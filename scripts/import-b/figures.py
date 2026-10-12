"""Redraw a site B figure script's outputs in site A's look.

    python scripts/import-b/figures.py <B figure script> [--out <dir>] [--prefix <s>]
    python scripts/import-b/figures.py tools/diagrams/gen_mlp_figures.py

Site B draws every figure through one shared style module,
`tools/diagrams/usaaio_theme.py`, and every generator imports it and nothing
else for colour, type, stroke or size. This runner imports a B generator
unchanged with `usaaio_theme` resolved to A's `scripts/figures/b_theme.py`, so
the generator draws the same figure in A's look.

It never edits B and never writes into B: B's folder is read-only input, and
the npz data files some generators load sit next to the script and are read
from there.

What it does not do: a generator that names a colour B's theme does not know
about, or that saves a figure itself, or that draws something the mapping in
`b_theme.py` cannot carry (a `series[0]` that means "the input layer" in one
figure and "the curve being explained" in another) comes out wrong, and this
runner says so rather than guessing. See `docs/merge-tooling.md` for the limits
and the per-script patch files.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
A_ROOT = HERE.parents[1]
B_ROOT = pathlib.Path("/home/jackwu/Desktop/Repos/USAAIO-Guide")
B_DIAGRAMS = B_ROOT / "tools" / "diagrams"

sys.path.insert(0, str(A_ROOT / "scripts" / "figures"))

import b_theme  # noqa: E402

# B's generators do `from usaaio_theme import ...`. Seeding sys.modules means
# the import resolves to A's theme and B's own theme module is never executed.
sys.modules["usaaio_theme"] = b_theme

# gen_step_figures.py does `from gen_transformer_figures import DASH, cells`, so
# B's folder has to be importable. It goes last: A's scripts win on collision.
for p in (str(B_DIAGRAMS),):
    if p not in sys.path:
        sys.path.append(p)


def import_script(path: pathlib.Path):
    """Import a B generator by path, without running its __main__ block."""
    if not path.is_absolute():
        path = B_DIAGRAMS / path
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    import importlib.util

    mod_name = path.stem
    # A second import of the same name would return the cached module; the
    # runner is one process per script, so this is only a safety check.
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_look(path: pathlib.Path) -> list[str]:
    """What can be measured without an eye: width, stray dark ink, text scale."""
    import re

    problems = []
    if path.suffix == ".svg":
        text = path.read_text(errors="replace")
        m = re.search(r"<svg\b[^>]*>", text)
        if not m:
            problems.append("no <svg> root element")
            return problems
        head = m.group(0)
        dim = re.search(r'width="(\d+)" height="(\d+)"', head)
        if not dim:
            problems.append("the SVG has no width/height attributes in CSS px")
        else:
            w, h = (int(v) for v in dim.groups())
            if w == 0 or h == 0:
                problems.append("zero-size SVG")
            elif w > 760:
                problems.append(f"width {w} px is wider than A's column (at most about 760)")
        # B's dark pass used #121212/#ECECEC and a teal signal; none should
        # survive the mapping.
        for stray in ("#121212", "#ECECEC", "#0F7363", "#5EEAD4", "#3A3A3A"):
            if stray.lower() in text.lower():
                problems.append(f"B's {stray} is still in the file")
    return problems


def load_patch(script: pathlib.Path, mod):
    """Apply a per-script patch from `scripts/import-b/figure-patches/`, if any.

    A patch is a small module named after the generator that defines
    `apply(module)`. It runs after the generator is imported and before any
    figure is drawn, so it can override a colour, a size or one call site. See
    `docs/merge-tooling.md`.
    """
    patch_path = HERE / "figure-patches" / script.name
    if not patch_path.exists():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"patch_{script.stem}", patch_path)
    patch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = patch
    spec.loader.exec_module(patch)
    if not hasattr(patch, "apply"):
        raise SystemExit(f"patch {patch_path} has no `apply(module)` function")
    patch.apply(mod)
    return patch_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("script", help="B figure script, e.g. tools/diagrams/gen_mlp_figures.py")
    ap.add_argument("--out", default=None, help="output dir (default: $AIO_FIG_OUT or assets/module-media)")
    ap.add_argument("--prefix", default=None, help="prefix on every output file name")
    ap.add_argument("--only", default=None, help="comma separated figure names to draw")
    ap.add_argument("--json", action="store_true", help="print a JSON report instead of prose")
    args = ap.parse_args(argv)

    # b_theme reads the env at save time, so set it here rather than pass it in.
    import os

    if args.out:
        os.environ["AIO_FIG_OUT"] = str(args.out)
    if args.prefix is not None:
        os.environ["AIO_FIG_PREFIX"] = args.prefix

    script = pathlib.Path(args.script)
    t0 = time.time()
    try:
        mod = import_script(script)
    except SystemExit:
        raise
    except Exception as e:
        report = {"script": script.name, "ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(report, indent=2) if args.json else f"FAILED to import {script.name}: {e}")
        return 2

    patch_path = None
    try:
        patch_path = load_patch(script, mod)
    except Exception as e:
        report = {"script": script.name, "ok": False,
                  "error": f"patch failed: {type(e).__name__}: {e}"}
        print(json.dumps(report, indent=2) if args.json
              else f"FAILED to patch {script.name}: {e}")
        return 2
    if patch_path and not args.json:
        print(f"patched with {patch_path.name}")

    figures = {k: v for k, v in vars(mod).items()
               if callable(v) and hasattr(v, "figure_name")}
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        figures = {k: v for k, v in figures.items() if v.figure_name in wanted}
        missing = wanted - {v.figure_name for v in figures.values()}
        if missing:
            print(f"warning: no figure named {sorted(missing)} in {script.name}")

    report = {"script": script.name, "figures": [], "seconds": round(time.time() - t0, 1)}
    failures = 0

    # With --json the caller parses stdout, so the per-figure progress lines
    # from the theme module are kept out of it.
    sink = io.StringIO() if args.json else sys.stdout
    for name, fn in sorted(figures.items(), key=lambda kv: kv[1].figure_name):
        t1 = time.time()
        try:
            with contextlib.redirect_stdout(sink):
                path = fn()
        except Exception as e:
            report["figures"].append({"name": fn.figure_name, "ok": False,
                                      "error": f"{type(e).__name__}: {e}",
                                      "seconds": round(time.time() - t1, 1)})
            failures += 1
            continue
        problems = check_look(path)
        info = b_theme.last.get(path.name, {})
        entry = {
            "name": fn.figure_name,
            "ok": not problems,
            "file": str(path),
            "size_kb": round(path.stat().st_size / 1024),
            "problems": problems,
            "seconds": round(time.time() - t1, 1),
        }
        entry.update(info)
        report["figures"].append(entry)
        failures += len(problems)

    report["ok"] = failures == 0
    report["n_figures"] = len(report["figures"])
    report["failures"] = failures
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for f in report["figures"]:
            mark = "ok " if f["ok"] else "PROBLEM"
            line = f"{mark} {f['name']}"
            if not f["ok"]:
                if "error" in f:
                    line += f"  {f['error']}"
                else:
                    line += "  " + "; ".join(f["problems"])
            print(line)
        print(f"{report['n_figures']} figure(s) from {script.name} in {report['seconds']}s"
              f"{' all ok' if report['ok'] else f', {failures} problem(s)'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
