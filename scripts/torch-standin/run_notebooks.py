"""Run the guide's PyTorch solution notebooks, on real PyTorch or on the stand-in, and report.

This is the parity check at notebook level: the parity tests in tests/ compare the stand-in
with real PyTorch name by name, and this runs the notebooks a reader actually runs, so
a stand-in gap that no test noticed still shows up as a check that stops passing.

For each notebook it prints the `Part N passed` lines the notebook printed, the first error with
the code cell it came from, and how long the notebook took. Run it twice and read the two runs
side by side: every part real PyTorch passes should also pass on the stand-in, and the ones that
do not are either a bug in src/ or a check that depends on real PyTorch's own random numbers.
The latter are written down in KNOWN_DIFFERENCES.md with what the check should test instead.

Run it from scripts/torch-standin with the venv's Python, which is the one that has real torch:

    .venv/bin/python run_notebooks.py --real
    .venv/bin/python run_notebooks.py --standin
    .venv/bin/python run_notebooks.py --standin --extra /tmp/other-notebooks
    .venv/bin/python run_notebooks.py --standin notebooks/modern-ai/cnn-train-pytorch-solution.ipynb
    .venv/bin/python run_notebooks.py --standin --json /tmp/standin.json

With no paths it runs every notebooks/**/*-solution.ipynb in the repository that imports torch or
torchvision, plus the solution notebooks under each --extra folder. Paths given on the command
line are run whatever they import; a relative path is read from the current folder, or else
from the repository. Each notebook runs in a child process, so its memory is returned when it
ends and a notebook that hangs is killed at --timeout instead of taking the run with it.

The children run in a scratch folder outside the repository (RUN_DIR), because the notebooks
write files where they stand: `datasets.MNIST("data", download=True)` downloads the full MNIST
there on real PyTorch, and the training-loop notebook saves checkpoints. Kept between runs, so
real MNIST is downloaded once.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
# A plotting library the notebooks need, kept beside the tool when the venv has none: the test
# venv is shared with the other stand-in worktrees, so it cannot be given matplotlib of its own.
# Appended, never prepended, so the venv's numpy and torch still win.
EXTRA_LIBS = HERE / ".notebooks-libs"
REPO = HERE.parents[1]
NOTEBOOKS = REPO / "notebooks"
# Where the children run, so what a notebook writes (data/, checkpoints) stays out of the repo.
RUN_DIR = Path(tempfile.gettempdir()) / "torch-standin-notebooks"

# Which torch a notebook has to import for the stand-in to matter, and so for the default run.
TORCH = ("torch", "torchvision")
# Cell magics that wrap ordinary Python. Any other %% magic makes the cell something else.
CODE_MAGICS = ("%%time", "%%timeit", "%%capture")
# What a check cell prints when the answer is right. The rewritten notebooks print
# `Part N passed`; the older practice ones still print `checks passed`, so both are read.
PASSED = re.compile(r"^(?:Part (\d+) passed|checks passed)$", re.MULTILINE)
# The same words inside a cell's own code, which is how the number of checks is counted.
MARKER = re.compile(r"Part \d+ passed|checks passed")
# The top-level module of every `import x` and `from x import y`, the way the site's notebook
# code reads them (src/lib/notebooks/collect.ts). A commented-out import is not an import.
IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import\b|import[ \t]+([^#\n]+))", re.MULTILINE
)
# How many errors a notebook may raise before the run is stopped: a break in the data flow makes
# every later cell fail, and a dozen copies of the same traceback says nothing new.
MAX_ERRORS = 3


def code_cells(path):
    """The Python source of a notebook's code cells, with the magics turned into no-ops.

    A cell that is not Python (a `%%bash` cell, say) is skipped, and a line that is a shell
    command or a magic becomes `pass` so the rest of the cell still runs. surface.py reads the
    notebooks the same way, so the cells run here are the cells the surface was measured from.
    """
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        lines = source.split("\n")
        if lines[0].startswith("%%"):
            if lines[0].split()[0] not in CODE_MAGICS:
                continue
            lines = lines[1:]
        cleaned = []
        for line in lines:
            body = line.lstrip()
            if body.startswith(("%", "!")):
                line = f"{line[: len(line) - len(body)]}pass"
            cleaned.append(line)
        cells.append((index, "\n".join(cleaned)))
    return cells


def imports_torch(path):
    """True when the notebook's own code imports torch or torchvision."""
    for _, source in code_cells(path):
        for from_module, plain in IMPORT.findall(source):
            names = [from_module] if from_module else plain.split(",")
            for name in names:
                words = name.strip().split()
                if words and words[0].split(".")[0] in TORCH:
                    return True
    return False


def notebooks_under(folder):
    """Every solution notebook under a folder, sorted, without checkpoints."""
    return sorted(
        path
        for path in Path(folder).rglob("*-solution.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def chosen_paths(paths, extras):
    """The notebooks to run, in order, and the ones passed over because they are not torch ones.

    With no paths on the command line the torch notebooks under notebooks/ and under each extra
    folder are run. A path given on the command line is always run, whatever it imports, and a
    folder stands for the solution notebooks under it, so a notebook can be checked even when it
    is not one of the defaults.
    """
    if paths:
        found = []
        for path in map(located, paths):
            if path.is_dir():
                found.extend(notebooks_under(path))
            else:
                found.append(path)
        return found, []

    everything, skipped = [], []
    for folder in [NOTEBOOKS, *map(located, extras)]:
        for path in notebooks_under(folder):
            (everything if imports_torch(path) else skipped).append(path)
    return everything, skipped


def located(path):
    """A command-line path made absolute: from the current folder when it is there, else from
    the repository, so `notebooks/...` works from scripts/torch-standin too. Absolute because the
    children run in RUN_DIR."""
    path = Path(path)
    if not path.is_absolute() and not path.exists() and (REPO / path).exists():
        path = REPO / path
    return path.resolve()


def run_one(path, mode, timeout):
    """Run one notebook in a child process and return its report, as a dictionary."""
    request = {"path": str(path), "mode": mode}
    environment = {**os.environ, "MPLBACKEND": "Agg"}
    # Two threads is enough and keeps several notebooks from adding up on a busy machine.
    environment.setdefault("OMP_NUM_THREADS", "2")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        finished = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(request)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            cwd=RUN_DIR,
        )
    except subprocess.TimeoutExpired:
        return {
            "path": str(path),
            "passes": [],
            "errors": [],
            "seconds": timeout,
            "timed_out": True,
            "output": "",
        }
    if finished.returncode != 0 or not finished.stdout.strip():
        # The child died before it could report, which is worth more than a bare exit code.
        return {
            "path": str(path),
            "passes": [],
            "errors": [{"cell": None, "text": (finished.stderr or finished.stdout).strip()[-2000:]}],
            "seconds": 0.0,
            "timed_out": False,
            "output": "",
        }
    return json.loads(finished.stdout)


def in_child(path, mode):
    """Run the notebook's cells in one namespace and return what happened.

    This is the half that runs inside the child, where `import torch` can be pointed at src/ for
    the stand-in. One namespace for every cell, as a notebook has, so a function defined in a
    solution cell is there for the check cell under it.
    """
    # Before anything imports torch, so the stand-in is the torch this process finds.
    if mode == "standin":
        sys.path.insert(0, str(SRC))
    if EXTRA_LIBS.is_dir():
        sys.path.append(str(EXTRA_LIBS))

    cells = code_cells(path)
    output = io.StringIO()
    errors = []
    namespace = {"__name__": "__main__"}
    started_all = time.perf_counter()
    for index, source in cells:
        try:
            # What the cell prints is kept for the report instead of going to the terminal.
            with redirect_stdout(output), redirect_stderr(output):
                exec(compile(source, f"{Path(path).name} cell {index}", "exec"), namespace)
        except BaseException:  # noqa: BLE001 - a notebook can raise anything, and it is reported
            errors.append({"cell": index, "text": traceback.format_exc()})
            if len(errors) >= MAX_ERRORS:
                break
    seconds = time.perf_counter() - started_all

    which, version = "", ""
    try:
        import torch

        which = "stand-in" if str(Path(torch.__file__).resolve()).startswith(str(SRC)) else "real torch"
        version = getattr(torch, "__version__", "")
    except BaseException as err:  # noqa: BLE001
        errors.insert(0, {"cell": None, "text": f"import torch failed: {err}"})
    printed = output.getvalue()
    return {
        "path": str(path),
        "torch": which,
        "version": version,
        "passes": pass_lines(printed),
        "expected": check_cells(cells),
        "errors": errors,
        "seconds": round(seconds, 1),
        "timed_out": False,
        "output": printed,
    }


def last_line(text):
    """The line of a traceback that says what went wrong, such as `AssertionError: ...`."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else text.strip()


def pass_lines(text):
    """The lines in what a run printed that say a check passed, in the order they were printed."""
    found = []
    for match in PASSED.finditer(text):
        number = match.group(1)
        found.append(f"Part {number} passed" if number else "checks passed")
    return found


def check_cells(cells):
    """How many check cells the notebook has: each one prints its marker on one line."""
    return sum(
        1
        for _, source in cells
        for line in source.split("\n")
        if line.lstrip().startswith("print(") and MARKER.search(line)
    )


def describe(path, repo=REPO):
    """A path as the reader would type it: relative to the repository when it is inside it."""
    path = Path(path)
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        return str(path)


def report(result, show_output=False):
    """Print one notebook's outcome: the parts it passed, its first error and its seconds."""
    print(describe(result["path"]))
    for line in result["passes"]:
        print(f"  {line}")
    if result["timed_out"]:
        print(f"  TIMED OUT after {result['seconds']:.1f} s")
    elif result["errors"]:
        first = result["errors"][0]
        where = "" if first["cell"] is None else f" (cell {first['cell']})"
        print(f"  first error{where}: {last_line(first['text'])}")
        if len(result["errors"]) > 1:
            print(f"  ... and {len(result['errors']) - 1} more errors")
        if show_output:
            for error in result["errors"]:
                print(f"  --- cell {error['cell']} ---")
                print(error["text"].rstrip())
    else:
        print("  no errors")
    if show_output and result["output"]:
        for line in result["output"].rstrip().split("\n"):
            print(f"  | {line}")
    expected = result.get("expected", len(result["passes"]))
    print(f"  {len(result['passes'])} of {expected} checks passed in {result['seconds']:.1f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--standin", action="store_true", help="run against the stand-in (the default)")
    mode.add_argument("--real", action="store_true", help="run against the real PyTorch in the venv")
    parser.add_argument(
        "--extra", action="append", default=[], type=Path, metavar="DIR",
        help="another notebooks folder to run (repeatable)",
    )
    parser.add_argument("--timeout", type=float, default=600.0, metavar="SECONDS",
                        help="give up on one notebook after this long (default 600)")
    parser.add_argument("--show-output", action="store_true", help="also print what the cells wrote")
    parser.add_argument("--json", type=Path, metavar="FILE", help="write the results as JSON here")
    parser.add_argument(
        "paths", nargs="*", type=Path, help="notebooks to run instead of the defaults"
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        request = json.loads(args.child)
        print(json.dumps(in_child(request["path"], request["mode"])))
        return 0

    which = "standin" if not args.real else "real"
    paths, skipped = chosen_paths(args.paths, args.extra)
    if not paths:
        print("No notebooks to run.", file=sys.stderr)
        return 1
    print(f"Running {len(paths)} notebook{'' if len(paths) == 1 else 's'} on the {which} torch.")
    if skipped:
        print(
            f"  left out {len(skipped)} that import no torch: "
            + ", ".join(describe(path) for path in skipped)
        )

    results = [run_one(path, which, args.timeout) for path in paths]
    for result in results:
        report(result, args.show_output)

    print()
    print(f"{'notebook':66} {'checks':>8} {'seconds':>8}")
    total = 0.0
    for result in results:
        total += result["seconds"]
        checks = f"{len(result['passes'])}/{result.get('expected', len(result['passes']))}"
        print(f"{describe(result['path']):66} {checks:>8} {result['seconds']:>8.1f}")
    print(f"{'total':66} {'':>8} {total:>8.1f}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")

    broken = [result for result in results if result["errors"] or result["timed_out"]]
    print(f"{len(results) - len(broken)} of {len(results)} notebooks ran without an error.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())