"""Renders every registered module figure and rewrites the module Markdown.

Each entry in FIGURES is (module_file, anchor, slug, alt, draw_fn).

The image is produced by *executing the snippet that gets published*, not by
calling draw_fn directly, so the picture in the module is by construction the
output of the code shown beneath it. `--verify` re-runs every published snippet
in a fresh namespace and compares the bytes against the committed PNG.

Re-running is safe: an existing figure block for a slug is replaced in place.
"""
import inspect
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import figlib  # noqa: E402
from figures import FIGURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / "assets" / "module-media"
CONTENT = ROOT / "content"

MARK = "<!-- figure:{slug} -->"


# Prepended to every published snippet so it is standalone: without it a reader
# would paste the body and get matplotlib defaults instead of the guide's look.
PREAMBLE = """import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#e5e7eb", "axes.labelcolor": "#4b5563",
    "axes.titlecolor": "#111827", "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.8,
    "xtick.color": "#4b5563", "ytick.color": "#4b5563",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "white",
    "legend.fontsize": 10, "font.size": 11,
    "figure.dpi": 280, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
})
"""


def source_of(fn):
    """The body of a draw function, dedented, with the signature line dropped."""
    lines = inspect.getsource(fn).split("\n")
    body = lines[1:]
    while body and not body[0].strip():
        body.pop(0)
    indent = min((len(l) - len(l.lstrip()) for l in body if l.strip()), default=0)
    out = [l[indent:] if len(l) >= indent else l for l in body]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def snippet(fn):
    """The standalone program published under the figure, and the one that runs.

    `return f` is replaced by `plt.show()` so the block reads as a script rather
    than a function body.
    """
    body = source_of(fn)
    body = re.sub(r"^return f\s*$", "plt.show()", body, flags=re.M)
    return f"{PREAMBLE}\n{body}\n"


def render(code, path):
    """Runs a published snippet and captures whatever it shows."""
    ns = {"__name__": "__figure__"}
    saved = {}

    def fake_show(*_args, **_kwargs):
        saved["fig"] = plt.gcf()

    real_show = plt.show
    plt.show = fake_show
    try:
        exec(compile(code, f"<figure {path.stem}>", "exec"), ns)
        fig = saved.get("fig") or plt.gcf()
        fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    finally:
        plt.show = real_show
        plt.close("all")


def block_for(slug, alt, fn):
    code = snippet(fn)
    return (
        f"{MARK.format(slug=slug)}\n"
        f"![{alt}](assets/module-media/{slug}.png)\n\n"
        f"<details>\n"
        f"<summary>the code that draws this</summary>\n\n"
        f"```python\n{code}\n```\n\n"
        f"</details>\n"
    )


def find_module(name):
    hits = list(CONTENT.glob(f"*/{name}"))
    if not hits:
        raise SystemExit(f"no module file matching {name}")
    return hits[0]


def insert(path, anchor, slug, alt, fn):
    text = path.read_text()
    block = block_for(slug, alt, fn)

    # Replace an existing block for this slug rather than stacking duplicates.
    # The anchor is authoritative, so an existing block is removed and
    # re-inserted rather than patched where it happens to sit.
    pattern = re.compile(
        r"\n*" + re.escape(MARK.format(slug=slug)) + r".*?</details>\n", re.S
    )
    text = pattern.sub("\n", text)

    lines = text.split("\n")

    # ("## Section", n) places the figure after the nth paragraph of that
    # section, so it breaks the prose instead of sitting under the heading.
    # A plain string anchors on the line it prefixes.
    if isinstance(anchor, tuple):
        heading, nth = anchor
        start = None
        for i, line in enumerate(lines):
            if line.strip() == heading.strip():
                start = i + 1
                break
        if start is None:
            return text, False
        end = len(lines)
        for i in range(start, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        # Paragraph scan mirroring parseBlocks in build-content.mjs: fenced
        # code, HTML comments, <details>, display math and list/table/heading
        # lines are all breaks, never paragraph text.
        paras, i = [], start
        while i < end:
            line = lines[i]
            st = line.strip()
            if st.startswith("```"):
                i += 1
                while i < end and not lines[i].strip().startswith("```"):
                    i += 1
                i += 1
                continue
            if st.startswith("<!--"):
                while i < end and "-->" not in lines[i]:
                    i += 1
                i += 1
                continue
            if st.startswith("<details"):
                while i < end and "</details>" not in lines[i]:
                    i += 1
                i += 1
                continue
            if st == "$$":
                i += 1
                while i < end and lines[i].strip() != "$$":
                    i += 1
                i += 1
                continue
            if not st or re.match(r"^(#|\||!\[|<|- |\* |\d+\. |> |\$\$)", st):
                i += 1
                continue
            begin, size = i, 0
            while i < end and lines[i].strip() and not lines[i].strip().startswith("<"):
                size += len(lines[i])
                i += 1
            paras.append([begin, i, size])

        if nth == "wall":
            # Longest run of paragraphs separated only by blank lines, split at
            # its midpoint so the figure lands inside the wall.
            runs, run = [], []
            for para in paras:
                if run and any(lines[t].strip() for t in range(run[-1][1], para[0])):
                    runs.append(run)
                    run = []
                run.append(para)
            if run:
                runs.append(run)
            best = max(runs, key=lambda r: sum(x[2] for x in r)) if runs else []
            if not best:
                j = end
            else:
                half, acc, j = sum(x[2] for x in best) / 2, 0, best[-1][1]
                for b, e, size in best:
                    acc += size
                    if acc >= half:
                        j = e
                        break
        else:
            j = paras[min(nth, len(paras)) - 1][1] if paras else end

        lines[j:j] = ["", block.rstrip("\n")]
        return "\n".join(lines), True

    for i, line in enumerate(lines):
        if line.strip().startswith(anchor.strip()):
            j = i + 1
            while j < len(lines) and lines[j].strip():
                j += 1
            lines[j:j] = ["", block.rstrip("\n")]
            return "\n".join(lines), True
    return text, False


def verify():
    """Re-runs every published snippet and diffs it against the committed PNG."""
    import hashlib
    import tempfile

    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for module_file, _anchor, slug, _alt, fn in FIGURES:
            committed = MEDIA / f"{slug}.png"
            if not committed.exists():
                bad.append(f"{slug}: no committed image")
                continue

            # Take the code from the module Markdown, not from figures.py, so
            # this checks what a reader actually sees.
            text = find_module(module_file).read_text()
            m = re.search(
                re.escape(MARK.format(slug=slug))
                + r".*?```python\n(.*?)```",
                text,
                re.S,
            )
            if not m:
                bad.append(f"{slug}: no published code block")
                continue

            out = Path(tmp) / f"{slug}.png"
            try:
                render(m.group(1), out)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{slug}: published code failed to run -- {exc}")
                continue

            a = hashlib.sha256(committed.read_bytes()).hexdigest()
            b = hashlib.sha256(out.read_bytes()).hexdigest()
            if a != b:
                bad.append(f"{slug}: image differs from what its code prints")

    if bad:
        print(f"{len(bad)} of {len(FIGURES)} figures do NOT match their code:")
        for line in bad:
            print(f"  - {line}")
        raise SystemExit(1)
    print(f"verified {len(FIGURES)} figures: every image is exactly what its code prints")


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    if "--verify" in sys.argv:
        verify()
        return
    only = sys.argv[1] if len(sys.argv) > 1 else None
    drawn = inserted = missed = 0
    for module_file, anchor, slug, alt, fn in FIGURES:
        if only and only not in module_file:
            continue
        render(snippet(fn), MEDIA / f"{slug}.png")
        drawn += 1

        path = find_module(module_file)
        text, ok = insert(path, anchor, slug, alt, fn)
        if ok:
            path.write_text(text)
            inserted += 1
        else:
            missed += 1
            print(f"  !! anchor not found in {module_file}: {anchor!r}")
    print(f"drew {drawn} figures, inserted {inserted}, missed {missed}")


if __name__ == "__main__":
    main()
