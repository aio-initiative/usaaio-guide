"""Contact sheet: one figure drawn two ways, side by side for L0 to judge.

    python scripts/import-b/contact_sheet.py <left> <right> -o <out.png> [--title ...]

The left image is the reference and the right is what a runner produced. Pass
the left label with `--left-label`, because the reference is not always the
same thing: against A's own hand-made figure it is "hand-made, in the module
today", against a B original it is "site B original". Both are shown at the
same width, so the comparison is about the look (stroke weight, text size,
colour, background) and not about a figure happening to be drawn at a
different size.

SVGs are rasterised with `rsvg-convert` at 2x. Everything is pasted onto white,
because the question is whether the runner's output belongs on A's light page.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

RIGHT_LABEL = "runner output"


def rasterise(path: pathlib.Path, tmp: pathlib.Path) -> pathlib.Path:
    """An SVG becomes a PNG at 2x; a PNG is used as it is."""
    if path.suffix.lower() == ".svg":
        out = tmp / (path.stem + ".png")
        subprocess.run(
            ["rsvg-convert", "-d", "2", "-p", "2", "-b", "white",
             str(path), "-o", str(out)],
            check=True, capture_output=True,
        )
        return out
    return path


def load(path: pathlib.Path):
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def fit(im, width: int):
    """Scale to `width` CSS px, keeping the aspect ratio."""
    from PIL import Image

    if im.width == width:
        return im
    h = max(1, round(im.height * width / im.width))
    return im.resize((width, h), Image.LANCZOS)


def sheet(left: pathlib.Path, right: pathlib.Path, out: pathlib.Path,
          title: str = "", width: int = 720, pad: int = 28,
          label_h: int = 34, left_label: str = "") -> None:
    from PIL import Image, ImageDraw, ImageFont

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        a, b = fit(load(rasterise(left, tmp)), width), fit(load(rasterise(right, tmp)), width)
        w = pad * 3 + a.width + b.width
        h = pad * 2 + label_h * 2 + max(a.height, b.height) + (26 if title else 0)
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)

        def font(size, bold=False):
            p = pathlib.Path("/usr/share/texmf-dist/fonts/opentype/public/inter")
            name = f"Inter-{'SemiBold' if bold else 'Medium'}.otf"
            if (p / name).exists():
                return ImageFont.truetype(str(p / name), size)
            return ImageFont.load_default()

        y = pad
        if title:
            draw.text((pad, y), title, font=font(26, bold=True), fill="#111827")
            y += 34
        draw.text((pad, y), left_label, font=font(17), fill="#5c5c5c")
        draw.text((pad * 2 + a.width, y), RIGHT_LABEL, font=font(17), fill="#5c5c5c")
        y += label_h
        canvas.paste(a, (pad, y))
        canvas.paste(b, (pad * 2 + a.width, y))
        draw.line([(pad + a.width + pad // 2, y), (pad + a.width + pad // 2, y + max(a.height, b.height))],
                  fill="#d8d8dd", width=1)
        for im, x in ((a, pad), (b, pad * 2 + a.width)):
            draw.rectangle([x - 1, y - 1, x + im.width, y + im.height],
                           outline="#d8d8dd", width=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"{out}  {canvas.width}x{canvas.height}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="side by side contact sheet of two figures")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--left-label", default="site B original",
                    help="what the left image is; the default is a B original")
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args(argv)
    sheet(pathlib.Path(args.left), pathlib.Path(args.right),
          pathlib.Path(args.out), args.title, args.width,
          left_label=args.left_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
