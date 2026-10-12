"""Normalises the team photos to a sane size for the circular avatars.

The avatars render at clamp(8.25rem, 18vw, 11rem) -- up to 176 CSS px -- with
`background: center / cover`, so anything past ~2x that is wasted bytes and
anything under it is blurry. This downscales the short edge to TARGET, never
upscales, and re-encodes. Cropping is left to CSS, which frames them correctly
already.

    python3 scripts/optimize-images.py
"""
import os
from pathlib import Path

from PIL import Image, ImageOps

TARGET = 640           # short edge; 176 CSS px needs 352 at 2x, this leaves headroom
TEAM = Path("public/assets/team")

before = after = 0
for f in sorted(TEAM.iterdir()):
    if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    size_before = f.stat().st_size
    before += size_before

    im = Image.open(f)
    im = ImageOps.exif_transpose(im)          # honour phone camera rotation
    short = min(im.size)
    note = ""
    if short > TARGET:
        scale = TARGET / short
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    elif short < 352:
        note = f"  (source is only {short}px on its short edge; cannot be sharpened)"

    if f.suffix.lower() == ".png":
        im.convert("RGBA" if "A" in im.getbands() else "RGB").save(f, "PNG", optimize=True)
    else:
        im.convert("RGB").save(f, "JPEG", quality=86, optimize=True, progressive=True)

    size_after = f.stat().st_size
    after += size_after
    print(f"  {f.name[:40]:42s} {size_before//1024:5d} KB -> {size_after//1024:4d} KB  {im.width}x{im.height}{note}")

print(f"\nteam photos: {before/1024:.0f} KB -> {after/1024:.0f} KB")
