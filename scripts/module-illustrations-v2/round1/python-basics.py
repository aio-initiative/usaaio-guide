"""The leaderboard example: seven log lines, two broken ones skipped, each model's best kept, sorted best first."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

lines = [("resnet", 0.91), ("vit", 0.94), ("resnet", 0.93), ("mlp", None), ("vit", 0.92), ("cnn", 0.88), (None, None)]
best = {}
for mdl, s in lines:
    if mdl is not None and s is not None:
        best[mdl] = max(best.get(mdl, 0), s)
top = sorted(best.items(), key=lambda kv: -kv[1])[:3]

def marker(kind, x, y, r, fill):
    if kind == "resnet":
        return dot(x, y, r, fill)
    if kind == "vit":
        return rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8, fill, "none", 0, 1.5)
    if kind == "cnn":
        return poly([(x, y - r * 1.1), (x + r * 1.1, y + r * 0.8), (x - r * 1.1, y + r * 0.8)], fill, 0.01, fill, close=True)
    return poly([(x, y - r * 1.2), (x + r, y), (x, y + r * 1.2), (x - r, y)], fill, 0.01, fill, close=True)

bar = lambda s, full: full * (s - 0.84) / (0.95 - 0.84)
x0, w, top_y, pitch, h = 40, 230, 38, 47, 34
body = ""
for i, (mdl, s) in enumerate(lines):
    y = top_y + i * pitch
    cy = y + h / 2
    broken = s is None
    kept = not broken and best[mdl] == s
    body += rect(x0, y, w, h, BG, GUIDE if broken else GREY, 1.6, 6)
    col = INK if kept else GREY
    if mdl:
        body += marker(mdl, x0 + 22, cy, 8, GUIDE if broken else col)
    if broken:
        body += line(x0 + 44, cy, x0 + w - 50, cy, GUIDE, 3, dash="6 6")
        xc = x0 + w - 24
        body += line(xc - 9, cy - 9, xc + 9, cy + 9, ACCENT, 3) + line(xc - 9, cy + 9, xc + 9, cy - 9, ACCENT, 3)
    else:
        body += f'<rect x="{x0 + 44}" y="{cy - 7}" width="{bar(s, 165):.1f}" height="14" rx="3" fill="{col}"/>'

body += arrow(298, 200, 352, 200, INK, 3)
rx, rw, rh, rp = 380, 180, 54, 72
ry0 = 200 - (len(top) * rp - (rp - rh)) / 2
for i, (mdl, s) in enumerate(top):
    y = ry0 + i * rp
    cy = y + rh / 2
    body += rect(rx, y, rw, rh, ACCENT_SOFT if i == 0 else BG, ACCENT, 2.2, 8)
    body += marker(mdl, rx + 26, cy, 11, ACCENT if i == 0 else INK)
    body += f'<rect x="{rx + 52}" y="{cy - 9}" width="{bar(s, 112):.1f}" height="18" rx="3" fill="{ACCENT if i == 0 else INK}"/>'
write(body)
