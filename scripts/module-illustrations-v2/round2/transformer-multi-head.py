"""Two heads side by side: for the same three tokens, head 1's attention from the last token lands on the middle token
and head 2's stays on itself; each head returns a short vector and the two are placed end to end."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

heads = [([0.115, 0.709, 0.175], (0.872, 1.135), ACCENT, ACCENT_SOFT),
         ([0.124, 0.105, 0.771], (0.893, 1.771), SECOND, SECOND_SOFT)]
body = ""
outs = []
for h, (wts, out, col, soft) in enumerate(heads):
    yc = 130 + h * 170
    body += rect(40, yc - 82, 300, 140, soft, "none", 0, 10)
    xs = [90, 180, 270]
    ty = yc + 22
    # Arcs from the last token to every token, thickness from the weight; the self weight as a loop.
    for j, a in enumerate(wts[:2]):
        x0, x1 = xs[2], xs[j]
        hgt = 28 + (x0 - x1) * 0.32
        pts = [(x0 + (x1 - x0) * t, ty - 16 - hgt * math.sin(math.pi * t)) for t in [i / 30 for i in range(31)]]
        body += poly(pts, col if a == max(wts) else GREY, 1.2 + 11 * a)
    a = wts[2]
    loop = [(xs[2] + 22 * math.sin(2 * math.pi * t), ty - 16 - 26 * (1 - math.cos(2 * math.pi * t)) / 2 * 1.6)
            for t in [i / 40 for i in range(41)]]
    body += poly(loop, col if a == max(wts) else GREY, 1.2 + 11 * a)
    for j, x in enumerate(xs):
        body += rect(x - 18, ty - 14, 36, 28, INK if j == 2 else BG, INK, 2.2, 5)
    # Head output: two bars.
    body += arrow(350, yc, 388, yc, GREY, 2.2)
    bx = 400
    for k, v in enumerate(out):
        body += rect(bx + k * 26, yc + 30 - v * 40, 20, v * 40, col, "none", 0, 2)
    body += line(bx - 6, yc + 30, bx + 52, yc + 30, GUIDE, 1.4)
    outs.append((bx + 52, yc))
# Concatenation: the two head outputs joined, then four bars end to end.
cx, cb, jx = 506, 250, 474
for (sx, sy) in outs:
    body += line(sx + 4, sy, jx, sy, GREY, 2)
body += line(jx, outs[0][1], jx, outs[1][1], GREY, 2)
body += arrow(jx, 215, cx - 6, 215, GREY, 2.2)
vals = [*heads[0][1], *heads[1][1]]
for k, v in enumerate(vals):
    col = heads[k // 2][2]
    body += rect(cx + k * 14, cb - v * 40, 11, v * 40, col, "none", 0, 2)
body += line(cx - 4, cb, cx + 58, cb, GUIDE, 1.4)
write(body)
