"""Discriminative learning rates (the lesson's Part C): a pretrained backbone of four blocks and a new head, drawn
as feature maps that shrink in size and grow in depth, and above each a bar of its learning rate, base 1e-3 at the
head decaying by 2.5 per block going down, so block 1 moves at 1/39 of the head's rate. The dashed curve is the
geometric decay through the bar tops; the head is terracotta, the pretrained blocks grey."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

rates = [1e-3 / 2.5 ** k for k in range(4, -1, -1)]       # block 1 .. block 4, head
X0, X1 = 50, 550
slot = (X1 - X0) / 5
BASE = 238                                                  # bar baseline
TOP = 46
body = line(X0, BASE, X1, BASE, GUIDE, THIN)
tops = []
for k, r in enumerate(rates):
    cx = X0 + slot * (k + 0.5)
    h = (BASE - TOP) * r / rates[-1]
    head = k == 4
    body += rect(cx - 26, BASE - h, 52, h, ACCENT if head else GREY, "none", 0, 3)
    tops.append((cx, BASE - h))
# geometric decay, evaluated continuously between the bars
pts = []
for i in range(81):
    t = 4 * i / 80
    pts.append((X0 + slot * (t + 0.5), BASE - (BASE - TOP) * 2.5 ** (t - 4)))
body += poly(pts, INK, 2, dash="6 6")
for x, y in tops: body += dot(x, y, 5, INK)
# the network underneath: blocks as feature-map slabs, spatial size shrinking, depth growing, then the head
CY = 318
sizes = [64, 52, 40, 30]
depth = [6, 10, 15, 22]
for k in range(4):
    cx = X0 + slot * (k + 0.5)
    s, d = sizes[k], depth[k]
    # a slab seen at an angle: front square plus a parallelogram top and side
    fx, fy = cx - s / 2 - d / 2, CY - s / 2 + d / 2
    body += f'<path d="M{fx:.1f},{fy:.1f} L{fx + d:.1f},{fy - d:.1f} L{fx + s + d:.1f},{fy - d:.1f} L{fx + s:.1f},{fy:.1f} Z" fill="#e7e9ec" stroke="{GREY}" stroke-width="2" stroke-linejoin="round"/>'
    body += f'<path d="M{fx + s:.1f},{fy:.1f} L{fx + s + d:.1f},{fy - d:.1f} L{fx + s + d:.1f},{fy + s - d:.1f} L{fx + s:.1f},{fy + s:.1f} Z" fill="#d9dce0" stroke="{GREY}" stroke-width="2" stroke-linejoin="round"/>'
    body += rect(fx, fy, s, s, "#f1f2f4", GREY, 2, 1)
    if k < 3:
        body += arrow(cx + s / 2 + d / 2 + 6, CY, X0 + slot * (k + 1.5) - sizes[k + 1] / 2 - depth[k + 1] / 2 - 6, CY, GREY, 2)
# head: a column of output nodes
hx = X0 + slot * 4.5
body += arrow(X0 + slot * 3.5 + sizes[3] / 2 + depth[3] / 2 + 6, CY, hx - 18, CY, GREY, 2)
for j in range(3):
    body += dot(hx, CY - 24 + 24 * j, 8, ACCENT_SOFT, ACCENT, 2.6)
body += rect(hx - 18, CY - 42, 36, 84, "none", ACCENT, 2, 8)
write(body)
