"""Why PyTorch, not TensorFlow: the lesson's computation graph of f = ab + sin a at a = 2, b = 3, recorded as it
runs. Inputs a, b (circles) feed v1 = ab and v2 = sin a (boxes), which feed f. Ink arrows run forward; terracotta
arrows run the backward pass in reverse, and a receives two of them, which add. Above each node a grey bar is its
value (2, 3, 6, 0.909, 6.909); below it a terracotta bar is its adjoint df/dv (2.584, 2, 1, 1, 1).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

a, b = 2.0, 3.0
v1, v2 = a * b, math.sin(a)
f = v1 + v2
fb = 1.0
v1b = v2b = fb
ab_ = v1b * b + v2b * math.cos(a)
bb = v1b * a
nodes = {  # name: (x, y, value, adjoint, kind)
    "a": (95, 130, a, ab_, "in"), "b": (95, 290, b, bb, "in"),
    "v2": (300, 110, v2, v2b, "op"), "v1": (300, 290, v1, v1b, "op"),
    "f": (515, 200, f, fb, "op"),
}
edges = [("a", "v2"), ("a", "v1"), ("b", "v1"), ("v1", "f"), ("v2", "f")]
NR, BW, BH = 26, 72, 50
VS, AS = 70 / 7, 70 / 2.8          # px per unit for value and adjoint bars


def edge_pt(n, toward):
    x, y, *_, kind = nodes[n]
    tx, ty = nodes[toward][:2]
    ang = math.atan2(ty - y, tx - x)
    if kind == "in":
        return x + (NR + 3) * math.cos(ang), y + (NR + 3) * math.sin(ang)
    # leave the box through its border
    sx = (BW / 2 + 3) / max(abs(math.cos(ang)), 1e-6)
    sy = (BH / 2 + 3) / max(abs(math.sin(ang)), 1e-6)
    s = min(sx, sy)
    return x + s * math.cos(ang), y + s * math.sin(ang)


body = ""
for u, v in edges:
    x1, y1 = edge_pt(u, v)
    x2, y2 = edge_pt(v, u)
    body += arrow(x1, y1, x2, y2, INK, 2.4)
    # backward arrow: parallel, offset to the lower-right side of the edge, pointing back
    ang = math.atan2(y2 - y1, x2 - x1)
    ox, oy = -math.sin(ang) * 12, math.cos(ang) * 12
    body += arrow(x2 + ox - 10 * math.cos(ang), y2 + oy - 10 * math.sin(ang),
                  x1 + ox + 20 * math.cos(ang), y1 + oy + 20 * math.sin(ang), ACCENT, 2.2)
for n, (x, y, val, adj, kind) in nodes.items():
    if kind == "in":
        body += f'<circle cx="{x}" cy="{y}" r="{NR}" fill="{BG}" stroke="{INK}" stroke-width="2.6"/>'
        top, bot = y - NR, y + NR
    else:
        body += rect(x - BW / 2, y - BH / 2, BW, BH, fill=ACCENT_SOFT if n == "f" else BG, stroke=INK, sw=2.6, rx=8)
        top, bot = y - BH / 2, y + BH / 2
        if n == "v1":      # multiply
            body += line(x - 10, y - 10, x + 10, y + 10, INK, 3) + line(x - 10, y + 10, x + 10, y - 10, INK, 3)
        elif n == "f":     # add
            body += line(x - 12, y, x + 12, y, INK, 3) + line(x, y - 12, x, y + 12, INK, 3)
        else:              # sine, one true period
            body += poly([(x - 22 + 44 * t / 40, y - 12 * math.sin(2 * math.pi * t / 40)) for t in range(41)], INK, 3)
    bx = x - (55 if kind == "in" else 35)
    body += line(bx, top - 14, bx + val * VS, top - 14, GREY, 7)
    body += line(bx, bot + 14, bx + adj * AS, bot + 14, ACCENT, 7)
write(body)
