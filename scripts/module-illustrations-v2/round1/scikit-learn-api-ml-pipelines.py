"""Scikit-Learn Pipelines: why KNN needs scaling. Two classes differ only in a feature with a small spread; the other
feature has a spread about 14 times larger. Both panels plot the same points after dividing each feature by its
spread. Left: the region holding the query's k = 5 nearest neighbours under raw distance, which is a tall thin
ellipse, so the neighbours come from both classes. Right: the k = 5 nearest under scaled distance, a circle whose
neighbours all share the query's class.
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(2757)
pts = []
for cls, mu in ((0, -0.55), (1, 0.55)):
    for _ in range(16):
        pts.append((random.gauss(0, 4.0), random.gauss(mu, 0.26), cls))
q = (-2.5, 0.2)
K = 5


def std(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))


s1, s2 = std([p[0] for p in pts]), std([p[1] for p in pts])
COL = {0: SECOND, 1: ACCENT}
SC = 46


def panel(cx, cy, raw):
    """raw=True ranks neighbours by distance in the original units, raw=False after scaling."""
    if raw:
        d = sorted(pts, key=lambda p: math.dist(p[:2], q))
        r = math.dist(d[K - 1][:2], q) * 1.12
        rx, ry = r / s1 * SC, r / s2 * SC
    else:
        z = lambda p: (p[0] / s1, p[1] / s2)
        d = sorted(pts, key=lambda p: math.dist(z(p), z(q)))
        r = math.dist(z(d[K - 1]), z(q)) * 1.12
        rx = ry = r * SC
    X = lambda x: cx + x / s1 * SC
    Y = lambda y: cy - y / s2 * SC
    out = line(cx - 118, cy, cx + 118, cy, GUIDE, THIN) + line(cx, cy - 125, cx, cy + 125, GUIDE, THIN)
    out += (f'<ellipse cx="{X(q[0]):.1f}" cy="{Y(q[1]):.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{ACCENT_SOFT if raw else SECOND_SOFT}" '
            f'fill-opacity="0.7" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>')
    near = d[:K]
    for p in pts:
        if p in near:
            out += dot(X(p[0]), Y(p[1]), 7, COL[p[2]], INK, 2.4)
        else:
            out += dot(X(p[0]), Y(p[1]), 5.5, COL[p[2]])
    qx, qy = X(q[0]), Y(q[1])
    out += (f'<path d="M{qx:.1f},{qy - 12:.1f} L{qx + 11:.1f},{qy:.1f} L{qx:.1f},{qy + 12:.1f} L{qx - 11:.1f},{qy:.1f} Z" '
            f'fill="{BG}" stroke="{INK}" stroke-width="2.6"/>')
    return out


body = panel(160, 200, True) + panel(440, 200, False)
body += line(300, 60, 300, 340, GUIDE, 1.2, dash="2 6")
write(body)
