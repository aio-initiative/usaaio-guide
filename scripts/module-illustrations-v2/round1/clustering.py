"""One Lloyd step: points coloured by nearest center (with the Voronoi boundary), and each old center moving to its cluster's mean."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(3)
blobs = [((145, 115), 30), ((455, 120), 30), ((310, 300), 30)]
pts = [(cx + random.gauss(0, s), cy + random.gauss(0, s)) for (cx, cy), s in blobs for _ in range(13)]
pts = [(min(max(x, 55), 545), min(max(y, 55), 345)) for x, y in pts]
old = [(245, 190), (430, 240), (195, 325)]          # poor starting centers


def nearest(p, cs):
    return min(range(len(cs)), key=lambda j: (p[0] - cs[j][0]) ** 2 + (p[1] - cs[j][1]) ** 2)


lab = [nearest(p, old) for p in pts]
new = []
for j in range(3):
    m = [p for p, l in zip(pts, lab) if l == j]
    new.append((sum(x for x, _ in m) / len(m), sum(y for _, y in m) / len(m)))

# Voronoi cell of each old center: the frame clipped by every bisector half-plane (Sutherland-Hodgman).
def clip(poly_, a, b):
    """Keep the part of poly_ closer to center a than to center b."""
    (ax, ay), (bx, by) = a, b
    f = lambda p: (p[0] - (ax + bx) / 2) * (bx - ax) + (p[1] - (ay + by) / 2) * (by - ay)
    out = []
    for k in range(len(poly_)):
        p, q = poly_[k], poly_[(k + 1) % len(poly_)]
        fp, fq = f(p), f(q)
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


FRAME_PTS = [(24, 24), (576, 24), (576, 376), (24, 376)]
FILL = [ACCENT_SOFT, SECOND_SOFT, "#eceef1"]
COL = [ACCENT, SECOND, INK]
body = ""
for j in range(3):
    cell = FRAME_PTS
    for k in range(3):
        if k != j:
            cell = clip(cell, old[j], old[k])
    body += poly(cell, "none", 0, fill=FILL[j], close=True)
for a in range(3):
    for b in range(a + 1, 3):
        cell = FRAME_PTS
        for k in range(3):
            if k != a:
                cell = clip(cell, old[a], old[k])
        # Boundary between a and b: the cell edges lying on their bisector.
        (ax, ay), (bx, by) = old[a], old[b]
        on = [p for p in cell if abs((p[0] - (ax + bx) / 2) * (bx - ax) + (p[1] - (ay + by) / 2) * (by - ay)) < 1e-6 * math.dist(old[a], old[b]) ** 2]
        if len(on) == 2:
            body += line(*on[0], *on[1], GREY, 1.8, dash="7 6")
for p, l in zip(pts, lab):
    body += dot(*p, 6, COL[l])
for (ox, oy), (nx, ny), c in zip(old, new, COL):
    body += ring(ox, oy, 11, c, 3)
    body += arrow(ox, oy, nx, ny, c, 2.6)
    body += dot(nx, ny, 9, c, BG, 2.5)
    body += line(nx - 5, ny, nx + 5, ny, BG, 2) + line(nx, ny - 5, nx, ny + 5, BG, 2)
write(body)
