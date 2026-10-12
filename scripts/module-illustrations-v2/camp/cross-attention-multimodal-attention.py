"""Cross-attention: a photo of a dog with a ball, cut into a 4 x 4 grid of patches, and one text token attending to them.

The query scores each patch by how much ball (and a little dog) it holds; softmax weights set the width of each
link, so the two patches the ball straddles win.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N = 4
GX, GY, SZ = 40, 70, 260
P = SZ / N
u = lambda fx: GX + fx * SZ
v = lambda fy: GY + fy * SZ

# The scene in unit coordinates: a dog standing on the ground, a ball beside it.
BALL = (0.775, 0.70, 0.105)
DOG_BODY = (0.36, 0.55, 0.22, 0.11)     # ellipse cx, cy, rx, ry
DOG_HEAD = (0.58, 0.40, 0.085)
LEGS = [(0.20, 0.58), (0.28, 0.58), (0.44, 0.58), (0.52, 0.58)]   # x, top
GROUND = 0.80


def what(fx, fy):
    bx, by, br = BALL
    if (fx - bx) ** 2 + (fy - by) ** 2 <= br ** 2:
        return "ball"
    cx, cy, rx, ry = DOG_BODY
    hx, hy, hr = DOG_HEAD
    if ((fx - cx) / rx) ** 2 + ((fy - cy) / ry) ** 2 <= 1 or (fx - hx) ** 2 + (fy - hy) ** 2 <= hr ** 2:
        return "dog"
    if any(lx - 0.022 <= fx <= lx + 0.022 and ly <= fy <= GROUND for lx, ly in LEGS):
        return "dog"
    return "other"


# Patch scores from a fine sampling of each patch, then a softmax.
score, centers = [], []
K = 24
for pi in range(N):
    for pj in range(N):
        cells = [what((pj + (b + 0.5) / K) / N, (pi + (a + 0.5) / K) / N) for a in range(K) for b in range(K)]
        score.append(9.0 * cells.count("ball") / K ** 2 + 2.5 * cells.count("dog") / K ** 2)
        centers.append((GX + (pj + 0.5) * P, GY + (pi + 0.5) * P))
m = max(score)
e = [math.exp(s - m) for s in score]
w = [x / sum(e) for x in e]

defs = f'<clipPath id="img"><rect x="{GX}" y="{GY}" width="{SZ}" height="{SZ}" rx="2"/></clipPath>'
scene = rect(GX, GY, SZ, SZ, SECOND_SOFT, "none", 0, 0)
scene += rect(GX, v(GROUND), SZ, SZ * (1 - GROUND), GUIDE, "none", 0, 0)
cx, cy, rx, ry = DOG_BODY
hx, hy, hr = DOG_HEAD
for lx, ly in LEGS:
    scene += line(u(lx), v(ly), u(lx), v(GROUND), GREY, 0.044 * SZ)
scene += line(u(cx - rx * 0.9), v(cy - 0.02), u(cx - rx - 0.07), v(cy - 0.12), GREY, 9)        # tail
scene += f'<ellipse cx="{u(cx):.1f}" cy="{v(cy):.1f}" rx="{rx * SZ:.1f}" ry="{ry * SZ:.1f}" fill="{GREY}"/>'
scene += dot(u(hx), v(hy), hr * SZ, GREY)
scene += f'<ellipse cx="{u(hx + 0.08):.1f}" cy="{v(hy + 0.025):.1f}" rx="{0.05 * SZ:.1f}" ry="{0.035 * SZ:.1f}" fill="{GREY}"/>'
scene += poly([(u(hx - 0.06), v(hy - 0.03)), (u(hx - 0.02), v(hy - 0.13)), (u(hx + 0.02), v(hy - 0.05))], GREY, 1, GREY, close=True)
scene += dot(u(hx + 0.02), v(hy - 0.01), 3.2, INK)
bx, by, br = BALL
scene += dot(u(bx), v(by), br * SZ, ACCENT)
scene += f'<path d="M{u(bx - br):.1f},{v(by):.1f} Q{u(bx):.1f},{v(by + 0.07):.1f} {u(bx + br):.1f},{v(by):.1f}" fill="none" stroke="{BG}" stroke-width="2.4"/>'
body = f'<g clip-path="url(#img)">{scene}</g>'

# Patch grid in white over the photo.
for k in range(1, N):
    body += line(GX + k * P, GY, GX + k * P, GY + SZ, BG, 3.5)
    body += line(GX, GY + k * P, GX + SZ, GY + k * P, BG, 3.5)
body += rect(GX, GY, SZ, SZ, "none", INK, 2.2, 2)

# Text tokens; the middle one is the query.
TX, TW, TH = 480, 80, 34
tys = [200 + (k - 2) * 58 for k in range(5)]
Q = 2
qx, qy = TX, tys[Q]
for k in sorted(range(N * N), key=lambda k: w[k]):
    if w[k] < 0.05:
        continue
    px, py = centers[k]
    hot = w[k] > 0.15
    d = f"M{qx:.1f},{qy:.1f} C{qx - 100:.1f},{qy:.1f} {px + 120:.1f},{py:.1f} {px:.1f},{py:.1f}"
    body += (f'<path d="{d}" fill="none" stroke="{ACCENT if hot else GREY}" stroke-width="{1.4 + 16 * w[k]:.1f}" '
             f'stroke-linecap="round"/>')
    body += dot(px, py, 7 if hot else 5.5, ACCENT if hot else INK, BG, 2.4)
for k, ty in enumerate(tys):
    q = k == Q
    body += rect(TX, ty - TH / 2, TW, TH, ACCENT_SOFT if q else BG, ACCENT if q else INK, 2.4, 7)
    body += line(TX + 16, ty, TX + 16 + (40 if k % 2 else 28), ty, ACCENT if q else GREY, 3.2)
write(body, defs)
