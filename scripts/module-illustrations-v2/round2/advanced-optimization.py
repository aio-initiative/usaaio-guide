"""The lesson's constrained worked example and its dual: minimize x^2 + y^2 subject to x + y >= 1.
Left: circular contours of x^2 + y^2, the feasible half-plane shaded terracotta, the constraint line, and the
optimum (1/2, 1/2) where the smallest feasible circle touches the line, with grad f (ink) and mu times the
constraint gradient (terracotta) equal and opposite there, so the Lagrangian's gradient is zero (KKT, mu = 1). Right: the dual function g(mu) = mu - mu^2/2, concave, its peak at mu = 1
touching the dashed primal optimum p* = 1/2 (strong duality), with a few lower-bound values below it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# ---- left panel: primal geometry ----
S = 118                                  # px per unit
OX, OY = 150, 250                        # canvas position of the origin
P = lambda x, y: (OX + x * S, OY - y * S)
X0, X1, Y0, Y1 = 40, 340, 40, 360        # panel box

defs = f'<clipPath id="lp"><rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}"/></clipPath>'
body = f'<g clip-path="url(#lp)">'
# feasible half-plane x + y >= 1: polygon above the line, clipped to the panel
far = 6
body += poly([P(1 - far, far), P(1 + far, -far), P(far * 2, far * 2)], "none", 0, ACCENT_SOFT, close=True)
# coordinate guides through the origin
body += line(X0, OY, X1, OY, GUIDE, THIN) + line(OX, Y0, OX, Y1, GUIDE, THIN)
# contours of x^2 + y^2
for r, col in [(0.35, GREY), (math.sqrt(0.5), INK), (1.05, GREY), (1.4, GREY), (1.75, GUIDE)]:
    pts = [P(r * math.cos(2 * math.pi * k / 120), r * math.sin(2 * math.pi * k / 120)) for k in range(121)]
    body += poly(pts, col, 2.6 if col == INK else 1.8, close=True)
# constraint line x + y = 1
body += line(*P(1 - far, far), *P(1 + far, -far), ACCENT, 2.8)
body += "</g>"
body += rect(X0, Y0, X1 - X0, Y1 - Y0, "none", GUIDE, THIN, 0)

xs = P(0.5, 0.5)
# grad f at x* = (1, 1) direction (outward), and the constraint normal grad(1 - x - y) = (-1, -1) times mu = 1
u = 1 / math.sqrt(2)
L1 = 92
body += arrow(*xs, xs[0] + u * L1, xs[1] - u * L1, INK, 3.0)
body += arrow(*xs, xs[0] - u * L1 * 0.85, xs[1] + u * L1 * 0.85, ACCENT, 3.0)
body += dot(OX, OY, 5, GREY)
body += dot(*xs, 8.5, ACCENT, BG, 2.5)

# ---- right panel: dual function ----
RX0, RX1, RY0, RY1 = 382, 560, 70, 340
mu_max = 2.0
gy = lambda g: RY1 - 20 - (g + 0.0) / 0.62 * (RY1 - 20 - RY0)   # g in [0, 0.62] -> canvas y
mx = lambda m: RX0 + m / mu_max * (RX1 - RX0)
body += axes(RX0, RY0, RX1, RY1 - 20)
pstar = 0.5
body += line(RX0, gy(pstar), RX1, gy(pstar), INK, 2.2, "9 7")
g = lambda m: m - m * m / 2
pts = [(mx(m), gy(g(m))) for m in [mu_max * k / 120 for k in range(121)]]
body += poly(pts, SECOND, 3.0)
# a few weak-duality bounds: vertical gaps from g(mu) up to p*
for m in (0.35, 0.65, 1.5):
    body += line(mx(m), gy(g(m)) - 7, mx(m), gy(pstar) + 7, GREY, 1.8)
    body += dot(mx(m), gy(g(m)), 5.5, SECOND)
body += dot(mx(1), gy(pstar), 8.5, ACCENT, BG, 2.5)

write(body, defs)
