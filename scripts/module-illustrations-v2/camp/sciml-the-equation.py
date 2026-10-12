"""What the equation gives and what it leaves to be found: the slope field of dy/dt = -y (a short segment of slope
-y at each grid point, steep where y is large, flat near y = 0), teal members of the family c e^(-t) that also follow
the slopes (teal), and the one heavy terracotta solution e^(-t) picked out by the starting dot y(0) = 1."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

PX0, PX1, PY0, PY1 = 70, 560, 40, 350
TMAX, YMAX = 3.0, 1.9
tx = lambda t: PX0 + (PX1 - PX0) * t / TMAX
vy = lambda v: PY1 - (PY1 - PY0) * v / YMAX
sx, sy = (PX1 - PX0) / TMAX, (PY1 - PY0) / YMAX   # pixels per unit

b = axes(PX0 - 10, PY0 - 6, PX1 + 14, PY1)

# slope field: segment of fixed pixel length along direction (1, -y) in data units
L = 26
for i in range(10):
    for j in range(1, 8):
        t, y = TMAX * (i + 0.5) / 10, YMAX * (j - 0.35) / 7
        dx, dy = sx, sy * y          # screen direction: right sx, down sy*y
        n = math.hypot(dx, dy)
        ux, uy = dx / n * L / 2, dy / n * L / 2
        cx, cy = tx(t), vy(y)
        b += line(cx - ux, cy - uy, cx + ux, cy + uy, GREY, 2.4)

fine = [i * TMAX / 200 for i in range(201)]
for c in (0.45, 1.75):
    b += poly([(tx(t), vy(c * math.exp(-t))) for t in fine], SECOND, 2.4)
b += poly([(tx(t), vy(math.exp(-t))) for t in fine], ACCENT, 4.2)
b += dot(tx(0), vy(1), 9, ACCENT, BG, 3)
write(b)
