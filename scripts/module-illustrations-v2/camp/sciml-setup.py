"""One equation, three settings, three separate solver runs: the drug-clearance law dy/dt = -k y from y(0) = 1 for
k = 0.5 (grey), 1 (teal) and 2 (terracotta). Each curve is the true e^(-k t); the dots on it are that run's own
solver steps (classical RK4 with h = 0.4), so every new k is a fresh march across the time axis."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

PX0, PX1, PY0, PY1 = 70, 560, 50, 350
TMAX = 4.0
tx = lambda t: PX0 + (PX1 - PX0) * t / TMAX
vy = lambda v: PY1 - (PY1 - PY0) * v / 1.05

def rk4(k, h, n):
    f = lambda y: -k * y
    y, out = 1.0, [(0.0, 1.0)]
    for i in range(n):
        a = f(y); b2 = f(y + h * a / 2); c = f(y + h * b2 / 2); d = f(y + h * c)
        y += h * (a + 2 * b2 + 2 * c + d) / 6
        out.append(((i + 1) * h, y))
    return out

b = axes(PX0 - 10, PY0 - 14, PX1 + 14, PY1)
b += line(PX0 - 10, vy(1), PX1 + 14, vy(1), GUIDE, THIN, "4 5")

runs = [(0.5, GREY), (1.0, SECOND), (2.0, ACCENT)]
fine = [i * TMAX / 200 for i in range(201)]
# soft fill under the fastest run
fast = [(tx(t), vy(math.exp(-2 * t))) for t in fine]
b += poly(fast + [(tx(TMAX), vy(0)), (tx(0), vy(0))], "none", 0, ACCENT_SOFT, close=True)
for k, col in runs:
    b += poly([(tx(t), vy(math.exp(-k * t))) for t in fine], col, 3.2)
for k, col in runs:
    for t, y in rk4(k, 0.4, 10)[1:]:
        b += ring(tx(t), vy(y), 6, col, 2.6)
b += dot(tx(0), vy(1), 8, INK)
write(b)
