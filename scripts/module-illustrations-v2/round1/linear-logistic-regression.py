"""Left: the least-squares line through the lesson's four points, with residuals. Right: a sigmoid fitted to 0/1 labels."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# left panel: points (0,1), (1,2), (2,2), (3,4), fitted by least squares
pts = [(0, 1), (1, 2), (2, 2), (3, 4)]
n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
w1 = (n * sxy - sx * sy) / (n * sxx - sx * sx); w0 = (sy - w1 * sx) / n
L0, L1, T, B = 50, 280, 50, 350
LX = lambda x: L0 + 20 + (x + 0.3) * (L1 - L0 - 20) / 3.7
LY = lambda y: B - (y - 0.0) * (B - T) / 4.6
body = axes(L0, T, L1, B)
body += line(LX(-0.3), LY(w0 - 0.3 * w1), LX(3.4), LY(w0 + 3.4 * w1), ACCENT, 3)
for x, y in pts:
    body += line(LX(x), LY(y), LX(x), LY(w0 + w1 * x), GREY, 2, dash="5 4")
for x, y in pts:
    body += dot(LX(x), LY(y), 8, INK)

# right panel: the sigmoid, with 0/1 labels on the floor and ceiling
R0, R1 = 320, 550
sig = lambda z: 1 / (1 + math.exp(-z))
RX = lambda z: R0 + (z + 6) * (R1 - R0) / 12
RY = lambda p: B - 15 - p * (B - T - 30)
body += axes(R0, T, R1, B)
body += f'<rect x="{RX(0):.1f}" y="{RY(1):.1f}" width="{R1 - RX(0):.1f}" height="{RY(0) - RY(1):.1f}" fill="{ACCENT_SOFT}"/>'
body += line(R0, RY(0.5), R1, RY(0.5), GUIDE, 1.6, dash="6 5")
body += line(RX(0), RY(1) - 6, RX(0), RY(0), GREY, 1.6, dash="6 5")
body += poly([(RX(z), RY(sig(z))) for z in [i / 20 - 6 for i in range(241)]], ACCENT, 3)
for z in (-0.6, 1.3, 2.4, 3.6, 4.9):
    body += dot(RX(z), RY(1), 7.5, ACCENT, INK, 1.6)
for z in (-5.0, -3.6, -2.5, -1.4, 0.8):
    body += ring(RX(z), RY(0), 7.5, INK, 2.2)
body += dot(RX(0), RY(0.5), 5.5, INK)
write(body)
