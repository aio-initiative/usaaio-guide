"""One FNO layer acting on a function, with the heat equation as the operator being learned. Left: an input
initial condition a(x), a sum of sines, sampled on a coarse grid. Middle: its Fourier mode amplitudes |a_k| (grey)
and the same modes multiplied by the heat-flow multiplier e^(-k^2 t) (terracotta), kept only up to the cutoff
(dashed) and dropped beyond it. Right: the output u(x) = inverse transform of the kept modes, read off on the coarse
training grid (rings) and on a four-times finer grid (small dots) with the same multipliers: discretization invariance."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

MODES = [(1, 0.85, 0.3), (2, -0.5, 1.1), (3, 0.55, 2.0), (4, 0.4, 0.4), (6, -0.32, 1.7), (7, 0.24, 0.9)]
T, KMAX = 0.04, 4

def a(x):
    return sum(c * math.sin(k * x + p) for k, c, p in MODES)

def u(x):
    return sum(c * math.exp(-k * k * T) * math.sin(k * x + p) for k, c, p in MODES if k <= KMAX)

L = 2 * math.pi
TOP, BOT = 70, 330
MID = (TOP + BOT) / 2
SCALE = 58

def panel(x0, x1, f, col, coarse, fine):
    out = line(x0, MID, x1, MID, GUIDE, THIN)
    out += line(x0, TOP, x0, BOT, GUIDE, THIN)
    X = lambda x: x0 + (x1 - x0) * x / L
    Y = lambda v: MID - SCALE * v
    out += poly([(X(i * L / 240), Y(f(i * L / 240))) for i in range(241)], col, 3.0)
    if fine:
        for i in range(fine + 1):
            out += dot(X(i * L / fine), Y(f(i * L / fine)), 3.6, INK)
    for i in range(coarse + 1):
        out += ring(X(i * L / coarse), Y(f(i * L / coarse)), 6.5, INK, 2.4)
    return out

b = panel(42, 182, a, INK, 8, 0)
b += arrow(194, MID, 228, MID, GREY, 2.6)

# spectrum
SX0, SX1 = 244, 372
ks = list(range(1, 8))
bw = (SX1 - SX0) / len(ks)
amp = {k: abs(c) for k, c, p in MODES}
base = BOT - 10
b += line(SX0 - 4, base, SX1 + 4, base, GUIDE, THIN)
cut = SX0 + bw * KMAX + 1
b += line(cut, TOP - 4, cut, base + 6, GREY, THIN, "5 5")
b += rect(SX0 - 4, TOP - 4, cut - SX0 + 4, base - TOP + 4, ACCENT_SOFT, "none", 0, 3)
H = (base - TOP - 20) / 0.85
for i, k in enumerate(ks):
    x = SX0 + bw * i + 2
    h = amp.get(k, 0.0) * H
    if h:
        b += rect(x, base - h, bw * 0.4, h, GUIDE if k > KMAX else GREY, "none", 0, 1.5)
    if k <= KMAX and h:
        h2 = h * math.exp(-k * k * T)
        b += rect(x + bw * 0.46, base - h2, bw * 0.4, h2, ACCENT, "none", 0, 1.5)

b += arrow(388, MID, 422, MID, GREY, 2.6)
b += panel(432, 562, u, ACCENT, 8, 32)
write(b)
