"""The ravine from the lesson: a long, narrow, tilted quadratic valley f = (x^2 + K y^2) / 2 in rotated coordinates.
From one start, plain SGD (grey) at nearly its largest stable learning rate bounces wall to wall and stops short,
momentum (teal) damps the bounce and reaches the floor, and Adam (terracotta) curves smoothly to the minimum.
Every path is the real update rule iterated for the same number of steps."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

K = 60.0                               # curvature ratio across / along the valley
TILT = math.radians(-14)
grad = lambda x, y: (x, K * y)
f = lambda x, y: (x * x + K * y * y) / 2
start = (-7.0, 1.4)
STEPS = 60


def sgd(eta):
    p, out = start, [start]
    for _ in range(STEPS):
        g = grad(*p)
        p = (p[0] - eta * g[0], p[1] - eta * g[1])
        out.append(p)
    return out


def momentum(eta, beta=0.9):
    p, v, out = start, (0, 0), [start]
    for _ in range(STEPS):
        g = grad(*p)
        v = (beta * v[0] + g[0], beta * v[1] + g[1])
        p = (p[0] - eta * v[0], p[1] - eta * v[1])
        out.append(p)
    return out


def adam(eta, b1=0.9, b2=0.999, eps=1e-8):
    p, m, s, out = start, [0, 0], [0, 0], [start]
    for t in range(1, STEPS + 1):
        g = grad(*p)
        m = [b1 * m[i] + (1 - b1) * g[i] for i in range(2)]
        s = [b2 * s[i] + (1 - b2) * g[i] ** 2 for i in range(2)]
        step = [eta * (m[i] / (1 - b1 ** t)) / (math.sqrt(s[i] / (1 - b2 ** t)) + eps) for i in range(2)]
        p = (p[0] - step[0], p[1] - step[1])
        out.append(p)
    return out


SX, SY = 50, 80                        # px per unit; the across-valley axis is stretched so the bounce shows
OX, OY = 436, 238
c, s_ = math.cos(TILT), math.sin(TILT)
P = lambda x, y: (OX + SX * x * c - SY * y * s_, OY - SX * x * s_ - SY * y * c)

defs = '<clipPath id="plot"><rect x="28" y="28" width="544" height="344" rx="6"/></clipPath>'
body = '<g clip-path="url(#plot)">'
for lev, col, sw in [(0.4, GUIDE, 1.8), (3, GUIDE, 1.8), (10, GUIDE, 1.8), (24, GUIDE, 1.8), (46, GUIDE, 1.8), (f(*start), GREY, 1.8)]:
    a, b = math.sqrt(2 * lev), math.sqrt(2 * lev / K)
    body += poly([P(a * math.cos(2 * math.pi * k / 160), b * math.sin(2 * math.pi * k / 160)) for k in range(161)],
                 col, sw, close=True)
body += "</g>"

paths = [(sgd(1.8 / K), GREY, 2.2), (momentum(0.004), SECOND, 3.0), (adam(0.17), ACCENT, 3.4)]
for pts, col, sw in paths:
    body += poly([P(*p) for p in pts], col, sw)
for pts, col, sw in paths:
    body += dot(*P(*pts[-1]), 6.5, col)
body += dot(*P(0, 0), 7, BG, INK, 2.6) + dot(*P(0, 0), 2.5, INK)
body += dot(*P(*start), 8, INK)

write(body, defs)
