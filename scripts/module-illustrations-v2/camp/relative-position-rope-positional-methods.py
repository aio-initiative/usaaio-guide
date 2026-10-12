"""RoPE: rotate the query by m*theta and the key by n*theta. Two places in the text with the same offset n - m
turn the pair to different absolute angles, but the angle between them, and so the dot product, is identical.

theta = 30 degrees, base angles q0 = 20 and k0 = 0 degrees. Left dial: m = 1, n = 3. Right dial: m = 5, n = 7.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

th, q0, k0 = math.radians(30), math.radians(20), 0.0
R = 100

def dial(cx, cy, m, n):
    P = lambda a, r: (cx + r * math.cos(a), cy - r * math.sin(a))
    out = poly([P(2 * math.pi * i / 120, R) for i in range(121)], GUIDE, 1.6)
    for i in range(12):                                     # one tick per position step
        out += line(*P(i * th, R - 9), *P(i * th, R + 9), GUIDE, 2.0)
    aq, ak = q0 + m * th, k0 + n * th
    sector = [P(0, 0)] + [P(aq + (ak - aq) * i / 40, 0.62 * R) for i in range(41)]
    out += poly(sector, "none", 0, ACCENT_SOFT, close=True)
    out += poly([P(aq + (ak - aq) * i / 40, 0.62 * R) for i in range(41)], ACCENT, 3.0)
    for a0, a1, c, r in ((q0, aq, INK, R + 18), (k0, ak, SECOND, R + 32)):   # how far each was turned
        steps = max(2, int(abs(a1 - a0) / 0.05))
        pts = [P(a0 + (a1 - a0) * i / steps, r) for i in range(steps + 1)]
        out += poly(pts[:-2], c, 2.2) + arrow(*pts[-3], *pts[-1], c, 2.2, 11)
        out += dot(*P(a0, r), 3.2, c)
    out += arrow(*P(0, 0), *P(ak, R - 4), SECOND, 4.0)
    out += arrow(*P(0, 0), *P(aq, R - 4), INK, 4.0)
    out += dot(*P(0, 0), 5.5, INK)
    return out

write(dial(150, 212, 1, 3) + dial(452, 212, 5, 7))
