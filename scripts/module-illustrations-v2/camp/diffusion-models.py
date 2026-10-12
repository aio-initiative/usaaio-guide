"""Diffusion as a whole: data diffuses into noise, and generation runs the flow backwards.

Time runs left (data, t = 0) to right (pure noise, t = 1) under the VP / DDPM schedule (beta from 0.1 to 20).
The data is a two-mode Gaussian mixture, so every marginal p_t is known exactly: the terracotta band is the region
where p_t is above a fixed fraction of its peak, two lobes that merge into one Gaussian. The side profiles are the
true densities at t = 0 (terracotta, two sharp modes) and t = 1 (grey, a standard normal).
The ink curves are probability-flow ODE trajectories integrated backwards with RK4 using the exact mixture score,
starting from evenly spaced normal quantiles at t = 1 and landing on the two modes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

MU, S0, PI = (1.25, -1.25), 0.13, (0.5, 0.5)
B0, B1 = 0.1, 20.0
T0 = 1e-3


def beta(t):
    return B0 + (B1 - B0) * t


def abar(t):
    return math.exp(-(B0 * t + 0.5 * (B1 - B0) * t * t))


def comps(t):
    a = abar(t)
    s = math.sqrt(a)
    v = a * S0 * S0 + 1 - a
    return [(p, s * m, v) for p, m in zip(PI, MU)]


def pdf(x, t):
    return sum(p * math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v) for p, m, v in comps(t))


def score(x, t):
    ws = [(p * math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(v), -(x - m) / v) for p, m, v in comps(t)]
    tot = sum(w for w, _ in ws)
    return sum(w * g for w, g in ws) / tot


def ode(x, t):
    return -0.5 * beta(t) * (x + score(x, t))


# Canvas mapping: t on a square-root axis so the early split gets room.
TX0, TX1 = 128, 472
VY0, VY1 = 40, 360
XR = 2.9


def tx(t):
    return TX0 + math.sqrt(t) * (TX1 - TX0)


def vy(x):
    return (VY0 + VY1) / 2 - x / XR * (VY1 - VY0) / 2


body = ""
# The high-density band, column by column.
N = 172
grid = [-XR + 2 * XR * j / 400 for j in range(401)]
for i in range(N):
    t = T0 + (1 - T0) * (i / (N - 1)) ** 2
    t_next = T0 + (1 - T0) * ((i + 1) / (N - 1)) ** 2
    x_l, x_r = tx(t), tx(min(t_next, 1))
    ps = [pdf(g, t) for g in grid]
    top = max(ps)
    inside = [p > 0.12 * top for p in ps]
    j = 0
    while j < len(grid):
        if inside[j]:
            k = j
            while k + 1 < len(grid) and inside[k + 1]:
                k += 1
            body += rect(x_l, vy(grid[k]), max(x_r - x_l, 0) + 1.2, vy(grid[j]) - vy(grid[k]),
                         ACCENT_SOFT, "none", 0, 0)
            j = k + 1
        else:
            j += 1

# Side profiles: true densities at t = 0 (left, outward) and t = 1 (right, outward).
def profile(t, x_edge, sign, color, width):
    xs = [-XR + 2 * XR * j / 600 for j in range(601)]
    ps = [pdf(x, t) for x in xs]
    m = max(ps)
    pts = [(x_edge + sign * width * p / m, vy(x)) for x, p in zip(xs, ps)]
    return poly(pts, color, 2.8)


body += profile(T0, TX0 - 8, -1, ACCENT, 84)
body += profile(1.0, TX1 + 8, 1, GREY, 84)

# Probability-flow trajectories from noise back to data.
def inv_norm(p):
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return mid


M, STEPS = 12, 600
for q in range(M):
    x = inv_norm((q + 0.5) / M)
    pts = []
    for n in range(STEPS + 1):
        u = 1 - n / STEPS
        t = T0 + (1 - T0) * u * u
        pts.append((tx(t), vy(x)))
        if n == STEPS:
            break
        u2 = 1 - (n + 1) / STEPS
        h = (T0 + (1 - T0) * u2 * u2) - t
        k1 = ode(x, t)
        k2 = ode(x + h / 2 * k1, t + h / 2)
        k3 = ode(x + h / 2 * k2, t + h / 2)
        k4 = ode(x + h * k3, t + h)
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    body += poly(pts, INK, 1.9)
    body += dot(*pts[0], 4.5, GREY)
    body += dot(*pts[-1], 4.5, INK)

# Direction of generation: noise to data.
body += arrow(TX1 - 30, VY1 + 18, TX0 + 30, VY1 + 18, ACCENT, 2.8)
write(body)
