"""Reinforcement learning: the module's 4x4 grid world, solved two ways.

Left: the grid shaded by V*, computed by value iteration (step cost 0.04, goal pays 1, gamma 0.95), darker
terracotta = more valuable. Every optimal action in each cell is drawn as an arrow, so the tied cells show both
"right" and "down". Start is a ring in the top-left corner, the goal a filled dot in the bottom-right.
Right: Q-learning on the same grid with no model, the gap max_s |max_a Q(s,a) - V*(s)| falling over the first 600 episodes on a log scale,
run with the lesson's settings (alpha 0.1, annealed epsilon, seed 0).
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, GOAL, COST, GAMMA = 4, (3, 3), -0.04, 0.95
ACT = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
STATES = [(r, c) for r in range(N) for c in range(N)]


def step(s, a):
    if s == GOAL:
        return s, 0.0
    r, c = s[0] + ACT[a][0], s[1] + ACT[a][1]
    if not (0 <= r < N and 0 <= c < N):
        r, c = s
    return (r, c), (1.0 if (r, c) == GOAL else COST)


V = {s: 0.0 for s in STATES}
for _ in range(100):
    V = {s: 0.0 if s == GOAL else max(step(s, a)[1] + GAMMA * V[step(s, a)[0]] for a in ACT) for s in STATES}

rng = random.Random(0)
Q = {(s, a): 0.0 for s in STATES for a in ACT}
EP, gaps = 4000, []
for ep in range(EP):
    s, eps = (0, 0), max(0.05, 1.0 - ep / (EP * 0.5))
    for _ in range(100):
        if s == GOAL:
            break
        a = rng.choice(list(ACT)) if rng.random() < eps else max(ACT, key=lambda a: Q[(s, a)])
        s2, r = step(s, a)
        nxt = 0.0 if s2 == GOAL else max(Q[(s2, b)] for b in ACT)
        Q[(s, a)] += 0.1 * (r + GAMMA * nxt - Q[(s, a)])
        s = s2
    if ep < 600:
        gaps.append(max(abs(max(Q[(s, a)] for a in ACT) - V[s]) for s in STATES if s != GOAL))


def mix(t):
    a, b = (0xfb, 0xf3, 0xf0), (0xd9, 0x8a, 0x70)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


body = ""
G0X, G0Y, C = 40, 40, 80
vmin, vmax = min(V.values()), max(v for s, v in V.items() if s != GOAL)
vmin = min(v for s, v in V.items() if s != GOAL)
for (r, c) in STATES:
    x, y = G0X + c * C, G0Y + r * C
    t = (V[(r, c)] - vmin) / (vmax - vmin) if (r, c) != GOAL else 1.0
    body += f'<rect x="{x}" y="{y}" width="{C}" height="{C}" fill="{mix(t)}" stroke="{BG}" stroke-width="3"/>'
body += rect(G0X, G0Y, N * C, N * C, "none", INK, 2.4, 2)
for (r, c) in STATES:
    cx, cy = G0X + c * C + C / 2, G0Y + r * C + C / 2
    if (r, c) == GOAL:
        body += dot(cx, cy, 15, INK)
        continue
    q = {a: step((r, c), a)[1] + GAMMA * V[step((r, c), a)[0]] for a in ACT}
    best = [a for a in ACT if q[a] > max(q.values()) - 1e-9]
    start = (r, c) == (0, 0)
    for a in best:
        dy, dx = ACT[a]
        if len(best) == 1:
            body += arrow(cx - dx * 22, cy - dy * 22, cx + dx * 26, cy + dy * 26, INK, 3.2, 14)
        else:
            body += arrow(cx, cy, cx + dx * 32, cy + dy * 32, INK, 3.2, 14)
    body += ring(cx, cy, 11, INK, 3) if start else (dot(cx, cy, 4.5, INK) if len(best) > 1 else "")

# Learning curve of model-free Q-learning.
PX0, PX1, PY0, PY1 = 395, 560, 60, 330
body += axes(PX0, PY0, PX1, PY1)
LO = -5.0                                   # log10 floor of the plotted gap
lg = [max(LO, math.log10(max(gv, 1e-12))) for gv in gaps]
pts = [(PX0 + 6 + (PX1 - PX0 - 10) * i / (len(lg) - 1), PY1 - 4 - (PY1 - PY0 - 10) * (v - LO) / (lg[0] - LO))
       for i, v in enumerate(lg)]
body += poly(pts + [(pts[-1][0], PY1), (pts[0][0], PY1)], "none", 0, ACCENT_SOFT, close=True)
body += poly(pts, ACCENT, 2.8)
body += line(PX0, PY1 - 4, PX1, PY1 - 4, GREY, THIN, "6 6")
write(body)
