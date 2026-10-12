"""Data Visualization: the lesson's overfitting plot. Five seeded runs of training and validation loss over 40
epochs; the plot shows the mean curve of each with a shaded min-max band across runs, and a dashed line with a dot
at the epoch where the mean validation loss is lowest (it falls, bottoms out near epoch 19, then climbs).
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(7)
E, R = 40, 5
train, val = [], []
for r in range(R):
    s = random.uniform(0.9, 1.1)
    noise_t = [random.gauss(0, 0.012) for _ in range(E)]
    noise_v = [random.gauss(0, 1) for _ in range(E)]
    train.append([1.6 * math.exp(-e / (7 * s)) + 0.08 + noise_t[e - 1] for e in range(1, E + 1)])
    # validation: fast fall, then a slow quadratic climb as the model overfits; spread widens with epochs
    o = random.uniform(-1, 1)
    val.append([1.5 * math.exp(-e / (6 * s)) + 0.33 + 0.0011 * s * max(0, e - 14) ** 2
                + o * (0.03 + 0.06 * e / E) + 0.008 * noise_v[e - 1] for e in range(1, E + 1)])
mean = lambda rows, e: sum(r[e] for r in rows) / R
X0, X1, Y0, Y1 = 60, 560, 40, 350
YMAX = 1.6
px = lambda e: X0 + 12 + (X1 - X0 - 24) * e / (E - 1)
py = lambda v: Y1 - (Y1 - Y0) * v / YMAX

body = axes(X0, Y0, X1, Y1)
for g in (0.5, 1.0, 1.5):
    body += line(X0, py(g), X1, py(g), "#eef0f2", 1.2)


def band(rows, fill):
    lo = [(px(e), py(min(r[e] for r in rows))) for e in range(E)]
    hi = [(px(e), py(max(r[e] for r in rows))) for e in range(E)]
    return poly(lo + hi[::-1], color="none", sw=0, fill=fill, close=True)


body += band(train, SECOND_SOFT) + band(val, ACCENT_SOFT)
body += poly([(px(e), py(mean(train, e))) for e in range(E)], SECOND, 3)
vm = [mean(val, e) for e in range(E)]
body += poly([(px(e), py(vm[e])) for e in range(E)], ACCENT, 3.2)
best = min(range(E), key=lambda e: vm[e])
body += line(px(best), Y1, px(best), Y0 + 10, INK, 1.8, dash="6 5")
body += dot(px(best), py(vm[best]), 8, INK, BG, 2.5)
write(body)
