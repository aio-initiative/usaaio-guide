"""The training loop: a batch enters a five-step cycle (forward, loss, zero_grad, backward, step) that repeats,
and per-epoch curves show the training loss falling while the validation loss turns up after its best epoch."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

body = ""
# --- the cycle of five steps
ccx, ccy, R = 186, 222, 104
angles = [-90 + 72 * k for k in range(5)]
nodes = [(ccx + R * math.cos(math.radians(t)), ccy + R * math.sin(math.radians(t))) for t in angles]
gap = 15  # degrees left clear around each node
for k in range(5):
    t0, t1 = angles[k] + gap, angles[k] + 72 - gap
    pts = [(ccx + R * math.cos(math.radians(t0 + (t1 - t0) * s / 20)),
            ccy + R * math.sin(math.radians(t0 + (t1 - t0) * s / 20))) for s in range(21)]
    color = ACCENT if k == 3 else INK      # backward: gradients flow from the loss into the weights
    body += poly(pts[:-3], color, 2.6)
    body += arrow(pts[-4][0], pts[-4][1], pts[-1][0], pts[-1][1], color, 2.6)
for k, (x, y) in enumerate(nodes):
    if k == 1:
        body += dot(x, y, 13, ACCENT)                 # the loss
    elif k == 2:
        body += ring(x, y, 12, INK, 2.6)              # zero_grad clears the gradients
    else:
        body += dot(x, y, 12, INK)
# a stack of batches feeding the first step
for s in range(3):
    body += rect(34 + s * 7, 52 + s * 7, 48, 34, fill=BG if s < 2 else ACCENT_SOFT, stroke=GREY if s < 2 else ACCENT, sw=2)
body += arrow(100, 82, nodes[0][0] - 20, nodes[0][1] - 4, GREY, 2.2)

# --- loss curves over epochs
x0, y0, x1, y1 = 352, 48, 562, 352
body += axes(x0, y0, x1, y1)
E = 40
tr = lambda e: 2.0 * math.exp(-e / 5) + 0.01
c = 0.5 * math.exp(-13 / 4)
va = lambda e: 0.3 + 2.0 * math.exp(-e / 4) + c * e
top = 2.0
X = lambda e: x0 + 12 + (x1 - x0 - 16) * (e - 1) / (E - 1)
Y = lambda v: y1 - 8 - (y1 - y0 - 16) * v / top
body += poly([(X(e), Y(tr(e))) for e in range(1, E + 1)], INK, 2.6)
body += poly([(X(e), Y(va(e))) for e in range(1, E + 1)], ACCENT, 2.6)
best = min(range(1, E + 1), key=va)
body += line(X(best), Y(va(best)) + 12, X(best), y1, GUIDE, THIN, "5 5")
body += ring(X(best), Y(va(best)), 9, ACCENT, 2.8)

write(body)
