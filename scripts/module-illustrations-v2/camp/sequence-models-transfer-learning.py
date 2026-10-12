"""An RNN unrolled over seven steps with the loss at the last one. Below each step, how much gradient reaches it:
the plain RNN's product of tanh slopes times w_h shrinks step by step (terracotta), while the LSTM cell path,
whose factor is just the forget gate near 1, keeps nearly all of it (teal)."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(3)
T = 7
xs = [75 + 75 * i for i in range(T)]
xin = [random.gauss(0, 0.3) for _ in range(T)]

# Plain RNN: h_t = tanh(w_h h_{t-1} + w_x x_t); dh_T/dh_t = prod over later steps of w_h (1 - h_k^2).
w_h, h, hs = 0.8, 0.0, []
for x in xin:
    h = math.tanh(w_h * h + x)
    hs.append(h)
rnn = [1.0] * T
for t in range(T - 2, -1, -1):
    rnn[t] = rnn[t + 1] * w_h * (1 - hs[t + 1] ** 2)
# LSTM cell path: dc_T/dc_t = prod of forget gates f_k = sigmoid(b_f + noise), forget bias b_f = 3.
f = [1 / (1 + math.exp(-(3 + random.gauss(0, 0.4)))) for _ in range(T)]
lstm = [1.0] * T
for t in range(T - 2, -1, -1):
    lstm[t] = lstm[t + 1] * f[t + 1]

body = ""
BY0, BY1, BW = 82, 130, 48
for i, x in enumerate(xs):
    # Input token feeding the hidden state from below.
    body += rect(x - 9, 160, 18, 18, GUIDE, GREY, 1.6, 3)
    body += arrow(x, 158, x, BY1 + 3, GREY, 2.0)
    if i < T - 1:
        body += arrow(x + BW / 2 + 2, (BY0 + BY1) / 2, xs[i + 1] - BW / 2 - 2, (BY0 + BY1) / 2, INK, 2.4)
    last = i == T - 1
    body += rect(x - BW / 2, BY0, BW, BY1 - BY0, ACCENT_SOFT if last else BG, ACCENT if last else INK, 2.4, 6)
    # tanh glyph inside each hidden box.
    body += poly([(x - 14 + 28 * k / 30, (BY0 + BY1) / 2 - 11 * math.tanh(3 * (k / 30 - 0.5) * 2))
                  for k in range(31)], GREY, 1.8)
# Loss at the final step.
body += arrow(xs[-1], BY0 - 2, xs[-1], 50, ACCENT, 2.4)
body += dot(xs[-1], 42, 9, ACCENT)
# Gradient reaching each step, as paired bars on a shared axis.
BASE, TOP = 362, 205
body += line(40, BASE, 560, BASE, GUIDE, THIN)
body += line(40, TOP, 560, TOP, GUIDE, THIN, "5 5")
for x, a, b in zip(xs, rnn, lstm):
    body += rect(x - 19, BASE - (BASE - TOP) * b, 17, (BASE - TOP) * b, SECOND_SOFT, SECOND, 2.0, 2)
    body += rect(x + 2, BASE - (BASE - TOP) * a, 17, (BASE - TOP) * a, ACCENT_SOFT, ACCENT, 2.0, 2)
# Decay curves through the bar tops.
body += poly([(x + 10.5, BASE - (BASE - TOP) * a) for x, a in zip(xs, rnn)], ACCENT, 1.6, dash="4 4")
write(body)
