"""The lesson's logits after "the cat sat" at temperatures 0.5, 1 and 2: sorted probability bars, the top-p 0.9
nucleus in terracotta and the dropped tail in grey, under a wedge that widens as the temperature rises. A low temperature keeps one token; a high one keeps six."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

z = [-1.667, -0.639, -1.028, 1.530, 3.197, -1.530, 0.776, -0.776]
T, B = 85, 360
panels = [(0.5, 45), (1.0, 230), (2.0, 415)]
PW = 140
# the wedge: temperature rising left to right
body = poly([(45, 42), (555, 30), (555, 54)], GREY, 1.2, fill=GUIDE, close=True)
for tau, x0 in panels:
    e = [math.exp(v / tau) for v in z]
    s = sum(e)
    p = sorted((v / s for v in e), reverse=True)
    cum, kept = 0.0, 0
    for q in p:
        cum += q; kept += 1
        if cum >= 0.9:
            break
    bw = PW / len(p)
    body += line(x0 - 4, B, x0 + PW + 4, B, GUIDE, THIN)
    for i, q in enumerate(p):
        h = q * (B - T)
        bx = x0 + i * bw + 2.5
        if i < kept:
            body += rect(bx, B - h, bw - 5, h, ACCENT, ACCENT, 1.5, rx=2)
        else:
            body += rect(bx, B - max(h, 1.5), bw - 5, max(h, 1.5), GUIDE, GREY, 1.2, rx=2)
    cx = x0 + kept * bw
    body += line(cx, T - 5, cx, B + 14, INK, 2, dash="6 5")
write(body)
