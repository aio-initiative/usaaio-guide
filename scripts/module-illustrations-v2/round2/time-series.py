"""The lesson's twelve-day series y = (4, 6, 5, 7, ..., 11) and its sliding windows: each row below is one window,
a bar over the three input days and a dot on the target day. The dashed cut sits between days 8 and 9; windows
whose target falls before it (terracotta) train, those whose target falls after it (teal, shaded period) validate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

y = [4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11]
X0, X1 = 64, 536
sx = lambda d: X0 + d * (X1 - X0) / 11
sy = lambda v: 188 - (v - 4) * 21
cut = (sx(8) + sx(9)) / 2
body = rect(cut, 34, 556 - cut, 336, SECOND_SOFT, "none", 0, 6)
body += axes(44, 34, 556, 204)
body += poly([(sx(d), sy(v)) for d, v in enumerate(y)], INK, 2.4)
for d, v in enumerate(y):
    body += dot(sx(d), sy(v), 6.5, INK)
for k, t in enumerate(range(3, 12)):
    ry = 224 + k * 17.5
    col = ACCENT if t <= 8 else SECOND
    body += line(sx(t - 3), ry, sx(t - 1), ry, GREY, 7)
    body += line(sx(t - 1) + 6, ry, sx(t) - 9, ry, GUIDE, 1.6)
    body += dot(sx(t), ry, 6.5, col)
body += line(cut, 28, cut, 376, INK, 2, dash="8 6")
write(body)
