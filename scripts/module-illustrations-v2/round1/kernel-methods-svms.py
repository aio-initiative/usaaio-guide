"""The maximum-margin classifier: two classes (terracotta dots +1, white rings -1), the hard-margin SVM
boundary w.x + b = 0 fitted by scikit-learn, the two dashed margin lines w.x + b = +-1 with the empty strip
between them shaded, the support vectors circled, and an arrow across the margin, width 2 / |w|."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
from sklearn.svm import SVC
pos = [(6.2, 6.0), (7.4, 4.6), (7.0, 7.2), (8.4, 6.2), (8.6, 3.8), (5.6, 7.6), (8.2, 8.0)]
neg = [(2.0, 3.0), (3.4, 1.6), (1.6, 5.0), (3.0, 4.4), (4.4, 2.6), (2.4, 1.2), (1.2, 2.2)]
Xd = np.array(pos + neg); yd = np.array([1] * len(pos) + [-1] * len(neg))
m = SVC(kernel="linear", C=1e6).fit(Xd, yd)
w, b0 = m.coef_[0], m.intercept_[0]
X0, X1, Y0, Y1 = 50, 550, 40, 360
# equal scale on both axes, centred on (5, 5), so the margin arrow is truly perpendicular
unit = (Y1 - Y0) / 7.6
ylo, yhi = 5.0 - 3.8, 5.0 + 3.8
def sxx(v): return (X0 + X1) / 2 + (v - 5.0) * unit
def sy(v): return (Y0 + Y1) / 2 - (v - 5.0) * unit
def clip(level):
    # points of w.x + b = level across the visible y range
    out = []
    for yv in (ylo - 1, yhi + 1):
        xv = (level - b0 - w[1] * yv) / w[0]
        out.append((sxx(xv), sy(yv)))
    return out
body = ""
strip = clip(-1) + clip(1)[::-1]
body += f'<clipPath id="c"><rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}"/></clipPath>'
g = poly(strip, ACCENT_SOFT, 0, fill=ACCENT_SOFT, close=True)
g += poly(clip(-1), GREY, 2, dash="8 6") + poly(clip(1), GREY, 2, dash="8 6")
g += poly(clip(0), INK, 3)
body += f'<g clip-path="url(#c)">{g}</g>'
for i in m.support_:
    body += f'<circle cx="{sxx(Xd[i][0]):.1f}" cy="{sy(Xd[i][1]):.1f}" r="17" fill="none" stroke="{INK}" stroke-width="2"/>'
for p in pos: body += dot(sxx(p[0]), sy(p[1]), 9, ACCENT)
for p in neg: body += ring(sxx(p[0]), sy(p[1]), 9, INK, 2.6)
# margin arrow, perpendicular to the boundary, between the margin lines
n = w / np.linalg.norm(w)
c = np.array([5.0, 5.0]); c = c - (w @ c + b0) / (w @ w) * w   # foot on the boundary
c = c + np.array([-n[1], n[0]]) * 2.6
a = c - n / np.linalg.norm(w); z = c + n / np.linalg.norm(w)
body += arrow(sxx(c[0]), sy(c[1]), sxx(a[0]), sy(a[1]), ACCENT, 2.4)
body += arrow(sxx(c[0]), sy(c[1]), sxx(z[0]), sy(z[1]), ACCENT, 2.4)
write(body)
