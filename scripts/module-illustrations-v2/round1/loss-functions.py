"""The module's example: one constant c for the targets 1, 2, 2, 3, 12. Mean squared error (terracotta) is a
parabola whose bottom is the mean, 4, dragged right by the outlier; mean absolute error (teal) is piecewise linear
with its bottom at the median, 2. The five targets sit on the axis below."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

ys = [1, 2, 2, 3, 12]
lo, hi = -0.5, 13.5
X0, X1, Y0, Y1 = 50, 550, 40, 330
def sx(c): return X0 + (c - lo) / (hi - lo) * (X1 - X0)
def mse(c): return sum((y - c) ** 2 for y in ys) / len(ys)
def mae(c): return sum(abs(y - c) for y in ys) / len(ys)
cs = [lo + (hi - lo) * i / 600 for i in range(601)]
# each curve on its own vertical scale: minimum near the bottom, value at c = 0 at a fixed height
def scaled(f, cmin, ybot, ytop_at0):
    fmin, f0 = f(cmin), f(0)
    k = (ybot - ytop_at0) / (f0 - fmin)
    return lambda c: ybot - (f(c) - fmin) * k
mean, med = sum(ys) / len(ys), 2
fs = scaled(mse, mean, 250, 205)
fa = scaled(mae, med, 290, 245)
b = axes(X0, Y0, X1, Y1)
sq = [(sx(c), fs(c)) for c in cs if fs(c) > Y0]
ab = [(sx(c), fa(c)) for c in cs if fa(c) > Y0]
b += line(sx(mean), fs(mean), sx(mean), Y1, ACCENT, THIN, dash="6 5")
b += line(sx(med), fa(med), sx(med), Y1, SECOND, THIN, dash="6 5")
b += poly(ab, SECOND, 2.8)
b += poly(sq, ACCENT, 2.8)
b += dot(sx(mean), fs(mean), 7, ACCENT)
b += dot(sx(med), fa(med), 7, SECOND)
# targets on the axis, the repeated 2 stacked
seen = {}
for y in ys:
    k = seen.get(y, 0); seen[y] = k + 1
    b += dot(sx(y), Y1 + 22 + 18 * k, 7, INK)
write(b)
