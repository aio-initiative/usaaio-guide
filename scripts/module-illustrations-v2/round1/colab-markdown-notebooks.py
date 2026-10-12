"""Colab Notebooks: a notebook run fresh from the top. Three cells in order, each with its run marker: a code cell
that loads the data, a code cell whose output is the lesson's least-squares fit (the five points of train.csv and
the true line w = 1.99, b = 0.05), and a text cell holding the normal equations as bracketed matrices. A terracotta
arrow down the gutter is the restart-and-run-all order.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

xs, ys = [1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1]
n = len(xs)
sx, sy = sum(xs), sum(ys)
sxx, sxy = sum(x * x for x in xs), sum(x * y for x, y in zip(xs, ys))
w = (n * sxy - sx * sy) / (n * sxx - sx * sx)
b = (sy - w * sx) / n

L, R = 120, 560            # cell left/right edges
body = ""
# run-order arrow in the gutter
body += arrow(56, 46, 56, 352, ACCENT, 3)


def code_lines(x, y, lens):
    out = ""
    for k, ln in enumerate(lens):
        out += line(x, y + k * 16, x + ln, y + k * 16, GREY, 6)
    return out


def marker(y):
    return dot(96, y, 8, INK)


# cell 1: code
c1 = 40
body += rect(L, c1, R - L, 52, fill="#f6f7f8", stroke=GUIDE, sw=1.6, rx=6) + marker(c1 + 26)
body += code_lines(L + 20, c1 + 18, [150, 230])
# cell 2: code + output plot
c2 = 110
body += rect(L, c2, R - L, 36, fill="#f6f7f8", stroke=GUIDE, sw=1.6, rx=6) + marker(c2 + 18)
body += code_lines(L + 20, c2 + 18, [190])
ox0, ox1, oy0, oy1 = L + 24, L + 200, c2 + 50, c2 + 150
body += axes(ox0, oy0, ox1, oy1)
X = lambda x: ox0 + 10 + (ox1 - ox0 - 20) * (x - 0.5) / 5
Y = lambda y: oy1 - (oy1 - oy0) * y / 11
body += line(X(0.6), Y(w * 0.6 + b), X(5.4), Y(w * 5.4 + b), ACCENT, 3)
for x, y in zip(xs, ys):
    body += line(X(x), Y(y), X(x), Y(w * x + b), GREY, 1.6, dash="3 3")
for x, y in zip(xs, ys):
    body += dot(X(x), Y(y), 6.5, INK)
# printed numbers beside the plot, as two short output lines
body += line(L + 240, oy0 + 30, L + 330, oy0 + 30, GREY, 6) + line(L + 240, oy0 + 50, L + 300, oy0 + 50, GREY, 6)
# cell 3: markdown with the normal equations as bracketed matrices
c3 = 278
body += rect(L, c3, R - L, 82, fill=BG, stroke=GUIDE, sw=1.6, rx=6) + ring(96, c3 + 41, 7, INK, 2.4)


def bracket(x, y, wd, ht):
    t = 7
    return (poly([(x + t, y), (x, y), (x, y + ht), (x + t, y + ht)], INK, 2.2) +
            poly([(x + wd - t, y), (x + wd, y), (x + wd, y + ht), (x + wd - t, y + ht)], INK, 2.2))


my = c3 + 14
mh = 54
cx = L + 40
body += bracket(cx, my, 70, mh)
for i in range(2):
    for j in range(2):
        body += dot(cx + 22 + 26 * j, my + 15 + 24 * i, 5, INK)
cx2 = cx + 86
body += bracket(cx2, my, 34, mh)
for i in range(2):
    body += dot(cx2 + 17, my + 15 + 24 * i, 5, ACCENT)
ex = cx2 + 52
body += line(ex, my + mh / 2 - 5, ex + 18, my + mh / 2 - 5, INK, 2.4) + line(ex, my + mh / 2 + 5, ex + 18, my + mh / 2 + 5, INK, 2.4)
cx3 = ex + 34
body += bracket(cx3, my, 34, mh)
for i in range(2):
    body += dot(cx3 + 17, my + 15 + 24 * i, 5, INK)
body += line(cx3 + 70, my + 20, R - 30, my + 20, GREY, 6) + line(cx3 + 70, my + 38, R - 70, my + 38, GREY, 6)
write(body)
