"""Reward hacking, from the lesson's worked example (beta = 0, no KL penalty). Horizontal axis: log(1 + KL) from the
SFT reference. The reward-model score (terracotta) climbs without end while the human rating (teal) peaks early and
collapses. The shaded band is where a KL controller holds the policy (KL near 6), and the teal ring is the
penalised run's human rating there, above anything the unpenalised run reaches."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

kl = [0.0, 3.1, 18.4, 91.0, 340.0]
rm = [0.0, 1.8, 4.2, 9.7, 18.3]
human = [6.2, 6.9, 6.4, 3.1, 1.2]
d = [math.log1p(k) for k in kl]

x0, x1, y0, y1 = 52, W - 44, 40, H - 46
D1 = 6.1


def px(v):
    return x0 + v / D1 * (x1 - x0)


def py_rm(v):
    return y1 - v / 19.5 * (y1 - y0)


def py_h(v):
    return y1 - v / 8.4 * (y1 - y0)


def monotone(xs, ys, n=60):
    """Fritsch-Carlson monotone cubic through the data points."""
    m = len(xs)
    dl = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(m - 1)]
    t = [dl[0]] + [0 if dl[i - 1] * dl[i] <= 0 else 2 / (1 / dl[i - 1] + 1 / dl[i]) for i in range(1, m - 1)] + [dl[-1]]
    out = []
    for i in range(m - 1):
        h = xs[i + 1] - xs[i]
        for j in range(n + (i == m - 2)):
            s = j / n
            h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
            out.append((xs[i] + s * h, h00*ys[i] + h10*h*t[i] + h01*ys[i+1] + h11*h*t[i+1]))
    return out


kt = math.log1p(6.0)            # KL target held by the adaptive controller
body = rect(x0, y0, px(kt) - x0, y1 - y0, fill=ACCENT_SOFT, stroke="none", rx=0)
body += axes(x0, y0, x1, y1)
body += line(px(kt), y0, px(kt), y1, GREY, 2, dash="6 6")
body += poly([(px(a), py_h(b)) for a, b in monotone(d, human)], SECOND, 3.4)
body += poly([(px(a), py_rm(b)) for a, b in monotone(d, rm)], ACCENT, 3.4)
for a, r, h in zip(d, rm, human):
    body += dot(px(a), py_rm(r), 7, ACCENT)
    body += dot(px(a), py_h(h), 7, SECOND)
body += ring(px(kt), py_h(7.8), 10, SECOND, 3.6)
write(body)
