"""FID on the lesson's toy example: real embeddings ~ N((0,0), I) (grey rings), generated ~ N((1,0), 0.5 I)
(terracotta dots). Each set gets its fitted Gaussian drawn as 1- and 2-sigma circles; the arrow between the means is
the ||mu_r - mu_g||^2 term and the smaller generated circle is the covariance term. On the right, FID = 1.17 as a
to-scale stacked bar: the mean term 1 (ink) and the trace term 0.1716 (terracotta)."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

S = 68                                   # px per unit, same in x and y so circles stay circles
cx, cy = 200, 200                        # screen position of the real mean (0, 0)
X = lambda x: cx + x * S
Y = lambda y: cy - y * S
rng = random.Random(5)


def sample(n, mx, sd):
    out = []
    while len(out) < n:
        x, y = rng.gauss(mx, sd), rng.gauss(0, sd)
        if (x - mx) ** 2 + y ** 2 < (2.1 * sd) ** 2:
            out.append((x, y))
    return out


real = sample(30, 0.0, 1.0)
gen = sample(20, 1.0, math.sqrt(0.5))

body = line(X(-2.4), cy, X(3.0), cy, GUIDE, THIN) + line(cx, Y(2.4), cx, Y(-2.4), GUIDE, THIN)
for k in (2, 1):
    body += f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="{k * S:.1f}" fill="none" stroke="{GREY}" stroke-width="2" stroke-dasharray="7 6"/>'
rg = math.sqrt(0.5) * S
body += f'<circle cx="{X(1):.1f}" cy="{Y(0):.1f}" r="{2 * rg:.1f}" fill="{ACCENT_SOFT}" fill-opacity="0.6" stroke="{ACCENT}" stroke-width="2.4"/>'
body += f'<circle cx="{X(1):.1f}" cy="{Y(0):.1f}" r="{rg:.1f}" fill="none" stroke="{ACCENT}" stroke-width="1.8" stroke-dasharray="6 5"/>'
for x, y in real:
    body += ring(X(x), Y(y), 5, GREY, 2.2)
for x, y in gen:
    body += dot(X(x), Y(y), 5, ACCENT, BG, 1.2)
# the mean shift
body += arrow(X(0), Y(0), X(1) - 11, Y(0), INK, 3.4)
body += dot(X(0), Y(0), 8, INK, BG, 2.5)
body += dot(X(1), Y(0), 8, ACCENT, BG, 2.5)

# FID as a stacked bar, to scale: mean term 1 + trace term 2 (1 + 0.5 - 2 sqrt(0.5))
mean_term = 1.0
trace_term = 2 * (1 + 0.5 - 2 * math.sqrt(0.5))
unit = 220
bx, bw, base = 470, 50, cy + (mean_term + trace_term) * unit / 2
body += line(bx - 30, base, bx + bw + 30, base, GUIDE, THIN)
h1, h2 = mean_term * unit, trace_term * unit
body += rect(bx, base - h1, bw, h1, INK, INK, 0, rx=0)
body += rect(bx, base - h1 - h2, bw, h2, ACCENT_SOFT, ACCENT, 2.4, rx=0)
write(body)
