"""Codes as ranges. Over the code axis z: LEFT's range (terracotta) and RIGHT's range (teal), the true posteriors
of the running example (mu = +-2/3, sigma = 0.577), overlapping around 0, with the dashed standard range N(0, 1).
Below the axis, the lesson's linear decoder 0.5 + z * (1, -1, 1, -1) / 2 turns codes into 2 x 2 images: RIGHT at
one end, LEFT at the other, gray blends in the middle (the blur). A code drawn from the standard range (ring) decodes to the
image framed in terracotta."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

X0, X1 = 50, 550                 # z from -2.2 to 2.2
ZMAX = 2.2
AY, TOP = 222, 40                # axis baseline and top of the tallest curve
sx = lambda z: X0 + (z + ZMAX) / (2 * ZMAX) * (X1 - X0)
gauss = lambda z, m, s: math.exp(-0.5 * ((z - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
pk = gauss(0, 0, 0.577)
sy = lambda p: AY - p / pk * (AY - TOP)

def curve(m, s, color, fill=None, dash=None):
    zs = [-ZMAX + i * 2 * ZMAX / 240 for i in range(241)]
    pts = [(sx(z), sy(gauss(z, m, s))) for z in zs]
    out = ""
    if fill:
        out += poly(pts + [(X1, AY), (X0, AY)], "none", 0, fill, close=True)
    return out + poly(pts, color, 3.0 if not dash else 2.4, dash=dash)

body = ""
body += f'<g opacity="0.55">{curve(-2/3, 0.577, SECOND, SECOND_SOFT)}</g>'
body += f'<g opacity="0.55">{curve(2/3, 0.577, ACCENT, ACCENT_SOFT)}</g>'
body += curve(-2/3, 0.577, SECOND) + curve(2/3, 0.577, ACCENT)
body += curve(0, 1, GREY, dash="7 6")
body += line(X0, AY, X1, AY, INK, 2.0)
for m, c in ((-2/3, SECOND), (2/3, ACCENT)):
    body += line(sx(m), AY, sx(m), sy(gauss(m, m, 0.577)), c, 1.6, "3 5")

# decoded images along the axis
C = 32                           # pixel size, images are 2C x 2C
IY = 314
zs = [-1.25, -0.625, 0.0, 0.625, 1.25]
draw = 0.625                       # a code drawn from the standard range
pattern = [1, -1, 1, -1]         # row-major: top-left, top-right, bottom-left, bottom-right

def shade(v):
    v = max(0.0, min(1.0, v))
    a, b = (255, 255, 255), (31, 35, 40)  # white -> INK
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * v) for i in range(3))

for z in zs:
    cx = sx(z)
    body += line(cx, AY + 8, cx, IY - C - 12, GUIDE, THIN)
    for k, p in enumerate(pattern):
        v = 0.5 + z * p / 2
        body += rect(cx - C + (k % 2) * C, IY - C + (k // 2) * C, C, C, shade(v), shade(v), 0.8, rx=0)
    edge = ACCENT if z == draw else INK
    body += rect(cx - C, IY - C, 2 * C, 2 * C, "none", edge, 3.4 if z == draw else 2.0, rx=2)
body += ring(sx(draw), AY, 9, ACCENT, 3.0)
body += arrow(sx(draw), AY + 14, sx(draw), IY - C - 8, ACCENT, 2.6)
write(body)
