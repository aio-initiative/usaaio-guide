"""Classifier-free guidance on the lesson's running example, in the plane of the first two pixel values, turned 45
degrees so the guidance line runs left to right. guided = unprompted + w (prompted - unprompted): the unprompted
estimate (0.28, 0.72) is a grey blend inside the valid square [0,1]^2 (the diamond), the prompted "left" estimate
(1, 0) is the LEFT image on its corner, and w = 2, 3 keep going in equal steps out of the square (w = 7.5 runs off the
card). The sampler clips w = 3 back onto the corner. 2x2 tiles show the two estimates as images."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

S = 128                                    # px per pixel-value unit
r2 = math.sqrt(2)
ox, oy = 148, 326                          # screen position of pixel values (0, 0), the diamond's bottom corner


def P(a, b):                               # pixel values (a, b) -> screen, rotated so (1, -1) points right
    return ox + (a - b) / r2 * S, oy - (a + b) / r2 * S


un = (0.28, 0.72)
pr = (1.0, 0.0)
at = lambda w: (un[0] + w * (pr[0] - un[0]), un[1] + w * (pr[1] - un[1]))


def tile(x, y, vals, size=30, stroke=INK):
    """2x2 image, row-major pixels; value 1 is ink, 0 is white."""
    out = ""
    for k, v in enumerate(vals):
        g = int(255 - max(0.0, min(1.0, v)) * (255 - 0x1f))
        out += rect(x + (k % 2) * size, y + (k // 2) * size, size, size, f"rgb({g},{g},{g})", GUIDE, 1.2, rx=0)
    return out + rect(x, y, 2 * size, 2 * size, "none", stroke, 2.2, rx=2)


body = ""
# the valid square of pixel values
body += poly([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], ACCENT, 2.4, fill=ACCENT_SOFT, close=True)
# the guidance line through unprompted and prompted, on in equal steps of w past the square
g3 = at(3)
yl = P(*un)[1]
body += line(P(*g3)[0], yl, W - 40, yl, GREY, 2, dash="6 7")
body += line(P(*un)[0], yl, P(*g3)[0], yl, INK, 3.2)
for w in (0, 1, 2, 3):
    x = P(*at(w))[0]
    body += line(x, yl - 12, x, yl + 12, INK if w < 2 else ACCENT, 2)
# clipping w = 3 back onto the valid corner
xc, x3 = P(*pr)[0], P(*g3)[0]
body += f'<path d="M{x3 - 6:.1f},{yl + 16:.1f} C{x3 - 30:.1f},{yl + 92:.1f} {xc + 34:.1f},{yl + 92:.1f} {xc + 9:.1f},{yl + 22:.1f}" fill="none" stroke="{GREY}" stroke-width="2" stroke-dasharray="5 5"/>'
body += arrow(xc + 13, yl + 32, xc + 8, yl + 15, GREY, 2)
body += dot(P(*at(2))[0], yl, 8, ACCENT)
body += dot(x3, yl, 10, ACCENT, BG, 2.5)
body += ring(*P(*un), 8, GREY, 2.8)
body += dot(xc, yl, 9, INK, BG, 2.5)
# the two estimates as images, above their points
xu = P(*un)[0]
body += line(xu, yl - 14, xu, 120, GUIDE, THIN, dash="4 4")
body += tile(xu - 30, 58, [un[0], un[1], un[0], un[1]], stroke=GREY)
body += line(xc, yl - 14, xc, 120, GUIDE, THIN, dash="4 4")
body += tile(xc - 30, 58, [pr[0], pr[1], pr[0], pr[1]])
write(body)
