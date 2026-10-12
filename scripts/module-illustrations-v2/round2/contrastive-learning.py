"""CLIP's shared space and its similarity matrix, from the lesson's objective.
Left: five images (teal squares) and their five captions (terracotta dots) as unit vectors on one circle, each
caption close to its own image. For the first image, its positive (drawn a little farther off) is joined by a solid terracotta line and the
negatives by dashed grey lines; arc arrows show the positive pulled in and the nearest negatives pushed away. Right: the 5 x 5 matrix of softmax weights along each row of
cos(image_i, caption_j) / tau with tau = 0.15, computed from those same angles; the correct pairs are the diagonal,
and the first image's row is outlined."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

base = [112, 182, 248, 318, 38]                      # image directions, degrees
off = [-24, -12, 9, -10, 8]                          # each caption sits near its own image
img = [math.radians(a) for a in base]
cap = [math.radians(a + o) for a, o in zip(base, off)]
tau = 0.15
N = 5

CX, CY, R = 172, 204, 122
P = lambda t, r=R: (CX + r * math.cos(t), CY - r * math.sin(t))

body = poly([P(2 * math.pi * k / 120) for k in range(121)], GUIDE, 2.0)
body += dot(CX, CY, 4, GREY)
q = P(img[0])
for j in range(1, N):
    body += line(*q, *P(cap[j]), GREY, 1.8, "6 6")
body += line(*q, *P(cap[0]), ACCENT, 3.2)


def arc_arrow(t0, t1, r, color, sw):
    pts = [P(t0 + (t1 - t0) * k / 24, r) for k in range(25)]
    return poly(pts[:-3], color, sw) + arrow(*pts[-4], *pts[-1], color, sw)


# InfoNCE pulls the positive caption toward the image and pushes the nearest negatives away
body += arc_arrow(cap[0] - 0.02, img[0] - 0.1, R + 22, ACCENT, 3.0)
body += arc_arrow(cap[4] + 0.04, cap[4] - 0.26, R + 22, GREY, 2.4)
body += arc_arrow(cap[1] - 0.04, cap[1] + 0.26, R + 22, GREY, 2.4)
for i in range(N):
    x, y = P(img[i])
    body += rect(x - 10, y - 10, 20, 20, SECOND, SECOND if i else INK, 2.4 if i else 3.0, 3)
    body += dot(*P(cap[i]), 9, ACCENT)

# row softmax of the similarity matrix
C = 42
MX, MY = 352, 112
for i in range(N):
    s = [math.cos(img[i] - cap[j]) / tau for j in range(N)]
    z = sum(math.exp(v) for v in s)
    for j in range(N):
        w = math.exp(s[j]) / z
        body += (f'<rect x="{MX + j * C + 2:.1f}" y="{MY + i * C + 2:.1f}" width="{C - 4}" height="{C - 4}" rx="4" '
                 f'fill="{ACCENT}" fill-opacity="{0.07 + 0.93 * w:.3f}"/>')
    body += rect(MX - 34, MY + i * C + C / 2 - 10, 20, 20, SECOND, SECOND, 2.4, 3)
for j in range(N):
    body += dot(MX + j * C + C / 2, MY - 24, 9, ACCENT)
body += rect(MX - 1, MY - 1, N * C + 2, C + 2, "none", INK, 2.4, 6)

write(body)
