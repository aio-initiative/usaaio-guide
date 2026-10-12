"""RAG: the question vector picks its two nearest notes by cosine angle, and those two chunks are pasted into the prompt.

Vectors are the lesson's handbook notes in the (library, closing) plane: d1 = (1, 0), d2 = (0.8, 0.6), d3 = (0, 1) (d4, the food note, points out of this plane and is left out)
and the question q = (0.6, 0.8). The top two cosines are d2 and d3.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

ox, oy, R = 62, 352, 262
X = lambda v: (ox + R * v[0], oy - R * v[1])
notes = [(1.0, 0.0), (0.8, 0.6), (0.0, 1.0)]
q = (0.6, 0.8)
cos = [(n[0] * q[0] + n[1] * q[1]) / math.hypot(*n) for n in notes]
top = sorted(range(3), key=lambda i: -cos[i])[:2]

def bez(p0, p1, p2, n=40):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (i / n for i in range(n + 1))]

body = axes(ox, oy - R - 18, ox + R + 18, oy)
arc = [X((math.cos(a), math.sin(a))) for a in (math.pi / 2 * i / 80 for i in range(81))]
body += poly(arc, GUIDE, 1.6, dash="5 6")
a0, a1 = sorted(math.atan2(notes[i][1], notes[i][0]) for i in top)   # cone spanned by the retrieved pair
wedge = [X((0, 0))] + [X((0.93 * math.cos(a), 0.93 * math.sin(a))) for a in (a0 + (a1 - a0) * i / 60 for i in range(61))]
body += poly(wedge, "none", 0, ACCENT_SOFT, close=True)
for i, n in enumerate(notes):
    c = ACCENT if i in top else GREY
    body += arrow(*X((0, 0)), *X(n), c, 3.0 if i in top else 2.2)
body += arrow(*X((0, 0)), *X(q), INK, 4.0)
body += dot(*X((0, 0)), 5.5, INK)

# prompt card: two retrieved chunks, then the question
cx, cy, cw, ch = 392, 58, 170, 284
body += rect(cx, cy, cw, ch, BG, INK, 2.2, 8)
slots = [(cx + 16, cy + 18 + k * 78, cw - 32, 62) for k in range(2)]
for (sx, sy, sw, sh) in slots:
    body += rect(sx, sy, sw, sh, ACCENT_SOFT, ACCENT, 2.0, 5)
    for j, f in enumerate((0.78, 0.6)):
        body += line(sx + 12, sy + 21 + 20 * j, sx + 12 + (sw - 24) * f, sy + 21 + 20 * j, ACCENT, 3.2)
qy = cy + 18 + 2 * 78 + 10
body += rect(cx + 16, qy, cw - 32, 50, BG, INK, 2.0, 5)
body += line(cx + 28, qy + 25, cx + 28 + (cw - 56) * 0.7, qy + 25, INK, 3.2)
for k, i in enumerate(sorted(top, key=lambda i: -notes[i][1])):     # higher note feeds the upper slot
    tip = X(notes[i]); tip = (tip[0] + 12, tip[1])
    sx, sy, sw, sh = slots[k]
    end = (sx - 6, sy + sh / 2)
    pts = bez(tip, ((tip[0] + end[0]) / 2, min(tip[1], end[1]) - 40), end)
    body += poly(pts[:-3], ACCENT, 2.0, dash="7 6") + arrow(*pts[-4], *end, ACCENT, 2.0)
write(body)
