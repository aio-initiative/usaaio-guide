"""The next-token task and why it is hard: "The students who failed the exam ___". The blank's prediction (a softmax
over a small vocabulary, one clear winner) depends on the plural subject five tokens back, not on the nearest noun."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

words = ["the", "students", "who", "failed", "the", "exam"]
UNIT, GAP, BLANK = 10.5, 11, 64
widths = [len(w) * UNIT + 14 for w in words]
total = sum(widths) + BLANK + GAP * len(words)
x = (W - total) / 2
TOP, TH = 196, 44
tiles = []
for w in widths + [BLANK]:
    tiles.append((x, w))
    x += w + GAP
SUBJ, NOUN, Q = 1, 5, 6

body = ""
# Arcs below the row, from the blank back to the tokens: solid accent to the far subject, dashed grey to the nearest noun.
def arc(i, j, depth, color, sw, dash=None):
    (xi, wi), (xj, wj) = tiles[i], tiles[j]
    a, b = (xi + wi / 2, TOP + TH + 8), (xj + wj / 2, TOP + TH + 8)
    pts = []
    for k in range(41):
        t = k / 40
        pts.append((a[0] + (b[0] - a[0]) * t, a[1] + depth * 4 * t * (1 - t)))
    out = poly(pts[:-2], color, sw, dash=dash)
    out += arrow(*pts[-6], *b, color, sw)
    return out

body += arc(Q, NOUN, 40, GREY, 2.2, dash="7 6")
body += arc(Q, SUBJ, 96, ACCENT, 4.2)

# The token row. Plural vs singular shown by two dots or one dot inside the nouns.
for k, (tx, tw) in enumerate(tiles):
    if k == Q:
        body += rect(tx, TOP, tw, TH, fill=ACCENT_SOFT, stroke=ACCENT, sw=2.6, rx=7).replace("/>", ' stroke-dasharray="7 5"/>')
        cx = tx + tw / 2
        body += dot(cx - 9, TOP + TH / 2, 5, ACCENT) + dot(cx + 9, TOP + TH / 2, 5, ACCENT)
        continue
    stroke, fill = (ACCENT, BG) if k == SUBJ else (INK, BG)
    body += rect(tx, TOP, tw, TH, fill=fill, stroke=stroke, sw=2.4, rx=7)
    cy = TOP + TH / 2
    if k == SUBJ:
        cx = tx + tw / 2
        body += dot(cx - 9, cy, 5, ACCENT) + dot(cx + 9, cy, 5, ACCENT)
    elif k == NOUN:
        body += dot(tx + tw / 2, cy, 5, GREY)
    else:   # plain word: a grey stroke standing in for its letters
        body += line(tx + 12, cy, tx + tw - 12, cy, GUIDE, 5)

# Above the blank: the model's next-token distribution, a true softmax of fixed logits.
logits = [0.4, 2.7, 1.1, 1.8, 0.2, 0.9, -0.2, 0.6]
WIN = 1
z = [math.exp(v) for v in logits]
p = [v / sum(z) for v in z]
qx, qw = tiles[Q]
CW, BW = 28, 20
cw = len(p) * CW
x0 = min(qx + qw / 2 - cw / 2, W - 40 - cw)
BASE, HMAX = 150, 112
body += line(x0 - 8, BASE, x0 + cw, BASE, GUIDE, THIN)
for k, v in enumerate(p):
    bx = x0 + k * CW + (CW - BW) / 2
    h = HMAX * v / max(p)
    col = ACCENT if k == WIN else "#c4c9cf"
    body += f'<rect x="{bx:.1f}" y="{BASE - h:.1f}" width="{BW}" height="{h:.1f}" rx="2" fill="{col}"/>'
body += arrow(qx + qw / 2, TOP - 8, qx + qw / 2, BASE + 10, GREY, 2.2)

write(body)
