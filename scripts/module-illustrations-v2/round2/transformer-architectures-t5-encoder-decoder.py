"""T5's span corruption and the cross-attention it trains, from the lesson's worked example.
Top: the 11-token sentence "Thank you for inviting me to your party last week ." with the hidden spans
(words 3-4 and 9) in teal. Below it: the 10-token encoder input, each hidden span collapsed into one terracotta
sentinel. Under that: the 6 x 10 cross-attention weights, one row per target token (sentinels in terracotta,
hidden words in teal) and one column per input token, each row a true softmax of scores peaked at the input
position the target token came from."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

hidden = {2, 3, 8}                         # "for inviting" and "last" (0-based)
# input tokens: kept words, with each hidden span replaced by one sentinel
inp, src = [], []                          # src: which original indices feed each input token
i = 0
while i < 11:
    if i in hidden:
        j = i
        while j in hidden:
            j += 1
        inp.append("s"); src.append(list(range(i, j))); i = j
    else:
        inp.append("w"); src.append([i]); i += 1
# target tokens: X0 for inviting X1 last X2, with the input column each one is about
tgt = [("s", 2), ("h", 2), ("h", 2.5), ("s", 7), ("h", 7.4), ("s", 9)]

C = 36                                     # cell size
GX = 147                                   # left edge of the attention grid / input row
OW = 34                                    # spacing of the original sentence
OX = GX + 5 * C - 5 * OW                   # centre the 11-token sentence over the 10 columns
Y0, Y1, YG = 48, 116, 152                  # original row, input row, grid top
B = 28                                     # token square size

body = ""
ox = lambda k: OX + k * OW
ix = lambda k: GX + k * C + C / 2
# links from the original sentence down to the input row
for k, s in enumerate(src):
    for o in s:
        col = ACCENT if inp[k] == "s" else GUIDE
        body += line(ox(o), Y0 + B / 2, ix(k), Y1 - B / 2, col, 2.0 if inp[k] == "s" else THIN)
for o in range(11):
    if o in hidden:
        body += rect(ox(o) - B / 2, Y0 - B / 2, B, B, SECOND, SECOND, 2.0, 4)
    else:
        body += rect(ox(o) - B / 2, Y0 - B / 2, B, B, BG, INK, 2.0, 4)
for k, t in enumerate(inp):
    if t == "s":
        body += rect(ix(k) - B / 2, Y1 - B / 2, B, B, ACCENT, ACCENT, 2.0, 4)
    else:
        body += rect(ix(k) - B / 2, Y1 - B / 2, B, B, BG, INK, 2.0, 4)

# cross-attention weights: softmax along each row
for r, (kind, c) in enumerate(tgt):
    sc = [3.2 * math.exp(-((j - c) / 1.1) ** 2) for j in range(10)]
    z = sum(math.exp(s) for s in sc)
    w = [math.exp(s) / z for s in sc]
    y = YG + r * C
    for j in range(10):
        a = min(1.0, w[j] / 0.4)
        body += (f'<rect x="{GX + j * C + 1.5:.1f}" y="{y + 1.5:.1f}" width="{C - 3}" height="{C - 3}" rx="3" '
                 f'fill="{ACCENT}" fill-opacity="{0.06 + 0.94 * a:.3f}"/>')
    tx = GX - 40
    if kind == "s":
        body += rect(tx - B / 2, y + C / 2 - B / 2, B, B, ACCENT, ACCENT, 2.0, 4)
    else:
        body += rect(tx - B / 2, y + C / 2 - B / 2, B, B, SECOND, SECOND, 2.0, 4)
body += rect(GX, YG, 10 * C, 6 * C, "none", GUIDE, THIN, 4)

write(body)
