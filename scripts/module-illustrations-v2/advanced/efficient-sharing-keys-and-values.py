"""Sharing keys and values: four heads under multi-head, grouped-query (2 groups) and multi-query attention.

Every head keeps its own query (a terracotta ring on top); below sit the stored key-value sets (a teal key half
and a grey value half). Heads are grouped in order, as np.repeat lines them up, so the panels store 4, 2 and 1
sets. Under each panel a bar shows the cache size, in proportion to the number of sets stored.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

HEADS = 4
SCHEMES = [4, 2, 1]                    # sets stored: multi-head, grouped-query, multi-query
PW, PX0 = 152, 40                       # panel width, left edge of the first panel
GAPX = (W - 2 * PX0 - 3 * PW) / 2
QY, KY = 62, 196                        # query row, top of the stored sets
SH = 90                                 # height of a stored set
BAR_Y, BAR_H = 322, 24

body = ""
for p, sets in enumerate(SCHEMES):
    x0 = PX0 + p * (PW + GAPX)
    qx = [x0 + PW * (i + 0.5) / HEADS for i in range(HEADS)]
    group = HEADS // sets
    sw = min(30, PW / sets - 8)
    sx = [x0 + PW * (j + 0.5) / sets for j in range(sets)]
    if p:
        body += line(x0 - GAPX / 2, 36, x0 - GAPX / 2, 364, GUIDE, THIN, "4 6")
    for i, x in enumerate(qx):
        j = i // group                  # heads grouped in order: set 1, set 1, set 2, set 2
        body += line(x, QY + 14, sx[j], KY - 2, INK, 2.4)
    for j, x in enumerate(sx):
        body += rect(x - sw / 2, KY, sw, SH / 2, SECOND_SOFT, SECOND, 2.4, 5)
        body += rect(x - sw / 2, KY + SH / 2, sw, SH / 2, "#eceef0", GREY, 2.4, 5)
    for x in qx:
        body += ring(x, QY, 13, ACCENT, 3)
        body += dot(x, QY, 5, ACCENT)
    # Cache size in proportion to the sets stored.
    body += rect(x0, BAR_Y, PW, BAR_H, BG, GUIDE, THIN, 4)
    body += rect(x0, BAR_Y, PW * sets / HEADS, BAR_H, ACCENT_SOFT, ACCENT, 2.2, 4)
write(body)
