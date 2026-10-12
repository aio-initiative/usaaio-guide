"""Bayes on the flagged-fraud example: 1% fraud, 90% of fraud flagged, 3% of legitimate flagged. The tree's branch
widths follow the probabilities, filled leaves are flagged, and the two flagged groups pour into one bar drawn to
scale: 90 fraud on top of 297 legitimate, so the fraud share of the flagged bar is about 0.233."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

pf, sens, fpr = 0.01, 0.90, 0.03
w = lambda p: 2.0 + 7.0 * p        # branch width from its probability

root = (62, 200)
fraud, legit = (200, 108), (200, 292)
ff, fn = (340, 66), (340, 150)     # fraud: flagged, not flagged
lf, ln = (340, 250), (340, 334)    # legitimate: flagged, not flagged

body = ""
body += line(*root, *fraud, ACCENT, w(pf)) + line(*root, *legit, INK, w(1 - pf))
body += line(*fraud, *ff, ACCENT, w(sens)) + line(*fraud, *fn, ACCENT, w(1 - sens))
body += line(*legit, *lf, INK, w(fpr)) + line(*legit, *ln, INK, w(1 - fpr))

# the flagged bar, drawn to scale
flag_f, flag_l = pf * sens, (1 - pf) * fpr
share = flag_f / (flag_f + flag_l)
bx, by, bw, bh = 470, 52, 66, 296
top_h = bh * share
body += rect(bx, by, bw, top_h, fill=ACCENT, stroke="none", rx=0)
body += rect(bx, by + top_h, bw, bh - top_h, fill=GUIDE, stroke="none", rx=0)
body += rect(bx, by, bw, bh, stroke=INK, sw=LINE, rx=3)
body += line(bx, by + top_h, bx + bw, by + top_h, INK, 1.6)
body += arrow(ff[0] + 16, ff[1] + 2, bx - 12, by + top_h / 2, ACCENT, 2.4)
body += arrow(lf[0] + 16, lf[1] - 2, bx - 12, by + top_h + (bh - top_h) / 2, INK, 2.4)

body += dot(*root, 10, INK)
body += dot(*fraud, 10, ACCENT) + dot(*legit, 10, INK)
body += dot(*ff, 10, ACCENT) + ring(*fn, 9, ACCENT, 2.6)
body += dot(*lf, 10, INK) + ring(*ln, 9, INK, 2.6)

write(body)
