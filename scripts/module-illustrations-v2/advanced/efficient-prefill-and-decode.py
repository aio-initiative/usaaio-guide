"""Prefill and decode. Left: the causal table of scores for a 7-token text (lower triangle only). The first four
rows, the prompt, are teal: the prefill computes them in one run and they never change. Row five, the first decode
token, is the one new terracotta row; rows six and seven are dashed, still to come. Right: scores computed per run,
to scale. The teal prefill bar (the 4 x 4 table) is followed by three decode runs: the dashed grey bars are a run from
scratch (n^2 = 25, 36, 49) and the terracotta bars inside them are the new row only (n = 5, 6, 7)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, P, C = 7, 4, 38                       # tokens, prompt length, cell size
gx, gy = 44, (H - N * C) / 2
body = ""
for r in range(N):
    for c in range(r + 1):
        x, y = gx + c * C + 2, gy + r * C + 2
        if r < P:
            body += rect(x, y, C - 4, C - 4, fill=SECOND_SOFT, stroke=SECOND, sw=1.8, rx=3)
        elif r == P:
            body += rect(x, y, C - 4, C - 4, fill=ACCENT, stroke=ACCENT, sw=1.8, rx=3)
        else:
            body += f'<rect x="{x:.1f}" y="{y:.1f}" width="{C-4}" height="{C-4}" rx="3" fill="none" stroke="{GREY}" stroke-width="1.6" stroke-dasharray="4 4"/>'
# outline of the decode row, and the hidden upper triangle as a faint guide
body += rect(gx - 3, gy + P * C - 3, (P + 1) * C + 6, C + 6, stroke=INK, sw=2.6, rx=6)
body += poly([(gx, gy), (gx + N * C, gy), (gx + N * C, gy + N * C)], GUIDE, THIN, dash="5 5")

# scores computed per run, to scale
ax0, ax1, ay0, ay1 = 360, 556, gy, gy + N * C
s = (ay1 - ay0 - 6) / 49
body += axes(ax0, ay0, ax1, ay1)
bw, gap = 34, 14
x = ax0 + 12
body += rect(x, ay1 - P * P * s, bw, P * P * s, fill=SECOND_SOFT, stroke=SECOND, sw=2.2, rx=2)
for n in range(P + 1, N + 1):
    x += bw + gap
    body += f'<rect x="{x:.1f}" y="{ay1 - n*n*s:.1f}" width="{bw}" height="{n*n*s:.1f}" rx="2" fill="none" stroke="{GREY}" stroke-width="1.8" stroke-dasharray="6 5"/>'
    body += rect(x, ay1 - n * s, bw, n * s, fill=ACCENT, stroke=ACCENT, sw=2, rx=2)
write(body)
