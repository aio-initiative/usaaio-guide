"""Sinks plus a window (StreamingLLM), and what evicting the sinks does.

Left: the causal attention mask over 16 tokens with 4 sink tokens (terracotta columns) and a sliding window of 5
(ink band); every other past entry (pale outline) has been evicted from the cache. The last row, framed, is the
newest query. Right top: that query's real softmax weights over the kept slots, most of the mass parked on the
sinks. Right bottom: the same query after the sinks are evicted: softmax renormalises over the window alone, so the
sinks' mass lands proportionally on the content tokens (the terracotta extra on top of each grey bar), the spurious
mass that breaks a head that wanted to do nothing.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, SINKS, WIN = 16, 4, 5
SINK_LOGITS = [3.2, 1.8, 1.4, 1.2]
WIN_LOGITS = [1.0, 0.4, 1.3, 0.7, 1.1]
es, ew = [math.exp(x) for x in SINK_LOGITS], [math.exp(x) for x in WIN_LOGITS]
Z = sum(es) + sum(ew)
w_sink = [x / Z for x in es]
w_win = [x / Z for x in ew]
w_win_only = [x / sum(ew) for x in ew]

body = ""
# The mask.
C = 19
MX, MY = 40, (H - N * C) / 2
for i in range(N):
    for j in range(i + 1):
        x, y = MX + j * C, MY + i * C
        if j < SINKS:
            body += rect(x + 1, y + 1, C - 2, C - 2, ACCENT, "none", 0, 2)
        elif j > i - WIN:
            body += rect(x + 1, y + 1, C - 2, C - 2, INK, "none", 0, 2)
        else:
            body += rect(x + 2, y + 2, C - 4, C - 4, BG, GUIDE, 1.3, 2)
body += rect(MX - 4, MY + (N - 1) * C - 4, N * C + 8, C + 8, "none", ACCENT, 2.4, 5)

# Bars of the newest query.
PX0, PX1 = 374, 562
slots = list(range(SINKS)) + [None] + list(range(WIN))
SW = (PX1 - PX0) / len(slots)
BW = SW * 0.74
SCALE = 250                         # px per unit of attention weight
TOP_BASE, BOT_BASE = 170, 352


def xs(k):
    return PX0 + SW * k + (SW - BW) / 2


# Top panel: sinks kept.
body += line(PX0 - 6, TOP_BASE, PX1 + 2, TOP_BASE, GUIDE, THIN)
for k, s in enumerate(slots):
    if s is None:
        continue
    sink = k < SINKS
    h = (w_sink[s] if sink else w_win[s]) * SCALE
    body += rect(xs(k), TOP_BASE - h, BW, h, ACCENT if sink else INK, "none", 0, 2)

# Bottom panel: sinks evicted, their mass redistributed over the window.
body += line(PX0 - 6, BOT_BASE, PX1 + 2, BOT_BASE, GUIDE, THIN)
for k, s in enumerate(slots):
    if s is None:
        continue
    if k < SINKS:
        body += rect(xs(k) + 1, BOT_BASE - 22, BW - 2, 22, "none", GREY, 2.0, 2).replace("/>", ' stroke-dasharray="4 3"/>')
        continue
    h0, h1 = w_win[s] * SCALE, w_win_only[s] * SCALE
    body += rect(xs(k), BOT_BASE - h1, BW, h1 - h0, ACCENT_SOFT, ACCENT, 2.0, 2)
    body += rect(xs(k), BOT_BASE - h0, BW, h0, INK, "none", 0, 2)
# The sink mass pours over into the window.
ax0, ax1 = xs(1) + BW / 2, xs(SINKS + 2) + BW / 2
body += f'<path d="M{ax0:.1f},{BOT_BASE - 28:.1f} C{ax0:.1f},{BOT_BASE - 120:.1f} {ax1 - 40:.1f},{BOT_BASE - 140:.1f} {ax1 - 6:.1f},{BOT_BASE - 120:.1f}" fill="none" stroke="{ACCENT}" stroke-width="2.2" stroke-dasharray="6 5" stroke-linecap="round"/>'
body += arrow(ax1 - 40, BOT_BASE - 134, ax1 - 4, BOT_BASE - 118, ACCENT, 2.2, 12).split("/>", 1)[1]
write(body)
