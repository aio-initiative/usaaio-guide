"""The two-stage recipe. Left, pretraining: a large unlabeled corpus with about 15% of tokens masked feeds a
backbone whose every layer is trained (soft terracotta) to reconstruct the masked tokens above it. Right,
adaptation: the same backbone copied over and frozen (grey), with only a small new head trained on a few
labeled examples, each with its class colour."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(11)
NTOK, TS, TG = 8, 20, 4                  # tokens per row, tile size, gap
ROWW = NTOK * TS + (NTOK - 1) * TG       # 188
LX, RX = 158, 446                        # centres of the two backbones
BW, LAY, LH, LG, LTOP = 200, 5, 22, 9, 102


def layers(cx, fill, stroke):
    out = ""
    for k in range(LAY):
        out += rect(cx - BW / 2, LTOP + k * (LH + LG), BW, LH, fill, stroke, 2.2, 5)
    return out


def row(cx, y, masked=(), faint=False):
    out, x0 = "", cx - ROWW / 2
    for j in range(NTOK):
        x = x0 + j * (TS + TG)
        if j in masked:
            out += rect(x, y, TS, TS, ACCENT, ACCENT, 1.6, 3)
        else:
            out += rect(x, y, TS, TS, BG, GUIDE if faint else GREY, 1.6, 3)
    return out


LBOT = LTOP + LAY * LH + (LAY - 1) * LG  # bottom of the stack
ROW0 = LBOT + 20
# Pretraining corpus: many unlabeled rows, each with ~15% of positions masked (at least one per row).
masks = []
for r in range(4):
    m = {j for j in range(NTOK) if random.random() < 0.15}
    masks.append(m or {random.randrange(NTOK)})
body = ""
for r in range(4):
    body += row(LX, ROW0 + 25 * r, masks[r])
body += arrow(LX, ROW0 - 3, LX, LBOT + 4, GREY, 2.2)
body += layers(LX, ACCENT_SOFT, INK)
# Reconstructions appear above the masked columns only.
x0 = LX - ROWW / 2
for j in sorted(masks[0]):
    cx = x0 + j * (TS + TG) + TS / 2
    body += arrow(cx, LTOP - 4, cx, 72, ACCENT, 2.2)
    body += rect(cx - TS / 2, 46, TS, TS, ACCENT_SOFT, ACCENT, 2.2, 3)

# Copy the weights across.
mid = LTOP + (LAY * LH + (LAY - 1) * LG) / 2
body += arrow(LX + BW / 2 + 10, mid, RX - BW / 2 - 10, mid, GREY, 2.6)

# Adaptation: frozen backbone, small trainable head on top, a few labeled rows below.
body += layers(RX, GUIDE, GREY)
body += arrow(RX, LTOP - 4, RX, 84, GREY, 2.0)
body += rect(RX - 40, 46, 80, 36, ACCENT_SOFT, ACCENT, 2.4, 6)
body += dot(RX - 14, 64, 7, ACCENT) + dot(RX + 14, 64, 7, SECOND)
body += arrow(RX, ROW0 - 3, RX, LBOT + 4, GREY, 2.2)
for r, cls in enumerate((ACCENT, SECOND)):
    y = ROW0 + 25 * r
    body += row(RX - 14, y)
    body += dot(RX - 14 + ROWW / 2 + 18, y + TS / 2, 7, cls)
write(body)
