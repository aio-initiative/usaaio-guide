"""Why shortcuts let networks go deep: a plain stack of ten blocks, where a gradient scaled by 0.8 at every block
shrinks to 0.8^10 by the first one, above the same stack with a shortcut around each block (output = x + F(x)),
where the gradient keeps a factor-1 path and reaches every block at full size."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, FACTOR = 10, 0.8
X0, X1 = 44, 540
STEP = (X1 - X0) / N
BW, BH, BARMAX = 22, 34, 60
body = ""


def row(cy, base, shortcut):
    out = ""
    out += line(X0, cy, X1, cy, INK, 2.2)
    out += dot(X0, cy, 6)
    for k in range(N):
        bx = X0 + STEP * (k + 0.5)
        if shortcut:
            # Shortcut: the block's input jumps over it and is added back at the plus node.
            xa, xb = bx - STEP * 0.5, bx + STEP * 0.5
            pts = [(xa + (xb - xa) * t / 20, cy - 4 * 32 * (t / 20) * (1 - t / 20)) for t in range(21)]
            out += poly(pts, ACCENT, 2.2)
        out += rect(bx - BW / 2, cy - BH / 2, BW, BH, BG, INK, 2.2, 3)
        if shortcut:
            px = bx + STEP * 0.5
            out += dot(px, cy, 6.5, BG, ACCENT, 2.0)
            out += line(px - 3.5, cy, px + 3.5, cy, ACCENT, 1.6) + line(px, cy - 3.5, px, cy + 3.5, ACCENT, 1.6)
        # Gradient reaching this block from the loss at the right end.
        g = 1.0 if shortcut else FACTOR ** (N - k)
        h = BARMAX * g
        col = ACCENT if shortcut else GREY
        out += rect(bx - 8, base - h, 16, h, col, "none", 0, 0)
    out += arrow(X1 + 7, cy, X1 + 22, cy, INK, 2.2)
    out += line(X0 + 4, base, X1 + 10, base, GUIDE, THIN)
    out += line(X0 + 4, base - BARMAX, X1 + 10, base - BARMAX, GUIDE, THIN, "5 5")
    return out


body += row(70, 172, False)
body += row(260, 366, True)
write(body)
