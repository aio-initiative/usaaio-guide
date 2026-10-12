"""The running example's autoencoder: noisy LEFT (4 pixels) squeezes through a one-number code and comes back clean.

Below, the code axis: RIGHT at -1, the grey mean at 0, LEFT at +1, and noisy LEFT's code 0.9 marked in the accent.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

PATTERN = [1, -1, 1, -1]
encode = lambda x: sum(a * b for a, b in zip(x, PATTERN)) / 2
decode = lambda z: [0.5 + z * p / 2 for p in PATTERN]
noisy = [0.9, 0.1, 1.0, 0.0]
z = encode(noisy)                      # 0.9
out = decode(z)                        # 0.95, 0.05, 0.95, 0.05


def shade(v):
    v = max(0.0, min(1.0, v))
    g = int(255 - (255 - 31) * v)
    return f"#{g:02x}{g:02x}{g:02x}"


def grid(x, y, px, cell, frame=GREY, sw=1.6):
    out = ""
    for i, v in enumerate(px):
        out += rect(x + (i % 2) * cell, y + (i // 2) * cell, cell, cell, shade(v), GUIDE, 0.8, 0)
    return out + rect(x, y, 2 * cell, 2 * cell, "none", frame, sw, 2)


CY = 148
body = ""
# Hourglass silhouette behind the layers.
cols = [(190, 4), (250, 3), (300, 1), (350, 3), (410, 4)]
sp = 34
ys = lambda n: [CY + (k - (n - 1) / 2) * sp for k in range(n)]
top = [(x, CY - (n - 1) / 2 * sp - 22) for x, n in cols]
bot = [(x, CY + (n - 1) / 2 * sp + 22) for x, n in cols][::-1]
body += poly(top + bot, SECOND, 1.4, SECOND_SOFT, close=True)
for (x1, n1), (x2, n2) in zip(cols, cols[1:]):
    for y1 in ys(n1):
        for y2 in ys(n2):
            body += line(x1, y1, x2, y2, GREY, 1.0)
for i, (x, n) in enumerate(cols):
    for y in ys(n):
        body += dot(x, y, 13, ACCENT, BG, 3) if n == 1 else ring(x, y, 9, INK, 2.2)

# Input and output images.
cell = 46
body += grid(40, CY - cell, noisy, cell)
body += arrow(40 + 2 * cell + 8, CY, 170, CY, GREY, 2.2)
body += arrow(430, CY, 560 - 2 * cell - 8, CY, GREY, 2.2)
body += grid(560 - 2 * cell, CY - cell, out, cell, INK, 2.2)

# The code axis: one number per image.
AY, X0, S = 318, 300, 150
body += line(70, AY, 530, AY, GUIDE, 2.0)
for zt in (-1, 0, 1):
    body += line(X0 + zt * S, AY - 8, X0 + zt * S, AY + 8, GREY, 2.0)
    body += grid(X0 + zt * S - 18, AY + 16, decode(zt), 18, GREY, 1.4)
zx = X0 + z * S
body += poly([(300, CY + 16)] + [(300 + (zx - 300) * t * t, CY + 16 + (AY - 8 - CY - 16) * t) for t in [k / 20 for k in range(1, 21)]],
             ACCENT, 2.0, dash="6 6")
body += dot(zx, AY, 9, ACCENT)
write(body)
