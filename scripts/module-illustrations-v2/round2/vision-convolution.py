"""The module's running example: the 6 x 6 image with one bright column, the 3 x 3 window outlined where the
column runs through its middle, the vertical filter (terracotta middle column of 2s, teal sides of -1s), and the
4 x 4 feature map actually computed by sliding it (-3, 6, -3, 0 in every row, terracotta for positive, teal for
negative, shade by size). Guide lines project the window onto the one output cell it produces."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N, K = 6, 3
img = [[1.0 if j == 2 else 0.0 for j in range(N)] for i in range(N)]
filt = [[-1, 2, -1] for _ in range(K)]
M = N - K + 1
fmap = [[sum(filt[u][v] * img[i + u][j + v] for u in range(K) for v in range(K)) for j in range(M)] for i in range(M)]
vmax = max(abs(v) for r in fmap for v in r)
wi, wj = 0, 1                                   # the window position the lesson works through (value 6)

def shade(v, top):
    if v == 0: return BG, 0
    return (ACCENT if v > 0 else SECOND), abs(v) / top

def grid(x0, y0, c, vals, top, ink_pixels=False):
    out = ""
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            x, y = x0 + j * c, y0 + i * c
            if ink_pixels:
                out += rect(x, y, c, c, INK if v else BG, GUIDE, THIN, 0)
            else:
                col, a = shade(v, top)
                out += rect(x, y, c, c, BG, GUIDE, THIN, 0)
                if a: out += f'<rect x="{x+2:.1f}" y="{y+2:.1f}" width="{c-4:.1f}" height="{c-4:.1f}" fill="{col}" fill-opacity="{a:.2f}"/>'
    n = len(vals)
    return out + rect(x0, y0, n * c, n * c, "none", INK, 2, 0)

C = 40                                   # image cell
IX, IY = 40, (H - N * C) / 2
FC = 42                                  # feature-map cell
FX, FY = W - 40 - M * FC, (H - M * FC) / 2
KC = 26                                  # filter cell, below the projection band
KX, KY = (IX + N * C + FX) / 2 - K * KC / 2, H - 40 - K * KC

# projection from the window's right corners to the output cell's left corners
wx, wy, ws = IX + wj * C, IY + wi * C, K * C
ox, oy = FX + wj * FC, FY + wi * FC
b = f'<path d="M{wx+ws:.1f},{wy:.1f} L{ox:.1f},{oy:.1f} L{ox:.1f},{oy+FC:.1f} L{wx+ws:.1f},{wy+ws:.1f} Z" fill="{ACCENT_SOFT}" fill-opacity="0.55"/>'
b += grid(IX, IY, C, img, 1, ink_pixels=True) + grid(FX, FY, FC, fmap, vmax) + grid(KX, KY, KC, filt, 2)
b += line(wx + ws, wy, ox, oy, ACCENT, 1.6, "6 5") + line(wx + ws, wy + ws, ox, oy + FC, ACCENT, 1.6, "6 5")
b += f'<rect x="{wx:.1f}" y="{wy:.1f}" width="{ws:.1f}" height="{ws:.1f}" fill="none" stroke="{ACCENT}" stroke-width="4" rx="2"/>'
b += f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{FC:.1f}" height="{FC:.1f}" fill="none" stroke="{ACCENT}" stroke-width="4" rx="2"/>'
# the filter feeds the step from window to output cell
mx = KX + K * KC / 2
band = wy + ws + (oy + FC - wy - ws) * (mx - wx - ws) / (ox - wx - ws)
b += arrow(mx, KY - 8, mx, band + 12, GREY, 2.2)
write(b)
