"""U-Net, drawn as the maps themselves. Down the left side the encoder: a 16 x 16 greyscale image of a tilted blob,
average-pooled to 8 x 8, 4 x 4 and a 2 x 2 bottleneck, each grid at its true resolution. Up the right side the
decoder grows the map back, 4 x 4, 8 x 8 and finally a 16 x 16 per-pixel mask (terracotta pixels = the blob's
pixels, the coverage pooled to that size and thresholded at one half). Dashed terracotta skip connections carry
each encoder map straight across to the decoder map of the same size."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

N = 16
SS = 6                                    # supersampling per pixel for the blob's coverage
def inside(x, y):                         # the object: a tilted ellipse, x and y in pixel units
    x, y = x - 7.6, y - 8.4
    u, v = 0.8 * x + 0.6 * y, -0.6 * x + 0.8 * y
    return (u / 6.0) ** 2 + (v / 3.9) ** 2 <= 1
cov = [[sum(inside(j + (a + 0.5) / SS, i + (b + 0.5) / SS) for a in range(SS) for b in range(SS)) / SS ** 2
        for j in range(N)] for i in range(N)]
img = [[0.18 + 0.12 * j / N + 0.55 * cov[i][j] for j in range(N)] for i in range(N)]   # darkness 0..1

def pool(m, n):
    k = len(m) // n
    return [[sum(m[i * k + a][j * k + b] for a in range(k) for b in range(k)) / k ** 2 for j in range(n)] for i in range(n)]

def grey(t):
    g = round(250 - 190 * t)
    return f"#{g:02x}{g:02x}{min(255, g + 4):02x}"

def grid(x0, y0, side, vals, mask=False):
    n = len(vals)
    c = side / n
    o = ""
    for i in range(n):
        for j in range(n):
            if mask:
                f = ACCENT if vals[i][j] >= 0.5 else BG
            else:
                f = grey(vals[i][j])
            o += f'<rect x="{x0 + j * c:.2f}" y="{y0 + i * c:.2f}" width="{c + 0.4:.2f}" height="{c + 0.4:.2f}" fill="{f}"/>'
    lw = 0.8 if n >= 16 else 1.1
    for k in range(1, n):
        o += line(x0 + k * c, y0, x0 + k * c, y0 + side, "#ffffff" if not mask else GUIDE, lw)
        o += line(x0, y0 + k * c, x0 + side, y0 + k * c, "#ffffff" if not mask else GUIDE, lw)
    return o + rect(x0, y0, side, side, "none", INK, 2.6, 0)

# Levels: resolution 16, 8, 4 on each side and 2 at the bottom; drawn sides shrink with depth.
res = [16, 8, 4]
side = [116, 76, 50]
ys = [40, 186, 290]
xl = [40, 112, 166]                        # encoder left edges, stepping inward down the U
xr = [W - 40 - s - (x - 40) for s, x in zip(side, xl)]
bs, bres = 34, 2
bx, by = (W - bs) / 2, H - 40 - bs

body = ""
for k in range(3):
    enc = pool(img, res[k]) if res[k] < N else img
    body += grid(xl[k], ys[k], side[k], enc)
    body += grid(xr[k], ys[k], side[k], pool(cov, res[k]) if res[k] < N else cov, mask=True)
    yc = ys[k] + side[k] / 2
    body += line(xl[k] + side[k] + 10, yc, xr[k] - 22, yc, ACCENT, 2.6, "9 7")
    body += arrow(xr[k] - 22, yc, xr[k] - 8, yc, ACCENT, 2.6)
body += grid(bx, by, bs, pool(img, bres))

# Down arrows on the encoder side, up arrows on the decoder side, joining bottom corners to top corners.
def link(x1, y1, x2, y2):
    a = math.atan2(y2 - y1, x2 - x1)
    return arrow(x1 + 8 * math.cos(a), y1 + 8 * math.sin(a), x2 - 8 * math.cos(a), y2 - 8 * math.sin(a), INK, 2.4)
for k in range(2):
    body += link(xl[k] + side[k] / 2, ys[k] + side[k], xl[k + 1] + side[k + 1] / 2, ys[k + 1])
    body += link(xr[k + 1] + side[k + 1] / 2, ys[k + 1], xr[k] + side[k] / 2, ys[k] + side[k])
body += link(xl[2] + side[2], ys[2] + side[2], bx, by + bs / 2)
body += link(bx + bs, by + bs / 2, xr[2], ys[2] + side[2])
write(body)
