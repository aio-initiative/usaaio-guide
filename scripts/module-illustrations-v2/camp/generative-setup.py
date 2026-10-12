"""Real images are a tiny part of all possible images. Left: the space of all pixel arrangements, filled with grey
random points, and a thin terracotta band where real images live. Right: a random 10 x 10 grid (static) that sits
out in the grey, and an 10 x 10 disc image that sits on the band."""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

rng = random.Random(7)

# the space of all images
sx, sy, S = 44, 44, 312
body = rect(sx, sy, S, S, fill=BG, stroke=INK, sw=LINE, rx=6)


def band(t):
    """centre line of the data region: a gentle S-curve across the square"""
    x = sx + 50 + (S - 100) * t
    y = sy + S / 2 + 70 * math.sin(math.pi * (t - 0.5) * 1.6)
    return x, y


def near_band(x, y):
    return min(math.hypot(x - bx, y - by) for bx, by in (band(i / 60) for i in range(61)))


# the band itself: a soft fill polygon
top, bot = [], []
for i in range(61):
    t = i / 60
    x, y = band(t)
    x2, y2 = band(min(1, t + 0.01))
    a = math.atan2(y2 - y, x2 - x) if t < 1 else math.atan2(y - band(t - 0.01)[1], x - band(t - 0.01)[0])
    w = 14 * math.sin(math.pi * t) ** 0.5
    top.append((x - w * math.sin(a), y + w * math.cos(a)))
    bot.append((x + w * math.sin(a), y - w * math.cos(a)))
body += poly(top + bot[::-1], ACCENT, 1.8, fill=ACCENT_SOFT, close=True)

# uniform random points: almost every arrangement of pixels, all static
for _ in range(150):
    x, y = sx + 14 + rng.random() * (S - 28), sy + 14 + rng.random() * (S - 28)
    if near_band(x, y) > 22:
        body += dot(x, y, 3.8, GREY)
# real images: points on the band
for i in range(14):
    t = 0.08 + 0.84 * i / 13 + rng.uniform(-0.02, 0.02)
    x, y = band(t)
    body += dot(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), 4.6, ACCENT)

# two 10 x 10 thumbnails
N, C = 10, 13
G = N * C
tx = W - 44 - G
ty = [52, 400 - 52 - G]


def thumb(y0, shade):
    out = ""
    for r in range(N):
        for c in range(N):
            v = shade(r, c)
            g = int(255 - v * 200)
            out += rect(tx + c * C, y0 + r * C, C, C, fill=f"#{g:02x}{g:02x}{g:02x}", stroke="none", sw=0, rx=0)
    return out + rect(tx, y0, G, G, stroke=INK, sw=LINE, rx=3)


noise = [[rng.random() for _ in range(N)] for _ in range(N)]
body += thumb(ty[0], lambda r, c: noise[r][c])
body += thumb(ty[1], lambda r, c: 0.9 if math.hypot(r - 4.5, c - 4.5) < 3.4 else 0.06)

# links: the static grid to a grey point far from the band, the disc to a point on the band
px, py = sx + 66, sy + 52
body += ring(px, py, 8, INK, 2.4)
body += poly([(px + 8, py), (tx - 8, ty[0] + G / 2)], GREY, 2.6, dash="8 6")
bx, by = band(0.78)
body += ring(bx, by, 8, ACCENT, 2.8)
body += poly([(bx + 8, by + 2), (tx - 8, ty[1] + G / 2)], ACCENT, 2.6, dash="8 6")
write(body)
