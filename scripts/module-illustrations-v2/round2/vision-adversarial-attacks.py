"""The running example's PGD attack at epsilon 0.4: the 6 x 6 image with one bright vertical line, plus the
perturbation (terracotta where a pixel rose, teal where it fell, no cell moves more than 0.4), equals the attacked
image with the line dimmed and two faint horizontal lines. Under each image, the network's two class
probabilities, computed by the lesson's p_vertical: the vertical bar wins before the attack, the horizontal after."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
def p_vertical(img):
    v = np.array([[-1, 2, -1]] * 3, dtype=float)
    scores = []
    for k in (v, v.T):
        m = np.array([[(img[i:i + 3, j:j + 3] * k).sum() for j in range(4)] for i in range(4)])
        pooled = np.maximum(m, 0).reshape(2, 2, 2, 2).max(axis=(1, 3))
        scores.append(0.25 * pooled.sum())
    return np.exp(scores[0]) / np.exp(scores).sum()

image = np.zeros((6, 6)); image[:, 2] = 1.0
attacked = image.copy()
attacked[[0, 2, 3, 5], 2] -= 0.4
attacked[[1, 4], :] = np.clip(attacked[[1, 4], :] + 0.4, 0, 1)
delta = attacked - image
p0, p1 = p_vertical(image), p_vertical(attacked)

def mix(hex_a, hex_b, t):
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]; b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

C = 25                                   # cell size, a 6 x 6 grid is 150 px
GY = 78
def grid(x0, vals, color_of):
    out = ""
    for i in range(6):
        for j in range(6):
            out += f'<rect x="{x0 + j * C:.1f}" y="{GY + i * C:.1f}" width="{C}" height="{C}" fill="{color_of(vals[i, j])}" stroke="{GUIDE}" stroke-width="1"/>'
    return out + rect(x0, GY, 6 * C, 6 * C, "none", INK, 2, 0)

gray = lambda v: mix(BG, INK, v)
def signed(v):
    if v > 0: return mix(BG, ACCENT, v / 0.4 * 0.85)
    if v < 0: return mix(BG, SECOND, -v / 0.4 * 0.85)
    return BG
xs = [40, 225, 410]
body = grid(xs[0], image, gray) + grid(xs[1], delta, signed) + grid(xs[2], attacked, gray)
cy = GY + 3 * C
# plus and equals signs drawn as strokes
mx = (xs[0] + 6 * C + xs[1]) / 2
body += line(mx - 10, cy, mx + 10, cy, INK, 3) + line(mx, cy - 10, mx, cy + 10, INK, 3)
mx = (xs[1] + 6 * C + xs[2]) / 2
body += line(mx - 10, cy - 5, mx + 10, cy - 5, INK, 3) + line(mx - 10, cy + 5, mx + 10, cy + 5, INK, 3)
# class-probability bars under the two images: a tiny vertical-line icon and a horizontal-line icon, then the bar
def bars(x0, pv):
    out = ""
    BW = 6 * C - 34
    for k, (p, horiz) in enumerate([(pv, False), (1 - pv, True)]):
        y = GY + 6 * C + 30 + k * 36
        ix = x0 + 8
        out += (line(ix - 2, y + 10, ix + 18, y + 10, INK, 3.4) if horiz else line(ix + 8, y, ix + 8, y + 20, INK, 3.4))
        win = p > 0.5
        out += rect(x0 + 34, y + 1, BW, 18, "#eef0f2", "none", 0, 3)
        out += rect(x0 + 34, y + 1, BW * p, 18, ACCENT if win else GREY, "none", 0, 3)
    return out
body += bars(xs[0], p0) + bars(xs[2], p1)
write(body)
