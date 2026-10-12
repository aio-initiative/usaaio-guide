"""Bias and variance, measured: degree 1, 3 and 9 polynomials fitted to 25 training sets (20 evenly spaced x, fresh noise each time) of a sine.
Thin grey curves are the separate fits, the bold terracotta curve is their average, the dashed curve is the truth.
Degree 1 is stable but far from the sine (bias), degree 3 is close and steady, degree 9 swings wildly (variance)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

import numpy as np
rng = np.random.default_rng(4)
truth = lambda x: np.sin(2 * np.pi * x)
xs = np.linspace(0, 1, 120)
PW, GAP, X0, Y0, Y1 = 160, 20, 40, 44, 356
body, defs = "", ""
for k, deg in enumerate((1, 3, 9)):
    px = X0 + k * (PW + GAP)
    sx = lambda v: px + v * PW
    sy = lambda v: (Y0 + Y1) / 2 - v * (Y1 - Y0) / 2 / 1.7
    defs += f'<clipPath id="p{k}"><rect x="{px}" y="{Y0}" width="{PW}" height="{Y1 - Y0}"/></clipPath>'
    fits = []
    for _ in range(25):
        x = np.linspace(0, 1, 20)
        y = truth(x) + rng.normal(0, 0.3, 20)
        fits.append(np.polyval(np.polyfit(x, y, deg), xs))
    body += rect(px, Y0, PW, Y1 - Y0, "none", GUIDE, THIN, 6)
    body += line(px, sy(0), px + PW, sy(0), GUIDE, THIN)
    g = f'<g clip-path="url(#p{k})">'
    for f in fits:
        g += poly(list(zip(map(sx, xs), map(sy, np.clip(f, -3, 3)))), GREY, 1.1)
    g += poly(list(zip(map(sx, xs), map(sy, truth(xs)))), INK, 2, dash="7 6")
    g += poly(list(zip(map(sx, xs), map(sy, np.clip(np.mean(fits, 0), -3, 3)))), ACCENT, 3.4)
    body += g + "</g>"
write(body, defs)
