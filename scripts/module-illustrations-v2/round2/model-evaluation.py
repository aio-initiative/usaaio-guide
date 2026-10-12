"""The lesson's screening test as a binormal classifier, threshold and ROC curve side by side.
The screen flags 40 of 50 sick patients and 90 of 950 healthy ones: recall 0.80 at false positive rate 90/950.
With healthy scores N(0, 1) and sick scores N(mu, 1), that operating point fixes the threshold t = Phi^-1(1 - 0.095)
and mu = t + Phi^-1(0.80). Left: the two score densities (healthy grey, sick terracotta) cut by the dashed
threshold; the shaded tails right of it are the false alarms and the true positives. Right: the ROC curve this
classifier traces as the threshold sweeps, its AUC shaded, the random diagonal dashed, the screen's operating
point with guides to both axes, and the flag-nobody test as a ring at the origin."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))


def Phi_inv(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


fpr0, tpr0 = 90 / 950, 40 / 50
t = Phi_inv(1 - fpr0)
mu = t + Phi_inv(tpr0)
pdf = lambda z, m: math.exp(-(z - m) ** 2 / 2) / math.sqrt(2 * math.pi)

# left panel: densities
L0, L1, T, B = 40, 268, 70, 330
zmin, zmax = -3.2, mu + 3.2
LX = lambda z: L0 + (z - zmin) / (zmax - zmin) * (L1 - L0)
LY = lambda d: B - d / 0.42 * (B - T)
zs = [zmin + (zmax - zmin) * k / 300 for k in range(301)]
body = axes(L0, T - 10, L1, B)
for m, curve, soft in [(mu, ACCENT, ACCENT_SOFT), (0, GREY, GUIDE)]:   # false alarms drawn over the true positives
    tail = [z for z in zs if z >= t]
    body += poly([(LX(t), B)] + [(LX(z), LY(pdf(z, m))) for z in [t] + tail] + [(LX(tail[-1]), B)], "none", 0, soft, close=True)
for m, curve, soft in [(0, GREY, GUIDE), (mu, ACCENT, ACCENT_SOFT)]:
    body += poly([(LX(z), LY(pdf(z, m))) for z in zs], curve, 3)
body += line(LX(t), T - 6, LX(t), B, INK, 2.2, "7 6")

# right panel: ROC
S, R0, RB = 250, 310, 330
RX = lambda f: R0 + f * S
RY = lambda r: RB - r * S
ths = [mu + 4 - (mu + 8) * k / 400 for k in range(401)]
roc = [(1 - Phi(h), 1 - Phi(h - mu)) for h in ths]
roc = [(0, 0)] + roc + [(1, 1)]
body += poly([(RX(f), RY(r)) for f, r in roc] + [(RX(1), RY(0))], "none", 0, ACCENT_SOFT, close=True)
body += rect(R0, RB - S, S, S, "none", GUIDE, 1.4, 0)
body += line(RX(0), RY(0), RX(1), RY(1), GREY, 1.8, "7 6")
body += poly([(RX(f), RY(r)) for f, r in roc], ACCENT, 3.2)
px, py = RX(fpr0), RY(tpr0)
body += line(px, py, px, RB, INK, 1.6, "4 5") + line(px, py, R0, py, INK, 1.6, "4 5")
body += dot(px, py, 8, INK)
body += ring(RX(0), RY(0), 8, INK, 2.6)

write(body)
