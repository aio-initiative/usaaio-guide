"""Non-maximum suppression on a detector grid: two objects, each found by several candidate boxes; the most confident
box per object is kept (terracotta) and the overlapping ones are dropped (dashed grey). Right: the IoU of the kept box
with the strongest dropped one, intersection over union, as a bar against the 0.5 threshold."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

X0, Y0, X1, Y1 = 40, 40, 384, 360           # the image
COLS, ROWS = 4, 4
cw, ch = (X1 - X0) / COLS, (Y1 - Y0) / ROWS

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)

# Ground-truth boxes (left, top, right, bottom): a standing person and a car.
person = (62, 78, 142, 330)
car = (206, 210, 362, 316)
# Candidate boxes with confidences, as a detector's cells would emit them.
cands = [
    ((66, 84, 140, 326), 0.92), ((54, 96, 150, 336), 0.71), ((70, 70, 134, 300), 0.64), ((58, 110, 156, 322), 0.55),
    ((210, 206, 358, 318), 0.88), ((196, 216, 352, 326), 0.69), ((220, 200, 370, 308), 0.58),
]
# Greedy NMS at IoU 0.5.
order = sorted(range(len(cands)), key=lambda i: -cands[i][1])
kept, dropped = [], {}
for i in order:
    hit = next((k for k in kept if iou(cands[i][0], cands[k][0]) > 0.5), None)
    if hit is None:
        kept.append(i)
    else:
        dropped[i] = hit

body = rect(X0, Y0, X1 - X0, Y1 - Y0, "#fafbfc", GUIDE, THIN, rx=3)
for k, (box, _) in enumerate([(person, 0), (car, 0)]):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    gi, gj = int((cx - X0) // cw), int((cy - Y0) // ch)
    body += rect(X0 + gi * cw, Y0 + gj * ch, cw, ch, ACCENT_SOFT, "none", 0, rx=0)
for c in range(1, COLS):
    body += line(X0 + c * cw, Y0, X0 + c * cw, Y1, GUIDE, THIN)
for r in range(1, ROWS):
    body += line(X0, Y0 + r * ch, X1, Y0 + r * ch, GUIDE, THIN)

# Silhouettes: a person (head and body) and a car (body, cabin, wheels).
SIL, SILE = "#e6e8eb", "#b9bfc6"
px0, py0, px1, py1 = person
pcx = (px0 + px1) / 2
body += dot(pcx, py0 + 26, 24, SIL, SILE, 2)
body += rect(pcx - 34, py0 + 56, 68, 110, SIL, SILE, 2, rx=22)
body += rect(pcx - 28, py0 + 150, 24, py1 - py0 - 150, SIL, SILE, 2, rx=10)
body += rect(pcx + 4, py0 + 150, 24, py1 - py0 - 150, SIL, SILE, 2, rx=10)
cx0, cy0, cx1, cy1 = car
body += poly([(cx0 + 40, cy0 + 40), (cx0 + 62, cy0 + 4), (cx1 - 50, cy0 + 4), (cx1 - 22, cy0 + 40)], SILE, 2, fill=SIL, close=True)
body += rect(cx0 + 2, cy0 + 38, cx1 - cx0 - 4, 44, SIL, SILE, 2, rx=14)
for wx in (cx0 + 38, cx1 - 38):
    body += dot(wx, cy1 - 20, 19, SIL, SILE, 2) + dot(wx, cy1 - 20, 7, SILE)

# Dropped candidates first, then the kept box, then the responsible cell's center.
for i in dropped:
    b = cands[i][0]
    body += rect(b[0], b[1], b[2] - b[0], b[3] - b[1], "none", GREY, 1.9, rx=2).replace("/>", ' stroke-dasharray="7 5"/>')
for i in kept:
    b = cands[i][0]
    body += rect(b[0], b[1], b[2] - b[0], b[3] - b[1], "none", ACCENT, 3.4, rx=2)
    body += dot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 6, ACCENT)

# Right: IoU of the kept car box with its strongest dropped rival, drawn to scale.
k = kept[1]
rival = max((i for i, h in dropped.items() if h == k), key=lambda i: cands[i][1])
A, B = cands[k][0], cands[rival][0]
ux0, uy0 = min(A[0], B[0]), min(A[1], B[1])
ux1, uy1 = max(A[2], B[2]), max(A[3], B[3])
s = 0.86
OX, OY = 484 - (ux0 + ux1) / 2 * s, 165 - (uy0 + uy1) / 2 * s
T = lambda b: (OX + b[0] * s, OY + b[1] * s, OX + b[2] * s, OY + b[3] * s)
a, b = T(A), T(B)
I = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
body += rect(a[0], a[1], a[2] - a[0], a[3] - a[1], ACCENT_SOFT, "none", 0, rx=0)
body += rect(b[0], b[1], b[2] - b[0], b[3] - b[1], ACCENT_SOFT, "none", 0, rx=0)
body += rect(I[0], I[1], I[2] - I[0], I[3] - I[1], ACCENT, "none", 0, rx=0).replace("/>", ' fill-opacity="0.55"/>')
body += rect(b[0], b[1], b[2] - b[0], b[3] - b[1], "none", GREY, 1.9, rx=2).replace("/>", ' stroke-dasharray="7 5"/>')
body += rect(a[0], a[1], a[2] - a[0], a[3] - a[1], "none", ACCENT, 3.2, rx=2)

# IoU bar: track = union, fill = intersection / union, tick = the 0.5 threshold.
v = iou(A, B)
bx0, bx1, by = 414, 554, 296
body += rect(bx0, by - 9, bx1 - bx0, 18, "#eceef1", "none", 0, rx=3)
body += rect(bx0, by - 9, (bx1 - bx0) * v, 18, ACCENT, "none", 0, rx=3)
tx = bx0 + (bx1 - bx0) * 0.5
body += line(tx, by - 17, tx, by + 17, INK, 2.2)
write(body)
print("IoU kept/rival", round(v, 3), "kept", kept, "dropped", dropped)
