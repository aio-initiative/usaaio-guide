"""One round of GCN message passing into one node, with the true symmetric normalization.

Left: a small graph. Every node carries a three-component feature vector, drawn as three bars inside it. The
terracotta node v has three neighbours (teal, ink, grey) plus a self-loop; the faded nodes two hops away send
nothing this round. Each incoming arrow's width is the real GCN edge weight 1 / sqrt(d_v d_u) with self-loop
degrees, so the leaf neighbour (degree 2) sends the thickest message and the hub (degree 5) the thinnest.
Right: the new state of v, (D^-1/2 (A+I) D^-1/2 H)_v, as three stacked bars. Each bar is built from one segment
per source, coloured like the node it came from and sized by weight times that node's feature.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

# Graph: v = 0, neighbours 1 (leaf), 2, 3 (hub); 4..7 are two hops away.
POS = {0: (350, 205), 1: (255, 75), 2: (285, 330), 3: (195, 215),
       4: (405, 335), 5: (75, 95), 6: (75, 250), 7: (130, 345)}
EDGES = [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5), (3, 6), (3, 7), (5, 6)]
H = {0: (0.8, 0.3, 0.5), 1: (0.9, 0.2, 0.7), 2: (0.2, 0.9, 0.4), 3: (0.5, 0.6, 0.2),
     4: (0.6, 0.4, 0.8), 5: (0.3, 0.7, 0.6), 6: (0.7, 0.5, 0.3), 7: (0.4, 0.8, 0.5)}
COL = {0: ACCENT, 1: SECOND, 2: INK, 3: GREY}

deg = {i: 1 for i in POS}                      # self-loop: A_hat = A + I
for a, b in EDGES:
    deg[a] += 1
    deg[b] += 1
w = {u: 1 / math.sqrt(deg[0] * deg[u]) for u in (0, 1, 2, 3)}
out = [sum(w[u] * H[u][k] for u in w) for k in range(3)]

S = 27                                          # node half-size
body = ""
# Edges: the two-hop ones faint, the ones into v as weighted arrows.
for a, b in EDGES:
    if 0 not in (a, b):
        body += line(*POS[a], *POS[b], GUIDE, 2.2)
for u in (1, 2, 3):
    (x1, y1), (x2, y2) = POS[u], POS[0]
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    sw = 22 * w[u]
    body += arrow(x1 + ux * (S + 8), y1 + uy * (S + 8), x2 - ux * (S + 10), y2 - uy * (S + 10),
                  COL[u], sw, sw * 2.2 + 8)
# Self-loop on v, above it.
x0, y0 = POS[0]
sw = 22 * w[0]
body += (f'<path d="M{x0 - 16},{y0 - S - 2} C{x0 - 34},{y0 - 92} {x0 + 34},{y0 - 92} {x0 + 16},{y0 - S - 6}" '
         f'fill="none" stroke="{ACCENT}" stroke-width="{sw:.1f}" stroke-linecap="round"/>')
body += f'<path d="M{x0 + 13},{y0 - S - 1} L{x0 + 25},{y0 - S - 15} L{x0 + 8},{y0 - S - 18} Z" fill="{ACCENT}"/>'

# Nodes with their feature bars.
def node(i):
    x, y = POS[i]
    c = COL.get(i, GUIDE)
    fill = ACCENT_SOFT if i == 0 else BG
    s = rect(x - S, y - S, 2 * S, 2 * S, fill, c, 3.0 if i in COL else 2.2, 12)
    bc = c if i in COL else GUIDE
    for k, v in enumerate(H[i]):
        bx = x - 13 + 13 * k
        s += line(bx, y + 15, bx, y + 15 - 30 * v, bc, 7)
    return s
for i in POS:
    body += node(i)

# Arrow from v's slot to the output, and the stacked new state of v.
OX, OY0, OY1, BW, GAPB = 450, 60, 330, 24, 10
scale = (OY1 - OY0 - 20) / max(out)
body += arrow(x0 + S + 12, y0, OX - 10, y0, ACCENT, 3.0, 15)
body += rect(OX, OY0, 3 * BW + 2 * GAPB + 32, OY1 - OY0 + 16, ACCENT_SOFT, ACCENT, 2.6, 12)
for k in range(3):
    bx = OX + 16 + k * (BW + GAPB)
    y = OY1
    for u in (0, 1, 2, 3):
        hseg = scale * w[u] * H[u][k]
        body += rect(bx, y - hseg, BW, hseg - 2, COL[u], "none", 0, 2)
        y -= hseg
write(body)
