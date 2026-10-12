"""Two elongated groups joined by single linkage: the minimum spanning tree with its longest edge cut, and the dendrogram it gives."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(11)
# Two long parallel diagonal bands, close across but long along their axis.
pts = []
for g, off in ((0, -46), (1, 46)):
    for i in range(15):
        t = i / 14 + random.uniform(-0.02, 0.02)
        side = random.uniform(-11, 11)          # spread across the band, perpendicular to its axis
        x = 95 + 175 * t + off * 0.8 + side * 0.83
        y = 340 - 270 * t + off * 0.6 + side * 0.55
        pts.append((x, y))
n = len(pts)
D = [[math.dist(a, b) for b in pts] for a in pts]

# Prim's minimum spanning tree.
inT, edges = {0}, []
while len(inT) < n:
    w, i, j = min((D[i][j], i, j) for i in inT for j in range(n) if j not in inT)
    edges.append((w, i, j))
    inT.add(j)
edges.sort()
cut = edges[-1]

# Single linkage = Kruskal over the MST edges; record merges for the dendrogram.
parent = list(range(n))


def find(a):
    while parent[a] != a:
        a = parent[a]
    return a


# Colour by the two components left after removing the longest edge.
parent = list(range(n))
for w, i, j in edges[:-1]:
    parent[find(j)] = find(i)
roots = sorted({find(i) for i in range(n)})
col = {i: (ACCENT if find(i) == roots[0] else SECOND) for i in range(n)}

body = ""
for w, i, j in edges[:-1]:
    body += line(*pts[i], *pts[j], col[i], 2.4)
body += line(*pts[cut[1]], *pts[cut[2]], GREY, 2.0, dash="5 5")
for i, p in enumerate(pts):
    body += dot(*p, 6.5, col[i])

# Dendrogram on the right: leaf order from the merge tree, heights = merge distances.
DX0, DX1, DY0, DY1 = 345, 565, 50, 350
parent = list(range(n))
node = {i: ("leaf", i) for i in range(n)}
for w, i, j in edges:
    a, b = find(i), find(j)
    node[a] = ("join", node[a], node[b], w)
    parent[b] = a
    del node[b]
top = list(node.values())[0]
order = []


def leaves(t):
    if t[0] == "leaf":
        order.append(t[1])
    else:
        leaves(t[1]); leaves(t[2])


leaves(top)
hmax = top[3] * 1.12
xpos = {lf: DX0 + (DX1 - DX0) * k / (n - 1) for k, lf in enumerate(order)}


def hy(h):
    return DY1 - (DY1 - DY0) * h / hmax


def draw(t):
    global body
    if t[0] == "leaf":
        return xpos[t[1]], DY1, col[t[1]]
    xa, ya, ca = draw(t[1])
    xb, yb, cb = draw(t[2])
    y = hy(t[3])
    c = ca if ca == cb else INK
    body += line(xa, ya, xa, y, ca, 2.2) + line(xb, yb, xb, y, cb, 2.2) + line(xa, y, xb, y, c, 2.2)
    return (xa + xb) / 2, y, c


body += line(DX0 - 12, DY1, DX1 + 12, DY1, GUIDE, THIN)
draw(top)
yc = hy((edges[-1][0] + edges[-2][0]) / 2)
body += line(DX0 - 14, yc, DX1 + 14, yc, GREY, 1.8, dash="7 6")
for lf in order:
    body += dot(xpos[lf], DY1, 3.5, col[lf])
write(body)
