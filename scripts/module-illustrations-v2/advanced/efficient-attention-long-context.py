"""The same attention matrix three ways, for 16 tokens drawn from four clusters (shared queries and keys, as in
Reformer), ordered by hash bucket. Left: exact softmax(QK^T / sqrt d), cut into 4 x 4 tiles the way FlashAttention
streams it through SRAM, one tile (terracotta frame) in flight. Middle: its best rank-2 approximation from the SVD,
the low-rank (Linformer, Nystrom) answer, which blurs the sharp entries. Right: LSH attention, where each token
attends only inside its own random-rotation hash bucket, so everything off the buckets is never computed."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

random.seed(48)
N, D = 16, 6
centers = [[random.gauss(0, 1.6) for _ in range(D)] for _ in range(4)]
X = [[c + random.gauss(0, 0.35) for c in centers[i % 4]] for i in range(N)]

# LSH: random rotation R (D x 2), bucket = argmax over [xR, -xR]
R = [[random.gauss(0, 1) for _ in range(2)] for _ in range(D)]
def bucket(x):
    p = [sum(x[j] * R[j][c] for j in range(D)) for c in range(2)]
    v = p + [-q for q in p]
    return max(range(4), key=lambda i: v[i])
order = sorted(range(N), key=lambda i: (bucket(X[i]), i))
X = [X[i] for i in order]
B = [bucket(x) for x in X]

def softmax_rows(S, allow):
    A = []
    for i in range(N):
        m = max(S[i][j] for j in range(N) if allow(i, j))
        e = [math.exp(S[i][j] - m) if allow(i, j) else 0.0 for j in range(N)]
        z = sum(e); A.append([v / z for v in e])
    return A

S = [[sum(X[i][k] * X[j][k] for k in range(D)) / math.sqrt(D) for j in range(N)] for i in range(N)]
exact = softmax_rows(S, lambda i, j: True)
lsh = softmax_rows(S, lambda i, j: B[i] == B[j])

# rank-2 truncation of exact via power iteration with deflation
def rank_k(M, k):
    M = [r[:] for r in M]; out = [[0.0] * N for _ in range(N)]
    for _ in range(k):
        v = [1.0 + 0.01 * i for i in range(N)]
        for _ in range(200):
            u = [sum(M[i][j] * v[j] for j in range(N)) for i in range(N)]
            v = [sum(M[i][j] * u[i] for i in range(N)) for j in range(N)]
            n = math.sqrt(sum(t * t for t in v)); v = [t / n for t in v]
        u = [sum(M[i][j] * v[j] for j in range(N)) for i in range(N)]
        for i in range(N):
            for j in range(N):
                out[i][j] += u[i] * v[j]; M[i][j] -= u[i] * v[j]
    return [[max(0.0, t) for t in r] for r in out]
low = rank_k(exact, 2)

def mix(a, b, t):
    ha = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; hb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ha, hb))

top = max(max(r) for r in exact)

def heat(x0, y0, size, M, col, mask=None):
    cell = size / N
    out = ""
    for i in range(N):
        for j in range(N):
            if mask and not mask(i, j):
                continue
            t = min(1.0, (M[i][j] / top) ** 0.6)
            out += rect(x0 + j * cell, y0 + i * cell, cell + 0.4, cell + 0.4, mix(BG, col, 0.05 + 0.95 * t), "none", 0, 0)
    return out

# exact, cut into FlashAttention tiles, one tile in flight
BX, BY, BS = 44, 64, 272
c = BS / N
b = heat(BX, BY, BS, exact, ACCENT)
for t in range(1, 4):
    b += line(BX + t * 4 * c, BY, BX + t * 4 * c, BY + BS, BG, 3.2)
    b += line(BX, BY + t * 4 * c, BX + BS, BY + t * 4 * c, BG, 3.2)
b += rect(BX, BY, BS, BS, "none", INK, 2.4, 2)
b += rect(BX + 8 * c, BY + 4 * c, 4 * c, 4 * c, "none", INK, 3.6, 2)

# the two approximations, stacked on the right
SX, SS = 424, 132
Y1, Y2 = 50, 218
b += arrow(BX + BS + 14, BY + BS * 0.28, SX - 14, Y1 + SS * 0.5, GREY, 2.4)
b += arrow(BX + BS + 14, BY + BS * 0.72, SX - 14, Y2 + SS * 0.5, GREY, 2.4)
b += heat(SX, Y1, SS, low, SECOND)
b += rect(SX, Y1, SS, SS, "none", INK, 2.2, 2)
b += rect(SX, Y2, SS, SS, "none", GUIDE, THIN, 2)
b += heat(SX, Y2, SS, lsh, SECOND, lambda i, j: B[i] == B[j])
c2 = SS / N
i = 0
while i < N:
    j = i
    while j < N and B[j] == B[i]:
        j += 1
    b += rect(SX + i * c2, Y2 + i * c2, (j - i) * c2, (j - i) * c2, "none", INK, 2.2, 1)
    i = j
write(b)
