"""Pandas Data Cleaning: the lesson's messy housing table -> the cleaned table -> mean price per city.

Each cell holds a bar whose length is the real value. Left: 7 rows with a repeated row (struck), a '?' size and a
blank rooms cell (terracotta), two badly written city names (hollow swatch). Right: 6 rows, the gaps filled with the
column medians 95 and 3 (teal). Far right: groupby city, the true means 530, 330, 375.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

CITY_COL = {"Boston": INK, "Austin": SECOND, "Denver": ACCENT}
# city, clean?, size, rooms, price   (None = missing)
messy = [("Boston", True, 80, 3, 500), ("Boston", False, 95, 3, 560), ("Austin", True, 120, 4, 410),
         ("Austin", True, None, 2, 250), ("Denver", True, 70, None, 300), ("Austin", True, 120, 4, 410),
         ("Denver", False, 110, 4, 450)]
DUP = 5
sizes = sorted(r[2] for i, r in enumerate(messy) if r[2] is not None and i != DUP)
rooms = sorted(r[3] for i, r in enumerate(messy) if r[3] is not None and i != DUP)
med_size, med_rooms = sizes[len(sizes) // 2], rooms[len(rooms) // 2]          # 95, 3
MAX = {2: 130, 3: 4.6, 4: 600}

body = ""
RH = 38


def table(x0, y0, rows, widths, marks):
    out = ""
    n = len(rows)
    tw = sum(widths)
    out += rect(x0, y0, tw, n * RH, fill=BG, stroke=INK, sw=2, rx=3)
    for i, (city, ok, *vals) in enumerate(rows):
        y = y0 + i * RH
        kind = marks.get(i, {})
        if kind.get("row") == "dup":
            out += rect(x0 + 1.5, y + 1.5, tw - 3, RH - 3, fill=ACCENT_SOFT, stroke="none", sw=0, rx=2)
        if i:
            out += line(x0, y, x0 + tw, y, GUIDE, THIN)
        cx = x0 + widths[0] / 2
        out += dot(cx, y + RH / 2, 8, CITY_COL[city]) if ok else ring(cx, y + RH / 2, 7, CITY_COL[city], 2.6)
        xx = x0 + widths[0]
        for j, v in enumerate(vals):
            w = widths[j + 1]
            col = j + 2
            cell = kind.get(col)
            if cell == "missing":
                out += rect(xx + 3, y + 4, w - 6, RH - 8, fill=ACCENT_SOFT, stroke=ACCENT, sw=2, rx=3)
            elif cell == "filled":
                out += rect(xx + 3, y + 4, w - 6, RH - 8, fill=SECOND_SOFT, stroke="none", sw=0, rx=3)
            if v is not None:
                bl = (w - 16) * v / MAX[col]
                out += line(xx + 8, y + RH / 2, xx + 8 + bl, y + RH / 2,
                            SECOND if cell == "filled" else GREY, 7)
            xx += w
        if kind.get("row") == "dup":
            out += line(x0 + 6, y + RH / 2, x0 + tw - 6, y + RH / 2, ACCENT, 3)
    xx = x0
    for w in widths[:-1]:
        xx += w
        out += line(xx, y0, xx, y0 + n * RH, GUIDE, THIN)
    return out


widths = [36, 50, 42, 58]
top = (H - 7 * RH) / 2
body += table(40, top, messy, widths,
              {DUP: {"row": "dup"}, 3: {2: "missing"}, 4: {3: "missing"}})

clean = [r for i, r in enumerate(messy) if i != DUP]
clean = [(c, True, s if s is not None else med_size, rm if rm is not None else med_rooms, p)
         for c, ok, s, rm, p in clean]
top2 = (H - 6 * RH) / 2
body += arrow(236, H / 2, 268, H / 2, INK, 2.6)
body += table(278, top2, clean, widths, {3: {2: "filled"}, 4: {3: "filled"}})

# groupby city -> mean price
cities = ["Boston", "Austin", "Denver"]
means = {c: sum(r[4] for r in clean if r[0] == c) / sum(1 for r in clean if r[0] == c) for c in cities}
bx0, base = 500, top2 + 6 * RH
body += arrow(278 + sum(widths) + 10, H / 2, bx0 - 10, H / 2, INK, 2.6)
body += line(bx0, base, 562, base, GUIDE, THIN)
for k, c in enumerate(cities):
    h = (6 * RH - 20) * means[c] / 600
    body += rect(bx0 + 4 + k * 20, base - h, 15, h, fill=CITY_COL[c], stroke="none", sw=0, rx=1)
write(body)
