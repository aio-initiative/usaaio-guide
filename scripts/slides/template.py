"""The CP Initiative slide template, measured from the reference deck.

Every constant here was read out of "2. Sorting with Custom Comparators.pdf"
with PyMuPDF rather than eyeballed: page geometry, fills, dot-grid coordinates,
font sizes, colours and baselines all match that file.
"""
import pymupdf

# --- page -------------------------------------------------------------------
W, H = 720.0, 405.0                     # 10in x 5.625in, 16:9

# --- palette ----------------------------------------------------------------
PURPLE      = (0x5b/255, 0x21/255, 0xb6/255)
BLUE        = (0x25/255, 0x63/255, 0xeb/255)
PANEL_BLUE  = (0xdb/255, 0xea/255, 0xfe/255)
WHITE       = (1, 1, 1)
TITLE_INK   = (0x18/255, 0x18/255, 0x1b/255)
BODY_INK    = (0x37/255, 0x41/255, 0x51/255)
SUBTITLE    = (0xdd/255, 0xd6/255, 0xfe/255)
CODE_BG     = (0xf9/255, 0xfa/255, 0xfb/255)
CODE_EDGE   = (0xe5/255, 0xe7/255, 0xeb/255)
CODE_PLAIN  = (0x37/255, 0x47/255, 0x4f/255)
CODE_KEY    = (0x3f/255, 0x51/255, 0xb5/255)
CODE_TYPE   = (0x9c/255, 0x27/255, 0xb0/255)
CODE_STR    = (0x0d/255, 0x90/255, 0x4f/255)
CODE_COMMENT= (0x8a/255, 0x8a/255, 0x8a/255)

# --- type scale -------------------------------------------------------------
T_TITLE_HERO   = 33.0   # title slide
T_SUB_HERO     = 25.0
T_TITLE        = 28.0   # content slide heading
T_BODY         = 18.0
T_CODE         = 16.0
LEAD_BODY      = 24.7   # measured line pitch
LEAD_CODE      = 28.8

# --- margins ----------------------------------------------------------------
X_MARGIN   = 34.1
TITLE_Y    = 53.7       # top of the 28pt heading
BODY_TOP   = 101.2      # top of the first body line
BODY_MAX_Y = 358.0      # above the footer logo
TOPBAR_H   = 12.9

FONT_DIR = "assets/slide-fonts"
FONTS = {
    "bold":    f"{FONT_DIR}/Roboto-Bold.ttf",
    "regular": f"{FONT_DIR}/Roboto-Regular.ttf",
    "mono":    f"{FONT_DIR}/RobotoMono-Regular.ttf",
}
_cache = {}


def font(kind):
    if kind not in _cache:
        _cache[kind] = pymupdf.Font(fontfile=FONTS[kind])
    return _cache[kind]


def width(text, kind, size):
    return font(kind).text_length(text, size)


def new_doc():
    return pymupdf.open()


def new_page(doc):
    return doc.new_page(width=W, height=H)


def put(page, pos, text, size, kind, colour):
    """insert_text that tolerates empty strings (PyMuPDF raises on them)."""
    if not text:
        return
    page.insert_text(pos, text, fontsize=size, fontfile=FONTS[kind],
                     fontname={"bold": "rbold", "regular": "rreg", "mono": "rmono"}[kind],
                     color=colour)


def _dot_triangle(page, x0, y0, d, pitch, rows, colour, flush="right"):
    """The stepped dot grid in the corner of every slide."""
    for r in range(rows):
        count = rows - r
        for c in range(count):
            x = x0 + (c + r) * pitch if flush == "right" else x0 + c * pitch
            page.draw_circle((x + d / 2, y0 + r * pitch + d / 2), d / 2,
                             color=None, fill=colour)


def draw_logo(page, x, y, scale=1.0):
    """CP Initiative mark: a purple ring with a three-node graph inside."""
    r = 13.7 * scale
    cx, cy = x + r, y + r
    page.draw_circle((cx, cy), r, color=PURPLE, fill=WHITE, width=1.6 * scale)
    top = (cx, cy - r * 0.42)
    left = (cx - r * 0.42, cy + r * 0.40)
    right = (cx + r * 0.42, cy + r * 0.40)
    for a, b in ((top, left), (top, right), (left, right)):
        page.draw_line(a, b, color=PURPLE, width=1.5 * scale)
    for p in (top, left, right):
        page.draw_circle(p, 2.6 * scale, color=PURPLE, fill=PURPLE)


def footer(page):
    """Small logo lock-up, bottom right, as on every content slide."""
    put(page, (610.0, 371.5), "CP Initiative", 11, "bold", BODY_INK)
    put(page, (627.9, 382.4), "joincpi.org", 8, "regular", BODY_INK)
    draw_logo(page, 683.5, 368.0, scale=1.0)


def content_page(doc, title):
    """Purple top bar, corner dots, heading, footer. Returns the page."""
    page = new_page(doc)
    page.draw_rect(pymupdf.Rect(0, 0, W, TOPBAR_H), color=None, fill=PURPLE)
    _dot_triangle(page, 646.5, 23.1, 10.0, 18.0, 4, BLUE)
    if title:
        put(page, (X_MARGIN, TITLE_Y + T_TITLE * 0.78 + 4.1), title,
            T_TITLE, "bold", TITLE_INK)
    footer(page)
    return page


def title_page(doc, title, subtitle):
    """The cover: blue panel, purple band, white rules, dot grids, logo."""
    page = new_page(doc)
    page.draw_rect(pymupdf.Rect(0, 0, W, H), color=None, fill=WHITE)
    page.draw_rect(pymupdf.Rect(0, 0, 447.1, H), color=None, fill=PANEL_BLUE)

    lines = _wrap(title, "bold", T_TITLE_HERO, 470.0)
    band_h = 162.2 if len(lines) <= 2 else 162.2 + (len(lines) - 2) * 40.0
    band_y = 109.4 - (band_h - 162.2) / 2
    page.draw_rect(pymupdf.Rect(0, band_y, 554.7, band_y + band_h),
                   color=None, fill=PURPLE)

    for i in range(5):                      # white rules on the left edge
        page.draw_rect(pymupdf.Rect(0, 144.5 + i * 21.15, 25.1, 144.5 + i * 21.15 + 9.2),
                       color=None, fill=WHITE)

    _dot_triangle(page, 18.4, 20.5, 10.0, 18.0, 4, BLUE)
    for r in range(8):                      # purple grid, right of the band
        for c in range(5):
            page.draw_circle((565.0 + c * 13.1 + 2.55, 109.4 + r * 22.44 + 2.55),
                             2.55, color=None, fill=PURPLE)

    y = band_y + 29.8 + T_TITLE_HERO * 0.78
    for line in lines:
        put(page, (61.0, y), line, T_TITLE_HERO, "bold", WHITE)
        y += 39.8
    # Shrink the subtitle until it fits inside the band rather than running past it.
    sub_size = T_SUB_HERO
    while sub_size > 11 and width(subtitle, "regular", sub_size) > 470.0:
        sub_size -= 0.5
    put(page, (61.0, band_y + band_h - 26.0), subtitle, sub_size, "regular", SUBTITLE)

    draw_logo(page, 57.3, 312.4, scale=47.0 / 27.4)
    put(page, (109.7, 334.2), "CP Initiative", 18, "bold", BODY_INK)
    put(page, (112.3, 351.0), "joincpi.org", 12, "regular", BODY_INK)
    return page


def _wrap(text, kind, size, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if width(trial, kind, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines
