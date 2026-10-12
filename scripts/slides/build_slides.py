"""Generates one slide deck per module, in the CP Initiative template.

    python3 scripts/slides/build_slides.py            # all modules
    python3 scripts/slides/build_slides.py clustering # one module

Content comes from content/lessons.js, the same file the reader renders, so a
deck never drifts from the module it is made of. Output: slides/<id>.pdf
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
import template as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slides"
SITE = "aioinitiative.com"

BODY_W = T.W - 2 * T.X_MARGIN          # usable text width
BASE = 2.6                              # glyph-top correction, measured against the reference
BULLET_X = 52.0
BULLET_TEXT_X = 98.0
BULLET_W = T.W - BULLET_TEXT_X - T.X_MARGIN

# --- inline markdown --------------------------------------------------------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOL = {
    r"\times": "×", r"\cdot": "·", r"\le": "≤", r"\leq": "≤", r"\ge": "≥",
    r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\in": "∈", r"\notin": "∉",
    r"\subset": "⊂", r"\subseteq": "⊆", r"\to": "→", r"\rightarrow": "→",
    r"\Rightarrow": "⇒", r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\sum": "Σ", r"\prod": "Π", r"\int": "∫", r"\pm": "±", r"\ldots": "…",
    r"\dots": "…", r"\cdots": "⋯", r"\mid": "|", r"\ll": "≪", r"\gg": "≫",
    r"\sqrt": "√", r"\propto": "∝", r"\equiv": "≡", r"\forall": "∀",
    r"\exists": "∃", r"\emptyset": "∅", r"\cup": "∪", r"\cap": "∩",
}
SUP = str.maketrans("0123456789+-n2i", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ²ⁱ")
SUB = str.maketrans("0123456789+-aeijkmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋ₐₑᵢⱼₖₘₙₒₚᵣₛₜᵤᵥₓ")


def tex_to_text(s):
    """Readable Unicode for maths mathtext refuses. Fallback only.

    Commands are matched as whole tokens: an earlier version substring-replaced
    "\\le" and turned "\\left" into "\u2264ft".
    """
    s = re.sub(r"\\(?:mathbb|mathcal|mathrm|mathbf|mathtt|text|texttt|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[tdc]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\(?:lVert|rVert)\b", "\u2016", s)
    s = re.sub(r"\\(?:lvert|rvert)\b", "|", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\(?:quad|qquad)\b", "  ", s)
    s = re.sub(r"\\[;,:!]", " ", s)

    named = {k.lstrip("\\"): v for k, v in SYMBOL.items()}
    named.update(GREEK)

    def swap(m):
        name = m.group(1)
        return named.get(name, "")          # unknown commands drop out entirely

    s = re.sub(r"\\([A-Za-z]+)", swap, s)
    s = re.sub(r"\^\{([^{}]+)\}", lambda m: m.group(1).translate(SUP), s)
    s = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUP), s)
    s = re.sub(r"_\{([^{}]+)\}", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"_(\w)", lambda m: m.group(1).translate(SUB), s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"\s*([=+\u2264\u2265\u2260\u2248<>])\s*", r" \1 ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def inline_runs(text):
    """Split markdown into (text, style) runs; style is 'regular' or 'mono'."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # links -> label
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)                # bold
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)       # italic
    text = html.unescape(text)
    runs, pos = [], 0
    for m in re.finditer(r"`([^`]+)`|\$([^$]+)\$", text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], "regular"))
        if m.group(1) is not None:
            runs.append((m.group(1), "mono"))
        else:
            runs.append((m.group(2), "math"))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], "regular"))
    return [(t, s) for t, s in runs if t]


def math_width(tex, size):
    img = math_image(tex, size)
    return img[1] if img else T.width(tex_to_text(tex), "regular", size)


def wrap_runs(runs, size, max_w):
    """Greedy wrap across styled runs. Returns list of lines of runs."""
    lines, cur, cur_w = [], [], 0.0
    for text, style in runs:
        if style == "math":
            w = math_width(text, size)
            if cur_w + w > max_w and cur:
                lines.append(cur); cur, cur_w = [], 0.0
            cur.append((text, style)); cur_w += w
            continue
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            w = T.width(word, style, size)
            if cur_w + w > max_w and cur and word.strip():
                lines.append(cur); cur, cur_w = [], 0.0
                if not word.strip():
                    continue
            if not cur and not word.strip():
                continue
            cur.append((word, style)); cur_w += w
    if cur:
        lines.append(cur)
    return lines


def draw_runs(page, x, y, line, size, colour):
    for text, style in line:
        if not text:
            continue
        if style == "math":
            img = math_image(text, size)
            if img:
                data, w, h = img
                # sit the image on the text baseline
                rect = pymupdf.Rect(x, y - h * 0.78, x + w, y + h * 0.22)
                page.insert_image(rect, stream=data)
                x += w
            else:
                fallback = tex_to_text(text)
                T.put(page, (x, y), fallback, size, "regular", colour)
                x += T.width(fallback, "regular", size)
            continue
        T.put(page, (x, y), text, size, "mono" if style == "mono" else "regular",
              T.PURPLE if style == "mono" else colour)
        x += T.width(text, style, size)


# --- syntax colouring for code slides ---------------------------------------
KEYWORDS = set("""import from as def class return if elif else for while in and or not
is None True False with try except finally raise lambda yield global nonlocal pass
break continue assert del print""".split())
TYPES = set("""int float str bool list dict set tuple np pd plt torch nn F self""".split())


def code_runs(line):
    out = []
    if line.lstrip().startswith("#"):
        return [(line, T.CODE_COMMENT)]
    for tok in re.split(r"(\W)", line):
        if not tok:
            continue
        if tok in KEYWORDS:
            out.append((tok, T.CODE_KEY))
        elif tok in TYPES:
            out.append((tok, T.CODE_TYPE))
        else:
            out.append((tok, T.CODE_PLAIN))
    merged = []
    for tok, col in out:
        if merged and merged[-1][1] == col:
            merged[-1] = (merged[-1][0] + tok, col)
        else:
            merged.append((tok, col))
    return merged



_thumb_cache = {}


def _downsampled(path, target_w):
    """PNG bytes for `path` scaled to target_w, cached per (path, width)."""
    key = (str(path), target_w)
    if key not in _thumb_cache:
        from PIL import Image
        im = Image.open(path)
        h = round(im.height * target_w / im.width)
        im = im.convert("RGB").resize((target_w, h), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        _thumb_cache[key] = buf.getvalue()
    return _thumb_cache[key]



# --- math -------------------------------------------------------------------
# The Unicode fallback below cannot handle real LaTeX (\left, \qquad, \mathbb
# and friends), so maths is rendered with matplotlib's mathtext and placed as an
# image. tex_to_text is kept only for when mathtext refuses a string.
_math_cache = {}



def _mathtext_ready(tex):
    """Rewrite the LaTeX mathtext does not implement into what it does."""
    s = tex.strip()
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", s)   # aligned, cases, ...
    s = s.replace("\\\\", " ").replace("&", " ")
    s = re.sub(r"\\boxed\s*\{", "{", s)
    s = re.sub(r"\\(?:bigg?|Bigg?)([lrm]?)", "", s)               # \big( -> (
    s = s.replace("\\lVert", "\\|").replace("\\rVert", "\\|")
    s = s.replace("\\lvert", "|").replace("\\rvert", "|")
    s = re.sub(r"\\(?:quad|qquad)", r"\\ \\ ", s)
    s = re.sub(r"\\[;,:!]", " ", s)
    s = s.replace("\\text{", "\\mathrm{").replace("\\operatorname{", "\\mathrm{")
    s = s.replace("\\texttt{", "\\mathtt{")
    s = re.sub(r"\\underbrace\s*\{", "{", s)
    s = re.sub(r"\\lor", r"\\vee", s)
    s = re.sub(r"\\land", r"\\wedge", s)
    s = s.replace("\\Longrightarrow", "\\Rightarrow").replace("\\Longleftarrow", "\\Leftarrow")
    s = s.replace("\\longrightarrow", "\\rightarrow")
    s = re.sub(r"\\[tdc]frac", r"\\frac", s)
    s = re.sub(r"\\xrightarrow\s*\{([^{}]*)\}", r" \\rightarrow ", s)
    s = re.sub(r"\\(?:mathbin|mathrel|phantom)\s*\{[^{}]*\}", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def math_image(tex, size, colour="#18181b"):
    """Render `tex` to (png_bytes, width_pt, height_pt), or None if it fails."""
    key = (tex, round(size, 1), colour)
    if key in _math_cache:
        return _math_cache[key]
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["mathtext.fontset"] = "dejavusans"
        cleaned = _mathtext_ready(tex)
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${cleaned}$", fontsize=size, color=colour)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=260, bbox_inches="tight",
                    pad_inches=0.02, transparent=True)
        plt.close(fig)
        data = buf.getvalue()
        px_w, px_h = _png_size(data)
        # 260 dpi render, placed at 72 pt/in
        result = (data, px_w * 72.0 / 260.0, px_h * 72.0 / 260.0)
    except Exception:
        result = None
    _math_cache[key] = result
    return result


def _png_size(data):
    import struct
    return struct.unpack(">II", data[16:24])


class Deck:
    def __init__(self, module):
        self.doc = T.new_doc()
        self.module = module
        self.page = None
        self.y = 0.0

    def title(self):
        T.title_page(self.doc, self.module["title"],
                     f"{SITE}/module/?id={self.module['id']}")

    def start(self, heading):
        self.page = T.content_page(self.doc, heading)
        self.y = T.BODY_TOP if heading else T.TITLE_Y

    def room(self, need):
        return self.y + need <= T.BODY_MAX_Y

    def ensure(self, need, heading):
        if self.page is None or not self.room(need):
            self.start(heading)

    def paragraph(self, text, heading):
        lines = wrap_runs(inline_runs(text), T.T_BODY, BODY_W)
        i = 0
        while i < len(lines):
            self.ensure(T.LEAD_BODY, heading)
            while i < len(lines) and self.room(T.LEAD_BODY):
                draw_runs(self.page, T.X_MARGIN, self.y + T.T_BODY * 0.78 + BASE,
                          lines[i], T.T_BODY, T.BODY_INK)
                self.y += T.LEAD_BODY
                i += 1
        self.y += 10.0

    def bullets(self, items, heading, ordered=False):
        for n, item in enumerate(items, 1):
            lines = wrap_runs(inline_runs(item), T.T_BODY, BULLET_W)
            self.ensure(T.LEAD_BODY * min(len(lines), 2), heading)
            marker = f"{n}." if ordered else "●"
            T.put(self.page, (BULLET_X, self.y + T.T_BODY * 0.78 + BASE), marker,
                  T.T_BODY if ordered else 11, "regular", T.BODY_INK)
            for k, line in enumerate(lines):
                if not self.room(T.LEAD_BODY):
                    self.start(heading)
                draw_runs(self.page, BULLET_TEXT_X, self.y + T.T_BODY * 0.78 + BASE,
                          line, T.T_BODY, T.BODY_INK)
                self.y += T.LEAD_BODY
            self.y += 6.0
        self.y += 6.0

    def code(self, lines, heading):
        chunk = []
        for line in lines + [None]:
            if line is not None:
                chunk.append(line)
            if line is None or len(chunk) == 9:
                if chunk:
                    self._code_panel(chunk, heading)
                    chunk = []

    def _code_panel(self, lines, heading):
        panel_h = len(lines) * T.LEAD_CODE + 22.0
        self.ensure(panel_h, heading)
        if not self.room(panel_h):
            self.start(heading)
        rect = pymupdf.Rect(T.X_MARGIN - 6, self.y, T.W - T.X_MARGIN + 6, self.y + panel_h)
        self.page.draw_rect(rect, color=T.CODE_EDGE, fill=T.CODE_BG, width=0.8)
        y = self.y + 11.0
        for line in lines:
            x = T.X_MARGIN + 6
            size = T.T_CODE
            while T.width(line, "mono", size) > BODY_W - 12 and size > 8:
                size -= 0.5
            for tok, col in code_runs(line):
                T.put(self.page, (x, y + size * 0.78), tok, size, "mono", col)
                x += T.width(tok, "mono", size)
            y += T.LEAD_CODE
        self.y += panel_h + 12.0

    def display_math(self, tex, heading):
        img = math_image(tex, T.T_BODY + 2)
        if img is None:                       # mathtext refused it
            text = tex_to_text(tex)
            for line in wrap_runs([(text, "regular")], T.T_BODY, BODY_W - 80):
                self.ensure(T.LEAD_BODY, heading)
                w = sum(math_width(a, T.T_BODY) if b == "math"
                    else T.width(a, b, T.T_BODY) for a, b in line)
                draw_runs(self.page, (T.W - w) / 2, self.y + T.T_BODY * 0.78 + BASE,
                          line, T.T_BODY, T.TITLE_INK)
                self.y += T.LEAD_BODY
            self.y += 12.0
            return
        data, w, h = img
        if w > BODY_W:                        # scale a wide equation to fit
            h *= BODY_W / w
            w = BODY_W
        self.ensure(h + 18, heading)
        if not self.room(h + 18):
            self.start(heading)
        rect = pymupdf.Rect((T.W - w) / 2, self.y + 6, (T.W - w) / 2 + w, self.y + 6 + h)
        self.page.insert_image(rect, stream=data)
        self.y += h + 18

    def table(self, tbl, heading):
        head, rows = tbl.get("head", []), tbl.get("rows", [])
        if not head:
            return
        cols = len(head)
        cw = BODY_W / cols
        need = (len(rows) + 1) * 26.0 + 10
        self.ensure(need, heading)
        if not self.room(need):
            self.start(heading)
        y = self.y
        for c, cell in enumerate(head):
            if not cell.strip():
                continue
            ls = wrap_runs(inline_runs(cell), 13, cw - 8)
            if ls:
                draw_runs(self.page, T.X_MARGIN + c * cw + 4, y + 12, ls[0], 13, T.TITLE_INK)
        y += 24
        self.page.draw_line((T.X_MARGIN, y - 6), (T.W - T.X_MARGIN, y - 6),
                            color=T.CODE_EDGE, width=0.8)
        for row in rows:
            for c, cell in enumerate(row[:cols]):
                ls = wrap_runs(inline_runs(cell), 13, cw - 8)
                if ls:
                    draw_runs(self.page, T.X_MARGIN + c * cw + 4, y + 12, ls[0], 13, T.BODY_INK)
            y += 24
        self.y = y + 12

    def image(self, src, heading):
        path = ROOT / src
        if not path.exists():
            return
        pix = pymupdf.Pixmap(str(path))
        avail_w, avail_h = BODY_W, T.BODY_MAX_Y - T.BODY_TOP - 10
        scale = min(avail_w / pix.width, avail_h / pix.height)
        w, h = pix.width * scale, pix.height * scale
        self.start(heading)
        rect = pymupdf.Rect((T.W - w) / 2, self.y, (T.W - w) / 2 + w, self.y + h)
        # The figures are rendered at 280 dpi for the website. A slide only ever
        # shows them at ~650pt wide, so embedding the original would bloat the
        # deck; downsample to twice the placed size and no more.
        target = int(w * 2)
        if pix.width > target:
            data = _downsampled(path, target)
            self.page.insert_image(rect, stream=data)
        else:
            self.page.insert_image(rect, filename=str(path))
        self.y += h + 10

    def save(self, path):
        # Each deck embeds three full TTFs otherwise; subsetting keeps only the
        # glyphs actually used.
        try:
            self.doc.subset_fonts()
        except Exception:
            pass
        self.doc.save(str(path), deflate=True, garbage=4)
        return self.doc.page_count



# --- content selection ------------------------------------------------------
# A module is long-form prose; a deck is not. Per topic we keep the framing
# paragraphs, one worked list, one equation, one figure and one code listing,
# which lands a typical module at roughly the density of the reference deck.
MAX_PARAS_PER_TOPIC = 2
MAX_CODE_LINES = 12
SLIDE_BUDGET = 40


def condense(section):
    """Pick the blocks worth presenting from one lesson section."""
    kept, paras = [], 0
    seen = {"list": False, "tex": False, "image": False, "code": False}
    for block in section["blocks"]:
        if "heading" in block:
            # sub-headings restart the per-topic budget rather than nesting
            paras = 0
            seen = dict.fromkeys(seen, False)
            kept.append(block)
        elif "p" in block:
            if paras < MAX_PARAS_PER_TOPIC:
                kept.append(block); paras += 1
        elif "quote" in block:
            kept.append(block)
        elif "list" in block or "olist" in block:
            if not seen["list"]:
                kept.append(block); seen["list"] = True
        elif "tex" in block:
            if not seen["tex"]:
                kept.append(block); seen["tex"] = True
        elif "image" in block:
            if not seen["image"]:
                kept.append(block); seen["image"] = True
        elif "code" in block:
            if not seen["code"]:
                lines = block["code"].get("lines", [])
                if len(lines) > MAX_CODE_LINES:
                    block = {"code": dict(block["code"],
                                          lines=lines[:MAX_CODE_LINES] + ["# ..."])}
                kept.append(block); seen["code"] = True
        elif "table" in block:
            kept.append(block)
    return kept


def build(module, lesson):
    deck = Deck(module)
    deck.title()
    for section in lesson["sections"]:
        if section["id"] == "prerequisites":
            continue
        if deck.doc.page_count >= SLIDE_BUDGET:
            break
        heading = section["title"] or module["title"]
        deck.start(heading)
        for block in condense(section):
            if deck.doc.page_count >= SLIDE_BUDGET:
                break
            if "p" in block:
                deck.paragraph(block["p"], heading)
            elif "heading" in block:
                deck.start(block["heading"])
            elif "list" in block:
                deck.bullets(block["list"], heading)
            elif "olist" in block:
                deck.bullets(block["olist"], heading, ordered=True)
            elif "code" in block:
                deck.code(block["code"].get("lines", []), heading)
            elif "tex" in block:
                deck.display_math(block["tex"], heading)
            elif "table" in block:
                deck.table(block["table"], heading)
            elif "image" in block:
                deck.image(block["image"]["src"], heading)
            elif "quote" in block:
                deck.paragraph(" ".join(block["quote"]), heading)
    OUT.mkdir(exist_ok=True)
    return deck.save(OUT / f"{module['id']}.pdf")


def load_content():
    """Modules and lesson blocks, read from the built site data."""
    node = subprocess.run(
        ["node", "-e", """
const fs=require('fs'),vm=require('vm');
const L={window:{}}; vm.runInNewContext(fs.readFileSync('content/lessons.js','utf8'),L);
const C={window:{}}; vm.runInNewContext(fs.readFileSync('content/catalog.js','utf8'),C);
process.stdout.write(JSON.stringify({modules:C.window.AIO_CONTENT.modules,
                                     lessons:L.window.AIO_LESSONS.lessons}));
"""], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(node.stdout)


def main():
    data = load_content()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    total = 0
    for module in data["modules"]:
        if only and only != module["id"]:
            continue
        lesson = data["lessons"].get(module["id"])
        if not lesson:
            continue
        n = build(module, lesson)
        total += n
        print(f"  {n:3d} slides  {module['id']}")
    print(f"\n{total} slides written to slides/")


if __name__ == "__main__":
    main()
