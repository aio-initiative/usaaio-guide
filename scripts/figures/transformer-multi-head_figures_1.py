"""Corrected copy of B tools/diagrams/gen_transformer_figures.py for the
transformer-multi-head module's two figures.

B's script hardcodes head 1's output for "sat" as 0.872, but the
published attention() code gives 0.871248, which rounds to 0.871
(pack-75 fix 1, bver-75; rechecked with numpy). The lesson's tables, code
and alt text all print 0.871, so the figure has to as well. Only the two
values inside transformer_multi_head() are changed; every other figure in
this file belongs to another lane and is left alone. Run with
scripts/import-b/figures.py, which resolves usaaio_theme to A's b_theme.

"""

"""Figures for the Build a Transformer steps.

Look and feel is entirely `usaaio_theme`. As in `gen_guide_figures_4.py`,
structure carries structure: a solid line is the main path, a dashed line is a
shortcut or a loop, and a heavier outline is the part the figure is about.
"""

from usaaio_theme import (HAIR, LINE, THIN, P, arrow, block, clean, label,
                          render, run_all)

DASH = (0, (4, 2.5))


def fmt(v, digits=3):
    """A number as the pages print it: fixed decimals and a real minus sign."""
    if isinstance(v, str):
        return v
    return f"{v:.{digits}f}".replace("-", "−")


def cells(ax, x0, y0, rows, cw=1.0, ch=0.5, *, on=True, digits=3, muted=()):
    """A grid of numbers with its top-left row first. `muted` lists (row, col)
    cells drawn in the receding ink, for values that are switched off."""
    from matplotlib.patches import Rectangle

    n = len(rows)
    for i, row in enumerate(rows):
        y = y0 + (n - 1 - i) * ch
        for j, v in enumerate(row):
            quiet = (i, j) in muted or not on
            ax.add_patch(Rectangle(
                (x0 + j * cw, y), cw, ch, facecolor="none",
                edgecolor=P.series[2] if quiet else P.ink,
                lw=HAIR if quiet else THIN, zorder=2 if quiet else 3))
            label(ax, x0 + (j + 0.5) * cw, y + ch / 2, fmt(v, digits),
                  rank="small", ha="center", va="center",
                  color=P.mute if quiet else P.ink)


# ------------------------------------------- the whole model, section by section
@render("transformer-steps", size="tower")
def transformer_steps(ax):
    cx, w, h = 0.47, 0.62, 0.048
    left = cx - w / 2
    ys = {
        "tok": 0.07, "emb": 0.16, "pos": 0.25,
        "attn": 0.37, "an1": 0.46, "ffn": 0.55, "an2": 0.64,
        "pred": 0.80,
    }

    label(ax, cx, 0.025, "“the cat sat”", rank="label", ha="center")
    block(ax, left, ys["tok"], w, h, "Tokenize")
    block(ax, left, ys["emb"], w, h, "Embed")
    block(ax, left, ys["pos"], w, h, "Add position")
    block(ax, left, ys["attn"], w, h, "Masked multi-head attention",
          emph=True)
    block(ax, left, ys["an1"], w, h, "Add & norm")
    block(ax, left, ys["ffn"], w, h, "Feed-forward")
    block(ax, left, ys["an2"], w, h, "Add & norm")
    block(ax, left, ys["pred"], w, h, "Predict")
    label(ax, cx, 0.905, r"next token $x_4$ = “on”", rank="label",
          ha="center")

    # Main path.
    arrow(ax, (cx, 0.045), (cx, ys["tok"] - 0.004))
    order = ["tok", "emb", "pos", "attn", "an1", "ffn", "an2", "pred"]
    for a, b in zip(order, order[1:]):
        arrow(ax, (cx, ys[a] + h), (cx, ys[b] - 0.004))
    arrow(ax, (cx, ys["pred"] + h), (cx, 0.893))
    # Tokens exist only after step 1, so they are named on its output.
    label(ax, cx + 0.025, (ys["tok"] + h + ys["emb"]) / 2,
          r"tokens $x_1, x_2, x_3$", rank="small", ha="left", va="center")

    # One block: a dashed frame around attention through the second add &
    # norm. Stacking it is step 11.
    fx0, fx1, fy0, fy1 = 0.055, 0.845, 0.335, 0.735
    ax.plot([fx0, fx1, fx1, fx0, fx0], [fy0, fy0, fy1, fy1, fy0],
            color=P.series[2], lw=THIN, ls=DASH, zorder=1)
    label(ax, fx0, fy1 + 0.02, r"one block, $\times N$ (Stack)",
          rank="small", ha="left", va="center")

    # Residual skips, on Manhattan routing: branch off the main flow, run down
    # a left-hand lane, enter the add & norm on its left edge.
    def skip(y_branch, y_target, lane):
        ax.plot([cx, lane], [y_branch] * 2, color=P.series[2], lw=THIN,
                ls=DASH, zorder=2)
        ax.plot([lane] * 2, [y_branch, y_target], color=P.series[2], lw=THIN,
                ls=DASH, zorder=2)
        arrow(ax, (lane, y_target), (left - 0.004, y_target),
              color=P.series[2], linestyle=DASH)
        ax.plot(cx, y_branch, "o", color=P.series[2], ms=3.5, zorder=5)

    skip(ys["attn"] - 0.022, ys["an1"] + h / 2, 0.085)
    skip(ys["ffn"] - 0.022, ys["an2"] + h / 2, 0.125)

    # Generation: the chosen word is appended and the whole model runs again.
    lane = 0.93
    top, bottom = 0.915, 0.025
    ax.plot([cx + 0.23, lane], [top] * 2, color=P.series[2], lw=THIN,
            ls=DASH, zorder=2)
    ax.plot([lane] * 2, [top, bottom + 0.008], color=P.series[2], lw=THIN,
            ls=DASH, zorder=2)
    arrow(ax, (lane, bottom + 0.008), (cx + 0.15, bottom + 0.008),
          color=P.series[2], linestyle=DASH)
    label(ax, lane + 0.025, 0.47, "Generate: append the token, run again",
          rank="small", ha="center", va="center", rotation=90)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.94)
    ax.axis("off")


# ------------------------------------------ generation as repeated prediction
@render("transformer-generation", size="wide")
def transformer_generation(ax):
    rows = [
        (["the", "cat", "sat"], "on"),
        (["the", "cat", "sat", "on"], "the"),
        (["the", "cat", "sat", "on", "the"], "mat"),
        (["the", "cat", "sat", "on", "the", "mat"], "<end>"),
    ]
    x0, w, gap, h = 0.02, 0.08, 0.02, 0.11
    out_x, out_w = 0.80, 0.18
    top = 0.80
    step = 0.235

    for j in range(len(rows[-1][0])):
        label(ax, x0 + j * (w + gap) + w / 2, top + h + 0.06, f"$x_{j + 1}$",
              rank="tick", ha="center")
    label(ax, 0.70, top + h + 0.06, "model", rank="small", ha="center")
    label(ax, out_x + out_w / 2, top + h + 0.06, "prediction", rank="small",
          ha="center")

    for i, (tokens, nxt) in enumerate(rows):
        y = top - i * step
        for j, tok in enumerate(tokens):
            # The token appended from the previous row is the one this row
            # adds, so it carries the heavier outline.
            block(ax, x0 + j * (w + gap), y, w, h, tok, emph=(i > 0 and
                  j == len(tokens) - 1))
        end = x0 + len(tokens) * (w + gap) - gap
        arrow(ax, (end + 0.015, y + h / 2), (out_x - 0.015, y + h / 2))
        block(ax, out_x, y, out_w, h, f"$x_{len(tokens) + 1}$ = {nxt}",
              emph=True)

        if i < len(rows) - 1:
            # The prediction becomes the last input token of the next row.
            nj = len(tokens)
            arrow(ax, (out_x + out_w / 2, y - 0.006),
                  (x0 + nj * (w + gap) + w / 2, y - step + h + 0.006),
                  color=P.series[2], linestyle=DASH, curve=-0.12)

    ax.set_xlim(0, 1)
    ax.set_ylim(top - (len(rows) - 1) * step - 0.03, top + h + 0.11)
    ax.axis("off")


# --------------------------- one block: tokens exchange, then each one thinks
@render("transformer-exchange-think", size="block")
def transformer_exchange_think(ax):
    xs = [0.12, 0.32, 0.52]
    names = [r"$x_1$ = the", r"$x_2$ = cat", r"$x_3$ = sat"]
    w, h = 0.13, 0.075
    y_in, y_mid, y_out = 0.14, 0.50, 0.84

    for x, name in zip(xs, names):
        label(ax, x, 0.05, name, rank="tick", ha="center")
        block(ax, x - w / 2, y_in, w, h, "vector")
        block(ax, x - w / 2, y_mid, w, h, "vector")
        block(ax, x - w / 2, y_out, w, h, "vector")
        # Feed-forward: each token on its own, straight up.
        block(ax, x - w / 2, 0.67, w, 0.07, "FFN", rank="small")
        arrow(ax, (x, y_mid + h), (x, 0.67 - 0.004))
        arrow(ax, (x, 0.74), (x, y_out - 0.004))

    # Attention: each token reads itself and every earlier token. The link
    # that matters in the running example, "sat" reading "cat", is heavier.
    for t in range(3):
        for s in range(t + 1):
            start = (xs[s], y_in + h)
            end = (xs[t] + (s - t) * 0.012, y_mid - 0.006)
            emph = (s, t) == (1, 2)
            curve = 0.0 if s == t else -0.18
            arrow(ax, start, end, emph=emph, curve=curve)

    # Bands and their meaning, on the right.
    for y0, y1 in ((0.235, 0.49), (0.585, 0.83)):
        ax.plot([0.63, 0.63], [y0, y1], color=P.series[2], lw=THIN, ls=DASH)
    label(ax, 0.66, 0.45, "Attention", rank="label", ha="left", va="center")
    label(ax, 0.66, 0.345, "tokens exchange information:\neach reads itself and the\ntokens before it",
          rank="small", ha="left", va="center")
    label(ax, 0.66, 0.74, "Feed-forward", rank="label", ha="left", va="center")
    label(ax, 0.66, 0.675, "each token works on what it\ngathered, on its own",
          rank="small", ha="left", va="center")

    # One block, repeated.
    label(ax, 0.32, 0.965, r"one block; stacking repeats it $N$ times",
          rank="small", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ----------------------------------- which earlier token decides the next one
@render("transformer-agreement", size="banner")
def transformer_agreement(ax):
    words = ["The", "students", "who", "failed", "the", "exam", "___"]
    x0, w, gap, h, y = 0.02, 0.118, 0.022, 0.2, 0.40
    cx = [x0 + j * (w + gap) + w / 2 for j in range(len(words))]

    for j, word in enumerate(words):
        block(ax, cx[j] - w / 2, y, w, h, word, emph=j in (1, 6))
        label(ax, cx[j], y - 0.1, f"$x_{j + 1}$", rank="tick", ha="center")

    # The token that decides the answer: an arc over the top.
    arrow(ax, (cx[6], y + h + 0.01), (cx[1], y + h + 0.01), emph=True,
          curve=0.28)
    label(ax, (cx[1] + cx[6]) / 2, y + h + 0.33, "subject, plural: “were”",
          rank="tick", ha="center")

    # The nearest noun, which points the wrong way: a short dashed arc below.
    arrow(ax, (cx[6] - 0.02, y - 0.16), (cx[5] + 0.02, y - 0.16),
          color=P.series[2], linestyle=DASH, curve=-0.6)
    label(ax, (cx[5] + cx[6]) / 2, y - 0.33, "nearest noun, singular: “was”",
          rank="small", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

# ------------------------------------------------ 2. what the table is
@render("transformer-embed-table", size="wide")
def transformer_embed_table(ax):
    from matplotlib.patches import Rectangle

    vocab = ["<end>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
    table = [
        [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
        [-1, -1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0],
    ]
    cw, ch, ex, pick = 0.62, 0.5, 2.6, 2
    top = len(vocab) * ch

    for i, (tok, row) in enumerate(zip(vocab, table)):
        y, on = (len(vocab) - 1 - i) * ch, i == pick
        label(ax, ex - 1.5, y + ch / 2, str(i), rank="tick", ha="center",
              va="center", color=P.ink if on else P.mute)
        label(ax, ex - 0.2, y + ch / 2, tok, rank="tick", ha="right",
              va="center", color=P.ink if on else P.mute)
        for j, v in enumerate(row):
            ax.add_patch(Rectangle(
                (ex + j * cw, y), cw, ch, facecolor="none",
                edgecolor=P.ink if on else P.series[2],
                lw=LINE if on else HAIR, zorder=3 if on else 2))
            label(ax, ex + (j + 0.5) * cw, y + ch / 2,
                  str(v).replace("-", "−"), rank="tick", ha="center",
                  va="center", color=P.ink if on else P.mute)

    label(ax, ex - 1.5, top + 0.3, "ID", rank="small", ha="center")
    label(ax, ex - 0.2, top + 0.3, "token", rank="small", ha="right")
    for j in range(4):
        label(ax, ex + (j + 0.5) * cw, top + 0.3, str(j + 1), rank="small",
              ha="center")
    label(ax, ex + 2 * cw, top + 0.8, r"$d = 4$ columns, one per dimension",
          rank="tick", ha="center")
    label(ax, 0.3, top / 2, "$V = 8$ rows,\none per token", rank="tick",
          ha="center", va="center", rotation=90)

    # The row that is one token's embedding.
    y = (len(vocab) - 1 - pick) * ch + ch / 2
    arrow(ax, (ex + 4 * cw + 0.1, y), (ex + 4 * cw + 1.0, y), emph=True)
    label(ax, ex + 4 * cw + 1.15, y + 0.22, "the embedding of “cat”",
          rank="label", ha="left", va="center")
    label(ax, ex + 4 * cw + 1.15, y - 0.25, "(0, 1, 1, 0)", rank="tick",
          ha="left", va="center")

    ax.set_xlim(0, ex + 4 * cw + 4.6)
    ax.set_ylim(-0.2, top + 1.1)
    ax.axis("off")


# ------------------------------------- 2. rows before and after training
@render("transformer-embed-training", size="wide", axes=False)
def transformer_embed_training(fig):
    axes = fig.subplots(1, 2)
    before = {"cat": (0.2, 0.75), "dog": (0.8, 0.2), "sat": (0.55, 0.85),
              "ran": (0.15, 0.25), "the": (0.75, 0.6), "on": (0.45, 0.45)}
    after = {"cat": (0.2, 0.78), "dog": (0.3, 0.68), "sat": (0.75, 0.75),
             "ran": (0.84, 0.66), "the": (0.25, 0.2), "on": (0.75, 0.22)}
    notes = ("random rows: no two\ntokens are similar",
             "tokens used alike\nhave moved together")
    for ax, pts, title, note in zip(axes, (before, after),
                                    ("Before training", "After training"),
                                    notes):
        for name, (x, y) in pts.items():
            ax.plot(x, y, "o", color=P.ink, ms=4.5, zorder=5)
            label(ax, x, y + 0.075, name, rank="tick", ha="center",
                  va="center", color=P.ink)
        label(ax, 0.5, 1.04, title, rank="label", ha="center", va="bottom",
              transform=ax.transAxes)
        label(ax, 0.5, -0.06, note, rank="small", ha="center", va="top",
              transform=ax.transAxes)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=P.series[2], lw=HAIR)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.axis("off")


# ------------------------------------------- 2. the embedding table as a lookup
@render("transformer-embed-lookup", size="block")
def transformer_embed_lookup(ax):
    from matplotlib.patches import Rectangle

    vocab = ["<end>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
    table = [
        [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
        [-1, -1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0],
    ]
    used = (1, 2, 3)
    cw, ch = 0.62, 0.5
    ex, xx = 3.9, 8.6             # left edge of the cells of E and of X

    def row_y(i):                 # row 0 on top
        return (len(vocab) - 1 - i) * ch

    def cells(x0, y, values, on):
        for j, v in enumerate(values):
            ax.add_patch(Rectangle(
                (x0 + j * cw, y), cw, ch, facecolor="none",
                edgecolor=P.ink if on else P.series[2],
                lw=LINE if on else HAIR, zorder=3 if on else 2))
            label(ax, x0 + (j + 0.5) * cw, y + ch / 2,
                  str(v).replace("-", "−"), rank="tick", ha="center",
                  va="center", color=P.ink if on else P.mute)

    for i, (tok, row) in enumerate(zip(vocab, table)):
        y, on = row_y(i), i in used
        label(ax, ex - 1.35, y + ch / 2, str(i), rank="tick", ha="center",
              va="center", color=P.ink if on else P.mute)
        label(ax, ex - 0.2, y + ch / 2, tok, rank="tick", ha="right",
              va="center", color=P.ink if on else P.mute)
        cells(ex, y, row, on)
        if on:
            # The input token and its ID on the left, its row of X on the right.
            block(ax, 0.1, y + 0.06, 1.25, ch - 0.12, f"{tok}, ID {i}")
            arrow(ax, (1.42, y + ch / 2), (ex - 1.6, y + ch / 2))
            arrow(ax, (ex + 4 * cw + 0.08, y + ch / 2),
                  (xx - 0.08, y + ch / 2), emph=True)
            cells(xx, y, row, True)

    top = len(vocab) * ch
    label(ax, 0.72, row_y(1) + ch + 0.3, "input tokens", rank="small",
          ha="center")
    label(ax, ex - 1.35, top + 0.3, "ID", rank="small", ha="center")
    for j in range(4):
        label(ax, ex + (j + 0.5) * cw, top + 0.3, str(j + 1), rank="small",
              ha="center")
        label(ax, xx + (j + 0.5) * cw, row_y(1) + ch + 0.3, str(j + 1),
              rank="small", ha="center")
    label(ax, ex + 2 * cw, top + 0.75, "dimension", rank="small", ha="center")
    label(ax, xx + 2 * cw, row_y(1) + ch + 0.75, "dimension", rank="small",
          ha="center")
    label(ax, ex + 2 * cw, -0.45, r"table $E$: $V = 8$ rows, $d = 4$ columns",
          rank="label", ha="center")
    label(ax, xx + 2 * cw, row_y(3) - 0.45, r"$X$: $T = 3$ rows",
          rank="label", ha="center")

    ax.set_xlim(0, xx + 4 * cw + 0.1)
    ax.set_ylim(-0.8, top + 1.0)
    ax.axis("off")


# --------------------------- 2. what the rows of a trained table look like
@render("transformer-embed-geometry", size="panel", axes=False)
def transformer_embed_geometry(fig):
    import numpy as np

    axes = fig.subplots(1, 3)

    def vec(ax, deg, r, name, emph=False, dx=0.0, dy=0.0):
        t = np.radians(deg)
        x, y = r * np.cos(t), r * np.sin(t)
        arrow(ax, (0, 0), (x, y), emph=emph, shrinkA=0, shrinkB=0)
        label(ax, x * 1.13 + dx, y * 1.13 + dy, name, rank="tick",
              ha="center", va="center", color=P.ink)

    def arc(ax, a0, a1, r, text):
        ts = np.radians(np.linspace(a0, a1, 40))
        ax.plot(r * np.cos(ts), r * np.sin(ts), color=P.series[2], lw=THIN)
        tm = np.radians((a0 + a1) / 2)
        label(ax, (r + 0.2) * np.cos(tm), (r + 0.2) * np.sin(tm), text,
              rank="small", ha="center", va="center")

    def frame(ax, title, note):
        label(ax, 0.5, 1.02, title, rank="label", ha="center", va="bottom",
              transform=ax.transAxes)
        label(ax, 0.5, -0.04, note, rank="small", ha="center", va="top",
              transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.axis("off")

    # One-hot: every pair of tokens at a right angle.
    ax = axes[0]
    vec(ax, 0, 1, "cat", emph=True, dx=0.08)
    vec(ax, 90, 1, "dog", emph=True)
    arc(ax, 0, 90, 0.3, "90°")
    ax.set_xlim(-0.35, 1.5)
    ax.set_ylim(-0.35, 1.5)
    frame(ax, "One-hot", "cosine similarity 0\nfor every pair")

    # Learned: tokens used alike point alike.
    ax = axes[1]
    vec(ax, 62, 1, "cat", emph=True)
    vec(ax, 40, 1, "dog", emph=True, dx=0.05)
    vec(ax, 150, 1, "the")
    arc(ax, 40, 62, 0.45, "")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.5, 1.8)
    frame(ax, "Learned", "cat and dog close:\ncosine similarity near 1")

    # Differences between rows.
    ax = axes[2]
    pts = {"man": (0.2, 0.2), "woman": (0.55, 1.0),
           "king": (1.25, 0.45), "queen": (1.6, 1.25)}
    for name, (x, y) in pts.items():
        ax.plot(x, y, "o", color=P.ink, ms=4, zorder=5)
        up = name in ("woman", "queen")
        label(ax, x, y + (0.16 if up else -0.16), name, rank="tick",
              ha="center", va="center", color=P.ink)
    arrow(ax, pts["man"], pts["woman"], emph=True)
    arrow(ax, pts["king"], pts["queen"], emph=True)
    arrow(ax, pts["man"], pts["king"], color=P.series[2], linestyle=DASH)
    arrow(ax, pts["woman"], pts["queen"], color=P.series[2], linestyle=DASH)
    ax.set_xlim(-0.15, 1.95)
    ax.set_ylim(-0.4, 1.7)
    frame(ax, "Differences", "man to woman matches\nking to queen")

# ------------------------------------- 3. the fast and the slow pair of waves
@render("transformer-position-waves", size="panel", axes=False)
def transformer_position_waves(fig):
    import numpy as np

    axes = fig.subplots(1, 2)
    panels = (
        (12, 1.0, "Dim 1, the fast pair: $\\sin p$", [0, 1, 2]),
        (700, 0.01, "Dim 3, the slow pair: $\\sin(p/100)$", []),
    )
    for ax, (end, w, title, marks) in zip(axes, panels):
        ps = np.linspace(0, end, 600)
        ax.plot(ps, np.sin(w * ps), **P.line(1))
        whole = np.arange(0, end + 1) if end <= 20 else np.arange(0, end + 1, 50)
        ax.plot(whole, np.sin(w * whole), "o", color=P.series[1], ms=3.5)
        for p_, name in zip(marks, ("the", "cat", "sat")):
            y = np.sin(w * p_)
            ax.plot(p_, y, "o", color=P.ink, ms=6, zorder=5)
            left = name == "cat"
            label(ax, p_ + (-0.4 if left else 0.4), y + (0.0 if left else -0.14),
                  name, rank="tick", ha="right" if left else "left",
                  va="center", color=P.ink)
        ax.axhline(0, **P.grid())
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("position $p$")
        ax.set_title(title, fontsize=10.5, color=P.ink)
        clean(ax, spines="lb")
        if end <= 20:
            ax.set_xticks(range(0, end + 1, 2))

# ------------------- 3. similarity of two positions against their distance
@render("transformer-position-similarity", size="wide")
def transformer_position_similarity(ax):
    import numpy as np

    gaps = np.linspace(0, 60, 601)
    for d, rank in ((4, 1), (512, 0)):
        w = 1 / 10000 ** (2 * np.arange(d // 2) / d)
        sim = np.cos(np.outer(gaps, w)).mean(axis=1)
        ax.plot(gaps, sim, **(P.emph() if rank == 0 else P.line(1)))
    whole = np.arange(0, 61)
    w4 = np.array([1, 0.01])
    ax.plot(whole, np.cos(np.outer(whole, w4)).mean(axis=1), "o",
            color=P.series[1], ms=2.5)
    label(ax, 45, 0.36, "width 512, 256 pairs", rank="tick", ha="center",
          color=P.ink)
    label(ax, 6.3, 1.06, "width 4, 2 pairs: distance 6 looks like 0",
          rank="tick", ha="left", color=P.series[1])
    ax.axhline(0, **P.grid())
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel("distance between the two positions")
    ax.set_ylabel("similarity")
    clean(ax, spines="lb")

# --------------------------- 3. two waves repeat when both line up at once
@render("transformer-position-lineup", size="wide", axes=False)
def transformer_position_lineup(fig):
    import numpy as np

    axes = fig.subplots(2, 1, sharex=True)
    ps = np.linspace(0, 25, 800)
    for ax, period in zip(axes, (4, 6)):
        ax.plot(ps, np.cos(2 * np.pi * ps / period), **P.line(1))
        starts = np.arange(0, 25, period)
        both = [x for x in starts if x % 12 == 0]
        ax.plot(starts, np.ones_like(starts), "o", color=P.series[1], ms=4,
                zorder=4)
        ax.plot(both, np.ones(len(both)), "o", color=P.ink, ms=7, zorder=5)
        for x in (12, 24):
            ax.axvline(x, **P.guide())
        label(ax, -0.4, 0, f"period {period}", rank="tick", ha="right",
              va="center", color=P.ink)
        ax.set_ylim(-1.3, 1.5)
        ax.set_xlim(-0.3, 25)
        clean(ax)
    label(axes[0], 12, 1.75, "both back at the start", rank="small",
          ha="center")
    axes[1].set_xticks(range(0, 25, 2))
    for ax in axes:
        ax.tick_params(axis="x", length=0)
    axes[1].set_xlabel("position $p$")

# ------------------------------ 4. one token vector becomes query, key, value
@render("transformer-qkv", size="wide")
def transformer_qkv(ax):
    h = [0.909, -0.416, 1.020, 2.000]
    outs = [("query", "$W_Q$", [1.020, 2.000, -0.416, 2.000], 3.6),
            ("key", "$W_K$", [-0.416, 1.020, -0.416, 2.000], 2.0),
            ("value", "$W_V$", [0.909, -0.416, 1.020, 2.000], 0.4)]
    cw, ch, gx = 1.3, 0.55, 9.3
    cells(ax, 0.2, 2.0, [h], cw, ch)
    label(ax, 0.2 + 2 * cw, 2.9, "vector of “sat”, from part 1", rank="tick",
          ha="center")
    for name, weights, vals, y in outs:
        arrow(ax, (0.3 + 4 * cw, 2.0 + ch / 2), (gx - 1.7, y + ch / 2))
        label(ax, gx - 1.45, y + ch / 2, weights, rank="tick", ha="left",
              va="center", color=P.ink)
        cells(ax, gx, y, [vals[:2]], cw, ch)
        cells(ax, gx + 2 * cw + 0.3, y, [vals[2:]], cw, ch)
        label(ax, gx + 4 * cw + 0.55, y + ch / 2, name, rank="label",
              ha="left", va="center")
    label(ax, gx + cw, 4.45, "head 1", rank="small", ha="center")
    label(ax, gx + 3 * cw + 0.3, 4.45, "head 2", rank="small", ha="center")
    ax.set_xlim(0, gx + 4 * cw + 1.9)
    ax.set_ylim(0.1, 4.8)
    ax.axis("off")


# ----------------------------- 5. one query scored against every key
@render("transformer-scores", size="wide")
def transformer_scores(ax):
    keys = [("the", "(1.000, 0.000)", "0.721", 3.3, False),
            ("cat", "(1.540, 1.010)", "2.539", 1.9, True),
            ("sat", "(−0.416, 1.020)", "1.142", 0.5, False)]
    block(ax, 0.2, 1.75, 3.0, 0.9, "query of “sat”\n(1.020, 2.000)", emph=True)
    for tok, vec, score, y, top in keys:
        block(ax, 7.0, y - 0.15, 3.0, 0.9, f"key of “{tok}”\n{vec}", emph=top)
        arrow(ax, (3.3, 2.2), (6.9, y + 0.3), emph=top)
        label(ax, 5.0, {3.3: 3.45, 1.9: 2.45, 0.5: 0.95}[y], f"score {score}",
              rank="tick", ha="center", va="center",
              color=P.ink if top else P.body)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.1, 4.4)
    ax.axis("off")


# ------------------------------------- 6. scores, masked scores, weights
@render("transformer-mask-softmax", size="banner")
def transformer_mask_softmax(ax):
    toks = ["the", "cat", "sat"]
    scores = [[0.000, 0.714, 0.721], [0.714, 1.814, 0.424],
              [0.721, 2.539, 1.142]]
    masked = [[0.000, "hidden", "hidden"], [0.714, 1.814, "hidden"],
              [0.721, 2.539, 1.142]]
    weights = [[1.000, 0, 0], [0.250, 0.750, 0], [0.115, 0.709, 0.175]]
    upper = {(0, 1), (0, 2), (1, 2)}
    cw, ch = 1.0, 0.55
    xs = (0.9, 6.0, 11.1)
    titles = ("scores", "after the mask", "weights after softmax")
    for x0, rows, title, mute in zip(xs, (scores, masked, weights), titles,
                                     (set(), upper, upper)):
        cells(ax, x0, 0, rows, cw, ch, muted=mute)
        for i, tok in enumerate(toks):
            label(ax, x0 - 0.12, (2 - i + 0.5) * ch, tok, rank="small",
                  ha="right", va="center")
            label(ax, x0 + (i + 0.5) * cw, 3 * ch + 0.2, tok, rank="small",
                  ha="center")
        label(ax, x0 + 1.5 * cw, 3 * ch + 0.65, title, rank="tick",
              ha="center", color=P.ink)
    for x0, text in ((xs[0], "mask"), (xs[1], "softmax")):
        arrow(ax, (x0 + 3 * cw + 0.2, 1.5 * ch), (x0 + 3 * cw + 1.15, 1.5 * ch),
              emph=True)
        label(ax, x0 + 3 * cw + 0.67, 1.5 * ch + 0.32, text, rank="small",
              ha="center")
    ax.set_xlim(0, xs[2] + 3 * cw + 0.1)
    ax.set_ylim(-0.1, 3 * ch + 0.95)
    ax.axis("off")


# ------------------------------- 7. the weighted sum of three value vectors
@render("transformer-weighted-sum", size="wide")
def transformer_weighted_sum(ax):
    values = [("the", 0.115, (1.000, 1.000), (0.08, 0.02)),
              ("cat", 0.709, (0.841, 1.540), (-0.05, 0.1)),
              ("sat", 0.175, (0.909, -0.416), (0.1, -0.08))]
    for tok, wt, (x, y), (dx, dy) in values:
        arrow(ax, (0, 0), (x, y), shrinkA=0, shrinkB=0)
        label(ax, x + dx, y + dy, f"value of “{tok}”, weight {wt:.3f}",
              rank="small", ha="left", va="center")
    arrow(ax, (0, 0), (0.872, 1.135), emph=True, shrinkA=0, shrinkB=0)
    label(ax, 0.3, 1.22, "output for “sat”\n(0.872, 1.135)", rank="tick",
          ha="right", va="center", color=P.ink)
    ax.axhline(0, **P.grid())
    ax.axvline(0, **P.grid())
    ax.set_xlim(-0.6, 2.5)
    ax.set_ylim(-0.6, 1.8)
    ax.set_aspect("equal")
    clean(ax, spines="lb")
    ax.set_xlabel("first number")
    ax.set_ylabel("second number")


# --------------------------------- 8. two heads, two sets of weights
@render("transformer-multi-head", size="wide")
def transformer_multi_head(ax):
    heads = [("head 1", [0.115, 0.709, 0.175], [0.871, 1.135], 2.6),
             ("head 2", [0.124, 0.105, 0.771], [0.893, 1.771], 0.6)]
    toks = ["the", "cat", "sat"]
    w, h, cw = 1.25, 0.95, 1.0
    for name, wts, out, y in heads:
        label(ax, 0.0, y + h / 2, name, rank="tick", ha="left", va="center",
              color=P.ink)
        for j, (tok, wt) in enumerate(zip(toks, wts)):
            block(ax, 1.7 + j * (w + 0.2), y, w, h, f"{tok}\n{wt:.3f}",
                  emph=wt == max(wts))
        arrow(ax, (1.7 + 3 * (w + 0.2), y + h / 2), (6.9, y + h / 2))
        cells(ax, 7.0, y + 0.2, [out], 1.15, 0.55)
    label(ax, 1.7 + 1.5 * (w + 0.2) - 0.1, 3.85, "weights of “sat”",
          rank="small", ha="center")
    label(ax, 8.15, 3.85, "head output", rank="small", ha="center")
    # Both outputs, side by side, are the token's row of M.
    cells(ax, 10.6, 1.8, [[0.871, 1.135, 0.893, 1.771]], 1.15, 0.55)
    arrow(ax, (9.4, 3.05), (10.5, 2.25))
    arrow(ax, (9.4, 1.1), (10.5, 1.9))
    label(ax, 12.9, 2.75, "row of “sat” in $M$", rank="tick", ha="center",
          color=P.ink)
    ax.set_xlim(0, 15.3)
    ax.set_ylim(0.3, 4.1)
    ax.axis("off")


# ------------------------ 9. the skip path around attention, then LayerNorm
@render("transformer-add-norm", size="banner")
def transformer_add_norm(ax):
    w, h, y = 2.3, 0.8, 1.6
    xs = {"h": 0.2, "attn": 3.4, "add": 6.9, "norm": 8.7, "y": 12.0}
    block(ax, xs["h"], y, w, h, "$H$")
    block(ax, xs["attn"], y, w, h, "attention")
    block(ax, xs["add"], y, 0.9, h, "+", rank="label")
    block(ax, xs["norm"], y, w, h, "LayerNorm", emph=True)
    block(ax, xs["y"], y, w, h, "$Y$")
    arrow(ax, (xs["h"] + w, y + h / 2), (xs["attn"], y + h / 2))
    arrow(ax, (xs["attn"] + w, y + h / 2), (xs["add"], y + h / 2))
    arrow(ax, (xs["add"] + 0.9, y + h / 2), (xs["norm"], y + h / 2))
    arrow(ax, (xs["norm"] + w, y + h / 2), (xs["y"], y + h / 2))
    label(ax, (xs["attn"] + w + xs["add"]) / 2, y + h / 2 + 0.25, "$M$",
          rank="tick", ha="center", color=P.ink)
    # The residual connection: the input goes round attention unchanged.
    top = y + h + 0.75
    x_from, x_to = xs["h"] + w / 2, xs["add"] + 0.45
    ax.plot([x_from, x_from, x_to], [y + h + 0.05, top, top],
            color=P.series[2], lw=THIN, ls=DASH)
    arrow(ax, (x_to, top), (x_to, y + h + 0.03), color=P.series[2],
          linestyle=DASH)
    label(ax, (x_from + x_to) / 2, top + 0.25,
          "residual connection: the input, unchanged", rank="small",
          ha="center")
    for x, text in ((xs["h"] + w / 2, "0.909, −0.416,\n1.020, 2.000"),
                    (xs["add"] + 0.45, "1.781, 0.719,\n1.913, 3.771"),
                    (xs["y"] + w / 2, "−0.241, −1.208,\n−0.121, 1.571")):
        label(ax, x, y - 0.55, text, rank="small", ha="center", va="center")
    label(ax, xs["attn"] + w / 2, y - 0.55, "0.872, 1.135,\n0.893, 1.771",
          rank="small", ha="center", va="center")
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0.6, 3.6)
    ax.axis("off")


# ----------------------------- 10. expand, switch off negatives, shrink
@render("transformer-feed-forward", size="wide")
def transformer_feed_forward(ax):
    cw, ch = 1.05, 0.5
    inp = [-0.241, -1.208, -0.121, 1.571]
    hid = [-1.812, -1.208, -0.121, 1.812, 1.208, -0.120, -2.779, -0.121]
    relu = [0, 0, 0, 1.812, 1.208, 0, 0, 0]
    out = [-0.604, 0.000, 0.000, 0.906]
    off = {(0, j) for j, v in enumerate(relu) if v == 0}
    ys = [4.5, 3.0, 1.5, 0.0]
    x8, x4 = 2.6, 2.6 + 2 * cw
    cells(ax, x4, ys[0], [inp], cw, ch)
    cells(ax, x8, ys[1], [hid], cw, ch)
    cells(ax, x8, ys[2], [relu], cw, ch, muted=off)
    cells(ax, x4, ys[3], [out], cw, ch)
    mid = x8 + 4 * cw
    for a, b in zip(ys, ys[1:]):
        arrow(ax, (mid, a - 0.05), (mid, b + ch + 0.05))
    notes = ["“sat” after add and norm", "expand: 4 numbers to 8",
             "ReLU: negatives to 0", "shrink: 8 numbers to 4"]
    for y, note in zip(ys, notes):
        label(ax, 2.35, y + ch / 2, note, rank="small", ha="right",
              va="center")
    ax.set_xlim(-1.6, x8 + 8 * cw + 0.1)
    ax.set_ylim(-0.15, 5.15)
    ax.axis("off")


# ----------------------------------- 12. probabilities for the next token
@render("transformer-predict", size="wide")
def transformer_predict(ax):
    vocab = ["<end>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
    probs = [0.006, 0.016, 0.011, 0.140, 0.741, 0.007, 0.066, 0.014]
    for i, (tok, p) in enumerate(zip(vocab, probs)):
        top = tok == "on"
        ax.bar(i, p, width=0.62, color=P.ink if top else P.series[2])
        label(ax, i, p + 0.03, f"{p:.3f}", rank="small", ha="center",
              color=P.ink if top else P.mute)
    ax.set_xticks(range(len(vocab)))
    ax.set_xticklabels(vocab)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("probability of coming next")
    clean(ax, spines="lb")
    ax.tick_params(axis="x", length=0, labelsize=9.5)


# ------------------------------- 13. the same logits at three temperatures
@render("transformer-temperature", size="panel", axes=False)
def transformer_temperature(fig):
    toks = ["on", "sat", "dog", "the", "ran", "cat", "mat", "<end>"]
    rows = [("Temperature 0.5", [0.957, 0.034, 0.008, 0, 0, 0, 0, 0]),
            ("Temperature 1", [0.741, 0.140, 0.066, 0.016, 0.014, 0.011,
                               0.007, 0.006]),
            ("Temperature 2", [0.431, 0.187, 0.128, 0.063, 0.059, 0.052,
                               0.041, 0.038])]
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (title, probs) in zip(axes, rows):
        ax.bar(range(len(toks)), probs, width=0.66,
               color=[P.ink] + [P.series[2]] * (len(toks) - 1))
        label(ax, 0.55, probs[0], f"“on” {probs[0]:.3f}", rank="small",
              ha="left", va="center", color=P.ink)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=90)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10.5, color=P.ink)
        clean(ax, spines="lb")
        ax.tick_params(axis="x", length=0, labelsize=8.5)
    axes[0].set_ylabel("probability")

# =================================================== Efficient Transformers

# ------------------------------- 1. the score table after one more token
@render("efficient-prefill-decode", size="wide")
def efficient_prefill_decode(ax):
    toks = ["the", "cat", "sat", "on"]
    rows = [[0.000, "", "", ""], [0.714, 1.814, "", ""],
            [0.721, 2.539, 1.142, ""], [0.021, 1.461, 1.433, 0.000]]
    cw, ch, x0 = 1.15, 0.55, 1.2
    upper = {(i, j) for i in range(4) for j in range(4) if j > i}
    old = {(i, j) for i in range(3) for j in range(4)}
    cells(ax, x0, 0, rows, cw, ch, muted=upper | old)
    cells(ax, x0, 0, [rows[3]], cw, ch)
    for i, tok in enumerate(toks):
        label(ax, x0 - 0.15, (3 - i + 0.5) * ch, tok, rank="tick", ha="right",
              va="center", color=P.ink if i == 3 else P.mute)
        label(ax, x0 + (i + 0.5) * cw, 4 * ch + 0.2, tok, rank="small",
              ha="center")
    right = x0 + 4 * cw + 0.3
    ax.plot([right, right], [ch + 0.05, 4 * ch], color=P.series[2], lw=THIN)
    label(ax, right + 0.2, 2.5 * ch, "prefill: computed once,\nnever changes",
          rank="small", ha="left", va="center")
    label(ax, right + 0.2, 0.5 * ch, "decode: the only new row", rank="tick",
          ha="left", va="center", color=P.ink)
    ax.set_xlim(0, right + 4.3)
    ax.set_ylim(-0.1, 4 * ch + 0.5)
    ax.axis("off")


# ------------------------------------ 2. the cache grows by one row
@render("efficient-kv-cache", size="wide")
def efficient_kv_cache(ax):
    toks = ["the", "cat", "sat", "on"]
    keys = [[1.000, 0.000], [1.540, 1.010], [-0.416, 1.020], [-1.990, 0.030]]
    vals = [[1.000, 1.000], [0.841, 1.540], [0.909, -0.416], [-0.859, -1.990]]
    cw, ch, kx, vx = 1.2, 0.55, 5.6, 8.6
    cells(ax, kx, ch, keys[:3], cw, ch)
    cells(ax, vx, ch, vals[:3], cw, ch)
    cells(ax, kx, 0, [keys[3]], cw, ch, muted={(0, 0), (0, 1)})
    cells(ax, vx, 0, [vals[3]], cw, ch, muted={(0, 0), (0, 1)})
    for i, tok in enumerate(toks):
        label(ax, kx - 0.15, (3 - i + 0.5) * ch, tok, rank="small",
              ha="right", va="center")
    label(ax, kx + cw, 4 * ch + 0.2, "cached keys", rank="tick", ha="center",
          color=P.ink)
    label(ax, vx + cw, 4 * ch + 0.2, "cached values", rank="tick",
          ha="center", color=P.ink)
    label(ax, vx + 2 * cw + 0.2, 0.5 * ch, "added by\nthis run", rank="small",
          ha="left", va="center")
    block(ax, 0.1, 1.5 * ch, 2.9, 1.1, "query of “on”\n(0.030, 2.000)",
          emph=True)
    for i in range(4):
        arrow(ax, (3.1, 1.5 * ch + 0.55), (kx - 1.0, (3 - i + 0.5) * ch))
    label(ax, 1.55, 1.5 * ch - 0.3, "computed, used once,\nnot stored",
          rank="small", ha="center", va="top")
    ax.set_xlim(0, vx + 2 * cw + 1.6)
    ax.set_ylim(-0.3, 4 * ch + 0.5)
    ax.axis("off")


# ------------------------ 3. how many sets of keys and values are stored
@render("efficient-sharing", size="panel")
def efficient_sharing(ax):
    panels = [("Multi-head", [[0], [1], [2], [3]], "4 sets stored"),
              ("Grouped-query, 2 groups", [[0, 1], [2, 3]], "2 sets stored"),
              ("Multi-query", [[0, 1, 2, 3]], "1 set stored")]
    w, h, gap, span = 0.8, 0.5, 0.25, 5.2
    for k, (title, groups, note) in enumerate(panels):
        x0 = k * span
        qx = [x0 + j * (w + gap) for j in range(4)]
        for j in range(4):
            block(ax, qx[j], 2.2, w, h, f"q{j + 1}", rank="small")
        for g in groups:
            cx = sum(qx[j] + w / 2 for j in g) / len(g)
            block(ax, cx - 0.4, 0.5, 0.8, h, "K, V", rank="small", emph=True)
            for j in g:
                arrow(ax, (qx[j] + w / 2, 2.2), (cx, 0.5 + h + 0.03))
        mid = x0 + 2 * w + 1.5 * gap
        label(ax, mid, 3.15, title, rank="tick", ha="center", color=P.ink)
        label(ax, mid, 0.05, note, rank="small", ha="center")
    label(ax, -0.25, 2.45, "queries", rank="small", ha="right", va="center")
    ax.set_xlim(-1.5, 2 * span + 4.2)
    ax.set_ylim(-0.2, 3.5)
    ax.axis("off")


# ----------------- 4. the row of "sat" processed one block of keys at a time
@render("efficient-flash", size="banner")
def efficient_flash(ax):
    w, h, y = 3.3, 1.7, 0.9
    steps = [("block 1: “the”", "largest 0.721\ntotal 1.000\nsum (1.000, 1.000)"),
             ("block 2: “cat”, “sat”", "largest 2.539\ntotal 1.409\nsum (1.228, 1.599)"),
             ("divide sum by total", "output for “sat”\n(0.872, 1.135)")]
    xs = [0.2, 5.0, 9.8]
    for i, (x, (title, body)) in enumerate(zip(xs, steps)):
        block(ax, x, y, w, h, body, emph=i == 2, rank="small")
        label(ax, x + w / 2, y + h + 0.25, title, rank="tick", ha="center",
              color=P.ink)
        if i:
            arrow(ax, (xs[i - 1] + w + 0.05, y + h / 2), (x - 0.05, y + h / 2))
    label(ax, (xs[0] + w + xs[1]) / 2, y + h / 2 + 0.2,
          "shrink by\n0.162", rank="small", ha="center", va="bottom")
    label(ax, xs[0], y - 0.45,
          "three running numbers per query, and the full row of scores is never stored",
          rank="small", ha="left")
    ax.set_xlim(0, 13.3)
    ax.set_ylim(0.2, 3.3)
    ax.axis("off")


# ------------------------ 5. softmax weights against linear weights
@render("efficient-linear", size="wide")
def efficient_linear(ax):
    toks = ["the", "cat", "sat"]
    soft, lin = [0.115, 0.709, 0.175], [0.275, 0.436, 0.289]
    for i in range(3):
        ax.bar(i - 0.19, soft[i], width=0.34, color=P.ink)
        ax.bar(i + 0.19, lin[i], width=0.34, color=P.series[2])
        label(ax, i - 0.19, soft[i] + 0.025, f"{soft[i]:.3f}", rank="small",
              ha="center", color=P.ink)
        label(ax, i + 0.19, lin[i] + 0.025, f"{lin[i]:.3f}", rank="small",
              ha="center")
    label(ax, 1.75, 0.66, "softmax attention", rank="tick", ha="left",
          color=P.ink)
    label(ax, 1.75, 0.58, "linear attention", rank="tick", ha="left",
          color=P.mute)
    ax.set_xticks(range(3))
    ax.set_xticklabels(toks)
    ax.set_xlim(-0.6, 3.1)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("weight of “sat” on each token")
    clean(ax, spines="lb")
    ax.tick_params(axis="x", length=0)

# ============================================== The problem, step by step
#
# One figure per "The problem" section. Each shows what the step is given and
# what is still missing, never the method that fills the gap.

# ------------------------------------------- 1. text has to become IDs
@render("transformer-problem-tokenize", size="banner")
def transformer_problem_tokenize(ax):
    rows = [("“the cat sat”", [1, 2, 3], "a short list of IDs", 1.9),
            ("“lowest”", ["?", "?"], "never seen in training", 0.4)]
    for text, ids, note, y in rows:
        block(ax, 0.2, y, 3.0, 0.8, text)
        arrow(ax, (3.2, y + 0.4), (4.4, y + 0.4))
        block(ax, 4.4, y, 2.2, 0.8, "tokenizer", emph=True)
        arrow(ax, (6.6, y + 0.4), (7.8, y + 0.4))
        cells(ax, 7.9, y + 0.1, [ids], 0.8, 0.6, digits=0)
        label(ax, 7.9 + len(ids) * 0.8 + 0.3, y + 0.4, note, rank="small",
              ha="left", va="center")
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0.1, 3.0)
    ax.axis("off")


# --------------------------------- 2. IDs sit on a line that means nothing
@render("transformer-problem-embed", size="banner")
def transformer_problem_embed(ax):
    vocab = ["<end>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
    ax.plot([-0.4, 7.4], [0, 0], color=P.series[1], lw=THIN)
    for i, tok in enumerate(vocab):
        hot = tok in ("cat", "sat", "dog")
        ax.plot(i, 0, "o", color=P.ink if hot else P.series[1],
                ms=7 if hot else 4.5, zorder=5)
        label(ax, i, -0.42, tok, rank="tick", ha="center",
              color=P.ink if hot else P.mute)
        label(ax, i, -0.85, f"ID {i}", rank="small", ha="center")
    arrow(ax, (2.05, 0.12), (5.95, 0.12), curve=-0.42, color=P.series[1],
          linestyle=DASH)
    label(ax, 4.0, 1.25, "cat and dog: used alike, 4 apart", rank="tick",
          ha="center", color=P.ink)
    arrow(ax, (2.03, 0.1), (2.97, 0.1), curve=-0.9)
    label(ax, 1.1, 0.75, "cat and sat:\nunrelated, 1 apart", rank="small",
          ha="center", va="center")
    ax.set_xlim(-0.8, 7.8)
    ax.set_ylim(-1.1, 1.6)
    ax.axis("off")


# ------------------------- 3. two sentences made of the same three vectors
@render("transformer-problem-position", size="banner")
def transformer_problem_position(ax):
    top, bottom = ["dog", "bites", "man"], ["man", "bites", "dog"]
    w, h, gap = 1.7, 0.7, 0.5
    xs = [0.2 + i * (w + gap) for i in range(3)]
    for i in range(3):
        block(ax, xs[i], 2.2, w, h, top[i])
        block(ax, xs[i], 0.3, w, h, bottom[i])
    for i, word in enumerate(top):
        j = bottom.index(word)
        ax.plot([xs[i] + w / 2, xs[j] + w / 2], [2.2 - 0.05, 0.3 + h + 0.05],
                color=P.series[2], lw=THIN, ls=DASH)
    label(ax, xs[2] + w + 0.5, 2.55, "“dog bites man”", rank="tick",
          ha="left", va="center", color=P.ink)
    label(ax, xs[2] + w + 0.5, 0.65, "“man bites dog”", rank="tick",
          ha="left", va="center", color=P.ink)
    label(ax, xs[2] + w + 0.5, 1.6,
          "the same three vectors,\nin a different order", rank="small",
          ha="left", va="center")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0.1, 3.1)
    ax.axis("off")


# ----------------------------- 4. three things a token has to be able to do
@render("transformer-problem-qkv", size="banner")
def transformer_problem_qkv(ax):
    w, h, y = 1.6, 0.75, 1.0
    xs = [0.6, 4.4, 8.2]
    for x, tok in zip(xs, ["the", "cat", "sat"]):
        block(ax, x, y, w, h, tok, emph=tok != "the")
    arrow(ax, (xs[1] + w / 2, y + h + 0.05), (xs[2] + w / 2, y + h + 0.05),
          emph=True, curve=-0.45)
    label(ax, (xs[1] + xs[2]) / 2 + w / 2, y + h + 1.15,
          "3. hands over information", rank="tick", ha="center", color=P.ink)
    label(ax, xs[2] + w / 2, y - 0.3, "1. says what it\nis looking for",
          rank="small", ha="center", va="top")
    label(ax, xs[1] + w / 2, y - 0.3, "2. shows what\nit offers", rank="small",
          ha="center", va="top")
    label(ax, xs[0] + w / 2, y - 0.3, "not what “sat”\nis looking for",
          rank="small", ha="center", va="top")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.3, 3.2)
    ax.axis("off")


# ------------------------------------- 5. how fast the pairs pile up
@render("transformer-problem-scores", size="wide")
def transformer_problem_scores(ax):
    import numpy as np

    n = np.logspace(np.log10(3), 4, 100)
    ax.plot(n, n ** 2, **P.emph())
    for x, text, dx, dy in ((3, "“the cat sat”: 9 scores", 1.4, 0.8),
                            (1000, "1,000 tokens:\n1,000,000 scores", 0.07,
                             1.0)):
        ax.plot(x, x * x, "o", color=P.ink, ms=6, zorder=5)
        label(ax, x * dx, x * x * dy, text, rank="tick",
              ha="left" if dx > 1 else "right", va="center", color=P.ink)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 3e8)
    ax.set_xlabel("tokens in the text")
    ax.set_ylabel("scores per head")
    clean(ax, spines="lb")


# ------------------------- 6. what each position may see, and its target
@render("transformer-problem-mask", size="banner")
def transformer_problem_mask(ax):
    toks, targets = ["the", "cat", "sat"], ["cat", "sat", "on"]
    w, h, gap, x0 = 1.3, 0.55, 0.2, 3.0
    for i in range(3):
        y = (2 - i) * 0.85
        label(ax, x0 - 0.3, y + h / 2, f"position {i + 1}", rank="small",
              ha="right", va="center")
        for j, tok in enumerate(toks):
            x = x0 + j * (w + gap)
            if j <= i:
                block(ax, x, y, w, h, tok)
            else:
                block(ax, x, y, w, h, tok, edgecolor=P.series[2],
                      linewidth=HAIR, linestyle=DASH)
        end = x0 + 3 * (w + gap)
        arrow(ax, (end, y + h / 2), (end + 1.0, y + h / 2))
        label(ax, end + 1.15, y + h / 2, f"must predict “{targets[i]}”",
              rank="tick", ha="left", va="center", color=P.ink)
    label(ax, x0 + 1.5 * w + gap, 2 * 0.85 + h + 0.3,
          "solid: may be seen. dashed: must stay hidden", rank="small",
          ha="center")
    ax.set_xlim(0.6, 12.4)
    ax.set_ylim(-0.15, 2.95)
    ax.axis("off")


# ------------------------ 7. three weights and three values, one vector
@render("transformer-problem-weighted-sum", size="wide")
def transformer_problem_weighted_sum(ax):
    toks = ["the", "cat", "sat"]
    wts = [0.115, 0.709, 0.175]
    vals = ["(1.000, 1.000)", "(0.841, 1.540)", "(0.909, −0.416)"]
    for i, wt in enumerate(wts):
        ax.bar(i, wt, width=0.5, color=P.ink if wt > 0.5 else P.series[2])
        label(ax, i, wt + 0.03, f"{wt:.3f}", rank="small", ha="center",
              color=P.ink)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{t}\nvalue {v}" for t, v in zip(toks, vals)])
    ax.set_ylim(0, 0.85)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("weight of “sat” on the token")
    clean(ax, spines="lb")
    ax.tick_params(axis="x", length=0, labelsize=8.5)


# ----------------------- 8. one set of weights can only give the average
@render("transformer-problem-multi-head", size="wide")
def transformer_problem_multi_head(ax):
    arrow(ax, (0, 0), (1, 0), shrinkA=0, shrinkB=0)
    arrow(ax, (0, 0), (0, 1), shrinkA=0, shrinkB=0)
    arrow(ax, (0, 0), (0.5, 0.5), emph=True, shrinkA=0, shrinkB=0)
    label(ax, 1.05, 0.0, "value of “cat” (1, 0)", rank="tick", ha="left",
          va="center")
    label(ax, 0.05, 1.08, "value of “sat” (0, 1)", rank="tick", ha="left",
          va="center")
    label(ax, 0.57, 0.52, "weights 0.5 and 0.5:\n(0.5, 0.5), which is neither",
          rank="tick", ha="left", va="center", color=P.ink)
    ax.plot([1, 0], [0, 1], color=P.series[2], lw=THIN, ls=DASH)
    ax.set_xlim(-0.15, 2.45)
    ax.set_ylim(-0.2, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------- 9. what many layers do to the signal and to the numbers
@render("transformer-problem-depth", size="panel", axes=False)
def transformer_problem_depth(fig):
    import numpy as np

    axes = fig.subplots(1, 2)
    n = np.arange(0, 13)
    panels = [(0.5 ** n, "Training signal reaching layer 1",
               "if each layer passes back half", "0.00024"),
              (2.0 ** n, "Size of the numbers",
               "if each layer doubles them", "4,096")]
    for ax, (y, title, note, last) in zip(axes, panels):
        ax.plot(n, y, marker="o", ms=3.5, **P.emph())
        ax.set_yscale("log")
        ax.set_xlabel("layers")
        ax.set_title(title, fontsize=10.5, color=P.ink)
        # The note goes in whichever corner the line leaves empty.
        falling = y[-1] < 1
        label(ax, 0.95, 0.92 if falling else 0.06, note, rank="small",
              ha="right", transform=ax.transAxes)
        label(ax, 10.3, y[-1], last, rank="tick", ha="right", va="center",
              color=P.ink)
        ax.set_xticks([0, 4, 8, 12])
        clean(ax, spines="lb")
    fig.subplots_adjust(wspace=0.4)


# ------------- 10. attention can only land between the values it averages
@render("transformer-problem-feed-forward", size="wide")
def transformer_problem_feed_forward(ax):
    from matplotlib.patches import Polygon

    vals = {"the": (1.000, 1.000), "cat": (0.841, 1.540),
            "sat": (0.909, -0.416)}
    outs = {"cat": (0.881, 1.405), "sat": (0.872, 1.135)}
    ax.add_patch(Polygon(list(vals.values()), closed=True, lw=THIN,
                         facecolor=P.series[2], alpha=0.25,
                         edgecolor=P.series[1]))
    side = {"the": (0.012, 0.0, "left"), "cat": (-0.012, 0.0, "right"),
            "sat": (0.012, 0.0, "left")}
    for tok, (x, y) in vals.items():
        dx, dy, ha = side[tok]
        ax.plot(x, y, "o", mfc="none", mec=P.series[1], ms=7, zorder=4)
        label(ax, x + dx, y + dy, f"value of “{tok}”", rank="small", ha=ha,
              va="center")
    for tok, (x, y) in outs.items():
        ax.plot(x, y, "o", color=P.ink, ms=5.5, zorder=5)
        label(ax, x + 0.012, y - 0.03, f"output for “{tok}”", rank="tick",
              ha="left", va="center", color=P.ink)
    ax.set_xlim(0.74, 1.14)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("first number")
    ax.set_ylabel("second number")
    clean(ax, spines="lb")


# ------------------------------- 11. a prediction that needs two rounds
@render("transformer-problem-stack", size="banner")
def transformer_problem_stack(ax):
    toks = ["Harry", "Potter", "went", "home", ".", "Later", ",", "Harry",
            "___"]
    w, h, gap, y = 1.15, 0.6, 0.18, 1.0
    cx = [0.2 + i * (w + gap) + w / 2 for i in range(len(toks))]
    for i, tok in enumerate(toks):
        block(ax, cx[i] - w / 2, y, w, h, tok, emph=i in (0, 1, 7, 8),
              rank="small")
    arrow(ax, (cx[7], y + h + 0.05), (cx[0], y + h + 0.05), emph=True,
          curve=0.22)
    label(ax, (cx[0] + cx[7]) / 2, y + h + 1.2,
          "round 1: find the earlier “Harry”", rank="tick", ha="center",
          color=P.ink)
    arrow(ax, (cx[0], y - 0.05), (cx[1], y - 0.05), emph=True, curve=0.9)
    label(ax, (cx[0] + cx[1]) / 2, y - 0.75,
          "round 2: take the\ntoken after it", rank="small", ha="center",
          va="top")
    label(ax, cx[8], y - 0.3, "“Potter”", rank="tick", ha="center", va="top",
          color=P.ink)
    ax.set_xlim(0, cx[8] + w / 2 + 0.2)
    ax.set_ylim(-0.6, 3.2)
    ax.axis("off")


# ----------------------- 12. one vector in, eight probabilities wanted
@render("transformer-problem-predict", size="banner")
def transformer_problem_predict(ax):
    vocab = ["<end>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
    cells(ax, 0.2, 0.8, [[-0.639, -0.891, -0.136, 1.667]], 1.1, 0.6)
    label(ax, 2.4, 1.75, "final vector of “sat”", rank="small", ha="center")
    arrow(ax, (4.75, 1.1), (6.25, 1.1), emph=True)
    label(ax, 5.5, 1.4, "?", rank="label", ha="center")
    cw = 0.95
    cells(ax, 6.4, 0.8, [["?"] * 8], cw, 0.6)
    for j, tok in enumerate(vocab):
        label(ax, 6.4 + (j + 0.5) * cw, 1.75, tok, rank="small", ha="center")
    label(ax, 6.4 + 4 * cw, 0.35,
          "a probability for every token, adding up to 1", rank="small",
          ha="center")
    ax.set_xlim(0, 6.4 + 8 * cw + 0.2)
    ax.set_ylim(0.1, 2.1)
    ax.axis("off")


# ------------------------------ 13. the two ways of choosing go wrong
@render("transformer-problem-generate", size="banner")
def transformer_problem_generate(ax):
    rows = [("always the most likely token",
             "the cat sat on the mat on the mat on the mat".split(), 6, 1.7),
            ("any token, however unlikely",
             "the cat sat mat ran the on dog sat the ran mat".split(), 3,
             0.3)]
    w, h, gap = 0.95, 0.5, 0.1
    for title, toks, start, y in rows:
        label(ax, 0.2, y + h + 0.22, title, rank="tick", ha="left",
              color=P.ink)
        for i, tok in enumerate(toks):
            block(ax, 0.2 + i * (w + gap), y, w, h, tok, emph=i >= start,
                  rank="small")
    label(ax, 0.2 + 12 * (w + gap) + 0.1, 1.95, "stuck in\na loop",
          rank="small", ha="left", va="center")
    label(ax, 0.2 + 12 * (w + gap) + 0.1, 0.55, "nonsense", rank="small",
          ha="left", va="center")
    ax.set_xlim(0, 0.2 + 12 * (w + gap) + 1.7)
    ax.set_ylim(0.1, 2.75)
    ax.axis("off")


# ---------------- 7. how fast the output of "sat" moves as each score rises
@render("transformer-gradient", size="square")
def transformer_gradient(ax):
    # Each arrow is weight x (value - output), drawn to scale, so it lies on
    # the dashed line to its value. The value of "sat" is below the frame.
    out = (0.872, 1.135)
    bottom = 0.6
    rows = [("the", (1.000, 1.000), (0.015, -0.016)),
            ("cat", (0.841, 1.540), (-0.021, 0.288)),
            ("sat", (0.909, -0.416), (0.007, -0.272))]
    for tok, (x, y), (dx, dy) in rows:
        if y < bottom:
            t = (out[1] - bottom) / (out[1] - y)
            ax.plot([out[0], out[0] + t * (x - out[0])], [out[1], bottom],
                    **P.guide())
            label(ax, out[0] + 0.03, bottom + 0.05,
                  f"towards the value of “{tok}”,\n({x:.3f}, {y:.3f})"
                  .replace("-", "−"), rank="small", ha="left", va="bottom")
        else:
            ax.plot([out[0], x], [out[1], y], **P.guide())
            ax.plot(x, y, "o", color=P.series[1], ms=5, zorder=4)
            label(ax, x + 0.03, y, f"value of “{tok}”", rank="small",
                  ha="left", va="center")
        arrow(ax, out, (out[0] + dx, out[1] + dy), emph=True, shrinkA=0,
              shrinkB=0)
    ax.plot(*out, "o", color=P.ink, ms=6, zorder=5)
    label(ax, out[0] - 0.04, out[1], "output for “sat”\n(0.872, 1.135)",
          rank="tick", ha="right", va="center", color=P.ink)
    label(ax, out[0] + 0.03, 1.33, "score of “cat” rises",
          rank="small", ha="left", va="center")
    label(ax, out[0] + 0.03, 0.88, "score of “sat” rises",
          rank="small", ha="left", va="center")
    ax.set_xlim(0.35, 1.45)
    ax.set_ylim(bottom, 1.7)
    ax.set_aspect("equal")
    clean(ax, spines="lb")
    ax.set_xlabel("first number")
    ax.set_ylabel("second number")


# ------------------- 3. one step along turns the fast pair by the same angle
@render("transformer-position-rotation", size="square")
def transformer_position_rotation(ax):
    import numpy as np

    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.sin(t), np.cos(t), **P.guide())
    ax.axhline(0, **P.grid())
    ax.axvline(0, **P.grid())
    names = {0: "0, the", 1: "1, cat", 2: "2, sat"}
    for q in range(7):
        x, y = np.sin(q), np.cos(q)
        key = q in (0, 1, 5, 6)
        ax.plot(x, y, "o", color=P.ink if key else P.series[1],
                ms=6 if key else 4.5, zorder=5)
        label(ax, 1.2 * x, 1.2 * y, names.get(q, str(q)), rank="small",
              ha="center", va="center", color=P.ink if key else P.mute)
    for q in range(6):
        r0 = 0.78
        arrow(ax, (r0 * np.sin(q + 0.12), r0 * np.cos(q + 0.12)),
              (r0 * np.sin(q + 0.88), r0 * np.cos(q + 0.88)), curve=-0.25,
              emph=q in (0, 5))
    for q in (0, 5):
        m = q + 0.5
        label(ax, 0.5 * np.sin(m), 0.5 * np.cos(m), "turn\nby 1", rank="small",
              ha="center", va="center", color=P.ink)
    label(ax, 1.45, 0.05, "dim 1: $\\sin p$", rank="tick", ha="right",
          va="bottom")
    label(ax, 0.06, 1.42, "dim 2: $\\cos p$", rank="tick", ha="left",
          va="top")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------- 9. where the norm sits: on the residual path or off it
@render("transformer-pre-post-norm", size="banner")
def transformer_pre_post_norm(ax):
    w, h, plus = 2.2, 0.75, 0.8
    rows = (("post-norm", 3.0, [("attention", w), ("+", plus), ("LayerNorm", w)],
             "the residual path passes through a LayerNorm"),
            ("pre-norm", 0.5, [("LayerNorm", w), ("attention", w), ("+", plus)],
             "nothing on the residual path is normalized"))
    for name, y, parts, note in rows:
        label(ax, 0.0, y + h / 2, name, rank="tick", ha="left", va="center",
              color=P.ink)
        block(ax, 2.2, y, 1.2, h, "$H$")
        x, xs = 4.2, {}
        prev = 2.2 + 1.2
        for text, width in parts:
            block(ax, x, y, width, h, text, emph=text == "LayerNorm",
                  rank="label" if text == "+" else "tick")
            arrow(ax, (prev, y + h / 2), (x, y + h / 2))
            xs[text] = (x, width)
            prev = x + width
            x += width + 0.8
        arrow(ax, (prev, y + h / 2), (prev + 0.7, y + h / 2))
        # The residual path: from H over the top, down into the plus.
        top = y + h + 0.45
        x_from = 2.8
        x_to = xs["+"][0] + plus / 2
        ax.plot([x_from, x_from, x_to], [y + h + 0.03, top, top],
                color=P.series[2], lw=THIN, ls=DASH)
        arrow(ax, (x_to, top), (x_to, y + h + 0.02), color=P.series[2],
              linestyle=DASH)
        label(ax, (x_from + x_to) / 2, top + 0.2, note, rank="small",
              ha="center")
    ax.set_xlim(0, 12.1)
    ax.set_ylim(0.3, 5.0)
    ax.axis("off")


# ================================== Efficient Transformers: the problems

# ------------------------------- 0. the work grows with the square
@render("efficient-problem-setup", size="wide")
def efficient_problem_setup(ax):
    import numpy as np

    n = np.linspace(0, 8000, 200)
    ax.plot(n, n ** 2 / 1e6, **P.emph())
    for x, dx in ((4000, -200), (8000, -200)):
        ax.plot(x, x * x / 1e6, "o", color=P.ink, ms=6, zorder=5)
        label(ax, x + dx, x * x / 1e6 + 2,
              f"{x:,} tokens:\n{x * x // 10 ** 6} million", rank="tick",
              ha="right", va="bottom", color=P.ink)
    ax.set_xlim(0, 8400)
    ax.set_ylim(0, 72)
    ax.set_xlabel("tokens in the text")
    ax.set_ylabel("scores per head (millions)")
    clean(ax, spines="lb")


# ---------------------- 1. scores computed while writing a reply
@render("efficient-problem-prefill", size="wide")
def efficient_problem_prefill(ax):
    import numpy as np

    n = np.arange(1, 1001)
    scratch = n * (n + 1) * (2 * n + 1) / 6
    rows = n * (n + 1) / 2
    ax.plot(n, scratch, **P.emph())
    ax.plot(n, rows, **P.line(1))
    ax.set_yscale("log")
    label(ax, 980, scratch[-1] * 2.2, "every run from scratch: 333,833,500",
          rank="tick", ha="right", color=P.ink)
    label(ax, 980, rows[-1] * 0.05, "only the new row each run: 500,500",
          rank="tick", ha="right")
    ax.set_xlim(0, 1010)
    ax.set_ylim(1, 5e9)
    ax.set_xlabel("tokens generated so far")
    ax.set_ylabel("scores computed, per head")
    clean(ax, spines="lb")


# ------------------------------- 2. the cache grows with every token
@render("efficient-problem-cache", size="wide")
def efficient_problem_cache(ax):
    import numpy as np

    n = np.linspace(0, 4096, 100)
    per_token = 2 * 80 * 64 * 128 * 2 / 2 ** 30          # GiB
    ax.plot(n, n * per_token, **P.emph())
    ax.plot(4096, 10, "o", color=P.ink, ms=6, zorder=5)
    label(ax, 3950, 10.3, "4,096 tokens: 10 GiB", rank="tick", ha="right",
          va="bottom", color=P.ink)
    label(ax, 250, 8.4, "Llama 2 70B, one conversation:\n2.5 MiB for every token",
          rank="small", ha="left", va="center")
    ax.set_xlim(0, 4300)
    ax.set_ylim(0, 11.8)
    ax.set_xlabel("tokens in the conversation")
    ax.set_ylabel("cache size (GiB)")
    clean(ax, spines="lb")


# ----------------------- 3. what one token stores in one layer
@render("efficient-problem-sharing", size="wide")
def efficient_problem_sharing(ax):
    from matplotlib.patches import Rectangle

    c = 0.42
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((0.3 + j * c, 0.3 + i * c), c * 0.86,
                                   c * 0.86, facecolor="none",
                                   edgecolor=P.series[1], lw=THIN))
    ax.add_patch(Rectangle((0.3, 0.3 + 7 * c), c * 0.86, c * 0.86,
                           facecolor=P.series[2], edgecolor=P.ink, lw=LINE))
    label(ax, 0.3 + 4 * c, 0.3 + 8 * c + 0.15, "one token, one layer: 64 heads",
          rank="tick", ha="center", color=P.ink)
    x = 0.3 + 8 * c + 0.5
    label(ax, x, 3.3, "one square: one head's key and value,\n128 + 128 = 256 numbers",
          rank="small", ha="left", va="center")
    label(ax, x, 2.2, "64 × 256 = 16,384 numbers", rank="tick", ha="left",
          va="center", color=P.ink)
    label(ax, x, 1.2, "for every token,\nin each of 80 layers", rank="small",
          ha="left", va="center")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0.1, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------- 4. the table of scores against the fast memory
@render("efficient-problem-flash", size="wide")
def efficient_problem_flash(ax):
    from matplotlib.patches import Rectangle

    big, small = 128 ** 0.5, 20 ** 0.5        # sides, so areas are to scale
    ax.add_patch(Rectangle((0.3, 0.3), big, big, facecolor="none",
                           edgecolor=P.ink, lw=LINE))
    ax.add_patch(Rectangle((0.3, 0.3), small, small, facecolor=P.series[2],
                           edgecolor=P.ink, lw=THIN, alpha=0.8))
    label(ax, 0.3 + big / 2, 0.3 + big * 0.72,
          "table of scores, one head,\n8,192 tokens: 128 MiB", rank="tick",
          ha="center", va="center", color=P.ink)
    label(ax, 0.3 + small / 2, 0.3 + small / 2, "fast memory:\nabout 20 MB",
          rank="small", ha="center", va="center", color=P.ink)
    x = 0.3 + big + 0.8
    for i, text in enumerate(["write the scores", "read them back",
                              "write the weights", "read them back"]):
        label(ax, x, 0.3 + big - 1.3 - i * 1.9, f"{i + 1}. {text}",
              rank="small", ha="left", va="center")
    label(ax, x, 0.3 + big - 9.6, "every trip is to\nslow main memory",
          rank="tick", ha="left", va="center", color=P.ink)
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 12.0)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------- 5. keys compared for each new token
@render("efficient-problem-linear", size="wide")
def efficient_problem_linear(ax):
    import numpy as np

    n = np.linspace(0, 100000, 50)
    ax.plot(n / 1000, n / 1000, **P.emph())
    ax.plot(n / 1000, np.full_like(n, 2.0), color=P.series[1], lw=LINE,
            ls=DASH)
    label(ax, 58, 68, "softmax attention:\none comparison per earlier token",
          rank="tick", ha="right", va="center", color=P.ink)
    label(ax, 97, 8, "the goal: the same work for every token", rank="tick",
          ha="right", va="center")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 108)
    ax.set_xlabel("tokens so far (thousands)")
    ax.set_ylabel("keys compared for the\nnext token (thousands)")
    clean(ax, spines="lb")


# --------------- 2. a chunk of two new tokens, with and without the shift
@render("efficient-chunk-mask", size="banner")
def efficient_chunk_mask(ax):
    toks, rows = ["the", "cat", "sat", "on", "the"], ["on", "the"]
    scores = [[0.021, 1.461, 1.433, 0.000, 0.043],
              [0.028, 0.757, 0.709, -0.035, 0.010]]
    cw, ch, gap = 1.0, 0.55, 1.6
    panels = [("shifted by the 3 cached tokens", 3, "“on” sees 4 keys"),
              ("not shifted", 0, "“on” sees only “the”")]
    for k, (title, shift, note) in enumerate(panels):
        x0 = 0.9 + k * (5 * cw + gap)
        hidden = {(i, j) for i in range(2) for j in range(5) if j > i + shift}
        shown = [[s if (i, j) not in hidden else "" for j, s in enumerate(r)]
                 for i, r in enumerate(scores)]
        cells(ax, x0, 0, shown, cw, ch, muted=hidden)
        for j, tok in enumerate(toks):
            label(ax, x0 + (j + 0.5) * cw, 2 * ch + 0.15, tok, rank="small",
                  ha="center")
        for i, tok in enumerate(rows):
            label(ax, x0 - 0.12, (1 - i + 0.5) * ch, tok, rank="small",
                  ha="right", va="center")
        label(ax, x0 + 2.5 * cw, 2 * ch + 0.62, title, rank="tick",
              ha="center", color=P.ink)
        label(ax, x0 + 2.5 * cw, -0.3, note, rank="small", ha="center",
              va="top")
    ax.set_xlim(0, 0.9 + 10 * cw + gap + 0.2)
    ax.set_ylim(-0.75, 2 * ch + 0.95)
    ax.axis("off")


# ---------------- 3. one stored latent, rebuilt into each head's key and value
@render("efficient-latent", size="wide")
def efficient_latent(ax):
    cw, ch = 1.0, 0.55
    lx, ly = 0.3, 1.6
    cells(ax, lx, ly, [[0.841, 1.540, 1.010, 1.000]], cw, ch)
    label(ax, lx + 2 * cw, ly + ch + 0.25, "latent of “cat”: the 4 numbers stored",
          rank="tick", ha="center", color=P.ink)
    parts = [("head 1 key", [1.540, 1.010]), ("head 1 value", [0.841, 1.540]),
             ("head 2 key", [1.540, 1.000]), ("head 2 value", [1.010, 1.000])]
    bx, bw = 6.4, 0.95
    for i, (name, nums) in enumerate(parts):
        y = 2.75 - i * 0.85
        cells(ax, bx, y, [nums], bw, ch, muted={(0, 0), (0, 1)})
        label(ax, bx + 2 * bw + 0.15, y + ch / 2, name, rank="small",
              ha="left", va="center")
        arrow(ax, (lx + 4 * cw + 0.1, ly + ch / 2), (bx - 0.1, y + ch / 2))
    label(ax, bx + bw, -0.35, "rebuilt from the latent when needed, not stored",
          rank="small", ha="center", va="center")
    ax.set_xlim(0, bx + 2 * bw + 1.6)
    ax.set_ylim(-0.6, 3.75)
    ax.axis("off")


# ------------------ 4. a window of 2 keys, and the 2-slot cache it needs
@render("efficient-window", size="wide")
def efficient_window(ax):
    toks = ["the", "cat", "sat", "on"]
    scores = [[0.000, 0.714, 0.721, 0.021], [0.714, 1.814, 0.424, -1.400],
              [0.721, 2.539, 1.142, -1.393], [0.021, 1.461, 1.433, 0.000]]
    cw, ch, x0 = 1.0, 0.55, 0.9
    outside = {(i, j) for i in range(4) for j in range(4) if j > i or j < i - 1}
    shown = [[s if (i, j) not in outside else "" for j, s in enumerate(r)]
             for i, r in enumerate(scores)]
    cells(ax, x0, 0, shown, cw, ch, muted=outside)
    for i, tok in enumerate(toks):
        label(ax, x0 - 0.12, (3 - i + 0.5) * ch, tok, rank="small",
              ha="right", va="center")
        label(ax, x0 + (i + 0.5) * cw, 4 * ch + 0.15, tok, rank="small",
              ha="center")
    label(ax, x0 + 2 * cw, 4 * ch + 0.62, "head 1 scores, window of 2",
          rank="tick", ha="center", color=P.ink)

    sx = x0 + 4 * cw + 2.3
    slots = [["the", ""], ["the", "cat"], ["sat", "cat"], ["sat", "on"]]
    cells(ax, sx, 0, slots, 1.0, ch)
    for i, tok in enumerate(toks):
        label(ax, sx - 0.12, (3 - i + 0.5) * ch, f"after “{tok}”",
              rank="small", ha="right", va="center")
    for j in range(2):
        label(ax, sx + (j + 0.5), 4 * ch + 0.15, f"slot {j}", rank="small",
              ha="center")
    label(ax, sx + 1.0, 4 * ch + 0.62, "the cache", rank="tick",
          ha="center", color=P.ink)
    ax.set_xlim(0, sx + 2.4)
    ax.set_ylim(-0.2, 4 * ch + 0.95)
    ax.axis("off")


# ------------ 5. the exponential as a dot product, cut at a degree
@render("efficient-taylor", size="panel", axes=False)
def efficient_taylor(fig):
    import numpy as np

    left, right = fig.subplots(1, 2)
    dots = [1.000, 3.539, 6.763, 9.491, 11.223, 12.103, 12.475, 12.610,
            12.653]
    left.axhline(12.669, color=P.series[1], lw=THIN, ls=DASH)
    left.plot(range(9), dots, "o-", color=P.ink, lw=LINE, ms=4.5)
    label(left, 8.2, 13.4, "exact: e$^{2.539}$ = 12.669", rank="small",
          ha="right")
    for n, feats, ha in ((1, 3, "left"), (4, 31, "left"), (8, 511, "right")):
        label(left, n + (0.15 if ha == "left" else 0.1), dots[n] - 1.4,
              f"{feats} features", rank="small", ha=ha, va="center")
    left.set_xlim(-0.4, 8.5)
    left.set_ylim(0, 14.5)
    left.set_xlabel("degree of the cut")
    left.set_ylabel("similarity of “sat” and “cat”")
    clean(left, spines="lb")

    s = np.linspace(-4, 2, 200)
    right.axhline(0, color=P.faint, lw=HAIR)
    right.plot(s, np.exp(s), color=P.ink, lw=LINE)
    right.plot(s, 1 + s, color=P.series[1], lw=THIN, ls=DASH)
    right.plot(s, 1 + s + s**2 / 2, color=P.series[1], lw=THIN)
    label(right, 1.95, 7.2, "e$^{s}$", rank="tick", ha="right",
          color=P.ink)
    label(right, -3.9, 6.7, "degree 2:\nnever below 0.5", rank="small",
          ha="left", va="center")
    label(right, -2.9, -2.6, "degree 1: below 0\nfor scores under −1",
          rank="small", ha="left", va="center")
    right.set_xlim(-4, 2)
    right.set_ylim(-3.5, 8)
    right.set_xlabel("score s")
    clean(right, spines="lb")


# ----------------- 5. linear weights of "sat" with a decay of 0.5
@render("efficient-decay", size="wide")
def efficient_decay(ax):
    toks = ["the", "cat", "sat"]
    lin, dec = [0.275, 0.436, 0.289], [0.119, 0.379, 0.502]
    for i in range(3):
        ax.bar(i - 0.19, lin[i], width=0.34, color=P.series[2])
        ax.bar(i + 0.19, dec[i], width=0.34, color=P.ink)
        label(ax, i - 0.19, lin[i] + 0.025, f"{lin[i]:.3f}", rank="small",
              ha="center")
        label(ax, i + 0.19, dec[i] + 0.025, f"{dec[i]:.3f}", rank="small",
              ha="center", color=P.ink)
    label(ax, -0.55, 0.72, "linear attention", rank="tick", ha="left",
          color=P.mute)
    label(ax, -0.55, 0.64, "with a decay of 0.5", rank="tick", ha="left",
          color=P.ink)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["the\n(×0.25)", "cat\n(×0.5)", "sat\n(×1)"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("weight of “sat” on each token")
    clean(ax, spines="lb")
    ax.tick_params(axis="x", length=0)


if __name__ == "__main__":
    run_all(globals())
