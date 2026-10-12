# Partly ported from PyTorch (torch/_tensor_str.py: the print options, the rules choosing
# precision or scientific notation, and the summarisation), BSD-3-Clause. See THIRD_PARTY_NOTICES.md.
"""How a tensor prints. This follows torch's own rules (torch/_tensor_str.py), so the output
students see in the browser is the output they would see in Colab."""

import numpy as np

PRECISION, THRESHOLD, EDGE_ITEMS, LINE_WIDTH = 4, 1000, 3, 80
SCI_MODE = None  # None lets the values decide, as in torch; True or False forces it.
PREFIX = "tensor("

PROFILES = {
    "default": (4, 1000, 3, 80),
    "short": (2, 1000, 2, 80),
    "full": (4, float("inf"), 3, 80),
}


def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None):
    """torch.set_printoptions: a profile resets all four numbers, then each one given
    overrides it. sci_mode is set every call, so leaving it out goes back to automatic."""
    global PRECISION, THRESHOLD, EDGE_ITEMS, LINE_WIDTH, SCI_MODE
    if profile is not None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile!r}")
        PRECISION, THRESHOLD, EDGE_ITEMS, LINE_WIDTH = PROFILES[profile]
    if precision is not None:
        PRECISION = precision
    if threshold is not None:
        THRESHOLD = threshold
    if edgeitems is not None:
        EDGE_ITEMS = edgeitems
    if linewidth is not None:
        LINE_WIDTH = linewidth
    SCI_MODE = sci_mode


def _summarized(array):
    """The first and last EDGE_ITEMS entries along every dimension."""
    if array.ndim == 0:
        return array
    if array.shape[0] > 2 * EDGE_ITEMS:
        array = np.concatenate([array[:EDGE_ITEMS], array[-EDGE_ITEMS:]])
    if array.ndim == 1:
        return array
    return np.stack([_summarized(row) for row in array])


def _formatter(array):
    """(format one value, column width) for the values that will be shown."""
    flat = array.reshape(-1)
    if array.dtype.kind != "f":
        width = max([1] + [len(str(value)) for value in flat.tolist()])
        return (lambda value: str(value).rjust(width)), width

    shown = flat[np.isfinite(flat) & (flat != 0)].astype(np.float64)
    whole = bool(np.all(shown == np.ceil(shown)))
    sci, width = False, 1
    if shown.size:
        low, high = np.abs(shown).min(), np.abs(shown).max()
        sci = bool(high / low > 1000.0 or high > 1.0e8 or (not whole and low < 1.0e-4))
    if SCI_MODE is not None:
        sci = bool(SCI_MODE)
    if shown.size:
        if sci:
            width = max(len(f"{value:.{PRECISION}e}") for value in shown.tolist())
        elif whole:
            width = max(len(f"{value:.0f}") + 1 for value in shown.tolist())
        else:
            width = max(len(f"{value:.{PRECISION}f}") for value in shown.tolist())

    def fmt(value):
        if sci:
            text = f"{value:{width}.{PRECISION}e}"
        elif whole:
            # Whole numbers print as "3."; nan and inf get no dot.
            text = f"{value:.0f}" + ("." if np.isfinite(value) else "")
        else:
            text = f"{value:.{PRECISION}f}"
        return text.rjust(width)

    return fmt, width


def _vector(array, indent, summarize, fmt, width):
    per_line = max(1, (LINE_WIDTH - indent) // (width + 2))
    if summarize and array.shape[0] > 2 * EDGE_ITEMS:
        cells = [fmt(v) for v in array[:EDGE_ITEMS].tolist()] + [" ..."]
        cells += [fmt(v) for v in array[-EDGE_ITEMS:].tolist()]
    else:
        cells = [fmt(v) for v in array.tolist()]
    lines = [", ".join(cells[i : i + per_line]) for i in range(0, len(cells), per_line)]
    return "[" + (",\n" + " " * (indent + 1)).join(lines) + "]"


def _nested(array, indent, summarize, fmt, width):
    if array.ndim == 0:
        return fmt(array.item())
    if array.ndim == 1:
        return _vector(array, indent, summarize, fmt, width)
    rows = range(array.shape[0])
    if summarize and array.shape[0] > 2 * EDGE_ITEMS:
        parts = [_nested(array[i], indent + 1, summarize, fmt, width) for i in rows[:EDGE_ITEMS]]
        parts.append("...")
        parts += [_nested(array[i], indent + 1, summarize, fmt, width) for i in rows[-EDGE_ITEMS:]]
    else:
        parts = [_nested(array[i], indent + 1, summarize, fmt, width) for i in rows]
    return "[" + ("," + "\n" * (array.ndim - 1) + " " * (indent + 1)).join(parts) + "]"


def tensor_repr(tensor):
    array, dtype = tensor._data, tensor.dtype
    indent, suffixes = len(PREFIX), []
    if array.size == 0:
        if array.ndim != 1:
            suffixes.append(f"size={tuple(array.shape)}")
        if dtype.name != "float32":
            suffixes.append(f"dtype={dtype}")
        body = "[]"
    else:
        # float32, int64 and bool are what plain Python values become, so they are not named.
        if dtype.name not in ("float32", "int64", "bool"):
            suffixes.append(f"dtype={dtype}")
        summarize = array.size > THRESHOLD
        fmt, width = _formatter(_summarized(array) if summarize else array)
        body = _nested(array, indent, summarize, fmt, width)
    if tensor.grad_fn is not None:
        suffixes.append(f"grad_fn=<{type(tensor.grad_fn).__name__}>")
    elif tensor.requires_grad:
        suffixes.append("requires_grad=True")

    text = PREFIX + body
    last_line = len(text) - text.rfind("\n") + 1
    for suffix in suffixes:
        if last_line + len(suffix) + 2 > LINE_WIDTH:
            text += ",\n" + " " * indent + suffix
            last_line = indent + len(suffix)
        else:
            text += ", " + suffix
            last_line += len(suffix) + 2
    return text + ")"
