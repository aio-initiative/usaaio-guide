#!/usr/bin/env python3
"""Import B's problem files into A's data.

Usage:
  scripts/import-b/problems.py <B module id> [--root <B repo>] [--out <path>]
      [--into <content/problems/<id>.json>] [--module <A module id>]
      [--stage <stage>] [--spec <spec file>] [--strict]

Reads every content/problems/<B module id>/*.mdx file in site B and writes one
per-module data file holding the module's problems.js index entries ("index")
and its theory.js theory problems ("theory"). B's sentences are copied
verbatim; only the container changes:

With --into, the run's entries are merged into that module's file, which may
already hold A's kept entries: entries whose id the run emits are replaced in
place, ids the file does not have are appended after its own entries in the
order the run emits them (B's order), and every other entry and key is left
untouched. The file is created as {"module", "stage", "index", "theory"} if it
does not exist. Running the import twice leaves the file byte-identical, and
the formatting matches scripts/split-content.mjs.

    a paragraph            -> {"p": "..."}
    $$ ... $$              -> {"tex": "..."}
    a markdown table       -> {"tex": "\\begin{array}..."} (KaTeX)
    a "- " list            -> {"list": [...]}
    <Choice ...>text       -> {"text": ..., "correct": true, "explain": ...}
    <Part> / <Solution>    -> one part with prompt and answer blocks
    answer="3"             -> a closing {"result": "3"} block

What the tool decides itself, because B does not carry it:
  - the A module id and the stage tag, from the module spec or the flags;
  - the index entry's source and tags (Custom / Multiple choice for an mcq,
    USAAIO-style / Multi-part for a written problem);
  - each written part's "kind" and "requirement", which are A's grading
    metadata. Defaults: a part with a numeric answer is a "computation" whose
    requirement is "Reasoning is not required.", a part without one is a
    "derivation" whose requirement is "A complete derivation is required.".
    Every part that falls back to the default is a warning, because a person
    should confirm it. A spec file can set both per part.
  - the written index card's prompt, which is a one-line summary. Default:
    the problem's own opening paragraphs with the math stripped, plus
    "N parts.". A spec file can set it.

A spec file (scripts/import-b/modules/<B module id>.json by default) may carry:
  aModule, stage, order (entry ids in the order they should be emitted),
  fixes (a list of {file, find, replace} confirmed corrections to B's text),
  parts ({id: {"1.1": {kind, requirement}}}), indexPrompt ({id: prompt}),
  indexAnswer ({id: answer}, an MCQ card's answer) and source ({id: label},
  when a card's source label is not the default).

Index cards, the lesson focus card and site search print a card's prompt and
answer with escapeHtml and no markdown, so $...$ math and `code` reach the
reader as literal characters. The tool warns whenever an emitted index field
still holds either (and exits 1 under --strict); the fix is an indexPrompt or
indexAnswer entry in the spec.

Warnings go to stderr. With --strict the tool exits 1 on the first warning.

The stage tag comes from --stage if it is given, else the spec file's "stage",
else the stage content/chapters.json puts the A module in. If none of the three
gives a stage the tool exits 2 rather than silently tagging the module
"Round 1".
"""
import argparse
import json
import re
import sys
from pathlib import Path

B_ROOT = Path("/home/jackwu/Desktop/Repos/USAAIO-Guide")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_DIR = Path(__file__).resolve().parent / "modules"
CHAPTERS = REPO_ROOT / "content" / "chapters.json"
STAGE_KEYS = {"round1": "Round 1", "round2": "Round 2",
              "camp": "Camp", "advanced": "Advanced"}

ALLOWED_DIFFICULTIES = {"Easy", "Medium", "Hard", "Very hard", "Very Hard"}

CHOICE = re.compile(r'<Choice( correct)? explain="([^"]*)">(.*?)</Choice>', re.S)
# <Part>, <Part answer="3">, <Part answer="3" tolerance={0.001}> and any
# attribute order in between.
PARTRE = re.compile(
    r'(?P<tag><Part[^>]*>)\s*(?P<prompt>.*?)\s*<Solution>\s*(?P<solution>.*?)\s*</Solution>\s*</Part>',
    re.S,
)
ANSWER_ATTR = re.compile(r'\banswer="([^"]*)"')
TOLERANCE_ATTR = re.compile(r'\btolerance=\{([^}]*)\}')
KNOWN_PART_ATTRS = re.compile(r'\s+(?:answer="[^"]*"|tolerance=\{[^}]*\})')


def part_attrs(tag, warn, where):
    """Read a <Part ...> tag's answer and tolerance, or warn about the shape."""
    known = KNOWN_PART_ATTRS.sub("", tag[len("<Part"):]).rstrip(">").strip()
    if known:
        warn(f"{where}: unknown <Part> attribute(s) {known!r}")
    answer = ANSWER_ATTR.search(tag)
    tolerance = TOLERANCE_ATTR.search(tag)
    return (answer.group(1) if answer else None), (tolerance.group(1) if tolerance else None)
DISPLAY_MATH = "$$"


def frontmatter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("no frontmatter delimiter")
    fm = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [x.strip() for x in value[1:-1].split(",") if x.strip()]
        fm[key.strip()] = value
    return fm, text[m.end():]


def cell_tex(cell):
    """One markdown table cell to a KaTeX array cell."""
    cell = cell.strip()
    if re.fullmatch(r"\$[^$]+\$", cell):
        return cell[1:-1]
    if re.fullmatch(r"-?\d+(\.\d+)?", cell):
        return cell
    return r"\text{" + cell + "}"


def table_to_tex(lines):
    rows = [line.strip().strip("|").split("|") for line in lines]
    head, body = rows[0], rows[2:]
    out = [" & ".join(cell_tex(c) for c in head) + r" \\ \hline"]
    out += [" & ".join(cell_tex(c) for c in row) + r" \\" for row in body]
    out[-1] = out[-1][:-3]
    return r"\begin{array}{" + "c" * len(head) + "}\n" + "\n".join(out) + "\n" + r"\end{array}"


def strip_center_tags(body, warn, where=""):
    """<center>...</center> only ever wraps a markdown table on B's site, and A
    renders a table as a centered KaTeX array, so the tags carry nothing."""
    lines = body.split("\n")
    out, inside = [], 0
    for line in lines:
        stripped = line.strip()
        if stripped == "<center>":
            inside += 1
            continue
        if stripped == "</center>":
            if inside:
                inside -= 1
            else:
                warn(f"{where}: a </center> with no opening <center>")
            continue
        if inside and not stripped.startswith("|") and stripped:
            warn(f"{where}: a <center> block holds something other than a table")
        out.append(line)
    if inside:
        warn(f"{where}: a <center> block is never closed")
    return "\n".join(out)


def blocks(md, warn, where):
    """B's markdown body to A's block list.

    Every pass of the loop must move `i` forward. An older fork of this
    function had a numbered-list case that only warned and then fell through
    to the paragraph collector, which rejected the line, so `i` stayed put and
    the loop appended empty paragraphs until the process was killed on memory.
    The `stuck` check below turns any such future regression into a hard error.
    """
    out, lines, i, prev = [], md.strip("\n").split("\n"), 0, -1
    while i < len(lines):
        if i == prev:
            raise SystemExit(f"{where}: block parsing is stuck at {lines[i]!r} "
                             "(a pass produced output without advancing the cursor)")
        prev = i
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip() == DISPLAY_MATH:
            j = i + 1
            while j < len(lines) and lines[j].strip() != DISPLAY_MATH:
                j += 1
            if j >= len(lines):
                warn(f"{where}: a $$ block is never closed")
                out.append({"tex": "\n".join(lines[i + 1:]).strip()})
                return out
            out.append({"tex": "\n".join(lines[i + 1:j]).strip()})
            i = j + 1
            continue
        if line.lstrip().startswith("```"):
            lang = line.strip()[3:]
            j = i + 1
            while j < len(lines) and lines[j].strip() != "```":
                j += 1
            if j >= len(lines):
                warn(f"{where}: a code fence is never closed")
                out.append({"code": {"language": lang, "lines": lines[i + 1:]}})
                return out
            out.append({"code": {"language": lang, "lines": lines[i + 1:j]}})
            i = j + 1
            continue
        if line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append({"tex": table_to_tex(lines[i:j])})
            i = j
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or (lines[i].startswith("  ") and items)):
                if lines[i].startswith("- "):
                    items.append(lines[i][2:].strip())
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append({"list": items})
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(re.sub(r"^\d+\. ", "", lines[i]).strip())
                i += 1
            out.append({"olist": items})
            continue
        if line.startswith("```"):
            language = line.strip()[3:]
            if language and language not in ("python", "text"):
                warn(f"{where}: a code fence names a language {language!r} A has no highlighter for")
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j >= len(lines):
                warn(f"{where}: a code fence is never closed")
                out.append({"code": {"language": language or "python",
                                     "lines": [l.rstrip() for l in lines[i + 1:]]}})
                return out
            out.append({"code": {"language": language or "python",
                                 "lines": [l.rstrip() for l in lines[i + 1:j]]}})
            i = j + 1
            continue
        if line.startswith(">"):
            warn(f"{where}: a blockquote line is kept as prose ({line[:40]}...)")
        if line.strip() in ("<center>", "</center>"):
            # The caller strips these (strip_center_tags); reaching the parser
            # means that pass was skipped, which used to leave the tags as
            # standalone prose paragraphs. Drop them, but say so.
            warn(f"{where}: a <center> tag reached the block parser; "
                 "strip_center_tags should have removed it")
            i += 1
            continue
        if line.startswith("<"):
            warn(f"{where}: a line starting with < is kept as prose ({line[:40]}...)")
        j, para = i, []
        while j < len(lines) and lines[j].strip() and lines[j].strip() != DISPLAY_MATH \
                and not lines[j].lstrip().startswith("|") and not lines[j].startswith("- ") \
                and not lines[j].lstrip().startswith("```") \
                and not re.match(r"^\d+\. ", lines[j]) and not lines[j].startswith(">"):
            para.append(lines[j].strip())
            j += 1
        if para:
            out.append({"p": " ".join(para)})
        else:
            warn(f"{where}: a line is not any block A knows ({line[:40]}...)")
            i += 1
            continue
        i = j
    return out


GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa",
    "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi",
    "varpi": "pi", "rho": "rho", "varrho": "rho", "sigma": "sigma",
    "varsigma": "sigma", "tau": "tau", "upsilon": "upsilon", "phi": "phi",
    "varphi": "phi", "chi": "chi", "psi": "psi", "omega": "omega",
    "Gamma": "Gamma", "Delta": "Delta", "Theta": "Theta", "Lambda": "Lambda",
    "Xi": "Xi", "Pi": "Pi", "Sigma": "Sigma", "Upsilon": "Upsilon",
    "Phi": "Phi", "Psi": "Psi", "Omega": "Omega",
}
SYMBOLS = {
    "times": "x", "cdot": ".", "cdots": "...", "ldots": "...", "dots": "...",
    "ge": ">=", "geq": ">=", "le": "<=", "leq": "<=", "ne": "!=", "neq": "!=",
    "equiv": "==", "approx": "~", "sim": "~", "simeq": "~", "cong": "~",
    "propto": "proportional to", "pm": "+-", "mp": "-+", "in": " in ",
    "notin": " not in ", "subset": "subset of", "supset": "superset of",
    "subseteq": "subset of", "supseteq": "superset of", "cup": " or ",
    "cap": " and ", "emptyset": "empty", "infty": "infinity",
    "partial": "partial", "nabla": "nabla", "forall": "for all",
    "exists": "exists", "nexists": "does not exist", "to": "->",
    "rightarrow": "->", "leftarrow": "<-", "Rightarrow": "=>",
    "Leftarrow": "<=", "leftrightarrow": "<->", "mapsto": "maps to",
    "mid": "|", "nmid": "not |", "parallel": "parallel to", "perp": "perp",
    "angle": "angle", "prime": "'", "circ": " degrees", "det": "det",
    "lVert": "||", "rVert": "||", "Vert": "||", "lvert": "|", "rvert": "|",
    "vert": "|", "langle": "<", "rangle": ">", "lfloor": "[", "rfloor": "]",
    "lceil": "[", "rceil": "]", "aleph": "aleph", "hbar": "hbar",
    "ell": "ell", "Re": "Re", "Im": "Im", "arg": "arg", "dim": "dim",
    "exp": "exp", "ln": "ln", "log": "log", "lg": "log", "cos": "cos",
    "sin": "sin", "tan": "tan", "sec": "sec", "csc": "csc", "cot": "cot",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arccos": "arccos",
    "arcsin": "arcsin", "arctan": "arctan", "max": "max", "min": "min",
    "sup": "sup", "inf": "inf", "gcd": "gcd", "lim": "lim", "Pr": "Pr",
    "bmod": " mod ", "pmod": " mod ", "top": "T", "sum": "sum",
    "prod": "product", "int": "integral", "sqrt": "sqrt", "partial": "partial ",
    "ldots": "...", "dots": "...",
}
# An accent becomes a Unicode combining mark on the letter it decorates, so
# \hat{x} keeps its meaning as a different symbol from x.
ACCENTS = {
    "hat": "̂", "widehat": "̂", "bar": "̄", "overline": "̄", "tilde": "̃",
    "vec": "⃗", "dot": "̇", "ddot": "̈", "underline": "",
}
ACCENT_PATTERN = re.compile(r"\\(?P<cmd>%s)\{([^{}]*)\}" % "|".join(ACCENTS))
# Spacing commands: KaTeX spacing is meaningless in plain text.
SPACING = {
    ",": " ", ";": " ", ":": " ", "!": "", "quad": " ", "qquad": "  ",
    "thinspace": " ", "medspace": " ", "thickspace": " ",
}
# Commands whose argument is the content, and braces are dropped.
CONTENT_COMMANDS = (
    "text", "operatorname", "mathbf", "mathcal", "mathbb", "mathrm",
    "boldsymbol", "mathit", "mathscr", "mathsf", "mathtt",
)
# Decoration that carries no text.
DECORATION = (
    "big", "Big", "bigg", "Bigg", "biggl", "Biggl", "biggr", "Biggr",
    "bigl", "Bigl", "bigr", "Bigr", "left", "right", "displaystyle",
    "limits", "nolimits", "scriptstyle", "textstyle", "rm", "it", "bf", "sf", "tt",
)
MATH_COMMAND = re.compile(r"\\([A-Za-z]+)")


def math_to_plain(tex):
    """One inline math span as the plain text the index card shows. KaTeX never
    runs on a card, so commands have to be flattened; anything unrecognised
    keeps its backslash so plain() can warn about it."""
    s = tex.replace("\\\\", " ").replace("\\ ", " ")
    s = s.replace("\\{", "{").replace("\\}", "}").replace("\\|", "||")
    s = re.sub(r"\\([,;:!])", lambda m: SPACING.get(m.group(1), " "), s)
    s = re.sub(r"\\(quad|qquad|thinspace|medspace|thickspace)\b",
               lambda m: SPACING.get(m.group(1), " "), s)
    s = re.sub(r"\\begin\{[a-zA-Z*]+\}", "(", s)
    s = re.sub(r"\\end\{[a-zA-Z*]+\}", ")", s)
    # Innermost braces first, then repeat so nested commands unwrap from the
    # inside out (\hat{\mathbf{y}} -> \hat{y} -> y).
    patterns = [
        (re.compile(r"\\(?:%s)\{([^{}]*)\}" % "|".join(CONTENT_COMMANDS)), r"\1"),
        (re.compile(r"\\(?:[dt]?frac)\{([^{}]*)\}\{([^{}]*)\}"), r"\1/\2"),
        (re.compile(r"\\(?:[dt]?frac)(\d)(\d)"), r"\1/\2"),
        (re.compile(r"\\sqrt\{([^{}]*)\}"), r"sqrt(\1)"),
        (re.compile(r"\\sqrt\[(\d+)\]\{([^{}]*)\}"), r"\2^(1/\1)"),
        (ACCENT_PATTERN, lambda m: m.group(2) + ACCENTS[m.group("cmd")]),
        (re.compile(r"\\(?:%s)\\?!?" % "|".join(DECORATION)), ""),
    ]
    for _ in range(12):
        new = s
        for pattern, repl in patterns:
            new = pattern.sub(repl, new)
        if new == s:
            break
        s = new

    def replace(match):
        # A command may sit flush against the next word once the braces around
        # a neighbour are gone (\top\mathbf{x} -> \topx), so match the longest
        # command prefix the tables know and keep the leftover letters.
        name = match.group(1)
        for i in range(len(name), 0, -1):
            head, tail = name[:i], name[i:]
            if head in GREEK:
                return GREEK[head] + (" " + tail if tail else "")
            if head in SYMBOLS:
                value = SYMBOLS[head]
                if tail and value and value[-1].isalpha():
                    return value + " " + tail
                return value + tail
        return match.group(0)

    s = MATH_COMMAND.sub(replace, s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s)


def plain(text, warn=None, where=""):
    """Math and code markup to the plain text the index card shows."""
    def flatten(match):
        out = math_to_plain(match.group(1))
        if "\\" in out and warn:
            warn(f"{where}: index text keeps LaTeX the flattener does not know ({out[:60]}...)")
        return out

    s = re.sub(r"\$([^$]*)\$", flatten, text).replace("`", "")
    if warn and "\\" in s:
        warn(f"{where}: index text still has a backslash after flattening ({s[:60]}...)")
    return s


def warn_index_markup(pid, field, text, spec_key, warn):
    """A card's prompt and answer are printed with escapeHtml and no markdown,
    so $math$ and `code` show up as literal characters. The spec key is the
    escape hatch the warning names."""
    if "$" in text or "`" in text:
        warn(f"{pid}: index {field} has math or code markup; "
             f"write {spec_key} in the spec ({text[:50]}...)")


def result_block(answer):
    """B's answer="..." to A's closing result block."""
    return {"result": f"${answer}$" if answer.startswith("-") else answer}


def theory_entry(pid, fm, body, spec, warn):
    difficulty = "Very Hard" if fm["difficulty"] == "Very hard" else fm["difficulty"]
    if difficulty not in ALLOWED_DIFFICULTIES:
        warn(f"{pid}: difficulty {fm['difficulty']!r} is not one A uses")
    entry = {
        "id": pid,
        "moduleId": spec["aModule"],
        "title": fm["title"],
        "difficulty": difficulty,
        "minutes": int(fm["minutes"]),
    }
    if fm["kind"] == "mcq":
        stem = blocks(body[: body.index("<Choice")], warn, pid)
        if not stem:
            warn(f"{pid}: no prompt text before the first <Choice>")
        choices = []
        for correct, explain, text in CHOICE.findall(body):
            choices.append({
                "text": text.strip(),
                **({"correct": True} if correct else {}),
                "explain": explain.strip(),
            })
        if len(choices) < 2:
            warn(f"{pid}: {len(choices)} choice(s), fewer than 2")
        if sum(1 for c in choices if c.get("correct")) != 1:
            warn(f"{pid}: {sum(1 for c in choices if c.get('correct'))} correct choice(s), want 1")
        right = next((c for c in choices if c.get("correct")), choices[0] if choices else None)
        entry["overview"] = []
        entry["parts"] = [{
            "id": "1.1",
            "kind": "mcq",
            "prompt": stem + [{"choices": choices}],
            "answer": [{"p": right["explain"]}, {"result": right["text"]}] if right else [],
        }]
        return entry

    intro = blocks(body[: body.index("<Part")], warn, pid)
    if intro and "p" in intro[0]:
        entry["overview"] = [intro[0]["p"]]
        rest = intro[1:]
    else:
        entry["overview"] = []
        rest = intro
    if rest:
        entry["setup"] = rest
    if not entry["overview"]:
        warn(f"{pid}: the problem does not open with a paragraph")
    if len(entry.get("setup", [])) > 4:
        warn(f"{pid}: {len(entry.get('setup', []))} setup blocks, check the split")
    part_specs = spec.get("parts", {}).get(pid, {})
    entry["parts"] = []
    for n, match in enumerate(PARTRE.finditer(body), 1):
        answer, tolerance = part_attrs(match.group("tag"), warn, f"{pid} part {n}")
        if answer is not None and tolerance is None and re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", answer):
            warn(f"{pid} part {n}: a numeric answer without a tolerance")
        override = part_specs.get(f"1.{n}")
        if override:
            kind = override.get("kind", "computation" if answer is not None else "derivation")
            requirement = override.get(
                "requirement",
                "Reasoning is not required." if answer is not None else "A complete derivation is required.",
            )
        else:
            kind = "computation" if answer is not None else "derivation"
            requirement = "Reasoning is not required." if answer is not None \
                else "A complete derivation is required."
            warn(f"{pid} part 1.{n}: no kind/requirement in the spec, using the default "
                 f"({kind!r}, {requirement!r})")
        answer_blocks = blocks(match.group("solution"), warn, f"{pid} part {n}")
        if answer is not None:
            answer_blocks.append(result_block(answer))
        entry["parts"].append({
            "id": f"1.{n}",
            "kind": kind,
            "requirement": requirement,
            "prompt": blocks(match.group("prompt"), warn, f"{pid} part {n}"),
            "answer": answer_blocks,
        })
    if len(entry["parts"]) != body.count("<Part"):
        warn(f"{pid}: {body.count('<Part>')} <Part> tags but {len(entry['parts'])} parsed")
    if not entry["parts"]:
        warn(f"{pid}: no parts parsed")
    return entry


def index_entry(pid, fm, theory, body, spec, warn):
    default_source = "Custom" if fm["kind"] == "mcq" else "USAAIO-style"
    source = spec.get("source", {}).get(pid, default_source)
    entry = {
        "id": pid,
        "moduleId": spec["aModule"],
        "title": fm["title"],
        "source": source,
        "difficulty": "Very Hard" if fm["difficulty"] == "Very hard" else fm["difficulty"],
        "url": f"/theory/?id={pid}",
    }
    if source != default_source and pid not in spec.get("source", {}):
        warn(f"{pid}: source {source!r} is not A's default for a {fm['kind']} problem ({default_source!r})")
    if fm["kind"] == "mcq":
        stem = blocks(body[: body.index("<Choice")], warn, pid)
        right = next((c for c in theory["parts"][0]["prompt"][-1]["choices"] if c.get("correct")), None)
        if pid in spec.get("indexAnswer", {}):
            answer = spec["indexAnswer"][pid].strip()
        elif right is None:
            warn(f"{pid}: no correct choice, the index answer is empty")
            answer = ""
        else:
            text = plain(right["text"], warn, pid).strip()
            explain = plain(right["explain"], warn, pid).strip()
            answer = (text if text.endswith(".") else text + ".") + " " + explain
        if pid in spec.get("indexPrompt", {}):
            prompt = spec["indexPrompt"][pid]
        else:
            skipped = [b for b in stem if "p" not in b]
            if skipped:
                warn(f"{pid}: the index prompt drops {len(skipped)} non-paragraph block(s) from the stem")
            prompt = plain(" ".join(b["p"] for b in stem if "p" in b), warn, pid).strip()
        entry.update({
            "tags": ["Theory", "Multiple choice", spec["stage"]],
            "prompt": prompt,
            "answer": answer,
        })
        return entry

    intro = blocks(body[: body.index("<Part")], warn, pid)
    paragraphs = [b["p"] for b in intro if "p" in b]
    skipped = [b for b in intro if "p" not in b]
    if pid in spec.get("indexPrompt", {}):
        prompt = spec["indexPrompt"][pid]
    else:
        prompt = " ".join(paragraphs)
        if skipped:
            warn(f"{pid}: the index prompt drops {len(skipped)} non-paragraph block(s) from the intro")
        warn(f"{pid}: no indexPrompt in the spec, using the opening paragraph(s)")
    entry.update({
        "tags": ["Theory", "Multi-part", spec["stage"]],
        "prompt": (prompt.strip() + f" {len(theory['parts'])} parts."),
        "answer": "Worked solutions are given part by part on the problem page. "
                  "Attempt each part on paper before revealing it.",
    })
    return entry


def convert(path, spec, warn):
    text = path.read_text()
    fm, body = frontmatter(text)
    for fix in spec.get("fixes", []):
        if fix["file"] != path.stem:
            continue
        if fix["find"] not in body:
            warn(f"{path.stem}: a spec fix does not match any text ({fix['find'][:40]}...)")
            continue
        body = body.replace(fix["find"], fix["replace"], 1)
    body = strip_center_tags(body, warn, path.stem)
    theory = theory_entry(path.stem, fm, body, spec, warn)
    index = index_entry(path.stem, fm, theory, body, spec, warn)
    for field, spec_key in (("prompt", "indexPrompt"), ("answer", "indexAnswer")):
        warn_index_markup(path.stem, field, index.get(field, ""), spec_key, warn)
    return theory, index


def stage_in_chapters(a_module):
    """The stage content/chapters.json puts an A module in, or None."""
    try:
        tree = json.loads(CHAPTERS.read_text())
    except (OSError, ValueError):
        return None
    for key, label in STAGE_KEYS.items():
        for chapter in tree.get(key, []):
            if a_module in chapter.get("items", []):
                return label
    return None


def load_spec(bmodule, args):
    spec_path = Path(args.spec) if args.spec else SPEC_DIR / f"{bmodule}.json"
    spec = {"aModule": args.module, "stage": args.stage, "parts": {},
            "indexPrompt": {}, "indexAnswer": {}, "fixes": [], "source": {}}
    if spec_path.exists():
        spec.update(json.loads(spec_path.read_text()))
    if args.module:
        spec["aModule"] = args.module
    if not spec.get("aModule"):
        spec["aModule"] = bmodule
        print(f"warning: {bmodule} has no A module id in a spec file or --module; "
              f"writing it under B's own id {bmodule!r} as a placeholder", file=sys.stderr)

    # --stage used to default to "Round 1", so it always clobbered the spec
    # file's stage and tagged every Round 2, Camp and Advanced entry "Round 1".
    # It now defaults to None: --stage wins, then the spec, then chapters.json,
    # and if none of them names a stage that is an error, not a silent default.
    spec_stage = spec.get("stage")
    chapters_stage = stage_in_chapters(spec["aModule"])
    if args.stage:
        stage = args.stage
    elif spec_stage is not None:
        stage = spec_stage
    else:
        stage = chapters_stage
    if stage is None:
        print(f"{bmodule}: no stage for A module {spec['aModule']} "
              f"(not in --stage, the spec or content/chapters.json)", file=sys.stderr)
        sys.exit(2)
    warnings = []
    if spec_stage is not None and chapters_stage is not None and spec_stage != chapters_stage:
        warnings.append(f"{bmodule}: the spec's stage {spec_stage!r} disagrees with "
                        f"content/chapters.json, which puts {spec['aModule']} in "
                        f"{chapters_stage!r}")
    spec["stage"] = stage
    return spec, spec_path.exists(), warnings


def merge_into(out, dest):
    """Fold this run's entries into an existing per-module file.

    A module file may already hold A's kept entries (and notebook cards from
    the notebook converter), so the run's entries replace the ones with the
    same id in place and the ids the file does not have are appended after the
    file's own entries, in the order the run emits them (B's order). Every
    other entry and every other key (module, order, stage, ...) is left alone.
    """
    try:
        file = json.loads(dest.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"{dest}: the module file is not valid JSON ({exc})")
    if not isinstance(file, dict):
        sys.exit(f"{dest}: the module file is not a JSON object")

    existing_module = file.get("module")
    if existing_module and existing_module != out["module"]:
        sys.exit(f"{dest}: holds module {existing_module!r}, this import is for "
                 f"{out['module']!r}; a module file holds one module")

    for field in ("index", "theory"):
        incoming = out[field]
        existing = file.get(field)
        if not isinstance(existing, list):
            file[field] = incoming
            continue
        incoming_by_id = {e["id"]: e for e in incoming if "id" in e}
        if len(incoming_by_id) != len(incoming):
            sys.exit(f"this run emits two entries with the same id in {field}")
        seen = set()
        merged = []
        for entry in existing:
            eid = entry.get("id") if isinstance(entry, dict) else None
            if eid in incoming_by_id:
                merged.append(incoming_by_id[eid])
                seen.add(eid)
            else:
                merged.append(entry)
        merged.extend(e for e in incoming if e.get("id") not in seen)
        file[field] = merged
    return file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bmodule")
    parser.add_argument("--root", default=str(B_ROOT))
    parser.add_argument("--out", help="output file, or a directory to write <A module id>.json into")
    parser.add_argument("--into", help="content/problems/<id>.json to merge this run's entries into "
                                       "(created if it does not exist)")
    parser.add_argument("--module", help="A module id (overrides the spec file)")
    parser.add_argument("--stage", default=None,
                        help="stage tag (default: the spec's stage, else the module's stage "
                             "in content/chapters.json)")
    parser.add_argument("--spec", help="path to the module spec file")
    parser.add_argument("--strict", action="store_true", help="exit 1 on the first warning")
    args = parser.parse_args(argv)

    problems_dir = Path(args.root) / "content" / "problems" / args.bmodule
    if not problems_dir.is_dir():
        sys.exit(f"no such B problem folder: {problems_dir}")
    files = sorted(problems_dir.glob("*.mdx"))
    if not files:
        sys.exit(f"{problems_dir} has no .mdx files")

    spec, had_spec, spec_warnings = load_spec(args.bmodule, args)
    warnings = []
    def warn(message):
        warnings.append(f"{args.bmodule}: {message}")
    warnings.extend(spec_warnings)
    if not had_spec:
        warn(f"no spec file at scripts/import-b/modules/{args.bmodule}.json "
             f"(module id, stage, part kinds and index prompts are all defaults)")

    by_order, entries = {}, []
    for path in files:
        theory, index = convert(path, spec, warn)
        by_order[path.stem] = (theory, index)
        entries.append((path, theory, index))

    order = spec.get("order")
    if order:
        missing = [pid for pid in order if pid not in by_order]
        extra = [pid for pid in by_order if pid not in order]
        if missing or extra:
            warn(f"the spec order list does not match the folder "
                 f"(missing {missing}, unexpected {extra})")
        entries = [(path, *by_order[pid]) for pid in order for path in [Path(problems_dir / f"{pid}.mdx")]]

    out = {
        "module": spec["aModule"],
        "stage": spec["stage"],
        "index": [index for _, _, index in entries],
        "theory": [theory for _, theory, _ in entries],
    }
    payload = json.dumps(out, indent=2, ensure_ascii=False) + "\n"

    if args.into and args.out:
        sys.exit("pass only one of --into and --out")
    if args.into:
        dest = Path(args.into)
        if dest.exists():
            out = merge_into(out, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    elif args.out:
        dest = Path(args.out)
        if not dest.suffix:
            dest = dest / f"{spec['aModule']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload)
    else:
        sys.stdout.write(payload)

    for message in warnings:
        print(f"warning: {message}", file=sys.stderr)
    if warnings:
        print(f"{args.bmodule}: {len(files)} file(s), {len(warnings)} warning(s)", file=sys.stderr)
        if args.strict:
            sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
