#!/usr/bin/env python3
"""check_verbatim.py - the stage-2 verbatim gate for the lesson lane.

Usage:
  python3 check_verbatim.py <B lesson.mdx> <A lesson.md> [--json] [--links links.json]

The contract for scripts/import-b/lesson.mjs is that every sentence of the A
lesson appears word for word in its B lesson, and every sentence of the B
lesson survives the import. This script proves it, and reports the differences
the importer is allowed to make in their own sections instead of counting them
as changed sentences:

  header      A's H1 / Goal / Stage / Prerequisites against B's frontmatter
  labels      sentences that differ only in a link's label or target
  figures     <Figure> src / alt / caption triples (B's leading ./ is mechanical)
  code        fenced blocks: bodies compared whole, language relabelling noted
  math        display-math blocks, compared whole
  tables      table blocks, compared whole
  cards       ProblemCard / NotebookCard ids, in order
  resources   <Resource> entries
  headings    ## / ### headings, unordered (A moves ## Resources to the end)
  tags        every other component tag with its attributes

Sections other than "sentences" never fail the gate on their own; they are the
inventory a reviewer reads before approving a lesson. The gate is the pair of
sentence counts: changed (in A but not word for word in B) and missing (in B
but gone from A), both of which are zero for an import that changed nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- regexes ----

FRONTMATTER_KEY = re.compile(r"^([A-Za-z_-]+):\s*(.*)$")
FRONTMATTER_ITEM = re.compile(r"^\s+-\s+(\S+)\s*$")
TAG = re.compile(r"^\s*<(/?)([A-Za-z][A-Za-z0-9]*)(.*?)(/?)>\s*$")
ATTRS = re.compile(r"([A-Za-z-]+)=" r'"([^"]*)"')
MARKDOWN_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
BULLET = re.compile(r"^[-*]\s+(.*)$")
ORDERED = re.compile(r"^\d+\.\s+(.*)$")
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
INLINE_MATH = re.compile(r"\$[^$\n]+\$")
CODE_SPAN = re.compile(r"`[^`\n]+`")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(\[\x01\x02])")
# B's <EmbeddedPyEditor> keeps its Python in an initialCode template literal.
INITIALCODE = re.compile(r"^\s*initialCode=\{`(.*)$")

# Tags whose body becomes a figure caption rather than lesson prose.
FIGURE_OPENERS = ("Figure", "FigureBox")
# Tags A's reader parses into component blocks whose ids or entries are
# compared as units instead of as prose.
CARD_TAGS = ("ProblemCard", "NotebookCard")
# The importer relabels these fence languages mechanically.
KNOWN_LANG_CHANGES = {("py", "python"), ("", "output")}


def attrs_of(rest: str) -> dict[str, str]:
    return dict(ATTRS.findall(rest))


def mask_inline(text: str) -> str:
    """Hide inline math and code spans so a period inside one cannot break a
    sentence. Identical math on both sides yields an identical placeholder, so
    changed math still shows up as a changed sentence."""
    text = INLINE_MATH.sub(" \x01 ", text)
    return CODE_SPAN.sub(" \x02 ", text)


def sentences(paragraphs: list[str]) -> list[str]:
    """Split prose into sentences. Links are stashed before splitting and
    restored afterwards: a link label may hold a period (B writes
    "[Build a Generative Model: 2. VAE](...)"), and splitting there would leave
    the tail of the link in a sentence of its own, where it no longer matches
    the link mask and would look like a change."""
    out: list[str] = []
    for para in paragraphs:
        links: list[str] = []

        def stash(match: re.Match) -> str:
            links.append(match.group(0))
            return f"\x04{len(links) - 1}\x04"

        masked = LINK.sub(stash, mask_inline(para))
        for piece in SENTENCE_SPLIT.split(masked):
            piece = piece.strip()
            if piece:
                piece = re.sub(
                    r"\x04(\d+)\x04",
                    lambda m: links[int(m.group(1))],
                    re.sub(r"\s+", " ", piece),
                )
                out.append(piece)
    return out


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def link_key(text: str) -> str:
    """The sentence with every link reduced to a placeholder, so a remapped
    label or target is not a changed sentence."""
    return norm_key(LINK.sub("\x03", text))


def match(a_list: list[str], b_list: list[str], key) -> tuple[list[bool], list[bool]]:
    b_used = [False] * len(b_list)
    a_matched = [False] * len(a_list)
    for ai, a in enumerate(a_list):
        ka = key(a)
        for bi, b in enumerate(b_list):
            if not b_used[bi] and key(b) == ka:
                b_used[bi] = True
                a_matched[ai] = True
                break
    return a_matched, b_used


# ----------------------------------------------------------------- parsing ----


def parse_frontmatter(lines: list[str], start: int) -> tuple[dict, int]:
    front: dict = {}
    i = start
    n = len(lines)
    while i < n and lines[i].strip() != "---":
        key_match = FRONTMATTER_KEY.match(lines[i])
        if key_match:
            key, value = key_match.group(1), key_match.group(2).strip()
            if value:
                front[key] = value.strip('"')
            else:
                items = []
                j = i + 1
                while j < n and FRONTMATTER_ITEM.match(lines[j]):
                    items.append(FRONTMATTER_ITEM.match(lines[j]).group(1))
                    j += 1
                front[key] = items
                i = j - 1
        i += 1
    return front, i + 1


def parse_header_a(lines: list[str]) -> tuple[dict, int]:
    """A's preamble: the H1, the Goal/Stage quote lines, then ## Prerequisites."""
    header: dict = {"prerequisites": []}
    i = 0
    n = len(lines)
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].startswith("# "):
        header["h1"] = lines[i][2:].strip()
        i += 1
    while i < n and not lines[i].strip():
        i += 1
    quote = []
    while i < n and lines[i].startswith(">"):
        quote.append(lines[i].lstrip("> ").strip())
        i += 1
    for line in quote:
        if line.startswith("**Goal:**"):
            header["goal"] = line[len("**Goal:**"):].strip()
        elif line.startswith("**Stage:**"):
            header["stage"] = line[len("**Stage:**"):].strip()
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].strip() == "## Prerequisites":
        i += 1
        while i < n and not HEADING.match(lines[i]):
            plain = MARKDOWN_LINK_LABEL.match(lines[i].strip())
            if plain:
                header["prerequisites"].append((plain.group(1), plain.group(2)))
            elif not lines[i].strip():
                pass
            elif lines[i].strip().startswith("-"):
                header["prerequisites"].append((lines[i].strip().lstrip("- ").strip(), ""))
            i += 1
    return header, i


MARKDOWN_LINK_LABEL = re.compile(r"^-\s+\[([^\]]*)\]\(([^)]*)\)\s*$")


def parse_lesson(path: Path, is_mdx: bool) -> dict:
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    units: dict = {
        "frontmatter": {},
        "header": {},
        "headings": [],
        "prose": [],
        "code": [],
        "math": [],
        "tables": [],
        "figures": [],
        "cards": [],
        "resources": [],
        "tags": [],
    }
    n = len(lines)
    i = 0

    if is_mdx:
        while i < n and not lines[i].strip():
            i += 1
        if i < n and lines[i].strip() == "---":
            front, i = parse_frontmatter(lines, i + 1)
            units["frontmatter"] = front
    else:
        header, i = parse_header_a(lines)
        units["header"] = header

    prose_buf: list[str] = []
    list_items: list[str] = []
    list_ordered: bool | None = None
    figure: dict | None = None
    prev_kind = "blank"

    def flush_prose() -> None:
        nonlocal prose_buf, prev_kind
        text = " ".join(line.strip() for line in prose_buf if line.strip()).strip()
        if text:
            units["prose"].append(text)
            prev_kind = "prose"
        prose_buf = []

    def flush_list() -> None:
        nonlocal list_items, list_ordered, prev_kind
        if list_items:
            units["prose"].extend(list_items)
            prev_kind = "list"
        list_items = []
        list_ordered = None

    def flush_figure() -> None:
        nonlocal figure, prev_kind
        if figure is not None:
            caption = " ".join(figure.pop("_caption_lines", [])).strip()
            figure["caption"] = caption
            units["figures"].append(figure)
            prev_kind = "other"
        figure = None

    while i < n:
        line = lines[i]
        trimmed = line.strip()

        if figure is not None:
            tag = TAG.match(trimmed)
            if tag and tag.group(1) == "/" and tag.group(2) in FIGURE_OPENERS:
                flush_figure()
                i += 1
                continue
            inner = TAG.match(trimmed)
            if inner and inner.group(2) == "img" and not inner.group(1):
                inner_attrs = attrs_of(inner.group(3))
                figure.setdefault("src", inner_attrs.get("src", inner_attrs.get("image", "")))
                figure.setdefault("alt", inner_attrs.get("alt", ""))
                if inner_attrs.get("imageDark"):
                    figure["imageDark"] = inner_attrs["imageDark"]
                i += 1
                continue
            if trimmed:
                figure.setdefault("_caption_lines", []).append(trimmed)
            i += 1
            continue

        if not trimmed:
            flush_prose()
            flush_list()
            prev_kind = "blank"
            i += 1
            continue

        if trimmed.startswith("```"):
            flush_prose()
            flush_list()
            lang = trimmed[3:].strip()
            body: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                body.append(lines[i].rstrip())
                i += 1
            units["code"].append({"lang": lang, "body": "\n".join(body)})
            prev_kind = "other"
            i += 1
            continue

        if trimmed == "$$" or (trimmed.startswith("$$") and trimmed.endswith("$$")):
            flush_prose()
            flush_list()
            math_lines: list[str] = []
            if trimmed == "$$":
                i += 1
                while i < n and lines[i].strip() != "$$":
                    math_lines.append(lines[i].rstrip())
                    i += 1
            else:
                math_lines.append(trimmed[2:-2])
            units["math"].append("\n".join(math_lines).strip())
            prev_kind = "other"
            i += 1
            continue

        if trimmed.startswith("|"):
            flush_prose()
            flush_list()
            table_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            units["tables"].append("\n".join(table_lines))
            prev_kind = "other"
            continue

        if trimmed.startswith("<!--"):
            flush_prose()
            flush_list()
            while i < n and "-->" not in lines[i]:
                i += 1
            prev_kind = "other"
            i += 1
            continue

        heading_match = HEADING.match(trimmed)
        if heading_match and not (not is_mdx and heading_match.group(1) == "#" and not units["headings"] and not units["prose"]):
            flush_prose()
            flush_list()
            level, text = heading_match.group(1), heading_match.group(2)
            if level == "#" and is_mdx:
                units["headings"].append((len(level), text))
            elif level != "#":
                units["headings"].append((len(level), text))
            prev_kind = "other"
            i += 1
            continue

        if is_mdx and trimmed.startswith("<EmbeddedPyEditor"):
            # B opens this tag on a line of its own and puts the Python code in
            # an initialCode template literal over the following lines, so the
            # line never matches TAG. Read the code out as a code unit and skip
            # to the tag's self-closing line; A's file holds the same code as
            # the body of a matching tag.
            flush_prose()
            flush_list()
            code_lines: list[str] = []
            j = i
            head = None
            while j < n:
                m = INITIALCODE.match(lines[j])
                if m:
                    head = m.group(1)
                    break
                j += 1
            if head is None:
                units["tags"].append({"name": "EmbeddedPyEditor", "attrs": {}})
                prev_kind = "other"
                i += 1
                continue
            if head != "":
                code_lines.append(head)
            j += 1
            while j < n:
                tick = lines[j].find("`")
                if tick != -1:
                    if lines[j][:tick]:
                        code_lines.append(lines[j][:tick])
                    break
                code_lines.append(lines[j])
                j += 1
            while code_lines and code_lines[0].strip() == "":
                code_lines.pop(0)
            while code_lines and code_lines[-1].strip() == "":
                code_lines.pop()
            units["code"].append({"lang": "python", "body": "\n".join(code_lines)})
            prev_kind = "other"
            while j < n and "/>" not in lines[j]:
                j += 1
            i = j + 1
            continue

        tag = TAG.match(trimmed)
        if tag:
            closing, name, rest, self_closing = tag.groups()
            attrs = attrs_of(rest)
            flush_prose()
            if not closing:
                flush_list()
            if name in FIGURE_OPENERS and not closing:
                figure = {
                    "src": attrs.get("src", attrs.get("image", "")),
                    "alt": attrs.get("alt", ""),
                    "imageDark": attrs.get("imageDark", ""),
                    "_caption_lines": [],
                }
                if self_closing:
                    flush_figure()
                prev_kind = "other"
                i += 1
                continue
            if not closing and name == "Resources":
                i += 1
                while i < n and lines[i].strip() != "</Resources>":
                    inner_tag = TAG.match(lines[i].strip())
                    if inner_tag and inner_tag.group(2) == "Resource" and not inner_tag.group(1):
                        inner_attrs = attrs_of(inner_tag.group(3))
                        units["resources"].append(
                            {
                                "source": inner_attrs.get("source", ""),
                                "title": inner_attrs.get("title", ""),
                                "url": inner_attrs.get("url", ""),
                                "starred": bool(inner_attrs.get("starred")),
                            }
                        )
                    i += 1
                prev_kind = "other"
                i += 1
                continue
            if not closing and name in CARD_TAGS and self_closing:
                # The importer turns B's <NotebookCard notebookPath="notebooks/x.ipynb">
                # into A's <NotebookCard id="x" /> by the notebook's filename stem,
                # so mirror that here or every notebook card looks remapped.
                card_id = attrs.get("id", "")
                if name == "NotebookCard" and not card_id and attrs.get("notebookPath"):
                    card_id = attrs["notebookPath"].rpartition("/")[2]
                    if card_id.endswith(".ipynb"):
                        card_id = card_id[: -len(".ipynb")]
                units["cards"].append((name, card_id))
                prev_kind = "other"
                i += 1
                continue
            if not closing and name == "Resource" and self_closing:
                units["resources"].append(
                    {
                        "source": attrs.get("source", ""),
                        "title": attrs.get("title", ""),
                        "url": attrs.get("url", ""),
                        "starred": bool(attrs.get("starred")),
                    }
                )
                prev_kind = "other"
                i += 1
                continue
            if not closing and name == "EmbeddedPyEditor":
                # A's file carries the Python code as the tag body; B keeps it
                # in an initialCode template literal (handled before the tag
                # match, because B's opening line has no ">"). Compare the two
                # as code so the tag is never an A-only tag.
                code_lines: list[str] = []
                j = i + 1
                while j < n and lines[j].strip() != "</EmbeddedPyEditor>":
                    code_lines.append(lines[j])
                    j += 1
                while code_lines and code_lines[0].strip() == "":
                    code_lines.pop(0)
                while code_lines and code_lines[-1].strip() == "":
                    code_lines.pop()
                units["code"].append({"lang": "python", "body": "\n".join(code_lines)})
                prev_kind = "other"
                i = j + 1
                continue
            # Every other tag (Warning, Info, Optional,
            # details, center, anything B invented) is recorded with its
            # attributes and its body is left in the prose stream, which is
            # what A's reader does with the ones it does not parse.
            units["tags"].append({"name": ("</" if closing else "") + name, "attrs": attrs})
            prev_kind = "other"
            i += 1
            continue

        image = MARKDOWN_IMAGE.match(trimmed)
        if image:
            flush_prose()
            flush_list()
            units["figures"].append({"src": image.group(2), "alt": image.group(1), "caption": ""})
            prev_kind = "other"
            i += 1
            continue

        bullet = BULLET.match(trimmed)
        ordered = ORDERED.match(trimmed)
        is_bullet = bool(bullet)
        is_ordered = bool(ordered)
        if (is_bullet or is_ordered) and (prev_kind in ("blank", "list", "none") or list_items):
            # A list item starts a list only at a block boundary; a wrapped
            # prose line that happens to begin with digits and a full stop
            # (B's "which is / 297.") continues the paragraph instead.
            if list_items and ((is_ordered != list_ordered) if (is_bullet or is_ordered) else False):
                flush_list()
            if not list_items:
                flush_prose()
                list_ordered = is_ordered
            list_items.append((bullet or ordered).group(1))
            prev_kind = "list"
            i += 1
            continue

        if list_items:
            # A's list parser folds every continuation line into the last item.
            list_items[-1] += " " + trimmed
            i += 1
            continue

        prose_buf.append(trimmed)
        prev_kind = "prose"
        i += 1

    flush_prose()
    flush_list()
    flush_figure()
    return units


# --------------------------------------------------------------- reporting ----


def compare(b: dict, a: dict, links: dict) -> dict:
    report: dict = {}

    b_sents = sentences(b["prose"])
    a_sents = sentences(a["prose"])
    a_matched, b_used = match(a_sents, b_sents, norm_key)
    label_only = []
    a_label_only = [False] * len(a_sents)
    for i, sent in enumerate(a_sents):
        if a_matched[i]:
            continue
        for j, other in enumerate(b_sents):
            if not b_used[j] and link_key(other) == link_key(sent):
                label_only.append((sent, other))
                b_used[j] = True
                a_label_only[i] = True
                break

    # A sentence that matches once its links are masked differs only in a
    # remapped label or target, which the importer is allowed to do. It is
    # reported under "labels", not as a changed or missing sentence.
    changed = [s for i, s in enumerate(a_sents) if not a_matched[i] and not a_label_only[i]]
    missing = [s for j, s in enumerate(b_sents) if not b_used[j]]

    report["sentences"] = {
        "a_total": len(a_sents),
        "b_total": len(b_sents),
        "matched": len(a_sents) - len(changed) - len(label_only),
        "changed": changed,
        "missing": missing,
        "label_only": len(label_only),
    }
    report["labels"] = label_only

    # Header: A's preamble against B's frontmatter.
    front = b["frontmatter"]
    head = a["header"]
    header_diff: dict = {}
    title = front.get("title", "")
    h1 = head.get("h1", "")
    header_diff["title"] = {
        "b": title,
        "a": h1,
        "ok": bool(title) and h1.endswith(title),
    }
    header_diff["description"] = {
        "b": front.get("description", ""),
        "a": head.get("goal", ""),
        "ok": front.get("description", "") == head.get("goal", ""),
    }
    header_diff["stage"] = {"b": front.get("tier", ""), "a": head.get("stage", "")}
    modules = links.get("modules", {}) if links else {}
    prereq_pairs = []
    mapped = {Path(v).name: k for k, v in modules.items()} if modules else {}
    for label, href in head.get("prerequisites", []):
        b_id = mapped.get(href, "")
        prereq_pairs.append(
            {
                "b_id": b_id,
                "a_label": label,
                "a_href": href,
                "in_b_frontmatter": b_id in front.get("prerequisites", []),
                "mapped": b_id != "",
            }
        )
    b_only = [p for p in front.get("prerequisites", []) if p not in {d["b_id"] for d in prereq_pairs}]
    header_diff["prerequisites"] = {
        "a": prereq_pairs,
        "b_only": b_only,
        "b_declared": bool(front.get("prerequisites")),
    }
    report["header"] = header_diff

    # Figures: the importer rewrites the directory mechanically (B's
    # "./assets/..." or "/diagrams/..." becomes "assets/module-media/<file>"),
    # so the file's basename is what has to match. imageDark has no A
    # equivalent and is dropped, which is noted rather than flagged.
    def fig_key(entry: dict) -> tuple:
        src = entry["src"] or ""
        return (os.path.basename(src), entry["alt"].strip(), entry["caption"].strip())

    b_figs = [fig_key(f) for f in b["figures"]]
    a_figs = [fig_key(f) for f in a["figures"]]
    dark = [f["imageDark"] for f in b["figures"] if f.get("imageDark")]
    report["figures"] = {
        "count_b": len(b_figs),
        "count_a": len(a_figs),
        "b_only": sorted(set(b_figs) - set(a_figs)),
        "a_only": sorted(set(a_figs) - set(b_figs)),
        "b_imageDark_dropped": dark,
    }

    # Code: bodies compared whole; the language relabelling is reported.
    code_notes = []
    n_code = min(len(b["code"]), len(a["code"]))
    for k in range(n_code):
        b_lang, a_lang = b["code"][k]["lang"], a["code"][k]["lang"]
        same_body = b["code"][k]["body"] == a["code"][k]["body"]
        note = {"index": k, "b_lang": b_lang, "a_lang": a_lang, "body_same": same_body}
        if (b_lang, a_lang) not in KNOWN_LANG_CHANGES and b_lang != a_lang:
            note["unexpected_lang"] = True
        if not same_body:
            note["diff"] = diff_lines(b["code"][k]["body"], a["code"][k]["body"])
        code_notes.append(note)
    report["code"] = {
        "count_b": len(b["code"]),
        "count_a": len(a["code"]),
        "blocks": code_notes,
    }

    report["math"] = {
        "count_b": len(b["math"]),
        "count_a": len(a["math"]),
        "b_only": sorted(set(b["math"]) - set(a["math"])),
        "a_only": sorted(set(a["math"]) - set(b["math"])),
    }
    report["tables"] = {
        "count_b": len(b["tables"]),
        "count_a": len(a["tables"]),
        "b_only": sorted(set(b["tables"]) - set(a["tables"])),
        "a_only": sorted(set(a["tables"]) - set(a["tables"])),
    }

    report["cards"] = {
        "count_b": len(b["cards"]),
        "count_a": len(a["cards"]),
        "b_only": [c for c in b["cards"] if c not in a["cards"]],
        "a_only": [c for c in a["cards"] if c not in b["cards"]],
        "order_same": b["cards"] == a["cards"],
    }

    def res_key(entry: dict) -> tuple:
        return (entry["source"], entry["title"], entry["url"], entry["starred"])

    report["resources"] = {
        "count_b": len(b["resources"]),
        "count_a": len(a["resources"]),
        "b_only": [res_key(r) for r in b["resources"] if res_key(r) not in {res_key(x) for x in a["resources"]}],
        "a_only": [res_key(r) for r in a["resources"] if res_key(r) not in {res_key(x) for x in b["resources"]}],
    }

    b_heads = sorted(h for _, h in b["headings"])
    # A's "## Prerequisites" is the header section, reported under "header".
    a_heads = sorted(h for _, h in a["headings"] if h != "Prerequisites")
    report["headings"] = {
        "count_b": len(b_heads),
        "count_a": len(a_heads),
        "b_only": sorted(set(b_heads) - set(a_heads)),
        "a_only": sorted(set(a_heads) - set(b_heads)),
        "order_same": [h for _, h in b["headings"]] == [h for _, h in a["headings"]],
    }

    def tag_key(tag: dict) -> tuple:
        return (tag["name"], tuple(sorted(tag["attrs"].items())))

    # <center> wrappers have no A equivalent; the importer drops them and warns.
    def is_dropped_tag(key: tuple) -> bool:
        return key[0] in ("center",)

    b_tags = [tag_key(t) for t in b["tags"]]
    a_tags = [tag_key(t) for t in a["tags"]]
    dropped = [key for key in b_tags if is_dropped_tag(key)]
    b_tags = [key for key in b_tags if not is_dropped_tag(key)]
    a_tags = [key for key in a_tags if not is_dropped_tag(key)]
    report["tags"] = {
        "count_b": len(b_tags),
        "count_a": len(a_tags),
        "b_only": [key for key in b_tags if key not in set(a_tags)],
        "a_only": [key for key in a_tags if key not in set(b_tags)],
        "b_dropped_by_design": dropped,
    }
    return report


def diff_lines(b: str, a: str, context: int = 2) -> list[str]:
    b_lines, a_lines = b.splitlines(), a.splitlines()
    out: list[str] = []
    for k in range(min(len(b_lines), len(a_lines))):
        if b_lines[k] != a_lines[k]:
            out.append(f"      line {k + 1}: B={b_lines[k][:100]!r}")
            out.append(f"      line {k + 1}: A={a_lines[k][:100]!r}")
    if len(b_lines) != len(a_lines):
        out.append(f"      line count: B={len(b_lines)} A={len(a_lines)}")
    return out[:8]


def print_report(report: dict, b_path: Path, a_path: Path) -> None:
    sent = report["sentences"]
    print(f"check_verbatim: {a_path.name} vs {b_path.name}")
    print(
        f"sentences: A={sent['a_total']} B={sent['b_total']} matched={sent['matched']} "
        f"changed={len(sent['changed'])} missing={len(sent['missing'])}"
    )
    if sent.get("label_only"):
        print(f"--- link labels remapped ({sent['label_only']}): reported under labels, not as changed")
    for text in sent["changed"]:
        print(f"  + in A, not in B: {text[:150]}")
    for text in sent["missing"]:
        print(f"  - in B, missing from A: {text[:150]}")
    labels = report["labels"]
    if labels:
        print(f"--- link labels remapped ({len(labels)}): not counted as changed")
        for a_text, b_text in labels[:12]:
            print(f"  ~ A: {a_text[:110]}")
            print(f"      B: {b_text[:110]}")
    header = report["header"]
    print(
        f"header: title={'ok' if header['title']['ok'] else 'DIFFERS'} "
        f"description={'ok' if header['description']['ok'] else 'DIFFERS'} "
        f"stage B={header['stage']['b']!r} A={header['stage']['a']!r}"
    )
    b_declared = bool(header["prerequisites"]["b_declared"])
    for entry in header["prerequisites"]["a"]:
        if entry["mapped"] and entry["in_b_frontmatter"]:
            flag = "ok"
        elif entry["mapped"] and not b_declared:
            flag = "inherited"
        else:
            flag = "CHECK"
        print(f"  prereq {flag}: {entry['b_id']!r} -> [{entry['a_label']}]({entry['a_href']})")
    for b_id in header["prerequisites"]["b_only"]:
        print(f"  prereq MISSING: B frontmatter {b_id!r} has no A link")
    figs = report["figures"]
    print(f"figures: B={figs['count_b']} A={figs['count_a']}")
    for triple in figs["b_only"][:6]:
        print(f"  - B only: src={triple[0][:60]} alt={triple[1][:60]!r}")
    for triple in figs["a_only"][:6]:
        print(f"  + A only: src={triple[0][:60]} alt={triple[1][:60]!r}")
    code = report["code"]
    unexpected = [n for n in code["blocks"] if n.get("unexpected_lang") or not n["body_same"]]
    print(f"code: B={code['count_b']} A={code['count_a']} blocks differing={len(unexpected)}")
    for note in unexpected[:6]:
        lang = f"lang {note['b_lang']!r} -> {note['a_lang']!r}"
        print(f"  * block {note['index']}: {lang} body_same={note['body_same']}")
        for line in note.get("diff", []):
            print(line)
    math = report["math"]
    print(f"math: B={math['count_b']} A={math['count_a']} b_only={len(math['b_only'])} a_only={len(math['a_only'])}")
    tables = report["tables"]
    print(f"tables: B={tables['count_b']} A={tables['count_a']} b_only={len(tables['b_only'])} a_only={len(tables['a_only'])}")
    cards = report["cards"]
    print(f"cards: B={cards['count_b']} A={cards['count_a']} order_same={cards['order_same']}")
    for card in cards["a_only"]:
        print(f"  + A only: {card}")
    for card in cards["b_only"]:
        print(f"  - B only: {card}")
    res = report["resources"]
    print(f"resources: B={res['count_b']} A={res['count_a']} b_only={len(res['b_only'])} a_only={len(res['a_only'])}")
    heads = report["headings"]
    print(f"headings: B={heads['count_b']} A={heads['count_a']} order_same={heads['order_same']}")
    for text in heads["a_only"]:
        print(f"  + A only: {text!r}")
    for text in heads["b_only"]:
        print(f"  - B only: {text!r}")
    tags = report["tags"]
    print(f"tags: B={tags['count_b']} A={tags['count_a']} b_only={len(tags['b_only'])} a_only={len(tags['a_only'])}")
    for key in tags["a_only"]:
        print(f"  + A only: {key}")
    for key in tags["b_only"]:
        print(f"  - B only: {key}")
    verdict = "PASS" if not sent["changed"] and not sent["missing"] else "FAIL"
    print(f"verdict: {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b_file", help="B lesson .mdx")
    parser.add_argument("a_file", help="imported A lesson .md")
    parser.add_argument("--links", help="scripts/import-b/links.json for prerequisite mapping")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    b_path = Path(args.b_file)
    a_path = Path(args.a_file)
    if not b_path.is_file() or not a_path.is_file():
        print("check_verbatim: both files must exist", file=sys.stderr)
        return 2

    links_path = Path(args.links) if args.links else Path(__file__).parent / "links.json"
    links = {}
    if links_path.is_file():
        links = json.loads(links_path.read_text(encoding="utf-8"))

    b = parse_lesson(b_path, True)
    a = parse_lesson(a_path, False)
    report = compare(b, a, links)

    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        print_report(report, b_path, a_path)

    sent = report["sentences"]
    return 0 if not sent["changed"] and not sent["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
