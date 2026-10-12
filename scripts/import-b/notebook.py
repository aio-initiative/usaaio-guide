#!/usr/bin/env python3
"""Turn a site B notebook pair into one site A notebook.

Usage:
  scripts/import-b/notebook.py <B task notebook>
      [--solution <B solution notebook>] --module <A module id> --out <file>
      [--colab "<sentence>"] [--difficulty <word>] [--minutes <n>]
      [--tags a,b,c] [--description "<text>"]

B keeps every notebook as a pair: `<name>.ipynb`, the exercise the student
works on, and `<name>-solution.ipynb`, the same notebook with each task cell
replaced by its worked solution. A keeps one file: the exercise cells first,
untouched, and the whole worked solution appended at the end behind a
markdown cell that starts `## Solution`, so the editor can fold it and
`?embed=1` can strip it.

The solution section keeps B's order. Each part of the exercise gets a
heading (`### Part 1: <B's part title>`, the `(coding task)` dropped) followed
by the cells of B's solution for that part that actually differ from the
exercise. Cells B's solution repeats word for word are dropped: the student
already has them above. Explanations B adds to its solution ("Why it works.")
are kept.

Nothing is written that is not in B. The metadata comes from B's
`metadata.usaaio` where B has it (id, title, difficulty, minutes, tags), and
from the notebook itself otherwise (the id from the file name, the title from
the first heading, the description from the exercise's own task sentence).
B's `runSeconds` and `colab` are how B says a notebook runs, not what it
teaches, so they are not copied: `colab` only lands in A's notebook when it is
passed with `--colab`, after someone has run the notebook against the page's
own data subset and written a sentence that names the parts that still miss
their check.

Anything the script cannot map is printed as a WARNING and left out; it never
guesses. Warnings go to stdout, one per line, and the exit code stays 0.
"""

import argparse
import difflib
import json
import os
import re
import sys

# B writes its part headings as "## Part 2 (coding task): Title".
PART_HEADING = re.compile(r"^##\s+Part\s+(\d+)\s+\((\w+)\s+task\)\s*:\s*(.+)$", re.M)
# Any other H2 section of the exercise ("## Setup"), which can also hold
# solution-only cells.
H2 = re.compile(r"^##\s+(.+)$", re.M)
# B's own rule for the two fields that say how a notebook runs, not what it
# teaches (src/lib/notebooks/collect.ts in site B).
RUN_KEYS = {"runSeconds", "colab"}
DIFFICULTIES = ("Easy", "Medium", "Hard", "Very hard")
SOLUTION_FLAG = {"usaaio_solution": True}


def warn(message):
    print(f"WARNING: {message}")


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def source(cell):
    text = cell.get("source")
    if isinstance(text, list):
        return "".join(text)
    return text or ""


def parse_sections(cells, label):
    """Split a notebook into ordered sections at H2 headings.

    Returns a list of (kind, number, title, [cells]). kind is "part" for
    "## Part N (coding task): Title" and "section" for any other H2. Cells
    before the first H2 belong to an implicit section named None.
    """
    sections = []
    current = ("head", None, None, [])
    for cell in cells:
        text = source(cell)
        part = PART_HEADING.search(text)
        if part:
            if current[3]:
                sections.append(current)
            current = ("part", int(part.group(1)), part.group(3).strip(), [])
            continue
        h2 = H2.search(text)
        if h2 and cell.get("cell_type") == "markdown" and not text.lstrip().startswith("#"):
            # A part heading is also an H2; the check above already took it.
            if current[3]:
                sections.append(current)
            current = ("section", None, h2.group(1).strip(), [])
            continue
        current[3].append(cell)
    if current[3]:
        sections.append(current)
    if not any(kind == "part" for kind, _, _, _ in sections):
        warn(f"{label}: no '## Part N (coding task)' heading; the solution "
             "section is one undivided block")
    return sections


def exercise_sections(cells, label):
    """The exercise's sections, keyed so the solution can be matched to them."""
    out = []
    for kind, number, title, cells_of in parse_sections(cells, label):
        if kind == "part":
            key = f"part {number}"
        elif kind == "section":
            key = f"section {title}"
        else:
            key = "head"
        out.append((key, kind, number, title, cells_of))
    return out


def solution_cells_for(task_cells, sol_cells, where):
    """The cells of B's solution that differ from its exercise, in order.

    Cells B repeats word for word are dropped: the exercise above already has
    them, and the solution section has to read as one worked notebook.
    """
    a = [source(c) for c in task_cells]
    b = [source(c) for c in sol_cells]
    kept = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        kept.extend(sol_cells[j1:j2])
        if tag == "delete":
            for task_cell in task_cells[i1:i2]:
                if is_todo(source(task_cell)):
                    continue
                warn(f"{where}: B's solution drops a cell the exercise has, "
                     f"which the solution section cannot carry: "
                     f"{source(task_cell).splitlines()[0][:70]!r}")
        elif tag == "replace" and any(
            c.get("cell_type") != "code" for c in task_cells[i1:i2]
        ):
            warn(f"{where}: B's solution rewrites an exercise markdown cell; "
                 "the solution's version is carried into the solution section")
    return kept


def is_todo(text):
    return (
        "TODO" in text
        or "raise NotImplementedError" in text
        or "YOUR SOLUTION GOES HERE" in text
        or re.search(r"=\s*\.\.\.\s*($|#)", text, re.M) is not None
    )


def clean_cell(cell, flag=False):
    """One of B's cells as A keeps it: source untouched, outputs stripped."""
    new = {
        "cell_type": cell["cell_type"],
        "metadata": dict(SOLUTION_FLAG) if flag else {},
    }
    if cell["cell_type"] == "code":
        new["execution_count"] = None
        new["outputs"] = []
    text = cell.get("source")
    new["source"] = list(text) if isinstance(text, list) else (
        [text] if text else []
    )
    if isinstance(new["source"], list) and new["source"]:
        # nbformat joins the list; every entry but the last ends with a newline.
        new["source"] = [
            (line + "\n") if not line.endswith("\n") else line
            for line in new["source"][:-1]
        ] + [new["source"][-1]]
    return new


def h1(cells):
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        match = re.match(r"^\s*#\s+(.+?)\s*$", source(cell), re.M)
        if match:
            return match.group(1).strip()
    return None


def plain(text):
    """B's own cleanup for a notebook's opening text (summaryOf in site B),
    without its mangling of identifiers: math becomes its body, backticks go,
    whitespace collapses."""
    text = re.sub(r"\$\$?(.+?)\$\$?", r"\1", text, flags=re.S)
    text = text.replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return " ".join(text.split())


def derive_description(cells, where):
    """The exercise's own task sentence, as plain text.

    B's exercises state the task one way: "Here you ...". That sentence is the
    description, with "Here you" dropped. When the exercise does not use that
    phrasing the description falls back to the exercise's opening paragraph,
    which is what B itself shows as the notebook's summary.
    """
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        text = source(cell)
        text = re.sub(r"^\s*#\s+.+\n?", "", text.lstrip(), count=1)
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
            and not paragraph.strip().startswith("**How to work through")
            and not paragraph.strip().startswith("Original USAAIO Guide")
        ]
        if not paragraphs:
            continue
        body = " ".join(paragraphs[0].split())
        sentences = re.split(r"(?<=[.!?])\s+", body)
        here = next(
            (sentence for sentence in sentences if re.search(r"\bHere you\b", sentence)),
            None,
        )
        if here is not None:
            body = re.sub(r"^\s*Here you\s+", "", here)
        else:
            warn(f"{where}: the exercise does not state its task with 'Here "
                 "you', so the description is its whole opening paragraph; "
                 "shorten it by hand if the card needs one line")
        body = plain(body)
        if body:
            body = body[0].upper() + body[1:]
        if "**" in body:
            warn(f"{where}: the derived description still has markdown "
                 f"emphasis: {body[:80]!r}")
        return body
    warn(f"{where}: no markdown cell to take a description from")
    return ""


def is_coding_problem(usaaio):
    """B's rule (isCodingProblem in site B): a notebook is a coding problem
    when its metadata.usaaio holds anything beyond how it runs."""
    if usaaio is None:
        return False
    if not isinstance(usaaio, dict):
        return True
    return any(key not in RUN_KEYS for key in usaaio)


def convert(task_path, solution_path, module, out_path, overrides):
    if not os.path.isfile(task_path):
        fail(f"no such task notebook: {task_path}")
    with open(task_path, encoding="utf-8") as handle:
        task = json.load(handle)
    cells = task.get("cells")
    if not isinstance(cells, list) or not cells:
        fail(f"{task_path}: no cells")
    where = os.path.basename(task_path)

    meta = task.get("metadata") or {}
    usaaio = meta.get("usaaio")
    if usaaio is not None and not isinstance(usaaio, dict):
        fail(f"{task_path}: metadata.usaaio is not an object")

    # ---- the exercise cells, untouched -------------------------------------
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in ("markdown", "code"):
            fail(f"{task_path}: cell {index} has cell_type "
                 f"{cell.get('cell_type')!r}, which A has no place for")
        if (cell.get("metadata") or {}).get("usaaio_solution"):
            warn(f"{task_path}: cell {index} is already flagged "
                 "usaaio_solution in B's exercise; the flag is dropped")

    out_cells = [clean_cell(cell) for cell in cells]

    # ---- the worked solution appended at the end ---------------------------
    if solution_path is None:
        guessed = os.path.splitext(task_path)[0] + "-solution.ipynb"
        if os.path.isfile(guessed):
            solution_path = guessed
            warn(f"{where}: --solution not given, using {guessed}")
    solution = None
    if solution_path is not None:
        if not os.path.isfile(solution_path):
            warn(f"{where}: no solution notebook at {solution_path}; the A "
                 "notebook has no solution section")
        else:
            with open(solution_path, encoding="utf-8") as handle:
                solution = json.load(handle)
            sol_cells = solution.get("cells")
            if not isinstance(sol_cells, list):
                fail(f"{solution_path}: no cells")

    solution_section = []
    if solution is not None:
        task_parts = exercise_sections(cells, where)
        sol_sections = parse_sections(solution["cells"], where)
        # B's solution repeats the exercise's headings; match them by title.
        by_key = {}
        for kind, number, title, cells_of in sol_sections:
            if kind == "part":
                key = f"part {number}"
            elif kind == "section":
                key = f"section {title}"
            else:
                key = "head"
            by_key.setdefault(key, []).extend(cells_of)
        order = []
        seen = set()
        for key, kind, number, title, _ in task_parts:
            if key not in seen:
                seen.add(key)
                order.append((key, kind, number, title))
        for key in by_key:
            if key not in seen:
                seen.add(key)
                order.append((key, "section", None, key[len("section "):] if key.startswith("section ") else None))

        for key, kind, number, title in order:
            sol_of = by_key.get(key, [])
            if not sol_of:
                continue
            task_of = next(
                (cells_of for k, _, _, _, cells_of in task_parts if k == key), []
            )
            kept = solution_cells_for(task_of, sol_of, f"{where} {key}")
            if not kept:
                continue
            if kind == "part":
                solution_section.append(
                    markdown_cell(f"### Part {number}: {title}", flag=True))
            elif title is not None:
                solution_section.append(markdown_cell(f"### {title}", flag=True))
            elif key != "head":
                warn(f"{where}: B's solution has cells under a heading the "
                     f"exercise does not have ({key!r}); they are carried "
                     "into the solution section without a heading")
            for cell in kept:
                solution_section.append(clean_cell(cell, flag=True))

    # ---- metadata ----------------------------------------------------------
    new_usaaio = {}
    stem = os.path.splitext(os.path.basename(task_path))[0]
    new_id = overrides.get("description_id") or None
    if isinstance(usaaio, dict) and isinstance(usaaio.get("id"), str) and usaaio["id"]:
        new_id = usaaio["id"]
    elif new_id is None:
        new_id = stem
    out_stem = os.path.splitext(os.path.basename(out_path))[0]
    if new_id != out_stem:
        warn(f"{where}: the A notebook's id is {new_id!r} but the output file "
             f"is named {out_stem!r}; A's validator wants them to match")
    new_usaaio["id"] = new_id

    title = overrides.get("title")
    if not title and isinstance(usaaio, dict):
        maybe = usaaio.get("title")
        if isinstance(maybe, str) and maybe.strip():
            title = maybe.strip()
    if not title:
        title = h1(cells)
        if title:
            warn(f"{where}: B's metadata has no title; taking the notebook's "
                 f"first heading, {title!r}")
    if not title:
        warn(f"{where}: no title anywhere; the A notebook's title is empty "
             "and its card will not render")
    new_usaaio["title"] = title or ""

    new_usaaio["module"] = module

    description = overrides.get("description")
    if description is None:
        description = derive_description(cells, where)
    new_usaaio["description"] = description

    difficulty = overrides.get("difficulty")
    if difficulty is None and isinstance(usaaio, dict):
        maybe = usaaio.get("difficulty")
        if isinstance(maybe, str) and maybe.strip():
            difficulty = maybe.strip()
    if difficulty is None:
        difficulty = "Medium"
        warn(f"{where}: B's metadata says nothing about difficulty; the A "
             "notebook says Medium. Pass --difficulty to change it")
    if difficulty not in DIFFICULTIES:
        fail(f"{where}: difficulty {difficulty!r} is not one of "
             f"{', '.join(DIFFICULTIES)}")
    new_usaaio["difficulty"] = difficulty

    minutes = overrides.get("minutes")
    if minutes is None and isinstance(usaaio, dict):
        maybe = usaaio.get("minutes")
        if isinstance(maybe, int) and maybe > 0:
            minutes = maybe
    if minutes is None:
        minutes = 30
        warn(f"{where}: B's metadata says nothing about how long it takes; "
             "the A notebook says 30 minutes. Pass --minutes to change it")
    if not (isinstance(minutes, int) and minutes > 0) or isinstance(minutes, bool):
        fail(f"{where}: minutes must be a whole number of minutes, got {minutes!r}")
    new_usaaio["minutes"] = minutes

    tags = overrides.get("tags")
    if tags is None and isinstance(usaaio, dict):
        maybe = usaaio.get("tags")
        if isinstance(maybe, list) and all(isinstance(t, str) for t in maybe):
            tags = maybe
    if tags is None:
        tags = []
        warn(f"{where}: B's metadata has no tags; the A notebook has none. "
             "Pass --tags to list the ideas it teaches")
    new_usaaio["tags"] = tags

    guided = not is_coding_problem(usaaio)
    new_usaaio["guided"] = guided
    if guided:
        if usaaio is None:
            warn(f"{where}: B's notebook carries no metadata.usaaio at all, so "
                 "A's notebook is guided (a lesson's companion, not a problem)")
        else:
            warn(f"{where}: B lists this notebook as a lesson's companion rather "
                 "than a problem (its metadata only says how it runs), so A's "
                 "notebook is guided")

    colab = overrides.get("colab")
    if colab is not None:
        if not isinstance(colab, str) or not colab.strip():
            fail(f"{where}: --colab must be a sentence saying why the page "
                 "cannot run the notebook")
        new_usaaio["colab"] = colab.strip()
    elif isinstance(usaaio, dict) and usaaio.get("colab"):
        warn(f"{where}: B's notebook carries a colab sentence that the A "
             "notebook does not repeat, because it is about B's own data "
             "subset. Run the notebook on the page's subset and pass --colab "
             "with a sentence naming the parts that still miss their check. "
             f"B's sentence: {str(usaaio['colab']).strip()!r}")

    # ---- write --------------------------------------------------------------
    notebook = {"cells": out_cells, "metadata": {}, "nbformat": 4,
                "nbformat_minor": 5}
    for key in ("kernelspec", "language_info"):
        if isinstance(meta.get(key), dict):
            notebook["metadata"][key] = meta[key]
    notebook["metadata"]["usaaio"] = new_usaaio

    if solution_section:
        notebook["cells"].append(markdown_cell("## Solution", flag=True))
        notebook["cells"].extend(solution_section)
    else:
        warn(f"{where}: no solution section; the A notebook has no folded "
             "worked solution")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {out_path} ({len(out_cells)} exercise cells, "
          f"{len(solution_section)} solution cells)")


def markdown_cell(text, flag=False):
    return {
        "cell_type": "markdown",
        "metadata": dict(SOLUTION_FLAG) if flag else {},
        "source": [text],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", help="B's exercise notebook")
    parser.add_argument("--solution", help="B's solution notebook (default: "
                        "the -solution.ipynb beside the exercise)")
    parser.add_argument("--module", required=True,
                        help="the A module id the notebook belongs to")
    parser.add_argument("--out", required=True, help="the A notebook to write")
    parser.add_argument("--colab", help="a sentence saying why the page "
                        "cannot run the notebook, after running it there")
    parser.add_argument("--difficulty", help="one of Easy, Medium, Hard, "
                        "Very hard (default: B's, or Medium when B is silent)")
    parser.add_argument("--minutes", type=int, help="a whole number of "
                        "minutes (default: B's, or 30 when B is silent)")
    parser.add_argument("--tags", help="comma separated tags (default: B's)")
    parser.add_argument("--description", help="override the derived "
                        "one-line description, for example with the lesson's "
                        "own sentence about the notebook")
    parser.add_argument("--title", help="override the title")
    args = parser.parse_args()

    overrides = {
        "colab": args.colab,
        "difficulty": args.difficulty,
        "minutes": args.minutes,
        "description": args.description,
        "title": args.title,
    }
    if args.tags is not None:
        overrides["tags"] = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    convert(args.task, args.solution, args.module, args.out, overrides)


if __name__ == "__main__":
    main()
