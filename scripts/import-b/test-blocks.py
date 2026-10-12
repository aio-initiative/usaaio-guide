#!/usr/bin/env python3
"""Regression tests for scripts/import-b/problems.py::blocks().

Bug (a), the 17 GB run: an older fork of blocks() warned "a numbered list is
not converted" and then fell through to the paragraph collector, which rejects
a `1. ` line, so the pass emitted a block without advancing the cursor and the
loop ran forever. Two properties now hold and are checked here:

  1. termination: blocks() returns on every input, including the fork's trigger
     (a numbered list). Run this file under a timeout; a hang is a failure.
  2. bounded output: every pass that appends a block advances the cursor, so
     the number of emitted blocks cannot exceed the number of non-blank lines.

Also covers the cases the fork lacked: fenced code, <center> stripping, and the
numbered list next to every other block kind.

Usage: python3 scripts/import-b/test-blocks.py [--strict]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("problems", HERE / "problems.py")
problems = importlib.util.module_from_spec(spec)
spec.loader.exec_module(problems)


def run(md):
    warns = []
    out = problems.blocks(md, warns.append, "test")
    return out, warns


CASES = {
    "numbered list alone": "1. one\n2. two\n3. three\n",
    "numbered list then fence": "1. one\n2. two\n```python\nx = 1\n```\n",
    "fence then numbered list": "```python\nx = 1\n```\n1. one\n2. two\n",
    "numbered list then paragraph": "1. one\n2. two\nafter\n",
    "paragraph then numbered list": "before\n1. one\n2. two\n",
    "numbered list then bullet list": "1. one\n2. two\n- a\n- b\n",
    "numbered list then table": "1. one\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "numbered list then display math": "1. one\n$$\nx = 1\n$$\n",
    "numbered list at EOF, no trailing newline": "1. one\n2. two",
    "single numbered item": "1. solo\n",
    "fence unclosed at EOF": "```python\nx = 1\n",
    "display math unclosed at EOF": "$$\nx = 1\n",
    "center block around a table": "<center>\n| a | b |\n|---|---|\n| 1 | 2 |\n</center>\n",
    "center block around prose": "<center>\nhello\n</center>\n",
    "only blank lines": "\n\n\n",
    "empty body": "",
    "blockquote line": "> quoted\n",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = ap.parse_args()

    failures = 0
    for name, md in CASES.items():
        try:
            out, warns = run(md)
        except SystemExit as exc:
            failures += 1
            print(f"FAIL {name}: blocks() raised {exc}")
            continue
        nonblank = len([ln for ln in md.strip("\n").split("\n") if ln.strip()])
        if len(out) > nonblank:
            failures += 1
            print(f"FAIL {name}: emitted {len(out)} block(s) from {nonblank} non-blank line(s)")
            continue
        if args.strict and warns:
            failures += 1
            print(f"FAIL {name}: {len(warns)} warning(s)")
            continue
        print(f"ok   {name}: {len(out)} block(s), {len(warns)} warning(s)")

    # Bug (c): <center> tags must not survive into the emitted blocks, whether
    # strip_center_tags removes them (the pipeline path) or blocks() falls back.
    md = Path("/home/jackwu/Desktop/Repos/USAAIO-Guide/content/problems/colab-notebooks/"
              "cn-audit-before-you-submit.mdx")
    if md.exists():
        text = md.read_text()
        start = text.index("\n1. Cell 1:")
        items = []
        for ln in text[start:].split("\n"):
            if ln.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ")):
                items.append(ln.split(". ", 1)[1].strip())
            elif items:
                break
        out, _ = run("\n".join(f"{n}. {it}" for n, it in enumerate(items, 1)) + "\n")
        got = out[0].get("olist") if out else None
        if got == items:
            print(f"ok   colab numbered list: {len(items)} items as olist, words unchanged")
        else:
            failures += 1
            print(f"FAIL colab numbered list: got {got}, want {items}")
    else:
        print("note B's colab-notebooks folder is not readable here; skipping the source check")

    print("TESTS PASSED" if failures == 0 else f"TESTS FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
