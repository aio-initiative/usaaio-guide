# Content Structure

This folder mirrors the content-first structure used by guide projects like USACO Guide and USAMO Guide.

- `1_Round_1`, `2_Round_2`, `3_Camp`, and `4_Advanced` hold the 55 module documents in Markdown. **These are the source of truth.**
- `ordering.js`, `catalog.js`, and `lessons.js` are **generated** from those Markdown files by `npm run build:content`. Do not edit them by hand.
- `problems.js` and `theory.js` are hand-maintained: the practice bank and the written-problem bank.
- `problemBank` stores structured practice problem metadata.

## Writing a module

Each file is named `<stage>.<n>-<CODE>-<slug>.md`, where `<CODE>` is the syllabus subject code (`PY1`, `MA3`, `TF2`, ...). The slug becomes the module id on the site, so `2.8-RP2-contrastive-learning.md` is served at `/module/?id=contrastive-learning`.

Every module opens with:

```markdown
# 2.8 · RP2 — Contrastive Learning

> **Goal:** learn by comparison — InfoNCE, SimCLR, MoCo, BYOL, CLIP, and why they do not collapse.
> **Stage:** Round 2
```

The `Goal` line becomes the module summary shown on stage pages and in search.

After that a module is a **linear lesson**: it names its own topics, and every
`## ` heading becomes one reader section in document order. There is no fixed
section vocabulary — write the headings the material actually needs, in the
order you would teach them, and keep each topic's explanation, math, diagram,
worked example and code together under that topic.

Three headings are structural:

| Heading | What it does |
| --- | --- |
| `## Prerequisites` | required; its Markdown links become the catalog `prerequisites`, and the reader hoists it into a box under the title |
| `## Overview` | optional lead-in; rendered with **no heading of its own** so the module opens straight into prose |
| `## Practice Problems` | dropped — the module page ends with that module's slice of `problems.js` |

A typical module therefore looks like:

```markdown
## Prerequisites

- [Supervised Learning Basics](1.14-ML2-supervised-learning-foundations.md)

## Overview

One or two paragraphs framing the module.

## Confusion Matrix

Prose, a table, a diagram.

### Worked Example

The numbers, then the code that reproduces them.

## Precision, Recall, Specificity, And FPR

...

## Quick Summary

- one line per idea
```

Inside a section you can use paragraphs, `###`/`####` sub-headings, bullet and numbered lists, tables, blockquotes, fenced code, images, `$inline$` and `$$display$$` math, and `<details><summary>Answer</summary>...</details>` for hidden solutions. Each becomes one block in `lessons.js`; the reader renders blocks in order and typesets the math with KaTeX.

Links to other modules are written as ordinary relative Markdown links and are rewritten to site routes at build time. **Use the target module's title as the link text**, not its number and subject code — `[Supervised Learning Basics](1.14-ML2-supervised-learning-foundations.md)`, never `[1.14 ML2](...)`.

Diagrams live in `assets/module-media/` and are referenced with a normal image link: `![Confusion matrix layout](assets/module-media/ml6-04-confusion-matrix-layout.svg)`.

Generated diagrams are managed by `scripts/figures/` and carry a `<!-- figure:slug -->`
marker plus a collapsible block holding the code that drew them. A few concepts are
architecture diagrams that matplotlib draws badly; those use a freely-licensed diagram
from Wikimedia Commons instead, marked `<!-- web-figure:slug -->` and followed by a
credit line. They are declared in `scripts/figures/web_media.py`, downloaded by running
it, and their provenance is recorded in `content/media-credits.json`. Attribution is not
optional — every one is CC BY or CC BY-SA and the credit line is the licence condition.

## Grouping

`scripts/build-content.mjs` also owns the stage -> chapter -> module grouping (`Math Foundations`, `Classical Machine Learning`, ...). Adding a module means adding its id to a chapter in that file; the build fails if a module is in no chapter, in two chapters, or in a chapter belonging to another stage.

## Workflow

```bash
npm run build:content   # Markdown -> ordering.js + catalog.js + lessons.js
npm run validate        # cross-checks catalog/ordering/lessons/problems/theory
```

The site loads content through classic browser scripts rather than a build-time MDX compiler, so `index.html` can still be opened directly. `lessons.js` is large and is only loaded by `/module` and `/search`, the two pages that read it.
