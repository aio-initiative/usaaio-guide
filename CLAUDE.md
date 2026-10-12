# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AIO Initiative: a content-first, dependency-free static site for USA AI Olympiad prep, structured like USACO Guide / USAMO Guide. No framework, no bundler, no build step — plain HTML pages loading plain browser-global scripts via `<script src>` tags.

## Commands

```bash
npm run dev            # serve on http://127.0.0.1:4173 via python3 -m http.server (alias: npm run serve)
npm run build:content  # regenerate ordering.js/catalog.js/lessons.js from the module Markdown
npm run validate       # run scripts/validate-content.mjs (alias: npm run check)
```
     
There is no install step, no bundler, and no test suite beyond `validate-content.mjs`. You can also just open `index.html` directly in a browser, but route-relative links work more reliably through the local server.

Always run `npm run build:content` after editing a module's Markdown, then `npm run validate` after touching anything under `content/` — validation is the only automated check in this repo and it catches the most common mistakes (mismatched IDs, missing files, orphaned references).

## Architecture

### Script loading order is load-bearing

Every route page (e.g. `index.html`, `module/index.html`, `round1/index.html`) loads the same fixed sequence of scripts, each an IIFE that attaches its exports to `window`:

```
content/ordering.js   -> window.AIO_ORDERING
content/catalog.js    -> window.AIO_CONTENT
content/lessons.js    -> window.AIO_LESSONS   (only module/ and search/ — see below)
content/problems.js   -> window.AIO_PROBLEMS
src/data/site-data.js -> window.AIO_SITE_DATA
src/lib/progress.js   -> window.AIO_PROGRESS
src/components/renderers.js -> window.AIO_RENDERERS
src/app.js            -> reads all of the above, drives the page
```

There is no module system — later scripts depend on globals set by earlier ones, so this order must be preserved on every page.

`content/lessons.js` is the exception: it holds every lesson body (~2 MB) and is loaded **only** by `module/index.html` and `search/index.html`, the two pages that read it. `app.js` therefore treats `window.AIO_LESSONS` as optional. Do not add it back to the other pages. `src/app.js` is a single large IIFE that queries `document.body.dataset` (`page`, `root`, `section`) to decide what to render and wires up one delegated `click`/`change`/`submit` listener for the whole app rather than per-element handlers.

All `<script>`/`<link>` tags across every HTML page share one cache-busting query string (currently `?v=20260830-usaco-type`). When editing any JS/CSS asset, bump this string consistently across **all** HTML files (all files under the route folders plus `index.html`), or browsers/CDNs may serve stale cached versions.

### Route pages, not client-side routing

Each top-level folder (`dashboard/`, `planner/`, `round1/`, `round2/`, `camp/`, `advanced/`, `module/`, `problems/`, `contests/`, `resources/`, `official/`, `search/`, `contact/`, `editor/`, `theory/`) is a real folder with its own `index.html` (`roadmap/`, `settings/`, `login/`, `dl/`, `ml/`, `foundation/`, and `practice/` are redirect stubs for removed/moved routes). Pages differentiate behavior purely through `<body data-page="..." data-root="..." data-section="...">`:
- `data-page` selects which `render*` functions in `app.js` are relevant (e.g. `"module"`, `"section"`, `"dashboard"`).
- `data-root` is the relative path back to the repo root (`""` at top level, `"../"` one level down) used to build links.
- `data-section` (on `round1/`, `round2/`, `camp/`, `advanced/`) tells `app.js` which entry in `ordering.sectionOrdering` to render.

When adding a new route, copy the pattern of an existing sibling page (header/nav markup, script tag block, `data-*` attributes) rather than inventing a new structure.

### Content pipeline: Markdown is the source, the JS is generated

The 55 modules live as Markdown under `content/1_Round_1/`, `2_Round_2/`, `3_Camp/`, `4_Advanced/`, named `<stage>.<n>-<CODE>-<slug>.md` (e.g. `2.20-RP6-contrastive-learning.md`, module id `contrastive-learning`). See `content/README.md` for the module shape.

`npm run build:content` (`scripts/build-content.mjs`) parses them and **overwrites** three files. Never hand-edit these:

1. **`content/ordering.js`** — stage -> chapter -> module ID sequence. Drives nav order, prev/next links, and section pages. The chapter grouping itself (`Math Foundations`, `Classical Machine Learning`, ...) is declared in the `CHAPTERS` table at the top of `build-content.mjs`; the build fails if a module is in no chapter, in two, or in another stage's chapter.
2. **`content/catalog.js`** — flat array of module metadata (`id`, `title`, `section`, `level`, `summary`, `prerequisites`, `prerequisiteTitles`, `code`, `file`). `summary` comes from the `> **Goal:**` line; `prerequisites` from the Markdown links in the Prerequisites section, resolved to titles in `prerequisiteTitles`.
3. **`content/lessons.js`** — one reader section per `## ` heading, in document order. A module is a linear lesson: it names its own topics, so there is no fixed section vocabulary. Two headings are structural — `## Prerequisites` (hoisted by the reader into a box under the title) and `## Overview` (the lead-in, rendered with no heading of its own, id `intro`) — and `## Practice Problems` is dropped in favour of the module's slice of `problems.js`. Each section is `{ id, title, blocks[] }`, where a block is exactly one of `p` (paragraph), `tex` (display math), `heading`, `list`, `olist`, `table`, `quote`, `code`, or `details` (a collapsible answer). `renderers.renderBlocks` renders them in order; `escapeHtml` runs on everything, then `$...$` spans, `**bold**`, `*italic*`, `` `code` `` and `[links](...)` are reintroduced. Inline and display math are emitted as `data-tex` placeholders and typeset by `renderReaderMath` in `app.js` once KaTeX has loaded.

Hand-maintained alongside them:

- **`content/problems.js`** — the practice-problem bank and contest presets used by `problems/` and `contests/`. Each entry's `moduleId` must name a module in the catalog, and its stage tag (`Round 1`/`Round 2`/`Camp`/`Advanced`) must match that module's stage.
- **`content/problemBank/*.problems.json`** — optional structured practice metadata per module (`MODULE_ID` + `practice[]`), validated against `problemBank/problems.schema.json`.

### Two kinds of problems: Coding and Theory

Every entry in `content/problems.js` carries exactly one type tag, `"Coding"` or `"Theory"` (the validator enforces this, since the Type filter on `problems/` and the badge on each card both depend on it).

- **Coding** problems are the 56 auto-graded notebooks. `url` is `/editor/?nb=<id>`, backed by `notebooks/<id>.ipynb`.
- **Theory** problems are multi-part written problems in the format of the official USAAIO papers. `url` is `/theory/?id=<id>`, backed by an entry in **`content/theory.js`**.

`content/theory.js` (`window.AIO_THEORY`) holds each problem's `overview`, `setup`, and `parts[]`. A part has `id`, `kind` (`derivation` / `computation` / `mcq` / `explanation` / `coding`), a `requirement` string (the official grading instruction, e.g. "Reasoning is required."), and `prompt`/`answer` arrays of content blocks — see the schema comment at the top of the file. Inline math uses `$...$`; display math is a `{ tex }` block.

Part ids are stored as `"<problem>.<part>"` so they stay unique across the bank, but only the trailing number is displayed and exported: part `"3.1"` renders as "Part 1". There are no point values — this bank does not score.

`content/theory.js` and `src/theory.js` load **only** on `theory/index.html` (which also loads KaTeX), the same way `src/editor.js` loads only on `editor/index.html` — they are page-specific and deliberately not part of the shared script sequence. `src/theory.js` renders the parts, drives the per-part reveal-answer buttons, and builds the `.ipynb` export whose text cells use the official `# **Part N.M**` heading convention.

When adding a theory problem, add it to `content/theory.js` *and* register a matching entry in `content/problems.js` with the `Theory` tag and a `/theory/?id=<id>` url; `npm run validate` checks the two stay in sync, that every part has a requirement/prompt/answer, and that `mcq` parts actually list choices.

The `.ipynb` export is a **blank worksheet** — prompts plus an empty cell per part, with no solutions. Answers live only on the page, behind the per-part reveal buttons.

To add a new module: write the `.md` file in the right stage folder following the ten-section shape in `content/README.md`, add its id to a chapter in the `CHAPTERS` table in `scripts/build-content.mjs`, optionally add a `problemBank/*.problems.json` file, then run `npm run build:content && npm run validate`.

### Client-side state (`src/lib/progress.js`)

All user progress (module/problem status, bookmarks, scratchpad, contest run history, activity log) lives in `localStorage` under a single key (`usaaio-guide-state-v2`). There is no auth or per-user profile system. Any state mutation goes through `storage.updateState(...)` and returns a fresh normalized state object; `app.js` re-renders via `syncAppState(newState)` after every mutation rather than doing fine-grained DOM patching.

### Rendering (`src/components/renderers.js`)

Pure functions that take data + app state and return HTML strings (no templating engine, no JSX). `app.js` calls these and assigns the result to `element.innerHTML`. Always escape user/content strings with `renderers.escapeHtml(...)` when interpolating into HTML strings here.

## Content principles (from CONTRIBUTING.md)

- Write for middle and high school students; connect math, implementation, and contest explanation.
- Prefer small worked examples students can verify by hand.
- Keep unofficial guidance clearly separate from official USAAIO facts/links (rules, dates, syllabus, past problems should link to usaaio.org).
