# Chapters from data (merge tooling, platform lane)

The chapter grouping for each stage used to be hard-coded as the `CHAPTERS`
literal in `scripts/build-content.mjs`. With ~62 new modules arriving on
parallel branches, that table would have been edited by every branch, and every
edit would collide at merge time. The structure is now data instead.

## The three files

- **`content/chapters.json`** is the structure: `{ "<stageId>": [{ "name", "items": [ids] }] }`.
  The build reads it, the validator reads it, and both name it in their errors.
- **`scripts/import-b/chapters-from-map.mjs`** generates it from `MERGE-MAP.json`
  (`--map`, defaults to the merge map path, or `$MERGE_MAP`).
- **`scripts/build-content.mjs`** and **`scripts/validate-content.mjs`** consume it.

Nothing else reads the old literal. The sidebar (`src/app.js`) and every other
consumer read `content/ordering.js`, which the build emits from the data file.

## The placement rule

The map states the final structure. The filesystem states what is real today.
The generator reconciles them, and the reconciliation is deliberately
conservative, because the build derives a module's stage from the directory its
lesson file lives in:

- A module whose lesson file **is on disk** keeps the placement it has now.
  Listing it under another stage would fail the build (the file is not there),
  and listing it under another chapter would move it on the site before its
  file has moved.
- A module whose lesson file **is not on disk yet** is listed exactly where the
  map puts it, so the entry is already in place the day its file lands. No
  module branch ever has to edit `content/chapters.json`.
- A module on disk that the map does not name is carried forward where it is.
  The generator prints a `kept:` note for each.

Part 1 modules are placed in `order`. Part 2 modules are placed at their
`position` anchor ("after X"); when the anchor is not itself in the map, the
module is appended to the end of its chapter and the generator prints a note.
Five of the seven part 2 entries are also part 1 entries; their part 2 positions
are ignored with a note, since part 1 already places them.

## What the generator prints

Every decision that is not mechanical is reported: deferred moves (`deferred:`),
modules the map does not name (`kept:`), unresolvable anchors, and part 2
duplicates. Run it with `--dry-run` to read the notes without writing anything.

## Modes

```
node scripts/import-b/chapters-from-map.mjs [--map <path>] [--current <path>]
    [--out <path>] [--only-on-disk] [--dry-run]
```

`--current` is the file the on-disk modules keep (default `content/chapters.json`);
`--out` is the file written (same default). Splitting them lets a variant be
generated without touching the committed file. `--only-on-disk` emits only the
modules whose lesson files exist, which is the identity check.

## The identity proof

`content/chapters.json` as committed lists 117 module entries. Of those, 55 have
a lesson file and 62 are listed ahead of their files. The build prints a note for
each of the 62 and skips them, then emits a site for the 55.

With the committed 117-entry file, the build output is byte-identical to the
pre-refactor site. `content/ordering.js`, `content/catalog.js`,
`content/lessons.js`, `sitemap.xml` and `robots.txt` were captured before the
refactor and compared after it: all five match exactly, and the build prints 62
skip notes. The `--only-on-disk` variant was also generated and compared against
the old hard-coded `CHAPTERS` literal: identical for all four stages, 24
chapters, 55 items.

## Deferred placement changes (need an owner decision)

The map moves four modules that are on disk today. The generator keeps them
where they are and prints a `deferred:` note for each, so the site does not move
anything before its file moves:

| Module | Map target | Today |
| --- | --- | --- |
| `object-detection-fpn-yolo` | Round 2 / Computer Vision | Camp / Computer Vision |
| `vision-transformers-vit-attention` | Round 2 / Computer Vision | Camp / Computer Vision |
| `representation-learning-autoencoders-vaes-latent-spaces` | Camp / Generative Models | Camp / Representation Learning |
| `vae-encoder-decoder-generation` | Camp / Generative Models | Camp / Representation Learning |

The two stage moves need their files to move from `content/3_Camp` to
`content/2_Round_2` first. The two chapter moves are within Camp and need no
file move, but they were still deferred so the site stays identical until an
owner says otherwise.

16 on-disk modules are not named in the map and are carried forward (the
generator prints a `kept:` note for each): `advanced-linear-algebra-projections-svd-low-rank`,
`applied-information-theory`, `autoregressive-generative-models`,
`computer-vision-classification-augmentation`, `diffusion-models`,
`dimensionality-reduction`, `efficient-attention-long-context`,
`gqa-mla-attention-complexity`, `model-training`,
`modern-vision-architectures-resnet-efficientnet-transfer`, `multimodal-learning`,
`physics-informed-neural-networks`, `pytorch-essentials`,
`supervised-learning-foundations`, `transformer-architectures-gpt-autoregressive`,
`transformer-fundamentals`.

# The per-module readiness check (merge tooling, platform lane)

`npm run check:module -- <module id>` is the one command a module manager runs
before saying READY. It prints one line per check and exits non-zero if any
failed.

```
build        ok
validate     ok
figures      ok (3 named)
notebooks    ok (1 named)
problems     ok (5 named)
links        ok (4 to other modules)
em-dashes    ok
render 1440  ok (rendered "SVMs and Kernels")
render 375   ok (rendered "SVMs and Kernels")
```

## What each check does

- **build** and **validate** run the whole pipeline. They are global, so another
  lane's broken module fails your check too; that is deliberate, since the site
  is one build.
- **figures** / **notebooks** / **problems** walk this lesson's blocks and
  confirm every `<Figure src>`, `<NotebookCard id>` and `<ProblemCard id>`
  names a file or problem that exists.
- **links** resolves every `/module/?id=<id>` link in the lesson against the
  catalog and `content/redirects.json`. A link to a retired id is fine, because
  it redirects; a link to an id nobody knows fails.
- **b-notebooks** walks site B's `notebooks/<b_id>/` folder plus every
  `<NotebookCard notebookPath="...">` in the B lesson (B keeps some notebooks in
  shared folders: `round-2`, `classical-ml`, `deep-learning`, `modern-ai`), and
  requires each one to exist as `notebooks/<id>.ipynb` in A and to be reached by
  a `<NotebookCard id="<id>">` in the lesson or by a `nb=<id>` problem url.
  B's `-solution` notebooks are answers and are not carried over.
- **em-dashes** fails on any em dash outside the header block (the H1 and the
  `> ` metadata lines before the first `## ` section, which are template text).
- **render 1440** / **render 375** open the module page in headless Chrome, via
  `heavy.sh`, and require no console errors, no `.katex-error` element, nothing
  wider than the viewport, and that the reader actually rendered.

## Running it

The site must already be served for the render checks. One shared server serves
every worktree, so point the command at the copy it serves:

```
npm run check:module -- <module id>
AIO_BASE_URL=http://127.0.0.1:4200/<worktree folder name>/ npm run check:module -- <module id>
```

`--no-browser` skips the two render checks, and `--base <url>` (or the
`AIO_BASE_URL` env var) overrides where the page is read from. The Chrome pass
is `scripts/check-module-browser.py`; it needs `google-chrome` and the
`websocket-client` package, and it goes through `heavy.sh` when that is present.

The render checks fetch the module page from node once before they queue in
`heavy.sh`, so a base nothing serves fails right away instead of taking one of
the two machine-wide slots:

```
render 1440  FAIL  no server at http://127.0.0.1:4173/: use the shared server, AIO_BASE_URL=http://127.0.0.1:4200/<worktree folder name>/ (MODULE-PIPELINE.md step 7); never start one
```

Point `--base` (or `AIO_BASE_URL`) at the shared server that already serves the
worktree; lanes never start a server of their own.

## A note on the em dash check

46 of the 55 existing modules fail it today: their prose still uses em dashes.
The check is the gate L0 asked for, so a module that has not had its dashes
replaced is not READY. Two lines are exempt: the H1, whose title format uses an
em dash, and the `> **Stage:**` metadata line. The `> **Goal:**` line is not
exempt, because the site shows it as the module summary.

## Re-running the chapters generator
The generator is idempotent. Re-run it after a lesson file lands or moves:

```
node scripts/import-b/chapters-from-map.mjs
npm run build:content
npm run validate
```

If a module is listed under a stage whose directory does not hold its file, the
build fails with "filed under the wrong stage's chapters" and the validator
fails with the file named. That is the signal to move the file first, then
re-run.

# Redirects for retired module ids (merge tooling, platform lane)

When a B module takes over from an A module, `MERGE-MAP.json` carries the A id
in `redirect_from`. `content/redirects.json` is the flat old id -> new id table;
the module page honours it so an old link or bookmark still lands somewhere.

## The files
- **`content/redirects.json`** is the table (committed data file, 16 entries).
- **`scripts/import-b/redirects-from-map.mjs`** generates it from the map
  (`--map`, defaults to the merge map path, or `$MERGE_MAP`).
- **`content/redirects.js`** is generated output the module page loads
  (`window.AIO_REDIRECTS`), written by the build and committed like
  `content/lessons.js`. The site deploys straight from the repo with no build
  step (vercel.json: `buildCommand: null`, `outputDirectory: "."`), so a file
  the deployed page requests has to be in git. Module branches never stage it;
  the integration branch regenerates and commits it.
- **`module/index.html`**, **`src/app.js`**, **`scripts/validate-content.mjs`**
  consume it.

## When a redirect fires

The guard is deliberately one-sided:

```js
if (requestedId && window.AIO_REDIRECTS[requestedId] && !moduleLookup.has(requestedId)) {
  location.replace(`/module/?id=${encodeURIComponent(replacement)}`);
}
```

A redirect only fires when the requested id is **not itself a module**. While
both modules are on the site, the old one still renders; retiring it (deleting
its file and dropping it from `content/chapters.json`) is what arms the redirect.
That keeps the site identical today and makes each redirect active the day its
module goes.

`location.replace` rather than `assign`, so the redirecting URL is replaced in
history: the back button does not bounce back to it, and the canonical tag lands
on the new module. This is a static, client-routed site, so there is no server
301; the client redirect is the available equivalent.

## Validator checks

The validator loads `content/redirects.json` directly and fails on:

- a redirect that points at itself,
- a chain (`a -> b -> c`; `b` is itself a redirect source),
- a target that is not a module the chapter structure lists.

A source that is still a live module is not an error: it is the transition state
above.

## Today's 16 entries

All 16 sources are the carried-forward modules from the chapters tool: they arm
when retired.

| Retired A module | Replaced by |
| --- | --- |
| `advanced-linear-algebra-projections-svd-low-rank` | `linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory` |
| `applied-information-theory` | `information-theory-core` |
| `autoregressive-generative-models` | `gpt-causal-lm` |
| `computer-vision-classification-augmentation` | `convolutional-neural-networks` |
| `diffusion-models` | `generative-adding-noise` |
| `dimensionality-reduction` | `principal-component-analysis` |
| `efficient-attention-long-context` | `efficient-flashattention` |
| `gqa-mla-attention-complexity` | `efficient-sharing-keys-and-values` |
| `model-training` | `bias-variance-cross-validation` |
| `modern-vision-architectures-resnet-efficientnet-transfer` | `vision-deeper-networks` |
| `multimodal-learning` | `contrastive-learning` |
| `physics-informed-neural-networks` | `sciml-setup` |
| `pytorch-essentials` | `pytorch-tensors-autograd` |
| `supervised-learning-foundations` | `linear-logistic-regression` |
| `transformer-architectures-gpt-autoregressive` | `gpt-causal-lm` |
| `transformer-fundamentals` | `transformer-add-position` |

Two sources share one target (`gpt-causal-lm`), which is fine: no chains, since
no target is itself a source.

## Re-running it

```
node scripts/import-b/redirects-from-map.mjs
npm run build:content
npm run validate
```

The generator prints a note for anything odd: a duplicate source, a self-redirect,
or a target the chapter structure does not list. `--dry-run` prints without
writing.


# Step chapters in the sidebar and prev/next (merge tooling, platform lane)

Site B groups some modules into numbered step chapters: "Build a Transformer"
holds 14 modules, "Build a Vision Model" 8, "Efficient Transformers" 9. In B
those sit nested inside a topic. A's `content/chapters.json` is one flat list of
chapters per stage, so the map flattens the nesting: Round 2's "Transformers and
Attention" chapter carries all 17 modules in reading order, the 14 build steps
followed by the three architecture modules. The order survives, so prev/next
needs no work, but a chapter that long needed a look at the sidebar.

## What was wrong

The sidebar is a sticky column whose scroll area (`overflow-y: auto`) is the
viewport height, and every page load is a full navigation, so the list always
starts at the top. A reader on module 16 of 17 saw modules 1 through 9 instead
of their own. This was already visible with today's chapters: Round 1's six open
chapters are taller than the viewport, so "Clustering Methods" in the fourth
chapter was below the fold at both 1440 and 375.

## What changed

`src/app.js` now scrolls the sidebar to the active module after the reader
renders, and only when it is actually out of view. It moves the sidebar's own
scroll container by the offset between the item's centre and the container's
centre, so the surrounding items stay visible rather than parking the active one
at the edge. The sidebar scroll position is also captured and restored across
re-renders the way open details and picked options already were, so marking a
module complete no longer snaps the list back to the top.

Nothing else changed: the chapter names, the dot-and-rule list, the section
page's numbered rows and its progress meter (`0/17`) already read a long
chapter well, so no numbering or counters were added to the sidebar.

## How it was checked

A scratch `content/chapters.json` put 17 on-disk Round 1 modules in one chapter
(the real 17-module chapter's lessons are not on disk yet) and the build was
served locally. Measured, at both 1440 and 375:

- item 16 of 17 is scrolled into view, sidebar scrollTop 308 at 1440 and 610 at
  375, where before it was 0 and the item sat 863 px down;
- the section page shows one card with 17 numbered rows, no element wider than
  the viewport, page scrollWidth equal to the viewport width;
- zero console errors on seven module and section pages;
- with the real `content/chapters.json` back in place, `ordering.js`,
  `catalog.js` and `lessons.js` are byte-identical to the baseline, and the
  active item is visible on every module page.

Screenshots in `/home/jackwu/Desktop/hp/merge/shots/platform/`:
`v2-item3-1440.png`, `v2-item3-375.png` (the reader with a 17-item chapter),
`v2-item3-sidebar-full-1440.png` and `v2-item3-sidebar-full-375.png` (the whole
sidebar), `v2-item3-section-1440.png` and `v2-item3-section-375.png` (the
section page with the 17-module chapter card).

The scratch `content/chapters.json` was reverted; the committed file is
unchanged.

# The per-module problem banks (merge tooling, platform lane)

`content/problems.js` and `content/theory.js` were the merge bottleneck: every
module lane edited the same two files, and the split of concerns between them
was by card (index card vs. theory page), not by module. Both banks are now
assembled from per-module files, so a lane touches one file.

## The layout

- **`content/problems/<module id>.json`** is the source of truth for one
  module: `{ "module": <id>, "stage": <stage tag>, "index": [<problems.js
  entries>], "theory": [<theory.js entries>, `[]` when the module has none] }`.
  The file name must match `module`, and every entry's `moduleId` must be the
  module's own. Entries keep their exact fields and their bank order.
- **`content/problems/contests.json`** is `{ "contests": [...] }`, unchanged in
  shape.
- **`content/problems.js`** and **`content/theory.js`** are generated output the
  site loads (`window.AIO_PROBLEMS`, `window.AIO_THEORY`), written by the build
  and committed like `content/lessons.js` and `content/redirects.js`: the site
  deploys straight from the repo with no build step, so a file the deployed page
  requests has to be in git. Module branches never stage them; the build
  regenerates them.

## What a lane edits

A module's problems live **only** in `content/problems/<module id>.json`. Lanes
edit that file (`scripts/import-b/problems.py <B module> --spec <spec> --out
content/problems` writes it), never `content/problems.js` or
`content/theory.js`, and never another module's file. After editing it, run
`npm run build:content` (regenerates the two banks) and `npm run validate`.

A module's `stage` comes from `content/chapters.json`, not from the entries' own
tags: a handful of entries carry the tag of the module they were written for
(two modules on this branch), and the stage is a property of where the module
sits in the curriculum, so the entry tags are not authority. The validator
checks the tags against the chapters file (see below).

## Assembly order

`scripts/lib/assemble-problems.mjs` orders the module files by each module's
position in `content/chapters.json` (stage key order `round1`, `round2`, `camp`,
`advanced`, then chapter, then item), not by any ordering field in the module
files. The problems index lists problems in array order (`src/app.js`
`sortProblems`), so the default order now follows the curriculum. Today's
first-appearance order was arbitrary: the index opened on SVMs, and 444 of 1485
module pairs were inverted against the curriculum. Inside a module, the entries
stay in the file's own order.

A module file whose `module` id is not in `content/chapters.json` is a build
error naming `content/problems/<id>.json`: its entries belong in the file of the
module the chapters file lists instead (a redirected old id, for example).

## What the build and the validator guarantee

- The build writes both banks on every run, so the shipped files cannot fall
  behind the module files.
- The validator fails with `content/problems.js is out of date: run npm run
  build:content` (and the `theory.js` twin) when a committed bank is not exactly
  what the assembler builds from the module files, compared after canonicalising
  object key order. Array order is not normalised, because the index lists in
  it.
- The validator also fails when an index entry's stage tag is missing or
  differs from its module's `content/chapters.json` stage, naming
  `content/problems/<module id>.json` and the entry id. Theory entries carry no
  stage tag, so they are skipped. This is the backstop for an import that tags a
  module for the wrong round.

## The round-trip proof

`scripts/import-b/proof-split.mjs` loads the two banks, splits them into
scratch, reassembles, and compares after `JSON.parse`:

```
node scripts/import-b/proof-split.mjs
```

It checks that every module's `index` and `theory` arrays are deep-equal to the
original subsequence for that module, that the assembled entries are the same
multiset globally, that the contests are equal, that the module order is the
chapters order, and that the `.js` files the assembler writes, loaded back the
way the site loads them, give the same data. What it cannot claim is
byte-identical global order: the old `problems.js` interleaves modules in 133
runs over 54 modules, and a per-module split has one run per module. Order
inside a module is exact.

