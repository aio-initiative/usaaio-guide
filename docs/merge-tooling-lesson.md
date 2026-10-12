# Merge tooling: the lesson lane

This is the stage 2 lesson tooling on `merge/mlp-lesson`. It supersedes `docs/merge-tooling.md`,
which is left untouched on this branch because it collided with another branch's change; L0
resolves that file at merge. Tools are in this repo, proofs and notes in
`/home/jackwu/Desktop/hp/merge/`. Run everything from the repo root.

The contract the importer keeps: **no sentence is changed.** Prose, math, code, tables,
captions and titles pass through verbatim. Only the container syntax is converted, and B's
~80-column wrapping is reflowed into the one-paragraph-per-line style this site's files use
(whitespace only; the reader joins wrapped lines the same way). `check_verbatim.py` is the
gate that proves it, and the 94-lesson dry run is the proof at scale
(`/home/jackwu/Desktop/hp/merge/LESSON-IMPORT-INVENTORY.md`).

## scripts/import-b/lesson.mjs: B lesson to A lesson

```
node scripts/import-b/lesson.mjs <B mdx file> <A md file> [A content dir]
node scripts/import-b/lesson.mjs --from-map <b_id> [--map MERGE-MAP.json] [--b-repo <B repo>]
```

The first form converts one file. The second looks up `<b_id>` in the merge map, reads the B
source from the map's `b_path` and writes the map's `file` (a full `content/<stage>/<file>.md`
path). `B_REPO` defaults to `/home/jackwu/Desktop/Repos/USAAIO-Guide` and `MERGE_MAP` to
`/home/jackwu/Desktop/hp/merge/MERGE-MAP.json`; both are env-overridable as well as flags.
Every line of output goes to stderr.

`--from-map` resolves its output path from the script's own location, so run it inside a
scratch copy of the worktree unless you mean to write the real repo.

### What it converts

| B | A |
|---|---|
| frontmatter `title` / `description` / `tier` | `# <n> · <CODE> · <title>`, `> **Goal:**`, `> **Stage:**` (the number and code come from the output filename). The H1's second separator is a middle dot; the parser also accepts the older em dash form, so A files written before the change still give up their titles |
| frontmatter `prerequisites` (B ids) | `## Prerequisites` with `- [A title](A file)` links, in B's order |
| `## Resources` + `<Resources>` near the top | the same block, unindented, moved to the end of the file |
| `<FigureBox image="..." imageDark="..." alt="...">caption</FigureBox>` | `<Figure src="assets/module-media/<B image basename>" alt="...">caption</Figure>`; `imageDark` is dropped (no dark variants on this site) |
| `<ProblemCard id="..." />`, `<Resources>`/`<Resource ...>`, `<Optional title="...">` | identical syntax |
| `<NotebookCard notebookPath="notebooks/<dir>/<name>.ipynb" />` | `<NotebookCard id="<name>" />` |
| `<Warning>`, `<Info>` | identical syntax, passed through untouched; the site parses both as B writes them |
| `<EmbeddedPyEditor packages={['numpy']} initialCode={`code`} />` | `<EmbeddedPyEditor packages="numpy">` then the code as plain lines, then `</EmbeddedPyEditor>`. The code is copied byte for byte, with only leading and trailing blank lines trimmed. The `packages` array is joined with commas; the attribute is omitted when B has none. Any other attribute is a warning naming the file and line |
| ` ```py ` fence | ` ```python ` |
| a plain fence directly after a python/py fence | ` ```output ` (measured over all 94 lessons: every one of the 322 python/py fences is followed by exactly one plain fence holding its printed output) |
| ` ```latex ` / ` ```markdown ` fences | kept as fenced code with their language (B uses latex fences for "type this" examples next to the math they produce; unwrapping them into display math was defect 5) |
| `[label](/round-1/<id>)` and the other tiers | `[label](<A file>)` via the module table in `links.json` |
| ` ```text `, tables, math, lists | unchanged |

Figure paths are mechanical: `<Figure src="assets/module-media/<basename of B's image>">`.
The tool draws nothing; after writing the lesson it prints the image paths the lesson needs,
which is what the figure lane redraws.

### What it warns about

Every warning names the file and the line, and every one is expected at this stage of the
merge. Counts below are from the 94-lesson dry run.

- **link not merged yet** (88): a prerequisite or inline link points at a module whose A file
  is not in this copy; the link is written and reads as plain text until that module lands.
- **frontmatter redirects** (37 lessons): B's frontmatter carries `redirects`, which A does
  not use. Dropped.
- **no prerequisites in B** (48 lessons): 39 omit the `## Prerequisites` section with the note
  "no prerequisites in B; section omitted", and 9 keep the A file's existing bullets verbatim.
  The platform lane is making the section optional, so an omitted section is not an error; no
  placeholder is added.
- **NotebookCard mapped by filename stem** (15): no `links.json` notebook entry, so the id is
  the notebook's basename; confirm the notebook exists on A's side.
- **anchor links** (11): a `#anchor` is dropped (A has no anchors); the link keeps its target.
- **other, a few each**: a `<center>` wrapper dropped (the table it held is unchanged), a
  plain fence after a python block whose body looks like code rather than output.

It throws (no output written) on a frontmatter or filename it cannot read, a tag that never
closes, an unknown component tag, or an `EmbeddedPyEditor` that never closes.

### The six defects found by the pilots and the dry run

Each is fixed in `scripts/import-b/lesson.mjs` with a regression case among the four pilots:
`multilayer_perceptron`, `probability_and_bayes`, `colab_notebooks`, `data_visualization`.

1. **Dropped blank lines between paragraphs.** The paragraph buffer flushed on every blank
   line, so two paragraphs separated by a blank line became one. Regression: MLP's Overview.
2. **Splitting a paragraph at a wrapped line starting with digits and a full stop.** B writes
   "which is / 297. Put the truth..." across a line break; the importer treated "297." as a
   numbered list item and split the sentence. A list item now starts a new line only at a
   block boundary or when the buffer already holds a list. Regression: probability, line 297.
3. **`<Warning>` carrying its own tag into the blockquote body.** Regression: MLP line 346.
4. **An empty `## Prerequisites` when B declares none.** The section is inherited verbatim from
   the A file being replaced when it exists, and omitted with a plain note when it does not;
   the platform lane is making an omitted section valid, so nothing is invented. Regression:
   probability (inherited from A's `1.10-PS1`), colab (from `1.4-PY4`).
5. **Unwrapping ` ```latex ` fences into display math.** Regression: MLP and colab.
6. **Joining consecutive list items onto one line when an item wraps onto a second line.**
   The buffer's kind is tracked explicitly instead of inspecting its last line for a marker,
   because a wrapped item leaves plain text there. Regression: `data_visualization.mdx`,
   which has five such lists.

### scripts/import-b/links.json and scripts/import-b/fill-links.mjs

`links.json` holds the remap tables keyed by B identifiers: `modules` (B id -> A lesson
filename), `notebooks` (B notebookPath -> A notebook id) and `problems` (B problem id -> A
problem id). `fill-links.mjs` writes only `modules`, sorted by B id, from `MERGE-MAP.json`,
preserving `notebooks`, `problems` and `_comment`; it prints the entry count (95 today).

## scripts/import-b/check_verbatim.py: the verbatim gate

```
python3 scripts/import-b/check_verbatim.py <B lesson.mdx> <A lesson.md> [--json] [--links links.json]
```

Exit 0 only when no sentence changed and no sentence is missing. It splits each side's prose
into sentences (links stashed first, so a period inside a link label like B's
"[Build a Generative Model: 2. VAE]" cannot split a sentence) and compares them as multisets,
with a second pass that masks link labels and targets. A sentence that matches only after
masking is a remapped link, reported under `labels` and **not** counted as changed.

Everything that is not prose is compared as a unit in its own section, so a clean import
reports 0 changed and 0 missing with the differences that remain listed separately:

- **header**: A’s title, Goal, Stage and `## Prerequisites` preamble against B’s frontmatter.
  B's frontmatter. Prerequisites are `ok`, `inherited` (B declares none, A's file supplied
  them) or `CHECK`.
- **figures**: by the image's basename, since the importer rewrites the directory
  mechanically. B's `imageDark` is noted as dropped, not flagged.
- **code fences**, **math**, **tables**, **cards** (ProblemCard / NotebookCard ids, the
  notebookPath-to-id conversion mirrored), **resources**, **headings** (A's inserted
  `## Prerequisites` excluded; the `## Resources` move makes the orders differ by design) and
  **tags** (`<center>` reported as dropped by design).
- **`<EmbeddedPyEditor>` bodies** are code, not tags and not prose. B's opening line has no
  `>`, so the gate reads the `initialCode` template literal itself and the tag body on A's
  side, and compares both as one `python` code unit; the tag is never reported as A-only.
  A converted lesson reports `tags 2/2 a_only 0` and `code` blocks differing 0.

`--json` emits the whole report for scripting; the dry run uses it.

## Proof

**Pilot lessons.** All four pilots import, build and validate, and pass both checks:

- the render check (`/home/jackwu/Desktop/hp/merge/scratch/lesson/render-check.mjs`, scratch
  only) confirms that every paragraph of B is one paragraph block of the built lesson, word
  for word: MLP 73/73, probability 63/63, colab 42/42, data visualization 47/47;
- `check_verbatim.py` reports 0 changed and 0 missing sentences for all four, and catches a
  deliberate one-word mutation (changed=1, missing=1, FAIL).

**All 94 lessons.** `/home/jackwu/Desktop/hp/merge/LESSON-IMPORT-INVENTORY.md` is the dry run:
94 of 94 imported and every one reports 0 changed and 0 missing sentences, with no A-only unit
anywhere (the single B-only unit is one `</center>` the importer drops by design). 72 lessons
reference 173 image files under `assets/`.

The build in that copy carries scratch-only stubs for what the other lanes have not landed yet:
the 322 `<ProblemCard>` ids and 15 notebook ids that do not exist yet, plus the two platform
changes that are announced but not merged (the H1 middle-dot separator with the em dash kept as
a legacy form, and the `## Prerequisites` section becoming optional). With those in place 112
modules build and the build reports no error at all; remove the stubs and the only failures are
the missing problem and notebook cards. No failure is a tag, an H1 or a prerequisites section.

**Items 1 to 5 (2026-10-10).** Proven in `/tmp/opencode/lp3`, a scratch copy of the worktree:

- `python-basics` (six `<EmbeddedPyEditor>` tags, none with `packages`) and
  `k-means-clustering` (`packages={['numpy']}`) convert, and all six code bodies in
  `python-basics` are byte-identical to B's `initialCode` template literals.
- The build parses the converted tags; `scratch-build-errors.json` holds no tag, H1 or
  prerequisites error.
- `check_verbatim.py` on `python_basics.mdx` reports `tags: B=2 A=2 a_only 0` and
  `code: B=21 A=21 blocks differing=0`, verdict PASS. Before item 3 the same lesson reported 12
  A-only tags.
- The importer writes `# 1.23 · PY7 · Python Basics`, and `titleFromH1` parses both that and
  the older em dash form, so existing A files still give up their titles.
- `git grep -c $'\u2014' -- scripts/import-b` prints nothing; the file holds no em dash, and
  the one the regex matches is written as the escape `\u2014`.

The MLP lesson's own proof is the diff of the imported file against the hand-merged one,
`/home/jackwu/Desktop/hp/merge/scratch/lesson/imported-vs-hand.diff`: every difference is a
correction logged in `scratch/lesson/decisions.md`, an A extra named in the merge map, figure
src/alt text, or a link label, and there is no fifth kind.

## How a lane runs a rollout

1. Copy the worktree to a scratch directory (`cp -a`; `--from-map` writes into the copy it
   runs in).
2. `node scripts/import-b/lesson.mjs --from-map <b_id>` per lesson, reading the warnings.
3. `node scripts/import-b/check_verbatim.py <B mdx> <A md>` per lesson; anything other than
   0 changed / 0 missing is a hand comparison.
4. Add new module ids to the chapter lists, supply the printed image assets, then
   `npm run build:content` and `npm run validate`. A lesson whose B source declares no
   prerequisites simply has no `## Prerequisites` section once the platform lane lands the
   optional-section change; add none by hand.
5. Commit the lesson files; never commit the scratch copy.

## What the importer cannot do

It does not add A extras, apply a module's confirmed corrections, write alt text, draw
figures, convert notebooks, or choose link labels; the merge map and the other lanes own those.
It does not touch B's wording: `check_verbatim.py` is the gate that proves it.
