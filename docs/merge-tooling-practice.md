# Merge tooling: practice lane

Tools this lane built for the content merge. Each is committed on
`merge/mlp-practice`; nothing here is generated content, and none of the split
files they write to scratch are committed.

## scripts/import-b/problems.py: B's problems to A's data

Converts B's problem files (`content/problems/<B module id>/*.mdx`) into the
index and theory entries A's two banks hold. B's text is copied as written; the
only edits are the confirmed corrections a module spec lists.

```
python3 scripts/import-b/problems.py <B module id> [--out FILE|DIR]
                                    [--into content/problems/<id>.json]
                                    [--module <A module id>]
                                    [--stage <stage>] [--spec PATH] [--strict]
```

- No `--out` prints to stdout. `--out FILE.json` writes that file.
  `--out DIR` writes `DIR/<A module id>.json`, which is the same file name the
  per-module layout uses (below), so an imported module drops straight into
  `content/problems/`.
- `--into content/problems/<id>.json` merges this run's entries into the
  module file that already exists, which is what wave 2 lanes use (below).
  Only one of `--into` and `--out`.
- `--stage <stage>` sets the stage tag. Its default is the precedence below,
  not a fixed value.
- `--strict` exits 1 on the first warning. Warnings always go to stderr and
  never into the JSON.

Output shape, one file per module:

```json
{ "module": "<A module id>", "stage": "<stage>",
  "index":  [ <problems.js entries> ],
  "theory": [ <theory.js entries> ] }
```

The stage tag comes from the first of these that names one, and the tool exits
2 if none does (it never silently tags a module "Round 1" again):

1. `--stage <stage>`
2. the spec file's `"stage"` (`"Round 1"`, `"Round 2"`, `"Camp"` or `"Advanced"`)
3. the stage `content/chapters.json` puts the A module in

When the spec's stage and the stage in `content/chapters.json` disagree the tool
warns naming both, which `--strict` treats as an error.

What it decides itself, and warns about every time:

- **MCQ entry**: `source` "Custom", tags `["Theory", "Multiple choice", stage]`,
  `url` `/theory/?id=<id>`, `prompt` the stem's paragraphs as plain text,
  `answer` the correct choice's text, period-normalised, then its explanation.
- **Written entry**: `source` "USAAIO-style", tags
  `["Theory", "Multi-part", stage]`, `prompt` the intro paragraphs plus
  "N parts.", `answer` the fixed sentence pointing at the problem page.
- **Part kinds**: a part with a numeric `answer` is `computation` with
  "Reasoning is not required.", any other written part is `derivation` with
  "A complete derivation is required." `kind` and `requirement` are A's grading
  metadata, not B's text.
- **Plain text for cards**: the index card renders `prompt` and `answer` with
  `escapeHtml` and no KaTeX, so inline math is flattened (`\frac{a}{b}` to a/b,
  `\mathbf{x}` to x, accents to combining marks, Greek to names). A backslash
  the flattener does not recognise survives and warns.
- `tolerance` is dropped: A's theory page reveals answers and never auto-judges.
- Difficulty "Very hard" is normalised to "Very Hard".

### The module spec

`scripts/import-b/modules/<B module id>.json`. Every key is optional unless the
run needs it:

| key | holds |
|---|---|
| `aModule` | A's module id for this folder |
| `stage` | stage tag |
| `order` | entry ids in the order they should be emitted |
| `fixes` | `[{file, find, replace, why}]`, confirmed corrections applied before conversion |
| `parts` | `{id: {"1.1": {kind, requirement}}}`, overriding the defaults |
| `indexPrompt` | `{id: prompt}`, replacing the mechanical card prompt |
| `source` | `{id: label}`, when a card's source is not the kind's default |

`scripts/import-b/modules/multilayer-perceptron.json` is the finished spec for
the MLP module and the model to copy: 2 fixes, 5 part overrides, 2 indexPrompt
overrides.

### scripts/import-b/proof-mlp.mjs

The control. Runs the importer on `multilayer-perceptron` and compares the
output to the entries already committed in `content/theory.js` and
`content/problems.js`, key-order-insensitively. All 9 theory and 9 index
entries are deep-equal, so the defaults above reproduce a hand-reviewed module
exactly. It also names the 4 notebook cards the importer does not produce
(those come from the notebook converter, lane toolnb).

```
node scripts/import-b/proof-mlp.mjs
```

### Index markup: indexPrompt and indexAnswer

The problems index, the lesson focus card and site search print a card's `prompt` and `answer`
with `escapeHtml` and no markdown, so `$...$` math and backticks reach the reader as literal
characters. The importer warns on any emitted index field that still holds either:

```
warning: statistical-inference: si-three-labs: index prompt has math or code markup; write indexPrompt in the spec (Three labs each measure the same unknown concentration $\mu$...)
```

Under `--strict` that exits 1. The fix is a spec entry, nothing else in the problem changes:

- `indexPrompt` (`{id: prompt}`) replaces the card's prompt, for a written problem or an MCQ.
- `indexAnswer` (`{id: answer}`) replaces an MCQ card's answer, and exists for the same reason.

A dry run over all 88 folders with no specs flags 232 prompts (203 carry math, 37 carry code
markup, 8 carry both; no answer is flagged). The platform lane is adding the same check to
`scripts/validate-content.mjs`, so a lane that merges with the warning outstanding fails
validate.

### scripts/import-b/proof-colab.mjs

A second control, added when the `m-colab` pilot reported four bugs in an
older fork of the importer (a numbered-list line sent the fork into an
unbounded loop that reached 17 GB; the pilot fixed the four by hand). This
proof runs the importer on `colab-notebooks` with the pilot's committed spec
and compares against the entries `merge/m-colab` committed: 5 theory and 5
index entries are deep-equal, and the field order of every entry matches the
committed entries key for key. It then re-runs the importer with the spec's
`fixes` removed so B's numbered list reaches the block parser itself, and
checks the run terminates and converts the list to `{"olist": [...]}` with
the words unchanged. Each of the four bugs has a case that fails loudly if it
returns.

```
node scripts/import-b/proof-colab.mjs
```

### scripts/import-b/test-blocks.py

Unit tests for `blocks()`, the markdown-to-block-list function: 18 adversarial
inputs (numbered lists against fences, tables, bullet lists, display math and
end-of-file; unclosed fences and math; `<center>` blocks; empty bodies) plus
B's own five-item list from `cn-audit-before-you-submit.mdx`. Two properties
are asserted: the run terminates, and the number of emitted blocks never
exceeds the number of non-blank input lines, which is exactly the invariant
the fork's stuck loop broke. Run it under a timeout; a hang is a failure.

```
timeout 60 python3 scripts/import-b/test-blocks.py
```

### Dry run and inventory

`scratch/practice/dry_run.py` runs the importer over all 88 of B's problem
folders into `scratch/practice/import-dry/` and classifies every warning. The
report of that run is `merge/PROBLEM-IMPORT-INVENTORY.md`: 470 files, 470 index
and 470 theory entries, 1004 parts (172 mcq, 298 written), zero nonzero exits,
and a taxonomy of all 1817 warnings. 87 of the 88 modules still need a spec;
the inventory's per-module table is the work list.

Run it under the memory cap so a stuck loop dies instead of taking the
machine: `capped.sh 3 python3 scratch/practice/dry_run.py`. The last run under
the cap had exit 0 and totals unchanged.

## Per-module problem files

The two hand-maintained banks are split by module so each module lane owns one
file. The layout is proposed in `merge/CONTRACT-CHANGES.md` (2026-10-10 08:35
AM); the split is written and proven but not applied, and the split files are
not committed.

`content/problems/<module id>.json` per module:

```json
{ "module": "<module id>", "order": 3, "stage": "Round 1",
  "index":  [ <problems.js entries for this module> ],
  "theory": [ <theory.js entries for this module> ] }
```

`content/problems/contests.json` is `{contests: [...]}`. Entries are copied
verbatim; `stage` is the one stage tag the module's entries already carry,
`order` its first position in today's problems.js. Files without `order` sort
by file name, which is where imported modules land.

### Merging a module's entries in: `--into`

Once the per-module files exist, a module lane merges B's entries into its own
file, which may already hold A's kept entries and notebook cards from the
notebook converter. That is what `--into` is for, so every lane does it the
same way instead of writing a splice script:

```
python3 scripts/import-b/problems.py <B module id> --into content/problems/<A module id>.json
```

- The file does not exist: it is created as
  `{"module", "stage", "index", "theory"}` with this run's entries.
- The file exists: every entry whose id this run emits is replaced in place,
  ids the file does not have are appended after its own entries in the order
  the run emits them (B's order), and every other entry and every other key
  (`module`, `order`, `stage`) is left untouched.
- The formatting is what `scripts/split-content.mjs` writes, so a merged file
  diffs clean against a split one. Running the import twice leaves the file
  byte-identical.
- A file holding another module is refused, not written.

### scripts/split-content.mjs

The one-off that writes today's banks into the new layout. `--out DIR` writes
there (scratch until the platform lane applies it); without it, it writes
`content/problems/`.

```
node scripts/split-content.mjs --out /home/jackwu/Desktop/hp/merge/scratch/practice/split
```

It throws rather than guesses: a module whose entries carry no stage or more
than one, a theory entry whose module is absent from problems.js, a module id
with no usable file name.

### scripts/lib/assemble-problems.mjs

The module the platform build will call. Dependency-free, throws on anything
that would build a half bank: a module file whose name does not match its
`module`, an entry whose `moduleId` is not the file's module, a duplicated id,
a `contests.json` without a contests array.

```js
import { assembleProblems, assembleTheory, moduleIds, writeContentFiles }
  from "./scripts/lib/assemble-problems.mjs";

assembleProblems("content/problems");  // -> { problems, contests }
assembleTheory("content/problems");    // -> { problems }
moduleIds("content/problems");         // -> module ids in assembly order
writeContentFiles("content/problems", repoRoot);  // writes the two .js files
```

`writeContentFiles` emits the same IIFE wrappers the site loads
(`window.AIO_PROBLEMS = { problems, contests }`, `window.AIO_THEORY = { problems }`),
so consumers see no change until the two .js files are deleted.

### scripts/import-b/proof-into.mjs

The `--into` proof, 13 checks, all passing:

```
node scripts/import-b/proof-into.mjs
```

- Merging into the module file `split-content.mjs` writes for the MLP module
  leaves it **byte-identical**. That file holds the 9 theory cards this
  importer reproduces exactly, the 9 theory entries, and 4 notebook cards from
  the notebook converter the importer never emits, so the check covers both
  "reproduces the module's entries" and "leaves the rest alone".
- Running the import again is byte-identical.
- With no file there it creates `{"module","stage","index","theory"}`, holds
  the module's 9 index and 9 theory entries, and a second run is
  byte-identical.
- Entries the file already has stay where they were; ids it lacked are
  appended after its own entries in B's order; a notebook card sitting between
  two dropped ids is untouched; as a set the file then equals the reproduced
  module file.
- A file holding another module is refused and left alone.

The importer is run under `capped.sh 3`.

### scripts/import-b/proof-split.mjs

The round-trip proof: loads today's two .js files, splits to scratch,
reassembles, and compares after JSON.parse. All 10 checks pass.

```
node scripts/import-b/proof-split.mjs
```

What is proven: every module's index and theory arrays are deep-equal to the
original subsequences for that module, the entries are the same multiset
globally, the contests are equal, the module order is the original
first-appearance order, and the reassembled .js files load to the same data.

What is not, and cannot be: byte-identical global order. Today's problems.js
interleaves its 55 modules in 133 runs, and a per-module split has one run per
module. Order within a module is exact.
