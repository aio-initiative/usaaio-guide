# Merge tooling

Tools the module managers use in stage 2. Each section is one tool: what it does,
how to run it, and what it cannot do. One section per lane; add yours, do not
edit another lane's.

## Lesson lane: the lesson importer and the verbatim check

Built on `merge/mlp-lesson` for stage 2 of the merge. Tools in this repo, proof and
notes in `/home/jackwu/Desktop/hp/merge/`. Run them from the repo root.

### scripts/import-b/lesson.mjs: B lesson to A lesson

```
node scripts/import-b/lesson.mjs <B mdx file> <A md file> [A content dir]
```

Reads one B lesson (MDX, YAML frontmatter, B's components) and writes the A lesson
file with this site's contract-1 tags. The optional third argument is the A directory
the sibling lessons live in, used to read the titles for the Prerequisites list; it
defaults to the output file's own directory.

**No sentence is changed.** Prose, math, code, tables, captions and titles pass through
verbatim. The tool only converts the container syntax, and it reflows B's ~80-column
wrapping into the one-paragraph-per-line style this site's files use (whitespace only;
A's own lessons are written that way, and the reader joins wrapped lines the same way).

### What it converts

| B | A |
|---|---|
| frontmatter `title` / `description` / `tier` | `# <n> · <CODE> — <title>`, `> **Goal:**`, `> **Stage:**` (the number and code come from the output filename) |
| frontmatter `prerequisites` (B ids) | `## Prerequisites` with `- [A title](A file)` links, in B's order |
| `## Resources` + `<Resources>` at the top | the same block, unindented, moved to the end of the file |
| `<FigureBox image="..." imageDark="..." alt="...">caption</FigureBox>` | `<Figure src="assets/module-media/<B image basename>" alt="...">caption</Figure>`; `imageDark` is dropped (this site has no dark variants) |
| `<ProblemCard id="..." />`, `<Resources>`/`<Resource ...>` | identical syntax |
| `<NotebookCard notebookPath="notebooks/<dir>/<name>.ipynb" />` | `<NotebookCard id="<name>" />` |
| `<Optional title="...">` | identical syntax |
| ` ```py ` fence | ` ```python ` |
| a plain fence directly after a python/py fence | ` ```output ` (measured over all 94 lessons: every one of the 322 python/py fences is followed by exactly one plain fence holding its printed output) |
| ` ```latex ` fence whose body is one `$$ ... $$` block | unwrapped to a display-math block |
| `[label](/round-1/<id>)` and the other tiers | `[label](<A file>)` via the module table in `links.json` |
| ` ```text `, ` ```markdown `, tables, math, lists | unchanged |

Figure paths are derived mechanically: `<Figure src="assets/module-media/<basename of B's image>">`.
The tool does not draw anything; after writing the lesson it prints the list of image
paths the lesson needs, which is what the figure lane redraws.

`<Warning>`, `<Info>` and `<EmbeddedPyEditor>` (20 + 1 + 20 uses across B's 94 lessons)
have no place on this site yet: the platform lane's item 6 adds them under those same
names. Until it lands the importer passes all three through untouched, so nothing is
lost, and reports every use as "needs platform item 6".

### What it warns about

Every warning goes to stderr and names the file and line.

- a link to a B module that has no A target in `links.json` yet: the link is replaced by
  its label as plain text (B's sentence survives, no broken link);
- a link carrying an anchor: the anchor is kept in the source, but A's link rewriter
  drops anchors, so it needs a manual check;
- a `<ProblemCard>` id that is neither in `links.json` nor in `content/problems.js`;
- a `<NotebookCard>` mapped by filename stem only (no `links.json` entry);
- a `<Resource>` that does not close itself on one line, an attribute value containing a
  double quote (A's tag parser cannot read it), a latex fence that is not one `$$` block,
  a plain fence after a python block whose body looks like code rather than output,
  a `<center>` wrapper (dropped; the table it held is unchanged), frontmatter `redirects`
  (unused on A), a missing `<Resources>` block, an unclosed `$$` block;
- the three tags above, each reported as "needs platform item 6".

It throws (no output written) on a frontmatter or filename it cannot read, a tag that
never closes, an unknown component tag, or an `EmbeddedPyEditor` that never closes.

### scripts/import-b/links.json

The remap tables, keyed by B identifiers: `modules` (B id -> A lesson filename),
`notebooks` (B notebookPath -> A notebook id) and `problems` (B problem id -> A problem
id, identity by default). L0 fills it from `MERGE-MAP.json` as modules are merged; today
it holds the five module links and the one notebook link mapped by hand for the MLP
lesson, and an empty `problems` table, because the MLP lesson's nine problem ids already
exist on A and pass through unchanged. The `_comment` keys are skipped.

### Proof for the MLP lesson

`node scripts/import-b/lesson.mjs ~/Desktop/Repos/USAAIO-Guide/content/1_Round_1/multilayer_perceptron.mdx <out> content/1_Round_1`
re-creates `content/1_Round_1/1.18-DL1-neural-networks-perceptrons-mlps-activations.md`.
The diff is in `scratch/lesson/imported-vs-hand.diff` (205 lines). Every difference is one
of the four kinds below; there is no fifth kind, and no difference is a bug.

**1. Corrections logged in `scratch/lesson/decisions.md`** (L0's corrections and the stage
2 fixes; the importer cannot know about them, because they are this module's confirmed
corrections, not conversions). Examples: "simplest neural network" -> "simplest neural
network with hidden layers"; "the part every larger network is built from" -> "a part of
most larger networks"; pixels "a number from 0 to 1" -> "a whole number from 0 to 255";
"94.8%" -> "94.76%"; inputs `$a_1$` -> `$x_1$`; "a bias $b$" -> "one **bias** $b$"; "large
and positive"; "0 to 1 for this sigmoid"; "use ReLU, or a variant of it"; "negative for
every training input"; "the signal training needs to move a weight"; "Transformers such as
BERT and GPT-2"; "It holds for inputs in a closed, bounded range"; "a grid fine enough for
error $\varepsilon$"; the split of the collapse paragraph into three; "### Inside
`nn.Linear`" in B's position.

**2. A extras named in the merge map** (added by hand, not conversions): the
activation-slope table and its lead-in sentence, `$\tanh(z) = 2\sigma(2z) - 1$` and the
note about the three panels' scales, the `mlp-hinges` figure and its caption, "Each row is
the same $W\mathbf{x} + \mathbf{b}$", "softmax (10 non-negative numbers that sum to 1)",
"the same code with 128 and 64 neurons (109,386 parameters...)", the quadratic-activation
sentence with its $1/32$ bound, and the `$K = 2$, error 0.01 needs about 100 units"
worked example.

**3. Figure src and alt text.** The tool writes `assets/module-media/<B basename>`:
`mnist-digits.png`, `neuron.png`, `sigmoid.png`, `activation-functions.png`,
`mlp-network.png`, `relu-regions.png`, `relu-sawtooth.png`. The hand-merged file uses
`mlp-` prefixed names (and `.svg` for the six that were redrawn) and alt text rewritten
for the redrawn figures, since the figures themselves were redrawn in this site's look.
Alt text belongs to the figure asset, so the figure lane writes it; the tool carries B's
alt text through unchanged.

**4. Link labels.** The tool keeps B's labels: "logistic regression", "Forward
Propagation", "PyTorch Tensors & Autograd", "convolutional network". The hand-merged file
used the A module titles ("Supervised Learning Basics", "Matrix Backpropagation", "CNN
Fundamentals"). Per L0's rule that the other modules get B's wording as B wrote it, the
tool's behaviour is the right one; the MLP lesson's labels are a hand change, not a
conversion, and the merge map decides each module's labels.

**Structural check.** Both files have 20 headings, 8 `<ProblemCard>`, 1 `<NotebookCard>`,
3 `<Optional>`, 6 `<Resource>` in 1 `<Resources>`, and identical fence counts (13 python,
12 output, 25 plain). The only tag count that differs is `<Figure>` (7 vs 8), the extra
being the A-extra `mlp-hinges`.

The three links with no A module yet (`loss-functions`, `optimizers-training-tricks`,
`training-loop-in-pytorch`) come out as plain text with a warning, as designed; the
hand-merged file drops them the same way, in sentences rewritten for the A extras.

### Build and validation proof

The imported MLP lesson was built and validated in a throwaway copy of this worktree
(`/tmp/opencode/lesson-build-1`, copy with `cp -a`, never committed), with the importer's
output written over the real lesson file there:

- `npm run build:content` -> "Built 55 modules across 4 stages (24 chapters, 7813 lesson
  blocks)", exit 0;
- `npm run validate` -> "Content validation passed for 55 modules, 197 problems (20 theory,
  142 parts), and 4 contests", exit 0.

That the imported file was actually parsed, not silently skipped, is provable from the
generated content: `content/lessons.js` in the copy contains "part every larger network is
built from" (B's uncorrected Overview sentence, which the hand-merged file does not have)
and zero occurrences of the hand-merged file's "a part of most larger networks".

Neither script checks that a `<Figure src>` asset exists or that a lesson's `<ProblemCard>`
id is in the problem bank, so the missing `mnist-digits.png`-style assets and the eight
`mlp-*` problem ids fail nothing here; both are the importer's expected warnings, and both
are resolved by the figure lane and the merge map.

### scripts/import-b/check_verbatim.py

See its own section below (added with the gate).

### What the importer cannot do

It does not add A extras, apply a module's confirmed corrections, write alt text, draw
figures, convert notebooks, or choose link labels; the merge map and the other lanes own
those. It does not touch B's wording: `check_verbatim.py` is the gate that proves it.


## Figures: redrawing site B's figures in site A's look

Site B draws every figure through one shared style module,
`tools/diagrams/usaaio_theme.py`. Every generator imports it and nothing else for
colour, type, stroke or size, and no generator names a hex value of its own (the
survey checked all 30). That makes the whole figure pipeline a swap of one module.

Two files do it, both on `merge/tool-fig`:

- `scripts/figures/b_theme.py` is A's look written with B's API: the same names,
  the same signatures, A's palette, fonts, line widths and sizes behind them.
- `scripts/import-b/figures.py` is the runner. It imports a B generator unchanged
  with `usaaio_theme` resolved to `b_theme`, so the generator draws its own figure
  on A's white page.

### How to run it

```bash
# one B generator, its figures into a scratch folder
python scripts/import-b/figures.py gen_mlp_figures.py --out /tmp/opencode/figs

# one figure from a generator
python scripts/import-b/figures.py gen_mlp_figures.py --only sigmoid --out /tmp/opencode/figs

# machine-readable result: one entry per figure with the checks it passed
python scripts/import-b/figures.py gen_mlp_figures.py --json --out /tmp/opencode/figs
```

`--prefix b-` keeps trial output apart from the module's real figures. Heavy runs
go through `/home/jackwu/Desktop/hp/merge/heavy.sh`, one at a time.

Every output lands next to the lesson that uses it under `assets/module-media/`,
as SVG for diagrams and 2x PNG for dense plots. The split is decided at save time
from what the figure actually drew: an image with a real colour ramp, or a grid of
250 thousand cells or more, is rasterised, so a digit photograph is PNG and a
two-colour decision mask stays vector.

A side-by-side review sheet for a figure you are unsure about:

```bash
python scripts/import-b/contact_sheet.py assets/module-media/x.svg /tmp/opencode/figs/b-x.svg -o /tmp/opencode/x.png
```

It rasterises both at the same width and pastes them onto white, because the
question is whether the new figure belongs on A's page. Label the left panel
with `--left-label`: "site B original" when the reference is B's own figure,
"hand-made, in the module today" when it is one of A's. The two are not
interchangeable, and a sheet that mislabels its reference says the runner
changed something A drew by hand. Save the sheet under
`/home/jackwu/Desktop/hp/merge/shots/<lane>/` and ask L0 to look at it; a manager
does not sign off a figure's look alone.

### What the runner guarantees

The rules below are what makes two figures from two different modules look like
one set on the same page. They come from `b_theme.py`, not from a convention
anyone has to remember.

**One colour per role, in every figure.**

| role | colour | where it comes from |
|---|---|---|
| the main objects and the curves being explained | blue `#2563eb` | `P.emph()`, `P.line(rank=0)`, the heaviest marks |
| the one highlighted result or point | terracotta `#b84e30` | `P.signal`; twice in one figure is wrong |
| neutral structure: arrows, box outlines, a second series, body text | dark grey `#4b5563` | `series[1]`, `arrow()`, `box()` |
| guides and grid | light grey `#e5e7eb` gridlines, `#d8d8dd` dashes, `#9ca3af` the frame | `grid()`, `guide()`, `clean()` |

**One text scale and one stroke ladder, in every figure.** Type is fixed at
12/10.5/9.5/8.5 pt and strokes at 0.8/1.2/1.7/2.4 pt. Every figure is drawn at one
pt-to-CSS-px scale of 1.427, so a 10.5 pt label is the same size on the page in a
4.4 in square as in a 7.2 in banner; a narrower figure is narrower on the page, it
is not stretched. The width written into the file comes from what the figure
actually drew, not its nominal size: `bbox_inches="tight"` crops the canvas to the
content, and sizing from the nominal value would scale two figures of the same
nominal size differently and the text with them. Text is Inter, matching A's lesson
body, and maths uses Computer Modern so symbols match the page's KaTeX. No figure is
wider than A's column, and no `-dark` variant is ever written: A has one light theme.

The smallest text the theme draws is 8.5 pt, which is **12.1 CSS px** at that scale.
L0's floor is 11 px, so the theme clears it, and the runner reports the smallest
label per figure so a manager does not have to measure. A figure wider than the
column is scaled to 760 px to fit, and its text scales with it; any figure that then
falls under 11 px is listed in `FIGURE-INVENTORY.md` for its module manager to fix.
As of 2026-10-10 one figure hit the cap and none fell below the floor.

The runner checks what it can: the file exists, is no wider than the column, no
ink touches an edge, no colour of B's dark palette or teal signal survived, and
the figure is neither blank nor a solid slab. It exits non-zero if any of those
fail, so a module manager can put it in a script.

### What the runner cannot do, and you must check by hand

The swap is mechanical. The judgement is not.

- **Role semantics.** B's `series[0]` is its darkest step, so generators use it
  both for "the curve being explained" and for "the input layer". The mapping
  turns it blue everywhere. In a network schematic that paints the input layer
  blue where A's own MLP figure paints it grey. Look at the figure and decide
  whether the blue belongs there; if not, re-colour that one call in a copy of
  the generator, and say so in the module's hand-off notes.
- **What the figure is about.** `signal` is one mark per figure by construction,
  and the mapping keeps that rule. A figure whose lesson text points at two
  things will have one terracotta mark and one that is not.
- **Overlapping labels.** The theme does not detect them and neither does the
  runner. If B's original has two labels that collide, the runner reproduces the
  collision faithfully, which makes it B's problem rather than the theme's. Only
  an eye finds these: of the twelve figures L0 looked at, one had an overlap and
  B's original has it too. Ask L0 to look at each figure before its module ships.
- **Data correctness.** The runner re-draws what the script computes. If the
  script's numbers were tuned for a different dataset size or seed, they come out
  as they were. Rerun any number the lesson quotes.
- **Nothing it draws is new.** The runner never invents a figure, never splits one
  into two, and never decides a figure should be dropped. B's lesson asked for it;
  the module manager decides whether A's version still needs it.

### How to fix a script the runner cannot carry

Per-script patch files live in `scripts/import-b/figure-patches/`. A patch is a
small module the runner applies before drawing: it overrides a colour, a size, or
one call site in the generator. Keep them small, name them after the script, and
say in the patch why the generator could not run unchanged. When a patch is more
than a few lines, the figure is being redrawn and should be a new generator in
`scripts/figures/` instead.

As of 2026-10-10, all 30 of B's generators run unchanged and no patch file exists.
The full result, one row per script with its outputs and the lessons that use
them, is in `/home/jackwu/Desktop/hp/merge/FIGURE-INVENTORY.md`, along with the
twelve figures L0 sampled and the smallest label size per figure.

### The MLP module is not redrawn by the runner

`assets/module-media/mlp-*` was drawn by hand for this module and stays. The
runner reproduces B's `gen_mlp_figures.py` faithfully, but the hand-made figures
carry this module's own worked numbers (784-16-16-10, the weights 3, -2, 1, the
bias -1) and the runner draws B's generic ones. The runner is for the other 93
modules; the comparison sheets under `shots/toolfig/` are the proof it matches
A's look.
