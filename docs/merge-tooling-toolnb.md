# Merge tooling: notebooks

The notebook lane's two tools, documented here. The shared `docs/merge-tooling.md` links to this file.

## Notebooks: import from site B, then gate

Site B keeps every notebook as a pair: `notebooks/<module>/<name>.ipynb`, the
exercise the student works on, and `<name>-solution.ipynb` beside it, the same
notebook with each task cell replaced by its worked solution. Site A keeps one
file: the exercise cells, untouched, and the whole worked solution appended at
the end behind a markdown cell that starts `## Solution`, so the editor can fold
it and `?embed=1` can strip it. Two scripts do the conversion and then prove the
result runs.

### `scripts/import-b/notebook.py`

```
scripts/import-b/notebook.py <B task notebook> [--solution <B solution notebook>]
    --module <A module id> --out <file>
    [--colab "<sentence>"] [--difficulty <word>] [--minutes <n>]
    [--tags a,b,c] [--description "<text>"] [--title "<text>"]
```

Reads B's pair, writes one A notebook. What it does:

- B's exercise cells come first, verbatim, with their outputs stripped.
- The solution notebook is split into sections at its H2 headings the same way
  B's own site code splits them. `## Part 3 (coding task): Title` becomes
  `### Part 3: Title`; any other section (`## Setup`) becomes `### Title`. Only
  the solution cells that actually differ from the exercise are carried across,
  so a cell B repeats word for word is not duplicated. A section whose solution
  is identical to its exercise gets no heading, because there is nothing new to
  show.
- Every cell from `## Solution` on, headings included, is flagged
  `{"usaaio_solution": true}`, which is what `scripts/validate-content.mjs`
  requires and what the editor folds.
- `metadata.usaaio` is filled from B's own `metadata.usaaio`: id, title,
  difficulty, minutes, tags. `guided` is B's own judgement, mirrored from its
  `isCodingProblem` rule: a notebook is a problem when its metadata holds
  anything beyond `runSeconds` and `colab`, and a guided lesson companion
  otherwise.
- B's `runSeconds` and `colab` say how a notebook runs, not what it teaches, so
  they are never copied. `colab` lands in the A notebook only when it is passed
  with `--colab`, after someone has run the notebook against the page's own data
  subset and written a sentence naming the parts that still miss their check.
- The description is the exercise's own task sentence: B's exercises state the
  task as "Here you ...", so that sentence with "Here you" dropped is the
  description. When an exercise does not use that phrasing the script falls back
  to the exercise's opening paragraph, which is what B itself shows as the
  notebook's summary, and warns.

It changes no sentence and no line of code. Anything it cannot map is printed as
a `WARNING:` on stdout and left out; it never guesses, and the exit code stays 0
so a conversion log can be scanned for warnings rather than for failure. When B
is silent on a field the A notebook carries a default (difficulty Medium,
minutes 30, tags empty) and a warning naming the flag that overrides it. That
leaves a person to decide what a lesson's companion should say about itself,
which is not something the pair of notebooks can answer.

### `scripts/check_notebooks.py`

```
scripts/check_notebooks.py [--timeout SECONDS] [--python INTERPRETER] <notebook>...
```

The gate. For each A-convention notebook it runs two executions in its own
throwaway folder, each in a worker subprocess of the conda Python
(`~/miniforge3/envs/Main/bin/python`), so one notebook that crashes a library
cannot kill the run:

1. **The solved version.** Each task cell is replaced by its part's solution
   cells, the way the editor's "Run all" sees a notebook the student has
   finished, and the solution section is dropped. The part a solution cell
   belongs to is read from the `### Part N` headings in the solution section;
   site A's older notebooks, which keep the whole solution as a fenced
   ```python block inside the `## Solution` markdown cell, are read too.
2. **The blank template.** The exercise as the student gets it.

A self-check cell is one that prints the word "passed": site B's convention is
the cell's last statement (`print("Part 3 passed")`), site A's older notebooks
print it from inside an if/else (`print("PASSED, nice work!" if _ok else "NOT
PASSED ...")`). The cell passes when the last verdict line it printed is not a
"NOT PASSED" line. The solved run must pass every self-check; the blank run must
pass none of them, and a check cell in the blank that dies on the unimplemented
task cell above it counts as failing, which is what the gate wants to see.

MNIST comes from the `MNIST_ROOT` environment variable: torchvision is pointed
at it with `download=False`, whatever the notebook asks for, so nothing is ever
downloaded and nothing is written into the repo. A notebook whose package, GPU
or data is missing says so instead:

```
PASS     mlp-train-pytorch (solved: 3/3 checks ok in 1.6s; blank: 3 checks fail as they should)
FAIL     mlp-mnist (solved check cell 9 printed 'accuracy 0.8905 is below the 0.93 bar')
TIMEOUT  tiny-diffusion-pytorch (solved run, after 600s)
NEEDS    bert-answer-spans-numpy package: No module named 'transformers'
```

The exit code is 1 when any notebook FAILs. TIMEOUT and NEEDS are reported, not
counted as failures: they say something about the machine, not about the
notebook. Run it through `heavy.sh`, one notebook at a time, so notebook
execution never overlaps another lane's heavy job.
