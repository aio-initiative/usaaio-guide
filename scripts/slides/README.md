# Module slide decks

One 16:9 deck per module, in the CP Initiative template, in two formats:
PDF for handing out, `.pptx` for Google Slides.

```bash
npm run slides:setup                 # once — builds .venv, installs deps
npm run build:slides                 # 55 PDFs  -> slides/
npm run build:slides:pptx            # 55 .pptx -> slides/ + manifest + .gs
npm run slides                       # browse the decks locally
sh scripts/slides/run.sh scripts/slides/build_slides.py clustering   # just one
```

## Just looking at them

`npm run slides` regenerates `slides/index.html`, serves the repo and opens
a browser: deck list on the left, the PDF in the pane on the right, filter
box, arrow keys to move between decks. Nothing to upload and no Google
account involved — this is the fast path when you only want to read them.

It serves rather than opening `file://` because Chrome will not load a local
PDF into an iframe from a `file://` page.

The builders need pymupdf, python-pptx and matplotlib, which are rarely
installed system-wide, so `run.sh` routes every invocation through the
repo's `.venv` and prints the setup command if the packages are missing.
Calling `python3 scripts/slides/...` directly skips that and will fail
with `ModuleNotFoundError`.

`slides/` is gitignored: the decks are build output and regenerate from
`content/lessons.js`, the same file the site reader uses, so a deck cannot
drift from the module it is made of.

## Getting them into Google Slides

Google Slides imports `.pptx`, not PDF, which is why both formats exist.
The `.pptx` decks are built from native shapes and text boxes — every
heading, paragraph and table stays editable after import, and the fonts
(Roboto, Roboto Mono) are ones Google Slides ships with, so nothing gets
substituted.

`npm run build:slides:pptx` also regenerates `slides/manifest.csv` and
`scripts/slides/import_to_drive.gs`, so those never go stale against the
decks.

Then: upload all 55 `.pptx` to one Drive folder, paste
`import_to_drive.gs` into a new script.google.com project, set
`FOLDER_ID`, and run `importDecks`. It converts each deck to Google
Slides and drops a spreadsheet in the same folder with stage, code,
module, slide count, the Slides link and the module page URL. Re-running
is safe — anything already converted is reused, so a run cut short by
the 6-minute execution limit just needs running again.

`slides/manifest.csv` is the same table with the link column blank, for
filling in by hand if you would rather not run the script.

Only display equations are images; inline maths is flattened to Unicode
so the sentence around it stays editable. The PDF decks keep inline maths
as rendered images and are the better-looking of the two.

## Requirements

`npm run slides:setup` handles it — `python3 -m venv .venv` plus
`scripts/slides/requirements.txt` (matplotlib, numpy, pymupdf, python-pptx).

## Where the template came from

Every constant in `template.py` was measured out of the reference deck
(`2. Sorting with Custom Comparators.pdf`) with PyMuPDF rather than estimated:

| | |
| --- | --- |
| Page | 720x405pt (10 x 5.625in, 16:9) |
| Purple | `#5b21b6` — 12.9pt top bar, 554.7x162.2 cover band |
| Blue dots | `#2563eb`, 10pt, 18pt pitch, stepped 4-3-2-1 from (646.5, 23.1) |
| Cover panel | `#dbeafe`, 447.1x405 |
| Cover title | Roboto Bold 33pt white at x=61 |
| Heading | Roboto Bold 28pt `#18181b` at (34.1, 53.7) |
| Body | Roboto Regular 18pt `#374151`, 24.7pt line pitch |
| Code | Roboto Mono 16pt on `#f9fafb`; keywords `#3f51b5`, types `#9c27b0` |
| Logo | 47pt on the cover, 27.4pt on content slides |

The Roboto and Roboto Mono TTFs live in `assets/slide-fonts/` so the type
matches instead of being substituted.

## What ends up on a slide

Modules are long-form prose and a deck is not, so `condense()` keeps the
framing paragraphs of each topic plus one list, one equation, one figure and
one code listing, capped at `SLIDE_BUDGET` slides. Without it a module like
reinforcement-learning runs to 277 slides.

Maths is rendered through matplotlib's mathtext and placed as an image, inline
and display. An earlier attempt converted LaTeX to Unicode and produced
nonsense — `\left` collided with the `\le` rule and rendered as "≤ft" — so that
path survives only as a fallback when mathtext rejects a string.
