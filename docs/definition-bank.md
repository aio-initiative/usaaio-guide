# Definition bank

Edit `content/definitions.js` to add or revise a shared entry. Each entry contains
a stable ID, title, aliases to recognize in prose, short hover definition,
full explanation, destination module ID, and an optional proof.

Use `String.raw` for paragraphs or equations containing TeX. Full explanations
use the same paragraph and display-math blocks as lessons. Keep definitions
self-contained and give theorem assumptions explicitly. Use specific aliases:
ordinary words can have unrelated meanings in other modules.

The lesson and theory readers underline the first occurrence of each entry in
each section. They leave links, headings, math, and code alone. Hover or focus
shows the short definition; click or Enter opens a native modal dialog. Escape
closes it and returns focus to the term. The Definition bank button opens a
searchable list. The full explanation links to a matching section in its module,
or to the first topic when no matching phrase is found.

Run `node scripts/validate-definitions.mjs` and the usual content build and
validation after edits. Browser-check a short definition, a proof with math,
search, keyboard dismissal, and the destination link. When publishing, change
the asset versions in the module and theory HTML so cached readers get the update.
