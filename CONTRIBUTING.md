# Contributing

Thanks for helping improve AIO Initiative.

## Add a Module

1. Copy the ten-section structure documented in `content/README.md`.
2. Add a `.md` file under the correct `content/` stage folder, named `<stage>.<n>-<CODE>-<slug>.md`.
3. Add the module id to the right chapter in the `CHAPTERS` table in `scripts/build-content.mjs`.
4. Add practice metadata under `content/problemBank/` when useful.
5. Run `npm run build:content` — this regenerates `content/ordering.js`, `catalog.js`, and `lessons.js`, which are never edited by hand.
6. Run `npm run validate`.

## Content Principles

- Write for middle and high school students.
- Connect math, implementation, and contest explanation.
- Prefer small examples students can verify by hand.
- Link to official USAAIO pages for rules, dates, syllabus, and past problems.
- Keep unofficial guidance clearly separate from official competition facts.
