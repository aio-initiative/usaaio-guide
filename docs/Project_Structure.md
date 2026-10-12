# Project Structure

The repo is organized like a lightweight version of guide sites such as USACO Guide and USAMO Guide.

```text
content/
  ordering.js                 stage -> chapter -> module ordering
  catalog.js                  module metadata consumed by the site
  lessons.js                  structured lesson sections for the browser reader
  problems.js                 local problem bank and contest presets
  1_Round_1/                  MDX-style lessons
  2_Round_2/                  MDX-style lessons
  3_Camp/                     MDX-style lessons
  4_Advanced/                 MDX-style lessons
  problemBank/                structured practice metadata
dashboard/
roadmap/
round1/
round2/
camp/
advanced/
module/
problems/
contests/
resources/
search/
settings/
login/
contact/                      route pages
docs/
  AI_Topic_Template.md        contributor template
public/
  assets/                     static media
scripts/
  validate-content.mjs        no-dependency content validation
src/
  app.js                      page orchestration
  components/renderers.js     HTML render helpers
  data/site-data.js           tracks, resources, dates, checklist
  lib/progress.js             local profile, progress, settings, export/import
  styles/main.css             site styles
```

This project intentionally avoids a dependency install for now. It can open directly through `index.html`, while still keeping a migration path to Next.js, Gatsby, Astro, or another MDX compiler later.

The route folders follow the same broad separation used by guide projects: pages provide the surface, `content/ordering.js` provides stage/chapter/module order, `content/catalog.js` provides module metadata, and `src/components/renderers.js` turns shared data into reusable cards, tables, and readers.

## Supported Static-App Functions

- Round 1, Round 2, Camp, and Advanced stage pages through `content/ordering.js`.
- Consolidated top navigation rendered from `src/app.js`.
- Module reader with table of contents, previous/next navigation, MDX draft links, bookmarks, and status selectors.
- Dashboard with module, problem, bookmark, and recent contest-run summaries.
- Browser-local sign-up/sign-in so multiple students on one browser can keep separate progress profiles.
- Searchable module catalog and guide map.
- Global search across module metadata, lesson sections, practice problems, and resources.
- Searchable/filterable problem bank with local problem statuses and answer reveals.
- Timed contest arena with local attempt history.
- Scratchpad saved in browser storage.
- Settings for theme, preferred language mode, compact mode, hide-hero mode, progress export/import, and local reset.
- Contributor docs and validation scripts.
