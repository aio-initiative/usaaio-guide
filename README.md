# AIO Initiative

A content-first USA AI Olympiad preparation guide for middle and high school students. 

The site is a lightweight guide repo: route pages live at the top level, content lives in `content/`, UI code lives in `src/`, static media lives in `public/`, contributor docs live in `docs/`, and validation scripts live in `scripts/`.

## Quick Start  

Open `index.html` directly in a browser, or use the local server for the cleanest route testing.
 
For local-server testing:

```bash
npm run dev
```

Then visit `http://127.0.0.1:4173`.

No dependency install is required for the current static version.

## Route Structure

- `index.html` - home and guide overview.
- `dashboard/` - progress, bookmarks, scratchpad, and next steps.
- `roadmap/` - study tracks and full stage/chapter map.
- `round1/`, `round2/`, `camp/`, `advanced/` - competition-stage catalogs with search, filters, status selectors, and module cards.
- `module/?id=<module-id>` - module reader with table of contents, related practice, bookmark, and previous/next controls.
- `problems/` - searchable/filterable problem bank of clickable auto-graded notebook problems with status tracking.
- `contests/` - timed local practice sets with attempt history.
- `resources/` - official links, resource shelf, season planner, and FAQ.
- `search/` - global search across modules, lesson sections, problems, and resources.
- `contact/` - feedback, contribution, and official-contact links.
- `settings/` and `login/` - compatibility redirect stubs; the Login control is available from every page.

## Scripts

```bash
npm run validate
npm run check
```

Both commands run `scripts/validate-content.mjs`, which checks that:

- Every route page exists.
- Every catalog module has a matching MDX file.
- Every ordered module exists in the catalog.
- Every catalog module appears in `content/ordering.js`.
- Module frontmatter IDs match catalog IDs.
- Problem-bank files reference known modules.
- Structured lessons, local problems, and contest presets reference known modules/problems.

## Repository Layout

```text
content/
  ordering.js
  catalog.js
  lessons.js
  problems.js
  1_Round_1/
  2_Round_2/
  3_Camp/
  4_Advanced/
  problemBank/
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
contact/
docs/
public/
scripts/
src/
  app.js
  components/
  data/
  lib/
  styles/
```

See `docs/Project_Structure.md` for more detail.

## Content Workflow 

1. Add or edit the module's `.md` file in the correct `content/` stage folder.
2. Add its id to a chapter in the `CHAPTERS` table in `scripts/build-content.mjs`.
3. Add a `.problems.json` file under `content/problemBank/` if needed.
4. Run `npm run build:content` to regenerate `ordering.js`, `catalog.js`, and `lessons.js`.
5. Run `npm run validate`.

`content/README.md` documents the ten-section shape every module follows; `docs/AI_Topic_Template.md` is a fuller worked template.

## Implemented Guide Functions

- Multi-page route structure inspired by USACO Guide and USAMO Guide.
- Consolidated USACO Guide-style top bar with Guide, Problems, Resources, Contact Us, and Search.
- Dashboard with browser-local progress tracking that syncs to a signed-in account.
- USACO Guide-style stage/chapter guide map using Round 1, Round 2, Camp, and Advanced sections.
- Module reader with table of contents, previous/next controls, bookmarks, MDX draft links, and statuses.
- Searchable module catalog.
- Global search page for modules, lesson sections, practice problems, and resources.
- Searchable/filterable problem bank of clickable notebook problem cards with status tracking.
- Timed local contest arena with recent attempts.
- Scratchpad notes saved in browser storage.
- Contributor templates and no-dependency content validation.
- Google and GitHub login through Supabase, with account-backed module, problem, planner, and scratchpad state.
- Public access to all guide modules and problems, with signed-out progress stored locally in the browser.

## Authentication and account linking

The guide does not require an account. Module and problem status, the planner, and scratchpad are stored in local storage while signed out. The site uses Supabase Auth only for persistent sessions and cross-device sync. Enable Google and GitHub under the Supabase project's Auth providers, and add the deployed site URL to the OAuth redirect allow list.

Supabase automatically links Google and GitHub identities when their email address is the same and verified, so both sign-ins resolve to the same user and the same `user_progress` record. The signed-in Login panel shows which providers are connected. To connect providers with different emails, enable Supabase's **Allow manual linking** setting and use the Connect button in that panel.

## Source Notes 

This guide is unofficial and uses original explanatory content. Competition facts and links were checked on August 17, 2026 from:

- <https://www.usaaio.org/>. 
- <https://www.usaaio.org/2027-usa-na-aio>
- <https://www.usaaio.org/syllabus>
- <https://www.usaaio.org/past-problems>
- <https://www.usaaio.org/contest-rules>

The structure was modeled after public patterns in:

- <https://github.com/cpinitiative/usaco-guide>
- <https://github.com/usamoguide/usamo-guide>

The home hero uses a lightweight animated neural-network canvas drawn by `src/app.js` (no image asset).
