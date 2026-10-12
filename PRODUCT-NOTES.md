# PRODUCT-NOTES (DRAFT)

**Status: DRAFT. Written by the product lane manager for the owner and a lawyer. Nothing here is
legal advice; it is plain-English wording plus a list of facts a lawyer will need. Every fact below
was verified against the site's code and files on 2026-10-10.**

The site has no visible "unofficial" statement, no accuracy disclaimer, no privacy policy and no
terms of use. This file drafts the first two and lists what the last two must cover. It is a
starting point for the owner's lawyer, not a substitute for one.

## 1. The not-affiliated wording (DRAFT)

The site currently says it is unofficial only in places a reader never sees: the meta description
and the social previews (`index.html`, `og:description` and `twitter:description`), and the og-card
artwork. No page the reader lands on says it. The home page's structured data goes further and
asserts the opposite, listing `https://www.usaaio.org/` under `sameAs` for
`https://www.aioinitiative.com/#org`, which in structured data states that the two are the same
organisation. That entry should be removed or changed to `about`/`mentions`.

Two notes that shape the wording. The owner runs two live sites under two names, AIO Initiative and
USAAIO Guide, and this merge has to settle on one name and one footer, so the disclaimer should be
written once and used on both. And the guide's name is built on another organisation's mark, so a
disclaimer alone may not be enough; that is the lawyer's call.

Draft wording, one line for the footer of every page and a short paragraph for the About page:

> **Footer, every page:** An unofficial, independent study guide. Not affiliated with, endorsed by,
> or sponsored by the USA Artificial Intelligence Olympiad or USAAIO. All trademarks are the
> property of their owners. USAAIO does not review, approve or check this guide.

> **About page, fuller version:** This is an unofficial, independently written study guide. We are
> not affiliated with, endorsed by, sponsored by, or in any way officially connected to the USA
> Artificial Intelligence Olympiad (USAAIO), the International Olympiad in Artificial Intelligence,
> or the International Artificial Intelligence Olympiad. The names of those organisations and their
> contests are their trademarks; we use them here only to identify what our lessons prepare students
> for. Nothing on this site has been reviewed or approved by any of those organisations. For the
> official contest, its rules, its dates and its past papers, go to
> [https://www.usaaio.org/](https://www.usaaio.org/).

The About page is also where the team's USAAIO and Team-USA badges and bios lean on USAAIO
credentials, so the fuller paragraph belongs there.

## 2. The accuracy disclaimer (DRAFT)

The site grades problems, teaches, and states competition facts. Two of those statements were
checked against the official site and found unsupported, so the disclaimer should be real rather
than boilerplate, and the two statements should be fixed regardless.

Draft wording for the footer or the About page:

> **Accuracy.** Lessons, problems and answers here were written by the authors of this guide and
> checked by them, not by any official body. Contest rules, dates and formats change every year;
> what we say about them was correct to the best of our knowledge when written, and we do not
> promise it still is. Before you rely on any of it for a registration, a deadline or a submission,
> check the official contest site. If you find an error, tell us and we will correct it.

And the accuracy policy that wording implies, which the site should state somewhere a reader can
find: what a report should contain, where it goes, and that corrections are credited if the reporter
wants. Right now error reporting is a bare mailto link with no instructions.

## 3. What a privacy policy for this site must cover (DRAFT list, for a lawyer)

This is the important part: a policy written from a template will be wrong, because the list below
is what this site actually stores. Each item was verified in the code.

**What the site stores in the visitor's own browser (localStorage, no account needed):**

- `usaaio-guide-state-v2` - the whole progress state: module and problem completion status, the
  study plan, the free-text scratchpad, and the activity log of every problem solved with its date.
- `usaaio-editor-draft-v2:<notebook-id>` - one entry per notebook, holding the code the visitor
  typed into the editor and has not yet run or submitted.
- `usaaio-editor-layout` - the preferred column layout of the editor (split or single).
- `usaaio-reader-nav-collapsed` - whether the reader sidebar is open.
- `usaaio-mock-session-v1` - the state of a practice mock session, which by design keeps a mock in
  progress away from the worked solution.

**What the site stores on a server, only when the visitor signs in:**

- A Supabase record in the table `user_progress`, written on essentially every progress change,
  holding the same state object as `usaaio-guide-state-v2` above, keyed by the user id, with a
  timestamp. That includes the scratchpad's free text and the full activity log.
- `usaaio-guide-remote-user-v1` - the marker that a session exists locally.
- The email address and the display name the OAuth provider returns, from Google or GitHub.

**What third parties see, on every page load, with no choice or notice:**

- The visitor's IP address and the page URL, sent to Google Fonts (the CSS for Inter, Outfit and
  JetBrains Mono) and to jsdelivr (KaTeX, Pyodide, the Supabase client). Google's and jsdelivr's
  policies apply to what they receive.
- The sign-in flow hands the visitor to Google or GitHub, which then return the email and the name.

**What a policy therefore has to say, at minimum:**

1. Who the data controller is and how to contact them, including a named channel for data requests.
   The site currently offers a shared mailbox and a Discord link, which is not a data-request
   channel, and no contact address for privacy matters at all.
2. Each of the localStorage keys above, in plain English, and that none of it leaves the browser
   unless the visitor signs in.
3. That signing in syncs the whole progress state, including the scratchpad's free text and the
   activity log, to a server, and how often.
4. The email and name collected via OAuth, what they are used for, and that the site does not
   collect a password.
5. The three third parties above, what each receives, and links to their policies.
6. How long the data is kept and how a user exports or deletes it. Today there is no way for a
   signed-in user to see, export or delete what is stored about them; that is a platform fix, not
   just a policy sentence.
7. Children. The audience is middle and high school students. COPPA requires verifiable parental
   consent before collecting personal information from a child under 13, and Google and GitHub
   accounts both require 13 or over, so an under-13 visitor cannot lawfully sign in at all. The
   policy has to say that, and the site has to enforce it; today there is no age gate or
   parental-consent step anywhere in the sign-up path.
8. That progress protection depends on Supabase's row-level security on `user_progress`, which no
   outside party has audited. An access-control review should happen before launch.
9. Cookies and tracking. The site sets no cookies of its own, but the third-party loads above carry
   their own, so a notice is still required.

**An option the owner should know about.** All of the learning features work signed out; progress,
drafts, the scratchpad and the mock sessions are stored locally already. Dropping accounts removes
items 3, 4, 6 and 7 almost entirely and leaves the IP-address disclosure to disclose. That is a
product decision, not a legal one.

## 4. Terms of use (DRAFT list, for a lawyer)

There are no terms, while the footer asserts "All rights reserved." and the site plans to sell
access. The terms will have to settle, at minimum: what licence the content is under (unsettled;
see PRODUCT-AUDIT.md part 1.1), what licence the code is under (also unsettled; part 1.2), what an
account is and what it does not guarantee, that the lessons and graders are unofficial, the
accuracy position from section 2 above, and what happens to a paid subscription if content is
replaced because a licence or a copyright holder requires it. The "All rights reserved." line should
stay only if the owner is confident it is true; as written it conflicts with both the CC-licensed
content the site merges and with any open-licence declaration.

---

Product lane, branch `merge/product`. Questions to the owner are in PRODUCT-AUDIT.md.
