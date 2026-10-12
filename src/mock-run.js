/*
 * Mock Arena, the sitting itself.
 *
 * One frame holds every problem: the header card the problem pages use
 * (problem n of N, type, difficulty), a strip of problem tabs, then one panel
 * with a sticky toolbar, the problem, and Previous / Next at its foot.
 *
 * The problem itself is drawn by the problem bank's own code:
 *   - a coding problem is the notebook editor (editor/?nb=...&embed=1) in a
 *     frame that grows with its content. Statement, Code and Split view are
 *     three layouts of that one editor, switched in place, so the running
 *     Python and the code typed so far survive a switch.
 *   - a Text or MCQ problem is drawn with src/theory.js (AIO_THEORY_VIEW):
 *     the same overview, setup and part cards as theory/?id=...
 *
 * Solutions are deliberately unreachable here:
 *   - theory problems render prompts only, never the `answer` blocks. An
 *     option is drawn from its text alone, so no verdict reaches the page, and
 *     a part with a number for its answer gets a box to write it in, never
 *     a checker.
 *   - the editor drops every cell from the first one carrying
 *     `usaaio_solution` (or a "## Solution" heading), and so does its download.
 * Links out to the full problem pages appear only on the summary, after the
 * mock has ended.
 */
(function () {
  if (document.body.dataset.page !== "mock-run") return;

  const root = document.body.dataset.root || "";
  const mock = window.AIO_MOCK;
  const storage = window.AIO_PROGRESS;
  const theoryView = window.AIO_THEORY_VIEW;
  const els = { shell: document.querySelector("#mockRun") };
  if (!els.shell || !mock) return;

  const problemData = window.AIO_PROBLEMS || { problems: [] };
  const theoryData = window.AIO_THEORY || { problems: [] };
  const problemLookup = new Map(problemData.problems.map((p) => [p.id, p]));
  const theoryLookup = new Map(theoryData.problems.map((p) => [p.id, p]));
  const knownIds = new Set(problemLookup.keys());

  let session = mock.load(knownIds);
  let ticker = null;

  // The editor's own Split view switch, shared with the coding problem page,
  // and the narrowest window it is offered at there.
  const LAYOUT_KEY = "usaaio-editor-layout";
  const SPLIT_MIN_WIDTH = 1024;
  // What a coding problem shows: "text" (Statement), "single" (Code) or
  // "split". Kept in memory. A problem opens on its statement, or in Split
  // view when that is on, since Split view shows the statement too.
  let view = "text";

  /* --------------------------------- helpers ------------------------------- */

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  const problemType = (p) => (p.tags.includes("Theory") ? "Theory" : "Coding");

  // The three problem kinds the site shows: a notebook is Code, one
  // multiple-choice question is MCQ, every other written problem is Text.
  // `page` is the name the problem page gives it in its type pill.
  function problemKind(p) {
    if (problemType(p) === "Coding") return { id: "code", label: "Code", page: "Code" };
    return p.tags.includes("Multiple choice")
      ? { id: "mcq", label: "MCQ", page: "Multiple choice" }
      : { id: "text", label: "Text", page: "Text answer" };
  }

  // Same pill as the problems table: one small mark per kind, then its label.
  const TYPE_ICON = {
    code: '<path d="m6 5-3.5 3L6 11M10 5l3.5 3L10 11" />',
    mcq: '<circle cx="3.6" cy="4.5" r="1.6" /><path d="M8 4.5h5.5M8 11.5h5.5" /><path d="m2.2 11.6 1.1 1.1 2-2.4" />',
    text: '<path d="M2.5 4h11M2.5 8h11M2.5 12h6.5" />',
  };
  const typePill = (p) => {
    const kind = problemKind(p);
    return `<span class="mock-type mock-type--${kind.id}"><svg class="mock-type-icon" viewBox="0 0 16 16" aria-hidden="true">${TYPE_ICON[kind.id]}</svg>${kind.label}</span>`;
  };
  const difficultyChip = (p) =>
    `<span class="problem-chip difficulty-${escapeHtml(String(p.difficulty).toLowerCase().replace(/\s+/g, "-"))}">${escapeHtml(p.difficulty)}</span>`;
  const problems = () => session.problemIds.map((id) => problemLookup.get(id)).filter(Boolean);
  const activeProblem = () => problems()[session.activeIndex];

  function problemHref(problem) {
    if (!problem.url) return "";
    return problem.url.startsWith("/") ? `${root}${problem.url.slice(1)}` : problem.url;
  }

  // The notebook filename lives in the problem's "?nb=" param and is NOT the
  // moduleId. Many coding problems disagree (the "orientation" notebook, for
  // one, belongs to the data-handling-visualization module).
  function notebookId(problem) {
    const query = String(problem.url || "").split("?")[1] || "";
    const nb = new URLSearchParams(query).get("nb");
    return (nb || problem.moduleId || "").replace(/[^a-z0-9-]/g, "");
  }

  function statuses() {
    return storage ? storage.loadState().problemStatus || {} : {};
  }

  function completedCount() {
    const map = statuses();
    return session.problemIds.filter((id) => map[id] === "complete").length;
  }

  /* ------------------------------- split view ------------------------------ */

  const splitAllowed = () => window.innerWidth >= SPLIT_MIN_WIDTH;

  function splitStored() {
    try {
      return window.localStorage.getItem(LAYOUT_KEY) === "split";
    } catch (error) {
      return false;
    }
  }

  function storeSplit(on) {
    try {
      window.localStorage.setItem(LAYOUT_KEY, on ? "split" : "single");
    } catch (error) {
      /* private mode: the choice lasts for this page only */
    }
  }

  const openingView = () => (splitStored() && splitAllowed() ? "split" : "text");

  /* ------------------------------- full screen ----------------------------- */

  // The expanded layout is a class on the shell rather than a native-fullscreen
  // element, because render() replaces the shell's innerHTML. Anything inside
  // it would be destroyed (and drop out of fullscreen) on the next redraw. The
  // class survives, and we ask the browser to fullscreen <html>, which is never
  // re-rendered either.
  const fsActive = () => els.shell.classList.contains("mock-fs");

  function setFullscreen(on) {
    els.shell.classList.toggle("mock-fs", on);
    document.body.classList.toggle("mock-fs-active", on);
    if (on) {
      if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {
          /* denied, and the in-page expanded layout still applies */
        });
      }
    } else if (document.fullscreenElement && document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
    syncFullscreenButton();
  }

  // The button is patched rather than redrawn: a redraw would reload the
  // notebook editor and lose the running session.
  function syncFullscreenButton() {
    const button = els.shell.querySelector("#mockFullscreen");
    if (!button) return;
    const on = fsActive();
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.innerHTML = `${on ? ICONS.shrink : ICONS.expand}<span>${on ? "Exit full screen" : "Full screen"}</span>`;
  }

  // Escape leaves native fullscreen without touching our class, so resync.
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && fsActive()) {
      els.shell.classList.remove("mock-fs");
      document.body.classList.remove("mock-fs-active");
      syncFullscreenButton();
    }
  });

  /* ------------------------------ kept answers ----------------------------- */

  // The option a student picks on a multiple-choice problem, and the number
  // they write for a part whose answer is a number, are kept for the sitting.
  // They survive switching problems and a reload, are never judged here, and
  // show on the summary next to the link to the full problem.
  const PICKS_KEY = "usaaio-mock-picks-v1";

  function loadKept() {
    try {
      const raw = JSON.parse(window.localStorage.getItem(PICKS_KEY) || "null");
      if (raw && raw.startedAt === session.startedAt) {
        return {
          picks: raw.picks && typeof raw.picks === "object" ? raw.picks : {},
          answers: raw.answers && typeof raw.answers === "object" ? raw.answers : {},
        };
      }
    } catch (error) {
      /* unreadable storage, start with nothing kept */
    }
    return { picks: {}, answers: {} };
  }

  const { picks, answers } = loadKept();

  function saveKept() {
    try {
      window.localStorage.setItem(PICKS_KEY, JSON.stringify({ startedAt: session.startedAt, picks, answers }));
    } catch (error) {
      /* private browsing, what was kept lasts until the page reloads */
    }
  }

  const letterOf = (index) => String.fromCharCode(65 + index);
  const choiceText = (choice) => (typeof choice === "string" ? choice : String((choice && choice.text) || ""));
  const partLabel = (part) => String(part.id).split(".").pop();

  // The first multiple-choice block of a problem, as { key, choices }.
  function choiceBlock(problem) {
    const data = theoryLookup.get(problem.id);
    if (!data) return null;
    for (const part of data.parts) {
      const block = (part.prompt || []).find((item) => item && Array.isArray(item.choices));
      if (block) return { key: `${problem.id}:${part.id}`, choices: block.choices };
    }
    return null;
  }

  /* ------------------------------ theory problems -------------------------- */

  // The problem page's option cards (.mcq-card), drawn from the option text
  // alone. A pick is marked, never judged. Options written as plain strings
  // already carry their own "A." labels and stay a list, as on the page.
  function renderPicks(choices, pickKey) {
    if (!choices.every((item) => item && typeof item === "object")) {
      return `<ul class="theory-choices">${choices.map((item) => `<li>${theoryView.inline(choiceText(item))}</li>`).join("")}</ul>`;
    }
    const picked = picks[pickKey];
    return `
      <section class="mcq-card mcq-bare mock-picks" aria-label="Choices">
        <ol class="mcq-options">
          ${choices
            .map((item, index) => {
              const on = picked === index;
              return `
            <li class="mcq-option" data-picked="${on}">
              <button class="mcq-choice" type="button" data-pick="${escapeHtml(pickKey)}" data-pick-index="${index}" aria-pressed="${on}">
                <span class="mcq-letter" aria-hidden="true">${letterOf(index)}</span>
                <span class="mcq-text"><span class="sr-only">${letterOf(index)}. </span>${theoryView.inline(choiceText(item))}</span>
              </button>
            </li>`;
            })
            .join("")}
        </ol>
      </section>
      <p class="mock-kept-note">Your pick is kept for this mock and shown on the summary. Answers stay hidden until the mock ends.</p>`;
  }

  // Where the problem page checks a number, the mock only keeps it: the same
  // box, with no Check button and no verdict.
  function renderAnswerBox(problem, part) {
    if (!theoryView.numericAnswer(part)) return "";
    const key = `${problem.id}:${part.id}`;
    const id = `answer-${escapeHtml(key.replace(/[^A-Za-z0-9_-]/g, "-"))}`;
    return `
      <form class="theory-check mock-answer" novalidate>
        <label class="theory-check-label" for="${id}">Your Answer</label>
        <div class="theory-check-row">
          <input class="theory-check-input" id="${id}" type="text" inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="Type a number" data-answer="${escapeHtml(key)}" value="${escapeHtml(answers[key] || "")}" />
        </div>
        <p class="theory-check-result">Kept for this mock and shown on the summary. Answers stay hidden until the mock ends.</p>
      </form>`;
  }

  // The structure of theory/?id=...: the overview and setup, then one card per
  // part. A lone multiple-choice question has no part heading.
  function renderTheoryStatement(problem) {
    const data = theoryLookup.get(problem.id);
    if (!data || !theoryView) return `<div class="mock-statement"><p>${escapeHtml(problem.prompt || "")}</p></div>`;
    const single = problemKind(problem).id === "mcq" && data.parts.length === 1;
    // A pick is kept under "<problem>:<part id>", the key the summary reads.
    const options = { hideAnswers: true, choices: (choices, partKey) => renderPicks(choices, `${problem.id}:${partKey}`) };
    const parts = data.parts
      .map(
        (part) => `
          <section class="theory-part${single ? " theory-part-single" : ""}">
            ${single ? "" : `<h3 class="theory-part-head">${theoryView.partHeading(part)}</h3>`}
            <div class="theory-part-body">${theoryView.renderBlocks(part.prompt, part.id, options)}</div>
            ${part.requirement && !single ? `<p class="theory-requirement">${theoryView.inline(part.requirement)}</p>` : ""}
            ${single ? "" : renderAnswerBox(problem, part)}
          </section>`,
      )
      .join("");
    const lead = (data.overview || []).length || data.setup;
    return `
      <div class="mock-statement">
        ${
          lead
            ? `<div class="theory-overview">
          ${theoryView.renderBlocks((data.overview || []).map((text) => ({ p: text })))}
          ${data.setup ? `<div class="theory-setup">${theoryView.renderBlocks(data.setup, "setup", options)}</div>` : ""}
        </div>`
            : ""
        }
        <div class="theory-parts">${parts}</div>
      </div>`;
  }

  /* ----------------------------- coding problems --------------------------- */

  const editorFrame = () => els.shell.querySelector(".mock-ide");
  const editorApi = () => {
    const frame = editorFrame();
    try {
      return (frame && frame.contentWindow && frame.contentWindow.AIO_EDITOR_EMBED) || null;
    } catch (error) {
      return null;
    }
  };

  // The frame is as tall as the notebook inside it, so the page scrolls the
  // way the coding problem page does, with no second scroll bar.
  function fitFrame(frame) {
    const doc = frame.contentDocument;
    const main = doc && doc.querySelector("main");
    if (!main) return;
    const height = Math.ceil(main.offsetTop + main.offsetHeight);
    if (height > 0) frame.style.height = `${height}px`;
  }

  function wireFrame(frame, problem) {
    frame.addEventListener("load", () => {
      const doc = frame.contentDocument;
      if (!doc) return;
      const RO = frame.contentWindow.ResizeObserver;
      if (RO) new RO(() => fitFrame(frame)).observe(doc.querySelector("main") || doc.body);
      fitFrame(frame);
      // The view may have changed while the notebook was loading.
      const apply = () => {
        const api = editorApi();
        if (api && editorFrame() === frame) api.setLayout(view);
      };
      if (editorApi()) apply();
      else doc.addEventListener("aio-editor-ready", apply, { once: true });
      // Passing every self-check marks the problem done in the editor. Mirror
      // it here without a redraw.
      doc.addEventListener("aio-problem-status-set", (event) => {
        const detail = event.detail || {};
        if (detail.id === problem.id) patchDone(problem.id, detail.status === "complete");
      });
    });
  }

  // Shows the view in the toolbar and hands it to the editor, in place.
  function setView(next) {
    view = next === "split" && !splitAllowed() ? "single" : next;
    els.shell.querySelectorAll("[data-view]").forEach((button) => {
      const on = button.dataset.view === view;
      button.classList.toggle("active", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const split = els.shell.querySelector("#mockSplit");
    if (split) {
      split.hidden = !splitAllowed();
      split.setAttribute("aria-checked", view === "split" ? "true" : "false");
    }
    const body = els.shell.querySelector("#mockBody");
    if (body) body.dataset.view = view;
    const api = editorApi();
    if (api) api.setLayout(view);
  }

  window.addEventListener("resize", () => {
    const active = activeProblem();
    if (!active || session.status === "done" || problemType(active) !== "Coding") return;
    if (view === "split" && !splitAllowed()) setView("single");
    else setView(view);
  });

  /* -------------------------------- download ------------------------------- */

  function saveFile(name, data) {
    const blob = new Blob([data], { type: "application/x-ipynb+json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  // A coding problem downloads from the editor, with the code typed so far and
  // without the solution. A theory problem downloads the problem page's own
  // blank worksheet, which never holds the answers.
  function downloadActive(problem) {
    if (problemType(problem) === "Coding") {
      const api = editorApi();
      if (api) api.download();
      return;
    }
    const data = theoryLookup.get(problem.id);
    if (!data || !theoryView) return;
    saveFile(`${problem.id}.ipynb`, JSON.stringify(theoryView.buildNotebook(data), null, 1));
  }

  /* -------------------------------- rendering ------------------------------ */

  const ICONS = {
    check: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M3.5 8.5l3 3 6-7"/></svg>',
    download:
      '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 2.25a.75.75 0 0 1 .75.75v5.19l1.72-1.72a.75.75 0 1 1 1.06 1.06l-3 3a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 0 1 1.06-1.06l1.72 1.72V3A.75.75 0 0 1 8 2.25zM3.5 12a.75.75 0 0 0 0 1.5h9a.75.75 0 0 0 0-1.5z"/></svg>',
    expand: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 2h4v1.5H3.5V6H2V2zm8 0h4v4h-1.5V3.5H10V2zM2 10h1.5v2.5H6V14H2v-4zm10.5 0H14v4h-4v-1.5h2.5V10z"/></svg>',
    shrink: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6 2H4.5v1.5H2V5h4V2zm4 0v3h4V3.5h-2.5V2H10zM2 11v1.5h2.5V14H6v-3H2zm8 0v3h1.5v-1.5H14V11h-4z"/></svg>',
    clock: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.75" fill="none" stroke="currentColor" stroke-width="1.5"/><path fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" d="M8 4.9V8l2 1.4"/></svg>',
    stop: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect fill="currentColor" x="4" y="4" width="8" height="8" rx="1.2"/></svg>',
    prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" d="M9.5 3.5L5 8l4.5 4.5"/></svg>',
    next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" d="M6.5 3.5L11 8l-4.5 4.5"/></svg>',
  };

  // The clock turns amber for the last five minutes and red once time is up.
  const LOW_TIME_MS = 5 * 60 * 1000;
  const clockState = (left) => (left === 0 ? "is-failed" : left <= LOW_TIME_MS ? "is-warning" : "");

  const endButton = () =>
    `<button class="button pp-action mock-end" type="button" id="mockEnd">${ICONS.stop}End mock early</button>`;

  function renderRunning() {
    const list = problems();
    const map = statuses();
    const active = list[session.activeIndex] || list[0];
    const left = mock.remainingMs(session);
    const expired = left === 0;
    const done = completedCount();

    if (!active) {
      return `
        <section class="page-shell mock-run-body">
          <header class="theory-head pp-head mock-head">
            <div class="mock-head-top"><h1 id="mock-run-title">Mock in Progress</h1>${endButton()}</div>
          </header>
          <p class="mock-empty">This mock has no problems.</p>
        </section>`;
    }

    const coding = problemType(active) === "Coding";
    const kind = problemKind(active);
    const isDone = map[active.id] === "complete";
    const fs = fsActive();
    const split = coding && view === "split";

    const switcher = list
      .map((p, i) => {
        const solved = map[p.id] === "complete";
        const current = i === session.activeIndex;
        return `<button class="mock-tab ${current ? "active" : ""} ${solved ? "done" : ""}" type="button" data-go="${i}" ${current ? 'aria-current="true"' : ""}>
            <span class="mock-tab-num">${solved ? ICONS.check : i + 1}</span>
            <span class="mock-tab-copy">
              <strong>${escapeHtml(p.title)}</strong>
              <small>${solved ? `<span class="sr-only">Done. </span>` : ""}${escapeHtml(problemKind(p).label)} · ${escapeHtml(p.difficulty)}</small>
            </span>
          </button>`;
      })
      .join("");

    const viewButton = (id, label) =>
      `<button class="mock-view-option ${view === id ? "active" : ""}" type="button" data-view="${id}" aria-pressed="${view === id}">${label}</button>`;

    const body = coding
      ? `<iframe class="mock-ide" title="Notebook for ${escapeHtml(active.title)}" src="${escapeHtml(
          `${root}editor/?nb=${encodeURIComponent(notebookId(active))}&embed=1&layout=${view}`,
        )}"></iframe>`
      : renderTheoryStatement(active);

    return `
      <section class="page-shell mock-run-body">
        <header class="theory-head pp-head mock-head" aria-labelledby="mock-run-title">
          <div class="mock-head-top">
            <p class="theory-kicker pp-kicker">
              <span class="mock-head-step">Problem ${session.activeIndex + 1} of ${list.length}</span>
              <span class="pp-type pp-type--${kind.id}">${kind.page}</span>
            </p>
            ${endButton()}
          </div>
          <h1 id="mock-run-title">${escapeHtml(active.title)}</h1>
          <div class="theory-meta">${difficultyChip(active)}</div>
        </header>
        <nav class="mock-switcher" aria-label="Problems in this mock">${switcher}</nav>
        <section class="panel editor-panel mock-panel">
          <div class="editor-toolbar mock-toolbar">
            <div class="mock-tools">
              <button class="button button-primary mock-mark ${isDone ? "done" : ""}" type="button" data-toggle-done="${escapeHtml(active.id)}" aria-pressed="${isDone ? "true" : "false"}">
                ${ICONS.check}${isDone ? "Marked done" : "Mark as done"}
              </button>
              ${
                coding
                  ? `<div class="mock-view" role="group" aria-label="What to show">${viewButton("text", "Statement")}${viewButton("single", "Code")}</div>
                    <button class="editor-split mock-split" type="button" id="mockSplit" role="switch" aria-checked="${split}" ${splitAllowed() ? "" : "hidden"}>
                      Split view
                      <span class="editor-split-track" aria-hidden="true"><span></span></span>
                    </button>`
                  : ""
              }
              <button class="button button-secondary dark-text mock-download" type="button" data-download="${escapeHtml(active.id)}">${ICONS.download}Download .ipynb</button>
              <button class="button button-secondary dark-text mock-fullscreen" type="button" id="mockFullscreen" aria-pressed="${fs ? "true" : "false"}">
                ${fs ? ICONS.shrink : ICONS.expand}<span>${fs ? "Exit full screen" : "Full screen"}</span>
              </button>
            </div>
            <div class="mock-status">
              <span class="mock-progress"><strong>${done}</strong> / ${list.length} done</span>
              <span class="editor-kernel mock-clock ${clockState(left)}" id="mockClock" role="timer" aria-live="off">
                ${ICONS.clock}
                <span class="mock-clock-label">${expired ? "Time is up" : "Time remaining"}</span>
                <span class="mock-timer" id="mockTimer">${mock.clockText(left)}</span>
              </span>
            </div>
          </div>
          <div class="mock-body mock-body--${kind.id}" id="mockBody" ${coding ? `data-view="${view}"` : ""}>${body}</div>
          <nav class="mock-pager" aria-label="Previous and next problem">
            <button class="button button-secondary dark-text" type="button" data-go="${session.activeIndex - 1}" ${session.activeIndex === 0 ? "disabled" : ""}>${ICONS.prev}Previous</button>
            <span class="mock-pager-where">Problem ${session.activeIndex + 1} of ${list.length}</span>
            <button class="button button-secondary dark-text" type="button" data-go="${session.activeIndex + 1}" ${session.activeIndex >= list.length - 1 ? "disabled" : ""}>Next${ICONS.next}</button>
          </nav>
        </section>
      </section>
    `;
  }

  // What the student kept on a problem, for its row on the summary.
  function keptLine(problem) {
    if (problemKind(problem).id === "mcq") {
      const block = choiceBlock(problem);
      const picked = block ? picks[block.key] : undefined;
      return Number.isInteger(picked) && block.choices[picked] ? `<small>You picked ${letterOf(picked)}</small>` : "";
    }
    const data = theoryLookup.get(problem.id);
    if (!data) return "";
    const written = data.parts
      .filter((part) => String(answers[`${problem.id}:${part.id}`] || "").trim())
      .map((part) => `Part ${escapeHtml(partLabel(part))}: ${escapeHtml(String(answers[`${problem.id}:${part.id}`]).trim())}`);
    return written.length ? `<small>Your answers: ${written.join(", ")}</small>` : "";
  }

  function renderDone() {
    const list = problems();
    const map = statuses();
    const done = completedCount();
    const used = session.startedAt && session.finishedAt ? session.finishedAt - session.startedAt : 0;
    const limit = session.minutes * 60 * 1000;
    const ranOut = used >= limit;

    const rows = list
      .map((p, i) => {
        const href = problemHref(p);
        const solved = map[p.id] === "complete";
        return `
          <tr class="mock-row ${solved ? "done" : ""}">
            <td class="mock-col-num"><span class="mock-chosen-index">${i + 1}</span></td>
            <td class="mock-col-title">
              <span class="mock-run-copy">
                <strong>${escapeHtml(p.title)}</strong>
                ${keptLine(p)}
              </span>
            </td>
            <td data-label="Type">${typePill(p)}</td>
            <td data-label="Difficulty">${difficultyChip(p)}</td>
            <td data-label="Result">
              <span class="mock-result-pill ${solved ? "done" : ""}">${solved ? `${ICONS.check}Done` : "Not done"}</span>
            </td>
            <td class="mock-col-action">
              ${href ? `<a class="mock-solution-link" href="${escapeHtml(href)}">View problem and solutions${ICONS.next}</a>` : ""}
            </td>
          </tr>`;
      })
      .join("");

    return `
      <section class="page-shell mock-run-body">
        <header class="theory-head pp-head mock-head" aria-labelledby="mock-run-title">
          <h1 id="mock-run-title">Mock Complete</h1>
          <p class="mock-head-note">${
            ranOut ? "You used the full time limit." : `You finished with ${mock.clockText(limit - used)} to spare.`
          } Open any problem below to read the full write-up and its solutions.</p>
          <div class="theory-toolbar pp-actions">
            <a class="button pp-action pp-complete" href="${root}mock/">Build Another Mock</a>
            <a class="button pp-action" href="${root}problems/">Problem Bank</a>
          </div>
        </header>
        <div class="mock-summary">
          <div class="mock-stat mock-stat--solved"><span>Solved</span><strong>${done} / ${list.length}</strong></div>
          <div class="mock-stat"><span>Time Used</span><strong>${mock.clockText(Math.min(used, limit))}</strong></div>
          <div class="mock-stat"><span>Limit</span><strong>${session.minutes}m</strong></div>
        </div>
        <div class="mock-table-wrap">
          <table class="mock-table mock-review-table">
            <thead>
              <tr>
                <th scope="col" class="mock-col-num">#</th>
                <th scope="col">Problem</th>
                <th scope="col">Type</th>
                <th scope="col">Difficulty</th>
                <th scope="col">Result</th>
                <th scope="col" class="mock-col-action"><span class="sr-only">Open</span></th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>
    `;
  }

  function render() {
    stopTicker();
    if (session.status === "done") {
      els.shell.innerHTML = renderDone();
      document.title = "Mock complete | USAAIO Guide";
      return;
    }
    els.shell.innerHTML = renderRunning();
    document.title = "Mock in progress | USAAIO Guide";
    startTicker();
    const active = activeProblem();
    const frame = editorFrame();
    if (frame && active) wireFrame(frame, active);
  }

  /* --------------------------------- timer --------------------------------- */

  function startTicker() {
    stopTicker();
    ticker = window.setInterval(() => {
      const node = document.querySelector("#mockTimer");
      if (!node) return stopTicker();
      const left = mock.remainingMs(session);
      const text = mock.clockText(left);
      if (node.textContent !== text) node.textContent = text;
      const clock = document.querySelector("#mockClock");
      if (clock) clock.classList.toggle("is-warning", clockState(left) === "is-warning");
      if (left === 0) {
        stopTicker();
        finish();
      }
    }, 250);
  }

  function stopTicker() {
    if (ticker) window.clearInterval(ticker);
    ticker = null;
  }

  function finish() {
    // The summary is a normal page. Never leave it stuck in the expanded layout.
    if (fsActive()) {
      els.shell.classList.remove("mock-fs");
      document.body.classList.remove("mock-fs-active");
      if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    }
    session.status = "done";
    session.finishedAt = Date.now();
    mock.save(session);
    render();
    window.scrollTo(0, 0);
  }

  /* --------------------------------- events -------------------------------- */

  els.shell.addEventListener("click", (event) => {
    const go = event.target.closest("[data-go]");
    if (go && !go.disabled) {
      const index = Number(go.dataset.go);
      const list = problems();
      if (index < 0 || index >= list.length) return;
      session.activeIndex = index;
      view = openingView();
      mock.save(session);
      render();
      // The pager sits below the problem, so bring the new title into view.
      const head = els.shell.querySelector(".mock-head");
      if (head && head.getBoundingClientRect().top < 0) {
        if (fsActive()) els.shell.scrollTop = 0;
        else window.scrollTo(0, 0);
      }
      return;
    }

    if (event.target.closest("#mockFullscreen")) {
      setFullscreen(!fsActive());
      return;
    }

    const viewOption = event.target.closest("[data-view]");
    if (viewOption) {
      setView(viewOption.dataset.view === "single" ? "single" : "text");
      return;
    }

    if (event.target.closest("#mockSplit")) {
      const on = view !== "split";
      storeSplit(on);
      setView(on ? "split" : "single");
      return;
    }

    // Pick (or unpick) a multiple-choice option. Patched in place, with no
    // redraw, so the page does not jump.
    const option = event.target.closest("[data-pick]");
    if (option) {
      const key = option.dataset.pick;
      const index = Number(option.dataset.pickIndex);
      if (picks[key] === index) delete picks[key];
      else picks[key] = index;
      saveKept();
      option.closest(".mcq-options").querySelectorAll("[data-pick]").forEach((node) => {
        const on = picks[key] === Number(node.dataset.pickIndex);
        node.setAttribute("aria-pressed", on ? "true" : "false");
        node.closest(".mcq-option").dataset.picked = String(on);
      });
      return;
    }

    const download = event.target.closest("[data-download]");
    if (download) {
      const problem = problemLookup.get(download.dataset.download);
      if (problem) downloadActive(problem);
      return;
    }

    const toggle = event.target.closest("[data-toggle-done]");
    if (toggle && storage) {
      const id = toggle.dataset.toggleDone;
      const isDone = statuses()[id] === "complete";
      storage.setProblemStatus(id, isDone ? "not-started" : "complete");
      // A redraw would reload the notebook editor and lose its place, so only
      // the parts that changed are patched.
      if (!patchDone(id, !isDone)) render();
      return;
    }

    if (event.target.closest("#mockEnd")) {
      const left = mock.remainingMs(session);
      if (left > 0 && !window.confirm(`End the mock now? You still have ${mock.clockText(left)} left.`)) return;
      finish();
    }
  });

  // A number written for a part is kept as it is typed.
  els.shell.addEventListener("input", (event) => {
    const input = event.target.closest("[data-answer]");
    if (!input) return;
    if (input.value.trim()) answers[input.dataset.answer] = input.value.slice(0, 80);
    else delete answers[input.dataset.answer];
    saveKept();
  });

  // Enter in an answer box does not send the form anywhere.
  els.shell.addEventListener("submit", (event) => {
    if (event.target.closest(".mock-answer")) event.preventDefault();
  });

  // Updates the mark button, the active tab and the done count without a redraw.
  function patchDone(id, nowDone) {
    const button = els.shell.querySelector(".mock-mark");
    const tab = els.shell.querySelector(".mock-tab.active");
    const count = els.shell.querySelector(".mock-progress strong");
    if (!button || !tab || !count || button.dataset.toggleDone !== id) return false;
    button.classList.toggle("done", nowDone);
    button.setAttribute("aria-pressed", nowDone ? "true" : "false");
    button.innerHTML = `${ICONS.check}${nowDone ? "Marked done" : "Mark as done"}`;
    tab.classList.toggle("done", nowDone);
    const num = tab.querySelector(".mock-tab-num");
    if (num) num.innerHTML = nowDone ? ICONS.check : String(session.activeIndex + 1);
    const small = tab.querySelector(".mock-tab-copy small");
    const label = small && small.querySelector(".sr-only");
    if (small && nowDone && !label) small.insertAdjacentHTML("afterbegin", '<span class="sr-only">Done. </span>');
    if (label && !nowDone) label.remove();
    count.textContent = String(completedCount());
    return true;
  }

  /* ---------------------------------- boot --------------------------------- */

  // Nothing to sit, so send them back to build a set.
  if (session.status === "setup" || !session.problemIds.length) {
    window.location.replace(`${root}mock/`);
    return;
  }
  // Clock ran out while the tab was closed.
  if (session.status === "running" && mock.remainingMs(session) === 0) {
    session.status = "done";
    session.finishedAt = session.endsAt;
    mock.save(session);
  }
  view = openingView();
  render();
})();
