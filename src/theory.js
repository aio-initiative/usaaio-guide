/*
 * Theory problem reader. Renders one multi-part problem from
 * window.AIO_THEORY in the layout the official USAAIO papers use, gives each
 * part its own reveal-answer button, and exports the whole problem as an
 * .ipynb whose text cells follow the official submission convention
 * ("Each part's name/number must appear in a text cell in bold section format").
 */
(function () {
  // The Mock Arena renders its Text and MCQ problems with these same functions
  // (window.AIO_THEORY_VIEW, set at the foot), so this file loads there too and
  // only draws a page on /theory/.
  const root = document.body.dataset.root || "";
  const params = new URLSearchParams(window.location.search);
  const problemId = (params.get("id") || "").replace(/[^a-z0-9-]/g, "");
  // Opened from a module's problem list: previous and next stay in that module.
  const fromModule = params.get("from") === "module";

  const els = {
    paper: document.querySelector("#theoryPaper"),
    empty: document.querySelector("#theoryEmptyState"),
  };

  const storage = window.AIO_PROGRESS;
  const theoryData = window.AIO_THEORY || { problems: [] };
  const catalog = window.AIO_CONTENT || { modules: [] };
  const moduleLookup = new Map((catalog.modules || []).map((module) => [module.id, module]));

  let problem = null;
  // A multiple-choice problem is one question judged in the page; it has a
  // single part and no part heading.
  let singleQuestion = false;
  let registered = null;
  // Links and status the one-question card needs, set by render().
  let questionLinks = {};

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  /* ------------------------------ math + inline ----------------------------- */

  function katexInline(tex) {
    if (!window.katex) return `<code>${escapeHtml(tex)}</code>`;
    try {
      return window.katex.renderToString(tex, { throwOnError: false, displayMode: false });
    } catch (error) {
      return `<code>${escapeHtml(tex)}</code>`;
    }
  }

  function katexDisplay(tex) {
    if (!window.katex) return `<pre class="theory-tex-fallback">${escapeHtml(tex)}</pre>`;
    try {
      return window.katex.renderToString(tex, { throwOnError: false, displayMode: true });
    } catch (error) {
      return `<pre class="theory-tex-fallback">${escapeHtml(tex)}</pre>`;
    }
  }

  // A `$` that is preceded by a backslash is a literal dollar sign, not a
  // delimiter. Without this, "\$10" would open a math span and swallow the rest
  // of the sentence looking for a partner.
  function findDelimiter(source, from) {
    for (let i = from; i < source.length; i += 1) {
      if (source[i] !== "$") continue;
      let backslashes = 0;
      while (i - backslashes - 1 >= 0 && source[i - backslashes - 1] === "\\") backslashes += 1;
      if (backslashes % 2 === 0) return i;
    }
    return -1;
  }

  // The text halves carry the same small markup the lesson reader supports:
  // `code`, [links](...), **bold**, *italic*. Reuse renderers.js's implementation
  // so the two readers stay in step, and fall back to plain escaped text if that
  // file has not loaded.
  function markup(text) {
    const inlineMarkup = window.AIO_RENDERERS && window.AIO_RENDERERS.inlineMarkup;
    if (!inlineMarkup) return escapeHtml(text).replaceAll("\\$", "$");
    return String(text ?? "")
      .split("`")
      .map((part, index) => (index % 2 ? `<code>${escapeHtml(part)}</code>` : inlineMarkup(part, root)))
      .join("");
  }

  // Splits on $...$ so the text halves get HTML-escaped and the math halves go
  // to KaTeX untouched (escaping first would break \begin{bmatrix} and friends).
  function inline(text) {
    const source = String(text ?? "");
    let html = "";
    let cursor = 0;
    while (cursor < source.length) {
      const open = findDelimiter(source, cursor);
      const close = open < 0 ? -1 : findDelimiter(source, open + 1);
      if (open < 0 || close < 0) {
        html += markup(source.slice(cursor));
        break;
      }
      html += markup(source.slice(cursor, open));
      html += katexInline(source.slice(open + 1, close));
      cursor = close + 1;
    }
    return html;
  }

  function renderCode(code) {
    const language = (code && code.language) || "python";
    const lines = Array.isArray(code && code.lines) ? code.lines : String((code && code.body) || code || "").split("\n");
    return `
      <div class="theory-code">
        <div class="theory-code-head"><span>${escapeHtml(language)}</span></div>
        <pre><code>${lines.map((line) => escapeHtml(line || " ")).join("\n")}</code></pre>
      </div>
    `;
  }

  // Options written as objects { text, correct, explain } are asked in the
  // page with the lesson's question component; plain strings stay a list.
  function renderChoices(choices, partKey) {
    const renderers = window.AIO_RENDERERS;
    const objects = choices.every((item) => item && typeof item === "object");
    if (!objects || !renderers || !renderers.renderChoiceQuestion) {
      const text = (item) => (renderers ? renderers.choiceText(item) : String(item));
      return `<ul class="theory-choices">${choices.map((item) => `<li>${inline(text(item))}</li>`).join("")}</ul>`;
    }
    return renderers.renderChoiceQuestion(
      {
        key: `theory-${partKey}`,
        problemId: problem.id,
        title: problem.title,
        solves: singleQuestion,
        bare: true,
        submit: singleQuestion,
        ...(singleQuestion ? questionLinks : {}),
        choices,
      },
      root,
    );
  }

  // `options.choices(choices, partKey)` draws the options in place of the
  // judged question, and `options.hideAnswers` leaves out `result` blocks.
  // The mock uses both, so no verdict reaches its page.
  function renderBlocks(blocks, partKey, options) {
    if (!Array.isArray(blocks)) return "";
    const opts = options || {};
    return blocks
      .map((block) => {
        if (!block) return "";
        if (block.p) return `<p>${inline(block.p)}</p>`;
        if (block.tex) {
          const note = block.note ? `<p class="theory-tex-note">${inline(block.note)}</p>` : "";
          return `<div class="theory-tex">${katexDisplay(block.tex)}</div>${note}`;
        }
        if (block.list) {
          return `<ul class="theory-list">${block.list.map((item) => `<li>${inline(item)}</li>`).join("")}</ul>`;
        }
        if (block.olist) {
          return `<ol class="theory-list">${block.olist.map((item) => `<li>${inline(item)}</li>`).join("")}</ol>`;
        }
        if (block.choices) return (opts.choices || renderChoices)(block.choices, partKey || "part");
        if (block.code) return renderCode(block.code);
        if (block.result && !opts.hideAnswers) return `<p class="theory-result"><span>Answer</span>${inline(block.result)}</p>`;
        return "";
      })
      .join("");
  }

  /* -------------------------------- rendering ------------------------------- */

  // Part ids are stored as "<problem>.<part>" so they stay unique across the
  // bank, but a reader only ever sees one problem, so the problem number is
  // noise: "3.1" is displayed and exported as "Part 1".
  function partLabel(part) {
    return String(part.id).split(".").pop();
  }

  function partHeading(part) {
    const title = part.title ? ` ${escapeHtml(part.title)}` : "";
    return `Part ${escapeHtml(partLabel(part))}.${title}`;
  }

  /* ----------------------------- answer checker ----------------------------- */

  // A part whose answer has one `result` block holding a single number gets a
  // small checker: the reader types a number and the page compares it with the
  // stored result. Fractions (3/4, \frac{3}{4}), scientific notation (1e-3,
  // 2 \times 10^{-3}) and percentages are understood. Anything else (code, a
  // list, a word) gets no checker.
  function parseNumber(raw) {
    let text = String(raw == null ? "" : raw).trim();
    text = text.replace(/^`+|`+$/g, "").replace(/^\$+|\$+$/g, "").trim();
    text = text
      .replace(/\\[,;!: ]/g, "")
      .replace(/\\(?:text|mathrm)\{([^}]*)\}/g, "$1")
      .replace(/−/g, "-");
    // Two numbers side by side ("3 0.84") are a list, not one number.
    if (/\d\s+[-+]?[\d.]/.test(text)) return null;
    text = text.replace(/\s+/g, "");
    text = text.replace(/^(-?)\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$/, "$1$2/$3");
    text = text.replace(/(?:\\times|\\cdot|×|x|\*)10\^\{?([-+]?\d+)\}?$/, "e$1");
    text = text.replace(/^(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?)$/, (match) => match.replace(/,/g, ""));
    let percent = false;
    if (/\\?%$/.test(text)) {
      percent = true;
      text = text.replace(/\\?%$/, "");
    }
    let value = null;
    let decimals = 0;
    if (/^[-+]?\d+(?:\.\d+)?\/[-+]?\d+(?:\.\d+)?$/.test(text)) {
      const [top, bottom] = text.split("/").map(Number);
      if (bottom) value = top / bottom;
      decimals = 4;
    } else if (/^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(text)) {
      value = Number(text);
      const mantissa = text.replace(/e.*$/i, "");
      decimals = mantissa.includes(".") ? mantissa.split(".")[1].length : 0;
      if (/e/i.test(text)) decimals = Math.max(decimals, 3);
    }
    if (value === null || !Number.isFinite(value)) return null;
    return { value, percent, decimals };
  }

  function numericAnswer(part) {
    const results = (part.answer || []).filter((block) => block && block.result);
    return results.length === 1 ? parseNumber(results[0].result) : null;
  }

  // Close enough: half a unit in the answer's last stated decimal place or
  // 0.1% of its size, whichever is looser. An integer answer must match
  // exactly. A percentage also accepts the same value written as a fraction.
  function answerMatches(expected, given) {
    const candidates = [given.value];
    if (expected.percent && !given.percent) candidates.push(given.value * 100);
    if (!expected.percent && given.percent) candidates.push(given.value / 100);
    const integer = expected.decimals === 0 && Number.isInteger(expected.value);
    const tolerance = integer
      ? 1e-9
      : Math.max(0.5 * 10 ** -expected.decimals, Math.abs(expected.value) * 0.001);
    return candidates.some((value) => Math.abs(value - expected.value) <= tolerance + 1e-12);
  }

  // The checker sits on one line with the reveal button: a number box and
  // Check, with the verdict on its own line under them.
  function renderChecker(part, label) {
    if (!numericAnswer(part)) return "";
    const id = `check-${escapeHtml(label)}`;
    return `
        <form class="theory-check" data-check-part="${escapeHtml(label)}" novalidate>
          <label class="sr-only" for="${id}">Check your answer to Part ${escapeHtml(label)}</label>
          <div class="theory-check-row">
            <input class="theory-check-input" id="${id}" type="text" inputmode="decimal" autocomplete="off" spellcheck="false" placeholder="Your answer, as a number" />
            <button class="theory-check-button" type="submit">Check</button>
          </div>
        </form>`;
  }

  // The first answer checked marks the problem in progress. One already in
  // progress or done stays as it is. app.js redraws the status circle.
  function markInProgress() {
    if (!storage || (storage.loadState().problemStatus[problem.id] || "not-started") !== "not-started") return;
    storage.setProblemStatus(problem.id, "in-progress");
    document.dispatchEvent(new CustomEvent("aio-problem-status-set", { detail: { id: problem.id, status: "in-progress" } }));
  }

  function checkPart(form) {
    const part = problem.parts.find((item) => partLabel(item) === form.dataset.checkPart);
    const expected = part && numericAnswer(part);
    const input = form.querySelector(".theory-check-input");
    const result = els.paper.querySelector(`#check-result-${CSS.escape(form.dataset.checkPart)}`);
    if (!expected || !input || !result) return;
    const given = parseNumber(input.value);
    form.classList.remove("is-correct", "is-wrong");
    result.classList.remove("is-correct", "is-wrong");
    if (!given) {
      result.textContent = input.value.trim() ? "That is not a number this checker can read. Try 0.25, 3/4 or 1e-3." : "Type a number first.";
      return;
    }
    const right = answerMatches(expected, given);
    markInProgress();
    form.classList.add(right ? "is-correct" : "is-wrong");
    result.classList.add(right ? "is-correct" : "is-wrong");
    result.textContent = right ? "Correct. Reveal the answer to compare your working." : "Not yet. Check your working, or reveal the answer.";
  }

  // A part reads top to bottom: its number and what the grader asks for,
  // the question, then one row of tools (the checker and the reveal button)
  // and the solution under them once revealed.
  function renderPart(part) {
    const label = partLabel(part);
    // A one-question multiple-choice page judges each option in place and shows
    // its explanation there, so it has no reveal button, checker or heading.
    if (singleQuestion) {
      return `
      <section class="theory-part theory-part-single" id="part-${escapeHtml(label)}" data-part="${escapeHtml(label)}">
        <div class="theory-part-body">${renderBlocks(part.prompt, label)}</div>
      </section>
    `;
    }
    const checker = renderChecker(part, label);
    return `
      <section class="theory-part" id="part-${escapeHtml(label)}" data-part="${escapeHtml(label)}">
        <div class="theory-part-top">
          <h2 class="theory-part-head">${partHeading(part)}</h2>
          ${part.requirement ? `<p class="theory-requirement">${inline(part.requirement)}</p>` : ""}
        </div>
        <div class="theory-part-body">${renderBlocks(part.prompt, label)}</div>
        <div class="theory-part-tools">
          ${checker}
          <button class="theory-reveal" type="button" data-reveal-part="${escapeHtml(label)}" aria-expanded="false" aria-controls="answer-${escapeHtml(label)}">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 3C4.7 3 2 5.4 1 8c1 2.6 3.7 5 7 5s6-2.4 7-5c-1-2.6-3.7-5-7-5zm0 8a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0-1.6a1.4 1.4 0 1 0 0-2.8 1.4 1.4 0 0 0 0 2.8z"/></svg>
            <span>Reveal Answer</span>
          </button>
        </div>
        ${checker ? `<p class="theory-check-result" id="check-result-${escapeHtml(label)}" role="status" aria-live="polite"></p>` : ""}
        <div class="theory-answer" id="answer-${escapeHtml(label)}" hidden>
          <p class="theory-answer-label">Solution</p>
          ${renderBlocks(part.answer)}
        </div>
      </section>
    `;
  }

  const ICONS = {
    eye: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 3C4.7 3 2 5.4 1 8c1 2.6 3.7 5 7 5s6-2.4 7-5c-1-2.6-3.7-5-7-5zm0 8a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0-1.6a1.4 1.4 0 1 0 0-2.8 1.4 1.4 0 0 0 0 2.8z"/></svg>',
    download: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 2.25a.75.75 0 0 1 .75.75v5.19l1.72-1.72a.75.75 0 1 1 1.06 1.06l-3 3a.75.75 0 0 1-1.06 0l-3-3a.75.75 0 0 1 1.06-1.06l1.72 1.72V3A.75.75 0 0 1 8 2.25zM3.5 12a.75.75 0 0 0 0 1.5h9a.75.75 0 0 0 0-1.5z"/></svg>',
    book: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" d="M2.5 3.2c1.9-.6 3.8-.4 5.5.8v9c-1.7-1.2-3.6-1.4-5.5-.8zM13.5 3.2c-1.9-.6-3.8-.4-5.5.8v9c1.7-1.2 3.6-1.4 5.5-.8z"/></svg>',
    tick: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="m3.8 8.3 2.7 2.7 5.7-6"/></svg>',
  };

  // The lesson part that asks this problem, from content/problem-sections.js:
  // its own module's lesson first, then any other lesson that asks it.
  function lessonPlace() {
    const places = (window.AIO_PROBLEM_SECTIONS || {})[problem.id] || [];
    const place = places.find((entry) => entry[0] === problem.moduleId) || places[0];
    if (!place || !moduleLookup.has(place[0])) return null;
    return { module: moduleLookup.get(place[0]), anchor: place[1], title: place[2] };
  }

  // A few problems to try next: the nearest ones in the same module, then
  // problems of the same type and stage from the modules closest to it.
  function similarProblems(problems, limit) {
    const renderers = window.AIO_RENDERERS;
    const here = problems.indexOf(registered);
    if (!registered || here < 0 || !renderers) return [];
    const moduleOrder = new Map((catalog.modules || []).map((module, index) => [module.id, index]));
    const moduleAt = (entry) => (moduleOrder.has(entry.moduleId) ? moduleOrder.get(entry.moduleId) : 1e6);
    const near = (a, b) => Math.abs(problems.indexOf(a) - here) - Math.abs(problems.indexOf(b) - here);
    const sameModule = problems
      .filter((entry) => entry.moduleId === problem.moduleId && entry.id !== problem.id)
      .sort(near)
      .slice(0, limit - 1);
    const type = renderers.problemType(registered);
    const stage = (registered.tags || []).find((tag) => ["Round 1", "Round 2", "Camp", "Advanced"].includes(tag));
    const myModule = moduleAt(registered);
    const elsewhere = problems
      .filter(
        (entry) =>
          entry.moduleId !== problem.moduleId &&
          renderers.problemType(entry) === type &&
          (!stage || (entry.tags || []).includes(stage)),
      )
      .sort((a, b) => Math.abs(moduleAt(a) - myModule) - Math.abs(moduleAt(b) - myModule) || near(a, b))
      .slice(0, limit - sameModule.length);
    return [...sameModule, ...elsewhere];
  }

  function renderAside(module, problems, appState) {
    const renderers = window.AIO_RENDERERS;
    const place = lessonPlace();
    const sourceModule = (place && place.module) || module;
    const reviewHref = place
      ? `${root}module/?id=${escapeHtml(place.module.id)}#${escapeHtml(place.anchor)}`
      : module
        ? `${root}module/?id=${escapeHtml(problem.moduleId)}`
        : "";
    const source = sourceModule
      ? `
        <section class="pp-panel pp-source" aria-labelledby="ppSourceTitle">
          <h2 class="pp-panel-title" id="ppSourceTitle">From the Lesson</h2>
          <a class="pp-source-module" href="${root}module/?id=${escapeHtml(sourceModule.id)}">
            <span class="pp-source-icon">${ICONS.book}</span>
            <span><small>Module ${escapeHtml(sourceModule.number || "")}</small><strong>${escapeHtml(sourceModule.title)}</strong></span>
          </a>
          ${
            place
              ? `<p class="pp-source-part"><span>Taught In</span> <a href="${reviewHref}">${inline(place.title)}</a></p>`
              : `<p class="pp-source-part">${inline(sourceModule.summary || "")}</p>`
          }
          <a class="pp-panel-button" href="${reviewHref}">${place ? "Review This Part" : "Review the Module"}<span aria-hidden="true"> →</span></a>
        </section>`
      : "";
    const similar = similarProblems(problems, 4);
    const done = (entry) => (appState.problemStatus || {})[entry.id] === "complete";
    const list = similar.length
      ? `
        <section class="pp-panel pp-similar" aria-labelledby="ppSimilarTitle">
          <h2 class="pp-panel-title" id="ppSimilarTitle">Similar Problems</h2>
          <ul class="pp-similar-list">
            ${similar
              .map((entry) => {
                const sameModule = entry.moduleId === problem.moduleId;
                const href = renderers.problemHref(entry, root, fromModule && sameModule).href;
                const other = sameModule ? "" : moduleLookup.get(entry.moduleId);
                return `
              <li>
                <a class="pp-similar-item" href="${escapeHtml(href)}">
                  <span class="pp-similar-title">${escapeHtml(entry.title)}${done(entry) ? `<span class="pp-similar-done" title="Done"><span class="sr-only"> (done)</span>${ICONS.tick}</span>` : ""}</span>
                  <span class="pp-similar-meta">
                    ${renderers.typePill(entry)}
                    <span class="problem-chip ${escapeHtml(renderers.difficultyClass(entry.difficulty))}">${escapeHtml(entry.difficulty)}</span>
                    ${other ? `<span class="pp-similar-module">${escapeHtml(other.title)}</span>` : ""}
                  </span>
                </a>
              </li>`;
              })
              .join("")}
          </ul>
          ${module ? `<a class="pp-panel-more" href="${root}problems/?module=${escapeHtml(problem.moduleId)}">All ${escapeHtml(module.title)} problems<span aria-hidden="true"> →</span></a>` : ""}
        </section>`
      : "";
    return { html: source + list, reviewHref, reviewLabel: place ? "Review This Part of the Lesson" : "Review the Module" };
  }

  function render() {
    const module = moduleLookup.get(problem.moduleId);
    const appState = storage ? storage.loadState() : { problemStatus: {} };
    const status = (appState.problemStatus || {})[problem.id] || "not-started";
    const complete = status === "complete";

    const renderers = window.AIO_RENDERERS;
    const problems = (window.AIO_PROBLEMS || {}).problems || [];
    // Opened from a module's list, the back link returns to that list.
    const back = fromModule && module
      ? `<a class="theory-back" href="${root}module/?id=${escapeHtml(problem.moduleId)}#lesson-problems">← ${escapeHtml(module.title)}</a>`
      : `<a class="theory-back" href="${root}problems/">← Back to problems</a>`;
    const fact = (label, value) => `<div class="pp-fact pp-fact--${label.toLowerCase()}"><dt>${label}</dt><dd>${value}</dd></div>`;
    const facts = [
      fact("Type", `<span class="pp-type pp-type--${singleQuestion ? "mcq" : "text"}">${singleQuestion ? "Multiple choice" : "Text answer"}</span>`),
      module ? fact("Module", `<a class="pp-fact-link" href="${root}module/?id=${escapeHtml(problem.moduleId)}">${escapeHtml(module.title)}</a>`) : "",
      fact("Difficulty", `<span class="problem-chip difficulty-${escapeHtml(String(problem.difficulty).toLowerCase().replace(/\s+/g, "-"))}">${escapeHtml(problem.difficulty)}</span>`),
      singleQuestion ? "" : fact("Parts", escapeHtml(problem.parts.length)),
      problem.minutes && !singleQuestion ? fact("Time", `About ${escapeHtml(problem.minutes)} min`) : "",
    ].join("");

    const aside = renderAside(module, problems, appState);
    const stepper = renderers && renderers.renderProblemStepper ? renderers.renderProblemStepper(problem.id, problems, root, fromModule) : "";
    // The question card's Next problem button goes where the header's next arrow does.
    const scope = fromModule ? problems.filter((entry) => entry.moduleId === problem.moduleId) : problems;
    const next = registered ? scope[scope.indexOf(registered) + 1] : null;
    questionLinks = {
      nextHref: next && renderers ? renderers.problemHref(next, root, fromModule).href : "",
      reviewHref: aside.reviewHref.replaceAll("&amp;", "&"),
      reviewLabel: aside.reviewLabel,
      status: complete ? "complete" : "",
    };

    els.paper.innerHTML = `
      <header class="theory-head pp-head">
        <div class="pp-head-top">
          ${back}
          ${stepper}
        </div>
        <div class="pp-title-row">
          ${renderers && renderers.statusCircle && storage ? renderers.statusCircle(problem.id, problem.title, status, "pp-check") : ""}
          <h1>${escapeHtml(problem.title)}</h1>
        </div>
        <dl class="pp-facts">${facts}</dl>
        <div class="theory-toolbar pp-actions">
          <div class="pp-actions-group">
            ${module ? `<a class="button pp-action pp-action-primary theory-meta-link" href="${root}module/?id=${escapeHtml(problem.moduleId)}">Read the Module</a>` : ""}
            ${singleQuestion ? "" : `<button class="button pp-action" type="button" id="theoryDownload">${ICONS.download}Download .ipynb</button>`}
          </div>
          <div class="pp-actions-group pp-actions-group--end">
            ${singleQuestion ? "" : `<button class="button pp-action" type="button" id="theoryRevealAll">${ICONS.eye}<span>Reveal All Answers</span></button>`}
            <button class="button pp-action pp-action-quiet definition-bank-button" type="button" data-definition-bank aria-haspopup="dialog">Definition Bank</button>
          </div>
        </div>
      </header>
      <div class="pp-layout">
        <div class="pp-main">
          ${
            (problem.overview || []).length || problem.setup
              ? `<div class="theory-overview">
            ${renderBlocks((problem.overview || []).map((text) => ({ p: text })))}
            ${problem.setup ? `<div class="theory-setup">${renderBlocks(problem.setup)}</div>` : ""}
          </div>`
              : ""
          }
          <div class="theory-parts">${problem.parts.map(renderPart).join("")}</div>
        </div>
        <aside class="pp-aside" aria-label="About this problem">${aside.html}</aside>
      </div>
    `;

    typesetPlaceholders();
    window.AIO_DEFINITION_LINKS?.annotate(els.paper);
    // Every theory problem shares /theory/?id=..., so the markup's title and
    // description describe none of them until this runs.
    document.title = `${problem.title} | USAAIO Guide`;
    // overview is an array of content blocks; take the first plain paragraph.
    const firstText = (problem.overview || []).find((block) => typeof block === "string");
    const summary = firstText || `Multi-part USAAIO-style theory problem: ${problem.title}.`;
    [
      ['meta[name="description"]', summary],
      ['meta[property="og:title"]', document.title],
      ['meta[property="og:description"]', summary],
      ['meta[name="twitter:title"]', document.title],
      ['meta[name="twitter:description"]', summary],
    ].forEach(([selector, value]) => {
      const el = document.head.querySelector(selector);
      if (el && value) el.setAttribute("content", String(value).slice(0, 300));
    });
    const here = `https://www.aioinitiative.com${window.location.pathname}${window.location.search}`;
    const canonical = document.head.querySelector('link[rel="canonical"]');
    if (canonical) canonical.setAttribute("href", here);
    const ogUrl = document.head.querySelector('meta[property="og:url"]');
    if (ogUrl) ogUrl.setAttribute("content", here);
    wireEvents();
  }

  // The shared question component writes math as data-tex placeholders, the
  // way the lesson reader does; this page typesets them straight away.
  function typesetPlaceholders() {
    if (!window.katex) return;
    els.paper.querySelectorAll("[data-tex]").forEach((element) => {
      window.katex.render(element.getAttribute("data-tex"), element, {
        displayMode: element.classList.contains("reader-math-display"),
        throwOnError: false,
      });
    });
  }

  /* --------------------------------- events -------------------------------- */

  function togglePart(partId, force) {
    const panel = els.paper.querySelector(`#answer-${CSS.escape(partId)}`);
    const button = els.paper.querySelector(`[data-reveal-part="${CSS.escape(partId)}"]`);
    if (!panel || !button) return;
    const show = typeof force === "boolean" ? force : panel.hidden;
    panel.hidden = !show;
    button.setAttribute("aria-expanded", show ? "true" : "false");
    button.querySelector("span").textContent = show ? "Hide answer" : "Reveal answer";
    button.classList.toggle("revealed", show);
  }

  function allRevealed() {
    return problem.parts.every((part) => {
      const panel = els.paper.querySelector(`#answer-${CSS.escape(partLabel(part))}`);
      return panel && !panel.hidden;
    });
  }

  function wireEvents() {
    els.paper.addEventListener("submit", (event) => {
      const form = event.target.closest("[data-check-part]");
      if (!form) return;
      event.preventDefault();
      checkPart(form);
    });

    els.paper.addEventListener("click", (event) => {
      const revealButton = event.target.closest("[data-reveal-part]");
      if (revealButton) {
        togglePart(revealButton.dataset.revealPart);
        syncRevealAllLabel();
      }
    });

    const revealAll = els.paper.querySelector("#theoryRevealAll");
    if (revealAll) {
      revealAll.addEventListener("click", () => {
        const show = !allRevealed();
        problem.parts.forEach((part) => togglePart(partLabel(part), show));
        syncRevealAllLabel();
      });
    }

    const download = els.paper.querySelector("#theoryDownload");
    if (download) download.addEventListener("click", downloadNotebook);

    // The status circle in the header is toggled by app.js, like every other
    // copy of it, and solving the question in the page ticks it there too.
  }

  function syncRevealAllLabel() {
    const revealAll = els.paper.querySelector("#theoryRevealAll");
    const label = revealAll && revealAll.querySelector("span");
    if (label) label.textContent = allRevealed() ? "Hide all answers" : "Reveal all answers";
  }

  /* ------------------------------ ipynb export ----------------------------- */

  function blocksToMarkdown(blocks) {
    if (!Array.isArray(blocks)) return [];
    const out = [];
    blocks.forEach((block) => {
      if (!block) return;
      if (block.p) out.push(block.p);
      else if (block.tex) {
        out.push(`$$\n${block.tex}\n$$`);
        if (block.note) out.push(`*${block.note}*`);
      } else if (block.list) out.push(block.list.map((item) => `- ${item}`).join("\n"));
      else if (block.olist) out.push(block.olist.map((item, index) => `${index + 1}. ${item}`).join("\n"));
      else if (block.choices) {
        const letters = "ABCDEFGHIJ";
        out.push(
          block.choices
            .map((item, index) => (typeof item === "string" ? `- ${item}` : `- ${letters[index] || index + 1}. ${item.text}`))
            .join("\n"),
        );
      }
      else if (block.code) {
        const lines = Array.isArray(block.code.lines) ? block.code.lines : [String(block.code)];
        out.push(["```" + (block.code.language || "python"), ...lines, "```"].join("\n"));
      } else if (block.result) out.push(`**Answer:** ${block.result}`);
    });
    return out;
  }

  // nbformat 4.5 requires a unique id per cell (only a warning today, a hard
  // error in future versions). Ids must match /^[a-zA-Z0-9-_]+$/.
  let cellSeq = 0;
  function nextCellId(hint) {
    cellSeq += 1;
    return `${String(hint || "cell").replace(/[^a-zA-Z0-9-_]/g, "-")}-${cellSeq}`.slice(0, 64);
  }

  function markdownCell(paragraphs, metadata, idHint) {
    return {
      id: nextCellId(idHint),
      cell_type: "markdown",
      metadata: metadata || {},
      // Notebook `source` is a list of lines that each keep their trailing \n.
      source: paragraphs.join("\n\n").split(/(?<=\n)/),
    };
  }

  function codeCell(lines, idHint) {
    return {
      id: nextCellId(idHint),
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: lines.join("\n").split(/(?<=\n)/),
    };
  }

  // The export is a blank worksheet: prompts plus an empty cell to work in.
  // Solutions stay on the page behind the per-part reveal buttons so the
  // downloaded notebook can be attempted without the answers sitting in it.
  function buildNotebook(target) {
    const sheet = target || problem;
    const module = moduleLookup.get(sheet.moduleId);
    const cells = [];
    cellSeq = 0;

    cells.push(
      markdownCell(
        [
          `# ${sheet.title}`,
          `**Theory problem** · ${sheet.difficulty}${module ? ` · ${module.title}` : ""} · ${sheet.parts.length} parts`,
          "Answer each part in the cell that follows its heading. Following the official USAAIO submission format, every part's number appears in a text cell in bold section format, for example `# **Part 1**`.",
          "Worked solutions are on the problem page, revealable one part at a time.",
          ...(sheet.overview || []),
          ...blocksToMarkdown(sheet.setup),
        ],
        {},
        "intro",
      ),
    );

    sheet.parts.forEach((part) => {
      const label = partLabel(part);
      cells.push(
        markdownCell(
          [
            `# **Part ${label}**`,
            ...blocksToMarkdown(part.prompt),
            ...(part.requirement ? [`*${part.requirement}*`] : []),
          ],
          {},
          `part-${label}`,
        ),
      );
      if (part.kind === "coding") {
        cells.push(codeCell([`# Your solution for Part ${label}`, ""], `work-${label}`));
      } else {
        cells.push(markdownCell([`*Your answer for Part ${label}:*`, ""], {}, `work-${label}`));
      }
    });

    return {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {
        kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
        language_info: { name: "python", version: "3.x" },
        usaaio: { problem: sheet.id, title: sheet.title, type: "theory", parts: sheet.parts.length },
      },
      cells,
    };
  }

  function downloadNotebook() {
    const blob = new Blob([JSON.stringify(buildNotebook(), null, 1)], { type: "application/x-ipynb+json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${problem.id}.ipynb`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /* ---------------------------------- boot --------------------------------- */

  window.AIO_THEORY_VIEW = { inline, renderBlocks, partLabel, partHeading, numericAnswer, buildNotebook };
  if (document.body.dataset.page !== "theory") return;

  problem = theoryData.problems.find((item) => item.id === problemId);
  registered = ((window.AIO_PROBLEMS || {}).problems || []).find((item) => item.id === problemId) || null;
  singleQuestion = Boolean(
    problem &&
      registered &&
      (registered.tags || []).includes("Multiple choice") &&
      problem.parts.length === 1 &&
      problem.parts[0].kind === "mcq",
  );
  if (!problem || !els.paper) {
    if (els.empty) els.empty.hidden = false;
    if (els.paper) els.paper.hidden = true;
    return;
  }
  render();
})();
