(function () {
  // The asset version this script was loaded with (?v=...). Module pictures carry it too, so a new picture under
  // the same file name is fetched fresh instead of served from the browser cache.
  const ASSET_VERSION = (() => {
    try {
      return new URL(document.currentScript.src).searchParams.get("v") || "";
    } catch (error) {
      return "";
    }
  })();
  const STATUS_LABELS = {
    "not-started": "Not started",
    "in-progress": "In progress",
    complete: "Completed",
    skipped: "Skipped",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function slugify(value) {
    return (
      String(value ?? "")
        .toLowerCase()
        .replace(/&/g, " and ")
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "") || "section"
    );
  }

  /* ---------------------------- inline markdown ---------------------------- */

  // Lesson text comes straight from the Markdown modules, so it carries inline
  // math and a small amount of inline markup. Everything is escaped first and
  // the markup is re-introduced afterwards, so content can never inject HTML.
  //
  // Math is emitted as a placeholder span carrying the TeX in a data attribute
  // rather than being typeset here: KaTeX loads with `defer`, so it is usually
  // not available at render time. app.js typesets these once it is.
  function mathSpan(tex) {
    return `<span class="reader-math-inline" data-tex="${escapeHtml(tex)}"><code>${escapeHtml(tex)}</code></span>`;
  }

  // A `$` preceded by a backslash is a literal dollar sign, not a delimiter.
  function findClosingDollar(source, from) {
    for (let i = from; i < source.length; i += 1) {
      if (source[i] !== "$") continue;
      if (i > 0 && source[i - 1] === "\\") continue;
      return i;
    }
    return -1;
  }

  const INLINE_LINK = /\[([^\]]+)\]\(([^)\s]+)\)/g;

  function inlineMarkup(text, root) {
    return escapeHtml(text)
      .replaceAll("\\$", "$")
      .replace(INLINE_LINK, (match, label, href) => {
        // Only http(s), mailto and site-relative links become anchors; any
        // other scheme falls back to plain text, the way the build's
        // rewriteLinks already does for lesson markdown.
        if (!/^(https?:|mailto:|[/#])/i.test(href)) return label;
        const internal = href.startsWith("/");
        const target = internal ? `${root || ""}${href.slice(1)}` : href;
        const attrs = internal ? "" : ' target="_blank" rel="noopener"';
        return `<a href="${target}"${attrs}>${label}</a>`;
      })
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  // Math and code spans are lifted out behind placeholders before emphasis is
  // applied, then put back afterwards. Doing it the other way round breaks the
  // common case where emphasis wraps a formula or an identifier — the two `**`
  // markers of "**`axis` is the axis that disappears.**" would land in separate
  // segments and never pair up.
  const MATH_TOKEN = "␟";
  const CODE_TOKEN = "␞";
  const TOKEN_PATTERN = new RegExp(`[${MATH_TOKEN}${CODE_TOKEN}](\\d+)[${MATH_TOKEN}${CODE_TOKEN}]`, "g");

  function extractMath(source, formulas) {
    let stripped = "";
    let cursor = 0;
    while (cursor < source.length) {
      const open = findClosingDollar(source, cursor);
      const close = open === -1 ? -1 : findClosingDollar(source, open + 1);
      if (open === -1 || close === -1) {
        stripped += source.slice(cursor);
        break;
      }
      stripped += `${source.slice(cursor, open)}${MATH_TOKEN}${formulas.length}${MATH_TOKEN}`;
      formulas.push({ math: source.slice(open + 1, close) });
      cursor = close + 1;
    }
    return stripped;
  }

  function inline(text, root) {
    const spans = [];
    // Strip any stray placeholder characters so content cannot forge a token.
    const source = String(text ?? "")
      .replaceAll(MATH_TOKEN, "")
      .replaceAll(CODE_TOKEN, "");

    const stripped = source
      .split("`")
      .map((part, index) => {
        if (index % 2 === 0) return extractMath(part, spans);
        spans.push({ code: part });
        return `${CODE_TOKEN}${spans.length - 1}${CODE_TOKEN}`;
      })
      .join("");

    return inlineMarkup(stripped, root).replace(TOKEN_PATTERN, (match, index) => {
      const span = spans[Number(index)];
      return span.code === undefined ? mathSpan(span.math) : `<code>${escapeHtml(span.code)}</code>`;
    });
  }

  /* ------------------------------ lesson blocks ---------------------------- */

  function resolveAssetPath(src, root) {
    const raw = String(src || "");
    if (/^(https?:|mailto:|data:)/.test(raw)) return raw;
    if (raw.startsWith("/")) return `${root || ""}${raw.slice(1)}`;
    return `${root || ""}${raw}`;
  }

  function renderImageBlock(image, root) {
    if (!image || !image.src) return "";
    const src = resolveAssetPath(image.src, root);
    const alt = image.alt || "Module diagram";
    return `
      <figure class="reader-figure">
        <img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}" loading="lazy" />
        ${image.alt ? `<figcaption>${escapeHtml(image.alt)}</figcaption>` : ""}
      </figure>
    `;
  }

  function renderTable(table, root) {
    const head = Array.isArray(table.head) ? table.head : [];
    const rows = Array.isArray(table.rows) ? table.rows : [];
    return `
      <div class="reader-table-wrap">
        <table class="reader-table">
          ${head.length ? `<thead><tr>${head.map((cell) => `<th>${inline(cell, root)}</th>`).join("")}</tr></thead>` : ""}
          <tbody>
            ${rows
              .map((row) => `<tr>${row.map((cell) => `<td>${inline(cell, root)}</td>`).join("")}</tr>`)
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  /* --------------------------- lesson components --------------------------- */

  // <Figure>: an image drawn for the lesson, its alt text, and a caption that
  // may carry inline markup. width/height are its drawn size in CSS pixels, so
  // the page reserves the space and never scales the figure up.
  function renderFigure(figure, root) {
    if (!figure || !figure.src) return "";
    // The width goes in a style too: A's reader stretches every figure image
    // to the column, and a figure should never grow past the size it was drawn.
    const size = figure.width && figure.height
      ? ` width="${escapeHtml(figure.width)}" height="${escapeHtml(figure.height)}" style="width: ${escapeHtml(figure.width)}px"`
      : "";
    return `
      <figure class="reader-figure lesson-figure">
        <img src="${escapeHtml(resolveAssetPath(figure.src, root))}" alt="${escapeHtml(figure.alt || "")}"${size} loading="lazy" decoding="async" />
        ${figure.caption ? `<figcaption>${inline(figure.caption, root)}</figcaption>` : ""}
      </figure>
    `;
  }

  // <Optional>: material past what the module needs, collapsed by default.
  function renderOptional(optional, root, options, index) {
    const prefix = `${options.headingPrefix || "optional"}-opt${index}`;
    return `
      <details class="lesson-optional">
        <summary>
          <span>${inline(optional.title, root)}</span>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.05 4.55 12.5 10l-5.45 5.45" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </summary>
        <div class="lesson-optional-body">${renderBlocks(optional.blocks, root, { ...options, headingPrefix: prefix })}</div>
      </details>
    `;
  }

  // <Warning> and <Info>: a short callout. The label is the bold first line
  // ("Warning", "Warning: <title>", or an Info block's title); the body is
  // markdown like everywhere else. A's amber and navy tokens are the two tints,
  // so no new colours.
  function renderCallout(callout, root, options) {
    if (!callout || !callout.blocks) return "";
    return `
      <div class="lesson-callout lesson-callout--${callout.variant === "info" ? "info" : "warning"}">
        <div class="lesson-callout-label">${inline(callout.label || "", root)}</div>
        <div class="lesson-callout-body">${renderBlocks(callout.blocks, root, options)}</div>
      </div>
    `;
  }

  // <EmbeddedPyEditor>: a Python cell the reader edits and runs, with Run and
  // Reset and nothing else. app.js attaches the behaviour; the initial code is
  // the block's own text so Reset can restore it without a second copy.
  function renderPyEditor(pyEditor, root, options, index) {
    if (!pyEditor || !pyEditor.code) return "";
    const id = `${options.headingPrefix || "py"}-editor-${index}`;
    return `
      <div class="lesson-py-editor" id="${escapeHtml(id)}" data-packages="${escapeHtml((pyEditor.packages || []).join(","))}">
        <div class="lesson-py-editor-shell">
          <textarea class="lesson-py-editor-code" spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="Python code">${escapeHtml(pyEditor.code)}</textarea>
        </div>
        <div class="lesson-py-editor-actions">
          <button class="lesson-py-editor-run" type="button">Run</button>
          <button class="lesson-py-editor-reset" type="button">Reset</button>
          <span class="lesson-py-editor-status" role="status">Python: not started</span>
        </div>
        <div class="lesson-py-editor-output" hidden></div>
      </div>
    `;
  }


  function renderOutputBlock(output) {
    const lines = Array.isArray(output && output.lines) ? output.lines : [];
    return `
      <div class="reader-output">
        <div class="reader-output-title">Output</div>
        <pre>${escapeHtml(lines.join("\n"))}</pre>
      </div>
    `;
  }

  // <Resources>: one row per link, starred ones marked.
  function renderResourceList(resources, root) {
    const rows = resources
      .filter((resource) => resource && resource.url)
      .map((resource) => {
        const external = !/^[/#]/.test(resource.url);
        const attrs = external ? ' target="_blank" rel="noopener"' : "";
        const href = resource.url.startsWith("/") ? `${root || ""}${resource.url.slice(1)}` : resource.url;
        return `
          <li>
            <a href="${escapeHtml(href)}"${attrs}>${escapeHtml(resource.title)}</a>
            ${resource.starred ? '<span class="lesson-resource-star" title="Recommended"><span aria-hidden="true">★</span><span class="sr-only">Recommended</span></span>' : ""}
            ${resource.source ? `<span class="lesson-resource-source">${escapeHtml(resource.source)}</span>` : ""}
          </li>
        `;
      })
      .join("");
    return `<ul class="lesson-resources">${rows}</ul>`;
  }

  // A page inside this site opened from a module (`fromModule`) carries
  // ?from=module, so its previous and next buttons stay in that module.
  function problemHref(problem, root, fromModule) {
    const internal = Boolean(problem.url) && problem.url.startsWith("/");
    let href = internal ? `${root || ""}${problem.url.slice(1)}` : problem.url || "#";
    if (internal && fromModule) href += `${href.includes("?") ? "&" : "?"}from=module`;
    return { href, external: Boolean(problem.url) && !internal };
  }

  function difficultyClass(difficulty) {
    return `difficulty-${String(difficulty || "").toLowerCase().replace(/\s+/g, "-")}`;
  }

  // How often a module's topic has come up on past papers, as usaco.guide
  // shows it: four dots, filled up to the level, then the label. Labels and
  // hints match the main guide's Frequency component; the levels are set by
  // hand in content/frequency.json.
  const FREQUENCY_LABELS = ["Has Not Appeared", "Rare", "Not Frequent", "Somewhat Frequent", "Very Frequent"];
  const FREQUENCY_HINTS = [
    "Has not appeared on a past paper, but could still show up.",
    "Barely tested so far: at most a small part of one past paper.",
    "Appears infrequently on past papers.",
    "Appears somewhat frequently on past papers.",
    "Has appeared on nearly every past paper for its round. No guarantee for future ones.",
  ];

  function frequencyBadge(module) {
    const level = module && Number.isInteger(module.frequency) ? module.frequency : null;
    if (level === null || !FREQUENCY_LABELS[level]) return "";
    const dots = [1, 2, 3, 4].map((step) => `<i class="${step <= level ? "is-on" : ""}"></i>`).join("");
    return `<span class="frequency" data-frequency="${level}" title="${escapeHtml(FREQUENCY_HINTS[level])}"><span class="frequency-dots" aria-hidden="true">${dots}</span><span class="frequency-label">${FREQUENCY_LABELS[level]}</span><span class="sr-only">. ${escapeHtml(FREQUENCY_HINTS[level])}</span></span>`;
  }

  const NOTEBOOK_ICON = '<svg viewBox="0 0 24 24"><path d="M6 3.5h9.5L19 7v13.5H6z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 11.5l2 1.5-2 1.5M12.5 15.5h3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  // One problem inside a lesson, as a full-width row like usaco.guide's: the
  // status control on the left (the same dropdown as everywhere else), then
  // the link to the problem with its kind, title and difficulty. The status
  // sits outside the link, since a button cannot live inside an anchor.
  // "Round 1 - Medium": the problem's stage, from its module, then its
  // difficulty, the way usaco.guide writes "Bronze - Easy".
  function problemStageLine(problem) {
    if (!problem) return "";
    const ordering = window.AIO_ORDERING || {};
    const sectionId = (ordering.moduleIdToSectionMap || {})[problem.moduleId];
    const section = sectionId && (ordering.sectionOrdering || {})[sectionId];
    const stage = section
      ? section.label
      : ["Round 1", "Round 2", "Camp", "Advanced"].find((label) => (problem.tags || []).includes(label)) || "";
    return [stage, problem.difficulty].filter(Boolean).join(" - ");
  }

  function lessonProblemRow({ problem, href, external, title, anchorId, status, notebook }) {
    const control = problem ? statusCircle(problem.id, problem.title, status, "lesson-problem-status") : "";
    const line = problemStageLine(problem);
    return `
      <div class="lesson-problem-card${notebook ? " lesson-notebook-card" : ""}"${anchorId ? ` id="${escapeHtml(anchorId)}"` : ""}>
        <div class="lesson-problem-main">
          ${control}
          <a class="lesson-problem-link" href="${escapeHtml(href)}"${external ? ' target="_blank" rel="noopener"' : ""}>
            ${notebook ? `<span class="lesson-notebook-icon" aria-hidden="true">${NOTEBOOK_ICON}</span>` : ""}
            <span class="lesson-problem-copy">
              <strong>${escapeHtml(title)}</strong>
              ${line ? `<span class="lesson-problem-meta">${escapeHtml(line)}</span>` : ""}
            </span>
            <span class="lesson-problem-arrow" aria-hidden="true">→</span>
          </a>
        </div>
        <p class="lesson-problem-note">Focus Problem – try your best to solve this problem before continuing!</p>
      </div>
    `;
  }

  function renderProblemLinkCard(problem, root, appState) {
    const { href, external } = problemHref(problem, root, true);
    return lessonProblemRow({
      problem,
      href,
      external,
      title: problem.title,
      anchorId: `card-${problem.id}`,
      status: appState && appState.problemStatus ? appState.problemStatus[problem.id] : "",
    });
  }

  const CHOICE_LETTERS = "ABCDEFGHIJ";

  // The one status control for a problem, on every page that shows one: a
  // button whose icon reads the current status (empty circle, half filled
  // while in progress, filled green with a tick once done) and which opens a
  // dropdown to set the status directly. app.js owns one shared menu for the
  // whole page and keeps every copy of the button in step with storage (see
  // openStatusMenu and patchProblemStatus there).
  const PROBLEM_STATUS_LABELS = {
    "not-started": "Not Started",
    "in-progress": "In Progress",
    complete: "Done",
  };

  // Menu order, top to bottom.
  const PROBLEM_STATUS_ORDER = ["not-started", "in-progress", "complete"];

  const PROBLEM_STATUS_PRESSED = { "not-started": "false", "in-progress": "mixed", complete: "true" };

  // Older callers pass true or false for done or not.
  function problemStatusValue(status) {
    if (status === true) return "complete";
    return PROBLEM_STATUS_LABELS[status] ? status : "not-started";
  }

  function statusTitle(status) {
    return PROBLEM_STATUS_LABELS[problemStatusValue(status)];
  }

  function statusIcon(extraClass) {
    return `
      <svg class="pcheck-icon${extraClass ? ` ${extraClass}` : ""}" viewBox="0 0 24 24" aria-hidden="true">
        <circle class="pcheck-ring" cx="12" cy="12" r="10" />
        <path class="pcheck-half" d="M12 4a8 8 0 0 1 0 16z" />
        <path class="pcheck-tick" d="M7.4 12.4l3.1 3.1 6.1-6.7" />
      </svg>`;
  }

  function statusCircle(problemId, title, status, extraClass) {
    const value = problemStatusValue(status);
    return `
      <button class="pcheck${extraClass ? ` ${extraClass}` : ""}" type="button"
        data-cycle-problem-status="${escapeHtml(problemId)}"
        data-problem-title="${escapeHtml(title)}"
        data-status="${value}"
        aria-haspopup="menu"
        aria-expanded="false"
        aria-pressed="${PROBLEM_STATUS_PRESSED[value]}"
        aria-label="${statusTitle(value)}: ${escapeHtml(title)}. Change status"
        title="${statusTitle(value)}">
        ${statusIcon()}
      </button>`;
  }

  // ---- module resources ---------------------------------------------------
  // usaco.guide's resource states. Each has an icon that shows in the row's
  // status square and in the dropdown, so the two always agree.
  const RESOURCE_STATUS_ORDER = ["not-started", "reading", "practicing", "complete", "skipped", "reviewing", "ignored"];
  const RESOURCE_STATUS_LABELS = {
    "not-started": "Not Started",
    reading: "Reading",
    practicing: "Practicing",
    complete: "Complete",
    skipped: "Skipped",
    reviewing: "Reviewing",
    ignored: "Ignored",
  };
  const RESOURCE_STATUS_ICONS = {
    "not-started": "",
    reading: '<path d="M3.5 5.5c2.9-.9 5.7-.6 8.5 1v12c-2.8-1.6-5.6-1.9-8.5-1z" /><path d="M20.5 5.5c-2.9-.9-5.7-.6-8.5 1v12c2.8-1.6 5.6-1.9 8.5-1z" />',
    practicing: '<path d="M8.5 7.5 4 12l4.5 4.5" /><path d="M15.5 7.5 20 12l-4.5 4.5" /><path d="m13.5 5-3 14" />',
    complete: '<path d="M5.5 12.5 10 17l8.5-9.5" />',
    skipped: '<path d="m6 6.5 5.5 5.5L6 17.5" /><path d="m12.5 6.5 5.5 5.5-5.5 5.5" />',
    reviewing: '<path d="m18 6.5-5.5 5.5 5.5 5.5" /><path d="m11.5 6.5-5.5 5.5 5.5 5.5" />',
    ignored: '<path d="m7 7 10 10" /><path d="M17 7 7 17" />',
  };

  function resourceStatusValue(status) {
    return RESOURCE_STATUS_LABELS[status] ? status : "not-started";
  }

  function resourceStatusIcon(status) {
    const paths = RESOURCE_STATUS_ICONS[resourceStatusValue(status)];
    return paths ? `<svg viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>` : "";
  }

  function resourceStatusButton(url, title, status) {
    const value = resourceStatusValue(status);
    return `<button class="rstatus" type="button" data-resource-status="${escapeHtml(url)}" data-resource-title="${escapeHtml(title)}"
        data-status="${value}" aria-haspopup="menu" aria-expanded="false"
        aria-label="${RESOURCE_STATUS_LABELS[value]}: ${escapeHtml(title)}. Change status" title="${RESOURCE_STATUS_LABELS[value]}">${resourceStatusIcon(value)}</button>`;
  }

  function resourceStatusMenu() {
    const items = RESOURCE_STATUS_ORDER.map((value) => `
      <button class="pstatus-item rstatus-item" type="button" role="menuitemradio" aria-checked="false"
        data-status-item data-set-resource-status="${value}" data-status="${value}">
        <span class="rstatus-item-icon">${resourceStatusIcon(value)}</span>
        <span>${RESOURCE_STATUS_LABELS[value]}</span>
      </button>`).join("");
    return `<div class="pstatus-menu rstatus-menu" role="menu" aria-label="Set resource status" hidden>${items}</div>`;
  }

  // The module's resources as one card at the top of the page: a status
  // square, the source, a star for the recommended ones, and the link.
  function renderModuleResources(resources, appState, root) {
    const list = (resources || []).filter((resource) => resource && resource.url);
    if (!list.length) return "";
    const statuses = (appState && appState.resourceStatus) || {};
    const rows = list
      .map((resource) => {
        const external = !/^[/#]/.test(resource.url);
        const href = resource.url.startsWith("/") ? `${root || ""}${resource.url.slice(1)}` : resource.url;
        return `
          <li class="mres-row">
            ${resourceStatusButton(resource.url, resource.title, statuses[resource.url])}
            <span class="mres-source">${escapeHtml(resource.source || "")}</span>
            <span class="mres-title">
              ${resource.starred ? '<span class="mres-star" title="Recommended"><span aria-hidden="true">★</span><span class="sr-only">Recommended: </span></span>' : ""}
              <a href="${escapeHtml(href)}"${external ? ' target="_blank" rel="noopener"' : ""}>${escapeHtml(resource.title)}</a>
            </span>
          </li>`;
      })
      .join("");
    return `
      <section class="mres" id="lesson-resources" aria-labelledby="lesson-resources-title">
        <h2 class="mres-head" id="lesson-resources-title">Resources</h2>
        <ul class="mres-list">${rows}</ul>
      </section>
    `;
  }

  // The shared dropdown app.js moves to whichever button is open.
  function statusMenu() {
    const items = PROBLEM_STATUS_ORDER.map((value) => `
      <button class="pstatus-item" type="button" role="menuitemradio" aria-checked="false"
        data-status-item data-set-problem-status="${value}">
        ${statusIcon("pstatus-item-icon")}
        <span>${PROBLEM_STATUS_LABELS[value]}</span>
      </button>`).join("");
    return `<div class="pstatus-menu" role="menu" aria-label="Set problem status" hidden>${items}</div>`;
  }

  // Previous and next problem for a problem page, in the problem bank's order.
  // A reader who came from a module (?from=module) steps through that
  // module's problems only, and the links carry the flag on.
  function problemNeighbours(problemId, problems, fromModule) {
    const list = Array.isArray(problems) ? problems : [];
    const current = list.find((problem) => problem.id === problemId);
    if (!current) return null;
    const scope = fromModule ? list.filter((problem) => problem.moduleId === current.moduleId) : list;
    const index = scope.indexOf(current);
    return {
      prev: index > 0 ? scope[index - 1] : null,
      next: index < scope.length - 1 ? scope[index + 1] : null,
      index,
      total: scope.length,
    };
  }

  const STEP_ARROWS = {
    prev: '<path d="M14.5 6 8.5 12l6 6" />',
    next: '<path d="M9.5 6l6 6-6 6" />',
  };

  function renderProblemStepper(problemId, problems, root, fromModule) {
    const around = problemNeighbours(problemId, problems, fromModule);
    if (!around) return "";
    const step = (direction, target) => {
      const label = direction === "prev" ? "Previous problem" : "Next problem";
      const icon = `<svg viewBox="0 0 24 24" aria-hidden="true">${STEP_ARROWS[direction]}</svg>`;
      if (!target) {
        return `<span class="pp-step is-disabled" aria-disabled="true" title="${label}"><span class="sr-only">${label}</span>${icon}</span>`;
      }
      return `
        <a class="pp-step" href="${escapeHtml(problemHref(target, root, fromModule).href)}" rel="${direction}"
          data-problem-step="${direction}" aria-label="${label}: ${escapeHtml(target.title)}" title="${label}: ${escapeHtml(target.title)}">${icon}</a>`;
    };
    return `
      <nav class="pp-stepper" aria-label="${fromModule ? "Problems in this module" : "Problems in the bank"}">
        ${step("prev", around.prev)}
        <span class="pp-step-count">${around.index + 1}<span aria-hidden="true"> / </span><span class="sr-only"> of </span>${around.total}</span>
        ${step("next", around.next)}
      </nav>`;
  }

  // Text shown for a choice: an object choice's text, or a legacy string.
  function choiceText(choice) {
    return typeof choice === "string" ? choice : String((choice && choice.text) || "");
  }

  /**
   * A multiple-choice question judged in the page. `choices` are objects
   * { text, correct, explain }. Every verdict is rendered up front and hidden,
   * so math in it is typeset with the rest of the page; app.js shows the
   * verdict of the option picked (see judgeChoice there).
   *   key       unique DOM id base
   *   problemId the problem this question solves, if it solves one
   *   status    the problem's status, for the status dot
   *   showStatus draw the status dot in the head (the lesson's cards)
   *   promptHtml the question, already rendered
   *   submit    pick first and judge on Submit, with a result card (the
   *             problem page). Without it a click judges at once (lessons).
   */
  function renderChoiceQuestion(spec, root) {
    if (spec.submit) return renderSubmitQuestion(spec, root);
    const key = String(spec.key || spec.problemId || "mcq").replace(/[^A-Za-z0-9_-]/g, "-");
    const head = spec.showStatus && spec.problemId
      ? `
        <div class="mcq-head">
          ${statusCircle(spec.problemId, spec.title || "this question", spec.status, "mcq-status")}
          <span class="mcq-kicker">Question</span>
        </div>`
      : "";
    const options = (spec.choices || [])
      .map((choice, index) => {
        const correct = Boolean(choice && choice.correct);
        const verdictId = `${key}-verdict-${index}`;
        return `
          <li class="mcq-option" data-correct="${correct ? "true" : "false"}">
            <button class="mcq-choice" type="button" data-mcq-choice="${index}" aria-describedby="${escapeHtml(verdictId)}">
              <span class="mcq-letter" aria-hidden="true">${CHOICE_LETTERS[index] || index + 1}</span>
              <span class="mcq-text"><span class="sr-only">${CHOICE_LETTERS[index] || index + 1}. </span>${inline(choiceText(choice), root)}</span>
            </button>
            <div class="mcq-verdict" id="${escapeHtml(verdictId)}" aria-live="polite">
              <div class="mcq-verdict-body" hidden>
                <strong>${correct ? "Correct" : "Not quite"}</strong>
                ${choice && choice.explain ? `<p>${inline(choice.explain, root)}</p>` : ""}
              </div>
            </div>
          </li>
        `;
      })
      .join("");
    return `
      <section class="mcq-card${spec.bare ? " mcq-bare" : ""}" id="${escapeHtml(spec.anchor || `${key}-question`)}" data-mcq="${escapeHtml(key)}"
        ${spec.problemId ? `data-mcq-problem="${escapeHtml(spec.problemId)}"` : ""} ${spec.solves ? 'data-mcq-solves="true"' : ""}
        ${spec.promptHtml ? `aria-labelledby="${escapeHtml(key)}-prompt"` : 'aria-label="Choices"'}>
        ${head}
        ${spec.promptHtml ? `<div class="mcq-prompt" id="${escapeHtml(key)}-prompt">${spec.promptHtml}</div>` : ""}
        <ol class="mcq-options">${options}</ol>
      </section>
    `;
  }

  const RESULT_ICON = {
    right: '<path d="M6.5 12.5l3.5 3.5 7.5-8" />',
    wrong: '<path d="M7.5 7.5l9 9M16.5 7.5l-9 9" />',
  };

  /**
   * The problem page's question card. Options are radio cards the reader can
   * change freely, Submit judges the pick, and a result card says whether it
   * was right, explains that choice and gives the answer. Try again clears it.
   * Every explanation is rendered up front and hidden so its math is typeset
   * with the page. app.js judges it (see submitChoice there).
   *   nextHref   the next problem, offered once the answer is right
   *   reviewHref, reviewLabel  the lesson part to reread after a wrong answer
   */
  function renderSubmitQuestion(spec, root) {
    const key = String(spec.key || spec.problemId || "mcq").replace(/[^A-Za-z0-9_-]/g, "-");
    const choices = spec.choices || [];
    const letter = (index) => CHOICE_LETTERS[index] || String(index + 1);
    const answerIndex = choices.findIndex((choice) => choice && choice.correct);
    const options = choices
      .map(
        (choice, index) => `
          <li class="mcq-option" data-correct="${choice && choice.correct ? "true" : "false"}" data-index="${index}">
            <label class="mcq-choice">
              <input class="mcq-radio" type="radio" name="${escapeHtml(key)}-pick" value="${index}" />
              <span class="mcq-letter" aria-hidden="true">${letter(index)}</span>
              <span class="mcq-text"><span class="sr-only">${letter(index)}. </span>${inline(choiceText(choice), root)}</span>
              <span class="mcq-tag" aria-hidden="true"></span>
            </label>
          </li>`,
      )
      .join("");
    const explains = choices
      .map(
        (choice, index) => `
          <div class="mcq-explain" data-explain="${index}" hidden>
            <p class="mcq-explain-label">Why ${letter(index)} ${choice && choice.correct ? "is right" : "is not it"}</p>
            ${choice && choice.explain ? `<p>${inline(choice.explain, root)}</p>` : ""}
          </div>`,
      )
      .join("");
    const icon = (name) => `<svg class="mcq-result-icon mcq-result-icon--${name}" viewBox="0 0 24 24" aria-hidden="true">${RESULT_ICON[name]}</svg>`;
    const answer = answerIndex >= 0
      ? `
          <div class="mcq-answer" hidden>
            <p class="mcq-answer-line"><span>Correct Answer</span><span class="mcq-letter">${letter(answerIndex)}</span><span class="mcq-answer-text">${inline(choiceText(choices[answerIndex]), root)}</span></p>
          </div>`
      : "";
    const next = spec.nextHref
      ? `<a class="mcq-btn mcq-btn-primary mcq-next" href="${escapeHtml(spec.nextHref)}" hidden>Next Problem<span aria-hidden="true"> →</span></a>`
      : "";
    const review = spec.reviewHref
      ? `<a class="mcq-btn mcq-review" href="${escapeHtml(spec.reviewHref)}" hidden>${escapeHtml(spec.reviewLabel || "Review the lesson")}</a>`
      : "";
    return `
      <section class="mcq-card mcq-submit-card${spec.bare ? " mcq-bare" : ""}" id="${escapeHtml(spec.anchor || `${key}-question`)}" data-mcq="${escapeHtml(key)}" data-mcq-mode="submit"
        ${spec.problemId ? `data-mcq-problem="${escapeHtml(spec.problemId)}"` : ""} ${spec.solves ? 'data-mcq-solves="true"' : ""} aria-label="Question">
        <form class="mcq-form" data-mcq-form novalidate>
          <fieldset class="mcq-fieldset">
            <legend class="mcq-legend">Choose One Answer</legend>
            <ol class="mcq-options">${options}</ol>
          </fieldset>
          <div class="mcq-actions">
            <p class="mcq-hint" data-mcq-hint>${spec.status === "complete" ? "Solved before. Answer again any time." : "Pick an answer, then submit."}</p>
            <button class="mcq-btn mcq-btn-primary mcq-submit" type="submit" disabled>Submit</button>
          </div>
        </form>
        <div class="mcq-result" data-mcq-result hidden tabindex="-1">
          <div class="mcq-result-head" role="status">
            ${icon("right")}${icon("wrong")}
            <p class="mcq-result-title"><strong data-mcq-verdict></strong> <span data-mcq-picked></span></p>
          </div>
          <div class="mcq-result-body">
            ${explains}
            ${answer}
          </div>
          <div class="mcq-result-actions">
            <button class="mcq-btn mcq-retry" type="button" data-mcq-retry>Try Again</button>
            ${review}
            ${next}
          </div>
        </div>
      </section>
    `;
  }

  // <ProblemCard id>: a multiple-choice problem is asked right here; any other
  // kind is a one-line link to its page.
  function renderProblemCardBlock(card, root, options) {
    const lookup = options.problemLookup;
    const problem = lookup && lookup.get(card.id);
    if (!problem) return "";
    const prompt = card.mcq && Array.isArray(card.mcq.prompt) ? card.mcq.prompt : null;
    const choicesBlock = prompt && prompt.find((block) => block && Array.isArray(block.choices));
    if (!choicesBlock || !choicesBlock.choices.every((choice) => choice && typeof choice === "object")) {
      return renderProblemLinkCard(problem, root, options.appState);
    }
    const status = options.appState && options.appState.problemStatus ? options.appState.problemStatus[problem.id] : "";
    return renderChoiceQuestion(
      {
        key: `q-${problem.id}`,
        anchor: `card-${problem.id}`,
        problemId: problem.id,
        title: problem.title,
        solves: true,
        status,
        showStatus: true,
        promptHtml: renderBlocks(
          prompt.filter((block) => !Array.isArray(block.choices)),
          root,
          options.headingPrefix ? { headingPrefix: `${options.headingPrefix}-q` } : {},
        ),
        choices: choicesBlock.choices,
      },
      root,
    );
  }

  // <NotebookCard id>: the lesson's notebook, opened in the editor. It is a
  // problem like any other, so it carries the same status control.
  function renderNotebookCard(card, root, options = {}) {
    if (!card || !card.id || !card.title) return "";
    const lookup = options.problemLookup;
    const url = `/editor/?nb=${card.id}`;
    let problem = lookup ? lookup.get(card.id) : null;
    if ((!problem || problem.url !== url) && lookup) {
      problem = Array.from(lookup.values()).find((entry) => entry.url === url) || null;
    }
    return lessonProblemRow({
      problem,
      href: `${root || ""}editor/?nb=${encodeURIComponent(card.id)}`,
      title: card.title,
      anchorId: problem ? `card-${problem.id}` : "",
      status: problem && options.appState && options.appState.problemStatus ? options.appState.problemStatus[problem.id] : "",
      notebook: true,
    });
  }

  function renderBlocks(blocks, root, options = {}) {
    if (!Array.isArray(blocks)) return "";
    return blocks
      .map((block, index) => {
        if (!block) return "";
        // Code followed by its output share one frame, laid out like a runnable cell: the code, then the
        // output in an inset panel below it.
        if (block.code && blocks[index + 1] && blocks[index + 1].output) {
          return `<div class="reader-run">${renderCodeBlock(block.code)}${renderOutputBlock(blocks[index + 1].output)}</div>`;
        }
        if (block.output && index > 0 && blocks[index - 1] && blocks[index - 1].code) return "";
        if (block.p) return `<p>${inline(block.p, root)}</p>`;
        if (block.figure) return renderFigure(block.figure, root);
        if (block.optional) return renderOptional(block.optional, root, options, index);
        if (block.callout) return renderCallout(block.callout, root, options);
        if (block.pyEditor) return renderPyEditor(block.pyEditor, root, options, index);
        if (block.problemCard) return renderProblemCardBlock(block.problemCard, root, options);
        if (block.notebookCard) return renderNotebookCard(block.notebookCard, root, options);
        if (block.resources) return renderResourceList(block.resources, root);
        if (block.output) return renderOutputBlock(block.output);
        if (block.tex) {
          return `<div class="reader-math-display" data-tex="${escapeHtml(block.tex)}"><code>${escapeHtml(block.tex)}</code></div>`;
        }
        if (block.heading) {
          const tag = block.level === 4 ? "h4" : "h3";
          const headingId = options.headingPrefix
            ? `${options.headingPrefix}-${index}-${slugify(block.heading)}`
            : slugify(block.heading);
          return `<${tag} class="reader-subhead" id="${escapeHtml(headingId)}">${inline(block.heading, root)}</${tag}>`;
        }
        if (Array.isArray(block.list)) {
          return `<ul class="reader-block-list">${block.list.map((item) => `<li>${inline(item, root)}</li>`).join("")}</ul>`;
        }
        if (Array.isArray(block.olist)) {
          return `<ol class="reader-block-list">${block.olist.map((item) => `<li>${inline(item, root)}</li>`).join("")}</ol>`;
        }
        if (block.table) return renderTable(block.table, root);
        if (block.image) return renderImageBlock(block.image, root);
        if (Array.isArray(block.quote)) {
          return `<blockquote class="reader-quote">${block.quote.map((line) => `<p>${inline(line, root)}</p>`).join("")}</blockquote>`;
        }
        if (block.code) return renderCodeBlock(block.code);
        if (block.details) {
          return `
            <details class="reader-answer">
              <summary>${escapeHtml(block.details.summary || "Answer")}</summary>
              <div class="reader-answer-body">${renderBlocks(block.details.blocks, root, options)}</div>
            </details>
          `;
        }
        return "";
      })
      .join("");
  }

  function statusOptions(current) {
    return Object.entries(STATUS_LABELS)
      .map(([value, label]) => `<option value="${value}" ${current === value ? "selected" : ""}>${label}</option>`)
      .join("");
  }

  function renderFilterChips(ordering, activeFilter) {
    const allChip = `<button class="filter-chip ${activeFilter === "all" ? "active" : ""}" type="button" data-filter="all">All</button>`;
    const sectionChips = ordering.sections
      .map((sectionId) => {
        const label = ordering.sectionOrdering[sectionId].label;
        const active = activeFilter === sectionId ? "active" : "";
        return `<button class="filter-chip ${active}" type="button" data-filter="${escapeHtml(sectionId)}">${escapeHtml(label)}</button>`;
      })
      .join("");

    return allChip + sectionChips;
  }

  function renderSearchFilters(activeFilter) {
    const types = [
      ["all", "All"],
      ["module", "Modules"],
      ["lesson", "Lessons"],
      ["problem", "Problems"],
      ["resource", "Resources"],
    ];
    return types
      .map(
        ([value, label]) =>
          `<button class="filter-chip ${activeFilter === value ? "active" : ""}" type="button" data-search-filter="${escapeHtml(
            value,
          )}">${escapeHtml(label)}</button>`,
      )
      .join("");
  }

  function countStatuses(ids, statusMap) {
    const counts = {
      complete: 0,
      "in-progress": 0,
      skipped: 0,
      "not-started": 0,
      total: ids.length,
    };
    ids.forEach((id) => {
      const status = statusMap[id] || "not-started";
      counts[status] = (counts[status] || 0) + 1;
    });
    return counts;
  }

  // Where the reader left off in this section: the last module they opened if
  // it is here, else the newest module or problem activity here, else the
  // first module still in progress. Null when nothing here was started.
  function sectionResumeModule(moduleIds, moduleLookup, problems, appState) {
    const moduleSet = new Set(moduleIds);
    const isOpen = (id) => moduleSet.has(id) && appState.moduleStatus[id] !== "complete";
    if (appState.lastModuleId && isOpen(appState.lastModuleId)) return appState.lastModuleId;
    const problemModule = new Map(problems.map((problem) => [problem.id, problem.moduleId]));
    for (const entry of appState.activityLog || []) {
      const id = entry.type === "problem" ? problemModule.get(entry.id) : entry.type === "module" ? entry.id : null;
      if (id && isOpen(id) && moduleLookup.has(id)) return id;
    }
    return moduleIds.find((id) => appState.moduleStatus[id] === "in-progress") || null;
  }

  function renderSectionProgress(sectionId, ordering, moduleLookup, problems, appState, root = "") {
    const section = ordering.sectionOrdering[sectionId];
    if (!section) return "";
    const moduleIds = section.chapters.flatMap((chapter) => chapter.items);
    const moduleSet = new Set(moduleIds);
    const problemIds = problems
      .filter((problem) => {
        const module = moduleLookup.get(problem.moduleId);
        return module && moduleSet.has(module.id);
      })
      .map((problem) => problem.id);
    const resume = moduleLookup.get(sectionResumeModule(moduleIds, moduleLookup, problems, appState));
    const resumeLink = resume
      ? `<a class="stage-resume" href="${escapeHtml(`${root}module/?id=${encodeURIComponent(resume.id)}`)}" data-open-module="${escapeHtml(resume.id)}">Continue: <strong>${escapeHtml(resume.title)}</strong> <span aria-hidden="true">→</span></a>`
      : "";

    return `
      <div class="stage-progress-card">
        <div class="stage-meter-card">${renderStageMeter("Modules", countStatuses(moduleIds, appState.moduleStatus), resumeLink)}</div>
        <div class="stage-meter-card">${renderStageMeter(
          "Practice problems",
          countStatuses(problemIds, appState.problemStatus),
        )}</div>
      </div>
    `;
  }

  function renderStageMeter(label, counts, detail = "") {
    const percent = counts.total ? Math.round((counts.complete / counts.total) * 100) : 0;
    return `
      <div class="stage-meter">
        <div class="stage-meter-head">
          <span>${escapeHtml(label)}</span>
          <b>${escapeHtml(counts.complete)} / ${escapeHtml(counts.total)} complete</b>
        </div>
        <div class="meter-track" role="progressbar" aria-valuenow="${escapeHtml(percent)}" aria-valuemin="0" aria-valuemax="100" aria-label="${escapeHtml(label)} progress"><span class="meter-fill" style="width: ${escapeHtml(percent)}%"></span></div>
        <div class="stage-meter-count">${escapeHtml(percent)}%</div>
        ${detail ? `<div class="stage-meter-detail">${detail}</div>` : ""}
      </div>
    `;
  }

  // Each section-listing row gets one small, topic-specific sketch. The art is
  // deliberately decorative and repeated title text remains the accessible
  // name of the link.
  const MODULE_ILLUSTRATIONS = {
    "advanced-optimization": "/assets/module-illustrations/round2/advanced-optimization.svg",
    "advanced-optimization-methods": "/assets/module-illustrations/round2/advanced-optimization-methods.svg",
    "autoregressive-generative-models": "/assets/module-illustrations/camp/autoregressive-generative-models.svg",
    "backpropagation-mechanics": "/assets/module-illustrations/round2/backpropagation-mechanics.svg",
    "bias-variance-cross-validation": "/assets/module-illustrations/round2/bias-variance-cross-validation.svg",
    "calculus-gradients": "/assets/module-illustrations/round1/calculus-gradients.svg",
    "clustering": "/assets/module-illustrations/round1/clustering.svg",
    "clustering-methods": "/assets/module-illustrations/round1/clustering-methods.svg",
    "colab-markdown-notebooks": "/assets/module-illustrations/round1/colab-markdown-notebooks.svg",
    "contrastive-learning": "/assets/module-illustrations/round2/contrastive-learning.svg",
    "convolutional-neural-networks": "/assets/module-illustrations/round1/convolutional-neural-networks.svg",
    "cross-attention-multimodal-attention": "/assets/module-illustrations/camp/cross-attention-multimodal-attention.svg",
    "data-processing-features": "/assets/module-illustrations/round1/data-processing-features.svg",
    "decision-trees-knn": "/assets/module-illustrations/round1/decision-trees-knn.svg",
    "efficient-attention-long-context": "/assets/module-illustrations/advanced/efficient-attention-long-context.svg",
    "efficient-flashattention": "/assets/module-illustrations/advanced/efficient-flashattention.svg",
    "efficient-kv-cache": "/assets/module-illustrations/advanced/efficient-kv-cache.svg",
    "efficient-linear-attention": "/assets/module-illustrations/advanced/efficient-linear-attention.svg",
    "efficient-prefill-and-decode": "/assets/module-illustrations/advanced/efficient-prefill-and-decode.svg",
    "efficient-setup": "/assets/module-illustrations/advanced/efficient-setup.svg",
    "efficient-sharing-keys-and-values": "/assets/module-illustrations/advanced/efficient-sharing-keys-and-values.svg",
    "evaluating-generative-models": "/assets/module-illustrations/camp/evaluating-generative-models.svg",
    "expectations-multivariate-distributions": "/assets/module-illustrations/round1/expectations-multivariate-distributions.svg",
    "forward-propagation": "/assets/module-illustrations/round2/forward-propagation.svg",
    "generative-adding-noise": "/assets/module-illustrations/camp/generative-adding-noise.svg",
    "generative-adversarial-networks": "/assets/module-illustrations/camp/generative-adversarial-networks.svg",
    "generative-guidance-and-stable-diffusion": "/assets/module-illustrations/camp/generative-guidance-and-stable-diffusion.svg",
    "generative-removing-noise": "/assets/module-illustrations/camp/generative-removing-noise.svg",
    "generative-setup": "/assets/module-illustrations/camp/generative-setup.svg",
    "gpt-causal-lm": "/assets/module-illustrations/round2/gpt-causal-lm.svg",
    "graph-neural-networks": "/assets/module-illustrations/advanced/graph-neural-networks.svg",
    "hierarchical-vision-models": "/assets/module-illustrations/round2/hierarchical-vision-models.svg",
    "information-theory-core": "/assets/module-illustrations/camp/information-theory-core.svg",
    "kernel-methods-svms": "/assets/module-illustrations/round1/kernel-methods-svms.svg",
    "linear-algebra-i-vectors-matrices-systems-transformations": "/assets/module-illustrations/round1/linear-algebra-i-vectors-matrices-systems-transformations.svg",
    "linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory": "/assets/module-illustrations/round1/linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory.svg",
    "linear-logistic-regression": "/assets/module-illustrations/round1/linear-logistic-regression.svg",
    "loss-functions": "/assets/module-illustrations/round1/loss-functions.svg",
    "mixture-of-experts": "/assets/module-illustrations/advanced/mixture-of-experts.svg",
    "model-evaluation": "/assets/module-illustrations/round2/model-evaluation.svg",
    "modern-vision-architectures-resnet-efficientnet-transfer": "/assets/module-illustrations/round2/modern-vision-architectures-resnet-efficientnet-transfer.svg",
    "natural-language-processing-tokenization-embeddings": "/assets/module-illustrations/camp/natural-language-processing-tokenization-embeddings.svg",
    "neural-network-regularization-dropout-batchnorm": "/assets/module-illustrations/round1/neural-network-regularization-dropout-batchnorm.svg",
    "neural-networks-perceptrons-mlps-activations": "/assets/module-illustrations/round1/neural-networks-perceptrons-mlps-activations.svg",
    "neural-operators": "/assets/module-illustrations/camp/neural-operators.svg",
    "nonlinear-dimensionality-reduction": "/assets/module-illustrations/round2/nonlinear-dimensionality-reduction.svg",
    "object-detection-fpn-yolo": "/assets/module-illustrations/camp/object-detection-fpn-yolo.svg",
    "optimization-adam-sgd": "/assets/module-illustrations/round1/optimization-adam-sgd.svg",
    "pandas-data-cleaning-storing": "/assets/module-illustrations/round1/pandas-data-cleaning-storing.svg",
    "plotting-matplotlib-seaborn": "/assets/module-illustrations/round1/plotting-matplotlib-seaborn.svg",
    "policy-gradients": "/assets/module-illustrations/advanced/policy-gradients.svg",
    "pre-training-fine-tuning": "/assets/module-illustrations/camp/pre-training-fine-tuning.svg",
    "preference-optimization-rlhf-and-dpo": "/assets/module-illustrations/advanced/preference-optimization-rlhf-and-dpo.svg",
    "principal-component-analysis": "/assets/module-illustrations/round2/principal-component-analysis.svg",
    "probability-foundations": "/assets/module-illustrations/round1/probability-foundations.svg",
    "programming-essentials": "/assets/module-illustrations/round1/programming-essentials.svg",
    "python-basics": "/assets/module-illustrations/round1/python-basics.svg",
    "pytorch-tensors-autograd": "/assets/module-illustrations/round1/pytorch-tensors-autograd.svg",
    "rag-retrieval-embeddings": "/assets/module-illustrations/camp/rag-retrieval-embeddings.svg",
    "reinforcement-learning": "/assets/module-illustrations/advanced/reinforcement-learning.svg",
    "relative-position-rope-positional-methods": "/assets/module-illustrations/camp/relative-position-rope-positional-methods.svg",
    "representation-learning-autoencoders-vaes-latent-spaces": "/assets/module-illustrations/camp/representation-learning-autoencoders-vaes-latent-spaces.svg",
    "scikit-learn-api-ml-pipelines": "/assets/module-illustrations/round1/scikit-learn-api-ml-pipelines.svg",
    "sciml-derivatives": "/assets/module-illustrations/camp/sciml-derivatives.svg",
    "sciml-network-as-solution": "/assets/module-illustrations/camp/sciml-network-as-solution.svg",
    "sciml-physics-loss": "/assets/module-illustrations/camp/sciml-physics-loss.svg",
    "sciml-setup": "/assets/module-illustrations/camp/sciml-setup.svg",
    "sciml-the-equation": "/assets/module-illustrations/camp/sciml-the-equation.svg",
    "sciml-training-and-checking": "/assets/module-illustrations/camp/sciml-training-and-checking.svg",
    "semantic-segmentation-unet-masks": "/assets/module-illustrations/round2/semantic-segmentation-unet-masks.svg",
    "semi-supervised-learning": "/assets/module-illustrations/round2/semi-supervised-learning.svg",
    "sequence-models-transfer-learning": "/assets/module-illustrations/camp/sequence-models-transfer-learning.svg",
    "sparse-attention-sinks-and-eviction": "/assets/module-illustrations/advanced/sparse-attention-sinks-and-eviction.svg",
    "state-space-models": "/assets/module-illustrations/advanced/state-space-models.svg",
    "statistics-hypothesis-testing": "/assets/module-illustrations/round1/statistics-hypothesis-testing.svg",
    "time-series": "/assets/module-illustrations/round2/time-series.svg",
    "training-loop-in-pytorch": "/assets/module-illustrations/round1/training-loop-in-pytorch.svg",
    "transformer-add-position": "/assets/module-illustrations/round2/transformer-add-position.svg",
    "transformer-architectures-bert-mlm": "/assets/module-illustrations/round2/transformer-architectures-bert-mlm.svg",
    "transformer-architectures-t5-encoder-decoder": "/assets/module-illustrations/round2/transformer-architectures-t5-encoder-decoder.svg",
    "transformer-embed": "/assets/module-illustrations/round2/transformer-embed.svg",
    "transformer-feed-forward": "/assets/module-illustrations/round2/transformer-feed-forward.svg",
    "transformer-fundamentals": "/assets/module-illustrations/round2/transformer-fundamentals.svg",
    "transformer-generate": "/assets/module-illustrations/round2/transformer-generate.svg",
    "transformer-mask-and-softmax": "/assets/module-illustrations/round2/transformer-mask-and-softmax.svg",
    "transformer-multi-head": "/assets/module-illustrations/round2/transformer-multi-head.svg",
    "transformer-predict": "/assets/module-illustrations/round2/transformer-predict.svg",
    "transformer-qkv": "/assets/module-illustrations/round2/transformer-qkv.svg",
    "transformer-residual-and-layernorm": "/assets/module-illustrations/round2/transformer-residual-and-layernorm.svg",
    "transformer-scores": "/assets/module-illustrations/round2/transformer-scores.svg",
    "transformer-setup": "/assets/module-illustrations/round2/transformer-setup.svg",
    "transformer-stack": "/assets/module-illustrations/round2/transformer-stack.svg",
    "transformer-tokenize": "/assets/module-illustrations/round2/transformer-tokenize.svg",
    "transformer-weighted-sum": "/assets/module-illustrations/round2/transformer-weighted-sum.svg",
    "tree-based-models-ensembles": "/assets/module-illustrations/round1/tree-based-models-ensembles.svg",
    "vae-encoder-decoder-generation": "/assets/module-illustrations/camp/vae-encoder-decoder-generation.svg",
    "vision-adversarial-attacks": "/assets/module-illustrations/round2/vision-adversarial-attacks.svg",
    "vision-classify": "/assets/module-illustrations/round2/vision-classify.svg",
    "vision-convolution": "/assets/module-illustrations/round2/vision-convolution.svg",
    "vision-deeper-networks": "/assets/module-illustrations/round2/vision-deeper-networks.svg",
    "vision-pixels": "/assets/module-illustrations/round2/vision-pixels.svg",
    "vision-relu-and-pooling": "/assets/module-illustrations/round2/vision-relu-and-pooling.svg",
    "vision-setup": "/assets/module-illustrations/round2/vision-setup.svg",
    "vision-transfer-learning": "/assets/module-illustrations/round2/vision-transfer-learning.svg",
    "vision-transformers-vit-attention": "/assets/module-illustrations/camp/vision-transformers-vit-attention.svg",
    "why-pytorch-not-tensorflow": "/assets/module-illustrations/round1/why-pytorch-not-tensorflow.svg",
  };

  const MODULE_ILLUSTRATION_STAGES = new Set(["round1", "round2", "camp", "advanced"]);

  function renderModuleIllustration(moduleId, moduleSection, root) {
    const source = MODULE_ILLUSTRATIONS[moduleId] ||
      (MODULE_ILLUSTRATION_STAGES.has(moduleSection)
        ? `/assets/module-illustrations/${moduleSection}/${moduleId}.svg`
        : "");
    if (!source) return "";
    return `
      <span class="topic-illustration" aria-hidden="true">
        <img src="${escapeHtml(resolveAssetPath(source, root))}${ASSET_VERSION ? `?v=${escapeHtml(ASSET_VERSION)}` : ""}" alt="" loading="lazy" decoding="async" onerror="this.parentNode.remove()" />
      </span>
    `;
  }

  function renderSectionTimeline(sectionId, ordering, moduleLookup, appState, root = "") {
    const section = ordering.sectionOrdering[sectionId];
    if (!section) return "";

    return section.chapters
      .map((chapter, chapterIndex) => {
        const completeCount = chapter.items.filter((moduleId) => appState.moduleStatus[moduleId] === "complete").length;
        const chapterPercent = chapter.items.length ? Math.round((completeCount / chapter.items.length) * 100) : 0;
        const topics = chapter.items
          .map((moduleId, topicIndex) => {
            const module = moduleLookup.get(moduleId);
            const title = module ? module.title : moduleId;
            const summary = module ? module.summary : "";
            const meta = module ? module.level : "";
            const href = `${root}module/?id=${encodeURIComponent(moduleId)}`;
            return `
              <li>
                <a class="topic-row" href="${escapeHtml(href)}" data-open-module="${escapeHtml(moduleId)}">
                  <span class="topic-index">${String(topicIndex + 1).padStart(2, "0")}</span>
                  <span class="topic-copy">
                    <strong>${escapeHtml(title)}</strong>
                    <small>${escapeHtml(meta)}</small>
                    ${frequencyBadge(module)}
                    <em>${escapeHtml(summary)}</em>
                  </span>
                  ${renderModuleIllustration(moduleId, module ? module.section : "", root)}
                </a>
              </li>
            `;
          })
          .join("");

        return `
          <article class="chapter-card">
            <header class="chapter-card-head">
              <div class="chapter-card-title">
                <span class="chapter-kicker">Chapter ${escapeHtml(String(chapterIndex + 1).padStart(2, "0"))}</span>
                <h2>${escapeHtml(chapter.name)}</h2>
                ${chapter.description && !/ modules in this stage of the guide\.$/.test(chapter.description) ? `<p>${escapeHtml(chapter.description)}</p>` : ""}
              </div>
              <div class="chapter-meter" aria-label="${escapeHtml(completeCount)} of ${escapeHtml(chapter.items.length)} modules complete">
                <span><i style="width: ${escapeHtml(chapterPercent)}%"></i></span>
                <b>${escapeHtml(completeCount)}/${escapeHtml(chapter.items.length)}</b>
              </div>
            </header>
            <ol class="chapter-topics">${topics}</ol>
          </article>
        `;
      })
      .join("");
  }

  function renderModuleCard(module, sectionLabel, appState) {
    const prerequisiteNames = module.prerequisiteTitles || module.prerequisites;
    const prerequisites = prerequisiteNames.length
      ? `<p class="module-prereq">Prereq: ${prerequisiteNames.map(escapeHtml).join(", ")}</p>`
      : `<p class="module-prereq">Prereq: Start here</p>`;
    const status = (appState && appState.moduleStatus[module.id]) || "not-started";

    return `
      <article class="module-card" data-category="${escapeHtml(module.section)}" data-module-id="${escapeHtml(module.id)}">
        <div class="card-topline">
          <span class="card-kicker">${escapeHtml(sectionLabel)} - ${escapeHtml(module.level)}</span>
          <span class="mini-status status-${escapeHtml(status)}">${escapeHtml(cardStatusLabel(status))}</span>
        </div>
        <h3>${escapeHtml(module.title)}</h3>
        <p>${escapeHtml(module.summary)}</p>
        ${prerequisites}
        <div class="module-actions">
          <button class="complete-button" type="button" data-open-module="${escapeHtml(module.id)}">Open Module</button>
        </div>
      </article>
    `;
  }

  // Sentence case status for the module cards on the section pages.
  function cardStatusLabel(status) {
    const label = STATUS_LABELS[status] || STATUS_LABELS["not-started"];
    return label.charAt(0) + label.slice(1).toLowerCase();
  }

  // One decorative sketch per stage, used as the prominent art on the home
  // stage cards. Picked from a module that sits at the heart of that stage.
  const STAGE_ILLUSTRATIONS = {
    round1: "/assets/module-illustrations/round1/convolutional-neural-networks.svg",
    round2: "/assets/module-illustrations/round2/transformer-fundamentals.svg",
    camp: "/assets/module-illustrations/camp/diffusion-models.svg",
    advanced: "/assets/module-illustrations/advanced/efficient-attention-long-context.svg",
  };

  const STAGE_INDEX = { round1: 1, round2: 2, camp: 3, advanced: 4 };

  function stageStats(sectionId, ordering, moduleLookup, problems) {
    const section = ordering.sectionOrdering[sectionId];
    const moduleIds = section.chapters.flatMap((chapter) => chapter.items);
    const moduleSet = new Set(moduleIds);
    const problemCount = problems
      ? problems.filter((problem) => {
          const module = moduleLookup.get(problem.moduleId);
          return module && moduleSet.has(module.id);
        }).length
      : 0;
    return {
      chapters: section.chapters.length,
      modules: moduleIds.length,
      problems: problemCount,
    };
  }

  function renderStageCards(ordering, moduleLookup, problems, appState, root = "") {
    return ordering.sections
      .map((sectionId) => {
        const section = ordering.sectionOrdering[sectionId];
        const stats = stageStats(sectionId, ordering, moduleLookup, problems);
        const complete = section.chapters
          .flatMap((chapter) => chapter.items)
          .filter((moduleId) => appState.moduleStatus[moduleId] === "complete").length;
        const percent = stats.modules ? Math.round((complete / stats.modules) * 100) : 0;
        const source = STAGE_ILLUSTRATIONS[sectionId] || "";
        const art =
          source && moduleLookup.size
            ? `<span class="stage-card-art" aria-hidden="true"><img src="${escapeHtml(resolveAssetPath(source, root))}${ASSET_VERSION ? `?v=${escapeHtml(ASSET_VERSION)}` : ""}" alt="" loading="lazy" decoding="async" onerror="this.parentNode.remove()" /></span>`
            : "";
        return `
          <a class="stage-card" data-stage="${escapeHtml(sectionId)}" href="${escapeHtml(root + section.route)}">
            ${art}
            <span class="stage-card-body">
              <span class="stage-card-top">
                <span class="stage-card-kicker">Stage ${escapeHtml(String(STAGE_INDEX[sectionId] || ""))}</span>
                <span class="stage-card-progress">${escapeHtml(percent)}%</span>
              </span>
              <strong class="stage-card-title">${escapeHtml(section.label)}</strong>
              <em class="stage-card-desc">${escapeHtml(section.description)}</em>
              <span class="stage-card-facts">
                <span class="stage-card-fact"><b>${escapeHtml(stats.modules)}</b> modules</span>
                <span class="stage-card-fact"><b>${escapeHtml(stats.chapters)}</b> chapters</span>
                <span class="stage-card-fact"><b>${escapeHtml(stats.problems)}</b> problems</span>
              </span>
              <span class="stage-card-cta">Browse Modules <span aria-hidden="true">→</span></span>
            </span>
          </a>
        `;
      })
      .join("");
  }

  function renderGlanceTable(ordering, moduleLookup, problems, appState, root = "") {
    const rows = ordering.sections
      .map((sectionId) => {
        const section = ordering.sectionOrdering[sectionId];
        const stats = stageStats(sectionId, ordering, moduleLookup, problems);
        const moduleIds = section.chapters.flatMap((chapter) => chapter.items);
        const complete = moduleIds.filter(
          (moduleId) => appState.moduleStatus[moduleId] === "complete",
        ).length;
        const percent = stats.modules ? Math.round((complete / stats.modules) * 100) : 0;
        return `
          <tr>
            <th scope="row"><a href="${escapeHtml(root + section.route)}">${escapeHtml(section.label)}</a></th>
            <td data-label="Chapters">${escapeHtml(stats.chapters)}</td>
            <td data-label="Modules">${escapeHtml(stats.modules)}</td>
            <td data-label="Practice problems">${escapeHtml(stats.problems)}</td>
            <td data-label="Complete">${escapeHtml(complete)}</td>
            <td data-label="Progress">
              <span class="glance-bar" role="img" aria-label="${escapeHtml(percent)} percent complete"><i style="width: ${escapeHtml(percent)}%"></i></span>
              <span class="glance-pct">${escapeHtml(percent)}%</span>
            </td>
          </tr>
        `;
      })
      .join("");

    const totals = ordering.sections.reduce(
      (acc, sectionId) => {
        const stats = stageStats(sectionId, ordering, moduleLookup, problems);
        acc.chapters += stats.chapters;
        acc.modules += stats.modules;
        acc.problems += stats.problems;
        return acc;
      },
      { chapters: 0, modules: 0, problems: 0 },
    );
    const allModules = ordering.sections.flatMap((sectionId) =>
      ordering.sectionOrdering[sectionId].chapters.flatMap((chapter) => chapter.items),
    );
    const totalComplete = allModules.filter(
      (moduleId) => appState.moduleStatus[moduleId] === "complete",
    ).length;
    const totalPercent = totals.modules ? Math.round((totalComplete / totals.modules) * 100) : 0;

    return `
      <table class="glance-table">
        <caption>Every stage of the guide at a glance</caption>
        <thead>
          <tr>
            <th scope="col">Stage</th>
            <th scope="col">Chapters</th>
            <th scope="col">Modules</th>
            <th scope="col">Practice Problems</th>
            <th scope="col">Complete</th>
            <th scope="col">Progress</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
          <tr class="glance-total">
            <th scope="row">All Stages</th>
            <td data-label="Chapters">${escapeHtml(totals.chapters)}</td>
            <td data-label="Modules">${escapeHtml(totals.modules)}</td>
            <td data-label="Practice problems">${escapeHtml(totals.problems)}</td>
            <td data-label="Complete">${escapeHtml(totalComplete)}</td>
            <td data-label="Progress">
              <span class="glance-bar" role="img" aria-label="${escapeHtml(totalPercent)} percent complete"><i style="width: ${escapeHtml(totalPercent)}%"></i></span>
              <span class="glance-pct">${escapeHtml(totalPercent)}%</span>
            </td>
          </tr>
        </tbody>
      </table>
    `;
  }

  function dateKey(date) {
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${date.getFullYear()}-${month}-${day}`;
  }

  function buildActivity(appState) {
    const activityLog = Array.isArray(appState.activityLog) ? appState.activityLog : [];
    const activityByDay = {};
    const now = new Date();
    activityLog.forEach((entry) => {
      if (!entry || entry.type !== "problem" || entry.status !== "complete") return;
      if (typeof entry.id !== "string" || !entry.id.trim()) return;
      if (typeof entry.at !== "string" && typeof entry.at !== "number") return;
      const date = new Date(entry.at);
      if (Number.isNaN(date.getTime()) || date > now) return;
      const key = dateKey(date);
      if (!activityByDay[key]) activityByDay[key] = new Set();
      activityByDay[key].add(entry.id);
    });

    const today = new Date(now);
    today.setHours(0, 0, 0, 0);
    const start = new Date(today);
    start.setDate(today.getDate() - 363);
    const days = Array.from({ length: 364 }, (_, index) => {
      const date = new Date(start);
      date.setDate(start.getDate() + index);
      return date;
    });
    const months = days
      .map((date, index) => ({ date, index, column: Math.floor(index / 7) + 1 }))
      .filter(({ date, index }) => index === 0 || date.getDate() === 1);
    const monthLabels = months
      .map(({ date, column }, index) => {
        const span = (months[index + 1]?.column || 53) - column;
        if (span < 2 || (index === 0 && span < 3)) return "";
        return `<span style="grid-column: ${column} / span ${span}">${date.toLocaleString("en-US", { month: "short" })}</span>`;
      })
      .join("");
    const squares = days
      .map((date) => {
        const key = dateKey(date);
        const count = activityByDay[key]?.size || 0;
        const level = Math.min(count, 4);
        const label = `${escapeHtml(key)}: ${escapeHtml(count)} problem${count === 1 ? "" : "s"} solved`;
        return `<span class="activity-square level-${level}" title="${label}"><span class="sr-only">${label}</span></span>`;
      })
      .join("");

    let streak = 0;
    const cursor = new Date(today);
    const completedToday = activityByDay[dateKey(today)]?.size || 0;
    if (!completedToday) cursor.setDate(cursor.getDate() - 1);
    while (activityByDay[dateKey(cursor)]) {
      streak += 1;
      cursor.setDate(cursor.getDate() - 1);
    }

    return { monthLabels, squares, streak, completedToday };
  }

  function kpiTile(label, value, total, percent, note = "") {
    return `
      <article class="dashboard-card kpi-tile">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}${total ? `<small>/ ${escapeHtml(total)}</small>` : ""}</strong>
        ${percent === null ? "" : `<div class="meter-track"><span class="meter-fill" style="width: ${escapeHtml(percent)}%"></span></div>`}
        ${note ? `<p>${escapeHtml(note)}</p>` : ""}
      </article>
    `;
  }

  function renderDashboard(stats, nextModules, appState, context) {
    const lastModule = context.moduleLookup.get(appState.lastModuleId);
    const continueModule = lastModule || nextModules[0] || context.moduleOrder[0];
    const activity = buildActivity(appState);
    const modulePercent = stats.modules.total ? Math.round((stats.modules.complete / stats.modules.total) * 100) : 0;
    const problemPercent = stats.problems.total ? Math.round((stats.problems.complete / stats.problems.total) * 100) : 0;

    return `
      <section class="dash-hero">
        <div>
          <h1>Welcome back</h1>
          ${continueModule ? `<p>Your last opened module is "${escapeHtml(continueModule.title)}".</p>` : ""}
        </div>
        ${continueModule ? `
          <a class="dashboard-continue" href="${escapeHtml(`${context.root}module/?id=${encodeURIComponent(continueModule.id)}`)}" data-open-module="${escapeHtml(continueModule.id)}">
            Continue learning
          </a>
        ` : ""}
      </section>
      <div class="kpi-grid">
        ${kpiTile("Modules completed", stats.modules.complete, stats.modules.total, modulePercent)}
        ${kpiTile("Problems completed", stats.problems.complete, stats.problems.total, problemPercent)}
        ${kpiTile("Day streak", activity.streak, "", null, "")}
      </div>
      <article class="dashboard-card activity-card">
        <div class="activity-head">
          <h2>Activity</h2>
          <div class="activity-legend" aria-hidden="true">
            <span>Less</span>
            <i class="activity-square level-0"></i>
            <i class="activity-square level-1"></i>
            <i class="activity-square level-2"></i>
            <i class="activity-square level-3"></i>
            <i class="activity-square level-4"></i>
            <span>More</span>
          </div>
        </div>
        <div class="activity-scroll" tabindex="0" role="region" aria-label="Problems solved over the last 52 weeks, scroll to explore">
          <div class="activity-months">${activity.monthLabels}</div>
          <div class="activity-heatmap" role="group" aria-label="Problems solved by day over the last 52 weeks. ${activity.completedToday} solved today. Current streak: ${activity.streak} days.">${activity.squares}</div>
        </div>
      </article>
    `;
  }

  function renderSectionBody(body) {
    const paragraphs = Array.isArray(body) ? body : [body];
    return paragraphs
      .filter((paragraph) => paragraph)
      .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
      .join("");
  }

  function renderSectionList(items, className, ordered = false, label = "") {
    if (!Array.isArray(items) || !items.length) return "";
    const tag = ordered ? "ol" : "ul";
    const list = `
      <${tag} class="${className}">
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </${tag}>
    `;
    if (!label) return list;
    return `
      <div class="reader-subblock">
        <h3>${escapeHtml(label)}</h3>
        ${list}
      </div>
    `;
  }

  const PY_KEYWORDS = new Set(
    ("False None True and as assert async await break case class continue def del elif else except " +
      "finally for from global if import in is lambda match nonlocal not or pass raise return try while with yield").split(" ")
  );
  const PY_BUILTINS = new Set(
    ("abs all any bin bool bytes callable chr classmethod complex dict dir divmod enumerate eval filter float format " +
      "frozenset getattr hasattr hash hex id input int isinstance issubclass iter len list map max min next object oct " +
      "open ord pow print property range repr reversed round set setattr slice sorted staticmethod str sum super tuple " +
      "type vars zip self cls __init__ __name__ Exception ValueError TypeError KeyError IndexError AttributeError " +
      "ZeroDivisionError RuntimeError StopIteration NotImplementedError AssertionError").split(" ")
  );

  const PY_TOKEN_RE = new RegExp(
    [
      "(#[^\\n]*)",                                                            // 1 comment
      "((?:[rRbBuUfF]{1,2})?(?:\"\"\"[\\s\\S]*?(?:\"\"\"|$)|'''[\\s\\S]*?(?:'''|$)|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*'))", // 2 string
      "(@[A-Za-z_][\\w.]*)",                                                   // 3 decorator
      "(\\d[\\d_]*(?:\\.[\\d_]*)?(?:[eE][+-]?\\d+)?[jJ]?|\\.\\d[\\d_]*(?:[eE][+-]?\\d+)?[jJ]?|0[xX][\\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+)", // 4 number
      "([A-Za-z_]\\w*)",                                                       // 5 identifier
      "([+\\-*/%=<>!&|^~:@]+)",                                                // 6 operator
    ].join("|"),
    "g"
  );

  function token(kind, text) {
    return text.split("\n").map(line => `<span class="tok-${kind}">${escapeHtml(line)}</span>`).join("\n");
  }

  function highlightPython(code) {
    let html = "";
    let last = 0;
    let prevWord = "";
    PY_TOKEN_RE.lastIndex = 0;
    let match;
    while ((match = PY_TOKEN_RE.exec(code))) {
      if (match.index > last) {
        const gap = code.slice(last, match.index);
        html += escapeHtml(gap);
        if (gap.trim()) prevWord = "";
      }
      const [text, comment, str, dec, num, word] = match;
      if (comment !== undefined) {
        html += token("com", text);
      } else if (str !== undefined) {
        html += token("str", text);
      } else if (dec !== undefined) {
        html += token("dec", text);
      } else if (num !== undefined) {
        html += token("num", text);
      } else if (word !== undefined) {
        if (PY_KEYWORDS.has(word)) html += token("kw", word);
        else if (prevWord === "def" || prevWord === "class") html += token("def", word);
        else if (PY_BUILTINS.has(word)) html += token("blt", word);
        else if (code[PY_TOKEN_RE.lastIndex] === "(") html += token("fn", word);
        else html += escapeHtml(word);
        prevWord = word;
      } else {
        html += token("op", text);
        prevWord = "";
      }
      if (word === undefined) prevWord = "";
      last = PY_TOKEN_RE.lastIndex;
    }
    html += escapeHtml(code.slice(last));
    return html;
  }


  function renderCodeBlock(code) {
    if (!code) return "";
    const language = typeof code === "object" ? code.language || "pseudocode" : "pseudocode";
    const rawLines = typeof code === "object" && Array.isArray(code.lines) ? code.lines : String(code.body || code).split("\n");
    const coloredLines = /^(python|py|python3)$/i.test(language) ? highlightPython(rawLines.join("\n")).split("\n") : rawLines.map(line => escapeHtml(line));
    return `
      <div class="reader-code-block">
        <div class="reader-code-title">
          <span>${escapeHtml(language)}</span>
          <button class="reader-code-copy" type="button" data-copy-code>Copy</button>
        </div>
        <pre>${rawLines
          .map((line, index) => `<span><b>${escapeHtml(index + 1)}</b><code>${coloredLines[index] || " "}</code></span>`)
          .join("")}</pre>
      </div>
    `;
  }

  function renderMathBlock(math) {
    if (!Array.isArray(math) || !math.length) return "";
    const blocks = math
      .map((entry) => {
        const note = entry && entry.note ? `<p class="reader-math-note">${escapeHtml(entry.note)}</p>` : "";
        const tex = entry && entry.tex
          ? `<div class="reader-math-display" data-tex="${escapeHtml(entry.tex)}"><code>${escapeHtml(entry.tex)}</code></div>`
          : "";
        return note + tex;
      })
      .join("");
    return `<div class="reader-math">${blocks}</div>`;
  }

  function renderFocusProblem(problemId, problemLookup) {
    if (!problemId) return "";
    const problem = problemLookup.get(problemId);
    if (!problem) return "";
    return `
      <button class="reader-focus-card" type="button" data-focus-problem="${escapeHtml(problem.id)}">
        <span class="reader-focus-copy">
          <small>Focus problem · ${escapeHtml(problem.source)} · ${escapeHtml(problem.difficulty)}</small>
          <strong>${escapeHtml(problem.title)}</strong>
          <em>${escapeHtml(problem.prompt)}</em>
        </span>
        <span class="reader-focus-open" aria-hidden="true">Open →</span>
      </button>
    `;
  }

  function renderSectionLinks(links, root) {
    if (!Array.isArray(links) || !links.length) return "";
    const items = links
      .map((link) => {
        const isModule = Boolean(link.module);
        const href = isModule ? `${root || ""}module/?id=${encodeURIComponent(link.module)}` : link.href;
        const attrs = isModule
          ? `href="${escapeHtml(href)}" data-open-module="${escapeHtml(link.module)}"`
          : `href="${escapeHtml(href)}" target="_blank" rel="noopener"`;
        return `<li><a ${attrs}>${escapeHtml(link.label)}</a>${link.note ? ` <small>${escapeHtml(link.note)}</small>` : ""}</li>`;
      })
      .join("");
    return `<ul class="reader-link-list">${items}</ul>`;
  }

  // Mirrors DIFFICULTY_ORDER in app.js. Anything unrecognised sorts last rather
  // than jumping to the front on an index of -1.
  const DIFFICULTY_ORDER = ["easy", "medium", "hard", "very hard", "mixed"];
  function difficultyRank(difficulty) {
    const index = DIFFICULTY_ORDER.indexOf(String(difficulty || "").toLowerCase());
    return index === -1 ? DIFFICULTY_ORDER.length : index;
  }

  // Compact problem table shared by the lesson sections and a module's problem
  // list. Title, type and difficulty in a box that scrolls sideways on a narrow
  // screen instead of stretching the page.
  // `withStatus` adds the status circle as a last column, for the module's
  // own problem list.
  function problemTable(rows, root, withStatus) {
    return `
      <div class="mptable-scroll">
        <table class="mptable">
          <caption class="sr-only">Problems</caption>
          <thead>
            <tr>
              <th scope="col" class="mptable-th">Problem</th>
              <th scope="col" class="mptable-th mptable-th-type">Type</th>
              <th scope="col" class="mptable-th mptable-th-difficulty">Difficulty</th>
              ${withStatus ? '<th scope="col" class="mptable-th mptable-th-status">Status</th>' : ""}
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // `options.appState` draws the status column, and `options.fromModule`
  // makes the problem page step through this module's problems only.
  function problemRow(problem, root, options = {}) {
    const { href, external } = problemHref(problem, root, options.fromModule);
    const difficultyClass = `difficulty-${escapeHtml(String(problem.difficulty).toLowerCase().replace(/\s+/g, "-"))}`;
    const status = options.appState
      ? `<td class="mptable-cell mptable-cell-status">${statusCircle(
          problem.id,
          problem.title,
          (options.appState.problemStatus || {})[problem.id],
        )}</td>`
      : "";
    return `
      <tr class="mptable-row">
        <td class="mptable-cell mptable-cell-title">
          <a class="mptable-title" href="${escapeHtml(href)}"${external ? ' target="_blank" rel="noopener"' : ""}>
            ${escapeHtml(problem.title)}
          </a>
        </td>
        <td class="mptable-cell mptable-cell-type">${typePill(problem)}</td>
        <td class="mptable-cell mptable-cell-difficulty">
          <span class="problem-chip ${difficultyClass}">${escapeHtml(problem.difficulty)}</span>
        </td>
        ${status}
      </tr>`;
  }

  function renderSimilarProblems(problemIds, problemLookup, root, appState) {
    if (!Array.isArray(problemIds) || !problemIds.length) return "";
    const rows = problemIds
      .map((problemId) => problemLookup.get(problemId))
      .filter(Boolean)
      // Easiest first. Array.prototype.sort is stable, so problems sharing a
      // difficulty keep the order the lesson author listed them in.
      .sort((a, b) => difficultyRank(a.difficulty) - difficultyRank(b.difficulty))
      .map((problem) => problemRow(problem, root, { appState, fromModule: true }))
      .join("");
    return problemTable(rows, root, Boolean(appState));
  }

  // The module's slice of the problemset as a compact table, with the same
  // status circle as the problem bank.
  function renderModuleProblems(moduleId, problems, root, appState) {
    const mine = problems
      .filter((problem) => problem.moduleId === moduleId)
      .sort((a, b) => {
        const setA = a.tags.includes("Problem Set") ? 0 : 1;
        const setB = b.tags.includes("Problem Set") ? 0 : 1;
        return setA - setB || difficultyRank(a.difficulty) - difficultyRank(b.difficulty);
      });
    if (!mine.length) {
      return `
        <p class="reader-problems-empty">
          No problems for this module yet.
          <a href="${escapeHtml(`${root || ""}problems/`)}">Browse the full problemset →</a>
        </p>
      `;
    }
    return problemTable(
      mine.map((problem) => problemRow(problem, root, { appState, fromModule: true })).join(""),
      root,
      Boolean(appState),
    );
  }

  function renderLessonSection(section, problemLookup, root, appState) {
    const anchor = `lesson-${escapeHtml(section.id)}`;
    const headingPrefix = `lesson-${slugify(section.id)}`;
    // The lead-in carries no title: its prose runs straight under the module
    // heading rather than under a heading of its own.
    const label = section.title
      ? `<h2 class="reader-section-label">${escapeHtml(section.title)}<a class="reader-anchor" href="#${anchor}" aria-label="Link to ${escapeHtml(section.title)}">#</a></h2>`
      : "";
    return `
      <section class="reader-section reader-notebook-cell" id="${anchor}">
        ${label}
        ${renderBlocks(section.blocks, root, { headingPrefix, problemLookup, appState })}
        ${renderSectionBody(section.body)}
        ${renderSectionList(section.keyIdeas, "reader-key-list", false, "Key ideas")}
        ${renderSectionList(section.steps, "reader-steps", true, "Steps")}
        ${section.note ? `<div class="reader-callout">${escapeHtml(section.note)}</div>` : ""}
        ${renderSectionLinks(section.links, root)}
        ${renderMathBlock(section.math)}
        ${renderCodeBlock(section.code)}
        ${renderFocusProblem(section.focusProblem, problemLookup)}
        ${renderSimilarProblems(section.problems, problemLookup, root, appState)}
      </section>
    `;
  }

  function renderModuleReader(module, lesson, context) {
    const problemLookup = new Map(context.problems.map((problem) => [problem.id, problem]));
    const section = context.ordering.sectionOrdering[module.section];
    const stageModules = section.chapters.flatMap((chapter) => chapter.items);
    const moduleIndex = stageModules.indexOf(module.id);
    const currentStatus = context.appState.moduleStatus[module.id] || "not-started";
    const moduleStatusControl = `
        <label class="reader-head-status">
          <span class="sr-only">Module Status</span>
          <select class="reader-status-select" data-module-status="${escapeHtml(module.id)}" aria-label="Module status for ${escapeHtml(module.title)}">
            ${statusOptions(currentStatus)}
          </select>
        </label>
      `;
    // Prerequisites are hoisted out of the section flow into a box directly
    // under the title, so the reader sees what it needs before the prose
    // starts rather than as body section two.
    const bodySections = lesson.sections.filter((entry) => entry.id !== "prerequisites" && entry.id !== "resources");
    // Resources move from the end of the lesson to a card at the top, as on
    // usaco.guide, so a reader sees what to read alongside before starting.
    const resourceSection = lesson.sections.find((entry) => entry.id === "resources");
    const resourceList = resourceSection
      ? resourceSection.blocks.flatMap((block) => (block && Array.isArray(block.resources) ? block.resources : []))
      : [];
    const resourcesCard = renderModuleResources(resourceList, context.appState, context.root);
    const sections = `${bodySections
      .map((section) => renderLessonSection(section, problemLookup, context.root, context.appState))
      .join("")}
      <section class="reader-section reader-notebook-cell reader-problems-section" id="lesson-problems">
        <h2>Problems<a class="reader-anchor" href="#lesson-problems" aria-label="Link to Problems">#</a></h2>
        ${renderModuleProblems(module.id, context.problems, context.root, context.appState)}
      </section>`;
    // Built from the resolved prerequisite ids rather than the Markdown blocks,
    // so the box is always a row of titled links to real modules.
    const prerequisiteLinks = (module.prerequisites || [])
      .map((id) => {
        const target = context.moduleLookup.get(id);
        if (!target) return "";
        const href = `${context.root || ""}module/?id=${encodeURIComponent(id)}`;
        return `<li><a href="${escapeHtml(href)}" data-open-module="${escapeHtml(id)}">${escapeHtml(target.title)}</a></li>`;
      })
      .filter(Boolean)
      .join("");
    // The row above the title: level on the left where usaco.guide puts its
    // frequency dots, solved-count meter on the right.
    const moduleProblems = context.problems.filter((problem) => problem.moduleId === module.id);
    const solvedCount = moduleProblems.filter(
      (problem) => context.appState && context.appState.problemStatus[problem.id] === "complete",
    ).length;
    const solvedPct = moduleProblems.length ? Math.round((solvedCount / moduleProblems.length) * 100) : 0;
    const progressRow = `
      <div class="reader-progress-row">
        <span class="reader-level">${escapeHtml(module.level)}</span>
        ${
          moduleProblems.length
            ? `<span class="reader-progress">
                <span class="reader-progress-track"><span class="reader-progress-fill" style="width: ${solvedPct}%"></span></span>
                <span class="reader-progress-count${solvedCount === moduleProblems.length ? " is-done" : solvedCount ? " is-started" : ""}" title="Problems solved in this module"><b>${escapeHtml(solvedCount)}</b>/${escapeHtml(moduleProblems.length)} <span>solved</span></span>
              </span>`
            : ""
        }
      </div>
    `;

    const prerequisites = prerequisiteLinks
      ? `
        <section class="reader-infocard" aria-labelledby="reader-prereqs-title">
          <h2 id="reader-prereqs-title">Prerequisites</h2>
          <ul class="reader-prereq-list">${prerequisiteLinks}</ul>
        </section>
      `
      : "";
    const logoMark = `
      <svg class="reader-logo-mark" viewBox="0 0 64 64" aria-hidden="true">
        <circle cx="32" cy="32" r="29" fill="#f5f0e8" stroke="#b84e30" stroke-width="3"/>
        <g stroke="#d4805c" stroke-width="3" stroke-linecap="round">
          <path d="M25 47 L16 30"/><path d="M25 47 L32 30"/><path d="M25 47 L48 30"/>
          <path d="M39 47 L16 30"/><path d="M39 47 L32 30"/><path d="M39 47 L48 30"/>
          <path d="M16 30 L32 16"/><path d="M32 30 L32 16"/><path d="M48 30 L32 16"/>
        </g>
        <g fill="#f5f0e8" stroke="#a8442a" stroke-width="2">
          <circle cx="25" cy="47" r="3.4"/><circle cx="39" cy="47" r="3.4"/><circle cx="16" cy="30" r="3.4"/>
          <circle cx="32" cy="30" r="3.4"/><circle cx="48" cy="30" r="3.4"/><circle cx="32" cy="16" r="3.4"/>
        </g>
      </svg>
    `;
    const sidebarChapters = section.chapters
      .map(
        (chapter) => `
          <details class="reader-chapter ${chapter.items.includes(module.id) ? "active" : ""}" open>
            <summary>${escapeHtml(chapter.name)}</summary>
            <ol>
              ${chapter.items
                .map((moduleId) => {
                  const item = context.moduleLookup.get(moduleId);
                  const active = moduleId === module.id;
                  return `
                    <li class="${active ? "active" : ""}">
                      <a class="${active ? "active" : ""}" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(moduleId)}`)}" data-open-module="${escapeHtml(moduleId)}">
                        <span class="reader-nav-dot" aria-hidden="true"></span>
                        <span class="reader-nav-title">${escapeHtml(item ? item.title : moduleId)}</span>
                      </a>
                    </li>
                  `;
                })
                .join("")}
            </ol>
          </details>
        `,
      )
      .join("");
    const stageMenu = context.ordering.sections
      .map((stageId) => {
        const stage = context.ordering.sectionOrdering[stageId];
        if (!stage) return "";
        const firstChapter = stage.chapters.find((chapter) => chapter.items.length);
        const firstModule = firstChapter ? firstChapter.items[0] : "";
        const active = stageId === module.section;
        return `
          <a class="${active ? "active" : ""}" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(firstModule)}`)}" data-open-module="${escapeHtml(firstModule)}">
            ${escapeHtml(stage.label)}
          </a>
        `;
      })
      .join("");
    // Prev · breadcrumb · Next, sitting above the article the way usaco.guide
    // frames a module. The same two targets are repeated as cards at the foot.
    const stageRoute = `${context.root || ""}${section.route || ""}`;
    const crumbSep = `<svg class="reader-crumb-sep" viewBox="0 0 20 20" aria-hidden="true"><path d="M7.05 4.55 12.5 10l-5.45 5.45" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const topBar = `
      <div class="reader-topbar">
        ${
          context.prev
            ? `<a class="reader-step" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(context.prev.id)}`)}" data-open-module="${escapeHtml(context.prev.id)}">
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12 4.5 6.5 10 12 15.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
                Prev
              </a>`
            : `<span class="reader-step is-disabled" aria-hidden="true"></span>`
        }
        <nav class="reader-crumbs" aria-label="Breadcrumb">
          <a href="${escapeHtml(`${context.root || ""}`)}">Home</a>
          ${crumbSep}
          <a href="${escapeHtml(stageRoute)}">${escapeHtml(section.label)}</a>
          ${crumbSep}
          <span aria-current="page">${escapeHtml(module.title)}</span>
        </nav>
        ${
          context.next
            ? `<a class="reader-step" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(context.next.id)}`)}" data-open-module="${escapeHtml(context.next.id)}">
                Next
                <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M8 4.5 13.5 10 8 15.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
              </a>`
            : `<span class="reader-step is-disabled" aria-hidden="true"></span>`
        }
      </div>
    `;

    const pageNav = `
      <nav class="reader-pagenav" aria-label="Module navigation">
        ${
          context.prev
            ? `<a class="reader-pagenav-card" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(context.prev.id)}`)}" data-open-module="${escapeHtml(context.prev.id)}">
                <small>← Previous</small>
                <strong>${escapeHtml(context.prev.title)}</strong>
              </a>`
            : `<span class="reader-pagenav-card is-disabled" aria-disabled="true">
                <small>← Previous</small>
                <strong>Start of ${escapeHtml(section.label)}</strong>
              </span>`
        }
        ${
          context.next
            ? `<a class="reader-pagenav-card next" href="${escapeHtml(`${context.root || ""}module/?id=${encodeURIComponent(context.next.id)}`)}" data-open-module="${escapeHtml(context.next.id)}">
                <small>Next →</small>
                <strong>${escapeHtml(context.next.title)}</strong>
              </a>`
            : `<span class="reader-pagenav-card next is-disabled" aria-disabled="true">
                <small>Next →</small>
                <strong>End of ${escapeHtml(section.label)}</strong>
              </span>`
        }
      </nav>
    `;

    return `
      <aside class="reader-sidenav" aria-label="${escapeHtml(section.label)} modules">
        <div class="reader-nav-top">
          <a class="reader-logo" href="${escapeHtml(`${context.root || "../"}dashboard/`)}">
            ${logoMark}
            <span class="reader-logo-text"><strong>AIO Initiative</strong><small>USAAIO Guide</small></span>
          </a>
          <button class="reader-nav-toggle" type="button" data-reader-nav aria-label="Collapse sidebar" title="Collapse sidebar">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
        </div>
        <details class="reader-stage">
          <summary>${escapeHtml(section.label)}</summary>
          <div class="reader-stage-menu">${stageMenu}</div>
        </details>
        <div class="reader-nav-scroll">
          ${sidebarChapters}
        </div>
        <div class="reader-nav-footer">
          <a href="${escapeHtml(`${context.root || ""}problems/`)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
            Problems
          </a>
          <a href="${escapeHtml(`${context.root || ""}glossary/`)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5v-15ZM5 19.5A1.5 1.5 0 0 0 6.5 21H19v-3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Glossary
          </a>
          <a href="${escapeHtml(`${context.root || ""}contact/`)}">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Contact us
          </a>
        </div>
      </aside>
      <button class="reader-nav-fab" type="button" data-reader-nav aria-label="Open sidebar" title="Open sidebar">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>
      <article class="reader-article reader-notebook-article">
        ${topBar}
        ${progressRow}
        <header class="reader-head">
          <div class="reader-head-main">
            <h1>${escapeHtml(module.title)}</h1>
          </div>
        </header>
        <div class="reader-meta-row">
          <p class="reader-meta">
            <span>${escapeHtml(section.label)} · module ${escapeHtml(moduleIndex + 1)} of ${escapeHtml(stageModules.length)}</span>
            ${frequencyBadge(module)}
          </p>
          <div class="reader-head-actions">
            <button class="definition-bank-button" type="button" data-definition-bank aria-haspopup="dialog">Definition Bank</button>
            ${moduleStatusControl}
          </div>
        </div>
        <p class="reader-lede">${escapeHtml(module.summary)}</p>
        ${prerequisites}
        ${resourcesCard}
        ${sections}
        ${pageNav}
        <aside class="reader-help">
          <p><strong>Found an error?</strong> Send a correction or suggestion for this module.</p>
          <a class="reader-help-link" href="${escapeHtml(`${context.root || ""}contact/`)}">Report It</a>
        </aside>
      </article>
    `;
  }


  function plannerKey(item) {
    return `${item.type}:${item.id}`;
  }

  function plannerBlock(item, context) {
    if (item.type === "module") {
      const module = context.moduleLookup.get(item.id);
      if (!module) return null;
      const section = context.ordering.sectionOrdering[module.section];
      return {
        title: module.title,
        meta: `${section ? section.label : module.section} · ${module.level}`,
        href: `${context.root || ""}module/?id=${encodeURIComponent(module.id)}`,
        external: false,
      };
    }
    const problem = context.problemLookup.get(item.id);
    if (!problem) return null;
    const isInternalLink = Boolean(problem.url) && problem.url.startsWith("/");
    return {
      title: problem.title,
      meta: `${problem.difficulty} · ${problem.source}`,
      href: isInternalLink ? `${context.root || ""}${problem.url.slice(1)}` : problem.url,
      external: Boolean(problem.url) && !isInternalLink,
    };
  }

  function renderPlannerLibrary(tab, query, plan, context) {
    const inPlan = new Set(plan.map(plannerKey));
    const normalizedQuery = (query || "").trim().toLowerCase();
    const source =
      tab === "problem"
        ? context.problems.map((problem) => ({ type: "problem", id: problem.id }))
        : context.moduleOrder.map((module) => ({ type: "module", id: module.id }));
    const blocks = source
      .map((item) => ({ item, block: plannerBlock(item, context) }))
      .filter(({ block }) => {
        if (!block) return false;
        if (!normalizedQuery) return true;
        return `${block.title} ${block.meta}`.toLowerCase().includes(normalizedQuery);
      })
      .map(({ item, block }) => {
        const added = inPlan.has(plannerKey(item));
        return `
          <div class="plan-block plan-type-${escapeHtml(item.type)} ${added ? "in-plan" : ""}" draggable="${added ? "false" : "true"}"
            data-plan-source data-plan-type="${escapeHtml(item.type)}" data-plan-id="${escapeHtml(item.id)}">
            <span class="plan-grip" aria-hidden="true"></span>
            <span class="plan-block-copy">
              <strong>${escapeHtml(block.title)}</strong>
              <small>${escapeHtml(block.meta)}</small>
            </span>
            <button class="plan-add" type="button" data-plan-add aria-label="${added ? "Already in plan" : `Add ${escapeHtml(block.title)} to plan`}" ${added ? "disabled" : ""}>${added ? "✓" : "+"}</button>
          </div>
        `;
      })
      .join("");
    return blocks || `<p class="empty-state">No blocks match that search.</p>`;
  }

  function renderPlannerBoard(plan, context) {
    if (!plan.length) {
      return `
        <li class="planner-empty">
          <strong>Your plan is empty.</strong>
          <span>Drag blocks here from the library, or use the + buttons.</span>
        </li>
      `;
    }
    return plan
      .map((item, index) => {
        const block = plannerBlock(item, context);
        if (!block) return "";
        return `
          <li class="plan-block plan-block-board plan-type-${escapeHtml(item.type)}" draggable="true" data-plan-index="${index}">
            <span class="plan-order" aria-hidden="true">${index + 1}</span>
            <span class="plan-block-copy">
              <strong>${escapeHtml(block.title)}</strong>
              <small>${escapeHtml(item.type === "module" ? "Topic" : "Problem")} · ${escapeHtml(block.meta)}</small>
            </span>
            <span class="plan-actions">
              <button class="plan-tool" type="button" data-plan-move="up" data-plan-target="${index}" aria-label="Move up" ${index === 0 ? "disabled" : ""}>↑</button>
              <button class="plan-tool" type="button" data-plan-move="down" data-plan-target="${index}" aria-label="Move down" ${index === plan.length - 1 ? "disabled" : ""}>↓</button>
              <a class="plan-tool plan-open" href="${escapeHtml(block.href)}" ${block.external ? 'target="_blank" rel="noreferrer"' : ""} aria-label="Open ${escapeHtml(block.title)}">Open</a>
              <button class="plan-tool plan-remove" type="button" data-plan-remove="${index}" aria-label="Remove ${escapeHtml(block.title)} from plan">×</button>
            </span>
          </li>
        `;
      })
      .join("");
  }

  function renderPager(currentPage, totalPages) {
    if (totalPages <= 1) return "";
    const windowSize = 2;
    const pages = [];
    for (let page = 1; page <= totalPages; page += 1) {
      if (page === 1 || page === totalPages || Math.abs(page - currentPage) <= windowSize) {
        pages.push(page);
      } else if (pages[pages.length - 1] !== "gap") {
        pages.push("gap");
      }
    }
    const numbers = pages
      .map((page) =>
        page === "gap"
          ? `<span class="pager-gap" aria-hidden="true">…</span>`
          : `<button class="pager-button ${page === currentPage ? "active" : ""}" type="button" data-problem-page="${page}" ${
              page === currentPage ? 'aria-current="page"' : ""
            }>${page}</button>`,
      )
      .join("");
    return `
      <button class="pager-button pager-step" type="button" data-problem-page="${currentPage - 1}" ${currentPage === 1 ? "disabled" : ""} aria-label="Previous page">←</button>
      ${numbers}
      <button class="pager-button pager-step" type="button" data-problem-page="${currentPage + 1}" ${currentPage === totalPages ? "disabled" : ""} aria-label="Next page">→</button>
    `;
  }

  // The three problem UIs. Coding problems open in the code editor, a theory
  // problem that is a single multiple-choice question is MCQ, and every other
  // theory problem is a Text answer. The index table and the module lists all
  // show the same label for a problem.
  function problemType(problem) {
    if (problem.url && problem.url.startsWith("/editor/")) return "Code";
    return problem.tags.includes("Multiple choice") ? "MCQ" : "Text";
  }

  const TYPE_CLASS = { Code: "ptype-code", MCQ: "ptype-mcq", Text: "ptype-text" };

  // One small mark per type, so the tag reads at a glance and never looks
  // like a difficulty chip: angle brackets, a ticked list, lines of writing.
  const TYPE_ICON = {
    Code: '<path d="m6 5-3.5 3L6 11M10 5l3.5 3L10 11" />',
    MCQ: '<circle cx="3.6" cy="4.5" r="1.6" /><path d="M8 4.5h5.5M8 11.5h5.5" /><path d="m2.2 11.6 1.1 1.1 2-2.4" />',
    Text: '<path d="M2.5 4h11M2.5 8h11M2.5 12h6.5" />',
  };

  function typePill(problem) {
    const type = problemType(problem);
    return `<span class="ptype ${TYPE_CLASS[type]}"><svg class="ptype-icon" viewBox="0 0 16 16" aria-hidden="true">${TYPE_ICON[type]}</svg>${escapeHtml(type)}</span>`;
  }

  // The sortable columns of the problems index table.
  const PTABLE_COLUMNS = [
    { key: "title", label: "Title" },
    { key: "type", label: "Type" },
    { key: "difficulty", label: "Difficulty" },
    { key: "module", label: "Module" },
    { key: "section", label: "Section" },
    { key: "status", label: "Status" },
  ];

  function ptableHead(sort) {
    return PTABLE_COLUMNS.map((column) => {
      const active = Boolean(sort) && sort.key === column.key;
      const direction = active && sort.dir === "desc" ? "descending" : "ascending";
      return `
        <th scope="col" class="ptable-th ptable-th-${column.key}" aria-sort="${active ? direction : "none"}">
          <button class="ptable-sort" type="button" data-sort="${column.key}" title="Sort by ${column.label.toLowerCase()}">
            <span class="ptable-sort-label">${column.label}</span>
            <svg class="ptable-sort-arrow" viewBox="0 0 12 16" aria-hidden="true">
              <path class="ptable-sort-up" d="M2.5 6.2 6 2.7l3.5 3.5" />
              <path class="ptable-sort-down" d="M2.5 9.8 6 13.3l3.5-3.5" />
            </svg>
          </button>
        </th>`;
    }).join("");
  }

  function statusPill(problem, status) {
    return statusCircle(problem.id, problem.title, status);
  }

  function renderProblemCards(problems, moduleLookup, appState, root = "", sectionOrdering = {}, sort = {}) {
    const head = `
      <thead>
        <tr>
          ${ptableHead(sort)}
        </tr>
      </thead>`;
    const rows = problems
      .map((problem) => {
        const module = moduleLookup.get(problem.moduleId);
        const status = appState.problemStatus[problem.id] || "not-started";
        // A leading "/" marks a route inside this site (the editor and theory
        // readers). Those get rewritten relative to `root` so the site still
        // works when served from a subpath; anything else is an external link.
        const isInternalLink = Boolean(problem.url) && problem.url.startsWith("/");
        const href = isInternalLink ? `${root}${problem.url.slice(1)}` : problem.url;
        const difficultyClass = `difficulty-${escapeHtml(String(problem.difficulty).toLowerCase().replace(/\s+/g, "-"))}`;
        const sectionLabel =
          module && sectionOrdering[module.section] ? sectionOrdering[module.section].label : "";
        // A problem with no URL of its own shows its answer inline instead.
        const titleCell = problem.url
          ? `<a
               class="ptable-title-link"
               href="${escapeHtml(href)}"${isInternalLink ? "" : ' target="_blank" rel="noreferrer"'}
             >${escapeHtml(problem.title)}</a>`
          : `<button
               class="ptable-title-link"
               type="button"
               data-reveal-problem="${escapeHtml(problem.id)}"
               aria-label="Show answer for ${escapeHtml(problem.title)}"
             >${escapeHtml(problem.title)}</button>`;
        const moduleCell = module
          ? `<a class="ptable-module-link" href="${root}module/?id=${escapeHtml(problem.moduleId)}">${escapeHtml(module.title)}</a>`
          : `<span class="ptable-module-id">${escapeHtml(problem.moduleId)}</span>`;
        return `
          <tr class="ptable-row" data-problem-row="${escapeHtml(problem.id)}">
            <td class="ptable-cell ptable-cell-title">${titleCell}</td>
            <td class="ptable-cell ptable-cell-type">${typePill(problem)}</td>
            <td class="ptable-cell ptable-cell-difficulty">
              <span class="problem-chip ${difficultyClass}">${escapeHtml(problem.difficulty)}</span>
            </td>
            <td class="ptable-cell ptable-cell-module">${moduleCell}</td>
            <td class="ptable-cell ptable-cell-section">${escapeHtml(sectionLabel)}</td>
            <td class="ptable-cell ptable-cell-status">${statusPill(problem, status)}</td>
          </tr>
          ${
            problem.url
              ? ""
              : `<tr class="ptable-answer-row" id="answer-${escapeHtml(problem.id)}" hidden>
                  <td class="ptable-answer-cell" colspan="6">${escapeHtml(problem.answer)}</td>
                </tr>`
          }`;
      })
      .join("");
    return `
      <table class="ptable">
        <caption class="sr-only">Practice problems</caption>
        ${head}
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderSearchResults(results) {
    // Sentence case on the pill, so a result's type reads as a label not a tag.
    const typeLabels = {
      module: "Module",
      lesson: "Lesson",
      problem: "Problem",
      resource: "Resource",
    };
    return results
      .map(
        (result) => `
          <article class="search-result-card" data-search-type="${escapeHtml(result.type)}">
            <div class="search-result-copy">
              <div class="search-result-meta">
                <span class="card-pill">${escapeHtml(typeLabels[result.type] || result.type)}</span>
                <span class="search-result-source">${escapeHtml(result.meta)}</span>
              </div>
              <h3>${escapeHtml(result.title)}</h3>
              <p>${escapeHtml(result.body)}</p>
            </div>
            <a class="button-lite search-result-open" href="${escapeHtml(result.href)}" ${
              result.external ? 'target="_blank" rel="noreferrer"' : ""
            }>
              Open
            </a>
          </article>
        `,
      )
      .join("");
  }

  function renderResources(resources) {
    return resources
      .map(
        (resource) => `
          <article class="resource-card">
            <small>${escapeHtml(resource.type)}</small>
            <h3>${escapeHtml(resource.title)}</h3>
            <p>${escapeHtml(resource.summary)}</p>
            <a class="resource-open" href="${escapeHtml(resource.href)}" target="_blank" rel="noreferrer">Open Resource</a>
          </article>
        `,
      )
      .join("");
  }

  // The season is a list of dated facts, so it reads best as a table. Each
  // cell carries a data-label so the narrow layout can stack a row into a card.
  function renderSeason(phases) {
    const rows = phases
      .map(
        (phase, index) => `
          <tr>
            <td data-label="When"><span class="season-date">${escapeHtml(phase.date)}</span></td>
            <th scope="row" data-label="Phase"><span class="season-step">${index + 1}</span>${escapeHtml(phase.title)}</th>
            <td data-label="What happens">${escapeHtml(phase.body)}</td>
          </tr>
        `,
      )
      .join("");
    return `
      <table class="season-table">
        <thead>
          <tr><th scope="col">When</th><th scope="col">Phase</th><th scope="col">What Happens</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function renderChecklist(items) {
    return items
      .map(
        (item, index) => `
          <label class="check-item">
            <input type="checkbox" data-check="${index}" />
            <span>${escapeHtml(item)}</span>
          </label>
        `,
      )
      .join("");
  }

  window.AIO_RENDERERS = {
    STATUS_LABELS,
    escapeHtml,
    inlineMarkup,
    inline,
    renderBlocks,
    renderChoiceQuestion,
    choiceText,
    statusCircle,
    statusMenu,
    resourceStatusMenu,
    resourceStatusButton,
    RESOURCE_STATUS_LABELS,
    frequencyBadge,
    PROBLEM_STATUS_ORDER,
    PROBLEM_STATUS_LABELS,
    statusTitle,
    problemStatusPressed: (status) => PROBLEM_STATUS_PRESSED[problemStatusValue(status)],
    problemHref,
    renderProblemStepper,
    renderFilterChips,
    renderSearchFilters,
    renderSectionProgress,
    renderSectionTimeline,
    renderModuleCard,
    renderStageCards,
    renderGlanceTable,
    renderDashboard,
    renderModuleReader,
    renderPlannerLibrary,
    renderPlannerBoard,
    renderPager,
    renderProblemCards,
    problemType,
    typePill,
    difficultyClass,
    renderSearchResults,
    renderResources,
    renderSeason,
    renderChecklist,
  };
})();
