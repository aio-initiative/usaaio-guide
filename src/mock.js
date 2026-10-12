/*
 * Mock Arena, setup page.
 *
 * Builds the set (time limit and problems), then hands off to /mock/run/, which
 * owns the sitting itself. Shared state lives in src/mock-session.js.
 */
(function () {
  if (document.body.dataset.page !== "mock") return;

  const mock = window.AIO_MOCK;
  const els = { arena: document.querySelector("#mockArena") };
  if (!els.arena || !mock) return;

  const MAX_RESULTS = 8;
  // Sized against the real papers. Round 1 runs for hours and each Round 2 day
  // is a half-day sitting, so the quick picks start where a serious set starts.
  const QUICK_MINUTES = [60, 90, 120, 180, 240, 300];

  // Three tiers over the bank's four difficulty labels. "Hard" folds in the
  // handful of "Very Hard" problems so nothing is unreachable.
  const RANDOM_TIERS = [
    { id: "easy", label: "Easy", levels: ["Easy"] },
    { id: "medium", label: "Medium", levels: ["Medium"] },
    { id: "hard", label: "Hard", levels: ["Hard", "Very Hard"] },
  ];
  const RANDOM_COUNTS = [3, 5, 8, 10];

  const problemData = window.AIO_PROBLEMS || { problems: [] };
  const catalog = window.AIO_CONTENT || { modules: [] };
  const moduleLookup = new Map((catalog.modules || []).map((m) => [m.id, m]));
  const problemLookup = new Map(problemData.problems.map((p) => [p.id, p]));
  const knownIds = new Set(problemLookup.keys());

  let session = mock.load(knownIds);
  let query = "";
  let randomCount = 5;

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  // The three problem kinds the site shows: a notebook is Code, one
  // multiple-choice question is MCQ, every other written problem is Text.
  function problemKind(p) {
    if (!p.tags.includes("Theory")) return { id: "code", label: "Code" };
    return p.tags.includes("Multiple choice") ? { id: "mcq", label: "MCQ" } : { id: "text", label: "Text" };
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
  const moduleTitle = (p) => moduleLookup.get(p.moduleId)?.title || p.moduleId;

  const selectedProblems = () => session.problemIds.map((id) => problemLookup.get(id)).filter(Boolean);
  const tierPool = (tier) => problemData.problems.filter((p) => tier.levels.includes(p.difficulty));

  // Fisher-Yates on a copy, so the pool itself is never reordered.
  function sample(list, n) {
    const pool = [...list];
    for (let i = pool.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, n);
  }

  function searchMatches() {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return problemData.problems.filter((problem) => {
      if (session.problemIds.includes(problem.id)) return false;
      const haystack = `${problem.title} ${problem.difficulty} ${problem.tags.join(" ")} ${problemKind(problem).label} ${moduleTitle(
        problem,
      )}`.toLowerCase();
      return haystack.includes(q);
    });
  }

  function cardHead(step, title, aside) {
    return `
      <header class="mock-card-head">
        <span class="mock-step" aria-hidden="true">${step}</span>
        <h3>${title}</h3>
        ${aside || ""}
      </header>`;
  }

  function render() {
    const matches = searchMatches();
    const results = matches.slice(0, MAX_RESULTS);
    const chosen = selectedProblems();

    const quick = QUICK_MINUTES.map(
      (m) =>
        `<button class="mock-quick ${session.minutes === m ? "active" : ""}" type="button" data-minutes="${m}" aria-pressed="${
          session.minutes === m ? "true" : "false"
        }">${m} min</button>`,
    ).join("");

    const counts = RANDOM_COUNTS.map(
      (count) => `<button class="mock-size" type="button" data-random-count="${count}" aria-pressed="${count === randomCount}">${count}</button>`,
    ).join("");

    const generators = RANDOM_TIERS.map((tier) => {
      const pool = tierPool(tier).length;
      return `<button class="mock-generator mock-generator--${escapeHtml(tier.id)}" type="button" data-random="${escapeHtml(tier.id)}" ${pool ? "" : "disabled"}>
          <strong>${escapeHtml(tier.label)}</strong>
          <small>${pool ? `${Math.min(randomCount, pool)} from ${pool}` : "none available"}</small>
        </button>`;
    }).join("");

    const resultList = query.trim()
      ? results.length
        ? results
            .map(
              (p) => `
                <li>
                  <button class="mock-result" type="button" data-add="${escapeHtml(p.id)}">
                    <span class="mock-result-copy">
                      <strong>${escapeHtml(p.title)}</strong>
                      <small>${escapeHtml(moduleTitle(p))}</small>
                    </span>
                    <span class="mock-result-tags">${typePill(p)}${difficultyChip(p)}</span>
                    <span class="mock-result-add"><span aria-hidden="true">+</span> Add</span>
                  </button>
                </li>`,
            )
            .join("")
        : `<li class="mock-empty">No unadded problems match “${escapeHtml(query.trim())}”.</li>`
      : "";

    const resultCount =
      matches.length > results.length
        ? `<p class="helper-text mock-result-count">Showing ${results.length} of ${matches.length} matches. Type more to narrow the list.</p>`
        : "";

    const chosenTable = chosen.length
      ? `
        <div class="mock-table-wrap">
          <table class="mock-table mock-set-table">
            <thead>
              <tr>
                <th scope="col" class="mock-col-num">#</th>
                <th scope="col">Problem</th>
                <th scope="col">Type</th>
                <th scope="col">Difficulty</th>
                <th scope="col" class="mock-col-action"><span class="sr-only">Remove</span></th>
              </tr>
            </thead>
            <tbody>
              ${chosen
                .map(
                  (p, i) => `
                <tr class="mock-row">
                  <td class="mock-col-num"><span class="mock-chosen-index">${i + 1}</span></td>
                  <td class="mock-col-title">
                    <span class="mock-chosen-copy">
                      <strong>${escapeHtml(p.title)}</strong>
                      <small>${escapeHtml(moduleTitle(p))}</small>
                    </span>
                  </td>
                  <td data-label="Type">${typePill(p)}</td>
                  <td data-label="Difficulty">${difficultyChip(p)}</td>
                  <td class="mock-col-action">
                    <button class="mock-remove" type="button" data-remove="${escapeHtml(p.id)}" aria-label="Remove ${escapeHtml(p.title)}">
                      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" /></svg>
                      <span>Remove</span>
                    </button>
                  </td>
                </tr>`,
                )
                .join("")}
            </tbody>
          </table>
        </div>`
      : `<p class="mock-empty mock-empty-box">No problems yet. Search above, or generate a random set.</p>`;

    els.arena.innerHTML = `
      <div class="section-heading compact"><h2>Set Up Your Mock</h2></div>

      <section class="mock-card mock-block">
        ${cardHead(1, `<label for="mockMinutes">Time Limit</label>`)}
        <div class="mock-duration">
          <span class="mock-minutes-field">
            <input class="mock-minutes" id="mockMinutes" type="number" min="1" max="${mock.MAX_MINUTES}" step="1" value="${escapeHtml(session.minutes)}" aria-label="Time limit in minutes" />
            <span class="mock-unit">minutes</span>
          </span>
          <div class="mock-quick-row" role="group" aria-label="Common time limits">${quick}</div>
        </div>
      </section>

      <section class="mock-card mock-block">
        ${cardHead(2, `<label for="mockSearch">Add Problems</label>`)}
        <label class="problems-search mock-search">
          <span class="sr-only">Search the Problem Bank</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m16 16 4 4" /></svg>
          <input id="mockSearch" type="search" placeholder="Search by title, topic, or difficulty" value="${escapeHtml(query)}" />
        </label>
        ${resultList ? `<ul class="mock-results">${resultList}</ul>` : ""}
        ${resultCount}
        <div class="mock-random">
          <p class="mock-label subtle">Or generate a random set</p>
          <fieldset class="mock-sizes">
            <legend>Number of problems</legend>
            <div class="mock-size-options">${counts}</div>
          </fieldset>
          <div class="mock-generators">${generators}</div>
          <p class="helper-text mock-hint">Picks ${randomCount} problems at random from that difficulty and replaces your set. Click again for a new draw. Your time limit is left alone.</p>
        </div>
      </section>

      <section class="mock-card mock-block">
        ${cardHead(3, "Your Set", `<span class="mock-count">${chosen.length} problem${chosen.length === 1 ? "" : "s"}</span>`)}
        ${chosenTable}
        <div class="mock-actions">
          <button class="button button-primary mock-start" type="button" id="mockStart" ${chosen.length ? "" : "disabled"}>
            Start ${escapeHtml(session.minutes)}-Minute Mock
          </button>
          ${chosen.length ? `<button class="button button-secondary dark-text" type="button" id="mockClear">Clear Set</button>` : ""}
        </div>
      </section>
    `;
  }

  function update(mutate) {
    mutate();
    mock.save(session);
    render();
  }

  els.arena.addEventListener("click", (event) => {
    const add = event.target.closest("[data-add]");
    if (add) {
      const id = add.dataset.add;
      return update(() => {
        if (!session.problemIds.includes(id)) session.problemIds.push(id);
      });
    }

    const remove = event.target.closest("[data-remove]");
    if (remove) {
      const id = remove.dataset.remove;
      return update(() => {
        session.problemIds = session.problemIds.filter((item) => item !== id);
      });
    }

    const minutes = event.target.closest("[data-minutes]");
    if (minutes) {
      return update(() => {
        session.minutes = Number(minutes.dataset.minutes);
      });
    }

    const count = event.target.closest("[data-random-count]");
    if (count) {
      const value = Number(count.dataset.randomCount);
      if (!RANDOM_COUNTS.includes(value)) return;
      randomCount = value;
      render();
      els.arena.querySelector(`[data-random-count="${value}"]`).focus();
      return;
    }

    const random = event.target.closest("[data-random]");
    if (random) {
      const tier = RANDOM_TIERS.find((t) => t.id === random.dataset.random);
      if (!tier) return;
      const picked = sample(tierPool(tier), randomCount);
      if (!picked.length) return;
      // Replaces the set (so clicking again draws a new one) but leaves the
      // time limit alone. Choosing the duration is the student's call.
      return update(() => {
        session.problemIds = picked.map((p) => p.id);
      });
    }

    if (event.target.closest("#mockClear")) {
      return update(() => {
        session.problemIds = [];
      });
    }

    if (event.target.closest("#mockStart")) {
      if (!session.problemIds.length) return;
      session.status = "running";
      session.activeIndex = 0;
      session.startedAt = Date.now();
      session.endsAt = Date.now() + session.minutes * 60 * 1000;
      session.finishedAt = null;
      mock.save(session);
      window.location.href = "run/";
    }
  });

  els.arena.addEventListener("input", (event) => {
    if (event.target.matches("#mockSearch")) {
      query = event.target.value;
      render();
      // Re-focus and restore the caret, since render() replaced the input.
      const field = document.querySelector("#mockSearch");
      if (field) {
        field.focus();
        field.setSelectionRange(field.value.length, field.value.length);
      }
      return;
    }
    if (event.target.matches("#mockMinutes")) {
      const value = Number(event.target.value);
      if (!Number.isFinite(value) || value <= 0) return;
      session.minutes = Math.min(Math.round(value), mock.MAX_MINUTES);
      mock.save(session);
      const startButton = document.querySelector("#mockStart");
      if (startButton) startButton.textContent = `Start ${session.minutes}-minute mock`;
      document.querySelectorAll("[data-minutes]").forEach((chip) => {
        const on = Number(chip.dataset.minutes) === session.minutes;
        chip.classList.toggle("active", on);
        chip.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }
  });

  // A mock already under way belongs on the run page, not back at setup.
  if (session.status === "running" && mock.remainingMs(session) > 0) {
    window.location.replace("run/");
    return;
  }
  if (session.status !== "setup") {
    session.status = "setup";
    mock.save(session);
  }
  render();
})();
