/*
 * The glossary page (glossary/index.html). Renders window.AIO_GLOSSARY, set by
 * content/glossary.js: [{ id, term, aka, definition, moduleId, tags }]. Terms
 * are grouped by first letter with a jump bar, and the search box filters by
 * term, other names, tags and definition. Each term links to its module.
 */
(function () {
  if (document.body.dataset.page !== "glossary") return;

  const root = document.body.dataset.root || "";
  const renderers = window.AIO_RENDERERS;
  const escapeHtml = renderers.escapeHtml;
  const modules = new Map(((window.AIO_CONTENT && window.AIO_CONTENT.modules) || []).map((module) => [module.id, module]));
  const entries = (Array.isArray(window.AIO_GLOSSARY) ? window.AIO_GLOSSARY : [])
    .filter((entry) => entry && entry.term && entry.definition)
    .sort((a, b) => a.term.localeCompare(b.term, "en", { sensitivity: "base" }));

  const els = {
    search: document.querySelector("#glossarySearch"),
    clear: document.querySelector("#glossaryClear"),
    tools: document.querySelector(".glossary-tools"),
    letters: document.querySelector("#glossaryLetters"),
    count: document.querySelector("#glossaryCount"),
    list: document.querySelector("#glossaryList"),
  };

  // Lower case with accents removed, so "naive" finds "naïve".
  const fold = (value) => String(value || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const haystacks = new Map(
    entries.map((entry) => [entry, fold([entry.term, entry.aka, (entry.tags || []).join(" "), entry.definition].join(" "))]),
  );

  const anchor = (entry) =>
    `term-${String(entry.id || entry.term).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;

  function letterOf(entry) {
    const first = fold(entry.term).charAt(0).toUpperCase();
    return /[A-Z]/.test(first) ? first : "#";
  }

  function renderTerm(entry) {
    const id = anchor(entry);
    const module = entry.moduleId ? modules.get(entry.moduleId) : null;
    const moduleLink = module
      ? `<a class="glossary-module" href="${escapeHtml(`${root}module/?id=${encodeURIComponent(module.id)}`)}"><span aria-hidden="true">&rarr;</span> ${escapeHtml(module.title)}</a>`
      : "";
    const tags = (entry.tags || [])
      .map((tag) => `<span class="glossary-tag">${escapeHtml(tag.replace(/-/g, " "))}</span>`)
      .join("");
    return `
      <article class="glossary-term" id="${escapeHtml(id)}">
        <header class="glossary-term-head">
          <h3><a href="#${escapeHtml(id)}">${renderers.inline(entry.term, root)}</a></h3>
          ${entry.aka ? `<span class="glossary-aka">(${renderers.inline(entry.aka, root)})</span>` : ""}
        </header>
        <p>${renderers.inline(entry.definition, root)}</p>
        ${moduleLink || tags ? `<footer class="glossary-term-foot">${moduleLink}${tags ? `<div class="glossary-tags">${tags}</div>` : ""}</footer>` : ""}
      </article>
    `;
  }

  function typeset() {
    if (!window.katex) return;
    els.list.querySelectorAll("[data-tex]").forEach((element) => {
      window.katex.render(element.getAttribute("data-tex"), element, { displayMode: false, throwOnError: false });
    });
  }

  function render() {
    const raw = els.search ? els.search.value.trim() : "";
    const query = fold(raw);
    const shown = query ? entries.filter((entry) => haystacks.get(entry).includes(query)) : entries;
    const groups = new Map();
    shown.forEach((entry) => {
      const letter = letterOf(entry);
      if (!groups.has(letter)) groups.set(letter, []);
      groups.get(letter).push(entry);
    });
    const letters = [...groups.keys()].sort((a, b) => (a === "#") - (b === "#") || a.localeCompare(b));

    els.letters.innerHTML = letters
      .map((letter) => `<a href="#letter-${letter === "#" ? "other" : letter}">${escapeHtml(letter)}</a>`)
      .join("");
    els.letters.hidden = !letters.length;
    els.list.innerHTML = letters
      .map((letter) => {
        const key = letter === "#" ? "other" : letter;
        const terms = groups.get(letter);
        return `
          <section class="glossary-group" id="letter-${key}" aria-labelledby="letter-${key}-head">
            <div class="glossary-group-head">
              <h2 class="glossary-letter" id="letter-${key}-head">${escapeHtml(letter)}</h2>
              <span class="glossary-rule" aria-hidden="true"></span>
              <span class="glossary-group-count">${terms.length} ${terms.length === 1 ? "term" : "terms"}</span>
            </div>
            <div class="glossary-grid">${terms.map(renderTerm).join("")}</div>
          </section>
        `;
      })
      .join("");
    els.count.classList.toggle("is-empty", !shown.length);
    if (els.clear) els.clear.hidden = !raw;
    if (!entries.length) els.count.textContent = "The glossary is empty.";
    else if (!shown.length) els.count.textContent = `No terms match “${raw}”.`;
    else els.count.textContent = query ? `${shown.length} of ${entries.length} terms` : `${entries.length} terms`;
    typeset();
    activeLetter();
  }

  // ?q= opens the page already filtered, and the box keeps it up to date, so
  // a search can be linked to.
  const params = new URLSearchParams(window.location.search);
  if (els.search) {
    els.search.value = (params.get("q") || "").trim();
    els.search.addEventListener("input", () => {
      render();
      const next = new URLSearchParams(window.location.search);
      if (els.search.value.trim()) next.set("q", els.search.value.trim());
      else next.delete("q");
      const query = next.toString();
      window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`);
    });
  }
  render();

  // The letter that owns the text on screen is the one a jump would take the
  // reader away from, so it earns the filled pill. The line sits a third of the
  // way into the reading area: a jumped-to heading lights its own letter the
  // moment it appears under the sticky bar. On a narrow screen the tools are not
  // sticky, so the reading area starts under the fixed header instead.
  function activeLetter() {
    if (!els.list || !els.letters) return;
    const header = document.querySelector(".site-header");
    const bar = Math.max(
      els.tools ? els.tools.getBoundingClientRect().bottom : 0,
      header ? header.getBoundingClientRect().bottom : 0,
    );
    const line = bar + (window.innerHeight - bar) * 0.35;
    const groups = els.list.querySelectorAll(".glossary-group");
    let current = groups.length ? groups[0].id : null;
    for (const group of groups) {
      if (group.getBoundingClientRect().top <= line) current = group.id;
      else break;
    }
    for (const link of els.letters.querySelectorAll("a")) {
      const on = current !== null && link.getAttribute("href") === `#${current}`;
      link.classList.toggle("is-active", on);
      if (on) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }

  let spyFrame = 0;
  function scheduleSpy() {
    if (spyFrame) return;
    spyFrame = requestAnimationFrame(() => {
      spyFrame = 0;
      activeLetter();
    });
  }

  window.addEventListener("scroll", scheduleSpy, { passive: true });
  window.addEventListener("resize", scheduleSpy);
  activeLetter();

  // Clearing starts over in place: the input event keeps the URL and the list in
  // sync, so the button only has to empty the box and hand focus back.
  if (els.clear) {
    els.clear.addEventListener("click", () => {
      if (!els.search) return;
      els.search.value = "";
      els.search.dispatchEvent(new Event("input", { bubbles: true }));
      els.search.focus();
    });
  }

  // A link to one term (#term-...) lands on it once the list exists. A pasted
  // hash with a bad percent sequence is not a real id, and must not throw.
  if (window.location.hash) {
    let target = null;
    try {
      target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
    } catch (error) {
      target = null;
    }
    if (target) {
      target.scrollIntoView({ block: "start" });
    }
  }
})();
