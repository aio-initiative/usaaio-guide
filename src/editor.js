(function () {
  if (document.body.dataset.page !== "editor") return;

  const root = document.body.dataset.root || "";
  const params = new URLSearchParams(window.location.search);
  const notebookId = (params.get("nb") || "").replace(/[^a-z0-9-]/g, "");
  // ?embed=1 is used by the Mock Arena, which frames this editor as the body of
  // a coding problem. It drops the site chrome AND strips the solution, so a
  // mock in progress cannot reach the answer through the IDE. The mock picks
  // the layout (?layout=text, single or split) and changes it through
  // window.AIO_EDITOR_EMBED, so the running Python survives a switch.
  const EMBED = params.get("embed") === "1";
  // Opened from a module's problem list: previous and next stay in that module.
  const FROM_MODULE = params.get("from") === "module";
  const EMBED_LAYOUTS = ["text", "single", "split"];
  const DRAFT_KEY = `usaaio-editor-draft-v2:${notebookId}`;
  const LAYOUT_KEY = "usaaio-editor-layout";
  // The narrowest window, in CSS pixels, with room for text and code side by side.
  const SPLIT_MIN_WIDTH = 1024;
  // How much printed text a cell keeps (about 200 KB), so a loop that prints
  // without end cannot fill the page.
  const OUTPUT_LIMIT = 200000;
  const OUTPUT_REFRESH_MS = 100;
  const LOAD_TIMEOUT_MS = 120000;
  // This file's own ?v= string, so the worker, the notebook and the stand-in
  // are fetched fresh whenever the site's cache string is bumped.
  const VERSION = (() => {
    try {
      return new URL(document.currentScript.src).search;
    } catch (error) {
      return "";
    }
  })();

  const STATUS_LABELS = {
    off: "Python: not started",
    loading: "Python: downloading runtime (~30-60s on first use)...",
    ready: "Python: ready",
    running: "Python: running",
    restarted: "Python: restarted",
    failed: "Python: failed to start",
  };
  const STOPPED_MESSAGE = "Stopped.";
  const CRASHED_MESSAGE = "Python stopped unexpectedly.";
  const RESTART_NOTICE = "Python restarted. Run the cells above again, or use Run all.";
  const TODO_HINT = "This runs code that still has a TODO. Fill it in, run that cell, then run this one again.";
  const COLAB_INSTEAD = "Download it and run it in Colab or on your machine.";
  const PASSED_LINE = /^(?:checks passed|Part \d+ passed)\b/im;
  // A self-check cell is one that prints that line.
  const CHECK_CELL = /(?:^|\n)[ \t]*print\(\s*f?["'](?:checks passed|Part \d+ passed)\b/i;

  const renderers = window.AIO_RENDERERS;
  const storage = window.AIO_PROGRESS;

  const els = {
    title: document.querySelector("#editor-title"),
    subtitle: document.querySelector("#editorSubtitle"),
    toolbar: document.querySelector("#editorToolbar"),
    note: document.querySelector("#editorNote"),
    notice: document.querySelector("#editorNotice"),
    cells: document.querySelector("#editorCells"),
    panel: document.querySelector(".editor-panel"),
    empty: document.querySelector("#editorEmptyState"),
    kernel: document.querySelector("#kernelStatus"),
    runAll: document.querySelector("#runAllBtn"),
    stop: document.querySelector("#stopBtn"),
    reset: document.querySelector("#resetBtn"),
    download: document.querySelector("#downloadBtn"),
    split: document.querySelector("#splitBtn"),
    moduleLink: document.querySelector("#editorModuleLink"),
    meta: document.querySelector("#editorMeta"),
    actions: document.querySelector("#editorActions"),
    back: document.querySelector("#editorBack"),
    stepper: document.querySelector("#editorStepper"),
    check: document.querySelector("#editorCheck"),
  };

  let originalNotebook = null;
  let meta = {};
  let problem = null;
  // The cells the reader works in, and the worked solution kept apart from
  // them: shown folded and read-only, never run, never saved in a draft.
  let cells = []; // [{ type, source, original }]
  let solutionCells = []; // [{ type, source }]
  let cellEls = [];
  let runs = new Map(); // cell index -> { status, session, count, output, truncated, error, figures }
  let session = 0;
  let execCounter = 0;
  let busy = false;
  let restarted = false;
  let storedLayout = "single";

  const escapeHtml = (value) =>
    renderers
      ? renderers.escapeHtml(value)
      : String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");

  /* --------------------------- syntax highlighting -------------------------- */

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
    return `<span class="tok-${kind}">${escapeHtml(text)}</span>`;
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

  /* -------------------------------- markdown ------------------------------- */

  // Inline text uses the lesson reader's markup: `code`, **bold**, *italic*,
  // [links](...) and $math$ (typeset by KaTeX after the cell is drawn).
  function inline(text) {
    // A real no-break space, not an ordinary one, so "60,000 images" does not
    // wrap between the number and the word.
    const source = String(text).replace(/&nbsp;/g, "\u00A0");
    if (renderers && renderers.inline) return renderers.inline(source, root);
    return escapeHtml(source);
  }

  function isTableRow(line) {
    return /^\s*\|.*\|\s*$/.test(line);
  }

  // A pipe inside a code span or inside inline math does not end the cell.
  function tableCells(line) {
    const cells = [];
    let buf = "";
    let inCode = false;
    let inMath = false;
    let escaped = false;
    for (const ch of line.trim().slice(1, -1)) {
      if (escaped) {
        buf += ch;
        escaped = false;
        continue;
      }
      if (ch === "\\") {
        buf += ch;
        escaped = true;
        continue;
      }
      if (ch === "`") inCode = !inCode;
      else if (!inCode && ch === "$") inMath = !inMath;
      if (ch === "|" && !inCode && !inMath) {
        cells.push(buf.trim());
        buf = "";
      } else buf += ch;
    }
    cells.push(buf.trim());
    return cells;
  }

  /* A small markdown renderer: enough for the notebooks (headings, paragraphs,
     lists, tables, fenced code, display math, inline markup). */
  function renderMarkdown(md) {
    const lines = md.split("\n");
    const out = [];
    let paraBuf = [];
    let listBuf = [];
    let listOrdered = false;

    function flushPara() {
      if (paraBuf.length) out.push(`<p>${inline(paraBuf.join(" "))}</p>`);
      paraBuf = [];
    }
    function flushList() {
      if (listBuf.length) {
        const tag = listOrdered ? "ol" : "ul";
        out.push(`<${tag}>${listBuf.map((item) => `<li>${inline(item)}</li>`).join("")}</${tag}>`);
      }
      listBuf = [];
    }
    function flushAll() {
      flushPara();
      flushList();
    }

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const trimmed = line.trim();
      if (trimmed.startsWith("```")) {
        flushAll();
        const lang = trimmed.slice(3).trim().toLowerCase();
        const body = [];
        i += 1;
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          body.push(lines[i]);
          i += 1;
        }
        const code = body.join("\n");
        out.push(`<pre class="md-code"><code>${/^(python|py)$/.test(lang) ? highlightPython(code) : escapeHtml(code)}</code></pre>`);
        continue;
      }
      if (trimmed.startsWith("$$")) {
        flushAll();
        let tex = trimmed.slice(2);
        if (tex.endsWith("$$") && tex.length >= 2) {
          tex = tex.slice(0, -2);
        } else {
          const body = [tex];
          i += 1;
          while (i < lines.length && !lines[i].trim().endsWith("$$")) {
            body.push(lines[i]);
            i += 1;
          }
          if (i < lines.length) body.push(lines[i].trim().replace(/\$\$$/, ""));
          tex = body.join("\n");
        }
        out.push(`<div class="reader-math-display" data-tex="${escapeHtml(tex.trim())}"><code>${escapeHtml(tex.trim())}</code></div>`);
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flushAll();
        const level = Math.min(heading[1].length + 1, 5); // demote: nb h1 -> page h2
        out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        continue;
      }
      if (isTableRow(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        flushAll();
        const head = tableCells(line);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableRow(lines[i])) {
          rows.push(tableCells(lines[i]));
          i += 1;
        }
        i -= 1;
        out.push(
          `<div class="editor-md-table"><table><thead><tr>${head.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead>` +
            `<tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`,
        );
        continue;
      }
      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (bullet || ordered) {
        flushPara();
        if (listBuf.length && listOrdered !== Boolean(ordered)) flushList();
        listOrdered = Boolean(ordered);
        listBuf.push((bullet || ordered)[1]);
        continue;
      }
      if (!trimmed) {
        flushAll();
        continue;
      }
      if (listBuf.length) {
        listBuf[listBuf.length - 1] += ` ${trimmed}`;
        continue;
      }
      paraBuf.push(trimmed);
    }
    flushAll();
    return out.join("\n");
  }

  function typeset(container) {
    if (!window.katex || !container) return;
    container.querySelectorAll("[data-tex]").forEach((element) => {
      window.katex.render(element.getAttribute("data-tex"), element, {
        displayMode: element.classList.contains("reader-math-display"),
        throwOnError: false,
      });
    });
  }

  /* --------------------------------- Python -------------------------------- */

  // The page's handle on the notebook's Python (src/notebook-worker.js). The
  // worker starts on the first run. Python cannot be interrupted from outside,
  // so stop() ends the worker; the next run starts a fresh one.
  const kernel = {
    status: "off",
    worker: null,
    loaded: false,
    loadTimer: null,
    nextRunId: 1,
    pending: new Map(),

    run(code, onOutput) {
      const worker = this.worker || this.start();
      const runId = this.nextRunId;
      this.nextRunId += 1;
      return new Promise((resolve) => {
        this.pending.set(runId, { onOutput, resolve });
        if (this.status === "ready") this.setStatus("running");
        // A worker that is still loading holds the message until it is ready.
        worker.postMessage({ type: "run", runId, code, assetBase: new URL(`${root}public/pyodide/`, window.location.href).href });
      });
    },

    stop() {
      if (!this.worker) return;
      this.endWorker();
      this.finishAll(STOPPED_MESSAGE);
      this.setStatus("restarted");
    },

    start() {
      let worker;
      try {
        worker = new Worker(`${root}src/notebook-worker.js${VERSION}`);
      } catch (error) {
        this.setStatus("failed");
        throw error;
      }
      this.worker = worker;
      worker.onmessage = (event) => {
        if (worker === this.worker) this.receive(event.data || {});
      };
      worker.onerror = (event) => {
        if (worker !== this.worker) return;
        if (this.loaded) {
          // A Python that was working broke. Its names are gone, as after stop().
          this.endWorker();
          this.finishAll(CRASHED_MESSAGE);
          this.setStatus("restarted");
          return;
        }
        this.fail(event && event.message ? `Python could not be loaded: ${event.message}` : "Python could not be loaded.");
      };
      this.loadTimer = window.setTimeout(
        () => this.fail("Loading Python timed out. Check your connection and try again."),
        LOAD_TIMEOUT_MS,
      );
      this.setStatus("loading");
      return worker;
    },

    receive(message) {
      if (message.type === "status") {
        if (message.status === "ready") {
          this.loaded = true;
          this.clearLoadTimer();
          this.setStatus(this.pending.size ? "running" : "ready");
        } else if (message.status === "failed") {
          this.fail(message.message || "Python could not be loaded.");
        }
        return;
      }
      const run = this.pending.get(message.runId);
      if (!run) return;
      if (message.type === "stdout" || message.type === "figure") {
        run.onOutput(message);
        return;
      }
      if (message.type === "done") {
        this.pending.delete(message.runId);
        if (!this.pending.size) this.setStatus("ready");
        run.resolve({ error: message.error, seconds: message.seconds });
      }
    },

    fail(message) {
      this.endWorker();
      this.finishAll(message);
      this.setStatus("failed");
    },

    endWorker() {
      this.clearLoadTimer();
      if (this.worker) this.worker.terminate();
      this.worker = null;
      this.loaded = false;
    },

    finishAll(error) {
      const waiting = [...this.pending.values()];
      this.pending.clear();
      waiting.forEach((run) => run.resolve({ error, seconds: 0 }));
    },

    clearLoadTimer() {
      if (this.loadTimer !== null) window.clearTimeout(this.loadTimer);
      this.loadTimer = null;
    },

    setStatus(status) {
      if (this.status === status) return;
      const everLoaded = this.everLoaded || status === "ready" || status === "running";
      this.everLoaded = everLoaded;
      if (status === "restarted" || status === "failed") {
        // That Python is gone, and what it ran no longer counts.
        session += 1;
        execCounter = 0;
      }
      this.status = status;
      onKernelStatus(status, everLoaded);
    },
  };

  let quietStop = false;

  function onKernelStatus(status, everLoaded) {
    let shown = status;
    if (status === "restarted") {
      if (!everLoaded) {
        // Stopped while the first Python was still loading: nothing ran, so
        // nothing was lost and there is nothing to run again.
        shown = "off";
        kernel.status = "off";
      } else if (!quietStop) {
        restarted = true;
      }
    }
    if (status === "restarted" || status === "failed") {
      // A dead Python's check marks and run counts are no longer true, and
      // nothing else recomputes them until the next cell runs.
      refreshCounts();
      refreshPassed();
    }
    if (els.kernel) {
      // Nothing to say until Python starts; the pill stays out of the toolbar.
      els.kernel.hidden = shown === "off";
      els.kernel.textContent = STATUS_LABELS[shown];
      els.kernel.classList.toggle("busy", shown === "loading" || shown === "running");
      els.kernel.classList.toggle("is-warning", shown === "restarted");
      els.kernel.classList.toggle("is-failed", shown === "failed");
    }
    refreshNotice();
  }

  function refreshNotice() {
    if (!els.notice) return;
    // Clear the text while the region is still visible, then hide it, so a screen
    // reader does not announce the empty string. And set the new text only after
    // unhiding, because content written to a hidden region can be missed.
    if (!restarted) {
      els.notice.textContent = "";
      els.notice.hidden = true;
      return;
    }
    els.notice.hidden = false;
    els.notice.textContent = RESTART_NOTICE;
  }

  /* ------------------------------ persistence ------------------------------ */

  function saveDraft() {
    try {
      const edited = cells.some((cell) => cell.source !== cell.original);
      if (!edited) {
        window.localStorage.removeItem(DRAFT_KEY);
        return;
      }
      const payload = { fp: fingerprint(cells), cells: cells.map((cell) => ({ type: cell.type, source: cell.source })) };
      window.localStorage.setItem(DRAFT_KEY, JSON.stringify(payload));
    } catch (error) {
      /* storage full or blocked: editing still works, just not persisted */
    }
  }

  // A draft is kept only while it still lines up with the notebook: the same
  // cells, in the same order, with the same original text. A draft saved
  // against an older version of the notebook is dropped rather than silently
  // overwriting a fix the author made. Drafts saved before the fingerprint
  // was written are bare arrays, and are read the same way.
  function fingerprint(list) {
    let h = 0x811c9dc5;
    for (const cell of list) {
      for (const ch of `${cell.type}\u0000${cell.original}\u0000`) {
        h ^= ch.charCodeAt(0);
        h = Math.imul(h, 0x01000193);
      }
    }
    return h >>> 0;
  }

  function loadDraft() {
    try {
      const raw = window.localStorage.getItem(DRAFT_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      const saved = Array.isArray(parsed) ? parsed : parsed && Array.isArray(parsed.cells) ? parsed.cells : null;
      if (!saved) return null;
      const list = saved.filter((cell) => cell && typeof cell.source === "string");
      const cut = list.findIndex((cell) => cell.type === "markdown" && /^#{1,3}\s+Solution\b/.test(cell.source.trim()));
      const draft = cut >= 0 ? list.slice(0, cut) : list;
      if (draft.length !== cells.length || draft.some((cell, index) => cell.type !== cells[index].type)) return null;
      if (parsed.fp !== undefined && parsed.fp !== fingerprint(cells)) return null;
      return draft;
    } catch (error) {
      return null;
    }
  }

  const cellSource = (cell) => (Array.isArray(cell.source) ? cell.source.join("") : String(cell.source || ""));

  // The worked solution is the tail of the notebook: from the first cell
  // marked {"usaaio_solution": true}, or the first "## Solution" heading in a
  // notebook written before the flag existed.
  const SOLUTION_HEADING = /^#{1,3}\s+Solution\b/;

  function solutionStart(raw) {
    return raw.findIndex(
      (cell) =>
        (cell.metadata && cell.metadata.usaaio_solution) ||
        (cell.cell_type === "markdown" && SOLUTION_HEADING.test(cellSource(cell).trim())),
    );
  }

  /* -------------------------------- rendering ------------------------------ */

  function autoSize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight + 4}px`;
  }

  function countLabel(index) {
    const run = runs.get(index);
    if (run && run.status === "running") return "[*]";
    if (run && run.session === session && run.count) return `[${run.count}]`;
    return "[ ]";
  }

  function refreshCounts() {
    cellEls.forEach((wrap, index) => {
      const count = wrap && wrap.querySelector(".cell-count");
      if (count) count.textContent = countLabel(index);
    });
  }

  function errorLine(traceback) {
    const lines = String(traceback).trim().split("\n").filter(Boolean);
    return lines[lines.length - 1] || "";
  }

  function renderOutput(index) {
    const wrap = cellEls[index];
    const out = wrap && wrap.querySelector(".editor-output");
    if (!out) return;
    const run = runs.get(index);
    const text = run ? run.output.replace(/\s+$/, "") : "";
    if (!run || (!text && !run.truncated && !run.error && !run.figures.length)) {
      out.hidden = true;
      out.innerHTML = "";
      return;
    }
    // The frame says what happened before the reader reads the text: still
    // running, an error, or a self-check that printed its passed line.
    const isCheck = CHECK_CELL.test(cells[index].original);
    let state = "done";
    let status = "";
    if (run.status === "running") {
      state = "running";
      status = "Running";
    } else if (run.error === STOPPED_MESSAGE) {
      state = "stopped";
    } else if (run.error) {
      state = "error";
      status = isCheck && /^AssertionError\b/.test(errorLine(run.error)) ? "Check failed" : "Error";
    } else if (isCheck && PASSED_LINE.test(run.output)) {
      state = "passed";
      status = "Check passed";
    }
    const parts = [];
    if (run.truncated) parts.push('<p class="editor-output-note">Earlier output hidden</p>');
    if (text) parts.push(`<pre class="editor-output-text">${escapeHtml(text)}</pre>`);
    run.figures.forEach((png, n) => {
      parts.push(`<img class="editor-output-figure" src="data:image/png;base64,${escapeHtml(png)}" alt="Figure ${n + 1} drawn by this cell" />`);
    });
    if (run.error === STOPPED_MESSAGE) {
      parts.push(`<p class="editor-output-note">${STOPPED_MESSAGE}</p>`);
    } else if (run.error) {
      const hint = /(?:^|[\n.])NotImplementedError$/.test(run.error.trim()) ? TODO_HINT : "";
      const restartHint = restarted && run.session === session ? RESTART_NOTICE : "";
      parts.push(`
        <div class="editor-output-error">
          <p class="editor-error-line">${escapeHtml(errorLine(run.error))}</p>
          ${hint ? `<p class="editor-error-hint">${escapeHtml(hint)}</p>` : ""}
          ${restartHint ? `<p class="editor-error-hint">${escapeHtml(restartHint)}</p>` : ""}
          ${
            run.error.trim().includes("\n")
              ? `<details class="editor-traceback"><summary>Full Traceback</summary><pre>${escapeHtml(run.error)}</pre></details>`
              : ""
          }
        </div>
      `);
    }
    // Unhide before writing: content written to a hidden aria-live region can be
    // missed by screen readers.
    out.hidden = false;
    out.dataset.state = state;
    out.innerHTML = `
      <div class="editor-output-head">
        <span class="editor-output-label">Output</span>
        ${status ? `<span class="editor-output-status">${status}</span>` : ""}
      </div>
      <div class="editor-output-body">${parts.join("")}</div>
    `;
    out.setAttribute("aria-busy", run.status === "running" ? "true" : "false");
  }

  function buildMarkdownCell(cell, index) {
    const wrap = document.createElement("div");
    wrap.className = "editor-cell editor-cell-markdown";
    wrap.dataset.index = String(index);
    const main = document.createElement("div");
    main.className = "cell-main";
    const body = document.createElement("div");
    body.className = "editor-md";
    const shown = cell.display !== undefined ? cell.display : cell.source;
    body.innerHTML = renderMarkdown(shown);
    typeset(body);
    if (!shown.trim()) wrap.hidden = true;
    // Terms in the text open the definition bank, as on the other problem pages.
    // The mock frame has no room for that dialog, so it leaves them plain.
    else if (!EMBED && window.AIO_DEFINITION_LINKS) window.AIO_DEFINITION_LINKS.annotate(body);
    main.appendChild(body);
    wrap.appendChild(main);
    return wrap;
  }

  function buildCodeCell(cell, index, position) {
    const wrap = document.createElement("div");
    wrap.className = "editor-cell editor-cell-code";
    wrap.dataset.index = String(index);
    const isCheck = CHECK_CELL.test(cell.original);
    if (isCheck) wrap.classList.add("is-check");
    const main = document.createElement("div");
    main.className = "cell-main";
    // The frame holds the head (label, run count, Run) and the code under it.
    const frame = document.createElement("div");
    frame.className = "cell-frame";
    const head = document.createElement("div");
    head.className = "cell-head";

    const label = document.createElement("span");
    label.className = "cell-label";
    label.textContent = `Code cell ${position}`;
    const count = document.createElement("span");
    count.className = "cell-count";
    count.textContent = countLabel(index);
    head.appendChild(label);
    head.appendChild(count);
    if (isCheck) {
      const tag = document.createElement("span");
      tag.className = "cell-tag";
      tag.textContent = "Self-check";
      head.appendChild(tag);
    }

    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.className = "cell-run";
    runBtn.title = "Run cell (Shift+Enter)";
    runBtn.setAttribute("aria-label", `Run code cell ${position}`);
    runBtn.innerHTML =
      '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.2 3.4a.6.6 0 0 1 .91-.51l7.2 4.6a.6.6 0 0 1 0 1.02l-7.2 4.6a.6.6 0 0 1-.91-.51z"/></svg>' +
      '<span class="cell-run-label">Run</span>';
    runBtn.addEventListener("click", () => runOne(index));
    head.appendChild(runBtn);

    const editorWrap = document.createElement("div");
    editorWrap.className = "cell-editor";
    const highlight = document.createElement("pre");
    highlight.className = "editor-highlight";
    highlight.setAttribute("aria-hidden", "true");
    const highlightCode = document.createElement("code");
    highlight.appendChild(highlightCode);

    const textarea = document.createElement("textarea");
    textarea.className = "editor-code";
    textarea.spellcheck = false;
    textarea.setAttribute("autocapitalize", "off");
    textarea.setAttribute("autocorrect", "off");
    textarea.value = cell.source;
    textarea.setAttribute("aria-label", `Code cell ${position}. Shift+Enter runs it, Escape leaves it.`);

    const syncHighlight = () => {
      highlightCode.innerHTML = `${highlightPython(textarea.value)}\n`;
    };
    syncHighlight();
    const changed = () => {
      markInProgress();
      cells[index].source = textarea.value;
      syncHighlight();
      autoSize(textarea);
      saveDraft();
    };

    textarea.addEventListener("input", changed);
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.shiftKey || event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        runOne(index);
        return;
      }
      // Escape leaves the editor, so Tab moves focus again rather than indenting.
      if (event.key === "Escape") {
        textarea.blur();
        return;
      }
      if (event.key === "Tab" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        // insertText keeps the browser's undo history; setting the value wipes it.
        // Some engines run it without firing an input event, so changed() is
        // called either way; it only re-reads the textarea, so it is idempotent.
        if (!document.execCommand || !document.execCommand("insertText", false, "    ")) {
          textarea.setRangeText("    ", textarea.selectionStart, textarea.selectionEnd, "end");
        }
        changed();
      }
    });

    editorWrap.appendChild(highlight);
    editorWrap.appendChild(textarea);
    frame.appendChild(head);
    frame.appendChild(editorWrap);
    main.appendChild(frame);

    const output = document.createElement("div");
    output.className = "editor-output";
    output.setAttribute("aria-live", "polite");
    output.hidden = true;
    main.appendChild(output);

    wrap.appendChild(main);
    return wrap;
  }

  function buildCells() {
    let position = 0;
    cellEls = cells.map((cell, index) => {
      if (cell.type === "markdown") return buildMarkdownCell(cell, index);
      position += 1;
      return buildCodeCell(cell, index, position);
    });
    cellEls.forEach((wrap, index) => {
      wrap.style.animationDelay = `${Math.min(index * 40, 320)}ms`;
    });
  }

  /* --------------------------------- layout -------------------------------- */

  // The part each cell belongs to: a markdown cell with a "## Part N" heading
  // starts part N, and every later cell belongs to it until the next one.
  function partOfCells() {
    let part = null;
    return cells.map((cell) => {
      if (cell.type === "markdown") {
        const heading = cell.source.match(/^##[ \t]+Part[ \t]+(\d+)\b/m);
        if (heading) part = Number(heading[1]);
      }
      return part;
    });
  }

  // The cells in runs that share a part: one group per part, and one for the
  // cells before the first part. The split view puts a group's text on the
  // left and its code on the right.
  function partGroups() {
    const groups = [];
    partOfCells().forEach((part, index) => {
      let group = groups[groups.length - 1];
      if (!group || group.part !== part) {
        group = { part, cells: [], markdown: [], code: [] };
        groups.push(group);
      }
      group.cells.push(index);
      (cells[index].type === "code" ? group.code : group.markdown).push(index);
    });
    return groups;
  }

  // The box drawn around each part, and the line at its foot that says whether
  // the part's self-checks passed. Rebuilt with the layout.
  let partEls = [];

  function partBox(tag, className, part) {
    const box = document.createElement(tag);
    box.className = className;
    if (part === null) return box;
    box.classList.add("editor-part");
    box.dataset.part = String(part);
    box.setAttribute("aria-label", `Part ${part}`);
    partEls.push(box);
    return box;
  }

  function partFoot(box) {
    if (!box.dataset.part) return;
    const foot = document.createElement("p");
    foot.className = "editor-part-status";
    foot.hidden = true;
    box.appendChild(foot);
  }

  // The mock checks the window width itself, since a frame is narrower than
  // the window it sits in.
  function canSplit() {
    return (EMBED || window.innerWidth >= SPLIT_MIN_WIDTH) && cells.some((cell) => cell.type === "markdown");
  }

  // "text" is the mock's Statement view: the text cells alone.
  function effectiveLayout() {
    if (EMBED && storedLayout === "text") return "text";
    return storedLayout === "split" && canSplit() ? "split" : "single";
  }

  let currentLayout = null;

  // Cells are built once and moved between layouts, so edits, outputs and the
  // caret survive a switch.
  function layoutCells(force) {
    const layout = effectiveLayout();
    if (els.split) {
      // In the mock the switch sits in the mock's own toolbar.
      els.split.hidden = EMBED || !canSplit();
      els.split.setAttribute("aria-checked", storedLayout === "split" ? "true" : "false");
    }
    if (layout === currentLayout && !force) return;
    currentLayout = layout;
    const focused = document.activeElement;
    els.cells.innerHTML = "";
    els.cells.classList.toggle("is-split", layout === "split");
    if (els.panel) els.panel.classList.toggle("is-split", layout === "split");
    // Run all, Stop and Reset have nothing to act on while only text shows.
    if (EMBED && els.toolbar) els.toolbar.hidden = layout === "text";
    partEls = [];
    if (layout !== "split") {
      const shown = (index) => layout !== "text" || cells[index].type === "markdown";
      partGroups().forEach((group) => {
        if (group.part === null) {
          group.cells.filter(shown).forEach((index) => els.cells.appendChild(cellEls[index]));
          return;
        }
        const box = partBox("section", "editor-part-card", group.part);
        group.cells.filter(shown).forEach((index) => box.appendChild(cellEls[index]));
        partFoot(box);
        els.cells.appendChild(box);
      });
    } else {
      partGroups().forEach((row) => {
        // A row with no code (the notebook's opening text) reads across both columns.
        const rowEl = partBox("div", row.code.length ? "editor-row" : "editor-row is-text-only", row.part);
        const text = document.createElement("div");
        text.className = "editor-row-text";
        const code = document.createElement("div");
        code.className = "editor-row-code";
        row.markdown.forEach((index) => text.appendChild(cellEls[index]));
        row.code.forEach((index) => code.appendChild(cellEls[index]));
        rowEl.appendChild(text);
        rowEl.appendChild(code);
        partFoot(rowEl);
        els.cells.appendChild(rowEl);
      });
    }
    paintPassed();
    els.cells.querySelectorAll("textarea.editor-code").forEach((textarea) => autoSize(textarea));
    if (focused && els.cells.contains(focused)) focused.focus({ preventScroll: true });
  }

  function chooseLayout(next) {
    storedLayout = next;
    // The mock keeps the Split view choice itself, under the same key.
    if (!EMBED) {
      try {
        window.localStorage.setItem(LAYOUT_KEY, next);
      } catch (error) {
        /* private mode: the choice lasts for this page only */
      }
    }
    layoutCells();
  }

  /* ------------------------------ guided parts ----------------------------- */

  // A guided notebook's self-checks end by printing "checks passed" (or
  // "Part N passed"). A part is passed when every check cell in it finished
  // cleanly in the Python running now and printed that line. When every part
  // with checks has passed, the notebook's problem is marked solved. The page
  // says so in three places: the cell's output, the foot of the part, and the
  // count in the header.
  let partStates = new Map(); // part -> "idle" | "passed" | "failed"

  function paintPassed() {
    partEls.forEach((box) => {
      const part = Number(box.dataset.part);
      const state = partStates.get(part) || "none";
      box.dataset.check = state;
      const foot = box.querySelector(":scope > .editor-part-status");
      if (!foot) return;
      foot.hidden = state !== "passed" && state !== "failed";
      foot.textContent = state === "passed" ? `Part ${part} passed` : state === "failed" ? `Part ${part} did not pass yet` : "";
    });
    const chip = els.meta && els.meta.querySelector("#editorChecks");
    if (!chip) return;
    const total = partStates.size;
    const passed = [...partStates.values()].filter((state) => state === "passed").length;
    const holder = chip.closest(".pp-fact") || chip;
    holder.hidden = !total;
    chip.classList.toggle("is-done", total > 0 && passed === total);
    chip.textContent = `${passed} of ${total} passed`;
  }

  function refreshPassed() {
    const partOf = partOfCells();
    const checks = new Map();
    cells.forEach((cell, index) => {
      const part = partOf[index];
      if (part === null || cell.type !== "code" || !CHECK_CELL.test(cell.original)) return;
      if (!checks.has(part)) checks.set(part, []);
      checks.get(part).push(index);
    });
    const passed = new Set();
    partStates = new Map();
    checks.forEach((indexes, part) => {
      const current = indexes.map((index) => runs.get(index)).map((run) => (run && run.session === session && run.status === "done" ? run : null));
      const ok = current.every((run) => run && !run.error && PASSED_LINE.test(run.output));
      const failed = current.some((run) => run && run.error && run.error !== STOPPED_MESSAGE);
      if (ok) passed.add(part);
      partStates.set(part, ok ? "passed" : failed ? "failed" : "idle");
    });
    paintPassed();
    if (checks.size && passed.size === checks.size && problem && storage) {
      const status = storage.loadState().problemStatus[problem.id];
      if (status !== "complete") {
        storage.setProblemStatus(problem.id, "complete");
        // app.js ticks the header's status circle when it hears this.
        document.dispatchEvent(new CustomEvent("aio-problem-status-set", { detail: { id: problem.id, status: "complete" } }));
      }
    }
  }

  // The first cell run or edit marks the problem in progress. One already in
  // progress or done stays as it is.
  let started = false;
  function markInProgress() {
    if (started || !problem || !storage) return;
    started = true;
    if ((storage.loadState().problemStatus[problem.id] || "not-started") !== "not-started") return;
    storage.setProblemStatus(problem.id, "in-progress");
    document.dispatchEvent(new CustomEvent("aio-problem-status-set", { detail: { id: problem.id, status: "in-progress" } }));
  }

  /* --------------------------- status and stepping --------------------------- */

  // The header's status circle and its previous and next buttons, drawn by
  // the shared renderers. app.js toggles the circle and keeps it in step with
  // storage, and turns the arrow keys into previous and next.
  function renderHeaderControls(module) {
    if (!problem || !renderers) return;
    if (els.check && storage && renderers.statusCircle) {
      const status = storage.loadState().problemStatus[problem.id];
      els.check.outerHTML = renderers.statusCircle(problem.id, problem.title, status, "pp-check");
    }
    if (els.stepper && renderers.renderProblemStepper) {
      const problems = (window.AIO_PROBLEMS || {}).problems || [];
      els.stepper.outerHTML = renderers.renderProblemStepper(problem.id, problems, root, FROM_MODULE);
    }
    // Opened from a module's list, the back link returns to that list.
    if (els.back && FROM_MODULE && module) {
      els.back.href = `${root}module/?id=${encodeURIComponent(module.id)}#lesson-problems`;
      els.back.textContent = `← ${module.title}`;
    }
  }

  /* -------------------------------- execution ------------------------------ */

  // The control the user invoked to start the current run. setBusy(true) disables
  // that control, which would drop focus to <body> on every run; remember it so
  // focus can be restored when the run ends. WCAG 2.1 SC 2.4.3.
  let lastInvoke = null;

  function firstEnabledControl() {
    return els.toolbar ? els.toolbar.querySelector("button:not([disabled])") : null;
  }

  function setBusy(next) {
    const active = document.activeElement;
    busy = next;
    if (els.runAll) els.runAll.disabled = next;
    if (els.reset) els.reset.disabled = next;
    if (els.stop) els.stop.disabled = !next;
    cellEls.forEach((wrap) => {
      const button = wrap.querySelector(".cell-run");
      if (button) button.disabled = next;
    });
    if (next) {
      // Focus leaves the control that was just disabled. Park it on Stop, which
      // was just enabled, so it does not fall back to <body>.
      if (active === els.runAll || active === els.reset || (active && active.classList.contains("cell-run"))) {
        lastInvoke = active;
        if (els.stop) els.stop.focus();
      }
    } else if (active === els.stop) {
      // Stop is disabled now, so focus has to move. The other controls are
      // enabled again, so it can go back to the one the user used, or to the
      // first enabled toolbar button if that one is gone.
      const target = lastInvoke && !lastInvoke.disabled ? lastInvoke : firstEnabledControl();
      if (target) target.focus();
      lastInvoke = null;
    }
  }

  function markRunning(wrap, on) {
    if (!wrap) return;
    wrap.classList.toggle("running", on);
    const label = wrap.querySelector(".cell-run-label");
    if (label) label.textContent = on ? "Running" : "Run";
  }

  // Runs one code cell. Resolves to false when it raised or was stopped.
  async function runCell(index) {
    const cell = cells[index];
    const wrap = cellEls[index];
    if (!cell || cell.type !== "code") return true;
    markInProgress();
    const startedIn = session;
    const run = { status: "running", session: startedIn, count: null, output: "", truncated: false, error: null, figures: [] };
    runs.set(index, run);
    markRunning(wrap, true);
    refreshCounts();
    renderOutput(index);

    let redraw = null;
    let result;
    try {
      result = await kernel.run(cell.source, (event) => {
        if (event.type === "stdout") {
          const all = run.output + event.text;
          if (all.length > OUTPUT_LIMIT) {
            let kept = all.slice(all.length - OUTPUT_LIMIT);
            const lineEnd = kept.indexOf("\n");
            if (all[all.length - OUTPUT_LIMIT - 1] !== "\n" && lineEnd !== -1 && lineEnd < kept.length - 1) {
              kept = kept.slice(lineEnd + 1);
            }
            run.output = kept;
            run.truncated = true;
          } else {
            run.output = all;
          }
        } else if (event.type === "figure") {
          run.figures.push(event.png);
        } else {
          return;
        }
        // A timer, not an animation frame: those never fire in a hidden tab.
        if (redraw === null) {
          redraw = window.setTimeout(() => {
            redraw = null;
            renderOutput(index);
          }, OUTPUT_REFRESH_MS);
        }
      });
    } catch (error) {
      result = { error: String((error && error.message) || error), seconds: 0 };
    }
    if (redraw !== null) window.clearTimeout(redraw);

    // A cell that Python was stopped under did not finish, so has no count.
    const finished = session === startedIn;
    run.status = "done";
    run.error = result.error;
    if (finished) {
      execCounter += 1;
      run.count = execCounter;
    }
    if (result.error === null) {
      restarted = false;
      refreshNotice();
    }
    markRunning(wrap, false);
    refreshCounts();
    renderOutput(index);
    refreshPassed();
    return result.error === null;
  }

  async function runOne(index) {
    if (busy) return;
    setBusy(true);
    try {
      await runCell(index);
    } finally {
      setBusy(false);
    }
  }

  // Runs every code cell in order and stops at the first one that fails.
  async function runAll() {
    if (busy) return;
    setBusy(true);
    try {
      restarted = false;
      refreshNotice();
      for (let index = 0; index < cells.length; index += 1) {
        if (cells[index].type !== "code") continue;
        // eslint-disable-next-line no-await-in-loop
        if (!(await runCell(index))) break;
      }
    } finally {
      setBusy(false);
    }
  }

  /* --------------------------------- actions ------------------------------- */

  function resetNotebook() {
    if (!originalNotebook) return;
    const confirmed = window.confirm("Reset the notebook to the original problem? Your edits in this browser will be discarded.");
    if (!confirmed) return;
    cells.forEach((cell) => {
      cell.source = cell.original;
    });
    try {
      window.localStorage.removeItem(DRAFT_KEY);
    } catch (error) {
      /* ignore */
    }
    runs = new Map();
    restarted = false;
    quietStop = true;
    kernel.stop();
    quietStop = false;
    refreshNotice();
    buildCells();
    layoutCells(true);
    refreshPassed();
  }

  // The notebook file with the reader's code: every cell keeps its metadata,
  // code cells carry the edited source, and the solution stays at the end.
  // In the mock the solution is left out.
  function downloadNotebook() {
    if (!originalNotebook) return;
    const raw = EMBED ? (originalNotebook.cells || []).slice(0, workEnd) : originalNotebook.cells || [];
    let next = 0;
    const out = raw.map((cell, rawIndex) => {
      const isSolution = rawIndex >= workEnd;
      const source = isSolution ? cellSource(cell) : cells[next++].source;
      const base = { cell_type: cell.cell_type === "markdown" ? "markdown" : "code", metadata: cell.metadata || {}, source: source.split(/(?<=\n)/) };
      if (base.cell_type === "code") {
        base.execution_count = null;
        base.outputs = [];
      }
      return base;
    });
    const nb = { nbformat: 4, nbformat_minor: 4, metadata: originalNotebook.metadata || {}, cells: out };
    const blob = new Blob([JSON.stringify(nb, null, 1)], { type: "application/x-ipynb+json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${notebookId || "notebook"}.ipynb`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /* -------------------------------- solution ------------------------------- */

  function renderSolution() {
    if (EMBED || !solutionCells.length) return;
    const fold = document.createElement("details");
    fold.className = "editor-solution-fold";
    const summary = document.createElement("summary");
    summary.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.05 4.55 12.5 10l-5.45 5.45" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><span>Show the Solution</span>';
    fold.appendChild(summary);
    const body = document.createElement("div");
    body.className = "editor-solution-body";
    fold.appendChild(body);
    // Drawn on first opening: until then it is not worth the work.
    fold.addEventListener("toggle", () => {
      summary.querySelector("span").textContent = fold.open ? "Hide the solution" : "Show the solution";
      if (!fold.open || body.childElementCount) return;
      body.innerHTML = solutionCells
        .map((cell, n) => {
          if (cell.type === "markdown") {
            const source = n === 0 ? cell.source.replace(/^#{1,3}\s+Solution[^\n]*\n?/, "") : cell.source;
            return source.trim() ? `<div class="editor-md">${renderMarkdown(source)}</div>` : "";
          }
          return `<pre class="md-code editor-solution-code"><code>${highlightPython(cell.source.replace(/\s+$/, ""))}</code></pre>`;
        })
        .join("");
      typeset(body);
    });
    els.cells.insertAdjacentElement("afterend", fold);
  }

  /* ---------------------------------- boot --------------------------------- */

  function showMissing() {
    if (els.title) els.title.textContent = "Notebook not found";
    if (els.empty) els.empty.hidden = false;
  }

  let workEnd = 0;

  async function boot() {
    if (!notebookId) {
      showMissing();
      return;
    }
    let nb;
    try {
      const response = await fetch(`${root}notebooks/${notebookId}.ipynb${VERSION}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      nb = await response.json();
    } catch (error) {
      showMissing();
      return;
    }

    originalNotebook = nb;
    meta = (nb.metadata && nb.metadata.usaaio) || {};
    const problemData = window.AIO_PROBLEMS || { problems: [] };
    problem = problemData.problems.find((item) => item.url === `/editor/?nb=${notebookId}`) || null;
    const catalog = (window.AIO_CONTENT && window.AIO_CONTENT.modules) || [];
    const module = catalog.find((item) => item.id === meta.module || (problem && item.id === problem.moduleId));
    const title = meta.title || (problem && problem.title) || notebookId;

    if (els.title) els.title.textContent = title;
    if (els.moduleLink) {
      if (module) els.moduleLink.href = `${root}module/?id=${encodeURIComponent(module.id)}`;
      els.moduleLink.hidden = !module;
    }
    document.title = `${title} | USAAIO Guide`;

    const raw = Array.isArray(nb.cells) ? nb.cells : [];
    const start = solutionStart(raw);
    workEnd = start >= 0 ? start : raw.length;
    cells = raw.slice(0, workEnd).map((cell) => {
      const source = cellSource(cell);
      return { type: cell.cell_type === "markdown" ? "markdown" : "code", source, original: source };
    });
    solutionCells = raw.slice(workEnd).map((cell) => ({
      type: cell.cell_type === "markdown" ? "markdown" : "code",
      source: cellSource(cell),
    }));

    // The page title already names the notebook, so a first cell that only
    // repeats it as "# Title" loses that line.
    if (cells[0] && cells[0].type === "markdown") {
      const first = cells[0].source.match(/^#\s+(.+)\n?/);
      if (first && first[1].trim().toLowerCase() === String(title).trim().toLowerCase()) {
        cells[0].display = cells[0].source.slice(first[0].length);
      }
    }

    const draft = loadDraft();
    if (draft) draft.forEach((cell, index) => { cells[index].source = cell.source; });

    if (EMBED) {
      document.body.classList.add("editor-embed");
      // The solution is dropped, and the mock's own toolbar holds the
      // download, which calls downloadNotebook without it.
      solutionCells = [];
      if (els.download) els.download.hidden = true;
    }

    try {
      storedLayout = window.localStorage.getItem(LAYOUT_KEY) === "split" ? "split" : "single";
    } catch (error) {
      storedLayout = "single";
    }
    if (EMBED) storedLayout = EMBED_LAYOUTS.includes(params.get("layout")) ? params.get("layout") : "single";

    if (els.note) {
      if (meta.colab) {
        // A notebook that needs a GPU says why here, and where else to run it.
        els.note.textContent = `${meta.colab} ${COLAB_INSTEAD}`;
      }
      els.note.hidden = false;
    }
    if (els.toolbar) els.toolbar.hidden = false;

    // The header's facts strip, the same one the multiple choice and text
    // pages use: type, module, difficulty, parts, time. The last fact counts
    // the parts whose self-checks have passed, and stays hidden until a
    // notebook with checks has run them.
    if (els.meta) {
      const difficulty = (problem && problem.difficulty) || meta.difficulty;
      const partCount = new Set(partOfCells().filter((part) => part !== null)).size;
      const fact = (label, value, extra = "") => `<div class="pp-fact pp-fact--${label.toLowerCase()}"${extra}><dt>${label}</dt><dd>${value}</dd></div>`;
      const facts = [fact("Type", '<span class="pp-type pp-type--code">Code</span>')];
      if (module) {
        facts.push(fact("Module", `<a class="pp-fact-link" href="${escapeHtml(`${root}module/?id=${encodeURIComponent(module.id)}`)}">${escapeHtml(module.title)}</a>`));
      }
      if (difficulty) {
        const slug = String(difficulty).toLowerCase().replace(/[^a-z]+/g, "-");
        facts.push(fact("Difficulty", `<span class="problem-chip difficulty-${escapeHtml(slug)}">${escapeHtml(difficulty)}</span>`));
      }
      if (partCount) facts.push(fact("Parts", escapeHtml(partCount)));
      if (Number(meta.minutes) > 0) facts.push(fact("Time", `About ${escapeHtml(meta.minutes)} min`));
      facts.push(fact("Checks", '<span class="editor-checks" id="editorChecks"></span>', ' hidden'));
      els.meta.innerHTML = facts.join("");
      els.meta.hidden = false;
    }
    if (!EMBED) renderHeaderControls(module);
    if (els.actions) els.actions.hidden = false;

    buildCells();
    layoutCells(true);
    refreshPassed();
    renderSolution();
    setBusy(false);

    if (els.runAll) els.runAll.addEventListener("click", runAll);
    if (els.stop) els.stop.addEventListener("click", () => kernel.stop());
    if (els.reset) els.reset.addEventListener("click", resetNotebook);
    if (els.download) els.download.addEventListener("click", downloadNotebook);
    if (els.split) els.split.addEventListener("click", () => chooseLayout(storedLayout === "split" ? "single" : "split"));
    window.addEventListener("resize", () => layoutCells());
    // KaTeX loads with defer; typeset anything drawn before it arrived.
    window.addEventListener("load", () => typeset(els.panel));
    if (EMBED) {
      window.AIO_EDITOR_EMBED = {
        setLayout(next) {
          if (EMBED_LAYOUTS.includes(next)) chooseLayout(next);
        },
        download: downloadNotebook,
      };
      document.dispatchEvent(new CustomEvent("aio-editor-ready"));
    }
  }

  boot();
})();
