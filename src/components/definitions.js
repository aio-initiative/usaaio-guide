(function () {
  const entries = window.AIO_DEFINITIONS || [];
  if (!entries.length) return;
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const aliases = new Map(entries.flatMap((entry) => entry.aliases.map((alias) => [alias.toLowerCase(), entry])));
  const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(`(?<![\\p{L}\\p{N}_])(${[...aliases.keys()].sort((a, b) => b.length - a.length).map(escapeRegex).join('|')})(?![\\p{L}\\p{N}_])`, 'giu');
  const root = document.body.dataset.root || '';
  const tip = document.createElement('div');
  tip.id = 'definition-tooltip';
  tip.className = 'definition-tooltip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  const dialog = document.createElement('dialog');
  dialog.className = 'definition-dialog';
  dialog.setAttribute('aria-labelledby', 'definition-title');
  dialog.innerHTML = '<div class="definition-toolbar"><button type="button" data-definition-bank>All Definitions</button><button type="button" data-definition-close aria-label="Close definition">Close <span aria-hidden="true">×</span></button></div><div class="definition-content"></div>';
  document.body.append(tip, dialog);
  const content = dialog.querySelector('.definition-content');
  let lastTrigger = null;
  let tipTrigger = null;
  let hideTimer;
  const escape = (value) => window.AIO_RENDERERS.escapeHtml(value);

  function hideTip() {
    tip.hidden = true;
    tipTrigger?.removeAttribute('aria-describedby');
    tipTrigger = null;
  }
  function showTip(button) {
    if (dialog.open) return;
    clearTimeout(hideTimer);
    hideTip();
    const entry = byId.get(button.dataset.definition);
    if (!entry) return;
    tip.textContent = `${entry.brief} Click for the full definition.`;
    tip.hidden = false;
    tipTrigger = button;
    button.setAttribute('aria-describedby', tip.id);
    // Measured in screen pixels; the desktop page zoom scales style values
    // again, so positions are divided by it.
    const zoom = parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const rect = button.getBoundingClientRect();
    const width = tip.getBoundingClientRect().width;
    tip.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - width - 8)) / zoom}px`;
    const height = tip.getBoundingClientRect().height;
    tip.style.top = `${Math.max(8, rect.bottom + height + 12 < innerHeight ? rect.bottom + 8 : rect.top - height - 8) / zoom}px`;
  }
  function typeset() {
    if (!window.katex) return;
    content.querySelectorAll('[data-tex]').forEach((el) => {
      window.katex.render(el.dataset.tex, el, { displayMode: el.classList.contains('reader-math-display'), throwOnError: false });
    });
  }
  function openDialog(trigger) {
    hideTip();
    if (!dialog.open) {
      lastTrigger = trigger || document.activeElement;
      dialog.showModal();
      document.body.classList.add('definition-open');
    }
    content.scrollTop = 0;
  }
  function showEntry(id, trigger) {
    const entry = byId.get(id);
    if (!entry) return;
    const render = window.AIO_RENDERERS.renderBlocks;
    const lesson = window.AIO_LESSONS?.lessons[entry.moduleId];
    const target = lesson?.sections.find((section) => section.id !== 'prerequisites' && section.id !== 'intro' && JSON.stringify(section.blocks).toLowerCase().includes(entry.aliases[0].toLowerCase())) || lesson?.sections.find((section) => section.id !== 'prerequisites' && section.id !== 'intro');
    const href = `${root}module/?id=${encodeURIComponent(entry.moduleId)}${target ? '#lesson-' + encodeURIComponent(target.id) : ''}`;
    const module = window.AIO_CONTENT?.modules.find((item) => item.id === entry.moduleId);
    content.innerHTML = `<h2 id="definition-title" tabindex="-1">${escape(entry.title)}</h2><p class="definition-brief">${escape(entry.brief)}</p>${render(entry.body, root)}${entry.proof.length ? `<h3>Why it holds</h3>${render(entry.proof, root)}` : ''}<p class="definition-further"><a href="${escape(href)}">Read more in ${escape(module?.title || 'the guide')} →</a></p>`;
    openDialog(trigger);
    typeset();
    content.querySelector('h2').focus();
  }
  function showBank(trigger) {
    content.innerHTML = '<h2 id="definition-title">Definition bank</h2><label class="definition-search-label" for="definition-search">Find a Term</label><input id="definition-search" type="search" placeholder="Try Jensen, gradient, or KL…" autocomplete="off"><p class="definition-count" aria-live="polite"></p><ul class="definition-results"></ul>';
    const input = content.querySelector('input');
    const list = content.querySelector('ul');
    const count = content.querySelector('.definition-count');
    function filter() {
      const query = input.value.trim().toLowerCase();
      const found = entries.filter((entry) => `${entry.title} ${entry.aliases.join(' ')} ${entry.brief}`.toLowerCase().includes(query)).sort((a, b) => a.title.localeCompare(b.title));
      count.textContent = found.length ? `${found.length} ${found.length === 1 ? 'definition' : 'definitions'}` : 'No matching terms. Try another spelling.';
      list.innerHTML = found.map((entry) => `<li><button type="button" data-definition="${escape(entry.id)}">${escape(entry.title)}</button><p>${escape(entry.brief)}</p></li>`).join('');
    }
    input.addEventListener('input', filter);
    filter();
    openDialog(trigger);
    input.focus();
  }

  function annotate(container) {
    if (!container) return;
    // Work only in prose. Links, formulas, headings and executable examples
    // retain their original DOM and semantics. Longest aliases match first.
    const seen = new WeakMap();
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent?.closest('p, li, td, blockquote') || parent.closest('a, button, code, pre, .katex, [data-tex], h1, h2, h3, h4, .definition-dialog, .reader-nav, .reader-prereqs')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const scope = node.parentElement.closest('.reader-section, .theory-part') || container;
      if (!seen.has(scope)) seen.set(scope, new Set());
      const used = seen.get(scope);
      const fragment = document.createDocumentFragment();
      let offset = 0;
      for (const match of node.textContent.matchAll(pattern)) {
        const entry = aliases.get(match[0].toLowerCase());
        if (used.has(entry.id)) continue;
        fragment.append(document.createTextNode(node.textContent.slice(offset, match.index)));
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'definition-term';
        button.dataset.definition = entry.id;
        button.setAttribute('aria-haspopup', 'dialog');
        button.setAttribute('aria-label', `${match[0]}: open definition`);
        button.textContent = match[0];
        fragment.append(button);
        offset = match.index + match[0].length;
        used.add(entry.id);
      }
      if (offset) {
        fragment.append(document.createTextNode(node.textContent.slice(offset)));
        node.replaceWith(fragment);
      }
    }
  }

  document.addEventListener('click', (event) => {
    const term = event.target.closest('[data-definition]');
    if (term) showEntry(term.dataset.definition, term);
    else if (event.target.closest('[data-definition-bank]')) showBank(event.target);
    else if (event.target.closest('[data-definition-close]')) dialog.close();
  });
  dialog.addEventListener('click', (event) => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('definition-open');
    lastTrigger?.isConnected && lastTrigger.focus();
  });
  document.addEventListener('pointerover', (event) => {
    const button = event.target.closest('.definition-term');
    if (button && event.pointerType !== 'touch') showTip(button);
  });
  document.addEventListener('pointerout', (event) => {
    if (event.target.closest('.definition-term') && !tip.contains(event.relatedTarget)) hideTimer = setTimeout(hideTip, 150);
  });
  tip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  tip.addEventListener('pointerleave', hideTip);
  document.addEventListener('focusin', (event) => {
    if (event.target.matches('.definition-term')) showTip(event.target);
  });
  document.addEventListener('focusout', (event) => {
    if (event.target.matches('.definition-term')) hideTip();
  });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') hideTip(); });
  window.addEventListener('scroll', hideTip, true);
  window.addEventListener('resize', hideTip);
  window.AIO_DEFINITION_LINKS = { annotate };
})();
