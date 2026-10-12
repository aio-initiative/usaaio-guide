/*
 * Shared mock-test session state.
 *
 * The Mock Arena spans two pages — /mock/ builds the set, /mock/run/ sits it —
 * so the session lives in localStorage rather than in either page's memory.
 * An absolute `endsAt` timestamp (rather than a remaining duration) is what
 * lets the clock stay honest across the navigation between them.
 */
(function () {
  const KEY = "usaaio-mock-session-v1";
  const DEFAULT_MINUTES = 120;
  const MAX_MINUTES = 600;

  function blank() {
    return {
      status: "setup", // "setup" | "running" | "done"
      minutes: DEFAULT_MINUTES,
      problemIds: [],
      activeIndex: 0,
      startedAt: null,
      endsAt: null,
      finishedAt: null,
    };
  }

  // `knownIds` lets the caller drop problems that no longer exist in the bank,
  // so a stale session cannot render a blank row or crash on lookup.
  function normalize(raw, knownIds) {
    if (!raw || typeof raw !== "object") return blank();
    const minutes = Number(raw.minutes);
    const ids = Array.isArray(raw.problemIds) ? raw.problemIds : [];
    const problemIds = knownIds ? ids.filter((id) => knownIds.has(id)) : ids;
    const activeIndex = Number(raw.activeIndex);
    return {
      status: ["setup", "running", "done"].includes(raw.status) ? raw.status : "setup",
      minutes: Number.isFinite(minutes) && minutes > 0 ? Math.min(Math.round(minutes), MAX_MINUTES) : DEFAULT_MINUTES,
      problemIds,
      activeIndex: Number.isFinite(activeIndex) ? Math.min(Math.max(0, Math.trunc(activeIndex)), Math.max(0, problemIds.length - 1)) : 0,
      startedAt: Number(raw.startedAt) || null,
      endsAt: Number(raw.endsAt) || null,
      finishedAt: Number(raw.finishedAt) || null,
    };
  }

  function load(knownIds) {
    try {
      return normalize(JSON.parse(window.localStorage.getItem(KEY) || "null"), knownIds);
    } catch (error) {
      return blank();
    }
  }

  function save(session) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(session));
    } catch (error) {
      /* private browsing — the session simply will not survive a reload */
    }
  }

  function remainingMs(session) {
    return session && session.endsAt ? Math.max(0, session.endsAt - Date.now()) : 0;
  }

  // "1:04:09" for anything an hour or longer, "04:09" below that.
  function clockText(ms) {
    const total = Math.max(0, Math.round(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }

  window.AIO_MOCK = { KEY, DEFAULT_MINUTES, MAX_MINUTES, blank, normalize, load, save, remainingMs, clockText };
})();
