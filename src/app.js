(function () {
  const content = window.AIO_CONTENT;
  const ordering = window.AIO_ORDERING;
  // Full lesson bodies are large, so content/lessons.js is only loaded by the
  // two pages that read them: the module reader and global search.
  const lessons = (window.AIO_LESSONS || { lessons: {} }).lessons;
  const problemData = window.AIO_PROBLEMS;
  const siteData = window.AIO_SITE_DATA;
  const renderers = window.AIO_RENDERERS;
  const storage = window.AIO_PROGRESS;

  const page = document.body.dataset.page || "home";
  const root = document.body.dataset.root || "";
  const sectionId = document.body.dataset.section || "";
  const params = new URLSearchParams(window.location.search);
  const SUPABASE_URL = "https://tjrkhyvfrqeuasmgaerg.supabase.co";
  const SUPABASE_ANON_KEY =
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRqcmtoeXZmcnFldWFzbWdhZXJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODcyMzgxMDcsImV4cCI6MjEwMjgxNDEwN30._RPC-2UkJsDS70Sc-nWsVjhalShLHgH7qjrz-PnUvnI";
  // Pinned to the exact version the site was built against (read off the page's own network
  // request): jsdelivr resolves the floating "@2" to 2.117.3 today. A floating major means a
  // release we never tested can replace the auth library at any time. The cost of pinning is that
  // security fixes no longer arrive on their own: bump this deliberately and re-test sign-in.
  const SUPABASE_CDN = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2.117.3";
  const AUTH_PROVIDERS = ["google", "github"];

  let authIdentitiesUserId = "";
  let authIdentities = [];
  let authIdentitiesLoading = false;
  let authIdentitiesError = "";

  const state = {
    filter: sectionId || "all",
    problemFilter: "all",
    problemPage: 1,
    problemTypeFilter: problemTypeParam(params.get("type")),
    problemStatusFilter: params.get("status") || "all",
    problemModuleFilter: params.get("module") || "all",
    problemSectionFilter: params.get("section") || "all",
    problemSort: "relevance",
    problemSortDir: "asc",
    problemShuffle: null,
    search: "",
    problemSearch: params.get("q") || "",
    globalSearch: params.get("q") || "",
    searchFilter: params.get("type") || "all",
    selectedModuleId: params.get("id") || "programming-essentials",
    plannerTab: "module",
    plannerSearch: "",
    appState: storage.loadState(),
  };

  const moduleLookup = new Map(content.modules.map((module) => [module.id, module]));
  const problemLookup = new Map(problemData.problems.map((problem) => [problem.id, problem]));

  const els = {
    navToggle: document.querySelector(".nav-toggle"),
    navLinks: document.querySelector("#site-menu"),
    dashboardGrid: document.querySelector("#dashboardGrid"),
    plannerLibrary: document.querySelector("#plannerLibrary"),
    plannerBoard: document.querySelector("#plannerBoard"),
    plannerSearch: document.querySelector("#plannerSearch"),
    plannerCount: document.querySelector("#plannerCount"),
    clearPlan: document.querySelector("#clearPlan"),
    moduleGrid: document.querySelector("#moduleGrid"),
    emptyState: document.querySelector("#emptyState"),
    moduleSearch: document.querySelector("#moduleSearch"),
    filterRow: document.querySelector("#filterRow"),
    sectionProgressCards: document.querySelector("#sectionProgressCards"),
    sectionRoadmap: document.querySelector("#sectionRoadmap"),
    homeStageCards: document.querySelector("#homeStageCards"),
    homeGlanceTable: document.querySelector("#homeGlanceTable"),
    moduleReader: document.querySelector("#moduleReader"),
    problemSearch: document.querySelector("#problemSearch"),
    problemTypeFilter: document.querySelector("#problemTypeFilter"),
    problemStatusFilter: document.querySelector("#problemStatusFilter"),
    problemDifficultyFilter: document.querySelector("#problemDifficultyFilter"),
    problemModuleFilter: document.querySelector("#problemModuleFilter"),
    problemSectionFilter: document.querySelector("#problemSectionFilter"),
    problemSort: document.querySelector("#problemSort"),
    problemTable: document.querySelector("#problemTable"),
    problemPager: document.querySelector("#problemPager"),
    problemPageInfo: document.querySelector("#problemPageInfo"),
    problemEmptyState: document.querySelector("#problemEmptyState"),
    problemStats: document.querySelector("#problemStats"),
    problemTypeTabs: document.querySelector("#problemTypeTabs"),
    globalSearch: document.querySelector("#globalSearch"),
    searchFilterRow: document.querySelector("#searchFilterRow"),
    searchResults: document.querySelector("#searchResults"),
    searchEmptyState: document.querySelector("#searchEmptyState"),
    resourceGrid: document.querySelector("#resourceGrid"),
    seasonTimeline: document.querySelector("#seasonTimeline"),
    checklist: document.querySelector("#contestChecklist"),
    scratchpadInput: document.querySelector("#scratchpadInput"),
    scratchpadSaved: document.querySelector("#scratchpadSaved"),
    sectionTitle: document.querySelector("#sectionTitle"),
    sectionDescription: document.querySelector("#sectionDescription"),
  };

  function orderedModules() {
    const ids = [];
    ordering.sections.forEach((id) => {
      ordering.sectionOrdering[id].chapters.forEach((chapter) => {
        chapter.items.forEach((moduleId) => ids.push(moduleId));
      });
    });
    return ids.map((id) => moduleLookup.get(id)).filter(Boolean);
  }

  const moduleOrder = orderedModules();

  function localPath(path) {
    return `${root}${path}`;
  }

  // aria-current marks the page the reader is on, not just a colour change.
  // A page that lives under a link (a problem under Problems) gets "true".
  function navLink(path, label, pageId, within = false) {
    const exact = page === pageId;
    const active = exact || within ? "active" : "";
    const current = exact ? ` aria-current="page"` : within ? ` aria-current="true"` : "";
    return `<a class="${active}"${current} href="${localPath(path)}">${label}</a>`;
  }

  function navSearchLink() {
    const active = page === "search" ? "active" : "";
    const current = active ? ` aria-current="page"` : "";
    return `
      <a class="nav-search ${active}"${current} href="${localPath("search/")}">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m21 21-4.35-4.35M10.8 18a7.2 7.2 0 1 1 0-14.4 7.2 7.2 0 0 1 0 14.4Z" /></svg>
        <span>Search</span>
      </a>
    `;
  }

  // The planner stays reachable by direct link only; the dashboard is in the nav.
  const SHOW_DASHBOARD = true;
  const SHOW_PLANNER = false;

  function renderTopNav() {
    if (!els.navLinks) return;
    const guideActive = page === "planner" || page === "section";
    const resourcesActive = page === "resources" || page === "mock" || page === "mock-run" || page === "official" || page === "glossary";
    const contactActive = page === "contact";
    els.navLinks.innerHTML = `
      ${SHOW_DASHBOARD ? navLink("dashboard/", "Dashboard", "dashboard") : ""}
      <details class="nav-menu ${guideActive ? "active" : ""}">
        <summary aria-haspopup="true">Sections</summary>
        <div class="nav-menu-panel">
          ${SHOW_PLANNER ? navLink("planner/", "Planner", "planner") : ""}
          ${navLink("round1/", "Round 1", sectionId === "round1" ? "section" : "round1")}
          ${navLink("round2/", "Round 2", sectionId === "round2" ? "section" : "round2")}
          ${navLink("camp/", "Camp", sectionId === "camp" ? "section" : "camp")}
          ${navLink("advanced/", "Advanced", sectionId === "advanced" ? "section" : "advanced")}
        </div>
      </details>
      ${navLink("problems/", "Problems", "problems", page === "editor" || page === "theory")}
      <details class="nav-menu ${resourcesActive ? "active" : ""}">
        <summary aria-haspopup="true">Resources</summary>
        <div class="nav-menu-panel">
          ${navLink("mock/", "Timed Mock", "mock")}
          ${navLink("resources/", "Resource Shelf", "resources")}
          ${navLink("glossary/", "Glossary", "glossary")}
          ${navLink("official/", "USAAIO Links", "official")}
          ${SHOW_PLANNER ? navLink("planner/", "Planner", "planner") : ""}
        </div>
      </details>
      ${navLink("about/", "About", "about")}
      <details class="nav-menu ${contactActive ? "active" : ""}">
        <summary aria-haspopup="true">Contact Us</summary>
        <div class="nav-menu-panel">
          ${navLink("contact/", "Contact Us", "contact")}
          <a href="${localPath("contact/#sponsors")}">Sponsors</a>
        </div>
      </details>
      ${navSearchLink()}
      <span class="nav-auth" id="authNavSlot" aria-live="polite">
        <button class="nav-login-button" type="button" data-auth-open>Login</button>
      </span>
    `;
  }

  function loadSupabaseClient() {
    if (window.supabase && window.supabase.createClient) return Promise.resolve(window.supabase);
    if (window.AIO_SUPABASE_LOADING) return window.AIO_SUPABASE_LOADING;
    window.AIO_SUPABASE_LOADING = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = SUPABASE_CDN;
      script.async = true;
      script.onload = () => {
        if (window.supabase && window.supabase.createClient) resolve(window.supabase);
        else reject(new Error("Supabase client did not load."));
      };
      script.onerror = () => reject(new Error("Could not load Supabase client."));
      document.head.appendChild(script);
    });
    return window.AIO_SUPABASE_LOADING;
  }

  function authRedirectUrl() {
    return new URL(root || "./", window.location.href).href;
  }

  const AUTH_CONTROL_SELECTOR = "#authNavSlot, [data-auth-open], [data-auth-login], [data-auth-logout], [data-auth-status]";

  function displayUser(user) {
    if (!user) return "";
    return (
      user.user_metadata?.user_name ||
      user.user_metadata?.preferred_username ||
      user.user_metadata?.full_name ||
      user.email ||
      "Signed in"
    );
  }

  function authProviderLabel(provider) {
    return provider === "google" ? "Google" : "GitHub";
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, (char) => {
      const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      };
      return entities[char];
    });
  }

  function updateAuthUi(session, statusText) {
    const user = session && session.user;
    window.AIO_AUTH_SESSION = session || null;
    const navSlot = document.querySelector("#authNavSlot");
    if (navSlot) {
      const userLabel = escapeHtml(displayUser(user));
      navSlot.innerHTML = user
        ? `<span class="nav-user"><button class="nav-account-button" type="button" data-auth-open title="Open account settings for ${userLabel}">${userLabel}</button><button type="button" data-auth-logout>Logout</button></span>`
        : `<button class="nav-login-button" type="button" data-auth-open>Login</button>`;
    }

    document.querySelectorAll("[data-auth-when='signed-out']").forEach((item) => {
      item.hidden = Boolean(user);
    });
    document.querySelectorAll("[data-auth-when='signed-in']").forEach((item) => {
      item.hidden = !user;
    });
    document.querySelectorAll("[data-auth-user]").forEach((item) => {
      item.textContent = displayUser(user);
    });
    document.querySelectorAll("[data-auth-status]").forEach((item) => {
      item.textContent = statusText || (user ? "You are signed in." : "You are signed out.");
    });
    if (!user) {
      authIdentitiesUserId = "";
      authIdentities = [];
      authIdentitiesLoading = false;
      authIdentitiesError = "";
    }
    renderAuthIdentities();
  }

  async function getAuthClient() {
    if (window.AIO_AUTH) return window.AIO_AUTH;
    const supabaseLib = await loadSupabaseClient();
    window.AIO_AUTH = supabaseLib.createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    });
    return window.AIO_AUTH;
  }

  function ensureAuthModal() {
    if (document.querySelector("#authModal")) return;
    const modal = document.createElement("section");
    modal.className = "auth-modal";
    modal.id = "authModal";
    modal.hidden = true;
    modal.setAttribute("aria-labelledby", "auth-modal-title");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("role", "dialog");
    modal.innerHTML = `
      <div class="auth-modal-backdrop" data-auth-close></div>
      <div class="auth-modal-card">
        <button class="auth-modal-close" type="button" data-auth-close aria-label="Close sign in">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18" /></svg>
        </button>
        <div class="auth-modal-copy">
          <h2 id="auth-modal-title">Sign In</h2>
          <p>
            Sign in to save your module status, problem progress, planner, and scratchpad across devices. Google and GitHub accounts with the same verified email are automatically connected.
          </p>
        </div>
        <div class="auth-modal-actions" data-auth-when="signed-out">
          <button class="auth-provider-button" type="button" data-auth-login="google">
            <span class="auth-provider-mark auth-provider-mark--google" aria-hidden="true">G</span>
            Continue with Google
          </button>
          <button class="auth-provider-button" type="button" data-auth-login="github">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.69c-2.78.61-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.04 1.53 1.04.89 1.53 2.34 1.09 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.56-1.11-4.56-4.95 0-1.09.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.03A9.56 9.56 0 0 1 12 5.84c.85 0 1.7.11 2.5.33 1.91-1.3 2.75-1.03 2.75-1.03.55 1.38.2 2.4.1 2.65.64.7 1.03 1.6 1.03 2.69 0 3.85-2.34 4.69-4.57 4.94.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
            </svg>
            Continue with GitHub
          </button>
        </div>
        <div class="auth-modal-signed-in" data-auth-when="signed-in" hidden>
          <p>Signed in as <strong data-auth-user></strong>.</p>
          <div class="auth-identities" data-auth-identities></div>
          <p class="auth-link-help">
            Connect the other provider if its email is different. Once connected, either sign-in will open this same account.
          </p>
          <div class="auth-link-actions" data-auth-link-actions></div>
          <button class="button button-secondary dark-text" type="button" data-auth-logout>Log Out</button>
        </div>
        <p class="auth-status" data-auth-status aria-live="polite">Checking sign-in status...</p>
      </div>
    `;
    document.body.appendChild(modal);
  }

  function renderAuthIdentities() {
    const identityList = document.querySelector("[data-auth-identities]");
    const linkActions = document.querySelector("[data-auth-link-actions]");
    const user = window.AIO_AUTH_SESSION?.user;
    if (!identityList || !linkActions) return;
    if (!user) {
      identityList.innerHTML = "";
      linkActions.innerHTML = "";
      linkActions.hidden = true;
      return;
    }

    if (authIdentitiesLoading || authIdentitiesUserId !== user.id) {
      identityList.innerHTML = '<p class="auth-identities-status">Checking connected sign-ins...</p>';
      linkActions.innerHTML = "";
      linkActions.hidden = true;
      return;
    }

    if (authIdentitiesError) {
      identityList.innerHTML = `<p class="auth-identities-status">${escapeHtml(authIdentitiesError)}</p>`;
      linkActions.innerHTML = "";
      linkActions.hidden = true;
      return;
    }

    const connectedProviders = new Set(authIdentities.map((identity) => identity.provider));
    identityList.innerHTML = AUTH_PROVIDERS.map(
      (provider) => `
        <div class="auth-identity-row">
          <span>${authProviderLabel(provider)}</span>
          <span class="auth-identity-state ${connectedProviders.has(provider) ? "is-connected" : ""}">
            ${connectedProviders.has(provider) ? "Connected" : "Not connected"}
          </span>
        </div>
      `,
    ).join("");

    const missingProviders = AUTH_PROVIDERS.filter((provider) => !connectedProviders.has(provider));
    linkActions.innerHTML = missingProviders
      .map(
        (provider) => `
          <button class="auth-link-button" type="button" data-auth-link="${provider}">
            Connect ${authProviderLabel(provider)}
          </button>
        `,
      )
      .join("");
    linkActions.hidden = missingProviders.length === 0;
  }

  async function refreshAuthIdentities(session) {
    const user = session && session.user;
    if (!user) return;
    authIdentitiesLoading = true;
    authIdentitiesError = "";
    renderAuthIdentities();
    try {
      const client = await getAuthClient();
      const { data, error } = await client.auth.getUserIdentities();
      if (error) throw error;
      if (window.AIO_AUTH_SESSION?.user?.id !== user.id) return;
      authIdentities = data?.identities || [];
      authIdentitiesUserId = user.id;
    } catch (error) {
      if (window.AIO_AUTH_SESSION?.user?.id === user.id) {
        authIdentitiesUserId = user.id;
        authIdentitiesError = "Could not load connected sign-ins. You can still use either provider login.";
      }
    } finally {
      authIdentitiesLoading = false;
      renderAuthIdentities();
    }
  }

  function openAuthModal() {
    ensureAuthModal();
    const modal = document.querySelector("#authModal");
    if (!modal) return;
    modal.hidden = false;
    document.body.classList.add("auth-modal-open");
    const firstAction = modal.querySelector("[data-auth-login], [data-auth-link], [data-auth-logout], [data-auth-close]");
    if (firstAction) firstAction.focus();
  }

  function closeAuthModal() {
    const modal = document.querySelector("#authModal");
    if (!modal) return;
    modal.hidden = true;
    document.body.classList.remove("auth-modal-open");
  }

  async function initAuth() {
    if (!document.querySelector(AUTH_CONTROL_SELECTOR)) return;
    ensureAuthModal();
    updateAuthUi(null, "Checking sign-in status...");
    try {
      const client = await getAuthClient();
      const { data, error } = await client.auth.getSession();
      if (error) throw error;
      window.AIO_AUTH_READY = true;
      updateAuthUi(data.session, data.session ? "You are signed in." : "You are signed out.");
      renderAll();
      void storage.syncWithSession(data.session);
      void refreshAuthIdentities(data.session);
      client.auth.onAuthStateChange((_event, session) => {
        window.AIO_AUTH_READY = true;
        updateAuthUi(session, session ? "You are signed in." : "You are signed out.");
        renderAll();
        void storage.syncWithSession(session);
        void refreshAuthIdentities(session);
      });
    } catch (error) {
      updateAuthUi(null, "Could not load login. Check your connection and Supabase settings.");
    }
  }

  function openModule(moduleId) {
    window.location.href = `${localPath("module/")}?id=${encodeURIComponent(moduleId)}`;
  }

  function getStats() {
    const modules = {
      total: content.modules.length,
      complete: 0,
      "in-progress": 0,
      skipped: 0,
      "not-started": 0,
    };
    content.modules.forEach((module) => {
      const status = state.appState.moduleStatus[module.id] || "not-started";
      modules[status] += 1;
    });

    const problems = {
      total: problemData.problems.length,
      complete: 0,
      "in-progress": 0,
      skipped: 0,
      "not-started": 0,
    };
    problemData.problems.forEach((problem) => {
      const status = state.appState.problemStatus[problem.id] || "not-started";
      problems[status] += 1;
    });

    return {
      modules: { ...modules, inProgress: modules["in-progress"] },
      problems,
    };
  }

  function filteredModules() {
    const query = state.search.trim().toLowerCase();
    return moduleOrder.filter((module) => {
      const matchesFilter = state.filter === "all" || module.section === state.filter;
      const haystack =
        `${module.title} ${module.summary} ${module.level} ${module.id}`.toLowerCase();
      return matchesFilter && (!query || haystack.includes(query));
    });
  }

  const DIFFICULTY_ORDER = ["Easy", "Medium", "Hard", "Very Hard", "Mixed"];

  // The three problem UIs: Code, MCQ and Text. The rule lives in renderers.js
  // so the index table, the module lists and this filter always agree.
  const problemType = renderers.problemType;
  const PROBLEM_TYPES = ["Code", "MCQ", "Text"];

  // Reads the ?type= link parameter. Older links used the Coding tag name.
  function problemTypeParam(value) {
    const wanted = String(value || "").toLowerCase();
    if (wanted === "coding") return "Code";
    return ["Code", "MCQ", "Text"].find((type) => type.toLowerCase() === wanted) || "all";
  }

  // Completed first when the Status column sorts ascending.
  const STATUS_RANK = { complete: 0, "in-progress": 1, "not-started": 2 };

  function difficultyRank(difficulty) {
    const wanted = String(difficulty || "").toLowerCase();
    const index = DIFFICULTY_ORDER.findIndex((name) => name.toLowerCase() === wanted);
    return index === -1 ? DIFFICULTY_ORDER.length : index;
  }

  // Every filter except Type. The type tabs count with this, so each tab shows
  // how many problems it would list under the other filters.
  function matchesProblemFilters(problem, query) {
    const module = moduleLookup.get(problem.moduleId);
    // Compared without case, so a stray "Very hard" still lands under Very Hard.
    const matchesDifficulty =
      state.problemFilter === "all" || difficultyRank(problem.difficulty) === difficultyRank(state.problemFilter);
    const matchesModule = state.problemModuleFilter === "all" || problem.moduleId === state.problemModuleFilter;
    const matchesSection =
      state.problemSectionFilter === "all" || (module && module.section === state.problemSectionFilter);
    const status = state.appState.problemStatus[problem.id] || "not-started";
    const matchesStatus = state.problemStatusFilter === "all" || status === state.problemStatusFilter;
    if (!(matchesDifficulty && matchesModule && matchesSection && matchesStatus)) return false;
    if (!query) return true;
    const haystack = `${problem.title} ${problem.prompt} ${problem.source} ${problem.difficulty} ${problem.tags.join(" ")} ${
      module ? module.title : ""
    }`.toLowerCase();
    return haystack.includes(query);
  }

  function filteredProblems() {
    const query = state.problemSearch.trim().toLowerCase();
    const visible = problemData.problems.filter(
      (problem) =>
        (state.problemTypeFilter === "all" || problemType(problem) === state.problemTypeFilter) &&
        matchesProblemFilters(problem, query),
    );
    return sortProblems(visible);
  }

  function sortProblems(problems) {
    // A shuffle overrides the sort dropdown until the next filter change.
    if (state.problemShuffle) {
      const order = state.problemShuffle;
      return [...problems].sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
    }
    if (!state.problemSort || state.problemSort === "relevance") return problems;
    const direction = state.problemSortDir === "desc" ? -1 : 1;
    return [...problems].sort((a, b) => {
      const rankA = problemRank(a);
      const rankB = problemRank(b);
      if (typeof rankA === "string" || typeof rankB === "string") {
        return String(rankA).localeCompare(String(rankB)) * direction;
      }
      return (rankA - rankB) * direction;
    });
  }

  // Where each module sits in the guide. The Module and Section columns sort
  // in this order, so a sorted list reads Round 1 to Advanced like the rest of
  // the site.
  const moduleRank = new Map(moduleOrder.map((module, index) => [module.id, index]));

  // The value one row sorts on for the column the reader picked. Strings
  // compare with localeCompare, numbers with a plain subtraction.
  function problemRank(problem) {
    const module = moduleLookup.get(problem.moduleId);
    switch (state.problemSort) {
      case "title":
        return problem.title;
      case "type":
        return problemType(problem);
      case "difficulty":
        return difficultyRank(problem.difficulty);
      case "module":
        return moduleRank.get(problem.moduleId) ?? moduleRank.size;
      case "section": {
        const index = module ? ordering.sections.indexOf(module.section) : -1;
        return index === -1 ? ordering.sections.length : index;
      }
      case "status":
        return STATUS_RANK[state.appState.problemStatus[problem.id]] ?? STATUS_RANK["not-started"];
      default:
        return 0;
    }
  }

  function scoreSearchResult(query, title, body) {
    if (!query) return 1;
    const normalizedTitle = title.toLowerCase();
    const normalizedBody = body.toLowerCase();
    if (normalizedTitle === query) return 100;
    if (normalizedTitle.includes(query)) return 60;
    if (normalizedBody.includes(query)) return 25;
    return 0;
  }

  // Flattening every lesson block on each keystroke is wasteful, so the plain
  // text of a section is computed once and cached on the section object.
  const lessonTextCache = new WeakMap();

  function blockText(block) {
    if (!block) return "";
    if (block.p) return block.p;
    if (block.heading) return block.heading;
    if (block.tex) return block.tex;
    if (Array.isArray(block.list)) return block.list.join(" ");
    if (Array.isArray(block.olist)) return block.olist.join(" ");
    if (Array.isArray(block.quote)) return block.quote.join(" ");
    if (block.table) {
      return [...(block.table.head || []), ...(block.table.rows || []).flat()].join(" ");
    }
    if (block.code) return Array.isArray(block.code.lines) ? block.code.lines.join(" ") : "";
    if (block.details) return block.details.blocks.map(blockText).join(" ");
    if (block.optional) return `${block.optional.title || ""} ${block.optional.blocks.map(blockText).join(" ")}`;
    if (block.figure) return block.figure.caption || block.figure.alt || "";
    if (Array.isArray(block.resources)) return block.resources.map((entry) => `${entry.title} ${entry.source || ""}`).join(" ");
    if (block.notebookCard) return `${block.notebookCard.title || ""} ${block.notebookCard.description || ""}`;
    return "";
  }

  function lessonSectionText(section) {
    const cached = lessonTextCache.get(section);
    if (cached) return cached;
    const legacy = [
      Array.isArray(section.body) ? section.body.join(" ") : section.body || "",
      Array.isArray(section.keyIdeas) ? section.keyIdeas.join(" ") : "",
      Array.isArray(section.steps) ? section.steps.join(" ") : "",
      section.code && Array.isArray(section.code.lines) ? section.code.lines.join(" ") : "",
      section.note || "",
    ].join(" ");
    const text = `${section.title} ${(section.blocks || []).map(blockText).join(" ")} ${legacy}`.trim();
    lessonTextCache.set(section, text);
    return text;
  }

  // Search results show a plain-text preview, so the Markdown markers the
  // reader would turn into emphasis, code, and math are stripped instead.
  function plainText(text) {
    return String(text || "")
      .replace(/\[([^\]]+)\]\([^)\s]+\)/g, "$1")
      .replace(/[*`$]/g, "")
      .trim();
  }

  function lessonSectionExcerpt(section) {
    const firstParagraph = (section.blocks || []).find((block) => block && block.p);
    if (firstParagraph) return plainText(firstParagraph.p);
    return Array.isArray(section.body) ? section.body[0] || "" : section.body || "";
  }

  function buildSearchResults() {
    const query = state.globalSearch.trim().toLowerCase();
    const results = [];

    moduleOrder.forEach((module) => {
      const section = ordering.sectionOrdering[module.section];
      const body = `${module.summary} ${module.level} ${module.prerequisites.join(" ")}`;
      const score = scoreSearchResult(query, module.title, body);
      if (score) {
        results.push({
          type: "module",
          title: module.title,
          meta: searchMeta(section.label, module.level),
          body: module.summary,
          href: `${localPath("module/")}?id=${encodeURIComponent(module.id)}`,
          score,
        });
      }
    });

    Object.entries(lessons).forEach(([moduleId, lesson]) => {
      const module = moduleLookup.get(moduleId);
      if (!module) return;
      lesson.sections.forEach((section) => {
        const body = `${lessonSectionText(section)} ${module.title}`;
        const score = scoreSearchResult(query, `${module.title}: ${section.title}`, body);
        if (score) {
          results.push({
            type: "lesson",
            title: section.title,
            meta: module.title,
            body: lessonSectionExcerpt(section),
            href: `${localPath("module/")}?id=${encodeURIComponent(module.id)}#lesson-${encodeURIComponent(section.id)}`,
            score: Math.max(1, score - 5),
          });
        }
      });
    });

    problemData.problems.forEach((problem) => {
      const module = moduleLookup.get(problem.moduleId);
      const body = `${problem.prompt} ${problem.answer} ${problem.source} ${problem.difficulty} ${problem.tags.join(" ")} ${
        module ? module.title : ""
      }`;
      const score = scoreSearchResult(query, problem.title, body);
      if (score) {
        results.push({
          type: "problem",
          title: problem.title,
          meta: searchMeta(problem.difficulty, problem.source),
          body: problem.prompt,
          href: `${localPath("problems/")}?q=${encodeURIComponent(problem.title)}`,
          score,
        });
      }
    });

    siteData.resources.forEach((resource) => {
      const body = `${resource.summary} ${resource.type} ${resource.href}`;
      const score = scoreSearchResult(query, resource.title, body);
      if (score) {
        results.push({
          type: "resource",
          title: resource.title,
          meta: resource.type,
          body: resource.summary,
          href: resource.href,
          external: true,
          score: Math.max(1, score - 10),
        });
      }
    });

    return results
      .filter((result) => state.searchFilter === "all" || result.type === state.searchFilter)
      .sort((left, right) => right.score - left.score || left.title.localeCompare(right.title))
      .slice(0, 40);
  }

  // The meta line under a search result. An Advanced module has the level
  // "Advanced" too, so a repeated part is shown once.
  function searchMeta(...parts) {
    return [...new Set(parts.filter(Boolean))].join(" · ");
  }

  function findNeighbors(moduleId) {
    const index = moduleOrder.findIndex((module) => module.id === moduleId);
    return {
      prev: index > 0 ? moduleOrder[index - 1] : null,
      next: index >= 0 && index < moduleOrder.length - 1 ? moduleOrder[index + 1] : null,
    };
  }

  function defaultLessonFor(module) {
    return {
      sections: [
        { id: "overview", title: "Overview", body: module.summary },
      ],
    };
  }

  function renderFilters() {
    if (els.filterRow) {
      els.filterRow.innerHTML = renderers.renderFilterChips(ordering, state.filter);
    }
    if (els.searchFilterRow) {
      els.searchFilterRow.innerHTML = renderers.renderSearchFilters(state.searchFilter);
    }
    if (els.problemStatusFilter) {
      els.problemStatusFilter.value = state.problemStatusFilter;
    }
    if (els.problemTypeFilter) {
      els.problemTypeFilter.value = state.problemTypeFilter;
    }
    if (els.problemDifficultyFilter) {
      // One option per difficulty the bank uses, in the order of DIFFICULTY_ORDER.
      const used = new Set(problemData.problems.map((problem) => difficultyRank(problem.difficulty)));
      const difficulties = DIFFICULTY_ORDER.filter((name, index) => used.has(index));
      els.problemDifficultyFilter.innerHTML =
        `<option value="all">All Difficulties</option>` +
        difficulties.map((item) => optionTag(item, item, state.problemFilter)).join("");
    }
    if (els.problemModuleFilter) {
      const moduleIds = [...new Set(problemData.problems.map((problem) => problem.moduleId))];
      const options = moduleIds
        .map((id) => ({ id, title: moduleLookup.get(id)?.title || id }))
        .sort((a, b) => a.title.localeCompare(b.title));
      els.problemModuleFilter.innerHTML =
        `<option value="all">All Modules</option>` +
        options.map((item) => optionTag(item.id, item.title, state.problemModuleFilter)).join("");
    }
    if (els.problemSectionFilter) {
      els.problemSectionFilter.innerHTML =
        `<option value="all">All Sections</option>` +
        ordering.sections
          .map((id) => optionTag(id, ordering.sectionOrdering[id].label, state.problemSectionFilter))
          .join("");
    }
    if (els.problemSort) {
      els.problemSort.value = state.problemSort;
    }
  }

  function optionTag(value, label, selected) {
    return `<option value="${renderers.escapeHtml(value)}" ${value === selected ? "selected" : ""}>${renderers.escapeHtml(
      label,
    )}</option>`;
  }

  function renderDashboard() {
    if (!els.dashboardGrid) return;
    const stats = getStats();
    const nextModules = moduleOrder
      .filter((module) => (state.appState.moduleStatus[module.id] || "not-started") !== "complete")
      .slice(0, 3);
    els.dashboardGrid.innerHTML = renderers.renderDashboard(
      stats,
      nextModules,
      state.appState,
      {
        moduleLookup,
        problemLookup,
        moduleOrder,
        ordering,
        root,
      },
    );
    const activityScroll = els.dashboardGrid.querySelector(".activity-scroll");
    if (activityScroll) activityScroll.scrollLeft = activityScroll.scrollWidth;
  }

  function renderModules() {
    if (!els.moduleGrid) return;
    const visible = filteredModules();
    els.moduleGrid.innerHTML = visible
      .map((module) => {
        const sectionLabel = ordering.sectionOrdering[module.section].label;
        return renderers.renderModuleCard(module, sectionLabel, state.appState);
      })
      .join("");
    if (els.emptyState) els.emptyState.hidden = visible.length > 0;
  }

  function renderSectionPage() {
    if (!sectionId || !ordering.sectionOrdering[sectionId]) return;
    const section = ordering.sectionOrdering[sectionId];
    if (els.sectionTitle) els.sectionTitle.textContent = section.label;
    if (els.sectionDescription) els.sectionDescription.textContent = section.description;
    if (els.sectionProgressCards) {
      els.sectionProgressCards.innerHTML = renderers.renderSectionProgress(
        sectionId,
        ordering,
        moduleLookup,
        problemData.problems,
        state.appState,
        root,
      );
    }
    if (els.sectionRoadmap) {
      els.sectionRoadmap.innerHTML = renderers.renderSectionTimeline(sectionId, ordering, moduleLookup, state.appState, root);
    }
    renderModules();
  }

  function renderHomePage() {
    if (page !== "home") return;
    if (els.homeStageCards) {
      els.homeStageCards.innerHTML = renderers.renderStageCards(
        ordering,
        moduleLookup,
        problemData.problems,
        state.appState,
        root,
      );
    }
    if (els.homeGlanceTable) {
      els.homeGlanceTable.innerHTML = renderers.renderGlanceTable(
        ordering,
        moduleLookup,
        problemData.problems,
        state.appState,
        root,
      );
    }
  }

  function plannerContext() {
    return {
      moduleLookup,
      problemLookup,
      moduleOrder,
      ordering,
      problems: problemData.problems,
      root,
    };
  }

  function renderPlannerLibrary() {
    if (!els.plannerLibrary) return;
    els.plannerLibrary.innerHTML = renderers.renderPlannerLibrary(
      state.plannerTab,
      state.plannerSearch,
      state.appState.plan,
      plannerContext(),
    );
  }

  function renderPlanner() {
    if (!els.plannerBoard) return;
    renderPlannerLibrary();
    els.plannerBoard.innerHTML = renderers.renderPlannerBoard(state.appState.plan, plannerContext());
    if (els.plannerCount) {
      const count = state.appState.plan.length;
      els.plannerCount.textContent = count ? `${count} block${count === 1 ? "" : "s"}` : "";
    }
    if (els.clearPlan) els.clearPlan.hidden = state.appState.plan.length === 0;
  }

  const READER_NAV_KEY = "usaaio-reader-nav-collapsed";

  function readerNavCollapsed() {
    // On a phone the sidebar stacks above the lesson, so open it would put a
    // screen of links before the title. There it starts closed every time.
    if (window.matchMedia && window.matchMedia("(max-width: 760px)").matches) return true;
    try {
      return localStorage.getItem(READER_NAV_KEY) === "1";
    } catch (error) {
      return false;
    }
  }

  function renderReaderMath() {
    if (!window.katex || !els.moduleReader) return;
    const typeset = (selector, displayMode) => {
      els.moduleReader.querySelectorAll(selector).forEach((element) => {
        window.katex.render(element.getAttribute("data-tex"), element, {
          displayMode,
          throwOnError: false,
        });
      });
    };
    typeset(".reader-math-display[data-tex]", true);
    typeset(".reader-math-inline[data-tex]", false);
  }

  // The reader is re-rendered whole when sign-in or progress sync finishes,
  // often while the page is being read. These two keep what the reader had
  // open or answered: every <details> by position, each question's pick, and
  // focus on a question's option.
  function captureReaderUi() {
    if (!els.moduleReader || !els.moduleReader.firstElementChild) return null;
    const picks = {};
    els.moduleReader.querySelectorAll("[data-mcq]").forEach((card) => {
      const picked = [...card.querySelectorAll(".mcq-option")].findIndex((option) => option.classList.contains("is-picked"));
      if (picked >= 0) picks[card.dataset.mcq] = picked;
    });
    const active = document.activeElement;
    const choice = active && els.moduleReader.contains(active) ? active.closest("[data-mcq-choice]") : null;
    const navScroll = document.querySelector(".reader-nav-scroll");
    return {
      open: [...els.moduleReader.querySelectorAll("details")].map((details) => details.open),
      picks,
      focus: choice ? { card: choice.closest("[data-mcq]").dataset.mcq, index: choice.dataset.mcqChoice } : null,
      navScroll: navScroll ? navScroll.scrollTop : 0,
    };
  }

  function restoreReaderUi(saved) {
    if (!saved) return;
    const all = els.moduleReader.querySelectorAll("details");
    if (all.length === saved.open.length) all.forEach((details, index) => { details.open = saved.open[index]; });
    els.moduleReader.querySelectorAll("[data-mcq]").forEach((card) => {
      const option = card.querySelectorAll(".mcq-option")[saved.picks[card.dataset.mcq]];
      if (option) showPick(card, option);
    });
    if (saved.focus) {
      const card = els.moduleReader.querySelector(`[data-mcq="${CSS.escape(saved.focus.card)}"]`);
      const choice = card && card.querySelector(`[data-mcq-choice="${CSS.escape(saved.focus.index)}"]`);
      if (choice) choice.focus({ preventScroll: true });
    }
    const navScroll = document.querySelector(".reader-nav-scroll");
    if (navScroll && saved.navScroll != null) navScroll.scrollTop = saved.navScroll;
  }

  /* ----------------------------- figure zoom ------------------------------
     A lesson figure drawn smaller than its own size (a phone, a narrow
     window) opens at that size over the page when tapped or clicked. Nothing
     is added to the page: the figure only gains a zoom-in cursor, a tab stop
     and its role. Tap, Escape or Back closes it, and focus returns to it. */
  let figureDialog = null;
  let figureTrigger = null;
  let figureSrc = "";
  let panStart = null;

  function markZoomableFigures() {
    if (!els.moduleReader) return;
    els.moduleReader.querySelectorAll(".lesson-figure img").forEach((img) => {
      const drawn = Number(img.getAttribute("width")) || img.naturalWidth;
      const zoomable = Boolean(drawn) && img.clientWidth > 0 && img.clientWidth < drawn - 1;
      img.closest(".lesson-figure").classList.toggle("is-zoomable", zoomable);
      if (zoomable) {
        img.tabIndex = 0;
        img.setAttribute("role", "button");
        img.setAttribute("aria-label", `Enlarge: ${img.alt}`);
      } else if (img.hasAttribute("role")) {
        img.removeAttribute("tabindex");
        img.removeAttribute("role");
        img.removeAttribute("aria-label");
      }
    });
  }

  function closeFigure(fromHistory) {
    if (!figureDialog || !figureDialog.open) return;
    figureDialog.close();
    if (!fromHistory && window.history.state && window.history.state.figureZoom) window.history.back();
  }

  function openFigure(img) {
    if (!figureDialog) {
      figureDialog = document.createElement("dialog");
      figureDialog.className = "figure-zoom";
      figureDialog.setAttribute("aria-label", "Enlarged figure");
      // A tap that pans the enlarged image scrolls the dialog, so it is not a
      // tap on the backdrop: the click a drag dispatches must not close it.
      figureDialog.addEventListener("pointerdown", () => {
        panStart = { x: figureDialog.scrollLeft, y: figureDialog.scrollTop };
      });
      figureDialog.addEventListener("click", () => {
        // panStart is only set by a pointerdown, so a click that came from the
        // keyboard would otherwise be judged by a pan that has already ended.
        const moved = panStart && (figureDialog.scrollLeft !== panStart.x || figureDialog.scrollTop !== panStart.y);
        panStart = null;
        if (moved) return;
        closeFigure(false);
      });
      figureDialog.addEventListener("cancel", (event) => {
        event.preventDefault();
        closeFigure(false);
      });
      figureDialog.addEventListener("close", () => {
        // A re-render while the overlay was open detached the trigger img, so
        // fall back to the same figure in the new DOM to keep the reader's place.
        const target =
          figureTrigger && figureTrigger.isConnected
            ? figureTrigger
            : els.moduleReader && els.moduleReader.querySelector(`.lesson-figure.is-zoomable img[src="${CSS.escape(figureSrc)}"]`);
        if (target) target.focus({ preventScroll: true });
      });
      window.addEventListener("popstate", () => closeFigure(true));
      document.body.appendChild(figureDialog);
    }
    figureTrigger = img;
    figureSrc = img.currentSrc || img.src;
    const width = Number(img.getAttribute("width")) || img.naturalWidth;
    const height = Number(img.getAttribute("height")) || 0;
    figureDialog.innerHTML = `<img src="${renderers.escapeHtml(figureSrc)}" alt="${renderers.escapeHtml(img.alt)}" style="width: ${Math.round(width)}px"${height ? ` width="${Math.round(width)}" height="${Math.round(height)}"` : ""} />`;
    figureDialog.showModal();
    figureDialog.scrollLeft = (figureDialog.scrollWidth - figureDialog.clientWidth) / 2;
    window.history.pushState({ figureZoom: true }, "");
  }

  if (page === "module" && els.moduleReader) {
    // A figure inside a collapsed <Optional> has no width until the box opens,
    // so the zoomable mark is taken again when one does.
    els.moduleReader.addEventListener(
      "toggle",
      (event) => {
        if (event.target.matches("details")) markZoomableFigures();
      },
      true,
    );
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(markZoomableFigures, 150);
    });
    document.addEventListener("keydown", (event) => {
      const img = event.target.closest && event.target.closest(".lesson-figure.is-zoomable img");
      if (img && (event.key === "Enter" || event.key === " ")) {
        event.preventDefault();
        openFigure(img);
      }
    });
  }

  function renderReader() {
    if (!els.moduleReader) return;
    const savedUi = captureReaderUi();
    const module = moduleLookup.get(state.selectedModuleId) || moduleOrder[0];
    state.selectedModuleId = module.id;
    const currentStatus = state.appState.moduleStatus[module.id] || "not-started";
    if (page === "module" && currentStatus === "not-started") {
      state.appState = storage.setModuleStatus(module.id, "in-progress");
    }
    const sectionLabel = ordering.sectionOrdering[module.section].label;
    const lesson = lessons[module.id] || defaultLessonFor(module);
    els.moduleReader.classList.toggle("nav-collapsed", readerNavCollapsed());
    els.moduleReader.innerHTML = renderers.renderModuleReader(module, lesson, {
      sectionLabel,
      appState: state.appState,
      problems: problemData.problems,
      ordering,
      moduleLookup,
      prev: findNeighbors(module.id).prev,
      next: findNeighbors(module.id).next,
      root,
    });
    renderReaderMath();
    restoreReaderUi(savedUi);
    markZoomableFigures();
    scrollSidebarToActive();
    window.AIO_DEFINITION_LINKS?.annotate(els.moduleReader.querySelector('.reader-article'));
    setPageMeta(`${module.title} | USAAIO Guide`, module.summary);
    scrollToHashOnce();
  }

  // The lesson is drawn by script after the page loads, so the browser's own
  // jump to "#lesson-..." finds nothing. Do it once, after the first render;
  // later re-renders (sign-in, progress sync) must not yank the reader back.
  let hashScrolled = false;
  function scrollToHashOnce() {
    if (hashScrolled || !location.hash) return;
    hashScrolled = true;
    let target = null;
    try {
      target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    } catch (error) {
      return;
    }
    if (!target) return;
    target.scrollIntoView({ block: "start" });
    // Figures load after the text and push the section down, so settle once
    // more when they are in.
    if (document.readyState !== "complete") {
      window.addEventListener("load", () => target.scrollIntoView({ block: "start" }), { once: true });
    }
  }

  /**
   * A step chapter can carry 13 or more modules, and the sidebar starts scrolled
   * to the top on every page load. Without this, a reader deep in the chapter
   * sees the top of the list instead of their own module. Only the sidebar's own
   * scroll container moves, and only when the item is actually out of view.
   */
  function scrollSidebarToActive() {
    const scroll = document.querySelector(".reader-nav-scroll");
    const item = scroll && scroll.querySelector("li.active");
    if (!scroll || !item) return;
    const box = scroll.getBoundingClientRect();
    const rect = item.getBoundingClientRect();
    if (rect.bottom <= box.bottom - 8 && rect.top >= box.top + 8) return;
    const zoom = parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    scroll.scrollTop += (rect.top + rect.height / 2 - (box.top + box.height / 2)) / zoom;
  }

  /**
   * The 55 modules and every theory problem share one URL each
   * (/module/?id=... and /theory/?id=...), so the title and description in the
   * markup describe none of them. Update both, plus the social tags, once the
   * page knows which one it is rendering.
   */
  function setPageMeta(title, description) {
    document.title = title;
    const set = (selector, value) => {
      const el = document.head.querySelector(selector);
      if (el && value) el.setAttribute("content", value);
    };
    set('meta[name="description"]', description);
    // The markup's canonical points at /module/ with no query, which would tell
    // search engines all 55 modules are the same page. Point it at this one.
    const canonical = document.head.querySelector('link[rel="canonical"]');
    if (canonical) canonical.setAttribute("href", `https://www.aioinitiative.com${window.location.pathname}${window.location.search}`);
    set('meta[property="og:url"]', `https://www.aioinitiative.com${window.location.pathname}${window.location.search}`);
    set('meta[property="og:title"]', title);
    set('meta[property="og:description"]', description);
    set('meta[name="twitter:title"]', title);
    set('meta[name="twitter:description"]', description);
  }

  function notebookSource(lines) {
    return lines.map((line, index) => `${line}${index === lines.length - 1 ? "" : "\n"}`);
  }

  function moduleNotebookFilename(module) {
    const base = `${module.code || module.id}-problems`
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return `${base || "module-problems"}.ipynb`;
  }

  function moduleProblemNotebook(module, problems) {
    const cells = [
      {
        cell_type: "markdown",
        metadata: {},
        source: notebookSource([
          `# ${module.code || ""}${module.code ? ": " : ""}${module.title} Problems`,
          "",
          "This notebook contains the practice problems attached to this USAAIO Guide module.",
        ]),
      },
    ];

    problems.forEach((problem, index) => {
      const lines = [
        `## Problem ${index + 1}: ${problem.title}`,
        "",
        `**Difficulty:** ${problem.difficulty}`,
        `**Source:** ${problem.source}`,
      ];
      if (problem.url && !problem.url.startsWith("/editor/")) lines.push(`**Link:** ${problem.url}`);
      if (problem.prompt) lines.push("", problem.prompt);
      cells.push({ cell_type: "markdown", metadata: {}, source: notebookSource(lines) });
      if (problem.tags.includes("Coding") || problem.tags.includes("Notebook") || problem.url?.includes("/editor/")) {
        cells.push({
          cell_type: "code",
          execution_count: null,
          metadata: {},
          outputs: [],
          source: notebookSource(["# Write your solution here"]),
        });
      }
    });

    return {
      cells,
      metadata: {
        kernelspec: {
          display_name: "Python 3",
          language: "python",
          name: "python3",
        },
        language_info: {
          name: "python",
          pygments_lexer: "ipython3",
        },
      },
      nbformat: 4,
      nbformat_minor: 5,
    };
  }

  function downloadModuleNotebook(moduleId) {
    const module = moduleLookup.get(moduleId);
    if (!module) return;
    const problems = problemData.problems.filter((problem) => problem.moduleId === module.id);
    if (!problems.length) return;
    const notebook = moduleProblemNotebook(module, problems);
    const blob = new Blob([JSON.stringify(notebook, null, 1)], { type: "application/x-ipynb+json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = moduleNotebookFilename(module);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
  }

  const PROBLEM_PAGE_SIZE = 20;

  function renderProblems() {
    if (!els.problemTable) return;
    const visible = filteredProblems();
    const totalPages = Math.max(1, Math.ceil(visible.length / PROBLEM_PAGE_SIZE));
    state.problemPage = Math.min(Math.max(1, state.problemPage), totalPages);
    const start = (state.problemPage - 1) * PROBLEM_PAGE_SIZE;
    const pageItems = visible.slice(start, start + PROBLEM_PAGE_SIZE);
    els.problemTable.innerHTML = renderers.renderProblemCards(
      pageItems,
      moduleLookup,
      state.appState,
      root,
      ordering.sectionOrdering,
      // A shuffle overrides the sort, so no header shows as the sorted one.
      state.problemShuffle ? {} : { key: state.problemSort, dir: state.problemSortDir },
    );
    if (els.problemPager) {
      els.problemPager.innerHTML = renderers.renderPager(state.problemPage, totalPages);
      els.problemPager.hidden = visible.length === 0;
    }
    if (els.problemPageInfo) {
      els.problemPageInfo.textContent = visible.length
        ? `Showing ${start + 1} to ${Math.min(start + PROBLEM_PAGE_SIZE, visible.length)} of ${visible.length} problems`
        : "";
    }
    if (els.problemEmptyState) els.problemEmptyState.hidden = visible.length > 0;
    // With nothing to list, the box shows the tabs and the empty note only.
    els.problemTable.hidden = visible.length === 0;
    els.problemTable.closest(".problem-table-wrap")?.toggleAttribute("data-empty", visible.length === 0);
    if (els.problemStats) renderProblemStats();
    if (els.problemTypeTabs) renderProblemTypeTabs();
  }

  // Progress under the lede, as bars: everything completed out of the whole
  // bank, then each type out of its own count. The counts come from the bank,
  // so they follow it as problems are added.
  function renderProblemStats() {
    const tally = { all: { done: 0, total: 0 } };
    PROBLEM_TYPES.forEach((type) => {
      tally[type] = { done: 0, total: 0 };
    });
    problemData.problems.forEach((problem) => {
      const done = state.appState.problemStatus[problem.id] === "complete";
      [tally.all, tally[problemType(problem)]].forEach((entry) => {
        entry.total += 1;
        if (done) entry.done += 1;
      });
    });
    const share = (entry) => (entry.total ? (entry.done / entry.total) * 100 : 0);
    const bar = (entry, label) => `
      <span class="pprog-track" role="progressbar" aria-label="${label}" aria-valuemin="0" aria-valuemax="${entry.total}" aria-valuenow="${entry.done}">
        <span class="pprog-fill" style="width: ${share(entry).toFixed(2)}%"></span>
      </span>`;
    const typeRow = (type) => `
      <div class="pprog-type pprog-type--${type.toLowerCase()}">
        <div class="pprog-line">
          <span class="pprog-label">${type}</span>
          <span class="pprog-count"><strong>${tally[type].done}</strong>/${tally[type].total}</span>
        </div>
        ${bar(tally[type], `${type} problems completed`)}
      </div>`;
    els.problemStats.innerHTML = `
      <div class="pprog-total">
        <div class="pprog-line">
          <span class="pprog-label">Completed</span>
          <span class="pprog-count"><strong>${tally.all.done}</strong> of ${tally.all.total}</span>
          <span class="pprog-pct">${tally.all.done && share(tally.all) < 1 ? "&lt;1" : Math.round(share(tally.all))}%</span>
        </div>
        ${bar(tally.all, "Problems completed")}
      </div>
      <div class="pprog-types">${PROBLEM_TYPES.map(typeRow).join("")}</div>`;
  }

  // The split by type, as tabs on top of the table. Each count is what the tab
  // would list under the other filters and the search.
  function renderProblemTypeTabs() {
    const query = state.problemSearch.trim().toLowerCase();
    const counts = { all: 0, Code: 0, MCQ: 0, Text: 0 };
    problemData.problems.forEach((problem) => {
      if (!matchesProblemFilters(problem, query)) return;
      counts.all += 1;
      counts[problemType(problem)] += 1;
    });
    const tab = (value, label) => {
      const active = state.problemTypeFilter === value;
      return `
        <button class="ptabs-tab${active ? " active" : ""}" type="button" data-problem-type="${value}" aria-pressed="${active ? "true" : "false"}">
          <span class="ptabs-label">${label}</span>
          <span class="ptabs-count">${counts[value]}</span>
        </button>`;
    };
    els.problemTypeTabs.innerHTML = tab("all", "All") + PROBLEM_TYPES.map((type) => tab(type, type)).join("");
  }

  // Jumps to the page holding a problem and flashes its card (used by "Random"
  // when the pick has no link to open).
  function highlightProblem(problemId, pool) {
    const index = pool.findIndex((problem) => problem.id === problemId);
    if (index < 0) return;
    state.problemPage = Math.floor(index / PROBLEM_PAGE_SIZE) + 1;
    renderProblems();
    const card = els.problemTable?.querySelector(`[data-problem-row="${problemId}"]`);
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1600);
  }

  function renderSearch() {
    if (!els.searchResults) return;
    const results = buildSearchResults();
    els.searchResults.innerHTML = renderers.renderSearchResults(results);
    if (els.searchEmptyState) els.searchEmptyState.hidden = results.length > 0;
  }

  function renderStatic() {
    if (els.resourceGrid) els.resourceGrid.innerHTML = renderers.renderResources(siteData.resources);
    if (els.seasonTimeline) els.seasonTimeline.innerHTML = renderers.renderSeason(siteData.seasonPhases);
    if (els.checklist) els.checklist.innerHTML = renderers.renderChecklist(siteData.checklistItems);
    if (els.scratchpadInput) els.scratchpadInput.value = state.appState.scratchpad || "";
    renderTheoryIndex();
  }

  // The theory index lists every problem in window.AIO_THEORY, in module order.
  // That file is deferred after app.js, so the first pass sees no data and the
  // table is filled from the DOMContentLoaded call at the bottom of this file.
  // While a problem is open the index stays hidden, so it is not built.
  function renderTheoryIndex() {
    const rows = document.querySelector("#theoryIndexRows");
    const wrap = document.querySelector("#theoryEmptyState");
    if (!rows || !window.AIO_THEORY || (wrap && wrap.hidden)) return;
    const position = new Map(moduleOrder.map((module, index) => [module.id, index]));
    const problems = (window.AIO_THEORY.problems || [])
      .map((problem, index) => ({ problem, index, at: position.has(problem.moduleId) ? position.get(problem.moduleId) : Infinity }))
      .sort((left, right) => left.at - right.at || left.index - right.index)
      .map((entry) => entry.problem);
    rows.innerHTML = problems
      .map((problem) => {
        const module = moduleLookup.get(problem.moduleId);
        const section = module ? ordering.sectionOrdering[module.section] : null;
        const moduleTitle = module ? module.title : problem.moduleId;
        const sectionLabel = section ? section.label : "";
        return `
          <tr data-theory-filter="${escapeHtml(`${problem.title} ${moduleTitle} ${sectionLabel} ${problem.difficulty}`.toLowerCase())}">
            <th scope="row" data-label="Problem">
              <a href="${localPath("theory/")}?id=${encodeURIComponent(problem.id)}">${escapeHtml(problem.title)}</a>
            </th>
            <td data-label="Module">${escapeHtml(moduleTitle)}</td>
            <td data-label="Section">${escapeHtml(sectionLabel)}</td>
            <td data-label="Parts">${problem.parts.length}</td>
            <td data-label="Difficulty"><span class="theory-difficulty">${escapeHtml(problem.difficulty)}</span></td>
          </tr>
        `;
      })
      .join("");
    filterTheoryIndex();
  }

  // Hides the index rows that do not contain every word typed in the filter box.
  function filterTheoryIndex() {
    const input = document.querySelector("#theoryIndexSearch");
    const rows = document.querySelectorAll("#theoryIndexRows tr");
    const words = (input ? input.value : "").toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    rows.forEach((row) => {
      const text = row.dataset.theoryFilter || "";
      const match = words.every((word) => text.includes(word));
      row.hidden = !match;
      if (match) shown += 1;
    });
    const count = document.querySelector("#theoryIndexCount");
    if (count) count.textContent = `${shown} of ${rows.length} problems`;
    const empty = document.querySelector("#theoryIndexEmpty");
    if (empty) empty.hidden = shown > 0 || rows.length === 0;
  }

  function renderAll() {
    renderTopNav();
    if (window.AIO_AUTH_READY) updateAuthUi(window.AIO_AUTH_SESSION);
    renderStatic();
    renderFilters();
    renderDashboard();
    renderPlanner();
    renderReader();
    renderSectionPage();
    renderModules();
    renderProblems();
    renderSearch();
    renderHomePage();
  }

  function syncAppState(newState) {
    state.appState = newState || storage.loadState();
    renderAll();
  }

  document.addEventListener("aio-progress-sync", (event) => {
    state.appState = event.detail?.state || storage.loadState();
    renderAll();
    // The problem pages draw their own status circle, outside renderAll.
    if (page === "theory" || page === "editor") patchAllStatus();
  });

  function revealProblem(problemId) {
    const answer = document.querySelector(`#answer-${problemId}`);
    if (answer) answer.hidden = !answer.hidden;
  }

  // A circle that has just turned done pops once. The class comes off when
  // the animation ends, so the next toggle can play it again.
  function popStatus(dot) {
    dot.classList.remove("is-popping");
    // Reading the width restarts the animation when the class goes back on.
    void dot.offsetWidth;
    dot.classList.add("is-popping");
    dot.addEventListener("animationend", () => dot.classList.remove("is-popping"), { once: true });
  }

  // ---- the shared status dropdown -----------------------------------------
  // One menu serves every status button on the page, so a 700-row problem
  // table carries 700 buttons and a single menu rather than 700 menus. It is
  // moved under whichever button is open and closed on Escape, an outside
  // click, a scroll or a resize.
  // Problems and resources each get one menu; statusMenuEl is whichever is
  // open (or was last opened).
  const statusMenus = {};
  let statusMenuEl = null;
  let statusMenuOwner = null;
  // Focusing the first item can scroll the page, which would otherwise close
  // the menu through the scroll handler below the moment it opened.
  let statusMenuOpenedAt = 0;

  function ensureStatusMenu(kind) {
    if (!statusMenus[kind]) {
      const host = document.createElement("div");
      host.innerHTML = kind === "resource" ? renderers.resourceStatusMenu() : renderers.statusMenu();
      statusMenus[kind] = host.firstElementChild;
      document.body.appendChild(statusMenus[kind]);
    }
    return statusMenus[kind];
  }

  // The page is zoomed on desktop (main.css). Measurements come back in screen
  // pixels, but a position written to style is scaled by the zoom again, so
  // writes divide by it.
  function pageZoom() {
    return parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
  }

  function placeStatusMenu(trigger) {
    const menu = statusMenuEl;
    const box = trigger.getBoundingClientRect();
    menu.hidden = false;
    const size = menu.getBoundingClientRect();
    const width = size.width;
    const height = size.height;
    const margin = 8;
    // Below the button by default, above it when the bottom is close.
    let top = box.bottom + 6;
    if (top + height > window.innerHeight - margin) top = Math.max(margin, box.top - height - 6);
    // Left-aligned with the button, pulled in when it would run off the side.
    let left = box.left;
    if (left + width > window.innerWidth - margin) left = window.innerWidth - width - margin;
    if (left < margin) left = margin;
    // The menu is fixed, so these are viewport coordinates and no page or
    // inner-container scroll offset comes into it.
    top = Math.min(Math.max(top, margin), Math.max(margin, window.innerHeight - height - margin));
    const zoom = pageZoom();
    menu.style.top = `${Math.round(top / zoom)}px`;
    menu.style.left = `${Math.round(left / zoom)}px`;
  }

  function openStatusMenu(trigger) {
    const kind = trigger.dataset.resourceStatus !== undefined ? "resource" : "problem";
    if (statusMenuEl && statusMenuEl !== statusMenus[kind]) closeStatusMenu();
    const menu = ensureStatusMenu(kind);
    statusMenuEl = menu;
    let current;
    if (kind === "resource") {
      menu.dataset.resourceUrl = trigger.dataset.resourceStatus;
      current = (state.appState.resourceStatus || {})[trigger.dataset.resourceStatus] || "not-started";
    } else {
      menu.dataset.problemId = trigger.dataset.cycleProblemStatus;
      current = state.appState.problemStatus[trigger.dataset.cycleProblemStatus] || "not-started";
    }
    menu.querySelectorAll("[data-status-item]").forEach((item) => {
      const value = item.dataset.setProblemStatus || item.dataset.setResourceStatus;
      const mine = value === current;
      item.setAttribute("aria-checked", mine ? "true" : "false");
      item.classList.toggle("is-current", mine);
      item.dataset.status = value;
    });
    if (statusMenuOwner && statusMenuOwner !== trigger) statusMenuOwner.setAttribute("aria-expanded", "false");
    statusMenuOwner = trigger;
    trigger.setAttribute("aria-expanded", "true");
    placeStatusMenu(trigger);
    statusMenuOpenedAt = Date.now();
    const checked = menu.querySelector('[aria-checked="true"]') || menu.querySelector("[data-status-item]");
    if (checked) checked.focus({ preventScroll: true });
  }

  function closeStatusMenu(returnFocus) {
    if (!statusMenuEl || statusMenuEl.hidden) return;
    statusMenuEl.hidden = true;
    if (statusMenuOwner) {
      statusMenuOwner.setAttribute("aria-expanded", "false");
      if (returnFocus) statusMenuOwner.focus();
    }
    statusMenuOwner = null;
  }

  function setProblemStatusFromMenu(next) {
    const problemId = statusMenuEl && statusMenuEl.dataset.problemId;
    if (!problemId) return;
    closeStatusMenu(true);
    // A lesson or problem page holds open answers and folded boxes that a
    // full re-render would close, so there the change is patched in place.
    if (PATCH_STATUS_PAGES.includes(page)) {
      state.appState = storage.setProblemStatus(problemId, next);
      patchProblemStatus(problemId);
    } else {
      syncAppState(storage.setProblemStatus(problemId, next));
      // The table drew new buttons, so the pop plays on the new one.
      if (next === "complete") {
        document.querySelectorAll(`[data-cycle-problem-status="${CSS.escape(problemId)}"]`).forEach(popStatus);
      }
    }
  }

  // A resource's status lives only in its own row's square, so the change is
  // patched there and nothing else re-renders.
  function setResourceStatusFromMenu(next) {
    const url = statusMenuEl && statusMenuEl.dataset.resourceUrl;
    if (!url) return;
    closeStatusMenu(true);
    state.appState = storage.setResourceStatus(url, next);
    document.querySelectorAll(`[data-resource-status="${CSS.escape(url)}"]`).forEach((button) => {
      const host = document.createElement("div");
      host.innerHTML = renderers.resourceStatusButton(url, button.dataset.resourceTitle || "", next);
      const fresh = host.firstElementChild;
      const hadFocus = document.activeElement === button;
      button.replaceWith(fresh);
      if (hadFocus) fresh.focus();
    });
  }

  // Arrow keys walk the menu, Escape closes it, and Enter or Space picks.
  document.addEventListener("keydown", (event) => {
    if (!statusMenuEl || statusMenuEl.hidden) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeStatusMenu(true);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "Home" && event.key !== "End") return;
    const items = Array.from(statusMenuEl.querySelectorAll("[data-status-item]"));
    if (!items.length) return;
    event.preventDefault();
    const here = items.indexOf(document.activeElement);
    let next = 0;
    if (event.key === "ArrowDown") next = here < 0 ? 0 : (here + 1) % items.length;
    else if (event.key === "ArrowUp") next = here < 0 ? items.length - 1 : (here - 1 + items.length) % items.length;
    else if (event.key === "End") next = items.length - 1;
    items[next].focus();
  });

  document.addEventListener("pointerdown", (event) => {
    if (!statusMenuEl || statusMenuEl.hidden) return;
    if (event.target.closest(".pstatus-menu") || event.target.closest("[data-cycle-problem-status], [data-resource-status]")) return;
    closeStatusMenu();
  });

  window.addEventListener("resize", () => closeStatusMenu());
  window.addEventListener("scroll", () => {
    if (Date.now() - statusMenuOpenedAt < 250) return;
    closeStatusMenu();
  }, true);

  // Brings every control showing a problem's status up to date without
  // re-rendering the page: the status circles, the module's solved meter, and
  // (through the event) the problem pages.
  function patchProblemStatus(problemId) {
    const status = state.appState.problemStatus[problemId] || "not-started";
    const complete = status === "complete";
    const label = renderers.statusTitle(status);
    document.querySelectorAll(`[data-cycle-problem-status="${CSS.escape(problemId)}"]`).forEach((dot) => {
      const turnedDone = complete && dot.dataset.status !== "complete";
      dot.dataset.status = renderers.PROBLEM_STATUS_LABELS[status] ? status : "not-started";
      dot.setAttribute("aria-pressed", renderers.problemStatusPressed(status));
      dot.setAttribute("aria-label", `${label}: ${dot.dataset.problemTitle || ""}. Change status`);
      dot.title = label;
      if (turnedDone) popStatus(dot);
    });
    // A question whose problem is no longer solved can be answered again.
    if (!complete) {
      document.querySelectorAll(`[data-mcq-problem="${CSS.escape(problemId)}"]`).forEach((card) => {
        if (card.dataset.mcqMode === "submit") {
          if (card.classList.contains("is-solved")) resetSubmitCard(card);
          return;
        }
        // A first wrong pick marks the problem in progress and keeps its verdict.
        if (status === "in-progress" && !card.classList.contains("is-solved")) return;
        delete card.dataset.locked;
        card.classList.remove("is-solved");
        card.querySelectorAll(".mcq-option").forEach((option) => {
          option.classList.remove("is-picked", "is-correct", "is-wrong");
          const body = option.querySelector(".mcq-verdict-body");
          if (body) body.hidden = true;
        });
        card.querySelectorAll("[data-mcq-choice]").forEach((choice) => choice.removeAttribute("aria-disabled"));
      });
    }
    const module = moduleLookup.get(state.selectedModuleId);
    const meter = document.querySelector(".reader-progress");
    if (page === "module" && module && meter) {
      const mine = problemData.problems.filter((problem) => problem.moduleId === module.id);
      const solved = mine.filter((problem) => state.appState.problemStatus[problem.id] === "complete").length;
      const fill = meter.querySelector(".reader-progress-fill");
      const count = meter.querySelector(".reader-progress-count");
      if (fill) fill.style.width = `${mine.length ? Math.round((solved / mine.length) * 100) : 0}%`;
      if (count) {
        count.innerHTML = `<b>${solved}</b>/${mine.length} <span>solved</span>`;
        count.classList.toggle("is-done", mine.length > 0 && solved === mine.length);
        count.classList.toggle("is-started", solved > 0 && solved < mine.length);
      }
    }
    document.dispatchEvent(new CustomEvent("aio-problem-status", { detail: { id: problemId, status } }));
  }

  // A problem page (the editor's passed checks) changed a status behind app.js.
  document.addEventListener("aio-problem-status-set", (event) => {
    if (!event.detail || !event.detail.id) return;
    state.appState = storage.loadState();
    patchProblemStatus(event.detail.id);
  });

  // Pages that hold open answers, folded boxes or a running notebook, where a
  // status change is patched in place rather than re-rendered.
  const PATCH_STATUS_PAGES = ["module", "theory", "editor"];

  // Storage changed under the page: another tab, or a page restored from the
  // back-forward cache after the reader ticked a problem elsewhere.
  function patchAllStatus() {
    const ids = new Set();
    document.querySelectorAll("[data-cycle-problem-status]").forEach((dot) => ids.add(dot.dataset.cycleProblemStatus));
    ids.forEach((id) => patchProblemStatus(id));
  }

  function refreshFromStorage() {
    state.appState = storage.loadState();
    if (PATCH_STATUS_PAGES.includes(page)) patchAllStatus();
    else renderAll();
  }

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) refreshFromStorage();
  });
  window.addEventListener("storage", (event) => {
    if (event.key === null || String(event.key).startsWith("usaaio-guide-state")) refreshFromStorage();
  });

  // One option of a multiple-choice question was picked. Its verdict and
  // explanation show; a wrong pick can be retried, and the right one locks the
  // question and, when the question is a whole problem, marks it solved.
  // Draws a pick on a question card; returns whether it was the right one.
  function showPick(card, option) {
    const correct = option.dataset.correct === "true";
    card.querySelectorAll(".mcq-option").forEach((entry) => {
      const picked = entry === option;
      entry.classList.toggle("is-picked", picked);
      entry.classList.toggle("is-correct", picked && correct);
      entry.classList.toggle("is-wrong", picked && !correct);
      const body = entry.querySelector(".mcq-verdict-body");
      if (body) body.hidden = !picked;
    });
    if (correct) {
      card.dataset.locked = "true";
      card.classList.add("is-solved");
      card.querySelectorAll("[data-mcq-choice]").forEach((choice) => choice.setAttribute("aria-disabled", "true"));
    }
    return correct;
  }

  // A first attempt that is not yet right marks a problem in progress. A
  // problem already in progress or done stays as it is.
  function markInProgress(problemId) {
    if (!problemId || (state.appState.problemStatus[problemId] || "not-started") !== "not-started") return;
    state.appState = storage.setProblemStatus(problemId, "in-progress");
    patchProblemStatus(problemId);
  }

  function judgeChoice(button) {
    const card = button.closest("[data-mcq]");
    const option = button.closest(".mcq-option");
    if (!card || !option || card.dataset.locked === "true") return;
    const problemId = card.dataset.mcqProblem;
    if (!showPick(card, option)) {
      if (card.dataset.mcqSolves === "true") markInProgress(problemId);
      return;
    }
    if (card.dataset.mcqSolves === "true" && problemId && state.appState.problemStatus[problemId] !== "complete") {
      state.appState = storage.setProblemStatus(problemId, "complete");
      patchProblemStatus(problemId);
    }
  }

  // The problem page's question card (renderSubmitQuestion): a pick only
  // selects, Submit judges it and opens the result card, Try again clears it.
  const choiceLetter = (index) => "ABCDEFGHIJ"[index] || String(index + 1);

  function pickChoice(radio) {
    const card = radio.closest('[data-mcq-mode="submit"]');
    if (!card || card.classList.contains("is-submitted")) return;
    card.querySelectorAll(".mcq-option").forEach((option) => {
      option.classList.toggle("is-selected", option.contains(radio));
    });
    const submit = card.querySelector(".mcq-submit");
    if (submit) submit.disabled = false;
    const hint = card.querySelector("[data-mcq-hint]");
    if (hint) hint.textContent = `You picked ${choiceLetter(Number(radio.value))}. Submit when you are ready.`;
  }

  function submitChoice(form) {
    const card = form.closest('[data-mcq-mode="submit"]');
    const radio = form.querySelector(".mcq-radio:checked");
    if (!card || !radio || card.classList.contains("is-submitted")) return;
    const index = Number(radio.value);
    const option = radio.closest(".mcq-option");
    const correct = option.dataset.correct === "true";
    card.classList.add("is-submitted");
    card.classList.toggle("is-solved", correct);
    card.classList.toggle("is-missed", !correct);
    card.querySelectorAll(".mcq-option").forEach((entry) => {
      const picked = entry === option;
      const right = entry.dataset.correct === "true";
      entry.classList.toggle("is-picked", picked);
      entry.classList.toggle("is-correct", right);
      entry.classList.toggle("is-wrong", picked && !right);
      const tag = entry.querySelector(".mcq-tag");
      if (tag) tag.textContent = right ? "Answer" : picked ? "Your pick" : "";
    });
    form.querySelector(".mcq-fieldset").disabled = true;
    form.querySelector(".mcq-actions").hidden = true;
    const result = card.querySelector("[data-mcq-result]");
    result.hidden = false;
    result.querySelector("[data-mcq-verdict]").textContent = correct ? "Correct." : "Not quite.";
    result.querySelector("[data-mcq-picked]").textContent = `You picked ${choiceLetter(index)}.`;
    result.querySelectorAll("[data-explain]").forEach((entry) => {
      entry.hidden = Number(entry.dataset.explain) !== index;
    });
    const answer = result.querySelector(".mcq-answer");
    if (answer) answer.hidden = correct;
    const next = result.querySelector(".mcq-next");
    if (next) next.hidden = !correct;
    const review = result.querySelector(".mcq-review");
    if (review) review.hidden = correct;
    result.focus({ preventScroll: true });
    if (result.getBoundingClientRect().bottom > window.innerHeight) {
      result.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    const problemId = card.dataset.mcqProblem;
    if (!correct && card.dataset.mcqSolves === "true") markInProgress(problemId);
    if (correct && card.dataset.mcqSolves === "true" && problemId && state.appState.problemStatus[problemId] !== "complete") {
      state.appState = storage.setProblemStatus(problemId, "complete");
      patchProblemStatus(problemId);
    }
  }

  function resetSubmitCard(card) {
    card.classList.remove("is-submitted", "is-solved", "is-missed");
    card.querySelectorAll(".mcq-option").forEach((option) => {
      option.classList.remove("is-selected", "is-picked", "is-correct", "is-wrong");
      const tag = option.querySelector(".mcq-tag");
      if (tag) tag.textContent = "";
    });
    card.querySelectorAll(".mcq-radio").forEach((radio) => {
      radio.checked = false;
    });
    const fieldset = card.querySelector(".mcq-fieldset");
    if (fieldset) fieldset.disabled = false;
    const actions = card.querySelector(".mcq-actions");
    if (actions) actions.hidden = false;
    const submit = card.querySelector(".mcq-submit");
    if (submit) submit.disabled = true;
    const hint = card.querySelector("[data-mcq-hint]");
    if (hint) hint.textContent = "Pick an answer, then submit.";
    const result = card.querySelector("[data-mcq-result]");
    if (result) result.hidden = true;
  }

  document.addEventListener("change", (event) => {
    const radio = event.target.closest && event.target.closest(".mcq-radio");
    if (radio) pickChoice(radio);
  });

  document.addEventListener("submit", (event) => {
    const form = event.target.closest && event.target.closest("[data-mcq-form]");
    if (!form) return;
    event.preventDefault();
    submitChoice(form);
  });

  // Opens the problem itself. Falls back to the filtered problem bank only when a
  // problem has no URL of its own.
  function focusProblem(problemId) {
    const problem = problemLookup.get(problemId);
    if (!problem) return;
    if (!problem.url) {
      window.location.href = `${localPath("problems/")}?q=${encodeURIComponent(problem.title)}`;
      return;
    }
    if (problem.url.startsWith("/editor/")) {
      window.location.href = `${root}${problem.url.slice(1)}`;
      return;
    }
    window.open(problem.url, "_blank", "noopener");
  }

  document.addEventListener("click", (event) => {
    const authOpen = event.target.closest("[data-auth-open]");
    if (authOpen) {
      event.preventDefault();
      openAuthModal();
      if (els.navLinks?.classList.contains("open")) {
        els.navLinks.classList.remove("open");
        els.navToggle?.setAttribute("aria-expanded", "false");
      }
      return;
    }

    const authClose = event.target.closest("[data-auth-close]");
    if (authClose) {
      event.preventDefault();
      closeAuthModal();
      return;
    }

    const authLogin = event.target.closest("[data-auth-login]");
    if (authLogin) {
      event.preventDefault();
      const provider = authLogin.dataset.authLogin || "github";
      const providerName = provider === "google" ? "Google" : "GitHub";
      updateAuthUi(null, `Opening ${providerName} sign-in...`);
      getAuthClient()
        .then((client) =>
          client.auth.signInWithOAuth({
            provider,
            options: {
              redirectTo: authRedirectUrl(),
            },
          }),
        )
        .then(({ error }) => {
          if (error) updateAuthUi(null, `Login failed: ${error.message}`);
        })
        .catch(() => updateAuthUi(null, "Login failed. Check your connection and Supabase settings."));
      return;
    }

    const authLink = event.target.closest("[data-auth-link]");
    if (authLink) {
      event.preventDefault();
      const session = window.AIO_AUTH_SESSION;
      const provider = authLink.dataset.authLink || "github";
      if (!session?.user) {
        updateAuthUi(null, "Sign in before connecting another provider.");
        return;
      }
      updateAuthUi(session, `Opening ${authProviderLabel(provider)} connection...`);
      getAuthClient()
        .then((client) => {
          if (typeof client.auth.linkIdentity !== "function") {
            throw new Error("Manual provider linking is not enabled in this Supabase project.");
          }
          return client.auth.linkIdentity({
            provider,
            options: {
              redirectTo: authRedirectUrl(),
            },
          });
        })
        .then(({ error }) => {
          if (error) throw error;
          updateAuthUi(session, `Connected ${authProviderLabel(provider)}. Refreshing account details...`);
          void refreshAuthIdentities(session);
        })
        .catch((error) => {
          updateAuthUi(session, `Could not connect ${authProviderLabel(provider)}: ${error.message}`);
          void refreshAuthIdentities(session);
        });
      return;
    }

    const authLogout = event.target.closest("[data-auth-logout]");
    if (authLogout) {
      event.preventDefault();
      updateAuthUi(null, "Signing out...");
      getAuthClient()
        .then((client) => client.auth.signOut())
        .then(({ error }) => {
          if (error) updateAuthUi(null, `Logout failed: ${error.message}`);
          else {
            updateAuthUi(null, "You are signed out.");
            void storage.syncWithSession(null);
            closeAuthModal();
          }
        })
        .catch(() => updateAuthUi(null, "Logout failed. Check your connection and Supabase settings."));
      return;
    }

    const readerNavToggle = event.target.closest("[data-reader-nav]");
    if (readerNavToggle && els.moduleReader) {
      const collapsed = els.moduleReader.classList.toggle("nav-collapsed");
      try {
        localStorage.setItem(READER_NAV_KEY, collapsed ? "1" : "0");
      } catch (error) {
        /* private browsing */
      }
      // The column width changed without a window resize, so a figure that
      // fitted now may be drawn smaller than its own size (or the reverse).
      markZoomableFigures();
      return;
    }

    const moduleNotebookButton = event.target.closest("[data-download-module-notebook]");
    if (moduleNotebookButton) {
      downloadModuleNotebook(moduleNotebookButton.dataset.downloadModuleNotebook);
      return;
    }

    // Copy a listing without its line numbers: the gutter lives in <b>, the
    // source in <code>, so reading the <code> elements gives paste-ready text.
    const copyButton = event.target.closest("[data-copy-code]");
    if (copyButton) {
      const block = copyButton.closest(".reader-code-block");
      const source = block ? [...block.querySelectorAll("pre code")].map((line) => line.textContent).join("\n") : "";
      const done = () => {
        copyButton.textContent = "Copied";
        window.setTimeout(() => {
          copyButton.textContent = "Copy";
        }, 1400);
      };
      // execCommand is the fallback for insecure origins and for browsers that
      // refuse the async clipboard without a permission grant.
      const legacyCopy = () => {
        const scratch = document.createElement("textarea");
        scratch.value = source;
        scratch.setAttribute("readonly", "");
        scratch.style.cssText = "position:fixed;top:-1000px;opacity:0";
        document.body.appendChild(scratch);
        scratch.select();
        let ok = false;
        try {
          ok = document.execCommand("copy");
        } catch (error) {
          ok = false;
        }
        scratch.remove();
        if (ok) done();
        else copyButton.textContent = "Press ⌘C";
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(source).then(done, legacyCopy);
      } else {
        legacyCopy();
      }
      return;
    }

    const moduleButton = event.target.closest("[data-open-module]");
    if (moduleButton) {
      event.preventDefault();
      openModule(moduleButton.dataset.openModule);
      return;
    }

    const filterButton = event.target.closest("[data-filter]");
    if (filterButton) {
      state.filter = filterButton.dataset.filter;
      renderFilters();
      renderModules();
      return;
    }

    const problemPageButton = event.target.closest("[data-problem-page]");
    if (problemPageButton && !problemPageButton.disabled) {
      state.problemPage = Number(problemPageButton.dataset.problemPage);
      renderProblems();
      if (els.problemTable) {
        els.problemTable.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      return;
    }

    const zoomFigure = event.target.closest(".lesson-figure.is-zoomable img");
    if (zoomFigure) {
      openFigure(zoomFigure);
      return;
    }

    const mcqChoice = event.target.closest("[data-mcq-choice]");
    if (mcqChoice) {
      judgeChoice(mcqChoice);
      return;
    }

    const mcqRetry = event.target.closest("[data-mcq-retry]");
    if (mcqRetry) {
      const card = mcqRetry.closest('[data-mcq-mode="submit"]');
      if (card) {
        resetSubmitCard(card);
        const first = card.querySelector(".mcq-radio");
        if (first) first.focus();
      }
      return;
    }

    const resourceItem = event.target.closest("[data-set-resource-status]");
    if (resourceItem) {
      setResourceStatusFromMenu(resourceItem.dataset.setResourceStatus);
      return;
    }

    const resourceSquare = event.target.closest("[data-resource-status]");
    if (resourceSquare) {
      if (statusMenuOwner === resourceSquare) closeStatusMenu();
      else openStatusMenu(resourceSquare);
      return;
    }

    const statusItem = event.target.closest("[data-set-problem-status]");
    if (statusItem) {
      setProblemStatusFromMenu(statusItem.dataset.setProblemStatus);
      return;
    }

    const statusDot = event.target.closest("[data-cycle-problem-status]");
    if (statusDot) {
      if (statusMenuOwner === statusDot) closeStatusMenu();
      else openStatusMenu(statusDot);
      return;
    }

    if (event.target.closest("#problemShuffle")) {
      const order = new Map();
      filteredProblems().forEach((problem) => order.set(problem.id, Math.random()));
      state.problemShuffle = order;
      state.problemPage = 1;
      renderProblems();
      return;
    }

    // A column header click sorts by that column, and a second click flips the
    // direction. The sort dropdown stays in sync because both write the same key.
    const sortHeader = event.target.closest("[data-sort]");
    if (sortHeader) {
      const key = sortHeader.dataset.sort;
      if (state.problemSort === key) {
        state.problemSortDir = state.problemSortDir === "asc" ? "desc" : "asc";
      } else {
        state.problemSort = key;
        state.problemSortDir = "asc";
      }
      state.problemShuffle = null;
      state.problemPage = 1;
      renderProblems();
      renderFilters();
      return;
    }

    // A type tab and the Type dropdown write the same filter.
    const typeTab = event.target.closest("[data-problem-type]");
    if (typeTab) {
      state.problemTypeFilter = typeTab.dataset.problemType;
      state.problemPage = 1;
      if (els.problemTypeFilter) els.problemTypeFilter.value = state.problemTypeFilter;
      renderProblems();
      return;
    }

    if (event.target.closest("#problemReset")) {
      state.problemFilter = "all";
      state.problemTypeFilter = "all";
      state.problemStatusFilter = "all";
      state.problemModuleFilter = "all";
      state.problemSectionFilter = "all";
      state.problemSort = "relevance";
      state.problemSortDir = "asc";
      state.problemShuffle = null;
      state.problemSearch = "";
      state.problemPage = 1;
      if (els.problemSearch) els.problemSearch.value = "";
      renderFilters();
      renderProblems();
      return;
    }

    if (event.target.closest("#problemRandom")) {
      const pool = filteredProblems();
      if (pool.length) {
        const pick = pool[Math.floor(Math.random() * pool.length)];
        const isInternalLink = Boolean(pick.url) && pick.url.startsWith("/");
        if (pick.url) {
          window.open(isInternalLink ? `${root}${pick.url.slice(1)}` : pick.url, isInternalLink ? "_self" : "_blank");
        } else {
          highlightProblem(pick.id, pool);
        }
      }
      return;
    }

    const searchFilterButton = event.target.closest("[data-search-filter]");
    if (searchFilterButton) {
      state.searchFilter = searchFilterButton.dataset.searchFilter;
      renderFilters();
      renderSearch();
      return;
    }

    const plannerTabButton = event.target.closest("[data-planner-tab]");
    if (plannerTabButton) {
      state.plannerTab = plannerTabButton.dataset.plannerTab;
      document.querySelectorAll("[data-planner-tab]").forEach((button) => {
        button.classList.toggle("active", button === plannerTabButton);
      });
      renderPlannerLibrary();
      return;
    }

    const planAddButton = event.target.closest("[data-plan-add]");
    if (planAddButton) {
      const block = planAddButton.closest("[data-plan-source]");
      if (block) addPlanItem(block.dataset.planType, block.dataset.planId, state.appState.plan.length);
      return;
    }

    const planRemoveButton = event.target.closest("[data-plan-remove]");
    if (planRemoveButton) {
      const plan = [...state.appState.plan];
      plan.splice(Number(planRemoveButton.dataset.planRemove), 1);
      syncAppState(storage.setPlan(plan));
      return;
    }

    const planMoveButton = event.target.closest("[data-plan-move]");
    if (planMoveButton) {
      const from = Number(planMoveButton.dataset.planTarget);
      const to = planMoveButton.dataset.planMove === "up" ? from - 1 : from + 1;
      const plan = [...state.appState.plan];
      if (to < 0 || to >= plan.length) return;
      const [moved] = plan.splice(from, 1);
      plan.splice(to, 0, moved);
      syncAppState(storage.setPlan(plan));
      return;
    }

    if (event.target.matches("#clearPlan")) {
      syncAppState(storage.setPlan([]));
      return;
    }

    const revealButton = event.target.closest("[data-reveal-problem]");
    if (revealButton) {
      revealProblem(revealButton.dataset.revealProblem);
      return;
    }

    const focusProblemButton = event.target.closest("[data-focus-problem]");
    if (focusProblemButton) {
      focusProblem(focusProblemButton.dataset.focusProblem);
      return;
    }

  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAuthModal();
  });

  document.addEventListener("change", (event) => {
    if (event.target.matches("[data-module-status]")) {
      syncAppState(storage.setModuleStatus(event.target.dataset.moduleStatus, event.target.value));
      return;
    }
    if (event.target.matches("#problemStatusFilter")) {
      state.problemStatusFilter = event.target.value;
      state.problemPage = 1;
      renderProblems();
      return;
    }
    if (event.target.matches("#problemTypeFilter")) {
      state.problemTypeFilter = event.target.value;
      state.problemPage = 1;
      renderProblems();
      return;
    }
    const problemSelects = {
      "#problemDifficultyFilter": "problemFilter",
      "#problemModuleFilter": "problemModuleFilter",
      "#problemSectionFilter": "problemSectionFilter",
      "#problemSort": "problemSort",
    };
    const matched = Object.keys(problemSelects).find((selector) => event.target.matches(selector));
    if (matched) {
      state[problemSelects[matched]] = event.target.value;
      // Any explicit sort or filter change clears a pending shuffle.
      state.problemShuffle = null;
      // The dropdown lists each column ascending, so a pick from it restarts
      // the direction. The arrow on the header follows.
      if (matched === "#problemSort") state.problemSortDir = "asc";
      state.problemPage = 1;
      renderProblems();
    }
  });

  if (els.moduleSearch) {
    els.moduleSearch.addEventListener("input", (event) => {
      state.search = event.target.value;
      renderModules();
    });
  }

  if (els.problemSearch) {
    els.problemSearch.value = state.problemSearch;
    els.problemSearch.addEventListener("input", (event) => {
      state.problemSearch = event.target.value;
      state.problemPage = 1;
      renderProblems();
    });
  }

  if (els.globalSearch) {
    els.globalSearch.value = state.globalSearch;
    els.globalSearch.addEventListener("input", (event) => {
      state.globalSearch = event.target.value;
      renderSearch();
      const nextParams = new URLSearchParams(window.location.search);
      if (state.globalSearch.trim()) {
        nextParams.set("q", state.globalSearch.trim());
      } else {
        nextParams.delete("q");
      }
      if (state.searchFilter !== "all") {
        nextParams.set("type", state.searchFilter);
      } else {
        nextParams.delete("type");
      }
      const queryString = nextParams.toString();
      window.history.replaceState(null, "", `${window.location.pathname}${queryString ? `?${queryString}` : ""}`);
    });
  }

  if (els.scratchpadInput) {
    els.scratchpadInput.addEventListener("input", (event) => {
      storage.saveScratchpad(event.target.value);
      if (els.scratchpadSaved) {
        els.scratchpadSaved.textContent = window.AIO_AUTH_SESSION?.user
          ? "Saved to your account."
          : "Saved locally in this browser.";
      }
    });
  }

  if (els.navToggle && els.navLinks) {
    els.navToggle.addEventListener("click", () => {
      const open = els.navLinks.classList.toggle("open");
      els.navToggle.setAttribute("aria-expanded", String(open));
    });
    els.navLinks.addEventListener("click", (event) => {
      if (event.target.matches("a")) {
        els.navLinks.classList.remove("open");
        els.navToggle.setAttribute("aria-expanded", "false");
      }
    });
    // Crossing back to a wide viewport closes the sheet and resets the toggle.
    window.addEventListener("resize", () => {
      if (window.innerWidth > 1024 && els.navLinks.classList.contains("open")) {
        els.navLinks.classList.remove("open");
        els.navToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.addEventListener("click", (event) => {
    document.querySelectorAll(".nav-menu[open], .reader-stage[open]").forEach((menu) => {
      if (!menu.contains(event.target)) menu.removeAttribute("open");
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll(".nav-menu[open], .reader-stage[open]").forEach((menu) => menu.removeAttribute("open"));
    if (els.navLinks && els.navLinks.classList.contains("open")) {
      els.navLinks.classList.remove("open");
      if (els.navToggle) els.navToggle.setAttribute("aria-expanded", "false");
    }
  });

  // On a problem page the left and right arrow keys follow the header's
  // previous and next buttons, unless the reader is typing or a dialog is open.
  if (page === "theory" || page === "editor") {
    document.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target;
      if (target && target.closest && target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])")) return;
      if (document.querySelector("dialog[open]")) return;
      const link = document.querySelector(`[data-problem-step="${event.key === "ArrowLeft" ? "prev" : "next"}"]`);
      if (!link) return;
      event.preventDefault();
      window.location.href = link.href;
    });
  }

  if (els.plannerSearch) {
    els.plannerSearch.addEventListener("input", (event) => {
      state.plannerSearch = event.target.value;
      renderPlannerLibrary();
    });
  }

  function addPlanItem(type, id, index) {
    const exists = state.appState.plan.some((item) => item.type === type && item.id === id);
    if (exists) return;
    const plan = [...state.appState.plan];
    plan.splice(Math.max(0, Math.min(index, plan.length)), 0, { type, id });
    syncAppState(storage.setPlan(plan));
  }

  function planDropIndex(event) {
    const target = event.target.closest("[data-plan-index]");
    if (!target) return state.appState.plan.length;
    const rect = target.getBoundingClientRect();
    const after = event.clientY > rect.top + rect.height / 2;
    return Number(target.dataset.planIndex) + (after ? 1 : 0);
  }

  document.addEventListener("dragstart", (event) => {
    const sourceBlock = event.target.closest("[data-plan-source]");
    if (sourceBlock && els.plannerBoard) {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData(
        "text/plain",
        JSON.stringify({ kind: "add", type: sourceBlock.dataset.planType, id: sourceBlock.dataset.planId }),
      );
      return;
    }
    const boardBlock = event.target.closest("[data-plan-index]");
    if (boardBlock) {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", JSON.stringify({ kind: "move", index: Number(boardBlock.dataset.planIndex) }));
      boardBlock.classList.add("dragging");
    }
  });

  document.addEventListener("dragover", (event) => {
    if (!els.plannerBoard) return;
    if (event.target.closest("#plannerBoard")) {
      event.preventDefault();
      els.plannerBoard.classList.add("drag-over");
    }
  });

  document.addEventListener("dragleave", (event) => {
    if (!els.plannerBoard) return;
    if (event.target === els.plannerBoard && !els.plannerBoard.contains(event.relatedTarget)) {
      els.plannerBoard.classList.remove("drag-over");
    }
  });

  document.addEventListener("dragend", () => {
    if (!els.plannerBoard) return;
    els.plannerBoard.classList.remove("drag-over");
    document.querySelectorAll(".plan-block.dragging").forEach((block) => block.classList.remove("dragging"));
  });

  document.addEventListener("drop", (event) => {
    if (!els.plannerBoard || !event.target.closest("#plannerBoard")) return;
    event.preventDefault();
    els.plannerBoard.classList.remove("drag-over");
    let payload;
    try {
      payload = JSON.parse(event.dataTransfer.getData("text/plain"));
    } catch (error) {
      return;
    }
    if (!payload) return;
    const insertIndex = planDropIndex(event);
    if (payload.kind === "add") {
      addPlanItem(payload.type, payload.id, insertIndex);
      return;
    }
    if (payload.kind === "move") {
      const plan = [...state.appState.plan];
      const from = payload.index;
      if (!(from >= 0 && from < plan.length)) return;
      const [moved] = plan.splice(from, 1);
      const to = from < insertIndex ? insertIndex - 1 : insertIndex;
      plan.splice(Math.max(0, Math.min(to, plan.length)), 0, moved);
      syncAppState(storage.setPlan(plan));
    }
  });

  function initHeroCanvas() {
    const canvas = document.querySelector("#heroCanvas");
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let width = 0;
    let height = 0;
    let particles = [];
    let rafId = null;

    function seed() {
      const count = Math.max(28, Math.min(90, Math.floor((width * height) / 16000)));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: 1.2 + Math.random() * 1.8,
      }));
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      seed();
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      const linkDistance = Math.min(150, width / 7);
      for (let i = 0; i < particles.length; i += 1) {
        for (let j = i + 1; j < particles.length; j += 1) {
          const a = particles[i];
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distance = Math.hypot(dx, dy);
          if (distance < linkDistance) {
            ctx.strokeStyle = `rgba(184, 78, 48, ${(1 - distance / linkDistance) * 0.3})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      ctx.fillStyle = "rgba(184, 78, 48, 0.5)";
      particles.forEach((particle) => {
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function step() {
      particles.forEach((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        if (particle.x < -20) particle.x = width + 20;
        if (particle.x > width + 20) particle.x = -20;
        if (particle.y < -20) particle.y = height + 20;
        if (particle.y > height + 20) particle.y = -20;
      });
    }

    function frame() {
      step();
      draw();
      rafId = requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener("resize", resize);
    if (reduceMotion) return;
    frame();
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        cancelAnimationFrame(rafId);
        rafId = null;
      } else if (!rafId) {
        frame();
      }
    });
  }

  function initScrollReveal() {
    if (page !== "home" || !("IntersectionObserver" in window)) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const targets = document.querySelectorAll(
      ".page-shell .panel, .route-card, .section-link-grid a, .site-footer",
    );
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -36px" },
    );
    targets.forEach((element, index) => {
      element.classList.add("reveal");
      element.style.setProperty("--reveal-delay", `${(index % 5) * 70}ms`);
      observer.observe(element);
    });
  }

  /* --------------------------- embedded py editors -------------------------- */

  // <EmbeddedPyEditor> cells share one Python worker per page, like the notebook
  // editor: the runtime is large and loads once, on the first Run. The protocol
  // is the one src/notebook-worker.js documents, and the worker loads any package
  // the code imports, so there is nothing to configure but Run and Reset.
  const PY_STATUS_LABELS = {
    off: "Python: not started",
    loading: "Python: downloading runtime (~30-60s on first use)...",
    ready: "Python: ready",
    running: "Python: running",
    failed: "Python: failed to start",
  };

  const pyVersion = (() => {
    try {
      return new URL(document.currentScript.src).search;
    } catch (error) {
      return "";
    }
  })();

  const pySession = {
    worker: null,
    status: "off",
    nextRunId: 1,
    pending: new Map(),

    setStatus(status) {
      this.status = status;
      document.querySelectorAll(".lesson-py-editor-status").forEach((el) => {
        el.textContent = PY_STATUS_LABELS[status] || status;
      });
    },

    ensure() {
      if (this.worker) return this.worker;
      const worker = new Worker(`${root}src/notebook-worker.js${pyVersion}`);
      this.worker = worker;
      worker.onmessage = (event) => {
        if (worker !== this.worker) return;
        this.receive(event.data || {});
      };
      worker.onerror = () => {
        if (worker !== this.worker) return;
        this.fail("Python could not be loaded.");
      };
      this.setStatus("loading");
      return worker;
    },

    receive(message) {
      if (message.type === "status") {
        if (message.status === "ready") {
          if (this.pending.size) this.setStatus("running");
          else this.setStatus("ready");
        } else if (message.status === "failed") {
          this.fail(message.message || "Python could not be loaded.");
        }
        return;
      }
      const run = this.pending.get(message.runId);
      if (!run) return;
      if (message.type === "stdout") {
        run.output += message.text;
        run.render();
        return;
      }
      if (message.type === "figure") {
        run.figures.push(message.png);
        run.render();
        return;
      }
      if (message.type === "done") {
        this.pending.delete(message.runId);
        run.error = message.error || null;
        run.finish();
        if (!this.pending.size && this.status !== "failed") this.setStatus("ready");
      }
    },

    fail(message) {
      if (this.worker) {
        this.worker.terminate();
        this.worker = null;
      }
      const waiting = [...this.pending.values()];
      this.pending.clear();
      waiting.forEach((run) => {
        run.error = message;
        run.finish();
      });
      this.setStatus("failed");
    },

    run(code, editor) {
      const worker = this.ensure();
      const runId = this.nextRunId;
      this.nextRunId += 1;
      const run = {
        output: "",
        figures: [],
        error: null,
        render() { editor.renderOutput(this); },
        finish() { editor.renderOutput(this); editor.setRunning(false); },
      };
      this.pending.set(runId, run);
      if (this.status === "ready") this.setStatus("running");
      worker.postMessage({
        type: "run",
        runId,
        code,
        assetBase: new URL(`${root}public/pyodide/`, window.location.href).href,
      });
    },
  };

  function initEmbeddedEditors() {
    const editors = [...document.querySelectorAll(".lesson-py-editor")];
    if (!editors.length) return;

    editors.forEach((wrap) => {
      const codeEl = wrap.querySelector(".lesson-py-editor-code");
      const outEl = wrap.querySelector(".lesson-py-editor-output");
      const runBtn = wrap.querySelector(".lesson-py-editor-run");
      const resetBtn = wrap.querySelector(".lesson-py-editor-reset");
      const initial = codeEl.value;
      let running = false;
      // Grow the code box to fit its code, so a block shows whole instead of four lines and a scrollbar.
      // Once the reader drags the resize handle, their size wins until Reset: refitting during a drag
      // snapped the box back to the code's height on every frame, which made dragging stutter.
      let sizedByHand = false;
      let dragging = false;
      let lastWidth = 0;
      const fitCode = () => {
        if (sizedByHand) return;
        codeEl.style.height = "auto";
        codeEl.style.height = `${codeEl.scrollHeight + 2}px`;
      };
      fitCode();
      codeEl.addEventListener("input", fitCode);
      codeEl.addEventListener("pointerdown", (event) => {
        const box = codeEl.getBoundingClientRect();
        if (event.clientX > box.right - 24 && event.clientY > box.bottom - 24) dragging = true;
      });
      window.addEventListener("pointerup", () => {
        if (!dragging) return;
        dragging = false;
        sizedByHand = true;
      });
      // Refit when the width changes (first layout, window resize, split view), since wrapping changes height.
      // Height changes are the reader's own drag or this function's own fit, so they are ignored.
      // A fit before the stylesheet and fonts load measures the wrong box, so refit once they have loaded too.
      if (window.ResizeObserver) {
        new ResizeObserver((entries) => {
          const width = entries[0].contentRect.width;
          if (dragging || Math.abs(width - lastWidth) < 1) return;
          lastWidth = width;
          fitCode();
        }).observe(codeEl);
      }
      window.addEventListener("resize", fitCode);
      window.addEventListener("load", fitCode);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitCode);

      const editor = {
        renderOutput(run) {
          const text = run.output.replace(/\s+$/, "");
          const parts = [];
          if (text) parts.push(`<pre class="lesson-py-editor-output-text">${escapeHtml(text)}</pre>`);
          run.figures.forEach((png, n) => {
            parts.push(`<img class="lesson-py-editor-output-figure" src="data:image/png;base64,${escapeHtml(png)}" alt="Figure ${n + 1} drawn by this cell" />`);
          });
          if (run.error) {
            const line = String(run.error).trim().split("\n").filter(Boolean).pop() || run.error;
            parts.push(`<p class="lesson-py-editor-output-error">${escapeHtml(line)}</p>`);
          }
          outEl.innerHTML = parts.join("");
          outEl.hidden = !parts.length;
        },
        setRunning(next) {
          running = next;
          runBtn.disabled = next;
          wrap.classList.toggle("is-running", next);
        },
      };

      runBtn.addEventListener("click", () => {
        if (running) return;
        editor.setRunning(true);
        pySession.run(codeEl.value, editor);
      });
      resetBtn.addEventListener("click", () => {
        codeEl.value = initial;
        sizedByHand = false;
        fitCode();
        outEl.innerHTML = "";
        outEl.hidden = true;
      });
    });
  }

  // A module id that has been retired points at its replacement, so old links
  // and bookmarks still land somewhere. It only fires when the requested id is
  // not itself a module: while both are on the site, the old module still
  // renders, and its retirement is what arms the redirect.
  if (page === "module" && window.AIO_REDIRECTS) {
    const requestedId = params.get("id");
    const replacement = requestedId && window.AIO_REDIRECTS[requestedId];
    if (replacement && !moduleLookup.has(requestedId)) {
      location.replace(`${localPath("module/")}?id=${encodeURIComponent(replacement)}`);
      return;
    }
  }

  renderAll();
  initAuth();
  initEmbeddedEditors();
  initHeroCanvas();
  initScrollReveal();
  // content/theory.js runs after this deferred script, so the theory index
  // table is filled once window.AIO_THEORY is available.
  document.addEventListener("DOMContentLoaded", renderTheoryIndex);
  document.querySelector("#theoryIndexSearch")?.addEventListener("input", filterTheoryIndex);
})();
