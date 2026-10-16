(function () {
  const STORAGE_KEY = "usaaio-guide-state-v2";
  const REMOTE_USER_KEY = "usaaio-guide-remote-user-v1";
  const REMOTE_TABLE = "user_progress";

  const DEFAULT_STATE = {
    moduleStatus: {},
    problemStatus: {},
    // Where a reader is with each module resource, keyed by the resource's
    // URL: reading, practicing, complete and so on, as on usaco.guide.
    resourceStatus: {},
    scratchpad: "",
    plan: [],
    activityLog: [],
    lastModuleId: null,
  };

  function cloneDefaultState() {
    return JSON.parse(JSON.stringify(DEFAULT_STATE));
  }

  function normalizeState(rawState) {
    const state = { ...cloneDefaultState(), ...(rawState || {}) };
    state.moduleStatus = state.moduleStatus || {};
    state.problemStatus = state.problemStatus || {};
    state.resourceStatus = state.resourceStatus && typeof state.resourceStatus === "object" ? state.resourceStatus : {};
    state.plan = Array.isArray(state.plan)
      ? state.plan.filter(
          (item) => item && (item.type === "module" || item.type === "problem") && typeof item.id === "string",
        )
      : [];
    state.activityLog = Array.isArray(state.activityLog) ? state.activityLog : [];
    state.lastModuleId = state.lastModuleId || null;
    return state;
  }

  function loadState() {
    try {
      return normalizeState(JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"));
    } catch (error) {
      return cloneDefaultState();
    }
  }

  function saveState(state) {
    const normalized = normalizeState(state);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    queueRemoteSave(normalized);
    return normalized;
  }

  function saveLocalState(state) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(normalizeState(state)));
  }

  let remoteWriteQueue = Promise.resolve();

  function queueRemoteSave(state) {
    const session = window.AIO_AUTH_SESSION;
    const client = window.AIO_AUTH;
    const userId = session && session.user && session.user.id;
    if (!client || !userId) return;
    const payload = normalizeState(state);
    remoteWriteQueue = remoteWriteQueue
      .then(() => client.from(REMOTE_TABLE).upsert({
        user_id: userId,
        progress: payload,
        updated_at: new Date().toISOString(),
      }))
      .then(({ error }) => {
        if (error) throw error;
      })
      .catch((error) => {
        // Local state remains the source of truth if the network is unavailable.
        console.warn("Could not sync USAAIO Guide progress.", error);
      });
  }

  function clearLocalState() {
    saveLocalState(cloneDefaultState());
  }

  function remoteUserId() {
    try {
      return localStorage.getItem(REMOTE_USER_KEY) || "";
    } catch (error) {
      return "";
    }
  }

  function setRemoteUserId(userId) {
    try {
      if (userId) localStorage.setItem(REMOTE_USER_KEY, userId);
      else localStorage.removeItem(REMOTE_USER_KEY);
    } catch (error) {
      // Private browsing can block storage; the session still works in memory.
    }
  }

  async function syncWithSession(session) {
    const userId = session && session.user && session.user.id;
    if (!userId || !window.AIO_AUTH) {
      if (!userId) {
        if (remoteUserId()) {
          clearLocalState();
          setRemoteUserId("");
        }
        document.dispatchEvent(new CustomEvent("aio-progress-sync", { detail: { state: loadState() } }));
      }
      return loadState();
    }

    const previousUserId = remoteUserId();
    if (previousUserId && previousUserId !== userId) clearLocalState();
    setRemoteUserId(userId);

    try {
      const { data, error } = await window.AIO_AUTH
        .from(REMOTE_TABLE)
        .select("progress")
        .eq("user_id", userId)
        .maybeSingle();
      if (error) throw error;

      const localState = loadState();
      const remoteState = data && data.progress && typeof data.progress === "object"
        ? normalizeState(data.progress)
        : null;
      // Existing account data wins field-by-field; a first-time account adopts
      // any anonymous progress already in this browser.
      const merged = remoteState
        ? normalizeState({ ...localState, ...remoteState })
        : localState;
      saveLocalState(merged);
      if (!remoteState) queueRemoteSave(merged);
      document.dispatchEvent(new CustomEvent("aio-progress-sync", { detail: { state: merged } }));
      return merged;
    } catch (error) {
      console.warn("Could not load USAAIO Guide progress.", error);
      return loadState();
    }
  }

  function updateState(updater) {
    const state = loadState();
    updater(state);
    saveState(state);
    return state;
  }

  function recordActivity(state, entry) {
    if (!entry || entry.status === "not-started") return;
    state.activityLog = [
      {
        ...entry,
        at: new Date().toISOString(),
      },
      ...(state.activityLog || []),
    ].slice(0, 500);
  }

  function completedSetFromStatuses(moduleStatus) {
    return new Set(
      Object.entries(moduleStatus)
        .filter(([, status]) => status === "complete")
        .map(([id]) => id),
    );
  }

  function loadCompleted() {
    return completedSetFromStatuses(loadState().moduleStatus);
  }

  function saveCompleted(completed) {
    updateState((state) => {
      Object.keys(state.moduleStatus).forEach((moduleId) => {
        if (state.moduleStatus[moduleId] === "complete" && !completed.has(moduleId)) {
          state.moduleStatus[moduleId] = "not-started";
        }
      });
      completed.forEach((moduleId) => {
        state.moduleStatus[moduleId] = "complete";
      });
    });
  }

  function toggle(completed, id) {
    if (completed.has(id)) {
      completed.delete(id);
    } else {
      completed.add(id);
    }
    saveCompleted(completed);
    return completed;
  }

  function setModuleStatus(moduleId, status) {
    return updateState((state) => {
      const previous = state.moduleStatus[moduleId] || "not-started";
      state.moduleStatus[moduleId] = status;
      state.lastModuleId = moduleId;
      if (previous !== status) {
        recordActivity(state, { type: "module", id: moduleId, status });
      }
    });
  }

  function setProblemStatus(problemId, status) {
    return updateState((state) => {
      const previous = state.problemStatus[problemId] || "not-started";
      state.problemStatus[problemId] = status;
      if (previous !== status) {
        recordActivity(state, { type: "problem", id: problemId, status });
      }
    });
  }

  // Resources are reading, not solving, so they stay out of the activity log
  // that feeds the streak and the heatmap.
  function setResourceStatus(url, status) {
    return updateState((state) => {
      if (!status || status === "not-started") delete state.resourceStatus[url];
      else state.resourceStatus[url] = status;
    });
  }

  function setPlan(plan) {
    return updateState((state) => {
      state.plan = Array.isArray(plan) ? plan : [];
    });
  }

  function saveScratchpad(value) {
    return updateState((state) => {
      state.scratchpad = value;
    });
  }

  window.AIO_PROGRESS = {
    loadState,
    saveState,
    updateState,
    loadCompleted,
    saveCompleted,
    toggle,
    setModuleStatus,
    setProblemStatus,
    setResourceStatus,
    setPlan,
    saveScratchpad,
    syncWithSession,
  };
})();
