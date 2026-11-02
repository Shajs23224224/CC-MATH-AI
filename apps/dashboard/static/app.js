(() => {
  "use strict";

  const API = "/api/v1";
  const EVENT_TYPES = ["run_created", "state_changed", "progress", "metrics", "warning", "error"];
  const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
  const state = {
    runs: [],
    selectedRunId: null,
    selectedRun: null,
    events: [],
    eventSource: null,
    refreshInFlight: false,
    lastUpdated: null,
    taskCatalog: [],
    selectedJob: null,
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    apiStatus: $("api-status"),
    apiStatusText: $("api-status-text"),
    refresh: $("refresh-button"),
    openSubmitDialog: $("open-submit-dialog"),
    submitDialog: $("submit-dialog"),
    closeSubmitDialog: $("close-submit-dialog"),
    submitForm: $("submit-form"),
    taskSelect: $("task-select"),
    runName: $("run-name"),
    taskParameters: $("task-parameters"),
    taskMetadata: $("task-metadata"),
    submitJobButton: $("submit-job-button"),
    taskCatalogEmpty: $("task-catalog-empty"),
    cancelJobButton: $("cancel-job-button"),
    jobControlNote: $("job-control-note"),
    lastUpdated: $("last-updated"),
    total: $("total-runs"),
    running: $("running-runs"),
    completed: $("completed-runs"),
    failed: $("failed-runs"),
    count: $("runs-count"),
    search: $("search-input"),
    filter: $("status-filter"),
    list: $("run-list"),
    empty: $("empty-runs"),
    footnote: $("runs-footnote"),
    clearFilters: $("clear-filters"),
    detailEmpty: $("detail-empty"),
    detailContent: $("detail-content"),
    detailStatus: $("detail-status"),
    detailAlgorithm: $("detail-algorithm"),
    detailName: $("detail-name"),
    detailId: $("detail-id"),
    progressValue: $("detail-progress-value"),
    progressBar: $("detail-progress-bar"),
    stage: $("detail-stage"),
    step: $("detail-step"),
    created: $("detail-created"),
    duration: $("detail-duration"),
    eventsCount: $("detail-events-count"),
    shortId: $("detail-short-id"),
    metrics: $("metrics-grid"),
    metricsCaption: $("metrics-caption"),
    activity: $("activity-list"),
    activityCount: $("activity-count"),
    streamStatus: $("stream-status"),
    toast: $("toast"),
  };

  let toastTimer = null;

  function setText(element, value) {
    element.textContent = value == null || value === "" ? "—" : String(value);
  }

  function formatDate(value, compact = false) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return new Intl.DateTimeFormat(undefined, compact
      ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
      : { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    ).format(date);
  }

  function relativeTime(value) {
    if (!value) return "—";
    const time = new Date(value).getTime();
    if (!Number.isFinite(time)) return "—";
    const seconds = Math.max(0, Math.floor((Date.now() - time) / 1000));
    if (seconds < 45) return "Just now";
    if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
    return Math.floor(seconds / 86400) + "d ago";
  }

  function durationFor(run) {
    const start = run.started_at || run.created_at;
    if (!start) return "—";
    const end = run.ended_at || new Date().toISOString();
    const milliseconds = Math.max(0, new Date(end).getTime() - new Date(start).getTime());
    if (!Number.isFinite(milliseconds)) return "—";
    const seconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    if (hours) return hours + "h " + minutes + "m";
    if (minutes) return minutes + "m " + remainder + "s";
    return remainder + "s";
  }

  function formatMetric(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
    const absolute = Math.abs(value);
    if (absolute !== 0 && (absolute >= 1e6 || absolute < 1e-4)) return value.toExponential(2);
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 5 }).format(value);
  }

  function readableName(value) {
    return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  async function requestJson(path) {
    const response = await fetch(path, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {
        // Keep the HTTP status as the fallback error message.
      }
      throw new Error("HTTP " + response.status + ": " + detail);
    }
    return response.json();
  }

  function setApiStatus(mode, label) {
    els.apiStatus.classList.remove("online", "offline");
    if (mode === "online") els.apiStatus.classList.add("online");
    if (mode === "offline") els.apiStatus.classList.add("offline");
    setText(els.apiStatusText, label);
  }

  function showToast(message) {
    setText(els.toast, message);
    els.toast.classList.remove("hidden");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => els.toast.classList.add("hidden"), 4200);
  }

  async function loadTaskCatalog() {
    try {
      const tasks = await requestJson(API + "/tasks");
      state.taskCatalog = Array.isArray(tasks) ? tasks : [];
      const fragment = document.createDocumentFragment();
      for (const task of state.taskCatalog) {
        const option = document.createElement("option");
        option.value = task.task_name;
        option.textContent = task.description
          ? task.task_name + " — " + task.description
          : task.task_name;
        fragment.append(option);
      }
      els.taskSelect.replaceChildren(fragment);
      if (!state.taskCatalog.length) {
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = "No registered tasks";
        els.taskSelect.append(empty);
      }
      els.taskCatalogEmpty.classList.toggle("hidden", state.taskCatalog.length !== 0);
      els.submitJobButton.disabled = state.taskCatalog.length === 0;
    } catch (error) {
      state.taskCatalog = [];
      els.taskSelect.replaceChildren();
      const unavailable = document.createElement("option");
      unavailable.value = "";
      unavailable.textContent = "Task registry unavailable";
      els.taskSelect.append(unavailable);
      els.taskCatalogEmpty.classList.remove("hidden");
      els.taskCatalogEmpty.textContent = "Could not load registered tasks: " + error.message;
      els.submitJobButton.disabled = true;
    }
  }

  function parseJsonObject(value, label) {
    let parsed;
    try {
      parsed = JSON.parse(value || "{}");
    } catch (error) {
      throw new Error(label + " must contain valid JSON: " + error.message);
    }
    if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
      throw new Error(label + " must be a JSON object.");
    }
    return parsed;
  }

  async function sendJson(path, method, payload) {
    const response = await fetch(path, {
      method,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      cache: "no-store",
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {
        // Keep the HTTP status as the fallback error message.
      }
      throw new Error("HTTP " + response.status + ": " + detail);
    }
    return response.json();
  }

  async function submitTrainingJob(event) {
    event.preventDefault();
    try {
      const parameters = parseJsonObject(els.taskParameters.value, "Task parameters");
      const metadata = parseJsonObject(els.taskMetadata.value, "Run metadata");
      const taskName = els.taskSelect.value;
      if (!taskName) throw new Error("Select a registered training task.");
      const job = await sendJson(API + "/runs", "POST", {
        task_name: taskName,
        name: els.runName.value.trim(),
        parameters,
        metadata,
      });
      els.submitDialog.close();
      showToast("Training job accepted: " + job.run_id);
      await refreshRuns();
      await selectRun(job.run_id);
    } catch (error) {
      showToast("Could not submit training job: " + error.message);
    }
  }

  async function cancelSelectedJob() {
    const runId = state.selectedRunId;
    if (!runId || !state.selectedJob) return;
    els.cancelJobButton.disabled = true;
    try {
      const job = await sendJson(
        API + "/jobs/" + encodeURIComponent(runId) + "/cancel",
        "POST",
      );
      if (state.selectedRunId === runId) state.selectedJob = job;
      if (job.status === "CANCELLED") {
        showToast("Queued training job cancelled.");
      } else if (job.cancellation_requested) {
        showToast("Cancellation requested. The worker will stop at its next safe checkpoint.");
      } else {
        showToast("Training job is already terminal.");
      }
      await refreshRuns({ reloadSelected: true });
    } catch (error) {
      showToast("Could not cancel training job: " + error.message);
    } finally {
      if (state.selectedRunId === runId) {
        els.cancelJobButton.disabled = false;
        renderDetails();
      }
    }
  }

  function currentFilters() {
    const query = els.search.value.trim().toLocaleLowerCase();
    const status = els.filter.value;
    return state.runs.filter((run) => {
      const matchesStatus = status === "ALL" || run.status === status;
      const matchesQuery = !query || [run.name, run.algorithm, run.run_id]
        .some((part) => String(part || "").toLocaleLowerCase().includes(query));
      return matchesStatus && matchesQuery;
    });
  }

  function makeStatusBadge(status) {
    const badge = document.createElement("span");
    const normalized = TERMINAL.has(status) || status === "RUNNING" || status === "QUEUED"
      ? status.toLowerCase()
      : "cancelled";
    badge.className = "status-badge status-" + normalized;
    badge.textContent = status || "UNKNOWN";
    return badge;
  }

  function renderSummary() {
    const total = state.runs.length;
    const count = (status) => state.runs.filter((run) => run.status === status).length;
    setText(els.total, total);
    setText(els.running, count("RUNNING"));
    setText(els.completed, count("COMPLETED"));
    setText(els.failed, count("FAILED") + count("CANCELLED"));
    setText(els.count, total);
    setText(els.lastUpdated, state.lastUpdated
      ? new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(state.lastUpdated)
      : "—");
  }

  function renderRuns() {
    const visible = currentFilters();
    const fragment = document.createDocumentFragment();

    for (const run of visible) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "run-row" + (run.run_id === state.selectedRunId ? " selected" : "");
      row.setAttribute("role", "row");
      row.setAttribute("aria-pressed", run.run_id === state.selectedRunId ? "true" : "false");

      const primary = document.createElement("span");
      primary.className = "run-primary";
      const name = document.createElement("span");
      name.className = "run-name";
      name.textContent = run.name || "Unnamed run";
      const algorithm = document.createElement("span");
      algorithm.className = "run-algorithm";
      algorithm.textContent = run.algorithm || "Unknown algorithm";
      primary.append(name, algorithm);

      const statusCell = document.createElement("span");
      statusCell.append(makeStatusBadge(run.status));

      const progress = document.createElement("span");
      progress.className = "row-progress";
      const progressCopy = document.createElement("span");
      progressCopy.className = "row-progress-copy";
      const progressNumber = Number.isFinite(run.progress_percent) ? run.progress_percent : null;
      const progressLabel = document.createElement("span");
      progressLabel.textContent = progressNumber == null ? "—" : Math.round(progressNumber) + "%";
      const progressStatus = document.createElement("span");
      progressStatus.textContent = run.current_stage ? readableName(run.current_stage) : "—";
      progressCopy.append(progressLabel, progressStatus);
      const track = document.createElement("span");
      track.className = "mini-track";
      const fill = document.createElement("span");
      fill.className = "mini-fill";
      fill.style.width = (progressNumber == null ? 0 : Math.max(0, Math.min(100, progressNumber))) + "%";
      track.append(fill);
      progress.append(progressCopy, track);

      const updated = document.createElement("span");
      updated.className = "row-updated";
      updated.textContent = relativeTime(run.updated_at);
      row.append(primary, statusCell, progress, updated);
      row.addEventListener("click", () => selectRun(run.run_id));
      fragment.append(row);
    }

    els.list.replaceChildren(fragment);
    els.empty.classList.toggle("hidden", visible.length !== 0);
    els.list.classList.toggle("hidden", visible.length === 0);
    setText(els.footnote, visible.length + " of " + state.runs.length + " runs");
  }

  function renderMetrics(run) {
    const metrics = run && run.latest_metrics ? Object.entries(run.latest_metrics) : [];
    const fragment = document.createDocumentFragment();
    const usable = metrics.filter(([, value]) =>
      typeof value === "number" && Number.isFinite(value)
    ).slice(0, 12);

    if (!usable.length) {
      const empty = document.createElement("div");
      empty.className = "metrics-empty";
      empty.textContent = "No numeric metrics reported yet.";
      fragment.append(empty);
    } else {
      for (const [key, value] of usable) {
        const card = document.createElement("div");
        card.className = "metric-card";
        const name = document.createElement("span");
        name.className = "metric-name";
        name.textContent = readableName(key);
        const metricValue = document.createElement("strong");
        metricValue.className = "metric-value";
        metricValue.textContent = formatMetric(value);
        card.append(name, metricValue);
        fragment.append(card);
      }
    }
    els.metrics.replaceChildren(fragment);
    setText(els.metricsCaption, usable.length + " values");
  }

  function eventDescription(event) {
    if (event.message) return event.message;
    if (event.error_code) return "Error code: " + event.error_code;
    if (event.stage) return "Stage: " + readableName(event.stage);
    if (event.status) return "Run status: " + readableName(event.status);
    return readableName(event.event_type);
  }

  function renderActivity() {
    const recent = [...state.events].sort((a, b) => b.sequence - a.sequence).slice(0, 80);
    const fragment = document.createDocumentFragment();
    if (!recent.length) {
      const empty = document.createElement("div");
      empty.className = "metrics-empty";
      empty.textContent = "No events to display.";
      fragment.append(empty);
    } else {
      for (const event of recent) {
        const item = document.createElement("div");
        const kind = String(event.event_type || "").toLowerCase();
        item.className = "activity-item" +
          (kind === "warning" ? " activity-warning" : "") +
          (kind === "error" ? " activity-error" : "") +
          (kind === "state_changed" ? " activity-state" : "");

        const rail = document.createElement("span");
        rail.className = "activity-rail";
        const dot = document.createElement("span");
        dot.className = "activity-dot";
        rail.append(dot);

        const main = document.createElement("span");
        main.className = "activity-main";
        const title = document.createElement("strong");
        title.className = "activity-title";
        title.textContent = readableName(event.event_type);
        const description = document.createElement("span");
        description.className = "activity-description";
        description.textContent = eventDescription(event);
        const time = document.createElement("span");
        time.className = "activity-time";
        time.textContent = "#" + event.sequence + " · " + formatDate(event.occurred_at, true);
        main.append(title, description, time);
        item.append(rail, main);
        fragment.append(item);
      }
    }
    els.activity.replaceChildren(fragment);
    const totalEvents = state.selectedRun ? state.selectedRun.event_count || state.events.length : state.events.length;
    setText(
      els.activityCount,
      state.events.length < totalEvents
        ? state.events.length + " recent / " + totalEvents + " total"
        : state.events.length + " events",
    );
  }

  function renderDetails() {
    const run = state.selectedRun;
    els.detailEmpty.classList.toggle("hidden", Boolean(run));
    els.detailContent.classList.toggle("hidden", !run);
    els.streamStatus.classList.toggle("hidden", !run || run.status !== "RUNNING");
    if (!run) {
      els.cancelJobButton.classList.add("hidden");
      setText(els.jobControlNote, "Select a run to inspect its execution controls.");
      return;
    }

    const cancellable = state.selectedJob &&
      (state.selectedJob.status === "QUEUED" || state.selectedJob.status === "RUNNING");
    els.cancelJobButton.classList.toggle("hidden", !cancellable);
    els.cancelJobButton.disabled = Boolean(state.selectedJob && state.selectedJob.cancellation_requested);
    setText(
      els.cancelJobButton,
      state.selectedJob && state.selectedJob.cancellation_requested
        ? "Cancellation requested"
        : "Request cancellation",
    );
    if (cancellable) {
      setText(
        els.jobControlNote,
        state.selectedJob.status === "QUEUED"
          ? "Queued · position " + (state.selectedJob.queue_position || "—")
          : "Running · cancellation is cooperative",
      );
    } else {
      setText(els.jobControlNote, "No active scheduler job for this run.");
    }

    setText(els.detailAlgorithm, run.algorithm);
    els.detailStatus.replaceWith(makeStatusBadge(run.status));
    const newStatus = els.detailContent.querySelector(".status-badge");
    newStatus.id = "detail-status";
    els.detailStatus = newStatus;
    setText(els.detailName, run.name);
    setText(els.detailId, run.run_id);
    setText(els.detailShortId, run.run_id);
    const progress = Number.isFinite(run.progress_percent) ? run.progress_percent : null;
    setText(els.detailProgressValue, progress == null ? "—" : Math.round(progress) + "%");
    els.progressBar.style.width = (progress == null ? 0 : Math.max(0, Math.min(100, progress))) + "%";
    setText(els.detailStage, run.current_stage ? readableName(run.current_stage) : "Awaiting stage");
    setText(els.detailStep, run.current_step != null && run.total_steps != null
      ? run.current_step + " / " + run.total_steps + " steps"
      : "Step not reported");
    setText(els.detailCreated, formatDate(run.created_at, true));
    setText(els.detailDuration, durationFor(run));
    setText(els.detailEventsCount, run.event_count);
    renderMetrics(run);
    renderActivity();
    if (run.status === "RUNNING") {
      setText(els.streamStatus, "Streaming");
    } else {
      setText(els.streamStatus, "Run " + run.status.toLowerCase());
    }
  }

  function closeEventStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    els.streamStatus.classList.add("hidden");
  }

  function mergeEvent(event) {
    if (!event || !Number.isInteger(event.sequence)) return;
    if (state.events.some((item) => item.sequence === event.sequence)) return;
    state.events.push(event);
    state.events.sort((a, b) => a.sequence - b.sequence);
    if (state.events.length > 500) state.events = state.events.slice(-500);

    const current = state.selectedRun;
    if (current && current.run_id === event.run_id) {
      const updated = { ...current, event_count: Math.max(current.event_count || 0, event.sequence) };
      if (event.status) {
        updated.status = event.status;
        if (event.status === "RUNNING" && !updated.started_at) updated.started_at = event.occurred_at;
        if (TERMINAL.has(event.status)) updated.ended_at = event.occurred_at;
      }
      if (event.stage) updated.current_stage = event.stage;
      if (event.progress_percent != null) updated.progress_percent = event.progress_percent;
      if (event.step != null) {
        updated.current_step = event.step;
        updated.total_steps = event.total_steps;
      }
      if (event.metrics && Object.keys(event.metrics).length) {
        updated.latest_metrics = { ...(updated.latest_metrics || {}), ...event.metrics };
      }
      if (event.error_code) updated.error_code = event.error_code;
      if (event.message) updated.message = event.message;
      updated.updated_at = event.occurred_at || updated.updated_at;
      state.selectedRun = updated;
      state.runs = state.runs.map((run) => run.run_id === updated.run_id ? updated : run);
      state.runs.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
      if (state.selectedJob && event.status) {
        state.selectedJob = { ...state.selectedJob, status: event.status };
      }
      if (TERMINAL.has(updated.status)) closeEventStream();
    }

    renderSummary();
    renderRuns();
    renderDetails();
  }

  function openEventStream(runId, afterSequence) {
    closeEventStream();
    if (!window.EventSource) {
      setText(els.streamStatus, "Live stream unavailable");
      return;
    }
    const source = new EventSource(
      API + "/runs/" + encodeURIComponent(runId) +
      "/events/stream?after_sequence=" + encodeURIComponent(afterSequence)
    );
    state.eventSource = source;
    els.streamStatus.classList.remove("hidden");
    setText(els.streamStatus, "Connecting");

    source.onopen = () => {
      if (state.eventSource !== source) return;
      setText(els.streamStatus, "Streaming");
    };
    source.onerror = () => {
      if (state.eventSource !== source) return;
      setText(els.streamStatus, source.readyState === EventSource.CLOSED ? "Stream closed" : "Reconnecting");
    };

    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (message) => {
        if (state.eventSource !== source || state.selectedRunId !== runId) return;
        try {
          mergeEvent(JSON.parse(message.data));
        } catch (_) {
          showToast("An unreadable event was received from the stream.");
        }
      });
    }
    source.addEventListener("stream_end", () => {
      if (state.eventSource !== source || state.selectedRunId !== runId) return;
      source.close();
      state.eventSource = null;
      setText(els.streamStatus, "History replayed");
    });
  }

  async function selectRun(runId) {
    state.selectedRunId = runId;
    state.selectedRun = state.runs.find((run) => run.run_id === runId) || null;
    state.events = [];
    closeEventStream();
    renderRuns();
    renderDetails();

    try {
      const run = await requestJson(API + "/runs/" + encodeURIComponent(runId));
      const afterSequence = Math.max(0, Number(run.event_count || 0) - 200);
      const [events, job] = await Promise.all([
        requestJson(
          API + "/runs/" + encodeURIComponent(runId) +
          "/events?after_sequence=" + afterSequence + "&limit=200"
        ),
        requestJson(API + "/jobs/" + encodeURIComponent(runId)).catch(() => null),
      ]);
      if (state.selectedRunId !== runId) return;
      state.selectedRun = run;
      state.selectedJob = job;
      state.events = events;
      renderDetails();
      renderRuns();
      if (run.status === "RUNNING") {
        const latestSequence = events.reduce(
          (maximum, event) => Math.max(maximum, event.sequence),
          Number(run.event_count || 0) - 200,
        );
        openEventStream(runId, latestSequence);
      }
    } catch (error) {
      if (state.selectedRunId !== runId) return;
      showToast("Could not load run details: " + error.message);
      renderDetails();
    }
  }

  async function refreshRuns(options = {}) {
    if (state.refreshInFlight) return;
    state.refreshInFlight = true;
    try {
      const runs = await requestJson(API + "/runs");
      state.runs = Array.isArray(runs) ? runs : [];
      state.runs.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
      state.lastUpdated = new Date();
      setApiStatus("online", "API connected");
      renderSummary();
      renderRuns();

      if (state.selectedRunId) {
        const selected = state.runs.find((run) => run.run_id === state.selectedRunId);
        if (!selected) {
          state.selectedRunId = null;
          state.selectedRun = null;
          state.selectedJob = null;
          state.events = [];
          closeEventStream();
          renderDetails();
        } else if (options.reloadSelected) {
          await selectRun(state.selectedRunId);
        } else {
          state.selectedRun = selected;
          state.selectedJob = await requestJson(
            API + "/jobs/" + encodeURIComponent(state.selectedRunId)
          ).catch(() => null);
          renderDetails();
          if (selected.status === "RUNNING" && !state.eventSource) {
            const latestSequence = state.events.reduce(
              (maximum, event) => Math.max(maximum, event.sequence),
              0,
            );
            openEventStream(state.selectedRunId, latestSequence);
          } else if (selected.status !== "RUNNING" && state.eventSource) {
            closeEventStream();
          }
        }
      }
      if (options.showToast) showToast("Training runs refreshed.");
    } catch (error) {
      setApiStatus("offline", "API unavailable");
      setText(els.footnote, "Unable to reach the training API");
      if (options.showToast) showToast("Refresh failed: " + error.message);
    } finally {
      state.refreshInFlight = false;
    }
  }

  els.refresh.addEventListener("click", () => refreshRuns({ showToast: true, reloadSelected: true }));
  els.openSubmitDialog.addEventListener("click", async () => {
    await loadTaskCatalog();
    els.submitDialog.showModal();
  });
  els.closeSubmitDialog.addEventListener("click", () => els.submitDialog.close());
  els.submitForm.addEventListener("submit", submitTrainingJob);
  els.cancelJobButton.addEventListener("click", cancelSelectedJob);
  els.search.addEventListener("input", renderRuns);
  els.filter.addEventListener("change", renderRuns);
  els.clearFilters.addEventListener("click", () => {
    els.search.value = "";
    els.filter.value = "ALL";
    renderRuns();
  });

  loadTaskCatalog();
  refreshRuns();
  window.setInterval(() => refreshRuns(), 30000);
})();
