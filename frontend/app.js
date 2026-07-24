const TERMINAL_STAGES = new Set(["complete", "cancelled", "failed"]);

const state = {
  jobs: [],
  jobsSignature: "",
  hasLoaded: false,
  isPolling: false,
  uploading: false,
  connectionLost: false,
  showExcluded: false,
  openDisclosures: new Set(),
  busyActions: new Set(),
  statusTimer: null,
};

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const appStatusEl = document.getElementById("appStatus");
const currentActivityEl = document.getElementById("currentActivity");
const waitingQueueEl = document.getElementById("waitingQueue");
const waitingCountEl = document.getElementById("waitingCount");
const recentResultsEl = document.getElementById("recentResults");
const excludedResultsEl = document.getElementById("excludedResults");
const showExcludedBtn = document.getElementById("showExcludedBtn");
const cleanTerminalBtn = document.getElementById("cleanTerminalBtn");
const clearResultsBtn = document.getElementById("clearResultsBtn");
const stopAllBtn = document.getElementById("stopAllBtn");

fileInput?.addEventListener("change", () => {
  handleUploadedFiles(fileInput.files);
  fileInput.value = "";
});

dropzone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  if (!state.uploading) dropzone.classList.add("dragover");
});

dropzone?.addEventListener("dragleave", (event) => {
  if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove("dragover");
});

dropzone?.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  if (!state.uploading) handleUploadedFiles(event.dataTransfer.files);
});

showExcludedBtn?.addEventListener("click", () => {
  state.showExcluded = !state.showExcluded;
  renderInterface();
});

cleanTerminalBtn?.addEventListener("click", cleanTerminalJobs);
clearResultsBtn?.addEventListener("click", clearResults);
stopAllBtn?.addEventListener("click", stopAllProcesses);

for (const container of [currentActivityEl, waitingQueueEl, recentResultsEl, excludedResultsEl]) {
  container?.addEventListener("click", handleJobAction);
}

async function handleUploadedFiles(fileList) {
  if (!fileList || !fileList.length || state.uploading) return;

  const allFiles = Array.from(fileList);
  const pdfFiles = allFiles.filter((file) => {
    return file.type === "application/pdf" || String(file.name || "").toLowerCase().endsWith(".pdf");
  });
  const ignoredCount = allFiles.length - pdfFiles.length;

  if (!pdfFiles.length) {
    setAppStatus("Choose one or more PDF files.", "error", { sticky: true });
    return;
  }

  state.uploading = true;
  setUploadDisabled(true);
  setAppStatus(`Uploading ${pluralize(pdfFiles.length, "PDF")}…`, "working", { sticky: true });

  try {
    const created = await submitUploadBatch(pdfFiles);
    for (const job of created) {
      state.jobs.push({
        job_id: job.job_id,
        filename: job.filename,
        stage: "upload",
        progress: 0,
        message: "Queued",
        queue_state: "waiting",
        artifacts: {},
      });
    }
    renderInterface();
    const ignored = ignoredCount ? ` ${pluralize(ignoredCount, "non-PDF file")} ignored.` : "";
    setAppStatus(`${pluralize(created.length, "PDF")} added to the queue.${ignored}`, "success");
    await pollJobs();
  } catch (error) {
    setAppStatus(error.message || "The PDFs could not be uploaded.", "error", { sticky: true });
  } finally {
    state.uploading = false;
    setUploadDisabled(false);
  }
}

async function submitUploadBatch(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);

  const response = await fetch("/api/jobs", { method: "POST", body: form });
  const data = await parseJsonResponse(response);
  if (!response.ok) throw new Error(data.detail || "The upload could not be submitted.");
  return Array.isArray(data.jobs) ? data.jobs : [];
}

function setUploadDisabled(disabled) {
  if (fileInput) fileInput.disabled = disabled;
  dropzone?.classList.toggle("disabled", disabled);
  dropzone?.setAttribute("aria-busy", String(disabled));
}

async function pollJobs() {
  if (state.isPolling) return;
  state.isPolling = true;
  try {
    const response = await fetch("/api/jobs?include_archived=true", { cache: "no-store" });
    if (!response.ok) throw new Error(`Job service returned ${response.status}`);
    const jobs = await response.json();
    if (!Array.isArray(jobs)) throw new Error("Job service returned an invalid response");

    const isInitialLoad = !state.hasLoaded;
    const wasDisconnected = state.connectionLost;
    const jobsSignature = JSON.stringify(jobs);
    const jobsChanged = jobsSignature !== state.jobsSignature;
    state.jobs = jobs;
    state.jobsSignature = jobsSignature;
    state.hasLoaded = true;
    state.connectionLost = false;
    if (isInitialLoad || jobsChanged) renderInterface();

    if (wasDisconnected && !state.uploading) {
      setAppStatus("Connection restored.", "success");
    } else if (isInitialLoad && !state.uploading) {
      setAppStatus("Ready for PDF files.", "neutral");
    }
  } catch (_error) {
    state.connectionLost = true;
    if (!state.uploading) {
      setAppStatus("Connection lost. Existing results remain visible while TranslaTHOR retries.", "error", {
        sticky: true,
      });
    }
  } finally {
    state.isPolling = false;
  }
}

function renderInterface() {
  captureDisclosureState();

  const activeJobs = state.jobs.filter((job) => !isArchived(job) && jobQueueState(job) === "active");
  const waitingJobs = state.jobs
    .filter((job) => !isArchived(job) && jobQueueState(job) === "waiting")
    .sort(compareWaitingJobs);
  const recentJobs = state.jobs
    .filter((job) => !isArchived(job) && jobQueueState(job) === "terminal")
    .sort(compareNewestJobs);
  const excludedJobs = state.jobs.filter(isArchived).sort(compareNewestJobs);

  renderCurrentActivity(activeJobs[0] || null, waitingJobs.length);
  renderWaitingQueue(waitingJobs, activeJobs.length > 0);
  renderRecentResults(recentJobs, excludedJobs);
  bindDisclosureState();
}

function renderCurrentActivity(job, waitingCount) {
  if (!currentActivityEl) return;
  currentActivityEl.setAttribute("aria-busy", String(Boolean(job)));

  if (!job) {
    const message = waitingCount
      ? "Nothing is processing yet. The next document will start automatically."
      : "No document is processing right now.";
    currentActivityEl.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
    return;
  }

  const progress = progressPercent(job);
  const stopKey = actionKey("stop", job.job_id);
  currentActivityEl.innerHTML = `
    <article class="current-job job-card">
      <div class="activity-status">
        <h3>${escapeHtml(plainStageLabel(job.stage))}</h3>
        <strong>${progress}%</strong>
      </div>
      ${progressBar(job, progress)}
      <p class="activity-meta">
        <strong>${escapeHtml(job.filename || "Untitled PDF")}</strong>
        <code>${escapeHtml(shortJobId(job))}</code>
        <span>${escapeHtml(waitingCount ? `${pluralize(waitingCount, "document")} waiting` : "Nothing waiting")}</span>
      </p>
      <div class="card-actions">
        ${actionButton({
          action: "stop",
          job,
          label: "Stop",
          busyLabel: "Stopping…",
          busyKey: stopKey,
          className: "danger-outline-button",
        })}
      </div>
      ${jobDetails(job)}
    </article>`;
}

function renderWaitingQueue(jobs, hasActiveJob) {
  if (waitingCountEl) waitingCountEl.textContent = String(jobs.length);
  if (!waitingQueueEl) return;

  if (!jobs.length) {
    waitingQueueEl.innerHTML = '<p class="empty-state">No documents are waiting.</p>';
    return;
  }

  waitingQueueEl.innerHTML = jobs
    .map((job, index) => {
      const ahead = jobsAhead(job, index, hasActiveJob);
      const removeKey = actionKey("remove", job.job_id);
      return `
        <article class="waiting-job job-card">
          <div class="waiting-position" aria-label="Queue position ${index + 1}">${index + 1}</div>
          <div class="waiting-body">
            <div class="job-title-row">
              <h3>${escapeHtml(job.filename || "Untitled PDF")}</h3>
              <code>${escapeHtml(shortJobId(job))}</code>
            </div>
            <p class="job-summary">${ahead ? `${pluralize(ahead, "job")} ahead` : "Next to start"}</p>
            <div class="card-actions">
              ${actionButton({
                action: "remove",
                job,
                label: "Remove from queue",
                busyLabel: "Removing…",
                busyKey: removeKey,
                className: "quiet-button",
              })}
            </div>
            ${jobDetails(job)}
          </div>
        </article>`;
    })
    .join("");
}

function renderRecentResults(recentJobs, excludedJobs) {
  if (recentResultsEl) {
    recentResultsEl.innerHTML = recentJobs.length
      ? recentJobs.map((job) => resultCard(job, false)).join("")
      : '<p class="empty-state">No completed documents yet.</p>';
  }

  if (showExcludedBtn) {
    showExcludedBtn.hidden = excludedJobs.length === 0;
    showExcludedBtn.setAttribute("aria-pressed", String(state.showExcluded));
    showExcludedBtn.textContent = state.showExcluded
      ? `Hide excluded (${excludedJobs.length})`
      : `Show excluded (${excludedJobs.length})`;
  }

  if (excludedResultsEl) {
    excludedResultsEl.hidden = !state.showExcluded || excludedJobs.length === 0;
    excludedResultsEl.innerHTML = state.showExcluded
      ? `
        <div class="excluded-heading"><h3>Excluded</h3><p>Hidden from recent results, but still stored locally.</p></div>
        ${excludedJobs.map((job) => resultCard(job, true)).join("")}`
      : "";
  }
}

function resultCard(job, archived) {
  const complete = job.stage === "complete";
  const archiveAction = archived ? "unarchive" : "archive";
  const archiveKey = actionKey(archiveAction, job.job_id);
  const statusClass = job.stage === "complete" ? "success" : job.stage === "failed" ? "failed" : "stopped";

  return `
    <article class="result-job job-card ${archived ? "is-archived" : ""}">
      <div class="job-title-row">
        <div>
          <span class="status-pill ${statusClass}">${escapeHtml(resultStatusLabel(job.stage))}</span>
          <h3>${escapeHtml(job.filename || "Untitled PDF")}</h3>
        </div>
        <div class="result-identity">
          <code>${escapeHtml(shortJobId(job))}</code>
          ${job.created_at ? `<time datetime="${escapeHtml(job.created_at)}">${escapeHtml(formatDate(job.created_at))}</time>` : ""}
        </div>
      </div>
      <p class="job-summary ${job.stage === "failed" ? "error-summary" : ""}">${escapeHtml(resultSummary(job.stage))}</p>
      ${complete ? primaryPdfActions(job) : ""}
      <div class="card-actions">
        ${actionButton({
          action: archiveAction,
          job,
          label: archived ? "Restore to recent" : "Exclude",
          busyLabel: archived ? "Restoring…" : "Excluding…",
          busyKey: archiveKey,
          className: "quiet-button",
        })}
      </div>
      ${jobDetails(job)}
    </article>`;
}

function primaryPdfActions(job) {
  const readableKey = actionKey("pdf", job.job_id, "readable");
  const originalKey = actionKey("pdf", job.job_id, "original-layout");
  return `
    <div class="primary-outputs" role="group" aria-label="Translated PDF downloads">
      ${actionButton({
        action: "pdf",
        job,
        mode: "readable",
        label: "Readable PDF",
        busyLabel: "Preparing readable PDF…",
        busyKey: readableKey,
        className: "primary-button",
      })}
      ${actionButton({
        action: "pdf",
        job,
        mode: "original-layout",
        label: "Original layout PDF",
        busyLabel: "Preparing original layout…",
        busyKey: originalKey,
        className: "secondary-button",
      })}
    </div>`;
}

function jobDetails(job) {
  const disclosureKey = `job:${job.job_id}`;
  const warnings = jobWarnings(job);
  const reconstructionWarnings = reconstructionWarningCount(job);
  const warningCount = warnings.length + reconstructionWarnings;
  const warningBadge = warningCount
    ? `<span class="warning-badge">${warningCount} ${warningCount === 1 ? "warning" : "warnings"}</span>`
    : "";

  return `
    <details class="job-details" data-disclosure-key="${escapeHtml(disclosureKey)}"${openAttribute(disclosureKey)}>
      <summary><span>View details</span>${warningBadge}</summary>
      <div class="detail-content">
        ${warningsSection(warnings)}
        ${runDetailsSection(job)}
        ${reconstructionDetailsSection(job)}
        ${technicalErrorSection(job)}
        ${permanentDeleteSection(job)}
      </div>
    </details>`;
}

function warningsSection(warnings) {
  return `
    <section class="detail-section warnings-section">
      <h4>Warnings</h4>
      ${
        warnings.length
          ? `<ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
          : '<p class="detail-empty">No run warnings were reported.</p>'
      }
    </section>`;
}

function runDetailsSection(job) {
  const translation = job.translation || {};
  const config = savedRunConfig(job);
  const modelConfig = config.translation_model && typeof config.translation_model === "object"
    ? config.translation_model
    : {};
  const rows = [
    ["Job ID", job.job_id],
    ["Created", job.created_at ? formatDate(job.created_at, true) : ""],
    ["Current status", plainStageLabel(job.stage)],
    ["Status detail", job.message],
    ["Translation model", shortModelName(translation.model || modelConfig.model_id || config.model)],
    ["Document handling", humanizeValue(config.extraction_mode)],
    ["PDF type", humanizeValue(translation.pdf_classification)],
    ["Extraction method", humanizeValue(translation.marker_mode)],
    ["OCR used", booleanLabel(translation.ocr_used)],
    ["OCR fallback", humanizeValue(translation.fallback_engine)],
    ["Local repair", booleanLabel(translation.local_vlm_repair_used)],
    ["Chunk size", config.chunk_size],
    ["Maximum output tokens", modelConfig.max_tokens ?? config.max_tokens],
    ["Temperature", translation.temperature ?? modelConfig.temperature ?? config.temperature],
    ["Top P", translation.top_p ?? modelConfig.top_p ?? config.top_p],
    ["Top K", translation.top_k ?? modelConfig.top_k ?? config.top_k],
    ["Min P", translation.min_p ?? modelConfig.min_p ?? config.min_p],
    ["Presence penalty", translation.presence_penalty ?? modelConfig.presence_penalty ?? config.presence_penalty],
    ["Repetition penalty", translation.repetition_penalty ?? modelConfig.repetition_penalty ?? config.repetition_penalty],
  ].filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "");

  const configDisclosureKey = `config:${job.job_id}`;
  const savedConfig = Object.keys(config).length
    ? `
      <details class="nested-details" data-disclosure-key="${escapeHtml(configDisclosureKey)}"${openAttribute(configDisclosureKey)}>
        <summary>Full saved configuration</summary>
        <pre>${escapeHtml(JSON.stringify(config, null, 2))}</pre>
      </details>`
    : "";

  return `
    <section class="detail-section">
      <h4>Run details and configuration</h4>
      <dl class="detail-grid">
        ${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
      </dl>
      ${savedConfig}
    </section>`;
}

function reconstructionDetailsSection(job) {
  const reconstruction = job.translation?.original_layout_reconstruction;
  const reportLink = artifactLink(job, "reconstruction_report", "Open reconstruction report");
  let content;

  if (reconstruction) {
    const rows = [
      ["Status", humanizeValue(reconstruction.status)],
      ["Pages reconstructed", reconstruction.pages_successfully_reconstructed],
      ["Pages using fallback", reconstruction.pages_using_fallback_behavior],
      ["Reconstruction warnings", reconstruction.warning_count],
    ].filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "");
    const caution = reconstruction.status === "partial"
      ? '<p class="reconstruction-note">Some content was retained unchanged. Use the readable PDF as the safe fallback.</p>'
      : "";
    content = `
      <dl class="detail-grid compact">
        ${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
      </dl>
      ${caution}
      ${reportLink}`;
  } else if (job.stage === "complete") {
    content = `
      <p class="detail-empty">Not prepared yet. The original-layout PDF is built when you request it.</p>
      <p class="reconstruction-note">Scanned or unreliable pages may be retained unchanged. The readable PDF is the safe fallback.</p>`;
  } else {
    content = '<p class="detail-empty">Reconstruction details become available after translation.</p>';
  }

  return `<section class="detail-section"><h4>Original-layout reconstruction</h4>${content}</section>`;
}

function technicalErrorSection(job) {
  if (!job.error) return "";
  const disclosureKey = `error:${job.job_id}`;
  return `
    <section class="detail-section technical-error-section">
      <details class="nested-details" data-disclosure-key="${escapeHtml(disclosureKey)}"${openAttribute(disclosureKey)}>
        <summary>Technical error</summary>
        <pre>${escapeHtml(job.error)}</pre>
      </details>
    </section>`;
}

function permanentDeleteSection(job) {
  if (!TERMINAL_STAGES.has(job.stage)) return "";
  const busyKey = actionKey("delete", job.job_id);
  return `
    <section class="detail-section danger-zone">
      <h4>Permanent deletion</h4>
      <p>Delete this job, its uploaded PDF, and all generated files. This cannot be undone.</p>
      ${actionButton({
        action: "delete",
        job,
        label: "Delete permanently",
        busyLabel: "Deleting…",
        busyKey,
        className: "danger-button",
      })}
    </section>`;
}

function artifactLink(job, type, label) {
  if (!job.artifacts?.[type]) return "";
  return `<a class="detail-link" href="/api/jobs/${encodeURIComponent(job.job_id)}/artifacts/${encodeURIComponent(type)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
}

function actionButton({ action, job, label, busyLabel, busyKey, className, mode = "" }) {
  const busy = state.busyActions.has(busyKey);
  return `
    <button
      type="button"
      class="${escapeHtml(className)}"
      data-action="${escapeHtml(action)}"
      data-job-id="${escapeHtml(job.job_id)}"
      ${mode ? `data-mode="${escapeHtml(mode)}"` : ""}
      ${busy ? "disabled aria-busy=\"true\"" : ""}
    >${escapeHtml(busy ? busyLabel : label)}</button>`;
}

async function handleJobAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button || !event.currentTarget.contains(button)) return;

  const action = button.dataset.action;
  const jobId = button.dataset.jobId;
  const mode = button.dataset.mode || "";
  const job = state.jobs.find((item) => item.job_id === jobId);
  if (!job) {
    setAppStatus("That job is no longer available.", "error", { sticky: true });
    return;
  }

  if (action === "pdf") {
    await preparePdf(job, mode);
  } else if (action === "remove") {
    if (!window.confirm(`Remove “${job.filename}” from the waiting queue?`)) return;
    await changeJob(job, "remove", `/api/jobs/${encodeURIComponent(jobId)}/cancel`, "POST", "Removed from the waiting queue.");
  } else if (action === "stop") {
    if (!window.confirm(`Stop processing “${job.filename}”?`)) return;
    await changeJob(job, "stop", `/api/jobs/${encodeURIComponent(jobId)}/cancel`, "POST", "Stop requested.");
  } else if (action === "archive") {
    await changeJob(job, "archive", `/api/jobs/${encodeURIComponent(jobId)}/archive`, "POST", "Excluded from recent results.");
  } else if (action === "unarchive") {
    await changeJob(job, "unarchive", `/api/jobs/${encodeURIComponent(jobId)}/unarchive`, "POST", "Restored to recent results.");
  } else if (action === "delete") {
    if (!window.confirm(`Permanently delete “${job.filename}” and all of its files?`)) return;
    await changeJob(job, "delete", `/api/jobs/${encodeURIComponent(jobId)}`, "DELETE", "Job permanently deleted.", {
      removeLocally: true,
    });
  }
}

async function changeJob(job, action, url, method, successMessage, options = {}) {
  const key = actionKey(action, job.job_id);
  if (state.busyActions.has(key)) return;
  state.busyActions.add(key);
  renderInterface();

  try {
    const response = await fetch(url, { method });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.detail || `The job could not be ${actionPastTense(action)}.`);
    if (options.removeLocally) state.jobs = state.jobs.filter((item) => item.job_id !== job.job_id);
    setAppStatus(successMessage, "success");
    await pollJobs();
  } catch (error) {
    setAppStatus(error.message || "The job could not be updated.", "error", { sticky: true });
  } finally {
    state.busyActions.delete(key);
    renderInterface();
  }
}

async function preparePdf(job, mode) {
  const key = actionKey("pdf", job.job_id, mode);
  if (state.busyActions.has(key)) return;
  state.busyActions.add(key);
  renderInterface();
  setAppStatus(`Preparing ${mode === "original-layout" ? "the original-layout" : "the readable"} PDF…`, "working", {
    sticky: true,
  });

  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(job.job_id)}/pdf/${encodeURIComponent(mode)}`);
    if (!response.ok) {
      const data = await parseJsonResponse(response);
      throw new Error(data.detail || "The PDF could not be prepared.");
    }
    const blob = await response.blob();
    const fallbackName = `${filenameStem(job.filename || "translation")}_${mode.replaceAll("-", "_")}.pdf`;
    downloadBlob(blob, responseFilename(response) || fallbackName);
    setAppStatus(`${mode === "original-layout" ? "Original-layout" : "Readable"} PDF is ready.`, "success");
    await pollJobs();
  } catch (error) {
    setAppStatus(error.message || "The PDF could not be prepared.", "error", { sticky: true });
  } finally {
    state.busyActions.delete(key);
    renderInterface();
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function responseFilename(response) {
  const disposition = response.headers.get("content-disposition") || "";
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1].replace(/^"|"$/g, ""));
    } catch (_error) {
      return encoded[1].replace(/^"|"$/g, "");
    }
  }
  const plain = disposition.match(/filename="?([^";]+)"?/i);
  return plain ? plain[1] : "";
}

async function clearResults() {
  if (!window.confirm("Permanently delete every uploaded PDF, job, and generated result?")) return;
  await runBulkAction(clearResultsBtn, "/api/jobs", "DELETE", "All jobs and files were deleted.", () => {
    state.jobs = [];
  });
}

async function cleanTerminalJobs() {
  if (!window.confirm("Permanently delete all failed and stopped jobs?")) return;
  await runBulkAction(
    cleanTerminalBtn,
    "/api/jobs/cleanup-terminal",
    "DELETE",
    "Failed and stopped jobs were deleted.",
  );
}

async function stopAllProcesses() {
  if (!window.confirm("Stop the active job and remove every waiting job from the queue?")) return;
  await runBulkAction(stopAllBtn, "/api/jobs/stop-all", "POST", "Stop requested for active and waiting jobs.");
}

async function runBulkAction(button, url, method, successMessage, onSuccess = null) {
  if (!button || button.disabled) return;
  button.disabled = true;
  try {
    const response = await fetch(url, { method });
    const data = await parseJsonResponse(response);
    if (!response.ok) throw new Error(data.detail || "The management action failed.");
    if (onSuccess) onSuccess(data);
    setAppStatus(successMessage, "success");
    await pollJobs();
  } catch (error) {
    setAppStatus(error.message || "The management action failed.", "error", { sticky: true });
  } finally {
    button.disabled = false;
    renderInterface();
  }
}

function setAppStatus(message, tone = "neutral", options = {}) {
  if (!appStatusEl) return;
  if (state.statusTimer) window.clearTimeout(state.statusTimer);
  appStatusEl.textContent = message;
  appStatusEl.className = `app-status ${tone ? `is-${tone}` : ""}`;
  appStatusEl.setAttribute("role", tone === "error" ? "alert" : "status");
  appStatusEl.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  appStatusEl.setAttribute("aria-atomic", "true");

  if (!options.sticky && (tone === "success" || tone === "neutral")) {
    state.statusTimer = window.setTimeout(() => {
      if (!state.uploading && !state.connectionLost) {
        appStatusEl.textContent = "";
        appStatusEl.className = "app-status is-empty";
      }
    }, 5000);
  }
}

function jobQueueState(job) {
  if (TERMINAL_STAGES.has(job.stage)) return "terminal";
  const queueState = String(job.queue_state || "").toLowerCase();
  if (["waiting", "queued", "pending"].includes(queueState)) return "waiting";
  if (["active", "running", "processing"].includes(queueState)) return "active";
  return job.stage === "upload" ? "waiting" : "active";
}

function isArchived(job) {
  return Boolean(job.archived_at);
}

function compareWaitingJobs(left, right) {
  const leftPosition = finiteNumber(left.queue_position, Number.POSITIVE_INFINITY);
  const rightPosition = finiteNumber(right.queue_position, Number.POSITIVE_INFINITY);
  if (leftPosition !== rightPosition) return leftPosition - rightPosition;
  return createdTimestamp(left) - createdTimestamp(right);
}

function compareNewestJobs(left, right) {
  const timestampDifference = resultTimestamp(right) - resultTimestamp(left);
  if (timestampDifference) return timestampDifference;
  // Legacy statuses may not have timestamps. The old API returned FIFO order,
  // so reverse their stable source order for the recent-results view.
  return state.jobs.indexOf(right) - state.jobs.indexOf(left);
}

function createdTimestamp(job) {
  const value = Date.parse(job.queued_at || job.created_at || "");
  return Number.isFinite(value) ? value : 0;
}

function resultTimestamp(job) {
  const explicit = Date.parse(job.completed_at || job.created_at || job.archived_at || "");
  if (Number.isFinite(explicit)) return explicit;
  const embedded = String(job.message || "").match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/);
  const parsed = embedded ? Date.parse(embedded[0]) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : 0;
}

function jobsAhead(job, index, hasActiveJob) {
  const reported = finiteNumber(job.jobs_ahead, Number.NaN);
  if (Number.isFinite(reported) && reported >= 0) return reported;
  return index + (hasActiveJob ? 1 : 0);
}

function progressPercent(job) {
  const value = Number(job.progress);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

function progressBar(job, progress) {
  return `
    <div
      class="progress"
      role="progressbar"
      aria-label="${escapeHtml(`${plainStageLabel(job.stage)} progress`)}"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow="${progress}"
    >
      <span style="width:${progress}%"></span>
    </div>`;
}

function plainStageLabel(stage) {
  const labels = {
    upload: "Waiting to start",
    extraction: "Inspecting the document",
    ocr_layout_parsing: "Reading the document",
    structure_generation: "Preparing the document",
    translation: "Translating",
    pdf_generation: "Finishing the translation",
    complete: "Ready to download",
    cancelled: "Stopped",
    failed: "Could not finish",
  };
  return labels[stage] || "Processing";
}

function resultStatusLabel(stage) {
  return stage === "complete" ? "Ready" : stage === "failed" ? "Failed" : "Stopped";
}

function resultSummary(stage) {
  if (stage === "complete") return "Translation is ready to download.";
  if (stage === "failed") return "This document could not be processed. Open details for technical information.";
  return "Processing was stopped before the translation completed.";
}

function jobWarnings(job) {
  const values = [
    ...(Array.isArray(job.warnings) ? job.warnings : []),
    ...(Array.isArray(job.translation?.warnings) ? job.translation.warnings : []),
  ];
  return [...new Set(values.map((value) => String(value).trim()).filter(Boolean))];
}

function reconstructionWarningCount(job) {
  const value = Number(job.translation?.original_layout_reconstruction?.warning_count);
  return Number.isFinite(value) && value > 0 ? Math.round(value) : 0;
}

function savedRunConfig(job) {
  const candidates = [
    job.run_config,
    job.settings,
    job.config,
    job.translation?.run_config,
    job.translation?.settings,
  ];
  return candidates.find((value) => value && typeof value === "object" && !Array.isArray(value)) || {};
}

function booleanLabel(value) {
  return typeof value === "boolean" ? (value ? "Yes" : "No") : "";
}

function shortModelName(value) {
  const model = String(value || "").trim();
  return model ? model.split("/").pop() : "";
}

function humanizeValue(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  return text.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatDate(value, includeSeconds = false) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "");
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: includeSeconds ? "medium" : "short",
  }).format(date);
}

function shortJobId(job) {
  return String(job.job_id || "").slice(0, 8);
}

function finiteNumber(value, fallback) {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function pluralize(count, singular) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function actionKey(action, jobId, mode = "") {
  return [action, jobId, mode].filter(Boolean).join(":");
}

function actionPastTense(action) {
  const labels = {
    archive: "excluded",
    unarchive: "restored",
    delete: "deleted",
    remove: "removed",
    stop: "stopped",
  };
  return labels[action] || "updated";
}

function filenameStem(filename) {
  return String(filename || "translation").replace(/\.pdf$/i, "").replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function captureDisclosureState() {
  for (const details of document.querySelectorAll("details[data-disclosure-key]")) {
    const key = details.dataset.disclosureKey;
    if (!key) continue;
    if (details.open) state.openDisclosures.add(key);
    else state.openDisclosures.delete(key);
  }
}

function bindDisclosureState() {
  for (const details of document.querySelectorAll("details[data-disclosure-key]")) {
    details.addEventListener("toggle", () => {
      const key = details.dataset.disclosureKey;
      if (!key) return;
      if (details.open) state.openDisclosures.add(key);
      else state.openDisclosures.delete(key);
    });
  }
}

function openAttribute(key) {
  return state.openDisclosures.has(key) ? " open" : "";
}

async function parseJsonResponse(response) {
  try {
    return await response.json();
  } catch (_error) {
    return {};
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.setInterval(pollJobs, 2000);
pollJobs();
