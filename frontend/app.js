const BOX_TYPES = ["page", "text", "table", "figure", "caption", "header", "footer", "reference", "other"];
const WORKFLOW_STEPS = ["upload", "select", "ocr", "translate", "export"];

const state = {
  jobs: [],
  staged: [],
  stagingCounter: 0,
  submittingStaged: false,
  region: createEmptyRegionState(),
};

function createEmptyRegionState() {
  return {
    jobId: null,
    filename: "",
    pageCount: 0,
    currentPage: 1,
    pages: {},
    loading: false,
    regionsDetected: false,
    hasSavedBoxes: false,
    statusMessage: "",
    hasUnsavedChanges: false,
    ocrStatus: "not_started",
    translationStatus: "not_started",
    workflowStep: "upload",
  };
}

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const pickBtn = document.getElementById("pickBtn");
const queueEl = document.getElementById("queue");
const stagedQueueEl = document.getElementById("stagedQueue");
const parseTranslateBtn = document.getElementById("parseTranslateBtn");
const clearStagedBtn = document.getElementById("clearStagedBtn");
const cleanTerminalBtn = document.getElementById("cleanTerminalBtn");
const clearResultsBtn = document.getElementById("clearResultsBtn");
const stopAllBtn = document.getElementById("stopAllBtn");

const openRegionFromFileBtn = document.getElementById("openRegionFromFileBtn");
const startSelectedModeBtn = document.getElementById("startSelectedModeBtn");
const startFullPageModeBtn = document.getElementById("startFullPageModeBtn");
const regionStatusEl = document.getElementById("regionStatus");
const prevPageBtn = document.getElementById("prevPageBtn");
const nextPageBtn = document.getElementById("nextPageBtn");
const useFullPageBtn = document.getElementById("useFullPageBtn");
const useFullPageAllBtn = document.getElementById("useFullPageAllBtn");
const autoDetectBtn = document.getElementById("autoDetectBtn");
const saveBoxesBtn = document.getElementById("saveBoxesBtn");
const reloadSavedBoxesBtn = document.getElementById("reloadSavedBoxesBtn");
const selectAllBoxesBtn = document.getElementById("selectAllBoxesBtn");
const deselectAllBoxesBtn = document.getElementById("deselectAllBoxesBtn");
const deleteSelectedBoxesBtn = document.getElementById("deleteSelectedBoxesBtn");
const clearPageRegionsBtn = document.getElementById("clearPageRegionsBtn");
const addBoxBtn = document.getElementById("addBoxBtn");
const runOcrSelectedBtn = document.getElementById("runOcrSelectedBtn");
const boxListEl = document.getElementById("boxList");
const regionCanvasEl = document.getElementById("regionCanvas");
const workflowStepsEl = document.getElementById("workflowSteps");

const canvasState = {
  stage: null,
  imageLayer: null,
  boxLayer: null,
  transformer: null,
  boxNodes: new Map(),
  activeBoxId: null,
};

pickBtn?.addEventListener("click", () => fileInput.click());
fileInput?.addEventListener("change", () => {
  handleUploadedFiles(fileInput.files);
  fileInput.value = "";
});
clearResultsBtn?.addEventListener("click", clearResults);
stopAllBtn?.addEventListener("click", stopAllProcesses);
parseTranslateBtn?.addEventListener("click", submitStagedJobs);
clearStagedBtn?.addEventListener("click", clearStagedJobs);
cleanTerminalBtn?.addEventListener("click", cleanTerminalJobs);

openRegionFromFileBtn?.addEventListener("click", openRegionFromFilePicker);
startSelectedModeBtn?.addEventListener("click", () => startRegionJob("selected_regions"));
startFullPageModeBtn?.addEventListener("click", () => startRegionJob("full_page"));
prevPageBtn?.addEventListener("click", () => gotoPage(state.region.currentPage - 1));
nextPageBtn?.addEventListener("click", () => gotoPage(state.region.currentPage + 1));
useFullPageBtn?.addEventListener("click", useFullPageForCurrentPage);
useFullPageAllBtn?.addEventListener("click", useFullPageForAllPages);
autoDetectBtn?.addEventListener("click", detectDetailedBoxes);
saveBoxesBtn?.addEventListener("click", saveLoadedBoxesFromButton);
reloadSavedBoxesBtn?.addEventListener("click", reloadSavedBoxes);
selectAllBoxesBtn?.addEventListener("click", () => setAllBoxesSelected(true));
deselectAllBoxesBtn?.addEventListener("click", () => setAllBoxesSelected(false));
deleteSelectedBoxesBtn?.addEventListener("click", deleteSelectedBoxes);
clearPageRegionsBtn?.addEventListener("click", clearCurrentPageRegions);
addBoxBtn?.addEventListener("click", addManualBox);
runOcrSelectedBtn?.addEventListener("click", runSelectedOcr);

window.addEventListener("resize", () => {
  if (!state.region.jobId) return;
  renderCurrentPageCanvas();
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  handleUploadedFiles(e.dataTransfer.files);
});

async function handleUploadedFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const pdfFiles = Array.from(fileList).filter((file) => {
    return file.type === "application/pdf" || String(file.name || "").toLowerCase().endsWith(".pdf");
  });
  if (!pdfFiles.length) return;

  try {
    await openRegionFromFile(pdfFiles[0]);
    if (pdfFiles.length > 1) window.alert("Opened the first PDF. Add another PDF after finishing this one.");
  } catch (error) {
    window.alert(error.message || "Unable to upload PDF for OCR region selection");
  }
}

function stageFiles(fileList) {
  if (!fileList || !fileList.length) return;
  for (const file of fileList) {
    const isPdf = file.type === "application/pdf" || String(file.name || "").toLowerCase().endsWith(".pdf");
    if (!isPdf) continue;
    state.staged.unshift({
      local_id: nextStagedId(),
      type: "upload",
      filename: file.name,
      file,
    });
  }
  renderStagedQueue();
}

function stageReuseJob(job) {
  state.staged.unshift({
    local_id: nextStagedId(),
    type: "reuse",
    filename: job.filename,
    source_job_id: job.job_id,
  });
  renderStagedQueue();
}

function nextStagedId() {
  state.stagingCounter += 1;
  return `staged-${Date.now()}-${state.stagingCounter}`;
}

function removeStagedItem(localId) {
  state.staged = state.staged.filter((item) => item.local_id !== localId);
  renderStagedQueue();
}

function renderStagedQueue() {
  if (!stagedQueueEl || !parseTranslateBtn || !clearStagedBtn) return;
  parseTranslateBtn.disabled = state.submittingStaged || state.staged.length === 0;
  clearStagedBtn.disabled = state.submittingStaged || state.staged.length === 0;
  parseTranslateBtn.textContent = state.staged.length
    ? `Start (${state.staged.length})`
    : "Start";

  stagedQueueEl.innerHTML = "";
  if (!state.staged.length) {
    stagedQueueEl.innerHTML = '<p class="empty-state">No staged documents.</p>';
    return;
  }

  for (const item of state.staged) {
    const row = document.createElement("div");
    row.className = "staged-item";
    const subtype = item.type === "reuse" ? `Reuse OCR from ${item.source_job_id.slice(0, 8)}` : "New PDF upload";
    const regionBtn = item.type === "upload"
      ? `<button type="button" data-region-staged="${item.local_id}" class="secondary-button">Open OCR Regions</button>`
      : "";

    row.innerHTML = `
      <div class="staged-item-main">
        <strong>${escapeHtml(item.filename)}</strong>
        <small>${escapeHtml(subtype)}</small>
      </div>
      <div class="queue-actions">
        ${regionBtn}
        <button type="button" data-remove-staged="${item.local_id}" class="secondary-button">Remove</button>
      </div>
    `;

    const regionOpenBtn = row.querySelector("button[data-region-staged]");
    if (regionOpenBtn) {
      regionOpenBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        await openRegionFromStagedUpload(item);
      });
    }

    const removeBtn = row.querySelector("button[data-remove-staged]");
    if (removeBtn) {
      removeBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        removeStagedItem(item.local_id);
      });
    }
    stagedQueueEl.appendChild(row);
  }
}

async function submitStagedJobs() {
  if (state.submittingStaged || !state.staged.length) return;
  state.submittingStaged = true;
  renderStagedQueue();

  const uploadItems = state.staged.filter((item) => item.type === "upload");
  const reuseItems = state.staged.filter((item) => item.type === "reuse");
  const failed = [];

  try {
    if (uploadItems.length) {
      try {
        await submitUploadBatch(uploadItems);
      } catch (_error) {
        failed.push(...uploadItems);
      }
    }

    for (const item of reuseItems) {
      try {
        await submitReuseItem(item);
      } catch (_error) {
        failed.push(item);
      }
    }

    state.staged = failed;
    if (failed.length) {
      window.alert(`${failed.length} staged item(s) failed to submit. They were kept in the staged list.`);
    }
    await pollJobs();
  } finally {
    state.submittingStaged = false;
    renderStagedQueue();
  }
}

async function submitUploadBatch(uploadItems) {
  const form = new FormData();
  for (const item of uploadItems) {
    form.append("files", item.file, item.filename);
  }
  appendTranslationFormFields(form);

  const res = await fetch("/api/jobs", { method: "POST", body: form });
  const data = await parseJsonResponse(res);
  if (!res.ok) throw new Error(data.detail || "Upload submission failed");

  if (data.jobs) {
    for (const job of data.jobs) {
      state.jobs.unshift({
        job_id: job.job_id,
        filename: job.filename,
        stage: "upload",
        progress: 0,
        message: "Queued",
        artifacts: {},
      });
    }
    renderQueue();
  }
}

async function submitReuseItem(item) {
  const payload = buildTranslationPayload();
  const res = await fetch(`/api/jobs/${item.source_job_id}/retranslate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await parseJsonResponse(res);
  if (!res.ok) throw new Error(data.detail || "Reuse OCR submission failed");

  if (data.job) {
    state.jobs.unshift({
      job_id: data.job.job_id,
      filename: data.job.filename || item.filename,
      stage: "upload",
      progress: 0,
      message: "Queued (reusing OCR cache)",
      artifacts: {},
    });
    renderQueue();
  }
}

function appendTranslationFormFields(form) {
  form.append("chunk_size", getInputValue("chunkSize", "1800"));
  form.append("temperature", getInputValue("temp", "0.2"));
  form.append("max_tokens", getInputValue("maxTokens", "2048"));
  form.append("model", getInputValue("model", "mlx-community/Qwen3.5-9B-8bit"));
  form.append("top_p", getInputValue("topP", "0.9"));
  form.append("output_mode", "readable");
}

function buildTranslationPayload() {
  return {
    chunk_size: numberInput("chunkSize", 1800),
    model: getInputValue("model", "mlx-community/Qwen3.5-9B-8bit"),
    temperature: numberInput("temp", 0.2),
    top_p: numberInput("topP", 0.9),
    max_tokens: numberInput("maxTokens", 2048),
    output_mode: "readable",
    profile_pipeline: false,
    translation_input_mode: "continuous_document",
  };
}

async function parseJsonResponse(res) {
  try {
    return await res.json();
  } catch (_error) {
    return {};
  }
}

async function pollJobs() {
  const res = await fetch("/api/jobs");
  const jobs = await res.json();
  state.jobs = jobs;
  syncWorkflowFromCurrentJob();
  renderQueue();
  renderRegionStatus();
}

function syncWorkflowFromCurrentJob() {
  if (!state.region.jobId) return;
  const job = state.jobs.find((item) => item.job_id === state.region.jobId);
  if (!job) {
    state.region = createEmptyRegionState();
    return;
  }

  if (job.stage === "complete") {
    state.region.ocrStatus = "completed";
    state.region.translationStatus = "completed";
    state.region.workflowStep = "export";
  } else if (job.stage === "failed" || job.stage === "cancelled") {
    if (state.region.ocrStatus === "running") state.region.ocrStatus = "failed";
    if (state.region.translationStatus === "running") {
      state.region.translationStatus = "failed";
      state.region.workflowStep = "translate";
    }
  } else if (job.stage === "ocr_layout_parsing") {
    state.region.ocrStatus = "running";
    state.region.workflowStep = "ocr";
    state.region.statusMessage = job.message || "Running OCR on selected regions";
  } else if (state.region.translationStatus === "running") {
    state.region.workflowStep = "translate";
  }
}

async function clearResults() {
  const confirmed = window.confirm("Delete all uploaded PDFs and generated results?");
  if (!confirmed) return;

  clearResultsBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs", { method: "DELETE" });
    if (!res.ok) throw new Error("Cleanup failed");
    state.jobs = [];
    state.region = createEmptyRegionState();
    renderQueue();
    renderRegionStatus();
  } catch (error) {
    window.alert(error.message || "Cleanup failed");
  } finally {
    clearResultsBtn.disabled = false;
  }
}

async function cleanTerminalJobs() {
  const confirmed = window.confirm("Remove cancelled and failed jobs from the list?");
  if (!confirmed) return;

  cleanTerminalBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs/cleanup-terminal", { method: "DELETE" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Cleanup failed");
    await pollJobs();
  } catch (error) {
    window.alert(error.message || "Cleanup failed");
  } finally {
    cleanTerminalBtn.disabled = false;
  }
}

async function stopAllProcesses() {
  const confirmed = window.confirm("Stop the active job and cancel all queued jobs?");
  if (!confirmed) return;

  stopAllBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs/stop-all", { method: "POST" });
    if (!res.ok) throw new Error("Stop-all failed");
    await pollJobs();
  } catch (error) {
    window.alert(error.message || "Stop-all failed");
  } finally {
    stopAllBtn.disabled = false;
  }
}

async function clearStagedJobs() {
  if (!state.staged.length) return;
  const confirmed = window.confirm("Remove all staged documents?");
  if (!confirmed) return;
  state.staged = [];
  renderStagedQueue();
}

function stageLabel(stage) {
  const map = {
    upload: "Upload",
    extraction: "Extraction",
    ocr_layout_parsing: "OCR/Layout",
    structure_generation: "Structure",
    translation: "Translation",
    pdf_generation: "PDF Generation",
    complete: "Complete",
    cancelled: "Cancelled",
    failed: "Failed",
  };
  return map[stage] || stage;
}

function renderQueue() {
  if (!queueEl) return;
  queueEl.innerHTML = "";
  if (!state.jobs.length) {
    queueEl.innerHTML = '<p class="empty-state">No documents in the queue.</p>';
    return;
  }
  for (const job of state.jobs) {
    const item = document.createElement("div");
    item.className = "job-item";

    const progressWidth = Math.max(2, Math.round((job.progress || 0) * 100));
    const progressText = progressBarLabel(job, progressWidth);

    item.innerHTML = `
      <div class="job-head">
        <strong>${escapeHtml(job.filename)}</strong>
        <code>${job.job_id.slice(0, 8)}</code>
      </div>
      <div class="progress" aria-label="${escapeHtml(progressText)}">
        <span style="width:${progressWidth}%"></span>
        <small>${escapeHtml(progressText)}</small>
      </div>
      <div class="stage">${stageLabel(job.stage)} - ${escapeHtml(job.message || "")}</div>
      ${translationInfoLine(job)}
      ${job.error ? `<div class="error">${escapeHtml(job.error)}</div>` : ""}
      <div class="downloads">
        ${cancelQueuedButton(job)}
        ${reloadOcrButton(job)}
        ${pdfDownloadLink(job, "readable", "Readable PDF")}
        ${pdfDownloadLink(job, "faithful", "Faithful PDF")}
        ${downloadLink(job, "markdown", "Markdown")}
        ${downloadLink(job, "json", "JSON")}
        ${downloadLink(job, "profile_summary", "Timing Summary")}
        ${downloadLink(job, "profile_json", "Timing JSON")}
        ${downloadLink(job, "profile_csv", "Timing CSV")}
      </div>
    `;

    const cancelBtn = item.querySelector("button[data-cancel-queued]");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        cancelQueuedJob(job);
      });
    }

    const reloadBtn = item.querySelector("button[data-reload-ocr]");
    if (reloadBtn) {
      reloadBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        await createEditableOcrRerun(job);
      });
    }

    queueEl.appendChild(item);
  }
}

function progressBarLabel(job, progressWidth) {
  const message = String(job.message || "");
  const counter = message.match(/\b\d+\s*\/\s*\d+\b/);
  if (job.stage === "ocr_layout_parsing" && counter) {
    return `OCR ${counter[0].replace(/\s+/g, "")}`;
  }
  return `${progressWidth}%`;
}

function downloadLink(job, type, label) {
  if (!job.artifacts || !job.artifacts[type]) return "";
  return `<a href="/api/jobs/${job.job_id}/artifacts/${type}" target="_blank"><button>${label}</button></a>`;
}

function reloadOcrButton(job) {
  const hasSavedBoxes = Boolean(job.artifacts && job.artifacts.ocr_regions);
  if (!hasSavedBoxes) return "";
  return `<button type="button" data-reload-ocr="${job.job_id}" class="secondary-button">Reload OCR</button>`;
}

async function createEditableOcrRerun(job) {
  if (!job || !job.job_id) return;
  try {
    const res = await fetch(`/api/jobs/${job.job_id}/duplicate-for-ocr-rerun`, { method: "POST" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Unable to create OCR rerun");
    await pollJobs();
    await openRegionEditor(data.new_job_id, data.filename || `${job.filename} OCR rerun`, {
      reloadSaved: true,
      statusMessage: data.message || "Created editable OCR rerun from previous job.",
    });
  } catch (error) {
    window.alert(error.message || "Unable to create OCR rerun");
  }
}

function pdfDownloadLink(job, mode, label) {
  const canGeneratePdf = job.stage === "complete" || Boolean(job.artifacts && job.artifacts.markdown);
  if (!canGeneratePdf) return "";
  return `<a href="/api/jobs/${job.job_id}/pdf/${mode}" target="_blank"><button>${label}</button></a>`;
}

function translationInfoLine(job) {
  if (job.stage !== "complete" || !job.translation) return "";
  const model = String(job.translation.model || "").trim();
  if (!model) return "";
  const modelLabel = model.split("/").pop() || model;
  const temp = Number(job.translation.temperature);
  const topP = Number(job.translation.top_p);
  const tempText = Number.isFinite(temp) ? temp.toFixed(2) : "n/a";
  const topPText = Number.isFinite(topP) ? topP.toFixed(2) : "n/a";
  return `<div class="meta-line">Model: ${escapeHtml(modelLabel)} | temp: ${escapeHtml(tempText)} | top-p: ${escapeHtml(topPText)}</div>`;
}

function retranslateButton(job) {
  return "";
}

function cancelQueuedButton(job) {
  const canCancelQueued = job.stage === "upload";
  if (!canCancelQueued) return "";
  return `<button type="button" data-cancel-queued="${job.job_id}">Cancel Queued Job</button>`;
}

async function cancelQueuedJob(job) {
  const confirmed = window.confirm("Cancel this queued job?");
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Unable to cancel queued job");
    await pollJobs();
  } catch (error) {
    window.alert(error.message || "Unable to cancel queued job");
  }
}

async function openRegionFromStagedUpload(item) {
  if (!item || item.type !== "upload" || !item.file) return;
  const draft = await openRegionFromFile(item.file);
  removeStagedItem(item.local_id);
  return draft;
}

async function openRegionFromFilePicker() {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = "application/pdf";
  picker.multiple = false;
  picker.addEventListener("change", async () => {
    const file = picker.files && picker.files[0];
    if (!file) return;
    try {
      await openRegionFromFile(file);
    } catch (error) {
      window.alert(error.message || "Unable to open region editor");
    }
  });
  picker.click();
}

async function openRegionFromFile(file) {
  setWorkflowBusy("upload", true);
  try {
    const draft = await createDraftJob(file);
    await openRegionEditor(draft.job_id, draft.filename || file.name);
    return draft;
  } finally {
    setWorkflowBusy("upload", false);
  }
}

async function createDraftJob(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch("/api/jobs/draft", { method: "POST", body: form });
  const data = await parseJsonResponse(res);
  if (!res.ok) throw new Error(data.detail || "Draft upload failed");
  return data;
}

async function openRegionEditor(jobId, filename, options = {}) {
  state.region = {
    ...createEmptyRegionState(),
    jobId,
    filename: filename || "",
    workflowStep: "select",
  };
  await loadPdfMetadata();
  await loadSavedBoxesSummary();
  if (options.statusMessage) {
    state.region.statusMessage = options.statusMessage;
  } else if (options.reloadSaved && state.region.hasSavedBoxes) {
    state.region.statusMessage = "Loaded saved OCR regions from previous session.";
  }
  await gotoPage(1);
  document.querySelector(".region-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  renderRegionStatus();
}

async function loadPdfMetadata() {
  const jobId = state.region.jobId;
  if (!jobId) return;
  const res = await fetch(`/api/jobs/${jobId}/pdf-metadata`);
  const data = await parseJsonResponse(res);
  if (!res.ok) throw new Error(data.detail || "Unable to read PDF metadata");
  state.region.pageCount = Number(data.page_count || 0);
}

async function loadSavedBoxesSummary() {
  const jobId = state.region.jobId;
  if (!jobId) return;
  const res = await fetch(`/api/jobs/${jobId}/boxes/summary`);
  const data = await parseJsonResponse(res);
  if (!res.ok) {
    state.region.hasSavedBoxes = false;
    return;
  }
  state.region.hasSavedBoxes = Boolean(data.has_saved_boxes);
}

async function gotoPage(pageNumber) {
  if (!state.region.jobId) return;
  const pageCount = state.region.pageCount;
  if (pageCount < 1) return;
  if (state.region.pages[String(state.region.currentPage)]) {
    await saveCurrentPageBoxes();
  }
  const next = Math.min(pageCount, Math.max(1, Number(pageNumber) || 1));
  state.region.currentPage = next;
  await loadPageState(next, false);
  renderRegionStatus();
}

async function loadPageState(pageNumber, refreshDetect) {
  const key = String(pageNumber);
  const jobId = state.region.jobId;
  if (!jobId) return;

  state.region.loading = true;
  renderRegionStatus();
  try {
    const boxesPayload = await fetchPageBoxes(pageNumber, refreshDetect, false);

    const imageUrl = await fetchPageImageUrl(jobId, pageNumber);

    state.region.pages[key] = pageStateFromBoxesPayload(boxesPayload, imageUrl);
    state.region.regionsDetected = true;

    await renderCurrentPageCanvas();
    renderBoxList();
  } finally {
    state.region.loading = false;
    renderRegionStatus();
  }
}

function currentPageState() {
  return state.region.pages[String(state.region.currentPage)] || null;
}

async function fetchPageImageUrl(jobId, pageNumber) {
  const imageRes = await fetch(`/api/jobs/${jobId}/pages/${pageNumber}/image?dpi=150`);
  if (!imageRes.ok) {
    const err = await parseJsonResponse(imageRes);
    throw new Error(err.detail || "Unable to load page image");
  }
  const imageBlob = await imageRes.blob();
  return URL.createObjectURL(imageBlob);
}

function renderRegionStatus() {
  const hasJob = Boolean(state.region.jobId);
  const selectedCount = countSelectedRegions();
  if (!hasJob) {
    regionStatusEl.textContent = "Upload a PDF to start selecting OCR regions.";
  } else if (state.region.loading) {
    regionStatusEl.textContent = `Loading ${state.region.filename || state.region.jobId} page ${state.region.currentPage}...`;
  } else {
    const dirty = state.region.hasUnsavedChanges ? " | unsaved edits" : "";
    const ocrLabel = state.region.ocrStatus.replace("_", " ");
    const translationLabel = state.region.translationStatus.replace("_", " ");
    const message = state.region.statusMessage ? ` | ${state.region.statusMessage}` : "";
    regionStatusEl.textContent = `${state.region.filename} | page ${state.region.currentPage}/${state.region.pageCount} | OCR regions: ${selectedCount} | OCR: ${ocrLabel} | translation: ${translationLabel}${dirty}${message}`;
  }

  const disabled = !hasJob || state.region.loading;
  const hasSelectedRegions = selectedCount > 0;
  if (runOcrSelectedBtn) {
    const hideOcrAction = hasJob && state.region.ocrStatus === "completed";
    runOcrSelectedBtn.hidden = hideOcrAction;
    runOcrSelectedBtn.textContent =
      state.region.ocrStatus === "running" ? `OCR running (${selectedCount} regions)` : "Next: OCR selected regions";
    runOcrSelectedBtn.disabled = disabled || !hasSelectedRegions || state.region.ocrStatus === "running";
  }
  if (startSelectedModeBtn) {
    startSelectedModeBtn.hidden = !hasJob || (
      state.region.ocrStatus !== "completed"
      || state.region.translationStatus === "completed"
    );
    startSelectedModeBtn.textContent = state.region.translationStatus === "running"
      ? "Translating OCR output"
      : state.region.translationStatus === "failed"
        ? "Retry translation"
        : "Translate now";
    startSelectedModeBtn.disabled = disabled || state.region.ocrStatus !== "completed" || state.region.translationStatus === "running";
  }
  if (prevPageBtn) prevPageBtn.disabled = disabled || state.region.currentPage <= 1;
  if (nextPageBtn) nextPageBtn.disabled = disabled || state.region.currentPage >= state.region.pageCount;
  if (autoDetectBtn) autoDetectBtn.disabled = disabled;
  if (useFullPageBtn) useFullPageBtn.disabled = disabled;
  if (useFullPageAllBtn) useFullPageAllBtn.disabled = disabled;
  if (saveBoxesBtn) saveBoxesBtn.disabled = disabled;
  if (reloadSavedBoxesBtn) {
    reloadSavedBoxesBtn.hidden = !hasJob || !state.region.hasSavedBoxes;
    reloadSavedBoxesBtn.disabled = disabled || state.region.ocrStatus === "running" || state.region.translationStatus === "running";
  }
  if (selectAllBoxesBtn) selectAllBoxesBtn.disabled = disabled;
  if (deselectAllBoxesBtn) deselectAllBoxesBtn.disabled = disabled;
  if (deleteSelectedBoxesBtn) deleteSelectedBoxesBtn.disabled = disabled;
  if (clearPageRegionsBtn) clearPageRegionsBtn.disabled = disabled;
  if (addBoxBtn) addBoxBtn.disabled = disabled;
  if (startFullPageModeBtn) startFullPageModeBtn.disabled = !hasJob || state.region.translationStatus === "running";
  if (openRegionFromFileBtn) {
    openRegionFromFileBtn.disabled = state.region.loading || state.region.ocrStatus === "running" || state.region.translationStatus === "running";
  }
  renderWorkflowSteps();
}

function renderWorkflowSteps() {
  if (!workflowStepsEl) return;
  const currentIndex = WORKFLOW_STEPS.indexOf(state.region.workflowStep);
  workflowStepsEl.querySelectorAll("[data-step]").forEach((item) => {
    const step = item.dataset.step;
    const index = WORKFLOW_STEPS.indexOf(step);
    item.classList.toggle("active", step === state.region.workflowStep);
    item.classList.toggle("done", currentIndex > index);
    item.classList.toggle("blocked", currentIndex < index);
  });
}

function countSelectedRegions() {
  return Object.values(state.region.pages).reduce((total, page) => {
    if (!page || !Array.isArray(page.regions)) return total;
    return total + page.regions.length;
  }, 0);
}

function markRegionEditsDirty() {
  state.region.hasUnsavedChanges = true;
  state.region.statusMessage = "";
  if (state.region.ocrStatus === "completed") state.region.ocrStatus = "not_started";
  if (state.region.translationStatus === "completed") state.region.translationStatus = "not_started";
  if (state.region.workflowStep !== "upload") state.region.workflowStep = "select";
  renderRegionStatus();
}

function setWorkflowBusy(step, isBusy) {
  state.region.loading = Boolean(isBusy);
  if (isBusy) state.region.workflowStep = step;
  renderRegionStatus();
}

async function renderCurrentPageCanvas() {
  const page = currentPageState();
  if (!page) {
    regionCanvasEl.innerHTML = "";
    boxListEl.innerHTML = "";
    return;
  }

  const image = await loadImage(page.imageUrl);
  const containerWidth = Math.max(420, regionCanvasEl.clientWidth || 900);
  const scale = Math.min(1, containerWidth / image.width);
  const canvasWidth = Math.round(image.width * scale);
  const canvasHeight = Math.round(image.height * scale);

  if (canvasState.stage) canvasState.stage.destroy();
  canvasState.stage = new Konva.Stage({
    container: "regionCanvas",
    width: canvasWidth,
    height: canvasHeight,
  });

  canvasState.imageLayer = new Konva.Layer();
  canvasState.boxLayer = new Konva.Layer();

  const background = new Konva.Image({
    x: 0,
    y: 0,
    width: canvasWidth,
    height: canvasHeight,
    image,
    listening: false,
  });
  canvasState.imageLayer.add(background);

  canvasState.transformer = new Konva.Transformer({
    rotateEnabled: false,
    keepRatio: false,
    enabledAnchors: ["top-left", "top-right", "bottom-left", "bottom-right"],
    boundBoxFunc: (_oldBox, newBox) => {
      if (newBox.width < 10 || newBox.height < 10) return _oldBox;
      return newBox;
    },
  });

  canvasState.boxNodes = new Map();
  for (const box of page.regions) {
    const node = createBoxNode(box, canvasWidth, canvasHeight);
    canvasState.boxLayer.add(node.group);
    canvasState.boxNodes.set(box.id, node);
  }
  canvasState.boxLayer.add(canvasState.transformer);

  canvasState.stage.add(canvasState.imageLayer);
  canvasState.stage.add(canvasState.boxLayer);
  canvasState.stage.draw();

  canvasState.stage.on("click", (event) => {
    if (event.target === canvasState.stage || event.target === background) {
      canvasState.activeBoxId = null;
      canvasState.transformer.nodes([]);
      for (const item of canvasState.boxNodes.values()) {
        item.deleteGroup?.visible(false);
      }
      canvasState.boxLayer.draw();
      renderBoxList();
    }
  });

  if (canvasState.activeBoxId && canvasState.boxNodes.has(canvasState.activeBoxId)) {
    focusBox(canvasState.activeBoxId);
  }
}

function createBoxNode(box, canvasWidth, canvasHeight) {
  box.selected = true;
  box.type = box.type || "page";
  const x = box.x0 * canvasWidth;
  const y = box.y0 * canvasHeight;
  const width = Math.max(6, (box.x1 - box.x0) * canvasWidth);
  const height = Math.max(6, (box.y1 - box.y0) * canvasHeight);

  const group = new Konva.Group();
  const rect = new Konva.Rect({
    x,
    y,
    width,
    height,
    draggable: true,
    strokeWidth: 2,
    stroke: "#0e7a52",
    dash: [],
    fill: "rgba(13, 110, 74, 0.08)",
    shadowEnabled: false,
  });

  const label = new Konva.Text({
    x,
    y: Math.max(0, y - 18),
    text: `Region #${box.reading_order}`,
    fontSize: 12,
    fill: "#0f5539",
    padding: 2,
    listening: false,
  });

  const deleteGroup = new Konva.Group({
    x: x + width - 11,
    y: y + 11,
    visible: false,
  });
  const deleteCircle = new Konva.Circle({
    radius: 10,
    fill: "#8f1d21",
    stroke: "#fffdf7",
    strokeWidth: 2,
  });
  const deleteText = new Konva.Text({
    x: -5,
    y: -8,
    text: "x",
    fontSize: 15,
    fontStyle: "bold",
    fill: "#fff",
    listening: false,
  });
  deleteGroup.add(deleteCircle);
  deleteGroup.add(deleteText);

  const syncChrome = () => {
    label.x(rect.x());
    label.y(Math.max(0, rect.y() - 18));
    deleteGroup.x(rect.x() + rect.width() - 11);
    deleteGroup.y(rect.y() + 11);
  };

  const syncFromRect = () => {
    const page = currentPageState();
    if (!page) return;
    const rectWidth = rect.width() * rect.scaleX();
    const rectHeight = rect.height() * rect.scaleY();
    const left = rect.x();
    const top = rect.y();
    const right = left + rectWidth;
    const bottom = top + rectHeight;

    box.x0 = clamp(left / canvasWidth, 0, 1);
    box.y0 = clamp(top / canvasHeight, 0, 1);
    box.x1 = clamp(right / canvasWidth, 0, 1);
    box.y1 = clamp(bottom / canvasHeight, 0, 1);

    rect.width(rectWidth);
    rect.height(rectHeight);
    rect.scaleX(1);
    rect.scaleY(1);

    syncChrome();
    markRegionEditsDirty();
    renderBoxList();
  };

  let dragInProgress = false;
  const focusFromPage = () => {
    if (dragInProgress) return;
    focusBox(box.id);
  };
  rect.on("click", focusFromPage);
  rect.on("tap", focusFromPage);
  rect.on("mouseenter", () => {
    deleteGroup.visible(true);
    canvasState.boxLayer.batchDraw();
  });
  rect.on("mouseleave", () => {
    deleteGroup.visible(canvasState.activeBoxId === box.id);
    canvasState.boxLayer.batchDraw();
  });
  deleteGroup.on("mouseenter", () => {
    deleteGroup.visible(true);
    canvasState.boxLayer.batchDraw();
  });
  deleteGroup.on("click tap", (event) => {
    event.cancelBubble = true;
    deleteBoxesById([box.id]);
  });
  rect.on("dragstart", () => {
    dragInProgress = true;
    focusBox(box.id);
  });
  rect.on("dragmove", () => {
    syncChrome();
    canvasState.boxLayer.batchDraw();
  });
  rect.on("dragend", () => {
    syncFromRect();
    window.setTimeout(() => {
      dragInProgress = false;
    }, 0);
  });
  rect.on("transformend", syncFromRect);

  group.add(rect);
  group.add(label);
  group.add(deleteGroup);
  return { group, rect, label, deleteGroup, box };
}

function focusBox(boxId) {
  const page = currentPageState();
  if (!page) return;
  const node = canvasState.boxNodes.get(boxId);
  if (!node) return;
  canvasState.activeBoxId = boxId;
  canvasState.transformer.nodes([node.rect]);
  for (const [id, item] of canvasState.boxNodes.entries()) {
    item.deleteGroup?.visible(id === boxId);
  }
  canvasState.boxLayer.draw();
  renderBoxList();
}

function toggleBoxSelection(boxId) {
  // Kept for compatibility with older event handlers. A visible box now always
  // means "OCR this region"; clicking only focuses it for editing.
  focusBox(boxId);
}

function updateBoxStyles() {
  const page = currentPageState();
  if (!page) return;
  for (const box of page.regions) {
    const node = canvasState.boxNodes.get(box.id);
    if (!node) continue;
    box.selected = true;
    node.rect.stroke("#0e7a52");
    node.rect.fill("rgba(13, 110, 74, 0.08)");
    node.rect.dash([]);
    node.label.text(`Region #${box.reading_order}`);
    node.label.fill("#0f5539");
    node.deleteGroup?.visible(box.id === canvasState.activeBoxId);
  }
  if (canvasState.boxLayer) canvasState.boxLayer.draw();
}

function renderBoxList() {
  const page = currentPageState();
  boxListEl.innerHTML = "";
  if (!page || !Array.isArray(page.regions) || !page.regions.length) {
    boxListEl.innerHTML = '<p class="empty-state">No OCR regions on this page. Add a region or use full page.</p>';
    return;
  }

  const sorted = [...page.regions].sort((a, b) => (a.reading_order - b.reading_order) || a.id.localeCompare(b.id));
  for (const box of sorted) {
    const row = document.createElement("div");
    row.className = `box-row ${box.id === canvasState.activeBoxId ? "selected" : ""}`;

    row.innerHTML = `
      <div class="box-row-head">
        <strong>${escapeHtml(box.id)}</strong>
        <button type="button" class="tiny secondary-button" data-ocr-box="${box.id}">OCR this region</button>
      </div>
      <label>Reading order
        <input data-order-box="${box.id}" type="number" min="1" step="1" value="${Number(box.reading_order) || 1}" />
      </label>
      <div class="box-geometry">
        <label>X %
          <input data-geom-box="${box.id}" data-geom-field="x" type="number" min="0" max="100" step="0.1" value="${percentValue(box.x0)}" />
        </label>
        <label>Y %
          <input data-geom-box="${box.id}" data-geom-field="y" type="number" min="0" max="100" step="0.1" value="${percentValue(box.y0)}" />
        </label>
        <label>W %
          <input data-geom-box="${box.id}" data-geom-field="w" type="number" min="0.5" max="100" step="0.1" value="${percentValue(box.x1 - box.x0)}" />
        </label>
        <label>H %
          <input data-geom-box="${box.id}" data-geom-field="h" type="number" min="0.5" max="100" step="0.1" value="${percentValue(box.y1 - box.y0)}" />
        </label>
      </div>
    `;

    row.querySelectorAll("input, select").forEach((control) => {
      control.addEventListener("click", (event) => event.stopPropagation());
    });

    row.addEventListener("click", () => focusBox(box.id));

    const orderInput = row.querySelector("input[data-order-box]");
    if (orderInput) {
      orderInput.addEventListener("change", (event) => {
        const value = Number(event.target.value);
        box.reading_order = Number.isFinite(value) && value > 0 ? Math.round(value) : 1;
        normalizeReadingOrder(page.regions);
        markRegionEditsDirty();
        renderBoxList();
        updateBoxStyles();
      });
    }

    row.querySelectorAll("input[data-geom-box]").forEach((input) => {
      input.addEventListener("change", (event) => {
        const field = event.target.dataset.geomField;
        applyBoxGeometryInput(box, field, event.target.value);
      });
    });

    const ocrBoxButton = row.querySelector("button[data-ocr-box]");
    if (ocrBoxButton) {
      ocrBoxButton.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        await runSingleRegionOcr(box);
      });
    }

    boxListEl.appendChild(row);
  }
}

function percentValue(value) {
  const percent = Number(value) * 100;
  return Number.isFinite(percent) ? percent.toFixed(1) : "0.0";
}

function applyBoxGeometryInput(box, field, rawValue) {
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return;

  const minSize = 0.005;
  const x = clamp(box.x0, 0, 1);
  const y = clamp(box.y0, 0, 1);
  const w = clamp(box.x1 - box.x0, minSize, 1);
  const h = clamp(box.y1 - box.y0, minSize, 1);
  const normalized = clamp(value / 100, field === "w" || field === "h" ? minSize : 0, 1);

  if (field === "x") {
    box.x0 = clamp(normalized, 0, 1 - w);
    box.x1 = clamp(box.x0 + w, box.x0 + minSize, 1);
  } else if (field === "y") {
    box.y0 = clamp(normalized, 0, 1 - h);
    box.y1 = clamp(box.y0 + h, box.y0 + minSize, 1);
  } else if (field === "w") {
    box.x1 = clamp(x + normalized, x + minSize, 1);
  } else if (field === "h") {
    box.y1 = clamp(y + normalized, y + minSize, 1);
  }

  updateBoxNodeGeometry(box);
  markRegionEditsDirty();
  renderBoxList();
}

function updateBoxNodeGeometry(box) {
  const node = canvasState.boxNodes.get(box.id);
  if (!node || !canvasState.stage) return;
  const canvasWidth = canvasState.stage.width();
  const canvasHeight = canvasState.stage.height();
  const x = box.x0 * canvasWidth;
  const y = box.y0 * canvasHeight;
  const width = Math.max(6, (box.x1 - box.x0) * canvasWidth);
  const height = Math.max(6, (box.y1 - box.y0) * canvasHeight);
  node.rect.position({ x, y });
  node.rect.size({ width, height });
  node.rect.scaleX(1);
  node.rect.scaleY(1);
  node.label.position({ x, y: Math.max(0, y - 18) });
  node.deleteGroup?.position({ x: x + width - 11, y: y + 11 });
  if (canvasState.activeBoxId === box.id) {
    canvasState.transformer.nodes([node.rect]);
  }
  canvasState.boxLayer.draw();
}

function normalizeReadingOrder(regions) {
  regions.sort((a, b) => (a.reading_order - b.reading_order) || (a.y0 - b.y0) || (a.x0 - b.x0));
  regions.forEach((box, idx) => {
    box.reading_order = idx + 1;
  });
}

async function detectDetailedBoxes() {
  if (!state.region.jobId) return;
  const confirmed = window.confirm(
    state.region.hasSavedBoxes
      ? "This will replace saved boxes on the current page with detailed detected boxes and may create many OCR regions. Continue?"
      : "Detailed detection can create many small boxes and substantially increase OCR time. Continue?"
  );
  if (!confirmed) return;
  await loadPageStateWithMode(state.region.currentPage, { refresh: true, detailed: true, replaceSaved: true });
  state.region.ocrStatus = "not_started";
  state.region.translationStatus = "not_started";
  state.region.workflowStep = "select";
  state.region.hasUnsavedChanges = false;
  renderRegionStatus();
}

async function useFullPageForCurrentPage() {
  if (!state.region.jobId) return;
  const page = currentPageState();
  if (!page) return;
  if (state.region.hasSavedBoxes) {
    const confirmed = window.confirm("Replace saved boxes on this page with one full-page box?");
    if (!confirmed) return;
  }
  page.regions = [buildFullPageBox(page.pageNumber)];
  canvasState.activeBoxId = page.regions[0].id;
  state.region.ocrStatus = "not_started";
  state.region.translationStatus = "not_started";
  state.region.workflowStep = "select";
  markRegionEditsDirty();
  await renderCurrentPageCanvas();
  focusBox(page.regions[0].id);
  renderBoxList();
  renderRegionStatus();
}

async function useFullPageForAllPages() {
  if (!state.region.jobId || !state.region.pageCount) return;
  if (state.region.hasSavedBoxes) {
    const confirmed = window.confirm("Replace saved boxes on all pages with one full-page box per page?");
    if (!confirmed) return;
  }
  const current = state.region.currentPage;
  state.region.loading = true;
  renderRegionStatus();
  try {
    for (let pageNumber = 1; pageNumber <= state.region.pageCount; pageNumber += 1) {
      const payload = await fetchPageBoxes(pageNumber, true, false, true);
      const key = String(pageNumber);
      const existingImageUrl = state.region.pages[key]?.imageUrl || null;
      state.region.pages[key] = pageStateFromBoxesPayload(payload, existingImageUrl);
    }
    state.region.currentPage = current;
    state.region.ocrStatus = "not_started";
    state.region.translationStatus = "not_started";
    state.region.workflowStep = "select";
    state.region.hasUnsavedChanges = false;
    await renderCurrentPageCanvas();
    renderBoxList();
  } finally {
    state.region.loading = false;
    renderRegionStatus();
  }
}

async function loadPageStateWithMode(pageNumber, { refresh, detailed, replaceSaved = false }) {
  const key = String(pageNumber);
  const jobId = state.region.jobId;
  if (!jobId) return;

  state.region.loading = true;
  renderRegionStatus();
  try {
    const boxesPayload = await fetchPageBoxes(pageNumber, refresh, detailed, replaceSaved);
    const imageUrl = state.region.pages[key]?.imageUrl || await fetchPageImageUrl(jobId, pageNumber);
    state.region.pages[key] = pageStateFromBoxesPayload(boxesPayload, imageUrl);
    state.region.currentPage = pageNumber;
    state.region.regionsDetected = true;
    await renderCurrentPageCanvas();
    renderBoxList();
  } finally {
    state.region.loading = false;
    renderRegionStatus();
  }
}

async function ensureAllPageRegionsDetected() {
  if (!state.region.jobId || !state.region.pageCount) return;
  for (let pageNumber = 1; pageNumber <= state.region.pageCount; pageNumber += 1) {
    const key = String(pageNumber);
    if (state.region.pages[key]) continue;
    const payload = await fetchPageBoxes(pageNumber, false, false);
    state.region.pages[key] = pageStateFromBoxesPayload(payload, null);
  }
  state.region.regionsDetected = true;
  renderRegionStatus();
}

async function fetchPageBoxes(pageNumber, refreshDetect, detailed, replaceSaved = false) {
  const params = new URLSearchParams({
    refresh: refreshDetect ? "true" : "false",
    detailed: detailed ? "true" : "false",
    replace_saved: replaceSaved ? "true" : "false",
  });
  const res = await fetch(`/api/jobs/${state.region.jobId}/pages/${pageNumber}/boxes?${params.toString()}`);
  const payload = await parseJsonResponse(res);
  if (!res.ok) throw new Error(payload.detail || `Unable to load boxes for page ${pageNumber}`);
  return payload;
}

function pageStateFromBoxesPayload(boxesPayload, imageUrl) {
  const regions = Array.isArray(boxesPayload.regions) ? boxesPayload.regions : [];
  for (const box of regions) {
    box.selected = true;
    box.type = box.type || "page";
  }
  return {
    pageNumber: boxesPayload.page_number,
    pageWidth: boxesPayload.page_width,
    pageHeight: boxesPayload.page_height,
    imageWidth: boxesPayload.image_width,
    imageHeight: boxesPayload.image_height,
    detector: boxesPayload.detector,
    imageUrl,
    regions,
  };
}

async function saveAllLoadedPageBoxes() {
  const pages = Object.values(state.region.pages).filter(Boolean);
  for (const page of pages) {
    await savePageBoxes(page, { renderAfterSave: false });
  }
  state.region.hasUnsavedChanges = false;
  state.region.hasSavedBoxes = true;
  renderRegionStatus();
}

async function saveCurrentPageBoxes() {
  const page = currentPageState();
  if (!page || !state.region.jobId) return;
  try {
    await savePageBoxes(page, { renderAfterSave: true });
    state.region.hasUnsavedChanges = false;
    state.region.hasSavedBoxes = true;
    state.region.statusMessage = "Saved OCR boxes.";
    await pollJobs();
    renderRegionStatus();
  } catch (error) {
    window.alert(error.message || "Failed to save boxes");
    throw error;
  }
}

async function saveLoadedBoxesFromButton() {
  if (!state.region.jobId) return;
  try {
    await saveAllLoadedPageBoxes();
    state.region.statusMessage = "Saved OCR boxes.";
    await pollJobs();
    renderRegionStatus();
  } catch (error) {
    window.alert(error.message || "Failed to save boxes");
    throw error;
  }
}

async function reloadSavedBoxes() {
  if (!state.region.jobId || !state.region.hasSavedBoxes) return;
  if (state.region.hasUnsavedChanges) {
    const confirmed = window.confirm("Discard unsaved box edits and reload the saved boxes?");
    if (!confirmed) return;
  }
  state.region.pages = {};
  state.region.hasUnsavedChanges = false;
  state.region.statusMessage = "Loaded saved OCR regions from previous session.";
  await loadSavedBoxesSummary();
  await loadPageState(state.region.currentPage || 1, false);
  renderRegionStatus();
}

async function savePageBoxes(page, options = {}) {
  normalizeReadingOrder(page.regions);
  const regions = page.regions.map((box) => ({
    ...box,
    selected: true,
    type: box.type || "page",
  }));
  const payload = {
    pdf_file_id: state.region.jobId,
    page_number: page.pageNumber,
    page_width: page.pageWidth,
    page_height: page.pageHeight,
    image_width: page.imageWidth,
    image_height: page.imageHeight,
    coordinate_space: "normalized",
    detector: page.detector || "manual",
    regions,
  };

  const res = await fetch(`/api/jobs/${state.region.jobId}/pages/${page.pageNumber}/boxes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await parseJsonResponse(res);
  if (!res.ok) {
    throw new Error(data.detail || "Failed to save boxes");
  }

  page.regions = Array.isArray(data.regions) ? data.regions : page.regions;
  page.detector = data.detector || page.detector;
  if (options.renderAfterSave) {
    await renderCurrentPageCanvas();
    renderBoxList();
  }
  renderRegionStatus();
}

function setAllBoxesSelected(value) {
  const page = currentPageState();
  if (!page) return;
  for (const box of page.regions) box.selected = true;
  updateBoxStyles();
  renderBoxList();
}

function deleteSelectedBoxes() {
  const page = currentPageState();
  if (!page) return;
  const ids = canvasState.activeBoxId ? [canvasState.activeBoxId] : [];
  if (!ids.length) return;
  deleteBoxesById(ids);
}

function clearCurrentPageRegions() {
  const page = currentPageState();
  if (!page) return;
  page.regions = [];
  canvasState.activeBoxId = null;
  markRegionEditsDirty();
  renderCurrentPageCanvas();
  renderBoxList();
}

function deleteBoxesById(ids) {
  const page = currentPageState();
  if (!page) return;
  const set = new Set(ids);
  page.regions = page.regions.filter((box) => !set.has(box.id));
  normalizeReadingOrder(page.regions);
  if (set.has(canvasState.activeBoxId)) {
    canvasState.activeBoxId = null;
  }
  markRegionEditsDirty();
  renderCurrentPageCanvas();
  renderBoxList();
}

async function addManualBox() {
  const page = currentPageState();
  if (!page) return;

  const active = page.regions.find((box) => box.id === canvasState.activeBoxId);
  const newBox = buildManualBox(active);
  page.regions.push(newBox);
  normalizeReadingOrder(page.regions);
  markRegionEditsDirty();
  await renderCurrentPageCanvas();
  focusBox(newBox.id);
  renderBoxList();
}

async function duplicateBox(boxId) {
  const page = currentPageState();
  if (!page) return;
  const source = page.regions.find((box) => box.id === boxId);
  if (!source) return;

  const newBox = buildManualBox(source);
  newBox.type = source.type;
  newBox.selected = true;
  page.regions.push(newBox);
  normalizeReadingOrder(page.regions);
  markRegionEditsDirty();
  await renderCurrentPageCanvas();
  focusBox(newBox.id);
  renderBoxList();
}

function buildManualBox(sourceBox) {
  const nextOrder = (currentPageState()?.regions.length || 0) + 1;
  const base = sourceBox || {
    x0: 0.2,
    y0: 0.2,
    x1: 0.6,
    y1: 0.35,
    type: "text",
  };
  const width = clamp(base.x1 - base.x0, 0.08, 0.8);
  const height = clamp(base.y1 - base.y0, 0.05, 0.6);
  const offset = sourceBox ? 0.035 : 0;
  const x0 = clamp(base.x0 + offset, 0, 1 - width);
  const y0 = clamp(base.y0 + offset, 0, 1 - height);

  return {
    id: `manual-${state.region.currentPage}-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
    page_number: state.region.currentPage,
    x0,
    y0,
    x1: clamp(x0 + width, x0 + 0.005, 1),
    y1: clamp(y0 + height, y0 + 0.005, 1),
    coordinate_space: "normalized",
    type: base.type || "text",
    selected: true,
    reading_order: nextOrder,
    source: "manual",
  };
}

function buildFullPageBox(pageNumber) {
  return {
    id: `full-page-${pageNumber}-${Date.now()}`,
    page_number: pageNumber,
    x0: 0,
    y0: 0,
    x1: 1,
    y1: 1,
    coordinate_space: "normalized",
    type: "page",
    selected: true,
    reading_order: 1,
    source: "manual",
  };
}

async function runSelectedOcr() {
  if (!state.region.jobId) return;
  if (state.region.ocrStatus === "running") return;
  runOcrSelectedBtn.disabled = true;
  state.region.ocrStatus = "running";
  state.region.workflowStep = "ocr";
  renderRegionStatus();
  try {
    await ensureAllPageRegionsDetected();
    if (countSelectedRegions() === 0) {
      throw new Error("Select at least one OCR region before running OCR.");
    }
    await saveAllLoadedPageBoxes();
    const res = await fetch(`/api/jobs/${state.region.jobId}/ocr/selected`, { method: "POST" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Selected OCR failed");
    state.region.ocrStatus = "completed";
    state.region.workflowStep = "translate";
    const pagesWithText = Number(data.pages_with_text_count || 0);
    const selectedPages = Number(data.selected_page_count || 0);
    const emptyPages = Array.isArray(data.pages_without_text) ? data.pages_without_text : [];
    const emptySuffix = emptyPages.length
      ? `\n\nWarning: no OCR text was produced for page(s): ${emptyPages.join(", ")}. Translation will only include pages with OCR text.`
      : "";
    window.alert(
      `OCR completed for ${data.total_region_count || data.count || 0} region(s). Text was found on ${pagesWithText}/${selectedPages} selected page(s).${emptySuffix}`
    );
  } catch (error) {
    state.region.ocrStatus = "failed";
    state.region.workflowStep = "ocr";
    window.alert(error.message || "Selected OCR failed");
  } finally {
    runOcrSelectedBtn.disabled = false;
    renderRegionStatus();
  }
}

async function runSingleRegionOcr(box) {
  if (!state.region.jobId || !box) return;
  if (state.region.ocrStatus === "running") return;
  state.region.ocrStatus = "running";
  state.region.workflowStep = "ocr";
  renderRegionStatus();
  try {
    await saveCurrentPageBoxes();
    const params = new URLSearchParams({
      page_number: String(state.region.currentPage),
      box_id: String(box.id),
    });
    const res = await fetch(`/api/jobs/${state.region.jobId}/ocr/selected?${params.toString()}`, { method: "POST" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Single-region OCR failed");
    state.region.ocrStatus = "completed";
    state.region.workflowStep = "translate";
    const chars = Array.isArray(data.pages_with_text) && data.pages_with_text.length ? "Text found." : "No text found.";
    window.alert(`OCR completed for region "${box.id}". ${chars}`);
  } catch (error) {
    state.region.ocrStatus = "failed";
    state.region.workflowStep = "ocr";
    window.alert(error.message || "Single-region OCR failed");
  } finally {
    renderRegionStatus();
  }
}

async function startRegionJob(mode) {
  if (!state.region.jobId) return;
  if (mode === "selected_regions" && state.region.ocrStatus !== "completed") {
    window.alert("Run OCR selected regions before translating OCR output.");
    return;
  }
  if (state.region.translationStatus === "running") return;

  if (mode === "selected_regions") {
    await saveAllLoadedPageBoxes();
  }

  if (mode === "full_page") state.region.ocrStatus = "running";
  state.region.translationStatus = "running";
  state.region.workflowStep = mode === "full_page" ? "ocr" : "translate";
  renderRegionStatus();
  const payload = {
    ...buildTranslationPayload(),
    ocr_input_mode: mode,
    ocr_full_page_fallback: mode === "full_page",
  };

  try {
    const res = await fetch(`/api/jobs/${state.region.jobId}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Unable to start job");
    await pollJobs();
  } catch (error) {
    if (mode === "full_page") state.region.ocrStatus = "failed";
    state.region.translationStatus = "failed";
    state.region.workflowStep = mode === "full_page" ? "ocr" : "translate";
    window.alert(error.message || "Unable to start job");
  } finally {
    renderRegionStatus();
  }
}

function getInputValue(id, fallback) {
  const input = document.getElementById(id);
  if (!input) return fallback;
  const value = String(input.value ?? "").trim();
  return value || fallback;
}

function numberInput(id, fallback) {
  const value = Number(getInputValue(id, String(fallback)));
  return Number.isFinite(value) ? value : fallback;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Unable to decode page image"));
    img.src = url;
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

setInterval(pollJobs, 2000);
pollJobs();
renderStagedQueue();
renderRegionStatus();
