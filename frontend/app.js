const DEFAULT_LLM_TEMPERATURE = 0.4;
const DEFAULT_LLM_TOP_P = 0.7;
const DEFAULT_LLM_TOP_K = 10;
const DEFAULT_LLM_MIN_P = 0.0;
const DEFAULT_LLM_PRESENCE_PENALTY = 1.5;
const DEFAULT_LLM_REPETITION_PENALTY = 1.0;
const WORKFLOW_STEPS = ["upload", "select", "ocr", "translate", "export"];

const state = { jobs: [] };

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const pickBtn = document.getElementById("pickBtn");
const queueEl = document.getElementById("queue");
const workflowStatusEl = document.getElementById("workflowStatus");
const workflowStepsEl = document.getElementById("workflowSteps");
const cleanTerminalBtn = document.getElementById("cleanTerminalBtn");
const clearResultsBtn = document.getElementById("clearResultsBtn");
const stopAllBtn = document.getElementById("stopAllBtn");

pickBtn?.addEventListener("click", () => fileInput.click());
fileInput?.addEventListener("change", () => {
  handleUploadedFiles(fileInput.files);
  fileInput.value = "";
});
clearResultsBtn?.addEventListener("click", clearResults);
cleanTerminalBtn?.addEventListener("click", cleanTerminalJobs);
stopAllBtn?.addEventListener("click", stopAllProcesses);

dropzone?.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone?.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  handleUploadedFiles(event.dataTransfer.files);
});

async function handleUploadedFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const pdfFiles = Array.from(fileList).filter((file) => {
    return file.type === "application/pdf" || String(file.name || "").toLowerCase().endsWith(".pdf");
  });
  if (!pdfFiles.length) return;

  try {
    await submitUploadBatch(pdfFiles);
    await pollJobs();
  } catch (error) {
    window.alert(error.message || "Unable to submit PDF for automatic extraction");
  }
}

async function submitUploadBatch(files) {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file, file.name);
  }
  appendTranslationFormFields(form);

  const res = await fetch("/api/jobs", { method: "POST", body: form });
  const data = await parseJsonResponse(res);
  if (!res.ok) throw new Error(data.detail || "Upload submission failed");

  for (const job of data.jobs || []) {
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
  renderWorkflowStatus();
}

function appendTranslationFormFields(form) {
  form.append("chunk_size", getInputValue("chunkSize", "1800"));
  form.append("temperature", getInputValue("temp", String(DEFAULT_LLM_TEMPERATURE)));
  form.append("max_tokens", getInputValue("maxTokens", "2048"));
  form.append("model", getInputValue("model", "mlx-community/Qwen3.5-9B-MLX-4bit"));
  form.append("top_p", getInputValue("topP", String(DEFAULT_LLM_TOP_P)));
  form.append("top_k", String(DEFAULT_LLM_TOP_K));
  form.append("min_p", String(DEFAULT_LLM_MIN_P));
  form.append("presence_penalty", String(DEFAULT_LLM_PRESENCE_PENALTY));
  form.append("repetition_penalty", String(DEFAULT_LLM_REPETITION_PENALTY));
  form.append("output_mode", "readable");
  form.append("extraction_mode", getInputValue("extractionMode", "auto"));
  form.append("use_local_vlm_repair", checkboxValue("useLocalVlmRepair"));
  form.append("keep_debug_artifacts", checkboxValue("keepDebugArtifacts"));
}

async function pollJobs() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    state.jobs = await res.json();
    renderQueue();
    renderWorkflowStatus();
  } catch (_error) {
    // Keep the current queue visible during transient backend restarts.
  }
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
      ${translationWarningLine(job)}
      ${originalLayoutWarningLine(job)}
      ${job.error ? `<div class="error">${escapeHtml(job.error)}</div>` : ""}
      <div class="downloads">
        ${cancelQueuedButton(job)}
        ${sourceMarkdownDownloadLink(job)}
        ${sourcePdfDownloadLink(job, "readable", "OCR PDF")}
        ${pdfDownloadLink(job, "readable", "Readable PDF")}
        ${pdfDownloadLink(job, "faithful", "Faithful PDF")}
        ${pdfDownloadLink(job, "original-layout", "Original layout PDF")}
        ${downloadLink(job, "reconstruction_report", "Reconstruction Report")}
        ${downloadLink(job, "markdown", "Markdown")}
        ${downloadLink(job, "json", "JSON")}
        ${downloadLink(job, "extraction_result", "Extraction JSON")}
        ${downloadLink(job, "marker_detection", "Detection JSON")}
        ${downloadLink(job, "profile_summary", "Timing Summary")}
        ${downloadLink(job, "profile_json", "Timing JSON")}
        ${downloadLink(job, "profile_csv", "Timing CSV")}
      </div>
    `;
    item.querySelector("button[data-cancel-queued]")?.addEventListener("click", (event) => {
      event.preventDefault();
      cancelQueuedJob(job);
    });
    queueEl.appendChild(item);
  }
}

function renderWorkflowStatus() {
  if (!workflowStatusEl || !workflowStepsEl) return;
  const current = state.jobs.find((job) => !["complete", "cancelled", "failed"].includes(job.stage)) || state.jobs[0];
  const activeStep = workflowStepForStage(current?.stage);
  for (const item of workflowStepsEl.querySelectorAll("[data-step]")) {
    const step = item.dataset.step;
    const stepIndex = WORKFLOW_STEPS.indexOf(step);
    const activeIndex = WORKFLOW_STEPS.indexOf(activeStep);
    item.classList.toggle("active", step === activeStep);
    item.classList.toggle("done", activeIndex >= 0 && stepIndex < activeIndex);
  }
  workflowStatusEl.textContent = current ? `${current.filename} | ${current.message || stageLabel(current.stage)}` : "No active document.";
}

function workflowStepForStage(stage) {
  const map = {
    upload: "upload",
    extraction: "select",
    ocr_layout_parsing: "ocr",
    structure_generation: "ocr",
    translation: "translate",
    pdf_generation: "export",
    complete: "export",
    cancelled: "upload",
    failed: "upload",
  };
  return map[stage] || "upload";
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

function progressBarLabel(job, progressWidth) {
  const counter = String(job.message || "").match(/\b\d+\s*\/\s*\d+\b/);
  if (job.stage === "ocr_layout_parsing" && counter) {
    return `OCR ${counter[0].replace(/\s+/g, "")}`;
  }
  return `${progressWidth}%`;
}

function downloadLink(job, type, label) {
  if (!job.artifacts || !job.artifacts[type]) return "";
  return `<a href="/api/jobs/${job.job_id}/artifacts/${type}" target="_blank"><button>${label}</button></a>`;
}

function pdfDownloadLink(job, mode, label) {
  const canGenerate = job.stage === "complete";
  return canGenerate ? `<a href="/api/jobs/${job.job_id}/pdf/${mode}" target="_blank"><button>${label}</button></a>` : "";
}

function sourceMarkdownDownloadLink(job) {
  if (!job.artifacts?.source_markdown) return "";
  return `<a href="/api/jobs/${job.job_id}/artifacts/source_markdown" target="_blank"><button>OCR Markdown</button></a>`;
}

function sourcePdfDownloadLink(job, mode, label) {
  if (!job.artifacts?.source_markdown) return "";
  return `<a href="/api/jobs/${job.job_id}/ocr-pdf/${mode}" target="_blank"><button>${label}</button></a>`;
}

function translationInfoLine(job) {
  if (!job.translation) return "";
  const model = String(job.translation.model || "").trim();
  if (!model && !job.translation.pdf_classification && !job.translation.marker_mode) return "";
  const temp = numericLabel(job.translation.temperature, 2);
  const topP = numericLabel(job.translation.top_p, 2);
  const topK = numericLabel(job.translation.top_k, 0);
  const minP = numericLabel(job.translation.min_p, 2);
  const presence = numericLabel(job.translation.presence_penalty, 2);
  const repeat = numericLabel(job.translation.repetition_penalty, 2);
  const classification = job.translation.pdf_classification ? ` | PDF: ${job.translation.pdf_classification}` : "";
  const markerMode = job.translation.marker_mode ? ` | Marker: ${job.translation.marker_mode}` : "";
  const fallback = job.translation.fallback_engine ? ` | Fallback: ${job.translation.fallback_engine}` : "";
  const ocr = typeof job.translation.ocr_used === "boolean" ? ` | OCR: ${job.translation.ocr_used ? "yes" : "no"}` : "";
  const repair = job.translation.local_vlm_repair_used ? " | VLM repair: yes" : "";
  const modelParams = [
    `Translation model: ${model.split("/").pop() || model}`,
    `temp: ${temp}`,
    `top-p: ${topP}`,
    `top-k: ${topK}`,
    `min-p: ${minP}`,
    `presence: ${presence}`,
    `repeat: ${repeat}`,
  ].join(" | ");
  return `<div class="meta-line">${escapeHtml(`${modelParams}${classification}${markerMode}${fallback}${ocr}${repair}`)}</div>`;
}

function translationWarningLine(job) {
  const warnings = Array.isArray(job.translation?.warnings) ? job.translation.warnings.filter(Boolean) : [];
  return warnings.length ? `<div class="warning-line">${escapeHtml(warnings.slice(0, 3).join(" | "))}</div>` : "";
}

function originalLayoutWarningLine(job) {
  if (job.stage !== "complete") return "";
  const result = job.translation?.original_layout_reconstruction;
  if (result?.status === "complete") return "";
  const detail = result?.status === "partial"
    ? ` Partial result: ${result.pages_using_fallback_behavior || 0} page(s) retained or skipped; see the report.`
    : " Scanned, hidden-OCR, rotated, or unreliable pages may remain unchanged.";
  return `<div class="original-layout-warning">Original layout PDF is conservative.${escapeHtml(detail)} Use Readable PDF as the safe fallback.</div>`;
}

function numericLabel(value, digits) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "n/a";
}

function cancelQueuedButton(job) {
  return job.stage === "upload"
    ? `<button type="button" data-cancel-queued="${job.job_id}">Cancel Queued Job</button>`
    : "";
}

async function cancelQueuedJob(job) {
  if (!window.confirm("Cancel this queued job?")) return;
  try {
    const res = await fetch(`/api/jobs/${job.job_id}/cancel`, { method: "POST" });
    const data = await parseJsonResponse(res);
    if (!res.ok) throw new Error(data.detail || "Unable to cancel queued job");
    forcePageRefresh();
  } catch (error) {
    window.alert(error.message || "Unable to cancel queued job");
  }
}

async function clearResults() {
  if (!window.confirm("Delete all uploaded PDFs and generated results?")) return;
  clearResultsBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs", { method: "DELETE" });
    if (!res.ok) throw new Error("Cleanup failed");
    state.jobs = [];
    renderQueue();
    renderWorkflowStatus();
  } catch (error) {
    window.alert(error.message || "Cleanup failed");
  } finally {
    clearResultsBtn.disabled = false;
  }
}

async function cleanTerminalJobs() {
  if (!window.confirm("Remove cancelled and failed jobs from the list?")) return;
  cleanTerminalBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs/cleanup-terminal", { method: "DELETE" });
    if (!res.ok) throw new Error("Cleanup failed");
    forcePageRefresh();
  } catch (error) {
    window.alert(error.message || "Cleanup failed");
  } finally {
    cleanTerminalBtn.disabled = false;
  }
}

async function stopAllProcesses() {
  if (!window.confirm("Stop the active job and cancel all queued jobs?")) return;
  stopAllBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs/stop-all", { method: "POST" });
    if (!res.ok) throw new Error("Stop-all failed");
    forcePageRefresh();
  } catch (error) {
    window.alert(error.message || "Stop-all failed");
  } finally {
    stopAllBtn.disabled = false;
  }
}

function forcePageRefresh() {
  const url = new URL(window.location.href);
  url.searchParams.set("_refresh", Date.now().toString());
  window.location.replace(url.toString());
}

function checkboxValue(id) {
  const item = document.getElementById(id);
  return item && item.checked ? "true" : "false";
}

function getInputValue(id, fallback) {
  const input = document.getElementById(id);
  const value = String(input?.value ?? "").trim();
  return value || fallback;
}

async function parseJsonResponse(res) {
  try {
    return await res.json();
  } catch (_error) {
    return {};
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

setInterval(pollJobs, 2000);
pollJobs();
