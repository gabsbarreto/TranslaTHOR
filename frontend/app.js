const state = {
  jobs: [],
  staged: [],
  stagingCounter: 0,
  submittingStaged: false,
};

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

pickBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  stageFiles(fileInput.files);
  fileInput.value = "";
});
clearResultsBtn.addEventListener("click", clearResults);
stopAllBtn.addEventListener("click", stopAllProcesses);
parseTranslateBtn.addEventListener("click", submitStagedJobs);
clearStagedBtn.addEventListener("click", clearStagedJobs);
cleanTerminalBtn.addEventListener("click", cleanTerminalJobs);

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  stageFiles(e.dataTransfer.files);
});

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
  parseTranslateBtn.disabled = state.submittingStaged || state.staged.length === 0;
  clearStagedBtn.disabled = state.submittingStaged || state.staged.length === 0;
  parseTranslateBtn.textContent = state.staged.length
    ? `Parse and Translate (${state.staged.length})`
    : "Parse and Translate";

  stagedQueueEl.innerHTML = "";
  if (!state.staged.length) {
    stagedQueueEl.innerHTML = '<p class="empty-state">No staged documents. Add PDFs or click "Reuse OCR File" from a completed job.</p>';
    return;
  }

  for (const item of state.staged) {
    const row = document.createElement("div");
    row.className = "staged-item";
    const subtype =
      item.type === "reuse"
        ? `Reuse OCR from ${item.source_job_id.slice(0, 8)}`
        : "New PDF upload";
    row.innerHTML = `
      <div class="staged-item-main">
        <strong>${escapeHtml(item.filename)}</strong>
        <small>${escapeHtml(subtype)}</small>
      </div>
      <button type="button" data-remove-staged="${item.local_id}" class="secondary-button">Remove</button>
    `;
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
  renderQueue();
}

async function clearResults() {
  const confirmed = window.confirm("Delete all uploaded PDFs and generated results?");
  if (!confirmed) return;

  clearResultsBtn.disabled = true;
  try {
    const res = await fetch("/api/jobs", { method: "DELETE" });
    if (!res.ok) throw new Error("Cleanup failed");
    state.jobs = [];
    renderQueue();
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
  queueEl.innerHTML = "";
  if (!state.jobs.length) {
    queueEl.innerHTML = '<p class="empty-state">No documents in the queue.</p>';
    return;
  }
  for (const job of state.jobs) {
    const item = document.createElement("div");
    item.className = "job-item";

    const progressWidth = Math.max(2, Math.round((job.progress || 0) * 100));

    item.innerHTML = `
      <div class="job-head">
        <strong>${escapeHtml(job.filename)}</strong>
        <code>${job.job_id.slice(0, 8)}</code>
      </div>
      <div class="progress"><span style="width:${progressWidth}%"></span></div>
      <div class="stage">${stageLabel(job.stage)} - ${escapeHtml(job.message || "")}</div>
      ${translationInfoLine(job)}
      ${job.error ? `<div class="error">${escapeHtml(job.error)}</div>` : ""}
      <div class="downloads">
        ${cancelQueuedButton(job)}
        ${retranslateButton(job)}
        ${pdfDownloadLink(job, "readable", "Readable PDF")}
        ${pdfDownloadLink(job, "faithful", "Faithful PDF")}
        ${downloadLink(job, "markdown", "Markdown")}
        ${downloadLink(job, "json", "JSON")}
        ${downloadLink(job, "profile_summary", "Timing Summary")}
        ${downloadLink(job, "profile_json", "Timing JSON")}
        ${downloadLink(job, "profile_csv", "Timing CSV")}
      </div>
    `;

    const reuseBtn = item.querySelector("button[data-reuse-ocr]");
    if (reuseBtn) {
      reuseBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        stageReuseJob(job);
      });
    }

    const cancelBtn = item.querySelector("button[data-cancel-queued]");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        cancelQueuedJob(job);
      });
    }

    queueEl.appendChild(item);
  }
}

function downloadLink(job, type, label) {
  if (!job.artifacts || !job.artifacts[type]) return "";
  return `<a href="/api/jobs/${job.job_id}/artifacts/${type}" target="_blank"><button>${label}</button></a>`;
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
  const canRetranslate = job.stage === "complete";
  if (!canRetranslate) return "";
  return `<button type="button" data-reuse-ocr="${job.job_id}">Reuse OCR File</button>`;
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

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getInputValue(id, fallback) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  return typeof el.value === "string" ? el.value : fallback;
}

function numberInput(id, fallback) {
  const value = Number(getInputValue(id, String(fallback)));
  return Number.isFinite(value) ? value : fallback;
}

setInterval(pollJobs, 2000);
renderStagedQueue();
pollJobs();
