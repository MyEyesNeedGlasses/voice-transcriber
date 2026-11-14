const recordBtn = document.getElementById("record-btn");
const recordLabel = document.getElementById("record-label");
const timerEl = document.getElementById("timer");
const preview = document.getElementById("preview");
const transcribeBtn = document.getElementById("transcribe-btn");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result-card");
const resultMeta = document.getElementById("result-meta");
const resultText = document.getElementById("result-text");
const resultTurns = document.getElementById("result-turns");
const speakersField = document.getElementById("speakers-field");
const speakersInput = document.getElementById("speakers-input");
const copyBtn = document.getElementById("copy-btn");
const historyEl = document.getElementById("history");
const historyEmpty = document.getElementById("history-empty");

let mediaRecorder = null;
let chunks = [];
let audioBlob = null;
let audioName = null;
let timerId = null;
let startedAt = 0;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function formatSeconds(total) {
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(Math.floor(total % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

function extensionFor(mimeType) {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  return "webm";
}

function setAudio(blob, name) {
  audioBlob = blob;
  audioName = name;
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(blob);
  preview.hidden = false;
  transcribeBtn.disabled = false;
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not support audio recording.", true);
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus(`Microphone access denied: ${err.message}`, true);
    return;
  }

  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const type = mediaRecorder.mimeType || "audio/webm";
    const blob = new Blob(chunks, { type });
    setAudio(blob, `recording-${Date.now()}.${extensionFor(type)}`);
    setStatus("Recording ready. Click Transcribe.");
  };

  mediaRecorder.start();
  startedAt = Date.now();
  timerEl.textContent = "00:00";
  timerId = setInterval(() => {
    timerEl.textContent = formatSeconds((Date.now() - startedAt) / 1000);
  }, 250);

  recordBtn.classList.add("recording");
  recordLabel.textContent = "Stop recording";
  transcribeBtn.disabled = true;
  setStatus("Recording…");
}

function stopRecording() {
  clearInterval(timerId);
  mediaRecorder.stop();
  recordBtn.classList.remove("recording");
  recordLabel.textContent = "Start recording";
}

recordBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    startRecording();
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  setAudio(file, file.name);
  setStatus(`Loaded ${file.name}. Click Transcribe.`);
  fileInput.value = "";
});

transcribeBtn.addEventListener("click", async () => {
  if (!audioBlob) return;

  const form = new FormData();
  form.append("file", audioBlob, audioName);
  if (!speakersField.hidden && speakersInput.value) {
    form.append("num_speakers", speakersInput.value);
  }

  transcribeBtn.disabled = true;
  recordBtn.disabled = true;
  setStatus("Transcribing… this can take a while on CPU.");

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

    showResult(data);
    setStatus("Done.");
    loadHistory();
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    transcribeBtn.disabled = false;
    recordBtn.disabled = false;
  }
});

function describe(item) {
  const parts = [new Date(item.created_at).toLocaleString()];
  if (item.duration != null) parts.push(formatSeconds(item.duration));
  if (item.language) parts.push(item.language.toUpperCase());
  const speakers = new Set(item.segments.map((seg) => seg.speaker).filter(Boolean));
  if (speakers.size) parts.push(`${speakers.size} speaker${speakers.size === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

function hasSpeakers(item) {
  return item.segments.some((seg) => seg.speaker);
}

// "SPEAKER_00" -> "Speaker 1"
function speakerName(label) {
  const match = /^SPEAKER_(\d+)$/.exec(label || "");
  if (match) return `Speaker ${Number(match[1]) + 1}`;
  return label || "Unknown";
}

function speakerColor(label) {
  const match = /(\d+)$/.exec(label || "");
  return match ? Number(match[1]) % 6 : 0;
}

function transcriptText(item) {
  if (!hasSpeakers(item)) return item.text;
  return item.segments.map((seg) => `${speakerName(seg.speaker)}: ${seg.text}`).join("\n");
}

let currentItem = null;

function showResult(item) {
  currentItem = item;
  resultCard.hidden = false;
  resultMeta.textContent = describe(item);

  const diarized = hasSpeakers(item);
  resultText.hidden = diarized;
  resultTurns.hidden = !diarized;
  resultTurns.replaceChildren();

  if (!diarized) {
    resultText.textContent = item.text || "(no speech detected)";
    return;
  }

  for (const seg of item.segments) {
    const who = document.createElement("span");
    who.className = `speaker speaker-${speakerColor(seg.speaker)}`;
    who.textContent = speakerName(seg.speaker);

    const when = document.createElement("span");
    when.className = "muted small";
    when.textContent = formatSeconds(seg.start);

    const head = document.createElement("div");
    head.className = "turn-head";
    head.append(who, when);

    const text = document.createElement("p");
    text.textContent = seg.text;

    const row = document.createElement("div");
    row.className = "turn";
    row.append(head, text);
    resultTurns.append(row);
  }
}

copyBtn.addEventListener("click", async () => {
  if (!currentItem) return;
  try {
    await navigator.clipboard.writeText(transcriptText(currentItem));
    copyBtn.textContent = "Copied";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
  } catch {
    setStatus("Could not copy to clipboard.", true);
  }
});

async function loadHistory() {
  try {
    const res = await fetch("/api/transcriptions");
    const items = await res.json();
    renderHistory(items);
  } catch (err) {
    setStatus(`Could not load history: ${err.message}`, true);
  }
}

function renderHistory(items) {
  historyEl.replaceChildren();
  historyEmpty.hidden = items.length > 0;

  for (const item of items) {
    const li = document.createElement("li");

    const body = document.createElement("div");
    const meta = document.createElement("p");
    meta.className = "muted small";
    meta.textContent = describe(item);
    const text = document.createElement("p");
    text.className = "text";
    text.textContent = item.text || "(no speech detected)";
    body.append(meta, text);
    body.style.cursor = "pointer";
    body.addEventListener("click", () => showResult(item));

    const del = document.createElement("button");
    del.className = "btn btn-ghost btn-small";
    del.type = "button";
    del.textContent = "Delete";
    del.addEventListener("click", async () => {
      const res = await fetch(`/api/transcriptions/${item.id}`, { method: "DELETE" });
      if (res.ok) loadHistory();
      else setStatus("Delete failed.", true);
    });

    li.append(body, del);
    historyEl.append(li);
  }
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const health = await res.json();
    speakersField.hidden = !health.diarization;
  } catch {
    // Health is only used to show the speakers field
  }
}

loadHealth();
loadHistory();
