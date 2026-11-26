/* Personalization layer: patient profiles, multilingual voice cues, progress chart,
   physio-approved adaptive targets, and Gemini-powered session reports / Q&A. */

const coachEls = {
  patientSelect: document.getElementById("patient-select"),
  btnNewPatient: document.getElementById("btn-new-patient"),
  guestGroup: document.getElementById("guest-label-group"),
  languageSelect: document.getElementById("language-select"),
  voiceToggle: document.getElementById("voice-toggle"),
  sideSelect: document.getElementById("side-select"),
  exerciseSelect: document.getElementById("exercise-select"),
  targetAngle: document.getElementById("target-angle"),
  startAngle: document.getElementById("start-angle"),
  targetSource: document.getElementById("target-source"),
  lastRep: document.getElementById("last-rep"),
  progressTitle: document.getElementById("progress-title"),
  suggestion: document.getElementById("target-suggestion"),
  aiStatus: document.getElementById("ai-status"),
  askInput: document.getElementById("ask-input"),
  btnAsk: document.getElementById("btn-ask"),
  askAnswer: document.getElementById("ask-answer"),
  patientDialog: document.getElementById("patient-dialog"),
  patientForm: document.getElementById("patient-form"),
  btnCancelPatient: document.getElementById("btn-cancel-patient"),
  reportDialog: document.getElementById("report-dialog"),
  reportBody: document.getElementById("report-body"),
  reportMeta: document.getElementById("report-meta"),
  reportLanguage: document.getElementById("report-language"),
  btnSpeakReport: document.getElementById("btn-speak-report"),
  btnCloseReport: document.getElementById("btn-close-report"),
  sessionsTbody: document.getElementById("sessions-tbody"),
};

let patients = [];
let currentReportSession = null;
let currentReport = null;

// ------------------------------------------------------------------ voice (browser TTS, works offline)
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", mr: "mr-IN" };
const REP_WORD = { en: "Rep", hi: "दोहराव", mr: "पुनरावृत्ती" };
const WARNING_KEYS = new Set(["too_fast", "insufficient_range", "posture_warning", "low_visibility"]);

function pickVoice(lang) {
  const voices = window.speechSynthesis ? speechSynthesis.getVoices() : [];
  const want = SPEECH_LANG[lang] || "en-IN";
  return (
    voices.find((v) => v.lang === want) ||
    voices.find((v) => v.lang.startsWith(want.slice(0, 2))) ||
    // Marathi voices are rare; Devanagari text reads acceptably with a Hindi voice
    (lang === "mr" ? voices.find((v) => v.lang.startsWith("hi")) : null) ||
    null
  );
}

function speak(text, lang, { interrupt = false } = {}) {
  if (!window.speechSynthesis || !text) return;
  if (interrupt) speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = SPEECH_LANG[lang] || "en-IN";
  const voice = pickVoice(lang);
  if (voice) u.voice = voice;
  u.rate = 0.92; // slightly slower for elderly patients
  speechSynthesis.speak(u);
}
if (window.speechSynthesis) speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();

const voiceState = { lastReps: 0, lastKey: "", lastWarnTs: 0, wasActive: false };

function handleVoice(data) {
  const lang = coachEls.languageSelect.value;
  if (!data.session_active) {
    voiceState.wasActive = false;
    voiceState.lastReps = data.reps;
    return;
  }
  if (!voiceState.wasActive) {
    voiceState.wasActive = true;
    voiceState.lastReps = data.reps;
    if (coachEls.voiceToggle.checked) speak(data.feedback_i18n?.[lang], lang, { interrupt: true });
    return;
  }
  if (!coachEls.voiceToggle.checked) {
    voiceState.lastReps = data.reps;
    return;
  }
  const now = Date.now();
  if (data.reps > voiceState.lastReps) {
    voiceState.lastReps = data.reps;
    speak(`${REP_WORD[lang]} ${data.reps}`, lang, { interrupt: true });
    voiceState.lastKey = "completed";
    return;
  }
  const key = data.feedback_key;
  if (WARNING_KEYS.has(key) && key !== voiceState.lastKey && now - voiceState.lastWarnTs > 4000) {
    voiceState.lastWarnTs = now;
    speak(data.feedback_i18n?.[lang], lang);
  }
  voiceState.lastKey = key;
}

// ------------------------------------------------------------------ telemetry hook
window.onTelemetryExtra = (data) => {
  handleVoice(data);
  coachEls.targetAngle.textContent = data.target_threshold != null ? `${data.target_threshold}°` : "--°";
  coachEls.startAngle.textContent = data.start_threshold != null ? `${data.start_threshold}°` : "--°";
  const patient = patients.find((p) => p.id === data.patient_id);
  const hasOverride = patient && patient.threshold_overrides?.[data.exercise_key]?.target_angle_threshold != null;
  coachEls.targetSource.textContent = hasOverride ? "(personalized)" : "(default)";
  if (document.activeElement !== coachEls.sideSelect && coachEls.sideSelect.value !== data.side) {
    coachEls.sideSelect.value = data.side;
  }
  const r = data.last_rep;
  coachEls.lastRep.textContent = r
    ? `Last rep #${r.rep}: peak ${r.peak_angle}° · ROM ${r.rom_deg}° · ${r.duration_s}s · max speed ${r.max_velocity}°/s` +
      (r.left_peak != null && r.right_peak != null ? ` · L ${r.left_peak}° / R ${r.right_peak}°` : "")
    : "Last rep: --";
};

// ------------------------------------------------------------------ patients
async function loadPatients(selectId) {
  const res = await fetch("/api/patients");
  patients = (await res.json()).patients || [];
  const current = selectId ?? coachEls.patientSelect.value;
  coachEls.patientSelect.innerHTML =
    '<option value="">— Guest (no profile) —</option>' +
    patients
      .map((p) => `<option value="${p.id}">${esc(p.name)}${p.age ? ` (${p.age})` : ""} — ${esc(p.condition || "")}</option>`)
      .join("");
  coachEls.patientSelect.value = current ? String(current) : "";
}

async function selectPatient(id) {
  await postJson("/api/patient/select", { patient_id: id ? Number(id) : null });
  coachEls.guestGroup.hidden = Boolean(id);
  const patient = patients.find((p) => String(p.id) === String(id));
  if (patient?.language) {
    coachEls.languageSelect.value = patient.language;
    coachEls.reportLanguage.value = patient.language;
  }
  await loadProgress();
}

coachEls.patientSelect.addEventListener("change", () => selectPatient(coachEls.patientSelect.value));
coachEls.btnNewPatient.addEventListener("click", () => coachEls.patientDialog.showModal());
coachEls.btnCancelPatient.addEventListener("click", () => coachEls.patientDialog.close());
coachEls.patientForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = Object.fromEntries(new FormData(coachEls.patientForm).entries());
  const res = await postJson("/api/patients", body);
  if (res.error) return alert(res.error);
  coachEls.patientDialog.close();
  coachEls.patientForm.reset();
  await loadPatients(res.patient.id);
  await selectPatient(res.patient.id);
});

// ------------------------------------------------------------------ progress chart + adaptive target
const progressChart = new Chart(document.getElementById("progressChart").getContext("2d"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      { label: "Best angle (°)", data: [], borderColor: "#38bdf8", yAxisID: "y", tension: 0.25 },
      { label: "Avg ROM / rep (°)", data: [], borderColor: "#a78bfa", yAxisID: "y", tension: 0.25 },
      { label: "Consistency (%)", data: [], borderColor: "#22c55e", yAxisID: "y1", tension: 0.25, borderDash: [5, 4] },
      { label: "Target (°)", data: [], borderColor: "#f59e0b", yAxisID: "y", pointRadius: 0, borderDash: [2, 3] },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { title: { display: true, text: "Degrees" } },
      y1: { position: "right", min: 0, max: 100, grid: { drawOnChartArea: false }, title: { display: true, text: "%" } },
    },
  },
});

async function loadProgress() {
  const pid = coachEls.patientSelect.value;
  const exKey = coachEls.exerciseSelect.value;
  coachEls.suggestion.hidden = true;
  if (!pid) {
    coachEls.progressTitle.textContent = "select a patient";
    progressChart.data.labels = [];
    progressChart.data.datasets.forEach((d) => (d.data = []));
    progressChart.update();
    return;
  }
  const res = await fetch(`/api/patients/${pid}/progress`);
  if (!res.ok) return;
  const ctx = await res.json();
  const ex = ctx.exercises?.[exKey];
  const patient = patients.find((p) => String(p.id) === pid);
  coachEls.progressTitle.textContent = `${patient?.name ?? "Patient"} · ${coachEls.exerciseSelect.selectedOptions[0].text}`;
  const trend = ex?.trend || [];
  progressChart.data.labels = trend.map((t) => t.date.slice(5, 16));
  progressChart.data.datasets[0].data = trend.map((t) => t.best_peak_angle);
  progressChart.data.datasets[1].data = trend.map((t) => t.avg_rom_deg);
  progressChart.data.datasets[2].data = trend.map((t) => t.consistency_score);
  progressChart.data.datasets[3].data = trend.map(() => ex.current_target);
  progressChart.update();
  renderSuggestion(ex?.target_suggestion, Number(pid), exKey);
}

function renderSuggestion(sug, patientId, exerciseKey) {
  const box = coachEls.suggestion;
  if (!sug) {
    box.hidden = false;
    box.className = "suggestion";
    box.textContent = "No sessions yet for this exercise.";
    return;
  }
  box.hidden = false;
  if (sug.action === "keep") {
    box.className = "suggestion";
    box.innerHTML = `🎯 Target ${sug.current_target}° — ${esc(sug.reason)}`;
    return;
  }
  const verb = sug.action === "increase_difficulty" ? "Increase" : "Reduce";
  box.className = "suggestion suggestion-action";
  box.innerHTML = `
    <div>💡 <strong>Suggested:</strong> ${verb} target ${sug.current_target}° → <strong>${sug.suggested_target}°</strong><br>
    <small>${esc(sug.reason)} Requires physiotherapist approval.</small></div>
    <button class="btn btn-success btn-sm" id="btn-approve-target">Approve (Physio)</button>`;
  document.getElementById("btn-approve-target").onclick = async () => {
    if (!confirm(`Physiotherapist approval: set target to ${sug.suggested_target}° for this patient?`)) return;
    const res = await postJson(`/api/patients/${patientId}/targets`, {
      exercise_key: exerciseKey,
      target_angle_threshold: sug.suggested_target,
    });
    if (res.error) return alert(res.error);
    await loadPatients(patientId);
    await loadProgress();
  };
}

coachEls.exerciseSelect.addEventListener("change", () => setTimeout(loadProgress, 150));

// ------------------------------------------------------------------ AI status + Q&A
async function loadAiStatus() {
  const s = await (await fetch("/api/ai/status")).json();
  coachEls.aiStatus.textContent = s.available ? `AI: Gemini (${s.model})` : "AI: offline mode (no API key)";
  coachEls.aiStatus.className = s.available ? "pill pill-ok" : "pill pill-warn";
}

coachEls.btnAsk.addEventListener("click", async () => {
  const pid = coachEls.patientSelect.value;
  const question = coachEls.askInput.value.trim();
  if (!pid) return (coachEls.askAnswer.textContent = "Select a patient profile first.");
  if (!question) return;
  coachEls.btnAsk.disabled = true;
  coachEls.askAnswer.textContent = "Thinking…";
  try {
    const res = await postJson(`/api/patients/${pid}/ask`, { question, language: coachEls.languageSelect.value });
    coachEls.askAnswer.textContent = res.answer || res.error;
    coachEls.askAnswer.classList.remove("muted");
  } finally {
    coachEls.btnAsk.disabled = false;
  }
});
coachEls.askInput.addEventListener("keydown", (e) => e.key === "Enter" && coachEls.btnAsk.click());

// ------------------------------------------------------------------ session reports
function listHtml(value) {
  if (Array.isArray(value)) return `<ul>${value.map((v) => `<li>${esc(v)}</li>`).join("")}</ul>`;
  return `<p>${esc(value)}</p>`;
}

async function openReport(sessionId, refresh = false) {
  currentReportSession = sessionId;
  if (!coachEls.reportDialog.open) coachEls.reportDialog.showModal();
  coachEls.reportBody.innerHTML = '<p class="muted">Generating report…</p>';
  coachEls.reportMeta.textContent = `Session #${sessionId}`;
  const r = await postJson(`/api/sessions/${sessionId}/report`, {
    language: coachEls.reportLanguage.value,
    refresh,
  });
  if (r.error) {
    coachEls.reportBody.textContent = r.error;
    return;
  }
  currentReport = r;
  const sug = r.target_suggestion;
  coachEls.reportMeta.textContent = `Session #${sessionId} · ${r.source}${r.cached ? " (saved)" : ""}`;
  coachEls.reportBody.innerHTML = `
    <h3>For the patient</h3>
    <p class="report-patient">${esc(r.patient_summary)}</p>
    <p><strong>Next time:</strong> ${esc(r.next_session_tip)}</p>
    <h3>For the physiotherapist</h3>
    ${listHtml(r.clinician_summary)}
    ${r.flags && r.flags.length ? `<h3>⚠️ Flags to review</h3>${listHtml(r.flags)}` : ""}
    ${sug && sug.action !== "keep" ? `<p class="suggestion suggestion-action">💡 Target suggestion: ${sug.current_target}° → ${sug.suggested_target}° (${esc(sug.reason)}) — approve it in the Patient Progress panel.</p>` : ""}
    ${r.ai_error ? `<p class="muted small">AI unavailable, showing offline report: ${esc(r.ai_error)}</p>` : ""}
    <p class="muted small">AI-generated from measured metrics. Not a medical assessment.</p>
    <button class="btn btn-secondary btn-sm" id="btn-regen-report">↻ Regenerate</button>`;
  document.getElementById("btn-regen-report").onclick = () => openReport(sessionId, true);
}

coachEls.sessionsTbody.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-report]");
  if (btn) openReport(Number(btn.dataset.report));
});
coachEls.reportLanguage.addEventListener("change", () => currentReportSession && openReport(currentReportSession));
coachEls.btnCloseReport.addEventListener("click", () => {
  speechSynthesis?.cancel();
  coachEls.reportDialog.close();
});
coachEls.btnSpeakReport.addEventListener("click", () => {
  if (!currentReport) return;
  speak(`${currentReport.patient_summary} ${currentReport.next_session_tip}`, coachEls.reportLanguage.value, {
    interrupt: true,
  });
});

window.onSessionSaved = async (stopped) => {
  if (stopped?.ok && stopped.session) {
    await loadProgress();
    coachEls.reportLanguage.value = coachEls.languageSelect.value;
    openReport(stopped.session.id);
  }
};

// ------------------------------------------------------------------ init
(async () => {
  // Sync with the server's current patient without resetting an in-progress session
  const tel = await (await fetch("/api/telemetry")).json();
  await loadPatients(tel.patient_id || "");
  coachEls.guestGroup.hidden = Boolean(tel.patient_id);
  if (tel.exercise_key) coachEls.exerciseSelect.value = tel.exercise_key;
  await loadAiStatus();
  await loadProgress();
})();
