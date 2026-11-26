/* Phase 7, 8 & 9: Offline-ready dashboard telemetry, calibration, side selector, hardware stats & charts */

const els = {
  pillBackend: document.getElementById("pill-backend"),
  pillAngleMode: document.getElementById("pill-angle-mode"),
  pillFps: document.getElementById("pill-fps"),
  pillLatency: document.getElementById("pill-latency"),
  pillCam: document.getElementById("pill-cam"),
  hwSource: document.getElementById("hw-source"),
  hwCpu: document.getElementById("hw-cpu"),
  hwGpu: document.getElementById("hw-gpu"),
  hwRam: document.getElementById("hw-ram"),
  hwTemp: document.getElementById("hw-temp"),
  hwPower: document.getElementById("hw-power"),
  cameraInput: document.getElementById("camera-input"),
  cameraError: document.getElementById("camera-error"),
  btnConnectCamera: document.getElementById("btn-connect-camera"),
  btnDemoMode: document.getElementById("btn-demo-mode"),
  exerciseSelect: document.getElementById("exercise-select"),
  sideSelect: document.getElementById("side-select"),
  patientLabel: document.getElementById("patient-label"),
  btnCalibrate: document.getElementById("btn-calibrate"),
  calibrationStatus: document.getElementById("calibration-status"),
  btnStartSession: document.getElementById("btn-start-session"),
  btnStopSession: document.getElementById("btn-stop-session"),
  btnRefreshSessions: document.getElementById("btn-refresh-sessions"),
  stageBadge: document.getElementById("stage-badge"),
  feedbackBanner: document.getElementById("feedback-banner"),
  kpiReps: document.getElementById("kpi-reps"),
  kpiAngle: document.getElementById("kpi-angle"),
  kpiScore: document.getElementById("kpi-score"),
  kpiDuration: document.getElementById("kpi-duration"),
  kpiMinAngle: document.getElementById("kpi-min-angle"),
  kpiMaxAngle: document.getElementById("kpi-max-angle"),
  sessionsTbody: document.getElementById("sessions-tbody"),
};

// 1. Live Joint Angle Chart (offline local renderer)
const liveAngleChart = new Chart(
  document.getElementById("liveAngleChart").getContext("2d"),
  {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Primary Joint Angle (°)",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.14)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { display: false },
        y: { min: 0, max: 185 },
      },
    },
  }
);

// 2. Session History Comparison Chart (offline local renderer)
const sessionHistoryChart = new Chart(
  document.getElementById("sessionHistoryChart").getContext("2d"),
  {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Total Reps",
          data: [],
          backgroundColor: "#22c55e",
          yAxisID: "y",
        },
        {
          label: "Consistency (%)",
          data: [],
          backgroundColor: "#38bdf8",
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  }
);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function repQualityCell(d) {
  if (!d || !d.rep_count) return '<span class="muted">--</span>';
  const parts = [`ROM ${d.avg_rom_deg ?? "--"}°`, `${d.avg_rep_duration_s ?? "--"}s/rep`];
  if (d.symmetry_gap_deg !== null && d.symmetry_gap_deg !== undefined) parts.push(`L/R gap ${d.symmetry_gap_deg}°`);
  return `<small>${parts.join(" · ")}</small>`;
}

function setStatusClass(el, status) {
  el.classList.remove("status-idle", "status-good", "status-warning", "status-caution");
  el.classList.add(`status-${status || "idle"}`);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res.json();
}

function updateHardwareStrip(hw) {
  if (!hw) return;
  els.hwSource.textContent = hw.source || "--";
  els.hwCpu.textContent = `CPU: ${hw.cpu_percent !== null && hw.cpu_percent !== undefined ? hw.cpu_percent + "%" : "--%"}`;
  els.hwGpu.textContent = `GPU: ${hw.gpu_percent !== null && hw.gpu_percent !== undefined ? hw.gpu_percent + "%" : "N/A"}`;
  if (hw.ram_used_mb !== null && hw.ram_total_mb !== null && hw.ram_used_mb !== undefined) {
    els.hwRam.textContent = `RAM: ${hw.ram_used_mb}/${hw.ram_total_mb} MB (${hw.ram_percent ?? "--"}%)`;
  } else {
    els.hwRam.textContent = "RAM: --";
  }
  els.hwTemp.textContent = `Temp: ${hw.temp_c !== null && hw.temp_c !== undefined ? hw.temp_c + "°C" : "N/A"}`;
  els.hwPower.textContent = `Power: ${hw.power_w !== null && hw.power_w !== undefined ? hw.power_w + " W" : "N/A"}`;
}

async function pollTelemetry() {
  try {
    const res = await fetch("/api/telemetry");
    if (!res.ok) return;
    const data = await res.json();

    els.pillBackend.textContent = `Backend: ${data.pose_backend}`;
    els.pillAngleMode.textContent = `Angles: ${data.angle_mode || "2D"}`;
    els.pillFps.textContent = `FPS: ${data.fps}`;
    els.pillLatency.textContent = `Inference: ${data.inference_ms} ms`;

    if (data.camera_connected) {
      els.pillCam.textContent = `Camera: Connected (${data.camera_source})`;
      els.pillCam.className = "pill pill-ok";
    } else {
      els.pillCam.textContent = "Camera: Disconnected";
      els.pillCam.className = "pill pill-warn";
    }

    updateHardwareStrip(data.hardware_stats);

    if (data.camera_error) {
      els.cameraError.textContent = data.camera_error;
      els.cameraError.hidden = false;
    } else {
      els.cameraError.hidden = true;
    }

    els.kpiReps.textContent = data.reps;
    els.kpiAngle.textContent = data.primary_angle !== null ? `${data.primary_angle}°` : "--°";
    els.kpiScore.textContent = `${data.consistency_score}%`;
    els.kpiDuration.textContent = data.session_active
      ? `${data.duration_seconds}s (${data.logged_frames}f)`
      : `${data.duration_seconds}s`;
    els.kpiMinAngle.textContent = data.min_angle !== null ? `${data.min_angle}°` : "--°";
    els.kpiMaxAngle.textContent = data.max_angle !== null ? `${data.max_angle}°` : "--°";

    if (data.calibrating) {
      els.btnCalibrate.disabled = true;
      els.calibrationStatus.textContent = `Hold still: ${data.calibration_remaining_s}s`;
    } else {
      els.btnCalibrate.disabled = false;
      els.calibrationStatus.textContent = data.calibration_message || "Not calibrated";
    }

    els.stageBadge.textContent = String(data.stage || "ready").replace("_", " ").toUpperCase();
    setStatusClass(els.stageBadge, data.status);

    const lang = document.getElementById("language-select").value;
    els.feedbackBanner.textContent =
      (data.feedback_i18n && data.feedback_i18n[lang]) || data.feedback;
    setStatusClass(els.feedbackBanner, data.status);

    els.btnStartSession.disabled = Boolean(data.session_active);
    els.btnStopSession.disabled = !Boolean(data.session_active);

    if (typeof window.onTelemetryExtra === "function") window.onTelemetryExtra(data);

    if (Array.isArray(data.angle_history)) {
      liveAngleChart.data.labels = data.angle_history.map((_, i) => i);
      liveAngleChart.data.datasets[0].data = data.angle_history.map((p) => p.angle);
      liveAngleChart.update("none");
    }
  } catch (err) {
    console.error("Telemetry poll error:", err);
  }
}

async function loadSessions() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const { sessions } = await res.json();

    if (!sessions || sessions.length === 0) {
      els.sessionsTbody.innerHTML =
        '<tr><td colspan="10" class="empty-row">No sessions recorded yet. Click "Start Session" then "Stop &amp; Save".</td></tr>';
      return;
    }

    els.sessionsTbody.innerHTML = sessions
      .map(
        (s) => `
        <tr>
          <td>${esc(s.created_at)}</td>
          <td>${esc(s.patient_label)}</td>
          <td>${esc(s.exercise_name)} <small>(${esc(s.side || "right")})</small></td>
          <td><strong>${s.total_reps}</strong></td>
          <td>${s.duration_seconds}</td>
          <td>${s.min_angle ?? "--"}° / ${s.max_angle ?? "--"}°</td>
          <td>${s.consistency_score}%</td>
          <td>${repQualityCell(s.rep_details)}</td>
          <td>
            <a href="/api/sessions/${s.id}/frames.csv" class="btn btn-secondary btn-sm">
              CSV (${s.frame_count ?? 0} frames)
            </a>
          </td>
          <td><button class="btn btn-primary btn-sm" data-report="${s.id}">✨ Report</button></td>
        </tr>`
      )
      .join("");

    const recent = sessions.slice(0, 8).reverse();
    sessionHistoryChart.data.labels = recent.map((s) => `#${s.id} ${s.exercise_name}`);
    sessionHistoryChart.data.datasets[0].data = recent.map((s) => s.total_reps);
    sessionHistoryChart.data.datasets[1].data = recent.map((s) => s.consistency_score);
    sessionHistoryChart.update();
  } catch (err) {
    console.error("Failed to load sessions:", err);
  }
}

// Event listeners
els.btnConnectCamera.addEventListener("click", async () => {
  const src = els.cameraInput.value.trim();
  if (!src) return;
  await postJson("/api/camera", { camera_source: src });
});

els.btnDemoMode.addEventListener("click", async () => {
  els.cameraInput.value = "demo";
  await postJson("/api/camera", { camera_source: "demo" });
});

els.exerciseSelect.addEventListener("change", async () => {
  await postJson("/api/exercise", { exercise_key: els.exerciseSelect.value });
});

els.sideSelect.addEventListener("change", async () => {
  await postJson("/api/side", { side: els.sideSelect.value });
});

els.btnCalibrate.addEventListener("click", async () => {
  await postJson("/api/calibrate", { duration_s: 3.0 });
  await pollTelemetry();
});

els.btnStartSession.addEventListener("click", async () => {
  await postJson("/api/session/start", { patient_label: els.patientLabel.value });
  await pollTelemetry();
});

els.btnStopSession.addEventListener("click", async () => {
  const stopped = await postJson("/api/session/stop", {});
  await pollTelemetry();
  await loadSessions();
  if (typeof window.onSessionSaved === "function") window.onSessionSaved(stopped);
});

els.btnRefreshSessions.addEventListener("click", loadSessions);

// Initial load & polling loop
loadSessions();
pollTelemetry();
setInterval(pollTelemetry, 250);
