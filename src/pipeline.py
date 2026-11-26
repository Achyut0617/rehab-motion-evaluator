"""End-to-end real-time rehabilitation processing pipeline for Flask dashboard."""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Generator, List, Optional, Union

import cv2
import numpy as np

from src.angle_calculator import JointAngleCalculator
from src.camera import VideoStream
from src.config import (
    CAMERA_SOURCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MEDIAPIPE_MODEL_PATH,
    POSE_BACKEND,
    YOLO_MODEL_PATH,
    load_exercise_configs,
    parse_camera_source,
)
from src.evaluator import MovementEvaluator
from src.exercise_tracker import ExerciseTracker
from src.jetson_monitor import JetsonHardwareMonitor
from src.pose_estimator import PoseEstimator
from src.progress import progress_trend, suggest_target, summarize_reps
from src.session_manager import SessionManager


class RehabPipeline:
    """Coordinates camera, pose estimation, 3D/2D angles, calibration, telemetry logging, and hardware stats."""

    def __init__(
        self,
        camera_source: Union[int, str] = CAMERA_SOURCE,
        pose_backend: str = POSE_BACKEND,
    ) -> None:
        self.exercise_configs = load_exercise_configs()
        self.current_exercise = next(iter(self.exercise_configs.keys()))

        self.stream = VideoStream(
            source=camera_source, width=FRAME_WIDTH, height=FRAME_HEIGHT
        )
        self.estimator = PoseEstimator(
            backend=pose_backend,
            mediapipe_model_path=MEDIAPIPE_MODEL_PATH,
            yolo_model_path=YOLO_MODEL_PATH,
        )
        self.estimator.set_demo_exercise(self.current_exercise)
        self.angle_calc = JointAngleCalculator(use_3d=True)
        self.tracker = ExerciseTracker(
            self.current_exercise, self.exercise_configs[self.current_exercise]
        )
        self.evaluator = MovementEvaluator(self.exercise_configs[self.current_exercise])
        self.session_manager = SessionManager()
        self.hw_monitor = JetsonHardwareMonitor(interval_ms=1000)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None

        self._session_active: bool = False
        self._session_start_ts: Optional[float] = None
        self._patient_label: str = "Anonymous"
        self._patient: Optional[Dict[str, Any]] = None
        self._session_frames: List[Dict[str, Any]] = []
        self._last_logged_frame_ts: float = 0.0

        # 3-second resting-pose calibration state
        self._calibrating: bool = False
        self._cal_end_ts: float = 0.0
        self._cal_samples: List[float] = []
        self._cal_message: str = "Not calibrated"

        self._angle_history: Deque[Dict[str, Any]] = deque(maxlen=90)
        self._inference_ms: float = 0.0
        self._telemetry: Dict[str, Any] = self._build_initial_telemetry()

    def _build_initial_telemetry(self) -> Dict[str, Any]:
        cfg = self.exercise_configs[self.current_exercise]
        return {
            "exercise_key": self.current_exercise,
            "exercise_name": cfg["display_name"],
            "side": self.tracker.side,
            "reps": 0,
            "stage": "ready",
            "primary_angle": None,
            "secondary_angle": None,
            "left_angle": None,
            "right_angle": None,
            "min_angle": None,
            "max_angle": None,
            "angular_velocity": 0.0,
            "angle_mode": self.angle_calc.last_mode,
            "status": "idle",
            "feedback": cfg.get("feedback", {}).get("ready", "Ready"),
            "consistency_score": 0.0,
            "session_active": False,
            "duration_seconds": 0.0,
            "logged_frames": 0,
            "fps": 0.0,
            "inference_ms": 0.0,
            "camera_source": str(self.stream.source),
            "camera_connected": False,
            "camera_error": "",
            "pose_backend": self.estimator.active_backend,
            "calibrating": False,
            "calibration_remaining_s": 0.0,
            "calibration_message": self._cal_message,
            "hardware_stats": self.hw_monitor.get_stats(),
            "angle_history": [],
        }

    def start(self) -> None:
        if self._running:
            return
        self.stream.start()
        self.hw_monitor.start()
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.stream.stop()
        self.hw_monitor.stop()
        self.estimator.close()

    def select_exercise(self, exercise_key: str) -> bool:
        if exercise_key not in self.exercise_configs:
            return False
        with self._lock:
            self.current_exercise = exercise_key
            cfg = self.exercise_configs[exercise_key]
            self.estimator.set_demo_exercise(exercise_key)
            self.angle_calc.reset()
            self.tracker.configure(exercise_key, cfg, side=self.tracker.side)
            self._apply_patient_overrides()
            self.evaluator.configure(cfg)
            self._angle_history.clear()
            self._calibrating = False
            self._cal_samples.clear()
            self._cal_message = "Not calibrated"
            if self._session_active:
                self._session_start_ts = time.time()
                self._session_frames.clear()
        return True

    # ------------------------------------------------------------------ patients
    def _apply_patient_overrides(self) -> None:
        """Apply this patient's physio-approved thresholds for the current exercise (lock held)."""
        if self._patient:
            overrides = self._patient.get("threshold_overrides", {}).get(self.current_exercise)
            self.tracker.apply_overrides(overrides)

    def set_patient(self, patient_id: Optional[int]) -> Optional[Dict[str, Any]]:
        patient = self.session_manager.get_patient(patient_id) if patient_id else None
        with self._lock:
            self._patient = patient
            cfg = self.exercise_configs[self.current_exercise]
            self.tracker.configure(self.current_exercise, cfg, side=self.tracker.side)
            if patient:
                self._patient_label = patient["name"]
                if patient.get("affected_side") in ("left", "right", "both"):
                    self.tracker.set_side(patient["affected_side"])
            self._apply_patient_overrides()
        return patient

    def refresh_patient(self) -> None:
        if self._patient:
            self.set_patient(self._patient["id"])

    def set_side(self, side: str) -> str:
        with self._lock:
            chosen = self.tracker.set_side(side)
            self.angle_calc.reset()
            self._angle_history.clear()
            self._calibrating = False
            self._cal_samples.clear()
            self._cal_message = "Not calibrated"
            return chosen

    def start_calibration(self, duration_s: float = 3.0) -> Dict[str, Any]:
        with self._lock:
            self._calibrating = True
            self._cal_end_ts = time.time() + duration_s
            self._cal_samples.clear()
            self._cal_message = f"Hold resting pose ({duration_s:.0f}s)..."
            return {"calibrating": True, "duration_s": duration_s}

    def set_camera_source(self, raw_source: str) -> str:
        parsed = parse_camera_source(raw_source)
        self.stream.switch_source(parsed)
        return str(parsed)

    def start_session(self, patient_label: str = "Anonymous") -> Dict[str, Any]:
        with self._lock:
            if self._patient:
                self._patient_label = self._patient["name"]
            else:
                self._patient_label = patient_label.strip() or "Anonymous"
            self.angle_calc.reset()
            self.tracker.reset()
            self.evaluator.reset()
            self._angle_history.clear()
            self._session_frames.clear()
            self._last_logged_frame_ts = 0.0
            self._session_active = True
            self._session_start_ts = time.time()
        return {"session_active": True, "patient_label": self._patient_label}

    def stop_session(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._session_active:
                return None
            duration = (
                time.time() - self._session_start_ts if self._session_start_ts else 0.0
            )
            cfg = self.exercise_configs[self.current_exercise]
            summary = self._telemetry.get("feedback", "Session completed")
            saved = self.session_manager.save_session(
                exercise_key=self.current_exercise,
                exercise_name=cfg["display_name"],
                total_reps=self.tracker.reps,
                duration_seconds=duration,
                min_angle=self.tracker.min_observed_angle,
                max_angle=self.tracker.max_observed_angle,
                consistency_score=self.evaluator.consistency_score,
                evaluation_summary=summary,
                patient_label=self._patient_label,
                side=self.tracker.side,
                frames=list(self._session_frames),
                patient_id=self._patient["id"] if self._patient else None,
                rep_details=summarize_reps(
                    self.tracker.rep_log, cfg.get("direction", "extend")
                ),
            )
            self._session_active = False
            self._session_start_ts = None
            self._session_frames.clear()
            return saved

    def _draw_overlay(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, Any],
        angles: Dict[str, Optional[float]],
        telemetry: Dict[str, Any],
    ) -> np.ndarray:
        self.estimator.draw_skeleton(frame, landmarks)

        for joint_name, angle_val in angles.items():
            if angle_val is None or joint_name not in landmarks:
                continue
            lm = landmarks[joint_name]
            pos = (int(lm.x) + 10, int(lm.y) - 8)
            cv2.putText(
                frame,
                f"{int(round(angle_val))} deg",
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (250, 204, 21),
                2,
                cv2.LINE_AA,
            )

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 58), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        hud_left = (
            f"{telemetry['exercise_name']} ({telemetry['side']}) | "
            f"Reps: {telemetry['reps']} | Score: {telemetry['consistency_score']}%"
        )
        cv2.putText(
            frame, hud_left, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (248, 250, 252), 1, cv2.LINE_AA
        )
        hud_right = (
            f"FPS: {telemetry['fps']} | {telemetry['inference_ms']}ms | "
            f"{telemetry['pose_backend']} [{telemetry['angle_mode']}]"
        )
        cv2.putText(
            frame, hud_right, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA
        )

        if telemetry.get("calibrating"):
            rem = telemetry.get("calibration_remaining_s", 0.0)
            cv2.putText(
                frame,
                f"CALIBRATING RESTING POSE: {rem:.1f}s",
                (14, frame.shape[0] - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (56, 189, 248),
                2,
                cv2.LINE_AA,
            )
        return frame

    def _make_waiting_frame(self) -> np.ndarray:
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (18, 24, 34), dtype=np.uint8)
        msg = self.stream.last_error or f"Connecting to {self.stream.source}..."
        cv2.putText(
            frame,
            "Waiting for camera stream",
            (24, FRAME_HEIGHT // 2 - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            msg[:72],
            (24, FRAME_HEIGHT // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (248, 113, 113),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _process_loop(self) -> None:
        while self._running:
            ok, frame = self.stream.read()
            if not ok or frame is None:
                waiting = self._make_waiting_frame()
                ret, buf = cv2.imencode(".jpg", waiting, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._telemetry["camera_connected"] = False
                        self._telemetry["camera_error"] = self.stream.last_error
                        self._telemetry["camera_source"] = str(self.stream.source)
                        self._telemetry["hardware_stats"] = self.hw_monitor.get_stats()
                time.sleep(0.08)
                continue

            t0 = time.perf_counter()
            landmarks = self.estimator.detect(frame, force_demo=self.stream.is_demo)
            self._inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            now = time.time()
            with self._lock:
                angles = self.angle_calc.calculate_all(landmarks)
                rep_update = self.tracker.update(angles, timestamp=now)
                eval_result = self.evaluator.evaluate(rep_update)

                # Handle 3-second resting-pose calibration
                cal_remaining = 0.0
                if self._calibrating:
                    if rep_update.primary_angle is not None:
                        self._cal_samples.append(rep_update.primary_angle)
                    cal_remaining = max(0.0, self._cal_end_ts - now)
                    if cal_remaining <= 0.0:
                        self._calibrating = False
                        if self._cal_samples:
                            median_rest = float(statistics.median(self._cal_samples))
                            cal_info = self.tracker.apply_calibration(median_rest)
                            self._cal_message = (
                                f"Calibrated (rest {cal_info['resting_angle']}°, "
                                f"start {cal_info['start_angle_threshold']}°)"
                            )
                        else:
                            self._cal_message = "Calibration failed — joint not visible"
                        self._cal_samples.clear()

                duration = (
                    round(now - self._session_start_ts, 1)
                    if self._session_active and self._session_start_ts
                    else 0.0
                )

                if rep_update.primary_angle is not None:
                    self._angle_history.append(
                        {"t": round(now, 2), "angle": rep_update.primary_angle}
                    )

                # Record per-frame telemetry at up to 10 Hz during active sessions
                if (
                    self._session_active
                    and self._session_start_ts is not None
                    and (now - self._last_logged_frame_ts) >= 0.1
                ):
                    self._last_logged_frame_ts = now
                    self._session_frames.append(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            )[:-3],
                            "elapsed_s": round(now - self._session_start_ts, 3),
                            "stage": rep_update.stage,
                            "primary_angle": rep_update.primary_angle,
                            "secondary_angle": rep_update.secondary_angle,
                            "left_angle": rep_update.left_angle,
                            "right_angle": rep_update.right_angle,
                            "angular_velocity": rep_update.angular_velocity,
                            "reps": rep_update.reps,
                            "consistency_score": eval_result.consistency_score,
                            "angle_mode": self.angle_calc.last_mode,
                            "feedback": eval_result.feedback,
                        }
                    )

                cfg = self.exercise_configs[self.current_exercise]
                telemetry = {
                    "exercise_key": self.current_exercise,
                    "exercise_name": cfg["display_name"],
                    "side": self.tracker.side,
                    "reps": rep_update.reps,
                    "stage": rep_update.stage,
                    "primary_angle": rep_update.primary_angle,
                    "secondary_angle": rep_update.secondary_angle,
                    "left_angle": rep_update.left_angle,
                    "right_angle": rep_update.right_angle,
                    "min_angle": rep_update.min_observed_angle,
                    "max_angle": rep_update.max_observed_angle,
                    "angular_velocity": rep_update.angular_velocity,
                    "angle_mode": self.angle_calc.last_mode,
                    "status": eval_result.status,
                    "feedback": eval_result.feedback,
                    "consistency_score": eval_result.consistency_score,
                    "session_active": self._session_active,
                    "duration_seconds": duration,
                    "logged_frames": len(self._session_frames),
                    "fps": self.stream.fps,
                    "inference_ms": self._inference_ms,
                    "camera_source": str(self.stream.source),
                    "camera_connected": self.stream.connected,
                    "camera_error": self.stream.last_error,
                    "pose_backend": "demo" if self.stream.is_demo else self.estimator.active_backend,
                    "calibrating": self._calibrating,
                    "calibration_remaining_s": round(cal_remaining, 1),
                    "calibration_message": self._cal_message,
                    "hardware_stats": self.hw_monitor.get_stats(),
                    "angle_history": list(self._angle_history),
                    "feedback_key": eval_result.feedback_key,
                    "feedback_i18n": {
                        lang: cfg.get(field, {}).get(eval_result.feedback_key, eval_result.feedback)
                        for lang, field in (("en", "feedback"), ("hi", "feedback_hi"), ("mr", "feedback_mr"))
                    },
                    "last_rep": self.tracker.rep_log[-1] if self.tracker.rep_log else None,
                    "target_threshold": self.tracker.cfg.get("target_angle_threshold"),
                    "start_threshold": self.tracker.cfg.get("start_angle_threshold"),
                    "patient_id": self._patient["id"] if self._patient else None,
                    "patient_name": self._patient["name"] if self._patient else None,
                }
                self._telemetry = telemetry

            annotated = self._draw_overlay(frame, landmarks, angles, telemetry)
            ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            time.sleep(0.01)

    # ------------------------------------------------------------------ AI context
    def _patient_profile(self, patient: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not patient:
            return None
        # Deliberately excludes the patient name (privacy when calling a cloud LLM)
        return {
            "age": patient.get("age"),
            "condition": patient.get("condition"),
            "affected_side": patient.get("affected_side"),
            "physio_notes": patient.get("notes"),
        }

    def _effective_target(self, patient: Optional[Dict[str, Any]], exercise_key: str) -> float:
        cfg = self.exercise_configs[exercise_key]
        override = ((patient or {}).get("threshold_overrides") or {}).get(exercise_key, {})
        return float(override.get("target_angle_threshold", cfg["target_angle_threshold"]))

    def build_session_context(self, session_id: int) -> Optional[Dict[str, Any]]:
        session = self.session_manager.get_session(session_id)
        if session is None:
            return None
        key = session["exercise_key"]
        cfg = self.exercise_configs.get(key, {})
        direction = cfg.get("direction", "extend")
        patient = (
            self.session_manager.get_patient(session["patient_id"]) if session.get("patient_id") else None
        )
        history: List[Dict[str, Any]] = []
        suggestion = None
        if patient and cfg:
            all_sessions = self.session_manager.patient_sessions(patient["id"], key)
            prior = [s for s in all_sessions if s["id"] < session_id]
            history = progress_trend(prior[-5:], direction)
            upto = [s for s in all_sessions if s["id"] <= session_id]
            suggestion = suggest_target(upto, cfg, self._effective_target(patient, key))
        rep_details = dict(session.get("rep_details") or {})
        rep_details.pop("reps", None)  # keep the prompt compact
        return {
            "patient_profile": self._patient_profile(patient),
            "exercise": {
                "name": session["exercise_name"],
                "direction": direction,
                "target_angle": self._effective_target(patient, key) if cfg else None,
                "max_velocity_limit": cfg.get("max_angular_velocity_deg_per_sec"),
            },
            "session": {
                k: session.get(k)
                for k in (
                    "exercise_name", "side", "total_reps", "duration_seconds",
                    "min_angle", "max_angle", "consistency_score",
                )
            },
            "rep_analysis": rep_details,
            "previous_sessions": history,
            "target_suggestion": suggestion,
        }

    def build_patient_context(self, patient_id: int) -> Optional[Dict[str, Any]]:
        patient = self.session_manager.get_patient(patient_id)
        if patient is None:
            return None
        per_exercise = {}
        for key, cfg in self.exercise_configs.items():
            sessions = self.session_manager.patient_sessions(patient_id, key)
            if not sessions:
                continue
            direction = cfg.get("direction", "extend")
            target = self._effective_target(patient, key)
            per_exercise[key] = {
                "name": cfg["display_name"],
                "direction": direction,
                "current_target": target,
                "trend": progress_trend(sessions, direction),
                "target_suggestion": suggest_target(sessions, cfg, target),
            }
        return {"patient_profile": self._patient_profile(patient), "exercises": per_exercise}

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        while True:
            with self._lock:
                frame_bytes = self._latest_jpeg
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.033)
