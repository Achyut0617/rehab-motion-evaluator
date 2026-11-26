"""Unit tests for 2D/3D angle math, side switching, calibration, per-frame telemetry, and tegrastats."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.angle_calculator import JointAngleCalculator, compute_angle_3pts
from src.config import load_exercise_configs
from src.evaluator import MovementEvaluator
from src.exercise_tracker import ExerciseTracker
from src.jetson_monitor import parse_tegrastats_line
from src.pose_estimator import Landmark
from src.session_manager import SessionManager


def test_compute_angle_2d_and_3d():
    assert pytest.approx(compute_angle_3pts((0, 1), (0, 0), (1, 0)), abs=0.1) == 90.0
    assert pytest.approx(compute_angle_3pts((-1, 0), (0, 0), (1, 0)), abs=0.1) == 180.0
    # 3D orthogonal vectors: (0,1,0) - (0,0,0) - (0,0,1) => 90 degrees
    assert pytest.approx(compute_angle_3pts((0, 1, 0), (0, 0, 0), (0, 0, 1)), abs=0.1) == 90.0


def test_joint_angle_calculator_prefers_3d_world_landmarks():
    calc = JointAngleCalculator(min_visibility=0.6, smoothing_alpha=1.0, use_3d=True)
    # 2D pixels look nearly collinear (angle ~180), but 3D world coords form a 90 deg angle
    landmarks = {
        "right_hip": Landmark(100, 100, 0.9, wx=0.0, wy=1.0, wz=0.0),
        "right_knee": Landmark(100, 200, 0.9, wx=0.0, wy=0.0, wz=0.0),
        "right_ankle": Landmark(100, 300, 0.9, wx=0.0, wy=0.0, wz=1.0),
    }
    angles = calc.calculate_all(landmarks)
    assert calc.last_mode == "3D"
    assert pytest.approx(angles["right_knee"], abs=0.2) == 90.0


def test_side_switching_and_calibration():
    cfgs = load_exercise_configs()
    tracker = ExerciseTracker("knee_extension", cfgs["knee_extension"])

    # Switch to left side
    tracker.set_side("left")
    upd = tracker.update({"left_knee": 98.0, "right_knee": 170.0}, timestamp=100.0)
    assert upd.primary_angle == 98.0

    # Switch to both (average)
    tracker.set_side("both")
    upd_both = tracker.update({"left_knee": 100.0, "right_knee": 120.0}, timestamp=101.0)
    assert upd_both.primary_angle == 110.0

    # Calibrate resting pose at 90 degrees -> start threshold adjusted to 102.0
    cal = tracker.apply_calibration(90.0)
    assert cal["resting_angle"] == 90.0
    assert cal["start_angle_threshold"] == 102.0


def test_knee_extension_rep_counting_and_cooldown():
    cfgs = load_exercise_configs()
    tracker = ExerciseTracker("knee_extension", cfgs["knee_extension"])
    evaluator = MovementEvaluator(cfgs["knee_extension"])

    sequence = [95, 130, 170, 130, 95, 135, 168, 125, 92]
    t = 1000.0
    for angle in sequence:
        upd = tracker.update({"right_knee": float(angle)}, timestamp=t)
        res = evaluator.evaluate(upd)
        assert res.feedback
        t += 0.5

    assert tracker.reps == 2
    assert tracker.min_observed_angle == 92.0
    assert tracker.max_observed_angle == 170.0
    assert evaluator.consistency_score > 70.0


def test_session_manager_per_frame_sqlite_and_csv(tmp_path: Path):
    mgr = SessionManager(
        db_path=tmp_path / "test.db",
        export_dir=tmp_path / "exports",
    )
    sample_frames = [
        {
            "timestamp_utc": "2026-11-22 10:00:00.100",
            "elapsed_s": 0.1,
            "stage": "in_progress",
            "primary_angle": 95.0,
            "secondary_angle": None,
            "left_angle": 94.0,
            "right_angle": 95.0,
            "angular_velocity": 12.5,
            "reps": 0,
            "consistency_score": 88.0,
            "angle_mode": "3D",
            "feedback": "Try extending your knee further",
        },
        {
            "timestamp_utc": "2026-11-22 10:00:01.100",
            "elapsed_s": 1.1,
            "stage": "target_reached",
            "primary_angle": 168.0,
            "secondary_angle": None,
            "left_angle": 166.0,
            "right_angle": 168.0,
            "angular_velocity": 40.0,
            "reps": 1,
            "consistency_score": 95.0,
            "angle_mode": "3D",
            "feedback": "Movement completed",
        },
    ]
    saved = mgr.save_session(
        exercise_key="knee_extension",
        exercise_name="Knee Extension",
        total_reps=1,
        duration_seconds=5.0,
        min_angle=95.0,
        max_angle=168.0,
        consistency_score=91.5,
        evaluation_summary="Movement completed",
        patient_label="Test Patient",
        side="right",
        frames=sample_frames,
    )
    assert saved["id"] == 1
    assert saved["frame_count"] == 2

    frames = mgr.get_session_frames(1)
    assert len(frames) == 2
    assert frames[1]["angle_mode"] == "3D"

    frame_csv = tmp_path / "exports" / "session_1_frames.csv"
    assert frame_csv.exists()
    content = frame_csv.read_text(encoding="utf-8")
    assert "angular_velocity" in content
    assert "168.0" in content


def test_parse_tegrastats_line():
    sample = (
        "RAM 3210/7620MB (lfb 12x4MB) CPU [24%@1510,18%@1510,30%@1510,28%@1510] "
        "GR3D_FREQ 62% cpu@49.5C gpu@47.2C tj@51.0C VDD_IN 8450mW/8450mW"
    )
    parsed = parse_tegrastats_line(sample)
    assert parsed["ram_used_mb"] == 3210
    assert parsed["ram_total_mb"] == 7620
    assert parsed["cpu_percent"] == 25.0
    assert parsed["gpu_percent"] == 62
    assert parsed["temp_c"] == 51.0
    assert parsed["power_w"] == 8.45


# ---------------------------------------------------------------- personalization & AI layer
from src.llm_coach import LLMCoach, offline_report  # noqa: E402
from src.progress import suggest_target, summarize_reps  # noqa: E402


def test_per_rep_analytics_and_symmetry():
    cfgs = load_exercise_configs()
    tracker = ExerciseTracker("knee_extension", cfgs["knee_extension"])
    tracker.set_side("both")
    t = 0.0
    for left, right in [(95, 95), (130, 120), (170, 150), (130, 120), (95, 95), (172, 150), (100, 100)]:
        tracker.update({"left_knee": float(left), "right_knee": float(right)}, timestamp=t)
        t += 1.0
    assert tracker.reps == 2
    first = tracker.rep_log[0]
    assert first["left_peak"] == 170.0 and first["right_peak"] == 150.0
    assert first["rom_deg"] > 0 and first["duration_s"] > 0
    summary = summarize_reps(tracker.rep_log, "extend")
    assert summary["rep_count"] == 2
    assert summary["symmetry_gap_deg"] == pytest.approx(21.0, abs=0.1)


def test_calibration_keeps_reps_and_overrides_apply():
    cfgs = load_exercise_configs()
    tracker = ExerciseTracker("knee_extension", cfgs["knee_extension"])
    for i, a in enumerate([95, 170, 95]):
        tracker.update({"right_knee": float(a)}, timestamp=float(i))
    assert tracker.reps == 1
    tracker.apply_calibration(90.0)
    assert tracker.reps == 1
    tracker.apply_overrides({"target_angle_threshold": 150})
    assert tracker.cfg["target_angle_threshold"] == 150.0


def test_evaluator_in_range_message_is_encouraging():
    cfgs = load_exercise_configs()
    tracker = ExerciseTracker("knee_extension", cfgs["knee_extension"])
    ev = MovementEvaluator(cfgs["knee_extension"])
    tracker.update({"right_knee": 100.0}, timestamp=0.0)
    res = ev.evaluate(tracker.update({"right_knee": 120.0}, timestamp=1.0))
    assert res.feedback_key == "keep_going"


def test_adaptive_target_suggestion():
    cfg = load_exercise_configs()["knee_extension"]
    good = [{"total_reps": 10, "max_angle": 170.0, "consistency_score": 90.0} for _ in range(3)]
    sug = suggest_target(good, cfg)
    assert sug["action"] == "increase_difficulty" and sug["suggested_target"] == 165.0
    weak = [{"total_reps": 5, "max_angle": 150.0, "consistency_score": 70.0} for _ in range(3)]
    assert suggest_target(weak, cfg)["action"] == "decrease_difficulty"
    assert suggest_target(good[:2], cfg)["action"] == "keep"


def test_patients_and_session_link(tmp_path: Path):
    mgr = SessionManager(db_path=tmp_path / "p.db", export_dir=tmp_path / "ex")
    p = mgr.create_patient("Ram", age=67, condition="TKR", affected_side="left", language="hi")
    saved = mgr.save_session(
        exercise_key="knee_extension", exercise_name="Knee Extension", total_reps=4,
        duration_seconds=30, min_angle=90, max_angle=165, consistency_score=88,
        evaluation_summary="ok", patient_id=p["id"], rep_details={"rep_count": 4, "best_peak_angle": 165},
    )
    assert mgr.patient_sessions(p["id"])[0]["rep_details"]["best_peak_angle"] == 165
    mgr.set_patient_overrides(p["id"], "knee_extension", {"target_angle_threshold": 165})
    assert mgr.get_patient(p["id"])["threshold_overrides"]["knee_extension"]["target_angle_threshold"] == 165
    mgr.save_ai_report(saved["id"], {"hi": {"patient_summary": "बढ़िया"}})
    assert mgr.get_session(saved["id"])["ai_report"]["hi"]["patient_summary"] == "बढ़िया"
    assert mgr.export_csv().exists()


def test_llm_coach_falls_back_offline_without_key():
    coach = LLMCoach(api_key="")
    ctx = {
        "exercise": {"direction": "extend", "max_velocity_limit": 180},
        "session": {"exercise_name": "Knee Extension", "total_reps": 5, "duration_seconds": 40,
                    "max_angle": 160, "consistency_score": 85},
        "rep_analysis": {"best_peak_angle": 162, "max_velocity": 90},
        "previous_sessions": [{"best_peak_angle": 150}],
    }
    for lang in ("en", "hi", "mr"):
        r = coach.session_report(ctx, lang)
        assert r["source"] == "offline-template" and r["patient_summary"]
    assert "पिछले" in offline_report(ctx, "hi")["patient_summary"]
