"""Phase 7 & 10: Flask web server and REST API for the Rehabilitation Motion Evaluator."""

from __future__ import annotations

import argparse
import atexit
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request, send_file

from src.config import (
    CAMERA_SOURCE,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    POSE_BACKEND,
    parse_camera_source,
)
from src.llm_coach import LANGUAGE_NAMES, LLMCoach
from src.pipeline import RehabPipeline


def create_app(
    camera_source=CAMERA_SOURCE,
    pose_backend: str = POSE_BACKEND,
    start_pipeline: bool = True,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static",
    )

    pipeline = RehabPipeline(camera_source=camera_source, pose_backend=pose_backend)
    if start_pipeline:
        pipeline.start()
        atexit.register(pipeline.stop)

    app.config["PIPELINE"] = pipeline
    coach = LLMCoach()
    app.config["COACH"] = coach

    @app.route("/")
    def index():
        exercises = [
            {"key": k, "name": v["display_name"]}
            for k, v in pipeline.exercise_configs.items()
        ]
        return render_template(
            "index.html",
            exercises=exercises,
            initial_camera=str(pipeline.stream.source),
        )

    @app.route("/video_feed")
    def video_feed():
        return Response(
            pipeline.generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/telemetry")
    def api_telemetry():
        return jsonify(pipeline.get_telemetry())

    @app.route("/api/exercise", methods=["POST"])
    def api_select_exercise():
        payload = request.get_json(silent=True) or {}
        exercise_key = str(payload.get("exercise_key", "")).strip()
        if not pipeline.select_exercise(exercise_key):
            return jsonify({"error": f"Unknown exercise: {exercise_key}"}), 400
        return jsonify({"ok": True, "exercise_key": exercise_key})

    @app.route("/api/side", methods=["POST"])
    def api_set_side():
        payload = request.get_json(silent=True) or {}
        side = str(payload.get("side", "")).strip().lower()
        try:
            active_side = pipeline.set_side(side)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "side": active_side})

    @app.route("/api/calibrate", methods=["POST"])
    def api_calibrate():
        payload = request.get_json(silent=True) or {}
        duration_s = float(payload.get("duration_s", 3.0))
        duration_s = max(1.0, min(duration_s, 10.0))
        result = pipeline.start_calibration(duration_s=duration_s)
        return jsonify({"ok": True, **result})

    @app.route("/api/camera", methods=["POST"])
    def api_set_camera():
        payload = request.get_json(silent=True) or {}
        raw_source = str(payload.get("camera_source", "")).strip()
        if not raw_source:
            return jsonify({"error": "camera_source is required"}), 400
        active = pipeline.set_camera_source(raw_source)
        return jsonify({"ok": True, "camera_source": active})

    @app.route("/api/session/start", methods=["POST"])
    def api_start_session():
        payload = request.get_json(silent=True) or {}
        patient_label = str(payload.get("patient_label", "Anonymous"))
        result = pipeline.start_session(patient_label=patient_label)
        return jsonify({"ok": True, **result})

    @app.route("/api/session/stop", methods=["POST"])
    def api_stop_session():
        saved = pipeline.stop_session()
        if saved is None:
            return jsonify({"ok": False, "message": "No active session to stop"})
        return jsonify({"ok": True, "session": saved})

    @app.route("/api/sessions")
    def api_list_sessions():
        return jsonify({"sessions": pipeline.session_manager.list_sessions(limit=50)})

    @app.route("/api/sessions/<int:session_id>/frames")
    def api_session_frames(session_id: int):
        return jsonify(
            {
                "session_id": session_id,
                "frames": pipeline.session_manager.get_session_frames(session_id),
            }
        )

    @app.route("/api/sessions/<int:session_id>/frames.csv")
    def api_session_frames_csv(session_id: int):
        csv_path = pipeline.session_manager.export_session_frames_csv(session_id)
        return send_file(
            csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=csv_path.name,
        )

    @app.route("/api/sessions/export.csv")
    def api_export_csv():
        csv_path = pipeline.session_manager.export_csv()
        return send_file(
            csv_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=csv_path.name,
        )

    # ------------------------------------------------------------ patients
    @app.route("/api/patients", methods=["GET"])
    def api_list_patients():
        return jsonify({"patients": pipeline.session_manager.list_patients()})

    @app.route("/api/patients", methods=["POST"])
    def api_create_patient():
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip()
        if not name:
            return jsonify({"error": "name is required"}), 400
        age_raw = payload.get("age")
        try:
            age = int(age_raw) if age_raw not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify({"error": "age must be a number"}), 400
        side = str(payload.get("affected_side", "right")).lower()
        language = str(payload.get("language", "en")).lower()
        patient = pipeline.session_manager.create_patient(
            name=name,
            age=age,
            condition=str(payload.get("condition", "")).strip(),
            affected_side=side if side in ("left", "right", "both") else "right",
            language=language if language in LANGUAGE_NAMES else "en",
            notes=str(payload.get("notes", "")).strip(),
        )
        return jsonify({"ok": True, "patient": patient})

    @app.route("/api/patient/select", methods=["POST"])
    def api_select_patient():
        payload = request.get_json(silent=True) or {}
        pid = payload.get("patient_id")
        patient = pipeline.set_patient(int(pid) if pid else None)
        return jsonify({"ok": True, "patient": patient})

    @app.route("/api/patients/<int:patient_id>/progress")
    def api_patient_progress(patient_id: int):
        context = pipeline.build_patient_context(patient_id)
        if context is None:
            return jsonify({"error": "Unknown patient"}), 404
        return jsonify(context)

    @app.route("/api/patients/<int:patient_id>/targets", methods=["POST"])
    def api_approve_target(patient_id: int):
        """Physiotherapist approves a (suggested) per-patient target angle."""
        payload = request.get_json(silent=True) or {}
        exercise_key = str(payload.get("exercise_key", ""))
        if exercise_key not in pipeline.exercise_configs:
            return jsonify({"error": "Unknown exercise"}), 400
        try:
            target = float(payload["target_angle_threshold"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "target_angle_threshold is required"}), 400
        try:
            overrides = pipeline.session_manager.set_patient_overrides(
                patient_id, exercise_key, {"target_angle_threshold": round(target, 1)}
            )
        except KeyError:
            return jsonify({"error": "Unknown patient"}), 404
        pipeline.refresh_patient()
        return jsonify({"ok": True, "threshold_overrides": overrides})

    @app.route("/api/patients/<int:patient_id>/ask", methods=["POST"])
    def api_ask_about_patient(patient_id: int):
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()[:500]
        if not question:
            return jsonify({"error": "question is required"}), 400
        context = pipeline.build_patient_context(patient_id)
        if context is None:
            return jsonify({"error": "Unknown patient"}), 404
        return jsonify(coach.answer_question(question, context, str(payload.get("language", "en"))))

    # ------------------------------------------------------------ AI reports
    @app.route("/api/ai/status")
    def api_ai_status():
        return jsonify(coach.status())

    @app.route("/api/sessions/<int:session_id>/report", methods=["POST"])
    def api_session_report(session_id: int):
        payload = request.get_json(silent=True) or {}
        language = str(payload.get("language", "en")).lower()
        refresh = bool(payload.get("refresh", False))
        session = pipeline.session_manager.get_session(session_id)
        if session is None:
            return jsonify({"error": "Unknown session"}), 404
        cached = session.get("ai_report") or {}
        if not refresh and cached.get(language):
            return jsonify({**cached[language], "cached": True})
        context = pipeline.build_session_context(session_id)
        report = coach.session_report(context, language)
        report["target_suggestion"] = context.get("target_suggestion")
        report["exercise_key"] = session["exercise_key"]
        report["patient_id"] = session.get("patient_id")
        if report.get("source", "").startswith("gemini"):
            cached[language] = report
            pipeline.session_manager.save_ai_report(session_id, cached)
        return jsonify(report)

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient Rehabilitation Motion Evaluator — Edge AI Dashboard"
    )
    parser.add_argument(
        "--camera",
        default=str(CAMERA_SOURCE),
        help="Android IP Camera URL (e.g. http://192.168.1.100:8080/video), USB index (0), or 'demo'",
    )
    parser.add_argument(
        "--backend",
        default=POSE_BACKEND,
        choices=["auto", "mediapipe", "yolo", "demo"],
        help="Pose estimation backend",
    )
    parser.add_argument("--host", default=FLASK_HOST, help="Flask bind host")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help="Flask bind port")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    application = create_app(
        camera_source=parse_camera_source(args.camera),
        pose_backend=args.backend,
        start_pipeline=True,
    )
    application.run(
        host=args.host,
        port=args.port,
        debug=FLASK_DEBUG,
        threaded=True,
        use_reloader=False,
    )
