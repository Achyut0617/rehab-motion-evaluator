"""Phase 8: Local SQLite session & per-frame telemetry storage with CSV export (no raw video stored)."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CSV_EXPORT_DIR, DB_PATH


class SessionManager:
    """Stores physiotherapy session summaries and per-frame telemetry in SQLite and CSV."""

    def __init__(self, db_path: Path = DB_PATH, export_dir: Path = CSV_EXPORT_DIR) -> None:
        self.db_path = Path(db_path)
        self.export_dir = Path(export_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    patient_label TEXT DEFAULT 'Anonymous',
                    exercise_key TEXT NOT NULL,
                    exercise_name TEXT NOT NULL,
                    side TEXT DEFAULT 'right',
                    total_reps INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    min_angle REAL,
                    max_angle REAL,
                    consistency_score REAL NOT NULL,
                    frame_count INTEGER DEFAULT 0,
                    evaluation_summary TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    elapsed_s REAL NOT NULL,
                    stage TEXT NOT NULL,
                    primary_angle REAL,
                    secondary_angle REAL,
                    left_angle REAL,
                    right_angle REAL,
                    angular_velocity REAL NOT NULL,
                    reps INTEGER NOT NULL,
                    consistency_score REAL NOT NULL,
                    angle_mode TEXT DEFAULT '2D',
                    feedback TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_frames_sid ON session_frames(session_id)"
            )
            # Lightweight schema migration if upgrading from an older DB
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "side" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN side TEXT DEFAULT 'right'")
            if "frame_count" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN frame_count INTEGER DEFAULT 0")
            if "patient_id" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN patient_id INTEGER")
            if "rep_details" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN rep_details TEXT")
            if "ai_report" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN ai_report TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    age INTEGER,
                    condition TEXT DEFAULT '',
                    affected_side TEXT DEFAULT 'right',
                    language TEXT DEFAULT 'en',
                    notes TEXT DEFAULT '',
                    threshold_overrides TEXT DEFAULT '{}'
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ patients
    @staticmethod
    def _patient_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["threshold_overrides"] = json.loads(d.get("threshold_overrides") or "{}")
        except json.JSONDecodeError:
            d["threshold_overrides"] = {}
        return d

    def create_patient(
        self,
        name: str,
        age: Optional[int] = None,
        condition: str = "",
        affected_side: str = "right",
        language: str = "en",
        notes: str = "",
    ) -> Dict[str, Any]:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO patients (created_at, name, age, condition, affected_side, language, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (created_at, name.strip() or "Anonymous", age, condition, affected_side, language, notes),
            )
            pid = int(cur.lastrowid)
            conn.commit()
        return self.get_patient(pid)  # type: ignore[return-value]

    def get_patient(self, patient_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return self._patient_row(row) if row else None

    def list_patients(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM patients ORDER BY name COLLATE NOCASE").fetchall()
        return [self._patient_row(r) for r in rows]

    def set_patient_overrides(
        self, patient_id: int, exercise_key: str, overrides: Dict[str, float]
    ) -> Dict[str, Any]:
        patient = self.get_patient(patient_id)
        if patient is None:
            raise KeyError(patient_id)
        all_overrides = patient["threshold_overrides"]
        all_overrides.setdefault(exercise_key, {}).update(overrides)
        with self._connect() as conn:
            conn.execute(
                "UPDATE patients SET threshold_overrides = ? WHERE id = ?",
                (json.dumps(all_overrides), patient_id),
            )
            conn.commit()
        return all_overrides

    def patient_sessions(
        self, patient_id: int, exercise_key: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Chronological (oldest first) sessions for one patient, optionally one exercise."""
        query = "SELECT * FROM sessions WHERE patient_id = ?"
        params: list = [patient_id]
        if exercise_key:
            query += " AND exercise_key = ?"
            params.append(exercise_key)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._session_row(r) for r in reversed(rows)]

    # ------------------------------------------------------------------ sessions
    @staticmethod
    def _session_row(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for key in ("rep_details", "ai_report"):
            raw = d.get(key)
            try:
                d[key] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                d[key] = None
        return d

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session_row(row) if row else None

    def save_ai_report(self, session_id: int, report: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ai_report = ? WHERE id = ?",
                (json.dumps(report, ensure_ascii=False), session_id),
            )
            conn.commit()

    def save_session(
        self,
        exercise_key: str,
        exercise_name: str,
        total_reps: int,
        duration_seconds: float,
        min_angle: Optional[float],
        max_angle: Optional[float],
        consistency_score: float,
        evaluation_summary: str,
        patient_label: str = "Anonymous",
        side: str = "right",
        frames: Optional[List[Dict[str, Any]]] = None,
        patient_id: Optional[int] = None,
        rep_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        frame_list = frames or []

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    created_at, patient_label, exercise_key, exercise_name, side,
                    total_reps, duration_seconds, min_angle, max_angle,
                    consistency_score, frame_count, evaluation_summary,
                    patient_id, rep_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    patient_label.strip() or "Anonymous",
                    exercise_key,
                    exercise_name,
                    side,
                    int(total_reps),
                    round(float(duration_seconds), 1),
                    round(float(min_angle), 1) if min_angle is not None else None,
                    round(float(max_angle), 1) if max_angle is not None else None,
                    round(float(consistency_score), 1),
                    len(frame_list),
                    evaluation_summary,
                    patient_id,
                    json.dumps(rep_details) if rep_details is not None else None,
                ),
            )
            session_id = int(cursor.lastrowid)

            if frame_list:
                conn.executemany(
                    """
                    INSERT INTO session_frames (
                        session_id, timestamp_utc, elapsed_s, stage,
                        primary_angle, secondary_angle, left_angle, right_angle,
                        angular_velocity, reps, consistency_score, angle_mode, feedback
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            f.get("timestamp_utc", created_at),
                            round(float(f.get("elapsed_s", 0.0)), 3),
                            str(f.get("stage", "ready")),
                            f.get("primary_angle"),
                            f.get("secondary_angle"),
                            f.get("left_angle"),
                            f.get("right_angle"),
                            round(float(f.get("angular_velocity", 0.0)), 1),
                            int(f.get("reps", 0)),
                            round(float(f.get("consistency_score", 0.0)), 1),
                            str(f.get("angle_mode", "2D")),
                            str(f.get("feedback", "")),
                        )
                        for f in frame_list
                    ],
                )
            conn.commit()

        # Also persist a per-session frame telemetry CSV for easy clinical review
        if frame_list:
            self.export_session_frames_csv(session_id)

        return {
            "id": session_id,
            "created_at": created_at,
            "patient_label": patient_label.strip() or "Anonymous",
            "exercise_key": exercise_key,
            "exercise_name": exercise_name,
            "side": side,
            "total_reps": int(total_reps),
            "duration_seconds": round(float(duration_seconds), 1),
            "min_angle": round(float(min_angle), 1) if min_angle is not None else None,
            "max_angle": round(float(max_angle), 1) if max_angle is not None else None,
            "consistency_score": round(float(consistency_score), 1),
            "frame_count": len(frame_list),
            "evaluation_summary": evaluation_summary,
            "patient_id": patient_id,
            "rep_details": rep_details,
        }

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._session_row(r) for r in rows]

    def get_session_frames(self, session_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_frames WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def export_session_frames_csv(self, session_id: int) -> Path:
        frames = self.get_session_frames(session_id)
        out_path = self.export_dir / f"session_{session_id}_frames.csv"
        fieldnames = [
            "id",
            "session_id",
            "timestamp_utc",
            "elapsed_s",
            "stage",
            "primary_angle",
            "secondary_angle",
            "left_angle",
            "right_angle",
            "angular_velocity",
            "reps",
            "consistency_score",
            "angle_mode",
            "feedback",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in frames:
                writer.writerow(row)
        return out_path

    def export_csv(self, filename: Optional[str] = None) -> Path:
        sessions = self.list_sessions(limit=500)
        if not filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rehab_sessions_{stamp}.csv"
        out_path = self.export_dir / filename
        fieldnames = [
            "id",
            "created_at",
            "patient_label",
            "exercise_key",
            "exercise_name",
            "side",
            "total_reps",
            "duration_seconds",
            "min_angle",
            "max_angle",
            "consistency_score",
            "frame_count",
            "evaluation_summary",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in sessions:
                writer.writerow(row)
        return out_path
