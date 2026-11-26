"""Centralized configuration loader for the Rehabilitation Motion Evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
EXERCISES_CONFIG_PATH = CONFIG_DIR / "exercises.json"
DB_PATH = DATA_DIR / "rehab_sessions.db"
CSV_EXPORT_DIR = DATA_DIR / "exports"


def parse_camera_source(raw: str) -> Union[int, str]:
    """Parse CAMERA_SOURCE into an integer USB index or URL/demo string."""
    value = raw.strip()
    if value.isdigit():
        return int(value)
    return value


CAMERA_SOURCE: Union[int, str] = parse_camera_source(
    os.getenv("CAMERA_SOURCE", "demo")
)
FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "480"))
POSE_BACKEND: str = os.getenv("POSE_BACKEND", "auto").strip().lower()
MEDIAPIPE_MODEL_PATH: str = os.getenv("MEDIAPIPE_MODEL_PATH", "").strip()
YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolov8n-pose.pt").strip()

FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"


def load_exercise_configs(path: Path = EXERCISES_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    """Load physiotherapist-configurable exercise thresholds from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {k: v for k, v in raw.items() if not k.startswith("_")}
