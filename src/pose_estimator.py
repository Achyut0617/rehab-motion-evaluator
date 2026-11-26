"""Phase 3: Pluggable human pose estimation (MediaPipe 3D/2D, YOLOv8-Pose TensorRT, Demo mode)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import BASE_DIR

# Standardized 12-landmark anatomical subset used across all backends
LANDMARK_NAMES: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# MediaPipe 33-landmark indices
MP_LANDMARK_INDEX: Dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# COCO 17-keypoint indices (used by YOLOv8-Pose / TensorRT fallback)
COCO_KEYPOINT_INDEX: Dict[str, int] = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass
class Landmark:
    """2D pixel coordinate, confidence visibility, and optional 3D world coordinates (meters)."""

    x: float
    y: float
    visibility: float
    wx: Optional[float] = None
    wy: Optional[float] = None
    wz: Optional[float] = None

    @property
    def has_world_3d(self) -> bool:
        return self.wx is not None and self.wy is not None and self.wz is not None


class PoseEstimator:
    """Unified pose estimation wrapper with MediaPipe (3D world landmarks), YOLOv8-Pose TensorRT, and Demo backends."""

    def __init__(
        self,
        backend: str = "auto",
        mediapipe_model_path: str = "",
        yolo_model_path: str = "yolov8n-pose.pt",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.requested_backend = backend.lower()
        self.active_backend = "demo"
        self._mp_pose = None
        self._mp_landmarker = None
        self._yolo_model = None
        self._yolo_is_engine = False
        self._demo_start_time = time.time()
        self._demo_exercise = "knee_extension"

        if self.requested_backend == "yolo":
            if self._init_yolo(yolo_model_path):
                return
        elif self.requested_backend == "auto":
            engine_file = BASE_DIR / "models" / "yolov8n-pose.engine"
            if engine_file.exists() and self._init_yolo(str(engine_file)):
                return
            if self._init_mediapipe(
                mediapipe_model_path,
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence,
            ):
                return
            if self._init_yolo(yolo_model_path):
                return
        elif self.requested_backend == "mediapipe":
            if self._init_mediapipe(
                mediapipe_model_path,
                model_complexity,
                min_detection_confidence,
                min_tracking_confidence,
            ):
                return

        self.active_backend = "demo"

    def _init_mediapipe(
        self,
        model_path: str,
        model_complexity: int,
        min_det: float,
        min_trk: float,
    ) -> bool:
        try:
            import mediapipe as mp
        except ImportError:
            return False

        if model_path:
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                options = mp_vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=min_det,
                    min_tracking_confidence=min_trk,
                )
                self._mp_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                self.active_backend = "mediapipe_tasks"
                return True
            except Exception:
                self._mp_landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=min_det,
                min_tracking_confidence=min_trk,
            )
            self.active_backend = "mediapipe"
            return True

        return False

    def _resolve_yolo_path(self, model_path: str) -> str:
        """Prefer an exported FP16 TensorRT .engine file in models/ when available."""
        candidate_engine = BASE_DIR / "models" / "yolov8n-pose.engine"
        if model_path == "yolov8n-pose.pt" and candidate_engine.exists():
            return str(candidate_engine)
        rel = BASE_DIR / model_path
        if rel.exists():
            return str(rel)
        return model_path

    def _init_yolo(self, model_path: str) -> bool:
        try:
            from ultralytics import YOLO

            resolved = self._resolve_yolo_path(model_path)
            self._yolo_is_engine = resolved.endswith(".engine")
            self._yolo_model = YOLO(resolved, task="pose")
            self.active_backend = "yolo (TensorRT)" if self._yolo_is_engine else "yolo (PyTorch)"
            return True
        except Exception:
            self._yolo_model = None
            return False

    def set_demo_exercise(self, exercise_name: str) -> None:
        self._demo_exercise = exercise_name

    def detect(self, frame: np.ndarray, force_demo: bool = False) -> Dict[str, Landmark]:
        """Run pose estimation on a BGR OpenCV frame and return named landmarks."""
        h, w = frame.shape[:2]
        if force_demo or self.active_backend == "demo":
            return self._detect_demo(w, h)
        if self.active_backend == "mediapipe" and self._mp_pose is not None:
            return self._detect_mediapipe_solutions(frame, w, h)
        if self.active_backend == "mediapipe_tasks" and self._mp_landmarker is not None:
            return self._detect_mediapipe_tasks(frame, w, h)
        if self.active_backend.startswith("yolo") and self._yolo_model is not None:
            return self._detect_yolo(frame)
        return {}

    def _detect_mediapipe_solutions(
        self, frame: np.ndarray, w: int, h: int
    ) -> Dict[str, Landmark]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mp_pose.process(rgb)
        if not result.pose_landmarks:
            return {}
        lm_list = result.pose_landmarks.landmark
        world_list = (
            result.pose_world_landmarks.landmark
            if getattr(result, "pose_world_landmarks", None) is not None
            else None
        )
        out: Dict[str, Landmark] = {}
        for name, idx in MP_LANDMARK_INDEX.items():
            lm = lm_list[idx]
            wlm = world_list[idx] if world_list is not None else None
            out[name] = Landmark(
                x=lm.x * w,
                y=lm.y * h,
                visibility=float(getattr(lm, "visibility", 1.0)),
                wx=float(wlm.x) if wlm is not None else None,
                wy=float(wlm.y) if wlm is not None else None,
                wz=float(wlm.z) if wlm is not None else None,
            )
        return out

    def _detect_mediapipe_tasks(
        self, frame: np.ndarray, w: int, h: int
    ) -> Dict[str, Landmark]:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_landmarker.detect(mp_img)
        if not result.pose_landmarks:
            return {}
        lm_list = result.pose_landmarks[0]
        world_list = (
            result.pose_world_landmarks[0]
            if getattr(result, "pose_world_landmarks", None)
            and len(result.pose_world_landmarks) > 0
            else None
        )
        out: Dict[str, Landmark] = {}
        for name, idx in MP_LANDMARK_INDEX.items():
            lm = lm_list[idx]
            wlm = world_list[idx] if world_list is not None else None
            out[name] = Landmark(
                x=lm.x * w,
                y=lm.y * h,
                visibility=float(getattr(lm, "visibility", 1.0)),
                wx=float(wlm.x) if wlm is not None else None,
                wy=float(wlm.y) if wlm is not None else None,
                wz=float(wlm.z) if wlm is not None else None,
            )
        return out

    def _detect_yolo(self, frame: np.ndarray) -> Dict[str, Landmark]:
        predict_kwargs = {"verbose": False, "max_det": 1}
        if self._yolo_is_engine:
            predict_kwargs["half"] = True
        results = self._yolo_model.predict(frame, **predict_kwargs)
        if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return {}
        xy = results[0].keypoints.xy[0].cpu().numpy()
        conf = (
            results[0].keypoints.conf[0].cpu().numpy()
            if results[0].keypoints.conf is not None
            else np.ones(len(xy), dtype=float)
        )
        landmarks: Dict[str, Landmark] = {}
        for name, idx in COCO_KEYPOINT_INDEX.items():
            if idx < len(xy):
                landmarks[name] = Landmark(
                    x=float(xy[idx][0]),
                    y=float(xy[idx][1]),
                    visibility=float(conf[idx]),
                )
        return landmarks

    def _detect_demo(self, w: int, h: int) -> Dict[str, Landmark]:
        """Generate a smooth animated skeleton (with 3D world coordinates) for testing."""
        t = time.time() - self._demo_start_time
        phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / 3.6))

        cx, cy = w * 0.5, h * 0.45
        scale = min(w, h) / 480.0

        l_sh = (cx - 42 * scale, cy - 110 * scale)
        r_sh = (cx + 42 * scale, cy - 110 * scale)
        l_hip = (cx - 32 * scale, cy + 10 * scale)
        r_hip = (cx + 32 * scale, cy + 10 * scale)

        l_elb = (l_sh[0] - 18 * scale, l_sh[1] + 68 * scale)
        l_wri = (l_elb[0] - 8 * scale, l_elb[1] + 62 * scale)

        r_elb = (r_sh[0] + 18 * scale, r_sh[1] + 68 * scale)
        r_wri = (r_elb[0] + 8 * scale, r_elb[1] + 62 * scale)

        l_knee = (l_hip[0] - 10 * scale, l_hip[1] + 95 * scale)
        l_ank = (l_knee[0] - 6 * scale, l_knee[1] + 95 * scale)

        r_knee = (r_hip[0] + 10 * scale, r_hip[1] + 95 * scale)
        r_ank = (r_knee[0] + 6 * scale, r_knee[1] + 95 * scale)

        if self._demo_exercise == "bicep_curl":
            angle_deg = 165.0 - phase * 123.0
            rad = math.radians(180.0 - angle_deg)
            forearm = 64 * scale
            r_wri = (
                r_elb[0] + forearm * math.sin(rad),
                r_elb[1] + forearm * math.cos(rad),
            )
            l_wri = (
                l_elb[0] - forearm * math.sin(rad),
                l_elb[1] + forearm * math.cos(rad),
            )
        elif self._demo_exercise == "knee_extension":
            r_knee = (r_hip[0] + 88 * scale, r_hip[1] + 6 * scale)
            l_knee = (l_hip[0] - 88 * scale, l_hip[1] + 6 * scale)
            angle_deg = 92.0 + phase * 80.0
            rad = math.radians(180.0 - angle_deg)
            shin = 92 * scale
            r_ank = (
                r_knee[0] + shin * math.cos(rad),
                r_knee[1] + shin * math.sin(rad),
            )
            l_ank = (
                l_knee[0] - shin * math.cos(rad),
                l_knee[1] + shin * math.sin(rad),
            )
        elif self._demo_exercise == "sit_to_stand":
            drop = (1.0 - phase) * 75 * scale
            r_sh = (cx + 15 * scale + (1.0 - phase) * 28 * scale, cy - 110 * scale + drop)
            l_sh = (r_sh[0] - 32 * scale, r_sh[1])
            r_hip = (cx + 10 * scale - (1.0 - phase) * 35 * scale, cy + 10 * scale + drop)
            l_hip = (r_hip[0] - 28 * scale, r_hip[1])
            r_knee = (cx + 35 * scale, cy + 102 * scale + drop * 0.25)
            l_knee = (r_knee[0] - 28 * scale, r_knee[1])
            r_ank = (cx + 22 * scale, cy + 195 * scale)
            l_ank = (r_ank[0] - 28 * scale, r_ank[1])

        pts = {
            "left_shoulder": l_sh,
            "right_shoulder": r_sh,
            "left_elbow": l_elb,
            "right_elbow": r_elb,
            "left_wrist": l_wri,
            "right_wrist": r_wri,
            "left_hip": l_hip,
            "right_hip": r_hip,
            "left_knee": l_knee,
            "right_knee": r_knee,
            "left_ankle": l_ank,
            "right_ankle": r_ank,
        }
        # Map pixel coordinates to metric-scale 3D world coords (z=0 plane in demo)
        px_to_m = 1.0 / (220.0 * scale)
        return {
            name: Landmark(
                x=p[0],
                y=p[1],
                visibility=0.98,
                wx=(p[0] - cx) * px_to_m,
                wy=(p[1] - cy) * px_to_m,
                wz=0.0,
            )
            for name, p in pts.items()
        }

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        landmarks: Dict[str, Landmark],
        min_visibility: float = 0.5,
    ) -> np.ndarray:
        """Draw skeletal connections and joint points on the frame."""
        for a, b in SKELETON_CONNECTIONS:
            if a in landmarks and b in landmarks:
                la, lb = landmarks[a], landmarks[b]
                if la.visibility >= min_visibility and lb.visibility >= min_visibility:
                    cv2.line(
                        frame,
                        (int(la.x), int(la.y)),
                        (int(lb.x), int(lb.y)),
                        (56, 189, 248),
                        3,
                        cv2.LINE_AA,
                    )

        for lm in landmarks.values():
            if lm.visibility >= min_visibility:
                pt = (int(lm.x), int(lm.y))
                cv2.circle(frame, pt, 6, (15, 23, 42), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, (34, 197, 94), -1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        if self._mp_pose is not None:
            self._mp_pose.close()
        if self._mp_landmarker is not None:
            self._mp_landmarker.close()
