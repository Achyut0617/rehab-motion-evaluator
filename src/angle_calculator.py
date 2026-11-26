"""Phase 4: Joint angle calculation supporting 3D world landmarks (with 2D fallback) and EMA smoothing."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src.pose_estimator import Landmark

# Triplets (proximal, vertex/joint, distal)
JOINT_TRIPLETS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
}


def compute_angle_3pts(
    a: Tuple[float, ...],
    b: Tuple[float, ...],
    c: Tuple[float, ...],
) -> float:
    """Calculate the interior angle (0-180 degrees) at vertex b for 2D or 3D points."""
    pa = np.array(a, dtype=np.float64)
    pb = np.array(b, dtype=np.float64)
    pc = np.array(c, dtype=np.float64)

    ba = pa - pb
    bc = pc - pb

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class JointAngleCalculator:
    """Computes smoothed joint angles using 3D world landmarks when available, falling back to 2D."""

    def __init__(
        self,
        min_visibility: float = 0.55,
        smoothing_alpha: float = 0.35,
        use_3d: bool = True,
    ) -> None:
        self.min_visibility = min_visibility
        self.smoothing_alpha = smoothing_alpha
        self.use_3d = use_3d
        self.last_mode: str = "2D"
        self._ema_state: Dict[str, float] = {}

    def reset(self) -> None:
        self._ema_state.clear()

    def calculate_all(
        self, landmarks: Dict[str, Landmark]
    ) -> Dict[str, Optional[float]]:
        """Return dictionary of joint_name -> smoothed angle in degrees (or None if occluded)."""
        angles: Dict[str, Optional[float]] = {}
        used_3d_any = False

        for joint_name, (p1, p2, p3) in JOINT_TRIPLETS.items():
            if p1 not in landmarks or p2 not in landmarks or p3 not in landmarks:
                angles[joint_name] = None
                continue

            la, lb, lc = landmarks[p1], landmarks[p2], landmarks[p3]
            if min(la.visibility, lb.visibility, lc.visibility) < self.min_visibility:
                angles[joint_name] = None
                continue

            if self.use_3d and la.has_world_3d and lb.has_world_3d and lc.has_world_3d:
                raw_angle = compute_angle_3pts(
                    (la.wx, la.wy, la.wz),
                    (lb.wx, lb.wy, lb.wz),
                    (lc.wx, lc.wy, lc.wz),
                )
                used_3d_any = True
            else:
                raw_angle = compute_angle_3pts((la.x, la.y), (lb.x, lb.y), (lc.x, lc.y))

            prev = self._ema_state.get(joint_name)
            if prev is None:
                smoothed = raw_angle
            else:
                smoothed = self.smoothing_alpha * raw_angle + (1.0 - self.smoothing_alpha) * prev
            self._ema_state[joint_name] = smoothed
            angles[joint_name] = round(smoothed, 1)

        self.last_mode = "3D" if used_3d_any else "2D"
        return angles
