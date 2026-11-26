"""Phase 6: Rule-based movement evaluation, real-time feedback, and consistency scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.exercise_tracker import RepUpdate


@dataclass
class EvaluationResult:
    """Real-time movement evaluation snapshot."""

    status: str  # "good" | "warning" | "caution" | "idle"
    feedback: str
    consistency_score: float
    in_range: bool
    feedback_key: str = "ready"


class MovementEvaluator:
    """Evaluates movement range, speed, posture, and visibility against configurable rules."""

    def __init__(self, exercise_cfg: Dict[str, Any]) -> None:
        self.cfg = exercise_cfg
        self.reset()

    def configure(self, exercise_cfg: Dict[str, Any]) -> None:
        self.cfg = exercise_cfg
        self.reset()

    def reset(self) -> None:
        self._frame_scores: List[float] = []
        self._last_feedback: str = self.cfg.get("feedback", {}).get(
            "ready", "Ready to begin"
        )
        self._last_status: str = "idle"
        self._last_key: str = "ready"

    @property
    def consistency_score(self) -> float:
        if not self._frame_scores:
            return 0.0
        return round(sum(self._frame_scores) / len(self._frame_scores), 1)

    def evaluate(self, update: RepUpdate) -> EvaluationResult:
        fb = self.cfg.get("feedback", {})
        if update.primary_angle is None:
            self._last_status = "caution"
            self._last_feedback = fb.get(
                "low_visibility", "Ensure key joints are clearly visible"
            )
            return EvaluationResult(
                status=self._last_status,
                feedback=self._last_feedback,
                consistency_score=self.consistency_score,
                in_range=False,
                feedback_key="low_visibility",
            )

        max_vel = float(self.cfg.get("max_angular_velocity_deg_per_sec", 180.0))
        ideal_min = float(self.cfg.get("ideal_min_angle", 40.0))
        ideal_max = float(self.cfg.get("ideal_max_angle", 175.0))
        in_range = ideal_min <= update.primary_angle <= ideal_max

        frame_score = 100.0
        status = "good"
        message = fb.get("ready", "Continue movement")
        key = "ready"

        # 1. Rep just completed cleanly
        if update.rep_completed:
            status = "good"
            message = fb.get("completed", "Movement completed")
            key = "completed"
            frame_score = 100.0
        # 2. Movement speed check
        elif update.angular_velocity > max_vel:
            status = "warning"
            message = fb.get("too_fast", "Move slowly")
            key = "too_fast"
            frame_score = 65.0
        # 3. Secondary posture check (e.g., hip extension during sit-to-stand)
        elif (
            update.stage == "target_reached"
            and update.secondary_angle is not None
            and "secondary_target_threshold" in self.cfg
            and update.secondary_angle < float(self.cfg["secondary_target_threshold"])
        ):
            status = "warning"
            message = fb.get("posture_warning", "Maintain your posture")
            key = "posture_warning"
            frame_score = 72.0
        # 4. Target reached
        elif update.stage == "target_reached":
            status = "good"
            message = fb.get("target_reached", "Target reached — return slowly")
            key = "target_reached"
            frame_score = 95.0
        # 5. Mid-movement range guidance
        elif update.stage == "in_progress":
            status = "good" if in_range else "warning"
            if not in_range:
                message = fb.get("insufficient_range", "Adjust your range of motion")
                key = "insufficient_range"
                frame_score = 75.0
            else:
                message = fb.get("keep_going", "Keep moving smoothly toward target")
                key = "keep_going"
                frame_score = 88.0

        self._frame_scores.append(frame_score)
        # Keep rolling history bounded to last 3000 frames (~100s at 30 FPS)
        if len(self._frame_scores) > 3000:
            self._frame_scores.pop(0)

        self._last_status = status
        self._last_feedback = message
        self._last_key = key
        return EvaluationResult(
            status=status,
            feedback=message,
            consistency_score=self.consistency_score,
            in_range=in_range,
            feedback_key=key,
        )
