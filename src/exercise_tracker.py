"""Phase 5: State-based exercise recognition and repetition counter with side switching & calibration."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RepUpdate:
    """Output of a single exercise tracker update step."""

    reps: int
    stage: str
    rep_completed: bool
    primary_angle: Optional[float]
    secondary_angle: Optional[float]
    left_angle: Optional[float]
    right_angle: Optional[float]
    peak_angle_this_rep: Optional[float]
    min_observed_angle: Optional[float]
    max_observed_angle: Optional[float]
    angular_velocity: float
    last_rep: Optional[Dict[str, Any]] = None


class ExerciseTracker:
    """State-machine repetition counter with configurable side (left/right/both), thresholds, and calibration."""

    def __init__(self, exercise_key: str, exercise_cfg: Dict[str, Any]) -> None:
        self.exercise_key = exercise_key
        self.cfg = copy.deepcopy(exercise_cfg)
        self.side: str = str(self.cfg.get("side", "right")).lower()
        self.calibrated_resting_angle: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        self.reps: int = 0
        self.stage: str = "ready"  # ready -> in_progress -> target_reached
        self._last_rep_ts: float = 0.0
        self._prev_angle: Optional[float] = None
        self._prev_ts: Optional[float] = None
        self._current_rep_peak: Optional[float] = None
        self.min_observed_angle: Optional[float] = None
        self.max_observed_angle: Optional[float] = None
        # Per-rep analytics (range of motion, tempo, symmetry, peak speed)
        self.rep_log: List[Dict[str, Any]] = []
        self._rep_start_ts: Optional[float] = None
        self._rep_min: Optional[float] = None
        self._rep_max: Optional[float] = None
        self._rep_left_peak: Optional[float] = None
        self._rep_right_peak: Optional[float] = None
        self._rep_max_velocity: float = 0.0

    def apply_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        """Apply physiotherapist-approved per-patient threshold overrides."""
        for key, value in (overrides or {}).items():
            if key in self.cfg and isinstance(value, (int, float)):
                self.cfg[key] = float(value)

    def _start_rep(self, now: float, angle: float) -> None:
        self._rep_start_ts = now
        self._rep_min = angle
        self._rep_max = angle
        self._rep_left_peak = None
        self._rep_right_peak = None
        self._rep_max_velocity = 0.0

    def _better_peak(self, current: Optional[float], value: Optional[float]) -> Optional[float]:
        if value is None:
            return current
        if current is None:
            return value
        if self.cfg.get("direction", "extend") == "extend":
            return max(current, value)
        return min(current, value)

    def _track_rep(
        self, angle: float, left: Optional[float], right: Optional[float], velocity: float
    ) -> None:
        if self._rep_start_ts is None:
            return
        self._rep_min = angle if self._rep_min is None else min(self._rep_min, angle)
        self._rep_max = angle if self._rep_max is None else max(self._rep_max, angle)
        self._rep_left_peak = self._better_peak(self._rep_left_peak, left)
        self._rep_right_peak = self._better_peak(self._rep_right_peak, right)
        self._rep_max_velocity = max(self._rep_max_velocity, velocity)

    def _finish_rep(self, now: float, peak: Optional[float]) -> Dict[str, Any]:
        rom = (
            (self._rep_max - self._rep_min)
            if self._rep_max is not None and self._rep_min is not None
            else 0.0
        )
        record = {
            "rep": self.reps,
            "peak_angle": round(peak, 1) if peak is not None else None,
            "rom_deg": round(rom, 1),
            "duration_s": round(now - self._rep_start_ts, 2) if self._rep_start_ts is not None else None,
            "left_peak": round(self._rep_left_peak, 1) if self._rep_left_peak is not None else None,
            "right_peak": round(self._rep_right_peak, 1) if self._rep_right_peak is not None else None,
            "max_velocity": round(self._rep_max_velocity, 1),
        }
        self.rep_log.append(record)
        return record

    def configure(
        self,
        exercise_key: str,
        exercise_cfg: Dict[str, Any],
        side: Optional[str] = None,
    ) -> None:
        self.exercise_key = exercise_key
        self.cfg = copy.deepcopy(exercise_cfg)
        self.side = (side or str(self.cfg.get("side", "right"))).lower()
        self.calibrated_resting_angle = None
        self.reset()

    def set_side(self, side: str) -> str:
        normalized = side.strip().lower()
        if normalized not in ("left", "right", "both"):
            raise ValueError(f"Unsupported side: {side}")
        self.side = normalized
        self.reset()
        return self.side

    def apply_calibration(self, resting_angle: float) -> Dict[str, float]:
        """Adjust start_angle_threshold around the patient's measured 3s resting angle."""
        direction = self.cfg.get("direction", "extend")
        target_th = float(self.cfg["target_angle_threshold"])
        margin = 12.0

        if direction == "extend":
            new_start = min(resting_angle + margin, target_th - 15.0)
        else:
            new_start = max(resting_angle - margin, target_th + 15.0)

        self.cfg["start_angle_threshold"] = round(new_start, 1)
        self.calibrated_resting_angle = round(resting_angle, 1)
        # Re-arm the state machine but keep reps/analytics so mid-session calibration is safe
        self.stage = "ready"
        self._current_rep_peak = None
        self._rep_start_ts = None
        return {
            "resting_angle": self.calibrated_resting_angle,
            "start_angle_threshold": self.cfg["start_angle_threshold"],
            "target_angle_threshold": target_th,
        }

    def _resolve_joint_angles(
        self, angles: Dict[str, Optional[float]], joint_field: str
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        joint = self.cfg.get(joint_field)
        if not joint:
            return None, None, None

        left_val = angles.get(f"left_{joint}")
        right_val = angles.get(f"right_{joint}")

        if self.side == "left":
            return left_val, left_val, right_val
        if self.side == "right":
            return right_val, left_val, right_val

        # side == "both": average both sides when both are visible, else use whichever is visible
        if left_val is not None and right_val is not None:
            return round((left_val + right_val) / 2.0, 1), left_val, right_val
        return (left_val if left_val is not None else right_val), left_val, right_val

    def update(
        self,
        angles: Dict[str, Optional[float]],
        timestamp: Optional[float] = None,
    ) -> RepUpdate:
        now = timestamp if timestamp is not None else time.time()
        primary_angle, left_angle, right_angle = self._resolve_joint_angles(
            angles, "primary_joint"
        )
        secondary_angle, _, _ = self._resolve_joint_angles(angles, "secondary_joint")

        angular_velocity = 0.0
        if primary_angle is not None:
            if self.min_observed_angle is None or primary_angle < self.min_observed_angle:
                self.min_observed_angle = primary_angle
            if self.max_observed_angle is None or primary_angle > self.max_observed_angle:
                self.max_observed_angle = primary_angle

            if self._prev_angle is not None and self._prev_ts is not None:
                dt = max(now - self._prev_ts, 1e-3)
                angular_velocity = abs(primary_angle - self._prev_angle) / dt

            self._prev_angle = primary_angle
            self._prev_ts = now

        rep_completed = False
        completed_peak = self._current_rep_peak
        last_rep: Optional[Dict[str, Any]] = None
        if primary_angle is not None and self.stage != "ready":
            self._track_rep(primary_angle, left_angle, right_angle, angular_velocity)

        if primary_angle is not None:
            direction = self.cfg.get("direction", "extend")
            start_th = float(self.cfg["start_angle_threshold"])
            target_th = float(self.cfg["target_angle_threshold"])
            cooldown = float(self.cfg.get("cooldown_seconds", 0.7))

            if direction == "extend":
                if self.stage == "ready" and primary_angle <= start_th:
                    self.stage = "in_progress"
                    self._current_rep_peak = primary_angle
                    self._start_rep(now, primary_angle)
                elif self.stage == "in_progress":
                    if self._current_rep_peak is None or primary_angle > self._current_rep_peak:
                        self._current_rep_peak = primary_angle
                    if primary_angle >= target_th:
                        self.stage = "target_reached"
                elif self.stage == "target_reached":
                    if self._current_rep_peak is None or primary_angle > self._current_rep_peak:
                        self._current_rep_peak = primary_angle
                    if primary_angle <= start_th and (now - self._last_rep_ts) >= cooldown:
                        self.reps += 1
                        self._last_rep_ts = now
                        rep_completed = True
                        completed_peak = self._current_rep_peak
                        last_rep = self._finish_rep(now, completed_peak)
                        self.stage = "in_progress"
                        self._current_rep_peak = primary_angle
                        self._start_rep(now, primary_angle)
            else:
                if self.stage == "ready" and primary_angle >= start_th:
                    self.stage = "in_progress"
                    self._current_rep_peak = primary_angle
                    self._start_rep(now, primary_angle)
                elif self.stage == "in_progress":
                    if self._current_rep_peak is None or primary_angle < self._current_rep_peak:
                        self._current_rep_peak = primary_angle
                    if primary_angle <= target_th:
                        self.stage = "target_reached"
                elif self.stage == "target_reached":
                    if self._current_rep_peak is None or primary_angle < self._current_rep_peak:
                        self._current_rep_peak = primary_angle
                    if primary_angle >= start_th and (now - self._last_rep_ts) >= cooldown:
                        self.reps += 1
                        self._last_rep_ts = now
                        rep_completed = True
                        completed_peak = self._current_rep_peak
                        last_rep = self._finish_rep(now, completed_peak)
                        self.stage = "in_progress"
                        self._current_rep_peak = primary_angle
                        self._start_rep(now, primary_angle)

        return RepUpdate(
            reps=self.reps,
            stage=self.stage,
            rep_completed=rep_completed,
            primary_angle=primary_angle,
            secondary_angle=secondary_angle,
            left_angle=left_angle,
            right_angle=right_angle,
            peak_angle_this_rep=completed_peak,
            min_observed_angle=self.min_observed_angle,
            max_observed_angle=self.max_observed_angle,
            angular_velocity=round(angular_velocity, 1),
            last_rep=last_rep,
        )
