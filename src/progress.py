"""Per-rep analytics, cross-session progress trends, and adaptive target suggestions.

Everything here is deterministic and offline. The LLM layer (src/llm_coach.py) only
*explains* these numbers; it never computes or changes them. Target changes are
suggestions that a physiotherapist must approve in the dashboard.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

TARGET_STEP_DEG = 5.0
SESSIONS_FOR_SUGGESTION = 3


def summarize_reps(rep_log: List[Dict[str, Any]], direction: str = "extend") -> Dict[str, Any]:
    """Aggregate per-rep records into session-level movement-quality metrics."""
    summary: Dict[str, Any] = {
        "rep_count": len(rep_log),
        "avg_rom_deg": None,
        "rom_variability_pct": None,
        "avg_rep_duration_s": None,
        "avg_peak_angle": None,
        "best_peak_angle": None,
        "max_velocity": None,
        "left_avg_peak": None,
        "right_avg_peak": None,
        "symmetry_gap_deg": None,
        "fatigue_rom_drop_pct": None,
        "reps": rep_log,
    }
    if not rep_log:
        return summary

    roms = [r["rom_deg"] for r in rep_log if r.get("rom_deg") is not None]
    durations = [r["duration_s"] for r in rep_log if r.get("duration_s")]
    peaks = [r["peak_angle"] for r in rep_log if r.get("peak_angle") is not None]
    velocities = [r["max_velocity"] for r in rep_log if r.get("max_velocity") is not None]
    lefts = [r["left_peak"] for r in rep_log if r.get("left_peak") is not None]
    rights = [r["right_peak"] for r in rep_log if r.get("right_peak") is not None]

    if roms:
        mean_rom = statistics.fmean(roms)
        summary["avg_rom_deg"] = round(mean_rom, 1)
        if len(roms) > 1 and mean_rom > 0:
            summary["rom_variability_pct"] = round(statistics.pstdev(roms) / mean_rom * 100, 1)
        # Fatigue indicator: ROM of the last third vs the first third of reps
        if len(roms) >= 6:
            third = len(roms) // 3
            first, last = statistics.fmean(roms[:third]), statistics.fmean(roms[-third:])
            if first > 0:
                summary["fatigue_rom_drop_pct"] = round((first - last) / first * 100, 1)
    if durations:
        summary["avg_rep_duration_s"] = round(statistics.fmean(durations), 2)
    if peaks:
        summary["avg_peak_angle"] = round(statistics.fmean(peaks), 1)
        summary["best_peak_angle"] = max(peaks) if direction == "extend" else min(peaks)
    if velocities:
        summary["max_velocity"] = max(velocities)
    if lefts:
        summary["left_avg_peak"] = round(statistics.fmean(lefts), 1)
    if rights:
        summary["right_avg_peak"] = round(statistics.fmean(rights), 1)
    if lefts and rights:
        summary["symmetry_gap_deg"] = round(
            abs(summary["left_avg_peak"] - summary["right_avg_peak"]), 1
        )
    return summary


def _session_peak(session: Dict[str, Any], direction: str) -> Optional[float]:
    details = session.get("rep_details") or {}
    if details.get("best_peak_angle") is not None:
        return float(details["best_peak_angle"])
    key = "max_angle" if direction == "extend" else "min_angle"
    return session.get(key)


def progress_trend(sessions: List[Dict[str, Any]], direction: str = "extend") -> List[Dict[str, Any]]:
    """Chronological per-session points for charts and LLM context."""
    points = []
    for s in sessions:
        details = s.get("rep_details") or {}
        points.append(
            {
                "session_id": s["id"],
                "date": s["created_at"],
                "exercise_key": s["exercise_key"],
                "reps": s["total_reps"],
                "best_peak_angle": _session_peak(s, direction),
                "avg_rom_deg": details.get("avg_rom_deg"),
                "consistency_score": s["consistency_score"],
                "symmetry_gap_deg": details.get("symmetry_gap_deg"),
            }
        )
    return points


def suggest_target(
    sessions: List[Dict[str, Any]],
    exercise_cfg: Dict[str, Any],
    current_target: Optional[float] = None,
) -> Dict[str, Any]:
    """Rule-based progressive-overload suggestion from the last few sessions.

    extend exercises: higher target angle = harder. flex exercises: lower = harder.
    """
    direction = exercise_cfg.get("direction", "extend")
    target = float(current_target if current_target is not None else exercise_cfg["target_angle_threshold"])
    ideal_min = float(exercise_cfg.get("ideal_min_angle", 30.0))
    ideal_max = float(exercise_cfg.get("ideal_max_angle", 178.0))
    recent = [s for s in sessions if s.get("total_reps", 0) > 0][-SESSIONS_FOR_SUGGESTION:]

    result = {
        "current_target": target,
        "suggested_target": target,
        "action": "keep",
        "reason": "",
    }
    if len(recent) < SESSIONS_FOR_SUGGESTION:
        result["reason"] = (
            f"Need {SESSIONS_FOR_SUGGESTION} completed sessions for a suggestion "
            f"({len(recent)} so far)."
        )
        return result

    peaks = [p for p in (_session_peak(s, direction) for s in recent) if p is not None]
    scores = [s["consistency_score"] for s in recent]
    if len(peaks) < SESSIONS_FOR_SUGGESTION:
        result["reason"] = "Not enough angle data in recent sessions."
        return result

    if direction == "extend":
        exceeded = all(p >= target + TARGET_STEP_DEG for p in peaks)
        struggling = all(p < target for p in peaks)
        harder = min(target + TARGET_STEP_DEG, ideal_max - 3.0)
        easier = max(target - TARGET_STEP_DEG, ideal_min + 10.0)
    else:
        exceeded = all(p <= target - TARGET_STEP_DEG for p in peaks)
        struggling = all(p > target for p in peaks)
        harder = max(target - TARGET_STEP_DEG, ideal_min + 3.0)
        easier = min(target + TARGET_STEP_DEG, ideal_max - 10.0)

    if exceeded and statistics.fmean(scores) >= 80 and harder != target:
        result.update(
            suggested_target=round(harder, 1),
            action="increase_difficulty",
            reason=(
                f"Patient beat the {target:.0f}° target by at least {TARGET_STEP_DEG:.0f}° "
                f"in the last {len(recent)} sessions with good consistency."
            ),
        )
    elif struggling and easier != target:
        result.update(
            suggested_target=round(easier, 1),
            action="decrease_difficulty",
            reason=f"Patient has not reached the {target:.0f}° target in the last {len(recent)} sessions.",
        )
    else:
        result["reason"] = "Current target is appropriately challenging."
    return result
