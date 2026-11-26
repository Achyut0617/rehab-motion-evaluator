"""Phase 9: Benchmark pose inference FPS/latency and capture Jetson Orin Nano tegrastats."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.angle_calculator import JointAngleCalculator
from src.config import FRAME_HEIGHT, FRAME_WIDTH, POSE_BACKEND
from src.pose_estimator import PoseEstimator


def sample_tegrastats(duration_sec: int = 3) -> list[str]:
    if not shutil.which("tegrastats"):
        return []
    proc = subprocess.Popen(
        ["tegrastats", "--interval", "1000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    lines: list[str] = []
    start = time.time()
    try:
        while time.time() - start < duration_sec:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if line:
                lines.append(line.strip())
    finally:
        proc.terminate()
        proc.wait(timeout=2)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference FPS and Jetson tegrastats")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to benchmark")
    parser.add_argument("--backend", default=POSE_BACKEND, choices=["auto", "mediapipe", "yolo", "demo"])
    args = parser.parse_args()

    estimator = PoseEstimator(backend=args.backend)
    angle_calc = JointAngleCalculator()
    dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    latencies_ms: list[float] = []
    t_start = time.perf_counter()
    for _ in range(args.frames):
        t0 = time.perf_counter()
        lm = estimator.detect(dummy_frame)
        angle_calc.calculate_all(lm)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_sec = time.perf_counter() - t_start
    estimator.close()

    arr = np.array(latencies_ms)
    print(f"=== Benchmark Results ({estimator.active_backend}) ===")
    print(f"Frames processed: {args.frames} ({FRAME_WIDTH}x{FRAME_HEIGHT})")
    print(f"Average FPS:      {args.frames / total_sec:.1f}")
    print(f"Mean latency:     {arr.mean():.2f} ms (p95: {np.percentile(arr, 95):.2f} ms)")

    tegra_lines = sample_tegrastats(duration_sec=2)
    if tegra_lines:
        print("\nJetson tegrastats snapshot:")
        for line in tegra_lines:
            print(" ", line)
    else:
        print("\ntegrastats not available on this host (run directly on Jetson Orin Nano).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
