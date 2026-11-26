"""Phase 3 & 4: Standalone test script for human pose estimation and live joint angles."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.angle_calculator import JointAngleCalculator
from src.camera import VideoStream
from src.config import CAMERA_SOURCE, FRAME_HEIGHT, FRAME_WIDTH, POSE_BACKEND, parse_camera_source
from src.pose_estimator import PoseEstimator


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Pose Estimation and Joint Angle Calculation")
    parser.add_argument("--camera", default=str(CAMERA_SOURCE), help="IP Camera URL, USB index, or 'demo'")
    parser.add_argument("--backend", default=POSE_BACKEND, choices=["auto", "mediapipe", "yolo", "demo"])
    args = parser.parse_args()

    src = parse_camera_source(args.camera)
    stream = VideoStream(source=src, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()
    estimator = PoseEstimator(backend=args.backend)
    angle_calc = JointAngleCalculator()

    print(f"Active pose backend: {estimator.active_backend} | Press 'q' to quit.")
    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            landmarks = estimator.detect(frame, force_demo=stream.is_demo)
            angles = angle_calc.calculate_all(landmarks)
            estimator.draw_skeleton(frame, landmarks)

            for joint_name, val in angles.items():
                if val is not None and joint_name in landmarks:
                    lm = landmarks[joint_name]
                    cv2.putText(
                        frame,
                        f"{joint_name.split('_')[1]}: {int(round(val))} deg",
                        (int(lm.x) + 8, int(lm.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (250, 204, 21),
                        2,
                    )

            cv2.imshow("Phase 3 & 4 - Pose & Joint Angles", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        estimator.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
