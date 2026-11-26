"""Phase 2: Standalone test script for Android IP Camera URL or USB webcam via OpenCV."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.camera import VideoStream
from src.config import CAMERA_SOURCE, FRAME_HEIGHT, FRAME_WIDTH, parse_camera_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Android IP Camera or USB webcam stream")
    parser.add_argument(
        "--camera",
        default=str(CAMERA_SOURCE),
        help="IP Camera URL (e.g. http://192.168.1.100:8080/video), USB index (0), or 'demo'",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (5s FPS check)")
    args = parser.parse_args()

    src = parse_camera_source(args.camera)
    print(f"Connecting to camera source: {src}")
    stream = VideoStream(source=src, width=FRAME_WIDTH, height=FRAME_HEIGHT).start()

    try:
        if args.headless:
            end_time = time.time() + 5.0
            frames = 0
            while time.time() < end_time:
                ok, frame = stream.read()
                if ok and frame is not None:
                    frames += 1
                time.sleep(0.01)
            print(f"Headless check complete: {frames} frames read, stream FPS={stream.fps}")
            return 0 if frames > 0 else 1

        print("Press 'q' in the video window to quit.")
        while True:
            ok, frame = stream.read()
            if ok and frame is not None:
                cv2.putText(
                    frame,
                    f"Source: {src} | FPS: {stream.fps}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (34, 197, 94),
                    2,
                )
                cv2.imshow("Phase 2 - Camera Feed Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
