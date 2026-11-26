"""Phase 1: Verify Jetson Orin Nano Python, OpenCV, MediaPipe, Flask, CUDA, TensorRT, and tegrastats."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_module(import_name: str, dist_name: str, label: str) -> bool:
    try:
        importlib.import_module(import_name)
        try:
            version = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
        print(f"  [OK]   {label:<20} ({version})")
        return True
    except ImportError:
        print(f"  [MISS] {label:<20} (not installed)")
        return False


def main() -> int:
    print("=== Jetson Orin Nano Environment Check ===")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()} ({platform.machine()})\n")

    print("Core Python packages:")
    ok_cv = check_module("cv2", "opencv-python", "OpenCV")
    ok_np = check_module("numpy", "numpy", "NumPy")
    ok_fl = check_module("flask", "Flask", "Flask")
    ok_mp = check_module("mediapipe", "mediapipe", "MediaPipe")

    print("\nOptional / Hardware Acceleration:")
    check_module("ultralytics", "ultralytics", "Ultralytics YOLO")
    ok_trt = check_module("tensorrt", "tensorrt", "TensorRT")
    check_module("torch", "torch", "PyTorch")

    engine_path = ROOT / "models" / "yolov8n-pose.engine"
    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"  [OK]   {'TensorRT Engine':<20} ({engine_path.name}, {size_mb:.1f} MB)")
    else:
        print(f"  [INFO] {'TensorRT Engine':<20} (not built yet — run: python3 scripts/export_tensorrt.py)")

    if ok_cv:
        import cv2

        cuda_count = 0
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            try:
                cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception:
                cuda_count = 0
        print(f"  [INFO] OpenCV CUDA devices: {cuda_count}")

    tegrastats = shutil.which("tegrastats")
    print(f"  [INFO] tegrastats CLI:      {'found at ' + tegrastats if tegrastats else 'not found (non-Jetson host)'}")

    if ok_cv and ok_np and ok_fl:
        print("\nCore stack ready! Run: python3 app.py")
        if not ok_trt:
            print("To enable TensorRT on Jetson: sudo apt install -y python3-libnvinfer tensorrt")
        if not ok_mp:
            print("Note: MediaPipe is not installed; app will fall back to YOLO/TensorRT or Demo mode.")
        return 0
    print("\nMissing core packages. Run: pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
