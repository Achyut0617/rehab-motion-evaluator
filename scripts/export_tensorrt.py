"""Export YOLOv8-Pose (.pt) to an FP16 TensorRT (.engine) file on NVIDIA Jetson Orin Nano."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8-Pose model to FP16 TensorRT engine for Jetson Orin Nano"
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Ultralytics pose model name or path (default: yolov8n-pose.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 and export in FP32")
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install dependencies first: pip install -r requirements.txt")
        return 1

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("TensorRT Python bindings not found in this environment.")
        print("On Jetson Orin Nano, install them via:")
        print("  sudo apt update && sudo apt install -y python3-libnvinfer tensorrt")
        print("Or inside .venv: pip install tensorrt")
        return 1

    if not torch.cuda.is_available():
        print("CUDA GPU not detected. Run this export directly on your NVIDIA Jetson Orin Nano.")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} and exporting to TensorRT (imgsz={args.imgsz}, half={not args.fp32})...")
    model = YOLO(args.model)
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=args.imgsz,
            half=not args.fp32,
            device=0,
        )
    )

    target_path = MODELS_DIR / exported_path.name
    if exported_path.resolve() != target_path.resolve():
        shutil.move(str(exported_path), str(target_path))

    print(f"\nTensorRT engine saved to: {target_path}")
    print("POSE_BACKEND=auto will now automatically use TensorRT, or set POSE_BACKEND=yolo in .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
