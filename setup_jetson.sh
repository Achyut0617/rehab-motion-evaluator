#!/usr/bin/env bash
# Automated setup script for NVIDIA Jetson Orin Nano (Ubuntu / JetPack)
set -e

echo "=== Patient Rehabilitation Motion Evaluator: Jetson Setup ==="

# 1. Create virtual environment with access to JetPack system packages (CUDA/TensorRT/OpenCV)
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment (.venv) with --system-site-packages..."
    python3 -m venv .venv --system-site-packages
else
    echo "[1/5] Virtual environment (.venv) already exists."
fi

# 2. Activate virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Upgrade pip and install requirements
echo "[2/5] Upgrading pip and installing requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create .env from template if missing
if [ ! -f ".env" ]; then
    echo "[3/5] Creating .env from .env.example..."
    cp .env.example .env
else
    echo "[3/5] .env already exists."
fi

# 5. Export TensorRT FP16 engine if TensorRT + CUDA are available and engine is missing
if [ ! -f "models/yolov8n-pose.engine" ]; then
    if python3 -c "import tensorrt, torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[4/5] Building FP16 TensorRT engine (this takes 2-5 minutes)..."
        python3 scripts/export_tensorrt.py --model yolov8n-pose.pt --imgsz 640
    else
        echo "[4/5] Skipping TensorRT engine build (install python3-libnvinfer tensorrt on Jetson, then run: python3 scripts/export_tensorrt.py)."
    fi
else
    echo "[4/5] TensorRT engine already exists at models/yolov8n-pose.engine."
fi

# 6. Run environment check
echo "[5/5] Running environment diagnostics..."
python3 scripts/check_environment.py

echo ""
echo "Setup complete! To start the dashboard:"
echo "  source .venv/bin/activate"
echo "  python3 app.py"
