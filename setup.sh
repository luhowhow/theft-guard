#!/usr/bin/env bash
# One-step setup for Theft Guard (Ubuntu 24.04 / 26.04).
# Safe to run again: steps that are already done are skipped.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/5 Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libopencv-dev \
  ffmpeg alsa-utils espeak-ng curl

echo "==> 2/5 Downloading models (YOLOX, Apache 2.0)"
mkdir -p models
BASE="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
for m in yolox_tiny yolox_s; do
  if [ -f "models/$m.onnx" ]; then
    echo "    models/$m.onnx already exists"
  else
    curl -L --fail -o "models/$m.onnx" "$BASE/$m.onnx"
  fi
done

echo "==> 3/5 Creating the warning sound"
mkdir -p sounds
if [ -f sounds/alert.wav ]; then
  echo "    sounds/alert.wav already exists"
else
  espeak-ng -v en -s 140 -w sounds/alert.wav \
    "Warning. This area is under surveillance. The manager has been notified."
fi

echo "==> 4/5 Preparing .env"
if [ -f .env ]; then
  echo "    .env already exists"
else
  cp .env.example .env
  echo "    Created .env from .env.example. Edit it to add your Discord webhook URL."
fi
chmod +x scripts/*.sh

echo "==> 5/5 Building"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo
echo "Done. Environment:"
bash scripts/env_info.sh
echo
echo "Run:  ./build/guard <video.mp4 | rtsp://... | 0> [more cameras...]"
