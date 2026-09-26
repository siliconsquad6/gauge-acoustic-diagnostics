#!/bin/bash
set -e
WAV="$1"
MACHINE=$(echo "$WAV" | grep -oiE 'valve|fan|pump|slider' | head -1 | tr A-Z a-z)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate gauge
echo "[1/4] Listening ($MACHINE)..."; python ~/Desktop/edge-ai/gauge/app/listen.py "$WAV" "${MACHINE:-machine}"
echo "[2/4] Drawing...";   python ~/Desktop/edge-ai/gauge/app/draw.py
echo "[3/4] 3D shape...";  python ~/Desktop/edge-ai/gauge/app/shape.py
echo "[4/4] Locating defect..."; python ~/Desktop/edge-ai/gauge/app/tighten.py
echo "Done. Refresh viewer.html"
