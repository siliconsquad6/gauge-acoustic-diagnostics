#!/bin/bash
O=~/Desktop/edge-ai/gauge/out
for m in fan pump valve; do
  find ~ -ipath "*/$m/*abnormal*" -name '*.wav' 2>/dev/null | shuf -n2 | while read WAV; do
    N="${m}_$(echo "$WAV" | grep -oiE 'id_0[0-9]')_$(basename "$WAV" .wav)"
    echo "=== $N ($(date +%T)) ==="
    ~/Desktop/edge-ai/gauge/run.sh "$WAV" || { echo "FAIL $N"; continue; }
    mkdir -p "$O/runs/$N" && cp "$O"/{diag.json,machine.png,machine.glb} "$O/runs/$N/"
    echo "$N $(tr -d '\n' < "$O/diag.json")" >> "$O/runs/results.txt"
  done
done
echo "ALL DONE $(date +%T)"
