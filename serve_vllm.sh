#!/bin/bash
# Serve Qwen2.5-Omni with vLLM on the HP ZGX Nano (GB10) as an OpenAI-compatible endpoint on port 8001.
# Uses NVIDIA's vLLM container. Run from the gauge folder:  bash serve_vllm.sh
# Then start the app with vLLM turned on:                  GAUGE_VLLM=1 python app/server.py
#
# VLLM_IMAGE  container tag (check NGC for the newest nvcr.io/nvidia/vllm tag)
# VLLM_MEM    share of GPU memory for vLLM. The app server also needs memory, keep this low.
cd "$(dirname "$0")"
IMAGE=${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}
MEM=${VLLM_MEM:-0.35}

docker rm -f gauge-vllm >/dev/null 2>&1
docker run -d --name gauge-vllm --gpus all --ipc=host -p 8001:8001 \
  -v "$PWD/models:/models" -e HF_HOME=/models/hf \
  "$IMAGE" \
  vllm serve /models/omni \
    --served-model-name gauge-omni \
    --host 0.0.0.0 --port 8001 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization "$MEM" \
    --enable-auto-tool-choice --tool-call-parser hermes

echo "Starting vLLM (first start takes a few minutes). Logs: docker logs -f gauge-vllm"
until curl -sf http://localhost:8001/v1/models >/dev/null; do
  docker ps -q -f name=gauge-vllm | grep -q . || { echo "vLLM container stopped. Check: docker logs gauge-vllm"; exit 1; }
  sleep 5; printf "."
done
echo; curl -s http://localhost:8001/v1/models; echo
echo "vLLM READY on http://localhost:8001/v1"