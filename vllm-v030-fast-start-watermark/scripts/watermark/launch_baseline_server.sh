#!/usr/bin/env bash
# Baseline (non-watermarked) server — same model, different port.
# Used for the false-positive detector control.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8001}"

echo "Starting BASELINE (no watermark) server: model=$MODEL port=$PORT"
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 512
