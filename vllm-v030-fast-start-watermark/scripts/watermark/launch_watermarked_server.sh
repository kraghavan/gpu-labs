#!/usr/bin/env bash
# Watermarked server — Qwen3-0.6B, Gumbel-max, fixed key for reproducibility.
# Runs anywhere v0.30.0 is installed: M4 Mac Mini (vllm-metal) or GH200 (CUDA).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
KEY="${WATERMARK_KEY:-12345}"

echo "Starting WATERMARKED server: model=$MODEL port=$PORT key=$KEY"
exec vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len 512 \
    --watermark-config "{\"algorithm\":\"gumbel\",\"key\":${KEY}}"
