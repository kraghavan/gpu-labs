#!/usr/bin/env bash
# Minimal HiSparse probe — NOT a validation of HiSparse. Just checks whether
# any HiSparse-related Prometheus series appear on a non-sparse-MLA model
# (Qwen3-0.6B, dense GQA). Expected result: absent or all-zero. That's the
# point — a documented negative result, same move as Part 2's
# gpu_cache_usage_perc gotcha.
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"

echo "Querying /metrics on $HOST:$PORT for HiSparse-related series..."
curl -s "http://${HOST}:${PORT}/metrics" | grep -i -E "hisparse|sparse_mla|host_pool|hot_buffer" \
    || echo "No HiSparse-related metrics found (expected on a non-sparse-MLA model)."
