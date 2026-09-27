#!/usr/bin/env bash
# Survival checklist — run this first, every session, on whatever box you're on.
# Adapted from the Part 3 GH200 "survival checklist" pattern.
set -euo pipefail

echo "=== vLLM version ==="
python -c "import vllm; print(vllm.__version__)" || echo "vllm not importable — check venv/activation"

echo
echo "=== GPU visibility ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
else
    echo "nvidia-smi not found — are you on Apple Silicon? (expected for the watermark experiment)"
    command -v system_profiler >/dev/null 2>&1 && system_profiler SPDisplaysDataType | grep -A3 "Chipset Model" || true
fi

echo
echo "=== Ports 8000 / 8001 free? ==="
for p in 8000 8001; do
    if lsof -ti ":$p" >/dev/null 2>&1; then
        echo "port $p IN USE — kill before starting a fresh server:"
        lsof -i ":$p"
    else
        echo "port $p free"
    fi
done

echo
echo "=== Cost tracking reminder (GH200 only) ==="
echo "Session start: $(date -u +'%Y-%m-%dT%H:%M:%SZ') — Lambda GH200 billed at ~\$2.29/hr per Part 3."
echo "Note the instance IP and start time in results/ before proceeding."
