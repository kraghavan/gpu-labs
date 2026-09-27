#!/usr/bin/env bash
# `pkill -f 'vllm serve'` only kills the APIServer process. The EngineCore
# child renames its own process title via setproctitle (shows up as
# `VLLM::EngineCore` in ps/nvidia-smi, not `vllm serve`), gets reparented to
# init when its parent dies, and keeps holding GPU memory. Confirmed this
# pattern 3 times in one session (2026-09-26/27, A100, v0.30.0).
#
# This kills both the APIServer AND whatever's actually holding GPU memory,
# found via nvidia-smi rather than trusting a name-based pkill pattern.
set -euo pipefail

echo "Killing vllm serve / weight-cache-daemon processes by name..."
pkill -9 -f 'vllm serve' 2>/dev/null || true
pkill -9 -f 'weight_cache.daemon' 2>/dev/null || true
sleep 2

echo "Checking nvidia-smi for anything still holding GPU memory..."
LEFTOVER=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)

if [ -n "$LEFTOVER" ]; then
    echo "Found leftover process(es) still holding GPU memory:"
    echo "$LEFTOVER"
    echo "$LEFTOVER" | while IFS=',' read -r pid _; do
        pid=$(echo "$pid" | tr -d ' ')
        echo "Killing PID $pid directly..."
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
    echo "GPU memory after cleanup:"
    nvidia-smi --query-gpu=memory.used --format=csv
else
    echo "Clean — no leftover compute processes found."
fi
