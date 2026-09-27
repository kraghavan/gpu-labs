#!/usr/bin/env bash
# Launches the vLLM weight-cache daemon for one TP rank (TP=1 for this experiment).
#
# Verified against a real v0.30.0 install (2026-09-26, A100): the CLI
# subcommand `vllm weight-cache-daemon` does NOT exist in this build (real
# subcommands are chat, complete, serve, launch, bench, collect-env,
# run-batch). Only the module form works:
#   python -m vllm.model_executor.model_loader.weight_cache.daemon \
#       --model /path/to/model --tensor-parallel-size 4
# The fallback logic below still checks for the CLI form first in case a
# future release adds it — harmless, just falls through to the module form
# on any current v0.30.0 install.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TP="${TP:-1}"

echo "Starting weight-cache daemon for model=$MODEL tp=$TP"
echo "(Ctrl-C to stop; timing_harness.py expects this running in the background for --mode ipc_cache)"

# Preferred: CLI subcommand form. Fall back to module form if unavailable.
if vllm weight-cache-daemon --help >/dev/null 2>&1; then
    exec vllm weight-cache-daemon --model "$MODEL" --tensor-parallel-size "$TP"
else
    exec python -m vllm.model_executor.model_loader.weight_cache.daemon \
        --model "$MODEL" --tensor-parallel-size "$TP"
fi
