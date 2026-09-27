# vLLM v0.30.0 — Fast Start & Watermarking

Scripts behind [Part 7 of "LLM Inference from First
Principles"](https://kraghavan.ca) — measuring two features from vLLM
v0.30.0 (released 2026-09-22) on a rented Lambda Cloud A100: the Fast
Start weight-cache daemon and CUDA IPC restart, and Gumbel-max text
watermarking. (A third feature, HiSparse, needs a 300B+ sparse-MLA model
and an 8-GPU node to exercise at all — that gets its own post later.)

## Layout

```
scripts/
  fast_start/
    run_daemon.sh              # launches the weight-cache daemon (module-form fallback included)
    timing_harness.py          # cold vs ipc_cache restart timing, with graph-capture extraction
    values-ipc-cache-k8s.yaml  # sketch only, not run — see notes in the file
  watermark/
    launch_watermarked_server.sh
    launch_baseline_server.sh  # for the false-positive detection control
    generate_and_detect.py     # generate + GumbelWatermarkDetector round trip
  common/
    env_check.sh
    hisparse_metric_probe.sh   # confirms HiSparse Prometheus metrics are absent on non-sparse-MLA models
    kill_zombie_engines.sh     # see "gotcha" below — you will need this
```

## Results summary

- **Fast Start, Qwen3-0.6B:** `ipc_cache` restart ~4.4% faster than cold
  (37.7s vs 39.5s total init). Graph-capture time showed no difference
  between arms — see the post for why that specific comparison has a real
  methodological gap.
- **Fast Start, Qwen3-8B:** ~8.0% faster (40.2s vs 43.7s) — the effect
  roughly doubled going from 0.6B to 8B, consistent with disk-load time
  being a bigger fraction of total init at larger sizes.
- **Watermarking:** 3/3 true positives (p-values < 1e-5), 3/3 correct true
  negatives against a non-watermarked baseline (p-values 0.19–0.49),
  temperature=0 bypass confirmed verbatim in server logs.

Full numbers, raw logs, and the complete "what broke" list are in the blog
post, not duplicated here.

## The one gotcha worth knowing before you run any of this

`pkill -f 'vllm serve'` does not reliably stop the engine. The EngineCore
subprocess renames its own process title to `VLLM::EngineCore` and gets
reparented to init if its parent dies — it survives the pkill and keeps
holding GPU memory. Confirmed 3 times in one session. Run
`scripts/common/kill_zombie_engines.sh` between trials, or at minimum
check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
before assuming the GPU is free.

## Requirements

- A CUDA GPU with enough VRAM for whatever model you point these at
  (tested on a 40GB A100 with Qwen3-0.6B and Qwen3-8B)
- `pip install vllm==0.30.0` into a **fresh venv** — installing system-wide
  alongside an existing `scipy`/apt Python will likely hit a NumPy 2.0
  incompatibility (`ImportError: cannot import name 'Inf' from numpy`)
- `httpx`, `huggingface_hub` (both pulled in by vLLM's own deps)

## Contributing

These scripts are exactly what ran during the experiment, warts and all.
If you find a cleaner way to do the timing harness, a better fix for the
zombie-process issue, or you've reproduced this on different hardware —
open an issue or a PR. Genuinely welcome the feedback.
