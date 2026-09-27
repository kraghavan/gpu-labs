#!/usr/bin/env python3
"""Fast Start timing harness.

Measures wall-clock time from `vllm serve` process start to the engine
becoming ready (first 200 from /v1/models), across N trials, in either
"cold" (--load-format auto) or "ipc_cache" mode.

Usage:
    # Terminal 1 (ipc_cache mode only): ./run_daemon.sh
    # Terminal 2:
    python timing_harness.py --mode ipc_cache --trials 5 --out ../../results/fast_start_ipc.csv
    python timing_harness.py --mode cold      --trials 5 --out ../../results/fast_start_cold.csv

Known gaps, verify before trusting the numbers:
  - Graph-capture-time extraction (`GRAPH_CAPTURE_LOG_PATTERN` below) is a
    best-effort regex against server stdout. The exact log line format for
    v0.30.0 was not confirmed from documentation search — check your actual
    server logs and update the pattern before relying on t_graph_capture_s.
    The total-init timing (t_total_init_s) does NOT depend on this and is
    robust regardless of log format, since it's measured by HTTP polling.
  - This assumes a single GPU, TP=1. Not measuring the TP-sharding benefit
    Fast Start is partly designed for.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

GRAPH_CAPTURE_LOG_PATTERN = re.compile(
    r"Graph capturing finished in (\d+) secs"
)  # Confirmed against real v0.30.0 log output on 2026-09-26 (A100, Qwen3-0.6B).
# There are TWO such lines per engine start (PIECEWISE capture, then FULL capture)
# — sum all matches, don't take the first. Integer seconds, not decimal, contrary
# to the original guess this replaced.


def total_graph_capture_seconds(log_text: str) -> int | None:
    matches = GRAPH_CAPTURE_LOG_PATTERN.findall(log_text)
    if not matches:
        return None
    return sum(int(m) for m in matches)


def wait_for_ready(port: int, timeout_s: float = 300.0) -> float:
    """Poll /v1/models until it returns 200. Returns wall-clock seconds elapsed."""
    start = time.perf_counter()
    url = f"http://localhost:{port}/v1/models"
    while time.perf_counter() - start < timeout_s:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Server on port {port} did not become ready within {timeout_s}s")


def run_trial(mode: str, model: str, port: int, log_path: Path, gpu_mem_util: float = 0.90,
              max_model_len: int | None = None) -> dict:
    load_format = "ipc_cache" if mode == "ipc_cache" else "auto"
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--load-format", load_format,
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    with open(log_path, "w") as logf:
        t_process_start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        try:
            t_total_init_s = wait_for_ready(port)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    t_graph_capture_s = None
    try:
        log_text = log_path.read_text(errors="ignore")
        t_graph_capture_s = total_graph_capture_seconds(log_text)
    except OSError:
        pass

    return {
        "mode": mode,
        "t_total_init_s": round(t_total_init_s, 3),
        "t_graph_capture_s": t_graph_capture_s,
        "log_path": str(log_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "ipc_cache"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="../../results/fast_start_results.csv")
    ap.add_argument(
        "--cooldown-s", type=float, default=5.0,
        help="pause between trials to let the port fully release",
    )
    ap.add_argument(
        "--gpu-mem-util", type=float, default=0.90,
        help="--gpu-memory-utilization passed to vllm serve. Lower this if the "
             "daemon (in ipc_cache mode) already holds enough weight memory that "
             "the default leaves too little free at engine-startup memory check "
             "time — this is a real constraint at larger model sizes, not a bug.",
    )
    ap.add_argument(
        "--max-model-len", type=int, default=None,
        help="Bound context length to reduce KV cache memory footprint — this "
             "experiment measures load time, not long-context serving capacity, "
             "and an unbounded max_model_len can push KV cache requirements to "
             "the edge of what --gpu-mem-util leaves available.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mode", "trial", "t_total_init_s", "t_graph_capture_s", "log_path"]
        )
        if write_header:
            writer.writeheader()

        for i in range(args.trials):
            print(f"[{args.mode}] trial {i+1}/{args.trials} ...", file=sys.stderr)
            safe_model = args.model.replace("/", "_")
            log_path = log_dir / f"{safe_model}_{args.mode}_trial{i+1}.log"
            result = run_trial(args.mode, args.model, args.port, log_path, args.gpu_mem_util, args.max_model_len)
            result["trial"] = i + 1
            writer.writerow(result)
            f.flush()
            print(f"  -> t_total_init_s={result['t_total_init_s']} "
                  f"t_graph_capture_s={result['t_graph_capture_s']}", file=sys.stderr)
            time.sleep(args.cooldown_s)

    print(f"Done. Results appended to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
