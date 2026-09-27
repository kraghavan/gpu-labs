#!/usr/bin/env python3
"""Generate/detect round-trip for the Gumbel-max watermarking feature.

Requires:
    - launch_watermarked_server.sh running on WATERMARK_PORT (default 8000)
    - launch_baseline_server.sh running on BASELINE_PORT (default 8001)
    Run these SEQUENTIALLY, not concurrently — both default to ~92% GPU
    memory utilization, and launching them at the same time on one GPU
    starves the second of memory. Confirmed the hard way on an A100 40GB.

Produces a small results table covering:
    - true positive: watermarked output, temperature>0, detected?
    - true negative: baseline output, same key attempted, NOT detected?
    - bypass check: watermarked server, temperature=0, still unwatermarked?

Verified against a real vLLM v0.30.0 install (2026-09-26/27, A100):
    - Detector API: `from vllm.v1.watermarking.gumbel import GumbelWatermarkDetector`
      `GumbelWatermarkDetector(key=int).detect(token_ids: list[int])` returns
      `WatermarkDetection(score, p_value, num_scored_tokens, is_watermarked)`.
    - The completions endpoint does NOT return `token_ids` unless you pass
      `"return_token_ids": true` in the request body — without it you'll
      get `TypeError: 'NoneType' object is not iterable` from `.detect()`.
"""

import argparse
import json
import sys

import httpx

PROMPTS = [
    "Explain what a KV cache is in one paragraph.",
    "Write a short recipe for scrambled eggs.",
    "Describe the water cycle in three sentences.",
    "What is the capital of France, and why is it significant?",
    "Summarize why continuous batching improves LLM throughput.",
]

KEY = 12345  # must match WATERMARK_KEY used to launch the watermarked server


def generate(base_url: str, prompt: str, temperature: float, max_tokens: int = 100):
    resp = httpx.post(
        f"{base_url}/v1/completions",
        json={
            "model": "Qwen/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "return_token_ids": True,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    token_ids = data["choices"][0].get("token_ids")
    return text, token_ids


def detect(token_ids, key: int):
    from vllm.v1.watermarking.gumbel import GumbelWatermarkDetector

    detector = GumbelWatermarkDetector(key=key)
    result = detector.detect(token_ids)
    return {
        "score": result.score,
        "p_value": result.p_value,
        "num_scored_tokens": result.num_scored_tokens,
        "is_watermarked": result.is_watermarked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watermark-url", default="http://localhost:8000")
    ap.add_argument("--baseline-url", default="http://localhost:8001")
    ap.add_argument("--out", default="../../results/watermark_results.json")
    args = ap.parse_args()

    results = []

    print("=== True positive: watermarked server, temperature=0.8 ===", file=sys.stderr)
    for prompt in PROMPTS:
        text, token_ids = generate(args.watermark_url, prompt, temperature=0.8)
        det = detect(token_ids, KEY) if token_ids else None
        results.append({"case": "watermarked_temp0.8", "prompt": prompt,
                         "text": text, **(det or {})})
        print(f"  prompt={prompt[:40]!r} -> {det}", file=sys.stderr)

    print("=== True negative: baseline server (no watermark config) ===", file=sys.stderr)
    for prompt in PROMPTS:
        text, token_ids = generate(args.baseline_url, prompt, temperature=0.8)
        det = detect(token_ids, KEY) if token_ids else None
        results.append({"case": "baseline_temp0.8", "prompt": prompt,
                         "text": text, **(det or {})})
        print(f"  prompt={prompt[:40]!r} -> {det} (expect: is_watermarked=False)", file=sys.stderr)

    print("=== Bypass check: watermarked server, temperature=0 (greedy) ===", file=sys.stderr)
    for prompt in PROMPTS[:2]:
        text, token_ids = generate(args.watermark_url, prompt, temperature=0.0)
        det = detect(token_ids, KEY) if token_ids else None
        results.append({"case": "watermarked_temp0_greedy", "prompt": prompt,
                         "text": text, **(det or {})})
        print(f"  prompt={prompt[:40]!r} -> {det} "
              f"(expect: bypass warning in server logs, detection score meaningless)", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
