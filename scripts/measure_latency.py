import time
import httpx
import json

MODEL = "mlx-community/Qwen3-0.6B-4bit"
BASE_URL = "http://localhost:8000"

def measure_streaming(prompt: str, max_tokens: int = 100):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    first_token_time = None
    token_times = []
    start = time.perf_counter()

    with httpx.Client(timeout=60) as client:
        with client.stream("POST", f"{BASE_URL}/v1/chat/completions",
                           json=payload,
                           headers={"Content-Type": "application/json"}) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        print(f"TTFT: {(first_token_time - start)*1000:.1f}ms")
                    else:
                        token_times.append(now - token_times[-1] if token_times else now - first_token_time)
                    token_times.append(now)

    total_time = time.perf_counter() - start
    if len(token_times) > 2:
        intervals = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        avg_tpot = sum(intervals) / len(intervals) * 1000
        print(f"Avg TPOT: {avg_tpot:.1f}ms")
        print(f"Throughput: {len(token_times) / total_time:.1f} tok/s")

# Test 1: Short prompt (fast prefill)
print("=== Short prompt ===")
measure_streaming("What is 2+2?", max_tokens=50)

# Test 2: Long prompt (slow prefill, watch TTFT increase)
print("\n=== Long prompt ===")
long_prompt = "Summarize the following text: " + ("The quick brown fox jumps over the lazy dog. " * 100)
measure_streaming(long_prompt, max_tokens=50)