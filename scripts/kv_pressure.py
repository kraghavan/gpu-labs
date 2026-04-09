import httpx, concurrent.futures, time

def send_long_request(i):
    start = time.perf_counter()
    r = httpx.post("http://localhost:8000/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3-0.6B-4bit",
            "messages": [{"role": "user", "content": f"Write a very long essay about distributed systems topic {i}. Be extremely verbose and detailed."}],
            "max_tokens": 500,
        },
        timeout=120)
    elapsed = time.perf_counter() - start
    tokens = r.json()["usage"]["completion_tokens"]
    print(f"Request {i}: {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)")
    return elapsed

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(send_long_request, i) for i in range(4)]
    results = [f.result() for f in futures]