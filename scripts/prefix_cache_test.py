import time, httpx, json

MODEL = "mlx-community/Qwen3-0.6B-4bit"
SYSTEM_PROMPT = "You are a helpful assistant. " * 50  # long shared prefix

def chat(user_msg, label):
    start = time.perf_counter()
    r = httpx.post("http://localhost:8000/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 30,
            "stream": True
        }, headers={"Content-Type": "application/json"}, timeout=30)

    first_token = None
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            if chunk["choices"][0]["delta"].get("content") and first_token is None:
                first_token = time.perf_counter()
                break

    ttft = (first_token - start) * 1000 if first_token else -1
    print(f"{label}: TTFT={ttft:.0f}ms")

chat("What is 2+2?", "First request (cold cache)")
chat("What is 3+3?", "Second request (warm cache - same system prompt)")
chat("What is 4+4?", "Third request (warmer cache)")
