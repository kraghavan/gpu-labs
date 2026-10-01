"""Load generator for pd-ratio-coordinator validation.

Two task sets, run against whichever pool's port-forward you point Locust
at (--host). Short prompts stress decode (sustained TPOT pressure); long
prompts stress prefill (queue depth / velocity).

Usage:
    # Test 1 (TPOT breach) -- against decode, port-forwarded to 8001
    locust -f locustfile.py --host http://localhost:8001 \
        --headless --users 30 --spawn-rate 3 --run-time 90s DecodeLoad

    # Test 2 (velocity spike) -- against prefill, port-forwarded to 8000
    locust -f locustfile.py --host http://localhost:8000 \
        --headless --users 20 --spawn-rate 20 --run-time 30s PrefillBurst
"""

from locust import HttpUser, task, between

MODEL = "Qwen/Qwen3-0.6B"


class DecodeLoad(HttpUser):
    """Sustained short-prompt load -- generates enough tokens per request
    to keep decode busy and push TPOT p95 up under concurrency."""

    wait_time = between(0.1, 0.5)

    @task
    def short_prompt(self):
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": "Write a short paragraph about distributed systems.",
                "max_tokens": 150,
                "temperature": 0.8,
            },
            timeout=30,
        )


class PrefillBurst(HttpUser):
    """Burst of long prompts -- spawn-rate should be set high (near-simultaneous)
    to trigger velocity detection rather than just steady queue depth."""

    wait_time = between(0.05, 0.2)

    @task
    def long_prompt(self):
        long_text = "Analyze the following in detail: " + "word " * 300
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": long_text,
                "max_tokens": 20,  # short output -- we care about prefill queue, not decode time
                "temperature": 0.0,
            },
            timeout=30,
        )
