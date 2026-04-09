import random
from locust import HttpUser, task, between

MODEL = "mlx-community/Qwen3-0.6B-4bit"

SHORT_PROMPTS = [
    "What is 2+2?",
    "Name a color.",
    "What is the capital of France?",
]

LONG_PROMPTS = [
    "Explain in detail how transformer attention mechanisms work. Cover queries, keys, values, and the softmax scaling factor.",
    "Describe the history of distributed computing from mainframes to microservices in extensive detail.",
    "Explain PagedAttention in vLLM and why it improves KV cache efficiency compared to static allocation.",
]

class InferenceUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def short_request(self):
        """High-frequency, short prompt — simulates chatbot queries"""
        self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": random.choice(SHORT_PROMPTS)}],
            "max_tokens": 50,
        }, name="short_prompt")

    @task(1)
    def long_request(self):
        """Low-frequency, long prompt — simulates doc summarization"""
        self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": random.choice(LONG_PROMPTS)}],
            "max_tokens": 200,
        }, name="long_prompt")