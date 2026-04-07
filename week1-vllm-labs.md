# Week 1: vLLM Hands-On Lab — M4 Mac Mini
> Goal: Go from zero to serving a model, understanding the API surface, wiring up
> Prometheus/Grafana, load testing, and reading KV cache metrics live.

---

## Prerequisites Checklist

Before Day 1, verify these are installed:

```bash
# Check existing tools
kubectl version --client    # should be 1.27+
helm version                # should be 3.x
docker info                 # should be running
python3 --version           # should be 3.11+
brew --version              # should exist

# Install anything missing
brew install kind kubectl helm k9s vegeta
pip install huggingface_hub[cli] locust
huggingface-cli login       # paste your HF token when prompted
```

---

## Day 1 — Install vllm-mlx and Serve Your First Model

### Why vllm-mlx (not vllm-metal)?
- Single `pip install` vs 15-min build from source
- Published benchmarks: 525 tok/s on Qwen3-0.6B 4-bit on M4 Max
- Full OpenAI-compatible API — same interface you'd hit on a cloud vLLM instance
- Accepted at EuroMLSys '26 — this is serious research, not a toy

### Step 1.1 — Install

```bash
# Always isolate ML tooling
python3 -m venv ~/.venv-vllm-mlx
source ~/.venv-vllm-mlx/bin/activate

# Add this to your ~/.zshrc so you can activate quickly
echo 'alias vllm-env="source ~/.venv-vllm-mlx/bin/activate"' >> ~/.zshrc

# Install
pip install git+https://github.com/waybarrios/vllm-mlx.git

# Verify
vllm-mlx --help
```

### Step 1.2 — Download a Model

We use Qwen3-0.6B (4-bit quantized) — small enough for 16GB RAM, fast enough to be useful.

```bash
# This downloads ~400MB from HuggingFace mlx-community
huggingface-cli download mlx-community/Qwen3-0.6B-4bit --local-dir ~/models/qwen3-0.6b-4bit
```

**Why Qwen3-0.6B?**
- Fits comfortably in 16GB (model ~400MB, KV cache grows with context)
- MLX-quantized format — runs on Metal GPU, not CPU
- Same model llm-d docs use for prefix cache routing examples

### Step 1.3 — Start the Server

```bash
# Single-user mode first (simpler, max speed per request)
vllm-mlx serve mlx-community/Qwen3-0.6B-4bit --port 8000

# Expected startup output:
# INFO: Loading model mlx-community/Qwen3-0.6B-4bit
# INFO: Model loaded in X.Xs
# INFO: Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal open. Open a new terminal for the rest.

### Step 1.4 — Hit the API

```bash
# Verify the server is alive
curl http://localhost:8000/health | jq .

# List available models (OpenAI-compatible)
curl http://localhost:8000/v1/models | jq .

# Your first inference request
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "What is KV cache in LLM inference?"}],
    "max_tokens": 200
  }' | jq .choices[0].message.content
```

**What to notice in the response:**
- `usage.prompt_tokens` — how many tokens the prompt was
- `usage.completion_tokens` — how many tokens generated
- These are the building blocks of TTFT and TPOT

### Day 1 Checkpoint ✅
- [ ] `vllm-mlx` installed in a venv
- [ ] Model downloaded
- [ ] Server running and responding to `/v1/chat/completions`

---

## Day 2 — Explore the Full API Surface

### Step 2.1 — Streaming Responses

Streaming is how you see decode tokens arrive one-by-one (the UX you see in ChatGPT).

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Count slowly from 1 to 20, one number per line."}],
    "max_tokens": 200,
    "stream": true
  }'
```

Watch the `data: {"choices":[{"delta":{"content":"..."}}]}` chunks arrive.
Each chunk = one decoded token being returned. This is TPOT visible to the naked eye.

### Step 2.2 — Measure TTFT and TPOT Manually

```python
# save as ~/labs/measure_latency.py
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
```

```bash
python ~/labs/measure_latency.py
```

**What you should see:** TTFT is higher for the long prompt (more prefill work),
but TPOT stays roughly constant (decode is independent of prompt length).
This is the core insight that motivates P/D disaggregation.

### Step 2.3 — Prefix Cache in Action

```python
# save as ~/labs/prefix_cache_test.py
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
```

```bash
python ~/labs/prefix_cache_test.py
```

**Expected:** TTFT drops significantly on the 2nd and 3rd requests because the
system prompt tokens are already in the KV cache. This is prefix caching — one of
the key features llm-d's scheduler routes around.

### Day 2 Checkpoint ✅
- [ ] Observed streaming tokens arriving
- [ ] Measured TTFT vs TPOT difference between short/long prompts
- [ ] Observed prefix cache speedup

---

## Day 3 — Wire Up Prometheus and Grafana

### Step 3.1 — Enable Metrics

Restart vllm-mlx with continuous batching (required for metrics + concurrency):

```bash
# Stop the existing server (Ctrl+C), restart with continuous batching
vllm-mlx serve mlx-community/Qwen3-0.6B-4bit \
  --port 8000 \
  --continuous-batching

# Verify metrics endpoint
curl http://localhost:8000/metrics | grep "^vllm" | head -30
```

**Key metrics to find in the output:**
```
vllm:num_requests_running      # inflight requests right now
vllm:num_requests_waiting      # queued requests
vllm:gpu_cache_usage_perc      # KV cache utilization (0–1)
vllm:time_to_first_token_seconds_bucket   # TTFT histogram
vllm:time_per_output_token_seconds_bucket # TPOT histogram
```

### Step 3.2 — Docker Compose for Prometheus + Grafana

```bash
mkdir -p ~/labs/observability
cd ~/labs/observability
```

**Create `prometheus.yml`:**
```yaml
# ~/labs/observability/prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - 'host.docker.internal:8000'  # reaches your Mac's localhost from inside Docker
```

**Create `docker-compose.yml`:**
```yaml
# ~/labs/observability/docker-compose.yml
version: "3"
services:
  prometheus:
    image: prom/prometheus:latest
    extra_hosts:
      - "host.docker.internal:host-gateway"
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
```

```bash
cd ~/labs/observability
docker compose up -d

# Verify both are running
docker compose ps
```

### Step 3.3 — Wire Grafana to Prometheus

1. Open http://localhost:3000 (no login needed — anonymous admin enabled)
2. Go to **Connections → Data Sources → Add data source → Prometheus**
3. URL: `http://prometheus:9090`
4. Click **Save & Test** — should get green "Successfully queried"

### Step 3.4 — Import the Official vLLM Dashboard

```bash
# Download the official vLLM Grafana dashboard JSON
curl -o ~/labs/observability/vllm-dashboard.json \
  https://raw.githubusercontent.com/vllm-project/vllm/main/examples/online_serving/prometheus_grafana/grafana.json
```

In Grafana: **Dashboards → Import → Upload JSON file** → select `vllm-dashboard.json`
Select your Prometheus datasource. Click **Import**.

You now have: E2E latency, TTFT, TPOT, queue depth, KV cache utilization — all live.

### Step 3.5 — Generate Some Traffic to See Data

```bash
# Quick loop to generate requests
for i in $(seq 1 20); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"mlx-community/Qwen3-0.6B-4bit\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Tell me fact number $i about distributed systems.\"}],
      \"max_tokens\": 100
    }" > /dev/null &
done
wait
```

Watch the Grafana dashboard update as the 20 requests process.

### Day 3 Checkpoint ✅
- [ ] `/metrics` endpoint returning vllm-prefixed metrics
- [ ] Prometheus scraping successfully (check http://localhost:9090/targets)
- [ ] Grafana dashboard showing live data

---

## Day 4 — Load Testing with Locust

### Step 4.1 — Write the Locust Test

```python
# save as ~/labs/locustfile.py
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
    wait_time = between(0.5, 2)  # think time between requests

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
```

### Step 4.2 — Run the Load Test

```bash
cd ~/labs

# Terminal 1: Start vllm-mlx (if not already running)
vllm-mlx serve mlx-community/Qwen3-0.6B-4bit --port 8000 --continuous-batching

# Terminal 2: Run Locust
locust -f locustfile.py --host http://localhost:8000

# Open http://localhost:8089 in browser
# Start with: 5 users, spawn rate 1/s
# Then ramp to 10, then 20 — watch what breaks
```

### Step 4.3 — What to Watch in Grafana While Load Testing

Open Grafana alongside Locust. Watch these panels simultaneously:

| Grafana Panel | What it tells you |
|---|---|
| `vllm:num_requests_running` | How many requests are inflight — your batch size |
| `vllm:num_requests_waiting` | Queue depth — non-zero means you're saturated |
| `vllm:gpu_cache_usage_perc` | KV cache pressure — approaching 1.0 = OOM risk |
| `vllm:time_to_first_token_seconds` | TTFT p50/p95 — watch it degrade under load |
| `vllm:time_per_output_token_seconds` | TPOT — should stay stable even under load |

**The key insight to observe:**
As you increase users, TTFT will rise (more queuing time before prefill starts)
but TPOT stays roughly stable (once a request gets the GPU, decode is consistent).
This is exactly what P/D disaggregation is designed to fix.

### Step 4.4 — Custom PromQL Queries (Add to Grafana)

Add these panels manually in Grafana (Edit → Add panel → Prometheus query):

```promql
# Request throughput (tokens/sec)
rate(vllm:generation_tokens_total[1m])

# TTFT p95
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[2m]))

# TPOT p95
histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[2m]))

# Queue saturation ratio
vllm:num_requests_waiting / (vllm:num_requests_running + 1)

# KV cache pressure
vllm:gpu_cache_usage_perc
```

### Day 4 Checkpoint ✅
- [ ] Locust load test running with mixed short/long prompts
- [ ] Observed TTFT degradation under load in Grafana
- [ ] Confirmed TPOT stays stable (the key P/D insight)
- [ ] Queue depth visible and non-zero at saturation

---

## Day 5 — Continuous Batching Deep Dive

### Step 5.1 — Understand the Batching Config Knobs

```bash
# Restart server with explicit batching params
vllm-mlx serve mlx-community/Qwen3-0.6B-4bit \
  --port 8000 \
  --continuous-batching \
  --max-num-seqs 8        # max concurrent sequences in a batch
```

Run your Locust test again. Then change `--max-num-seqs 4` and compare:
- Lower = less memory pressure, higher individual latency, lower throughput
- Higher = more memory pressure, better throughput, higher tail latency

### Step 5.2 — Simulate KV Cache Pressure

```python
# save as ~/labs/kv_pressure.py
# Send requests with very long max_tokens to fill the KV cache
import httpx, concurrent.futures, time

def send_long_request(i):
    start = time.perf_counter()
    r = httpx.post("http://localhost:8000/v1/chat/completions",
        json={
            "model": "mlx-community/Qwen3-0.6B-4bit",
            "messages": [{"role": "user", "content": f"Write a very long essay about distributed systems topic {i}. Be extremely verbose and detailed."}],
            "max_tokens": 500,  # long output = lots of KV cache
        },
        timeout=120)
    elapsed = time.perf_counter() - start
    tokens = r.json()["usage"]["completion_tokens"]
    print(f"Request {i}: {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)")
    return elapsed

# Send 4 concurrent long requests
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(send_long_request, i) for i in range(4)]
    results = [f.result() for f in futures]
```

Watch `vllm:gpu_cache_usage_perc` in Grafana climb during this test.
When it approaches 1.0, you'll see latency spike — that's KV cache exhaustion.

### Step 5.3 — Compare Ollama vs vllm-mlx

You already have Ollama running Mistral 7B. Let's benchmark both:

```bash
# Make sure Ollama is running with mistral
ollama serve &  # if not already running

# Run vegeta load test against Ollama
echo '{"model":"mistral","messages":[{"role":"user","content":"What is 2+2?"}]}' > /tmp/ollama_body.json

vegeta attack -rate=3/s -duration=30s \
  -targets=<(echo "POST http://localhost:11434/api/chat
Content-Type: application/json
@/tmp/ollama_body.json") | vegeta report

# Now same against vllm-mlx
echo '{"model":"mlx-community/Qwen3-0.6B-4bit","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' > /tmp/vllm_body.json

vegeta attack -rate=3/s -duration=30s \
  -targets=<(echo "POST http://localhost:8000/v1/chat/completions
Content-Type: application/json
@/tmp/vllm_body.json") | vegeta report
```

**What to compare:** latency p50/p95/p99, throughput, success rate at higher concurrency.
Ollama processes requests sequentially. vllm-mlx batches them — the gap widens with concurrency.

### Day 5 Checkpoint ✅
- [ ] Observed KV cache pressure in Grafana under long-output load
- [ ] Compared `max-num-seqs` settings and their throughput/latency tradeoff
- [ ] Benchmarked Ollama vs vllm-mlx under concurrent load

---

## Day 6 — kind Cluster + vllm-mlx Behind a Service

### Why This Matters
This bridges local learning to how llm-d actually works: vLLM processes
running as K8s pods, behind a gateway/load balancer that does smart routing.

### Step 6.1 — Create a kind Cluster

```bash
# Create cluster with port-forwarding for our services
cat <<EOF > ~/labs/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30000  # vllm service
        hostPort: 9000
      - containerPort: 30001  # prometheus
        hostPort: 9091
EOF

kind create cluster --name vllm-lab --config ~/labs/kind-config.yaml
kubectl cluster-info --context kind-vllm-lab
```

### Step 6.2 — Deploy vllm-mlx as a K8s Deployment

```bash
# For local learning we use a mock vLLM-compatible server
# (vllm-mlx runs natively, not in containers, due to Metal GPU access)
# We simulate the K8s routing layer using a real OpenAI proxy

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-proxy
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
        - name: proxy
          image: nginx:alpine
          ports:
            - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  type: NodePort
  selector:
    app: vllm
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30000
EOF
```

### Step 6.3 — Use k9s to Explore

```bash
k9s --context kind-vllm-lab
```

**Keyboard shortcuts in k9s:**
- `:pods` — list all pods
- `:svc` — list services
- `d` on a pod — describe it
- `l` on a pod — tail logs
- `ctrl+k` — kill a pod (watch it restart)

Get comfortable navigating K8s resources this way — you'll use k9s constantly
when debugging llm-d InferencePools and EPP pods next week.

### Step 6.4 — Port-Forward Your Real vllm-mlx Into the Cluster Mesh Concept

While vllm-mlx can't run inside kind (Metal GPU passthrough not supported),
understand the pattern:

```
[your request]
     │
     ▼
[K8s Service: vllm-service:80]   ← In llm-d, this is the Inference Gateway (Envoy)
     │
     ▼
[K8s Pod: vllm-worker-1]          ← In llm-d, this is a vLLM pod (prefill or decode)
[K8s Pod: vllm-worker-2]
```

On cloud GPU next week, each of these pods will be a real vLLM process.
The EPP (Endpoint Picker Plugin) will replace the generic K8s load balancer
with KV-cache-aware, TTFT-aware smart routing.

### Day 6 Checkpoint ✅
- [ ] kind cluster running
- [ ] k9s navigation comfortable
- [ ] Mental model of how local vllm-mlx maps to K8s pod topology

---

## Day 7 — Review, Solidify, and Build Your Reference Dashboard

### Step 7.1 — Build a Custom Grafana Dashboard

Create a single "vLLM Learning" dashboard with exactly 6 panels:

| Panel | PromQL | Visualization |
|---|---|---|
| Requests Inflight | `vllm:num_requests_running` | Stat |
| Queue Depth | `vllm:num_requests_waiting` | Stat (alert when > 0) |
| KV Cache Usage | `vllm:gpu_cache_usage_perc` | Gauge (0–100%) |
| TTFT p50/p95 | `histogram_quantile(0.5, rate(vllm:time_to_first_token_seconds_bucket[2m]))` | Time series |
| TPOT p50/p95 | `histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[2m]))` | Time series |
| Token Throughput | `rate(vllm:generation_tokens_total[1m])` | Time series |

Export this dashboard as JSON and save to `~/labs/observability/my-vllm-dashboard.json`.
You'll import a version of this when monitoring llm-d next week.

### Step 7.2 — Document Your Findings

Fill in your own numbers from this week's experiments:

```
Model: Qwen3-0.6B-4bit on M4 Mac Mini 16GB

Baseline (single request):
  TTFT (short prompt ~20 tokens): ___ms
  TTFT (long prompt ~500 tokens): ___ms
  TPOT: ___ms
  Throughput: ___ tok/s

Under load (10 concurrent users):
  TTFT p95: ___ms
  TPOT p95: ___ms
  Max queue depth seen: ___
  KV cache peak: ___%

Ollama vs vllm-mlx (3 req/s, 30s):
  Ollama p95: ___ms
  vllm-mlx p95: ___ms
  vllm-mlx throughput advantage: ___x
```

### Step 7.3 — Connect This Week to Week 2

The gap you observed between short and long prompt TTFT is exactly what
P/D disaggregation fixes in production:

```
Week 1 (what you observed):
  Short prompt → 50ms TTFT  ✓ good
  Long prompt  → 400ms TTFT ✗ bad — prefill blocks decode queue

Week 2 (what llm-d solves):
  Long prompt → prefill pod handles it in parallel
              → decode pod is never blocked
              → TTFT stays low for ALL request types
```

Your Week 2 goal: deploy this exact pattern on a cloud GPU node with
llm-d's Helm charts and see the TTFT difference in Grafana.

---

## Quick Reference Card

### Start/Stop Everything

```bash
# vllm-mlx
source ~/.venv-vllm-mlx/bin/activate
vllm-mlx serve mlx-community/Qwen3-0.6B-4bit --port 8000 --continuous-batching

# Prometheus + Grafana
cd ~/labs/observability && docker compose up -d
cd ~/labs/observability && docker compose down

# kind cluster
kind create cluster --name vllm-lab --config ~/labs/kind-config.yaml
kind delete cluster --name vllm-lab
```

### Key URLs

| Service | URL |
|---|---|
| vllm-mlx API | http://localhost:8000/v1 |
| vllm-mlx Metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Locust UI | http://localhost:8089 |

### Important Metrics at a Glance

| Metric | Good | Warning | Critical |
|---|---|---|---|
| `num_requests_waiting` | 0 | 1–5 | > 10 |
| `gpu_cache_usage_perc` | < 0.7 | 0.7–0.9 | > 0.9 |
| TTFT p95 | < 200ms | 200–500ms | > 500ms |
| TPOT p95 | < 50ms | 50–100ms | > 100ms |

---

## Week 2 Preview — llm-d on Cloud GPU

What you'll build:
1. Single GPU node: llm-d quickstart with prefix-cache routing (K3s + Helm)
2. Two GPU nodes: P/D disaggregation — prefill pool + decode pool separated
3. Load test both configurations and compare TTFT histograms in Grafana
4. Scale decode pool independently while prefill pool stays fixed — watch TPOT improve
5. Observe EPP scheduler routing decisions in logs

Estimated cloud cost: ~$5–15 total on RunPod (L40S) for the full week of experiments.
