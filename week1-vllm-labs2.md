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

## Day 1 — Install vllm-metal and Serve Your First Model

### Why vllm-metal (not vllm-mlx)?

`vllm-metal` is the **official community plugin** under the vllm-project GitHub
organisation. It brings vLLM to Apple Silicon via MLX and Metal GPU acceleration.
`vllm-mlx` is a third-party wrapper with a broken dependency on mlx-lm>=0.31.0
as of April 2026 — avoid it.

- Official vllm-project plugin — same codebase, pinned compatible dependencies
- Uses MLX models from mlx-community on HuggingFace
- Full OpenAI-compatible API — same interface as a cloud vLLM instance
- Zero-copy tensor ops via Apple Silicon unified memory

### Step 1.1 — Install

```bash
# One-liner install script — handles venv creation and pinned deps
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# Takes ~10-15 min (builds vLLM 0.13.0 from source)
# Creates venv at: ~/.venv-vllm-metal

# Activate
source ~/.venv-vllm-metal/bin/activate

# Add alias to ~/.bash_profile (your shell on M4 Mac Mini)
echo 'alias vllm-env="source ~/.venv-vllm-metal/bin/activate"' >> ~/.bash_profile

# Verify — should print help text
vllm --help
```

### Step 1.2 — Download a Model

We use Qwen3-0.6B (4-bit quantized) — small enough for 16GB RAM, fast enough to be useful.

```bash
# Download ~400MB from HuggingFace mlx-community
huggingface-cli download mlx-community/Qwen3-0.6B-4bit
```

**Why Qwen3-0.6B?**
- Fits comfortably in 16GB (model ~400MB, KV cache grows with context)
- MLX-quantized format — runs on Metal GPU, not CPU
- Same model llm-d docs use for prefix cache routing examples

### Step 1.3 — Start the Server

```bash
source ~/.venv-vllm-metal/bin/activate

vllm serve mlx-community/Qwen3-0.6B-4bit --port 8000

# Expected startup output (normal, not errors):
# INFO: Available plugins for group vllm.platform_plugins:
# INFO:  - metal -> vllm_metal:register
# INFO: Platform plugin metal is activated
# INFO: Triton not installed or not compatible  ← HARMLESS, see note below
# INFO: Uvicorn running on http://0.0.0.0:8000  ← ready when you see this
```

> **Note on Triton warning:** Triton is NVIDIA's CUDA kernel compiler. On Apple
> Silicon, Metal replaces it entirely. This warning is printed automatically but
> has no effect — vllm-metal uses Metal kernels, not Triton. Ignore it.

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
- [ ] vllm-metal installed via install script
- [ ] `source ~/.venv-vllm-metal/bin/activate` works
- [ ] `vllm serve` running and responding to `/v1/chat/completions`

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
# save as scripts/measure_latency.py
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
pip install httpx   # if not already installed
python3 scripts/measure_latency.py
```

**What you should see:** TTFT is higher for the long prompt (more prefill work),
but TPOT stays roughly constant (decode is independent of prompt length).
This is the core insight that motivates P/D disaggregation.

### Step 2.3 — Prefix Cache in Action

```python
# save as scripts/prefix_cache_test.py
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
python3 scripts/prefix_cache_test.py
```

**Expected:** TTFT drops ~50% on the 2nd request because the system prompt tokens
are already in the KV cache. This is prefix caching — one of the key features
llm-d's EPP scheduler routes around.

### Day 2 Checkpoint ✅
- [ ] Observed streaming tokens arriving
- [ ] Measured TTFT vs TPOT difference between short/long prompts
- [ ] Observed prefix cache speedup (~50% TTFT reduction on warm requests)

---

## Day 3 — Wire Up Prometheus and Grafana

### Step 3.1 — Verify Metrics Endpoint

vllm-metal exposes Prometheus metrics by default on the same port:

```bash
# Server must be running first
curl http://localhost:8000/metrics | grep "^vllm" | head -30
```

**Key metrics to find in the output:**
```
vllm:num_requests_running      # inflight requests right now
vllm:num_requests_waiting      # queued requests
vllm:kv_cache_usage_perc       # KV cache utilization (0–1)
                               # Note: this is kv_cache not gpu_cache
vllm:time_to_first_token_seconds_bucket   # TTFT histogram
vllm:time_per_output_token_seconds_bucket # TPOT histogram
vllm:prefix_cache_hits_total   # cumulative prefix cache hits
vllm:prefix_cache_queries_total # total cache lookups
```

> **Note:** The official vLLM Grafana dashboard references `gpu_cache_usage_perc`
> but vllm-metal exposes `kv_cache_usage_perc`. Fix that panel query in Grafana
> after import.

### Step 3.2 — Docker Compose for Prometheus + Grafana

```bash
mkdir -p labs/observability
cd labs/observability
```

**Create `prometheus.yml`:**
```yaml
# labs/observability/prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - 'host.docker.internal:8000'  # reaches Mac localhost from inside Docker
```

**Create `docker-compose.yml`:**
```yaml
# labs/observability/docker-compose.yml
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
cd labs/observability
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
curl -o labs/observability/vllm-dashboard.json \
  https://raw.githubusercontent.com/vllm-project/vllm/main/examples/online_serving/prometheus_grafana/grafana.json
```

In Grafana: **Dashboards → Import → Upload JSON file** → select `vllm-dashboard.json`
Select your Prometheus datasource. Click **Import**.

**Fix the KV cache panel:** find the `gpu_cache_usage_perc` panel, edit its query to
`vllm:kv_cache_usage_perc` — this is what vllm-metal actually exposes.

### Step 3.5 — Generate Some Traffic to See Data

```bash
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
- [ ] Fixed `gpu_cache_usage_perc` → `kv_cache_usage_perc` panel query

---

## Day 4 — Load Testing with Locust

### Step 4.1 — Write the Locust Test

```python
# save as scripts/locust.py
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
```

### Step 4.2 — Run the Load Test

```bash
# Terminal 1: ensure server is running
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen3-0.6B-4bit --port 8000

# Terminal 2: run Locust
locust -f scripts/locust.py --host http://localhost:8000

# Open http://localhost:8089
# Start with: 5 users, spawn rate 1/s
# Then ramp to 10, then 20 — watch what breaks
```

### Step 4.3 — What to Watch in Grafana While Load Testing

| Grafana Panel | What it tells you |
|---|---|
| `vllm:num_requests_running` | Inflight requests — your batch size |
| `vllm:num_requests_waiting` | Queue depth — non-zero means saturated |
| `vllm:kv_cache_usage_perc` | KV cache pressure — approaching 1.0 = OOM risk |
| `vllm:time_to_first_token_seconds` | TTFT p50/p95 — watch it degrade under load |
| `vllm:time_per_output_token_seconds` | TPOT — should stay stable even under load |

**The key insight:** as users increase, TTFT rises (queuing before prefill)
but TPOT stays roughly stable (once a request gets the GPU, decode is consistent).
This is exactly what P/D disaggregation is designed to fix.

### Step 4.4 — Custom PromQL Queries (Add to Grafana)

```promql
# Request throughput (tokens/sec)
rate(vllm:generation_tokens_total[1m])

# TTFT p95
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[2m]))

# TPOT p95
histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[2m]))

# Queue saturation ratio
vllm:num_requests_waiting / (vllm:num_requests_running + 1)

# KV cache pressure (vllm-metal metric name)
vllm:kv_cache_usage_perc
```

### Day 4 Checkpoint ✅
- [ ] Locust load test running with mixed short/long prompts
- [ ] Observed TTFT degradation under load in Grafana
- [ ] Confirmed TPOT stays stable (the key P/D insight)
- [ ] Queue depth visible and non-zero at saturation

---

## Day 5 — Batching and KV Cache Deep Dive

### Step 5.1 — Understand the Batching Config Knobs

```bash
# vllm-metal uses standard vLLM flags
# Restart server with explicit concurrency limit
vllm serve mlx-community/Qwen3-0.6B-4bit \
  --port 8000 \
  --max-num-seqs 8        # max concurrent sequences in a batch
```

Run your Locust test again. Then change `--max-num-seqs 4` and compare:
- Lower = less memory pressure, higher individual latency, lower throughput
- Higher = more memory pressure, better throughput, higher tail latency

### Step 5.2 — Simulate KV Cache Pressure

```python
# save as scripts/kv_pressure.py
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

# 4 concurrent long requests — watch continuous batching run them in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(send_long_request, i) for i in range(4)]
    results = [f.result() for f in futures]

# All 4 complete at roughly the same time = continuous batching working
# Sequential would be ~4x the time of a single request
```

Watch `vllm:kv_cache_usage_perc` in Grafana climb during this test.

### Step 5.3 — Compare Ollama vs vllm-metal

```bash
# Ollama load test (must have ollama running: ollama serve)
echo '{"model":"mistral","messages":[{"role":"user","content":"What is 2+2?"}]}' > /tmp/ollama_body.json

vegeta attack -rate=3/s -duration=30s \
  -targets=<(echo "POST http://localhost:11434/api/chat
Content-Type: application/json
@/tmp/ollama_body.json") | vegeta report

# vllm-metal load test
echo '{"model":"mlx-community/Qwen3-0.6B-4bit","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}' > /tmp/vllm_body.json

vegeta attack -rate=3/s -duration=30s \
  -targets=<(echo "POST http://localhost:8000/v1/chat/completions
Content-Type: application/json
@/tmp/vllm_body.json") | vegeta report
```

Ollama processes requests sequentially. vllm-metal batches them — the gap widens with concurrency.

### Day 5 Checkpoint ✅
- [ ] Observed KV cache pressure in Grafana under long-output load
- [ ] All 4 kv_pressure requests completed simultaneously (continuous batching confirmed)
- [ ] Benchmarked Ollama vs vllm-metal — noted p50/p95 difference

---

## Day 6 — kind Cluster + nginx Reverse Proxy to vLLM

### Architecture Explanation First

**This is critical to understand before touching any YAML.**

On Apple Silicon, Metal GPU cannot be passed through to Docker containers.
This means vllm-metal **must run natively on the Mac host** — it cannot run inside
a kind pod.

So our local topology is:

```
curl localhost:9000                    (your Mac)
  → kind NodePort 30000               (kind routes into the cluster)
  → nginx pod port 80                 (proxy pod inside kind)
  → host.docker.internal:8000         (vllm-metal back on the Mac host)
```

This is an **artificial two-hop** that exists only because of Metal GPU limitations.

**In production (Week 2 with real GPU), the topology is clean and single-hop:**

```
curl gateway:80
  → Envoy gateway pod                 (load balancing, KV-cache-aware routing)
  → vLLM pod                          (FastAPI/uvicorn server running INSIDE
                                       the pod with direct GPU access)
```

vLLM runs a FastAPI server inside each pod — Envoy sits in front of it.
The nginx pod we deploy locally is simulating where Envoy sits in llm-d.
The routing lesson is identical; only the backend location differs.

### Step 6.1 — Create a kind Cluster

```bash
cat <<EOF > labs/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30000   # maps to host port 9000
        hostPort: 9000
      - containerPort: 30001
        hostPort: 9091
EOF

kind create cluster --name vllm-lab --config labs/kind-config.yaml
kubectl cluster-info --context kind-vllm-lab
```

### Step 6.2 — Deploy nginx as a Reverse Proxy to vllm-metal

This is a **working proxy** — requests entering the cluster on port 9000 are
forwarded to vllm-metal running on your Mac at port 8000.

> **IPv6 gotcha on Apple Silicon:** `host.docker.internal` inside kind resolves
> to an IPv6 address, but vllm-metal only listens on IPv4. nginx will log
> `connect() failed (101: Network unreachable)` and the proxy silently fails.
> **Fix: use the kind bridge gateway IP directly.**

```bash
# Step 1: find the host IP as seen from inside kind (do this first)
HOST_IP=$(docker inspect vllm-lab-control-plane \
  --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}')
echo "Host IP: $HOST_IP"
# Expected: 172.19.0.1  (may vary — use whatever this prints)
```

```bash
# Step 2: write the manifest using the real IP, not host.docker.internal
# Replace 172.19.0.1 below with your HOST_IP value if it differs
cat <<EOF > labs/nginx-vllm.yaml
# ConfigMap: nginx reverse proxy config
# Forwards all traffic to vllm-metal on the Mac host via IPv4
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-vllm-config
  namespace: default
data:
  default.conf: |
    server {
        listen 80;

        # Use kind bridge gateway IP directly — avoids IPv6 resolution issue
        # with host.docker.internal on Apple Silicon + kind.
        location / {
            proxy_pass http://172.19.0.1:8000;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;

            # vLLM requests can take tens of seconds — set generous timeouts
            proxy_read_timeout    300s;
            proxy_connect_timeout  10s;
            proxy_send_timeout    300s;
        }
    }
---
# Deployment: nginx pod mounting the config above
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
        - name: nginx
          image: nginx:alpine
          ports:
            - containerPort: 80
          volumeMounts:
            - name: nginx-config
              mountPath: /etc/nginx/conf.d   # overrides nginx default config
      volumes:
        - name: nginx-config
          configMap:
            name: nginx-vllm-config
---
# Service: NodePort exposes nginx pod on cluster port 30000
# kind maps host port 9000 → cluster port 30000 (via kind-config.yaml above)
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: default
spec:
  type: NodePort
  selector:
    app: vllm
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30000
EOF

# Apply everything
kubectl apply -f labs/nginx-vllm.yaml

# Wait for pod to be Running
kubectl get pods -w
# Expected: vllm-proxy-xxxx   1/1   Running
```

### Step 6.3 — Verify the Full Request Path

Make sure vllm-metal is running on port 8000 first, then:

```bash
# Step 1: verify nginx pod is healthy
kubectl get pods
kubectl logs deploy/vllm-proxy   # should show nginx startup, no errors

# Step 2: hit the models endpoint through the full K8s path
# localhost:9000 → kind → nginx pod → host vllm-metal
curl http://localhost:9000/v1/models | jq .

# Expected: same response as hitting localhost:8000 directly
# {"object":"list","data":[{"id":"mlx-community/Qwen3-0.6B-4bit",...}]}

# Step 3: full inference request through the gateway
curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "What is KV cache?"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content

# Step 4: load test through the gateway (not localhost:8000 directly)
echo '{"model":"mlx-community/Qwen3-0.6B-4bit","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20}' > /tmp/vllm_gateway.json

vegeta attack -rate=3/s -duration=15s \
  -targets=<(echo "POST http://localhost:9000/v1/chat/completions
Content-Type: application/json
@/tmp/vllm_gateway.json") | vegeta report

# Compare p50/p95 here vs direct localhost:8000 — gateway adds ~1-2ms overhead
```

### Step 6.4 — Explore with k9s

```bash
k9s --context kind-vllm-lab
```

**Keyboard shortcuts in k9s:**
- `:pods` — list all pods
- `:svc` — list services
- `:cm` — list ConfigMaps (see your nginx-vllm-config here)
- `d` on a resource — describe it
- `l` on a pod — tail logs (watch nginx access logs as you send requests)
- `ctrl+k` — kill a pod (watch it restart — K8s self-healing)

**What to look for in nginx logs:**
```
# In k9s, press 'l' on the vllm-proxy pod
# You should see lines like:
10.244.0.1 - - "POST /v1/chat/completions HTTP/1.1" 200 ...
# 200 = nginx successfully proxied to vllm-metal and got a response
```

### Step 6.5 — The Topology Pattern

```
Local setup (Apple Silicon constraint):

  curl localhost:9000
       │
       ▼ host port 9000
  kind NodePort 30000
       │
       ▼ inside cluster
  nginx pod :80  (simulates Envoy position in llm-d)
       │
       ▼ proxy_pass http://172.19.0.1:8000
  Mac host IPv4 address (kind bridge gateway)
       │
       ▼ native on Mac
  vllm-metal (Metal GPU)

  Note: host.docker.internal resolves to IPv6 inside kind on Apple Silicon —
  use the bridge gateway IP from: docker inspect vllm-lab-control-plane

Production (Week 2 — real GPU):

  curl gateway:80
       │
       ▼
  Envoy gateway pod  (EPP does KV-cache-aware routing here)
       │
       ▼ routes to selected pod
  vLLM pod  (FastAPI server + real GPU inside the pod)
```

The only difference between local and production: in production, vLLM is
**inside the pod** with direct GPU access. The gateway layer (nginx here,
Envoy in production) sits in exactly the same position in both topologies.

### Day 6 Checkpoint ✅
- [ ] kind cluster running with NodePort 9000→30000 mapping
- [ ] Host IP obtained: `docker inspect vllm-lab-control-plane --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}'`
- [ ] nginx ConfigMap deployed with `proxy_pass http://<HOST_IP>:8000` (not host.docker.internal)
- [ ] `curl http://localhost:9000/v1/models` returns model list
- [ ] Full inference request works through the gateway path
- [ ] nginx pod logs show `200` status codes (not `404` or `Network unreachable`)
- [ ] k9s navigation: can see pods, services, ConfigMaps, tail logs

> **If you see `connect() failed (101: Network unreachable)` in nginx logs:**
> `host.docker.internal` resolved to IPv6. Re-apply the ConfigMap with
> `proxy_pass http://172.19.0.1:8000` (your HOST_IP) and
> `kubectl rollout restart deployment/vllm-proxy`.

---

## Day 7 — Review and Build Your Reference Dashboard

### Step 7.1 — Build a Custom Grafana Dashboard

Create a single "vLLM Learning" dashboard with 6 panels:

| Panel | PromQL | Visualization |
|---|---|---|
| Requests Inflight | `vllm:num_requests_running` | Stat |
| Queue Depth | `vllm:num_requests_waiting` | Stat |
| KV Cache Usage | `vllm:kv_cache_usage_perc` | Gauge |
| TTFT p50/p95 | `histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[2m]))` | Time series |
| TPOT p50/p95 | `histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[2m]))` | Time series |
| Token Throughput | `rate(vllm:generation_tokens_total[1m])` | Time series |

Export as JSON → `labs/observability/my-vllm-dashboard.json`

### Step 7.2 — Record Your Benchmark Numbers

```
Model: Qwen3-0.6B-4bit on M4 Mac Mini 16GB (vllm-metal)

Baseline (single request):
  TTFT short prompt (~20 tokens): 341.8ms
  TTFT long prompt (~500 tokens): 487.3ms
  TPOT short:                      3.4ms
  TPOT long:                       4.2ms
  Throughput:                      145.8 tok/s

Prefix cache:
  Cold TTFT:   753ms
  Warm TTFT:   367ms  (51% reduction)

KV pressure (4 concurrent, 500 tokens each):
  All completed simultaneously: ~24.9s  (continuous batching confirmed)
  Per-request throughput:       20.1 tok/s

Under concurrent load (5 users, Locust):
  short_prompt p50:   1,900ms
  long_prompt p50:   31,000ms  ← the P/D disaggregation motivation

Ollama vs vllm-metal (3 req/s, 30s, similar model size):
  Ollama/qwen2.5:0.5b p50:   14,062ms
  vllm-metal/Qwen3-0.6B p50:  6,543ms
  vllm-metal advantage:        2.15x faster at p50
```

### Step 7.3 — Connect This Week to Week 2

```
Week 1 (what you observed):
  Short prompt → TTFT 341ms  ✓ good
  Long prompt  → TTFT 487ms  ← prefill work visible even on 0.6B model
  Under load   → long_prompt p50 = 31,000ms  ✗ queuing kills TTFT

Week 2 (what llm-d solves):
  Long prompt → dedicated prefill pod handles prompt processing
              → decode pod never blocked by prefill queue
              → TTFT stays low for ALL request types under load
```

---

## Quick Reference Card

### Start/Stop Everything

```bash
# vllm-metal server
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen3-0.6B-4bit --port 8000

# With concurrency limit
vllm serve mlx-community/Qwen3-0.6B-4bit --port 8000 --max-num-seqs 8

# Prometheus + Grafana
cd labs/observability && docker compose up -d
cd labs/observability && docker compose down

# kind cluster
kind create cluster --name vllm-lab --config labs/kind-config.yaml
kind delete cluster --name vllm-lab

# nginx proxy (after cluster is up)
kubectl apply -f labs/nginx-vllm.yaml
kubectl delete -f labs/nginx-vllm.yaml
```

### Key URLs

| Service | URL | Notes |
|---|---|---|
| vllm-metal direct | http://localhost:8000/v1 | Bypass gateway |
| vllm-metal via K8s | http://localhost:9000/v1 | Through nginx proxy |
| vllm-metal Metrics | http://localhost:8000/metrics | Prometheus scrape |
| Prometheus | http://localhost:9090 | |
| Grafana | http://localhost:3000 | |
| Locust UI | http://localhost:8089 | |

### Important Metrics at a Glance

| Metric | Good | Warning | Critical |
|---|---|---|---|
| `num_requests_waiting` | 0 | 1–5 | > 10 |
| `vllm:kv_cache_usage_perc` | < 0.7 | 0.7–0.9 | > 0.9 |
| TTFT p95 | < 500ms | 500ms–2s | > 2s |
| TPOT p95 | < 50ms | 50–100ms | > 100ms |

### Known Issues / Notes

| Issue | Explanation |
|---|---|
| `Triton not installed` warning | Harmless — Metal replaces Triton on Apple Silicon |
| `gpu_cache_usage_perc` shows no data | Fix panel query to `vllm:kv_cache_usage_perc` |
| vllm-mlx NoneType error | vllm-mlx broken with mlx-lm>=0.31.0 — use vllm-metal |
| nginx `Network unreachable` error | `host.docker.internal` resolved to IPv6 inside kind — use bridge IP: `docker inspect vllm-lab-control-plane --format '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}'` then set `proxy_pass http://<IP>:8000` |
| nginx 404 on `/v1/chat/completions` | ConfigMap missing or not mounted — apply labs/nginx-vllm.yaml and restart pod |
| Model re-downloads on serve | Pre-download via `huggingface-cli download <model>` first |

---

## Week 2 Preview — llm-d on Cloud GPU

What you'll build:
1. Single GPU node: llm-d quickstart with prefix-cache routing (K3s + Helm)
2. P/D disaggregation — prefill pool + decode pool on separate GPU pods
3. Load test both configurations — compare TTFT histograms in Grafana
4. Scale decode pool independently — watch TPOT improve
5. Read EPP scheduler routing decisions in logs

Estimated cloud cost: ~$15–20 total on Vultr L40S ($250 free credit covers it entirely).