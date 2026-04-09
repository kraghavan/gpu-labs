# Week 2: llm-d on Cloud GPU — End-to-End Lab
> Goal: Deploy llm-d on a real GPU cluster, wire observability, understand EPP
> scheduler decisions, run P/D disaggregation, and scale prefill/decode independently.
> Estimated cloud cost: $10–20 total (RunPod L40S at ~$0.50/hr)

---

## Architecture You're Building This Week

```
Week 2, Days 1–4 (single GPU node):
  Client → Inference Gateway (Envoy) → EPP (KV-cache router) → vLLM pod
                                                                ↑
                                             prefix-cache-scorer scores each pod

Week 2, Days 5–7 (two GPU nodes):
  Client → Inference Gateway (Envoy) → EPP → decode pod (token gen, memory BW)
                                              decode pod ←NIXL KV transfer← prefill pod
                                                                              (compute BW)
```

---

## Cloud Provider Setup — RunPod

### Why RunPod for This
- Pay per second, no minimum commitment
- Network volumes persist across pod restarts (avoid re-downloading 7B models)
- Supports multi-GPU pods on a single node for P/D disaggregation simulation
- Cheaper than Lambda/Vast for L40S class GPUs

### What to Rent

| Phase | Pod Config | Approx Cost |
|---|---|---|
| Days 1–4 (prefix cache routing) | 1× L40S 48GB, Ubuntu 22.04 | ~$0.50/hr |
| Days 5–7 (P/D disaggregation) | 1× 2×L40S or 2× separate L40S pods | ~$1.00/hr |

**STOP instances when not actively working — don't leave them running overnight.**

### RunPod Setup Steps

1. Create account at runpod.io
2. Create a **Network Volume** (50GB, same region as pod) — this caches model weights
3. Deploy pod: select **RunPod PyTorch 2.4** template, attach your network volume
4. SSH into the pod: `ssh root@<pod-ip> -p <pod-port>`

---

## Day 1 — Cloud Node Bootstrap

### Step 1.1 — Verify GPU and CUDA

```bash
nvidia-smi
# Expected: L40S 48GB, CUDA 12.x

docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
# If this fails → NVIDIA container runtime not set up, see Step 1.2
```

### Step 1.2 — Install K3s (Lightweight Kubernetes)

```bash
# Install K3s — disabling Traefik (we use the llm-d gateway instead)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -

# Set up kubectl access
mkdir -p $HOME/.kube
cp /etc/rancher/k3s/k3s.yaml $HOME/.kube/config
export KUBECONFIG=$HOME/.kube/config
echo 'export KUBECONFIG=$HOME/.kube/config' >> ~/.bashrc

# Verify
kubectl get nodes
# Expected: 1 node, Ready
```

### Step 1.3 — Install NVIDIA GPU Operator

This makes Kubernetes GPU-aware — without it, pods can't request `nvidia.com/gpu`.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \   # driver already installed on RunPod image
  --wait --timeout 10m

# Verify GPU is visible to K8s (takes 2-3 min)
kubectl describe node | grep -A5 "nvidia.com/gpu"
# Expected: nvidia.com/gpu: 1 (or 2 for dual-GPU nodes)
```

### Step 1.4 — Install Supporting Tools

```bash
# Helm (already installed with K3s but verify)
helm version

# helmfile (llm-d uses this for composing charts)
wget https://github.com/helmfile/helmfile/releases/download/v0.167.1/helmfile_0.167.1_linux_amd64.tar.gz
tar xzf helmfile_*.tar.gz && mv helmfile /usr/local/bin/
helmfile --version

# yq (required by llm-d install scripts)
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
chmod +x /usr/local/bin/yq

# k9s for terminal K8s dashboard
wget https://github.com/derailed/k9s/releases/latest/download/k9s_Linux_amd64.tar.gz
tar xzf k9s_*.tar.gz && mv k9s /usr/local/bin/

# jq
apt-get install -y jq
```

### Step 1.5 — Clone llm-d repos

```bash
mkdir -p ~/llm-d && cd ~/llm-d

# Main guides and Helm charts (replaces deprecated llm-d-deployer)
git clone https://github.com/llm-d/llm-d-infra.git
git clone https://github.com/llm-d/llm-d.git

# Explore the structure — understand before deploying
ls llm-d-infra/
ls llm-d/guides/
```

**Key directories to know:**
```
llm-d/guides/
  ├── precise-prefix-cache-aware/   ← Day 2: single GPU starting point
  ├── pd-disaggregation/            ← Day 5: the main P/D event
  ├── inference-scheduling/         ← EPP scheduler config examples
  └── wide-ep-lws/                  ← Expert parallelism (week 3)

llm-d-infra/
  ├── charts/                       ← Helm charts: infra, gaie, modelservice
  └── scripts/                      ← install-prometheus-grafana.sh etc.
```

### Step 1.6 — Set Up HuggingFace Secret

```bash
export HF_TOKEN=<your-hf-token>
export NAMESPACE=llm-d-lab

kubectl create namespace $NAMESPACE

kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=$HF_TOKEN \
  -n $NAMESPACE
```

### Step 1.7 — Pull Model Weights to Network Volume

Start this now — it runs in the background while you read.

```bash
pip install huggingface_hub[cli]
huggingface-cli login

# Download to your mounted network volume (adjust path to your mount)
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir /workspace/models/qwen3-0.6b

# Also grab Llama-3.2-3B for later experiments
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir /workspace/models/llama-3.2-3b
```

### Day 1 Checkpoint ✅
- [ ] `kubectl get nodes` shows 1 Ready node
- [ ] `kubectl describe node | grep nvidia.com/gpu` shows allocatable GPU
- [ ] Both repos cloned
- [ ] HF secret created in namespace
- [ ] Model download running in background

---

## Day 2 — Deploy llm-d: Prefix Cache Routing (Single GPU)

**Why start here, not P/D disaggregation?**
Single GPU = simpler debug surface. You learn the three-chart pattern (infra + gaie + modelservice),
the gateway flow, and EPP routing without NIXL complexity. P/D disaggregation builds on this.

### The Three Helm Charts (learn this pattern — it repeats everywhere)

```
Chart 1: llm-d-infra     → Inference Gateway (Envoy + Istio or Envoy standalone)
Chart 2: inferencepool   → EPP pod + InferencePool CRD + scheduling config
Chart 3: modelservice    → vLLM pods (decode workers, optional prefill workers)
```

### Step 2.1 — Install Prometheus and Grafana First

Always install observability before the workload — you want metrics from t=0.

```bash
cd ~/llm-d/llm-d-infra

# One-liner install script included in the repo
bash scripts/install-prometheus-grafana.sh

# Verify
kubectl get pods -n llm-d-monitoring
# Expected: prometheus, grafana, alertmanager pods Running

# Port-forward Grafana to your laptop (run from your Mac, not the cloud node)
# First get the RunPod public IP/port from the RunPod dashboard
ssh -L 3000:localhost:3000 root@<runpod-ip> -p <runpod-port> \
  "kubectl port-forward -n llm-d-monitoring svc/llmd-grafana 3000:80"

# Open http://localhost:3000 on your Mac
# Default creds: admin / prom-operator
```

### Step 2.2 — Deploy the Prefix Cache Stack

```bash
export NAMESPACE=llm-d-lab
cd ~/llm-d/llm-d/guides/precise-prefix-cache-aware

# Review what you're about to deploy
cat helmfile.yaml
cat gaie-kv-events/values.yaml    # EPP config
cat ms-kv-events/values.yaml      # modelservice config

# Deploy (helmfile installs all 3 charts in order)
helmfile -n $NAMESPACE sync

# Watch pods come up
kubectl get pods -n $NAMESPACE -w
```

**Expected healthy state (takes 3–5 min for model download into pod):**
```
pod/gaie-kv-events-epp-*              1/1 Running  ← EPP scheduler
pod/infra-kv-events-inference-gateway-* 1/1 Running  ← Envoy gateway
pod/ms-kv-events-llm-d-modelservice-decode-* 2/2 Running  ← vLLM + sidecar
pod/ms-kv-events-llm-d-modelservice-decode-* 2/2 Running  ← second decode worker
```

The `2/2` on decode pods = vLLM container + KV events sidecar (what reports cache state to EPP).

### Step 2.3 — Send Your First Request Through llm-d

```bash
# Get the gateway service external IP (or use port-forward)
kubectl get svc -n $NAMESPACE | grep inference-gateway

# Port-forward the gateway to local (from your Mac)
ssh -L 8080:localhost:8080 root@<runpod-ip> -p <port> \
  "kubectl port-forward -n $NAMESPACE svc/infra-kv-events-inference-gateway-istio 8080:80"

# From your Mac — hit the gateway (not vLLM directly!)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is KV cache?"}],
    "max_tokens": 100
  }' | jq .

# Note the x-went-to-pod header — that's which decode pod the EPP chose
curl -sv http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"max_tokens":10}' \
  2>&1 | grep -i "x-went-to\|x-request-id"
```

### Step 2.4 — Import llm-d Grafana Dashboards

In Grafana (http://localhost:3000):
- Dashboards → Import → Upload JSON

Dashboard files are in the llm-d repo:
```bash
# On the cloud node
find ~/llm-d -name "*.json" | grep -i grafana
# Look for: vllm-dashboard.json, epp-dashboard.json, inference-gateway.json
```

Import all three. You now have: vLLM per-pod metrics, EPP routing metrics, gateway throughput.

### Day 2 Checkpoint ✅
- [ ] All 4 pods in `2/2` or `1/1 Running`
- [ ] Inference request returning valid response through the gateway
- [ ] Grafana showing vLLM metrics (TTFT, KV cache, queue depth)
- [ ] EPP dashboard showing scheduling activity

---

## Day 3 — Observability Deep Dive: llm-d-Specific Metrics

This is where your SRE background pays off. llm-d adds a new metric namespace
on top of vLLM — learn both layers.

### The Two Metric Layers

```
Layer 1: vllm:* metrics (from each vLLM pod)
  → TTFT, TPOT, KV cache usage, queue depth, generation tokens
  → Scraped via PodMonitor (enabled by default in modelservice chart)

Layer 2: llmd:* metrics (from EPP)
  → Cache hit ratios per pool, routing decisions, scheduling latency
  → Scraped via ServiceMonitor on EPP service
```

### Step 3.1 — Key llm-d PromQL Queries

Add these panels to a custom Grafana dashboard named "llm-d SRE View":

```promql
# === TTFT — the primary SLO for llm-d ===
# Per-pod TTFT p95 (compare prefill vs decode pods)
histogram_quantile(0.95,
  sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le, pod)
)

# === TPOT — decode health ===
histogram_quantile(0.95,
  sum(rate(vllm:time_per_output_token_seconds_bucket[2m])) by (le, pod)
)

# === KV Cache hit ratio (the EPP's raison d'être) ===
sum(rate(llmd_kvcache_hits_total[5m])) by (pool)
/
sum(rate(llmd_kvcache_requests_total[5m])) by (pool)

# === Queue saturation per pod ===
vllm:num_requests_waiting by (pod)

# === Token throughput per pod ===
sum(rate(vllm:generation_tokens_total[1m])) by (pod)

# === EPP scheduling latency ===
histogram_quantile(0.95, rate(inference_extension_request_total[2m]))

# === GPU KV cache utilization (watch this stay below 0.9) ===
vllm:gpu_cache_usage_perc by (pod)
```

### Step 3.2 — Read EPP Scheduler Logs Live

The EPP logs every routing decision at debug level. This is the X-ray for understanding
how KV-cache-aware routing works:

```bash
# Stream EPP routing decisions (run on cloud node)
kubectl logs -l inferencepool=gaie-kv-events-epp \
  -n $NAMESPACE --tail 100 -f \
  | grep "Calculated score"

# What you'll see per request (one line per candidate pod):
# {"msg":"Calculated score",
#  "x-request-id":"...",
#  "plugin":"precise-prefix-cache-scorer",
#  "endpoint":{"name":"decode-pod-A"},
#  "score":1}    ← this pod has a KV cache hit
# {"msg":"Calculated score",
#  "plugin":"precise-prefix-cache-scorer",
#  "endpoint":{"name":"decode-pod-B"},
#  "score":0}    ← this pod has no cached prefix → EPP picks pod-A
```

**Experiment: force a cache hit vs miss**

Send the same long system prompt twice, watch the score flip:
```bash
LONG_SYSTEM="You are an expert distributed systems engineer with 20 years of experience. " \
  "You specialize in Kubernetes, Kafka, real-time data pipelines, and LLM inference. "

# Request 1: cold cache — score:0 on both pods, random pick
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"system\",\"content\":\"$LONG_SYSTEM\"},{\"role\":\"user\",\"content\":\"What is Kafka?\"}],\"max_tokens\":50}"

# Request 2: same prefix — score:1 on the pod that served request 1
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"system\",\"content\":\"$LONG_SYSTEM\"},{\"role\":\"user\",\"content\":\"What is Kubernetes?\"}],\"max_tokens\":50}"

# Watch EPP logs during both — you'll see score change from 0 to 1
```

### Step 3.3 — Understand the Three EPP Scorer Plugins

```
Plugin 1: prefix-cache-scorer
  Input: prompt token hash vs KV cache block index per pod
  Output: 0 (cold) or 1+ (warm, proportional to matched blocks)
  Effect: Routes repeated system prompts to the same pod → reuse cached KV

Plugin 2: queue-scorer
  Input: vllm:num_requests_waiting per pod
  Output: higher score for lower queue depth
  Effect: Load balances away from saturated pods

Plugin 3: kv-cache-utilization-scorer
  Input: vllm:gpu_cache_usage_perc per pod
  Output: penalizes pods approaching KV cache exhaustion
  Effect: Prevents OOM by routing away before 0.9 threshold
```

These three scorers run for every request, scores are summed, highest-score pod wins.
This is an SRE-owned scheduling system — you can tune weights.

### Step 3.4 — Wire OpenTelemetry Tracing (Optional but Powerful)

llm-d supports end-to-end tracing: Gateway → EPP → vLLM worker.

```bash
# One-liner from llm-d-infra (installs OTel collector + Jaeger)
bash ~/llm-d/llm-d-infra/scripts/install-otel-collector-jaeger.sh

# Port-forward Jaeger UI
kubectl port-forward -n llm-d-monitoring svc/jaeger-query 16686:16686

# From Mac:
ssh -L 16686:localhost:16686 root@<runpod-ip> -p <port> \
  "kubectl port-forward -n llm-d-monitoring svc/jaeger-query 16686:16686"
# Open http://localhost:16686 → trace individual requests end-to-end
```

### Day 3 Checkpoint ✅
- [ ] Custom "llm-d SRE View" Grafana dashboard with all 6 PromQL panels
- [ ] EPP scheduler scoring logs readable and interpretable
- [ ] Confirmed cache hit/miss experiment: score:0 cold → score:1 warm
- [ ] (Optional) Jaeger traces showing request path through all components

---

## Day 4 — Load Testing llm-d and Observing Smart Routing

### Step 4.1 — Install Locust on Cloud Node (or Run from Mac)

```bash
# On cloud node or your Mac (Mac is fine — it hits the port-forwarded gateway)
pip install locust

# Save as ~/labs/llmd_locustfile.py
cat > ~/labs/llmd_locustfile.py << 'EOF'
import random
from locust import HttpUser, task, between

MODEL = "Qwen/Qwen3-0.6B"
GATEWAY = "http://localhost:8080"

# Simulate 3 different "tenants" each with their own system prompt
TENANTS = [
    "You are a financial analyst assistant specializing in market data. " * 5,
    "You are a DevOps engineer assistant specializing in Kubernetes and CI/CD. " * 5,
    "You are a data scientist assistant specializing in ML pipelines. " * 5,
]

class LLMDUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        # Each simulated user picks a "tenant" at start — keeps same system prompt
        self.tenant_prompt = random.choice(TENANTS)

    @task(4)
    def tenant_request(self):
        """Simulates session-affine requests — same system prompt = cache hits"""
        self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": self.tenant_prompt},
                {"role": "user", "content": random.choice([
                    "Summarize the key points.", "What should I focus on?",
                    "Give me 3 recommendations.", "Explain the tradeoffs.",
                ])}
            ],
            "max_tokens": 80,
        }, name="tenant_session")

    @task(1)
    def cold_request(self):
        """Random prompt — no prefix cache benefit"""
        self.client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": f"Random question {random.randint(1,1000)}"}],
            "max_tokens": 50,
        }, name="cold_request")
EOF
```

### Step 4.2 — Run the Load Test

```bash
# From your Mac (hitting the port-forwarded gateway)
locust -f ~/labs/llmd_locustfile.py --host http://localhost:8080

# Open http://localhost:8089
# Start: 10 users, spawn 2/s
# Let it run 5 min, then ramp to 20 users
```

### Step 4.3 — What to Watch Simultaneously

Open 3 windows: Locust UI, Grafana, EPP logs.

**In Grafana, watch:**
- KV cache hit ratio — should INCREASE as tenant sessions warm up (target: >0.5)
- TTFT p95 — should be lower than a vanilla round-robin baseline
- KV cache utilization per pod — should trend toward the "hot" pod for each tenant

**In EPP logs:**
```bash
# Count score:1 vs score:0 decisions — hit ratio should climb
kubectl logs -l inferencepool=gaie-kv-events-epp -n $NAMESPACE -f \
  | grep "Calculated score" \
  | awk -F'"score":' '{print $2}' \
  | tr -d '}' \
  | sort | uniq -c
```

**The expected pattern:**
- First 2 min: mostly score:0 (cold caches) → EPP does random-ish routing
- After 2 min: score:1 appears for tenant sessions → routing becomes sticky to warm pod
- Watch TTFT p95 drop as cache hit ratio rises — this is the llm-d value proposition

### Step 4.4 — Baseline Comparison: Smart Routing vs Round-Robin

```bash
# Temporarily bypass EPP and hit vLLM pods directly (round-robin via K8s service)
# Port-forward the raw vLLM service (not the gateway)
kubectl port-forward svc/ms-kv-events-llm-d-modelservice-decode 8081:8000 -n $NAMESPACE

# Run same locust test against :8081 (raw round-robin)
# Compare p95 TTFT between :8080 (EPP-routed) vs :8081 (round-robin)
# Expected: EPP route is 20-40% faster on tenant_session tasks after warmup
```

### Day 4 Checkpoint ✅
- [ ] Locust running 20 concurrent users through llm-d gateway
- [ ] KV cache hit ratio visible in Grafana and trending upward
- [ ] EPP score:1 decisions increasing over test duration
- [ ] Measured TTFT difference: smart routing vs round-robin

---

## Day 5 — P/D Disaggregation: The Main Event

### Conceptual Review Before Touching YAML

What changes from prefix-cache-routing → P/D disaggregation:

```
Prefix cache (Days 2-4):        P/D Disaggregation (Days 5-7):

Request → EPP → decode-pod-A    Request → EPP → decode-pod-A
               (handles full              decode-pod-A sends x-prefiller-url header
                prefill+decode)           → prefill-pod-X processes prompt → KV cache
                                         → transfers KV via NIXL sidecar to decode-pod-A
                                         → decode-pod-A generates tokens
```

The `2/2` on decode pods you saw earlier = `vLLM container + NIXL sidecar`.
The sidecar is what receives KV cache transfers from prefill pods over RDMA/TCP.

### Step 5.1 — Understand the P/D Values Files

```bash
cd ~/llm-d/llm-d/guides/pd-disaggregation

# Read these carefully before deploying
cat helmfile.yaml
cat gaie-pd/values.yaml    # EPP config — note pd-disaggregation-handler plugin
cat ms-pd/values.yaml      # modelservice — note prefill: and decode: sections
```

**Key things in `ms-pd/values.yaml` to understand:**

```yaml
# Prefill workers: compute-bound, fewer needed
prefill:
  replicaCount: 2           # 2 prefill pods
  resources:
    limits:
      nvidia.com/gpu: 1

# Decode workers: memory-bandwidth-bound, scale these
decode:
  replicaCount: 1           # 1 decode pod
  resources:
    limits:
      nvidia.com/gpu: 1

# The threshold that decides whether to disaggregate
env:
  - name: PD_PROMPT_LEN_THRESHOLD
    value: "1000"           # requests with >1000 prompt tokens → use P/D split
                            # shorter requests → single worker handles both
```

### Step 5.2 — Tear Down Day 2 Stack, Deploy P/D Stack

```bash
# Tear down prefix cache stack
cd ~/llm-d/llm-d/guides/precise-prefix-cache-aware
helmfile -n $NAMESPACE destroy

# Deploy P/D stack
cd ~/llm-d/llm-d/guides/pd-disaggregation
helmfile -n $NAMESPACE sync

# Watch pods — you'll now see SEPARATE prefill and decode pods
kubectl get pods -n $NAMESPACE -w
```

**Expected healthy state:**
```
pod/gaie-pd-epp-*                           1/1 Running  ← EPP with P/D handler plugin
pod/infra-pd-inference-gateway-*            1/1 Running  ← Envoy gateway
pod/ms-pd-llm-d-modelservice-decode-*      2/2 Running  ← decode + NIXL sidecar
pod/ms-pd-llm-d-modelservice-prefill-*     1/1 Running  ← prefill worker 1
pod/ms-pd-llm-d-modelservice-prefill-*     1/1 Running  ← prefill worker 2
pod/ms-pd-llm-d-modelservice-prefill-*     1/1 Running  ← prefill worker 3
pod/ms-pd-llm-d-modelservice-prefill-*     1/1 Running  ← prefill worker 4
```

More prefill pods than decode: prefill is parallelizable (stateless per request),
decode is stateful (must hold KV cache for its active sequences).

### Step 5.3 — Verify P/D Routing Is Happening

```bash
# Watch EPP logs — you should see pd-disaggregation-handler decisions
kubectl logs -l inferencepool=gaie-pd-epp -n $NAMESPACE -f \
  | grep -E "disaggregat|prefiller|x-prefiller"

# Send a SHORT prompt — should NOT disaggregate (below threshold)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}' \
  | jq .usage

# Send a LONG prompt (>1000 tokens) — SHOULD disaggregate
LONG_PROMPT=$(python3 -c "print('Analyze the following text in great detail: ' + ('The quick brown fox jumps over the lazy dog. ' * 60))")

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"meta-llama/Llama-3.2-3B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROMPT\"}],\"max_tokens\":50}" \
  | jq .usage
```

### Step 5.4 — Measure the P/D Benefit

```python
# save as ~/labs/pd_benchmark.py — run from your Mac
import time, httpx, json

GATEWAY = "http://localhost:8080"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"

def measure_ttft(prompt, label, max_tokens=50):
    start = time.perf_counter()
    first_token = None

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{GATEWAY}/v1/chat/completions",
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens,
                  "stream": True},
            headers={"Content-Type": "application/json"}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk["choices"][0]["delta"].get("content") and first_token is None:
                        first_token = time.perf_counter()
                        break

    ttft = (first_token - start) * 1000 if first_token else -1
    print(f"{label}: TTFT={ttft:.0f}ms")
    return ttft

SHORT = "What is Kubernetes?"
LONG = "Analyze and summarize: " + ("The quick brown fox jumps over the lazy dog. " * 60)

print("=== Short prompts (below threshold — no disaggregation) ===")
for i in range(3):
    measure_ttft(SHORT, f"Short request {i+1}")

print("\n=== Long prompts (above threshold — P/D disaggregation) ===")
for i in range(3):
    measure_ttft(LONG, f"Long request {i+1}")

print("""
Expected:
  Short prompts: moderate TTFT (single worker, fast prefill)
  Long prompts: TTFT should be LOWER than equivalent aggregated serving
  because prefill runs in parallel with previous decode, not blocking it.
""")
```

### Day 5 Checkpoint ✅
- [ ] P/D stack deployed with 4 prefill pods + 1 decode pod
- [ ] EPP logs showing disaggregation decisions for long prompts
- [ ] Measured TTFT: long prompt with P/D vs short prompt without P/D
- [ ] Verified threshold behavior: short prompts bypass disaggregation

---

## Day 6 — Scale Prefill and Decode Independently

This is the core SRE skill for llm-d production operations.

### Step 6.1 — Scale Decode Pods and Watch TPOT

```bash
# Start with 1 decode pod (default)
kubectl get deploy -n $NAMESPACE | grep decode

# Run load test while watching TPOT
locust -f ~/labs/llmd_locustfile.py --host http://localhost:8080 &
# Start 20 users in the UI

# Watch TPOT p95 in Grafana — it will rise as decode is saturated

# Scale decode to 2 pods (if you have 2 GPUs, or use time-slicing on 1 GPU)
kubectl scale deployment ms-pd-llm-d-modelservice-decode \
  --replicas=2 -n $NAMESPACE

# Watch TPOT drop — decode is now less saturated
```

### Step 6.2 — Scale Prefill Pods and Watch TTFT

```bash
# Default is 4 prefill pods. Let's test with fewer to see TTFT impact.

# Scale DOWN prefill (simulate prefill bottleneck)
kubectl scale deployment ms-pd-llm-d-modelservice-prefill \
  --replicas=1 -n $NAMESPACE

# Run long-prompt load test
locust -f ~/labs/llmd_locustfile.py --host http://localhost:8080
# Use LONG prompts only — change locustfile to only use the long_request task

# Watch TTFT p95 rise in Grafana — prefill is now the bottleneck

# Scale prefill back up
kubectl scale deployment ms-pd-llm-d-modelservice-prefill \
  --replicas=4 -n $NAMESPACE

# Watch TTFT recover
```

### Step 6.3 — The Ratio Game (Core Scaling Intuition)

Use this decision table when observing your Grafana dashboard:

| Symptom | Root Cause | Action |
|---|---|---|
| High TTFT, low TPOT | Prefill bottleneck | Add prefill pods |
| Low TTFT, high TPOT | Decode bottleneck | Add decode pods |
| Both high | KV cache saturation or GPU memory | Scale out both or reduce batch size |
| TTFT spiky, TPOT stable | Prefill queue bursting | Add prefill pods OR raise PD_PROMPT_LEN_THRESHOLD |
| KV cache > 0.9 | Decode pod OOM risk | Add decode pods immediately |

**Record your observations in your notes table from Week 1 Day 7.**

### Step 6.4 — Wire HPA for Automatic Scaling

```bash
# Create Prometheus adapter to expose vLLM metrics as K8s custom metrics
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace llm-d-monitoring \
  -f - <<EOF
rules:
  custom:
    - seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "vllm:num_requests_waiting"
        as: "llm_queue_depth"
      metricsQuery: 'avg(vllm:num_requests_waiting{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
EOF

# Create HPA for decode pods — scale on queue depth
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: decode-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ms-pd-llm-d-modelservice-decode
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Pods
      pods:
        metric:
          name: llm_queue_depth
        target:
          type: AverageValue
          averageValue: "3"   # scale up when avg queue depth > 3 per pod
EOF

# Watch the HPA in action
kubectl get hpa -n $NAMESPACE -w
```

### Day 6 Checkpoint ✅
- [ ] Scaled decode pods and observed TPOT improvement in Grafana
- [ ] Scaled prefill pods and observed TTFT change under long-prompt load
- [ ] Can correctly diagnose: "this metric means this bottleneck"
- [ ] HPA deployed and responding to queue depth changes

---

## Day 7 — Review, Debug Patterns, and Week 3 Preview

### Step 7.1 — Common llm-d Failure Modes and How to Debug Them

**Failure 1: Decode pod stuck in Pending**
```bash
kubectl describe pod <decode-pod> -n $NAMESPACE
# Look for: "0/1 nodes are available: insufficient nvidia.com/gpu"
# Fix: GPU operator not installed, or node has 0 allocatable GPUs
kubectl describe node | grep -A10 Allocatable
```

**Failure 2: NIXL KV transfer timeouts (P/D disaggregation)**
```bash
# Check NIXL sidecar logs on decode pod
kubectl logs <decode-pod> -c nixl-sidecar -n $NAMESPACE | tail -50
# Look for: connection refused, RDMA errors, timeout
# Cause: prefill pod can't reach decode pod's NIXL port
kubectl get svc -n $NAMESPACE | grep nixl
```

**Failure 3: EPP routing all traffic to one pod**
```bash
# Check queue-scorer is working
kubectl logs -l inferencepool=gaie-pd-epp -n $NAMESPACE \
  | grep "queue-scorer" | tail -20
# If all scores are equal: queue metric not being scraped
kubectl get servicemonitor -n $NAMESPACE
```

**Failure 4: KV cache hit ratio stuck at 0**
```bash
# KV events sidecar not reporting to EPP
kubectl logs <decode-pod> -c kv-events-sidecar -n $NAMESPACE
# EPP not receiving events
kubectl logs -l inferencepool=gaie-kv-events-epp -n $NAMESPACE \
  | grep -i "kv.event\|cache.event"
```

**Failure 5: Model download loop (pod keeps restarting)**
```bash
kubectl logs <prefill-pod> -n $NAMESPACE | tail -30
# Look for: 401 Unauthorized, invalid HF token, disk full
kubectl get secret llm-d-hf-token -n $NAMESPACE -o jsonpath='{.data.HF_TOKEN}' | base64 -d
```

### Step 7.2 — Your Week 2 Benchmark Table

Fill this in from your Grafana data:

```
Deployment: llm-d P/D Disaggregation on RunPod L40S 48GB
Model: meta-llama/Llama-3.2-3B-Instruct

Prefix Cache Routing (Day 4):
  TTFT p95 (cold cache): ___ms
  TTFT p95 (warm cache): ___ms
  Cache hit ratio at 20 users: ___%
  Token throughput: ___ tok/s

P/D Disaggregation (Day 5-6):
  TTFT p95 (short prompt, no disagg): ___ms
  TTFT p95 (long prompt, with disagg): ___ms
  TPOT p95 at 1 decode pod: ___ms
  TPOT p95 at 2 decode pods: ___ms
  Prefill pod count at which TTFT stops improving: ___

Autoscaler (Day 6):
  Time from queue spike to scale-up: ___s
  Queue depth threshold that triggered scale: ___
```

### Step 7.3 — STOP Your RunPod Instances

Before anything else — stop/terminate pods to avoid billing:
```
RunPod dashboard → My Pods → Stop (or Terminate if done)
Keep the Network Volume — it has your downloaded models
```

### Step 7.4 — Week 3 Preview

What comes after P/D disaggregation:

**Option A: Wide Expert-Parallelism (MoE models)**
Deploy DeepSeek-R1 or a Mixture-of-Experts model using llm-d's wide-EP mode.
Each GPU handles a subset of experts. The llm-d guide is `guides/wide-ep-lws/`.
Requires: 4–8 GPUs. Cost: ~$2–4/hr on RunPod.

**Option B: KV Cache Tiering**
Configure the `llm-d filesystem backend` to offload cold KV cache blocks from
GPU RAM → CPU RAM → NVMe, using `guides/kv-cache-tiering/`.
Useful for long-context workloads. Works on single GPU.

**Option C: Scale-to-Zero Autoscaling (v0.5 feature)**
Configure the workload variant autoscaler to scale decode pods to zero during
idle periods and back up on traffic. New in v0.5.

**Option D: Bring This Back to threatgraph**
Your stateful multi-agent system has inference embedded in its loop.
llm-d as the serving layer for your Mistral 7B agents, with session
stickiness routing agent calls to warm KV cache = direct TTFT reduction
for your threat classification pipeline.

---

## Quick Reference Card — Week 2

### Key Commands

```bash
# Deploy a guide
cd ~/llm-d/llm-d/guides/<guide-name>
helmfile -n $NAMESPACE sync

# Tear down
helmfile -n $NAMESPACE destroy

# Watch all pods
kubectl get pods -n $NAMESPACE -w
k9s -n $NAMESPACE

# EPP routing decisions
kubectl logs -l inferencepool=<epp-label> -n $NAMESPACE -f | grep "Calculated score"

# Scale workers
kubectl scale deployment ms-pd-llm-d-modelservice-decode --replicas=2 -n $NAMESPACE
kubectl scale deployment ms-pd-llm-d-modelservice-prefill --replicas=4 -n $NAMESPACE

# Port-forward everything (from Mac)
ssh -L 8080:localhost:8080 -L 3000:localhost:3000 -L 9090:localhost:9090 \
  root@<runpod-ip> -p <port> \
  "kubectl port-forward -n $NAMESPACE svc/infra-pd-inference-gateway-istio 8080:80 &
   kubectl port-forward -n llm-d-monitoring svc/llmd-grafana 3000:80 &
   kubectl port-forward -n llm-d-monitoring svc/llmd-kube-prometheus-stack-prometheus 9090:9090 &
   wait"
```

### Helm Chart → Component Mapping

| Chart | Creates | Purpose |
|---|---|---|
| `llm-d-infra` | Inference Gateway (Envoy/Istio pod) | Traffic entry point |
| `inferencepool` | EPP pod + InferencePool CRD | Smart routing/scoring |
| `modelservice` | vLLM pods (prefill + decode) | Actual inference |

### Key URLs (after port-forwarding)

| Service | URL | Notes |
|---|---|---|
| llm-d Gateway | http://localhost:8080/v1 | Send inference requests here |
| Grafana | http://localhost:3000 | admin / prom-operator |
| Prometheus | http://localhost:9090 | PromQL exploration |
| Jaeger (optional) | http://localhost:16686 | Distributed traces |
| Locust | http://localhost:8089 | Load test control |

### Scheduling Plugins Cheat Sheet

| Plugin | Scores based on | Fixes |
|---|---|---|
| `prefix-cache-scorer` | Cached KV block match count | Low cache hit ratio |
| `queue-scorer` | Pending request count | Uneven load distribution |
| `kv-cache-utilization-scorer` | GPU KV cache % used | Pod OOM events |
| `pd-disaggregation-handler` | Prompt length vs threshold | TTFT on long prompts |
| `slo-scorer` (experimental) | Predicted TTFT/TPOT | Per-request SLO enforcement |