# gpu-labs

A structured, hands-on learning lab for LLM inference infrastructure —
from local vLLM serving on Apple Silicon to distributed P/D disaggregation
on Kubernetes using [llm-d](https://github.com/llm-d/llm-d).

Built as part of a deliberate study path into **LLM Inference Engineering**,
grounded in 11 years of distributed systems and SRE experience.

---

## What This Covers

### Week 1 — vLLM on Apple Silicon (M4 Mac Mini)
Local inference serving, observability, and load testing without any cloud spend.

| Topic | What You Learn |
|---|---|
| vllm-mlx setup | Serve any HuggingFace model on Apple Silicon via OpenAI-compatible API |
| TTFT / TPOT measurement | Manually instrument streaming responses to measure inference latency |
| Prefix cache behaviour | Observe KV cache reuse across requests with shared system prompts |
| Prometheus + Grafana | Wire vLLM's `/metrics` endpoint into a full observability stack |
| Locust load testing | Saturate the server and observe queue depth, KV cache pressure, latency degradation |
| kind + k9s | Bridge local inference to Kubernetes pod topology mental model |

### Week 2 — llm-d on Cloud GPU (Vultr / RunPod L40S)
Production-grade distributed inference on Kubernetes using the llm-d stack.

| Topic | What You Learn |
|---|---|
| K3s + GPU Operator | Bootstrap a GPU-aware Kubernetes cluster from scratch |
| llm-d three-chart pattern | infra (Envoy gateway) + inferencepool (EPP) + modelservice (vLLM pods) |
| Prefix cache routing | Deploy EPP with KV-cache-aware scoring, observe routing decisions in logs |
| P/D Disaggregation | Split prefill and decode into separate pod pools, transfer KV cache via NIXL |
| EPP scheduler internals | Read `Calculated score` logs, understand prefix/queue/kv-utilization scorers |
| Independent scaling | Scale prefill vs decode pools separately based on TTFT vs TPOT signals |
| HPA on inference metrics | Wire `vllm:num_requests_waiting` as a custom metric for autoscaling |

---

## Repository Structure

```
gpu-labs/
├── week1-vllm/
│   ├── README.md                  # Day-by-day lab guide
│   ├── measure_latency.py         # Manual TTFT/TPOT measurement via streaming
│   ├── prefix_cache_test.py       # Cache hit/miss experiment
│   ├── kv_pressure.py             # KV cache saturation under concurrent load
│   ├── locustfile.py              # Mixed short/long prompt load test
│   └── observability/
│       ├── prometheus.yml         # Scrape config targeting vLLM /metrics
│       └── docker-compose.yml     # Prometheus + Grafana stack
└── week2-llmd/
    ├── README.md                  # Day-by-day lab guide
    └── llmd_locustfile.py         # Tenant-session load test for EPP routing
```

---

## Hardware Used

| Component | Spec |
|---|---|
| Local machine | Apple M4 Mac Mini, 16GB unified memory |
| Local inference | vllm-mlx (MLX backend, Metal GPU acceleration) |
| Cloud GPU | NVIDIA L40S 48GB (Vultr / RunPod) |
| K8s local | kind |
| K8s cloud | K3s |

---

## Key Metrics Tracked

| Metric | What It Measures |
|---|---|
| `vllm:time_to_first_token_seconds` | Time from request received to first token returned (TTFT) |
| `vllm:time_per_output_token_seconds` | Per-token decode latency (TPOT) |
| `vllm:num_requests_waiting` | Queue depth — primary autoscaling signal |
| `vllm:gpu_cache_usage_perc` | KV cache pressure (alert above 0.85) |
| `llmd_kvcache_hits_total` | EPP-level cache hit count across the pool |

---

## What Comes Next

These labs are the foundation for two open-source tools in active design:

- **`llmd-pd-controller`** — A Kubernetes operator that autonomously rebalances
  prefill and decode pod ratios based on live TTFT/TPOT signals, replacing
  manual scaling with a closed-loop controller.

- **`llmd-cache-guard`** — A custom EPP scorer plugin that reserves KV cache
  slots per tenant, preventing noisy-neighbour eviction in multi-tenant
  inference clusters.

---

## Related Projects

- [inference-sentinel](https://github.com/kraghavan/inference-sentinel) — Privacy-aware LLM routing gateway with Prometheus/Grafana/Loki observability
- [schema-travels](https://github.com/kraghavan/schema-travels) — LLM-assisted database schema migration analysis
- [go-fast-token](https://github.com/kraghavan/go-fast-token) — Parallel BPE tokenizer in Go, 3M tokens/sec on M4

---

## Author

**Karthika Raghavan** — Senior Software Engineer, Distributed Systems & LLM Infrastructure  
[linkedin.com/in/karthikaraghavan](https://linkedin.com/in/karthikaraghavan)
