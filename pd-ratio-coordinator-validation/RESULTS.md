# pd-ratio-coordinator Validation — Results

**Date:** 2026-10-01. **Hardware:** 1x A100 (40GB SXM4), Lambda Cloud.
**Setup:** single-node k3s, plain vLLM Deployments labeled `prefill`/`decode`
(Qwen3-0.6B, sharing the physical GPU via `--gpu-memory-utilization=0.2`
each, `runtimeClassName: nvidia`), annotation-based Prometheus, controller
run via `make run` on the host (not in-cluster).

## Repo state before this session

~1,600 lines of real Go, but did not compile as committed: a DeepCopy type
bug (`register.go`), unqualified `Bottleneck*` constants (`decode.go`), and
a missing `go.sum`. All three fixed and pushed to `KRAG-1` prior to this
GPU session. `config/crd/` also didn't exist despite the README instructing
users to apply it — generated and pushed.

## What broke getting the cluster up (kept in full)

1. **No GPU access at all.** Deliberately skipped `nvidia.com/gpu` resource
   requests (to let multiple replicas share one physical GPU), which also
   meant containerd never injected GPU device access into the containers —
   `RuntimeError: Failed to infer device type`. Fixed with a Kubernetes
   `RuntimeClass` object (`handler: nvidia`) and `runtimeClassName: nvidia`
   on the prefill/decode pod specs, rather than forcing nvidia as the
   cluster-wide containerd default (which would also apply to Prometheus,
   and which hit a hard TOML conflict — see next item).
2. **`containerd: failed to unmarshal TOML: toml: table containerd already
   exists`**, then `table nvidia already exists` after a first fix attempt.
   k3s on this GPU-ready Lambda box auto-detects `nvidia-container-runtime`
   and already configures a working `nvidia` runtime handler in its
   generated containerd config — no custom template was needed at all.
   Removed the custom `config.toml.tmpl` entirely once this was confirmed.
3. **Stale containerd-shim processes block a clean k3s restart.** Same
   "child survives killing the parent" pattern as the zombie EngineCore
   issue from the earlier vLLM experiment, now in k3s/containerd. Fixed
   with `k3s-killall.sh` before restarting the service.
4. **In-cluster service DNS doesn't resolve from the host.** Running the
   controller via `make run` outside the cluster meant
   `http://prometheus.pd-validation:9090` (the default-style URL) couldn't
   resolve. Fixed with a persistent `kubectl port-forward` and pointing
   `prometheusURL` at `http://localhost:9090`.
5. **`go run` spawns a child binary that survives killing the parent.**
   Same zombie-child pattern a third time, now in Go's own tooling —
   `pkill -f 'go run main.go'` didn't kill the actual compiled binary at
   `/tmp/go-build.../exe/main`. Had to find and kill it by PID directly.
6. **Two real metric-name drifts in vLLM v0.30.0**, found by comparing the
   operator's hardcoded PromQL against the actual `/metrics` output:
   - `vllm:gpu_cache_usage_perc` → renamed to `vllm:kv_cache_usage_perc`
     (the *same* rename already documented as a gotcha in Part 2 of the
     blog series, on an earlier vLLM version — this has been drifting for
     a while).
   - `vllm:time_per_output_token_seconds` → renamed to
     `vllm:request_time_per_output_token_seconds`.
   Both fixed in `internal/metrics/prometheus.go`, rebuilt, redeployed.

## Test results

### Test 1 — TPOT SLO breach → scale decode: **PASSED**

Real detection, real cooldown enforcement, real scale action.

- Load: 30 users, 60s, against decode. **1,970 requests, 0 failures**,
  p50=530ms, p95=560ms (Locust-measured HTTP latency, not vLLM's internal
  TPOT).
- Measured `decode.tpot_p95_ms`: **9.5ms** at steady state under this load
  — far below the first "aggressive-sounding" 20ms SLO we set, which never
  triggered. Had to lower the SLO to 5ms (below the measured baseline) to
  get a clean, deterministic breach signal. Stated plainly: this is a
  hand-tuned threshold to exercise the code path, not a realistic
  production SLO for this model/hardware pairing.
- Controller log: `tpot_slo_breach(10ms>5ms)` → after the 60s cooldown
  cleared → `"scaled decode", "from": 1, "to": 2, "bottleneck": "decode"`.
  A real second decode pod was created.

### Test 2 — Queue velocity spike → scale prefill: **INCONCLUSIVE (real limitation, not a bug)**

- Escalated three times: 20 users/20s, 150 users/15s, 300 users/25s.
- Final run: **12,517 requests, 0 failures**, 520.7 req/s sustained,
  p50=470ms, p95=560ms, p99=680ms.
- `prefill.queue` (`vllm:num_requests_waiting`) stayed at **0 for the
  entire duration of all three runs**, confirmed by both the controller's
  10s reconcile snapshots and a direct 2-second-interval Prometheus poll
  during the final run.
- **Why, most likely:** vLLM's continuous batching on a 0.6B model with
  `gpu-memory-utilization=0.2` admits requests into the running batch fast
  enough that nothing backs up into a "waiting" state, even at 300
  concurrent requests and 500+ req/s. This is a property of the
  model/hardware combination, not of the detection logic — `AnalysePrefill`
  in `prefill.go` is already covered by passing unit tests with synthetic
  queue data.
- **Open item for a v2 validation round:** either a larger model (where
  prefill compute per request is high enough to genuinely queue) or a
  deliberately resource-constrained vLLM config (e.g. a very low
  `--max-num-seqs`) would be needed to produce real backlog and exercise
  this path operationally.

### Test 3 — GPU budget enforcement: **PASSED**

- Budget set equal to current total (prefill=1, decode=2, budget=3).
- Load: 30 users, 40s, against decode. **1,376 requests, 0 failures**,
  p50=530ms, p95=550ms.
- Real pressure confirmed: `decode.under_pressure: true`,
  `tpot_slo_breach(10ms>5ms)`.
- Decode replicas stayed at 2 the entire time — the controller correctly
  refused to scale past the budget even with genuine, sustained pressure
  present. `prefill + decode <= gpuBudget` held throughout (1+2=3≤3).

### Test 4 — Drain before scale-down: **NUANCED — real finding, not a clean pass**

- Load: 8 users, 70s, against decode. **696 requests, 0 failures**,
  p50=470ms, p95=480ms.
- Budget cut from 3 to 2 mid-flight to force a scale-down.
- `llmd.io/draining=true` label was correctly applied to the targeted pod.
- **But:** the drain timed out after its configured 20s window — controller
  log: `"drain timeout, force scaling down", "error": "drain timeout after
  20s for pod decode-699544896-ctffv"` — then force-scaled down anyway
  (`"scaled decode", "from": 2, "to": 1`).
- **The more important finding:** this validation setup uses a plain
  Kubernetes `Service` (kube-proxy round-robin), not llm-d's actual EPP
  gateway. The `llmd.io/draining` label is designed to be read by that EPP
  to stop routing new requests to the draining pod — a plain Service has
  no concept of this label at all. So while the drain mechanism's *write
  path* (apply label, poll running-request count, force-scale on timeout)
  is confirmed to execute correctly end-to-end, its *actual protective
  effect* was not validated here, because half of the mechanism (routing
  exclusion) isn't present in this de-risked setup.
- The zero-failures result is good news but should not be read as proof
  the drain protects production traffic — it may simply reflect that this
  particular test's light load (8 concurrent users) didn't happen to have
  an in-flight request on the targeted pod at the moment it was
  force-killed. Real validation of this specific guarantee needs llm-d's
  EPP in front of the pods — explicitly out of this round's scope (see
  design plan's scope decision).

## Summary table

| Test | Result | Requests | Failures | p50 | p95 |
|---|---|---|---|---|---|
| 1. TPOT breach → scale decode | **Passed** | 1,970 | 0 | 530ms | 560ms |
| 2. Velocity spike → scale prefill | **Inconclusive** (real limitation) | 12,517 | 0 | 470ms | 560ms |
| 3. GPU budget enforcement | **Passed** | 1,376 | 0 | 530ms | 550ms |
| 4. Drain before scale-down | **Nuanced** (write path works, protection unverified) | 696 | 0 | 470ms | 480ms |

**Total requests across all tests: 16,559. Total failures: 0.**

## Cost

Session total: instance rented for this round, A100 at $1.99/hr. Exact
duration/cost should be confirmed against the Lambda console directly —
not tracked to the second in this log.
