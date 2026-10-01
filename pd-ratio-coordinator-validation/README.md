# pd-ratio-coordinator Validation

Scripts behind [the blog post](https://kraghavan.ca) validating
[pd-ratio-coordinator](https://github.com/kraghavan/pd-ratio-coordinator)
(branch `KRAG-1`) on a rented Lambda Cloud A100 — real Prometheus metrics,
real Locust load, real `kubectl scale` actions, deliberately scoped to
skip real P/D disaggregation (NIXL/RDMA), since the operator never
inspects KV-transfer correctness, only metrics and replica counts.

## Layout

```
scripts/
  k8s/
    namespace.yaml
    runtimeclass.yaml          # nvidia RuntimeClass — see RESULTS.md for why
    prefill-deployment.yaml    # plain vLLM, shares physical GPU
    decode-deployment.yaml     # same
    prometheus.yaml            # annotation-based scrape, no Helm chart
    pdratiopolicy.yaml         # the CR under test
  locustfile.py                # DecodeLoad + PrefillBurst task sets
  00-setup-cluster.sh          # k3s install through cluster-ready
  01-run-tests.sh              # all 4 tests, adapted from the repo's own docs/testing.md
RESULTS.md                     # full results, including the honest non-passes
```

## Results summary

4 tests run, 16,559 total requests, 0 failures across all of them:

- **TPOT breach → scale decode:** passed
- **Queue velocity spike → scale prefill:** inconclusive — vLLM's
  continuous batching on a 0.6B model never produced real queue backlog,
  even at 300 concurrent users / 500+ req/s. Real limitation, not a bug.
- **GPU budget enforcement:** passed
- **Drain before scale-down:** nuanced — the drain mechanism's bookkeeping
  works, but this setup's plain Kubernetes Service has no concept of the
  `llmd.io/draining` label, so the actual routing-exclusion effect wasn't
  validated. See `RESULTS.md` for the full explanation.

Full numbers and the complete "what broke getting the cluster up" list
(6 real issues, including a RuntimeClass fix and two vLLM v0.30.0 metric
name drifts) are in `RESULTS.md`.

## Requirements

Single CUDA GPU (tested on an A100 40GB), k3s, Go 1.22+, Python 3.10+ with
`locust` (install in a venv — system Python package conflicts are likely).

## Contributing

If you find a workload that genuinely saturates prefill admission on a
small model, or want to wire up a real llm-d EPP to validate the drain
test's routing-exclusion claim properly, open an issue or a PR — both are
named as open items in the writeup.
