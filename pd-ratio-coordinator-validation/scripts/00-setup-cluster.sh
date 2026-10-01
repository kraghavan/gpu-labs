#!/usr/bin/env bash
# Run on the rented GPU box. Brings up a single-node k3s cluster, deploys
# prefill/decode/prometheus, applies the CRD, and leaves the operator
# ready to run via `make run` in a separate terminal (see 01-run-tests.sh).
set -euo pipefail

echo "=== Installing k3s (if not already present) ==="
if ! command -v k3s >/dev/null 2>&1; then
    curl -sfL https://get.k3s.io | sh -
fi
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
export KUBECONFIG="$HOME/.kube/config"
echo "export KUBECONFIG=$HOME/.kube/config" >> ~/.bashrc

echo "=== Waiting for node Ready ==="
kubectl wait --for=condition=Ready node --all --timeout=120s

echo "=== Applying namespace, CRD, deployments, prometheus, CR ==="
kubectl apply -f k8s/namespace.yaml
kubectl apply -f "$HOME/pd-ratio-coordinator/config/crd/llmd.io_pdratiopolicies.yaml"
kubectl apply -f k8s/prefill-deployment.yaml
kubectl apply -f k8s/decode-deployment.yaml
kubectl apply -f k8s/prometheus.yaml

echo "=== Waiting for prefill/decode/prometheus pods Ready (model download can take a few minutes) ==="
kubectl wait --for=condition=Ready pod -l app=prefill -n pd-validation --timeout=300s
kubectl wait --for=condition=Ready pod -l app=decode -n pd-validation --timeout=300s
kubectl wait --for=condition=Ready pod -l app=prometheus -n pd-validation --timeout=120s

echo "=== Confirming Prometheus is actually scraping vLLM metrics ==="
kubectl port-forward -n pd-validation svc/prometheus 9090:9090 &
PF_PID=$!
sleep 3
RESULT=$(curl -s 'http://localhost:9090/api/v1/query?query=vllm:num_requests_waiting' | python3 -c "import json,sys; print(len(json.load(sys.stdin)['data']['result']))")
kill $PF_PID 2>/dev/null || true
if [ "$RESULT" = "0" ]; then
    echo "WARNING: Prometheus query returned 0 series for vllm:num_requests_waiting."
    echo "Check pod annotations and Prometheus targets at http://localhost:9090/targets before proceeding."
else
    echo "OK: Prometheus is scraping vLLM metrics ($RESULT series found)."
fi

echo "=== Applying PDRatioPolicy CR ==="
kubectl apply -f k8s/pdratiopolicy.yaml

echo ""
echo "=== Cluster ready. Next: run the controller in a separate terminal ==="
echo "cd ~/pd-ratio-coordinator && export KUBECONFIG=$HOME/.kube/config && make run"
