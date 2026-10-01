#!/usr/bin/env bash
# Run after 00-setup-cluster.sh. Starts the controller in the background
# (logs to controller.log), runs all 4 tests from docs/testing.md adapted
# to this plain-Service setup, and reports results. Run from the
# pd-ratio-coordinator repo root's parent, or adjust REPO_DIR below.
set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
REPO_DIR="${REPO_DIR:-$HOME/pd-ratio-coordinator}"
RESULTS_DIR="$(pwd)/../results"
mkdir -p "$RESULTS_DIR"

echo "=== Starting controller in background (log: controller.log) ==="
(cd "$REPO_DIR" && nohup make run > "$RESULTS_DIR/controller.log" 2>&1 &)
CONTROLLER_PID=$!
sleep 5
echo "Controller started. Watch $RESULTS_DIR/controller.log for scale decisions."

cleanup() {
    echo "=== Killing any leftover port-forwards ==="
    pkill -f "kubectl port-forward" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_portforward() {
    local port=$1
    for i in $(seq 1 10); do
        if curl -s -o /dev/null "http://localhost:$port/v1/models"; then return 0; fi
        sleep 1
    done
    echo "WARNING: port-forward on $port never became reachable"
}

echo ""
echo "############################################"
echo "# Test 1 — TPOT SLO breach -> scale decode #"
echo "############################################"
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p '{"spec":{"decode":{"tpotSLOMs":20}}}'
kubectl port-forward -n pd-validation svc/decode 8001:8001 > /dev/null 2>&1 &
wait_for_portforward 8001
locust -f locustfile.py --host http://localhost:8001 \
    --headless --users 30 --spawn-rate 3 --run-time 90s DecodeLoad \
    --csv "$RESULTS_DIR/test1_decode_load" || true
sleep 15  # let a reconcile cycle catch up
echo "--- PDRatioPolicy status after Test 1 ---"
kubectl get pdratiopolicy validation -n pd-validation -o wide | tee "$RESULTS_DIR/test1_status.txt"
grep "scaled decode" "$RESULTS_DIR/controller.log" | tail -5 | tee "$RESULTS_DIR/test1_scale_events.txt" || echo "No 'scaled decode' events found yet"
pkill -f "port-forward.*decode" 2>/dev/null || true

# Reset for next test
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p '{"spec":{"decode":{"tpotSLOMs":80}}}'
sleep 65  # clear cooldown

echo ""
echo "###################################################"
echo "# Test 2 — Queue velocity spike -> scale prefill  #"
echo "###################################################"
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p '{"spec":{"prefill":{"queueVelocityTrigger":2}}}'
kubectl port-forward -n pd-validation svc/prefill 8000:8000 > /dev/null 2>&1 &
wait_for_portforward 8000
locust -f locustfile.py --host http://localhost:8000 \
    --headless --users 20 --spawn-rate 20 --run-time 20s PrefillBurst \
    --csv "$RESULTS_DIR/test2_prefill_burst" || true
sleep 15
echo "--- PDRatioPolicy status after Test 2 ---"
kubectl get pdratiopolicy validation -n pd-validation -o wide | tee "$RESULTS_DIR/test2_status.txt"
grep "scaled prefill" "$RESULTS_DIR/controller.log" | tail -5 | tee "$RESULTS_DIR/test2_scale_events.txt" || echo "No 'scaled prefill' events found yet"
pkill -f "port-forward.*prefill" 2>/dev/null || true
sleep 65

echo ""
echo "#####################################"
echo "# Test 3 — GPU budget enforcement   #"
echo "#####################################"
CURRENT_TOTAL=$(kubectl get pdratiopolicy validation -n pd-validation -o jsonpath='{.status.currentPrefillReplicas}{" "}{.status.currentDecodeReplicas}' | awk '{print $1+$2}')
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p "{\"spec\":{\"gpuBudget\":$CURRENT_TOTAL}}"
kubectl port-forward -n pd-validation svc/decode 8001:8001 > /dev/null 2>&1 &
kubectl port-forward -n pd-validation svc/prefill 8000:8000 > /dev/null 2>&1 &
wait_for_portforward 8001
wait_for_portforward 8000
locust -f locustfile.py --host http://localhost:8001 --headless --users 20 --spawn-rate 5 --run-time 60s DecodeLoad --csv "$RESULTS_DIR/test3_decode" &
locust -f locustfile.py --host http://localhost:8000 --headless --users 20 --spawn-rate 5 --run-time 60s PrefillBurst --csv "$RESULTS_DIR/test3_prefill" &
wait
sleep 15
echo "--- PDRatioPolicy status after Test 3 (prefill+decode must be <= budget=$CURRENT_TOTAL) ---"
kubectl get pdratiopolicy validation -n pd-validation -o wide | tee "$RESULTS_DIR/test3_status.txt"
pkill -f "kubectl port-forward" 2>/dev/null || true
sleep 65

echo ""
echo "#####################################"
echo "# Test 4 — Drain before scale-down  #"
echo "#####################################"
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p '{"spec":{"gpuBudget":4}}'
kubectl port-forward -n pd-validation svc/decode 8001:8001 > /dev/null 2>&1 &
wait_for_portforward 8001
locust -f locustfile.py --host http://localhost:8001 \
    --headless --users 5 --spawn-rate 1 --run-time 120s DecodeLoad \
    --csv "$RESULTS_DIR/test4_decode" &
LOCUST_PID=$!
sleep 30
kubectl patch pdratiopolicy validation -n pd-validation --type=merge \
    -p '{"spec":{"gpuBudget":2}}'
echo "--- Watching for draining label ---"
for i in $(seq 1 20); do
    kubectl get pods -n pd-validation --show-labels 2>/dev/null | grep -i draining && break
    sleep 3
done
wait $LOCUST_PID || true
echo "--- Test 4 Locust failure count (expect 0) ---"
tail -5 "$RESULTS_DIR/test4_decode_stats.csv" 2>/dev/null || echo "check test4_decode_stats.csv manually"
pkill -f "port-forward.*decode" 2>/dev/null || true

echo ""
echo "=== All 4 tests complete. Results in $RESULTS_DIR ==="
echo "=== Full controller log: $RESULTS_DIR/controller.log ==="
