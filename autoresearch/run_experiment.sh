#!/bin/bash
# Standard experiment runner: cleanup → launch → wait → benchmark → report
# Usage: run_experiment.sh <exp_number> <local_script> <bdea_script> <description>
set -euo pipefail

EXP="$1"
LOCAL_SCRIPT="$2"
BDEA_SCRIPT="${3:-$2}"
DESC="$4"

echo "============================================"
echo "EXPERIMENT $EXP: $DESC"
echo "============================================"

# --- SAFETY: Cleanup ---
echo "[safety] Cleaning up previous containers..."
docker rm -f ds4-rank0 2>/dev/null || true
ssh -o ConnectTimeout=5 edgexpert-bdea 'docker rm -f ds4-rank1 2>/dev/null' 2>/dev/null || true
sleep 3

# --- SAFETY: Kill stale GPU processes ---
echo "[safety] Checking for stale GPU procs..."
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c "[0-9]" || echo "0")
if [ "$PROCS" != "0" ]; then
  echo "[safety] Killing $PROCS stale GPU procs"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  sleep 2
fi

# --- SAFETY: Kill dangling NCCL workers ---
ssh -o ConnectTimeout=5 edgexpert-bdea 'pkill -f "nccl" 2>/dev/null || true; pkill -f "vllm" 2>/dev/null || true' 2>/dev/null || true
pkill -f "nccl" 2>/dev/null || true

echo "[safety] Cleanup complete"

# --- Copy script to bdea ---
scp "$BDEA_SCRIPT" edgexpert-bdea:/tmp/serve_exp.sh 2>/dev/null || true

# --- LAUNCH RANK 1 (bdea) ---
echo "[launch] Starting rank 1 (bdea)..."
ssh edgexpert-bdea "docker run -d --name ds4-rank1 --network host --gpus all --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/infiniband/ -v /dev/infiniband:/dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v /home/vikassridhar/.cache/huggingface/hub/models--drowzeys--keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32:/model:ro \
  -v /tmp/serve_exp.sh:/serve.sh:ro \
  -e NODE_RANK=1 -e HEADLESS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c /serve.sh" 2>/dev/null
echo "[launch] Rank 1 started"
sleep 5

# --- LAUNCH RANK 0 (9105) ---
echo "[launch] Starting rank 0 (9105)..."
docker run -d --name ds4-rank0 --network host --gpus all --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/infiniband/ -v /dev/infiniband:/dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v /mnt/ds4-ablit:/model:ro \
  -v "$LOCAL_SCRIPT:/serve.sh:ro" \
  -e NODE_RANK=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  vllm-dspark-runtime:dspark-nvfp4-stage-c /serve.sh
echo "[launch] Rank 0 started"

# --- WAIT FOR HEALTH ---
echo "[wait] Waiting for model to load..."
ELAPSED=0
MAX_WAIT=600
while [ $ELAPSED -lt $MAX_WAIT ]; do
  if curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "deepseek"; then
    echo "[wait] HEALTHY after ${ELAPSED}s"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "ds4-rank0"; then
    echo "[wait] RANK0 CRASHED at ${ELAPSED}s"
    docker logs ds4-rank0 2>&1 | tail -20
    echo "RESULT|CRASHED|0|0|0|0/3"
    exit 1
  fi
  sleep 15
  ELAPSED=$((ELAPSED + 15))
  if [ $((ELAPSED % 60)) = 0 ]; then
    LAST=$(docker logs ds4-rank0 2>&1 | tail -1 | head -c 100)
    echo "[wait] ${ELAPSED}s: $LAST"
  fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo "[wait] TIMEOUT after ${MAX_WAIT}s"
  echo "RESULT|TIMEOUT|0|0|0|0/3"
  exit 1
fi

# --- BENCHMARK ---
echo "[bench] Running benchmark..."
bash /home/vikassridhar/ds4-autoresearch/benchmark.sh "exp${EXP}"
