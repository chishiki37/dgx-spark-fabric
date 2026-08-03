#!/usr/bin/env bash
# launch_experiment.sh — Safe launch for DS4-Ablit experiments
# Usage: launch_experiment.sh [serve_script_path]
# Includes: pre-launch GPU check, rank-1-first ordering, health wait with timeout
set -euo pipefail

SERVE_SCRIPT="${1:-/home/vikassridhar/ds4_docker_serve_optimized.sh}"
IMAGE="vllm-dspark-runtime:dspark-nvfp4-stage-c"
MODEL_LOCAL_BDEA="/home/vikassridhar/.cache/huggingface/hub/models--drowzeys--keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32"
MODEL_NFS_9105="/mnt/ds4-ablit"
HEALTH_TIMEOUT=600  # 10 min max for startup
HEALTH_INTERVAL=15

echo "[launch] Serve script: $SERVE_SCRIPT"

# Step 1: Verify clean GPU state
echo "[launch] Checking GPU is clean..."
PROC=$(nvidia-smi --query-compute-apps=pid 2>/dev/null | grep -c "[0-9]" || echo "0")
if [ "$PROC" != "0" ]; then
  echo "[launch] ⚠️ GPU has $PROC compute procs — run safe_cleanup.sh first!"
  exit 1
fi
echo "[launch] GPU clean ✅"

# Step 2: Verify NFS mount
if ! ls ${MODEL_NFS_9105}/config.json >/dev/null 2>&1; then
  echo "[launch] ⚠️ NFS mount stale — remounting..."
  mount -t nfs -o ro,vers=3,proto=rdma,port=20049,mountproto=tcp,rsize=1048576,wsize=1048576 \
    10.10.10.2:${MODEL_LOCAL_BDEA} ${MODEL_NFS_9105} 2>/dev/null || {
    echo "[launch] ❌ NFS remount failed — cannot proceed"
    exit 1
  }
fi
echo "[launch] NFS mount OK ✅"

# Step 3: Verify serve script exists on both nodes
if [ ! -f "$SERVE_SCRIPT" ]; then
  echo "[launch] ❌ Serve script not found: $SERVE_SCRIPT"
  exit 1
fi
ssh -o ConnectTimeout=5 edgexpert-bdea "test -f $SERVE_SCRIPT" 2>/dev/null || {
  echo "[launch] Copying serve script to bdea..."
  scp "$SERVE_SCRIPT" edgexpert-bdea:"$SERVE_SCRIPT" 2>/dev/null
}

# Step 4: Launch rank 1 (bdea) FIRST
echo "[launch] Starting rank 1 (bdea, headless worker)..."
ssh edgexpert-bdea "docker run -d --name ds4-rank1 --network host --gpus all --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/infiniband/ -v /dev/infiniband:/dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v ${MODEL_LOCAL_BDEA}:/model:ro \
  -v ${SERVE_SCRIPT}:/serve.sh:ro \
  -e NODE_RANK=1 -e HEADLESS=1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  ${IMAGE} /serve.sh" 2>/dev/null
echo "[launch] Rank 1 started ✅"

sleep 5

# Step 5: Launch rank 0 (9105, API server)
echo "[launch] Starting rank 0 (9105, API server)..."
docker run -d --name ds4-rank0 --network host --gpus all --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/infiniband/ -v /dev/infiniband:/dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v ${MODEL_NFS_9105}:/model:ro \
  -v ${SERVE_SCRIPT}:/serve.sh:ro \
  -e NODE_RANK=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  ${IMAGE} /serve.sh
echo "[launch] Rank 0 started ✅"

# Step 6: Wait for health with hard timeout
echo "[launch] Waiting for API health (max ${HEALTH_TIMEOUT}s)..."
ELAPSED=0
while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
  if curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "deepseek-v4-ablit"; then
    echo "[launch] API healthy after ${ELAPSED}s ✅"
    echo "METRIC: startup_time_s=${ELAPSED}"
    exit 0
  fi

  # Check for crash
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "ds4-rank0"; then
    echo "[launch] ❌ rank0 container died after ${ELAPSED}s"
    docker logs ds4-rank0 2>&1 | tail -20
    exit 1
  fi
  if ! ssh -o ConnectTimeout=5 edgexpert-bdea 'docker ps --format "{{.Names}}"' 2>/dev/null | grep -q "ds4-rank1"; then
    echo "[launch] ❌ rank1 container died after ${ELAPSED}s"
    ssh edgexpert-bdea 'docker logs ds4-rank1 2>&1 | tail -20' 2>&1
    exit 1
  fi

  sleep $HEALTH_INTERVAL
  ELAPSED=$((ELAPSED + HEALTH_INTERVAL))
  echo "[launch] Still waiting... ${ELAPSED}s"
done

echo "[launch] ❌ Health check timed out after ${HEALTH_TIMEOUT}s"
exit 1
