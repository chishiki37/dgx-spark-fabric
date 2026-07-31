#!/bin/bash
# manual-2node-vllm-launch.sh — Manual 2-node vLLM launch for DGX Spark cluster
# when sparkrun's rsync model distribution fails (root-owned .no_exist stubs).
#
# Usage: edit MODEL_ID, SERVED_NAME, and IMAGE below, then run on spark-head.
# The model MUST already be present in ~/.cache/huggingface on BOTH nodes.
#
# Verified 2026-07-23 with MiniMax-M2.7-AWQ-4bit (eugr-nightly, ~4 min to serving).
#
# TOPOLOGY SWITCH (Jul 31, 2026): set TOPOLOGY=crs812 when running through the
# MikroTik switch fabric (RoCE works, needs --privileged + f0 ports), or
# TOPOLOGY=direct for the legacy direct DAC (Socket NCCL, f1 ports on 9105).

set -euo pipefail

# ---- EDIT THESE ----
MODEL_ID="cyankiwi/MiniMax-M2.7-AWQ-4bit"
SERVED_NAME="minimax-m27"
IMAGE="ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
PORT=8600
GPU_MEM=0.85
MAX_LEN=32768
EXTRA_FLAGS="--load-format fastsafetensors --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2"
CONTAINER_NAME="vllm-2node"
WORKER_SSH="spark-worker"
TOPOLOGY="${TOPOLOGY:-crs812}"   # crs812 | direct

# ---- Cluster topology ----
HEAD_CX7_IP="10.10.10.1"
WORKER_CX7_IP="10.10.10.2"
MASTER_PORT=29500

if [ "$TOPOLOGY" = "crs812" ]; then
  HEAD_IFACE="enp1s0f0np0"
  WORKER_IFACE="enp1s0f0np0"
  PRIV="--privileged"                 # required: exposes /dev/infiniband for RDMA
  NCCL_HCA="rocep1s0f0"
  NCCL_EXTRA="-e NCCL_IB_GID_INDEX=3" # RoCE+GDR through switch: 40-51 GB/s BusBW
else
  HEAD_IFACE="enp1s0f1np1"     # 9105 direct-DAC port
  WORKER_IFACE="enp1s0f0np0"   # bdea direct-DAC port (asymmetric!)
  PRIV=""
  NCCL_HCA="mlx5_0"
  NCCL_EXTRA=""
fi

# ---- Cleanup any previous run ----
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true
ssh "$WORKER_SSH" "docker stop $CONTAINER_NAME 2>/dev/null || true; docker rm $CONTAINER_NAME 2>/dev/null || true" || true
sleep 2

# ---- HEAD node (rank 0, serves the API) ----
echo "=== Launching head node ==="
docker run -d --name "$CONTAINER_NAME" --gpus all --network host --ipc host $PRIV \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e NCCL_IB_HCA="$NCCL_HCA" \
  $NCCL_EXTRA \
  -e NCCL_SOCKET_IFNAME="$HEAD_IFACE" \
  -e GLOO_SOCKET_IFNAME="$HEAD_IFACE" \
  -e TP_SOCKET_IFNAME="$HEAD_IFACE" \
  -e NCCL_DEBUG=WARN \
  "$IMAGE" \
  vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  $EXTRA_FLAGS \
  --nnodes 2 --node-rank 0 \
  --master-addr "$HEAD_CX7_IP" --master-port "$MASTER_PORT"

# ---- WORKER node (rank 1, headless — no API server) ----
echo "=== Launching worker node ==="
ssh "$WORKER_SSH" "docker run -d --name $CONTAINER_NAME --gpus all --network host --ipc host $PRIV \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_IB_HCA=$NCCL_HCA \
  $NCCL_EXTRA \
  -e NCCL_SOCKET_IFNAME=$WORKER_IFACE \
  -e GLOO_SOCKET_IFNAME=$WORKER_IFACE \
  -e TP_SOCKET_IFNAME=$WORKER_IFACE \
  -e NCCL_DEBUG=WARN \
  $IMAGE \
  vllm serve $MODEL_ID \
  --served-model-name $SERVED_NAME \
  --trust-remote-code \
  --host 0.0.0.0 --port $PORT \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization $GPU_MEM \
  --max-model-len $MAX_LEN \
  $EXTRA_FLAGS \
  --nnodes 2 --node-rank 1 \
  --master-addr $HEAD_CX7_IP --master-port $MASTER_PORT \
  --headless"

# ---- Wait for readiness (~4-5 min for 122 GB AWQ) ----
echo "=== Waiting for $SERVED_NAME on port $PORT ==="
for i in $(seq 1 60); do
  if curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "READY after ~$((i * 10))s"
    curl -s "http://localhost:$PORT/v1/models"
    exit 0
  fi
  # Fail fast if head container died
  if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    echo "ERROR: head container exited. Logs:"
    docker logs "$CONTAINER_NAME" --tail 40 2>&1 || true
    exit 1
  fi
  sleep 10
done
echo "TIMEOUT after 10 min — check: docker logs $CONTAINER_NAME --tail 50"
exit 1
