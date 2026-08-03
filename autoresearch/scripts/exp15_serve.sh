#!/usr/bin/env bash
# DS4 Abliterated serve — OPTIMIZED for speed (Aug 3, 2026)
# Changes from original: reasoning_effort=medium, dual-rail NCCL, proper 10.10.10.x fabric
set -euo pipefail

PY=/opt/env/bin/python

# Install cutlass-dsl if missing
if [ "$($PY -c 'import cutlass;print(cutlass.__version__)' 2>/dev/null)" != "4.5.1" ]; then
  echo "[serve] installing cutlass-dsl 4.5.1 ..."
  $PY -m pip install --no-cache-dir "nvidia-cutlass-dsl==4.5.1" "nvidia-cutlass-dsl-libs-base==4.5.1" "nvidia-cutlass-dsl-libs-cu13==4.5.1" >/tmp/cutlass_install.log 2>&1
fi
echo "[serve] cutlass=$($PY -c 'import cutlass;print(cutlass.__version__)' 2>/dev/null)"

# Env vars for B12X MoE and caching
export VLLM_USE_B12X_MOE=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export TRITON_CACHE_DIR=/tmp/triton-cache-rank${NODE_RANK:-0}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache-rank${NODE_RANK:-0}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# RoCE GID index detection (find IPv4 RoCE v2 GID)
for HCA in rocep1s0f0 roceP2p1s0f0; do
  for i in $(seq 0 7); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null)
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null)
    case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
  done
done
echo "[serve] NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset}"

exec /usr/local/bin/dsv4-vllm-entrypoint serve /model \
  --served-model-name deepseek-v4-ablit \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 65536 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.75 \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=medium \
  --enable-flashinfer-autotune \
  -cc '{"pass_config": {"fuse_allreduce_rms": true, "fuse_gemm_comms": true}}' \
  --nnodes 2 \
  --node-rank "${NODE_RANK}" \
  --master-addr 10.10.10.1 \
  --master-port 25000 \
  ${HEADLESS:+--headless}
