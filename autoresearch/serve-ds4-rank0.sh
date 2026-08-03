#!/bin/bash
# DS4 vLLM serve - Rank 0 (9105/leader)
# Using unholy-fusion image (no B12X baked in, cleaner multi-node support)

docker run -d --name ds4-rank0 \
  --privileged --gpus all --network host \
  --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device=/dev/infiniband \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NODE_IP=10.10.30.1 \
  -v /mnt/deepseek-nfs:/model:ro \
  -v vllm-cache:/cache/huggingface/vllm-cache \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready \
  /opt/env/bin/vllm serve /model \
    --served-model-name deepseek-v4-ablit \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 65536 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --distributed-executor-backend mp \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --enable-flashinfer-autotune \
    --nnodes 2 --node-rank 0 \
    --master-addr 10.10.30.1 --master-port 25000
