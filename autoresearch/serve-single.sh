#!/bin/bash
# DS4 vLLM serve - SINGLE NODE (9105), TP=1
# GB10 has 128GB unified memory - model weights are 74GB, leaving ~35GB for KV cache
# This avoids the vLLV V1 multi-node collective_rpc bug

docker run -d --name ds4-single \
  --privileged --gpus all --network host \
  --ipc=host --shm-size=2g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -e NCCL_DEBUG=WARN \
  -e GLOO_SOCKET_IFNAME=wlP9s9 \
  -v /mnt/deepseek-nfs:/model:ro \
  -v vllm-cache:/cache/huggingface/vllm-cache \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready \
  /opt/env/bin/vllm serve /model \
    --served-model-name deepseek-v4-ablit \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --enable-flashinfer-autotune
