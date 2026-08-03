# DS4-Ablit Autoresearch — Final Results

## Best Configuration

| Setting | Value |
|---|---|
| Model | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` |
| Nodes | 9105 (rank 0) + bdea (rank 1), 2× DGX Spark |
| Docker image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` |
| TP | 2 (1 GPU per node) |
| max_num_seqs | **8** (up from 4) |
| VLLM_USE_BREAKABLE_CUDAGRAPH | **1** |
| reasoning_effort | medium |
| gpu_memory_utilization | 0.75 |
| max_model_len | 65536 |
| kv_cache_dtype | fp8 |
| block_size | 256 |
| Fabric MTU | **9000** (jumbo) |
| NCCL_IB_HCA | rocep1s0f0,roceP2p1s0f0 |

## Performance Metrics

| Metric | Baseline | Final Best | Improvement |
|---|---|---|---|
| Single-stream decode | 27.4 tok/s | 27.1 tok/s | ~same |
| C=4 aggregate | 64.6 tok/s | 73.6 tok/s | **+14.0%** |
| C=8 aggregate | N/A | 87.3 tok/s | **NEW** |
| Fabric latency | 0.8ms | 0.17ms | **4.7x** |
| Correctness | 3/3 | 3/3 | — |

## What Worked
1. **`max_num_seqs 8`** — enabled C=8 concurrency (was limited to 4)
2. **`VLLM_USE_BREAKABLE_CUDAGRAPH=1`** — better batch-size transitions at high concurrency
3. **Jumbo MTU 9000** — 4.7x latency reduction on fabric, +4.7% C=8 throughput

## What Didn't Work (and why)
- **Larger batched tokens (16K)**: OOM from CUDA graph memory expansion
- **Profiler disabled**: Doubled KV cache but caused runtime memory contention
- **block_size 128**: flashinfer attention incompatibility (crash)
- **reasoning_effort=low**: More decode pressure from non-thinking responses
- **max_num_seqs 12**: C=12 reached 99 tok/s but C=4 regressed 3.9%
- **gpu_memory_utilization 0.80**: bdea OOM (unified memory: 93 GiB free < 97 GiB requested)
- **Smaller batched tokens (4K)**: Hurt prefill pipeline
- **No prefix caching**: Hashing overhead is minimal, caching helps
- **fuse_allreduce_rms**: Disabled by vLLM for world_size=2 ("not supported")
- **NCCL multi-QP (4 QPs)**: QP scheduling overhead exceeded bandwidth gain
- **Inductor compilation mode**: Worse for concurrency than breakable graphs

## Why We Hit the Ceiling
At 27 tok/s decode, we're at the **LPDDR5X memory bandwidth ceiling** (~273 GB/s). Each decode step reads ~16 GB of active MoE expert weights. With TP=2 parallel decode:
- 273 GB/s × 2 GPUs ÷ 16 GB ≈ 34 tok/s theoretical
- 27 tok/s observed = ~79% efficiency (excellent for this hardware)
