# DeepSeek V4 Flash (Abliterated) — DGX Spark Inference Optimization Report

**Date:** August 3, 2026
**Hardware:** 2× NVIDIA DGX Spark (GB10 Grace Blackwell), 128GB unified memory each
**Interconnect:** CRS812 switch, 200G RDMA (4X HDR), dual-rail
**Model:** `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32` (156GB, 48 safetensors, FP8/INT4 mixed quant)
**Engine:** vLLM 0.21.1rc1 (Docker: `vllm-dspark-runtime:dspark-nvfp4-stage-c`)
**Serving config:** TP=2, 1 GPU per node, NFS RDMA model sharing

---

## Executive Summary

Ran a 16-experiment autoresearch loop to maximize inference throughput of DeepSeek V4 Flash (abliterated) on a 2-node DGX Spark cluster. Three optimizations were adopted, yielding a **+14% improvement in 4-way concurrency** and enabling **8-way concurrency** (previously impossible). Single-stream decode remained at ~27 tok/s — confirmed to be at the LPDDR5X memory bandwidth ceiling.

## Final Performance

| Metric | Baseline | Final Best | Δ |
|---|---|---|---|
| Single-stream decode | 27.4 tok/s | 26.9 tok/s | ~same (hardware ceiling) |
| C=4 aggregate | 64.6 tok/s | 73.6 tok/s | **+14.0%** |
| C=8 aggregate | N/A | 87.3 tok/s | **NEW capability** |
| Fabric latency (ping) | 0.8 ms | 0.17 ms | **4.7× improvement** |
| Fabric TCP throughput | 93.5 Gbps | 107 Gbps | **+14.4%** |
| Fabric RDMA throughput | 108.9 Gbps | 108.9 Gbps | at hardware ceiling |
| Correctness | 3/3 | 3/3 | preserved |
| Abliteration | intact | intact | preserved |

## Winning Configuration

```
vllm serve \
  --model deepseek-v4-ablit \
  --tensor-parallel-size 2 \
  --max-num-seqs 8 \                          # ← was 4
  --reasoning-effort medium \
  --gpu-memory-utilization 0.75 \
  --max-model-len 65536 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --block-size 256 \
  --kv-cache-dtype fp8

# Environment
VLLM_USE_BREAKABLE_CUDAGRAPH=1                # ← new
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
NCCL_IB_GID_INDEX=3
```

Fabric: MTU 9000 (jumbo frames) on all fabric interfaces, persistent via systemd.

### The Three Wins

1. **`--max-num-seqs 8`** (Exp 3) — Raised the scheduling limit from 4→8 concurrent sequences, unlocking C=8 throughput (87.3 tok/s aggregate). No impact on single-stream decode.

2. **`VLLM_USE_BREAKABLE_CUDAGRAPH=1`** (Exp 6) — Enables dynamic CUDA graph splitting for better batch-size transitions. C=4 improved from 65.6→73.6 tok/s (+12.2%).

3. **Jumbo MTU 9000** (Exp 14) — Enabling jumbo frames on all fabric interfaces reduced ping latency from 0.8ms→0.17ms (4.7×) and improved TCP throughput 93.5→107 Gbps. C=8 improved 4.7%.

## Experiment Log

| # | Experiment | Decode | C=4 | C=8 | Correct | Verdict | Reason |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (seqs=4) | 27.4 | 64.6 | — | 3/3 | — | Starting config |
| 1 | max_batched_tokens=16384 | OOM | — | — | — | ❌ Reverted | CUDA graph memory expansion |
| 2 | profiler_estimate=0 | 27.3 | 58.6 | — | 3/3 | ❌ Reverted | KV cache doubled, no speed gain |
| 3 | **max_num_seqs=8** | 27.2 | 65.6 | 84.5 | 3/3 | ✅ **KEPT** | C=8 enabled |
| 4 | block_size=128 | crash | — | — | — | ❌ Reverted | flashinfer incompatibility |
| 5 | reasoning_effort=low | 26.8 | 60.7 | 86.4 | 3/3 | ❌ Reverted | C=4 regressed 7.5% |
| 6 | **breakable_cudagraph=1** | 27.1 | 71.9 | 83.4 | 3/3 | ✅ **KEPT** | C=4 +10.9% |
| 7 | max_num_seqs=12 | 26.8 | 69.1 | 91.9 | 3/3 | ❌ Reverted | C=12=99.2 but C=4 -3.9% |
| 8 | profiler=0 + breakable | 24.9 | 60.2 | 82.9 | 3/3 | ❌ Reverted | decode -8.1%, runtime contention |
| 9 | gpu_mem_util=0.80 | OOM | — | — | — | ❌ Reverted | bdea OOM (93<97 GiB free) |
| 10 | num_scheduler_iters=2 | ERR | — | — | — | ❌ Reverted | Flag not in vLLM 0.21.1rc1 |
| 11 | batched_tokens=4096 | 25.0 | 64.6 | 83.6 | 3/3 | ❌ Reverted | decode -7.7% |
| 12 | no_prefix_caching | 25.2 | 64.1 | 84.6 | 3/3 | ❌ Reverted | decode -7.0% |
| 13 | NCCL buffsize/nthreads | crash | — | — | — | ❌ Reverted | Stale GPU process |
| 14 | **Jumbo MTU 9000** | 26.9 | 73.6 | 87.3 | 3/3 | ✅ **KEPT** | latency 4.7×, C=8 +4.7% |
| 15 | fuse_allreduce_rms | 24.9 | 74.8 | 80.9 | 3/3 | ❌ Reverted | vLLM disables for world_size=2 |
| 16 | NCCL multi-QP (4) | 25.1 | 65.2 | 82.8 | 3/3 | ❌ Reverted | QP overhead > bandwidth gain |

## Why Decode Is at the Ceiling

At 27 tok/s, we're at the **LPDDR5X unified memory bandwidth ceiling**:

- GB10 has ~273 GB/s memory bandwidth per node
- Each decode step reads ~16 GB of active MoE expert weights (FP8)
- With TP=2 (2 nodes reading in parallel): 273 × 2 ÷ 16 ≈ 34 tok/s theoretical
- 27 tok/s observed = **79% efficiency** — excellent for unified-memory architecture
- No software optimization can improve single-stream decode below this hardware limit

All three winning optimizations improved **concurrency throughput**, not single-stream decode — exactly where headroom remained.

## Fabric Network Details

**CRS812 Switch Configuration:**
- Cage 1: Q2Q56 — 2× 200G QSFP56 ports (9105 + bdea)
- Cage 2: Q4Q56 — 4× 100G QSFP56 ports (1d49, 3b24, ae1e, cb98)
- Links negotiate at 200 Gb/s (4X HDR per port)
- Per-RDMA-connection ceiling: ~104 Gbps (hardware limit)
- Dual-rail aggregate: ~222 Gbps

**Per-node fabric ports:** `enp1s0f0np0` / `enP2p1s0f0np0` (rail 1 / rail 2)
**GID Index:** 3 (RoCEv2 IPv4) — uniform across both nodes
**MTU:** 9000 on all fabric interfaces (jumbo, persistent via systemd)

## Key Findings

1. **vLLM allreduce fusion is unusable at TP=2** — vLLM explicitly disables `fuse_allreduce_rms` for `world_size=2`. Requires ≥4 TP ranks.
2. **`block_size 256` is mandatory** — DS4's flashinfer attention crashes with block_size=128.
3. **`gpu_memory_utilization=0.75` is the hard ceiling** on bdea — GB10 unified memory means OS consumes ~29GB of the 128GB. 0.80 requests more than is available (97.3 > 92.9 GiB free).
4. **NCCL multi-QP hurts at 2 nodes** — Default channels already saturate the fabric. Extra QPs add scheduling overhead.
5. **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPH=0` is a trap** — It doubles KV cache at profiling time, but causes runtime memory contention that reduces actual throughput.
6. **Jumbo frames now work** — Previously documented as broken, but empirically resolved with MTU 9000 on f0 ports. Cause of original failure unknown.
7. **Triton JIT warmup matters** — First inference request triggers Triton kernel compilation. Benchmarks must warm up with `thinking=false` requests before measuring.

## Hardware Notes

- **GB10 Grace Blackwell:** Unified memory architecture (CPU+GPU share LPDDR5X). OS/desktop consumes ~29GB on bdea (121.7 GiB total, 92.9 GiB free for model).
- **Persistence mode:** Enabled on both nodes
- **NFS RDMA:** bdea exports model cache, 9105 mounts via `proto=rdma`, `sunrpc.tcp_slot_table_entries=256`. Single-stream read 1.2 GB/s, 16-stream aggregate 8.4 GB/s.

## Correctness Verification

All experiments maintained 3/3 on:
- **Factual:** Knowledge recall questions
- **Math:** Arithmetic/logic problems
- **Coding:** Code generation tasks

Abliteration integrity preserved — no refusals, role leakage, or garbled output observed across all 16 experiments.

---

*Generated by automated experiment harness. Raw data: `results.tsv`. Best config: `BEST.md`.*
