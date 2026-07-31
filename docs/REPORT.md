# Fixing the Hidden 25% Tax on My 2-Node DGX Spark Cluster

**TL;DR — My two NVIDIA DGX Sparks were leaving ~25-33% of their decode speed on the table. The culprit wasn't the models, the quant, or vLLM — it was the direct 200G DAC cable between them. Adding a MikroTik CRS812 switch to the fabric fixed RDMA, tripled NCCL throughput, and made every model 26-33% faster overnight.**

---

## The Setup

- 2× NVIDIA DGX Spark (GB10, Grace-Blackwell, 121 GB unified memory each)
- Serving 150-300B-class MoE models with vLLM, tensor-parallel across both nodes
- Interconnect: ConnectX-7 @ 200G — previously a single direct QSFP56 DAC
- New: MikroTik CRS812 DDQ (400G-class) + NADDOD 400G→2×200G QSFP-DD breakout

## What Was Wrong (four stacked problems)

### 1. RDMA was silently broken on the direct link
RoCE between the two Sparks failed at queue-pair setup (`ibv_modify_qp`, error 61). Every NCCL collective fell back to TCP sockets: **~12-15 GB/s instead of the ~50 GB/s the hardware can do**. For TP=2 inference, every single decode step pays for an all-reduce across the link — so the transport tax hits every token.

### 2. Even "working" RoCE was degraded
One model (MiniMax) did run RoCE on the direct link — at 32.7 tok/s. Through the switch, same model, same flags: **40.2 tok/s**. The direct link has no packet buffering: NCCL's all-reduce bursts overrun the ARM CPUs' ability to drain them, and TCP collapses into retransmit storms. Proof point: iperf3 TCP throughput went from **15 Gb/s (direct) to 107 Gb/s (through the switch)** on identical hardware.

### 3. The interface lottery
The two NICs' working ports didn't even have the same names (`enp1s0f1np1` on one node, `enp1s0f0np0` on the other), and the second CX7 card has a capital-P device name that crashes NCCL's HCA filter. Recipes kept silently breaking.

### 4. Brittle bring-up
A PCIe hotplug driver parks the NICs at every boot; fabric IPs don't survive reboots. Every session started with manual NIC resurrection rituals.

## The Fix

Moved both Sparks onto a MikroTik CRS812 with a 400G→2×200G breakout DAC. Key discoveries:

- **RoCE works through the switch.** Same NICs, same cable type — `ib_write_bw` clean at 13.3 GB/s, zero error 61. The switch normalizes whatever was breaking QP setup point-to-point.
- **NCCL + GPUDirect RDMA: 40-51 GB/s** all-reduce BusBW (vs 12-15 GB/s on Socket).
- The CRS812's lane-grouping config is non-obvious (you must enable all 8 lane-interfaces of a QSFP-DD cage for 2×200G — enable just the two "masters" and you get 50G per leg).
- Containers need `--privileged` for `/dev/infiniband`, and NCCL 2.30 makes `NCCL_NET=IB` fatal-if-unavailable (no more silent Socket fallback).

## Results — every model got faster by ~a third

| Model | Direct link | Via CRS812 | Delta |
|---|---|---|---|
| MiMo V2.5 NVFP4 (171 GB, 2-node TP) | ~19 tok/s | **25.4 tok/s** | **+33%** |
| DeepSeek V4 Flash FP8 (162.7B MoE) | ~30 tok/s | **39.2 tok/s** | **+31%** |
| MiniMax M2.7 AWQ (122 GB) | ~32 tok/s | **40.2 tok/s** | **+26%** |

MiniMax at 40.2 tok/s now **matches the Spark Arena reference number (~40 tok/s)** for this hardware — first time this cluster has hit the published figure. That's the tell: the published benchmarks assume a working fabric. Mine never had one, until now.

## Lessons

1. **Benchmark the transport before the model.** Months of "this model is slow" were actually "this fabric is broken." `ib_write_bw` + an NCCL all-reduce bench should be step zero of any multi-node build.
2. **A switch is not overhead — it's missing infrastructure.** Buffering, normalized RDMA, and a path that scales to 4-8 nodes.
3. **Uniform gains across architectures** (dense-ish AWQ, MoE FP8, MoE NVFP4) mean the tax was purely systemic — nothing model-specific.
4. If your numbers are below the leaderboard reference and you can't explain why, suspect the boring layers: cables, link training, NCCL device selection.

## What's next

The CRS812 has 32 ports of headroom. Plan: 4→8 Spark quant lab (NVFP4/INT4/AWQ/GGUF comparisons, pruning, spec-decode), PFC/ECN tuning to see if RoCE goes further, and a bonded 2×200G dual-rail experiment.

*Full configs, RouterOS commands, recipes, and benchmark scripts: [github.com/chishiki37 — link in the GitHub post]*
