# 4×100G vs 2×200G Breakout: Does Cable Geometry Matter for DGX Spark Clusters?

**TL;DR — We swapped the NADDOD 400G→2×200G breakout DAC for a 400G→4×100G breakout on the same MikroTik CRS812 fabric. RoCE worked at 100G. Decode speed on all three models was identical (within ±1%). The DGX Spark interconnect at TP=2 is compute-bound, not network-bound — 100G is more than enough for every model we tested.**

---

## The Question

Our CRS812 fabric was built with a NADDOD `Q2Q56-400G-CU1` — a 400G→2×200G QSFP-DD breakout DAC. That gave us 200G-CR4 per leg, 40-51 GB/s NCCL BusBW, and +26-33% decode improvements over the direct-link baseline (see [REPORT.md](REPORT.md)).

The natural follow-up: **does halving the per-leg speed to 100G change inference performance?** At TP=2, every decode step does an all-reduce across the link. If the collective is network-bound, 100G should cut throughput roughly in half. If it's compute-bound, we shouldn't see any difference.

## Hardware

- **Switch:** MikroTik CRS812-8DS-2DQ-2DDQ-RM (RouterOS v7, Marvell switch chip)
- **Nodes:** 2× NVIDIA DGX Spark (GB10 Grace-Blackwell, 121 GB unified memory each)
- **NICs:** ConnectX-7 (2 cards per node, 2 ports per card)
- **Original cable:** NADDOD `Q2Q56-400G-CU1` (QSFP-DD 400G → 2× QSFP56 200G, 1m, CMIS 5.0)
- **New cable:** 400G → 4×100G QSFP-DD breakout DAC (QSFP56 100G per leg)
- **Software:** vLLM v0.23.1rc1 (eugr-nightly), NCCL 2.30.7, CUDA 13.0, Ray 2.x

## RouterOS Lane Grouping for 4×100G

This was the first non-obvious discovery. The CRS812 exposes each QSFP-DD cage as 8 lane interfaces (`qsfp56-dd-<cage>-1` through `-8`). An enabled interface claims all contiguous serdes lanes below the next enabled interface (see [crs812-setup.md](crs812-setup.md)).

For a 4×100G breakout, each leg maps to a 2-lane pair:

| Leg | Lanes | 100G-CR2 Group Master |
|-----|-------|-----------------------|
| 1   | 1-2   | `dd-X-1`              |
| 2   | 3-4   | `dd-X-3`              |
| 3   | 5-6   | `dd-X-5`              |
| 4   | 7-8   | `dd-X-7`              |

**To light up legs 1 and 3** (the two we need — one per Spark):

```
/interface ethernet disable [find name~"qsfp56-dd-1"]
/interface ethernet enable qsfp56-dd-1-1,qsfp56-dd-1-2,qsfp56-dd-1-5,qsfp56-dd-1-6
/interface ethernet disable [find name~"qsfp56-dd-2"]
/interface ethernet enable qsfp56-dd-2-1,qsfp56-dd-2-2,qsfp56-dd-2-5,qsfp56-dd-2-6
```

This creates two 100G-CR2 groups per cage: `dd-X-1` (lanes 1-2, leg 1) and `dd-X-5` (lanes 5-6, leg 3). **Legs 2 and 4 stay dark** — no lane group covers them.

**Critical:** Do NOT enable `dd-X-1` through `dd-X-4` contiguously. That collapses into one 4-lane run (200G-CR4) and no 100G leg will link.

### Bridge ports

The lane masters must be added to the bridge, or ARP fails silently:

```
/interface bridge port add bridge=bridge interface=qsfp56-dd-1-1
/interface bridge port add bridge=bridge interface=qsfp56-dd-1-5
/interface bridge port add bridge=bridge interface=qsfp56-dd-2-1
/interface bridge port add bridge=bridge interface=qsfp56-dd-2-5
```

Verify: `rate=100Gbps`, `fec=fec91`, `status=link-ok`.

## The GID Index Trap

This was the hardest bug of the session. When we switched from the first ports (`enp1s0f0np0` / `rocep1s0f0`) to the second ports (`enp1s0f1np1` / `rocep1s0f1`), NCCL started failing with `ibv_modify_qp` errors — the same "error 61" class of failure we saw on the direct link.

**Root cause:** The two DGX Sparks have **different GID index layouts** for their second ports. The IPv4-mapped RoCE GID (the one that resolves to a routable address) lives at a different index on each machine:

**9105 (`rocep1s0f1`):**
```
GID 0: fe80::... (link-local v1)
GID 1: fe80::... (link-local v2)
GID 2: fe80::... (random, not IPv4)
GID 3: fe80::... (random, not IPv4)  ← was pinned here, BROKEN
GID 4: ::ffff:10.10.10.1 (IPv4 v1)
GID 5: ::ffff:10.10.10.1 (IPv4 v2)  ← correct
```

**bdea (`rocep1s0f1`):**
```
GID 0: fe80::... (link-local v1)
GID 1: fe80::... (link-local v2)
GID 2: ::ffff:10.10.10.2 (IPv4 v1)  ← correct
GID 3: ::ffff:10.10.10.2 (IPv4 v2)
```

The 200G config used `NCCL_IB_GID_INDEX=3` because both nodes had their IPv4-mapped GID at index 3 on the first ports. On the second ports, index 3 points to a non-routable link-local on 9105 but a valid IPv4 GID on bdea — so `ibv_modify_qp` fails on the 9105 side.

**Fix: Remove `NCCL_IB_GID_INDEX` entirely and let NCCL auto-select.** NCCL 2.30's auto-selection correctly picks the IPv4-mapped RoCEv2 GID on both nodes regardless of index position. Verified: `ncclCommInitRank` Init COMPLETE on both ranks with zero errors.

## The Ray OOM Problem

Three consecutive OOM crashes on bdea (the worker node) during model loading nearly killed this experiment. The symptom was always the same: bdea's memory climbed from ~5 GB to ~119 GB during checkpoint shard loading, then the raylet died and the node went dark.

**Root cause:** Ray's distributed executor creates an object store and worker processes that consume significant unified memory overhead (the Spark's GPU memory = host memory on Grace-Blackwell). On a 121 GB node loading a ~61 GB model shard, Ray's overhead pushed total memory past the limit.

**The original 200G MiniMax benchmark (40.2 tok/s) was NOT run via Ray** — it used `manual-2node-vllm-launch.sh`, which launches vLLM's native multi-node backend (`--nnodes 2 --node-rank 0/1`) with a single container per node. No Ray object store, no Ray worker processes.

**Fix:** Use the same native multi-node launcher for the 100G comparison. Added a `TOPOLOGY=crs812-100g` branch to `manual-2node-vllm-launch.sh`:

```bash
elif [ "$TOPOLOGY" = "crs812-100g" ]; then
  HEAD_IFACE="enp1s0f1np1"       # 4x100G breakout, second ports
  WORKER_IFACE="enp1s0f1np1"
  PRIV="--privileged"
  NCCL_HCA="rocep1s0f1"
  NCCL_EXTRA=""                   # GID auto-select (f1 GID layout differs per node)
```

DeepSeek V4 and MiMo were smaller and survived Ray — but for consistency, we validated that their launch scripts also had `NCCL_IB_GID_INDEX=3` removed.

## Methodology

### Transport benchmarks
- `ib_write_bw -d rocep1s0f1 -s 1048576 -q 4 --report_gbits` (both rails)
- Pings on both subnets (10.10.10.x rail 1, 10.10.20.x rail 2)

### Model benchmarks
- **MiniMax M2.7 AWQ** (122 GB, dense-ish MoE): native `--nnodes 2` launch, `gpu-memory-utilization=0.85`, `max-model-len=32768`, port 8600
- **DeepSeek V4 Flash FP8** (162.7B MoE): Ray executor, `gpu-memory-utilization=0.80`, `max-model-len=200000`, MTP spec-decode (2 speculative tokens), port 8000
- **MiMo V2.5 NVFP4** (171 GB, MoE): Ray executor, NVFP4 quant, `gpu-memory-utilization=0.82`, `max-model-len=32768`, port 8000

### Decode test
4 runs per model, 400 tokens each, temperature=0. First run discarded as warmup (spec-decode cold start for DSV4). Results are the average of runs 2-4. Prompts were identical across all tests — a detailed technical explanation prompt (~30 tokens).

## Results

### Transport Layer

| Metric | 2×200G (Jul 31) | 4×100G (Aug 1) | Delta |
|--------|-----------------|----------------|-------|
| `ib_write_bw` per rail | 13.3 GB/s | 12.7 GB/s | −4.5% |
| Link rate | 200G-CR4 per leg | 100G-CR2 per leg | −50% |
| NCCL transport | NET/IB (RoCE) | NET/IB (RoCE) | same |
| NCCL channels | 64 | 64 | same |
| NCCL init | Complete, no errors | Complete, no errors | same |
| Ping latency | ~0.6 ms | ~0.6 ms | same |

RoCE works cleanly through the switch at 100G. `ib_write_bw` shows a small ~4.5% drop — close to theoretical expectation since the CX7 NIC processes are not the bottleneck even at 200G.

### Decode Performance

| Model | 2×200G (Jul 31) | 4×100G (Aug 1) | Delta | Notes |
|-------|-----------------|----------------|-------|-------|
| MiniMax M2.7 AWQ | **40.2 tok/s** | **40.0 tok/s** | **−0.5%** | Within noise. Matches Spark Arena reference. |
| DeepSeek V4 Flash FP8 | **39.2 tok/s** | **39.1 tok/s** | **−0.3%** | With MTP spec-decode (2 tokens). |
| MiMo V2.5 NVFP4 | **25.4 tok/s** | **25.1 tok/s** | **−1.2%** | Heaviest model, largest MoE. |

**All three models are within ±1.2% of their 200G numbers.** This is well within run-to-run variance (~2-3% based on our 4-run samples).

## Analysis: Why 100G = 200G for Decode

The interconnect is **not the bottleneck** at these batch sizes. Here's why:

### 1. Decode is latency-bound, not bandwidth-bound
During single-request decode (batch size = 1), the all-reduce collective transfers a small tensor — the model's hidden dimension. For MiniMax M2.7, that's roughly 6 KB per all-reduce. At 100G, that takes microseconds. The GPU compute (MoE expert routing, attention, FFN) takes milliseconds. The network is 1000× faster than the compute.

### 2. `ib_write_bw` confirms headroom
Even at 100G, `ib_write_bw` shows 12.7 GB/s. The all-reduce during decode moves KB of data, not GB. The fabric has orders of magnitude more bandwidth than decode needs.

### 3. The original +26-33% improvement was about fixing broken RoCE, not raw speed
The Jul 31 gains came from fixing `ibv_modify_qp` error 61 (Socket fallback → RoCE), not from 200G bandwidth. Once RoCE works, the transport cost is negligible at both 100G and 200G for decode. The 200G link would matter more for:
- Large-batch prefill (many concurrent requests)
- Training (gradient sync is bandwidth-bound)
- KV cache transfer during pipeline parallelism

### 4. When would 200G actually matter?
If you're running high-concurrency serving (batch size 16+, multiple simultaneous users), the all-reduce tensor grows proportionally. At some batch size, 100G becomes the bottleneck. Our single-request benchmark doesn't hit that regime. Future testing with `bench-suite.sh` (multi-request lm_eval tasks) could reveal the crossover point.

## Lessons

1. **GID index is not portable across ports or nodes.** Always verify with `show_gids` before pinning `NCCL_IB_GID_INDEX`. When in doubt, omit it — NCCL's auto-selection works.
2. **Ray adds ~15-20 GB of unified memory overhead** on Grace-Blackwell. For memory-tight configs, use vLLM's native `--nnodes` multi-node launch instead.
3. **100G breakout is a viable, zero-penalty alternative to 200G** for DGX Spark decode workloads. The CRS812 supports both cable geometries with different lane-grouping configs.
4. **Don't assume faster interconnect = faster inference.** Profile the transport (`ib_write_bw`, NCCL bench) before chasing bandwidth upgrades. Decode is almost always compute-bound.
5. **Lane grouping for 4×100G differs from 2×200G.** 2×200G needs all 8 lanes enabled (group masters at 1 and 5). 4×100G needs lane pairs at 1-2 and 5-6 (for legs 1 and 3). Wiring legs 1 and 2 is a trap — leg 2 isn't covered by this grouping.

## Config Artifacts

### RouterOS lane config (4×100G, legs 1+3)
```
/interface ethernet disable [find name~"qsfp56-dd-1"]
/interface ethernet enable qsfp56-dd-1-1,qsfp56-dd-1-2,qsfp56-dd-1-5,qsfp56-dd-1-6
/interface ethernet disable [find name~"qsfp56-dd-2"]
/interface ethernet enable qsfp56-dd-2-1,qsfp56-dd-2-2,qsfp56-dd-2-5,qsfp56-dd-2-6
/interface bridge port add bridge=bridge interface=qsfp56-dd-1-1
/interface bridge port add bridge=bridge interface=qsfp56-dd-1-5
/interface bridge port add bridge=bridge interface=qsfp56-dd-2-1
/interface bridge port add bridge=bridge interface=qsfp56-dd-2-5
```

### NCCL environment (100G RoCE, second ports)
```
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
# Do NOT set NCCL_IB_GID_INDEX — GID layout differs per node on f1 ports
```

### Spark fabric IPs (second ports, don't survive reboot)
```bash
# 9105 (head)
sudo nmcli device set enp1s0f1np1 managed no
sudo ip addr add 10.10.10.1/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

# bdea (worker)
sudo nmcli device set enp1s0f1np1 managed no
sudo ip addr add 10.10.10.2/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

### Launch command
```bash
# MiniMax via native multi-node (no Ray)
TOPOLOGY=crs812-100g bash scripts/manual-2node-vllm-launch.sh

# DeepSeek V4 / MiMo via Ray (smaller models, Ray overhead tolerable)
bash spark-cluster-manager/scripts/<model>/launch.sh
```

## What's Next

- **High-concurrency test:** Run the full `bench-suite.sh` (lm_eval, multi-request) to find the batch size where 100G starts to lag 200G
- **Dual-rail bonding:** Use both 100G legs per cage (rails 1+2) with NCCL multi-rail to test if aggregate 200G via 2×100G matches native 200G
- **PFC/ECN:** Enable priority flow control on the CRS812 for 100G ports — may improve RoCE under bursty collective patterns
- **NCCL_NCHANNELS tuning:** Test whether reducing channels from 64 to 16-32 reduces memory overhead without hurting throughput
