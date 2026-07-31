# DGX Spark Fabric — 2-Node Cluster via MikroTik CRS812

**How I fixed a hidden 25-33% decode-speed tax on my 2× NVIDIA DGX Spark (GB10) cluster by replacing the direct 200G DAC with a switched 200G fabric — RDMA finally worked, NCCL tripled, and every model got ~a third faster. Then I tested 4×100G vs 2×200G breakouts and found zero difference — decode is compute-bound, not network-bound.**

## Results

### CRS812 fabric vs direct link (Jul 31, 2026 — 2×200G breakout)

| Model (2-node TP=2, vLLM) | Direct DAC | Via CRS812 | Delta |
|---|---|---|---|
| MiMo V2.5 NVFP4 (171 GB) | ~19 tok/s (Socket NCCL) | **25.4 tok/s** | **+33%** |
| DeepSeek V4 Flash FP8 (162.7B MoE) | ~30 tok/s (Socket) | **39.2 tok/s** | **+31%** |
| MiniMax M2.7 AWQ (122 GB) | ~32 tok/s (RoCE direct) | **40.2 tok/s** | **+26%** |

Fabric-level numbers: NCCL all-reduce BusBW **40-51 GB/s** with GPUDirect RDMA (vs 12-15 GB/s Socket), TCP iperf3 **107 Gbit/s** per rail (vs ~15 direct), `ib_write_bw` 13.3 GB/s with zero `ibv_modify_qp` errors.

### 4×100G vs 2×200G breakout comparison (Aug 1, 2026)

| Model | 2×200G | 4×100G | Delta |
|---|---|---|---|
| MiniMax M2.7 AWQ | 40.2 tok/s | **40.0 tok/s** | −0.5% |
| DeepSeek V4 Flash FP8 | 39.2 tok/s | **39.1 tok/s** | −0.3% |
| MiMo V2.5 NVFP4 | 25.4 tok/s | **25.1 tok/s** | −1.2% |

**Verdict: no difference.** Decode at single-request batch size is compute-bound — the interconnect has orders of magnitude more bandwidth than the all-reduce needs. See [`docs/100g-vs-200g-comparison.md`](docs/100g-vs-200g-comparison.md) for the full analysis.

## Why the direct link was slow

1. **RoCE broken point-to-point** — QP setup fails (`ibv_modify_qp` error 61) on the CX7↔CX7 DAC; NCCL silently falls back to TCP sockets (~3-4× slower).
2. **No buffering** — NCCL all-reduce bursts overrun the ARM CPUs' drain rate; without switch buffers, TCP melts into retransmit storms (15 vs 107 Gb/s on identical hardware).
3. **Interface asymmetry** — working ports had different names per node (`f1` vs `f0`), and the second CX7 card's capital-P device name crashes naive `NCCL_IB_HCA` filters.
4. **Brittle bring-up** — PCIe hotplug parks NICs at boot; fabric IPs don't survive reboots.

## Hardware

- 2× NVIDIA DGX Spark (GB10, 121 GB unified memory each)
- MikroTik CRS812 DDQ (CRS812-8DS-2DQ-2DDQ-RM, RouterOS v7)
- NADDOD Q2Q56-400G-CU1 — 1m passive DAC, QSFP56-DD 400G → 2× QSFP56 200G
- Each breakout leg → one Spark's CX7 Card 1 (`enp1s0f0np0`)

## Repo contents

- `docs/REPORT.md` — full narrative writeup: direct-link fix, CRS812 fabric, +26-33% gains
- `docs/100g-vs-200g-comparison.md` — exhaustive 4×100G vs 2×200G breakout comparison (GID traps, Ray OOM, zero decode difference)
- `docs/crs812-setup.md` — complete switch setup guide incl. the lane-grouping discovery
- `docs/cluster-operations.md` — day-2 ops: topologies, per-model containers, pitfalls
- `routeros/crs812-config.rsc` — annotated RouterOS commands (copy-paste into terminal)
- `scripts/nccl-bench.py` — NCCL all-reduce bandwidth test (run in a `--privileged` container on both nodes)
- `scripts/manual-2node-vllm-launch.sh` — 2-node vLLM launcher with `TOPOLOGY=crs812|crs812-100g|direct` switch
- `recipes/` — sparkrun recipes for MiniMax M2.7 AWQ + DeepSeek V4 Flash on this fabric

## Quick start

1. Cable: breakout DD-end → CRS812 QSFP-DD cage 1; legs → each Spark's CX7 Card-1 port. Push until click (QSFP-DD latches lie).
2. Switch: paste `routeros/crs812-config.rsc` (lanes + bridge).
3. Sparks: `sudo nmcli device set enp1s0f0np0 managed no && sudo ip addr add 10.10.10.X/24 dev enp1s0f0np0 && sudo ip link set enp1s0f0np0 up` (X=1 head, X=2 worker).
4. Verify fabric: `ping`, then `scripts/nccl-bench.py` in a `--privileged` vLLM container (`NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3`).
5. Serve: `TOPOLOGY=crs812 scripts/manual-2node-vllm-launch.sh` or the sparkrun recipes in `recipes/`.

## Gotchas (each cost me real time)

- **Enable ALL 8 lane-interfaces** of the QSFP-DD cage on the MikroTik. Enabling only the two 200G "masters" (`dd-1-1`+`dd-1-5`) gives 50G per leg. The masters need their filler lanes enabled to absorb 4 serdes lanes each.
- **For 4×100G breakouts:** enable lane pairs `dd-X-1,dd-X-2` + `dd-X-5,dd-X-6` (legs 1 and 3). Do NOT wire legs 1+2 — leg 2 isn't covered by this grouping. Do NOT enable `dd-X-1` through `dd-X-4` contiguously — that collapses into one 200G group.
- **Add the two masters to the bridge** — default RouterOS bridge only has ether ports; without this, links come up but ARP fails.
- **GID index is NOT portable across ports or nodes.** `NCCL_IB_GID_INDEX=3` works for `rocep1s0f0` (first ports) but breaks on `rocep1s0f1` (second ports) because the GID layout differs per machine. When in doubt, omit it — NCCL auto-selects correctly.
- **Ray adds ~15-20 GB unified memory overhead** on Grace-Blackwell. For memory-tight models (MiniMax at 122 GB), use vLLM's native `--nnodes` multi-node launch instead of Ray.
- **Containers need `--privileged`** (or `/dev/infiniband` mapped) or NCCL reports `NET/IB : No device found`.
- **NCCL 2.30:** `NCCL_NET=IB` is fatal if the IB plugin can't init — omit it, pin `NCCL_IB_HCA` instead.
- The RJ45 port labeled **CONSOLE is serial**, not Ethernet. Use the MGMT port (factory IP 192.168.88.1, no DHCP server — set your client static).

## Scaling

This fabric is the foundation for an 8-node quant lab (968 GB unified memory): GLM-5.x, Kimi-K2.6-NVFP4, Nemotron-550B-class models, NVFP4/INT4/AWQ/GGUF comparisons, pruning, and speculative decoding. The CRS812 has the port headroom; each Spark needs one breakout leg.

## License

MIT. Benchmarks and configs provided as-is — your mileage (and cable vendor) may vary.
