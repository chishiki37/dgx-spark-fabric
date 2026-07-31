# DGX Spark Multi-Node Cluster Operations

## Two Fabric Topologies (both valid)

**A. Switched fabric via MikroTik CRS812 (current, preferred)** — 2×200G breakout into QSFP-DD cage 1, RoCE+GDR works (40-51 GB/s NCCL BusBW), scales to 4-8 nodes. Full recipe: `references/mikrotik-crs812-setup.md`. CRS812-topology recipe templates carry the `-crs812` suffix.
- Head: `spark-head` — CX7 iface `enp1s0f0np0`, 10.10.10.1; RDMA dev `rocep1s0f0`
- Worker: `spark-worker` / `gb10` — CX7 iface `enp1s0f0np0`, 10.10.10.2; RDMA dev `rocep1s0f0`
- Containers need `--privileged` (or /dev/infiniband mapped) for RDMA
- NCCL 2.30: `NCCL_NET=IB` is fatal-if-unavailable (no silent Socket fallback)

**B. Direct DAC Spark↔Spark (legacy fallback)** — single 200G DAC, no switch; RoCE broken (error 61), Socket NCCL only (~12-15 GB/s). Recipe retained in `references/multi-node-setup-guide.md`.
- Head: `spark-head` — iface `enp1s0f1np1`, 10.10.10.1; RDMA `rocep1s0f1`
- Worker: `spark-worker` — iface `enp1s0f0np0`, 10.10.10.2; RDMA `rocep1s0f0`

**Check which topology is active before launching:** `ethtool <iface> | grep Speed` — or just `ping -c 2 10.10.10.2`. Interface/device names differ between topologies (f0 vs f1 ports); recipes must match the active one.

## Critical Warnings

**⚠️ CX7 IPs don't survive reboots:** After every reboot, re-add them with `sudo ip addr add`. NetworkManager will fight manual assignments — use `nmcli device set <iface> managed no` to stop it. Without CX7 IPs, RoCE fails (`ibv_modify_qp` error 61) and NCCL falls back to Socket (~12–15 Gb/s vs ~200 Gb/s RoCE; measured decode delta 1.6–2×).

**⚠️ Before deploying any model:** check if a working recipe already exists on disk. Every model that was previously served has a recipe or launch script saved — check `~/spark-vllm-docker/recipes/`, `/tmp/*.yaml`, `/tmp/*-head.sh`, and session history via `session_search` before building anything from scratch. Reuse is ALWAYS faster than reinventing.

**⚠️ Before launching ANY multi-node model:** verify CX7 IPs are up. Check with `ping -c 2 10.10.10.2`. If down, give the user the exact `sudo ip addr add` commands — never ask for their sudo password.

## Torchrun Direct Launch (bypasses sparkrun)

Launch containers directly with `torchrun` for TP=2 across both nodes:

- Both nodes use `--network host --ipc host`, mount `~/.cache/huggingface:/root/.cache/huggingface`
- **Worker first** (`--node-rank=1`, `--master-addr=10.10.10.1 --master-port=29500`), then **head** (`--node-rank=0`)
- Worker waits for head; model loads in ~4 min after both connect
- Model files must exist in HF cache on BOTH nodes

**⚠️ Caveat (found Jul 23, 2026):** vLLM V1's default multiprocess executor requires all TP ranks on ONE node — it fails with `World size (2) is larger than the number of available GPUs (1) in this node` when launched via torchrun across 2 nodes. Cross-node TP=2 then requires `--distributed-executor-backend ray` (manual Ray cluster).

## Per-Model Containers

| Model | Container | Notes |
|---|---|---|
| DeepSeek V4 Flash (FP8) | `vllm-node-dsv4` (jasl fork) | Needs MTP/MLA/FP8 kernels; use `--moe-backend auto` (NOT wna16/marlin). Mainline eugr-nightly gives only ~6 tok/s; dsv4: ~30 tok/s direct-link (Jul 23 fleet run) → **39.2 tok/s via CRS812 RoCE** (Jul 31, +31%). |
| MiMo V2.5 NVFP4 | `vllm-node-mimo-v25-nvfp4` + 2 mods | 171 GB. Mods `fix-mimo-v2-vllm` + `fix-modelopt-mixed-mxfp8` on BOTH nodes, `--load-format instanttensor`, `--attention-backend triton_attn_diffkv`, Omni mode, GMU 0.82, ctx 32768, `RAY_memory_monitor_refresh_ms=0`. ~19 tok/s direct-link Socket → **25.4 tok/s via CRS812 RoCE** (Jul 31, +33%). |
| MiniMax M2.7 AWQ | `eugr-nightly` | 122 GB. ~32 tok/s direct-link (Jul 22-23 fleet run) → **40.2 tok/s via CRS812 RoCE** (Jul 31, +26%); ~40 tok/s is Spark Arena ref — fabric now matches it. **Reasoning model** — generates CoT in `reasoning_content` before answer in `content`. |
| Hy3 295B / Laguna S 2.1 | `eugr-nightly` | Recipe: GMU 0.90, --enforce-eager, marlin, MTP spec-1. **Laguna rev 07614121 (Jul 2026):** requantized (67 GiB/15 shards) + thinking-by-default — inline-thinking model now; benchmarks need `max_gen_toks=1024`. |
| Qwen3.6-35B-A3B NVFP4 | `eugr-nightly` | 24 GB — runs **single-node** on head (no Ray, no worker). `VLLM_MARLIN_USE_ATOMIC_ADD=1`, GMU 0.85, kv fp8, 32K ctx. NOT a structured reasoning model (thinks inline in content) → IFEval-chat is safe. Full suite ~45-60 min. |

## Benchmark Suite

- GEN: gsm8k, mbpp, ifeval — `--limit 100`, `HF_ALLOW_CODE_EVAL=1`
- **MBPP also needs the CLI flag `--confirm_run_unsafe_code`** — `HF_ALLOW_CODE_EVAL=1` alone is NOT sufficient on the current lm_eval
- LL: mmlu_stem, arc_challenge, hellaswag — `--limit 100`

**Ready-made suite runner:** `scripts/bench-suite.sh` — copy + set `MODEL_ID`/`TOKPATH` env vars; encodes all pitfalls above.

**Parametrized runner:** `scripts/bench-standard-suite.sh` — runs all 5-6 tasks separately and verifies results JSON.

## Key Pitfalls

1. **CX7 link lost after reboot breaks multi-node launches** — Always verify CX7 IPs before launch
2. **NaN 400 crashes (DeepSeek V4 FP8)** — Run each task as a SEPARATE lm_eval invocation
3. **`tee` masks exit codes** — Use `${PIPESTATUS[0]}` to detect real failure
4. **Exit-code-0 ≠ success** — Always check for the results JSON under `--output_path`
5. **Sparkrun is unreliable (Jul 2026)** — Use direct `torchrun` instead
6. **Ray memory monitor kills large-model workers** — Set `RAY_memory_monitor_refresh_ms=0`
7. **Inline-thinking models truncate GEN benchmarks** — Pass `--gen_kwargs max_gen_toks=1024`
8. **Don't conflate tok/s numbers** — Always attribute model + quantization + transport + container

## Spark Cluster Manager

A web app wraps most operations: FastAPI on **spark-head:8700** (`http://spark-head:8700`), systemd user service `spark-manager.service`, code at `~/spark-cluster-manager/`.

**Check the app/API before hand-rolling launches or benchmark runs** — `POST /api/models/{id}/load`, `/benchmark`, `GET /api/benchmarks/matrix`.

See `references/spark-cluster-manager.md` for the full API + ops map.