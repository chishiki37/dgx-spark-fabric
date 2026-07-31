#!/usr/bin/env python3
"""NCCL all-reduce bandwidth test across 2 GPUs on 2 DGX Spark nodes.

Usage: Run inside the vLLM container on BOTH nodes simultaneously (worker first).

⚠️ Container MUST have RDMA char devices: launch with --privileged (or
--device=/dev/infiniband/uverbs* --device=/dev/infiniband/rdma_cm). Without
them NCCL reports "NET/IB : No device found" even though sysfs shows the HCAs.

⚠️ NCCL 2.30+: NCCL_NET=IB is FATAL if the IB plugin can't init (no silent
Socket fallback). Omit NCCL_NET and pin NCCL_IB_HCA instead — NCCL
auto-selects IB when the device opens, Socket otherwise.

CRS812 switch fabric (current, Jul 31 2026 — RoCE works, 40-51 GB/s BusBW):
On 9105 (rank 0):
  docker run --rm --gpus all --network host --ipc host --privileged \
    -v /tmp/nccl-bench.py:/tmp/nccl-bench.py \
    -e RANK=0 -e WORLD_SIZE=2 -e MASTER_ADDR=10.10.10.1 -e MASTER_PORT=29500 \
    -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_DEBUG=INFO \
    ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest \
    python3 /tmp/nccl-bench.py

On bdea (rank 1): same but RANK=1 (device names identical on both nodes
in the switch topology: cabled port is f0 on Card 1).

Direct DAC fallback (legacy — Socket NCCL only, RoCE broken err 61):
  drop NCCL_IB_* vars; head iface enp1s0f1np1, worker enp1s0f0np0.

Expected via CRS812 RoCE: ~40-51 GB/s busbw at 128-512MB (NET/IB/0 in debug).
Socket fallback: ~12-15 GB/s. Check NCCL_DEBUG for "NET/IB" vs "NET/Socket".
"""
import torch
import torch.distributed as dist
import os
import time


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    print(f"[Rank {rank}] Initializing NCCL... world_size={world_size}, master={master_addr}:{master_port}")
    print(f"[Rank {rank}] CUDA device: {torch.cuda.get_device_name(0)}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(0)

    # Warmup
    tensor = torch.randn(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark different sizes
    sizes_mb = [1, 4, 16, 64, 128, 256, 512]

    if rank == 0:
        print(f"\n{'Size (MB)':>10} | {'ms/iter':>10} | {'Algo BW (GB/s)':>15} | {'Bus BW (GB/s)':>14}")
        print("-" * 60)

    for size_mb in sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 4
        tensor = torch.randn(n_elements, device="cuda", dtype=torch.float32)

        n_iters = 50
        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_per_iter = size_mb * 1024 * 1024 * 4  # float32
        algobw = (bytes_per_iter * n_iters) / elapsed
        busbw = algobw * (2 * (world_size - 1)) / world_size

        if rank == 0:
            print(f"  {size_mb:6d} MB | {elapsed/n_iters*1000:8.2f} ms | {algobw/1e9:12.2f} | {busbw/1e9:12.2f}")

    if rank == 0:
        print("\nDone!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
