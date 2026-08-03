#!/usr/bin/env bash
# safe_cleanup.sh — Anti-hang safety: kill all DS4 containers, free GPU, verify clean state
# Called BEFORE and AFTER every experiment. NEVER hangs.
set -euo pipefail

TIMEOUT=30
echo "[cleanup] Starting safe cleanup..."

# Step 1: Stop containers (hard timeout per node)
echo "[cleanup] Stopping ds4 containers on 9105..."
timeout $TIMEOUT docker rm -f ds4-rank0 2>/dev/null || true
echo "[cleanup] Stopping ds4 containers on bdea..."
timeout $TIMEOUT ssh -o ConnectTimeout=5 edgexpert-bdea 'docker rm -f ds4-rank1 2>/dev/null' 2>/dev/null || true

# Step 2: Kill any lingering vLLM/python processes on GPU
echo "[cleanup] Killing lingering GPU processes..."
timeout 10 bash -c 'pkill -f "dsv4-vllm-entrypoint" 2>/dev/null || true'
timeout 10 ssh -o ConnectTimeout=5 edgexpert-bdea 'pkill -f "dsv4-vllm-entrypoint" 2>/dev/null || true' 2>/dev/null || true

# Step 3: Wait for GPU memory to settle
echo "[cleanup] Waiting for GPU memory to settle..."
for i in $(seq 1 10); do
  sleep 2
  # Check if any compute processes are still on GPU
  PROC=$(nvidia-smi --query-compute-apps=pid 2>/dev/null | grep -c "[0-9]" || echo "0")
  PROC_BDEA=$(timeout 5 ssh -o ConnectTimeout=5 edgexpert-bdea 'nvidia-smi --query-compute-apps=pid 2>/dev/null | grep -c "[0-9]"' 2>/dev/null || echo "0")
  if [ "$PROC" = "0" ] && [ "$PROC_BDEA" = "0" ]; then
    echo "[cleanup] GPU processes cleared after ${i}x2s"
    break
  fi
  echo "[cleanup] Still waiting (9105=$PROC, bdea=$PROC_BDEA procs)..."
done

# Step 4: Verify clean state
echo ""
echo "[cleanup] === GPU STATE ==="
echo "9105:"
nvidia-smi 2>&1 | grep -E "GB10|python|Xorg" | head -5
echo "bdea:"
timeout 5 ssh -o ConnectTimeout=5 edgexpert-bdea 'nvidia-smi 2>&1 | grep -E "GB10|python|Xorg"' 2>/dev/null | head -5

# Step 5: Verify fabric still up (quick ping)
if timeout 3 ping -c 1 -W 1 10.10.10.2 >/dev/null 2>&1; then
  echo "[cleanup] Fabric link: UP ✅"
else
  echo "[cleanup] ⚠️ Fabric link: DOWN — may need bounce before next experiment"
fi

# Step 6: Verify NFS mount still accessible
if ls /mnt/ds4-ablit/config.json >/dev/null 2>&1; then
  echo "[cleanup] NFS mount: OK ✅"
else
  echo "[cleanup] ⚠️ NFS mount: STALE — remounting..."
  timeout 10 mount -t nfs -o ro,vers=3,proto=rdma,port=20049,mountproto=tcp,rsize=1048576,wsize=1048576 \
    10.10.10.2:/home/vikassridhar/.cache/huggingface/hub/models--drowzeys--keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32 \
    /mnt/ds4-ablit 2>/dev/null || echo "[cleanup] ❌ NFS remount failed"
fi

echo "[cleanup] Done."
