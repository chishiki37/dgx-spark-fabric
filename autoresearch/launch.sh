#!/bin/bash
# Launch helper: starts both nodes, waits for health, returns
set -e

RANK0_SCRIPT="/home/vikassridhar/ds4-autoresearch/serve-ds4-rank0.sh"
BDEA="edgexpert-bdea"

echo "[launch] Cleaning up old containers..."
docker stop ds4-rank0 2>/dev/null && docker rm ds4-rank0 2>/dev/null || true
ssh $BDEA "docker stop ds4-rank1 2>/dev/null && docker rm ds4-rank1 2>/dev/null" || true
echo "[launch] Clean."

echo "[launch] Copying rank1 script to bdea..."
scp /home/vikassridhar/ds4-autoresearch/serve-ds4-rank1.sh $BDEA:/home/vikassridhar/ds4-autoresearch/serve-ds4-rank1.sh 2>/dev/null || \
scp /home/vikassridhar/ds4-autoresearch/serve-ds4-rank1.sh $BDEA:/tmp/serve-ds4-rank1.sh

echo "[launch] Starting rank 1 (bdea)..."
ssh $BDEA "bash /home/vikassridhar/ds4-autoresearch/serve-ds4-rank1.sh 2>/dev/null || bash /tmp/serve-ds4-rank1.sh"

echo "[launch] Starting rank 0 (9105)..."
sleep 2
bash $RANK0_SCRIPT

echo "[launch] Both nodes started. Waiting for health..."
# Wait up to 10 minutes for model load
for i in $(seq 1 120); do
    if curl -sf --max-time 5 http://localhost:8000/health >/dev/null 2>&1; then
        echo "[launch] Server healthy after ${i}x5s!"
        exit 0
    fi
    # Check if containers are still alive
    if ! docker ps --format '{{.Names}}' | grep -q ds4-rank0; then
        echo "[launch] ERROR: rank0 container died"
        docker logs ds4-rank0 2>&1 | tail -20
        exit 1
    fi
    ssh $BDEA "docker ps --format '{{.Names}}'" 2>/dev/null | grep -q ds4-rank1 || {
        echo "[launch] ERROR: rank1 container died"
        ssh $BDEA "docker logs ds4-rank1 2>&1 | tail -20"
        exit 1
    }
    sleep 5
done

echo "[launch] ERROR: Server didn't become healthy in 10 minutes"
docker logs ds4-rank0 2>&1 | tail -20
exit 1
