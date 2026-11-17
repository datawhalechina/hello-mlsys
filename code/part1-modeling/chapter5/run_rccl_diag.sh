#!/usr/bin/env bash
# 双卡 RCCL 最小诊断：等两卡（GPU[0]+GPU[1]）真正空闲 → 跑 rccl_minidiag.py（90s 看门狗）。
# 目的：定位 run2 中"首个 all_reduce 600s hang"的根因，不是标定。
#   Config B 优先：HSA_FORCE_FINE_GRAIN_PCIE=1（GPU↔GPU DMA 走 2-hop PCIe 直接路径，
#                  与 nominal 档案同口径；可绕开 CPU 中转的潜在死锁）。
#   Config B 失败再跑 Config A：默认（P2P off，CPU 中转）。
# 每个 config 外层 timeout 200s 兜底（内层 90s 看门狗会先触发）。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

# 任一卡上有 KFD 进程 → busy（返回该 pid）。
busy_pids_any() {
  rocm-smi --showpids 2>/dev/null | awk '
    $1=="PID" {gcol=3; next}
    gcol && $1 ~ /^[0-9]+$/ {
      n=split($gcol, g, ","); for (i=1;i<=n;i++) if (g[i] !~ /^$/) print $1
    }'
}
max_used() { rocm-smi --showmeminfo vram 2>/dev/null | grep 'Used' | awk '{print $NF}' | sort -n | tail -1; }
THRESH=524288000      # 500 MB：任一卡超过视为忙

idle=0
for i in $(seq 1 540); do
  p=$(busy_pids_any); u=$(max_used)
  if [ -z "$p" ] && { [ -z "$u" ] || [ "$u" -lt "$THRESH" ]; }; then
    idle=$((idle+1)); [ "$idle" -ge 3 ] && break
  else
    idle=0
  fi
  [ $((i % 9)) -eq 0 ] && echo "waiting GPUs free (pids=${p:-none}, max_used=${u:-?} B, check $i/540)..."
  sleep 20
done
p=$(busy_pids_any); u=$(max_used)
echo "state at launch: pids=${p:-none} max_used=${u:-?} B"

echo "=== Config B: HSA_FORCE_FINE_GRAIN_PCIE=1 (GPU<->GPU DMA, 2-hop PCIe) ==="
set +e
HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_DEBUG=INFO timeout 200 \
  torchrun --nproc_per_node=2 rccl_minidiag.py 2>&1 | tee results/rccl_diag_b.log
RC_B=${PIPESTATUS[0]}
set -e
echo "Config B exit=$RC_B"

if [ "$RC_B" -ne 0 ]; then
  echo "=== Config A: default (P2P off, CPU staging) ==="
  set +e
  NCCL_DEBUG=INFO timeout 200 torchrun --nproc_per_node=2 rccl_minidiag.py 2>&1 | tee results/rccl_diag_a.log
  RC_A=${PIPESTATUS[0]}
  set -e
  echo "Config A exit=$RC_A"
fi
echo "diag done"
