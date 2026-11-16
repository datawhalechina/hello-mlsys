#!/usr/bin/env bash
# 诊断2：验证 NCCL_P2P_DISABLE=1（强制 proxy/CPU-staging 路径）能否让异构双卡首个 all_reduce 完成。
#
# 背景（diag1 结论，见 results/rccl_diag_stdout.log + rccl_diag_a/b.log）：
#   init/ring/tree 全部 OK，但 RCCL 拓扑误判 isAllCudaP2p=1，而 HSA 实际
#   "Could not enable P2P between dev 1 and dev 0"（×8）→ 走 SHM/direct/direct
#   直连通道 → GPU↔GPU DMA 永不完成 → 首个 all_reduce 干净超时（90s/600s 看门狗
#   均触发，sum_ok=False，双 rank SIGABRT）。RCCL 自警缺 "iommu=pt"（"can lead to
#   hang"），且 cuMem 自动启用上限 gfx1250（gfx1100 无 cuMem）。HSA_FORCE_FINE_GRAIN_PCIE=1 无效。
#
# 本脚本（两卡都需空闲）：
#   ① mini（90s 看门狗，NCCL_P2P_DISABLE=1）→ 快速判定首个集合能否完成；
#   ② mini 通过（exit 0）→ 立即跑全量 14 点探针（run_rccl_probe.sh，NCCL_P2P_DISABLE=1
#      由本脚本 export 传入 torchrun），含 GPU[1] 污染门。
#   ③ mini 失败 → exit 4（引擎口径判 not_covered，见 rccl_diag_c.log）。
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
echo "state at launch: pids=$(busy_pids_any || true) max_used=$(max_used || true) B"

export NCCL_P2P_DISABLE=1
echo "=== mini-diag: NCCL_P2P_DISABLE=1 (proxy / CPU-staging) ==="
set +e
NCCL_DEBUG=INFO timeout 200 torchrun --nproc_per_node=2 rccl_minidiag.py 2>&1 | tee results/rccl_diag_c.log
RC=${PIPESTATUS[0]}
set -e
echo "mini exit=$RC"
if [ "$RC" -ne 0 ]; then
  echo "NCCL_P2P_DISABLE=1 也无法完成首个 all_reduce → RCCL 引擎口径判定 not_covered（详见 rccl_diag_c.log）"
  exit 4
fi

echo "=== mini OK → 全量 14 点探针（NCCL_P2P_DISABLE=1 已由环境传入）==="
set +e
bash run_rccl_probe.sh 2>&1 | tee results/rccl_probe_p2poff.log
RC2=${PIPESTATUS[0]}
set -e
echo "full probe exit=$RC2"
[ "$RC2" -eq 0 ] || exit "$RC2"
echo "done"
