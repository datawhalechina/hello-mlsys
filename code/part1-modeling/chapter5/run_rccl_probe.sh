#!/usr/bin/env bash
# 第 5 章 RCCL 双卡 α–β 标定（P=2，PCIe 2-hop）——需 GPU[0]+GPU[1] 全程独占。
# GPU[1] 与他人短生命周期容器共享：
#   空闲判据 = rocm-smi --showpids 中 GPU[1] 无任何 KFD 进程（主）
#              且 VRAM used < 500 MB（兜底，防 KFD 输出解析失败时误判忙）；
#   连续 3 次（间隔 20s）判定空闲；上限 ~3h，超时 exit 2。
# 测量全程由监视线程每 5s 采样 GPU[1]（pids + used），写入 results/rccl_gpu1_watch.log。
# 污染判定（测后）：窗口内任一采样出现"非本探针"的 GPU[1] 进程，或 VRAM 远超本探针足迹
#   → 标记 contaminated（数据不可用，需重跑）。
#   本探针 GPU[1] 足迹实测 ~0.5 GB；判定阈值 VRAM_DIRTY=2 GB（4× 足迹，远小于他人
#   容器 >8 GB 的占用，且与 PID 时序无关——上次竞态即他人 5s 后挤入且占用 8.9–15 GB）。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

gpu1_busy_pids() {
  rocm-smi --showpids 2>/dev/null | awk '
    $1=="PID" {gcol=3; next}
    gcol && $1 ~ /^[0-9]+$/ {
      n=split($gcol, g, ","); for (i=1;i<=n;i++) if (g[i]=="1") print $1
    }'
}
used1() { rocm-smi --showmeminfo vram 2>/dev/null | grep '^GPU\[1\]' | grep 'Used' | awk '{print $NF}'; }
THRESH=524288000      # 500 MB 空闲判据
VRAM_DIRTY=2147483648 # 2 GB 污染判据

# 递归收集 $1 的进程树（不含根），用于"哪些 PID 属于本探针"。
collect_tree() {
  local root=$1 out="" k kids
  kids=$(pgrep -P "$root" 2>/dev/null || true)
  for k in $kids; do
    out="$out $k"
    out="$out $(collect_tree "$k")"
  done
  echo "$out"
}

idle=0
for i in $(seq 1 540); do
  p=$(gpu1_busy_pids); u=$(used1)
  if [ -z "$p" ] && { [ -z "$u" ] || [ "$u" -lt "$THRESH" ]; }; then
    idle=$((idle+1)); [ "$idle" -ge 3 ] && break
  else
    idle=0
  fi
  [ $((i % 9)) -eq 0 ] && echo "waiting GPU[1] free (pids=${p:-none}, used=${u:-?} B, check $i/540)..."
  sleep 20
done
p=$(gpu1_busy_pids)
if [ -n "$p" ]; then
  echo "TIMEOUT: GPU[1] still busy after ~3h (pids=$p); 需人工确认后再跑"
  exit 2
fi
echo "GPU[1] free, running RCCL probe"

T_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
rm -f results/rccl_gpu1_watch.log results/.rccl_dirty
set +e
torchrun --nproc_per_node=2 rccl_probe.py &
TORCH_PID=$!
set -e
# 等 worker 拉起后收集本探针进程树（纯 ps 口径，与 GPU 时序竞态无关）
OUR_PIDS=" $TORCH_PID"
for i in $(seq 1 30); do
  OUR_PIDS=" $TORCH_PID $(collect_tree "$TORCH_PID")"
  [ "$(echo $OUR_PIDS | wc -w)" -ge 3 ] && break
  sleep 2
done
echo "our pids: $OUR_PIDS"

( for j in $(seq 1 600); do
    pids=$(gpu1_busy_pids | tr '\n' ','); u=$(used1)
    echo "$(date -u +%H:%M:%S) pids=$pids used=$u" >> results/rccl_gpu1_watch.log
    dirty=0
    [ -n "$u" ] && [ "$u" -gt "$VRAM_DIRTY" ] && dirty=1
    if [ -n "$pids" ]; then
      for p in $(echo "$pids" | tr ',' ' '); do
        case " $OUR_PIDS " in *" $p "*) : ;; *) dirty=1;; esac
      done
    fi
    [ "$dirty" -eq 1 ] && echo "$(date -u +%H:%M:%S) DIRTY pids=$pids used=$u our=[$OUR_PIDS]" >> results/.rccl_dirty
    sleep 5
  done ) &
WATCH=$!
set +e
wait "$TORCH_PID"
RC=$?
set -e
kill "$WATCH" 2>/dev/null || true
wait "$WATCH" 2>/dev/null || true
T_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "run window: $T_START .. $T_END (torchrun rc=$RC, our_pids=$OUR_PIDS)"
if [ -s results/.rccl_dirty ]; then
  echo "CONTAMINATED: GPU[1] 在测量窗口内出现他人进程或异常 VRAM，results/rccl_probe.json 作废需重跑"
  cat results/.rccl_dirty
  exit 3
fi
[ "$RC" -eq 0 ] || { echo "torchrun failed rc=$RC"; exit "$RC"; }
echo "clean window, RCCL probe OK"
