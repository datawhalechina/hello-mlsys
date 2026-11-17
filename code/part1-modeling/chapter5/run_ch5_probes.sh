#!/usr/bin/env bash
# 第 5 章真机探针（宿主直跑：miniconda3 torch 2.16.0.dev20261019+rocm10.1）
# 纪律：experiment-draft.yaml 的 prediction.stored_at 已先于本脚本任何一步。
#   ① pcie_probe   设备口径：pinned H2D/D2H（7900 XTX；主卡被他人任务占用 not_covered）
#   ② gloo 2/4 rank 语义验证（纯 CPU，显式非多卡性能）
# RCCL 双卡标定见 run_rccl_probe.sh（需等主卡 9070 XT 释放，避免带宽竞争）。
set -euo pipefail
cd "$(dirname "$0")"
R=results
mkdir -p "$R"

echo "== ① PCIe 1-hop pinned H2D/D2H（7900 XTX, GPU[0]）=="
HSA_VISIBLE_DEVICES=0 python3 pcie_probe.py

echo "== ② gloo CPU All-Reduce 语义（2 rank / 4 rank）=="
torchrun --nproc_per_node=2 gloo_semantics.py
torchrun --nproc_per_node=4 gloo_semantics.py
python3 - "$R" <<'EOF'
import json, sys
from pathlib import Path
R = Path(sys.argv[1])
merged = {f"{n}rank": json.loads((R / f"gloo_semantics_{n}rank.json").read_text())
          for n in (2, 4)}
(R / "gloo_semantics.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False))
print("merged ->", R / "gloo_semantics.json")
EOF

echo "== done =="
