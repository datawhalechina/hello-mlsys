"""第 5 章引擎口径（NVLink 层）：v100x2 双卡 NCCL All-Reduce α–β 标定（P=2，NV6 bond）。

本地宿主（7900 XTX + 9070 XT）的 PCIe 2-hop 引擎口径经 5 次 run 法医判定
not_covered（§5.8）。NVLink 层的引擎口径改在 v100x2（6×V100-SXM2-16GB，
GPU 4/5 为 NV6 bond，单 NUMA）上补齐；GPU 0–3 被用户 VLLM 占用，一律不碰。

口径对齐（与本地 rccl_probe.py 同协议，保证跨宿主可比）：
  - calibration 12 点（64KB–128MB，×2 跨距）+ holdout 2 点（16KB / 256MB，
    测前冻结，experiment-draft.yaml measurement_plan，用后即烧）；
  - 每点 5 warmup + 30 次 CUDA event 计时取中位数；
  - 最小二乘 T = a + b·Data（P=2：a = 2α_eff，b = 1/BW_eff）；
  - holdout 只外推验证，不参与拟合；
  - nominal 对比对象 = comm_ledger 的 nvlink300 档案（300 GB/s 单向，
    α=1 µs/跳，A100 口径，Feishu 指定，测前冻结）；本机为 V100 NVLink2，
    代际不同，gap 如实记录，不硬凑。

用法（v100x2 宿主，仅 GPU 4/5）：
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 nccl_probe_v100nv6.py
"""
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from comm_ledger import ring_ar_s  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

CAL_KB = [2 ** k for k in range(6, 18)]      # 64KB .. 128MB, 12 点（yaml 冻结 2^16 B..2^27 B，KB 指数）
HOLDOUT_KB = [16, 262144]                    # 16KB（α 外推）/ 256MB（β 外推）
WARMUP, ITERS = 5, 30
DTYPE = torch.bfloat16                       # 与账本 BYTES_ELEM=2 同口径
TIMEOUT_S = 600                              # experiment-draft.yaml measurement_plan.timeout_s


def bench_ar(t: torch.Tensor) -> dict:
    """单次 all_reduce 时延统计（ms：中位数/最小/p90）。与本地 rccl_probe 同实现。"""
    for _ in range(WARMUP):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(ITERS):
        ev0.record()
        dist.all_reduce(t)
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts_sorted = sorted(ts)
    return {"t_ms": round(statistics.median(ts), 6),
            "min_ms": round(ts_sorted[0], 6),
            "p90_ms": round(ts_sorted[int(0.9 * (len(ts) - 1))], 6)}


def env_block() -> dict:
    """环境与硬件快照（nvidia-smi 版；本地 common.env_block 为 rocm-smi，不通用）。"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=30).stdout
    gpus = []
    for line in out.splitlines():
        if not line.strip():
            continue
        name, mem, drv = [x.strip() for x in line.split(",")]
        gpus.append({"name": name,
                     "vram_mib": int(mem.split()[0]),
                     "driver": drv})
    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
                          timeout=30).stdout
    link = next((l.strip() for l in topo.splitlines() if "NV6" in l), None)
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus_visible": gpus,
        "nvlink_pair_line": link,
        "note": ("CUDA_VISIBLE_DEVICES=4,5（物理 GPU 4/5，NV6 bond，单 NUMA）；"
                 "GPU 0–3 用户 VLLM 占用，全程未触碰"),
    }


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dev = f"cuda:{rank}"
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", timeout=timedelta(seconds=TIMEOUT_S))
    checks = {}

    # ① 正确性（bf16 整数域，避免舍入争议；与本地同实现）
    t = torch.full((4096,), rank + 1, dtype=DTYPE, device=dev)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    checks["sum_correct"] = bool(torch.equal(t, torch.full((4096,),
                                                           float(world + 1), dtype=DTYPE, device=dev)))

    # ② P2P 可达性（NVLink 直连预期 True）
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))

    # ③ 计时网格
    rows = []
    for kb in CAL_KB + HOLDOUT_KB:
        nb = kb * 1024
        numel = nb // DTYPE.itemsize
        t = torch.randn(numel, dtype=DTYPE, device=dev)
        r = bench_ar(t)
        rows.append({"msg_kb": kb, "bytes": nb, "holdout": kb in HOLDOUT_KB, **r})
        if rank == 0:
            print(f"  {kb:>7} KB ({'holdout' if kb in HOLDOUT_KB else 'calib  '}): "
                  f"t={r['t_ms']} ms", flush=True)
        del t
    torch.cuda.synchronize()

    dist.destroy_process_group()
    if rank != 0:
        return

    cal = [r for r in rows if not r["holdout"]]
    hold = [r for r in rows if r["holdout"]]

    # ④ 最小二乘 T_ms = a_ms + b_ms·Data_KB（P=2：a = 2α_eff，b = 1/BW_eff）
    X = np.array([[1.0, r["bytes"]] for r in cal])
    y = np.array([r["t_ms"] for r in cal])
    (a_ms, b_ms), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ np.array([a_ms, b_ms])
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    alpha_eff_us = a_ms * 1e3 / (2 * (world - 1))
    bw_eff_gbps = 1e3 / b_ms / 1e9                     # b [ms/B] → BW [GB/s]

    # ⑤ holdout 外推验证（用拟合值预测，不用 nominal）
    for r in hold:
        t_pred = a_ms + b_ms * r["bytes"]
        r["pred_t_ms"] = round(t_pred, 6)
        r["err_pct"] = round((t_pred - r["t_ms"]) / r["t_ms"] * 100, 2)

    # ⑥ 与 nominal 账本对比：nvlink300（α=1 µs/跳 → P=2 两跳 a=2 µs；BW=300 GB/s）。
    #    公式直接复用 comm_ledger.ring_ar_s，与账本单源同口径。
    a2 = 2 * common.TOPOLGY["nvlink300"]["alpha_us"]
    bw = common.TOPOLGY["nvlink300"]["bw_gbps"]
    nominal = []
    for r in cal + hold:
        t_nom = ring_ar_s(world, r["bytes"], a2, bw) * 1e3
        nominal.append({"msg_kb": r["msg_kb"], "t_nominal_ms": round(t_nom, 6),
                        "t_measured_ms": r["t_ms"],
                        "measured_over_nominal": round(r["t_ms"] / t_nom, 3)})

    payload = {
        "backend": "NCCL (torch.distributed 'nccl', CUDA build; V100-SXM2-16GB x2, NV6 bond)",
        "host": "v100x2 (z-lab-v100x2, 192.168.1.211)",
        "world_size": world,
        "dtype": str(DTYPE),
        "warmup": WARMUP, "iters": ITERS,
        "checks": checks,
        "p2p_access_0_1": p2p,
        "fit": {"a_ms": a_ms, "b_ms_per_byte": b_ms, "r2": r2,
                "alpha_eff_us": alpha_eff_us,
                "bw_eff_gbps": bw_eff_gbps,
                "n_cal_points": len(cal),
                "resid_ms": [round(v, 6) for v in resid]},
        "rows": rows,
        "nominal_comparison": nominal,
        "nominal_profile": "nvlink300（300 GB/s 单向，α=1 µs/跳，A100 口径；本机为 V100 NVLink2，代际差如实记录）",
        "ledger_run": "ch5-ledger-20261116T063929Z（nominal-topology-profiles-v1，测前冻结）",
        "env": env_block(),
    }
    RESULTS = Path(__file__).parent / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "nccl_probe_v100nv6.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved {p}")
    print(f"  fit: α_eff={alpha_eff_us:.2f} µs  BW_eff={bw_eff_gbps:.2f} GB/s  R²={r2:.5f}")
    for r in hold:
        print(f"  holdout {r['msg_kb']} KB: measured {r['t_ms']} ms, "
              f"pred {r['pred_t_ms']} ms, err {r['err_pct']}%")
    print(f"  p2p access 0<->1: {p2p}")


if __name__ == "__main__":
    main()
