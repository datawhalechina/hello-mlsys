"""第 5 章引擎口径：双卡 RCCL All-Reduce α–β 标定（P=2，PCIe 2-hop）。

- calibration 12 点（64KB–128MB，×2 跨距）+ holdout 2 点（16KB / 256MB，测前冻结）；
- 每点 5 warmup + 30 次 CUDA event 计时取中位数；
- 最小二乘拟合 T = a + b·Data（P=2：a = 2α_eff，b = 1/BW_eff）；
- holdout 只外推验证，不参与拟合；
- 与 nominal 账本（pcie4_x16 2-hop：α=4µs，25 GB/s）逐点对比，
  gap 分解：α 差 = 引擎启动/同步开销，β 差 = 有效带宽折损。

用法（宿主，双卡）：
  torchrun --nproc_per_node=2 code/part1-modeling/chapter5/rccl_probe.py
"""
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from comm_ledger import ring_ar_s  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

# experiment-draft.yaml measurement_plan 冻结网格（KB）
# yaml 按字节指数记（2^16 B..2^27 B = 64KB..128MB）；KB 指数 = 6..17。
# 历史 bug：曾误用 range(16,28)（把字节指数当 KB 指数 → 16MB..128GB），
# 本机因首个 AR hang 未及暴露；v100x2 上 16GB 点 OOM 后修正（2026-11-16）。
CAL_KB = [2 ** k for k in range(6, 18)]      # 64KB .. 128MB, 12 点
HOLDOUT_KB = [16, 262144]                    # 16KB（α 外推）/ 256MB（β 外推）
WARMUP, ITERS = 5, 30
DTYPE = torch.bfloat16                       # 与账本 BYTES_ELEM=2 同口径


def bench_ar(t: torch.Tensor) -> dict:
    """单次 all_reduce 时延统计（ms：中位数/最小/p90）。"""
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


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dev = f"cuda:{rank}"
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl")         # ROCm 上即 RCCL
    checks = {}

    # ① 正确性（bf16 整数域，避免舍入争议）
    t = torch.full((4096,), rank + 1, dtype=DTYPE, device=dev)
    dist.all_reduce(t)
    torch.cuda.synchronize()
    checks["sum_correct"] = bool(torch.equal(t, torch.full((4096,),
                                                           float(world + 1), dtype=DTYPE, device=dev)))

    # ② P2P 可达性（无 P2P 时 RCCL 走 host 中转，2-hop 成立）
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
                  f"t={r['t_ms']} ms")
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

    # P=2：T_ms = a_ms + b_ms·Data 中 a = 2α_eff，b = 1e3/BW（BW: B/s）
    alpha_eff_us = a_ms * 1e3 / (2 * (world - 1))
    bw_eff_gbps = 1e3 / b_ms / 1e9                     # b [ms/B] → BW [GB/s]

    # ⑤ holdout 外推验证（用拟合值预测，不用 nominal）
    for r in hold:
        t_pred = a_ms + b_ms * r["bytes"]
        r["pred_t_ms"] = round(t_pred, 6)
        r["err_pct"] = round((t_pred - r["t_ms"]) / r["t_ms"] * 100, 2)

    # ⑥ 与 nominal 账本对比（2-hop：α=4µs，25 GB/s）。公式直接复用
    #    comm_ledger.ring_ar_s（Ring-AR 带宽项系数 2(P−1)/P、ms=D/(BW·1e6)），
    #    与账本单源同口径，避免手写系数/单位出错。
    a2 = 2 * common.TOPOLGY["pcie4_x16"]["alpha_us"]   # 4 µs（2 跳 × 2 µs/跳）
    bw = common.TOPOLGY["pcie4_x16"]["bw_gbps"]
    nominal = []
    for r in cal + hold:
        t_nom = ring_ar_s(world, r["bytes"], a2, bw) * 1e3
        nominal.append({"msg_kb": r["msg_kb"], "t_nominal_ms": round(t_nom, 6),
                        "t_measured_ms": r["t_ms"],
                        "measured_over_nominal": round(r["t_ms"] / t_nom, 3)})

    payload = {
        "backend": "RCCL (torch.distributed 'nccl' on ROCm)",
        "world_size": world,
        "dtype": str(DTYPE),
        "warmup": WARMUP, "iters": ITERS,
        "checks": checks,
        "p2p_access_0_1": p2p,
        "p2p_link_line": common.env_block()["p2p_link_line"],
        "fit": {"a_ms": a_ms, "b_ms_per_byte": b_ms, "r2": r2,
                "alpha_eff_us": alpha_eff_us,
                "bw_eff_gbps": bw_eff_gbps,
                "n_cal_points": len(cal),
                "resid_ms": [round(v, 6) for v in resid]},
        "rows": rows,
        "nominal_comparison": nominal,
        "ledger_run": "ch5-ledger-20261116T063929Z（nominal 2-hop 预测 256MB=10.745 ms）",
        "env": common.env_block(),
    }
    p = common.save_json("rccl_probe.json", payload)
    print(f"\nsaved {p}")
    print(f"  fit: α_eff={alpha_eff_us:.2f} µs  BW_eff={bw_eff_gbps:.2f} GB/s  R²={r2:.5f}")
    for r in hold:
        print(f"  holdout {r['msg_kb']} KB: measured {r['t_ms']} ms, "
              f"pred {r['pred_t_ms']} ms, err {r['err_pct']}%")
    print(f"  p2p access 0<->1: {p2p}")


if __name__ == "__main__":
    main()
