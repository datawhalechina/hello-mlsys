"""第 5 章纸面账本：通信代价模型（nominal 拓扑档案 × Ring-AllReduce 公式）。

产出 results/comm_ledger.json（正文数字的唯一来源之一）：
  1. 四档拓扑 × P∈{2,4} 的 Ring-AllReduce / All-Gather 时间曲线（1KB–1GB，21 点）
     + 延迟主导/带宽主导区拐点 D* = P·α·BW；
  2. TP=2/4 单层 Transformer 通信占比预测（Qwen3-1.7B，Megatron 口径：每层 2 次 AR）；
  3. NVLink(300GB/s) 相对其余三档的收益区间；
  4. 真机预测块（先于实测冻结）：PCIe 1-hop H2D/D2H、双卡 2-hop All-Reduce。

可选 --msysim /path/to/mlsys-simulator：与自研模拟器 ring_allreduce_ms 做
"账本对模拟器"公式级对账（同 nominal，应一致到 1e-9 相对误差）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def ring_ar_s(p: int, data_b: float, alpha_us: float, bw_gbps: float) -> float:
    """T_AR = 2(P−1)α + 2·(P−1)/P · D/BW（秒）。"""
    return (2 * (p - 1) * alpha_us * 1e-6
            + 2 * (p - 1) / p * data_b / (bw_gbps * 1e9))


def allgather_s(p: int, data_b: float, alpha_us: float, bw_gbps: float) -> float:
    """T_AG = (P−1)α + (P−1)/P · D/BW（秒；total_bytes 为全量逻辑张量）。"""
    return ((p - 1) * alpha_us * 1e-6
            + (p - 1) / p * data_b / (bw_gbps * 1e9))


def crossover_b(p: int, alpha_us: float, bw_gbps: float) -> float:
    """拐点：α 项 = β 项 ⟺ D* = P·α·BW（字节）。"""
    return p * alpha_us * 1e-6 * bw_gbps * 1e9


def curves() -> dict:
    out = {}
    for name, prof in common.TOPOLGY.items():
        a, b = prof["alpha_us"], prof["bw_gbps"]
        for p in (2, 4):
            pts = []
            for kb in common.MSG_GRID_KB:
                d = kb * 1024
                t = ring_ar_s(p, d, a, b)
                lat_term = 2 * (p - 1) * a * 1e-6
                bw_term = t - lat_term
                pts.append({
                    "msg_kb": kb,
                    "t_ar_ms": round(t * 1e3, 9),
                    "t_allgather_ms": round(allgather_s(p, d, a, b) * 1e3, 9),
                    "latency_share": round(lat_term / t, 4),   # 1=纯延迟区, 0=纯带宽区
                })
            out[f"{name}_P{p}"] = {
                "alpha_us": a, "bw_gbps": b,
                "crossover_kb": round(crossover_b(p, a, b) / 1024, 2),
                "points": pts,
            }
    return out


def tp_layer_share() -> dict:
    """TP=P 单层 fwd：2 次 All-Reduce（payload = B·S·D·bf16）+ 计算时间（Ch2 校准）。

    计算口径与 Ch1/Ch2 一致：单层 matmul FLOPs ≈ 32·N·D²（N = B·S tokens），
    每 rank 摊 P 份：t_comp = 32·N·D²/(P·PEAK·η)。
    """
    c = common.model_cfg()
    d = c["hidden"]
    n_layers = c["n_layers"]
    out = {"model": c["name"], "hidden": d, "n_layers": n_layers,
           "bytes_elem": common.BYTES_ELEM, "rows": []}
    for n_tok in (64, 256, 1024, 4096):
        payload = n_tok * d * common.BYTES_ELEM      # 每次 AR 的逻辑张量
        flops_layer = 32 * n_tok * d * d             # 单层 matmul（全模型口径）
        for p in (2, 4):
            t_comp = flops_layer / (p * common.PEAK_TFLOPS * 1e12
                                    * common.ETA)     # 每 rank，Ch2 η
            row = {"tokens": n_tok, "tp": p,
                   "payload_mb": round(payload / 2 ** 20, 3),
                   "t_comp_ms": round(t_comp * 1e3, 4)}
            for name in common.TOPOLGY:
                prof = common.TOPOLGY[name]
                t_ar = ring_ar_s(p, payload, prof["alpha_us"], prof["bw_gbps"])
                row[name] = {
                    "t_2ar_ms": round(2 * t_ar * 1e3, 4),
                    "comm_share_pct": round(2 * t_ar / t_comp * 100, 2),
                }
            out["rows"].append(row)
    return out


def nvlink_gain() -> dict:
    """NVLink(300GB/s) 相对其余三档的加速比与收益区间（P=2，Ring AR）。"""
    ref = common.TOPOLGY["nvlink300"]
    out = {"tp": 2, "per_topology": {}}
    for name, prof in common.TOPOLGY.items():
        if name == "nvlink300":
            continue
        gains = []
        for kb in common.MSG_GRID_KB:
            d = kb * 1024
            t_ref = ring_ar_s(2, d, ref["alpha_us"], ref["bw_gbps"])
            t_x = ring_ar_s(2, d, prof["alpha_us"], prof["bw_gbps"])
            gains.append({"msg_kb": kb, "speedup": round(t_x / t_ref, 3)})
        best = max(g2["speedup"] for g2 in gains)
        out["per_topology"][name] = {
            "max_speedup": best,
            "speedup_at_1kb": gains[0]["speedup"],
            "speedup_at_1gb": gains[-1]["speedup"],
            "points": gains,
        }
    return out


def realhw_predictions() -> dict:
    """真机预测块（先于测量冻结，供 experiment-draft.yaml 引用）：

    - PCIe 1-hop（H2D/D2H pinned）：nominal 25 GB/s 上限，工程带 20–25 GB/s；
    - 双卡 2-hop Ring AR（P=2，PCIe 经 CPU 根复合体）：α_eff = 2 跳 × 2 µs，
      BW_eff = 单链 25 GB/s（两个方向复用同一物理链的独立 lane 对）。
    """
    a2 = 2 * common.TOPOLGY["pcie4_x16"]["alpha_us"]
    b = common.TOPOLGY["pcie4_x16"]["bw_gbps"]
    ar2 = []
    for kb in (64, 256, 1024, 4096, 16384, 65536, 262144):
        d = kb * 1024
        ar2.append({"msg_kb": kb,
                    "t_ar_ms": round(ring_ar_s(2, d, a2, b) * 1e3, 6)})
    return {
        "pcie1hop_h2d_d2h": {"bw_gbps_upper": b, "engineering_band_gbps": [20.0, b]},
        "dual_gpu_2hop_ring_ar": {"p": 2, "alpha_eff_us": a2, "bw_eff_gbps": b,
                                  "predictions_ms": ar2},
    }


def msysim_crosscheck(msysim_dir: str) -> dict:
    """账本 vs msys-simulator：同 nominal 参数下公式级对账（应 ~0 相对误差）。"""
    sys.path.insert(0, msysim_dir)
    from mlsys_sim.performance import ring_allreduce_ms  # noqa: E402
    worst = 0.0
    checked = 0
    for name, prof in common.TOPOLGY.items():
        for p in (2, 4):
            for kb in common.MSG_GRID_KB[::4]:
                d = kb * 1024
                mine = ring_ar_s(p, d, prof["alpha_us"], prof["bw_gbps"])
                theirs = ring_allreduce_ms(p, d, prof["bw_gbps"],
                                           prof["alpha_us"]) / 1e3
                rel = abs(mine - theirs) / max(mine, 1e-18)
                worst = max(worst, rel)
                checked += 1
    import subprocess
    ver = subprocess.run(["git", "-C", msysim_dir, "log", "-1", "--format=%h %cd",
                          "--date=iso"], capture_output=True, text=True).stdout.strip()
    return {"msysim_commit": ver or "unknown", "points_checked": checked,
            "max_rel_err": worst,
            "formula": "T_ms=2*(P-1)*alpha_us/1000+2*(P-1)/P*msg_bytes/(bw_gbps*1e6)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msysim", default=None,
                    help="mlsys-simulator 仓库路径（公式级对账）")
    args = ap.parse_args()

    run_id = f"ch5-ledger-{common.now_iso().replace(':', '').replace('-', '')}"
    payload = {
        "run_id": run_id,
        "env": common.env_block(),
        "calibration_version": "nominal-topology-profiles-v1（测前冻结）",
        "formula": {
            "ring_ar": "T = 2(P-1)*alpha + 2*(P-1)/P * Data/BW",
            "allgather": "T = (P-1)*alpha + (P-1)/P * Data/BW",
            "crossover": "D* = P * alpha * BW（alpha 项 = beta 项）",
        },
        "topologies": {k: v["note"] for k, v in common.TOPOLGY.items()},
        "curves": curves(),
        "tp_layer_share": tp_layer_share(),
        "nvlink_gain": nvlink_gain(),
        "realhw_predictions": realhw_predictions(),
    }
    if args.msysim:
        payload["msysim_crosscheck"] = msysim_crosscheck(args.msysim)
    p = common.save_json("comm_ledger.json", payload)
    print(f"saved {p}")
    for k, v in payload["curves"].items():
        print(f"  {k}: crossover {v['crossover_kb']} KB "
              f"(alpha={v['alpha_us']}us, bw={v['bw_gbps']}GB/s)")


if __name__ == "__main__":
    main()
