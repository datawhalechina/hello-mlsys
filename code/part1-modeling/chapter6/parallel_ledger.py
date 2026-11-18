#!/usr/bin/env python3
"""第 6 章主模拟器（Feishu：本章模拟器是主角）：DP / TP / PP 切分账本。

产出 results/parallel_ledger.json（正文数字的唯一来源之一）：
  1. 显存账本（推理口径）：三方案每 rank 水位（权重/KV/激活缓冲/运行时预留）
     × 在盘 Qwen3-4B 与公开 config Qwen3-8B/14B/32B × 9070 XT 15.9 GiB /
     W7900D 48 GB（TODO 占位）；含"显存冗余"指标（P 卡上权重总占用 vs 单份）；
  2. 通信时间占比：每 step T_comm/(t_step+T_comm)
     —— DP=0（推理口径，副本独立）；TP=逐层 2×Ring-AR（Megatron 口径）；
     PP=激活 p2p + 流水线气泡 (P−1)/(m+P−1)（GPipe 稳态口径）；
     计算时间 t_step = F(N)/(P·PEAK·η)（Ch2 常数）—— 三方案每 rank 同值，
     差异全在显存与通信；
  3. 判定条件（"切分收益被通信吃掉"）：
     —— TP：prefill 占比随 S 的曲线 + share≤10% 的最小 S（S_min）；
        decode（N=B）占比 vs B；
     —— PP：气泡比 (P−1)/(m+P−1) 表 + 气泡≤10% 所需 m_min = 9(P−1)；
     —— 跨节点：节点内（PCIe/NVLink）vs 跨节点（RoCE 100G）占比对比，
        "TP 组必须限制在同一节点内"判定；holdout = Qwen3-32B（用后即烧）；
  4. 边界场景（Feishu 实验项 ②）：8B 卡 16 GB 边界 / 14B 卡 2×16 GB 与 48 GB
     （TODO 占位）的可行性矩阵 → 切分决策输入；
  5. 训练口径对照（Feishu 核心内容：DP 梯度同步 / FSDP）：16B/param 账本 +
     每 step 梯度 AR / 逐层 all-gather 占比；
  6. animation_data：动画演示数据源（每 rank 显存水位段 + TP 单层通信时序 +
     PP 气泡 Gantt），供 docs 交互动画消费。

公式与 Ch5 同源（ring_ar_s/allgather_s/p2p_s + nominal 拓扑档案 + v100x2 实测档）；
计算口径与 Ch1/Ch2 同源（PEAK_TFLOPS×η）。FLOPs 逐层精确分解（Ch5 32·N·D² 同族）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

GIB = 2 ** 30
COMMBUDGET_PCT = 10.0            # 工程预算：通信占比上限（正文注明为工程惯例，非标准）
S_GRID = [2 ** k for k in range(0, 16)]          # 1 .. 32768 tokens
B_GRID = [1, 2, 4, 8, 16, 32]
M_GRID = [1, 2, 4, 8, 16, 32]                    # PP 微批数
P_GRID = [2, 4]                                  # 账本主网格（真机仅 P=2，P=4 模拟器口径）


def all_models() -> dict:
    out = {name: dict(cfg) for name, cfg in common.PUBLISHED_MODELS.items()}
    c4b = common.model_cfg_4b()
    out["Qwen3-4B"] = dict(c4b, source=c4b["source"])
    return out


def kv_per_token_bytes(c: dict, kv_heads_per_rank: int, n_layers_per_rank: int) -> float:
    """每 rank 每 token KV 字节数（GQA：KV 头按 rank 切分；层按 rank 切分）。"""
    return 2 * n_layers_per_rank * kv_heads_per_rank * c["head_dim"] * common.BYTES_ELEM


# ---------------- 显存账本（推理口径） ----------------

def memory_ledger(c: dict, vram_gib: float, scheme: str, p: int,
                  batch: int, seq: int) -> dict:
    """一个模型 × 一个方案 × P rank × 固定服务负载（batch×seq）的每 rank 水位。"""
    w = common.weight_params(c)
    w_gib = w["total"] * common.BYTES_ELEM / GIB
    d = c["hidden"]
    act_chunk = seq * d * common.BYTES_ELEM / GIB        # 一个激活块（S tokens）

    # KV：全模型每 token 字节（GQA 全头）
    kv_tok = 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * common.BYTES_ELEM / GIB

    if scheme == "dp":
        # 副本独立：权重全量 + 自己那 1/P 批次的 KV 池
        weights = w_gib
        kv = batch / p * seq * kv_tok
        act = act_chunk
        note = "权重全量冗余 ×P；每副本独立服务 B/P 请求，KV 池 = (B/P)·S"
    elif scheme == "tp":
        weights = w_gib / p
        kvh = max(1, -(-c["n_kv_heads"] // p))           # ceil；P>kv_heads 时整头冗余
        kv = batch * seq * (2 * c["n_layers"] * kvh * c["head_dim"]
                            * common.BYTES_ELEM / GIB)
        act = 2 * act_chunk                              # 2 个 in-flight AR 缓冲
        note = (f"权重切 1/P；KV 头 {c['n_kv_heads']}→{kvh}/rank"
                + ("（整头冗余）" if kvh > c["n_kv_heads"] / p else ""))
    elif scheme == "pp":
        layers_r = -(-c["n_layers"] // p)                # 均匀 PP 每 rank 层数
        weights = w_gib * layers_r / c["n_layers"]
        kv = batch * seq * kv_tok * layers_r / c["n_layers"]
        act = 2 * act_chunk                              # workspace + 跨边界 transit
        note = f"层均匀切 {layers_r}/rank；KV 随层切；transit 缓冲 = 一个激活块"
    else:
        raise ValueError(scheme)

    overhead = common.OVERHEAD_GIB
    per_rank = round(weights + kv + act + overhead, 3)
    usable = vram_gib * common.UTIL - overhead
    return {
        "scheme": scheme, "p": p, "card_gib": vram_gib,
        "workload": {"batch": batch, "seq": seq},
        "per_rank_gib": {"weights": round(weights, 3), "kv_pool": round(kv, 3),
                         "act_buffer": round(act, 3), "overhead": overhead,
                         "total": per_rank},
        "usable_gib": round(usable, 3),
        "feasible": per_rank <= usable,
        "weights_redundancy_total_gib": round(weights * p, 3),   # P 卡上权重合计
        "note": note,
    }


# ---------------- 计算时间 ----------------

def step_time(c: dict, n_tok: int, seq: int, p: int, scheme: str) -> dict:
    """每 rank 每 step 计算时间（秒）。三方案每 rank 同值 = F(N)/(P·PEAK·η)；
    PP 含 embed/lm_head 不均衡（末 rank 多 lm_head），按 rank 精确。"""
    hw = common.HARDWARE["9070xt"]
    peak_eta = hw["peak_tflops"] * 1e12 * hw["eta"]
    F = common.fwd_flops(c, n_tok, seq)
    out = {"n_tok": n_tok, "seq": seq, "p": p, "scheme": scheme,
           "flops_total": int(F)}
    if scheme == "dp":
        t = common.fwd_flops(c, n_tok / p, seq) / peak_eta
        out["per_rank"] = [round(t, 9)] * p
        out["note"] = "DP 切 token：每副本跑全模型，N/P tokens"
    elif scheme == "tp":
        t = F / (p * peak_eta)
        out["per_rank"] = [round(t, 9)] * p
        out["note"] = "TP 切层内权重：每 rank F/N 份，精确 1/P"
    elif scheme == "pp":
        per_layer = (F - 2 * n_tok * c["vocab"] * c["hidden"]) / c["n_layers"]
        lm = 2 * n_tok * c["vocab"] * c["hidden"]
        ts = []
        for r in range(p):
            layers_r = -(-c["n_layers"] // p) - (1 if r == p - 1 and c["n_layers"] % p else 0)
            t = (layers_r * per_layer + (lm if r == p - 1 else 0)) / peak_eta
            ts.append(round(t, 9))
        out["per_rank"] = ts
        out["note"] = "PP 切层间：每 rank L/P 层；末 rank 多 lm_head（轻微不均衡）"
    else:
        raise ValueError(scheme)
    return out


# ---------------- 通信时间占比 ----------------

def tp_comm(c: dict, n_tok: int, seq: int, p: int, topo: str) -> dict:
    """TP=P 每 step：逐层 2×Ring-AR（payload = N·d·bf16，Megatron 口径）。"""
    prof = common.TOPOLOGY[topo]
    payload = n_tok * c["hidden"] * common.BYTES_ELEM
    t_ar = common.ring_ar_s(p, payload, prof["alpha_us"], prof["bw_gbps"])
    t_comm = c["n_layers"] * 2 * t_ar
    t_comp = common.fwd_flops(c, n_tok, seq) / (
        p * common.HARDWARE["9070xt"]["peak_tflops"] * 1e12
        * common.HARDWARE["9070xt"]["eta"])
    return {"topo": topo, "payload_mb": round(payload / 2 ** 20, 3),
            "t_ar_ms": round(t_ar * 1e3, 4),
            "t_comm_ms": round(t_comm * 1e3, 4),
            "t_comp_ms": round(t_comp * 1e3, 4),
            "share_pct": round(t_comm / (t_comp + t_comm) * 100, 3)}


def pp_comm(c: dict, n_tok: int, p: int, m: int, topo: str) -> dict:
    """PP=P：激活 p2p（单跳）+ 流水线气泡。

    m=1（单流无重叠）：总时间 = P·t_step + (P−1)·T_p2p，
      占比 = (P−1)·T_p2p / 总时间（传输开销口径）；
    m≥2（GPipe 稳态，m 微批）：气泡比 = (P−1)/(m+P−1)（计算侧空转口径，
      与带宽无关——气泡只由级数与微批数决定）+ 摊薄后的 p2p 项。
    """
    prof = common.TOPOLOGY[topo]
    payload = n_tok * c["hidden"] * common.BYTES_ELEM
    t_p2p = common.p2p_s(payload, prof["alpha_us"], prof["bw_gbps"])
    # 每 step 计算（单 rank，L/P 层，近似 F/P）
    t_step = common.fwd_flops(c, n_tok, n_tok) / (
        p * common.HARDWARE["9070xt"]["peak_tflops"] * 1e12
        * common.HARDWARE["9070xt"]["eta"])
    if m == 1:
        total = p * t_step + (p - 1) * t_p2p
        share = (p - 1) * t_p2p / total
    else:
        bubble = (p - 1) / (m + p - 1)
        # 稳态每 step 有效时间 = t_step·(m+P−1)/m 中占 t_step 的比例 = m/(m+P−1)
        share = bubble
    return {"topo": topo, "m": m, "payload_mb": round(payload / 2 ** 20, 3),
            "t_p2p_ms": round(t_p2p * 1e3, 4),
            "t_step_ms": round(t_step * 1e3, 4),
            "bubble_or_transfer_share_pct": round(share * 100, 3),
            "caliber": "m=1: 传输开销占比；m≥2: GPipe 稳态气泡比 (P−1)/(m+P−1)"}


def tp_smin(c: dict, p: int, topo: str, budget: float = COMMBUDGET_PCT) -> float | None:
    """prefill（B=1）share≤budget 的最小序列长 S（网格内；带宽主导区近似单调）。"""
    for s in S_GRID:
        r = tp_comm(c, s, s, p, topo)
        if r["share_pct"] <= budget:
            return s
    return None


def crossnode_verdict(c: dict, p: int, batch: int, seq_ctx: int = 4096,
                      budget: float = COMMBUDGET_PCT) -> dict:
    """"TP 组必须限制在同一节点内"判定，双工作负载各判一次：
      decode（N=B，ctx=seq_ctx）：share 对 B 平移不变（t_comm、t_comp 均 ∝N）；
      prefill（B=1，N=S=seq_ctx）：计算含 S² 注意力项，share 随 S 下降——
        "必须同节点"判定实际落在这个区间。
    三档标签：
      infeasible_even_innode：连最优节点内链路（NVLink）都 > 预算
        → 该规模+负载下 TP 不可行，与拓扑无关，需降 P 或改 PP；
      must_innode：最优节点内 ≤ 预算 < 跨节点 → TP 仅节点内可行；
      crossnode_ok：两档都 ≤ 预算（本数据集未出现，如实留空）。
    """
    out = {"p": p, "batch": batch, "seq_ctx": seq_ctx}
    for wl, (n_tok, seq) in {"decode": (batch, seq_ctx),
                             "prefill": (seq_ctx, seq_ctx)}.items():
        rows = {f"innode_{t}": tp_comm(c, n_tok, seq, p, t)
                for t in ("pcie4_x16", "nvlink300")}
        rows["cross_roce100g"] = tp_comm(c, n_tok, seq, p, "roce100g")
        inn = min(r["share_pct"] for k, r in rows.items() if k.startswith("innode"))
        cross = rows["cross_roce100g"]["share_pct"]
        if inn > budget:
            verdict = "infeasible_even_innode"
        elif cross > budget:
            verdict = "must_innode"
        else:
            verdict = "crossnode_ok"
        out[wl] = {"shares_pct": {k: r["share_pct"] for k, r in rows.items()},
                   "best_innode_pct": inn, "crossnode_pct": cross,
                   "verdict": verdict, "detail": rows}
    return out


# ---------------- 训练口径对照 ----------------

def training_caliber(c: dict, p: int, n_tok: int, seq: int) -> dict:
    """Feishu 核心内容：DP 显存冗余与梯度同步；FSDP 切分 + 逐层 all-gather。
    混合精度 Adam：16 B/param（2 fp16 W + 2 fp16 grad + 4 fp32 master + 8 fp32 m/v）。"""
    w = common.weight_params(c)["total"]
    hw = common.HARDWARE["9070xt"]
    peak_eta = hw["peak_tflops"] * 1e12 * hw["eta"]
    # fwd+bwd ≈ 3×F
    F = common.fwd_flops(c, n_tok, seq)
    t_step_train = 3 * F / (p * peak_eta)            # DP：N/P tokens ×3
    # DP：每 rank 全量 16B/param；每 step 梯度 AR（bf16 梯度 2B/param）
    dp_mem = w * 16 / GIB
    grad_payload = w * common.BYTES_ELEM
    t_ar = common.ring_ar_s(p, grad_payload,
                            common.TOPOLOGY["pcie4_x16"]["alpha_us"],
                            common.TOPOLOGY["pcie4_x16"]["bw_gbps"])
    dp_share = t_ar / (t_step_train + t_ar)
    # FSDP：16B/P 每 rank；fwd+bwd 逐层权重 all-gather（2×L 次，payload=层参数×2B）
    fsdp_mem = w * 16 / p / GIB
    per_layer_bytes = common.weight_params(c)["per_layer"] * common.BYTES_ELEM
    t_ag = common.allgather_s(p, per_layer_bytes,
                              common.TOPOLOGY["pcie4_x16"]["alpha_us"],
                              common.TOPOLOGY["pcie4_x16"]["bw_gbps"])
    fsdp_share = c["n_layers"] * 2 * t_ag / (t_step_train + c["n_layers"] * 2 * t_ag)
    return {
        "n_tok": n_tok, "seq": seq, "p": p,
        "bytes_per_param_mixed_precision": 16,
        "dp": {"per_rank_gib": round(dp_mem, 3),
               "grad_ar_payload_gib": round(grad_payload / GIB, 3),
               "t_ar_ms": round(t_ar * 1e3, 3),
               "t_step_train_ms": round(t_step_train * 1e3, 3),
               "comm_share_pct": round(dp_share * 100, 3),
               "topo": "pcie4_x16（节点内档）"},
        "fsdp": {"per_rank_gib": round(fsdp_mem, 3),
                 "allgather_payload_per_layer_mib": round(per_layer_bytes / 2 ** 20, 3),
                 "t_ag_ms": round(t_ag * 1e3, 4),
                 "comm_share_pct": round(fsdp_share * 100, 3),
                 "topo": "pcie4_x16（节点内档）"},
        "note": "训练口径对照（Feishu 核心内容 DP/FSDP）：fwd+bwd ≈ 3×F，Ch2 常数"
                "（195 TFLOPS bf16 × η=0.6243）；显存为方案间对比账本，非单卡可行性"
                "（8B DP 122 GiB/rank 远超 16/48 GB 档，需 FSDP 切分或更大卡档）；"
                "非 serving 主线",
    }


# ---------------- 动画数据 ----------------

def animation_data(mname: str, c: dict, vram_gib: float, p: int, batch: int, seq: int) -> dict:
    """动画演示数据源（Feishu 作业④：数据来自模拟器逐 rank 显存账本与通信时间线）。"""
    mem = {s: memory_ledger(c, vram_gib, s, p, batch, seq) for s in ("dp", "tp", "pp")}
    # TP 单层时序（prefill 1 request，S tokens）：attn 计算 → AR → mlp 计算 → AR
    prof = common.TOPOLOGY["pcie4_x16"]
    n_tok = seq
    payload = n_tok * c["hidden"] * common.BYTES_ELEM
    t_ar = common.ring_ar_s(p, payload, prof["alpha_us"], prof["bw_gbps"]) * 1e3
    d, d_q, d_kv = c["hidden"], c["n_q_heads"] * c["head_dim"], c["n_kv_heads"] * c["head_dim"]
    per_layer = (2 * n_tok * (d * d_q + 2 * d * d_kv + d_q * d)
                 + 6 * n_tok * d * c["inter"] + 4 * n_tok * seq * d_q)
    t_comp_layer = per_layer / (p * common.HARDWARE["9070xt"]["peak_tflops"] * 1e12
                                * common.HARDWARE["9070xt"]["eta"]) * 1e3
    t_attn = (2 * n_tok * (d * d_q + 2 * d * d_kv + d_q * d)
              + 4 * n_tok * seq * d_q) / (p * common.HARDWARE["9070xt"]["peak_tflops"]
                                          * 1e12 * common.HARDWARE["9070xt"]["eta"]) * 1e3
    t_mlp = (6 * n_tok * d * c["inter"]) / (p * common.HARDWARE["9070xt"]["peak_tflops"]
                                            * 1e12 * common.HARDWARE["9070xt"]["eta"]) * 1e3
    tp_layer = {"attn_ms": round(t_attn, 4), "ar_ms": round(t_ar, 4),
                "mlp_ms": round(t_mlp, 4),
                "n_layers": c["n_layers"],
                "payload_mb": round(payload / 2 ** 20, 2),
                "topo": "pcie4_x16"}
    # PP 气泡 Gantt：P=4（演示口径，真机仅 P=2）、m=8 微批，GPipe 稳态
    p_g, m_g = 4, 8
    t_stage = common.fwd_flops(c, seq, seq) / (
        p_g * common.HARDWARE["9070xt"]["peak_tflops"] * 1e12
        * common.HARDWARE["9070xt"]["eta"]) * 1e3
    pp_gantt = {"p": p_g, "m": m_g, "t_stage_ms": round(t_stage, 3),
                "bubble_pct": round((p_g - 1) / (m_g + p_g - 1) * 100, 2),
                "note": "GPipe 稳态：每 stage 先等 (P−1) 个微批进入（气泡区），"
                        "之后连续流水；气泡比与带宽无关"}
    return {"model": mname, "card_gib": vram_gib, "p": p,
            "workload": {"batch": batch, "seq": seq},
            "memory": {s: {"per_rank_gib": mem[s]["per_rank_gib"],
                           "feasible": mem[s]["feasible"],
                           "usable_gib": mem[s]["usable_gib"]} for s in mem},
            "tp_layer_timeline": tp_layer,
            "pp_gantt": pp_gantt,
    }


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-holdout", action="store_true",
                    help="跳过 32B holdout（holdout 纪律：用后即烧，复跑需显式）")
    args = ap.parse_args()

    models = all_models()
    cards = {"9070xt": common.HARDWARE["9070xt"], "w7900d": common.HARDWARE["w7900d"]}
    B, S = 8, 4096                       # 固定服务负载（跨方案可比）

    res = {
        "generated_at": common.now_iso(),
        "usage": "python3 parallel_ledger.py [--skip-holdout]",
        "calibration_version":
            "ch5-nominal-topology-profiles-v1 + ch5-nvlink-measured-v100x2（测前冻结）；"
            "ch2-compute-constants（PEAK=195 TFLOPS bf16, η=0.6243）",
        "workload": {"batch": B, "seq": S,
                     "note": "固定服务负载：batch B=8 × 上下文 S=4096，跨方案可比"},
        "comm_budget_pct": COMMBUDGET_PCT,
        "formulas": {
            "t_step": "t_step = F(N)/(P·PEAK·η)（三方案每 rank 同值；PP 末 rank 多 lm_head）",
            "tp_comm": "逐层 2×Ring-AR，payload = N·d·bf16（Megatron 口径）",
            "pp_bubble": "GPipe 稳态气泡比 = (P−1)/(m+P−1)（与带宽无关）；m=1 单流：(P−1)·T_p2p 传输项",
            "dp_inference": "副本独立，前向通信 = 0（训练口径见 training_caliber）",
            "flops": "逐层精确：2N(d·d_q+2d·d_kv+d_q·d)+6N·d·inter+4N·S·d_q，+2N·V·d（lm_head）",
        },
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "source"} | {"source": v["source"]}
                   for k, v in models.items()},
        "memory": {}, "comm_share": {}, "judgment": {}, "boundary": {},
        "training_caliber": {}, "animation_data": {},
    }

    # 1. 显存账本（键含卡号：{card}_{scheme}_p{P}）
    for mname, c in models.items():
        rows = {}
        for ckey, hw in cards.items():
            for s in ("dp", "tp", "pp"):
                for p in P_GRID:
                    rows[f"{ckey}_{s}_p{p}"] = memory_ledger(c, hw["vram_gib"], s, p, B, S)
        res["memory"][mname] = rows

    # 1b. 计算等价（本章主线）：三方案每 rank t_step = F(N)/(P·PEAK·η) 同值
    eq = {}
    for mname, c in models.items():
        eq[mname] = {}
        for s in ("dp", "tp", "pp"):
            pre = step_time(c, S, S, 2, s)      # prefill：B=1，S tokens
            dec = step_time(c, B, S, 2, s)      # decode：N=B tokens
            eq[mname][s] = {
                "prefill_p2_ms": [round(t * 1e3, 4) for t in pre["per_rank"]],
                "decode_p2_ms": [round(t * 1e3, 4) for t in dec["per_rank"]],
                "note": pre["note"],
            }
    res["compute_equivalence"] = eq

    # 2. 通信占比（prefill S 曲线 + decode B 点）
    for mname, c in models.items():
        rows = {}
        for s in ("tp",):
            for p in P_GRID:
                pref = {}
                for t in common.TOPOLOGY:
                    pref[t] = [tp_comm(c, s_, s_, p, t)["share_pct"] for s_ in S_GRID]
                decode = {}
                for t in common.TOPOLOGY:
                    decode[t] = [tp_comm(c, b_, 4096, p, t)["share_pct"] for b_ in B_GRID]
                rows[f"{s}_p{p}"] = {"prefill_share_by_s": dict(zip(S_GRID, pref["pcie4_x16"])) |
                                     {f"{t}_by_s": pref[t] for t in common.TOPOLOGY if t != "pcie4_x16"},
                                     "decode_share_by_b": dict(zip(B_GRID, decode["pcie4_x16"])) |
                                     {f"{t}_by_b": decode[t] for t in common.TOPOLOGY if t != "pcie4_x16"},
                                     "prefill_points_pcie": [tp_comm(c, s_, s_, p, "pcie4_x16") for s_ in S_GRID],
                                     "decode_points_pcie": [tp_comm(c, b_, 4096, p, "pcie4_x16") for b_ in B_GRID]}
        # PP 气泡表
        pp_rows = {}
        for p in P_GRID:
            for m in M_GRID:
                for t in ("pcie4_x16",):
                    pp_rows[f"p{p}_m{m}"] = pp_comm(c, B * 1, p, m, t)
        rows["pp"] = pp_rows
        res["comm_share"][mname] = rows

    # 3. 判定条件
    j = {}
    for mname, c in models.items():
        j[mname] = {
            "tp_smin_prefill": {
                f"p{p}_{t}": tp_smin(c, p, t) for p in P_GRID for t in common.TOPOLOGY},
            "pp_bubble": {f"p{p}": {"m_min_for_10pct": 9 * (p - 1),
                                    "table": {m: round((p - 1) / (m + p - 1) * 100, 2)
                                              for m in M_GRID}} for p in P_GRID},
        }
    res["judgment"] = j

    # 4. 边界场景（Feishu 实验项 ②：卡 16/48 GB 边界，迫使真实切分决策）
    bd = {}
    c8 = models["Qwen3-8B"]
    single8 = memory_ledger(c8, common.VRAM_9070_GIB, "dp", 1, B, S)   # p=1 = 单卡不切分
    bd["Qwen3-8B_16gb_boundary"] = {
        "single_card_9070xt": single8,
        "single_card_verdict": "weights 16.4 GB > 15.9×0.9−0.5=13.81 GiB 可用 → 单卡不可行，必须切分",
        "dp_p2": res["memory"]["Qwen3-8B"]["9070xt_dp_p2"],
        "tp_p2": res["memory"]["Qwen3-8B"]["9070xt_tp_p2"],
        "pp_p2": res["memory"]["Qwen3-8B"]["9070xt_pp_p2"],
    }
    c14 = models["Qwen3-14B"]
    single48 = memory_ledger(c14, common.VRAM_W7900D_GIB, "dp", 1, B, S)
    bd["Qwen3-14B_2x16gb_vs_48gb"] = {
        "tp_p2_9070xt": res["memory"]["Qwen3-14B"]["9070xt_tp_p2"],
        "tp_p4_9070xt": res["memory"]["Qwen3-14B"]["9070xt_tp_p4"],
        "pp_p2_9070xt": res["memory"]["Qwen3-14B"]["9070xt_pp_p2"],
        "single_48gb": single48,
        "note": "48 GB 卡参数 TODO 嘉航（容量占位）：14B 权重 29.5 GB 单张 48 GB 可放下 → "
                "切分决策取决于卡容量档位，正是 Feishu 要求的'真实切分决策'压力场景",
    }
    res["boundary"] = bd

    # 5. 训练口径对照（8B 为例）
    res["training_caliber"] = {
        "Qwen3-8B": training_caliber(models["Qwen3-8B"], p=2, n_tok=B, seq=S),
    }

    # 6. 动画数据（8B × 2×15.9GiB 主场景）
    res["animation_data"] = animation_data("Qwen3-8B", models["Qwen3-8B"],
                                           common.VRAM_9070_GIB, p=2, batch=B, seq=S)

    # 7. holdout（Feishu 实验项 ③：换模型规模复查"TP 组必须同节点"；用后即烧）
    if not args.skip_holdout:
        hold = {}
        c32 = models["Qwen3-32B"]
        for p in P_GRID:
            hold[f"tp_p{p}"] = crossnode_verdict(c32, p, batch=B)
        # 对照：8B/14B 同口径
        hold["control_8B"] = {f"p{p}": crossnode_verdict(models["Qwen3-8B"], p, B)
                              for p in P_GRID}
        hold["control_14B"] = {f"p{p}": crossnode_verdict(models["Qwen3-14B"], p, B)
                               for p in P_GRID}
        hold["burned_at"] = common.now_iso()
        res["holdout_32b"] = hold

    p = common.save_json("parallel_ledger.json", res)
    print(f"saved {p}")

    # 控制台速览
    for mname, c in models.items():
        w_gib = common.weight_params(c)["total"] * common.BYTES_ELEM / GIB
        print(f"\n=== {mname}（weights {w_gib:.2f} GiB bf16）")
        for key in ("9070xt_dp_p2", "9070xt_tp_p2", "9070xt_pp_p2", "9070xt_tp_p4"):
            r = res["memory"][mname][key]
            ok = "OK " if r["feasible"] else "NO "
            print(f"  [{ok}] {key:6s} 9070xt: per-rank {r['per_rank_gib']['total']:7.3f} GiB"
                  f" (w {r['per_rank_gib']['weights']:7.3f} + kv {r['per_rank_gib']['kv_pool']:5.3f}"
                  f" + act {r['per_rank_gib']['act_buffer']:5.3f} + o/{r['per_rank_gib']['overhead']})"
                  f" / usable {r['usable_gib']:.2f}")
        for p in P_GRID:
            sm = res["judgment"][mname]["tp_smin_prefill"][f"p{p}_pcie4_x16"]
            print(f"  TP P={p}: prefill S_min(≤{COMMBUDGET_PCT}%, PCIe) = {sm}")


if __name__ == "__main__":
    main()
