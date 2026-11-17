#!/usr/bin/env python3
"""第 6 章真机行为验证探针（Feishu：gloo 后端多卡 2–4 rank 真实 TP/PP）。

启动：torchrun --nproc_per_node=2 gloo_tp_pp_probe.py
（2-rank；4-rank 本机不可行——仅 2 GPU，not_covered 见 experiment-draft.yaml）

口径声明（正文必引）：
  gloo 后端只支持 CPU 张量 → TP/PP 通信路径 = GPU 分片 → D2H → gloo
  （CPU 共享内存）→ H2D。这里量到的通信耗时是"行为验证"口径，
  **不是** GPU 间通信性能（性能口径见 Ch5：v100x2 NVLink 实测档 + nominal 档案）。
  选 gloo 而非 RCCL 的原因：本机 2-process RCCL 跨 GPU kernel 互等死锁
  （Ch5 §5.8，10-run 法医，vendor 级）；gloo 全程不依赖 GPU P2P，与死锁解耦。

四阶段（顺序执行，rank0 写 results/gloo_tp_pp_4b.json）：
  ref          rank0 上 HF Qwen3ForCausalLM（bf16）——正确性锚点（真 RoPE/RMSNorm）；
  manual_full  rank0 上手写全模型（P=1，无切分无 AR）——锚定手写基座，
               TP/PP 与它的差即"切分本身"的误差贡献；
  tp           双 rank：列并行 q/k/v + 行并行 o_proj/mlp-down（Megatron 口径），
               GQA 32q/8kv → 每 rank 16q/4kv（组比 4 保持），逐层 2×AR，
               期望每前向 2×L=72 次，D2H/CPU/H2D 三段计时；
  pp           双 rank：层 0–17 / 18–35（权重不切，整层归本 rank），
               rank0→rank1 单跳 send/recv；tied lm_head → rank1 持 embed。
每 rank 记录分片 VRAM（memory_allocated 加载后 / max 峰值）、AR 次数与顺序、
输出 vs 锚点的 cos 与 top-10 重叠。

数值回退纪律：gloo 若不支持 bf16 张量（版本相关），AR/send-recv 回退 float32
（bf16→fp32 精确上转换，不引入数值差），并在 JSON 的 ar_dtype 字段如实登记。
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

MODEL_DIR = common.MODEL_4B
B, S = 2, 512            # 探针负载（experiment-draft.yaml 冻结）
SEED = 42
WARMUP, SAMPLES = 2, 3
GIB = 2 ** 30

# 预测值（experiment-draft.yaml，stored_at 2026-11-16T12:45:00Z，只读对照）
PRED = {
    "tp2": {"ar_calls": 2 * 36, "per_rank_weights_gib": 4.109,
            "allocated_band_gib": [4.1, 5.8],
            "d2h_ms_per_ar": 0.366, "h2d_ms_per_ar": 0.367,
            "gloo_cpu_ar_ms_band": [0.05, 5.0]},
    "pp2": {"collectives": 0, "per_rank_weights_gib": 4.109,
            "allocated_band_gib": [4.1, 5.8]},
    "correctness": {"cos_min": 0.998, "top10_min": 5},
}


def cfg() -> dict:
    c = json.loads(Path(MODEL_DIR, "config.json").read_text())
    return {
        "L": c["num_hidden_layers"], "d": c["hidden_size"],
        "qh": c["num_attention_heads"], "kvh": c["num_key_value_heads"],
        "hd": c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        "inter": c["intermediate_size"], "vocab": c["vocab_size"],
        "tie": c.get("tie_word_embeddings", False),
        "rms_eps": c["rms_norm_eps"], "rope_theta": c["rope_theta"],
    }


def weight_index() -> dict:
    idx = json.loads(Path(MODEL_DIR, "model.safetensors.index.json").read_text())
    return idx["weight_map"]   # 权重名 -> 分片文件名


# ---------------- 手写基座（TP/PP/全模型共用同一实现） ----------------

class ManualModel(torch.nn.Module):
    """Qwen3 手写实现。

    scheme:
      "full"  P=1，全层全权重（manual_full 锚点）
      "tp"    各 rank 跑全层；权重列并行 q/k/v（切 out 维）+
              行并行 o_proj/gate/up（切 out 维）与 down（切 in 维）
      "pp"    各 rank 只跑自己的层段；权重不切（整层归本 rank）
    """

    def __init__(self, c: dict, rank: int, world: int, device: str, scheme: str):
        super().__init__()
        self.c, self.rank, self.world, self.scheme = c, rank, world, scheme
        p = world if scheme == "tp" else 1        # 权重切分粒度
        self.p = p
        wmap = weight_index()
        d, dq, dkv = c["d"], c["qh"] * c["hd"], c["kvh"] * c["hd"]

        def sl(name, dim, size_full):
            r = rank if p > 1 else 0            # PP（p=1）权重不切，整张归本 rank
            i0, i1 = r * size_full // p, (r + 1) * size_full // p
            with safe_open(Path(MODEL_DIR, wmap[name]), framework="pt") as f:
                t = f.get_tensor(name)
            return t.narrow(dim, i0, i1 - i0).to(device=device, dtype=torch.bfloat16)

        def full(name):
            with safe_open(Path(MODEL_DIR, wmap[name]), framework="pt") as f:
                return f.get_tensor(name).to(device=device, dtype=torch.bfloat16)

        # embed：全量（tied lm_head 复用 → TP/PP 下每 rank 各持一份）
        self.embed_w = full("model.embed_tokens.weight")
        per = -(-c["L"] // world)
        if scheme in ("full", "tp"):
            self.l0, self.l1 = 0, c["L"]
        else:
            self.l0, self.l1 = rank * per, min(c["L"], (rank + 1) * per)
        self.layer_w = {}
        for i in range(self.l0, self.l1):
            a, m = f"model.layers.{i}.self_attn.", f"model.layers.{i}.mlp."
            self.layer_w[i] = {
                "q": sl(a + "q_proj.weight", 0, dq),
                "k": sl(a + "k_proj.weight", 0, dkv),
                # Qwen3 特有：q/k 投出后逐 head RMSNorm（hd 维），RoPE 之前；
                # 权重仅 [hd]，各 rank 全量持有（TP 下逐 head 独立，无需通信）
                "q_norm": full(a + "q_norm.weight"),
                "k_norm": full(a + "k_norm.weight"),
                "v": sl(a + "v_proj.weight", 0, dkv),
                "o": sl(a + "o_proj.weight", 1, dq),
                "gate": sl(m + "gate_proj.weight", 0, c["inter"]),
                "up": sl(m + "up_proj.weight", 0, c["inter"]),
                "down": sl(m + "down_proj.weight", 1, c["inter"]),
            }
        self.in_ln = torch.nn.ParameterList(
            [torch.nn.Parameter(full(f"model.layers.{i}.input_layernorm.weight"))
             for i in range(self.l0, self.l1)])
        self.post_ln = torch.nn.ParameterList(
            [torch.nn.Parameter(full(f"model.layers.{i}.post_attention_layernorm.weight"))
             for i in range(self.l0, self.l1)])
        self.run_final = (scheme == "full") or (rank == world - 1)
        # final_norm 各 rank 都加载：TP rank0（不跑 final）也要用它做 top-10 对照
        self.final_norm_w = full("model.norm.weight")

    @staticmethod
    def rms(x, w, eps):
        # 缩放因子 fp32 计算（数值稳定），结果回 x 的 dtype
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
                * w).to(x.dtype)

    @staticmethod
    def rope(q, k, cos, sin):
        """HF 同式：cos/sin 全 hd 维（cat(freqs,freqs)），rotate_half，
        fp32 计算后回 bf16。"""
        def rot(x):
            n = x.shape[-1] // 2
            return torch.cat((-x[..., n:], x[..., :n]), dim=-1)
        qf, kf = q.float(), k.float()
        return (qf * cos + rot(qf) * sin).to(q.dtype), \
               (kf * cos + rot(kf) * sin).to(k.dtype)

    def forward(self, x, ar_fn=None, ar_log=None, cap=None):
        """x: [B,S,d] 全量残差（TP 下各 rank 同一份）。
        ar_fn(t, name) → 行并行部分和归约后的全量张量；P=1/PP 时为 None。
        cap（可选）：非 None 时逐层 append 层后残差（层 i 后、无 final_norm），
        供逐层对账（debug/forensics 用，主流程不传）。
        返回 (hidden, logits)；logits 仅 run_final 的 rank 非 None。
        """
        c = self.c
        p = self.p
        dev = x.device
        B_, S_ = x.shape[:2]
        hd, hq, hkv = c["hd"], c["qh"] // p, c["kvh"] // max(1, p)
        inv = 1.0 / (c["rope_theta"] ** (
            torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
        pos = torch.arange(S_, device=dev, dtype=torch.float32)
        freqs = torch.outer(pos, inv)                       # [S, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)             # [S, hd]（HF 同式）
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]   # [1,1,S,hd]
        mask = torch.triu(torch.ones(S_, S_, device=dev, dtype=torch.bool),
                          diagonal=1)[None, None]                 # [1,1,Sq,Sk]
        ln = 0
        for i in range(self.l0, self.l1):
            w = self.layer_w[i]
            res = x
            h = self.rms(x, self.in_ln[ln], c["rms_eps"])
            # 注意力布局 [B, H, S, hd]（batched matmul 的 head 维在 batch 里）
            q = (h @ w["q"].t()).view(B_, S_, hq, hd).transpose(1, 2)
            k = (h @ w["k"].t()).view(B_, S_, hkv, hd).transpose(1, 2)
            v = (h @ w["v"].t()).view(B_, S_, hkv, hd).transpose(1, 2)
            q = self.rms(q, w["q_norm"], c["rms_eps"])   # 逐 head（hd 维）
            k = self.rms(k, w["k_norm"], c["rms_eps"])
            q, k = self.rope(q, k, cos, sin)
            rep = hq // hkv
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)    # [B,hq,S,S]
            att = torch.softmax(scores.float().masked_fill(mask, float("-inf")),
                                dim=-1).to(q.dtype)
            att = (att @ v).transpose(1, 2).reshape(B_, S_, hq * hd)
            o = att @ w["o"].t()                            # 行并行部分和 [B,S,d]
            if ar_fn is not None:
                o = ar_fn(o, f"L{i}.o_proj")
                ar_log.append(f"L{i}.o_proj")
            x = res + o
            res = x
            h = self.rms(x, self.post_ln[ln], c["rms_eps"])
            g, u = h @ w["gate"].t(), h @ w["up"].t()
            dpart = (F.silu(g) * u) @ w["down"].t()         # 行并行部分和 [B,S,d]
            if ar_fn is not None:
                dpart = ar_fn(dpart, f"L{i}.mlp_down")
                ar_log.append(f"L{i}.mlp_down")
            x = res + dpart
            ln += 1
            if cap is not None:
                cap.append(x.detach())          # 层 i 后、无 final_norm
        logits = None
        if self.run_final:
            x = self.rms(x, self.final_norm_w, c["rms_eps"])
            logits = x @ self.embed_w.t()
        return x, logits


# ---------------- 工具 ----------------

def sync():
    torch.cuda.synchronize()


def median(v):
    s = sorted(v)
    return s[len(s) // 2]


def make_input():
    c = cfg()
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, c["vocab"], (B, S), generator=g)
    chk = hashlib.md5(ids.numpy().tobytes()).hexdigest()[:12]
    return ids, chk


def top10(logits):
    v, idx = torch.topk(logits.float(), 10, dim=-1)
    return idx.cpu(), v.cpu()


def overlap(a, b):
    return (a[..., :, None] == b[..., None, :]).any(-1).float().mean().item()


def cos_flat(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(),
                               dim=0).item()


def probe_gloo_bf16():
    """gloo 是否支持 bf16 张量通信（AR 与 send/recv 同族）。
    返回 (dtype, bf16_ok)；不支持则回退 float32（精确上转换，无数值差）。"""
    try:
        t = torch.zeros(4, dtype=torch.bfloat16)
        dist.all_reduce(t)
        return torch.bfloat16, True
    except Exception:
        return torch.float32, False


def run_ref(ids_gpu):
    """HF 锚点（仅 rank0）。"""
    from transformers import Qwen3ForCausalLM
    t0 = time.perf_counter()
    m = Qwen3ForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = m(input_ids=ids_gpu, output_hidden_states=True)
    sync()
    return {"load_s": round(load_s, 2),
            "mem_gib": round(torch.cuda.memory_allocated() / GIB, 3),
            "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
            # 本 transformers 版约定（实测判定）：hidden_states 共 L+1 条，
            # 末条 = 最后一层后 **且过 final_norm**（与手写实现同口径）
            "hidden": out.hidden_states[-1],
            "t10": top10(out.logits)[0]}


def build_manual(rank, world, dev, scheme):
    t0 = time.perf_counter()
    m = ManualModel(cfg(), rank, world, dev, scheme)
    load_s = time.perf_counter() - t0
    sync()
    return m, load_s, torch.cuda.memory_allocated() / GIB


def make_ar(dev, dtype):
    """D2H → gloo AR → H2D 三段计时（行为口径）。返回 (ar_fn, rec)。"""
    rec = []

    def ar(t, name):
        sync()
        ta = time.perf_counter()
        c = t.detach().to("cpu").to(dtype)
        sync()
        tb = time.perf_counter()
        dist.all_reduce(c)
        tc = time.perf_counter()
        out = c.to(dev, non_blocking=True).to(torch.bfloat16)
        sync()
        td = time.perf_counter()
        rec.append({"op": name,
                    "d2h_ms": round((tb - ta) * 1e3, 4),
                    "cpu_ms": round((tc - tb) * 1e3, 4),
                    "h2d_ms": round((td - tc) * 1e3, 4),
                    "total_ms": round((td - ta) * 1e3, 4)})
        return out

    return ar, rec


def forward_manual(m, ids, dev, ar_fn, ar_log):
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        x = m.embed_w[ids]
        x, logits = m(x, ar_fn=ar_fn, ar_log=ar_log)
    sync()
    return x, logits, (time.perf_counter() - t0) * 1e3


def main():
    assert "RANK" in os.environ, "必须经 torchrun 启动（--nproc_per_node=2）"
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, "本机 2 GPU，仅支持 world=2（4-rank not_covered）"
    # 默认双卡设计（rank r → cuda:r）；CH6_DEV 可覆盖为同卡部署（如实登记，非静默改配置）
    dev = os.environ.get("CH6_DEV", f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("gloo")

    c = cfg()
    ids, chk = make_input()
    ids_dev = ids.to(dev)
    ar_dtype, bf16_ok = probe_gloo_bf16()

    out = {
        "probe": ("gloo_tp_pp_probe.py（行为验证口径：gloo=CPU 共享内存，"
                  "通信耗时非 GPU 间通信性能——性能口径见 Ch5）"),
        "run_at": common.now_iso(),
        "model": {k: c[k] for k in ("L", "d", "qh", "kvh", "hd", "inter",
                                    "vocab", "tie", "rope_theta")} | {"path": MODEL_DIR},
        "workload": {"batch": B, "seq": S, "seed": SEED,
                     "ar_payload_mb": round(B * S * c["d"] * 2 / 1e6, 2),
                     "input_md5_64tok": chk},
        "gpu": {str(r): {"name": torch.cuda.get_device_name(r),
                         "vram_gib": round(
                             torch.cuda.get_device_properties(r).total_memory / GIB, 2)}
                for r in range(world)},
        "ar_dtype": str(ar_dtype).replace("torch.", "") +
                    ("" if bf16_ok else "（bf16 不支持，回退 float32 精确上转换）"),
        "device": dev,
        "prediction_frozen": PRED,
    }
    # 写 JSON 的只有 rank0；同卡偏离（CH6_DEV）由 rank0 登记一次即可
    if os.environ.get("CH6_DEV") and rank == 0:
        out["device_note"] = (f"CH6_DEV={dev}：两 rank 同卡部署（GPU1 被其他用户容器作业"
                              "占用，不抢卡）。gloo 走 CPU 共享内存、每 rank 显存为进程内"
                              "口径，测量语义不变；偏离默认双卡设计，如实登记。")
    dist.barrier()

    # ---- 阶段 1：HF 锚点（rank0） ----
    ref = None
    if rank == 0:
        ref = run_ref(ids.to("cuda:0"))
    dist.barrier()

    # ---- 阶段 2：手写全模型锚点（rank0，P=1） ----
    mf = None
    mf_hidden = None
    if rank == 0:
        m, load_s, mem = build_manual(0, 1, "cuda:0", "full")
        x, logits, wall = forward_manual(m, ids.to("cuda:0"), "cuda:0", None, [])
        mf = {"load_s": round(load_s, 2), "mem_gib": round(mem, 3),
              "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
              "wall_ms": round(wall, 2),
              # 双侧均 post final_norm 同口径（见 run_ref 约定注释）
              "cos_vs_hf": round(cos_flat(x, ref["hidden"]), 6),
              "top10_overlap_vs_hf": round(overlap(top10(logits)[0], ref["t10"]), 4)}
        mf_hidden = x.detach().clone()
        del m, x, logits
        torch.cuda.empty_cache()
    dist.barrier()

    # ---- 阶段 3：TP=2 ----
    m, load_s, mem_load = build_manual(rank, world, dev, "tp")
    torch.cuda.reset_peak_memory_stats()
    ar, rec = make_ar(dev, ar_dtype)
    ar_log, walls = [], []
    for s in range(WARMUP + SAMPLES):
        rec.clear()
        ar_log.clear()
        x, logits, wall = forward_manual(m, ids_dev, dev, ar, ar_log)
        if s >= WARMUP:
            walls.append(wall)
    peak = torch.cuda.max_memory_allocated() / GIB
    seg = {k: median(r[k] for r in rec) for k in ("d2h_ms", "cpu_ms", "h2d_ms", "total_ms")}
    tp = {"load_s": round(load_s, 2), "mem_after_load_gib": round(mem_load, 3),
          "peak_gib": round(peak, 3), "wall_ms_median": round(median(walls), 2),
          "ar_calls": len(ar_log), "ar_sequence": ar_log,
          **{f"ar_{k[:-3]}_median_ms": round(seg[k], 4) for k in seg},  # d2h_ms→ar_d2h_median_ms
          "comm_ms_per_forward": round(sum(seg[k] for k in ("d2h_ms", "cpu_ms", "h2d_ms")), 3)}
    tp["comm_share_pct"] = round(tp["comm_ms_per_forward"] / tp["wall_ms_median"] * 100, 2)
    # rank1 最终 hidden（post final_norm，其 run_final=True）→ rank0（gloo p2p）
    if rank == 1:
        sync()
        dist.send(x.detach().to("cpu").to(ar_dtype), dst=0)
        del x
    else:
        buf = torch.empty(B, S, c["d"], dtype=ar_dtype)
        dist.recv(buf, src=1)
        x1 = buf.to(dev).to(torch.bfloat16)
        x_fin = m.rms(x, m.final_norm_w, c["rms_eps"])   # rank0 补 final_norm
        # rank1 送来的是 post-norm（其 run_final=True）→ 双侧同口径比，公平
        tp["rank_agreement_cos"] = round(cos_flat(x_fin, x1), 6)
        tp["cos_vs_manualfull"] = round(cos_flat(x_fin, mf_hidden), 6)
        tp["top10_overlap_vs_hf"] = round(
            overlap(top10(x_fin @ m.embed_w.t())[0], ref["t10"]), 4)
        del x1, x_fin, x
    del m, logits
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- 阶段 4：PP=2（rank1 先 recv 再跑层；send 含 D2H+gloo，recv 含 gloo+H2D） ----
    m, load_s, mem_load = build_manual(rank, world, dev, "pp")
    torch.cuda.reset_peak_memory_stats()
    walls, send_ts, recv_ts = [], [], []
    x = logits = None
    for s in range(WARMUP + SAMPLES):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            if rank == 0:
                x = m.embed_w[ids_dev]
                x, _ = m(x, ar_fn=None, ar_log=None)
                sync()
                ts = time.perf_counter()
                cpu = x.detach().to("cpu").to(ar_dtype)
                sync()
                tr = time.perf_counter()
                dist.send(cpu, dst=1)
                td = time.perf_counter()
                send_ts.append((td - ts) * 1e3)
            else:
                buf = torch.empty(B, S, c["d"], dtype=ar_dtype)
                ts = time.perf_counter()
                dist.recv(buf, src=0)
                x = buf.to(dev).to(torch.bfloat16)
                sync()
                tr = time.perf_counter()
                recv_ts.append((tr - ts) * 1e3)
                x, logits = m(x, ar_fn=None, ar_log=None)  # 内含 final_norm + lm_head
        sync()
        if s >= WARMUP:
            walls.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / GIB
    pp = {"load_s": round(load_s, 2), "mem_after_load_gib": round(mem_load, 3),
          "peak_gib": round(peak, 3), "wall_ms_median": round(median(walls), 2),
          "collectives": 0,
          "send_ms_median": round(median(send_ts), 3) if send_ts else None,
          "recv_ms_median": round(median(recv_ts), 3) if recv_ts else None}
    # rank1 最终 hidden（post final_norm）→ rank0
    if rank == 1:
        sync()
        dist.send(x.detach().to("cpu").to(ar_dtype), dst=0)
        del x
    else:
        buf = torch.empty(B, S, c["d"], dtype=ar_dtype)
        dist.recv(buf, src=1)
        x1 = buf.to(dev).to(torch.bfloat16)
        pp["rank1_cos_vs_manualfull"] = round(cos_flat(x1, mf_hidden), 6)
        pp["rank1_top10_overlap_vs_hf"] = round(
            overlap(top10(x1 @ m.embed_w.t())[0], ref["t10"]), 4)
        del x1, x
    del m, logits
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- 汇总：rank0 写 JSON ----
    mine = {"rank": rank, "tp": tp, "pp": pp}
    # 本 torch 版 API：gather_list 仅允许在 dst rank 上给出
    if rank == 0:
        gathered = [None] * world
        dist.gather_object(mine, gathered, dst=0)
    else:
        gathered = None
        dist.gather_object(mine, dst=0)
    if rank == 0:
        res = dict(out)
        res["ref_hf"] = {"load_s": ref["load_s"], "mem_gib": ref["mem_gib"],
                         "peak_gib": ref["peak_gib"]}
        res["manual_full_rank0"] = mf
        res["tp"] = gathered[0]["tp"]
        res["pp"] = gathered[0]["pp"]
        res["rank1"] = {"tp": gathered[1]["tp"], "pp": gathered[1]["pp"]}
        exp_seq = [f"L{i}.{n}" for i in range(c["L"]) for n in ("o_proj", "mlp_down")]
        # 正确性链：HF ← manual_full ← {TP, PP}，每一跳都要过（对各自直接参考）
        mf_cos = res["manual_full_rank0"]["cos_vs_hf"]
        mf_t10 = res["manual_full_rank0"]["top10_overlap_vs_hf"]
        cos_tp = res["tp"]["cos_vs_manualfull"]
        cos_pp = res["pp"]["rank1_cos_vs_manualfull"]
        t10_tp = res["tp"]["top10_overlap_vs_hf"]
        t10_pp = res["pp"]["rank1_top10_overlap_vs_hf"]
        res["judgment"] = {
            "tp_ar_count": {"expected": PRED["tp2"]["ar_calls"],
                            "rank0": res["tp"]["ar_calls"],
                            "rank1": res["rank1"]["tp"]["ar_calls"]},
            "tp_ar_order_matches_megatron": res["tp"]["ar_sequence"] == exp_seq,
            "pp_collectives": {"expected": PRED["pp2"]["collectives"],
                               "rank0": res["pp"]["collectives"],
                               "rank1": res["rank1"]["pp"]["collectives"]},
            "shard_vram_vs_pred": {
                "tp_rank0_gib": res["tp"]["mem_after_load_gib"],
                "tp_rank1_gib": res["rank1"]["tp"]["mem_after_load_gib"],
                "pp_rank0_gib": res["pp"]["mem_after_load_gib"],
                "pp_rank1_gib": res["rank1"]["pp"]["mem_after_load_gib"],
                "pred_weights_gib": PRED["tp2"]["per_rank_weights_gib"],
                "pred_allocated_band_gib": PRED["tp2"]["allocated_band_gib"],
            },
            "d2h_h2d_vs_ch5_caliber": {
                "d2h_ms": res["tp"]["ar_d2h_median_ms"], "d2h_pred_ms": PRED["tp2"]["d2h_ms_per_ar"],
                "h2d_ms": res["tp"]["ar_h2d_median_ms"], "h2d_pred_ms": PRED["tp2"]["h2d_ms_per_ar"],
            },
            "correctness": {
                # 链上每一跳对各自直接参考；TP/PP 继承 manual_full，
                # 故 manual_full↔HF 也须过，否则 TP/PP 对 HF 必然偏
                "manualfull_cos_vs_hf": mf_cos,
                "manualfull_top10_vs_hf": mf_t10,
                "tp_cos_vs_manualfull": cos_tp,
                "pp_cos_vs_manualfull": cos_pp,
                "tp_top10_vs_hf": t10_tp,
                "pp_top10_vs_hf": t10_pp,
                "cos_min": PRED["correctness"]["cos_min"],
                "top10_min": PRED["correctness"]["top10_min"],
                "pass": bool(mf_cos >= PRED["correctness"]["cos_min"]
                             and cos_tp >= PRED["correctness"]["cos_min"]
                             and cos_pp >= PRED["correctness"]["cos_min"]
                             and mf_t10 * 10 >= PRED["correctness"]["top10_min"]
                             and t10_tp * 10 >= PRED["correctness"]["top10_min"]
                             and t10_pp * 10 >= PRED["correctness"]["top10_min"]),
            },
        }
        p = common.save_json("gloo_tp_pp_4b.json", res)
        print(f"[rank0] saved {p}", flush=True)
        print(json.dumps(res["judgment"], ensure_ascii=False, indent=1), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
