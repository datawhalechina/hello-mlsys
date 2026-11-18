"""第 6 章共享工具：模型档案（在盘 + 公开 config）、硬件、拓扑、计算常数。

与 Ch2/Ch3/Ch4/Ch5 common.py 对齐（环境快照 schema 跨章一致）。本章维度：
  并行切分 = 显存账本（每卡水位）× 通信占比（每 step 通信时间 / 总时间）
三口径（对账表用）：
  公式口径：parallel_ledger.py（公开 config 纸面账本 × Ch5 拓扑档案 × Ch2 计算常数）
  引擎口径：Ch5 实测（v100x2 NVLink fit：α_eff=24.45 µs/跳，BW_eff=102.24 GB/s）
            —— 本章以 nvlink_measured_v100x2 拓扑档复用，本机 PCIe 2-hop 仍 not_covered
  行为口径：gloo_tp_pp_probe.py（gloo CPU 共享内存路径，2-rank 真机）
            —— 显式声明：行为验证 ≠ 性能验证（Feishu CH6 原文）

模型档案纪律：
  - Qwen3-4B 在盘（/mnt/public/models），gloo 探针用真权重；
  - Qwen3-8B/14B/32B 不在盘，config 取自 HuggingFace Qwen 官方仓库
    （raw/main/config.json，2026-11-16 逐一 curl 核实），纸面账本用，
    来源逐条标注，不做实测。
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

MODEL_4B = "/mnt/public/models/Qwen3-4B"

RESULTS = Path(__file__).parent / "results"

# --- Ch2 校准常数（来源：code/part1-modeling/chapter2/results/cal_9070xt.json）---
PEAK_TFLOPS = 195.0      # bf16 名义峰值（9070 XT，Ch0/Ch1 冻结）
ETA = 0.6243             # 计算侧系数（Ch2 §2.3 拟合）
BYTES_ELEM = 2           # bf16 激活/权重

# --- 硬件档案 ---
VRAM_9070_GIB = 15.9     # 9070 XT 15.9 GiB（Ch3 口径：torch 实测 17095983104 B）
VRAM_7900X_GIB = 24.0    # 7900 XTX 24 GiB（本机辅助卡，仅 gloo 行为探针用）
VRAM_W7900D_GIB = 48     # W7900D 48 GB（参数待确认，TODO 嘉航；容量占位）
UTIL = 0.9               # 可用显存比例（Ch3 口径）
OVERHEAD_GIB = 0.5       # 每 rank 运行时预留（driver+runtime+workspace，Ch3 口径）

HARDWARE = {
    "9070xt": {"vram_gib": VRAM_9070_GIB, "peak_tflops": PEAK_TFLOPS, "eta": ETA,
               "confirmed": True,
               "note": "主线卡，Ch2 计算常数已校准"},
    "7900xtx": {"vram_gib": VRAM_7900X_GIB, "peak_tflops": None, "eta": None,
                "confirmed": True,
                "note": "本机辅助卡（异构 gfx1100）；未做峰值/η 校准，本章不用于性能账本"},
    "w7900d": {"vram_gib": VRAM_W7900D_GIB, "peak_tflops": None, "eta": None,
               "confirmed": False,
               "note": "参数待确认（TODO 嘉航），仅用容量占位"},
}

# --- 模型档案 ---
# 在盘（gloo 探针真权重）
MODEL_4B_CFG = None     # 运行时从 config.json 读（model_cfg 函数）

# 公开 config（HuggingFace Qwen 官方仓库，2026-11-16 逐一 curl 核实 raw/main/config.json；
# sliding_window 均为 None → 全量注意力口径，滑动窗口忽略并在正文注明）
PUBLISHED_MODELS = {
    "Qwen3-8B": {
        "n_layers": 36, "hidden": 4096, "n_q_heads": 32, "n_kv_heads": 8,
        "head_dim": 128, "inter": 12288, "vocab": 151936, "tie_weights": False,
        "source": "huggingface.co/Qwen/Qwen3-8B config.json（raw/main，2026-11-16 核实）",
    },
    "Qwen3-14B": {
        "n_layers": 40, "hidden": 5120, "n_q_heads": 40, "n_kv_heads": 8,
        "head_dim": 128, "inter": 17408, "vocab": 151936, "tie_weights": False,
        "source": "huggingface.co/Qwen/Qwen3-14B config.json（raw/main，2026-11-16 核实）",
    },
    "Qwen3-32B": {
        "n_layers": 64, "hidden": 5120, "n_q_heads": 64, "n_kv_heads": 8,
        "head_dim": 128, "inter": 25600, "vocab": 151936, "tie_weights": False,
        "source": "huggingface.co/Qwen/Qwen3-32B config.json（raw/main，2026-11-16 核实）",
    },
}


def model_cfg_4b() -> dict:
    """Qwen3-4B 在盘 config（gloo 探针 + 账本同源）。"""
    c = json.loads(Path(MODEL_4B, "config.json").read_text())
    return {
        "name": "Qwen3-4B",
        "n_layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "n_q_heads": c["num_attention_heads"],
        "n_kv_heads": c.get("num_key_value_heads", c["num_attention_heads"]),
        "head_dim": c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        "inter": c["intermediate_size"],
        "vocab": c["vocab_size"],
        "tie_weights": c.get("tie_word_embeddings", False),
        "source": "/mnt/public/models/Qwen3-4B/config.json（在盘）",
    }


# --- 拓扑参数档案（Ch5 同档 + Ch5 实测档）---
# alpha_us：单跳启动/延迟开销；bw_gbps：单向有效带宽。
# 前三档 = Ch5 §5.1 nominal 档（测前冻结口径一致）；第四档 = Ch5 §5.9 v100x2 实测拟合。
TOPOLOGY = {
    "pcie4_x16": {"bw_gbps": 25.0, "alpha_us": 2.0,
                  "note": "Ch5 nominal：PCIe 4.0 ×16，25 GB/s 单向有效，α=2 µs/跳"},
    "nvlink300": {"bw_gbps": 300.0, "alpha_us": 1.0,
                  "note": "Ch5 nominal：NVLink A100 口径 300 GB/s 单向，α=1 µs/跳"},
    "nvlink_measured_v100x2": {"bw_gbps": 102.24, "alpha_us": 24.45,
                               "note": "Ch5 §5.9 实测：v100x2 6×V100 NV6 bond（GPU 4/5），"
                                       "12 calib + 2 holdout 拟合（R²=0.99965），"
                                       "α 含引擎启动/同步开销（引擎口径）"},
    "roce100g": {"bw_gbps": 11.0, "alpha_us": 2.0,
                 "note": "Ch5 nominal：100 Gbps RoCEv2 跨节点，11 GB/s，α=2 µs/跳"},
}


def ring_ar_s(p: int, data_b: float, alpha_us: float, bw_gbps: float) -> float:
    """T_AR = 2(P−1)α + 2·(P−1)/P · D/BW（秒；Ch5 同式）。"""
    return (2 * (p - 1) * alpha_us * 1e-6
            + 2 * (p - 1) / p * data_b / (bw_gbps * 1e9))


def allgather_s(p: int, data_b: float, alpha_us: float, bw_gbps: float) -> float:
    """T_AG = (P−1)α + (P−1)/P · D/BW（秒）。"""
    return ((p - 1) * alpha_us * 1e-6
            + (p - 1) / p * data_b / (bw_gbps * 1e9))


def p2p_s(data_b: float, alpha_us: float, bw_gbps: float) -> float:
    """点对点单跳：T = α + D/BW（秒；PP 激活传输用）。"""
    return alpha_us * 1e-6 + data_b / (bw_gbps * 1e9)


def fwd_flops(c: dict, n_tok: int, seq_len: int) -> float:
    """整模型一次前向 FLOPs（bf16，含 lm_head；与 Ch5 32·N·D² 口径同族——
    当 d_q=d_kv=d、inter=4d 时退化为逐层 32·N·D²）。

    逐层线性层（每 token 批 N）：
      q/k/v/o: 2·N·(d·d_q + d·d_kv + d·d_kv + d_q·d)
      gate/up/down: 6·N·d·inter
    注意力打分（全量注意力，忽略 Qwen3 可选滑窗）：
      QK^T + 加权 V：4·N·S·d_q
    lm_head: 2·N·V·d（未 tie 时；tie 时 embed 复用，前向仍有该 GEMM，计入）
    """
    d = c["hidden"]
    d_q = c["n_q_heads"] * c["head_dim"]
    d_kv = c["n_kv_heads"] * c["head_dim"]
    per_layer = (2 * n_tok * (d * d_q + d * d_kv + d * d_kv + d_q * d)
                 + 6 * n_tok * d * c["inter"]
                 + 4 * n_tok * seq_len * d_q)
    total = c["n_layers"] * per_layer + 2 * n_tok * c["vocab"] * d
    return total


def weight_params(c: dict) -> dict:
    """参数量逐组件（Ch3 weight_components 同口径）。"""
    d = c["hidden"]
    d_q = c["n_q_heads"] * c["head_dim"]
    d_kv = c["n_kv_heads"] * c["head_dim"]
    per_layer = (d * d_q + d * d_kv + d * d_kv + d_q * d
                 + 3 * d * c["inter"]
                 + d_q + d_kv + 2 * d)
    embed = c["vocab"] * d
    out = {"embed": embed, "per_layer": per_layer,
           "layers": c["n_layers"] * per_layer,
           "final_norm": d}
    if not c["tie_weights"]:
        out["lm_head"] = c["vocab"] * d
    out["total"] = sum(v for k, v in out.items() if k != "per_layer")
    return out


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def env_block() -> dict:
    """环境与硬件快照（宿主侧 rocm-smi；与 Ch2/Ch3/Ch4/Ch5 同 schema，双卡扩展）。"""
    out = subprocess.run(["rocm-smi", "--showproductname", "--showmeminfo", "vram",
                          "--showdriverversion"],
                         capture_output=True, text=True, timeout=30).stdout
    gpus = []
    pending = None
    for line in out.splitlines():
        if "Card Series:" in line:
            pending = line.split("Card Series:")[1].strip()
        if "GFX Version:" in line and pending:
            gpus.append({"name": pending,
                         "arch": line.split("GFX Version:")[1].strip()})
            pending = None
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d)\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)", line)
        if m and int(m.group(1)) < len(gpus):
            gpus[int(m.group(1))]["vram_gib"] = round(int(m.group(2)) / 2 ** 30, 2)
    m = re.search(r"Driver version\s*:\s*(\d+\.\d+\.\d+)", out)
    driver = m.group(1) if m else "unknown"
    topo = subprocess.run(["rocm-smi", "--showtopo"],
                          capture_output=True, text=True, timeout=30).stdout
    link = None
    for line in topo.splitlines():
        if "PCIE" in line or "XGMI" in line:
            link = line.strip()
    return {
        "timestamp_utc": now_iso(),
        "python": __import__("sys").version.split()[0],
        "os": __import__("platform").platform(),
        "gpus": gpus,
        "driver": driver,
        "p2p_link_line": link,
    }
