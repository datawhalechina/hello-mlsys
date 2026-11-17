"""第 5 章共享工具：环境快照、拓扑参数档案、名义常数（跨章同 schema）。

与 Ch0 §0.3 协议、Ch2/Ch3/Ch4 common.py 对齐。本章维度：通信代价 ——
  集合通信时间 = 启动项 α + 带宽项 β × Data（Ring-AllReduce 一阶模型）
  T_AR(P, D) = 2(P−1)α + 2·(P−1)/P · D/BW
三个口径（对账表用）：
  公式口径：comm_ledger（nominal 拓扑档案 × Ring 公式，纸面预测）
  引擎口径：RCCL 实测曲线拟合（双卡 PCIe 2-hop）/ gloo 行为记录
  设备口径：torch pinned H2D/D2H 实测（PCIe 路径物理边界）+ rocm-smi 拓扑读数
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

MODEL_17 = "/mnt/public/models/Qwen3-1.7B"

RESULTS = Path(__file__).parent / "results"

# --- Ch2 校准常数（来源：code/part1-modeling/chapter2/results/cal_9070xt.json）---
PEAK_TFLOPS = 195.0      # bf16 名义峰值（Ch0/Ch1 冻结，9070 XT）
ETA = 0.6243             # 计算侧系数（Ch2 §2.3 拟合）
BYTES_ELEM = 2           # bf16 激活

# --- nominal 拓扑参数档案（测前冻结；来源见正文 §5.1 表注）---
# alpha_us：单跳启动/延迟开销；bw_gbps：单向有效带宽。
# pcie4_x16 按 Feishu 要求取单跳；本机双卡实测为 2-hop（PCIe 经 CPU 根复合体）。
TOPOLGY = {
    "pcie4_x16":  {"bw_gbps": 25.0,  "alpha_us": 2.0,
                   "note": "PCIe 4.0 x16：31.5 GB/s 单向 raw（16 GT/s×16 lane，128b/130b），"
                           "取 ~79% 有效；α 取 PCIe doorbell+传输启动典型 2 µs/跳"},
    "xgmi_ring":  {"bw_gbps": 64.0,  "alpha_us": 1.0,
                   "note": "AMD xGMI 环网（MI300 类公开参数：8 链 × 64 GB/s 双向/链），"
                           "α 取 1 µs/跳（片上网络级）"},
    "roce100g":   {"bw_gbps": 11.0,  "alpha_us": 2.0,
                   "note": "100 Gbps RoCEv2 跨节点：12.5 GB/s raw，取 ~88% 有效；"
                           "α 含 RDMA 发送+确认+协议栈，取 2 µs/跳"},
    "nvlink300":  {"bw_gbps": 300.0, "alpha_us": 1.0,
                   "note": "NVLink（Feishu 指定 300 GB/s，对应 A100 单向口径）；"
                           "α 取 1 µs/跳"},
}

# 消息体量网格（Feishu：Data ∈ [1 KB, 1 GB]，对数，21 点 = 2^0 .. 2^20 KB）
MSG_GRID_KB = [2 ** k for k in range(0, 21)]   # 1KB .. 1GB, 21 点


def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def model_cfg(path: str = MODEL_17) -> dict:
    """从 HF config.json 读结构参数（TP 通信占比算例用；与 Ch3 同口径）。"""
    c = json.loads(Path(path, "config.json").read_text())
    return {
        "name": Path(path).name,
        "path": path,
        "n_layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "n_q_heads": c["num_attention_heads"],
        "n_kv_heads": c.get("num_key_value_heads", c["num_attention_heads"]),
        "head_dim": c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
        "inter": c["intermediate_size"],
        "vocab": c["vocab_size"],
        "tie_weights": c.get("tie_word_embeddings", False),
    }


def env_block() -> dict:
    """环境与硬件快照（宿主侧 rocm-smi；与 Ch2/Ch3/Ch4 同 schema，双卡扩展）。"""
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
