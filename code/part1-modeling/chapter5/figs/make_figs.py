#!/usr/bin/env python3
"""第 5 章配图生成：4 张中文标签 PNG。

数据来源（全部来自 results/，机器产物，正文数字同源）：
  fig1  results/comm_ledger.json   curves        四档拓扑 Ring-AR 曲线 + 拐点 D*（P=2/4）
  fig2  results/comm_ledger.json   tp_layer_share  TP=2/4 单层通信占比（Qwen3-1.7B）
  fig3  results/comm_ledger.json   nvlink_gain    NVLink 300GB/s 收益区间（P=2）
  fig4  results/pcie_probe_*.json + comm_ledger.json
        三口径对账：A PCIe 1-hop 实测 / B 双卡 RCCL not_covered 根因示意 /
        C α–β 参数对账（公式×设备×引擎-not_covered）

配色（dataviz 中性浅色板，light 模式，与第 2/3/4 章一致）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948；
  文字一律用墨色 token，不写灰色脚注；轴带单位；标题含硬件与口径。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter5/images/fig1-ring-ar-curves.png
  docs/part1-modeling/chapter5/images/fig2-tp-comm-share.png
  docs/part1-modeling/chapter5/images/fig3-nvlink-gain.png
  docs/part1-modeling/chapter5/images/fig4-reconciliation.png

运行：
  cd code/part1-modeling/chapter5
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 comm_ledger.py 与三个探针。）

字体：Noto Sans CJK SC。figs/fonts/*.otf 因版权考虑不入仓（已 .gitignore）。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_DIR = HERE.parents[3] / "docs" / "part1-modeling" / "chapter5" / "images"

# ---------- 调色板（light，与第 2/3/4 章一致） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"

TOPO_COLOR = {
    "pcie4_x16": BLUE,      # 本机档
    "xgmi_ring": ORANGE,
    "roce100g": RED,
    "nvlink300": AQUA,
}
TOPO_CN = {
    "pcie4_x16": "PCIe 4.0 ×16（25 GB/s）",
    "xgmi_ring": "xGMI 环网（64 GB/s）",
    "roce100g": "100G RoCEv2（11 GB/s）",
    "nvlink300": "NVLink（300 GB/s）",
}

# ---------- 中文字体 ----------
FONT_DIR = HERE / "fonts"


def setup_font():
    for p in (FONT_DIR / "NotoSansCJKsc-Regular.otf",
              FONT_DIR / "NotoSansCJKsc-Bold.otf"):
        font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(BASELINE)
        s.set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=10)


def _load(name):
    return json.loads((RESULTS / name).read_text())


def fig1_ring_ar_curves(ledger):
    """四档拓扑 Ring-AR 时间曲线（log-log）+ 拐点 D*，左 P=2 / 右 P=4。"""
    curves = ledger["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, p in zip(axes, (2, 4)):
        for name in TOPO_COLOR:
            key = f"{name}_P{p}"
            d = curves[key]
            ks = [pt["msg_kb"] for pt in d["points"]]
            ts = [pt["t_ar_ms"] for pt in d["points"]]
            ax.loglog(ks, ts, color=TOPO_COLOR[name], linewidth=1.8,
                      label=TOPO_CN[name])
            dx = d["crossover_kb"]
            if 1 <= dx <= 1e6:
                tx = 2 * (p - 1) * d["alpha_us"] * 1e-3   # 拐点处 t ≈ 2×α 项
                ax.axvline(dx, color=MUTED, linewidth=0.8, linestyle=":")
                ax.annotate(f"D*={g(dx)}", xy=(dx, tx), xytext=(dx * 1.15, tx * 1.35),
                            fontsize=8.2, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        style_ax(ax, title=f"P={p}：Ring-AllReduce 时间 vs 消息体量",
                 xlabel="消息体量 Data（KB）", ylabel="T_AR（ms）")
        ax.set_xlim(0.7, 2e6)
    axes[0].legend(frameon=False, fontsize=8.6, labelcolor=INK2, loc="upper left")
    fig.suptitle("nominal 拓扑档案 × Ring 公式（T = 2(P−1)α + 2·(P−1)/P·Data/BW），"
                 "账本 ch5-ledger", color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_fig(fig, "fig1-ring-ar-curves.png")


def g(kb):
    return f"{kb/1024:.0f}MB" if kb >= 1024 else f"{kb:.0f}KB"


def fig2_tp_share(ledger):
    """TP=2/4 单层 Transformer 通信占比（Qwen3-1.7B），100% 平衡线。"""
    t = ledger["tp_layer_share"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, p in zip(axes, (2, 4)):
        rows = [r for r in t["rows"] if r["tp"] == p]
        toks = [r["tokens"] for r in rows]
        w = 0.19
        x = np.arange(len(toks))
        for i, name in enumerate(TOPO_COLOR):
            vals = [r[name]["comm_share_pct"] for r in rows]
            bars = ax.bar(x + (i - 1.5) * w, vals, w,
                          color=TOPO_COLOR[name], label=TOPO_CN[name],
                          edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}",
                        ha="center", va="bottom", color=INK, fontsize=7.6)
        ax.axhline(100, color=RED, linewidth=1.0, linestyle="--")
        ax.text(len(toks) - 0.45, 105, "100%：通信=计算（盈亏线）",
                color=RED, fontsize=8.8, ha="right")
        ax.set_xticks(x, [f"{v}" for v in toks])
        ax.set_ylim(0, max(max(r[name]["comm_share_pct"] for r in rows
                               for name in TOPO_COLOR) * 1.12, 115))
        style_ax(ax, title=f"TP={p}：单层 fwd 两次 All-Reduce 占计算时间比",
                 xlabel="批内 token 数 B·S", ylabel="通信占比（%）")
    axes[0].legend(frameon=False, fontsize=8.2, labelcolor=INK2, loc="upper left")
    fig.suptitle(f"{t['model']}（hidden={t['hidden']}）：Megatron 口径每层 2 次 AR，"
                 "payload=B·S·D·bf16；计算=32·N·D²/(P·195TFLOPS·0.6243)",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "fig2-tp-comm-share.png")


def fig3_nvlink_gain(ledger):
    """NVLink 300GB/s 相对其余三档的加速比（P=2，Ring AR）+ 收益区间。"""
    nv = ledger["nvlink_gain"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for name, d in nv["per_topology"].items():
        pts = d["points"]
        ks = [pt["msg_kb"] for pt in pts]
        sp = [pt["speedup"] for pt in pts]
        ax.loglog(ks, sp, color=TOPO_COLOR[name], linewidth=1.8,
                  label=f"{TOPO_CN[name]}：最大 {d['max_speedup']}×")
        ax.scatter([ks[0]], [sp[0]], color=TOPO_COLOR[name], s=18, zorder=5)
        # RoCE 曲线在 1KB 附近平在 17×，标签放曲线上方避免压线；其余放下方
        if name == "roce100g":
            xytext = (ks[0] * 2.2, sp[0] * 1.25)
        else:
            xytext = (ks[0] * 1.4, sp[0] * 0.62)
        ax.annotate(f"{sp[0]}× @1KB", xy=(ks[0], sp[0]), xytext=xytext,
                    fontsize=8.2, color=INK2)
    style_ax(ax, title="NVLink（300 GB/s, α=1µs）相对三档的 P=2 Ring-AR 加速比",
             xlabel="消息体量 Data（KB）", ylabel="加速比（×）")
    ax.set_xlim(0.7, 2e6)
    ax.legend(frameon=False, fontsize=8.6, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "fig3-nvlink-gain.png")


def _schematic_box(ax, x0, y0, w, h, text, fc, ec):
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=True, facecolor=fc,
                               edgecolor=ec, linewidth=1.2, zorder=3))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            color=INK, fontsize=8.2, fontweight="bold", zorder=4)


def fig4_reconciliation():
    """三口径对账：A PCIe 1-hop 实测带宽；B 双卡 RCCL not_covered 根因示意
    （isAllCudaP2p 误判 → SHM/direct/direct → GPU↔GPU DMA 永不完成）；
    C α–β 参数对账（公式×设备，引擎口径 not_covered）。

    引擎口径（双卡 RCCL 实测）在本机不可得（平台层 P2P 无法建立：两卡无 ReBAR，
    仅 256MB legacy aperture，iommu=pt 已启用仍不充分；RDNA3 无 cuMem，
    首个 all_reduce 永不完成，见 §5.8 / results/rccl_diag*.log 五次 run），
    故 B 画根因示意、C 只画公式×设备两口径实测条 + 引擎 not_covered 标记，
    不画拟合曲线。
    """
    pcie = _load("pcie_probe_7900xtx.json")
    # 设备口径两参数（同源 pcie_probe_7900xtx.json）：
    #   α_device = 64KB H2D 单发 t（µs，设备侧启动+传输）；BW_device = 1GB H2D 稳态
    dev_alpha_us = pcie["rows"][0]["h2d_ms"] * 1e3            # 0.049 ms → 49 µs（iommu=pt 复测版 JSON）
    dev_bw_gbps = pcie["rows"][-1]["h2d_gbps"]                # 1GB → 14.3 GB/s
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.35, 1]})

    # A：PCIe 1-hop H2D/D2H（7900 XTX；主卡被他人任务占用，见 experiment-draft not_covered）
    ax = axes[0]
    rows = [r for r in pcie["rows"] if r["msg_kb"] >= 1024]
    ks = [r["msg_kb"] for r in rows]
    ax.loglog(ks, [r["h2d_gbps"] for r in rows], "o-", color=BLUE, lw=1.6,
              ms=4, label="H2D pinned")
    ax.loglog(ks, [r["d2h_gbps"] for r in rows], "s-", color=ORANGE, lw=1.6,
              ms=4, label="D2H pinned")
    ax.axhspan(20, 25, color=NEUTRAL, alpha=0.55, zorder=0)
    ax.text(1.4e3, 22.5, "工程带 20–25 GB/s（预测）", color=INK2, fontsize=8.4)
    for r in rows:
        ax.text(r["msg_kb"], r["h2d_gbps"] * 1.06, f"{r['h2d_gbps']:.1f}",
                fontsize=7.4, color=INK, ha="center")
    style_ax(ax, title="A · PCIe 1-hop：pinned 传输实测（7900 XTX）",
             xlabel="传输体量（KB）", ylabel="带宽（GB/s）")
    ax.set_ylim(12, 32)
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK2, loc="upper left")

    # B：双卡 RCCL 引擎口径 not_covered —— 根因拓扑示意（无 rccl_probe.json 数据）
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    _schematic_box(ax, 0.30, 0.80, 0.40, 0.15, "CPU 根复合体\n单 NUMA（3900X）",
                   "#efe9dd", INK2)
    _schematic_box(ax, 0.05, 0.46, 0.40, 0.20, "GPU 0\nRX 7900 XTX\n(gfx1100)",
                   SURFACE, INK2)
    _schematic_box(ax, 0.55, 0.46, 0.40, 0.20, "GPU 1\nRX 9070 XT\n(gfx1201)",
                   SURFACE, INK2)
    # host 中转 2-hop 路径（可行，H2D/D2H 14 GB/s）
    ax.plot([0.25, 0.40], [0.66, 0.80], color=BLUE, lw=1.6, zorder=2)
    ax.plot([0.75, 0.60], [0.66, 0.80], color=BLUE, lw=1.6, zorder=2)
    ax.text(0.5, 0.745, "2-hop：经 CPU 根复合体（无 xGMI）", ha="center",
            va="center", color=BLUE, fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=BASELINE, lw=0.7))
    # GPU↔GPU 直连 P2P（不可用）：红线段 + 手绘 X（✗ 字形不在 CJK 字体内）
    ax.plot([0.45, 0.55], [0.56, 0.56], color=RED, lw=2.4, zorder=2)
    ax.plot([0.472, 0.528], [0.532, 0.588], color=RED, lw=3.2, zorder=5, solid_capstyle="round")
    ax.plot([0.472, 0.528], [0.588, 0.532], color=RED, lw=3.2, zorder=5, solid_capstyle="round")
    # 根因文字块
    ax.text(0.5, 0.335, "RCCL 拓扑误判 isAllCudaP2p=1，\n"
                        "HSA 实为 “Could not enable P2P” ×8",
            ha="center", va="top", color=INK, fontsize=8.0, fontweight="bold")
    ax.text(0.5, 0.245, "→ 环落 SHM/direct/direct + GDR 1，\n"
                        "GPU↔GPU DMA 永不完成；NCCL_P2P_DISABLE=1\n"
                        "后 isAllDirectP2p 1→0 但通道不变，首个 1KB AR 90s 超时",
            ha="center", va="top", color=INK2, fontsize=7.5)
    ax.text(0.5, 0.105, "平台层：两卡无 ReBAR（仅 256MB aperture），iommu=pt 已启用仍不足，\n"
                        "gfx1100 无 cuMem → 5 次 run 一致失败（§5.8）",
            ha="center", va="center", color=RED, fontsize=7.8, fontweight="bold")
    ax.set_title("B · 引擎口径（双卡 RCCL）：not_covered", color=INK,
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)

    # C：α–β 参数对账（公式×设备两口径实测条 + 引擎 not_covered 标记）
    ax = axes[2]
    cats = ["α 启动项（µs，对数）", "有效带宽（GB/s）"]
    nominal_a, device_a = 4.0, dev_alpha_us   # nominal 2 跳协议 4 µs；设备=64KB H2D 单发 t
    nominal_b, device_b = 25.0, dev_bw_gbps   # nominal 链路档案；设备=1GB H2D 稳态
    y = np.arange(2)
    ax.barh(y + 0.19, [nominal_a, nominal_b], 0.34, color=NEUTRAL,
            edgecolor="white", label="公式口径（nominal 档案）")
    ax.barh(y - 0.19, [device_a, device_b], 0.34, color=BLUE,
            edgecolor="white", label="设备口径（pinned 实测）")
    for i, (nv, dv) in enumerate([(nominal_a, device_a), (nominal_b, device_b)]):
        ax.text(nv * 1.10, i + 0.19, f"{nv:g}", va="center", color=INK2, fontsize=8.4)
        ax.text(dv * 1.10, i - 0.19, f"{dv:g}", va="center", color=INK,
                fontsize=8.4, fontweight="bold")
    # 引擎口径 not_covered 标记（两行右侧）
    for i in range(2):
        ax.text(280, i, "× 引擎\nnot_covered", va="center", ha="center",
                color=RED, fontsize=7.4, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fbe9e7", ec=RED, lw=0.8))
    ax.set_xscale("log")
    ax.set_yticks(y, cats)
    ax.set_xlim(1, 500)
    style_ax(ax, title="C · α–β 参数对账（P=2，PCIe 2-hop）")
    ax.legend(frameon=False, fontsize=8.0, labelcolor=INK2, loc="lower right")

    fig.suptitle("三口径对账：公式（nominal 档案）× 引擎（RCCL，not_covered）× 设备（pinned 传输）"
                 "｜P2P=无（host 中转；双卡直连不可用，§5.8）",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "fig4-reconciliation.png")


# ---------- 保存 + 字节级验证 ----------
GEOM = {}


def save_fig(fig, name):
    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"  {out}  md5={md5}")
    return out, md5


if __name__ == "__main__":
    setup_font()
    ledger = _load("comm_ledger.json")
    print("figs:")
    fig1_ring_ar_curves(ledger)
    fig2_tp_share(ledger)
    fig3_nvlink_gain(ledger)
    fig4_reconciliation()
    print("done")
