#!/usr/bin/env python3
"""第 6 章配图生成：4 张中文标签 PNG。

数据来源（全部来自 results/，机器产物，正文数字同源）：
  fig6-1 results/parallel_ledger.json  memory          四模型 × 两卡档 × 三方案（P=2/4）每卡显存水位
  fig6-2 results/parallel_ledger.json  comm_share.tp_* TP 通信占比 vs S（prefill，TP=2/4 × 四拓扑）+ 10% 预算线
  fig6-3 results/parallel_ledger.json  comm_share.pp   PP 气泡比 vs m（P=2/4）+ P=4/m=8 GPipe 稳态甘特
  fig6-4 results/parallel_ledger.json  holdout_32b     32B 留出验证：三档拓扑 prefill/decode 占比 + 判定

配色（dataviz 中性浅色板，light 模式，与第 2/3/4/5 章一致）：
  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a / 红 #e34948；
  文字一律用墨色 token，不写灰色脚注；轴带单位；标题含硬件与口径。

输出（index.md 以相对路径引用）：
  docs/part1-modeling/chapter6/images/fig6-1-memory-water-level.png
  docs/part1-modeling/chapter6/images/fig6-2-tp-comm-share.png
  docs/part1-modeling/chapter6/images/fig6-3-pp-bubble-gantt.png
  docs/part1-modeling/chapter6/images/fig6-4-holdout-verdict.png

运行：
  cd code/part1-modeling/chapter6
  python3 figs/make_figs.py
（无 GPU 依赖，纯画图；需先跑完 parallel_ledger.py。）

三件套验证（本脚本内建）：
  ① 几何 dump：save_fig 前打印每个 ax 的 xlim/ylim 与关键 artist 位置；
  ② 像素采样：render 后在期望数据坐标采样，断言命中精确调色板色；
  ③ md5 确定性：save_fig 打印 md5，重复运行不变。

字体：Noto Sans CJK SC。figs/fonts/*.otf 因版权考虑不入仓（已 .gitignore）。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT_DIR = HERE.parents[3] / "docs" / "part1-modeling" / "chapter6" / "images"

# ---------- 调色板（light，与第 2/3/4/5 章一致） ----------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"

HEX_RGB = {
    "ink": (11, 11, 11), "ink2": (82, 81, 78), "muted": (137, 135, 129),
    "surface": (252, 252, 251), "grid": (225, 224, 217), "baseline": (195, 194, 183),
    "blue": (42, 120, 214), "orange": (235, 104, 52), "aqua": (27, 175, 122),
    "red": (227, 73, 72), "neutral": (216, 215, 208),
}

# 拓扑配色（本章四条：本机 PCIe / nominal NVLink / 实测 NVLink / 跨节点 RoCE）
TOPO_COLOR = {
    "pcie4_x16": BLUE,
    "nvlink300": AQUA,
    "nvlink_measured_v100x2": ORANGE,
    "roce100g": RED,
}
TOPO_CN = {
    "pcie4_x16": "PCIe 4.0 ×16（节点内 25 GB/s）",
    "nvlink300": "NVLink nominal（300 GB/s）",
    "nvlink_measured_v100x2": "NVLink 实测 v100x2（α=24.45µs）",
    "roce100g": "100G RoCEv2（跨节点 11 GB/s）",
}

MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
MODEL_LABEL = {"Qwen3-4B": "4B", "Qwen3-8B": "8B",
               "Qwen3-14B": "14B", "Qwen3-32B": "32B"}
CARDS = {
    "9070xt": ("9070 XT · 16 GB", 13.81),
    "w7900d": ("W7900D · 48 GB", 42.7),
}
# 三方案 P=2 / P=4（TP ≡ PP 显存相同，图中标记为 TP/PP）
SCHEMES = [
    ("dp_p2", "DP（P=2）", ORANGE),
    ("tp_p2", "TP/PP（P=2）", BLUE),
    ("tp_p4", "TP/PP（P=4）", AQUA),
]

BUDGET = 10.0  # comm_budget_pct（测前冻结）

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


def _pixel_check(fig, ax, data_xy, key, label, tol=2):
    """三件套②：在 data_xy 采样像素，断言命中调色板色 HEX_RGB[key]。

    坐标变换自校准：transData 的输出单位因 matplotlib 构建而异
    （本机 build 输出即 fig.dpi 下的 figure 像素，而非 points），
    故用 ax 两角在 transData 与 renderer 像素空间（get_window_extent）
    中的位置解出缩放系数 sx/sy，不预设单位；对 log 轴同样成立。
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    renderer = fig.canvas.get_renderer()
    bb = ax.get_window_extent(renderer)          # renderer 像素空间（= 缓冲区像素）
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    p00 = np.array(ax.transData.transform((xl, yl)))
    p10 = np.array(ax.transData.transform((xr, yl)))
    p01 = np.array(ax.transData.transform((xl, yr)))
    sx = bb.width / (p10[0] - p00[0])
    sy = bb.height / (p01[1] - p00[1])
    d = np.array(ax.transData.transform(data_xy))
    xi = int(round(bb.x0 + (d[0] - p00[0]) * sx))
    yi = int(round(buf.shape[0] - (bb.y0 + (d[1] - p00[1]) * sy)))
    got = tuple(int(v) for v in buf[yi, xi, :3])
    exp = HEX_RGB[key]
    ok = all(abs(g - e) <= tol for g, e in zip(got, exp))
    tag = "OK " if ok else "FAIL"
    print(f"    [{tag}] pixel {label}: got {got} expect {exp} (±{tol})")
    if not ok:
        raise AssertionError(f"pixel sampling failed: {label} got {got} expect {exp}")
    return got


# ---------- fig6-1：每卡显存水位 ----------
def fig6_1_memory(ledger):
    mem = ledger["memory"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    x = np.arange(len(MODELS))
    w = 0.26
    for ax, (ckey, (clabel, usable)) in zip(axes, CARDS.items()):
        for i, (skey, slabel, scol) in enumerate(SCHEMES):
            vals, feas = [], []
            for mk in MODELS:
                v = mem[mk][f"{ckey}_{skey}"]
                vals.append(v["per_rank_gib"]["total"])
                feas.append(v["feasible"])
            xpos = x + (i - 1) * w
            for j, (mk, val, fk) in enumerate(zip(MODELS, vals, feas)):
                ax.bar(xpos[j], val, w,
                       color=scol,
                       edgecolor=RED if not fk else "white",
                       linewidth=1.4 if not fk else 0.5,
                       hatch="//" if not fk else None,
                       label=slabel if j == 0 else None,
                       zorder=3)
                ax.text(xpos[j], val + usable * 0.012, f"{val:.1f}",
                        ha="center", va="bottom", color=INK, fontsize=7.4)
        ax.axhline(usable, color=RED, linewidth=1.2, linestyle="--", zorder=2)
        ax.text(len(MODELS) - 0.5, usable * 0.965, f"可用 {usable} GiB",
                color=RED, fontsize=8.6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
        ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS])
        ax.set_ylim(0, max(mem[m][f"{ckey}_dp_p2"]["per_rank_gib"]["total"] for m in MODELS) * 1.14)
        style_ax(ax, title=f"{clabel}：每卡显存水位（B=8 × S=4096）",
                 xlabel="模型规模", ylabel="每卡显存（GiB）")
    axes[0].legend(frameon=False, fontsize=8.6, labelcolor=INK2,
                   loc="upper left", ncols=1)
    fig.suptitle("三方案每卡显存：DP 权重冗余 ×P 最高，TP ≡ PP 切 1/P；红框/斜纹 = 超可用线（不可行）",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 三件套②：采样可行蓝条内部（8B / TP-PP P=2）—— 先采样后保存（save_fig 会 close 图）
    ax = axes[1]
    _pixel_check(fig, ax, (x[1] + 0 * w, 6.0), "blue", "fig6-1 TP/PP P=2 bar interior")
    save_fig(fig, "fig6-1-memory-water-level.png")


# ---------- fig6-2：TP 通信占比 vs S ----------
S_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def fig6_2_tp_share(ledger):
    cs = ledger["comm_share"]["Qwen3-8B"]
    smin = ledger["judgment"]["Qwen3-8B"]["tp_smin_prefill"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for ax, p in zip(axes, (2, 4)):
        blk = cs[f"tp_p{p}"]["prefill_share_by_s"]
        series = {
            "pcie4_x16": [blk[str(s)] for s in S_GRID],
            "nvlink300": blk["nvlink300_by_s"],
            "nvlink_measured_v100x2": blk["nvlink_measured_v100x2_by_s"],
            "roce100g": blk["roce100g_by_s"],
        }
        for name in TOPO_COLOR:
            ax.loglog(S_GRID, series[name], color=TOPO_COLOR[name],
                      linewidth=1.9, label=TOPO_CN[name], zorder=3)
            sv = smin[f"p{p}_{name}"]
            if sv is not None:
                ax.scatter([sv], [BUDGET], color=TOPO_COLOR[name], s=34,
                           zorder=5, edgecolor=INK, linewidth=0.6)
                ax.annotate(f"S≥{sv}", xy=(sv, BUDGET),
                            xytext=(sv * 1.35, BUDGET + 4.5),
                            fontsize=7.6, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        ax.axhline(BUDGET, color=RED, linewidth=1.1, linestyle="--", zorder=2)
        ax.text(1.05, BUDGET + 1.4, "10% 预算线（comm_budget）",
                color=RED, fontsize=8.4, ha="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(S_GRID[::2], [f"{v}" for v in S_GRID[::2]], rotation=0)
        ax.set_xlim(1, 32768)
        ax.set_ylim(1, 105)   # log 轴下界必须为正（0 会被静默忽略并告警）
        style_ax(ax, title=f"TP={p}：prefill 通信占比 vs 序列长度 S（B=8）",
                 xlabel="序列长度 S（token）", ylabel="通信占比（%）")
    axes[0].legend(frameon=False, fontsize=8.0, labelcolor=INK2, loc="upper right")
    fig.suptitle("Qwen3-8B：TP 每层 2 次 AR，payload=B·S·D·bf16；"
                 "实测 NVLink（α=24.45µs）因延迟地板整体劣于 nominal nvlink300",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 三件套②：采样 TP=2 面板实测 NVLink 的 smin 标记点（4096, 10%）—— 先采样后保存
    ax = axes[0]
    _pixel_check(fig, ax, (4096, BUDGET),
                 "orange", "fig6-2 measured-NVLink smin marker @S=4096 (TP=2)")
    save_fig(fig, "fig6-2-tp-comm-share.png")


# ---------- fig6-3：PP 气泡 + GPipe 甘特 ----------
def fig6_3_pp_bubble(ledger):
    pp = ledger["comm_share"]["Qwen3-8B"]["pp"]
    g = ledger["animation_data"]["pp_gantt"]
    mgrid = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # 左：气泡比 vs m（P=2 / P=4）
    ax = axes[0]
    for p, scol in [(2, BLUE), (4, ORANGE)]:
        vals = [pp[f"p{p}_m{m}"]["bubble_or_transfer_share_pct"] for m in mgrid]
        ax.plot(mgrid, vals, "o-", color=scol, linewidth=1.9, ms=5,
                label=f"P={p}：(P−1)/(m+P−1)")
    ax.axhline(BUDGET, color=RED, linewidth=1.1, linestyle="--", zorder=2)
    ax.text(1.05, BUDGET + 1.4, "10% 预算线", color=RED, fontsize=8.4, ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mgrid, [f"{v}" for v in mgrid])
    ax.set_xlim(0.8, 48)
    ax.set_ylim(0, 78)
    ax.legend(frameon=False, fontsize=8.4, labelcolor=INK2, loc="upper right")
    style_ax(ax, title="PP 气泡比 vs 微批数 m（m=1 为传输项）",
             xlabel="微批数 m（log）", ylabel="气泡 / 传输占比（%）")

    # 右：P=4、m=8 GPipe 稳态甘特
    ax = axes[1]
    P, m, ts = g["p"], g["m"], g["t_stage_ms"]
    makespan = (m + P - 1) * ts
    for s in range(P):  # stage s
        yrow = P - 1 - s  # stage 0 在最上
        # 气泡（前导空闲）：0 → s*ts
        if s > 0:
            ax.add_patch(plt.Rectangle((0, yrow), s * ts, 1,
                                       facecolor="#fbe9e7", edgecolor="none",
                                       hatch="///", zorder=2))
        # 计算块：microbatch b → [(b+s)*ts, (b+s+1)*ts]
        for b in range(m):
            xc = (b + s) * ts
            ax.add_patch(plt.Rectangle((xc, yrow), ts, 1,
                                       facecolor=(BLUE if b % 2 == 0 else AQUA),
                                       edgecolor="white", linewidth=0.5, zorder=3))
    ax.set_xlim(0, makespan)
    ax.set_ylim(0, P)
    ax.set_yticks([P - 1 - s + 0.5 for s in range(P)],
                  [f"stage {s}" for s in range(P)], fontsize=8.4)
    ax.set_xticks(np.arange(0, makespan + 1, 2 * ts))
    ax.set_xticklabels([f"{t/ts:.0f}·t_stage" for t in np.arange(0, makespan + 1, 2 * ts)],
                       fontsize=7.6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(BASELINE)
    style_ax(ax, title=f"GPipe 稳态：P={P}，m={m}，气泡 {g['bubble_pct']}%",
             xlabel="时间（t_stage=147.639 ms）", ylabel="")
    # 气泡区标注
    ax.text(0.5 * ts, P - 1 + 0.05, "气泡（前导空闲）", color=RED, fontsize=8.0,
            ha="left", va="bottom")
    fig.suptitle("PP：气泡比与带宽无关，只由 P 与 m 决定；P=4/m=8 时气泡占 27.27%",
                 color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 三件套②：左面板采样 P=2 曲线在 m=8 的点 —— 先采样后保存
    axL = axes[0]
    _pixel_check(fig, axL, (8, pp["p2_m8"]["bubble_or_transfer_share_pct"]),
                 "blue", "fig6-3 P=2 bubble curve @m=8")
    save_fig(fig, "fig6-3-pp-bubble-gantt.png")


# ---------- fig6-4：32B 留出验证 ----------
def fig6_4_holdout(ledger):
    h = ledger["holdout_32b"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    tcs = {"innode_pcie4_x16": ("节点内 PCIe", BLUE),
           "innode_nvlink300": ("节点内 NVLink", AQUA),
           "cross_roce100g": ("跨节点 RoCE", RED)}
    for ax, (ph, ptitle, pverdict) in zip(axes, [
        ("prefill", "prefill（B=8 × S=4096）", "判定 must_innode（只有节点内可行）"),
        ("decode", "decode（小 payload）", "判定 infeasible_even_innode（连节点内都 >10%）"),
    ]):
        x = np.arange(2)
        w = 0.26
        for i, (tkey, (tlabel, tcol)) in enumerate(tcs.items()):
            vals = [h[f"tp_p{p}"][ph]["shares_pct"][tkey] for p in (2, 4)]
            xpos = x + (i - 1) * w
            for j, (p, val) in enumerate(zip((2, 4), vals)):
                ax.bar(xpos[j], val, w, color=tcol,
                       edgecolor="white", linewidth=0.5,
                       label=tlabel if j == 0 else None, zorder=3)
                ax.text(xpos[j], val + 1.5, f"{val:.1f}", ha="center", va="bottom",
                        color=INK, fontsize=7.6)
        ax.axhline(BUDGET, color=INK, linewidth=1.1, linestyle="--", zorder=2)
        ax.text(1.55, BUDGET + 1.5, "10% 预算线", color=INK, fontsize=8.4, ha="right")
        ax.set_xticks(x, ["TP=2", "TP=4"])
        ax.set_ylim(0, max(h[f"tp_p4"][ph]["shares_pct"]["cross_roce100g"],
                           h[f"tp_p2"][ph]["shares_pct"]["cross_roce100g"]) * 1.18)
        style_ax(ax, title=f"32B 留出验证 · {ptitle}",
                 xlabel="TP 切分（P）", ylabel="通信占比（%）")
        ax.legend(frameon=False, fontsize=8.2, labelcolor=INK2, loc="upper left")
        ax.text(0.02, 0.03, pverdict, transform=ax.transAxes, color=RED,
                fontsize=8.6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=RED, lw=0.8))
    fig.suptitle("留出验证（32B，未参与调参）：prefill 必须节点内、decode 连节点内都不可行 →"
                 "「TP 组必须限制在同一节点内」换规模后依然成立",
                 color=INK, fontsize=12.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # 三件套②：采样 prefill 面板节点内 NVLink 条（TP=4，居中 x=1）内部 —— 先采样后保存
    # （TP=2 条高仅 1.465%，左下角 verdict 框红色边框压底边，x=0 处安全带宽 <2px，不可靠）
    ax = axes[0]
    _pixel_check(fig, ax, (1, h["tp_p4"]["prefill"]["shares_pct"]["innode_nvlink300"] * 0.5),
                 "aqua", "fig6-4 prefill NVLink bar (TP=4) interior")
    save_fig(fig, "fig6-4-holdout-verdict.png")


# ---------- 保存 + 字节级验证 ----------
def save_fig(fig, name):
    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    # 三件套①：几何 dump
    print(f"  [geom] {name}")
    for ax in fig.axes:
        if ax.get_visible():
            print(f"     ax xlim={ax.get_xlim()} ylim={ax.get_ylim()}")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"  {out.name}  md5={md5}")
    return out, md5


if __name__ == "__main__":
    setup_font()
    ledger = _load("parallel_ledger.json")
    print("figs:")
    fig6_1_memory(ledger)
    fig6_2_tp_share(ledger)
    fig6_3_pp_bubble(ledger)
    fig6_4_holdout(ledger)
    print("done")
