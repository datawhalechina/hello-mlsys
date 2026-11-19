#!/usr/bin/env python3
"""Chapter 7 figures, all from JSON machine artifacts.

Data sources:
  results/analysis.json       real sweep, DES holdout, open-loop, pressure
  results/service_model.json  calibration coefficients

Outputs (index.md references these relative paths):
  docs/part1-modeling/chapter7/images/fig7-1-timeline.png
  docs/part1-modeling/chapter7/images/fig7-2-load-models.png
  docs/part1-modeling/chapter7/images/fig7-3-serving-path.png
  docs/part1-modeling/chapter7/images/fig7-4-knee.png
  docs/part1-modeling/chapter7/images/fig7-5-open-loop.png
  docs/part1-modeling/chapter7/images/fig7-6-pressure.png
  docs/part1-modeling/chapter7/images/fig7-7-p99-growth.png

Run: cd code/part1-modeling/chapter7 && python3 figs/make_figs.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE.parents[3] / "docs" / "part1-modeling" / "chapter7" / "images"
FONT_DIR = HERE / "fonts"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"
HEX_RGB = {"blue": (42, 120, 214), "orange": (235, 104, 52),
           "aqua": (27, 175, 122), "red": (227, 73, 72), "neutral": (216, 215, 208)}


def setup_font() -> None:
    for name in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"):
        p = FONT_DIR / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style(ax, title=None, xlabel=None, ylabel=None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(BASELINE)
        s.set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK2)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold",
                     loc="left", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": None})
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"saved {path}  md5={md5}")
    for i, ax in enumerate(fig.axes):
        print(f"  ax{i}: xlim={ax.get_xlim()} ylim={ax.get_ylim()} "
              f"patches={len(ax.patches)} texts={len(ax.texts)}")
    plt.close(fig)


def pixel_check(fig, ax, xy, key, label, tol=3) -> None:
    """Sample a solid bar at data coordinates and assert the palette colour."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    renderer = fig.canvas.get_renderer()
    bb = ax.get_window_extent(renderer)
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    p00 = np.array(ax.transData.transform((xl, yl)))
    p10 = np.array(ax.transData.transform((xr, yl)))
    p01 = np.array(ax.transData.transform((xl, yr)))
    sx = bb.width / (p10[0] - p00[0])
    sy = bb.height / (p01[1] - p00[1])
    d = np.array(ax.transData.transform(xy))
    xi = int(round(bb.x0 + (d[0] - p00[0]) * sx))
    yi = int(round(buf.shape[0] - (bb.y0 + (d[1] - p00[1]) * sy)))
    got = tuple(int(v) for v in buf[yi, xi, :3])
    exp = HEX_RGB[key]
    ok = all(abs(g - e) <= tol for g, e in zip(got, exp))
    print(f"  [{'OK' if ok else 'FAIL'}] pixel {label}: got {got} expect {exp}")
    if not ok:
        raise AssertionError(f"pixel check failed: {label}")


def box(ax, x, y, w, h, text, color, textcolor="white", fs=10) -> None:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.2,rounding_size=0.6",
                                        facecolor=color, edgecolor="white", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fs, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.4) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


def fig71() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.set_xlim(-2, 104)
    ax.set_ylim(-0.6, 5.0)
    ax.axis("off")
    ax.set_title("一次流式请求的时间线：TTFT、ITL、TPOT、E2E 不是同一个口径",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    y, h = 2.0, 1.1
    segs = [(0, 8, "网络 / 客户端", NEUTRAL, INK),
            (8, 25, "排队 + 预处理", ORANGE, "white"),
            (25, 40, "Prefill", BLUE, "white"),
            (40, 92, "逐 token Decode（ITL）", AQUA, "white"),
            (92, 100, "收尾", NEUTRAL, INK)]
    for x1, x2, label, color, tc in segs:
        ax.add_patch(patches.Rectangle((x1, y), x2 - x1, h, facecolor=color,
                                       edgecolor="white", linewidth=1.5))
        ax.text((x1 + x2) / 2, y + h / 2, label, ha="center", va="center",
                color=tc, fontsize=10, fontweight="bold")
    for t in (0, 8, 25, 40, 92, 100):
        ax.plot([t, t], [y - 0.18, y], color=INK, lw=1)
    for t, lab in [(0, "t0"), (8, "t1"), (25, "t2"), (40, "t3"), (92, "tn")]:
        ax.text(t, y - 0.42, lab, ha="center", va="top", color=INK2, fontsize=10)
    for i in range(6):
        t = 48 + i * 7
        ax.plot([t, t], [y, y + h], color="white", lw=1.2, alpha=.9)
    ax.annotate("", xy=(40, 4.2), xytext=(0, 4.2),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.8))
    ax.text(20, 4.42, "TTFT = t3 − t0", ha="center", color=BLUE,
            fontsize=10.5, fontweight="bold")
    ax.annotate("", xy=(100, 1.35), xytext=(0, 1.35),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.4))
    ax.text(50, 1.05, "E2E = tn − t0", ha="center", color=INK2, fontsize=10.5)
    ax.text(66, 0.48, "TPOT = (E2E − TTFT)/(N−1)", color=AQUA,
            fontsize=10.5, fontweight="bold", ha="center")
    ax.text(66, 0.08, "P99 TPOT 先在请求内取平均，再对请求求分位；P99 ITL 在全部区间上求。",
            color=INK2, fontsize=9.2, ha="center")
    save(fig, "fig7-1-timeline.png")


def fig72() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("三种“并发”：Open-loop λ、Closed-loop C、Engine Batch B",
                 color=INK, fontsize=12, fontweight="bold", loc="left")
    # open loop
    ax.text(2, 88, "① Open-loop：外部按 λ 独立到达", color=INK, fontsize=11, fontweight="bold")
    for i, x in enumerate((12, 24, 36, 48, 60, 72)):
        arrow(ax, x, 80, x, 70, ORANGE)
    ax.text(66, 82, "到达间隔 ~ Poisson", color=ORANGE, fontsize=9.5)
    box(ax, 8, 58, 68, 9, "网关 / 客户端队列（不设 C 上限）", NEUTRAL, INK)
    # closed loop
    ax.text(2, 50, "② Closed-loop：C 个客户端，完成一个再发下一个", color=INK,
            fontsize=11, fontweight="bold")
    for i, x in enumerate((12, 24, 36, 48)):
        ax.add_patch(patches.Circle((x, 38), 2.2, facecolor=BLUE, edgecolor="white"))
        arrow(ax, x + 1.8, 40.2, 80, 42, BLUE, 1.1)
    box(ax, 72, 46, 22, 8, "C 个在途", BLUE)
    ax.annotate("", xy=(12, 35.6), xytext=(70, 35.6),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(41, 30.5, "响应完成 → 同一客户端再提交", color=BLUE, fontsize=9.5, ha="center")
    # engine batch
    ax.text(2, 22, "③ Engine：waiting queue → scheduler → running batch", color=INK,
            fontsize=11, fontweight="bold")
    box(ax, 5, 7, 22, 9, "Waiting queue", ORANGE)
    box(ax, 32, 7, 22, 9, "Scheduler", NEUTRAL, INK)
    box(ax, 59, 7, 18, 9, "Running B", AQUA)
    box(ax, 82, 7, 13, 9, "KV pool", BLUE)
    arrow(ax, 27, 11.5, 32, 11.5)
    arrow(ax, 54, 11.5, 59, 11.5)
    arrow(ax, 77, 11.5, 82, 11.5)
    ax.text(50, 2.2, "B ≤ max_num_seqs；B 是每一步真正上 GPU 的序列数，不等于 C 或在线用户数。",
            color=INK2, fontsize=9.4, ha="center")
    save(fig, "fig7-2-load-models.png")


def fig73() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("LLM Serving 的三段等待与四个资源事件", color=INK,
                 fontsize=12, fontweight="bold", loc="left")
    y = 46
    boxes = [
        (2, "Client\n/ gateway", NEUTRAL, INK),
        (20, "Waiting\nqueue", ORANGE, "white"),
        (38, "Scheduler\n(chunked prefill)", NEUTRAL, INK),
        (57, "Running\nbatch", AQUA, "white"),
        (75, "KV blocks\n24,464 tokens", BLUE, "white"),
    ]
    for x, label, color, tc in boxes:
        box(ax, x, y, 16, 18, label, color, tc, 10)
    for x in (18, 36, 54, 73):
        arrow(ax, x, y + 9, x + 2, y + 9)
    notes = [
        (28, 78, "排队：TTFT 的第一来源", ORANGE),
        (46, 28, "Prefill：输入越长越重", BLUE),
        (66, 78, "Decode：batch 越大，单步越慢", AQUA),
        (82, 28, "KV 不足 → 推迟/抢占/重算", RED),
    ]
    for x, yy, text, color in notes:
        ax.text(x, yy, text, color=color, fontsize=10, ha="center", fontweight="bold")
    ax.plot([20, 36], [y + 20, y + 20], color=GRID, lw=1)
    ax.text(50, 5, "本章把 P99 陡升拆成：排队等待、prefill、decode、KV 抢占/重算、算力饱和；"
            "第 8 章再下钻到 kernel 与框架 Profile。", color=INK2, fontsize=9.4, ha="center")
    save(fig, "fig7-3-serving-path.png")


def fig74(a: dict) -> None:
    sweep = a["closed_sweep"]
    cs = [r["concurrency"] for r in sweep]
    real_ttft = [r["ttft_p99_ms"] for r in sweep]
    real_tpot = [r["tpot_p99_ms"] for r in sweep]
    real_tp = [r["request_throughput"] for r in sweep]
    real_gp = [r["request_goodput"] for r in sweep]
    des = [r.get("des_v2") for r in sweep]
    des_cs = [c for c, d in zip(cs, des) if d]
    des_ttft = [d["ttft_p99_ms"] for d in des if d]
    des_gp = [d["goodput_rps"] for d in des if d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.plot(des_cs, des_ttft, color=BLUE, lw=2, marker="o", ms=4,
             label="DES v2（校准 C=1/2/4/8）")
    ax1.scatter(cs, real_ttft, color=ORANGE, s=26, zorder=4, label="vLLM 实测")
    ax1.axhline(800, color=RED, ls="--", lw=1.2)
    ax1.text(cs[-1], 870, "TTFT P99 SLO 800 ms", color=RED, ha="right", fontsize=9)
    ax1.axvline(48, color=RED, ls=":", lw=1.2)
    ax1.text(48 * 1.05, 48, "C=48\n首次失守", color=RED, fontsize=9.2, va="bottom")
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    style(ax1, "(a) P99 TTFT：预测拐点与实测拐点对齐",
          "闭环节点 C（请求数）", "P99 TTFT（ms，log）")
    ax1.legend(frameon=False, loc="upper left", fontsize=9)

    x = np.arange(len(cs))
    w = 0.38
    b1 = ax2.bar(x - w / 2, real_tp, w, color=BLUE, label="request throughput")
    b2 = ax2.bar(x + w / 2, real_gp, w, color=AQUA, label="SLO goodput")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in cs], rotation=45, fontsize=8.5)
    ax2.axvline(x[cs.index(48)] - 0.5, color=RED, ls=":", lw=1.2)
    ax2.text(x[cs.index(48)] - 0.35, max(real_tp) * 0.93, "C=48", color=RED, fontsize=9)
    style(ax2, "(b) 吞吐还在涨，goodput 已先崩", "闭环节点 C", "req/s")
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("RX 9070 XT · Qwen3-4B bf16 · ISL 256 / OSL 128 · 200 prompts/点",
                 color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # sample the last blue bar near its top
    pixel_check(fig, ax2, (x[-1] - w / 2, real_tp[-1] * 0.55), "blue", "throughput bar")
    save(fig, "fig7-4-knee.png")


def fig75(a: dict) -> None:
    real = [r for r in a["open_loop_real"] if "rate" in r]
    des = a.get("open_loop_des", [])
    rates = [r["rate"] for r in real]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax1.plot(rates, rates, color=BASELINE, ls="--", lw=1.2, label="offered = achieved")
    ax1.plot(rates, [r["request_throughput"] for r in real], color=BLUE, marker="o",
             lw=2, label="实测 achieved")
    ax1.plot(rates, [r["request_goodput"] for r in real], color=AQUA, marker="s",
             lw=2, label="实测 goodput")
    if des:
        ax1.plot([d["offered_rps"] for d in des], [d["achieved_rps"] for d in des],
                 color=ORANGE, marker="^", lw=1.5, label="DES achieved")
    ax1.set_xlim(0, max(rates) + 2)
    ax1.set_ylim(0, max(rates) + 2)
    style(ax1, "(a) 开环到达：吞吐饱和后，增益不再变成产出",
          "offered λ（req/s）", "achieved / goodput（req/s）")
    ax1.legend(frameon=False, fontsize=8.8, loc="upper left")

    ax2.plot(rates, [r["queue_share_p99_est_pct"] for r in real], color=ORANGE,
             marker="o", lw=2, label="实测 P99 排队占比（估计）")
    if des:
        ax2.plot([d["offered_rps"] for d in des], [d["queue_share_pct"] for d in des],
                 color=BLUE, marker="^", lw=1.5, label="DES 排队占比")
    ax2.axhline(50, color=RED, ls="--", lw=1.1)
    ax2.text(rates[-1], 52, "一半时间在排队", color=RED, ha="right", fontsize=9)
    ax2.set_ylim(0, 100)
    style(ax2, "(b) 饱和后，排队时间挤占实际执行时间",
          "offered λ（req/s）", "排队时间 / E2E（%）")
    ax2.legend(frameon=False, fontsize=8.8, loc="upper left")
    fig.suptitle("开环 λ 扫描：Qwen3-4B · 9070 XT · 120 prompts/点",
                 color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig7-5-open-loop.png")


def fig76(a: dict) -> None:
    rows = a["pressure"]
    labels = [f"C={r['concurrency']}\nOSL={r['output_len']}" for r in rows]
    pre = [r["preemptions"] for r in rows]
    kv = [r["kv_cache_peak_pct"] for r in rows]
    gp = [r["request_goodput"] for r in rows]
    ttft = [r["ttft_p99_ms"] for r in rows]
    x = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax1.bar(x, pre, 0.5, color=BLUE, label="preemptions")
    ax1.set_ylabel("preemptions", color=INK2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    style(ax1, "(a) KV 压力：抢占次数随水位跳升", None, "preemptions")
    ax1b = ax1.twinx()
    ax1b.plot(x, kv, color=RED, marker="o", lw=2, label="KV peak %")
    ax1b.set_ylim(0, 110)
    ax1b.set_ylabel("KV peak（%）", color=RED)
    ax1b.tick_params(colors=RED)
    for i, v in enumerate(kv):
        ax1b.text(i, v + 3, f"{v:.1f}%", ha="center", color=RED, fontsize=9)

    ax2.bar(x - 0.18, gp, 0.36, color=AQUA, label="goodput")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("goodput（req/s）", color=INK2)
    style(ax2, "(b) 重算把 goodput 拉回去", None, "goodput（req/s）")
    ax2b = ax2.twinx()
    ax2b.plot(x, ttft, color=ORANGE, marker="o", lw=2, label="P99 TTFT")
    ax2b.set_ylabel("P99 TTFT（ms）", color=ORANGE)
    ax2b.tick_params(colors=ORANGE)
    for i, v in enumerate(ttft):
        ax2b.text(i, v + 180, f"{v:.0f}", ha="center", color=ORANGE, fontsize=9)
    fig.suptitle("OSL=512 压力实验：KV 100% 时，抢占/重算与尾延迟同时爆发",
                 color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig7-6-pressure.png")


def fig77(a: dict) -> None:
    sweep = a["closed_sweep"]
    cs = [r["concurrency"] for r in sweep]
    real_ttft = [r["ttft_p99_ms"] for r in sweep]
    real_tpot = [r["tpot_p99_ms"] for r in sweep]
    des = [r.get("des_v2") for r in sweep]
    des_cs = [c for c, d in zip(cs, des) if d]
    des_ttft = [d["ttft_p99_ms"] for d in des if d]
    des_tpot = [d["tpot_p99_ms"] for d in des if d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    for ax, real, pred, slo, ylabel, title in (
        (ax1, real_ttft, des_ttft, 800, "P99 TTFT（ms，log）",
         "(a) TTFT P99：排队主导向"),
        (ax2, real_tpot, des_tpot, 100, "P99 TPOT（ms，log）",
         "(b) TPOT P99：decode 调度与干扰"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.plot(des_cs, pred, color=BLUE, lw=2, marker="o", ms=4,
                label="DES v2")
        ax.scatter(cs, real, color=ORANGE, s=26, zorder=4, label="vLLM 实测")
        ax.axhline(slo, color=RED, ls="--", lw=1.2)
        ax.text(cs[-1], slo * 1.06, f"SLO {slo} ms", color=RED,
                ha="right", fontsize=9)
        ax.axvline(48, color=RED, ls=":", lw=1.2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        style(ax, title, "闭环节点 C（请求数）", ylabel)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("P99 随并发增长：TTFT 由排队主导，TPOT 由 decode 调度主导",
                 color=INK2, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig7-7-p99-growth.png")


def main() -> None:
    setup_font()
    a = load("analysis.json")
    fig71()
    fig72()
    fig73()
    fig74(a)
    fig75(a)
    fig76(a)
    fig77(a)


if __name__ == "__main__":
    main()
