#!/usr/bin/env python3
"""Chapter 8 figures from machine artifacts.

Run: cd code/part1-modeling/chapter8 && python3 figs/make_figs.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent
RESULTS = CHAPTER / "results"
FRAMEWORK = RESULTS / "framework"
OUT = CHAPTER.parents[2] / "docs" / "part1-modeling" / "chapter8" / "images"
FONT_DIR = CHAPTER.parent / "chapter7" / "figs" / "fonts"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
NEUTRAL = "#d8d7d0"
FRAMEWORK_COLORS = {"vllm": BLUE, "llamacpp": ORANGE}
FRAMEWORK_LABELS = {"vllm": "vLLM", "llamacpp": "llama.cpp"}


def setup_font() -> None:
    for name in ("NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Bold.otf"):
        path = FONT_DIR / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
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
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
        spine.set_linewidth(1.0)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK2)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_points() -> dict[str, dict[int, dict]]:
    analysis = load("framework/analysis.json")
    holdout = load("framework/holdout-analysis.json")
    points: dict[str, dict[int, dict]] = {"vllm": {}, "llamacpp": {}}
    for framework in ("vllm", "llamacpp"):
        for point in analysis["points"][framework]:
            points[framework][point["concurrency"]] = {
                "ttft": point["ttft_ms"]["p99"],
                "tpot": point["tpot_corrected_ms"]["p99"],
                "e2el": point["e2el_ms"]["p99"],
                "raw_tok_s": point["output_tokens_per_s"],
                "goodput": point["slo_goodput_requests_per_s"],
                "verdict": point["slo_verdict"],
                "source": "main",
            }
    for row in holdout["rows"]:
        actual = row["actual"]
        points[row["framework"]][row["concurrency"]] = {
            "ttft": actual["ttft_p99_ms"],
            "tpot": actual["tpot_p99_ms"],
            "e2el": actual["e2el_p99_ms"],
            "raw_tok_s": actual["output_tokens_per_s"],
            "goodput": actual["slo_goodput_requests_per_s"],
            "verdict": actual["slo_verdict"],
            "source": "holdout",
        }
    return points


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": None})
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"saved {path}  md5={md5}")
    for index, ax in enumerate(fig.axes):
        print(f"  ax{index}: xlim={ax.get_xlim()} ylim={ax.get_ylim()} "
              f"patches={len(ax.patches)} texts={len(ax.texts)}")
    plt.close(fig)


def annotate(ax, x, y, text, color=INK2, dy=6) -> None:
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, dy),
                ha="center", va="bottom", fontsize=8.5, color=color)


def fig81(points: dict[str, dict[int, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axhline(800, color=RED, lw=1.3, ls="--")
    ax.text(1.03, 830, "SLO: TTFT P99 = 800 ms", color=RED, fontsize=9, ha="left")
    for framework in ("vllm", "llamacpp"):
        ordered = sorted(points[framework].items())
        xs = [item[0] for item in ordered]
        ys = [item[1]["ttft"] for item in ordered]
        color = FRAMEWORK_COLORS[framework]
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=6.5,
                label=FRAMEWORK_LABELS[framework])
        for x, y, (_, point) in zip(xs, ys, ordered):
            if point["verdict"] == "FAIL":
                ax.plot(x, y, marker="X", ms=10, color=RED, markeredgecolor="white",
                        markeredgewidth=0.8, zorder=5)
                annotate(ax, x, y, "FAIL", color=RED, dy=7)
            elif point["source"] == "holdout":
                annotate(ax, x, y, f"{y:.0f}", color=color, dy=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_xticklabels(["1", "8", "16", "32", "48", "64"])
    ax.set_ylim(0, 4200)
    style(ax, "同一负载下 TTFT P99 随并发上升：vLLM 在 C48 越线，llama.cpp 在 C16 已越线",
          "闭环并发 C", "TTFT P99 (ms)")
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, -0.02, "数据：C1/8/32/64 主扫描 + C16/48 留出测量；Qwen3-4B，ISL=256，OSL=128，200 prompts/点。",
             color=MUTED, fontsize=8.5)
    save(fig, "fig8-1-ttft-p99.png")


def fig82(points: dict[str, dict[int, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for framework in ("vllm", "llamacpp"):
        ordered = sorted(points[framework].items())
        xs = [item[0] for item in ordered]
        raw = [item[1]["raw_tok_s"] for item in ordered]
        good = [item[1]["goodput"] for item in ordered]
        color = FRAMEWORK_COLORS[framework]
        axes[0].plot(xs, raw, color=color, lw=2.0, marker="o", ms=6,
                     label=FRAMEWORK_LABELS[framework])
        axes[1].plot(xs, good, color=color, lw=2.0, marker="o", ms=6,
                     label=FRAMEWORK_LABELS[framework])
        for x, y, (_, point) in zip(xs, raw, ordered):
            if point["verdict"] == "FAIL":
                axes[0].plot(x, y, marker="X", ms=9, color=RED, markeredgecolor="white",
                             markeredgewidth=0.8, zorder=5)
        for x, y, (_, point) in zip(xs, good, ordered):
            if point["verdict"] == "FAIL":
                axes[1].plot(x, y, marker="X", ms=9, color=RED, markeredgecolor="white",
                             markeredgewidth=0.8, zorder=5)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 16, 32, 48, 64])
        ax.set_xticklabels(["1", "8", "16", "32", "48", "64"])
        ax.set_xlabel("闭环并发 C", color=INK2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("输出 token / s", color=INK2)
    axes[1].set_ylabel("SLO goodput (合格请求 / s)", color=INK2)
    style(axes[0], "裸吞吐一直涨，但不是有效容量", None, None)
    style(axes[1], "SLO goodput 在越线点塌陷或停滞", None, None)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    fig.text(0.01, -0.02, "X 标记表示该点整体 SLO FAIL；goodput 只统计同时满足 TTFT/TPOT/E2E 的请求。",
             color=MUTED, fontsize=8.5)
    save(fig, "fig8-2-throughput-goodput.png")


def fig83() -> None:
    holdout = load("framework/holdout-analysis.json")
    rows = holdout["rows"]
    labels = [f"{'V' if row['framework']=='vllm' else 'L'} C{row['concurrency']}" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    metrics = (("ttft", "TTFT P99 (ms)", axes[0], 800.0),
               ("tpot", "TPOT P99 (ms)", axes[1], 100.0))
    positions = list(range(len(rows)))
    width = 0.36
    for metric, ylabel, ax, threshold in metrics:
        predicted = [row["predicted"][f"{metric}_p99_ms"] for row in rows]
        actual = [row["actual"][f"{metric}_p99_ms"] for row in rows]
        ax.bar([p - width / 2 for p in positions], predicted, width,
               color=NEUTRAL, edgecolor="white", label="预测（测前冻结）")
        ax.bar([p + width / 2 for p in positions], actual, width,
               color=AQUA, edgecolor="white", label="实测")
        ax.axhline(threshold, color=RED, lw=1.2, ls="--")
        ax.text(-0.45, threshold, f"SLO {threshold:g}", color=RED,
                fontsize=8.5, ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        top = max(max(predicted), max(actual))
        ax.set_ylim(0, top * 1.18)
        for index, row in enumerate(rows):
            error = row["error_pct"][metric]
            ax.text(index, max(predicted[index], actual[index]) * 1.02,
                    f"{error:+.0f}%", ha="center", va="bottom", fontsize=8.5,
                    color=INK2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        style(ax, "留出点 TTFT" if metric == "ttft" else "留出点 TPOT",
              None, ylabel)
        ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, -0.02,
             "留出点 C16/48 在预测写盘后才测量；误差百分比 = (实测−预测)/预测。四个 SLO 判定全部命中。",
             color=MUTED, fontsize=8.5)
    save(fig, "fig8-3-holdout-prediction.png")


def fig84() -> None:
    profile = load("framework/profile-summary.json")
    frameworks = ("vllm", "llama")
    labels = ["vLLM", "llama.cpp"]
    colors = [BLUE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    dispatch = [profile[framework]["kernels_per_output_token"] for framework in frameworks]
    busy = [profile[framework]["gpu_busy_pct"] for framework in frameworks]
    axes[0].bar(labels, dispatch, color=colors, edgecolor="white", width=0.55)
    for index, value in enumerate(dispatch):
        axes[0].text(index, value * 1.02, f"{value:.1f}", ha="center", va="bottom",
                     color=INK, fontweight="bold")
    axes[0].set_ylim(0, max(dispatch) * 1.22)
    style(axes[0], "kernel dispatch 数 / 输出 token", None, "dispatches / token")
    axes[1].bar(labels, busy, color=colors, edgecolor="white", width=0.55)
    for index, value in enumerate(busy):
        axes[1].text(index, value + 1.5, f"{value:.1f}%", ha="center", va="bottom",
                     color=INK, fontweight="bold")
    axes[1].set_ylim(0, 105)
    style(axes[1], "GPU busy（kernel 时间 / trace 跨度）", None, "%")
    fig.text(0.01, -0.02,
             "C=8，20 请求，ISL=256/OSL=128；vLLM 269,892 dispatches/8.50 s，llama.cpp 497,906/13.95 s。profile 有扰动，只做相对归因。",
             color=MUTED, fontsize=8.5)
    save(fig, "fig8-4-kernel-profile.png")


def memory_rows() -> list[dict]:
    plan = load("screen-qwen4b-7900xtx-v2/plan.json")
    weights = 4_022_468_096 * 2
    rows = []
    for candidate in plan["candidates"]:
        name = candidate["candidate_id"]
        strategy = candidate["experiment"]["strategy"]
        memory = strategy["memory"]
        other = (memory["graph_bytes"] + memory["workspace_base_bytes"]
                 + memory["runtime_bytes"] + memory["communication_buffer_bytes"]
                 + memory["other_bytes"] + memory["safety_margin_bytes"])
        rows.append({
            "name": name,
            "weights": weights,
            "kv": memory["kv_pool_bytes"],
            "other": other,
            "total": weights + memory["kv_pool_bytes"] + other,
        })
    return rows


def fig85() -> None:
    rows = memory_rows()
    labels = ["vLLM\n4K ctx", "vLLM\n128K ctx", "llama.cpp\n512 ctx", "llama.cpp\n131K ctx"]
    order = ["vllm-bf16-4096-c48", "vllm-bf16-131072-c1-memory-fail",
             "llamacpp-f16-512-c48", "llamacpp-f16-131072-c1-memory-fail"]
    by_name = {row["name"]: row for row in rows}
    selected = [by_name[name] for name in order]
    gib = 1024 ** 3
    weights = [row["weights"] / gib for row in selected]
    kv = [row["kv"] / gib for row in selected]
    other = [row["other"] / gib for row in selected]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    positions = list(range(len(selected)))
    ax.bar(positions, weights, color=BLUE, edgecolor="white", width=0.58, label="权重")
    ax.bar(positions, kv, bottom=weights, color=ORANGE, edgecolor="white", width=0.58,
           label="KV 池")
    ax.bar(positions, other, bottom=[w + k for w, k in zip(weights, kv)],
           color=NEUTRAL, edgecolor="white", width=0.58, label="graph / workspace / runtime")
    limit = 25753026560 / gib
    ax.axhline(limit, color=RED, lw=1.4, ls="--",
               label=f"物理显存 {limit:.2f} GiB")
    for index, row in enumerate(selected):
        total = row["total"] / gib
        ax.text(index, total + 0.35, f"{total:.2f} GiB", ha="center", va="bottom",
                color=INK, fontweight="bold")
        if total > limit:
            ax.text(index, total * 0.55, "STATIC\nOOM", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 34)
    style(ax, "显存账本先判死：131K 上下文在任何延迟比较之前就装不下", None, "GiB")
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.01, -0.02,
             "权重按 4,022,468,096 参数 × 2 B；KV = 147,456 B/token；物理容量 25,753,026,560 B。",
             color=MUTED, fontsize=8.5)
    save(fig, "fig8-5-memory-first.png")


def main() -> None:
    setup_font()
    points = load_points()
    fig81(points)
    fig82(points)
    fig83()
    fig84()
    fig85()


if __name__ == "__main__":
    main()
