#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path


CATEGORY_PATTERNS = [
    ("gemm", ("cijk", "gemm", "matmul", "hipblas", "tensile")),
    ("attention", ("attn", "flash", "fmha", "attention")),
    ("softmax", ("softmax",)),
    ("norm", ("rmsnorm", "layernorm", "norm_kernel", "groupnorm")),
    ("rope", ("rope", "rotary")),
    ("reduce", ("reduce",)),
    ("argmax", ("argmax", "topk")),
    ("copy_or_cat", ("catarray", "copy", "concat")),
    ("elementwise", ("elementwise", "binary", "unary", "neg_kernel", "silu", "gelu", "add_kernel", "mul_kernel")),
]


def categorize(name: str) -> str:
    lower = name.lower()
    for category, needles in CATEGORY_PATTERNS:
        if any(needle in lower for needle in needles):
            return category
    return "other"


def summarize_window(records, symbols, label, start, end):
    selected = [
        r for r in records
        if start <= r["start_timestamp"] <= end
    ]
    by_name = collections.defaultdict(lambda: [0, 0.0])
    by_category = collections.defaultdict(lambda: [0, 0.0])
    for record in selected:
        name = symbols.get(record["dispatch_info"]["kernel_id"], "unknown")
        duration_ms = (record["end_timestamp"] - record["start_timestamp"]) / 1e6
        by_name[name][0] += 1
        by_name[name][1] += duration_ms
        category = categorize(name)
        by_category[category][0] += 1
        by_category[category][1] += duration_ms

    kernel_total = sum(value[1] for value in by_name.values())
    wall_ms = (end - start) / 1e6
    top = []
    for name, (count, total) in sorted(by_name.items(), key=lambda item: item[1][1], reverse=True)[:20]:
        top.append({
            "kernel": name,
            "count": count,
            "total_ms": round(total, 6),
            "avg_ms": round(total / count, 6),
            "pct_of_kernel_time": round(100.0 * total / kernel_total, 3) if kernel_total else None,
        })
    categories = []
    for category, (count, total) in sorted(by_category.items(), key=lambda item: item[1][1], reverse=True):
        categories.append({
            "category": category,
            "count": count,
            "total_ms": round(total, 6),
            "pct_of_kernel_time": round(100.0 * total / kernel_total, 3) if kernel_total else None,
        })
    return {
        "label": label,
        "kernel_count": len(selected),
        "kernel_time_ms": round(kernel_total, 6),
        "wall_ms": round(wall_ms, 6),
        "gpu_busy_ratio": round(kernel_total / wall_ms, 4) if wall_ms else None,
        "categories": categories,
        "top_kernels": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    root = data["rocprofiler-sdk-tool"][0]
    symbols = {item["kernel_id"]: item["kernel_name"] for item in root["kernel_symbols"]}
    records = root["buffer_records"]["kernel_dispatch"]
    markers = root["buffer_records"]["marker_api"]
    labels = [item for item in args.labels.split(",") if item]
    windows = []
    for index, marker in enumerate(markers):
        label = labels[index] if index < len(labels) else f"window{index}"
        windows.append(summarize_window(
            records,
            symbols,
            label,
            marker["start_timestamp"],
            marker["end_timestamp"],
        ))

    all_by_name = collections.defaultdict(lambda: [0, 0.0])
    for record in records:
        name = symbols.get(record["dispatch_info"]["kernel_id"], "unknown")
        duration_ms = (record["end_timestamp"] - record["start_timestamp"]) / 1e6
        all_by_name[name][0] += 1
        all_by_name[name][1] += duration_ms
    total = sum(value[1] for value in all_by_name.values())
    overall_top = []
    for name, (count, duration) in sorted(all_by_name.items(), key=lambda item: item[1][1], reverse=True)[:20]:
        overall_top.append({
            "kernel": name,
            "count": count,
            "total_ms": round(duration, 6),
            "avg_ms": round(duration / count, 6),
            "pct_of_kernel_time": round(100.0 * duration / total, 3) if total else None,
        })

    result = {
        "source": args.results,
        "process": root["metadata"].get("command"),
        "marker_count": len(markers),
        "total_kernel_dispatches": len(records),
        "total_kernel_time_ms": round(total, 6),
        "windows": windows,
        "overall_top_kernels": overall_top,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
