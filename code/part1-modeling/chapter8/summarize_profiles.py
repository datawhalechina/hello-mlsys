#!/usr/bin/env python3
"""Summarize framework-level kernel traces for chapter 8.

Inputs are intentionally read from outside the repository because the raw
traces are hundreds of megabytes.  The committed result contains the aggregate
evidence, hashes and top-kernel tables; the trace itself stays in /tmp.

Examples:
  python summarize_profiles.py \
    --llama-trace /tmp/ch8-llama-kernel_trace.csv \
    --vllm-trace /tmp/ch8-vllm-torch-trace.json
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "framework"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], p: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return data[low]
    return data[low] * (high - position) + data[high] * (position - low)


def category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("flash_attn", "paged_attention", "attention", "_fwd_kernel")):
        return "attention"
    if any(token in lowered for token in ("cijk_", "wvsplitk", "mul_mat", "gemm", "matmul", "rocblas", "hipblas")):
        return "gemm"
    if any(token in lowered for token in ("rms_norm", "rope", "layernorm", "layer_norm")):
        return "norm_rope"
    if any(token in lowered for token in ("reshape_and_cache", "copybuffer", "memcpy", "copy", "k_set_rows", "convert_unary")):
        return "copy_convert"
    if any(token in lowered for token in ("triton_poi_fused", "unary_gated", "silu", "k_bin_bcast", "elementwise", "scatter", "gather", "index", "fill", "arange")):
        return "elementwise"
    if any(token in lowered for token in ("reduce", "triton_red", "combine_results", "argmax", "sum")):
        return "reduce"
    if any(token in lowered for token in ("sample", "gumbel", "slot", "block_table", "prepare", "post_update", "write", "combine_sampled")):
        return "sampling_control"
    return "other"


def metric_summary(durations_ns: list[float]) -> dict:
    total = sum(durations_ns)
    return {
        "kernels": len(durations_ns),
        "kernel_ms_total": round(total / 1e6, 3),
        "kernel_ms_p50": round(percentile(durations_ns, 50) / 1e6, 6),
        "kernel_ms_p95": round(percentile(durations_ns, 95) / 1e6, 6),
        "kernel_ms_p99": round(percentile(durations_ns, 99) / 1e6, 6),
        "kernel_ms_max": round(max(durations_ns, default=0) / 1e6, 6),
    }


def aggregate(events: list[tuple[str, int, int]]) -> dict:
    start = min(event[1] for event in events)
    end = max(event[2] for event in events)
    duration_total = sum(event[2] - event[1] for event in events)
    by_category: dict[str, list[int]] = defaultdict(list)
    by_name: dict[str, list[int]] = defaultdict(list)
    for name, begin, finish in events:
        duration = finish - begin
        by_category[category(name)].append(duration)
        by_name[name].append(duration)
    category_rows = []
    for name, durations in sorted(by_category.items(), key=lambda item: -sum(item[1])):
        category_rows.append({
            "category": name,
            "kernels": len(durations),
            "kernel_ms_total": round(sum(durations) / 1e6, 3),
            "share_pct": round(100.0 * sum(durations) / duration_total, 3),
            **metric_summary(durations),
        })
    top_rows = []
    for name, durations in sorted(by_name.items(), key=lambda item: -sum(item[1]))[:30]:
        top_rows.append({
            "kernel_name": name,
            "category": category(name),
            "kernels": len(durations),
            "kernel_ms_total": round(sum(durations) / 1e6, 3),
            "share_pct": round(100.0 * sum(durations) / duration_total, 3),
            **metric_summary(durations),
        })
    return {
        "kernel_dispatches": len(events),
        "span_ms": round((end - start) / 1e6, 3),
        "kernel_ms_total": round(duration_total / 1e6, 3),
        "gpu_busy_pct": round(100.0 * duration_total / (end - start), 3),
        **metric_summary([event[2] - event[1] for event in events]),
        "categories": category_rows,
        "top_kernels": top_rows,
    }


def read_llama(path: Path) -> list[tuple[str, int, int]]:
    events = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Kind") != "KERNEL_DISPATCH":
                continue
            events.append((row["Kernel_Name"], int(row["Start_Timestamp"]), int(row["End_Timestamp"])))
    return events


ANNOTATION = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def annotation_phase(name: str) -> str:
    match = ANNOTATION.search(name)
    if not match:
        return "unattributed"
    prefetched_context = int(match.group(2))
    generation = int(match.group(4))
    if prefetched_context == 0:
        return "decode"
    if generation == 0:
        return "prefill"
    return "mixed_chunked_prefill"


def read_vllm(path: Path) -> tuple[list[tuple[str, int, int]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = []
    annotations = []
    for event in payload.get("traceEvents", []):
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            begin = int(event["ts"] * 1000)
            events.append((event["name"], begin, begin + int(event["dur"] * 1000)))
        elif event.get("cat") == "gpu_user_annotation" and event.get("ph") == "X":
            begin = int(event["ts"] * 1000)
            annotations.append((event["name"], begin, begin + int(event["dur"] * 1000)))
    annotations.sort(key=lambda item: item[1])
    starts = [item[1] for item in annotations]
    phases: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    phase_windows: dict[str, list[int]] = defaultdict(list)
    unattributed = []
    for name, begin, finish in events:
        index = bisect.bisect_right(starts, begin) - 1
        chosen = None
        for candidate in range(index, max(-1, index - 32), -1):
            annotation_name, annotation_begin, annotation_finish = annotations[candidate]
            if annotation_begin <= begin and finish <= annotation_finish:
                if chosen is None or annotation_finish - annotation_begin < chosen[2] - chosen[1]:
                    chosen = annotations[candidate]
        if chosen is None:
            unattributed.append((name, begin, finish))
        else:
            phases[annotation_phase(chosen[0])].append((name, begin, finish))
    for name, begin, finish in annotations:
        phase_windows[annotation_phase(name)].append(finish - begin)

    phase_summary = {}
    for phase in ("prefill", "decode", "mixed_chunked_prefill", "unattributed"):
        phase_events = phases.get(phase, [])
        durations = [finish - begin for _, begin, finish in phase_events]
        windows = phase_windows.get(phase, [])
        phase_summary[phase] = {
            "annotations": len(windows),
            "annotation_ms_total": round(sum(windows) / 1e6, 3),
            "kernels": len(durations),
            "kernel_ms_total": round(sum(durations) / 1e6, 3),
            "kernel_ms_p50": round(percentile(durations, 50) / 1e6, 6),
            "kernel_ms_p99": round(percentile(durations, 99) / 1e6, 6),
            "kernels_per_annotation": round(len(durations) / len(windows), 3) if windows else None,
            "kernel_ms_per_annotation": round(sum(durations) / 1e6 / len(windows), 3) if windows else None,
            "annotation_ms_per_annotation": round(sum(windows) / 1e6 / len(windows), 3) if windows else None,
            "busy_pct": round(100.0 * sum(durations) / sum(windows), 3) if windows else None,
            "categories": aggregate(phase_events)["categories"] if phase_events else [],
        }
    if unattributed:
        durations = [finish - begin for _, begin, finish in unattributed]
        phase_summary["unattributed"] = {
            "annotations": 0,
            "annotation_ms_total": 0,
            "kernels": len(durations),
            "kernel_ms_total": round(sum(durations) / 1e6, 3),
            "kernel_ms_p50": round(percentile(durations, 50) / 1e6, 6),
            "kernel_ms_p99": round(percentile(durations, 99) / 1e6, 6),
            "categories": aggregate(unattributed)["categories"],
        }
    return events, {"phases": phase_summary}


def per_request_corrected(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    corrected = []
    for row in rows:
        if row.get("error") or row.get("ttft_ms") is None:
            continue
        tokens = int(row.get("n_completion_tokens") or 0)
        if tokens > 1:
            corrected.append((row["e2el_ms"] - row["ttft_ms"]) / (tokens - 1))
    return {
        "requests": len(rows),
        "corrected_tpot_ms_p50": round(percentile(corrected, 50), 6),
        "corrected_tpot_ms_p99": round(percentile(corrected, 99), 6),
        "stream_chunk_mismatch_requests": sum(
            1 for row in rows if row.get("n_stream_chunks") != row.get("n_completion_tokens")
        ),
    }


def write_top_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-trace", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--llama-requests", type=Path, default=RESULTS / "profiles" / "llama" / "requests.jsonl")
    parser.add_argument("--vllm-requests", type=Path, default=RESULTS / "profiles" / "vllm" / "requests.jsonl")
    parser.add_argument("--out", type=Path, default=RESULTS / "profile-summary.json")
    args = parser.parse_args()

    llama_events = read_llama(args.llama_trace)
    vllm_events, vllm_phases = read_vllm(args.vllm_trace)
    profile_output_tokens = 2560
    profile_input_tokens = 5120

    def normalize(item: dict) -> dict:
        return {
            "kernels_per_output_token": round(item["kernel_dispatches"] / profile_output_tokens, 3),
            "kernel_ms_per_output_token": round(item["kernel_ms_total"] / profile_output_tokens, 6),
            "kernel_ms_per_processed_token": round(
                item["kernel_ms_total"] / (profile_output_tokens + profile_input_tokens), 6
            ),
        }

    llama_summary = aggregate(llama_events)
    vllm_summary = aggregate(vllm_events)
    summary = {
        "generated_from_raw_traces": True,
        "gpu": "AMD Radeon RX 7900 XTX 23.98 GiB (gfx1100, GPU0)",
        "model": "Qwen3-4B",
        "workload": {
            "input_tokens": 256,
            "output_tokens": 128,
            "concurrency": 8,
            "requests": 20,
            "warmup_requests": 2,
            "successful_output_tokens": profile_output_tokens,
            "successful_input_tokens": profile_input_tokens,
        },
        "trace_tools": {
            "llama": "rocprofv3 1.4.1 attach, --kernel-trace --hip-graph-trace, 40 s window",
            "vllm": "vLLM /start_profile + /stop_profile, torch profiler, rocprofiler-sdk 1.4.1 backend",
        },
        "trace_files": {
            "llama": {
                "path": str(args.llama_trace),
                "bytes": args.llama_trace.stat().st_size,
                "sha256": sha256(args.llama_trace),
            },
            "vllm": {
                "path": str(args.vllm_trace),
                "bytes": args.vllm_trace.stat().st_size,
                "sha256": sha256(args.vllm_trace),
            },
        },
        "llama": {
            **llama_summary,
            **normalize(llama_summary),
            "phase_split": None,
            "phase_split_note": "llama.cpp server has no per-iteration annotation in this trace; do not invent a phase split.",
            "profile_request_metrics": per_request_corrected(args.llama_requests),
        },
        "vllm": {
            **vllm_summary,
            **normalize(vllm_summary),
            **vllm_phases,
            "profile_request_metrics": per_request_corrected(args.vllm_requests),
        },
        "caveats": [
            "Trace collection perturbs wall-clock latency; use the low-overhead sweep for SLO numbers.",
            "llama.cpp trace is attach-mode rocprofv3; vLLM trace is torch profiler. The kernel attribution is comparable at the dispatch-family level, not as an identical instrumentation path.",
            "vLLM phase labels come from execute_context_*_generation_* GPU annotations. llama.cpp phase labels are absent.",
            "The raw traces are local /tmp artifacts and are not committed; hashes bind this summary to them.",
        ],
    }
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for framework in ("llama", "vllm"):
        rows = summary[framework]["top_kernels"]
        write_top_csv(RESULTS / "profiles" / framework / "top-kernels.csv", rows)
    print(f"wrote {args.out}")
    for framework in ("llama", "vllm"):
        item = summary[framework]
        print(
            f"{framework:5s} dispatches={item['kernel_dispatches']} "
            f"span={item['span_ms']/1000:.2f}s busy={item['gpu_busy_pct']:.1f}% "
            f"kernel={item['kernel_ms_total']/1000:.2f}s"
        )
    print(json.dumps(vllm_phases, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
