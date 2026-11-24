#!/usr/bin/env python3
"""Recompute chapter-8 framework results from raw request records.

The original summary files are kept as measured artifacts.  This script only
adds a derived view: it fixes TPOT to ``(E2EL - TTFT) / (completion_tokens - 1)``
because an SSE chunk may contain more than one token.  It also recomputes the
request-level SLO verdict, failure reasons and cross-framework ratios.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "framework"
OUT = RESULTS / "analysis.json"
FRAMEWORKS = ("vllm", "llamacpp")
CONCURRENCIES = (1, 8, 32, 64)
SLO = {
    "ttft_p99_ms": 800.0,
    "tpot_p99_ms": 100.0,
    "e2el_p99_ms": 15000.0,
    "failure_rate_max": 0.01,
}
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2


def percentile(values: Iterable[float], p: float) -> float | None:
    data = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return data[low]
    return data[low] * (high - position) + data[high] * (position - low)


def finite(value: float | None) -> float | None:
    return round(value, 6) if value is not None and math.isfinite(value) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corrected_tpot_ms(row: dict) -> float | None:
    if row.get("error") or row.get("ttft_ms") is None or row.get("e2el_ms") is None:
        return None
    token_count = int(row.get("n_completion_tokens") or 0)
    if token_count <= 1:
        return None
    return (float(row["e2el_ms"]) - float(row["ttft_ms"])) / (token_count - 1)


def failure_reasons(row: dict) -> list[str]:
    if row.get("error") or row.get("ttft_ms") is None:
        return ["transport"]
    reasons: list[str] = []
    if row["ttft_ms"] > SLO["ttft_p99_ms"]:
        reasons.append("ttft")
    tpot = row.get("tpot_corrected_ms")
    if tpot is None or tpot > SLO["tpot_p99_ms"]:
        reasons.append("tpot")
    if row["e2el_ms"] > SLO["e2el_p99_ms"]:
        reasons.append("e2el")
    return reasons


def analyze_point(
    framework: str,
    concurrency: int,
    root: Path = RESULTS,
    point_dir: Path | None = None,
) -> dict:
    if point_dir is None:
        point_dir = root / framework / f"c{concurrency}"
    requests_path = point_dir / "requests.jsonl"
    summary_path = point_dir / "summary.json"
    rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]
    old = json.loads(summary_path.read_text(encoding="utf-8"))

    for row in rows:
        row["tpot_corrected_ms"] = corrected_tpot_ms(row)
        row["slo_failure_reasons"] = failure_reasons(row)
        row["slo_pass"] = not row["slo_failure_reasons"]

    completed = [row for row in rows if not row.get("error") and row.get("ttft_ms") is not None]
    passed = [row for row in rows if row["slo_pass"]]
    ttft_p99 = percentile([row["ttft_ms"] for row in completed], 99)
    tpot_p99 = percentile(
        [row["tpot_corrected_ms"] for row in completed if row.get("tpot_corrected_ms") is not None],
        99,
    )
    e2el_p99 = percentile([row["e2el_ms"] for row in completed], 99)
    failure_rate = (len(rows) - len(completed)) / len(rows)
    verdict = (
        "PASS"
        if (
            ttft_p99 is not None
            and ttft_p99 <= SLO["ttft_p99_ms"]
            and tpot_p99 is not None
            and tpot_p99 <= SLO["tpot_p99_ms"]
            and e2el_p99 is not None
            and e2el_p99 <= SLO["e2el_p99_ms"]
            and failure_rate <= SLO["failure_rate_max"]
        )
        else "FAIL"
    )
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["slo_failure_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    mismatch = [
        row for row in completed
        if row.get("n_stream_chunks") != row.get("n_completion_tokens")
    ]
    headrooms = {}
    if ttft_p99 is not None:
        headrooms["ttft"] = 1.0 - ttft_p99 / SLO["ttft_p99_ms"]
    if tpot_p99 is not None:
        headrooms["tpot"] = 1.0 - tpot_p99 / SLO["tpot_p99_ms"]
    if e2el_p99 is not None:
        headrooms["e2el"] = 1.0 - e2el_p99 / SLO["e2el_p99_ms"]
    headrooms["failure_rate"] = 1.0 - failure_rate / SLO["failure_rate_max"]
    tightest = min(headrooms, key=headrooms.get)

    wall_s = float(old["wall_s"])
    output_tokens = sum(int(row.get("n_completion_tokens") or 0) for row in completed)
    goodput_tokens = sum(int(row.get("n_completion_tokens") or 0) for row in passed)
    stream_itls = [
        interval for row in completed for interval in row.get("itls_ms", [])
    ]
    return {
        "framework": framework,
        "concurrency": concurrency,
        "requests": len(rows),
        "successful_requests": len(completed),
        "failed_requests": len(rows) - len(completed),
        "failure_rate": round(failure_rate, 6),
        "slo_passed_requests": len(passed),
        "slo_pass_rate": round(len(passed) / len(rows), 6),
        "slo_verdict": verdict,
        "slo_failure_reason_counts": reason_counts,
        "tightest_constraint": tightest,
        "headroom": {key: round(value, 6) for key, value in headrooms.items()},
        "ttft_ms": {
            "p50": finite(percentile([row["ttft_ms"] for row in completed], 50)),
            "p95": finite(percentile([row["ttft_ms"] for row in completed], 95)),
            "p99": finite(ttft_p99),
            "max": finite(max((row["ttft_ms"] for row in completed), default=None)),
        },
        "tpot_corrected_ms": {
            "p50": finite(percentile(
                [row["tpot_corrected_ms"] for row in completed if row.get("tpot_corrected_ms") is not None],
                50,
            )),
            "p95": finite(percentile(
                [row["tpot_corrected_ms"] for row in completed if row.get("tpot_corrected_ms") is not None],
                95,
            )),
            "p99": finite(tpot_p99),
        },
        "tpot_stream_chunk_ms_old": old.get("tpot_ms"),
        "stream_chunk_interval_ms_aux": {
            "count": len(stream_itls),
            "p50": finite(percentile(stream_itls, 50)),
            "p99": finite(percentile(stream_itls, 99)),
        },
        "e2el_ms": {
            "p50": finite(percentile([row["e2el_ms"] for row in completed], 50)),
            "p95": finite(percentile([row["e2el_ms"] for row in completed], 95)),
            "p99": finite(e2el_p99),
        },
        "wall_s": round(wall_s, 6),
        "request_throughput_per_s": round(len(completed) / wall_s, 6),
        "output_tokens_per_s": round(output_tokens / wall_s, 6),
        "slo_goodput_requests_per_s": round(len(passed) / wall_s, 6),
        "slo_goodput_output_tokens_per_s": round(goodput_tokens / wall_s, 6),
        "stream_chunk_token_mismatch_requests": len(mismatch),
        "stream_chunk_token_mismatch_max_tokens": max(
            (int(row.get("n_completion_tokens") or 0) for row in mismatch), default=0
        ),
        "metrics_delta": old.get("metrics_delta", {}),
        "vram": old.get("vram", {}),
        "sources": {
            "requests": str(requests_path.relative_to(HERE)),
            "requests_sha256": sha256(requests_path),
            "original_summary": str(summary_path.relative_to(HERE)),
            "original_summary_sha256": sha256(summary_path),
        },
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 6)


def main() -> None:
    points = {
        framework: [analyze_point(framework, concurrency) for concurrency in CONCURRENCIES]
        for framework in FRAMEWORKS
    }
    comparison = []
    for concurrency in CONCURRENCIES:
        vllm = next(point for point in points["vllm"] if point["concurrency"] == concurrency)
        llama = next(point for point in points["llamacpp"] if point["concurrency"] == concurrency)
        comparison.append({
            "concurrency": concurrency,
            "vllm_slo": vllm["slo_verdict"],
            "llamacpp_slo": llama["slo_verdict"],
            "ttft_p99_ratio_llamacpp_over_vllm": ratio(
                llama["ttft_ms"]["p99"], vllm["ttft_ms"]["p99"]
            ),
            "tpot_p99_ratio_llamacpp_over_vllm": ratio(
                llama["tpot_corrected_ms"]["p99"], vllm["tpot_corrected_ms"]["p99"]
            ),
            "raw_token_throughput_ratio_vllm_over_llamacpp": ratio(
                vllm["output_tokens_per_s"], llama["output_tokens_per_s"]
            ),
            "slo_goodput_ratio_vllm_over_llamacpp": ratio(
                vllm["slo_goodput_requests_per_s"], llama["slo_goodput_requests_per_s"]
            ),
        })

    max_safe = {}
    first_breach = {}
    for framework in FRAMEWORKS:
        safe = [point["concurrency"] for point in points[framework] if point["slo_verdict"] == "PASS"]
        failed = [point["concurrency"] for point in points[framework] if point["slo_verdict"] == "FAIL"]
        max_safe[framework] = max(safe) if safe else None
        first_breach[framework] = min(failed) if failed else None

    analysis = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "DERIVED_FROM_RAW_REQUESTS",
        "hardware_validated": True,
        "correction": {
            "field": "tpot_ms",
            "old_formula": "(last streamed token time - first streamed token time)/(stream chunks-1)",
            "new_formula": "(E2EL - TTFT)/(completion_tokens-1)",
            "reason": "vLLM and llama.cpp can coalesce multiple completion tokens into one SSE chunk",
            "raw_requests_unchanged": True,
            "original_summaries_unchanged": True,
        },
        "contract": {
            "model": "Qwen3-4B",
            "gpu": "AMD Radeon RX 7900 XTX 23.98 GiB (gfx1100, GPU0)",
            "input_tokens": 256,
            "output_tokens": 128,
            "num_prompts_per_point": 200,
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "concurrency": list(CONCURRENCIES),
            "slo": SLO,
        },
        "models": {
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "kv_kib_per_token": round(KV_BYTES_PER_TOKEN / 1024, 3),
            "vllm_kv_tokens": 77248,
            "vllm_kv_pool_gib": round(77248 * KV_BYTES_PER_TOKEN / 1024**3, 3),
            "vllm_context_capacity_4096": round(77248 / 4096, 3),
            "llamacpp_slots": 64,
            "llamacpp_context_per_slot": 512,
            "llamacpp_total_context_tokens": 64 * 512,
            "llamacpp_context_capacity_384": round(64 * 512 / 384, 3),
            "failed_kv_probe_context_tokens": 131072,
            "failed_kv_probe_gib": round(131072 * KV_BYTES_PER_TOKEN / 1024**3, 3),
        },
        "points": points,
        "comparison": comparison,
        "capacity": {
            "max_slo_safe_tested_concurrency": max_safe,
            "first_slo_breach_concurrency": first_breach,
            "raw_throughput_peak": {
                framework: max(
                    points[framework], key=lambda point: point["output_tokens_per_s"]
                )["output_tokens_per_s"]
                for framework in FRAMEWORKS
            },
            "slo_goodput_peak": {
                framework: max(
                    points[framework], key=lambda point: point["slo_goodput_requests_per_s"]
                )["slo_goodput_requests_per_s"]
                for framework in FRAMEWORKS
            },
        },
        "notes": [
            "Raw requests.jsonl and original summary.json files are preserved; this file is a derived view.",
            "Stream-chunk intervals are retained as auxiliary evidence, not as TPOT.",
            "The two engines are not equivalent implementations; see experiment-draft.yaml for the ledger.",
            "This analysis does not create a pre-measurement prediction or a held-out framework ranking.",
        ],
    }
    OUT.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for framework in FRAMEWORKS:
        for point in points[framework]:
            print(
                f"{framework:8s} C={point['concurrency']:2d} "
                f"TTFT99={point['ttft_ms']['p99']:.2f}ms "
                f"TPOT99={point['tpot_corrected_ms']['p99']:.2f}ms "
                f"tok/s={point['output_tokens_per_s']:.2f} "
                f"goodput={point['slo_goodput_requests_per_s']:.3f} "
                f"{point['slo_verdict']} tight={point['tightest_constraint']}"
            )
    print(json.dumps(analysis["capacity"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
