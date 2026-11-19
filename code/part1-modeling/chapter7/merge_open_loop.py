#!/usr/bin/env python3
"""Rebuild an open-loop summary from raw JSON files.

The long λ=1 point can outlive the driver process; the raw benchmark JSON is
still valid.  This script merges all open_R*.json files and keeps /metrics
summaries where they exist.  For points without a metrics sample it estimates
the engine running batch with Little's Law (L = X * W), explicitly labelled.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from common import RESULTS, now_iso, save_json


def main() -> None:
    old = {}
    summary_path = RESULTS / "open_loop" / "summary.json"
    if summary_path.exists():
        for p in json.loads(summary_path.read_text(encoding="utf-8"))["points"]:
            if "tag" in p:
                old[p["tag"]] = p

    points = []
    for path in glob.glob(str(RESULTS / "open_loop" / "open_R*.json")):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rate = float(raw["request_rate"])
        tag = f"open_R{rate:g}"
        prior = old.get(tag, {})
        metrics = prior.get("metrics") or {}
        mean_in_system = raw["request_throughput"] * raw["p50_e2el_ms"] / 1000.0
        running_est = min(64, max(1, round(mean_in_system)))
        points.append({
            "tag": tag,
            "mode": "open",
            "point": rate,
            "input_len": raw["input_lens"][0],
            "output_len": raw["output_lens"][0],
            "num_prompts": raw["num_prompts"],
            "completed": raw["completed"],
            "failed": raw["failed"],
            "failure_rate": raw["failed"] / max(raw["completed"] + raw["failed"], 1),
            "duration_s": round(raw["duration"], 3),
            "request_throughput": round(raw["request_throughput"], 3),
            "request_goodput": round(raw["request_goodput"], 3),
            "output_throughput": round(raw["output_throughput"], 3),
            "ttft_p50_ms": round(raw["p50_ttft_ms"], 2),
            "ttft_p95_ms": round(raw["p95_ttft_ms"], 2),
            "ttft_p99_ms": round(raw["p99_ttft_ms"], 2),
            "tpot_p50_ms": round(raw["p50_tpot_ms"], 2),
            "tpot_p95_ms": round(raw["p95_tpot_ms"], 2),
            "tpot_p99_ms": round(raw["p99_tpot_ms"], 2),
            "itl_p99_ms": round(raw["p99_itl_ms"], 2),
            "e2el_p50_ms": round(raw["p50_e2el_ms"], 2),
            "e2el_p95_ms": round(raw["p95_e2el_ms"], 2),
            "e2el_p99_ms": round(raw["p99_e2el_ms"], 2),
            "max_concurrent_requests": raw.get("max_concurrent_requests"),
            "metrics": metrics or None,
            "running_peak_est": running_est,
            "metrics_note": ("observed /metrics" if metrics
                             else "estimated with Little's Law from X and p50 E2E"),
            "raw_json": str(Path(path).relative_to(RESULTS)),
        })
    points.sort(key=lambda p: p["point"])
    payload = {
        "generated_at": now_iso(),
        "mode": "open",
        "out_dir": "open_loop",
        "points": points,
    }
    path = save_json("open_loop/summary_all.json", payload)
    print(f"Wrote {path} with {len(points)} points")
    for p in points:
        print(f"  {p['tag']:>9} achieved={p['request_throughput']:.2f} "
              f"goodput={p['request_goodput']:.2f} TTFTp99={p['ttft_p99_ms']:.0f} "
              f"batch_est={p['running_peak_est']}")


if __name__ == "__main__":
    main()
