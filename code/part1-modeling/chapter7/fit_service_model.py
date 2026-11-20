#!/usr/bin/env python3
"""Fit the DES service-time model from real vLLM calibration runs.

Calibration inputs are `results/calibration/calibration_c{1,2,4,8}.json`.
The fresh v2 holdout (C=12/24/48/96) must not exist yet when this script is
frozen.  C=16/32/64 was measured under v1 and is burned; it cannot be used to
select the model.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from common import RESULTS, now_iso, save_json


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return my - slope * mx, slope


def main() -> None:
    paths = sorted(glob.glob(str(RESULTS / "calibration" / "calibration_c*.json")))
    if not paths:
        raise SystemExit("no calibration files found")
    points = []
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        c = int(raw["max_concurrency"])
        points.append({
            "concurrency": c,
            "file": Path(path).name,
            "num_prompts": raw["num_prompts"],
            "completed": raw["completed"],
            "failed": raw["failed"],
            "duration_s": raw["duration"],
            "request_throughput": raw["request_throughput"],
            "output_throughput": raw["output_throughput"],
            "ttft_p50_ms": raw["p50_ttft_ms"],
            "ttft_p95_ms": raw["p95_ttft_ms"],
            "ttft_p99_ms": raw["p99_ttft_ms"],
            "tpot_p50_ms": raw["p50_tpot_ms"],
            "tpot_p95_ms": raw["p95_tpot_ms"],
            "tpot_p99_ms": raw["p99_tpot_ms"],
            "e2el_p50_ms": raw["p50_e2el_ms"],
            "e2el_p99_ms": raw["p99_e2el_ms"],
            "max_concurrent_requests": raw.get("max_concurrent_requests"),
        })
    points.sort(key=lambda r: r["concurrency"])
    if [p["concurrency"] for p in points] != [1, 2, 4, 8]:
        raise SystemExit("expected calibration points C=1,2,4,8; got "
                         + str([p["concurrency"] for p in points]))

    # Prefill: compare two calibration-only candidates.  The first model used
    # log2(B); the calibration residuals already favoured linear-in-B, so v2
    # selects linear before any *new* holdout is opened.  The old C=16/32/64
    # holdout from v1 is burned and is not used for this selection.
    ys = [p["ttft_p50_ms"] for p in points]
    xs_log = [math.log2(p["concurrency"]) for p in points]
    xs_lin = [float(p["concurrency"]) for p in points]
    p0_log, p1_log = _fit_line(xs_log, ys)
    p0_lin, p1_lin = _fit_line(xs_lin, ys)
    p1_log = max(0.0, p1_log)
    p1_lin = max(0.0, p1_lin)

    def _rmse(a0: float, a1: float, xs: list[float]) -> float:
        return math.sqrt(sum((a0 + a1 * x - y) ** 2
                             for x, y in zip(xs, ys)) / len(xs))

    rmse_log = _rmse(p0_log, p1_log, xs_log)
    rmse_lin = _rmse(p0_lin, p1_lin, xs_lin)
    use_linear = rmse_lin <= rmse_log
    p0 = p0_lin if use_linear else p0_log
    p1 = p1_lin if use_linear else p1_log

    # Decode: TPOT p50 vs C on log-log axes.  In steady state the running
    # batch is approximately C for the closed-loop calibration.
    xs = [math.log(p["concurrency"]) for p in points]
    ys = [math.log(p["tpot_p50_ms"]) for p in points]
    intercept, gamma = _fit_line(xs, ys)
    gamma = min(max(gamma, 0.0), 0.5)
    t1 = math.exp(intercept)

    payload = {
        "source": "vllm bench serve calibration C=1,2,4,8 (real RX 9070 XT)",
        "frozen_at": now_iso(),
        "measurement_contract": {
            "model": "Qwen3-4B bf16",
            "input_len": 256,
            "output_len": 128,
            "request_rate": "inf",
            "max_concurrency": [1, 2, 4, 8],
            "num_prompts": 200,
            "seed": 42,
            "container": "rocm/vllm 0.29.1.dev0 (image tag vllm-0.29.0)",
            "gpu": "RX 9070 XT 15.9 GiB, HIP_VISIBLE_DEVICES=1",
        },
        "model_version": 2,
        "model_selection": {
            "policy": "choose the lower calibration RMSE on C=1,2,4,8 only",
            "v1_holdout_note": ("C=16/32/64 was measured under v1 log2 and is "
                                "burned; it is not used for this selection"),
            "candidates": {
                "log2": {"p0_ms": round(p0_log, 4),
                         "p1_ms_per_log2": round(p1_log, 4),
                         "rmse_ms": round(rmse_log, 3)},
                "linear": {"p0_ms": round(p0_lin, 4),
                           "p1_ms_per_seq": round(p1_lin, 4),
                           "rmse_ms": round(rmse_lin, 3)},
            },
            "selected": "linear" if use_linear else "log2",
        },
        "prefill": ({
            "mode": "linear",
            "p0_ms": round(p0_lin, 4),
            "p1_ms_per_seq": round(p1_lin, 4),
            "p1_ms_per_log2": round(p1_log, 4),
            "formula": "t_prefill(B) = p0 + p1*B",
            "fit": "least squares on p50 TTFT vs C (calibration only)",
        } if use_linear else {
            "mode": "log2",
            "p0_ms": round(p0_log, 4),
            "p1_ms_per_log2": round(p1_log, 4),
            "p1_ms_per_seq": round(p1_lin, 4),
            "formula": "t_prefill(B) = p0 + p1*log2(max(B,1))",
            "fit": "least squares on p50 TTFT vs log2(C) (calibration only)",
        }),
        "decode": {
            "t1_ms": round(t1, 4),
            "gamma": round(gamma, 6),
            "formula": "t_decode(B) = t1*max(B,1)^gamma",
            "fit": "least squares on log(p50 TPOT) vs log(C)",
        },
        "calibration_points": points,
        "predictions_for_this_file": {
            "t_prefill_ms": {str(p["concurrency"]): round(
                p0 + p1 * (p["concurrency"] if use_linear else math.log2(p["concurrency"])), 3)
                for p in points},
            "t_decode_ms": {str(p["concurrency"]): round(t1 * p["concurrency"] ** gamma, 3)
                            for p in points},
        },
    }
    path = save_json("service_model.json", payload)
    save_json("calibration_summary.json", {
        "generated_at": now_iso(),
        "points": points,
        "fit": payload["prefill"] | payload["decode"],
    })
    print(f"Wrote {path}")
    if use_linear:
        print(f"prefill: {p0:.3f} + {p1:.3f}*B ms  (RMSE {rmse_lin:.2f} ms, "
              f"log2 RMSE {rmse_log:.2f} ms)")
    else:
        print(f"prefill: {p0:.3f} + {p1:.3f}*log2(B) ms  (RMSE {rmse_log:.2f} ms, "
              f"linear RMSE {rmse_lin:.2f} ms)")
    print(f"decode : {t1:.3f} * B^{gamma:.4f} ms")


if __name__ == "__main__":
    main()
