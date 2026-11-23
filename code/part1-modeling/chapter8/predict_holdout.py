#!/usr/bin/env python3
"""Freeze C=16/48 predictions before the holdout measurements.

Calibration uses only the already-measured C=1/8/32 points.  C=64 is excluded
from the fit.  The model is deliberately small: a two-parameter power law for
TTFT P99 and TPOT P99.  E2E P99 is reconstructed from fixed OSL=128.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "results" / "framework" / "analysis.json"
OUT = HERE / "results" / "framework" / "prediction-holdout.json"
FRAMEWORKS = ("vllm", "llamacpp")
CALIBRATION = (1, 8, 32)
HOLDOUT = (16, 48)
EXCLUDED = (64,)
OUTPUT_TOKENS = 128
SLO = {
    "ttft_p99_ms": 800.0,
    "tpot_p99_ms": 100.0,
    "e2el_p99_ms": 15000.0,
    "failure_rate_max": 0.01,
}


def fit_loglog(points: list[tuple[float, float]]) -> dict:
    xs = [math.log(point[0]) for point in points]
    ys = [math.log(point[1]) for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    residual_std = math.sqrt(sum(value ** 2 for value in residuals) / max(1, len(residuals) - 2))
    return {
        "intercept_ln": intercept,
        "slope": slope,
        "residual_log_std": residual_std,
        "calibration_mape_pct": 100.0 * sum(abs(math.expm1(value)) for value in residuals) / len(residuals),
    }


def predict(model: dict, concurrency: float) -> float:
    return math.exp(model["intercept_ln"] + model["slope"] * math.log(concurrency))


def slo_verdict(ttft: float, tpot: float, e2e: float, failure_rate: float = 0.0) -> str:
    return (
        "PASS"
        if ttft <= SLO["ttft_p99_ms"]
        and tpot <= SLO["tpot_p99_ms"]
        and e2e <= SLO["e2el_p99_ms"]
        and failure_rate <= SLO["failure_rate_max"]
        else "FAIL"
    )


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    predictions = {}
    for framework in FRAMEWORKS:
        points = analysis["points"][framework]
        calibration = [point for point in points if point["concurrency"] in CALIBRATION]
        ttft_model = fit_loglog([
            (point["concurrency"], point["ttft_ms"]["p99"]) for point in calibration
        ])
        tpot_model = fit_loglog([
            (point["concurrency"], point["tpot_corrected_ms"]["p99"]) for point in calibration
        ])
        rows = []
        for concurrency in HOLDOUT:
            ttft = predict(ttft_model, concurrency)
            tpot = predict(tpot_model, concurrency)
            e2e = ttft + (OUTPUT_TOKENS - 1) * tpot
            ttft_band = math.exp(1.96 * ttft_model["residual_log_std"])
            tpot_band = math.exp(1.96 * tpot_model["residual_log_std"])
            rows.append({
                "concurrency": concurrency,
                "ttft_p99_ms": round(ttft, 3),
                "ttft_p99_heuristic_band_ms": [
                    round(ttft / ttft_band, 3), round(ttft * ttft_band, 3)
                ],
                "tpot_p99_ms": round(tpot, 3),
                "tpot_p99_heuristic_band_ms": [
                    round(tpot / tpot_band, 3), round(tpot * tpot_band, 3)
                ],
                "e2el_p99_ms": round(e2e, 3),
                "predicted_slo_verdict": slo_verdict(ttft, tpot, e2e),
            })
        predictions[framework] = {
            "calibration_points": list(CALIBRATION),
            "excluded_points": list(EXCLUDED),
            "ttft_model": ttft_model,
            "tpot_model": tpot_model,
            "predictions": rows,
        }

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "FROZEN_BEFORE_HOLDOUT_MEASUREMENT",
        "experiment": "chapter8 framework holdout",
        "contract": {
            "model": "Qwen3-4B",
            "gpu": "AMD Radeon RX 7900 XTX 23.98 GiB (gfx1100, GPU0)",
            "input_tokens": 256,
            "output_tokens": OUTPUT_TOKENS,
            "num_prompts_per_point": 200,
            "warmup_per_point": 2,
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "concurrency": list(HOLDOUT),
            "slo": SLO,
        },
        "method": {
            "ttft": "log(TTFT_p99_ms) = a + b*log(concurrency), fit on C=1/8/32",
            "tpot": "log(TPOT_p99_ms) = a + b*log(concurrency), fit on C=1/8/32",
            "e2e": "TTFT_p99 + 127*TPOT_p99 (fixed OSL=128)",
            "slo_verdict": "predicted P99 thresholds; failure rate assumed 0 because no transport errors were observed",
            "interval": "exp(+/-1.96*residual_log_std) heuristic band; with 3 calibration points this is not a statistical confidence interval",
        },
        "selection_disclosure": {
            "c64_observed_before_this_file": True,
            "c64_used_in_fit": False,
            "c16_c48_measured_before_this_file": False,
            "note": "C=16/48 are the holdout points. C=64 is a burned point and is excluded from the fit and from acceptance.",
        },
        "predictions": predictions,
        "acceptance": {
            "primary": "compare measured TTFT/TPOT/E2E P99 against the frozen point predictions",
            "secondary": "compare measured SLO verdict against predicted SLO verdict",
            "do_not": "do not refit or change the holdout points after this file is written",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} at {payload['generated_at']}")
    for framework in FRAMEWORKS:
        print(framework)
        for row in predictions[framework]["predictions"]:
            print(
                f"  C={row['concurrency']:2d} TTFT99={row['ttft_p99_ms']:.2f} "
                f"TPOT99={row['tpot_p99_ms']:.2f} E2E99={row['e2el_p99_ms']:.2f} "
                f"{row['predicted_slo_verdict']}"
            )


if __name__ == "__main__":
    main()
