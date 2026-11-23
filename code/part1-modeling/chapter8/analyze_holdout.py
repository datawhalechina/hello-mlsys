#!/usr/bin/env python3
"""Compare frozen C=16/48 predictions with the holdout measurements."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import analyze_framework as af

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "framework"
PREDICTION = ROOT / "prediction-holdout.json"
OUT = ROOT / "holdout-analysis.json"


def pct_error(predicted: float, actual: float) -> float:
    return 100.0 * (actual - predicted) / predicted


def main() -> None:
    prediction = json.loads(PREDICTION.read_text(encoding="utf-8"))
    rows = []
    errors: dict[str, list[float]] = {"ttft": [], "tpot": [], "e2el": []}
    verdict_matches = 0
    for framework in af.FRAMEWORKS:
        predicted_rows = {
            row["concurrency"]: row
            for row in prediction["predictions"][framework]["predictions"]
        }
        for concurrency in (16, 48):
            actual = af.analyze_point(
                framework,
                concurrency,
                point_dir=ROOT / f"{framework}-holdout" / f"c{concurrency}",
            )
            predicted = predicted_rows[concurrency]
            actual_ttft = actual["ttft_ms"]["p99"]
            actual_tpot = actual["tpot_corrected_ms"]["p99"]
            actual_e2el = actual["e2el_ms"]["p99"]
            row = {
                "framework": framework,
                "concurrency": concurrency,
                "predicted": predicted,
                "actual": {
                    "ttft_p99_ms": actual_ttft,
                    "tpot_p99_ms": actual_tpot,
                    "e2el_p99_ms": actual_e2el,
                    "slo_verdict": actual["slo_verdict"],
                    "slo_passed_requests": actual["slo_passed_requests"],
                    "slo_failure_reason_counts": actual["slo_failure_reason_counts"],
                    "output_tokens_per_s": actual["output_tokens_per_s"],
                    "slo_goodput_requests_per_s": actual["slo_goodput_requests_per_s"],
                    "tightest_constraint": actual["tightest_constraint"],
                    "headroom": actual["headroom"],
                    "wall_s": actual["wall_s"],
                },
                "error_pct": {
                    "ttft": round(pct_error(predicted["ttft_p99_ms"], actual_ttft), 3),
                    "tpot": round(pct_error(predicted["tpot_p99_ms"], actual_tpot), 3),
                    "e2el": round(pct_error(predicted["e2el_p99_ms"], actual_e2el), 3),
                },
                "slo_verdict_match": predicted["predicted_slo_verdict"] == actual["slo_verdict"],
                "sources": actual["sources"],
            }
            rows.append(row)
            for metric in errors:
                errors[metric].append(abs(row["error_pct"][metric]))
            verdict_matches += int(row["slo_verdict_match"])

    output = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "HOLDOUT_MEASURED",
        "software_validated": True,
        "prediction_file": str(PREDICTION.relative_to(HERE)),
        "prediction_generated_at": prediction["generated_at"],
        "holdout_measured_after_prediction": True,
        "rows": rows,
        "summary": {
            "points": len(rows),
            "slo_verdict_matches": verdict_matches,
            "slo_verdict_accuracy": round(verdict_matches / len(rows), 3),
            "mape_pct": {
                metric: round(sum(values) / len(values), 3)
                for metric, values in errors.items()
            },
            "max_abs_error_pct": {
                metric: round(max(values), 3)
                for metric, values in errors.items()
            },
        },
        "notes": [
            "The prediction file was frozen before C=16/48 were measured; its timestamp is the audit boundary.",
            "C=64 was observed before the prediction but excluded from the fit and from this holdout.",
            "The power law has three calibration points and two parameters; the heuristic band is not a confidence interval.",
            "All four SLO verdicts matched, but llama.cpp C=16 is only 38.66 ms above the TTFT threshold; do not call it a robust pass boundary.",
        ],
    }
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for row in rows:
        print(
            f"{row['framework']:8s} C={row['concurrency']:2d} "
            f"pred={row['predicted']['predicted_slo_verdict']} "
            f"actual={row['actual']['slo_verdict']} "
            f"err TTFT={row['error_pct']['ttft']:+.1f}% "
            f"TPOT={row['error_pct']['tpot']:+.1f}% "
            f"E2E={row['error_pct']['e2el']:+.1f}%"
        )
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
