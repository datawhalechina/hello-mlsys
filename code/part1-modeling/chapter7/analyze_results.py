#!/usr/bin/env python3
"""Join DES predictions, real vLLM runs, /metrics and build the chapter tables."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common import (KV_CACHE_TOKENS, RESULTS, env_block, load_json, now_iso,
                    percentile, save_json)


def _num(x, default=math.nan):
    return default if x is None else float(x)


def real_from_raw(concurrency: int, raw: dict, source: str) -> dict:
    return {
        "concurrency": concurrency,
        "source": source,
        "num_prompts": raw["num_prompts"],
        "completed": raw["completed"],
        "failed": raw["failed"],
        "failure_rate": raw["failed"] / max(raw["completed"] + raw["failed"], 1),
        "duration_s": raw["duration"],
        "request_throughput": raw["request_throughput"],
        "request_goodput": raw["request_goodput"],
        "output_throughput": raw["output_throughput"],
        "ttft_p50_ms": raw["p50_ttft_ms"],
        "ttft_p95_ms": raw["p95_ttft_ms"],
        "ttft_p99_ms": raw["p99_ttft_ms"],
        "tpot_p50_ms": raw["p50_tpot_ms"],
        "tpot_p95_ms": raw["p95_tpot_ms"],
        "tpot_p99_ms": raw["p99_tpot_ms"],
        "itl_p99_ms": raw["p99_itl_ms"],
        "e2el_p50_ms": raw["p50_e2el_ms"],
        "e2el_p95_ms": raw["p95_e2el_ms"],
        "e2el_p99_ms": raw["p99_e2el_ms"],
        "max_concurrent_requests": raw.get("max_concurrent_requests"),
        "metrics": None,
    }


def real_from_summary(point: dict) -> dict:
    return {
        "concurrency": int(point["point"]),
        "source": point["tag"],
        "num_prompts": point["num_prompts"],
        "completed": point["completed"],
        "failed": point["failed"],
        "failure_rate": point["failure_rate"],
        "duration_s": point["duration_s"],
        "request_throughput": point["request_throughput"],
        "request_goodput": point["request_goodput"],
        "output_throughput": point["output_throughput"],
        "ttft_p50_ms": point["ttft_p50_ms"],
        "ttft_p95_ms": point["ttft_p95_ms"],
        "ttft_p99_ms": point["ttft_p99_ms"],
        "tpot_p50_ms": point["tpot_p50_ms"],
        "tpot_p95_ms": point["tpot_p95_ms"],
        "tpot_p99_ms": point["tpot_p99_ms"],
        "itl_p99_ms": point["itl_p99_ms"],
        "e2el_p50_ms": point["e2el_p50_ms"],
        "e2el_p95_ms": point["e2el_p95_ms"],
        "e2el_p99_ms": point["e2el_p99_ms"],
        "max_concurrent_requests": point.get("max_concurrent_requests"),
        "metrics": point.get("metrics"),
    }


def des_point(point: dict) -> dict:
    return {
        "concurrency": int(point["concurrency"]),
        "ttft_p99_ms": point["ttft_p99_ms"],
        "tpot_p99_ms": point["tpot_p99_ms"],
        "e2e_p99_ms": point["e2e_p99_ms"],
        "achieved_rps": point["achieved_rps"],
        "goodput_rps": point["goodput_rps"],
        "slo_pass_rate": point["slo_pass_rate"],
        "queue_share_pct": point["queue_share_pct"],
        "preemptions": point["preemptions"],
        "mean_batch": point["mean_batch"],
        "kv_peak_pct": point["kv_peak_pct"],
    }


def _residual(real: dict, pred: dict) -> dict:
    out = {}
    for key in ("ttft_p99_ms", "tpot_p99_ms", "e2el_p99_ms"):
        pkey = "e2e_p99_ms" if key == "e2el_p99_ms" else key
        r, p = real[key], pred[pkey]
        out[key] = {
            "real": r,
            "pred": p,
            "abs_ms": round(r - p, 2),
            "rel_pct": round(100 * (r - p) / p, 2) if p else None,
        }
    return out


def main() -> None:
    model = load_json("service_model.json")
    real: dict[int, dict] = {}
    for c in (1, 2, 4, 8):
        raw = load_json(f"calibration/calibration_c{c}.json")
        real[c] = real_from_raw(c, raw, "calibration")
    for c in (16, 32, 64):
        raw = load_json(f"holdout/holdout_C{c}.json")
        real[c] = real_from_raw(c, raw, "holdout_v1_burned")
    for p in load_json("holdout_v2/summary.json")["points"]:
        real[int(p["point"])] = real_from_summary(p)

    des_all = {p["concurrency"]: des_point(p)
               for p in load_json("des_closed_v2.json")["points"]}
    des_v1 = {p["concurrency"]: des_point(p)
              for p in load_json("des_prediction_v1_holdout16_32_64.json")["points"]}
    des_v2_holdout = {p["concurrency"]: des_point(p)
                      for p in load_json("des_prediction_holdout_v2.json")["points"]}

    sweep = []
    for c in sorted(real):
        r = dict(real[c])
        r["des_v2"] = des_all.get(c)
        r["residual_v2"] = _residual(r, des_all[c]) if c in des_all else None
        sweep.append(r)

    holdout = []
    for c in (12, 24, 48, 96):
        r = dict(real[c])
        pred = des_v2_holdout[c]
        holdout.append({"concurrency": c, "real": r, "pred": pred,
                        "residual": _residual(r, pred)})

    burned = []
    for c in (16, 32, 64):
        r = dict(real[c])
        burned.append({
            "concurrency": c,
            "real": r,
            "v1_log2_pred": des_v1[c],
            "v1_residual": _residual(r, des_v1[c]),
            "v2_posthoc_pred": des_all[c],
            "v2_posthoc_residual": _residual(r, des_all[c]),
        })

    # Business knee: first tested C violating any P99 SLO or 1% failure.
    first_fail = None
    last_pass = None
    for r in sweep:
        fails = []
        if r["ttft_p99_ms"] > 800:
            fails.append("ttft_p99")
        if r["tpot_p99_ms"] > 100:
            fails.append("tpot_p99")
        if r["e2el_p99_ms"] > 15000:
            fails.append("e2el_p99")
        if r["failure_rate"] > 0.01:
            fails.append("failure_rate")
        if fails:
            first_fail = {"concurrency": r["concurrency"], "fails": fails,
                          "ttft_p99_ms": r["ttft_p99_ms"],
                          "tpot_p99_ms": r["tpot_p99_ms"],
                          "e2el_p99_ms": r["e2el_p99_ms"],
                          "goodput_rps": r["request_goodput"]}
            break
        last_pass = {"concurrency": r["concurrency"],
                     "request_throughput": r["request_throughput"],
                     "request_goodput": r["request_goodput"],
                     "ttft_p99_ms": r["ttft_p99_ms"],
                     "tpot_p99_ms": r["tpot_p99_ms"],
                     "e2el_p99_ms": r["e2el_p99_ms"]}

    max_tp = max(sweep, key=lambda r: r["request_throughput"])
    max_goodput = max(sweep, key=lambda r: r["request_goodput"])
    safe_c = math.floor(0.8 * last_pass["concurrency"]) if last_pass else None
    knee = {
        "first_fail": first_fail,
        "last_pass": last_pass,
        "safe_concurrency_80pct": safe_c,
        "safe_request_rate_80pct": (round(0.8 * last_pass["request_throughput"], 3)
                                    if last_pass else None),
        "peak_request_throughput": {
            "concurrency": max_tp["concurrency"],
            "request_throughput": max_tp["request_throughput"],
            "goodput": max_tp["request_goodput"],
        },
        "peak_goodput": {
            "concurrency": max_goodput["concurrency"],
            "goodput": max_goodput["request_goodput"],
            "request_throughput": max_goodput["request_throughput"],
        },
        "slo": {"ttft_p99_ms": 800, "tpot_p99_ms": 100,
                "e2el_p99_ms": 15000, "failure_rate_max": 0.01},
    }

    # Open-loop queue share.  The low-load TTFT is the prefill + fixed
    # overhead floor; the extra TTFT at high load is the server-side queue.
    # This is a tail estimate, not a per-request trace event.
    open_real = []
    open_path = (RESULTS / "open_loop" / "summary_all.json")
    open_payload = (load_json("open_loop/summary_all.json") if open_path.exists()
                    else load_json("open_loop/summary.json"))
    valid_open = [p for p in open_payload["points"] if "error" not in p]
    base_ttft_p50 = min(p["ttft_p50_ms"] for p in valid_open)
    base_ttft_p99 = min(p["ttft_p99_ms"] for p in valid_open)
    for p in open_payload["points"]:
        if "error" in p:
            open_real.append(p)
            continue
        m = p.get("metrics") or {}
        q50 = max(p["ttft_p50_ms"] - base_ttft_p50, 0.0)
        q99 = max(p["ttft_p99_ms"] - base_ttft_p99, 0.0)
        open_real.append({
            "rate": p["point"],
            "request_throughput": p["request_throughput"],
            "request_goodput": p["request_goodput"],
            "ttft_p50_ms": p["ttft_p50_ms"],
            "ttft_p99_ms": p["ttft_p99_ms"],
            "tpot_p99_ms": p["tpot_p99_ms"],
            "e2el_p50_ms": p["e2el_p50_ms"],
            "e2el_p99_ms": p["e2el_p99_ms"],
            "failed": p["failed"],
            "failure_rate": p["failure_rate"],
            "running_peak": m.get("running_peak"),
            "running_peak_est": p.get("running_peak_est"),
            "waiting_peak": m.get("waiting_peak"),
            "kv_cache_peak_pct": m.get("kv_cache_peak_pct"),
            "preemptions_delta": m.get("preemptions_delta"),
            "baseline_ttft_p50_ms": round(base_ttft_p50, 2),
            "baseline_ttft_p99_ms": round(base_ttft_p99, 2),
            "queue_share_p50_est_pct": round(100 * q50 / p["e2el_p50_ms"], 2),
            "queue_share_p99_est_pct": round(100 * q99 / p["e2el_p99_ms"], 2),
        })

    des_open_path = RESULTS / "des_open.json"
    des_open = []
    if des_open_path.exists():
        des_open = load_json("des_open.json")["points"]

    pressure = []
    for p in load_json("pressure/summary.json")["points"]:
        pressure.append({
            "concurrency": int(p["point"]),
            "output_len": p["output_len"],
            "request_throughput": p["request_throughput"],
            "request_goodput": p["request_goodput"],
            "ttft_p99_ms": p["ttft_p99_ms"],
            "tpot_p99_ms": p["tpot_p99_ms"],
            "e2el_p99_ms": p["e2el_p99_ms"],
            "preemptions": p["metrics"].get("preemptions_delta"),
            "kv_cache_peak_pct": p["metrics"].get("kv_cache_peak_pct"),
            "waiting_peak": p["metrics"].get("waiting_peak"),
            "running_peak": p["metrics"].get("running_peak"),
        })

    def _mape(rows: list[dict], metric: str) -> float:
        vals = [abs(r["residual"][metric]["rel_pct"])
                for r in rows if r["residual"][metric]["rel_pct"] is not None]
        return round(sum(vals) / len(vals), 2) if vals else math.nan

    payload = {
        "generated_at": now_iso(),
        "environment": env_block(),
        "service_model": model,
        "kv_cache_tokens": KV_CACHE_TOKENS,
        "closed_sweep": sweep,
        "holdout_v2": holdout,
        "burned_v1": burned,
        "holdout_v2_mape_pct": {
            "ttft_p99": _mape(holdout, "ttft_p99_ms"),
            "tpot_p99": _mape(holdout, "tpot_p99_ms"),
            "e2el_p99": _mape(holdout, "e2el_p99_ms"),
        },
        "knee": knee,
        "open_loop_real": open_real,
        "open_loop_des": des_open,
        "pressure": pressure,
    }
    path = save_json("analysis.json", payload)
    print(f"Wrote {path}")
    print("knee:", json.dumps(knee, ensure_ascii=False))
    print("holdout MAPE:", payload["holdout_v2_mape_pct"])
    print("holdout rows:")
    for r in holdout:
        res = r["residual"]
        print(f"  C={r['concurrency']:>3}  TTFT real={res['ttft_p99_ms']['real']:.0f} "
              f"pred={res['ttft_p99_ms']['pred']:.0f} ({res['ttft_p99_ms']['rel_pct']:+.1f}%)  "
              f"TPOT {res['tpot_p99_ms']['real']:.1f}/{res['tpot_p99_ms']['pred']:.1f}  "
              f"E2E {res['e2el_p99_ms']['real']:.0f}/{res['e2el_p99_ms']['pred']:.0f}")
    print("open loop:")
    for r in open_real:
        if "rate" in r:
            print(f"  lambda={r['rate']:>4}  achieved={r['request_throughput']:.2f}  "
                  f"goodput={r['request_goodput']:.2f}  TTFTp99={r['ttft_p99_ms']:.0f}  "
                  f"queue%={r['queue_share_p99_est_pct']:.1f}")


if __name__ == "__main__":
    main()
