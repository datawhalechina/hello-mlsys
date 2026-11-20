#!/usr/bin/env python3
"""Drive real `vllm bench serve` runs and sample vLLM /metrics alongside.

The service is assumed to run in the `ch7-vllm` container on host port 18000.
Results are written both by the benchmark (inside the container, mounted at
`/tmp/ch7-vllm`) and by this script (repo `results/<out-dir>/`).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from common import (INPUT_LEN, OUTPUT_LEN, RESULTS, now_iso, percentile,
                    save_json)

HOST = "http://127.0.0.1:18000"
CONTAINER = "ch7-vllm"
CONTAINER_RESULTS = "/tmp/ch7-vllm"


def _fetch_metrics() -> str:
    with urllib.request.urlopen(f"{HOST}/metrics", timeout=3) as resp:
        return resp.read().decode("utf-8")


def _metric_values(text: str, name: str) -> list[float]:
    pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9eE+.\-]+)$",
                     re.MULTILINE)
    return [float(m.group(1)) for m in pat.finditer(text)]


def _metric_sum(text: str, name: str) -> float:
    return sum(_metric_values(text, name))


def _snapshot() -> dict:
    text = _fetch_metrics()
    return {
        "t": time.time(),
        "running": _metric_sum(text, "vllm:num_requests_running"),
        "waiting": _metric_sum(text, "vllm:num_requests_waiting"),
        "kv_cache_pct": _metric_sum(text, "vllm:kv_cache_usage_perc") * 100,
        "preemptions": _metric_sum(text, "vllm:num_preemptions_total"),
        "prompt_tokens": _metric_sum(text, "vllm:prompt_tokens_total"),
        "generation_tokens": _metric_sum(text, "vllm:generation_tokens_total"),
        "success": _metric_sum(text, "vllm:request_success_total"),
    }


class Sampler(threading.Thread):
    def __init__(self, interval_s: float = 0.25):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: list[dict] = []
        self.error: str | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.samples.append(_snapshot())
            except Exception as exc:  # keep the benchmark result useful
                self.error = repr(exc)
                break
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)


def _summarise_samples(samples: list[dict]) -> dict:
    if not samples:
        return {}
    # Drop the first/last sample so an in-flight scrape does not dominate.
    core = samples[1:-1] if len(samples) > 2 else samples
    return {
        "n_samples": len(samples),
        "running_peak": max(s["running"] for s in core),
        "waiting_peak": max(s["waiting"] for s in core),
        "kv_cache_peak_pct": round(max(s["kv_cache_pct"] for s in core), 2),
        "kv_cache_mean_pct": round(sum(s["kv_cache_pct"] for s in core) / len(core), 2),
        "preemptions_delta": round(core[-1]["preemptions"] - core[0]["preemptions"], 3),
        "prompt_tokens_delta": round(core[-1]["prompt_tokens"] - core[0]["prompt_tokens"], 3),
        "generation_tokens_delta": round(core[-1]["generation_tokens"] - core[0]["generation_tokens"], 3),
    }


def _run_bench(*, mode: str, point: float, num_prompts: int, input_len: int,
               output_len: int, out_dir: str, filename: str,
               num_warmups: int = 2) -> tuple[dict, str]:
    common = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", "http://127.0.0.1:8000",
        "--endpoint", "/v1/completions",
        "--model", "qwen3-4b",
        "--tokenizer", "/models/Qwen3-4B",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--num-warmups", str(num_warmups),
        "--seed", "42",
        "--ignore-eos",
        "--temperature", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--goodput", "ttft:800", "tpot:100", "e2el:15000",
        "--save-result", "--save-detailed",
        "--result-dir", f"/results/{out_dir}",
        "--result-filename", filename,
        "--disable-tqdm",
    ]
    if mode == "closed":
        common += ["--max-concurrency", str(int(point)), "--request-rate", "inf"]
    else:
        common += ["--request-rate", str(point), "--burstiness", "1"]
    cmd = ["docker", "exec", CONTAINER, "bash", "-lc", " ".join(shlex.quote(x) for x in common)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    stdout, stderr = proc.stdout, proc.stderr
    return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr}, filename


def _copy_result(out_dir: str, filename: str) -> dict:
    src = Path(CONTAINER_RESULTS) / out_dir / filename
    if not src.exists():
        raise SystemExit(f"bench result not found: {src}")
    dest_dir = RESULTS / out_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    dest.write_bytes(src.read_bytes())
    return json.loads(dest.read_text(encoding="utf-8"))


def _summary_from_result(raw: dict, metrics: dict) -> dict:
    total = raw.get("completed", 0) + raw.get("failed", 0)
    return {
        "num_prompts": raw.get("num_prompts"),
        "completed": raw.get("completed"),
        "failed": raw.get("failed"),
        "failure_rate": round(raw.get("failed", 0) / total, 6) if total else None,
        "duration_s": round(raw.get("duration", math.nan), 3),
        "request_throughput": round(raw.get("request_throughput", math.nan), 3),
        "request_goodput": round(raw.get("request_goodput", math.nan), 3),
        "output_throughput": round(raw.get("output_throughput", math.nan), 3),
        "total_token_throughput": round(raw.get("total_token_throughput", math.nan), 3),
        "ttft_p50_ms": round(raw.get("p50_ttft_ms", math.nan), 2),
        "ttft_p95_ms": round(raw.get("p95_ttft_ms", math.nan), 2),
        "ttft_p99_ms": round(raw.get("p99_ttft_ms", math.nan), 2),
        "tpot_p50_ms": round(raw.get("p50_tpot_ms", math.nan), 2),
        "tpot_p95_ms": round(raw.get("p95_tpot_ms", math.nan), 2),
        "tpot_p99_ms": round(raw.get("p99_tpot_ms", math.nan), 2),
        "itl_p99_ms": round(raw.get("p99_itl_ms", math.nan), 2),
        "e2el_p50_ms": round(raw.get("p50_e2el_ms", math.nan), 2),
        "e2el_p95_ms": round(raw.get("p95_e2el_ms", math.nan), 2),
        "e2el_p99_ms": round(raw.get("p99_e2el_ms", math.nan), 2),
        "max_concurrent_requests": raw.get("max_concurrent_requests"),
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["closed", "open"], required=True)
    ap.add_argument("--points", default="", help="closed-loop concurrency list")
    ap.add_argument("--rates", default="", help="open-loop req/s list")
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--input-len", type=int, default=INPUT_LEN)
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--num-warmups", type=int, default=2)
    a = ap.parse_args()

    points: list[float]
    if a.mode == "closed":
        points = [float(x) for x in a.points.split(",") if x.strip()]
    else:
        points = [float(x) for x in a.rates.split(",") if x.strip()]
    if not points:
        raise SystemExit("no points supplied")

    rows = []
    for point in points:
        tag = f"{a.prefix}_C{int(point)}" if a.mode == "closed" else f"{a.prefix}_R{point:g}"
        filename = f"{tag}.json"
        print(f"=== {tag} ===", flush=True)
        sampler = Sampler()
        sampler.start()
        started = time.time()
        run_info, _ = _run_bench(mode=a.mode, point=point,
                                 num_prompts=a.num_prompts,
                                 input_len=a.input_len, output_len=a.output_len,
                                 out_dir=a.out_dir, filename=filename,
                                 num_warmups=a.num_warmups)
        elapsed = time.time() - started
        sampler.stop()
        metrics = _summarise_samples(sampler.samples)
        log_path = RESULTS / a.out_dir / f"{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(run_info["stdout"] + "\n---STDERR---\n" + run_info["stderr"],
                            encoding="utf-8")
        try:
            raw = _copy_result(a.out_dir, filename)
            summary = _summary_from_result(raw, metrics)
            summary.update({
                "tag": tag,
                "mode": a.mode,
                "point": point,
                "input_len": a.input_len,
                "output_len": a.output_len,
                "elapsed_s": round(elapsed, 3),
                "returncode": run_info["returncode"],
                "raw_json": str(Path(a.out_dir) / filename),
            })
            rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except Exception as exc:
            rows.append({"tag": tag, "mode": a.mode, "point": point,
                         "error": repr(exc), "returncode": run_info["returncode"],
                         "metrics": metrics})
            print(f"FAILED {tag}: {exc!r}", flush=True)

    payload = {
        "generated_at": now_iso(),
        "mode": a.mode,
        "out_dir": a.out_dir,
        "input_len": a.input_len,
        "output_len": a.output_len,
        "num_prompts": a.num_prompts,
        "points": rows,
    }
    path = save_json(f"{a.out_dir}/summary.json", payload)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
