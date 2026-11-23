#!/usr/bin/env python3
"""Closed-loop OpenAI-compatible benchmark for chapter 8 framework comparisons.

The same client drives vLLM and llama.cpp server.  It deliberately keeps the
measurement boundary in one place: one request starts when the HTTP body is
sent, TTFT ends at the first non-empty streamed token, and E2E ends after the
final SSE chunk / [DONE] marker.  TPOT is derived from streamed token arrival
times, not from server-specific timings, so the result is comparable across
frameworks.

Example:
  python bench_framework.py --framework vllm --base-url http://127.0.0.1:18001 \
    --model qwen3-4b --concurrency 1,8,32,64 --num-prompts 200 \
    --out-dir results/framework/vllm
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def percentile(values: Iterable[float], p: float) -> float | None:
    data = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def _vram_used_bytes(gpu_index: int) -> int | None:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    pat = re.compile(
        rf"GPU\[{gpu_index}\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)"
    )
    m = pat.search(out)
    return int(m.group(1)) if m else None


class VramSampler(threading.Thread):
    def __init__(self, gpu_index: int, interval_s: float = 0.2):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self.stop_event.is_set():
            used = _vram_used_bytes(self.gpu_index)
            self.samples.append({"t": time.time(), "used_bytes": used})
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        values = [s["used_bytes"] for s in self.samples if s["used_bytes"] is not None]
        if not values:
            return {"samples": 0}
        return {
            "samples": len(values),
            "min_bytes": min(values),
            "max_bytes": max(values),
            "median_bytes": int(statistics.median(values)),
        }


def _metric_values(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name, raw = line.rsplit(None, 1)
            value = float(raw)
        except ValueError:
            continue
        base = name.split("{", 1)[0]
        values[base] = values.get(base, 0.0) + value
    return values


def _metrics_snapshot(base_url: str) -> tuple[dict[str, float], str | None]:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return _metric_values(text), text
    except Exception as exc:  # metrics are auxiliary evidence
        return {}, f"ERROR: {exc!r}"


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    names = {
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:request_success_total",
        "vllm:num_preemptions_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:tokens_evaluated_total",
        "llamacpp:requests_processed_total",
        "llamacpp:prompt_seconds_total",
        "llamacpp:predict_seconds_total",
    }
    return {
        name: round(after.get(name, 0.0) - before.get(name, 0.0), 6)
        for name in sorted(names)
        if name in before or name in after
    }


def _parse_chunk(payload: dict) -> tuple[str, str | None, dict | None, dict | None]:
    text = ""
    finish_reason = None
    usage = None
    timings = None
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0] or {}
        text = choice.get("text") or ""
        finish_reason = choice.get("finish_reason")
    usage = payload.get("usage")
    timings = payload.get("timings")
    return text, finish_reason, usage, timings


def request_once(
    *,
    base_url: str,
    model: str,
    prompt_ids: list[int],
    output_tokens: int,
    framework: str,
    timeout_s: float,
    request_id: str,
) -> dict:
    body: dict = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
    }
    if framework == "vllm":
        body["stream_options"] = {"include_usage": True}
    elif framework == "llamacpp":
        body["cache_prompt"] = False
    payload_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=payload_bytes,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_t = None
    token_times: list[float] = []
    text_parts: list[str] = []
    finish_reason = None
    usage = None
    server_timings = None
    error = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", None)
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text, finish, chunk_usage, timings = _parse_chunk(obj)
                if text:
                    now = time.perf_counter()
                    if first_token_t is None:
                        first_token_t = now
                    token_times.append(now)
                    text_parts.append(text)
                if finish:
                    finish_reason = finish
                if chunk_usage:
                    usage = chunk_usage
                if timings:
                    server_timings = timings
    except Exception as exc:
        error = repr(exc)
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            try:
                error += " body=" + exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass
    ended = time.perf_counter()

    completion_tokens = None
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    n_tokens = int(completion_tokens) if completion_tokens else len(token_times)
    ttft_ms = (first_token_t - started) * 1000.0 if first_token_t is not None else None
    e2el_ms = (ended - started) * 1000.0
    if first_token_t is not None and n_tokens > 1:
        # A server may coalesce several tokens into one SSE chunk.  Dividing by
        # stream chunks would then overstate TPOT, so use the token count from
        # the final usage block and the request-level TTFT/E2E boundary.
        tpot_ms = (ended - first_token_t) * 1000.0 / (n_tokens - 1)
        itls_ms = [
            (token_times[i] - token_times[i - 1]) * 1000.0
            for i in range(1, len(token_times))
        ]
    else:
        tpot_ms = None
        itls_ms = []
    return {
        "request_id": request_id,
        "status": status,
        "error": error,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_ms,
        "e2el_ms": e2el_ms,
        "tpot_ms": tpot_ms,
        "itls_ms": itls_ms,
        "n_stream_chunks": len(token_times),
        "n_completion_tokens": n_tokens,
        "n_prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "text_preview": "".join(text_parts)[:80],
        "server_timings": server_timings,
        "started_epoch_s": time.time() - (ended - started),
        "ended_epoch_s": time.time(),
    }


def _slo_pass(row: dict, slo: dict) -> bool:
    if row.get("error") or row.get("ttft_ms") is None:
        return False
    return (
        row["ttft_ms"] <= slo["ttft_p99_ms"]
        and row.get("tpot_ms") is not None
        and row["tpot_ms"] <= slo["tpot_p99_ms"]
        and row["e2el_ms"] <= slo["e2el_p99_ms"]
    )


def summarize(rows: list[dict], wall_s: float, slo: dict) -> dict:
    ok = [r for r in rows if not r.get("error") and r.get("ttft_ms") is not None]
    failed = len(rows) - len(ok)
    passed = [r for r in rows if _slo_pass(r, slo)]
    itls = [x for r in ok for x in r.get("itls_ms", [])]
    out_tokens = sum(int(r.get("n_completion_tokens") or 0) for r in ok)
    goodput_tokens = sum(
        int(r.get("n_completion_tokens") or 0) for r in passed
    )
    return {
        "requests": len(rows),
        "successful_requests": len(ok),
        "failed_requests": failed,
        "failure_rate": failed / len(rows) if rows else None,
        "slo_passed_requests": len(passed),
        "slo_pass_rate": len(passed) / len(rows) if rows else None,
        "wall_s": round(wall_s, 6),
        "request_throughput_per_s": round(len(ok) / wall_s, 6) if wall_s else None,
        "output_tokens_total": out_tokens,
        "output_tokens_per_s": round(out_tokens / wall_s, 6) if wall_s else None,
        "slo_goodput_requests_per_s": round(len(passed) / wall_s, 6) if wall_s else None,
        "slo_goodput_output_tokens_per_s": round(goodput_tokens / wall_s, 6) if wall_s else None,
        "ttft_ms": {
            "p50": percentile([r["ttft_ms"] for r in ok], 50),
            "p95": percentile([r["ttft_ms"] for r in ok], 95),
            "p99": percentile([r["ttft_ms"] for r in ok], 99),
            "max": max((r["ttft_ms"] for r in ok), default=None),
        },
        "tpot_ms": {
            "p50": percentile([r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None], 50),
            "p95": percentile([r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None], 95),
            "p99": percentile([r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None], 99),
        },
        "e2el_ms": {
            "p50": percentile([r["e2el_ms"] for r in ok], 50),
            "p95": percentile([r["e2el_ms"] for r in ok], 95),
            "p99": percentile([r["e2el_ms"] for r in ok], 99),
        },
        "itl_ms_all": {
            "count": len(itls),
            "p50": percentile(itls, 50),
            "p95": percentile(itls, 95),
            "p99": percentile(itls, 99),
        },
        "completion_tokens_per_request": {
            "min": min((r.get("n_completion_tokens") or 0 for r in ok), default=None),
            "p50": percentile([r.get("n_completion_tokens") or 0 for r in ok], 50),
            "max": max((r.get("n_completion_tokens") or 0 for r in ok), default=None),
        },
        "slo": slo,
        "slo_verdict": (
            "PASS"
            if (
                (percentile([r["ttft_ms"] for r in ok], 99) or math.inf) <= slo["ttft_p99_ms"]
                and (percentile([r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None], 99) or math.inf) <= slo["tpot_p99_ms"]
                and (percentile([r["e2el_ms"] for r in ok], 99) or math.inf) <= slo["e2el_p99_ms"]
                and failed / len(rows) <= slo["failure_rate_max"]
            )
            else "FAIL"
        ),
    }


def make_prompts(num_prompts: int, input_tokens: int, seed: int) -> list[list[int]]:
    import random

    rng = random.Random(seed)
    # Qwen3 vocabulary is 151936.  Stay away from the special-token tail and
    # keep the same IDs for every framework.
    lo, hi = 1000, 150000
    return [[rng.randrange(lo, hi) for _ in range(input_tokens)] for _ in range(num_prompts)]


def run_point(
    *,
    args: argparse.Namespace,
    point: int,
    prompts: list[list[int]],
    out_dir: Path,
) -> dict:
    warmup_n = args.warmup
    for i in range(warmup_n):
        row = request_once(
            base_url=args.base_url,
            model=args.model,
            prompt_ids=prompts[-(i + 1)],
            output_tokens=args.output_len,
            framework=args.framework,
            timeout_s=args.request_timeout_s,
            request_id=f"warmup-{i}",
        )
        if row.get("error"):
            raise RuntimeError(f"warmup failed: {row['error']}")
        print(f"  warmup {i + 1}/{warmup_n}: ttft={row['ttft_ms']:.1f}ms", flush=True)

    barrier = threading.Barrier(point + 1)
    rows: list[dict] = []
    rows_lock = threading.Lock()
    vram = VramSampler(args.gpu_index)
    vram.start()
    before_metrics, before_text = _metrics_snapshot(args.base_url)
    wall_start = time.perf_counter()

    def worker(wid: int) -> None:
        barrier.wait()
        for idx in range(wid, len(prompts), point):
            row = request_once(
                base_url=args.base_url,
                model=args.model,
                prompt_ids=prompts[idx],
                output_tokens=args.output_len,
                framework=args.framework,
                timeout_s=args.request_timeout_s,
                request_id=f"c{point}-r{idx}",
            )
            row["worker"] = wid
            row["request_index"] = idx
            with rows_lock:
                rows.append(row)
            if row.get("error"):
                print(f"    request {idx} failed: {row['error'][:200]}", flush=True)

    with ThreadPoolExecutor(max_workers=point) as pool:
        futures = [pool.submit(worker, wid) for wid in range(point)]
        barrier.wait()
        for fut in futures:
            fut.result()

    wall_s = time.perf_counter() - wall_start
    vram.stop()
    after_metrics, after_text = _metrics_snapshot(args.base_url)
    rows.sort(key=lambda r: r["request_index"])
    summary = summarize(rows, wall_s, {
        "ttft_p99_ms": args.slo_ttft_ms,
        "tpot_p99_ms": args.slo_tpot_ms,
        "e2el_p99_ms": args.slo_e2el_ms,
        "failure_rate_max": args.slo_failure_rate_max,
    })
    summary.update({
        "concurrency": point,
        "num_prompts": len(prompts),
        "vram": vram.summary(),
        "metrics_delta": _metric_delta(before_metrics, after_metrics),
    })

    point_dir = out_dir / f"c{point}"
    point_dir.mkdir(parents=True, exist_ok=True)
    with (point_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (point_dir / "metrics_before.txt").write_text(before_text or "", encoding="utf-8")
    (point_dir / "metrics_after.txt").write_text(after_text or "", encoding="utf-8")
    (point_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True, choices=["vllm", "llamacpp", "generic"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--concurrency", default="1,8,32,64")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--slo-ttft-ms", type=float, default=800.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=100.0)
    parser.add_argument("--slo-e2el-ms", type=float, default=15000.0)
    parser.add_argument("--slo-failure-rate-max", type=float, default=0.01)
    parser.add_argument("--framework-version", default="unspecified")
    parser.add_argument("--server-command", default="")
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if any(p <= 0 for p in points):
        raise SystemExit("concurrency points must be positive")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts(args.num_prompts, args.input_len, args.seed)
    run_started = now_iso()
    print(
        f"[{run_started}] {args.framework} {args.base_url} model={args.model} "
        f"points={points} n={args.num_prompts} isl={args.input_len} osl={args.output_len}",
        flush=True,
    )
    results = []
    for point in points:
        print(f"\n=== concurrency={point} ===", flush=True)
        results.append(run_point(args=args, point=point, prompts=prompts, out_dir=out_dir))

    top = {
        "generated_at": now_iso(),
        "run_started_at": run_started,
        "framework": args.framework,
        "framework_version": args.framework_version,
        "base_url": args.base_url,
        "model": args.model,
        "server_command": args.server_command,
        "note": args.note,
        "contract": {
            "protocol": "OpenAI-compatible /v1/completions streaming SSE",
            "prompt": "random token ids, same IDs for every framework",
            "input_tokens": args.input_len,
            "output_tokens": args.output_len,
            "num_prompts_per_point": args.num_prompts,
            "warmup_per_point": args.warmup,
            "seed": args.seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "request_timeout_s": args.request_timeout_s,
            "gpu_index_for_vram": args.gpu_index,
            "ttft": "first non-empty SSE token minus HTTP send",
            "tpot": "(E2E - TTFT)/(completion_tokens-1); SSE chunk intervals are auxiliary",
            "e2e": "last SSE chunk/[DONE] minus HTTP send",
            "slo": {
                "ttft_p99_ms": args.slo_ttft_ms,
                "tpot_p99_ms": args.slo_tpot_ms,
                "e2el_p99_ms": args.slo_e2el_ms,
                "failure_rate_max": args.slo_failure_rate_max,
            },
        },
        "host": {
            "hostname": os.uname().nodename,
            "python": os.sys.version.split()[0],
        },
        "points": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(top, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
