#!/usr/bin/env python3
"""A small, calibrated DES for chapter 7.

What is modelled explicitly
---------------------------
* Open-loop Poisson arrivals or a closed-loop client with C workers.
* vLLM-style continuous batching: waiting -> prefill -> running decode.
* A finite KV pool (24,464 tokens, 16-token pages) with RECOMPUTE preemption.
* Per-request TTFT / TPOT / E2E, queue wait, prefill and decode time.
* Goodput under the chapter SLOs (TTFT<=800 ms, TPOT<=100 ms, E2E<=15 s).

What is deliberately *not* modelled
-----------------------------------
* Kernel-level roofline, prefix-cache hits, chunked-prefill token budget,
  CUDA graph replay, CPU scheduling, network and tokenizer time.
* vLLM's exact victim policy.  We preempt the most recently admitted running
  request, which is the same RECOMPUTE family but not a bit-exact clone.

The service-time constants are fitted from real `vllm bench serve` calibration
points (C=1/2/4/8) and are frozen before the fresh C=12/24/48/96 holdout is
opened.  C=16/32/64 was measured under the earlier v1 log2 model and is burned:
it is never used to select v2.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Iterable

from common import (INPUT_LEN, KV_BLOCK_TOKENS, KV_CACHE_TOKENS, MAX_NUM_SEQS,
                    OUTPUT_LEN, RESULTS, SLO, load_json, now_iso, percentile,
                    save_json)


@dataclass
class Req:
    rid: int
    arrival: float
    input_len: int = INPUT_LEN
    output_len: int = OUTPUT_LEN
    generated: int = 0
    kv_tokens: int = 0
    first_token_time: float | None = None
    completion_time: float | None = None
    admitted_time: float | None = None
    has_first_token: bool = False
    preemptions: int = 0
    recompute_tokens: int = 0

    @property
    def blocks(self) -> int:
        return (self.kv_tokens + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS

    @property
    def queue_wait_s(self) -> float:
        if self.admitted_time is None:
            return math.nan
        return self.admitted_time - self.arrival

    @property
    def e2e_s(self) -> float:
        if self.completion_time is None:
            return math.nan
        return self.completion_time - self.arrival

    @property
    def ttft_s(self) -> float:
        if self.first_token_time is None:
            return math.nan
        return self.first_token_time - self.arrival

    @property
    def tpot_s(self) -> float:
        if self.first_token_time is None or self.completion_time is None:
            return math.nan
        if self.output_len <= 1:
            return math.nan
        return (self.completion_time - self.first_token_time) / (self.output_len - 1)


@dataclass
class ServiceModel:
    prefill_p0_ms: float
    prefill_p1_ms_per_log2: float
    decode_t1_ms: float
    decode_gamma: float
    prefill_mode: str = "log2"
    prefill_p1_ms_per_seq: float = 0.0
    source: str = "default"

    def t_prefill_ms(self, batch: int) -> float:
        if batch <= 0:
            return 0.0
        if self.prefill_mode == "linear":
            return self.prefill_p0_ms + self.prefill_p1_ms_per_seq * batch
        return self.prefill_p0_ms + self.prefill_p1_ms_per_log2 * math.log2(max(batch, 1))

    def t_decode_ms(self, batch: int) -> float:
        if batch <= 0:
            return 0.0
        return self.decode_t1_ms * max(batch, 1) ** self.decode_gamma

    def as_dict(self) -> dict:
        return {
            "prefill_p0_ms": self.prefill_p0_ms,
            "prefill_p1_ms_per_log2": self.prefill_p1_ms_per_log2,
            "prefill_p1_ms_per_seq": self.prefill_p1_ms_per_seq,
            "prefill_mode": self.prefill_mode,
            "decode_t1_ms": self.decode_t1_ms,
            "decode_gamma": self.decode_gamma,
            "source": self.source,
            "formulas": {
                "prefill": ("t_prefill(B) = p0 + p1 * B [ms]" if self.prefill_mode == "linear"
                             else "t_prefill(B) = p0 + p1 * log2(max(B,1)) [ms]"),
                "decode": "t_decode(B) = t1 * max(B,1)^gamma [ms]",
            },
        }


def default_service_model() -> ServiceModel:
    # Conservative initial values; fit_service_model.py replaces these with
    # coefficients from the real calibration runs.
    return ServiceModel(32.0, 2.0, 14.3, 0.06, prefill_mode="log2",
                        source="chapter-default")


def load_service_model(path: str | Path | None) -> ServiceModel:
    if not path:
        return default_service_model()
    p = Path(path)
    if not p.is_absolute():
        p = RESULTS / p
    if not p.exists():
        raise SystemExit(f"service model not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    prefill = raw["prefill"]
    decode = raw["decode"]
    return ServiceModel(
        prefill_p0_ms=float(prefill["p0_ms"]),
        prefill_p1_ms_per_log2=float(prefill.get("p1_ms_per_log2", 0.0)),
        decode_t1_ms=float(decode["t1_ms"]),
        decode_gamma=float(decode["gamma"]),
        prefill_mode=str(prefill.get("mode", "log2")),
        prefill_p1_ms_per_seq=float(prefill.get("p1_ms_per_seq", 0.0)),
        source=str(raw.get("source", str(p))),
    )


class Simulator:
    def __init__(self, model: ServiceModel, *, max_num_seqs: int = MAX_NUM_SEQS,
                 kv_tokens: int = KV_CACHE_TOKENS,
                 block_tokens: int = KV_BLOCK_TOKENS, seed: int = 42):
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.kv_tokens = kv_tokens
        self.block_tokens = block_tokens
        self.max_blocks = kv_tokens // block_tokens
        self.free_blocks = self.max_blocks
        self.rng = random.Random(seed)

        self.now = 0.0
        self.waiting: Deque[Req] = deque()
        self.prefilling: list[Req] = []
        self.running: list[Req] = []
        self.completed: list[Req] = []
        self.arrival_heap: list[tuple[float, int, Req]] = []
        self._arrival_seq = 0
        self.next_rid = 0

        self.preemptions = 0
        self.recompute_tokens = 0
        self.peak_batch = 0
        self.peak_kv_tokens = 0
        self.batch_area = 0.0
        self.busy_time = 0.0
        self.trace: list[dict] = []
        self._last_trace_t = -1.0

    # ---------- arrivals ----------
    def _new_req(self, arrival: float) -> Req:
        req = Req(rid=self.next_rid, arrival=arrival)
        self.next_rid += 1
        return req

    def push_arrival(self, t: float, req: Req) -> None:
        self._arrival_seq += 1
        heapq.heappush(self.arrival_heap, (t, self._arrival_seq, req))

    def _release_arrivals(self) -> bool:
        moved = False
        while self.arrival_heap and self.arrival_heap[0][0] <= self.now + 1e-12:
            _, _, req = heapq.heappop(self.arrival_heap)
            self.waiting.append(req)
            moved = True
        return moved

    # ---------- KV ----------
    def _free_req_kv(self, req: Req) -> None:
        self.free_blocks += req.blocks
        req.kv_tokens = 0

    def _preempt_one(self, exclude: Req | None = None) -> Req | None:
        if not self.running:
            return None
        victim = None
        for candidate in reversed(self.running):
            if candidate is not exclude:
                victim = candidate
                break
        if victim is None:
            victim = self.running[-1]
        self.running.remove(victim)
        self._free_req_kv(victim)
        victim.preemptions += 1
        victim.recompute_tokens += victim.input_len + victim.generated
        self.preemptions += 1
        self.recompute_tokens += victim.input_len + victim.generated
        # RECOMPUTE: keep already delivered tokens, rebuild the KV in prefill.
        victim.admitted_time = None
        self.waiting.appendleft(victim)
        return victim

    def _admit(self) -> None:
        while self.waiting and len(self.running) + len(self.prefilling) < self.max_num_seqs:
            req = self.waiting[0]
            prefill_tokens = req.input_len if not req.has_first_token else req.input_len + req.generated
            # Reserve input + the first token for a fresh request.
            need = prefill_tokens + (0 if req.has_first_token else 1)
            blocks = (need + self.block_tokens - 1) // self.block_tokens
            if blocks > self.free_blocks:
                break
            self.waiting.popleft()
            self.free_blocks -= blocks
            req.kv_tokens = need
            req.admitted_time = self.now
            self.prefilling.append(req)

    # ---------- one engine iteration ----------
    def step(self) -> bool:
        if self._release_arrivals():
            pass
        self._admit()
        if not self.running and not self.prefilling:
            if self.waiting:
                # Cannot admit because KV is exhausted by nothing (should not
                # happen); move time forward defensively.
                self.now += 0.001
                return True
            if self.arrival_heap:
                self.now = max(self.now, self.arrival_heap[0][0])
                return True
            return False

        bp, br = len(self.prefilling), len(self.running)
        dt_ms = self.model.t_prefill_ms(bp) + self.model.t_decode_ms(br)
        dt = max(dt_ms / 1000.0, 1e-6)

        # Ensure every running request can get its next token; preempt if not.
        while True:
            extra = 0
            for req in self.running:
                extra += ((req.kv_tokens + 1 + self.block_tokens - 1)
                          // self.block_tokens - req.blocks)
            if extra <= self.free_blocks:
                break
            if not self._preempt_one():
                break

        self.now += dt
        self.busy_time += dt
        self.batch_area += br * dt

        # Decode one token for each running request.
        for req in list(self.running):
            old_blocks = req.blocks
            req.kv_tokens += 1
            self.free_blocks -= req.blocks - old_blocks
            req.generated += 1
            if req.generated >= req.output_len:
                self.running.remove(req)
                self._free_req_kv(req)
                req.completion_time = self.now
                self.completed.append(req)

        # Finish prefill; fresh requests receive their first output token,
        # recomputed requests resume decode on the next iteration.
        for req in list(self.prefilling):
            self.prefilling.remove(req)
            if not req.has_first_token:
                req.has_first_token = True
                req.generated = 1
                req.first_token_time = self.now
                if req.generated >= req.output_len:
                    self._free_req_kv(req)
                    req.completion_time = self.now
                    self.completed.append(req)
                    continue
            self.running.append(req)

        self.peak_batch = max(self.peak_batch, len(self.running) + len(self.prefilling))
        used = self.max_blocks - self.free_blocks
        self.peak_kv_tokens = max(self.peak_kv_tokens, used * self.block_tokens)

        if self.now - self._last_trace_t >= 0.05:
            self._last_trace_t = self.now
            self.trace.append({
                "t": round(self.now, 4),
                "waiting": len(self.waiting),
                "prefill": len(self.prefilling),
                "running": len(self.running),
                "batch": len(self.running) + len(self.prefilling),
                "kv_tokens": used * self.block_tokens,
                "preemptions": self.preemptions,
            })
        return True

    def run(self, *, total_requests: int, mode: str, concurrency: int = 1,
            rate: float | None = None, max_time_s: float = 600.0,
            on_complete=None) -> None:
        if mode == "closed":
            initial = min(concurrency, total_requests)
            for _ in range(initial):
                self.push_arrival(0.0, self._new_req(0.0))
            remaining = total_requests - initial
            while True:
                before = len(self.completed)
                if not self.step():
                    break
                new_done = len(self.completed) - before
                for _ in range(new_done):
                    if remaining > 0:
                        req = self._new_req(self.now)
                        self.push_arrival(self.now, req)
                        remaining -= 1
                if len(self.completed) >= total_requests:
                    break
                if self.now > max_time_s:
                    break
        elif mode == "open":
            if rate is None or rate <= 0:
                raise ValueError("open-loop mode needs a positive rate")
            t = 0.0
            for _ in range(total_requests):
                t += self.rng.expovariate(rate)
                self.push_arrival(t, self._new_req(t))
            while True:
                if not self.step():
                    break
                if self.now > max_time_s:
                    break
        else:
            raise ValueError(mode)


def summarize(sim: Simulator, *, mode: str, concurrency: int | None,
              rate: float | None, total_requests: int, warmup_s: float = 5.0,
              warmup_frac: float = 0.20) -> dict:
    completed = [r for r in sim.completed if r.completion_time is not None]
    if not completed:
        return {"error": "no completed requests"}
    if mode == "closed":
        warm_n = int(len(completed) * warmup_frac)
        measured = completed[warm_n:]
    else:
        measured = [r for r in completed if r.arrival >= warmup_s]
    if not measured:
        measured = completed

    ttfts = [r.ttft_s * 1000 for r in measured]
    tpots = [r.tpot_s * 1000 for r in measured if not math.isnan(r.tpot_s)]
    e2es = [r.e2e_s * 1000 for r in measured]
    waits = [r.queue_wait_s * 1000 for r in measured]
    prefills = [(r.first_token_time - r.admitted_time) * 1000
                for r in measured
                if r.first_token_time is not None and r.admitted_time is not None]
    decodes = [(r.completion_time - r.first_token_time) * 1000
               for r in measured
               if r.first_token_time is not None and r.completion_time is not None]

    first_arrival = min(r.arrival for r in measured)
    last_done = max(r.completion_time for r in measured)
    duration = max(last_done - first_arrival, 1e-9)
    passed = sum(1 for r in measured
                 if r.ttft_s * 1000 <= SLO["ttft_p99_ms"]
                 and (math.isnan(r.tpot_s) or r.tpot_s * 1000 <= SLO["tpot_p99_ms"])
                 and r.e2e_s * 1000 <= SLO["e2el_p99_ms"])
    mean_e2e = sum(e2es) / len(e2es)
    mean_wait = sum(waits) / len(waits)
    mean_prefill = sum(prefills) / len(prefills) if prefills else math.nan
    mean_decode = sum(decodes) / len(decodes) if decodes else math.nan

    return {
        "mode": mode,
        "concurrency": concurrency,
        "offered_rps": rate,
        "total_requests": total_requests,
        "completed": len(completed),
        "measured": len(measured),
        "duration_s": round(duration, 3),
        "achieved_rps": round(len(measured) / duration, 3),
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p95_ms": round(percentile(ttfts, 95), 2),
        "ttft_p99_ms": round(percentile(ttfts, 99), 2),
        "tpot_p50_ms": round(percentile(tpots, 50), 2),
        "tpot_p95_ms": round(percentile(tpots, 95), 2),
        "tpot_p99_ms": round(percentile(tpots, 99), 2),
        "e2e_p50_ms": round(percentile(e2es, 50), 2),
        "e2e_p95_ms": round(percentile(e2es, 95), 2),
        "e2e_p99_ms": round(percentile(e2es, 99), 2),
        "queue_p50_ms": round(percentile(waits, 50), 2),
        "queue_p95_ms": round(percentile(waits, 95), 2),
        "queue_p99_ms": round(percentile(waits, 99), 2),
        "prefill_mean_ms": round(mean_prefill, 2) if prefills else None,
        "decode_mean_ms": round(mean_decode, 2) if decodes else None,
        "queue_share_pct": round(100 * mean_wait / mean_e2e, 2) if mean_e2e else None,
        "exec_share_pct": round(100 * (1 - mean_wait / mean_e2e), 2) if mean_e2e else None,
        "slo_pass_rate": round(passed / len(measured), 4),
        "goodput_rps": round(passed / duration, 3),
        "preemptions": sim.preemptions,
        "recompute_tokens": sim.recompute_tokens,
        "mean_batch": round(sim.batch_area / sim.busy_time, 2) if sim.busy_time else 0,
        "peak_batch": sim.peak_batch,
        "kv_peak_tokens": sim.peak_kv_tokens,
        "kv_peak_pct": round(100 * sim.peak_kv_tokens / sim.kv_tokens, 2),
        "slo": SLO,
    }


def request_rows(sim: Simulator) -> list[dict]:
    rows = []
    for r in sim.completed:
        rows.append({
            "rid": r.rid,
            "arrival": round(r.arrival, 4),
            "admitted": round(r.admitted_time, 4) if r.admitted_time is not None else None,
            "first_token": round(r.first_token_time, 4) if r.first_token_time else None,
            "completion": round(r.completion_time, 4) if r.completion_time else None,
            "queue_ms": round(r.queue_wait_s * 1000, 2),
            "ttft_ms": round(r.ttft_s * 1000, 2),
            "tpot_ms": round(r.tpot_s * 1000, 2) if not math.isnan(r.tpot_s) else None,
            "e2e_ms": round(r.e2e_s * 1000, 2),
            "preemptions": r.preemptions,
            "recompute_tokens": r.recompute_tokens,
        })
    return rows


def run_closed(model: ServiceModel, concurrency: int, n: int, seed: int) -> tuple[dict, list[dict], list[dict]]:
    sim = Simulator(model, seed=seed)
    sim.run(total_requests=n, mode="closed", concurrency=concurrency)
    return (summarize(sim, mode="closed", concurrency=concurrency, rate=None,
                      total_requests=n),
            request_rows(sim), sim.trace)


def run_open(model: ServiceModel, rate: float, n: int, seed: int,
             max_time_s: float = 300.0) -> tuple[dict, list[dict], list[dict]]:
    sim = Simulator(model, seed=seed)
    sim.run(total_requests=n, mode="open", rate=rate, max_time_s=max_time_s)
    return (summarize(sim, mode="open", concurrency=None, rate=rate,
                      total_requests=n),
            request_rows(sim), sim.trace)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--rates", default="1,2,4,6,8,10,12,16,20,24")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--service-model", default="service_model.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--trace-run", default=None,
                    help="write a per-request timeline for the animation (open-loop rate)")
    ap.add_argument("--trace-requests", type=int, default=300)
    ap.add_argument("--trace-max-time-s", type=float, default=90.0)
    a = ap.parse_args()

    model = load_service_model(a.service_model)
    if a.mode == "closed":
        points = [int(x) for x in a.concurrency.split(",") if x.strip()]
        rows, details = [], []
        for c in points:
            summary, reqs, trace = run_closed(model, c, a.num_requests, a.seed)
            rows.append(summary)
            details.append({"concurrency": c, "summary": summary,
                            "requests": reqs, "trace": trace})
            print(f"C={c:>3}  achieved={summary['achieved_rps']:.2f} rps  "
                  f"TTFT p99={summary['ttft_p99_ms']:.0f} ms  "
                  f"TPOT p99={summary['tpot_p99_ms']:.0f} ms  "
                  f"SLO={summary['slo_pass_rate']*100:.1f}%  "
                  f"preempt={summary['preemptions']}")
        payload = {
            "generated_at": now_iso(),
            "kind": "des_closed_loop",
            "service_model": model.as_dict(),
            "config": {"concurrency": points, "num_requests": a.num_requests,
                       "seed": a.seed, "input_len": INPUT_LEN,
                       "output_len": OUTPUT_LEN, "max_num_seqs": MAX_NUM_SEQS,
                       "kv_cache_tokens": KV_CACHE_TOKENS,
                       "kv_block_tokens": KV_BLOCK_TOKENS},
            "points": rows,
            "details": details,
        }
        out = a.output or "des_closed.json"
    else:
        rates = [float(x) for x in a.rates.split(",") if x.strip()]
        rows, details = [], []
        for rate in rates:
            summary, reqs, trace = run_open(model, rate, a.num_requests, a.seed)
            rows.append(summary)
            details.append({"offered_rps": rate, "summary": summary,
                            "requests": reqs, "trace": trace})
            print(f"lambda={rate:5.1f}  achieved={summary['achieved_rps']:.2f} rps  "
                  f"TTFT p99={summary['ttft_p99_ms']:.0f} ms  "
                  f"E2E p99={summary['e2e_p99_ms']:.0f} ms  "
                  f"SLO={summary['slo_pass_rate']*100:.1f}%  "
                  f"queue={summary['queue_share_pct']:.1f}%  "
                  f"preempt={summary['preemptions']}")
        payload = {
            "generated_at": now_iso(),
            "kind": "des_open_loop",
            "service_model": model.as_dict(),
            "config": {"rates": rates, "num_requests": a.num_requests,
                       "seed": a.seed, "input_len": INPUT_LEN,
                       "output_len": OUTPUT_LEN, "max_num_seqs": MAX_NUM_SEQS,
                       "kv_cache_tokens": KV_CACHE_TOKENS,
                       "kv_block_tokens": KV_BLOCK_TOKENS},
            "points": rows,
            "details": details,
        }
        out = a.output or "des_open.json"

    path = save_json(out, payload)
    print(f"Wrote {path}")

    if a.trace_run:
        rate = float(a.trace_run)
        summary, reqs, trace = run_open(model, rate, a.trace_requests, a.seed,
                                        max_time_s=a.trace_max_time_s)
        save_json("des_animation_trace.json", {
            "generated_at": now_iso(),
            "kind": "des_animation_trace",
            "rate": rate,
            "service_model": model.as_dict(),
            "summary": summary,
            "requests": reqs,
            "trace": trace,
        })
        print(f"Animation trace: {len(reqs)} requests, {len(trace)} steps")


if __name__ == "__main__":
    main()
