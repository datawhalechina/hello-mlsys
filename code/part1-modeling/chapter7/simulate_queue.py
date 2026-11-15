#!/usr/bin/env python3
"""Pedagogical M/G/1 FCFS model; NOT an LLM serving simulator.

No third-party dependencies. Exponential inter-arrival times and lognormal
service times. Runs are finite and overload rows MUST NOT be read as a
steady-state latency prediction.
"""
import argparse
import csv
import math
import random
from pathlib import Path


def percentile(data: list[float], p: float) -> float:
    if not data:
        return math.nan
    sorted_data = sorted(data)
    pos = (len(sorted_data) - 1) * p / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_data[lo] * (hi - pos) + sorted_data[hi] * (pos - lo) if hi != lo else sorted_data[lo]


def simulate(rate: float, mean_service_s: float, cv: float,
             duration_s: float, warmup_s: float, seed: int):
    rng = random.Random(seed)
    sigma2 = math.log1p(cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean_service_s) - sigma2 / 2
    t = 0.0
    server_free = 0.0
    waits, latencies = [], []
    offered = finished_in_window = 0
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            break
        service_s = rng.lognormvariate(mu, sigma)
        start = max(t, server_free)
        finish = start + service_s
        server_free = finish
        if t >= warmup_s:
            offered += 1
            waits.append(start - t)
            latencies.append(finish - t)
        if warmup_s <= finish < duration_s:
            finished_in_window += 1
    return waits, latencies, offered, finished_in_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rates', default='1,2,3,4,4.5,5,5.2,5.5,6',
                   help='Offered rates (requests/second), comma separated')
    p.add_argument('--mean-service-ms', type=float, default=180.0)
    p.add_argument('--cv', type=float, default=0.65)
    p.add_argument('--duration-s', type=float, default=1800)
    p.add_argument('--warmup-s', type=float, default=180)
    p.add_argument('--replications', type=int, default=3)
    p.add_argument('--slo-p99-ms', type=float, default=1200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='queue_results.csv')
    a = p.parse_args()
    assert 0 < a.warmup_s < a.duration_s
    assert a.mean_service_ms > 0 and a.cv >= 0 and a.replications >= 1
    rates = [float(x) for x in a.rates.split(',')]
    assert rates and all(x > 0 for x in rates)
    fields = ['offered_rps', 'rho', 'stable_analytic', 'n_observed',
              'achieved_rps', 'p50_e2e_ms', 'p95_e2e_ms', 'p99_e2e_ms',
              'p99_queue_ms', 'slo_pass']
    records = []
    mean_s = a.mean_service_ms / 1000
    for i, rate in enumerate(rates):
        waits, lats, n, done = [], [], 0, 0
        for rep in range(a.replications):
            w, l, o, f = simulate(rate, mean_s, a.cv, a.duration_s,
                                  a.warmup_s, a.seed + i * 1009 + rep)
            waits.extend(w)
            lats.extend(l)
            n += o
            done += f
        rho = rate * mean_s
        stable = rho < 1
        p99 = percentile(lats, 99) * 1000
        row = {'offered_rps': f'{rate:.3f}', 'rho': f'{rho:.4f}',
               'stable_analytic': str(stable).lower(), 'n_observed': n,
               'achieved_rps': f'{done / (a.duration_s-a.warmup_s) / a.replications:.4f}',
               'p50_e2e_ms': f'{percentile(lats,50)*1000:.2f}',
               'p95_e2e_ms': f'{percentile(lats,95)*1000:.2f}',
               'p99_e2e_ms': f'{p99:.2f}',
               'p99_queue_ms': f'{percentile(waits,99)*1000:.2f}',
               'slo_pass': str(stable and p99 <= a.slo_p99_ms).lower()}
        records.append(row)
        status = 'unstable; finite-trace only' if not stable else ('SLO PASS' if p99 <= a.slo_p99_ms else 'SLO FAIL')
        print(f'{rate:5.2f} rps | rho={rho:.2f} | P99={p99:8.1f} ms | {status}')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
