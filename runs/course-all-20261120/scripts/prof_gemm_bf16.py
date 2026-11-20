#!/usr/bin/env python3
import json
import time

import torch


def main() -> None:
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    n = 2048
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    torch.cuda.nvtx.range_push("WARMUP")
    for _ in range(2):
        torch.mm(a, b)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    times = []
    torch.cuda.nvtx.range_push("MEASURED")
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mm(a, b)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    torch.cuda.nvtx.range_pop()

    flops = 2 * n * n * n
    tflops = [flops / (t / 1000.0) / 1e12 for t in times]
    print(json.dumps({
        "device": name,
        "shape": [n, n, n],
        "dtype": "bfloat16",
        "times_ms": times,
        "tflops": tflops,
        "median_ms": sorted(times)[len(times) // 2],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
