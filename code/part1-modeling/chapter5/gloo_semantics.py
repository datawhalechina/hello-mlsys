"""第 5 章设备口径②：gloo CPU 多进程 All-Reduce 语义验证（Feishu CH5 ①）。

显式声明：本节只验证"集合通信的语义"（求和正确、各 rank 结果一致、
in-place、阻塞/同步、dtype 保持），**不做多卡性能测量**——多卡性能
归 RCCL 探针（rccl_probe.py）。

用法（宿主机）：
  torchrun --nproc_per_node=2 code/part1-modeling/chapter5/gloo_semantics.py
  torchrun --nproc_per_node=4 code/part1-modeling/chapter5/gloo_semantics.py
rank0 把结果写入 results/gloo_semantics_<n>rank.json，由 run_gloo.sh 合并。
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    checks = {}

    # ① 求和正确性：rank i 贡献 (i+1)·arange → 期望 Σ(i+1)·arange
    x = torch.arange(2048, dtype=torch.float64) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expect = (world * (world + 1) / 2) * torch.arange(2048, dtype=torch.float64)
    checks["sum_correct"] = bool(torch.equal(x, expect))

    # ② 各 rank 结果逐元素一致
    gathered = [None] * world
    dist.all_gather_object(gathered, x[:16].tolist())
    checks["consistent_across_ranks"] = all(
        torch.equal(torch.tensor(g), torch.tensor(gathered[0])) for g in gathered)

    # ③ in-place 语义：同一存储地址被原地改写
    a = torch.ones(4, dtype=torch.float32)
    ptr_before = a.data_ptr()
    dist.all_reduce(a)
    checks["in_place"] = bool(a.data_ptr() == ptr_before
                              and torch.equal(a, torch.full((4,), float(world))))

    # ④ dtype 保持（bf16 走 gloo 会报错，属预期：记录支持边界）
    try:
        b = torch.ones(8, dtype=torch.bfloat16)
        dist.all_reduce(b)
        checks["dtype_bf16"] = "supported"
    except Exception as e:  # noqa: BLE001 — 边界记录
        checks["dtype_bf16"] = f"not_supported: {type(e).__name__}: {str(e)[:120]}"

    # ⑤ 阻塞语义：任何 rank 的退出时间 ≥ 最晚 rank 的进入时间
    #    （若 all_reduce 不等待其他 rank，先到的 rank 会在最晚者进入前退出）
    dist.barrier()   # 先对齐，排除 init 抖动
    enter = time.monotonic()
    y = torch.ones(1 << 20, dtype=torch.float32)
    dist.all_reduce(y)
    exit_t = time.monotonic()
    enters, exits = [None] * world, [None] * world
    dist.all_gather_object(enters, enter)
    dist.all_gather_object(exits, exit_t)
    latest_enter = max(enters)
    checks["blocking"] = all(t >= latest_enter - 1e-9 for t in exits)
    checks["blocking_evidence"] = {
        "enter_spread_s": round(max(enters) - min(enters), 6),
        "all_exit_ge_latest_enter": checks["blocking"],
    }

    dist.destroy_process_group()
    if rank != 0:
        return
    payload = {
        "backend": "gloo (CPU, torch.distributed)",
        "world_size": world,
        "declaration": "语义验证，非多卡性能测量（Feishu CH5 ①；多卡性能见 rccl_probe）",
        "torch": torch.__version__,
        "checks": checks,
        "env": common.env_block(),
    }
    p = common.save_json(f"gloo_semantics_{world}rank.json", payload)
    print(f"[world={world}] saved {p}")
    for k, v in checks.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
