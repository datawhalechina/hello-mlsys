"""最小 RCCL 连通性诊断（双卡 gfx1100+gfx1201，P2P=off）。

只回答三个问题，快速失败不占卡：
  ① 首个 all_reduce 是否会 hang（90s 看门狗，短超时）；
  ② RCCL 实际选什么传输（NCCL_DEBUG=INFO 抓 init 日志）；
  ③ 正确性：小消息 all_reduce 求和是否 = world。
不做标定拟合、不走 14 点网格——那是 run_rccl_probe.sh 的职责。
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dev = f"cuda:{rank}"
torch.cuda.set_device(dev)

t0 = time.time()
_timeout_sec = int(os.environ.get("MINIDIAG_TIMEOUT_SEC", "90"))
dist.init_process_group("nccl", timeout=timedelta(seconds=_timeout_sec))
torch.cuda.synchronize()
print(f"[rank{rank}] init_process_group OK in {time.time() - t0:.1f}s", flush=True)

for kb in (1, 256, 1024):
    n = kb * 1024 // 2                    # bf16 = 2B
    x = torch.ones(n, dtype=torch.bfloat16, device=dev)
    t1 = time.time()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = time.time() - t1
    ok = bool(torch.equal(x, torch.full_like(x, float(world))))
    print(f"[rank{rank}] all_reduce {kb}KB: {dt * 1e3:.2f} ms, sum_ok={ok}", flush=True)
    del x

print(f"[rank{rank}] DIAG-OK", flush=True)
dist.destroy_process_group()
