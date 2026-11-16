"""本机（z-lab-7900）ReBAR 后 P2P 路径法医：绕过 RCCL，直接测 GPU0↔GPU1 裸 D2D copy。

背景（run6 = ReBAR 启用后首个 RCCL run，见 results/rccl_diag_rebar_b.log）：
- ReBAR 前（run3/run5）RCCL 通道 = `via SHM/direct/direct`（host staging，P2P 被拒）；
- ReBAR 后（run6）RCCL 通道 = `via P2P/IPC`（isAllDirectP2p 1），但首个 all_reduce
  仍 60s 看门狗超时 → SIGABRT；dmesg 故障窗口零 IOMMU fault。
本脚本回答：P2P DMA 路径本身是否走通（与 RCCL/cuMem/IPC 无关）？

单进程、单卡对，不占卡不拟合，结果只作法医证据（不入冻结 α–β 网格、
不碰 holdout、不影响 v100x2 口径）：
  ① can_device_access_peer(0,1)/(1,0) + 设备名（排除双句柄同卡）；
  ② 裸 D2D copy 双向：64KB / 1MB / 16MB——**数据完整性校验**（a 随机、
     b 清零、copy 后 b 必须逐元素等于 a；校验不过 = 数据没真走）+
     **双卡各自 synchronize** 后的 wall-clock（synchronize() 只等当前设备，
     首版脚本单设备同步测出 13 TB/s 假象——copy 在源设备队列上、目的设备
     无 pending 操作即返回，量的是 launch 而非搬运）；
  ③ rocm-smi --showp2p 矩阵。

用法（宿主本地）：
  timeout 120 python p2p_forensic_local.py 2>&1 | tee results/p2p_forensic_local.log
"""
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

SIZES_KB = [64, 1024, 16384]
ITERS = 3


def d2d_copy_time(src_dev: int, dst_dev: int, n_bytes: int) -> dict:
    """src→dst 裸 copy：随机源 + 逐元素校验 + 双设备同步后 wall-clock。

    uint8 逐字节计数（避免 dtype 字节数与 n_bytes 换算错位）。
    """
    a = torch.randint(0, 255, (n_bytes,), dtype=torch.uint8,
                      device=f"cuda:{src_dev}")
    b = torch.zeros(n_bytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")
    torch.cuda.set_device(dst_dev)
    b.copy_(a)                                    # warmup（含首次 P2P 映射）
    torch.cuda.synchronize(src_dev)
    torch.cuda.synchronize(dst_dev)
    ts = []
    for _ in range(ITERS):
        b.zero_()
        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        t1 = time.time()
        torch.cuda.set_device(dst_dev)
        b.copy_(a)
        torch.cuda.synchronize(src_dev)
        torch.cuda.synchronize(dst_dev)
        dt = time.time() - t1
        ok = bool(torch.equal(b, a.to(f"cuda:{dst_dev}")))
        if not ok:
            del a, b
            return {"ms": None, "gbps": None, "verified": False,
                    "note": "数据校验失败——copy 未真实搬运"}
        ts.append(dt)
    del a, b
    med = statistics.median(ts)                       # seconds（wall-clock）
    return {"ms": round(med * 1e3, 3), "gbps": round(n_bytes / med / 1e9, 2),
            "verified": True}


def rocm_smi_p2p() -> str:
    try:
        out = subprocess.run(["rocm-smi", "--showp2p"], capture_output=True,
                             text=True, timeout=30).stdout
        return out.strip()
    except Exception as e:
        return f"unavailable: {e}"


def main():
    payload = {
        "what": "ReBAR 后本机 P2P 裸 copy 法医（绕过 RCCL；不入冻结网格、不碰 holdout/拟合）",
        "host": "z-lab-7900 (Ryzen 9 3900X / MSI MEG X570 UNIFY, iommu=pt, ReBAR on)",
        "pair": "GPU0 RX 7900 XTX gfx1100 @ 2f:00.0 <-> GPU1 RX 9070 XT gfx1201 @ 32:00.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "checks": {},
        "d2d": {},
        "rocm_smi_p2p": rocm_smi_p2p(),
    }
    payload["checks"] = {
        "device_count": torch.cuda.device_count(),
        "name_0": torch.cuda.get_device_name(0),
        "name_1": torch.cuda.get_device_name(1),
        "p2p_0_1": bool(torch.cuda.can_device_access_peer(0, 1)),
        "p2p_1_0": bool(torch.cuda.can_device_access_peer(1, 0)),
    }
    print(f"checks: {payload['checks']}", flush=True)

    for kb in SIZES_KB:
        nb = kb * 1024
        payload["d2d"][f"{kb}kb_0to1"] = d2d_copy_time(0, 1, nb)
        print(f"  {kb}KB 0->1: {payload['d2d'][f'{kb}kb_0to1']}", flush=True)
        payload["d2d"][f"{kb}kb_1to0"] = d2d_copy_time(1, 0, nb)
        print(f"  {kb}KB 1->0: {payload['d2d'][f'{kb}kb_1to0']}", flush=True)

    p = Path(__file__).parent / "results" / "p2p_forensic_local.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved {p}", flush=True)
    print("FORENSIC-OK", flush=True)


if __name__ == "__main__":
    main()
