"""第 5 章设备口径①：PCIe 1-hop H2D/D2H 实测（torch pinned 传输）。

宿主无 rocm-bandwidth-test（experiment-draft.yaml not_covered 已声明），
PCIe 基线由 torch pinned-memory 传输承担：
  - 主卡 9070 XT（HSA_VISIBLE_DEVICES=1）与辅助卡 7900 XTX 各跑一遍；
  - 64KB 延迟端 + 1MB–1GB × 6 点（×4 跨距）；
  - 每点 5 warmup + 30 次计时取中位数（CUDA event 计时）；
  - 顺带记录 pageable（非 pinned）对照组，展示 pin 的意义（不进入拟合）。

产出 results/pcie_probe_<card>.json。
"""
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import torch  # noqa: E402


LATENCY_KB = 64
SIZE_KB = [2 ** k for k in range(10, 21, 2)]   # 1MB .. 1GB, ×4, 6 点（KB 口径：2^10..2^20）
WARMUP, ITERS = 5, 30


def pinned_bw(device: str, nbytes: int, kind: str, iters: int = ITERS,
              warmup: int = WARMUP) -> float:
    """返回 GB/s：host pinned → device（h2d）或反向（d2h）。"""
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
    src, dst = (host, dev) if kind == "h2d" else (dev, host)
    for _ in range(warmup):
        src.copy_(dst, non_blocking=(kind == "h2d"))
    torch.cuda.synchronize(device)
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    times_ms = []
    for _ in range(iters):
        ev0.record()
        src.copy_(dst, non_blocking=(kind == "h2d"))
        ev1.record()
        torch.cuda.synchronize(device)
        times_ms.append(ev0.elapsed_time(ev1))
    t_med_s = statistics.median(times_ms) * 1e-3
    return nbytes / t_med_s / 1e9


def pageable_bw(device: str, nbytes: int, kind: str, iters: int = 10) -> float:
    """pageable 对照（非 pinned），仅记录不拟合。"""
    host = torch.empty(nbytes, dtype=torch.uint8)
    dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
    src, dst = (host, dev) if kind == "h2d" else (dev, host)
    for _ in range(3):
        src.copy_(dst)
    torch.cuda.synchronize(device)
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    times_ms = []
    for _ in range(iters):
        ev0.record()
        src.copy_(dst)
        ev1.record()
        torch.cuda.synchronize(device)
        times_ms.append(ev0.elapsed_time(ev1))
    return nbytes / (statistics.median(times_ms) * 1e-3) / 1e9


def probe(card: str, device: str) -> dict:
    out = {"card": card, "device": device, "rows": []}
    for kb in [LATENCY_KB] + SIZE_KB:
        nb = kb * 1024
        row = {
            "msg_kb": kb,
            "h2d_gbps": round(pinned_bw(device, nb, "h2d"), 2),
            "d2h_gbps": round(pinned_bw(device, nb, "d2h"), 2),
            "h2d_ms": None, "d2h_ms": None,
        }
        if kb == LATENCY_KB:   # 延迟端：同时记绝对时间
            for kind, key in (("h2d", "h2d_ms"), ("d2h", "d2h_ms")):
                host = torch.empty(nb, dtype=torch.uint8, pin_memory=True)
                dev = torch.empty(nb, dtype=torch.uint8, device=device)
                src, dst = (host, dev) if kind == "h2d" else (dev, host)
                for _ in range(WARMUP):
                    src.copy_(dst)
                torch.cuda.synchronize(device)
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                ts = []
                for _ in range(ITERS):
                    ev0.record()
                    src.copy_(dst)
                    ev1.record()
                    torch.cuda.synchronize(device)
                    ts.append(ev0.elapsed_time(ev1))
                row[key] = round(statistics.median(ts), 3)
        out["rows"].append(row)
    # pageable 对照（只取 64MB 一点，避免整表慢）
    nb = 64 * 1024 * 1024
    out["pageable_64mb_gbps"] = {
        "h2d": round(pageable_bw(device, nb, "h2d"), 2),
        "d2h": round(pageable_bw(device, nb, "d2h"), 2),
    }
    return out


def main():
    # 双卡都在，但每张卡单独进程跑，避免互相抢 PCIe（主卡先）
    idx = int(os.environ.get("HSA_VISIBLE_DEVICES", "1").split(",")[0])
    card = "9070xt" if idx == 1 else "7900xtx"
    name = torch.cuda.get_device_name(0)
    res = probe(card, "cuda:0")
    res["device_name"] = name
    res["env"] = common.env_block()
    p = common.save_json(f"pcie_probe_{card}.json", res)
    print(f"saved {p}")
    for r in res["rows"]:
        extra = f"  t={r['h2d_ms']}ms/{r['d2h_ms']}ms" if r["msg_kb"] == LATENCY_KB else ""
        print(f"  {r['msg_kb']:>7} KB  h2d={r['h2d_gbps']:>6} GB/s  "
              f"d2h={r['d2h_gbps']:>6} GB/s{extra}")
    print(f"  pageable 64MB: {res['pageable_64mb_gbps']}")


if __name__ == "__main__":
    main()
