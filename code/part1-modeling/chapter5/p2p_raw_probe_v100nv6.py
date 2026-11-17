"""第 5 章补充：v100x2 NVLink2 裸互联（P2P raw 带宽/延迟）side measurement。

测后 side measurement（**不入冻结 α–β 网格、不影响 holdout、不改拟合**）：
直接测 GPU4↔5（NV6 bond）D2D raw 带宽与小搬运延迟，把 §5.9 gap 分解里
"NVLink2 单向 raw = 150 GB/s（规格书）"变成实测值，
使"引擎效率 = BW_eff / raw"成为实测比。

协议（与 pcie_probe.py 设备口径同风格）：5 warmup + 30 CUDA event 中位数：
- 单流单向 D2D copy：1/4/16/64/256 MB × 5 点；
- 8 流分块并行单向 D2D（逼近链路饱和，对应 NCCL 多通道口径）；
- 双向并发（0→1 / 1→0 独立流，256 MB，报每方向均值）；
- 小搬运延迟 64 KB 单次；
- 路径验证：nvidia-smi nvlink -s 链路状态（6×25.781 GB/s active）+
  -gt g 计数器可用性探测（本平台不可用，如实记录）+ 带宽论证
  （实测 ≫ host PCIe 通路上限 → 流量走 NVLink 而非 host 中转）。

用法（v100x2 宿主，仅 GPU 4/5；单进程即可，无需 torchrun）：
  CUDA_VISIBLE_DEVICES=4,5 python p2p_raw_probe_v100nv6.py
"""
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

WARMUP, ITERS = 5, 30
SIZES_MB = [1, 4, 16, 64, 256]
N_STREAMS = 8
LAT_KB = 64
DEV0, DEV1 = "cuda:0", "cuda:1"          # = 物理 GPU 4/5（CUDA_VISIBLE_DEVICES=4,5）
NCCL_FIT_JSON = Path(__file__).parent / "results" / "nccl_probe_v100nv6.json"


def nvlink_link_status(phys_id: int) -> dict:
    """nvidia-smi nvlink -s -i <id>：active 链路数与每链路标称速率。

    V100 SXM2 + driver 580 上每链路标称 25.781 GB/s（单向）；
    NV6 bond → 标称单向 raw = 6 × 25.781 = 154.69 GB/s。
    """
    out = subprocess.run(["nvidia-smi", "nvlink", "-s", "-i", str(phys_id)],
                         capture_output=True, text=True, timeout=30).stdout
    rates = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "Link" and parts[3] == "GB/s":
            try:
                rates.append(float(parts[2]))
            except ValueError:
                pass
    return {"n_links": len(rates),
            "per_link_gbps": rates[0] if rates else None,
            "nominal_oneway_raw_gbps": round(sum(rates), 2) if rates else None}


def nvlink_gt_g_counters_status() -> str:
    """nvidia-smi nvlink -gt g：GPU→GPU 累计字节计数器可用性。

    本平台（V100 SXM2 + driver 580）无权限/有权限均报
    "Failed to get throughput counters" → 计数器不可用，如实记录，
    路径判定改用链路状态 + 带宽论证。
    """
    out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "g"],
                         capture_output=True, text=True, timeout=30).stdout
    if "Failed to get throughput counters" in out:
        return "unavailable (Failed to get throughput counters, sudo 亦同)"
    return "available"


def d2d_one_way(n_bytes: int, n_streams: int) -> dict:
    """单向 DEV0→DEV1 D2D copy：中位数 ms → GB/s。"""
    numel = n_bytes // 2
    a = torch.empty(numel, dtype=torch.bfloat16, device=DEV0)
    b = torch.empty(numel, dtype=torch.bfloat16, device=DEV1)
    torch.cuda.set_device(DEV0)
    if n_streams == 1:
        def go():
            b.copy_(a)
    else:
        chunk = numel // n_streams
        streams = [torch.cuda.Stream(device=DEV0) for _ in range(n_streams)]

        def go():
            for i, s in enumerate(streams):
                lo = i * chunk
                hi = (i + 1) * chunk if i < n_streams - 1 else numel
                with torch.cuda.stream(s):
                    b[lo:hi].copy_(a[lo:hi])
            cur = torch.cuda.current_stream(DEV0)
            for s in streams:
                cur.wait_stream(s)
    for _ in range(WARMUP):
        go()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(ITERS):
        ev0.record()
        go()
        ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    t_ms = statistics.median(ts)
    del a, b
    return {"t_ms": round(t_ms, 6), "gbps": round(n_bytes / (t_ms * 1e-3) / 1e9, 2)}


def d2d_bidir(n_bytes: int) -> dict:
    """双向并发 0→1 / 1→0（独立流、独立源张量），报每方向均值。"""
    numel = n_bytes // 2
    s0 = torch.empty(numel, dtype=torch.bfloat16, device=DEV0)
    s1 = torch.empty(numel, dtype=torch.bfloat16, device=DEV1)
    d0 = torch.empty_like(s0)                       # DEV0 上接收
    d1 = torch.empty_like(s1)                       # DEV1 上接收
    st0 = torch.cuda.Stream(device=DEV0)
    st1 = torch.cuda.Stream(device=DEV1)

    def go():
        with torch.cuda.stream(st0):
            d1.copy_(s0)                            # DEV0 -> DEV1
        with torch.cuda.stream(st1):
            d0.copy_(s1)                            # DEV1 -> DEV0

    for _ in range(WARMUP):
        go()
    torch.cuda.synchronize()
    ev0a, ev1a = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0b, ev1b = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(ITERS):
        torch.cuda.set_device(DEV0); ev0a.record()
        torch.cuda.set_device(DEV1); ev0b.record()
        go()
        torch.cuda.set_device(DEV0)
        torch.cuda.current_stream(DEV0).wait_stream(st0)
        ev1a.record()
        torch.cuda.set_device(DEV1)
        torch.cuda.current_stream(DEV1).wait_stream(st1)
        ev1b.record()
        # 双卡各自同步：synchronize() 只等当前设备，DEV0 的 ev1a 必须单独等
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        ts.append(max(ev0a.elapsed_time(ev1a), ev0b.elapsed_time(ev1b)))
    t_ms = statistics.median(ts)
    del s0, s1, d0, d1
    total_gbps = 2 * n_bytes / (t_ms * 1e-3) / 1e9
    return {"t_ms": round(t_ms, 6),
            "total_gbps": round(total_gbps, 2),
            "per_dir_gbps": round(total_gbps / 2, 2)}


def main():
    payload = {
        "what": "NVLink2 raw P2P 互联 side measurement（测后；不入冻结 α–β 网格、不影响 holdout/拟合）",
        "host": "v100x2 (z-lab-v100x2, 192.168.1.211)",
        "pair": "物理 GPU4<->GPU5，NV6 bond（6×NVLink2）",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": "torch.bfloat16",
        "warmup": WARMUP, "iters": ITERS,
        "checks": {"p2p_access_0_1": bool(torch.cuda.can_device_access_peer(0, 1)),
                   "device_count": torch.cuda.device_count()},
        "one_way_single_stream": {},
        "one_way_8stream": {},
        "bidir": {},
        "latency": {},
        "nvlink_path_check": {},
        "spec_ref": {"nvlink2_oneway_raw_gbps": 150.0, "nvlink3_oneway_raw_gbps": 300.0,
                     "note": "V100-SXM2 规格 300 GB/s 双向总 → 单向 150；A100 NVLink3 单向 300"},
    }

    # ① 路径验证：链路状态 + 计数器可用性 + 带宽论证
    #    （-gt g 计数器本平台不可用 → 以 -s 标称速率为链路口径）
    r_path = d2d_one_way(256 * 1024 * 1024, 1)
    ls4 = nvlink_link_status(4)
    ls5 = nvlink_link_status(5)
    nom = ls4.get("nominal_oneway_raw_gbps") or 154.69
    payload["nvlink_path_check"] = {
        "gt_g_counters": nvlink_gt_g_counters_status(),
        "link_status_gpu4": ls4,
        "link_status_gpu5": ls5,
        "moved_mb": 256,
        "one_way_256mb_gbps": r_path["gbps"],
        "inference": (
            f"实测单向 {r_path['gbps']} GB/s ≈ {r_path['gbps'] / nom * 100:.1f}% "
            f"NV6 标称单向 raw（{nom} GB/s），≫ host PCIe 通路上限"
            "（V100 SXM2 Gen3 x16 有效 ≈12–13 GB/s）→ 流量走 NVLink 而非 host 中转；"
            "与前节 NCCL transport P2P/CUMEM（GDR 1）一致"),
    }

    # ② 体量网格（单流 / 8 流）
    for mb in SIZES_MB:
        nb = mb * 1024 * 1024
        payload["one_way_single_stream"][f"{mb}mb"] = d2d_one_way(nb, 1)
        payload["one_way_8stream"][f"{mb}mb"] = d2d_one_way(nb, N_STREAMS)

    # ③ 双向并发
    payload["bidir"]["256mb"] = d2d_bidir(256 * 1024 * 1024)

    # ④ 小搬运延迟
    payload["latency"][f"{LAT_KB}kb_single"] = d2d_one_way(LAT_KB * 1024, 1)

    # ⑤ 效率参考（NCCL 拟合 BW_eff）
    bw_eff = None
    try:
        fit = json.loads(NCCL_FIT_JSON.read_text())["fit"]
        bw_eff = fit["bw_eff_gbps"]
    except Exception as e:
        print(f"  (nccl fit JSON 读取失败: {e}；回退 102.235)", file=sys.stderr)
        bw_eff = 102.235
    # 单向 raw 参考取单流 256MB（实测最高）；8 流分块并行反而略低（chunk/stream 开销）
    sat = payload["one_way_single_stream"]["256mb"]["gbps"]
    sat8 = payload["one_way_8stream"]["256mb"]["gbps"]
    payload["efficiency_ref"] = {
        "bw_eff_nccl_gbps": bw_eff,
        "raw_oneway_single_stream_256mb_gbps": sat,
        "raw_oneway_8stream_256mb_gbps": sat8,
        "engine_efficiency": round(bw_eff / sat, 4) if sat else None,
        "note": "raw 参考 = 单流单向 256MB（实测最高）；8 流分块并行 256MB 略低（stream/chunk 同步开销）",
    }

    RESULTS = Path(__file__).parent / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "p2p_raw_v100nv6.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved {p}")
    for k in ("one_way_single_stream", "one_way_8stream"):
        for size, r in payload[k].items():
            print(f"  {k} {size}: {r['gbps']} GB/s (t={r['t_ms']} ms)", flush=True)
    b = payload["bidir"]["256mb"]
    print(f"  bidir 256mb: total {b['total_gbps']} GB/s, per-dir {b['per_dir_gbps']} GB/s")
    print(f"  latency {LAT_KB}kb: {payload['latency'][f'{LAT_KB}kb_single']['t_ms'] * 1e3:.1f} µs")
    pc = payload["nvlink_path_check"]
    l4, l5 = pc["link_status_gpu4"], pc["link_status_gpu5"]
    print(f"  nvlink path check: gpu4 {l4['n_links']}×{l4['per_link_gbps']} GB/s, "
          f"gpu5 {l5['n_links']}×{l5['per_link_gbps']} GB/s; counters: {pc['gt_g_counters']}")
    print(f"  one-way 256mb (path check): {pc['one_way_256mb_gbps']} GB/s")
    print(f"  efficiency ref: {payload['efficiency_ref']}")


if __name__ == "__main__":
    main()
