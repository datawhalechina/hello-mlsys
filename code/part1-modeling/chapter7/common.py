#!/usr/bin/env python3
"""Shared constants and small helpers for chapter 7 (serving / queueing / SLA).

Everything that the chapter quotes in prose comes from a JSON file written by
one of the scripts in this directory.  This module only centralises the
measurement contract so the DES, the vLLM driver and the figure builder cannot
silently disagree about the model, card or SLOs.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Measurement contract (frozen before the holdout runs).
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/public/models/Qwen3-4B")
MODEL_NAME = "Qwen3-4B"
MODEL_DTYPE = "bfloat16"
CARD_NAME = "AMD Radeon RX 9070 XT"
CARD_ARCH = "gfx1201"
CARD_VRAM_GIB = 15.9
CONTAINER_IMAGE = ("rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-"
                   "pytorch_2.13.0_vllm-0.29.0")

INPUT_LEN = 256
OUTPUT_LEN = 128
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
KV_CACHE_TOKENS = 24464       # vLLM startup log, 2026-11-18
KV_BLOCK_TOKENS = 16          # vLLM default page size
KV_CACHE_GIB = 3.36

SLO = {
    "ttft_p99_ms": 800.0,
    "tpot_p99_ms": 100.0,
    "e2el_p99_ms": 15000.0,
    "failure_rate_max": 0.01,
}


def percentile(values: Iterable[float], p: float) -> float:
    """Linear-interpolated percentile; p is 0..100.  NaN on empty input."""
    data = sorted(float(v) for v in values)
    if not data:
        return math.nan
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def save_json(name: str | Path, payload: dict) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = RESULTS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_json(name: str | Path) -> dict:
    path = Path(name)
    if not path.is_absolute():
        path = RESULTS / path
    return json.loads(path.read_text(encoding="utf-8"))


def env_block() -> dict:
    """Host-side snapshot, same spirit as Ch2--Ch6 env_block()."""
    out = subprocess.run(
        ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showdriverversion"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    gpus: list[dict] = []
    pending = None
    for line in out.splitlines():
        if "Card Series:" in line:
            pending = line.split("Card Series:")[1].strip()
        elif "GFX Version:" in line and pending:
            gpus.append({"name": pending,
                         "arch": line.split("GFX Version:")[1].strip()})
            pending = None
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d)\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)", line)
        if m and int(m.group(1)) < len(gpus):
            gpus[int(m.group(1))]["vram_gib"] = round(int(m.group(2)) / 2**30, 2)
    driver = re.search(r"Driver version\s*:\s*(\d+\.\d+\.\d+)", out)
    return {
        "captured_at": now_iso(),
        "driver": driver.group(1) if driver else "unknown",
        "gpus": gpus,
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "dtype": MODEL_DTYPE,
        "container_image": CONTAINER_IMAGE,
        "device_selection": "HIP_VISIBLE_DEVICES=1 (host GPU[1] = 9070 XT)",
        "kv_cache_tokens": KV_CACHE_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
    }


def slo_pass(ttft_ms: float, tpot_ms: float, e2el_ms: float) -> bool:
    return (ttft_ms <= SLO["ttft_p99_ms"]
            and tpot_ms <= SLO["tpot_p99_ms"]
            and e2el_ms <= SLO["e2el_p99_ms"])
