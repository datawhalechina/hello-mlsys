#!/usr/bin/env python3
"""Build the Qwen3-4B / RX 7900 XTX joint-constraint screen plan.

This is a mechanism screen, not a hardware prediction.  The memory ledger uses
the measured 7900 XTX capacity and the chapter's KV geometry.  Service times
use the simulator's analytical roofline path with declared efficiencies.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "screen-qwen4b-7900xtx.plan.json"

GIB = 1024**3
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
VLLM_KV_POOL = 77248 * KV_BYTES_PER_TOKEN
LLAMA_512_KV_POOL = 64 * 512 * KV_BYTES_PER_TOKEN
LLAMA_131072_KV_POOL = 131072 * KV_BYTES_PER_TOKEN


def workload(count: int) -> list[dict]:
    return [
        {
            "request_id": str(index),
            "arrival_ms": 0,
            "input_tokens": 256,
            "output_tokens": 128,
            "priority": 0,
            "timeout_ms": None,
            "prompt_token_ids": [],
            "cache_namespace": "",
        }
        for index in range(count)
    ]


def model(dtype: str) -> dict:
    return {
        "name": "Qwen3-4B",
        "num_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "weight_bytes": 2.0,
        "kv_bytes": 2.0,
        "activation_bytes": 2.0,
        "revision": "local",
        "weight_dtype": dtype,
        "kv_dtype": dtype,
        "activation_dtype": dtype,
        "tie_word_embeddings": True,
        "qk_norm": True,
        "final_norm": True,
        "weight_dequant": "none",
        "architecture_config_sha256": "",
    }


def experiment(
    *,
    requests: int,
    max_batch_size: int,
    kv_pool_bytes: int,
    graph_bytes: int,
    workspace_bytes: int,
    runtime_bytes: int,
    chunked_prefill: bool,
    dtype: str = "bfloat16",
) -> dict:
    return {
        "workload": workload(requests),
        "infra": {
            "devices": [{
                "id": "gpu0",
                "name": "AMD Radeon RX 7900 XTX",
                "memory_bytes": 25753026560,
                "memory_bandwidth_gbps": 960.0,
                "compute_tflops": 122.8,
            }],
            "link_bandwidth_gbps": 20.0,
            "link_latency_us": 5.0,
            "topology": "single-node-uncontended",
        },
        "strategy": {
            "model": model(dtype),
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "stage_layers": [],
            "replicas": 1,
            "scheduler": "fifo",
            "max_batch_size": max_batch_size,
            "max_batch_tokens": 8192,
            "preemption": "recompute",
            "max_preemptions_per_request": 100,
            "memory": {
                "kv_block_tokens": 16,
                "kv_pool_bytes": kv_pool_bytes,
                "workspace_base_bytes": workspace_bytes,
                "workspace_bytes_per_token": 0,
                "graph_bytes": graph_bytes,
                "runtime_bytes": runtime_bytes,
                "communication_buffer_bytes": 0,
                "other_bytes": 0,
                "safety_margin_bytes": 0,
                "guard_blocks": 0,
                "cpu_swap_bytes": 0,
                "swap_d2h_bandwidth_gbps": 12.0,
                "swap_h2d_bandwidth_gbps": 12.0,
                "swap_latency_us": 10.0,
            },
            "pipeline_microbatch_size": 1,
            "pipeline_max_inflight_microbatches": 8,
            "pipeline_feedback_latency_us": 0.0,
            "pipeline_p2p": True,
            "pipeline_transport": "full_hidden",
            "pipeline_transport_max_tokens": 0,
            "swap_fallback": "none",
            "prefix_cache": False,
            "prefix_cache_max_blocks": 128,
            "chunked_prefill": chunked_prefill,
            "prefill_chunk_tokens": 512,
            "mixed_batching": chunked_prefill,
            "replica_routing": "least_loaded",
        },
        "calibration": {
            "backend": "analytical",
            "profile_version": "ch8-roofline-declared-v1",
            "compute_efficiency": 0.5,
            "bandwidth_efficiency": 0.8,
            "launch_overhead_us": 2.0,
            "framework_overhead_us": 5.0,
            "prefill_ms": 0.5,
            "decode_ms": 0.1,
            "recompute_ms": 10.0,
            "mixed_ms": 10.0,
            "dequant_flops_per_weight": 1.0,
            "profile_path": None,
            "replica_profile_paths": [],
            "allow_extrapolation": False,
            "allow_test_fixture": False,
            "runtime_name": "analytical-roofline",
            "runtime_version": "ch8-v1",
            "driver_version": "amdgpu",
            "rocm_version": "10.1",
            "rccl_version": "not-applicable",
            "timing_boundary": "step_includes_output_and_sampling_v1",
            "runtime_config_sha256": "",
            "native_pp_communication_profile_path": None,
            "replica_native_pp_communication_profile_paths": [],
            "native_pp_communication_profile_version": "",
            "native_pp_communication_allow_extrapolation": False,
            "workspace_profile_path": None,
            "replica_workspace_profile_paths": [],
            "workspace_profile_version": "",
        },
        "evaluation": {
            "ttft_sla_ms": 800.0,
            "tpot_sla_ms": 100.0,
            "measurement_window_ms": None,
            "drain": True,
            "max_drain_ms": None,
        },
        "execution": {
            "detailed_events": False,
            "initial_state": "empty_cache",
            "max_events": 200000,
            "max_simulated_ms": 180000.0,
            "seed": 42,
            "warmup": [],
        },
        "schema_version": 1,
    }


def main() -> None:
    plan = {
        "schema_version": 1,
        "kind": "simulation",
        "source": "chapter8_qwen3_4b_rx7900xtx_analytical_screen",
        "description": (
            "Qwen3-4B on one RX 7900 XTX. 48 concurrent 256/128 requests are a "
            "teaching stand-in for the C=48 holdout, not the 200-request measurement. "
            "Memory uses the measured capacity and chapter KV geometry; timings use "
            "the analytical roofline with declared efficiencies, not a measured profile."
        ),
        "constraints": {
            "schema_version": 1,
            "min_completed_samples": 1,
            "memory": {"max_budget_fraction": 1.0},
            "latency": {
                "p99_ttft_ms": 800.0,
                "p99_tpot_ms": 100.0,
                "p99_e2e_ms": 15000.0,
            },
        },
        "objective": {"metric": "elapsed_ms", "direction": "min"},
        "candidates": [
            {
                "candidate_id": "vllm-bf16-4096-c48",
                "experiment": experiment(
                    requests=48,
                    max_batch_size=48,
                    kv_pool_bytes=VLLM_KV_POOL,
                    graph_bytes=int(1.68 * GIB),
                    workspace_bytes=int(2.37 * GIB),
                    runtime_bytes=GIB,
                    chunked_prefill=True,
                    dtype="bfloat16",
                ),
            },
            {
                "candidate_id": "llamacpp-f16-512-c48",
                "experiment": experiment(
                    requests=48,
                    max_batch_size=48,
                    kv_pool_bytes=LLAMA_512_KV_POOL,
                    graph_bytes=0,
                    workspace_bytes=int(0.25 * GIB),
                    runtime_bytes=int(0.1 * GIB),
                    chunked_prefill=False,
                    dtype="float16",
                ),
            },
            {
                "candidate_id": "llamacpp-f16-131072-c1-memory-fail",
                "experiment": experiment(
                    requests=48,
                    max_batch_size=1,
                    kv_pool_bytes=LLAMA_131072_KV_POOL,
                    graph_bytes=0,
                    workspace_bytes=int(0.25 * GIB),
                    runtime_bytes=int(0.1 * GIB),
                    chunked_prefill=False,
                    dtype="float16",
                ),
            },
            {
                "candidate_id": "vllm-bf16-131072-c1-memory-fail",
                "experiment": experiment(
                    requests=48,
                    max_batch_size=1,
                    kv_pool_bytes=LLAMA_131072_KV_POOL,
                    graph_bytes=int(1.68 * GIB),
                    workspace_bytes=int(2.37 * GIB),
                    runtime_bytes=GIB,
                    chunked_prefill=True,
                    dtype="bfloat16",
                ),
            },
        ],
    }
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"KV bytes/token={KV_BYTES_PER_TOKEN} vllm_pool={VLLM_KV_POOL} ({VLLM_KV_POOL/GIB:.2f} GiB)")
    print(f"llama512_pool={LLAMA_512_KV_POOL} ({LLAMA_512_KV_POOL/GIB:.2f} GiB)")
    print(f"llama131072_pool={LLAMA_131072_KV_POOL} ({LLAMA_131072_KV_POOL/GIB:.2f} GiB)")


if __name__ == "__main__":
    main()
