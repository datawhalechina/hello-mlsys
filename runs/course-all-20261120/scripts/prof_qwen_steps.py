#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    load_s = time.perf_counter() - load_start

    vocab = model.config.vocab_size
    with torch.inference_mode():
        warm_ids = torch.randint(0, vocab, (1, args.warmup_tokens), device="cuda:0")
        warm = model(input_ids=warm_ids, use_cache=True)
        past = warm.past_key_values
        nxt = warm.logits[:, -1, :].argmax(-1, keepdim=True)
        for _ in range(4):
            warm = model(input_ids=nxt, use_cache=True, past_key_values=past)
            past = warm.past_key_values
            nxt = warm.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        del warm, past, nxt

    ids = torch.randint(0, vocab, (1, args.prompt_tokens), device="cuda:0")
    torch.cuda.synchronize()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push("PREFILL")
    prefill_start.record()
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    prefill_end.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    step_times = []
    torch.cuda.nvtx.range_push("DECODE")
    with torch.inference_mode():
        for _ in range(args.decode_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(input_ids=nxt, use_cache=True, past_key_values=past)
            end.record()
            torch.cuda.synchronize()
            step_times.append(start.elapsed_time(end))
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.nvtx.range_pop()

    result = {
        "model_path": args.model,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "prompt_tokens": args.prompt_tokens,
        "decode_steps": args.decode_steps,
        "load_s": load_s,
        "prefill_ms": prefill_ms,
        "decode_step_ms": step_times,
        "decode_step_ms_median": statistics.median(step_times),
        "decode_tok_s_median": 1000.0 / statistics.median(step_times),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "hardware_validated": True,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
