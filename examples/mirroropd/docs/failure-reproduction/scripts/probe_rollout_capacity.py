#!/usr/bin/env python3
"""Bounded long-prefix KV pressure probe; synthetic tokens, not math accuracy."""

import argparse
import json
import os
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--prompt-tokens", type=int, default=28672)
    parser.add_argument("--response-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NVTX_INJECTION64_PATH", "/nonexistent/verl-disabled-nvtx-injection.so")
    from vllm import LLM, SamplingParams

    captures, size = [], 1
    while size < args.max_num_seqs:
        captures.append(size)
        size *= 2
    captures.append(args.max_num_seqs)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=False,
        gpu_memory_utilization=0.65,
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": captures},
    )
    snapshots, stop = [], threading.Event()

    def observe():
        while not stop.is_set():
            values = {
                m.name: m.value
                for m in llm.get_metrics()
                if hasattr(m, "value")
                and any(
                    key in m.name
                    for key in ("preempt", "kv_cache_usage", "num_requests_running", "num_requests_waiting")
                )
            }
            snapshots.append({"time": time.time(), **values})
            stop.wait(0.5)

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    # Each prefix starts with a different valid token. Prefix caching is disabled
    # to model distinct long student trajectories, not repeated shared prompts.
    inputs = [
        {
            "prompt_token_ids": [1000 + i]
            + [17, 18, 19] * ((args.prompt_tokens - 1) // 3)
            + [17] * ((args.prompt_tokens - 1) % 3)
        }
        for i in range(args.requests)
    ]
    params = SamplingParams(temperature=0, max_tokens=args.response_tokens, allowed_token_ids=[17], detokenize=False)
    started = time.monotonic()
    try:
        outputs = llm.generate(inputs, params, use_tqdm=True)
    finally:
        stop.set()
        thread.join()
    elapsed = time.monotonic() - started
    lengths = [len(r.outputs[0].token_ids) for r in outputs]
    assert lengths == [args.response_tokens] * args.requests
    metrics = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value") and "preempt" in m.name}
    result = {
        "model": str(args.model),
        "max_num_seqs": args.max_num_seqs,
        "requests": args.requests,
        "prompt_tokens": args.prompt_tokens,
        "response_tokens": args.response_tokens,
        "gpu_memory_utilization": 0.65,
        "token_budget": 8192,
        "prefix_caching": False,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(lengths) / elapsed,
        "preemption_metrics": metrics,
        "snapshots": snapshots,
        "scope": "Synthetic long-prefix scheduler pressure; not a task-quality benchmark or full validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "snapshots"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
