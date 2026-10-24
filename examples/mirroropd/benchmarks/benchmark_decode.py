#!/usr/bin/env python3
"""Measure one local TP1 replica; keep profiler runs separate from throughput.

This diagnostic deliberately fixes output length (ignore_eos) and does not run
teacher scoring, Ray, or training. It is not a quality or end-to-end benchmark.
The explicit physical-device allowlist matches this machine's debug allocation.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, choices=range(4), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["eager", "graph"], required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-start", type=int, default=32)
    parser.add_argument("--trace-iterations", type=int, default=40)
    args = parser.parse_args()
    if min(args.batches) < 1 or max(args.batches) > 32 or args.tokens < 128 or args.repeats < 1:
        parser.error("batches must be 1..32, tokens >= 128, repeats >= 1")
    if args.out.exists():
        parser.error("Use a new output directory")
    if args.trace and not (0 <= args.trace_start < args.trace_start + args.trace_iterations < args.tokens):
        parser.error("Trace window must be positive and finish before the output token budget")
    # Set before importing Torch/vLLM. This process and its children see one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # A preloaded NVTX injection handler can consume the only CUPTI subscriber.
    os.environ["NVTX_INJECTION64_PATH"] = "/nonexistent/verl-disabled-nvtx-injection.so"

    import pyarrow.parquet as pq
    import pynvml
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    torch.set_num_threads(1)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    if used > 1024**3:
        raise RuntimeError(f"GPU {args.device} already uses {used / 1024**3:.2f} GiB; refusing to launch")
    args.out.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    rows = pq.read_table(args.data, columns=["prompt"]).slice(0, max(args.batches)).to_pylist()
    prompts = []
    for row in rows:
        # Transformers 5 may return a BatchEncoding by default; vLLM needs the
        # integer IDs, not its dictionary keys. Keep this explicit across versions.
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False)
        if not isinstance(ids, list) or any(not isinstance(token, int) for token in ids):
            raise TypeError("Chat template did not return a flat list of token IDs")
        if not 0 < len(ids) <= 4096:
            raise ValueError(f"Unexpected prompt length: {len(ids)}")
        prompts.append({"prompt_token_ids": ids})
    if len(prompts) != max(args.batches):
        raise ValueError("Dataset has too few prompts")

    kwargs = dict(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=35841,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.5,
        enforce_eager=args.mode == "eager",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        seed=42,
        logprobs_mode="processed_logprobs",
    )
    if args.mode == "graph":
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}
    if args.trace:
        kwargs["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=str((args.out / "trace").resolve()),
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=False,
            torch_profiler_with_memory=False,
            ignore_frontend=True,
            delay_iterations=args.trace_start,
            max_iterations=args.trace_iterations,
        )
    metadata = {
        "kind": "local-single-replica-fixed-output-diagnostic",
        "device": args.device,
        "gpu": pynvml.nvmlDeviceGetName(handle),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "vllm", "transformers"]},
        "model": str(args.model),
        "data": str(args.data),
        "engine": kwargs,
        "prompt_tokens": [len(p["prompt_token_ids"]) for p in prompts],
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": 0, "ignore_eos": True},
        "tokens": args.tokens,
        "repeats": args.repeats,
        "limitations": (
            "Initial checkpoint; fixed output length; warmed prefix cache; no Ray/teacher/update; H800, not A800."
        ),
    }
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    llm = LLM(**kwargs)

    def generate(batch, tokens):
        return llm.generate(
            prompts[:batch],
            SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=tokens, ignore_eos=True, logprobs=0),
            use_tqdm=False,
        )

    results = []
    for batch in args.batches:
        # Warm this decode shape and prefix before collecting wall-clock timings.
        generate(batch, 64)
        for repeat in range(args.repeats):
            samples = []
            stop = threading.Event()

            def sample(stop, samples):
                while not stop.is_set():
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    samples.append(
                        {
                            "time": time.monotonic(),
                            "gpu_busy_percent": util.gpu,
                            "memory_gib": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3,
                            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        }
                    )
                    stop.wait(0.25)

            thread = threading.Thread(target=sample, args=(stop, samples), daemon=True)
            thread.start()
            begin = time.monotonic()
            try:
                outputs = generate(batch, args.tokens)
                elapsed = time.monotonic() - begin
            finally:
                stop.set()
                thread.join()
            lengths = [len(output.outputs[0].token_ids) for output in outputs]
            if len(lengths) != batch or any(length != args.tokens for length in lengths):
                raise RuntimeError(f"Incomplete fixed-output workload: {lengths}")
            result = {
                "batch": batch,
                "repeat": repeat,
                "seconds": elapsed,
                "output_tokens": sum(lengths),
                "output_tokens_per_second": sum(lengths) / elapsed,
                # This is batch wall time / output length, not measured per-request TPOT.
                "batch_wall_ms_per_output_position": elapsed * 1000 / args.tokens,
                "gpu_busy_probe_mean": statistics.mean(s["gpu_busy_percent"] for s in samples),
                "memory_gib_probe_max": max(s["memory_gib"] for s in samples),
                "samples": samples,
            }
            results.append(result)
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    if args.trace:
        # Capture a separate generation. Skip prefill/early decode and stop after
        # 40 engine iterations; these profiled timings never enter results.json.
        llm.start_profile()
        try:
            generate(max(args.batches), args.tokens)
        finally:
            llm.stop_profile()
        print(f"Trace directory: {args.out / 'trace'}", flush=True)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
