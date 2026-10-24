#!/usr/bin/env python3
"""Bounded, continuously supplied TP1 throughput probe; never an accuracy evaluation."""

import argparse
import json
import os
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--settle-seconds", type=float, default=20)
    parser.add_argument("--measure-seconds", type=float, default=60)
    parser.add_argument("--queue-depth", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_num_seqs < 1 or args.queue_depth < 4 * args.max_num_seqs:
        parser.error("queue-depth must be at least four times max-num-seqs")
    if args.measure_seconds <= 0 or args.settle_seconds < 0:
        parser.error("invalid timing window")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must lie between zero and one")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NVTX_INJECTION64_PATH", "/nonexistent/verl-disabled-nvtx-injection.so")
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Use a fresh output directory; completed results are immutable")
    family = "qwen3" if args.model.name.startswith("Qwen3") else "deepseek"
    prompt_cap = 1024 if args.model.name.endswith("Base") else 4096
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_config = AutoConfig.from_pretrained(args.model)
    builder = create_continuous_token_builder(
        tokenizer,
        model_family=family,
        hf_model_type=model_config.model_type,
        chat_template_kwargs={"enable_thinking": False} if family == "qwen3" else {},
    )
    cases = []
    for dataset in ("MATH-500", "AMC23", "AIME25"):
        for row_index, row in enumerate(pq.read_table(args.subset_dir / f"{dataset}.parquet").to_pylist()):
            ids = builder.build_initial_tokens(row["prompt"])
            if len(ids) > prompt_cap:
                raise ValueError(f"Overlong prompt: {dataset}/{row_index}: {len(ids)}")
            cases.append({"dataset": dataset, "row": row_index, "prompt_token_ids": ids})
    # Fixed ordering across candidates; spread the three datasets through the stream.
    random.Random(args.seed).shuffle(cases)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "family": family,
        "max_model_len": prompt_cap + 31744,
        "response_cap": 31744,
        "temperature": 0.7,
        "top_p": 0.9,
        "prefix_caching": True,
        "tp": 1,
        "dtype": "bfloat16",
        "scope": (
            "Time-bounded throughput only. In-flight requests are aborted at deadline. "
            "Short windows do not certify 32K steady-state KV stability or task accuracy."
        ),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "prompts.json").write_text(json.dumps(cases) + "\n")
    if args.prepare_only:
        print(json.dumps({"prepared_prompts": len(cases), "config": config}), flush=True)
        return

    import pynvml

    pynvml.nvmlInit()
    physical_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # Do not silently benchmark alongside somebody else's process or lower the budget.
    if memory.used > 1024**3:
        raise RuntimeError(f"GPU {physical_gpu} is occupied: {memory.used / 1024**3:.2f} GiB in use")

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    captures, size = [], 1
    while size < args.max_num_seqs:
        captures.append(size)
        size *= 2
    captures.append(args.max_num_seqs)
    started = time.monotonic()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=prompt_cap + 31744,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.token_budget,
        enable_prefix_caching=True,
        disable_log_stats=False,
        seed=args.seed,
        override_generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": captures},
    )
    initialization_seconds = time.monotonic() - started
    llm.generate(
        [{"prompt_token_ids": cases[i % len(cases)]["prompt_token_ids"]} for i in range(args.max_num_seqs)],
        SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=-1,
            max_tokens=32,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        ),
        use_tqdm=False,
    )
    llm.reset_prefix_cache()
    engine = llm.llm_engine
    live, sequence = set(), 0

    def replenish():
        nonlocal sequence
        count = args.queue_depth - len(live)
        requests, parameters = [], []
        for _ in range(count):
            case = cases[sequence % len(cases)]
            requests.append({"prompt_token_ids": case["prompt_token_ids"]})
            parameters.append(
                SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=-1,
                    max_tokens=31744,
                    seed=args.seed + sequence,
                    detokenize=False,
                    stop_token_ids=[151645] if family == "qwen3" else [],
                    output_kind=RequestOutputKind.FINAL_ONLY,
                )
            )
            sequence += 1
        live.update(llm.enqueue(requests, parameters, use_tqdm=False))

    def snapshot():
        selected = (
            "generation_tokens",
            "prompt_tokens",
            "num_preemptions",
            "kv_cache_usage_perc",
            "num_requests_running",
            "num_requests_waiting",
        )
        values = {
            m.name: m.value
            for m in llm.get_metrics()
            if hasattr(m, "value") and m.name in {"vllm:" + x for x in selected}
        }
        values.update(
            elapsed_seconds=time.monotonic() - begin,
            gpu_memory_MiB=pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
            gpu_utilization=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        )
        return values

    replenish()
    begin = time.monotonic()
    measurement_start = None
    snapshots, completed = [], []
    next_snapshot, next_progress = 0.0, 15.0
    try:
        while True:
            for output in engine.step():
                if output.finished:
                    live.discard(output.request_id)
                    response = output.outputs[0]
                    completed.append(
                        {
                            "id": output.request_id,
                            "length": len(response.token_ids),
                            "finish_reason": response.finish_reason,
                            "elapsed_seconds": time.monotonic() - begin,
                        }
                    )
            elapsed = time.monotonic() - begin
            if elapsed >= next_snapshot:
                sample = snapshot()
                snapshots.append(sample)
                next_snapshot = elapsed + 1
                if measurement_start is None and elapsed >= args.settle_seconds:
                    measurement_start = sample
            if measurement_start is not None:
                if elapsed - measurement_start["elapsed_seconds"] >= args.measure_seconds:
                    break
            if len(live) < args.queue_depth - max(32, args.max_num_seqs):
                replenish()
            if elapsed >= next_progress:
                print("PROGRESS " + json.dumps(snapshots[-1]), flush=True)
                next_progress = elapsed + 15
        end = snapshot()
        snapshots.append(end)
    finally:
        # Abort only this engine's requests, including queued ones; do not wait for long tails.
        censored = len(live)
        engine.abort_request(list(live))

    duration = end["elapsed_seconds"] - measurement_start["elapsed_seconds"]
    measured = [s for s in snapshots if s["elapsed_seconds"] >= measurement_start["elapsed_seconds"]]
    generated = end["vllm:generation_tokens"] - measurement_start["vllm:generation_tokens"]
    result = {
        "config": config,
        "complete": True,
        "gpu": physical_gpu,
        "initialization_seconds": initialization_seconds,
        "measured_seconds": duration,
        "output_tokens": generated,
        "output_tps": generated / duration,
        "preemptions": end["vllm:num_preemptions"] - measurement_start["vllm:num_preemptions"],
        "peak_kv_fraction": max(s["vllm:kv_cache_usage_perc"] for s in measured),
        "peak_gpu_MiB": max(s["gpu_memory_MiB"] for s in measured),
        "mean_running": sum(s["vllm:num_requests_running"] for s in measured) / len(measured),
        "waiting_sample_fraction": sum(s["vllm:num_requests_waiting"] > 0 for s in measured) / len(measured),
        "completed_requests_total": len(completed),
        "censored_requests_at_deadline": censored,
        "submitted_requests": sequence,
    }
    (args.output / "telemetry.json").write_text(json.dumps(snapshots) + "\n")
    (args.output / "completed-requests.json").write_text(json.dumps(completed) + "\n")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
