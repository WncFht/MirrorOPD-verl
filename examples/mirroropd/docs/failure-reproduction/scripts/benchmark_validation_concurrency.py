#!/usr/bin/env python3
"""Measure one TP1 validation replica: natural math requests and controlled KV stress."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--stress-requests", type=int, default=48)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--sampling-warmup", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NVTX_INJECTION64_PATH", "/nonexistent/verl-disabled-nvtx-injection.so")
    import pyarrow.parquet as pq
    import pynvml
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    args.output.mkdir(parents=True, exist_ok=True)
    family = "qwen3" if args.model.name.startswith("Qwen3") else "deepseek"
    prompt_cap = 1024 if args.model.name.endswith("Base") else 4096
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    builder = create_continuous_token_builder(
        tokenizer,
        model_family=family,
        hf_model_type=config.model_type,
        chat_template_kwargs={"enable_thinking": False} if family == "qwen3" else {},
    )
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
        gpu_memory_utilization=0.65,
        max_model_len=prompt_cap + 31744,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        disable_log_stats=False,
        seed=42,
        override_generation_config={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0},
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": captures},
    )
    result = {
        "model": str(args.model),
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": 0.65,
        "token_budget": 8192,
        "max_model_len": prompt_cap + 31744,
        "response_cap": 31744,
        "prefix_caching": True,
        "initialization_seconds": time.monotonic() - started,
        "subset_dir": str(args.subset_dir),
        "samples_per_prompt": args.samples,
        "sampling_warmup": args.sampling_warmup,
        "scope": "One isolated TP1 replica per GPU; excludes Ray/FSDP startup and grading. Not four-GPU throughput.",
    }
    try:
        kv = llm.llm_engine.vllm_config.cache_config
        result["cache_config"] = {k: getattr(kv, k, None) for k in ("block_size", "num_gpu_blocks", "cache_dtype")}
    except AttributeError:
        pass
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))

    def metrics():
        return {
            m.name: m.value
            for m in llm.get_metrics()
            if hasattr(m, "value")
            and any(
                s in m.name
                for s in (
                    "preempt",
                    "kv_cache_usage",
                    "num_requests_running",
                    "num_requests_waiting",
                    "generation_tokens",
                    "prompt_tokens",
                )
            )
        }

    def persist():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def run_phase(name, inputs, params):
        snapshots, errors, stop = [], [], threading.Event()
        before = metrics()

        def observe():
            while not stop.is_set():
                try:
                    snapshots.append(
                        {
                            "time": time.time(),
                            **metrics(),
                            "gpu_memory_MiB": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
                            "gpu_utilization": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        }
                    )
                except Exception as exc:
                    errors.append(repr(exc))
                stop.wait(0.5)

        thread = threading.Thread(target=observe, daemon=True)
        thread.start()
        t = time.monotonic()
        try:
            outputs = llm.generate(inputs, params, use_tqdm=True)
        finally:
            elapsed = time.monotonic() - t
            stop.set()
            thread.join()
        after = metrics()
        counts = [len(r.outputs[0].token_ids) for r in outputs]
        data = {
            "seconds": elapsed,
            "requests": len(outputs),
            "output_tokens": sum(counts),
            "input_tokens": sum(len(r.prompt_token_ids) for r in outputs),
            "output_tps": sum(counts) / elapsed,
            "length_count": sum(r.outputs[0].finish_reason == "length" for r in outputs),
            "stop_count": sum(r.outputs[0].finish_reason == "stop" for r in outputs),
            "peak_gpu_MiB": max((s["gpu_memory_MiB"] for s in snapshots), default=None),
            "peak_kv_fraction": max(
                (v for s in snapshots for k, v in s.items() if "kv_cache_usage" in k), default=None
            ),
            "max_running": max(
                (v for s in snapshots for k, v in s.items() if "num_requests_running" in k), default=None
            ),
            "preemptions": {k: v - before.get(k, 0) for k, v in after.items() if "preempt" in k},
            "observation_errors": errors,
        }
        (args.output / f"{name}-telemetry.json").write_text(json.dumps(snapshots) + "\n")
        result[name] = data
        persist()
        print("PHASE_RESULT " + json.dumps({"phase": name, **data}), flush=True)
        return outputs

    # Warm engine kernels without counting initialization or first-call work as throughput.
    warm = [{"prompt_token_ids": [1000 + i, 17, 18]} for i in range(min(args.max_num_seqs, 32))]
    warm_params = (
        SamplingParams(temperature=0.7, top_p=0.9, top_k=-1, max_tokens=32, ignore_eos=True, detokenize=False)
        if args.sampling_warmup
        else SamplingParams(temperature=0, max_tokens=32, allowed_token_ids=[17], detokenize=False)
    )
    llm.generate(warm, warm_params, use_tqdm=False)
    llm.reset_prefix_cache()
    requests, params, metadata = [], [], []
    for dataset in ("MATH-500", "AMC23", "AIME25"):
        rows = pq.read_table(args.subset_dir / f"{dataset}.parquet").to_pylist()
        for row_index, row in enumerate(rows):
            ids = builder.build_initial_tokens(row["prompt"])
            if len(ids) > prompt_cap:
                raise ValueError(f"Overlong prompt: {dataset}/{row_index}: {len(ids)}")
            for sample in range(args.samples):
                requests.append({"prompt_token_ids": ids})
                params.append(
                    SamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=-1,
                        max_tokens=31744,
                        seed=42 + len(params),
                        detokenize=False,
                        stop_token_ids=[151645] if family == "qwen3" else [],
                    )
                )
                metadata.append({"dataset": dataset, "row": row_index, "sample": sample})
    outputs = run_phase("natural", requests, params)
    with (args.output / "natural-outputs.jsonl").open("w") as stream:
        for meta, output in zip(metadata, outputs, strict=True):
            completion = output.outputs[0]
            stream.write(
                json.dumps(
                    {
                        **meta,
                        "prompt_ids": output.prompt_token_ids,
                        "response_ids": list(completion.token_ids),
                        "text": tokenizer.decode(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    if not args.skip_stress:
        llm.reset_prefix_cache()
        # Distinct first blocks prevent shared-prefix hits, even with caching enabled.
        requests = [{"prompt_token_ids": [1000 + i] + [17, 18, 19] * 9557} for i in range(args.stress_requests)]
        outputs = run_phase(
            "stress",
            requests,
            SamplingParams(
                temperature=0,
                max_tokens=1024,
                allowed_token_ids=[17],
                detokenize=False,
            ),
        )
        assert all(len(r.outputs[0].token_ids) == 1024 for r in outputs)
    result["complete"] = True
    persist()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
