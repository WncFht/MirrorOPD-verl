# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reproducible dual-KV workload with natural EOS, independent replica processes."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import statistics
import threading
import time
from contextlib import nullcontext
from pathlib import Path


def worker(rank, args, barrier):
    import psutil
    import pynvml
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from verl.experimental.rollout_policy.hf_decoder import DecodeRequest, HFFusionDecoder
    from verl.experimental.rollout_policy.sampling import SamplingConfig, validate_tokenizers
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    torch.set_num_threads(1)
    pynvml.nvmlInit()
    devices = [rank] if args.layout == "colocated" else [2 * rank, 2 * rank + 1]
    physical_ids = [int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[device]) for device in devices]
    handles = [pynvml.nvmlDeviceGetHandleByIndex(device) for device in physical_ids]
    physical, handle = physical_ids[0], handles[0]
    if any(pynvml.nvmlDeviceGetMemoryInfo(item).used > 1024**3 for item in handles):
        raise RuntimeError(f"GPU group {physical_ids} is occupied")
    torch.cuda.set_device(devices[0])

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank-{rank}.jsonl"
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")

    def record(value):
        with path.open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    student_config = AutoConfig.from_pretrained(args.student, local_files_only=True)
    teacher_config = AutoConfig.from_pretrained(args.teacher, local_files_only=True)
    legal = validate_tokenizers(
        student_tokenizer, teacher_tokenizer, student_config.vocab_size, teacher_config.vocab_size
    )
    prompt_tokenizer = AutoTokenizer.from_pretrained(args.prompt_model or args.student, local_files_only=True)
    if prompt_tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError("Prompt tokenizer differs from the learner token-ID mapping")
    builder = create_continuous_token_builder(
        prompt_tokenizer,
        model_family="qwen3",
        hf_model_type=student_config.model_type,
        chat_template_kwargs=args.chat_template_kwargs,
    )
    import pyarrow.parquet as pq

    rows = pq.read_table(args.data, columns=["prompt"]).slice(0, args.batch).to_pylist()
    if len(rows) != args.batch:
        raise ValueError("Insufficient rows for the declared global batch")
    requests = [
        DecodeRequest(str(i), tuple(builder.build_initial_tokens(row["prompt"])), args.seed + i)
        for i, row in enumerate(rows)
    ]
    stop_ids = tuple(sorted(set(args.stop_ids)))
    config = SamplingConfig(max_tokens=args.cap, stop_ids=() if args.replay else stop_ids, legal_ids=legal)
    cold_start = time.monotonic()

    def load(model_path, device):
        return (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, attn_implementation=args.attention, local_files_only=True
            )
            .eval()
            .to(device)
        )

    student = load(args.student, devices[0]) if args.alpha < 1 else None
    teacher = load(args.teacher, devices[-1]) if args.alpha > 0 else None
    from verl.experimental.rollout_policy.precision import configure_precision

    for model in (student, teacher):
        if model is not None:
            configure_precision(
                model, args.precision, attention_dtype=torch.float16 if args.attention_dtype == "fp16" else None
            )
    synchronize()
    from fingerprint import source_fingerprint

    record(
        {
            "kind": "metadata",
            "source": source_fingerprint(),
            "rank": rank,
            "physical_gpu": physical,
            "physical_gpus": physical_ids,
            "gpu_names": [str(pynvml.nvmlDeviceGetName(item)) for item in handles],
            "gpu_memory_bytes": [pynvml.nvmlDeviceGetMemoryInfo(item).total for item in handles],
            "driver": str(pynvml.nvmlSystemGetDriverVersion()),
            "cuda_runtime": torch.version.cuda,
            "sampling": {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k},
            "legal_vocab_size": len(legal),
            "vocabulary_sha256": hashlib.sha256(
                json.dumps(
                    student_tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "inference_math": (
                "cuBLASLt; BF16 reduced-precision reduction and split-K disabled; "
                "cuDNN SDPA disabled unless explicitly requested"
            ),
            "args": vars(args),
            "resolved_precision": {
                name: getattr(model, "_fusion_precision", "bf16")
                for name, model in (("student", student), ("teacher", teacher))
                if model is not None
            },
            "prefill_attention": {
                name: model.config._attn_implementation
                for name, model in (("student", student), ("teacher", teacher))
                if model is not None
            },
            "prompt_lengths": [len(r.prompt_ids) for r in requests],
            "prompt_tokens_sha256": hashlib.sha256(
                json.dumps([r.prompt_ids for r in requests], separators=(",", ":")).encode()
            ).hexdigest(),
            "prompt_source_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
            "cold_load_seconds": time.monotonic() - cold_start,
            "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers"]},
            "stop_ids": stop_ids,
            "native_eos": {"student": student_tokenizer.eos_token_id, "teacher": teacher_tokenizer.eos_token_id},
            "chat_template_kwargs": args.chat_template_kwargs,
            "profiler": args.profile,
            "cache_policy": (
                "retain weights and CUDA allocator cache; reset both KV and rebuild graphs for every microbatch"
            ),
            "preemption": "unavailable",
            "recompute": "unavailable",
        }
    )
    decoder_class = HFFusionDecoder
    if args.layout == "paired":
        from paired_reference import PairedReference

        decoder_class = PairedReference
    for compact in args.compact:
        decoder = decoder_class(
            student,
            teacher,
            pad_id=student_tokenizer.pad_token_id,
            compact=bool(compact),
            compact_power_of_two=compact == 2,
            stop_diagnostics=args.stop_diagnostics,
            native_stop_ids=(student_tokenizer.eos_token_id, teacher_tokenizer.eos_token_id),
            allow_cudnn=args.allow_cudnn,
            profile=args.profile,
            preallocate_kv=args.preallocate_kv,
            flash_kv=args.flash_kv,
            flash_kv_splits=args.flash_kv_splits,
            cuda_graph=args.cuda_graph,
        )
        if args.replay:
            from types import MethodType

            def fixed_tokens(self, log_probs, generators, step):
                # Operator/cache comparison only. Every implementation sees the
                # same decimal-token prefix; natural-EOS runs remain separate.
                tokens = torch.full((log_probs.shape[0],), 16 + step % 10, device=log_probs.device, dtype=torch.long)
                return tokens, log_probs.gather(-1, tokens[:, None]).squeeze(-1)

            decoder._select_tokens = MethodType(fixed_tokens, decoder)
        for microbatch in args.microbatch:
            global_order = sorted(requests, key=lambda r: len(r.prompt_ids)) if compact else requests
            chunks = [global_order[i : i + microbatch] for i in range(0, len(global_order), microbatch)]
            ordered = [request for i, chunk in enumerate(chunks) if i % args.replicas == rank for request in chunk]
            warm_start = time.monotonic()
            warm_config = (
                dataclasses.replace(config, max_tokens=min(config.max_tokens, args.warmup_cap))
                if args.warmup_cap
                else config
            )
            for offset in range(0, len(ordered), microbatch):
                decoder.generate(
                    ordered[offset : offset + microbatch], args.alpha, warm_config, deadline_seconds=args.deadline
                )
            synchronize()
            record(
                {
                    "kind": "warmup",
                    "compact": compact,
                    "microbatch": microbatch,
                    "seconds": time.monotonic() - warm_start,
                }
            )
            for repeat in range(args.repeats):
                telemetry = []
                stopped = threading.Event()

                def monitor(stopped=stopped, telemetry=telemetry):
                    while not stopped.is_set():
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        telemetry.append(
                            {
                                "at": time.time(),
                                "util": util.gpu,
                                "used_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                                "gpus": [
                                    {
                                        "index": device,
                                        "util": pynvml.nvmlDeviceGetUtilizationRates(item).gpu,
                                        "used_bytes": pynvml.nvmlDeviceGetMemoryInfo(item).used,
                                    }
                                    for device, item in zip(physical_ids, handles, strict=True)
                                ],
                                "rss_bytes": psutil.Process().memory_info().rss,
                            }
                        )
                        stopped.wait(0.25)

                thread = threading.Thread(target=monitor, daemon=True)
                for device in devices:
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize()
                thread.start()
                barrier.wait(timeout=args.deadline)
                start = time.monotonic()
                results, stages = [], []
                try:
                    profile_stem = f"rank{rank}-compact{compact}-mb{microbatch}-{repeat}"

                    def export_profile(profiler, stem=profile_stem):
                        profiler.export_chrome_trace(str(output_dir / f"trace-{stem}.json.gz"))
                        table = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=60)
                        (output_dir / f"profile-{stem}.txt").write_text(table)

                    profile_context = (
                        torch.profiler.profile(
                            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                            schedule=torch.profiler.schedule(
                                wait=args.profile_start, warmup=0, active=args.profile_steps, repeat=1
                            ),
                            on_trace_ready=export_profile,
                        )
                        if args.profile
                        else nullcontext()
                    )
                    with profile_context as profiler:
                        decoder.profile_step = profiler.step if args.profile else None
                        for offset in range(0, len(ordered), microbatch):
                            results.extend(
                                decoder.generate(
                                    ordered[offset : offset + microbatch],
                                    args.alpha,
                                    config,
                                    deadline_seconds=args.deadline,
                                )
                            )
                            stages.append(dict(decoder.metrics))
                    decoder.profile_step = None
                    synchronize()
                    local_elapsed = time.monotonic() - start
                    barrier.wait(timeout=args.deadline)
                    elapsed = time.monotonic() - start
                finally:
                    stopped.set()
                    thread.join()
                lengths = [len(r.token_ids) for r in results]
                record(
                    {
                        "kind": "measurement",
                        "rank": rank,
                        "compact": compact,
                        "microbatch": microbatch,
                        "repeat": repeat,
                        "seconds": elapsed,
                        "local_decode_seconds": local_elapsed,
                        "effective_tokens": sum(lengths),
                        "tokens_per_second": sum(lengths) / elapsed,
                        "lengths": lengths,
                        "eos": sum(r.finish_reason == "eos" for r in results),
                        "completed": len(results),
                        "peak_allocated": max(torch.cuda.max_memory_allocated(device) for device in devices),
                        "peak_reserved": max(torch.cuda.max_memory_reserved(device) for device in devices),
                        "gpu_peaks": [
                            {
                                "index": physical_id,
                                "allocated": torch.cuda.max_memory_allocated(device),
                                "reserved": torch.cuda.max_memory_reserved(device),
                            }
                            for physical_id, device in zip(physical_ids, devices, strict=True)
                        ],
                        "telemetry": telemetry,
                        "host_stage_times": stages,
                        "results": [dataclasses.asdict(r) for r in results],
                    }
                )
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "compact": compact,
                            "microbatch": microbatch,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "local_decode_seconds": local_elapsed,
                            "tokens": sum(lengths),
                        }
                    ),
                    flush=True,
                )

        if hasattr(decoder, "close"):
            decoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dtype", choices=["bf16", "fp16"], default="fp16")
    parser.add_argument(
        "--replay", action="store_true", help="Fixed decimal-token prefixes; disables natural EOS and random sampling"
    )
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="flash_attention_2")
    parser.add_argument(
        "--precision",
        choices=["bf16", "head", "residual", "linear", "intermediate", "stable", "stable_moe", "stable_moe_attention"],
        default="stable",
    )
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--prompt-model")
    parser.add_argument(
        "--chat-template-kwargs", type=json.loads, default={}, help="Explicit prompt protocol as a JSON object"
    )
    parser.add_argument("--stop-ids", type=int, nargs="+", default=[151643, 151645])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout", choices=["colocated", "paired"], default="colocated")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--cap", type=int, default=7168)
    parser.add_argument("--microbatch", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--compact", type=int, nargs="+", choices=[0, 1, 2], default=[0, 1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--flash-kv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shared decode arithmetic for reference and optimized paths; disable only for native-kernel diagnostics",
    )
    parser.add_argument("--flash-kv-splits", type=int, default=1)
    parser.add_argument("--preallocate-kv", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-start", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument(
        "--warmup-cap", type=int, default=0, help="Diagnostic profiling only; formal measurements warm a full batch"
    )
    parser.add_argument("--allow-cudnn", action="store_true")
    parser.add_argument("--stop-diagnostics", action="store_true")
    parser.add_argument("--deadline", type=float, default=7200)
    args = parser.parse_args()
    if not isinstance(args.chat_template_kwargs, dict):
        parser.error("--chat-template-kwargs must be a JSON object")
    if args.warmup_cap and (not args.profile or args.warmup_cap < 1):
        parser.error("Shortened warmup is allowed only for separate profiler diagnostics")
    if args.cuda_graph:
        if not args.flash_kv:
            parser.error("CUDA graphs require Flash KV decode")
        args.preallocate_kv = True
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        parser.error("Set CUDA_VISIBLE_DEVICES=0,1,2,3 explicitly")
    if not 1 <= args.gpus <= 4 or args.batch % args.gpus or min(args.microbatch) < 1:
        parser.error("Invalid GPU/batch/microbatch combination")
    if args.layout == "paired" and args.gpus % 2:
        parser.error("Paired layout requires an even GPU count")
    args.replicas = args.gpus if args.layout == "colocated" else args.gpus // 2
    if max(args.microbatch) > args.batch // args.replicas:
        parser.error("Microbatch exceeds per-replica workload; it would leave reserved GPUs idle")
    os.environ["NVTX_INJECTION64_PATH"] = "/nonexistent/verl-disabled-nvtx-injection.so"
    import torch.multiprocessing as mp

    barrier = mp.get_context("spawn").Barrier(args.replicas)
    mp.spawn(worker, args=(args, barrier), nprocs=args.replicas, join=True)
    rows = [json.loads(line) for p in Path(args.out).glob("rank-*.jsonl") for line in p.read_text().splitlines()]
    measurements = [r for r in rows if r["kind"] == "measurement"]
    summary = []
    for compact in args.compact:
        for microbatch in args.microbatch:
            values = []
            for repeat in range(args.repeats):
                group = [
                    r
                    for r in measurements
                    if (r["compact"], r["microbatch"], r["repeat"]) == (compact, microbatch, repeat)
                ]
                identities = [result["request_id"] for row in group for result in row["results"]]
                if len(group) != args.replicas or len(identities) != args.batch or len(set(identities)) != args.batch:
                    raise RuntimeError("Benchmark result does not cover the declared global batch exactly once")
                wall = max(r["seconds"] for r in group)
                tokens = sum(r["effective_tokens"] for r in group)
                values.append({"seconds": wall, "effective_tokens": tokens, "tokens_per_second": tokens / wall})
            summary.append(
                {
                    "compact": compact,
                    "microbatch": microbatch,
                    "raw": values,
                    "median_seconds": statistics.median(r["seconds"] for r in values),
                    "range_seconds": [min(r["seconds"] for r in values), max(r["seconds"] for r in values)],
                }
            )
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
