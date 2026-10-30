# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure a fixed global batch with TP2 or two TP1 replicas on one protected allocation."""

import argparse
import dataclasses
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def assigned_devices(allocation, tp, replicas, rank=None):
    """Map an explicitly allocated physical pair to each replica's CUDA mask."""
    from examples.mirroropd.benchmarks.fusion.resource_guard import parse_gpus

    devices = parse_gpus(allocation)
    if tp * replicas != 2 or len(devices) != 2 or tp not in {1, 2} or replicas not in {1, 2}:
        raise ValueError("Topology requires one protected physical GPU pair")
    if rank is None:
        return allocation
    if not 0 <= rank < replicas:
        raise ValueError("Replica rank is outside the protected topology")
    return ",".join(str(device) for device in devices[rank * tp : (rank + 1) * tp])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def wait_files(paths, processes, deadline):
    while not all(path.exists() for path in paths):
        failed = [p.pid for p in processes if p.poll() is not None]
        if failed:
            raise RuntimeError(f"Replica exited before completing its barrier: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Replica barrier deadline exceeded")
        time.sleep(0.02)


def coordinate(out, processes, repeats, timeout):
    """Every global batch starts after all replicas finish initialization/the previous batch."""
    rows = []
    for repeat in range(-1, repeats):
        deadline = time.monotonic() + timeout
        wait_files([out / f"ready-{repeat}-{rank}.json" for rank in range(len(processes))], processes, deadline)
        released = time.monotonic()
        write_json(out / f"go-{repeat}.json", {"released_monotonic": released})
        paths = [out / f"batch-{repeat}-{rank}.json" for rank in range(len(processes))]
        wait_files(paths, processes, deadline)
        batches = [json.loads(path.read_text()) for path in paths]
        collected = time.monotonic()
        if any(
            not all(math.isfinite(batch[key]) for key in ("started_monotonic", "ended_monotonic"))
            or not released <= batch["started_monotonic"] <= batch["ended_monotonic"]
            or batch["replica"] != rank
            for rank, batch in enumerate(batches)
        ):
            raise RuntimeError("Invalid replica identity or generation interval")
        rows.append(
            {
                "repeat": repeat,
                "seconds": collected - released,
                "collected_monotonic": collected,
                "engine_completion_seconds": max(b["ended_monotonic"] for b in batches) - released,
                "result_collection_seconds": collected - max(b["ended_monotonic"] for b in batches),
                "launch_skew_seconds": max(b["started_monotonic"] for b in batches)
                - min(b["started_monotonic"] for b in batches),
                "replicas": batches,
            }
        )
    return rows


def worker(args):
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion import register_models
    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config, fusion_hf_overrides

    register_models()
    prompts = json.loads((args.out / "prompts.json").read_text())
    indices = list(range(args.rank, args.batch, args.replicas))
    local = [prompts[index] for index in indices]
    llm = LLM(
        model=args.student,
        tokenizer=args.student,
        dtype="bfloat16",
        hf_overrides=fusion_hf_overrides(args.teacher),
        tensor_parallel_size=args.tp,
        enforce_eager=not args.graph,
        compilation_config=decode_graph_config(args.max_num_seqs) if args.graph else None,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
        gpu_memory_utilization=0.9,
        logprobs_mode="processed_logprobs",
        disable_log_stats=False,
        attention_config={"backend": "TRITON_ATTN"},
        worker_extension_cls="verl.workers.rollout.vllm_fusion.worker_extension.FusionWorkerExtension",
    )
    audit = llm.collective_rpc("fusion_prepare", args=(args.alpha,))
    write_json(args.out / f"replica-{args.rank}-audit.json", audit)
    for repeat in range(-1, args.repeats):
        params = [
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=args.cap,
                stop_token_ids=args.stop_token_ids,
                logprobs=0,
                seed=42 + index,
            )
            for index in indices
        ]
        write_json(args.out / f"ready-{repeat}-{args.rank}.json", {"indices": indices})
        deadline = time.monotonic() + args.timeout
        gate = args.out / f"go-{repeat}.json"
        while not gate.exists():
            if os.getppid() != args.parent_pid or time.monotonic() >= deadline:
                raise RuntimeError("Benchmark coordinator disappeared or timed out")
            time.sleep(0.02)
        started = time.monotonic()
        outputs = llm.generate(local, params, use_tqdm=False)
        ended = time.monotonic()
        if len(outputs) != len(indices):
            raise RuntimeError("Replica returned an incomplete batch")
        responses = []
        for index, output in zip(indices, outputs, strict=True):
            if list(output.prompt_token_ids) != prompts[index]["prompt_token_ids"]:
                raise RuntimeError("Replica output order differs from the assigned input order")
            response = output.outputs[0]
            probs = [item[token].logprob for token, item in zip(response.token_ids, response.logprobs, strict=True)]
            if not probs or not all(math.isfinite(x) for x in probs):
                raise RuntimeError("Missing finite behavior probabilities")
            responses.append(
                {
                    "global_index": index,
                    "seed": 42 + index,
                    "prompt_ids": output.prompt_token_ids,
                    "token_ids": response.token_ids,
                    "log_probs": probs,
                    "finish_reason": response.finish_reason,
                }
            )
        write_json(
            args.out / f"batch-{repeat}-{args.rank}.json",
            {
                "replica": args.rank,
                "started_monotonic": started,
                "ended_monotonic": ended,
                "responses": responses,
                "backend_metrics_cumulative": [dataclasses.asdict(x) for x in llm.get_metrics()],
            },
        )
    # Let the coordinator consume the last result before process teardown.
    while not (args.out / "finish.json").exists():
        if os.getppid() != args.parent_pid or time.monotonic() >= deadline:
            raise RuntimeError("Coordinator did not acknowledge the final results")
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--replicas", type=int, choices=[1, 2], required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--cap", type=int, default=7168)
    parser.add_argument("--max-model-len", type=int, default=11264)
    parser.add_argument("--max-num-seqs", type=int, required=True, help="Per-replica active request limit")
    parser.add_argument("--kv-gib", type=float, required=True, help="Per-GPU KV allocation")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--stop-token-ids", type=int, nargs="+", default=[151643, 151645])
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.student = str(Path(args.student).resolve())
    args.teacher = str(Path(args.teacher).resolve())
    args.data = str(Path(args.data).resolve())
    if (
        min(args.batch, args.cap, args.max_model_len, args.max_num_seqs) < 1
        or not math.isfinite(args.kv_gib)
        or args.kv_gib <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.alpha)
        or not 0 <= args.alpha <= 1
        or args.tp * args.replicas != 2
        or args.batch % args.replicas
        or args.repeats < 1
    ):
        parser.error("Invalid topology, workload, allocation, or timeout")
    from examples.mirroropd.benchmarks.fusion.resource_guard import verify_allocation

    allocation = os.environ.get("FUSION_GPUS", "")
    try:
        expected = assigned_devices(allocation, args.tp, args.replicas, args.rank)
        verify_allocation(Path(__file__).resolve().parents[5], allocation, visible=expected)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Run inside the protected two-GPU allocation: {error}")
    if args.rank is not None:
        if args.parent_pid != os.getppid():
            parser.error("Replica must be launched by its protected coordinator")
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    from fingerprint import source_fingerprint
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    frame = pd.read_parquet(args.data)
    if len(frame) < args.batch:
        raise ValueError("Insufficient workload rows")
    prompts = [
        {"prompt_token_ids": tokenizer.apply_chat_template(list(row), add_generation_prompt=True, return_dict=False)}
        for row in frame["prompt"].iloc[: args.batch]
    ]
    if any(len(p["prompt_token_ids"]) + args.cap > args.max_model_len for p in prompts):
        raise ValueError("Declared cap plus prompt exceeds context")
    write_json(args.out / "prompts.json", prompts)
    write_json(args.out / "config.json", vars(args) | {"out": str(args.out), "physical_gpus": allocation})
    write_json(args.out / "source.json", source_fingerprint())
    monitor = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("monitor.py")),
            "--parent",
            os.environ["FUSION_RESOURCE_PARENT"],
            "--out",
            str(args.out / "resources.jsonl"),
        ]
    )
    assert monitor.pid > 0
    processes, logs = [], []
    success = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank in range(args.replicas):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=assigned_devices(allocation, args.tp, args.replicas, rank))
            log = (args.out / f"replica-{rank}.log").open("x")
            logs.append(log)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--rank",
                str(rank),
                "--parent-pid",
                str(os.getpid()),
            ]
            processes.append(
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
        rows = coordinate(args.out, processes, args.repeats, args.timeout)
        for row in rows:
            indices = [r["global_index"] for batch in row["replicas"] for r in batch["responses"]]
            if sorted(indices) != list(range(args.batch)):
                raise RuntimeError("Global batch has missing or duplicate request identities")
        write_json(args.out / "finish.json", {"global_batches_complete": len(rows)})
        for process in processes:
            if process.wait(timeout=120) != 0:
                raise RuntimeError("Replica failed during teardown")
        write_json(
            args.out / "results.json",
            {"scope": "Global batch latency includes replica release skew; warmup separate", "batches": rows},
        )
        success = True
    finally:
        for process in processes:
            if not success or process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if not success:
            # A failed leader can leave vLLM descendants in its owned session.
            # Never use global process-name cleanup on this shared machine.
            for process in processes:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
