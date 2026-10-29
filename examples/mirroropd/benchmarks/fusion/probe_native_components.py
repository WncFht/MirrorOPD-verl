# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay identical prefixes with independent native models or the composite."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--component", choices=["student", "teacher", "fusion"], required=True)
    parser.add_argument("--workload", required=True, help="Frozen JSON list of prompt_ids, suffix_ids and checkpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--engine-max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--kv-gib", type=float, default=4.0)
    parser.add_argument("--arrival-mode", choices=["synchronized", "asynchronous"], default="synchronized")
    args = parser.parse_args()
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != args.tensor_parallel_size:
        raise ValueError("Allocate exactly one GPU per TP rank for this probe")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    workloads = json.loads(Path(args.workload).read_text())
    capacity = args.batch if args.engine_max_num_seqs is None else args.engine_max_num_seqs
    if capacity < args.batch:
        raise ValueError("Engine capacity must cover the controlled batch")
    from fingerprint import source_fingerprint
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion import register_models
    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config, fusion_hf_overrides

    register_models()
    model_path = args.teacher if args.component == "teacher" else args.student
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=not args.graph,
        compilation_config=decode_graph_config(capacity) if args.graph else None,
        hf_overrides=fusion_hf_overrides(args.teacher) if args.component == "fusion" else {},
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * args.batch,
        max_num_seqs=capacity,
        kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config={"backend": "TRITON_ATTN"},
        worker_extension_cls="examples.mirroropd.benchmarks.fusion.probe_worker.PrefixProbe",
    )
    manifest = {"config": vars(args), "source": source_fingerprint(), "cases": []}
    for index, workload in enumerate(workloads):
        prompt, suffix = workload["prompt_ids"], workload["suffix_ids"]
        if not suffix or len(prompt) + len(suffix) > args.max_model_len:
            raise ValueError("Invalid replay length")
        llm.collective_rpc(
            "configure_probe", args=(suffix, workload["checkpoints"], args.component == "fusion", args.batch)
        )
        params = SamplingParams(temperature=0, max_tokens=len(suffix), ignore_eos=True, logprobs=0)
        prompts = [{"prompt_token_ids": prompt}] * args.batch
        if args.arrival_mode == "synchronized":
            core = llm.llm_engine.engine_core
            core.call_utility("pause_scheduler", "keep", False)
            try:
                llm.enqueue(prompts, params, use_tqdm=False)
            finally:
                core.call_utility("resume_scheduler")
            outputs = llm.wait_for_completion(use_tqdm=False)
        else:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        if any(list(output.outputs[0].token_ids) != suffix for output in outputs):
            raise RuntimeError("Replay changed the prescribed prefix")
        path = out / f"case-{index}.pt"
        audit = llm.collective_rpc("save_probe", args=(str(path),))
        output_ranks = [row for row in audit if row["output_rank"]]
        if len(output_ranks) != 1:
            raise RuntimeError("Numerical probe needs exactly one output rank")
        shapes = output_ranks[0]["batch_shapes"]
        if args.batch > 1 and max(shapes) < args.batch:
            raise RuntimeError("The calibration run never exercised the requested batch width")
        if args.arrival_mode == "synchronized" and set(shapes) != {args.batch}:
            raise RuntimeError("Synchronized numerical replay changed its batch width")
        manifest["cases"].append(
            {
                "index": index,
                "path": str(path),
                "checkpoints": workload["checkpoints"],
                "audit": audit,
            }
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
