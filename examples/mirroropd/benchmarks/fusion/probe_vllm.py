# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run native fusion on an explicitly allocated GPU and retain request evidence."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--kv-gib", type=float, default=1)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--skip-lifecycle-audit", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--chat-template-kwargs", default="{}")
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != args.tensor_parallel_size:
        raise ValueError("Allocate exactly one GPU per TP rank for this standalone probe")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    monitor = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("monitor.py")),
            "--parent",
            os.environ.get("FUSION_RESOURCE_PARENT", str(os.getpid())),
            "--out",
            str(out / "resources.jsonl"),
        ]
    )

    # Retain the child handle until process teardown; the monitor observes this PID.
    assert monitor.pid > 0
    from fingerprint import source_fingerprint

    (out / "source.json").write_text(json.dumps(source_fingerprint(), indent=2) + "\n")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.workers.rollout.vllm_fusion import register_models
    from verl.workers.rollout.vllm_fusion.compat import decode_graph_config, fusion_hf_overrides

    register_models()
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    questions = [
        "What is 2 + 3?",
        "Compute 12 * 13 and explain your calculation.",
        "What is 7 squared?",
        "Solve x + 9 = 21.",
    ]
    template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(template_kwargs, dict):
        raise ValueError("chat-template-kwargs must be a JSON object")
    if args.data:
        import pandas as pd

        frame = pd.read_parquet(args.data)
        if len(frame) < args.batch:
            raise ValueError("Insufficient workload rows")
        conversations = [list(row) for row in frame["prompt"].iloc[: args.batch]]
    else:
        conversations = [[{"role": "user", "content": questions[i % len(questions)]}] for i in range(args.batch)]
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_dict=False, **template_kwargs
            )
        }
        for conversation in conversations
    ]
    if any(len(prompt["prompt_token_ids"]) + args.cap > args.max_model_len for prompt in prompts):
        raise ValueError("A prompt plus declared cap exceeds max-model-len")
    (out / "prompts.json").write_text(json.dumps(prompts) + "\n")
    tick = time.monotonic()
    llm = LLM(
        model=args.student,
        tokenizer=args.student,
        hf_overrides=fusion_hf_overrides(args.teacher),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=not args.graph,
        compilation_config=decode_graph_config(args.max_num_seqs) if args.graph else None,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        kv_cache_memory_bytes=int(args.kv_gib * 1024**3),
        gpu_memory_utilization=0.9,
        enable_sleep_mode=True,
        logprobs_mode="processed_logprobs",
        worker_extension_cls="verl.workers.rollout.vllm_fusion.worker_extension.FusionWorkerExtension",
        attention_config={"backend": "TRITON_ATTN"},
    )
    startup = time.monotonic() - tick
    with (out / "results.jsonl").open("x") as stream:
        for alpha in args.alphas:
            audit = llm.collective_rpc("fusion_prepare", args=(alpha,))
            for repeat in range(-1, args.repeats):
                params = [
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.cap,
                        stop_token_ids=[151643, 151645],
                        logprobs=0,
                        seed=42 + i,
                    )
                    for i in range(args.batch)
                ]
                tick = time.monotonic()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.monotonic() - tick
                row = {
                    "alpha": alpha,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "startup_seconds": startup,
                    "audit": audit,
                    "responses": [],
                }
                for output in outputs:
                    response = output.outputs[0]
                    row["responses"].append(
                        {
                            "request_id": output.request_id,
                            "prompt_ids": output.prompt_token_ids,
                            "token_ids": response.token_ids,
                            "log_probs": [
                                v[t].logprob for t, v in zip(response.token_ids, response.logprobs, strict=True)
                            ],
                            "finish_reason": response.finish_reason,
                        }
                    )
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            "alpha": alpha,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "tokens": sum(len(r["token_ids"]) for r in row["responses"]),
                        }
                    ),
                    flush=True,
                )
        if args.skip_lifecycle_audit:
            stream.write(json.dumps({"lifecycle_audit": "not requested for this workload measurement"}) + "\n")
        else:
            before = llm.collective_rpc("fusion_weight_digests")
            llm.sleep(level=1)
            llm.wake_up()
            after = llm.collective_rpc("fusion_weight_digests")
            if before != after:
                raise RuntimeError("Fusion weights changed across sleep/wake")
            resumed = llm.generate(prompts, params, use_tqdm=False)
            same_tokens = [list(item.outputs[0].token_ids) for item in resumed] == [
                item["token_ids"] for item in row["responses"]
            ]
            stream.write(
                json.dumps(
                    {
                        "sleep_wake_audit": llm.collective_rpc("fusion_audit"),
                        "weight_digests_before": before,
                        "weight_digests_after": after,
                        "resumed_tokens_equal": same_tokens,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
