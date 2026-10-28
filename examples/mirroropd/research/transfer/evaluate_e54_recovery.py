# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed 64-question MATH diagnostic for E54 recovery, using the production cap.

Preparation uses a predeclared random seed without reading model outcomes.
Generate and grade are separate processes; grading never shares CUDA state.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import random
import statistics
import time
from pathlib import Path


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    rows = pq.read_table(args.data).to_pylist()
    assert len(rows) == 500
    indices = sorted(random.Random(20261028).sample(range(500), 64))
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for index in indices:
        row = rows[index]
        ids = tok.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        assert len(ids) <= 1024
        prompts.append(
            {
                "prompt_id": f"MATH-500:{index}",
                "data_source": "MATH-500",
                "prompt_ids": ids,
                "gts": row["reward_model"]["ground_truth"],
                "seed": 42 + index * 23,
            }
        )
    write(args.output / "prompts.json", prompts)
    write(
        args.output / "protocol.json",
        {
            "scope": "64 fixed MATH-500 questions, one sample each; diagnostic, not full benchmark",
            "selection_seed": 20261028,
            "indices": indices,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
            "response_cap": 31744,
            "max_context": 32768,
            "stop_ids": [151643, 151645],
            "template": "Base enable_thinking=False; unchanged from E54 training",
        },
    )


def generate(args):
    from vllm import LLM, SamplingParams

    prompts = json.loads(args.prompts.read_text())
    started = time.time()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.8,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        seed=42,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_ids"]} for r in prompts],
        sampling_params=[
            SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=-1,
                max_tokens=31744,
                stop_token_ids=[151643, 151645],
                ignore_eos=False,
                seed=r["seed"],
            )
            for r in prompts
        ],
    )
    assert len(outputs) == len(prompts) == 64
    with (args.output / "generated.jsonl").open("x") as stream:
        for prompt, result in zip(prompts, outputs, strict=True):
            assert list(result.prompt_token_ids) == prompt["prompt_ids"]
            response = result.outputs[0]
            assert response.finish_reason in ("stop", "length")
            record = prompt | {
                "model": str(args.model),
                "output": response.text,
                "response_ids": list(response.token_ids),
                "response_length": len(response.token_ids),
                "finish_reason": response.finish_reason,
                "stop_reason": response.stop_reason,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write(args.output / "generation-time.json", {"including_model_load_seconds": time.time() - started})


def grade(args):
    from .evaluate_qwen_moe import grade as grade_record
    from .evaluate_qwen_moe import init_grader

    records = [json.loads(line) for line in (args.output / "generated.jsonl").read_text().splitlines()]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
    ) as pool:
        results = list(pool.map(grade_record, records))
    with (args.output / "scored.jsonl").open("x") as stream:
        for record in results:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = {
        "responses": len(results),
        "accuracy": statistics.mean(float(r["acc"]) for r in results),
        "mean_length": statistics.mean(r["response_length"] for r in results),
        "median_length": statistics.median(r["response_length"] for r in results),
        "cap_count": sum(r["finish_reason"] == "length" for r in results),
        "boxed_format_fraction": statistics.mean(r["format_score"] for r in results),
        "repeat8_mean": statistics.mean(r["repeat8"] for r in results),
    }
    write(args.output / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "generate", "grade"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare" and (args.model is None or args.data is None):
        parser.error("prepare requires --model and --data")
    if args.action == "generate" and (args.model is None or args.prompts is None):
        parser.error("generate requires --model and --prompts")
    args.output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "generate": generate, "grade": grade}[args.action](args)


if __name__ == "__main__":
    main()
