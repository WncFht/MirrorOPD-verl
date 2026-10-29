# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Teacher training-question lengths and frozen student continuation beyond cap.

All generation uses E54's common prompt, temperature 1 and top-p 1. Shorter
budgets are censored prefixes of the same long draw. Continuations preserve
the original 7168 response IDs but use a fresh, explicitly recorded RNG seed.
This is inference, not a claim about retraining with a larger response cap.
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import random
import time
from pathlib import Path

CAPS = (7168, 16384, 31744)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = pq.read_table(args.data).to_pylist()
    order = list(range(len(rows)))
    random.Random(20261028).shuffle(order)
    teacher, excluded = [], 0
    for idx in order:
        row = rows[idx]
        ids = tok.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        if len(ids) > 1024:
            excluded += 1
            continue
        teacher.append(
            dict(
                request_id=f"train-{idx}",
                cohort="random_training",
                prompt_ids=ids,
                prefix_ids=[],
                data_source="DAPO",
                gts=row["reward_model"]["ground_truth"],
                seed=42 + idx * 23,
            )
        )
        if len(teacher) == 320:
            break
    assert len(teacher) == 320
    observed = []
    for outer, model_label in [(6, "student5"), (11, "student10")]:
        raw = read(args.run / "continue-r16/debug" / f"rollout-{outer}.jsonl")
        scored = {r["uid"]: r for r in read(args.run / "continue-r16/rollouts" / f"{outer}.jsonl")}
        assert len(raw) == len(scored) == 64
        continuation = []
        for i, row in enumerate(raw):
            gt = scored[row["sample_key"]]["gts"]
            assert row["score"] == scored[row["sample_key"]]["score"]
            question_id = f"observed-{outer}-{i}"
            question = dict(
                request_id=question_id,
                cohort=f"observed_outer{outer}",
                prompt_ids=row["prompt_ids"],
                prefix_ids=[],
                data_source="DAPO",
                gts=gt,
                seed=20261028 + outer * 1000 + i,
                original_score=row["score"],
                original_length=row["length"],
                original_stopped=row["stopped_on_eos"],
                sample_key=row["sample_key"],
            )
            teacher.append(question)
            observed.append(question)
            if row["at_response_cap"]:
                assert len(row["response_ids"]) == 7168 and not row["stopped_on_eos"]
                continuation.append(question | {"prefix_ids": row["response_ids"], "model_label": model_label})
        write(args.output / f"{model_label}-requests.json", continuation)
    assert len(teacher) == 448
    write(args.output / "teacher-requests.json", teacher)
    write(args.output / "observed-student.json", observed)
    write(
        args.output / "protocol.json",
        {
            "training_rows": len(rows),
            "random_teacher_requests": 320,
            "matched_teacher_requests": 128,
            "selection_seed": 20261028,
            "excluded_overlong_while_sampling": excluded,
            "prompt_cap": 1024,
            "response_caps": CAPS,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "common_prompt": "Base tokenizer with enable_thinking=False, unchanged from training",
            "student_scope": "All cap-truncated trajectories in outer 6 and 11; same frozen generating checkpoint",
            "continuation_rng": "New recorded seed after original 7168 IDs; not restored original sampler RNG",
            "cap_comparison": "Censor the same 31744-cap draw at 7168 and 16384; no separate short-cap resampling",
        },
    )
    print(
        json.dumps(
            {
                "teacher": len(teacher),
                "student5": len(read_json(args.output / "student5-requests.json")),
                "student10": len(read_json(args.output / "student10-requests.json")),
            }
        ),
        flush=True,
    )


def read_json(path):
    return json.loads(path.read_text())


def generate(args):
    from vllm import LLM, SamplingParams

    requests = read_json(args.requests)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.8,
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        seed=42,
    )
    started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": r["prompt_ids"] + r["prefix_ids"]} for r in requests],
        sampling_params=[
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=31744 - len(r["prefix_ids"]),
                stop_token_ids=[151643, 151645],
                ignore_eos=False,
                seed=r["seed"],
                detokenize=False,
            )
            for r in requests
        ],
    )
    assert len(outputs) == len(requests)
    with (args.output / "generated.jsonl").open("x") as stream:
        for request, output in zip(requests, outputs, strict=True):
            assert list(output.prompt_token_ids) == request["prompt_ids"] + request["prefix_ids"]
            response = output.outputs[0]
            ids = request["prefix_ids"] + list(response.token_ids)
            assert response.finish_reason in ("stop", "length") and len(ids) <= 31744
            if response.finish_reason == "length":
                assert len(ids) == 31744
            stream.write(
                json.dumps(
                    request
                    | {
                        "model": str(args.model),
                        "response_ids": ids,
                        "response_length": len(ids),
                        "finish_reason": response.finish_reason,
                        "stop_reason": response.stop_reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write(
        args.output / "generation.json",
        {
            "responses": len(outputs),
            "generation_seconds": time.time() - started,
            "tensor_parallel": args.tp,
            "max_num_seqs": args.max_num_seqs,
        },
    )


def grade(args):
    from transformers import AutoTokenizer

    from .evaluate_qwen_moe import grade as grade_record
    from .evaluate_qwen_moe import init_grader

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generated = read(args.output / "generated.jsonl")
    tasks = []
    for row in generated:
        for cap in CAPS:
            ids = row["response_ids"][:cap]
            tasks.append(
                row
                | {
                    "cap": cap,
                    "response_ids": ids,
                    "response_length": len(ids),
                    "output": tok.decode(ids, skip_special_tokens=True),
                    "stopped_within_cap": row["finish_reason"] == "stop" and len(row["response_ids"]) <= cap,
                }
            )
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=init_grader
        ) as pool,
        (args.output / "scored.jsonl").open("x") as stream,
    ):
        for result in pool.map(grade_record, tasks):
            compact = {
                k: v for k, v in result.items() if k not in ["response_ids", "prefix_ids", "prompt_ids", "output"]
            }
            stream.write(json.dumps(compact, ensure_ascii=False) + "\n")
    print(f"Graded {len(tasks)} censored prefixes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "generate", "grade"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare" and (args.data is None or args.run is None):
        parser.error("prepare requires --data and --run")
    if args.action == "generate" and args.requests is None:
        parser.error("generate requires --requests")
    {"prepare": prepare, "generate": generate, "grade": grade}[args.action](args)


if __name__ == "__main__":
    main()
