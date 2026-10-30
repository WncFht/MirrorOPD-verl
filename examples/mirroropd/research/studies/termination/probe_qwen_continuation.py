# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded frozen-teacher generation on questions and recorded student prefixes."""

import argparse
import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer

from examples.mirroropd.assets.paths import registered_model
from examples.mirroropd.research.studies.termination.probe_qwen_termination import MODELS, STOP, text_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model", choices=["teacher_4b", "teacher_30b"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(
        registered_model(MODELS["initial_student"], root=args.models), local_files_only=True
    )
    cases = json.loads((args.data / "probes/cases.json").read_text())
    suffix = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    llm = LLM(
        model=str(registered_model(MODELS[args.model], root=args.models)),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.88 if args.model == "teacher_30b" else 0.4,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        seed=42,
        attention_config={"backend": "TRITON_ATTN"},
    )
    destination = args.data / "probes" / f"{args.model}-continuations.jsonl"
    with destination.open("w") as stream:
        requests = []
        for case in cases:
            if case["category"] not in ["archived_terminal", "e41_correct_end"]:
                continue
            requests.append(
                dict(
                    kind="end_continuation",
                    id=case["id"],
                    variant="training_prompt",
                    tokens=case["prompt_ids"] + case["response_ids"],
                    cap=128,
                    temperature=0.0,
                )
            )
        for case in cases:
            if case["category"] == "e54_late_loop":
                requests.append(
                    dict(
                        kind="loop_continuation",
                        id=case["id"],
                        variant="training_prompt",
                        tokens=case["prompt_ids"] + case["response_ids"][:4096],
                        cap=128,
                        temperature=0.0,
                    )
                )
        with (args.data / "E54/rollouts/1.jsonl").open() as source:
            originals = sorted([json.loads(line) for line in source], key=lambda row: row["input"])[:16]
        for i, row in enumerate(originals):
            case = text_case(tok, row, "question", f"question-{i}")
            for variant in ["training_prompt", "without_empty_think"]:
                prompt = case["prompt_ids"]
                if variant == "without_empty_think":
                    assert prompt[-len(suffix) :] == suffix
                    prompt = prompt[: -len(suffix)]
                requests.append(
                    dict(
                        kind="question",
                        id=case["id"],
                        variant=variant,
                        tokens=prompt,
                        cap=7168,
                        temperature=1.0,
                        ground_truth=row["gts"],
                        seed=42 + i,
                    )
                )
        for kind in ["end_continuation", "loop_continuation", "question"]:
            subset = [r for r in requests if r["kind"] == kind]
            params = [
                SamplingParams(
                    temperature=r["temperature"],
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=r["cap"],
                    stop_token_ids=STOP,
                    seed=r.get("seed", 42),
                    detokenize=False,
                )
                for r in subset
            ]
            start = time.time()
            outputs = llm.generate([{"prompt_token_ids": r["tokens"]} for r in subset], params, use_tqdm=False)
            for request, output in zip(subset, outputs, strict=True):
                result = output.outputs[0]
                record = {k: v for k, v in request.items() if k != "tokens"}
                record.update(
                    model=args.model,
                    prompt_ids=request["tokens"],
                    response_ids=list(result.token_ids),
                    response_length=len(result.token_ids),
                    output=tok.decode(result.token_ids, skip_special_tokens=True),
                    finish_reason=result.finish_reason,
                    stop_reason=result.stop_reason,
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(args.model, kind, "completed", len(subset), "seconds", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
