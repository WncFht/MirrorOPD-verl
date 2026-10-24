#!/usr/bin/env python3
"""Real vLLM stopping and Continuous Token mask checks; no training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--family", choices=["deepseek", "qwen3"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forced-only", action="store_true")
    parser.add_argument("--generation-eos", type=int, nargs="+")
    args = parser.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    config = AutoConfig.from_pretrained(str(args.model))
    generation = json.loads((args.model / "generation_config.json").read_text())
    template_kwargs = {"enable_thinking": False} if args.family == "qwen3" else {}
    builder = create_continuous_token_builder(
        tokenizer, model_family=args.family, hf_model_type=config.model_type, chat_template_kwargs=template_kwargs
    )
    extra_stop = [tokenizer.convert_tokens_to_ids("<|im_end|>")] if args.family == "qwen3" else []
    result = {
        "label": args.label,
        "model": str(args.model),
        "generation_config": generation,
        "tokenizer_eos": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "additional_chat_stop_ids": extra_stop,
        "forced": [],
        "natural": [],
        "scope": "vLLM eager, actual model weights, no optimizer updates; max 1024 generated tokens",
    }
    overrides = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}
    if args.generation_eos is not None:
        overrides["eos_token_id"] = args.generation_eos
        saved = GenerationConfig.from_pretrained(str(args.model))
        saved.eos_token_id = args.generation_eos
        config_dir = args.output.parent / (args.label + "-generation")
        saved.save_pretrained(config_dir)
        assert GenerationConfig.from_pretrained(config_dir).eos_token_id == args.generation_eos
        result["exported_generation_config"] = str(config_dir)
    result["generation_overrides"] = overrides
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.30,
        max_model_len=4096,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=42,
        attention_config={"backend": "TRITON_ATTN"},
        override_generation_config=overrides,
    )
    prompt = builder.build_initial_tokens([{"role": "user", "content": "What is 1+1?"}])
    eos = args.generation_eos or generation["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    forced_ids = sorted(set([tokenizer.eos_token_id, 17] + extra_stop + eos))
    for policy in ("native", "chat"):
        for target in forced_ids:
            params = SamplingParams(
                temperature=0,
                max_tokens=4,
                allowed_token_ids=[target],
                detokenize=False,
                logprobs=0,
                stop_token_ids=extra_stop if policy == "chat" else [],
            )
            output = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0].outputs[0]
            ids = list(output.token_ids)
            lp = [float(x[t].logprob) for x, t in zip(output.logprobs, ids, strict=True)]
            merge = builder.merge_assistant_tokens(prompt, ids)
            mask, aligned_lp = builder.align_response_metadata(merge, [], [], assistant_logprobs=lp)
            assert merge.token_ids == prompt + ids
            assert mask == [1] * len(ids) and aligned_lp == lp
            result["forced"].append(
                {
                    "policy": policy,
                    "forced_id": target,
                    "token": tokenizer.convert_ids_to_tokens(target),
                    "output_ids": ids,
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "response_mask": mask,
                    "padded_ids": ids + [tokenizer.pad_token_id] * 2,
                    "correct_padded_mask": mask + [0, 0],
                    "incorrect_id_based_mask": [
                        int(x != tokenizer.pad_token_id) for x in ids + [tokenizer.pad_token_id] * 2
                    ],
                }
            )
    print(f"{args.label}: forced checks complete", flush=True)
    if args.forced_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return
    simple = [
        "What is 1+1? Reply with only the answer.",
        "What is 7*8? Reply with only the answer.",
        "What is 100-37? Reply with only the answer.",
        "Solve x+3=5. Reply with only x.",
        "What is 12/4? Reply with only the answer.",
        "What is the square of 9? Reply with only the answer.",
        "What is the next prime after 7? Reply with only the answer.",
        "What is 0+0? Reply with only the answer.",
    ]
    cases = [("simple", i, [{"role": "user", "content": q}]) for i, q in enumerate(simple)]
    table = pq.read_table(args.dataset, columns=["prompt"])
    indices = np.random.default_rng(42).choice(len(table), size=16, replace=False).tolist()
    cases.extend(("dapo", i, table["prompt"][i].as_py()) for i in indices)
    requests, params, meta = [], [], []
    for case_id, (source, index, messages) in enumerate(cases):
        ids = builder.build_initial_tokens(messages)
        assert len(ids) <= 1024
        for policy in ("native", "chat"):
            requests.append({"prompt_token_ids": ids})
            params.append(
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=1024,
                    seed=42 + case_id,
                    detokenize=False,
                    stop_token_ids=extra_stop if policy == "chat" else [],
                )
            )
            meta.append({"source": source, "index": index, "policy": policy, "seed": 42 + case_id})
    outputs = llm.generate(requests, params, use_tqdm=True)
    for item, request, output in zip(meta, requests, outputs, strict=True):
        out = output.outputs[0]
        ids = list(out.token_ids)
        item.update(
            prompt_ids=request["prompt_token_ids"],
            response_ids=ids,
            response_length=len(ids),
            finish_reason=out.finish_reason,
            stop_reason=out.stop_reason,
            im_end_positions=[i for i, x in enumerate(ids) if x in extra_stop],
            text=tokenizer.decode(ids, skip_special_tokens=False),
        )
        result["natural"].append(item)
    result["summary"] = {}
    for source in ("simple", "dapo"):
        for policy in ("native", "chat"):
            rows = [x for x in result["natural"] if x["source"] == source and x["policy"] == policy]
            result["summary"][f"{source}/{policy}"] = {
                "count": len(rows),
                "mean_length": float(np.mean([x["response_length"] for x in rows])),
                "stopped": sum(x["finish_reason"] == "stop" for x in rows),
                "length_capped": sum(x["finish_reason"] == "length" for x in rows),
                "has_im_end": sum(bool(x["im_end_positions"]) for x in rows),
            }
    result["paired_prefix_matches"] = []
    for case_id in range(len(cases)):
        native, chat = result["natural"][2 * case_id : 2 * case_id + 2]
        ids = native["response_ids"]
        expected = ids[: native["im_end_positions"][0] + 1] if native["im_end_positions"] else ids
        result["paired_prefix_matches"].append(expected == chat["response_ids"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
