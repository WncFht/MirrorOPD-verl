#!/usr/bin/env python3
"""Export every math validation prompt using the existing rollout token builder."""

import argparse
import json
from pathlib import Path

DATASETS = {"MATH-500": 500, "AMC23": 40, "AIME25": 30}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_rows(data_root):
    import pyarrow.parquet as pq

    rows = []
    for name, expected in DATASETS.items():
        source = data_root / "math" / name / "test.parquet"
        table = pq.read_table(source).to_pylist()
        if len(table) != expected:
            raise ValueError(f"{source}: expected {expected} rows, got {len(table)}")
        rows.extend({"dataset": name, "row": index, "data": row} for index, row in enumerate(table))
    return rows


def render_rows(rows, tokenizer, builder, prompt_cap, samples, response_cap):
    requests, metadata = [], []
    for item in rows:
        token_ids = list(builder.build_initial_tokens(item["data"]["prompt"]))
        if len(token_ids) > prompt_cap:
            raise ValueError(f"Overlong prompt {item['dataset']}/{item['row']}: {len(token_ids)} > {prompt_cap}")
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != token_ids:
            raise ValueError(f"Prompt token roundtrip mismatch: {item['dataset']}/{item['row']}")
        metadata.append(
            {
                "dataset": item["dataset"],
                "row": item["row"],
                "prompt_tokens": len(token_ids),
                "token_ids": token_ids,
                "ground_truth": item["data"]["reward_model"]["ground_truth"],
                "client_default_length": len(tokenizer.encode(prompt)),
            }
        )
        requests.extend({"prompt": prompt, "output_tokens": response_cap} for _ in range(samples))
    return requests, metadata


def prepare(plan, model_root, data_root, output):
    from transformers import AutoConfig, AutoTokenizer

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    rows = load_rows(data_root)
    expected = sum(DATASETS.values()) * plan["samples_per_prompt"]
    reports = {}
    for profile in plan["models"]:
        path = model_root / profile["directory"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        builder = create_continuous_token_builder(
            tokenizer,
            model_family=profile["family"],
            hf_model_type=cfg.model_type,
            chat_template_kwargs={"enable_thinking": False} if profile["family"] == "qwen3" else {},
        )
        requests, metadata = render_rows(
            rows,
            tokenizer,
            builder,
            profile["prompt_cap"],
            plan["samples_per_prompt"],
            plan["response_cap"],
        )
        assert len(requests) == expected
        target = output / f"{profile['id']}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            for request in requests:
                stream.write(json.dumps(request, ensure_ascii=False) + "\n")
        temporary.replace(target)
        write_json(output / f"{profile['id']}-metadata.json", metadata)
        reports[profile["id"]] = {
            "model": str(path),
            "dataset": str(target),
            "requests": expected,
            "prompts": len(rows),
            "max_prompt_tokens": max(x["prompt_tokens"] for x in metadata),
            "tokenizer_width": len(tokenizer),
            "model_vocab_size": cfg.vocab_size,
            "client_length_mismatches": sum(x["client_default_length"] != x["prompt_tokens"] for x in metadata),
        }
        print(json.dumps({"prepared": profile["id"], **reports[profile["id"]]}), flush=True)
    write_json(output / "preparation.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        json.loads(args.plan.read_text()), args.model_root.resolve(), args.data_root.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()
