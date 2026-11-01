# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse frozen question texts and training split; build new native evaluation banks separately."""

import argparse
import random
import shutil
from pathlib import Path

from examples.mirroropd.launchers.qwen_prefix import BENCHMARKS, LEGACY_PROTOCOL, SOURCE_PROTOCOL
from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read


def prepare(source, out, *, protocol=LEGACY_PROTOCOL, tokenizer_path=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.mirroropd.research.studies.prefix_fusion.data import user_text
    from verl.trainer.evaluation.prepare import QUESTION, question

    receipt = read(source / "complete.json")
    for name, digest in receipt["artifacts"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Prepared legacy question artifact changed: {name}")
    if protocol not in (LEGACY_PROTOCOL, SOURCE_PROTOCOL):
        raise ValueError(f"Unknown Qwen preparation protocol: {protocol}")
    expanded = protocol == SOURCE_PROTOCOL
    heldout_pool = [r for r in read(source / "heldout-questions.json") if r["domain"] == "math"]
    heldout = (
        random.Random(42).sample(sorted(heldout_pool, key=lambda r: r["id"]), 128)
        if expanded
        else [r for r in heldout_pool if r["primary"]]
    )
    tasks = read(source / "task-prompts.json")
    assert len(heldout) == (128 if expanded else 64) and len(tasks) == 630
    reserved = {user_text(r["messages"]) for r in heldout + tasks}
    training = pq.read_table(source / "train.parquet")
    training_texts = [user_text(r["prompt"]) for r in training.select(["prompt"]).to_pylist()]
    if len(set(training_texts)) != len(training_texts):
        raise ValueError("Prepared training pool contains duplicate question texts")
    for row in training.select(["prompt"]).to_pylist():
        if user_text(row["prompt"]) in reserved:
            raise ValueError("Training text overlaps native evaluation questions")
    order = None
    max_prompt_tokens = None
    if expanded:
        if len(training) < 12800:
            raise ValueError("The b128/r16/o100 budget requires 12800 distinct questions")
        if tokenizer_path is None:
            raise ValueError("The b128 study requires a local tokenizer for prompt-length preflight")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        order = random.Random(42).sample(range(len(training)), 12800)
        training = training.take(pa.array(order))
        messages = [r["prompt"] for r in training.select(["prompt"]).to_pylist()]
        lengths = [
            len(
                tokenizer.apply_chat_template(
                    m, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
                )
            )
            for m in messages
        ]
        max_prompt_tokens = max(lengths)
        if max_prompt_tokens > 4096:
            raise ValueError("A selected training question exceeds prompt cap4096; no silent filtering")
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, count in (("source", len(heldout)), *BENCHMARKS):
        rows = heldout if name == "source" else [r for r in tasks if r["data_source"] == name]
        if len(rows) != count:
            raise ValueError(f"Incorrect question count for {name}")
        converted = []
        for index, row in enumerate(rows):
            messages = row["messages"]
            truth = row.get("ground_truth", row.get("gts"))
            value = question(
                name,
                "math",
                user_text(messages),
                ground_truth="" if truth is None else truth,
                prompt=messages,
            )
            value["id"] = f"{name}:{index}"
            converted.append(value)
        path = out / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(converted, schema=QUESTION), path)
        files[path.name] = file_hash(path)
    if expanded:
        pq.write_table(training, out / "train.parquet")
        dump(out / "training-order.json", dict(seed=42, source_indices=order, rows=12800, shuffle=False))
        files["training-order.json"] = file_hash(out / "training-order.json")
    else:
        shutil.copyfile(source / "train.parquet", out / "train.parquet")
    files["train.parquet"] = file_hash(out / "train.parquet")
    dump(
        out / "complete.json",
        dict(
            files=files,
            source_receipt=file_hash(source / "complete.json"),
            protocol=protocol,
            training_rows=len(training),
            source_count=len(heldout),
            max_prompt_tokens=max_prompt_tokens,
            scope="Same frozen training/question texts; new native sampling, banks, scoring and aggregation contract",
        ),
    )


def configs(path, out):
    from omegaconf import OmegaConf, open_dict

    out.mkdir(parents=True, exist_ok=False)
    for producer, seqs, kv in [("initial", 16, 64), ("teacher", 8, 40)]:
        config = OmegaConf.load(path)
        with open_dict(config.actor_rollout_ref.rollout):
            config.actor_rollout_ref.rollout.max_num_seqs = seqs
            config.actor_rollout_ref.rollout.cudagraph_capture_sizes = [n for n in [1, 2, 4, 8, 16] if n <= seqs]
            config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes = kv * 2**30
        OmegaConf.save(config, out / f"{producer}.yaml")
    dump(
        out / "complete.json",
        dict(
            source_config=file_hash(path),
            overrides={"initial": {"seqs": 16, "kv_gib": 64}, "teacher": {"seqs": 8, "kv_gib": 40}},
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["questions", "configs"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", choices=[LEGACY_PROTOCOL, SOURCE_PROTOCOL], default=LEGACY_PROTOCOL)
    parser.add_argument("--tokenizer", type=Path, help="Local learner tokenizer, required by source_b128_v1")
    args = parser.parse_args()
    if args.action == "questions":
        prepare(args.source, args.out, protocol=args.protocol, tokenizer_path=args.tokenizer)
    else:
        configs(args.source, args.out)


if __name__ == "__main__":
    main()
