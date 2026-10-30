# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze disjoint questions and an auditable E20 sampler order without touching source data."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint, receipt
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.data import question_messages
from examples.mirroropd.research.studies.prefix_fusion.protocol import (
    LONG_GROUPS,
    MAIN_GROUPS,
    PROTOCOL,
    protocol_identity,
)


def user_text(messages):
    return re.sub(r"\s+", " ", "\n".join(m["content"] for m in messages if m["role"] == "user")).strip()


def is_holdout_group(group):
    return re.fullmatch(r"(?:M|Code|IF)\d{2}", group) is not None


def select_training(rows, heldout, benchmarks):
    excluded = {user_text(question_messages(r)) for r in heldout}
    excluded |= {user_text(r["messages"]) for r in benchmarks}
    selected, seen, removed = [], set(), Counter()
    for index, row in enumerate(rows):
        key = user_text(row["prompt"])
        if not key:
            raise ValueError(f"Empty training question at source row {index}")
        if key in excluded:
            removed["heldout_or_task_overlap"] += 1
        elif key in seen:
            removed["duplicate_training_question"] += 1
        else:
            selected.append(index)
            seen.add(key)
    if seen & excluded:
        raise AssertionError("Training/evaluation split is not disjoint")
    return selected, dict(removed)


def prepare(workspace, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from torchdata.stateful_dataloader.sampler import RandomSampler
    from transformers import AutoTokenizer

    workspace, out = Path(workspace).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = workspace / "data/math/DAPO/train.parquet"
    bank_source = workspace / "wandb-logs/MirrorOPD-verl/analysis/teacher-boundary-20261028/prompts.json"
    task_source = bank_source.with_name("evaluation-prompts.json")
    source_hashes = {str(p): file_hash(p) for p in (source, bank_source, task_source)}
    table = pq.read_table(source)
    original = table.to_pylist()
    heldout = [r for r in read(bank_source) if is_holdout_group(r["group"])]
    benchmarks = read(task_source)
    assert len(heldout) == 320 and len(benchmarks) == 630
    assert len({user_text(question_messages(r)) for r in heldout}) == 320
    if {user_text(question_messages(r)) for r in heldout} & {user_text(r["messages"]) for r in benchmarks}:
        raise ValueError("Fixed-risk development questions overlap the task benchmark")
    selected, removed = select_training(original, heldout, benchmarks)
    tokenizer = AutoTokenizer.from_pretrained(workspace / "model/DeepSeek-R1-Distill-Qwen-1.5B", local_files_only=True)

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)

    kept, lengths = [], []
    for index in selected:
        length = len(render(original[index]["prompt"]))
        if length <= PROTOCOL["training"]["prompt_cap"]:
            kept.append(index)
            lengths.append(length)
    removed["overlong_prompt"] = len(selected) - len(kept)
    if len(kept) < 40 * 64:
        raise ValueError("Insufficient distinct training questions for forty batches")
    clean = table.take(pa.array(kept, type=pa.int64()))
    pq.write_table(clean, out / "train.parquet")
    questions = []
    for row in heldout:
        messages = question_messages(row)
        ids = render(messages)
        if len(ids) > 1024:
            raise ValueError("Evaluation question exceeds the shared task prompt budget")
        questions.append(
            dict(
                id=row["id"],
                group=row["group"],
                domain=row["domain"],
                split="heldout",
                messages=messages,
                prompt_ids=ids,
                ground_truth=row.get("ground_truth"),
                source_index=row.get("source_index"),
                question_sha256=fingerprint(user_text(messages)),
                primary=row["group"] in MAIN_GROUPS,
                long=row["group"] in LONG_GROUPS,
            )
        )
    tasks = []
    for row in benchmarks:
        ids = render(row["messages"])
        if len(ids) > 1024:
            raise ValueError("Task prompt does not fit the declared 32K evaluation window")
        tasks.append(dict(row, prompt_ids=ids, input=tokenizer.decode(ids, skip_special_tokens=False)))
    generator = torch.Generator().manual_seed(PROTOCOL["training"]["seed"])
    order = list(RandomSampler(range(len(kept)), generator=generator))[: 40 * 64]
    training_order = [
        dict(
            outer=i // 64 + 1,
            row=i % 64,
            clean_index=k,
            source_row=kept[k],
            source_index=original[kept[k]].get("extra_info", {}).get("index"),
            question_sha256=fingerprint(user_text(original[kept[k]]["prompt"])),
        )
        for i, k in enumerate(order)
    ]
    dump(out / "protocol.json", PROTOCOL)
    dump(out / "heldout-questions.json", questions)
    dump(out / "task-prompts.json", tasks)
    dump(out / "training-order.json", training_order)
    if any(file_hash(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError("Source data changed during preparation")
    inputs = dict(
        protocol=protocol_identity(),
        source=file_hash(source),
        heldout_source=file_hash(bank_source),
        task_source=file_hash(task_source),
        tokenizer_chat_template=fingerprint(tokenizer.chat_template),
    )
    artifacts = {
        name: file_hash(out / name)
        for name in [
            "train.parquet",
            "protocol.json",
            "heldout-questions.json",
            "task-prompts.json",
            "training-order.json",
        ]
    }
    receipt(
        out,
        inputs,
        source_rows=len(original),
        training_pool=len(kept),
        removed=removed,
        primary_counts=dict(Counter(r["domain"] for r in questions if r["primary"])),
        long_questions=sum(r["long"] for r in questions),
        reserved_holdout=len(questions),
        task_questions=len(tasks),
        training_questions=2560,
        updates_per_arm=640,
        conditions=12,
        total_updates=7680,
        artifacts=artifacts,
        dedup_scope="exact whitespace-normalized user messages; not semantic dedup",
        sampler="torchdata.stateful_dataloader.sampler.RandomSampler; independent seed42 generator",
        torch_version=torch.__version__,
        max_train_prompt=max(lengths),
    )
    print(json.dumps(read(out / "complete.json"), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.workspace, args.out)


if __name__ == "__main__":
    main()
