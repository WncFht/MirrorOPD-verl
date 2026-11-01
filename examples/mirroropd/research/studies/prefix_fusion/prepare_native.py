# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse frozen question texts and training split; build new native evaluation banks separately."""

import argparse
import shutil
from pathlib import Path

from examples.mirroropd.launchers.qwen_prefix import BENCHMARKS
from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read


def prepare(source, out):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.mirroropd.research.studies.prefix_fusion.data import user_text
    from verl.trainer.evaluation.prepare import QUESTION, question

    receipt = read(source / "complete.json")
    for name, digest in receipt["artifacts"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Prepared legacy question artifact changed: {name}")
    heldout = [r for r in read(source / "heldout-questions.json") if r["domain"] == "math" and r["primary"]]
    tasks = read(source / "task-prompts.json")
    assert len(heldout) == 64 and len(tasks) == 630
    reserved = {user_text(r["messages"]) for r in heldout + tasks}
    for row in pq.read_table(source / "train.parquet", columns=["prompt"]).to_pylist():
        if user_text(row["prompt"]) in reserved:
            raise ValueError("Training text overlaps native evaluation questions")
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, count in (("source", 64), *BENCHMARKS):
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
    shutil.copyfile(source / "train.parquet", out / "train.parquet")
    files["train.parquet"] = file_hash(out / "train.parquet")
    dump(
        out / "complete.json",
        dict(
            files=files,
            source_receipt=file_hash(source / "complete.json"),
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
    args = parser.parse_args()
    (prepare if args.action == "questions" else configs)(args.source, args.out)


if __name__ == "__main__":
    main()
