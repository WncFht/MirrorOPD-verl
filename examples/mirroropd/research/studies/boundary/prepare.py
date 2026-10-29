# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze shared prompts and audit interfaces without loading GPU weights."""

import random
import re

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary.common import (
    ARMS,
    CACHE,
    DATASETS,
    GROUPS,
    MODELS,
    OUT,
    WORK,
    model_path,
)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    tokenizer = AutoTokenizer.from_pretrained(model_path("r1"), local_files_only=True)
    evaluations = []
    for dataset, expected in DATASETS.items():
        records = pq.read_table(WORK / "data/math" / dataset / "test.parquet").to_pylist()
        assert len(records) == expected
        for index, row in enumerate(records):
            ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            evaluations.append(
                dict(
                    data_source=dataset,
                    prompt_id=f"{dataset}:{index}",
                    seed_index=len(evaluations),
                    messages=row["prompt"],
                    prompt_ids=ids,
                    input=tokenizer.decode(ids, skip_special_tokens=False),
                    gts=row["reward_model"]["ground_truth"],
                )
            )
    dump(OUT / "evaluation-prompts.json", evaluations)
    evaluation_texts = {norm(r["input"]) for r in evaluations}
    rows = read(WORK / "wandb-logs/MirrorOPD-verl/analysis/twenty-batch-pass-20261027/prompts.json")
    used = {norm(r["prompt"]) for r in rows} | evaluation_texts
    candidates = pq.read_table(WORK / "data/math/DAPO/train.parquet").to_pylist()
    random.Random(2026102801).shuffle(candidates)
    replacements = []
    for row in rows:
        if row["split"] != "train" or norm(row["prompt"]) not in evaluation_texts:
            continue
        for candidate in candidates:
            prompt = tokenizer.apply_chat_template(candidate["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if norm(prompt) in used or not 24 <= len(ids) <= 384:
                continue
            replacements.append(dict(id=row["id"], previous_source_index=row["source_index"]))
            row.update(
                prompt=prompt,
                prompt_ids=ids,
                ground_truth=candidate["reward_model"]["ground_truth"],
                source_index=str(candidate["extra_info"].get("index")),
            )
            used.add(norm(prompt))
            break
        else:
            raise RuntimeError("No eligible DAPO replacement")
    assert len(rows) == 640 and len({norm(r["prompt"]) for r in rows}) == 640
    assert not {norm(r["prompt"]) for r in rows if r["split"] == "train"} & evaluation_texts
    for row in rows:
        assert tokenizer.encode(row["prompt"], add_special_tokens=False) == row["prompt_ids"]
    dump(OUT / "prompts.json", rows)
    audit = {}
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(model_path(name), local_files_only=True)
        cfg = AutoConfig.from_pretrained(model_path(name), local_files_only=True)
        inverse = {v: k for k, v in tok.get_vocab().items()}
        conflicts = [(v, k, inverse[v]) for k, v in tokenizer.get_vocab().items() if v in inverse and inverse[v] != k]
        assert not conflicts, (name, conflicts[:10])
        native = [
            tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
            for r in evaluations
        ]
        different = [i for i, (r, ids) in enumerate(zip(evaluations, native, strict=True)) if r["prompt_ids"] != ids]
        audit[name] = dict(
            path=str(model_path(name)),
            revision=MODELS[name][1],
            shared_id_conflicts=conflicts,
            template_different_indices=different,
            eos_token_id=tok.eos_token_id,
            config_eos=cfg.eos_token_id,
            vocab_size=cfg.vocab_size,
            native_prompt_max=max(map(len, native)),
            shared_prompt_max=max(len(r["prompt_ids"]) for r in evaluations),
        )
        if different:
            dump(
                OUT / "native-prompts" / f"{name}.json",
                [dict(r, prompt_ids=ids, input=tok.decode(ids)) for r, ids in zip(evaluations, native, strict=True)],
            )
    dump(OUT / "interface-audit.json", audit)
    dump(
        OUT / "protocol.json",
        dict(
            status="authorized-TB0-TB1-TB2",
            physical_gpus=[4, 5, 6, 7],
            models=MODELS,
            arms=ARMS,
            updates_per_arm=64,
            train_groups=list("ABCD"),
            scoring_groups=GROUPS,
            batch_size=16,
            reuse=16,
            horizon=1024,
            generation="frozen initial student; per-prompt RNG; BF16 backbone and FP32 head",
            optimizer=dict(name="AdamW", lr=1e-6, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01, clip=1),
            seed=42,
            evaluation=dict(samples=4, cap=31744, temperature=0.7, top_p=0.9, seed_formula="72000000+100*i+r"),
            replaced_training_prompts=replacements,
            historical_holdout_overlap="L prompts reused; development is not a pristine test set",
            cache=str(CACHE),
        ),
    )
    print("Prepared", len(rows), "prompts; replacements", replacements, "interfaces", audit, flush=True)


if __name__ == "__main__":
    main()
