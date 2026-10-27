#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze 40 disjoint batches for a single-pass cross-state transfer study."""

import argparse
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from ..common import native as core
from ..common.protocol import LONG_PASS


def prepare(a):
    assert not (a.out / "protocol.json").exists(), "Choose a new output directory."
    tokenizer = AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    excluded = set()
    earlier = [
        a.repo / "examples/mirroropd/docs/experiments/polaris-transfer/data/state-transfer-20261027/prompts.json",
        a.repo / "examples/mirroropd/docs/experiments/polaris-transfer/data/intersection-study-20261027/prompts.json",
        a.old / "bank.json",
    ] + [a.old / f"rich-shard-{i}.json" for i in range(4)]
    for path in earlier:
        for row in core.read(path):
            value = row.get("prompt", row.get("text"))
            if isinstance(value, str):
                excluded.add(norm(value))
    rng = random.Random(2026102711)
    records = []
    for domain, file, count in [
        ("math", "math/DAPO/train.parquet", 512),
        ("code", "code/TACO/train.parquet", 64),
        ("if", "if/MultiIF/train.parquet", 64),
    ]:
        candidates = pq.read_table(a.data / file, columns=["prompt", "extra_info", "reward_model"]).to_pylist()
        rng.shuffle(candidates)
        accepted = 0
        for row in candidates:
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not 24 <= len(ids) <= 384 or norm(prompt) in excluded:
                continue
            if (
                any("\u4e00" <= c <= "\u9fff" for c in prompt)
                or "http" in prompt
                or "interactive problem" in prompt.lower()
            ):
                continue
            if domain == "if" and "English" not in str(row["extra_info"]):
                continue
            i = len(records)
            if domain == "math" and accepted < 320:
                split, group, batch_index = "train", chr(65 + accepted // 16), accepted // 16
            elif domain == "math":
                split, group, batch_index = "heldout", f"M{(accepted - 320) // 16 + 1:02d}", 20 + (accepted - 320) // 16
            elif domain == "code":
                split, group, batch_index = "heldout", f"Code{accepted // 16 + 1:02d}", 32 + accepted // 16
            else:
                split, group, batch_index = "heldout", f"IF{accepted // 16 + 1:02d}", 36 + accepted // 16
            records.append(
                dict(
                    id=i,
                    domain=domain,
                    split=split,
                    group=group,
                    batch_index=batch_index,
                    prompt=prompt,
                    prompt_ids=ids,
                    source="long-pass-s0-2026102711",
                    source_index=str(row["extra_info"].get("index")),
                    ground_truth=row["reward_model"]["ground_truth"],
                )
            )
            excluded.add(norm(prompt))
            accepted += 1
            if accepted == count:
                break
        assert accepted == count, (domain, accepted)
    assert len(records) == LONG_PASS.examples
    core.dump(a.out / "prompts.json", records)
    core.dump(
        a.out / "protocol.json",
        dict(
            selection_seed=2026102711,
            training_seed=LONG_PASS.seed,
            models=core.NAMES,
            total_batches=LONG_PASS.total_batches,
            prompts_per_batch=LONG_PASS.batch_size,
            train_batches=LONG_PASS.train_batches,
            heldout_batches=dict(math=12, code=4, instruction_following=4),
            train_order=list("ABCDEFGHIJKLMNOPQRST"),
            visits_to_each_training_batch=1,
            consecutive_updates_per_batch=LONG_PASS.reuse,
            optimizer_updates_per_teacher=LONG_PASS.updates,
            cap=LONG_PASS.horizon,
            lr=LONG_PASS.lr,
            betas=list(LONG_PASS.betas),
            weight_decay=LONG_PASS.weight_decay,
            clip=LONG_PASS.clip,
            loss="native full-vocabulary RKL, all response positions, fixed denominator 16*1024",
            rollout="frozen initial-student BF16 HF generation; T=1, top_p=1, top_k=0",
            precision="FP32 student master/head; BF16 backbone autocast; BF16 frozen teacher and FP32 head projection",
            state_evaluation="initial and after every batch: all 40 fixed batches, same complete token prefixes",
            gradient_probe=(
                "before the first update of every batch: read-only M01-M04 held-out "
                "gradient; compare raw train gradient, actual AdamW displacement, and "
                "finite held-out risk change"
            ),
            fresh_batches=[0, 5, 10, 15, 20],
            fresh_groups=["M01", "M02"],
            fresh_base_seeds=[77101, 77102],
            fresh_sampling="unrestricted current student, H1024; seed offset +1000*rank",
            checkpoint_batches=[1, 5, 10, 15, 20],
            checkpoint_contents="HF student, tokenizer, full AdamW state and RNG",
            distribution=(
                "two independent torchrun groups; two GPUs per teacher; identical input/order and one pass each"
            ),
            limitations="A longer optimization run, not a longer response horizon or an E48 checkpoint replay.",
            previous_inputs_excluded=[str(p) for p in earlier],
        ),
    )
    print("Prepared 640 prompts: 20 train math + 12 held math + 4 code + 4 IF batches.", flush=True)


@torch.no_grad()
def generate(a):
    destination = a.out / f"bank-shard-{a.shard}.json"
    assert not destination.exists(), "Generation shard already exists."
    records = core.read(a.out / "prompts.json")[160 * a.shard : 160 * (a.shard + 1)]
    model = core.load(a, "student")
    tokenizer = AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    result = []
    for start in range(0, len(records), 32):
        batch = records[start : start + 32]
        seed = 92000 + 160 * a.shard + start
        torch.manual_seed(seed)
        inputs = tokenizer(
            [r["prompt"] for r in batch], padding=True, return_tensors="pt", add_special_tokens=False
        ).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=LONG_PASS.horizon,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        for row, tokens in zip(batch, outputs[:, inputs.input_ids.shape[1] :].tolist(), strict=False):
            stop = tokenizer.eos_token_id in tokens
            if stop:
                tokens = tokens[: tokens.index(tokenizer.eos_token_id) + 1]
            result.append(
                dict(
                    row,
                    response_ids=tokens,
                    finish_reason="stop" if stop else "length",
                    cap=LONG_PASS.horizon,
                    generation_seed=seed,
                )
            )
        core.dump(a.out / f"generation-progress-{a.shard}.json", dict(generated=len(result), total=160))
        print("generated", a.shard, len(result), "/", 160, flush=True)
    core.dump(destination, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "generate"])
    for name in ["models", "data", "repo", "old", "out"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
