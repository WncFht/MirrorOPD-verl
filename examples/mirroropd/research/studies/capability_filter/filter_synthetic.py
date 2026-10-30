#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-parameter counterexample probe: shared rule versus fixed random codebook.

One action terminates the task. All states are prompt-only, so on-policy state
sampling is identical for both losses. q=.9*one_hot(label)+.1*p_initial keeps
full support. Random held-out labels are independent of training labels.
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from examples.mirroropd.assets.paths import checkpoint_path, registered_model
from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump


def inputs(a, tokenizer):
    prefixes = ["dax", "wug", "zup", "vem"]
    mapping = [2, 4, 1, 3]
    rng = random.Random(2026102725)
    rows = []
    for split, values in [("train", range(4)), ("test", range(4, 20))]:
        for group, prefix in enumerate(prefixes):
            labels = [1, 2, 3, 4] * (len(values) // 4)
            rng.shuffle(labels)
            for suffix, random_label in zip(values, labels, strict=False):
                user = (
                    f"Use the learned codebook to label item {prefix}_{suffix:02d}. "
                    "The tag is one digit: 1, 2, 3, or 4. Return only its tag."
                )
                prompt = (
                    tokenizer.apply_chat_template(
                        [dict(role="user", content=user)], tokenize=False, add_generation_prompt=True
                    )
                    + "</think>\n\nTag: "
                )
                rows.append(
                    dict(
                        id=len(rows),
                        split=split,
                        prefix=prefix,
                        suffix=suffix,
                        prompt=prompt,
                        shared_label=mapping[group],
                        random_label=random_label,
                    )
                )
    assert len(rows) == 80 and sum(r["split"] == "train" for r in rows) == 16
    return rows


def logits(model, encoded):
    positions = encoded.attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(encoded.attention_mask == 0, 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, position_ids=positions, use_cache=False
        ).last_hidden_state[:, -1]
    return torch.nn.functional.linear(h.float(), model.lm_head.weight)


@torch.no_grad()
def evaluate(model, encoded, lq, targets, label_ids):
    lp = logits(model, encoded).log_softmax(-1)
    p = lp.exp()
    q = lq.exp()
    mix = torch.logaddexp(lp, lq) - math.log(2)
    correct = p.gather(1, targets[:, None]).squeeze(1)
    pred = lp.argmax(-1)
    conditional = label_ids[lp[:, label_ids].argmax(-1)]
    return dict(
        rkl=(p * (lp - lq)).sum(-1).tolist(),
        fkl=(q * (lq - lp)).sum(-1).tolist(),
        js=(0.5 * ((p * (lp - mix)).sum(-1) + (q * (lq - mix)).sum(-1))).tolist(),
        correct_probability=correct.tolist(),
        top1_correct=(pred == targets).tolist(),
        candidate_conditional_correct=(conditional == targets).tolist(),
        candidate_mass=p[:, label_ids].sum(-1).tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    for k in ["models", "out", "cache"]:
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--kind", choices=["shared", "random"], required=True)
    ap.add_argument("--direction", choices=["reverse", "forward"], required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        registered_model(core.NAMES["student"], root=a.models), local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = inputs(a, tokenizer)
    dest = a.out / f"synthetic-{a.kind}-{a.direction}"
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "training.json").exists()
    dump(dest / "prompts.json", rows)
    encoded = tokenizer([r["prompt"] for r in rows], padding=True, return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )
    train = tokenizer([r["prompt"] for r in rows[:16]], padding=True, return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in [1, 2, 3, 4]]
    assert all(len(v) == 1 for v in ids)
    label_ids = torch.tensor([v[0] for v in ids], device="cuda")
    targets = label_ids[torch.tensor([r[a.kind + "_label"] - 1 for r in rows], device="cuda")]
    p = core.load(a, "student", student=True)
    with torch.no_grad():
        lp0 = logits(p, encoded).log_softmax(-1)
        lq = lp0 + math.log(0.1)
        lq[torch.arange(80, device="cuda"), targets] = torch.logaddexp(
            lq[torch.arange(80, device="cuda"), targets], torch.tensor(math.log(0.9), device="cuda")
        )
        assert float(lq.logsumexp(-1).abs().max()) < 2e-6
        baseline = evaluate(p, encoded, lq, targets, label_ids)
    dump(dest / "eval-000.json", baseline)
    protocol = dict(
        kind=a.kind,
        direction=a.direction,
        train=16,
        test=64,
        full_parameter_count=sum(v.numel() for v in p.parameters()),
        horizon=1,
        lr=1e-6,
        betas=[0.9, 0.999],
        weight_decay=0.01,
        clip=1.0,
        max_steps=512,
        teacher="q=.9*onehot(label)+.1*p0, full support, frozen initial probabilities",
        state="prompt-only, single terminal action; no policy-dependent continuation states",
        selection="fixed codebook; random labels balanced within each prefix and split",
        early_stop=(
            "at predefined checkpoints when all 16 training top-1 outputs are "
            "correct AND mean target probability >=.8; no test-based stopping"
        ),
        checkpoints=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        scope="Counterexample to a hard RKL learning gate; not a mathematical-reasoning skill benchmark.",
    )
    dump(dest / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(p.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    history = []
    started = time.time()
    criterion = False
    for step in range(1, 513):
        optimizer.zero_grad(set_to_none=True)
        lp = logits(p, train).log_softmax(-1)
        loss = (
            (lp.exp() * (lp - lq[:16])).sum() / 16
            if a.direction == "reverse"
            else (lq[:16].exp() * (lq[:16] - lp)).sum() / 16
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        history.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), seconds=time.time() - started))
        dump(dest / "training.json", history)
        if step in protocol["checkpoints"]:
            optimizer.zero_grad(set_to_none=True)
            values = evaluate(p, encoded, lq, targets, label_ids)
            dump(dest / f"eval-{step:03d}.json", values)
            train_acc = sum(values["top1_correct"][:16]) / 16
            test_acc = sum(values["top1_correct"][16:]) / 64
            meanprob = sum(values["correct_probability"][:16]) / 16
            print(
                "SYNTHETIC",
                a.kind,
                a.direction,
                step,
                "train",
                train_acc,
                "test",
                test_acc,
                "train_probability",
                meanprob,
                flush=True,
            )
            criterion = train_acc == 1 and meanprob >= 0.8
            if criterion:
                break
    target = checkpoint_path(a.cache, "checkpoints") / dest.name
    target.mkdir(parents=True, exist_ok=True)
    p.save_pretrained(target, max_shard_size="4GB")
    tokenizer.save_pretrained(target)
    dump(dest / "complete.json", dict(finished=True, steps=step, training_criterion_reached=criterion))


if __name__ == "__main__":
    main()
