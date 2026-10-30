#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-state batch-transfer and schedule controls with matched exposure.

A-D are four independently grouped student-generated math batches. E-H are
new math prompts, and Code/IF are separate domains. Frozen states remove
rollout drift from the cross-batch influence and forgetting measurements.
"""

import argparse
import math
import os
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.assets.paths import checkpoint_path, registered_model
from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.metrics import distances


def prepare(a):
    old = read(a.previous / "reference-student.json")
    bank = []
    mathrows = [r for r in old if r["domain"] == "math"]
    assert len(mathrows) == 64
    for i, r in enumerate(mathrows):
        bank.append(dict(r, id=i, origin_id=r["id"], group="ABCD"[i // 16], split="train", source="fixed-s0"))
    tok = AutoTokenizer.from_pretrained(registered_model(core.NAMES["student"], root=a.models), local_files_only=True)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    excluded = {norm(r["prompt"]) for r in read(a.previous / "prompts.json")}
    raw = pq.read_table(a.data / "math/DAPO/train.parquet").to_pylist()
    random.Random(2026102704).shuffle(raw)
    new = []
    for r in raw:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt, add_special_tokens=False)
        key = norm(prompt)
        if key in excluded or not 24 <= len(ids) <= 384 or any("\u4e00" <= c <= "\u9fff" for c in prompt):
            continue
        if "http" in prompt:
            continue
        i = len(new)
        new.append(
            dict(
                id=64 + i,
                domain="math",
                group="EFGH"[i // 16],
                split="heldout",
                source="fixed-s0",
                prompt=prompt,
                prompt_ids=ids,
                source_index=r["extra_info"].get("index"),
                ground_truth=r["reward_model"]["ground_truth"],
            )
        )
        excluded.add(key)
        if len(new) == 64:
            break
    for domain, group, start in [("code", "Code", 128), ("if", "IF", 144)]:
        rs = [r for r in old if r["domain"] == domain]
        assert len(rs) == 16
        for i, r in enumerate(rs):
            bank.append(dict(r, id=start + i, origin_id=r["id"], group=group, split="heldout", source="fixed-s0"))
    dump(a.out / "existing-bank.json", bank)
    dump(a.out / "new-prompts.json", new)
    dump(
        a.out / "protocol.json",
        dict(
            training_groups=list("ABCD"),
            heldout_math_groups=list("EFGH"),
            other_groups=["Code", "IF"],
            prompts_per_group=16,
            cap=1024,
            steps=128,
            lr=1e-6,
            clip=1,
            source="frozen initial-student trajectories for every group",
            schedules=["single", "blocked", "roundrobin", "mixed"],
            matched="blocked/roundrobin/mixed use exactly the same trajectory multiset per 64 updates",
            selection_seed=2026102704,
            reference_input="complete fixed input for every loss/evaluation; all response positions",
            loss_normalizer=(
                "16 * H for every update, zero contribution after refe"
                "rence EOS; matched per-token weights across schedules"
            ),
        ),
    )


@torch.no_grad()
def generate(a):
    prompts = read(a.out / "new-prompts.json")
    tok = AutoTokenizer.from_pretrained(registered_model(core.NAMES["student"], root=a.models), local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = core.load(a, "student")
    generated = []
    for start in [0, 32]:
        rs = prompts[start : start + 32]
        torch.manual_seed(98100 + start)
        enc = tok([r["prompt"] for r in rs], padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
        y = model.generate(
            **enc,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        for r, ids in zip(rs, y[:, enc.input_ids.shape[1] :].tolist(), strict=False):
            stop = tok.eos_token_id in ids
            if stop:
                ids = ids[: ids.index(tok.eos_token_id) + 1]
            generated.append(dict(r, response_ids=ids, finish_reason="stop" if stop else "length", cap=1024))
        print("new heldout generated", len(generated), flush=True)
    bank = read(a.out / "existing-bank.json") + generated
    bank.sort(key=lambda r: r["id"])
    assert len(bank) == 160
    dump(a.out / "bank.json", bank)


def prepare_rich(a):
    tok = AutoTokenizer.from_pretrained(registered_model(core.NAMES["student"], root=a.models), local_files_only=True)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    excluded = {norm(r["prompt"]) for r in read(a.out / "bank.json")}
    rows = pq.read_table(a.data / "math/DAPO/train.parquet").to_pylist()
    random.Random(2026102705).shuffle(rows)
    out = []
    for r in rows:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False)
        if (
            norm(text) in excluded
            or not 24 <= len(ids) <= 384
            or any("\u4e00" <= c <= "\u9fff" for c in text)
            or "http" in text
        ):
            continue
        i = len(out)
        out.append(
            dict(
                id=160 + i,
                domain="math",
                split="train",
                group=f"R{i // 16:02d}",
                source="fixed-s0",
                prompt=text,
                prompt_ids=ids,
                source_index=r["extra_info"].get("index"),
                ground_truth=r["reward_model"]["ground_truth"],
            )
        )
        excluded.add(norm(text))
        if len(out) == 448:
            break
    assert len(out) == 448
    dump(a.out / "rich-prompts.json", out)


@torch.no_grad()
def generate_rich(a):
    rows = read(a.out / "rich-prompts.json")[112 * a.shard : 112 * (a.shard + 1)]
    tok = AutoTokenizer.from_pretrained(registered_model(core.NAMES["student"], root=a.models), local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = core.load(a, "student")
    out = []
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        torch.manual_seed(88200 + 112 * a.shard + start)
        enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        y = model.generate(
            **enc,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=True,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        for r, ids in zip(batch, y[:, enc.input_ids.shape[1] :].tolist(), strict=False):
            stop = tok.eos_token_id in ids
            if stop:
                ids = ids[: ids.index(tok.eos_token_id) + 1]
            out.append(dict(r, response_ids=ids, finish_reason="stop" if stop else "length", cap=1024))
        dump(a.out / f"rich-shard-{a.shard}.json", out)
        print("rich generated", a.shard, len(out), flush=True)


def choose(bank, schedule, step):
    groups = {g: [r for r in bank if r["group"] == g] for g in "ABCD"}
    if schedule == "single":
        return groups["A"]
    if schedule == "blocked":
        return groups["ABCD"[(step // 16) % 4]]
    if schedule == "roundrobin":
        return groups["ABCD"[step % 4]]
    if schedule == "mixed":
        shard = 4 * (step % 4)
        return [r for g in "ABCD" for r in groups[g][shard : shard + 4]]
    raise ValueError(schedule)


@torch.no_grad()
def evaluate(a, p, q, bank, step, path):
    result = []
    wq = q.lm_head.weight.float()
    for r in bank:
        hq = core.teacher_hidden(a, q, r)
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        values = {}
        for start in range(0, len(hp), 128):
            sp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
            tq = torch.nn.functional.linear(hq[start : start + 128].float(), wq)
            for key, val in distances(sp, tq).items():
                values.setdefault(key, []).extend(val.tolist())
        bands = {}
        for name, lo, hi in [("all", 0, 1024), ("early", 0, 256), ("middle", 256, 512), ("late", 512, 1024)]:
            bands[name] = dict(tokens=max(0, min(len(hp), hi) - lo), **{k: sum(v[lo:hi]) for k, v in values.items()})
        result.append(
            dict(
                id=r["id"],
                group=r["group"],
                domain=r["domain"],
                split=r["split"],
                step=step,
                length=len(hp),
                stats=bands,
            )
        )
        del hp, hq, sp, tq, values
    dump(path, result)
    del wq
    torch.cuda.empty_cache()
    print("evaluated", a.teacher, a.schedule, step, flush=True)


@torch.no_grad()
def fresh(a, p, q, bank, step, path):
    prompts = [r for r in bank if r["group"] in ["E", "F"]]
    out = []
    for seed in [65101, 65102]:
        policy = "initial-student" if step == 0 and not a.initial else (a.name or f"{a.teacher}-{a.schedule}")
        out.extend(
            [
                dict(r, step=step, reference_source=r.get("source"), source=f"fresh-{policy}-step{step}-seed{seed}")
                for r in core.decode(a, p, q, prompts, "student", seed)
            ]
        )
        dump(path, out)


def run(a):
    bank = read(a.out / "bank.json")
    assert len(bank) == 160
    dest = a.out / (a.name or f"{a.teacher}-{a.schedule}")
    dest.mkdir(exist_ok=True)
    p = (
        AutoModelForCausalLM.from_pretrained(
            a.initial, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
        )
        .cuda()
        .eval()
        if a.initial
        else core.load(a, "student", student=True)
    )
    q = core.load(a, a.teacher)
    optimizer = torch.optim.AdamW(p.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    training_pool = [r for r in bank if r["split"] == "train"]
    if a.rich:
        training_pool += sum([read(a.out / f"rich-shard-{i}.json") for i in range(4)], [])
    assert len(training_pool) == (512 if a.rich else 64)
    dump(
        dest / "protocol.json",
        read(a.out / "protocol.json")
        | dict(
            teacher=a.teacher,
            schedule=a.schedule,
            steps=a.steps,
            params=sum(v.numel() for v in p.parameters()),
            fresh_optimizer=True,
            all_training_frozen=True,
            direction=a.direction,
            initial=str(a.initial) if a.initial else "original student",
            lr=a.lr,
            training_pool_size=len(training_pool),
            rich=a.rich,
            teacher_body_offload=a.offload_teacher,
        ),
    )
    baseline = a.out / (f"baseline-warm-{a.teacher}" if a.initial else f"baseline-{a.teacher}")
    lock = baseline.with_name("." + baseline.name + ".lock")
    owner = False
    if not (baseline / "done").exists():
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owner = True
        except FileExistsError:
            pass
    if owner:
        baseline.mkdir(exist_ok=True)
        evaluate(a, p, q, bank, 0, baseline / "fixed.json")
        fresh(a, p, q, bank, 0, baseline / "fresh.json")
        (baseline / "done").write_text("complete\n")
    teacher_h = {}
    with torch.no_grad():
        for r in training_pool:
            teacher_h[r["id"]] = core.teacher_hidden(a, q, r)
    wq = q.lm_head.weight.float()
    if a.offload_teacher:
        q.cpu()
        torch.cuda.empty_cache()
    history = []
    start_time = time.time()
    for step in range(a.steps):
        batch = training_pool[16 * (step % 32) : 16 * (step % 32 + 1)] if a.rich else choose(bank, a.schedule, step)
        total = sum(len(r["response_ids"]) for r in batch)
        normalizer = len(batch) * 1024
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for r in batch:
            hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
            hq = teacher_h[r["id"]]
            losses = []
            for start in range(0, len(hp), 128):
                sp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
                with torch.no_grad():
                    tq = torch.nn.functional.linear(hq[start : start + 128].float(), wq)
                lp = sp.log_softmax(-1)
                if a.direction == "reverse":
                    lq = tq.log_softmax(-1)[:, : sp.shape[-1]]
                    objective = (lp.exp() * (lp - lq)).sum()
                elif a.direction == "forward":
                    lq = tq[:, : sp.shape[-1]].log_softmax(-1)
                    objective = (lq.exp() * (lq - lp)).sum()
                else:
                    lq = tq.log_softmax(-1)
                    qprob = lq.exp()
                    mix = torch.logaddexp(lp, lq[:, : sp.shape[-1]]) - math.log(2)
                    objective = 0.5 * (
                        (lp.exp() * (lp - mix)).sum()
                        + (qprob[:, : sp.shape[-1]] * (lq[:, : sp.shape[-1]] - mix)).sum()
                        + qprob[:, sp.shape[-1] :].sum() * math.log(2)
                    )
                losses.append(objective / normalizer)
            loss = sum(losses)
            loss.backward()
            value += float(loss.detach())
            del hp, sp, tq, lq, lp, objective, loss, losses
        norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        assert torch.isfinite(norm) and math.isfinite(value)
        measure = step + 1 in [1, 16, 32, 64, 96, a.steps]
        before = [param.detach().clone() for param in p.parameters()] if measure else None
        optimizer.step()
        row = dict(
            step=step + 1,
            loss=value,
            grad_norm=float(norm),
            prompt_ids=[r["id"] for r in batch],
            tokens=total,
            normalizer=normalizer,
            seconds=time.time() - start_time,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        if measure:
            with torch.no_grad():
                delta = torch.zeros((), device="cuda")
                for old, param in zip(before, p.parameters(), strict=False):
                    delta += (param - old).square().sum()
            row["parameter_step_norm"] = float(delta.sqrt())
            del before, delta
        history.append(row)
        dump(dest / "training.json", history)
        print(
            "update",
            a.teacher,
            a.schedule,
            step + 1,
            "loss",
            round(value, 6),
            "grad",
            round(float(norm), 4),
            flush=True,
        )
        if (step + 1) % 16 == 0:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if a.offload_teacher:
                q.cuda()
            evaluate(a, p, q, bank, step + 1, dest / f"fixed-{step + 1}.json")
            if a.offload_teacher:
                q.cpu()
                torch.cuda.empty_cache()
    del teacher_h, wq
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if a.offload_teacher:
        q.cuda()
    if a.rich:
        evaluate(a, p, q, training_pool, a.steps, dest / f"train-pool-{a.steps}.json")
    fresh(a, p, q, bank, a.steps, dest / f"fresh-{a.steps}.json")
    checkpoint = checkpoint_path(a.cache, "checkpoints") / dest.name
    checkpoint.mkdir(parents=True, exist_ok=True)
    p.save_pretrained(checkpoint, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(
        registered_model(core.NAMES["student"], root=a.models), local_files_only=True
    ).save_pretrained(checkpoint)
    while not (baseline / "done").exists():
        time.sleep(5)
    dump(dest / "complete.json", dict(steps=a.steps, finished=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "generate", "prepare_rich", "generate_rich", "run"])
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--teacher", choices=["justrl", "polaris"], default="polaris")
    p.add_argument("--schedule", choices=["single", "blocked", "roundrobin", "mixed"], default="blocked")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--initial", type=Path)
    p.add_argument("--name")
    p.add_argument("--direction", choices=["reverse", "forward", "js"], default="reverse")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--rich", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--offload-teacher", action="store_true")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    a.cache.mkdir(exist_ok=True, parents=True)
    a.cap = 1024
    a.k = 16
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if a.stage not in ["prepare", "prepare_rich"]:
        torch.cuda.set_per_process_memory_fraction(0.88)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
