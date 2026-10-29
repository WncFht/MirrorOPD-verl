#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-LR effective-batch controls at initial and warm checkpoints.

The cyclic trajectory stream is identical across B16/B64/B128. Accumulation
changes the number of sequences per optimizer step, not microbatch precision.
Noise measurements use exact full-parameter Gram matrices, not sketches.
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.cross_batch import batch_generalization as base


def student(a):
    path = (
        a.models / core.NAMES["student"]
        if a.phase == "initial"
        else a.previous_cache / "checkpoints" / f"{a.teacher}-blocked"
    )
    return (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="flash_attention_2", local_files_only=True
        )
        .cuda()
        .eval()
    )


def backward(p, wq, rows, teacher_h):
    value = 0.0
    for r in rows:
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        hq = teacher_h[r["id"]]
        terms = []
        for start in range(0, len(hp), 128):
            lp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight).log_softmax(-1)
            with torch.no_grad():
                lq = torch.nn.functional.linear(hq[start : start + 128].float(), wq).log_softmax(-1)[:, : lp.shape[-1]]
            terms.append((lp.exp() * (lp - lq)).sum() / (len(rows) * 1024))
        loss = sum(terms)
        loss.backward()
        value += float(loss.detach())
        del hp, hq, lp, lq, loss, terms
    return value


def noise(a):
    bank, pool = read(a.out / "bank.json"), read(a.out / "pool.json")
    groups = [pool[i : i + 16] for i in range(0, 512, 16)]
    groups += [[r for r in bank if r["group"] in ["E", "F", "G", "H"]], [r for r in bank if r["id"] < 64]]
    p, q = student(a), core.load(a, a.teacher)
    wq = q.lm_head.weight.float()
    hs = {r["id"]: core.teacher_hidden(a, q, r) for r in pool + bank}
    q.cpu()
    torch.cuda.empty_cache()
    grads, losses = [], []
    started = time.time()
    for i, rows in enumerate(groups):
        p.zero_grad(set_to_none=True)
        losses.append(backward(p, wq, rows, hs))
        grads.append(
            torch.cat(
                [(v.grad if v.grad is not None else torch.zeros_like(v)).detach().reshape(-1) for v in p.parameters()]
            ).cpu()
        )
        p.zero_grad(set_to_none=True)
        print("noise", a.teacher, a.phase, i + 1, len(groups), "seconds", round(time.time() - started, 1), flush=True)
    gram = torch.zeros((len(groups), len(groups)), device="cuda", dtype=torch.float64)
    for start in range(0, len(grads[0]), 2**18):
        chunk = torch.stack([g[start : start + 2**18] for g in grads]).cuda()
        gram += (chunk @ chunk.T).double()
        del chunk
    result = dict(
        teacher=a.teacher,
        phase=a.phase,
        parameter_count=len(grads[0]),
        microbatch_size=16,
        training_microbatches=32,
        groups=[f"Pool16-{i}" for i in range(32)] + ["Held64", "Seen64"],
        prompt_ids=[[r["id"] for r in rows] for rows in groups],
        losses=losses,
        gram=gram.cpu().tolist(),
        seconds=time.time() - started,
        definition="Raw full-parameter gradients, fixed-H normalized; finite 512-prompt pool.",
    )
    dump(a.out / f"noise-{a.teacher}-{a.phase}.json", result)


def train(a):
    bank, pool = read(a.out / "bank.json"), read(a.out / "pool.json")
    assert a.examples % a.batch_size == 0
    a.steps = a.examples // a.batch_size
    a.name = f"{a.teacher}-{a.phase}-b{a.batch_size}"
    dest = a.out / a.name
    dest.mkdir(exist_ok=True)
    assert not (dest / "complete.json").exists(), "Use a separate name/output to repeat an arm."
    p, q = student(a), core.load(a, a.teacher)
    optimizer = torch.optim.AdamW(p.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    wq = q.lm_head.weight.float()
    hs = {r["id"]: core.teacher_hidden(a, q, r) for r in pool}
    q.cpu()
    torch.cuda.empty_cache()
    protocol = read(a.out / "design.json") | dict(
        teacher=a.teacher,
        phase=a.phase,
        batch_size=a.batch_size,
        steps=a.steps,
        parameter_count=sum(v.numel() for v in p.parameters()),
        warm_optimizer="reset for every arm",
        examples=a.examples,
        evaluation_updates=sorted(
            s for s in set([1, 4, 8, 16, a.steps] + list(range(32, a.steps + 1, 16))) if s <= a.steps
        ),
    )
    dump(dest / "protocol.json", protocol)
    history = []
    started = time.time()
    for step in range(1, a.steps + 1):
        offset = ((step - 1) * a.batch_size) % 512
        rows = pool[offset : offset + a.batch_size]
        optimizer.zero_grad(set_to_none=True)
        value = backward(p, wq, rows, hs)
        norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
        assert torch.isfinite(norm) and value == value and abs(value) < 1e6
        measure = step in [1, 16, a.steps]
        old = [v.detach().clone() for v in p.parameters()] if measure else None
        optimizer.step()
        row = dict(
            step=step,
            examples=step * a.batch_size,
            batch_size=a.batch_size,
            loss=value,
            grad_norm=float(norm),
            prompt_ids=[r["id"] for r in rows],
            tokens=sum(len(r["response_ids"]) for r in rows),
            normalizer=a.batch_size * 1024,
            seconds=time.time() - started,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        if measure:
            with torch.no_grad():
                squared = sum((v - before).square().sum() for v, before in zip(p.parameters(), old, strict=False))
            row["parameter_step_norm"] = float(squared.sqrt())
            del squared, old
        history.append(row)
        dump(dest / "training.json", history)
        print("train", a.name, step, "/", a.steps, "loss", round(value, 7), "grad", round(float(norm), 5), flush=True)
        if step in protocol["evaluation_updates"]:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            q.cuda()
            base.evaluate(a, p, q, bank, step, dest / f"fixed-{step}.json")
            if a.batch_size == 512 and step * a.batch_size == 2048 and step != a.steps:
                base.fresh(a, p, q, bank, step, dest / f"fresh-{step}.json")
            q.cpu()
            torch.cuda.empty_cache()
    del hs, wq, optimizer
    p.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    q.cuda()
    base.fresh(a, p, q, bank, a.steps, dest / f"fresh-{a.steps}.json")
    checkpoint = a.checkpoints / a.name
    checkpoint.mkdir(exist_ok=True, parents=True)
    p.save_pretrained(checkpoint, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True).save_pretrained(checkpoint)
    dump(dest / "complete.json", dict(finished=True, steps=a.steps, examples=a.examples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noise", "train"])
    for name in ["models", "out", "cache", "previous-cache", "checkpoints"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--teacher", choices=["justrl", "polaris"], required=True)
    ap.add_argument("--phase", choices=["initial", "warm128"], required=True)
    ap.add_argument("--batch-size", type=int, choices=[16, 64, 128, 512], default=16)
    ap.add_argument("--examples", type=int, default=2048)
    a = ap.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    a.cap = 1024
    a.k = 16
    a.schedule = "matched-stream"
    a.initial = a.phase != "initial"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.80)
    globals()[a.mode](a)


if __name__ == "__main__":
    main()
