#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Cold-start loss/source controls, preserving the previous native probe recipe."""

import argparse
import gc
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..common import native as core
from . import filter_common as fc


def measures(sp, tq, z0, actions):
    lp, lq, l0 = sp.log_softmax(-1), tq.log_softmax(-1), z0.log_softmax(-1)
    p, q, p0 = lp.exp(), lq.exp(), l0.exp()
    v = sp.shape[-1]
    lqc = tq[:, :v].log_softmax(-1)
    mix = torch.logaddexp(lp, lq[:, :v]) - math.log(2)
    mix0 = torch.logaddexp(lp, l0) - math.log(2)
    si, ti = sp.topk(16, -1).indices, tq.topk(16, -1).indices
    common = (si[:, :, None] == ti[:, None, :]).any(-1)
    initial_top = z0.topk(16, -1).indices
    teacher_only = ~(ti[:, :, None] == initial_top[:, None, :]).any(-1)
    valid_teacher = ti < v
    safe_teacher = ti.clamp(max=v - 1)
    return dict(
        rkl=(p * (lp - lq[:, :v])).sum(-1),
        fkl_shared=(lqc.exp() * (lqc - lp)).sum(-1),
        js=0.5 * ((p * (lp - mix)).sum(-1) + (q[:, :v] * (lq[:, :v] - mix)).sum(-1) + q[:, v:].sum(-1) * math.log(2)),
        tv=0.5 * ((p - q[:, :v]).abs().sum(-1) + q[:, v:].sum(-1)),
        hp=-(p * lp).sum(-1),
        hq=-(q * lq).sum(-1),
        retain_fkl=(p0 * (l0 - lp)).sum(-1),
        retain_rkl=(p * (lp - l0)).sum(-1),
        retain_js=0.5 * ((p * (lp - mix0)).sum(-1) + (p0 * (l0 - mix0)).sum(-1)),
        reference_nll=-lp.gather(1, actions[:, None]).squeeze(1),
        teacher_reference_nll=-lqc.gather(1, actions[:, None]).squeeze(1),
        outside_student_mass=q[:, v:].sum(-1),
        overlap16=common.float().mean(-1),
        common_p16=(p.gather(1, si) * common).sum(-1),
        common_q16=(q.gather(1, si) * common).sum(-1),
        q_outside_initial_top16=1 - q.gather(1, initial_top).sum(-1),
        p_outside_initial_top16=1 - p.gather(1, initial_top).sum(-1),
        p0_outside_initial_top16=1 - p0.gather(1, initial_top).sum(-1),
        teacher_only_initial_p=(p.gather(1, safe_teacher) * teacher_only * valid_teacher).sum(-1),
        teacher_only_initial_p0=(p0.gather(1, safe_teacher) * teacher_only * valid_teacher).sum(-1),
        teacher_only_initial_q=(q.gather(1, ti) * teacher_only).sum(-1),
    )


@torch.no_grad()
def evaluate(a, p, wq, w0, hq, h0, rows, step, path):
    result = []
    for r in rows:
        key = (r["source"], r["id"])
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        teacher_h, old_h = hq[key].cuda(), h0[key].cuda()
        values = {}
        for start in range(0, len(hp), 128):
            sp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
            tq = torch.nn.functional.linear(teacher_h[start : start + 128].float(), wq)
            z0 = torch.nn.functional.linear(old_h[start : start + 128].float(), w0)
            acts = torch.tensor(r["response_ids"][start : start + 128], device="cuda")
            for name, v in measures(sp, tq, z0, acts).items():
                values.setdefault(name, []).extend(v.tolist())
        bands = {}
        for name, lo, hi in [("all", 0, 1024), ("early", 0, 256), ("middle", 256, 512), ("late", 512, 1024)]:
            bands[name] = dict(tokens=max(0, min(len(hp), hi) - lo), **{k: sum(v[lo:hi]) for k, v in values.items()})
        result.append(
            dict(
                id=r["id"],
                source=r["source"],
                view=r.get("view", "active"),
                role=r["role"],
                domain=r["domain"],
                group=r["group"],
                step=step,
                length=len(hp),
                stats=bands,
            )
        )
        del hp, teacher_h, old_h, sp, tq, z0, acts, values
    core.dump(path, result)
    torch.cuda.empty_cache()
    print("EVAL", a.name, step, len(rows), flush=True)


def backward(a, p, wq, hq, rows):
    total = 0.0
    for r in rows:
        hp = core.hidden(p, r["prompt_ids"] + r["response_ids"][:-1])[len(r["prompt_ids"]) - 1 :]
        ht = hq[r["source"], r["id"]].cuda() if a.direction != "sft" else None
        terms = []
        for start in range(0, len(hp), 128):
            sp = torch.nn.functional.linear(hp[start : start + 128].float(), p.lm_head.weight)
            lp = sp.log_softmax(-1)
            if a.direction == "sft":
                acts = torch.tensor(r["response_ids"][start : start + 128], device="cuda")
                objective = -lp.gather(1, acts[:, None]).sum()
            else:
                with torch.no_grad():
                    tq = torch.nn.functional.linear(ht[start : start + 128].float(), wq)
                    lq = (
                        tq.log_softmax(-1)[:, : sp.shape[-1]]
                        if a.direction == "reverse"
                        else tq[:, : sp.shape[-1]].log_softmax(-1)
                    )
                objective = (lp.exp() * (lp - lq)).sum() if a.direction == "reverse" else (lq.exp() * (lq - lp)).sum()
            terms.append(objective / (16 * 1024))
        loss = sum(terms)
        loss.backward()
        total += float(loss.detach())
        del hp, ht, sp, lp, loss, terms, objective
    return total


def save_model(a, p, step):
    target = a.cache / "checkpoints" / a.name / f"step-{step:03d}"
    target.mkdir(parents=True, exist_ok=True)
    p.save_pretrained(target, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(a.models / core.NAMES["student"], local_files_only=True).save_pretrained(target)
    core.dump(target / "complete.json", dict(step=step, weights_only=True))


def fresh(a, p, q, step, dest):
    q.cuda()
    prompts = [r for r in core.read(a.out / "test-bank.json") if r["group"] in ["M05", "M06"]]
    result = []
    for seed in [33710, 33711]:
        result.extend(
            dict(r, step=step, source=f"fresh-{a.name}-{step}-{seed}")
            for r in core.decode(a, p, q, prompts, "student", seed)
        )
    core.dump(dest / f"fresh-{step:03d}.json", result)
    q.cpu()
    torch.cuda.empty_cache()


def task(a, p, dest):
    prompts = core.read(a.out / "task-prompts.json")
    result = []
    for start in range(0, 64, 16):
        result.extend(fc.generate(a, p, prompts[start : start + 16], 88700 + start, 8192))
        core.dump(dest / "task.json", result)
    fc.grade_math(result, a.repo)
    core.dump(dest / "task.json", result)


def run(a):
    a.name = a.name or f"{a.teacher}-{a.source}-{a.direction}"
    dest = a.out / a.name
    dest.mkdir(exist_ok=True)
    assert not (dest / "training.json").exists() and not (dest / "complete.json").exists(), (
        "Use a new arm/output for a retry."
    )
    s0 = [dict(r, view="student") for r in core.read(a.out / "reference-student.json")]
    qt = [dict(r, view="teacher") for r in core.read(a.out / f"reference-{a.teacher}.json")]
    test = [dict(r, view="student") for r in core.read(a.out / "test-bank.json")]
    bank = s0 + qt + test
    assert len(bank) == 576
    p = core.load(a, "student", student=True)
    q = core.load(a, a.teacher)
    wq = q.lm_head.weight.detach().float()
    w0 = p.lm_head.weight.detach().clone()
    fc.cache_hidden(a.cache, a.teacher, q, bank)
    q.cpu()
    torch.cuda.empty_cache()
    fc.cache_hidden(a.cache, "initial-student", p, bank)
    hq = fc.load_hidden(a.cache, a.teacher, bank, model=q)
    h0 = fc.load_hidden(a.cache, "initial-student", bank, model=p)
    ref = None
    if a.source == "online":
        ref = (
            AutoModelForCausalLM.from_pretrained(
                a.models / core.NAMES["student"],
                dtype=torch.float32,
                attn_implementation="flash_attention_2",
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
    optimizer = torch.optim.AdamW(p.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, foreach=True)
    core.dump(
        dest / "protocol.json",
        core.read(a.out / "protocol.json")
        | dict(
            teacher=a.teacher,
            source=a.source,
            direction=a.direction,
            name=a.name,
            parameter_count=sum(v.numel() for v in p.parameters()),
        ),
    )
    evaluate(a, p, wq, w0, hq, h0, bank, 0, dest / "fixed-000.json")
    initial = core.read(dest / "fixed-000.json")
    assert max(abs(r["stats"]["all"]["retain_fkl"]) for r in initial) < 1e-4
    history = []
    started = time.time()
    for batch in range(8):
        ids = set(range(batch * 16, (batch + 1) * 16))
        optimizer.zero_grad(set_to_none=True)
        if a.source == "online":
            prompts = [r for r in s0 if r["id"] in ids]
            current = fc.generate(a, p, prompts, 93000 + batch, 1024)
            current = [
                dict(r, source=f"filter-online-{a.name}-pid{os.getpid()}-b{batch}", view="online") for r in current
            ]
            if batch == 0:
                assert [r["response_ids"] for r in current] == [r["response_ids"] for r in prompts]
            core.dump(dest / f"rollout-{batch + 1:02d}.json", current)
            q.cuda()
            fc.cache_hidden(a.cache, a.teacher, q, current)
            q.cpu()
            torch.cuda.empty_cache()
            ref.cuda()
            fc.cache_hidden(a.cache, "initial-student", ref, current)
            ref.cpu()
            torch.cuda.empty_cache()
            hq.update(fc.load_hidden(a.cache, a.teacher, current, model=q))
            h0.update(fc.load_hidden(a.cache, "initial-student", current, model=ref))
        else:
            current = [r for r in (s0 if a.source == "student" else qt) if r["id"] in ids]
        assert len(current) == 16
        evaluate(a, p, wq, w0, hq, h0, current, batch * 16, dest / f"active-{batch + 1:02d}-before.json")
        for repeat in range(16):
            step = batch * 16 + repeat + 1
            optimizer.zero_grad(set_to_none=True)
            value = backward(a, p, wq, hq, current)
            norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
            assert math.isfinite(value) and torch.isfinite(norm)
            measure = step in [1, 16, 32, 64, 96, 128]
            old = [v.detach().clone() for v in p.parameters()] if measure else None
            optimizer.step()
            row = dict(
                step=step,
                batch=batch + 1,
                repeat=repeat + 1,
                loss=value,
                grad_norm=float(norm),
                normalizer=16384,
                prompt_ids=[r["id"] for r in current],
                actual_tokens=sum(len(r["response_ids"]) for r in current),
                seconds=time.time() - started,
            )
            if measure:
                with torch.no_grad():
                    delta = sum((v - before).square().sum() for v, before in zip(p.parameters(), old, strict=False))
                row["parameter_step_norm"] = float(delta.sqrt())
                del old, delta
            history.append(row)
            core.dump(dest / "training.json", history)
            print("UPDATE", a.name, step, "loss", round(value, 6), "grad", round(float(norm), 4), flush=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        evaluate(a, p, wq, w0, hq, h0, current, (batch + 1) * 16, dest / f"active-{batch + 1:02d}-after.json")
        if (batch + 1) * 16 in [32, 64, 96, 128]:
            evaluate(a, p, wq, w0, hq, h0, bank, (batch + 1) * 16, dest / f"fixed-{(batch + 1) * 16:03d}.json")
            if a.source == "student" or batch == 7:
                save_model(a, p, (batch + 1) * 16)
        if batch == 3:
            fresh(a, p, q, 64, dest)
    del optimizer, hq, h0, ref
    gc.collect()
    torch.cuda.empty_cache()
    fresh(a, p, q, 128, dest)
    del q, wq
    gc.collect()
    torch.cuda.empty_cache()
    task(a, p, dest)
    core.dump(dest / "complete.json", dict(finished=True, updates=128, batches=8, task_answers=64))


def main():
    ap = argparse.ArgumentParser()
    for key in ["models", "out", "cache", "repo"]:
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--teacher", choices=["justrl", "polaris"], required=True)
    ap.add_argument("--source", choices=["student", "teacher", "online"], required=True)
    ap.add_argument("--direction", choices=["reverse", "forward", "sft"], required=True)
    ap.add_argument("--name")
    a = ap.parse_args()
    assert a.direction != "sft" or a.source == "teacher"
    a.out.mkdir(parents=True, exist_ok=True)
    a.cap = 1024
    a.k = 16
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(0.8)
    run(a)


if __name__ == "__main__":
    main()
