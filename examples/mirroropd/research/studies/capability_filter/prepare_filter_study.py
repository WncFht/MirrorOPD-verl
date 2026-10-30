#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Preregister matched objective/source comparisons and independent test prompts."""

import argparse
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from examples.mirroropd.assets.paths import registered_model
from examples.mirroropd.research.backends import hf_native as core
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.capability_filter.filter_common import generate, grade_math


def prepare(a):
    assert not (a.out / "protocol.json").exists()
    old = read(a.previous / "bank.json")
    # Source-independent prompt identities. The student training paths are regenerated
    # using the same FP32-head sampler as the online branch.
    train = [dict(r, role="train") for r in old if r["id"] < 128]
    diagnostic = [dict(r, role="diagnostic") for r in old if 320 <= r["id"] < 384]
    test = [dict(r, role="test") for r in old if r["id"] >= 384]
    dump(a.out / "training-prompts.json", train)
    dump(a.out / "diagnostic-bank.json", diagnostic)
    dump(a.out / "test-bank.json", test)
    task_teacher = [dict(r) for r in test if 384 <= r["id"] < 448]
    dump(a.out / "teacher-reference-prompts.json", train + task_teacher)
    tok = AutoTokenizer.from_pretrained(registered_model(core.NAMES["student"], root=a.models), local_files_only=True)

    def norm(s):
        return re.sub(r"\s+", " ", s).strip()

    excluded = {norm(r["prompt"]) for r in old}
    earlier = a.repo / (
        "examples/mirroropd/docs/experiments/teacher-transfer/transfer-mechanism/"
        "data/intersection-study-20261027/task-prompts.json"
    )
    excluded.update(norm(r["prompt"]) for r in read(earlier))
    raw = pq.read_table(a.data / "math/MATH-500/test.parquet").to_pylist()
    random.Random(2026102723).shuffle(raw)
    task = []
    for r in raw:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        if norm(prompt) in excluded:
            continue
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) > 768:
            continue
        task.append(
            dict(
                id=10000 + len(task),
                domain="math500",
                role="task_test",
                prompt=prompt,
                prompt_ids=ids,
                source_index=str(r["extra_info"].get("index")),
                ground_truth=r["reward_model"]["ground_truth"],
            )
        )
        excluded.add(norm(prompt))
        if len(task) == 64:
            break
    assert len(task) == 64
    dump(a.out / "task-prompts.json", task)
    arms = []
    for teacher in ["justrl", "polaris"]:
        for source, direction in [
            ("student", "reverse"),
            ("student", "forward"),
            ("teacher", "reverse"),
            ("teacher", "forward"),
            ("teacher", "sft"),
            ("online", "reverse"),
            ("online", "forward"),
        ]:
            arms.append(
                dict(name=f"{teacher}-{source}-{direction}", teacher=teacher, source=source, direction=direction)
            )
    dump(
        a.out / "protocol.json",
        dict(
            phase="cold start only",
            models=core.NAMES,
            seed=42,
            train_prompts=128,
            batches=8,
            updates_per_batch=16,
            batch_size=16,
            updates=128,
            cap=1024,
            lr=1e-6,
            betas=[0.9, 0.999],
            weight_decay=0.01,
            clip=1.0,
            normalization="fixed 16*1024, zero beyond reference EOS",
            train_order=list("ABCDEFGH"),
            revisits=False,
            evaluation_steps=[0, 32, 64, 96, 128],
            diagnostic_math=64,
            independent_test_math=128,
            test_code=64,
            test_if=64,
            teacher_test_reference_math=64,
            source_sampling="T=1, full student support; teacher conditioned to student output rows",
            directions=(
                "RKL native teacher; FKL teacher conditioned to student output "
                "support; SFT hard teacher-sampled token NLL"
            ),
            precision="FP32 student master/head; BF16 backbone autocast; BF16 frozen teacher, FP32 teacher head",
            function_preservation=(
                "fixed-state KL(initial||current), KL(current||initial), "
                "JS(initial,current); not task forgetting by definition"
            ),
            transfer_proxy=(
                "Pre-training full-parameter JS gradient alignment of each training "
                "prompt with diagnostic math64; test math128 gradient only validates "
                "proxy, never selects training"
            ),
            transfer_bins=(
                "Top/bottom eight initial JS-alignment scores WITHIN each training "
                "batch; scores independent of later outcomes"
            ),
            task_eval=dict(
                prompts=64, source="MATH-500", cap=8192, temperature=1.0, samples_per_prompt=1, seed_base=88700
            ),
            comparison=(
                "matched update/data budgets; also inspect common JS acquisition "
                "versus initial-policy drift, not only native training loss"
            ),
            hypothesis_strong="Only transferable target information can be acquired by RKL/OPD.",
            hypothesis_weak=(
                "At comparable acquisition or drift, RKL/OPD shows stronger preference "
                "for initially transferable corrections than FKL/SFT."
            ),
            limitations=(
                "Distribution-level proxies are not abilities. Teacher-source H1024 "
                "SFT is truncated-prefix SFT, not full-solution SFT. Single training "
                "seed."
            ),
            arms=arms,
            design_revision=2,
            posthoc_calibration=dict(
                arms=[
                    "initial",
                    "polaris-student-reverse",
                    "polaris-student-forward",
                    "polaris-teacher-reverse",
                    "polaris-teacher-forward",
                ],
                temperatures=[0.6, 0.8, 1.0, 1.25, 1.5, 2.0],
                selection="minimum mean fixed-prefix JS on 64 diagnostic math prompts only",
                test="128 independent math prompts, full saved H1024 prefix positions",
                secondary_teacher_test="64 teacher-prefix held-out prompts; same diagnostic-selected temperature",
                model_parameter_updates=0,
                calibration_scalars=1,
                purpose="Quantify scalar logit calibration contribution; not a task-accuracy intervention",
            ),
            revision_note="Complete online source x loss factor; OPD sampling is not conflated with reverse KL.",
            synthetic_control=dict(
                arms=4,
                kinds=["shared", "random"],
                directions=["reverse", "forward"],
                max_updates_per_arm=512,
                early_stop="predefined train-only accuracy/confidence criterion",
                reason="Test hard filter on a one-action problem with fixed random held-out labels.",
            ),
            if_eval=dict(
                prompts=64,
                source="fixed IF holdout; single-turn diagnostic, not official multi-turn benchmark",
                cap=4096,
                temperature=1.0,
                samples_per_prompt=1,
                seed_base=99700,
                grading="local MultiIF strict and loose checkers; strip control tokens and reasoning",
                models="initial student, two teachers, fourteen endpoints",
                natural_stop="report strict success and strict success with natural stop separately",
            ),
        ),
    )
    print("Prepared 14 cold-start arms, independent diagnostic/test panels, and 64 task prompts.", flush=True)


def reference(a):
    name = a.source
    model = core.load(a, name, student=name == "student")
    rows = read(a.out / ("training-prompts.json" if name == "student" else "teacher-reference-prompts.json"))
    result = []
    for start in range(0, len(rows), 16):
        block = rows[start : start + 16]
        generated = generate(
            a, model, block, 93000 + start // 16, 1024, vocab_limit=None if name == "student" else 151936
        )
        result.extend(dict(r, source=f"filter-{name}-initial") for r in generated)
        print("REFERENCE", name, len(result), "/", len(rows), flush=True)
    dump(a.out / f"reference-{name}.json", result)


def task(a):
    model = core.load(a, a.source, student=a.source == "student")
    prompts = read(a.out / "task-prompts.json")
    result = []
    for start in range(0, 64, 16):
        result.extend(
            generate(
                a,
                model,
                prompts[start : start + 16],
                88700 + start,
                8192,
                vocab_limit=None if a.source == "student" else 151936,
            )
        )
        dump(a.out / f"task-{a.source}-baseline.json", result)
    grade_math(result, a.repo)
    dump(a.out / f"task-{a.source}-baseline.json", result)


def baseline_fresh(a):
    assert a.source in ["justrl", "polaris"]
    p = core.load(a, "student", student=True)
    q = core.load(a, a.source)
    prompts = [r for r in read(a.out / "test-bank.json") if r["group"] in ["M05", "M06"]]
    result = []
    for seed in [33710, 33711]:
        result.extend(
            dict(r, step=0, source=f"fresh-initial-{seed}") for r in core.decode(a, p, q, prompts, "student", seed)
        )
    dump(a.out / f"baseline-fresh-{a.source}.json", result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "reference", "task", "baseline_fresh"])
    for k in ["models", "out", "previous", "data", "repo"]:
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--source", choices=["student", "justrl", "polaris"], default="student")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cap = 1024
    a.k = 16
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
