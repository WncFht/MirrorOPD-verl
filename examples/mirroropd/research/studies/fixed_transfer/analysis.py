# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Streaming task audits and shared risk/GapRecovery statistics for any study plan."""

import hashlib
import json

import numpy as np

from examples.mirroropd.research.core.artifacts import complete
from examples.mirroropd.research.core.io import dump as write
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.core.io import write_csv as csv_write
from examples.mirroropd.research.core.transfer_metrics import gap_row, summarize_gap
from examples.mirroropd.research.studies.fixed_transfer.data import required_rows
from examples.mirroropd.research.studies.fixed_transfer.evaluation import evaluation_inputs, validate_records
from examples.mirroropd.research.studies.fixed_transfer.models import same_parameter_layout
from examples.mirroropd.research.studies.fixed_transfer.runner import check_inputs
from examples.mirroropd.research.studies.fixed_transfer.spec import fingerprint
from examples.mirroropd.research.studies.fixed_transfer.training import (
    arm_directory,
    checkpoint_directory,
    scientific_inputs,
)

PANELS = {
    "train_ad": list("ABCD"),
    "train_eh": list("EFGH"),
    "train_it": list("IJKLMNOPQRST"),
    "held_math": ["M01", "M02", "M03", "M04"],
    "held_code": ["Code01", "Code02"],
    "held_if": ["IF01", "IF02"],
}


def task_data(study, name, weight_identity=None):
    inputs = evaluation_inputs(study, name, weight_identity)
    directory = study.cache_directory / "evaluations" / fingerprint(inputs)
    if not complete(directory, inputs):
        return None
    source = directory / "records.jsonl"
    cached = directory / "prompt-statistics-v1.json"
    stat = source.stat()
    if cached.exists():
        saved = read(cached)
        if saved["size"] == stat.st_size and saved["mtime_ns"] == stat.st_mtime_ns:
            assert saved["input_fingerprint"] == fingerprint(inputs)
            return saved
    prompts = read(study.directory / "prepared/evaluation-prompts.json")
    expected = {(p["prompt_id"], i) for p in prompts for i in range(study.evaluation.samples)}
    seen = set()
    by_prompt = {}
    totals = {}
    digest = hashlib.sha256()
    chunk = []
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            key = (row["prompt_id"], row["sample_index"])
            if key in seen:
                raise ValueError(f"Duplicate response slot in {source}")
            seen.add(key)
            chunk.append(row)
            by_prompt.setdefault(row["data_source"], {}).setdefault(row["prompt_id"], []).append(float(row["acc"]))
            acc = totals.setdefault(row["data_source"], dict(responses=0, length=0, repeat8=0.0, truncated=0, empty=0))
            acc["responses"] += 1
            acc["length"] += row["response_length"]
            acc["repeat8"] += row["repeat8"]
            acc["truncated"] += int(row["finish_reason"] == "length")
            acc["empty"] += int(not row["output"].strip())
            if len(chunk) == 32:
                validate_records(chunk, prompts, study.evaluation)
                chunk = []
    if chunk:
        validate_records(chunk, prompts, study.evaluation)
    assert seen == expected and read(directory / "complete.json")["records"] == len(expected)
    vectors = {dataset: {p: float(np.mean(values[p])) for p in sorted(values)} for dataset, values in by_prompt.items()}
    for values in by_prompt.values():
        assert all(len(v) == study.evaluation.samples for v in values.values())
    if cached.exists() and read(cached)["record_sha256"] != digest.hexdigest():
        raise ValueError(f"Completed response artifact changed: {source}")
    result = dict(
        input_fingerprint=fingerprint(inputs),
        record_sha256=digest.hexdigest(),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        vectors=vectors,
        totals=totals,
        records=len(expected),
    )
    write(cached, result)
    return result


def analyze(study, partial=False):
    directory = study.directory / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    prompts = read(study.directory / "prepared/evaluation-prompts.json")
    datasets = list(dict.fromkeys(r["data_source"] for r in prompts))
    rng = np.random.default_rng(42)
    missing = []
    accuracy = []
    gaps = []
    risks = []
    curves = []
    performance = []
    checks = []
    copies = []
    numerical_profiles = {}
    baselines = {name: task_data(study, name) for name in study.models}

    def accuracy_rows(label, kind, name, data, arm=None, step=0):
        for dataset in datasets:
            values = data["vectors"][dataset]
            total = data["totals"][dataset]
            n = total["responses"]
            accuracy.append(
                dict(
                    label=label,
                    kind=kind,
                    model=name,
                    arm=arm,
                    step=step,
                    dataset=dataset,
                    accuracy_percent=100 * np.mean(list(values.values())),
                    prompts=len(values),
                    responses=n,
                    mean_length=total["length"] / n,
                    mean_repeat8=total["repeat8"] / n,
                    length_stop_percent=100 * total["truncated"] / n,
                    empty_percent=100 * total["empty"] / n,
                )
            )

    for name, data in baselines.items():
        if data is None:
            missing.append("baseline-" + name)
        else:
            accuracy_rows(name, "baseline", name, data)
    for arm in study.arms:
        root = arm_directory(study, arm)
        finished = (root / "complete.json").exists()
        if not finished:
            missing.append("train-" + arm.name)
        else:
            check_inputs(root / "complete.json", scientific_inputs(study, arm))
            training_receipt = read(root / "complete.json")
            profile_id = training_receipt["inputs"]["numerical_profile"]
            profile = read(study.directory / "numerical" / profile_id / "complete.json")
            assert profile["fingerprint"] == fingerprint(profile["inputs"])
            assert profile["inputs"]["student"] == study.models[study.student].identity
            assert fingerprint(profile["execution"]) == fingerprint(training_receipt["inputs"]["execution"])
            best = profile["execution"]["microbatch"]
            assert {p["point"] for p in profile["probes"]} == {"S0", "teacher-copy"}
            for probe in profile["probes"]:
                selected = next(p for p in probe["candidates"] if p["microbatch"] == best)
                assert selected["loss_error"] < 1e-5 and selected["relative_l2"] < 0.01
                assert selected["cosine"] > 0.9999 and selected["estimated_peak_with_adam_gib"] < 74
            assert len(profile["disposable_adam_steps"]) == 2
            for index, smoke in enumerate(profile["disposable_adam_steps"], 1):
                assert smoke["optimizer_steps"] == [index] and smoke["sampled_parameter_change_l2"] > 0
                assert smoke["actual_peak_with_adam_gib"] + profile["teacher_head_reserve_gib"] < 74
            numerical_profiles[profile_id] = dict(execution=profile["execution"], student=profile["inputs"]["student"])
            history = read(root / "training.json")
            assert [r["step"] for r in history] == list(range(1, arm.updates + 1))
            assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in history)
            rows = required_rows(study, arm)
            ids = [r["id"] for r in rows]
            initial = read(root / "fixed-000.json")
            assert [r["id"] for r in initial] == ids
            snapshots = {}
            for step in study.training.score_steps(arm.updates, arm.evaluate_at):
                scored = read(root / f"fixed-{step:03d}.json")
                snapshots[step] = scored
                assert [r["id"] for r in scored] == ids
                assert all(
                    r["source"] == base["source"] and r["length"] == len(base["response_ids"])
                    for r, base in zip(scored, rows, strict=True)
                )
                entropy_error = max(
                    abs(a["stats"]["all"]["hq"] - b["stats"]["all"]["hq"]) for a, b in zip(initial, scored, strict=True)
                )
                assert entropy_error < 1e-4, (arm.name, step, entropy_error)
                if step < arm.updates and step % study.training.reuse == 0:
                    group = chr(65 + step // study.training.reuse)
                    expected = np.mean(
                        [r["stats"]["all"]["rkl"] / study.training.horizon for r in scored if r["group"] == group]
                    )
                    error = abs(expected - history[step]["loss"])
                    assert error < 1e-5, (arm.name, step, error)
                    checks.append(
                        dict(arm=arm.name, step=step + 1, loss_error=error, teacher_entropy_error=entropy_error)
                    )
                for panel, groups in PANELS.items():
                    selected = [r for r in scored if r["group"] in groups]
                    before = [r for r in initial if r["group"] in groups]
                    if not selected:
                        continue
                    for metric in ["rkl", "js"]:
                        a = np.array([r["stats"]["all"][metric] / study.training.horizon for r in selected])
                        b = np.array([r["stats"]["all"][metric] / study.training.horizon for r in before])
                        assert np.isfinite(a).all() and np.isfinite(b).all()
                        curves.append(
                            dict(
                                arm=arm.name,
                                teacher=arm.teacher,
                                step=step,
                                panel=panel,
                                metric=metric,
                                value=float(a.mean()),
                                initial=float(b.mean()),
                            )
                        )
                        if step in arm.evaluate_at:
                            sample = rng.integers(0, len(a), size=(5000, len(a)))
                            denominator = b[sample].mean(1)
                            defined = b.mean() > 0 and np.all(denominator > 0)
                            drops = 100 * (1 - a[sample].mean(1) / denominator) if defined else None
                            risks.append(
                                dict(
                                    arm=arm.name,
                                    teacher=arm.teacher,
                                    student=study.student,
                                    step=step,
                                    panel=panel,
                                    metric=metric,
                                    prompts=len(a),
                                    initial=float(b.mean()),
                                    final=float(a.mean()),
                                    drop_percent=float(100 * (1 - a.mean() / b.mean())) if b.mean() > 0 else None,
                                    ci_low=float(np.quantile(drops, 0.025)) if defined else None,
                                    ci_high=float(np.quantile(drops, 0.975)) if defined else None,
                                )
                            )
            performance.append(
                dict(
                    arm=arm.name,
                    updates=arm.updates,
                    median_update_seconds=float(np.median([r["update_seconds"] for r in history[1:]])),
                    training_seconds=sum(r["update_seconds"] + r["cache_seconds"] for r in history),
                    wall_seconds=read(root / "complete.json")["elapsed"],
                    peak_gib=max(r["peak_gib"] for r in history),
                )
            )
        for step in arm.evaluate_at:
            checkpoint = checkpoint_directory(study, arm, step) / "complete.json"
            label = f"{arm.name}-{step:03d}"
            if not checkpoint.exists():
                missing.append("endpoint-" + label)
                continue
            check_inputs(checkpoint, dict(scientific_inputs(study, arm), step=step))
            saved = read(checkpoint)
            data = task_data(study, study.student, saved["weight_identity"])
            if data is None:
                missing.append("endpoint-" + label)
                continue
            accuracy_rows(label, "endpoint", study.student, data, arm.name, step)
            before, teacher = baselines[study.student], baselines[arm.teacher]
            if before is None or teacher is None:
                continue
            means_all = []
            boots_all = []
            for dataset in datasets:
                means, boots = summarize_gap(
                    before["vectors"][dataset], teacher["vectors"][dataset], data["vectors"][dataset], rng
                )
                means_all.append(means)
                boots_all.append(boots)
                gaps.append(
                    dict(
                        arm=arm.name,
                        teacher=arm.teacher,
                        student=study.student,
                        step=step,
                        dataset=dataset,
                        **gap_row(means, boots),
                    )
                )
            gaps.append(
                dict(
                    arm=arm.name,
                    teacher=arm.teacher,
                    student=study.student,
                    step=step,
                    dataset="macro5",
                    **gap_row(np.mean(means_all, axis=0), np.mean(boots_all, axis=0)),
                )
            )
    from examples.mirroropd.research.studies.fixed_transfer.validation import copy_inputs

    interfaces = read(study.directory / "interfaces.json")
    for teacher in sorted({a.teacher for a in study.arms}):
        if not same_parameter_layout(interfaces[study.student]["configuration"], interfaces[teacher]["configuration"]):
            continue
        root = study.directory / "teacher-copy" / teacher
        if not complete(root, copy_inputs(study, teacher)):
            missing.append("copy-" + teacher)
            continue
        item = read(root / "complete.json")
        scored = read(root / "scores.json")
        assert len(scored) == 64 and item["training_updates"] == 0
        for metric in ["rkl", "js"]:
            measured = np.mean([r["stats"]["all"][metric] / study.training.horizon for r in scored])
            assert abs(item[metric] - measured) < 1e-10
        copies.append(item)
    for name, values in [
        ("accuracies", accuracy),
        ("gap-recovery", gaps),
        ("endpoints", risks),
        ("risk-curves", curves),
        ("performance", performance),
        ("block-checks", checks),
    ]:
        if values:
            csv_write(directory / f"{name}.csv", values)
    summary = dict(
        study=study.name,
        student=study.student,
        accuracy=accuracy,
        gap_recovery=gaps,
        risks=risks,
        performance=performance,
        copies=copies,
        numerical_profiles=numerical_profiles,
        missing=missing,
        statistics="5000 paired prompt bootstrap draws, seed42; one training seed",
    )
    write(directory / "summary.json", summary)
    if not partial:
        expected_evaluations = len(study.models) + sum(len(a.evaluate_at) for a in study.arms)
        assert not missing, missing
        assert len(accuracy) == expected_evaluations * len(datasets)
        assert len(gaps) == sum(len(a.evaluate_at) for a in study.arms) * (len(datasets) + 1)
        expected_risks = sum(
            2
            * len(arm.evaluate_at)
            * sum(any(r["group"] in groups for r in required_rows(study, arm)) for groups in PANELS.values())
            for arm in study.arms
        )
        assert len(risks) == expected_risks
        assert len(checks) == sum(arm.updates // study.training.reuse for arm in study.arms)
        write(
            directory / "verification.json",
            dict(
                complete=True,
                study=study.identity,
                scientific_updates=sum(a.updates for a in study.arms),
                evaluations=expected_evaluations,
                answers=expected_evaluations * len(prompts) * study.evaluation.samples,
                gap_rows=len(gaps),
                risk_rows=len(risks),
                block_checks=len(checks),
            ),
        )
    return summary
