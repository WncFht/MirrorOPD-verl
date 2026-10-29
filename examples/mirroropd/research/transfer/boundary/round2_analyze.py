# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired prompt-level GapRecovery and full-vocabulary endpoint evidence."""

import argparse
import json

import numpy as np

from ...common.transfer_metrics import gap_row, summarize_gap
from .analyze import PANELS, csv_write
from .round2_common import ARMS, DATASETS, OLD_OUT, OUT, dump, read


def answers(directory):
    assert read(directory / "complete.json")["records"] == 2520
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    assert len(rows) == 2520
    ids = {(r["prompt_id"], r["sample_index"]) for r in rows}
    assert len(ids) == 2520
    assert all(r["status"] == "ok" and r["sample_index"] in range(4) for r in rows)
    result = {}
    for dataset, count in DATASETS.items():
        subset = [r for r in rows if r["data_source"] == dataset]
        prompts = sorted({r["prompt_id"] for r in subset})
        assert len(prompts) == count and len(subset) == 4 * count
        result[dataset] = {p: np.mean([float(r["acc"]) for r in subset if r["prompt_id"] == p]) for p in prompts}
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    dest = OUT / "analysis"
    dest.mkdir(exist_ok=True)
    specs = read(OUT / "eval-specs.json")
    rng = np.random.default_rng(2026102905)
    memo = {}

    def data(path):
        if path not in memo:
            memo[path] = answers(path)
        return memo[path]

    gaps, accuracies, risks, performance, missing, curves, block_checks = [], [], [], [], [], [], []
    for name, spec in specs.items():
        directory = OUT / "evaluations" / name
        if not (directory / "complete.json").exists():
            missing.append(name)
            continue
        after, raw = data(directory)
        for dataset, mapping in after.items():
            rows = [r for r in raw if r["data_source"] == dataset]
            accuracies.append(
                dict(
                    name=name,
                    kind=spec["kind"],
                    teacher=spec["teacher"],
                    student=spec["student"],
                    dataset=dataset,
                    template=spec["template"],
                    cap=spec["response_cap"],
                    acc_percent=100 * np.mean(list(mapping.values())),
                    mean_length=np.mean([r["response_length"] for r in rows]),
                    mean_repeat8=np.mean([r["repeat8"] for r in rows]),
                    length_stop_percent=100 * np.mean([r["finish_reason"] == "length" for r in rows]),
                )
            )
        if spec["kind"] == "teacher":
            continue
        teacher_dir = OUT / "evaluations" / ("teacher-" + spec["teacher"])
        if not (teacher_dir / "complete.json").exists():
            teacher_dir = OLD_OUT / "baseline" / (spec["teacher"] + "-shared")
        teacher, _ = data(teacher_dir)
        before, _ = data(OLD_OUT / "baseline" / (spec["student"] + "-shared"))
        means_all, boots_all = [], []
        for dataset in DATASETS:
            means, boots = summarize_gap(before[dataset], teacher[dataset], after[dataset], rng)
            means_all.append(means)
            boots_all.append(boots)
            gaps.append(
                dict(
                    name=name,
                    arm=spec["arm"],
                    step=spec["step"],
                    teacher=spec["teacher"],
                    student=spec["student"],
                    dataset=dataset,
                    **gap_row(means, boots),
                )
            )
        gaps.append(
            dict(
                name=name,
                arm=spec["arm"],
                step=spec["step"],
                teacher=spec["teacher"],
                student=spec["student"],
                dataset="macro5",
                **gap_row(np.mean(means_all, axis=0), np.mean(boots_all, axis=0)),
            )
        )
    continuation = []
    for arm, config in ARMS.items():
        if not config["old_arm"]:
            continue
        left = OUT / "evaluations" / f"after-{config['old_arm']}-064"
        right = OUT / "evaluations" / f"after-{arm}-128"
        if not all((d / "complete.json").exists() for d in [left, right]):
            continue
        a64, _ = data(left)
        a128, _ = data(right)
        baseline, _ = data(OLD_OUT / "baseline/r1-shared")
        teacher, _ = data(OLD_OUT / "baseline" / (config["teacher"] + "-shared"))
        means_all, boots_all = [], []
        for dataset in list(DATASETS) + ["macro5"]:
            if dataset == "macro5":
                means, boots = np.mean(means_all, axis=0), np.mean(boots_all, axis=0)
            else:
                keys = list(a64[dataset])
                assert keys == list(a128[dataset]) == list(baseline[dataset]) == list(teacher[dataset])
                values = np.array(
                    [[a64[dataset][k], a128[dataset][k], teacher[dataset][k] - baseline[dataset][k]] for k in keys]
                )
                means = values.mean(0)
                boots = values[rng.integers(0, len(keys), size=(5000, len(keys)))].mean(1)
                means_all.append(means)
                boots_all.append(boots)
            gain = means[1] - means[0]
            ci = np.quantile(boots[:, 1] - boots[:, 0], [0.025, 0.975])
            continuation.append(
                dict(
                    arm=arm,
                    teacher=config["teacher"],
                    dataset=dataset,
                    acc64_percent=100 * means[0],
                    acc128_percent=100 * means[1],
                    gain_pp=100 * gain,
                    gain_ci_low_pp=100 * ci[0],
                    gain_ci_high_pp=100 * ci[1],
                    gap_recovery_gain_pp=100 * gain / means[2] if abs(means[2]) > 1e-12 else None,
                )
            )
    for arm, config in ARMS.items():
        directory = OUT / "arms" / arm
        if not (directory / "complete.json").exists():
            missing.append("train-" + arm)
            continue
        history = read(directory / "training.json")
        assert [r["step"] for r in history] == list(range(config["start"] + 1, config["stop"] + 1))
        assert [r["group"] for r in history] == [
            chr(65 + (s - 1) // 16) for s in range(config["start"] + 1, config["stop"] + 1)
        ]
        assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in history)
        if config["old_arm"]:
            initial = read(OLD_OUT / "arms" / config["old_arm"] / "fixed-000.json")
        else:
            initial = read(directory / "fixed-000.json")
        resumed = read(directory / f"fixed-{config['start']:03d}.json")
        final = read(directory / f"fixed-{config['stop']:03d}.json")
        assert [r["id"] for r in initial] == [r["id"] for r in final] == [r["id"] for r in resumed]
        for step in range(config["start"], config["stop"] + 1, 16):
            scored = read(directory / f"fixed-{step:03d}.json")
            assert len(scored) == 256 and [r["id"] for r in scored] == [r["id"] for r in resumed]
            entropy_error = max(
                abs(a["stats"]["all"]["hq"] - b["stats"]["all"]["hq"]) for a, b in zip(resumed, scored, strict=True)
            )
            assert entropy_error < 1e-4, (arm, step, entropy_error)
            if step < config["stop"]:
                group = chr(65 + step // 16)
                risk = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in scored if r["group"] == group])
                error = abs(risk - history[step - config["start"]]["loss"])
                assert error < 1e-5, (arm, step, error)
                block_checks.append(
                    dict(
                        arm=arm,
                        step=step + 1,
                        group=group,
                        first_update_loss_error=error,
                        teacher_entropy_sum_error=entropy_error,
                    )
                )
            for pname, groups in PANELS.items():
                for metric in ["rkl", "js"]:
                    curves.append(
                        dict(
                            arm=arm,
                            step=step,
                            panel=pname,
                            metric=metric,
                            value=np.mean([r["stats"]["all"][metric] / 1024 for r in scored if r["group"] in groups]),
                        )
                    )
        for panel, groups in PANELS.items():
            for metric in ["rkl", "js"]:
                vectors = [
                    np.array([r["stats"]["all"][metric] / 1024 for r in rows if r["group"] in groups])
                    for rows in [initial, resumed, final]
                ]
                b, s, a = [v.mean() for v in vectors]
                sample = rng.integers(0, len(vectors[0]), size=(5000, len(vectors[0])))
                drops = 100 * (1 - vectors[2][sample].mean(1) / vectors[0][sample].mean(1))
                risks.append(
                    dict(
                        arm=arm,
                        teacher=config["teacher"],
                        student="r1",
                        step=config["stop"],
                        panel=panel,
                        metric=metric,
                        initial=b,
                        start=s,
                        final=a,
                        drop_percent=100 * (1 - a / b),
                        continuation_drop_percent=100 * (1 - a / s),
                        ci_low=np.quantile(drops, 0.025),
                        ci_high=np.quantile(drops, 0.975),
                    )
                )
        performance.append(
            dict(
                arm=arm,
                median_update_seconds=np.median([r["update_seconds"] for r in history[1:]]),
                cache_seconds=sum(r["cache_seconds"] for r in history),
                training_seconds=sum(r["update_seconds"] + r["cache_seconds"] for r in history),
                wall_seconds=read(directory / "complete.json")["elapsed"],
                peak_gib=max(r["peak_gib"] for r in history),
            )
        )
    references = []
    for path, (mapping, raw) in memo.items():
        if path.parent != OLD_OUT / "baseline":
            continue
        for dataset, values in mapping.items():
            selected = [r for r in raw if r["data_source"] == dataset]
            references.append(
                dict(
                    model=path.name,
                    dataset=dataset,
                    acc_percent=100 * np.mean(list(values.values())),
                    mean_length=np.mean([r["response_length"] for r in selected]),
                    mean_repeat8=np.mean([r["repeat8"] for r in selected]),
                    length_stop_percent=100 * np.mean([r["finish_reason"] == "length" for r in selected]),
                )
            )
    for name, rows in [
        ("gap-recovery", gaps),
        ("accuracies", accuracies),
        ("endpoints", risks),
        ("performance", performance),
        ("risk-curves", curves),
        ("block-checks", block_checks),
        ("continuation-accuracy", continuation),
        ("baseline-reference", references),
    ]:
        if rows:
            csv_write(dest / f"{name}.csv", rows)
    dump(
        dest / "summary.json",
        dict(
            gap_recovery=gaps,
            accuracies=accuracies,
            endpoints=risks,
            performance=performance,
            missing=missing,
            block_checks=block_checks,
            continuation_accuracy=continuation,
            statistical_unit="prompt; paired bootstrap5000; one training seed",
            math15="native baseline only, no after model or GapRecovery",
        ),
    )
    if not args.partial:
        assert not missing, missing
        assert len(gaps) == 15 * 6 and len(accuracies) == 19 * 5 and len(risks) == 6 * 5 * 2
        assert len(continuation) == 18
        dump(
            dest / "verification.json",
            dict(complete=True, updates=384, eval_answers=47880, endpoint_pairs=15, gap_rows=90, training_arms=6),
        )
    print(dict(gap_rows=len(gaps), accuracy_rows=len(accuracies), risk_rows=len(risks), missing=missing))


if __name__ == "__main__":
    main()
