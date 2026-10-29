# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Audit completed TB0--TB2 evidence and aggregate prompt-level paired risks.

Ledger: each arm is a distinct initial-student/teacher pair. One line point is
the mean risk over a fixed prompt panel after an actual update count. Risk is
the per-response sum divided by H1024; relative changes use each arm's own
initial risk. Confidence intervals resample prompts, never training seeds.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .common import ARMS, CACHE, DATASETS, GROUPS, MODELS, OUT, dump, read

PANELS = {
    "train_ad": list("ABCD"),
    "future_eh": list("EFGH"),
    "held_math": ["M01", "M02", "M03", "M04"],
    "held_code": ["Code01", "Code02"],
    "held_if": ["IF01", "IF02"],
}
METRICS = ["rkl", "js", "tv", "fkl_shared", "hp", "hq", "overlap16", "common_p16", "common_q16"]


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = OUT / "analysis"
    out.mkdir(exist_ok=True)
    risks, endpoints, baselines, cross, blocks, performance = [], [], [], [], [], []
    influence = []
    rng = np.random.default_rng(2026102804)
    assert read(OUT / "numerical/cpu-vjp.json")["max_gradient_error"] < 1e-12
    for name in MODELS:
        audit = read(OUT / "numerical" / f"{name}.json")
        assert max(p["max_rkl"] for p in audit["probes"]) < 1e-5
    for name in ["r1", "r7"]:
        assert read(OUT / "numerical" / f"self-{name}.json")["grad_norm"] < 1e-5
    for arm in ["TB1-J", "TB2-P7"]:
        assert read(OUT / "smoke" / arm / "complete.json")["updates"] == 1
    for arm in ARMS:
        directory = OUT / "arms" / arm
        assert read(directory / "complete.json")["updates"] == 64
        history = read(directory / "training.json")
        assert [x["step"] for x in history] == list(range(1, 65))
        assert [x["group"] for x in history] == [g for g in "ABCD" for _ in range(16)]
        assert all(np.isfinite(x["loss"]) and x["parameter_step_norm"] > 0 for x in history)
        keys = ["backward_seconds", "snapshot_seconds", "optimizer_seconds", "displacement_measure_seconds"]
        perf = dict(
            arm=arm,
            student=ARMS[arm][0],
            updates=64,
            measurement=history[-1].get("displacement_measurement", "CPU snapshots and CPU reduction"),
            measured_update_seconds=sum(sum(r[k] for k in keys) for r in history),
            wall_after_initial_score=read(directory / "complete.json")["elapsed"],
            max_peak_gib=max(max(r["peak_gib"]) for r in history),
        )
        for key in keys:
            perf["median_" + key] = float(np.median([r[key] for r in history[1:]]))
        performance.append(perf)
        for step in [32, 64]:
            checkpoint = read(CACHE / "checkpoints" / arm / f"step{step:03d}/complete.json")
            assert checkpoint["optimizer_steps"] == [step]
        initial = read(directory / "fixed-000.json")
        assert len(initial) == 256
        ids = [r["id"] for r in initial]
        for index, group in enumerate("ABCD"):
            left = read(directory / f"fixed-{16 * index:03d}.json")
            right = read(directory / f"fixed-{16 * (index + 1):03d}.json")
            local0 = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in left if r["group"] == group])
            local1 = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in right if r["group"] == group])
            held0 = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in left if r["group"] in PANELS["held_math"]])
            held1 = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in right if r["group"] in PANELS["held_math"]])
            assert abs(history[16 * index]["loss"] - local0) < 1e-5
            for evaluated in GROUPS:
                initial_risk = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in initial if r["group"] == evaluated])
                before_risk = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in left if r["group"] == evaluated])
                after_risk = np.mean([r["stats"]["all"]["rkl"] / 1024 for r in right if r["group"] == evaluated])
                influence.append(
                    dict(
                        arm=arm,
                        training_group=group,
                        evaluated_group=evaluated,
                        initial=float(initial_risk),
                        before=float(before_risk),
                        after=float(after_risk),
                        drop_initial_pp=float(100 * (before_risk - after_risk) / initial_risk),
                        old_trained_batch=evaluated in list("ABCD")[:index],
                    )
                )
            blocks.append(
                dict(
                    arm=arm,
                    group=group,
                    update_start=16 * index,
                    update_end=16 * (index + 1),
                    local_before=float(local0),
                    local_after=float(local1),
                    local_drop_percent=float(100 * (1 - local1 / local0)),
                    held_before=float(held0),
                    held_after=float(held1),
                    held_drop_percent=float(100 * (1 - held1 / held0)),
                )
            )
        for step in [0, 16, 32, 48, 64]:
            rows = read(directory / f"fixed-{step:03d}.json")
            assert [r["id"] for r in rows] == ids
            for panel, groups in PANELS.items():
                before = [r for r in initial if r["group"] in groups]
                after = [r for r in rows if r["group"] in groups]
                assert len(before) == len(after)
                for metric in METRICS:
                    b = np.array([r["stats"]["all"][metric] / 1024 for r in before])
                    a = np.array([r["stats"]["all"][metric] / 1024 for r in after])
                    risks.append(
                        dict(
                            arm=arm,
                            step=step,
                            panel=panel,
                            metric=metric,
                            prompts=len(a),
                            initial=float(b.mean()),
                            value=float(a.mean()),
                            relative_drop=float(1 - a.mean() / b.mean()) if b.mean() else None,
                        )
                    )
                    if step == 64 and metric in ["rkl", "js"]:
                        sample = rng.integers(0, len(a), size=(4000, len(a)))
                        drops = 1 - a[sample].mean(1) / b[sample].mean(1)
                        endpoints.append(
                            dict(
                                arm=arm,
                                panel=panel,
                                metric=metric,
                                initial=float(b.mean()),
                                final=float(a.mean()),
                                drop_percent=float((1 - a.mean() / b.mean()) * 100),
                                ci_low=float(np.quantile(drops, 0.025) * 100),
                                ci_high=float(np.quantile(drops, 0.975) * 100),
                            )
                        )
            # Teacher entropy must be invariant on identical cached inputs.
            assert (
                max(abs(a["stats"]["all"]["hq"] - b["stats"]["all"]["hq"]) for a, b in zip(initial, rows, strict=True))
                < 1e-4
            )
    for name in MODELS:
        for template in ["shared", "native"]:
            directory = OUT / "baseline" / f"{name}-{template}"
            if template == "native" and not read(OUT / "interface-audit.json")[name]["template_different_indices"]:
                continue
            assert read(directory / "complete.json")["records"] == 2520
            records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
            assert len({(r["prompt_id"], r["sample_index"]) for r in records}) == 2520
            for dataset, n in DATASETS.items():
                selected = [r for r in records if r["data_source"] == dataset]
                assert len(selected) == n * 4 and all(r["status"] == "ok" for r in selected)
                by_prompt = {}
                for r in selected:
                    by_prompt.setdefault(r["prompt_id"], []).append(float(r["acc"]))
                acc = np.array([np.mean(v) for v in by_prompt.values()])
                boots = acc[rng.integers(0, len(acc), size=(4000, len(acc)))].mean(1)
                baselines.append(
                    dict(
                        model=name,
                        template=template,
                        dataset=dataset,
                        prompts=n,
                        samples=4,
                        mean_percent=float(acc.mean() * 100),
                        ci_low=float(np.quantile(boots, 0.025) * 100),
                        ci_high=float(np.quantile(boots, 0.975) * 100),
                        mean_length=float(np.mean([r["response_length"] for r in selected])),
                        format_percent=float(np.mean([float(r["format_score"]) for r in selected]) * 100),
                        missing_answer_count=sum(r.get("pred") in [None, ""] for r in selected),
                        length_stop_percent=float(np.mean([r["finish_reason"] == "length" for r in selected]) * 100),
                    )
                )
    for arm in ["TB1-P", "TB2-P7", "TB2-L1", "TB2-L7"]:
        for source in ["r1", "r7"]:
            b = read(OUT / "cross" / arm / f"{source}-000.json")
            a = read(OUT / "cross" / arm / f"{source}-064.json")
            assert len(a) == len(b) == 64
            if source == ARMS[arm][0]:
                for step, scored in [(0, b), (64, a)]:
                    original = [
                        r
                        for r in read(OUT / "arms" / arm / f"fixed-{step:03d}.json")
                        if r["group"] in PANELS["held_math"]
                    ]
                    assert [r["id"] for r in original] == [r["id"] for r in scored]
                    difference = max(
                        abs(x["stats"]["all"]["rkl"] - y["stats"]["all"]["rkl"]) / 1024
                        for x, y in zip(original, scored, strict=True)
                    )
                    assert difference < 1e-5, (arm, step, "checkpoint readout mismatch", difference)
            for metric in ["rkl", "js"]:
                before = np.mean([r["stats"]["all"][metric] / 1024 for r in b])
                after = np.mean([r["stats"]["all"][metric] / 1024 for r in a])
                cross.append(
                    dict(
                        arm=arm,
                        source=source,
                        metric=metric,
                        initial=float(before),
                        final=float(after),
                        drop_percent=float(100 * (1 - after / before)),
                    )
                )
    csv_write(out / "risk-curves.csv", risks)
    csv_write(out / "endpoints.csv", endpoints)
    csv_write(out / "baselines.csv", baselines)
    csv_write(out / "cross-input.csv", cross)
    csv_write(out / "block-transfer.csv", blocks)
    csv_write(out / "performance.csv", performance)
    csv_write(out / "batch-influence.csv", influence)
    matched = []
    for arm in ARMS:
        train = sorted(
            [r for r in risks if r["arm"] == arm and r["panel"] == "train_ad" and r["metric"] == "js"],
            key=lambda r: r["step"],
        )
        held = {r["step"]: r for r in risks if r["arm"] == arm and r["panel"] == "held_math" and r["metric"] == "js"}
        # Coordinates for a descriptive matched-fit plot, not success thresholds.
        for target in [0.10, 0.25, 0.50]:
            result = dict(
                arm=arm, train_js_drop_target=target, covered=False, update_coordinate=None, held_js_drop=None
            )
            for left, right in zip(train[:-1], train[1:], strict=True):
                lo, hi = left["relative_drop"], right["relative_drop"]
                if lo <= target <= hi and hi > lo:
                    fraction = (target - lo) / (hi - lo)
                    result.update(
                        covered=True,
                        update_coordinate=left["step"] + fraction * (right["step"] - left["step"]),
                        held_js_drop=held[left["step"]]["relative_drop"]
                        + fraction * (held[right["step"]]["relative_drop"] - held[left["step"]]["relative_drop"]),
                    )
                    break
            matched.append(result)
    csv_write(out / "matched-fit.csv", matched)
    dump(
        out / "summary.json",
        dict(
            endpoints=endpoints,
            baselines=baselines,
            cross=cross,
            matched_fit=matched,
            blocks=blocks,
            performance=performance,
        ),
    )
    dump(
        out / "verification.json",
        dict(
            arms=9,
            updates=576,
            fixed_scores=9 * 5 * 256,
            checkpoint_counts=18,
            baseline_models=8,
            native_extra="light",
            cross_cells=16,
            baseline_responses=9 * 2520,
            statistical_unit="prompt; one training seed",
            all_checks_passed=True,
        ),
    )
    print(json.dumps(endpoints, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
