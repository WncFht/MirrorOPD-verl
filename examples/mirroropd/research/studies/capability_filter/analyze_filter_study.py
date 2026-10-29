#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate teacher acquisition, transfer, behavioral drift and task retention."""

import argparse
import collections
import statistics as st
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.statistics import mean_ci as ci


def panels(rows):
    result = {}
    for view in ["student", "teacher"]:
        for role in ["train", "test"]:
            for domain in ["math", "code", "if"]:
                rs = [r for r in rows if r["view"] == view and r["role"] == role and r["domain"] == domain]
                if rs:
                    result[f"{view}-{role}-{domain}"] = rs
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    root = a.data
    summaries = []
    fresh = []
    tasks = []
    bins = []
    selectivity = []
    controls = []
    progress = []
    base_task = read(root / "task-student-baseline.json")
    base_task = (
        {r["id"]: r for r in base_task} if len(base_task) == 64 and all("grade" in r for r in base_task) else None
    )
    for teacher in ["justrl", "polaris"]:
        proxy = read(root / f"transfer-proxy-{teacher}.json")
        if not proxy["complete"]:
            continue
        records = proxy["records"]
        controls.append(
            dict(
                teacher=teacher,
                diagnostic_test_gradient_cosine=proxy["diagnostic_test_cosine"],
                train_score_diagnostic_test_correlation=float(
                    np.corrcoef([r["diagnostic_cosine"] for r in records], [r["test_cosine"] for r in records])[0, 1]
                ),
                diagnostic_range=[
                    min(r["diagnostic_cosine"] for r in records),
                    max(r["diagnostic_cosine"] for r in records),
                ],
            )
        )
    for cfg in read(root / "protocol.json")["arms"]:
        arm = cfg["name"]
        d = root / arm
        if not (d / "fixed-000.json").exists():
            continue
        before = panels(read(d / "fixed-000.json"))
        proxy = {r["id"]: r for r in read(root / f"transfer-proxy-{cfg['teacher']}.json")["records"]}
        for file in sorted(d.glob("fixed-*.json")):
            step = int(file.stem.split("-")[-1])
            after = panels(read(file))
            for panel, rs in after.items():
                initial = {r["id"]: r for r in before[panel]}
                out = cfg | dict(step=step, panel=panel, prompts=len(rs))
                for metric in [
                    "rkl",
                    "fkl_shared",
                    "js",
                    "tv",
                    "retain_fkl",
                    "retain_rkl",
                    "retain_js",
                    "reference_nll",
                    "teacher_reference_nll",
                ]:
                    b = st.mean(initial[r["id"]]["stats"]["all"][metric] / 1024 for r in rs)
                    value = st.mean(r["stats"]["all"][metric] / 1024 for r in rs)
                    out[metric] = value
                    out[metric + "_before"] = b
                    if metric in ["rkl", "fkl_shared", "js", "tv"]:
                        out[metric + "_drop"] = 1 - value / b
                tokens = sum(r["length"] for r in rs)
                for metric in [
                    "hp",
                    "hq",
                    "overlap16",
                    "common_p16",
                    "common_q16",
                    "q_outside_initial_top16",
                    "p_outside_initial_top16",
                    "p0_outside_initial_top16",
                    "teacher_only_initial_p",
                    "teacher_only_initial_p0",
                    "teacher_only_initial_q",
                    "outside_student_mass",
                ]:
                    out[metric] = sum(r["stats"]["all"][metric] for r in rs) / tokens
                summaries.append(out)
            # Initial proxy is shared across all methods; interpretation focuses on fixed-student R/F controls.
            rs = after["student-train-math"]
            initial = {r["id"]: r for r in before["student-train-math"]}
            for batch in range(1, 9):
                for label in ["high", "low"]:
                    selected = [
                        r for r in rs if proxy[r["id"]]["batch"] == batch and proxy[r["id"]]["transfer_bin"] == label
                    ]
                    assert len(selected) == 8
                    base = st.mean(initial[r["id"]]["stats"]["all"]["js"] / 1024 for r in selected)
                    value = st.mean(r["stats"]["all"]["js"] / 1024 for r in selected)
                    tokens = sum(r["length"] for r in selected)
                    bins.append(
                        cfg
                        | dict(
                            step=step,
                            batch=batch,
                            transfer_bin=label,
                            js_before=base,
                            js_after=value,
                            js_drop=1 - value / base,
                            initial_entropy=st.mean(initial[r["id"]]["stats"]["all"]["hp"] / 1024 for r in selected),
                            initial_outside_mass=st.mean(
                                initial[r["id"]]["stats"]["all"]["q_outside_initial_top16"] / 1024 for r in selected
                            ),
                        )
                    )
        baseline_fresh = root / f"baseline-fresh-{cfg['teacher']}.json"
        if baseline_fresh.exists():
            initial = read(baseline_fresh)
            old = {(r["id"], r["seed"]): r for r in initial}
            for file in sorted(d.glob("fresh-*.json")):
                rows = read(file)
                if len(rows) != 64:
                    continue
                delta = collections.defaultdict(list)
                for r in rows:
                    delta[r["id"]].append(r["stats"]["all"]["rkl"] - old[r["id"], r["seed"]]["stats"]["all"]["rkl"])
                values = [st.mean(v) for v in delta.values()]
                fresh.append(
                    cfg
                    | dict(
                        step=rows[0]["step"],
                        sequence_rkl=st.mean(r["stats"]["all"]["rkl"] for r in rows),
                        before=st.mean(r["stats"]["all"]["rkl"] for r in initial),
                        difference=st.mean(values),
                        paired_prompt_ci=ci(values),
                        mean_length=st.mean(len(r["response_ids"]) for r in rows),
                    )
                )
        file = d / "task.json"
        if file.exists() and base_task:
            rows = read(file)
            if len(rows) == 64 and all("grade" in r for r in rows):
                differences = [int(r["grade"]["acc"]) - int(base_task[r["id"]]["grade"]["acc"]) for r in rows]
                cells = collections.Counter(
                    (bool(base_task[r["id"]]["grade"]["acc"]), bool(r["grade"]["acc"])) for r in rows
                )
                tasks.append(
                    cfg
                    | dict(
                        answers=64,
                        correct=sum(r["grade"]["acc"] for r in rows),
                        accuracy=st.mean(r["grade"]["acc"] for r in rows),
                        accuracy_delta=st.mean(differences),
                        paired_prompt_ci=ci(differences),
                        retained_correct=cells[True, True],
                        lost_correct=cells[True, False],
                        lost_correct_capped=sum(
                            base_task[r["id"]]["grade"]["acc"]
                            and not r["grade"]["acc"]
                            and r["finish_reason"] == "length"
                            for r in rows
                        ),
                        lost_correct_natural_stop=sum(
                            base_task[r["id"]]["grade"]["acc"]
                            and not r["grade"]["acc"]
                            and r["finish_reason"] == "stop"
                            for r in rows
                        ),
                        new_correct=cells[False, True],
                        both_wrong=cells[False, False],
                        mean_length=st.mean(len(r["response_ids"]) for r in rows),
                        length_stops=sum(r["finish_reason"] == "length" for r in rows),
                    )
                )
        history = read(d / "training.json") if (d / "training.json").exists() else []
        progress.append(
            cfg
            | dict(
                updates=len(history),
                complete=(d / "complete.json").exists(),
                clipped_updates=sum(r["grad_norm"] > 1 for r in history),
            )
        )
    for teacher in ["justrl", "polaris"]:
        for step in [32, 64, 96, 128]:

            def get(direction, batch, label, teacher=teacher, step=step):
                return next(
                    (
                        r
                        for r in bins
                        if r["teacher"] == teacher
                        and r["source"] == "student"
                        and r["direction"] == direction
                        and r["step"] == step
                        and r["batch"] == batch
                        and r["transfer_bin"] == label
                    ),
                    None,
                )

            values = []
            for batch in range(1, 9):
                rh, rl, fh, fl = [
                    get(d, batch, left_values)
                    for d, left_values in [
                        ("reverse", "high"),
                        ("reverse", "low"),
                        ("forward", "high"),
                        ("forward", "low"),
                    ]
                ]
                if any(x is None for x in [rh, rl, fh, fl]):
                    continue
                values.append((rh["js_drop"] - rl["js_drop"]) - (fh["js_drop"] - fl["js_drop"]))
            if len(values) == 8:
                selectivity.append(
                    dict(
                        teacher=teacher,
                        step=step,
                        interaction=st.mean(values),
                        paired_batch_ci=ci(values),
                        batches=8,
                        definition=(
                            "(RKL high-low initial-proxy JS acquisition) minus (FKL high-low); "
                            "positive favors stronger RKL selectivity. Exploratory initial-gradient proxy, "
                            "not an intrinsic skill label."
                        ),
                    )
                )
    paired = []
    pairs = []
    for t in ["justrl", "polaris"]:
        pairs += [(f"{t}-{source}-reverse", f"{t}-{source}-forward") for source in ["student", "teacher", "online"]]
        pairs += [(f"{t}-teacher-sft", f"{t}-teacher-forward"), (f"{t}-teacher-sft", f"{t}-online-reverse")]
    for left, right in pairs:
        left_values = root / left / "fixed-128.json"
        r = root / right / "fixed-128.json"
        if not (left_values.exists() and r.exists()):
            continue
        li = panels(read(left_values))
        ri = panels(read(r))
        for panel in ["student-test-math", "student-test-code", "student-test-if", "teacher-test-math"]:
            right_index = {v["id"]: v for v in ri[panel]}
            for metric in ["js", "rkl", "fkl_shared", "retain_fkl"]:
                delta = [
                    (v["stats"]["all"][metric] - right_index[v["id"]]["stats"]["all"][metric]) / 1024 for v in li[panel]
                ]
                paired.append(
                    dict(
                        left=left,
                        right=right,
                        panel=panel,
                        metric=metric,
                        difference=st.mean(delta),
                        paired_prompt_ci=ci(delta),
                    )
                )
    for name, value in [
        ("risk-summary.json", summaries),
        ("fresh-summary.json", fresh),
        ("task-summary.json", tasks),
        ("transfer-bins.json", bins),
        ("selectivity-summary.json", selectivity),
        ("paired-controls.json", paired),
        ("proxy-summary.json", controls),
        ("progress-summary.json", progress),
    ]:
        dump(root / name, value)
    print(
        "arms with fixed baseline",
        len(progress),
        "completed",
        sum(r["complete"] for r in progress),
        "graded endpoints",
        len(tasks),
    )
    print("selectivity", selectivity[-2:])


if __name__ == "__main__":
    main()
