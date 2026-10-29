#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired method contrasts and an exact empirical 2x2 state/model decomposition."""

import argparse
import collections
import json
import statistics as st
from pathlib import Path

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.core.statistics import mean_ci as ci


def byprompt(rows, field="rkl"):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r.get("prompt_id", r.get("id"))].append(r["stats"]["all"][field])
    return {k: st.mean(v) for k, v in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    initial_categories = []
    for teacher in ["justrl", "polaris"]:
        rows = read(root / f"baseline-{teacher}/fresh.json")
        n = sum(len(r["response_ids"]) for r in rows)
        for k in [8, 16, 64]:
            counts = [sum(r["stats"]["all"][f"category{k}_{c}"] for r in rows) for c in range(4)]
            initial_categories.append(
                dict(
                    teacher=teacher,
                    k=k,
                    tokens=n,
                    category_order=["neither", "teacher_only", "student_only", "both"],
                    counts=counts,
                    fractions=[v / n for v in counts],
                    overlap=sum(r["stats"]["all"][f"overlap{k}"] for r in rows) / n,
                    common_student_mass=sum(r["stats"]["all"][f"common_p{k}"] for r in rows) / n,
                    common_teacher_mass=sum(r["stats"]["all"][f"common_q{k}"] for r in rows) / n,
                )
            )
    dump(root / "summary-baseline-categories.json", initial_categories)
    endpoint_categories = []
    for arm in [
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]:
        rows = read(root / arm / "fresh-64.json")
        n = sum(len(r["response_ids"]) for r in rows)
        counts = [sum(r["stats"]["all"][f"category16_{c}"] for r in rows) for c in range(4)]
        endpoint_categories.append(
            dict(
                arm=arm,
                k=16,
                tokens=n,
                category_order=["neither", "teacher_only", "student_only", "both"],
                counts=counts,
                fractions=[v / n for v in counts],
                overlap=sum(r["stats"]["all"]["overlap16"] for r in rows) / n,
            )
        )
    dump(root / "summary-endpoint-categories.json", endpoint_categories)
    contrasts = []
    for left, right, step in [
        ("polaris-intersection-k16", "polaris-student-k16", 64),
        ("polaris-intersection-k16", "polaris-topk-k16", 64),
        ("polaris-teacher-k16", "polaris-student-k16", 64),
        ("selection-high", "selection-random", 16),
        ("selection-low", "selection-random", 16),
    ]:
        left_values = byprompt(read(root / left / f"fresh-{step}.json"))
        r = byprompt(read(root / right / f"fresh-{step}.json"))
        assert left_values.keys() == r.keys()
        d = [left_values[k] - r[k] for k in sorted(left_values)]
        contrasts.append(
            dict(
                left=left,
                right=right,
                step=step,
                metric="fresh sequence KL, left minus right; negative favors left",
                prompts=len(d),
                mean_difference=st.mean(d),
                paired_prompt_ci=ci(d),
            )
        )
    dump(root / "paired-method-contrasts.json", contrasts)
    result = []
    droot = root / "decomposition"
    for arm in [
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]:
        teacher = arm.split("-")[0]
        oo = byprompt(read(droot / f"{teacher}-oldmodel-oldstates.json"))
        on = byprompt(read(droot / f"{arm}-oldmodel-newstates.json"))
        no = byprompt(read(droot / f"{arm}-newmodel-oldstates.json"))
        nn = byprompt(read(droot / f"{arm}-newmodel-newstates.json"))
        gain = [oo[k] - no[k] for k in sorted(oo)]
        cost = [nn[k] - no[k] for k in sorted(oo)]
        net = [oo[k] - nn[k] for k in sorted(oo)]
        assert max(abs(g - c - n) for g, c, n in zip(gain, cost, net, strict=False)) < 1e-10
        actual0 = st.mean(byprompt(read(root / f"baseline-{teacher}/fresh.json")).values())
        actual1 = st.mean(byprompt(read(root / arm / "fresh-64.json")).values())
        result.append(
            dict(
                arm=arm,
                old_model_old_states=st.mean(oo.values()),
                old_model_new_states=st.mean(on.values()),
                new_model_old_states=st.mean(no.values()),
                new_model_new_states=st.mean(nn.values()),
                parameter_gain_on_old_states=st.mean(gain),
                visitation_cost_at_new_model=st.mean(cost),
                net_gain=st.mean(net),
                offset_fraction=st.mean(cost) / st.mean(gain),
                parameter_gain_ci=ci(gain),
                visitation_cost_ci=ci(cost),
                old_diagonal_scoring_minus_decode=st.mean(oo.values()) - actual0,
                new_diagonal_scoring_minus_decode=st.mean(nn.values()) - actual1,
            )
        )
    dump(root / "summary-decomposition.json", result)
    entropy = []
    fixed_summary = read(root / "summary-fixed.json")
    for arm in [
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]:
        teacher = arm.split("-")[0]
        before = read(root / f"baseline-{teacher}/fixed.json")
        after = read(root / arm / "fixed-64.json")
        for domain in ["math", "code", "if"]:
            for source in ["student", "justrl", "polaris"]:
                old = [r for r in before if r["domain"] == domain and r["source"] == source]
                new = [r for r in after if r["domain"] == domain and r["source"] == source]
                n = sum(r["length"] for r in old)
                metric = next(
                    r
                    for r in fixed_summary
                    if r["arm"] == arm
                    and r["step"] == 64
                    and r["domain"] == domain
                    and r["source"] == source
                    and r["band"] == "all"
                )
                entropy.append(
                    dict(
                        arm=arm,
                        domain=domain,
                        source=source,
                        tokens=n,
                        student_entropy_before=sum(r["stats"]["all"]["hp"] for r in old) / n,
                        student_entropy_after=sum(r["stats"]["all"]["hp"] for r in new) / n,
                        teacher_entropy=sum(r["stats"]["all"]["hq"] for r in old) / n,
                        rkl_drop=metric["drop"],
                        js_change=metric["js_after"] / metric["js_before"] - 1,
                        tv_change=metric["tv_after"] / metric["tv_before"] - 1,
                    )
                )
    dump(root / "entropy-by-source.json", entropy)
    task_names = [
        "initial",
        "justrl-student-k16",
        "polaris-student-k16",
        "polaris-teacher-k16",
        "polaris-intersection-k16",
        "polaris-topk-k16",
    ]
    tasks = {name: read(root / f"task-{name}.json") for name in task_names}
    assert all(len(rows) == 32 and all("grade" in r for r in rows) for rows in tasks.values())
    taskcomparisons = []
    if "initial" in tasks and all(all("grade" in r for r in rows) for rows in tasks.values()):
        baseline = {r["id"]: r for r in tasks["initial"]}
        for name, rows in tasks.items():
            if name == "initial":
                continue
            d = [float(r["grade"]["acc"]) - float(baseline[r["id"]]["grade"]["acc"]) for r in rows]
            taskcomparisons.append(dict(arm=name, accuracy_difference=st.mean(d), paired_prompt_ci=ci(d)))
    dump(root / "task-paired-contrasts.json", taskcomparisons)
    print(json.dumps(contrasts, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
