#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prompt-paired summaries; no token-independent significance assumptions."""

import argparse
import collections
import json
import statistics as st
from pathlib import Path

from ..common.io import dump
from ..common.statistics import mean_ci as paired_ci


def groups(rows, keys):
    out = collections.defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def mean(rs, k):
    return st.mean(r[k] for r in rs)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", type=Path, required=True)
    args = a.parse_args()
    p = args.data
    init = json.loads((p / "initial-state-metrics.json").read_text())
    stats = []
    rollouts = [
        r
        for source in ["student", "justrl", "polaris"]
        for r in json.loads((p / f"rollouts-{source}.json").read_text())
    ]
    endpoints = {
        (r["id"], r["source"]): r["response_ids"].index(151649)
        if 151649 in r["response_ids"]
        else len(r["response_ids"])
        for r in rollouts
    }
    for r in init:
        r["phase"] = "reasoning" if r["position"] <= endpoints[(r["prompt_id"], r["source"])] else "answer"
    keys = ["teacher", "source", "domain", "region"]
    metrics = [
        "kl",
        "overlap1",
        "overlap8",
        "overlap16",
        "overlap32",
        "overlap64",
        "overlap_student_mass16",
        "overlap_teacher_mass16",
        "overlap_conditional_kl16",
        "overlap_advantage16",
        "entropy_student",
        "entropy_teacher",
        "positive_sample_signal",
        "positive_signal_mass_under_student",
        "sample_log_gap",
        "total_variation",
        "logit_gradient_norm",
    ]
    for group, rs in groups(init, keys).items():
        stats.append(
            dict(
                zip(keys, group, strict=False),
                n_positions=len(rs),
                n_prompts=len({r["prompt_id"] for r in rs}),
                **{k: mean(rs, k) for k in metrics},
            )
        )
    dump(p / "initial-summary.json", stats)
    phase = []
    for group, rs in groups(init, ["teacher", "source", "domain", "phase"]).items():
        phase.append(
            dict(
                zip(["teacher", "source", "domain", "phase"], group, strict=False),
                n_positions=len(rs),
                **{k: mean(rs, k) for k in metrics},
            )
        )
    dump(p / "phase-summary.json", phase)
    contrasts = []
    for teacher in ["justrl", "polaris"]:
        for region in ["early", "late"]:
            for metric in ["kl", "overlap64"]:
                selected = [
                    r for r in init if r["teacher"] == teacher and r["domain"] == "math" and r["region"] == region
                ]
                by = groups(selected, ["prompt_id", "source"])
                delta = [mean(by[(i, teacher)], metric) - mean(by[(i, "student")], metric) for i in range(16)]
                contrasts.append(
                    dict(
                        teacher=teacher,
                        region=region,
                        metric=metric,
                        comparison="teacher-source minus student-source",
                        prompt_count=16,
                        mean_delta=st.mean(delta),
                        prompt_paired_ci=paired_ci(delta),
                        positive_prompt_count=sum(x > 0 for x in delta),
                    )
                )
    dump(p / "source-contrasts.json", contrasts)
    output = []
    for arm in sorted(p.glob("*-split*-t*-lr*")):
        if not (arm / "evaluation.json").exists():
            continue
        config = json.loads((arm / "protocol.json").read_text())
        rows = json.loads((arm / "evaluation.json").read_text())
        ids = config["train_ids"]
        base = {(r["prompt_id"], r["source"], r["position"]): r for r in rows if r["step"] == 0}
        for r in rows:
            if r["prompt_id"] in ids:
                r["partition"] = "fitted_prompts"
            elif r["split"] == "heldout":
                r["partition"] = "heldout"
            else:
                r["partition"] = "other_train_fold"
        keys = ["step", "partition", "source", "domain", "region"]
        for group, rs in groups([r for r in rows if r["step"] > 0], keys).items():
            initial = [base[(r["prompt_id"], r["source"], r["position"])] for r in rs]
            byprompt = groups(rs, ["prompt_id"])
            deltas = []
            rel = []
            for prs in byprompt.values():
                bef = [base[(r["prompt_id"], r["source"], r["position"])]["kl"] for r in prs]
                aft = [r["kl"] for r in prs]
                deltas.append(st.mean(bef) - st.mean(aft))
                rel.append(1 - st.mean(aft) / st.mean(bef))
            output.append(
                dict(
                    arm=arm.name,
                    teacher=config["teacher"],
                    training_source=config["source"],
                    temperature=config["teacher_temperature"],
                    **dict(zip(keys, group, strict=False)),
                    n_prompts=len(byprompt),
                    n_positions=len(rs),
                    kl_before=mean(initial, "kl"),
                    kl_after=mean(rs, "kl"),
                    relative_drop=1 - mean(rs, "kl") / mean(initial, "kl"),
                    prompt_mean_absolute_drop=st.mean(deltas),
                    prompt_delta_ci=paired_ci(deltas),
                    improved_prompt_fraction=st.mean(x > 0 for x in deltas),
                    overlap64_before=mean(initial, "overlap64"),
                    overlap64_after=mean(rs, "overlap64"),
                    tv_before=mean(initial, "total_variation"),
                    tv_after=mean(rs, "total_variation"),
                )
            )
    dump(p / "transfer-summary.json", output)
    for arm in sorted({r["arm"] for r in output}):
        last = max(r["step"] for r in output if r["arm"] == arm)
        print(arm, "step", last)
        for r in output:
            if (
                r["arm"] == arm
                and r["step"] == last
                and r["source"] == "student"
                and r["partition"] != "other_train_fold"
            ):
                print(
                    r["partition"],
                    r["domain"],
                    r["region"],
                    "KL",
                    round(r["kl_before"], 5),
                    "->",
                    round(r["kl_after"], 5),
                    "drop",
                    round(100 * r["relative_drop"], 2),
                )


if __name__ == "__main__":
    main()
