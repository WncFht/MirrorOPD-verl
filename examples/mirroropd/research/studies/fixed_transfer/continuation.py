# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired prompt-level absolute risk changes across completed training stages."""

import argparse
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import read, write_csv
from examples.mirroropd.research.core.statistics import paired_mean_difference
from examples.mirroropd.research.studies.fixed_transfer.analysis import PANELS
from examples.mirroropd.research.studies.fixed_transfer.spec import Study
from examples.mirroropd.research.studies.fixed_transfer.training import arm_directory


def compare(study):
    results = []
    for arm in study.arms:
        root = arm_directory(study, arm)
        if arm.updates <= 64 or not (root / "complete.json").exists():
            continue
        panels = dict(PANELS, first_batch_a=["A"])
        for panel, groups in panels.items():
            before_step = study.training.reuse if panel == "first_batch_a" else 64
            states = []
            for step in [before_step, arm.updates]:
                rows = read(root / f"fixed-{step:03d}.json")
                states.append({r["id"]: r for r in rows if r["group"] in groups})
            before, after = states
            if not before:
                continue
            if before.keys() != after.keys():
                raise ValueError("Comparison prompt identities changed")
            for key in before:
                if any(before[key][field] != after[key][field] for field in ["source", "length", "group"]):
                    raise ValueError("Comparison states changed")
            ids = sorted(before)
            for metric in ["rkl", "js"]:
                values = [
                    np.array([state[i]["stats"]["all"][metric] / state[i]["stats"]["all"]["tokens"] for i in ids])
                    for state in states
                ]
                results.append(
                    dict(
                        arm=arm.name,
                        panel=panel,
                        metric=metric,
                        before_step=before_step,
                        after_step=arm.updates,
                        prompts=len(ids),
                        **paired_mean_difference(*values),
                        inference="paired prompt bootstrap; 5000 draws; seed42; one training run",
                    )
                )
    write_csv(study.directory / "analysis/continuation-risk.csv", results)
    return results


def compare_windows(study, window=64):
    """Use the same H-budget normalization and pooled prompts as the main RKL figures."""
    if window <= 0 or window % study.training.score_interval:
        raise ValueError("Window must be a positive multiple of the scoring interval")
    results = []
    panels = dict(PANELS)
    panels["held_all"] = PANELS["held_math"] + PANELS["held_code"] + PANELS["held_if"]
    for arm in study.arms:
        root = arm_directory(study, arm)
        if not (root / "complete.json").exists():
            continue
        end = arm.updates
        intervals = {(0, end)}
        for stop in [end - window, end]:
            if stop >= window:
                intervals.add((stop - window, stop))
        origin = (study.continuation or {}).get("resume_from", {}).get(arm.name)
        if origin:
            intervals.add((read(Path(origin) / "complete.json")["updates"], end))
        for start, stop in sorted(intervals):
            snapshots = [read(root / f"fixed-{step:03d}.json") for step in [start, stop]]
            for panel, groups in panels.items():
                states = [{r["id"]: r for r in snapshot if r["group"] in groups} for snapshot in snapshots]
                before, after = states
                if not before:
                    continue
                if before.keys() != after.keys() or any(
                    before[key][field] != after[key][field] for key in before for field in ["source", "length", "group"]
                ):
                    raise ValueError("Window comparison states changed")
                ids = sorted(before)
                for metric in ["rkl", "js"]:
                    values = [
                        [state[i]["stats"]["all"][metric] / study.training.horizon for i in ids] for state in states
                    ]
                    result = paired_mean_difference(*values)
                    results.append(
                        dict(
                            arm=arm.name,
                            panel=panel,
                            metric=metric,
                            before_step=start,
                            after_step=stop,
                            prompts=len(ids),
                            normalization="sum / (prompts * horizon)",
                            relative_drop_percent=100 * (1 - result["after"] / result["before"])
                            if result["before"] > 0
                            else None,
                            **result,
                        )
                    )
    write_csv(study.directory / "analysis/continuation-windows.csv", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window", type=int, help="Compare equal windows and stage origin with H normalization")
    args = parser.parse_args()
    study = Study.load(args.plan)
    result = compare(study) if args.window is None else compare_windows(study, args.window)
    print("comparison rows", len(result))


if __name__ == "__main__":
    main()
