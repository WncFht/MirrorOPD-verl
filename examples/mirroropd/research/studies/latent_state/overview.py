# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Global teacher comparisons from saved labels, without changing the fitted geometry."""

import numpy as np

from .geometry import js_divergence


def teacher_overview(points, groups, *, seed=42, draws=30):
    """Compare equal trajectory budgets; teacher calibration uses disjoint halves.

    Initial queries match the target teacher prompts. Training queries have different
    prompts. Whole trajectories, including their correlated states, are sampled together.
    """
    eligible = [p for p in points if p["metadata"]["action"] > 0 and p["metadata"]["split"] in ("test", "query")]
    trajectories = {}
    for p in eligible:
        trajectories.setdefault(p["group"], {}).setdefault(p["trajectory"], []).append(p)
    pools = {g: sorted(trajectories[g]) for g in groups}
    teacher = pools["teacher"]
    budget = min(len(teacher) // 2, *(len(pool) for pool in pools.values()))
    if budget < 2:
        raise ValueError("Need at least four teacher trajectories for disjoint calibration")
    prompt_to_initial = {rows[0]["metadata"]["prompt"]: key for key, rows in trajectories["initial"].items()}
    teacher_prompts = {rows[0]["metadata"]["prompt"] for rows in trajectories["teacher"].values()}
    if set(prompt_to_initial) != teacher_prompts:
        raise ValueError("Initial and teacher banks must share prompts")
    result = {}
    for k in (50, 100, 200):
        counts = {
            g: np.array(
                [
                    np.bincount([p["metadata"]["clusters"][str(k)] for p in trajectories[g][t]], minlength=k)
                    for t in pool
                ]
            )
            for g, pool in pools.items()
        }
        lookup = {g: {key: i for i, key in enumerate(pool)} for g, pool in pools.items()}
        rng = np.random.default_rng(seed)
        values = {g: dict(js=[], coverage=[]) for g in groups}
        for _ in range(draws):
            first, second = np.split(rng.choice(len(teacher), budget * 2, replace=False), 2)
            target_counts = counts["teacher"][first].sum(axis=0)
            target = target_counts / target_counts.sum()
            for g in groups:
                if g == "teacher":
                    chosen = second
                elif g == "initial":
                    chosen = [
                        lookup[g][prompt_to_initial[trajectories["teacher"][teacher[i]][0]["metadata"]["prompt"]]]
                        for i in first
                    ]
                else:
                    chosen = rng.choice(len(pools[g]), budget, replace=False)
                query = counts[g][chosen].sum(axis=0)
                values[g]["js"].append(js_divergence(query, target))
                values[g]["coverage"].append(float(np.count_nonzero(query) / k))
        result[str(k)] = {
            g: {
                metric: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), values=v)
                for metric, v in measures.items()
            }
            for g, measures in values.items()
        }
        for g in groups:
            total = counts[g].sum(axis=0)
            result[str(k)][g]["occupancy"] = (total / total.sum()).tolist()
    return dict(
        budget=budget,
        draws=draws,
        seed=seed,
        target="teacher query bank",
        baseline="Disjoint teacher subsets; initial uses matched target prompts",
        metrics=result,
    )
