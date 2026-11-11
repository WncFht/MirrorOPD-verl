"""Broad teacher-only metrics, matched initial parent and budget.

Exclude student/remaining-weight features from predictors. Use one canonical
input condition per teacher. Permute outcomes only within initial-parent strata;
all 5! * 3! * 3! = 4320 permutations are enumerated. These describe this panel.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    out = args.output or root / "direct-pairs-analysis-v4"
    out.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(root / "direct-pairs-analysis-v4/conditions.csv")
    old = pd.read_csv(root / "curve-spectrum-v1/curve-feature-table.csv")
    inv = json.loads((root / "direct-pairs-v1/inventory.json").read_text())
    conditions = {c["run_id"]: c for c in inv["conditions"]}
    primary = current[
        (current.cohort == "continuous") & (current.panel == "historical H1024") & (current.curve_step == 320)
    ].copy()
    primary = primary[primary.run_id.map(lambda x: conditions[x]["student"] != "q3-1.7-nonthinking")]
    primary["parent"] = primary.run_id.map(lambda x: conditions[x]["base"])
    primary = primary.sort_values(["parent", "teacher"]).reset_index(drop=True)
    extra = [
        c
        for c in old.columns
        if c.startswith(("base_", "teacher_", "delta_"))
        and c not in current.columns
        and pd.api.types.is_numeric_dtype(old[c])
    ]
    primary = primary.merge(old[["run_id"] + extra], on="run_id", how="left")
    excluded = {
        "initial_K",
        "final_K",
        "remaining",
        "absolute_drop",
        "early_log_drop",
        "late_log_drop",
        "curve_step",
        "matrices",
        "seed_spread_max",
    }
    features = [c for c in primary.columns if pd.api.types.is_numeric_dtype(primary[c]) and c not in excluded]
    groups = [g.index.to_numpy() for _, g in primary.groupby("parent", sort=False)]
    assert sorted(map(len, groups)) == [3, 3, 5]
    permutations = []
    for choices in itertools.product(*(list(itertools.permutations(g)) for g in groups)):
        order = np.arange(len(primary))
        for group, choice in zip(groups, choices, strict=True):
            order[group] = choice
        permutations.append(order)
    permutations = np.asarray(permutations)

    def centered_rank(values):
        result = np.empty(len(values))
        for group in groups:
            result[group] = rankdata(values[group]) - np.mean(rankdata(values[group]))
        return result

    rows = []
    for target in ["final_K", "remaining", "late_log_drop"]:
        y = centered_rank(primary[target].to_numpy(float))
        for feature in features:
            values = primary[feature].to_numpy(float)
            if not np.isfinite(values).all():
                continue
            x = centered_rank(values)
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            if denominator < 1e-12:
                continue
            observed = float(x @ y / denominator)
            permuted = y[permutations] @ x / denominator
            p = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))
            rows.append(
                dict(
                    target=target,
                    feature=feature,
                    parent_centered_rank_correlation=observed,
                    exact_within_parent_permutation_p=p,
                    teachers=len(primary),
                )
            )
    results = pd.DataFrame(rows)
    results["BH_q"] = np.nan
    for target, group in results.groupby("target"):
        ordered = group.sort_values("exact_within_parent_permutation_p")
        raw = ordered.exact_within_parent_permutation_p.to_numpy() * len(ordered) / np.arange(1, len(ordered) + 1)
        q = np.minimum.accumulate(raw[::-1])[::-1].clip(0, 1)
        results.loc[ordered.index, "BH_q"] = q
    results.sort_values(["target", "exact_within_parent_permutation_p"]).to_csv(
        out / "parent-matched-all-metrics.csv", index=False
    )
    primary.to_csv(out / "parent-matched-primary-panel.csv", index=False)
    summary = dict(
        teachers=len(primary),
        parent_sizes=[len(g) for g in groups],
        permutations=len(permutations),
        features_per_target=results.groupby("target").size().to_dict(),
        BH_below_05=results[results.BH_q < 0.05].to_dict("records"),
        definition="within-parent centered ranks; exact within-parent label permutations; BH within target",
        selection="continuous, H1024, step320; canonical thinking input, one condition per teacher",
    )
    (out / "parent-matched-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
