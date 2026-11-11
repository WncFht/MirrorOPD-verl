# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired functional comparisons with explicit denominators and random seed controls."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def value(row, metric, normalization):
    x = row["per_token"][metric]
    if normalization == "H128":
        return sum(x) / 128
    if normalization == "actual":
        return float(np.mean(x))
    if normalization == "first32":
        return float(np.mean(x[:32]))
    if normalization == "last32":
        return float(np.mean(x[-32:]))
    raise ValueError(normalization)


def compare(a, b, metric, normalization):
    # Positive improvement is less risk/entropy under b than a.
    left = {(r["origin"], r["row_id"]): r for r in a}
    right = {(r["origin"], r["row_id"]): r for r in b}
    if set(left) != set(right) or len(left) != len(a) or len(right) != len(b):
        raise ValueError("Unpaired or duplicated question identities")
    keys = sorted(left)
    if any(left[k]["tokens"] != right[k]["tokens"] for k in keys):
        raise ValueError("Changed fixed trajectory token count")
    x = np.array([value(left[k], metric, normalization) for k in keys])
    y = np.array([value(right[k], metric, normalization) for k in keys])
    delta = x - y
    rng = np.random.default_rng(20261110)
    boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
    return dict(
        n=len(delta),
        reference_absolute=float(x.mean()),
        condition_absolute=float(y.mean()),
        improvement=float(delta.mean()),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        lower_risk_questions=int((delta > 0).sum()),
        ratio=float(y.mean() / x.mean()) if x.mean() else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    output, coverage, source_hashes = [], [], []
    for job in manifest["jobs"]:
        folder = args.run_root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
        complete = (folder / "interventions-complete.json").exists()
        coverage.append(dict(job_id=job["job_id"], complete=complete))
        if not complete:
            continue
        scores = {}
        for path in sorted((folder / "scores").glob("*.json")):
            if path.name.endswith(".progress.json"):
                continue
            scores[path.stem] = json.loads(path.read_text())["rows"]
            source_hashes.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        pairs = [("base", name) for name in scores]
        pairs += [("endpoint", name) for name in scores]
        for band in ["off", "null"]:
            target = f"missing-{band}-beta-1.0"
            pairs += [
                (f"missing-{band}-{control}-{seed}", target)
                for control in ["gaussian", "random_frames"]
                for seed in range(3)
            ]
        for reference, condition in pairs:
            if reference == condition:
                continue
            groups_a, groups_b = defaultdict(list), defaultdict(list)
            for data, groups in [(scores[reference], groups_a), (scores[condition], groups_b)]:
                for row in data:
                    groups[row["origin"], row["split"], row["domain"]].append(row)
            if set(groups_a) != set(groups_b):
                raise ValueError("Unmatched domains/splits")
            for group in groups_a:
                for normalization in ["H128", "actual", "first32", "last32"]:
                    for metric in ["rkl", "fkl", "js"]:
                        result = compare(groups_a[group], groups_b[group], metric, normalization)
                        output.append(
                            dict(
                                job_id=job["job_id"],
                                run_id=job["run_id"],
                                family=job["family"],
                                reference=reference,
                                condition=condition,
                                origin=group[0],
                                split=group[1],
                                domain=group[2],
                                metric=metric,
                                normalization=normalization,
                                **result,
                            )
                        )
    (args.output_dir / "paired.json").write_text(json.dumps(output, indent=2))
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            dict(
                coverage=coverage,
                sources=source_hashes,
                uncertainty="2000 question-paired bootstrap within domain; Gaussian/frame seeds reported separately",
                windows="First/last32 are positions in the same frozen <=128 token trajectory; windows may overlap",
                normalization=(
                    "actual is equal-question mean per actual token; H128 is equal-question fixed denominator"
                ),
            ),
            indent=2,
        )
    )
    print(json.dumps(dict(complete_jobs=sum(x["complete"] for x in coverage), rows=len(output))))


if __name__ == "__main__":
    main()
