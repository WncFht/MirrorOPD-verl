"""Pairwise teacher geometry versus held-out trajectories within input mode.

Teacher-pair distances are dependent; permutations relabel whole teachers,
never individual distance entries. Associations remain exploratory. All
curves share the observed 0:320 window; levels and relative shapes separated.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "curve-spectrum-v1"


def main():
    df = pd.read_csv(OUT / "curve-feature-table.csv")
    curves = json.loads((OUT / "curve-clusters.json").read_text())
    lookup = {r["run_id"]: np.array(v) for r, v in zip(curves["runs"], curves["log10_absolute_risk"], strict=True)}
    neighbors = json.loads((OUT / "teacher-neighbors.json").read_text())["comparisons"]
    pairs = []
    results = []
    for group in ["r1", "q3-1.7", "q3-1.7-nonthinking"]:
        rows = df[(df.student == group) & df.compatible].to_dict("records")
        n = len(rows)
        fields = ["normalized_spectrum_distance", "norm_logratio", "delta_cosine", "relative_delta_distance"]
        G = {k: np.zeros((n, n)) for k in fields}
        C = {k: np.zeros((n, n)) for k in ["absolute_curve_distance", "relative_shape_distance"]}
        for i, j in itertools.combinations(range(n), 2):
            a, b = rows[i], rows[j]
            r = next(r for r in neighbors if {r["path_a"], r["path_b"]} == {a["teacher_path"], b["teacher_path"]})
            va, vb = lookup[a["run_id"]], lookup[b["run_id"]]
            entry = dict(student=group, teacher_a=a["teacher"], teacher_b=b["teacher"])
            for field in fields:
                value = (
                    abs(r[field]) if field == "norm_logratio" else 1 - r[field] if field == "delta_cosine" else r[field]
                )
                G[field][i, j] = G[field][j, i] = value
                entry[field] = value
            for field, value in [
                ("absolute_curve_distance", np.sqrt(np.mean((va - vb) ** 2))),
                ("relative_shape_distance", np.sqrt(np.mean(((va - va[0]) - (vb - vb[0])) ** 2))),
            ]:
                C[field][i, j] = C[field][j, i] = value
                entry[field] = float(value)
            pairs.append(entry)
        tri = np.triu_indices(n, 1)
        for field, matrix in G.items():
            for target, dist in C.items():
                observed = float(spearmanr(matrix[tri], dist[tri]).statistic)
                perms = []
                for perm in itertools.permutations(range(n)):
                    perm = np.array(perm)
                    perms.append(float(spearmanr(matrix[tri], dist[perm[:, None], perm[None, :]][tri]).statistic))
                results.append(
                    dict(
                        student=group,
                        teachers=n,
                        teacher_pairs=len(tri[0]),
                        geometry=field,
                        target=target,
                        spearman=observed,
                        whole_teacher_permutations=len(perms),
                        permutation95=np.quantile(perms, [0.025, 0.975]).tolist(),
                        permutation_tail_fraction=float((np.abs(perms) >= abs(observed) - 1e-12).mean()),
                    )
                )
    pd.DataFrame(pairs).to_csv(OUT / "pairwise-geometry-curve.csv", index=False)
    (OUT / "neighbor-associations.json").write_text(
        json.dumps(
            dict(
                results=results,
                protocol="Exact whole-teacher label permutations; exploratory, no multiple-test correction",
            ),
            indent=2,
        )
    )
    for r in results:
        if r["geometry"] in ["normalized_spectrum_distance", "delta_cosine"]:
            print(r, flush=True)


if __name__ == "__main__":
    main()
