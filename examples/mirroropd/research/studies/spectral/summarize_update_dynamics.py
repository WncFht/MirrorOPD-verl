# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate and freeze checkpoint spectra and module-level trajectory summaries.

Each matrix is an observation, not an independent model replicate. Module
summaries use medians; global teacher geometry sums energies and inner products.
Partial training references remain explicitly labeled in the source protocol.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarize(source, output):
    status = json.loads((source / "exit.json").read_text())
    if status["failed"] or status["complete"] != status["total"]:
        raise ValueError(f"Incomplete trajectory measurement: {status}")
    protocol = json.loads((source / "protocol.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(source.glob("matrix-*.json"))]
    if len(records) != status["total"]:
        raise ValueError("Matrix count disagrees with completion receipt")
    keys = [(r["task"]["job_id"], r["task"]["tensor"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate matrix condition")
    expected = {(t["job_id"], t["tensor"]) for t in protocol["tasks"]}
    if set(keys) != expected:
        raise ValueError("Measured matrices disagree with the frozen task manifest")
    groups, stages, geometry = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        task = record["task"]
        module = task["tensor"].split(".")[-2]
        layer = int(re.search(r"layers\.(\d+)", task["tensor"])[1])
        for facts in record["stages"]:
            singular = facts["singular_values"]
            energy = facts["energy"]
            if not math.isclose(sum(x * x for x in singular), energy, rel_tol=1e-8):
                raise ValueError("Update spectrum energy identity failed")
            stages[(task["job_id"], facts["label"], module)].append(facts)
            geometry[(task["job_id"], facts["label"])].append(facts)
        for row in record["rows"]:
            if not -1e-8 <= row["final_overlap"] <= 1 + 1e-8:
                raise ValueError("Invalid subspace overlap")
            if row["label"] == "endpoint" and abs(row["final_overlap"] - 1) > 1e-8:
                raise ValueError("Final self-overlap failed")
            if row["early_capture"] > row["own_energy"] + 1e-8:
                raise ValueError("Projected energy exceeds optimal rank-k approximation")
            groups[(task["job_id"], row["label"], module, row["requested_rank"], row["side"])].append(
                dict(row, layer=layer)
            )
    summaries = []
    for (job, label, module, rank, side), rows in sorted(groups.items()):
        summaries.append(
            dict(
                job_id=job,
                label=label,
                module=module,
                requested_rank=rank,
                side=side,
                matrix_count=len(rows),
                layers=sorted(r["layer"] for r in rows),
                **{
                    field: median(r[field] for r in rows if r[field] is not None)
                    if any(r[field] is not None for r in rows)
                    else None
                    for field in (
                        "final_overlap",
                        "adjusted_final_overlap",
                        "early_capture",
                        "own_energy",
                        "effective_rank",
                    )
                },
            )
        )
    spectral = []
    for (job, label, module), rows in sorted(stages.items()):
        per_matrix = []
        for r in rows:
            energy = r["energy"]
            probabilities = [s * s / energy for s in r["singular_values"] if s > 0]
            item = dict(
                energy=energy,
                stable_rank=energy / r["singular_values"][0] ** 2,
                entropy_rank=math.exp(-sum(p * math.log(p) for p in probabilities)),
            )
            for rank in (16, 64, 128, 256, 512):
                item[f"top{rank}"] = sum(s * s for s in r["singular_values"][:rank]) / energy
            if "weight_geometry" in r:
                item.update({k: r["weight_geometry"][k] for k in ("delta_sigma", "rho_sigma", "kappa_spec")})
            per_matrix.append(item)
        spectral.append(
            dict(
                job_id=job,
                label=label,
                module=module,
                matrix_count=len(rows),
                energy_sum=sum(r["energy"] for r in rows),
                medians={k: median(r[k] for r in per_matrix) for k in per_matrix[0]},
            )
        )
    global_geometry = []
    for (job, label), rows in sorted(geometry.items()):
        if not all("teacher_comparison" in r for r in rows):
            continue
        es = sum(r["energy"] for r in rows)
        et = sum(r["teacher_comparison"]["teacher_energy"] for r in rows)
        dot = sum(
            r["teacher_comparison"]["cosine"] * math.sqrt(r["energy"] * r["teacher_comparison"]["teacher_energy"])
            for r in rows
        )
        residual = (
            sum(
                r["teacher_comparison"]["residual_energy_ratio"] * r["teacher_comparison"]["teacher_energy"]
                for r in rows
            )
            / et
        )
        ratio = math.sqrt(es / et)
        cosine = dot / math.sqrt(es * et)
        if not math.isclose(residual, 1 + ratio * ratio - 2 * ratio * cosine, rel_tol=1e-8):
            raise ValueError("Teacher residual identity failed")
        global_geometry.append(
            dict(job_id=job, label=label, matrix_count=len(rows), norm_ratio=ratio, cosine=cosine, residual=residual)
        )
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(protocol=protocol, records=records), separators=(",", ":"), allow_nan=False).encode()
    (output / "cohort-dynamics.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    result = dict(
        conditions=protocol["conditions"],
        matrix_conditions=len(records),
        checkpoint_matrices=sum(len(r["stages"]) for r in records),
        rank_side_records=sum(len(r["rows"]) for r in records),
        subspaces=summaries,
        spectra=spectral,
        selected_matrix_teacher_geometry=global_geometry,
    )
    (output / "cohort-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "cohort-validation.json").write_text(
        json.dumps(
            dict(
                complete=status,
                update_energy_identity=True,
                self_overlap_identity=True,
                optimal_capture_bound=True,
                teacher_residual_identity=True,
                source_sha256=hashlib.sha256(raw).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.source, args.output)
    print({k: result[k] for k in ("matrix_conditions", "checkpoint_matrices", "rank_side_records")})


if __name__ == "__main__":
    main()
