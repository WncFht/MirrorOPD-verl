# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute first-pass H1024 risks and join frozen spectral measurements.

No GPU is used. Each run has exactly one risk source; steps after the first
20 batches are rejected. Endpoint geometry and stage frames remain distinct.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from .absorption.cohort import report_eligible


def analyze(report, clean, output):
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def read(path, kind="json"):
        raw = path.read_bytes()
        provenance[str(path)] = hashlib.sha256(raw).hexdigest()
        if kind == "csv":
            return list(csv.DictReader(raw.decode().splitlines()))
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

    layer = report / "data/layer-review"
    models = read(layer / "energy-amplitude-analysis.json")["models"]
    jobs = {j["job_id"]: j for j in read(report / "data/teacher-spectrum-20261110/expansion-v1/manifest.json")["jobs"]}
    runs = {r["run_id"]: r for r in read(clean / "runs.csv", "csv")}
    endpoints = {
        r["run_id"]: r
        for r in read(clean / "endpoints.csv", "csv")
        if r["panel"] == "All held-out" and r["band"] == "all"
    }
    question_rows, risks, identities = [], [], []
    rng = np.random.default_rng(20261112)
    for m in models:
        j = jobs[m["job_id"]]
        assert report_eligible(j)
        m["historical_h1024"] = None
        if j["run_id"] not in runs:
            continue
        item = runs[j["run_id"]]
        budget = int(item["budget"])
        if budget > 320 or budget != j["step"]:
            raise ValueError("Endpoint budget does not match the first pass")
        directory = Path(item["source"]).parent
        snapshots = {}
        initial_keys = None
        for source in sorted(directory.glob("fixed-*.json")):
            step = int(re.search(r"\d+", source.stem)[0])
            if step > budget:
                continue
            held = [r for r in read(source) if r["split"] == "heldout"]
            keyed = {(str(r["id"]), r["source"], r["group"], r["length"]): r for r in held}
            if len(keyed) != len(held) or len(held) != 128:
                raise ValueError("Duplicate questions or nonstandard H1024 panel")
            if initial_keys is None:
                initial_keys = set(keyed)
            if set(keyed) != initial_keys:
                raise ValueError("Frozen question/state identities changed")
            order = sorted(keyed)
            values = np.array([keyed[k]["stats"]["all"]["rkl"] / 1024 for k in order])
            snapshots[step] = values
            for key in order:
                r = keyed[key]
                question_rows.append(
                    dict(
                        job_id=m["job_id"],
                        step=step,
                        id=key[0],
                        source=key[1],
                        domain=r["domain"],
                        tokens=r["stats"]["all"]["tokens"],
                        rkl=r["stats"]["all"]["rkl"] / 1024,
                    )
                )
        baseline = snapshots[0]
        final = snapshots[budget]
        old = endpoints[j["run_id"]]
        assert abs(baseline.mean() - float(old["fixed_h_initial"])) < 1e-10
        assert abs(final.mean() - float(old["fixed_h_final"])) < 1e-10
        ratios = final[rng.integers(128, size=(2000, 128))].mean(axis=1)
        indices = rng.integers(128, size=(2000, 128))
        remaining_ci = np.quantile(
            final[indices].mean(axis=1) / baseline[indices].mean(axis=1), [0.025, 0.975]
        ).tolist()
        row = dict(
            initial=float(baseline.mean()),
            final=float(final.mean()),
            remaining=float(final.mean() / baseline.mean()),
            n=128,
            final_ci95=np.quantile(ratios, [0.025, 0.975]).tolist(),
            remaining_ci95=remaining_ci,
            step=budget,
            run_id=j["run_id"],
        )
        m["historical_h1024"] = row
        for step, values in sorted(snapshots.items()):
            risks.append(
                dict(
                    job_id=m["job_id"],
                    step=step,
                    rkl=float(values.mean()),
                    remaining=float(values.mean() / baseline.mean()),
                )
            )
        # The source directory may contain copied earlier snapshots. Select it once,
        # never append the same prefix from another continuation directory.
        protocol_file = directory / "protocol.json"
        protocol = read(protocol_file) if protocol_file.exists() else {}
        receipt_file = Path(j["endpoint"]) / "complete.json"
        receipt = read(receipt_file) if receipt_file.exists() else {}
        identities.append(
            dict(
                job_id=m["job_id"],
                run_id=j["run_id"],
                source_directory=str(directory),
                steps=sorted(snapshots),
                protocol=protocol,
                checkpoint_receipt=receipt,
            )
        )
    angles = read(layer / "module-angle-analysis.json.gz")["angle_summary"]
    frame_summary = read(layer / "expanded-update-summary.json")
    cohort = read(layer / "cohort-summary.json")
    frame_summary = [
        dict(r, step=int(r["label"][4:]), reference_step=64 if r["job_id"] == 6 else 320)
        for r in frame_summary
        if r["label"] != "endpoint"
    ]
    for row in cohort["subspaces"]:
        if row["job_id"] not in [109, 110] or row["label"] == "endpoint":
            continue
        frame_summary.append(
            dict(
                job_id=row["job_id"],
                module=row["module"],
                side=row["side"],
                rank=row["requested_rank"],
                label=row["label"],
                step=64,
                reference_step=320,
                raw=row["final_overlap"],
                adjusted=row["adjusted_final_overlap"],
                layers=row["matrix_count"],
            )
        )
    expanded = read(layer / "expanded-update-dynamics.json.gz")
    additional = read(layer / "cohort-dynamics.json.gz")
    records = expanded["records"] + [r for r in additional["records"] if r["task"]["job_id"] in [109, 110]]
    grouped = defaultdict(list)
    for r in records:
        job, module = r["task"]["job_id"], r["task"]["tensor"].split(".")[-2]
        point_steps = [
            int(label[4:]) if label != "endpoint" else (64 if job == 6 else 320) for label, _ in r["task"]["points"]
        ]
        assert max(point_steps) <= 320
        for stage, step in zip(r["stages"], point_steps, strict=True):
            grouped[(job, module, step)].append(stage)
    spectra = []
    for (job, module, step), rows in sorted(grouped.items()):
        spectra.append(
            dict(
                job_id=job,
                module=module,
                step=step,
                matrices=len(rows),
                energy=sum(r["energy"] for r in rows),
                top16=median(sum(s * s for s in r["singular_values"][:16]) / r["energy"] for r in rows),
                top128=median(sum(s * s for s in r["singular_values"][:128]) / r["energy"] for r in rows),
            )
        )
    # These selected frames refer only to the first-pass endpoint, never step1280.
    assert {r["reference_step"] for r in frame_summary} <= {64, 320}
    assert len({r["job_id"] for r in frame_summary}) == 6
    result = dict(
        protocol="First-pass steps<=320; H1024 heldout128 for 14 endpoint conditions; MiMo risk protocol separate",
        models=models,
        risks=risks,
        frames=frame_summary,
        spectra=spectra,
        full_weight_angles=angles,
    )
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "question-risks.json.gz").write_bytes(gzip.compress(json.dumps(question_rows).encode(), mtime=0))
    (output / "identities.json").write_text(json.dumps(identities, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        dict(
            models=len(models),
            historical_h1024=sum(m["historical_h1024"] is not None for m in models),
            risk_points=len(risks),
            frame_conditions=6,
            max_step=max(r["step"] for r in risks),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.report, args.clean, args.output)


if __name__ == "__main__":
    main()
