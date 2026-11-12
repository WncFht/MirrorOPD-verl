# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Align existing fixed-state risk curves, update spaces and parameter geometry.

The groups summarize observed panels, not universal learning thresholds. No GPU
or training is used. Fixed-H1024 risk, relative risk and paired window changes
remain separate from selected-matrix parameter distances and task success.
"""

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


def analyze(report, history, output):
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def read(path):
        raw = path.read_bytes()
        provenance[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

    layer = report / "data/layer-review"
    summary = read(layer / "cohort-summary.json")
    full = read(layer / "cohort-dynamics.json.gz")
    readouts = read(layer / "cohort-readouts.json")
    amplitudes = read(layer / "energy-amplitude-analysis.json")["models"]
    lower = {1, 2, 7, 9, 11, 12, 19}
    classification = []
    for row in amplitudes:
        group = "lower residual" if row["job_id"] in lower else "higher residual"
        if (row["remaining"] < 0.1) != (group == "lower residual"):
            raise ValueError("Historical H128 panel group membership changed")
        classification.append(dict(row, group=group, budget_note="short run" if row["steps"] == 64 else "320 steps"))

    groups = {
        100: "lower",
        101: "lower",
        102: "higher, improving",
        103: "lower",
        104: "partial, improving",
        105: "higher, improving",
        106: "higher, improving",
        107: "higher, improving",
        108: "higher, plateau",
        109: "lower",
        110: "higher",
    }
    trajectories = []
    protocols = []
    seen_directories = set()
    for condition in summary["conditions"]:
        job = condition["job_id"]
        if job not in groups:
            continue
        points = []
        raw_risks = {}
        for step, (_, path) in zip(condition["steps"], condition["points"], strict=True):
            checkpoint = Path(path)
            study, arm = checkpoint.parents[1].name, checkpoint.parent.name
            if study == "qwen3-lineage-20261029":
                study = "qwen3-teacher-lineage-20261029"
            directory = history / study / "arms" / arm
            if directory not in seen_directories:
                seen_directories.add(directory)
                protocol = read(directory / "protocol.json")
                training = read(directory / "training.json")
                boundaries = {b + offset for b in [0, 320, 640, 960] for offset in [0, 1, 16]}
                protocols.append(
                    dict(
                        job_id=job,
                        protocol=protocol,
                        boundary_updates=[
                            {k: r[k] for k in ["step", "group", "loss"]} for r in training if r["step"] in boundaries
                        ],
                    )
                )
            for source in directory.glob("fixed-*.json"):
                t = int(re.search(r"\d+", source.stem)[0])
                if t > condition["steps"][-1]:
                    continue
                records = read(source)
                held = [r for r in records if r["split"] == "heldout"]
                # Use the exact standard panel, including its fixed group identities.
                allowed = {"M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"}
                held = [r for r in held if r["group"] in allowed]
                values = {
                    (str(r["id"]), r["group"], r["length"], r["source"]): r["stats"]["all"]["rkl"] / 1024 for r in held
                }
                if len(values) != 128:
                    raise ValueError(f"Expected standard 128-question panel: {condition['teacher']}, {t}")
                if t in raw_risks and values != raw_risks[t]:
                    if max(abs(values[k] - raw_risks[t][k]) for k in values) > 1e-8:
                        raise ValueError("Overlapping historical readouts disagree")
                raw_risks[t] = values
        initial = raw_risks[0]
        for t, values in sorted(raw_risks.items()):
            if set(values) != set(initial):
                raise ValueError("Historical risk state identities changed")
            risk = float(np.mean(list(values.values())))
            points.append(dict(step=t, rkl=risk, remaining=risk / np.mean(list(initial.values()))))
        for old in readouts[str(job)]["rows"]:
            if "rkl" in old:
                current = next(p for p in points if p["step"] == old["step"])
                if abs(current["rkl"] - old["rkl"]) > 1e-10:
                    raise ValueError("Dense curves disagree with prior frozen readout")
        rng = np.random.default_rng(20261111)
        windows = []
        for start, stop in [(0, 64), (64, 320), (320, 640), (640, 1280)]:
            if start not in raw_risks or stop not in raw_risks:
                continue
            keys = sorted(initial)
            a = np.array([raw_risks[start][key] for key in keys])
            b = np.array([raw_risks[stop][key] for key in keys])
            difference = b - a
            boot = difference[rng.integers(len(keys), size=(2000, len(keys)))].mean(axis=1)
            windows.append(
                dict(
                    start=start,
                    stop=stop,
                    initial=float(a.mean()),
                    final=float(b.mean()),
                    absolute_reduction=float(-difference.mean()),
                    relative_reduction=float(-difference.mean() / a.mean()),
                    change_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                )
            )
        matrices = [r for r in full["records"] if r["task"]["job_id"] == job]
        stages = []
        for step, (label, _) in zip(condition["steps"], condition["points"], strict=True):
            rows = defaultdict(list)
            for matrix in matrices:
                module = matrix["task"]["tensor"].split(".")[-2]
                facts = next(r for r in matrix["stages"] if r["label"] == label)
                rows[module].append(facts)
            modules = {}
            for module, facts in rows.items():
                modules[module] = dict(
                    energy=sum(r["energy"] for r in facts),
                    top16=median(sum(s * s for s in r["singular_values"][:16]) / r["energy"] for r in facts),
                    **{
                        f"right_overlap_{rank}": next(
                            r["adjusted_final_overlap"]
                            for r in summary["subspaces"]
                            if r["job_id"] == job
                            and r["label"] == label
                            and r["module"] == module
                            and r["side"] == "right"
                            and r["requested_rank"] == rank
                        )
                        for rank in [16, 128]
                    },
                )
            geo = next(
                (r for r in summary["selected_matrix_teacher_geometry"] if r["job_id"] == job and r["label"] == label),
                None,
            )
            stages.append(dict(step=step, modules=modules, teacher_geometry=geo))
        trajectories.append(
            dict(
                job_id=job,
                label=condition["teacher"],
                group=groups[job],
                reference_step=condition["steps"][-1],
                risks=points,
                windows=windows,
                stages=stages,
            )
        )
    result = dict(
        protocol="Historical H1024 curves; exact selected-matrix delta geometry; no task-accuracy classification",
        h128_classification=classification,
        trajectories=trajectories,
    )
    (output / "grouped-dynamics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "training-protocols.json").write_text(json.dumps(protocols, indent=2, allow_nan=False) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "Grouped conditions",
        len(classification),
        "dense trajectories",
        len(trajectories),
        "risk points",
        sum(len(t["risks"]) for t in trajectories),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.report, args.history, args.output)


if __name__ == "__main__":
    main()
