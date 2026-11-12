# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze existing functional readouts for explicitly identified trajectories.

Historical risks use sum KL / fixed H1024 per question, including zero padding.
New training metrics retain their original names and outer-step units. Neither
teacher prefixes nor teacher rollouts imply forward KL.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


def collect(manifest, history, frozen_metrics, output):
    protocol = json.loads(manifest.read_text())
    result, proof, paired = {}, {}, {}

    def read(path):
        raw = path.read_bytes()
        proof[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def heldout(records):
        groups = {"M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"}
        return [r for r in records if r["group"] in groups]

    def identities(records):
        return {(str(r["id"]), r["group"], r["length"], r["source"]) for r in records}

    for condition in protocol["conditions"]:
        job = str(condition["job_id"])
        label = condition["teacher"]
        if "rollout_policy" in condition:
            arm = label.split()[0]
            result[job] = dict(
                label=label,
                kind="new public bank metrics; step is outer, optimizer step=outer*16",
                rows=read(frozen_metrics / f"{arm}-evaluation-frozen.json"),
            )
            continue
        rows, identity = [], None
        paired[job] = {}
        for step, (_, path) in zip(condition["steps"], condition["points"], strict=True):
            checkpoint = Path(path)
            study, arm = checkpoint.parents[1].name, checkpoint.parent.name
            if study == "qwen3-lineage-20261029":
                study = "qwen3-teacher-lineage-20261029"
            directory = history / study / "arms" / arm
            source = directory / f"fixed-{step:03d}.json"
            if not source.exists():
                rows.append(dict(step=step, missing=str(source)))
                continue
            held, initial = heldout(read(source)), heldout(read(directory / "fixed-000.json"))
            if (directory / "complete.json").exists():
                read(directory / "complete.json")
            current = identities(held)
            if not held or current != identities(initial) or (identity is not None and identity != current):
                raise ValueError(f"Changed heldout identities: {label}, step {step}")
            identity = current
            values = {str(r["id"]): r["stats"]["all"]["rkl"] / 1024 for r in held}
            if len(values) != len(held):
                raise ValueError("Duplicate question id")
            paired[job][str(step)] = values
            risk = mean(values.values())
            base_risk = mean(r["stats"]["all"]["rkl"] / 1024 for r in initial)
            rows.append(
                dict(
                    step=step,
                    n=len(held),
                    initial_rkl=base_risk,
                    rkl=risk,
                    remaining=risk / base_risk,
                    by_domain={
                        domain: mean(r["stats"]["all"]["rkl"] / 1024 for r in held if r["domain"] == domain)
                        for domain in sorted({r["domain"] for r in held})
                    },
                )
            )
        result[job] = dict(label=label, kind="historical fixed-H1024 heldout, per-question mean", rows=rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cohort-readouts.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "cohort-readout-provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
    (output / "cohort-question-risks.json.gz").write_bytes(gzip.compress(json.dumps(paired).encode(), mtime=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--frozen-metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    collect(args.manifest, args.history, args.frozen_metrics, args.output)


if __name__ == "__main__":
    main()
