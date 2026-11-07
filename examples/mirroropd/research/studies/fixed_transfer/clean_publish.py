# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Publish a small, reproducible selection from the complete local diagnostic atlas.

No new measurement or model loading. Files are copied verbatim with their audit
receipts and source hashes; complete normalized tables remain in the local atlas.
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump, read, write_csv
from examples.mirroropd.research.core.plotting import load_style
from examples.mirroropd.research.studies.fixed_transfer.clean_diagnostics import BASE, DEFAULT

SELECTED = {
    "qwen-rkl-log": "comparisons/q3-1.7-focal-log",
    "qwen-rkl-absolute": "comparisons/q3-1.7-focal-absolute",
    "qwen-rkl-decrease": "comparisons/q3-1.7-focal-decrease",
    "qwen-rkl-remaining": "comparisons/q3-1.7-focal-remaining",
    "qwen-position-rkl": "comparisons/q3-1.7-focal-position-rkl",
    "qwen-entropy-gap": "comparisons/q3-1.7-focal-entropy-gap",
    "samsung-position": "q3-1.7--sft-gsm8k--a19448d4/position-profile-rkl",
    "samsung-position-updates": "q3-1.7--sft-gsm8k--a19448d4/position-rkl",
    "samsung-entropy": "q3-1.7--sft-gsm8k--a19448d4/entropy",
    "samsung-training-loss": "q3-1.7--sft-gsm8k--a19448d4/training-loss",
    "rl-training-loss": "q3-1.7--grpo-kl--7a02ee6c/training-loss",
    "thinkingspace-position": "q3-1.7--sft-deepmath--20bfd6b7/position-profile-rkl",
    "thinkingspace-position-updates": "q3-1.7--sft-deepmath--20bfd6b7/position-rkl",
    "thinkingspace-position-heatmap": "q3-1.7--sft-deepmath--20bfd6b7/position-absolute-heatmap",
    "r1-top16": "comparisons/r1-focal-top16",
    "r1-position-top16": "comparisons/r1-focal-position-overlap16",
    "justrl-top16-mass": "r1--justrl--534ec3bd/top16-mass",
    "llama-rkl-log": "comparisons/llama3-3b-off-focal-log",
    "llama-position": "comparisons/llama3-3b-off-focal-position-rkl",
}


def validate(atlas, manifests):
    """Independently compare exported endpoints to the frozen overview tables."""
    original = list(csv.DictReader((BASE / "transfer-mechanism/data/overview-20261029/risks.csv").open()))
    endpoints = list(csv.DictReader((atlas / "comparisons/endpoints.csv").open()))
    domains = {"Math": "held_math", "Code": "held_code", "IF": "held_if"}
    checks = dict(domain=0, pooled_heldout=0)
    for row in endpoints:
        panel = row["panel"]
        if row["band"] != "all" or panel not in [*domains, "All held-out"]:
            continue
        chosen = [
            r
            for r in original
            if r["student"] == row["student"]
            and r["teacher"] == row["teacher"]
            and r["metric"] == "rkl"
            and r["panel"] in (list(domains.values()) if panel == "All held-out" else [domains[panel]])
        ]
        if len(chosen) != (3 if panel == "All held-out" else 1):
            raise ValueError(f"Missing independent reference: {row['run_id']}/{panel}")
        for source, exported in [("initial", "fixed_h_initial"), ("final", "fixed_h_final")]:
            value = sum(float(r[source]) * int(r["prompts"]) for r in chosen) / sum(int(r["prompts"]) for r in chosen)
            if not math.isclose(value, float(row[exported]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Independent endpoint mismatch: {row['run_id']}/{panel}/{source}")
        checks["pooled_heldout" if panel == "All held-out" else "domain"] += 1
    fv = load_style()
    audits = list(atlas.glob("*/*.audit.json"))
    for path in audits:
        if any(fv.problem_severity(issue) == "error" for issue in read(path)):
            raise ValueError(f"Figure audit errors: {path}")
    for manifest in manifests:
        target = atlas / manifest["spec"]["run_id"]
        for name, digest in manifest["artifacts"].items():
            if file_hash(target / name) != digest:
                raise ValueError(f"Changed completed artifact: {target / name}")
    return dict(
        endpoint_checks=checks,
        audited_figures=len(audits),
        audit_errors=0,
        teacher_entropy_max_drift=max(m["teacher_entropy_max_drift"] for m in manifests),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=BASE / "figures/fixed-state-clean-20261107")
    args = parser.parse_args()
    if not (args.input / "README.md").exists():
        raise ValueError("Build the complete atlas first")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = [read(p) for p in sorted(args.input.glob("*/manifest.json"))]
    sources = {s["path"]: s for m in manifests for s in m["sources"]}
    receipt = dict(
        local_atlas=str(args.input.resolve()),
        runs=len(manifests),
        run_figures=sum(len(m["figures"]) for m in manifests),
        comparison_figures=len(read(args.input / "comparisons/figures.json")),
        selected={},
        sources=list(sources.values()),
        code={p.name: file_hash(p) for p in Path(__file__).parent.glob("clean_*.py")},
        validation=validate(args.input, manifests),
    )
    for name, source in SELECTED.items():
        files = {}
        for suffix in [".png", ".pdf", ".audit.json"]:
            src, dst = args.input / (source + suffix), args.output / (name + suffix)
            shutil.copyfile(src, dst)
            files[dst.name] = file_hash(dst)
        receipt["selected"][name] = dict(source=source, files=files)
    for source in ["runs.csv", "comparisons/endpoints.csv", "comparisons/all-panel-trajectories.csv"]:
        shutil.copyfile(args.input / source, args.output / Path(source).name)
    write_csv(args.output / "figure-map.csv", [dict(figure=k, atlas_path=v) for k, v in SELECTED.items()])
    dump(args.output / "provenance.json", receipt)
    print(f"Published {len(SELECTED)} selected PNG/PDF pairs to {args.output}")


if __name__ == "__main__":
    main()
