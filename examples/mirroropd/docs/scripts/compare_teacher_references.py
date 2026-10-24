#!/usr/bin/env python3
"""Attach historical val-only references without relabeling them as run step zero."""

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_policy_run import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    docs = Path(__file__).resolve().parents[1]
    mapping = {"math500": "MATH-500", "amc23": "AMC23", "aime25": "AIME25"}
    refs = []
    for kind, run in (("student", "6uorrk2k"), ("teacher", "u87buult")):
        path = docs / "data" / f"oneshot-{kind}-{run}-api-20261020"
        rows = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        config = json.loads((path / "config.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        refs.append(
            {
                "kind": kind,
                "run_id": run,
                "source": json.loads((path / "source.json").read_text()),
                "source_path": str(path.relative_to(docs)),
                "model": config["actor_rollout_ref"]["model"]["path"],
                "sampling": config["actor_rollout_ref"]["rollout"]["val_kwargs"],
                "mean_accuracy": {slug: row[f"val-core/{name}/acc/mean@16"] for slug, name in mapping.items()},
                "format_score": {slug: row[f"val-aux/{name}/format_score/mean@16"] for slug, name in mapping.items()},
                "length_or_entropy_keys": [
                    k
                    for k in row.keys() | summary.keys()
                    if any(word in k.lower() for word in ("length", "entropy", "tokens", "len/"))
                ],
                "boundary": "Historical One-Shot-OPD val-only; same sampling budget, different execution/input path. "
                "Only mean accuracy is overlaid; legacy bootstrap best/majority are not modern pass/majority.",
            }
        )
    local_path = docs / "data/valid-parameters-20261022/analysis.json"
    local_key = "valid-full-s128-u0.65-t8192-a"
    local = json.loads(local_path.read_text())["runs"][local_key]
    local_student = {
        "run_id": local["run_id"],
        "source_path": str(local_path.relative_to(docs)),
        "source_key": local_key,
        "mean_accuracy": {slug: local["quality_metrics"][f"eval/{slug}/acc_mean"] for slug in mapping},
        "overall_mean_response_tokens": local["tokens"] / local["responses"],
        "responses": local["responses"],
        "output_tokens": local["tokens"],
        "boundary": "Initial checkpoint, four H800, graph rollout; length from engine generation-token counter. "
        "No per-dataset token lengths or entropy in archived history.",
    }
    analysis = json.loads((args.comparison_data / "analysis.json").read_text())
    convergence = {}
    terminal = []
    teacher = refs[1]["mean_accuracy"]
    for label, run in analysis["runs"].items():
        windows = []
        for lo, hi in ((1, 20), (21, 40), (81, 100), (141, 160), (159, 178), (181, 200)):
            selected = [b for b in run["reuse"] if lo <= b["outer"] <= hi]
            if not selected:
                continue
            windows.append(
                {
                    "outer_from": lo,
                    "outer_through": hi,
                    "batches": len(selected),
                    "first_kl_mean": st.mean(b["first"] for b in selected),
                    "last_preupdate_kl_mean": st.mean(b["last"] for b in selected),
                    "relative_drop_mean": st.mean(b["relative_drop"] for b in selected),
                    "last_pair_relative_drop_mean": (
                        st.mean(1 - b["values"][-1] / b["values"][-2] for b in selected)
                        if len(selected[0]["values"]) > 1
                        else None
                    ),
                }
            )
        convergence[label] = windows
        last = run["evaluation"][-1]
        values = {slug: last["metrics"][f"eval/{slug}/acc/mean"] for slug in mapping}
        responses = sum(last["metrics"][f"eval/{slug}/response_count"] for slug in mapping)
        tokens = sum(
            last["metrics"][f"eval/{slug}/response_count"]
            * last["metrics"][f"eval/{slug}/generation/response_length/mean"]
            for slug in mapping
        )
        terminal.append(
            {
                "label": label,
                "run_id": run["id"],
                "outer": last["outer"],
                "mean_accuracy": values,
                "teacher_minus_run_pp": {s: 100 * (teacher[s] - values[s]) for s in mapping},
                "overall_mean_response_tokens": tokens / responses,
                "per_dataset_mean_response_tokens": {
                    s: last["metrics"][f"eval/{s}/generation/response_length/mean"] for s in mapping
                },
            }
        )
    result = {
        "historical": refs,
        "local_initial_student": local_student,
        "terminal_comparison": terminal,
        "within_batch_convergence": convergence,
        "entropy_coverage": {
            "historical_student_eval": "not recorded",
            "historical_teacher_eval": "not recorded",
            "full_kl_runs": "not recorded",
            "E15": "training-prefix pre_update entropy; not evaluation entropy",
        },
        "scope": "Historical references and existing telemetry; no new model evaluation, IS or ESS estimates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
