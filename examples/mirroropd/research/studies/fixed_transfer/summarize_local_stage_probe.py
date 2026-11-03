# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute stage-probe endpoint comparisons without rerunning a model."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = []
    for run in args.runs:
        assert json.loads((run / "complete.json").read_text())["completed"]
        manifest = json.loads((run / "manifest.json").read_text())
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        for stage in ("sft", "rl", "sft_rl"):
            evaluations = [row for row in records if row["phase"] == f"opd_{stage}"]
            initial, final = evaluations[0], evaluations[-1]
            updates = [row for row in records if row["phase"] == f"update_{stage}"]
            assert initial["step"] == 0 and final["step"] == manifest["student_steps"]
            first = updates[0]
            prediction = first["step_size"] * first["grad_norm"] ** 2
            row = dict(
                run=str(run),
                stage=stage,
                steps=final["step"],
                initial_train_kl=initial["train_kl"],
                final_train_kl=final["train_kl"],
                initial_held_kl=initial["held_kl"],
                final_held_kl=final["held_kl"],
                train_reduction=1 - final["train_kl"] / initial["train_kl"],
                held_reduction=1 - final["held_kl"] / initial["held_kl"],
                initial_grad_norm=first["grad_norm"],
                final_grad_norm=updates[-1]["grad_norm"],
                first_step_actual_drop=initial["train_kl"] - first["loss"],
                first_step_linear_prediction=prediction,
                final_held_reward=final["held_reward"],
                final_held_greedy=final["held_greedy"],
            )
            summary.append(row)
            print(
                f"{run.name} {stage}: held KL {row['initial_held_kl']:.9g} -> "
                f"{row['final_held_kl']:.9g}; reduction {row['held_reduction']:.4%}"
            )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
