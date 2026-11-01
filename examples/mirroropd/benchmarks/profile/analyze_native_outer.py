# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Measured native outer costs, logical token exposure and explicitly labeled estimates."""

import argparse
import json
import statistics
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from examples.mirroropd.research.core.io import dump, write_csv
from verl.trainer.evaluation.artifacts import file_sha256


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line)["data"] for line in (run / "metrics.jsonl").read_text().splitlines()]
    outers = [row for row in records if "perf/timing/outer_step_seconds" in row]
    events = {}
    for path in (run / "evaluation/events").glob("outer-*/manifest.json"):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            raise ValueError("Profiling requires completed evaluation events")
        events[int(path.parent.name.split("-")[-1])] = record["metadata"]
    config = OmegaConf.load(run / "acceptance.yaml")
    reuse = int(config.actor_rollout_ref.actor.ppo_epochs)
    teacher_path = Path(config.distillation.teacher_models.teacher_model.model_path)
    hidden = json.loads((teacher_path / "config.json").read_text())["hidden_size"]
    from verl.trainer.evaluation.execution import hidden_dtype_bytes

    width = hidden_dtype_bytes(config)
    token_rows = []
    for path in sorted((run / "evaluation/trajectories").glob("*.parquet")):
        rows = pq.read_table(path, columns=["prompt_tokens", "response_tokens", "valid_tokens"]).to_pylist()
        totals = {key: sum(row[key] for row in rows) for key in ("prompt_tokens", "response_tokens", "valid_tokens")}
        token_rows.append(
            dict(
                outer=int(path.stem.split("-")[-1]),
                questions=len(rows),
                **totals,
                student_input_token_visits=reuse * (totals["prompt_tokens"] + totals["response_tokens"]),
                student_loss_token_visits=reuse * totals["valid_tokens"],
                logical_teacher_hidden_bytes=totals["valid_tokens"] * hidden * width,
            )
        )
    timings = sorted({key for row in outers for key in row if key.startswith("perf/timing/")})
    classes = {"ordinary": [], "fixed_bank": [], "bench": []}
    for row in outers:
        outer = int(row["step/outer"])
        tasks = events[outer]["plan"]["tasks"]
        group = (
            "bench"
            if events[outer]["plan"]["generate_current"]
            else "fixed_bank"
            if "public_initial" in tasks
            else "ordinary"
        )
        classes[group].append(row)
    stage_rows = []
    for group, rows in classes.items():
        for metric in timings:
            values = [row.get(metric, 0) for row in rows]
            stage_rows.append(
                dict(
                    group=group,
                    outers=len(rows),
                    metric=metric,
                    mean_seconds=statistics.mean(values),
                    median_seconds=statistics.median(values),
                    minimum_seconds=min(values),
                    maximum_seconds=max(values),
                )
            )
    eval_rows = []
    for outer, event in sorted(events.items()):
        metrics = event["metrics"]
        current = {k: v for k, v in metrics.items() if k.startswith("eval/accuracy/") and k.count("/") == 3}
        generated_tokens = sum(
            v * current[k.replace("response_tokens_mean", "samples")]
            for k, v in current.items()
            if k.endswith("response_tokens_mean")
        )
        new_banks = ["public_initial/all", "public_teacher/all"] + (["public_current/all"] if outer else [])
        eval_rows.append(
            dict(
                outer=outer,
                measured_rows=metrics["eval/rkl/rows"],
                source_history_off_rows=sum(metrics.get(f"eval/rkl/{bank}/rows", 0) for bank in new_banks),
                source_history_off_response_positions=sum(
                    metrics.get(f"eval/rkl/{bank}/tokens", 0) for bank in new_banks
                ),
                rkl_seconds=event["timing"]["seconds"],
                evaluation_seconds=metrics["perf/timing/evaluation_seconds"],
                current_generated_response_tokens=round(generated_tokens) if event["plan"]["generate_current"] else 0,
                rkl_windows=event["timing"]["windows"],
            )
        )
    ordinary = classes["ordinary"]
    estimate = statistics.mean(
        row["perf/timing/outer_step_seconds"] - row.get("perf/timing/evaluation_seconds", 0) for row in ordinary
    )
    phases = {
        name: json.loads((run / (name + "-exit.json")).read_text())
        for name in ("initial", "teacher", "finalize", "train", "figures")
    }
    summary = dict(
        run=str(run),
        gpus=json.loads((run / "gpus.json").read_text()),
        reuse=reuse,
        measurement_scope=(
            "Frozen completed run with source/history enabled; disabled timing is an estimate, not a new measurement"
        ),
        timing_notes=[
            "rollout_batch is nested within rollout_pipeline; do not add them",
            "update_actor includes all reuse updates, teacher-head projection, KL, backward, "
            "optimizer and communication",
            "evaluation includes overlapping generation/grading and scoring; "
            "CPU grading is not separately identifiable",
            "token visits count logical passes, not activation-checkpoint recomputation or measured bus traffic",
        ],
        phases=phases,
        stage_means=stage_rows,
        evaluation=eval_rows,
        mean_training_tokens={k: statistics.mean(row[k] for row in token_rows) for k in token_rows[0] if k != "outer"},
        total_training_tokens={k: sum(row[k] for row in token_rows) for k in token_rows[0] if k != "outer"},
        estimated_ordinary_outer_without_source_history_seconds=estimate,
        source_history_only_evaluation_seconds=sum(row.get("perf/timing/evaluation_seconds", 0) for row in ordinary),
        original_rkl_trajectory_evaluations=sum(row["measured_rows"] for row in eval_rows),
        source_history_off_rkl_trajectory_evaluations=sum(row["source_history_off_rows"] for row in eval_rows),
        source_hashes={
            str(p.relative_to(run)): file_sha256(p) for p in [run / "metrics.jsonl", run / "acceptance.yaml"]
        },
        generator_sha256=file_sha256(__file__),
    )
    dump(output / "summary.json", summary)
    write_csv(output / "stages.csv", stage_rows)
    write_csv(output / "training-tokens.csv", token_rows)
    write_csv(output / "evaluation-nodes.csv", eval_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
