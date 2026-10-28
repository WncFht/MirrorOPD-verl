# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize live E54 raw trajectories and paired, pre-update stop diagnostics."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    # The writer may currently be appending the last line.
    return [json.loads(line) for line in lines if line.endswith("\n")]


def summary(root):
    result = {"rollouts": [], "updates": [], "paired_terminal_states": []}
    answers = {}
    for path in sorted(root.glob("rollout-[0-9]*.jsonl")):
        rows = read(path)
        if not rows:
            continue
        answers.update({r["sample_key"]: r for r in rows})
        total_tokens = sum(r["length"] for r in rows)
        result["rollouts"].append(
            {
                "outer": rows[0]["outer"],
                "n": len(rows),
                "mean_length": statistics.mean(r["length"] for r in rows),
                "median_length": statistics.median(r["length"] for r in rows),
                "cap_count": sum(r["at_response_cap"] for r in rows),
                "eos_count": sum(r["stopped_on_eos"] for r in rows),
                "last_tokens": dict(Counter(str(r["last_token"]) for r in rows)),
                "boxed_count": sum("\\boxed" in r["text"] for r in rows),
                "mean_score": statistics.mean(r["score"] for r in rows),
                "cap_token_fraction": sum(r["length"] for r in rows if r["at_response_cap"]) / total_tokens,
                "eos_targets_per_1000_tokens": 1000 * sum(r["stopped_on_eos"] for r in rows) / total_tokens,
            }
        )
    updates, paired = defaultdict(list), defaultdict(dict)
    for path in sorted(root.glob("states-rank-*.jsonl")):
        for row in read(path):
            updates[row["update_attempt"]].append(row)
            if row["terminal_target"]:
                key = (row["outer"], row["sample_key"], row["position"])
                paired[key][row["inner"]] = row
    for update, rows in sorted(updates.items()):
        terminals = [r for r in rows if r["terminal_target"]]
        ends = [r for r in rows if r["position"] == r["response_length"] - 1]
        info = {"update": update, "outer": rows[0]["outer"], "inner": rows[0]["inner"], "positions": len(rows)}
        for label, group in [("sampled_eos", terminals), ("last_target", ends)]:
            info[label] = {
                "n": len(group),
                **(
                    {
                        "student_stop_mean": statistics.mean(r["student_stop"] for r in group),
                        "teacher_stop_mean": statistics.mean(r["teacher_stop"] for r in group),
                        "positive_stop_gradient_fraction": statistics.mean(r["stop_logit_gradient"] > 0 for r in group),
                    }
                    if group
                    else {}
                ),
            }
        result["updates"].append(info)
    deltas = defaultdict(list)
    for (outer, _key, _pos), records in paired.items():
        last_inner = records.get(1, {}).get("reuse", 16)
        if 1 in records and last_inner in records:
            first, last = records[1], records[last_inner]
            deltas[outer].append(
                {
                    "first": first["student_stop"],
                    "last": last["student_stop"],
                    "teacher": first["teacher_stop"],
                    "teacher_drift": abs(first["teacher_stop"] - last["teacher_stop"]),
                    "last_inner": last_inner,
                }
            )
    for outer, rows in sorted(deltas.items()):
        result["paired_terminal_states"].append(
            {
                "outer": outer,
                "pairs": len(rows),
                "pre_inner1_student_stop": statistics.mean(r["first"] for r in rows),
                f"pre_inner{rows[0]['last_inner']}_student_stop": statistics.mean(r["last"] for r in rows),
                "last_inner": rows[0]["last_inner"],
                "teacher_stop": statistics.mean(r["teacher"] for r in rows),
                "max_teacher_drift": max(r["teacher_drift"] for r in rows),
                "fraction_decreased": statistics.mean(r["last"] < r["first"] for r in rows),
            }
        )
    metrics = [r.get("data", r) for r in read(root.parent / "metrics.jsonl")]
    result["loss_by_outer"] = []
    for outer in sorted({r["step/outer"] for r in metrics if "optim/actor/loss" in r}):
        group = [r for r in metrics if r.get("step/outer") == outer and "optim/actor/loss" in r]
        result["loss_by_outer"].append(
            {
                "outer": outer,
                "updates": len(group),
                "first_loss": group[0]["optim/actor/loss"],
                "last_loss": group[-1]["optim/actor/loss"],
            }
        )
    fixed = read(root.parent / "probes/fixed-states.jsonl")
    grouped = defaultdict(list)
    for row in fixed:
        answer = answers[row["sample_key"]]
        if row["position"] == row["response_length"] - 1:
            group_key = (row["outer_source"], row["source_ended_on_eos"], answer["score"] > 0, row["checkpoint"])
            grouped[group_key].append(row["student_stop"])
    result["fixed_prefix_by_score"] = [
        {
            "source_outer": key[0],
            "source_ended_on_eos": key[1],
            "source_correct": key[2],
            "checkpoint": key[3],
            "n": len(values),
            "mean_stop": statistics.mean(values),
            "median_stop": statistics.median(values),
        }
        for key, values in sorted(grouped.items())
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run debug directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summary(args.root), indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    else:
        print(report)


if __name__ == "__main__":
    main()
