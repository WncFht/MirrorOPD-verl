# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

repo = Path("/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/fht_data/MirrorOPD-verl")
r = repo / "outputs/e39-q3base-8b-full-r16-200step-online-20261026-hppk13"
sys.path.insert(0, str(r / "source"))
from verl.utils.reward_score.one_shot_opd import reward_func  # noqa: E402, I001 - runtime source/style path is configured before import


def main():
    root = repo.parent / "wandb-logs/MirrorOPD-verl/analysis/e39-qwen3base-validation-20261027-step140"
    subset = root / "teacher-subsets"
    questions = {
        name: pq.read_table(subset / f"{name}.parquet").to_pylist() for name in ["MATH-500", "AMC23", "AIME25"]
    }
    teacher = []
    for line in (root / "teacher-outputs.jsonl").read_text().splitlines():
        x = json.loads(line)
        q = questions[x["dataset"]][x["row"]]
        score = reward_func(x["dataset"], x["text"], q["reward_model"]["ground_truth"])
        teacher.append(
            {
                "dataset": x["dataset"],
                "row": x["row"],
                "sample": x["sample"],
                "prompt": q["prompt"][0]["content"],
                "ground_truth": q["reward_model"]["ground_truth"],
                "acc": bool(score["acc"]),
                "length": len(x["response_ids"]),
                "finish_reason": x["finish_reason"],
                "format_score": score.get("format_score"),
                "pred": score.get("pred"),
            }
        )
        print("graded", len(teacher), flush=True) if len(teacher) % 32 == 0 else None

    def aggregate(xs):
        return {
            "responses": len(xs),
            "acc": statistics.mean(x["acc"] for x in xs),
            "length": statistics.mean(x["length"] for x in xs),
            "length_p50": statistics.median(x["length"] for x in xs),
            "cap_fraction": statistics.mean(x["finish_reason"] == "length" for x in xs),
            "short_under32_fraction": statistics.mean(x["length"] < 32 for x in xs),
            "correct_length": statistics.mean(x["length"] for x in xs if x["acc"])
            if any(x["acc"] for x in xs)
            else None,
            "wrong_length": statistics.mean(x["length"] for x in xs if not x["acc"])
            if any(not x["acc"] for x in xs)
            else None,
        }

    summary = {
        "observed_at": time.time(),
        "teacher_source": str(root / "teacher-outputs.jsonl"),
        "teacher": {name: aggregate([x for x in teacher if x["dataset"] == name]) for name in questions},
        "teacher_rows": teacher,
        "student": {},
        "matched_subset_student": {},
        "validation_history": [],
    }
    for stage in [0, 10, 140]:
        groups = collections.defaultdict(list)
        matched = collections.defaultdict(list)
        for line in (r / f"training/validation/records/{stage}.jsonl").open():
            x = json.loads(line)
            a = {
                "dataset": x["data_source"],
                "acc": bool(x["acc"]),
                "length": x["response_length"],
                "finish_reason": x["finish_reason"],
                "prompt_id": x["prompt_id"],
                "prompt": x["input"],
                "pred": x.get("pred"),
                "format_score": x.get("format_score"),
            }
            groups[a["dataset"]].append(a)
            for q in questions[a["dataset"]]:
                if q["prompt"][0]["content"] in a["prompt"]:
                    matched[a["dataset"]].append(a)
                    break
        summary["student"][stage] = {name: aggregate(xs) for name, xs in groups.items()}
        summary["matched_subset_student"][stage] = {name: aggregate(xs) for name, xs in matched.items()}
        print("student stage", stage, summary["student"][stage], flush=True)
    rows = [json.loads(line) for line in (r / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        d = row["data"]
        if "eval/math500/acc/mean" in d and d["step/outer"] <= 140:
            summary["validation_history"].append(d)
    Path("/tmp/e39-validation-analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("TEACHER", json.dumps(summary["teacher"], ensure_ascii=False), flush=True)
    print("MATCHED STUDENT", json.dumps(summary["matched_subset_student"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
