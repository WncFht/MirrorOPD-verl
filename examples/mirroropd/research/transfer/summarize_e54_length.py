# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute text diagnostics and grade frozen teacher generations on CPU.

The module entry point is required by the math verifier's spawn process pool.
Text re-encoding is explicit; its tokens are not original rollout token IDs.
"""

import argparse
import csv
import json
import re
import runpy
import statistics as st
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def text_metrics(args):
    if args.recompute_text:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.models / "Qwen3-1.7B-Base", local_files_only=True)
        script = Path(__file__).resolve().parents[2] / "research/analysis/termination/build_data.py"
        metric = runpy.run_path(str(script))["repeat_metrics"]
        output = []
        for tag in ["E41", "E54"]:
            for step in range(1, 30):
                rows = read_rows(args.data / tag / "rollouts" / f"{step}.jsonl")
                ids = tok([r["output"] for r in rows], add_special_tokens=False)["input_ids"]
                values = [metric(tokens) for tokens in ids]
                output.append(
                    dict(
                        experiment=tag,
                        outer=step,
                        responses=len(rows),
                        mean_reencoded_length=st.mean(r["length"] for r in values),
                        long_loop_fraction=st.mean(r["long_loop"] for r in values),
                        repeat8_mean=st.mean(r["repeat8"] for r in values),
                        score=st.mean(r["score"] for r in rows),
                    )
                )
        with (args.data / "text-repetition.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    summary = {}
    pattern = r"\b(?:wait|however|but|alternatively|reconsider|double.check)\b|等等|但是|不过|重新|检查"
    for tag in ["E41", "E54"]:
        rows = []
        for step in range(25, 30):
            for row in read_rows(args.data / tag / "rollouts" / f"{step}.jsonl"):
                text = row["output"]
                rows.append(
                    dict(
                        markers=len(re.findall(pattern, text, re.I)),
                        boxed=text.count("\\boxed"),
                        score=row["score"],
                        input=row["input"],
                        output=text,
                        step=step,
                    )
                )
        with (args.data / "text-repetition.csv").open() as stream:
            metrics = [r for r in csv.DictReader(stream) if r["experiment"] == tag and 25 <= int(r["outer"]) <= 29]
        count = sum(float(r["mean_reencoded_length"]) * int(r["responses"]) for r in metrics)
        summary[tag] = dict(
            responses=len(rows),
            mean_markers=st.mean(r["markers"] for r in rows),
            markers_per_1k_reencoded_tokens=sum(r["markers"] for r in rows) / count * 1000,
            no_boxed_fraction=st.mean(r["boxed"] == 0 for r in rows),
            long_loop_fraction=st.mean(float(r["long_loop_fraction"]) for r in metrics),
            repeat8_mean=st.mean(float(r["repeat8_mean"]) for r in metrics),
            mean_reencoded_length=count / len(rows),
        )
        if tag == "E54":
            selected = sorted(rows, key=lambda r: r["markers"], reverse=True)
            dump(args.data / "sample-examples.json", [selected[0], selected[len(selected) // 2]])
    dump(args.data / "text-summary.json", summary)


def grade_questions(args):
    from verl.utils.reward_score.one_shot_opd.math import reward_func

    summary, scores = {}, []
    for name in ["teacher_4b", "teacher_30b"]:
        rows = read_rows(args.data / "probes" / f"{name}-continuations.jsonl")
        assert len(rows) == 55
        summary[name] = {}
        for variant in ["training_prompt", "without_empty_think"]:
            selected = [r for r in rows if r["kind"] == "question" and r["variant"] == variant]
            values = []
            for row in selected:
                result = reward_func("math", row["output"], row["ground_truth"])
                values.append(result["score"])
                scores.append(
                    dict(
                        model=name,
                        variant=variant,
                        id=row["id"],
                        length=row["response_length"],
                        finish_reason=row["finish_reason"],
                        score=result["score"],
                    )
                )
            summary[name][variant] = dict(
                n=len(selected),
                mean_length=st.mean(r["response_length"] for r in selected),
                median_length=st.median(r["response_length"] for r in selected),
                cap_count=sum(r["finish_reason"] == "length" for r in selected),
                boxed_count=sum("\\boxed" in r["output"] for r in selected),
                correct=sum(values),
            )
    dump(args.data / "probes/question-summary.json", summary)
    dump(args.data / "probes/question-scores.json", scores)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--recompute-text", action="store_true")
    args = parser.parse_args()
    text_metrics(args)
    grade_questions(args)


if __name__ == "__main__":
    main()
