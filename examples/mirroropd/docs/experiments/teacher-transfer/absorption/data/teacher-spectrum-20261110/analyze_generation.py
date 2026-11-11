"""Paired generation analysis, including 8K truncation of the same 16K trajectory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from examples.mirroropd.research.core.math_grading import close_grader, grade  # noqa: E402


def summary(rows, base):
    result = {}
    for domain in sorted({r["data_source"] for r in rows}):
        selected = [r for r in rows if r["data_source"] == domain]
        differences = np.array([float(r["acc"]) - float(base[r["prompt_id"]]["acc"]) for r in selected])
        rng = np.random.default_rng(2026111007)
        indices = rng.integers(0, len(selected), (5000, len(selected)))
        result[domain] = dict(
            n=len(selected),
            correct=sum(bool(r["acc"]) for r in selected),
            truncated=sum(r["finish_reason"] == "length" for r in selected),
            mean_length=float(np.mean([r["response_length"] for r in selected])),
            wrong_to_correct=[r["prompt_id"] for r in selected if r["acc"] and not base[r["prompt_id"]]["acc"]],
            correct_to_wrong=[r["prompt_id"] for r in selected if not r["acc"] and base[r["prompt_id"]]["acc"]],
            paired_accuracy_difference=float(differences.mean()),
            paired_bootstrap_95=np.quantile(differences[indices].mean(1), [0.025, 0.975]).tolist(),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = sorted(args.directory.glob("*/*/records.jsonl"))
    assert files, args.directory
    inputs = json.loads((files[0].parents[1] / "inputs.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(inputs["conditions"][0]["model"])
    raw = {p.parent.name: [json.loads(x) for x in p.read_text().splitlines()] for p in files}
    base = {r["prompt_id"]: r for r in raw["endpoint"]}
    assert all({r["prompt_id"] for r in rows} == set(base) for rows in raw.values())
    report = dict(conditions={}, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    truncated_rows = {}
    for condition, rows in raw.items():
        short = []
        for r in rows:
            if len(r["response_ids"]) <= 8192:
                short.append(r)
                continue
            ids = r["response_ids"][:8192]
            record = r | dict(
                response_ids=ids,
                response_length=len(ids),
                output=tokenizer.decode(ids, skip_special_tokens=True),
                finish_reason="length",
            )
            short.append(grade(record))
        truncated_rows[condition] = short
    short_base = {r["prompt_id"]: r for r in truncated_rows["endpoint"]}
    teacher = {r["prompt_id"]: r for r in raw["teacher"]}
    for condition, rows in raw.items():
        short = {r["prompt_id"]: r for r in truncated_rows[condition]}
        report["conditions"][condition] = dict(
            cap16384=summary(rows, base),
            same_trajectory_cap8192=summary(list(short.values()), short_base),
            cap_extension_changes=[
                dict(
                    prompt_id=r["prompt_id"], correct_8k=bool(short[r["prompt_id"]]["acc"]), correct_16k=bool(r["acc"])
                )
                for r in rows
                if r["acc"] != short[r["prompt_id"]]["acc"]
            ],
            matches_teacher_correctness=sum(bool(r["acc"]) == bool(teacher[r["prompt_id"]]["acc"]) for r in rows),
        )
        if args.prior:
            prior = {
                r["prompt_id"]: r
                for r in map(json.loads, (args.prior / condition / "records.jsonl").read_text().splitlines())
            }
            divergence = []
            for r in rows:
                a, b = prior[r["prompt_id"]]["response_ids"], r["response_ids"]
                prefix = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
                divergence.append(
                    dict(prompt_id=r["prompt_id"], common_prefix=prefix, is_prefix=prefix == min(len(a), len(b)))
                )
            report["conditions"][condition]["prior_prefix_comparison"] = divergence
    close_grader()
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".truncated-records.json").write_text(json.dumps(truncated_rows, ensure_ascii=False))
    print(
        json.dumps(
            {
                k: {c: v[c] for c in ["cap16384", "same_trajectory_cap8192", "cap_extension_changes"]}
                for k, v in report["conditions"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
