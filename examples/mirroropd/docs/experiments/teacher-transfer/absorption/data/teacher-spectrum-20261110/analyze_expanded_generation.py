"""Read predeclared expanded generation pairs; only rank complete paired panels."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_generation import summary

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"pairs": {}, "sha256": {}}
    root = ROOT / "generation-readout-expanded-v1"
    for pair in ["sft", "rl"]:
        conditions = {}
        for lane in ["endpoint", "patch-off-b1", "teacher", "top1"]:
            path = root / lane / f"{pair}-{lane}/records.jsonl"
            if path.exists():
                conditions[lane] = [json.loads(x) for x in path.read_text().splitlines()]
                report["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                conditions[lane] = []
        expected = {p["prompt_id"] for p in json.loads((root / "endpoint/inputs.json").read_text())["prompts"]}
        identity_sets = [{r["prompt_id"] for r in rows} for rows in conditions.values()]
        complete = all(ids == expected for ids in identity_sets)
        result = dict(
            expected=len(expected),
            complete=complete,
            completed={k: len(v) for k, v in conditions.items()},
            common_completed=len(set.intersection(*identity_sets)),
        )
        if complete:
            base = {r["prompt_id"]: r for r in conditions["endpoint"]}
            teacher = {r["prompt_id"]: r for r in conditions["teacher"]}
            result["conditions"] = {k: summary(rows, base) for k, rows in conditions.items()}
            result["teacher_correctness_overlap"] = {}
            for lane, rows in conditions.items():
                result["teacher_correctness_overlap"][lane] = dict(
                    matches=sum(bool(r["acc"]) == bool(teacher[r["prompt_id"]]["acc"]) for r in rows),
                    endpoint_unique_success_lost=[
                        r["prompt_id"]
                        for r in rows
                        if base[r["prompt_id"]]["acc"] and not teacher[r["prompt_id"]]["acc"] and not r["acc"]
                    ],
                    teacher_unique_success_recovered=[
                        r["prompt_id"]
                        for r in rows
                        if not base[r["prompt_id"]]["acc"] and teacher[r["prompt_id"]]["acc"] and r["acc"]
                    ],
                )
        report["pairs"][pair] = result
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pairs"], indent=2))


if __name__ == "__main__":
    main()
