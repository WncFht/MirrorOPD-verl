"""Paired task and teacher-overlap readout after explicit-answer-only supplementation."""

import json
from pathlib import Path

from analyze_generation import summary

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / "answer-format-audit-008.json").read_text())
    result = {}
    for panel in ["original32", "expanded72"]:
        pairs = {}
        for pair in ["sft", "rl"]:
            rows = {}
            for lane in ["endpoint", "patch-off-b1", "teacher", "top1"]:
                if panel == "expanded72":
                    key = f"generation-readout-expanded-v1/{lane}/{pair}-{lane}/records.jsonl"
                elif lane == "top1":
                    key = f"generation-readout-top-v1/{pair}-top1/{pair}-top1/records.jsonl"
                else:
                    root = "generation-readout-v2" if pair == "sft" else "generation-readout-rl-v1"
                    key = f"{root}/{lane}/{lane}/records.jsonl"
                raw = [json.loads(x) for x in (ROOT / key).read_text().splitlines()]
                rescued = {pid for d in audit["files"][key]["groups"].values() for pid in d["rescued"]}
                rows[lane] = [
                    r | dict(canonical_acc=r["acc"], acc=bool(r["acc"] or r["prompt_id"] in rescued)) for r in raw
                ]
            base = {r["prompt_id"]: r for r in rows["endpoint"]}
            teacher = {r["prompt_id"]: r for r in rows["teacher"]}
            pairs[pair] = dict(
                conditions={lane: summary(values, base) for lane, values in rows.items()},
                teacher_correctness_overlap={
                    lane: dict(
                        matches=sum(r["acc"] == teacher[r["prompt_id"]]["acc"] for r in values),
                        student_unique_success_lost=[
                            r["prompt_id"]
                            for r in values
                            if base[r["prompt_id"]]["acc"] and not teacher[r["prompt_id"]]["acc"] and not r["acc"]
                        ],
                        teacher_unique_success_recovered=[
                            r["prompt_id"]
                            for r in values
                            if not base[r["prompt_id"]]["acc"] and teacher[r["prompt_id"]]["acc"] and r["acc"]
                        ],
                    )
                    for lane, values in rows.items()
                },
            )
        result[panel] = pairs
    output = ROOT / "supplemented-generation-analysis-008.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2))
    for panel, pairs in result.items():
        for pair, value in pairs.items():
            print(
                panel,
                pair,
                {
                    lane: {d: (r["correct"], r["n"], r["paired_bootstrap_95"]) for d, r in groups.items()}
                    for lane, groups in value["conditions"].items()
                },
            )
            print("off_overlap", value["teacher_correctness_overlap"]["patch-off-b1"])


if __name__ == "__main__":
    main()
