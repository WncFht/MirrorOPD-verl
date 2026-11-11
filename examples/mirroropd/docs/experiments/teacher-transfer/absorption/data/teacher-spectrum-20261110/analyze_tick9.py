"""Paired stages and output-format intervention, with explicit-answer audit retained."""

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
    audit_path = ROOT / "answer-format-audit-009.json"
    audit = json.loads(audit_path.read_text())
    hashes = {str(audit_path.relative_to(ROOT)): hashlib.sha256(audit_path.read_bytes()).hexdigest()}

    def read(key):
        path = ROOT / key
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = list(map(json.loads, path.read_text().splitlines()))
        checked = audit["files"][key]["explicit_checks"]
        return [
            r
            | dict(canonical_acc=r["acc"], acc=bool(r["acc"] or checked.get(r["prompt_id"], {}).get("verified", False)))
            for r in raw
        ]

    base = read("generation-readout-stages-v1/base/base/records.jsonl")
    base_by = {r["prompt_id"]: r for r in base}
    stages = {}
    for name, key in {
        "base": "generation-readout-stages-v1/base/base/records.jsonl",
        "sft-early": "generation-readout-stages-v1/sft-early/sft-early/records.jsonl",
        "rl-early": "generation-readout-stages-v1/rl-early/rl-early/records.jsonl",
        "sft-endpoint": "generation-readout-expanded-v1/endpoint/sft-endpoint/records.jsonl",
        "rl-endpoint": "generation-readout-expanded-v1/endpoint/rl-endpoint/records.jsonl",
        "sft-null": "generation-readout-stages-v1/sft-missing-null/sft-missing-null/records.jsonl",
    }.items():
        rows = read(key)
        assert {r["prompt_id"] for r in rows} == set(base_by)
        stages[name] = dict(
            supplemented=summary(rows, base_by),
            canonical=summary([r | dict(acc=r["canonical_acc"]) for r in rows], base_by),
            base_success_lost=[r["prompt_id"] for r in rows if base_by[r["prompt_id"]]["acc"] and not r["acc"]],
            base_failure_recovered=[r["prompt_id"] for r in rows if not base_by[r["prompt_id"]]["acc"] and r["acc"]],
        )
    formatting = {}
    for lane in ["endpoint", "patch-off-b1", "teacher", "top1"]:
        old = read(f"generation-readout-expanded-v1/{lane}/sft-{lane}/records.jsonl")
        new = read(f"generation-readout-format-v1/{lane}/{lane}/records.jsonl")
        old_by = {r["prompt_id"]: r for r in old}
        assert {r["prompt_id"] for r in new} == set(old_by)
        formatting[lane] = dict(
            original=summary(old, old_by),
            with_instruction=summary(new, old_by),
            canonical_with_instruction=summary([r | dict(acc=r["canonical_acc"]) for r in new], old_by),
            stats={
                label: dict(
                    boxed=sum("\\boxed" in r["output"] for r in values),
                    hash_final=sum(r["prompt_id"] in audit["files"][key]["explicit_checks"] for r in values),
                    truncated=sum(r["finish_reason"] == "length" for r in values),
                    mean_repeat8=sum(r.get("repeat8", 0) for r in values) / len(values),
                )
                for label, values, key in [
                    ("original", old, f"generation-readout-expanded-v1/{lane}/sft-{lane}/records.jsonl"),
                    ("with_instruction", new, f"generation-readout-format-v1/{lane}/{lane}/records.jsonl"),
                ]
            },
        )
    report = dict(stages=stages, formatting=formatting, sha256=hashes)
    args.output.write_text(json.dumps(report, indent=2))
    print("STAGES", {k: {d: v["correct"] for d, v in x["supplemented"].items()} for k, x in stages.items()})
    print(
        "FORMAT",
        {
            k: {
                label: {d: v["correct"] for d, v in x[label].items()}
                for label in ["original", "with_instruction", "canonical_with_instruction"]
            }
            for k, x in formatting.items()
        },
    )
    print("STATS", {k: v["stats"] for k, v in formatting.items()})


if __name__ == "__main__":
    main()
