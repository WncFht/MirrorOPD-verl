"""Compare fixed Gaussian deletion controls with actual deletion on identical prompts."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_generation import summary

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    result = {
        "protocol": (
            "Same 72 prompts; original and explicit-final-answer supplemented scores. "
            "One fixed Gaussian seed, per-matrix equal norm."
        ),
        "conditions": {},
        "sha256": {},
    }

    def read(key, supplemented):
        path = ROOT / key
        result["sha256"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = list(map(json.loads, path.read_text().splitlines()))
        assert len(rows) == 72 and len({r["prompt_id"] for r in rows}) == 72
        rescued = (
            {pid for g in audit["files"][key]["groups"].values() for pid in g["rescued"]} if supplemented else set()
        )
        return [r | dict(acc=bool(r["acc"] or r["prompt_id"] in rescued)) for r in rows]

    for pair in ["sft", "rl"]:
        for target, lane in [("teacher", "teacher"), ("opd", "endpoint")]:
            label = f"{pair}-{target}"
            keys = {
                "unedited": f"generation-readout-expanded-v1/{lane}/{pair}-{lane}/records.jsonl",
                "delete_off": f"generation-readout-delete-v1/{label}-delete-off/{label}-delete-off/records.jsonl",
                "gaussian": f"generation-readout-delete-noise-v1/{label}-gaussian/{label}-gaussian/records.jsonl",
            }
            result["conditions"][label] = {}
            for mode in ["canonical", "supplemented"]:
                panels = {name: read(key, mode == "supplemented") for name, key in keys.items()}
                base = {r["prompt_id"]: r for r in panels["unedited"]}
                noise = {r["prompt_id"]: r for r in panels["gaussian"]}
                assert all({r["prompt_id"] for r in rows} == set(base) for rows in panels.values())
                report = {name: summary(rows, base) for name, rows in panels.items()}
                report["deletion_vs_gaussian"] = summary(panels["delete_off"], noise)
                # Also bootstrap the complete fixed panel, retaining paired prompt identity.
                for name, rows in panels.items():
                    report[name]["all"] = summary([r | {"data_source": "all"} for r in rows], base)["all"]
                report["deletion_vs_gaussian"]["all"] = summary(
                    [r | {"data_source": "all"} for r in panels["delete_off"]], noise
                )["all"]
                result["conditions"][label][mode] = report
    args.output.write_text(json.dumps(result, indent=2))
    for label, values in result["conditions"].items():
        print(
            label,
            {name: data["all"]["correct"] for name, data in values["supplemented"].items()},
            values["supplemented"]["deletion_vs_gaussian"]["all"]["paired_bootstrap_95"],
        )


if __name__ == "__main__":
    main()
