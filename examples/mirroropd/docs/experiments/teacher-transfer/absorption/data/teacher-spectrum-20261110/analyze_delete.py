"""Same-prompt own-update off deletion compared with its unedited target."""

import json
from pathlib import Path

from analyze_generation import summary

ROOT = Path(__file__).resolve().parent


def main():
    audit = json.loads((ROOT / "answer-format-delete-audit-008.json").read_text())
    result = {}
    for pair in ["sft", "rl"]:
        for target, lane in [("teacher", "teacher"), ("opd", "endpoint")]:
            key = f"generation-readout-delete-v1/{pair}-{target}-delete-off/{pair}-{target}-delete-off/records.jsonl"
            basekey = f"generation-readout-expanded-v1/{lane}/{pair}-{lane}/records.jsonl"
            panels = []
            for path in [key, basekey]:
                rows = list(map(json.loads, (ROOT / path).read_text().splitlines()))
                rescued = {pid for group in audit["files"][path]["groups"].values() for pid in group["rescued"]}
                panels.append([r | dict(acc=bool(r["acc"] or r["prompt_id"] in rescued)) for r in rows])
            base = {r["prompt_id"]: r for r in panels[1]}
            assert {r["prompt_id"] for r in panels[0]} == set(base)
            result[f"{pair}-{target}"] = summary(panels[0], base)
    out = ROOT / "delete-paired-supplemented-008.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
