"""Continuous-training teacher cohort with local lineage evidence and MiMo endpoints."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOC = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
OUT = ROOT / "direct-pairs-v1"


def main():
    OUT.mkdir(exist_ok=True)
    inv = json.loads((ROOT / "spectrum-expanded-v4/inventory.json").read_text())["runs"]
    conditions = []
    for r in inv:
        direct = r["lineage"] == "published_direct_parent"
        llama = r["student"] == "llama3-3b-off" and r["architecture_compatible"]
        if not (direct or llama):
            continue
        row = dict(
            r,
            family="Llama" if llama else "R1" if r["student"].startswith("r") else "Qwen",
            cohort="continuous",
            label="SFT" if r["added_stage"] == "added SFT" else "RL",
            lineage_basis="local model card named parent; parent revision not independently pinned",
        )
        card = Path(r["teacher_path"]) / "README.md"
        if card.exists():
            target = OUT / "model-cards" / (r["run_id"] + ".md")
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(card.read_bytes())
            row["lineage_card"] = str(target)
            row["lineage_card_sha256"] = hashlib.sha256(card.read_bytes()).hexdigest()
        conditions.append(row)
    ancillary = json.loads((ROOT / "spectrum-expanded-v4/ancillary-complete.json").read_text())["conditions"]
    for r in ancillary:
        row = dict(r)
        if r["run_id"].startswith("controlled-"):
            row.update(
                family="Qwen",
                student="Qwen3-Base controlled",
                teacher=r["run_id"],
                cohort="continuous",
                label="SFT",
                lineage_basis="frozen controlled training protocol",
                added_stage="added SFT",
                lineage="exact_recipe",
            )
        else:
            control = r["run_id"] in ["MiMo-RL", "MiMo-Zero-from-SFT"]
            row.update(
                family="MiMo",
                student="MiMo-SFT" if "from-SFT" in r["run_id"] else "MiMo-Base",
                teacher=r["run_id"],
                cohort="branch_control" if control else "continuous",
                label="SFT" if r["run_id"] == "MiMo-SFT" else "RL",
                lineage_basis="documented MiMo stages",
                added_stage="added SFT" if r["run_id"] == "MiMo-SFT" else "RL",
                lineage="documented_stages",
            )
            if r["run_id"] == "MiMo-RL-from-SFT":
                row["checkpoints"] = {
                    "320": str(REPO / "runs/e93-mimo-rl-sft-20261108/formal/E93/evaluation/final-model/huggingface")
                }
            if r["run_id"] == "MiMo-Zero-from-SFT":
                row["checkpoints"] = {
                    "320": str(REPO / "runs/e91-mimo-rlzero-sft-20261107/formal/E91/evaluation/final-model/huggingface")
                }
        conditions.append(row)
    # Merge repeated input modes for weight work, retaining all output conditions.
    pairs = {}
    for r in conditions:
        ident = hashlib.sha256((r["base"] + "|" + r["teacher_path"]).encode()).hexdigest()[:16]
        p = pairs.setdefault(
            ident,
            dict(
                pair_id=ident,
                base=r["base"],
                teacher_path=r["teacher_path"],
                teacher=r["teacher"],
                family=r["family"],
                cohort=r["cohort"],
                label=r["label"],
                student=r["student"],
                conditions=[],
            ),
        )
        p["conditions"].append(
            dict(
                run_id=r["run_id"],
                student=r["student"],
                checkpoints=r.get("checkpoints", {}),
                lineage_basis=r["lineage_basis"],
            )
        )
    obj = dict(
        conditions=conditions,
        pairs=list(pairs.values()),
        definition="Actual initial student is the declared training parent; branch controls explicitly separate; repeated modes share teacher weights but preserve student checkpoint identity",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "inventory.json").write_text(json.dumps(obj, indent=2))
    print(
        "conditions",
        len(conditions),
        "pairs",
        len(pairs),
        "continuous pairs",
        sum(p["cohort"] == "continuous" for p in pairs.values()),
    )
    for fam in ["R1", "Qwen", "Llama", "MiMo"]:
        print(fam, [(p["teacher"], p["label"], p["cohort"]) for p in pairs.values() if p["family"] == fam])


if __name__ == "__main__":
    main()
