"""Complete all 42 historical inventory weights, including non-explorer controls.

The explorer excludes three Polaris conditions. Their original frozen-state
risks remain separately identified; they do not enter the explorer's primary
correlations. Math-1.5B has task evaluation only, without an OPD absorption curve.
"""

import csv
import hashlib
import json
from pathlib import Path

from measure_expanded_spectrum import OUT, ROOT, Measurements, dump

REPO = ROOT.parents[1]
BASE = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"


def main():
    import torch

    torch.set_num_threads(2)
    assert (OUT / "measurement-complete.json").exists()
    inventory_file = BASE / "transfer-mechanism/data/overview-20261029/inventory.json"
    historical = json.loads(inventory_file.read_text())
    runs = json.loads((OUT / "inventory.json").read_text())["runs"]
    models = {m["key"]: m for m in historical["inventory"]}
    existing = {str(Path(r[k]).resolve()) for r in runs for k in ["base", "teacher_path"]}
    m = Measurements()
    extras = []
    for model in historical["inventory"]:
        path = str(Path(model["path"]).resolve())
        if path not in existing:
            for key in m.keys(path):
                m.full(path, key)
            extras.append(model)
    risks_file = BASE / "transfer-mechanism/data/overview-20261029/risks.csv"
    allrisks = list(csv.DictReader(risks_file.open()))
    panels = {
        "Math": ["M01", "M02", "M03", "M04"],
        "Code": ["Code01", "Code02"],
        "IF": ["IF01", "IF02"],
        "All held-out": ["M01", "M02", "M03", "M04", "Code01", "Code02", "IF01", "IF02"],
    }
    results = []
    for condition in historical["pairs"]:
        if not condition["teacher"].startswith("polaris"):
            continue
        raw = next(
            r
            for r in allrisks
            if r["student"] == condition["student"] and r["teacher"] == condition["teacher"] and r["metric"] == "rkl"
        )
        r = dict(
            run_id="supplement-" + condition["student"] + "-" + condition["teacher"],
            base=str(Path(models[condition["student"]]["path"]).resolve()),
            teacher_path=str(Path(models[condition["teacher"]]["path"]).resolve()),
            checkpoints={},
        )
        ident, compatible = m.pair(r)
        destination = OUT / "supplement-runs" / r["run_id"]
        destination.parent.mkdir(exist_ok=True)
        (OUT / "pairs" / ident).rename(destination)
        initial = json.loads(Path(raw["initial_source"]).read_text())
        curves = {}
        for panel, groups in panels.items():
            points = []
            for step in range(0, int(condition["steps"]) + 1, 16):
                path = Path(raw["final_source"]).parent / f"fixed-{step:03d}.json"
                records = json.loads(path.read_text())
                selected = [x for x in records if x["group"] in groups]
                first = [x for x in initial if x["group"] in groups]
                assert {x["id"] for x in first} == {x["id"] for x in selected}
                points.append(
                    dict(
                        step=step,
                        rkl=sum(x["stats"]["all"]["rkl"] for x in selected) / (len(selected) * 1024),
                        source=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            curves[panel] = points
        results.append(dict(**r, compatible=compatible, curves=curves))
    measured = {
        json.loads(p.read_text())["path"]
        for p in (OUT / "weights").glob("*/model.layers.0.self_attn.k_proj.weight.json")
    }
    coverage = [
        dict(**model, measured=str(Path(model["path"]).resolve()) in measured) for model in historical["inventory"]
    ]
    assert len(coverage) == 42 and all(x["measured"] for x in coverage)
    dump(
        OUT / "historical-42-coverage.json",
        dict(
            weights=coverage,
            extra_models=extras,
            inventory_sha256=hashlib.sha256(inventory_file.read_bytes()).hexdigest(),
        ),
    )
    dump(OUT / "supplement-polaris-curves.json", results)
    print("All42 historical weights measured; three Polaris conditions attached; Math15 has no OPD curve.", flush=True)


if __name__ == "__main__":
    main()
