"""Calibration-selected intervention effects with question-paired uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tick", type=int, default=2)
    args = parser.parse_args()
    plan = json.loads((ROOT / "manifest.json").read_text())
    rng = np.random.default_rng(2026111002)
    summary = {}
    for config in plan["configs"]:
        label = config["label"]
        endpoint = json.loads((ROOT / f"results/{label}.risk-s320/risk.json").read_text())
        reference = {r["row_id"]: r for r in endpoint["rows"]}
        records = json.loads(Path(config["records"]).read_text())
        domains = {str(r.get("row_id", r.get("id", i))): r["domain"] for i, r in enumerate(records)}
        edits = {}
        for path in sorted((ROOT / "results").glob(f"{label}.intervention-*/risk.json")):
            data = json.loads(path.read_text())
            assert data["bank"] == endpoint["bank"] and data["cap"] == endpoint["cap"]
            rows = {r["row_id"]: r for r in data["rows"]}
            assert set(rows) == set(reference)
            conditions = {}
            for split in ["train", "heldout"]:
                for domain in ["all", "math", "code", "if"]:
                    ids = [
                        k
                        for k, r in reference.items()
                        if r["split"] == split and (domain == "all" or domains[k] == domain)
                    ]
                    if not ids:
                        continue
                    original = np.array([reference[k]["loss"] for k in ids])
                    edited = np.array([rows[k]["loss"] for k in ids])
                    indices = rng.integers(0, len(ids), (4000, len(ids)))
                    improvement = (original[indices].mean(1) - edited[indices].mean(1)) / original[indices].mean(1)
                    conditions[f"{split}/{domain}"] = dict(
                        n=len(ids),
                        loss=float(edited.mean()),
                        relative_endpoint_improvement=float(1 - edited.mean() / original.mean()),
                        paired_bootstrap_95=np.quantile(improvement, [0.025, 0.975]).tolist(),
                        improved_questions=int((edited < original).sum()),
                    )
            edits[path.parent.name] = conditions
        chosen = {}
        for band in ["diag", "off", "null"]:
            candidates = [(endpoint["rows"], 0, None)]
            for beta in [0.25, 0.5, 1]:
                name = f"{label}.intervention-patch-{band}-b{beta:g}"
                if name in edits:
                    candidates.append((None, beta, name))
            values = [(np.mean([r["loss"] for r in endpoint["rows"] if r["split"] == "train"]), 0, None)]
            values.extend((edits[n]["train/all"]["loss"], b, n) for _, b, n in candidates[1:])
            loss, beta, name = min(values)
            chosen[band] = dict(
                tested_betas=[v[1] for v in values],
                complete_grid=len(values) == 4,
                beta=beta,
                calibration_loss=float(loss),
                condition=name,
                heldout=edits[name]["heldout/all"] if name else None,
            )
        summary[label] = dict(edits=edits, calibration_selected=chosen)
    (ROOT / f"intervention-analysis-{args.tick:03d}.json").write_text(json.dumps(summary, indent=2))
    for label, result in summary.items():
        print(label, json.dumps(result["calibration_selected"]))
        for name, conditions in result["edits"].items():
            if "gaussian" in name or "random_frames" in name:
                print(name, conditions["heldout/all"])


if __name__ == "__main__":
    main()
