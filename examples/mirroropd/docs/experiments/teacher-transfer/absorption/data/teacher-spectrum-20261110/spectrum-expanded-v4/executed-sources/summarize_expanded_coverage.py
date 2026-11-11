"""Finish historical lineage joins without rerunning parameter measurements."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "spectrum-expanded-v4"
DOC = ROOT.parents[1] / "examples/mirroropd/docs/experiments/teacher-transfer"


def main():
    ancillary = json.loads((OUT / "ancillary-summary.json").read_text())
    source = DOC / "figures/mimo-stage-20261104-final/e91-comparison/rkl-comparison.csv"
    curves = list(csv.DictReader(source.open()))
    for row in ancillary["conditions"]:
        arm = {"MiMo-RL-from-SFT": "E93", "MiMo-Zero-from-SFT": "E91"}.get(row["run_id"])
        if arm:
            row["curves"] = [dict(x, bank="public_teacher", domain="all") for x in curves if x["arm"] == arm]
            row["curve_source"] = str(source)
            row["curve_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT / "ancillary-complete.json").write_text(json.dumps(ancillary, indent=2))
    historical = json.loads((OUT / "historical-42-coverage.json").read_text())["weights"]
    references = {str(Path(r["path"]).resolve()) for r in historical}
    for r in ancillary["conditions"]:
        references.update([r["base"], r["teacher_path"]])
    for path in sorted(references):
        directory = OUT / "weights" / hashlib.sha256(path.encode()).hexdigest()[:16]
        assert len(list(directory.glob("*.json"))) == 6, (path, directory)
    summary = dict(
        reference_weights=len(references),
        historical_reference_weights=len(historical),
        ancillary_pairs=len(ancillary["conditions"]),
        full_weight_checkpoints=len(list((OUT / "weights").iterdir())),
        full_weight_matrices=len(list((OUT / "weights").glob("*/*.json"))),
        reference_paths=sorted(references),
        scope="Six k/v projection matrices per checkpoint; CPU FP64 exact SVD; no GPU inference",
        original_explorer_conditions=47,
        supplemental_polaris_conditions=3,
    )
    (OUT / "final-coverage.json").write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "reference_paths"})
    for r in ancillary["conditions"]:
        a = [
            x
            for x in r.get("curves", [])
            if x["bank"] in ["public_teacher", "teacher-bank"] and x["domain"] in ["all", "macro4"]
        ]
        print(r["run_id"], r["features"], [(x.get("step", x.get("optimizer_updates")), float(x["rkl"])) for x in a])


if __name__ == "__main__":
    main()
