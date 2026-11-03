"""Sensitivity of the completed kappa0 coverage contrast to the two verifiers."""

import json
from pathlib import Path

from analyze_task_controls import ROOT, contrast, digest, load_condition


def main():
    out = ROOT / "coverage-k0-verifiers-20261102.json"
    assert not out.exists()
    labels = ["reuse64x16-k0-s42-block1", "multi128x8-k0-s42-block1"]
    conditions = [load_condition(label, "paper") for label in labels]
    native = contrast(*conditions)
    hashes = {}
    for label, condition in zip(labels, conditions, strict=True):
        path = ROOT / "evaluation" / label / "paper/scores.jsonl"
        groups = {}
        for line in path.open():
            score = json.loads(line)
            groups.setdefault((score["dataset"], score["question_id"]), []).append(
                float(score["official_result"]["acc"])
            )
        for (dataset, question), values in groups.items():
            assert len(values) == 4
            condition["means"][dataset][question][0] = sum(values) / 4
        hashes[label] = {"scores_sha256": digest(path), "completion_sha256": condition["completion_sha256"]}
    report = {
        "left": labels[0],
        "right": labels[1],
        "native": native,
        "official": contrast(*conditions),
        "identity": hashes,
        "script_sha256": digest(Path(__file__)),
        "limits": "Both are automatic grading protocols; intervals omit training seeds and multiplicity.",
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print({name: report[name]["AIME macro"]["accuracy"] for name in ["native", "official"]})


if __name__ == "__main__":
    main()
