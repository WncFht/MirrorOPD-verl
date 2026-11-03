"""Paired difference-in-differences for coverage and kappa0.5 versus1."""

import json
from pathlib import Path

from analyze_task_controls import ROOT, contrast, digest, load_condition


def main():
    output = ROOT / "coverage-kappa-interaction-20261102.json"
    assert not output.exists()
    effects, identities, disagreements = {}, {}, {}
    for arm in ["reuse64x16", "fresh1024x1", "multi128x8"]:
        half = load_condition(f"{arm}-k05-s42-block1", "paper")
        one = load_condition(f"{arm}-k1-s42-block1", "paper")
        assert half is not None and one is not None
        assert half["sources"] == one["sources"]
        assert half["identity"]["decoding"] == one["identity"]["decoding"]
        effects[arm] = {
            "means": {
                ds: {q: one["means"][ds][q] - half["means"][ds][q] for q in half["means"][ds]} for ds in half["means"]
            },
            "sources": half["sources"],
            "identity": half["identity"],
        }
        identities[arm] = {"k05": half["completion_sha256"], "k1": one["completion_sha256"]}
        disagreements[arm] = {
            k: [x for x in c["disagreements"] if x["request_id"].startswith("AIME")]
            for k, c in [("k05", half), ("k1", one)]
        }
    result = {
        "formula": "(kappa1-kappa0.5) under new coverage minus (kappa1-kappa0.5) under reuse64",
        "interactions": {arm: contrast(effects["reuse64x16"], effects[arm]) for arm in ["fresh1024x1", "multi128x8"]},
        "completion_hashes": identities,
        "aime_verifier_disagreements": disagreements,
        "script_sha256": digest(Path(__file__)),
        "analysis_helper_sha256": digest(ROOT / "analyze_task_controls.py"),
        "limits": [
            "10000 paired question bootstrap draws, grouped by benchmark and response samples.",
            "Exploratory interaction after examining the matrix; one training bank seed; no multiplicity adjustment.",
            "This is a direct paired interaction, not a comparison of separate confidence-interval overlap.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    for arm, values in result["interactions"].items():
        print(arm, values["AIME macro"]["accuracy"], values["five-benchmark macro"]["accuracy"])


if __name__ == "__main__":
    main()
