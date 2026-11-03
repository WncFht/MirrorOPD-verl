"""Complete paper-decoder input controls across the three kappa1 coverage arms."""

import json
from pathlib import Path

import numpy as np
from analyze_instruction_interaction import METRICS, ROOT, contrast, digest, load


def main():
    output = ROOT / "coverage-instruction-paper-20261102.json"
    assert not output.exists()
    labels = ["base", "reuse64x16-k1-s42-block1", "fresh1024x1-k1-s42-block1", "multi128x8-k1-s42-block1"]
    available = {
        label + "/" + variant: load(label, "paper", variant == "instruction")
        for label in labels
        for variant in ["native", "instruction"]
    }
    for label in labels:
        assert (
            available[label + "/native"]["identity"]["weight_sha256"]
            == available[label + "/instruction"]["identity"]["weight_sha256"]
        )
        for variant in ["native", "instruction"]:
            assert available[label + "/" + variant]["inputs"] == available["base/" + variant]["inputs"]
    results = {}
    for label in labels:
        terms = [(label + "/instruction", 1), (label + "/native", -1)]
        results[label + "/instruction_effect"] = {"terms": terms, "metrics": contrast(terms, available)}
        if label != "base":
            terms = terms + [("base/instruction", -1), ("base/native", 1)]
            results[label + "/training_input_interaction"] = {"terms": terms, "metrics": contrast(terms, available)}
    reuse = labels[1]
    for label in labels[2:]:
        terms = [(label + "/instruction", 1), (reuse + "/instruction", -1)]
        results[label + "/coverage_with_instruction"] = {"terms": terms, "metrics": contrast(terms, available)}
        terms = [
            (label + "/instruction", 1),
            (label + "/native", -1),
            (reuse + "/instruction", -1),
            (reuse + "/native", 1),
        ]
        results[label + "/coverage_input_interaction"] = {"terms": terms, "metrics": contrast(terms, available)}
    previous = json.loads((ROOT / "instruction-training-interaction.json").read_text())
    for old_name, new_key in [
        ("instruction_at_base", "base/instruction_effect"),
        ("instruction_after_training", reuse + "/instruction_effect"),
        ("interaction", reuse + "/training_input_interaction"),
    ]:
        old = next(c for c in previous["contrasts"] if c["protocol"] == "paper" and c["contrast"] == old_name)
        for metric in METRICS:
            for field in ["value", "question_bootstrap95"]:
                np.testing.assert_allclose(
                    old["metrics"][metric][field], results[new_key]["metrics"][metric][field], atol=1e-12, rtol=1e-12
                )
    report = {
        "scope": __doc__,
        "conditions": {
            k: {field: v[field] for field in ["identity", "aggregate", "source_sha256"]} for k, v in available.items()
        },
        "contrasts": results,
        "original_base_and_reuse_results_reproduced": True,
        "script_sha256": digest(Path(__file__)),
        "helper_sha256": digest(ROOT / "analyze_instruction_interaction.py"),
        "limits": [
            "Same frozen model within input contrasts; same questions, ground truths and seeds.",
            "10000 stratified paired question bootstrap draws; all four responses stay grouped.",
            "One initial training bank seed; no multiplicity adjustment or training-seed uncertainty.",
            "Inference instruction controls only; training instructions were unchanged.",
            "Both automatic verifiers retain their extraction and timeout-to-zero limitations.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    for key, value in results.items():
        print(key, value["metrics"]["native_accuracy"])


if __name__ == "__main__":
    main()
