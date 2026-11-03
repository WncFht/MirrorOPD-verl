"""Paired AIME input effects before/after the common kappa1 initial block.

The interaction is (trained instruction effect) minus (Base instruction effect).
All four samples stay together in each stratified question bootstrap. This is
inference-input evidence, not an experiment on instructions in training data.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRAINED = "reuse64x16-k1-s42-block1"
DATASETS = ["AIME24", "AIME25", "AIME26"]
METRICS = ["native_accuracy", "official_accuracy", "mean_length", "cap_rate"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(label, protocol, instruction):
    name = protocol + ("-official-instruction" if instruction else "")
    folder = ROOT / "evaluation" / label / name
    done = json.loads((folder / "complete.json").read_text())
    assert done["status"] == "complete"
    identity = done["identity"]
    assert identity["label"] == label and identity["protocol"] == protocol and identity["n"] == 4
    assert identity["input_variant"] == ("official_instruction" if instruction else "local")
    prompts = "prompt-control-prompts.json" if instruction else "evaluation-prompts.json"
    assert identity["question_plan_sha256"] == digest(ROOT / prompts)
    for key in ["responses", "scores", "summary"]:
        extension = "json" if key == "summary" else "jsonl"
        assert digest(folder / f"{key}.{extension}") == done[f"{key}_sha256"]
    with (folder / "scores.jsonl").open() as stream:
        scored = [json.loads(line) for line in stream]
    scores = {r["request_id"]: r for r in scored}
    assert len(scores) == len(scored)
    groups, sources, inputs = {}, {}, {}
    with (folder / "responses.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["dataset"] not in DATASETS:
                continue
            rid = row["request_id"]
            assert rid not in sources
            score = scores[rid]
            assert score["status"] == score["official_status"] == "scored"
            groups.setdefault(row["question_id"], []).append(
                [
                    float(score["result"]["acc"]),
                    float(score["official_result"]["acc"]),
                    row["response_length"],
                    float(row["finish_reason"] == "length"),
                ]
            )
            sources[rid] = {key: row[key] for key in ["seed", "ground_truth", "dataset", "question_id", "sample"]}
            inputs[rid] = row["prompt_ids"]
    assert len(groups) == 90 and len(sources) == 360 and all(len(v) == 4 for v in groups.values())
    means = {key: np.asarray(value).mean(0) for key, value in groups.items()}
    overall = np.mean(list(means.values()), axis=0)
    return {
        "means": means,
        "sources": sources,
        "inputs": inputs,
        "identity": identity,
        "aggregate": dict(zip(METRICS, overall.tolist(), strict=True)),
        "source_sha256": {key: done[f"{key}_sha256"] for key in ["responses", "scores", "summary"]},
    }


def contrast(terms, available):
    reference = available[terms[0][0]]
    assert all(available[name]["sources"] == reference["sources"] for name, _ in terms)
    assert all(available[name]["identity"]["decoding"] == reference["identity"]["decoding"] for name, _ in terms)
    rng = np.random.default_rng(20261102)
    point, boot = np.zeros(4), np.zeros((10000, 4))
    for dataset in DATASETS:
        ids = sorted(q for q in reference["means"] if q.startswith(dataset + ":"))
        assert len(ids) == 30
        delta = sum(weight * np.stack([available[name]["means"][q] for q in ids]) for name, weight in terms)
        boot += delta[rng.integers(0, 30, size=(10000, 30))].mean(1) / 3
        point += delta.mean(0) / 3
    return {
        key: {"value": float(point[i]), "question_bootstrap95": np.quantile(boot[:, i], [0.025, 0.975]).tolist()}
        for i, key in enumerate(METRICS)
    }


def main():
    out = ROOT / "instruction-training-interaction.json"
    assert not out.exists()
    available = {}
    for label in ["base", TRAINED]:
        for protocol in ["local", "paper"]:
            for instruction in [False, True]:
                key = f"{label}/{protocol}/{'instruction' if instruction else 'native'}"
                available[key] = load(label, protocol, instruction)
    for protocol in ["local", "paper"]:
        for variant in ["native", "instruction"]:
            assert (
                available[f"base/{protocol}/{variant}"]["inputs"]
                == available[f"{TRAINED}/{protocol}/{variant}"]["inputs"]
            )
    for label in ["base", TRAINED]:
        weights = [
            value["identity"]["weight_sha256"] for key, value in available.items() if key.startswith(label + "/")
        ]
        assert all(value == weights[0] for value in weights)
    results = []
    for protocol in ["local", "paper"]:
        base, trained = f"base/{protocol}", f"{TRAINED}/{protocol}"
        definitions = [
            ("instruction_at_base", [(base + "/instruction", 1), (base + "/native", -1)]),
            ("instruction_after_training", [(trained + "/instruction", 1), (trained + "/native", -1)]),
            ("training_native_input", [(trained + "/native", 1), (base + "/native", -1)]),
            ("training_instruction_input", [(trained + "/instruction", 1), (base + "/instruction", -1)]),
            (
                "interaction",
                [
                    (trained + "/instruction", 1),
                    (trained + "/native", -1),
                    (base + "/instruction", -1),
                    (base + "/native", 1),
                ],
            ),
        ]
        for name, terms in definitions:
            result = {"protocol": protocol, "contrast": name, "terms": terms, "metrics": contrast(terms, available)}
            results.append(result)
            print(protocol, name, json.dumps(result["metrics"]["native_accuracy"]), flush=True)
    original = json.loads((ROOT / "base-factorial-analysis.json").read_text())
    for protocol in ["local", "paper"]:
        old = next(
            c
            for c in original["contrasts"]
            if c["left"] == protocol and c["right"] == protocol + "-official-instruction"
        )
        new = next(c for c in results if c["protocol"] == protocol and c["contrast"] == "instruction_at_base")
        for before, after in [
            ("accuracy", "native_accuracy"),
            ("mean_length", "mean_length"),
            ("cap_rate", "cap_rate"),
        ]:
            np.testing.assert_allclose(
                old["right_minus_left"][before]["value"], new["metrics"][after]["value"], atol=1e-12
            )
            np.testing.assert_allclose(
                old["right_minus_left"][before]["question_bootstrap95"],
                new["metrics"][after]["question_bootstrap95"],
                atol=1e-12,
            )
    report = {
        "scope": __doc__,
        "conditions": {
            key: {field: value[field] for field in ["identity", "aggregate", "source_sha256"]}
            for key, value in available.items()
        },
        "contrasts": results,
        "original_base_result_reproduced": True,
        "script_sha256": digest(Path(__file__)),
        "limits": [
            "One initial training block and one data/rollout seed.",
            "Intervals resample 90 paired questions, stratified by year, keeping four samples together.",
            "No correction for multiple comparisons and no training-seed uncertainty.",
            "This tests inference instructions, not training instructions or later online checkpoints.",
            "Both automated verifiers have documented extraction limitations.",
        ],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
