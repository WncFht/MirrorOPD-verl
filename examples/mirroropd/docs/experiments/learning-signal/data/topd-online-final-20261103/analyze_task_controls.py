"""Completed task controls only; paired question bootstrap, fixed evaluation seeds.

The output lists incomplete conditions separately. It never treats a missing
completion as a zero, and never counts generations before the completion hashes
and question/seed identities have been checked.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ["AIME24", "AIME25", "AIME26", "AMC23", "MATH-500"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition(label, protocol):
    out = ROOT / "evaluation" / label / protocol
    if not (out / "complete.json").exists():
        return None
    done = json.loads((out / "complete.json").read_text())
    assert done["status"] == "complete"
    identity = done["identity"]
    assert identity["label"] == label and identity["protocol"] == protocol
    assert identity["input_variant"] == "local" and identity["n"] == 4
    assert identity["question_plan_sha256"] == digest(ROOT / "evaluation-prompts.json")
    assert identity["evaluation_plan_sha256"] == digest(ROOT / "evaluation-plan.json")
    for name in ["responses", "scores", "summary"]:
        suffix = ".json" if name == "summary" else ".jsonl"
        assert digest(out / (name + suffix)) == done[name + "_sha256"]
    with (out / "scores.jsonl").open() as stream:
        scores = [json.loads(line) for line in stream]
    indexed = {s["request_id"]: s for s in scores}
    assert len(scores) == len(indexed) == 776
    groups, sources, disagreements = {}, {}, []
    with (out / "responses.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["request_id"]
            assert rid not in sources
            score = indexed[rid]
            assert score["status"] == score["official_status"] == "scored"
            assert all(score[key] == row[key] for key in ["dataset", "question_id", "sample"])
            sources[rid] = hashlib.sha256(
                json.dumps(
                    {k: row[k] for k in ["seed", "prompt_ids", "ground_truth", "question_id", "dataset", "sample"]},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            groups.setdefault(row["dataset"], {}).setdefault(row["question_id"], []).append(
                [float(score["result"]["acc"]), row["response_length"], float(row["finish_reason"] == "length")]
            )
            if bool(score["result"]["acc"]) != bool(score["official_result"]["acc"]):
                disagreements.append(
                    {"request_id": rid, "native": score["result"], "official": score["official_result"]}
                )
    assert set(sources) == set(indexed)
    means = {}
    for ds in DATASETS:
        expected = 64 if ds == "MATH-500" else 40 if ds == "AMC23" else 30
        assert len(groups[ds]) == expected and all(len(v) == 4 for v in groups[ds].values())
        means[ds] = {q: np.asarray(values).mean(0) for q, values in groups[ds].items()}
    summary = json.loads((out / "summary.json").read_text())
    for ds in DATASETS:
        assert abs(np.mean([v[0] for v in means[ds].values()]) - summary["datasets"][ds]["observed_mean"]) < 1e-12
    return {
        "means": means,
        "sources": sources,
        "identity": identity,
        "summary": summary,
        "disagreements": disagreements,
        "completion_sha256": digest(out / "complete.json"),
    }


def contrast(left, right):
    assert left["sources"] == right["sources"], "Question, input, sample seed or answer identity changed"
    assert left["identity"]["decoding"] == right["identity"]["decoding"]
    random = np.random.default_rng(20261102)
    result = {}
    for name, datasets in [(ds, [ds]) for ds in DATASETS] + [
        ("AIME macro", DATASETS[:3]),
        ("five-benchmark macro", DATASETS),
    ]:
        boot, point = np.zeros((10000, 3)), np.zeros(3)
        for ds in datasets:
            ids = sorted(left["means"][ds])
            assert ids == sorted(right["means"][ds])
            delta = np.stack([right["means"][ds][q] - left["means"][ds][q] for q in ids])
            boot += delta[random.integers(0, len(ids), size=(10000, len(ids)))].mean(1) / len(datasets)
            point += delta.mean(0) / len(datasets)
        result[name] = {
            key: {"value": float(point[i]), "question_bootstrap95": np.quantile(boot[:, i], [0.025, 0.975]).tolist()}
            for i, key in enumerate(["accuracy", "mean_length", "cap_rate"])
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Preserve each dated analysis"
    labels = ["base"] + [
        f"{a}-{k}-s42-block1" for a in ["reuse64x16", "fresh1024x1", "multi128x8"] for k in ["k0", "k05", "k1"]
    ]
    labels += ["budget-b256-u4-k1-s42", "budget-b256-u16-k1-s42", "budget-b64-u64-k1-s42"]
    labels += [f"online-{a}-outer08" for a in ["reuse-k0", "reuse-k05", "reuse-k1", "fresh-k1"]]
    labels += [
        f"online-seed{seed}-{arm}-outer08"
        for seed in [43, 44]
        for arm in ["reuse-k0", "reuse-k05", "reuse-k1", "fresh-k1"]
    ]
    available, pending = {}, []
    for protocol in ["local", "paper"]:
        for label in labels:
            if label.startswith("online-") and protocol == "local":
                continue
            key = label + "/" + protocol
            data = load_condition(label, protocol)
            if data is None:
                pending.append(key)
            else:
                available[key] = data
    pairs = [("base", label) for label in labels[1:]]
    pairs += [
        (f"{arm}-k{k}-s42-block1", f"{arm}-k1-s42-block1")
        for arm in ["reuse64x16", "fresh1024x1", "multi128x8"]
        for k in ["0", "05"]
    ]
    pairs += [
        (f"reuse64x16-{k}-s42-block1", f"{arm}-{k}-s42-block1")
        for k in ["k0", "k05", "k1"]
        for arm in ["fresh1024x1", "multi128x8"]
    ]
    pairs += [("fresh1024x1-k1-s42-block1", label) for label in labels if label.startswith("budget-")]
    pairs += [
        ("budget-b256-u4-k1-s42", "budget-b256-u16-k1-s42"),
        ("budget-b256-u16-k1-s42", "budget-b64-u64-k1-s42"),
    ]
    pairs += [
        ("online-reuse-k1-outer08", "online-fresh-k1-outer08"),
        ("online-reuse-k05-outer08", "online-reuse-k1-outer08"),
        ("online-reuse-k0-outer08", "online-reuse-k1-outer08"),
    ]
    pairs += [
        (f"online-seed{seed}-{left}-outer08", f"online-seed{seed}-{right}-outer08")
        for seed in [43, 44]
        for left, right in [("reuse-k1", "fresh-k1"), ("reuse-k05", "reuse-k1"), ("reuse-k0", "reuse-k1")]
    ]
    contrasts = []
    for protocol in ["local", "paper"]:
        for a, b in pairs:
            left, right = a + "/" + protocol, b + "/" + protocol
            if left in available and right in available:
                contrasts.append(
                    {"left": left, "right": right, "right_minus_left": contrast(available[left], available[right])}
                )
    report = {
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "conditions": {
            k: {field: value[field] for field in ["summary", "identity", "completion_sha256", "disagreements"]}
            for k, value in available.items()
        },
        "pending": pending,
        "contrasts": contrasts,
        "limits": [
            "10000 paired question bootstrap draws, stratified by benchmark; four samples stay grouped.",
            "Each contrast fixes one training/rollout seed; intervals omit cross-seed variation and multiple testing.",
            "MATH is a fixed 64-question subset, not all MATH-500.",
            "Incomplete conditions are not ranked. Decoder and input settings must match within a contrast.",
            "Official wrapper returns zero on its one-second timeout; disagreement alone does not establish a bug.",
        ],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for item in contrasts:
        print(item["left"], "->", item["right"], json.dumps(item["right_minus_left"]["AIME macro"]))


if __name__ == "__main__":
    main()
