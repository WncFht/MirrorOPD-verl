"""Paired generation behavior on the same newly scheduled questions.

Outer-j generations use the weights after (j-1)*16 updates. Before/after training
risks and rollout behavior have different time indices. Comparisons keep input
tokens and request seeds identical, but include each arm's accumulated learning
and visitation effects. Length and cap rate are not mathematical correctness.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path):
    done = json.loads((path / "bank-complete.json").read_text())
    assert digest(path / "bank.jsonl") == done["sha256"]
    with (path / "bank.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == done["responses"] == len({r["request_id"] for r in rows})
    protocol = json.loads((path / "protocol.json").read_text())
    assert protocol["rollout"] == {"temperature": 1.0, "top_p": 1.0, "cap": 16384, "eos": [151643, 151645]}
    return {
        "rows": rows,
        "protocol": protocol,
        "bank_sha256": done["sha256"],
        "protocol_sha256": digest(path / "protocol.json"),
    }


def metrics(rows):
    values = np.array([[r["response_length"], float(r["finish_reason"] == "length")] for r in rows])
    assert all(len(r["response_ids"]) == r["response_length"] for r in rows)
    return {
        "responses": len(rows),
        "tokens": int(values[:, 0].sum()),
        "mean_length": float(values[:, 0].mean()),
        "median_length": float(np.median(values[:, 0])),
        "cap_count": int(values[:, 1].sum()),
        "cap_rate": float(values[:, 1].mean()),
    }


def paired(left, right):
    a = {r["request_id"]: r for r in left}
    b = {r["request_id"]: r for r in right}
    assert set(a) == set(b)
    keys = sorted(a)
    assert len(keys) == 64
    for key in keys:
        for field in ["id", "source_index", "prompt_ids", "seed", "ground_truth", "sample"]:
            assert a[key][field] == b[key][field]
    delta = np.asarray(
        [
            [
                b[k]["response_length"] - a[k]["response_length"],
                int(b[k]["finish_reason"] == "length") - int(a[k]["finish_reason"] == "length"),
            ]
            for k in keys
        ],
        dtype=float,
    )
    indices = np.random.default_rng(20261102).integers(0, 64, size=(10000, 64))
    boot = delta[indices].mean(1)
    return {
        name: {
            "value": float(delta[:, i].mean()),
            "question_bootstrap95": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
        }
        for i, name in enumerate(["mean_length", "cap_rate"])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    banks, summaries, comparisons = {}, {}, []
    for seed, directory in [
        (42, ROOT / "online"),
        (43, ROOT / "replications/seed43"),
        (44, ROOT / "replications/seed44"),
    ]:
        for receipt in sorted(directory.glob("*/outer*/bank-complete.json")):
            path = receipt.parent
            arm, outer = path.parent.name, int(path.name.removeprefix("outer"))
            data = load_bank(path)
            assert data["protocol"]["outer"] == outer
            rows = data["rows"]
            assert all(row["sample"] == 0 for row in rows)
            common = rows[:64]
            assert [r["id"] for r in common] == list(range((outer - 1) * 1024, (outer - 1) * 1024 + 64))
            key = f"seed{seed}/{arm}/outer{outer:02d}"
            banks[key] = common
            summaries[key] = {
                "parent_optimizer_updates": (outer - 1) * 16,
                "all_generated": metrics(rows),
                "common_first64": metrics(common),
                "bank_sha256": data["bank_sha256"],
                "protocol_sha256": data["protocol_sha256"],
                "parent_checkpoint_sha256": data["protocol"]["checkpoint_sha256"],
            }
        for outer in range(1, 9):
            for left, right in [("reuse-k0", "reuse-k1"), ("reuse-k05", "reuse-k1"), ("reuse-k1", "fresh-k1")]:
                a, b = f"seed{seed}/{left}/outer{outer:02d}", f"seed{seed}/{right}/outer{outer:02d}"
                if a in banks and b in banks:
                    if outer == 1:
                        assert all(
                            x["response_ids"] == y["response_ids"] for x, y in zip(banks[a], banks[b], strict=True)
                        )
                    comparisons.append(
                        {
                            "left": a,
                            "right": b,
                            "parent_updates": (outer - 1) * 16,
                            "right_minus_left": paired(banks[a], banks[b]),
                        }
                    )
    report = {
        "scope": __doc__,
        "banks": summaries,
        "comparisons": comparisons,
        "script_sha256": digest(Path(__file__)),
        "limits": [
            "Only complete, hashed banks are included; the corresponding training block may still be running.",
            "Each outer has new questions; cross-outer differences also change the question set.",
            "Within-outer contrasts pair the same64 questions and sampling seeds.",
            "Fresh1024 contrasts use its nested first64; full1024 summaries are reported separately.",
            "One answer per question; question intervals do not include training-seed variance.",
            "Initial replication banks share identical trajectories by design; not independent equality evidence.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for comparison in comparisons:
        if comparison["parent_updates"]:
            print(comparison["left"], "->", comparison["right"], comparison["right_minus_left"], flush=True)


if __name__ == "__main__":
    main()
