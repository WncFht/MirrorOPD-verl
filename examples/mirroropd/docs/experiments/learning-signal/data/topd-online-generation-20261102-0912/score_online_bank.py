"""Score saved online rollouts before their own training block, without GPU use.

Exploratory paired DAPO readout: one response per common question, original
native verifier, no change to training or the scheduled benchmark evaluations.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
from analyze_online_generation import digest, load_bank, paired

ROOT = Path(__file__).resolve().parent


def initialize():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT / "source"))
    sys.path.insert(0, str(ROOT / "evaluation-runtime-v5"))
    from verifier_cleanup import install_cleanup

    install_cleanup()


def grade(item):
    arm, row = item
    from verl.utils.reward_score.one_shot_opd import reward_func

    entry = {
        "arm": arm,
        "request_id": row["request_id"],
        "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }
    try:
        entry.update(status="scored", result=reward_func("DAPO", row["output"], row["ground_truth"]))
    except Exception as error:
        entry.update(status="verifier_error", error=repr(error))
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert 2 <= args.outer <= 8
    args.out.mkdir(exist_ok=False)
    arms = ["reuse-k0", "reuse-k05", "reuse-k1"]
    banks = {arm: load_bank(ROOT / "online" / arm / f"outer{args.outer:02d}") for arm in arms}
    for arm in arms:
        assert len(banks[arm]["rows"]) == 64
        paired(banks[arms[0]]["rows"], banks[arm]["rows"])
    inputs = [(arm, row) for arm in arms for row in banks[arm]["rows"]]
    scores = []
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
        ) as workers,
        (args.out / "scores.jsonl").open("x") as stream,
    ):
        for score in workers.map(grade, inputs, chunksize=1):
            stream.write(json.dumps(score, ensure_ascii=False) + "\n")
            stream.flush()
            scores.append(score)
            if len(scores) % 32 == 0:
                print("scored", len(scores), "/", len(inputs), flush=True)
    assert len(scores) == len(inputs) and all(s["status"] == "scored" for s in scores)
    values, summary = {}, {}
    for arm in arms:
        by_id = {s["request_id"]: s for s in scores if s["arm"] == arm}
        assert len(by_id) == 64
        rows = sorted(banks[arm]["rows"], key=lambda r: r["request_id"])
        values[arm] = np.asarray(
            [
                [
                    bool(by_id[r["request_id"]]["result"]["acc"]),
                    r["response_length"],
                    r["finish_reason"] == "length",
                ]
                for r in rows
            ],
            dtype=float,
        )
        a = values[arm]
        summary[arm] = {
            "correct": int(a[:, 0].sum()),
            "accuracy": float(a[:, 0].mean()),
            "mean_length": float(a[:, 1].mean()),
            "cap_count": int(a[:, 2].sum()),
            "correct_capped": int((a[:, 0] * a[:, 2]).sum()),
            "bank_sha256": banks[arm]["bank_sha256"],
            "protocol_sha256": banks[arm]["protocol_sha256"],
        }
    indices = np.random.default_rng(20261102).integers(0, 64, size=(10000, 64))
    contrasts = {}
    for left in arms[:2]:
        delta = values["reuse-k1"] - values[left]
        boot = delta[indices].mean(1)
        contrasts[f"reuse-k1_minus_{left}"] = {
            key: {
                "value": float(delta[:, i].mean()),
                "question_bootstrap95": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
            }
            for i, key in enumerate(["accuracy", "mean_length", "cap_rate"])
        }
        contrasts[f"reuse-k1_minus_{left}"]["paired_accuracy_counts"] = {
            "gain": int((delta[:, 0] == 1).sum()),
            "loss": int((delta[:, 0] == -1).sum()),
        }
    files = list((ROOT / "source/verl/utils/reward_score/one_shot_opd").glob("*.py"))
    files += [
        ROOT / "source/verl/utils/reward_score/math_verify.py",
        ROOT / "evaluation-runtime-v5/verifier_cleanup.py",
    ]
    report = {
        "status": "complete",
        "outer": args.outer,
        "parent_optimizer_updates": (args.outer - 1) * 16,
        "responses": len(scores),
        "summary": summary,
        "contrasts": contrasts,
        "script_sha256": digest(Path(__file__)),
        "scores_sha256": digest(args.out / "scores.jsonl"),
        "verifier_sources": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "limits": [
            "Original native automatic verifier; internal fallback errors/timeouts can be scored false.",
            "These are new scheduled training questions before this outer's update, not held-out benchmarks.",
            "One answer per question, one training seed; 10000 paired question resamples omit seed uncertainty.",
            "Exploratory round selected after observing length contrast; no multiplicity adjustment.",
            "Shorter generation and higher accuracy are distinct outcomes; no causal mediation claim.",
        ],
    }
    (args.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "contrasts": contrasts}), flush=True)


if __name__ == "__main__":
    main()
