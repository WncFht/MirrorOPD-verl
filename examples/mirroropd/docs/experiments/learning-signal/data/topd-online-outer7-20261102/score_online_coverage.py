"""Grade complete fresh banks and pair only their nested64 with reuse-k1.

The full1024 result is descriptive; the causal coverage contrast uses the same
64 questions, input tokens and request seeds before this outer's training.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from analyze_online_generation import paired
from score_online_bank import ROOT, digest, grade, initialize, load_bank


def row_identity(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def values(rows, scores):
    indexed = {s["request_id"]: s for s in scores}
    assert len(indexed) == len(scores)
    result = []
    for row in sorted(rows, key=lambda x: x["request_id"]):
        score = indexed[row["request_id"]]
        assert score["status"] == "scored" and score["row_sha256"] == row_identity(row)
        result.append([bool(score["result"]["acc"]), row["response_length"], row["finish_reason"] == "length"])
    return np.asarray(result, dtype=float)


def summarize(array):
    return {
        "responses": len(array),
        "correct": int(array[:, 0].sum()),
        "accuracy": float(array[:, 0].mean()),
        "mean_length": float(array[:, 1].mean()),
        "cap_count": int(array[:, 2].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outers", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.outers) == len(set(args.outers)) and all(2 <= o <= 8 for o in args.outers)
    args.out.mkdir(exist_ok=False)
    report = {"outers": {}, "script_sha256": digest(Path(__file__))}
    for outer in args.outers:
        fresh = load_bank(ROOT / "online/fresh-k1" / f"outer{outer:02d}")
        reuse = load_bank(ROOT / "online/reuse-k1" / f"outer{outer:02d}")
        assert len(fresh["rows"]) == 1024 and len(reuse["rows"]) == 64
        matched = fresh["rows"][:64]
        paired(reuse["rows"], matched)
        previous = ROOT / f"online-outer{outer}-scores-20261102"
        receipt = json.loads((previous / "complete.json").read_text())
        assert receipt["status"] == "complete" and receipt["scores_sha256"] == digest(previous / "scores.jsonl")
        assert receipt["summary"]["reuse-k1"]["bank_sha256"] == reuse["bank_sha256"]
        reuse_scores = [
            s for line in (previous / "scores.jsonl").open() if (s := json.loads(line))["arm"] == "reuse-k1"
        ]
        scores = []
        score_path = args.out / f"outer{outer:02d}-fresh-scores.jsonl"
        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
            ) as workers,
            score_path.open("x") as stream,
        ):
            for score in workers.map(grade, [("fresh-k1", row) for row in fresh["rows"]], chunksize=1):
                stream.write(json.dumps(score, ensure_ascii=False) + "\n")
                stream.flush()
                scores.append(score)
                if len(scores) % 128 == 0:
                    print("outer", outer, "scored", len(scores), "/1024", flush=True)
        assert len(scores) == 1024
        a, b = values(reuse["rows"], reuse_scores), values(matched, scores)
        delta = b - a
        indices = np.random.default_rng(20261102).integers(0, 64, size=(10000, 64))
        boot = delta[indices].mean(1)
        report["outers"][outer] = {
            "parent_optimizer_updates": (outer - 1) * 16,
            "reuse64": summarize(a),
            "fresh_common64": summarize(b),
            "fresh_full1024": summarize(values(fresh["rows"], scores)),
            "fresh_minus_reuse_common64": {
                name: {
                    "value": float(delta[:, i].mean()),
                    "question_bootstrap95": np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
                }
                for i, name in enumerate(["accuracy", "mean_length", "cap_rate"])
            },
            "identity": {
                "fresh_bank_sha256": fresh["bank_sha256"],
                "reuse_bank_sha256": reuse["bank_sha256"],
                "fresh_protocol_sha256": fresh["protocol_sha256"],
                "reuse_protocol_sha256": reuse["protocol_sha256"],
                "fresh_scores_sha256": digest(score_path),
                "reuse_scores_sha256": receipt["scores_sha256"],
            },
        }
        print(json.dumps(report["outers"][outer]), flush=True)
    report.update(
        status="complete",
        scoring_helper_sha256=digest(ROOT / "score_online_bank.py"),
        limits=[
            "Native automatic grading, including fallback false positives and internal timeout-to-zero behavior.",
            "The full1024 denominator differs from reuse64; compare the common64 only.",
            "Different outers use different questions; within-outer intervals omit seed uncertainty and multiplicity.",
            "These are new scheduled training questions before the next update, not the final benchmark evaluation.",
            "One training seed, one answer per question; early coverage screening only.",
        ],
    )
    (args.out / "complete.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
