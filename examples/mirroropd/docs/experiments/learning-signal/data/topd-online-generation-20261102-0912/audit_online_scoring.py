"""Repeat native scoring and compare the frozen official wrapper on outer4."""

import concurrent.futures
import importlib.util
import json
import multiprocessing
from pathlib import Path

from score_online_bank import ROOT, digest, grade, initialize, load_bank


def check(item):
    arm, row = item
    native = grade(item)
    spec = importlib.util.spec_from_file_location("official_reward", ROOT / "official-data/math_verify_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return native | {"official_result": module.compute_score("DAPO", row["output"], row["ground_truth"])}


def main():
    out = ROOT / "online-outer4-scores-20261102"
    assert not (out / "repeat-scoring.json").exists()
    original = [json.loads(line) for line in (out / "scores.jsonl").open()]
    indexed = {(s["arm"], s["request_id"]): s for s in original}
    inputs = [
        (arm, row)
        for arm in ["reuse-k0", "reuse-k05", "reuse-k1"]
        for row in load_bank(ROOT / "online" / arm / "outer04")["rows"]
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=initialize
    ) as workers:
        repeated = list(workers.map(check, inputs, chunksize=1))
    assert all(s["status"] == "scored" for s in repeated)
    differences = [s for s in repeated if s["result"] != indexed[(s["arm"], s["request_id"])]["result"]]
    report = {
        "native_repeat_differences": differences,
        "scores": repeated,
        "summary": {
            arm: {
                "native_correct": sum(bool(s["result"]["acc"]) for s in repeated if s["arm"] == arm),
                "official_correct": sum(bool(s["official_result"]["acc"]) for s in repeated if s["arm"] == arm),
                "native_boxed_correct": sum(
                    bool(s["result"]["acc"]) and s["result"]["format_score"] == 1 for s in repeated if s["arm"] == arm
                ),
            }
            for arm in ["reuse-k0", "reuse-k05", "reuse-k1"]
        },
        "script_sha256": digest(Path(__file__)),
        "official_source_sha256": digest(ROOT / "official-data/math_verify_reward.py"),
        "limits": "Both wrappers map internal timeouts to false; official uses a one-second limit.",
    }
    (out / "repeat-scoring.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]), "native repeat differences", len(differences), flush=True)


if __name__ == "__main__":
    main()
