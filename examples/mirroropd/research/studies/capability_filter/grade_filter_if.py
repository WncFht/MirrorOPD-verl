# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only grading entrypoint, importable without torch or verl initialization."""

import argparse
import importlib.util
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    a = ap.parse_args()
    from langdetect import DetectorFactory

    spec = importlib.util.spec_from_file_location(
        "filter_if_scorer", a.repo / "verl/utils/reward_score/one_shot_opd/if_eval.py"
    )
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    DetectorFactory.seed = 0
    rows = json.loads(a.input.read_text())
    assert len(rows) == 64
    for row in rows:
        random.seed(row["id"])
        row["grade"] = scorer.reward_func("MultiIF", row["clean_output"], row["ground_truth"])
    temporary = a.input.with_suffix(".grading.tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(a.input)
    print(a.input.stem, sum(r["grade"]["acc"] for r in rows), "/ 64", flush=True)


if __name__ == "__main__":
    main()
