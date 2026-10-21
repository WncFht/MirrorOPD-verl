#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay four hand-written LCB references through the production evaluator."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from benchmark_isolation import ROOT, real_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cgroup-parent")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from verl.utils.reward_score.one_shot_opd.code_eval_v2 import _lcb_eval_single

    cases, stats = real_cases(args.dataset)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lcb-reference-slots-") as slots:
        for key in list(os.environ):
            if key.startswith("MIRROROPD_LCB_"):
                del os.environ[key]
        os.environ.update(
            MIRROROPD_LCB_SLOT_DIR=slots, MIRROROPD_LCB_MEMORY_MIB="256", MIRROROPD_LCB_ADDRESS_SPACE_MIB="1024"
        )
        for backend in ["rss", *(["cgroup"] if args.cgroup_parent else [])]:
            os.environ["MIRROROPD_LCB_BACKEND"] = backend
            if backend == "cgroup":
                os.environ["MIRROROPD_LCB_CGROUP_PARENT"] = args.cgroup_parent
            for case in cases:
                start = time.monotonic()
                passed, detail = _lcb_eval_single(
                    (case["sample"], case["code"], 6, {"index": case["name"], "run_id": "implementation-replay"})
                )
                rows.append(
                    dict(
                        case=case["name"],
                        backend=backend,
                        passed=passed,
                        end_to_end_s=time.monotonic() - start,
                        detail=detail,
                    )
                )
                assert passed, (case["name"], backend, detail)
    args.out.write_text(
        json.dumps(
            {
                "scope": "Hand-written reference programs; production path; no model inference",
                "dataset": stats,
                "results": rows,
            },
            indent=2,
        )
    )
    print(f"{len(rows)} reference-program executions passed; wrote {args.out}")


if __name__ == "__main__":
    main()
