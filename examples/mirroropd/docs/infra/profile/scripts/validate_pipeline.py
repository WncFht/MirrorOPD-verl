# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Require fixed-prefix loss and preclip gradient norm agreement across backends."""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = [json.loads(f.read_text()) for f in (args.baseline, args.candidate)]
    assert a["input_tokens"] == b["input_tokens"] and a["loss_tokens"] == b["loss_tokens"]
    checks = []
    for x, y in zip(a["records"], b["records"], strict=True):
        for key in ("loss", "grad_norm"):
            assert x[key] is not None and y[key] is not None
            assert math.isfinite(x[key]) and math.isfinite(y[key])
            delta = abs(x[key] - y[key])
            tolerance = 2e-4 + 0.02 * abs(x[key])
            checks.append({"step": x["step"], "key": key, "absolute_delta": delta, "tolerance": tolerance})
            assert delta <= tolerance, checks[-1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
    main()
