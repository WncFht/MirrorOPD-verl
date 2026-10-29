# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze native-layout error envelopes, then evaluate held-out fusion replays."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def measure(reference, candidate, legal, selected):
    reference = reference.double()[legal].log_softmax(-1)
    candidate = candidate.double()[legal].log_softmax(-1)
    difference = (reference.exp() - candidate.exp()).abs()
    selected_index = (legal == selected).nonzero().item()
    return {
        "probability_abs": difference.max().item(),
        "total_variation": difference.sum().item() / 2,
        "selected_logprob_abs": abs(reference[selected_index].item() - candidate[selected_index].item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--component", choices=["student", "teacher"], required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--limits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    legal = torch.tensor(sorted(set(tokenizer.get_vocab().values())), dtype=torch.long)
    workload_path = Path(args.workload)
    workload = json.loads(workload_path.read_text())
    workload_hash = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    checks = []
    for index, case in enumerate(workload):
        reference = torch.load(Path(args.reference) / f"case-{index}.pt", map_location="cpu", weights_only=True)
        candidate = torch.load(Path(args.candidate) / f"case-{index}.pt", map_location="cpu", weights_only=True)
        if reference.keys() != candidate.keys() or set(reference) != set(case["checkpoints"]):
            raise ValueError("Replay checkpoint coverage differs")
        for step in sorted(reference):
            left = reference[step]["raw"]
            right = candidate[step]["raw" if args.calibrate else args.component]
            if left.shape[0] != right.shape[0] and not (args.calibrate and left.shape[0] == 1):
                raise ValueError("Reference and candidate request coverage differs")
            for row in range(right.shape[0]):
                reference_row = 0 if args.calibrate and left.shape[0] == 1 else row
                checks.append(
                    {
                        "case": index,
                        "step": step,
                        "row": row,
                        **measure(left[reference_row], right[row], legal, case["suffix_ids"][step]),
                    }
                )
    if not checks:
        raise ValueError("Empty numerical comparison")
    metrics = ["probability_abs", "total_variation", "selected_logprob_abs"]
    maxima = {key: max(row[key] for row in checks) for key in metrics}
    limits_path = Path(args.limits)
    if args.calibrate:
        limits = {
            "component": args.component,
            "calibration_workload_sha256": workload_hash,
            "rule": "observed native-layout maximum plus 2e-6 numerical floor",
            "limits": {key: value + 2e-6 for key, value in maxima.items()},
        }
        with limits_path.open("x") as stream:
            json.dump(limits, stream, indent=2)
    else:
        limits = json.loads(limits_path.read_text())
        if limits["component"] != args.component or limits["calibration_workload_sha256"] == workload_hash:
            raise ValueError("Acceptance needs the matching component and a held-out workload")
    failures = [row for row in checks if any(row[key] > limits["limits"][key] for key in metrics)]
    result = {
        "config": vars(args),
        "checks": len(checks),
        "maxima": maxima,
        "limits": limits,
        "failures": failures,
        "passed": not failures,
    }
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
    if failures:
        raise RuntimeError("Native component comparison exceeded the frozen error envelope")


if __name__ == "__main__":
    main()
