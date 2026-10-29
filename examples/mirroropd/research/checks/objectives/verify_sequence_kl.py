#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Enumerate finite distributions: conditional improvement need not improve sequence KL."""

import argparse
import json
import math
from pathlib import Path


def kl(p, t):
    return sum(a * math.log(a / b) for a, b in zip(p, t, strict=True))


def entropy(p):
    return -sum(a * math.log(a) for a in p)


def mirror_step(p, t, eta):
    weights = [a ** (1 - eta) * b**eta for a, b in zip(p, t, strict=True)]
    return [w / sum(weights) for w in weights]


def sequence(p):
    return [p[0][i] * p[1 + i][j] for i in range(2) for j in range(2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = [[0.99, 0.01], [0.5, 0.5], [1 - 1e-6, 1e-6]]
    t = [[0.01, 0.99], [0.5, 0.5], [1e-30, 1 - 1e-30]]
    new = [mirror_step(a, b, 0.1) for a, b in zip(p, t, strict=True)]
    before = [kl(a, b) for a, b in zip(p, t, strict=True)]
    after = [kl(a, b) for a, b in zip(new, t, strict=True)]
    sequence_before, sequence_after = kl(sequence(p), sequence(t)), kl(sequence(new), sequence(t))
    assert all(a <= b for a, b in zip(after, before, strict=True))
    assert sequence_after > sequence_before
    for policy, conditionals, joint in ((p, before, sequence_before), (new, after, sequence_after)):
        assert (
            abs(conditionals[0] + sum(a * b for a, b in zip(policy[0], conditionals[1:], strict=True)) - joint) < 1e-12
        )
    a, b = [0.9, 0.1], [0.1, 0.9]
    assert entropy(a) == entropy(b) and kl(a, b) > 1
    result = {
        "chain_rule_counterexample": {
            "conditional_before": before,
            "conditional_after": after,
            "sequence_kl_before": sequence_before,
            "sequence_kl_after": sequence_after,
        },
        "same_entropy_different_distribution": {
            "entropy_p": entropy(a),
            "entropy_teacher": entropy(b),
            "kl": kl(a, b),
        },
        "scope": "Finite distributions, direct enumeration; no model measurement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: chain rule, conditional/sequence counterexample, equal-entropy counterexample")


if __name__ == "__main__":
    main()
