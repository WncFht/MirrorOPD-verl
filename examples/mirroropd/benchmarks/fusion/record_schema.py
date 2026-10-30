# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read frozen experiment records; this module is never used to launch training."""


def prefix_policy_facts(policy):
    """Compare scientific meanings across recorded schemas without rewriting either."""
    mode = policy["mode"]
    if "right_weight" in policy:
        if mode == "single":
            return mode, (policy.get("model") or {"kind": "current"})["kind"], None, policy["right_weight"]
        return (
            mode,
            (policy.get("left") or {"kind": "current"})["kind"],
            (policy.get("right") or {"kind": "teacher"})["kind"],
            policy["right_weight"],
        )
    if mode == "single":
        kind = policy["student"] if policy["source"] == "student" else policy["source"]
        return mode, kind, None, policy["teacher_weight"]
    return mode, policy["student"], "teacher", policy["teacher_weight"]
