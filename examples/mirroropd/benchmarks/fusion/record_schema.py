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


def checkpoint_root(run, config=None):
    """Read the saved launch destination instead of assuming outputs/checkpoints."""
    from pathlib import Path

    import yaml

    run = Path(run).resolve()
    config = config if config is not None else yaml.safe_load((run / "resolved_config.yaml").read_text())
    root = Path(config["trainer"]["default_local_dir"]).expanduser()
    return root if root.is_absolute() else (run.parent.parent / root).resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read immutable launch artifact locations")
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()
    print(checkpoint_root(args.checkpoint_root))
