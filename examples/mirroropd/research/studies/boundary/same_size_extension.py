# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run legacy-state R1 continuations through the shared resource and dependency scheduler."""

import argparse
from pathlib import Path

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.io import read
from examples.mirroropd.research.studies.boundary.fast_train import extension_plan
from examples.mirroropd.research.studies.fixed_transfer.runner import ExecutionContext, run


def jobs(plan):
    spec = read(plan)
    return [
        dict(
            name="train-" + arm,
            args=[arm, "--engine", "validated"],
            module="examples.mirroropd.research.studies.boundary.fast_train",
            marker=str(Path(spec["output"]) / "arms" / arm / "complete.json"),
            dependencies=[],
            retries=0,
            expected_inputs=dict(plan_identity=fingerprint(spec)),
        )
        for arm in spec["arms"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wait-resource", action="store_true")
    args = parser.parse_args()
    spec = read(args.plan)
    if len(spec["gpus"]) != 2 or len(set(spec["gpus"])) != 2 or not set(spec["gpus"]) <= set(range(6)):
        raise ValueError("Allocate two distinct physical GPUs within 0-5")
    for arm in spec["arms"]:
        extension_plan(args.plan, arm)
    context = ExecutionContext(spec["name"], spec["output"], tuple(spec["gpus"]), fingerprint(spec))
    run(args.plan, "r1-extension", wait_resource=args.wait_resource, prepared_study=context, custom_job_builder=jobs)


if __name__ == "__main__":
    main()
