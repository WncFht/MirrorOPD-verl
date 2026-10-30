# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Combine already-defined jobs under one lease while preserving each frozen plan and source."""

import argparse
import re
from pathlib import Path

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.fixed_transfer.runner import ExecutionContext, run, validate_graph


def jobs(plan):
    spec = read(plan)
    result = []
    for entry in spec["entries"]:
        if not re.fullmatch(r"[a-z0-9_-]+", entry["name"]):
            raise ValueError("Unsafe batch entry name")
        prefix = entry["name"] + ":"
        child_plan = Path(plan).parent / (entry["name"] + "-plan.json")
        dump(child_plan, entry["plan"])
        source = entry.get("source")
        if source:
            for item in entry["source_files"]:
                if file_hash(Path(source) / item["path"]) != item["sha256"]:
                    raise ValueError("Pinned worker source changed")
        for original in entry["jobs"]:
            job = dict(
                original,
                name=prefix + original["name"],
                plan=str(child_plan),
                dependencies=[prefix + name for name in original["dependencies"]],
            )
            if job.get("producer"):
                job["producer"] = prefix + job["producer"]
            if source:
                job["source"] = source
            result.append(job)
    validate_graph(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wait-resource", action="store_true")
    args = parser.parse_args()
    spec = read(args.plan)
    if not spec["gpus"] or not set(spec["gpus"]) <= set(range(6)) or len(set(spec["gpus"])) != len(spec["gpus"]):
        raise ValueError("Only distinct physical GPUs within 0-5 are allowed")
    if len({e["name"] for e in spec["entries"]}) != len(spec["entries"]):
        raise ValueError("Batch entry names must be unique")
    context = ExecutionContext(spec["name"], spec["output"], tuple(spec["gpus"]), fingerprint(spec))
    run(args.plan, "batch", wait_resource=args.wait_resource, prepared_study=context, custom_job_builder=jobs)


if __name__ == "__main__":
    main()
