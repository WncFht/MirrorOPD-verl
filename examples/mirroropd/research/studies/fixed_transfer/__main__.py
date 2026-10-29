# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Single CLI for model preparation, banks, evaluations and guarded study execution."""

import argparse
from pathlib import Path

from examples.mirroropd.research.studies.fixed_transfer.spec import Study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "analyze",
            "assets",
            "prepare",
            "bank",
            "baseline",
            "cache",
            "validate",
            "train",
            "endpoint",
            "copy",
            "run",
            "guard",
        ],
    )
    parser.add_argument("--partial", action="store_true", help="Analyze only completed evidence")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--arm")
    parser.add_argument("--step", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--phase", choices=["baselines", "full"], default="baselines")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action == "guard":
        from examples.mirroropd.research.runtime.resources import guard

        guard(args.directory)
        return
    if args.plan is None:
        parser.error("--plan is required")
    study = Study.load(args.plan)
    if args.action == "analyze":
        from examples.mirroropd.research.studies.fixed_transfer.analysis import analyze

        analyze(study, partial=args.partial)
    elif args.action == "assets":
        from examples.mirroropd.research.studies.fixed_transfer.models import assets, audit

        assets(study)
        audit(study)
    elif args.action == "prepare":
        from examples.mirroropd.research.studies.fixed_transfer.data import prepare

        prepare(study)
    elif args.action == "bank":
        from examples.mirroropd.research.runtime.cuda import setup
        from examples.mirroropd.research.studies.fixed_transfer.data import generate

        setup()
        generate(study, args.shard)
    elif args.action == "baseline":
        from examples.mirroropd.research.studies.fixed_transfer.evaluation import evaluate

        evaluate(study, args.model)
    elif args.action in ["cache", "validate", "train", "endpoint", "copy"]:
        from examples.mirroropd.research.runtime.cuda import setup

        setup()
        if args.action == "cache":
            from examples.mirroropd.research.studies.fixed_transfer.training import cache_teacher

            cache_teacher(study, args.model)
        elif args.action == "validate":
            from examples.mirroropd.research.studies.fixed_transfer.validation import validate

            validate(study, args.model)
        elif args.action == "train":
            from examples.mirroropd.research.studies.fixed_transfer.training import train

            train(study, args.arm)
        elif args.action == "endpoint":
            from examples.mirroropd.research.studies.fixed_transfer.training import evaluate_endpoint

            evaluate_endpoint(study, args.arm, args.step)
        else:
            from examples.mirroropd.research.studies.fixed_transfer.validation import teacher_copy

            teacher_copy(study, args.model)
    else:
        from examples.mirroropd.research.studies.fixed_transfer.runner import run

        run(args.plan, args.phase)


if __name__ == "__main__":
    main()
