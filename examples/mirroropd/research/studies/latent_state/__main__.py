# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit stages; no generation API and no implicit GPU allocation."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--catalog", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--pairs", nargs="+", required=True)
    prep.add_argument("--per-outer", type=int, default=64)
    prep.add_argument("--seed", type=int, default=42)
    for name in ("extract", "analyze"):
        stage = commands.add_parser(name)
        stage.add_argument("--plan", type=Path, required=True)
        stage.add_argument("--pair", required=True)
        if name == "extract":
            stage.add_argument("--device", default="cuda:0")
            stage.add_argument("--token-budget", type=int, default=65536)
            stage.add_argument("--max-batch", type=int, default=8)
    report = commands.add_parser("report")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    site = commands.add_parser("serve")
    site.add_argument("--host", default="127.0.0.1")
    site.add_argument("--port", type=int, default=8766)
    site.add_argument("--report", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        from .data import prepare

        prepare(**args)
    elif command == "extract":
        from .extract import extract

        extract(args.pop("plan"), **args)
    elif command == "analyze":
        from .analysis import analyze

        analyze(args.pop("plan"), **args)
    elif command == "serve":
        from .serve import serve

        if args["report"] is None:
            args.pop("report")
        serve(**args)
    else:
        from .report import report

        report(args.pop("plan"), **args)


if __name__ == "__main__":
    main()
