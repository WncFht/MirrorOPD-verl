# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""List or run research modules without importing training backends for discovery."""

import argparse
import runpy
import sys
from pathlib import Path


def commands(family):
    directory = Path(__file__).parent / family
    return sorted(p.stem for p in directory.glob("*.py") if "if __name__ ==" in p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("transfer", "precision"))
    parser.add_argument("command", nargs="?", help="Module name; omit to list available experiments")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    available = commands(args.family)
    if args.command is None:
        print("\n".join(available))
        return
    if args.command not in available:
        parser.error(f"Unknown executable module {args.command!r}; omit command to list choices")
    module = f"examples.mirroropd.research.{args.family}.{args.command}"
    previous = sys.argv
    try:
        sys.argv = [module, *args.arguments]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
