# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""List or run research commands without importing model backends for discovery."""

import argparse
import ast
import runpy
import sys
from pathlib import Path


def commands():
    """Discover executable modules from source; no registry or backend imports."""
    root = Path(__file__).parent
    result = {}
    for folder in ("studies", "checks"):
        for path in sorted((root / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not any(
                isinstance(node, ast.If)
                and ast.unparse(node.test) in {"__name__ == '__main__'", "'__main__' == __name__"}
                for node in tree.body
            ):
                continue
            parts = path.relative_to(root).with_suffix("").parts
            module = ".".join(parts[:-1] if parts[-1] == "__main__" else parts)
            name = module.removeprefix("studies.")
            result[name] = f"examples.mirroropd.research.{module}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", help="Dotted command name; omit to list commands")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    available = commands()
    if args.command is None:
        print("\n".join(available))
        return
    if args.command not in available:
        parser.error(f"Unknown command {args.command!r}; omit command to list available commands")
    previous = sys.argv
    try:
        sys.argv = [available[args.command], *args.arguments]
        runpy.run_module(available[args.command], run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
