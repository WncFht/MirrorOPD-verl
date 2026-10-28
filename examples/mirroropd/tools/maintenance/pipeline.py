# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Scoped maintenance entry; explicit formatting and read-only staged checks."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from examples.mirroropd.tools import check_docs  # noqa: E402
from examples.mirroropd.tools.maintenance import contracts, policy  # noqa: E402

TOOL = Path(__file__).resolve().parent


def run(command, *, root, input=None, timeout=180):
    """Use argument arrays and propagate failure, including missing dependencies."""
    try:
        result = subprocess.run(command, cwd=root, input=input, text=True, timeout=timeout, check=False)
        return result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Maintenance command failed: {error}", file=sys.stderr)
        return 2


def format_paths(root, paths, write=False):
    return run(
        ["node", str(TOOL / "format.cjs"), "--write" if write else "--check"],
        root=root,
        input=json.dumps([str(p) for p in paths]),
    )


def lint_paths(root, paths):
    executable = TOOL / "node_modules/markdownlint-cli2/markdownlint-cli2-bin.mjs"
    # Pass explicit inputs; never let an empty glob turn this into an empty success.
    return run(["node", str(executable), "--config", str(TOOL / "markdownlint.cjs"), *map(str, paths)], root=root)


def check(root, *, history=None):
    paths = policy.markdown_files(root)
    status = 0
    for action in (lambda: format_paths(root, paths), lambda: lint_paths(root, paths)):
        status = max(status, action())
    counts, errors = check_docs.check_pages(paths, root=root)
    rule_errors, warnings = contracts.check(root, history)
    for warning in warnings:
        print("REVIEW:", warning)
    for error in errors + rule_errors:
        print("ERROR:", error)
    print("Document coverage:", ", ".join(f"{key}={value}" for key, value in counts.items()))
    if counts.get("unavailable_evidence"):
        print("Referenced external/frozen artifacts unavailable here were not validated for content or existence.")
    return max(status, int(bool(errors or rule_errors)))


def staged(repo):
    """Check the index without stashing, formatting, or changing its entries."""
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="mirroropd-index-") as temp:
        snapshot = Path(temp)
        subprocess.run(
            ["git", "checkout-index", "--all", "--prefix", str(snapshot) + os.sep], cwd=repo, env=env, check=True
        )
        return check(snapshot, history=repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("format", "check", "list", "test"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.staged:
            if args.action != "check" or args.paths:
                parser.error("--staged only accepts a complete read-only check")
            return staged(root)
        if args.action == "check":
            if args.paths:
                parser.error("check covers all maintained documents")
            return check(root, history=root)
        if args.action == "test":
            return run([sys.executable, "-m", "unittest", "discover", "-s", str(TOOL / "tests"), "-v"], root=root)
        paths = policy.markdown_files(root, args.paths)
        if args.action == "list":
            print("\n".join(str(p.relative_to(root)) for p in paths))
            return 0
        return format_paths(root, paths, write=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Maintenance failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
