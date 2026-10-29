# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute workers from a real source snapshot, keeping workspace data external."""

import os
import shutil
from pathlib import Path


def freeze(repo, destination):
    """Copy research and its production dependencies, never datasets or model weights."""
    repo, destination = Path(repo), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(repo / "verl", destination / "verl", ignore=ignore)
    target = destination / "examples/mirroropd"
    target.mkdir(parents=True)
    for relative in ("examples/__init__.py", "examples/mirroropd/__init__.py"):
        source = repo / relative
        if source.exists():
            shutil.copy2(source, destination / relative)
    shutil.copytree(repo / "examples/mirroropd/research", target / "research", ignore=ignore)
    return destination


def environment(source, workspace):
    """Ensure child imports and external resource paths refer to different explicit roots."""
    return dict(
        os.environ,
        PYTHONPATH=str(Path(source).resolve()),
        MIRROROPD_WORKSPACE=str(Path(workspace).resolve()),
        PYTHONDONTWRITEBYTECODE="1",
    )
