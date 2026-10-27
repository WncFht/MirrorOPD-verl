# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Load the external figure style explicitly, only when rendering is requested."""

import importlib
import os
import sys
from pathlib import Path


def load_style(path=None):
    value = path or os.environ.get("FIGURE_VIZ_MPL")
    if not value:
        raise ValueError("Set FIGURE_VIZ_MPL to the figure-viz assets/mpl directory")
    directory = Path(value).resolve()
    if not (directory / "fvstyle.py").exists() and not (directory / "fvstyle/__init__.py").exists():
        raise ValueError(f"No fvstyle module under {directory}")
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(directory))
        return importlib.import_module("fvstyle")
    finally:
        sys.path[:] = previous
