# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Source and workspace locations, independent of a study's package depth."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MIRROROPD = REPO / "examples/mirroropd"


def workspace():
    """Frozen workers keep data and resources in the original workspace."""
    return Path(os.environ.get("MIRROROPD_WORKSPACE", REPO.parent)).resolve()
