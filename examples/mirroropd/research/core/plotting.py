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


def save_figure(fig, path, caption, caption_size=8):
    import matplotlib.pyplot as plt

    fv = load_style()
    fig.text(0.5, 0.025, caption, ha="center", fontsize=caption_size, color=fv.NEUTRALS["baseline"])
    fig.canvas.draw()
    problems = fv.audit(fig, print_slot=fig.get_figwidth())
    for p in fv.format_audit_lines(problems):
        print(path.name + ": " + p)
    path.with_suffix(".audit.txt").write_text(
        "\n".join(fv.format_audit_lines(problems)) + "\n" if problems else "PASS: fv.audit(fig) returned no issues.\n"
    )
    if any(fv.problem_severity(p) == "error" for p in problems):
        raise RuntimeError("Figure audit failed")
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=180, facecolor=fv.NEUTRALS["paper"])
    plt.close(fig)
