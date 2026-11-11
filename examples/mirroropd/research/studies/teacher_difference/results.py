# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read completed measurements and export the common scalar result table."""

import json
from pathlib import Path

from examples.mirroropd.research.core.io import write_csv as write_csv


def loss_signal(summary):
    """Derived fixed-objective gradient-flow rates; not measured Adam improvement."""
    energy = summary["P"] * summary["Xi"]
    loss = summary["loss_mean"]
    return dict(PXi=energy, PXi_over_L=energy / loss if loss > 0 else float("nan"))


def load_results(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for task in manifest["tasks"]:
        path = manifest_path.parent / "results" / task["id"] / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if not result.get("finished") or result.get("signature", {}).get("task") != task:
            raise ValueError(f"Incomplete or stale result: {path}")
        yield task, result, path


def summary_rows(task, result):
    for subset, summary in result["summaries"].items():
        yield dict(
            task=task["id"],
            model=task["model"],
            branch=task.get("branch", task["target"]),
            label=task.get("label", task.get("branch", task["target"])),
            teacher=task["target"],
            stage=task["stage"],
            bank=task["bank"],
            subset=subset,
            **{key: value for key, value in summary.items() if isinstance(value, float | int)},
            **loss_signal(summary),
            **{
                f"batch_{key}": value
                for key, value in summary.get("geometry", {}).items()
                if isinstance(value, float | int)
            },
        )
