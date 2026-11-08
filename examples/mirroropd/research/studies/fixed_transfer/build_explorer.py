# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Build an offline Plotly explorer from the verified student-paths CSV exports.

Semantic ledger: color identifies a teacher within a student; circle/diamond
identifies same/cross size. All connecting lines are solid chronological paths.
Snapshots from one run are dependent; the explorer estimates no correlation.
Geometry ledger: linear absorption/decrease axes and log RKL axes; domains share
numeric ranges per view. Saved budgets stop at actual endpoints, never extend.
"""

import hashlib
import json
import re
from pathlib import Path

from examples.mirroropd.research.studies.fixed_transfer.explorer_schema import (
    METRICS,
    STATUSES,
    SYMBOLS,
    VIEWS,
    presentation,
)
from examples.mirroropd.research.studies.fixed_transfer.plot_correlations import PANELS, STUDENTS
from examples.mirroropd.research.studies.fixed_transfer.quantify_absorption import MIN_PROGRESS, diagnose, load_runs

BASE = Path(__file__).resolve().parents[3] / "docs/experiments/teacher-transfer/fixed-state-results"
DATA = BASE / "figures/student-paths"


def main():
    from plotly.offline import get_plotlyjs

    runs = {r["run_id"]: r for r in presentation(load_runs())}
    row_count = sum(len(points) for r in runs.values() for points in r["panels"].values())
    for run in runs.values():
        run["diagnostics"] = {
            panel: [diagnose(run, panel, p["step"]) for p in run["panels"][panel]] for panel in PANELS[1:]
        }
    payload = dict(
        students=STUDENTS,
        runs=list(runs.values()),
        panels=PANELS,
        min_progress=MIN_PROGRESS,
        metrics=METRICS,
        views=VIEWS,
        statuses=STATUSES,
        symbols=SYMBOLS,
    )
    template = Path(__file__).with_name("explorer.html").read_text()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template, replaced = re.subn(r"/\*__DATA__\*/\s*null", lambda _: encoded, template)
    if replaced != 1:
        raise ValueError("Expected exactly one data placeholder")
    script = Path(__file__).with_name("explorer.js").read_text()
    style = Path(__file__).with_name("explorer.css").read_text()
    html = (
        template.replace("/*__STYLE__*/", style)
        .replace("/*__PLOTLY__*/", get_plotlyjs())
        .replace("/*__APP__*/", script)
    )
    target = BASE / "explorer.html"
    target.write_text(html)
    receipt = dict(
        runs=len(runs),
        rows=row_count,
        source_hashes={
            name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
            for name in ("inventory.csv", "snapshots.csv", "provenance.json")
        },
        app_source_hashes={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("explorer.html", "explorer.css", "explorer.js", "explorer_schema.py", "quantify_absorption.py")
        },
        html_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        offline=True,
    )
    (BASE / "figures/student-paths/explorer-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
