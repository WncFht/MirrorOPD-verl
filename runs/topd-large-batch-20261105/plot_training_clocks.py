# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot training reward against outer, optimizer-update, and sample-update clocks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
E57_DATA = ROOT.parent / "e55-e57-refresh-20261102-0900" / "data.json"
E81_METRICS = ROOT / "formal" / "E81" / "metrics.jsonl"
OUT = ROOT / "figures"


def load_e57() -> dict[str, list[dict[str, float]]]:
    payload = json.loads(E57_DATA.read_text())
    result = {}
    for run in payload["runs"]:
        if run["name"] != "E57":
            continue
        result["outer"] = [
            {
                "outer": float(row["step"]),
                "score": float(row["score"]),
                "updates": float(row["step"]) * 16,
                "sample_updates": float(row["step"]) * 64 * 16,
            }
            for row in run["outer"]
        ]
    return result


def load_e81() -> list[dict[str, float]]:
    rows = []
    for line in E81_METRICS.read_text().splitlines():
        record = json.loads(line)
        data = record.get("data", {})
        if "step/outer" not in data or "rollout/score/mean" not in data:
            continue
        outer = float(data["step/outer"])
        if rows and rows[-1]["outer"] == outer:
            rows[-1] = {
                "outer": outer,
                "score": float(data["rollout/score/mean"]),
                "updates": float(data["step/optimizer"]),
                "sample_updates": outer * 256 * 16,
            }
        else:
            rows.append(
                {
                    "outer": outer,
                    "score": float(data["rollout/score/mean"]),
                    "updates": float(data["step/optimizer"]),
                    "sample_updates": outer * 256 * 16,
                }
            )
    return rows


def plot(series: dict[str, list[dict[str, float]]], key: str, label: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name, rows in series.items():
        if not rows:
            continue
        ax.plot([row[key] for row in rows], [row["score"] for row in rows], label=name, linewidth=2)
    ax.set_xlabel(label)
    ax.set_ylabel("training rollout reward / score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / f"{filename}.png", dpi=180)
    fig.savefig(OUT / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = {"E57 original (64 responses/outer)": load_e57()["outer"], "E81 current (256 responses/outer)": load_e81()}
    plot(series, "outer", "outer step", "reward-by-outer")
    plot(series, "updates", "optimizer update", "reward-by-optimizer-update")
    plot(series, "sample_updates", "cumulative response samples × optimizer updates", "reward-by-sample-update")
    (OUT / "README.md").write_text(
        "# Training clock comparison\n\n"
        "The plots compare the local E57 original run with the currently running E81 run. "
        "E57 uses 64 responses per outer and E81 uses 256; both use 16 optimizer updates per outer. "
        "The local TOP-D paper snapshot contains no machine-readable per-step coordinates, so no "
        "digitized TOP-D curve is drawn. Add audited coordinates before using a TOP-D overlay.\n"
    )


if __name__ == "__main__":
    main()
