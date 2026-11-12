"""CPU-only module/depth and angle aggregation from completed frozen evidence.

Model conditions are the unit for cross-condition correlations. Seven modules
per layer are dependent observations. No statistical significance is inferred.
Full-weight top-k angles and update-matrix top-k energy are distinct objects.
"""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
args = parser.parse_args()
data = json.loads(gzip.decompress((ROOT / "layer-data.json.gz").read_bytes()))
groups = defaultdict(list)
for row in data["matrices"]:
    n = max(x["layer"] for x in data["matrices"] if x["job_id"] == row["job_id"]) + 1
    band = min(2, row["layer"] * 3 // n)
    kind = "attention" if row["module"] in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
    groups[(row["job_id"], row["stage"], kind, band)].append(row)
modules = []
for (job, stage, kind, band), rows in sorted(groups.items()):
    modules.append(
        dict(
            job_id=job,
            stage=stage,
            kind=kind,
            depth_third=band,
            matrices=len(rows),
            top16=median(x["top16"] for x in rows),
            rho=median(x["rho_sigma"] for x in rows),
            kappa=median(x["kappa"] for x in rows),
            update_norm=math.sqrt(sum(x["update_energy"] for x in rows)),
            relative_update=math.sqrt(sum(x["update_energy"] for x in rows) / sum(x["base_energy"] for x in rows)),
        )
    )
angles = []
for directory in sorted((args.run / "offline-v2/tasks").glob("geometry-*/attempt-001/result")):
    for path in sorted(directory.glob("0*.json")):
        record = json.loads(path.read_text())
        layer = int(re.search(r"layers\.(\d+)", record["tensor"]).group(1))
        module = record["tensor"].split(".")[-2]
        for row in record["rows"]:
            for rank, values in row["subspaces"].items():
                for side in ["left", "right"]:
                    theta = [math.degrees(math.acos(min(1, max(0, v)))) for v in values[side + "_cosines"]]
                    angles.append(
                        dict(
                            job_id=row["job_id"],
                            transition=row["transition"],
                            tensor=record["tensor"],
                            layer=layer,
                            module=module,
                            rank=int(rank),
                            side=side,
                            mean_angle_deg=sum(theta) / len(theta),
                            max_angle_deg=max(theta),
                            overlap=values[side + "_overlap"],
                        )
                    )
angle_groups = defaultdict(list)
for a in angles:
    angle_groups[(a["job_id"], a["transition"], a["rank"], a["side"])].append(a)
angle_summary = [
    dict(
        job_id=k[0],
        transition=k[1],
        rank=k[2],
        side=k[3],
        matrices=len(v),
        median_mean_angle_deg=median(x["mean_angle_deg"] for x in v),
        median_max_angle_deg=median(x["max_angle_deg"] for x in v),
        median_overlap=median(x["overlap"] for x in v),
    )
    for k, v in sorted(angle_groups.items())
]
models = []
for m in data["models"]:
    rows = [x for x in data["matrices"] if x["job_id"] == m["job_id"]]
    et = sum(x["update_energy"] for x in rows if x["stage"] == "teacher")
    es = sum(x["update_energy"] for x in rows if x["stage"] == "endpoint")
    models.append(
        dict(
            m,
            teacher_update_norm=math.sqrt(et),
            student_update_norm=math.sqrt(es),
            student_teacher_norm_ratio=math.sqrt(es / et),
            absolute_top16_energy_ratio=(
                sum(x["top16"] * x["update_energy"] for x in rows if x["stage"] == "endpoint")
                / sum(x["top16"] * x["update_energy"] for x in rows if x["stage"] == "teacher")
            ),
        )
    )


def rank(values):
    vals = np.array(values)
    return np.array([(np.sum(vals < v) + (np.sum(vals == v) - 1) / 2) for v in vals])


correlations = []
for ids, label in [
    (None, "all16"),
    ([m["job_id"] for m in models if m["cohort"] == "continuous" and m["steps"] == 320], "continuous320"),
]:
    rows = [m for m in models if ids is None or m["job_id"] in ids]
    for feature in ["teacher_relative_update", "teacher_kappa", "teacher_top16", "teacher_rho", "initial_rkl"]:
        for target in ["endpoint_rkl", "remaining"]:
            corr = float(np.corrcoef(rank([x[feature] for x in rows]), rank([x[target] for x in rows]))[0, 1])
            correlations.append(dict(cohort=label, n=len(rows), feature=feature, target=target, spearman=corr))
result = dict(
    protocol={
        "depth": (
            "Three contiguous depth thirds by layer index; unweighted matrix median. "
            "Attention has 4 and MLP has 3 matrices per layer."
        ),
        "angle": (
            "Full weight W top16/top64 principal angles, singular values of U0.T@U1 or V0.T@V1, degrees. "
            "Median of matrix-level mean/max, not maximum across all layers."
        ),
        "correlations": (
            "Exploratory Spearman correlations; ties averaged; repeated teachers/families dependent. "
            "continuous320 excludes 64-step conditions and MiMo branch, retains shared Samsung teacher."
        ),
    },
    models=models,
    modules=modules,
    angles=angles,
    angle_summary=angle_summary,
    correlations=correlations,
)
(ROOT / "module-angle-analysis.json.gz").write_bytes(
    gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0)
)
print("models", len(models), "angles", len(angles))
for c in correlations:
    print(c)
for n in [2, 7, 8, 11, 13, 19, 20]:
    print(
        "MODULES",
        n,
        [
            (x["stage"], x["kind"], x["depth_third"], round(x["top16"] * 100, 1))
            for x in modules
            if x["job_id"] == n and x["stage"] in ["teacher", "endpoint"]
        ],
    )
    print(
        "ANGLES",
        n,
        [
            (x["transition"], x["side"], round(x["median_mean_angle_deg"], 3), round(x["median_max_angle_deg"], 3))
            for x in angle_summary
            if x["job_id"] == n and x["rank"] == 16 and x["transition"] in ["base-to-teacher", "base-to-endpoint"]
        ],
    )
