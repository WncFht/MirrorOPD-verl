import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
r = parser.parse_args().run
dest = Path(__file__).resolve().parent
dest.mkdir(exist_ok=True)
manifest = json.loads((r / "offline-manifest.json").read_text())["jobs"]
jobs = {j["job_id"]: j for j in manifest}
bybase = defaultdict(list)
for j in manifest:
    bybase[j["base"]].append(j)
rows = []
identity = hashlib.sha256()
count = 0
for p in sorted((r / "scan-v1").glob("shard-*/*.json")):
    if len(p.stem) != 64:
        continue
    raw = p.read_bytes()
    identity.update(hashlib.sha256(raw).digest())
    d = json.loads(raw)
    count += 1
    layer = int(re.search(r"layers\.(\d+)", d["tensor"]).group(1))
    module = d["tensor"].split(".")[-2]
    for j in bybase[d["base"]]:
        stages = {"teacher": j["teacher_path"], "endpoint": j["endpoint"]}
        stages.update({k: v for k, v in j.get("gradient_stages", {}).items() if k not in ["initial", "endpoint"]})
        for stage, path in stages.items():
            if path not in d["targets"]:
                continue
            v = d["targets"][path]
            s = v.get("update_spectrum")
            if not s:
                continue
            row = dict(
                job_id=j["job_id"],
                family=j["family"],
                teacher=j["teacher"],
                layer=layer,
                module=module,
                stage=stage,
                shape=d["shape"],
                delta_sigma=v["delta_sigma"],
                rho_sigma=v["rho_sigma"],
                kappa=v["kappa_spec"],
                update_energy=v["update_energy"],
                base_energy=v["base_energy"],
                top16=s["top_energy"]["16"],
                top64=s["top_energy"]["64"],
            )
            if stage != "teacher":
                pair = next((x for x in d["pairs"] if x["job_id"] == j["job_id"] and x["stage"] == stage), None)
                if pair:
                    row["missing_energy"] = pair["missing_spectrum"]["frobenius"] ** 2
            rows.append(row)
models = []
with (dest.parent / "synthesis/model-absorption.csv").open() as f:
    for x in csv.DictReader(f):
        n = int(x["job_id"])
        j = jobs[n]
        v = [a for a in rows if a["job_id"] == n and a["stage"] == "teacher"]
        s = [a for a in rows if a["job_id"] == n and a["stage"] == "endpoint"]
        et = sum(a["update_energy"] for a in v)
        es = sum(a["update_energy"] for a in s)
        em = sum(a["missing_energy"] for a in s)
        models.append(
            dict(
                job_id=n,
                family=x["family"],
                teacher=x["teacher"],
                cohort=x["cohort"],
                steps=int(x["step"]),
                initial_rkl=float(x["base"]),
                endpoint_rkl=float(x["endpoint"]),
                remaining=float(x["endpoint"]) / float(x["base"]),
                teacher_relative_update=math.sqrt(et / sum(a["base_energy"] for a in v)),
                teacher_top16=statistics.median(a["top16"] for a in v),
                student_top16=statistics.median(a["top16"] for a in s),
                teacher_rho=statistics.median(a["rho_sigma"] for a in v),
                student_rho=statistics.median(a["rho_sigma"] for a in s),
                teacher_kappa=statistics.median(a["kappa"] for a in v),
                student_kappa=statistics.median(a["kappa"] for a in s),
                parameter_residual_energy=em / et,
                update_cosine=(et + es - em) / (2 * math.sqrt(et * es)),
            )
        )
summary = json.loads(gzip.decompress((dest.parent / "iso-20261111/summary.json.gz").read_bytes()))
geometry = [
    x
    for x in summary["geometry_by_layer"]
    if x["rank_fraction"] == 0.9 and x["transition"] in ["base-to-teacher", "base-to-endpoint"]
]
reverse = json.loads((r / "init-frames-versus-initial-analysis.json").read_text())
result = dict(
    protocol={
        "scope": (
            "Full-layer seven attention/MLP projections; one matrix is one observation. "
            "Layer plots use median over 7 modules; initial/teacher prefixes are separate."
        ),
        "groups": (
            "Exploratory split at observed endpoint/initial RKL gap (<=.10 versus >=.20); "
            "describes observed panel, not complete absorption or prospective threshold."
        ),
        "scan_records": count,
        "scan_content_digest": identity.hexdigest(),
        "risk_scope": "model-absorption.csv: initial-student prefix, heldout Math/Code/IF 12 each, H128",
    },
    models=models,
    matrices=rows,
    geometry=geometry,
    reverse_frames=reverse["rows"],
)
(dest / "layer-data.json.gz").write_bytes(
    gzip.compress(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0)
)
print("records", count, "matrix stage rows", len(rows))
for m in models:
    print(
        m["job_id"],
        m["teacher"],
        "remaining",
        round(m["remaining"], 3),
        "residual",
        round(m["parameter_residual_energy"], 3),
        "cos",
        round(m["update_cosine"], 3),
    )
