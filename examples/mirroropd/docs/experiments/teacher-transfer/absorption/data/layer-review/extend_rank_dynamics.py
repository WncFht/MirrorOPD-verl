"""CPU-only larger-rank summaries and exact FP64 checkpoint subspace dynamics.

The full-layer extension reuses stored complete update singular values; no new
GPU decomposition. Temporal recomputation is restricted to historical three
k_proj matrices per pair. SHA checks match original temporal evidence. Baseline
subspace overlap is k/ambient dimension, not zero; full-rank saturation is not
locking. All checkpoint weights are promoted before subtraction.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[7]
sys.path.insert(0, str(REPO))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scan-run", type=Path, required=True)
parser.add_argument("--temporal-run", type=Path, required=True)
args = parser.parse_args()
ranks = [8, 16, 32, 64, 128, 256, 512]
manifest = json.loads((args.scan_run / "offline-manifest.json").read_text())["jobs"]
bybase = {}
for job in manifest:
    bybase.setdefault(job["base"], []).append(job)
rows = []
for path in sorted((args.scan_run / "scan-v1").glob("shard-*/*.json")):
    if len(path.stem) != 64:
        continue
    data = json.loads(path.read_text())
    for job in bybase[data["base"]]:
        for stage, target in [("teacher", job["teacher_path"]), ("endpoint", job["endpoint"])]:
            record = data["targets"].get(target, {})
            spectrum = record.get("update_spectrum", {}).get("singular_values")
            if spectrum is None:
                continue
            cumulative = np.cumsum(np.square(np.array(spectrum, dtype=np.float64)))
            fractions = {str(k): float(cumulative[min(k, len(spectrum)) - 1] / cumulative[-1]) for k in ranks}
            rows.append(
                dict(
                    job_id=job["job_id"],
                    stage=stage,
                    tensor=data["tensor"],
                    shape=data["shape"],
                    energy=float(cumulative[-1]),
                    top_energy=fractions,
                    stable_rank=float(cumulative[-1] / spectrum[0] ** 2),
                    rank90=int(np.searchsorted(cumulative, 0.9 * cumulative[-1]) + 1),
                )
            )
summary = []
for job in manifest:
    for stage in ["teacher", "endpoint"]:
        group = [r for r in rows if r["job_id"] == job["job_id"] and r["stage"] == stage]
        if not group:
            continue
        total = sum(r["energy"] for r in group)
        summary.append(
            dict(
                job_id=job["job_id"],
                stage=stage,
                matrices=len(group),
                top_median={str(k): median(r["top_energy"][str(k)] for r in group) for k in ranks},
                top_pooled={str(k): sum(r["energy"] * r["top_energy"][str(k)] for r in group) / total for k in ranks},
                rank90_median=median(r["rank90"] for r in group),
                relative_rank90_median=median(r["rank90"] / min(r["shape"]) for r in group),
                stable_rank_median=median(r["stable_rank"] for r in group),
            )
        )
(ROOT / "extended-rank-analysis.json.gz").write_bytes(
    gzip.compress(
        json.dumps(dict(ranks=ranks, matrices=rows, summary=summary), separators=(",", ":")).encode(), mtime=0
    )
)
print("rank extension", len(rows), "rows", flush=True)

import torch  # noqa: E402

from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest  # noqa: E402

torch.set_num_threads(2)
configs = json.loads((args.temporal_run / "manifest.json").read_text())["configs"]
trajectories = []
for config in configs:
    if not config["same_coordinates"]:
        continue
    base = TensorCheckpoint(config["base"])
    for layer in [0, 13, 27]:
        key = f"model.layers.{layer}.self_attn.k_proj.weight"
        original = json.loads((args.temporal_run / "temporal-svd" / f"{config['label']}-l{layer:02d}.json").read_text())
        raw_base = base.get(key)
        assert tensor_digest(raw_base) == original["base_sha256"]
        b = raw_base.double()
        decompositions = []
        for step, path in config["stages"].items():
            raw = TensorCheckpoint(path).get(key)
            prior = next(x for x in original["stages"] if x["step"] == int(step))
            assert tensor_digest(raw) == prior["sha256"]
            delta = raw.double() - b
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            assert torch.allclose(
                s, torch.tensor(prior["update_singular_values"], dtype=torch.float64), rtol=1e-8, atol=1e-10
            )
            decompositions.append((int(step), delta, u, s, vh))
        for step, delta, u, s, vh in decompositions:
            for requested in ranks:
                k = min(requested, len(s))
                for side, dim in [("left", b.shape[0]), ("right", b.shape[1])]:
                    current = u[:, :k] if side == "left" else vh[:k].T
                    values = {}
                    for label, ref in [("early", decompositions[0]), ("final", decompositions[-1])]:
                        basis = ref[2][:, :k] if side == "left" else ref[4][:k].T
                        overlap = float((basis.T @ current).square().sum() / k)
                        projected = basis.T @ delta if side == "left" else delta @ basis
                        baseline = k / dim
                        values[label + "_overlap"] = overlap
                        values[label + "_adjusted_overlap"] = (
                            (overlap - baseline) / (1 - baseline) if baseline < 1 else None
                        )
                        values[label + "_capture"] = float(projected.square().sum() / delta.square().sum())
                    trajectories.append(
                        dict(
                            pair=config["label"],
                            layer=layer,
                            tensor=key,
                            step=step,
                            side=side,
                            requested_rank=requested,
                            effective_rank=k,
                            ambient_dimension=dim,
                            random_overlap=k / dim,
                            own_energy=float(s[:k].square().sum() / s.square().sum()),
                            **values,
                        )
                    )
        print("temporal", config["label"], layer, "verified", flush=True)
(ROOT / "extended-temporal-analysis.json").write_text(
    json.dumps(
        dict(
            protocol=(
                "CPU FP64 exact SVD; checkpoint SHA and stored singular values verified; "
                "k clipped to matrix min dimension."
            ),
            rows=trajectories,
        ),
        indent=2,
    )
    + "\n"
)
for pair in ["qwen-grpo", "qwen-sft", "r1-justrl"]:
    group = [r for r in trajectories if r["pair"] == pair and r["requested_rank"] == 16]
    for step in sorted({r["step"] for r in group}):
        print(
            pair,
            step,
            [
                (side, round(median(r["final_overlap"] for r in group if r["side"] == side and r["step"] == step), 4))
                for side in ["left", "right"]
            ],
        )
