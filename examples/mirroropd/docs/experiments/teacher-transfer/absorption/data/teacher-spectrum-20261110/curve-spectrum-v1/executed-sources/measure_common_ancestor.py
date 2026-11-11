"""Full-layer attention/MLP decomposition around the shared Qwen Base gap.

For experimental initial B, common ancestor A and teacher T, T-B=(A-B)+(T-A).
Measure exact Frobenius energies/cross terms and shared-gap projection in all
layer attention and MLP matrices, without any GPU or matrix truncation.
Layernorm/embedding/head excluded explicitly. This tests six-matrix coverage.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import torch
from measure_expanded_spectrum import TensorCheckpoint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "common-ancestor-v1"
SRC = ROOT / "spectrum-expanded-v4"


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    inv = json.loads((SRC / "inventory.json").read_text())["runs"]
    candidates = [r for r in inv if r["student"] in ["q3-1.7", "q3-1.7-nonthinking"] and r["architecture_compatible"]]
    baseline = next(r for r in candidates if r["teacher"] == "q3-base")
    bc = TensorCheckpoint(baseline["base"])
    ac = TensorCheckpoint(baseline["teacher_path"])
    keys = bc.selected(r"model\.layers\.\d+\.(self_attn|mlp)\..*\.weight$")
    targets = {r["teacher_path"]: r for r in candidates}
    start = time.monotonic()
    for path, r in targets.items():
        dest = OUT / (r["teacher"] + ".json")
        if dest.exists():
            continue
        tc = TensorCheckpoint(path)
        bc.compatible(tc)
        records = []
        for key in keys:
            b = bc.get(key).double()
            gap = ac.get(key).double() - b
            d = tc.get(key).double() - b
            own = d - gap
            eg = float(gap.square().sum())
            ed = float(d.square().sum())
            eo = float(own.square().sum())
            cross = float((gap * own).sum())
            dot = float((gap * d).sum())
            assert abs(ed - eg - eo - 2 * cross) < 1e-8 * max(ed, 1)
            records.append(
                dict(
                    tensor=key,
                    layer=int(re.search(r"layers\.(\d+)", key)[1]),
                    module=key.split(".")[-2],
                    base_energy=float(b.square().sum()),
                    gap_energy=eg,
                    teacher_delta_energy=ed,
                    teacher_ancestor_energy=eo,
                    cross=cross,
                    dot_gap_delta=dot,
                    shared_gap_gain=dot / eg,
                    shared_axis_energy_fraction=dot * dot / (eg * ed) if ed else None,
                    teacher_ancestor_relative_norm=(eo / eg) ** 0.5,
                )
            )
        obj = dict(
            teacher=r["teacher"],
            teacher_path=path,
            initial=baseline["base"],
            ancestor=baseline["teacher_path"],
            matrices=records,
        )
        dest.write_text(json.dumps(obj, indent=2))
        print(r["teacher"], len(records), round(time.monotonic() - start, 1), flush=True)
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(
                teachers=len(targets),
                matrices_per_teacher=len(keys),
                elapsed=time.monotonic() - start,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="All layer attention and MLP 2D weights; exclude norm, embedding and head",
                device="CPU float64 threads2",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
