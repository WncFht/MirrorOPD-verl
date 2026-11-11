"""Same-initial-student teacher geometry: signed delta and normalized spectra.

Each comparison uses identical tensors and coordinates. Normalized-spectrum
similarity ignores direction, while cosine preserves direction. CPU only.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from measure_expanded_spectrum import Measurements, TensorCheckpoint

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "spectrum-expanded-v4"
OUT = ROOT / "curve-spectrum-v1"


def main():
    torch.set_num_threads(2)
    inv = json.loads((SRC / "inventory.json").read_text())["runs"]
    groups = {}
    for r in inv:
        if r["architecture_compatible"]:
            groups.setdefault(r["base"], {})[r["teacher_path"]] = r
    out = []
    for base, targets in groups.items():
        if len(targets) < 2:
            continue
        cache = {}
        bc = TensorCheckpoint(base)
        for key in Measurements().keys(base):
            b = bc.get(key).double()
            for path in targets:
                dt = TensorCheckpoint(path).get(key).double() - b
                s = torch.linalg.svdvals(dt)
                cache[path, key] = (dt, s / s.norm())
            for a, bpath in itertools.combinations(targets, 2):
                da, sa = cache[a, key]
                db, sb = cache[bpath, key]
                out.append(
                    dict(
                        base=base,
                        teacher_a=targets[a]["teacher"],
                        teacher_b=targets[bpath]["teacher"],
                        path_a=a,
                        path_b=bpath,
                        tensor=key,
                        delta_cosine=float((da * db).sum() / (da.norm() * db.norm())),
                        normalized_spectrum_distance=float((sa - sb).norm()),
                        norm_logratio=float(torch.log10(da.norm() / db.norm())),
                        relative_delta_distance=float((da - db).norm() / torch.sqrt(da.norm() * db.norm())),
                    )
                )
            cache.clear()
        print("base", base, "teachers", len(targets), flush=True)
    (OUT / "teacher-neighbor-matrices.json").write_text(json.dumps(out, indent=2))
    agg = []
    for a, b in sorted({(x["path_a"], x["path_b"]) for x in out}):
        rows = [x for x in out if x["path_a"] == a and x["path_b"] == b]
        row = {k: rows[0][k] for k in ["base", "teacher_a", "teacher_b", "path_a", "path_b"]}
        row.update(
            {
                k: float(np.median([x[k] for x in rows]))
                for k in ["delta_cosine", "normalized_spectrum_distance", "norm_logratio", "relative_delta_distance"]
            }
        )
        agg.append(row)
    (OUT / "teacher-neighbors.json").write_text(
        json.dumps(
            dict(comparisons=agg, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2
        )
    )
    print("complete", len(agg), "teacher comparisons", flush=True)


if __name__ == "__main__":
    main()
