"""Cross-family teacher-frame capture and equal-spectrum orientation controls.

Keep teacher top16 singular values fixed, replace left/right frames with three
isotropic orthonormal frames, and measure student parameter alignment. This
is a parameter-geometry intervention, not a new functional KL measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from measure_expanded_spectrum import Measurements, TensorCheckpoint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "direct-pairs-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pairs = [p for p in json.loads((OUT / "inventory.json").read_text())["pairs"] if p["family"] == args.family]
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    count = 0
    for p in pairs:
        students = {(c["run_id"], int(step), path) for c in p["conditions"] for step, path in c["checkpoints"].items()}
        if not students:
            continue
        bc = TensorCheckpoint(p["base"])
        tc = TensorCheckpoint(p["teacher_path"])
        bc.compatible(tc)
        for key in Measurements().keys(p["base"]):
            dest = OUT / "direction-controls" / p["pair_id"] / (key + ".json")
            if dest.exists():
                continue
            b = bc.get(key).double()
            dt = tc.get(key).double() - b
            u, s, v = torch.linalg.svd(dt, full_matrices=False)
            k = min(16, len(s))
            top = s[:k]
            energy = float(top.square().sum())
            random = []
            for seed in [311, 719, 2027]:
                g = torch.Generator(device="cpu").manual_seed(seed)
                left = torch.linalg.qr(torch.randn(dt.shape[0], k, generator=g, dtype=dt.dtype), mode="reduced")[0]
                right = torch.linalg.qr(torch.randn(dt.shape[1], k, generator=g, dtype=dt.dtype), mode="reduced")[0]
                random.append((seed, left, right))
            stages = []
            for run_id, step, path in sorted(students):
                sc = TensorCheckpoint(path)
                bc.compatible(sc)
                ds = sc.get(key).double() - b
                coords = u.T @ ds @ v.T
                a = coords.diagonal()
                es = float(ds.square().sum())
                bands = []
                cuts = [
                    0,
                    max(1, int(torch.ceil(torch.tensor(len(s) * 0.01)))),
                    max(1, int(torch.ceil(torch.tensor(len(s) * 0.05)))),
                    max(1, int(torch.ceil(torch.tensor(len(s) * 0.2)))),
                    len(s),
                ]
                for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
                    teacher = s[lo:hi]
                    coeff = a[lo:hi]
                    et = float(teacher.square().sum())
                    bands.append(
                        dict(
                            lo=lo,
                            hi=hi,
                            teacher_energy=et,
                            gain=float((teacher * coeff).sum()) / et,
                            residual=float((teacher - coeff).square().sum()) / et,
                        )
                    )
                dot = float((a[:k] * top).sum())
                controls = []
                for seed, left, right in random:
                    diagonal = (left.T @ ds @ right).diagonal()
                    value = float((diagonal * top).sum())
                    controls.append(
                        dict(seed=seed, dot=value, gain=value / energy, cosine=value / (energy * es) ** 0.5)
                    )
                stages.append(
                    dict(
                        run_id=run_id,
                        step=step,
                        path=path,
                        student_energy=es,
                        teacher_energy=float(s.square().sum()),
                        matched_fraction=float(a.square().sum()) / es,
                        top16_energy=energy,
                        top16_gain=dot / energy,
                        top16_cosine=dot / (energy * es) ** 0.5,
                        isotropic_cosine_RMS=dt.numel() ** -0.5,
                        random_frames=controls,
                        bands=bands,
                    )
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    dict(
                        pair_id=p["pair_id"], tensor=key, shape=list(dt.shape), stages=stages, source_sha256=source_sha
                    ),
                    indent=2,
                )
            )
            count += 1
            print(args.family, p["teacher"], key, len(stages), flush=True)
    (OUT / (args.family + "-controls-complete.json")).write_text(
        json.dumps(
            dict(
                matrices_this_attempt=count,
                source_sha256=source_sha,
                intervention="Exact same teacher top16 singular values, three randomized left/right frames; parameter alignment only",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
