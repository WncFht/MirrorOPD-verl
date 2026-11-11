"""CPU parameter experiments: student capture in teacher-delta singular bands.

The paired rank-one matrices u_i v_i^T are Frobenius orthonormal. Their
coefficients, residuals and off-diagonal energy distinguish singular-value
concentration from learning the associated parameter directions. Random
matched-diagonal energy has exact isotropic expectation ||deltaS||² k/(mn),
which supplies a geometric baseline, not a function-response experiment.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from measure_expanded_spectrum import Measurements, TensorCheckpoint, tensor_digest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "spectrum-expanded-v4"
OUT = ROOT / "teacher-band-capture-v1"


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    inv = json.loads((SRC / "inventory.json").read_text())["runs"]
    inv = [r for r in inv if r["architecture_compatible"]]
    ancillary = json.loads((SRC / "ancillary-complete.json").read_text())["conditions"]
    seen = set()
    rows = []
    for r in inv + ancillary:
        key = (r["base"], r["teacher_path"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    started = time.monotonic()
    count = 0
    for r in rows:
        ident = hashlib.sha256((r["base"] + "|" + r["teacher_path"]).encode()).hexdigest()[:16]
        bc = TensorCheckpoint(r["base"])
        tc = TensorCheckpoint(r["teacher_path"])
        bc.compatible(tc)
        for key in Measurements().keys(r["base"]):
            dest = OUT / ident / (key + ".json")
            if dest.exists():
                continue
            braw = bc.get(key)
            traw = tc.get(key)
            b = braw.double()
            dt = traw.double() - b
            u, s, v = torch.linalg.svd(dt, full_matrices=False)
            rank = len(s)
            total = float(s.square().sum())
            assert abs(total - float(dt.square().sum())) < 1e-8 * max(total, 1)
            cuts = sorted(set([0, math.ceil(0.01 * rank), math.ceil(0.05 * rank), math.ceil(0.2 * rank), rank]))
            stages = []
            for step, path in sorted(r.get("checkpoints", {}).items(), key=lambda x: int(x[0])):
                sc = TensorCheckpoint(path)
                bc.compatible(sc)
                raw = sc.get(key)
                ds = raw.double() - b
                coords = u.T @ ds @ v.T
                diag = coords.diagonal()
                dse = float(ds.square().sum())
                missing = float((dt - ds).square().sum())
                bands = []
                for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
                    ts = s[lo:hi]
                    a = diag[lo:hi]
                    energy = float(ts.square().sum())
                    cross = float((a * ts).sum())
                    learned = float(a.square().sum())
                    error = float((a - ts).square().sum())
                    bands.append(
                        dict(
                            lo=lo,
                            hi=hi,
                            teacher_energy=energy,
                            teacher_energy_fraction=energy / total,
                            student_matched_energy=learned,
                            missing_matched_energy=error,
                            amplitude_gain=cross / energy,
                            matched_residual=error / energy,
                            cosine=cross / math.sqrt(energy * learned) if learned else None,
                            isotropic_expected_energy=dse * (hi - lo) / dt.numel(),
                        )
                    )
                matched = float(diag.square().sum())
                off = float(coords.square().sum()) - matched
                outside = max(dse - float(coords.square().sum()), 0)
                # Exact distance identity: teacher-diagonal residual + student off/complement.
                reconstructed = sum(x["missing_matched_energy"] for x in bands) + off + outside
                assert abs(missing - reconstructed) < 1e-8 * max(missing, 1)
                stages.append(
                    dict(
                        step=int(step),
                        path=path,
                        tensor_sha256=tensor_digest(raw),
                        student_energy=dse,
                        teacher_energy=total,
                        missing_energy=missing,
                        matched_energy=matched,
                        off_energy=off,
                        outside_energy=outside,
                        matched_fraction=matched / dse if dse else None,
                        off_fraction=off / dse if dse else None,
                        outside_fraction=outside / dse if dse else None,
                        isotropic_matched_expectation=dse * rank / dt.numel(),
                        bands=bands,
                    )
                )
            out = dict(
                run_id=r["run_id"],
                base=r["base"],
                teacher_path=r["teacher_path"],
                tensor=key,
                shape=list(dt.shape),
                base_sha256=tensor_digest(braw),
                teacher_sha256=tensor_digest(traw),
                teacher_energy=total,
                stages=stages,
            )
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(json.dumps(out, indent=2))
            count += 1
            print(r["run_id"], key, len(stages), round(time.monotonic() - started, 1), flush=True)
            (OUT / "progress.json").write_text(
                json.dumps(dict(pair=r["run_id"], matrices=count, seconds=time.monotonic() - started))
            )
    (OUT / "complete.json").write_text(
        json.dumps(
            dict(
                pairs=len(rows),
                matrices=count,
                elapsed=time.monotonic() - started,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                CPU_threads=2,
                GPU=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
