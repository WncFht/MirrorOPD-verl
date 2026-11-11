"""Exact CPU spectral screen across all explorer model pairs.

A datum is a model matrix; teacher-level summaries aggregate prespecified six
k/v projection matrices at first/middle/last depth. Roles B/T/S and T-B/S-B/T-S
remain distinct. Cross-size checkpoints receive native full-weight spectra only.
No GPU, no training, and no inference. Stored tensors promoted before subtraction.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
from examples.mirroropd.research.studies.spectral.weights import TensorCheckpoint, tensor_digest  # noqa: E402

OUT = ROOT / "spectrum-expanded-v4"


def dump(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def spectrum(s, shape):
    p = s.square() / s.square().sum() if float(s.square().sum()) else None
    if p is None:
        return dict(
            frobenius=0.0,
            stable_rank=None,
            effective_rank=None,
            rank90=None,
            rank90_fraction=None,
            top1=None,
            top5=None,
            top16=None,
            top1pct=None,
            top5pct=None,
            stable_rank_fraction=None,
        )
    r90 = int(torch.searchsorted(p.cumsum(0), torch.tensor(0.9, dtype=p.dtype))) + 1
    return dict(
        frobenius=float(s.norm()),
        stable_rank=float(1 / p[0]),
        effective_rank=float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())),
        rank90=r90,
        rank90_fraction=r90 / len(s),
        stable_rank_fraction=float(1 / p[0]) / len(s),
        **{
            name: float(p[:k].sum())
            for name, k in [
                ("top1", 1),
                ("top5", 5),
                ("top16", 16),
                ("top1pct", math.ceil(0.01 * len(s))),
                ("top5pct", math.ceil(0.05 * len(s))),
            ]
        },
    )


class Measurements:
    def __init__(self):
        self.checkpoints = {}
        self.frames = {}
        self.started = time.monotonic()
        self.count = 0

    def checkpoint(self, path):
        if path not in self.checkpoints:
            self.checkpoints[path] = TensorCheckpoint(path)
        return self.checkpoints[path]

    def keys(self, path):
        config = json.loads((Path(path) / "config.json").read_text())
        n = config["num_hidden_layers"]
        return [
            f"model.layers.{layer}.self_attn.{module}.weight"
            for layer in [0, (n - 1) // 2, n - 1]
            for module in ["k_proj", "v_proj"]
        ]

    def full(self, path, key, frame=False):
        ident = hashlib.sha256(path.encode()).hexdigest()[:16]
        result = OUT / "weights" / ident / f"{key}.json"
        c = self.checkpoint(path)
        raw = c.get(key)
        x = raw.double()
        if frame and (path, key) not in self.frames:
            self.frames[path, key] = torch.linalg.svd(x, full_matrices=False)
        if result.exists():
            row = json.loads(result.read_text())
            assert row["tensor_sha256"] == tensor_digest(raw)
        else:
            s = self.frames[path, key][1] if frame else torch.linalg.svdvals(x)
            row = dict(
                path=path,
                tensor=key,
                shape=list(x.shape),
                dtype=str(raw.dtype),
                tensor_sha256=tensor_digest(raw),
                singular_values=s.tolist(),
                **spectrum(s, x.shape),
            )
            assert abs(float(s.square().sum() / x.square().sum()) - 1) < 1e-8
            dump(result, row)
        return x, row, self.frames.get((path, key))

    def delta(self, b, t, frame, target_s):
        u, s0, v = frame
        delta = t - b
        ud, s, vd = torch.linalg.svd(delta, full_matrices=False)
        energy = float(delta.square().sum())
        coords = u.T @ delta @ v.T
        diag = float(coords.diagonal().square().sum())
        inside = float(coords.square().sum())
        sd = (torch.tensor(target_s, dtype=torch.float64) - s0).norm()
        null = float((delta - u @ coords @ v).square().sum())
        assert abs(diag + max(inside - diag, 0) + null - energy) < 1e-8 * max(energy, 1)
        if energy:
            assert abs(float(s.square().sum()) / energy - 1) < 1e-8
        return dict(
            singular_values=s.tolist(),
            **spectrum(s, delta.shape),
            norm_over_base=float(delta.norm() / b.norm()),
            kappa_spec=b.numel() / min(b.shape) * diag / energy if energy else None,
            rho_sigma=float(sd / math.sqrt(energy)) if energy else None,
            frame_diag=diag,
            frame_off=max(inside - diag, 0),
            frame_null=null,
        ), (ud, s, vd)

    def pair(self, row):
        base_path, target_path = row["base"], row["teacher_path"]
        pairid = hashlib.sha256((base_path + "|" + target_path).encode()).hexdigest()[:16]
        bc, tc = self.checkpoint(base_path), self.checkpoint(target_path)
        try:
            bc.compatible(tc)
            compatible = True
        except ValueError:
            compatible = False
        keys = self.keys(base_path)
        # Native full-weight fingerprint for every teacher, including cross-size.
        for key in self.keys(target_path):
            self.full(target_path, key)
        for key in keys:
            path = OUT / "pairs" / pairid / f"{key}.json"
            if path.exists():
                continue
            started = time.monotonic()
            b, br, frame = self.full(base_path, key, frame=True)
            result = dict(
                pair_id=pairid,
                base=base_path,
                teacher_path=target_path,
                tensor=key,
                base_full=br,
                compatible=compatible,
                teacher_delta=None,
                stages=[],
            )
            if compatible:
                t, tr, _ = self.full(target_path, key)
                dt, tf = self.delta(b, t, frame, tr["singular_values"])
                result.update(teacher_full=tr, teacher_delta=dt)
            for step, student in sorted(row["checkpoints"].items(), key=lambda x: int(x[0])):
                bc.compatible(self.checkpoint(student))
                s, sr, _ = self.full(student, key)
                ds, sf = self.delta(b, s, frame, sr["singular_values"])
                stage = dict(
                    step=int(step),
                    path=student,
                    full=sr,
                    delta=ds,
                    missing=None,
                    update_cosine=None,
                    left_overlap16=None,
                    right_overlap16=None,
                )
                if compatible:
                    missing = torch.linalg.svdvals(t - s)
                    stage["missing"] = dict(
                        singular_values=missing.tolist(),
                        **spectrum(missing, t.shape),
                        norm_over_base=float((t - s).norm() / b.norm()),
                    )
                    if dt["frobenius"] and ds["frobenius"]:
                        stage["update_cosine"] = float(((t - b) * (s - b)).sum() / ((t - b).norm() * (s - b).norm()))
                        stage["left_overlap16"] = float((tf[0][:, :16].T @ sf[0][:, :16]).square().sum() / 16)
                        stage["right_overlap16"] = float((tf[2][:16] @ sf[2][:16].T).square().sum() / 16)
                result["stages"].append(stage)
            dump(path, result)
            self.count += 1
            dump(
                OUT / "progress.json",
                dict(
                    run_id=row["run_id"],
                    tensor=key,
                    matrices_this_attempt=self.count,
                    elapsed=time.monotonic() - self.started,
                    last_seconds=time.monotonic() - started,
                ),
            )
            print(row["run_id"], key, round(time.monotonic() - started, 2), flush=True)
        return pairid, compatible


def smoke():
    torch.manual_seed(55)
    b = torch.randn(6, 9, dtype=torch.float64)
    t = b + 0.1 * torch.randn_like(b)
    m = Measurements()
    d, _ = m.delta(b, t, torch.linalg.svd(b, full_matrices=False), torch.linalg.svdvals(t).tolist())
    assert abs(d["frobenius"] - float((t - b).norm())) < 1e-10
    z, _ = m.delta(b, b, torch.linalg.svd(b, full_matrices=False), torch.linalg.svdvals(b).tolist())
    assert z["kappa_spec"] is None and z["stable_rank"] is None
    print("synthetic energy, reconstruction and zero-delta checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.smoke:
        smoke()
        return
    protocol = OUT / "measurement-protocol.json"
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inventory = json.loads((OUT / "inventory.json").read_text())
    if protocol.exists():
        assert args.resume, "Use explicit --resume for completed matrix reuse"
        assert json.loads(protocol.read_text())["source_sha256"] == sha
    else:
        dump(
            protocol,
            dict(
                source_sha256=sha,
                device="cpu",
                dtype="float64",
                threads=2,
                inventory_sha256=hashlib.sha256((OUT / "inventory.json").read_bytes()).hexdigest(),
                scope="k_proj,v_proj; layers 0,floor((L-1)/2),L-1; exact SVD",
                subtraction="stored tensors promoted before subtraction; no inferred FP32 masters",
            ),
        )
    m = Measurements()
    # Same B,T may have different input-mode students: merge checkpoint identities.
    grouped = {}
    for r in inventory["runs"]:
        key = (r["base"], r["teacher_path"])
        g = grouped.setdefault(key, dict(r, checkpoints={}))
        for step, path in r["checkpoints"].items():
            g["checkpoints"][step + "@" + r["run_id"]] = path
    # Stage key must remain numeric; process each run and cache pair results per run,
    # because a shared teacher can have distinct thinking/nonthinking students.
    identities = []
    for r in inventory["runs"]:
        # Isolate mode-specific stage rows while reusing full tensor measurements.
        pairkey = hashlib.sha256((r["base"] + "|" + r["teacher_path"]).encode()).hexdigest()[:16]
        previous = OUT / "pairs" / pairkey
        runout = OUT / "runs" / r["run_id"]
        if runout.exists():
            continue
        if previous.exists():
            # Previous mode already owns that pair path; move its immutable rows to
            # its run output before the next mode is processed (below).
            raise RuntimeError("Unattached pair result; inspect prior attempt before resume")
        try:
            ident, compatible = m.pair(r)
            runout.parent.mkdir(exist_ok=True)
            previous.rename(runout)
            identities.append(dict(run_id=r["run_id"], pair_id=ident, compatible=compatible))
            dump(OUT / "run-receipts" / f"{r['run_id']}.json", identities[-1])
        except Exception as exc:
            dump(OUT / "errors" / f"{r['run_id']}.json", dict(type=type(exc).__name__, message=str(exc)))
            raise
    dump(
        OUT / "measurement-complete.json",
        dict(runs=len(list((OUT / "runs").iterdir())), elapsed=time.monotonic() - m.started, source_sha256=sha),
    )


if __name__ == "__main__":
    main()
