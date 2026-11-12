# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU right-frame angles and capture of later checkpoint increments.

All tensors are read-only. Each task fixes one matrix and an ordered first-pass
checkpoint series. Full-weight and cumulative-update angles remain separate.
Outputs retain complete principal-angle distributions, not only averages.
"""

import argparse
import gzip
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .update_dynamics import atomic_json


def angle_metrics(a, b, *, cross=None):
    """Columns of a and b are orthonormal bases of equal rank."""
    import torch

    k = a.shape[1]
    cosines = torch.linalg.svdvals(a.T @ b if cross is None else cross).clamp(0, 1).cpu()
    angles = torch.rad2deg(torch.acos(cosines))
    overlap = float(cosines.square().mean())
    baseline = k / a.shape[0]
    return dict(
        angles_deg=angles.tolist(),
        mean_angle_deg=float(angles.mean()),
        median_angle_deg=float(angles.median()),
        p90_angle_deg=float(torch.quantile(angles, 0.9)),
        max_angle_deg=float(angles.max()),
        overlap=overlap,
        random_overlap=baseline,
        adjusted_overlap=(overlap - baseline) / (1 - baseline) if baseline < 1 else None,
        chordal_distance=math.sqrt(max(0, 1 - overlap)),
    )


def spectrum(s, energy, ranks):
    return dict(
        energy=energy,
        stable_rank=energy / float(s[0].square()),
        rank90=int((s.square().cumsum(0) < 0.9 * energy).sum()) + 1,
        ranks=[
            dict(
                rank=min(k, len(s)),
                requested_rank=k,
                captured_energy=float(s[:k].square().sum()) / energy,
                boundary_relative_gap=float((s[k - 1] - s[k]) / s[k - 1])
                if k < len(s) and float(s[k - 1]) > 0
                else None,
            )
            for k in ranks
        ],
    )


def measure(task):
    import torch

    from .weights import TensorCheckpoint, tensor_digest

    torch.set_num_threads(task.get("threads", 2))
    started = time.monotonic()
    raw = TensorCheckpoint(task["base"]).get(task["tensor"])
    base = raw.double()
    hashes = {"initial": tensor_digest(raw)}
    ranks = task["ranks"]
    base_frame = torch.linalg.svd(base, full_matrices=False)[2] if task.get("weight_angles") else None
    frames = []
    result = dict(
        task=task, shape=list(base.shape), sha256=hashes, stages=[], pairs=[], increments=[], weight_angles=[]
    )
    for step, path in task["points"]:
        raw = TensorCheckpoint(path).get(task["tensor"])
        if raw.shape != base.shape:
            raise ValueError("Checkpoint tensor shape changed")
        hashes[str(step)] = tensor_digest(raw)
        delta = raw.double() - base
        _, s, vh = torch.linalg.svd(delta, full_matrices=False)
        energy = float(delta.square().sum())
        if energy <= 0 or not math.isclose(float(s.square().sum()), energy, rel_tol=1e-8):
            raise ValueError("Zero update or failed SVD energy identity")
        frames.append((step, delta, vh[: max(ranks)].T.contiguous(), s, energy))
        result["stages"].append(dict(step=step, **spectrum(s, energy, ranks)))
        if base_frame is not None:
            _, ws, wvh = torch.linalg.svd(delta + base, full_matrices=False)
            for rank in ranks:
                k = min(rank, len(ws))
                result["weight_angles"].append(dict(step=step, rank=k, **angle_metrics(base_frame[:k].T, wvh[:k].T)))
    for ia, (sa, da, va, _, ea) in enumerate(frames):
        for ib in range(ia, len(frames)):
            sb, db, vb, _, eb = frames[ib]
            for rank in ranks:
                k = min(rank, va.shape[1])
                result["pairs"].append(
                    dict(
                        step_a=sa,
                        step_b=sb,
                        rank=k,
                        **angle_metrics(va[:, :k], vb[:, :k]),
                        a_captures_b=float((db @ va[:, :k]).square().sum()) / eb,
                        b_captures_a=float((da @ vb[:, :k]).square().sum()) / ea,
                    )
                )
    # Each later interval is tested against all frames saved before it starts.
    for i in range(len(frames) - 1):
        sa, da, _, _, _ = frames[i]
        sb, db, _, _, _ = frames[i + 1]
        inc = db - da
        energy = float(inc.square().sum())
        if energy == 0:
            result["increments"].append(dict(start=sa, end=sb, energy=0, captures=[]))
            continue
        _, s, _ = torch.linalg.svd(inc, full_matrices=False)
        captures = []
        for anchor, _, v, _, _ in frames[: i + 1]:
            for rank in ranks:
                k = min(rank, v.shape[1])
                own = float(s[:k].square().sum()) / energy
                capture = float((inc @ v[:, :k]).square().sum()) / energy
                if capture > own + 1e-8:
                    raise ValueError("Projection exceeds optimal rank-k capture")
                captures.append(
                    dict(anchor=anchor, rank=k, capture=capture, optimal_capture=own, relative_to_optimal=capture / own)
                )
        result["increments"].append(dict(start=sa, end=sb, **spectrum(s, energy, ranks), captures=captures))
    result["wall_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", manifest)
    atomic_json(args.output / "process.json", dict(pid=os.getpid(), started=time.time()))
    done, failed = 0, 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure, task): i for i, task in enumerate(manifest["tasks"])}
        for future in as_completed(futures):
            i = futures[future]
            try:
                value = future.result()
                path = args.output / f"matrix-{i:04d}.json.gz"
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False).encode(), mtime=0))
                temporary.replace(path)
                done += 1
                print(
                    i, value["task"]["condition"], value["task"]["tensor"], round(value["wall_seconds"], 2), flush=True
                )
            except Exception as exc:
                failed += 1
                atomic_json(args.output / f"error-{i:04d}.json", dict(task=manifest["tasks"][i], error=repr(exc)))
            atomic_json(
                args.output / "progress.json",
                dict(complete=done, failed=failed, total=len(futures), wall_seconds=time.monotonic() - started),
            )
    atomic_json(
        args.output / "exit.json", dict(complete=done, failed=failed, total=len(futures), exit_code=int(failed > 0))
    )


if __name__ == "__main__":
    main()
