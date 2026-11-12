# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact CPU checkpoint-to-final update subspaces, with incremental evidence.

Uses saved same-coordinate checkpoints, never GPU or model inference. Per-tensor
records include SHA, ranks, singular values and elapsed time. Full-rank overlap
is structural saturation, not evidence of early locking. Checkpoint labels are
preserved; different earliest training fractions must be reported separately.
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure(task):
    import torch

    from .weights import TensorCheckpoint, tensor_digest

    torch.set_num_threads(task["threads"])
    started = time.monotonic()
    base = TensorCheckpoint(task["base"])
    key = task["tensor"]
    raw = base.get(key)
    hashes = {"initial": tensor_digest(raw)}
    b = raw.double()
    weight_frame = torch.linalg.svd(b, full_matrices=False) if task.get("weight_geometry") else None
    teacher_delta = None
    if task.get("teacher_path"):
        teacher_raw = TensorCheckpoint(task["teacher_path"]).get(key)
        if teacher_raw.shape != b.shape:
            raise ValueError("Teacher parameter metrics require matching matrix coordinates")
        hashes["teacher"] = tensor_digest(teacher_raw)
        teacher_delta = teacher_raw.double() - b
    frames = []
    for label, path in task["points"]:
        checkpoint = TensorCheckpoint(path)
        raw = checkpoint.get(key)
        if raw.shape != b.shape:
            raise ValueError(f"Incompatible checkpoint shape: {label}, {key}")
        hashes[label] = tensor_digest(raw)
        delta = raw.double() - b
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        energy = float(delta.square().sum())
        if energy == 0:
            raise ValueError(f"Zero cumulative update: {label}, {key}")
        frames.append((label, delta, u, s, vh, energy))
    result = dict(task=task, shape=list(b.shape), sha256=hashes, stages=[], rows=[])
    final = frames[-1]
    early = frames[0]
    for label, delta, u, s, vh, energy in frames:
        facts = dict(label=label, energy=energy, singular_values=s.tolist())
        if teacher_delta is not None:
            teacher_energy = float(teacher_delta.square().sum())
            dot = float((delta * teacher_delta).sum())
            facts["teacher_comparison"] = dict(
                teacher_energy=teacher_energy,
                norm_ratio=math.sqrt(energy / teacher_energy) if teacher_energy else None,
                cosine=dot / math.sqrt(energy * teacher_energy) if teacher_energy else None,
                residual_energy_ratio=float((teacher_delta - delta).square().sum()) / teacher_energy
                if teacher_energy
                else None,
            )
        if weight_frame is not None:
            from .scan import iso_scalars

            target = delta + b
            wu, ws, wvh = torch.linalg.svd(target, full_matrices=False)
            facts["weight_geometry"] = iso_scalars(b, target, weight_frame, ws)
            facts["weight_geometry"]["singular_values"] = ws.tolist()
            facts["weight_geometry"]["frames"] = []
            for requested in task["ranks"]:
                k = min(requested, len(ws))
                facts["weight_geometry"]["frames"].append(
                    dict(
                        requested_rank=requested,
                        effective_rank=k,
                        left_overlap=float((weight_frame[0][:, :k].T @ wu[:, :k]).square().sum() / k),
                        right_overlap=float((weight_frame[2][:k] @ wvh[:k].T).square().sum() / k),
                    )
                )
        result["stages"].append(facts)
        for requested in task["ranks"]:
            k = min(requested, len(s))
            for side, dimension in [("left", b.shape[0]), ("right", b.shape[1])]:
                basis = u[:, :k] if side == "left" else vh[:k].T
                overlap = float(((final[2][:, :k] if side == "left" else final[4][:k].T).T @ basis).square().sum() / k)
                random = k / dimension
                early_basis = early[2][:, :k] if side == "left" else early[4][:k].T
                projected = early_basis.T @ delta if side == "left" else delta @ early_basis
                result["rows"].append(
                    dict(
                        label=label,
                        side=side,
                        requested_rank=requested,
                        effective_rank=k,
                        random_overlap=random,
                        final_overlap=overlap,
                        adjusted_final_overlap=(overlap - random) / (1 - random) if random < 1 else None,
                        early_capture=float(projected.square().sum() / energy),
                        own_energy=float(s[:k].square().sum() / energy),
                    )
                )
        # SVD energy identity is checked for every checkpoint before accepting evidence.
        if not math.isclose(float(s.square().sum()), energy, rel_tol=1e-8):
            raise ValueError(f"SVD energy mismatch: {label}, {key}")
    result["wall_seconds"] = time.monotonic() - started
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", manifest)
    atomic_json(args.output / "process.json", dict(pid=os.getpid(), started=time.time()))
    total = len(manifest["tasks"])
    done, failed = 0, 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(measure, task): i for i, task in enumerate(manifest["tasks"])}
        for future in as_completed(pending):
            index = pending[future]
            try:
                value = future.result()
                atomic_json(args.output / f"matrix-{index:04d}.json", value)
                done += 1
                print(
                    index, value["task"]["job_id"], value["task"]["tensor"], round(value["wall_seconds"], 2), flush=True
                )
            except Exception as exc:
                failed += 1
                atomic_json(
                    args.output / f"error-{index:04d}.json", dict(task=manifest["tasks"][index], error=repr(exc))
                )
            atomic_json(
                args.output / "progress.json",
                dict(
                    complete=done,
                    failed=failed,
                    total=total,
                    wall_seconds=time.monotonic() - started,
                ),
            )
    atomic_json(args.output / "exit.json", dict(complete=done, failed=failed, total=total, exit_code=int(failed > 0)))


if __name__ == "__main__":
    main()
