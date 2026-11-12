# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only, streamed checkpoint geometry; one matrix, all stages and ranks.

Historical CPU entry points retain their original protocols. This engine combines
right-frame dynamics, update spectra, full-weight geometry and parameter residuals.
Only small frames survive between stages; checkpoint-sized updates are reread.
"""

import gzip
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

import torch

from examples.mirroropd.research.runtime.processes import identity

from .right_dynamics import angle_metrics
from .scan import Timings, iso_scalars, singular_values, thin_svd
from .update_dynamics import atomic_json
from .weights import TensorCheckpoint, tensor_digest


@lru_cache(maxsize=32)
def checkpoint(path):
    """Cache only checkpoint indexes and shapes, never model tensors."""
    return TensorCheckpoint(path)


def describe(s, energy, ranks):
    values = s.double().square()
    if energy <= 0 or not math.isclose(float(values.sum()), energy, rel_tol=1e-8):
        raise ValueError("Nonzero update and SVD energy identity required")
    return {
        "energy": energy,
        "singular_values": s.cpu().tolist(),
        "stable_rank": energy / float(values[0]),
        **{f"rank{p}": min(len(s), int((values.cumsum(0) < p / 100 * energy).sum()) + 1) for p in (90, 95)},
        "ranks": [
            {
                "requested_rank": rank,
                "rank": min(rank, len(s)),
                "captured_energy": float(values[:rank].sum()) / energy,
                "boundary_relative_gap": float((s[rank - 1] - s[rank]) / s[rank - 1])
                if rank < len(s) and float(s[rank - 1]) > 0
                else None,
            }
            for rank in ranks
        ],
    }


def measure(task, device="cpu", *, driver=None):
    """Use exact FP64 SVD, without training, forward passes or persisted frames."""
    ranks = task["ranks"]
    points = task["points"]
    if not ranks or ranks != sorted(set(ranks)) or any(type(k) is not int or k <= 0 for k in ranks):
        raise ValueError("Ranks must be positive, unique and increasing")
    steps = [s for s, _ in points]
    if not steps or steps != sorted(set(steps)) or any(type(s) is not int or s <= 0 for s in steps):
        raise ValueError("Points must have unique, increasing positive optimizer steps")
    torch.set_num_threads(task.get("threads", 2))
    device = torch.device(device)
    clock = Timings(device)
    key = task["tensor"]
    driver = (driver or task.get("driver")) if device.type == "cuda" else None

    def svd(x):
        return thin_svd(x, driver=driver, transpose_wide=True)

    hashes = {}

    def load(path, label):
        with clock.phase("read_hash_transfer"):
            raw = checkpoint(path).get(key)
            digest = tensor_digest(raw)
            if label in hashes and hashes[label] != digest:
                raise ValueError("Checkpoint tensor changed during measurement")
            hashes[label] = digest
            return raw.to(device=device, dtype=torch.float64)

    started = time.monotonic()
    base = load(task["base"], "initial")
    result = dict(
        schema="absorption-geometry-v1",
        task=task,
        shape=list(base.shape),
        sha256=hashes,
        device=str(device),
        dtype="float64",
        execution=dict(engine="trajectory-v2", driver=driver, projection="shared-max-rank"),
        stages=[],
        pairs=[],
        increments=[],
    )
    with torch.inference_mode():
        with clock.phase("base_svd"):
            base_frame = svd(base)
        result["initial_singular_values"] = base_frame[1].cpu().tolist()
        teacher = teacher_frame = None
        if task.get("teacher_path"):
            teacher = load(task["teacher_path"], "teacher")
            if teacher.shape != base.shape:
                raise ValueError("Teacher geometry requires matching coordinates")
            with clock.phase("teacher_weight_svd"):
                wu, ws, wvh = svd(teacher)
            teacher_weight = dict(singular_values=ws.cpu().tolist(), **iso_scalars(base, teacher, base_frame, ws))
            del wu, ws, wvh
            teacher = teacher - base
            with clock.phase("teacher_svd"):
                tu, ts, tvh = svd(teacher)
            te = float(teacher.square().sum())
            result["teacher"] = describe(ts, te, ranks) if te else {"energy": 0, "zero_update": True}
            result["teacher"]["weight"] = teacher_weight
            if te:
                teacher_frame = tvh[: min(max(ranks), len(ts))].T.contiguous()
            del tu, ts, tvh
        frames = []
        previous = None
        for step, path in points:
            target = load(path, str(step))
            if target.shape != base.shape:
                raise ValueError("Student checkpoint tensor coordinates changed")
            delta = target - base
            with clock.phase("update_svd"):
                u, s, vh = svd(delta)
            en = float(delta.square().sum())
            record = dict(step=step, **describe(s, en, ranks))
            frame = vh[: min(max(ranks), len(s))].T.contiguous()
            frames.append((step, frame.cpu()))
            del u, s, vh
            with clock.phase("full_weight_svd"):
                wu, ws, wvh = svd(target)
            record["weight"] = dict(singular_values=ws.cpu().tolist(), **iso_scalars(base, target, base_frame, ws))
            record["weight"]["frames"] = [
                dict(requested_rank=k, rank=min(k, len(ws)), **angle_metrics(base_frame[2][:k].T, wvh[:k].T))
                for k in ranks
            ]
            del wu, ws, wvh, target
            if teacher is not None:
                te = float(teacher.square().sum())
                dot = float((delta * teacher).sum())
                record["teacher_comparison"] = dict(
                    teacher_energy=te,
                    student_energy=en,
                    dot=dot,
                    norm_ratio=math.sqrt(en / te) if te else None,
                    cosine=dot / math.sqrt(en * te) if te else None,
                    residual_energy_ratio=float((teacher - delta).square().sum()) / te if te else None,
                )
            if teacher_frame is not None:
                record["teacher_comparison"]["frames"] = [
                    dict(
                        requested_rank=k,
                        rank=min(k, frame.shape[1]),
                        **angle_metrics(teacher_frame[:, :k], frame[:, :k]),
                    )
                    for k in ranks
                ]
            result["stages"].append(record)
            if previous is not None:
                start, prev = previous
                increment = delta - prev
                ie = float(increment.square().sum())
                item = dict(start=start, end=step, optimizer_interval=step - start, energy=ie, captures=[])
                if ie:
                    with clock.phase("increment_svd"):
                        iss = singular_values(increment, driver=driver, transpose_wide=True)
                    item.update(describe(iss, ie, ranks))
                    for anchor, basis in frames[:-1]:
                        v = basis.to(device)
                        captured = (increment @ v).square().sum(0).cumsum(0).cpu()
                        optimal_cdf = iss.square().cumsum(0).cpu()
                        for rank in ranks:
                            k = min(rank, v.shape[1])
                            optimal = float(optimal_cdf[k - 1]) / ie
                            capture = float(captured[k - 1]) / ie
                            if capture > optimal + 1e-8:
                                raise ValueError("Increment capture exceeds optimal rank-k capture")
                            item["captures"].append(
                                dict(
                                    anchor=anchor,
                                    requested_rank=rank,
                                    rank=k,
                                    capture=capture,
                                    optimal_capture=optimal,
                                    relative_to_optimal=capture / optimal,
                                )
                            )
                    del iss
                result["increments"].append(item)
                del increment, prev
            previous = (step, delta)
        del previous, delta, frame, base_frame, teacher
        # Second streamed pass: avoid retaining nine checkpoint-sized deltas.
        with clock.phase("stage_pairs"):
            for i, (step, path) in enumerate(points):
                delta = load(path, str(step)) - base
                en = result["stages"][i]["energy"]
                v = frames[i][1].to(device)
                for j in range(i + 1):
                    anchor, cpu_basis = frames[j]
                    a = cpu_basis.to(device)
                    cross = a.T @ v
                    captured = (delta @ a).square().sum(0).cumsum(0).cpu()
                    for rank in ranks:
                        k = min(rank, a.shape[1])
                        result["pairs"].append(
                            dict(
                                step_a=anchor,
                                step_b=step,
                                requested_rank=rank,
                                rank=k,
                                full_input_space=k == base.shape[1],
                                **angle_metrics(a[:, :k], v[:, :k], cross=cross[:k, :k]),
                                a_captures_b=float(captured[k - 1]) / en,
                            )
                        )
    result["timing_seconds"] = dict(clock.seconds)
    result["wall_seconds"] = time.monotonic() - started
    return result


def resume_records(manifest, output, sources):
    """Copy accepted immutable records; task identities and stage coverage must match."""
    import shutil

    indices = {json.dumps(t, sort_keys=True): i for i, t in enumerate(manifest["tasks"])}
    if len(indices) != len(manifest["tasks"]):
        raise ValueError("Duplicate matrix tasks")
    reused = {}
    for directory in map(Path, sources):
        source_plan = json.loads((directory / "protocol.json").read_text())
        if source_plan != manifest:
            raise ValueError("Resume protocol differs from requested manifest")
        for path in sorted(directory.glob("matrix-*.json.gz")):
            raw = path.read_bytes()
            record = json.loads(gzip.decompress(raw))
            index = indices.get(json.dumps(record["task"], sort_keys=True))
            if index is None or record.get("schema") != "absorption-geometry-v1" or record.get("dtype") != "float64":
                raise ValueError("Resume record identity or precision mismatch")
            if [s["step"] for s in record["stages"]] != [p[0] for p in record["task"]["points"]]:
                raise ValueError("Incomplete resume checkpoint coverage")
            target = output / f"matrix-{index:05d}.json.gz"
            if index in reused:
                if target.read_bytes() != raw:
                    raise ValueError("Conflicting resume records")
                continue
            temporary = target.with_suffix(".tmp")
            shutil.copyfile(path, temporary)
            temporary.replace(target)
            reused[index] = str(path)
    atomic_json(output / "reused.json", reused)
    return set(reused)


def run(manifest, output, devices=("cpu",), workers=1, *, driver=None, resume_from=()):
    """Bounded dynamic dispatch; STOP drains active matrices, then preserves a receipt."""
    if not manifest.get("tasks") or workers < 1:
        raise ValueError("Require nonempty tasks and positive workers")
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Provide unique devices")
    if devices != ("cpu",) and (any(not d.startswith("cuda:") for d in devices) or workers != 1):
        raise ValueError("CUDA uses one worker per explicit cuda:N device")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "protocol.json", manifest)
    atomic_json(output / "process.json", dict(pid=os.getpid(), identity=identity(os.getpid()), started_at=time.time()))
    atomic_json(output / "execution.json", dict(engine="trajectory-v2", driver=driver, devices=devices))
    started = time.monotonic()
    reused = resume_records(manifest, output, resume_from)
    done, failed = len(reused), 0
    tasks = iter((i, task) for i, task in enumerate(manifest["tasks"]) if i not in reused)
    pools = [ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) for _ in devices]
    pending = {}

    def submit(slot):
        if (output / "STOP").exists():
            return
        item = next(tasks, None)
        if item is not None:
            i, task = item
            future = pools[slot].submit(measure, task, devices[slot], driver=driver)
            pending[future] = (i, slot)

    def progress():
        return dict(
            complete=done,
            reused=len(reused),
            failed=failed,
            total=len(manifest["tasks"]),
            wall_seconds=time.monotonic() - started,
        )

    try:
        for slot in range(len(devices)):
            for _ in range(workers):
                submit(slot)
        atomic_json(output / "progress.json", progress())
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                i, slot = pending.pop(future)
                try:
                    record = future.result()
                    path = output / f"matrix-{i:05d}.json.gz"
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(gzip.compress(json.dumps(record, allow_nan=False).encode(), mtime=0))
                    tmp.replace(path)
                    done += 1
                except Exception as error:
                    failed += 1
                    atomic_json(output / f"error-{i:05d}.json", dict(task=manifest["tasks"][i], error=repr(error)))
                atomic_json(output / "progress.json", progress())
                submit(slot)
    finally:
        for pool in pools:
            pool.shutdown()
    stopped = done + failed < len(manifest["tasks"])
    atomic_json(output / "exit.json", dict(**progress(), stopped=stopped, exit_code=2 if stopped else int(failed > 0)))
    if failed:
        raise RuntimeError(f"{failed} matrix tasks failed; retain errors and completed records")
    if stopped:
        raise InterruptedError("STOP drained active matrices; resume unmeasured tasks in a new output")
