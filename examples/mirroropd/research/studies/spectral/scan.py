# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact spectra grouped by base matrix; no model forward or intervention directions.

One atomic record contains one base matrix and all requested teacher/student
stages. Frames remain transient; content-keyed scalar/spectrum caches persist.
Historical weights/analyze output and protocols are left unchanged.
"""

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from examples.mirroropd.research.core.artifacts import fingerprint
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.checkpoints import CheckpointStore

from .absorption.cohort import report_eligible
from .decomposition import energy, ratio, spectrum_summary
from .weights import DEFAULT_SCOPE, TensorCheckpoint, tensor_digest


class Timings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds = defaultdict(float)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += time.perf_counter() - start


def thin_svd(matrix, *, driver=None, transpose_wide=False):
    if transpose_wide and matrix.shape[0] < matrix.shape[1]:
        ut, s, vht = torch.linalg.svd(matrix.T.contiguous(), full_matrices=False, driver=driver)
        return vht.T.contiguous(), s, ut.T.contiguous()
    return torch.linalg.svd(matrix, full_matrices=False, driver=driver)


def singular_values(matrix, *, driver=None, transpose_wide=False):
    if transpose_wide and matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T.contiguous()
    return torch.linalg.svdvals(matrix, driver=driver)


def iso_scalars(base, target, frame, target_singular):
    """Same ISO scalars as weight_summary, without full diag/off/null reconstruction."""
    u, initial, vh = frame
    delta = target - base
    diagonal = ((delta @ vh.T) * u).sum(0) if base.shape[0] <= base.shape[1] else ((u.T @ delta) * vh).sum(1)
    drift = (target_singular - initial).double().norm()
    base_energy, delta_energy, diag_energy = energy(base), energy(delta), energy(diagonal)
    return dict(
        base_energy=float(base_energy),
        update_energy=float(delta_energy),
        spectral_distance=float(drift),
        delta_sigma=ratio(drift, base_energy.sqrt()),
        rho_sigma=ratio(drift, delta_energy.sqrt()),
        kappa_spec=ratio(base.numel() / min(base.shape) * diag_energy, delta_energy),
    )


def selected_jobs(manifest, ids):
    jobs = [j for j in manifest["jobs"] if report_eligible(j) and (ids is None or j["job_id"] in ids)]
    if ids is not None and {j["job_id"] for j in jobs} != set(ids):
        raise ValueError("Requested job missing or excluded from report cohort")
    if not jobs:
        raise ValueError("No eligible jobs selected")
    return jobs


def scan(
    manifest,
    output,
    *,
    job_ids=None,
    pattern=DEFAULT_SCOPE,
    device="cpu",
    driver=None,
    transpose_wide=False,
    updates=False,
    stages=False,
    cache=None,
    resume=False,
    stop_paths=(),
    shards=1,
    shard_index=0,
):
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Require shards >= 1 and 0 <= shard_index < shards")
    started = time.perf_counter()
    jobs = selected_jobs(json.loads(Path(manifest).read_text()), job_ids)
    checkpoints = {}
    groups = defaultdict(list)
    for job in jobs:
        groups[job["base"]].append(job)
        paths = [job["base"], job["teacher_path"], job["endpoint"]]
        if stages:
            paths.extend(job["gradient_stages"].values())
        for path in paths:
            if path not in checkpoints:
                checkpoints[path] = TensorCheckpoint(path)
    signature = dict(
        version=2,
        shards=shards,
        shard_index=shard_index,
        jobs=jobs,
        checkpoints={p: c.identity for p, c in checkpoints.items()},
        pattern=pattern,
        device=str(device),
        driver=driver,
        transpose_wide=transpose_wide,
        updates=updates,
        stages=stages,
        dtype="float64",
        torch_version=str(torch.__version__),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    totals = defaultdict(float)
    completed, skipped = 0, 0
    initialization_seconds = time.perf_counter() - started

    def check_stop():
        if any(Path(p).exists() for p in stop_paths):
            raise InterruptedError("STOP requested")

    with CheckpointStore(output, signature) as results:
        dump(output / "protocol.json", signature)
        for base_path, paired in groups.items():
            keys = checkpoints[base_path].selected(pattern)
            targets = sorted(
                {j[k] for j in paired for k in ("teacher_path", "endpoint")}
                | ({p for j in paired for p in j["gradient_stages"].values()} if stages else set())
            )
            for path in targets:
                if any(checkpoints[path].shapes.get(key) != checkpoints[base_path].shapes[key] for key in keys):
                    raise ValueError(f"Selected backbone coordinates differ: {path}")
            for index, key in enumerate(keys):
                if index % shards != shard_index:
                    continue
                check_stop()
                clock = Timings(device)
                start = time.perf_counter()
                with clock.phase("read_hash"):
                    raw_base = checkpoints[base_path].get(key)
                    raw = {p: checkpoints[p].get(key) for p in targets}
                    inputs = {
                        p: dict(sha256=tensor_digest(t), dtype=str(t.dtype), shape=list(t.shape))
                        for p, t in raw.items()
                    }
                    inputs[base_path] = dict(
                        sha256=tensor_digest(raw_base), dtype=str(raw_base.dtype), shape=list(raw_base.shape)
                    )
                record_key = fingerprint(dict(base=base_path, tensor=key))
                saved = results.json(record_key)
                if saved is not None:
                    if saved["inputs"] != inputs:
                        raise ValueError(f"Changed tensor in resumed scan: {key}")
                    skipped += 1
                    continue
                with clock.phase("h2d_base"):
                    base = raw_base.to(device=device, dtype=torch.float64)
                with clock.phase("base_svd"):
                    frame = thin_svd(base, driver=driver, transpose_wide=transpose_wide)
                target_stats, hits, calls = {}, 0, 1
                for path in targets:
                    check_stop()
                    identity = dict(
                        version=2,
                        tensor=key,
                        base=inputs[base_path],
                        target=inputs[path],
                        dtype="float64",
                        device=str(device),
                        driver=driver,
                        transpose_wide=transpose_wide,
                        updates=updates,
                        torch_version=str(torch.__version__),
                    )
                    cache_path = Path(cache) / fingerprint(identity) if cache else None
                    store = CheckpointStore(cache_path, identity) if cache_path else nullcontext(None)
                    with store as reference:
                        with clock.phase("cache_read"):
                            facts = reference.json("summary") if reference is not None else None
                        if facts is None:
                            with clock.phase("h2d_target"):
                                target = raw[path].to(device=device, dtype=torch.float64)
                            with clock.phase("target_svdvals"):
                                singular = (
                                    frame[1]
                                    if path == base_path
                                    else singular_values(target, driver=driver, transpose_wide=transpose_wide)
                                )
                            calls += int(path != base_path)
                            with clock.phase("iso_projection"):
                                facts = dict(
                                    **iso_scalars(base, target, frame, singular), singular_values=singular.tolist()
                                )
                            if updates:
                                with clock.phase("update_svdvals"):
                                    delta_singular = singular_values(
                                        target - base, driver=driver, transpose_wide=transpose_wide
                                    )
                                calls += 1
                                facts["update_spectrum"] = spectrum_summary(
                                    target - base, singular_values=delta_singular
                                )
                                facts["update_spectrum"]["singular_values"] = delta_singular.tolist()
                            if cache_path:
                                with clock.phase("cache_publish"):
                                    reference.save_json("summary", facts)
                            del target, singular
                        else:
                            hits += 1
                    target_stats[path] = facts
                pairs = []
                for job in paired:
                    points = {"endpoint": job["endpoint"]}
                    if stages:
                        points.update(job["gradient_stages"])
                    for stage, path in points.items():
                        item = dict(
                            job_id=job["job_id"],
                            teacher=job["teacher"],
                            stage=stage,
                            target=path,
                            teacher_path=job["teacher_path"],
                        )
                        if updates:
                            with clock.phase("missing_svdvals"):
                                missing = raw[job["teacher_path"]].to(device=device, dtype=torch.float64) - raw[
                                    path
                                ].to(device=device, dtype=torch.float64)
                                singular = singular_values(missing, driver=driver, transpose_wide=transpose_wide)
                                item["missing_spectrum"] = spectrum_summary(missing, singular_values=singular)
                                item["missing_spectrum"]["singular_values"] = singular.tolist()
                                del missing, singular
                            calls += 1
                        pairs.append(item)
                record = dict(
                    tensor=key,
                    shape=list(base.shape),
                    base=base_path,
                    inputs=inputs,
                    targets=target_stats,
                    pairs=pairs,
                    reference_cache_hits=hits,
                    svd_calls=calls,
                    phases_seconds=dict(clock.seconds),
                    compute_wall_seconds=time.perf_counter() - start,
                )
                with clock.phase("result_publish"):
                    results.save_json(record_key, record)
                for phase, seconds in clock.seconds.items():
                    totals[phase] += seconds
                completed += 1
                del frame, base, raw, raw_base
                dump(
                    output / "progress.json",
                    dict(completed=completed, skipped=skipped, tensor=key, wall_seconds=time.perf_counter() - started),
                )
        results.save_json("complete", dict(completed=completed, skipped=skipped))
    dump(
        output / "performance.json",
        dict(
            wall_seconds=time.perf_counter() - started,
            initialization_seconds=initialization_seconds,
            phases_seconds=dict(totals),
            completed=completed,
            skipped=skipped,
            note="Synchronized phase timing; wall includes initialization, publication and bookkeeping",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--job-ids", type=int, nargs="+")
    p.add_argument("--pattern", default=DEFAULT_SCOPE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--driver", choices=["gesvd", "gesvdj"])
    p.add_argument("--transpose-wide", action="store_true")
    p.add_argument("--updates", action="store_true")
    p.add_argument("--stages", action="store_true")
    p.add_argument("--cache")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-paths", nargs="*", default=[])
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    a = vars(p.parse_args())
    torch.set_num_threads(a.pop("threads"))
    scan(**a)


if __name__ == "__main__":
    main()
