# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact matrix decompositions and explicit intervention conditions."""

import json
import math
import time

import torch

from examples.mirroropd.research.core.io import dump

from ..decomposition import frame_components
from ..weights import TensorCheckpoint, tensor_digest
from .protocol import SCOPE, save_tensor


def variants():
    result = []
    for fraction in [0.01, 0.05]:
        for restore in [False, True]:
            result.append(
                dict(name=f"top-{fraction}-restore-{restore}", kind="top", fraction=fraction, restore=restore)
            )
    for band in ["off", "null"]:
        for beta in [0.25, 0.5, 1.0]:
            result.append(dict(name=f"missing-{band}-beta-{beta}", kind="missing", band=band, beta=beta))
        for control in ["gaussian", "random_frames"]:
            for seed in range(3):
                result.append(
                    dict(name=f"missing-{band}-{control}-{seed}", kind="control", band=band, seed=seed, control=control)
                )
    for depth in range(3):
        result.append(dict(name=f"missing-off-depth-{depth}", kind="depth", band="off", depth=depth, beta=1.0))
    return result


def prepare_directions(context, job, out, smoke=False):
    b, t, s = [TensorCheckpoint(job[k]) for k in ["base", "teacher_path", "endpoint"]]
    keys = b.selected(SCOPE)
    if smoke:
        keys = [k for k in keys if k.endswith("self_attn.k_proj.weight")][:1]
    records = []
    for index, key in enumerate(keys):
        context.check_stop()
        dest = out / "directions" / f"{index:04d}.pt"
        if dest.exists():
            records.append(json.loads(dest.with_suffix(".json").read_text()))
            continue
        started = time.monotonic()
        raw = [c.get(key) for c in [b, t, s]]
        wb, wt, ws = [x.cuda().double() for x in raw]
        learned, missing = ws - wb, wt - ws
        u, singular, vh = torch.linalg.svd(learned, full_matrices=False)
        parts = frame_components(wb, missing)
        # A square thin-SVD spans the complete matrix space; its null is structural zero.
        if wb.shape[0] == wb.shape[1]:
            parts["null"].zero_()
        reconstruction_error = (
            float((((u * singular) @ vh) - learned).norm() / learned.norm()) if learned.norm() > 0 else 0.0
        )
        decomposition_error = (
            float((sum(parts.values()) - missing).norm() / missing.norm()) if missing.norm() > 0 else 0.0
        )
        if max(reconstruction_error, decomposition_error) > 1e-8:
            raise ValueError("Exact SVD reconstruction/decomposition gate failed")
        data = dict(learned=learned.float().cpu())
        for frac in [0.01, 0.05]:
            rank = math.ceil(frac * len(singular))
            top = (u[:, :rank] * singular[:rank]) @ vh[:rank]
            data[f"top-{frac}"] = top.float().cpu()
        for band in ["off", "null"]:
            part = parts[band]
            data[band] = part.float().cpu()
            if part.double().norm() > 1e-15:
                sigma = torch.linalg.svdvals(part)
                for seed in range(3):
                    g = torch.Generator(device="cuda").manual_seed(20261110 + 1000 * seed + index)
                    a = torch.linalg.qr(
                        torch.randn((part.shape[0], len(sigma)), generator=g, device="cuda", dtype=torch.float64),
                        mode="reduced",
                    ).Q
                    z = torch.linalg.qr(
                        torch.randn((part.shape[1], len(sigma)), generator=g, device="cuda", dtype=torch.float64),
                        mode="reduced",
                    ).Q
                    data[f"{band}-random_frames-{seed}"] = ((a * sigma) @ z.T).float().cpu()
            else:
                for seed in range(3):
                    data[f"{band}-random_frames-{seed}"] = torch.zeros_like(data[band])
        record = dict(
            svd_reconstruction_error=reconstruction_error,
            frame_decomposition_error=decomposition_error,
            index=index,
            key=key,
            shape=list(wb.shape),
            input_sha256=[tensor_digest(x) for x in raw],
            learned_energy=float(learned.square().sum()),
            off_energy=float(parts["off"].square().sum()),
            null_energy=float(parts["null"].square().sum()),
            elapsed_seconds=time.monotonic() - started,
        )
        save_tensor(dest, data)
        dump(dest.with_suffix(".json"), record)
        records.append(record)
        dump(out / "decomposition-progress.json", dict(done=len(records), total=len(keys), latest=record))
        del data, raw, wb, wt, ws, learned, missing, parts, u, singular, vh
    dump(out / "matrix-index.json", records)
    return records


def apply(context, model, job, out, records, variant):
    base, student = TensorCheckpoint(job["base"]), TensorCheckpoint(job["endpoint"])
    maximum_layer = max(int(r["key"].split(".")[2]) for r in records)
    edits = []
    with torch.no_grad():
        for r in records:
            context.check_stop()
            key = r["key"]
            parameter = model.get_parameter(key)
            original = student.get(key).cuda().float()
            data = torch.load(out / "directions" / f"{r['index']:04d}.pt", weights_only=True, mmap=True)
            if variant["kind"] == "top":
                change = data[f"top-{variant['fraction']}"].cuda()
                if variant["restore"] and change.norm() > 0:
                    change *= math.sqrt(r["learned_energy"]) / change.double().norm()
                value = base.get(key).cuda().float() + change
            else:
                change = data[variant["band"]].cuda()
                if variant["kind"] == "control":
                    if variant["control"] == "random_frames":
                        change = data[f"{variant['band']}-random_frames-{variant['seed']}"].cuda()
                    else:
                        generator = torch.Generator(device="cuda").manual_seed(
                            20261110 + 1000 * variant["seed"] + r["index"]
                        )
                        noise = torch.randn(change.shape, device="cuda", generator=generator)
                        change = noise * (change.double().norm() / noise.double().norm())
                if variant["kind"] == "depth":
                    depth = min(2, int(key.split(".")[2]) * 3 // (maximum_layer + 1))
                    if depth != variant["depth"]:
                        change.zero_()
                value = original + variant.get("beta", 1.0) * change
            parameter.copy_(value)
            edits.append(dict(key=key, displacement=float((value - original).double().norm())))
    dump(out / "edit-receipts" / (variant["name"] + ".json"), edits)
