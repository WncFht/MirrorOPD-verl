# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only projection of preserved R1 Adam moments onto audited spectral directions.

Stored moments imply a local Adam direction, not an observed adjacent-checkpoint
increment. No optimizer is stepped and no model weights are changed.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from examples.mirroropd.research.core.io import dump


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    torch.set_num_threads(2)
    manifest = json.loads((root / "manifest.json").read_text())
    old = json.loads((args.inventory).read_text())
    conditions = {c["run_id"]: c for c in old["conditions"]}
    pending = []
    for job in manifest["jobs"]:
        c = conditions[job["run_id"]]
        if job["family"] != "R1" or not c.get("source_protocol"):
            continue
        folder = Path(c["source_protocol"]).parent / "resume"
        if (folder / "optimizer.pt").exists():
            pending.append((job, folder))
    output = root / "optimizer-audit"
    while pending and not (root / "STOP").exists():
        for job, folder in list(pending):
            out = root / "jobs" / f"{job['job_id']:03d}" / "attempt-001"
            if not (out / "matrix-index.json").exists():
                if (root / "exit.json").exists():
                    dump(output / f"{job['job_id']:03d}-missing.json", dict(reason="Matrix decomposition unavailable"))
                    pending.remove((job, folder))
                continue
            receipt = json.loads((folder / "complete.json").read_text())
            statefile = folder / "optimizer.pt"
            if receipt["step"] != job["step"]:
                dump(
                    output / f"{job['job_id']:03d}-skipped.json",
                    dict(
                        reason="Optimizer/checkpoint step mismatch",
                        optimizer_step=receipt["step"],
                        checkpoint_step=job["step"],
                    ),
                )
                pending.remove((job, folder))
                continue
            checkpoint = Path(job["endpoint"])
            weight_receipt = json.loads((checkpoint / "complete.json").read_text())
            if (weight_receipt["step"], weight_receipt["weight_identity"]) != (
                receipt["step"],
                receipt["weight_identity"],
            ):
                raise ValueError("Historical Adam state is not bound to this endpoint")
            verified_weights = []
            for item in weight_receipt["files"]:
                path = checkpoint / item["name"]
                if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise ValueError("Endpoint shard failed historical optimizer identity check")
                verified_weights.append(item)
            if statefile.stat().st_size != receipt["bytes"] or sha256(statefile) != receipt["sha256"]:
                raise ValueError("Historical Adam payload failed checksum")
            payload = torch.load(statefile, weights_only=True, mmap=True, map_location="cpu")
            if any(payload[k] != receipt[k] for k in ["step", "signature", "weight_identity"]):
                raise ValueError("Historical Adam payload/receipt mismatch")
            mapping = {}
            for names, group in zip(payload["parameter_names"], payload["optimizer"]["param_groups"], strict=True):
                for name, identifier in zip(names, group["params"], strict=True):
                    mapping[name] = (payload["optimizer"]["state"][identifier], group)
            records = json.loads((out / "matrix-index.json").read_text())
            rows = []
            for r in records:
                slot, group = mapping[r["key"]]
                step = int(slot["step"])
                if step != payload["step"]:
                    raise ValueError("Historical Adam slot step differs")
                beta1, beta2 = group["betas"]
                m = slot["exp_avg"].double() / (1 - beta1**step)
                v = slot["exp_avg_sq"].double() / (1 - beta2**step)
                descent = -group["lr"] * m / (v.sqrt() + group["eps"])
                directions = torch.load(out / "directions" / f"{r['index']:04d}.pt", weights_only=True, mmap=True)
                projections = {}
                for name in ["learned", "off", "null"]:
                    direction = directions[name].double()
                    norm = float(direction.norm())
                    dot = float((direction * descent).sum())
                    projections[name] = dict(dot=dot, direction_norm=norm, unit_projection=dot / norm if norm else None)
                rows.append(
                    dict(key=r["key"], step=step, adam_descent_norm=float(descent.norm()), projections=projections)
                )
                dump(output / f"{job['job_id']:03d}-progress.json", dict(done=len(rows), total=len(records)))
            dump(
                output / f"{job['job_id']:03d}.json",
                dict(
                    run_id=job["run_id"],
                    optimizer_file=str(statefile),
                    optimizer_sha256=receipt["sha256"],
                    checkpoint=job["endpoint"],
                    scope=(
                        "stored bias-corrected Adam moment direction; excludes weight decay; "
                        "not raw gradient or observed increment"
                    ),
                    weight_identity=payload["weight_identity"],
                    verified_endpoint_shards=verified_weights,
                    rows=rows,
                ),
            )
            pending.remove((job, folder))
            del payload, mapping
        dump(output / "status.json", dict(pending=[j["run_id"] for j, _ in pending], updated_at=time.time()))
        if pending:
            time.sleep(20)
    dump(output / "exit.json", dict(complete=not pending, stopped=(root / "STOP").exists(), finished_at=time.time()))


if __name__ == "__main__":
    main()
