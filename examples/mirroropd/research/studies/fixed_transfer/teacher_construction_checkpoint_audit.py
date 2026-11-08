# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only CPU audit of this experiment's complete, locally authored checkpoints.

The default validates model headers, optimizer coverage/steps and restorable RNG
metadata without scanning every moment tensor. --scan additionally reads all
model/moment values and hashes the files. Neither mode loads a model or uses CUDA.
Only audit trusted training-state.pt files produced by the frozen local trainer;
their NumPy and Python RNG objects require torch.load(weights_only=False).
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import struct
from collections import Counter
from pathlib import Path

import numpy as np


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def model_headers(path):
    tensors, files = {}, {}
    for file in sorted(path.glob("model*.safetensors")):
        with file.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {file}")
            length = struct.unpack("<Q", size_bytes)[0]
            if not 0 < length < 32 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header size: {file}")
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"Truncated safetensors header: {file}")
        header = json.loads(raw)
        for name, value in header.items():
            if name == "__metadata__":
                continue
            shape = value["shape"]
            start, end = value["data_offsets"]
            if name in tensors or value["dtype"] != "F32" or not 0 <= start <= end:
                raise ValueError(f"Duplicate/non-FP32/invalid model tensor: {name}")
            if end - start != math.prod(shape) * 4 or end + 8 + length > file.stat().st_size:
                raise ValueError(f"Truncated or malformed tensor: {file}/{name}")
            tensors[name] = dict(shape=tuple(shape), file=file.name)
        files[file.name] = dict(header_sha256=hashlib.sha256(size_bytes + raw).hexdigest())
    if not tensors:
        raise ValueError(f"No model tensors in {path}")
    index = path / "model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text())["weight_map"]
        if set(weights) != set(tensors) or any(weights[k] != v["file"] for k, v in tensors.items()):
            raise ValueError("Model shard index does not cover the tensor headers")
        files[index.name] = dict(sha256=file_hash(index))
    return tensors, files


def finite_tensor(tensor, label):
    import torch

    # Keep temporary allocations bounded even for the vocabulary projection.
    for block in tensor.reshape(-1).split(4 * 1024 * 1024):
        if not bool(torch.isfinite(block).all()):
            raise ValueError(f"Nonfinite checkpoint tensor: {label}")


def audit_checkpoint(path, *, expected_step, require_sampler=False, scan=False):
    import torch

    path = Path(path)
    marker = json.loads((path / "checkpoint-complete.json").read_text())
    if marker != dict(step=expected_step, parameter_dtype="float32"):
        raise ValueError("Checkpoint-complete marker identity mismatch")
    watched = sorted([*path.glob("model*.safetensors"), path / "training-state.pt"])
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
    tensors, files = model_headers(path)
    state = torch.load(path / "training-state.pt", map_location="cpu", mmap=True, weights_only=False)
    if state["step"] != expected_step:
        raise ValueError("Training state step differs from completion marker")
    optimizer = state["optimizer"]
    parameters = [key for group in optimizer["param_groups"] for key in group["params"]]
    if len(parameters) != len(set(parameters)) or set(parameters) != set(optimizer["state"]):
        raise ValueError("Optimizer parameter groups do not cover the saved states")
    shapes = []
    for key, value in optimizer["state"].items():
        if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Unexpected Adam state keys for parameter {key}")
        step = value["step"]
        if step.numel() != 1 or step.dtype != torch.float32 or float(step.item()) != expected_step:
            raise ValueError(f"Adam update count mismatch for parameter {key}")
        average, square = value["exp_avg"], value["exp_avg_sq"]
        if average.dtype != torch.float32 or square.dtype != torch.float32 or average.shape != square.shape:
            raise ValueError(f"Adam moment dtype/shape mismatch for parameter {key}")
        shapes.append(tuple(average.shape))
        if scan:
            finite_tensor(average, f"optimizer/{key}/exp_avg")
            finite_tensor(square, f"optimizer/{key}/exp_avg_sq")
    if Counter(shapes) != Counter(v["shape"] for v in tensors.values()):
        raise ValueError("Optimizer shapes do not cover all model parameters")
    for name in ["torch_rng", "cuda_rng"]:
        rng = state[name]
        if rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() < 16:
            raise ValueError(f"Invalid {name}")
    torch.Generator(device="cpu").set_state(state["torch_rng"])
    random.Random().setstate(state["python_rng"])
    np.random.RandomState().set_state(state["numpy_rng"])
    sampler = state["sampler_rng"]
    if require_sampler and sampler is None:
        raise ValueError("SFT candidate sampling RNG is missing")
    if sampler is not None:
        generator = np.random.default_rng()
        generator.bit_generator.state = sampler
    if scan:
        from safetensors import safe_open

        for file in sorted(path.glob("model*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    finite_tensor(handle.get_tensor(name), f"model/{name}")
        for file in watched:
            files.setdefault(file.name, {})["sha256"] = file_hash(file)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
    if before != after:
        raise ValueError("Checkpoint changed while being audited")
    for name, (size, modified) in after.items():
        files.setdefault(name, {}).update(bytes=size, mtime_ns=modified)
    return dict(
        path=str(path.resolve()),
        step=expected_step,
        model_tensors=len(tensors),
        model_parameters=sum(math.prod(v["shape"]) for v in tensors.values()),
        optimizer_parameter_states=len(parameters),
        master_and_moments_dtype="float32",
        torch_rng_bytes=state["torch_rng"].numel(),
        cuda_rng_bytes=state["cuda_rng"].numel(),
        python_numpy_torch_cpu_rng_restore_checked=True,
        sampler_rng_restore_checked=sampler is not None,
        cuda_rng_check="saved byte-vector only; no GPU state mutation",
        mode="full-finite-and-hash-scan" if scan else "metadata-and-rng-only",
        files=files,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scan", action="store_true", help="Read all model/moment tensors and hash files")
    parser.add_argument("--endpoints-only", action="store_true", help="Audit available teacher/student endpoints")
    args = parser.parse_args()
    source_snapshot = args.output.with_suffix(".source.py")
    if args.output.exists() or source_snapshot.exists():
        raise FileExistsError("Use a new audit receipt path")
    protocol = json.loads((args.root / "protocol.json").read_text())
    values = []
    endpoints = {}
    for stage in ["teachers", "opd"]:
        for arm in protocol["arms"]:
            endpoint_step = protocol["opd" if stage == "opd" else "ert" if arm == "ERT-verifier" else "sft"]["updates"]
            endpoint = args.root / stage / arm / f"step{endpoint_step:04d}" / "checkpoint-complete.json"
            endpoints[f"{stage}/{arm}"] = endpoint.exists()
            for path in sorted((args.root / stage / arm).glob("step*")):
                if not (path / "checkpoint-complete.json").exists():
                    continue
                step = int(path.name[4:])
                if args.endpoints_only and step != endpoint_step:
                    continue
                values.append(
                    dict(
                        stage=stage,
                        arm=arm,
                        **audit_checkpoint(
                            path,
                            expected_step=step,
                            require_sampler=stage == "teachers" and arm != "ERT-verifier",
                            scan=args.scan,
                        ),
                    )
                )
    if not values:
        raise ValueError("No complete checkpoints to audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, source_snapshot)
    receipt = dict(
        protocol_sha256=file_hash(args.root / "protocol.json"),
        source_sha256=file_hash(source_snapshot),
        source_snapshot=str(source_snapshot),
        checkpoints=values,
        scope="available-endpoints" if args.endpoints_only else "available-saved-checkpoints",
        endpoint_inventory=endpoints,
        all_endpoint_markers_present=all(endpoints.values()),
    )
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(audited=len(values), scan=args.scan, output=str(args.output))))


if __name__ == "__main__":
    main()
