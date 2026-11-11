# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared model execution and durable, input-addressed teacher caches.

Paths identify immutable checkpoints. Cache keys additionally include weight
file metadata so replacing a local shard invalidates cached hidden states.
All diagnostic students use FP32 parameters/head and BF16 backbone autocast;
teachers use BF16 parameters on CUDA with an FP32 output projection.
"""

import gc
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from examples.mirroropd.research.backends.hf_models import load
from examples.mirroropd.research.core.io import dump

from .inputs import source_fingerprint
from .metrics import tensor_norm_sq


def write_json(path, value):
    # Undefined normalized diagnostics retain NaN, as in legacy PAC artifacts.
    dump(path, value, allow_nan=True)


def write_tensors(path, value):
    """Publish a complete tensor artifact atomically for resumable workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_identity(path):
    """Validate every indexed weight shard without loading model tensors."""
    directory = Path(path).resolve()
    index = directory / "model.safetensors.index.json"
    names = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    )
    files = [directory / name for name in ["config.json", *names]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Model export has missing files: {missing}")
    return dict(
        path=str(directory),
        files=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files],
    )


def load_model(path, *, trainable=True, checkpointing=False, device="cuda"):
    device = torch.device(device)
    model = load(path=path, trainable=trainable, device=device, attention="sdpa", trust_remote_code=True)
    if trainable:
        mtp = getattr(model.model, "mtp_layers", None)
        if mtp is not None:
            mtp.requires_grad_(False)
    if trainable and checkpointing:
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
            raise ValueError("Checkpointed diagnostic requires zero dropout")
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def collect_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hidden_states(model, tokens):
    device = next(model.parameters()).device
    ids = torch.tensor([tokens], device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model.model(input_ids=ids, use_cache=False).last_hidden_state[0].float()


def response_hidden(model, row):
    """Frozen teacher states aligned with every response's prediction position."""
    with torch.no_grad():
        hidden = hidden_states(model, row["prompt_ids"] + row["response_ids"][:-1])
    return hidden[len(row["prompt_ids"]) - 1 :].cpu()


def prefix_logits(model, row, position):
    hidden = hidden_states(model, row["prompt_ids"] + row["response_ids"][:position])[-1]
    return torch.nn.functional.linear(hidden, model.lm_head.weight.float())


def prefix_batch_logits(model, prefixes):
    """Right-pad fixed prefixes and select each last real hidden state."""
    device = next(model.parameters()).device
    tokens = [row["prompt_ids"] + row["response_ids"][:position] for row, position in prefixes]
    lengths = torch.tensor([len(t) for t in tokens], device=device)
    ids = torch.zeros((len(tokens), int(lengths.max())), device=device, dtype=torch.long)
    for index, values in enumerate(tokens):
        ids[index, : len(values)] = torch.tensor(values, device=device)
    mask = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = hidden[torch.arange(len(tokens), device=device), lengths - 1].float()
    return torch.nn.functional.linear(last, model.lm_head.weight.float())


def teacher_cache(root, path, rows, *, device="cuda", student=False):
    identity = dict(
        model=checkpoint_identity(path),
        records=source_fingerprint(rows),
        protocol="student-fp32-master-bf16-autocast-v1" if student else "native-bf16-fp32-head-v1",
        device_type=torch.device(device).type,
        torch_version=str(torch.__version__),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = Path(root) / "teacher-v1" / f"{key}.pt"
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        if payload.get("identity") != identity:
            raise ValueError(f"Teacher cache identity mismatch: {destination}")
        return payload
    teacher = load_model(path, trainable=student, device=device)
    payload = dict(
        identity=identity,
        hidden={row["row_id"]: response_hidden(teacher, row) for row in rows},
        head=teacher.lm_head.weight.detach().float().cpu(),
    )
    del teacher
    collect_memory()
    write_tensors(destination, payload)
    return payload


@contextmanager
def gradient_norms(model):
    """Leaf hooks measure each backward's exact incoming norm before accumulation."""
    buffer = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)

    def capture(gradient):
        with torch.no_grad():
            buffer.add_(tensor_norm_sq(gradient))

    handles = [p.register_hook(capture) for p in model.parameters() if p.requires_grad]
    try:
        yield buffer
    finally:
        for handle in handles:
            handle.remove()


def flatten_grad(model, *, scale=1.0):
    """Copy model-order gradients into one FP32 CPU vector, including zeros."""
    parameters = list(model.parameters())
    return flatten_tensors(parameters, [p.grad for p in parameters], scale=scale)


def flatten_tensors(parameters, tensors, *, scale=1.0):
    """Pack parameter-shaped tensors (or missing zeros) using bounded DMA."""
    vector = torch.empty(sum(p.numel() for p in parameters), dtype=torch.float32, device="cpu")
    if parameters and parameters[0].is_cuda:
        # Bound accelerator workspace even for 7B students. Pack parameter slices
        # into <=1 GiB DMA blocks and normalize before copying to pageable RAM.
        width = 268435456
        pieces, filled, offset = [], 0, 0
        for parameter, value in zip(parameters, tensors, strict=True):
            gradient = None if value is None else value.detach().reshape(-1)
            for start in range(0, parameter.numel(), width):
                block = (
                    parameter.new_zeros(min(width, parameter.numel() - start))
                    if gradient is None
                    else gradient[start : start + width]
                )
                while block.numel():
                    take = min(width - filled, block.numel())
                    pieces.append(block[:take])
                    filled += take
                    block = block[take:]
                    if filled == width:
                        packed = torch.cat(pieces).mul_(scale)
                        vector[offset : offset + filled].copy_(packed)
                        del packed
                        offset += filled
                        pieces, filled = [], 0
        if pieces:
            packed = torch.cat(pieces).mul_(scale)
            vector[offset : offset + filled].copy_(packed)
        return vector
    offset = 0
    for parameter, value in zip(parameters, tensors, strict=True):
        part = vector[offset : offset + parameter.numel()]
        if value is None:
            part.zero_()
        else:
            part.copy_(value.detach().reshape(-1))
        offset += parameter.numel()
    return vector.mul_(scale)
