# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One-model-at-a-time frozen-bank extraction; no gradients or optimizer state."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.mirroropd.research.core.artifacts import file_hash
from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.teacher_difference.inputs import (
    load_records,
    normalize_records,
    source_fingerprint,
    validate_records,
)
from examples.mirroropd.research.studies.teacher_difference.runtime import (
    checkpoint_identity,
    collect_memory,
    load_model,
)


@torch.no_grad()
def extract(
    model_path, records, output, *, cap=1024, positions=16, chunk=16, device="cuda", precision="master", layers=()
):
    """Sample uniformly spaced prediction positions per question, including endpoints.

    Log probabilities use the entire native vocabulary and FP32 head projection.
    Layer hooks capture post-block (pre-final-norm) states; final states are post
    norm. A complete marker is published only after every memmap is flushed.
    """
    from transformers import AutoTokenizer

    if cap < 1 or positions < 1 or chunk < 1 or precision not in ("master", "native"):
        raise ValueError("Positive cap/positions/chunk and precision master/native are required")
    rows = normalize_records(load_records(records), cap=cap)
    validate_records(rows)
    if any(not row["prompt_ids"] or not row.get("split") for row in rows):
        raise ValueError("Every question needs a nonempty prompt and an explicit split")
    selected = [
        np.unique(
            np.linspace(0, len(r["response_ids"]) - 1, min(positions, len(r["response_ids"])), dtype=int)
        ).tolist()
        for r in rows
    ]
    positions_meta = [
        dict(row_id=r["row_id"], split=r["split"], position=p)
        for r, points in zip(rows, selected, strict=True)
        for p in points
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokens = tokenizer.get_vocab()
    vocabulary = hashlib.sha256(json.dumps(sorted(tokens.items()), ensure_ascii=False).encode()).hexdigest()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / "vocabulary.json", tokens)
    identity = checkpoint_identity(model_path)
    model = load_model(model_path, trainable=precision == "master", device=device)
    model.requires_grad_(False)
    head = model.get_output_embeddings()
    # Existing backend validates a bias-free head. Promote exactly once, avoiding
    # a fresh full-head FP32 allocation for each position block.
    projection = head.weight.detach().float()
    device = projection.device
    n, vocab = len(positions_meta), head.weight.shape[0]
    hidden_width = head.weight.shape[1]
    logprobs = np.lib.format.open_memmap(destination / "logprobs.npy", mode="w+", dtype="float32", shape=(n, vocab))
    states = {
        "final": np.lib.format.open_memmap(
            destination / "hidden-final.npy", mode="w+", dtype="float32", shape=(n, hidden_width)
        )
    }
    layers = sorted(set(layers))
    decoder_layers = getattr(model.model, "layers", ())
    if any(i < 0 or i >= len(decoder_layers) for i in layers):
        raise ValueError("Requested layer outside decoder layer range")
    states.update(
        {
            str(i): np.lib.format.open_memmap(
                destination / f"hidden-layer-{i}.npy", mode="w+", dtype="float32", shape=(n, hidden_width)
            )
            for i in layers
        }
    )
    metadata = dict(
        protocol="spectral-full-vocab-v1",
        model=identity,
        model_sha256={item["name"]: file_hash(Path(model_path) / item["name"]) for item in identity["files"]},
        bank=source_fingerprint(rows),
        positions=positions_meta,
        vocabulary=vocabulary,
        shape=[n, vocab],
        precision=precision,
        device_type=device.type,
        backbone_autocast="BF16 on CUDA; FP32 on CPU",
        head="FP32",
        layers=layers,
        hidden_semantics="layer: post-block/pre-final-norm; final: post-final-norm",
        torch_version=str(torch.__version__),
        complete=False,
    )
    dump(destination / "metadata.json", metadata)
    handles, current = [], {}

    def hook_for(name):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            states[name][current["start"] : current["end"]] = value[0, current["indices"]].float().cpu().numpy()

        return capture

    try:
        for i in layers:
            handles.append(decoder_layers[i].register_forward_hook(hook_for(str(i))))
        offset = 0
        for row, points in zip(rows, selected, strict=True):
            prediction_indices = [len(row["prompt_ids"]) - 1 + p for p in points]
            current.update(start=offset, end=offset + len(points), indices=prediction_indices)
            tokens = row["prompt_ids"] + row["response_ids"][:-1]
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, prediction_indices].float()
            states["final"][offset : offset + len(points)] = hidden.cpu().numpy()
            for start in range(0, len(points), chunk):
                logs = torch.nn.functional.linear(hidden[start : start + chunk], projection).log_softmax(-1)
                logprobs[offset + start : offset + start + len(logs)] = logs.cpu().numpy()
            offset += len(points)
        logprobs.flush()
        for state in states.values():
            state.flush()
        dump(
            destination / "metadata.json",
            dict(metadata, complete=True, sha256={p.name: file_hash(p) for p in destination.glob("*.npy")}),
        )
    finally:
        for handle in handles:
            handle.remove()
        del model, head, projection
        collect_memory()
