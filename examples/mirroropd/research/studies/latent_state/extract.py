# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay saved prefixes with one decoder load and resumable signature shards."""

import inspect
import time
from pathlib import Path

import numpy as np

from examples.mirroropd.research.core.io import read

from .artifacts import cached, digest, fingerprint, model_identity, publish
from .data import normalize, read_selected
from .geometry import state_positions


def batches(rows, token_budget, max_batch):
    """Bound padded tokens; a sequence longer than the budget is an explicit error."""
    batch = []
    for row in sorted(rows, key=lambda r: len(r["tokens"])):
        if len(row["tokens"]) > token_budget:
            raise ValueError("Token budget must accommodate the longest complete saved prefix")
        if batch and (len(batch) == max_batch or len(row["tokens"]) * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def signatures(model, rows, device, pad_id):
    """Left padding with explicit position IDs; transfer only selected final-norm states."""
    import torch

    width = max(len(row["tokens"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    batch_indices, positions = [], []
    for i, row in enumerate(rows):
        offset = width - len(row["tokens"])
        ids[i, offset:] = torch.tensor(row["tokens"], device=device)
        mask[i, offset:] = 1
        positions.extend(offset + p for p in row["positions"])
        batch_indices.extend([i] * len(row["positions"]))
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        hidden = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False).last_hidden_state
        result = hidden[batch_indices, positions].float().cpu().numpy()
    return result


def alignment_check(model, row, device, pad_id):
    """A full replay and a truncated causal prefix must encode the same state."""
    index = min(1, len(row["positions"]) - 1)
    position = row["positions"][index]
    prefix = dict(row, tokens=row["tokens"][: position + 1], positions=[position])
    a = signatures(model, [row], device, pad_id)[index]
    b = signatures(model, [prefix], device, pad_id)[0]
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if cosine < 0.999:
        raise ValueError(f"Causal prefix alignment failed: cosine={cosine}")
    return cosine


def task_key(plan, pair, task, settings):
    return fingerprint(
        dict(
            schema="final-norm-prefix-v3",
            teacher=plan["pairs"][pair]["teacher"],
            task=task,
            seed=plan["seed"],
            settings=settings,
            code_sha256=digest(__file__),
            adapter_sha256=digest(Path(__file__).with_name("data.py")),
            positions_sha256=fingerprint(inspect.getsource(state_positions)),
        )
    )


def extract(plan_path, pair, *, device="cuda:0", token_budget=65536, max_batch=8):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    plan_path = Path(plan_path)
    plan = read(plan_path)
    spec = plan["pairs"][pair]
    if model_identity(spec["teacher"]["path"]) != spec["teacher"]:
        raise ValueError("Teacher files changed since prepare; prepare a new study")
    settings = dict(
        dtype="bfloat16",
        attention="sdpa",
        token_budget=token_budget,
        max_batch=max_batch,
        torch=torch.__version__,
        transformers=transformers.__version__,
    )
    pending = []
    for task in spec["tasks"]:
        path = plan_path.parent / "signatures" / pair / f"{task['id']}.npz"
        key = task_key(plan, pair, task, settings)
        if not cached(path, key):
            pending.append((task, path, key))
    if not pending:
        return
    model = (
        AutoModel.from_pretrained(
            spec["teacher"]["path"], dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["teacher"]["path"], local_files_only=True)
    pad_id = model.config.pad_token_id or 0
    checked = None
    for task, path, key in pending:
        if digest(task["source"]) != task["source_sha256"]:
            raise ValueError(f"Saved rollout changed: {task['source']}")
        rows = [normalize(row, task, plan["seed"]) for row in read_selected(task)]
        if checked is None:
            checked = alignment_check(model, rows[0], device, pad_id)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        vectors, metadata = [], []
        for batch in batches(rows, token_budget, max_batch):
            vectors.append(signatures(model, batch, device, pad_id))
            for row in batch:
                for rank, position in enumerate(row["positions"]):
                    metadata.append(
                        dict(
                            trajectory=row["trajectory_id"],
                            group=row["group"],
                            outer=row["outer"],
                            prompt=row["prompt_key"],
                            split=row["split"],
                            rank=rank,
                            action=position - row["prompt_tokens"] + 1,
                            response_tokens=row["response_tokens"],
                            prompt_text=tokenizer.decode(
                                row["tokens"][: row["prompt_tokens"]], skip_special_tokens=True
                            )[-600:],
                            context=tokenizer.decode(
                                row["tokens"][max(row["prompt_tokens"], position - 64) : position + 1],
                                skip_special_tokens=True,
                            ),
                        )
                    )
        elapsed = time.monotonic() - start
        arrays = dict(features=np.concatenate(vectors), **{k: np.array([m[k] for m in metadata]) for k in metadata[0]})
        tokens = sum(len(r["tokens"]) for r in rows)
        publish(
            path,
            key,
            arrays,
            settings=settings,
            source=task,
            trajectories=len(rows),
            elapsed_seconds=elapsed,
            tokens=tokens,
            tokens_per_second=tokens / elapsed,
            alignment_cosine=checked,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )
        print(f"{pair} {task['group']} {task['id']} {len(rows)} responses {elapsed:.1f}s", flush=True)
