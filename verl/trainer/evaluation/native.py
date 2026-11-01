# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared original-token capture, balanced history and native TQ transport."""

import hashlib
import json
import uuid

import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from .artifacts import identity


def priority(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], separators=(",", ":")).encode()).hexdigest()


def balanced_history(batches, capacity, seed):
    """Deterministic random priorities; no loss-dependent selection."""
    if not batches:
        return []
    count = len(batches)
    if count > capacity:
        raise ValueError("History capacity must cover every observed batch")
    q, r = divmod(capacity, count)
    extras = set(sorted(batches, key=lambda b: priority(seed, "batch", b))[:r])
    result = []
    for batch_id, rows in sorted(batches.items()):
        quota = q + (batch_id in extras)
        if len(rows) < quota:
            raise ValueError("A training batch cannot fill its history quota")
        selected = sorted(rows, key=lambda x: priority(seed, "sample", batch_id, x["source_id"]))[:quota]
        result.extend(dict(row, history_batch=batch_id, history_quota=quota) for row in selected)
    return result


def capture_rows(batch, outer):
    fields = ["prompts", "responses", "response_mask", "extra_info"]
    # KVBatchMeta.fields=None means an unprojected batch. Native rollout rows
    # still contain metrics; it is not an empty list of available columns.
    available_fields = getattr(batch, "fields", None)
    if available_fields is None or "metrics" in available_fields:
        fields.append("metrics")
    data = tq.kv_batch_get(
        keys=list(batch.keys),
        partition_id=batch.partition_id,
        select_fields=fields,
    )
    info = list(data["extra_info"])
    observations = list(data["metrics"]) if "metrics" in data else [{} for _ in info]
    rows = []
    for i, (prompt, response, mask) in enumerate(
        zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
    ):
        tag = batch.tags[i]
        if tag.get("is_padding", False):
            continue
        source_id = str(info[i].get("index", identity(prompt.tolist())))
        rows.append(
            dict(
                id=f"{outer}:{source_id}",
                source_id=source_id,
                outer=outer,
                prompt_ids=prompt.tolist(),
                response_ids=response.tolist(),
                response_mask=mask.tolist(),
                rollout_key=batch.keys[i],
                min_global_steps=tag.get("min_global_steps"),
                max_global_steps=tag.get("max_global_steps"),
                data_source=info[i].get("data_source"),
                finish_reason=observations[i].get("finish_reason", tag.get("finish_reason")),
            )
        )
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate nonpadding source IDs in a rollout batch")
    return rows


def put_fixed_rows(rows, outer):
    partition = f"rkl-readout-{outer}-{uuid.uuid4().hex}"
    keys = [f"row-{i}" for i in range(len(rows))]
    values = {k: [] for k in ["prompts", "responses", "input_ids", "position_ids", "response_mask", "loss_mask"]}
    tags = []
    for row in rows:
        prompt, response = row["prompt_ids"], row["response_ids"]
        mask = row.get("response_mask", [1] * len(response))
        if not prompt or not response or len(mask) != len(response):
            raise ValueError("Empty or inconsistent fixed rollout")
        ids = prompt + response
        for k, v in dict(
            prompts=prompt,
            responses=response,
            input_ids=ids,
            position_ids=list(range(len(ids))),
            response_mask=mask,
            loss_mask=mask,
        ).items():
            values[k].append(torch.tensor(v, dtype=torch.int64))
        tags.append(dict(seq_len=len(ids), prompt_len=len(prompt), response_len=len(response), status="success"))
    fields = TensorDict(
        {k: torch.nested.nested_tensor(v, layout=torch.jagged) for k, v in values.items()}, batch_size=[len(rows)]
    )
    tq.kv_batch_put(keys=keys, partition_id=partition, tags=tags, fields=fields)
    return KVBatchMeta(keys=keys, partition_id=partition, tags=tags, fields=list(values)), dict(
        zip(keys, rows, strict=True)
    )
