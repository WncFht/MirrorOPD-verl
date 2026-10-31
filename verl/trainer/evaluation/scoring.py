# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial teacher/student readouts through the native FSDP workers."""

import random
import time

import numpy as np
import torch
import transfer_queue as tq

from verl.experimental.rollout_policy.collector import UnsafeDrainError
from verl.trainer.distillation.readout import put_fixed_rows

from .plan import payload_windows


def score_native_rows(trainer, rows, *, hidden_size, budget, live_payload_bytes=0, load_window=None):
    """Score compatible prefixes, excluding synthetic DP padding from every result.

    The caller parks generation first. No checkpoint or optimizer is created.
    An unresolved remote failure is fail-stop: do not clear a partition while a
    remote writer may still be using it, and do not start the next GPU phase.
    """
    rows = list(rows)
    if not rows:
        return [], {"windows": 0, "seconds": 0.0}
    zero = [r for r in rows if r.get("valid_tokens", sum(r.get("response_mask", ()))) == 0]
    rows = [r for r in rows if r not in zero]
    windows = payload_windows(rows, hidden_size=hidden_size, payload_bytes=budget.available_payload(live_payload_bytes))
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    step = trainer.actor_optimizer_step
    started = time.monotonic()
    timings, results = {}, []
    for row in zero:
        results.append(
            {
                k: row.get(k)
                for k in (
                    "id",
                    "source_id",
                    "bank",
                    "history_batch",
                    "dataset",
                    "domain",
                    "sample_index",
                )
            }
            | {"trajectory_id": row.get("trajectory_id", row["id"]), "rkl_sum": 0.0, "tokens": 0}
        )
    active_rpc, begun = False, False
    audit = None
    try:
        active_rpc = True
        trainer.actor_rollout_wg.begin_rkl_readout(audit_state=budget.audit_state)
        active_rpc, begun = False, True
        for references in windows:
            window = load_window(references) if load_window is not None else references
            meta, by_key = put_fixed_rows(window, trainer.global_steps)
            try:
                meta = trainer._balance_batch(meta, {}, logging_prefix="evaluation_balance", inference_only=True)
                active_rpc = True
                window_timing = {}
                trainer._compute_shared_teacher_log_probs(meta, window_timing)
                for key, value in window_timing.items():
                    timings[key] = timings.get(key, 0.0) + value
                active_rpc = False
                meta.fields = ["input_ids", "position_ids", "prompts", "responses", "loss_mask", "response_mask"]
                meta.extra_info = dict(
                    temperature=1.0,
                    compute_loss=False,
                    calculate_entropy=False,
                    full_vocab_score=True,
                    retain_sample_identity=True,
                    full_vocab_prefetch=True,
                    global_token_num=None,
                    full_vocab_score_token_chunk=budget.token_chunk,
                )
                active_rpc = True
                trainer.actor_rollout_wg.score_full_vocab(meta)
                active_rpc = False
                actual_keys = [key for key, tag in zip(meta.keys, meta.tags, strict=True) if not tag.get("is_padding")]
                if set(actual_keys) != set(by_key) or len(actual_keys) != len(by_key):
                    raise ValueError("Inference balancing changed logical trajectory identities")
                scores = tq.kv_batch_get(
                    keys=actual_keys, partition_id=meta.partition_id, select_fields=["rkl_sum", "rkl_count"]
                )
                for key, total, count in zip(
                    actual_keys, scores["rkl_sum"].unbind(), scores["rkl_count"].unbind(), strict=True
                ):
                    row = by_key[key]
                    if int(count.item()) != sum(row["response_mask"]):
                        raise ValueError("RKL score count differs from original mask")
                    results.append(
                        {
                            k: row.get(k)
                            for k in (
                                "id",
                                "source_id",
                                "bank",
                                "history_batch",
                                "dataset",
                                "domain",
                                "sample_index",
                            )
                        }
                        | {
                            "trajectory_id": row.get("trajectory_id", row["id"]),
                            "rkl_sum": float(total.item()),
                            "tokens": int(count.item()),
                        }
                    )
            finally:
                if not active_rpc:
                    tq.kv_clear(keys=list(meta.keys), partition_id=meta.partition_id)
        if trainer.actor_optimizer_step != step:
            raise RuntimeError("Read-only evaluation advanced the optimizer")
    except BaseException as exc:
        if active_rpc:
            raise UnsafeDrainError(
                "Scoring RPC did not settle; terminate run before model switching or TQ cleanup"
            ) from exc
        raise
    finally:
        try:
            if begun and not active_rpc:
                audit = trainer.actor_rollout_wg.end_rkl_readout()
        finally:
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
    if budget.audit_state and (not audit or not all(x.get("unchanged") and x.get("state_digest") for x in audit)):
        raise RuntimeError("Requested RKL audit was not verified")
    return results, {
        "windows": len(windows),
        "seconds": time.monotonic() - started,
        "teacher_timing": timings,
        "logical_payload_limit_bytes": budget.available_payload(live_payload_bytes),
        "audit": {"status": "passed" if budget.audit_state else "not_requested", "ranks": audit},
    }
