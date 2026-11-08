# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded serial teacher/student readouts through the native FSDP workers."""

import random
import time

import numpy as np
import torch
import transfer_queue as tq

from verl.experimental.rollout_policy.collector import UnsafeDrainError

from .artifacts import identity
from .execution import hidden_dtype_bytes, numerical_profile
from .native import put_fixed_rows
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
    windows = payload_windows(
        rows,
        hidden_size=hidden_size,
        payload_bytes=budget.available_payload(live_payload_bytes),
        dtype_bytes=hidden_dtype_bytes(trainer.config),
    )
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    from .execution import NativeEvaluationExecution

    execution = NativeEvaluationExecution(trainer)
    space = execution.distribution(objective=budget.objective_space)
    profile = numerical_profile(trainer.config)
    measure_id = identity({"distribution": space.metadata, "numerical_profile": profile})
    extra_names = (
        ["uncoarsened_rkl_sum", "student_excluded_mass_sum", "teacher_excluded_mass_sum"]
        if space.spec.mode != "native" or space.termination_ids
        else []
    )
    position_names = ["rkl_position_sum", "rkl_position_count"]
    if extra_names:
        position_names.append("uncoarsened_rkl_position_sum")
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
            | {
                "trajectory_id": row.get("trajectory_id", row["id"]),
                "rkl_sum": 0.0,
                "tokens": 0,
                **{name: 0.0 for name in extra_names},
                "rkl_position_sum": [0.0] * 10,
                "rkl_position_count": [0] * 10,
                **({"uncoarsened_rkl_position_sum": [0.0] * 10} if extra_names else {}),
                "measure_id": measure_id,
            }
        )
    stage_seconds = {}
    active_rpc, begun = False, False
    audit = None
    try:
        active_rpc = True
        trainer.actor_rollout_wg.begin_rkl_readout(audit_state=budget.audit_state)
        active_rpc, begun = False, True
        for references in windows:
            window = load_window(references) if load_window is not None else references
            stage_start = time.monotonic()
            meta, by_key = put_fixed_rows(window, trainer.global_steps)
            stage_seconds["trajectory_publish"] = (
                stage_seconds.get("trajectory_publish", 0.0) + time.monotonic() - stage_start
            )
            try:
                meta = execution.balance(meta)
                active_rpc = True
                window_timing = {}
                execution.score_teacher(meta, window_timing)
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
                    full_vocab_score_objective=budget.objective_space,
                )
                active_rpc = True
                stage_start = time.monotonic()
                trainer.actor_rollout_wg.score_full_vocab(meta)
                stage_seconds["student_readout_rpc"] = (
                    stage_seconds.get("student_readout_rpc", 0.0) + time.monotonic() - stage_start
                )
                active_rpc = False
                actual_keys = [key for key, tag in zip(meta.keys, meta.tags, strict=True) if not tag.get("is_padding")]
                if set(actual_keys) != set(by_key) or len(actual_keys) != len(by_key):
                    raise ValueError("Inference balancing changed logical trajectory identities")
                scores = tq.kv_batch_get(
                    keys=actual_keys,
                    partition_id=meta.partition_id,
                    select_fields=["rkl_sum", "rkl_count", *extra_names, *position_names],
                )
                for index, (key, total, count) in enumerate(
                    zip(actual_keys, scores["rkl_sum"].unbind(), scores["rkl_count"].unbind(), strict=True)
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
                            "measure_id": measure_id,
                            **{name: float(scores[name][index].item()) for name in extra_names},
                            **{name: scores[name][index].tolist() for name in position_names},
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
        "stage_seconds": stage_seconds,
        "distribution": space.metadata,
        "numerical_profile": profile,
        "measure_id": measure_id,
        "logical_payload_limit_bytes": budget.available_payload(live_payload_bytes),
        "audit": {"status": "passed" if budget.audit_state else "not_requested", "ranks": audit},
    }
