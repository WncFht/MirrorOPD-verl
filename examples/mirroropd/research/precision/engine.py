#!/usr/bin/env python3
# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Production FSDP2 primitives and independent FP64 scoring for research.

Requires four GPUs and saved fixed-prefix records. Packed SDPA comparisons are
valid here only because each rank/micro-batch contains exactly one sequence.
Use on an allocated GPU pool; the memory cap is not a scheduler reservation.

Uses the production FSDP2 engine, global token normalization, anchor scope and
AdamW. This creates a local low-KL state, not a reconstruction of the original
on-policy E45/E47 history. Pre/post evaluation uses FP64 row arithmetic.
"""

from types import SimpleNamespace

import torch

import verl.trainer.distillation.full_vocab as kernels
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.targets import build_log_weights
from verl.utils import tensordict_utils as tu
from verl.workers.config import (
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    TrainingWorkerConfig,
)
from verl.workers.engine_workers import TrainingWorker
from verl.workers.update_records import UpdateResult, finalize_update


def nested(rows):
    return torch.nested.nested_tensor(rows, layout=torch.jagged)


def loss_fn(model_output, data, dp_group, config):
    return distillation_loss(SimpleNamespace(loss_agg_mode="token-mean"), config, model_output, data)


def make_worker(path, world, teacher=False, precision="bfloat16", attention="flash_attention_2"):
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(
            path=str(path),
            enable_gradient_checkpointing=not teacher,
            use_fused_kernels=False,
            enable_full_vocab=True,
            use_remove_padding=True,
            override_config={"attn_implementation": attention},
        ),
        engine_config=FSDPEngineConfig(
            strategy="fsdp2",
            forward_only=teacher,
            model_dtype="bfloat16" if teacher else "float32",
            fsdp_size=world,
            mixed_precision={"param_dtype": precision, "reduce_dtype": "float32"},
            forward_only_cpu_offload=False,
            param_offload=True,
            optimizer_offload=not teacher,
            use_torch_compile=False,
            infer_max_token_len_per_gpu=1024,
            max_token_len_per_gpu=1024,
            use_no_sync_for_gradient_accumulation=True,
        ),
        optimizer_config=None
        if teacher
        else FSDPOptimizerConfig(
            lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01, lr_scheduler_type="constant", total_training_steps=1000
        ),
    )
    w = TrainingWorker(cfg)
    w.reset()
    return w


def reference(hs, ws, ht, wt, *, spec, hq=None, wq=None, **kwargs):
    zp = kernels.project(hs, ws).double()
    zt = kernels.project(ht, wt).double()
    lp = zp.log_softmax(-1)
    lt = zt.log_softmax(-1)[:, : len(ws)]
    if spec.teacher_only_kl:
        rows = (lp.exp() * (lp - lt)).sum(-1)
        return kernels.FullVocabObjectiveResult(rows.sum(), rows, {})
    zq = kernels.project(hq, wq).double()
    lq = zq.log_softmax(-1)
    b = build_log_weights(lt, lq, kappa=spec.kappa, eta=spec.eta)
    p = lp.exp()
    gap = lp - b
    raw_mdpo = (p * gap).sum(-1)
    if spec.objective == "bpo":
        q = lq.exp()
        centered = gap - (q * gap).sum(-1, keepdim=True)
        rows = (q * centered.square()).sum(-1) / 2
    else:
        rows = raw_mdpo
    lz = b.logsumexp(-1)
    return kernels.FullVocabObjectiveResult(
        rows.sum(),
        rows,
        {
            "target_kl": raw_mdpo + lz,
            "target_log_z": lz,
            "anchor_kl": (p * (lp - lq)).sum(-1),
            "teacher_kl": (p * (lp - lt)).sum(-1),
        },
    )


def evaluate(actor, data, *, reference_fn=reference):
    probe = data.clone()
    tu.assign_non_tensor(probe, compute_loss=True, disable_auto_offload=False, global_token_num=None)
    old = kernels.full_vocab_objective_result
    kernels.full_vocab_objective_result = reference_fn
    try:
        out = actor.infer_batch(probe)
    finally:
        kernels.full_vocab_objective_result = old
    _, rec = finalize_update(
        UpdateResult(None, tu.get(out, "metrics")),
        role="actor",
        collect_records=True,
        collect_metrics=True,
        update_index=16,
        minibatches_per_epoch=1,
    )
    return {k: v for k, v in rec["metrics"].items() if k.startswith("distill/") and "_position/" not in k}


def scalar_metrics(records):
    return [
        {
            k: v
            for k, v in r["metrics"].items()
            if k.startswith(("distill/", "optim/actor/", "rollout/data_reuse/")) and "_position/" not in k
        }
        for r in records
    ]


def local(p):
    return p.to_local() if hasattr(p, "to_local") else p


def snapshot(actor):
    return {n: local(p).detach().cpu().clone() for n, p in actor.engine.module.named_parameters()}


def assign(actor, values, rounded=False):
    with torch.no_grad():
        for n, p in actor.engine.module.named_parameters():
            v = values[n]
            if rounded:
                v = v.to(torch.bfloat16).float()
            local(p).copy_(v)
