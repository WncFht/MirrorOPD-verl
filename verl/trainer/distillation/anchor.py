# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One actor-call anchor scope. Uses the current model, never a third model/worker."""

from contextlib import contextmanager
from time import perf_counter

import torch

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import response_from_nested


class FrozenAnchorHead:
    def __init__(self, weight, version):
        from torch.distributed.tensor import DTensor

        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        self.cpu_weight = weight.detach().to(device="cpu", copy=True).contiguous()
        self.version = tuple(version)
        self.weight = None

    def get(self, device, dtype):
        if self.weight is None or self.weight.device != device or self.weight.dtype != dtype:
            self.weight = self.cpu_weight.to(device=device, dtype=dtype)
        return self.weight


def _check_deterministic_anchor(actor):
    module = actor.engine.module
    if any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in module.modules()):
        raise ValueError("Mirror anchor requires zero training dropout")
    config = getattr(module, "config", None)
    for name in ("attention_dropout", "hidden_dropout_prob", "embedding_dropout"):
        if getattr(config, name, 0):
            raise ValueError(f"Mirror anchor requires {name}=0")


@contextmanager
def anchor_scope(actor, data, spec):
    """Prepare the entire local shard before train_mini_batch, clean up even on failure."""
    if spec is None or spec.anchor_payload is None:
        yield 0.0
        return
    if getattr(actor.engine, "mirror_anchor_head", None) is not None:
        raise RuntimeError("An anchor scope is already active")
    started = perf_counter()
    _check_deterministic_anchor(actor)
    fields = [
        "input_ids",
        "position_ids",
        "loss_mask",
        "response_mask",
        "prompts",
        "responses",
        "sample_keys",
        "sample_partition",
        "full_vocab_keys",
        "full_vocab_partition",
    ]
    prepared = data.select(*(k for k in fields if k in data)).clone(recurse=False)
    version = tuple(tu.get_non_tensor_data(data, "mirror_anchor_version", default=(0, 0)))
    tu.assign_non_tensor(
        prepared,
        compute_loss=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        score_centering=False,
        temperature=1.0,
        full_vocab_train=False,
        full_vocab_capture="anchor" if spec.anchor_payload == "hidden" else None,
        full_vocab_async_write=tu.get_non_tensor_data(data, "full_vocab_async_write", default=False),
        mirror_anchor_version=version,
        global_token_num=None,
    )
    try:
        output = actor.infer_batch(prepared)
        if spec.anchor_payload == "scalar":
            data["old_log_probs"] = response_from_nested(output["log_probs"], data["response_mask"])
        else:
            head = getattr(actor.engine, "mirror_anchor_head", None)
            if head is None or head.version != version:
                raise RuntimeError("Anchor capture did not produce the expected frozen head")
            # Direct callers without TQ keys receive their small/test payload inline.
            for field in ("anchor_hidden", "anchor_positions", "anchor_version"):
                if field in output:
                    data[field] = output[field]
            tu.assign_non_tensor(data, full_vocab_sources=("teacher", "anchor"), mirror_anchor_version=version)
        yield perf_counter() - started
    finally:
        actor.engine.mirror_anchor_head = None
        if spec.anchor_payload == "scalar":
            data.pop("old_log_probs", None)
        else:
            for field in ("anchor_hidden", "anchor_positions", "anchor_version", "full_vocab_sources"):
                data.pop(field, None)
