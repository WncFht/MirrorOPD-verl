# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Actual TQ IO with the production anchor scope; GPU capture is tested separately."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import transfer_queue as tq
from tensordict import NonTensorStack, TensorDict
from test_rejection_denominator_on_cpu import runtime  # noqa: F401

from verl.trainer.distillation.anchor import FrozenAnchorHead
from verl.trainer.distillation.hidden_io import HiddenBatchIO
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.config.distillation_spec import resolve_distillation_spec
from verl.workers.engine_workers import ActorRolloutRefWorker


def nested(values):
    return torch.nested.nested_tensor(values, layout=torch.jagged)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fail_training", [False, True])
@pytest.mark.usefixtures("runtime")
def test_anchor_scope_real_tq_reorder_and_cleanup(asynchronous, fail_training):
    keys = [f"mirror-{asynchronous}-{fail_training}-{i}" for i in range(2)]
    data = TensorDict(
        {
            "input_ids": nested([torch.tensor([3, 4, 5]), torch.tensor([4, 5, 6])]),
            "position_ids": nested([torch.arange(3), torch.arange(3)]),
            "prompts": nested([torch.tensor([3]), torch.tensor([4])]),
            "responses": nested([torch.tensor([4, 5]), torch.tensor([5, 6])]),
            "loss_mask": nested([torch.tensor([1, 1]), torch.tensor([1, 1])]),
            "response_mask": nested([torch.tensor([1, 1]), torch.tensor([1, 1])]),
            "sample_keys": NonTensorStack(*keys),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        sample_partition="train",
        full_vocab_train=True,
        full_vocab_async_write=asynchronous,
        full_vocab_prefetch=asynchronous,
        mirror_anchor_version=(3, 0),
    )
    tq.kv_batch_put(keys=keys, partition_id="train", fields=data.select("input_ids", "loss_mask"))
    for i, key in enumerate(keys):
        tq.kv_batch_put(
            keys=[key],
            partition_id="train",
            fields=TensorDict(
                {
                    "teacher_hidden": nested([torch.full((2, 3), float(i + 10))]),
                    "teacher_positions": nested([torch.arange(2)]),
                },
                batch_size=[1],
            ),
        )
    engine = SimpleNamespace(module=torch.nn.Linear(3, 3), get_data_parallel_group=lambda: None)
    calls = []

    def infer(prepared):
        assert not tu.get_non_tensor_data(prepared, "full_vocab_train", default=True)
        engine.mirror_anchor_head = FrozenAnchorHead(engine.module.weight, (3, 0))
        with HiddenBatchIO(prepared, [tu.index_select_tensor_dict(prepared, [i]) for i in range(len(prepared))]) as io:
            for i, mb in enumerate(io):
                out = {
                    "model_output": {
                        "anchor_hidden": nested([torch.full((2, 3), float(i))]),
                        "anchor_positions": nested([torch.arange(2)]),
                        "anchor_version": torch.tensor([[3, 0]]),
                    }
                }
                io.publish(out, mb)
                assert "anchor_hidden" not in out["model_output"]
                calls.append(i)
        return TensorDict({}, batch_size=[2])

    spec = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"objective": "bpo", "eta": 1}})

    def train(data):
        batches = [tu.index_select_tensor_dict(data, [i]) for i in reversed(range(len(data)))]
        with HiddenBatchIO(data, batches) as io:
            for i, mb in enumerate(io):
                assert mb["anchor_hidden"].values()[0, 0].item() == 1 - i
                assert mb["teacher_hidden"].values()[0, 0].item() == 11 - i
                assert [v.tolist() for v in mb["anchor_version"].unbind()] == [[3, 0]]
        assert all("anchor_hidden" not in mb for mb in batches)
        if fail_training:
            raise RuntimeError("test failure")
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {}})

    actor = SimpleNamespace(engine=engine, infer_batch=infer, train_mini_batch=train, device_name="cpu")
    worker = SimpleNamespace(actor=actor, mirror_spec=spec)
    try:
        with pytest.raises(RuntimeError, match="test failure") if fail_training else nullcontext():
            out = inspect.unwrap(ActorRolloutRefWorker.update_actor)(worker, data)
            metrics = reduce_metrics(tu.get(out, "metrics"))
            assert metrics["perf/anchor/prepare_seconds_max"] >= 0
        assert calls == [0, 1]
        assert engine.mirror_anchor_head is None
        assert "full_vocab_sources" not in data
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
