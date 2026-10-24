# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real TQ field updates and the FSDP normalization boundary, with a CPU toy forward.

The model forward/microbatch packing are replaced; TQ storage, the FSDP batch
normalizer, Gloo reduction, agg_loss and backward are real. No existing Ray
cluster is attached or stopped.
"""

import tempfile
import uuid
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp import transformer_impl as fsdp


@pytest.fixture(scope="module")
def runtime():
    assert not ray.is_initialized() and not dist.is_initialized()
    with tempfile.TemporaryDirectory(prefix="mask-contract-") as root:
        ray.init(
            address="local",
            namespace="mask-contract",
            num_cpus=4,
            num_gpus=0,
            include_dashboard=False,
            _temp_dir=root,
            object_store_memory=128 * 1024**2,
            logging_level="ERROR",
        )
        tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 1}}}))
        dist.init_process_group("gloo", init_method="file://" + root + "/gloo", rank=0, world_size=1)
        try:
            yield
        finally:
            dist.destroy_process_group()
            tq.close()
            ray.shutdown()


def masks(rows):
    return torch.nested.nested_tensor([torch.tensor(row, dtype=torch.float32) for row in rows], layout=torch.jagged)


@pytest.mark.parametrize("micro_size", [1, 2])
@pytest.mark.parametrize("kept,expected", [([[1, 0, 0, 0], [1, 1]], 0.5), ([[0, 0, 0, 0], [0, 0]], 0.0)])
def test_rejected_tokens_keep_original_global_denominator(runtime, monkeypatch, micro_size, kept, expected):
    partition = "mask-" + uuid.uuid4().hex
    keys = ["a", "b"]
    original = masks([[1, 1, 1, 1], [1, 1]])
    data = TensorDict({"loss_mask": original, "response_mask": original}, batch_size=[2])
    tq.kv_batch_put(keys=keys, partition_id=partition, fields=data)
    # The trainer publishes the post-rejection response mask, not loss_mask.
    tq.kv_batch_put(
        keys=keys, partition_id=partition, fields=TensorDict({"response_mask": masks(kept)}, batch_size=[2])
    )
    restored = tq.kv_batch_get(keys=keys, partition_id=partition, select_fields=["loss_mask", "response_mask"])
    assert restored["loss_mask"].sum().item() == 6
    assert restored["response_mask"].sum().item() == 6 * expected
    parameter = torch.tensor(1.0, requires_grad=True)
    observed_denominators = []

    def forward_step(batch, **kwargs):
        denominator = tu.get(batch, "batch_num_tokens")
        observed_denominators.append(denominator)
        mask = batch["response_mask"].to_padded_tensor(0.0)
        loss = agg_loss(
            parameter * torch.ones_like(mask),
            mask,
            "token-mean",
            dp_size=tu.get(batch, "dp_size"),
            batch_num_tokens=denominator,
        )
        return loss, {"loss": loss.detach().item(), "metrics": {}, "model_output": {}}

    engine = SimpleNamespace(
        ulysses_sequence_parallel_size=1,
        get_data_parallel_group=lambda: dist.group.WORLD,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        _gradient_sync_context=lambda **kw: nullcontext(),
        forward_step=forward_step,
    )
    monkeypatch.setattr(fsdp, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        fsdp, "prepare_micro_batches", lambda data, **kw: (tu.chunk_tensordict(data, len(data) // micro_size), None)
    )
    monkeypatch.setattr(fsdp, "postprocess_batch_func", lambda output_lst, **kw: output_lst)
    result = fsdp.FSDPEngine.forward_backward_batch(engine, restored, lambda *a: None)
    assert observed_denominators == [6] * (2 // micro_size)
    assert sum(r["loss"] for r in result) == pytest.approx(expected)
    assert parameter.grad.item() == pytest.approx(expected)
    tq.kv_clear(keys=keys, partition_id=partition)


def test_explicit_sequence_denominator_keeps_empty_sequences():
    mask = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    loss = torch.ones_like(mask)
    explicit = agg_loss(loss, mask, "seq-mean-token-mean", global_batch_size=2)
    inferred = agg_loss(loss, mask, "seq-mean-token-mean")
    assert explicit.item() == pytest.approx(0.5)
    assert inferred.item() == pytest.approx(1.0)


def test_no_original_valid_tokens_is_rejected(runtime, monkeypatch):
    monkeypatch.setattr(fsdp, "get_device_id", lambda: "cpu")
    data = TensorDict({"loss_mask": masks([[0, 0], [0]])}, batch_size=[2])
    engine = SimpleNamespace(ulysses_sequence_parallel_size=1, get_data_parallel_group=lambda: dist.group.WORLD)
    with pytest.raises(ValueError, match="zero globally valid"):
        fsdp.FSDPEngine.forward_backward_batch(engine, data, lambda *a: None)


def test_actual_trainer_rejection_publishes_masks_without_changing_population(runtime, monkeypatch):
    from verl.trainer.ppo.v1 import trainer_base

    class Trainer(trainer_base.PPOTrainer):
        def on_step_end(self):
            pass

        def on_sample_end(self):
            pass

    partition = "trainer-mask-" + uuid.uuid4().hex
    keys = ["a", "b"]
    original = masks([[1, 1, 1, 1], [1, 1]])
    zeros = masks([[0, 0, 0, 0], [0, 0]])
    meta = tq.kv_batch_put(
        keys=keys,
        partition_id=partition,
        fields=TensorDict(
            {
                "uid": torch.tensor([1, 2]),
                "loss_mask": original,
                "response_mask": original,
                "rm_scores": zeros,
                "rollout_log_probs": masks([[-2, -2, -2, -2], [-2, -2]]),
                "old_log_probs": masks([[-2, -4, -4, -4], [-2, -2]]),
                "ref_log_prob": zeros,
                "values": zeros,
            },
            batch_size=[2],
        ),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": False,
                "rollout_correction": {"rollout_rs": "token_k1", "rollout_rs_threshold": 2.0},
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )

    def advantages(data, **kwargs):
        data.batch["advantages"] = torch.ones_like(data.batch["response_mask"])
        data.batch["returns"] = torch.zeros_like(data.batch["response_mask"])
        return data

    monkeypatch.setattr(trainer_base, "compute_advantage_for_multi_trajectories", advantages)
    trainer._compute_advantage(meta, {})
    stored = tq.kv_batch_get(
        keys=keys, partition_id=partition, select_fields=["loss_mask", "response_mask", "metrics_response_mask"]
    )
    assert stored["loss_mask"].values().tolist() == [1, 1, 1, 1, 1, 1]
    assert stored["response_mask"].values().tolist() == [1, 0, 0, 0, 1, 1]
    assert stored["metrics_response_mask"].values().tolist() == [1, 1, 1, 1, 1, 1]
    tq.kv_clear(keys=keys, partition_id=partition)


def _gloo_normalizer_worker(rank, store, rows):
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank, world_size=2)
    try:
        parameter = torch.tensor(1.0, requires_grad=True)
        data = TensorDict(
            {"loss_mask": masks([rows[rank][0]]), "response_mask": masks([rows[rank][1]])}, batch_size=[1]
        )

        def forward_step(batch, **kwargs):
            mask = batch["response_mask"].to_padded_tensor(0.0)
            loss = agg_loss(
                parameter * torch.ones_like(mask),
                mask,
                "token-mean",
                dp_size=tu.get(batch, "dp_size"),
                batch_num_tokens=tu.get(batch, "batch_num_tokens"),
            )
            return loss, {"loss": loss.detach().item(), "metrics": {}, "model_output": {}}

        engine = SimpleNamespace(
            ulysses_sequence_parallel_size=1,
            get_data_parallel_group=lambda: dist.group.WORLD,
            get_data_parallel_size=lambda: 2,
            get_data_parallel_rank=lambda: rank,
            _gradient_sync_context=lambda **kw: nullcontext(),
            forward_step=forward_step,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fsdp, "get_device_id", lambda: "cpu")
            patch.setattr(fsdp, "prepare_micro_batches", lambda data, **kw: ([data], None))
            patch.setattr(fsdp, "postprocess_batch_func", lambda output_lst, **kw: output_lst)
            fsdp.FSDPEngine.forward_backward_batch(engine, data, lambda *a: None)
        assert tu.get(data, "batch_num_tokens") == 6
        # The toy module has no FSDP wrapper; perform its real gradient collective explicitly.
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        assert parameter.grad.item() == pytest.approx(1 / 3)
    finally:
        dist.destroy_process_group()


def test_real_tq_masks_and_two_rank_gloo_denominator(runtime, tmp_path):
    import torch.multiprocessing as mp

    partition = "dp-mask-" + uuid.uuid4().hex
    original = masks([[1, 1, 1, 1], [1, 1]])
    tq.kv_batch_put(
        keys=["a", "b"],
        partition_id=partition,
        fields=TensorDict({"loss_mask": original, "response_mask": original}, batch_size=[2]),
    )
    tq.kv_batch_put(
        keys=["a", "b"],
        partition_id=partition,
        fields=TensorDict({"response_mask": masks([[0, 0, 0, 0], [1, 1]])}, batch_size=[2]),
    )
    stored = tq.kv_batch_get(keys=["a", "b"], partition_id=partition, select_fields=["loss_mask", "response_mask"])
    rows = [
        (a.tolist(), b.tolist())
        for a, b in zip(stored["loss_mask"].unbind(), stored["response_mask"].unbind(), strict=True)
    ]
    mp.spawn(_gloo_normalizer_worker, args=(str(tmp_path / "dp-normalizer"), rows), nprocs=2, join=True)
    tq.kv_clear(keys=["a", "b"], partition_id=partition)


def test_real_tq_identity_bridge_retains_reordered_keys(runtime):
    from transfer_queue import KVBatchMeta

    from verl.utils import transferqueue_utils as tqu

    partition = "identity-" + uuid.uuid4().hex
    tq.kv_batch_put(
        keys=["a", "b"], partition_id=partition, fields=TensorDict({"value": torch.tensor([10, 20])}, batch_size=[2])
    )
    meta = KVBatchMeta(
        keys=["b", "a"],
        tags=[{}, {}],
        partition_id=partition,
        fields=["value"],
        extra_info={"retain_sample_identity": True},
    )
    restored = tqu._meta_to_realdata(meta)
    assert list(restored["sample_keys"]) == ["b", "a"]
    assert tu.get_non_tensor_data(restored, "sample_partition", default=None) == partition
    assert restored["value"].tolist() == [20, 10]
    tq.kv_clear(keys=["a", "b"], partition_id=partition)
