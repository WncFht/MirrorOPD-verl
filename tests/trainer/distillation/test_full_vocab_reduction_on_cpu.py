# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Configuration, packed layouts, adapter, and optimizer-batch normalization."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.output import FullVocabOutput
from verl.trainer.distillation.reduction import PackedReduction, nonempty_sequence_count
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.distillation import DistillationLossConfig, FullVocabConfig
from verl.workers.config.distillation_spec import FutureCreditConfig, resolve_distillation_spec
from verl.workers.utils.padding import full_vocab_reduction, prediction_positions

AGGREGATES = ["token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]


@pytest.mark.parametrize("mode", AGGREGATES)
def test_hydra_future_and_actor_reduction_roundtrip(mode):
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "distillation.distillation_loss.loss_mode=mirror",
                "distillation.distillation_loss.mirror={objective:mdpo,estimator:full_vocab,kappa:0.5,eta:0.3}",
                "distillation.distillation_loss.full_vocab.backend=precompute",
                "distillation.distillation_loss.full_vocab.future.mode=discounted_mean",
                "distillation.distillation_loss.full_vocab.future.gamma=0.9",
                "distillation.distillation_loss.use_task_rewards=False",
                "distillation.distillation_loss.use_policy_gradient=False",
                "actor_rollout_ref.actor.loss_agg_mode=" + mode,
                "actor_rollout_ref.actor.loss_scale_factor=7",
            ],
        )
        parsed = omega_conf_to_dataclass(cfg.distillation.distillation_loss)
    assert isinstance(parsed.full_vocab.future, FutureCreditConfig)
    assert parsed.full_vocab.future.mode == "discounted_mean"
    assert parsed.full_vocab.future.gamma == 0.9
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == mode
    assert cfg.actor_rollout_ref.actor.loss_scale_factor == 7


@pytest.mark.parametrize("backend", ["recompute", "cached", "streaming", "liger"])
def test_config_rejects_future_backend(backend):
    with pytest.raises(ValueError, match="Future credit requires"):
        FullVocabConfig(backend=backend, future={"mode": "sum"})


def test_config_rejects_future_conditioned_support_and_sampled_objective():
    with pytest.raises(ValueError, match="common-conditioned"):
        FullVocabConfig(
            backend="precompute",
            future={"mode": "sum"},
            distribution={"mode": "common_conditioned", "common_vocab_size": 3},
        )
    with pytest.raises(ValueError, match="reverse KL and MDPO only"):
        DistillationLossConfig(loss_mode="k1", full_vocab={"backend": "precompute", "future": {"mode": "sum"}})


def test_support_declaration_is_parsed_once_and_not_serialized_as_config(monkeypatch):
    from dataclasses import asdict

    from verl.trainer.distillation.distribution import DistributionSpec

    config = FullVocabConfig(distribution={"mode": "common_conditioned", "common_vocab_size": 3})
    monkeypatch.setattr(DistributionSpec, "parse", lambda *a, **k: pytest.fail("Support was reparsed"))
    first = config.distribution_spec.compile(4, 5)
    second = config.distribution_spec.compile(6, 7)
    assert first.spec is second.spec is config.distribution_spec
    assert first.student_width == 4 and second.student_width == 6
    assert "distribution_spec" not in asdict(config) and "_distribution_spec" not in asdict(config)
    assert asdict(config)["distribution"] == {"mode": "common_conditioned", "common_vocab_size": 3}


def test_layout_preserves_gaps_empty_answers_and_exactly_one_target_shift():
    ids = torch.nested.nested_tensor(
        [torch.tensor([8, 7, 6, 5, 4, 3]), torch.tensor([9, 2, 1]), torch.tensor([8, 6, 4])], layout=torch.jagged
    )
    responses = torch.nested.nested_tensor(
        [torch.tensor([5, 4, 3]), torch.tensor([1]), torch.tensor([6, 4])], layout=torch.jagged
    )
    masks = torch.nested.nested_tensor(
        [torch.tensor([1, 0, 1]), torch.tensor([0]), torch.tensor([0, 1])], layout=torch.jagged
    )
    positions, packed = prediction_positions(ids, masks)
    layout = full_vocab_reduction(
        ids, responses, positions, packed, mode="seq-mean-token-mean", future=FutureCreditConfig(mode="sum")
    )
    assert layout.counts == (2, 0, 1)
    assert layout.times.tolist() == [0, 2, 1]
    assert layout.actions.tolist() == [5, 3, 4]
    assert nonempty_sequence_count(masks) == 2
    assert (
        full_vocab_reduction(ids, responses, positions, packed, mode="token-mean", future=FutureCreditConfig()) is None
    )
    assert nonempty_sequence_count(torch.empty(0, 3)) == 0


@pytest.mark.parametrize("mode", AGGREGATES)
@pytest.mark.parametrize("future_mode", ["none", "sum", "mean", "discounted_sequence"])
def test_dp_and_microbatch_partition_preserve_value_and_both_gradients(mode, future_mode):
    torch.manual_seed(17)
    counts = (4, 0, 1, 3)
    hs = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn_like(hs), torch.randn_like(ws)
    loss_cfg = DistillationLossConfig(
        loss_mode="mirror",
        mirror={"objective": "mdpo", "eta": 1},
        full_vocab={
            "backend": "precompute",
            "future": {"mode": future_mode, "gamma": 0.8 if future_mode.startswith("discounted") else 1.0},
        },
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    actor = SimpleNamespace(loss_agg_mode=mode, loss_scale_factor=7)
    distillation = SimpleNamespace(distillation_loss=loss_cfg)
    future = loss_cfg.full_vocab.future

    def loss_for(sequence_counts, start, dp_size):
        n = sum(sequence_counts)
        times = torch.cat([torch.arange(length) for length in sequence_counts])
        layout = PackedReduction(sequence_counts, times, torch.zeros(n, dtype=torch.long), mode, future)
        result = full_vocab_objective_result(
            hs[start : start + n],
            ws,
            ht[start : start + n],
            wt,
            spec=loss_cfg.loss_settings,
            reduction=layout,
            backend="precompute",
            token_chunk=2,
        )
        data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(
            data, dp_size=dp_size, batch_num_tokens=8, global_batch_size=99, full_vocab_num_sequences=3
        )
        return distillation_loss(
            actor, distillation, {"full_vocab": FullVocabOutput.from_result(result, loss_cfg.loss_settings, n)}, data
        )[0]

    expected = loss_for(counts, 0, 1)
    # Unequal DP partitions and micro-batches, including a completely empty rank.
    actual = (loss_for((4,), 0, 3) + loss_for((0,), 4, 3) + loss_for((1,), 4, 3) + loss_for((3,), 5, 3)) / 3
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (hs, ws)), torch.autograd.grad(expected, (hs, ws)), strict=True):
        torch.testing.assert_close(a, b)


def test_real_cpu_model_adapter_propagates_future_gradients_through_tied_head():
    import copy

    from transformers import AutoModelForCausalLM, Qwen2Config

    from verl.models.transformers.full_vocab import install_full_vocab_adapter

    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference = AutoModelForCausalLM.from_config(config).double().eval()
    adapted = copy.deepcopy(reference)
    install_full_vocab_adapter(adapted)
    ids = torch.tensor([[1, 2, 3, 4]])
    ht, wt = torch.randn(3, 4, dtype=torch.float64), torch.randn(11, 4, dtype=torch.float64)
    lp = reference(ids).logits[0, :3].log_softmax(-1)
    lt = (ht @ wt.T).log_softmax(-1)
    cost = (lp.exp() * (lp - lt)).sum(-1)
    suffix = torch.stack([cost[1:].detach().sum(), cost[2].detach(), cost.new_zeros(())])
    expected = (cost + suffix * lp[range(3), ids[0, 1:]]).sum()
    layout = PackedReduction((3,), torch.arange(3), ids[0, 1:], future=FutureCreditConfig(mode="sum"))
    settings = resolve_distillation_spec({"loss_mode": "mirror", "mirror": {"eta": 1}})

    def output(hidden, head):
        return head(
            hidden[0, :3],
            full_vocab_callback=lambda h, w: full_vocab_objective_result(
                h, w, ht, wt, spec=settings, reduction=layout, backend="precompute", token_chunk=2
            ),
        )

    result = adapted(ids, full_vocab_callback=output)
    torch.testing.assert_close(result.loss_sum, expected)
    expected.backward()
    result.loss_sum.backward()
    for (name, param), (other_name, other_param) in zip(
        reference.named_parameters(), adapted.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(param.grad, other_param.grad, atol=2e-10, rtol=2e-10)
