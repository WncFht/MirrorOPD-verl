# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent probability-space and gradient oracles for conditional head supports."""

import pytest
import torch

from verl.trainer.distillation.distribution import DistributionSpec
from verl.trainer.distillation.evaluation import score_distribution_rows
from verl.trainer.distillation.full_vocab import full_vocab_objective_result
from verl.workers.config.distillation_spec import resolve_distillation_spec


@pytest.mark.parametrize("backend", ["dense", "recompute", "streaming", "precompute"])
@pytest.mark.parametrize("mapped", [False, True])
def test_conditional_loss_gradient_and_mass_match_probability_oracle(backend, mapped):
    torch.manual_seed(42)
    hs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    wt = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    student, teacher = ([4, 0, 2], [1, 3, 0]) if mapped else (list(range(5)), list(range(5)))
    definition = (
        dict(mode="common_conditioned", student_ids=student, teacher_ids=teacher)
        if mapped
        else dict(mode="common_conditioned", common_vocab_size=5)
    )
    zp, zt = hs @ ws.T, ht.detach() @ wt.detach().T
    lp, lq = zp[:, student].log_softmax(-1), zt[:, teacher].log_softmax(-1)
    expected = (lp.exp() * (lp - lq)).sum(-1)
    gradient = torch.autograd.grad(expected.sum(), (hs, ws), retain_graph=True)
    result = full_vocab_objective_result(
        hs,
        ws,
        ht,
        wt,
        spec=resolve_distillation_spec(
            {"loss_mode": "mirror", "mirror": {"objective": "mdpo", "estimator": "full_vocab", "eta": 1}}
        ),
        backend=backend,
        token_chunk=2,
        vocab_chunk=3,
        distribution=definition,
    )
    torch.testing.assert_close(result.loss_sum, expected.sum(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(result.objective_rows, expected, rtol=1e-10, atol=1e-10)
    result.loss_sum.backward()
    torch.testing.assert_close(hs.grad, gradient[0], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(ws.grad, gradient[1], rtol=1e-9, atol=1e-9)
    assert ht.grad is None and wt.grad is None
    assert torch.count_nonzero(ws.grad[[i for i in range(7) if i not in student]]) == 0
    readout = score_distribution_rows(hs, ws, ht, wt, token_chunk=2, distribution=definition)
    torch.testing.assert_close(readout["rkl"], expected)
    for role, z, ids in (("student", zp, student), ("teacher", zt, teacher)):
        mass = 1 - z.softmax(-1)[:, ids].sum(-1)
        torch.testing.assert_close(readout[role + "_excluded_mass"], mass, atol=1e-14, rtol=1e-10)
        torch.testing.assert_close(result.token_diagnostics[role + "_excluded_mass"], mass, atol=1e-14, rtol=1e-10)


def test_coarse_objective_rows_are_distinct_from_legacy_raw_diagnostic():
    h = torch.eye(1, dtype=torch.float64)
    p = torch.tensor([[0.05], [0.45], [0.5]], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([[0.45], [0.05], [0.5]], dtype=torch.float64).log()
    result = full_vocab_objective_result(
        h,
        p,
        h,
        q,
        spec=resolve_distillation_spec(
            {"loss_mode": "mirror", "mirror": {"objective": "mdpo", "estimator": "full_vocab", "eta": 1}}
        ),
        backend="precompute",
        termination_token_ids=(0, 1),
    )
    torch.testing.assert_close(result.objective_rows.sum(), result.loss_sum)
    assert result.loss_sum.abs() < 1e-15 and result.token_objective.item() > 0.8
    readout = score_distribution_rows(h, p, h, q, token_chunk=1, termination_ids=(0, 1))
    torch.testing.assert_close(readout["rkl"], result.objective_rows)
    torch.testing.assert_close(readout["uncoarsened_rkl"], result.token_objective)


def test_partial_termination_classes_and_implicit_truncation_are_rejected():
    with pytest.raises(ValueError, match="teacher support"):
        DistributionSpec().compile(7, 5)
    with pytest.raises(ValueError, match="complete termination"):
        DistributionSpec(mode="common_conditioned", common_vocab_size=5).compile(7, 7, (4, 5))
    a = DistributionSpec(mode="common_conditioned", common_vocab_size=5).compile(7, 5)
    assert a.measure_id != DistributionSpec().compile(5, 7).measure_id
    assert (
        a.measure_id
        == DistributionSpec.parse(dict(mode="common_conditioned", common_vocab_size=5)).compile(7, 5).measure_id
    )


def test_verifier_channel_does_not_change_sampling_identity():
    from verl.trainer.evaluation.config import GenerationProtocol

    a, b = GenerationProtocol(), GenerationProtocol(answer_channel="whole_response")
    assert a.protocol_id == b.protocol_id
    assert a.metadata() != b.metadata()


def test_pair_acceptance_checks_real_tokenizer_semantics_before_gpu():
    from tokenizers import Tokenizer, normalizers
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, PreTrainedTokenizerFast

    from verl.trainer.distillation.compatibility import ModelPairContract

    def tokenizer(lower=False):
        backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
        if lower:
            backend.normalizer = normalizers.Lowercase()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    student = LlamaConfig(vocab_size=7, max_position_embeddings=128)
    teacher = LlamaConfig(vocab_size=5, max_position_embeddings=64)
    spec = dict(mode="common_conditioned", common_vocab_size=5)
    pair = ModelPairContract.build(student, teacher, tokenizer(), tokenizer(), distribution=spec)
    pair.validate_prefix([1, 2, 1])
    assert pair.distribution["spec"]["common_vocab_size"] == 5
    with pytest.raises(ValueError, match="context"):
        pair.validate_prefix([1] * 65)
    with pytest.raises(ValueError, match="normalization"):
        ModelPairContract.build(student, teacher, tokenizer(), tokenizer(True), distribution=spec)


def test_report_rejects_mixing_probability_spaces():
    from verl.trainer.evaluation.reporting import evaluation_metrics

    rows = [
        dict(id=str(i), source_id=str(i), bank="source_heldout", rkl_sum=1.0, tokens=2, measure_id=name)
        for i, name in enumerate(["native", "conditioned"])
    ]
    with pytest.raises(ValueError, match="distribution identities"):
        evaluation_metrics(rows, baselines={})


def test_hidden_budget_follows_forward_precision():
    from omegaconf import OmegaConf

    from verl.trainer.evaluation.execution import hidden_dtype_bytes

    config = OmegaConf.create({"distillation": {"shared_teacher": {"mixed_precision": {"param_dtype": "fp32"}}}})
    assert hidden_dtype_bytes(config) == 4
    config.distillation.shared_teacher.mixed_precision.param_dtype = "bf16"
    assert hidden_dtype_bytes(config) == 2
    config.distillation.shared_teacher.forward_precision = "bf16_fp32"
    assert hidden_dtype_bytes(config) == 4


def test_tiny_excluded_mass_is_not_lost_to_partition_subtraction():
    space = DistributionSpec(mode="common_conditioned", common_vocab_size=2).compile(3, 2)
    z = torch.tensor([[20.0, 20.0, -10.0]], dtype=torch.float32)
    actual = space.excluded_from_logits(z, "student")
    expected = z.double().softmax(-1)[:, -1]
    assert actual.item() > 0
    torch.testing.assert_close(actual.double(), expected, atol=1e-20, rtol=2e-6)
