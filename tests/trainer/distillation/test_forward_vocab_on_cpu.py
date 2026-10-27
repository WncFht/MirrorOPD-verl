from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.forward_vocab import forward_rows, full_forward_kl_result
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig, FullVocabConfig


@pytest.mark.parametrize("backend", ["dense", "recompute", "precompute", "precompute_fused"])
@pytest.mark.parametrize("tokens,extra", [(0, 0), (1, 0), (7, 0), (7, 4)])
def test_forward_value_vjp_native_reverse_and_support(backend, tokens, extra):
    torch.manual_seed(42)
    h = torch.randn(tokens, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 5, dtype=torch.float64, requires_grad=True)
    t = torch.randn(tokens, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(19 + extra, 8, dtype=torch.float64, requires_grad=True)
    lp = (h @ w.T).log_softmax(-1)
    lt = (t.detach() @ v.detach().T).log_softmax(-1)
    lq = (t.detach() @ v[:19].detach().T).log_softmax(-1)
    expected = (lq.exp() * (lq - lp)).sum()
    eg = torch.autograd.grad(expected * 0.37, (h, w))
    result = full_forward_kl_result(h, w, t, v, backend=backend, token_chunk=3, teacher_support="student")
    actual = torch.autograd.grad(result.loss_sum * 0.37, (h, w, t, v), allow_unused=True)
    torch.testing.assert_close(result.loss_sum, expected, atol=1e-10, rtol=1e-10)
    for g, ref in zip(actual[:2], eg, strict=True):
        torch.testing.assert_close(g, ref, atol=1e-10, rtol=1e-10)
    assert actual[2:] == (None, None)
    torch.testing.assert_close(
        result.token_diagnostics["full_kl"], (lp.exp() * (lp - lt[:, :19])).sum(-1), atol=1e-10, rtol=1e-10
    )
    if tokens:
        torch.testing.assert_close(
            result.token_diagnostics["teacher_outside_student_mass"], lt[:, 19:].exp().sum(-1), atol=1e-12, rtol=1e-10
        )
    if extra:
        with pytest.raises(ValueError, match="explicit"):
            full_forward_kl_result(h, w, t, v)


def test_close_distribution_shift_and_vjp():
    torch.manual_seed(5)
    q = torch.randn(4, 37, dtype=torch.float64) * 3
    z = (q + torch.randn_like(q) * 1e-6 + 200).requires_grad_()
    value, g, info = forward_rows(z, q)
    lp = z.log_softmax(-1)
    lq = q.log_softmax(-1)
    reference = (lq.exp() * (lq - lp)).sum(-1)
    torch.testing.assert_close(value, reference, atol=3e-14, rtol=1e-5)
    torch.testing.assert_close(g, lp.exp() - lq.exp(), atol=3e-14, rtol=1e-5)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), z)[0], g, atol=3e-14, rtol=1e-5)
    assert value.min() > 0 and g.sum(-1).abs().max() < 1e-20
    zero, g, _ = forward_rows(q, q)
    assert torch.count_nonzero(zero) == torch.count_nonzero(g) == 0


def test_forward_loss_uses_forward_total_and_global_token_weight():
    cfg = DistillationLossConfig(
        loss_mode="full_forward_kl",
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    assert cfg.loss_settings.anchor_payload is None and cfg.loss_settings.teacher_only_forward_kl
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=11)
    f = torch.tensor(3.0, requires_grad=True)
    r = torch.tensor(9.0)
    loss, metrics = distillation_loss(
        SimpleNamespace(loss_agg_mode="token-mean"),
        DistillationConfig(distillation_loss=cfg),
        dict(full_forward_kl_sum=f, full_forward_kl_count=3, forward_statistics={"full_kl": r}),
        data,
    )
    loss.backward()
    assert f.grad.item() == pytest.approx(2 / 11)
    assert "distillation/full_forward_kl" in metrics and "distillation/full_kl" in metrics
    with pytest.raises(ValueError, match="direct loss"):
        DistillationLossConfig(loss_mode="full_forward_kl")
    with pytest.raises(ValueError, match="supports"):
        DistillationLossConfig(
            loss_mode="full_forward_kl",
            use_policy_gradient=False,
            use_task_rewards=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            full_vocab=FullVocabConfig(backend="liger"),
        )
