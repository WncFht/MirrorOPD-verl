# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent objective, VJP, endpoint, configuration and frozen-state checks."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.distillation.anchor import FrozenAnchorHead, anchor_scope
from verl.trainer.distillation.full_vocab import full_reverse_kl_result, full_vocab_objective_result
from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.distillation.objectives import full_vocab_rows, mdpo_sampled_terms
from verl.trainer.distillation.targets import build_log_weights
from verl.workers.config.distillation import DistillationLossConfig, FullVocabConfig
from verl.workers.config.distillation_spec import resolve_distillation_spec


def cfg(objective="mdpo", estimator="full_vocab", eta=0.3, kappa=0.5, backend="precompute", diagnostics=False):
    return DistillationLossConfig(
        loss_mode="mirror",
        mirror=dict(objective=objective, estimator=estimator, eta=eta, kappa=kappa, diagnostics=diagnostics),
        full_vocab=FullVocabConfig(backend=backend),
        use_task_rewards=False,
        use_policy_gradient=estimator == "sampled",
        policy_loss_mode="token_is" if estimator == "sampled" else "vanilla",
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )


def target_reference(lq, lt, kappa, eta):
    """Independent power-mean numerator; small FP64 inputs avoid overflow."""
    if eta == 1:
        return lt.detach()
    if kappa == 0:
        return (1 - eta) * lq.detach() + eta * lt.detach()
    return ((1 - eta) * (kappa * lq.detach()).exp() + eta * (kappa * lt.detach()).exp()).log() / kappa


@pytest.mark.parametrize("kappa", [0, 1e-7, 0.5, 1])
@pytest.mark.parametrize("eta", [0.2, 0.8, 1])
def test_targets_and_sampled_expectation(kappa, eta):
    torch.manual_seed(14)
    lq, lt, lb = [torch.randn(7, dtype=torch.float64).log_softmax(-1) for _ in range(3)]
    z = torch.randn(7, dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    b = build_log_weights(lt, lq if eta != 1 else None, kappa=kappa, eta=eta)
    torch.testing.assert_close(b, target_reference(lq, lt, kappa, eta), atol=3e-9, rtol=1e-9)
    exact = (lp.exp() * (lp - b)).sum() / eta
    sampled = (lb.exp() * mdpo_sampled_terms(lp, lb, b)).sum() / eta
    torch.testing.assert_close(sampled, exact)
    torch.testing.assert_close(torch.autograd.grad(sampled, z, retain_graph=True)[0], torch.autograd.grad(exact, z)[0])


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
@pytest.mark.parametrize("backend", ["dense", "precompute", "precompute_fused"])
@pytest.mark.parametrize("tokens,extra", [(0, 0), (1, 3), (7, 3)])
def test_full_value_and_both_vjps(objective, backend, tokens, extra):
    torch.manual_seed(42)
    hs, ws = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in [(tokens, 5), (11, 5)]]
    ht, wt, hq, wq = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in [(tokens, 4), (11 + extra, 4), (tokens, 5), (11, 5)]
    ]
    config = cfg(objective, diagnostics=True)
    lp = (hs @ ws.T).log_softmax(-1)
    lt = (ht.detach() @ wt.detach().T).log_softmax(-1)[:, :11]
    lq = (hq.detach() @ wq.detach().T).log_softmax(-1)
    b = target_reference(lq, lt, 0.5, 0.3)
    residual = lp - b
    if objective == "mdpo":
        expected = (lp.exp() * residual).sum()
    else:
        q = lq.exp()
        expected = (q * (residual - (q * residual).sum(-1, keepdim=True)).square()).sum() / 2
    result = full_vocab_objective_result(
        hs, ws, ht, wt, hq=hq, wq=wq, spec=config.loss_settings, backend=backend, token_chunk=3, vocab_chunk=4
    )
    torch.testing.assert_close(result.loss_sum, expected, atol=1e-10, rtol=1e-10)
    grad = torch.autograd.grad(result.loss_sum * 0.37 / 0.3, (hs, ws, ht, wt, hq, wq), allow_unused=True)
    ref = torch.autograd.grad(expected * 0.37 / 0.3, (hs, ws))
    for actual, wanted in zip(grad[:2], ref, strict=True):
        torch.testing.assert_close(actual, wanted, atol=1e-10, rtol=1e-10)
    assert grad[2:] == (None,) * 4
    if tokens:
        lm = b.log_softmax(-1)
        torch.testing.assert_close(result.token_diagnostics["target_kl"], (lp.exp() * (lp - lm)).sum(-1))
        torch.testing.assert_close(result.token_diagnostics["teacher_kl"], (lp.exp() * (lp - lt)).sum(-1))


@pytest.mark.parametrize("backend", ["dense", "recompute", "cached", "streaming", "precompute", "precompute_fused"])
def test_endpoint_uses_original_kernel_without_anchor(backend):
    torch.manual_seed(2)
    hs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn(3, 6, dtype=torch.float64), torch.randn(12, 6, dtype=torch.float64)
    spec = cfg(eta=1, backend=backend).loss_settings
    assert spec.anchor_payload is None and spec.teacher_only_kl
    old = full_reverse_kl_result(hs, ws, ht, wt, backend=backend, token_chunk=2, vocab_chunk=4)
    new = full_vocab_objective_result(hs, ws, ht, wt, spec=spec, backend=backend, token_chunk=2, vocab_chunk=4)
    torch.testing.assert_close(new.loss_sum, old.loss_sum, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(new.loss_sum, (hs, ws), retain_graph=True),
        torch.autograd.grad(old.loss_sum, (hs, ws)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert cfg("bpo", eta=1).loss_settings.anchor_payload == "hidden"


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
def test_dense_rows_and_stable_small_bpo(objective):
    torch.manual_seed(7)
    z = torch.randn(3, 9, dtype=torch.float64, requires_grad=True)
    lq = torch.randn_like(z).log_softmax(-1)
    b = z.detach() + 1e-7 * torch.randn_like(z)
    value, dz, _ = full_vocab_rows(z, b, lq, objective=objective)
    torch.testing.assert_close(dz, torch.autograd.grad(value.sum(), z)[0], atol=1e-14, rtol=1e-7)
    if objective == "bpo":
        assert (value >= 0).all() and value.max() < 1e-12
        shifted, _, _ = full_vocab_rows(z + 3, b - 2, lq, objective="bpo")
        torch.testing.assert_close(value, shifted, atol=1e-20, rtol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(objective="bpo", estimator="sampled"),
        dict(eta=0),
        dict(eta=1.1),
        dict(kappa=-1),
        dict(eta=float("nan")),
        dict(kappa=float("inf")),
        dict(backend="recompute"),
        dict(objective="unknown"),
    ],
)
def test_invalid_combinations(kwargs):
    with pytest.raises(ValueError):
        cfg(**kwargs)


def test_metadata_resolution_and_conflicts():
    assert resolve_distillation_spec(OmegaConf.create({"loss_mode": "full_reverse_kl"})).teacher_only_kl
    with pytest.raises(ValueError, match="require loss_mode"):
        resolve_distillation_spec({"loss_mode": "full_reverse_kl", "mirror": {"eta": 0.3}})
    with pytest.raises(ValueError, match="requires mirror"):
        resolve_distillation_spec({"loss_mode": "mirror"})


def test_snapshot_owns_storage_and_endpoint_never_runs_inference():
    w = torch.randn(4, 3)
    frozen = FrozenAnchorHead(w, (1, 0))
    expected = w.clone()
    w.add_(10)
    torch.testing.assert_close(frozen.get("cpu", torch.float32), expected)
    actor = SimpleNamespace(infer_batch=lambda _: pytest.fail("endpoint must skip anchor"))
    with anchor_scope(actor, {}, cfg(eta=1).loss_settings):
        pass


def test_sampled_loss_mask_alignment_and_scale():
    from test_sampled_token_loss_on_cpu import inputs

    from verl.utils import tensordict_utils as tu

    lp = torch.tensor([[-0.7, -1.3], [-0.5, float("nan")]], requires_grad=True)
    lt = torch.tensor([[-0.6, -1.1], [-0.9, float("nan")]])
    lb = torch.tensor([[-0.8, -1.2], [-0.7, float("nan")]])
    out, data = inputs(lp, lt, lb, lengths=[2, 1], prompt_lengths=[2, 3])
    lq = torch.tensor([[-0.9, -1.0], [-0.6, float("nan")]])
    data["old_log_probs"] = lq
    config = cfg(estimator="sampled")
    actor = SimpleNamespace(loss_agg_mode="token-mean")
    loss, metrics = distillation_loss(actor, SimpleNamespace(distillation_loss=config), out, data)
    mask = data["response_mask"]
    b = build_log_weights(lt[mask], lq[mask], kappa=0.5, eta=0.3)
    expected = mdpo_sampled_terms(lp[mask], lb[mask], b).sum() / (3 * 0.3)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, lp, retain_graph=True)[0], torch.autograd.grad(expected, lp)[0]
    )
    assert "distillation/mdpo/objective" in metrics
    tu.assign_non_tensor(data, collect_metrics=True)
    observed, _ = distillation_loss(actor, SimpleNamespace(distillation_loss=config), out, data)
    torch.testing.assert_close(observed, loss)


@pytest.mark.parametrize("objective", ["mdpo", "bpo"])
def test_diagnostics_do_not_change_loss_or_gradients(objective):
    torch.manual_seed(11)
    hs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ws = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    ht, wt = torch.randn(3, 5, dtype=torch.float64), torch.randn(12, 5, dtype=torch.float64)
    hq, wq = torch.randn_like(hs), torch.randn_like(ws)
    values, gradients = [], []
    before = torch.random.get_rng_state().clone()
    for diagnostics in (False, True):
        result = full_vocab_objective_result(
            hs, ws, ht, wt, hq=hq, wq=wq, spec=cfg(objective, diagnostics=diagnostics).loss_settings, token_chunk=2
        )
        values.append(result.loss_sum)
        gradients.append(torch.autograd.grad(result.loss_sum, (hs, ws)))
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_anchor_scope_failure_releases_partially_created_head():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    rows = lambda values: torch.nested.nested_tensor(values, layout=torch.jagged)
    data = TensorDict(
        {
            "input_ids": rows([torch.tensor([1, 2])]),
            "position_ids": rows([torch.arange(2)]),
            "loss_mask": rows([torch.ones(1)]),
            "response_mask": rows([torch.ones(1)]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, mirror_anchor_version=(2, 0), full_vocab_train=True)
    engine = SimpleNamespace(module=torch.nn.Linear(2, 2))

    def fail(prepared):
        engine.mirror_anchor_head = FrozenAnchorHead(engine.module.weight, (2, 0))
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        with anchor_scope(SimpleNamespace(engine=engine, infer_batch=fail), data, cfg().loss_settings):
            pytest.fail("failed preparation must not begin training")
    assert engine.mirror_anchor_head is None
    assert tu.get_non_tensor_data(data, "full_vocab_train", default=False)


@pytest.mark.parametrize("objective,estimator", [("mdpo", "sampled"), ("mdpo", "full_vocab"), ("bpo", "full_vocab")])
def test_hydra_config_roundtrip(objective, estimator):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass

    root = Path(__file__).resolve().parents[3]
    sampled = estimator == "sampled"
    with initialize_config_dir(config_dir=str(root / "verl/trainer/config"), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "distillation.distillation_loss.loss_mode=mirror",
                "distillation.distillation_loss.mirror={objective:"
                + objective
                + ",estimator:"
                + estimator
                + ",kappa:0.5,eta:0.3}",
                "distillation.distillation_loss.full_vocab.backend=precompute",
                "distillation.distillation_loss.use_task_rewards=False",
                "distillation.distillation_loss.use_policy_gradient=" + str(sampled),
                "distillation.distillation_loss.policy_loss_mode=" + ("token_is" if sampled else "vanilla"),
            ],
        )
        parsed = omega_conf_to_dataclass(config.distillation.distillation_loss)
    assert parsed.loss_settings.objective == objective
    assert parsed.loss_settings.anchor_payload == ("scalar" if sampled else "hidden")


@pytest.mark.parametrize("fail", [False, True])
def test_worker_endpoint_and_teacher_cleanup(fail):
    import inspect

    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine_workers import ActorRolloutRefWorker

    events = []
    head = SimpleNamespace(release=lambda: events.append("release"))

    def train(data):
        events.append("train")
        if fail:
            raise RuntimeError("training failed")
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {}})

    worker = SimpleNamespace(
        mirror_spec=cfg(eta=1).loss_settings,
        actor=SimpleNamespace(
            engine=SimpleNamespace(full_vocab_teacher_head=head),
            train_mini_batch=train,
            infer_batch=lambda _: pytest.fail("endpoint must not score anchor"),
        ),
    )
    update = inspect.unwrap(ActorRolloutRefWorker.update_actor)
    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            update(worker, TensorDict({}, batch_size=[]))
    else:
        update(worker, TensorDict({}, batch_size=[]))
    assert events == ["train", "release"]


def test_worker_anchor_scope_and_preparation_metric():
    import inspect

    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine_workers import ActorRolloutRefWorker

    def nested(values):
        return torch.nested.nested_tensor(values, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested([torch.tensor([1, 2])]),
            "position_ids": nested([torch.arange(2)]),
            "loss_mask": nested([torch.ones(1)]),
            "response_mask": nested([torch.ones(1)]),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, mirror_anchor_version=(5, 0), full_vocab_train=True)
    events = []
    engine = SimpleNamespace(
        module=torch.nn.Linear(2, 2), full_vocab_teacher_head=SimpleNamespace(release=lambda: events.append("release"))
    )

    def infer(prepared):
        events.append("infer")
        engine.mirror_anchor_head = FrozenAnchorHead(engine.module.weight, (5, 0))
        return TensorDict(
            {
                "anchor_hidden": nested([torch.ones(1, 2)]),
                "anchor_positions": nested([torch.tensor([0])]),
                "anchor_version": torch.tensor([[5, 0]]),
            },
            batch_size=[1],
        )

    def train(data):
        events.append("train")
        assert "anchor_hidden" in data and engine.mirror_anchor_head is not None
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {}})

    worker = SimpleNamespace(
        mirror_spec=cfg("bpo", eta=1).loss_settings,
        actor=SimpleNamespace(engine=engine, infer_batch=infer, train_mini_batch=train),
    )
    out = inspect.unwrap(ActorRolloutRefWorker.update_actor)(worker, data)
    assert events == ["infer", "train", "release"]
    assert engine.mirror_anchor_head is None and "anchor_hidden" not in data
    assert tu.get(out, "metrics")["perf/anchor/prepare_seconds_max"].aggregate() >= 0


def test_shared_sampled_primitive_preserves_guard_difference():
    from verl.trainer.distillation.objectives import sampled_policy_terms

    for beta in (-2.0, -30.0):
        lp = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
        lb, lt = lp.new_tensor([beta]), lp.new_tensor([-2.0])
        old, _, guarded = sampled_policy_terms(lp, lb, lt - lp, mode="token_is", log_ratio_guard=20)
        mirror = mdpo_sampled_terms(lp, lb, lt)
        go = torch.autograd.grad(old.sum(), lp, retain_graph=True)[0]
        gm = torch.autograd.grad(mirror.sum(), lp)[0]
        if beta == -2.0:
            torch.testing.assert_close(old, mirror, rtol=0, atol=0)
            torch.testing.assert_close(go, gm, rtol=0, atol=0)
            assert not guarded.any()
        else:
            assert guarded.all() and go.item() == 0 and gm.item() != 0
