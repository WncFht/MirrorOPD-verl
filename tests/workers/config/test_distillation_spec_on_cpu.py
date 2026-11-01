# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import pytest

from verl.workers.config.distillation_spec import get_distillation_loss_settings


def test_config_construction_does_not_import_tensor_loss_implementations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from verl.workers.config.distillation import DistillationLossConfig; "
            "c=DistillationLossConfig(); assert c.loss_settings.payload_kind=='sampled'; "
            "assert 'verl.trainer.distillation.losses' not in sys.modules",
        ],
        check=True,
    )


def test_only_canonical_metadata_names_are_registered():
    from verl.trainer.distillation.losses import get_distillation_loss_fn

    assert callable(get_distillation_loss_fn("k1"))
    assert callable(get_distillation_loss_fn("k3"))
    for retired in ("kl", "low_var_kl"):
        with pytest.raises(ValueError, match="Unsupported"):
            get_distillation_loss_settings(retired)
        with pytest.raises(ValueError, match="Unsupported"):
            get_distillation_loss_fn(retired)
    with pytest.raises(ValueError, match="Unsupported"):
        get_distillation_loss_settings("full_reverse_kl")
    assert get_distillation_loss_settings("forward_kl_topk").payload_kind == "topk"
    with pytest.raises(ValueError, match="Unsupported loss mode"):
        get_distillation_loss_settings("unknown_objective")


@pytest.mark.parametrize("names", ["k1", ["k1"], ("k1", "k1"), ()])
def test_metadata_has_no_implicit_name_format_conversion(names):
    from verl.workers.config.distillation_spec import DistillationLossSettings

    with pytest.raises(ValueError, match="names"):
        DistillationLossSettings(names=names)


def test_resolved_metadata_is_not_user_configuration_and_mirror_is_typed():
    from dataclasses import FrozenInstanceError, asdict

    from verl.workers.config.distillation import DistillationLossConfig
    from verl.workers.config.distillation_spec import MirrorConfig

    config = DistillationLossConfig(
        loss_mode="mirror",
        mirror={"objective": "mdpo", "eta": 1},
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    assert isinstance(config.mirror, MirrorConfig)
    assert config.loss_settings.teacher_only_kl
    assert "loss_settings" not in asdict(config)
    with pytest.raises(FrozenInstanceError):
        config.mirror.eta = 0.3
    with pytest.raises(TypeError, match="loss_settings"):
        DistillationLossConfig(loss_settings={"payload_kind": "hidden"})
