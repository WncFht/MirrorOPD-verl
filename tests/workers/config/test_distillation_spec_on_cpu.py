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


def test_alias_metadata_matches_runtime_registration():
    from verl.trainer.distillation.losses import get_distillation_loss_fn

    assert get_distillation_loss_fn("k1") is get_distillation_loss_fn("kl")
    assert get_distillation_loss_fn("k3") is get_distillation_loss_fn("low_var_kl")
    assert get_distillation_loss_settings("full_reverse_kl").payload_kind == "hidden"
    assert get_distillation_loss_settings("forward_kl_topk").payload_kind == "topk"
    with pytest.raises(ValueError, match="Unsupported loss mode"):
        get_distillation_loss_settings("unknown_objective")
