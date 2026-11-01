# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Static objective metadata, importable without tensor kernels or worker configs."""

import math
from dataclasses import dataclass

from verl.base_config import BaseConfig


@dataclass
class MirrorConfig(BaseConfig):
    objective: str = "mdpo"
    estimator: str = "full_vocab"
    kappa: float = 0.5
    eta: float = 0.3
    diagnostics: bool = False

    def __post_init__(self):
        if self.objective not in ("mdpo", "bpo") or self.estimator not in ("sampled", "full_vocab"):
            raise ValueError("mirror requires mdpo/bpo and sampled/full_vocab")
        if self.objective == "bpo" and self.estimator != "full_vocab":
            raise ValueError("sampled BPO is not supported")
        if not math.isfinite(self.kappa) or not 0 <= self.kappa <= 1:
            raise ValueError("mirror.kappa must be finite and in [0, 1]")
        if not math.isfinite(self.eta) or not 0 < self.eta <= 1:
            raise ValueError("mirror.eta must be finite and in (0, 1]")


@dataclass
class DistillationLossSettings(BaseConfig):
    names: tuple[str, ...] = ()
    payload_kind: str = "sampled"
    objective: str | None = None
    kappa: float = 0.0
    eta: float = 1.0
    diagnostics: bool = False

    @property
    def anchor_payload(self):
        if self.objective in (None, "forward_kl") or (self.objective == "mdpo" and self.eta == 1):
            return None
        return "hidden" if self.payload_kind == "hidden" else "scalar"

    @property
    def teacher_only_kl(self):
        return self.objective == "mdpo" and self.eta == 1 and self.payload_kind == "hidden"

    @property
    def teacher_only_forward_kl(self):
        return self.objective == "forward_kl" and self.payload_kind == "hidden"

    @property
    def use_topk(self):
        return self.payload_kind == "topk"

    @property
    def use_estimator(self):
        return self.payload_kind == "sampled"

    def __post_init__(self):
        names = (self.names,) if isinstance(self.names, str) else tuple(self.names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Objective metadata requires nonempty registration names")
        object.__setattr__(self, "names", names)
        if self.payload_kind not in ("sampled", "topk", "hidden"):
            raise ValueError(f"Unknown teacher payload: {self.payload_kind}")


_BUILTINS = (
    DistillationLossSettings(names="full_forward_kl", payload_kind="hidden", objective="forward_kl"),
    DistillationLossSettings(names="full_reverse_kl", payload_kind="hidden", objective="mdpo"),
    DistillationLossSettings(names="forward_kl_topk", payload_kind="topk"),
    DistillationLossSettings(names=("kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"), payload_kind="sampled"),
)
DISTILLATION_SETTINGS_REGISTRY = {name: spec for spec in _BUILTINS for name in spec.names}


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY)}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def resolve_distillation_spec(config) -> DistillationLossSettings:
    """Resolve data requirements without importing tensor kernels or engine state."""
    mode = config.get("loss_mode", "k3")
    mirror = config.get("mirror")
    if mode != "mirror":
        if mirror is not None:
            raise ValueError("mirror settings require loss_mode=mirror")
        return get_distillation_loss_settings(mode)
    if mirror is None:
        raise ValueError("loss_mode=mirror requires mirror settings")
    if not isinstance(mirror, MirrorConfig):
        mirror = MirrorConfig(**{k: v for k, v in dict(mirror).items() if k != "_target_"})
    else:
        mirror.__post_init__()
    return DistillationLossSettings(
        names="mirror",
        payload_kind="hidden" if mirror.estimator == "full_vocab" else "sampled",
        objective=mirror.objective,
        kappa=mirror.kappa,
        eta=mirror.eta,
        diagnostics=mirror.diagnostics,
    )
