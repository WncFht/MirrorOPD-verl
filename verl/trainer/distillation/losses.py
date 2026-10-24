# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.policy_metrics import (
    applied_weight_observations,
    ppo_clip_observations,
    rejection_observations,
    update_observations,
)
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.utils.metric.observations import Observation
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import as_padded_response, no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    payload_kind: str | None = None

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if self.payload_kind is None:
            object.__setattr__(self, "payload_kind", "topk" if self.use_topk else "sampled")
        if self.payload_kind not in ("sampled", "topk", "hidden"):
            raise ValueError(f"Unknown teacher payload: {self.payload_kind}")
        if sum([self.use_topk, self.use_estimator, self.payload_kind == "hidden"]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Observation.minimum(distillation_losses_response),
        "distillation/loss_max": Observation.maximum(distillation_losses_response),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni" | "fsdp2":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if not distillation_loss_config.use_task_rewards and (
        not distillation_loss_config.use_policy_gradient or distillation_loss_config.policy_loss_mode != "vanilla"
    ):
        # no need to compute policy loss
        policy_loss = 0.0
        policy_metrics = {}
    else:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
        if not distillation_loss_config.use_task_rewards:
            policy_loss = 0.0
            policy_metrics = {}  # The task objective is not used; do not publish its unused loss or advantage.

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)
    if distillation_loss_config.use_task_rewards:
        policy_metrics["optim/actor/distillation_loss_weighted"] = Metric(
            value=distill_loss.detach() * distillation_loss_coef, aggregation=AggregationType.SUM
        )

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    if loss_config.loss_mode == "full_reverse_kl":
        if config.loss_agg_mode != "token-mean":
            raise ValueError("full_reverse_kl currently requires token-mean aggregation")
        total = model_output["full_kl_sum"]
        count = model_output["full_kl_count"]
        denominator = tu.get(data, "batch_num_tokens")
        if denominator <= 0:
            raise ValueError("full KL requires positive globally valid token count")
        loss = total * tu.get(data, "dp_size") / denominator
        metrics = {"distillation/full_kl": Observation.ratio(total.detach(), count)}
        if "full_kl_position_sums" in model_output:
            from verl.utils.metric.position import position_observations

            metrics.update(
                position_observations(
                    model_output["full_kl_position_sums"],
                    model_output["full_kl_position_counts"],
                    prefix="distillation/full_kl_position",
                )
            )
        return loss, metrics
    if loss_config.policy_loss_mode != "vanilla":
        return _sampled_token_loss(config, loss_config, model_output, data)
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        if rollout_is_weights is not None and rollout_is_weights.is_nested:
            rollout_is_weights = rollout_is_weights.to_padded_tensor(0.0)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        count = response_mask.sum()
        pg_metrics = {
            f"distillation/{key.removeprefix('actor/')}": Observation.ratio(
                torch.where(count > 0, torch.as_tensor(value, device=response_mask.device) * count, 0.0), count
            )
            for key, value in pg_metrics.items()
        }
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    if (
        loss_config.use_policy_gradient
        and loss_config.loss_mode == "k1"
        and tu.get_non_tensor_data(data, "collect_metrics", default=False)
    ):
        behavior = data.get("rollout_log_probs")
        if behavior is not None and behavior.is_nested:
            behavior = behavior.to_padded_tensor(0.0)
        teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
        raw_mask = data.get("metrics_response_mask")
        if raw_mask is not None and raw_mask.is_nested:
            raw_mask = raw_mask.to_padded_tensor(False)
        distillation_metrics.update(
            update_observations(
                log_prob,
                behavior,
                old_log_prob if tu.get_non_tensor_data(data, "metrics_has_anchor", default=True) else None,
                -distillation_losses.detach(),
                response_mask,
                component="distillation",
                observation_mask=raw_mask,
                teacher=teacher,
                mode="vanilla",
                quantile=tu.get_non_tensor_data(data, "metrics_quantile", default=False),
            )
        )
        low = loss_config.clip_ratio if loss_config.clip_ratio_low is None else loss_config.clip_ratio_low
        high = loss_config.clip_ratio if loss_config.clip_ratio_high is None else loss_config.clip_ratio_high
        scale = torch.ones_like(log_prob)
        if loss_agg_mode == "seq-mean-token-mean":
            scale = scale / response_mask.sum(-1, keepdim=True).clamp_min(1)
        elif loss_agg_mode == "seq-mean-token-sum-norm":
            scale = scale / (config.loss_scale_factor or response_mask.shape[-1])
        distillation_metrics.update(
            ppo_clip_observations(
                log_prob,
                old_log_prob,
                -distillation_losses.detach(),
                response_mask,
                low,
                high,
                loss_config.get("clip_ratio_c", 3.0),
                prefix="correction/anchor_ppo/"
                if tu.get_non_tensor_data(data, "metrics_has_anchor", default=True)
                else "correction/distillation_behavior_ppo/",
                weights=rollout_is_weights,
                scale=scale,
            )
        )
        if raw_mask is not None:
            distillation_metrics.update(rejection_observations(response_mask, raw_mask))
        if rollout_is_weights is not None:
            distillation_metrics.update(
                applied_weight_observations(
                    rollout_is_weights,
                    response_mask,
                    response_mask if raw_mask is None else raw_mask,
                    quantile=tu.get_non_tensor_data(data, "metrics_quantile", default=False),
                )
            )
    return distillation_loss, distillation_metrics


def _sampled_token_loss(config, loss_config, model_output, data):
    """Response-token surrogate with a generation-time behavior denominator.

    Advantages are detached in all modes. IS modes differentiate the ratio;
    reinforce differentiates log pi directly. Mask before arithmetic so padded
    NaNs/infinities neither contaminate the objective nor its gradients.
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data).float()
    teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1).detach().float()
    behavior = data["rollout_log_probs"].detach()
    mask = data["response_mask"]
    mask = as_padded_response(mask, False).bool()
    behavior = as_padded_response(behavior).float()
    if not (log_prob.shape == teacher.shape == behavior.shape == mask.shape):
        raise ValueError("Sampled-token log probabilities must match the response mask shape")
    log_prob = log_prob.masked_fill(~mask, 0.0)
    teacher = teacher.masked_fill(~mask, 0.0)
    behavior = behavior.masked_fill(~mask, 0.0)
    current_advantage = (teacher - log_prob).detach()
    frozen_advantage = teacher - behavior
    advantage = current_advantage if loss_config.advantage_source == "current" else frozen_advantage
    metrics = {
        "distillation/token_observations": Metric("sum", mask.sum()),
        "distillation/advantage_sign_flip_count": Metric(
            "sum", ((current_advantage * frozen_advantage < 0) & mask).sum()
        ),
    }
    clip_active = None
    if loss_config.policy_loss_mode == "reinforce":
        loss_mat = -advantage * log_prob
    else:
        log_ratio = log_prob - behavior
        ratio = log_ratio.clamp(-20.0, 20.0).exp()
        loss_mat = -ratio * advantage
        metrics["distillation/log_ratio_guard_count"] = Metric("sum", ((log_ratio.abs() > 20) & mask).sum())
        if loss_config.policy_loss_mode == "token_ppo":
            low = loss_config.clip_ratio if loss_config.clip_ratio_low is None else loss_config.clip_ratio_low
            high = loss_config.clip_ratio if loss_config.clip_ratio_high is None else loss_config.clip_ratio_high
            clipped_loss = -ratio.clamp(1 - low, 1 + high) * advantage
            clip_active = clipped_loss > loss_mat
            metrics["distillation/ppo_clip_count"] = Metric("sum", (clip_active & mask).sum())
            loss_mat = torch.maximum(loss_mat, clipped_loss)
    loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=mask,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=tu.get(data, "dp_size"),
        batch_num_tokens=tu.get(data, "batch_num_tokens"),
        global_batch_size=tu.get(data, "global_batch_size"),
        loss_scale_factor=config.loss_scale_factor,
    )
    if tu.get_non_tensor_data(data, "collect_metrics", default=False):
        anchor = as_padded_response(data.get("old_log_probs"))
        coefficient_scale = torch.ones_like(log_prob)
        if config.loss_agg_mode == "seq-mean-token-mean":
            coefficient_scale = coefficient_scale / (mask.sum(-1, keepdim=True).float() + 1e-8)
        elif config.loss_agg_mode == "seq-mean-token-sum-norm":
            scale = config.loss_scale_factor if config.loss_scale_factor is not None else mask.shape[-1]
            coefficient_scale = coefficient_scale / scale
        metrics.update(
            update_observations(
                log_prob,
                behavior,
                anchor,
                advantage,
                mask,
                component="distillation",
                teacher=teacher,
                mode=loss_config.policy_loss_mode,
                clip_active=clip_active,
                coefficient_scale=coefficient_scale,
                quantile=tu.get_non_tensor_data(data, "metrics_quantile", default=False),
            )
        )
    return loss, metrics


@register_distillation_loss(DistillationLossSettings(names="full_reverse_kl", payload_kind="hidden"))
def compute_full_reverse_kl(config, distillation_config, model_output, data):
    raise RuntimeError("full_reverse_kl must use the head-loss path, not a sampled estimator")


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = Observation.mean(valid_overlap_count.float() / k)
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        overlap_metrics["distillation/overlap_token_advantage"] = Observation.mean(
            overlap_token_advantage[overlap_position_mask]
        )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": Observation.mean(student_mass),
        "distillation/student_mass_min": Observation.minimum(student_mass),
        "distillation/student_mass_max": Observation.maximum(student_mass),
        "distillation/teacher_mass": Observation.mean(teacher_mass),
        "distillation/teacher_mass_min": Observation.minimum(teacher_mass),
        "distillation/teacher_mass_max": Observation.maximum(teacher_mass),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Observation.mean(distillation_losses[response_mask_bool].abs()),
    }
    return distillation_losses, metrics
