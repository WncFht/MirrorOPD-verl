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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.policy_metrics import applied_weight_observations, ppo_clip_observations, rejection_observations
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.metric.observations import Observation
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
    sc_outputs = {
        key: no_padding_2_padding(model_output[key], data)
        for key in (
            "sc_correction",
            "sc_sampler_head_mass",
            "sc_train_head_mass",
            "sc_support_aligned",
            "sc_sampler_tail_mass",
            "sc_support_unknown",
        )
        if key in model_output
    }

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}
    if (
        tu.get_non_tensor_data(data, "collect_metrics", default=False)
        and config.policy_loss.get("loss_mode", "vanilla") != "bypass_mode"
    ):
        from verl.trainer.ppo.policy_metrics import update_observations

        def padded(value):
            return value.to_padded_tensor(0.0) if value is not None and value.is_nested else value

        metrics.update(
            update_observations(
                log_prob,
                padded(data.get("rollout_log_probs")),
                padded(data["old_log_probs"])
                if tu.get_non_tensor_data(data, "metrics_has_anchor", default=True)
                else None,
                padded(data["advantages"]),
                padded(data["response_mask"]),
                component="task",
                observation_mask=padded(data.get("metrics_response_mask")),
                quantile=tu.get_non_tensor_data(data, "metrics_quantile", default=False),
            )
        )

    raw_mask = data.get("metrics_response_mask")
    if raw_mask is not None and raw_mask.is_nested:
        raw_mask = raw_mask.to_padded_tensor(False)
    collect_metrics = tu.get_non_tensor_data(data, "collect_metrics", default=False)
    quantile = tu.get_non_tensor_data(data, "metrics_quantile", default=False)
    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    policy_loss_kwargs = {"sc_correction": sc_outputs["sc_correction"]} if sc_outputs else {}
    if loss_mode == "bypass_mode":
        policy_loss_kwargs.update(collect_metrics=collect_metrics, metrics_quantile=quantile)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        **policy_loss_kwargs,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = {
        key: value
        if isinstance(value, Observation)
        else Observation.ratio(
            torch.as_tensor(value, device=response_mask.device) * response_mask.sum(), response_mask.sum()
        )
        for key, value in pg_metrics.items()
    }

    metrics.update(pg_metrics)
    if collect_metrics:
        if raw_mask is not None:
            metrics.update(rejection_observations(response_mask, raw_mask))
        if rollout_is_weights is not None:
            metrics.update(
                applied_weight_observations(
                    rollout_is_weights,
                    response_mask,
                    response_mask if raw_mask is None else raw_mask,
                    quantile=quantile,
                )
            )
        if loss_mode == "vanilla":
            low = config.clip_ratio if config.clip_ratio_low is None else config.clip_ratio_low
            high = config.clip_ratio if config.clip_ratio_high is None else config.clip_ratio_high
            scale = torch.ones_like(log_prob)
            if loss_agg_mode == "seq-mean-token-mean":
                scale = scale / response_mask.sum(-1, keepdim=True).clamp_min(1)
            elif loss_agg_mode == "seq-mean-token-sum-norm":
                scale = scale / (config.loss_scale_factor or response_mask.shape[-1])
            metrics.update(
                ppo_clip_observations(
                    log_prob,
                    old_log_prob,
                    advantages,
                    response_mask,
                    low,
                    high,
                    config.get("clip_ratio_c", 3.0),
                    prefix="correction/task_ppo/",
                    weights=rollout_is_weights,
                    scale=scale,
                )
            )
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    if sc_outputs:
        for key in (
            "sc_sampler_head_mass",
            "sc_train_head_mass",
            "sc_support_aligned",
            "sc_sampler_tail_mass",
            "sc_support_unknown",
        ):
            if key not in sc_outputs:
                continue
            if key in ("sc_support_aligned", "sc_sampler_tail_mass", "sc_support_unknown"):
                # Sum globally normalized token contributions across microbatches; the
                # worker's DP mean then yields a token-weighted fraction, even for uneven rows.
                global_tokens = config.global_batch_info["batch_num_tokens"]
                value = agg_loss(
                    sc_outputs[key],
                    response_mask,
                    "token-mean",
                    dp_size=config.global_batch_info["dp_size"],
                    batch_num_tokens=global_tokens,
                )
                aggregation = AggregationType.SUM if global_tokens is not None else AggregationType.MEAN
            else:
                value = masked_mean(sc_outputs[key], response_mask)
                aggregation = AggregationType.MEAN
            metrics[f"actor/{key}"] = Metric(value=value, aggregation=aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
        metrics["optim/actor/entropy_loss_weighted"] = Metric(
            value=-entropy_coeff * entropy_loss.detach(), aggregation=metric_aggregation
        )
        metrics["policy/entropy/update"] = Observation.mean(entropy.detach()[response_mask])

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef
        metrics["optim/actor/kl_loss_weighted"] = Metric(
            value=kl_loss.detach() * config.kl_loss_coef, aggregation=metric_aggregation
        )
        metrics["policy/current_reference/sampled_kl"] = Observation.mean(kld.detach()[response_mask])

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # Normalize the value loss over the global mini-batch (dp_size / batch_num_tokens /
    # global_batch_size) instead of the local micro-batch, so the accumulated critic gradient is
    # invariant to how the mini-batch is split into micro-batches (as the actor's ppo_loss does).
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]
    global_batch_size = data["global_batch_size"]

    # When the loss is normalized over the global batch, each micro-batch contributes a partial sum,
    # so the loss metric must be aggregated with SUM to reflect the global-batch mean.
    if (
        dp_size > 1
        or batch_num_tokens is not None
        or global_batch_size is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=config.loss_scale_factor,
    )

    metrics = {
        "critic/vf_loss": Metric(value=vf_loss, aggregation=metric_aggregation),
        "critic/vf_clipfrac": Observation.ratio(vf_clipfrac.detach() * response_mask.sum(), response_mask.sum()),
        "critic/vpred_mean": Observation.mean(vpreds.detach()[response_mask]),
    }

    return vf_loss, metrics
