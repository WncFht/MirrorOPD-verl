# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only prefix-probability probes on a separately resumed OPD branch."""

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from verl.trainer.ppo.readout_callback import ReadoutCallback

INNER = (0, 1, 2, 4, 8, 12, 16)
HORIZONS = (0, 32, 128, 256, 512, 1024, 2048, 4096, 6144)


def weight_summary(log_weights):
    """Stable empirical moments; do not interpret survivor samples as population moments."""
    values = np.asarray(log_weights, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("A nonempty finite log-weight sample is required")
    maximum = float(values.max())
    scaled = np.exp(values - maximum)
    sum1, sum2 = scaled.sum(), np.square(scaled).sum()
    logmean = maximum + np.log(sum1 / len(values))
    logsecond = 2 * maximum + np.log(sum2 / len(values))
    normalized = scaled / sum1
    ess = float(sum1 * sum1 / sum2)
    result = {
        "n": len(values),
        "ess": ess,
        "ess_fraction": ess / len(values),
        "log_mean_weight": float(logmean),
        "log_second_moment": float(logsecond),
        "mean_weight": float(np.exp(logmean)) if logmean < 700 else None,
        "second_moment": float(np.exp(logsecond)) if logsecond < 700 else None,
        "cv_squared": len(values) / ess - 1,
        "max_normalized_weight": float(normalized.max()),
        "top10pct_weight_mass": float(np.sort(normalized)[-max(1, math.ceil(len(values) / 10)) :].sum()),
        "logweight_quantiles": np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
    }
    if result["second_moment"] is not None and result["mean_weight"] is not None:
        result["empirical_variance"] = max(0.0, result["second_moment"] - result["mean_weight"] ** 2)
    for cap in (2, 5, 10):
        clipped_logs = np.minimum(values, np.log(cap))
        clipped_max = clipped_logs.max()
        clipped = np.exp(clipped_logs - clipped_max)
        retained_log_mass = clipped_max + np.log(clipped.sum()) - maximum - np.log(sum1)
        result[f"clip{cap}"] = {
            "ess_fraction": float(clipped.sum() ** 2 / np.square(clipped).sum() / len(values)),
            "removed_sample_weight_fraction": float(max(0.0, -np.expm1(retained_log_mass))),
        }
    return result


def prefix_table(current, reference, masks=None):
    """Prefix h includes actions 0..h-1; only trajectories with a state at h enter."""
    if len(current) != len(reference):
        raise ValueError("Trajectory count changed")
    if masks is None:
        masks = [[True] * len(row) for row in current]
    if len(masks) != len(current):
        raise ValueError("Mask count changed")
    cumulative = []
    for p, b in zip(current, reference, strict=True):
        p, b = np.asarray(p, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if p.shape != b.shape or not np.isfinite(p).all() or not np.isfinite(b).all():
            raise ValueError("Token probability shape or finiteness changed")
        cumulative.append(np.r_[0.0, np.cumsum(p - b, dtype=np.float64)])
    return {
        str(h): weight_summary([v[h] for v, m in zip(cumulative, masks, strict=True) if len(v) > h + 1 and m[h]])
        for h in HORIZONS
        if any(len(v) > h + 1 and m[h] for v, m in zip(cumulative, masks, strict=True))
    }


def selected_token_log_probs(hidden, weight, actions, legal_ids, token_chunk=128):
    """Bounded FP32 reference projection over the exact tokenizer support."""
    import torch

    if len(hidden) != len(actions) or token_chunk < 1:
        raise ValueError("Invalid selected-token projection dimensions")
    legal = torch.as_tensor(legal_ids, device=hidden.device, dtype=torch.long)
    inverse = torch.full((len(weight),), -1, device=hidden.device, dtype=torch.long)
    inverse[legal] = torch.arange(len(legal), device=hidden.device)
    targets = inverse[actions]
    if (targets < 0).any():
        raise ValueError("A sampled response token lies outside the behavior support")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        head = weight.index_select(0, legal).float()
        values = []
        for start in range(0, len(hidden), token_chunk):
            logits = hidden[start : start + token_chunk].float() @ head.T
            target = targets[start : start + token_chunk]
            chosen = logits.gather(-1, target[:, None]).squeeze(-1)
            values.append(chosen - torch.logsumexp(logits, dim=-1))
        return torch.cat(values)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PrefixISProbe(ReadoutCallback):
    """No reweighting: retain model/AdamW updates and measure their visitation drift."""

    handles_validation = True
    uses_validation_dataset = False
    requires_full_batch = True

    def __init__(self, *, config, options):
        from transformers import AutoConfig

        from verl.trainer.evaluation.artifacts import ArtifactStore
        from verl.trainer.evaluation.config import EvaluationConfig
        from verl.trainer.evaluation.trajectory import select_panel, trajectory_index

        self.config = config
        self.options = EvaluationConfig.parse(options)
        self.root = Path(self.options.output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows = None
        self.reference = None
        self.batch_count = 0
        self.hidden_size = AutoConfig.from_pretrained(
            config.distillation.teacher_models.teacher_model.model_path, local_files_only=True
        ).hidden_size
        self.refs = []
        for bank, child in (("public_initial", "initial"), ("public_teacher", "teacher")):
            refs = trajectory_index(ArtifactStore(Path(self.options.bootstrap) / child), bank=bank)
            self.refs.extend(select_panel(refs, self.options.datasets))
        atomic_json(self.root / "protocol.json", {"inner": INNER, "horizons": HORIZONS, "reweight_training": False})

    @staticmethod
    def preflight_generation_protocols(options):
        return {}

    def _banks(self, trainer, label):
        from verl.trainer.evaluation.scoring import score_native_rows
        from verl.trainer.evaluation.trajectory import load_trajectories

        trainer.rollout_service.park()
        live = 0 if self.rows is None else sum(sum(r["response_mask"]) for r in self.rows) * (self.hidden_size * 2 + 8)
        scores, timing = score_native_rows(
            trainer,
            self.refs,
            hidden_size=self.hidden_size,
            budget=self.options.score,
            live_payload_bytes=live,
            load_window=load_trajectories,
        )
        if len(scores) != len(self.refs) or any(r["tokens"] <= 0 or not math.isfinite(r["rkl_sum"]) for r in scores):
            raise RuntimeError("Diagnostic fixed-bank coverage is incomplete or nonfinite")
        groups = {}
        for r in scores:
            key = f"{r['bank']}/{r['dataset']}"
            groups.setdefault(key, []).append(r["rkl_sum"] / r["tokens"])
        metrics = {f"eval/rkl/{k}/question_mean": float(np.mean(v)) for k, v in groups.items()}
        atomic_json(
            self.root / f"banks-{label}.json",
            {"update": trainer.actor_optimizer_step, "metrics": metrics, "rows": scores, "timing": timing},
        )
        trainer.logger.log(data=metrics | trainer._metric_coordinates(), step=trainer.global_steps)
        return metrics

    def validate(self, trainer):
        metrics = self._banks(trainer, f"initial-u{trainer.actor_optimizer_step}")
        trainer.rollout_service.ensure_current(trainer.global_steps)
        return metrics

    def _probabilities(self, trainer):
        import torch
        import transfer_queue as tq

        from verl.trainer.evaluation.native import put_fixed_rows

        trainer.rollout_service.park()
        before = trainer.actor_optimizer_step
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        meta, by_key = put_fixed_rows(self.rows, trainer.global_steps)
        begun, settled = False, True
        try:
            meta = trainer._balance_batch(meta, {}, logging_prefix="prefix_probe", inference_only=True)
            meta.extra_info = {
                "temperature": 1.0,
                "compute_loss": False,
                "calculate_entropy": False,
                "prefix_is_score": True,
            }
            settled = False
            trainer.actor_rollout_wg.begin_rkl_readout(audit_state=False)
            begun, settled = True, True
            settled = False
            trainer.actor_rollout_wg.compute_log_prob(meta)
            settled = True
            data = tq.kv_batch_get(keys=meta.keys, partition_id=meta.partition_id, select_fields=["prefix_log_probs"])
            probs = data["prefix_log_probs"]
            scored = {}
            for key, lp, tag in zip(meta.keys, probs.unbind(), meta.tags, strict=True):
                if not tag.get("is_padding", False):
                    scored[by_key[key]["id"]] = lp.cpu().double().tolist()
            output = [scored[r["id"]] for r in self.rows]
            if trainer.actor_optimizer_step != before:
                raise RuntimeError("Readout advanced optimizer progress")
            return output
        finally:
            if begun and settled:
                trainer.actor_rollout_wg.end_rkl_readout()
            if settled:
                tq.kv_clear(keys=meta.keys, partition_id=meta.partition_id)
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])

    def on_inner_start(self, trainer, batch):
        import transfer_queue as tq

        from verl.trainer.evaluation.native import capture_rows

        self.rows = capture_rows(batch, trainer.global_steps)
        if len(self.rows) != 64:
            raise RuntimeError("Expected exactly 64 unique training trajectories")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"])
        mapping = {
            key: lp.cpu().double().tolist()
            for key, lp in zip(batch.keys, data["rollout_log_probs"].unbind(), strict=True)
        }
        self.behavior = [mapping[r["rollout_key"]] for r in self.rows]
        for row, lp in zip(self.rows, self.behavior, strict=True):
            if len(lp) != len(row["response_ids"]):
                raise RuntimeError("Behavior probabilities are not response-token aligned")
        self.batch_count += 1
        self.batch_root = self.root / f"outer-{self.batch_count:02d}"
        self.batch_root.mkdir()
        atomic_json(
            self.batch_root / "trajectories.json",
            {"rows": self.rows, "behavior_log_probs": self.behavior, "starting_update": trainer.actor_optimizer_step},
        )
        self.reference = self._probabilities(trainer)
        self._record(trainer, 0, self.reference)

    def _record(self, trainer, inner, current):
        result = {
            "update": trainer.actor_optimizer_step,
            "inner": inner,
            "batch": self.batch_count,
            "current_log_probs": current,
            "actual_behavior": prefix_table(current, self.behavior, [r["response_mask"] for r in self.rows]),
            "same_scorer_anchor": prefix_table(current, self.reference, [r["response_mask"] for r in self.rows]),
        }
        atomic_json(self.batch_root / f"inner-{inner:02d}.json", result)
        metrics = {"prefix_is/inner": inner, "prefix_is/batch": self.batch_count}
        for denominator in ("actual_behavior", "same_scorer_anchor"):
            for horizon, summary in result[denominator].items():
                for name in ("ess_fraction", "log_mean_weight", "max_normalized_weight"):
                    metrics[f"prefix_is/{denominator}/h{horizon}/{name}"] = summary[name]
        trainer.logger.log(data=metrics | trainer._metric_coordinates(), step=trainer.global_steps)
        print("PREFIX_IS_PROBE", self.batch_count, inner, trainer.actor_optimizer_step, flush=True)

    def on_inner_update(self, trainer, batch, inner):
        started = time.monotonic()
        if inner in INNER:
            self._record(trainer, inner, self._probabilities(trainer))
        if inner in (4, 8, 12, 16):
            self._banks(trainer, f"b{self.batch_count}-i{inner:02d}-u{trainer.actor_optimizer_step}")
        return time.monotonic() - started

    def finish_inner_batch(self):
        atomic_json(self.batch_root / "complete.json", {"inner_reads": list(INNER), "trajectories": len(self.rows)})
        self.rows = self.reference = None
