# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-prefix RKL readouts at synchronous verl optimizer boundaries.

No model or optimizer is constructed here: generation and scoring use the
trainer's existing rollout, shared teacher, and actor worker groups.
"""

import hashlib
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.evaluation.artifacts import HISTORY, RKL, TRAJECTORY, ArtifactStore, file_sha256, identity
from verl.trainer.evaluation.metrics import readout_metrics
from verl.trainer.evaluation.schedule import Schedule
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn


def priority(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], separators=(",", ":")).encode()).hexdigest()


def balanced_history(batches, capacity, seed):
    """Deterministic random priorities; no loss-dependent selection."""
    if not batches:
        return []
    count = len(batches)
    if count > capacity:
        raise ValueError("History capacity must cover every observed batch")
    q, r = divmod(capacity, count)
    extras = set(sorted(batches, key=lambda b: priority(seed, "batch", b))[:r])
    result = []
    for batch_id, rows in sorted(batches.items()):
        quota = q + (batch_id in extras)
        if len(rows) < quota:
            raise ValueError("A training batch cannot fill its history quota")
        selected = sorted(rows, key=lambda x: priority(seed, "sample", batch_id, x["source_id"]))[:quota]
        result.extend(dict(row, history_batch=batch_id, history_quota=quota) for row in selected)
    return result


def capture_rows(batch, outer):
    fields = ["prompts", "responses", "response_mask", "extra_info"]
    # KVBatchMeta.fields=None means an unprojected batch. Native rollout rows
    # still contain metrics; it is not an empty list of available columns.
    available_fields = getattr(batch, "fields", None)
    if available_fields is None or "metrics" in available_fields:
        fields.append("metrics")
    data = tq.kv_batch_get(
        keys=list(batch.keys),
        partition_id=batch.partition_id,
        select_fields=fields,
    )
    info = list(data["extra_info"])
    observations = list(data["metrics"]) if "metrics" in data else [{} for _ in info]
    rows = []
    for i, (prompt, response, mask) in enumerate(
        zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
    ):
        tag = batch.tags[i]
        if tag.get("is_padding", False):
            continue
        source_id = str(info[i].get("index", identity(prompt.tolist())))
        rows.append(
            dict(
                id=f"{outer}:{source_id}",
                source_id=source_id,
                outer=outer,
                prompt_ids=prompt.tolist(),
                response_ids=response.tolist(),
                response_mask=mask.tolist(),
                rollout_key=batch.keys[i],
                min_global_steps=tag.get("min_global_steps"),
                max_global_steps=tag.get("max_global_steps"),
                data_source=info[i].get("data_source"),
                finish_reason=observations[i].get("finish_reason", tag.get("finish_reason")),
            )
        )
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate nonpadding source IDs in a rollout batch")
    return rows


def put_fixed_rows(rows, outer):
    partition = f"rkl-readout-{outer}-{uuid.uuid4().hex}"
    keys = [f"row-{i}" for i in range(len(rows))]
    values = {k: [] for k in ["prompts", "responses", "input_ids", "position_ids", "response_mask", "loss_mask"]}
    tags = []
    for row in rows:
        prompt, response = row["prompt_ids"], row["response_ids"]
        mask = row.get("response_mask", [1] * len(response))
        if not prompt or not response or len(mask) != len(response):
            raise ValueError("Empty or inconsistent fixed rollout")
        ids = prompt + response
        for k, v in dict(
            prompts=prompt,
            responses=response,
            input_ids=ids,
            position_ids=list(range(len(ids))),
            response_mask=mask,
            loss_mask=mask,
        ).items():
            values[k].append(torch.tensor(v, dtype=torch.int64))
        tags.append(dict(seq_len=len(ids), prompt_len=len(prompt), response_len=len(response), status="success"))
    fields = TensorDict(
        {k: torch.nested.nested_tensor(v, layout=torch.jagged) for k, v in values.items()}, batch_size=[len(rows)]
    )
    tq.kv_batch_put(keys=keys, partition_id=partition, tags=tags, fields=fields)
    return KVBatchMeta(keys=keys, partition_id=partition, tags=tags, fields=list(values)), dict(
        zip(keys, rows, strict=True)
    )


class FixedRKLReadout:
    """Opt-in callback: one fixed initial bank, bounded batch-balanced history."""

    def __init__(self, *, config):
        self.config = config
        self.options = OmegaConf.to_container(config.trainer.rkl_readout, resolve=True)
        self.root = Path(self.options["output"])
        self.capacity = int(self.options.get("history_size", 128))
        self.every = int(self.options.get("every", 1))
        self.seed = int(self.options.get("seed", 42))
        if self.capacity <= 0 or self.every <= 0:
            raise ValueError("Positive history capacity and readout frequency required")
        self.schedules = {
            "source_heldout": Schedule(at=(0,), every=self.options.get("source_every", self.every)),
            "history": Schedule(every=self.options.get("history_every", self.every)),
        }
        if config.trainer.v1.trainer_mode != "sync" or config.distillation.execution_mode != "shared_sync":
            raise ValueError("Fixed RKL readout requires native synchronous shared-teacher execution")
        if config.trainer.get("resume_mode", "disable") != "disable":
            raise ValueError("This acceptance readout currently requires a fresh run, not checkpoint resume")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Choose a new output directory for a fresh training run")
        if not config.trainer.get("val_before_train", False):
            raise ValueError("Fixed RKL acceptance requires val_before_train=True")
        val_files = list(config.data.val_files)
        if val_files != [self.options["source_file"]]:
            raise ValueError("Native validation data must be the declared source held-out file")
        self.bank_path = Path(self.options.get("initial_bank", self.root / "initial-student-bank"))
        self.bank = None
        self.bank_sha256 = None
        rollout = config.actor_rollout_ref.rollout
        if rollout.get("n", 1) != 1:
            raise ValueError("Fixed RKL history requires one training rollout per question")
        self.generation = OmegaConf.to_container(rollout.val_kwargs, resolve=True)
        model_path = Path(config.actor_rollout_ref.model.path)
        model_files = {
            str(p.relative_to(model_path)): file_sha256(p)
            for p in sorted(model_path.glob("*"))
            if p.is_file()
            and (
                p.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
                or p.suffix == ".jinja"
            )
        }
        self.bank_metadata = dict(
            initial_student=str(model_path),
            model_files=model_files,
            source_sha256=file_sha256(self.options["source_file"]),
            response_cap=config.data.max_response_length,
            generation=self.generation,
            seed=OmegaConf.select(config, "actor_rollout_ref.rollout.seed"),
            sampling_defaults={
                key: OmegaConf.select(config, f"actor_rollout_ref.rollout.{key}")
                for key in ("temperature", "top_p", "top_k", "do_sample")
            },
            input_config={
                key: OmegaConf.select(config, f"data.{key}")
                for key in (
                    "prompt_key",
                    "max_prompt_length",
                    "truncation",
                    "apply_chat_template_kwargs",
                    "chat_template",
                )
            },
        )
        # Resolve OmegaConf containers before hashing or publishing metadata.
        self.bank_metadata = OmegaConf.to_container(OmegaConf.create(self.bank_metadata), resolve=True)
        if self.bank_path.exists():
            if not self.bank_path.is_dir():
                raise ValueError(
                    "initial_bank must be a Parquet artifact directory; legacy JSON banks require migration"
                )
            bank_store = ArtifactStore(self.bank_path)
            saved = bank_store.manifest(complete=True)
            if saved["kind"] != "source_bank" or saved["metadata"] != self.bank_metadata:
                raise ValueError("Initial student bank model, input, source or generation protocol differs")
            self.bank = bank_store.read_rows("trajectories.parquet")
            self._check_bank()
            self.bank_sha256 = file_sha256(bank_store.manifest_path)
        elif self.bank_path.suffix == ".json":
            raise ValueError("New banks use an artifact directory, not a JSON filename")
        self.store = ArtifactStore.create(
            self.root,
            kind="fixed_rkl_run",
            metadata={
                "source_bank": str(self.bank_path.resolve()),
                "config_sha256": identity(OmegaConf.to_container(config, resolve=True, enum_to_str=True)),
                "teacher_model": OmegaConf.select(config, "distillation.teacher_models.teacher_model.model_path"),
                "history_capacity": self.capacity,
                "history_seed": self.seed,
                "score_temperature": 1.0,
                "support": "native_student_support",
                "aggregation": "question_mean_of_token_mean; history equal_batch",
            },
        )
        self.batches = {}
        self.history = []
        self.scored = set()
        self.pending_outer = 0
        self.before_weight_sync = False
        self.baseline = None
        self.previous_history = []

    def _check_bank(self):
        if len(self.bank) != int(self.options.get("source_count", 128)):
            raise ValueError("Initial student bank has a different sample count")
        if len({r["source_id"] for r in self.bank}) != len(self.bank):
            raise ValueError("Duplicate source bank questions")
        if any(r["bank"] != "source_heldout" or r["outer"] != 0 for r in self.bank):
            raise ValueError("Source bank must contain initial student trajectories")

    def on_readout(self, trainer, outer):
        if outer == trainer.total_training_steps and outer in self.scored:
            if self.store.manifest()["status"] == "active":
                self.store.finish(metadata={"completed_outer": outer, "optimizer_steps": trainer.actor_optimizer_step})
            return {}
        if outer != 0 or outer in self.scored:
            return {}
        self.pending_outer = outer
        return trainer._validate()

    def _generate_initial_bank(self, trainer):
        rollout = trainer.config.actor_rollout_ref.rollout
        cap = rollout.val_kwargs.get("max_tokens") or rollout.response_length
        if cap != trainer.config.data.max_response_length or rollout.val_kwargs.n != 1:
            raise ValueError("Source bank generation requires n=1 and the agreed training response cap")
        dataset = trainer.val_dataset
        if dataset is None:
            raise ValueError("RKL source generation requires the native validation dataset")
        if len(dataset) != int(self.options.get("source_count", 128)):
            raise ValueError("Source held-out dataset was filtered or has the wrong count")
        rows = collate_fn([dataset[i] for i in range(len(dataset))])
        rows["uid"] = np.array([uuid.uuid4().hex for _ in range(len(dataset))], dtype=object)
        batch = tu.get_tensordict(rows)
        tu.assign_non_tensor_data(batch, "validate", True)
        tu.assign_non_tensor_data(batch, "global_steps", trainer.global_steps)
        tq.kv_batch_put(
            keys=list(batch["uid"]),
            partition_id="val",
            tags=[dict(is_prompt=True, status="pending", global_steps=trainer.global_steps) for _ in range(len(batch))],
        )
        trainer._stamp_rollout_submission(batch, validate=True)
        trainer.agent_loop_manager.generate_sequences(batch)
        generated, _ = trainer.replay_buffer.sample(
            global_steps=trainer.global_steps, partition_id="val", batch_size=len(batch)
        )
        try:
            self.bank = capture_rows(generated, 0)
            if len(self.bank) != len(dataset):
                raise ValueError("Incomplete initial student held-out bank")
            for row in self.bank:
                row["bank"] = "source_heldout"
            self._check_bank()
            bank_store = ArtifactStore.create(self.bank_path, kind="source_bank", metadata=self.bank_metadata)
            bank_store.write_rows("trajectories.parquet", self.bank, schema=TRAJECTORY)
            bank_store.finish()
            self.bank_sha256 = file_sha256(bank_store.manifest_path)
        finally:
            try:
                tq.kv_clear(keys=list(generated.keys), partition_id=generated.partition_id)
            finally:
                trainer.checkpoint_manager.sleep_replicas()

    def on_before_weight_sync(self, trainer, batch, outer):
        rows = capture_rows(batch, outer)
        seen = {r["source_id"] for prior in self.batches.values() for r in prior}
        heldout = {r["source_id"] for r in self.bank}
        if seen.intersection(r["source_id"] for r in rows) or heldout.intersection(r["source_id"] for r in rows):
            raise ValueError("Repeated training source IDs or training/held-out overlap")
        shard = f"trajectories/outer-{outer:03d}.parquet"
        self.store.write_rows(shard, rows, schema=TRAJECTORY)
        self.batches[outer] = [
            {key: row[key] for key in ("id", "source_id", "outer")} | dict(shard=shard, row_index=index)
            for index, row in enumerate(rows)
        ]
        self.history = balanced_history(self.batches, self.capacity, self.seed)
        self.store.write_rows(f"history/outer-{outer:03d}.parquet", self.history, schema=HISTORY)
        self.pending_outer = outer
        if not any(schedule.due(outer, trainer.total_training_steps) for schedule in self.schedules.values()):
            return {}
        self.before_weight_sync = True
        try:
            return trainer._validate()
        finally:
            self.before_weight_sync = False

    def validate_fixed_rkl(self, trainer):
        outer = self.pending_outer
        if outer in self.scored:
            return {}
        source_due = self.schedules["source_heldout"].due(outer, trainer.total_training_steps)
        history_due = bool(self.history) and self.schedules["history"].due(outer, trainer.total_training_steps)
        # Legacy test_freq may invoke _validate even when this callback has no due task.
        if not source_due and not history_due:
            return {}
        restore_rollout = not self.before_weight_sync
        if self.bank is None:
            if outer != 0 or not restore_rollout:
                raise ValueError("Cannot generate an initial bank from updated student weights")
            self._generate_initial_bank(trainer)
        elif restore_rollout:
            trainer.checkpoint_manager.sleep_replicas()
        if self.bank_sha256 is None or file_sha256(self.bank_path / "manifest.json") != self.bank_sha256:
            raise ValueError("Bound initial bank manifest changed or was never completed")
        rows = []
        if source_due:
            rows.extend(self.bank)
        if history_due:
            rows.extend(self.store.resolve(self.history))
        meta, by_key = put_fixed_rows(rows, outer)
        timing = {}
        metrics = {}
        before = trainer.actor_optimizer_step
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        started = time.perf_counter()
        begun = False
        audit = None
        scored = []
        try:
            trainer.actor_rollout_wg.begin_rkl_readout(audit_state=bool(self.options.get("audit_state", False)))
            begun = True
            meta = trainer._balance_batch(meta, {})
            if len(meta.keys) != len(rows):
                raise ValueError("Readout count must be divisible by the active worker layout")
            trainer._compute_shared_teacher_log_probs(meta, timing)
            meta.fields = ["input_ids", "position_ids", "prompts", "responses", "loss_mask", "response_mask"]
            meta.extra_info = dict(
                temperature=1.0,
                compute_loss=False,
                calculate_entropy=False,
                full_vocab_score=True,
                retain_sample_identity=True,
                full_vocab_prefetch=True,
                global_token_num=None,
            )
            trainer.actor_rollout_wg.score_full_vocab(meta)
            data = tq.kv_batch_get(
                keys=list(meta.keys), partition_id=meta.partition_id, select_fields=["rkl_sum", "rkl_count"]
            )
            if len(data) != len(by_key):
                raise ValueError("Incomplete RKL score batch")
            for key, total, count in zip(meta.keys, data["rkl_sum"].unbind(), data["rkl_count"].unbind(), strict=True):
                row = by_key[key]
                if int(count.item()) != sum(row["response_mask"]):
                    raise ValueError("RKL count differs from the stored response mask")
                scored.append(
                    {k: row[k] for k in ["id", "source_id", "bank"]}
                    | dict(
                        history_batch=row.get("history_batch"),
                        rkl_sum=float(total.item()),
                        tokens=int(count.item()),
                    )
                )
            if trainer.actor_optimizer_step != before:
                raise RuntimeError("Read-only evaluation changed the optimizer counter")
            score_seconds = time.perf_counter() - started
        finally:
            try:
                if begun:
                    audit = trainer.actor_rollout_wg.end_rkl_readout()
            finally:
                try:
                    tq.kv_clear(keys=list(meta.keys), partition_id=meta.partition_id)
                finally:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                    if restore_rollout:
                        trainer.checkpoint_manager.update_weights(global_steps=outer)
        # Publication follows successful scoring, state audit and resource cleanup.
        metrics = readout_metrics(scored, baseline=self.baseline, previous_history=self.previous_history)
        metrics.update(
            {
                "perf/timing/rkl_score_seconds": score_seconds,
                "perf/timing/rkl_readout_seconds": time.perf_counter() - started,
            }
        )
        audit_requested = bool(self.options.get("audit_state", False))
        if audit_requested and (not audit or not all(x.get("unchanged") and x.get("state_digest") for x in audit)):
            raise RuntimeError("Requested state audit did not return verified digests")
        event = ArtifactStore.create(
            self.root / "scores" / f"outer-{outer:03d}",
            kind="rkl_event",
            metadata={
                "outer": outer,
                "optimizer_steps": before,
                "teacher_timing": timing,
                "source_bank": str(self.bank_path.resolve()),
                "source_bank_manifest_sha256": self.bank_sha256,
                "aggregation": "question_mean_of_token_mean; history equal_batch",
                "audit": {"status": "passed" if audit_requested else "not_requested", "ranks": audit},
            },
        )
        event.write_rows("rkl.parquet", scored, schema=RKL)
        invalid = sum(value for key, value in metrics.items() if key.endswith("/invalid_rows"))
        event.finish(
            metadata={
                "metrics": metrics,
                "coverage": {
                    "expected_rows": len(rows),
                    "observed_rows": len(scored),
                    "invalid_rows": invalid,
                    "scientific_complete": invalid == 0,
                },
            }
        )
        if self.baseline is None and any(r["bank"] == "source_heldout" for r in scored):
            self.baseline = [r for r in scored if r["bank"] == "source_heldout"]
        if historical := [r for r in scored if r["bank"] == "history"]:
            self.previous_history = historical
        self.scored.add(outer)
        return metrics
