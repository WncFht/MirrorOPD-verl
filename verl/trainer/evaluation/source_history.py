# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-prefix RKL readouts at synchronous verl optimizer boundaries.

No model or optimizer is constructed here: generation and scoring use the
trainer's existing rollout, shared teacher, and actor worker groups.
"""

import time
import uuid
from pathlib import Path

import numpy as np
import transfer_queue as tq
from omegaconf import OmegaConf
from transformers import AutoConfig

from verl.experimental.rollout_policy.collector import UnsafeDrainError
from verl.trainer.evaluation.artifacts import HISTORY, RKL, TRAJECTORY, ArtifactStore, file_sha256, identity
from verl.trainer.evaluation.config import SourceHistoryConfig
from verl.trainer.evaluation.generation_contract import engine_contract, require_current_generation
from verl.trainer.evaluation.identity import model_identity
from verl.trainer.evaluation.metrics import readout_metrics
from verl.trainer.evaluation.native import balanced_history, capture_rows
from verl.trainer.evaluation.schedule import Schedule
from verl.trainer.evaluation.scoring import score_native_rows
from verl.trainer.ppo.readout_callback import ReadoutCallback
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.config import HFModelConfig


class SourceHistoryReadout(ReadoutCallback):
    """One fixed initial source bank and a bounded batch-balanced history."""

    handles_validation = True

    def __init__(self, *, config, options):
        self.config = config
        self.options = SourceHistoryConfig.parse(options)
        require_current_generation(config)
        input_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        generation_contract = engine_contract(config.actor_rollout_ref.rollout, input_model, input_model)
        teacher_path = config.distillation.teacher_models.teacher_model.model_path
        self.hidden_size = AutoConfig.from_pretrained(teacher_path, local_files_only=True).hidden_size
        self.score_budget = self.options.score
        self.root = Path(self.options.output)
        self.capacity = self.options.history_size
        self.seed = self.options.seed
        self.schedules = {
            "source_heldout": Schedule(at=(0,), every=self.options.source_every),
            "history": Schedule(every=self.options.history_every),
        }
        if config.trainer.v1.trainer_mode != "sync" or config.distillation.execution_mode != "shared_sync":
            raise ValueError("Fixed RKL readout requires native synchronous shared-teacher execution")
        if config.trainer.get("resume_mode", "disable") != "disable":
            raise ValueError("This acceptance readout currently requires a fresh run, not checkpoint resume")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Choose a new output directory for a fresh training run")
        if config.trainer.get("test_freq", -1) > 0:
            raise ValueError("Source/history owns evaluation schedules; disable trainer.test_freq")
        if not config.trainer.get("val_before_train", False):
            raise ValueError("Fixed RKL acceptance requires val_before_train=True")
        val_files = list(config.data.val_files)
        if val_files != [self.options.source_file]:
            raise ValueError("Native validation data must be the declared source held-out file")
        self.bank_path = Path(self.options.initial_bank or self.root / "initial-student-bank")
        self.bank = None
        self.bank_sha256 = None
        rollout = config.actor_rollout_ref.rollout
        if rollout.get("n", 1) != 1:
            raise ValueError("Fixed RKL history requires one training rollout per question")
        self.generation = OmegaConf.to_container(rollout.val_kwargs, resolve=True)
        model_path = Path(config.actor_rollout_ref.model.path)
        self.bank_metadata = dict(
            contract_version=2,
            engine_contract=generation_contract,
            initial_student_identity=model_identity(model_path),
            source_sha256=file_sha256(self.options.source_file),
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
                "teacher_model": teacher_path,
                "teacher_identity": model_identity(teacher_path),
                "student_identity": self.bank_metadata["initial_student_identity"],
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
        if len(self.bank) != self.options.source_count:
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
        if len(dataset) != self.options.source_count:
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
                trainer.rollout_service.park()

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

    def validate(self, trainer):
        outer = self.pending_outer
        if outer in self.scored:
            return {}
        source_due = self.schedules["source_heldout"].due(outer, trainer.total_training_steps)
        history_due = bool(self.history) and self.schedules["history"].due(outer, trainer.total_training_steps)
        # Manual validation still follows the declared task schedule.
        if not source_due and not history_due:
            return {}
        restore_rollout = not self.before_weight_sync
        if self.bank is None:
            if outer != 0 or not restore_rollout:
                raise ValueError("Cannot generate an initial bank from updated student weights")
            self._generate_initial_bank(trainer)
        elif restore_rollout:
            trainer.rollout_service.park()
        if self.bank_sha256 is None or file_sha256(self.bank_path / "manifest.json") != self.bank_sha256:
            raise ValueError("Bound initial bank manifest changed or was never completed")
        rows = []
        if source_due:
            rows.extend(self.bank)
        if history_due:
            rows.extend(self.store.resolve(self.history))
        before = trainer.actor_optimizer_step
        started = time.perf_counter()
        safe_to_restore = True
        try:
            scored, scoring = score_native_rows(trainer, rows, hidden_size=self.hidden_size, budget=self.score_budget)
        except UnsafeDrainError:
            safe_to_restore = False
            raise
        finally:
            if restore_rollout and safe_to_restore:
                trainer.rollout_service.ensure_current(outer)
        timing = scoring["teacher_timing"]
        score_seconds = scoring["seconds"]
        audit = scoring["audit"]
        # Publication follows successful scoring, state audit and resource cleanup.
        metrics = readout_metrics(scored, baseline=self.baseline, previous_history=self.previous_history)
        metrics.update(
            {
                "perf/timing/rkl_score_seconds": score_seconds,
                "perf/timing/rkl_readout_seconds": time.perf_counter() - started,
            }
        )
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
                "audit": audit,
            },
        )
        event.write_rows("rkl.parquet", ({key: row[key] for key in RKL.names} for row in scored), schema=RKL)
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
