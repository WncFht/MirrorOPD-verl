# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One native-verl boundary owner for generation, serial RKL and CPU accuracy."""

import asyncio
import dataclasses
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoConfig

from verl.trainer.evaluation.native import balanced_history, capture_rows
from verl.trainer.ppo.readout_callback import ReadoutCallback
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from .accuracy import AccuracyCollector, gap_recovery
from .artifacts import EVAL_RKL, GENERATION, HISTORY, ArtifactStore, file_sha256, identity
from .config import EvaluationConfig
from .coverage import rkl_coverage
from .execution import hidden_dtype_bytes
from .generation import NativeGenerator
from .generation_contract import engine_contract, require_current_generation
from .identity import active_bootstrap_contract, bootstrap_contract, model_identity
from .plan import compile_plan
from .reporting import evaluation_metrics
from .scoring import score_native_rows
from .trajectory import load_trajectories, select_panel, trajectory_index


def read_questions(spec):
    rows = pq.read_table(spec.path).to_pylist()
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate questions in {spec.name}")
    for row in rows:
        if isinstance(row["prompt"], str):
            row["prompt"] = json.loads(row["prompt"])
        if not isinstance(row["id"], str) or not row["prompt"]:
            raise ValueError("Prepared questions require string IDs and explicit messages")
    if not set(spec.panel_ids) <= {r["id"] for r in rows}:
        raise ValueError("Panel selection is not a subset of the full benchmark")
    return rows


class EvaluationCoordinator(ReadoutCallback):
    handles_validation = True
    uses_validation_dataset = False
    requires_full_batch = True
    owns_final_weight_sync = True

    def __init__(self, *, config, options):
        self.config = config
        require_current_generation(config, independent_engine=True)
        self.options = EvaluationConfig.parse(options)
        if config.trainer.v1.trainer_mode != "sync" or config.distillation.execution_mode != "shared_sync":
            raise ValueError("Unified evaluation requires synchronous serial shared-teacher training")
        if config.trainer.test_freq > 0 or config.trainer.get("resume_mode", "disable") != "disable":
            raise ValueError("Unified evaluation owns schedules and requires a fresh run")
        if not config.trainer.val_before_train or config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("Initial validation and one training trajectory per question are required")
        if config.trainer.save_freq > 0:
            raise ValueError("Unified evaluation owns the single final export; disable periodic saves")
        if self.options.final_export:
            checkpoint = config.actor_rollout_ref.actor.checkpoint
            if set(checkpoint.save_contents) != {"model", "hf_model"} or set(checkpoint.load_contents) != {"model"}:
                raise ValueError("Final export requires model+hf_model saves and model-only reloads")
            if not checkpoint.get("preserve_hf_master_weights", False):
                raise ValueError("Final export must preserve fp32 master weights")
        self.input_model = omega_conf_to_dataclass(config.actor_rollout_ref.model, dataclass_type=HFModelConfig)
        teacher_path = config.distillation.teacher_models.teacher_model.model_path
        self.hidden_size = AutoConfig.from_pretrained(
            teacher_path, trust_remote_code=config.actor_rollout_ref.model.get("trust_remote_code", False)
        ).hidden_size
        self.student_identity = model_identity(config.actor_rollout_ref.model.path)
        self.teacher_identity = model_identity(teacher_path)
        self.specifications = [(d, read_questions(d)) for d in self.options.datasets]
        self.bootstrap = ArtifactStore(self.options.bootstrap)
        frozen = self.bootstrap.manifest(complete=True)
        expected_generation = engine_contract(config.actor_rollout_ref.rollout, self.input_model, self.input_model)
        for name in ("initial", "source") if self.options.source_enabled else ("initial",):
            recorded = ArtifactStore(self.bootstrap.root / name).manifest(complete=True)["metadata"]
            if recorded.get("engine_contract_id") != identity(expected_generation):
                raise ValueError("Initial bank and current generation engine have different effective protocols")
        for name in self.options.bootstrap_children:
            child = ArtifactStore(self.bootstrap.root / name)
            if frozen["metadata"].get("children_sha256", {}).get(name) != file_sha256(child.manifest_path):
                raise ValueError("Bootstrap child manifest changed or was not bound")
        contract = bootstrap_contract(self.options, student=self.student_identity, teacher=self.teacher_identity)
        stored_contract = frozen["metadata"]["contract"]
        if frozen["metadata"].get("contract_sha256") != identity(stored_contract):
            raise ValueError("Bootstrap contract digest differs")
        if identity(active_bootstrap_contract(stored_contract, source_enabled=self.options.source_enabled)) != identity(
            contract
        ):
            raise ValueError("Bootstrap models, question sets or generation protocols differ")
        self.fixed = {}
        banks = {"public_initial": "initial", "public_teacher": "teacher"}
        if self.options.source_enabled:
            banks["source_heldout"] = "source"
        for bank, child in banks.items():
            refs = trajectory_index(ArtifactStore(self.bootstrap.root / child), bank=bank)
            self.fixed[bank] = refs if bank == "source_heldout" else select_panel(refs, self.options.datasets)
        if self.options.source_enabled and len(self.fixed.get("source_heldout", [])) != self.options.source_count:
            raise ValueError("Source bank does not cover the declared source holdout")
        self.store = ArtifactStore.create(
            self.options.output,
            kind="unified_evaluation_run",
            metadata={
                "bootstrap": str(self.bootstrap.root),
                "bootstrap_sha256": file_sha256(self.bootstrap.manifest_path),
                "contract_sha256": identity(contract),
                "configuration": dataclasses.asdict(self.options),
                "training_rollout_policy": OmegaConf.to_container(config.rollout_policy, resolve=True)
                if OmegaConf.select(config, "rollout_policy") is not None
                else None,
            },
        )
        self.batches, self.baselines = {}, {}
        self.previous_history, self.history = [], []
        self.completed, self.pending_outer = set(), 0
        self.pending_events = {}
        self.before_sync = False
        self.initial_accuracy = ArtifactStore(self.bootstrap.root / "initial-accuracy").manifest(complete=True)[
            "metadata"
        ]["metrics"]
        self.teacher_accuracy = ArtifactStore(self.bootstrap.root / "teacher-accuracy").manifest(complete=True)[
            "metadata"
        ]["metrics"]

    def on_readout(self, trainer, outer):
        if outer in self.pending_events:
            event, metadata = self.pending_events[outer]
            event.finish(metadata=metadata)
            del self.pending_events[outer]
            self.completed.add(outer)
        if outer == 0 and outer not in self.completed:
            return self.validate(trainer)
        if (
            outer == trainer.total_training_steps
            and outer in self.completed
            and self.store.manifest()["status"] == "active"
        ):
            self.store.finish(metadata={"completed_outer": outer, "optimizer_steps": trainer.actor_optimizer_step})
        return {}

    def on_before_weight_sync(self, trainer, batch, outer):
        rows = capture_rows(batch, outer)
        known = {r["source_id"] for prior in self.batches.values() for r in prior}
        known.update(r["source_id"] for r in self.fixed.get("source_heldout", []))
        if known.intersection(r["source_id"] for r in rows):
            raise ValueError("Training repeated a question or overlaps source heldout")
        shard = f"trajectories/outer-{outer:03d}.parquet"
        for row in rows:
            row.update(
                dataset="training",
                domain=self.options.training_domain,
                sample_index=0,
                producer="current_student"
                if getattr(getattr(trainer, "rollout_policy", None), "default_path", True)
                else f"policy_{trainer.rollout_policy.mode}",
                bank="history",
                prompt_tokens=len(row["prompt_ids"]),
                response_tokens=len(row["response_ids"]),
                valid_tokens=sum(row["response_mask"]),
            )
        self.store.write_rows(shard, rows, schema=GENERATION)
        self.batches[outer] = [
            {k: row[k] for k in ("id", "source_id", "outer", "valid_tokens", "dataset", "domain", "sample_index")}
            | {
                "artifact": str(self.store.root),
                "shard": shard,
                "row_index": i,
                "trajectory_id": row["id"],
                "bank": "history",
            }
            for i, row in enumerate(rows)
        ]
        if self.options.history_enabled:
            self.history = balanced_history(self.batches, self.options.history_size, self.options.seed)
            self.store.write_rows(
                f"history/outer-{outer:03d}.parquet",
                ({k: r.get(k) for k in HISTORY.names} for r in self.history),
                schema=HISTORY,
            )
        self.pending_outer, self.before_sync = outer, True
        try:
            # Training hidden states remain live until the native outer cleanup.
            live_bytes = sum(sum(r["response_mask"]) for r in rows) * (
                self.hidden_size * hidden_dtype_bytes(self.config) + 8
            )
            return self.validate(trainer, live_payload_bytes=live_bytes)
        finally:
            self.before_sync = False

    def on_update_records(self, events, role):
        schema = pa.schema([("step", pa.int64()), ("metrics", pa.map_(pa.string(), pa.float64()))])
        self.store.write_rows(
            f"updates/{role}-outer-{self.pending_outer + 1:03d}.parquet",
            [{"step": step, "metrics": [(k, float(v)) for k, v in values.items()]} for values, step in events],
            schema=schema,
        )

    def _progress(self, value):
        # Structured stdout is picked up by the existing run logger; no second service.
        print("EVALUATION_PROGRESS " + json.dumps(value, sort_keys=True), flush=True)

    def checkpoint_references(self):
        return self.fixed.get("source_heldout") or self.fixed.get("public_initial", [])

    def validate(self, trainer, *, live_payload_bytes=0):
        outer = self.pending_outer
        if outer in self.completed:
            return {}
        started = time.monotonic()
        plan = compile_plan(
            self.options, outer=outer, total=trainer.total_training_steps, optimizer_steps=trainer.actor_optimizer_step
        )
        if not plan.tasks and not (outer == trainer.total_training_steps and self.options.final_export):
            self.completed.add(outer)
            return {}
        service = trainer.rollout_service
        event = ArtifactStore.create(
            self.store.root / f"events/outer-{outer:03d}",
            kind="evaluation_event",
            metadata={
                "plan": dataclasses.asdict(plan),
                "outer": outer,
                "optimizer_steps": trainer.actor_optimizer_step,
            },
        )
        refs, metrics, aliases = [], {}, {}
        for task in plan.tasks:
            if task in self.fixed:
                refs.extend(self.fixed[task])
            elif task == "history":
                refs.extend(self.history)
        collector = None
        try:
            if plan.generate_current:
                service.ensure_current(outer)
                generator = NativeGenerator(
                    self.config,
                    [r.server_handle for r in trainer.checkpoint_manager.replicas],
                    self.input_model,
                    engine_config=self.config.actor_rollout_ref.rollout,
                    engine_model=self.input_model,
                    max_model_len=self.config.actor_rollout_ref.rollout.max_model_len,
                    concurrency=self.options.generation_concurrency,
                    chunk_size=self.options.generation_chunk_size,
                    timeout=self.options.request_timeout,
                )
                collector = AccuracyCollector(
                    event.root / "accuracy",
                    self.specifications,
                    self.input_model.tokenizer,
                    workers=self.options.verifier_workers,
                )
                collector.follow_generation(event.root / "current", self.options.protocols)
                current = asyncio.run(
                    generator.generate(
                        event.root / "current",
                        self.specifications,
                        self.options.protocols,
                        producer="current_student",
                        model_identity=self.student_identity,
                        outer=outer,
                        expected_version=outer,
                        on_batch=collector.check,
                        progress=self._progress,
                    )
                )
                refs.extend(select_panel(trajectory_index(current, bank="public_current"), self.options.datasets))
            elif outer == 0 and "public_current" in plan.tasks:
                if "public_initial" not in plan.tasks:
                    refs.extend(self.fixed["public_initial"])
                aliases["public_current"] = "public_initial"
            if outer == 0:
                metrics.update(self.initial_accuracy)
                metrics.update(gap_recovery(self.initial_accuracy, self.initial_accuracy, self.teacher_accuracy))
                for name, label in (("initial-accuracy", "initial_student"), ("teacher-accuracy", "teacher")):
                    accuracy = ArtifactStore(self.bootstrap.root / name).manifest(complete=True)["metadata"]["metrics"]
                    metrics.update(
                        {k.replace("eval/accuracy/", f"eval/accuracy/{label}/"): v for k, v in accuracy.items()}
                    )
            service.park()
            self._progress({"phase": "rkl", "outer": outer, "trajectories": len(refs), "tasks": plan.tasks})
            scores, timing = score_native_rows(
                trainer,
                refs,
                hidden_size=self.hidden_size,
                budget=self.options.score,
                live_payload_bytes=live_payload_bytes,
                load_window=load_trajectories,
            )
            coverage = rkl_coverage(refs, scores)
            if outer == 0:
                for bank in self.fixed:
                    self.baselines[bank] = [r for r in scores if r["bank"] == bank]
            metrics.update(
                evaluation_metrics(
                    scores, baselines=self.baselines, previous_history=self.previous_history, aliases=aliases
                )
            )
            if collector is not None:
                accuracy = collector.finish(current)
                metrics.update(accuracy)
                metrics.update(gap_recovery(accuracy, self.initial_accuracy, self.teacher_accuracy))
                collector = None
            if outer == trainer.total_training_steps and self.options.final_export:
                from .checkpoint import final_model_export

                metrics.update(
                    final_model_export(
                        trainer,
                        self.checkpoint_references(),
                        hidden_size=self.hidden_size,
                        budget=self.options.score,
                        destination=self.store.root / "final-model",
                        verify=self.options.verify_export,
                        live_payload_bytes=live_payload_bytes,
                    )
                )
            event.write_rows("rkl.parquet", scores, schema=EVAL_RKL)
            metrics["perf/timing/evaluation_seconds"] = time.monotonic() - started
            if not self.before_sync:
                service.ensure_current(outer)
            event_metadata = {"metrics": metrics, "timing": timing, "aliases": aliases, "coverage": coverage}
            if self.before_sync:
                self.pending_events[outer] = event, event_metadata
            else:
                event.finish(metadata=event_metadata)
                self.completed.add(outer)
            if any(r["bank"] == "history" for r in scores):
                self.previous_history = [r for r in scores if r["bank"] == "history"]
            # Before-update hook leaves restoration to the normal native on_step_end.
            return metrics
        finally:
            if collector is not None:
                collector.close()
