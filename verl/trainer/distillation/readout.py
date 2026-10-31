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

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


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
    data = tq.kv_batch_get(
        keys=list(batch.keys),
        partition_id=batch.partition_id,
        select_fields=["prompts", "responses", "response_mask", "extra_info"],
    )
    info = list(data["extra_info"])
    rows = []
    for i, (prompt, response, mask) in enumerate(
        zip(data["prompts"].unbind(), data["responses"].unbind(), data["response_mask"].unbind(), strict=True)
    ):
        tag = batch.tags[i]
        if tag.get("is_padding", False):
            continue
        source_id = str(info[i].get("index", batch.keys[i]))
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
        if config.trainer.v1.trainer_mode != "sync" or config.distillation.execution_mode != "shared_sync":
            raise ValueError("Fixed RKL readout requires native synchronous shared-teacher execution")
        if config.trainer.get("resume_mode", "disable") != "disable":
            raise ValueError("This acceptance readout currently requires a fresh run, not checkpoint resume")
        if any((self.root / "trajectories").glob("outer-*.json")):
            raise FileExistsError("Choose a new output directory for a fresh training run")
        if not config.trainer.get("val_before_train", False):
            raise ValueError("Fixed RKL acceptance requires val_before_train=True")
        val_files = list(config.data.val_files)
        if val_files != [self.options["source_file"]]:
            raise ValueError("Native validation data must be the declared source held-out file")
        self.bank_path = Path(self.options.get("initial_bank", self.root / "initial-student-bank.json"))
        self.bank = None
        rollout = config.actor_rollout_ref.rollout
        self.generation = OmegaConf.to_container(rollout.val_kwargs, resolve=True)
        if self.bank_path.exists():
            saved = json.loads(self.bank_path.read_text())
            if saved.get("initial_student") != config.actor_rollout_ref.model.path:
                raise ValueError("Initial student bank belongs to a different model")
            expected_hash = hashlib.sha256(Path(self.options["source_file"]).read_bytes()).hexdigest()
            if saved.get("source_sha256") != expected_hash:
                raise ValueError("Initial student bank belongs to a different held-out split")
            if saved.get("response_cap") != config.data.max_response_length:
                raise ValueError("Initial student bank has a different response cap")
            if saved.get("generation") != self.generation:
                raise ValueError("Initial student bank uses different generation settings")
            self.bank = saved["rows"]
            if len(self.bank) != int(self.options.get("source_count", 128)):
                raise ValueError("Initial student bank has a different sample count")
        self.batches = {}
        self.history = []
        self.scored = set()
        self.pending_outer = 0
        self.before_weight_sync = False

    def on_readout(self, trainer, outer):
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
        self.bank = capture_rows(generated, 0)
        if len(self.bank) != len(dataset):
            raise ValueError("Incomplete initial student held-out bank")
        for row in self.bank:
            row["bank"] = "source_heldout"
        write_json(
            self.bank_path,
            dict(
                rows=self.bank,
                initial_student=trainer.config.actor_rollout_ref.model.path,
                source_file=self.options["source_file"],
                source_sha256=hashlib.sha256(Path(self.options["source_file"]).read_bytes()).hexdigest(),
                response_cap=trainer.config.data.max_response_length,
                generation=self.generation,
            ),
        )
        tq.kv_clear(keys=list(generated.keys), partition_id=generated.partition_id)
        trainer.checkpoint_manager.sleep_replicas()

    def on_before_weight_sync(self, trainer, batch, outer):
        rows = capture_rows(batch, outer)
        seen = {r["source_id"] for prior in self.batches.values() for r in prior}
        heldout = {r["source_id"] for r in self.bank}
        if seen.intersection(r["source_id"] for r in rows) or heldout.intersection(r["source_id"] for r in rows):
            raise ValueError("Repeated training source IDs or training/held-out overlap")
        self.batches[outer] = rows
        write_json(self.root / "trajectories" / f"outer-{outer:03d}.json", rows)
        self.history = balanced_history(self.batches, self.capacity, self.seed)
        for row in self.history:
            row["bank"] = "history"
        write_json(self.root / "history" / f"outer-{outer:03d}.json", self.history)
        self.pending_outer = outer
        if outer % self.every and outer != trainer.total_training_steps:
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
        restore_rollout = not self.before_weight_sync
        if self.bank is None:
            if outer != 0 or not restore_rollout:
                raise ValueError("Cannot generate an initial bank from updated student weights")
            self._generate_initial_bank(trainer)
        elif restore_rollout:
            trainer.checkpoint_manager.sleep_replicas()
        rows = self.bank + self.history
        meta, by_key = put_fixed_rows(rows, outer)
        timing = {}
        metrics = {}
        before = trainer.actor_optimizer_step
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        started = time.perf_counter()
        begun = False
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
            scored = []
            for key, total, count in zip(meta.keys, data["rkl_sum"].unbind(), data["rkl_count"].unbind(), strict=True):
                row = by_key[key]
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
            elapsed = time.perf_counter() - started
            write_json(
                self.root / "scores" / f"outer-{outer:03d}.json",
                dict(
                    outer=outer,
                    optimizer_steps=before,
                    seconds=elapsed,
                    teacher_timing=timing,
                    initial_bank_sha256=hashlib.sha256(self.bank_path.read_bytes()).hexdigest(),
                    rows=scored,
                    aggregation="token-mean for engineering acceptance only; final scientific aggregation undecided",
                ),
            )
            metrics = {"perf/timing/rkl_score_seconds": elapsed, "eval/rkl/rows": len(scored)}
            held = [r for r in scored if r["bank"] == "source_heldout"]
            metrics["eval/rkl/source_heldout/token_mean"] = sum(r["rkl_sum"] for r in held) / sum(
                r["tokens"] for r in held
            )
            historical = [r for r in scored if r["bank"] == "history"]
            if historical:
                groups = {r["history_batch"] for r in historical}
                means = []
                for group in groups:
                    selected = [r for r in historical if r["history_batch"] == group]
                    means.append(sum(r["rkl_sum"] for r in selected) / sum(r["tokens"] for r in selected))
                metrics["eval/rkl/history/equal_batch_token_mean"] = sum(means) / len(means)
            self.scored.add(outer)
            return metrics
        finally:
            try:
                if begun:
                    audit = trainer.actor_rollout_wg.end_rkl_readout()
                    write_json(self.root / "audit" / f"outer-{outer:03d}.json", audit)
            finally:
                try:
                    tq.kv_clear(keys=list(meta.keys), partition_id=meta.partition_id)
                finally:
                    random.setstate(rng[0])
                    np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2])
                    if restore_rollout:
                        trainer.checkpoint_manager.update_weights(global_steps=outer)
                    metrics["perf/timing/rkl_readout_seconds"] = time.perf_counter() - started
