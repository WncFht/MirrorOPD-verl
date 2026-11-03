# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent native forward acceptance after initializing from a portable HF export."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from verl.trainer.ppo.readout_callback import ReadoutCallback

from .artifacts import EVAL_RKL, ArtifactStore, file_sha256
from .config import ScoreBudget, strict_record
from .scoring import score_native_rows
from .trajectory import load_trajectories, trajectory_index


@dataclass(frozen=True)
class HFReloadOptions:
    export_audit: str
    output: str
    reference_event: str
    generation_bank: str | None = None
    reference_bank: str | None = None
    score: ScoreBudget = field(default_factory=ScoreBudget)

    def __post_init__(self):
        if any(
            not isinstance(value, str) or not value for value in (self.export_audit, self.output, self.reference_event)
        ):
            raise ValueError("HF reload requires nonempty artifact paths")
        if self.generation_bank is not None and (not isinstance(self.generation_bank, str) or not self.generation_bank):
            raise ValueError("generation_bank must be a nonempty path or null")

    @classmethod
    def parse(cls, values):
        values = dict(values)
        if "source_bank" in values:
            if "generation_bank" in values:
                raise ValueError("Specify generation_bank or legacy source_bank, not both")
            values["generation_bank"] = values.pop("source_bank")
        values["score"] = strict_record(ScoreBudget, values.get("score", {}))
        return strict_record(cls, values)


class HFReloadReadout(ReadoutCallback):
    """Use only with a new val_only trainer whose student path is the final HF export."""

    handles_validation = True
    uses_validation_dataset = False

    def __init__(self, *, config, options):
        self.options = HFReloadOptions.parse(options)
        if not config.trainer.val_only or config.trainer.resume_mode != "disable":
            raise ValueError("HF export acceptance requires a fresh validation-only trainer")
        export = ArtifactStore(self.options.export_audit).manifest(complete=True)
        self.reference_bank = (
            self.options.reference_bank or export["metadata"].get("reference_bank") or "source_heldout"
        )
        self.reference_ids = (
            export["metadata"].get("reference_ids")
            if self.reference_bank == export["metadata"].get("reference_bank")
            else None
        )
        self.generation_bank = self.options.generation_bank or export["metadata"].get("reference_generation_bank")
        if not self.generation_bank:
            raise ValueError("Legacy exports require an explicit generation_bank (or source_bank)")
        expected_bank = export["metadata"].get("reference_generation_sha256")
        if expected_bank and self.reference_bank == export["metadata"].get("reference_bank"):
            if file_sha256(ArtifactStore(self.generation_bank).manifest_path) != expected_bank:
                raise ValueError("Reload reference generation differs from the audited export")
        hf = Path(self.options.export_audit).parent / "huggingface"
        if Path(config.actor_rollout_ref.model.path).resolve() != hf.resolve():
            raise ValueError("HF reload trainer did not initialize from the audited final model")
        for name, digest in export["metadata"]["hf_files"].items():
            if file_sha256(hf / name) != digest:
                raise ValueError("Export contents changed after publication")
        self.store = ArtifactStore.create(
            self.options.output,
            kind="hf_reload_forward",
            metadata={
                "hf_model": str(hf.resolve()),
                "export_audit": self.options.export_audit,
                "reference_event": self.options.reference_event,
            },
        )

    def validate(self, trainer):
        from transformers import AutoConfig

        event = ArtifactStore(self.options.reference_event)
        event.manifest(complete=True)
        expected = {r["id"]: r for r in event.read_rows("rkl.parquet") if r["bank"] == self.reference_bank}
        if self.reference_ids is not None:
            if not set(self.reference_ids) <= set(expected):
                raise ValueError("Reload reference event is missing the audited verification trajectories")
            expected = {key: value for key, value in expected.items() if key in self.reference_ids}
        references = trajectory_index(ArtifactStore(self.generation_bank), bank=self.reference_bank)
        if self.reference_ids is not None or self.reference_bank != "source_heldout":
            # Public generation contains all accuracy samples; only the fixed RKL panel is scored.
            references = [r for r in references if r["id"] in expected]
        if set(expected) != {r["id"] for r in references}:
            raise ValueError("HF reload acceptance does not use the exact original fixed trajectories")
        hidden_size = AutoConfig.from_pretrained(
            trainer.config.distillation.teacher_models.teacher_model.model_path,
            trust_remote_code=trainer.config.actor_rollout_ref.model.get("trust_remote_code", False),
        ).hidden_size
        trainer.rollout_service.park()
        actual, timing = score_native_rows(
            trainer,
            references,
            hidden_size=hidden_size,
            budget=self.options.score,
            load_window=load_trajectories,
        )
        measures = {row.get("measure_id") for row in expected.values()}
        if None in measures or measures != {row.get("measure_id") for row in actual}:
            raise ValueError("HF reload requires matching explicit distribution and precision identities")
        errors = []
        for row in actual:
            before = expected.pop(row["id"])
            if row["tokens"] != before["tokens"] or row["tokens"] <= 0:
                raise ValueError("HF reload changed the scored response positions")
            left, right = before["rkl_sum"] / row["tokens"], row["rkl_sum"] / row["tokens"]
            if not math.isclose(left, right, rel_tol=1e-4, abs_tol=1e-6):
                raise RuntimeError(f"HF reload question RKL mismatch: {left} vs {right}")
            errors.append(abs(left - right))
        if expected or not errors:
            raise ValueError("Incomplete HF reload comparison")
        self.store.write_rows("rkl.parquet", actual, schema=EVAL_RKL)
        metrics = {"eval/checkpoint/hf_reload_verified": 1, "eval/checkpoint/hf_reload_max_rkl_error": max(errors)}
        self.store.finish(metadata={"metrics": metrics, "timing": timing, "rtol": 1e-4, "atol": 1e-6})
        return metrics
