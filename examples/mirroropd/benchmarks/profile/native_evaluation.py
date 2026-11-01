# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validation-only long-prefix/window probes through native verl workers.

Inputs may be explicitly synthetic shape probes. These results are infrastructure
checks and must never be merged into a scientific teacher-transfer ranking.
"""

import dataclasses
import math
from dataclasses import dataclass

from transformers import AutoConfig

from verl.trainer.evaluation.artifacts import EVAL_RKL, ArtifactStore, file_sha256
from verl.trainer.evaluation.config import ScoreBudget, strict_record
from verl.trainer.evaluation.scoring import score_native_rows
from verl.trainer.evaluation.trajectory import load_trajectories, trajectory_index
from verl.trainer.ppo.readout_callback import ReadoutCallback


@dataclass(frozen=True)
class ProbeOptions:
    bank: str
    output: str
    cases: tuple[ScoreBudget, ...]

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.bank, self.output)):
            raise ValueError("Probe bank and output must be nonempty paths")

    @classmethod
    def parse(cls, value):
        value = dict(value)
        value["cases"] = tuple(strict_record(ScoreBudget, case) for case in value["cases"])
        result = strict_record(cls, value)
        if not result.cases:
            raise ValueError("A native RKL probe requires at least one scoring case")
        return result


class NativeRKLProbe(ReadoutCallback):
    handles_validation = True
    uses_validation_dataset = False

    def __init__(self, *, config, options):
        if not config.trainer.val_only or config.trainer.resume_mode != "disable":
            raise ValueError("RKL infrastructure probes require a fresh validation-only run")
        self.options = ProbeOptions.parse(options)
        self.teacher_copy = (
            config.actor_rollout_ref.model.path == config.distillation.teacher_models.teacher_model.model_path
        )
        self.bank = ArtifactStore(self.options.bank)
        self.bank.manifest(complete=True)
        self.store = ArtifactStore.create(
            self.options.output,
            kind="native_rkl_probe",
            metadata={
                "bank": str(self.bank.root),
                "bank_sha256": file_sha256(self.bank.manifest_path),
                "purpose": "infrastructure_only",
                "student": config.actor_rollout_ref.model.path,
                "teacher": config.distillation.teacher_models.teacher_model.model_path,
            },
        )

    def validate(self, trainer):
        trainer.rollout_service.park()
        hidden = AutoConfig.from_pretrained(
            trainer.config.distillation.teacher_models.teacher_model.model_path
        ).hidden_size
        references = trajectory_index(self.bank, bank="infrastructure_probe")
        baseline, reports = None, []
        for index, case in enumerate(self.options.cases):
            result, timing = score_native_rows(
                trainer,
                references,
                hidden_size=hidden,
                budget=case,
                load_window=load_trajectories,
            )
            values = {r["id"]: (r["rkl_sum"], r["tokens"]) for r in result}
            if len(values) != len(references) or any(
                n <= 0 or not math.isfinite(total) for total, n in values.values()
            ):
                raise ValueError("Probe produced incomplete or invalid RKL rows")
            if self.teacher_copy and any(abs(total / n) > 1e-6 for total, n in values.values()):
                raise RuntimeError("Identical teacher/student weights exceeded the numerical RKL tolerance")
            error = 0.0
            if baseline is not None:
                if values.keys() != baseline.keys():
                    raise ValueError("Probe cases scored different trajectories")
                for key, (total, count) in values.items():
                    old, old_count = baseline[key]
                    if count != old_count or not math.isclose(total / count, old / count, rel_tol=1e-4, abs_tol=1e-6):
                        raise RuntimeError("Window/token-chunk probe changed question RKL")
                    error = max(error, abs(total - old) / count)
            else:
                baseline = values
            self.store.write_rows(f"case-{index:02d}.parquet", result, schema=EVAL_RKL)
            reports.append(
                {
                    "case": dataclasses.asdict(case),
                    "timing": timing,
                    "max_question_rkl_error": error,
                    "max_question_rkl": max(total / n for total, n in values.values()),
                }
            )
        self.store.finish(
            metadata={
                "cases": reports,
                "rows": len(references),
                "tokens": sum(r["valid_tokens"] for r in references),
                "teacher_copy": self.teacher_copy,
            }
        )
        return {
            "eval/probe/cases": len(reports),
            "eval/probe/rows": len(references),
            "eval/probe/max_question_rkl_error": max(r["max_question_rkl_error"] for r in reports),
        }
