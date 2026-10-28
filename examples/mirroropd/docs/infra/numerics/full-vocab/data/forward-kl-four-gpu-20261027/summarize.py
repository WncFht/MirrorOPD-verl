"""Audit the two-round, four-GPU E49 acceptance run from its file metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

from omegaconf import OmegaConf


def summarize(run, gpu_csv):
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.trainer.total_training_steps == 2
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 16
    assert cfg.distillation.distillation_loss.loss_mode == "full_forward_kl"
    assert cfg.distillation.distillation_loss.full_vocab.forward_teacher_support == "student"
    rows = [json.loads(line)["data"] for line in (run / "history.jsonl").read_text().splitlines()]
    updates = [row for row in rows if "optim/actor/update_applied" in row]
    outer = [row for row in rows if "perf/timing/outer_step_seconds" in row]
    assert len(updates) == 32 and len(outer) == 2
    reconstruction = {"full_forward_kl": 0.0, "full_kl": 0.0}
    for index, row in enumerate(updates, start=1):
        assert row["step/update"] == row["step/optimizer"] == index
        assert row["step/outer"] == (index - 1) // 16 + 1
        assert row["optim/actor/update_applied"] == 1
        assert row["optim/actor/update_skipped"] == 0
        assert row["optim/actor/loss_nonfinite"] == row["optim/actor/grad_norm_preclip_nonfinite"] == 0
        for key in ("distill/full_forward_kl", "distill/full_kl", "optim/actor/grad_norm_preclip"):
            assert math.isfinite(row[key]) and row[key] >= 0, (index, key, row[key])
        assert abs(row["optim/actor/loss"] - row["distill/full_forward_kl"]) < 1e-6
        assert 0 <= row["distill/js"] <= math.log(2) + 1e-6
        assert 0 <= row["distill/tv"] <= 1
        assert 0 <= row["distill/teacher_outside_student_mass"] <= 1
        for metric in reconstruction:
            prefix = f"distill/{metric}_position/"
            total = sum(v for k, v in row.items() if k.startswith(prefix) and k.endswith("/sum"))
            count = sum(v for k, v in row.items() if k.startswith(prefix) and k.endswith("/token_count"))
            assert count > 0
            error = abs(total / count - row[f"distill/{metric}"])
            assert error < 1e-6, (metric, error)
            reconstruction[metric] = max(reconstruction[metric], error)
    assert [row["step/update"] for row in outer] == [16, 32]
    evaluations = [row for row in rows if "eval/math500/response_count" in row]
    validation_enabled = cfg.trainer.test_freq > 0
    assert len(evaluations) == int(validation_enabled)
    if validation_enabled:
        evaluation = evaluations[0]
        assert evaluation["step/update"] == 32
        for domain in ("math500", "amc23", "aime25"):
            assert evaluation[f"eval/{domain}/response_count"] == 2
            assert evaluation[f"eval/{domain}/failed_count"] == 0
            assert evaluation[f"eval/{domain}/nonfinite_score_count"] == 0
    gpu_peaks = {}
    with gpu_csv.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            gpu = row["index"].strip()
            used = float(row["memory.used [MiB]"].split()[0])
            gpu_peaks[gpu] = max(gpu_peaks.get(gpu, 0), used)
    assert set(gpu_peaks) == {"0", "1", "2", "3"}
    metrics = ("full_forward_kl", "full_kl", "js", "tv", "student_entropy", "teacher_entropy")
    rounds = []
    for index in range(2):
        first, last = updates[index * 16], updates[index * 16 + 15]
        rounds.append(
            {
                "outer": index + 1,
                "first": {key: first[f"distill/{key}"] for key in metrics},
                "last": {key: last[f"distill/{key}"] for key in metrics},
                "timing_seconds": {key: value for key, value in outer[index].items() if key.startswith("perf/timing/")},
                "actor_peak_gib": outer[index].get("perf/memory/fsdp_allocated_peak_gib"),
            }
        )
    return {
        "status": "pass",
        "physical_gpus": [0, 1, 2, 3],
        "successful_updates": 32,
        "source_commit": (run / "source_commit.txt").read_text().strip(),
        "skipped_updates": 0,
        "validation_enabled": validation_enabled,
        "validation_responses": 6 if validation_enabled else 0,
        "validation_failed": 0 if validation_enabled else None,
        "train_response_cap": cfg.data.max_response_length,
        "validation_response_cap": cfg.actor_rollout_ref.rollout.val_kwargs.max_tokens,
        "optimizer_overrides": OmegaConf.to_container(cfg.actor_rollout_ref.actor.optim)["override_optimizer_config"],
        "grad_norm_preclip_range": [
            min(row["optim/actor/grad_norm_preclip"] for row in updates),
            max(row["optim/actor/grad_norm_preclip"] for row in updates),
        ],
        "teacher_outside_student_mass_range": [
            min(row["distill/teacher_outside_student_mass"] for row in updates),
            max(row["distill/teacher_outside_student_mass"] for row in updates),
        ],
        "position_reconstruction_max_abs_error": reconstruction,
        "gpu_sampled_peak_mib": gpu_peaks,
        "rounds": rounds,
        "scope": (
            "Two rollout rounds on the same 64 prompts; engineering acceptance, "
            "not an accuracy or long-context comparison."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.gpu_csv)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
