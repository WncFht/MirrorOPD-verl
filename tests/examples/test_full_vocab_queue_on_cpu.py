# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ablation_queue", ROOT / "examples/mirroropd/queue_full_vocab_ablations.py"
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def plan():
    return {"name": "test-queue", "python": sys.executable, "model_root": "/models", "data_root": "/data"}


def test_job_budget_and_seed_pairs():
    jobs = queue.make_jobs()
    assert len(jobs) == 13
    quality = [j for j in jobs if j["kind"] == "quality"]
    assert len(quality) == 5 and all(j["steps"] == 100 for j in quality)
    assert {(j["method"], j["seed"]) for j in quality} == {
        ("vanilla", 42),
        ("full", 42),
        ("token_is", 42),
        ("full", 43),
        ("vanilla", 43),
    }
    engineering = [j for j in jobs if j["kind"] == "engineering"]
    assert {(j["async_write"], j["prefetch"]) for j in engineering} == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }


@pytest.mark.parametrize("method", ["vanilla", "full", "token_is"])
def test_exact_launch_composes_and_uses_matched_budget(tmp_path, method):
    job = next(j for j in queue.make_jobs() if j["kind"] == "quality" and j["method"] == method and j["seed"] == 42)
    command, public_env, _ = queue.public_launch(plan(), job, ROOT, tmp_path)
    assert "WANDB_API_KEY" not in public_env and "NETRC" not in public_env
    env = {**os.environ, **public_env, "DRY_RUN": "1"}
    output = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    argv = shlex.split(output.stdout)
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=argv[argv.index("verl.trainer.main_ppo") + 1 :])
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert cfg.trainer.total_training_steps == 100 and cfg.trainer.test_freq == 20
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert cfg.actor_rollout_ref.rollout.seed == cfg.data.seed == 42
    assert cfg.data.max_response_length == 7168 and cfg.actor_rollout_ref.rollout.val_kwargs.n == 16
    assert cfg.distillation.distillation_loss.loss_mode == ("full_reverse_kl" if method == "full" else "k1")
    assert cfg.distillation.distillation_loss.policy_loss_mode == ("token_is" if method == "token_is" else "vanilla")


@pytest.mark.parametrize("busy", [False, True])
def test_gpu_gate_rejects_even_idle_compute_process(monkeypatch, busy):
    gpu = "\n".join(f"{i}, GPU-{i}, 1, 0" for i in range(4))
    values = iter([SimpleNamespace(stdout=gpu), SimpleNamespace(stdout="GPU-0, 123\n" if busy else "GPU-7, 456\n")])
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **kw: next(values))
    idle, _ = queue.gpu_snapshot()
    assert idle is not busy


def test_nested_file_logger_and_validation_coverage(tmp_path):
    job = {"kind": "quality", "steps": 100}
    rows = []
    for step in (20, 40, 60, 80, 100):
        row = {"step/optimizer": step, "step/outer": step}
        for dataset, count in (("math500", 8000), ("amc23", 640), ("aime25", 480)):
            row[f"eval/{dataset}/response_count"] = count
            row[f"eval/{dataset}/failed_count"] = 0
        rows.append({"step": step, "data": row})
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert queue.validate_result(job, tmp_path)["optimizer_steps"] == 100
    rows[-1]["data"]["eval/aime25/failed_count"] = 1
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="Incomplete validation"):
        queue.validate_result(job, tmp_path)


def test_cleanup_does_not_terminate_reused_pid(monkeypatch):
    own = Mock()
    own.create_time.return_value = 10
    other = Mock()
    other.create_time.return_value = 30
    monkeypatch.setattr(queue.psutil, "Process", lambda pid: {1: own, 2: other}[pid])
    monkeypatch.setattr(queue.psutil, "wait_procs", lambda *a, **kw: ([], []))
    process = Mock()
    process.poll.return_value = 0
    queue.cleanup_owned(process, {1: 10, 2: 20})
    own.terminate.assert_called_once()
    other.terminate.assert_not_called()
    process.terminate.assert_not_called()


def setup_queue(tmp_path):
    config = {
        **plan(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "jobs": [queue.make_jobs()[0]],
    }
    queue.write_json(tmp_path / "plan.json", config)
    return SimpleNamespace(
        root=tmp_path, source=ROOT, netrc_file=tmp_path / "unused", idle_seconds=0, poll_seconds=0.01
    )


def test_stop_file_prevents_launch(tmp_path, monkeypatch):
    args = setup_queue(tmp_path)
    commit = json.loads((tmp_path / "plan.json").read_text())["source_commit"]
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: commit + "\n")
    (tmp_path / "STOP").touch()
    launch = Mock()
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.signal, "signal", lambda *a: None)
    queue.run_queue(args)
    launch.assert_not_called()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "stopped"


def test_preflight_fallback_is_global_and_bounded(tmp_path, monkeypatch):
    args = setup_queue(tmp_path)
    commit = json.loads((tmp_path / "plan.json").read_text())["source_commit"]
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: commit + "\n")
    launches = []

    def launch(command, *, env, **kwargs):
        launches.append(env["ROLLOUT_ENFORCE_EAGER"])
        code = 1 if len(launches) == 1 else 0
        return SimpleNamespace(returncode=code, poll=lambda: code)

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue.signal, "signal", lambda *a: None)
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: (True, {}))
    monkeypatch.setattr(queue, "cleanup_owned", lambda *a: None)
    monkeypatch.setattr(queue, "add_credentials", lambda *a: None)
    monkeypatch.setattr(queue, "validate_result", lambda *a: {"optimizer_steps": 2})
    queue.run_queue(args)
    state = json.loads((tmp_path / "status.json").read_text())
    assert launches == ["False", "True"]
    assert state["state"] == "finished" and state["eager"]
    assert state["jobs"]["preflight"]["attempt"] == 2
