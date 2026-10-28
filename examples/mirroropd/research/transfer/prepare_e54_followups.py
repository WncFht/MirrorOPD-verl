# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare bounded E54 continuation and reuse controls; does not launch GPUs."""

import argparse
import json
import shutil
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    old = repo / "outputs/e54-rope-debug-20261028-hppk13"
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    shutil.copytree(old / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "wandb", ".git"))
    shutil.copyfile(Path(__file__).with_name("e54_dynamic_debug.py"), source / "e54_dynamic_debug.py")
    controller = (
        (old / "controller.py")
        .read_text()
        .replace("len(results)==1", "len(results)==len(json.loads((ROOT/'plan.json').read_text())['jobs'])")
    )
    controller = controller.replace("E54 five-step diagnostic ended", "E54 bounded follow-ups ended")
    (root / "controller.py").write_text(controller)
    original = json.loads((old / "run/launch.json").read_text())
    jobs = []
    for name, resume, total, reuse, save in [("continue-r16", 5, 15, 16, 5), ("branch-r4", 2, 14, 4, 4)]:
        run = root / name
        run.mkdir()
        checkpoint = old / f"run/training/checkpoints/global_step_{resume}"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["actor_optimizer_step_attempts"] == state["actor_optimizer_step"] == resume * 16
        run_id = uuid.uuid4().hex[:8]

        def translate(text, run=run, run_id=run_id, name=name):
            return (
                text.replace(str(old / "source"), str(source))
                .replace(str(old / "run"), str(run))
                .replace(original["environment"]["WANDB_RUN_ID"], run_id)
                .replace(old.name, root.name + "-" + name)
            )

        env = {k: translate(v) for k, v in original["environment"].items()}
        # The source directory is shared, while run artifacts and Ray temp dirs are independent.
        env["PYTHONPATH"] = str(source)
        env["MIRROROPD_LONG_RUN_ID"] = root.name
        env.update(
            TOTAL_TRAINING_STEPS=str(total),
            PPO_EPOCHS=str(reuse),
            SAVE_FREQ=str(save),
            E54_DEBUG_DIR=str(run / "debug"),
            E54_DEBUG_UPDATE_START=str(state["actor_optimizer_step"]),
            E54_DEBUG_OUTER_START=str(resume),
            E54_DEBUG_REUSE=str(reuse),
        )
        command = [
            translate(v)
            for v in original["training_command"]
            if not v.startswith(("trainer.resume_mode=", "trainer.save_freq=", "trainer.max_actor_ckpt_to_keep="))
        ]
        # translate() replaces the old run name in Ray environment values as well.
        command = [
            f"++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={source}"
            if v.startswith("++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=")
            else v
            for v in command
        ]
        command = [
            f"++ray_kwargs.ray_init._temp_dir=/tmp/e54-{run_id}"
            if v.startswith("++ray_kwargs.ray_init._temp_dir=")
            else v
            for v in command
        ]
        command = [
            f"++ray_kwargs.ray_init.runtime_env.env_vars.MIRROROPD_LONG_RUN_ID={root.name}"
            if v.startswith("++ray_kwargs.ray_init.runtime_env.env_vars.MIRROROPD_LONG_RUN_ID=")
            else v
            for v in command
        ]
        command += [
            "trainer.resume_mode=resume_path",
            f"trainer.resume_from_path={checkpoint}",
            "trainer.del_local_ckpt_after_load=False",
            f"trainer.save_freq={save}",
            "trainer.max_actor_ckpt_to_keep=5",
        ]
        command += [
            f"++ray_kwargs.ray_init.runtime_env.env_vars.{k}='{env[k]}'"
            for k in ["E54_DEBUG_UPDATE_START", "E54_DEBUG_OUTER_START", "E54_DEBUG_REUSE"]
        ]
        launch = dict(environment=env, command=command, training_command=command, cwd=str(source))
        (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        jobs.append({"preset": name, "directory": str(run), "wandb_run_id": run_id})
    (root / "plan.json").write_text(json.dumps({"jobs": jobs, "restore_physical_gpus": [4, 5, 6, 7]}, indent=2))
    print(json.dumps({"root": str(root), "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
