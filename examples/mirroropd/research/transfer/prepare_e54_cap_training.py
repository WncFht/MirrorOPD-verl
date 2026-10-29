# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a four-GPU shared warm-up, two cap arms, and fixed evaluation.

Both arms restore the same four-GPU warm-up's step-2 model, Adam and loader.
The supervisor is scoped to GPUs 0-3 and restores occupancy on every exit.
"""

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
    shutil.copytree(old / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "wandb"))
    shutil.copyfile(Path(__file__).with_name("e54_dynamic_debug.py"), source / "e54_dynamic_debug.py")
    controller = (old / "controller.py").read_text()
    controller = controller.replace("CUDA_VISIBLE_DEVICES='4,5,6,7'", "CUDA_VISIBLE_DEVICES='0,1,2,3'")
    controller = controller.replace(
        "state.get('cuda_visible_devices') == '4,5,6,7'", "physical_gpus(status) == [0, 1, 2, 3]"
    )
    controller = controller.replace(
        "status.get('status', {}).get('cuda_visible_devices') != '4,5,6,7'", "physical_gpus(status) != [0, 1, 2, 3]"
    )
    controller = controller.replace(
        "state.get('status', {}).get('cuda_visible_devices') != '4,5,6,7'", "physical_gpus(state) != [0, 1, 2, 3]"
    )
    controller = controller.replace("'physical_gpus': [4, 5, 6, 7]", "'physical_gpus': [0, 1, 2, 3]")
    controller = controller.replace("GPU 4-7 mapping", "GPU 0-3 mapping")
    controller = controller.replace(
        "['nvidia-smi', '--query-compute-apps=pid'", "['nvidia-smi', '--id=0,1,2,3', '--query-compute-apps=pid'"
    )
    controller = controller.replace(
        "['nvidia-smi', '--query-gpu=index", "['nvidia-smi', '--id=0,1,2,3', '--query-gpu=index"
    )
    controller = controller.replace(
        "len(results)==1", "len(results)==len(json.loads((ROOT/'plan.json').read_text())['jobs'])"
    )
    controller = controller.replace("E54 five-step diagnostic ended", "E54 cap study ended")
    controller = controller.replace(
        "    return results\n\n\ndef run():",
        "        if code != 0:\n            break\n    return results\n\n\ndef run():",
    )
    mapping = """def physical_gpus(status):
    state = status.get('status', {})
    selected = state.get('gpus', [])
    visible = state.get('cuda_visible_devices')
    if visible in (None, '', '全部'):
        return selected
    try:
        devices = [int(x) for x in visible.split(',')]
        return [devices[i] for i in selected]
    except (ValueError, IndexError, TypeError):
        return None


"""
    assert "def restored(status):" in controller and "4,5,6,7" not in controller
    controller = controller.replace("def restored(status):", mapping + "def restored(status):")
    (root / "controller.py").write_text(controller)
    original = json.loads((old / "run/launch.json").read_text())
    jobs = []
    for label, cap, total, resume in [
        ("warmup", 7168, 2, False),
        ("cap7168", 7168, 5, True),
        ("cap16384", 16384, 5, True),
    ]:
        run = root / label
        run.mkdir()
        run_id = "c" + uuid.uuid4().hex[:7]

        def translate(text, run=run, run_id=run_id, label=label):
            return (
                text.replace(str(old / "source"), str(source))
                .replace(str(old / "run"), str(run))
                .replace(original["environment"]["WANDB_RUN_ID"], run_id)
                .replace(old.name, root.name + "-" + label)
            )

        env = {k: translate(v) for k, v in original["environment"].items()}
        env.update(
            CUDA_VISIBLE_DEVICES="0,1,2,3",
            STUDENT_GPUS="4",
            MAX_RESPONSE_LENGTH=str(cap),
            TOTAL_TRAINING_STEPS=str(total),
            SAVE_FREQ="1",
            TEST_FREQ="-1",
            PPO_EPOCHS="16",
            E54_DEBUG_DIR=str(run / "debug"),
            PYTHONPATH=str(source),
            MIRROROPD_LONG_RUN_ID=root.name,
        )
        command = [translate(x) for x in original["training_command"]]
        replacements = {
            "++ray_kwargs.ray_init.num_gpus=": "4",
            "++ray_kwargs.ray_init._temp_dir=": "/tmp/ecap-" + run_id,
            "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=": str(source),
            "++ray_kwargs.ray_init.runtime_env.env_vars.MIRROROPD_LONG_RUN_ID=": root.name,
        }
        command = [
            next((prefix + value for prefix, value in replacements.items() if x.startswith(prefix)), x) for x in command
        ]
        if resume:
            checkpoint = root / "warmup/training/checkpoints/global_step_2"
            command = [x for x in command if not x.startswith("trainer.resume_mode=")]
            command += [
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={checkpoint}",
                "trainer.del_local_ckpt_after_load=False",
            ]
            env.update(E54_DEBUG_UPDATE_START="32", E54_DEBUG_OUTER_START="2", E54_DEBUG_REUSE="16")
            command += [
                f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}='{env[key]}'"
                for key in ["E54_DEBUG_UPDATE_START", "E54_DEBUG_OUTER_START", "E54_DEBUG_REUSE"]
            ]
        launch = dict(environment=env, command=command, training_command=command, cwd=str(source))
        (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        jobs.append({"preset": label, "directory": str(run), "wandb_run_id": run_id})
    evaluation = root / "evaluation"
    evaluation.mkdir()
    launch = {
        "cwd": str(repo),
        "environment": {
            "PYTHONPATH": str(repo),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "OMP_NUM_THREADS": "2",
            "MIRROROPD_LONG_RUN_ID": root.name,
        },
        "command": [
            original["environment"]["PYTHON_BIN"],
            "-u",
            "-m",
            "examples.mirroropd.research.transfer.evaluate_e54_cap_training",
            "--root",
            str(root),
        ],
    }
    (evaluation / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    jobs.append({"preset": "evaluation", "directory": str(evaluation)})
    (root / "plan.json").write_text(json.dumps({"jobs": jobs, "restore_physical_gpus": [0, 1, 2, 3]}, indent=2) + "\n")
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "models": "Qwen3-1.7B-Base <- Qwen3-30B-A3B-Instruct-2507",
                "initialization": "Both arms restore the identical four-GPU warmup step-2 model, Adam and loader",
                "shared_warmup_outer_steps": 2,
                "new_outer_steps": {"warmup": 2, "cap7168": 3, "cap16384": 3},
                "gpus": [0, 1, 2, 3],
                "outer_steps_per_arm": 5,
                "reuse": 16,
                "batch": 64,
                "lr": 1e-6,
                "seed": 42,
                "train_caps": [7168, 16384],
                "checkpoint_every_outer": True,
                "rope": "Canonical FP32 buffers preserved",
                "template": "Unchanged Base empty-think common prompt",
                "comparison": "Only training response cap differs between the two four-GPU arms",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"root": str(root), "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
