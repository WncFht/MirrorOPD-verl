# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare, but do not launch, five E54 outer steps with correct RoPE and probes.

Keep the previous immutable E54 runtime, change its buffer cast, add read-only
diagnostics, and use the existing recovery supervisor. No experiment queue.
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path


def replace(path, before, after):
    text = path.read_text()
    assert text.count(before) == 1, (path, before, text.count(before))
    path.write_text(text.replace(before, after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    root = args.root.resolve()
    old_control = repo / "outputs/qwen-e54-retry-20261027-hppk13"
    old_run = repo / "outputs/e54-full-r16-100step-fivebench-retry-20261027-hppk13"
    root.mkdir(parents=True, exist_ok=False)
    source, run = root / "source", root / "run"
    run.mkdir()
    shutil.copytree(old_control / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Copy only the already accepted parameter-casting fix, not unrelated main changes.
    shutil.copyfile(repo / "verl/utils/torch_dtypes.py", source / "verl/utils/torch_dtypes.py")
    helper = Path(__file__).with_name("e54_dynamic_debug.py")
    shutil.copyfile(helper, source / "e54_dynamic_debug.py")
    engine = source / "verl/workers/engine/fsdp/transformer_impl.py"
    replace(
        engine,
        "            module.to(torch_dtype)",
        "            from verl.utils.torch_dtypes import cast_module_parameters\n"
        "            from e54_dynamic_debug import rope_check\n"
        "\n            cast_module_parameters(module, torch_dtype)\n            rope_check(module)",
    )
    replace(
        engine,
        "            def compute(h, w):\n"
        "                with torch.autocast(device_type=h.device.type, enabled=False):",
        "            def compute(h, w):\n                from e54_dynamic_debug import observe\n\n"
        "                with torch.autocast(device_type=h.device.type, enabled=False):\n"
        "                    observe(self, micro_batch, positions, h, w.to(h.dtype), "
        "teacher_hidden.to(h.dtype), weight)",
    )
    replace(
        source / "verl/workers/engine/base.py",
        '        with span("actor/zero_grad", role="fsdp"):',
        "        from e54_dynamic_debug import begin_update\n\n        begin_update(self)\n"
        '        with span("actor/zero_grad", role="fsdp"):',
    )
    replace(
        source / "verl/trainer/ppo/v1/trainer_base.py",
        '        """Update the actor network."""',
        '        """Update the actor network."""\n'
        "        from e54_dynamic_debug import dump_rollout\n\n        dump_rollout(self, batch)",
    )
    controller = root / "controller.py"
    shutil.copyfile(old_control / "controller.py", controller)
    text = controller.read_text()
    text = text[: text.index("        predecessor = json.loads")] + "        run()\n"
    text = text.replace("'E54 retry ended'", "'E54 five-step diagnostic ended'")
    controller.write_text(text)
    launch = json.loads((old_run / "launch.json").read_text())
    old_source = str(old_control / "source")
    old_id, new_id = launch["environment"]["WANDB_RUN_ID"], uuid.uuid4().hex[:8]

    def translate(text):
        return (
            text.replace(old_source, str(source))
            .replace(str(old_run), str(run))
            .replace(old_run.name, root.name)
            .replace(old_control.name, root.name)
            .replace(old_id, new_id)
            .replace("/tmp/e54-retry-h13-1027", "/tmp/" + root.name)
        )

    env = {k: translate(v) for k, v in launch["environment"].items()}
    env.update(
        TOTAL_TRAINING_STEPS="5",
        TEST_FREQ="-1",
        SAVE_FREQ="1",
        E54_DEBUG_DIR=str(run / "debug"),
    )
    command = [translate(v) for v in launch["training_command"]]
    command += [
        "trainer.test_freq=-1",
        "trainer.save_freq=1",
        "trainer.max_actor_ckpt_to_keep=5",
        "trainer.resume_mode=disable",
        f"++ray_kwargs.ray_init.runtime_env.env_vars.E54_DEBUG_DIR={env['E54_DEBUG_DIR']}",
    ]
    launch.update(environment=env, command=command, training_command=command, cwd=str(source))
    (run / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    plan = {"jobs": [{"preset": "E54", "directory": str(run), "wandb_run_id": new_id}]}
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (root / "changes.json").write_text(
        json.dumps(
            {
                "base_source": str(old_control / "source"),
                "training_changes": ["Preserve original FP32 RoPE buffers in actor and teacher loading"],
                "observability": ["Raw rollout IDs before update", "Pre-update fixed-prefix stop diagnostics"],
                "budget": {"outer": 5, "reuse": 16, "save_every": 1, "validation": False},
                "unchanged": ["template", "loss", "learning_rate", "batch_size", "cap", "seed", "sampling"],
                "restore_physical_gpus": [4, 5, 6, 7],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"root": str(root), "wandb_run_id": new_id}))


if __name__ == "__main__":
    main()
