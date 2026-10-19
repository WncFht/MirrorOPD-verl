# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Write one offline W&B record and verify the checkout's log directory."""

import os
from pathlib import Path


def main():
    """Verify SDK mode, run location and the completed local run file without loading a model."""
    root = Path(__file__).resolve().parent.parent
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = str(root)
    import wandb

    with wandb.init(project="MirrorOPD-verl", name="installation-check") as run:
        run.log({"installation_check": 1})
        assert run.settings.mode == "offline", run.settings.mode
        run_dir = Path(run.dir).resolve()
        assert run_dir.is_relative_to(root / "wandb"), run_dir
        run_file = run_dir.parent / f"run-{run.id}.wandb"
    assert run_file.is_file() and run_file.stat().st_size > 0, run_file
    print("W&B offline write passed:", run_file)


if __name__ == "__main__":
    main()
