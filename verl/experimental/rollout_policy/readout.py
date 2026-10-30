# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only FP32 HF exports at drained, single-rank scientific readout boundaries."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def readout_checkpoint_settings(manager):
    """Restore checkpoint retention and RNG even when a readout export fails."""
    if manager.world_size != 1:
        raise ValueError("Scientific readout export currently requires one FSDP rank")
    if manager.per_tensor_param_fn is None:
        raise ValueError("Scientific readout needs streamed full HF parameters")
    if manager.checkpoint_config and manager.checkpoint_config.get("async_save", False):
        raise ValueError("Scientific readout export must complete synchronously")
    contents, dtype, step = manager.checkpoint_save_contents, manager.hf_export_dtype, manager.previous_global_step
    paths = manager.previous_saved_paths
    prior_paths = list(paths)
    rng = manager.get_rng_state()
    manager.checkpoint_save_contents = ["hf_model"]
    manager.hf_export_dtype = torch.float32
    try:
        yield
    finally:
        manager.checkpoint_save_contents = contents
        manager.hf_export_dtype = dtype
        manager.previous_global_step = step
        paths[:] = prior_paths
        manager.previous_saved_paths = paths
        manager.load_rng_state(rng)


def export_scientific_readout(engine, local_path, global_step):
    """Save master-precision weights without saving or advancing optimizer state."""
    if Path(local_path).exists():
        raise FileExistsError(f"Readout directory already exists: {local_path}")
    manager = engine.checkpoint_manager
    with readout_checkpoint_settings(manager):
        engine.save_checkpoint(local_path=local_path, global_step=global_step, max_ckpt_to_keep=None)


def state_digest(value):
    """Content hash for the smoke gate, including actual CUDA RNG and Adam tensors."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            if hasattr(item, "to_local"):
                item = item.to_local()
            cpu = item.detach().cpu().contiguous()
            digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                digest.update(str(key).encode())
                visit(child)
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child)
        elif hasattr(item, "tobytes"):
            digest.update(item.tobytes())
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()
