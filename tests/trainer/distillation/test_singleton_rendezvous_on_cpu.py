# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""TP1 checkpoint groups never compete with another process's listening port."""

import socket

import pytest
import torch
import torch.distributed as dist

from verl.utils.distributed import initialize_global_process_group_ray


def test_singleton_cpu_group_works_with_a_busy_master_port(monkeypatch):
    if dist.is_initialized():
        pytest.skip("An enclosing distributed test owns the default group")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(listener.getsockname()[1]))
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        try:
            initialize_global_process_group_ray(timeout_second=10, backend="cpu:gloo")
            value = torch.tensor([7.0])
            dist.all_reduce(value)
            assert value.item() == 7 and dist.get_world_size() == 1
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
