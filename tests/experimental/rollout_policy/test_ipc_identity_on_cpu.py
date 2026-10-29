# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep matching endpoints isolated even when local Ray job counters repeat."""

from verl.workers.rollout.ipc_identity import ipc_job_key


def test_unscoped_existing_jobs_keep_their_identity(monkeypatch):
    monkeypatch.delenv("VERL_ROLLOUT_IPC_NAMESPACE", raising=False)
    assert ipc_job_key("01000000") == "01000000"


def test_same_job_counter_in_independent_runs_has_different_sockets(monkeypatch):
    monkeypatch.setenv("VERL_ROLLOUT_IPC_NAMESPACE", "run-one")
    first = ipc_job_key("01000000")
    assert first == ipc_job_key("01000000")
    monkeypatch.setenv("VERL_ROLLOUT_IPC_NAMESPACE", "run-two")
    assert first != ipc_job_key("01000000")


def test_namespace_is_safe_and_bounded_for_unix_socket_paths(monkeypatch):
    monkeypatch.setenv("VERL_ROLLOUT_IPC_NAMESPACE", "a/path with spaces" * 100)
    key = ipc_job_key("01000000")
    assert "/" not in key and " " not in key
    assert len(f"/tmp/rl-colocate-zmq-{key}-replica-4000-rank-0.sock") < 108
