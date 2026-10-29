# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in IPC isolation for independent Ray clusters on the same host."""

import hashlib
import os


def ipc_job_key(job_id):
    """Ray job counters can repeat across clusters; scope local development sockets."""
    namespace = os.environ.get("VERL_ROLLOUT_IPC_NAMESPACE")
    if not namespace:
        return str(job_id)
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:16]
    return f"{digest}-{job_id}"
