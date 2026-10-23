# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host spans shared by driver, worker and asynchronous IO threads.

No CUDA calls or synchronization. Per-process JSONL files use monotonic nanoseconds
and record clock anchors; payloads contain sizes/IDs only, never prompts or weights.
"""

import functools
import inspect
import json
import numbers
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_lock = threading.Lock()
_files = {}


def enabled():
    return bool(os.environ.get("VERL_E2E_TIMELINE_DIR"))


def emit(name, start_ns=None, **fields):
    root = os.environ.get("VERL_E2E_TIMELINE_DIR")
    if not root:
        return
    end = time.monotonic_ns()
    pid = os.getpid()
    try:
        domain = os.readlink("/proc/self/ns/time")
    except OSError:
        domain = None
    row = dict(
        name=name,
        pid=pid,
        tid=threading.get_native_id(),
        host=socket.gethostname(),
        clock_domain=domain,
        start_ns=end if start_ns is None else start_ns,
        end_ns=end,
        wall_ns=time.time_ns(),
        mono_ns=end,
        rank=os.environ.get("RANK"),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    row.update(fields)
    payload = (
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda v: int(v) if isinstance(v, numbers.Integral) else str(type(v).__name__),
        )
        + "\n"
    ).encode()
    with _lock:
        key = (root, pid)
        if key not in _files:
            Path(root).mkdir(parents=True, exist_ok=True)
            _files[key] = os.open(
                str(Path(root) / f"events-{pid}.jsonl"), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
            )
        os.write(_files[key], payload)


@contextmanager
def span(name, **fields):
    if not enabled():
        yield fields
        return
    start = time.monotonic_ns()
    status = "ok"
    try:
        yield fields
    except BaseException:
        status = "error"
        raise
    finally:
        emit(name, start, status=status, **fields)


def observe(name, role):
    """Observe an async request, preserving arguments, return values and exceptions."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            if not enabled():
                return await fn(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs).arguments
            obj = bound.get("self")
            extra = bound.get("kwargs", {})
            trajectory = bound.get("trajectory", {})
            fields = dict(
                role=role,
                request_id=bound.get("request_id"),
                replica=getattr(obj, "replica_rank", None),
                step=trajectory.get("step", getattr(obj, "global_steps", extra.get("global_steps"))),
                validate=trajectory.get("validate"),
                uid=extra.get("uid"),
            )
            if "prompt_ids" in bound:
                fields["input_tokens"] = len(bound["prompt_ids"])
            with span(name, **fields) as row:
                result = await fn(*args, **kwargs)
                if hasattr(result, "token_ids"):
                    row["output_tokens"] = len(result.token_ids)
                    row["finish_reason"] = result.extra_fields.get("finish_reason", result.stop_reason)
                return result

        return wrapped

    return decorate


def observe_sync(name, role):
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs).arguments
            fields = {k: bound[k] for k in ("device", "model", "optimizer", "grad") if k in bound}
            with span(name, role=role, **fields):
                return fn(*args, **kwargs)

        return wrapped

    return decorate
