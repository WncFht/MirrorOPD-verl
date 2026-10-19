# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One-Shot-OPD supplied-test execution with bounded time and process cleanup."""

import os
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

from ._base_imports import BASE_IMPORTS

_ERROR_MSG_PREFIX = "Failed to execute program: "
_MAX_TIMEOUT = 60
# Apply limits in the child interpreter, avoiding preexec_fn in reward threads.
_LIMITS = """
import platform
import resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    if platform.system() != "Darwin":
        resource.setrlimit(resource.RLIMIT_DATA, (4 * 1024**3, 4 * 1024**3))
except (OSError, ValueError):
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
except (OSError, ValueError):
    pass
"""


def run_test(code: str, test: str = None, timeout: int = _MAX_TIMEOUT) -> tuple[bool, str]:
    """Execute one supplied test suite; a timeout terminates the owned process group."""
    if not test:
        raise ValueError("No test provided.")
    timeout = min(timeout, _MAX_TIMEOUT)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    with TemporaryDirectory() as directory:
        path = os.path.join(directory, "solution.py")
        with open(path, "w") as output:
            output.write(f"{_LIMITS}\n{BASE_IMPORTS}\n\n{code}\n\n{test}\n")
        with subprocess.Popen(
            [sys.executable, path],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                return False, _ERROR_MSG_PREFIX + f"Execution timed out after {timeout} seconds."
            stdout = stdout.decode(errors="replace")
            stderr = stderr.decode(errors="replace").strip()
            if process.returncode == 0:
                return True, stdout
            return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
