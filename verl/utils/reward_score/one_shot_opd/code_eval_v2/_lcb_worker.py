# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Disposable lightweight LCB interpreter. Invoked by file path, never -m verl."""

import ctypes
import importlib
import io
import json
import os
import resource
import signal
import sys
import time
import types
from pathlib import Path

from _isolation import MIB, Limits, OutputLimitError


def run(directory, parent_pid, cgroup):
    # Kill this interpreter if its supervisor dies; handle the setup race too.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot install parent-death signal")
    if os.getppid() != parent_pid:
        return
    control = json.loads((directory / "control.json").read_text())
    limits = Limits(**control["limits"])
    if cgroup:
        (Path(cgroup) / "cgroup.procs").write_text(str(os.getpid()))
    resource.setrlimit(resource.RLIMIT_AS, (limits.address_space_mib * MIB,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.diagnostic_mib * MIB,) * 2)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    cpu_budget = int(limits.startup_timeout_s + (control["timeout"] + 1) * control["n_cases"] + 10)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_budget, cpu_budget + 1))
    output = (directory / "worker-result.json").open("w")
    result = dict(job_id=control["job_id"], status="INFRASTRUCTURE_ERROR")
    started = time.monotonic()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    ready = False
    try:
        request = json.loads((directory / "request.json").read_text())
        package = types.ModuleType("_mirroropd_lcb_worker")
        package.__path__ = [str(Path(__file__).parent)]
        sys.modules[package.__name__] = package
        runner = importlib.import_module(package.__name__ + "._lcb_runner")
        # Fail dependency errors before any generated code is evaluated.
        exec(runner.import_string, {})
        parsed = json.loads(request["sample"]["input_output"])
        budgets = iter(
            max(limits.output_mib * MIB, 4 * len(str(value).encode("utf-8")) + 65536) for value in parsed["outputs"]
        )
        del parsed

        class BoundedCapture:
            def __enter__(self):
                self.previous = sys.stdout
                self.stream = io.StringIO()
                self.limit = next(budgets)
                self.used = 0
                self.value = None
                sys.stdout = self
                return self

            def write(self, value):
                self.used += len(value.encode("utf-8"))
                if self.used > self.limit:
                    raise OutputLimitError()
                return self.stream.write(value)

            def flush(self):
                pass

            def close(self):
                pass

            def __exit__(self, typ, value, tb):
                sys.stdout = self.previous
                self.value = self.stream.getvalue() if typ is None else ""
                self.stream.close()

            def __getitem__(self, index):
                if index != 0:
                    raise IndexError(index)
                return self.value

        runner.Capturing = BoundedCapture
        # Keep comparison rules intact; suppress unbounded diagnostic payloads.
        runner.truncatefn = lambda value, length=300: value[:length] if isinstance(value, str) else "<non-text value>"
        result["startup_s"] = time.monotonic() - started
        (directory / "ready").write_text(control["job_id"])
        ready = True
        judge_started = time.monotonic()
        scores, metadata = runner.run_test(request["sample"], test=request["generation"], timeout=request["timeout"])
        if not scores:
            raise RuntimeError("Runner returned no test results")
        result["results"] = [value is True for value in scores]
        if all(result["results"]):
            result["status"] = "PASS"
        else:
            code = metadata.get("error_code")
            result["status"] = "TIME_LIMIT" if code == -3 else "RUNTIME_ERROR" if code == -4 else "WRONG_ANSWER"
        result["judge_s"] = time.monotonic() - judge_started
    except OutputLimitError:
        result["status"] = "OUTPUT_LIMIT" if ready else "INFRASTRUCTURE_ERROR"
    except MemoryError:
        result.update(status="MEMORY_LIMIT" if ready else "INFRASTRUCTURE_ERROR", error_type="MemoryError")
    except SyntaxError:
        result.update(status="SYNTAX_ERROR" if ready else "INFRASTRUCTURE_ERROR", error_type="SyntaxError")
    except BaseException as exc:
        if ready and type(exc).__name__ == "TimeoutException":
            result["status"] = "TIME_LIMIT"
        else:
            result.update(status="INFRASTRUCTURE_ERROR", error_type=type(exc).__name__, message=str(exc)[:500])
    finally:
        signal.alarm(0)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        json.dump(result, output)
        output.close()


if __name__ == "__main__":
    run(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
