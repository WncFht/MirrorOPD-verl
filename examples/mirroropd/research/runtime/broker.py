# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one explicit command through an existing local broker and resource guard."""

import os
import subprocess
import sys
import time
from pathlib import Path

from ..core.io import dump, read


def occupied(resource, gpu):
    state = read(Path(resource) / "status.json")["gpus"][str(gpu)]
    return state["phase"] == "occupying" and state.get("occupier_alive", False)


def run_leased(*, resource, repo, gpu, owner, output, command, stop_paths=(), env=None):
    """Reserve an idle occupier-owned GPU, guard/attach a child, and await restoration.

    Caller supplies the installed broker directory. No existing reservation is
    released or reassigned. The broker owns restoration after the guard exits.
    A private gate prevents the command from running before attach succeeds.
    Nonzero command exit is preserved in exit.json and returned to the caller.
    """
    resource, repo, output = Path(resource), Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if any(Path(path).exists() for path in stop_paths):
        raise InterruptedError("STOP exists before GPU reservation")
    if not occupied(resource, gpu):
        raise RuntimeError(f"GPU {gpu} has an active allocation; wait for broker occupancy")
    python = sys.executable
    gate = output / "ATTACHED"
    guarded = [
        python,
        str(resource / "runtime/resource_guard.py"),
        "run",
        "--repo",
        str(repo),
        "--gpus",
        str(gpu),
        "--",
        python,
        "-m",
        "examples.mirroropd.research.runtime.broker",
        "--gate",
        str(gate),
        "--",
        *command,
    ]
    child = None
    reserved = attached = False
    cleanup_errors = []
    with (output / "broker.log").open("x") as log:

        def broker_command(action, *arguments):
            return subprocess.run(
                [python, str(resource / "broker.py"), action, "--gpu", str(gpu), "--owner", owner, *arguments],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        try:
            broker_command("reserve")
            reserved = True
            child = subprocess.Popen(
                guarded, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            dump(output / "process.json", dict(pid=child.pid, argv=guarded, owner=owner, started_at=time.time()))
            broker_command("attach", "--pid", str(child.pid), "--exit-receipt", str(output / "exit.json"))
            attached = True
            if any(Path(path).exists() for path in stop_paths):
                raise InterruptedError("STOP arrived during broker attach")
            gate.touch()
            while child.poll() is None:
                if any(Path(path).exists() for path in stop_paths):
                    child.terminate()
                    break
                time.sleep(1)
            code = child.wait(timeout=90)
            return code
        except BaseException as error:
            dump(output / "error.json", dict(error=repr(error), reserved=reserved, attached=attached, at=time.time()))
            raise
        finally:
            if child is not None and child.poll() is None:
                # Signal only our own guard; it forwards STOP to its protected command.
                try:
                    child.terminate()
                    child.wait(timeout=90)
                except Exception as error:
                    cleanup_errors.append(repr(error))
            if child is not None:
                dump(output / "exit.json", dict(returncode=child.poll(), finished_at=time.time()))
            if reserved and not attached and (child is None or child.poll() is not None):
                try:
                    # The broker checks owner identity before releasing this failed launch.
                    broker_command("release", "--reason", "Guard launch/attach failed before execution gate")
                except Exception as error:
                    cleanup_errors.append(repr(error))
            restored = False
            if reserved:
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    try:
                        restored = occupied(resource, gpu)
                    except Exception as error:
                        cleanup_errors.append(repr(error))
                        break
                    if restored:
                        break
                    time.sleep(2)
                dump(
                    output / "occupancy.json",
                    dict(verified=restored, checked_at=time.time(), cleanup_errors=cleanup_errors),
                )
                if not restored:
                    # Preserve the original exception if setup failed; successful commands
                    # must still fail their caller when GPU restoration is unverified.
                    if sys.exc_info()[0] is None:
                        raise RuntimeError(f"GPU {gpu} occupancy not restored; inspect existing broker")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Internal attach gate, executed inside the existing guard")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    deadline = time.monotonic() + 120
    while not args.gate.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Broker attach gate timed out")
        time.sleep(0.1)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
