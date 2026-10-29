# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Local operation journals and subprocess diagnostics for manual transfers."""

import codecs
import json
import os
import platform
import re
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

_active = None
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
PALETTE = {
    "START": "1;36",
    "BEGIN": "36",
    "SUCCESS": "32",
    "DONE": "1;32",
    "SUCCEEDED": "32",
    "FAIL": "1;31",
    "FAILED": "31",
    "ERROR": "31",
    "PARTIAL": "1;33",
    "INCOMPLETE": "33",
    "WARN": "33",
    "WAIT": "34",
    "WAITING": "33",
    "PENDING": "34",
    "RUNNING": "36",
    "INTERRUPTED": "33",
    "SKIP": "90",
    "SKIPPED": "90",
    "SUPERSEDED": "90",
}


class PartialResultError(ValueError):
    """Some inputs need newer data; keep a nonzero exit without calling this a transport failure."""

    partial = True
    returncode = 2


def color_enabled(console):
    mode = os.environ.get("MIRROROPD_COLOR", "auto").lower()
    if "NO_COLOR" in os.environ or mode == "never":
        return False
    if mode == "always":
        return True
    return bool(getattr(console, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def paint(text, console):
    text = ANSI.sub("", text)
    if not color_enabled(console):
        return text
    text = re.sub(r"^\[[^\]\n]+\]", lambda m: f"\033[2m{m[0]}\033[0m", text)
    pattern = r"\b(" + "|".join(PALETTE) + r")\b"
    return re.sub(pattern, lambda m: f"\033[{PALETTE[m[0].upper()]}m{m[0]}\033[0m", text, flags=re.I)


def console_print(message=""):
    if _active:
        _active.emit(str(message))
    else:
        print(paint(redact(message), sys.stdout))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(text):
    """Mask common credential forms and known secret environment values, without logging the environment."""
    text = ANSI.sub("", str(text))
    for key, value in os.environ.items():
        if len(value) >= 8 and re.search(r"TOKEN|PASSWORD|SECRET|API_KEY", key, re.I):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(?i)(bearer\s+)[\w.\-]+", r"\1[REDACTED]", text)
    return re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)


def scrub(value):
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(item) for item in value]
    return redact(value) if isinstance(value, str | Path) else value


class Tee:
    def __init__(self, console, journal):
        self.console, self.journal = console, journal

    def write(self, text):
        self.journal.write(text, self.console)
        return len(text)

    def flush(self):
        self.console.flush()
        self.journal.stream.flush()

    def isatty(self):
        return False


class Journal:
    def __init__(self, directory, action, details):
        directory.mkdir(parents=True, exist_ok=True)
        identity = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}_{action}_{uuid.uuid4().hex[:8]}"
        self.log_path = directory / f"{identity}.log"
        self.report_path = directory / f"{identity}.json"
        descriptor = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)
        self.mutex = threading.RLock()
        self.console = sys.stdout
        self.started = time.monotonic()
        self.stack = []
        self.data = {
            "version": 1,
            "id": identity,
            "action": action,
            "status": "running",
            "started_at": now(),
            "host": platform.node(),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "cwd": str(Path.cwd()),
            "log": str(self.log_path),
            "details": details,
            "stages": [],
        }
        self.save()

    def write(self, text, console=None):
        with self.mutex:
            text = redact(text)
            if console is not None:
                console.write(paint(text, console))
                console.flush()
            self.stream.write(ANSI.sub("", text).replace("\r", "\n"))

    def emit(self, message, console=True):
        self.write(f"[{now()}] {message}\n", self.console if console else None)

    def save(self):
        with self.mutex:
            self.data["updated_at"] = now()
            self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
            temporary = self.report_path.with_suffix(".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(scrub(self.data), ensure_ascii=False, indent=2, default=str) + "\n")
            temporary.replace(self.report_path)

    def heartbeat(self, stop):
        while not stop.wait(15):
            with self.mutex:
                phase = self.stack[-1]["name"] if self.stack else self.data["action"]
                current = time.monotonic()
                progress = getattr(self, "process_progress", None)
                suffix = ""
                if progress:
                    suffix = (
                        f"; command elapsed {current - progress['started']:.0f}s"
                        f"; no new output {current - progress['last']:.0f}s"
                    )
                self.emit(f"WAIT {phase}; total elapsed {current - self.started:.0f}s{suffix}")
                self.save()


@contextmanager
def operation(directory, action, **details):
    """Record every invocation separately; failures and interruptions never look like completed uploads."""
    global _active
    previous = _active
    journal = Journal(Path(directory).resolve(), action, details)
    _active = journal
    stop = threading.Event()
    heartbeat = threading.Thread(target=journal.heartbeat, args=(stop,), daemon=True)
    try:
        journal.emit(f"START {action} | log: {journal.log_path}")
        journal.emit("Context: " + json.dumps(details, ensure_ascii=False, default=str), console=False)
        heartbeat.start()
        with redirect_stdout(Tee(sys.stdout, journal)), redirect_stderr(Tee(sys.stderr, journal)):
            try:
                yield journal
            except BaseException as error:
                journal.data["status"] = (
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else ("partial" if getattr(error, "partial", False) else "failed")
                )
                journal.data["exit_code"] = (
                    130 if isinstance(error, KeyboardInterrupt) else getattr(error, "returncode", 1)
                )
                journal.data["error"] = f"{type(error).__name__}: {error}"
                journal.write(traceback.format_exc())
                label = "PARTIAL" if getattr(error, "partial", False) else "FAIL"
                journal.emit(f"{label} {action}: {error}")
                raise
            else:
                journal.data["status"] = "success"
                journal.data["exit_code"] = 0
                journal.emit(f"DONE {action} in {time.monotonic() - journal.started:.1f}s")
    finally:
        stop.set()
        if heartbeat.ident is not None:
            heartbeat.join(timeout=2)
        journal.data["finished_at"] = now()
        journal.save()
        journal.emit(f"Report: {journal.report_path}")
        journal.stream.close()
        _active = previous


@contextmanager
def stage(name, hint=None, **details):
    journal = _active
    if journal is None:
        yield None
        return
    record = {"name": name, "status": "running", "started_at": now(), "details": details}
    started = time.monotonic()
    with journal.mutex:
        journal.data["stages"].append(record)
        journal.stack.append(record)
        journal.save()
    journal.emit(f"BEGIN {name}" + (" | " + json.dumps(details, ensure_ascii=False, default=str) if details else ""))
    try:
        yield record
    except BaseException as error:
        record["status"] = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else ("partial" if getattr(error, "partial", False) else "failed")
        )
        record["error"] = str(error)
        journal.data.setdefault("partial_stage" if getattr(error, "partial", False) else "failed_stage", name)
        if hint:
            record["hint"] = hint
            journal.emit(f"Recovery: {hint}")
        raise
    else:
        if record["status"] == "running":
            record["status"] = "success"
    finally:
        with journal.mutex:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["finished_at"] = now()
            journal.stack.pop()
            journal.save()
        journal.emit(f"{record['status'].upper()} {name} ({record['elapsed_seconds']:.1f}s)")


def note(message, *, console=True, **details):
    if _active:
        _active.emit(
            message + (" | " + json.dumps(details, ensure_ascii=False, default=str) if details else ""), console=console
        )


def update(**details):
    if _active:
        _active.data["details"].update(details)
        _active.save()


def skip(name, reason):
    if _active:
        _active.data["stages"].append({"name": name, "status": "skipped", "reason": reason})
        _active.save()
        _active.emit(f"SKIP {name}: {reason}")


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()


def streamed(command, journal, **kwargs):
    check = kwargs.pop("check", False)
    timeout = kwargs.pop("timeout", None)
    idle_timeout = kwargs.pop("idle_timeout", None)
    on_output = kwargs.pop("on_output", None)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, **kwargs
    )
    started = time.monotonic()
    last_progress = started
    if journal:
        journal.process_progress = {"started": started, "last": started}
    progress_lines = deque(maxlen=256)
    progress_buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    tail = ""
    secrets = [
        v
        for k, v in {**os.environ, **kwargs.get("env", {})}.items()
        if len(v) >= 8 and re.search(r"TOKEN|PASSWORD|SECRET|API_KEY", k, re.I)
    ]

    def emit(text):
        for value in secrets:
            text = text.replace(value, "[REDACTED]")
        if journal:
            journal.write(text, journal.console)
        else:
            sys.stdout.write(redact(text))
            sys.stdout.flush()

    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(command, timeout, output=tail)
                if idle_timeout is not None and time.monotonic() - last_progress >= idle_timeout:
                    raise subprocess.TimeoutExpired(
                        command,
                        idle_timeout,
                        output=tail,
                        stderr=f"No new SDK output for {idle_timeout}s (idle timeout)",
                    )
                events = selector.select(timeout=0.2)
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        pending += decoder.decode(b"", final=True)
                        selector.unregister(key.fileobj)
                        break
                    decoded = decoder.decode(chunk)
                    progress_buffer += decoded
                    lines = re.split(r"[\r\n]", progress_buffer)
                    progress_buffer = lines.pop()[-8192:]
                    for line in lines:
                        line = ANSI.sub("", line).strip()
                        if line and line not in progress_lines:
                            last_progress = time.monotonic()
                            if journal:
                                journal.process_progress["last"] = last_progress
                            progress_lines.append(line)
                    if on_output:
                        on_output(decoded)
                    tail = (tail + decoded)[-65536:]
                    pending += decoded
                # Display progress that uses carriage returns or no newline, while retaining split credential prefixes.
                for secret in secrets:
                    pending = pending.replace(secret, "[REDACTED]")
                held = 0
                for secret in secrets:
                    for length in range(1, min(len(secret), len(pending) + 1)):
                        if pending.endswith(secret[:length]):
                            held = max(held, length)
                if len(pending) > held:
                    emit(pending[:-held] if held else pending)
                    pending = pending[-held:] if held else ""
            emit(pending)
        remaining = None if timeout is None else max(0.001, timeout - (time.monotonic() - started))
        result = subprocess.CompletedProcess(command, process.wait(timeout=remaining), stdout=tail)
    except BaseException:
        stop_process(process)
        raise
    finally:
        process.stdout.close()
        if journal:
            journal.process_progress = None
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=tail)
    return result


def run(command, *, stream=False, **kwargs):
    """Capture diagnostics and stream partial lines with a real subprocess deadline."""
    journal = _active
    if journal is None and not stream:
        return subprocess.run(command, **kwargs)
    started = time.monotonic()
    label = shlex.join(str(arg) for arg in command)
    if journal:
        journal.emit(f"COMMAND {label}", console=False)
        if stream:
            short = shlex.join([Path(str(command[0])).name, *(str(arg) for arg in command[1:3])])
            journal.emit(f"RUN {short} … (full command in log)")
    try:
        if stream:
            result = streamed(command, journal, **kwargs)
        else:
            result = subprocess.run(command, **kwargs)
            for output in (result.stdout, result.stderr):
                if output:
                    journal.write(output.decode(errors="replace") if isinstance(output, bytes) else output)
    except subprocess.CalledProcessError as error:
        if journal:
            for output in (error.stdout, error.stderr):
                if output:
                    journal.write(output.decode(errors="replace") if isinstance(output, bytes) else output)
            journal.emit(f"COMMAND FAILED exit={error.returncode}: {label}")
        raise
    if journal:
        journal.emit(f"EXIT {result.returncode} in {time.monotonic() - started:.2f}s: {label}", console=False)
    return result
