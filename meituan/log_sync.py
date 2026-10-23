# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Commit received logs and upload immutable Git snapshots through a persistent queue."""

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    from meituan import sync_observe as observe
except ModuleNotFoundError:
    import sync_observe as observe

ROOT = Path(__file__).resolve().parent.parent


def git(repo, *args, check=True):
    return observe.run(
        ["git", "--literal-pathspecs", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def repository(destination, source_root=ROOT):
    parent = destination.resolve()
    while not parent.exists():
        parent = parent.parent
    repo = Path(git(parent, "rev-parse", "--show-toplevel").stdout.strip())
    if repo.resolve() == source_root.resolve():
        raise ValueError("Receive logs in the separate wandb-logs repository, not the source checkout")
    destination.resolve().relative_to(repo.resolve())
    return repo


def queue_dir(repo):
    return Path(git(repo, "rev-parse", "--absolute-git-dir").stdout.strip()) / "mirroropd-upload"


@contextmanager
def lock(directory, name, blocking=True):
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise ValueError(f"Another log operation owns {directory / name}") from error
        yield stream.fileno()


def load(directory):
    path = directory / "queue.json"
    state = json.loads(path.read_text()) if path.exists() else {"version": 1, "jobs": []}
    if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("jobs"), list):
        raise ValueError(f"Invalid queue state: {path}; keep it for recovery instead of deleting it")
    required = {"id", "path", "run", "tree", "commit", "cloud", "status", "attempts", "enqueued_at"}
    statuses = {"pending", "running", "failed", "incomplete", "succeeded", "superseded"}
    if any(
        not isinstance(job, dict)
        or not required.issubset(job)
        or job["status"] not in statuses
        or not isinstance(job["attempts"], int)
        or job["attempts"] < 0
        for job in state["jobs"]
    ):
        raise ValueError(f"Invalid task in {path}; preserve the queue for recovery")
    return state


def save(directory, state):
    path = directory / "queue.json.tmp"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(state, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    path.replace(directory / "queue.json")


def cloud_target():
    return {
        "base_url": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/"),
        "entity": os.environ.get("MIRROROPD_UPLOAD_ENTITY", ""),
        "project": os.environ.get("MIRROROPD_UPLOAD_PROJECT", ""),
    }


def commit_logs(repo, destination):
    path = destination.resolve().relative_to(repo.resolve()).as_posix()
    with observe.stage("logs.git_commit", repository=str(repo), path=path):
        git(repo, "add", "-A", "--", path)
        diff = git(repo, "diff", "--cached", "--quiet", "--", path, check=False)
        if diff.returncode == 1:
            git(repo, "commit", "--only", "-m", "chore(logs): receive W&B run snapshots", "--", path)
        else:
            diff.check_returncode()
            observe.note("No new log changes to commit")
        commit = git(repo, "rev-parse", "HEAD").stdout.strip()
        observe.update(logs_commit=commit, logs_repository=str(repo))
        return commit


def enqueue(repo, destination, commit):
    """Recover old backlog as well as this receipt; reuse Git tree IDs instead of custom file hashes."""
    relative = destination.resolve().relative_to(repo.resolve()).as_posix()
    listing = git(repo, "ls-tree", "-z", f"{commit}:{relative}").stdout
    directory = queue_dir(repo)
    target = cloud_target()
    added = 0
    with lock(directory, "queue.lock"):
        state = load(directory)
        for entry in listing.split("\0"):
            if not entry:
                continue
            metadata, name = entry.split("\t", 1)
            mode, kind, tree = metadata.split()
            if mode != "040000" or kind != "tree" or not name.startswith("offline-run-"):
                continue
            path = relative + "/" + name
            if any(job["path"] == path and job["tree"] == tree and job["cloud"] == target for job in state["jobs"]):
                continue
            identity = uuid.uuid4().hex
            # New snapshots supersede queued old snapshots of this run; running snapshots finish first.
            for older in state["jobs"]:
                if (
                    older["path"] == path
                    and older["cloud"] == target
                    and older["status"] in ("pending", "failed", "incomplete")
                ):
                    older.update(status="superseded", superseded_by=identity)
            git(repo, "update-ref", f"refs/mirroropd-upload/{identity}", commit)
            state["jobs"].append(
                {
                    "id": identity,
                    "path": path,
                    "run": name,
                    "tree": tree,
                    "commit": commit,
                    "cloud": target,
                    "status": "pending",
                    "attempts": 0,
                    "enqueued_at": observe.now(),
                }
            )
            added += 1
        save(directory, state)
    observe.update(upload_queue=str(directory / "queue.json"), queued_snapshots=added)
    print(f"Queued {added} new run snapshot(s); existing completed snapshots are not uploaded again.")


def receive(incoming, destination, source_root, *, push=True):
    repo = repository(destination, source_root)
    directory = queue_dir(repo)
    with lock(directory, "receive.lock", blocking=False):
        with observe.stage("logs.merge", destination=str(destination)):
            shutil.copytree(incoming, destination, dirs_exist_ok=True)
        commit = commit_logs(repo, destination)
        with observe.stage("logs.enqueue"):
            enqueue(repo, destination, commit)
    # A failed Git push must not erase queued uploads or block later cloud-only retries.
    if push:
        push_logs(repo)
    return repo


def push_logs(repo):
    with observe.stage("logs.git_push", hint="Local commit and upload queue are retained. Retry make sync-log-push."):
        observe.run(["git", "-C", str(repo), "push"], stream=True, check=True)


def materialize(repo, directory, job):
    """A persistent, per-snapshot copy absorbs all SDK writes; retries never rewrite it from raw logs."""
    cache = directory / "cache" / job["id"]
    copied = cache / job["run"]
    if not copied.is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache.parent) as temp:
            staging = Path(temp)
            bundle = staging / "source.tar"
            with bundle.open("wb") as output:
                subprocess.run(
                    ["git", "-C", str(repo), "archive", "--format=tar", job["tree"]], stdout=output, check=True
                )
            extracted = staging / job["run"]
            extracted.mkdir()
            with tarfile.open(bundle) as archive:
                archive.extractall(extracted, filter="data")
            bundle.unlink()
            # Old flags in the received raw snapshot cannot prove this queue task completed.
            for pattern in ("*.wandb.synced", "*.wandb.syncstate"):
                for marker in extracted.glob(pattern):
                    marker.unlink()
            staging.rename(cache)
    files = list(copied.glob("*.wandb"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one .wandb file in {copied}; found {len(files)}")
    return copied, files[0]


def sdk_command():
    default = ROOT / ".venv-wandb-sync/bin/wandb"
    executable = os.environ.get("MIRROROPD_WANDB_BIN") or (str(default) if default.is_file() else "wandb")
    executable = shutil.which(executable)
    if not executable:
        raise ValueError("W&B uploader not installed. Run make sync-log-setup on the uploading machine")
    version = observe.run(
        [executable, "--version"], capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()
    match = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (0, 31, 0):
        raise ValueError(f"Uploader {version!r} is older than the validated SDK. Run make sync-log-setup")
    help_text = observe.run(
        [executable, "beta", "sync", "--help"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    if not all(flag in help_text for flag in ("--verbose", "--yes", "--skip-synced")):
        raise ValueError("The uploader must support wandb beta sync; run make sync-log-setup. No legacy fallback.")
    observe.update(wandb_executable=executable, wandb_version=version, sync_engine="core")
    print(f"Uploader: {executable} ({version}); core sync, no legacy fallback")
    return [executable, "beta", "sync", "--yes", "--verbose"]


def selected(job, run):
    return not run or job["run"] == run or job["run"].endswith("-" + run)


def upload(repo, *, run="", concurrency=2, timeout=900, retry=False):
    if concurrency < 1 or concurrency > 8 or timeout <= 0:
        raise ValueError("Use WANDB_UPLOAD_JOBS=1..8 and a positive WANDB_UPLOAD_TIMEOUT")
    directory = queue_dir(repo)
    with lock(directory, "worker.lock", blocking=False) as worker_fd:
        with lock(directory, "queue.lock"):
            state = load(directory)
            # Owning the worker lock proves no earlier worker is still processing these jobs.
            for job in state["jobs"]:
                if job["status"] == "running":
                    job.update(status="failed", error="Previous worker was interrupted", retry_after=0)
            ready = [
                job.copy()
                for job in state["jobs"]
                if selected(job, run)
                and (job["status"] in ("pending", "failed") or (retry and job["status"] == "incomplete"))
                and (retry or (job["attempts"] < 3 and job.get("retry_after", 0) <= time.time()))
            ]
            save(directory, state)
        if run and not any(selected(job, run) for job in state["jobs"]):
            raise ValueError(f"No queued run matches {run!r}; use make sync-log-status")
        if not ready:
            unresolved = [
                j for j in state["jobs"] if selected(j, run) and j["status"] in ("pending", "failed", "incomplete")
            ]
            if unresolved:
                raise ValueError(
                    f"{len(unresolved)} unresolved jobs. Incomplete files need a newer snapshot. "
                    "Use RETRY=1 to override cooldown after inspecting status."
                )
            print("No pending uploads for this selection")
            return
        command = sdk_command()
        failed = []
        while ready:
            batch = []
            for job in ready[:]:
                if (not batch or job["cloud"] == batch[0]["cloud"]) and job["path"] not in {j["path"] for j in batch}:
                    batch.append(job)
                    ready.remove(job)
                    if len(batch) >= concurrency:
                        break
            prepared = []
            for job in batch:
                try:
                    copied, file = materialize(repo, directory, job)
                    prepared.append((job, copied, file))
                except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
                    finish_job(directory, job, "failed", str(error))
                    failed.append(job["run"])
            if not prepared:
                continue
            with lock(directory, "queue.lock"):
                state = load(directory)
                eligible = {
                    item["id"] for item in state["jobs"] if item["status"] in ("pending", "failed", "incomplete")
                }
                prepared = [item for item in prepared if item[0]["id"] in eligible]
                for current in state["jobs"]:
                    if current["id"] in {job["id"] for job, _, _ in prepared}:
                        current.update(status="running", attempts=current["attempts"] + 1, started_at=observe.now())
                save(directory, state)
            if not prepared:
                continue
            target = prepared[0][0]["cloud"]
            attempt_dir = directory / "attempts" / uuid.uuid4().hex
            (attempt_dir / "tmp").mkdir(parents=True)
            args = [*command, "-n", str(concurrency)]
            for key in ("entity", "project"):
                if target[key]:
                    args.extend(["--" + key, target[key]])
            args.extend(str(copied) for _, copied, _ in prepared)
            environment = {
                **os.environ,
                "WANDB_BASE_URL": target["base_url"],
                "WANDB_MODE": "online",
                "WANDB_DIR": str(attempt_dir),
                "TMPDIR": str(attempt_dir / "tmp"),
            }
            error_text = ""
            output = ""
            incomplete_names = set()
            line_buffer = ""

            def inspect_output(chunk, names=incomplete_names):
                nonlocal line_buffer
                line_buffer += chunk
                lines = re.split(r"[\r\n]", line_buffer)
                line_buffer = lines.pop()[-8192:]
                for line in lines:
                    match = re.search(r'Failed to sync all data from "([^"]+)": .*incomplete record', line)
                    if match:
                        names.update(match[1].split("/"))

            interrupted = False
            try:
                with observe.stage(
                    "wandb.batch",
                    runs=[job["run"] for job, _, _ in prepared],
                    timeout=timeout,
                    sdk_directory=str(attempt_dir),
                ):
                    result = observe.run(
                        args,
                        stream=True,
                        check=True,
                        timeout=timeout,
                        cwd=str(attempt_dir),
                        env=environment,
                        pass_fds=(worker_fd,),
                        on_output=inspect_output,
                    )
                    output = result.stdout or ""
                    if any(not file.with_name(file.name + ".synced").is_file() for _, _, file in prepared):
                        raise ValueError("SDK returned 0 without completing every snapshot; no completion marker")
            except KeyboardInterrupt:
                interrupted = True
                error_text = "Interrupted by user"
            except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                error_text = str(error)
                output = getattr(error, "stdout", "") or output
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
            inspect_output("\n")
            for job, copied, file in prepared:
                # A zero exit code alone is insufficient (e.g. skipped authentication or a missing run).
                complete = file.with_name(file.name + ".synced").is_file()
                incomplete = not complete and job["run"] in incomplete_names
                status = "succeeded" if complete else ("incomplete" if incomplete else "failed")
                message = (
                    ""
                    if complete
                    else (
                        "Run file has an incomplete record; receive a newer source snapshot"
                        if incomplete
                        else (error_text or "SDK exited without a completion marker; inspect SDK logs")
                    )
                )
                finish_job(directory, job, status, message, cache=str(copied), sdk_directory=str(attempt_dir))
                print(f"{job['run']}: {status}; snapshot={job['tree'][:12]}; cache={copied}")
                if not complete:
                    failed.append(job["run"])
            if interrupted:
                raise KeyboardInterrupt()
        if failed:
            raise ValueError(
                f"{len(failed)} upload(s) incomplete: {', '.join(failed)}. Cache retained; use sync-log-status"
            )


def finish_job(directory, job, status, error, **extra):
    with lock(directory, "queue.lock"):
        state = load(directory)
        current = next(item for item in state["jobs"] if item["id"] == job["id"])
        # Preparation failures also consume an attempt, instead of failing forever without a limit.
        if current["status"] != "running":
            current["attempts"] += 1
        if status in ("failed", "incomplete"):
            newer = next(
                (
                    item
                    for item in state["jobs"]
                    if item["path"] == job["path"]
                    and item["cloud"] == job["cloud"]
                    and item["id"] != job["id"]
                    and item["enqueued_at"] >= current["enqueued_at"]
                    and item["status"] == "pending"
                ),
                None,
            )
            if newer:
                status = "superseded"
                current["superseded_by"] = newer["id"]
        current.update(status=status, error=observe.redact(error), finished_at=observe.now(), **extra)
        if status == "failed":
            current["retry_after"] = time.time() + min(60 * 2 ** (current["attempts"] - 1), 3600)
        save(directory, state)
    observe.note("Upload task result", run=job["run"], snapshot=job["tree"], status=status, error=error)


def main():
    # Import path selection only at the CLI boundary; transfer.py calls the library without a circular import.
    try:
        from meituan import transfer
    except ModuleNotFoundError:
        import transfer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enqueue", "upload", "push"))
    args = parser.parse_args()
    destination = transfer.wandb_runs_dir("unpack")
    snap = transfer.repo_path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap"))
    diagnostics = Path(os.environ.get("MIRROROPD_OPERATION_DIR") or snap / "operations")
    try:
        with observe.operation(diagnostics, f"logs-{args.action}", destination=str(destination)):
            repo = repository(destination)
            if args.action == "enqueue":
                with lock(queue_dir(repo), "receive.lock", blocking=False):
                    enqueue(repo, destination, commit_logs(repo, destination))
            elif args.action == "push":
                push_logs(repo)
            else:
                upload(
                    repo,
                    run=os.environ.get("MIRROROPD_UPLOAD_RUN", ""),
                    concurrency=int(os.environ.get("MIRROROPD_UPLOAD_JOBS", "2")),
                    timeout=float(os.environ.get("MIRROROPD_UPLOAD_TIMEOUT", "900")),
                    retry=os.environ.get("MIRROROPD_UPLOAD_RETRY", "0") == "1",
                )
    except KeyboardInterrupt:
        parser.exit(130, "upload interrupted; queue and cached copies retained\n")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, observe.redact(f"log-sync: {error}\n"))


if __name__ == "__main__":
    main()
