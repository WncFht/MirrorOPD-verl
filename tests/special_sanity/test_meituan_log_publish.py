# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Test automatic publication with a local Git remote and a fake W&B uploader."""

import argparse
import importlib.util
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meituan import log_sync

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("log_publish_transfer", ROOT / "meituan/transfer.py")
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class LogPublishTest(unittest.TestCase):
    """Keep original data and unrelated index entries intact through uploads and retries."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="log-publish-test-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.logs = self.base / "wandb-logs"
        self.remote = self.base / "remote.git"
        self.real_run = subprocess.run
        self.git(self.base, "init", "--bare", str(self.remote))
        self.git(self.base, "init", "-b", "main", str(self.logs))
        self.git(self.logs, "config", "user.name", "Test")
        self.git(self.logs, "config", "user.email", "test@example.invalid")
        (self.logs / ".gitignore").write_text("*.wandb.synced\n*.wandb.syncstate\n")
        self.git(self.logs, "add", ".gitignore")
        self.git(self.logs, "commit", "-m", "initial")
        self.git(self.logs, "remote", "add", "origin", str(self.remote))
        self.git(self.logs, "push", "-u", "origin", "main")
        self.initial = self.git(self.logs, "rev-parse", "HEAD")
        self.run_name = "offline-run-test"
        self.source = self.base / self.run_name
        (self.source / "files").mkdir(parents=True)
        (self.source / "run-test.wandb").write_bytes(b"original run")
        (self.source / "run-test.wandb.synced").write_text("")
        (self.source / "run-test.wandb.syncstate").write_text("old sync state")
        (self.source / "files/config.yaml").write_text("original config")
        (self.source / "files/wandb-summary.json").write_text('{"loss": 1}')
        self.archive = self.base / "log_MirrorOPD-verl_test.tar.gz"
        with tarfile.open(self.archive, "w:gz") as bundle:
            bundle.add(self.source, arcname="wandb/" + self.run_name)
        self.destination = self.logs / "MirrorOPD-verl/wandb" / self.run_name
        env = {key: value for key, value in os.environ.items() if not key.startswith(("MIRROROPD_", "WANDB_"))}
        env["MIRROROPD_WANDB_LOGS_REPO"] = str(self.logs)
        env["MIRROROPD_WANDB_BIN"] = str(self.base / "must-not-upload")
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.uploaded = []
        self.upload_error = False

    def git(self, repo, *args):
        return self.real_run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    def unpack(self):
        args = argparse.Namespace(kind="logs", archive=str(self.archive), paths=[])
        transfer.unpack(args, self.base / "snap")

    def jobs(self):
        return log_sync.load(log_sync.queue_dir(self.logs))["jobs"]

    def test_defaults_queue_and_push_only_project_logs(self):
        unrelated = self.logs / "unrelated.txt"
        unrelated.write_text("staged elsewhere")
        self.git(self.logs, "add", "unrelated.txt")
        self.unpack()
        self.assertEqual(self.jobs()[0]["status"], "pending")
        self.assertEqual((self.destination / "files/config.yaml").read_text(), "original config")
        self.assertEqual((self.destination / "files/wandb-summary.json").read_text(), '{"loss": 1}')
        self.assertEqual((self.destination / "run-test.wandb").read_bytes(), b"original run")
        self.assertEqual(self.git(self.logs, "diff", "--cached", "--name-only"), "unrelated.txt")
        published = self.git(self.remote, "ls-tree", "-r", "--name-only", "main")
        self.assertNotIn("unrelated.txt", published)
        self.assertNotIn(".syncstate", published)
        self.assertNotIn(".synced", published)
        self.assertIn("run-test.wandb", published)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.git(self.logs, "rev-parse", "HEAD"))

    def test_import_also_commits_forgotten_changes_in_older_runs(self):
        old_run = self.logs / "MirrorOPD-verl/wandb/offline-run-older"
        old_run.mkdir(parents=True)
        config = old_run / "config.yaml"
        config.write_text("before")
        obsolete = old_run / "obsolete.txt"
        obsolete.write_text("remove later")
        self.git(self.logs, "add", "MirrorOPD-verl/wandb")
        self.git(self.logs, "commit", "-m", "older run")
        self.git(self.logs, "push")
        config.write_text("forgotten update")
        obsolete.unlink()
        forgotten = self.logs / "MirrorOPD-verl/wandb/offline-run-forgotten"
        forgotten.mkdir()
        (forgotten / "run-forgotten.wandb").write_bytes(b"forgotten run")
        (forgotten / "run-forgotten.wandb.syncstate").write_text("local state")
        self.unpack()
        prefix = "MirrorOPD-verl/wandb/"
        self.assertEqual(
            self.git(self.remote, "show", "main:" + prefix + "offline-run-older/config.yaml"), "forgotten update"
        )
        published = self.git(self.remote, "ls-tree", "-r", "--name-only", "main")
        self.assertIn("run-forgotten.wandb", published)
        self.assertNotIn("obsolete.txt", published)
        self.assertNotIn(".syncstate", published)
        self.assertEqual(len(self.jobs()), 3)

    def test_repeated_receipt_does_not_duplicate_queue_or_commit(self):
        self.unpack()
        commit = self.git(self.logs, "rev-parse", "HEAD")
        first = self.jobs()[0]["id"]
        self.unpack()
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(self.jobs()[0]["id"], first)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), commit)

    def test_failed_push_retries_without_duplicate_commit(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        with self.assertRaises(subprocess.CalledProcessError):
            self.unpack()
        commit = self.git(self.logs, "rev-parse", "HEAD")
        self.assertNotEqual(commit, self.initial)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.initial)
        self.assertTrue(self.archive.exists())
        self.assertEqual(self.jobs()[0]["status"], "pending")
        hook.unlink()
        self.unpack()
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), commit)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), commit)

    def test_upload_can_be_disabled_without_disabling_git(self):
        with patch.dict(os.environ, MIRROROPD_WANDB_SYNC="0"):
            self.unpack()
        self.assertEqual(self.uploaded, [])
        self.assertNotEqual(self.git(self.remote, "rev-parse", "main"), self.initial)

    def test_make_defaults_publish_without_extra_flags(self):
        bin_dir = self.base / "bin"
        bin_dir.mkdir()
        fake_cli = bin_dir / "wandb"
        fake_cli.write_text('#!/bin/sh\nset -e\nprintf "rewritten by fake uploader" > "$2/files/config.yaml"\n')
        fake_cli.chmod(0o755)
        env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
        self.real_run(
            [
                "make",
                "-C",
                str(ROOT),
                "sync-log-unpack",
                f"LOG_ARCHIVE={self.archive}",
                f"WANDB_LOGS_REPO={self.logs}",
                f"OPERATION_DIR={self.base / 'operations'}",
                f"WANDB_BIN={fake_cli}",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual((self.destination / "files/config.yaml").read_text(), "original config")
        self.assertNotEqual(self.git(self.remote, "rev-parse", "main"), self.initial)

    def test_refuse_automatic_commit_into_training_checkout(self):
        with patch.object(transfer, "ROOT", self.logs), self.assertRaisesRegex(ValueError, "separate wandb-logs"):
            self.unpack()
        self.assertEqual(self.uploaded, [])
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), self.initial)

    def test_enqueue_automatically_pushes_and_upload_retries_missed_push(self):
        shutil.copytree(self.source, self.destination)
        log_sync.enqueue_logs(self.logs, self.destination.parent)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.git(self.logs, "rev-parse", "HEAD"))
        (self.destination / "files/config.yaml").write_text("new raw config")
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        pending = self.git(self.logs, "rev-parse", "HEAD")
        self.assertNotEqual(self.git(self.remote, "rev-parse", "main"), pending)
        with patch.object(log_sync, "upload", return_value={"succeeded": 1}) as cloud:
            log_sync.upload_and_push(self.logs)
            cloud.assert_called_once()
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), pending)

    def test_failed_auto_push_does_not_block_cloud_attempt(self):
        shutil.copytree(self.source, self.destination)
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        hook = self.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        with patch.object(log_sync, "upload", return_value={"succeeded": 1}) as cloud:
            with self.assertRaisesRegex(ValueError, "Git synchronization failed"):
                log_sync.upload_and_push(self.logs)
            cloud.assert_called_once()
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.initial)
        self.assertNotEqual(self.git(self.logs, "rev-parse", "HEAD"), self.initial)

    def test_push_does_not_rebase_or_force_a_diverged_remote(self):
        other = self.base / "other"
        self.git(self.base, "clone", "-b", "main", str(self.remote), str(other))
        self.git(other, "config", "user.name", "Test")
        self.git(other, "config", "user.email", "test@example.invalid")
        (other / "remote-only.txt").write_text("remote data")
        self.git(other, "add", "remote-only.txt")
        self.git(other, "commit", "-m", "remote update")
        self.git(other, "push")
        remote_head = self.git(self.remote, "rev-parse", "main")
        shutil.copytree(self.source, self.destination)
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        local_head = self.git(self.logs, "rev-parse", "HEAD")
        with self.assertRaises(subprocess.CalledProcessError):
            log_sync.push_logs(self.logs)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), local_head)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), remote_head)

    def other_checkout(self):
        other = self.base / "other"
        self.git(self.base, "clone", "-b", "main", str(self.remote), str(other))
        self.git(other, "config", "user.name", "Test")
        self.git(other, "config", "user.email", "test@example.invalid")
        return other

    def publish_remote_run(self, other):
        run = other / "MirrorOPD-verl/wandb/offline-run-remote"
        run.mkdir(parents=True)
        (run / "run-remote.wandb").write_bytes(b"remote log")
        self.git(other, "add", ".")
        self.git(other, "commit", "-m", "remote log")
        self.git(other, "push")
        return self.git(other, "rev-parse", "HEAD")

    def test_upload_fetches_fast_forwards_and_enqueues_remote_runs(self):
        remote_head = self.publish_remote_run(self.other_checkout())

        def upload_queued(*args, **kwargs):
            self.assertEqual([job["run"] for job in self.jobs()], ["offline-run-remote"])
            self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), remote_head)

        with patch.object(log_sync, "upload", side_effect=upload_queued):
            log_sync.upload_and_push(self.logs)
        self.assertEqual(self.git(self.logs, "status", "--porcelain"), "")

    def test_upload_merges_diverged_histories_and_enqueues_both_sides(self):
        remote_head = self.publish_remote_run(self.other_checkout())
        shutil.copytree(self.source, self.destination)
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        local_head = self.git(self.logs, "rev-parse", "HEAD")
        with patch.object(log_sync, "upload"):
            log_sync.upload_and_push(self.logs)
        self.assertEqual(
            set(self.git(self.logs, "show", "-s", "--format=%P", "HEAD").split()), {local_head, remote_head}
        )
        self.assertEqual({job["run"] for job in self.jobs()}, {self.run_name, "offline-run-remote"})
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), self.git(self.remote, "rev-parse", "main"))

    def test_sync_uses_configured_upstream_for_fetch_and_push(self):
        remote_head = self.publish_remote_run(self.other_checkout())
        self.git(self.logs, "remote", "rename", "origin", "logs-origin")
        self.git(self.logs, "config", "remote.pushDefault", "not-the-log-remote")
        with patch.object(log_sync, "upload"):
            log_sync.upload_and_push(self.logs)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), remote_head)
        self.assertEqual(self.git(self.logs, "rev-parse", "@{upstream}"), remote_head)

    def test_binary_conflict_aborts_only_automatic_merge_and_keeps_queue(self):
        self.unpack()
        other = self.other_checkout()
        (other / self.destination.relative_to(self.logs) / "run-test.wandb").write_bytes(b"\0remote")
        self.git(other, "add", ".")
        self.git(other, "commit", "-m", "remote snapshot")
        self.git(other, "push")
        remote_head = self.git(other, "rev-parse", "HEAD")
        (self.destination / "run-test.wandb").write_bytes(b"\0local")
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        local_head = self.git(self.logs, "rev-parse", "HEAD")
        queue = self.jobs()
        with patch.object(log_sync, "upload") as cloud:
            with self.assertRaisesRegex(ValueError, "Automatic merge conflicted and was aborted"):
                log_sync.upload_and_push(self.logs)
            cloud.assert_called_once()
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), local_head)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), remote_head)
        self.assertEqual((self.destination / "run-test.wandb").read_bytes(), b"\0local")
        self.assertEqual(self.git(self.logs, "status", "--porcelain"), "")
        self.assertFalse((self.logs / ".git/MERGE_HEAD").exists())
        self.assertEqual(self.jobs(), queue)

    def test_remote_update_does_not_overwrite_unrelated_staged_changes(self):
        remote_head = self.publish_remote_run(self.other_checkout())
        (self.logs / "notes.txt").write_text("user staged content")
        self.git(self.logs, "add", "notes.txt")
        with patch.object(log_sync, "upload"):
            with self.assertRaisesRegex(ValueError, "Commit or stash local changes"):
                log_sync.upload_and_push(self.logs)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), self.initial)
        self.assertEqual(self.git(self.logs, "diff", "--cached", "--name-only"), "notes.txt")
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), remote_head)

    def test_existing_merge_is_not_aborted_or_received_over(self):
        (self.logs / ".git/MERGE_HEAD").write_text(self.initial + "\n")
        with self.assertRaisesRegex(ValueError, "existing Git operation"):
            self.unpack()
        self.assertTrue((self.logs / ".git/MERGE_HEAD").exists())
        self.assertFalse(self.destination.exists())

    def test_merge_preserves_ignored_local_file_when_remote_starts_tracking_it(self):
        other = self.other_checkout()
        (other / "local-cache.txt").write_text("remote tracked file")
        self.git(other, "add", ".")
        self.git(other, "commit", "-m", "track file")
        self.git(other, "push")
        (self.logs / ".git/info/exclude").write_text("local-cache.txt\n")
        (self.logs / "local-cache.txt").write_text("local ignored data")
        with patch.object(log_sync, "upload"):
            with self.assertRaisesRegex(ValueError, "Git synchronization failed"):
                log_sync.upload_and_push(self.logs)
        self.assertEqual((self.logs / "local-cache.txt").read_text(), "local ignored data")
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), self.initial)

    def test_fetch_failure_still_processes_existing_queue_and_disable_skips_network(self):
        shutil.copytree(self.source, self.destination)
        log_sync.enqueue_logs(self.logs, self.destination.parent, push=False)
        self.git(self.logs, "remote", "set-url", "origin", str(self.base / "missing.git"))
        with patch.object(log_sync, "upload") as cloud:
            with self.assertRaisesRegex(ValueError, "Git synchronization failed"):
                log_sync.upload_and_push(self.logs)
            cloud.assert_called_once()
        self.assertEqual(len(self.jobs()), 1)
        with patch.object(log_sync, "fetch_and_merge", side_effect=AssertionError("network disabled")):
            with patch.object(log_sync, "upload") as cloud:
                log_sync.upload_and_push(self.logs, push=False)
                cloud.assert_called_once()


if __name__ == "__main__":
    unittest.main()
