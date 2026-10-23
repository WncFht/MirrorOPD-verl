# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Test automatic publication with a local Git remote and a fake W&B uploader."""

import argparse
import importlib.util
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.uploaded = []
        self.upload_error = False

    def git(self, repo, *args):
        return self.real_run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    def fake_sync(self, command, **kwargs):
        if command[0] != "wandb":
            return self.real_run(command, **kwargs)
        copied = Path(command[2])
        self.uploaded.append(copied)
        self.assertNotEqual(copied, self.destination)
        self.assertFalse((copied / "run-test.wandb.synced").exists())
        self.assertFalse((copied / "run-test.wandb.syncstate").exists())
        (copied / "files/config.yaml").write_text("rewritten config")
        (copied / "files/wandb-summary.json").write_text("rewritten summary")
        (copied / "run-test.wandb.syncstate").write_text("new sync state")
        if self.upload_error:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    def unpack(self):
        args = argparse.Namespace(kind="logs", archive=str(self.archive), paths=[])
        with patch.object(transfer.subprocess, "run", side_effect=self.fake_sync):
            transfer.unpack(args, self.base / "snap")

    def test_defaults_upload_copies_and_push_only_project_logs(self):
        unrelated = self.logs / "unrelated.txt"
        unrelated.write_text("staged elsewhere")
        self.git(self.logs, "add", "unrelated.txt")
        self.unpack()
        self.assertEqual(len(self.uploaded), 1)
        self.assertFalse(self.uploaded[0].exists())
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
        self.assertEqual([run.name for run in self.uploaded], [self.run_name])

    def test_upload_failure_keeps_inputs_and_does_not_publish(self):
        self.upload_error = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.unpack()
        self.assertTrue(self.archive.exists())
        self.assertTrue((self.destination / "run-test.wandb").is_file())
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), self.initial)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.initial)

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


if __name__ == "__main__":
    unittest.main()
