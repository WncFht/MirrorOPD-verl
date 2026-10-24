# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate durable upload jobs with real local Git repositories and a fake core CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meituan import log_sync

FAKE_CLI = """import os,sys,time
from pathlib import Path
if "--version" in sys.argv:
    print("wandb, version 0.31.0"); raise SystemExit
if "--help" in sys.argv:
    print("--verbose --yes --skip-synced"); raise SystemExit
print("core progress without newline",end="",flush=True)
time.sleep(float(os.environ.get("FAKE_SLEEP","0")))
failed=False
for value in sys.argv[1:]:
    path=Path(value)
    if not path.is_dir() or not path.name.startswith("offline-run-"): continue
    file=next(path.glob("*.wandb"))
    (path/"sdk-attempt.txt").write_text("cached state")
    (path/"files/config.yaml").write_text("SDK modified the copy")
    if "z-bad" in path.name and os.environ.get("FAKE_FAIL")=="1":
        print("fake request failed",file=sys.stderr);failed=True;continue
    if os.environ.get("FAKE_INCOMPLETE")=="1":
        print(f'Failed to sync all data from "{path.name}": the file ends with an incomplete record')
        continue
    if os.environ.get("FAKE_NO_MARKER")!="1":
        file.with_name(file.name+".synced").write_text("")
print("done")
raise SystemExit(7 if failed else 0)
"""


class QueueTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.repo = self.base / "logs"
        self.repo.mkdir()
        self.destination = self.repo / "MirrorOPD-verl/wandb"
        self.destination.mkdir(parents=True)
        self.env = patch.dict(
            os.environ,
            {
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
                "WANDB_BASE_URL": "https://api.wandb.ai",
                "MIRROROPD_UPLOAD_PROJECT": "",
                "MIRROROPD_UPLOAD_ENTITY": "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        log_sync.git(self.repo, "init", "-b", "main")
        (self.repo / ".gitignore").write_text("*.wandb.synced\n*.wandb.syncstate\n")
        log_sync.git(self.repo, "add", ".gitignore")
        log_sync.git(self.repo, "commit", "-m", "initial")
        self.cli = self.base / "wandb"
        self.cli.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
        self.cli.chmod(0o755)
        setting = patch.dict(os.environ, MIRROROPD_WANDB_BIN=str(self.cli))
        setting.start()
        self.addCleanup(setting.stop)

    def write_run(self, name="offline-run-a-good", payload=b"data"):
        run = self.destination / name
        (run / "files").mkdir(parents=True, exist_ok=True)
        (run / "run.wandb").write_bytes(payload)
        (run / "files/config.yaml").write_text("raw config")
        return run

    def enqueue(self):
        log_sync.enqueue(self.repo, self.destination, log_sync.commit_logs(self.repo, self.destination))

    def jobs(self):
        return log_sync.load(log_sync.queue_dir(self.repo))["jobs"]

    def test_success_is_persistent_and_raw_logs_are_immutable(self):
        raw = self.write_run()
        self.enqueue()
        log_sync.upload(self.repo)
        job = self.jobs()[0]
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual((raw / "files/config.yaml").read_text(), "raw config")
        self.assertEqual((Path(job["cache"]) / "files/config.yaml").read_text(), "SDK modified the copy")
        self.enqueue()
        with patch.object(log_sync, "sdk_command", side_effect=AssertionError("must skip completed snapshot")):
            log_sync.upload(self.repo)
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(self.jobs()[0]["attempts"], 1)

    def test_new_snapshot_queues_after_success(self):
        self.write_run()
        self.enqueue()
        log_sync.upload(self.repo)
        self.write_run(payload=b"appended records")
        self.enqueue()
        self.assertEqual([j["status"] for j in self.jobs()], ["succeeded", "pending"])
        self.assertNotEqual(self.jobs()[0]["tree"], self.jobs()[1]["tree"])

    def test_failed_task_survives_a_new_archive_and_reuses_cache(self):
        self.write_run("offline-run-z-bad")
        self.enqueue()
        with patch.dict(os.environ, FAKE_FAIL="1"), self.assertRaisesRegex(ValueError, "incomplete"):
            log_sync.upload(self.repo)
        failed = self.jobs()[0]
        cache = Path(failed["cache"])
        self.assertTrue((cache / "sdk-attempt.txt").exists())
        self.write_run()
        self.enqueue()
        self.assertEqual([j["status"] for j in self.jobs()], ["failed", "pending"])
        log_sync.upload(self.repo, retry=True)
        self.assertEqual([j["status"] for j in self.jobs()], ["succeeded", "succeeded"])
        self.assertEqual(self.jobs()[0]["cache"], str(cache))

    def test_partial_batch_success_and_only_failed_task_retries(self):
        self.write_run()
        self.write_run("offline-run-z-bad")
        self.enqueue()
        with patch.dict(os.environ, FAKE_FAIL="1"), self.assertRaises(ValueError):
            log_sync.upload(self.repo, concurrency=2)
        self.assertEqual([j["status"] for j in self.jobs()], ["succeeded", "failed"])
        log_sync.upload(self.repo, retry=True)
        self.assertEqual([j["attempts"] for j in self.jobs()], [1, 2])

    def test_exit_zero_without_marker_is_not_success(self):
        self.write_run()
        self.enqueue()
        with patch.dict(os.environ, FAKE_NO_MARKER="1"), self.assertRaises(ValueError):
            log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["status"], "failed")
        self.assertIn("completion marker", self.jobs()[0]["error"])

    def test_incomplete_record_waits_for_new_snapshot(self):
        self.write_run()
        self.enqueue()
        with patch.dict(os.environ, FAKE_INCOMPLETE="1"):
            log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["status"], "incomplete")
        with patch.object(
            log_sync, "sdk_command", side_effect=AssertionError("do not retry an immutable partial file")
        ):
            log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["attempts"], 1)
        self.write_run(payload=b"complete snapshot")
        self.enqueue()
        self.assertEqual([j["status"] for j in self.jobs()], ["superseded", "pending"])
        log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[-1]["status"], "succeeded")

    def test_timeout_retains_retryable_snapshot(self):
        self.write_run()
        self.enqueue()
        with patch.dict(os.environ, FAKE_SLEEP="10"), self.assertRaises(ValueError):
            log_sync.upload(self.repo, timeout=0.2)
        job = self.jobs()[0]
        self.assertEqual(job["status"], "failed")
        self.assertTrue(Path(job["cache"]).is_dir())
        self.assertIn("timed out", job["error"])

    def test_incomplete_error_is_attributed_to_only_its_run(self):
        self.write_run()
        self.write_run("offline-run-z-bad")
        self.enqueue()
        with patch.dict(os.environ, FAKE_INCOMPLETE="1", FAKE_FAIL="1"), self.assertRaises(ValueError):
            log_sync.upload(self.repo, concurrency=1)
        self.assertEqual([j["status"] for j in self.jobs()], ["incomplete", "failed"])

    def test_batch_failure_is_not_hidden_by_incomplete_warning(self):
        self.write_run()
        self.enqueue()
        self.cli.write_text(self.cli.read_text().replace("raise SystemExit(7 if failed else 0)", "raise SystemExit(7)"))
        with patch.dict(os.environ, FAKE_INCOMPLETE="1"), self.assertRaisesRegex(ValueError, "SDK batch failed"):
            log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["status"], "failed")

    def test_superseded_snapshot_is_not_uploaded(self):
        self.write_run()
        self.enqueue()
        self.write_run(payload=b"newer")
        self.enqueue()
        self.assertEqual([j["status"] for j in self.jobs()], ["superseded", "pending"])
        log_sync.upload(self.repo)
        self.assertEqual([j["attempts"] for j in self.jobs()], [0, 1])

    def test_worker_lock_prevents_competing_uploaders(self):
        self.write_run()
        self.enqueue()
        with log_sync.lock(log_sync.queue_dir(self.repo), "worker.lock"):
            with self.assertRaisesRegex(ValueError, "Another log operation"):
                log_sync.upload(self.repo)

    def test_child_keeps_worker_lock_when_parent_releases_it(self):
        directory = log_sync.queue_dir(self.repo)
        with log_sync.lock(directory, "worker.lock") as descriptor:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], pass_fds=(descriptor,))
        try:
            with self.assertRaisesRegex(ValueError, "Another log operation"):
                with log_sync.lock(directory, "worker.lock", blocking=False):
                    pass
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_old_sdk_is_rejected_before_attempting_upload(self):
        self.write_run()
        self.enqueue()
        self.cli.write_text(self.cli.read_text().replace("version 0.31.0", "version 0.23.1"))
        with self.assertRaisesRegex(ValueError, "sync-log-setup"):
            log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["attempts"], 0)

    def test_recover_interrupted_worker_and_limit_attempts(self):
        self.write_run()
        self.enqueue()
        folder = log_sync.queue_dir(self.repo)
        state = log_sync.load(folder)
        state["jobs"][0].update(status="running", attempts=1)
        log_sync.save(folder, state)
        log_sync.upload(self.repo)
        self.assertEqual(self.jobs()[0]["attempts"], 2)
        state = log_sync.load(folder)
        state["jobs"][0].update(status="failed", attempts=3, retry_after=0)
        log_sync.save(folder, state)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            log_sync.upload(self.repo)

    def test_run_filter_and_cloud_destination_are_separate(self):
        self.write_run()
        self.write_run("offline-run-z-bad")
        self.enqueue()
        log_sync.upload(self.repo, run="a-good")
        self.assertEqual([j["status"] for j in self.jobs()], ["succeeded", "pending"])
        with patch.dict(os.environ, MIRROROPD_UPLOAD_PROJECT="other-project"):
            self.enqueue()
        self.assertEqual(len(self.jobs()), 4)

    def test_unknown_run_filter_does_not_report_success(self):
        self.write_run()
        self.enqueue()
        with self.assertRaisesRegex(ValueError, "No queued run"):
            log_sync.upload(self.repo, run="typo")

    def test_invalid_queue_is_not_overwritten(self):
        folder = log_sync.queue_dir(self.repo)
        folder.mkdir()
        state = folder / "queue.json"
        state.write_text("[]")
        with self.assertRaisesRegex(ValueError, "Invalid queue state"):
            log_sync.upload(self.repo)
        self.assertEqual(state.read_text(), "[]")

    def test_incomplete_snapshot_reports_success_with_waiting_summary(self):
        self.write_run()
        self.enqueue()
        reports = self.base / "reports"
        with patch.dict(os.environ, FAKE_INCOMPLETE="1"):
            with log_sync.observe.operation(reports, "logs-upload"):
                log_sync.upload(self.repo)
        report = json.loads(next(reports.glob("*.json")).read_text())
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["stages"][0]["status"], "waiting")
        self.assertEqual(report["details"]["upload_summary"]["waiting_for_snapshot"], 1)
        self.assertEqual(report["details"]["upload_summary"]["queue_counts"], {"incomplete": 1})
        self.assertNotIn("error", report)
        self.assertNotIn("failed_stage", report)
        self.assertNotIn("Traceback", Path(report["log"]).read_text())

    def test_strict_incomplete_snapshot_reports_partial_with_summary(self):
        self.write_run()
        self.enqueue()
        reports = self.base / "reports"
        with patch.dict(os.environ, FAKE_INCOMPLETE="1"):
            with self.assertRaises(log_sync.observe.PartialResultError):
                with log_sync.observe.operation(reports, "logs-upload"):
                    log_sync.upload(self.repo, strict=True)
        report = json.loads(next(reports.glob("*.json")).read_text())
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["exit_code"], 2)
        self.assertEqual(report["details"]["upload_summary"]["queue_counts"], {"incomplete": 1})
        self.assertNotIn("failed_stage", report)
        with patch.object(log_sync, "sdk_command", side_effect=AssertionError("strict check must not retry")):
            with self.assertRaises(log_sync.observe.PartialResultError):
                log_sync.upload(self.repo, strict=True)
        self.assertEqual(self.jobs()[0]["attempts"], 1)

    def test_nonfinite_timeouts_fail_before_starting_a_worker(self):
        for timeout in (float("nan"), float("inf"), -1):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                log_sync.upload(self.repo, timeout=timeout)


if __name__ == "__main__":
    unittest.main()
