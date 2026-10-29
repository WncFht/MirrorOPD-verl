# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check durable partial-success reports with local Git remotes and a fake W&B CLI."""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from meituan import sync_observe as observe
from meituan import sync_status

ROOT = Path(__file__).resolve().parents[2]


class JournalTest(unittest.TestCase):
    def test_idle_timeout_keeps_tail_and_ignores_repeated_output(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(subprocess.TimeoutExpired) as caught:
            observe.run(
                [
                    sys.executable,
                    "-c",
                    "import time\nfor _ in range(100):\n print('same progress',flush=True); time.sleep(.03)",
                ],
                stream=True,
                check=True,
                timeout=8,
                idle_timeout=1.0,
            )
        self.assertIn("same progress", caught.exception.output)
        self.assertIn("idle timeout", caught.exception.stderr)

    def test_color_is_console_only_and_no_color_is_honored(self):
        with tempfile.TemporaryDirectory() as temp:
            console = io.StringIO()
            env = {key: value for key, value in os.environ.items() if key != "NO_COLOR"}
            env["MIRROROPD_COLOR"] = "always"
            with patch.dict(os.environ, env, clear=True), redirect_stdout(console):
                with observe.operation(Path(temp), "color-test"):
                    with observe.stage("sample"):
                        print("\x1b[31mSDK output\x1b[0m")
            self.assertIn("\x1b[", console.getvalue())
            self.assertNotIn("\x1b", next(Path(temp).glob("*.log")).read_text())
            report = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertEqual(report["status"], "success")
            self.assertNotIn("\x1b", json.dumps(report))
            with patch.dict(os.environ, NO_COLOR="1", MIRROROPD_COLOR="always"):
                self.assertEqual(observe.paint("SUCCESS", console), "SUCCESS")

    def test_auto_color_disables_escape_codes_in_pipes(self):
        with patch.dict(os.environ, MIRROROPD_COLOR="auto"):
            self.assertEqual(observe.paint("FAILED", io.StringIO()), "FAILED")

    def test_streaming_timeout_stops_a_child_without_newlines(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                with observe.operation(Path(temp), "deadline-test"):
                    observe.run(
                        [
                            sys.executable,
                            "-c",
                            "import time; print('partial progress',end='',flush=True); time.sleep(20)",
                        ],
                        stream=True,
                        timeout=0.3,
                        check=True,
                    )
            report = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("partial progress", Path(report["log"]).read_text())

    def test_failed_command_keeps_output_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            console = io.StringIO()
            secret = "test-private-key-12345"
            with patch.dict(os.environ, WANDB_API_KEY=secret), redirect_stdout(console):
                with self.assertRaises(subprocess.CalledProcessError):
                    with observe.operation(directory, "test-failure", archive="sample.tar.gz"):
                        with observe.stage("upload", hint="retry the same archive"):
                            observe.run(
                                [
                                    sys.executable,
                                    "-c",
                                    "import os,sys; print(os.environ['WANDB_API_KEY']); "
                                    "print('token=private-value', file=sys.stderr); sys.exit(3)",
                                ],
                                stream=True,
                                check=True,
                            )
            report = json.loads(next(directory.glob("*.json")).read_text())
            log = next(directory.glob("*.log")).read_text()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_stage"], "upload")
            self.assertEqual(report["stages"][0]["hint"], "retry the same archive")
            self.assertIn("exit=3", log)
            self.assertIn("Traceback", log)
            self.assertNotIn(secret, log + console.getvalue() + json.dumps(report))
            self.assertNotIn("private-value", log + console.getvalue() + json.dumps(report))
            self.assertIsNone(observe._active)

    def test_interruption_is_distinct_from_success(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                with observe.operation(Path(temp), "test-interrupt"):
                    with observe.stage("extract"):
                        raise KeyboardInterrupt()
            report = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["stages"][0]["status"], "interrupted")

    def test_status_is_read_only_and_does_not_infer_missing_history(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "not-created"
            self.assertEqual(sync_status.reports(empty, "all"), [])
            self.assertFalse(empty.exists())
            self.assertIsNone(sync_status.code_status(Path(temp), "peer")["confirmed"])
            self.assertFalse((Path(temp) / ".git").exists())

    def test_corrupt_status_files_are_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "logs_state.json").write_text("[]")
            self.assertIn("error", sync_status.logs_status(directory)["packing_state"])
            (directory / "broken.json").write_text("{partial")
            self.assertTrue(all(report["status"] == "unreadable" for report in sync_status.reports(directory, "all")))


class LogObservabilityTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.logs = self.base / "logs-repo"
        self.remote = self.base / "remote.git"
        self.opdir = self.base / "operations"
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "MIRROROPD_OPERATION_DIR": str(self.opdir),
            "MIRROROPD_WANDB_LOGS_REPO": str(self.logs),
            "MIRROROPD_WANDB_DIR": "",
            "MIRROROPD_WANDB_SYNC": "0",
            "MIRROROPD_WANDB_GIT_SYNC": "1",
        }
        self.git(self.base, "init", "--bare", str(self.remote))
        self.git(self.base, "init", "-b", "main", str(self.logs))
        (self.logs / "README").write_text("logs\n")
        self.git(self.logs, "add", "README")
        self.git(self.logs, "commit", "-m", "initial")
        self.git(self.logs, "remote", "add", "origin", str(self.remote))
        self.git(self.logs, "push", "-u", "origin", "main")
        self.initial = self.git(self.logs, "rev-parse", "HEAD")
        binary = self.base / "bin"
        binary.mkdir()
        cli = binary / "wandb"
        cli.write_text(
            f"#!{sys.executable}\n"
            "import os,sys\nfrom pathlib import Path\n"
            "if '--version' in sys.argv: print('wandb, version 0.31.0'); raise SystemExit\n"
            "if '--help' in sys.argv: print('--verbose --yes --skip-synced'); raise SystemExit\n"
            "print('fake stderr diagnostic',file=sys.stderr)\nfailed=False\n"
            "for arg in sys.argv[1:]:\n"
            " p=Path(arg)\n"
            " if not p.is_dir() or not p.name.startswith('offline-run-'): continue\n"
            " if 'z-bad' in p.name and os.environ.get('FAIL_UPLOAD')=='1': failed=True; continue\n"
            " if os.environ.get('INCOMPLETE_UPLOAD')=='1':\n"
            "  print(f'Failed to sync all data from \"{p.name}\": the file ends with an incomplete record'); continue\n"
            " f=next(p.glob('*.wandb')); f.with_name(f.name+'.synced').write_text('')\n"
            "raise SystemExit(7 if failed else 0)\n"
        )
        self.env["MIRROROPD_WANDB_BIN"] = str(cli)
        cli.chmod(0o755)
        helper = binary / "python"
        helper.write_text(
            f"#!{sys.executable}\n"
            "import sys,json\nfrom pathlib import Path\n"
            "f=Path(sys.argv[2]); complete=f.with_name(f.name+'.synced').exists()\n"
            "result={'action':'covered' if complete else 'upload','reason':'test cloud','exit_code':0,"
            "'target':{'base_url':'https://api.wandb.ai','entity':'test','project':'test'}}\n"
            "Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(result))\n"
        )
        helper.chmod(0o755)
        self.env["PATH"] = str(binary) + os.pathsep + os.environ["PATH"]
        self.bundle = self.base / "logs.tar.gz"
        with tarfile.open(self.bundle, "w:gz") as tar:
            for name in ("offline-run-a-good", "offline-run-z-bad"):
                run = self.base / name
                run.mkdir()
                (run / "run.wandb").write_bytes(b"offline data")
                tar.add(run, arcname="wandb/" + name)

    def git(self, repo, *args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=self.env, capture_output=True, text=True, check=True
        ).stdout.strip()

    def unpack(self, **settings):
        result = subprocess.run(
            [sys.executable, str(ROOT / "meituan/transfer.py"), "logs", "unpack", str(self.bundle)],
            env={**self.env, **settings},
            capture_output=True,
            text=True,
        )
        reports = [json.loads(path.read_text()) for path in self.opdir.glob("*.json")]
        report = max(reports, key=lambda item: Path(item["log"]).stat().st_mtime_ns)
        return result, report

    def upload(self, **settings):
        result = subprocess.run(
            [sys.executable, str(ROOT / "meituan/log_sync.py"), "upload"],
            env={**self.env, **settings},
            capture_output=True,
            text=True,
        )
        reports = [json.loads(path.read_text()) for path in self.opdir.glob("*.json")]
        return result, max(reports, key=lambda item: Path(item["log"]).stat().st_mtime_ns)

    def test_upload_cli_fetches_and_uploads_new_remote_run_without_manual_enqueue(self):
        self.assertEqual(self.unpack()[0].returncode, 0)
        other = self.base / "other"
        self.git(self.base, "clone", "-b", "main", str(self.remote), str(other))
        run = other / "MirrorOPD-verl/wandb/offline-run-remote"
        run.mkdir()
        (run / "run-remote.wandb").write_bytes(b"remote snapshot")
        self.git(other, "add", ".")
        self.git(other, "commit", "-m", "remote run")
        self.git(other, "push")
        remote_head = self.git(other, "rev-parse", "HEAD")
        result, report = self.upload()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["details"]["upload_summary"]["queue_counts"], {"succeeded": 3})
        self.assertEqual(report["details"]["git_merge_after"], remote_head)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), remote_head)
        self.assertEqual(self.git(self.logs, "status", "--porcelain"), "")

    def test_waiting_snapshots_exit_zero_and_strict_checks_exit_two(self):
        self.assertEqual(self.unpack()[0].returncode, 0)
        result, report = self.upload(INCOMPLETE_UPLOAD="1", MIRROROPD_UPLOAD_STRICT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "success")
        self.assertIn("WAITING 2", result.stdout)
        self.assertEqual(report["details"]["upload_summary"]["waiting_for_snapshot"], 2)
        result, report = self.upload(MIRROROPD_UPLOAD_STRICT="1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(report["status"], "partial")
        self.assertIn("STRICT=1", result.stderr)
        self.assertEqual(report["details"]["upload_summary"]["attempted_this_call"], 0)
        result, report = self.upload(MIRROROPD_UPLOAD_STRICT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["details"]["upload_summary"]["attempted_this_call"], 0)

    def test_inline_upload_uses_same_waiting_and_strict_policy(self):
        result, report = self.unpack(MIRROROPD_WANDB_SYNC="1", INCOMPLETE_UPLOAD="1", MIRROROPD_UPLOAD_STRICT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "success")
        self.assertTrue(self.bundle.exists())
        result, report = self.unpack(MIRROROPD_WANDB_SYNC="1", MIRROROPD_UPLOAD_STRICT="1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(self.bundle.exists())

    def test_partial_upload_failure_is_durable_and_retry_records_success(self):
        self.assertEqual(self.unpack()[0].returncode, 0)
        committed = self.git(self.logs, "rev-parse", "HEAD")
        result, report = self.upload(FAIL_UPLOAD="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["failed_stage"], "wandb.batch")
        queue = json.loads((self.logs / ".git/mirroropd-upload/queue.json").read_text())
        self.assertEqual([job["status"] for job in queue["jobs"]], ["succeeded", "failed"])
        self.assertIn("fake stderr diagnostic", Path(report["log"]).read_text())
        self.assertNotEqual(committed, self.initial)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), committed)
        self.assertTrue(self.bundle.exists())
        result, report = self.upload(MIRROROPD_UPLOAD_RETRY="1")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(report["status"], "success")
        self.assertEqual(len(list(self.opdir.glob("*.json"))), 3)

    def test_push_failure_records_commit_and_disabled_upload_is_not_success(self):
        hook = self.remote / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        result, report = self.unpack()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["failed_stage"], "logs.git_push")
        committed = self.git(self.logs, "rev-parse", "HEAD")
        self.assertEqual(report["details"]["logs_commit"], committed)
        self.assertNotEqual(committed, self.initial)
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), self.initial)
        hook.unlink()
        result, report = self.unpack(MIRROROPD_WANDB_SYNC="0")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(self.git(self.logs, "rev-parse", "HEAD"), committed)
        self.assertTrue(
            any(step["name"] == "wandb.upload" and step["status"] == "skipped" for step in report["stages"])
        )
        self.assertEqual(self.git(self.remote, "rev-parse", "main"), committed)

    def test_status_json_exposes_failure_without_contacting_remote(self):
        self.unpack()
        self.upload(FAIL_UPLOAD="1")
        result = subprocess.run(
            [sys.executable, str(ROOT / "meituan/sync_status.py"), "--section", "logs", "--json"],
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )
        status = json.loads(result.stdout)
        self.assertEqual(status["operations"][0]["status"], "failed")
        self.assertIn("no live cloud query", status["logs"]["cloud_status"])
        self.assertEqual(status["logs"]["destination_runs"], 2)


if __name__ == "__main__":
    unittest.main()
