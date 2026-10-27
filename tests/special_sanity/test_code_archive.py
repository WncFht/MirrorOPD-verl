# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise offline source exchange with real Git repositories and no network or training dependencies."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("code_archive", ROOT / "meituan/code_archive.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class CodeArchiveTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.sender = self.base / "sender"
        self.receiver = self.base / "receiver"
        self.sender.mkdir()
        env = patch.dict(
            os.environ,
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "Archive Test",
                "GIT_AUTHOR_EMAIL": "archive@example.invalid",
                "GIT_COMMITTER_NAME": "Archive Test",
                "GIT_COMMITTER_EMAIL": "archive@example.invalid",
                "KEEP_ARCHIVE_AFTER_UNPACK": "1",
                "MIRROROPD_CHUNK_SIZE": "10485760",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.git(self.sender, "init", "-q", "-b", "main")
        self.write(self.sender, ".gitignore", "/hope/\n/outputs/\n/wandb/\n/transfer_snap/\n")
        self.write(self.sender, "src/code.py", "version = 1\n")
        self.write(self.sender, "remove.py", "obsolete\n")
        self.write(self.sender, "meituan/packages/deps.zip", os.urandom(2048))
        self.first = self.commit(self.sender)

    def git(self, repo, *args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
        return result.stdout.decode().strip()

    def write(self, repo, name, content):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def commit(self, repo):
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", "test: update source")
        return self.git(repo, "rev-parse", "HEAD")

    def pack(self, repo=None, **kwargs):
        repo = repo or self.sender
        return archive.pack(repo, repo / "transfer_snap", kwargs.pop("target", "peer"), **kwargs)

    def initialize(self):
        bundle = self.pack()[0]
        archive.unpack(self.receiver, bundle, "peer")
        archive.ack(self.sender, "peer")
        return bundle

    def test_cumulative_and_reverse_sync(self):
        self.initialize()
        self.write(self.sender, "src/code.py", "version = 2\n")
        second = self.commit(self.sender)
        skipped = self.pack()[0]
        (self.sender / "remove.py").unlink()
        self.write(self.sender, "renamed.py", "new\n")
        third = self.commit(self.sender)
        delivered = self.pack()[0]
        self.assertFalse(skipped.exists())
        self.assertEqual(archive.ref(self.sender, archive.refs("peer")[1]), self.first)
        archive.unpack(self.receiver, delivered, "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), third)
        self.assertTrue(archive.ancestor(self.receiver, second, third))
        self.assertFalse((self.receiver / "remove.py").exists())
        self.assertEqual((self.receiver / "src/code.py").read_text(), "version = 2\n")
        self.assertEqual(archive.ref(self.receiver, archive.refs("peer")[1]), third)
        archive.ack(self.sender, "peer")
        self.write(self.receiver, "from_dev.py", "reverse\n")
        reverse = self.commit(self.receiver)
        archive.unpack(self.sender, self.pack(self.receiver)[0], "peer")
        archive.ack(self.receiver, "peer")
        self.assertEqual(self.git(self.sender, "rev-parse", "HEAD"), reverse)
        self.assertEqual(archive.ref(self.sender, archive.refs("peer")[1]), reverse)
        self.assertEqual(self.git(self.sender, "status", "--porcelain"), "")

    def test_ack_uses_last_pack_not_later_head(self):
        self.pack()
        self.write(self.sender, "later.py", "later\n")
        later = self.commit(self.sender)
        archive.ack(self.sender, "peer")
        self.assertEqual(archive.ref(self.sender, archive.refs("peer")[1]), self.first)
        self.assertNotEqual(self.first, later)
        archive.ack(self.sender, "peer")  # Idempotent.

    def test_ack_without_pack_fails(self):
        with self.assertRaisesRegex(ValueError, "No successful pack"):
            archive.ack(self.sender, "peer")

    def test_no_change_and_explicit_full(self):
        self.initialize()
        self.assertEqual(self.pack(), [])
        full = self.pack(full=True)[0]
        fresh = self.base / "fresh"
        archive.unpack(fresh, full, "peer")
        self.assertEqual(self.git(fresh, "rev-parse", "HEAD"), self.first)

    def test_failed_pack_keeps_last_pack_and_archive(self):
        previous = self.pack()[0]
        self.write(self.sender, "later.py", "later\n")
        self.commit(self.sender)
        with patch.object(archive, "publish", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.pack()
        self.assertTrue(previous.exists())
        self.assertEqual(archive.ref(self.sender, archive.refs("peer")[0]), self.first)
        self.assertIsNone(archive.ref(self.sender, archive.EXPORT_REF))

    def test_uncommitted_files_are_not_packaged(self):
        self.write(self.sender, "src/code.py", "dirty\n")
        self.write(self.sender, "untracked.py", "untracked\n")
        archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual((self.receiver / "src/code.py").read_text(), "version = 1\n")
        self.assertFalse((self.receiver / "untracked.py").exists())

    def test_source_policy_checks_staged_blob_not_working_file(self):
        path = self.write(self.sender, "large.bin", b"x" * (2 * 1024 * 1024 + 1))
        self.git(self.sender, "add", "large.bin")
        path.write_bytes(b"small working copy")
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            archive.check_source_artifacts(self.sender, staged=True)

    def test_source_policy_rejects_raw_evidence_but_accepts_small_summary(self):
        name = "examples/mirroropd/docs/experiments/demo/data/summary.json"
        self.write(self.sender, name, '{"loss": 0.1}\n')
        self.commit(self.sender)
        archive.check_source_artifacts(self.sender)
        raw = "examples/mirroropd/docs/experiments/demo/data/history.jsonl"
        self.write(self.sender, raw, '{"loss": 0.1}\n')
        self.git(self.sender, "add", raw)
        with self.assertRaisesRegex(ValueError, "raw experiment artifact"):
            archive.check_source_artifacts(self.sender, staged=True)

    def test_pack_rejects_large_blob_even_after_later_deletion(self):
        path = self.write(self.sender, "large.bin", b"x" * (2 * 1024 * 1024 + 1))
        self.commit(self.sender)
        path.unlink()
        self.commit(self.sender)
        archive.check_source_artifacts(self.sender)
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            self.pack()
        self.assertIsNone(archive.ref(self.sender, archive.refs("peer")[0]))

    def test_source_policy_caps_experiment_summary_size(self):
        name = "examples/mirroropd/docs/experiments/demo/data/summary.json"
        self.write(self.sender, name, b"x" * (1024 * 1024 + 1))
        self.commit(self.sender)
        with self.assertRaisesRegex(ValueError, "1 MiB summary"):
            archive.check_source_artifacts(self.sender)

    def test_missing_baseline_is_rejected(self):
        self.initialize()
        self.write(self.sender, "later.py", "later\n")
        self.commit(self.sender)
        delta = self.pack()[0]
        empty = self.base / "empty"
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            archive.unpack(empty, delta, "peer")
        self.assertFalse(empty.exists())
        self.assertTrue(delta.exists())

    def test_dirty_receiver_and_divergence_preserve_head(self):
        self.initialize()
        self.write(self.sender, "src/code.py", "version = 2\n")
        self.commit(self.sender)
        delta = self.pack()[0]
        self.write(self.receiver, "src/code.py", "local edit\n")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            archive.unpack(self.receiver, delta, "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), self.first)
        self.assertEqual(archive.ref(self.receiver, archive.refs("peer")[1]), self.first)
        own = self.commit(self.receiver)
        with self.assertRaisesRegex(ValueError, "diverged"):
            archive.unpack(self.receiver, delta, "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), own)

    def test_repeated_bundle_does_not_claim_a_dirty_checkout_is_synced(self):
        bundle = self.initialize()
        self.write(self.receiver, "src/code.py", "local edit\n")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            archive.unpack(self.receiver, bundle, "peer")
        self.assertEqual((self.receiver / "src/code.py").read_text(), "local edit\n")

    def test_dirty_receiver_report_names_files_and_preserves_peer_baseline(self):
        bundle = self.initialize()
        self.write(self.receiver, "src/code.py", "local edit\n")
        reports = self.base / "operations"
        with self.assertRaisesRegex(ValueError, "src/code.py"):
            with archive.observe.operation(reports, "archive-unpack"):
                archive.unpack(self.receiver, bundle, "peer")
        report = json.loads(next(reports.glob("*.json")).read_text())
        self.assertEqual(report["failed_stage"], "checkout.update")
        self.assertEqual(report["details"]["head_before"], self.first)
        self.assertEqual(report["details"]["peer_before"], self.first)
        self.assertNotIn("head_after", report["details"])
        self.assertIn("src/code.py", Path(report["log"]).read_text())
        self.assertTrue(bundle.exists())

    def test_ignored_local_file_is_not_overwritten(self):
        self.initialize()
        self.write(self.sender, "outputs/new.py", "tracked\n")
        self.git(self.sender, "add", "-f", "outputs/new.py")
        self.commit(self.sender)
        self.write(self.receiver, "outputs/new.py", "local run\n")
        with self.assertRaisesRegex(ValueError, "overwrite a local"):
            archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual((self.receiver / "outputs/new.py").read_text(), "local run\n")

    def test_first_install_preserves_legacy_files_and_runtime(self):
        self.write(self.receiver, "src/code.py", "legacy change\n")
        self.write(self.receiver, "hope/mirroropd/job.hope", "local config\n")
        self.write(self.receiver, "wandb/offline-run/log", "local log\n")
        self.write(self.receiver, "outputs/run/result", "local result\n")
        archive.unpack(self.receiver, self.pack()[0], "peer")
        backup = next(self.base.glob(".receiver-before-bundle-*"))
        self.assertEqual((backup / "src/code.py").read_text(), "legacy change\n")
        self.assertEqual((self.receiver / "hope/mirroropd/job.hope").read_text(), "local config\n")
        self.assertEqual((self.receiver / "wandb/offline-run/log").read_text(), "local log\n")
        self.assertEqual((self.receiver / "outputs/run/result").read_text(), "local result\n")
        self.assertEqual(self.git(self.receiver, "status", "--porcelain"), "")

    def test_tracked_hope_migration_keeps_receiver_edits(self):
        self.write(self.sender, "hope/job.hope", "old tracked config\n")
        self.git(self.sender, "add", "-f", "hope/job.hope")
        self.commit(self.sender)
        self.git(self.base, "clone", "-q", str(self.sender), str(self.receiver))
        self.write(self.receiver, "hope/job.hope", "machine-specific edit\n")
        self.write(self.receiver, "hope/new.hope", "extra local config\n")
        self.git(self.sender, "rm", "-r", "--cached", "hope")
        removed = self.commit(self.sender)
        archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), removed)
        self.assertEqual(self.git(self.receiver, "ls-files", "hope"), "")
        self.assertEqual((self.receiver / "hope/job.hope").read_text(), "machine-specific edit\n")
        self.assertTrue((self.receiver / "hope/new.hope").exists())

    def test_pack_rejects_tracked_hope(self):
        self.write(self.sender, "hope/job.hope", "config\n")
        self.git(self.sender, "add", "-f", "hope/job.hope")
        self.commit(self.sender)
        with self.assertRaisesRegex(ValueError, "still tracks hope"):
            self.pack()

    def test_failed_hope_migration_restores_local_files(self):
        self.write(self.sender, "hope/job.hope", "tracked config\n")
        self.git(self.sender, "add", "-f", "hope/job.hope")
        old = self.commit(self.sender)
        self.git(self.base, "clone", "-q", str(self.sender), str(self.receiver))
        self.write(self.receiver, "hope/job.hope", "local settings\n")
        self.git(self.sender, "rm", "-r", "--cached", "hope")
        self.commit(self.sender)
        bundle = self.pack()[0]
        original_git = archive.git

        def fail_merge(repo, *args, **kwargs):
            if args[0] == "merge":
                raise ValueError("merge interrupted")
            return original_git(repo, *args, **kwargs)

        with patch.object(archive, "git", side_effect=fail_merge):
            with self.assertRaisesRegex(ValueError, "merge interrupted"):
                archive.unpack(self.receiver, bundle, "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), old)
        self.assertEqual((self.receiver / "hope/job.hope").read_text(), "local settings\n")
        self.assertIsNone(archive.ref(self.receiver, archive.refs("peer")[1]))
        self.assertTrue(bundle.exists())

    def test_interrupted_hope_backup_requires_recovery(self):
        self.initialize()
        self.write(self.sender, "later.py", "later\n")
        self.commit(self.sender)
        self.write(self.receiver, ".git/archive-hope-backup/job.hope", "recover me\n")
        with self.assertRaisesRegex(ValueError, "interrupted migration"):
            archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), self.first)

    def test_modes_symlinks_and_path_type_changes(self):
        executable = self.write(self.sender, "bin/entry.sh", "#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        (self.sender / "entry").symlink_to("bin/entry.sh")
        self.write(self.sender, "space in name.py", "first\n")
        self.commit(self.sender)
        self.initialize()
        self.assertEqual((self.receiver / "bin/entry.sh").stat().st_mode & 0o111, 0o111)
        self.assertEqual(os.readlink(self.receiver / "entry"), "bin/entry.sh")
        (self.sender / "space in name.py").unlink()
        self.write(self.sender, "space in name.py/nested.py", "now a directory\n")
        self.commit(self.sender)
        archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual((self.receiver / "space in name.py/nested.py").read_text(), "now a directory\n")

    def test_bootstrap_parent_symlink_is_rejected_before_writes(self):
        outside = self.base / "outside"
        outside.mkdir()
        self.receiver.mkdir()
        (self.receiver / "src").symlink_to(outside, target_is_directory=True)
        self.write(self.receiver, ".gitignore", "keep this\n")
        with self.assertRaisesRegex(ValueError, "Local parent obstructs"):
            archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual((self.receiver / ".gitignore").read_text(), "keep this\n")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.receiver / ".git").exists())

    def test_gitlink_keeps_local_submodule_directory(self):
        self.git(self.sender, "update-index", "--add", "--cacheinfo", f"160000,{self.first},recipe")
        self.git(self.sender, "commit", "-qm", "test: add submodule reference")
        self.write(self.receiver, "recipe/local.py", "keep\n")
        archive.unpack(self.receiver, self.pack()[0], "peer")
        self.assertEqual((self.receiver / "recipe/local.py").read_text(), "keep\n")
        self.assertTrue(self.git(self.receiver, "ls-tree", "HEAD", "recipe").startswith("160000"))

    def test_split_archive_and_incomplete_parts(self):
        with patch.dict(os.environ, MIRROROPD_CHUNK_SIZE="256"):
            parts = self.pack()
        self.assertGreater(len(parts), 2)
        original = parts[1].read_bytes()
        parts[1].unlink()
        with self.assertRaisesRegex(ValueError, "Missing archive part"):
            archive.unpack(self.receiver, parts[0], "peer")
        self.assertFalse(self.receiver.exists())
        parts[1].write_bytes(original)
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            archive.unpack(self.receiver, parts[0], "peer")
        self.assertTrue(all(not part.exists() for part in parts))

    def test_missing_last_part_does_not_initialize_receiver(self):
        with patch.dict(os.environ, MIRROROPD_CHUNK_SIZE="256"):
            parts = self.pack()
        parts[-1].unlink()
        with self.assertRaises(ValueError):
            archive.unpack(self.receiver, parts[0], "peer")
        self.assertFalse(self.receiver.exists())

    def test_peer_state_and_cleanup_are_separate(self):
        other = self.pack(target="peer_other")[0]
        self.pack()
        self.pack()
        self.assertTrue(other.exists())
        archive.ack(self.sender, "peer")
        self.assertIsNone(archive.ref(self.sender, archive.refs("peer_other")[1]))

    def test_unpack_retires_older_legacy_source_parts_only(self):
        downloads = self.base / "downloads"
        downloads.mkdir()
        source = downloads / "archive_MirrorOPD-verl_peer_20261023_151924_520190.bundle"
        shutil.copy2(self.pack()[0], source)
        old = [
            self.write(downloads, f"archive_MirrorOPD-verl_20261022_220146_190191.tar.gz.part.{i:03d}", "old")
            for i in range(3)
        ]
        old.append(self.write(downloads, "archive_MirrorOPD-verl_20261021_220146_190191.tar.gz", "old"))
        preserved = [
            self.write(downloads, name, "keep")
            for name in (
                "archive_MirrorOPD-verl_20261024_220146_190191.tar.gz.part.000",
                "archive_MirrorOPD-verl_custom.tar.gz",
                "archive_MirrorOPD-verl_peer_20261021_220146_190191.bundle",
                "log_MirrorOPD-verl_20261021_220146_190191.tar.gz",
                "MirrorOPD_results_20261021_220146_190191.tar.gz",
                "archive_bootstrap.py",
            )
        ]
        # Upload mtimes do not describe package generation order.
        for path in old:
            os.utime(path, (2200000000, 2200000000))
        elsewhere = self.write(
            self.base / "unrelated", "archive_MirrorOPD-verl_20261021_220146_190191.tar.gz", "elsewhere"
        )
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            archive.unpack(self.receiver, source, "peer")
        self.assertTrue(all(not path.exists() for path in [source, *old]))
        self.assertTrue(all(path.exists() for path in [elsewhere, *preserved]))

    def test_legacy_cleanup_respects_keep_and_failed_updates(self):
        source = self.pack()[0]
        old = self.write(source.parent, "archive_MirrorOPD-verl_20000101_000000_000000.tar.gz", "old")
        archive.unpack(self.receiver, source, "peer")
        self.assertTrue(source.exists())
        self.assertTrue(old.exists())
        self.write(self.receiver, "src/code.py", "local change\n")
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                archive.unpack(self.receiver, source, "peer")
        self.assertTrue(source.exists())
        self.assertTrue(old.exists())

    def test_custom_bundle_does_not_guess_legacy_archive_age(self):
        source = self.pack(supplied="custom.bundle")[0]
        old = self.write(source.parent, "archive_MirrorOPD-verl_20000101_000000_000000.tar.gz", "old")
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            archive.unpack(self.receiver, source, "peer")
        self.assertFalse(source.exists())
        self.assertTrue(old.exists())

    def test_cleanup_covers_checkout_and_snap_formats_without_deleting_tracked_files(self):
        tracked = "archive_MirrorOPD-verl_20000101_000000.zip"
        self.write(self.sender, tracked, "tracked content")
        self.commit(self.sender)
        source = self.pack()[0]
        old = [
            self.write(self.receiver, "archive_MirrorOPD-verl_20000101_010000.tar.gz.part.000", "old"),
            self.write(self.receiver / "transfer_snap", "archive_MirrorOPD-verl_peer_20000101_010000.zip", "old"),
            self.write(self.receiver / "transfer_snap", "archive_MirrorOPD-verl_20000101_010000.tgz", "old"),
        ]
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            archive.unpack(self.receiver, source, "peer")
        self.assertTrue(all(not path.exists() for path in old))
        self.assertEqual((self.receiver / tracked).read_text(), "tracked content")
        self.assertEqual(self.git(self.receiver, "status", "--porcelain"), "")

    def test_custom_bundle_cleanup_uses_conservative_age_and_keeps_symlinks(self):
        source = self.pack(supplied="custom.bundle")[0]
        old = self.write(source.parent, "archive_MirrorOPD-verl_20000101_000000.tar.gz", "old")
        os.utime(old, (1, 1))
        target = self.write(self.base, "retained.bin", "keep")
        link = source.parent / "archive_MirrorOPD-verl_20000101_000001.tar.gz"
        link.symlink_to(target)
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="0"):
            archive.unpack(self.receiver, source, "peer")
        self.assertFalse(old.exists())
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(), "keep")

    def test_pack_keeps_future_dated_delivery_and_dangling_link(self):
        snap = self.sender / "transfer_snap"
        future = self.write(snap, "archive_MirrorOPD-verl_peer_20990101_000000_000000.bundle", "future")
        link = snap / "archive_MirrorOPD-verl_peer_20000101_000000_000000.bundle"
        link.symlink_to(self.base / "missing")
        self.pack()
        self.assertTrue(future.exists())
        self.assertTrue(link.is_symlink())

    def test_old_bundle_does_not_rewind_checkout_or_peer(self):
        first_bundle = self.initialize()
        saved = self.base / "first.bundle"
        shutil.copy2(first_bundle, saved)
        self.write(self.sender, "next.py", "next\n")
        newer = self.commit(self.sender)
        archive.unpack(self.receiver, self.pack()[0], "peer")
        archive.unpack(self.receiver, saved, "peer")
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), newer)
        self.assertEqual(archive.ref(self.receiver, archive.refs("peer")[1]), newer)

    def test_standalone_bootstrap_and_make_entrypoints(self):
        for name in ("Makefile", "meituan/archive.sh", "meituan/code_archive.py", "meituan/sync_observe.py"):
            self.write(self.sender, name, (ROOT / name).read_bytes())
        self.commit(self.sender)
        subprocess.run(["make", "archive-pack", f"PYTHON={sys.executable}"], cwd=self.sender, check=True)
        snap = self.sender / "transfer_snap"
        bundle = next(snap.glob("*.bundle"))
        subprocess.run(
            [sys.executable, str(snap / "archive_bootstrap.py"), "unpack", str(bundle), str(self.receiver)], check=True
        )
        subprocess.run(["make", "archive-ack", f"PYTHON={sys.executable}"], cwd=self.sender, check=True)
        self.assertEqual(self.git(self.receiver, "rev-parse", "HEAD"), self.git(self.sender, "rev-parse", "HEAD"))

    def test_standalone_requires_destination_before_touching_parent_or_input(self):
        bundle = self.pack()[0]
        source = self.sender / "transfer_snap/archive_bootstrap.py"
        script = self.base / "uploaded-project/archive_bootstrap.py"
        script.parent.mkdir()
        shutil.copy2(source, script)
        sentinel = self.write(self.base, "model/keep.txt", "local model asset\n")
        for arguments in (["unpack"], ["unpack", str(bundle)], ["pack"], ["ack"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, str(script), *arguments], cwd=script.parent, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("unpack ARCHIVE DESTINATION", result.stderr)
                self.assertTrue(bundle.exists())
                self.assertFalse((self.base / ".git").exists())
                self.assertFalse((self.base / "transfer_snap").exists())
                self.assertFalse((script.parent / ".git").exists())
                self.assertEqual(sentinel.read_text(), "local model asset\n")

    def test_standalone_relative_destination_uses_callers_directory(self):
        bundle = self.pack()[0]
        script = self.sender / "transfer_snap/archive_bootstrap.py"
        caller = self.base / "caller"
        caller.mkdir()
        subprocess.run([sys.executable, str(script), "unpack", str(bundle), "fresh"], cwd=caller, check=True)
        self.assertEqual(self.git(caller / "fresh", "rev-parse", "HEAD"), self.first)
        self.assertFalse((self.sender / "fresh").exists())


if __name__ == "__main__":
    unittest.main()
