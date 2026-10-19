# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise source-archive cleanup without GPUs, package installation or network access."""

import argparse
import importlib.util
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("meituan_transfer", ROOT / "meituan/transfer.py")
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)


class SourceArchiveCleanupTest(unittest.TestCase):
    """Keep cleanup scoped to complete source snapshots and successfully consumed inputs."""

    def setUp(self):
        """Create an isolated checkout, receiving directory and archive destination."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.repo = self.base / "repo"
        (self.repo / "verl").mkdir(parents=True)
        (self.repo / "verl/payload.bin").write_bytes(os.urandom(4096))
        self.snap = self.repo / "transfer_snap"
        self.receiving = self.base / "receiving"
        self.downloads = self.base / "downloads"
        self.downloads.mkdir()
        root = patch.object(transfer, "ROOT", self.repo)
        root.start()
        self.addCleanup(root.stop)
        env = patch.dict(os.environ, {"KEEP_ARCHIVE_AFTER_UNPACK": "0", "MIRROROPD_CHUNK_SIZE": "10485760"})
        env.start()
        self.addCleanup(env.stop)

    def pack(self, name):
        """Create a full source snapshot using the real pack implementation."""
        transfer.pack(argparse.Namespace(kind="code", archive=name, paths=[]), self.snap)
        return self.snap / name

    def unpack(self, source):
        """Apply a downloaded source snapshot using the real unpack implementation."""
        transfer.unpack(argparse.Namespace(kind="code", archive=str(source), paths=[str(self.receiving)]), self.snap)

    def test_pack_removes_old_source_only(self):
        """Retain the new code archive, unrelated transfer types and dependency ZIPs."""
        old = self.pack("archive_MirrorOPD-verl_old.tar.gz")
        unrelated = [self.snap / "log_MirrorOPD-verl_old.tar.gz", self.snap / "dependencies.zip"]
        for path in unrelated:
            path.write_bytes(b"keep")
        current = self.pack("archive_MirrorOPD-verl_current.tar.gz")
        self.assertFalse(old.exists())
        self.assertTrue(current.exists())
        self.assertTrue(all(path.exists() for path in unrelated))

    def test_pack_excludes_hope_with_or_without_git(self):
        """Tracked Hope files and files found by fallback scanning stay local."""
        (self.repo / "hope").mkdir()
        (self.repo / "hope/local.hope").write_text("local job settings")
        archive = self.pack("archive_MirrorOPD-verl_current.tar.gz")
        with tarfile.open(archive) as bundle:
            self.assertEqual(bundle.getnames(), ["verl/payload.bin"])
        (self.repo / ".git").mkdir()
        with patch.object(transfer.subprocess, "check_output", return_value=b"hope/local.hope\0verl/payload.bin\0"):
            self.assertEqual([member for _, member in transfer.source_files([])], ["verl/payload.bin"])

    def test_unpack_legacy_archive_keeps_local_hope(self):
        """Even a previous delivery containing Hope files cannot replace local job settings."""
        (self.repo / "hope").mkdir()
        (self.repo / "hope/local.hope").write_text("incoming job settings")
        (self.repo / "hope/new.hope").write_text("incoming new job")
        (self.receiving / "hope").mkdir(parents=True)
        original = self.receiving / "hope/local.hope"
        original.write_text("machine-specific job settings")
        archive = self.downloads / "archive_MirrorOPD-verl_legacy.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(self.repo / "hope", arcname="hope")
            bundle.add(self.repo / "verl", arcname="verl")
        self.unpack(archive)
        self.assertEqual(original.read_text(), "machine-specific job settings")
        self.assertFalse((self.receiving / "hope/new.hope").exists())
        self.assertEqual(
            (self.receiving / "verl/payload.bin").read_bytes(), (self.repo / "verl/payload.bin").read_bytes()
        )

    def test_same_name_can_switch_from_parts_to_single_archive(self):
        """Publishing a smaller replacement removes obsolete same-name parts."""
        name = "custom.tar.gz"
        with patch.dict(os.environ, MIRROROPD_CHUNK_SIZE="1024"):
            self.pack(name)
        self.assertGreater(len(list(self.snap.glob(name + ".part.*"))), 1)
        current = self.pack(name)
        self.assertTrue(current.is_file())
        self.assertEqual(list(self.snap.glob(name + ".part.*")), [])

    def test_failed_pack_keeps_previous_archive(self):
        """A tar write failure must not delete the previous usable snapshot."""
        old = self.pack("archive_MirrorOPD-verl_old.tar.gz")
        expected = old.read_bytes()
        with patch.object(tarfile.TarFile, "add", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.pack("archive_MirrorOPD-verl_current.tar.gz")
        self.assertEqual(old.read_bytes(), expected)

    def test_unpack_deletes_consumed_and_older_but_keeps_newer(self):
        """Cleanup in the download directory never removes a newer source delivery."""
        current = self.pack("archive_MirrorOPD-verl_current.tar.gz")
        paths = []
        for index, name in enumerate(("old", "current", "newer"), start=1):
            path = self.downloads / f"archive_MirrorOPD-verl_{name}.tar.gz"
            shutil.copy2(current, path)
            os.utime(path, ns=(index * 1000000000, index * 1000000000))
            paths.append(path)
        self.unpack(paths[1])
        self.assertFalse(paths[0].exists())
        self.assertFalse(paths[1].exists())
        self.assertTrue(paths[2].exists())
        self.assertEqual(
            (self.receiving / "verl/payload.bin").read_bytes(), (self.repo / "verl/payload.bin").read_bytes()
        )

    def test_keep_flag_retains_consumed_archive(self):
        """The explicit keep flag preserves a successfully unpacked input."""
        current = self.pack("archive_MirrorOPD-verl_current.tar.gz")
        with patch.dict(os.environ, KEEP_ARCHIVE_AFTER_UNPACK="1"):
            self.unpack(current)
        self.assertTrue(current.exists())

    def test_successful_split_unpack_deletes_every_consumed_part(self):
        """Delete all consumed pieces only after a complete split archive is extracted."""
        name = "archive_MirrorOPD-verl_split.tar.gz"
        with patch.dict(os.environ, MIRROROPD_CHUNK_SIZE="1024"):
            self.pack(name)
        parts = sorted(self.snap.glob(name + ".part.*"))
        self.unpack(parts[0])
        self.assertTrue((self.receiving / "verl/payload.bin").exists())
        self.assertTrue(all(not part.exists() for part in parts))

    def test_incomplete_split_unpack_keeps_remaining_parts(self):
        """An interrupted upload must retain every piece that did arrive."""
        name = "archive_MirrorOPD-verl_split.tar.gz"
        with patch.dict(os.environ, MIRROROPD_CHUNK_SIZE="1024"):
            self.pack(name)
        parts = sorted(self.snap.glob(name + ".part.*"))
        parts[1].unlink()
        with self.assertRaisesRegex(ValueError, "Missing archive part"):
            self.unpack(parts[0])
        self.assertTrue(all(part.exists() for part in parts if part != parts[1]))
        self.assertFalse(self.receiving.exists())


if __name__ == "__main__":
    unittest.main()
