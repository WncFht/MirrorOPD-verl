# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Cross-machine history adoption and read-only transaction framing checks."""

import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from meituan import wandb_preflight as preflight


def frame(data, kind=1):
    return struct.pack("<IHB", zlib.crc32(bytes([kind]) + data) & 0xFFFFFFFF, len(data), kind) + data


class PreflightTest(unittest.TestCase):
    def read(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wandb"
            path.write_bytes(struct.pack("<4sHB", b":W&B", 0xBEE1, 0) + data)
            return list(preflight.records(path))

    def test_fragmented_record_and_block_padding(self):
        first = b"x" * (32768 - 14)
        self.assertEqual(self.read(frame(first, 2) + frame(b"end", 4)), [first + b"end"])
        first = b"x" * (32768 - 14 - 3)
        self.assertEqual(self.read(frame(first) + b"\0" * 3 + frame(b"next")), [first, b"next"])

    def test_incomplete_tail_and_corruption_are_different(self):
        for data in (frame(b"hello")[:-1], frame(b"first", 2), b"short"):
            with self.subTest(data=data), self.assertRaises(preflight.IncompleteRecord):
                self.read(data)
        damaged = bytearray(frame(b"hello"))
        damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.read(damaged)
        with self.assertRaisesRegex(ValueError, "first fragment"):
            self.read(frame(b"end", 4))

    def test_same_last_step_is_not_evidence_of_coverage(self):
        local = {0: {"_step": 0, "acc": 0.5}, 2: {"_step": 2, "loss": 0.2}}
        proof = preflight.compare(local, [{"_step": 2, "loss": 0.2}])
        self.assertFalse(proof["covered"])
        self.assertEqual(proof["missing_rows"], 1)

    def test_finished_run_with_no_history_can_be_verified(self):
        run = Mock()
        run.scan_history.return_value = []
        self.assertTrue(preflight.compare_run({}, run, history_line_count=0)["covered"])

    def test_duplicate_cloud_rows_are_reported_without_replaying(self):
        local = {0: {"_step": 0, "acc": 0.5}, 2: {"_step": 2, "loss": 0.2}}
        proof = preflight.compare(local, [local[0], local[0], local[2]])
        self.assertTrue(proof["covered"])
        self.assertEqual(proof["duplicate_rows"], 1)

    def test_missing_metrics_and_conflicting_duplicates_block(self):
        local = {0: {"_step": 0, "acc": 0.5}}
        for remote in ([{"_step": 0}], [local[0], {"_step": 0, "acc": 0.9}]):
            proof = preflight.compare(local, remote)
            self.assertFalse(proof["covered"])
            self.assertEqual(proof["conflicting_steps"], [0])

    def test_nested_sparse_values_nan_and_runtime(self):
        local = {"nested": {"v": [0.1, True, None]}, "nan": float("nan"), "_runtime": 30}
        self.assertTrue(preflight.matches(local, {"nested": {"v": [0.1000000001, True, None]}, "nan": None}))
        self.assertFalse(preflight.matches({"flag": True}, {"flag": 1}))
        self.assertFalse(preflight.matches({"loss": 0.1}, {"loss": None}))

    def test_duplicate_coordinates_stop_before_expensive_value_queries(self):
        row = {"_step": 0, "_timestamp": 1.0, "loss": 0.5}
        run = Mock()
        run.scan_history.return_value = [row, row]
        result = preflight.compare_run({0: row}, run)
        self.assertFalse(result["covered"])
        self.assertEqual(result["duplicate_rows"], 1)
        self.assertEqual(run.scan_history.call_count, 1)

    def test_sparse_wide_schema_queries_only_observed_steps(self):
        row = {**{f"metric{i}": 0.5 for i in range(300)}, "_step": 4000, "_timestamp": 1.0}
        run = Mock()
        run.scan_history.return_value = [row]
        self.assertTrue(preflight.compare_run({4000: row}, run)["covered"])
        self.assertEqual(run.scan_history.call_args.kwargs["min_step"], 4000)
        self.assertEqual(run.scan_history.call_args.kwargs["max_step"], 4001)

    def test_server_row_count_cannot_be_hidden_by_deduplicated_scan(self):
        row = {"_step": 0, "_timestamp": 1.0, "loss": 0.5}
        run = Mock()
        run.scan_history.return_value = [row]
        proof = preflight.compare_run({0: row}, run, history_line_count=2)
        self.assertFalse(proof["covered"])
        self.assertEqual(proof["unaccounted_rows"], 1)
        self.assertEqual(run.scan_history.call_count, 1)

    def test_query_error_is_not_treated_as_absent_run(self):
        api = Mock()
        api.runs.side_effect = ValueError("permission denied")
        local = {"identity": {"run_id": "test"}, "rows": {}, "partial": False, "exit_code": 0}
        with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=api))}):
            with patch.object(preflight, "snapshot", return_value=local):
                with self.assertRaisesRegex(ValueError, "permission denied"):
                    preflight.check("unused", "entity", "project", "https://api.wandb.ai")

    def test_unfinished_new_run_waits_and_complete_new_run_uploads(self):
        api = Mock()
        api.runs.return_value = []
        local = {"identity": {"run_id": "test"}, "rows": {}, "partial": True, "exit_code": None}
        with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=api))}):
            with patch.object(preflight, "snapshot", return_value=local):
                self.assertEqual(preflight.check("unused", "e", "p", "https://api.wandb.ai")["action"], "incomplete")
                local.update(partial=False, exit_code=1)
                self.assertEqual(preflight.check("unused", "e", "p", "https://api.wandb.ai")["action"], "upload")


if __name__ == "__main__":
    unittest.main()
