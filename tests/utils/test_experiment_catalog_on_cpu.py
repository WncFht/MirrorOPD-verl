# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Evidence boundaries for the local experiment catalog; no model dependencies."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples/mirroropd/research/analysis"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.mirroropd.research.analysis import refresh_tables as catalog  # noqa: E402


class CatalogEvidenceTests(unittest.TestCase):
    def test_precision_run_is_not_a_generic_method_override(self):
        result = catalog.classify({"run_id": "new", "display_name": "math-opd-E51-mdpo-full_vocab-test"})
        self.assertEqual(result["primary_question"], "teacher-transfer")

    def test_nonstandard_qwen_name_uses_explicit_identity(self):
        result = catalog.classify({"run_id": "53dd553d", "display_name": "local-name"})
        self.assertEqual((result["experiment"], result["primary_question"]), ("E41", "teacher-transfer"))

    def test_unknown_preset_requires_question_registration(self):
        with self.assertRaisesRegex(ValueError, "E99"):
            catalog.classify({"run_id": "new", "display_name": "math-opd-E99-test"})
        self.assertIsNone(catalog.classify({"run_id": "smoke", "display_name": "engineering-smoke"}))

    def test_dataset_coordinates_do_not_follow_later_training(self):
        rows = [
            {"step/outer": 32, "step/optimizer": 512, "eval/aime25/acc/mean": 0.3},
            {"step/outer": 64, "step/optimizer": 1020, "eval/math500/acc/mean": 0.8},
            {"step/outer": 70, "step/optimizer": 1116},
        ]
        result = catalog.last_evaluations(rows)
        self.assertEqual(result["aime25"]["outer"], 32)
        self.assertEqual(result["aime25"]["optimizer"], 512)
        self.assertEqual(result["math500"]["optimizer"], 1020)

    def test_missing_metrics_are_not_filled_from_older_checkpoint(self):
        result = catalog.last_evaluations(
            [
                {"step/outer": 1, "eval/aime25/acc/mean": 0.2, "eval/aime25/acc/pass@16": 0.5},
                {"step/outer": 2, "eval/aime25/acc/mean": 0.3},
            ]
        )
        self.assertIsNone(result["aime25"]["pass16"])

    def test_legacy_estimators_and_event_axes_keep_their_meaning(self):
        result = catalog.last_evaluations(
            [
                {"_step": 999, "val-core/MATH-500/acc/mean@16": 0.8, "val-core/MATH-500/acc/best@16": 0.9},
            ]
        )["math500"]
        self.assertIsNone(result["outer"])
        self.assertIsNone(result["optimizer"])
        self.assertIsNone(result["pass16"])
        self.assertEqual(len(result["legacy_estimators"]), 1)

    def fixture(self, root, folder="one", exit_code=0):
        raw = root / (folder + ".wandb")
        raw.write_bytes(b"fixture")
        stat = raw.stat()
        target = root / "derived/MirrorOPD-verl" / folder
        target.mkdir(parents=True)
        meta = {
            "run": {"run_id": "same-id", "display_name": "math-opd-E50-fixture", "project": "MirrorOPD-verl"},
            "status": "finished",
            "exit_code": exit_code,
            "history_rows": 0,
            "cache_key": {"source": str(raw), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        }
        (target / "run.json").write_text(json.dumps(meta))
        (target / "config.json").write_text("{}")
        (target / "history.jsonl").write_text("")
        return raw

    def test_nonzero_exit_is_not_a_completed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, exit_code=1)
            self.assertEqual(catalog.collect(root)[0]["execution_status"], "failed_exit")

    def test_duplicate_run_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, "one")
            self.fixture(root, "two")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                catalog.collect(root)

    def test_stale_decoder_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = self.fixture(root)
            raw.write_bytes(b"new snapshot")
            with self.assertRaisesRegex(ValueError, "Stale"):
                catalog.collect(root)


if __name__ == "__main__":
    unittest.main()
