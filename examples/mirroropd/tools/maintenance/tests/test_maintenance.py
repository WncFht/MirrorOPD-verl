# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Real format, parsing, scope and index failures, without training dependencies."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from examples.mirroropd.tools import check_docs  # noqa: E402
from examples.mirroropd.tools.maintenance import contracts, doc_checks, pipeline, policy  # noqa: E402


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.component = self.root / policy.COMPONENT
        self.component.mkdir(parents=True)

    def write(self, name, text):
        path = self.component / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def fixture(self):
        self.write("README.md", "# Fixture\n")
        self.write("AGENTS.md", "# Rules\n")
        self.write("docs/AGENTS.md", "# Document rules\n")
        self.write(".agents/notes/README.md", "# Notes\n")
        for name in policy.REQUIRED_SKILLS:
            self.write(f".agents/skills/{name}/SKILL.md", f"---\nname: {name}\ndescription: Fixture.\n---\n\n# Skill\n")
            self.write(
                f".agents/skills/{name}/agents/openai.yaml",
                "interface:\n  display_name: fixture\n  short_description: fixture\n",
            )
        result = pipeline.format_paths(self.root, policy.markdown_files(self.root), write=True)
        self.assertEqual(result, 0)

    def git(self, *args):
        env = dict(os.environ)
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        ):
            env.pop(key, None)
        return subprocess.check_output(["git", "-C", str(self.root), *args], env=env, text=True)

    def command(self, *args):
        return subprocess.run(
            [sys.executable, str(pipeline.TOOL / "pipeline.py"), *args, "--root", str(self.root)],
            text=True,
            capture_output=True,
        )

    def test_scope_includes_hidden_untracked_and_rejects_frozen(self):
        current = self.write("docs/current.md", "# Current\n")
        hidden = self.write(".agents/references/current.md", "# Rule\n")
        frozen = self.write("docs/archive/old.md", "# Old\n")
        source = self.write("docs/literature/sources/raw.md", "# Original\n")
        self.assertEqual(set(policy.markdown_files(self.root)), {current, hidden})
        for path in (frozen, source, self.root / "missing.md"):
            with self.assertRaises(ValueError):
                policy.markdown_files(self.root, [path])
        self.write("docs/link.md", "# x\n").unlink()
        (self.component / "docs/link.md").symlink_to(current)
        with self.assertRaises(ValueError):
            policy.markdown_files(self.root, [self.component / "docs/link.md"])

    def test_empty_scan_fails(self):
        with self.assertRaises(ValueError):
            policy.markdown_files(self.root)
        self.assertNotEqual(self.command("check").returncode, 0)

    def test_reference_links_html_directory_and_escaped_scope(self):
        page = self.write(
            "docs/page.md", '# Page\n\n[中文][ref]\n\n[ref]: other.md#中文-api\n\n<a href="guide/#guide">Guide</a>\n'
        )
        self.write("docs/other.md", "# 中文 `API`\n")
        self.write("docs/guide/README.md", "# Guide\n")
        _, errors = check_docs.check_pages([page], root=self.root)
        self.assertEqual(errors, [])
        page.write_text(page.read_text().replace("#中文-api", "#missing"))
        _, errors = check_docs.check_pages([page], root=self.root)
        self.assertTrue(any("anchor" in error for error in errors))
        self.assertIsNotNone(doc_checks.check_link(self.root, page, "../../../../escape.md", {}))

    def test_heading_collisions_html_and_code_literals(self):
        text = '# A\n# A-1\n# A\n<a name="legacy"></a>\n```html\n<a id="fake"></a>\n```\n'
        self.assertEqual(doc_checks.markdown_anchors(text), {"a", "a-1", "a-2", "legacy"})

    def test_unavailable_external_evidence_is_reported_not_claimed_verified(self):
        page = self.write("README.md", "# Scope\n\n[evidence](../../../wandb-logs/missing.json)\n")
        counts, errors = check_docs.check_pages([page], root=self.root)
        self.assertEqual(errors, [])
        self.assertEqual(counts["unavailable_evidence"], 1)
        page.write_text("# Scope\n\n[source](missing.py)\n")
        self.assertTrue(check_docs.check_pages([page], root=self.root)[1])

    def test_missing_frozen_artifacts_do_not_hide_missing_navigation(self):
        page = self.write("docs/page.md", "# Evidence\n\n[history](archive/evidence/run/history.jsonl)\n")
        counts, errors = check_docs.check_pages([page], root=self.root)
        self.assertEqual(errors, [])
        self.assertEqual(counts["unavailable_evidence"], 1)
        for target in ("archive/evidence/run/README.md", "archive/evidence/run/plot.svg"):
            page.write_text(f"# Evidence\n\n[reference]({target})\n")
            self.assertTrue(check_docs.check_pages([page], root=self.root)[1])

    def test_formatter_preserves_code_math_and_attributes_and_is_idempotent(self):
        text = (
            '# 中文English\n\n中文English正文。\n\n```python\nx = "中文English"\n'
            "# preserve $$ and $& literally\n```\n\n$$\np_\\theta = 1e-6 + \\text{中文English}\n$$\n"
            '\n<a id="中文English"></a>\n'
        )
        page = self.write("docs/format.md", text)
        self.assertNotEqual(pipeline.format_paths(self.root, [page]), 0)
        self.assertEqual(pipeline.format_paths(self.root, [page], write=True), 0)
        after = page.read_text()
        for literal in ('x = "中文English"', "p_\\theta = 1e-6 + \\text{中文English}", 'id="中文English"'):
            self.assertIn(literal, after)
        self.assertIn("中文 English 正文", after)
        self.assertEqual(after.count("$$"), 3)
        self.assertIn("# preserve $$ and $& literally", after)
        self.assertEqual(pipeline.format_paths(self.root, [page]), 0)
        self.assertEqual(page.read_text(), after)

    def test_lint_rejects_real_markdown_violation(self):
        page = self.write("docs/lint.md", "# First\n\n# Second\n")
        self.assertNotEqual(pipeline.lint_paths(self.root, [page]), 0)

    def test_footnote_description_is_not_a_destination(self):
        page = self.write("docs/footnote.md", "# Page\n\nClaim.[^ref]\n\n[^ref]: 同一版本。[源码](other.md)\n")
        self.write("docs/other.md", "# Source\n")
        self.assertEqual(check_docs.check_pages([page], root=self.root)[1], [])

    def test_hook_dependencies_match_local_parser_requirements(self):
        import yaml

        config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
        hook = next(h for group in config["repos"] for h in group["hooks"] if h["id"] == "check-mirroropd-docs")
        expected = set((pipeline.TOOL / "requirements.txt").read_text().splitlines())
        self.assertEqual(set(hook["additional_dependencies"]), expected)
        self.assertIn("--staged", hook["entry"])

    def test_hooks_preserve_frozen_python_but_check_active_tools(self):
        import re

        import yaml

        config = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
        hooks = {hook["id"]: hook for group in config["repos"] for hook in group["hooks"]}
        for name in ("ruff", "ruff-format", "mypy", "check-license", "compileall"):
            exclude = hooks[name]["exclude"]
            self.assertRegex("examples/mirroropd/docs/archive/source.py", exclude)
            self.assertRegex("examples/mirroropd/docs/literature/sources/model.py", exclude)
            self.assertIsNone(
                re.search(exclude, "examples/mirroropd/research/studies/production/analyze_equal_budget.py")
            )

    def test_rules_hard_limit_and_missing_skill_fail(self):
        self.fixture()
        self.assertEqual(contracts.check(self.root)[0], [])
        self.write("AGENTS.md", "x" * 5000)
        self.assertTrue(any("exceeds" in e for e in contracts.check(self.root)[0]))
        (self.component / ".agents/skills/maintain-docs/SKILL.md").unlink()
        self.assertTrue(any("maintain-docs" in e for e in contracts.check(self.root)[0]))

    def test_staged_bad_worktree_fixed_and_reverse_preserve_index(self):
        self.fixture()
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.test")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        page = self.component / "README.md"
        bad, good = "# Fixture\n\n[missing](missing.md)\n", "# Fixture\n"
        page.write_text(bad)
        self.git("add", ".")
        staged_before = self.git("ls-files", "--stage")
        page.write_text(good)
        result = self.command("check", "--staged")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.git("ls-files", "--stage"), staged_before)
        self.assertEqual(page.read_text(), good)
        self.git("add", ".")
        page.write_text(bad)
        result = self.command("check", "--staged")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(page.read_text(), bad)

    def test_sealed_note_cannot_be_rewritten_with_its_manifest(self):
        self.fixture()
        name = "archived/process/2026-10-01-choice.md"
        original = "# Agent Note: Choice\n\nStatus: implemented\nArchived: 2026-10-01\n\nOriginal reason.\n"
        note = self.write(".agents/notes/" + name, original)
        manifest = self.write(
            ".agents/notes/archive-seals.json",
            json.dumps({"version": 1, "files": {name: hashlib.sha256(note.read_bytes()).hexdigest()}}),
        )
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.test")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        note.write_text(original.replace("Original", "Changed"))
        manifest.write_text(json.dumps({"version": 1, "files": {name: hashlib.sha256(note.read_bytes()).hexdigest()}}))
        errors, _ = contracts.check(self.root, self.root)
        self.assertTrue(any("Previously sealed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
