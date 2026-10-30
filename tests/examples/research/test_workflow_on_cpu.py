# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Executable boundaries: frozen imports, dependency-free help and atomic evidence."""

import ast
import os
import subprocess
import sys

import pytest

from examples.mirroropd.research.__main__ import commands
from examples.mirroropd.research.core.io import write_csv
from examples.mirroropd.research.core.paths import REPO
from examples.mirroropd.research.runtime.source import environment, freeze


@pytest.mark.parametrize("command", [None, "fixed_transfer", "production.plot_policy_run"])
def test_discovery_and_help_need_only_standard_library(command):
    args = [sys.executable, "-S", "-m", "examples.mirroropd.research"]
    if command:
        args += [command, "--help"]
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout if command else "fixed_transfer" in result.stdout


def test_removed_entrypoints_have_no_aliases():
    names = commands()
    assert "cross_batch.long_batch_transfer" in names
    for old in ("transfer", "precision", "analysis", "common", "objectives", "rollout_policy"):
        assert old not in names
        assert not (REPO / "examples/mirroropd/research" / old).exists()


def test_research_imports_and_module_attributes_resolve():
    """Check consumers without loading optional model, plotting or grader packages."""
    prefix = "examples.mirroropd.research"
    trees, exports = {}, {}
    for path in (REPO / "examples/mirroropd/research").rglob("*.py"):
        name = ".".join(path.relative_to(REPO).with_suffix("").parts).removesuffix(".__init__")
        tree = ast.parse(path.read_text())
        trees[name] = tree
        names = {"__file__", "__name__", "__doc__"}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
        exports[name] = names
    failures = []
    for name, tree in trees.items():
        aliases = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(prefix):
                continue
            for alias in node.names:
                child = node.module + "." + alias.name
                if child in exports:
                    aliases[alias.asname or alias.name] = child
                elif alias.name not in exports.get(node.module, set()):
                    failures.append((name, node.lineno, child))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                owner = aliases.get(node.value.id)
                if owner and node.attr not in exports[owner] and owner + "." + node.attr not in exports:
                    failures.append((name, node.lineno, owner + "." + node.attr))
    assert not failures, failures


def test_workers_import_frozen_shared_code(tmp_path):
    original = tmp_path / "checkout"
    (original / "verl").mkdir(parents=True)
    (original / "verl/__init__.py").write_text("VALUE = 'original'\n")
    research = original / "examples/mirroropd/research"
    research.mkdir(parents=True)
    assets = original / "examples/mirroropd/assets"
    assets.mkdir(parents=True)
    (assets / "paths.py").write_text("ASSET = 'original asset resolver'\n")
    (assets / "model_registry.json").write_text('{"models": []}\n')
    (research / "__init__.py").write_text("from verl import VALUE\n")
    snapshot = freeze(original, tmp_path / "source")
    (original / "verl/__init__.py").write_text("VALUE = 'changed'\n")
    (research / "__init__.py").write_text("VALUE = 'changed research'\n")
    (assets / "paths.py").write_text("ASSET = 'changed asset resolver'\n")
    env = environment(snapshot, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import os; from examples.mirroropd.research import VALUE; "
            "from examples.mirroropd.assets.paths import ASSET; print(ASSET); "
            "print(VALUE); print(os.environ['MIRROROPD_WORKSPACE'])",
        ],
        cwd=snapshot,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["original asset resolver", "original", str(tmp_path)]
    assert env["PYTHONPATH"] == str(snapshot)
    assert "MIRROROPD_WORKSPACE" not in os.environ or os.environ["MIRROROPD_WORKSPACE"] != str(tmp_path)
    with pytest.raises(FileExistsError):
        freeze(original, snapshot)


def test_csv_uses_one_schema_and_rejects_empty_replacement(tmp_path):
    path = tmp_path / "result.csv"
    write_csv(path, [dict(step=0, risk=2), dict(step=1, gain=1)])
    before = path.read_bytes()
    assert before.startswith(b"step,risk,gain\r\n")
    with pytest.raises(ValueError, match="empty"):
        write_csv(path, [])
    assert path.read_bytes() == before
