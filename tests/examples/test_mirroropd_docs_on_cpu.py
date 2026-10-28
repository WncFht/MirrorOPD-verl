# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""The documentation check must reject broken references without importing code."""

import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "examples/mirroropd/tools/check_docs.py"
spec = importlib.util.spec_from_file_location("mirroropd_check_docs", SCRIPT)
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def test_valid_links_symbols_and_examples(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("raise RuntimeError('must not import')\nclass Worker:\n    def update(self): pass\n")
    target = tmp_path / "target.md"
    target.write_text('# 接口\n## 输入与输出\n## 输入与输出\n<a id="explicit"></a>\n')
    page = tmp_path / "page.md"
    page.write_text(
        "[Worker.update](source.py)\n[中文](target.md#输入与输出-1)\n[显式](target.md#explicit)\n"
        "```python\nx = 1\n```\n```text\n[ignored](missing.md)\n```\n"
    )
    counts, errors = docs.check_pages([page])
    assert not errors
    assert counts == {"pages": 1, "local_links": 3, "symbols": 1, "python_examples": 1}


def test_missing_target_anchor_and_qualified_symbol_are_errors(tmp_path):
    (tmp_path / "source.py").write_text("class Other:\n    def update(self): pass\n")
    (tmp_path / "target.md").write_text("# Existing\n")
    page = tmp_path / "page.md"
    page.write_text("[broken](missing.md)\n[anchor](target.md#missing)\n[Worker.update](source.py)\n")
    _, errors = docs.check_pages([page])
    assert len(errors) == 3
    assert "missing local target" in errors[0]
    assert "missing Markdown anchor" in errors[1]
    assert "missing Python definition Worker.update" in errors[2]


def test_invalid_python_reports_example_line(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("# Example\n\n```python\nif :\n```\n")
    _, errors = docs.check_pages([page])
    assert len(errors) == 1
    assert f"{page}:4: invalid Python" in errors[0]


def test_unclosed_fence_is_an_error(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("# Example\n```python\nx = 1\n")
    _, errors = docs.check_pages([page])
    assert len(errors) == 1
    assert "unclosed code fence at line 2" in errors[0]


def test_default_library_includes_untracked_authored_documents(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    directory = tmp_path / "examples/mirroropd/docs/infra/architecture"
    directory.mkdir(parents=True)
    tracked = directory / "tracked.md"
    tracked.write_text("# Ready\n")
    (directory / "draft.md").write_text("[unfinished](missing.md)\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", str(tracked)], check=True)
    for excluded in ("archive", "literature/sources", "infra/numerics/data"):
        source = tmp_path / "examples/mirroropd/docs" / excluded / "frozen.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("[original](missing.md)\n")
    pages = docs.maintained_pages(tmp_path)
    assert set(pages) == {tracked, directory / "draft.md"}
    _, errors = docs.check_pages(pages)
    assert len(errors) == 1 and "missing.md" in errors[0]


def test_ignored_local_output_does_not_hide_missing_source(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/outputs/\n")
    page = tmp_path / "page.md"
    page.write_text("[local evidence](outputs/run/README.md)\n[source](missing.md)\n")
    monkeypatch.setattr(docs, "ROOT", tmp_path)
    _, errors = docs.check_pages([page])
    assert len(errors) == 1
    assert "missing.md" in errors[0]
