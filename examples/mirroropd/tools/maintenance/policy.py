# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared scope for formatting and checking, adapted from agent-maintenance."""

from pathlib import Path

COMPONENT = Path("examples/mirroropd")
EXCLUDED = {
    "archive",
    "history",
    "archived",
    "data",
    "sources",
    "graph-expansion-sources",
    "node_modules",
    "__pycache__",
    ".venv",
    ".git",
    ".crossnote",
}
REQUIRED_SKILLS = {"maintain-docs", "maintain-prose", "maintain-decision-notes"}
HARD_LIMITS = {
    "AGENTS.md": 4000,
    "docs/AGENTS.md": 3000,
    **{f".agents/skills/{name}/SKILL.md": 3500 for name in REQUIRED_SKILLS},
}
SOFT_CHARACTERS = 20000


def owned(path, component):
    """Frozen input and symlinks never become formatter inputs."""
    if path.is_symlink() or not path.is_file() or path.suffix != ".md":
        return False
    relative = path.relative_to(component)
    if relative.is_relative_to("docs/infra/numerics/full-vocab/reference"):
        return False
    return not EXCLUDED.intersection(relative.parts)


def markdown_files(repo, explicit=()):
    component = repo / COMPONENT
    if explicit:
        result = []
        for value in explicit:
            path = Path(value)
            path = path if path.is_absolute() else repo / path
            if path.is_symlink():
                raise ValueError(f"Symbolic input is not editable: {value}")
            path = path.resolve()
            if not path.is_relative_to(component.resolve()) or not owned(path, component.resolve()):
                raise ValueError(f"Not an editable project Markdown file: {value}")
            result.append(path)
    else:
        result = [path for path in component.rglob("*.md") if owned(path, component)]
    if not result:
        raise ValueError("No maintained Markdown inputs; refusing an empty check")
    return sorted(set(result))
