# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Skills, document budgets and trusted local Notes history; no new manifest."""

import hashlib
import subprocess

import yaml

from .doc_checks import source_characters
from .notes_guard import load_manifest
from .notes_guard import run as check_notes
from .policy import COMPONENT, HARD_LIMITS, REQUIRED_SKILLS, SOFT_CHARACTERS, markdown_files


def check(repo, history=None):
    component = repo / COMPONENT
    errors, warnings = [], []
    skills = component / ".agents/skills"
    for name in sorted(REQUIRED_SKILLS):
        try:
            source = (skills / name / "SKILL.md").read_text()
            parts = source.split("---", 2)
            meta = yaml.safe_load(parts[1]) if len(parts) == 3 and not parts[0] else None
            if not isinstance(meta, dict) or meta.get("name") != name or not meta.get("description"):
                raise ValueError("Invalid skill metadata")
            ui = yaml.safe_load((skills / name / "agents/openai.yaml").read_text())["interface"]
            if not ui.get("display_name") or not ui.get("short_description"):
                raise ValueError("Missing skill UI metadata")
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            errors.append(f"{name}: {error}")
    for name, limit in HARD_LIMITS.items():
        path = component / name
        if not path.is_file():
            errors.append(f"Required rule file missing: {name}")
        elif source_characters(path.read_text()) > limit:
            errors.append(f"Rule entry exceeds {limit} non-whitespace characters: {name}")
    for page in markdown_files(repo):
        count = source_characters(page.read_text())
        if count > SOFT_CHARACTERS:
            warnings.append(
                f"Long topic ({count} characters): {page.relative_to(repo)}; review ownership, not deletion quota"
            )
    errors.extend(check_notes(component, None, False))
    if history is not None:
        name = (COMPONENT / ".agents/notes/archive-seals.json").as_posix()
        baseline = subprocess.run(["git", "-C", str(history), "show", "HEAD:" + name], capture_output=True, text=True)
        if baseline.returncode == 0:
            old = load_manifest(baseline.stdout)
            current_path = repo / name
            current = load_manifest(current_path.read_text()) if current_path.exists() else {}
            for relative, digest in old.items():
                path = component / ".agents/notes" / relative
                if (
                    current.get(relative) != digest
                    or not path.is_file()
                    or path.is_symlink()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    errors.append(f"Previously sealed Note was changed or removed: {relative}")
    return errors, warnings
