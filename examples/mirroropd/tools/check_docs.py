# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Check maintained MirrorOPD docs without importing training dependencies.

Python source links labelled with a qualified symbol, e.g. [LossContext.aggregate],
are checked against AST definitions. Filename labels link to the whole file.
Markdown coverage is intentionally limited to inline links, ATX headings and HTML
id anchors used by this library; historical pages are targets, not audited sources.
"""

import argparse
import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
DOCS = Path("examples/mirroropd/docs")
LINK = re.compile(r"!?\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)")
SYMBOL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def markdown_parts(text):
    """Return prose with source line numbers preserved, and fenced Python examples."""
    prose, examples, block = [], [], []
    fence = None
    language = ""
    start = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence is None and match:
            fence, language, start = match[1], match[2].strip(), line_number + 1
            block = []
            prose.append("")
        elif fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                if language == "python":
                    examples.append((start, "\n".join(block)))
                fence = None
            else:
                block.append(line)
            prose.append("")
        else:
            prose.append(line)
    if fence is not None:
        raise ValueError(f"unclosed code fence at line {start - 1}")
    return "\n".join(prose), examples


def markdown_anchors(text):
    prose, _ = markdown_parts(text)
    anchors = set(re.findall(r'\bid=["\']([^"\']+)["\']', prose))
    seen = {}
    for line in prose.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)(?:\s+#+)?$", line)
        if heading:
            title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading[1])
            title = re.sub(r"<[^>]*>", "", title)
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            occurrence = seen.get(slug, 0)
            seen[slug] = occurrence + 1
            anchors.add(slug if occurrence == 0 else f"{slug}-{occurrence}")
    return anchors


def python_symbols(text):
    symbols = set()

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                symbols.add(name)
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(text))
    return symbols


def maintained_pages(root):
    docs = root / DOCS
    return (
        [root / "examples/mirroropd/README.md", docs / "README.md", docs / "architecture.md"]
        + sorted((docs / "internals").glob("*.md"))
    )


def check_pages(pages):
    errors, source_cache, anchor_cache = [], {}, {}
    counts = {"pages": 0, "local_links": 0, "symbols": 0, "python_examples": 0}
    for page in pages:
        counts["pages"] += 1
        try:
            prose, examples = markdown_parts(page.read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"{page}: {exc}")
            continue
        for line, example in examples:
            counts["python_examples"] += 1
            try:
                compile(example, str(page), "exec")
            except SyntaxError as exc:
                errors.append(f"{page}:{line + (exc.lineno or 1) - 1}: invalid Python: {exc.msg}")
        for match in LINK.finditer(prose):
            label, target = match.groups()
            line = prose[: match.start()].count("\n") + 1
            target = target[1:-1] if target.startswith("<") else target
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            counts["local_links"] += 1
            path = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page
            error = None
            if not path.exists():
                error = f"missing local target: {target}"
            elif path.suffix == ".md" and parsed.fragment:
                if path not in anchor_cache:
                    anchor_cache[path] = markdown_anchors(path.read_text())
                if unquote(parsed.fragment) not in anchor_cache[path]:
                    error = f"missing Markdown anchor: {target}"
            elif path.suffix == ".py" and not label.endswith(".py") and SYMBOL.fullmatch(label):
                counts["symbols"] += 1
                if path not in source_cache:
                    source_cache[path] = python_symbols(path.read_text())
                if label not in source_cache[path]:
                    error = f"missing Python definition {label}: {target}"
            if error:
                errors.append(f"{page}:{line}: {error}")
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Explicit Markdown pages; defaults to maintained library")
    args = parser.parse_args()
    counts, errors = check_pages(args.paths or maintained_pages(ROOT))
    for error in errors:
        print(error)
    print("MirrorOPD docs: " + ", ".join(f"{value} {key}" for key, value in counts.items()) + f", {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
