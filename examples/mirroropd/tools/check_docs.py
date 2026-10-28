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
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
DOCS = Path("examples/mirroropd/docs")
LINK = re.compile(r"!?\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)")
SYMBOL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


sys.path.insert(0, str(ROOT))
from examples.mirroropd.tools.maintenance import doc_checks, policy  # noqa: E402


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
    return doc_checks.markdown_anchors(text)


def python_symbols(text):
    symbols = set()

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = prefix + child.name
                symbols.add(name)
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(text))
    return symbols


def maintained_pages(root):
    return policy.markdown_files(root)


def references(prose):
    """Keep display labels for AST checks; parse reference links and authored HTML."""
    tokens = doc_checks.markdown_parser().parse(prose)
    for token in tokens:
        children = token.children or []
        for index, child in enumerate(children):
            if child.type == "link_open":
                label = []
                for following in children[index + 1 :]:
                    if following.type == "link_close":
                        break
                    if following.type in {"text", "code_inline"}:
                        label.append(following.content)
                yield "".join(label), child.attrGet("href"), (token.map or [0])[0] + 1
            elif child.type == "image":
                yield child.content, child.attrGet("src"), (token.map or [0])[0] + 1
    for target in doc_checks.html_references(tokens).links:
        yield "", target, 1


def check_pages(pages, root=None):
    if root is None:
        root = (
            ROOT
            if all(page.is_relative_to(ROOT) for page in pages)
            else Path(os.path.commonpath([str(page.parent) for page in pages]))
        )
    errors, source_cache, anchor_cache = [], {}, {}
    counts = {"pages": 0, "local_links": 0, "symbols": 0, "python_examples": 0, "unavailable_evidence": 0}
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
        for label, target, line in references(prose):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            counts["local_links"] += 1
            path = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page.resolve()
            outside = tuple(root.parent / name for name in ("wandb-logs", "MirrorOPD", "model", "models", "runs"))

            def unavailable(destination, allowed=outside):
                allowed_evidence = any(destination.is_relative_to(base) for base in allowed)
                relative = destination.relative_to(root) if destination.is_relative_to(root) else None
                artifact = destination.suffix in {
                    ".json",
                    ".jsonl",
                    ".gz",
                    ".csv",
                    ".log",
                    ".txt",
                    ".pdf",
                    ".tex",
                    ".parquet",
                    ".pt",
                }
                local_artifact = relative is not None and (
                    relative.parts[0] in {"outputs", "runs"}
                    or (
                        artifact
                        and (
                            "data" in relative.parts
                            or relative.is_relative_to("examples/mirroropd/docs/archive/evidence")
                        )
                    )
                )
                if allowed_evidence or local_artifact:
                    counts["unavailable_evidence"] += 1
                    return True
                return False

            error = doc_checks.check_link(root, page, target, anchor_cache, external_roots=outside, missing=unavailable)
            if (
                error is None
                and path.is_file()
                and path.suffix == ".py"
                and not label.endswith(".py")
                and SYMBOL.fullmatch(label)
            ):
                counts["symbols"] += 1
                if path not in source_cache:
                    source_cache[path] = python_symbols(path.read_text())
                if label not in source_cache[path]:
                    error = f"missing Python definition {label}: {target}"
            if error:
                error = error.replace("本地链接不存在：", "missing local target: ").replace(
                    "章节锚点不存在：", "missing Markdown anchor: "
                )
                errors.append(f"{page}:{line}: {error}")
    return counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("paths", nargs="*", type=Path, help="Explicit Markdown pages; defaults to maintained library")
    args = parser.parse_args()
    counts, errors = check_pages(args.paths or maintained_pages(args.root.resolve()), root=args.root.resolve())
    for error in errors:
        print(error)
    print("MirrorOPD docs: " + ", ".join(f"{value} {key}" for key, value in counts.items()) + f", {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
