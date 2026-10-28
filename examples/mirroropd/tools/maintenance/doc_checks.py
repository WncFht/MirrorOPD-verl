# Copyright (c) 2026 DeepSeek
# SPDX-License-Identifier: MIT
# Adapted from agent-maintenance; see LICENSE.agent-maintenance and SOURCES.md.
"""Local Markdown anchors and explicit size budgets; never compare document content."""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin

if TYPE_CHECKING:
    from pathlib import Path


def markdown_parser():
    return MarkdownIt().enable("table").use(footnote_plugin)


class HTMLReferences(HTMLParser):
    """Read explicit anchors and links only from parsed HTML tokens."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if not value:
                continue
            if key == "id" or (tag == "a" and key == "name"):
                self.anchors.add(value)
            if key == "href" or (tag == "img" and key == "src"):
                self.links.append(value)


def html_references(tokens):
    """Exclude fenced code and code spans from HTML interpretation."""
    parser = HTMLReferences()
    for token in tokens:
        if token.type == "html_block":
            parser.feed(token.content)
        for child in token.children or []:
            if child.type == "html_inline":
                parser.feed(child.content)
    parser.close()
    return parser


def markdown_links(tokens):
    """Yield Markdown and authored HTML destinations, including local fragments."""
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                yield child.attrGet("href")
            elif child.type == "image":
                yield child.attrGet("src")
    yield from html_references(tokens).links


def heading_slug(text):
    """GitHub-style text slugs: lowercase, retain Unicode letters and hyphens."""
    text = text.lower()
    return "".join(
        "-" if char == " " else char for char in text if char in " _-" or unicodedata.category(char)[0] in "LMN"
    )


def markdown_anchors(source):
    """Collect heading collisions in source order and explicit HTML anchors."""
    tokens = markdown_parser().parse(source)
    result = set(html_references(tokens).anchors)
    used = set()
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1]
        text = "".join(
            child.content
            for child in inline.children or []
            if child.type in {"text", "code_inline", "image", "softbreak"}
        )
        slug = heading_slug(text)
        candidate = slug
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        used.add(candidate)
        result.add(candidate)
    return result


def check_link(root: Path, source: Path, target: str, cache: dict, *, external_roots=(), missing=None):  # noqa: PLR0911
    # Each early return identifies a distinct link-validation outcome.
    """Return a diagnostic for an invalid local destination, otherwise None."""
    try:
        url = urlsplit(target)
    except ValueError:
        return f"无效链接：{target}"
    if url.scheme or url.netloc:
        return None
    destination = (source.parent / unquote(url.path)).resolve() if url.path else source.resolve()
    allowed = (root.resolve(), *external_roots)
    if not any(destination.is_relative_to(base) for base in allowed):
        return f"本地链接越出允许范围：{target}"
    if not destination.exists():
        if missing is not None and missing(destination):
            return None
        return f"本地链接不存在：{target}"
    if url.fragment and destination.is_dir():
        readmes = sorted(p for p in destination.iterdir() if p.name.lower() == "readme.md")
        if not readmes:
            return f"目录片段缺少 README：{target}"
        destination = readmes[0].resolve()
        if not any(destination.is_relative_to(base) for base in allowed):
            return f"本地链接不存在或越出仓库：{target}"
    if not url.fragment or destination.suffix.lower() != ".md":
        return None
    if destination not in cache:
        cache[destination] = markdown_anchors(destination.read_text(encoding="utf-8"))
    if unquote(url.fragment) not in cache[destination]:
        return f"章节锚点不存在：{target}"
    return None


def source_characters(source):
    """Count all non-whitespace source characters, including code and tables."""
    return len(re.sub(r"\s", "", source))
