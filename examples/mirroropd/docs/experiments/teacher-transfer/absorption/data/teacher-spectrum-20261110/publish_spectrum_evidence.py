"""Publish cited small research artifacts beside the sole scientific document.

Keep original experiment attempts unchanged. PNG/PDF/SVG go to figures;
numeric evidence and source snapshots go to data. The paper never links to
runs after publication. Large checkpoint weights and caches are not copied.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

REPO = Path.cwd()
RUN = REPO / "runs/teacher-spectrum-followup-20261110"
DOC = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
PAGE = DOC / "2026-11-09-教师变化与OPD吸收的谱实验设计.md"
DATA = DOC / "data/teacher-spectrum-20261110"
FIG = DOC / "figures/teacher-spectrum-20261110"
VISUAL = {".png", ".pdf", ".svg"}
SMALL = {".json", ".csv", ".py", ".html", ".js", ".cjs", ".npz", ".txt", ".audit", ".md"}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    inventory = []

    def publish(source):
        rel = source.relative_to(RUN)
        dest = (FIG if source.suffix in VISUAL else DATA) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == source.read_bytes(), f"Archive conflict: {dest}"
        else:
            shutil.copy2(source, dest)
        inventory.append(
            dict(
                source=str(source),
                published=str(dest.relative_to(DOC)),
                bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
        return dest.relative_to(DOC).as_posix()

    content = PAGE.read_text()

    def replace(match):
        link = match.group(1)
        if "/runs/" not in link:
            return match.group(0)
        source = (PAGE.parent / link).resolve()
        if source == RUN.resolve():
            return "](data/teacher-spectrum-20261110/evidence-index.json)"
        if source.is_dir():
            files = [
                p
                for p in source.rglob("*")
                if p.is_file() and p.suffix in SMALL | VISUAL and p.stat().st_size < 30_000_000
            ]
            for file in files:
                publish(file)
            index = source / "publication-placeholder.json"
            rel = source.relative_to(RUN)
            target = DATA / rel / "index.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    dict(artifacts=[r["published"] for r in inventory if Path(r["source"]).is_relative_to(source)]),
                    indent=2,
                )
            )
            return "](" + target.relative_to(DOC).as_posix() + ")"
        assert source.is_file(), source
        target = publish(source)
        if source.suffix == ".png":
            for suffix in [".pdf", ".svg", ".audit.json"]:
                sibling = source.with_suffix(suffix)
                if sibling.exists():
                    publish(sibling)
        return "](" + target + ")"

    content = re.sub(r"\]\(([^)]+)\)", replace, content)
    # Preserve exact decompositions and plotting inputs, not only cited summaries.
    folders = [
        "spectrum-expanded-v4",
        "curve-spectrum-v1",
        "teacher-band-capture-v2",
        "common-ancestor-v1",
        "spectrum-screen-v1",
        "historical-dynamics-20261110",
    ]
    for folder in folders:
        for file in (RUN / folder).rglob("*"):
            if file.is_file() and file.suffix in SMALL | VISUAL and file.stat().st_size < 30_000_000:
                publish(file)
    for source in RUN.glob("*.py"):
        if any(word in source.name for word in ["analyze", "measure", "plot", "spectrum", "summarize"]):
            publish(source)
    content = content.replace(
        "原始数值、不可变 attempt、逐题输出和图源统一保存在[运行目录]", "图表、数值与复算源码见[证据索引]"
    )
    content = content.replace(
        "图源为运行目录中的 `summarize_spectrum_screen.py`，精确执行版本与输出哈希保留在 `spectrum-screen-v1/`。",
        "[图源快照](data/teacher-spectrum-20261110/summarize_spectrum_screen.py)与精确执行版本一并归档。",
    )
    content = content.replace("本文重构仅使用已有数据。", "后续 CPU 参数测量与重分析记录在相应小节。")
    PAGE.write_text(content)
    unique = {r["published"]: r for r in inventory}
    (DATA / "evidence-index.json").write_text(
        json.dumps(
            dict(
                artifacts=list(unique.values()),
                purpose="Self-contained cited figures, numeric outputs and exact source snapshots; original models and execution logs remain experiment provenance",
                archive_bytes=sum(r["bytes"] for r in unique.values()),
            ),
            indent=2,
        )
    )
    assert not re.search(r"\]\([^)]*/runs/", content)
    print("Published", len(unique), "artifacts;", round(sum(r["bytes"] for r in unique.values()) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
