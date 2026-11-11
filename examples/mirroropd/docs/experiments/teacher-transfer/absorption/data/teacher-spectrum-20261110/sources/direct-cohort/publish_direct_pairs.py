"""Publish immutable measured evidence and figure assets beside the scientific page."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DOC = REPO / "examples/mirroropd/docs/experiments/teacher-transfer"
DATA = DOC / "data/teacher-spectrum-20261110"
FIGURES = DOC / "figures/teacher-spectrum-20261110"


def main():
    index_path = DATA / "evidence-index.json"
    index = json.loads(index_path.read_text())
    records = {r["published"]: r for r in index["artifacts"]}
    files = []
    for dirname in ["direct-pairs-v1", "direct-pairs-analysis-v4"]:
        files.extend(p for p in (ROOT / dirname).rglob("*") if p.is_file())
    names = [
        "prepare_direct_pairs.py",
        "prepare_direct_pairs-executed.py",
        "direct_pair_coordinates.py",
        "measure_direct_pairs.py",
        "measure_direct_pairs_v2.py",
        "measure_direction_controls.py",
        "measure_direction_controls_v2.py",
        "analyze_direct_pairs.py",
        "plot_direct_dynamics.py",
        "publish_direct_pairs.py",
        "publish_spectrum_evidence.py",
        "publish_spectrum_evidence-executed.py",
    ]
    for name in names:
        src = ROOT / name
        dst = DATA / "sources/direct-cohort" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert dst.read_bytes() == src.read_bytes(), dst
        else:
            shutil.copy2(src, dst)
        records[str(dst.relative_to(DOC))] = dict(
            source=str(src),
            published=str(dst.relative_to(DOC)),
            bytes=dst.stat().st_size,
            sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
        )
    for src in files:
        visual = src.suffix in [".png", ".pdf", ".svg"]
        dst = (FIGURES if visual else DATA) / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert dst.read_bytes() == src.read_bytes(), dst
        else:
            shutil.copy2(src, dst)
        records[str(dst.relative_to(DOC))] = dict(
            source=str(src),
            published=str(dst.relative_to(DOC)),
            bytes=dst.stat().st_size,
            sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
        )
    index["artifacts"] = list(records.values())
    index["archive_bytes"] = sum(r["bytes"] for r in records.values())
    index["direct_cohort"] = dict(
        pairs=21,
        conditions=22,
        primary_continuous=19,
        data="data/teacher-spectrum-20261110/direct-pairs-v1",
        analysis="data/teacher-spectrum-20261110/direct-pairs-analysis-v4",
    )
    index_path.write_text(json.dumps(index, indent=2))
    print("Published new files", len(files) + len(names), "total", len(records))


if __name__ == "__main__":
    main()
