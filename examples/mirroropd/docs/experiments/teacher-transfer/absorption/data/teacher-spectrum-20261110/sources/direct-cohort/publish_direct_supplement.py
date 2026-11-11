"""Append broad-metric evidence and offline recomputation checks to the archive."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parents[1] / "examples/mirroropd/docs/experiments/teacher-transfer"
DATA = DOC / "data/teacher-spectrum-20261110"
FIG = DOC / "figures/teacher-spectrum-20261110"


def main():
    index_path = DATA / "evidence-index.json"
    index = json.loads(index_path.read_text())
    records = {r["published"]: r for r in index["artifacts"]}
    selected = []
    for name in ["analyze_direct_associations.py", "plot_direct_associations.py", "publish_direct_supplement.py"]:
        selected.append((ROOT / name, DATA / "sources/direct-cohort" / name))
    for src in (ROOT / "direct-pairs-analysis-v4").glob("*"):
        if src.is_file():
            target = FIG if src.suffix in [".png", ".pdf", ".svg"] else DATA
            selected.append((src, target / src.relative_to(ROOT)))
    for src, dst in selected:
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
    index_path.write_text(json.dumps(index, indent=2))
    print("Archived artifacts", len(records), "bytes", index["archive_bytes"])


if __name__ == "__main__":
    main()
