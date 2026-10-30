# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact Hub revisions, reusing existing weights only after SHA verification."""

import concurrent.futures
import time

from huggingface_hub import HfApi, hf_hub_download

from examples.mirroropd.research.core.io import dump
from examples.mirroropd.research.studies.boundary.common import MODELS, OUT, model_path


def prepare(name):
    repo, revision = MODELS[name]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    dest = model_path(name)
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for item in info.siblings:
        filename = item.rfilename
        if "/" in filename or not filename.endswith((".json", ".safetensors", ".md", ".jinja")):
            continue
        target = dest / filename
        method = "hub"
        if target.is_symlink():
            raise ValueError(f"Canonical model weights must be regular files: {target}")
        if not target.exists():
            for attempt in range(4):
                try:
                    hf_hub_download(repo, filename, revision=revision, local_dir=dest)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
        files.append({"file": filename, "method": method, "bytes": target.stat().st_size})
        print(name, filename, method, flush=True)
    receipt = {"repo": repo, "revision": revision, "path": str(dest), "files": files}
    dump(OUT / "assets" / f"{name}.json", receipt)
    return receipt


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(prepare, MODELS))
    dump(OUT / "assets.json", receipts)


if __name__ == "__main__":
    main()
