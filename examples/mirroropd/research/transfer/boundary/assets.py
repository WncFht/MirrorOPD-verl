# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact Hub revisions, reusing existing weights only after SHA verification."""

import concurrent.futures
import hashlib
import time

from huggingface_hub import HfApi, hf_hub_download

from .common import MODELS, OUT, WORK, dump, model_path


def prepare(name):
    repo, revision = MODELS[name]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    dest = model_path(name)
    dest.mkdir(parents=True, exist_ok=True)
    old = WORK / "model" / repo.split("/")[1]
    files = []
    for item in info.siblings:
        filename = item.rfilename
        if "/" in filename or not filename.endswith((".json", ".safetensors", ".md", ".jinja")):
            continue
        target = dest / filename
        source = old / filename
        method = "hub"
        if filename.endswith(".safetensors") and source.exists() and not target.exists():
            expected = item.lfs.sha256 if item.lfs else None
            if expected:
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == expected:
                    target.symlink_to(source.resolve())
                    method = "verified-local-weight"
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(prepare, MODELS))
    dump(OUT / "assets.json", receipts)


if __name__ == "__main__":
    main()
