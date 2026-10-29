# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Download pinned additional teachers without changing the completed run registry."""

import concurrent.futures
import os
import time

from huggingface_hub import HfApi, hf_hub_download

from examples.mirroropd.research.core.io import dump, read
from examples.mirroropd.research.studies.boundary.round2_common import NEW_MODELS, OUT, model_path


def one(name):
    receipt = OUT / "assets" / f"{name}.json"
    if receipt.exists():
        return read(receipt)
    repo, revision = NEW_MODELS[name]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    dest = model_path(name)
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for item in info.siblings:
        name_in_repo = item.rfilename
        if "/" in name_in_repo or not name_in_repo.endswith((".json", ".safetensors", ".md", ".jinja")):
            continue
        for attempt in range(4):
            try:
                path = hf_hub_download(repo, name_in_repo, revision=revision, local_dir=dest)
                files.append(dict(file=name_in_repo, bytes=os.stat(path).st_size))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        print(name, name_in_repo, flush=True)
    result = dict(repo=repo, revision=revision, path=str(dest), files=files)
    dump(receipt, result)
    return result


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, NEW_MODELS))
    dump(OUT / "assets.json", results)


if __name__ == "__main__":
    main()
