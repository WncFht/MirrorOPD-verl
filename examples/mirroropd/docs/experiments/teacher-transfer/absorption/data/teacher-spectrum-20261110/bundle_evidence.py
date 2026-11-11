"""Pack/verify/unpack per-matrix evidence next to its report, without deleting files.

Paths are relative to the evidence root. Only regular files declared in the
manifest may be unpacked, after all archive and per-member hashes pass.
"""

import argparse
import hashlib
import json
import lzma
import tarfile
from pathlib import Path, PurePosixPath

GROUPS = {
    "direct-matrices": ["direct-pairs-v1/matrices", "direct-pairs-v1/direction-controls"],
    "historical-spectra": [
        "spectrum-expanded-v4/weights",
        "spectrum-expanded-v4/runs",
        "spectrum-expanded-v4/ancillary-runs",
        "spectrum-expanded-v4/supplement-runs",
        "spectrum-expanded-v4/run-receipts",
    ],
    "teacher-bands": ["teacher-band-capture-v2"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "verify", "unpack"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "raw-evidence-bundles.json"
    if args.action == "pack":
        manifest = {"format": 1, "bundles": []}
        for name, folders in GROUPS.items():
            target = root / (name + ".tar.xz")
            if target.exists():
                raise FileExistsError(target)
            files = sorted({f for folder in folders for f in (root / folder).rglob("*") if f.is_file()})
            if name == "teacher-bands":
                files = [f for f in files if len(f.relative_to(root / "teacher-band-capture-v2").parts) > 1]
            members = []
            with tarfile.open(target, "w:xz", preset=6) as archive:
                for source in files:
                    relative = source.relative_to(root).as_posix()
                    info = archive.gettarinfo(str(source), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
                    members.append({"path": relative, "bytes": source.stat().st_size, "sha256": digest(source)})
            manifest["bundles"].append(
                {"archive": target.name, "sha256": digest(target), "bytes": target.stat().st_size, "members": members}
            )
            print(name, len(members), target.stat().st_size, flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    for bundle in manifest["bundles"]:
        if "members_index" in bundle:
            source = root / bundle["members_index"]
            assert digest(source) == bundle["members_index_sha256"], source
            bundle["members"] = json.loads(lzma.decompress(source.read_bytes()))
    # Verify the complete input before any unpacking.
    for bundle in manifest["bundles"]:
        target = root / bundle["archive"]
        assert digest(target) == bundle["sha256"], target
        expected = {item["path"]: item for item in bundle["members"]}
        with tarfile.open(target, "r:xz") as archive:
            seen = set()
            for member in archive:
                key = member.name
                assert member.isfile() and key in expected and key not in seen, key
                parts = PurePosixPath(key)
                assert not parts.is_absolute() and ".." not in parts.parts, key
                payload = archive.extractfile(member).read()
                assert len(payload) == expected[key]["bytes"], key
                assert hashlib.sha256(payload).hexdigest() == expected[key]["sha256"], key
                seen.add(key)
            assert seen == expected.keys()
    if args.action == "unpack":
        for bundle in manifest["bundles"]:
            with tarfile.open(root / bundle["archive"], "r:xz") as archive:
                for member in archive:
                    dest = root / member.name
                    payload = archive.extractfile(member).read()
                    if dest.exists():
                        assert dest.read_bytes() == payload, dest
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(payload)
    print("All archives and member hashes verified")


if __name__ == "__main__":
    main()
