# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Pack source snapshots and incremental logs/results for manual transfer."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
PREFIXES = {
    "code": "archive_MirrorOPD-verl",
    "logs": "log_MirrorOPD-verl",
    "results": "MirrorOPD_results",
}
RUNTIME_DIRS = {
    ".git",
    ".venvs",
    "venv",
    "ENV",
    "model",
    "models",
    "data",
    "dataset",
    "checkpoints",
    "outputs",
    "results",
    "wandb",
    "logs",
    "log",
    "transfer_snap",
    "log_snap",
    "result_snap",
    "offline-wheels",
    "hope",
}
CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    "build",
    "dist",
}


def repo_path(value):
    """Resolve user paths against this checkout rather than a machine-specific prefix."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def wandb_runs_dir(action):
    """Keep training logs local and receive downloads in the separate logs repository."""
    override = os.environ.get("MIRROROPD_WANDB_DIR")
    if override:
        return repo_path(override)
    if action == "pack":
        return ROOT / "wandb"
    logs_repo = os.environ.get("MIRROROPD_WANDB_LOGS_REPO")
    if logs_repo:
        return repo_path(logs_repo) / "MirrorOPD-verl/wandb"
    sibling = ROOT.parent / "wandb-logs"
    if (sibling / ".git").exists():
        return sibling / "MirrorOPD-verl/wandb"
    return Path.home() / "src/wandb-logs/MirrorOPD-verl/wandb"


def excluded(path):
    """Keep environments, training artifacts and prior transfer archives out of code snapshots."""
    return (
        path.parts[0] in RUNTIME_DIRS
        or any(part in CACHE_DIRS or part.startswith(".venv") or part.endswith(".egg-info") for part in path.parts)
        or path.name == ".DS_Store"
        or path.name.startswith("._")
        or path.name.endswith(".tar.gz")
        or ".tar.gz.part." in path.name
    )


def source_files(selections):
    """Include working-tree changes and the bundled ZIP, with or without Git on the receiving machine."""
    if any(len(PurePosixPath(path).parts) != 1 or path in (".", "..") for path in selections):
        raise ValueError("ARCHIVE_SOURCES must list repository top-level files or directories")
    for name in selections:
        if not (ROOT / name).exists():
            raise FileNotFoundError(ROOT / name)
    if (ROOT / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
        )
        candidates = [Path(os.fsdecode(name)) for name in output.split(b"\0") if name]
    else:
        candidates = []
        for directory, dirs, files in os.walk(ROOT, followlinks=False):
            parent = Path(directory).relative_to(ROOT)
            dirs[:] = [name for name in dirs if not excluded(parent / name)]
            candidates.extend(parent / name for name in files)
            candidates.extend(parent / name for name in dirs if (ROOT / parent / name).is_symlink())
    return [
        (ROOT / path, path.as_posix())
        for path in sorted(set(candidates))
        if not excluded(path)
        and (not selections or path.parts[0] in selections)
        and ((ROOT / path).is_file() or (ROOT / path).is_symlink())
    ]


def incremental_files(kind, snap_dir):
    """Select changed result files or complete changed W&B runs; advance state only after packing."""
    source = (
        wandb_runs_dir("pack") if kind == "logs" else repo_path(os.environ.get("MIRROROPD_RESULTS_SOURCE", "outputs"))
    )
    if not source.is_dir():
        raise FileNotFoundError(source)
    state_path = snap_dir / f"{kind}_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    state, candidates = {}, []
    max_bytes = int(os.environ.get("MIRROROPD_MAX_FILE_MB", "100")) * 1024 * 1024
    if max_bytes <= 0:
        raise ValueError("MAX_FILE_MB must be positive")
    skipped = 0
    for directory, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [name for name in dirs if name not in CACHE_DIRS and name != "checkpoints"]
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(source)
            if path.is_symlink() or name == ".DS_Store" or name.startswith("._"):
                continue
            if kind == "logs" and not rel.parts[0].startswith("offline-run-"):
                continue
            stat = path.stat()
            if kind == "results" and stat.st_size > max_bytes:
                skipped += 1
                continue
            key = rel.as_posix()
            state[key] = [stat.st_size, stat.st_mtime_ns]
            candidates.append((path, key))
    changed = {key for key, signature in state.items() if previous.get(key) != signature}
    if kind == "logs":
        runs = {PurePosixPath(key).parts[0].removesuffix(".wandb") for key in changed}
        selected = [
            (path, "wandb/" + key)
            for path, key in candidates
            if PurePosixPath(key).parts[0].removesuffix(".wandb") in runs
        ]
    else:
        selected = [(path, "results/" + key) for path, key in candidates if key in changed]
    if skipped:
        print(f"Skipped {skipped} result files larger than {max_bytes} bytes")
    return selected, state_path, state


def clean_archives(kind, archive, keep, older_than=None):
    """Remove same-kind archives in the same directory, excluding current outputs and newer downloads."""
    candidates = {archive}
    candidates.update(archive.parent.glob(PREFIXES[kind] + "*.tar.gz"))
    candidates.update(archive.parent.glob(PREFIXES[kind] + "*.tar.gz.part.[0-9]*"))
    candidates.update(archive.parent.glob(archive.name + ".part.[0-9]*"))
    removed = 0
    for path in candidates:
        if path in keep or not path.is_file():
            continue
        if not path.name.endswith(".tar.gz") and not path.suffix[1:].isdigit():
            continue
        if older_than is not None and path.stat().st_mtime_ns >= older_than:
            continue
        path.unlink()
        removed += 1
    if removed:
        print(f"Removed {removed} old {kind} archive file(s) from {archive.parent}")


def pack(args, snap_dir):
    """Publish a complete archive before cleaning old packages or advancing incremental state."""
    state_path, state = None, None
    if args.kind == "code":
        files = source_files(args.paths)
    else:
        if args.paths:
            raise ValueError("log/result pack accepts only an optional archive path")
        files, state_path, state = incremental_files(args.kind, snap_dir)
    if not files:
        print("Nothing to package: no new or updated files")
        return
    name = args.archive or f"{PREFIXES[args.kind]}_{datetime.now():%Y%m%d_%H%M%S_%f}.tar.gz"
    archive = Path(name).expanduser()
    if not archive.is_absolute():
        archive = snap_dir / archive
    if not archive.name.endswith(".tar.gz"):
        archive = archive.with_name(archive.name + ".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if args.kind != "code" and (archive.exists() or list(archive.parent.glob(archive.name + ".part.*"))):
        raise FileExistsError(f"Choose a new archive name: {archive}")
    chunk_size = int(os.environ.get("MIRROROPD_CHUNK_SIZE", "10485760"))
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be positive")
    with tempfile.TemporaryDirectory(prefix=".mirroropd-pack-", dir=archive.parent) as temp:
        packed = Path(temp) / archive.name
        with tarfile.open(packed, "w:gz", format=tarfile.PAX_FORMAT) as output:
            for path, member in files:
                output.add(path, arcname=member, recursive=False)
        if packed.stat().st_size <= chunk_size:
            packed.replace(archive)
            outputs = [archive]
        else:
            outputs = []
            part_count = (packed.stat().st_size + chunk_size - 1) // chunk_size
            width = max(3, len(str(part_count - 1)))
            with packed.open("rb") as stream:
                while data := stream.read(chunk_size):
                    part = Path(temp) / f"{archive.name}.part.{len(outputs):0{width}d}"
                    part.write_bytes(data)
                    outputs.append(archive.parent / part.name)
            for path in outputs:
                (Path(temp) / path.name).replace(path)
    if args.kind in ("code", "logs"):
        clean_archives(args.kind, archive, set(outputs))
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        next_state = state_path.with_suffix(".tmp")
        next_state.write_text(json.dumps(state, sort_keys=True) + "\n")
        next_state.replace(state_path)
    print(f"Packed {len(files)} files. Manually transfer all of:")
    for path in outputs:
        print(path)


def find_archive(kind, supplied, snap_dir):
    """Prefer an explicit input, then the newest matching download, checkout file or snapshot."""
    if supplied:
        return Path(supplied).expanduser().absolute()
    for directory in (Path.home() / "Downloads", Path.cwd(), snap_dir):
        candidates = list(directory.glob(PREFIXES[kind] + "*.tar.gz"))
        candidates += list(directory.glob(PREFIXES[kind] + "*.tar.gz.part.000*"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    raise FileNotFoundError(f"No {PREFIXES[kind]} archive found; pass its path explicitly")


def unpack(args, snap_dir):
    """Replace archived source paths, or merge logs/results, after validating the full archive."""
    source = find_archive(args.kind, args.archive, snap_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.kind != "code" and args.paths or len(args.paths) > 1:
        raise ValueError("only code unpack accepts an optional destination")
    source_mtime = source.stat().st_mtime_ns
    consumed = [source]
    archive_base = source
    with tempfile.TemporaryDirectory(prefix="mirroropd-unpack-") as temp:
        archive = source
        if ".tar.gz.part." in source.name:
            prefix = source.name.rsplit(".part.", 1)[0] + ".part."
            parts = sorted(
                source.parent.glob(prefix + "[0-9]*"),
                key=lambda path: int(path.name.rsplit(".", 1)[1]),
            )
            consumed = parts
            archive_base = source.with_name(source.name.rsplit(".part.", 1)[0])
            archive = Path(temp) / "joined.tar.gz"
            with archive.open("wb") as output:
                for index, part in enumerate(parts):
                    if int(part.name.rsplit(".", 1)[1]) != index:
                        raise ValueError(f"Missing archive part: {prefix}{index:03d}")
                    with part.open("rb") as stream:
                        shutil.copyfileobj(stream, output)
        extracted = Path(temp) / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = []
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                if not path.parts or path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                # Hope settings belong to the receiving machine, including when applying old bundles.
                if args.kind == "code" and path.parts[0] == "hope":
                    continue
                if args.kind == "code" and excluded(path):
                    raise ValueError(f"Protected runtime path in source archive: {member.name}")
                expected = "wandb" if args.kind == "logs" else "results"
                if args.kind != "code" and path.parts[0] != expected:
                    raise ValueError(f"Expected {expected}/ in archive, found: {member.name}")
                members.append(member)
            bundle.extractall(extracted, members=members, filter="data")
        if args.kind == "code":
            destination = repo_path(args.paths[0]) if args.paths else ROOT
            destination.mkdir(parents=True, exist_ok=True)
            for child in extracted.iterdir():
                target = destination / child.name
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
                shutil.move(str(child), target)
        else:
            logical = "wandb" if args.kind == "logs" else "results"
            destination = (
                wandb_runs_dir("unpack")
                if args.kind == "logs"
                else repo_path(os.environ.get("MIRROROPD_RESULTS_DEST", logical))
            )
            incoming = extracted / logical
            if not incoming.is_dir():
                raise ValueError(f"Archive contains no {logical}/ directory")
            shutil.copytree(incoming, destination, dirs_exist_ok=True)
            if args.kind == "logs" and os.environ.get("MIRROROPD_WANDB_SYNC", "0") == "1":
                for run in sorted(incoming.glob("offline-run-*")):
                    if run.is_dir():
                        subprocess.run(["wandb", "sync", str(destination / run.name)], check=True)
    print(f"Unpacked into {destination}")
    if args.kind == "code":
        clean_archives(args.kind, archive_base, set(consumed), older_than=source_mtime)
        if os.environ.get("KEEP_ARCHIVE_AFTER_UNPACK", "0") != "1":
            for path in consumed:
                path.unlink(missing_ok=True)
            print(f"Removed {len(consumed)} consumed source archive file(s)")
        else:
            print("Kept input archive(s): KEEP_ARCHIVE_AFTER_UNPACK=1")
    else:
        print("Kept the input archive(s)")


def main():
    """Expose the same pack/unpack interface through each shell entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=PREFIXES)
    parser.add_argument("action", choices=("pack", "unpack"))
    parser.add_argument("archive", nargs="?", default="")
    parser.add_argument(
        "paths",
        nargs="*",
        help="code pack: top-level sources; code unpack: destination",
    )
    args = parser.parse_args()
    snap_dir = repo_path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap"))
    if args.action == "pack":
        pack(args, snap_dir)
    else:
        unpack(args, snap_dir)


if __name__ == "__main__":
    main()
