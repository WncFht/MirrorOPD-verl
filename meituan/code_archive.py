# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exchange committed source through offline Git bundles; also usable as a bootstrap script."""

import argparse
import fcntl
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "archive_MirrorOPD-verl"
EXPORT_REF = "refs/archive/export"
INCOMING_REF = "refs/archive/incoming"


def git(repo, *args, check=True):
    """Run only local Git operations, without recursive submodule updates or deployment hooks."""
    result = subprocess.run(
        ["git", "-c", "submodule.recurse=false", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        capture_output=True,
    )
    if check and result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip() or f"git {args[0]} failed")
    return result


def value(repo, *args):
    return git(repo, *args).stdout.decode().strip()


def ref(repo, name):
    result = git(repo, "rev-parse", "--verify", name, check=False)
    return result.stdout.decode().strip() if result.returncode == 0 else None


def refs(target):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
        raise ValueError("TARGET must contain only letters, digits, underscores or hyphens")
    prefix = f"refs/archive/peers/{target}"
    return f"{prefix}/packed", f"{prefix}/confirmed"


def git_dir(repo):
    return Path(value(repo, "rev-parse", "--absolute-git-dir"))


def require_checkout(repo):
    if Path(value(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve():
        raise ValueError(f"Expected a Git checkout at {repo}")


@contextmanager
def locked(repo):
    """Serialize pack, unpack and ack in this checkout; OS locks survive interrupted commands safely."""
    with (git_dir(repo) / "archive.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another archive operation is running in this checkout") from error
        yield


def ancestor(repo, older, newer):
    result = git(repo, "merge-base", "--is-ancestor", older, newer, check=False)
    if result.returncode not in (0, 1):
        raise ValueError(result.stderr.decode(errors="replace"))
    return result.returncode == 0


def remember_peer(repo, target, commit):
    """Keep the newest known peer version when importing an old bundle or repeating an ack."""
    _, confirmed = refs(target)
    previous = ref(repo, confirmed)
    if previous and ancestor(repo, commit, previous):
        return previous
    if previous and not ancestor(repo, previous, commit):
        raise ValueError("Peer history diverged; use a separate TARGET for this history")
    git(repo, "update-ref", confirmed, commit)
    return commit


def check_hope_removed(repo, commit):
    if git(repo, "ls-tree", "-r", "--name-only", commit, "--", "hope").stdout:
        raise ValueError("Commit still tracks hope/. Add /hope/ to .gitignore, git rm -r --cached hope, then commit")


def publish(packed, archive, chunk_size):
    """Stage every part before publishing; existing names are never overwritten."""
    if archive.exists() or list(archive.parent.glob(archive.name + ".part.*")):
        raise FileExistsError(f"Choose a new archive name: {archive}")
    if packed.stat().st_size <= chunk_size:
        packed.replace(archive)
        return [archive]
    outputs = []
    with packed.open("rb") as stream:
        while data := stream.read(chunk_size):
            part = packed.with_name(f"{archive.name}.part.{len(outputs):03d}")
            part.write_bytes(data)
            outputs.append(archive.parent / part.name)
    published = []
    try:
        for output in outputs:
            (packed.parent / output.name).replace(output)
            published.append(output)
    except OSError:
        for output in published:
            output.unlink()
        raise
    return outputs


def cleanup(directory, target, keep):
    """Only prune this peer's automatic bundle names; never prune logs, custom names or other peers."""
    pattern = re.compile(rf"{re.escape(PREFIX + '_' + target)}_\d{{8}}_\d{{6}}_\d{{6}}\.bundle(?:\.part\.\d+)?")
    for path in directory.glob(f"{PREFIX}_{target}_*.bundle*"):
        if path not in keep and path.is_file() and pattern.fullmatch(path.name):
            path.unlink()


def pack(repo, snap, target, supplied="", full=False):
    require_checkout(repo)
    packed_ref, confirmed_ref = refs(target)
    with locked(repo):
        commit = value(repo, "rev-parse", "HEAD^{commit}")
        check_hope_removed(repo, commit)
        if git(repo, "status", "--porcelain", "--untracked-files=normal").stdout:
            print("Note: packaging committed HEAD only; uncommitted and untracked files are not included.")
        base = ref(repo, confirmed_ref)
        if base and not ancestor(repo, base, commit):
            raise ValueError("HEAD diverges from the peer baseline; resolve history or use a separate TARGET")
        if base == commit and not full:
            print(f"Nothing to package: {target} already has {commit}. No new pack or ack is needed.")
            return []
        full = full or base is None
        name = supplied or f"{PREFIX}_{target}_{datetime.now():%Y%m%d_%H%M%S_%f}.bundle"
        archive = Path(name).expanduser()
        archive = archive if archive.is_absolute() else snap / archive
        if archive.suffix != ".bundle":
            raise ValueError("Source archives must use a .bundle filename")
        chunk_size = int(os.environ.get("MIRROROPD_CHUNK_SIZE", "10485760"))
        if chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".archive-pack-", dir=archive.parent) as temp:
            bundle = Path(temp) / "source.bundle"
            git(repo, "update-ref", EXPORT_REF, commit)
            try:
                revisions = [EXPORT_REF] if full else [EXPORT_REF, f"^{base}"]
                git(repo, "bundle", "create", str(bundle), *revisions)
            finally:
                git(repo, "update-ref", "-d", EXPORT_REF)
            if full:
                bootstrap = archive.parent / "archive_bootstrap.py"
                staged_script = Path(temp) / "archive_bootstrap.py"
                shutil.copy2(__file__, staged_script)
                staged_script.replace(bootstrap)
            outputs = publish(bundle, archive, chunk_size)
        # This ref describes an actually published pack, never a later HEAD.
        git(repo, "update-ref", packed_ref, commit)
        cleanup(archive.parent, target, set(outputs))
        print(f"Packed {'full' if full else 'incremental'} commit {commit}; base: {base if not full else 'none'}")
        print("Manually transfer all of:")
        for output in outputs:
            print(output)
        if full:
            print(f"For first installation or old tar-based receivers, also transfer: {bootstrap}")
        print(f"After successful unpack, run on THIS sender: make archive-ack TARGET={target}")
        print("Acknowledge before generating another pack if you intend to confirm this delivery.")
        return outputs


def ack(repo, target):
    require_checkout(repo)
    packed_ref, _ = refs(target)
    with locked(repo):
        commit = ref(repo, packed_ref)
        if not commit:
            raise ValueError(f"No successful pack recorded for TARGET={target}")
        known = remember_peer(repo, target, commit)
        print(f"Confirmed {target} has {known} (last packed commit: {commit})")


def find_archive(supplied, snap):
    if supplied:
        return Path(supplied).expanduser().resolve()
    for directory in (Path.home() / "Downloads", Path.cwd(), snap):
        candidates = list(directory.glob(f"{PREFIX}_*.bundle"))
        candidates += list(directory.glob(f"{PREFIX}_*.bundle.part.000"))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()
    raise FileNotFoundError("No source bundle found; pass ARCHIVE=/path/to/file.bundle explicitly")


@contextmanager
def joined(source):
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.name.endswith(".bundle"):
        yield source, [source]
        return
    match = re.fullmatch(r"(.+\.bundle)\.part\.\d+", source.name)
    if not match:
        raise ValueError("Expected a .bundle or .bundle.part.000 file; old tar packages use transfer.py code unpack")
    prefix = match[1] + ".part."
    parts = sorted(
        (path for path in source.parent.glob(prefix + "*") if path.name[len(prefix) :].isdigit()),
        key=lambda path: int(path.name[len(prefix) :]),
    )
    with tempfile.TemporaryDirectory(prefix="archive-join-") as temp:
        bundle = Path(temp) / "source.bundle"
        with bundle.open("wb") as output:
            for index, part in enumerate(parts):
                if int(part.name[len(prefix) :]) != index:
                    raise ValueError(f"Missing archive part: {prefix}{index:03d}")
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        yield bundle, parts


def receive(repo, bundle):
    """Verify prerequisites and import objects without updating the working branch."""
    git(repo, "bundle", "verify", str(bundle))
    heads = value(repo, "bundle", "list-heads", str(bundle)).splitlines()
    if len(heads) != 1 or heads[0].split()[1] != EXPORT_REF:
        raise ValueError("Expected exactly one archive export ref in the bundle")
    git(repo, "fetch", "--no-tags", "--no-recurse-submodules", str(bundle), f"+{EXPORT_REF}:{INCOMING_REF}")
    commit = value(repo, "rev-parse", f"{INCOMING_REF}^{{commit}}")
    check_hope_removed(repo, commit)
    return commit


@contextmanager
def preserve_hope(repo):
    """Keep machine-local Hope files through the one-time tracked-to-ignored migration."""
    tracked = git(repo, "ls-files", "--", "hope").stdout
    if not tracked:
        yield
        return
    if git(repo, "diff", "--cached", "--quiet", "--", "hope", check=False).returncode:
        raise ValueError("Commit or unstage changes under hope/ before receiving its removal")
    local = repo / "hope"
    backup = git_dir(repo) / "archive-hope-backup"
    if backup.exists() or backup.is_symlink():
        raise ValueError(f"Restore the previous Hope backup before retrying: {backup}")
    saved = local.exists() or local.is_symlink()
    if saved:
        shutil.move(str(local), backup)
        print(f"Temporarily saved local Hope settings at {backup}")
    try:
        git(repo, "restore", "--source=HEAD", "--worktree", "--", "hope")
        yield
    finally:
        if local.is_symlink() or local.is_file():
            local.unlink()
        elif local.is_dir():
            shutil.rmtree(local)
        if saved:
            shutil.move(str(backup), local)


def update_checkout(repo, bundle, target):
    require_checkout(repo)
    with locked(repo):
        backup = git_dir(repo) / "archive-hope-backup"
        if backup.exists() or backup.is_symlink():
            raise ValueError(
                f"An interrupted migration left a Hope backup. Restore it to hope/ before retrying: {backup}"
            )
        if not ref(repo, "HEAD"):
            raise ValueError("Destination has an unborn branch; initialize in a directory without .git")
        commit = receive(repo, bundle)
        current = value(repo, "rev-parse", "HEAD")
        previous = ref(repo, refs(target)[1])
        if previous and not ancestor(repo, previous, commit) and not ancestor(repo, commit, previous):
            raise ValueError("Peer history diverged; use a separate TARGET for this history")
        if not ancestor(repo, current, commit) and not ancestor(repo, commit, current):
            raise ValueError("Source and destination histories diverged; no merge or overwrite was performed")
        changes = git(repo, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)hope").stdout
        if changes:
            raise ValueError("Destination has uncommitted source changes; commit or save them before unpack")
        if ancestor(repo, current, commit) and current != commit:
            # Git can overwrite ignored files on checkout. Preserve any newly tracked local paths as well.
            added = git(repo, "diff", "--name-only", "--diff-filter=A", "--no-renames", "-z", current, commit).stdout
            for name in added.split(b"\0"):
                if name:
                    path = repo / os.fsdecode(name)
                    if path.exists() or path.is_symlink():
                        raise ValueError(f"New tracked path would overwrite a local file: {path}")
            with preserve_hope(repo):
                git(repo, "merge", "--ff-only", "--no-autostash", commit)
        known = remember_peer(repo, target, commit)
        print(f"Checkout at {value(repo, 'rev-parse', 'HEAD')}; {target} is known to have {known}")


def bootstrap(destination, bundle, target):
    """Install a real checkout over a legacy tar tree, retaining backups of replaced local paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".archive-init-", dir=destination.parent) as temp:
        staging = Path(temp) / "checkout"
        staging.mkdir()
        git(staging, "init", "--initial-branch=main")
        commit = receive(staging, bundle)  # Incremental bundles fail here before touching the destination.
        git(staging, "checkout", "-B", "main", commit)
        remember_peer(staging, target, commit)
        destination.mkdir(parents=True, exist_ok=True)
        backup = destination.parent / f".{destination.name}-before-bundle-{datetime.now():%Y%m%d_%H%M%S_%f}"
        entries = git(staging, "ls-tree", "-rz", commit).stdout.split(b"\0")
        paths = []
        for entry in entries:
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            if metadata.split()[0] == b"160000":
                continue  # Submodule worktrees stay local; a parent bundle carries only their commit IDs.
            relative = Path(os.fsdecode(name))
            for parent in reversed(relative.parents):
                candidate = destination / parent
                if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
                    raise ValueError(f"Local parent obstructs checkout: {candidate}; resolve it and retry")
            paths.append(relative)
        if any((destination / path).exists() or (destination / path).is_symlink() for path in paths):
            print(f"Backing up replaced legacy files to {backup}")
        for relative in paths:
            incoming, output = staging / relative, destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.is_symlink():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), saved)
            shutil.move(str(incoming), output)
        # Publish .git last, so a failed installation can be retried as a bootstrap.
        shutil.move(str(staging / ".git"), destination / ".git")
        if backup.exists():
            print(f"Preserved replaced legacy files at {backup}")
        print(f"Initialized checkout at {commit}; {target} is known to have this commit")


def unpack(destination, source, target):
    refs(target)
    with joined(source) as (bundle, consumed):
        if (destination / ".git").exists() or (destination / ".git").is_symlink():
            update_checkout(destination, bundle, target)
        else:
            bootstrap(destination, bundle, target)
    print(f"Unpacked into {destination}. Run make archive-ack on the sender to confirm its last pack.")
    if os.environ.get("KEEP_ARCHIVE_AFTER_UNPACK", "0") != "1":
        for path in consumed:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "unpack", "ack"))
    parser.add_argument("archive", nargs="?", default="")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    target = os.environ.get("MIRROROPD_ARCHIVE_TARGET", "peer")
    snap = Path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap")).expanduser()
    snap = snap if snap.is_absolute() else ROOT / snap
    try:
        if args.action == "pack":
            if args.paths:
                raise ValueError("ARCHIVE_SOURCES is no longer supported: bundles carry the committed project")
            pack(ROOT, snap, target, args.archive, os.environ.get("MIRROROPD_ARCHIVE_FULL", "0") == "1")
        elif args.action == "ack":
            if args.archive or args.paths:
                raise ValueError("ack uses the last successful pack; no commit or archive argument is needed")
            ack(ROOT, target)
        else:
            if len(args.paths) > 1:
                raise ValueError("unpack accepts only one destination")
            destination = Path(args.paths[0]).expanduser() if args.paths else ROOT
            destination = destination if destination.is_absolute() else ROOT / destination
            unpack(destination.resolve(), find_archive(args.archive, snap), target)
    except (OSError, ValueError) as error:
        parser.exit(1, f"archive: {error}\n")


if __name__ == "__main__":
    main()
