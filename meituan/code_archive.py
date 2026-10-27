# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Exchange committed source through offline Git bundles; also usable as a bootstrap script."""

import argparse
import fcntl
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# BEGIN OBSERVE IMPORT
try:
    from meituan import sync_observe as observe
except ModuleNotFoundError:
    import sync_observe as observe
# END OBSERVE IMPORT

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "archive_MirrorOPD-verl"
EXPORT_REF = "refs/archive/export"
INCOMING_REF = "refs/archive/incoming"


def git(repo, *args, check=True, input_data=None):
    """Run only local Git operations, without recursive submodule updates or deployment hooks."""
    result = observe.run(
        ["git", "-c", "submodule.recurse=false", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        capture_output=True,
        input=input_data,
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


def source_artifact_reason(path, size):
    """Keep source bundles bounded while retaining small reports and test fixtures."""
    if size > 2 * 1024 * 1024:
        return "file exceeds the 2 MiB source limit"
    if not path.startswith("examples/mirroropd/docs/"):
        return None
    if re.search(r"\.(?:jsonl(?:\.gz)?|json\.gz|log|parquet|pt|npy|npz)$", path):
        return "raw experiment artifact belongs in local storage"
    if re.search(r"/papers/[^/]+\.(?:tex|pdf|zip|gz|txt)$", path):
        return "downloaded paper source belongs in local storage"
    if "/data/" in path:
        name = path.rsplit("/", 1)[-1]
        if re.fullmatch(r"(?:evaluation|initial-state-metrics|lineage-state-metrics|cases)\.json", name):
            return "per-state experiment details belong in local storage"
        if re.fullmatch(r"(?:rollouts?(?:[-_].*)?|fresh(?:[-_].*)?|task-.*)\.json", name):
            return "rollout details belong in local storage"
        if size > 1024 * 1024 and name.endswith((".json", ".csv")):
            return "experiment data exceeds the 1 MiB summary limit"
    return None


def check_source_artifacts(repo, revision="HEAD", *, staged=False, history=False):
    """Inspect Git blobs, so unstaged edits cannot hide staged or historical data."""
    if staged and history:
        raise ValueError("Choose either --staged or --history")
    entries = []
    if history:
        objects = git(repo, "rev-list", "--objects", revision).stdout.splitlines()
        names = {}
        for line in objects:
            oid, _, path = line.partition(b" ")
            names[oid] = os.fsdecode(path)
        sizes = git(
            repo,
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            input_data=b"\n".join(names) + b"\n",
        ).stdout.splitlines()
        for line in sizes:
            oid, kind, size = line.split()
            if kind == b"blob":
                entries.append((names[oid], int(size)))
    else:
        tree = value(repo, "write-tree") if staged else revision
        for entry in git(repo, "ls-tree", "-r", "-l", "-z", tree).stdout.split(b"\0"):
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            _, kind, _, size = metadata.split()
            if kind == b"blob":
                entries.append((os.fsdecode(name), int(size)))
    errors = []
    for path, size in entries:
        reason = source_artifact_reason(path, size)
        if reason:
            errors.append(f"{path}: {reason} ({size} bytes)")
    if errors:
        detail = "\n".join(errors[:20])
        raise ValueError(
            f"Source artifact policy rejected {len(errors)} blob(s):\n{detail}\n"
            "Keep raw data locally; deleting it in a later commit does not remove it from bundle history."
        )
    return len(entries)


def write_bootstrap(path):
    """Embed the shared journal so first-time receivers still need only one Python script."""
    source = Path(__file__).read_text()
    before, _, remaining = source.partition("# BEGIN OBSERVE IMPORT\n")
    _, _, after = remaining.partition("# END OBSERVE IMPORT\n")
    if not after:
        raise ValueError("Generate full packs from the installed meituan/code_archive.py, not the bootstrap copy")
    helper = Path(observe.__file__).read_text()
    embedded = f'import types\nobserve = types.ModuleType("sync_observe")\nexec({helper!r}, observe.__dict__)\n'
    path.write_text(before + embedded + after)


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


def archive_stamp(path):
    """Read supported automatic filenames, including older names without microseconds."""
    match = re.fullmatch(
        rf"{re.escape(PREFIX)}_(?:[A-Za-z0-9_-]+_)?"
        r"(\d{8}_\d{6}(?:_\d{6})?)\.(bundle|tar\.gz|tgz|zip)(?:\.part\.\d+)?",
        path.name,
    )
    if not match:
        return None
    try:
        stamp = datetime.strptime(match[1], "%Y%m%d_%H%M%S_%f" if len(match[1]) > 15 else "%Y%m%d_%H%M%S")
        return stamp, match[2]
    except ValueError:
        return None


def cleanup(directory, target, keep, current):
    """Prune only older bundles for this peer, preserving future deliveries and custom names."""
    pattern = re.compile(rf"{re.escape(PREFIX + '_' + target)}_\d{{8}}_\d{{6}}_\d{{6}}\.bundle(?:\.part\.\d+)?")
    cutoff = archive_stamp(current)
    current_mtime = min(path.stat().st_mtime_ns for path in keep)
    for path in directory.glob(f"{PREFIX}_{target}_*.bundle*"):
        if path in keep or not path.is_file() or path.is_symlink() or not pattern.fullmatch(path.name):
            continue
        stamp = archive_stamp(path)
        older = stamp[0] < cutoff[0] if stamp and cutoff else path.stat().st_mtime_ns < current_mtime
        if older:
            path.unlink()
            observe.note("Removed old bundle", path=str(path))


def cleanup_legacy_archives(source, destination=None):
    """Retire old source packages in the input folder, checkout root and configured snapshot folder."""
    folders = {source.parent.resolve()}
    protected = set()
    if destination is not None:
        destination = destination.resolve()
        snap = Path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap")).expanduser()
        folders.update((destination, (snap if snap.is_absolute() else destination / snap).resolve()))
        tracked = git(destination, "ls-files", "-z").stdout.split(b"\0")
        protected = {destination / os.fsdecode(name) for name in tracked if name}
    incoming = archive_stamp(source)
    source_mtime = source.stat().st_mtime_ns
    fallback_time = datetime.fromtimestamp(source_mtime / 1e9)
    removed, kept = [], []
    for folder in sorted(folders):
        for path in sorted(folder.glob(f"{PREFIX}_*")):
            if not re.search(r"\.(?:tar\.gz|tgz|zip)(?:\.part\.\d+)?$", path.name):
                continue
            stamp = archive_stamp(path)
            reason = None
            if path.is_symlink() or not path.is_file():
                reason = "not a regular archive file"
            elif path in protected:
                reason = "tracked by Git"
            elif stamp is None:
                reason = "not an automatic source archive name"
            elif incoming and stamp[0] >= incoming[0]:
                reason = "same age or newer than the received bundle"
            elif not incoming and (stamp[0] >= fallback_time or path.stat().st_mtime_ns >= source_mtime):
                reason = "custom bundle: archive is not provably older by both filename and modification time"
            if reason:
                kept.append({"path": str(path), "reason": reason})
                observe.note("Kept source archive", path=str(path), reason=reason)
                continue
            path.unlink()
            removed.append(str(path))
            observe.note("Removed obsolete legacy source archive", path=str(path))
    observe.update(cleanup_directories=[str(folder) for folder in sorted(folders)], kept_legacy_archives=kept)
    return removed


def pack(repo, snap, target, supplied="", full=False):
    require_checkout(repo)
    packed_ref, confirmed_ref = refs(target)
    with locked(repo):
        commit = value(repo, "rev-parse", "HEAD^{commit}")
        check_hope_removed(repo, commit)
        with observe.stage("source.artifacts"):
            check_source_artifacts(repo, commit)
            checked = check_source_artifacts(repo, commit, history=True)
            observe.note("Source artifact policy passed", blobs=checked)
        if git(repo, "status", "--porcelain", "--untracked-files=normal").stdout:
            print("Note: packaging committed HEAD only; uncommitted and untracked files are not included.")
        base = ref(repo, confirmed_ref)
        observe.update(head=commit, baseline=base, target=target)
        if base and not ancestor(repo, base, commit):
            raise ValueError("HEAD diverges from the peer baseline; resolve history or use a separate TARGET")
        if base == commit and not full:
            observe.update(outcome="no_changes")
            print(f"Nothing to package: {target} already has {commit}. No new pack or ack is needed.")
            return []
        full = full or base is None
        observe.note(
            "Bundle plan", mode="full" if full else "incremental", head=commit, baseline=None if full else base
        )
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
                with observe.stage("bundle.create", mode="full" if full else "incremental"):
                    git(repo, "bundle", "create", str(bundle), *revisions)
            finally:
                git(repo, "update-ref", "-d", EXPORT_REF)
            if full:
                bootstrap = archive.parent / "archive_bootstrap.py"
                staged_script = Path(temp) / "archive_bootstrap.py"
                write_bootstrap(staged_script)
                staged_script.replace(bootstrap)
            with observe.stage("bundle.publish", archive=str(archive), chunk_size=chunk_size):
                outputs = publish(bundle, archive, chunk_size)
                observe.update(
                    outputs=[str(path) for path in outputs], bytes=sum(path.stat().st_size for path in outputs)
                )
        # This ref describes an actually published pack, never a later HEAD.
        git(repo, "update-ref", packed_ref, commit)
        observe.note("Last packed ref updated; peer confirmation is unchanged", packed=commit, confirmed=base)
        cleanup(archive.parent, target, set(outputs), archive)
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
        observe.update(packed=commit, confirmed=known, head=value(repo, "rev-parse", "HEAD"))
        print(f"Confirmed {target} has {known} (last packed commit: {commit})")


def find_archive(supplied, snap):
    if supplied:
        source = Path(supplied).expanduser().resolve()
        observe.note("Using explicit archive", path=str(source))
        return source
    for directory in (Path.home() / "Downloads", Path.cwd(), snap):
        candidates = list(directory.glob(f"{PREFIX}_*.bundle"))
        candidates += list(directory.glob(f"{PREFIX}_*.bundle.part.000"))
        if candidates:
            source = max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()
            observe.note("Automatically selected archive", searched_directory=str(directory), selected=str(source))
            return source
    raise FileNotFoundError("No source bundle found; pass ARCHIVE=/path/to/file.bundle explicitly")


@contextmanager
def joined(source):
    observe.update(archive=str(source))
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.name.endswith(".bundle"):
        observe.note("Reading single bundle", bytes=source.stat().st_size)
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
        with observe.stage("bundle.join", parts=len(parts)), bundle.open("wb") as output:
            for index, part in enumerate(parts):
                if int(part.name[len(prefix) :]) != index:
                    raise ValueError(f"Missing archive part: {prefix}{index:03d}")
                observe.note(
                    "Joining part", part=str(part), index=index + 1, total=len(parts), bytes=part.stat().st_size
                )
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        yield bundle, parts


def receive(repo, bundle):
    """Verify prerequisites and import objects without updating the working branch."""
    with observe.stage(
        "bundle.verify", hint="Missing prerequisites require a full bundle; truncated packs require all parts."
    ):
        git(repo, "bundle", "verify", str(bundle))
    heads = value(repo, "bundle", "list-heads", str(bundle)).splitlines()
    if len(heads) != 1 or heads[0].split()[1] != EXPORT_REF:
        raise ValueError("Expected exactly one archive export ref in the bundle")
    with observe.stage("bundle.import", repository=str(repo)):
        git(repo, "fetch", "--no-tags", "--no-recurse-submodules", str(bundle), f"+{EXPORT_REF}:{INCOMING_REF}")
    commit = value(repo, "rev-parse", f"{INCOMING_REF}^{{commit}}")
    check_hope_removed(repo, commit)
    observe.update(incoming_commit=commit)
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
    with locked(repo), observe.stage("checkout.update", repository=str(repo)):
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
        observe.update(head_before=current, peer_before=previous)
        if previous and not ancestor(repo, previous, commit) and not ancestor(repo, commit, previous):
            raise ValueError("Peer history diverged; use a separate TARGET for this history")
        if not ancestor(repo, current, commit) and not ancestor(repo, commit, current):
            raise ValueError("Source and destination histories diverged; no merge or overwrite was performed")
        changes = git(repo, "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)hope").stdout
        if changes:
            listing = changes.decode(errors="replace").strip()
            observe.note("Uncommitted source paths (full list)", paths=listing, console=False)
            preview = "\n".join(listing.splitlines()[:30])
            raise ValueError(
                "Destination has uncommitted source changes; commit or save them before unpack.\n"
                f"{preview}\nInspect: git --no-pager diff --stat HEAD -- . ':(exclude)hope'\n"
                "A checkout previously overwritten by tar may need legacy migration; do not reset or commit it blindly."
            )
        if ancestor(repo, current, commit) and current != commit:
            # Git can overwrite ignored files on checkout. Preserve any newly tracked local paths as well.
            added = git(repo, "diff", "--name-only", "--diff-filter=A", "--no-renames", "-z", current, commit).stdout
            for name in added.split(b"\0"):
                if name:
                    path = repo / os.fsdecode(name)
                    if path.exists() or path.is_symlink():
                        raise ValueError(f"New tracked path would overwrite a local file: {path}")
            with (
                observe.stage("checkout.fast_forward", hint="Input is retained. Preserve local edits before retrying."),
                preserve_hope(repo),
            ):
                git(repo, "merge", "--ff-only", "--no-autostash", commit)
        known = remember_peer(repo, target, commit)
        observe.update(head_after=value(repo, "rev-parse", "HEAD"), peer_after=known)
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
            observe.update(legacy_backup=str(backup))
        observe.update(head_after=commit, peer_after=commit)
        print(f"Initialized checkout at {commit}; {target} is known to have this commit")


def unpack(destination, source, target):
    refs(target)
    observe.update(destination=str(destination), target=target)
    with joined(source) as (bundle, consumed):
        if (destination / ".git").exists() or (destination / ".git").is_symlink():
            update_checkout(destination, bundle, target)
        else:
            with observe.stage("checkout.bootstrap", destination=str(destination)):
                bootstrap(destination, bundle, target)
    print(f"Unpacked into {destination}. Run make archive-ack on the sender to confirm its last pack.")
    if os.environ.get("KEEP_ARCHIVE_AFTER_UNPACK", "0") != "1":
        with observe.stage("archive.cleanup", files=len(consumed)):
            # Keep this input available for retry if retiring an old package fails.
            legacy = cleanup_legacy_archives(source, destination)
            for path in consumed:
                path.unlink()
            observe.update(removed_input_archives=[str(path) for path in consumed], removed_legacy_archives=legacy)
        observe.note("Removed consumed source archives", files=len(consumed))
        print(f"Removed {len(consumed)} consumed bundle file(s) and {len(legacy)} older legacy source archive file(s)")
    else:
        observe.skip("archive.cleanup", "KEEP_ARCHIVE_AFTER_UNPACK=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "unpack", "ack", "check-source"))
    parser.add_argument("--staged", action="store_true", help="check the index instead of HEAD")
    parser.add_argument("--history", action="store_true", help="check all objects reachable from HEAD")
    parser.add_argument("archive", nargs="?", default="")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.action == "check-source":
        try:
            if args.archive or args.paths:
                raise ValueError("check-source accepts only --staged or --history")
            count = check_source_artifacts(ROOT, staged=args.staged, history=args.history)
            print(f"Source artifact policy passed: {count} blobs")
            return
        except (OSError, ValueError) as error:
            parser.exit(1, f"archive: {error}\n")
    if args.staged or args.history:
        parser.error("--staged and --history require check-source")
    target = os.environ.get("MIRROROPD_ARCHIVE_TARGET", "peer")
    snap = Path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap")).expanduser()
    snap = snap if snap.is_absolute() else ROOT / snap
    try:
        destination = Path(args.paths[0]).expanduser() if args.action == "unpack" and args.paths else ROOT
        destination = destination if destination.is_absolute() else ROOT / destination
        receiving_snap = Path(os.environ.get("MIRROROPD_SNAP_DIR", "transfer_snap")).expanduser()
        receiving_snap = receiving_snap if receiving_snap.is_absolute() else destination / receiving_snap
        default_log = receiving_snap / "operations" if args.action == "unpack" else snap / "operations"
        log_dir = Path(os.environ.get("MIRROROPD_OPERATION_DIR") or default_log).expanduser()
        with observe.operation(
            log_dir,
            f"archive-{args.action}",
            repository=str(destination if args.action == "unpack" else ROOT),
            entrypoint=str(Path(__file__).resolve()),
            target=target,
            archive=args.archive,
            full=os.environ.get("MIRROROPD_ARCHIVE_FULL", "0"),
            keep_archive=os.environ.get("KEEP_ARCHIVE_AFTER_UNPACK", "0"),
        ):
            with observe.stage(f"archive.{args.action}"):
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
                    unpack(destination.resolve(), find_archive(args.archive, snap), target)
    except KeyboardInterrupt:
        parser.exit(130, "archive: interrupted; see the operation log before retrying\n")
    except (OSError, ValueError) as error:
        parser.exit(1, observe.redact(f"archive: {error}\n"))


if __name__ == "__main__":
    main()
