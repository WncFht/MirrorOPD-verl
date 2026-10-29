# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only cloud history comparison, executed by the isolated uploader Python.

No init, sync, deletion or cloud writes occur here. A complete history match
proves metric coverage, not delivery of console, system metrics or artifacts.
"""

import argparse
import json
import math
import os
import struct
import zlib
from collections import defaultdict
from pathlib import Path


class IncompleteRecord(ValueError):
    """The immutable snapshot ends inside a physical or logical record."""


def records(path):
    """Read W&B v0 LevelDB framing with CRC checks, without opening for writes."""
    with Path(path).open("rb") as stream:
        if stream.read(7) != struct.pack("<4sHB", b":W&B", 0xBEE1, 0):
            raise ValueError("Invalid W&B transaction header")
        fragments = None
        while True:
            remaining = 32768 - stream.tell() % 32768
            if remaining < 7:
                padding = stream.read(remaining)
                if any(padding):
                    raise ValueError("Nonzero W&B block padding")
            header = stream.read(7)
            if not header:
                if fragments is not None:
                    raise IncompleteRecord("Unfinished fragmented record")
                return
            if len(header) != 7:
                raise IncompleteRecord("Unfinished physical record header")
            crc, length, kind = struct.unpack("<IHB", header)
            if length > 32768 - stream.tell() % 32768 and stream.tell() % 32768:
                raise ValueError("Record crosses a block boundary")
            data = stream.read(length)
            if len(data) != length:
                raise IncompleteRecord("Unfinished physical record body")
            if kind not in (1, 2, 3, 4) or zlib.crc32(bytes([kind]) + data) & 0xFFFFFFFF != crc:
                raise ValueError("Invalid record type or checksum; do not upload corrupt logs")
            if kind in (1, 2):
                if fragments is not None:
                    raise ValueError("Missing final fragment")
                if kind == 1:
                    yield data
                else:
                    fragments = bytearray(data)
            else:
                if fragments is None:
                    raise ValueError("Missing first fragment")
                fragments.extend(data)
                if kind == 4:
                    yield bytes(fragments)
                    fragments = None


def history_item(item):
    if "value" in item.DESCRIPTOR.oneofs_by_name:
        kind = item.WhichOneof("value")
        if kind:
            value = getattr(item, kind)
            return None if kind == "none" else json.loads(value) if kind == "json" else value
    if item.value_json:
        return json.loads(item.value_json)
    raise ValueError("Unsupported history value encoding")


def snapshot(path):
    from wandb.proto.wandb_internal_pb2 import Record

    rows, identity, exit_code = {}, None, None
    partial = False
    try:
        for data in records(path):
            record = Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                current = {k: getattr(record.run, k) for k in ("run_id", "entity", "project")}
                if identity and identity != current:
                    raise ValueError("Multiple run identities in one snapshot")
                identity = current
            elif kind == "exit":
                exit_code = record.exit.exit_code
            elif kind == "history":
                row = {}
                for item in record.history.item:
                    keys = [item.key] if item.key else list(item.nested_key)
                    if not keys:
                        raise ValueError("Missing history key")
                    node = row
                    for key in keys[:-1]:
                        node = node.setdefault(key, {})
                    node[keys[-1]] = history_item(item)
                step = row.get("_step")
                if not isinstance(step, int) or isinstance(step, bool) or step in rows:
                    raise ValueError("Missing, noninteger or duplicate local _step")
                rows[step] = row
    except IncompleteRecord:
        partial = True
    if not identity or not identity["run_id"]:
        raise ValueError("No run identity in snapshot")
    return {"identity": identity, "rows": rows, "partial": partial, "exit_code": exit_code}


def matches(expected, actual):
    """Cloud JSON may round floats; require every recorded non-runtime field."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and matches(value, actual[key]) for key, value in expected.items() if key != "_runtime"
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(matches(a, b) for a, b in zip(expected, actual, strict=True))
        )
    if isinstance(expected, float | int) and not isinstance(expected, bool):
        if isinstance(expected, float) and not math.isfinite(expected):
            return (
                actual is None
                or actual == expected
                or (isinstance(actual, float) and math.isnan(expected) and math.isnan(actual))
            )
        return (
            isinstance(actual, float | int)
            and not isinstance(actual, bool)
            and math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8)
        )
    return type(expected) is type(actual) and expected == actual


def compare(local, remote_rows):
    """Count duplicate cloud rows but never replay an existing partial history."""
    seen, count, conflicts = set(), 0, set()
    for row in remote_rows:
        step = row.get("_step")
        count += 1
        if step in local and not matches(local[step], row):
            conflicts.add(step)
        seen.add(step)
    missing = set(local) - seen
    return {
        "local_rows": len(local),
        "cloud_rows": count,
        "duplicate_rows": count - len(seen),
        "missing_rows": len(missing),
        "conflicting_steps": sorted(conflicts)[:10],
        "covered": not missing and not conflicts,
    }


def compare_run(local, run, history_line_count=None):
    """Fetch sparse schemas separately: an unfiltered scan expands thousands of null columns per row."""
    coordinates = {
        step: {key: row[key] for key in ("_step", "_timestamp") if key in row} for step, row in local.items()
    }
    proof = compare(coordinates, run.scan_history(keys=["_step", "_timestamp"], page_size=100000, use_cache=False))
    if history_line_count is not None:
        proof["cloud_history_line_count"] = history_line_count
        proof["unaccounted_rows"] = history_line_count - proof["cloud_rows"]
    if proof["duplicate_rows"] or proof.get("unaccounted_rows"):
        proof["covered"] = False
        return proof
    if not proof["covered"]:
        return proof
    schemas = defaultdict(dict)
    for step, row in local.items():
        schemas[tuple(sorted(key for key in row if key != "_runtime"))][step] = row
    for index, (keys, rows) in enumerate(schemas.items(), 1):
        print(f"Verifying history schema {index}/{len(schemas)}: {len(rows)} rows, {len(keys)} fields", flush=True)
        # SDK expands sparse columns inside each step window. Wide evaluation
        # schemas need explicit observed-step bounds, not thousands of null rows.
        if len(keys) > 256 and len(rows) < 100:
            remote = (
                row
                for step in rows
                for row in run.scan_history(keys=list(keys), min_step=step, max_step=step + 1, page_size=10000)
            )
        else:
            remote = run.scan_history(keys=list(keys), min_step=min(rows), max_step=max(rows) + 1, page_size=10000)
        details = compare(rows, remote)
        if not details["covered"]:
            proof.update(
                covered=False, missing_rows=details["missing_rows"], conflicting_steps=details["conflicting_steps"]
            )
            return proof
    return proof


def check(path, entity, project, base_url):
    import wandb

    local = snapshot(path)
    print(
        f"Local snapshot: {len(local['rows'])} history rows; partial={local['partial']}; exit={local['exit_code']}",
        flush=True,
    )
    identity = local["identity"]
    entity = entity or identity["entity"] or os.environ.get("MIRROROPD_UPLOAD_ENTITY", "")
    project = project or identity["project"] or os.environ.get("MIRROROPD_UPLOAD_PROJECT", "")
    if not entity or not project:
        raise ValueError("Set WANDB_ENTITY and WANDB_PROJECT explicitly; credential default destinations are not used")
    target = dict(entity=entity, project=project, base_url=base_url, run_id=identity["run_id"])
    result = dict(target=target, partial=local["partial"], exit_code=local["exit_code"])
    if local["partial"] or local["exit_code"] is None:
        result.update(action="incomplete", reason="Wait for a snapshot with complete records and an exit record")
        return result
    api = wandb.Api(overrides={"base_url": base_url}, timeout=30)
    # A permission, DNS or authentication error must never be interpreted as absence.
    found = list(api.runs(f"{entity}/{project}", filters={"name": identity["run_id"]}, per_page=1))
    if found:
        run = found[0]
        # Core's parquet/history view can hide repeated physical rows. Check
        # the server's resume counter too, using the pinned SDK service API.
        counts = api._service_api.execute_graphql(
            """query SyncHistoryCount($entity: String!, $project: String!, $run: String!) {
              project(name: $project, entityName: $entity) {
                run(name: $run) { historyLineCount }
              }
            }""",
            {"entity": entity, "project": project, "run": identity["run_id"]},
        )
        line_count = counts["project"]["run"]["historyLineCount"]
        if not isinstance(line_count, int) or line_count < 0:
            raise ValueError("Cloud did not provide a valid history row count")
        result.update(compare_run(local["rows"], run, line_count))
        result["url"] = run.url
        result["cloud_state"] = run.state
        if result["covered"]:
            result.update(action="covered", reason="Cloud history covers every local row; no SDK replay")
        else:
            result.update(
                action="blocked",
                reason="Existing cloud history has duplicate/unaccounted/missing/conflicting rows. "
                "Full replay would append duplicates; "
                "preserve the raw log and repair this run separately. RETRY=1 does not bypass this check.",
            )
    else:
        result.update(action="upload", reason="Run absent in accessible project; complete snapshot may be uploaded")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--entity", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--base-url", default="https://api.wandb.ai")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.file, args.entity, args.project, args.base_url)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
