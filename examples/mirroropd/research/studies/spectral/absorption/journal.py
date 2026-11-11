# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Atomic group checkpoints for exact gradient probes."""

from examples.mirroropd.research.core.artifacts import file_hash, fingerprint
from examples.mirroropd.research.core.io import dump, read


class GroupJournal:
    """Resume only whole groups with identical model, bank, matrix and backend identity."""

    def __init__(self, directory, identity):
        self.directory = directory
        self.identity = fingerprint(identity)
        protocol = directory / "journal-identity.json"
        if protocol.exists() and read(protocol) != self.identity:
            raise ValueError("Gradient journal identity changed; use a new attempt")
        dump(protocol, self.identity)
        self.completed = {}
        for path in sorted((directory / "groups").glob("*.json")):
            record = read(path)
            if record["identity"] != self.identity:
                raise ValueError(f"Group identity mismatch: {path}")
            tensor = directory / "domain-gradient-sums" / (record["group"].replace(":", "_") + ".pt")
            if not tensor.exists() or tensor.stat().st_size != record["tensor_bytes"]:
                raise ValueError(f"Missing or incomplete group gradient sum: {tensor}")
            if "tensor_sha256" not in record:
                raise ValueError(f"Legacy group lacks checksum; verify/migrate explicitly: {path}")
            if file_hash(tensor) != record["tensor_sha256"]:
                raise ValueError(f"Group gradient checksum mismatch: {tensor}")
            self.completed[record["group"]] = record

    def save(self, group, projections, summaries):
        tensor = self.directory / "domain-gradient-sums" / (group.replace(":", "_") + ".pt")
        record = dict(
            identity=self.identity,
            group=group,
            projections=projections,
            summaries=summaries,
            tensor_bytes=tensor.stat().st_size,
            tensor_sha256=file_hash(tensor),
        )
        dump(self.directory / "groups" / (group.replace(":", "_") + ".json"), record)
        self.completed[group] = record
