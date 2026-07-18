#!/usr/bin/env python3
"""Content-addressed run-cache primitives for independent KERC owners.

This module owns only byte identities and atomic cache receipts. It deliberately
contains no source parsing, semantic reconstruction, admission, or claim logic,
so producer and verifier implementations remain independent.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


CACHE_POLICY = "project_theseus_kerc_content_addressed_run_cache_v1"
CACHE_SCHEMA_VERSION = "1.0.0"
OBJECT_CACHE_POLICY = "project_theseus_kerc_content_addressed_object_cache_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: Path) -> dict[str, Any]:
    """Hash one file or a complete directory tree with relative-path binding."""
    path = path.resolve()
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries: list[dict[str, Any]] = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": candidate.relative_to(path).as_posix(),
                "sha256": sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            }
        )
    digest = hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()
    return {
        "kind": "directory_tree",
        "path": str(path),
        "sha256": digest,
        "file_count": len(entries),
        "size_bytes": sum(int(item["size_bytes"]) for item in entries),
    }


def dependency_bindings(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for identity, path in sorted(paths.items()):
        row = hash_path(path)
        row["id"] = identity
        rows.append(row)
    return rows


def cache_key(*, role: str, dependencies: Iterable[dict[str, Any]]) -> str:
    payload = {
        "policy": CACHE_POLICY,
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": role,
        "dependencies": list(dependencies),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def object_key(
    *, role: str, layer: str, dependencies: Mapping[str, Any]
) -> str:
    payload = {
        "policy": OBJECT_CACHE_POLICY,
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": role,
        "layer": layer,
        "dependencies": dependencies,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ContentObjectCache:
    """Integrity-checked transactional JSON object cache.

    Namespaces prevent producer and verifier objects from sharing authority. A
    corrupt row is deleted on read and therefore becomes a normal cache miss.
    """

    def __init__(
        self, path: Path, *, namespace: str, commit_interval: int = 128
    ) -> None:
        if commit_interval < 1:
            raise ValueError("commit_interval must be positive")
        self.path = path
        self.namespace = namespace
        self.commit_interval = commit_interval
        self._pending_writes = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                namespace TEXT NOT NULL,
                object_key TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_json BLOB NOT NULL,
                PRIMARY KEY (namespace, object_key)
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.flush()
        self.connection.close()

    def flush(self) -> None:
        if self._pending_writes:
            self.connection.commit()
            self._pending_writes = 0

    def __enter__(self) -> "ContentObjectCache":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def get(self, key: str) -> Any | None:
        row = self.connection.execute(
            "SELECT payload_sha256, payload_json FROM objects "
            "WHERE namespace = ? AND object_key = ?",
            (self.namespace, key),
        ).fetchone()
        if row is None:
            return None
        expected_sha256, encoded = str(row[0]), bytes(row[1])
        observed_sha256 = hashlib.sha256(encoded).hexdigest()
        if observed_sha256 != expected_sha256:
            self.delete(key)
            return None
        try:
            envelope = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.delete(key)
            return None
        if canonical_json(envelope).encode("utf-8") != encoded:
            self.delete(key)
            return None
        if not isinstance(envelope, dict) or envelope.get("policy") != OBJECT_CACHE_POLICY:
            self.delete(key)
            return None
        if envelope.get("schema_version") != CACHE_SCHEMA_VERSION:
            self.delete(key)
            return None
        if envelope.get("namespace") != self.namespace or envelope.get("object_key") != key:
            self.delete(key)
            return None
        return envelope.get("payload")

    def put(self, key: str, payload: Any) -> None:
        envelope = {
            "policy": OBJECT_CACHE_POLICY,
            "schema_version": CACHE_SCHEMA_VERSION,
            "namespace": self.namespace,
            "object_key": key,
            "payload": payload,
        }
        encoded = canonical_json(envelope).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self.connection.execute(
            "INSERT OR REPLACE INTO objects "
            "(namespace, object_key, payload_sha256, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (self.namespace, key, digest, encoded),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.commit_interval:
            self.flush()

    def delete(self, key: str) -> None:
        self.connection.execute(
            "DELETE FROM objects WHERE namespace = ? AND object_key = ?",
            (self.namespace, key),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.commit_interval:
            self.flush()

    def count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM objects WHERE namespace = ?", (self.namespace,)
        ).fetchone()
        return int(row[0]) if row else 0


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def receipt_path(cache_root: Path, *, role: str, key: str) -> Path:
    return cache_root / role / f"{key}.json"


def output_bindings(outputs: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for identity, path in sorted(outputs.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "id": identity,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def load_receipt(
    cache_root: Path,
    *,
    role: str,
    dependencies: list[dict[str, Any]],
    outputs: Mapping[str, Path],
    result_output_id: str,
) -> dict[str, Any] | None:
    key = cache_key(role=role, dependencies=dependencies)
    path = receipt_path(cache_root, role=role, key=key)
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (
            receipt.get("policy") != CACHE_POLICY
            or receipt.get("schema_version") != CACHE_SCHEMA_VERSION
            or receipt.get("role") != role
            or receipt.get("cache_key_sha256") != key
            or receipt.get("dependencies") != dependencies
        ):
            return None
        observed_outputs = output_bindings(outputs)
        if receipt.get("outputs") != observed_outputs:
            return None
        result_row = next(
            row for row in observed_outputs if row["id"] == result_output_id
        )
        result_path = Path(result_row["path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if receipt.get("result_payload_sha256") != hashlib.sha256(
            canonical_json(result).encode("utf-8")
        ).hexdigest():
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        return None


def publish_receipt(
    cache_root: Path,
    *,
    role: str,
    dependencies: list[dict[str, Any]],
    outputs: Mapping[str, Path],
    result_output_id: str,
) -> Path:
    key = cache_key(role=role, dependencies=dependencies)
    bound_outputs = output_bindings(outputs)
    result_row = next(row for row in bound_outputs if row["id"] == result_output_id)
    result = json.loads(Path(result_row["path"]).read_text(encoding="utf-8"))
    receipt = {
        "policy": CACHE_POLICY,
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": role,
        "cache_key_sha256": key,
        "dependencies": dependencies,
        "outputs": bound_outputs,
        "result_output_id": result_output_id,
        "result_payload_sha256": hashlib.sha256(
            canonical_json(result).encode("utf-8")
        ).hexdigest(),
    }
    path = receipt_path(cache_root, role=role, key=key)
    _atomic_json(path, receipt)
    return path
