"""Resolve manifest-backed archived artifacts.

Large historical report/checkpoint artifacts may be moved out of ``reports/``
latest-view space. A small JSON pointer remains at the original path so humans
and control-plane reports can see where the retained artifact lives.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_POINTER_POLICY = "project_theseus_archived_artifact_pointer_v1"


def resolve_archived_path(path: str | Path) -> Path:
    """Return the archived target when *path* is an archive pointer."""
    original = resolve(path)
    pointer = read_json(original, {})
    if pointer.get("policy") != ARCHIVE_POINTER_POLICY:
        return original
    archive_path = pointer.get("archive_path")
    if not archive_path:
        return original
    resolved = resolve(str(archive_path))
    return resolved if resolved.exists() else original


def read_json_follow_pointer(path: str | Path, default: Any = None) -> Any:
    return read_json(resolve_archived_path(path), default)


def read_text_follow_pointer(path: str | Path, default: str = "") -> str:
    try:
        target = resolve_archived_path(path)
        if target.suffix == ".gz":
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                return handle.read()
        return target.read_text(encoding="utf-8")
    except Exception:
        return default


def iter_jsonl_follow_pointer(path: str | Path) -> Iterator[Any]:
    """Yield JSONL rows from *path*, transparently following archive pointers."""
    target = resolve_archived_path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("policy") == ARCHIVE_POINTER_POLICY and row.get("archive_path"):
                yield from iter_jsonl_follow_pointer(str(row["archive_path"]))
                # Append-only ledgers may receive new rows after retention has
                # replaced their historical prefix with an archive pointer.
                # Replay the archived prefix, then continue through the live
                # tail instead of silently dropping post-archive events.
                continue
            yield row


def read_jsonl_follow_pointer(path: str | Path) -> list[Any]:
    return list(iter_jsonl_follow_pointer(path))


def is_archive_pointer(path: str | Path) -> bool:
    return read_json(resolve(path), {}).get("policy") == ARCHIVE_POINTER_POLICY


def read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
