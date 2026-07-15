#!/usr/bin/env python3
"""Append-only, fail-closed consumption records for frozen functional surfaces."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY = "project_theseus_functional_surface_consumption_v1"


class ConsumptionError(RuntimeError):
    """Raised when a frozen surface identity cannot be consumed safely."""


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def reserve_once(
    registry_path: Path,
    *,
    stage: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    if not stage or not identity:
        raise ConsumptionError("consumption stage and identity are required")
    key = stable_hash({"policy": POLICY, "stage": stage, "identity": identity})
    with _locked_registry(registry_path) as rows:
        if any(row.get("consumption_key") == key for row in rows):
            raise ConsumptionError(f"surface identity already consumed or reserved: {stage}:{key}")
        created_utc = now()
        reservation_id = stable_hash(
            {
                "consumption_key": key,
                "created_utc": created_utc,
                "pid": os.getpid(),
                "nonce": os.urandom(16).hex(),
            }
        )
        event = {
            "policy": POLICY,
            "event": "reserved",
            "created_utc": created_utc,
            "stage": stage,
            "consumption_key": key,
            "reservation_id": reservation_id,
            "identity": identity,
        }
        _append_locked(registry_path, event)
        return event


def complete_reservation(
    registry_path: Path,
    reservation: dict[str, Any],
    *,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    return _close_reservation(
        registry_path,
        reservation,
        event_name="completed",
        detail={"artifact": artifact},
    )


def fail_reservation(
    registry_path: Path,
    reservation: dict[str, Any],
    *,
    fault: str,
) -> dict[str, Any]:
    return _close_reservation(
        registry_path,
        reservation,
        event_name="failed",
        detail={"fault": fault[:2048]},
    )


def read_registry(registry_path: Path) -> list[dict[str, Any]]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_name(registry_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            return _read_rows(registry_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def require_completed_artifact(
    registry_path: Path,
    *,
    stage: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in read_registry(registry_path)
        if row.get("event") == "completed"
        and row.get("stage") == stage
        and (row.get("artifact") or {}).get("sha256") == artifact_sha256
    ]
    if len(matches) != 1:
        raise ConsumptionError(
            f"expected one completed artifact for {stage}:{artifact_sha256}; found {len(matches)}"
        )
    return matches[0]


class _locked_registry:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        self.handle: Any = None

    def __enter__(self) -> list[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return _read_rows(self.path)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _close_reservation(
    registry_path: Path,
    reservation: dict[str, Any],
    *,
    event_name: str,
    detail: dict[str, Any],
) -> dict[str, Any]:
    key = str(reservation.get("consumption_key") or "")
    reservation_id = str(reservation.get("reservation_id") or "")
    if not key or not reservation_id:
        raise ConsumptionError("invalid reservation receipt")
    with _locked_registry(registry_path) as rows:
        matching = [row for row in rows if row.get("consumption_key") == key]
        if not matching or matching[0].get("event") != "reserved":
            raise ConsumptionError("reservation is missing from consumption registry")
        if matching[0].get("reservation_id") != reservation_id:
            raise ConsumptionError("reservation identity mismatch")
        if len(matching) != 1:
            raise ConsumptionError("reservation is already closed")
        event = {
            "policy": POLICY,
            "event": event_name,
            "created_utc": now(),
            "stage": reservation["stage"],
            "consumption_key": key,
            "reservation_id": reservation_id,
            **detail,
        }
        _append_locked(registry_path, event)
        return event


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConsumptionError(
                    f"malformed consumption registry row {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or row.get("policy") != POLICY:
                raise ConsumptionError(
                    f"invalid consumption registry row {line_number}"
                )
            rows.append(row)
    return rows


def _append_locked(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
