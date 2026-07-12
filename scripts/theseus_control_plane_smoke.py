#!/usr/bin/env python3
"""Focused smokes for Theseus Control Plane v1.

These tests stay local and cheap: they do not launch training, benchmarks,
public calibration, or model growth.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402
import theseus_control_plane as tcp  # noqa: E402
import theseus_generated_artifact_gc as gc  # noqa: E402
from theseus_archive_resolver import ARCHIVE_POINTER_POLICY, resolve_archived_path  # noqa: E402


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="theseus_control_plane_smoke_") as tmp_raw:
        tmp = Path(tmp_raw)
        results.append(test_report_write_through_no_overwrite_loss(tmp))
        results.append(test_duplicate_heavy_action_lease(tmp))
        results.append(test_stale_report_detection(tmp))
        results.append(test_archive_pointer_resolution(tmp))
        results.append(test_gc_quarantine(tmp))
    passed = all(row["passed"] for row in results)
    report = {
        "policy": "project_theseus_control_plane_smoke_v1",
        "created_utc": tcp.now(),
        "trigger_state": "GREEN" if passed else "RED",
        "summary": {
            "passed": passed,
            "case_count": len(results),
            "failed_count": sum(1 for row in results if not row["passed"]),
        },
        "cases": results,
        "external_inference_calls": 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


def test_report_write_through_no_overwrite_loss(tmp: Path) -> dict[str, Any]:
    db = tmp / "evidence.sqlite"
    report = tmp / "volatile_report.json"
    snapshot_dir = tmp / "snapshots"
    old_limit = report_evidence_store.MAX_PAYLOAD_BYTES
    old_snapshot_dir = report_evidence_store.DEFAULT_SNAPSHOT_DIR
    try:
        report_evidence_store.MAX_PAYLOAD_BYTES = 64
        report_evidence_store.DEFAULT_SNAPSHOT_DIR = snapshot_dir
        first = {
            "policy": "smoke_report_v1",
            "created_utc": "2026-01-01T00:00:00Z",
            "trigger_state": "GREEN",
            "summary": {"version": 1},
            "large": "a" * 256,
        }
        second = {
            "policy": "smoke_report_v1",
            "created_utc": "2026-01-01T00:01:00Z",
            "trigger_state": "YELLOW",
            "summary": {"version": 2},
            "large": "b" * 256,
        }
        first_row = report_evidence_store.write_json_report(report, first, db_path=db)
        second_row = report_evidence_store.write_json_report(report, second, db_path=db)
        summary = report_evidence_store.evidence_store_summary(db)
        current = report_evidence_store.current_report_index(db, [report])
        snapshots = list(snapshot_dir.rglob("*.json"))
        passed = bool(
            first_row.get("run_id")
            and second_row.get("run_id")
            and first_row["content_hash"] != second_row["content_hash"]
            and summary["stored_run_count"] == 2
            and summary["latest_path_count"] == 1
            and summary["snapshot_count"] == 2
            and current["current_unstored_count"] == 0
            and current["current_truncated_without_snapshot_count"] == 0
            and len(snapshots) == 2
        )
        return {
            "id": "report_write_through_no_overwrite_loss",
            "passed": passed,
            "summary": summary,
            "current_index": {
                "current_unstored_count": current["current_unstored_count"],
                "current_truncated_without_snapshot_count": current["current_truncated_without_snapshot_count"],
            },
            "snapshot_count": len(snapshots),
        }
    finally:
        report_evidence_store.MAX_PAYLOAD_BYTES = old_limit
        report_evidence_store.DEFAULT_SNAPSHOT_DIR = old_snapshot_dir


def test_duplicate_heavy_action_lease(tmp: Path) -> dict[str, Any]:
    db = tmp / "actions.sqlite"
    conn = report_evidence_store.connect(db)
    try:
        tcp.ensure_action_schema(conn)
        gates = {
            "heavy_code_work_allowed": {"passed": True, "evidence": {}},
            "public_calibration_allowed": {"passed": False, "evidence": {}},
            "model_growth_allowed": {"passed": False, "evidence": {}},
            "candidate_promotion_allowed": {"passed": False, "evidence": {}},
        }
        first = tcp.request_action(
            conn,
            action_key="smoke_heavy_code_worker",
            action_type="heavy_code_worker",
            command="python scripts/code_lm_train_once_fanout.py --execute",
            gates=gates,
            active_workers=[],
            lease_seconds=3600,
        )
        second = tcp.request_action(
            conn,
            action_key="smoke_heavy_code_worker",
            action_type="heavy_code_worker",
            command="python scripts/code_lm_train_once_fanout.py --execute",
            gates=gates,
            active_workers=[],
            lease_seconds=3600,
        )
        summary = tcp.action_ledger_summary(conn)
    finally:
        conn.close()
    passed = bool(
        first["status"] == "reserved"
        and second["status"] == "blocked"
        and second["reason"] == "active_action_lease_exists"
        and summary["active_lease_count"] == 1
    )
    return {
        "id": "duplicate_heavy_action_lease_blocked",
        "passed": passed,
        "first_status": first["status"],
        "second_status": second["status"],
        "second_reason": second["reason"],
        "active_lease_count": summary["active_lease_count"],
    }


def test_stale_report_detection(tmp: Path) -> dict[str, Any]:
    report = tmp / "stale.json"
    report.write_text(
        json.dumps(
            {
                "policy": "stale_smoke_v1",
                "created_utc": "2020-01-01T00:00:00Z",
                "trigger_state": "GREEN",
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    spec = tcp.ReportSpec("stale_smoke", report, "smoke", max_age_hours=1.0)
    row = tcp.build_report_records((spec,))[0]
    passed = bool(row["exists"] and row["stale"] and not row["missing"] and row["age_hours"] and row["age_hours"] > 1.0)
    return {
        "id": "stale_report_detection",
        "passed": passed,
        "stale": row["stale"],
        "age_hours": row["age_hours"],
    }


def test_archive_pointer_resolution(tmp: Path) -> dict[str, Any]:
    archived = tmp / "archive" / "checkpoint.json"
    pointer = tmp / "reports" / "checkpoint.json"
    archived.parent.mkdir(parents=True, exist_ok=True)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text('{"policy":"checkpoint_smoke","ok":true}', encoding="utf-8")
    pointer.write_text(
        json.dumps(
            {
                "policy": ARCHIVE_POINTER_POLICY,
                "archive_path": str(archived),
                "original_path": str(pointer),
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_archived_path(pointer)
    passed = resolved == archived
    return {
        "id": "archive_pointer_resolution",
        "passed": passed,
        "resolved": str(resolved),
        "expected": str(archived),
    }


def test_gc_quarantine(tmp: Path) -> dict[str, Any]:
    source = tmp / "old_temp_file.tmp"
    quarantine = tmp / "quarantine"
    source.write_text("temporary", encoding="utf-8")
    row = {
        "path": str(source),
        "bytes": source.stat().st_size,
        "mib": 0.0,
        "age_hours": 48.0,
        "reason": "safe_generated_suffix_tmp",
    }
    args = type(
        "Args",
        (),
        {
            "delete": False,
            "compress_json": False,
            "quarantine_root": str(quarantine),
        },
    )()
    actions = gc.apply_actions([row], args)
    target = Path(actions[0].get("quarantine_path", ""))
    passed = bool(actions and actions[0].get("status") == "quarantined" and not source.exists() and target.exists())
    return {
        "id": "gc_quarantine",
        "passed": passed,
        "status": actions[0].get("status") if actions else "",
        "quarantine_path": str(target),
    }


if __name__ == "__main__":
    raise SystemExit(main())
