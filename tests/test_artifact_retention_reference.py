from __future__ import annotations

import json
import hashlib
import gzip
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import artifact_retention_reference as reference
import theseus_artifact_retention as retention
import theseus_artifact_retention_replay_gate as replay_gate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture_root(tmp_path: Path, monkeypatch) -> Path:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    monkeypatch.setattr(reference, "CHECKPOINT_ROOT", checkpoints)
    return checkpoints


def test_operational_config_and_route_evidence_protect_referenced_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoints = fixture_root(tmp_path, monkeypatch)
    protected = checkpoints / "active" / "weights.npz"
    route_protected = checkpoints / "route" / "weights.npz"
    unprotected = checkpoints / "old" / "weights.npz"
    for path in (protected, route_protected, unprotected):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.parent.name.encode("utf-8") * 100)
    write_json(tmp_path / "configs" / "runtime.json", {"checkpoint": "checkpoints/active/weights.npz"})
    write_json(tmp_path / "reports" / "route.json", {"trigger_state": "GREEN", "checkpoint": "checkpoints/route"})
    registry = {
        "route_evidence_contracts": [
            {"requirements": [{"path": "reports/route.json"}]}
        ]
    }

    index = reference.build_checkpoint_reference_index(registry)
    by_path = {row["path"]: row for row in index["file_records"]}

    assert by_path["checkpoints/active/weights.npz"]["protected"]
    assert by_path["checkpoints/route/weights.npz"]["protected"]
    assert not by_path["checkpoints/old/weights.npz"]["protected"]


def test_archive_candidates_exclude_protected_and_recent_payloads(tmp_path: Path, monkeypatch) -> None:
    checkpoints = fixture_root(tmp_path, monkeypatch)
    old = checkpoints / "old" / "weights.npz"
    recent = checkpoints / "recent" / "weights.npz"
    protected = checkpoints / "protected" / "weights.npz"
    for path in (old, recent, protected):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
    now_value = time.time()
    os.utime(old, (now_value - 72 * 3600, now_value - 72 * 3600))
    os.utime(protected, (now_value - 72 * 3600, now_value - 72 * 3600))
    rows = []
    for path, is_protected in ((old, False), (recent, False), (protected, True)):
        rows.append(
            {
                "path": reference.rel(path),
                "bytes": path.stat().st_size,
                "protected": is_protected,
                "mtime_utc": "",
            }
        )
    index = {"file_records": rows}

    candidates = reference.checkpoint_archive_candidates(
        index,
        min_bytes=1,
        min_age_hours=24,
        target_hot_bytes=2 * 1024,
        now_timestamp=now_value,
    )

    assert [row["path"] for row in candidates] == ["checkpoints/old/weights.npz"]


def test_checkpoint_deduplication_atomically_preserves_digest_and_paths(tmp_path: Path, monkeypatch) -> None:
    checkpoints = fixture_root(tmp_path, monkeypatch)
    left = checkpoints / "a" / "weights.npz"
    right = checkpoints / "b" / "weights.npz"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    payload = b"same-weights" * 100_000
    left.write_bytes(payload)
    right.write_bytes(payload)
    index = {
        "file_records": [
            {"path": reference.rel(left), "protected": True},
            {"path": reference.rel(right), "protected": False},
        ]
    }

    result = reference.deduplicate_checkpoint_payloads(index, execute=True, min_bytes=1)

    assert result["state"] == "GREEN"
    assert result["deduplicated_file_count"] == 1
    assert left.read_bytes() == payload
    assert right.read_bytes() == payload
    assert reference.same_inode(left, right)


def test_budget_unique_storage_bytes_does_not_double_count_hardlinks(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"x" * 8192)
    os.link(left, right)
    left_row = retention.budget_file_row(left)
    right_row = retention.budget_file_row(right)

    assert retention.unique_storage_bytes([left_row, right_row]) == left_row["allocated_bytes"]


def test_manifest_pointer_repair_requires_hash_and_never_overwrites_live_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(retention, "ROOT", tmp_path)
    archive = tmp_path / "archive" / "weights.npz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"verified-weights")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    pointer = tmp_path / "checkpoints" / "old" / "weights.npz.archive-pointer.json"
    manifest = {
        "entries": [
            {
                "original_path": "checkpoints/old/weights.npz",
                "pointer_path": "checkpoints/old/weights.npz.archive-pointer.json",
                "archive_path": "archive/weights.npz",
                "status": "archived",
                "bytes": len(archive.read_bytes()),
                "sha256": digest,
            }
        ]
    }

    repaired = retention.repair_manifest_pointers(manifest, execute=True)

    assert repaired["state"] == "GREEN"
    assert repaired["repaired_count"] == 1
    assert retention.is_pointer(pointer)

    live = tmp_path / "reports" / "live.json"
    live.parent.mkdir(parents=True)
    live.write_text('{"real":"payload"}\n', encoding="utf-8")
    conflict_manifest = {
        "entries": [
            {
                "original_path": "reports/live.json",
                "pointer_path": "reports/live.json",
                "archive_path": "archive/weights.npz",
                "status": "archived",
                "bytes": len(archive.read_bytes()),
                "sha256": digest,
            }
        ]
    }

    conflict = retention.repair_manifest_pointers(conflict_manifest, execute=True)

    assert conflict["state"] == "RED"
    assert conflict["live_path_conflict_count"] == 1
    assert live.read_text(encoding="utf-8") == '{"real":"payload"}\n'


def test_renewable_live_report_supersedes_pointer_without_losing_archive_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(retention, "ROOT", tmp_path)
    monkeypatch.setattr(replay_gate, "ROOT", tmp_path)
    encoded = (json.dumps({"generation": 1}, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    archive = tmp_path / "archive" / "report.json.gz"
    archive.parent.mkdir(parents=True)
    with gzip.open(archive, "wb") as handle:
        handle.write(encoded)
    live = tmp_path / "reports" / "theseus_artifact_budget_gate_current.json"
    write_json(live, {"generation": 2})
    entries = [
        {
            "original_path": "reports/theseus_artifact_budget_gate_current.json",
            "pointer_path": "reports/theseus_artifact_budget_gate_current.json",
            "archive_path": "archive/report.json.gz",
            "status": "archived",
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    ]

    reconciled, records = retention.reconcile_renewable_live_entries(entries)

    assert reconciled[0]["status"] == "archived_superseded_by_live_regeneration"
    assert records[0]["archive_hash_verified"] is True
    check = replay_gate.verify_action(
        reconciled[0],
        {"entries": reconciled},
        tmp_path / "manifest.json",
        manifest_entry_mode=True,
    )
    assert check["passed"] is True
    assert check["live_regeneration_verified"] is True


def test_nonrenewable_live_report_is_not_silently_superseded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(retention, "ROOT", tmp_path)
    archive = tmp_path / "archive" / "report.json"
    write_json(archive, {"generation": 1})
    live = tmp_path / "reports" / "unregistered_current.json"
    write_json(live, {"generation": 2})
    entries = [
        {
            "original_path": "reports/unregistered_current.json",
            "pointer_path": "reports/unregistered_current.json",
            "archive_path": "archive/report.json",
            "status": "archived",
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    ]

    reconciled, records = retention.reconcile_renewable_live_entries(entries)

    assert reconciled == entries
    assert records == []


def test_replay_records_are_digest_bound_and_constant_cardinality(tmp_path: Path) -> None:
    checks = [
        {
            "check_id": f"check-{index}",
            "passed": True,
            "original_path": f"reports/source-{index}.json",
            "pointer_path": f"reports/source-{index}.json",
            "archive_path": f"archive/source-{index}.json.gz",
            "expected_sha256": f"expected-{index}",
            "decoded_sha256": f"expected-{index}",
        }
        for index in range(100)
    ]

    digest = replay_gate.digest_rows(checks)
    records = replay_gate.build_records(
        checks,
        tmp_path / "reports" / "retention-manifest.json",
        checks_sha256=digest,
    )

    assert replay_gate.digest_rows(checks) == digest
    assert replay_gate.digest_rows([*checks[:-1], {**checks[-1], "passed": False}]) != digest
    assert set(records) == {
        "compressed_artifact_records",
        "compression_receipts",
        "proof_contract_receipt_records",
        "claim_records",
        "artifact_graph_records",
        "evidence_transition_records",
        "defeater_records",
    }
    assert all(len(rows) == 1 for rows in records.values())
    for rows in records.values():
        assert rows[0]["replay_checks_sha256"] == digest
        assert rows[0]["eligible_action_count"] == 100
        assert rows[0]["passed_replay_count"] == 100
        assert rows[0]["failed_replay_count"] == 0


def test_replay_records_fail_closed_when_any_check_fails(tmp_path: Path) -> None:
    checks = [
        {
            "check_id": "check-failed",
            "passed": False,
            "original_path": "reports/source.json",
            "pointer_path": "reports/source.json",
            "archive_path": "archive/source.json.gz",
            "expected_sha256": "expected",
            "decoded_sha256": "actual",
        }
    ]

    records = replay_gate.build_records(
        checks,
        tmp_path / "reports" / "retention-manifest.json",
    )

    assert records["claim_records"][0]["support_state"] == "UNSUPPORTED"
    assert records["compression_receipts"][0]["receipt_state"] == "failed"
    assert records["defeater_records"][0]["current_trigger_state"] == "RED"


def test_hot_report_compaction_protects_current_citations_and_ledgers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    monkeypatch.setattr(reference, "REPORTS_ROOT", reports)
    current = reports / "current.json"
    roadmap_current = reports / "roadmap.json"
    ledger = reports / "work_ledger.jsonl"
    historical = reports / "historical.json"
    nested_historical = reports / "report_snapshots" / "registry" / "old.json"
    nested_current = reports / "report_snapshots" / "registry" / "current.json"
    nested_historical.parent.mkdir(parents=True)
    for path in (
        current,
        roadmap_current,
        ledger,
        historical,
        nested_historical,
        nested_current,
    ):
        path.write_text('{"payload":"' + ("x" * 1024) + '"}\n', encoding="utf-8")
        os.utime(path, (time.time() - 72 * 3600, time.time() - 72 * 3600))
    write_json(
        tmp_path / "configs" / "active.json",
        {"report": "reports/report_snapshots/registry/current.json"},
    )
    registry = {"surfaces": [{"report_outputs": ["reports/current.json"]}]}
    matrix = {"phases": [{"phase": 14, "current_evidence": ["reports/roadmap.json"]}]}

    index = reference.build_hot_report_reference_index(registry, matrix)
    by_path = {row["path"]: row for row in index["file_records"]}
    candidates = reference.hot_report_archive_candidates(
        index,
        min_bytes=1,
        min_age_hours=24,
        target_hot_bytes=(
            sum(row["bytes"] for row in index["file_records"])
            - historical.stat().st_size
            - nested_historical.stat().st_size
        ),
    )

    assert by_path["reports/current.json"]["protected"]
    assert by_path["reports/roadmap.json"]["protected"]
    assert by_path["reports/work_ledger.jsonl"]["protected"]
    assert by_path["reports/report_snapshots/registry/current.json"]["protected"]
    assert {row["path"] for row in candidates} == {
        "reports/historical.json",
        "reports/report_snapshots/registry/old.json",
    }


def test_hot_report_default_threshold_catches_registry_sized_snapshots() -> None:
    assert retention.MIN_BYTES == 256 * 1024 * 1024
    assert retention.hot_report_candidate_min_bytes(None) == 1024 * 1024
    assert retention.hot_report_candidate_min_bytes(None) < 9 * 1024 * 1024
    assert retention.hot_report_candidate_min_bytes(4 * 1024 * 1024) == 4 * 1024 * 1024
    assert retention.HOT_REPORT_MIN_AGE_HOURS == 0.0
