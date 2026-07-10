from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_integrity as integrity
import report_evidence_store as store


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_standard_evidence_pack_is_digest_bound_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    gate = tmp_path / "reports" / "gate.json"
    payload = {
        "policy": "fixture_gate_v1",
        "trigger_state": "GREEN",
        "summary": {"pass_count": 2},
        "hard_gaps": [],
        "non_claims": ["fixture scope only"],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json(gate, payload)

    pack = integrity.build_standard_evidence_packs(
        [gate],
        commands={"reports/gate.json": ["python3 fixture_gate.py --gate"]},
    )[0]

    assert pack["validation_state"] == "GREEN"
    assert integrity.validate_evidence_pack(pack)
    assert pack["private_payload_copied"] is False
    assert pack["commands"] == ["python3 fixture_gate.py --gate"]
    assert not integrity.validate_evidence_pack({**pack, "source_sha256": "0" * 64})
    assert not integrity.validate_evidence_pack({**pack, "external_inference_calls": 1})


def test_material_claim_revision_discovers_dependents_and_downgrades_red_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    gate = tmp_path / "reports" / "gate.json"
    claim_report = tmp_path / "reports" / "claims.json"
    dependent = tmp_path / "reports" / "dependent.json"
    write_json(gate, {"policy": "gate", "trigger_state": "RED"})
    write_json(
        claim_report,
        {
            "policy": "claims",
            "trigger_state": "GREEN",
            "claim_ledger": [
                {
                    "record_type": "claim_record",
                    "claim_id": "claim.fixture",
                    "claim": "Fixture claim",
                    "support_state": "synthetic-test-backed",
                    "evidence_refs": ["reports/gate.json"],
                }
            ],
        },
    )
    write_json(
        dependent,
        {
            "policy": "dependent",
            "trigger_state": "GREEN",
            "claim_ref": "claim.fixture",
        },
    )

    result = integrity.build_material_claim_revision([gate, claim_report, dependent], [])

    assert result["state"] == "GREEN"
    assert result["material_claim_count"] == 1
    assert result["dependent_surface_edge_count"] >= 1
    assert result["dependent_invalidation_count"] == 1
    assert result["revision_transitions"][0]["to_state"] == "unsupported"
    assert "reports/dependent.json" in result["revision_transitions"][0]["dependent_surface_refs"]


def test_tcb_validation_rejects_self_audit() -> None:
    tcb = integrity.build_epistemic_tcb(rotation_epoch="2026-W28")
    assert tcb["state"] == "GREEN"
    forged = dict(tcb["audit_assignments"][0])
    forged["auditor_root_ids"] = [forged["subject_root_id"], forged["subject_root_id"]]

    errors = integrity.validate_tcb(tcb["trust_roots"], [forged])

    assert any(error.startswith("self_audit:") for error in errors)
    assert any(error.startswith("insufficient_independent_auditors:") for error in errors)


def test_receipt_audit_rejects_all_adversarial_traps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(integrity, "ROOT", tmp_path)
    gate = tmp_path / "reports" / "gate.json"
    write_json(
        gate,
        {
            "policy": "gate",
            "trigger_state": "GREEN",
            "non_claims": ["fixture"],
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    )
    pack = integrity.build_standard_evidence_packs([gate], commands={})[0]
    claim_revision = {
        "state": "GREEN",
    }
    roots = [
        {"root_id": "producer", "exists": True, "sha256": "a"},
        {"root_id": "audit_a", "exists": True, "sha256": "b"},
        {"root_id": "audit_b", "exists": True, "sha256": "c"},
    ]
    tcb = {
        "state": "GREEN",
        "trust_roots": roots,
        "audit_assignments": [
            {
                "subject_root_id": "producer",
                "auditor_root_ids": ["audit_a", "audit_b"],
            }
        ],
    }

    audit = integrity.audit_receipt_faithfulness([pack], claim_revision, tcb, sample_size=1)

    assert audit["state"] == "GREEN"
    assert audit["passed_deep_replay_count"] == 1
    assert audit["rejected_trap_fixture_count"] == audit["trap_fixture_count"]


def test_evidence_database_compaction_preserves_exact_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = tmp_path / "evidence.sqlite"
    report = tmp_path / "large.json"
    snapshots = tmp_path / "snapshots"
    monkeypatch.setattr(store, "DEFAULT_SNAPSHOT_DIR", snapshots)
    payload = {
        "policy": "large_fixture",
        "created_utc": "2026-07-10T00:00:00Z",
        "trigger_state": "GREEN",
        "summary": {"payload": "x" * 20_000},
    }
    write_json(report, payload)
    stored = store.ingest_report_path(db, report, payload)
    assert stored["payload_truncated"] is False

    result = store.compact_evidence_database(db, inline_limit=100)

    assert result["state"] == "GREEN"
    assert result["migrated_payload_count"] == 1
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT payload_json, snapshot_path, content_hash FROM report_runs").fetchone()
    conn.close()
    assert row[0] == "{}"
    snapshot = Path(row[1])
    assert snapshot.exists()
    assert hashlib.sha256(snapshot.read_bytes().rstrip(b"\n")).hexdigest() == row[2]


def test_material_claim_versions_survive_mutable_report_replacement(tmp_path: Path) -> None:
    db = tmp_path / "evidence.sqlite"
    report = tmp_path / "claims.json"
    first = {
        "policy": "claim_fixture",
        "created_utc": "2026-07-10T00:00:00Z",
        "trigger_state": "GREEN",
        "claim_ledger": [
            {
                "record_type": "claim_record",
                "claim_id": "claim.persisted",
                "claim": "Persisted claim",
                "support_state": "synthetic-test-backed",
                "evidence_refs": ["reports/gate.json"],
            }
        ],
    }
    second = {
        "policy": "claim_fixture",
        "created_utc": "2026-07-10T01:00:00Z",
        "trigger_state": "GREEN",
        "claim_ledger": [],
    }
    write_json(report, first)
    store.ingest_report_path(db, report, first)
    write_json(report, second)
    store.ingest_report_path(db, report, second)

    versions = store.material_claim_versions(db)

    assert len(versions) == 1
    assert versions[0]["claim_id"] == "claim.persisted"
    assert versions[0]["source_paths"] == [str(report)]


def test_evidence_store_does_not_reingest_its_own_claim_projection(tmp_path: Path) -> None:
    db = tmp_path / "evidence.sqlite"
    report = tmp_path / "report_evidence_store.json"
    payload = {
        "policy": "project_theseus_report_evidence_store_v1",
        "created_utc": "2026-07-10T00:00:00Z",
        "trigger_state": "GREEN",
        "material_claim_revision": {
            "claim_samples": [
                {
                    "claim_id": "claim.derived",
                    "claim": "Derived projection",
                    "support_state": "synthetic-test-backed",
                }
            ]
        },
    }
    write_json(report, payload)
    store.ingest_report_path(db, report, payload)

    assert store.material_claim_versions(db) == []


def test_report_ingest_follows_archive_pointer_instead_of_scoring_pointer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.setattr(store.theseus_archive_resolver, "ROOT", tmp_path)
    archive = tmp_path / "archive" / "gate.json"
    pointer = tmp_path / "reports" / "gate.json"
    write_json(archive, {"policy": "real_gate", "trigger_state": "RED", "summary": {"failed": 1}})
    write_json(
        pointer,
        {
            "policy": store.theseus_archive_resolver.ARCHIVE_POINTER_POLICY,
            "trigger_state": "GREEN",
            "archive_path": "archive/gate.json",
        },
    )

    payload = store.read_report_payload(pointer)

    assert payload["policy"] == "real_gate"
    assert payload["trigger_state"] == "RED"
