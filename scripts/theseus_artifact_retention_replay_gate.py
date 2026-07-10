#!/usr/bin/env python3
"""Verify retained artifact archive pointers by exact replay.

The retention service moves bulky generated evidence out of hot report paths
and leaves archive pointers behind. This gate independently follows those
pointers, decodes archived payloads, rehashes them, and emits VIEA-shaped
records proving whether the pointer can reconstruct the original artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402
import theseus_archive_resolver  # noqa: E402


DEFAULT_RETENTION_REPORT = ROOT / "reports" / "theseus_artifact_retention.json"
DEFAULT_RETENTION_MANIFEST = ROOT / "reports" / "theseus_artifact_retention_manifest.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_artifact_retention_replay_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "theseus_artifact_retention_replay_gate.md"
NO_CHEAT = {
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention-report", default=rel(DEFAULT_RETENTION_REPORT))
    parser.add_argument("--retention-manifest", default=rel(DEFAULT_RETENTION_MANIFEST))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    retention_path = resolve(args.retention_report)
    retention = read_json(retention_path)
    manifest_path = resolve(args.retention_manifest)
    manifest = read_json(manifest_path)
    manifest_rows = [
        row
        for row in list_dicts(manifest.get("entries"))
        if str(row.get("status") or "") in {"archived", "already_archived"}
    ]
    source_rows = manifest_rows or [
        row
        for row in list_dicts(retention.get("actions"))
        if str(row.get("status") or "") in {"archived", "already_archived"}
    ]
    source_payload = manifest if manifest_rows else retention
    source_path = manifest_path if manifest_rows else retention_path
    checks = [
        verify_action(
            row,
            source_payload,
            source_path,
            manifest_entry_mode=bool(manifest_rows),
        )
        for row in source_rows
    ]
    hard_gaps: list[dict[str, Any]] = []
    if not retention_path.exists():
        hard_gaps.append(gap("retention_report_missing", {"path": rel(retention_path)}))
    if not checks and not args.allow_empty:
        hard_gaps.append(gap("no_executed_retention_actions", {"retention_report": rel(retention_path), "retention_manifest": rel(manifest_path)}))
    for check in checks:
        if not check.get("passed"):
            hard_gaps.append(gap("archive_pointer_replay_failed", check))

    trigger_state = "GREEN" if not hard_gaps else "RED"
    records = build_records(checks, source_path)
    payload = {
        "policy": "project_theseus_artifact_retention_replay_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "retention_report": rel(retention_path),
            "retention_manifest": rel(manifest_path),
            "replay_source": "cumulative_manifest" if manifest_rows else "current_retention_report",
            "eligible_action_count": len(checks),
            "passed_replay_count": sum(1 for row in checks if row.get("passed")),
            "failed_replay_count": sum(1 for row in checks if not row.get("passed")),
            "pointer_verified_count": sum(1 for row in checks if row.get("pointer_verified")),
            "defeater_verified_count": sum(1 for row in checks if row.get("defeater_verified")),
            "json_parse_verified_count": sum(1 for row in checks if row.get("json_parse_verified")),
            "hard_gap_count": len(hard_gaps),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "hard_gaps": hard_gaps,
        "replay_checks": checks,
        **records,
        **NO_CHEAT,
        "non_claims": [
            "artifact retention replay is not learned generation evidence",
            "artifact retention replay is not model capability evidence",
            "retained public benchmark quarantine payloads remain calibration/quarantine data, not training data",
        ],
    }
    report_evidence_store.write_json_report(
        resolve(args.out),
        payload,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(payload),
    )
    print(json.dumps(gate_view(payload), indent=2, sort_keys=True))
    return 0 if trigger_state == "GREEN" else 2


def verify_action(
    row: dict[str, Any],
    retention: dict[str, Any],
    retention_path: Path,
    *,
    manifest_entry_mode: bool = False,
) -> dict[str, Any]:
    original = resolve(str(row.get("original_path") or row.get("path") or ""))
    pointer_path = resolve(str(row.get("pointer_path") or row.get("original_path") or row.get("path") or ""))
    archive = resolve(str(row.get("archive_path") or ""))
    expected_hash = str(row.get("sha256") or "")
    pointer = read_json(pointer_path)
    pointer_verified = pointer.get("policy") == theseus_archive_resolver.ARCHIVE_POINTER_POLICY
    resolver_target = theseus_archive_resolver.resolve_archived_path(pointer_path)
    decoded_hash = sha256_payload(archive)
    hash_verified = bool(expected_hash and decoded_hash == expected_hash)
    archive_exists = archive.exists()
    resolver_verified = archive_exists and resolver_target.resolve() == archive.resolve()
    json_parse_verified = False
    json_parse_error = ""
    if str(row.get("original_path") or "").endswith(".json"):
        try:
            loaded = theseus_archive_resolver.read_json_follow_pointer(pointer_path, default=None)
            json_parse_verified = isinstance(loaded, (dict, list))
        except Exception as exc:  # pragma: no cover - defensive report detail
            json_parse_error = repr(exc)
    else:
        json_parse_verified = True
    if manifest_entry_mode:
        defeater_verified = bool(
            pointer_verified
            and str(pointer.get("original_path") or "") == str(row.get("original_path") or row.get("path") or "")
            and str(pointer.get("archive_path") or "") == str(row.get("archive_path") or "")
            and str(pointer.get("original_sha256") or "") == expected_hash
        )
        compression_record_verified = bool(hash_verified and resolver_verified)
    else:
        defeater_verified = has_matching_defeater(row, retention)
        compression_record_verified = has_matching_record(
            row,
            list_dicts(retention.get("compression_records")),
            expected_hash=expected_hash,
        )
    passed = all(
        [
            archive_exists,
            pointer_verified,
            resolver_verified,
            hash_verified,
            json_parse_verified,
            defeater_verified,
            compression_record_verified,
        ]
    )
    return {
        "check_id": stable_id("artifact_retention_replay_check", row.get("original_path"), row.get("archive_path"), expected_hash),
        "status": str(row.get("status") or ""),
        "passed": passed,
        "original_path": rel(original),
        "pointer_path": rel(pointer_path),
        "archive_path": rel(archive),
        "expected_sha256": expected_hash,
        "decoded_sha256": decoded_hash,
        "archive_exists": archive_exists,
        "pointer_verified": pointer_verified,
        "resolver_verified": resolver_verified,
        "hash_verified": hash_verified,
        "json_parse_verified": json_parse_verified,
        "json_parse_error": json_parse_error,
        "defeater_verified": defeater_verified,
        "compression_record_verified": compression_record_verified,
        "payload_bytes": int(row.get("bytes") or pointer.get("original_bytes") or 0),
        "archived_bytes": int(row.get("archived_bytes") or (archive.stat().st_size if archive.exists() else 0)),
        "retention_report": rel(retention_path),
        "manifest_entry_mode": bool(manifest_entry_mode),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def has_matching_defeater(row: dict[str, Any], retention: dict[str, Any]) -> bool:
    original = str(row.get("original_path") or row.get("path") or "")
    archive = str(row.get("archive_path") or "")
    expected_hash = str(row.get("sha256") or "")
    for defeater in list_dicts(retention.get("defeater_records")):
        if str(defeater.get("original_path") or "") != original:
            continue
        if str(defeater.get("archive_path") or "") != archive:
            continue
        if expected_hash and str(defeater.get("previous_content_hash") or "") != expected_hash:
            continue
        if str(defeater.get("current_support_state") or "") != "ARCHIVED_POINTER":
            continue
        return True
    return False


def has_matching_record(row: dict[str, Any], records: list[dict[str, Any]], *, expected_hash: str) -> bool:
    original = str(row.get("original_path") or row.get("path") or "")
    archive = str(row.get("archive_path") or "")
    for record in records:
        if str(record.get("original_path") or "") != original:
            continue
        if str(record.get("archive_path") or "") != archive:
            continue
        if expected_hash and str(record.get("content_hash") or "") != expected_hash:
            continue
        if str(record.get("support_state") or "").upper() != "SUPPORTED":
            continue
        return True
    return False


def build_records(checks: list[dict[str, Any]], retention_path: Path) -> dict[str, list[dict[str, Any]]]:
    compressed_artifact_records = []
    compression_receipts = []
    proof_contract_receipt_records = []
    claim_records = []
    artifact_graph_records = []
    evidence_transition_records = []
    defeater_records = []
    for check in checks:
        rid = check["check_id"]
        support_state = "SUPPORTED" if check.get("passed") else "UNSUPPORTED"
        evidence_ref = "reports/theseus_artifact_retention_replay_gate.json"
        compressed_artifact_records.append(
            {
                "record_type": "compressed_artifact_record",
                "record_id": f"compressed-artifact-replay-{rid}",
                "artifact_id": f"retained-artifact-{rid}",
                "source_artifact": check["original_path"],
                "task_family": "artifact_retention_replay",
                "access_pattern": "archive_pointer_exact_replay",
                "admission_state": "archive_replay_supported" if check.get("passed") else "archive_replay_failed",
                "compression_method": "gzip" if check["archive_path"].endswith(".gz") else "move_or_pointer",
                "reconstruction_contract": "pointer_path resolves to archive_path and decoded payload sha256 equals expected_sha256",
                "declared_use_envelope": ["evidence replay", "operator audit citation", "storage retention"],
                "ratio_claim_state": "observed_not_benchmarked",
                "codec_parameters": [f"expected_sha256={check['expected_sha256']}", f"decoded_sha256={check['decoded_sha256']}"],
                "metadata_costs": [f"payload_bytes={check['payload_bytes']}", f"archived_bytes={check['archived_bytes']}"],
                "residual_coding": ["pointer json remains at original path", "archive payload remains authoritative for exact replay"],
                "probe_plan": ["follow pointer", "decode archive", "rehash payload", "parse json when applicable"],
                "fallback_artifact": check["archive_path"],
                "fallback_trigger": "Use archived payload whenever the original path is an archive pointer.",
                "decode_determinism": "exact gzip decode plus sha256 replay",
                "exact_replay_status": "exact_hash_match" if check.get("hash_verified") else "hash_mismatch",
                "consumer_policy": "May support evidence retention only after this replay gate is GREEN.",
                "utility_tests": ["archive pointer resolver", "sha256 replay", "json parse when applicable"],
                "support_state_effect": support_state,
                "evidence_refs": [evidence_ref, rel(retention_path)],
                **NO_CHEAT,
                "non_claims": ["not learned-generation evidence", "not a compression benchmark", "not model capability evidence"],
            }
        )
        compression_receipts.append(
            {
                "record_type": "compression_receipt",
                "record_id": f"compression-replay-receipt-{rid}",
                "artifact_id": f"retention-replay-receipt-{rid}",
                "receipt_state": "verified" if check.get("passed") else "failed",
                "reconstruction_contract": "archive pointer must replay to the original content hash",
                "public_law_family": "gzip" if check["archive_path"].endswith(".gz") else "move_or_pointer",
                "seed": check["expected_sha256"],
                "search_bound": "deterministic archive replay; no learned search credited",
                "generated_regions": [check["archive_path"]],
                "verification_result": "exact_replay_passed" if check.get("passed") else "exact_replay_failed",
                "repair_residual": "" if check.get("passed") else "inspect pointer/archive/hash mismatch before relying on retained artifact",
                "fallback_threshold": "fail closed for claim support unless replay gate is GREEN",
                "interface_costs": ["pointer read", "archive decode", "sha256 replay", "json parse when applicable"],
                "consumer_policy": "Storage/evidence consumers may follow the pointer only when this receipt is verified.",
                "use_permissions": ["artifact retention audit", "operator governance export"],
                "proxy_rate_status": "not_a_rate_claim",
                "final_serialization_status": "verified_exact_payload" if check.get("passed") else "not_verified",
                "rate_accounting": {
                    "payload_bytes": check["payload_bytes"],
                    "archived_bytes": check["archived_bytes"],
                    "ratio_claimed": False,
                },
                "support_state_effect": support_state,
                "evidence_refs": [evidence_ref, rel(retention_path)],
                **NO_CHEAT,
                "non_claims": ["not learned-generation evidence", "not public benchmark evidence"],
            }
        )
        proof_contract_receipt_records.append(
            {
                "record_type": "proof_contract_receipt_record",
                "record_id": f"proof-contract-retention-replay-{rid}",
                "receipt_id": f"retention-replay-proof-{rid}",
                "contract_id": "artifact_retention_archive_pointer_reconstruction_v1",
                "artifact_ref": check["original_path"],
                "verifier_state": support_state,
                "verification_result": "passed" if check.get("passed") else "failed",
                "evidence_ref": evidence_ref,
                "blocked_uses": ["learned_generation_claim", "model_quality_claim", "public_benchmark_training"],
                **NO_CHEAT,
                "non_claims": ["proof covers storage replay only"],
            }
        )
        claim_records.append(
            {
                "record_type": "claim_record",
                "claim_id": f"claim-retention-replay-{rid}",
                "claim": "Archived artifact pointer reconstructs the retained payload by exact hash replay.",
                "support_state": support_state,
                "evidence_refs": [evidence_ref, rel(retention_path), check["archive_path"]],
                "defeaters": [] if check.get("passed") else ["archive_pointer_replay_failed"],
                **NO_CHEAT,
                "non_claims": ["not a capability claim", "not a learned generation claim"],
            }
        )
        artifact_graph_records.append(
            {
                "record_type": "artifact_graph_record",
                "artifact_id": f"artifact-graph-retention-replay-{rid}",
                "artifact_type": "retained_artifact_replay",
                "parent_job": "artifact_retention_replay_gate",
                "source_refs": [rel(retention_path), check["original_path"], check["archive_path"]],
                "claim_refs": [f"claim-retention-replay-{rid}"],
                "test_refs": [f"proof-contract-retention-replay-{rid}"],
                "audit_events": ["archive_pointer_followed", "archive_payload_decoded", "sha256_recomputed"],
                "replay_metadata": check,
                "replay_grade": "exact_hash_replay" if check.get("passed") else "failed_replay",
                "provenance_status": "retained_generated_artifact",
                "evidence_gate": {"state": support_state, **NO_CHEAT},
                "residuals": [] if check.get("passed") else ["pointer/archive/hash replay mismatch"],
                **NO_CHEAT,
                "non_claims": ["not learned-generation evidence", "not model capability evidence"],
            }
        )
        evidence_transition_records.append(
            {
                "record_type": "evidence_transition_record",
                "record_id": f"evidence-transition-retention-replay-{rid}",
                "artifact_ref": check["original_path"],
                "previous_support_state": "ARCHIVED_POINTER_UNVERIFIED",
                "current_support_state": support_state,
                "transition_reason": "archive_pointer_exact_replay_gate",
                "evidence_ref": evidence_ref,
                **NO_CHEAT,
                "non_claims": ["storage evidence transition only"],
            }
        )
        defeater_records.append(
            {
                "record_type": "defeater_record",
                "record_id": f"defeater-retention-replay-{rid}",
                "defeater_type": "archive_pointer_unverified_until_exact_replay",
                "defeated_run_id": "artifact_retention_pointer_write",
                "defeating_run_id": f"retention_replay_gate:{rid}",
                "previous_content_hash": check["expected_sha256"],
                "current_content_hash": check["decoded_sha256"],
                "previous_support_state": "ARCHIVED_POINTER_UNVERIFIED",
                "current_support_state": support_state,
                "previous_trigger_state": "YELLOW",
                "current_trigger_state": "GREEN" if check.get("passed") else "RED",
                "original_path": check["original_path"],
                "archive_path": check["archive_path"],
                "pointer_path": check["pointer_path"],
                "support_state": support_state,
                "evidence_ref": evidence_ref,
                **NO_CHEAT,
                "non_claim": "Defeater is resolved only for archive replay; it does not make a model capability claim.",
            }
        )
    return {
        "compressed_artifact_records": compressed_artifact_records,
        "compression_receipts": compression_receipts,
        "proof_contract_receipt_records": proof_contract_receipt_records,
        "claim_records": claim_records,
        "artifact_graph_records": artifact_graph_records,
        "evidence_transition_records": evidence_transition_records,
        "defeater_records": defeater_records,
    }


def sha256_payload(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Theseus Artifact Retention Replay Gate",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- eligible actions: `{summary.get('eligible_action_count')}`",
        f"- passed replay: `{summary.get('passed_replay_count')}`",
        f"- failed replay: `{summary.get('failed_replay_count')}`",
        f"- pointer verified: `{summary.get('pointer_verified_count')}`",
        f"- defeater verified: `{summary.get('defeater_verified_count')}`",
        "",
        "## Checks",
        "",
    ]
    for row in payload.get("replay_checks", [])[:40]:
        lines.append(f"- `{row.get('passed')}` `{row.get('original_path')}` -> `{row.get('archive_path')}`")
    return "\n".join(lines) + "\n"


def gate_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "summary": payload.get("summary", {}),
        "hard_gaps": payload.get("hard_gaps", [])[:10],
        "failed_replay_checks": [row for row in payload.get("replay_checks", []) if not row.get("passed")][:10],
    }


def gap(kind: str, detail: Any) -> dict[str, Any]:
    return {"kind": kind, "detail": detail}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
