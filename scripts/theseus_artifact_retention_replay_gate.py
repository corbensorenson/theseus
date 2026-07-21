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
        if str(row.get("status") or "")
        in {
            "archived",
            "already_archived",
            "archived_superseded_by_live_regeneration",
        }
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
            hard_gaps.append(
                gap("archive_pointer_replay_failed", compact_replay_check(check))
            )

    trigger_state = "GREEN" if not hard_gaps else "RED"
    checks_sha256 = digest_rows(checks)
    failed_checks = [row for row in checks if not row.get("passed")]
    records = build_records(checks, source_path, checks_sha256=checks_sha256)
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
            "replay_checks_sha256": checks_sha256,
            "replay_check_sample_count": min(8, len(checks)),
            "failed_replay_check_detail_count": min(32, len(failed_checks)),
            "failed_replay_check_id_count": len(failed_checks),
            "hard_gap_count": len(hard_gaps),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "hard_gaps": hard_gaps,
        "replay_check_samples": [compact_replay_check(row) for row in checks[:8]],
        "failed_replay_check_ids": [str(row.get("check_id") or "") for row in failed_checks],
        "failed_replay_checks": [compact_replay_check(row) for row in failed_checks[:32]],
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
    superseded_live_generation = (
        str(row.get("status") or "")
        == "archived_superseded_by_live_regeneration"
    )
    pointer = read_json(pointer_path)
    pointer_verified = pointer.get("policy") == theseus_archive_resolver.ARCHIVE_POINTER_POLICY
    resolver_target = theseus_archive_resolver.resolve_archived_path(pointer_path)
    decoded_hash = sha256_payload(archive)
    hash_verified = bool(expected_hash and decoded_hash == expected_hash)
    archive_exists = archive.exists()
    resolver_verified = archive_exists and resolver_target.resolve() == archive.resolve()
    live_generation_hash = (
        sha256_file(pointer_path)
        if superseded_live_generation and pointer_path.is_file() and not pointer_verified
        else ""
    )
    live_regeneration_verified = bool(
        superseded_live_generation
        and archive_exists
        and not pointer_verified
        and live_generation_hash
        == str(row.get("live_generation_sha256") or "")
        and live_generation_hash != expected_hash
        and str(row.get("supersession_policy") or "")
        == "renewable_canonical_report_generation_v1"
    )
    json_parse_verified = False
    json_parse_error = ""
    if str(row.get("original_path") or "").endswith(".json"):
        try:
            loaded = (
                read_archive_json(archive)
                if superseded_live_generation
                else theseus_archive_resolver.read_json_follow_pointer(
                    pointer_path, default=None
                )
            )
            json_parse_verified = isinstance(loaded, (dict, list))
        except Exception as exc:  # pragma: no cover - defensive report detail
            json_parse_error = repr(exc)
    else:
        json_parse_verified = True
    if manifest_entry_mode:
        defeater_verified = bool(
            live_regeneration_verified
            or (
                pointer_verified
                and str(pointer.get("original_path") or "")
                == str(row.get("original_path") or row.get("path") or "")
                and str(pointer.get("archive_path") or "")
                == str(row.get("archive_path") or "")
                and str(pointer.get("original_sha256") or "") == expected_hash
            )
        )
        compression_record_verified = bool(
            hash_verified and (resolver_verified or live_regeneration_verified)
        )
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
            pointer_verified or live_regeneration_verified,
            resolver_verified or live_regeneration_verified,
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
        "superseded_live_generation": superseded_live_generation,
        "live_generation_sha256": live_generation_hash,
        "live_regeneration_verified": live_regeneration_verified,
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


def build_records(
    checks: list[dict[str, Any]],
    retention_path: Path,
    *,
    checks_sha256: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Emit one digest-bound record per contract type, not seven copies per check."""
    evidence_ref = "reports/theseus_artifact_retention_replay_gate.json"
    passed_count = sum(1 for row in checks if row.get("passed"))
    failed_count = len(checks) - passed_count
    support_state = "SUPPORTED" if checks and failed_count == 0 else "UNSUPPORTED"
    trigger_state = "GREEN" if support_state == "SUPPORTED" else "RED"
    digest = checks_sha256 or digest_rows(checks)
    aggregate_id = stable_id(
        "artifact_retention_replay_aggregate", len(checks), passed_count, digest
    )
    source_ref = rel(retention_path)
    replay_contract = (
        "Every cumulative manifest entry was independently decoded, hashed, "
        "and checked against its retained content hash; individual outcomes "
        "are committed by replay_checks_sha256."
    )
    common = {
        "replay_checks_sha256": digest,
        "eligible_action_count": len(checks),
        "passed_replay_count": passed_count,
        "failed_replay_count": failed_count,
        "support_state_effect": support_state,
        "evidence_refs": [evidence_ref, source_ref],
        **NO_CHEAT,
    }
    return {
        "compressed_artifact_records": [
            {
                "record_type": "compressed_artifact_record",
                "record_id": f"compressed-artifact-replay-{aggregate_id}",
                "artifact_id": f"retained-artifact-set-{aggregate_id}",
                "source_artifact": source_ref,
                "task_family": "artifact_retention_replay",
                "access_pattern": "cumulative_manifest_exact_replay",
                "admission_state": "archive_replay_supported" if failed_count == 0 else "archive_replay_failed",
                "compression_method": "mixed_manifest_archives",
                "reconstruction_contract": replay_contract,
                "declared_use_envelope": ["evidence replay", "operator audit citation", "storage retention"],
                "ratio_claim_state": "observed_not_benchmarked",
                "codec_parameters": [f"replay_checks_sha256={digest}"],
                "metadata_costs": [f"eligible_action_count={len(checks)}"],
                "residual_coding": ["manifest paths retain per-artifact archive metadata"],
                "probe_plan": ["decode every archive", "rehash every payload", "parse JSON when applicable"],
                "fallback_artifact": source_ref,
                "fallback_trigger": "Use the cumulative manifest to inspect a sampled or failed entry.",
                "decode_determinism": "exact archive decode plus sha256 replay",
                "exact_replay_status": "exact_hash_match" if failed_count == 0 else "replay_failure_present",
                "consumer_policy": "May support evidence retention only after this replay gate is GREEN.",
                "utility_tests": ["archive resolver", "sha256 replay", "JSON parse"],
                **common,
                "non_claims": ["not learned-generation evidence", "not a compression benchmark", "not model capability evidence"],
            }
        ],
        "compression_receipts": [
            {
                "record_type": "compression_receipt",
                "record_id": f"compression-replay-receipt-{aggregate_id}",
                "artifact_id": f"retention-replay-receipt-{aggregate_id}",
                "receipt_state": "verified" if failed_count == 0 else "failed",
                "reconstruction_contract": replay_contract,
                "public_law_family": "mixed_manifest_archives",
                "seed": digest,
                "search_bound": "deterministic archive replay; no learned search credited",
                "generated_regions": [source_ref],
                "verification_result": "exact_replay_passed" if failed_count == 0 else "exact_replay_failed",
                "repair_residual": "" if failed_count == 0 else "inspect failed_replay_checks before relying on retained artifacts",
                "fallback_threshold": "fail closed for claim support unless replay gate is GREEN",
                "interface_costs": ["manifest read", "archive decode", "sha256 replay", "JSON parse"],
                "consumer_policy": "Storage consumers may follow retained archives only when this receipt is verified.",
                "use_permissions": ["artifact retention audit", "operator governance export"],
                "proxy_rate_status": "not_a_rate_claim",
                "final_serialization_status": "verified_exact_payload_set" if failed_count == 0 else "not_verified",
                "rate_accounting": {"artifact_count": len(checks), "ratio_claimed": False},
                **common,
                "non_claims": ["not learned-generation evidence", "not public benchmark evidence"],
            }
        ],
        "proof_contract_receipt_records": [
            {
                "record_type": "proof_contract_receipt_record",
                "record_id": f"proof-contract-retention-replay-{aggregate_id}",
                "receipt_id": f"retention-replay-proof-{aggregate_id}",
                "contract_id": "artifact_retention_archive_pointer_reconstruction_v1",
                "artifact_ref": source_ref,
                "verifier_state": support_state,
                "verification_result": "passed" if failed_count == 0 else "failed",
                "evidence_ref": evidence_ref,
                "blocked_uses": ["learned_generation_claim", "model_quality_claim", "public_benchmark_training"],
                **common,
                "non_claims": ["proof covers storage replay only"],
            }
        ],
        "claim_records": [
            {
                "record_type": "claim_record",
                "claim_id": f"claim-retention-replay-{aggregate_id}",
                "claim": "All admitted cumulative-manifest artifacts reconstruct by exact hash replay.",
                "support_state": support_state,
                "defeaters": [] if failed_count == 0 else ["archive_pointer_replay_failed"],
                **common,
                "non_claims": ["not a capability claim", "not a learned generation claim"],
            }
        ],
        "artifact_graph_records": [
            {
                "record_type": "artifact_graph_record",
                "artifact_id": f"artifact-graph-retention-replay-{aggregate_id}",
                "artifact_type": "retained_artifact_set_replay",
                "parent_job": "artifact_retention_replay_gate",
                "source_refs": [source_ref],
                "claim_refs": [f"claim-retention-replay-{aggregate_id}"],
                "test_refs": [f"proof-contract-retention-replay-{aggregate_id}"],
                "audit_events": ["all_archives_decoded", "all_sha256_recomputed", "outcomes_digest_committed"],
                "replay_metadata": {"replay_checks_sha256": digest, "artifact_count": len(checks)},
                "replay_grade": "exact_hash_replay" if failed_count == 0 else "failed_replay",
                "provenance_status": "retained_generated_artifact_set",
                "evidence_gate": {"state": support_state, **NO_CHEAT},
                "residuals": [] if failed_count == 0 else ["one or more archive replay mismatches"],
                **common,
                "non_claims": ["not learned-generation evidence", "not model capability evidence"],
            }
        ],
        "evidence_transition_records": [
            {
                "record_type": "evidence_transition_record",
                "record_id": f"evidence-transition-retention-replay-{aggregate_id}",
                "artifact_ref": source_ref,
                "previous_support_state": "ARCHIVED_POINTER_UNVERIFIED",
                "current_support_state": support_state,
                "transition_reason": "cumulative_archive_pointer_exact_replay_gate",
                "evidence_ref": evidence_ref,
                **common,
                "non_claims": ["storage evidence transition only"],
            }
        ],
        "defeater_records": [
            {
                "record_type": "defeater_record",
                "record_id": f"defeater-retention-replay-{aggregate_id}",
                "defeater_type": "archive_pointer_unverified_until_exact_replay",
                "defeated_run_id": "artifact_retention_pointer_write_set",
                "defeating_run_id": f"retention_replay_gate:{aggregate_id}",
                "previous_content_hash": digest,
                "current_content_hash": digest,
                "previous_support_state": "ARCHIVED_POINTER_UNVERIFIED",
                "current_support_state": support_state,
                "previous_trigger_state": "YELLOW",
                "current_trigger_state": trigger_state,
                "original_path": source_ref,
                "archive_path": source_ref,
                "pointer_path": source_ref,
                "support_state": support_state,
                "evidence_ref": evidence_ref,
                **common,
                "non_claim": "Defeater is resolved only for exact archive replay; it is not a model capability claim.",
            }
        ],
    }


def digest_rows(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_replay_check(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "check_id",
            "status",
            "passed",
            "original_path",
            "pointer_path",
            "archive_path",
            "expected_sha256",
            "decoded_sha256",
            "archive_exists",
            "pointer_verified",
            "resolver_verified",
            "live_regeneration_verified",
            "hash_verified",
            "json_parse_verified",
            "json_parse_error",
            "defeater_verified",
            "compression_record_verified",
        )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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
    for row in payload.get("failed_replay_checks", []):
        lines.append(f"- failed `{row.get('original_path')}` -> `{row.get('archive_path')}`")
    for row in payload.get("replay_check_samples", [])[:8]:
        lines.append(f"- `{row.get('passed')}` `{row.get('original_path')}` -> `{row.get('archive_path')}`")
    return "\n".join(lines) + "\n"


def gate_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "summary": payload.get("summary", {}),
        "hard_gaps": payload.get("hard_gaps", [])[:10],
        "failed_replay_checks": payload.get("failed_replay_checks", [])[:10],
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
