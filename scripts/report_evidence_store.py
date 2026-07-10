"""Append-only SQLite evidence store for volatile report files.

JSON reports are still useful human-facing views, but many scripts write to
stable paths such as ``reports/foo.json``. This store preserves each distinct
payload by content hash so later schedulers can ask for the strongest evidence
instead of whichever file was written last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import report_evidence_integrity
import theseus_archive_resolver


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_DB = REPORTS / "report_evidence_store.sqlite"
DEFAULT_OUT = REPORTS / "report_evidence_store.json"
DEFAULT_MARKDOWN = REPORTS / "report_evidence_store.md"
DEFAULT_SNAPSHOT_DIR = REPORTS / "report_snapshots"
DEFAULT_EVIDENCE_PACKS_OUT = REPORTS / "theseus_book_importable_evidence_packs.json"
DEFAULT_REGISTRY = ROOT / "configs" / "project_manifest_registry.json"
DEFAULT_ROADMAP_MATRIX = ROOT / "configs" / "roadmap_implementation_matrix.json"
MAX_PAYLOAD_BYTES = 512 * 1024
NO_CHEAT_COUNTERS = ("public_training_rows_written", "external_inference_calls", "fallback_return_count")

DEFAULT_REPORT_PATHS = [
    REPORTS / "autonomy_watchdog.json",
    REPORTS / "learning_launch_supervisor.json",
    REPORTS / "vacation_mode_supervisor_overnight.json",
    REPORTS / "service_process_hygiene.json",
    REPORTS / "resource_aware_execution_policy.json",
    REPORTS / "windows_cuda_doctor.json",
    REPORTS / "system_efficiency_audit.json",
    REPORTS / "asi_wall_breaker_governor.json",
    REPORTS / "closed_loop_residual_ratchet.json",
    REPORTS / "theseus_control_plane.json",
    REPORTS / "theseus_plan_compiler.json",
    REPORTS / "theseus_plan_compiler_ablation.json",
    REPORTS / "theseus_plan_compiled_dags.json",
    REPORTS / "viea_execution_spine.json",
    REPORTS / "viea_verified_procedural_tools.json",
    REPORTS / "viea_research_implementation_matrix.json",
    REPORTS / "viea_spine_record_gate.json",
    REPORTS / "viea_spine_materialized_view.json",
    REPORTS / "roadmap_implementation_gate.json",
    REPORTS / "training_inference_execution_plan_gate.json",
    REPORTS / "book_to_theseus_crosswalk.json",
    REPORTS / "theseus_project_registry.json",
    REPORTS / "candidate_integrity_audit.json",
    REPORTS / "private_verifier_spine_smoke.json",
    REPORTS / "hive_scheduler.json",
    REPORTS / "hive_jobs.json",
    REPORTS / "hive_policy_first_scheduler.json",
    REPORTS / "theseus_workspace_hygiene_audit.json",
    REPORTS / "theseus_deprecation_registry.json",
    REPORTS / "theseus_artifact_retention.json",
    REPORTS / "theseus_generated_artifact_gc.json",
    REPORTS / "theseus_dirty_workspace_review.json",
    REPORTS / "agent_lane_transfer_gate.json",
    REPORTS / "decoder_v2_private_ablation_gate.json",
    REPORTS / "private_public_transfer_proof.json",
    REPORTS / "code_lm_train_once_fanout.json",
    REPORTS / "high_transfer_multi_turn_conversation.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard_v2.json",
    REPORTS / "high_transfer_multi_turn_conversation_hard_v3.json",
    REPORTS / "multi_turn_conversation_benchmark.json",
    REPORTS / "high_transfer_curriculum_scheduler.json",
    REPORTS / "hive_work_board_executor.json",
    REPORTS / "broad_transfer_matrix.json",
    REPORTS / "learning_scoreboard.json",
    REPORTS / "a_plus_operating_scorecard.json",
    REPORTS / "maturity_integrity_audit.json",
    REPORTS / "cross_domain_sts_capsules.json",
    REPORTS / "high_transfer_long_horizon_tool_use.json",
    REPORTS / "board_game_rl_benchmark.json",
    REPORTS / "board_game_learned_policy.json",
    REPORTS / "edge_contract_v2_private_verifier.json",
    REPORTS / "edge_obligation_decode_gate_v1_private.json",
    REPORTS / "edge_obligation_decode_gate_v1_private_pressure_private.json",
    REPORTS / "decoder_plan_ir_private_pressure.json",
    REPORTS / "decoder_plan_ir_code_lm_adapter.json",
    REPORTS / "public_transfer_residual_packet.json",
    REPORTS / "candidate_promotion_gate.json",
    REPORTS / "real_code_benchmark_graduation.json",
    REPORTS / "personality_runtime_audit.json",
    REPORTS / "deterministic_tool_substrate.json",
    REPORTS / "deterministic_tool_registry.json",
    REPORTS / "deterministic_tool_ablation.json",
    REPORTS / "deterministic_tool_loop_closure_candidates.json",
    REPORTS / "deterministic_tool_artifact_graph.json",
    REPORTS / "theseus_assistant_runtime.json",
    REPORTS / "theseus_assistant_e2e.json",
    REPORTS / "theseus_assistant_roadmap_integration_smoke.json",
    REPORTS / "theseus_assistant_tool_integration_smoke.json",
    REPORTS / "theseus_weekly_focus_20260706.json",
    REPORTS / "theseus_public_safe_reference_trace_20260706.json",
    REPORTS / "theseus_book_importable_evidence_packs_20260706.json",
    REPORTS / "report_evidence_store.json",
]

DEFAULT_REPORT_GLOBS = [
    "autonomy_watchdog*.json",
    "learning_launch_supervisor*.json",
    "vacation_mode_supervisor*.json",
    "service_process_hygiene*.json",
    "resource_aware_execution_policy*.json",
    "windows_cuda_doctor*.json",
    "system_efficiency_audit*.json",
    "asi_wall_breaker_governor*.json",
    "closed_loop_residual_ratchet*.json",
    "theseus_control_plane*.json",
    "theseus_plan_compiler*.json",
    "theseus_plan_compiled_dags*.json",
    "viea_execution_spine*.json",
    "viea_verified_procedural_tools*.json",
    "viea_research_implementation_matrix*.json",
    "viea_spine_record_gate*.json",
    "viea_spine_materialized_view*.json",
    "roadmap_implementation_gate*.json",
    "training_inference_execution_plan_gate*.json",
    "book_to_theseus_crosswalk*.json",
    "theseus_project_registry*.json",
    "candidate_integrity_audit*.json",
    "private_verifier_spine_smoke*.json",
    "hive_scheduler*.json",
    "hive_jobs*.json",
    "hive_policy_first_scheduler*.json",
    "theseus_workspace_hygiene_audit*.json",
    "theseus_deprecation_registry*.json",
    "theseus_artifact_retention*.json",
    "theseus_generated_artifact_gc*.json",
    "theseus_dirty_workspace_review*.json",
    "agent_lane_transfer_gate*.json",
    "decoder_v2_private_ablation_gate*.json",
    "private_public_transfer_proof*.json",
    "code_lm_train_once_fanout*.json",
    "broad_transfer_matrix*.json",
    "broad_transfer_closure*.json",
    "broad_code_calibration_scheduler*.json",
    "transfer_generalization_audit*.json",
    "real_code_benchmark_graduation*.json",
    "code_lm_closure*.json",
    "code_lm_closure_rust*.json",
    "code_lm_partial_artifact_score*.json",
    "code_transfer_bounded_recovery_chain*.json",
    "resource_aware_execution_policy*.json",
    "code_residual_curriculum*.json",
    "high_transfer_*_code_residual_curriculum.json",
    "code_residual_forge*.json",
    "code_transfer_artifacts*.json",
    "learning_scoreboard*.json",
    "candidate_promotion_gate*.json",
    "candidate_evidence_profile*.json",
    "sts_repair_ablation*.json",
    "sts_learning_forge*.json",
    "sts_native_parallel_probe*.json",
    "open_conversation_training_pantry*.json",
    "personality_runtime_audit*.json",
    "high_transfer_curriculum_scheduler*.json",
    "hive_work_board_executor*.json",
    "a_plus_operating_scorecard*.json",
    "maturity_integrity_audit*.json",
    "high_transfer_multi_turn_conversation*.json",
    "high_transfer_long_horizon_tool_use*.json",
    "cross_domain_sts_capsules*.json",
    "board_game_*benchmark*.json",
    "board_game_learned_policy*.json",
    "edge_contract_v2_private_verifier*.json",
    "edge_obligation_decode_gate_v1*.json",
    "decoder_plan_ir_private_pressure*.json",
    "decoder_plan_ir_code_lm_adapter*.json",
    "public_transfer_residual_packet*.json",
    "deterministic_tool*.json",
    "theseus_assistant*.json",
    "assistant_deterministic_tool*.json",
    "theseus_weekly_focus_*.json",
    "theseus_public_safe_reference_trace_*.json",
    "theseus_book_importable_evidence_packs_*.json",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--ingest", nargs="*", default=[])
    parser.add_argument("--family", default="")
    parser.add_argument("--compact-db", action="store_true")
    parser.add_argument("--integrity-sample-size", type=int, default=64)
    parser.add_argument("--evidence-packs-out", default=str(DEFAULT_EVIDENCE_PACKS_OUT.relative_to(ROOT)))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--roadmap-matrix", default=str(DEFAULT_ROADMAP_MATRIX.relative_to(ROOT)))
    args = parser.parse_args()

    started = time.perf_counter()
    db_path = resolve(args.db)
    paths = [resolve(path) for path in args.ingest] if args.ingest else default_report_paths()
    ingested = ingest_reports(db_path, paths)
    db_compaction = compact_evidence_database(db_path) if args.compact_db else {
        "state": "NOT_REQUESTED",
        "migrated_payload_count": 0,
        "migrated_payload_bytes": 0,
        "failed_payload_count": 0,
    }
    compression_records = [row["compression_record"] for row in ingested if isinstance(row.get("compression_record"), dict) and row.get("compression_record")]
    compressed_artifact_records = [
        compressed_artifact_record_from_compression_record(row, source_system="report_evidence_store")
        for row in compression_records
    ]
    compression_receipts = [
        compression_receipt_from_compression_record(row, source_system="report_evidence_store")
        for row in compression_records
    ]
    defeater_records = [row["defeater_record"] for row in ingested if isinstance(row.get("defeater_record"), dict) and row.get("defeater_record")]
    families = sorted(families_in_store(db_path))
    store_summary = evidence_store_summary(db_path)
    stored_claim_versions = material_claim_versions(db_path)
    best = {
        family: compact_payload(best_payload_for_family(db_path, family) or {})
        for family in families
        if not args.family or family == args.family
    }
    current_index = current_report_index(db_path, paths)
    registry = read_json(resolve(args.registry), {})
    roadmap_matrix = read_json(resolve(args.roadmap_matrix), {})
    pack_out = resolve(args.evidence_packs_out)
    excluded_pack_sources = {resolve(args.out).resolve(), pack_out.resolve()}
    citeable_paths = [
        path
        for path in report_evidence_integrity.citeable_green_report_paths(registry, roadmap_matrix)
        if path.resolve() not in excluded_pack_sources
    ]
    evidence_packs = report_evidence_integrity.build_standard_evidence_packs(
        citeable_paths,
        commands=report_evidence_integrity.command_index(registry, roadmap_matrix),
    )
    evidence_pack_export = {
        "policy": "project_theseus_book_importable_evidence_pack_export_v2",
        "created_utc": now(),
        "trigger_state": "GREEN" if evidence_packs and all(row.get("validation_state") == "GREEN" for row in evidence_packs) else "RED",
        "summary": {
            "evidence_pack_count": len(evidence_packs),
            "valid_evidence_pack_count": sum(1 for row in evidence_packs if row.get("validation_state") == "GREEN"),
            "citeable_green_source_count": len(citeable_paths),
            "private_payload_copied_count": sum(1 for row in evidence_packs if row.get("private_payload_copied")),
        },
        "evidence_packs": evidence_packs,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "Evidence packs export compact public-safe receipts; source report payloads are not copied.",
            "A GREEN pack preserves the source gate scope and does not create a broader capability claim.",
        ],
    }
    write_json(pack_out, evidence_pack_export)
    epistemic_tcb = report_evidence_integrity.build_epistemic_tcb()
    claim_revision = report_evidence_integrity.build_material_claim_revision(
        citeable_paths,
        evidence_packs,
        stored_claim_versions=stored_claim_versions,
    )
    receipt_audit = report_evidence_integrity.audit_receipt_faithfulness(
        evidence_packs,
        claim_revision,
        epistemic_tcb,
        sample_size=max(1, int(args.integrity_sample_size)),
    )
    integrity_states = {
        "evidence_packs": evidence_pack_export["trigger_state"],
        "epistemic_tcb": epistemic_tcb["state"],
        "claim_revision": claim_revision["state"],
        "receipt_faithfulness": receipt_audit["state"],
        "db_compaction": db_compaction["state"],
    }
    trigger_state = "GREEN" if all(state in {"GREEN", "NOT_REQUESTED"} for state in integrity_states.values()) else "RED"
    payload = {
        "policy": "project_theseus_report_evidence_store_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "database": rel_or_abs(db_path),
        "summary": {
            "ingested_this_run": len(ingested),
            "stored_run_count": store_summary["stored_run_count"],
            "family_count": len(families),
            "latest_path_count": store_summary["latest_path_count"],
            "truncated_payload_count": store_summary["truncated_payload_count"],
            "snapshot_count": store_summary["snapshot_count"],
            "truncated_without_snapshot_count": store_summary["truncated_without_snapshot_count"],
            "current_report_count": current_index["current_report_count"],
            "current_unstored_count": current_index["current_unstored_count"],
            "current_truncated_without_snapshot_count": current_index["current_truncated_without_snapshot_count"],
            "compression_record_count": len(compression_records),
            "compressed_artifact_record_count": len(compressed_artifact_records),
            "compression_receipt_count": len(compression_receipts),
            "defeater_record_count": len(defeater_records),
            "citeable_green_gate_count": len(citeable_paths),
            "standard_evidence_pack_count": len(evidence_packs),
            "valid_standard_evidence_pack_count": sum(1 for row in evidence_packs if row.get("validation_state") == "GREEN"),
            "epistemic_tcb_state": epistemic_tcb["state"],
            "epistemic_tcb_root_count": len(epistemic_tcb["trust_roots"]),
            "epistemic_tcb_audit_assignment_count": len(epistemic_tcb["audit_assignments"]),
            "material_claim_count": claim_revision["material_claim_count"],
            "claim_emitting_run_family_count": claim_revision["claim_emitting_run_family_count"],
            "claim_dependent_surface_edge_count": claim_revision["dependent_surface_edge_count"],
            "claim_dependent_invalidation_count": claim_revision["dependent_invalidation_count"],
            "stored_material_claim_version_count": len(stored_claim_versions),
            "receipt_randomized_deep_replay_count": receipt_audit["randomized_deep_replay_count"],
            "receipt_rejected_trap_fixture_count": receipt_audit["rejected_trap_fixture_count"],
            "db_compacted_payload_count": db_compaction["migrated_payload_count"],
            "db_compacted_payload_bytes": db_compaction["migrated_payload_bytes"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "ingested": ingested,
        "compression_records": compression_records,
        "compressed_artifact_records": compressed_artifact_records,
        "compression_receipts": compression_receipts,
        "defeater_records": defeater_records,
        "current_index": current_index,
        "best_by_family": best,
        "db_compaction": db_compaction,
        "standard_evidence_pack_export": rel_or_abs(pack_out),
        "epistemic_trusted_computing_base": epistemic_tcb,
        "material_claim_revision": claim_revision,
        "receipt_faithfulness_audit": receipt_audit,
        "integrity_states": integrity_states,
        "rules": {
            "append_only": "Distinct report payloads are stored by content hash instead of overwriting prior evidence.",
            "large_payload_snapshots": "Reports larger than the inline payload limit are copied into content-addressed immutable snapshots.",
            "latest_views": "Stable JSON paths are mutable latest views; report_latest_by_path points to their last observed immutable run.",
            "reports_are_views": "JSON reports remain views; this DB is the durable recent-run evidence index.",
            "best_evidence": "Schedulers should choose strongest valid evidence by family, not newest file path.",
            "claim_revision": "Missing or RED evidence dependencies downgrade material claims and invalidate discovered dependent surfaces; they never promote a claim.",
            "epistemic_tcb": "Trust roots are digest-bound and high-consequence verifiers receive rotated primary and shadow auditors.",
            "public_safe_packs": "Every registry/roadmap-citeable GREEN report receives a digest-bound compact pack without copying private payloads.",
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if trigger_state == "GREEN" else 2


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS report_runs (
            run_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            report_path TEXT NOT NULL,
            policy TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            observed_utc TEXT NOT NULL,
            source_mtime REAL NOT NULL,
            trigger_state TEXT NOT NULL,
            passed INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            payload_bytes INTEGER NOT NULL,
            payload_truncated INTEGER NOT NULL,
            snapshot_path TEXT NOT NULL DEFAULT '',
            summary_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_report_runs_family ON report_runs(family);
        CREATE INDEX IF NOT EXISTS idx_report_runs_hash ON report_runs(content_hash);
        CREATE INDEX IF NOT EXISTS idx_report_runs_created ON report_runs(created_utc);
        CREATE TABLE IF NOT EXISTS report_latest_by_path (
            report_path TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            family TEXT NOT NULL,
            policy TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            observed_utc TEXT NOT NULL,
            source_mtime REAL NOT NULL,
            trigger_state TEXT NOT NULL,
            passed INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            payload_bytes INTEGER NOT NULL,
            payload_truncated INTEGER NOT NULL,
            snapshot_path TEXT NOT NULL DEFAULT '',
            summary_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_report_latest_family ON report_latest_by_path(family);
        CREATE TABLE IF NOT EXISTS material_claim_versions (
            version_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            source_run_id TEXT NOT NULL,
            report_path TEXT NOT NULL,
            support_state TEXT NOT NULL,
            observed_utc TEXT NOT NULL,
            claim_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_material_claim_id ON material_claim_versions(claim_id);
        CREATE INDEX IF NOT EXISTS idx_material_claim_source_run ON material_claim_versions(source_run_id);
        """
    )
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(report_runs)").fetchall()}
    if "snapshot_path" not in columns:
        conn.execute("ALTER TABLE report_runs ADD COLUMN snapshot_path TEXT NOT NULL DEFAULT ''")
    latest_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(report_latest_by_path)").fetchall()}
    if "snapshot_path" not in latest_columns:
        conn.execute("ALTER TABLE report_latest_by_path ADD COLUMN snapshot_path TEXT NOT NULL DEFAULT ''")
    conn.commit()


def compact_evidence_database(
    db_path: Path,
    *,
    inline_limit: int = MAX_PAYLOAD_BYTES,
) -> dict[str, Any]:
    """Move oversized inline payloads to exact snapshots, then reclaim DB pages."""

    if not db_path.exists():
        return {
            "state": "GREEN",
            "migrated_payload_count": 0,
            "migrated_payload_bytes": 0,
            "failed_payload_count": 0,
            "database_bytes_before": 0,
            "database_bytes_after": 0,
        }
    before = int(db_path.stat().st_size)
    migrated_count = 0
    migrated_bytes = 0
    failures: list[dict[str, Any]] = []
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM report_runs WHERE payload_json != '{}' AND length(payload_json) > ? ORDER BY run_id",
            (max(1, int(inline_limit)),),
        ).fetchall()
        for row in rows:
            raw = str(row["payload_json"] or "{}").encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            expected = str(row["content_hash"] or "")
            if digest != expected:
                failures.append({"run_id": row["run_id"], "reason": "inline_payload_hash_mismatch"})
                continue
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                failures.append({"run_id": row["run_id"], "reason": "inline_payload_json_invalid"})
                continue
            source = resolve(str(row["report_path"] or "report.json"))
            snapshot_path = write_payload_snapshot(source, payload, raw, expected, str(row["family"] or "unknown"))
            snapshot = resolve(snapshot_path)
            if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes().rstrip(b"\n")).hexdigest() != expected:
                failures.append({"run_id": row["run_id"], "reason": "snapshot_replay_hash_mismatch", "snapshot_path": snapshot_path})
                continue
            conn.execute(
                "UPDATE report_runs SET payload_json='{}', payload_truncated=1, snapshot_path=? WHERE run_id=?",
                (snapshot_path, row["run_id"]),
            )
            conn.execute(
                "UPDATE report_latest_by_path SET payload_truncated=1, snapshot_path=? WHERE run_id=?",
                (snapshot_path, row["run_id"]),
            )
            migrated_count += 1
            migrated_bytes += len(raw)
        conn.commit()
    finally:
        conn.close()
    if migrated_count and not failures:
        vacuum = sqlite3.connect(str(db_path))
        try:
            vacuum.execute("VACUUM")
        finally:
            vacuum.close()
    after = int(db_path.stat().st_size) if db_path.exists() else 0
    return {
        "state": "GREEN" if not failures else "RED",
        "inline_payload_limit_bytes": int(inline_limit),
        "migrated_payload_count": migrated_count,
        "migrated_payload_bytes": migrated_bytes,
        "failed_payload_count": len(failures),
        "failures": failures[:50],
        "database_bytes_before": before,
        "database_bytes_after": after,
        "database_bytes_reclaimed": max(0, before - after),
        "reconstruction_contract": "snapshot JSON replays to the original report_runs.content_hash before inline payload removal",
        "non_claim": "Database compaction preserves evidence payloads and does not support model capability claims.",
        **zero_no_cheat_counters(),
    }


def ingest_reports(db_path: Path, paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    conn = connect(db_path)
    try:
        for path in paths:
            row = ingest_report_path(conn, path)
            if row:
                rows.append(row)
        conn.commit()
    finally:
        conn.close()
    return rows


def ingest_default_reports(db_path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    return ingest_reports(db_path, default_report_paths())


def write_json_report(
    out_path: str | Path,
    payload: dict[str, Any],
    *,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
    db_path: str | Path = DEFAULT_DB,
    ingest: bool = True,
) -> dict[str, Any]:
    """Write a mutable report view and optionally ingest it immediately.

    Most scripts should keep their stable ``reports/foo.json`` path for humans
    and automation. This helper makes that path a view over durable evidence
    instead of the only copy of the run.
    """
    out = resolve(out_path)
    write_json(out, payload)
    if markdown_path is not None and markdown_text is not None:
        write_text(resolve(markdown_path), markdown_text)
    if not ingest:
        return {}
    return ingest_report_path(resolve(db_path), out, payload)


def default_report_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in DEFAULT_REPORT_PATHS:
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    for pattern in DEFAULT_REPORT_GLOBS:
        for path in sorted(REPORTS.glob(pattern)):
            if path.exists() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def ingest_report_path(conn_or_db: sqlite3.Connection | Path, path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    owns_conn = isinstance(conn_or_db, Path)
    conn = connect(conn_or_db) if owns_conn else conn_or_db
    try:
        path = path if path.is_absolute() else ROOT / path
        if payload is None:
            payload = read_report_payload(path)
        if not isinstance(payload, dict) or not payload:
            return {}
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        content_hash = hashlib.sha256(raw).hexdigest()
        family = report_family(payload, path)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        metrics = extract_metrics(payload, family)
        payload_bytes = len(raw)
        payload_truncated = payload_bytes > MAX_PAYLOAD_BYTES
        snapshot_path = write_payload_snapshot(path, payload, raw, content_hash, family) if payload_truncated else ""
        payload_json = "{}" if payload_truncated else raw.decode("utf-8")
        created_utc = str(payload.get("created_utc") or summary.get("created_utc") or now())
        source_mtime = path.stat().st_mtime if path.exists() else datetime.now(timezone.utc).timestamp()
        run_id = stable_id("report_run", family, rel_or_abs(path), created_utc, content_hash)
        observed_utc = now()
        previous_latest = conn.execute(
            "SELECT * FROM report_latest_by_path WHERE report_path=?",
            (rel_or_abs(path),),
        ).fetchone()
        compression_record = build_compression_record(
            run_id=run_id,
            family=family,
            report_path=rel_or_abs(path),
            content_hash=content_hash,
            payload_bytes=payload_bytes,
            payload_truncated=payload_truncated,
            snapshot_path=snapshot_path,
        )
        defeater_record = build_defeater_record(
            previous_latest,
            run_id=run_id,
            family=family,
            report_path=rel_or_abs(path),
            policy=str(payload.get("policy") or ""),
            trigger_state=str(payload.get("trigger_state") or ""),
            passed=bool(payload.get("passed")),
            content_hash=content_hash,
            observed_utc=observed_utc,
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO report_runs (
                run_id, family, report_path, policy, created_utc, observed_utc,
                source_mtime, trigger_state, passed, content_hash, payload_bytes,
                payload_truncated, snapshot_path, summary_json, metrics_json, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                family,
                rel_or_abs(path),
                str(payload.get("policy") or ""),
                created_utc,
                observed_utc,
                float(source_mtime),
                str(payload.get("trigger_state") or ""),
                1 if bool(payload.get("passed")) else 0,
                content_hash,
                payload_bytes,
                1 if payload_truncated else 0,
                snapshot_path,
                json.dumps(summary, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
                payload_json,
            ),
        )
        claims_to_store = []
        if str(payload.get("policy") or "") != "project_theseus_report_evidence_store_v1":
            claims_to_store = report_evidence_integrity.extract_material_claims(
                payload,
                source_path=rel_or_abs(path),
            )
        for claim in claims_to_store:
            claim_id = str(claim.get("claim_id") or "")
            if not claim_id:
                continue
            version_id = stable_id("material_claim_version", run_id, claim_id, claim)
            conn.execute(
                """
                INSERT OR IGNORE INTO material_claim_versions (
                    version_id, claim_id, source_run_id, report_path,
                    support_state, observed_utc, claim_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    claim_id,
                    run_id,
                    rel_or_abs(path),
                    str(claim.get("support_state") or "argument"),
                    observed_utc,
                    json.dumps(claim, sort_keys=True),
                ),
            )
        if snapshot_path:
            conn.execute(
                "UPDATE report_runs SET snapshot_path=? WHERE run_id=? AND (snapshot_path IS NULL OR snapshot_path='')",
                (snapshot_path, run_id),
            )
        conn.execute(
            """
            INSERT INTO report_latest_by_path (
                report_path, run_id, family, policy, created_utc, observed_utc,
                source_mtime, trigger_state, passed, content_hash, payload_bytes,
                payload_truncated, snapshot_path, summary_json, metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_path) DO UPDATE SET
                run_id=excluded.run_id,
                family=excluded.family,
                policy=excluded.policy,
                created_utc=excluded.created_utc,
                observed_utc=excluded.observed_utc,
                source_mtime=excluded.source_mtime,
                trigger_state=excluded.trigger_state,
                passed=excluded.passed,
                content_hash=excluded.content_hash,
                payload_bytes=excluded.payload_bytes,
                payload_truncated=excluded.payload_truncated,
                snapshot_path=excluded.snapshot_path,
                summary_json=excluded.summary_json,
                metrics_json=excluded.metrics_json
            """,
            (
                rel_or_abs(path),
                run_id,
                family,
                str(payload.get("policy") or ""),
                created_utc,
                observed_utc,
                float(source_mtime),
                str(payload.get("trigger_state") or ""),
                1 if bool(payload.get("passed")) else 0,
                content_hash,
                payload_bytes,
                1 if payload_truncated else 0,
                snapshot_path,
                json.dumps(summary, sort_keys=True),
                json.dumps(metrics, sort_keys=True),
            ),
        )
        if owns_conn:
            conn.commit()
        result = {
            "run_id": run_id,
            "family": family,
            "report_path": rel_or_abs(path),
            "content_hash": content_hash[:16],
            "created_utc": created_utc,
            "payload_bytes": payload_bytes,
            "payload_truncated": payload_truncated,
            "snapshot_path": snapshot_path,
            "metrics": metrics,
        }
        if compression_record:
            result["compression_record"] = compression_record
        if defeater_record:
            result["defeater_record"] = defeater_record
        return result
    finally:
        if owns_conn:
            conn.close()


def best_payload_for_family(db_path: Path = DEFAULT_DB, family: str = "") -> dict[str, Any]:
    if not family or not db_path.exists():
        return {}
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM report_runs WHERE family=? ORDER BY observed_utc DESC",
            (family,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return {}
    best = max(rows, key=lambda row: evidence_sort_key(row))
    payload_json = str(best["payload_json"] or "{}")
    payload = json.loads(payload_json) if payload_json and payload_json != "{}" else {}
    if not payload:
        payload = {
            "policy": best["policy"],
            "trigger_state": best["trigger_state"],
            "passed": bool(best["passed"]),
            "summary": json.loads(str(best["summary_json"] or "{}")),
            "_payload_truncated": bool(best["payload_truncated"]),
        }
    if payload:
        payload["_evidence_store"] = {
            "run_id": best["run_id"],
            "family": best["family"],
            "report_path": best["report_path"],
            "content_hash": best["content_hash"],
            "observed_utc": best["observed_utc"],
            "payload_truncated": bool(best["payload_truncated"]),
            "snapshot_path": best["snapshot_path"] if "snapshot_path" in best.keys() else "",
        }
    return payload


def material_claim_versions(db_path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT claim_json FROM material_claim_versions
            WHERE report_path != 'reports/report_evidence_store.json'
            ORDER BY observed_utc, version_id
            """
        ).fetchall()
    finally:
        conn.close()
    versions: list[dict[str, Any]] = []
    for row in rows:
        try:
            value = json.loads(str(row["claim_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value:
            versions.append(value)
    return versions


def families_in_store(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT family FROM report_runs").fetchall()
    finally:
        conn.close()
    return {str(row["family"]) for row in rows}


def run_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM report_runs").fetchone()
    finally:
        conn.close()
    return int(row["n"]) if row else 0


def evidence_store_summary(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {
            "stored_run_count": 0,
            "latest_path_count": 0,
            "truncated_payload_count": 0,
            "snapshot_count": 0,
            "truncated_without_snapshot_count": 0,
        }
    conn = connect(db_path)
    try:
        runs = int(conn.execute("SELECT COUNT(*) AS n FROM report_runs").fetchone()["n"])
        latest = int(conn.execute("SELECT COUNT(*) AS n FROM report_latest_by_path").fetchone()["n"])
        truncated = int(conn.execute("SELECT COUNT(*) AS n FROM report_runs WHERE payload_truncated=1").fetchone()["n"])
        snapshots = int(
            conn.execute("SELECT COUNT(*) AS n FROM report_runs WHERE snapshot_path IS NOT NULL AND snapshot_path != ''").fetchone()["n"]
        )
        missing = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM report_runs WHERE payload_truncated=1 AND (snapshot_path IS NULL OR snapshot_path='')"
            ).fetchone()["n"]
        )
    finally:
        conn.close()
    return {
        "stored_run_count": runs,
        "latest_path_count": latest,
        "truncated_payload_count": truncated,
        "snapshot_count": snapshots,
        "truncated_without_snapshot_count": missing,
    }


def current_report_index(db_path: Path, paths: list[Path], sample_limit: int = 20) -> dict[str, Any]:
    current_rows = []
    unstored = []
    truncated_without_snapshot = []
    conn = connect(db_path)
    try:
        for path in paths:
            path = path if path.is_absolute() else ROOT / path
            if not path.exists():
                continue
            payload = read_report_payload(path)
            if not isinstance(payload, dict) or not payload:
                continue
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            content_hash = hashlib.sha256(raw).hexdigest()
            report_path = rel_or_abs(path)
            latest = conn.execute(
                "SELECT * FROM report_latest_by_path WHERE report_path=?",
                (report_path,),
            ).fetchone()
            stored = conn.execute(
                "SELECT run_id, snapshot_path FROM report_runs WHERE report_path=? AND content_hash=? LIMIT 1",
                (report_path, content_hash),
            ).fetchone()
            snapshot_path = str(stored["snapshot_path"] or "") if stored else ""
            payload_truncated = len(raw) > MAX_PAYLOAD_BYTES
            row = {
                "report_path": report_path,
                "family": report_family(payload, path),
                "created_utc": str(payload.get("created_utc") or ""),
                "trigger_state": str(payload.get("trigger_state") or ""),
                "content_hash": content_hash[:16],
                "payload_bytes": len(raw),
                "stored": bool(stored),
                "latest_run_id": str(latest["run_id"]) if latest else "",
                "snapshot_path": snapshot_path,
                "payload_truncated": payload_truncated,
            }
            current_rows.append(row)
            if not stored:
                unstored.append(row)
            if payload_truncated and not snapshot_path:
                truncated_without_snapshot.append(row)
    finally:
        conn.close()
    return {
        "current_report_count": len(current_rows),
        "current_unstored_count": len(unstored),
        "current_truncated_without_snapshot_count": len(truncated_without_snapshot),
        "unstored_samples": unstored[:sample_limit],
        "truncated_without_snapshot_samples": truncated_without_snapshot[:sample_limit],
        "latest_samples": sorted(current_rows, key=lambda row: row["payload_bytes"], reverse=True)[:sample_limit],
    }


def write_payload_snapshot(path: Path, payload: dict[str, Any], raw: bytes, content_hash: str, family: str) -> str:
    created = str(payload.get("created_utc") or now())
    stamp = safe_filename(created.replace("+00:00", "Z")).strip("_") or safe_filename(now())
    family_slug = safe_filename(family) or "unknown_family"
    stem = safe_filename(path.stem) or "report"
    out_dir = DEFAULT_SNAPSHOT_DIR / family_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}_{stem}_{content_hash[:16]}.json"
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(raw + b"\n")
        tmp.replace(out)
    return rel_or_abs(out)


def build_compression_record(
    *,
    run_id: str,
    family: str,
    report_path: str,
    content_hash: str,
    payload_bytes: int,
    payload_truncated: bool,
    snapshot_path: str,
) -> dict[str, Any]:
    if not payload_truncated:
        return {}
    return {
        "record_type": "compression_record",
        "record_id": stable_id("compression_record", run_id, snapshot_path, content_hash),
        "run_id": run_id,
        "family": family,
        "report_path": report_path,
        "content_hash": content_hash,
        "payload_bytes": int(payload_bytes),
        "payload_truncated": True,
        "snapshot_path": snapshot_path,
        "codec": "json_snapshot",
        "compression_scope": "large_report_payload_snapshot",
        "reconstruction_contract": "snapshot_path stores the exact JSON payload omitted from inline SQLite payload_json; content_hash is the canonical sha256 over sorted JSON bytes",
        "support_state": "SUPPORTED" if snapshot_path else "UNSUPPORTED",
        "status": "SUPPORTED" if snapshot_path else "BLOCKED",
        "evidence_ref": report_path,
        **zero_no_cheat_counters(),
        "non_claim": "Compression record proves large-report snapshot traceability; it is not model capability evidence.",
    }


def compressed_artifact_record_from_compression_record(record: dict[str, Any], *, source_system: str) -> dict[str, Any]:
    record_id = str(record.get("record_id") or stable_id("compression_record", record))
    status = str(record.get("status") or "").upper()
    source_artifact = str(record.get("report_path") or record.get("original_path") or record.get("snapshot_path") or "")
    fallback_artifact = str(record.get("snapshot_path") or record.get("archive_path") or source_artifact)
    admission_state = "cold_archive_candidate" if status in {"SUPPORTED", "PLANNED"} and fallback_artifact else "not_admitted"
    exact_replay_status = "not_run"
    if status not in {"SUPPORTED", "PLANNED"}:
        exact_replay_status = "blocked"
    return {
        "record_type": "compressed_artifact_record",
        "record_id": stable_id("compressed_artifact_record", source_system, record_id),
        "artifact_id": f"compressed-artifact-{record_id}",
        "source_artifact": source_artifact,
        "task_family": str(record.get("family") or record.get("compression_scope") or "report_evidence"),
        "access_pattern": str(record.get("compression_scope") or "large_report_payload_snapshot"),
        "admission_state": admission_state,
        "compression_method": str(record.get("codec") or "json_snapshot"),
        "reconstruction_contract": str(record.get("reconstruction_contract") or "No reconstruction contract recorded."),
        "declared_use_envelope": [
            "evidence-store browsing",
            "artifact retention routing",
            "operator audit citation",
        ],
        "ratio_claim_state": "not_measured",
        "codec_parameters": [
            f"codec={record.get('codec') or 'unknown'}",
            f"content_hash={record.get('content_hash') or ''}",
        ],
        "metadata_costs": [
            f"payload_bytes={int(record.get('payload_bytes') or 0)}",
            f"archived_bytes={int(record.get('archived_bytes') or 0)}",
        ],
        "residual_coding": [
            "full source artifact remains authoritative",
            "payload hash and fallback pointer must travel with compressed representation",
        ],
        "probe_plan": [
            "verify fallback artifact exists",
            "rehash decoded payload before exact evidence use",
            "route to full artifact when exact replay, benchmark, or training evidence is required",
        ],
        "fallback_artifact": fallback_artifact,
        "fallback_trigger": "Use the full source artifact or archived snapshot for exact replay, training evidence, benchmark evidence, or support-state promotion.",
        "decode_determinism": "Deterministic only after an explicit decode/rehash replay check; this record does not run that check.",
        "exact_replay_status": exact_replay_status,
        "consumer_policy": "May support routing and audit orientation. Exact claims require a replay command that resolves the fallback artifact and matches the recorded content hash.",
        "utility_tests": [
            "VIEA schema-shape gate",
            "report-evidence snapshot-presence gate",
        ],
        "support_state_effect": "record_shape_only",
        "evidence_refs": [str(record.get("evidence_ref") or source_artifact or "reports/report_evidence_store.json")],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "no compression ratio measured",
            "no downstream utility result",
            "not learned-generation evidence",
        ],
    }


def compression_receipt_from_compression_record(record: dict[str, Any], *, source_system: str) -> dict[str, Any]:
    record_id = str(record.get("record_id") or stable_id("compression_record", record))
    payload_bytes = int(record.get("payload_bytes") or 0)
    archived_bytes = int(record.get("archived_bytes") or 0)
    return {
        "record_type": "compression_receipt",
        "record_id": stable_id("compression_receipt", source_system, record_id),
        "artifact_id": f"compression-receipt-{record_id}",
        "receipt_state": "candidate",
        "reconstruction_contract": str(record.get("reconstruction_contract") or "No reconstruction contract recorded."),
        "public_law_family": str(record.get("codec") or "json_snapshot"),
        "seed": str(record.get("content_hash") or record_id),
        "search_bound": "No learned or stochastic search is credited; this is a deterministic storage/retention receipt.",
        "generated_regions": [
            str(record.get("snapshot_path") or record.get("archive_path") or record.get("report_path") or record.get("original_path") or "")
        ],
        "verification_result": str(record.get("support_state") or record.get("status") or "metadata_recorded"),
        "repair_residual": "No semantic repair is claimed; route to the full artifact if replay or utility probes fail.",
        "fallback_threshold": "Use literal/full artifact whenever exact replay, benchmark evidence, training evidence, or support-state promotion is requested without a passing decode check.",
        "interface_costs": [
            "record id",
            "content hash",
            "fallback pointer",
            "codec metadata",
        ],
        "consumer_policy": "Receipts can guide storage and audit routing only; they cannot replace full artifacts for claim support without replay verification.",
        "use_permissions": [
            "artifact retention audit",
            "report evidence compaction",
            "operator governance export",
        ],
        "proxy_rate_status": "not_run",
        "final_serialization_status": "not_run",
        "rate_accounting": {
            "payload_bytes": payload_bytes,
            "archived_bytes": archived_bytes,
            "metadata_bytes": len(json.dumps(record, sort_keys=True)),
            "ratio_claimed": False,
        },
        "support_state_effect": "record_shape_only",
        "evidence_refs": [str(record.get("evidence_ref") or record.get("report_path") or "reports/report_evidence_store.json")],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claims": [
            "This receipt is not a compression benchmark.",
            "no codec correctness result",
            "no final serialized-rate result",
            "not learned-generation evidence",
        ],
    }


def build_defeater_record(
    previous_latest: sqlite3.Row | None,
    *,
    run_id: str,
    family: str,
    report_path: str,
    policy: str,
    trigger_state: str,
    passed: bool,
    content_hash: str,
    observed_utc: str,
) -> dict[str, Any]:
    if not previous_latest:
        return {}
    previous_hash = str(previous_latest["content_hash"] or "")
    if previous_hash == content_hash:
        return {}
    previous_support = support_state_for(str(previous_latest["trigger_state"] or ""), bool(previous_latest["passed"]))
    current_support = support_state_for(trigger_state, passed)
    defeater_type = "latest_view_support_state_change" if previous_support != current_support else "latest_view_content_supersession"
    return {
        "record_type": "defeater_record",
        "record_id": stable_id("defeater_record", report_path, str(previous_latest["run_id"] or ""), run_id, previous_hash, content_hash),
        "family": family,
        "report_path": report_path,
        "policy": policy,
        "defeater_type": defeater_type,
        "defeated_run_id": str(previous_latest["run_id"] or ""),
        "defeating_run_id": run_id,
        "previous_content_hash": previous_hash,
        "current_content_hash": content_hash,
        "previous_support_state": previous_support,
        "current_support_state": current_support,
        "previous_trigger_state": str(previous_latest["trigger_state"] or ""),
        "current_trigger_state": trigger_state,
        "observed_utc": observed_utc,
        "support_state": "SUPPORTED",
        "status": "SUPPORTED",
        "evidence_ref": report_path,
        **zero_no_cheat_counters(),
        "non_claim": "Defeater record supersedes a mutable latest-view pointer only; immutable historical report runs remain retained.",
    }


def support_state_for(trigger_state: str, passed: bool) -> str:
    state = str(trigger_state or "").upper()
    if state == "GREEN":
        return "SUPPORTED"
    if state == "YELLOW":
        return "PARTIAL"
    if state == "RED":
        return "UNSUPPORTED"
    if passed:
        return "SUPPORTED"
    return state or "UNKNOWN"


def zero_no_cheat_counters() -> dict[str, int]:
    return {key: 0 for key in NO_CHEAT_COUNTERS}


def safe_filename(value: str) -> str:
    chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("._")[:160]


def evidence_sort_key(row: sqlite3.Row) -> tuple[Any, ...]:
    family = str(row["family"] or "")
    metrics = json.loads(str(row["metrics_json"] or "{}"))
    passed = int(row["passed"] or 0)
    source_mtime = float(row["source_mtime"] or 0.0)
    if family == "conversation_multiturn":
        graduated = 1 if metrics.get("graduated") else 0
        case_count = int(metrics.get("case_count") or 0)
        accuracy = float(metrics.get("accuracy") or 0.0)
        large = 1 if str(metrics.get("suite_mode") or "").lower() == "large" else 0
        return graduated, case_count, large, passed, accuracy, source_mtime
    if family == "broad_transfer":
        task_count = int(metrics.get("real_public_task_count") or 0)
        pass_rate = float(metrics.get("real_public_pass_rate") or 0.0)
        clean = 1 if int(metrics.get("no_cheat_violation_count") or 0) == 0 else 0
        return clean, task_count, pass_rate, source_mtime
    if family == "real_code_benchmark":
        task_count = int(metrics.get("public_task_count") or metrics.get("total_case_count") or 0)
        pass_rate = float(metrics.get("real_public_task_pass_rate") or metrics.get("multi_stream_pass_rate") or 0.0)
        clean = 1 if int(metrics.get("template_like_candidate_count") or 0) == 0 and int(metrics.get("loop_closure_candidate_count") or 0) == 0 else 0
        return clean, task_count, pass_rate, source_mtime
    if family in {"code_residual_curriculum", "high_transfer_code_residual_curriculum"}:
        rows = int(metrics.get("private_train_rows") or metrics.get("row_count") or metrics.get("selected_rows") or 0)
        clean = 1 if metrics.get("no_public_training_data_used", True) else 0
        return clean, rows, passed, source_mtime
    if family == "code_lm_closure":
        steps = int(metrics.get("steps_completed") or metrics.get("trained_steps") or metrics.get("max_work_steps") or 0)
        eval_score = float(metrics.get("private_eval_pass_rate") or metrics.get("eval_pass_rate") or 0.0)
        return passed, steps, eval_score, source_mtime
    if family == "decoder_plan_ir_private_pressure":
        rows = int(metrics.get("plan_ir_row_count") or 0)
        complete = float(metrics.get("complete_plan_order_rate") or 0.0)
        return passed, rows, complete, source_mtime
    if family == "decoder_plan_ir_code_lm_adapter":
        rows = int(metrics.get("code_lm_row_count") or metrics.get("joined_row_count") or 0)
        contract_rate = float(metrics.get("contract_row_rate") or 0.0)
        return passed, rows, contract_rate, source_mtime
    if family == "deterministic_tool_substrate":
        cases = int(metrics.get("private_case_result_count") or metrics.get("private_case_count") or 0)
        solve_rate = float(metrics.get("exact_solve_rate") or metrics.get("tool_on_solve_rate") or 0.0)
        clean = 1 if int(metrics.get("fallback_return_count") or 0) == 0 and int(metrics.get("public_training_rows_written") or 0) == 0 else 0
        return clean, cases, solve_rate, passed, source_mtime
    if family == "theseus_plan_compiler":
        nodes = int(metrics.get("compiled_node_count") or metrics.get("node_count") or 0)
        failures = int(metrics.get("hard_gate_failure_count") or metrics.get("goal_lint_hard_failure_count") or 0)
        return 1 if failures == 0 else 0, nodes, passed, source_mtime
    if family == "viea_execution_spine":
        cases = int(metrics.get("compiled_case_count") or metrics.get("case_count") or 0)
        pass_rate = float(metrics.get("compiled_useful_completion_rate") or metrics.get("verifier_pass_rate") or 0.0)
        failures = int(metrics.get("hard_gate_failure_count") or 0)
        clean = 1 if int(metrics.get("fallback_return_count") or 0) == 0 and int(metrics.get("public_training_rows_written") or 0) == 0 else 0
        return clean, 1 if failures == 0 else 0, cases, pass_rate, passed, source_mtime
    if family == "viea_procedural_tools":
        verified = int(metrics.get("verified_procedural_tool_count") or 0)
        total = int(metrics.get("procedural_tool_count") or 0)
        return verified, total, passed, source_mtime
    return passed, source_mtime


def report_family(payload: dict[str, Any], path: Path) -> str:
    policy = str(payload.get("policy") or "").lower()
    stem = path.stem.lower()
    if "multi_turn_conversation_benchmark" in policy or "multi_turn_conversation" in stem:
        return "conversation_multiturn"
    if "high_transfer_curriculum_scheduler" in policy or stem == "high_transfer_curriculum_scheduler":
        return "high_transfer_curriculum"
    if stem.startswith("high_transfer_") and "code_residual_curriculum" in stem:
        return "high_transfer_code_residual_curriculum"
    if "code_residual_curriculum" in stem:
        return "code_residual_curriculum"
    if "code_lm_closure" in stem:
        return "code_lm_closure"
    if "edge_obligation_decode_gate" in stem:
        return "edge_obligation_decode_gate"
    if "decoder_plan_ir_private_pressure" in stem or "decoder_plan_ir_private_pressure" in policy:
        return "decoder_plan_ir_private_pressure"
    if "decoder_plan_ir_code_lm_adapter" in stem or "decoder_plan_ir_code_lm_adapter" in policy:
        return "decoder_plan_ir_code_lm_adapter"
    if "public_transfer_residual_packet" in stem or "public_transfer_residual_packet" in policy:
        return "public_transfer_residual_packet"
    if "deterministic_tool" in stem or "deterministic_tool" in policy:
        return "deterministic_tool_substrate"
    if "viea_execution_spine" in stem or "viea_execution_spine" in policy:
        return "viea_execution_spine"
    if "viea_verified_procedural_tools" in stem or "viea_verified_procedural_tools" in policy:
        return "viea_procedural_tools"
    if "viea_research_implementation_matrix" in stem or "viea_research_implementation_matrix" in policy:
        return "viea_research_matrix"
    if "theseus_plan_compiler" in stem or "theseus_plan_compiled_dags" in stem or "plan_compiler" in policy:
        return "theseus_plan_compiler"
    if "real_code_benchmark_graduation" in stem:
        return "real_code_benchmark"
    if "broad_transfer" in stem:
        return "broad_transfer"
    if "broad_code_calibration_scheduler" in stem:
        return "broad_code_calibration_scheduler"
    if "transfer_generalization_audit" in stem:
        return "transfer_generalization"
    if "learning_scoreboard" in policy or stem == "learning_scoreboard":
        return "learning_scoreboard"
    if "candidate_promotion" in stem:
        return "candidate_promotion"
    if "personality_runtime" in stem:
        return "personality_runtime"
    if "hive_work_board" in stem:
        return "hive_work_board"
    return stem


def extract_metrics(payload: dict[str, Any], family: str) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    metrics: dict[str, Any] = {
        "trigger_state": payload.get("trigger_state"),
        "passed": payload.get("passed"),
    }
    for key in [
        "accuracy",
        "average_score",
        "case_count",
        "turn_count",
        "graduated",
        "saturated",
        "suite_mode",
        "real_public_task_count",
        "real_public_pass_rate",
        "no_cheat_violation_count",
        "ready_task_count",
        "critical_task_count",
        "public_task_count",
        "total_case_count",
        "real_public_task_pass_rate",
        "multi_stream_pass_rate",
        "template_like_candidate_count",
        "loop_closure_candidate_count",
        "private_train_rows",
        "row_count",
        "selected_rows",
        "steps_completed",
        "trained_steps",
        "max_work_steps",
        "private_eval_pass_rate",
        "eval_pass_rate",
        "no_public_training_data_used",
        "plan_ir_row_count",
        "complete_plan_order_rate",
        "return_contract_rate",
        "skeleton_obligation_rate",
        "repair_policy_rate",
        "public_candidate_count_inspected_for_pressure_only",
        "reason_for_teacher",
        "aggregate_public_pass_rate",
        "cards_below_floor",
        "dominant_residuals",
        "private_plan_ir_rows",
        "code_lm_row_count",
        "joined_row_count",
        "contract_row_count",
        "private_solution_row_count",
        "public_leak_flag_count",
        "contract_row_rate",
        "private_solution_row_rate",
        "tool_card_count",
        "available_tool_count",
        "private_case_result_count",
        "private_case_count",
        "solved_count",
        "verified_solved_count",
        "exact_solve_rate",
        "tool_on_solve_rate",
        "tool_off_solve_rate",
        "fallback_return_count",
        "public_training_rows_written",
        "compiled_goal_count",
        "compiled_node_count",
        "compiled_edge_count",
        "node_count",
        "trace_row_count",
        "goal_lint_hard_failure_count",
        "hard_gate_failure_count",
        "local_deterministic_tool_packet_count",
        "deterministic_tool_requirement_count",
        "compiled_case_count",
        "old_case_count",
        "compiled_useful_completion_rate",
        "old_useful_completion_rate",
        "compiled_lease_count",
        "compiled_checkpoint_count",
        "residual_count",
        "training_evidence_row_count",
        "verified_procedural_tool_count",
        "procedural_tool_count",
        "candidate_needs_repetition_count",
    ]:
        if key in summary:
            metrics[key] = summary.get(key)
    if family == "viea_procedural_tools":
        for key in ["procedural_tool_count", "verified_procedural_tool_count", "candidate_needs_repetition_count"]:
            if key in payload:
                metrics[key] = payload.get(key)
    if family == "viea_execution_spine":
        compiled_execution = payload.get("compiled_execution") if isinstance(payload.get("compiled_execution"), dict) else {}
        for key in ["verifier_pass_rate", "duplicate_work_count", "retry_count", "unknown_count", "tool_fault_count"]:
            if key in compiled_execution:
                metrics[key] = compiled_execution.get(key)
    if "no_public_training_data_used" in payload:
        metrics["no_public_training_data_used"] = payload.get("no_public_training_data_used")
    if family == "conversation_multiturn":
        metrics.setdefault("accuracy", summary.get("accuracy", summary.get("average_score", 0.0)))
        metrics.setdefault("case_count", summary.get("case_count", 0))
        metrics.setdefault("suite_mode", summary.get("suite_mode", "unknown"))
        metrics.setdefault("graduated", summary.get("graduated", False))
    if family == "decoder_plan_ir_private_pressure":
        coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
        for key in ["complete_plan_order_rate", "return_contract_rate", "skeleton_obligation_rate", "repair_policy_rate"]:
            if key in coverage:
                metrics[key] = coverage.get(key)
        metrics.setdefault("plan_ir_row_count", summary.get("plan_ir_row_count", 0))
        metrics.setdefault("private_task_count", summary.get("private_task_count", 0))
        metrics.setdefault(
            "public_candidate_count_inspected_for_pressure_only",
            summary.get("public_candidate_count_inspected_for_pressure_only", 0),
        )
    if family == "decoder_plan_ir_code_lm_adapter":
        for key in [
            "code_lm_row_count",
            "joined_row_count",
            "contract_row_count",
            "private_solution_row_count",
            "public_leak_flag_count",
            "contract_row_rate",
            "private_solution_row_rate",
            "private_plan_ir_rows",
        ]:
            if key in summary:
                metrics[key] = summary.get(key)
    return metrics


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "policy": payload.get("policy"),
        "trigger_state": payload.get("trigger_state"),
        "passed": payload.get("passed"),
        "summary": payload.get("summary", {}),
        "evidence_store": payload.get("_evidence_store", {}),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Report Evidence Store",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- database: `{payload.get('database')}`",
        f"- stored_run_count: `{summary.get('stored_run_count')}`",
        f"- family_count: `{summary.get('family_count')}`",
        f"- latest_path_count: `{summary.get('latest_path_count')}`",
        f"- current_report_count: `{summary.get('current_report_count')}`",
        f"- current_unstored_count: `{summary.get('current_unstored_count')}`",
        f"- truncated_payload_count: `{summary.get('truncated_payload_count')}`",
        f"- snapshot_count: `{summary.get('snapshot_count')}`",
        f"- current_truncated_without_snapshot_count: `{summary.get('current_truncated_without_snapshot_count')}`",
        f"- standard_evidence_pack_count: `{summary.get('standard_evidence_pack_count')}` valid=`{summary.get('valid_standard_evidence_pack_count')}`",
        f"- epistemic_tcb_state: `{summary.get('epistemic_tcb_state')}` roots=`{summary.get('epistemic_tcb_root_count')}`",
        f"- material_claim_count: `{summary.get('material_claim_count')}` dependents=`{summary.get('claim_dependent_surface_edge_count')}`",
        f"- receipt_randomized_deep_replay_count: `{summary.get('receipt_randomized_deep_replay_count')}`",
        f"- db_compacted_payload_count: `{summary.get('db_compacted_payload_count')}` bytes=`{summary.get('db_compacted_payload_bytes')}`",
        "",
        "## Current Report Index",
        "",
    ]
    current_index = payload.get("current_index") if isinstance(payload.get("current_index"), dict) else {}
    unstored = current_index.get("unstored_samples") if isinstance(current_index.get("unstored_samples"), list) else []
    missing_snapshots = (
        current_index.get("truncated_without_snapshot_samples")
        if isinstance(current_index.get("truncated_without_snapshot_samples"), list)
        else []
    )
    if unstored:
        lines.append("### Unstored Current Reports")
        for row in unstored[:10]:
            lines.append(f"- `{row.get('report_path')}` hash={row.get('content_hash')}")
        lines.append("")
    if missing_snapshots:
        lines.append("### Current Truncated Reports Missing Snapshots")
        for row in missing_snapshots[:10]:
            lines.append(f"- `{row.get('report_path')}` bytes={row.get('payload_bytes')}")
        lines.append("")
    if not unstored and not missing_snapshots:
        lines.extend(["- Current indexed reports are stored; current truncated reports have snapshots.", ""])

    lines.extend([
        "## Best By Family",
        "",
    ])
    for family, report in (payload.get("best_by_family") or {}).items():
        report_summary = report.get("summary") or {}
        lines.append(f"- `{family}` state={report.get('trigger_state')} summary={json.dumps(report_summary, sort_keys=True)[:500]}")
    lines.append("")
    return "\n".join(lines)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_report_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    if isinstance(payload, dict) and payload.get("policy") == theseus_archive_resolver.ARCHIVE_POINTER_POLICY:
        resolved = theseus_archive_resolver.read_json_follow_pointer(path, default={})
        return resolved if isinstance(resolved, dict) else {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
