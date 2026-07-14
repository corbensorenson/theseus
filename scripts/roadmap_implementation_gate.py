#!/usr/bin/env python3
"""Governed implementation gate for the AI-book-derived Theseus roadmap.

The roadmap is allowed to be ambitious. This gate keeps that ambition tied to
registered surfaces, stable capability fields, runnable integration paths, and
honest completion states instead of another prose-only checklist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


DEFAULT_MATRIX = ROOT / "configs" / "roadmap_implementation_matrix.json"
DEFAULT_REGISTRY = ROOT / "configs" / "project_manifest_registry.json"
DEFAULT_OUT = ROOT / "reports" / "roadmap_implementation_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "roadmap_implementation_gate.md"
DEFAULT_PRE_TRAINING_OUT = ROOT / "reports" / "roadmap_pre_training_architecture_readiness_gate.json"
DEFAULT_PRE_TRAINING_MARKDOWN = ROOT / "reports" / "roadmap_pre_training_architecture_readiness_gate.md"
DEFAULT_CROSSWALK = ROOT / "reports" / "book_to_theseus_crosswalk.json"
DEFAULT_PROJECT_STEWARD = ROOT / "configs" / "project_steward.json"
DEFAULT_AI_BOOK_ROOT = ROOT.parent / "AI_book"
REQUIRED_PHASES = set(range(20))
ALLOWED_PHASE_STATES = {"implemented", "wired", "partial", "missing", "frozen"}
DONE_STATES = {"implemented", "wired"}
EXTERNAL_FREEZE_TERMS = {"peer", "reachable", "external", "travel", "network", "coordinator_unreachable", "no route to host"}
DISALLOWED_OUT_OF_SCOPE_TERMS = {
    "public_benchmark_training",
    "serve_external_inference",
    "count_router_as_learned_generation",
    "count_template_as_learned_generation",
    "long_training_as_implementation_proof",
}
REQUIRED_SUPPORT_STATES = {
    "argument",
    "prototype-backed",
    "synthetic-test-backed",
    "empirical-test-backed",
    "replayable-reference-backed",
}
REQUIRED_BOOK_CROSSWALK_FIELDS = {
    "chapter_ordinal",
    "chapter_id",
    "chapter_title",
    "part_id",
    "book_file",
    "book_minimal_implementation",
    "book_beyond_state_of_art",
    "primary_track_id",
    "phase_refs",
    "minimum_theseus_slice",
    "beyond_sota_theseus_endpoint",
    "support_state_target",
    "current_theseus_state",
    "required_gate_or_fixture",
    "no_claims",
}
REQUIRED_PLANNED_CODEX_BACKLOG_FIELDS = {
    "backlog_id",
    "source_basis",
    "technique_family",
    "owned_phase_refs",
    "track_id",
    "status",
    "dependency",
    "acceptance_gate",
    "support_state_target",
    "no_claim_boundary",
}
REQUIRED_BOOK_FUTURE_CANDIDATE_FIELDS = {
    "candidate_id",
    "title",
    "item_kind",
    "book_disposition",
    "phase_refs",
    "entry_condition",
    "theseus_scope",
    "acceptance_boundary",
    "non_claim_boundary",
}
ALLOWED_BOOK_FUTURE_ITEM_KINDS = {"chapter_candidate", "cross_cutting_section"}
BOOK_CROSSWALK_SOURCE_FIELDS = {
    "chapter_ordinal": "chapter_ordinal",
    "chapter_id": "chapter_id",
    "chapter_title": "chapter_title",
    "part_id": "part_id",
    "part_title": "part_title",
    "book_file": "book_file",
    "book_claim_label": "book_claim_label",
    "book_evidence_level": "book_evidence_level",
    "book_minimal_implementation": "book_minimal_implementation",
    "book_beyond_state_of_art": "book_beyond_state_of_art",
    "book_interfaces": "book_interfaces",
    "book_invariants": "book_invariants",
    "book_failure_modes": "book_failure_modes",
    "codex_test_count": "codex_test_count",
    "representative_codex_tests": "representative_codex_tests",
}
HIVE_ARTIFACT_CITATION_REPORTS = [
    "reports/hive_installer_artifacts.json",
    "reports/hive_artifact_index_smoke.json",
    "reports/hive_artifact_sync.json",
    "reports/hive_artifact_merge_summary.json",
]
AI_BOOK_SOURCE_EXTENSIONS = {".md", ".qmd", ".json", ".lean", ".toml", ".yml", ".yaml"}
AI_BOOK_IGNORED_PARTS = {
    ".git",
    ".github",
    ".quarto",
    ".lake",
    "_archive",
    "_freeze",
    "_site",
    ".venv",
    "__pycache__",
    "build",
    "node_modules",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=rel(DEFAULT_MATRIX))
    parser.add_argument("--registry", default=rel(DEFAULT_REGISTRY))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--crosswalk-out", default=rel(DEFAULT_CROSSWALK))
    parser.add_argument("--project-steward", default=rel(DEFAULT_PROJECT_STEWARD))
    parser.add_argument("--ai-book-root", default=str(DEFAULT_AI_BOOK_ROOT))
    parser.add_argument("--gate", action="store_true", help="Print compact gate summary.")
    parser.add_argument(
        "--require-pre-training-ready",
        action="store_true",
        help="Fail closed unless the book-derived architecture is ready for training/public calibration focus.",
    )
    args = parser.parse_args()
    if args.require_pre_training_ready:
        if args.out == rel(DEFAULT_OUT):
            args.out = rel(DEFAULT_PRE_TRAINING_OUT)
        if args.markdown_out == rel(DEFAULT_MARKDOWN):
            args.markdown_out = rel(DEFAULT_PRE_TRAINING_MARKDOWN)

    started = time.perf_counter()
    matrix_path = resolve(args.matrix)
    registry_path = resolve(args.registry)
    matrix = read_json(matrix_path)
    registry = read_json(registry_path)
    project_steward = read_json(resolve(args.project_steward))
    ai_book_root = resolve_external(args.ai_book_root)
    report = build_report(
        matrix_path,
        registry_path,
        matrix,
        registry,
        started,
        ai_book_root=ai_book_root,
    )
    crosswalk_path = resolve(args.crosswalk_out)
    crosswalk = build_book_to_theseus_crosswalk(
        matrix_path,
        registry_path,
        matrix,
        report,
        crosswalk_path,
        ai_book_root=ai_book_root,
        project_steward=project_steward,
    )
    report["outputs"] = {"book_to_theseus_crosswalk": rel(crosswalk_path)}
    report["summary"]["book_to_theseus_crosswalk_item_count"] = crosswalk["summary"]["crosswalk_item_count"]
    report["summary"]["book_to_theseus_ai_book_source_file_count"] = crosswalk["summary"]["ai_book_source_file_count"]
    report["summary"]["book_to_theseus_ai_book_source_manifest_hash"] = crosswalk["summary"]["ai_book_source_manifest_hash"]
    report["summary"]["book_to_theseus_stale_phase_count"] = crosswalk["summary"]["stale_phase_count"]
    report["summary"]["book_to_theseus_roadmap_backlog_item_count"] = crosswalk["summary"]["roadmap_backlog_item_count"]
    report["summary"]["theseus_to_book_evidence_count"] = crosswalk["summary"]["theseus_to_book_evidence_count"]
    report["summary"]["book_to_theseus_source_sync_smoke_passed"] = crosswalk["summary"]["source_sync_smoke_passed"]
    report["summary"]["public_safe_evidence_smoke_passed"] = crosswalk["summary"]["public_safe_evidence_smoke_passed"]
    if args.require_pre_training_ready and not report["pre_training_architecture_readiness"]["ready"]:
        readiness_gap = gap(
            "pre_training_architecture_readiness",
            "architecture_not_ready_for_training",
            {
                "blocker_count": report["pre_training_architecture_readiness"]["blocker_count"],
                "ready": False,
                "rule": "complete book-derived architecture slices before training, public calibration, or score-chasing",
            },
        )
        report["hard_gaps"].append(readiness_gap)
        report["summary"]["hard_gap_count"] = len(report["hard_gaps"])
        report["trigger_state"] = "RED"
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    report_evidence_store.write_json_report(crosswalk_path, crosswalk)
    view = gate_view(report) if args.gate else report
    print(json.dumps(view, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(
    matrix_path: Path,
    registry_path: Path,
    matrix: dict[str, Any],
    registry: dict[str, Any],
    started: float,
    ai_book_root: Path = DEFAULT_AI_BOOK_ROOT,
) -> dict[str, Any]:
    surfaces = {str(row.get("id") or ""): row for row in list_dicts(registry.get("surfaces"))}
    abstractions = {str(row.get("id") or ""): row for row in list_dicts(registry.get("abstractions"))}
    implementations = {str(row.get("id") or ""): row for row in list_dicts(registry.get("implementations"))}
    phases = list_dicts(matrix.get("phases"))
    phase_reports = [audit_phase(row, surfaces, abstractions, implementations) for row in phases]
    artifact_citation_report = audit_hive_artifact_citations()
    book_contract_report = audit_book_implementation_contract(matrix, ai_book_root)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    hard_gaps.extend(audit_matrix_shape(matrix, phases))
    hard_gaps.extend(audit_non_cheat_scope(matrix))
    hard_gaps.extend(artifact_citation_report["hard_gaps"])
    hard_gaps.extend(book_contract_report["hard_gaps"])
    warnings.extend(book_contract_report["warnings"])
    for row in phase_reports:
        hard_gaps.extend(row["hard_gaps"])
        warnings.extend(row["warnings"])
    pre_training_readiness = audit_pre_training_architecture_readiness(
        matrix=matrix,
        phase_reports=phase_reports,
        book_contract_report=book_contract_report,
        current_hard_gap_count=len(hard_gaps),
    )

    implemented = sum(1 for row in phase_reports if row["status"] in DONE_STATES and not row["hard_gaps"])
    partial = sum(1 for row in phase_reports if row["status"] == "partial")
    missing = sum(1 for row in phase_reports if row["status"] == "missing")
    frozen = sum(1 for row in phase_reports if row["status"] == "frozen")
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings or implemented < len(REQUIRED_PHASES):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_roadmap_implementation_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "matrix": rel(matrix_path),
            "registry": rel(registry_path),
            "phase_count": len(phases),
            "required_phase_count": len(REQUIRED_PHASES),
            "implemented_or_wired_count": implemented,
            "partial_count": partial,
            "missing_count": missing,
            "frozen_count": frozen,
            "hard_gap_count": len(hard_gaps),
            "warning_count": len(warnings),
            "phase_13_to_19_preserved": all(any(int_or(row.get("phase"), -1) == idx for row in phases) for idx in range(13, 20)),
            "hive_artifact_citation_report_count": artifact_citation_report["report_count"],
            "hive_artifact_citation_ready_count": artifact_citation_report["ready_count"],
            "book_implementation_track_count": book_contract_report["summary"]["track_count"],
            "book_chapter_implementation_crosswalk_count": book_contract_report["summary"]["chapter_crosswalk_count"],
            "book_manifest_chapter_count": book_contract_report["summary"]["book_manifest_chapter_count"],
            "book_manifest_order_match": book_contract_report["summary"]["book_manifest_order_match"],
            "book_manifest_digest_match": book_contract_report["summary"]["book_manifest_digest_match"],
            "book_manifest_sha256": book_contract_report["summary"]["book_manifest_sha256"],
            "book_manifest_source": book_contract_report["summary"]["book_manifest_source"],
            "book_manifest_commit": book_contract_report["summary"]["book_manifest_commit"],
            "live_book_manifest_sha256": book_contract_report["summary"]["live_book_manifest_sha256"],
            "live_book_manifest_differs_from_pin": book_contract_report["summary"][
                "live_book_manifest_differs_from_pin"
            ],
            "book_manifest_source_field_drift_chapter_count": book_contract_report["summary"][
                "book_manifest_source_field_drift_chapter_count"
            ],
            "book_manifest_source_field_drift_count": book_contract_report["summary"][
                "book_manifest_source_field_drift_count"
            ],
            "book_codex_test_count": book_contract_report["summary"]["book_codex_test_count"],
            "book_pending_or_partial_codex_test_count": book_contract_report["summary"][
                "book_pending_or_partial_codex_test_count"
            ],
            "book_chapter_crosswalk_missing_required_field_count": book_contract_report["summary"]["missing_required_field_count"],
            "book_chapter_invalid_phase_ref_count": book_contract_report["summary"]["invalid_phase_ref_count"],
            "book_active_flagship_lane_id": book_contract_report["summary"]["active_flagship_lane_id"],
            "book_active_core_slice_count": book_contract_report["summary"]["active_core_slice_count"],
            "book_active_core_slice_support_states": book_contract_report["summary"]["active_core_slice_support_states"],
            "book_core_slice_support_states": book_contract_report["summary"]["core_slice_support_states"],
            "book_support_state_ladder_ready": book_contract_report["summary"]["support_state_ladder_ready"],
            "book_future_candidate_count": book_contract_report["summary"]["future_candidate_count"],
            "book_future_candidate_chapter_count": book_contract_report["summary"]["future_candidate_chapter_count"],
            "book_future_cross_cutting_section_count": book_contract_report["summary"]["future_cross_cutting_section_count"],
            "book_future_candidate_missing_required_field_count": book_contract_report["summary"][
                "future_candidate_missing_required_field_count"
            ],
            "book_future_candidate_invalid_ref_count": book_contract_report["summary"]["future_candidate_invalid_ref_count"],
            "book_future_candidate_disposition_counts": book_contract_report["summary"]["future_candidate_disposition_counts"],
            "planned_codex_test_backlog_count": book_contract_report["summary"]["planned_codex_test_backlog_count"],
            "planned_codex_test_backlog_missing_required_field_count": book_contract_report["summary"][
                "planned_codex_test_backlog_missing_required_field_count"
            ],
            "planned_codex_test_backlog_invalid_ref_count": book_contract_report["summary"][
                "planned_codex_test_backlog_invalid_ref_count"
            ],
            "planned_codex_test_backlog_status_counts": book_contract_report["summary"][
                "planned_codex_test_backlog_status_counts"
            ],
            "planned_codex_test_backlog_technique_family_counts": book_contract_report["summary"][
                "planned_codex_test_backlog_technique_family_counts"
            ],
            "planned_codex_test_backlog_blocked_or_queued_count": book_contract_report["summary"][
                "planned_codex_test_backlog_blocked_or_queued_count"
            ],
            "pre_training_architecture_ready": pre_training_readiness["ready"],
            "pre_training_architecture_blocker_count": pre_training_readiness["blocker_count"],
            "pre_training_architecture_warning_count": pre_training_readiness["warning_count"],
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "rules": {
            "registry_binding": "Every phase must bind to an existing registry surface and abstraction.",
            "completion": "A phase may be implemented/wired only when it has evidence, gates, docs, and execution spine hooks.",
            "source_of_truth": "The matrix is the machine-readable roadmap state; roadmap.md remains the human narrative.",
            "no_cheat": "Benchmarks stay calibration-only; routers/templates/tools/fallbacks do not count as learned generation.",
            "scope": "This gate does not authorize long training, public benchmark spends, or new strict-generator target-mode experiments.",
        },
        "phase_reports": phase_reports,
        "book_implementation_contract": book_contract_report,
        "pre_training_architecture_readiness": pre_training_readiness,
        "hive_artifact_citations": artifact_citation_report["reports"],
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_hive_artifact_citations() -> dict[str, Any]:
    rows = []
    hard_gaps: list[dict[str, Any]] = []
    for rel_path in HIVE_ARTIFACT_CITATION_REPORTS:
        path = resolve(rel_path)
        payload = read_json(path)
        citation = payload.get("viea_artifact_citation") if isinstance(payload.get("viea_artifact_citation"), dict) else {}
        row = {
            "path": rel_path,
            "exists": path.exists(),
            "policy": payload.get("policy"),
            "citation_ready": citation.get("ready") is True,
            "citation_id": citation.get("citation_id"),
            "claim_ledger_ref_count": len(citation.get("claim_ledger_refs") if isinstance(citation.get("claim_ledger_refs"), list) else []),
            "artifact_record_ref_count": len(citation.get("artifact_record_refs") if isinstance(citation.get("artifact_record_refs"), list) else []),
            "evidence_transition_ref_count": len(citation.get("evidence_transition_refs") if isinstance(citation.get("evidence_transition_refs"), list) else []),
            "public_training_rows_written": int_or(payload.get("public_training_rows_written"), 0),
            "external_inference_calls": int_or(payload.get("external_inference_calls"), 0),
            "fallback_return_count": int_or(payload.get("fallback_return_count"), 0),
        }
        rows.append(row)
        if not path.exists():
            hard_gaps.append(gap("hive_artifact_citations", "required_artifact_report_missing", {"path": rel_path}))
            continue
        if not citation:
            hard_gaps.append(gap("hive_artifact_citations", "viea_artifact_citation_missing", {"path": rel_path, "policy": payload.get("policy")}))
            continue
        if citation.get("ready") is not True:
            hard_gaps.append(gap("hive_artifact_citations", "viea_artifact_citation_not_ready", {"path": rel_path, "missing_required_groups": citation.get("missing_required_groups")}))
        if row["claim_ledger_ref_count"] <= 0:
            hard_gaps.append(gap("hive_artifact_citations", "claim_ledger_refs_missing", {"path": rel_path, "citation_id": citation.get("citation_id")}))
        if any(row[key] != 0 for key in ["public_training_rows_written", "external_inference_calls", "fallback_return_count"]):
            hard_gaps.append(gap("hive_artifact_citations", "no_cheat_counter_fault", {"path": rel_path, "counters": {key: row[key] for key in ["public_training_rows_written", "external_inference_calls", "fallback_return_count"]}}))
        for list_key in ["artifacts", "fetched", "promoted"]:
            values = payload.get(list_key)
            if not isinstance(values, list) or not values:
                continue
            missing_rows = [idx for idx, item in enumerate(values[:100]) if isinstance(item, dict) and not item.get("viea_artifact_citation_id")]
            if missing_rows:
                hard_gaps.append(gap("hive_artifact_citations", "artifact_rows_missing_citation_id", {"path": rel_path, "list": list_key, "first_missing_indexes": missing_rows[:20]}))
    return {
        "report_count": len(rows),
        "ready_count": sum(1 for row in rows if row["citation_ready"]),
        "reports": rows,
        "hard_gaps": hard_gaps,
    }


def audit_matrix_shape(matrix: dict[str, Any], phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if str(matrix.get("policy") or "") != "project_theseus_roadmap_implementation_matrix_v1":
        gaps.append(gap("matrix", "wrong_or_missing_policy", {"policy": matrix.get("policy")}))
    seen = {int_or(row.get("phase"), -999) for row in phases}
    missing = sorted(REQUIRED_PHASES - seen)
    extra = sorted(idx for idx in seen if idx not in REQUIRED_PHASES)
    if missing:
        gaps.append(gap("matrix", "missing_required_phases", {"missing": missing}))
    if extra:
        gaps.append(gap("matrix", "unexpected_phase_ids", {"extra": extra}))
    preserved = {idx for idx in range(13, 20) if idx in seen}
    if len(preserved) != 7:
        gaps.append(gap("matrix", "phase_13_to_19_not_preserved", {"present": sorted(preserved)}))
    return gaps


def audit_non_cheat_scope(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    scope = json.dumps(matrix.get("out_of_scope_now") or [], sort_keys=True).lower()
    for term in DISALLOWED_OUT_OF_SCOPE_TERMS:
        if term.lower() not in scope:
            gaps.append(gap("matrix", "missing_explicit_out_of_scope_guard", {"term": term}))
    return gaps


def audit_phase(
    phase: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
    abstractions: dict[str, dict[str, Any]],
    implementations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    phase_id = int_or(phase.get("phase"), -1)
    status = str(phase.get("status") or "")
    phase_key = f"phase_{phase_id}"
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    required_non_empty = [
        "title",
        "priority",
        "roadmap_section",
        "registry_surface_id",
        "abstraction_id",
        "execution_spine_hooks",
        "required_records",
        "required_gates",
        "required_docs",
        "smallest_next_patch",
        "completion_rule",
    ]
    for field in required_non_empty:
        if empty(phase.get(field)):
            hard_gaps.append(gap(phase_key, "missing_required_field", {"field": field}))
    if status not in ALLOWED_PHASE_STATES:
        hard_gaps.append(gap(phase_key, "invalid_status", {"status": status, "allowed": sorted(ALLOWED_PHASE_STATES)}))
    surface_id = str(phase.get("registry_surface_id") or "")
    abstraction_id = str(phase.get("abstraction_id") or "")
    if surface_id not in surfaces:
        hard_gaps.append(gap(phase_key, "registry_surface_missing", {"surface_id": surface_id}))
    if abstraction_id not in abstractions:
        hard_gaps.append(gap(phase_key, "abstraction_missing", {"abstraction_id": abstraction_id}))
    if surface_id in surfaces and abstraction_id and str(surfaces[surface_id].get("abstraction_id") or "") != abstraction_id:
        warnings.append(
            gap(
                phase_key,
                "phase_abstraction_differs_from_surface_binding",
                {"surface_id": surface_id, "surface_abstraction_id": surfaces[surface_id].get("abstraction_id"), "phase_abstraction_id": abstraction_id},
                severity="warning",
            )
        )
    impl_ids = [str(x) for x in list_values(phase.get("implementation_ids"))]
    missing_impls = [impl_id for impl_id in impl_ids if impl_id not in implementations]
    if missing_impls:
        hard_gaps.append(gap(phase_key, "implementation_binding_missing", {"implementation_ids": missing_impls}))
    if status in DONE_STATES:
        if empty(phase.get("current_evidence")):
            hard_gaps.append(gap(phase_key, "done_state_without_current_evidence", {}))
        if empty(phase.get("integration_smoke")):
            hard_gaps.append(gap(phase_key, "done_state_without_integration_smoke", {}))
        if not empty(phase.get("missing_items")):
            hard_gaps.append(gap(phase_key, "done_state_still_has_missing_items", {"missing_items": phase.get("missing_items")}))
    else:
        if empty(phase.get("missing_items")):
            hard_gaps.append(gap(phase_key, "unfinished_phase_without_missing_items", {}))
        if empty(phase.get("smallest_next_patch")):
            hard_gaps.append(gap(phase_key, "unfinished_phase_without_next_patch", {}))
    if int_or(phase.get("phase"), -1) >= 13 and status in {"retired", "removed", "deleted"}:
        hard_gaps.append(gap(phase_key, "ai_book_phase_removed", {"status": status}))
    return {
        "phase": phase_id,
        "title": str(phase.get("title") or ""),
        "status": status,
        "priority": str(phase.get("priority") or ""),
        "registry_surface_id": surface_id,
        "abstraction_id": abstraction_id,
        "implementation_ids": impl_ids,
        "execution_spine_hooks": list_values(phase.get("execution_spine_hooks")),
        "required_records": list_values(phase.get("required_records")),
        "required_gates": list_values(phase.get("required_gates")),
        "current_evidence_count": len(list_values(phase.get("current_evidence"))),
        "missing_item_count": len(list_values(phase.get("missing_items"))),
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_book_implementation_contract(
    matrix: dict[str, Any], ai_book_root: Path = DEFAULT_AI_BOOK_ROOT
) -> dict[str, Any]:
    tracks = list_dicts(matrix.get("book_implementation_tracks"))
    track_ids = {str(row.get("track_id") or "") for row in tracks}
    support_ladder = list_dicts(matrix.get("claim_support_ladder"))
    support_states = {str(row.get("state") or "") for row in support_ladder}
    crosswalk = list_dicts(matrix.get("book_chapter_implementation_crosswalk"))
    future_candidates = list_dicts(matrix.get("book_future_candidate_crosswalk"))
    planned_backlog = list_dicts(matrix.get("planned_codex_test_backlog"))
    flagship = dict_value(matrix.get("flagship_lane_governance"))
    core = dict_value(matrix.get("book_reference_core_before_training"))
    core_slices = list_dicts(core.get("required_slices"))
    phases = {int_or(row.get("phase"), -1) for row in list_dicts(matrix.get("phases"))}
    reconciliation = dict_value(matrix.get("latest_ai_book_reconciliation"))
    pinned_book_commit = str(reconciliation.get("book_commit") or "").strip()
    pinned_manifest_payload, pinned_manifest_bytes, pinned_manifest_error = load_ai_book_manifest_at_commit(
        ai_book_root,
        pinned_book_commit,
    )
    book_manifest_chapters = load_ai_book_manifest_chapters_from_payload(pinned_manifest_payload)
    book_manifest_count = len(book_manifest_chapters)
    hard_gaps: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if pinned_manifest_error:
        hard_gaps.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "pinned_book_manifest_unavailable",
                {
                    "book_commit": pinned_book_commit,
                    "error": pinned_manifest_error,
                },
            )
        )

    if not tracks:
        hard_gaps.append(gap("book_implementation_contract", "missing_book_implementation_tracks", {}))
    if not support_ladder:
        hard_gaps.append(gap("book_implementation_contract", "missing_claim_support_ladder", {}))
    if not planned_backlog:
        hard_gaps.append(gap("book_implementation_contract", "missing_planned_codex_test_backlog", {}))
    if not future_candidates:
        hard_gaps.append(gap("book_implementation_contract", "missing_book_future_candidate_crosswalk", {}))
    missing_support_states = sorted(REQUIRED_SUPPORT_STATES - support_states)
    if missing_support_states:
        hard_gaps.append(gap("book_implementation_contract", "missing_required_support_states", {"missing": missing_support_states}))

    active_flagship_lane_id = str(flagship.get("active_flagship_lane_id") or "")
    active_track_id = str(flagship.get("active_track_id") or "")
    if not active_flagship_lane_id:
        hard_gaps.append(gap("book_implementation_contract", "active_flagship_lane_missing", {}))
    if active_track_id and active_track_id not in track_ids:
        hard_gaps.append(gap("book_implementation_contract", "active_flagship_track_missing", {"active_track_id": active_track_id}))

    active_slices = [row for row in core_slices if row.get("active") is True]
    if len(active_slices) != 1:
        hard_gaps.append(
            gap(
                "book_implementation_contract",
                "wrong_active_core_slice_count",
                {"active_count": len(active_slices), "active_slice_ids": [row.get("slice_id") for row in active_slices]},
            )
        )
    for row in core_slices:
        slice_id = str(row.get("slice_id") or "")
        track_id = str(row.get("track_id") or "")
        target = str(row.get("target_support_state") or "")
        phase_refs = [int_or(value, -1) for value in list_values(row.get("phases"))]
        if not slice_id:
            hard_gaps.append(gap("book_reference_core", "core_slice_missing_id", {"row": row}))
        if track_id not in track_ids:
            hard_gaps.append(gap("book_reference_core", "core_slice_track_missing", {"slice_id": slice_id, "track_id": track_id}))
        if target not in support_states:
            hard_gaps.append(gap("book_reference_core", "core_slice_support_target_missing", {"slice_id": slice_id, "target": target}))
        invalid = [phase for phase in phase_refs if phase not in phases]
        if invalid:
            hard_gaps.append(gap("book_reference_core", "core_slice_invalid_phase_refs", {"slice_id": slice_id, "invalid_phase_refs": invalid}))
        if row.get("active") is True:
            current_support_state = str(row.get("current_support_state") or "")
            current_evidence_refs = [str(item) for item in list_values(row.get("current_evidence_refs"))]
            if current_support_state not in support_states:
                hard_gaps.append(
                    gap(
                        "book_reference_core",
                        "active_core_slice_missing_current_support_state",
                        {"slice_id": slice_id, "current_support_state": current_support_state},
                    )
                )
            elif current_support_state in {"argument", "unsupported"}:
                hard_gaps.append(
                    gap(
                        "book_reference_core",
                        "active_core_slice_below_prototype_backed",
                        {"slice_id": slice_id, "current_support_state": current_support_state},
                    )
                )
            if not current_evidence_refs:
                hard_gaps.append(gap("book_reference_core", "active_core_slice_missing_evidence_refs", {"slice_id": slice_id}))

    if book_manifest_count and len(crosswalk) != book_manifest_count:
        hard_gaps.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "book_manifest_chapter_count_mismatch",
                {"crosswalk_count": len(crosswalk), "book_manifest_chapter_count": book_manifest_count},
            )
        )

    expected_chapter_ids = [str(row.get("chapter_id") or "") for row in book_manifest_chapters]
    actual_chapter_ids = [str(row.get("chapter_id") or "") for row in crosswalk]
    manifest_order_match = actual_chapter_ids == expected_chapter_ids
    if book_manifest_count and not manifest_order_match:
        hard_gaps.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "book_manifest_chapter_id_order_mismatch",
                {
                    "expected": expected_chapter_ids,
                    "actual": actual_chapter_ids,
                },
            )
        )

    crosswalk_by_id = {
        str(row.get("chapter_id") or ""): row
        for row in crosswalk
        if str(row.get("chapter_id") or "")
    }
    source_field_drifts: list[dict[str, Any]] = []
    for expected in book_manifest_chapters:
        chapter_id = str(expected.get("chapter_id") or "")
        actual = crosswalk_by_id.get(chapter_id)
        if actual is None:
            continue
        drift_fields = [
            matrix_field
            for matrix_field, expected_field in BOOK_CROSSWALK_SOURCE_FIELDS.items()
            if actual.get(matrix_field) != expected.get(expected_field)
        ]
        if drift_fields:
            source_field_drifts.append(
                {
                    "chapter_id": chapter_id,
                    "drift_fields": drift_fields,
                }
            )
    if source_field_drifts:
        hard_gaps.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "book_manifest_source_field_drift",
                {
                    "chapter_count": len(source_field_drifts),
                    "field_count": sum(len(row["drift_fields"]) for row in source_field_drifts),
                    "chapters": source_field_drifts,
                },
            )
        )

    manifest_path = ai_book_root / "book_structure.json"
    live_manifest_sha256 = hash_file(manifest_path) if manifest_path.exists() else ""
    actual_manifest_sha256 = (
        hashlib.sha256(pinned_manifest_bytes).hexdigest() if pinned_manifest_bytes else ""
    )
    expected_manifest_sha256 = str(reconciliation.get("manifest_sha256") or "")
    manifest_digest_match = bool(expected_manifest_sha256) and expected_manifest_sha256 == actual_manifest_sha256
    if book_manifest_count and not manifest_digest_match:
        hard_gaps.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "book_manifest_digest_mismatch",
                {
                    "expected_sha256": expected_manifest_sha256,
                    "actual_sha256": actual_manifest_sha256,
                },
            )
        )
    elif not book_manifest_count:
        warnings.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "book_manifest_unavailable",
                {"ai_book_root": str(DEFAULT_AI_BOOK_ROOT)},
                severity="warning",
            )
        )
    live_manifest_differs_from_pin = bool(
        live_manifest_sha256
        and actual_manifest_sha256
        and live_manifest_sha256 != actual_manifest_sha256
    )
    if live_manifest_differs_from_pin:
        warnings.append(
            gap(
                "book_chapter_implementation_crosswalk",
                "live_book_worktree_differs_from_pinned_snapshot",
                {
                    "book_commit": pinned_book_commit,
                    "pinned_manifest_sha256": actual_manifest_sha256,
                    "live_manifest_sha256": live_manifest_sha256,
                    "rule": "Live book edits are intake work, not an architecture-readiness regression. Reconcile and advance the pin in a separate reviewed change.",
                },
                severity="warning",
            )
        )

    chapter_ids: list[str] = []
    missing_required_field_count = 0
    invalid_phase_ref_count = 0
    invalid_track_count = 0
    invalid_support_target_count = 0
    missing_no_claim_count = 0
    planned_backlog_missing_required_field_count = 0
    planned_backlog_invalid_ref_count = 0
    for row in crosswalk:
        chapter_id = str(row.get("chapter_id") or "")
        chapter_ids.append(chapter_id)
        missing_fields = sorted(field for field in REQUIRED_BOOK_CROSSWALK_FIELDS if empty(row.get(field)))
        if missing_fields:
            missing_required_field_count += len(missing_fields)
            hard_gaps.append(gap("book_chapter_implementation_crosswalk", "chapter_row_missing_required_fields", {"chapter_id": chapter_id, "missing_fields": missing_fields}))
        track_id = str(row.get("primary_track_id") or "")
        if track_id not in track_ids:
            invalid_track_count += 1
            hard_gaps.append(gap("book_chapter_implementation_crosswalk", "chapter_row_invalid_track", {"chapter_id": chapter_id, "primary_track_id": track_id}))
        target = str(row.get("support_state_target") or "")
        if target not in support_states:
            invalid_support_target_count += 1
            hard_gaps.append(gap("book_chapter_implementation_crosswalk", "chapter_row_invalid_support_target", {"chapter_id": chapter_id, "support_state_target": target}))
        phase_refs = [int_or(value, -1) for value in list_values(row.get("phase_refs"))]
        invalid_phases = [phase for phase in phase_refs if phase not in phases]
        if invalid_phases:
            invalid_phase_ref_count += len(invalid_phases)
            hard_gaps.append(gap("book_chapter_implementation_crosswalk", "chapter_row_invalid_phase_refs", {"chapter_id": chapter_id, "invalid_phase_refs": invalid_phases}))
        no_claims = [str(value) for value in list_values(row.get("no_claims"))]
        if len(no_claims) < 2:
            missing_no_claim_count += 1
            hard_gaps.append(gap("book_chapter_implementation_crosswalk", "chapter_row_missing_no_claim_boundaries", {"chapter_id": chapter_id, "no_claims": no_claims}))
        if int_or(row.get("codex_test_count"), 0) <= 0:
            warnings.append(
                gap(
                    "book_chapter_implementation_crosswalk",
                    "chapter_row_has_no_representative_codex_tests",
                    {"chapter_id": chapter_id},
                    severity="warning",
                )
            )

    duplicates = sorted(item for item, count in count_values(chapter_ids).items() if item and count > 1)
    if duplicates:
        hard_gaps.append(gap("book_chapter_implementation_crosswalk", "duplicate_chapter_ids", {"duplicates": duplicates}))

    future_candidate_ids: list[str] = []
    future_candidate_kinds: list[str] = []
    future_candidate_dispositions: list[str] = []
    future_candidate_missing_required_field_count = 0
    future_candidate_invalid_ref_count = 0
    for row in future_candidates:
        candidate_id = str(row.get("candidate_id") or "")
        item_kind = str(row.get("item_kind") or "")
        disposition = str(row.get("book_disposition") or "")
        future_candidate_ids.append(candidate_id)
        future_candidate_kinds.append(item_kind or "missing_item_kind")
        future_candidate_dispositions.append(disposition or "missing_disposition")
        missing_fields = sorted(field for field in REQUIRED_BOOK_FUTURE_CANDIDATE_FIELDS if empty(row.get(field)))
        if missing_fields:
            future_candidate_missing_required_field_count += len(missing_fields)
            hard_gaps.append(
                gap(
                    "book_future_candidate_crosswalk",
                    "future_candidate_missing_required_fields",
                    {"candidate_id": candidate_id, "missing_fields": missing_fields},
                )
            )
        if item_kind not in ALLOWED_BOOK_FUTURE_ITEM_KINDS:
            future_candidate_invalid_ref_count += 1
            hard_gaps.append(
                gap(
                    "book_future_candidate_crosswalk",
                    "future_candidate_invalid_item_kind",
                    {"candidate_id": candidate_id, "item_kind": item_kind},
                )
            )
        phase_refs = [int_or(value, -1) for value in list_values(row.get("phase_refs"))]
        invalid_phases = [phase for phase in phase_refs if phase not in phases]
        if invalid_phases:
            future_candidate_invalid_ref_count += len(invalid_phases)
            hard_gaps.append(
                gap(
                    "book_future_candidate_crosswalk",
                    "future_candidate_invalid_phase_refs",
                    {"candidate_id": candidate_id, "invalid_phase_refs": invalid_phases},
                )
            )
        if disposition == "admitted_current_local_manifest":
            admitted_chapter_id = str(row.get("chapter_id") or "")
            if not admitted_chapter_id or admitted_chapter_id not in chapter_ids:
                future_candidate_invalid_ref_count += 1
                hard_gaps.append(
                    gap(
                        "book_future_candidate_crosswalk",
                        "admitted_future_candidate_missing_chapter_crosswalk",
                        {"candidate_id": candidate_id, "chapter_id": admitted_chapter_id},
                    )
                )

    future_duplicates = sorted(item for item, count in count_values(future_candidate_ids).items() if item and count > 1)
    if future_duplicates:
        hard_gaps.append(gap("book_future_candidate_crosswalk", "duplicate_future_candidate_ids", {"duplicates": future_duplicates}))
    future_cross_cutting_obligations = [str(value) for value in list_values(matrix.get("book_future_cross_cutting_obligations"))]
    if not future_cross_cutting_obligations:
        hard_gaps.append(gap("book_future_candidate_crosswalk", "missing_cross_cutting_obligations", {}))

    planned_backlog_ids: list[str] = []
    planned_backlog_statuses: list[str] = []
    planned_backlog_technique_families: list[str] = []
    for row in planned_backlog:
        backlog_id = str(row.get("backlog_id") or "")
        status = str(row.get("status") or "")
        technique_family = str(row.get("technique_family") or "")
        planned_backlog_ids.append(backlog_id)
        planned_backlog_statuses.append(status or "missing_status")
        planned_backlog_technique_families.append(technique_family or "missing_technique_family")
        missing_fields = sorted(field for field in REQUIRED_PLANNED_CODEX_BACKLOG_FIELDS if empty(row.get(field)))
        if missing_fields:
            planned_backlog_missing_required_field_count += len(missing_fields)
            hard_gaps.append(
                gap(
                    "planned_codex_test_backlog",
                    "backlog_row_missing_required_fields",
                    {"backlog_id": backlog_id, "missing_fields": missing_fields},
                )
            )
        track_id = str(row.get("track_id") or "")
        if track_id not in track_ids:
            planned_backlog_invalid_ref_count += 1
            hard_gaps.append(
                gap(
                    "planned_codex_test_backlog",
                    "backlog_row_invalid_track",
                    {"backlog_id": backlog_id, "track_id": track_id},
                )
            )
        target = str(row.get("support_state_target") or "")
        if target not in support_states:
            planned_backlog_invalid_ref_count += 1
            hard_gaps.append(
                gap(
                    "planned_codex_test_backlog",
                    "backlog_row_invalid_support_target",
                    {"backlog_id": backlog_id, "support_state_target": target},
                )
            )
        phase_refs = [int_or(value, -1) for value in list_values(row.get("owned_phase_refs"))]
        invalid_phases = [phase for phase in phase_refs if phase not in phases]
        if invalid_phases:
            planned_backlog_invalid_ref_count += len(invalid_phases)
            hard_gaps.append(
                gap(
                    "planned_codex_test_backlog",
                    "backlog_row_invalid_phase_refs",
                    {"backlog_id": backlog_id, "invalid_phase_refs": invalid_phases},
                )
            )
        if not str(row.get("no_claim_boundary") or "").strip():
            planned_backlog_missing_required_field_count += 1
            hard_gaps.append(
                gap(
                    "planned_codex_test_backlog",
                    "backlog_row_missing_no_claim_boundary",
                    {"backlog_id": backlog_id},
                )
            )

    planned_duplicates = sorted(item for item, count in count_values(planned_backlog_ids).items() if item and count > 1)
    if planned_duplicates:
        hard_gaps.append(gap("planned_codex_test_backlog", "duplicate_backlog_ids", {"duplicates": planned_duplicates}))

    return {
        "policy": "project_theseus_book_implementation_contract_audit_v1",
        "summary": {
            "track_count": len(tracks),
            "chapter_crosswalk_count": len(crosswalk),
            "book_manifest_chapter_count": book_manifest_count,
            "book_manifest_order_match": manifest_order_match,
            "book_manifest_digest_match": manifest_digest_match,
            "book_manifest_sha256": actual_manifest_sha256,
            "book_manifest_source": "pinned_git_commit",
            "book_manifest_commit": pinned_book_commit,
            "live_book_manifest_sha256": live_manifest_sha256,
            "live_book_manifest_differs_from_pin": live_manifest_differs_from_pin,
            "book_manifest_source_field_drift_chapter_count": len(source_field_drifts),
            "book_manifest_source_field_drift_count": sum(
                len(row["drift_fields"]) for row in source_field_drifts
            ),
            "book_codex_test_count": sum(
                int_or(row.get("codex_test_count"), 0) for row in book_manifest_chapters
            ),
            "book_pending_or_partial_codex_test_count": sum(
                int_or(row.get("pending_or_partial_codex_test_count"), 0)
                for row in book_manifest_chapters
            ),
            "active_flagship_lane_id": active_flagship_lane_id,
            "active_track_id": active_track_id,
            "active_core_slice_count": len(active_slices),
            "support_state_ladder_ready": not missing_support_states and bool(support_ladder),
            "support_state_count": len(support_states),
            "active_core_slice_support_states": {
                str(row.get("slice_id") or ""): str(row.get("current_support_state") or "")
                for row in active_slices
            },
            "core_slice_support_states": {
                str(row.get("slice_id") or ""): str(row.get("current_support_state") or "not_yet_supported")
                for row in core_slices
            },
            "missing_required_field_count": missing_required_field_count,
            "invalid_phase_ref_count": invalid_phase_ref_count,
            "invalid_track_count": invalid_track_count,
            "invalid_support_target_count": invalid_support_target_count,
            "missing_no_claim_count": missing_no_claim_count,
            "duplicate_chapter_id_count": len(duplicates),
            "future_candidate_count": len(future_candidates),
            "future_candidate_chapter_count": sum(1 for kind in future_candidate_kinds if kind == "chapter_candidate"),
            "future_cross_cutting_section_count": sum(1 for kind in future_candidate_kinds if kind == "cross_cutting_section"),
            "future_cross_cutting_obligation_count": len(future_cross_cutting_obligations),
            "future_candidate_missing_required_field_count": future_candidate_missing_required_field_count,
            "future_candidate_invalid_ref_count": future_candidate_invalid_ref_count,
            "future_candidate_duplicate_id_count": len(future_duplicates),
            "future_candidate_disposition_counts": count_values(future_candidate_dispositions),
            "planned_codex_test_backlog_count": len(planned_backlog),
            "planned_codex_test_backlog_missing_required_field_count": planned_backlog_missing_required_field_count,
            "planned_codex_test_backlog_invalid_ref_count": planned_backlog_invalid_ref_count,
            "planned_codex_test_backlog_duplicate_id_count": len(planned_duplicates),
            "planned_codex_test_backlog_status_counts": count_values(planned_backlog_statuses),
            "planned_codex_test_backlog_technique_family_counts": count_values(planned_backlog_technique_families),
            "planned_codex_test_backlog_blocked_or_queued_count": sum(
                1
                for status in planned_backlog_statuses
                if status.startswith("blocked") or status.startswith("queued")
            ),
        },
        "hard_gaps": hard_gaps,
        "warnings": warnings,
    }


def audit_pre_training_architecture_readiness(
    *,
    matrix: dict[str, Any],
    phase_reports: list[dict[str, Any]],
    book_contract_report: dict[str, Any],
    current_hard_gap_count: int,
) -> dict[str, Any]:
    """Decide whether architecture is ready to make training the main focus.

    This is intentionally stricter than the normal roadmap gate. The ordinary
    gate can be YELLOW with no hard gaps while local work continues; this view
    answers a different question: are the book-derived implementation surfaces
    complete enough that training, public calibration, or score chasing should
    become the main path again?
    """

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    phases = list_dicts(matrix.get("phases"))
    phase_by_id = {int_or(row.get("phase"), -1): row for row in phases}
    phase_report_by_id = {int_or(row.get("phase"), -1): row for row in phase_reports}
    support_rank = {
        str(row.get("state") or ""): idx
        for idx, row in enumerate(list_dicts(matrix.get("claim_support_ladder")))
    }
    architecture_contract = dict_value(matrix.get("pre_training_architecture_contract"))
    required_architecture_phase_ids = {
        int_or(value, -1)
        for value in list_values(architecture_contract.get("required_phase_ids"))
    }
    deferred_phase_ids = {
        int_or(value, -1)
        for value in list_values(architecture_contract.get("training_or_behavior_qualification_phase_ids"))
    }
    external_environment_phase_ids = {
        int_or(value, -1)
        for value in list_values(architecture_contract.get("external_environment_phase_ids"))
    }
    declared_phase_ids = set(phase_by_id)
    contract_phase_ids = required_architecture_phase_ids | deferred_phase_ids | external_environment_phase_ids
    overlap_phase_ids = (
        (required_architecture_phase_ids & deferred_phase_ids)
        | (required_architecture_phase_ids & external_environment_phase_ids)
        | (deferred_phase_ids & external_environment_phase_ids)
    )
    if not required_architecture_phase_ids:
        blockers.append(
            {
                "kind": "pre_training_architecture_contract_missing_required_phases",
                "required_action": "Declare architecture prerequisites separately from training and behavior qualification phases.",
            }
        )
    if contract_phase_ids != declared_phase_ids or overlap_phase_ids:
        blockers.append(
            {
                "kind": "pre_training_architecture_contract_phase_partition_invalid",
                "missing_phase_ids": sorted(declared_phase_ids - contract_phase_ids),
                "unknown_phase_ids": sorted(contract_phase_ids - declared_phase_ids),
                "overlap_phase_ids": sorted(overlap_phase_ids),
                "required_action": "Partition every roadmap phase into exactly one readiness class.",
            }
        )

    if current_hard_gap_count:
        blockers.append(
            {
                "kind": "roadmap_hard_gaps_present",
                "hard_gap_count": current_hard_gap_count,
                "required_action": "Clear hard roadmap/registry/book-contract gaps before training focus.",
            }
        )

    externally_frozen = [
        {
            "phase": int_or(row.get("phase"), -1),
            "title": str(row.get("title") or ""),
            "status": str(row.get("status") or ""),
            "reason": " ".join(
                [
                    str(row.get("smallest_next_patch") or ""),
                    " ".join(str(item) for item in list_values(row.get("missing_items"))),
                ]
            )[:500],
        }
        for row in phases
        if phase_is_external_frozen(row)
    ]
    unfinished = [
        {
            "phase": int_or(row.get("phase"), -1),
            "title": str(row.get("title") or ""),
            "status": str(row.get("status") or ""),
            "missing_item_count": len(list_values(row.get("missing_items"))),
            "smallest_next_patch": str(row.get("smallest_next_patch") or ""),
        }
        for row in phases
        if int_or(row.get("phase"), -1) in required_architecture_phase_ids
        and str(row.get("status") or "") not in DONE_STATES
        and not phase_is_external_frozen(row)
    ]
    if unfinished:
        blockers.append(
            {
                "kind": "unfinished_architecture_prerequisite_phases",
                "count": len(unfinished),
                "phases": unfinished,
                "required_action": "Complete or explicitly external-freeze every architecture prerequisite before making training the primary roadmap focus.",
            }
        )

    deferred_unfinished = [
        {
            "phase": int_or(row.get("phase"), -1),
            "title": str(row.get("title") or ""),
            "status": str(row.get("status") or ""),
            "reason": "training_or_behavior_qualification_follow_through",
        }
        for row in phases
        if int_or(row.get("phase"), -1) in deferred_phase_ids
        and str(row.get("status") or "") not in DONE_STATES
    ]

    frozen_without_external_reason = []
    for row in phases:
        if str(row.get("status") or "") != "frozen":
            continue
        if not phase_is_external_frozen(row):
            frozen_without_external_reason.append({"phase": int_or(row.get("phase"), -1), "title": str(row.get("title") or "")})
    if frozen_without_external_reason:
        blockers.append(
            {
                "kind": "frozen_phase_without_external_environment_reason",
                "phases": frozen_without_external_reason,
                "required_action": "Frozen phases must name a real external environment blocker rather than hiding unfinished architecture work.",
            }
        )

    core = dict_value(matrix.get("book_reference_core_before_training"))
    core_slices = list_dicts(core.get("required_slices"))
    core_blockers = []
    for row in core_slices:
        current = str(row.get("current_support_state") or "")
        target = str(row.get("target_support_state") or "")
        current_rank = support_rank.get(current, -1)
        target_rank = support_rank.get(target, len(support_rank) + 1)
        current_evidence = [str(item) for item in list_values(row.get("current_evidence_refs"))]
        current_commands = [str(item) for item in list_values(row.get("current_validation_commands"))]
        if current_rank < target_rank or not current_evidence or not current_commands or str(row.get("current_state") or "") != "GREEN":
            core_blockers.append(
                {
                    "slice_id": str(row.get("slice_id") or ""),
                    "current_state": str(row.get("current_state") or ""),
                    "current_support_state": current,
                    "target_support_state": target,
                    "current_support_rank": current_rank,
                    "target_support_rank": target_rank,
                    "evidence_ref_count": len(current_evidence),
                    "validation_command_count": len(current_commands),
                    "active": row.get("active") is True,
                }
            )
    if core_blockers:
        blockers.append(
            {
                "kind": "book_reference_core_below_target_support",
                "count": len(core_blockers),
                "slices": core_blockers,
                "required_action": "Raise every pre-training book-reference core slice to its target support state with runnable validation commands and evidence refs.",
            }
        )

    out_of_scope = {str(item) for item in list_values(matrix.get("out_of_scope_now"))}
    required_guards = {
        "public_benchmark_training",
        "serve_external_inference",
        "count_router_as_learned_generation",
        "count_template_as_learned_generation",
        "long_training_as_implementation_proof",
        "training_score_chase_before_book_reference_core",
        "capability_claim_from_assisted_or_tool_output",
    }
    missing_guards = sorted(required_guards - out_of_scope)
    if missing_guards:
        blockers.append(
            {
                "kind": "missing_pre_training_no_cheat_guards",
                "missing_guards": missing_guards,
                "required_action": "Keep training/public-calibration/tool-output boundaries explicit in the roadmap matrix.",
            }
        )

    for report in phase_reports:
        phase_id = int_or(report.get("phase"), -1)
        phase = phase_by_id.get(phase_id, {})
        if str(report.get("status") or "") in DONE_STATES:
            gate_count = len(list_values(phase.get("required_gates")))
            evidence_count = len(list_values(phase.get("current_evidence")))
            smoke_count = len(list_values(phase.get("integration_smoke")))
            if gate_count <= 0 or evidence_count <= 0 or smoke_count <= 0:
                blockers.append(
                    {
                        "kind": "done_phase_missing_training_readiness_evidence",
                        "phase": phase_id,
                        "title": str(report.get("title") or ""),
                        "required_gate_count": gate_count,
                        "current_evidence_count": evidence_count,
                        "integration_smoke_count": smoke_count,
                    }
                )

    phase_status_counts: dict[str, int] = {}
    for report in phase_reports:
        status = str(report.get("status") or "")
        phase_status_counts[status] = phase_status_counts.get(status, 0) + 1

    return {
        "policy": "project_theseus_pre_training_architecture_readiness_v1",
        "ready": not blockers,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "phase_status_counts": phase_status_counts,
        "externally_frozen_phase_count": len(externally_frozen),
        "externally_frozen_phases": externally_frozen,
        "required_architecture_phase_ids": sorted(required_architecture_phase_ids),
        "training_or_behavior_qualification_phase_ids": sorted(deferred_phase_ids),
        "external_environment_phase_ids": sorted(external_environment_phase_ids),
        "deferred_unfinished_phases": deferred_unfinished,
        "core_slice_count": len(core_slices),
        "support_rank": support_rank,
        "rules": {
            "scope": "This gate decides whether architecture is ready for training/public calibration focus; it does not run training.",
            "training_boundary": "No long training, public calibration, or score chasing should be primary while required architecture remains unfinished; training and behavior qualification phases cannot circularly block architecture readiness.",
            "external_frozen_exception": "A frozen item can remain only when it names a concrete external-environment blocker such as unreachable trusted peers.",
            "claim_boundary": "Tools, routers, templates, deterministic solvers, and assisted product traces stay separate from learned-generation claims.",
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def phase_is_external_frozen(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") != "frozen":
        return False
    text = " ".join(
        [
            str(row.get("smallest_next_patch") or ""),
            " ".join(str(item) for item in list_values(row.get("missing_items"))),
        ]
    ).lower()
    return any(term in text for term in EXTERNAL_FREEZE_TERMS)


def count_ai_book_manifest_chapters(ai_book_root: Path) -> int:
    return len(load_ai_book_manifest_chapters(ai_book_root))


def load_ai_book_manifest_chapters(ai_book_root: Path) -> list[dict[str, Any]]:
    manifest = ai_book_root / "book_structure.json"
    if not manifest.exists():
        return []
    return load_ai_book_manifest_chapters_from_payload(read_json(manifest))


def load_ai_book_manifest_at_commit(
    ai_book_root: Path,
    book_commit: str,
) -> tuple[dict[str, Any], bytes, str]:
    if not book_commit:
        return {}, b"", "latest_ai_book_reconciliation.book_commit is empty"
    if len(book_commit) != 40 or any(char not in "0123456789abcdef" for char in book_commit.lower()):
        return {}, b"", "book_commit must be a full 40-character hexadecimal Git object id"
    try:
        completed = subprocess.run(
            ["git", "-C", str(ai_book_root), "show", f"{book_commit}:book_structure.json"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError:
        return {}, b"", "git executable is unavailable"
    except subprocess.TimeoutExpired:
        return {}, b"", "git show timed out after 10 seconds"
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        return {}, b"", error or f"git show failed with exit {completed.returncode}"
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, completed.stdout, f"pinned manifest is not valid UTF-8 JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, completed.stdout, "pinned manifest root is not an object"
    return payload, completed.stdout, ""


def load_ai_book_manifest_chapters_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in list_dicts(payload.get("parts")):
        for chapter in list_dicts(part.get("chapters")):
            codex_tests = list_values(chapter.get("codex_tests"))
            test_names = [book_codex_test_name(item) for item in codex_tests]
            rows.append(
                {
                    "chapter_ordinal": len(rows) + 1,
                    "chapter_id": str(chapter.get("id") or ""),
                    "chapter_title": str(chapter.get("title") or ""),
                    "part_id": str(part.get("id") or ""),
                    "part_title": str(part.get("title") or ""),
                    "book_file": str(chapter.get("file") or ""),
                    "book_claim_label": str(chapter.get("claim_label") or ""),
                    "book_evidence_level": str(chapter.get("evidence_level") or ""),
                    "book_minimal_implementation": str(chapter.get("minimal_implementation") or ""),
                    "book_beyond_state_of_art": str(chapter.get("beyond_state_of_art") or ""),
                    "book_interfaces": list_values(chapter.get("interfaces")),
                    "book_invariants": list_values(chapter.get("invariants")),
                    "book_failure_modes": list_values(chapter.get("failure_modes")),
                    "codex_test_count": len(codex_tests),
                    "pending_or_partial_codex_test_count": sum(
                        1 for item in codex_tests if not book_codex_test_is_implemented(item)
                    ),
                    "representative_codex_tests": test_names[:3],
                }
            )
    return rows


def book_codex_test_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value)


def book_codex_test_is_implemented(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    status = str(value.get("implementation_status") or value.get("status") or "").strip().lower()
    return status == "implemented" or status.startswith("implemented ") or status.startswith("implemented;")


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Roadmap Implementation Gate",
        "",
        f"- Trigger state: `{report['trigger_state']}`",
        f"- Matrix: `{summary['matrix']}`",
        f"- Phases: `{summary['phase_count']}/{summary['required_phase_count']}`",
        f"- Implemented/wired: `{summary['implemented_or_wired_count']}`",
        f"- Partial: `{summary['partial_count']}`",
        f"- Missing: `{summary['missing_count']}`",
        f"- Frozen: `{summary['frozen_count']}`",
        f"- Book crosswalk items: `{summary.get('book_to_theseus_crosswalk_item_count', 0)}`",
        f"- AI_book authored source files: `{summary.get('book_to_theseus_ai_book_source_file_count', 0)}`",
        f"- Stale book-linked phases: `{summary.get('book_to_theseus_stale_phase_count', 0)}`",
        f"- Unresolved book-to-roadmap backlog items: `{summary.get('book_to_theseus_roadmap_backlog_item_count', 0)}`",
        f"- Public-safe Theseus-to-book evidence pointers: `{summary.get('theseus_to_book_evidence_count', 0)}`",
        f"- Source-sync smoke passed: `{summary.get('book_to_theseus_source_sync_smoke_passed', False)}`",
        f"- Public-safe evidence smoke passed: `{summary.get('public_safe_evidence_smoke_passed', False)}`",
        f"- Book implementation tracks: `{summary.get('book_implementation_track_count', 0)}`",
        f"- Book chapter implementation crosswalk: `{summary.get('book_chapter_implementation_crosswalk_count', 0)}/{summary.get('book_manifest_chapter_count', 0)}`",
        f"- Book manifest order match: `{summary.get('book_manifest_order_match', False)}`",
        f"- Book manifest digest match: `{summary.get('book_manifest_digest_match', False)}`",
        f"- Book source-field drift: `{summary.get('book_manifest_source_field_drift_count', 0)}` fields across `{summary.get('book_manifest_source_field_drift_chapter_count', 0)}` chapters",
        f"- Book Codex tests: `{summary.get('book_codex_test_count', 0)}` total; `{summary.get('book_pending_or_partial_codex_test_count', 0)}` pending/partial",
        f"- Book future candidates/sections: `{summary.get('book_future_candidate_count', 0)}`",
        f"- Book future candidate missing fields: `{summary.get('book_future_candidate_missing_required_field_count', 0)}`",
        f"- Book future candidate invalid refs: `{summary.get('book_future_candidate_invalid_ref_count', 0)}`",
        f"- Book future candidate dispositions: `{summary.get('book_future_candidate_disposition_counts', {})}`",
        f"- Planned Codex test backlog: `{summary.get('planned_codex_test_backlog_count', 0)}`",
        f"- Planned Codex backlog missing fields: `{summary.get('planned_codex_test_backlog_missing_required_field_count', 0)}`",
        f"- Planned Codex backlog invalid refs: `{summary.get('planned_codex_test_backlog_invalid_ref_count', 0)}`",
        f"- Planned Codex backlog blocked/queued: `{summary.get('planned_codex_test_backlog_blocked_or_queued_count', 0)}`",
        f"- Planned Codex backlog status counts: `{summary.get('planned_codex_test_backlog_status_counts', {})}`",
        f"- Planned Codex backlog technique families: `{summary.get('planned_codex_test_backlog_technique_family_counts', {})}`",
        f"- Active flagship lane: `{summary.get('book_active_flagship_lane_id', '')}`",
        f"- Active core slices: `{summary.get('book_active_core_slice_count', 0)}`",
        f"- Active core slice support states: `{summary.get('book_active_core_slice_support_states', {})}`",
        f"- Core slice support states: `{summary.get('book_core_slice_support_states', {})}`",
        f"- Support-state ladder ready: `{summary.get('book_support_state_ladder_ready', False)}`",
        f"- Pre-training architecture ready: `{summary.get('pre_training_architecture_ready', False)}`",
        f"- Pre-training architecture blockers: `{summary.get('pre_training_architecture_blocker_count', 0)}`",
        f"- Book crosswalk missing fields: `{summary.get('book_chapter_crosswalk_missing_required_field_count', 0)}`",
        f"- Book crosswalk invalid phase refs: `{summary.get('book_chapter_invalid_phase_ref_count', 0)}`",
        f"- Hard gaps: `{summary['hard_gap_count']}`",
        f"- Warnings: `{summary['warning_count']}`",
        f"- Phases 13-19 preserved: `{summary['phase_13_to_19_preserved']}`",
        "",
        "## Phase Matrix",
        "",
        "| Phase | Status | Priority | Surface | Missing | Hard Gaps |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for phase in report["phase_reports"]:
        lines.append(
            f"| {phase['phase']} {phase['title']} | {phase['status']} | {phase['priority']} | "
            f"`{phase['registry_surface_id']}` | {phase['missing_item_count']} | {phase['hard_gap_count']} |"
        )
    if report["hard_gaps"]:
        lines.extend(["", "## Hard Gaps", ""])
        for item in report["hard_gaps"][:80]:
            lines.append(f"- `{item['id']}`: {item['kind']} {json.dumps(item.get('evidence', {}), sort_keys=True)}")
    readiness = report.get("pre_training_architecture_readiness")
    if isinstance(readiness, dict):
        lines.extend(["", "## Pre-Training Architecture Readiness", ""])
        lines.append(f"- Ready: `{readiness.get('ready', False)}`")
        lines.append(f"- Blockers: `{readiness.get('blocker_count', 0)}`")
        for item in list_dicts(readiness.get("blockers"))[:20]:
            lines.append(f"- {item.get('kind')}: {json.dumps(item, sort_keys=True)}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for item in report["warnings"][:80]:
            lines.append(f"- `{item['id']}`: {item['kind']} {json.dumps(item.get('evidence', {}), sort_keys=True)}")
    lines.append("")
    return "\n".join(lines)


def build_book_to_theseus_crosswalk(
    matrix_path: Path,
    registry_path: Path,
    matrix: dict[str, Any],
    report: dict[str, Any],
    crosswalk_path: Path,
    *,
    ai_book_root: Path,
    project_steward: dict[str, Any],
) -> dict[str, Any]:
    phase_reports = {int_or(row.get("phase"), -1): row for row in list_dicts(report.get("phase_reports"))}
    previous_crosswalk = read_json(crosswalk_path)
    source_inventory = build_ai_book_source_inventory(ai_book_root)
    previous_source_hashes = source_hashes_by_path(previous_crosswalk)
    current_source_hashes = {str(row.get("path") or ""): str(row.get("sha256") or "") for row in source_inventory["sources"]}
    previous_inventory_available = bool(previous_source_hashes)
    changed_source_paths = sorted(
        path
        for path, digest in current_source_hashes.items()
        if previous_inventory_available and previous_source_hashes.get(path) != digest
    )
    removed_source_paths = sorted(path for path in previous_source_hashes if path not in current_source_hashes) if previous_inventory_available else []
    changed_source_set = set(changed_source_paths) | set(removed_source_paths)
    source_sync_decisions = source_sync_review_decisions(project_steward)
    source_sync_smoke = source_sync_detection_smoke()
    items = []
    stale_phases = []
    for phase in list_dicts(matrix.get("phases")):
        phase_id = int_or(phase.get("phase"), -1)
        matched_sources = match_sources_for_phase(phase, source_inventory["sources"])
        matched_source_paths = [str(row.get("path") or "") for row in matched_sources]
        changed_matches = sorted(path for path in matched_source_paths if path in changed_source_set)
        matched_source_hash = stable_hash([row.get("path") for row in matched_sources] + [row.get("sha256") for row in matched_sources])
        review_decision = matching_source_sync_decision(
            source_sync_decisions,
            phase_id=phase_id,
            changed_source_paths=changed_matches,
            matched_source_hash=matched_source_hash,
        )
        stale = bool(changed_matches) and not review_decision
        if stale:
            stale_phases.append(phase_id)
        items.append(
            {
                "phase": phase_id,
                "title": str(phase.get("title") or ""),
                "ai_book_source_basis": [str(item) for item in list_values(phase.get("ai_book_source_basis"))],
                "source_sync": {
                    "ai_book_root": source_inventory["root"],
                    "matched_source_count": len(matched_sources),
                    "matched_source_paths": matched_source_paths[:20],
                    "matched_source_hash": matched_source_hash,
                    "changed_matched_source_count": len(changed_matches),
                    "changed_matched_source_paths": changed_matches[:20],
                    "stale_source_candidate": stale,
                    "review_decision": review_decision,
                    "support_state": (
                        "STALE_REVIEW_NEEDED"
                        if stale
                        else (
                            "REVIEWED_MATRIX_UPDATED"
                            if review_decision and review_decision.get("decision") == "matrix_updated"
                            else (
                                "REVIEWED_NO_CONTRACT_CHANGE"
                                if review_decision
                                else ("BASELINE_CREATED" if not previous_inventory_available else "SYNCED")
                            )
                        )
                    ),
                },
                "registry_surface_id": str(phase.get("registry_surface_id") or ""),
                "abstraction_id": str(phase.get("abstraction_id") or ""),
                "implementation_ids": [str(item) for item in list_values(phase.get("implementation_ids"))],
                "status": str(phase.get("status") or ""),
                "current_evidence": [str(item) for item in list_values(phase.get("current_evidence"))],
                "required_records": [str(item) for item in list_values(phase.get("required_records"))],
                "required_gates": [str(item) for item in list_values(phase.get("required_gates"))],
                "missing_items": [str(item) for item in list_values(phase.get("missing_items"))],
                "smallest_next_patch": str(phase.get("smallest_next_patch") or ""),
                "completion_rule": str(phase.get("completion_rule") or ""),
                "gate_summary": {
                    "hard_gap_count": int_or(phase_reports.get(phase_id, {}).get("hard_gap_count"), 0),
                    "warning_count": int_or(phase_reports.get(phase_id, {}).get("warning_count"), 0),
                    "missing_item_count": int_or(phase_reports.get(phase_id, {}).get("missing_item_count"), 0),
                },
            }
        )
    missing_source_basis = [row["phase"] for row in items if not row["ai_book_source_basis"]]
    missing_evidence = [row["phase"] for row in items if row["status"] in DONE_STATES and not row["current_evidence"]]
    detection_time = now()
    new_backlog_items = [
        {
            "record_type": "roadmap_backlog_item",
            "backlog_id": stable_id("book_to_theseus_backlog", row["phase"], row["source_sync"]["matched_source_hash"]),
            "phase": row["phase"],
            "title": row["title"],
            "state": "STALE_REVIEW_NEEDED",
            "reason": "matched_ai_book_source_changed_since_previous_crosswalk",
            "changed_source_paths": row["source_sync"]["changed_matched_source_paths"],
            "source_manifest_hash_at_detection": source_inventory["manifest_hash"],
            "first_detected_utc": detection_time,
            "last_seen_utc": detection_time,
            "required_action": "Review changed AI_book source files and update roadmap matrix evidence, missing_items, or smallest_next_patch if the implementation contract changed.",
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        }
        for row in items
        if row["source_sync"]["stale_source_candidate"]
    ]
    backlog_merge = merge_unresolved_backlog_items(new_backlog_items, previous_crosswalk, detection_time, source_sync_decisions)
    backlog_items = backlog_merge["active"]
    theseus_to_book_evidence = build_theseus_to_book_evidence(items)
    public_safe_evidence_smoke = public_safe_evidence_export_smoke()
    trigger_state = "GREEN"
    if missing_source_basis or missing_evidence or not source_inventory["exists"] or not source_sync_smoke["passed"] or not public_safe_evidence_smoke["passed"]:
        trigger_state = "YELLOW"
    if stale_phases:
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_book_to_theseus_crosswalk_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "matrix": rel(matrix_path),
            "registry": rel(registry_path),
            "crosswalk": rel(crosswalk_path),
            "ai_book_root": source_inventory["root"],
            "ai_book_source_file_count": source_inventory["source_count"],
            "ai_book_source_manifest_hash": source_inventory["manifest_hash"],
            "previous_source_inventory_available": previous_inventory_available,
            "changed_source_file_count": len(changed_source_paths),
            "removed_source_file_count": len(removed_source_paths),
            "stale_phase_count": len(stale_phases),
            "roadmap_backlog_item_count": len(backlog_items),
            "source_sync_review_decision_count": len(source_sync_decisions),
            "cleared_roadmap_backlog_item_count": len(backlog_merge["cleared"]),
            "theseus_to_book_evidence_count": len(theseus_to_book_evidence),
            "public_safe_evidence_smoke_passed": public_safe_evidence_smoke["passed"],
            "source_sync_smoke_passed": source_sync_smoke["passed"],
            "crosswalk_item_count": len(items),
            "missing_source_basis_count": len(missing_source_basis),
            "done_phase_missing_evidence_count": len(missing_evidence),
        },
        "rules": {
            "source_basis": "Every roadmap phase keeps its AI_book source basis attached to the registry surface and abstraction it affects.",
            "evidence_sync": "Completion state must be traceable to current evidence and gates, not prose.",
            "no_bloat": "New book-derived work should improve registered surfaces before creating new surfaces.",
            "source_sync": "The crosswalk stores checksums and support states for AI_book source files, not raw source text.",
            "theseus_to_book_evidence": "Public-safe book evidence is pointer-only and excludes private/model/checkpoint/benchmark payload families.",
        },
        "source_inventory": source_inventory,
        "changed_sources": {
            "changed_source_paths": changed_source_paths,
            "removed_source_paths": removed_source_paths,
            "stale_phases": stale_phases,
        },
        "source_sync_smoke": source_sync_smoke,
        "public_safe_evidence_smoke": public_safe_evidence_smoke,
        "roadmap_backlog_items": backlog_items,
        "cleared_roadmap_backlog_items": backlog_merge["cleared"],
        "theseus_to_book_evidence": theseus_to_book_evidence,
        "items": items,
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
    }


def build_ai_book_source_inventory(ai_book_root: Path) -> dict[str, Any]:
    root = ai_book_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return {
            "root": str(root),
            "exists": False,
            "source_count": 0,
            "manifest_hash": stable_hash({"root": str(root), "exists": False}),
            "sources": [],
        }
    sources: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root)
        parts = set(rel_path.parts)
        if parts & AI_BOOK_IGNORED_PARTS:
            continue
        if path.suffix.lower() not in AI_BOOK_SOURCE_EXTENSIONS:
            continue
        digest = hash_file(path)
        stat = path.stat()
        sources.append(
            {
                "source_id": stable_id("ai_book_source", str(rel_path), digest),
                "path": str(rel_path),
                "extension": path.suffix.lower(),
                "role": classify_ai_book_source(rel_path),
                "bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": digest,
                "text_fingerprint": digest[:16],
            }
        )
    manifest_payload = [
        {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"], "role": row["role"]}
        for row in sources
    ]
    return {
        "root": str(root),
        "exists": True,
        "source_count": len(sources),
        "manifest_hash": stable_hash(manifest_payload),
        "sources": sources,
    }


def classify_ai_book_source(rel_path: Path) -> str:
    first = rel_path.parts[0] if rel_path.parts else rel_path.name
    stem = rel_path.stem.lower()
    if first in {"chapters", "parts", "appendices", "frontmatter"}:
        return "book_content"
    if first in {"schemas", "schema"}:
        return "schema"
    if first in {"proofs", "lean"} or rel_path.suffix.lower() == ".lean":
        return "formal_proof"
    if first in {"experiments", "evals", "benchmarks"}:
        return "experiment"
    if first in {"manifests", "registry", "registries"} or "manifest" in stem:
        return "manifest"
    if first in {"docs", "notes", "sources"}:
        return "source_note"
    if first in {"release_records", "editions"}:
        return "release_record"
    return "root_source"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes_by_path(crosswalk: dict[str, Any]) -> dict[str, str]:
    inventory = crosswalk.get("source_inventory")
    if not isinstance(inventory, dict):
        return {}
    return {
        str(row.get("path") or ""): str(row.get("sha256") or "")
        for row in list_dicts(inventory.get("sources"))
        if row.get("path") and row.get("sha256")
    }


def match_sources_for_phase(phase: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    basis = [str(item) for item in list_values(phase.get("ai_book_source_basis"))]
    explicit_paths = {normalize_source_path(item) for item in basis if looks_like_source_path(item)}
    tokens = phase_match_tokens(phase)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for row in sources:
        path = str(row.get("path") or "")
        search_text = f"{path} {row.get('role') or ''} {Path(path).stem}".lower().replace("_", " ").replace("-", " ")
        score = 0
        normalized_path = normalize_source_path(path)
        if normalized_path in explicit_paths:
            score += 1000
        for explicit in explicit_paths:
            if explicit and (explicit in normalized_path or normalized_path in explicit):
                score += 400
        for token in tokens:
            if token in search_text:
                score += 10
        if str(row.get("role") or "") in {"schema", "manifest", "formal_proof"}:
            score += 1
        if score > 0:
            scored.append((score, path, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "source_id": row.get("source_id"),
            "path": row.get("path"),
            "role": row.get("role"),
            "sha256": row.get("sha256"),
            "score": score,
        }
        for score, _path, row in scored[:20]
    ]


def phase_match_tokens(phase: dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            str(phase.get("title") or ""),
            str(phase.get("roadmap_section") or ""),
            " ".join(str(item) for item in list_values(phase.get("ai_book_source_basis"))),
            " ".join(str(item) for item in list_values(phase.get("execution_spine_hooks"))),
            " ".join(str(item) for item in list_values(phase.get("required_records"))),
        ]
    )
    raw = text.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    ignored = {
        "and",
        "the",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "phase",
        "v0",
        "v1",
        "theseus",
        "implementation",
        "record",
        "records",
    }
    return {token for token in raw.split() if len(token) >= 4 and token not in ignored}


def looks_like_source_path(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.endswith(ext) for ext in AI_BOOK_SOURCE_EXTENSIONS) or "/" in value


def normalize_source_path(value: str) -> str:
    path = value.strip()
    if path.startswith(str(DEFAULT_AI_BOOK_ROOT)):
        try:
            path = str(Path(path).resolve().relative_to(DEFAULT_AI_BOOK_ROOT.resolve()))
        except ValueError:
            pass
    return path.lstrip("./").replace("\\", "/")


def source_sync_detection_smoke() -> dict[str, Any]:
    previous = {"stable.md": "aaa", "changed.md": "old", "removed.md": "gone"}
    current = {"stable.md": "aaa", "changed.md": "new", "added.md": "fresh"}
    changed = sorted(path for path, digest in current.items() if previous.get(path) not in {None, digest})
    removed = sorted(path for path in previous if path not in current)
    passed = changed == ["changed.md"] and removed == ["removed.md"]
    return {
        "policy": "roadmap_source_sync_detection_smoke_v1",
        "passed": passed,
        "changed_source_paths": changed,
        "removed_source_paths": removed,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def source_sync_review_decisions(project_steward: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = []
    for row in list_dicts(project_steward.get("source_sync_review_decisions")):
        if str(row.get("status") or "") != "active":
            continue
        phase = int_or(row.get("phase"), -1)
        decision = str(row.get("decision") or "")
        if phase < 0 or decision not in {"matrix_updated", "reviewed_no_contract_change", "superseded"}:
            continue
        decisions.append(
            {
                **row,
                "phase": phase,
                "changed_source_paths": sorted(normalize_source_path(str(path)) for path in list_values(row.get("changed_source_paths"))),
                "backlog_ids": [str(item) for item in list_values(row.get("backlog_ids"))],
                "matched_source_hash": str(row.get("matched_source_hash") or ""),
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "fallback_return_count": 0,
            }
        )
    return decisions


def matching_source_sync_decision(
    decisions: list[dict[str, Any]],
    *,
    phase_id: int,
    changed_source_paths: list[str],
    matched_source_hash: str,
) -> dict[str, Any]:
    changed = set(normalize_source_path(path) for path in changed_source_paths)
    if not changed:
        return {}
    for decision in decisions:
        if int_or(decision.get("phase"), -1) != phase_id:
            continue
        decision_paths = set(str(path) for path in list_values(decision.get("changed_source_paths")))
        decision_hash = str(decision.get("matched_source_hash") or "")
        if decision_hash and decision_hash == matched_source_hash:
            return decision
        if decision_paths and changed.issubset(decision_paths):
            return decision
    return {}


def backlog_item_clearing_decision(item: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    phase = int_or(item.get("phase"), -1)
    backlog_id = str(item.get("backlog_id") or "")
    paths = set(normalize_source_path(str(path)) for path in list_values(item.get("changed_source_paths")))
    for decision in decisions:
        if int_or(decision.get("phase"), -1) != phase:
            continue
        if backlog_id and backlog_id in set(str(item) for item in list_values(decision.get("backlog_ids"))):
            return decision
        decision_paths = set(str(path) for path in list_values(decision.get("changed_source_paths")))
        if paths and decision_paths and paths.issubset(decision_paths):
            return decision
    return {}


def merge_unresolved_backlog_items(
    new_items: list[dict[str, Any]],
    previous_crosswalk: dict[str, Any],
    seen_time: str,
    source_sync_decisions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    previous_manifest_hash = str(dict_value(previous_crosswalk.get("summary")).get("ai_book_source_manifest_hash") or "")
    merged: dict[str, dict[str, Any]] = {}
    cleared: list[dict[str, Any]] = []
    for item in list_dicts(previous_crosswalk.get("roadmap_backlog_items")):
        state = str(item.get("state") or "")
        if state in {"CLEARED", "RESOLVED", "SUPERSEDED"}:
            continue
        backlog_id = str(item.get("backlog_id") or "")
        if not backlog_id:
            continue
        clearing_decision = backlog_item_clearing_decision(item, source_sync_decisions)
        if clearing_decision:
            cleared.append(
                {
                    **item,
                    "state": "RESOLVED",
                    "resolved_utc": seen_time,
                    "resolution_decision_id": clearing_decision.get("id"),
                    "resolution_decision": clearing_decision.get("decision"),
                    "resolution_reason": clearing_decision.get("reason"),
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                }
            )
            continue
        carried = dict(item)
        carried.setdefault("first_detected_utc", seen_time)
        if previous_manifest_hash:
            carried.setdefault("source_manifest_hash_at_detection", previous_manifest_hash)
        carried["last_seen_utc"] = seen_time
        carried["carry_forward_reason"] = "unresolved_stale_ai_book_source_review"
        merged[backlog_id] = carried
    for item in new_items:
        backlog_id = str(item.get("backlog_id") or "")
        if not backlog_id:
            continue
        clearing_decision = backlog_item_clearing_decision(item, source_sync_decisions)
        if clearing_decision:
            cleared.append(
                {
                    **item,
                    "state": "RESOLVED",
                    "resolved_utc": seen_time,
                    "resolution_decision_id": clearing_decision.get("id"),
                    "resolution_decision": clearing_decision.get("decision"),
                    "resolution_reason": clearing_decision.get("reason"),
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                }
            )
            continue
        if backlog_id in merged:
            preserved_first_seen = merged[backlog_id].get("first_detected_utc")
            merged[backlog_id].update(item)
            if preserved_first_seen:
                merged[backlog_id]["first_detected_utc"] = preserved_first_seen
            merged[backlog_id]["last_seen_utc"] = seen_time
        else:
            merged[backlog_id] = item
    return {
        "active": sorted(merged.values(), key=lambda row: (int_or(row.get("phase"), -1), str(row.get("backlog_id") or ""))),
        "cleared": sorted(cleared, key=lambda row: (int_or(row.get("phase"), -1), str(row.get("backlog_id") or ""))),
    }


def build_theseus_to_book_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for ref in [str(value) for value in list_values(item.get("current_evidence"))]:
            safety = public_safe_evidence_ref_state(ref)
            if not safety["exportable"]:
                continue
            rows.append(
                {
                    "record_type": "theseus_to_book_evidence",
                    "evidence_id": stable_id("theseus_to_book_evidence", item.get("phase"), ref),
                    "phase": item.get("phase"),
                    "title": item.get("title"),
                    "registry_surface_id": item.get("registry_surface_id"),
                    "abstraction_id": item.get("abstraction_id"),
                    "evidence_ref": ref,
                    "public_safe_state": safety["state"],
                    "public_safe_reason": safety["reason"],
                    "support_state": "PUBLIC_SAFE_POINTER_ONLY",
                    "book_use": "implementation_reference_pointer_not_raw_payload",
                    "non_claims": [
                        "evidence pointer is not learned-model capability evidence by itself",
                        "book export does not include raw private traces, benchmark payloads, checkpoints, or candidate bodies",
                    ],
                    "public_training_rows_written": 0,
                    "external_inference_calls": 0,
                    "fallback_return_count": 0,
                }
            )
    rows.sort(key=lambda row: (int_or(row.get("phase"), -1), str(row.get("evidence_ref") or "")))
    return rows


def public_safe_evidence_ref_state(ref: str) -> dict[str, Any]:
    normalized = ref.lower()
    blocked_markers = [
        "checkpoint",
        ".pt",
        ".pth",
        ".bin",
        "candidate",
        "strict_generator",
        "private_train",
        "teacher",
        "benchmark_payload",
        "public_calibration",
        "invite_private",
        "secret",
        "token",
    ]
    if any(marker in normalized for marker in blocked_markers):
        return {"exportable": False, "state": "BLOCKED_PRIVATE_OR_MODEL_PAYLOAD", "reason": "blocked_marker"}
    if normalized.startswith(("docs/", "roadmap.md", "agents.md", "configs/", "scripts/")):
        return {"exportable": True, "state": "PUBLIC_SAFE_SOURCE_OR_CONFIG_POINTER", "reason": "source_or_config_pointer"}
    if normalized.startswith("reports/"):
        safe_report_markers = [
            "roadmap_implementation_gate",
            "book_to_theseus_crosswalk",
            "module_definition_of_done",
            "theseus_project_registry",
            "viea_spine",
            "report_evidence_store",
            "theseus_artifact_retention",
            "hive_artifact",
            "hive_installer_artifacts",
            "theseus_control_plane",
        ]
        if any(marker in normalized for marker in safe_report_markers):
            return {"exportable": True, "state": "PUBLIC_SAFE_REPORT_POINTER", "reason": "governance_or_registry_report_pointer"}
        return {"exportable": False, "state": "BLOCKED_UNCLASSIFIED_REPORT_POINTER", "reason": "unclassified_report_family"}
    return {"exportable": False, "state": "BLOCKED_UNKNOWN_POINTER_CLASS", "reason": "unknown_ref_class"}


def public_safe_evidence_export_smoke() -> dict[str, Any]:
    allowed = public_safe_evidence_ref_state("reports/book_to_theseus_crosswalk.json")
    blocked_checkpoint = public_safe_evidence_ref_state("reports/student_checkpoint.pt")
    blocked_private = public_safe_evidence_ref_state("reports/strict_generator_private_train_replay.json")
    passed = allowed["exportable"] and not blocked_checkpoint["exportable"] and not blocked_private["exportable"]
    return {
        "policy": "project_theseus_public_safe_evidence_export_smoke_v1",
        "passed": passed,
        "allowed_probe_state": allowed["state"],
        "blocked_checkpoint_state": blocked_checkpoint["state"],
        "blocked_private_report_state": blocked_private["state"],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    digest = stable_hash([str(part) for part in parts])[:16]
    return f"{prefix}-{digest}"


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "trigger_state": report["trigger_state"],
        "phase_count": summary["phase_count"],
        "implemented_or_wired_count": summary["implemented_or_wired_count"],
        "partial_count": summary["partial_count"],
        "missing_count": summary["missing_count"],
        "frozen_count": summary["frozen_count"],
        "book_to_theseus_crosswalk_item_count": summary.get("book_to_theseus_crosswalk_item_count", 0),
        "book_to_theseus_ai_book_source_file_count": summary.get("book_to_theseus_ai_book_source_file_count", 0),
        "book_to_theseus_stale_phase_count": summary.get("book_to_theseus_stale_phase_count", 0),
        "book_to_theseus_roadmap_backlog_item_count": summary.get("book_to_theseus_roadmap_backlog_item_count", 0),
        "theseus_to_book_evidence_count": summary.get("theseus_to_book_evidence_count", 0),
        "book_to_theseus_source_sync_smoke_passed": summary.get("book_to_theseus_source_sync_smoke_passed", False),
        "public_safe_evidence_smoke_passed": summary.get("public_safe_evidence_smoke_passed", False),
        "book_implementation_track_count": summary.get("book_implementation_track_count", 0),
        "book_chapter_implementation_crosswalk_count": summary.get("book_chapter_implementation_crosswalk_count", 0),
        "book_manifest_chapter_count": summary.get("book_manifest_chapter_count", 0),
        "book_manifest_order_match": summary.get("book_manifest_order_match", False),
        "book_manifest_digest_match": summary.get("book_manifest_digest_match", False),
        "book_manifest_source": summary.get("book_manifest_source", ""),
        "book_manifest_commit": summary.get("book_manifest_commit", ""),
        "live_book_manifest_differs_from_pin": summary.get(
            "live_book_manifest_differs_from_pin", False
        ),
        "book_manifest_source_field_drift_chapter_count": summary.get(
            "book_manifest_source_field_drift_chapter_count", 0
        ),
        "book_manifest_source_field_drift_count": summary.get("book_manifest_source_field_drift_count", 0),
        "book_codex_test_count": summary.get("book_codex_test_count", 0),
        "book_pending_or_partial_codex_test_count": summary.get("book_pending_or_partial_codex_test_count", 0),
        "book_future_candidate_count": summary.get("book_future_candidate_count", 0),
        "book_future_candidate_chapter_count": summary.get("book_future_candidate_chapter_count", 0),
        "book_future_cross_cutting_section_count": summary.get("book_future_cross_cutting_section_count", 0),
        "book_future_candidate_missing_required_field_count": summary.get("book_future_candidate_missing_required_field_count", 0),
        "book_future_candidate_invalid_ref_count": summary.get("book_future_candidate_invalid_ref_count", 0),
        "book_future_candidate_disposition_counts": summary.get("book_future_candidate_disposition_counts", {}),
        "planned_codex_test_backlog_count": summary.get("planned_codex_test_backlog_count", 0),
        "planned_codex_test_backlog_missing_required_field_count": summary.get(
            "planned_codex_test_backlog_missing_required_field_count", 0
        ),
        "planned_codex_test_backlog_invalid_ref_count": summary.get("planned_codex_test_backlog_invalid_ref_count", 0),
        "planned_codex_test_backlog_status_counts": summary.get("planned_codex_test_backlog_status_counts", {}),
        "planned_codex_test_backlog_technique_family_counts": summary.get(
            "planned_codex_test_backlog_technique_family_counts", {}
        ),
        "planned_codex_test_backlog_blocked_or_queued_count": summary.get(
            "planned_codex_test_backlog_blocked_or_queued_count", 0
        ),
        "book_active_flagship_lane_id": summary.get("book_active_flagship_lane_id", ""),
        "book_active_core_slice_count": summary.get("book_active_core_slice_count", 0),
        "book_active_core_slice_support_states": summary.get("book_active_core_slice_support_states", {}),
        "book_core_slice_support_states": summary.get("book_core_slice_support_states", {}),
        "book_support_state_ladder_ready": summary.get("book_support_state_ladder_ready", False),
        "pre_training_architecture_ready": summary.get("pre_training_architecture_ready", False),
        "pre_training_architecture_blocker_count": summary.get("pre_training_architecture_blocker_count", 0),
        "pre_training_architecture_warning_count": summary.get("pre_training_architecture_warning_count", 0),
        "book_chapter_crosswalk_missing_required_field_count": summary.get("book_chapter_crosswalk_missing_required_field_count", 0),
        "book_chapter_invalid_phase_ref_count": summary.get("book_chapter_invalid_phase_ref_count", 0),
        "hard_gap_count": summary["hard_gap_count"],
        "warning_count": summary["warning_count"],
        "phase_13_to_19_preserved": summary["phase_13_to_19_preserved"],
    }


def gap(identifier: str, kind: str, evidence: dict[str, Any], *, severity: str = "hard") -> dict[str, Any]:
    return {"id": identifier, "kind": kind, "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_external(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
