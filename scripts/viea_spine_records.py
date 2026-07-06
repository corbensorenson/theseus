"""Shared VIEA spine record helpers.

This module is intentionally small: producers can keep domain-specific payloads
while agreeing on canonical record families, no-cheat counters, and profile
validation semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ASSISTANT_RUNTIME_REQUIRED_RECORDS = {
    "intent_contract",
    "command_contract",
    "context_abi_record",
    "context_transaction",
    "context_adequacy",
    "typed_job",
    "planforge_dag",
    "runtime_adapter_invocation",
    "authority_transition",
    "authority_use_receipt",
    "resource_budget",
    "generation_mode",
    "failure_boundary",
    "artifact_graph_record",
    "claim_record",
    "evidence_transition_record",
    "residual_record",
    "policy_optimization_record",
}

CANONICAL_ALIASES = {
    "artifact_graph": "artifact_graph_record",
    "authority_transition_record": "authority_transition",
    "constitutional_predicate_record": "constitutional_predicate",
    "context_adequacy_record": "context_adequacy",
    "context_transaction_record": "context_transaction",
    "costed_route_record": "costed_route",
    "evidence_transition": "evidence_transition_record",
    "evidence_transitions": "evidence_transition_record",
    "failure_boundary_map": "failure_boundary",
    "generation_mode_record": "generation_mode",
    "constitutional_predicates": "constitutional_predicate",
    "governance_right_record": "governance_right",
    "governance_rights": "governance_right",
    "proof_carrying_claim": "proof_carrying_claim",
    "proof_carrying_claims": "proof_carrying_claim",
    "resource_budget_record": "resource_budget",
    "semantic_ir_atom": "semantic_atom",
    "semantic_node_record": "semantic_node",
    "simulation_contract_record": "simulation_contract",
}

NO_CHEAT_COUNTERS = ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIALIZED_VIEW = ROOT / "reports" / "viea_spine_materialized_view.json"
ARTIFACT_CITATION_GROUPS = (
    "claim_ledger_entries",
    "artifact_records",
    "evidence_transitions",
    "authority_records",
    "failure_boundaries",
    "resource_route_records",
)


def canonical_record_type(record_type: Any, aliases: dict[str, str] | None = None) -> str:
    raw = str(record_type or "")
    merged = dict(CANONICAL_ALIASES)
    if aliases:
        merged.update({str(key): str(value) for key, value in aliases.items()})
    return merged.get(raw, raw)


def collect_record_types(rows: list[dict[str, Any]], aliases: dict[str, str] | None = None) -> set[str]:
    out: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.add(canonical_record_type(row.get("record_type"), aliases))
    return out


def missing_required_record_types(
    rows: list[dict[str, Any]],
    required: set[str] | list[str] | tuple[str, ...] = ASSISTANT_RUNTIME_REQUIRED_RECORDS,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    observed = collect_record_types(rows, aliases)
    canonical_required = {canonical_record_type(item, aliases) for item in required}
    return sorted(canonical_required - observed)


def trace_complete(
    rows: list[dict[str, Any]],
    required: set[str] | list[str] | tuple[str, ...] = ASSISTANT_RUNTIME_REQUIRED_RECORDS,
    aliases: dict[str, str] | None = None,
) -> bool:
    return not missing_required_record_types(rows, required, aliases)


def no_cheat_counters_clean(payload: dict[str, Any]) -> bool:
    return all(int_or(payload.get(key), 0) == 0 for key in NO_CHEAT_COUNTERS)


def compact_record_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep materialized spine records compact and free of raw prompt text."""
    keep_keys = [
        "record_id",
        "artifact_id",
        "artifact_type",
        "parent_job",
        "claim_id",
        "proof_claim_id",
        "transaction_id",
        "operation",
        "snapshot_id",
        "node_id",
        "goal_id",
        "run_id",
        "case_id",
        "tool_id",
        "adapter_id",
        "route_id",
        "route_validator_receipt_id",
        "route_validator_ready",
        "placement_id",
        "route_phase",
        "task_kind",
        "target",
        "node_name",
        "source_refs",
        "context_refs",
        "context_transaction_refs",
        "semantic_certificate_refs",
        "tool_refs",
        "claim_refs",
        "test_refs",
        "audit_events",
        "replay_metadata",
        "replay_grade",
        "environment_assumptions",
        "provenance_status",
        "replay_limits",
        "evidence_gate",
        "residuals",
        "non_claims",
        "mounts",
        "read_set",
        "write_set",
        "branch_policy",
        "taint_labels",
        "deletion_obligations",
        "declassification_refs",
        "derivative_refs",
        "contradiction_refs",
        "materialization_state",
        "closure_state",
        "faults",
        "audit_refs",
        "replay_boundary",
        "audit_scope",
        "verifier_surface",
        "candidate_family",
        "candidate_count",
        "candidate_attempt_count",
        "audited_candidate_count",
        "integrity_verified_candidate_count",
        "integrity_mismatch_count",
        "syntax_invalid_count",
        "runtime_load_rate",
        "intended_behavior_pass_rate",
        "candidate_generation_credit",
        "job_id",
        "job_family",
        "arm_id",
        "backend_requirements",
        "authority_scope",
        "support_state",
        "verifier_state",
        "verifier_passed",
        "evidence_ref",
        "artifact_ref",
        "replay_checksum",
        "content_hash",
        "semantic_hash",
        "payload_bytes",
        "payload_truncated",
        "snapshot_path",
        "original_path",
        "archive_path",
        "pointer_path",
        "codec",
        "compression_scope",
        "reconstruction_contract",
        "archived_bytes",
        "source_artifact",
        "task_family",
        "access_pattern",
        "admission_state",
        "compression_method",
        "declared_use_envelope",
        "ratio_claim_state",
        "codec_parameters",
        "metadata_costs",
        "residual_coding",
        "probe_plan",
        "fallback_artifact",
        "fallback_trigger",
        "decode_determinism",
        "exact_replay_status",
        "consumer_policy",
        "utility_tests",
        "support_state_effect",
        "evidence_refs",
        "receipt_state",
        "public_law_family",
        "seed",
        "search_bound",
        "generated_regions",
        "verification_result",
        "repair_residual",
        "fallback_threshold",
        "interface_costs",
        "use_permissions",
        "proxy_rate_status",
        "final_serialization_status",
        "rate_accounting",
        "system_id",
        "target_system",
        "compact_seed",
        "rule_system",
        "memory_state",
        "generation_status",
        "residual_channel",
        "correction_mechanism",
        "verification_contract",
        "verification_status",
        "verifier_independence",
        "governance_interface",
        "authority_boundary",
        "use_envelope",
        "burden_ledger",
        "cost_accounting",
        "generative_leverage",
        "hidden_complexity_risks",
        "fallback_path",
        "fallback_status",
        "residual_burden_status",
        "promotion_state",
        "promotion_blockers",
        "retirement_condition",
        "source_refs",
        "defeater_type",
        "defeated_run_id",
        "defeating_run_id",
        "previous_content_hash",
        "current_content_hash",
        "previous_support_state",
        "current_support_state",
        "previous_trigger_state",
        "current_trigger_state",
        "gas_estimate_micro_twc",
        "provider_payout_micro_twc",
        "estimated_latency_ms",
        "network_class",
        "task_fit",
        "worker_limit",
        "quote_id",
        "currency_symbol",
        "simulation_id",
        "fidelity_level",
        "right_type",
        "predicate_id",
        "failure_id",
        "task_id",
        "route_state",
        "task_contract_ref",
        "quality_predicate",
        "authority_ceiling",
        "candidate_routes",
        "selected_route",
        "rejected_lower_cost_routes",
        "outcome_state",
        "cost_accounting",
        "cost_classes",
        "hidden_cost_checks",
        "residual_obligations",
        "fallback_route",
        "promotion_candidate",
        "blocked_reason",
        "state",
        "status",
        "terminal",
        "structured_non_solved",
        "fallback_return_used",
        "learned_generation_claim_allowed",
        "raw_prompt_stored",
        "raw_private_text_stored",
        *NO_CHEAT_COUNTERS,
    ]
    return {key: payload.get(key) for key in keep_keys if key in payload}


def load_materialized_view(path: str | Path | None = None) -> dict[str, Any]:
    view_path = resolve_path(path or DEFAULT_MATERIALIZED_VIEW)
    if not view_path.exists():
        return {
            "policy": "project_theseus_viea_spine_materialized_view_missing_v1",
            "trigger_state": "MISSING",
            "summary": {},
            "view_path": rel(view_path),
        }
    try:
        with view_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return {
            "policy": "project_theseus_viea_spine_materialized_view_invalid_v1",
            "trigger_state": "RED",
            "summary": {"json_error": str(exc)},
            "view_path": rel(view_path),
        }
    return data if isinstance(data, dict) else {"trigger_state": "RED", "summary": {}, "view_path": rel(view_path)}


def materialized_view_consumer_receipt(
    consumer_surface: str,
    *,
    required_groups: list[str] | tuple[str, ...],
    view: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    view_path = resolve_path(path or DEFAULT_MATERIALIZED_VIEW)
    payload = view if isinstance(view, dict) else load_materialized_view(view_path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    missing_groups = [group for group in required_groups if group_count(payload, group) <= 0]
    no_cheat_fault_count = int_or(summary.get("no_cheat_fault_count"), 0)
    ready = (
        payload.get("trigger_state") == "GREEN"
        and not missing_groups
        and no_cheat_fault_count == 0
        and int_or(summary.get("record_count"), 0) > 0
    )
    return {
        "record_type": "viea_spine_view_consumer_receipt",
        "receipt_id": stable_id("viea_spine_consumer", consumer_surface, rel(view_path), stable_hash(summary), required_groups),
        "consumer_surface": str(consumer_surface),
        "view_path": rel(view_path),
        "view_trigger_state": payload.get("trigger_state"),
        "view_content_hash": stable_hash(payload),
        "required_groups": list(required_groups),
        "missing_required_groups": missing_groups,
        "ready": ready,
        "record_count": int_or(summary.get("record_count"), 0),
        "claim_ledger_entry_count": group_count(payload, "claim_ledger_entries"),
        "semantic_ir_record_count": group_count(payload, "semantic_ir_records"),
        "simulation_fidelity_record_count": group_count(payload, "simulation_fidelity_records"),
        "governance_record_count": group_count(payload, "governance_records"),
        "failure_boundary_count": group_count(payload, "failure_boundaries"),
        "artifact_record_count": group_count(payload, "artifact_records"),
        "evidence_transition_count": group_count(payload, "evidence_transitions"),
        "authority_record_count": group_count(payload, "authority_records"),
        "runtime_adapter_record_count": group_count(payload, "runtime_adapter_records"),
        "resource_route_record_count": group_count(payload, "resource_route_records"),
        "generation_mode_record_count": group_count(payload, "generation_mode_records"),
        "context_record_count": group_count(payload, "context_records"),
        "no_cheat_fault_count": no_cheat_fault_count,
        "boundaries": {
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
        },
        "non_claim": "Consumer receipt proves this surface read the shared VIEA view; it is not learned-generation capability evidence.",
    }


def materialized_artifact_citation(
    consumer_surface: str,
    *,
    artifact_id: str = "",
    artifact_path: str = "",
    max_refs: int = 8,
    view: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Compact evidence citation for Hive artifact/package manifests.

    This intentionally cites materialized record ids and hashes rather than
    copying raw prompts, payloads, or report bodies into artifact manifests.
    """

    view_path = resolve_path(path or DEFAULT_MATERIALIZED_VIEW)
    payload = view if isinstance(view, dict) else load_materialized_view(view_path)
    receipt = materialized_view_consumer_receipt(
        consumer_surface,
        required_groups=list(ARTIFACT_CITATION_GROUPS),
        view=payload,
        path=view_path,
    )
    selection_context = {
        "consumer_surface": str(consumer_surface),
        "artifact_id": str(artifact_id or ""),
        "artifact_path": str(artifact_path or ""),
    }
    refs = {
        "claim_ledger_entries": compact_materialized_refs(payload.get("claim_ledger_entries"), max_refs=max_refs, **selection_context),
        "artifact_records": compact_materialized_refs(payload.get("artifact_records"), max_refs=max_refs, **selection_context),
        "evidence_transitions": compact_materialized_refs(payload.get("evidence_transitions"), max_refs=max_refs, **selection_context),
    }
    citation_id = stable_id(
        "viea_artifact_citation",
        consumer_surface,
        artifact_id,
        artifact_path,
        receipt.get("receipt_id"),
        refs,
    )
    return {
        "policy": "project_theseus_viea_artifact_citation_v1",
        "citation_id": citation_id,
        "consumer_surface": str(consumer_surface),
        "artifact_id": str(artifact_id or ""),
        "artifact_path": str(artifact_path or ""),
        "view_path": rel(view_path),
        "view_trigger_state": receipt.get("view_trigger_state"),
        "view_content_hash": receipt.get("view_content_hash"),
        "route_validator_receipt_id": receipt.get("receipt_id"),
        "ready": bool(receipt.get("ready")) and int_or(receipt.get("claim_ledger_entry_count"), 0) > 0,
        "required_groups": list(ARTIFACT_CITATION_GROUPS),
        "missing_required_groups": receipt.get("missing_required_groups"),
        "record_count": receipt.get("record_count"),
        "claim_ledger_entry_count": receipt.get("claim_ledger_entry_count"),
        "artifact_record_count": receipt.get("artifact_record_count"),
        "evidence_transition_count": receipt.get("evidence_transition_count"),
        "authority_record_count": receipt.get("authority_record_count"),
        "failure_boundary_count": receipt.get("failure_boundary_count"),
        "resource_route_record_count": receipt.get("resource_route_record_count"),
        "selection_policy": "best_valid_support_by_artifact_family_support_state_and_no_cheat_cleanliness_v1",
        "selection_context": selection_context,
        "claim_ledger_refs": refs["claim_ledger_entries"],
        "artifact_record_refs": refs["artifact_records"],
        "evidence_transition_refs": refs["evidence_transitions"],
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "non_claim": "Artifact citation anchors this manifest to the shared VIEA evidence view; it does not claim model capability or artifact quality.",
    }


def compact_materialized_refs(
    rows: Any,
    *,
    max_refs: int,
    consumer_surface: str = "",
    artifact_id: str = "",
    artifact_path: str = "",
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        if not record_no_cheat_clean(row):
            continue
        score, reasons = materialized_ref_score(row, consumer_surface=consumer_surface, artifact_id=artifact_id, artifact_path=artifact_path)
        candidates.append(
            {
                "selection_score": score,
                "selection_reasons": reasons,
                "record_id": row.get("record_id"),
                "canonical_record_type": row.get("canonical_record_type"),
                "producer_surface": row.get("producer_surface"),
                "source_profile": row.get("source_profile"),
                "source_path": row.get("source_path"),
                "source_locator": row.get("source_locator"),
                "content_hash": row.get("content_hash"),
                "support_state": compact_payload_value(row, "support_state")
                or compact_payload_value(row, "verifier_state")
                or compact_payload_value(row, "status")
                or compact_payload_value(row, "state")
                or ("ROUTE_VALIDATOR_READY" if compact_payload_value(row, "route_validator_ready") is True else None),
            }
        )
    candidates.sort(key=lambda item: (-int_or(item.get("selection_score")), str(item.get("source_path") or ""), str(item.get("record_id") or "")))
    return candidates[: max(1, int(max_refs))]


def materialized_ref_score(row: dict[str, Any], *, consumer_surface: str, artifact_id: str, artifact_path: str) -> tuple[int, list[str]]:
    text = materialized_ref_text(row)
    tokens = artifact_family_tokens(consumer_surface, artifact_id, artifact_path)
    score = 0
    reasons: list[str] = []
    for token in tokens:
        if token and token in text:
            score += 20
            reasons.append(f"token:{token}")
    support = str(
        compact_payload_value(row, "support_state")
        or compact_payload_value(row, "verifier_state")
        or compact_payload_value(row, "status")
        or compact_payload_value(row, "state")
        or ""
    ).lower()
    if support in {"supported", "verified", "green", "passed", "ready", "complete", "accepted"}:
        score += 40
        reasons.append(f"support:{support}")
    elif support in {"yellow", "partial", "warning", "planned"}:
        score += 10
        reasons.append(f"support:{support}")
    elif support in {"red", "failed", "blocked", "rejected", "invalid"}:
        score -= 30
        reasons.append(f"support:{support}")
    producer = str(row.get("producer_surface") or "")
    if producer and producer.lower() in text:
        score += 1
    if str(row.get("content_hash") or ""):
        score += 5
        reasons.append("content_hash")
    if str(row.get("record_id") or ""):
        score += 5
        reasons.append("record_id")
    if compact_payload_value(row, "raw_prompt_stored") is False:
        score += 5
        reasons.append("raw_prompt_not_stored")
    if compact_payload_value(row, "route_validator_ready") is True:
        score += 40
        reasons.append("route_validator_ready")
    if record_no_cheat_clean(row):
        score += 20
        reasons.append("no_cheat_clean")
    return score, reasons[:8]


def artifact_family_tokens(consumer_surface: str, artifact_id: str, artifact_path: str) -> list[str]:
    raw = " ".join([consumer_surface, artifact_id, artifact_path]).lower().replace("/", " ").replace(".", " ").replace("-", " ").replace("_", " ")
    stop = {"json", "reports", "report", "artifact", "artifacts", "sync", "index", "manifest", "payload", "metadata", "v1", "v0"}
    tokens = [part for part in raw.split() if len(part) >= 4 and part not in stop]
    if "hive" in raw and "hive" not in tokens:
        tokens.insert(0, "hive")
    seen: set[str] = set()
    out = []
    for token in tokens:
        if token not in seen:
            out.append(token)
            seen.add(token)
    return out[:12]


def materialized_ref_text(row: dict[str, Any]) -> str:
    compact = row.get("compact_payload") if isinstance(row.get("compact_payload"), dict) else {}
    fields = [
        row.get("record_id"),
        row.get("canonical_record_type"),
        row.get("source_record_type"),
        row.get("producer_surface"),
        row.get("source_profile"),
        row.get("source_path"),
        row.get("source_locator"),
        compact.get("claim_id"),
        compact.get("proof_claim_id"),
        compact.get("goal_id"),
        compact.get("node_id"),
        compact.get("tool_id"),
        compact.get("adapter_id"),
        compact.get("route_id"),
        compact.get("task_kind"),
        compact.get("job_family"),
        compact.get("arm_id"),
        compact.get("support_state"),
        compact.get("verifier_state"),
        compact.get("status"),
        compact.get("state"),
    ]
    return " ".join(str(item).lower() for item in fields if item not in (None, ""))


def compact_payload_value(row: dict[str, Any], key: str) -> Any:
    compact = row.get("compact_payload") if isinstance(row.get("compact_payload"), dict) else {}
    return row.get(key) if key in row else compact.get(key)


def record_no_cheat_clean(row: dict[str, Any]) -> bool:
    return all(int_or(compact_payload_value(row, key), 0) == 0 for key in NO_CHEAT_COUNTERS)


def group_count(view: dict[str, Any], group: str) -> int:
    value = view.get(group)
    if isinstance(value, list):
        return len(value)
    summary = view.get("summary") if isinstance(view.get("summary"), dict) else {}
    summary_key = {
        "claim_ledger_entries": "claim_ledger_entry_count",
        "semantic_ir_records": "semantic_ir_record_count",
        "simulation_fidelity_records": "simulation_fidelity_record_count",
        "governance_records": "governance_record_count",
        "failure_boundaries": "failure_boundary_count",
        "artifact_records": "artifact_record_count",
        "evidence_transitions": "evidence_transition_count",
        "authority_records": "authority_record_count",
        "runtime_adapter_records": "runtime_adapter_record_count",
        "resource_route_records": "resource_route_record_count",
        "generation_mode_records": "generation_mode_record_count",
        "context_records": "context_record_count",
    }.get(group, f"{group}_count")
    return int_or(summary.get(summary_key), 0)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{stable_hash(parts)[:16]}"


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
