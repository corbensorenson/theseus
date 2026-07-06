"""Shared progress-integrity rules for ASI-facing automation.

The project can keep diagnostic probes around, but the unattended frontier
runner must not promote or repeatedly train on concepts that only prove that
templates/skeletons can be emitted. This module centralizes that boundary so
the scheduler, Hive board, and command router do not drift apart.
"""

from __future__ import annotations

from typing import Any


NON_PROMOTABLE_DIAGNOSTIC_CONCEPT_MARKERS = (
    "typed_interface_private_closure",
    "edge_contract_private_closure",
    "edge_contract_balanced_private_closure",
    "edge_case_full_body_private_closure",
    "edge_contract_v2_private_closure",
    "candidate_floor_v2_private_closure",
    "candidate_floor_v2",
    "typed_interface_skeleton",
)

PROMOTION_SAFE_REPLACEMENT_CONCEPTS = {
    "typed_interface_private_closure": "private_pressure_private_closure",
    "edge_contract_private_closure": "private_pressure_private_closure",
    "edge_contract_balanced_private_closure": "private_pressure_private_closure",
    "edge_case_full_body_private_closure": "private_pressure_private_closure",
    "edge_contract_v2_private_closure": "private_pressure_private_closure",
    "candidate_floor_v2_private_closure": "private_pressure_private_closure",
    "candidate_floor_v2": "admissibility_and_interface",
    "typed_interface_skeleton": "type_and_return_shape",
}


def normalize_concept(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def non_promotable_diagnostic_marker(value: Any) -> str:
    concept = normalize_concept(value)
    for marker in NON_PROMOTABLE_DIAGNOSTIC_CONCEPT_MARKERS:
        if marker in concept:
            return marker
    return ""


def is_non_promotable_diagnostic_concept(value: Any) -> bool:
    return bool(non_promotable_diagnostic_marker(value))


def non_promotable_diagnostic_reason(value: Any) -> str:
    marker = non_promotable_diagnostic_marker(value)
    if not marker:
        return ""
    return (
        f"{marker}_is_diagnostic_only_for_progress_integrity:"
        "it may expose residuals, but it is not capability promotion evidence"
    )


def promotion_safe_replacement_concept(value: Any) -> str:
    marker = non_promotable_diagnostic_marker(value)
    return PROMOTION_SAFE_REPLACEMENT_CONCEPTS.get(marker, "")


def apply_non_promotable_diagnostic_policy(row: dict[str, Any]) -> dict[str, Any]:
    """Mutate a scheduler/board row so diagnostic concepts cannot be selected."""

    concept = row.get("concept")
    reason = non_promotable_diagnostic_reason(concept)
    if not reason:
        return row
    previous_status = str(row.get("status") or "")
    if previous_status in {"ready", "queued", "failed", "waiting_private_closure", "waiting_recalibration"}:
        row["status"] = "diagnostic_only"
    row["progress_integrity"] = {
        "promotion_safe": False,
        "diagnostic_only": True,
        "reason": reason,
        "previous_status": previous_status or None,
        "replacement_concept": promotion_safe_replacement_concept(concept) or None,
    }
    evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    evidence["progress_integrity"] = row["progress_integrity"]
    row["evidence"] = evidence
    return row
