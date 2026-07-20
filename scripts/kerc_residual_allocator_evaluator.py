#!/usr/bin/env python3
"""Independent consequence evaluator for KERC per-unit allocator decisions.

This module does not produce training labels.  It reconstructs the payload chosen
by a policy, applies an independently implemented set of executable invariants,
and reports hard violations, exact preservation, bit cost, regret, and confidence
calibration.  Keeping it separate from ``kerc_residual_interventions`` prevents a
single semantic proxy from both manufacturing and judging held-out targets.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from kerc_residual_economics import (
    FIDELITY_ORDER,
    digest,
    digest_bytes,
    project_residual_unit_payload,
    residual_unit_sources,
    residual_wire_bytes,
)


POLICY = "project_theseus_kerc_independent_allocator_evaluation_v1"
EVALUATOR_ID = "kerc_human_annotation_consequence_evaluator_v2"
_PARTS = re.compile(r"[^a-z0-9]+")
_ANNOTATION_SEMANTIC_FIELDS = {
    "answer_span", "attitude", "attributes", "category", "centering",
    "complete_component", "disposition", "edge_kind", "entity_type",
    "event_type", "excerpt_span", "excerpt_spans", "frame_elements",
    "frame_name", "group_id", "information_status", "label", "layer",
    "lexical_unit", "mention_id", "mention_ids", "modality", "nuclearity",
    "object_type", "polarity", "primary_nuclearity", "primary_relation",
    "primary_relation_base", "question_form", "relation", "relation_type",
    "response", "responses", "role", "spans", "stable_identity", "status",
    "support_relation", "target_spans", "text",
}
_STOPWORDS = {
    "annotation", "complete", "false", "manual", "none", "source", "true",
}
_SAFETY_PATH_PARTS = {
    "access", "access_policy", "authority", "credential", "permission", "privacy",
    "secret", "security",
}


class AllocatorEvaluationFault(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def __reduce__(self) -> tuple[Any, tuple[str, str]]:
        return self.__class__, (self.code, self.detail)


def _semantic_values(value: Any) -> set[str]:
    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {f"number:{value}"}
    raw = str(value).strip()
    if not raw or raw.startswith("sha256:"):
        return set()
    parts = [part for part in _PARTS.split(raw.casefold()) if len(part) >= 2]
    values = {part for part in parts if part not in _STOPWORDS}
    if parts:
        values.add(".".join(parts))
        for width in range(2, min(5, len(parts) + 1)):
            values.add(".".join(parts[-width:]))
    return {item for item in values if item and item not in _STOPWORDS}


def _annotation_values(value: Any, *, admitted: bool = False) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            child_admitted = admitted or key in _ANNOTATION_SEMANTIC_FIELDS
            if key.endswith("_sha256") or key in {
                "line_number", "source_row_sha256", "source_annotation_sha256",
            }:
                continue
            if child_admitted:
                values.update(_semantic_values(key))
            values.update(_annotation_values(child, admitted=child_admitted))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.update(_annotation_values(child, admitted=admitted))
    elif admitted:
        values.update(_semantic_values(value))
    return values


def _payload_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key.endswith("_sha256") or key in {
                "authority", "policy", "record_sha256", "source_annotation_sha256",
            }:
                continue
            values.update(_semantic_values(key))
            values.update(_payload_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.update(_payload_values(child))
    else:
        values.update(_semantic_values(value))
    return values


def _requires_exact_safety(unit: dict[str, Any]) -> bool:
    if str(unit.get("unit_kind") or "") == "exact_object":
        return True
    if str(unit.get("unit_kind") or "") != "interaction_entry":
        return False
    parts = set(_PARTS.split(str(unit.get("source_path") or "").casefold()))
    return bool(parts & _SAFETY_PATH_PARTS)


def _independent_action_checks(
    unit: dict[str, Any], source: Any, candidate_payload: Any
) -> list[dict[str, Any]]:
    kind = str(unit["unit_kind"])
    exact = residual_wire_bytes(source) == residual_wire_bytes(candidate_payload)
    checks = [
        {
            "check": "k2_hard_minimum",
            "pass": not bool(
                next(
                    row
                    for row in unit["candidates"]
                    if row["payload_sha256"]
                    == digest_bytes(residual_wire_bytes(candidate_payload))
                )["hard_blocked"]
            ),
        }
    ]
    if kind == "exact_object":
        checks.append({"check": "protected_object_byte_identity", "pass": exact})
    elif _requires_exact_safety(unit):
        checks.append({"check": "security_contract_byte_identity", "pass": exact})
    return checks


def evaluate_allocator_predictions(
    *,
    unit_packet: dict[str, Any],
    source_record_sha256: str,
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    concept_capsules: dict[str, Any],
    exact_objects: dict[str, Any],
    source_family: str,
    source_annotation: dict[str, Any],
    predictions: Iterable[dict[str, Any]],
    training_target_producer_id: str,
) -> dict[str, Any]:
    if training_target_producer_id == EVALUATOR_ID:
        raise AllocatorEvaluationFault(
            "KERC_ALLOCATOR_EVALUATOR_NOT_INDEPENDENT", training_target_producer_id
        )
    sources = residual_unit_sources(
        source_record_sha256=source_record_sha256,
        global_state=global_state,
        segment_residual=segment_residual,
        token_residuals=token_residuals,
        concept_capsules=concept_capsules,
        exact_objects=exact_objects,
    )
    source_by_id = {str(row["unit_id"]): row["source_payload"] for row in sources}
    source_values_by_id = {
        unit_id: _payload_values(payload) for unit_id, payload in source_by_id.items()
    }
    units = {str(row["unit_id"]): row for row in unit_packet.get("units") or []}
    prediction_rows = list(predictions)
    if set(source_by_id) != set(units) or {str(row.get("unit_id")) for row in prediction_rows} != set(units):
        raise AllocatorEvaluationFault(
            "KERC_ALLOCATOR_EVALUATION_UNIT_INVENTORY_INVALID", source_record_sha256
        )
    decisions = []
    annotation_values = _annotation_values(source_annotation)
    annotation_policy = str(
        source_annotation.get("policy")
        or source_annotation.get("source_kind")
        or "untyped_annotation"
    )
    hard_violations = 0
    total_bits = 0
    oracle_bits = 0
    rate_regret_bits = 0
    raw_rate_delta_bits = 0
    exact_count = 0
    brier_sum = 0.0
    brier_count = 0
    adjudicated_count = 0
    semantic_agreement_count = 0
    semantic_violation_count = 0
    semantic_fact_count = 0
    retained_semantic_fact_count = 0
    unadjudicated_count = 0
    kind_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction in prediction_rows:
        unit_id = str(prediction["unit_id"])
        unit = units[unit_id]
        source = source_by_id[unit_id]
        selected_index = int(prediction.get("selected_fidelity_index", -1))
        if selected_index not in range(len(FIDELITY_ORDER)):
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_SELECTION_INVALID", unit_id
            )
        fidelity = FIDELITY_ORDER[selected_index]
        candidate = unit["candidates"][selected_index]
        projected = project_residual_unit_payload(
            source, fidelity=fidelity, unit_kind=str(unit["unit_kind"])
        )
        if digest_bytes(residual_wire_bytes(projected)) != candidate["payload_sha256"]:
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_PAYLOAD_MISMATCH", unit_id
            )
        checks = _independent_action_checks(unit, source, projected)
        passes = all(bool(row["pass"]) for row in checks)
        other_values = {
            value
            for other_id, values in source_values_by_id.items()
            if other_id != unit_id
            for value in values
        }
        source_values = source_values_by_id[unit_id]
        gold_values = (source_values - other_values) & annotation_values
        retained_gold_values = gold_values & (other_values | _payload_values(projected))
        semantic_loss = (
            (len(gold_values) - len(retained_gold_values)) / max(1, len(gold_values))
        )
        adjudicated = bool(gold_values) or _requires_exact_safety(unit)
        hard_violations += int(not passes)
        total_bits += int(candidate["encoded_bits"])
        admissible = []
        for index, action in enumerate(unit["candidates"]):
            payload = project_residual_unit_payload(
                source,
                fidelity=FIDELITY_ORDER[index],
                unit_kind=str(unit["unit_kind"]),
            )
            action_passes = all(
                row["pass"] for row in _independent_action_checks(unit, source, payload)
            )
            action_retains_gold = not gold_values or gold_values <= (
                other_values | _payload_values(payload)
            )
            if action_passes and (not adjudicated or action_retains_gold):
                admissible.append((index, action))
        if not admissible:
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_NO_ADMISSIBLE_ACTION", unit_id
            )
        oracle_index, oracle_action = min(
            admissible,
            key=lambda row: (int(row[1]["encoded_bits"]), int(row[0])),
        )
        independent_oracle_bits = int(oracle_action["encoded_bits"])
        oracle_bits += independent_oracle_bits
        if adjudicated:
            raw_rate_delta_bits += int(candidate["encoded_bits"]) - independent_oracle_bits
            rate_regret_bits += max(
                0, int(candidate["encoded_bits"]) - independent_oracle_bits
            )
        exact = residual_wire_bytes(source) == residual_wire_bytes(projected)
        exact_count += int(exact)
        confidence = float(prediction.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_CONFIDENCE_INVALID", unit_id
            )
        semantic_agreement = selected_index == oracle_index if adjudicated else None
        if adjudicated:
            adjudicated_count += 1
            semantic_agreement_count += int(bool(semantic_agreement))
            semantic_violation_count += int(not passes or semantic_loss > 0.0)
            semantic_fact_count += len(gold_values)
            retained_semantic_fact_count += len(retained_gold_values)
            brier_sum += (confidence - float(bool(semantic_agreement) and passes)) ** 2
            brier_count += 1
        else:
            unadjudicated_count += 1
        kind = str(unit["unit_kind"])
        kind_counts[kind]["count"] += 1
        kind_counts[kind]["hard_violation"] += int(not passes)
        kind_counts[kind]["adjudicated"] += int(adjudicated)
        kind_counts[kind]["semantic_decision_agreement"] += int(
            bool(semantic_agreement)
        )
        row = {
            "unit_id": unit_id,
            "unit_kind": kind,
            "selected_fidelity": fidelity,
            "selected_encoded_bits": int(candidate["encoded_bits"]),
            "independent_oracle_encoded_bits": independent_oracle_bits,
            "raw_rate_delta_bits": (
                int(candidate["encoded_bits"]) - independent_oracle_bits
                if adjudicated
                else None
            ),
            "rate_regret_bits": (
                max(0, int(candidate["encoded_bits"]) - independent_oracle_bits)
                if adjudicated
                else None
            ),
            "hard_checks": checks,
            "hard_checks_pass": passes,
            "human_gold_adjudicated": adjudicated,
            "human_gold_fact_count": len(gold_values),
            "retained_human_gold_fact_count": len(retained_gold_values),
            "human_gold_semantic_loss": round(semantic_loss, 8) if adjudicated else None,
            "human_gold_oracle_fidelity": FIDELITY_ORDER[oracle_index]
            if adjudicated
            else None,
            "semantic_decision_agreement": semantic_agreement,
            "exact_payload": exact,
            "confidence": confidence,
        }
        row["decision_sha256"] = digest(row)
        decisions.append(row)
    count = len(decisions)
    result = {
        "policy": POLICY,
        "evaluator_id": EVALUATOR_ID,
        "training_target_producer_id": training_target_producer_id,
        "organizationally_separate_from_target_producer": True,
        "source_record_sha256": source_record_sha256,
        "source_family": source_family,
        "unit_packet_sha256": str(unit_packet["packet_sha256"]),
        "decisions": decisions,
        "summary": {
            "unit_count": count,
            "hard_violation_count": hard_violations,
            "hard_violation_rate": round(hard_violations / max(1, count), 8),
            "exact_payload_rate": round(exact_count / max(1, count), 8),
            "selected_encoded_bits": total_bits,
            "independent_oracle_encoded_bits": oracle_bits,
            "raw_rate_delta_bits": raw_rate_delta_bits,
            "rate_regret_bits": rate_regret_bits,
            "confidence_brier": round(brier_sum / max(1, brier_count), 8),
            "human_gold_adjudicated_unit_count": adjudicated_count,
            "human_gold_unadjudicated_unit_count": unadjudicated_count,
            "human_gold_panel_coverage": round(adjudicated_count / max(1, count), 8),
            "semantic_decision_agreement_count": semantic_agreement_count,
            "human_gold_semantic_violation_count": semantic_violation_count,
            "human_gold_semantic_violation_rate": round(
                semantic_violation_count / max(1, adjudicated_count), 8
            ),
            "semantic_decision_agreement_rate": round(
                semantic_agreement_count / max(1, adjudicated_count), 8
            ),
            "human_gold_fact_count": semantic_fact_count,
            "retained_human_gold_fact_count": retained_semantic_fact_count,
            "human_gold_fact_recall": round(
                retained_semantic_fact_count / max(1, semantic_fact_count), 8
            ),
            "by_unit_kind": {
                kind: dict(counts) for kind, counts in sorted(kind_counts.items())
            },
        },
        "training_labels_produced": 0,
        "source_annotation_policy": annotation_policy,
        "source_annotation_used_for_training": False,
        "public_or_hidden_target_used": False,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    result["receipt_sha256"] = digest(result)
    return result
