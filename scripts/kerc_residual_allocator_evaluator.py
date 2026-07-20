#!/usr/bin/env python3
"""Independent consequence evaluator for KERC per-unit allocator decisions.

This module does not produce training labels.  It reconstructs the payload chosen
by a policy, applies an independently implemented set of executable invariants,
and reports hard violations, exact preservation, bit cost, regret, and confidence
calibration.  Keeping it separate from ``kerc_residual_interventions`` prevents a
single semantic proxy from both manufacturing and judging held-out targets.
"""

from __future__ import annotations

import copy
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
EVALUATOR_ID = "kerc_source_replay_consequence_evaluator_v1"


class AllocatorEvaluationFault(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def __reduce__(self) -> tuple[Any, tuple[str, str]]:
        return self.__class__, (self.code, self.detail)


def _has_key(value: Any, names: set[str]) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in names or _has_key(child, names) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_key(child, names) for child in value)
    return False


def _preserves_keyed_values(source: Any, candidate: Any, names: set[str]) -> bool:
    def collect(value: Any) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = []
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in names:
                    rows.append((str(key).lower(), copy.deepcopy(child)))
                rows.extend(collect(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                rows.extend(collect(child))
        return rows

    required = collect(source)
    observed = collect(candidate)
    return all(row in observed for row in required)


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
    elif kind == "concept_realization":
        checks.append(
            {
                "check": "concept_identity_preserved",
                "pass": _preserves_keyed_values(
                    source,
                    candidate_payload,
                    {"stable_identity", "preferred_realization", "realization"},
                ),
            }
        )
    elif kind == "segment_frame":
        checks.append(
            {
                "check": "frame_and_role_contract_preserved",
                "pass": _preserves_keyed_values(
                    source,
                    candidate_payload,
                    {"frame_name", "frame_roles", "target_spans", "operator", "relation"},
                ),
            }
        )
    elif kind == "token_residue":
        checks.append(
            {
                "check": "token_scope_and_tag_preserved",
                "pass": _preserves_keyed_values(
                    source, candidate_payload, {"source_span", "tag", "authority"}
                ),
            }
        )
    elif kind == "interaction_entry" and any(
        term in str(unit["source_path"]).lower()
        for term in ("security", "privacy", "access_policy", "terminology", "alias", "unit")
    ):
        checks.append({"check": "shared_contract_preserved", "pass": exact})
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
    units = {str(row["unit_id"]): row for row in unit_packet.get("units") or []}
    prediction_rows = list(predictions)
    if set(source_by_id) != set(units) or {str(row.get("unit_id")) for row in prediction_rows} != set(units):
        raise AllocatorEvaluationFault(
            "KERC_ALLOCATOR_EVALUATION_UNIT_INVENTORY_INVALID", source_record_sha256
        )
    decisions = []
    hard_violations = 0
    total_bits = 0
    oracle_bits = 0
    exact_count = 0
    brier_sum = 0.0
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
        hard_violations += int(not passes)
        total_bits += int(candidate["encoded_bits"])
        admissible = []
        for index, action in enumerate(unit["candidates"]):
            payload = project_residual_unit_payload(
                source,
                fidelity=FIDELITY_ORDER[index],
                unit_kind=str(unit["unit_kind"]),
            )
            if all(row["pass"] for row in _independent_action_checks(unit, source, payload)):
                admissible.append(action)
        if not admissible:
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_NO_ADMISSIBLE_ACTION", unit_id
            )
        independent_oracle_bits = min(int(row["encoded_bits"]) for row in admissible)
        oracle_bits += independent_oracle_bits
        exact = residual_wire_bytes(source) == residual_wire_bytes(projected)
        exact_count += int(exact)
        confidence = float(prediction.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise AllocatorEvaluationFault(
                "KERC_ALLOCATOR_EVALUATION_CONFIDENCE_INVALID", unit_id
            )
        brier_sum += (confidence - float(passes)) ** 2
        kind = str(unit["unit_kind"])
        kind_counts[kind]["count"] += 1
        kind_counts[kind]["hard_violation"] += int(not passes)
        row = {
            "unit_id": unit_id,
            "unit_kind": kind,
            "selected_fidelity": fidelity,
            "selected_encoded_bits": int(candidate["encoded_bits"]),
            "independent_oracle_encoded_bits": independent_oracle_bits,
            "rate_regret_bits": int(candidate["encoded_bits"]) - independent_oracle_bits,
            "hard_checks": checks,
            "hard_checks_pass": passes,
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
            "rate_regret_bits": total_bits - oracle_bits,
            "confidence_brier": round(brier_sum / max(1, count), 8),
            "by_unit_kind": {
                kind: dict(counts) for kind, counts in sorted(kind_counts.items())
            },
        },
        "training_labels_produced": 0,
        "public_or_hidden_target_used": False,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    result["receipt_sha256"] = digest(result)
    return result
