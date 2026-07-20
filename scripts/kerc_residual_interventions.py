#!/usr/bin/env python3
"""Causal per-unit downgrade targets for the KERC residual allocator.

K2 provides concrete payload candidates and exact codec costs.  This module adds
the K3 target-production half of the contract: execute every candidate for one
unit while all other units remain fixed, independently measure typed effects,
preserve uncertainty, enforce infinite-cost constraints, and choose the lowest
rate admissible action.  It does not evaluate a learned allocator and therefore
cannot establish K3 utility by itself.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import re
from typing import Any, Iterable

from kerc_residual_economics import (
    FIDELITY_ORDER,
    ResidualEconomicsFault,
    digest,
    digest_bytes,
    project_residual_unit_payload,
    residual_unit_sources,
    residual_wire_bytes,
)


POLICY = "project_theseus_kerc_unit_intervention_targets_v1"
SCHEMA_VERSION = "KERC-IT-2.0"
TARGET_PRODUCER_ID = "kerc_typed_semantic_consequence_target_producer_v2"
DIMENSIONS = (
    "semantic_proposition",
    "entity_identity",
    "value_unit_precision",
    "scope",
    "polarity",
    "modality",
    "temporal",
    "causal",
    "attribution",
    "quote",
    "terminology",
    "style",
    "byte",
)
UNIT_KINDS = (
    "interaction_entry",
    "segment_frame",
    "token_residue",
    "concept_realization",
    "exact_object",
)

_FIELD_DIMENSIONS = {
    "semantic_proposition": {
        "claim", "concept", "edge", "frame", "frame_name", "frame_roles",
        "operator", "predicate", "relation", "role", "tag", "target",
    },
    "entity_identity": {
        "actor_id", "alias", "aliases", "entity", "handle", "identity",
        "object_type", "stable_identity",
    },
    "value_unit_precision": {
        "amount", "currency", "decimal", "number", "precision", "quantity",
        "unit", "units", "value",
    },
    "scope": {
        "byte_end", "byte_start", "character_end", "character_start", "scope",
        "source_span", "target_spans",
    },
    "polarity": {"negated", "negation", "polarity"},
    "modality": {
        "access_policy", "authority", "locked", "modal", "modality",
        "permission", "required",
    },
    "temporal": {
        "after", "before", "date", "duration", "expiry", "temporal", "time",
        "timestamp",
    },
    "causal": {"cause", "causal", "condition", "purpose", "reason", "result"},
    "attribution": {
        "actor_id", "agent", "attitude", "attribution", "author", "source",
        "speaker",
    },
    "quote": {
        "code", "copy_policy", "exactness", "formula", "inline_bytes_b64",
        "quote",
    },
    "terminology": {
        "alias", "aliases", "lexical_unit", "morphology", "realization",
        "stable_identity", "tag", "term", "terminology",
    },
    "style": {
        "capitalization", "dialect", "formatting", "language", "morphology",
        "register", "style",
    },
    "byte": {"byte_end", "byte_start", "content_ref", "encoding", "inline_bytes_b64"},
}
_NAME_PARTS = re.compile(r"[^a-z0-9]+")
_SEMANTIC_STOPWORDS = {
    "affirmed", "asserted", "confidence", "derivation", "none", "preserved",
    "required", "source", "source_authored", "supported", "unverified",
}
_SAFETY_PATH_PARTS = {
    "access", "access_policy", "authority", "credential", "permission", "privacy",
    "secret", "security",
}


class InterventionTargetFault(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    def __reduce__(self) -> tuple[Any, tuple[str, str]]:
        return self.__class__, (self.code, self.detail)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _field_tokens(path: tuple[str, ...]) -> set[str]:
    result: set[str] = set()
    for component in path:
        result.update(part for part in _NAME_PARTS.split(component.lower()) if part)
        result.add(component.lower())
    return result


def _leaves(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        return [
            row
            for key in sorted(value)
            for row in _leaves(value[key], (*path, str(key)))
        ]
    if isinstance(value, (list, tuple)):
        return [
            row
            for index, item in enumerate(value)
            for row in _leaves(item, (*path, str(index)))
        ]
    return [(path, value)]


def _all_dimension_signatures(value: Any) -> dict[str, dict[str, Any]]:
    rows_by_dimension: dict[str, list[list[Any]]] = {
        dimension: [] for dimension in DIMENSIONS
    }
    for path, leaf in _leaves(value):
        tokens = _field_tokens(path)
        row = [list(path), type(leaf).__name__, copy.deepcopy(leaf)]
        for dimension, fields in _FIELD_DIMENSIONS.items():
            if tokens & fields:
                rows_by_dimension[dimension].append(row)
    return {
        dimension: {
            "dimension": dimension,
            "rows": rows,
            "applicable": bool(rows),
            "signature_sha256": digest(rows),
        }
        for dimension, rows in rows_by_dimension.items()
    }


def _normalized_semantic_values(value: Any) -> set[str]:
    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {f"number:{value}"}
    raw = str(value).strip()
    if not raw or raw.startswith("sha256:"):
        return set()
    lowered = raw.casefold()
    parts = [part for part in _NAME_PARTS.split(lowered) if len(part) >= 2]
    values = {part for part in parts if part not in _SEMANTIC_STOPWORDS}
    if parts:
        values.add(".".join(parts))
        for width in range(2, min(5, len(parts) + 1)):
            values.add(".".join(parts[-width:]))
    return {item for item in values if item and item not in _SEMANTIC_STOPWORDS}


def _reference_semantic_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        value_type = str(value.get("type") or "")
        if value_type == "byte_literal" and isinstance(value.get("value"), str):
            try:
                decoded = base64.b64decode(value["value"], validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            values.update(_normalized_semantic_values(decoded))
        for key, child in value.items():
            if str(key).endswith("_sha256") or str(key) in {
                "confidence", "program_sha256", "answer_packet_sha256",
            }:
                continue
            values.update(_normalized_semantic_values(key))
            values.update(_reference_semantic_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.update(_reference_semantic_values(child))
    else:
        values.update(_normalized_semantic_values(value))
    return values


def _payload_semantic_atoms(value: Any) -> set[tuple[str, str]]:
    atoms: set[tuple[str, str]] = set()
    for path, leaf in _leaves(value):
        if any(part.endswith("_sha256") for part in path):
            continue
        path_tokens = _field_tokens(path)
        dimensions = [
            dimension
            for dimension, fields in _FIELD_DIMENSIONS.items()
            if path_tokens & fields
        ]
        for dimension in dimensions:
            atoms.update(
                (dimension, normalized)
                for normalized in _normalized_semantic_values(leaf)
            )
    return atoms


def _target_obligations(
    source_payload: Any,
    *,
    unit_kind: str,
    kernel_program: dict[str, Any],
    answer_packet: dict[str, Any],
    surface_target: str,
    other_source_payloads: Iterable[Any],
) -> set[tuple[str, str]]:
    reference = _reference_semantic_values(
        {
            "kernel_program": kernel_program,
            "answer_packet": answer_packet,
            "surface_target": surface_target,
        }
    )
    other_values = {
        atom[1]
        for payload in other_source_payloads
        for atom in _payload_semantic_atoms(payload)
    }
    obligations = {
        atom
        for atom in _payload_semantic_atoms(source_payload)
        if atom[1] in reference and atom[1] not in other_values
    }
    if unit_kind == "concept_realization":
        obligations.update(
            atom
            for atom in _payload_semantic_atoms(source_payload)
            if atom[0] == "entity_identity" and atom[1] not in other_values
        )
    return obligations


def _requires_exact_safety(unit: dict[str, Any]) -> bool:
    if str(unit.get("unit_kind") or "") == "exact_object":
        return True
    if str(unit.get("unit_kind") or "") != "interaction_entry":
        return False
    path_parts = set(_NAME_PARTS.split(str(unit.get("source_path") or "").casefold()))
    return bool(path_parts & _SAFETY_PATH_PARTS)


def _typed_effect(
    source_signature: dict[str, Any],
    candidate_signature: dict[str, Any],
    dimension: str,
) -> dict[str, Any]:
    if not source_signature["applicable"]:
        return {
            "dimension": dimension,
            "state": "NOT_APPLICABLE",
            "loss": None,
            "source_signature_sha256": source_signature["signature_sha256"],
            "candidate_signature_sha256": candidate_signature["signature_sha256"],
        }
    source_rows = {_canonical(row): row for row in source_signature["rows"]}
    candidate_rows = {_canonical(row): row for row in candidate_signature["rows"]}
    retained = len(set(source_rows) & set(candidate_rows))
    loss = (len(source_rows) - retained) / max(1, len(source_rows))
    return {
        "dimension": dimension,
        "state": "PRESERVED" if loss == 0.0 else "CHANGED",
        "loss": round(loss, 8),
        "source_fact_count": len(source_rows),
        "retained_fact_count": retained,
        "source_signature_sha256": source_signature["signature_sha256"],
        "candidate_signature_sha256": candidate_signature["signature_sha256"],
    }


def _executable_checks(unit_kind: str, source: Any, candidate: Any) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    serializable = True
    try:
        residual_wire_bytes(candidate)
    except (ResidualEconomicsFault, TypeError, ValueError):
        serializable = False
    checks.append({"check": "typed_wire_serializable", "state": "PASS" if serializable else "FAIL"})
    if unit_kind == "token_residue" and candidate is not None:
        row = candidate if isinstance(candidate, dict) else {}
        span = row.get("source_span")
        valid = (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            and 0 <= span[0] <= span[1]
        )
        checks.append({"check": "token_source_span_well_formed", "state": "PASS" if valid else "FAIL"})
    elif unit_kind == "segment_frame" and candidate is not None:
        row = candidate if isinstance(candidate, dict) else {}
        frames = row.get("frames") if isinstance(row.get("frames"), list) else [row]
        valid = all(isinstance(frame, dict) for frame in frames)
        checks.append({"check": "segment_frame_typed", "state": "PASS" if valid else "FAIL"})
    elif unit_kind == "concept_realization" and candidate is not None:
        checks.append(
            {
                "check": "concept_projection_typed",
                "state": "PASS" if isinstance(candidate, dict) else "FAIL",
            }
        )
    elif unit_kind == "exact_object" and candidate is not None:
        valid = residual_wire_bytes(source) == residual_wire_bytes(candidate)
        checks.append({"check": "protected_object_exact_roundtrip", "state": "PASS" if valid else "FAIL"})
    elif unit_kind == "interaction_entry" and candidate is not None:
        checks.append({"check": "interaction_delta_replayable", "state": "PASS" if serializable else "FAIL"})
    return checks


def _observed_dimensions(
    source_signatures: dict[str, dict[str, Any]]
) -> list[str]:
    return [
        dimension
        for dimension in DIMENSIONS
        if source_signatures[dimension]["applicable"]
    ]


def _candidate_intervention(
    unit: dict[str, Any],
    source_payload: Any,
    candidate: dict[str, Any],
    *,
    source_signatures: dict[str, dict[str, Any]],
    observed_dimensions: list[str],
    target_obligations: set[tuple[str, str]],
    exact_safety_required: bool,
) -> dict[str, Any]:
    fidelity = str(candidate["fidelity"])
    projected = project_residual_unit_payload(
        source_payload,
        fidelity=fidelity,
        unit_kind=str(unit["unit_kind"]),
    )
    observed_payload_sha256 = digest_bytes(residual_wire_bytes(projected))
    if observed_payload_sha256 != candidate.get("payload_sha256"):
        raise InterventionTargetFault(
            "KERC_INTERVENTION_CANDIDATE_PAYLOAD_MISMATCH",
            f"{unit['unit_id']}:{fidelity}",
        )
    candidate_signatures = _all_dimension_signatures(projected)
    effects = [
        _typed_effect(
            source_signatures[dimension], candidate_signatures[dimension], dimension
        )
        for dimension in DIMENSIONS
    ]
    hard_reasons = []
    if candidate.get("hard_blocked") is True:
        hard_reasons.append("k2_source_bound_minimum")
    executable = _executable_checks(str(unit["unit_kind"]), source_payload, projected)
    hard_reasons.extend(
        f"executable_check_failed:{row['check']}"
        for row in executable
        if row["state"] == "FAIL"
    )
    measured = [row for row in effects if row["loss"] is not None]
    mean_loss = sum(float(row["loss"]) for row in measured) / max(1, len(measured))
    missing_count = len(DIMENSIONS) - len(measured)
    exact = residual_wire_bytes(source_payload) == residual_wire_bytes(projected)
    if exact_safety_required and not exact:
        hard_reasons.append("exact_safety_constraint")
    candidate_atoms = _payload_semantic_atoms(projected)
    retained_obligations = target_obligations & candidate_atoms
    obligation_loss = (
        (len(target_obligations) - len(retained_obligations))
        / max(1, len(target_obligations))
    )
    # Missing semantic dimensions widen uncertainty; they are never silently zero.
    uncertainty_upper = 0.0 if exact else min(1.0, mean_loss + missing_count / len(DIMENSIONS))
    result = {
        "fidelity": fidelity,
        "fidelity_index": FIDELITY_ORDER.index(fidelity),
        "source_payload_sha256": digest_bytes(residual_wire_bytes(source_payload)),
        "intervened_payload_sha256": observed_payload_sha256,
        "encoded_bits": int(candidate["encoded_bits"]),
        "typed_effects": effects,
        "observed_dimensions": observed_dimensions,
        "target_obligation_count": len(target_obligations),
        "retained_target_obligation_count": len(retained_obligations),
        "target_obligation_loss": round(obligation_loss, 8),
        "target_obligation_sha256": digest(sorted(target_obligations)),
        "executable_checks": executable,
        "mean_measured_loss": round(mean_loss, 8),
        "measured_dimension_count": len(measured),
        "missing_dimension_count": missing_count,
        "distortion_interval": {
            "lower": round(mean_loss, 8),
            "upper": round(uncertainty_upper, 8),
        },
        "hard_constraint_cost": "INFINITY" if hard_reasons else 0,
        "hard_constraint_reasons": sorted(set(hard_reasons)),
        "exact_payload": exact,
        "public_or_hidden_target_used": False,
        "evaluator_only_answer_used": False,
    }
    result["intervention_sha256"] = digest(result)
    return result


def build_unit_intervention_targets(
    *,
    unit_packet: dict[str, Any],
    source_record_sha256: str,
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    concept_capsules: dict[str, Any],
    exact_objects: dict[str, Any],
    source_family: str,
    kernel_program: dict[str, Any],
    answer_packet: dict[str, Any],
    surface_target: str,
) -> dict[str, Any]:
    """Execute and measure all one-unit fidelity interventions for one record."""

    if unit_packet.get("source_record_sha256") != source_record_sha256:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_SOURCE_IDENTITY_MISMATCH", source_record_sha256
        )
    sources = residual_unit_sources(
        source_record_sha256=source_record_sha256,
        global_state=global_state,
        segment_residual=segment_residual,
        token_residuals=token_residuals,
        concept_capsules=concept_capsules,
        exact_objects=exact_objects,
    )
    source_by_id = {str(row["unit_id"]): row for row in sources}
    if list(source_by_id) != [str(row["unit_id"]) for row in unit_packet.get("units") or []]:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_UNIT_INVENTORY_MISMATCH", source_record_sha256
        )
    targets = []
    authoritative = 0
    source_dependent = 0
    class_counts: dict[str, int] = {str(index): 0 for index in range(len(FIDELITY_ORDER))}
    for unit in unit_packet.get("units") or []:
        source = source_by_id[str(unit["unit_id"])]["source_payload"]
        source_signatures = _all_dimension_signatures(source)
        observed_dimensions = _observed_dimensions(source_signatures)
        target_obligations = _target_obligations(
            source,
            unit_kind=str(unit["unit_kind"]),
            kernel_program=kernel_program,
            answer_packet=answer_packet,
            surface_target=surface_target,
            other_source_payloads=(
                row["source_payload"]
                for unit_id, row in source_by_id.items()
                if unit_id != str(unit["unit_id"])
            ),
        )
        exact_safety_required = _requires_exact_safety(unit)
        interventions = [
            _candidate_intervention(
                unit,
                source,
                candidate,
                source_signatures=source_signatures,
                observed_dimensions=observed_dimensions,
                target_obligations=target_obligations,
                exact_safety_required=exact_safety_required,
            )
            for candidate in unit.get("candidates") or []
        ]
        admissible = [
            row for row in interventions if row["hard_constraint_cost"] != "INFINITY"
        ]
        if not admissible:
            raise InterventionTargetFault(
                "KERC_INTERVENTION_NO_ADMISSIBLE_ACTION", str(unit["unit_id"])
            )
        consequence_preserving = [
            row for row in admissible if float(row["target_obligation_loss"]) == 0.0
        ]
        if target_obligations and not consequence_preserving:
            raise InterventionTargetFault(
                "KERC_INTERVENTION_NO_SEMANTICALLY_PRESERVING_ACTION",
                str(unit["unit_id"]),
            )
        selected = min(
            consequence_preserving or admissible,
            key=lambda row: (int(row["encoded_bits"]), int(row["fidelity_index"])),
        )
        determinate = bool(target_obligations) or exact_safety_required
        authority = (
            "source_dependent_semantic_consequence"
            if target_obligations
            else "hard_safety_exact"
            if exact_safety_required
            else "uncertain_target_withheld"
        )
        authoritative += int(determinate)
        cheapest_hard_safe = min(
            admissible,
            key=lambda row: (int(row["encoded_bits"]), int(row["fidelity_index"])),
        )
        source_dependent += int(
            determinate
            and int(selected["fidelity_index"]) != int(cheapest_hard_safe["fidelity_index"])
        )
        if determinate:
            class_counts[str(selected["fidelity_index"])] += 1
        row = {
            "unit_id": str(unit["unit_id"]),
            "unit_kind": str(unit["unit_kind"]),
            "source_path": str(unit["source_path"]),
            "source_payload_sha256": digest_bytes(residual_wire_bytes(source)),
            "source_payload_wire_b64": base64.b64encode(
                residual_wire_bytes(source)
            ).decode("ascii"),
            "source_visible_candidates": [
                {
                    "fidelity_index": index,
                    "encoded_bits": int(candidate["encoded_bits"]),
                    "uncompressed_bits": int(candidate["uncompressed_bits"]),
                    "structural_loss": float(candidate["structural_loss"]),
                    "distortion_vector": copy.deepcopy(candidate["distortion_vector"]),
                    "k2_hard_blocked": bool(candidate["hard_blocked"]),
                    "payload_sha256": str(candidate["payload_sha256"]),
                }
                for index, candidate in enumerate(unit.get("candidates") or [])
            ],
            "maximum_structural_distortion": float(
                unit.get("maximum_structural_distortion") or 0.0
            ),
            "interventions": interventions,
            "selected_fidelity": selected["fidelity"],
            "selected_fidelity_index": selected["fidelity_index"],
            "selected_encoded_bits": selected["encoded_bits"],
            "target_authority": authority,
            "allocator_loss_enabled": determinate,
            "semantic_obligation_count": len(target_obligations),
            "semantic_obligation_sha256": digest(sorted(target_obligations)),
            "source_dependent_beyond_constrained_rate": (
                int(selected["fidelity_index"])
                != int(cheapest_hard_safe["fidelity_index"])
            ),
            "confidence_target": 1.0 if determinate else 0.0,
        }
        row["target_sha256"] = digest(row)
        targets.append(row)
    fold = int(hashlib.sha256(source_family.encode("utf-8")).hexdigest()[:8], 16) % 5
    result = {
        "policy": POLICY,
        "schema_version": SCHEMA_VERSION,
        "target_producer_id": TARGET_PRODUCER_ID,
        "source_record_sha256": source_record_sha256,
        "source_family": source_family,
        "cross_fit_fold": fold,
        "unit_packet_sha256": str(unit_packet["packet_sha256"]),
        "intervention_contract": "one_unit_changed_all_other_units_fixed",
        "candidate_payloads_executed": True,
        "dimensions": list(DIMENSIONS),
        "hard_constraint_value": "INFINITY",
        "targets": targets,
        "summary": {
            "unit_count": len(targets),
            "candidate_intervention_count": sum(len(row["interventions"]) for row in targets),
            "allocator_authority_unit_count": authoritative,
            "withheld_uncertain_unit_count": len(targets) - authoritative,
            "source_dependent_decision_count": source_dependent,
            "authoritative_class_counts": class_counts,
        },
        "target_producer_is_final_evaluator": False,
        "target_inputs": [
            "kernel_program",
            "answer_packet",
            "surface_target",
            "source_bound_unit",
        ],
        "surface_target_visible_to_model": False,
        "source_annotation_used": False,
        "human_gold_absence_preserved_as_uncertainty": True,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    result["receipt_sha256"] = digest(result)
    return result


def validate_unit_intervention_targets(
    receipt: dict[str, Any],
    *,
    unit_packet: dict[str, Any],
    source_record_sha256: str,
    global_state: dict[str, Any],
    segment_residual: dict[str, Any],
    token_residuals: list[dict[str, Any]],
    concept_capsules: dict[str, Any],
    exact_objects: dict[str, Any],
    source_family: str,
    kernel_program: dict[str, Any],
    answer_packet: dict[str, Any],
    surface_target: str,
) -> dict[str, Any]:
    expected = build_unit_intervention_targets(
        unit_packet=unit_packet,
        source_record_sha256=source_record_sha256,
        global_state=global_state,
        segment_residual=segment_residual,
        token_residuals=token_residuals,
        concept_capsules=concept_capsules,
        exact_objects=exact_objects,
        source_family=source_family,
        kernel_program=kernel_program,
        answer_packet=answer_packet,
        surface_target=surface_target,
    )
    if receipt != expected:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_TARGET_RECEIPT_INVALID", digest(receipt)
        )
    return expected


def iter_authoritative_unit_examples(receipt: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if receipt.get("policy") != POLICY or receipt.get("schema_version") != SCHEMA_VERSION:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_TARGET_SCHEMA_INVALID", str(receipt.get("schema_version"))
        )
    for target in receipt.get("targets") or []:
        if target.get("allocator_loss_enabled") is True:
            yield target


def compact_allocator_targets(receipt: dict[str, Any]) -> dict[str, Any]:
    """Project the auditable intervention receipt into bounded model supervision."""

    if receipt.get("policy") != POLICY or receipt.get("schema_version") != SCHEMA_VERSION:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_TARGET_SCHEMA_INVALID", str(receipt.get("schema_version"))
        )
    targets = []
    for row in receipt.get("targets") or []:
        candidates = []
        for candidate in row.get("interventions") or []:
            candidates.append(
                {
                    "fidelity_index": int(candidate["fidelity_index"]),
                    "encoded_bits": int(candidate["encoded_bits"]),
                    "hard_blocked": candidate["hard_constraint_cost"] == "INFINITY",
                }
            )
        target = {
            "unit_id": str(row["unit_id"]),
            "unit_kind": str(row["unit_kind"]),
            "source_path": str(row["source_path"]),
            "source_payload_sha256": str(row["source_payload_sha256"]),
            "source_payload_wire_b64": str(row["source_payload_wire_b64"]),
            "source_visible_candidates": copy.deepcopy(
                row["source_visible_candidates"]
            ),
            "maximum_structural_distortion": float(
                row["maximum_structural_distortion"]
            ),
            "candidates": candidates,
            "selected_fidelity_index": int(row["selected_fidelity_index"]),
            "confidence_target": float(row["confidence_target"]),
            "allocator_loss_enabled": bool(row["allocator_loss_enabled"]),
            "target_authority": str(row["target_authority"]),
        }
        target["compact_target_sha256"] = digest(target)
        targets.append(target)
    result = {
        "policy": "project_theseus_kerc_compact_unit_allocator_targets_v1",
        "source_intervention_receipt_sha256": str(receipt["receipt_sha256"]),
        "source_record_sha256": str(receipt["source_record_sha256"]),
        "source_family": str(receipt["source_family"]),
        "cross_fit_fold": int(receipt["cross_fit_fold"]),
        "dimensions": list(DIMENSIONS),
        "targets": targets,
        "summary": copy.deepcopy(receipt["summary"]),
        "target_producer_is_final_evaluator": False,
        "target_producer_id": str(receipt["target_producer_id"]),
        "answer_packet_visible_to_model": False,
        "surface_target_visible_to_model": False,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    result["receipt_sha256"] = digest(result)
    return result


def compact_allocator_targets_for_record(record: dict[str, Any]) -> dict[str, Any]:
    """Derive K3 supervision from one canonical, already-verified K2 record.

    The governed source record and its semantic-verification identity stay
    unchanged.  This projection is attached only to the compiled training view
    after corpus admission, which lets an immutable K2 corpus migrate to K3
    without reparsing the licensed source material.
    """

    packet = (
        record.get("kernel_packet")
        if isinstance(record.get("kernel_packet"), dict)
        else {}
    )
    residual = (
        packet.get("residual") if isinstance(packet.get("residual"), dict) else {}
    )
    hrl_state = (
        record.get("hrl_state") if isinstance(record.get("hrl_state"), dict) else {}
    )
    provenance = (
        record.get("provenance")
        if isinstance(record.get("provenance"), dict)
        else {}
    )
    unit_packet = (
        residual.get("unit_packet")
        if isinstance(residual.get("unit_packet"), dict)
        else {}
    )
    source_record_sha256 = str(unit_packet.get("source_record_sha256") or "")
    source_family = str(provenance.get("source_group") or "")
    if not source_record_sha256 or not source_family:
        raise InterventionTargetFault(
            "KERC_INTERVENTION_RECORD_BINDING_INCOMPLETE",
            str(record.get("record_sha256") or source_record_sha256),
        )
    receipt = build_unit_intervention_targets(
        unit_packet=unit_packet,
        source_record_sha256=source_record_sha256,
        global_state=(hrl_state.get("global") or {}),
        segment_residual=(residual.get("segment_frame") or {}),
        token_residuals=list(residual.get("token_tags") or []),
        concept_capsules=(packet.get("concept_capsules") or {}),
        exact_objects=(packet.get("protected_objects") or {}),
        source_family=source_family,
        kernel_program=(packet.get("program") or {}),
        answer_packet=(record.get("answer_packet") or {}),
        surface_target=str(record.get("surface_target") or ""),
    )
    return compact_allocator_targets(receipt)


def validate_compact_allocator_targets(
    compact: dict[str, Any], *, unit_packet: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on compact-target tampering without trusting model-visible data."""

    if compact.get("policy") != "project_theseus_kerc_compact_unit_allocator_targets_v1":
        raise InterventionTargetFault(
            "KERC_COMPACT_ALLOCATOR_TARGET_POLICY_INVALID", str(compact.get("policy"))
        )
    core = {key: copy.deepcopy(value) for key, value in compact.items() if key != "receipt_sha256"}
    if compact.get("receipt_sha256") != digest(core):
        raise InterventionTargetFault(
            "KERC_COMPACT_ALLOCATOR_TARGET_RECEIPT_INVALID", str(compact.get("receipt_sha256"))
        )
    units = list(unit_packet.get("units") or [])
    targets = list(compact.get("targets") or [])
    if compact.get("source_record_sha256") != unit_packet.get("source_record_sha256"):
        raise InterventionTargetFault(
            "KERC_COMPACT_ALLOCATOR_TARGET_SOURCE_INVALID", str(compact.get("source_record_sha256"))
        )
    if [row.get("unit_id") for row in targets] != [row.get("unit_id") for row in units]:
        raise InterventionTargetFault(
            "KERC_COMPACT_ALLOCATOR_TARGET_UNITS_INVALID", str(len(targets))
        )
    for unit, target in zip(units, targets):
        target_core = {
            key: copy.deepcopy(value)
            for key, value in target.items()
            if key != "compact_target_sha256"
        }
        if target.get("compact_target_sha256") != digest(target_core):
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_UNIT_TARGET_INVALID", str(target.get("unit_id"))
            )
        candidates = list(target.get("candidates") or [])
        source_visible = list(target.get("source_visible_candidates") or [])
        if len(candidates) != len(FIDELITY_ORDER) or [
            int(row.get("fidelity_index", -1)) for row in candidates
        ] != list(range(len(FIDELITY_ORDER))):
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_CANDIDATES_INVALID", str(target.get("unit_id"))
            )
        expected_source_visible = [
            {
                "fidelity_index": index,
                "encoded_bits": int(candidate["encoded_bits"]),
                "uncompressed_bits": int(candidate["uncompressed_bits"]),
                "structural_loss": float(candidate["structural_loss"]),
                "distortion_vector": copy.deepcopy(candidate["distortion_vector"]),
                "k2_hard_blocked": bool(candidate["hard_blocked"]),
                "payload_sha256": str(candidate["payload_sha256"]),
            }
            for index, candidate in enumerate(unit.get("candidates") or [])
        ]
        if (
            source_visible != expected_source_visible
            or float(target.get("maximum_structural_distortion", -1.0))
            != float(unit.get("maximum_structural_distortion") or 0.0)
        ):
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_SOURCE_FEATURES_INVALID",
                str(target.get("unit_id")),
            )
        selected = int(target.get("selected_fidelity_index", -1))
        if selected not in range(len(FIDELITY_ORDER)) or candidates[selected]["hard_blocked"]:
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_SELECTION_INVALID", str(target.get("unit_id"))
            )
        if target.get("source_payload_sha256") != unit.get("source_residual", {}).get("payload_sha256"):
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_PAYLOAD_INVALID", str(target.get("unit_id"))
            )
        try:
            source_wire = base64.b64decode(
                str(target.get("source_payload_wire_b64") or ""), validate=True
            )
        except ValueError as exc:
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_PAYLOAD_ENCODING_INVALID",
                str(target.get("unit_id")),
            ) from exc
        if digest_bytes(source_wire) != target.get("source_payload_sha256"):
            raise InterventionTargetFault(
                "KERC_COMPACT_ALLOCATOR_PAYLOAD_HASH_INVALID", str(target.get("unit_id"))
            )
    return copy.deepcopy(compact)
