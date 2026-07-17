#!/usr/bin/env python3
"""Versioned Kernel English packet protocol under the Semantic-IR/VCM owners.

This module is exact substrate, not a learned language model. It captures source
identity, protects form-sensitive objects before correction, validates a typed
Kernel program, serializes the V_K/V_P code spaces, binds VCM-owned residual
state, and verifies structured answer constraints. Surface-to-Kernel inference,
surface rendering, and semantic equivalence remain learned/evaluated components.

No deterministic path in this module receives generation credit. Unknown state,
invalid handles, malformed scope, and round-trip mismatches fail closed without
templates, literal renderers, or best-effort interpretation.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


POLICY = "project_theseus_kernel_packet_protocol_v1"
KERNEL_VERSION = "KE-1.0"
PACKET_VERSION = "KPP-1.0"
HRL_VERSION = "HRL-1.0"
CODEBOOK_VERSION = "KE-CODEBOOK-1.0"
CODE_SPACES = ("V_S", "V_K", "V_P")
RESIDUAL_MODES = {"SOURCE_RECONSTRUCTION", "OUTPUT_REALIZATION"}
FIDELITY_MODES = {"semantic", "faithful", "lexical", "exact"}
DERIVATIONS = {
    "preserved",
    "high_confidence_correction",
    "low_confidence_hypothesis",
    "compiler_inference",
    "learned_reasoner",
    "tool_evidence",
}
MODALITIES = {
    "ASSERTED",
    "POSSIBLE",
    "PROBABLE",
    "HYPOTHESIS",
    "REQUIRED",
    "PERMITTED",
    "FORBIDDEN",
    "UNKNOWN",
}
POLARITIES = {"AFFIRMED", "NEGATED", "UNKNOWN"}
QUANTIFIERS = {"NONE", "EXISTS", "FORALL", "EXACT", "AT_LEAST", "AT_MOST", "UNKNOWN"}
VALUE_TYPES = {"handle", "concept", "number", "node_ref", "list", "ambiguity", "byte_literal", "boolean", "null"}
HANDLE_PREFIX_BY_TYPE = {
    "ENTITY": "E",
    "PERSON": "E",
    "ORGANIZATION": "E",
    "PLACE": "E",
    "PRODUCT": "E",
    "QUOTE": "Q",
    "NUMBER": "N",
    "MONEY": "N",
    "DATE_TIME": "N",
    "URL": "D",
    "EMAIL": "D",
    "FILE_PATH": "D",
    "HASH": "D",
    "CODE": "K",
    "FORMULA": "X",
    "MARKUP": "X",
    "EXACT_TEXT": "X",
}
COPY_POLICIES = {"EXACT", "VALUE_AND_STYLE", "NORMALIZED_VALUE", "REFERENCE_ONLY"}
NO_CHEAT = {
    "candidate_generation_credit": 0,
    "deterministic_compiler_credit": 0,
    "renderer_credit": 0,
    "public_training_rows_written": 0,
    "external_inference_calls": 0,
    "fallback_return_count": 0,
}
TRAINING_RECORD_POLICY = "project_theseus_kernel_english_training_record_v1"
TRAINING_VIEW_POLICY = "project_theseus_kernel_english_training_view_v1"
TRAINING_CONTRACT_POLICY = "project_theseus_kernel_english_learned_pipeline_v1"
TRAINING_VERIFICATION_POLICY = "project_theseus_kernel_english_verification_receipt_v1"
TRAINING_DISPOSITION_POLICY = "project_theseus_kerc_pretraining_disposition_v1"
SEMANTIC_SUPERVISION_POLICY = "project_theseus_kerc_semantic_supervision_evidence_v1"
TRAINING_SPLITS = {"private_train", "private_dev", "private_eval"}
TRAINING_OBJECTIVES = (
    "surface_direct_control_v1",
    "surface_to_kernel_program_v1",
    "kernel_program_to_answer_packet_v1",
    "answer_packet_to_surface_v1",
)
TRAINING_TASK_TAGS = {
    "surface_direct_control_v1": "<KERC_TASK_SURFACE_DIRECT>",
    "surface_to_kernel_program_v1": "<KERC_TASK_SURFACE_TO_KERNEL>",
    "kernel_program_to_answer_packet_v1": "<KERC_TASK_KERNEL_TO_ANSWER>",
    "answer_packet_to_surface_v1": "<KERC_TASK_ANSWER_TO_SURFACE>",
}
KERC_RESIDUAL_CHANNELS = ("interaction", "segment", "token", "exact")
KERC_FIDELITY_LABELS = {"semantic": 0, "faithful": 1, "lexical": 2, "exact": 3}
KERC_VERIFIER_DIMENSIONS = (
    "semantic_consistency",
    "protected_object_consistency",
    "numeric_value_consistency",
    "surface_fidelity",
)
SEMANTIC_EVIDENCE_TIERS = {
    "audited_human_gold": {
        "producer_kind": "human_annotation",
        "claim_authority": "decision_grade_reference",
        "model_derived": False,
        "allowed_splits": TRAINING_SPLITS,
        "maximum_optimizer_sampling_weight": 1.0,
        "verification_methods": {"human_dual_review"},
    },
    "licensed_human_semantic_gold": {
        "producer_kind": "licensed_semantic_dataset",
        "claim_authority": "decision_grade_reference",
        "model_derived": False,
        "allowed_splits": TRAINING_SPLITS,
        "maximum_optimizer_sampling_weight": 1.0,
        "verification_methods": {
            "licensed_semantic_dataset_plus_independent_schema_review"
        },
    },
    "local_parser_silver": {
        "producer_kind": "local_semantic_parser",
        "claim_authority": "training_only_silver",
        "model_derived": True,
        "allowed_splits": {"private_train"},
        "maximum_optimizer_sampling_weight": 0.25,
        "verification_methods": {"local_parser_plus_independent_schema_review"},
    },
    "governed_openai_residual": {
        "producer_kind": "governed_openai_teacher",
        "claim_authority": "residual_training_only",
        "model_derived": True,
        "allowed_splits": {"private_train"},
        "maximum_optimizer_sampling_weight": 0.02,
        "verification_methods": {"governed_teacher_plus_independent_verifier"},
    },
}


class KernelProtocolFault(ValueError):
    """Typed protocol fault with fail-closed behavior."""

    def __init__(self, code: str, detail: str, *, path: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.path = path

    def record(self) -> dict[str, Any]:
        return {
            "fault_type": self.code,
            "detail": self.detail,
            "path": self.path,
            "failure_behavior": "reject_without_fallback",
        }


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    object_type: str
    copy_policy: str
    priority: int
    source: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, (bytes, bytearray)) else canonical_json(value).encode("utf-8")
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def kernel_training_contract() -> dict[str, Any]:
    """Return the checkpoint-shaping learned KERC contract.

    The exact protocol validates records and learned outputs. It does not infer a
    Kernel program, answer packet, or surface answer and therefore earns no model
    capability credit.
    """

    contract = {
        "policy": TRAINING_CONTRACT_POLICY,
        "record_policy": TRAINING_RECORD_POLICY,
        "view_policy": TRAINING_VIEW_POLICY,
        "packet_version": PACKET_VERSION,
        "kernel_version": KERNEL_VERSION,
        "hrl_version": HRL_VERSION,
        "codebook_version": CODEBOOK_VERSION,
        "objective_order": list(TRAINING_OBJECTIVES),
        "task_tags": dict(TRAINING_TASK_TAGS),
        "task_mode_transport": "trusted_internal_source_prefix_token",
        "raw_text_may_select_task_mode": False,
        "trusted_source_control_tokens": list(TRAINING_TASK_TAGS.values()),
        "residual_channels": list(KERC_RESIDUAL_CHANNELS),
        "residual_fidelity_labels": dict(KERC_FIDELITY_LABELS),
        "verifier_dimensions": list(KERC_VERIFIER_DIMENSIONS),
        "semantic_supervision": {
            "policy": SEMANTIC_SUPERVISION_POLICY,
            "tiers": {
                tier: {
                    "producer_kind": contract["producer_kind"],
                    "claim_authority": contract["claim_authority"],
                    "model_derived": contract["model_derived"],
                    "allowed_splits": sorted(contract["allowed_splits"]),
                    "maximum_optimizer_sampling_weight": contract[
                        "maximum_optimizer_sampling_weight"
                    ],
                }
                for tier, contract in SEMANTIC_EVIDENCE_TIERS.items()
            },
            "heldout_requires_decision_grade_reference": True,
            "public_semantic_benchmark_training_forbidden": True,
        },
        "verifier_training": {
            "positive_pairs": "independently_verified_governed_records",
            "negative_pairs": "deterministic_targeted_corruptions",
            "negative_generator_loss_enabled": False,
            "shared_generator_verifier_parameters": False,
        },
        "pipeline": [
            "deterministic_protect_visible_source",
            "learned_surface_to_kernel_program",
            "independent_kernel_program_validation",
            "learned_kernel_program_to_answer_packet",
            "independent_answer_packet_validation",
            "learned_answer_packet_to_surface",
            "learned_surface_recompile_and_roundtrip_verification",
        ],
        "derived_views_receive_unique_source_credit": False,
        "surface_direct_control_required": True,
        "failure_behavior": "reject_without_template_literal_tool_or_router_fallback",
        **NO_CHEAT,
    }
    contract["contract_sha256"] = stable_hash(contract)
    return contract


def validate_training_disposition(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate whether full KERC joins or is retired from the first campaign.

    A bounded negative result may remove the checkpoint-shaping KERC objective
    without erasing the exact-object and scoped-residual mechanisms that survived
    independently. The disposition is architecture selection, not a broad claim
    that KERC can never work.
    """

    required = cfg.get("required") is True
    if required:
        return {
            "state": "CANDIDATE_REQUIRED",
            "full_kerc_training_enabled": True,
            "retained_mechanisms": [],
        }

    disposition = cfg.get("disposition")
    if not isinstance(disposition, dict) or disposition.get("policy") != (
        TRAINING_DISPOSITION_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_MISSING",
            str(disposition),
            path="kernel_english_training.disposition",
        )
    expected_scalars = {
        "state": "RETIRED_FROM_FIRST_LONG_RUN",
        "retirement_scope": "full_compiler_core_renderer_training_path_only",
        "evidence_scope": "bounded_authored_synthetic_campaign",
        "broad_efficiency_gate_passed": False,
        "full_kerc_training_enabled": False,
        "general_kerc_falsification_claimed": False,
        "learned_capability_claimed": False,
    }
    for key, expected in expected_scalars.items():
        if disposition.get(key) != expected:
            raise KernelProtocolFault(
                "KERC_TRAINING_DISPOSITION_INVALID",
                f"{key}={disposition.get(key)!r}",
                path=f"kernel_english_training.disposition.{key}",
            )
    retained = tuple(disposition.get("retained_mechanisms") or ())
    if retained != (
        "protected_exact_object_path",
        "scoped_interaction_glossary_residual",
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_RETAINED_MECHANISMS_INVALID",
            canonical_json(retained),
            path="kernel_english_training.disposition.retained_mechanisms",
        )
    evidence = disposition.get("evidence")
    if not isinstance(evidence, dict):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_EVIDENCE_MISSING",
            str(evidence),
            path="kernel_english_training.disposition.evidence",
        )
    hashes = evidence.get("artifact_sha256")
    required_hashes = {
        "preregistration",
        "design",
        "corpus",
        "raw_run",
        "confirmatory_result",
        "design_validator",
        "result_validator",
    }
    if not isinstance(hashes, dict) or set(hashes) != required_hashes:
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_HASH_SET_INVALID",
            canonical_json(hashes),
            path="kernel_english_training.disposition.evidence.artifact_sha256",
        )
    for key, value in hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise KernelProtocolFault(
                "KERC_TRAINING_DISPOSITION_HASH_INVALID",
                f"{key}={value}",
                path=f"kernel_english_training.disposition.evidence.artifact_sha256.{key}",
            )
    denominators = evidence.get("denominators")
    if denominators != {
        "corpus": 192,
        "train": 128,
        "heldout": 64,
        "seeds": 5,
        "ablations": 13,
        "attacks": 20,
    }:
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_DENOMINATOR_INVALID",
            canonical_json(denominators),
            path="kernel_english_training.disposition.evidence.denominators",
        )
    measurements = evidence.get("measurements")
    if not isinstance(measurements, dict):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_MEASUREMENTS_MISSING",
            str(measurements),
            path="kernel_english_training.disposition.evidence.measurements",
        )
    if not (
        float(measurements.get("kernel_native_mean_accuracy", -1.0))
        == float(measurements.get("best_surface_mean_accuracy", -2.0))
        == float(measurements.get("simple_handle_mean_accuracy", -3.0))
        == 0.5
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_MATCHED_UTILITY_INVALID",
            canonical_json(measurements),
            path="kernel_english_training.disposition.evidence.measurements",
        )
    if not (
        float(measurements.get("packet_mean_bytes", 0.0))
        > float(measurements.get("best_simple_total_description_bytes", math.inf))
        > 0.0
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_COST_INVALID",
            canonical_json(measurements),
            path="kernel_english_training.disposition.evidence.measurements",
        )
    if int(measurements.get("attack_false_allow_count", -1)) != 1:
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_ATTACK_RESULT_INVALID",
            canonical_json(measurements),
            path="kernel_english_training.disposition.evidence.measurements.attack_false_allow_count",
        )
    return copy.deepcopy(disposition)


def validate_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one governed source/Kernel/answer/surface training record."""

    if not isinstance(record, dict) or record.get("policy") != TRAINING_RECORD_POLICY:
        raise KernelProtocolFault(
            "KERC_TRAINING_RECORD_POLICY_INVALID",
            str(record.get("policy") if isinstance(record, dict) else type(record)),
            path="record.policy",
        )
    split = str(record.get("split") or "")
    if split not in TRAINING_SPLITS:
        raise KernelProtocolFault("KERC_TRAINING_SPLIT_INVALID", split, path="record.split")
    if str(record.get("language") or "") != "en":
        raise KernelProtocolFault(
            "KERC_TRAINING_LANGUAGE_INVALID", str(record.get("language")), path="record.language"
        )
    source = str(record.get("source_text") or "")
    surface_target = str(record.get("surface_target") or "")
    if not source.strip() or not surface_target.strip():
        raise KernelProtocolFault(
            "KERC_TRAINING_TEXT_MISSING", "source and surface target are required", path="record"
        )
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    for key in ("source_id", "source_group", "license_spdx", "permitted_use"):
        if not str(provenance.get(key) or "").strip():
            raise KernelProtocolFault(
                "KERC_TRAINING_PROVENANCE_INCOMPLETE", key, path=f"record.provenance.{key}"
            )
    if provenance.get("permitted_use") != "model_training":
        raise KernelProtocolFault(
            "KERC_TRAINING_USE_NOT_PERMITTED",
            str(provenance.get("permitted_use")),
            path="record.provenance.permitted_use",
        )
    verification = (
        record.get("verification_receipt")
        if isinstance(record.get("verification_receipt"), dict)
        else {}
    )
    if verification.get("policy") != TRAINING_VERIFICATION_POLICY:
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFICATION_POLICY_INVALID",
            str(verification.get("policy")),
            path="record.verification_receipt.policy",
        )
    if verification.get("accepted") is not True or not str(
        verification.get("verifier_id") or ""
    ).strip():
        raise KernelProtocolFault(
            "KERC_TRAINING_RECORD_UNVERIFIED",
            canonical_json(verification),
            path="record.verification_receipt",
        )
    if not str(verification.get("receipt_id") or "").strip():
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFICATION_RECEIPT_ID_MISSING",
            "receipt_id",
            path="record.verification_receipt.receipt_id",
        )
    if verification.get("reviewer_independent_of_record_producer") is not True:
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFIER_NOT_INDEPENDENT",
            str(verification.get("verifier_id")),
            path="record.verification_receipt",
        )
    if verification.get("method") not in {
        "human_dual_review",
        "governed_teacher_plus_independent_verifier",
        "licensed_semantic_dataset_plus_independent_schema_review",
        "local_parser_plus_independent_schema_review",
    }:
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFICATION_METHOD_INVALID",
            str(verification.get("method")),
            path="record.verification_receipt.method",
        )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(verification.get("evidence_sha256") or "")):
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFICATION_EVIDENCE_INVALID",
            str(verification.get("evidence_sha256")),
            path="record.verification_receipt.evidence_sha256",
        )
    semantic_supervision = validate_semantic_supervision_evidence(
        record.get("semantic_supervision"),
        split=split,
        verification_method=str(verification.get("method") or ""),
    )
    for key in (
        "public_benchmark",
        "public_tests_included",
        "public_benchmark_solutions_included",
        "external_inference",
    ):
        if record.get(key) is not False:
            raise KernelProtocolFault(
                "KERC_TRAINING_BOUNDARY_INVALID", key, path=f"record.{key}"
            )

    residual_supervision = _validate_residual_supervision(record, packet=record.get("kernel_packet") or {})
    for key in (
        "fallback_return_count",
        "template_credit",
        "deterministic_renderer_credit",
        "candidate_generation_credit",
    ):
        if int(record.get(key) or 0):
            raise KernelProtocolFault(
                "KERC_TRAINING_CREDIT_BOUNDARY_INVALID", key, path=f"record.{key}"
            )

    hrl_state = record.get("hrl_state") if isinstance(record.get("hrl_state"), dict) else {}
    packet = record.get("kernel_packet") if isinstance(record.get("kernel_packet"), dict) else {}
    packet_replay = validate_kernel_packet(packet, local_hrl_state=hrl_state)
    expected_source = capture_source(source)
    if (packet.get("source") or {}).get("source_sha256") != expected_source["source_sha256"]:
        raise KernelProtocolFault(
            "KERC_TRAINING_SOURCE_PACKET_MISMATCH",
            str((packet.get("source") or {}).get("source_sha256")),
            path="record.kernel_packet.source",
        )
    _validate_packet_objects_against_source(
        source,
        packet.get("protected_objects")
        if isinstance(packet.get("protected_objects"), dict)
        else {},
    )
    answer = validate_answer_packet_against_context(
        record.get("answer_packet") if isinstance(record.get("answer_packet"), dict) else {},
        protected_objects=packet.get("protected_objects") or {},
        concept_capsules=packet.get("concept_capsules") or {},
    )
    semantic_payload_sha256 = training_semantic_payload_sha256(
        {
            **record,
            "answer_packet": answer,
        }
    )
    if verification.get("semantic_payload_sha256") != semantic_payload_sha256:
        raise KernelProtocolFault(
            "KERC_TRAINING_VERIFICATION_BINDING_MISMATCH",
            f"expected={semantic_payload_sha256} observed={verification.get('semantic_payload_sha256')}",
            path="record.verification_receipt.semantic_payload_sha256",
        )
    canonical = copy.deepcopy(record)
    canonical["semantic_supervision"] = semantic_supervision
    canonical["residual_supervision"] = residual_supervision
    canonical["answer_packet"] = answer
    canonical["raw_source_sha256"] = expected_source["source_sha256"]
    canonical["surface_target_sha256"] = stable_hash(surface_target.encode("utf-8"))
    canonical["packet_replay"] = packet_replay
    canonical["semantic_payload_sha256"] = semantic_payload_sha256
    canonical["record_sha256"] = stable_hash(
        {key: value for key, value in canonical.items() if key != "record_sha256"}
    )
    return canonical


def validate_semantic_supervision_evidence(
    evidence: Any,
    *,
    split: str,
    verification_method: str,
) -> dict[str, Any]:
    """Validate semantic-target authority separately from schema validity.

    Silver and residual rows may shape the optimizer, but only independently
    audited human semantic targets may occupy development/evaluation splits or
    support a decision-grade semantic claim.
    """

    if not isinstance(evidence, dict) or evidence.get("policy") != SEMANTIC_SUPERVISION_POLICY:
        raise KernelProtocolFault(
            "KERC_SEMANTIC_SUPERVISION_POLICY_INVALID",
            str(evidence.get("policy") if isinstance(evidence, dict) else type(evidence)),
            path="record.semantic_supervision.policy",
        )
    tier = str(evidence.get("evidence_tier") or "")
    contract = SEMANTIC_EVIDENCE_TIERS.get(tier)
    if contract is None:
        raise KernelProtocolFault(
            "KERC_SEMANTIC_EVIDENCE_TIER_INVALID",
            tier,
            path="record.semantic_supervision.evidence_tier",
        )
    expected = {
        "producer_kind": contract["producer_kind"],
        "claim_authority": contract["claim_authority"],
        "model_derived": contract["model_derived"],
        "public_calibration_surface": False,
        "benchmark_payload_used": False,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise KernelProtocolFault(
                "KERC_SEMANTIC_EVIDENCE_CONTRACT_INVALID",
                f"{key}={evidence.get(key)!r}",
                path=f"record.semantic_supervision.{key}",
            )
    if split not in contract["allowed_splits"]:
        raise KernelProtocolFault(
            "KERC_SEMANTIC_EVIDENCE_SPLIT_FORBIDDEN",
            f"{tier}:{split}",
            path="record.semantic_supervision.evidence_tier",
        )
    if verification_method not in contract["verification_methods"]:
        raise KernelProtocolFault(
            "KERC_SEMANTIC_VERIFICATION_METHOD_MISMATCH",
            f"{tier}:{verification_method}",
            path="record.verification_receipt.method",
        )
    for key in ("producer_id", "annotation_source_id"):
        if not str(evidence.get(key) or "").strip():
            raise KernelProtocolFault(
                "KERC_SEMANTIC_EVIDENCE_IDENTITY_MISSING",
                key,
                path=f"record.semantic_supervision.{key}",
            )
    for key in ("producer_artifact_sha256", "annotation_source_sha256"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence.get(key) or "")):
            raise KernelProtocolFault(
                "KERC_SEMANTIC_EVIDENCE_HASH_INVALID",
                key,
                path=f"record.semantic_supervision.{key}",
            )
    objectives = evidence.get("objective_authority")
    if not isinstance(objectives, dict) or set(objectives) != set(TRAINING_OBJECTIVES):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_OBJECTIVE_AUTHORITY_INVALID",
            canonical_json(objectives),
            path="record.semantic_supervision.objective_authority",
        )
    if any(objectives.get(objective) is not True for objective in TRAINING_OBJECTIVES):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_OBJECTIVE_AUTHORITY_INCOMPLETE",
            canonical_json(objectives),
            path="record.semantic_supervision.objective_authority",
        )
    weight = evidence.get("optimizer_sampling_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_SAMPLING_WEIGHT_INVALID",
            str(weight),
            path="record.semantic_supervision.optimizer_sampling_weight",
        )
    weight = float(weight)
    if not 0.0 < weight <= float(contract["maximum_optimizer_sampling_weight"]):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_SAMPLING_WEIGHT_INVALID",
            f"{tier}:{weight}",
            path="record.semantic_supervision.optimizer_sampling_weight",
        )
    canonical = copy.deepcopy(evidence)
    canonical["optimizer_sampling_weight"] = weight
    return canonical


def training_semantic_payload_sha256(record: dict[str, Any]) -> str:
    packet = record.get("kernel_packet") if isinstance(record.get("kernel_packet"), dict) else {}
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    answer = validate_answer_packet(
        record.get("answer_packet") if isinstance(record.get("answer_packet"), dict) else {}
    )
    return stable_hash(
        {
            "source_text": str(record.get("source_text") or ""),
            "kernel_packet_sha256": str(packet.get("packet_sha256") or ""),
            "answer_packet": answer,
            "surface_target": str(record.get("surface_target") or ""),
            "residual_supervision": record.get("residual_supervision") or {},
            "semantic_supervision": record.get("semantic_supervision") or {},
            "source_id": str(provenance.get("source_id") or ""),
            "source_group": str(provenance.get("source_group") or ""),
            "license_spdx": str(provenance.get("license_spdx") or ""),
            "permitted_use": str(provenance.get("permitted_use") or ""),
        }
    )


def compile_training_views(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile matched learned objectives without multiplying unique-data credit."""

    record = validate_training_record(record)
    packet = record["kernel_packet"]
    protected_context = {
        "protected_objects": packet["protected_objects"],
        "concept_capsules": packet["concept_capsules"],
        "source_character_length": packet["source"]["character_length"],
    }
    compiler_input = {
        **protected_context,
        "masked_surface": _masked_surface_from_packet_objects(
            record["source_text"], packet["protected_objects"]
        ),
        "correction_lattice": packet["correction_lattice"],
    }
    compiler_target = {
        "kernel_version": KERNEL_VERSION,
        "program": packet["program"],
    }
    core_input = {
        **protected_context,
        "program": packet["program"],
        "residual": packet["residual"],
    }
    renderer_input = {
        "answer_packet": record["answer_packet"],
        "protected_objects": packet["protected_objects"],
        "residual_mode": packet["residual"]["mode"],
        "fidelity": packet["residual"]["fidelity"],
    }
    source = record["source_text"]
    rows = (
        ("surface_direct_control_v1", source, record["surface_target"]),
        (
            "surface_to_kernel_program_v1",
            canonical_json(compiler_input),
            canonical_json(compiler_target),
        ),
        (
            "kernel_program_to_answer_packet_v1",
            canonical_json(core_input),
            canonical_json(record["answer_packet"]),
        ),
        (
            "answer_packet_to_surface_v1",
            canonical_json(renderer_input),
            record["surface_target"],
        ),
    )
    compiled = []
    for objective, visible, target in rows:
        prompt = visible
        trusted_prefix = [TRAINING_TASK_TAGS[objective]]
        identity = stable_hash(
            {
                "record_sha256": record["record_sha256"],
                "objective": objective,
                "trusted_source_prefix_tokens": trusted_prefix,
                "prompt": prompt,
                "target": target,
            }
        )
        verifier_negative = _targeted_verifier_corruption(
            objective,
            target,
            protected_objects=packet["protected_objects"],
        )
        compiled.append(
            {
                "policy": TRAINING_VIEW_POLICY,
                "row_id": "kerc-view:" + identity.split(":", 1)[1][:24],
                "split": record["split"],
                "arm_id": "english",
                "objective": objective,
                "task_tag": TRAINING_TASK_TAGS[objective],
                "trusted_source_prefix_tokens": trusted_prefix,
                "trusted_prefix_authority": "internal_objective_route_only",
                "prompt": prompt,
                "prompt_sha256": stable_hash(prompt.encode("utf-8")),
                "target": target,
                "target_sha256": stable_hash(target.encode("utf-8")),
                "source_record_sha256": record["record_sha256"],
                "raw_source_sha256": record["raw_source_sha256"],
                "source_group": record["provenance"]["source_group"],
                "license_spdx": record["provenance"]["license_spdx"],
                "derived_view": True,
                "unique_source_credit": 0,
                "optimizer_exposure_credit": 1,
                "semantic_evidence_tier": record["semantic_supervision"]["evidence_tier"],
                "semantic_claim_authority": record["semantic_supervision"]["claim_authority"],
                "decision_grade_reference": record["semantic_supervision"][
                    "claim_authority"
                ]
                == "decision_grade_reference",
                "optimizer_sampling_weight": record["semantic_supervision"][
                    "optimizer_sampling_weight"
                ],
                "kerc_residual_channels": list(KERC_RESIDUAL_CHANNELS),
                "kerc_residual_labels": [
                    int(record["residual_supervision"]["labels_by_channel"][channel])
                    for channel in KERC_RESIDUAL_CHANNELS
                ],
                "kerc_verifier_dimensions": list(KERC_VERIFIER_DIMENSIONS),
                "kerc_verifier_positive_labels": [1, 1, 1, 1],
                "kerc_verifier_negative": verifier_negative,
                "generator_visible_fields": ["trusted_source_prefix_tokens", "prompt"],
                "evaluator_only_fields": [
                    "target",
                    "target_sha256",
                    "source_record_sha256",
                    "kerc_verifier_negative",
                ],
                "public_benchmark": False,
                "public_tests_included": False,
                "public_benchmark_solutions_included": False,
                "external_inference": False,
                **NO_CHEAT,
            }
        )
    return compiled


def compiler_input_from_source(source: str) -> dict[str, Any]:
    """Build the deterministic, source-only front end seen by the compiler."""

    protected = extract_protected_objects(source)
    return {
        "protected_objects": protected["protected_objects"],
        "concept_capsules": {},
        "source_character_length": len(source),
        "masked_surface": protected["masked_surface"],
        "correction_lattice": build_correction_lattice(
            source, protected["protected_objects"], []
        ),
    }


def execute_learned_pipeline(
    source: str,
    *,
    hrl_state: dict[str, Any],
    stage_executor: Callable[[str, str], tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    """Execute compiler, core, renderer, and learned round-trip verification.

    ``stage_executor`` is the only learned boundary. It receives one trusted
    objective id and one source-only prompt. Every intermediate is parsed and
    independently validated before it can become input to the next stage.
    There is no literal, template, tool, direct-surface, or best-effort route.
    """

    stage_receipts: list[dict[str, Any]] = []

    def run_stage(objective: str, prompt: str) -> str:
        output, receipt = stage_executor(objective, prompt)
        if not isinstance(receipt, dict) or receipt.get("state") != "GREEN":
            raise KernelProtocolFault(
                "KERC_LEARNED_STAGE_FAULT",
                f"{objective}:{receipt.get('reason') if isinstance(receipt, dict) else type(receipt)}",
                path=objective,
            )
        if int(receipt.get("fallback_return_count") or 0):
            raise KernelProtocolFault(
                "KERC_LEARNED_STAGE_FALLBACK_FORBIDDEN", objective, path=objective
            )
        if not str(output).strip():
            raise KernelProtocolFault(
                "KERC_LEARNED_STAGE_EMPTY", objective, path=objective
            )
        stage_receipts.append(
            {
                "objective": objective,
                "output_sha256": stable_hash(str(output).encode("utf-8")),
                "receipt": copy.deepcopy(receipt),
            }
        )
        return str(output)

    def parse_object(objective: str, text: str) -> dict[str, Any]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KernelProtocolFault(
                "KERC_LEARNED_STAGE_JSON_INVALID", str(exc), path=objective
            ) from exc
        if not isinstance(payload, dict):
            raise KernelProtocolFault(
                "KERC_LEARNED_STAGE_OBJECT_REQUIRED", type(payload).__name__, path=objective
            )
        return payload

    def compile_surface(surface: str) -> dict[str, Any]:
        front_end = compiler_input_from_source(surface)
        compiler_text = run_stage(
            "surface_to_kernel_program_v1", canonical_json(front_end)
        )
        compiler_output = parse_object(
            "surface_to_kernel_program_v1", compiler_text
        )
        if compiler_output.get("kernel_version") != KERNEL_VERSION:
            raise KernelProtocolFault(
                "KERC_LEARNED_COMPILER_VERSION_INVALID",
                str(compiler_output.get("kernel_version")),
                path="surface_to_kernel_program_v1.kernel_version",
            )
        program = compiler_output.get("program")
        if not isinstance(program, dict):
            raise KernelProtocolFault(
                "KERC_LEARNED_COMPILER_PROGRAM_MISSING",
                str(type(program)),
                path="surface_to_kernel_program_v1.program",
            )
        packet = build_kernel_packet(
            surface,
            program,
            hrl_state=hrl_state,
            residual_mode="OUTPUT_REALIZATION",
            fidelity="faithful",
            provenance={"source": "learned_kerc_pipeline_v1"},
        )
        validate_kernel_packet(packet, local_hrl_state=hrl_state)
        return packet

    def reason(packet: dict[str, Any]) -> dict[str, Any]:
        core_input = {
            "protected_objects": packet["protected_objects"],
            "concept_capsules": packet["concept_capsules"],
            "source_character_length": packet["source"]["character_length"],
            "program": packet["program"],
            "residual": packet["residual"],
        }
        answer_text = run_stage(
            "kernel_program_to_answer_packet_v1", canonical_json(core_input)
        )
        answer = parse_object("kernel_program_to_answer_packet_v1", answer_text)
        return validate_answer_packet_against_context(
            answer,
            protected_objects=packet["protected_objects"],
            concept_capsules=packet["concept_capsules"],
        )

    packet = compile_surface(source)
    intended_answer = reason(packet)
    renderer_input = {
        "answer_packet": intended_answer,
        "protected_objects": packet["protected_objects"],
        "residual_mode": packet["residual"]["mode"],
        "fidelity": packet["residual"]["fidelity"],
    }
    surface = run_stage(
        "answer_packet_to_surface_v1", canonical_json(renderer_input)
    )
    reconstructed_packet = compile_surface(surface)
    reconstructed_answer = reason(reconstructed_packet)
    roundtrip = verify_answer_roundtrip(
        intended_answer,
        reconstructed_answer,
        protected_objects=packet["protected_objects"],
    )
    if roundtrip.get("passes") is not True:
        raise KernelProtocolFault(
            "KERC_LEARNED_ROUNDTRIP_MISMATCH",
            canonical_json(roundtrip.get("hard_failures") or []),
            path="roundtrip",
        )
    receipt = {
        "policy": "project_theseus_kerc_learned_pipeline_execution_v1",
        "state": "GREEN",
        "stage_count": len(stage_receipts),
        "stage_objectives": [row["objective"] for row in stage_receipts],
        "stage_receipts": stage_receipts,
        "source_sha256": stable_hash(source.encode("utf-8")),
        "surface_sha256": stable_hash(surface.encode("utf-8")),
        "initial_packet_sha256": packet["packet_sha256"],
        "recompiled_packet_sha256": reconstructed_packet["packet_sha256"],
        "roundtrip": roundtrip,
        "direct_surface_route_used": False,
        "semantic_equivalence_claimed": False,
        "truth_verified": False,
        "failure_behavior": "reject_without_fallback",
        **NO_CHEAT,
    }
    return surface, receipt


def _validate_residual_supervision(
    record: dict[str, Any], *, packet: dict[str, Any]
) -> dict[str, Any]:
    supervision = (
        record.get("residual_supervision")
        if isinstance(record.get("residual_supervision"), dict)
        else {}
    )
    if supervision.get("policy") != "project_theseus_kerc_residual_supervision_v1":
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_POLICY_INVALID",
            str(supervision.get("policy")),
            path="record.residual_supervision.policy",
        )
    labels = (
        supervision.get("labels_by_channel")
        if isinstance(supervision.get("labels_by_channel"), dict)
        else {}
    )
    if set(labels) != set(KERC_RESIDUAL_CHANNELS):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_CHANNELS_INVALID",
            canonical_json(labels),
            path="record.residual_supervision.labels_by_channel",
        )
    for channel in KERC_RESIDUAL_CHANNELS:
        value = labels[channel]
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
            raise KernelProtocolFault(
                "KERC_RESIDUAL_SUPERVISION_LABEL_INVALID",
                f"{channel}:{value}",
                path=f"record.residual_supervision.labels_by_channel.{channel}",
            )
    residual = packet.get("residual") if isinstance(packet.get("residual"), dict) else {}
    expected_fidelity = KERC_FIDELITY_LABELS.get(str(residual.get("fidelity") or ""))
    if expected_fidelity is None or int(supervision.get("record_fidelity_label", -1)) != expected_fidelity:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_FIDELITY_MISMATCH",
            canonical_json(supervision),
            path="record.residual_supervision.record_fidelity_label",
        )
    if residual.get("exact_object_handles") and labels["exact"] != 3:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_EXACT_OBJECT_UNDERSPECIFIED",
            str(labels["exact"]),
            path="record.residual_supervision.labels_by_channel.exact",
        )
    if not residual.get("segment_frame") and labels["segment"] != 0:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_EMPTY_SEGMENT_NONZERO",
            str(labels["segment"]),
            path="record.residual_supervision.labels_by_channel.segment",
        )
    if not residual.get("token_tags") and labels["token"] != 0:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_EMPTY_TOKEN_NONZERO",
            str(labels["token"]),
            path="record.residual_supervision.labels_by_channel.token",
        )
    evidence = str(supervision.get("evidence_sha256") or "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", evidence):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_EVIDENCE_INVALID",
            evidence,
            path="record.residual_supervision.evidence_sha256",
        )
    if supervision.get("annotator_independent_of_model") is not True:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_ANNOTATOR_INVALID",
            canonical_json(supervision),
            path="record.residual_supervision.annotator_independent_of_model",
        )
    return copy.deepcopy(supervision)


def _targeted_verifier_corruption(
    objective: str,
    target: str,
    *,
    protected_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dimension_by_objective = {
        "surface_direct_control_v1": "protected_object_consistency",
        "surface_to_kernel_program_v1": "semantic_consistency",
        "kernel_program_to_answer_packet_v1": "numeric_value_consistency",
        "answer_packet_to_surface_v1": "surface_fidelity",
    }
    dimension = dimension_by_objective[objective]
    corrupted = target
    strategy = ""
    if objective == "surface_to_kernel_program_v1":
        payload = json.loads(target)
        payload["program"]["nodes"][0]["operator"] = "CORRUPTED_OPERATOR"
        corrupted = canonical_json(payload)
        strategy = "replace_first_kernel_operator"
    elif objective == "kernel_program_to_answer_packet_v1":
        payload = json.loads(target)
        changed = _increment_first_numeric_value(payload)
        if not changed:
            payload["claims"][0]["confidence"] = max(
                0.0, min(1.0, float(payload["claims"][0].get("confidence", 0.5)) / 2.0)
            )
            dimension = "semantic_consistency"
            strategy = "change_first_claim_confidence"
        else:
            strategy = "increment_first_numeric_value"
        corrupted = canonical_json(payload)
    elif objective == "surface_direct_control_v1":
        for row in protected_objects.values():
            raw = row.get("inline_bytes_b64")
            if not raw:
                continue
            exact = base64.b64decode(str(raw)).decode("utf-8")
            if exact in corrupted:
                corrupted = corrupted.replace(exact, "[CORRUPTED_OBJECT]", 1)
                strategy = "replace_first_protected_object"
                break
        if not strategy:
            corrupted += " [CORRUPTED_OBJECT]"
            strategy = "append_protected_object_mismatch_marker"
    else:
        corrupted += " [CORRUPTED_SURFACE_FIDELITY]"
        strategy = "append_surface_fidelity_mismatch_marker"
    if corrupted == target:
        raise KernelProtocolFault(
            "KERC_VERIFIER_CORRUPTION_NOOP",
            objective,
            path="training_view.kerc_verifier_negative",
        )
    labels = [1, 1, 1, 1]
    labels[KERC_VERIFIER_DIMENSIONS.index(dimension)] = 0
    return {
        "policy": "project_theseus_kerc_targeted_verifier_corruption_v1",
        "target": corrupted,
        "target_sha256": stable_hash(corrupted.encode("utf-8")),
        "labels": labels,
        "failed_dimension": dimension,
        "strategy": strategy,
        "generator_loss_enabled": False,
        "unique_source_credit": 0,
        "candidate_generation_credit": 0,
    }


def _increment_first_numeric_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "value" and isinstance(item, (int, float)) and not isinstance(item, bool):
                value[key] = item + 1
                return True
            if _increment_first_numeric_value(item):
                return True
    elif isinstance(value, list):
        for item in value:
            if _increment_first_numeric_value(item):
                return True
    return False


def parse_learned_compiler_output(
    output: str,
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_character_length: int,
) -> dict[str, Any]:
    """Parse and independently validate a learned compiler result."""

    try:
        decoded = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KernelProtocolFault(
            "KERC_LEARNED_COMPILER_OUTPUT_INVALID", str(exc), path="compiler_output"
        ) from exc
    if not isinstance(decoded, dict) or decoded.get("kernel_version") != KERNEL_VERSION:
        raise KernelProtocolFault(
            "KERC_LEARNED_COMPILER_VERSION_INVALID",
            str(decoded.get("kernel_version") if isinstance(decoded, dict) else type(decoded)),
            path="compiler_output.kernel_version",
        )
    return validate_kernel_program(
        decoded.get("program") if isinstance(decoded.get("program"), dict) else {},
        protected_objects=protected_objects,
        concept_capsules=concept_capsules,
        source_character_length=source_character_length,
    )


def parse_learned_answer_output(output: str) -> dict[str, Any]:
    """Parse and independently validate a learned core answer packet."""

    try:
        decoded = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_OUTPUT_INVALID", str(exc), path="answer_output"
        ) from exc
    return validate_answer_packet(decoded if isinstance(decoded, dict) else {})


def _masked_surface_from_packet_objects(
    source: str, objects: dict[str, dict[str, Any]]
) -> str:
    spans = sorted(
        (
            int(row["source_span"]["character_start"]),
            int(row["source_span"]["character_end"]),
            handle,
        )
        for handle, row in objects.items()
    )
    pieces: list[str] = []
    cursor = 0
    for start, end, handle in spans:
        if start < cursor or not 0 <= start < end <= len(source):
            raise KernelProtocolFault(
                "KERC_TRAINING_OBJECT_SPAN_INVALID",
                f"{handle}:{start}:{end}",
                path="record.kernel_packet.protected_objects",
            )
        pieces.extend((source[cursor:start], handle))
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def _validate_packet_objects_against_source(
    source: str, objects: dict[str, dict[str, Any]]
) -> None:
    cursor = 0
    for start, end, handle in sorted(
        (
            int(row["source_span"]["character_start"]),
            int(row["source_span"]["character_end"]),
            handle,
        )
        for handle, row in objects.items()
    ):
        row = objects[handle]
        if start < cursor or not 0 <= start < end <= len(source):
            raise KernelProtocolFault(
                "KERC_TRAINING_OBJECT_SPAN_INVALID",
                f"{handle}:{start}:{end}",
                path="record.kernel_packet.protected_objects",
            )
        raw = source[start:end].encode("utf-8")
        if stable_hash(raw) != row.get("content_ref"):
            raise KernelProtocolFault(
                "KERC_TRAINING_OBJECT_CONTENT_MISMATCH",
                handle,
                path=f"record.kernel_packet.protected_objects.{handle}",
            )
        inline = row.get("inline_bytes_b64")
        if inline is not None:
            try:
                decoded = base64.b64decode(str(inline), validate=True)
            except Exception as exc:
                raise KernelProtocolFault(
                    "KERC_TRAINING_OBJECT_INLINE_INVALID",
                    handle,
                    path=f"record.kernel_packet.protected_objects.{handle}",
                ) from exc
            if decoded != raw:
                raise KernelProtocolFault(
                    "KERC_TRAINING_OBJECT_INLINE_MISMATCH",
                    handle,
                    path=f"record.kernel_packet.protected_objects.{handle}",
                )
        cursor = end


def capture_source(
    source: str | bytes,
    *,
    retain_inline: bool = False,
    retention: str = "transient",
    language: str = "en",
) -> dict[str, Any]:
    raw = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    try:
        text = raw.decode("utf-8")
        encoding = "UTF-8"
    except UnicodeDecodeError as exc:
        raise KernelProtocolFault("KERC_SOURCE_ENCODING_UNSUPPORTED", str(exc), path="source") from exc
    record = {
        "record_type": "immutable_kernel_source_record",
        "source_sha256": stable_hash(raw),
        "byte_length": len(raw),
        "character_length": len(text),
        "encoding": encoding,
        "unicode_normalization": _normalization_form(text),
        "language_hint": language,
        "retention": retention,
        "inline_bytes_b64": base64.b64encode(raw).decode("ascii") if retain_inline else None,
        "content_address": stable_hash(raw),
    }
    record["record_sha256"] = stable_hash(record)
    return record


def extract_protected_objects(
    source: str,
    *,
    explicit_spans: Sequence[dict[str, Any]] = (),
    retain_inline: bool = True,
) -> dict[str, Any]:
    """Protect exact/form-sensitive spans before any correction or compilation."""

    candidates = _explicit_span_candidates(source, explicit_spans)
    patterns = (
        (r"```[\s\S]*?```", "CODE", "EXACT", 90, "fenced_code"),
        (r"`[^`\n]+`", "CODE", "EXACT", 85, "inline_code"),
        (r"https?://[^\s<>\]\[(){}]+", "URL", "EXACT", 80, "url"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "EMAIL", "EXACT", 80, "email"),
        (r"\b(?:sha(?:1|224|256|384|512):)?[0-9a-fA-F]{32,128}\b", "HASH", "EXACT", 78, "hash"),
        (r"(?<!\w)(?:~?/|\./|\.\./)[A-Za-z0-9._~+@%/\-]+", "FILE_PATH", "EXACT", 75, "file_path"),
        (r"\"[^\"\n]+\"|“[^”\n]+”|‘[^’\n]+’", "QUOTE", "EXACT", 70, "quotation"),
        (
            r"(?<!\w)(?:[$€£¥]\s*)?-?\d+(?:[,.]\d+)*(?:\s*(?:%|ms|s|min|h|days?|bytes?|KB|MB|GB|TB|kg|g|km|m|cm|mm|°[CF]|USD|EUR|GBP))?(?!\w)",
            "NUMBER",
            "VALUE_AND_STYLE",
            60,
            "number_or_unit",
        ),
    )
    for pattern, object_type, copy_policy, priority, label in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            candidates.append(
                SpanCandidate(match.start(), match.end(), object_type, copy_policy, priority, label)
            )
    selected = _select_non_overlapping(candidates)
    counters: dict[str, int] = {}
    objects: dict[str, dict[str, Any]] = {}
    alignments: list[dict[str, Any]] = []
    pieces: list[str] = []
    cursor = 0
    for span in selected:
        pieces.append(source[cursor : span.start])
        prefix = HANDLE_PREFIX_BY_TYPE[span.object_type]
        counters[prefix] = counters.get(prefix, 0) + 1
        handle = f"@{prefix}{counters[prefix]}"
        exact = source[span.start : span.end]
        raw = exact.encode("utf-8")
        char_to_byte_start = len(source[: span.start].encode("utf-8"))
        char_to_byte_end = char_to_byte_start + len(raw)
        record = {
            "record_type": "kernel_protected_object",
            "handle": handle,
            "object_type": span.object_type,
            "copy_policy": span.copy_policy,
            "content_ref": stable_hash(raw),
            "inline_bytes_b64": base64.b64encode(raw).decode("ascii") if retain_inline else None,
            "encoding": "UTF-8",
            "source_span": {
                "character_start": span.start,
                "character_end": span.end,
                "byte_start": char_to_byte_start,
                "byte_end": char_to_byte_end,
            },
            "protection_source": span.source,
            "access_policy": "task_scoped_least_privilege",
        }
        record["object_sha256"] = stable_hash(record)
        objects[handle] = record
        alignments.append(
            {
                "handle": handle,
                "source_span": record["source_span"],
                "derivation": "preserved",
                "confidence": 1.0,
            }
        )
        pieces.append(handle)
        cursor = span.end
    pieces.append(source[cursor:])
    return {
        "policy": "project_theseus_kernel_protected_object_extraction_v1",
        "masked_surface": "".join(pieces),
        "protected_objects": objects,
        "source_alignment": alignments,
        "protected_character_count": sum(span.end - span.start for span in selected),
        "unprotected_character_count": len(source) - sum(span.end - span.start for span in selected),
        **NO_CHEAT,
    }


def build_correction_lattice(
    source: str,
    protected_objects: dict[str, dict[str, Any]],
    proposals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate correction hypotheses while preserving uncertainty and source form."""

    protected_ranges = [
        (
            int(row["source_span"]["character_start"]),
            int(row["source_span"]["character_end"]),
        )
        for row in protected_objects.values()
    ]
    rows: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        path = f"corrections[{index}]"
        start = _required_int(proposal, "start", path)
        end = _required_int(proposal, "end", path)
        if not 0 <= start < end <= len(source):
            raise KernelProtocolFault("KERC_CORRECTION_SPAN_INVALID", f"{start}:{end}", path=path)
        if any(start < other_end and end > other_start for other_start, other_end in protected_ranges):
            raise KernelProtocolFault("KERC_CORRECTION_TOUCHES_PROTECTED_OBJECT", f"{start}:{end}", path=path)
        alternatives = proposal.get("alternatives")
        if not isinstance(alternatives, list) or len(alternatives) < 2:
            raise KernelProtocolFault("KERC_CORRECTION_ALTERNATIVES_INSUFFICIENT", "need at least two", path=path)
        normalized = []
        seen = set()
        probability_sum = 0.0
        for alt_index, alternative in enumerate(alternatives):
            if not isinstance(alternative, dict):
                raise KernelProtocolFault("KERC_CORRECTION_ALTERNATIVE_INVALID", str(alternative), path=f"{path}.alternatives[{alt_index}]")
            form = str(alternative.get("form") or "")
            probability = float(alternative.get("probability") or 0.0)
            if not form or form in seen or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise KernelProtocolFault("KERC_CORRECTION_ALTERNATIVE_INVALID", canonical_json(alternative), path=f"{path}.alternatives[{alt_index}]")
            seen.add(form)
            probability_sum += probability
            normalized.append({"form": form, "probability": probability, "evidence": alternative.get("evidence")})
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-6):
            raise KernelProtocolFault("KERC_CORRECTION_PROBABILITY_MASS_INVALID", str(probability_sum), path=path)
        original = source[start:end]
        if original not in seen:
            raise KernelProtocolFault("KERC_CORRECTION_ORIGINAL_NOT_RETAINED", original, path=path)
        rows.append(
            {
                "source_span": [start, end],
                "source_form": original,
                "alternatives": sorted(normalized, key=lambda row: (-row["probability"], row["form"])),
                "decision": "UNRESOLVED_REQUIRES_CALIBRATED_COMPILER",
                "form_sensitive": False,
            }
        )
    return {
        "policy": "project_theseus_kernel_correction_lattice_v1",
        "corrections": rows,
        "uncertainty_preserved": True,
        "automatic_corrections_applied": 0,
        **NO_CHEAT,
    }


def validate_kernel_program(
    program: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_character_length: int,
) -> dict[str, Any]:
    nodes = program.get("nodes")
    roots = program.get("roots")
    if not isinstance(nodes, list) or not nodes:
        raise KernelProtocolFault("KERC_PROGRAM_NODES_MISSING", "nodes must be non-empty", path="program.nodes")
    if not isinstance(roots, list) or not roots:
        raise KernelProtocolFault("KERC_PROGRAM_ROOTS_MISSING", "roots must be non-empty", path="program.roots")
    by_id: dict[str, dict[str, Any]] = {}
    refs: dict[str, set[str]] = {}
    handles_seen: set[str] = set()
    for index, node in enumerate(nodes):
        path = f"program.nodes[{index}]"
        if not isinstance(node, dict):
            raise KernelProtocolFault("KERC_NODE_INVALID", str(node), path=path)
        node_id = str(node.get("node_id") or "")
        if not re.fullmatch(r"k[0-9]+", node_id) or node_id in by_id:
            raise KernelProtocolFault("KERC_NODE_ID_INVALID", node_id, path=f"{path}.node_id")
        operator = str(node.get("operator") or "")
        if not re.fullmatch(r"(?:[A-Z][A-Z0-9_]*|@M[0-9]+)", operator):
            raise KernelProtocolFault("KERC_OPERATOR_INVALID", operator, path=f"{path}.operator")
        modality = str(node.get("modality") or "ASSERTED")
        polarity = str(node.get("polarity") or "AFFIRMED")
        quantifier = str(node.get("quantifier") or "NONE")
        confidence = float(node.get("confidence", 1.0))
        derivation = str(node.get("derivation") or "")
        if modality not in MODALITIES:
            raise KernelProtocolFault("KERC_MODALITY_INVALID", modality, path=f"{path}.modality")
        if polarity not in POLARITIES:
            raise KernelProtocolFault("KERC_POLARITY_INVALID", polarity, path=f"{path}.polarity")
        if quantifier not in QUANTIFIERS:
            raise KernelProtocolFault("KERC_QUANTIFIER_INVALID", quantifier, path=f"{path}.quantifier")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise KernelProtocolFault("KERC_CONFIDENCE_INVALID", str(confidence), path=f"{path}.confidence")
        if derivation not in DERIVATIONS:
            raise KernelProtocolFault("KERC_DERIVATION_INVALID", derivation, path=f"{path}.derivation")
        arguments = node.get("arguments")
        if not isinstance(arguments, list):
            raise KernelProtocolFault("KERC_ARGUMENTS_INVALID", "arguments must be a list", path=f"{path}.arguments")
        node_refs: set[str] = set()
        for arg_index, argument in enumerate(arguments):
            arg_path = f"{path}.arguments[{arg_index}]"
            if not isinstance(argument, dict) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(argument.get("role") or "")):
                raise KernelProtocolFault("KERC_ROLE_INVALID", canonical_json(argument), path=arg_path)
            _validate_value(
                argument.get("value"),
                path=f"{arg_path}.value",
                protected_objects=protected_objects,
                concept_capsules=concept_capsules,
                node_refs=node_refs,
                handles_seen=handles_seen,
            )
        for span_index, span in enumerate(node.get("source_spans") or []):
            if not isinstance(span, list) or len(span) != 2:
                raise KernelProtocolFault("KERC_ALIGNMENT_SPAN_INVALID", canonical_json(span), path=f"{path}.source_spans[{span_index}]")
            start, end = int(span[0]), int(span[1])
            if not 0 <= start < end <= source_character_length:
                raise KernelProtocolFault("KERC_ALIGNMENT_SPAN_INVALID", f"{start}:{end}", path=f"{path}.source_spans[{span_index}]")
        by_id[node_id] = copy.deepcopy(node)
        refs[node_id] = node_refs
    unknown_roots = sorted(set(str(root) for root in roots) - set(by_id))
    unknown_refs = sorted({ref for values in refs.values() for ref in values if ref not in by_id})
    if unknown_roots:
        raise KernelProtocolFault("KERC_ROOT_REFERENCE_UNKNOWN", ",".join(unknown_roots), path="program.roots")
    if unknown_refs:
        raise KernelProtocolFault("KERC_NODE_REFERENCE_UNKNOWN", ",".join(unknown_refs), path="program.nodes")
    _reject_reference_cycles(refs)
    canonical_program = {
        "record_type": "kernel_program",
        "kernel_version": KERNEL_VERSION,
        "roots": [str(root) for root in roots],
        "nodes": [by_id[node_id] for node_id in sorted(by_id, key=_node_sort_key)],
    }
    canonical_program["program_sha256"] = stable_hash(canonical_program)
    return {
        "state": "READY",
        "canonical_program": canonical_program,
        "node_count": len(by_id),
        "referenced_handles": sorted(handles_seen),
        "acyclic": True,
        **NO_CHEAT,
    }


def serialize_kernel_program(
    canonical_program: dict[str, Any],
    *,
    macros: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    tokens: list[dict[str, str]] = [token("V_P", f"VERSION:{KERNEL_VERSION}")]
    for node in canonical_program["nodes"]:
        tokens.extend(
            [
                token("V_P", "NODE_BEGIN"),
                token("V_P", f"NODE_ID:{node['node_id']}"),
                token("V_K", f"OP:{node['operator']}"),
                token("V_K", f"MOD:{node.get('modality', 'ASSERTED')}"),
                token("V_K", f"POL:{node.get('polarity', 'AFFIRMED')}"),
                token("V_K", f"QUANT:{node.get('quantifier', 'NONE')}"),
            ]
        )
        for argument in node.get("arguments") or []:
            tokens.append(token("V_K", f"ROLE:{argument['role']}"))
            tokens.extend(_serialize_value(argument["value"]))
        tokens.append(token("V_P", "NODE_END"))
    tokens.append(token("V_P", "PROGRAM_END"))
    macro_registry = validate_macro_registry(macros)
    compact = apply_macros(tokens, macro_registry)
    expanded = expand_macros(compact, macro_registry)
    if expanded != tokens:
        raise KernelProtocolFault("KERC_MACRO_ROUNDTRIP_MISMATCH", "expanded tokens differ", path="serialization")
    return {
        "policy": "project_theseus_kernel_three_code_space_serialization_v1",
        "expanded_tokens": tokens,
        "compact_tokens": compact,
        "macro_registry": macro_registry,
        "expanded_sha256": stable_hash(tokens),
        "compact_sha256": stable_hash(compact),
        "macro_roundtrip_exact": True,
        "code_space_counts": {
            space: sum(1 for row in compact if row["space"] == space) for space in CODE_SPACES
        },
        **NO_CHEAT,
    }


def validate_macro_registry(macros: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(macros):
        path = f"macros[{index}]"
        if not isinstance(raw, dict):
            raise KernelProtocolFault("KERC_MACRO_INVALID", str(raw), path=path)
        macro_id = str(raw.get("macro_id") or "")
        if not re.fullmatch(r"M[0-9]+", macro_id) or macro_id in seen:
            raise KernelProtocolFault("KERC_MACRO_ID_INVALID", macro_id, path=f"{path}.macro_id")
        expansion = raw.get("expansion")
        if not isinstance(expansion, list) or len(expansion) < 2:
            raise KernelProtocolFault("KERC_MACRO_EXPANSION_INVALID", "need at least two tokens", path=f"{path}.expansion")
        normalized = [_validate_token(row, path=f"{path}.expansion") for row in expansion]
        if any(row["space"] != "V_K" for row in normalized):
            raise KernelProtocolFault(
                "KERC_MACRO_CROSSES_PROTECTED_OR_CONTROL_BOUNDARY",
                "v1 macros may fuse only V_K tokens",
                path=f"{path}.expansion",
            )
        record = {
            "macro_id": macro_id,
            "expansion": normalized,
            "expansion_sha256": stable_hash(normalized),
            "typed_expansion": True,
            "may_cross_protected_boundary": False,
            "may_cross_scope_boundary": False,
        }
        registry.append(record)
        seen.add(macro_id)
    return sorted(registry, key=lambda row: (-len(row["expansion"]), row["macro_id"]))


def apply_macros(tokens: Sequence[dict[str, str]], registry: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        matched = None
        for macro in registry:
            expansion = macro["expansion"]
            if list(tokens[index : index + len(expansion)]) == expansion:
                matched = macro
                break
        if matched is None:
            output.append(dict(tokens[index]))
            index += 1
            continue
        output.append(token("V_P", f"MACRO:{matched['macro_id']}:{matched['expansion_sha256']}"))
        index += len(matched["expansion"])
    return output


def expand_macros(tokens: Sequence[dict[str, str]], registry: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    by_id = {row["macro_id"]: row for row in registry}
    output: list[dict[str, str]] = []
    for index, raw in enumerate(tokens):
        row = _validate_token(raw, path=f"tokens[{index}]")
        if row["space"] != "V_P" or not row["token"].startswith("MACRO:"):
            output.append(row)
            continue
        parts = row["token"].split(":", 2)
        if len(parts) != 3 or parts[1] not in by_id:
            raise KernelProtocolFault("KERC_MACRO_REFERENCE_UNKNOWN", row["token"], path=f"tokens[{index}]")
        macro = by_id[parts[1]]
        if parts[2] != macro["expansion_sha256"]:
            raise KernelProtocolFault("KERC_MACRO_EXPANSION_HASH_MISMATCH", row["token"], path=f"tokens[{index}]")
        output.extend(copy.deepcopy(macro["expansion"]))
    return output


def build_kernel_packet(
    source: str,
    program: dict[str, Any],
    *,
    hrl_state: dict[str, Any],
    correction_lattice: dict[str, Any] | None = None,
    concept_capsules: dict[str, dict[str, Any]] | None = None,
    explicit_spans: Sequence[dict[str, Any]] = (),
    macros: Sequence[dict[str, Any]] = (),
    residual_mode: str = "SOURCE_RECONSTRUCTION",
    fidelity: str = "faithful",
    retain_source_inline: bool = False,
    retain_objects_inline: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if residual_mode not in RESIDUAL_MODES:
        raise KernelProtocolFault("KERC_RESIDUAL_MODE_INVALID", residual_mode, path="residual.mode")
    if fidelity not in FIDELITY_MODES:
        raise KernelProtocolFault("KERC_FIDELITY_INVALID", fidelity, path="residual.fidelity")
    _validate_hrl_reference(hrl_state)
    source_record = capture_source(source, retain_inline=retain_source_inline)
    protected = extract_protected_objects(source, explicit_spans=explicit_spans, retain_inline=retain_objects_inline)
    capsules = copy.deepcopy(concept_capsules or {})
    _validate_concept_capsules(capsules)
    lattice = correction_lattice or build_correction_lattice(source, protected["protected_objects"], [])
    validated = validate_kernel_program(
        program,
        protected_objects=protected["protected_objects"],
        concept_capsules=capsules,
        source_character_length=len(source),
    )
    serialization = serialize_kernel_program(validated["canonical_program"], macros=macros)
    residual = {
        "mode": residual_mode,
        "fidelity": fidelity,
        "hrl_version": hrl_state["hrl_version"],
        "global_state_hash": hrl_state["state_hash"],
        "interaction_id": hrl_state["interaction_id"],
        "segment_frame": {},
        "token_tags": [],
        "exact_object_handles": sorted(protected["protected_objects"]),
        "missing_state_behavior": "reject_or_request_checkpoint_without_approximation",
    }
    packet_core = {
        "policy": POLICY,
        "packet_version": PACKET_VERSION,
        "kernel_version": KERNEL_VERSION,
        "codebook_version": CODEBOOK_VERSION,
        "source": source_record,
        "protected_objects": protected["protected_objects"],
        "concept_capsules": capsules,
        "correction_lattice": lattice,
        "program": validated["canonical_program"],
        "serialization": serialization,
        "residual": residual,
        "source_alignment": [
            *protected["source_alignment"],
            *[
                {
                    "kernel_node": node["node_id"],
                    "source_spans": node.get("source_spans") or [],
                    "derivation": node["derivation"],
                    "confidence": node.get("confidence", 1.0),
                }
                for node in validated["canonical_program"]["nodes"]
            ],
        ],
        "uncertainty": {
            "unresolved_correction_count": len(lattice.get("corrections") or []),
            "semantic_equivalence_claimed": False,
        },
        "provenance": copy.deepcopy(provenance or {}),
        "compatibility": {
            "kernel_version": KERNEL_VERSION,
            "packet_version": PACKET_VERSION,
            "hrl_version": hrl_state["hrl_version"],
            "concept_registry_hash": stable_hash(capsules),
            "macro_registry_hash": stable_hash(serialization["macro_registry"]),
        },
        **NO_CHEAT,
    }
    packet_core["packet_id"] = "kpacket:" + stable_hash(packet_core).split(":", 1)[1][:24]
    packet_core["packet_sha256"] = stable_hash(packet_core)
    validate_kernel_packet(packet_core, local_hrl_state=hrl_state)
    return packet_core


def validate_kernel_packet(packet: dict[str, Any], *, local_hrl_state: dict[str, Any]) -> dict[str, Any]:
    if packet.get("policy") != POLICY or packet.get("packet_version") != PACKET_VERSION:
        raise KernelProtocolFault("KERC_PACKET_PROTOCOL_INCOMPATIBLE", str(packet.get("packet_version")), path="packet")
    expected_packet_sha256 = stable_hash(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    if packet.get("packet_sha256") != expected_packet_sha256:
        raise KernelProtocolFault(
            "KERC_PACKET_IDENTITY_MISMATCH",
            f"expected={expected_packet_sha256} observed={packet.get('packet_sha256')}",
            path="packet.packet_sha256",
        )
    _validate_hrl_reference(local_hrl_state)
    residual = packet.get("residual") if isinstance(packet.get("residual"), dict) else {}
    if residual.get("global_state_hash") != local_hrl_state.get("state_hash"):
        raise KernelProtocolFault(
            "KERC_HRL_STATE_DESYNCHRONIZED",
            f"packet={residual.get('global_state_hash')} local={local_hrl_state.get('state_hash')}",
            path="packet.residual.global_state_hash",
        )
    source = packet.get("source") if isinstance(packet.get("source"), dict) else {}
    objects = packet.get("protected_objects") if isinstance(packet.get("protected_objects"), dict) else {}
    concepts = packet.get("concept_capsules") if isinstance(packet.get("concept_capsules"), dict) else {}
    expected_source_record_sha256 = stable_hash(
        {key: value for key, value in source.items() if key != "record_sha256"}
    )
    if source.get("record_sha256") != expected_source_record_sha256:
        raise KernelProtocolFault(
            "KERC_SOURCE_RECORD_IDENTITY_MISMATCH",
            str(source.get("record_sha256")),
            path="packet.source.record_sha256",
        )
    for handle, row in objects.items():
        expected_object_sha256 = stable_hash(
            {key: value for key, value in row.items() if key != "object_sha256"}
        )
        if row.get("object_sha256") != expected_object_sha256:
            raise KernelProtocolFault(
                "KERC_PROTECTED_OBJECT_IDENTITY_MISMATCH",
                str(handle),
                path=f"packet.protected_objects.{handle}.object_sha256",
            )
    validated = validate_kernel_program(
        packet.get("program") if isinstance(packet.get("program"), dict) else {},
        protected_objects=objects,
        concept_capsules=concepts,
        source_character_length=int(source.get("character_length") or 0),
    )
    if (packet.get("program") or {}).get("program_sha256") != validated[
        "canonical_program"
    ]["program_sha256"]:
        raise KernelProtocolFault(
            "KERC_PROGRAM_IDENTITY_MISMATCH",
            str((packet.get("program") or {}).get("program_sha256")),
            path="packet.program.program_sha256",
        )
    serialization = packet.get("serialization") if isinstance(packet.get("serialization"), dict) else {}
    expanded = expand_macros(serialization.get("compact_tokens") or [], serialization.get("macro_registry") or [])
    if stable_hash(expanded) != serialization.get("expanded_sha256"):
        raise KernelProtocolFault("KERC_PACKET_SERIALIZATION_REPLAY_MISMATCH", "expanded token hash differs", path="packet.serialization")
    missing_objects = sorted(set(validated["referenced_handles"]) - set(objects) - set(concepts))
    if missing_objects:
        raise KernelProtocolFault("KERC_PACKET_HANDLE_MISSING", ",".join(missing_objects), path="packet.program")
    return {
        "state": "READY",
        "packet_id": packet.get("packet_id"),
        "program_sha256": validated["canonical_program"]["program_sha256"],
        "state_hash_match": True,
        "serialization_replay_match": True,
        "semantic_equivalence_claimed": False,
        **NO_CHEAT,
    }


def validate_answer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    claims = packet.get("claims")
    if not isinstance(claims, list) or not claims:
        raise KernelProtocolFault("KERC_ANSWER_CLAIMS_MISSING", "claims must be non-empty", path="answer.claims")
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        path = f"answer.claims[{index}]"
        if not isinstance(claim, dict):
            raise KernelProtocolFault("KERC_ANSWER_CLAIM_INVALID", str(claim), path=path)
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id or claim_id in claim_ids:
            raise KernelProtocolFault("KERC_ANSWER_CLAIM_ID_INVALID", claim_id, path=f"{path}.claim_id")
        claim_ids.add(claim_id)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(claim.get("predicate") or "")):
            raise KernelProtocolFault("KERC_ANSWER_PREDICATE_INVALID", str(claim.get("predicate")), path=f"{path}.predicate")
        if str(claim.get("modality") or "") not in MODALITIES:
            raise KernelProtocolFault("KERC_ANSWER_MODALITY_INVALID", str(claim.get("modality")), path=f"{path}.modality")
        if str(claim.get("polarity") or "") not in POLARITIES:
            raise KernelProtocolFault("KERC_ANSWER_POLARITY_INVALID", str(claim.get("polarity")), path=f"{path}.polarity")
        confidence = float(claim.get("confidence", -1.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise KernelProtocolFault("KERC_ANSWER_CONFIDENCE_INVALID", str(confidence), path=f"{path}.confidence")
        if not isinstance(claim.get("arguments"), list):
            raise KernelProtocolFault("KERC_ANSWER_ARGUMENTS_INVALID", "arguments must be a list", path=f"{path}.arguments")
    canonical = copy.deepcopy(packet)
    canonical["answer_packet_sha256"] = stable_hash({key: value for key, value in canonical.items() if key != "answer_packet_sha256"})
    return canonical


def validate_answer_packet_against_context(
    packet: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate answer value types and references against its packet context."""

    canonical = validate_answer_packet(packet)
    handles_seen: set[str] = set()
    node_refs: set[str] = set()
    for claim_index, claim in enumerate(canonical["claims"]):
        for argument_index, argument in enumerate(claim["arguments"]):
            path = f"answer.claims[{claim_index}].arguments[{argument_index}]"
            if not isinstance(argument, dict) or not re.fullmatch(
                r"[A-Z][A-Z0-9_]*", str(argument.get("role") or "")
            ):
                raise KernelProtocolFault(
                    "KERC_ANSWER_ROLE_INVALID", canonical_json(argument), path=path
                )
            _validate_value(
                argument.get("value"),
                path=f"{path}.value",
                protected_objects=protected_objects,
                concept_capsules=concept_capsules,
                node_refs=node_refs,
                handles_seen=handles_seen,
            )
    if node_refs:
        raise KernelProtocolFault(
            "KERC_ANSWER_NODE_REFERENCE_FORBIDDEN",
            canonical_json(sorted(node_refs)),
            path="answer.claims",
        )
    return canonical


def verify_answer_roundtrip(
    intended: dict[str, Any],
    reconstructed: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare hard structured constraints after learned surface recompilation."""

    intended = validate_answer_packet(intended)
    reconstructed = validate_answer_packet(reconstructed)
    intended_constraints = _answer_constraints(intended, protected_objects)
    reconstructed_constraints = _answer_constraints(reconstructed, protected_objects)
    failures = []
    for field in (
        "entity_handles",
        "number_values",
        "claim_polarities",
        "claim_modalities",
        "quantifiers",
        "temporal_relations",
        "causal_relations",
        "attributions",
        "quotation_handles",
        "required_terms",
        "required_caveats",
    ):
        if intended_constraints[field] != reconstructed_constraints[field]:
            failures.append(
                {
                    "fault_type": f"KERC_ROUNDTRIP_{field.upper()}_MISMATCH",
                    "intended": intended_constraints[field],
                    "reconstructed": reconstructed_constraints[field],
                }
            )
    return {
        "policy": "project_theseus_kernel_answer_roundtrip_verifier_v1",
        "passes": not failures,
        "hard_failure_count": len(failures),
        "hard_failures": failures,
        "intended_constraints_sha256": stable_hash(intended_constraints),
        "reconstructed_constraints_sha256": stable_hash(reconstructed_constraints),
        "semantic_equivalence_claimed": False,
        "truth_verified": False,
        "failure_behavior": "reject_or_regenerate_learned_surface_without_literal_or_template_fallback",
        **NO_CHEAT,
    }


def token(space: str, value: str) -> dict[str, str]:
    if space not in CODE_SPACES or not value:
        raise KernelProtocolFault("KERC_CODE_SPACE_TOKEN_INVALID", f"{space}:{value}")
    return {"space": space, "token": value}


def _validate_token(value: Any, *, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise KernelProtocolFault("KERC_TOKEN_INVALID", str(value), path=path)
    return token(str(value.get("space") or ""), str(value.get("token") or ""))


def _validate_value(
    value: Any,
    *,
    path: str,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    node_refs: set[str],
    handles_seen: set[str],
) -> None:
    if not isinstance(value, dict) or str(value.get("type") or "") not in VALUE_TYPES:
        raise KernelProtocolFault("KERC_VALUE_INVALID", canonical_json(value), path=path)
    value_type = str(value["type"])
    payload = value.get("value")
    if value_type == "handle":
        handle = str(payload or "")
        if handle not in protected_objects and handle not in concept_capsules:
            raise KernelProtocolFault("KERC_HANDLE_REFERENCE_UNKNOWN", handle, path=path)
        handles_seen.add(handle)
    elif value_type == "concept":
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", str(payload or "")):
            raise KernelProtocolFault("KERC_CONCEPT_ID_INVALID", str(payload), path=path)
    elif value_type == "number":
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), (int, float)):
            raise KernelProtocolFault("KERC_NUMBER_INVALID", canonical_json(payload), path=path)
        if not math.isfinite(float(payload["value"])):
            raise KernelProtocolFault("KERC_NUMBER_INVALID", canonical_json(payload), path=path)
    elif value_type == "node_ref":
        ref = str(payload or "")
        if not re.fullmatch(r"k[0-9]+", ref):
            raise KernelProtocolFault("KERC_NODE_REFERENCE_INVALID", ref, path=path)
        node_refs.add(ref)
    elif value_type == "list":
        if not isinstance(payload, list):
            raise KernelProtocolFault("KERC_LIST_INVALID", canonical_json(payload), path=path)
        for index, item in enumerate(payload):
            _validate_value(item, path=f"{path}[{index}]", protected_objects=protected_objects, concept_capsules=concept_capsules, node_refs=node_refs, handles_seen=handles_seen)
    elif value_type == "ambiguity":
        if not isinstance(payload, list) or len(payload) < 2:
            raise KernelProtocolFault("KERC_AMBIGUITY_INVALID", canonical_json(payload), path=path)
        mass = 0.0
        for index, alternative in enumerate(payload):
            if not isinstance(alternative, dict):
                raise KernelProtocolFault("KERC_AMBIGUITY_INVALID", canonical_json(alternative), path=f"{path}[{index}]")
            probability = float(alternative.get("probability") or 0.0)
            mass += probability
            _validate_value(alternative.get("value"), path=f"{path}[{index}].value", protected_objects=protected_objects, concept_capsules=concept_capsules, node_refs=node_refs, handles_seen=handles_seen)
        if not math.isclose(mass, 1.0, abs_tol=1e-6):
            raise KernelProtocolFault("KERC_AMBIGUITY_PROBABILITY_MASS_INVALID", str(mass), path=path)
    elif value_type == "byte_literal":
        try:
            base64.b64decode(str(payload or ""), validate=True)
        except Exception as exc:
            raise KernelProtocolFault("KERC_BYTE_LITERAL_INVALID", str(payload), path=path) from exc
    elif value_type == "boolean" and not isinstance(payload, bool):
        raise KernelProtocolFault("KERC_BOOLEAN_INVALID", canonical_json(payload), path=path)
    elif value_type == "null" and payload is not None:
        raise KernelProtocolFault("KERC_NULL_INVALID", canonical_json(payload), path=path)


def _serialize_value(value: dict[str, Any]) -> list[dict[str, str]]:
    value_type = str(value["type"])
    payload = value.get("value")
    if value_type == "handle":
        return [token("V_P", f"HANDLE:{payload}")]
    if value_type == "concept":
        return [token("V_K", f"CONCEPT:{payload}")]
    if value_type == "number":
        return [
            token("V_P", f"NUMBER:{canonical_json(payload)}"),
        ]
    if value_type == "node_ref":
        return [token("V_P", f"NODE_REF:{payload}")]
    if value_type == "list":
        rows = [token("V_P", "LIST_BEGIN")]
        for item in payload:
            rows.extend(_serialize_value(item))
        rows.append(token("V_P", "LIST_END"))
        return rows
    if value_type == "ambiguity":
        rows = [token("V_P", "AMBIG_BEGIN")]
        for alternative in payload:
            rows.append(token("V_P", f"PROB:{alternative['probability']:.12g}"))
            rows.extend(_serialize_value(alternative["value"]))
        rows.append(token("V_P", "AMBIG_END"))
        return rows
    if value_type == "byte_literal":
        return [token("V_P", f"BYTE:{payload}")]
    if value_type == "boolean":
        return [token("V_K", "BOOL:TRUE" if payload else "BOOL:FALSE")]
    return [token("V_K", "NULL")]


def _answer_constraints(packet: dict[str, Any], objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    handles: set[str] = set()
    numbers: set[str] = set()
    polarities = []
    modalities = []
    quantifiers = []
    temporal = []
    causal = []
    attributions = []
    for claim in packet["claims"]:
        polarities.append((claim["claim_id"], claim["polarity"]))
        modalities.append((claim["claim_id"], claim["modality"]))
        quantifiers.append((claim["claim_id"], claim.get("quantifier", "NONE")))
        temporal.append((claim["claim_id"], canonical_json(claim.get("temporal") or {})))
        attributions.append((claim["claim_id"], canonical_json(claim.get("attribution") or {})))
        if claim["predicate"] in {"CAUSE", "PREVENT", "ENABLE", "RESULT_IN"}:
            causal.append((claim["claim_id"], claim["predicate"], canonical_json(claim["arguments"])))
        _collect_answer_values(claim["arguments"], handles, numbers)
    quote_handles = sorted(handle for handle in handles if objects.get(handle, {}).get("object_type") == "QUOTE")
    entity_handles = sorted(handle for handle in handles if objects.get(handle, {}).get("object_type") != "QUOTE")
    return {
        "entity_handles": entity_handles,
        "number_values": sorted(numbers),
        "claim_polarities": sorted(polarities),
        "claim_modalities": sorted(modalities),
        "quantifiers": sorted(quantifiers),
        "temporal_relations": sorted(temporal),
        "causal_relations": sorted(causal),
        "attributions": sorted(attributions),
        "quotation_handles": quote_handles,
        "required_terms": sorted(canonical_json(row) for row in packet.get("required_terms") or []),
        "required_caveats": sorted(str(row) for row in packet.get("required_caveats") or []),
    }


def _collect_answer_values(arguments: Any, handles: set[str], numbers: set[str]) -> None:
    if isinstance(arguments, list):
        for value in arguments:
            _collect_answer_values(value, handles, numbers)
    elif isinstance(arguments, dict):
        if arguments.get("type") == "handle":
            handles.add(str(arguments.get("value") or ""))
        elif arguments.get("type") == "number":
            numbers.add(canonical_json(arguments.get("value") or {}))
        else:
            for value in arguments.values():
                _collect_answer_values(value, handles, numbers)


def _validate_hrl_reference(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or state.get("hrl_version") != HRL_VERSION:
        raise KernelProtocolFault("KERC_HRL_VERSION_INCOMPATIBLE", str(state.get("hrl_version") if isinstance(state, dict) else None), path="hrl_state")
    if not str(state.get("interaction_id") or "") or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(state.get("state_hash") or "")):
        raise KernelProtocolFault("KERC_HRL_REFERENCE_INVALID", "interaction_id/state_hash missing", path="hrl_state")


def _validate_concept_capsules(capsules: dict[str, dict[str, Any]]) -> None:
    for handle, capsule in capsules.items():
        if not re.fullmatch(r"@C[0-9]+", str(handle)) or not isinstance(capsule, dict):
            raise KernelProtocolFault("KERC_CONCEPT_CAPSULE_INVALID", str(handle), path="concept_capsules")
        identity = str(capsule.get("stable_identity") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", identity):
            raise KernelProtocolFault("KERC_CONCEPT_ID_INVALID", identity, path=f"concept_capsules.{handle}")
        if not capsule.get("provenance"):
            raise KernelProtocolFault("KERC_CONCEPT_PROVENANCE_MISSING", handle, path=f"concept_capsules.{handle}")


def _reject_reference_cycles(refs: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise KernelProtocolFault("KERC_NODE_REFERENCE_CYCLE", node_id, path="program.nodes")
        if node_id in visited:
            return
        visiting.add(node_id)
        for target in refs.get(node_id, set()):
            visit(target)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in refs:
        visit(node_id)


def _explicit_span_candidates(source: str, rows: Sequence[dict[str, Any]]) -> list[SpanCandidate]:
    output = []
    for index, row in enumerate(rows):
        path = f"explicit_spans[{index}]"
        if not isinstance(row, dict):
            raise KernelProtocolFault("KERC_EXPLICIT_SPAN_INVALID", str(row), path=path)
        start = _required_int(row, "start", path)
        end = _required_int(row, "end", path)
        object_type = str(row.get("object_type") or "EXACT_TEXT")
        copy_policy = str(row.get("copy_policy") or "EXACT")
        if not 0 <= start < end <= len(source) or object_type not in HANDLE_PREFIX_BY_TYPE or copy_policy not in COPY_POLICIES:
            raise KernelProtocolFault("KERC_EXPLICIT_SPAN_INVALID", canonical_json(row), path=path)
        output.append(SpanCandidate(start, end, object_type, copy_policy, 100, "explicit_user_or_caller_span"))
    return output


def _select_non_overlapping(candidates: Iterable[SpanCandidate]) -> list[SpanCandidate]:
    selected: list[SpanCandidate] = []
    for candidate in sorted(candidates, key=lambda row: (-row.priority, row.start, -(row.end - row.start), row.object_type)):
        if any(candidate.start < existing.end and candidate.end > existing.start for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda row: (row.start, row.end))


def _required_int(row: dict[str, Any], key: str, path: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KernelProtocolFault("KERC_INTEGER_FIELD_INVALID", f"{key}={value}", path=f"{path}.{key}")
    return value


def _normalization_form(text: str) -> str:
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        if unicodedata.is_normalized(form, text):
            return form
    return "NONE"


def _node_sort_key(node_id: str) -> tuple[int, str]:
    return (int(node_id[1:]), node_id)
