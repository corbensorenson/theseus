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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Iterable, Sequence

from kerc_residual_economics import (
    ALLOCATION_POLICY,
    CODEC_POLICY,
    PROMOTION_POLICY,
    UNIT_PACKET_POLICY,
    UNIT_PACKET_VERSION,
    ResidualEconomicsFault,
    build_residual_codec,
    build_residual_unit_packet,
    validate_residual_codec,
    validate_residual_unit_packet,
    validate_residual_unit_allocation_receipt,
    validate_structural_rate_distortion_allocation,
)
from vcm_semantic_memory import (
    HRLStateFault,
    create_hierarchical_residual_state,
    replay_hierarchical_residual_deltas,
    validate_hierarchical_residual_state,
)


POLICY = "project_theseus_kernel_packet_protocol_v1"
KERNEL_VERSION = "KE-1.0"
PACKET_VERSION = "KPP-1.2"
LEGACY_PACKET_VERSION = "KPP-1.1"
HRL_VERSION = "HRL-1.0"
CODEBOOK_VERSION = "KE-CODEBOOK-1.0"
SERIALIZATION_VERSION = "KE-SERIALIZATION-2.0"
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
    "prior_chunk_claim",
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
VALUE_TYPES = {
    "handle",
    "concept",
    "number",
    "quantity",
    "temporal",
    "text",
    "symbol",
    "node_ref",
    "list",
    "ambiguity",
    "byte_literal",
    "boolean",
    "null",
}
QUANTITY_KINDS = {
    "COUNT",
    "LENGTH",
    "MASS",
    "DURATION",
    "CURRENCY",
    "PERCENT",
    "TEMPERATURE",
    "AREA",
    "VOLUME",
    "RATE",
    "OTHER",
}
QUANTITY_RELATIONS = {"EXACT", "APPROX", "AT_LEAST", "AT_MOST", "BETWEEN", "UNKNOWN"}
TEMPORAL_KINDS = {"DATE", "TIME", "DATETIME", "DURATION", "INTERVAL", "RELATIVE"}
TEMPORAL_RELATIONS = {
    "AT",
    "BEFORE",
    "AFTER",
    "DURING",
    "OVERLAPS",
    "STARTS",
    "ENDS",
    "UNKNOWN",
}
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
LEARNED_RESIDUAL_TAG_NAMESPACE_CODES = {
    "ENTITY": "E",
    "ENTITY_MENTION": "EM",
    "ERST_RELATION": "DR",
    "MPQA": "Q",
    "CB": "B",
    "EVENT": "V",
    "EVENT_COREFERENCE": "VC",
}
LEARNED_RESIDUAL_EXACT_TAG_CODES = {
    "ERST_DISCOURSE_UNIT": "DU",
    "MPQA_RELATION_EXPRESSION": "QE",
    "MPQA_RELATION_SOURCE": "QS",
    "MPQA_RELATION_ATTITUDE": "QA",
    "MPQA_RELATION_TARGET": "QT",
}
LEARNED_PROGRAM_TRANSPORT_POLICY = "project_theseus_kerc_learned_program_tokens_v1"
LEARNED_ANSWER_TRANSPORT_POLICY = "project_theseus_kerc_learned_answer_tokens_v1"
KERC_HIERARCHICAL_CORE_POLICY = "project_theseus_kerc_hierarchical_core_v1"
KERC_CORE_CHUNK_MAX_NODES = 8
KERC_HIERARCHICAL_COMPILER_POLICY = "project_theseus_kerc_hierarchical_compiler_v1"
KERC_COMPILER_CHUNK_MAX_NODES = 8
KERC_PRIOR_CLAIM_CONTEXT_POLICY = "project_theseus_kerc_prior_claim_context_v1"
KERC_RESIDUAL_CHANNELS = ("interaction", "segment", "token", "exact")
KERC_FIDELITY_LABELS = {"semantic": 0, "faithful": 1, "lexical": 2, "exact": 3}
KERC_VERIFIER_DIMENSIONS = (
    "semantic_consistency",
    "protected_object_consistency",
    "numeric_value_consistency",
    "surface_fidelity",
    "answer_decision_consistency",
)
ANSWER_DECISION_POLICY = "project_theseus_kernel_answer_decision_v1"
ANSWER_DISPOSITION_ORDER = ("ANSWER", "PARTIAL", "CLARIFY", "ABSTAIN")
ANSWER_DISPOSITIONS = set(ANSWER_DISPOSITION_ORDER)
ANSWER_EVIDENCE_STATES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNVERIFIED",
    "AMBIGUOUS",
    "INSUFFICIENT_CONTEXT",
    "CONFLICTING_EVIDENCE",
}
ANSWER_UNCERTAINTY_STATES = {
    "RESOLVED",
    "AMBIGUOUS",
    "INSUFFICIENT_CONTEXT",
    "CONFLICTING",
}
SEMANTIC_EVIDENCE_TIERS = {
    "audited_human_gold": {
        "producer_kind": "human_annotation",
        "claim_authority": "decision_grade_reference",
        "model_derived": False,
        "allowed_splits": TRAINING_SPLITS,
        "maximum_optimizer_sampling_weight": 1.0,
        "verification_methods": {"human_dual_review"},
    },
    "licensed_human_task_gold": {
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
    payload = (
        value
        if isinstance(value, (bytes, bytearray))
        else canonical_json(value).encode("utf-8")
    )
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
        "learned_program_transport": LEARNED_PROGRAM_TRANSPORT_POLICY,
        "learned_answer_transport": LEARNED_ANSWER_TRANSPORT_POLICY,
        "hierarchical_compiler": {
            "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
            "maximum_target_nodes_per_chunk": KERC_COMPILER_CHUNK_MAX_NODES,
            "state_transport": (
                "previous_compact_program_plus_integrity_bound_accumulated_state"
            ),
            "termination": "learned_continuation_bit_with_bounded_chunk_limit",
            "final_program_validation": "exact_merged_program_revalidation",
            "chunking_credit": 0,
            "fallback_return_count": 0,
        },
        "protected_span_detection": {
            "input": "immutable_raw_source_plus_scoped_interaction_state",
            "learned_output": "typed_copy_policy_character_spans",
            "exact_bytes_and_hashes": "deterministically_materialized_from_immutable_source",
            "training_annotations_visible_to_prompt": False,
            "shared_with_surface_compiler": True,
            "fallback_return_count": 0,
        },
        "hierarchical_core": {
            "policy": KERC_HIERARCHICAL_CORE_POLICY,
            "maximum_target_nodes_per_chunk": KERC_CORE_CHUNK_MAX_NODES,
            "dependency_context": (
                "topological_direct_predecessor_stubs_plus_exact_compact_prior_claims"
            ),
            "prior_claim_transport": KERC_PRIOR_CLAIM_CONTEXT_POLICY,
            "answer_assembly": (
                "validated_claim_disjoint_union_with_decision_union_and_canonical_order"
            ),
            "chunking_credit": 0,
            "fallback_return_count": 0,
        },
        "raw_text_may_select_task_mode": False,
        "trusted_source_control_tokens": list(TRAINING_TASK_TAGS.values()),
        "residual_channels": list(KERC_RESIDUAL_CHANNELS),
        "residual_fidelity_labels": dict(KERC_FIDELITY_LABELS),
        "residual_economics": {
            "codec_policy": CODEC_POLICY,
            "allocation_policy": ALLOCATION_POLICY,
            "promotion_policy": PROMOTION_POLICY,
            "codec_model": "adaptive_order1_byte_fsm_conditioned_arithmetic_v1",
            "codec_channels": list(KERC_RESIDUAL_CHANNELS),
            "fidelity_target_authority": "independently_measured_rate_distortion_required",
            "bootstrap_fidelity_labels_are_optimality_evidence": False,
            "unit_packet_policy": UNIT_PACKET_POLICY,
            "unit_packet_version": UNIT_PACKET_VERSION,
            "packet_wide_fidelity_drives_training": False,
        },
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
            "authority_granularity": "per_objective",
            "unauthorized_objectives_are_not_compiled": True,
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
            "learned_protected_span_detection_and_surface_to_kernel_program",
            "independent_exact_object_materialization_and_kernel_program_validation",
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
    """Validate whether faithful KERC joins or a bounded proxy stays retired.

    A bounded negative result may remove the checkpoint-shaping KERC objective
    without erasing the exact-object and scoped-residual mechanisms that survived
    independently. The disposition is architecture selection, not a broad claim
    that KERC can never work.
    """

    required = cfg.get("required") is True
    disposition = cfg.get("disposition")
    # Minimal synthetic/unit fixtures may omit the canonical evidence packet.
    # Production configs carry it and are validated content-for-content below.
    if required and disposition is None:
        return {
            "state": "CANDIDATE_REQUIRED",
            "full_kerc_training_enabled": True,
            "general_kerc_falsification_claimed": False,
            "learned_capability_claimed": False,
            "retained_mechanisms": [],
        }
    if not isinstance(disposition, dict) or disposition.get("policy") != (
        TRAINING_DISPOSITION_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_MISSING",
            str(disposition),
            path="kernel_english_training.disposition",
        )
    expected_scalars = (
        {
            "state": "CANDIDATE_REQUIRED",
            "qualification_scope": "faithful_full_compiler_core_renderer_candidate",
            "basis": "adequacy_audit_reopened_after_toy_proxy",
            "full_kerc_training_enabled": True,
            "general_kerc_falsification_claimed": False,
            "learned_capability_claimed": False,
        }
        if required
        else {
            "state": "RETIRED_FROM_FIRST_LONG_RUN",
            "retirement_scope": "full_compiler_core_renderer_training_path_only",
            "evidence_scope": "bounded_authored_synthetic_campaign",
            "broad_efficiency_gate_passed": False,
            "full_kerc_training_enabled": False,
            "general_kerc_falsification_claimed": False,
            "learned_capability_claimed": False,
        }
    )
    for key, expected in expected_scalars.items():
        if disposition.get(key) != expected:
            raise KernelProtocolFault(
                "KERC_TRAINING_DISPOSITION_INVALID",
                f"{key}={disposition.get(key)!r}",
                path=f"kernel_english_training.disposition.{key}",
            )
    retained = tuple(disposition.get("retained_mechanisms") or ())
    expected_retained = (
        ()
        if required
        else (
            "protected_exact_object_path",
            "scoped_interaction_glossary_residual",
        )
    )
    if retained != expected_retained:
        raise KernelProtocolFault(
            "KERC_TRAINING_RETAINED_MECHANISMS_INVALID",
            canonical_json(retained),
            path="kernel_english_training.disposition.retained_mechanisms",
        )
    evidence_key = "superseded_proxy_evidence" if required else "evidence"
    evidence = disposition.get(evidence_key)
    evidence_path = f"kernel_english_training.disposition.{evidence_key}"
    if not isinstance(evidence, dict):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_EVIDENCE_MISSING",
            str(evidence),
            path=evidence_path,
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
            path=f"{evidence_path}.artifact_sha256",
        )
    for key, value in hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
            raise KernelProtocolFault(
                "KERC_TRAINING_DISPOSITION_HASH_INVALID",
                f"{key}={value}",
                path=f"{evidence_path}.artifact_sha256.{key}",
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
            path=f"{evidence_path}.denominators",
        )
    measurements = evidence.get("measurements")
    if not isinstance(measurements, dict):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_MEASUREMENTS_MISSING",
            str(measurements),
            path=f"{evidence_path}.measurements",
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
            path=f"{evidence_path}.measurements",
        )
    if not (
        float(measurements.get("packet_mean_bytes", 0.0))
        > float(measurements.get("best_simple_total_description_bytes", math.inf))
        > 0.0
    ):
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_COST_INVALID",
            canonical_json(measurements),
            path=f"{evidence_path}.measurements",
        )
    if int(measurements.get("attack_false_allow_count", -1)) != 1:
        raise KernelProtocolFault(
            "KERC_TRAINING_DISPOSITION_ATTACK_RESULT_INVALID",
            canonical_json(measurements),
            path=f"{evidence_path}.measurements.attack_false_allow_count",
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
        raise KernelProtocolFault(
            "KERC_TRAINING_SPLIT_INVALID", split, path="record.split"
        )
    if str(record.get("language") or "") != "en":
        raise KernelProtocolFault(
            "KERC_TRAINING_LANGUAGE_INVALID",
            str(record.get("language")),
            path="record.language",
        )
    source = str(record.get("source_text") or "")
    surface_target = str(record.get("surface_target") or "")
    if not source.strip() or not surface_target.strip():
        raise KernelProtocolFault(
            "KERC_TRAINING_TEXT_MISSING",
            "source and surface target are required",
            path="record",
        )
    provenance = (
        record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    )
    for key in ("source_id", "source_group", "license_spdx", "permitted_use"):
        if not str(provenance.get(key) or "").strip():
            raise KernelProtocolFault(
                "KERC_TRAINING_PROVENANCE_INCOMPLETE",
                key,
                path=f"record.provenance.{key}",
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
    if (
        verification.get("accepted") is not True
        or not str(verification.get("verifier_id") or "").strip()
    ):
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
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(verification.get("evidence_sha256") or "")
    ):
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

    hrl_state = (
        record.get("hrl_state") if isinstance(record.get("hrl_state"), dict) else {}
    )
    validate_hierarchical_residual_state(hrl_state)
    hrl_deltas = record.get("hrl_deltas")
    if not isinstance(hrl_deltas, list):
        raise KernelProtocolFault(
            "KERC_TRAINING_HRL_DELTAS_INVALID",
            str(type(hrl_deltas)),
            path="record.hrl_deltas",
        )
    initial_hrl_state = create_hierarchical_residual_state(
        str(hrl_state.get("interaction_id") or ""),
        scope=copy.deepcopy(hrl_state.get("scope") or {}),
        language=str((hrl_state.get("global") or {}).get("language") or "en"),
    )
    if hrl_deltas:
        try:
            replayed_hrl = replay_hierarchical_residual_deltas(
                initial_hrl_state, copy.deepcopy(hrl_deltas)
            )["state"]
        except HRLStateFault as exc:
            raise KernelProtocolFault(
                "KERC_TRAINING_HRL_REPLAY_INVALID",
                str(exc),
                path="record.hrl_deltas",
            ) from exc
        if replayed_hrl != hrl_state:
            raise KernelProtocolFault(
                "KERC_TRAINING_HRL_REPLAY_MISMATCH",
                str(hrl_state.get("state_hash")),
                path="record.hrl_deltas",
            )
    elif hrl_state != initial_hrl_state:
        raise KernelProtocolFault(
            "KERC_TRAINING_HRL_UNJOURNALED_STATE",
            str(hrl_state.get("state_hash")),
            path="record.hrl_state",
        )
    packet = (
        record.get("kernel_packet")
        if isinstance(record.get("kernel_packet"), dict)
        else {}
    )
    packet_replay = validate_kernel_packet(packet, local_hrl_state=hrl_state)
    residual_supervision = _validate_residual_supervision(record, packet=packet)
    expected_source = capture_source(source)
    if (packet.get("source") or {}).get("source_sha256") != expected_source[
        "source_sha256"
    ]:
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
        record.get("answer_packet")
        if isinstance(record.get("answer_packet"), dict)
        else {},
        protected_objects=packet.get("protected_objects") or {},
        concept_capsules=packet.get("concept_capsules") or {},
        correction_lattice=packet.get("correction_lattice") or {},
    )
    valid_realizations = validate_valid_realizations(
        record.get("valid_realizations"),
        canonical_surface_target=surface_target,
        canonical_answer_packet=answer,
        protected_objects=packet.get("protected_objects") or {},
        concept_capsules=packet.get("concept_capsules") or {},
        correction_lattice=packet.get("correction_lattice") or {},
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
    canonical["valid_realizations"] = valid_realizations
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

    if (
        not isinstance(evidence, dict)
        or evidence.get("policy") != SEMANTIC_SUPERVISION_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_SUPERVISION_POLICY_INVALID",
            str(
                evidence.get("policy") if isinstance(evidence, dict) else type(evidence)
            ),
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
    if any(
        not isinstance(objectives.get(objective), bool)
        for objective in TRAINING_OBJECTIVES
    ):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_OBJECTIVE_AUTHORITY_INVALID",
            canonical_json(objectives),
            path="record.semantic_supervision.objective_authority",
        )
    if not any(objectives.values()):
        raise KernelProtocolFault(
            "KERC_SEMANTIC_OBJECTIVE_AUTHORITY_EMPTY",
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


def validate_valid_realizations(
    realizations: Any,
    *,
    canonical_surface_target: str,
    canonical_answer_packet: dict[str, Any],
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    correction_lattice: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate source-bound alternative answers without exposing them to generation.

    A missing list means the ordinary single observed target.  Multiple targets must
    be independently source-bound, unique, and carry complete answer packets.  They
    are compiled as separate target views over one prompt and never enter the prompt.
    """

    if realizations is None:
        return [
            {
                "realization_id": "canonical",
                "surface_target": canonical_surface_target,
                "answer_packet": copy.deepcopy(canonical_answer_packet),
                "source_rank": 0,
                "human_source_bound": True,
                "source_message_sha256": "",
            }
        ]
    if not isinstance(realizations, list) or not 1 <= len(realizations) <= 8:
        raise KernelProtocolFault(
            "KERC_VALID_REALIZATIONS_INVALID",
            str(type(realizations)),
            path="record.valid_realizations",
        )
    canonical: list[dict[str, Any]] = []
    ids: set[str] = set()
    targets: set[str] = set()
    for index, row in enumerate(realizations):
        path = f"record.valid_realizations[{index}]"
        if not isinstance(row, dict):
            raise KernelProtocolFault(
                "KERC_VALID_REALIZATION_INVALID", str(type(row)), path=path
            )
        realization_id = str(row.get("realization_id") or "")
        surface_target = str(row.get("surface_target") or "")
        if (
            not realization_id
            or realization_id in ids
            or not surface_target.strip()
            or surface_target in targets
            or row.get("human_source_bound") is not True
        ):
            raise KernelProtocolFault(
                "KERC_VALID_REALIZATION_IDENTITY_INVALID",
                realization_id,
                path=path,
            )
        rank = row.get("source_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise KernelProtocolFault(
                "KERC_VALID_REALIZATION_RANK_INVALID",
                str(rank),
                path=f"{path}.source_rank",
            )
        answer_packet = validate_answer_packet_against_context(
            row.get("answer_packet")
            if isinstance(row.get("answer_packet"), dict)
            else {},
            protected_objects=protected_objects,
            concept_capsules=concept_capsules,
            correction_lattice=correction_lattice,
        )
        ids.add(realization_id)
        targets.add(surface_target)
        canonical.append(
            {
                "realization_id": realization_id,
                "surface_target": surface_target,
                "answer_packet": answer_packet,
                "source_rank": rank,
                "human_source_bound": True,
                "source_message_sha256": str(row.get("source_message_sha256") or ""),
            }
        )
        if canonical[-1]["source_message_sha256"] and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", canonical[-1]["source_message_sha256"]
        ):
            raise KernelProtocolFault(
                "KERC_VALID_REALIZATION_SOURCE_HASH_INVALID",
                canonical[-1]["source_message_sha256"],
                path=f"{path}.source_message_sha256",
            )
    if (
        canonical[0]["surface_target"] != canonical_surface_target
        or canonical[0]["answer_packet"] != canonical_answer_packet
    ):
        raise KernelProtocolFault(
            "KERC_VALID_REALIZATION_CANONICAL_MISMATCH",
            canonical[0]["realization_id"],
            path="record.valid_realizations[0]",
        )
    if len(canonical) > 1 and sorted(row["source_rank"] for row in canonical) != list(
        range(len(canonical))
    ):
        raise KernelProtocolFault(
            "KERC_VALID_REALIZATION_RANK_SET_INVALID",
            canonical_json([row["source_rank"] for row in canonical]),
            path="record.valid_realizations",
        )
    return canonical


def training_semantic_payload_sha256(record: dict[str, Any]) -> str:
    packet = (
        record.get("kernel_packet")
        if isinstance(record.get("kernel_packet"), dict)
        else {}
    )
    provenance = (
        record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    )
    answer = validate_answer_packet(
        record.get("answer_packet")
        if isinstance(record.get("answer_packet"), dict)
        else {}
    )
    valid_realizations = validate_valid_realizations(
        record.get("valid_realizations"),
        canonical_surface_target=str(record.get("surface_target") or ""),
        canonical_answer_packet=answer,
        protected_objects=packet.get("protected_objects") or {},
        concept_capsules=packet.get("concept_capsules") or {},
        correction_lattice=packet.get("correction_lattice") or {},
    )
    return stable_hash(
        {
            "source_text": str(record.get("source_text") or ""),
            "kernel_packet_sha256": str(packet.get("packet_sha256") or ""),
            "answer_packet": answer,
            "surface_target": str(record.get("surface_target") or ""),
            "valid_realizations": valid_realizations,
            "residual_supervision": record.get("residual_supervision") or {},
            "semantic_supervision": record.get("semantic_supervision") or {},
            "source_id": str(provenance.get("source_id") or ""),
            "source_group": str(provenance.get("source_group") or ""),
            "license_spdx": str(provenance.get("license_spdx") or ""),
            "permitted_use": str(provenance.get("permitted_use") or ""),
        }
    )


def learned_protected_object_view(
    protected_objects: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose only generation-relevant object fields to learned KERC stages.

    Packet hashes, authority, provenance, byte offsets, and audit metadata remain
    in the deterministic packet/evidence plane. The model receives the typed
    handle, copy rule, exact authorized bytes, and character alignment needed to
    compile, reason, and render without learning on audit-envelope repetition.
    """

    output: dict[str, dict[str, Any]] = {}
    for handle, row in sorted(protected_objects.items()):
        source_span = (
            row.get("source_span") if isinstance(row.get("source_span"), dict) else {}
        )
        output[str(handle)] = {
            "object_type": str(row.get("object_type") or ""),
            "copy_policy": str(row.get("copy_policy") or ""),
            "inline_bytes_b64": row.get("inline_bytes_b64"),
            "source_span": {
                "character_start": int(source_span.get("character_start", -1)),
                "character_end": int(source_span.get("character_end", -1)),
            },
        }
    return output


def learned_protected_span_view(
    protected_objects: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project exact objects into the source-grounded span target learned by KERC.

    The model predicts only type, copy policy, and character alignment. Exact bytes,
    hashes, authority, and provenance are reconstructed from the immutable source by
    the deterministic object owner after generation.
    """

    output: list[dict[str, Any]] = []
    for handle, row in sorted(
        protected_objects.items(),
        key=lambda item: (
            int((item[1].get("source_span") or {}).get("character_start", -1)),
            int((item[1].get("source_span") or {}).get("character_end", -1)),
            str(item[0]),
        ),
    ):
        source_span = (
            row.get("source_span") if isinstance(row.get("source_span"), dict) else {}
        )
        output.append(
            {
                "handle": str(handle),
                "object_type": str(row.get("object_type") or ""),
                "copy_policy": str(row.get("copy_policy") or ""),
                "character_start": int(source_span.get("character_start", -1)),
                "character_end": int(source_span.get("character_end", -1)),
            }
        )
    return output


def materialize_learned_protected_objects(
    source: str, declarations: Any
) -> dict[str, dict[str, Any]]:
    """Turn model-emitted spans into exact source-owned protected objects.

    The learned channel never emits bytes or hashes. This function validates the
    complete declaration set, extracts exact bytes from ``source``, assigns handles
    with the canonical object owner, and requires byte-for-byte declaration replay.
    """

    if not isinstance(declarations, list) or len(declarations) > 256:
        raise KernelProtocolFault(
            "KERC_LEARNED_PROTECTED_SPANS_INVALID",
            str(
                type(declarations)
                if not isinstance(declarations, list)
                else len(declarations)
            ),
            path="compiler_output.protected_objects",
        )
    if len(canonical_json(declarations).encode("utf-8")) > 65536:
        raise KernelProtocolFault(
            "KERC_LEARNED_PROTECTED_SPANS_BUDGET_EXCEEDED",
            str(len(canonical_json(declarations).encode("utf-8"))),
            path="compiler_output.protected_objects",
        )
    expected_fields = {
        "handle",
        "object_type",
        "copy_policy",
        "character_start",
        "character_end",
    }
    normalized: list[dict[str, Any]] = []
    seen_handles: set[str] = set()
    spans: list[dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        path = f"compiler_output.protected_objects[{index}]"
        if not isinstance(declaration, dict) or set(declaration) != expected_fields:
            raise KernelProtocolFault(
                "KERC_LEARNED_PROTECTED_SPAN_SCHEMA_INVALID",
                canonical_json(declaration),
                path=path,
            )
        handle = str(declaration.get("handle") or "")
        object_type = str(declaration.get("object_type") or "")
        copy_policy = str(declaration.get("copy_policy") or "")
        start = declaration.get("character_start")
        end = declaration.get("character_end")
        prefix = HANDLE_PREFIX_BY_TYPE.get(object_type)
        if (
            handle in seen_handles
            or prefix is None
            or not re.fullmatch(rf"@{re.escape(prefix)}[1-9][0-9]*", handle)
            or copy_policy not in COPY_POLICIES
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(source)
        ):
            raise KernelProtocolFault(
                "KERC_LEARNED_PROTECTED_SPAN_INVALID",
                canonical_json(declaration),
                path=path,
            )
        seen_handles.add(handle)
        normalized.append(
            {
                "handle": handle,
                "object_type": object_type,
                "copy_policy": copy_policy,
                "character_start": start,
                "character_end": end,
            }
        )
        spans.append(
            {
                "start": start,
                "end": end,
                "object_type": object_type,
                "copy_policy": copy_policy,
            }
        )
    if normalized != sorted(
        normalized,
        key=lambda row: (row["character_start"], row["character_end"], row["handle"]),
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_PROTECTED_SPAN_ORDER_INVALID",
            canonical_json(normalized),
            path="compiler_output.protected_objects",
        )
    materialized = extract_protected_objects(source, explicit_spans=spans)[
        "protected_objects"
    ]
    replayed = learned_protected_span_view(materialized)
    if replayed != normalized:
        raise KernelProtocolFault(
            "KERC_LEARNED_PROTECTED_SPAN_REPLAY_MISMATCH",
            canonical_json({"declared": normalized, "replayed": replayed}),
            path="compiler_output.protected_objects",
        )
    return materialized


def learned_concept_capsule_view(
    concept_capsules: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose capsule semantics without evidence-plane authority metadata.

    Stable registry identities and provenance receipts are assigned and checked
    outside the learned channel. Requiring a model to reproduce source IDs or
    annotation hashes would create an impossible target; allowing it to mint
    them would create an authority injection path.
    """

    output: dict[str, dict[str, Any]] = {}
    for handle, capsule in sorted(concept_capsules.items()):
        if not re.fullmatch(r"@C[0-9]+", str(handle)) or not isinstance(capsule, dict):
            raise KernelProtocolFault(
                "KERC_CONCEPT_CAPSULE_INVALID", str(handle), path="concept_capsules"
            )
        output[str(handle)] = {
            str(key): copy.deepcopy(value)
            for key, value in capsule.items()
            if key
            not in {
                "stable_identity",
                "provenance",
                "registry_resolution",
                "registry_semantics",
            }
        }
    return output


def materialize_learned_concept_capsules(
    learned_capsules: dict[str, dict[str, Any]],
    *,
    concept_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attach local or registry identity after validating learned capsule fields.

    A model can request registry resolution but cannot select a candidate or
    mint authority fields. Only an unambiguous deterministic registry result
    receives the pre-existing stable identity. Ambiguous and unresolved
    requests retain packet-local identity and their explicit status.
    """

    if len(learned_capsules) > 64:
        raise KernelProtocolFault(
            "KERC_LEARNED_CONCEPT_CAPSULE_BUDGET_EXCEEDED",
            str(len(learned_capsules)),
            path="compiler_output.concept_capsules",
        )
    output: dict[str, dict[str, Any]] = {}
    for handle, capsule in sorted(learned_capsules.items()):
        if not re.fullmatch(r"@C[0-9]+", str(handle)) or not isinstance(capsule, dict):
            raise KernelProtocolFault(
                "KERC_LEARNED_CONCEPT_CAPSULE_INVALID",
                str(handle),
                path="compiler_output.concept_capsules",
            )
        forbidden = sorted(
            set(capsule)
            & {
                "stable_identity",
                "provenance",
                "registry_resolution",
                "registry_semantics",
            }
        )
        if forbidden:
            raise KernelProtocolFault(
                "KERC_LEARNED_CONCEPT_AUTHORITY_FORBIDDEN",
                canonical_json(forbidden),
                path=f"compiler_output.concept_capsules.{handle}",
            )
        learned = copy.deepcopy(capsule)
        resolution_request = learned.get("resolution_request")
        identity = f"local.concept.{str(handle)[2:]}"
        provenance: dict[str, Any] = {
            "source": "learned_compiler_output_v1",
            "scope": "packet_local",
            "registry_promotion_allowed": False,
        }
        registry_resolution: dict[str, Any] | None = None
        registry_semantics: dict[str, Any] | None = None
        if resolution_request is not None:
            if not isinstance(resolution_request, dict):
                raise KernelProtocolFault(
                    "KERC_CONCEPT_RESOLUTION_REQUEST_INVALID",
                    str(type(resolution_request)),
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            forbidden_request_fields = sorted(
                set(resolution_request) - {"surface", "pos", "sense"}
            )
            surface = str(resolution_request.get("surface") or "").strip()
            pos = str(resolution_request.get("pos") or "")
            sense = str(resolution_request.get("sense") or "")
            if (
                forbidden_request_fields
                or not surface
                or len(surface.encode("utf-8")) > 512
                or pos not in {"", "a", "n", "r", "s", "v"}
                or len(sense.encode("utf-8")) > 256
            ):
                raise KernelProtocolFault(
                    "KERC_CONCEPT_RESOLUTION_REQUEST_INVALID",
                    canonical_json(
                        {
                            "forbidden_fields": forbidden_request_fields,
                            "surface_present": bool(surface),
                            "surface_bytes": len(surface.encode("utf-8")),
                            "pos": pos,
                            "sense_bytes": len(sense.encode("utf-8")),
                        }
                    ),
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            if concept_resolver is None:
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_UNAVAILABLE",
                    str(handle),
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            try:
                resolution = concept_resolver(copy.deepcopy(resolution_request))
            except Exception as exc:
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_TOOL_FAULT",
                    str(exc),
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                ) from exc
            status = (
                str(resolution.get("status") or "")
                if isinstance(resolution, dict)
                else ""
            )
            candidates = (
                resolution.get("candidates") if isinstance(resolution, dict) else None
            )
            candidate_count = (
                int(resolution.get("candidate_count", -1))
                if isinstance(resolution, dict)
                else -1
            )
            candidates_truncated = (
                bool(resolution.get("candidates_truncated"))
                if isinstance(resolution, dict)
                else False
            )
            authority_basis = (
                str(resolution.get("authority_basis") or "")
                if isinstance(resolution, dict)
                else ""
            )
            if (
                status not in {"RESOLVED", "AMBIGUOUS", "UNRESOLVED"}
                or authority_basis != "exact_normalized_surface_has_one_global_identity"
                or not isinstance(candidates, list)
                or len(candidates) > 32
                or candidate_count < len(candidates)
                or candidates_truncated != (candidate_count > len(candidates))
                or (
                    status == "RESOLVED"
                    and (candidate_count != 1 or candidates_truncated)
                )
                or (status == "AMBIGUOUS" and candidate_count < 2)
                or (status == "UNRESOLVED" and (candidate_count != 0 or candidates))
            ):
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_RESPONSE_INVALID",
                    status,
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            candidate_identities: list[str] = []
            for candidate in candidates:
                candidate_identity = (
                    str(candidate.get("stable_identity") or "")
                    if isinstance(candidate, dict)
                    else ""
                )
                if not re.fullmatch(
                    r"conceptnet\.uri\.[0-9a-f]{64}", candidate_identity
                ):
                    raise KernelProtocolFault(
                        "KERC_CONCEPT_REGISTRY_IDENTITY_INVALID",
                        candidate_identity,
                        path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                    )
                candidate_identities.append(candidate_identity)
            if len(candidate_identities) != len(set(candidate_identities)):
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_RESPONSE_INVALID",
                    "duplicate candidate identity",
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            selected_identity = str(resolution.get("selected_identity") or "")
            if status == "RESOLVED":
                if len(candidates) != 1 or selected_identity != str(
                    candidates[0].get("stable_identity") or ""
                ):
                    raise KernelProtocolFault(
                        "KERC_CONCEPT_REGISTRY_RESPONSE_INVALID",
                        "resolved candidate mismatch",
                        path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                    )
                if not re.fullmatch(
                    r"conceptnet\.uri\.[0-9a-f]{64}", selected_identity
                ):
                    raise KernelProtocolFault(
                        "KERC_CONCEPT_REGISTRY_IDENTITY_INVALID",
                        selected_identity,
                        path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                    )
                identity = selected_identity
                registry_semantics = copy.deepcopy(candidates[0])
                provenance = {
                    "source": "kerc_concept_registry_v1",
                    "scope": "cross_document_registered",
                    "registry_promotion_allowed": False,
                    "registry_schema_version": str(
                        resolution.get("registry_schema_version") or ""
                    ),
                }
            elif selected_identity:
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_RESPONSE_INVALID",
                    "nonresolved response selected an identity",
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            registry_resolution = {
                "status": status,
                "request_sha256": stable_hash(resolution_request),
                "candidate_count": candidate_count,
                "candidates_truncated": candidates_truncated,
                "candidate_identities": candidate_identities,
                "authority_basis": authority_basis,
                "non_authoritative_hint_match_count": int(
                    resolution.get("non_authoritative_hint_match_count", 0)
                ),
                "external_inference_calls": int(
                    resolution.get("external_inference_calls", -1)
                ),
            }
            if registry_resolution["external_inference_calls"] != 0:
                raise KernelProtocolFault(
                    "KERC_CONCEPT_REGISTRY_EXTERNAL_INFERENCE_FORBIDDEN",
                    str(registry_resolution["external_inference_calls"]),
                    path=f"compiler_output.concept_capsules.{handle}.resolution_request",
                )
            provenance["resolution_status"] = status
        materialized = {
            "stable_identity": identity,
            "provenance": provenance,
            **learned,
        }
        if registry_resolution is not None:
            materialized["registry_resolution"] = registry_resolution
        if registry_semantics is not None:
            materialized["registry_semantics"] = registry_semantics
        output[str(handle)] = materialized
    if len(canonical_json(output).encode("utf-8")) > 65536:
        raise KernelProtocolFault(
            "KERC_LEARNED_CONCEPT_CAPSULE_BUDGET_EXCEEDED",
            str(len(canonical_json(output).encode("utf-8"))),
            path="compiler_output.concept_capsules",
        )
    _validate_concept_capsules(output)
    return output


def learned_interaction_residual_view(hrl_state: dict[str, Any]) -> list[list[Any]]:
    """Expose bounded, segment-scoped VCM state without widening authority."""

    _validate_hrl_reference(hrl_state)
    if hrl_state.get("cross_user_reuse_allowed") is not False:
        raise KernelProtocolFault(
            "KERC_INTERACTION_CROSS_USER_REUSE_FORBIDDEN",
            str(hrl_state.get("cross_user_reuse_allowed")),
            path="hrl_state.cross_user_reuse_allowed",
        )
    scope = hrl_state.get("scope") if isinstance(hrl_state.get("scope"), dict) else {}
    if scope.get("privacy") != "private_local":
        raise KernelProtocolFault(
            "KERC_INTERACTION_PRIVACY_INVALID",
            str(scope.get("privacy")),
            path="hrl_state.scope.privacy",
        )
    compiled: list[list[Any]] = []
    segments = (
        hrl_state.get("segments") if isinstance(hrl_state.get("segments"), dict) else {}
    )
    for segment_id, segment in sorted(segments.items()):
        entries = segment.get("entries") if isinstance(segment, dict) else None
        if not isinstance(entries, dict):
            raise KernelProtocolFault(
                "KERC_INTERACTION_SEGMENT_INVALID",
                str(segment_id),
                path=f"hrl_state.segments.{segment_id}",
            )
        for key, row in sorted(entries.items()):
            if not isinstance(row, dict) or row.get("privacy") != "interaction_private":
                raise KernelProtocolFault(
                    "KERC_INTERACTION_ENTRY_PRIVACY_INVALID",
                    str(key),
                    path=f"hrl_state.segments.{segment_id}.entries.{key}",
                )
            if row.get("authority") not in {"document", "user", "system"}:
                raise KernelProtocolFault(
                    "KERC_INTERACTION_ENTRY_AUTHORITY_INVALID",
                    str(row.get("authority")),
                    path=f"hrl_state.segments.{segment_id}.entries.{key}",
                )
            compiled.append(
                [str(segment_id), str(key), copy.deepcopy(row.get("value"))]
            )
    if len(compiled) > 64 or len(canonical_json(compiled).encode("utf-8")) > 8192:
        raise KernelProtocolFault(
            "KERC_INTERACTION_VIEW_BUDGET_EXCEEDED",
            f"entries={len(compiled)} bytes={len(canonical_json(compiled).encode('utf-8'))}",
            path="hrl_state.segments",
        )
    return compiled


def learned_residual_view(
    residual: dict[str, Any], *, hrl_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Compile governed residual state into the compact model-visible ABI.

    Authority, provenance, lifecycle, and full state hashes remain in VCM and the
    packet evidence plane.  The learned stages receive only the realization
    choices and alignments they can act on.
    """

    segment = residual.get("segment_frame") or {}
    composite_frames = segment.get("frames") or []
    if composite_frames:
        roles = list(
            dict.fromkeys(
                str(role)
                for frame in composite_frames
                for role in (frame.get("frame_roles") or [])
            )
        )
        learned_segment: list[Any] = [
            "COMPOSITE",
            [
                [
                    str(frame.get("node_id") or ""),
                    str(frame.get("claim_id") or ""),
                    str(frame.get("frame_name") or ""),
                    str(frame.get("lexical_unit") or ""),
                    list(frame.get("target_spans") or []),
                    list(frame.get("frame_roles") or []),
                ]
                for frame in composite_frames
            ],
        ]
    else:
        roles = list(segment.get("frame_roles") or [])
        learned_segment = (
            [
                str(segment.get("frame_name") or ""),
                str(segment.get("lexical_unit") or ""),
                list(segment.get("target_spans") or []),
                roles,
            ]
            if segment
            else []
        )
    compiled_tags: list[list[Any]] = []
    for row in residual.get("token_tags") or []:
        tag = str(row.get("tag") or "")
        start, end = (int(value) for value in (row.get("source_span") or [0, 0]))
        if tag.startswith("FRAME_TARGET:"):
            compiled_tags.append(["T", start, end])
        elif tag.startswith("FRAME_ROLE:"):
            role = tag.split(":", 1)[1]
            if role not in roles:
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_ROLE_UNKNOWN",
                    role,
                    path="residual.token_tags",
                )
            compiled_tags.append(["R", roles.index(role), start, end])
        elif tag in LEARNED_RESIDUAL_EXACT_TAG_CODES:
            compiled_tags.append([LEARNED_RESIDUAL_EXACT_TAG_CODES[tag], start, end])
        elif ":" in tag:
            namespace, payload = tag.split(":", 1)
            code = LEARNED_RESIDUAL_TAG_NAMESPACE_CODES.get(namespace)
            if code is None or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", payload):
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_TAG_UNKNOWN",
                    tag,
                    path="residual.token_tags",
                )
            compiled_tags.append([code, payload, start, end])
        else:
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_TAG_UNKNOWN", tag, path="residual.token_tags"
            )
    interaction = (
        learned_interaction_residual_view(hrl_state)
        if isinstance(hrl_state, dict)
        else []
    )
    if int((hrl_state or {}).get("sequence") or 0) > 0 and not interaction:
        raise KernelProtocolFault(
            "KERC_INTERACTION_STATE_NOT_VISIBLE",
            str((hrl_state or {}).get("state_hash")),
            path="hrl_state.segments",
        )
    return {
        "mode": str(residual.get("mode") or ""),
        "unit_fidelity": [
            [str(unit.get("unit_id") or ""), str(unit.get("selected_fidelity") or "")]
            for unit in (residual.get("unit_packet") or {}).get("units") or []
        ],
        "interaction": interaction,
        "segment": learned_segment,
        "tokens": compiled_tags,
        "exact_handles": list(residual.get("exact_object_handles") or []),
    }


def validate_learned_residual_view(
    view: Any,
    *,
    source_character_length: int,
    protected_objects: dict[str, dict[str, Any]],
    hrl_state: dict[str, Any],
) -> dict[str, Any]:
    """Validate the model-emitted residual ABI without granting evidence authority."""

    required = {
        "mode",
        "unit_fidelity",
        "interaction",
        "segment",
        "tokens",
        "exact_handles",
    }
    if not isinstance(view, dict) or set(view) != required:
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_SCHEMA_INVALID",
            canonical_json(view),
            path="compiler_output.residual",
        )
    if (
        view["mode"] != "SOURCE_RECONSTRUCTION"
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_POLICY_INVALID",
            canonical_json({"mode": view["mode"]}),
            path="compiler_output.residual",
        )
    unit_fidelity = view["unit_fidelity"]
    if (
        not isinstance(unit_fidelity, list)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not re.fullmatch(r"ru:[0-9a-f]{24}", str(row[0]))
            or str(row[1]) not in FIDELITY_MODES
            for row in unit_fidelity
        )
        or len({str(row[0]) for row in unit_fidelity}) != len(unit_fidelity)
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_UNIT_FIDELITY_INVALID",
            canonical_json(unit_fidelity),
            path="compiler_output.residual.unit_fidelity",
        )
    expected_interaction = learned_interaction_residual_view(hrl_state)
    if view["interaction"] != expected_interaction:
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_INTERACTION_MISMATCH",
            canonical_json(view["interaction"]),
            path="compiler_output.residual.interaction",
        )
    exact_handles = view["exact_handles"]
    if (
        not isinstance(exact_handles, list)
        or len(exact_handles) != len(set(exact_handles))
        or not set(exact_handles) <= set(protected_objects)
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_EXACT_HANDLE_INVALID",
            canonical_json(exact_handles),
            path="compiler_output.residual.exact_handles",
        )

    segment = view["segment"]
    roles: list[str] = []
    if segment:
        if not isinstance(segment, list):
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_SEGMENT_INVALID",
                canonical_json(segment),
                path="compiler_output.residual.segment",
            )
        if segment[0] == "COMPOSITE":
            if len(segment) != 2 or not isinstance(segment[1], list):
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_SEGMENT_INVALID",
                    canonical_json(segment),
                    path="compiler_output.residual.segment",
                )
            frames = segment[1]
            for frame in frames:
                if not isinstance(frame, list) or len(frame) != 6:
                    raise KernelProtocolFault(
                        "KERC_LEARNED_RESIDUAL_FRAME_INVALID",
                        canonical_json(frame),
                        path="compiler_output.residual.segment",
                    )
                roles.extend(str(value) for value in frame[5])
                for start, end in frame[4]:
                    if not 0 <= int(start) < int(end) <= source_character_length:
                        raise KernelProtocolFault(
                            "KERC_LEARNED_RESIDUAL_SPAN_INVALID",
                            canonical_json([start, end]),
                            path="compiler_output.residual.segment",
                        )
        else:
            if (
                len(segment) != 4
                or not all(isinstance(value, str) for value in segment[:2])
                or not isinstance(segment[2], list)
                or not isinstance(segment[3], list)
            ):
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_SEGMENT_INVALID",
                    canonical_json(segment),
                    path="compiler_output.residual.segment",
                )
            roles = [str(value) for value in segment[3]]
            for start, end in segment[2]:
                if not 0 <= int(start) < int(end) <= source_character_length:
                    raise KernelProtocolFault(
                        "KERC_LEARNED_RESIDUAL_SPAN_INVALID",
                        canonical_json([start, end]),
                        path="compiler_output.residual.segment",
                    )

    allowed_codes = {
        "T",
        "R",
        *LEARNED_RESIDUAL_EXACT_TAG_CODES.values(),
        *LEARNED_RESIDUAL_TAG_NAMESPACE_CODES.values(),
    }
    tokens = view["tokens"]
    if not isinstance(tokens, list):
        raise KernelProtocolFault(
            "KERC_LEARNED_RESIDUAL_TOKENS_INVALID",
            canonical_json(tokens),
            path="compiler_output.residual.tokens",
        )
    for index, row in enumerate(tokens):
        if (
            not isinstance(row, list)
            or len(row) not in (3, 4)
            or row[0] not in allowed_codes
        ):
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_TOKEN_INVALID",
                canonical_json(row),
                path=f"compiler_output.residual.tokens[{index}]",
            )
        start, end = row[-2:]
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= source_character_length
        ):
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_SPAN_INVALID",
                canonical_json(row),
                path=f"compiler_output.residual.tokens[{index}]",
            )
        if row[0] == "R" and (
            len(row) != 4
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or not 0 <= row[1] < len(roles)
        ):
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_ROLE_INVALID",
                canonical_json(row),
                path=f"compiler_output.residual.tokens[{index}]",
            )
    return copy.deepcopy(view)


_LEARNED_SPACE_CODE = {"V_K": "K", "V_P": "P", "V_S": "S"}
_LEARNED_CODE_SPACE = {code: space for space, code in _LEARNED_SPACE_CODE.items()}


def _learned_token_view(rows: Sequence[dict[str, str]]) -> list[str]:
    compact: list[str] = []
    for index, row in enumerate(rows):
        validated = _validate_token(row, path=f"learned_tokens[{index}]")
        compact.append(_LEARNED_SPACE_CODE[validated["space"]] + validated["token"])
    return compact


def _materialize_learned_token_view(
    rows: Any, *, path: str, maximum_tokens: int = 16384
) -> list[dict[str, str]]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum_tokens:
        raise KernelProtocolFault(
            "KERC_LEARNED_TOKEN_STREAM_INVALID", str(type(rows)), path=path
        )
    materialized: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, str)
            or len(row) < 2
            or row[0] not in _LEARNED_CODE_SPACE
        ):
            raise KernelProtocolFault(
                "KERC_LEARNED_TOKEN_INVALID",
                canonical_json(row),
                path=f"{path}[{index}]",
            )
        materialized.append(
            _validate_token(
                {"space": _LEARNED_CODE_SPACE[row[0]], "token": row[1:]},
                path=f"{path}[{index}]",
            )
        )
    return materialized


def learned_kernel_program_view(packet: dict[str, Any]) -> dict[str, Any]:
    """Project an authoritative program into its exact compact model ABI."""

    serialization = packet.get("serialization") or {}
    if serialization.get("macro_registry"):
        raise KernelProtocolFault(
            "KERC_LEARNED_PROGRAM_MACRO_UNSUPPORTED",
            str(len(serialization.get("macro_registry") or [])),
            path="packet.serialization.macro_registry",
        )
    expanded = serialization.get("expanded_tokens")
    if not isinstance(expanded, list) or stable_hash(expanded) != serialization.get(
        "expanded_sha256"
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_PROGRAM_SERIALIZATION_INVALID",
            str(serialization.get("expanded_sha256")),
            path="packet.serialization",
        )
    return {
        "policy": LEARNED_PROGRAM_TRANSPORT_POLICY,
        "tokens": _learned_token_view(expanded),
    }


def learned_kernel_program_view_from_program(
    canonical_program: dict[str, Any],
) -> dict[str, Any]:
    serialization = serialize_kernel_program(canonical_program)
    return {
        "policy": LEARNED_PROGRAM_TRANSPORT_POLICY,
        "tokens": _learned_token_view(serialization["expanded_tokens"]),
    }


def materialize_learned_kernel_program(
    view: Any,
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_character_length: int,
) -> dict[str, Any]:
    """Decode learned program tokens and independently revalidate exact semantics."""

    if (
        not isinstance(view, dict)
        or set(view) != {"policy", "tokens"}
        or view.get("policy") != LEARNED_PROGRAM_TRANSPORT_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_PROGRAM_TRANSPORT_INVALID",
            canonical_json(view),
            path="program",
        )
    expanded = _materialize_learned_token_view(view["tokens"], path="program.tokens")
    envelope = {
        "policy": "project_theseus_kernel_three_code_space_serialization_v2",
        "serialization_version": SERIALIZATION_VERSION,
        "expanded_tokens": expanded,
        "compact_tokens": expanded,
        "macro_registry": [],
        "expanded_sha256": stable_hash(expanded),
        "compact_sha256": stable_hash(expanded),
    }
    return deserialize_kernel_program(
        envelope,
        protected_objects=protected_objects,
        concept_capsules=concept_capsules,
        source_character_length=source_character_length,
    )["canonical_program"]


def learned_answer_packet_view(packet: dict[str, Any]) -> dict[str, Any]:
    """Encode a canonical answer as a compact, exact typed token stream."""

    canonical = validate_answer_packet(packet, require_explicit_decision=True)
    tokens: list[dict[str, str]] = [token("V_P", "ANSWER_VERSION:1")]
    for claim in canonical["claims"]:
        tokens.extend(
            [
                token("V_P", "CLAIM_BEGIN"),
                token("V_P", f"CLAIM_ID:{claim['claim_id']}"),
                token("V_K", f"PRED:{claim['predicate']}"),
                token("V_K", f"MOD:{claim['modality']}"),
                token("V_K", f"POL:{claim['polarity']}"),
                token("V_K", f"QUANT:{claim['quantifier']}"),
                token("V_P", f"CONF:{float(claim['confidence']):.17g}"),
            ]
        )
        for argument in claim["arguments"]:
            tokens.append(token("V_K", f"ROLE:{argument['role']}"))
            tokens.extend(_serialize_value(argument["value"]))
        claim_metadata = {
            key: copy.deepcopy(value)
            for key, value in claim.items()
            if key
            not in {
                "claim_id",
                "predicate",
                "modality",
                "polarity",
                "quantifier",
                "confidence",
                "arguments",
            }
        }
        if claim_metadata:
            tokens.append(
                token("V_P", f"CLAIM_METADATA:{canonical_json(claim_metadata)}")
            )
        tokens.append(token("V_P", "CLAIM_END"))
    decision = canonical["decision"]
    tokens.extend(
        [
            token("V_P", "DECISION_BEGIN"),
            token("V_K", f"DISPOSITION:{decision['disposition']}"),
            token("V_K", f"EVIDENCE:{decision['evidence_status']}"),
            token("V_K", f"UNCERTAINTY:{decision['uncertainty_state']}"),
            token("V_P", f"DECISION_CONF:{float(decision['confidence']):.17g}"),
        ]
    )
    for claim_id in decision["controlling_claim_ids"]:
        tokens.append(token("V_P", f"CONTROLLING:{claim_id}"))
    for ambiguity_id in decision["unresolved_ambiguity_ids"]:
        tokens.append(token("V_P", f"AMBIGUITY_ID:{ambiguity_id}"))
    tokens.append(token("V_P", "DECISION_END"))
    for term in canonical.get("required_terms") or []:
        tokens.append(token("V_P", f"REQUIRED_TERM:{canonical_json(term)}"))
    for caveat in canonical.get("required_caveats") or []:
        tokens.append(token("V_P", f"REQUIRED_CAVEAT:{canonical_json(caveat)}"))
    tokens.append(token("V_P", f"STYLE:{canonical_json(canonical.get('style') or {})}"))
    tokens.append(token("V_P", "ANSWER_END"))
    return {
        "policy": LEARNED_ANSWER_TRANSPORT_POLICY,
        "tokens": _learned_token_view(tokens),
    }


def learned_prior_claim_context_view(
    claims: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Encode verified prior claims exactly without asserting a final decision."""

    if not claims:
        return {
            "policy": KERC_PRIOR_CLAIM_CONTEXT_POLICY,
            "claim_count": 0,
            "transport": None,
        }
    claim_ids = [str(claim.get("claim_id") or "") for claim in claims]
    context_packet = validate_answer_packet(
        {
            "claims": copy.deepcopy(list(claims)),
            "decision": {
                "policy": ANSWER_DECISION_POLICY,
                "disposition": "ANSWER",
                "evidence_status": "UNVERIFIED",
                "uncertainty_state": "RESOLVED",
                "confidence": min(float(claim["confidence"]) for claim in claims),
                "controlling_claim_ids": claim_ids,
                "unresolved_ambiguity_ids": [],
            },
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "internal_dependency_context"},
        },
        require_explicit_decision=True,
    )
    return {
        "policy": KERC_PRIOR_CLAIM_CONTEXT_POLICY,
        "claim_count": len(claims),
        "transport": learned_answer_packet_view(context_packet),
        "final_decision_authority": False,
    }


def materialize_learned_answer_packet(view: Any) -> dict[str, Any]:
    """Decode a learned answer stream and independently validate its packet."""

    if (
        not isinstance(view, dict)
        or set(view) != {"policy", "tokens"}
        or view.get("policy") != LEARNED_ANSWER_TRANSPORT_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_TRANSPORT_INVALID", canonical_json(view), path="answer"
        )
    materialized_tokens = _materialize_learned_token_view(
        view["tokens"], path="answer.tokens"
    )
    if token("V_P", "DECISION_BEGIN") not in materialized_tokens:
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_POLICY_INVALID",
            "compact learned answer omitted decision contract",
            path="answer.tokens",
        )
    cursor = _KernelTokenCursor(materialized_tokens)
    cursor.expect("V_P", "ANSWER_VERSION:1")
    claims: list[dict[str, Any]] = []
    while cursor.peek() == {"space": "V_P", "token": "CLAIM_BEGIN"}:
        cursor.take()
        claim_id = cursor.expect_prefix("V_P", "CLAIM_ID:")
        predicate = cursor.expect_prefix("V_K", "PRED:")
        modality = cursor.expect_prefix("V_K", "MOD:")
        polarity = cursor.expect_prefix("V_K", "POL:")
        quantifier = cursor.expect_prefix("V_K", "QUANT:")
        confidence_text = cursor.expect_prefix("V_P", "CONF:")
        arguments: list[dict[str, Any]] = []
        while cursor.peek()["space"] == "V_K" and cursor.peek()["token"].startswith(
            "ROLE:"
        ):
            role = cursor.expect_prefix("V_K", "ROLE:")
            arguments.append({"role": role, "value": _deserialize_value(cursor)})
        metadata: dict[str, Any] = {}
        if cursor.peek()["space"] == "V_P" and cursor.peek()["token"].startswith(
            "CLAIM_METADATA:"
        ):
            encoded_metadata = cursor.expect_prefix("V_P", "CLAIM_METADATA:")
            try:
                decoded_metadata = json.loads(encoded_metadata)
            except json.JSONDecodeError as exc:
                raise KernelProtocolFault(
                    "KERC_LEARNED_ANSWER_CLAIM_METADATA_INVALID",
                    str(exc),
                    path="answer.tokens",
                ) from exc
            if not isinstance(decoded_metadata, dict) or set(decoded_metadata) & {
                "claim_id",
                "predicate",
                "modality",
                "polarity",
                "quantifier",
                "confidence",
                "arguments",
            }:
                raise KernelProtocolFault(
                    "KERC_LEARNED_ANSWER_CLAIM_METADATA_INVALID",
                    canonical_json(decoded_metadata),
                    path="answer.tokens",
                )
            metadata = decoded_metadata
        cursor.expect("V_P", "CLAIM_END")
        try:
            confidence = float(confidence_text)
        except ValueError as exc:
            raise KernelProtocolFault(
                "KERC_LEARNED_ANSWER_CONFIDENCE_INVALID",
                confidence_text,
                path="answer.tokens",
            ) from exc
        claims.append(
            {
                **metadata,
                "claim_id": claim_id,
                "predicate": predicate,
                "modality": modality,
                "polarity": polarity,
                "quantifier": quantifier,
                "confidence": confidence,
                "arguments": arguments,
            }
        )
    cursor.expect("V_P", "DECISION_BEGIN")
    decision = {
        "policy": ANSWER_DECISION_POLICY,
        "disposition": cursor.expect_prefix("V_K", "DISPOSITION:"),
        "evidence_status": cursor.expect_prefix("V_K", "EVIDENCE:"),
        "uncertainty_state": cursor.expect_prefix("V_K", "UNCERTAINTY:"),
    }
    confidence_text = cursor.expect_prefix("V_P", "DECISION_CONF:")
    controlling: list[str] = []
    ambiguity_ids: list[str] = []
    while cursor.peek()["token"] != "DECISION_END":
        current = cursor.peek()
        if current["space"] == "V_P" and current["token"].startswith("CONTROLLING:"):
            controlling.append(cursor.expect_prefix("V_P", "CONTROLLING:"))
        elif current["space"] == "V_P" and current["token"].startswith("AMBIGUITY_ID:"):
            ambiguity_ids.append(cursor.expect_prefix("V_P", "AMBIGUITY_ID:"))
        else:
            raise KernelProtocolFault(
                "KERC_LEARNED_ANSWER_DECISION_TOKEN_INVALID",
                canonical_json(current),
                path="answer.tokens",
            )
    cursor.take()
    try:
        decision["confidence"] = float(confidence_text)
    except ValueError as exc:
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_CONFIDENCE_INVALID",
            confidence_text,
            path="answer.tokens",
        ) from exc
    decision["controlling_claim_ids"] = controlling
    decision["unresolved_ambiguity_ids"] = ambiguity_ids
    required_terms: list[Any] = []
    required_caveats: list[Any] = []
    while cursor.peek()["token"].startswith("REQUIRED_TERM:"):
        required_terms.append(json.loads(cursor.expect_prefix("V_P", "REQUIRED_TERM:")))
    while cursor.peek()["token"].startswith("REQUIRED_CAVEAT:"):
        required_caveats.append(
            json.loads(cursor.expect_prefix("V_P", "REQUIRED_CAVEAT:"))
        )
    try:
        style = json.loads(cursor.expect_prefix("V_P", "STYLE:"))
    except json.JSONDecodeError as exc:
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_STYLE_INVALID", str(exc), path="answer.tokens"
        ) from exc
    cursor.expect("V_P", "ANSWER_END")
    if not cursor.done:
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_TRAILING_TOKENS",
            canonical_json(cursor.remaining()),
            path="answer.tokens",
        )
    return validate_answer_packet(
        {
            "claims": claims,
            "decision": decision,
            "required_terms": required_terms,
            "required_caveats": required_caveats,
            "style": style,
        },
        require_explicit_decision=True,
    )


def compiler_training_io(
    *, packet: dict[str, Any], source_text: str, hrl_state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the leak-free compiler input and its source-derived target.

    Packet concept semantics are annotation targets unless a separate runtime
    context explicitly provides them. They must therefore be generated beside
    the Kernel program, never exposed in the source-only compiler prompt.
    Evidence-plane stable identities and provenance are stripped and attached
    deterministically after learned output validation.
    """

    compiler_input = compiler_input_from_source(source_text, hrl_state=hrl_state)
    compiler_target = {
        "kernel_version": KERNEL_VERSION,
        "protected_objects": learned_protected_span_view(packet["protected_objects"]),
        "concept_capsules": learned_concept_capsule_view(packet["concept_capsules"]),
        "program": learned_kernel_program_view(packet),
        "residual": learned_residual_view(packet["residual"], hrl_state=hrl_state),
    }
    return compiler_input, compiler_target


def _node_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        if value.get("type") == "node_ref" and isinstance(value.get("value"), str):
            references.add(str(value["value"]))
        else:
            for child in value.values():
                references.update(_node_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_node_references(child))
    return references


def partition_kernel_program(
    canonical_program: dict[str, Any], *, maximum_nodes: int = KERC_CORE_CHUNK_MAX_NODES
) -> list[dict[str, Any]]:
    """Build topological chunks with direct, causally available dependencies."""

    nodes = list(canonical_program.get("nodes") or [])
    if not nodes or maximum_nodes <= 0:
        raise KernelProtocolFault(
            "KERC_HIERARCHICAL_PROGRAM_INVALID", str(len(nodes)), path="program.nodes"
        )
    by_id = {str(node["node_id"]): node for node in nodes}
    references_by_node: dict[str, set[str]] = {}
    for node_id, node in by_id.items():
        references = _node_references(node.get("arguments") or [])
        unknown = references - set(by_id)
        if unknown:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_NODE_REFERENCE_UNKNOWN",
                canonical_json(sorted(unknown)),
                path=f"program.nodes.{node_id}",
            )
        references_by_node[node_id] = references

    order = {str(node["node_id"]): index for index, node in enumerate(nodes)}
    dependants: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    indegree = {
        node_id: len(references) for node_id, references in references_by_node.items()
    }
    for node_id, references in references_by_node.items():
        for dependency in references:
            dependants[dependency].append(node_id)
    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=order.__getitem__,
    )
    targets: list[str] = []
    while ready:
        current = ready.pop(0)
        targets.append(current)
        for dependant in sorted(dependants[current], key=order.__getitem__):
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                ready.append(dependant)
                ready.sort(key=order.__getitem__)
    if len(targets) != len(nodes):
        raise KernelProtocolFault(
            "KERC_HIERARCHICAL_PROGRAM_CYCLE",
            f"{len(targets)}:{len(nodes)}",
            path="program.nodes",
        )
    packed = [
        targets[index : index + maximum_nodes]
        for index in range(0, len(targets), maximum_nodes)
    ]
    full_roots = set(str(value) for value in canonical_program.get("roots") or [])
    fragments: list[dict[str, Any]] = []
    for index, target_node_ids in enumerate(packed):
        target_set = set(target_node_ids)
        context_node_ids = sorted(
            {
                dependency
                for node_id in target_node_ids
                for dependency in references_by_node[node_id]
                if dependency not in target_set
            },
            key=order.__getitem__,
        )
        prior_targets = {node_id for chunk in packed[:index] for node_id in chunk}
        if not set(context_node_ids) <= prior_targets:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_DEPENDENCY_NOT_CAUSALLY_AVAILABLE",
                canonical_json(sorted(set(context_node_ids) - prior_targets)),
                path=f"program.chunks[{index}]",
            )
        local_roots = [node_id for node_id in target_node_ids if node_id in full_roots]
        if not local_roots:
            referenced = {
                target
                for node_id in target_node_ids
                for target in references_by_node[node_id]
            }
            local_roots = [
                node_id for node_id in target_node_ids if node_id not in referenced
            ]
        if not local_roots:
            local_roots = [target_node_ids[-1]]
        context_stubs = []
        for node_id in context_node_ids:
            source_node = by_id[node_id]
            context_stubs.append(
                {
                    **{
                        key: copy.deepcopy(value)
                        for key, value in source_node.items()
                        if key != "arguments"
                    },
                    "derivation": "prior_chunk_claim",
                    "arguments": [],
                }
            )
        fragment_program = {
            "record_type": "kernel_program",
            "kernel_version": KERNEL_VERSION,
            "roots": local_roots,
            "nodes": [
                *context_stubs,
                *[copy.deepcopy(by_id[node_id]) for node_id in target_node_ids],
            ],
        }
        fragment_program["program_sha256"] = stable_hash(fragment_program)
        fragments.append(
            {
                "policy": KERC_HIERARCHICAL_CORE_POLICY,
                "chunk_index": index,
                "chunk_count": len(packed),
                "node_ids": target_node_ids,
                "context_node_ids": context_node_ids,
                "claim_ordinals": [order[node_id] for node_id in target_node_ids],
                "full_root_node_ids": [
                    node_id for node_id in target_node_ids if node_id in full_roots
                ],
                "full_program_sha256": canonical_program["program_sha256"],
                "program": fragment_program,
            }
        )
    return fragments


def partition_answer_for_program_fragments(
    answer: dict[str, Any], fragments: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Project one-to-one node/claim supervision into exactly mergeable packets."""

    canonical = validate_answer_packet(answer, require_explicit_decision=True)
    claims = canonical["claims"]
    node_count = sum(len(fragment["node_ids"]) for fragment in fragments)
    if len(claims) != node_count:
        raise KernelProtocolFault(
            "KERC_HIERARCHICAL_NODE_CLAIM_ALIGNMENT_UNAVAILABLE",
            f"{node_count}:{len(claims)}",
            path="answer.claims",
        )
    controlling = set(canonical["decision"]["controlling_claim_ids"])
    packets: list[dict[str, Any]] = []
    claim_by_node: dict[str, dict[str, Any]] = {}
    claim_ordinals = [
        ordinal for fragment in fragments for ordinal in fragment["claim_ordinals"]
    ]
    if sorted(claim_ordinals) != list(range(len(claims))):
        raise KernelProtocolFault(
            "KERC_HIERARCHICAL_NODE_CLAIM_ALIGNMENT_UNAVAILABLE",
            f"{len(claim_ordinals)}:{len(claims)}",
            path="answer.claims",
        )
    for fragment in fragments:
        for node_id, ordinal in zip(fragment["node_ids"], fragment["claim_ordinals"]):
            claim_by_node[str(node_id)] = claims[int(ordinal)]
    for fragment in fragments:
        local_claims = copy.deepcopy(
            [claim_by_node[node_id] for node_id in fragment["node_ids"]]
        )
        nodes_by_id = {
            str(node["node_id"]): node for node in fragment["program"]["nodes"]
        }
        local_nodes = [nodes_by_id[node_id] for node_id in fragment["node_ids"]]
        if any(
            claim["predicate"] != node["operator"]
            for claim, node in zip(local_claims, local_nodes)
        ):
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_NODE_CLAIM_ALIGNMENT_UNAVAILABLE",
                str(fragment["chunk_index"]),
                path="answer.claims",
            )
        local_controlling = [
            claim["claim_id"]
            for claim in local_claims
            if claim["claim_id"] in controlling
        ]
        if not local_controlling:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_DECISION_ALIGNMENT_UNAVAILABLE",
                str(fragment["chunk_index"]),
                path="answer.decision.controlling_claim_ids",
            )
        local = {
            "claims": local_claims,
            "decision": {
                **copy.deepcopy(canonical["decision"]),
                "controlling_claim_ids": local_controlling,
            },
            "required_terms": copy.deepcopy(canonical.get("required_terms") or []),
            "required_caveats": copy.deepcopy(canonical.get("required_caveats") or []),
            "style": copy.deepcopy(canonical.get("style") or {}),
        }
        packets.append(validate_answer_packet(local, require_explicit_decision=True))
    return packets


def dependency_claims_for_program_fragments(
    answer: dict[str, Any], fragments: Sequence[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Expose only already-computed direct claims required by each chunk."""

    canonical = validate_answer_packet(answer, require_explicit_decision=True)
    claims = canonical["claims"]
    claim_by_node: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        for node_id, ordinal in zip(fragment["node_ids"], fragment["claim_ordinals"]):
            claim_by_node[str(node_id)] = claims[int(ordinal)]
    contexts: list[list[dict[str, Any]]] = []
    available: set[str] = set()
    for fragment in fragments:
        context_ids = [str(value) for value in fragment["context_node_ids"]]
        if not set(context_ids) <= available:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_DEPENDENCY_NOT_CAUSALLY_AVAILABLE",
                canonical_json(sorted(set(context_ids) - available)),
                path=f"answer_chunks[{fragment['chunk_index']}]",
            )
        contexts.append(
            [copy.deepcopy(claim_by_node[node_id]) for node_id in context_ids]
        )
        available.update(str(value) for value in fragment["node_ids"])
    return contexts


def merge_hierarchical_answer_packets(
    packets: Sequence[dict[str, Any]],
    *,
    expected_chunk_count: int,
    claim_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Losslessly assemble independently validated core chunks without fallback."""

    if len(packets) != expected_chunk_count or not packets:
        raise KernelProtocolFault(
            "KERC_HIERARCHICAL_ANSWER_CHUNK_COUNT_INVALID",
            f"{len(packets)}:{expected_chunk_count}",
            path="answer_chunks",
        )
    canonical = [
        validate_answer_packet(packet, require_explicit_decision=True)
        for packet in packets
    ]
    first = canonical[0]
    invariant_decision = {
        key: value
        for key, value in first["decision"].items()
        if key not in {"controlling_claim_ids", "unresolved_ambiguity_ids"}
    }
    claims: list[dict[str, Any]] = []
    controlling: set[str] = set()
    ambiguities: set[str] = set()
    observed_claim_ids: set[str] = set()
    for packet in canonical:
        if (
            any(
                packet.get(field) != first.get(field)
                for field in ("required_terms", "required_caveats", "style")
            )
            or {
                key: value
                for key, value in packet["decision"].items()
                if key not in {"controlling_claim_ids", "unresolved_ambiguity_ids"}
            }
            != invariant_decision
        ):
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_ANSWER_INVARIANT_MISMATCH",
                packet["answer_packet_sha256"],
                path="answer_chunks",
            )
        local_ids = {str(claim["claim_id"]) for claim in packet["claims"]}
        if observed_claim_ids & local_ids:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_ANSWER_CLAIM_COLLISION",
                canonical_json(sorted(observed_claim_ids & local_ids)),
                path="answer_chunks",
            )
        observed_claim_ids.update(local_ids)
        claims.extend(copy.deepcopy(packet["claims"]))
        controlling.update(packet["decision"]["controlling_claim_ids"])
        ambiguities.update(packet["decision"]["unresolved_ambiguity_ids"])
    if claim_order is not None:
        requested = [str(value) for value in claim_order]
        by_id = {str(claim["claim_id"]): claim for claim in claims}
        if len(requested) != len(set(requested)) or set(requested) != set(by_id):
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_ANSWER_CLAIM_ORDER_INVALID",
                canonical_json(requested),
                path="answer_chunks",
            )
        claims = [by_id[claim_id] for claim_id in requested]
    merged = {
        "claims": claims,
        "decision": {
            **copy.deepcopy(invariant_decision),
            "controlling_claim_ids": sorted(controlling),
            "unresolved_ambiguity_ids": sorted(ambiguities),
        },
        "required_terms": copy.deepcopy(first.get("required_terms") or []),
        "required_caveats": copy.deepcopy(first.get("required_caveats") or []),
        "style": copy.deepcopy(first.get("style") or {}),
    }
    return validate_answer_packet(merged, require_explicit_decision=True)


def compile_training_views(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile matched learned objectives without multiplying unique-data credit."""

    record = validate_training_record(record)
    packet = record["kernel_packet"]
    learned_objects = learned_protected_object_view(packet["protected_objects"])
    protected_context = {
        "protected_objects": learned_objects,
        "concept_capsules": learned_concept_capsule_view(packet["concept_capsules"]),
        "source_character_length": packet["source"]["character_length"],
    }
    compiler_input, compiler_target = compiler_training_io(
        packet=packet,
        source_text=record["source_text"],
        hrl_state=record["hrl_state"],
    )
    core_input = {
        **protected_context,
        "program": learned_kernel_program_view(packet),
        "residual": learned_residual_view(
            packet["residual"], hrl_state=record["hrl_state"]
        ),
    }
    renderer_input = {
        "protected_objects": learned_objects,
        "residual": learned_residual_view(
            packet["residual"], hrl_state=record["hrl_state"]
        ),
    }
    source = record["source_text"]
    interaction = learned_interaction_residual_view(record["hrl_state"])
    direct_prompt = (
        canonical_json({"current_surface": source, "interaction": interaction})
        if interaction
        else source
    )
    objective_authority = record["semantic_supervision"]["objective_authority"]
    rows: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for objective in TRAINING_OBJECTIVES:
        if objective_authority[objective] is not True:
            continue
        if objective == "surface_to_kernel_program_v1":
            fragments = partition_kernel_program(
                packet["program"], maximum_nodes=KERC_COMPILER_CHUNK_MAX_NODES
            )
            prior_nodes: list[dict[str, Any]] = []
            previous_program: dict[str, Any] | None = None
            for fragment in fragments:
                compiler_contract = {
                    "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
                    "chunk_index": fragment["chunk_index"],
                    "prior_node_count": len(prior_nodes),
                    "previous_program": copy.deepcopy(previous_program),
                    "accumulated_program_sha256": stable_hash(prior_nodes),
                }
                visible = canonical_json(
                    {
                        **compiler_input,
                        "hierarchical_compiler": compiler_contract,
                    }
                )
                target = canonical_json(
                    {
                        "kernel_version": KERNEL_VERSION,
                        "protected_objects": copy.deepcopy(
                            compiler_target["protected_objects"]
                        ),
                        "concept_capsules": (
                            compiler_target["concept_capsules"]
                            if fragment["chunk_index"] == 0
                            else {}
                        ),
                        "program": learned_kernel_program_view_from_program(
                            fragment["program"]
                        ),
                        "residual": copy.deepcopy(compiler_target["residual"]),
                        "hierarchical_compiler": {
                            "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
                            "chunk_index": fragment["chunk_index"],
                            "continuation": fragment["chunk_index"]
                            < fragment["chunk_count"] - 1,
                            "root_node_ids": fragment["full_root_node_ids"],
                        },
                    }
                )
                rows.append(
                    (
                        objective,
                        visible,
                        target,
                        f"compiler:chunk-{fragment['chunk_index']}",
                        {
                            "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
                            "chunk_index": fragment["chunk_index"],
                            "chunk_count": fragment["chunk_count"],
                            "prior_node_count": len(prior_nodes),
                        },
                    )
                )
                nodes_by_id = {
                    str(node["node_id"]): node for node in fragment["program"]["nodes"]
                }
                prior_nodes.extend(
                    {
                        key: copy.deepcopy(nodes_by_id[node_id][key])
                        for key in (
                            "node_id",
                            "operator",
                            "modality",
                            "polarity",
                            "quantifier",
                            "confidence",
                            "source_spans",
                        )
                    }
                    for node_id in fragment["node_ids"]
                )
                previous_program = learned_kernel_program_view_from_program(
                    fragment["program"]
                )
            continue
        for realization in record["valid_realizations"]:
            realization_id = str(realization["realization_id"])
            answer_packet = realization["answer_packet"]
            if objective == "surface_direct_control_v1":
                visible = direct_prompt
                target = str(realization["surface_target"])
                rows.append((objective, visible, target, realization_id, {}))
            elif objective == "kernel_program_to_answer_packet_v1":
                fragments = partition_kernel_program(packet["program"])
                if len(fragments) == 1:
                    visible = canonical_json(core_input)
                    target = canonical_json(learned_answer_packet_view(answer_packet))
                    rows.append((objective, visible, target, realization_id, {}))
                else:
                    partial_answers = partition_answer_for_program_fragments(
                        answer_packet, fragments
                    )
                    dependency_contexts = dependency_claims_for_program_fragments(
                        answer_packet, fragments
                    )
                    for fragment, partial_answer, dependency_claims in zip(
                        fragments, partial_answers, dependency_contexts
                    ):
                        chunk_contract = {
                            "policy": KERC_HIERARCHICAL_CORE_POLICY,
                            "chunk_index": fragment["chunk_index"],
                            "chunk_count": fragment["chunk_count"],
                            "node_ids": fragment["node_ids"],
                            "context_node_ids": fragment["context_node_ids"],
                            "full_program_sha256": fragment["full_program_sha256"],
                        }
                        visible = canonical_json(
                            {
                                **protected_context,
                                "program": learned_kernel_program_view_from_program(
                                    fragment["program"]
                                ),
                                "residual": learned_residual_view(
                                    packet["residual"], hrl_state=record["hrl_state"]
                                ),
                                "prior_claims": learned_prior_claim_context_view(
                                    dependency_claims
                                ),
                                "hierarchical_core": chunk_contract,
                            }
                        )
                        target = canonical_json(
                            learned_answer_packet_view(partial_answer)
                        )
                        rows.append(
                            (
                                objective,
                                visible,
                                target,
                                f"{realization_id}:chunk-{fragment['chunk_index']}",
                                chunk_contract,
                            )
                        )
                continue
            else:
                visible = canonical_json(
                    {
                        **renderer_input,
                        "answer_packet": learned_answer_packet_view(answer_packet),
                    }
                )
                target = str(realization["surface_target"])
                rows.append((objective, visible, target, realization_id, {}))
    compiled = []
    for objective, visible, target, realization_id, hierarchical_transport in rows:
        prompt = visible
        trusted_prefix = [TRAINING_TASK_TAGS[objective]]
        identity = stable_hash(
            {
                "record_sha256": record["record_sha256"],
                "objective": objective,
                "trusted_source_prefix_tokens": trusted_prefix,
                "prompt": prompt,
                "target": target,
                "realization_id": realization_id,
            }
        )
        verifier_negative = _targeted_verifier_corruption(
            objective,
            target,
            protected_objects=packet["protected_objects"],
            record_identity=f"{record['record_sha256']}:{realization_id}",
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
                "semantic_evidence_tier": record["semantic_supervision"][
                    "evidence_tier"
                ],
                "semantic_claim_authority": record["semantic_supervision"][
                    "claim_authority"
                ],
                "objective_semantic_authority": True,
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
                "kerc_residual_unit_ids": [
                    str(row["unit_id"])
                    for row in record["residual_supervision"][
                        "residual_unit_allocation"
                    ]["unit_allocations"]
                ],
                "kerc_residual_unit_fidelity_labels": [
                    int(
                        KERC_FIDELITY_LABELS[str(row["selected_fidelity"])]
                    )
                    for row in record["residual_supervision"][
                        "residual_unit_allocation"
                    ]["unit_allocations"]
                ],
                "kerc_residual_unit_allocator_loss_enabled": False,
                "kerc_residual_unit_target_authority": (
                    "k2_structural_baseline_only_not_intervention_or_semantic_utility"
                ),
                "kerc_verifier_dimensions": list(KERC_VERIFIER_DIMENSIONS),
                "kerc_verifier_positive_labels": [1] * len(KERC_VERIFIER_DIMENSIONS),
                "kerc_verifier_negative": verifier_negative,
                "kerc_answer_disposition": str(
                    record["answer_packet"]["decision"]["disposition"]
                ),
                "kerc_hierarchical_transport": copy.deepcopy(hierarchical_transport),
                "generator_visible_fields": ["trusted_source_prefix_tokens", "prompt"],
                "evaluator_only_fields": [
                    "target",
                    "target_sha256",
                    "source_record_sha256",
                    "kerc_verifier_negative",
                    "kerc_answer_disposition",
                    "realization_id",
                ],
                "public_benchmark": False,
                "public_tests_included": False,
                "public_benchmark_solutions_included": False,
                "external_inference": False,
                **NO_CHEAT,
            }
        )
    return compiled


def compiler_input_from_source(
    source: str, *, hrl_state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build the deterministic, source-only front end seen by the compiler."""

    return {
        "source_surface": source,
        "concept_capsules": {},
        "source_character_length": len(source),
        "interaction": (
            learned_interaction_residual_view(hrl_state)
            if isinstance(hrl_state, dict)
            else []
        ),
    }


def execute_learned_pipeline(
    source: str,
    *,
    hrl_state: dict[str, Any],
    stage_executor: Callable[[str, str], tuple[str, dict[str, Any]]],
    concept_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
                "KERC_LEARNED_STAGE_OBJECT_REQUIRED",
                type(payload).__name__,
                path=objective,
            )
        return payload

    def compile_surface(surface: str) -> dict[str, Any]:
        front_end = compiler_input_from_source(surface, hrl_state=hrl_state)
        generated_objects: dict[str, dict[str, Any]] | None = None
        prior_nodes: list[dict[str, Any]] = []
        canonical_nodes: list[dict[str, Any]] = []
        generated_capsules: dict[str, dict[str, Any]] = {}
        generated_residual: dict[str, Any] | None = None
        root_node_ids: list[str] = []
        previous_program: dict[str, Any] | None = None
        for chunk_index in range(256):
            compiler_contract = {
                "policy": KERC_HIERARCHICAL_COMPILER_POLICY,
                "chunk_index": chunk_index,
                "prior_node_count": len(prior_nodes),
                "previous_program": copy.deepcopy(previous_program),
                "accumulated_program_sha256": stable_hash(prior_nodes),
            }
            compiler_text = run_stage(
                "surface_to_kernel_program_v1",
                canonical_json(
                    {
                        **front_end,
                        "hierarchical_compiler": compiler_contract,
                    }
                ),
            )
            decoded = parse_object("surface_to_kernel_program_v1", compiler_text)
            hierarchy = decoded.get("hierarchical_compiler")
            if (
                not isinstance(hierarchy, dict)
                or set(hierarchy)
                != {"policy", "chunk_index", "continuation", "root_node_ids"}
                or hierarchy.get("policy") != KERC_HIERARCHICAL_COMPILER_POLICY
                or int(hierarchy.get("chunk_index", -1)) != chunk_index
                or not isinstance(hierarchy.get("continuation"), bool)
                or not isinstance(hierarchy.get("root_node_ids"), list)
            ):
                raise KernelProtocolFault(
                    "KERC_HIERARCHICAL_COMPILER_CONTRACT_INVALID",
                    canonical_json(hierarchy),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            compiler_output = parse_learned_compiler_output(
                compiler_text,
                protected_objects=generated_objects or {},
                concept_capsules=generated_capsules,
                source_character_length=len(surface),
                source=surface,
                hrl_state=hrl_state,
                concept_resolver=concept_resolver,
            )
            chunk_objects = compiler_output.get("generated_protected_objects")
            if not isinstance(chunk_objects, dict):
                raise KernelProtocolFault(
                    "KERC_LEARNED_PROTECTED_OBJECTS_MISSING",
                    str(chunk_index),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            if generated_objects is None:
                generated_objects = copy.deepcopy(chunk_objects)
            elif chunk_objects != generated_objects:
                raise KernelProtocolFault(
                    "KERC_LEARNED_PROTECTED_OBJECTS_CHUNK_MISMATCH",
                    str(chunk_index),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            chunk_residual = compiler_output.get("learned_residual")
            if not isinstance(chunk_residual, dict):
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_MISSING",
                    str(chunk_index),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            if generated_residual is None:
                generated_residual = copy.deepcopy(chunk_residual)
            elif chunk_residual != generated_residual:
                raise KernelProtocolFault(
                    "KERC_LEARNED_RESIDUAL_CHUNK_MISMATCH",
                    str(chunk_index),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            generated_capsules.update(compiler_output["generated_concept_capsules"])
            chunk_program = compiler_output["canonical_program"]
            prior_by_id = {str(row["node_id"]): row for row in prior_nodes}
            target_nodes: list[dict[str, Any]] = []
            for node in chunk_program["nodes"]:
                node_id = str(node["node_id"])
                if node.get("derivation") == "prior_chunk_claim":
                    prior = prior_by_id.get(node_id)
                    observed = {
                        key: copy.deepcopy(node[key])
                        for key in (
                            "node_id",
                            "operator",
                            "modality",
                            "polarity",
                            "quantifier",
                            "confidence",
                            "source_spans",
                        )
                    }
                    if prior != observed or node.get("arguments") != []:
                        raise KernelProtocolFault(
                            "KERC_HIERARCHICAL_COMPILER_CONTEXT_STUB_INVALID",
                            node_id,
                            path=f"compiler_chunks[{chunk_index}]",
                        )
                    continue
                if node_id in prior_by_id or any(
                    node_id == str(existing["node_id"]) for existing in canonical_nodes
                ):
                    raise KernelProtocolFault(
                        "KERC_HIERARCHICAL_COMPILER_NODE_COLLISION",
                        node_id,
                        path=f"compiler_chunks[{chunk_index}]",
                    )
                target_nodes.append(copy.deepcopy(node))
            if not target_nodes or len(target_nodes) > KERC_COMPILER_CHUNK_MAX_NODES:
                raise KernelProtocolFault(
                    "KERC_HIERARCHICAL_COMPILER_TARGET_COUNT_INVALID",
                    str(len(target_nodes)),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            target_ids = {str(node["node_id"]) for node in target_nodes}
            emitted_roots = [str(value) for value in hierarchy["root_node_ids"]]
            if (
                len(set(emitted_roots)) != len(emitted_roots)
                or not set(emitted_roots) <= target_ids
            ):
                raise KernelProtocolFault(
                    "KERC_HIERARCHICAL_COMPILER_ROOTS_INVALID",
                    canonical_json(emitted_roots),
                    path=f"compiler_chunks[{chunk_index}]",
                )
            canonical_nodes.extend(target_nodes)
            root_node_ids.extend(emitted_roots)
            prior_nodes.extend(
                {
                    key: copy.deepcopy(node[key])
                    for key in (
                        "node_id",
                        "operator",
                        "modality",
                        "polarity",
                        "quantifier",
                        "confidence",
                        "source_spans",
                    )
                }
                for node in target_nodes
            )
            previous_program = learned_kernel_program_view_from_program(chunk_program)
            if hierarchy["continuation"] is False:
                break
        else:
            raise KernelProtocolFault(
                "KERC_HIERARCHICAL_COMPILER_CHUNK_LIMIT",
                "256",
                path="compiler_chunks",
            )
        compiler_output = validate_kernel_program(
            {"roots": root_node_ids, "nodes": canonical_nodes},
            protected_objects=generated_objects or {},
            concept_capsules=generated_capsules,
            source_character_length=len(surface),
        )
        if generated_objects is None:
            raise KernelProtocolFault(
                "KERC_LEARNED_PROTECTED_OBJECTS_MISSING", "all", path="compiler_chunks"
            )
        packet = build_kernel_packet(
            surface,
            compiler_output["canonical_program"],
            hrl_state=hrl_state,
            explicit_spans=[
                {
                    "start": int(row["source_span"]["character_start"]),
                    "end": int(row["source_span"]["character_end"]),
                    "object_type": str(row["object_type"]),
                    "copy_policy": str(row["copy_policy"]),
                }
                for row in generated_objects.values()
            ],
            concept_capsules=generated_capsules,
            residual_mode="SOURCE_RECONSTRUCTION",
            fidelity="lexical",
            provenance={"source": "learned_kerc_pipeline_v1"},
        )
        validate_kernel_packet(packet, local_hrl_state=hrl_state)
        if generated_residual is None:
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_MISSING", "all", path="compiler_chunks"
            )
        return packet, generated_residual

    def reason(
        packet: dict[str, Any], learned_residual: dict[str, Any]
    ) -> dict[str, Any]:
        common_input = {
            "protected_objects": learned_protected_object_view(
                packet["protected_objects"]
            ),
            "concept_capsules": learned_concept_capsule_view(
                packet["concept_capsules"]
            ),
            "source_character_length": packet["source"]["character_length"],
            "residual": copy.deepcopy(learned_residual),
        }
        fragments = partition_kernel_program(packet["program"])
        partial_answers: list[dict[str, Any]] = []
        claims_by_node: dict[str, dict[str, Any]] = {}
        for fragment in fragments:
            missing_dependency_claims = set(fragment["context_node_ids"]) - set(
                claims_by_node
            )
            if missing_dependency_claims:
                raise KernelProtocolFault(
                    "KERC_HIERARCHICAL_DEPENDENCY_CLAIM_MISSING",
                    canonical_json(sorted(missing_dependency_claims)),
                    path=f"answer_chunks[{fragment['chunk_index']}]",
                )
            chunk_contract = {
                "policy": KERC_HIERARCHICAL_CORE_POLICY,
                "chunk_index": fragment["chunk_index"],
                "chunk_count": fragment["chunk_count"],
                "node_ids": fragment["node_ids"],
                "context_node_ids": fragment["context_node_ids"],
                "full_program_sha256": fragment["full_program_sha256"],
            }
            core_input = {
                **common_input,
                "program": (
                    learned_kernel_program_view(packet)
                    if len(fragments) == 1
                    else learned_kernel_program_view_from_program(fragment["program"])
                ),
                **(
                    {
                        "hierarchical_core": chunk_contract,
                        "prior_claims": learned_prior_claim_context_view(
                            [
                                copy.deepcopy(claims_by_node[node_id])
                                for node_id in fragment["context_node_ids"]
                            ]
                        ),
                    }
                    if len(fragments) > 1
                    else {}
                ),
            }
            answer_text = run_stage(
                "kernel_program_to_answer_packet_v1", canonical_json(core_input)
            )
            partial = materialize_learned_answer_packet(
                parse_object("kernel_program_to_answer_packet_v1", answer_text)
            )
            if len(fragments) > 1:
                nodes_by_id = {
                    str(node["node_id"]): node for node in fragment["program"]["nodes"]
                }
                expected_predicates = [
                    nodes_by_id[node_id]["operator"] for node_id in fragment["node_ids"]
                ]
                observed_predicates = [
                    claim["predicate"] for claim in partial["claims"]
                ]
                if observed_predicates != expected_predicates:
                    raise KernelProtocolFault(
                        "KERC_HIERARCHICAL_NODE_CLAIM_ALIGNMENT_INVALID",
                        canonical_json(
                            {
                                "expected": expected_predicates,
                                "observed": observed_predicates,
                            }
                        ),
                        path="answer_chunks",
                    )
                claims_by_node.update(
                    {
                        node_id: copy.deepcopy(claim)
                        for node_id, claim in zip(
                            fragment["node_ids"], partial["claims"]
                        )
                    }
                )
            partial_answers.append(partial)
        answer = (
            partial_answers[0]
            if len(partial_answers) == 1
            else merge_hierarchical_answer_packets(
                partial_answers,
                expected_chunk_count=len(fragments),
                claim_order=[
                    claims_by_node[str(node["node_id"])]["claim_id"]
                    for node in packet["program"]["nodes"]
                ],
            )
        )
        return validate_answer_packet_against_context(
            answer,
            protected_objects=packet["protected_objects"],
            concept_capsules=packet["concept_capsules"],
            correction_lattice=packet["correction_lattice"],
            require_explicit_decision=True,
        )

    packet, source_residual = compile_surface(source)
    intended_answer = reason(packet, source_residual)
    renderer_input = {
        "answer_packet": learned_answer_packet_view(intended_answer),
        "protected_objects": learned_protected_object_view(packet["protected_objects"]),
        "residual": copy.deepcopy(source_residual),
    }
    surface = run_stage("answer_packet_to_surface_v1", canonical_json(renderer_input))
    reconstructed_packet, reconstructed_residual = compile_surface(surface)
    reconstructed_answer = reason(reconstructed_packet, reconstructed_residual)
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
        "learned_protected_object_count": len(packet["protected_objects"]),
        "learned_protected_object_types": sorted(
            str(row["object_type"]) for row in packet["protected_objects"].values()
        ),
        "recompiled_protected_object_count": len(
            reconstructed_packet["protected_objects"]
        ),
        "protected_span_route": "learned_compiler_span_output_then_exact_source_materialization",
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
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in range(4)
        ):
            raise KernelProtocolFault(
                "KERC_RESIDUAL_SUPERVISION_LABEL_INVALID",
                f"{channel}:{value}",
                path=f"record.residual_supervision.labels_by_channel.{channel}",
            )
    residual = (
        packet.get("residual") if isinstance(packet.get("residual"), dict) else {}
    )
    expected_fidelity = KERC_FIDELITY_LABELS.get(str(residual.get("fidelity") or ""))
    if (
        expected_fidelity is None
        or int(supervision.get("record_fidelity_label", -1)) != expected_fidelity
    ):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SUPERVISION_FIDELITY_MISMATCH",
            canonical_json(supervision),
            path="record.residual_supervision.record_fidelity_label",
        )
    if (
        supervision.get("record_fidelity_label_training_authority") is not False
        or supervision.get("packet_wide_fidelity_drives_training") is not False
    ):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_PACKET_WIDE_TRAINING_AUTHORITY_FORBIDDEN",
            canonical_json(supervision),
            path="record.residual_supervision",
        )
    try:
        validate_residual_unit_allocation_receipt(
            supervision.get("residual_unit_allocation")
            if isinstance(supervision.get("residual_unit_allocation"), dict)
            else {},
            unit_packet=residual.get("unit_packet")
            if isinstance(residual.get("unit_packet"), dict)
            else {},
        )
    except ResidualEconomicsFault as exc:
        raise KernelProtocolFault(
            exc.code,
            exc.detail,
            path="record.residual_supervision.residual_unit_allocation",
        ) from exc
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
    if (
        supervision.get("allocation_target_authority")
        != (
            "measured_structural_rate_distortion_with_calibrated_source_visible_importance"
        )
        or supervision.get("rate_distortion_optimality_claimed") is not False
    ):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_ALLOCATION_AUTHORITY_INVALID",
            canonical_json(supervision),
            path="record.residual_supervision.allocation_target_authority",
        )
    importance = supervision.get("importance")
    allocation = supervision.get("rate_distortion_allocation")
    if not isinstance(importance, dict) or not isinstance(allocation, dict):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_ECONOMICS_SUPERVISION_MISSING",
            canonical_json(supervision),
            path="record.residual_supervision",
        )
    importance_core = {
        key: copy.deepcopy(value)
        for key, value in importance.items()
        if key != "receipt_sha256"
    }
    if (
        importance.get("policy")
        != "project_theseus_kerc_calibrated_importance_policy_v1"
        or importance.get("receipt_sha256") != stable_hash(importance_core)
        or importance.get("target_fields_visible_to_policy") != []
    ):
        raise KernelProtocolFault(
            "KERC_IMPORTANCE_RECEIPT_INVALID",
            canonical_json(importance),
            path="record.residual_supervision.importance",
        )
    hrl_state = (
        record.get("hrl_state") if isinstance(record.get("hrl_state"), dict) else {}
    )
    try:
        validated_allocation = validate_structural_rate_distortion_allocation(
            allocation,
            kernel_program=packet.get("program") or {},
            global_state=hrl_state.get("global") or {},
            segment_residual=residual.get("segment_frame") or {},
            token_residuals=residual.get("token_tags") or [],
            exact_objects=packet.get("protected_objects") or {},
            exact_codec=residual.get("codec") or None,
        )
    except ResidualEconomicsFault as exc:
        raise KernelProtocolFault(
            exc.code,
            exc.detail,
            path="record.residual_supervision.rate_distortion_allocation",
        ) from exc
    if (
        validated_allocation["selected_fidelity"] != residual.get("fidelity")
        or int(supervision.get("record_fidelity_label", -1))
        != KERC_FIDELITY_LABELS[validated_allocation["selected_fidelity"]]
    ):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_ALLOCATION_PACKET_MISMATCH",
            canonical_json(validated_allocation),
            path="record.residual_supervision.rate_distortion_allocation",
        )
    return copy.deepcopy(supervision)


def _targeted_verifier_corruption(
    objective: str,
    target: str,
    *,
    protected_objects: dict[str, dict[str, Any]],
    record_identity: str,
) -> dict[str, Any]:
    if objective not in TRAINING_OBJECTIVES:
        raise KernelProtocolFault(
            "KERC_VERIFIER_CORRUPTION_OBJECTIVE_INVALID",
            objective,
            path="training_view.kerc_verifier_negative",
        )
    selector = int(
        stable_hash({"record_identity": record_identity, "objective": objective}).split(
            ":", 1
        )[1][:8],
        16,
    )
    if objective in {
        "surface_to_kernel_program_v1",
        "kernel_program_to_answer_packet_v1",
    }:
        corrupted, dimension, strategy = _structured_verifier_corruption(
            objective,
            target,
            protected_objects=protected_objects,
            selector=selector,
        )
    else:
        corrupted, dimension, strategy = _surface_verifier_corruption(
            objective,
            target,
            protected_objects=protected_objects,
            selector=selector,
        )
    if corrupted == target:
        raise KernelProtocolFault(
            "KERC_VERIFIER_CORRUPTION_NOOP",
            objective,
            path="training_view.kerc_verifier_negative",
        )
    labels = [1] * len(KERC_VERIFIER_DIMENSIONS)
    labels[KERC_VERIFIER_DIMENSIONS.index(dimension)] = 0
    return {
        "policy": "project_theseus_kerc_targeted_verifier_corruption_v1",
        "target": corrupted,
        "target_sha256": stable_hash(corrupted.encode("utf-8")),
        "labels": labels,
        "failed_dimension": dimension,
        "strategy": strategy,
        "strategy_selector": selector % len(KERC_VERIFIER_DIMENSIONS),
        "record_identity_sha256": stable_hash(record_identity.encode("utf-8")),
        "generator_loss_enabled": False,
        "unique_source_credit": 0,
        "candidate_generation_credit": 0,
    }


def _structured_verifier_corruption(
    objective: str,
    target: str,
    *,
    protected_objects: dict[str, dict[str, Any]],
    selector: int,
) -> tuple[str, str, str]:
    payload = json.loads(target)
    view = (
        payload.get("program")
        if objective == "surface_to_kernel_program_v1"
        else payload
    )
    expected_policy = (
        LEARNED_PROGRAM_TRANSPORT_POLICY
        if objective == "surface_to_kernel_program_v1"
        else LEARNED_ANSWER_TRANSPORT_POLICY
    )
    if (
        not isinstance(view, dict)
        or view.get("policy") != expected_policy
        or not isinstance(view.get("tokens"), list)
        or not view["tokens"]
    ):
        raise KernelProtocolFault(
            "KERC_VERIFIER_STRUCTURED_TARGET_INVALID",
            objective,
            path="training_view.kerc_verifier_negative",
        )
    options = list(range(5 if objective == "kernel_program_to_answer_packet_v1" else 4))
    options = options[selector % len(options) :] + options[: selector % len(options)]
    for option in options:
        candidate = copy.deepcopy(payload)
        candidate_view = (
            candidate["program"]
            if objective == "surface_to_kernel_program_v1"
            else candidate
        )
        candidate_tokens = candidate_view["tokens"]
        if option == 4:
            changed = _replace_compact_token_prefix(
                candidate_tokens,
                space="V_P",
                prefix="DECISION_CONF:",
                replacement=lambda value: "0" if float(value) > 0.0 else "1",
            )
            if not changed:
                continue
            return (
                canonical_json(candidate),
                "answer_decision_consistency",
                "change_answer_decision_confidence",
            )
        if option == 0:
            prefix = "OP:" if objective == "surface_to_kernel_program_v1" else "PRED:"
            if not _replace_compact_token_prefix(
                candidate_tokens,
                space="V_K",
                prefix=prefix,
                replacement=lambda value: "SEMANTIC_CONTRADICTION_" + value,
            ):
                continue
            return (
                canonical_json(candidate),
                "semantic_consistency",
                "replace_first_predicate",
            )
        if option == 1:
            if not _replace_compact_token_prefix(
                candidate_tokens,
                space="V_K",
                prefix="MOD:",
                replacement=lambda value: (
                    "POSSIBLE" if value != "POSSIBLE" else "REQUIRED"
                ),
            ):
                continue
            return (
                canonical_json(candidate),
                "semantic_consistency",
                "change_first_modality",
            )
        if option == 2:
            if not _replace_compact_token_prefix(
                candidate_tokens,
                space="V_K",
                prefix="POL:",
                replacement=lambda value: (
                    "NEGATED" if value != "NEGATED" else "AFFIRMED"
                ),
            ):
                continue
            return (
                canonical_json(candidate),
                "semantic_consistency",
                "flip_first_polarity",
            )
        handles = sorted(protected_objects)
        if len(handles) >= 2 and _replace_compact_token_prefix(
            candidate_tokens,
            space="V_P",
            prefix="HANDLE:",
            replacement=lambda value: next(
                (handle for handle in handles if handle != value), value
            ),
            require_change=True,
        ):
            return (
                canonical_json(candidate),
                "protected_object_consistency",
                "swap_first_protected_handle",
            )
        if _replace_compact_json_number(candidate_tokens):
            return (
                canonical_json(candidate),
                "numeric_value_consistency",
                "increment_first_numeric_value",
            )
    raise KernelProtocolFault(
        "KERC_VERIFIER_CORRUPTION_UNAVAILABLE",
        objective,
        path="training_view.kerc_verifier_negative",
    )


def _replace_compact_token_prefix(
    rows: list[Any],
    *,
    space: str,
    prefix: str,
    replacement: Callable[[str], str],
    require_change: bool = False,
) -> bool:
    code = _LEARNED_SPACE_CODE[space]
    for index, row in enumerate(rows):
        encoded_prefix = code + prefix
        if isinstance(row, str) and row.startswith(encoded_prefix):
            observed = row[len(encoded_prefix) :]
            changed = replacement(observed)
            if require_change and changed == observed:
                continue
            rows[index] = encoded_prefix + changed
            return True
    return False


def _replace_compact_json_number(rows: list[Any]) -> bool:
    for index, row in enumerate(rows):
        if not isinstance(row, str) or not row.startswith("P"):
            continue
        for prefix in ("NUMBER:", "QUANTITY:"):
            encoded_prefix = "P" + prefix
            if not row.startswith(encoded_prefix):
                continue
            try:
                payload = json.loads(row[len(encoded_prefix) :])
            except json.JSONDecodeError:
                continue
            if _increment_first_numeric_value(payload):
                rows[index] = encoded_prefix + canonical_json(payload)
                return True
    return False


def _surface_verifier_corruption(
    objective: str,
    target: str,
    *,
    protected_objects: dict[str, dict[str, Any]],
    selector: int,
) -> tuple[str, str, str]:
    options = list(range(4))
    options = options[selector % len(options) :] + options[: selector % len(options)]
    for option in options:
        if option == 3 and objective == "answer_packet_to_surface_v1":
            if '"' in target:
                return (
                    target.replace('"', "\u201c", 1),
                    "surface_fidelity",
                    "change_first_quote_glyph",
                )
            if "\n" in target:
                return (
                    target.replace("\n", " ", 1),
                    "surface_fidelity",
                    "replace_first_line_break",
                )
            return target + " ", "surface_fidelity", "append_exact_surface_space"
        if option == 0:
            exact_values = []
            for row in protected_objects.values():
                raw = row.get("inline_bytes_b64")
                if raw:
                    exact_values.append(base64.b64decode(str(raw)).decode("utf-8"))
            for exact in exact_values:
                if exact and exact in target:
                    replacement = "different protected value"
                    return (
                        target.replace(exact, replacement, 1),
                        "protected_object_consistency",
                        "replace_first_protected_object",
                    )
        if option == 1:
            match = re.search(r"(?<!\w)-?\d+(?:[,.]\d+)*(?!\w)", target)
            if match:
                raw = match.group(0)
                digits = re.sub(r"\D", "", raw)
                if digits:
                    changed = str(int(digits) + 1)
                    return (
                        target[: match.start()] + changed + target[match.end() :],
                        "numeric_value_consistency",
                        "increment_first_surface_number",
                    )
        if option == 2:
            negated = re.search(r"\b(?:not|never|no)\b", target, flags=re.IGNORECASE)
            if negated:
                changed = (target[: negated.start()] + target[negated.end() :]).strip()
                if changed:
                    return (
                        changed,
                        "semantic_consistency",
                        "remove_first_surface_negation",
                    )
            auxiliary = re.search(
                r"\b(?:is|are|was|were|can|could|will|would|should|has|have|had|does|do|did)\b",
                target,
                flags=re.IGNORECASE,
            )
            if auxiliary:
                return (
                    target[: auxiliary.end()] + " not" + target[auxiliary.end() :],
                    "semantic_consistency",
                    "insert_surface_negation",
                )
        if option == 3:
            word = re.search(r"\S+", target)
            if word:
                observed = word.group(0)
                normalized = observed.strip(".,!?;:\"'").lower()
                replacement = (
                    "no"
                    if normalized in {"yes", "true"}
                    else "yes"
                    if normalized in {"no", "false"}
                    else "different"
                )
                return (
                    target[: word.start()] + replacement + target[word.end() :],
                    "semantic_consistency",
                    "replace_first_surface_token",
                )
    raise KernelProtocolFault(
        "KERC_VERIFIER_CORRUPTION_UNAVAILABLE",
        "surface",
        path="training_view.kerc_verifier_negative",
    )


def _replace_first_handle(value: Any, handles: list[str]) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "handle" and value.get("value") in handles:
            observed = str(value["value"])
            value["value"] = next(handle for handle in handles if handle != observed)
            return True
        return any(_replace_first_handle(item, handles) for item in value.values())
    if isinstance(value, list):
        return any(_replace_first_handle(item, handles) for item in value)
    return False


def _replace_first_byte_literal(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "byte_literal" and isinstance(value.get("value"), str):
            value["value"] = base64.b64encode(b"semantically different value").decode(
                "ascii"
            )
            return True
        return any(_replace_first_byte_literal(item) for item in value.values())
    if isinstance(value, list):
        return any(_replace_first_byte_literal(item) for item in value)
    return False


def _increment_first_numeric_value(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key == "value"
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
            ):
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
    source: str | None = None,
    hrl_state: dict[str, Any] | None = None,
    concept_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
            str(
                decoded.get("kernel_version")
                if isinstance(decoded, dict)
                else type(decoded)
            ),
            path="compiler_output.kernel_version",
        )
    available_objects = copy.deepcopy(protected_objects)
    if "protected_objects" in decoded:
        if source is None or len(source) != source_character_length:
            raise KernelProtocolFault(
                "KERC_LEARNED_PROTECTED_SOURCE_REQUIRED",
                str(source_character_length),
                path="compiler_output.protected_objects",
            )
        generated_objects = materialize_learned_protected_objects(
            source, decoded["protected_objects"]
        )
        if protected_objects:
            expected_spans = learned_protected_span_view(protected_objects)
            observed_spans = learned_protected_span_view(generated_objects)
            if observed_spans != expected_spans:
                raise KernelProtocolFault(
                    "KERC_LEARNED_PROTECTED_OBJECTS_CHANGED",
                    canonical_json(
                        {"expected": expected_spans, "observed": observed_spans}
                    ),
                    path="compiler_output.protected_objects",
                )
            # Preserve deterministic evidence-plane provenance supplied by the
            # caller; the learned channel owns only the replayed span contract.
            generated_objects = copy.deepcopy(protected_objects)
        available_objects = generated_objects
    else:
        generated_objects = None
    learned_capsules = decoded.get("concept_capsules")
    if not isinstance(learned_capsules, dict):
        raise KernelProtocolFault(
            "KERC_LEARNED_COMPILER_CONCEPT_CAPSULES_MISSING",
            str(type(learned_capsules)),
            path="compiler_output.concept_capsules",
        )
    generated = materialize_learned_concept_capsules(
        learned_capsules,
        concept_resolver=concept_resolver,
    )
    conflicts = set(concept_capsules) & set(generated)
    if conflicts:
        raise KernelProtocolFault(
            "KERC_LEARNED_COMPILER_CONCEPT_COLLISION",
            canonical_json(sorted(conflicts)),
            path="compiler_output.concept_capsules",
        )
    available = {**copy.deepcopy(concept_capsules), **copy.deepcopy(generated)}
    canonical_program = materialize_learned_kernel_program(
        decoded.get("program"),
        protected_objects=available_objects,
        concept_capsules=available,
        source_character_length=source_character_length,
    )
    validated = validate_kernel_program(
        canonical_program,
        protected_objects=available_objects,
        concept_capsules=available,
        source_character_length=source_character_length,
    )
    validated["generated_concept_capsules"] = copy.deepcopy(generated)
    if generated_objects is not None:
        validated["generated_protected_objects"] = copy.deepcopy(generated_objects)
    if "residual" in decoded:
        if not isinstance(hrl_state, dict):
            raise KernelProtocolFault(
                "KERC_LEARNED_RESIDUAL_STATE_REQUIRED",
                "missing hrl_state",
                path="compiler_output.residual",
            )
        validated["learned_residual"] = validate_learned_residual_view(
            decoded["residual"],
            source_character_length=source_character_length,
            protected_objects=available_objects,
            hrl_state=hrl_state,
        )
    return validated


def parse_learned_answer_output(output: str) -> dict[str, Any]:
    """Parse and independently validate a learned core answer packet."""

    try:
        decoded = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise KernelProtocolFault(
            "KERC_LEARNED_ANSWER_OUTPUT_INVALID", str(exc), path="answer_output"
        ) from exc
    return materialize_learned_answer_packet(decoded)


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
        raise KernelProtocolFault(
            "KERC_SOURCE_ENCODING_UNSUPPORTED", str(exc), path="source"
        ) from exc
    record = {
        "record_type": "immutable_kernel_source_record",
        "source_sha256": stable_hash(raw),
        "byte_length": len(raw),
        "character_length": len(text),
        "encoding": encoding,
        "unicode_normalization": _normalization_form(text),
        "language_hint": language,
        "retention": retention,
        "inline_bytes_b64": base64.b64encode(raw).decode("ascii")
        if retain_inline
        else None,
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
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "EMAIL",
            "EXACT",
            80,
            "email",
        ),
        (
            r"\b(?:sha(?:1|224|256|384|512):)?[0-9a-fA-F]{32,128}\b",
            "HASH",
            "EXACT",
            78,
            "hash",
        ),
        (
            r"(?<!\w)(?:~?/|\./|\.\./)[A-Za-z0-9._~+@%/\-]+",
            "FILE_PATH",
            "EXACT",
            75,
            "file_path",
        ),
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
                SpanCandidate(
                    match.start(),
                    match.end(),
                    object_type,
                    copy_policy,
                    priority,
                    label,
                )
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
            "inline_bytes_b64": base64.b64encode(raw).decode("ascii")
            if retain_inline
            else None,
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
        "unprotected_character_count": len(source)
        - sum(span.end - span.start for span in selected),
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
            raise KernelProtocolFault(
                "KERC_CORRECTION_SPAN_INVALID", f"{start}:{end}", path=path
            )
        if any(
            start < other_end and end > other_start
            for other_start, other_end in protected_ranges
        ):
            raise KernelProtocolFault(
                "KERC_CORRECTION_TOUCHES_PROTECTED_OBJECT", f"{start}:{end}", path=path
            )
        alternatives = proposal.get("alternatives")
        if not isinstance(alternatives, list) or len(alternatives) < 2:
            raise KernelProtocolFault(
                "KERC_CORRECTION_ALTERNATIVES_INSUFFICIENT",
                "need at least two",
                path=path,
            )
        normalized = []
        seen = set()
        probability_sum = 0.0
        for alt_index, alternative in enumerate(alternatives):
            if not isinstance(alternative, dict):
                raise KernelProtocolFault(
                    "KERC_CORRECTION_ALTERNATIVE_INVALID",
                    str(alternative),
                    path=f"{path}.alternatives[{alt_index}]",
                )
            form = str(alternative.get("form") or "")
            probability = float(alternative.get("probability") or 0.0)
            if (
                not form
                or form in seen
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise KernelProtocolFault(
                    "KERC_CORRECTION_ALTERNATIVE_INVALID",
                    canonical_json(alternative),
                    path=f"{path}.alternatives[{alt_index}]",
                )
            seen.add(form)
            probability_sum += probability
            normalized.append(
                {
                    "form": form,
                    "probability": probability,
                    "evidence": alternative.get("evidence"),
                }
            )
        if not math.isclose(probability_sum, 1.0, abs_tol=1e-6):
            raise KernelProtocolFault(
                "KERC_CORRECTION_PROBABILITY_MASS_INVALID",
                str(probability_sum),
                path=path,
            )
        original = source[start:end]
        if original not in seen:
            raise KernelProtocolFault(
                "KERC_CORRECTION_ORIGINAL_NOT_RETAINED", original, path=path
            )
        rows.append(
            {
                "source_span": [start, end],
                "source_form": original,
                "alternatives": sorted(
                    normalized, key=lambda row: (-row["probability"], row["form"])
                ),
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
        raise KernelProtocolFault(
            "KERC_PROGRAM_NODES_MISSING",
            "nodes must be non-empty",
            path="program.nodes",
        )
    if not isinstance(roots, list) or not roots:
        raise KernelProtocolFault(
            "KERC_PROGRAM_ROOTS_MISSING",
            "roots must be non-empty",
            path="program.roots",
        )
    by_id: dict[str, dict[str, Any]] = {}
    refs: dict[str, set[str]] = {}
    handles_seen: set[str] = set()
    node_fields = {
        "node_id",
        "operator",
        "modality",
        "polarity",
        "quantifier",
        "confidence",
        "derivation",
        "source_spans",
        "arguments",
    }
    for index, node in enumerate(nodes):
        path = f"program.nodes[{index}]"
        if not isinstance(node, dict):
            raise KernelProtocolFault("KERC_NODE_INVALID", str(node), path=path)
        if set(node) - node_fields:
            raise KernelProtocolFault(
                "KERC_NODE_SCHEMA_INVALID",
                canonical_json(sorted(set(node) - node_fields)),
                path=path,
            )
        node_id = str(node.get("node_id") or "")
        if not re.fullmatch(r"k[0-9]+", node_id) or node_id in by_id:
            raise KernelProtocolFault(
                "KERC_NODE_ID_INVALID", node_id, path=f"{path}.node_id"
            )
        operator = str(node.get("operator") or "")
        if not re.fullmatch(r"(?:[A-Z][A-Z0-9_]*|@M[0-9]+)", operator):
            raise KernelProtocolFault(
                "KERC_OPERATOR_INVALID", operator, path=f"{path}.operator"
            )
        modality = str(node.get("modality") or "ASSERTED")
        polarity = str(node.get("polarity") or "AFFIRMED")
        quantifier = str(node.get("quantifier") or "NONE")
        confidence = float(node.get("confidence", 1.0))
        derivation = str(node.get("derivation") or "")
        if modality not in MODALITIES:
            raise KernelProtocolFault(
                "KERC_MODALITY_INVALID", modality, path=f"{path}.modality"
            )
        if polarity not in POLARITIES:
            raise KernelProtocolFault(
                "KERC_POLARITY_INVALID", polarity, path=f"{path}.polarity"
            )
        if quantifier not in QUANTIFIERS:
            raise KernelProtocolFault(
                "KERC_QUANTIFIER_INVALID", quantifier, path=f"{path}.quantifier"
            )
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise KernelProtocolFault(
                "KERC_CONFIDENCE_INVALID", str(confidence), path=f"{path}.confidence"
            )
        if derivation not in DERIVATIONS:
            raise KernelProtocolFault(
                "KERC_DERIVATION_INVALID", derivation, path=f"{path}.derivation"
            )
        arguments = node.get("arguments")
        if not isinstance(arguments, list):
            raise KernelProtocolFault(
                "KERC_ARGUMENTS_INVALID",
                "arguments must be a list",
                path=f"{path}.arguments",
            )
        node_refs: set[str] = set()
        for arg_index, argument in enumerate(arguments):
            arg_path = f"{path}.arguments[{arg_index}]"
            if (
                not isinstance(argument, dict)
                or set(argument) != {"role", "value"}
                or not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(argument.get("role") or ""))
            ):
                raise KernelProtocolFault(
                    "KERC_ROLE_INVALID", canonical_json(argument), path=arg_path
                )
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
                raise KernelProtocolFault(
                    "KERC_ALIGNMENT_SPAN_INVALID",
                    canonical_json(span),
                    path=f"{path}.source_spans[{span_index}]",
                )
            start, end = int(span[0]), int(span[1])
            if not 0 <= start < end <= source_character_length:
                raise KernelProtocolFault(
                    "KERC_ALIGNMENT_SPAN_INVALID",
                    f"{start}:{end}",
                    path=f"{path}.source_spans[{span_index}]",
                )
        by_id[node_id] = {
            "node_id": node_id,
            "operator": operator,
            "modality": modality,
            "polarity": polarity,
            "quantifier": quantifier,
            "confidence": confidence,
            "derivation": derivation,
            "source_spans": copy.deepcopy(node.get("source_spans") or []),
            "arguments": copy.deepcopy(arguments),
        }
        refs[node_id] = node_refs
    unknown_roots = sorted(set(str(root) for root in roots) - set(by_id))
    unknown_refs = sorted(
        {ref for values in refs.values() for ref in values if ref not in by_id}
    )
    if unknown_roots:
        raise KernelProtocolFault(
            "KERC_ROOT_REFERENCE_UNKNOWN", ",".join(unknown_roots), path="program.roots"
        )
    if unknown_refs:
        raise KernelProtocolFault(
            "KERC_NODE_REFERENCE_UNKNOWN", ",".join(unknown_refs), path="program.nodes"
        )
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
    tokens: list[dict[str, str]] = [
        token("V_P", f"VERSION:{KERNEL_VERSION}"),
        token("V_P", f"SERIALIZATION:{SERIALIZATION_VERSION}"),
    ]
    for node in canonical_program["nodes"]:
        tokens.extend(
            [
                token("V_P", "NODE_BEGIN"),
                token("V_P", f"NODE_ID:{node['node_id']}"),
                token("V_K", f"OP:{node['operator']}"),
                token("V_K", f"MOD:{node.get('modality', 'ASSERTED')}"),
                token("V_K", f"POL:{node.get('polarity', 'AFFIRMED')}"),
                token("V_K", f"QUANT:{node.get('quantifier', 'NONE')}"),
                token("V_P", f"CONF:{float(node.get('confidence', 1.0)):.17g}"),
                token("V_P", f"DERIV:{node.get('derivation', '')}"),
                token("V_P", f"SPANS:{canonical_json(node.get('source_spans') or [])}"),
            ]
        )
        for argument in node.get("arguments") or []:
            tokens.append(token("V_K", f"ROLE:{argument['role']}"))
            tokens.extend(_serialize_value(argument["value"]))
        tokens.append(token("V_P", "NODE_END"))
    for root in canonical_program["roots"]:
        tokens.append(token("V_P", f"ROOT:{root}"))
    tokens.append(token("V_P", "PROGRAM_END"))
    macro_registry = validate_macro_registry(macros)
    compact = apply_macros(tokens, macro_registry)
    expanded = expand_macros(compact, macro_registry)
    if expanded != tokens:
        raise KernelProtocolFault(
            "KERC_MACRO_ROUNDTRIP_MISMATCH",
            "expanded tokens differ",
            path="serialization",
        )
    return {
        "policy": "project_theseus_kernel_three_code_space_serialization_v2",
        "serialization_version": SERIALIZATION_VERSION,
        "expanded_tokens": tokens,
        "compact_tokens": compact,
        "macro_registry": macro_registry,
        "expanded_sha256": stable_hash(tokens),
        "compact_sha256": stable_hash(compact),
        "macro_roundtrip_exact": True,
        "code_space_counts": {
            space: sum(1 for row in compact if row["space"] == space)
            for space in CODE_SPACES
        },
        **NO_CHEAT,
    }


def deserialize_kernel_program(
    serialization: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    source_character_length: int,
) -> dict[str, Any]:
    """Decode the complete typed token stream and revalidate its program.

    Macro replay alone is not a semantic round trip. This decoder requires every
    program field that affects identity, including roots, source alignment,
    confidence, and derivation, to survive the serialized form.
    """

    if (
        not isinstance(serialization, dict)
        or serialization.get("policy")
        != "project_theseus_kernel_three_code_space_serialization_v2"
        or serialization.get("serialization_version") != SERIALIZATION_VERSION
    ):
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_VERSION_INVALID",
            str(
                serialization.get("serialization_version")
                if isinstance(serialization, dict)
                else type(serialization)
            ),
            path="serialization",
        )
    compact = serialization.get("compact_tokens")
    macros = serialization.get("macro_registry")
    if not isinstance(compact, list) or not isinstance(macros, list):
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_TOKEN_STREAM_INVALID",
            "compact tokens and macro registry must be lists",
            path="serialization",
        )
    if stable_hash(compact) != serialization.get("compact_sha256"):
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_COMPACT_STREAM_MISMATCH",
            "compact token hash differs",
            path="serialization.compact_tokens",
        )
    expanded = expand_macros(compact, macros)
    stored_expanded = serialization.get("expanded_tokens")
    if expanded != stored_expanded or stable_hash(expanded) != serialization.get(
        "expanded_sha256"
    ):
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_EXPANDED_STREAM_MISMATCH",
            "expanded tokens differ from the committed stream",
            path="serialization.expanded_tokens",
        )
    cursor = _KernelTokenCursor(expanded)
    cursor.expect("V_P", f"VERSION:{KERNEL_VERSION}")
    cursor.expect("V_P", f"SERIALIZATION:{SERIALIZATION_VERSION}")
    nodes: list[dict[str, Any]] = []
    roots: list[str] = []
    program_end_seen = False
    while not cursor.done:
        current = cursor.peek()
        if current == {"space": "V_P", "token": "PROGRAM_END"}:
            cursor.take()
            program_end_seen = True
            break
        if current["space"] == "V_P" and current["token"].startswith("ROOT:"):
            roots.append(cursor.take()["token"].removeprefix("ROOT:"))
            continue
        cursor.expect("V_P", "NODE_BEGIN")
        node_id = cursor.expect_prefix("V_P", "NODE_ID:")
        operator = cursor.expect_prefix("V_K", "OP:")
        modality = cursor.expect_prefix("V_K", "MOD:")
        polarity = cursor.expect_prefix("V_K", "POL:")
        quantifier = cursor.expect_prefix("V_K", "QUANT:")
        confidence_text = cursor.expect_prefix("V_P", "CONF:")
        derivation = cursor.expect_prefix("V_P", "DERIV:")
        spans_text = cursor.expect_prefix("V_P", "SPANS:")
        try:
            confidence = float(confidence_text)
            source_spans = json.loads(spans_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise KernelProtocolFault(
                "KERC_SERIALIZATION_NODE_METADATA_INVALID",
                f"{node_id}:{exc}",
                path="serialization.expanded_tokens",
            ) from exc
        arguments: list[dict[str, Any]] = []
        while cursor.peek() != {"space": "V_P", "token": "NODE_END"}:
            role = cursor.expect_prefix("V_K", "ROLE:")
            arguments.append({"role": role, "value": _deserialize_value(cursor)})
        cursor.take()
        nodes.append(
            {
                "node_id": node_id,
                "operator": operator,
                "modality": modality,
                "polarity": polarity,
                "quantifier": quantifier,
                "confidence": confidence,
                "derivation": derivation,
                "source_spans": source_spans,
                "arguments": arguments,
            }
        )
    if not program_end_seen:
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_PROGRAM_END_MISSING",
            "typed program stream ended without PROGRAM_END",
            path="serialization.expanded_tokens",
        )
    if not cursor.done:
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_TRAILING_TOKENS",
            canonical_json(cursor.remaining()),
            path="serialization.expanded_tokens",
        )
    validated = validate_kernel_program(
        {"roots": roots, "nodes": nodes},
        protected_objects=protected_objects,
        concept_capsules=concept_capsules,
        source_character_length=source_character_length,
    )
    return {
        "state": "READY",
        "canonical_program": validated["canonical_program"],
        "expanded_sha256": stable_hash(expanded),
        "exact_program_roundtrip": True,
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
            raise KernelProtocolFault(
                "KERC_MACRO_ID_INVALID", macro_id, path=f"{path}.macro_id"
            )
        expansion = raw.get("expansion")
        if not isinstance(expansion, list) or len(expansion) < 2:
            raise KernelProtocolFault(
                "KERC_MACRO_EXPANSION_INVALID",
                "need at least two tokens",
                path=f"{path}.expansion",
            )
        normalized = [
            _validate_token(row, path=f"{path}.expansion") for row in expansion
        ]
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


def apply_macros(
    tokens: Sequence[dict[str, str]], registry: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
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
        output.append(
            token("V_P", f"MACRO:{matched['macro_id']}:{matched['expansion_sha256']}")
        )
        index += len(matched["expansion"])
    return output


def expand_macros(
    tokens: Sequence[dict[str, str]], registry: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
    by_id = {row["macro_id"]: row for row in registry}
    output: list[dict[str, str]] = []
    for index, raw in enumerate(tokens):
        row = _validate_token(raw, path=f"tokens[{index}]")
        if row["space"] != "V_P" or not row["token"].startswith("MACRO:"):
            output.append(row)
            continue
        parts = row["token"].split(":", 2)
        if len(parts) != 3 or parts[1] not in by_id:
            raise KernelProtocolFault(
                "KERC_MACRO_REFERENCE_UNKNOWN", row["token"], path=f"tokens[{index}]"
            )
        macro = by_id[parts[1]]
        if parts[2] != macro["expansion_sha256"]:
            raise KernelProtocolFault(
                "KERC_MACRO_EXPANSION_HASH_MISMATCH",
                row["token"],
                path=f"tokens[{index}]",
            )
        output.extend(copy.deepcopy(macro["expansion"]))
    return output


def _validate_residual_spans(
    spans: Any, *, source_character_length: int, path: str
) -> list[list[int]]:
    if not isinstance(spans, list):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_SPANS_INVALID", canonical_json(spans), path=path
        )
    normalized: list[list[int]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, list) or len(span) != 2:
            raise KernelProtocolFault(
                "KERC_RESIDUAL_SPAN_INVALID",
                canonical_json(span),
                path=f"{path}[{index}]",
            )
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= source_character_length:
            raise KernelProtocolFault(
                "KERC_RESIDUAL_SPAN_INVALID", f"{start}:{end}", path=f"{path}[{index}]"
            )
        normalized.append([start, end])
    unique = sorted(set(tuple(row) for row in normalized))
    return [list(row) for row in unique]


def _normalize_segment_frame(
    segment_frame: dict[str, Any] | None,
    *,
    source_character_length: int,
    path: str,
) -> dict[str, Any]:
    segment = copy.deepcopy(segment_frame or {})
    if not segment:
        return {}
    single_fields = {"frame_name", "lexical_unit", "target_spans", "frame_roles"}
    if set(segment) == single_fields:
        if not str(segment["frame_name"]).strip():
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_NAME_MISSING", "", path=f"{path}.frame_name"
            )
        segment["target_spans"] = _validate_residual_spans(
            segment["target_spans"],
            source_character_length=source_character_length,
            path=f"{path}.target_spans",
        )
        roles = segment["frame_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() for role in roles)
        ):
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_ROLES_INVALID",
                canonical_json(roles),
                path=f"{path}.frame_roles",
            )
        segment["frame_roles"] = sorted(set(roles))
        return segment
    if segment.get("schema") == "erst_discourse_graph_v1":
        required = {"schema", "document_id", "anchor_edu_id", "units", "edges"}
        if set(segment) != required:
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_SCHEMA_INVALID", canonical_json(segment), path=path
            )
        document_id = str(segment.get("document_id") or "")
        anchor_edu_id = segment.get("anchor_edu_id")
        if (
            not re.fullmatch(r"GUM_[a-z0-9_]+", document_id)
            or not isinstance(anchor_edu_id, int)
            or anchor_edu_id <= 0
        ):
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_IDENTITY_INVALID",
                f"{document_id}:{anchor_edu_id}",
                path=path,
            )
        units = segment.get("units")
        unit_fields = {
            "edu_id",
            "node_id",
            "target_spans",
            "tree_depth",
            "source_row_sha256",
        }
        if not isinstance(units, list) or not 2 <= len(units) <= 32:
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_UNIT_CARDINALITY_INVALID",
                canonical_json(units),
                path=f"{path}.units",
            )
        normalized_units = []
        units_by_node: dict[str, dict[str, Any]] = {}
        edu_ids: set[int] = set()
        for index, unit in enumerate(units):
            unit_path = f"{path}.units[{index}]"
            if not isinstance(unit, dict) or set(unit) != unit_fields:
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_UNIT_SCHEMA_INVALID",
                    canonical_json(unit),
                    path=unit_path,
                )
            edu_id = unit.get("edu_id")
            node_id = str(unit.get("node_id") or "")
            tree_depth = unit.get("tree_depth")
            source_hash = str(unit.get("source_row_sha256") or "")
            if (
                not isinstance(edu_id, int)
                or edu_id <= 0
                or edu_id in edu_ids
                or not re.fullmatch(r"k[0-9]+", node_id)
                or node_id in units_by_node
                or not isinstance(tree_depth, int)
                or tree_depth < 0
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash)
            ):
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_UNIT_VALUE_INVALID",
                    canonical_json(unit),
                    path=unit_path,
                )
            normalized = copy.deepcopy(unit)
            normalized["target_spans"] = _validate_residual_spans(
                unit["target_spans"],
                source_character_length=source_character_length,
                path=f"{unit_path}.target_spans",
            )
            if len(normalized["target_spans"]) != 1:
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_UNIT_SPAN_INVALID",
                    canonical_json(unit["target_spans"]),
                    path=f"{unit_path}.target_spans",
                )
            edu_ids.add(edu_id)
            units_by_node[node_id] = normalized
            normalized_units.append(normalized)
        if anchor_edu_id not in edu_ids:
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_ANCHOR_UNKNOWN",
                str(anchor_edu_id),
                path=f"{path}.anchor_edu_id",
            )
        edges = segment.get("edges")
        edge_fields = {
            "edge_id",
            "edge_order",
            "node_id",
            "edge_kind",
            "child_node_id",
            "parent_node_id",
            "relation",
            "nuclearity",
            "source_annotation_sha256",
            "signal_count",
        }
        if not isinstance(edges, list) or not 1 <= len(edges) <= 32:
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_EDGE_CARDINALITY_INVALID",
                canonical_json(edges),
                path=f"{path}.edges",
            )
        normalized_edges = []
        observed_edge_ids: set[str] = set()
        observed_node_ids = set(units_by_node)
        primary_count = 0
        for index, edge in enumerate(edges):
            edge_path = f"{path}.edges[{index}]"
            if not isinstance(edge, dict) or set(edge) != edge_fields:
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_EDGE_SCHEMA_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            edge_id = str(edge.get("edge_id") or "")
            node_id = str(edge.get("node_id") or "")
            edge_kind = str(edge.get("edge_kind") or "")
            child = str(edge.get("child_node_id") or "")
            parent = str(edge.get("parent_node_id") or "")
            relation = str(edge.get("relation") or "")
            nuclearity = str(edge.get("nuclearity") or "")
            source_hash = str(edge.get("source_annotation_sha256") or "")
            edge_order = edge.get("edge_order")
            signal_count = edge.get("signal_count")
            if (
                not re.fullmatch(r"edge-[0-9]+", edge_id)
                or edge_id in observed_edge_ids
                or not re.fullmatch(r"k[0-9]+", node_id)
                or node_id in observed_node_ids
                or edge_kind not in {"primary", "secondary"}
                or child not in units_by_node
                or parent not in units_by_node
                or child == parent
                or not re.fullmatch(r"[a-z][a-z0-9-]*(?:_[rm])?", relation)
                or nuclearity
                not in {"multinuclear", "satellite_nucleus", "secondary_unspecified"}
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash)
                or not isinstance(edge_order, int)
                or edge_order < 0
                or not isinstance(signal_count, int)
                or signal_count < 0
            ):
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_EDGE_VALUE_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            expected_nuclearity = (
                "multinuclear"
                if relation.endswith("_m")
                else "satellite_nucleus"
                if relation.endswith("_r")
                else "secondary_unspecified"
            )
            if nuclearity != expected_nuclearity:
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_NUCLEARITY_INVALID",
                    f"{relation}:{nuclearity}",
                    path=f"{edge_path}.nuclearity",
                )
            if edge_kind == "primary":
                primary_count += 1
                if edge_order != 0 or not relation.endswith(("_m", "_r")):
                    raise KernelProtocolFault(
                        "KERC_ERST_DISCOURSE_PRIMARY_INVALID",
                        canonical_json(edge),
                        path=edge_path,
                    )
            elif edge_order < 0 or relation.endswith(("_m", "_r")):
                raise KernelProtocolFault(
                    "KERC_ERST_DISCOURSE_SECONDARY_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            observed_edge_ids.add(edge_id)
            observed_node_ids.add(node_id)
            normalized_edges.append(copy.deepcopy(edge))
        if primary_count != 1:
            raise KernelProtocolFault(
                "KERC_ERST_DISCOURSE_PRIMARY_COUNT_INVALID",
                str(primary_count),
                path=f"{path}.edges",
            )
        segment["units"] = sorted(normalized_units, key=lambda row: row["edu_id"])
        segment["edges"] = sorted(
            normalized_edges,
            key=lambda row: (
                row["edge_kind"] != "primary",
                row["edge_order"],
                row["edge_id"],
            ),
        )
        return segment
    if segment.get("schema") == "event_coreference_group_v1":
        required = {
            "schema",
            "group_id",
            "group_node_id",
            "group_claim_id",
            "mentions",
        }
        if set(segment) != required:
            raise KernelProtocolFault(
                "KERC_EVENT_COREFERENCE_SCHEMA_INVALID",
                canonical_json(segment),
                path=path,
            )
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", str(segment.get("group_id") or "")):
            raise KernelProtocolFault(
                "KERC_EVENT_COREFERENCE_GROUP_ID_INVALID",
                str(segment.get("group_id")),
                path=f"{path}.group_id",
            )
        if not re.fullmatch(r"k[0-9]+", str(segment.get("group_node_id") or "")):
            raise KernelProtocolFault(
                "KERC_EVENT_COREFERENCE_NODE_ID_INVALID",
                str(segment.get("group_node_id")),
                path=f"{path}.group_node_id",
            )
        if not str(segment.get("group_claim_id") or "").strip():
            raise KernelProtocolFault(
                "KERC_EVENT_COREFERENCE_CLAIM_ID_INVALID",
                str(segment.get("group_claim_id")),
                path=f"{path}.group_claim_id",
            )
        mentions = segment.get("mentions")
        if not isinstance(mentions, list) or not 2 <= len(mentions) <= 128:
            raise KernelProtocolFault(
                "KERC_EVENT_COREFERENCE_CARDINALITY_INVALID",
                canonical_json(mentions),
                path=f"{path}.mentions",
            )
        mention_fields = {
            "node_id",
            "claim_id",
            "event_type",
            "target_spans",
            "source_annotation_sha256",
        }
        node_ids = {str(segment["group_node_id"])}
        claim_ids = {str(segment["group_claim_id"])}
        normalized_mentions = []
        for index, mention in enumerate(mentions):
            mention_path = f"{path}.mentions[{index}]"
            if not isinstance(mention, dict) or set(mention) != mention_fields:
                raise KernelProtocolFault(
                    "KERC_EVENT_COREFERENCE_MENTION_SCHEMA_INVALID",
                    canonical_json(mention),
                    path=mention_path,
                )
            node_id = str(mention.get("node_id") or "")
            claim_id = str(mention.get("claim_id") or "")
            event_type = str(mention.get("event_type") or "")
            source_hash = str(mention.get("source_annotation_sha256") or "")
            if (
                not re.fullmatch(r"k[0-9]+", node_id)
                or node_id in node_ids
                or not claim_id.strip()
                or claim_id in claim_ids
                or not re.fullmatch(r"[a-z][a-z0-9_.:-]*", event_type)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash)
            ):
                raise KernelProtocolFault(
                    "KERC_EVENT_COREFERENCE_MENTION_VALUE_INVALID",
                    canonical_json(mention),
                    path=mention_path,
                )
            node_ids.add(node_id)
            claim_ids.add(claim_id)
            normalized_mention = copy.deepcopy(mention)
            normalized_mention["target_spans"] = _validate_residual_spans(
                mention["target_spans"],
                source_character_length=source_character_length,
                path=f"{mention_path}.target_spans",
            )
            normalized_mentions.append(normalized_mention)
        segment["mentions"] = sorted(
            normalized_mentions,
            key=lambda row: (row["target_spans"], _node_sort_key(row["node_id"])),
        )
        return segment
    if segment.get("schema") == "mpqa_relation_chain_v1":
        required = {"schema", "relation_id", "members", "edges"}
        if set(segment) != required:
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_SCHEMA_INVALID", canonical_json(segment), path=path
            )
        relation_id = str(segment.get("relation_id") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", relation_id):
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_ID_INVALID",
                relation_id,
                path=f"{path}.relation_id",
            )
        members = segment.get("members")
        if not isinstance(members, list) or not 4 <= len(members) <= 128:
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_MEMBER_CARDINALITY_INVALID",
                canonical_json(members),
                path=f"{path}.members",
            )
        member_fields = {
            "node_id",
            "claim_id",
            "member_type",
            "concept_id",
            "target_spans",
            "source_annotation_sha256",
            "implicit",
            "span_status",
        }
        allowed_member_types = {"expression", "source", "attitude", "target"}
        node_ids: set[str] = set()
        claim_ids: set[str] = set()
        members_by_node: dict[str, dict[str, Any]] = {}
        member_type_counts: Counter[str] = Counter()
        normalized_members = []
        for index, member in enumerate(members):
            member_path = f"{path}.members[{index}]"
            if not isinstance(member, dict) or set(member) != member_fields:
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_MEMBER_SCHEMA_INVALID",
                    canonical_json(member),
                    path=member_path,
                )
            node_id = str(member.get("node_id") or "")
            claim_id = str(member.get("claim_id") or "")
            member_type = str(member.get("member_type") or "")
            concept_id = str(member.get("concept_id") or "")
            source_hash = str(member.get("source_annotation_sha256") or "")
            implicit = member.get("implicit")
            span_status = str(member.get("span_status") or "")
            if (
                not re.fullmatch(r"k[0-9]+", node_id)
                or node_id in node_ids
                or not claim_id.strip()
                or claim_id in claim_ids
                or member_type not in allowed_member_types
                or not re.fullmatch(r"[a-z][a-z0-9_.:-]*", concept_id)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash)
                or not isinstance(implicit, bool)
                or span_status
                not in {"explicit", "declared_implicit", "zero_width_annotation"}
                or implicit != (span_status == "declared_implicit")
                or (
                    span_status == "declared_implicit"
                    and member_type not in {"expression", "source"}
                )
            ):
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_MEMBER_VALUE_INVALID",
                    canonical_json(member),
                    path=member_path,
                )
            normalized = copy.deepcopy(member)
            normalized["target_spans"] = _validate_residual_spans(
                member["target_spans"],
                source_character_length=source_character_length,
                path=f"{member_path}.target_spans",
            )
            if (span_status == "explicit" and not normalized["target_spans"]) or (
                span_status in {"declared_implicit", "zero_width_annotation"}
                and normalized["target_spans"]
            ):
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_MEMBER_SPAN_INVALID",
                    canonical_json(member),
                    path=f"{member_path}.target_spans",
                )
            node_ids.add(node_id)
            claim_ids.add(claim_id)
            members_by_node[node_id] = normalized
            member_type_counts[member_type] += 1
            normalized_members.append(normalized)
        if (
            member_type_counts["expression"] != 1
            or member_type_counts["source"] < 1
            or member_type_counts["attitude"] < 1
            or member_type_counts["target"] < 1
        ):
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_MEMBER_TYPES_INCOMPLETE",
                canonical_json(member_type_counts),
                path=f"{path}.members",
            )
        edges = segment.get("edges")
        if not isinstance(edges, list) or not edges:
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_EDGES_INVALID",
                canonical_json(edges),
                path=f"{path}.edges",
            )
        edge_fields = {
            "edge_type",
            "from_node_id",
            "to_node_id",
            "order",
            "source_field",
        }
        edge_rules = {
            "nested_source_member": ("expression", "source", "nested-source"),
            "attitude_link": ("expression", "attitude", "attitude-link"),
            "target_link": ("attitude", "target", "target-link"),
        }
        normalized_edges = []
        observed_edges: set[tuple[str, str, str, int]] = set()
        incoming_nodes: set[str] = set()
        source_orders: list[int] = []
        for index, edge in enumerate(edges):
            edge_path = f"{path}.edges[{index}]"
            if not isinstance(edge, dict) or set(edge) != edge_fields:
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_EDGE_SCHEMA_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            edge_type = str(edge.get("edge_type") or "")
            from_node = str(edge.get("from_node_id") or "")
            to_node = str(edge.get("to_node_id") or "")
            source_field = str(edge.get("source_field") or "")
            order = edge.get("order")
            if (
                edge_type not in edge_rules
                or from_node not in members_by_node
                or to_node not in members_by_node
                or from_node == to_node
                or not isinstance(order, int)
            ):
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_EDGE_VALUE_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            expected_from, expected_to, expected_field = edge_rules[edge_type]
            if (
                members_by_node[from_node]["member_type"] != expected_from
                or members_by_node[to_node]["member_type"] != expected_to
                or source_field != expected_field
                or (edge_type == "nested_source_member" and order < 0)
                or (edge_type != "nested_source_member" and order != -1)
            ):
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_EDGE_TYPE_INVALID",
                    canonical_json(edge),
                    path=edge_path,
                )
            identity = (edge_type, from_node, to_node, order)
            if identity in observed_edges:
                raise KernelProtocolFault(
                    "KERC_MPQA_RELATION_EDGE_DUPLICATE",
                    canonical_json(edge),
                    path=edge_path,
                )
            observed_edges.add(identity)
            incoming_nodes.add(to_node)
            if edge_type == "nested_source_member":
                source_orders.append(order)
            normalized_edges.append(copy.deepcopy(edge))
        expression_node = next(
            node_id
            for node_id, member in members_by_node.items()
            if member["member_type"] == "expression"
        )
        if incoming_nodes != node_ids - {expression_node}:
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_GRAPH_INCOMPLETE",
                canonical_json(sorted(node_ids - {expression_node} - incoming_nodes)),
                path=f"{path}.edges",
            )
        if sorted(source_orders) != list(range(len(source_orders))):
            raise KernelProtocolFault(
                "KERC_MPQA_RELATION_SOURCE_ORDER_INVALID",
                canonical_json(source_orders),
                path=f"{path}.edges",
            )
        segment["members"] = sorted(
            normalized_members, key=lambda row: _node_sort_key(row["node_id"])
        )
        segment["edges"] = sorted(
            normalized_edges,
            key=lambda row: (
                row["edge_type"],
                _node_sort_key(row["from_node_id"]),
                row["order"],
                _node_sort_key(row["to_node_id"]),
            ),
        )
        return segment
    if (
        set(segment) != {"schema", "frames"}
        or segment.get("schema") != "framenet_composite_v1"
    ):
        raise KernelProtocolFault(
            "KERC_SEGMENT_FRAME_SCHEMA_INVALID",
            canonical_json(segment),
            path=path,
        )
    frames = segment.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 8:
        raise KernelProtocolFault(
            "KERC_SEGMENT_FRAME_SET_CARDINALITY_INVALID",
            canonical_json(frames),
            path=f"{path}.frames",
        )
    frame_fields = {
        "node_id",
        "claim_id",
        "annotation_set_node",
        "annotation_set_id",
        "frame_name",
        "lexical_unit",
        "target_spans",
        "frame_roles",
        "source_annotation_sha256",
    }
    node_ids: set[str] = set()
    claim_ids: set[str] = set()
    for index, frame in enumerate(frames):
        frame_path = f"{path}.frames[{index}]"
        if not isinstance(frame, dict) or set(frame) != frame_fields:
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_SET_ENTRY_SCHEMA_INVALID",
                canonical_json(frame),
                path=frame_path,
            )
        required_strings = (
            "node_id",
            "claim_id",
            "annotation_set_node",
            "frame_name",
            "lexical_unit",
        )
        if any(
            not isinstance(frame[field], str) or not frame[field].strip()
            for field in required_strings
        ):
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_SET_ENTRY_VALUE_INVALID",
                canonical_json(frame),
                path=frame_path,
            )
        if not isinstance(frame["source_annotation_sha256"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", frame["source_annotation_sha256"]
        ):
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_SET_SOURCE_HASH_INVALID",
                str(frame["source_annotation_sha256"]),
                path=f"{frame_path}.source_annotation_sha256",
            )
        if frame["node_id"] in node_ids or frame["claim_id"] in claim_ids:
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_SET_ID_DUPLICATE",
                canonical_json(frame),
                path=frame_path,
            )
        node_ids.add(frame["node_id"])
        claim_ids.add(frame["claim_id"])
        frame["target_spans"] = _validate_residual_spans(
            frame["target_spans"],
            source_character_length=source_character_length,
            path=f"{frame_path}.target_spans",
        )
        roles = frame["frame_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() for role in roles)
        ):
            raise KernelProtocolFault(
                "KERC_SEGMENT_FRAME_ROLES_INVALID",
                canonical_json(roles),
                path=f"{frame_path}.frame_roles",
            )
        frame["frame_roles"] = sorted(set(roles))
    return segment


def build_kernel_packet(
    source: str,
    program: dict[str, Any],
    *,
    hrl_state: dict[str, Any],
    correction_lattice: dict[str, Any] | None = None,
    concept_capsules: dict[str, dict[str, Any]] | None = None,
    explicit_spans: Sequence[dict[str, Any]] = (),
    macros: Sequence[dict[str, Any]] = (),
    segment_frame: dict[str, Any] | None = None,
    token_tags: Sequence[dict[str, Any]] = (),
    residual_mode: str = "SOURCE_RECONSTRUCTION",
    fidelity: str = "faithful",
    retain_source_inline: bool = False,
    retain_objects_inline: bool = True,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if residual_mode not in RESIDUAL_MODES:
        raise KernelProtocolFault(
            "KERC_RESIDUAL_MODE_INVALID", residual_mode, path="residual.mode"
        )
    if fidelity not in FIDELITY_MODES:
        raise KernelProtocolFault(
            "KERC_FIDELITY_INVALID", fidelity, path="residual.fidelity"
        )
    _validate_hrl_reference(hrl_state)
    source_record = capture_source(source, retain_inline=retain_source_inline)
    protected = extract_protected_objects(
        source, explicit_spans=explicit_spans, retain_inline=retain_objects_inline
    )
    capsules = copy.deepcopy(concept_capsules or {})
    _validate_concept_capsules(capsules)
    lattice = correction_lattice or build_correction_lattice(
        source, protected["protected_objects"], []
    )
    validated = validate_kernel_program(
        program,
        protected_objects=protected["protected_objects"],
        concept_capsules=capsules,
        source_character_length=len(source),
    )
    serialization = serialize_kernel_program(
        validated["canonical_program"], macros=macros
    )
    normalized_segment = _normalize_segment_frame(
        segment_frame,
        source_character_length=len(source),
        path="residual.segment_frame",
    )
    normalized_tags: list[dict[str, Any]] = []
    for index, raw_tag in enumerate(token_tags):
        if not isinstance(raw_tag, dict) or set(raw_tag) != {
            "tag",
            "source_span",
            "authority",
        }:
            raise KernelProtocolFault(
                "KERC_TOKEN_TAG_SCHEMA_INVALID",
                canonical_json(raw_tag),
                path=f"residual.token_tags[{index}]",
            )
        if (
            not str(raw_tag["tag"]).strip()
            or raw_tag["authority"] != "licensed_manual_annotation"
        ):
            raise KernelProtocolFault(
                "KERC_TOKEN_TAG_AUTHORITY_INVALID",
                canonical_json(raw_tag),
                path=f"residual.token_tags[{index}]",
            )
        span = _validate_residual_spans(
            [raw_tag["source_span"]],
            source_character_length=len(source),
            path=f"residual.token_tags[{index}].source_span",
        )[0]
        normalized_tags.append(
            {
                "tag": str(raw_tag["tag"]),
                "source_span": span,
                "authority": raw_tag["authority"],
            }
        )
    normalized_tags.sort(key=lambda row: (row["source_span"], row["tag"]))
    residual = {
        "mode": residual_mode,
        "fidelity": fidelity,
        "hrl_version": hrl_state["hrl_version"],
        "global_state_hash": hrl_state["state_hash"],
        "interaction_id": hrl_state["interaction_id"],
        "segment_frame": normalized_segment,
        "token_tags": normalized_tags,
        "exact_object_handles": sorted(protected["protected_objects"]),
        "missing_state_behavior": "reject_or_request_checkpoint_without_approximation",
    }
    residual["codec"] = build_residual_codec(
        kernel_program=validated["canonical_program"],
        global_state=hrl_state["global"],
        segment_residual=normalized_segment,
        token_residuals=normalized_tags,
        exact_objects=protected["protected_objects"],
    )
    residual["unit_packet"] = build_residual_unit_packet(
        source_record_sha256=source_record["record_sha256"],
        residual_mode=residual_mode,
        kernel_program=validated["canonical_program"],
        global_state=hrl_state["global"],
        segment_residual=normalized_segment,
        token_residuals=normalized_tags,
        concept_capsules=capsules,
        exact_objects=protected["protected_objects"],
    )
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
    packet_core["packet_id"] = (
        "kpacket:" + stable_hash(packet_core).split(":", 1)[1][:24]
    )
    packet_core["packet_sha256"] = stable_hash(packet_core)
    validate_kernel_packet(packet_core, local_hrl_state=hrl_state)
    return packet_core


def revise_kernel_packet_fidelity(
    packet: dict[str, Any],
    fidelity: str,
    *,
    local_hrl_state: dict[str, Any],
) -> dict[str, Any]:
    """Change only the declared residual fidelity and rebind packet identities."""

    if fidelity not in FIDELITY_MODES:
        raise KernelProtocolFault(
            "KERC_FIDELITY_INVALID", fidelity, path="residual.fidelity"
        )
    validate_kernel_packet(packet, local_hrl_state=local_hrl_state)
    revised = copy.deepcopy(packet)
    revised["residual"]["fidelity"] = fidelity
    revised.pop("packet_id", None)
    revised.pop("packet_sha256", None)
    revised["packet_id"] = "kpacket:" + stable_hash(revised).split(":", 1)[1][:24]
    revised["packet_sha256"] = stable_hash(revised)
    expected = copy.deepcopy(packet)
    expected["residual"]["fidelity"] = fidelity
    expected.pop("packet_id", None)
    expected.pop("packet_sha256", None)
    expected["packet_id"] = "kpacket:" + stable_hash(expected).split(":", 1)[1][:24]
    expected["packet_sha256"] = stable_hash(expected)
    if revised != expected:
        raise KernelProtocolFault(
            "KERC_FIDELITY_REVISION_SCOPE_VIOLATION",
            stable_hash(revised),
            path="packet",
        )
    return revised


def migrate_kernel_packet_kpp_1_1(
    packet: dict[str, Any], *, local_hrl_state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate a hash-valid KPP-1.1 packet without changing legacy payload fields."""

    if packet.get("policy") != POLICY or packet.get("packet_version") != LEGACY_PACKET_VERSION:
        raise KernelProtocolFault(
            "KERC_PACKET_MIGRATION_SOURCE_UNSUPPORTED",
            str(packet.get("packet_version")),
            path="packet.packet_version",
        )
    expected_sha256 = stable_hash(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    if packet.get("packet_sha256") != expected_sha256:
        raise KernelProtocolFault(
            "KERC_PACKET_MIGRATION_SOURCE_IDENTITY_INVALID",
            str(packet.get("packet_sha256")),
            path="packet.packet_sha256",
        )
    legacy_payload = {
        key: copy.deepcopy(value)
        for key, value in packet.items()
        if key not in {"packet_id", "packet_sha256", "packet_version"}
    }
    migrated = copy.deepcopy(packet)
    migrated["packet_version"] = PACKET_VERSION
    residual = migrated.get("residual")
    source = migrated.get("source")
    if not isinstance(residual, dict) or not isinstance(source, dict):
        raise KernelProtocolFault(
            "KERC_PACKET_MIGRATION_SOURCE_SCHEMA_INVALID",
            str(type(residual)),
            path="packet",
        )
    residual["unit_packet"] = build_residual_unit_packet(
        source_record_sha256=str(source.get("record_sha256") or ""),
        residual_mode=str(residual.get("mode") or ""),
        kernel_program=migrated.get("program") or {},
        global_state=local_hrl_state.get("global") or {},
        segment_residual=residual.get("segment_frame") or {},
        token_residuals=residual.get("token_tags") or [],
        concept_capsules=migrated.get("concept_capsules") or {},
        exact_objects=migrated.get("protected_objects") or {},
    )
    migrated.pop("packet_id", None)
    migrated.pop("packet_sha256", None)
    migrated["packet_id"] = "kpacket:" + stable_hash(migrated).split(":", 1)[1][:24]
    migrated["packet_sha256"] = stable_hash(migrated)
    validate_kernel_packet(migrated, local_hrl_state=local_hrl_state)
    migrated_payload = {
        key: copy.deepcopy(value)
        for key, value in migrated.items()
        if key not in {"packet_id", "packet_sha256", "packet_version"}
    }
    migrated_payload["residual"] = {
        key: value
        for key, value in migrated_payload["residual"].items()
        if key != "unit_packet"
    }
    if migrated_payload != legacy_payload:
        raise KernelProtocolFault(
            "KERC_PACKET_MIGRATION_LEGACY_PAYLOAD_CHANGED",
            stable_hash(migrated_payload),
            path="packet",
        )
    receipt = {
        "policy": "project_theseus_kerc_packet_migration_kpp_1_1_to_1_2_v1",
        "source_packet_sha256": expected_sha256,
        "target_packet_sha256": migrated["packet_sha256"],
        "legacy_payload_sha256": stable_hash(legacy_payload),
        "legacy_payload_preserved": True,
        "unit_count": len(residual["unit_packet"]["units"]),
        "fallback_return_count": 0,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return migrated, receipt


def validate_kernel_packet(
    packet: dict[str, Any], *, local_hrl_state: dict[str, Any]
) -> dict[str, Any]:
    if packet.get("policy") != POLICY or packet.get("packet_version") != PACKET_VERSION:
        raise KernelProtocolFault(
            "KERC_PACKET_PROTOCOL_INCOMPATIBLE",
            str(packet.get("packet_version")),
            path="packet",
        )
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
    residual = (
        packet.get("residual") if isinstance(packet.get("residual"), dict) else {}
    )
    if residual.get("global_state_hash") != local_hrl_state.get("state_hash"):
        raise KernelProtocolFault(
            "KERC_HRL_STATE_DESYNCHRONIZED",
            f"packet={residual.get('global_state_hash')} local={local_hrl_state.get('state_hash')}",
            path="packet.residual.global_state_hash",
        )
    if (
        residual.get("mode") not in RESIDUAL_MODES
        or residual.get("fidelity") not in FIDELITY_MODES
    ):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_CONTRACT_INVALID",
            canonical_json(residual),
            path="packet.residual",
        )
    if residual.get("hrl_version") != local_hrl_state.get(
        "hrl_version"
    ) or residual.get("interaction_id") != local_hrl_state.get("interaction_id"):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_STATE_IDENTITY_MISMATCH",
            canonical_json(residual),
            path="packet.residual",
        )
    source = packet.get("source") if isinstance(packet.get("source"), dict) else {}
    objects = (
        packet.get("protected_objects")
        if isinstance(packet.get("protected_objects"), dict)
        else {}
    )
    if residual.get("exact_object_handles") != sorted(objects):
        raise KernelProtocolFault(
            "KERC_RESIDUAL_EXACT_OBJECT_SET_MISMATCH",
            canonical_json(residual.get("exact_object_handles")),
            path="packet.residual.exact_object_handles",
        )
    segment = residual.get("segment_frame")
    if not isinstance(segment, dict):
        raise KernelProtocolFault(
            "KERC_SEGMENT_FRAME_SCHEMA_INVALID",
            canonical_json(segment),
            path="packet.residual.segment_frame",
        )
    normalized_segment = _normalize_segment_frame(
        segment,
        source_character_length=int(source.get("character_length") or 0),
        path="packet.residual.segment_frame",
    )
    if normalized_segment != segment:
        raise KernelProtocolFault(
            "KERC_SEGMENT_FRAME_NOT_CANONICAL",
            canonical_json(segment),
            path="packet.residual.segment_frame",
        )
    tags = residual.get("token_tags")
    if not isinstance(tags, list):
        raise KernelProtocolFault(
            "KERC_TOKEN_TAG_SCHEMA_INVALID",
            canonical_json(tags),
            path="packet.residual.token_tags",
        )
    for index, tag in enumerate(tags):
        if not isinstance(tag, dict) or set(tag) != {"tag", "source_span", "authority"}:
            raise KernelProtocolFault(
                "KERC_TOKEN_TAG_SCHEMA_INVALID",
                canonical_json(tag),
                path=f"packet.residual.token_tags[{index}]",
            )
        if (
            tag.get("authority") != "licensed_manual_annotation"
            or not str(tag.get("tag") or "").strip()
        ):
            raise KernelProtocolFault(
                "KERC_TOKEN_TAG_AUTHORITY_INVALID",
                canonical_json(tag),
                path=f"packet.residual.token_tags[{index}]",
            )
        _validate_residual_spans(
            [tag.get("source_span")],
            source_character_length=int(source.get("character_length") or 0),
            path=f"packet.residual.token_tags[{index}].source_span",
        )
    try:
        codec_receipt = validate_residual_codec(
            residual.get("codec") if isinstance(residual.get("codec"), dict) else {},
            kernel_program=packet.get("program")
            if isinstance(packet.get("program"), dict)
            else {},
            global_state=local_hrl_state.get("global")
            if isinstance(local_hrl_state.get("global"), dict)
            else {},
            segment_residual=segment,
            token_residuals=tags,
            exact_objects=objects,
        )
    except ResidualEconomicsFault as exc:
        raise KernelProtocolFault(
            exc.code, exc.detail, path="packet.residual.codec"
        ) from exc
    concepts = (
        packet.get("concept_capsules")
        if isinstance(packet.get("concept_capsules"), dict)
        else {}
    )
    try:
        unit_packet_receipt = validate_residual_unit_packet(
            residual.get("unit_packet")
            if isinstance(residual.get("unit_packet"), dict)
            else {},
            source_record_sha256=str(source.get("record_sha256") or ""),
            residual_mode=str(residual.get("mode") or ""),
            kernel_program=packet.get("program") or {},
            global_state=local_hrl_state.get("global") or {},
            segment_residual=segment,
            token_residuals=tags,
            concept_capsules=concepts,
            exact_objects=objects,
        )
    except ResidualEconomicsFault as exc:
        raise KernelProtocolFault(
            exc.code, exc.detail, path="packet.residual.unit_packet"
        ) from exc
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
    serialization = (
        packet.get("serialization")
        if isinstance(packet.get("serialization"), dict)
        else {}
    )
    decoded = deserialize_kernel_program(
        serialization,
        protected_objects=objects,
        concept_capsules=concepts,
        source_character_length=int(source.get("character_length") or 0),
    )
    if decoded["canonical_program"] != validated["canonical_program"]:
        raise KernelProtocolFault(
            "KERC_PACKET_SERIALIZATION_PROGRAM_MISMATCH",
            "decoded token stream differs from authoritative program",
            path="packet.serialization",
        )
    missing_objects = sorted(
        set(validated["referenced_handles"]) - set(objects) - set(concepts)
    )
    if missing_objects:
        raise KernelProtocolFault(
            "KERC_PACKET_HANDLE_MISSING",
            ",".join(missing_objects),
            path="packet.program",
        )
    return {
        "state": "READY",
        "packet_id": packet.get("packet_id"),
        "program_sha256": validated["canonical_program"]["program_sha256"],
        "state_hash_match": True,
        "serialization_replay_match": True,
        "serialization_exact_program_roundtrip": True,
        "residual_codec": codec_receipt,
        "residual_unit_packet": {
            "policy": unit_packet_receipt["policy"],
            "schema_version": unit_packet_receipt["schema_version"],
            "unit_count": unit_packet_receipt["summary"]["unit_count"],
            "packet_sha256": unit_packet_receipt["packet_sha256"],
        },
        "semantic_equivalence_claimed": False,
        **NO_CHEAT,
    }


def _default_answer_decision(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Migrate pre-decision source packets without granting calibrated authority."""

    return {
        "policy": ANSWER_DECISION_POLICY,
        "disposition": "ANSWER",
        "evidence_status": "UNVERIFIED",
        "uncertainty_state": "RESOLVED",
        "confidence": min(float(claim.get("confidence", 0.0)) for claim in claims),
        "controlling_claim_ids": sorted(str(claim["claim_id"]) for claim in claims),
        "unresolved_ambiguity_ids": [],
    }


def _validate_answer_decision(
    decision: Any,
    *,
    claims: list[dict[str, Any]],
    require_explicit: bool,
) -> dict[str, Any]:
    if decision is None and not require_explicit:
        return _default_answer_decision(claims)
    if (
        not isinstance(decision, dict)
        or decision.get("policy") != ANSWER_DECISION_POLICY
    ):
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_POLICY_INVALID",
            str(
                decision.get("policy") if isinstance(decision, dict) else type(decision)
            ),
            path="answer.decision.policy",
        )
    required = {
        "policy",
        "disposition",
        "evidence_status",
        "uncertainty_state",
        "confidence",
        "controlling_claim_ids",
        "unresolved_ambiguity_ids",
    }
    if set(decision) != required:
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_SCHEMA_INVALID",
            canonical_json(sorted(decision)),
            path="answer.decision",
        )
    disposition = str(decision.get("disposition") or "")
    evidence_status = str(decision.get("evidence_status") or "")
    uncertainty_state = str(decision.get("uncertainty_state") or "")
    if disposition not in ANSWER_DISPOSITIONS:
        raise KernelProtocolFault(
            "KERC_ANSWER_DISPOSITION_INVALID",
            disposition,
            path="answer.decision.disposition",
        )
    if evidence_status not in ANSWER_EVIDENCE_STATES:
        raise KernelProtocolFault(
            "KERC_ANSWER_EVIDENCE_STATUS_INVALID",
            evidence_status,
            path="answer.decision.evidence_status",
        )
    if uncertainty_state not in ANSWER_UNCERTAINTY_STATES:
        raise KernelProtocolFault(
            "KERC_ANSWER_UNCERTAINTY_STATE_INVALID",
            uncertainty_state,
            path="answer.decision.uncertainty_state",
        )
    confidence = decision.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_CONFIDENCE_INVALID",
            str(confidence),
            path="answer.decision.confidence",
        )
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_CONFIDENCE_INVALID",
            str(confidence),
            path="answer.decision.confidence",
        )
    claim_ids = {str(claim["claim_id"]) for claim in claims}
    controlling = decision.get("controlling_claim_ids")
    if (
        not isinstance(controlling, list)
        or not controlling
        or len(controlling) != len(set(controlling))
        or not set(controlling) <= claim_ids
    ):
        raise KernelProtocolFault(
            "KERC_ANSWER_DECISION_CLAIM_REFERENCE_INVALID",
            canonical_json(controlling),
            path="answer.decision.controlling_claim_ids",
        )
    ambiguity_ids = decision.get("unresolved_ambiguity_ids")
    if (
        not isinstance(ambiguity_ids, list)
        or len(ambiguity_ids) != len(set(ambiguity_ids))
        or any(
            not re.fullmatch(r"amb-[a-z0-9][a-z0-9_.:-]*", str(value))
            for value in ambiguity_ids
        )
    ):
        raise KernelProtocolFault(
            "KERC_ANSWER_AMBIGUITY_REFERENCE_INVALID",
            canonical_json(ambiguity_ids),
            path="answer.decision.unresolved_ambiguity_ids",
        )
    predicates = {
        str(claim["predicate"]) for claim in claims if claim["claim_id"] in controlling
    }
    if disposition == "CLARIFY" and "REQUEST_CLARIFICATION" not in predicates:
        raise KernelProtocolFault(
            "KERC_CLARIFICATION_CLAIM_MISSING",
            canonical_json(sorted(predicates)),
            path="answer.decision.disposition",
        )
    if disposition == "ABSTAIN" and "ABSTAIN" not in predicates:
        raise KernelProtocolFault(
            "KERC_ABSTENTION_CLAIM_MISSING",
            canonical_json(sorted(predicates)),
            path="answer.decision.disposition",
        )
    if disposition in {"CLARIFY", "ABSTAIN"} and uncertainty_state == "RESOLVED":
        raise KernelProtocolFault(
            "KERC_UNCERTAIN_DISPOSITION_MARKED_RESOLVED",
            disposition,
            path="answer.decision.uncertainty_state",
        )
    if uncertainty_state == "AMBIGUOUS" and not ambiguity_ids:
        raise KernelProtocolFault(
            "KERC_AMBIGUITY_ID_MISSING",
            disposition,
            path="answer.decision.unresolved_ambiguity_ids",
        )
    if uncertainty_state == "RESOLVED" and ambiguity_ids:
        raise KernelProtocolFault(
            "KERC_RESOLVED_DECISION_HAS_AMBIGUITY",
            canonical_json(ambiguity_ids),
            path="answer.decision.unresolved_ambiguity_ids",
        )
    canonical = copy.deepcopy(decision)
    canonical["confidence"] = confidence
    canonical["controlling_claim_ids"] = sorted(controlling)
    canonical["unresolved_ambiguity_ids"] = sorted(
        str(value) for value in ambiguity_ids
    )
    return canonical


def validate_answer_packet(
    packet: dict[str, Any], *, require_explicit_decision: bool = False
) -> dict[str, Any]:
    claims = packet.get("claims")
    if not isinstance(claims, list) or not claims:
        raise KernelProtocolFault(
            "KERC_ANSWER_CLAIMS_MISSING",
            "claims must be non-empty",
            path="answer.claims",
        )
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        path = f"answer.claims[{index}]"
        if not isinstance(claim, dict):
            raise KernelProtocolFault(
                "KERC_ANSWER_CLAIM_INVALID", str(claim), path=path
            )
        claim_id = str(claim.get("claim_id") or "")
        if not claim_id or claim_id in claim_ids:
            raise KernelProtocolFault(
                "KERC_ANSWER_CLAIM_ID_INVALID", claim_id, path=f"{path}.claim_id"
            )
        claim_ids.add(claim_id)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(claim.get("predicate") or "")):
            raise KernelProtocolFault(
                "KERC_ANSWER_PREDICATE_INVALID",
                str(claim.get("predicate")),
                path=f"{path}.predicate",
            )
        if str(claim.get("modality") or "") not in MODALITIES:
            raise KernelProtocolFault(
                "KERC_ANSWER_MODALITY_INVALID",
                str(claim.get("modality")),
                path=f"{path}.modality",
            )
        if str(claim.get("polarity") or "") not in POLARITIES:
            raise KernelProtocolFault(
                "KERC_ANSWER_POLARITY_INVALID",
                str(claim.get("polarity")),
                path=f"{path}.polarity",
            )
        confidence = float(claim.get("confidence", -1.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise KernelProtocolFault(
                "KERC_ANSWER_CONFIDENCE_INVALID",
                str(confidence),
                path=f"{path}.confidence",
            )
        if not isinstance(claim.get("arguments"), list):
            raise KernelProtocolFault(
                "KERC_ANSWER_ARGUMENTS_INVALID",
                "arguments must be a list",
                path=f"{path}.arguments",
            )
    canonical = copy.deepcopy(packet)
    canonical["decision"] = _validate_answer_decision(
        packet.get("decision"),
        claims=claims,
        require_explicit=require_explicit_decision,
    )
    canonical["answer_packet_sha256"] = stable_hash(
        {
            key: value
            for key, value in canonical.items()
            if key != "answer_packet_sha256"
        }
    )
    return canonical


def validate_answer_packet_against_context(
    packet: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
    concept_capsules: dict[str, dict[str, Any]],
    correction_lattice: dict[str, Any] | None = None,
    require_explicit_decision: bool = False,
) -> dict[str, Any]:
    """Validate answer value types and references against its packet context."""

    canonical = validate_answer_packet(
        packet, require_explicit_decision=require_explicit_decision
    )
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
    unresolved_corrections = len((correction_lattice or {}).get("corrections") or [])
    decision = canonical["decision"]
    if (
        unresolved_corrections
        and decision["disposition"] == "ANSWER"
        and decision["uncertainty_state"] == "RESOLVED"
    ):
        raise KernelProtocolFault(
            "KERC_UNRESOLVED_CORRECTION_CERTAINTY_LAUNDERING",
            str(unresolved_corrections),
            path="answer.decision",
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
        "claim_predicates",
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
        "answer_decision",
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
        raise KernelProtocolFault(
            "KERC_VALUE_INVALID", canonical_json(value), path=path
        )
    value_type = str(value["type"])
    payload = value.get("value")
    if value_type == "handle":
        handle = str(payload or "")
        if handle not in protected_objects and handle not in concept_capsules:
            raise KernelProtocolFault(
                "KERC_HANDLE_REFERENCE_UNKNOWN", handle, path=path
            )
        handles_seen.add(handle)
    elif value_type == "concept":
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", str(payload or "")):
            raise KernelProtocolFault(
                "KERC_CONCEPT_ID_INVALID", str(payload), path=path
            )
    elif value_type == "number":
        if (
            not isinstance(payload, dict)
            or isinstance(payload.get("value"), bool)
            or not isinstance(payload.get("value"), (int, float))
        ):
            raise KernelProtocolFault(
                "KERC_NUMBER_INVALID", canonical_json(payload), path=path
            )
        if not math.isfinite(float(payload["value"])):
            raise KernelProtocolFault(
                "KERC_NUMBER_INVALID", canonical_json(payload), path=path
            )
    elif value_type == "quantity":
        _validate_quantity(payload, path=path)
    elif value_type == "temporal":
        _validate_temporal(payload, path=path)
    elif value_type == "text":
        if (
            not isinstance(payload, dict)
            or set(payload) != {"text", "language"}
            or not isinstance(payload.get("text"), str)
            or not payload["text"]
            or payload["text"] != unicodedata.normalize("NFC", payload["text"])
            or payload.get("language") not in {None, "en"}
        ):
            raise KernelProtocolFault(
                "KERC_TEXT_VALUE_INVALID", canonical_json(payload), path=path
            )
    elif value_type == "symbol":
        if not isinstance(payload, str) or not re.fullmatch(
            r"[A-Za-z?][A-Za-z0-9_.:+?/-]*", payload
        ):
            raise KernelProtocolFault(
                "KERC_SYMBOL_VALUE_INVALID", canonical_json(payload), path=path
            )
    elif value_type == "node_ref":
        ref = str(payload or "")
        if not re.fullmatch(r"k[0-9]+", ref):
            raise KernelProtocolFault("KERC_NODE_REFERENCE_INVALID", ref, path=path)
        node_refs.add(ref)
    elif value_type == "list":
        if not isinstance(payload, list):
            raise KernelProtocolFault(
                "KERC_LIST_INVALID", canonical_json(payload), path=path
            )
        for index, item in enumerate(payload):
            _validate_value(
                item,
                path=f"{path}[{index}]",
                protected_objects=protected_objects,
                concept_capsules=concept_capsules,
                node_refs=node_refs,
                handles_seen=handles_seen,
            )
    elif value_type == "ambiguity":
        if not isinstance(payload, list) or len(payload) < 2:
            raise KernelProtocolFault(
                "KERC_AMBIGUITY_INVALID", canonical_json(payload), path=path
            )
        mass = 0.0
        for index, alternative in enumerate(payload):
            if (
                not isinstance(alternative, dict)
                or set(alternative) != {"probability", "value", "evidence"}
                or not isinstance(alternative.get("evidence"), str)
                or not alternative["evidence"].strip()
            ):
                raise KernelProtocolFault(
                    "KERC_AMBIGUITY_INVALID",
                    canonical_json(alternative),
                    path=f"{path}[{index}]",
                )
            probability = float(alternative.get("probability") or 0.0)
            mass += probability
            _validate_value(
                alternative.get("value"),
                path=f"{path}[{index}].value",
                protected_objects=protected_objects,
                concept_capsules=concept_capsules,
                node_refs=node_refs,
                handles_seen=handles_seen,
            )
        if not math.isclose(mass, 1.0, abs_tol=1e-6):
            raise KernelProtocolFault(
                "KERC_AMBIGUITY_PROBABILITY_MASS_INVALID", str(mass), path=path
            )
    elif value_type == "byte_literal":
        try:
            base64.b64decode(str(payload or ""), validate=True)
        except Exception as exc:
            raise KernelProtocolFault(
                "KERC_BYTE_LITERAL_INVALID", str(payload), path=path
            ) from exc
    elif value_type == "boolean" and not isinstance(payload, bool):
        raise KernelProtocolFault(
            "KERC_BOOLEAN_INVALID", canonical_json(payload), path=path
        )
    elif value_type == "null" and payload is not None:
        raise KernelProtocolFault(
            "KERC_NULL_INVALID", canonical_json(payload), path=path
        )


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
    if value_type == "quantity":
        return [token("V_P", f"QUANTITY:{canonical_json(payload)}")]
    if value_type == "temporal":
        return [token("V_P", f"TEMPORAL:{canonical_json(payload)}")]
    if value_type == "text":
        return [token("V_P", f"TEXT:{canonical_json(payload)}")]
    if value_type == "symbol":
        return [token("V_K", f"SYMBOL:{payload}")]
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
            rows.append(token("V_P", f"PROB:{alternative['probability']:.17g}"))
            rows.append(
                token("V_P", f"EVIDENCE:{canonical_json(alternative['evidence'])}")
            )
            rows.extend(_serialize_value(alternative["value"]))
        rows.append(token("V_P", "AMBIG_END"))
        return rows
    if value_type == "byte_literal":
        return [token("V_P", f"BYTE:{payload}")]
    if value_type == "boolean":
        return [token("V_K", "BOOL:TRUE" if payload else "BOOL:FALSE")]
    return [token("V_K", "NULL")]


def _validate_decimal_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value
    ):
        raise KernelProtocolFault(
            "KERC_DECIMAL_VALUE_INVALID", canonical_json(value), path=path
        )
    if value.startswith("-0") and value not in {"-0", "-0.0"}:
        raise KernelProtocolFault("KERC_DECIMAL_VALUE_NONCANONICAL", value, path=path)
    if "." in value and (value.endswith("0") or value.endswith(".")):
        raise KernelProtocolFault("KERC_DECIMAL_VALUE_NONCANONICAL", value, path=path)
    return value


def _validate_quantity(payload: Any, *, path: str) -> None:
    required = {
        "kind",
        "relation",
        "value",
        "lower",
        "upper",
        "unit_concept",
        "approximate",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise KernelProtocolFault(
            "KERC_QUANTITY_SCHEMA_INVALID", canonical_json(payload), path=path
        )
    if (
        payload["kind"] not in QUANTITY_KINDS
        or payload["relation"] not in QUANTITY_RELATIONS
    ):
        raise KernelProtocolFault(
            "KERC_QUANTITY_KIND_INVALID", canonical_json(payload), path=path
        )
    if not isinstance(payload["approximate"], bool):
        raise KernelProtocolFault(
            "KERC_QUANTITY_APPROXIMATION_INVALID", canonical_json(payload), path=path
        )
    unit = payload["unit_concept"]
    if unit is not None and not re.fullmatch(r"[a-z][a-z0-9_.:-]*", str(unit)):
        raise KernelProtocolFault(
            "KERC_QUANTITY_UNIT_INVALID", canonical_json(unit), path=path
        )
    for field in ("value", "lower", "upper"):
        if payload[field] is not None:
            _validate_decimal_text(payload[field], path=f"{path}.{field}")
    relation = payload["relation"]
    if relation == "BETWEEN":
        if (
            payload["lower"] is None
            or payload["upper"] is None
            or payload["value"] is not None
        ):
            raise KernelProtocolFault(
                "KERC_QUANTITY_BOUNDS_INVALID", canonical_json(payload), path=path
            )
        if Decimal(payload["lower"]) > Decimal(payload["upper"]):
            raise KernelProtocolFault(
                "KERC_QUANTITY_BOUNDS_REVERSED", canonical_json(payload), path=path
            )
    elif relation == "UNKNOWN":
        if any(payload[field] is not None for field in ("value", "lower", "upper")):
            raise KernelProtocolFault(
                "KERC_QUANTITY_UNKNOWN_HAS_VALUE", canonical_json(payload), path=path
            )
    elif (
        payload["value"] is None
        or payload["lower"] is not None
        or payload["upper"] is not None
    ):
        raise KernelProtocolFault(
            "KERC_QUANTITY_VALUE_INVALID", canonical_json(payload), path=path
        )
    if payload["approximate"] != (relation == "APPROX"):
        raise KernelProtocolFault(
            "KERC_QUANTITY_APPROXIMATION_MISMATCH", canonical_json(payload), path=path
        )


def _validate_temporal(payload: Any, *, path: str) -> None:
    required = {"kind", "relation", "value", "anchor", "calendar"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise KernelProtocolFault(
            "KERC_TEMPORAL_SCHEMA_INVALID", canonical_json(payload), path=path
        )
    if (
        payload["kind"] not in TEMPORAL_KINDS
        or payload["relation"] not in TEMPORAL_RELATIONS
    ):
        raise KernelProtocolFault(
            "KERC_TEMPORAL_KIND_INVALID", canonical_json(payload), path=path
        )
    value = payload["value"]
    if (
        not isinstance(value, str)
        or not value
        or value != unicodedata.normalize("NFC", value)
    ):
        raise KernelProtocolFault(
            "KERC_TEMPORAL_VALUE_INVALID", canonical_json(value), path=path
        )
    anchor = payload["anchor"]
    if anchor is not None and not (
        re.fullmatch(r"@[A-Z][0-9]+", str(anchor))
        or re.fullmatch(r"[a-z][a-z0-9_.:-]*", str(anchor))
    ):
        raise KernelProtocolFault(
            "KERC_TEMPORAL_ANCHOR_INVALID", canonical_json(anchor), path=path
        )
    if payload["calendar"] not in {None, "ISO8601", "GREGORIAN"}:
        raise KernelProtocolFault(
            "KERC_TEMPORAL_CALENDAR_INVALID",
            canonical_json(payload["calendar"]),
            path=path,
        )
    kind = payload["kind"]
    if kind == "DATE":
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_DATE_INVALID", value, path=path
            ) from exc
        if payload["calendar"] is None:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_CALENDAR_REQUIRED", value, path=path
            )
    elif kind == "TIME":
        if not re.fullmatch(
            r"(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\.[0-9]+)?)?(?:Z|[+-][0-2][0-9]:[0-5][0-9])?",
            value,
        ):
            raise KernelProtocolFault("KERC_TEMPORAL_TIME_INVALID", value, path=path)
    elif kind == "DATETIME":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_DATETIME_INVALID", value, path=path
            ) from exc
        if "T" not in value:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_DATETIME_INVALID", value, path=path
            )
    elif kind == "DURATION":
        if not re.fullmatch(
            r"P(?=\d|T\d)(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?",
            value,
        ):
            raise KernelProtocolFault(
                "KERC_TEMPORAL_DURATION_INVALID", value, path=path
            )
        if payload["calendar"] is not None:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_DURATION_CALENDAR_INVALID",
                canonical_json(payload["calendar"]),
                path=path,
            )
    elif kind == "INTERVAL":
        if value.count("/") != 1 or any(not endpoint for endpoint in value.split("/")):
            raise KernelProtocolFault(
                "KERC_TEMPORAL_INTERVAL_INVALID", value, path=path
            )
    elif kind == "RELATIVE":
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", value) or anchor is None:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_RELATIVE_INVALID", canonical_json(payload), path=path
            )
        if payload["calendar"] is not None:
            raise KernelProtocolFault(
                "KERC_TEMPORAL_RELATIVE_CALENDAR_INVALID",
                canonical_json(payload["calendar"]),
                path=path,
            )


class _KernelTokenCursor:
    def __init__(self, rows: Sequence[dict[str, str]]) -> None:
        self.rows = [
            _validate_token(row, path=f"serialization.expanded_tokens[{index}]")
            for index, row in enumerate(rows)
        ]
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index == len(self.rows)

    def peek(self) -> dict[str, str]:
        if self.done:
            raise KernelProtocolFault(
                "KERC_SERIALIZATION_UNEXPECTED_END",
                "token stream ended",
                path="serialization.expanded_tokens",
            )
        return self.rows[self.index]

    def take(self) -> dict[str, str]:
        row = self.peek()
        self.index += 1
        return row

    def expect(self, space: str, value: str) -> None:
        row = self.take()
        if row != {"space": space, "token": value}:
            raise KernelProtocolFault(
                "KERC_SERIALIZATION_TOKEN_UNEXPECTED",
                canonical_json(row),
                path=f"serialization.expanded_tokens[{self.index - 1}]",
            )

    def expect_prefix(self, space: str, prefix: str) -> str:
        row = self.take()
        if row["space"] != space or not row["token"].startswith(prefix):
            raise KernelProtocolFault(
                "KERC_SERIALIZATION_TOKEN_UNEXPECTED",
                canonical_json(row),
                path=f"serialization.expanded_tokens[{self.index - 1}]",
            )
        return row["token"][len(prefix) :]

    def remaining(self) -> list[dict[str, str]]:
        return self.rows[self.index :]


def _deserialize_json_value(
    cursor: _KernelTokenCursor, prefix: str, value_type: str
) -> dict[str, Any]:
    payload = cursor.expect_prefix("V_P", prefix)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise KernelProtocolFault(
            "KERC_SERIALIZATION_VALUE_JSON_INVALID",
            payload,
            path="serialization.expanded_tokens",
        ) from exc
    return {"type": value_type, "value": decoded}


def _deserialize_value(cursor: _KernelTokenCursor) -> dict[str, Any]:
    row = cursor.peek()
    space, value = row["space"], row["token"]
    if space == "V_P" and value.startswith("HANDLE:"):
        cursor.take()
        return {"type": "handle", "value": value.removeprefix("HANDLE:")}
    if space == "V_K" and value.startswith("CONCEPT:"):
        cursor.take()
        return {"type": "concept", "value": value.removeprefix("CONCEPT:")}
    for prefix, value_type in (
        ("NUMBER:", "number"),
        ("QUANTITY:", "quantity"),
        ("TEMPORAL:", "temporal"),
        ("TEXT:", "text"),
    ):
        if space == "V_P" and value.startswith(prefix):
            return _deserialize_json_value(cursor, prefix, value_type)
    if space == "V_K" and value.startswith("SYMBOL:"):
        cursor.take()
        return {"type": "symbol", "value": value.removeprefix("SYMBOL:")}
    if space == "V_P" and value.startswith("NODE_REF:"):
        cursor.take()
        return {"type": "node_ref", "value": value.removeprefix("NODE_REF:")}
    if row == {"space": "V_P", "token": "LIST_BEGIN"}:
        cursor.take()
        values = []
        while cursor.peek() != {"space": "V_P", "token": "LIST_END"}:
            values.append(_deserialize_value(cursor))
        cursor.take()
        return {"type": "list", "value": values}
    if row == {"space": "V_P", "token": "AMBIG_BEGIN"}:
        cursor.take()
        values = []
        while cursor.peek() != {"space": "V_P", "token": "AMBIG_END"}:
            probability_text = cursor.expect_prefix("V_P", "PROB:")
            evidence_text = cursor.expect_prefix("V_P", "EVIDENCE:")
            try:
                probability = float(probability_text)
                evidence = json.loads(evidence_text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise KernelProtocolFault(
                    "KERC_SERIALIZATION_PROBABILITY_INVALID",
                    probability_text,
                    path="serialization.expanded_tokens",
                ) from exc
            values.append(
                {
                    "probability": probability,
                    "evidence": evidence,
                    "value": _deserialize_value(cursor),
                }
            )
        cursor.take()
        return {"type": "ambiguity", "value": values}
    if space == "V_P" and value.startswith("BYTE:"):
        cursor.take()
        return {"type": "byte_literal", "value": value.removeprefix("BYTE:")}
    if row == {"space": "V_K", "token": "BOOL:TRUE"}:
        cursor.take()
        return {"type": "boolean", "value": True}
    if row == {"space": "V_K", "token": "BOOL:FALSE"}:
        cursor.take()
        return {"type": "boolean", "value": False}
    if row == {"space": "V_K", "token": "NULL"}:
        cursor.take()
        return {"type": "null", "value": None}
    raise KernelProtocolFault(
        "KERC_SERIALIZATION_VALUE_TOKEN_INVALID",
        canonical_json(row),
        path="serialization.expanded_tokens",
    )


def _answer_constraints(
    packet: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    handles: set[str] = set()
    numbers: set[str] = set()
    polarities = []
    modalities = []
    quantifiers = []
    predicates = []
    temporal = []
    causal = []
    attributions = []
    for claim in packet["claims"]:
        predicates.append((claim["claim_id"], claim["predicate"]))
        polarities.append((claim["claim_id"], claim["polarity"]))
        modalities.append((claim["claim_id"], claim["modality"]))
        quantifiers.append((claim["claim_id"], claim.get("quantifier", "NONE")))
        temporal.append(
            (claim["claim_id"], canonical_json(claim.get("temporal") or {}))
        )
        attributions.append(
            (claim["claim_id"], canonical_json(claim.get("attribution") or {}))
        )
        if claim["predicate"] in {"CAUSE", "PREVENT", "ENABLE", "RESULT_IN"}:
            causal.append(
                (
                    claim["claim_id"],
                    claim["predicate"],
                    canonical_json(claim["arguments"]),
                )
            )
        _collect_answer_values(claim["arguments"], handles, numbers)
    quote_handles = sorted(
        handle
        for handle in handles
        if objects.get(handle, {}).get("object_type") == "QUOTE"
    )
    entity_handles = sorted(
        handle
        for handle in handles
        if objects.get(handle, {}).get("object_type") != "QUOTE"
    )
    return {
        "claim_predicates": sorted(predicates),
        "entity_handles": entity_handles,
        "number_values": sorted(numbers),
        "claim_polarities": sorted(polarities),
        "claim_modalities": sorted(modalities),
        "quantifiers": sorted(quantifiers),
        "temporal_relations": sorted(temporal),
        "causal_relations": sorted(causal),
        "attributions": sorted(attributions),
        "quotation_handles": quote_handles,
        "required_terms": sorted(
            canonical_json(row) for row in packet.get("required_terms") or []
        ),
        "required_caveats": sorted(
            str(row) for row in packet.get("required_caveats") or []
        ),
        "answer_decision": copy.deepcopy(packet["decision"]),
    }


def _collect_answer_values(
    arguments: Any, handles: set[str], numbers: set[str]
) -> None:
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
        raise KernelProtocolFault(
            "KERC_HRL_VERSION_INCOMPATIBLE",
            str(state.get("hrl_version") if isinstance(state, dict) else None),
            path="hrl_state",
        )
    if not str(state.get("interaction_id") or "") or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(state.get("state_hash") or "")
    ):
        raise KernelProtocolFault(
            "KERC_HRL_REFERENCE_INVALID",
            "interaction_id/state_hash missing",
            path="hrl_state",
        )


def _validate_concept_capsules(capsules: dict[str, dict[str, Any]]) -> None:
    for handle, capsule in capsules.items():
        if not re.fullmatch(r"@C[0-9]+", str(handle)) or not isinstance(capsule, dict):
            raise KernelProtocolFault(
                "KERC_CONCEPT_CAPSULE_INVALID", str(handle), path="concept_capsules"
            )
        identity = str(capsule.get("stable_identity") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]*", identity):
            raise KernelProtocolFault(
                "KERC_CONCEPT_ID_INVALID", identity, path=f"concept_capsules.{handle}"
            )
        if not capsule.get("provenance"):
            raise KernelProtocolFault(
                "KERC_CONCEPT_PROVENANCE_MISSING",
                handle,
                path=f"concept_capsules.{handle}",
            )


def _reject_reference_cycles(refs: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise KernelProtocolFault(
                "KERC_NODE_REFERENCE_CYCLE", node_id, path="program.nodes"
            )
        if node_id in visited:
            return
        visiting.add(node_id)
        for target in refs.get(node_id, set()):
            visit(target)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in refs:
        visit(node_id)


def _explicit_span_candidates(
    source: str, rows: Sequence[dict[str, Any]]
) -> list[SpanCandidate]:
    output = []
    for index, row in enumerate(rows):
        path = f"explicit_spans[{index}]"
        if not isinstance(row, dict):
            raise KernelProtocolFault("KERC_EXPLICIT_SPAN_INVALID", str(row), path=path)
        start = _required_int(row, "start", path)
        end = _required_int(row, "end", path)
        object_type = str(row.get("object_type") or "EXACT_TEXT")
        copy_policy = str(row.get("copy_policy") or "EXACT")
        if (
            not 0 <= start < end <= len(source)
            or object_type not in HANDLE_PREFIX_BY_TYPE
            or copy_policy not in COPY_POLICIES
        ):
            raise KernelProtocolFault(
                "KERC_EXPLICIT_SPAN_INVALID", canonical_json(row), path=path
            )
        output.append(
            SpanCandidate(
                start,
                end,
                object_type,
                copy_policy,
                100,
                "explicit_user_or_caller_span",
            )
        )
    return output


def _select_non_overlapping(candidates: Iterable[SpanCandidate]) -> list[SpanCandidate]:
    selected: list[SpanCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (
            -row.priority,
            row.start,
            -(row.end - row.start),
            row.object_type,
        ),
    ):
        if any(
            candidate.start < existing.end and candidate.end > existing.start
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda row: (row.start, row.end))


def _required_int(row: dict[str, Any], key: str, path: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise KernelProtocolFault(
            "KERC_INTEGER_FIELD_INVALID", f"{key}={value}", path=f"{path}.{key}"
        )
    return value


def _normalization_form(text: str) -> str:
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        if unicodedata.is_normalized(form, text):
            return form
    return "NONE"


def _node_sort_key(node_id: str) -> tuple[int, str]:
    return (int(node_id[1:]), node_id)
