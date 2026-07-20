#!/usr/bin/env python3
"""Produce source-bound KERC candidates from licensed human task data.

This producer is intentionally not an admission authority. It parses pinned
source artifacts and emits candidate records; ``kerc_semantic_corpus_verify``
must independently replay the raw sources before canonical rows exist.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq

from kernel_english_protocol import (
    ANSWER_DECISION_POLICY,
    PACKET_VERSION,
    SEMANTIC_SUPERVISION_POLICY,
    TRAINING_OBJECTIVES,
    TRAINING_RECORD_POLICY,
    build_kernel_packet,
    extract_protected_objects,
    revise_kernel_packet_fidelity,
    stable_hash,
)
from kerc_importance_policy import fit_importance_policy, predict_importance
from kerc_masc_event_coreference import (
    ALIGNMENT_CONTRACT as MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT,
    COMPACTION_CONTRACT as MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT,
    POLICY as MASC_EVENT_COREFERENCE_POLICY,
    reconstruct_event_coreference_groups,
)
from kerc_masc_mpqa_relations import (
    COMPACTION_CONTRACT as MASC_MPQA_RELATION_COMPACTION_CONTRACT,
    POLICY as MASC_MPQA_RELATION_POLICY,
    RELATION_CONTRACT as MASC_MPQA_RELATION_CONTRACT,
    reconstruct_mpqa_relation_chains,
)
from kerc_gum_discourse_relations import (
    POLICY as GUM_DISCOURSE_POLICY,
    PROJECTION_CONTRACT as GUM_DISCOURSE_PROJECTION_CONTRACT,
    RELATION_CONTRACT as GUM_DISCOURSE_RELATION_CONTRACT,
    SPLIT_CONTRACT as GUM_DISCOURSE_SPLIT_CONTRACT,
    reconstruct_gum_discourse_relations,
)
from kerc_gum_entity_coreference import (
    COMPACTION_CONTRACT as GUM_ENTITY_COREFERENCE_COMPACTION_CONTRACT,
    POLICY as GUM_ENTITY_COREFERENCE_POLICY,
    RELATION_CONTRACT as GUM_ENTITY_COREFERENCE_RELATION_CONTRACT,
    SPLIT_CONTRACT as GUM_ENTITY_COREFERENCE_SPLIT_CONTRACT,
    reconstruct_gum_entity_coreference,
)
from kerc_content_cache import (
    ContentObjectCache,
    cache_storage_telemetry,
    dependency_bindings,
    load_receipt,
    object_key,
    publish_receipt,
)
from kerc_residual_economics import (
    UNIT_PACKET_POLICY,
    UNIT_PACKET_VERSION,
    build_structural_rate_distortion_allocation,
    calibrate_allocation_lambda,
    reallocate_structural_receipt,
    residual_unit_allocation_receipt,
    residual_wire_bytes,
)
from kerc_residual_interventions import (
    build_unit_intervention_targets,
    compact_allocator_targets,
)
from kerc_scoped_semantics import (
    POLICY as KERC_SCOPED_SEMANTIC_POLICY,
    compile_scoped_semantic_graph,
)
from kerc_source_family_identity import (
    PRODUCER_FAMILY_ROOTS,
    family_identity_receipts,
    source_closure_receipt,
)
from moecot_source_conditioned_pretraining import (
    KERC_SEMANTIC_CORPUS_POLICY,
    validate_kernel_english_config,
)
from vcm_semantic_memory import (
    apply_hierarchical_residual_delta,
    create_hierarchical_residual_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
GRAF = "{http://www.xces.org/ns/GrAF/1.0/}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
PRODUCER_POLICY = "project_theseus_kerc_semantic_corpus_producer_v1"
PACKET_CACHE_ABI_IDENTITY = stable_hash(
    {
        "packet_version": PACKET_VERSION,
        "unit_packet_policy": UNIT_PACKET_POLICY,
        "unit_packet_version": UNIT_PACKET_VERSION,
        "kernel_protocol_sha256": hashlib.sha256(
            (ROOT / "scripts" / "kernel_english_protocol.py").read_bytes()
        ).hexdigest(),
        "residual_economics_sha256": hashlib.sha256(
            (ROOT / "scripts" / "kerc_residual_economics.py").read_bytes()
        ).hexdigest(),
    }
)
MASC_ENTITY_TYPES = {
    "person": "PERSON",
    "location": "PLACE",
    "org": "ORGANIZATION",
    "date": "DATE_TIME",
}
DOLLY_GROUNDED_QUESTION_POLICY = (
    "project_theseus_dolly_unique_extractive_question_support_v1"
)
MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY = (
    "project_theseus_kerc_masc_train_only_contextual_frame_ambiguity_v1"
)
MASC_DECISION_SEMANTICS_POLICY = (
    "project_theseus_kerc_masc_decision_semantics_v1"
)
MASC_MPQA_SEMANTIC_LABELS = {
    "agent",
    "attitude",
    "direct-subjective",
    "expressive-subjectivity",
    "target",
}
GUM_SCOPED_SEMANTIC_SUPERVISION_POLICY = (
    "project_theseus_kerc_gum_source_grounded_scoped_semantics_v1"
)
GUM_SCOPED_RELATION_CONTRACT = {
    "contingency-condition_r": {
        "operator": "CONDITION",
        "roles": {"child": "ANTECEDENT", "parent": "CONSEQUENT"},
    },
    "causal-cause_r": {
        "operator": "CONSEQUENCE",
        "roles": {"child": "CAUSE", "parent": "RESULT"},
    },
    "causal-result_r": {
        "operator": "CONSEQUENCE",
        "roles": {"child": "RESULT", "parent": "CAUSE"},
    },
    "explanation-evidence_r": {
        "operator": "EXPLANATION",
        "roles": {"child": "EVIDENCE", "parent": "CLAIM"},
    },
    "adversative-contrast_m": {
        "operator": "CONTRAST",
        "ordered_roles": ["LEFT", "RIGHT"],
    },
    "joint-disjunction_m": {
        "operator": "ALTERNATION",
        "ordered_roles": ["MEMBER", "MEMBER"],
    },
    "joint-sequence_m": {
        "operator": "CONTINUATION",
        "ordered_roles": ["PREVIOUS", "NEXT"],
    },
}
GUM_SCOPED_RELATION_DESCRIPTIONS = {
    "adversative-contrast_m": "CONTRAST:source_order_LEFT_RIGHT",
    "causal-cause_r": "CONSEQUENCE:child_CAUSE_parent_RESULT",
    "causal-result_r": "CONSEQUENCE:parent_CAUSE_child_RESULT",
    "contingency-condition_r": "CONDITION:child_ANTECEDENT_parent_CONSEQUENT",
    "explanation-evidence_r": "EXPLANATION:parent_CLAIM_child_EVIDENCE",
    "joint-disjunction_m": "ALTERNATION:source_order_MEMBER_MEMBER",
    "joint-sequence_m": "CONTINUATION:source_order_PREVIOUS_NEXT",
}
DOLLY_QUESTION_FORM_RE = re.compile(
    r"^(who|what|when|where|which|how(?:\s+(?:many|much|long|old|far))?)\b",
    re.IGNORECASE,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def producer_cache_dependency_paths(
    config_path: Path, corpus: dict[str, Any]
) -> dict[str, Path]:
    scripts = ROOT / "scripts"
    paths = {
        "config": config_path.resolve(),
        "producer": Path(__file__).resolve(),
        "cache_integrity": scripts / "kerc_content_cache.py",
        "source_family_identity": scripts / "kerc_source_family_identity.py",
        "kernel_protocol": scripts / "kernel_english_protocol.py",
        "importance_policy": scripts / "kerc_importance_policy.py",
        "residual_economics": scripts / "kerc_residual_economics.py",
        "masc_event_coreference_producer": scripts / "kerc_masc_event_coreference.py",
        "masc_mpqa_relation_producer": scripts / "kerc_masc_mpqa_relations.py",
        "gum_entity_coreference_producer": scripts
        / "kerc_gum_entity_coreference.py",
        "semantic_config_validator": scripts / "moecot_source_conditioned_pretraining.py",
        "vcm_residual_lifecycle": scripts / "vcm_semantic_memory.py",
        "dolly_source": resolve(corpus["dolly"]["path"]),
        "masc_archive": resolve(corpus["masc"]["archive_path"]),
        "masc_extracted_tree": resolve(corpus["masc"]["extracted_root"]),
        "gum_source_tree": resolve(corpus["gum"]["source_root"]),
    }
    for split, row in sorted(corpus["oasst2"]["files"].items()):
        paths[f"oasst2_{split}_source"] = resolve(row["path"])
    return paths


def producer_family_identity_receipts() -> dict[str, dict[str, Any]]:
    scripts = ROOT / "scripts"
    common_external = {
        "kernel_protocol": scripts / "kernel_english_protocol.py",
        "vcm_residual_lifecycle": scripts / "vcm_semantic_memory.py",
    }
    family_external = {
        "masc_event_coreference": {
            "raw_relation_producer": scripts / "kerc_masc_event_coreference.py"
        },
        "masc_mpqa_relation": {
            "raw_relation_producer": scripts / "kerc_masc_mpqa_relations.py"
        },
        "gum_discourse": {
            "raw_relation_producer": scripts / "kerc_gum_discourse_relations.py",
            "scoped_semantic_compiler": scripts / "kerc_scoped_semantics.py",
        },
        "gum_entity_coreference": {
            "raw_relation_producer": scripts / "kerc_gum_entity_coreference.py"
        },
    }
    return family_identity_receipts(
        source_path=Path(__file__).resolve(),
        source_label="scripts/kerc_semantic_corpus.py",
        role="candidate_record_producer",
        family_roots=PRODUCER_FAMILY_ROOTS,
        external_paths=common_external,
        family_external_paths=family_external,
    )


def producer_finalization_identity_receipt() -> dict[str, Any]:
    return source_closure_receipt(
        source_path=Path(__file__).resolve(),
        source_label="scripts/kerc_semantic_corpus.py",
        role="candidate_record_finalization",
        family="all_source_families",
        root_function="finalize_candidate_record",
        external_paths={
            "kernel_protocol": ROOT / "scripts" / "kernel_english_protocol.py",
            "residual_economics": ROOT / "scripts" / "kerc_residual_economics.py",
        },
    )


def cached_candidate_record(
    *,
    store: ContentObjectCache | None,
    role: str,
    layer: str,
    family: str,
    family_identity: str,
    inputs: dict[str, Any],
    expected_source_id: str,
    refresh_cache: bool,
    build: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    key = object_key(
        role=role,
        layer=layer,
        dependencies={
            "family": family,
            "family_identity": family_identity,
            "packet_cache_abi_identity": PACKET_CACHE_ABI_IDENTITY,
            "inputs": inputs,
        },
    )
    cached = store.get(key) if store is not None and not refresh_cache else None
    if (
        isinstance(cached, dict)
        and cached.get("policy") == TRAINING_RECORD_POLICY
        and cached.get("provenance", {}).get("source_id") == expected_source_id
        and cached.get("semantic_supervision", {}).get("producer_artifact_sha256")
        == family_identity
        and "importance" not in (cached.get("residual_supervision") or {})
        and (cached.get("verification_receipt") or {}).get("accepted") is False
        and cached.get("external_inference") is False
        and cached.get("fallback_return_count") == 0
        and cached.get("template_credit") == 0
    ):
        return cached, True
    record = build()
    if (
        record.get("policy") != TRAINING_RECORD_POLICY
        or record.get("provenance", {}).get("source_id") != expected_source_id
        or record.get("semantic_supervision", {}).get("producer_artifact_sha256")
        != family_identity
        or "importance" in (record.get("residual_supervision") or {})
        or record.get("external_inference") is not False
        or record.get("fallback_return_count") != 0
        or record.get("template_credit") != 0
    ):
        raise ValueError(f"candidate-record cache contract mismatch: {family}")
    if store is not None:
        store.put(key, record)
    return record, False


def finalize_candidate_record(
    *,
    record: dict[str, Any],
    importance: dict[str, Any],
    provisional_allocation: dict[str, Any],
    lambda_value: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    allocation = reallocate_structural_receipt(
        provisional_allocation,
        lambda_value=lambda_value,
    )
    packet = revise_kernel_packet_fidelity(
        record["kernel_packet"],
        allocation["selected_fidelity"],
        local_hrl_state=record["hrl_state"],
    )
    record["kernel_packet"] = packet
    record["residual_supervision"] = residual_supervision(
        str(record["provenance"]["source_id"]),
        packet=packet,
        hrl_state=record["hrl_state"],
        source_family=str(record["provenance"]["source_group"]),
        importance=importance,
        allocation=allocation,
    )
    return record, allocation


def cached_finalized_candidate(
    *,
    store: ContentObjectCache | None,
    role: str,
    layer: str,
    finalization_identity: str,
    record: dict[str, Any],
    importance: dict[str, Any],
    provisional_allocation: dict[str, Any],
    lambda_value: float,
    refresh_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    candidate_sha256 = stable_hash(record)
    dependencies = {
        "finalization_identity": finalization_identity,
        "candidate_sha256": candidate_sha256,
        "source_id": record["provenance"]["source_id"],
        "importance": importance,
        "provisional_allocation": provisional_allocation,
        "lambda_value": lambda_value,
    }
    key = object_key(role=role, layer=layer, dependencies=dependencies)
    cached = store.get(key) if store is not None and not refresh_cache else None
    if isinstance(cached, dict):
        finalized = cached.get("record")
        allocation = cached.get("allocation")
        if (
            cached.get("candidate_sha256") == candidate_sha256
            and cached.get("finalization_identity") == finalization_identity
            and isinstance(finalized, dict)
            and isinstance(allocation, dict)
            and finalized.get("provenance", {}).get("source_id")
            == record["provenance"]["source_id"]
            and finalized.get("residual_supervision", {}).get("importance")
            == importance
            and finalized.get("residual_supervision", {}).get(
                "rate_distortion_allocation"
            )
            == allocation
            and finalized.get("kernel_packet", {}).get("residual", {}).get(
                "fidelity"
            )
            == allocation.get("selected_fidelity")
            and finalized.get("external_inference") is False
            and finalized.get("fallback_return_count") == 0
            and finalized.get("template_credit") == 0
        ):
            return finalized, allocation, True
    finalized, allocation = finalize_candidate_record(
        record=record,
        importance=importance,
        provisional_allocation=provisional_allocation,
        lambda_value=lambda_value,
    )
    if store is not None:
        store.put(
            key,
            {
                "candidate_sha256": candidate_sha256,
                "finalization_identity": finalization_identity,
                "record": finalized,
                "allocation": allocation,
            },
        )
    return finalized, allocation, False


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
                count += 1
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return count, sha256_file(path)


def byte_literal(text: str) -> dict[str, Any]:
    return {
        "type": "byte_literal",
        "value": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def safe_symbol(value: str, *, prefix: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    if not symbol or not symbol[0].isalpha():
        symbol = prefix + "_" + symbol
    return symbol[:96]


def semantic_evidence(
    *,
    source: dict[str, Any],
    source_id: str,
    objectives: set[str],
    producer_sha256: str,
) -> dict[str, Any]:
    return {
        "policy": SEMANTIC_SUPERVISION_POLICY,
        "evidence_tier": "licensed_human_task_gold",
        "producer_kind": "licensed_semantic_dataset",
        "producer_id": PRODUCER_POLICY,
        "producer_artifact_sha256": producer_sha256,
        "annotation_source_id": source_id,
        "annotation_source_sha256": source["content_sha256"],
        "claim_authority": "decision_grade_reference",
        "model_derived": False,
        "public_calibration_surface": False,
        "benchmark_payload_used": False,
        "objective_authority": {
            objective: objective in objectives for objective in TRAINING_OBJECTIVES
        },
        "optimizer_sampling_weight": 1.0,
    }


def scope(identity: str) -> dict[str, Any]:
    return {
        "user": "project-theseus-corpus",
        "project": "theseus",
        "conversation": identity,
        "privacy": "private_local",
    }


def bit_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "mean": None, "p50": None, "p95": None, "p99": None, "maximum": None}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": ordered[-1],
    }


def residual_supervision(
    identity: str,
    *,
    packet: dict[str, Any],
    hrl_state: dict[str, Any],
    source_family: str = "",
    importance: dict[str, Any] | None = None,
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    residual = packet["residual"]
    fidelity_labels = {"semantic": 0, "faithful": 1, "lexical": 2, "exact": 3}
    labels = {
        "interaction": 1 if (hrl_state.get("segments") or {}) else 0,
        "segment": 1 if residual["segment_frame"] else 0,
        "token": 2 if residual["token_tags"] else 0,
        "exact": 3 if residual["exact_object_handles"] else 0,
    }
    finalized = isinstance(importance, dict) and isinstance(allocation, dict)
    unit_allocation = residual_unit_allocation_receipt(residual["unit_packet"])
    compact_targets = None
    if finalized and "concept_capsules" in packet and "protected_objects" in packet:
        intervention_receipt = build_unit_intervention_targets(
            unit_packet=residual["unit_packet"],
            source_record_sha256=str(residual["unit_packet"]["source_record_sha256"]),
            global_state=(hrl_state.get("global") or {}),
            segment_residual=(residual.get("segment_frame") or {}),
            token_residuals=list(residual.get("token_tags") or []),
            concept_capsules=(packet.get("concept_capsules") or {}),
            exact_objects=(packet.get("protected_objects") or {}),
            source_family=source_family or identity,
        )
        compact_targets = compact_allocator_targets(intervention_receipt)
    return {
        "policy": "project_theseus_kerc_residual_supervision_v1",
        "labels_by_channel": labels,
        "record_fidelity_label": fidelity_labels[residual["fidelity"]],
        "record_fidelity_label_training_authority": False,
        "packet_wide_fidelity_drives_training": False,
        "residual_unit_allocation": unit_allocation,
        **(
            {
                "unit_intervention_targets": compact_targets,
                "unit_intervention_target_authority": (
                    "source_visible_typed_causal_interventions_provisional_until_independent_heldout_evaluation"
                ),
                "unit_intervention_target_producer_is_final_evaluator": False,
            }
            if compact_targets is not None
            else {}
        ),
        "allocation_target_authority": (
            "measured_structural_rate_distortion_with_calibrated_source_visible_importance"
            if finalized
            else "bootstrap_structural_label_only"
        ),
        "rate_distortion_optimality_claimed": False,
        **(
            {
                "importance": copy.deepcopy(importance),
                "rate_distortion_allocation": copy.deepcopy(allocation),
            }
            if finalized
            else {}
        ),
        "annotator_independent_of_model": True,
        "evidence_sha256": stable_hash(
            {
                "identity": identity,
                "labels_by_channel": labels,
                "rule": (
                    "measured_structural_rate_distortion_v1"
                    if finalized
                    else "source_fidelity_v1"
                ),
                "importance_receipt_sha256": (
                    importance.get("receipt_sha256") if finalized else None
                ),
                "allocation_sha256": (
                    allocation.get("allocation_sha256") if finalized else None
                ),
                "residual_unit_allocation_sha256": unit_allocation[
                    "receipt_sha256"
                ],
                "unit_intervention_targets_sha256": (
                    compact_targets["receipt_sha256"]
                    if compact_targets is not None
                    else None
                ),
            }
        ),
    }


def provisional_receipt(identity: str) -> dict[str, Any]:
    return {
        "policy": "project_theseus_kernel_english_verification_receipt_v1",
        "receipt_id": "pending:" + identity,
        "accepted": False,
        "verifier_id": "independent_verifier_required",
        "reviewer_independent_of_record_producer": False,
        "method": "licensed_semantic_dataset_plus_independent_schema_review",
        "evidence_sha256": "sha256:" + "0" * 64,
        "semantic_payload_sha256": "sha256:" + "0" * 64,
    }


def base_record(
    *,
    split: str,
    source_text: str,
    surface_target: str,
    program: dict[str, Any],
    answer_packet: dict[str, Any],
    source: dict[str, Any],
    source_id: str,
    source_group: str,
    objectives: set[str],
    producer_sha256: str,
    source_annotation: dict[str, Any],
    exact_residual: bool,
    explicit_spans: list[dict[str, Any]] | None = None,
    segment_frame: dict[str, Any] | None = None,
    token_tags: list[dict[str, Any]] | None = None,
    interaction_annotation: dict[str, Any] | None = None,
    interaction_entries: list[dict[str, Any]] | None = None,
    interaction_actor_id: str = "licensed_source_context",
    valid_realizations: list[dict[str, Any]] | None = None,
    concept_capsules: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    identity_payload = {
        "split": split,
        "source_id": source_id,
        "source_group": source_group,
        "source_text": source_text,
        "surface_target": surface_target,
        "source_annotation": source_annotation,
        "interaction_annotation": interaction_annotation,
        "valid_realizations": valid_realizations,
    }
    if concept_capsules is not None:
        identity_payload["concept_capsules"] = concept_capsules
    identity = stable_hash(identity_payload).split(":", 1)[1]
    state = create_hierarchical_residual_state(
        "kerc-corpus-" + identity[:24], scope=scope(identity[:24])
    )
    hrl_deltas: list[dict[str, Any]] = []
    if interaction_entries:
        operations = [
            {
                "op": "OVERRIDE",
                "segment_id": str(entry["segment_id"]),
                "key": str(entry["key"]),
                "value": copy.deepcopy(entry["value"]),
                "privacy": "interaction_private",
            }
            for entry in interaction_entries
        ]
        state, delta = apply_hierarchical_residual_delta(
            state,
            operations,
            expected_state_hash=state["state_hash"],
            actor_authority="document",
            actor_id=interaction_actor_id,
            provenance={
                "source": source["dataset_id"],
                "interaction_annotation_sha256": stable_hash(interaction_annotation),
            },
        )
        hrl_deltas.append(delta)
    packet = build_kernel_packet(
        source_text,
        program,
        hrl_state=state,
        provenance={
            "source": source["dataset_id"],
            "source_id": source_id,
            "source_annotation_sha256": stable_hash(source_annotation),
        },
        explicit_spans=explicit_spans or [],
        segment_frame=segment_frame,
        token_tags=token_tags or [],
        concept_capsules=concept_capsules or {},
        fidelity="exact" if exact_residual else "faithful",
    )
    return {
        "policy": TRAINING_RECORD_POLICY,
        "split": split,
        "language": "en",
        "source_text": source_text,
        "kernel_packet": packet,
        "hrl_state": state,
        "hrl_deltas": hrl_deltas,
        "answer_packet": answer_packet,
        "surface_target": surface_target,
        "provenance": {
            "source_id": source_id,
            "source_group": source_group,
            "license_spdx": source["license_spdx"],
            "permitted_use": "model_training",
            "dataset_id": source["dataset_id"],
            "dataset_revision": source["dataset_revision"],
        },
        "semantic_supervision": semantic_evidence(
            source=source,
            source_id=source_id,
            objectives=objectives,
            producer_sha256=producer_sha256,
        ),
        "residual_supervision": residual_supervision(
            identity,
            packet=packet,
            hrl_state=state,
            source_family=source_group,
        ),
        "verification_receipt": provisional_receipt(identity[:24]),
        "source_annotation": source_annotation,
        "interaction_annotation": interaction_annotation,
        "valid_realizations": valid_realizations,
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
        "external_inference": False,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }


def dolly_prompt(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or "").strip()


def load_dolly_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    path = resolve(source["path"])
    if sha256_file(path) != source["content_sha256"]:
        raise ValueError("Dolly source hash mismatch")
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        source_row = json.loads(raw)
        prompt = dolly_prompt(source_row)
        target = str(source_row.get("response") or "").strip()
        context = str(source_row.get("context") or "").strip()
        if (
            not 12 <= len(prompt) <= maximum_characters
            or not 2 <= len(target) <= maximum_characters
            or len(context) > maximum_characters
        ):
            rejects["length"] += 1
            continue
        prompt_hash = stable_hash(prompt.encode("utf-8"))
        target_hash = stable_hash(target.encode("utf-8"))
        if prompt_hash in seen_prompts or target_hash in seen_targets:
            rejects["duplicate_prompt_or_target"] += 1
            continue
        seen_prompts.add(prompt_hash)
        seen_targets.add(target_hash)
        annotation = {
            "source_kind": "dolly_human_instruction_response",
            "line_number": line_number,
            "instruction": str(source_row.get("instruction") or ""),
            "context": str(source_row.get("context") or ""),
            "response": str(source_row.get("response") or ""),
            "category": str(source_row.get("category") or ""),
            "source_row_sha256": stable_hash(source_row),
        }
        rows.append(
            {
                "selection_key": stable_hash(
                    {"dataset": source["dataset_id"], "annotation": annotation}
                ),
                "prompt": prompt,
                "target": target,
                "annotation": annotation,
                "category": annotation["category"],
            }
        )
    return sorted(rows, key=lambda row: row["selection_key"]), rejects


def dolly_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "RESPOND",
                "modality": "REQUIRED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[0, len(row["prompt"])]],
                "arguments": [
                    {"role": "TASK", "value": byte_literal(row["prompt"])},
                ],
            }
        ],
    }
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "RESPOND",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
                    {"role": "OUTPUT", "value": byte_literal(row["target"])},
                ],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored"},
    }
    identity = row["selection_key"].split(":", 1)[1]
    context = str(row["annotation"].get("context") or "").strip()
    interaction_annotation = (
        {
            "kind": "licensed_document_context",
            "source_row_sha256": row["annotation"]["source_row_sha256"],
            "context_sha256": stable_hash(context.encode("utf-8")),
            "context": context,
            "source_category": row["annotation"]["category"],
        }
        if context
        else None
    )
    interaction_entries = (
        [{"segment_id": "document_context", "key": "content", "value": context}]
        if context
        else []
    )
    return base_record(
        split=split,
        source_text=row["prompt"],
        surface_target=row["target"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="dolly:" + identity[:24],
        source_group="dolly-row:" + identity,
        objectives={"surface_direct_control_v1"},
        producer_sha256=producer_sha256,
        source_annotation=row["annotation"],
        exact_residual=False,
        interaction_annotation=interaction_annotation,
        interaction_entries=interaction_entries,
        interaction_actor_id="databricks_dolly_human_context",
    )


def load_dolly_grounded_question_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select narrow human QA rows with independently replayable extractive support."""

    path = resolve(source["path"])
    if sha256_file(path) != source["content_sha256"]:
        raise ValueError("Dolly grounded-question source hash mismatch")
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        source_row = json.loads(raw)
        prompt = dolly_prompt(source_row)
        target = str(source_row.get("response") or "").strip()
        context = str(source_row.get("context") or "").strip()
        if str(source_row.get("category") or "") != "closed_qa":
            rejects["not_closed_qa"] += 1
            continue
        question_match = DOLLY_QUESTION_FORM_RE.match(prompt)
        if not question_match or not prompt.endswith("?"):
            rejects["question_form"] += 1
            continue
        if (
            not 12 <= len(prompt) <= maximum_characters
            or not 2 <= len(target) <= min(512, maximum_characters)
            or not 16 <= len(context) <= maximum_characters
        ):
            rejects["length"] += 1
            continue
        if context.count(target) != 1 or len(target) / len(context) > 0.5:
            rejects["not_unique_bounded_exact_support"] += 1
            continue
        prompt_hash = stable_hash(prompt.encode("utf-8"))
        target_hash = stable_hash(target.encode("utf-8"))
        if prompt_hash in seen_prompts or target_hash in seen_targets:
            rejects["duplicate_prompt_or_target"] += 1
            continue
        seen_prompts.add(prompt_hash)
        seen_targets.add(target_hash)
        answer_start = context.index(target)
        question_form = question_match.group(1).lower().replace(" ", "_")
        annotation = {
            "source_kind": "dolly_human_unique_extractive_question_answer",
            "line_number": line_number,
            "instruction": str(source_row.get("instruction") or ""),
            "context": str(source_row.get("context") or ""),
            "response": str(source_row.get("response") or ""),
            "category": "closed_qa",
            "question_form": question_form,
            "answer_span": [answer_start, answer_start + len(target)],
            "support_relation": "unique_contiguous_exact_span",
            "support_claim_scope": "extractive_source_support_only",
            "broad_entailment_or_truth_claimed": False,
            "source_row_sha256": stable_hash(source_row),
        }
        rows.append(
            {
                "selection_key": stable_hash(
                    {"dataset": source["dataset_id"], "annotation": annotation}
                ),
                "prompt": prompt,
                "target": target,
                "context": context,
                "question_form": question_form,
                "annotation": annotation,
            }
        )
    return sorted(rows, key=lambda row: row["selection_key"]), rejects


def select_dolly_grounded_questions(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    required_question_forms: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Create source-disjoint, form-diverse deterministic private splits."""

    available = list(rows)
    output: dict[str, list[dict[str, Any]]] = {}
    split_rows = list(counts.items())
    for split_index, (split, raw_count) in enumerate(split_rows):
        count = int(raw_count)
        future_split_count = len(split_rows) - split_index - 1
        selected: list[dict[str, Any]] = []
        for form in required_question_forms:
            match = next(
                (row for row in available if row["question_form"] == form), None
            )
            if match is None:
                raise ValueError(
                    f"Dolly grounded-question form unavailable: {split}:{form}"
                )
            selected.append(match)
            available.remove(match)
        if len(selected) > count:
            raise ValueError("grounded-question split smaller than required form set")
        form_counts = Counter(row["question_form"] for row in selected)
        while len(selected) < count:
            available_form_counts = Counter(row["question_form"] for row in available)
            fillable = [
                row
                for row in available
                if row["question_form"] not in required_question_forms
                or available_form_counts[row["question_form"]] > future_split_count
            ]
            if not fillable:
                raise ValueError(f"insufficient Dolly grounded questions: {split}")
            match = min(
                fillable,
                key=lambda row: (form_counts[row["question_form"]], row["selection_key"]),
            )
            selected.append(match)
            available.remove(match)
            form_counts[match["question_form"]] += 1
        output[split] = sorted(selected, key=lambda row: row["selection_key"])
    return output


def dolly_grounded_question_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    context_sha256 = stable_hash(row["context"].encode("utf-8"))
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "ANSWER_FROM_CONTEXT",
                "modality": "REQUIRED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[0, len(row["prompt"])]],
                "arguments": [
                    {"role": "QUESTION", "value": byte_literal(row["prompt"])},
                    {"role": "QUESTION_FORM", "value": byte_literal(row["question_form"])},
                    {"role": "CONTEXT_SHA256", "value": byte_literal(context_sha256)},
                ],
            }
        ],
    }
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "SUPPORTED_ANSWER",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
                    {"role": "ANSWER_SPAN", "value": byte_literal(row["target"])},
                    {"role": "CONTEXT_SHA256", "value": byte_literal(context_sha256)},
                ],
            }
        ],
        "decision": {
            "policy": ANSWER_DECISION_POLICY,
            "disposition": "ANSWER",
            "evidence_status": "SUPPORTED",
            "uncertainty_state": "RESOLVED",
            "confidence": 1.0,
            "controlling_claim_ids": ["claim-1"],
            "unresolved_ambiguity_ids": [],
        },
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored_extractive_answer"},
    }
    identity = row["selection_key"].split(":", 1)[1]
    interaction_annotation = {
        "kind": "licensed_grounded_question_context",
        "policy": DOLLY_GROUNDED_QUESTION_POLICY,
        "source_row_sha256": row["annotation"]["source_row_sha256"],
        "context_sha256": context_sha256,
        "answer_span": list(row["annotation"]["answer_span"]),
        "support_relation": "unique_contiguous_exact_span",
        "support_claim_scope": "extractive_source_support_only",
    }
    return base_record(
        split=split,
        source_text=row["prompt"],
        surface_target=row["target"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="dolly-grounded:" + identity[:24],
        source_group="dolly-grounded-row:"
        + row["annotation"]["source_row_sha256"].split(":", 1)[1],
        objectives=set(TRAINING_OBJECTIVES),
        producer_sha256=producer_sha256,
        source_annotation=row["annotation"],
        exact_residual=False,
        interaction_annotation=interaction_annotation,
        interaction_entries=[
            {"segment_id": "document_context", "key": "content", "value": row["context"]},
            {
                "segment_id": "question_contract",
                "key": "context_sha256",
                "value": context_sha256,
            },
        ],
        interaction_actor_id="databricks_dolly_grounded_question",
    )


def source_labels(row: dict[str, Any]) -> dict[str, float]:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    names = labels.get("name") if isinstance(labels.get("name"), list) else []
    values = labels.get("value") if isinstance(labels.get("value"), list) else []
    return {str(name): float(value) for name, value in zip(names, values)}


def oasst_row_eligible(row: dict[str, Any], source: dict[str, Any]) -> bool:
    labels = source_labels(row)
    safety_dimensions = tuple(source["maximum_label_values"])
    return bool(
        str(row.get("lang") or "").lower() == "en"
        and row.get("review_result") is True
        and row.get("deleted") is not True
        and row.get("synthetic") is not True
        and str(row.get("tree_state") or "") == "ready_for_export"
        and str(row.get("text") or "").strip()
        and float(labels.get("quality", -1.0)) >= float(source["minimum_quality"])
        and all(
            float(labels.get(dimension, 0.0))
            <= float(source["maximum_label_values"][dimension])
            for dimension in safety_dimensions
        )
    )


def oasst_answer_packet(text: str) -> dict[str, Any]:
    return {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "DIALOGUE_RESPONSE",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [{"role": "CONTENT", "value": byte_literal(text)}],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored_conversation"},
    }


OASST_BEHAVIOR_POLICY = "project_theseus_oasst_explicit_answer_behavior_v1"
_CLARIFICATION_RULES = (
    ("request_more_detail", re.compile(r"^(?:i(?:'m| am) sorry[^?!.]{0,120}[,.]?\s*)?(?:could|can|would|will) you (?:please )?(?:clarify|provide|specify|explain|elaborate|tell me|give me)\b")),
    ("imperative_more_detail", re.compile(r"^please (?:clarify|provide|specify|explain|elaborate)\b")),
    ("meaning_question", re.compile(r"^(?:what do you mean|which .{0,100} do you mean|do you mean)\b")),
    ("explicit_choice_question", re.compile(r"^there (?:are|is) .{0,120}(?:which|what).{0,120}\?")),
)
_ABSTENTION_RULES = (
    ("explicit_unknown", re.compile(r"^(?:i(?:'m| am) sorry,? (?:but )?)?i (?:do not|don't) know\b")),
    ("explicit_cannot_determine", re.compile(r"^(?:as [^.!?]{1,120}[,.]\s*)?[^.!?]{0,120}\bi (?:cannot|can't) (?:determine|answer|know|tell)\b")),
    ("explicit_insufficient", re.compile(r"^(?:there is|there's) (?:insufficient|not enough) (?:information|context|details)\b")),
    ("explicit_missing_context", re.compile(r"^i (?:do not|don't) have (?:enough |sufficient )?(?:information|context|details)\b")),
)


def explicit_answer_behavior(text: str) -> tuple[str, str] | None:
    """Admit only surface-explicit human behavior; this is not truth verification."""

    normalized = " ".join(text.strip().lower().split())[:400]
    if "?" in normalized:
        for rule_id, pattern in _CLARIFICATION_RULES:
            if pattern.search(normalized):
                return "CLARIFY", rule_id
    for rule_id, pattern in _ABSTENTION_RULES:
        if pattern.search(normalized):
            return "ABSTAIN", rule_id
    return None


def oasst_behavior_answer_packet(
    text: str, *, disposition: str, ambiguity_id: str
) -> dict[str, Any]:
    predicate = "REQUEST_CLARIFICATION" if disposition == "CLARIFY" else "ABSTAIN"
    uncertainty = "AMBIGUOUS" if disposition == "CLARIFY" else "INSUFFICIENT_CONTEXT"
    evidence = "AMBIGUOUS" if disposition == "CLARIFY" else "INSUFFICIENT_CONTEXT"
    return {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": predicate,
                "modality": "REQUIRED" if disposition == "CLARIFY" else "UNKNOWN",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [{"role": "CONTENT", "value": byte_literal(text)}],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored_conversation"},
        "decision": {
            "policy": ANSWER_DECISION_POLICY,
            "disposition": disposition,
            "evidence_status": evidence,
            "uncertainty_state": uncertainty,
            "confidence": 1.0,
            "controlling_claim_ids": ["claim-1"],
            "unresolved_ambiguity_ids": [ambiguity_id] if disposition == "CLARIFY" else [],
        },
    }


def load_oasst_candidates(
    source: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    nodes_by_official_split: dict[str, dict[str, dict[str, Any]]] = {}
    rejects: Counter[str] = Counter()
    observed_file_hashes: dict[str, str] = {}
    for official_split, contract in source["files"].items():
        path = resolve(contract["path"])
        observed_file_hashes[official_split] = sha256_file(path)
        if observed_file_hashes[official_split] != contract["content_sha256"]:
            raise ValueError(f"OASST2 source hash mismatch: {official_split}")
        rows = pq.read_table(path).to_pylist()
        nodes_by_official_split[official_split] = {
            str(row["message_id"]): row for row in rows if row.get("message_id")
        }
    if stable_hash(observed_file_hashes) != source["content_sha256"]:
        raise ValueError("OASST2 aggregate source identity mismatch")

    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for official_split, nodes in nodes_by_official_split.items():
        assistants_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in nodes.values():
            if (
                oasst_row_eligible(row, source)
                and str(row.get("role") or "") == "assistant"
                and row.get("rank") in (0, 1)
            ):
                assistants_by_parent[str(row.get("parent_id") or "")].append(row)
        for parent_id, responses in assistants_by_parent.items():
            if sorted(int(row["rank"]) for row in responses) != [0, 1]:
                rejects["rank_zero_one_pair_missing"] += 1
                continue
            chain: list[dict[str, Any]] = []
            current = nodes.get(parent_id)
            seen: set[str] = set()
            while current and str(current["message_id"]) not in seen and len(chain) < 8:
                if not oasst_row_eligible(current, source):
                    chain = []
                    break
                seen.add(str(current["message_id"]))
                chain.append(current)
                current = nodes.get(str(current.get("parent_id") or ""))
            chain.reverse()
            if current is not None or len(chain) < 3:
                rejects["context_free_or_incomplete_ancestry"] += 1
                continue
            roles = [
                "user" if str(row.get("role")) == "prompter" else "assistant"
                for row in chain
            ]
            if roles != ["user" if index % 2 == 0 else "assistant" for index in range(len(chain))]:
                rejects["nonalternating_ancestry"] += 1
                continue
            current_prompt = str(chain[-1]["text"]).strip()
            responses = sorted(responses, key=lambda row: int(row["rank"]))
            targets = [str(row["text"]).strip() for row in responses]
            if len(set(targets)) != len(targets):
                rejects["duplicate_ranked_realization"] += 1
                continue
            prior_turns = chain[:-1]
            compiled_context = [
                [f"turn_{index:03d}", key, value]
                for index, prior in enumerate(prior_turns)
                for key, value in (
                    ("content", str(prior["text"]).strip()),
                    (
                        "role",
                        "user"
                        if str(prior.get("role") or "") == "prompter"
                        else "assistant",
                    ),
                )
            ]
            if (
                len(prior_turns) < int(source["minimum_prior_turns"])
                or len(prior_turns) > int(source["maximum_prior_turns"])
                or len(current_prompt) > int(source["maximum_current_characters"])
                or any(len(target) > int(source["maximum_response_characters"]) for target in targets)
                or sum(len(str(row["text"])) for row in prior_turns)
                > int(source["maximum_context_characters"])
                or len(canonical_json(compiled_context).encode("utf-8"))
                > int(source["maximum_compiled_context_bytes"])
            ):
                rejects["bounded_context_or_response_length"] += 1
                continue
            tree_hash = stable_hash(str(chain[-1]["message_tree_id"]).encode("utf-8"))
            split = "private_train"
            if official_split == "validation":
                split = (
                    "private_dev"
                    if int(tree_hash.split(":", 1)[1][:2], 16) % 2 == 0
                    else "private_eval"
                )
            interaction_turns = [
                {
                    "turn_index": index,
                    "role": roles[index],
                    "content": str(row["text"]).strip(),
                    "source_message_sha256": stable_hash(
                        str(row["message_id"]).encode("utf-8")
                    ),
                }
                for index, row in enumerate(prior_turns)
            ]
            annotation = {
                "source_kind": "oasst2_reviewed_conversation_tree",
                "official_split": official_split,
                "message_tree_sha256": tree_hash,
                "parent_message_sha256": stable_hash(parent_id.encode("utf-8")),
                "current_prompt": current_prompt,
                "interaction_turns": interaction_turns,
                "responses": [
                    {
                        "rank": int(row["rank"]),
                        "text": str(row["text"]).strip(),
                        "source_message_sha256": stable_hash(
                            str(row["message_id"]).encode("utf-8")
                        ),
                        "quality": source_labels(row).get("quality"),
                    }
                    for row in responses
                ],
            }
            selection_key = stable_hash(
                {"dataset": source["dataset_id"], "annotation": annotation}
            )
            selected[split].append(
                {
                    "selection_key": selection_key,
                    "source_group": "oasst2-tree:" + tree_hash.split(":", 1)[1],
                    "prompt": current_prompt,
                    "interaction_turns": interaction_turns,
                    "targets": targets,
                    "annotation": annotation,
                }
            )
    return {
        split: sorted(rows, key=lambda row: row["selection_key"])
        for split, rows in selected.items()
    }, rejects


def load_oasst_behavior_candidates(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Extract conservative, human-written clarification/abstention examples.

    The rule only identifies an explicit surface behavior. It does not establish
    that the behavior was optimal, factually correct, or semantically exhaustive.
    """

    contract = source["files"]["train"]
    path = resolve(contract["path"])
    if sha256_file(path) != contract["content_sha256"]:
        raise ValueError("OASST2 behavior source hash mismatch")
    rows = pq.read_table(path).to_pylist()
    nodes = {str(row["message_id"]): row for row in rows if row.get("message_id")}
    rejects: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for response in nodes.values():
        if not (
            oasst_row_eligible(response, source)
            and str(response.get("role") or "") == "assistant"
        ):
            continue
        behavior = explicit_answer_behavior(str(response.get("text") or ""))
        if behavior is None:
            continue
        parent_id = str(response.get("parent_id") or "")
        parent = nodes.get(parent_id)
        if not (
            parent
            and oasst_row_eligible(parent, source)
            and str(parent.get("role") or "") == "prompter"
        ):
            rejects["missing_reviewed_user_parent"] += 1
            continue
        chain: list[dict[str, Any]] = []
        current = parent
        seen: set[str] = set()
        while current and str(current["message_id"]) not in seen and len(chain) < 16:
            if not oasst_row_eligible(current, source):
                chain = []
                break
            seen.add(str(current["message_id"]))
            chain.append(current)
            current = nodes.get(str(current.get("parent_id") or ""))
        chain.reverse()
        if current is not None or not chain:
            rejects["incomplete_ancestry"] += 1
            continue
        roles = [
            "user" if str(row.get("role") or "") == "prompter" else "assistant"
            for row in chain
        ]
        if roles != ["user" if index % 2 == 0 else "assistant" for index in range(len(chain))]:
            rejects["nonalternating_ancestry"] += 1
            continue
        prompt = str(parent["text"]).strip()
        target = str(response["text"]).strip()
        prior = chain[:-1][-int(source["maximum_prior_turns"]) :]
        interaction_turns = [
            {
                "turn_index": index,
                "role": "user" if str(row.get("role") or "") == "prompter" else "assistant",
                "content": str(row["text"]).strip(),
                "source_message_sha256": stable_hash(str(row["message_id"]).encode("utf-8")),
            }
            for index, row in enumerate(prior)
        ]
        compiled_context = [
            [f"turn_{turn['turn_index']:03d}", key, turn[key]]
            for turn in interaction_turns
            for key in ("content", "role")
        ]
        if (
            len(prompt) > int(source["maximum_current_characters"])
            or len(target) > int(source["maximum_response_characters"])
            or sum(len(turn["content"]) for turn in interaction_turns)
            > int(source["maximum_context_characters"])
            or len(canonical_json(compiled_context).encode("utf-8"))
            > int(source["maximum_compiled_context_bytes"])
        ):
            rejects["bounded_context_or_response_length"] += 1
            continue
        disposition, rule_id = behavior
        tree_hash = stable_hash(str(response["message_tree_id"]).encode("utf-8"))
        annotation = {
            "source_kind": "oasst2_reviewed_explicit_answer_behavior",
            "official_split": "train",
            "message_tree_sha256": tree_hash,
            "parent_message_sha256": stable_hash(parent_id.encode("utf-8")),
            "response_message_sha256": stable_hash(
                str(response["message_id"]).encode("utf-8")
            ),
            "prompt": prompt,
            "target": target,
            "interaction_turns": interaction_turns,
            "behavior_policy": OASST_BEHAVIOR_POLICY,
            "disposition": disposition,
            "matched_rule_id": rule_id,
            "behavior_claim_scope": "explicit_human_surface_behavior_only",
            "optimality_or_truth_verified": False,
        }
        candidates.append(
            {
                "selection_key": stable_hash(
                    {"dataset": source["dataset_id"], "annotation": annotation}
                ),
                "source_group": "oasst2-tree:" + tree_hash.split(":", 1)[1],
                "prompt": prompt,
                "target": target,
                "interaction_turns": interaction_turns,
                "annotation": annotation,
                "disposition": disposition,
            }
        )
    by_tree: dict[str, dict[str, Any]] = {}
    for row in sorted(candidates, key=lambda value: value["selection_key"]):
        if row["source_group"] in by_tree:
            rejects["additional_behavior_in_same_tree"] += 1
            continue
        by_tree[row["source_group"]] = row
    return list(by_tree.values()), rejects


def select_oasst_behavior(
    rows: list[dict[str, Any]], counts: dict[str, dict[str, int]]
) -> dict[str, list[dict[str, Any]]]:
    pools = {
        disposition: sorted(
            (row for row in rows if row["disposition"] == disposition),
            key=lambda row: row["selection_key"],
        )
        for disposition in ("CLARIFY", "ABSTAIN")
    }
    cursors = {disposition: 0 for disposition in pools}
    output: dict[str, list[dict[str, Any]]] = {}
    for split in ("private_train", "private_dev", "private_eval"):
        selected: list[dict[str, Any]] = []
        for disposition in ("CLARIFY", "ABSTAIN"):
            count = int(counts[split][disposition])
            start = cursors[disposition]
            end = start + count
            if len(pools[disposition]) < end:
                raise ValueError(
                    f"insufficient OASST2 behavior rows: {split}:{disposition}:"
                    f"{len(pools[disposition]) - start}:{count}"
                )
            selected.extend(pools[disposition][start:end])
            cursors[disposition] = end
        output[split] = sorted(selected, key=lambda row: row["selection_key"])
    return output


def oasst_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "DIALOGUE_RESPOND",
                "modality": "REQUIRED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[0, len(row["prompt"])]],
                "arguments": [
                    {"role": "USER_UTTERANCE", "value": byte_literal(row["prompt"])}
                ],
            }
        ],
    }
    valid_realizations = [
        {
            "realization_id": f"oasst-rank-{rank}",
            "surface_target": target,
            "answer_packet": oasst_answer_packet(target),
            "source_rank": rank,
            "human_source_bound": True,
            "source_message_sha256": row["annotation"]["responses"][rank][
                "source_message_sha256"
            ],
        }
        for rank, target in enumerate(row["targets"])
    ]
    interaction_entries = [
        {
            "segment_id": f"turn_{turn['turn_index']:03d}",
            "key": key,
            "value": turn[key],
        }
        for turn in row["interaction_turns"]
        for key in ("role", "content")
    ]
    identity = row["selection_key"].split(":", 1)[1]
    return base_record(
        split=split,
        source_text=row["prompt"],
        surface_target=row["targets"][0],
        program=program,
        answer_packet=valid_realizations[0]["answer_packet"],
        source=source,
        source_id="oasst2:" + identity[:24],
        source_group=row["source_group"],
        objectives=set(source["allowed_objectives"]),
        producer_sha256=producer_sha256,
        source_annotation=row["annotation"],
        exact_residual=False,
        interaction_annotation={
            "kind": "reviewed_conversation_ancestry",
            "turns": row["interaction_turns"],
        },
        interaction_entries=interaction_entries,
        interaction_actor_id="openassistant_oasst2_reviewed_tree",
        valid_realizations=valid_realizations,
    )


def oasst_behavior_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    disposition = str(row["disposition"])
    ambiguity_id = "amb-" + row["selection_key"].split(":", 1)[1][:16]
    answer = oasst_behavior_answer_packet(
        row["target"], disposition=disposition, ambiguity_id=ambiguity_id
    )
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "DIALOGUE_RESPOND",
                "modality": "REQUIRED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[0, len(row["prompt"])]],
                "arguments": [
                    {"role": "USER_UTTERANCE", "value": byte_literal(row["prompt"])}
                ],
            }
        ],
    }
    realization = {
        "realization_id": "oasst-explicit-behavior",
        "surface_target": row["target"],
        "answer_packet": answer,
        "source_rank": 0,
        "human_source_bound": True,
        "source_message_sha256": row["annotation"]["response_message_sha256"],
    }
    entries = [
        {
            "segment_id": f"turn_{turn['turn_index']:03d}",
            "key": key,
            "value": turn[key],
        }
        for turn in row["interaction_turns"]
        for key in ("role", "content")
    ]
    return base_record(
        split=split,
        source_text=row["prompt"],
        surface_target=row["target"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="oasst2-behavior:" + row["selection_key"].split(":", 1)[1][:24],
        source_group=row["source_group"],
        objectives=set(source["allowed_objectives"]),
        producer_sha256=producer_sha256,
        source_annotation=row["annotation"],
        exact_residual=False,
        interaction_annotation={
            "kind": "reviewed_conversation_ancestry",
            "turns": row["interaction_turns"],
        },
        interaction_entries=entries,
        interaction_actor_id="openassistant_oasst2_reviewed_tree",
        valid_realizations=[realization],
    )


def parse_graf(path: Path) -> tuple[
    dict[str, list[tuple[str, dict[str, str]]]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    root = ET.parse(path).getroot()
    annotations: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    edges: dict[str, list[str]] = defaultdict(list)
    links: dict[str, list[str]] = {}
    for node in root.findall(GRAF + "node"):
        link = node.find(GRAF + "link")
        if link is not None:
            links[str(node.get(XML_ID))] = str(link.get("targets") or "").split()
    for annotation in root.findall(GRAF + "a"):
        fields: dict[str, str] = {}
        feature_structure = annotation.find(GRAF + "fs")
        if feature_structure is not None:
            for field in feature_structure.findall(GRAF + "f"):
                fields[str(field.get("name"))] = str(field.get("value") or "")
        annotations[str(annotation.get("ref"))].append(
            (str(annotation.get("label") or ""), fields)
        )
    for edge in root.findall(GRAF + "edge"):
        edges[str(edge.get("from"))].append(str(edge.get("to")))
    return dict(annotations), dict(edges), links


def parse_direct_graf_annotations(path: Path) -> list[dict[str, Any]]:
    """Read annotations whose nodes link directly to local character regions."""

    root = ET.parse(path).getroot()
    regions = {
        str(region.get(XML_ID)): tuple(
            int(value) for value in str(region.get("anchors") or "").split()
        )
        for region in root.findall(GRAF + "region")
    }
    links: dict[str, list[str]] = {}
    for node in root.findall(GRAF + "node"):
        link = node.find(GRAF + "link")
        if link is not None:
            links[str(node.get(XML_ID))] = str(link.get("targets") or "").split()
    output: list[dict[str, Any]] = []
    for annotation in root.findall(GRAF + "a"):
        spans = [regions[target] for target in links.get(str(annotation.get("ref")), ()) if target in regions]
        if not spans:
            continue
        fields: dict[str, str] = {}
        feature_structure = annotation.find(GRAF + "fs")
        if feature_structure is not None:
            for field in feature_structure.findall(GRAF + "f"):
                fields[str(field.get("name") or "")] = str(field.get("value") or "")
        output.append(
            {
                "annotation_id": str(annotation.get(XML_ID) or ""),
                "node_id": str(annotation.get("ref") or ""),
                "label": str(annotation.get("label") or ""),
                "fields": dict(sorted(fields.items())),
                "start": min(start for start, _end in spans),
                "end": max(end for _start, end in spans),
            }
        )
    return output


def masc_decision_semantic_instances(base: Path, root: Path) -> list[dict[str, Any]]:
    """Join MASC's human epistemic, opinion, and event layers by source sentence."""

    source_text = Path(str(base) + ".txt").read_text(encoding="utf-8", errors="replace")
    document_id = str(base.relative_to(root)).replace(os.sep, "/")
    sentence_rows = parse_direct_graf_annotations(Path(str(base) + "-s.xml"))
    annotations: list[dict[str, Any]] = []
    for layer in ("cb", "event", "mpqa"):
        path = Path(str(base) + f"-{layer}.xml")
        if not path.exists():
            continue
        for row in parse_direct_graf_annotations(path):
            if layer == "cb" and row["label"] == "Not Applicable":
                continue
            if layer == "mpqa" and row["label"] not in MASC_MPQA_SEMANTIC_LABELS:
                continue
            annotations.append({"layer": layer, **row})
    output: list[dict[str, Any]] = []
    for sentence in sentence_rows:
        start, end = int(sentence["start"]), int(sentence["end"])
        members = [
            {
                **row,
                "start": int(row["start"]) - start,
                "end": int(row["end"]) - start,
                "text": source_text[int(row["start"]):int(row["end"])],
            }
            for row in annotations
            if start <= int(row["start"]) < int(row["end"]) <= end
        ]
        if not members:
            continue
        members.sort(
            key=lambda row: (
                row["start"], row["end"], row["layer"], row["label"], row["annotation_id"]
            )
        )
        annotation = {
            "policy": MASC_DECISION_SEMANTICS_POLICY,
            "document_id": document_id,
            "sentence_id": sentence["fields"].get("id", sentence["node_id"]),
            "sentence_start": start,
            "sentence_end": end,
            "sentence": source_text[start:end],
            "annotations": members,
            "missingness": {
                "cb": not any(row["layer"] == "cb" for row in members),
                "event": not any(row["layer"] == "event" for row in members),
                "mpqa": not any(row["layer"] == "mpqa" for row in members),
                "event_coreference_grouping": True,
                "complete_sentence_semantics": True,
                "truth": True,
            },
        }
        output.append(
            {
                "selection_key": stable_hash(annotation),
                "document_id": document_id,
                "annotation": annotation,
            }
        )
    return output


def load_masc_decision_semantic_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> dict[str, list[dict[str, Any]]]:
    root = resolve(source["extracted_root"]) / "data"
    dev_groups = set(source["document_groups"]["private_dev"])
    eval_groups = set(source["document_groups"]["private_eval"])
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for text_path in sorted(root.rglob("*.txt")):
        base = Path(str(text_path)[:-4])
        if not Path(str(base) + "-s.xml").exists() or not any(
            Path(str(base) + f"-{layer}.xml").exists() for layer in ("cb", "event", "mpqa")
        ):
            continue
        document_id = str(base.relative_to(root)).replace(os.sep, "/")
        split = (
            "private_dev" if document_id in dev_groups else
            "private_eval" if document_id in eval_groups else
            "private_train"
        )
        for row in masc_decision_semantic_instances(base, root):
            if 0 < len(row["annotation"]["sentence"]) <= maximum_characters:
                selected[split].append(row)
    return {
        split: sorted(rows, key=lambda row: row["selection_key"])
        for split, rows in selected.items()
    }


def descendants(node: str, edges: dict[str, list[str]]) -> set[str]:
    output: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in output:
            continue
        output.add(current)
        stack.extend(edges.get(current, ()))
    return output


def field_rows(
    annotations: dict[str, list[tuple[str, dict[str, str]]]], node: str, label: str
) -> list[dict[str, str]]:
    return [fields for observed, fields in annotations.get(node, ()) if observed == label]


def token_spans(
    node: str,
    *,
    edges: dict[str, list[str]],
    token_links: dict[str, list[str]],
    anchors: dict[str, tuple[int, int]],
) -> list[tuple[int, int]]:
    spans = {
        anchors[region]
        for descendant in descendants(node, edges)
        for region in token_links.get(descendant, ())
        if region in anchors
    }
    return sorted(spans)


def masc_named_entities(
    base: Path,
    *,
    source_text: str,
    anchors: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Read MASC's manual named entities through their Penn-token anchors."""

    entity_path = Path(str(base) + "-ne.xml")
    penn_path = Path(str(base) + "-penn.xml")
    if not entity_path.exists() or not penn_path.exists():
        return []
    entity_annotations, entity_edges, _ = parse_graf(entity_path)
    _, _, penn_links = parse_graf(penn_path)
    candidates: list[dict[str, Any]] = []
    for node, rows in entity_annotations.items():
        for label, fields in rows:
            object_type = MASC_ENTITY_TYPES.get(label)
            if object_type is None:
                continue
            spans = token_spans(
                node,
                edges=entity_edges,
                token_links=penn_links,
                anchors=anchors,
            )
            if not spans:
                continue
            start = min(value[0] for value in spans)
            end = max(value[1] for value in spans)
            if not 0 <= start < end <= len(source_text):
                continue
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "object_type": object_type,
                    "copy_policy": "EXACT",
                    "source_label": label,
                    "source_features": dict(sorted(fields.items())),
                    "text": source_text[start:end],
                }
            )
    unique = {
        (row["start"], row["end"], row["object_type"], row["source_label"]): row
        for row in candidates
    }
    selected: list[dict[str, Any]] = []
    for row in sorted(
        unique.values(),
        key=lambda value: (
            value["start"],
            -(value["end"] - value["start"]),
            value["object_type"],
        ),
    ):
        if any(row["start"] < prior["end"] and row["end"] > prior["start"] for prior in selected):
            continue
        selected.append(row)
    return sorted(selected, key=lambda value: (value["start"], value["end"]))


def masc_document_instances(path: Path, root: Path) -> list[dict[str, Any]]:
    base = Path(str(path)[: -len("-fn.xml")])
    document_id = str(base.relative_to(root)).replace(os.sep, "/")
    source_text = Path(str(base) + ".txt").read_text(
        encoding="utf-8", errors="replace"
    )
    annotations, edges, _ = parse_graf(path)
    _, _, token_links = parse_graf(Path(str(base) + "-fntok.xml"))
    segment_root = ET.parse(Path(str(base) + "-seg.xml")).getroot()
    anchors = {
        str(region.get(XML_ID)): tuple(
            int(value) for value in str(region.get("anchors") or "").split()
        )
        for region in segment_root.findall(GRAF + "region")
    }
    named_entities = masc_named_entities(
        base,
        source_text=source_text,
        anchors=anchors,
    )
    output: list[dict[str, Any]] = []
    sentence_nodes = [
        node
        for node, rows in annotations.items()
        if any(label == "sentence" for label, _fields in rows)
    ]
    for sentence_node in sentence_nodes:
        sentence_token_spans = token_spans(
            sentence_node, edges=edges, token_links=token_links, anchors=anchors
        )
        if not sentence_token_spans:
            continue
        sentence_start = min(start for start, _end in sentence_token_spans)
        sentence_end = max(end for _start, end in sentence_token_spans)
        sentence = source_text[sentence_start:sentence_end]
        protected_spans = [
            {
                **entity,
                "start": entity["start"] - sentence_start,
                "end": entity["end"] - sentence_start,
            }
            for entity in named_entities
            if sentence_start <= entity["start"] < entity["end"] <= sentence_end
        ]
        for annotation_node in edges.get(sentence_node, ()):
            sets = field_rows(annotations, annotation_node, "annotationSet")
            if not sets:
                continue
            frame = sets[0]
            if frame.get("status") != "MANUAL" or not frame.get("frameName"):
                continue
            target_spans: list[tuple[int, int]] = []
            frame_elements: list[dict[str, Any]] = []
            for child in edges.get(annotation_node, ()):
                spans = token_spans(child, edges=edges, token_links=token_links, anchors=anchors)
                relative_spans = [
                    [start - sentence_start, end - sentence_start]
                    for start, end in spans
                    if sentence_start <= start < end <= sentence_end
                ]
                if field_rows(annotations, child, "Target"):
                    target_spans.extend((int(a), int(b)) for a, b in relative_spans)
                for fields in field_rows(annotations, child, "FE"):
                    if relative_spans:
                        frame_elements.append(
                            {
                                "role": str(fields.get("FE") or "ROLE"),
                                "rank": str(fields.get("rank") or ""),
                                "gf": str(fields.get("GF") or ""),
                                "pt": str(fields.get("PT") or ""),
                                "spans": relative_spans,
                                "text": " ".join(
                                    sentence[start:end] for start, end in relative_spans
                                ),
                            }
                        )
            if not target_spans or not frame_elements:
                continue
            annotation = {
                "source_kind": "masc_manual_framenet",
                "document_id": document_id,
                "sentence_node": sentence_node,
                "annotation_set_node": annotation_node,
                "annotation_set_id": str(frame.get("ID") or ""),
                "frame_name": str(frame["frameName"]),
                "lexical_unit": str(frame.get("luName") or ""),
                "status": str(frame.get("status") or ""),
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "sentence": sentence,
                "target_spans": [list(value) for value in sorted(set(target_spans))],
                "frame_elements": sorted(
                    frame_elements,
                    key=lambda row: (row["role"], row["spans"], row["text"]),
                ),
                "protected_spans": protected_spans,
            }
            output.append(
                {
                    "selection_key": stable_hash(annotation),
                    "document_id": document_id,
                    "sentence_identity": stable_hash(
                        {"document_id": document_id, "sentence_node": sentence_node}
                    ),
                    "annotation": annotation,
                }
            )
    return output


def load_masc_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    archive = resolve(source["archive_path"])
    if sha256_file(archive) != source["content_sha256"]:
        raise ValueError("MASC source hash mismatch")
    root = resolve(source["extracted_root"]) / "data"
    dev_groups = set(source["document_groups"]["private_dev"])
    eval_groups = set(source["document_groups"]["private_eval"])
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejects: Counter[str] = Counter()
    for path in sorted(root.rglob("*-fn.xml")):
        document_id = str(Path(str(path)[: -len("-fn.xml")]).relative_to(root)).replace(
            os.sep, "/"
        )
        split = (
            "private_dev"
            if document_id in dev_groups
            else "private_eval"
            if document_id in eval_groups
            else "private_train"
        )
        for row in masc_document_instances(path, root):
            sentence = row["annotation"]["sentence"]
            if not 12 <= len(sentence) <= maximum_characters:
                rejects["sentence_length"] += 1
                continue
            selected[split].append(row)
    return {
        split: sorted(rows, key=lambda row: row["selection_key"])
        for split, rows in selected.items()
    }, rejects


def masc_annotation_with_prior(
    row: dict[str, Any],
    priors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    annotation = copy.deepcopy(row["annotation"])
    lexical_unit = str(annotation["lexical_unit"]).strip().lower()
    prior = priors.get(lexical_unit)
    if prior is not None and annotation["frame_name"] in {
        alternative["frame_name"] for alternative in prior["alternatives"]
    }:
        annotation["contextual_frame_ambiguity"] = copy.deepcopy(prior)
    return annotation


def masc_semantic_value(
    element: dict[str, Any],
    *,
    sentence: str,
    protected_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spans = element["spans"]
    if spans:
        bounds = (min(value[0] for value in spans), max(value[1] for value in spans))
        for handle, value in protected_objects.items():
            source_span = value.get("source_span") or {}
            if (
                source_span.get("character_start") == bounds[0]
                and source_span.get("character_end") == bounds[1]
                and value.get("protection_source") == "explicit_user_or_caller_span"
                and sentence[bounds[0] : bounds[1]] == element["text"]
            ):
                return {"type": "handle", "value": handle}
    return byte_literal(element["text"])


def masc_frame_arguments(
    annotation: dict[str, Any],
    *,
    protected_objects: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    arguments = [
        {
            "role": safe_symbol(element["role"], prefix="ROLE"),
            "value": masc_semantic_value(
                element,
                sentence=annotation["sentence"],
                protected_objects=protected_objects,
            ),
        }
        for element in annotation["frame_elements"]
    ]
    contextual = annotation.get("contextual_frame_ambiguity")
    if contextual is not None:
        arguments.append(
            {
                "role": "CONTEXTUAL_FRAME_ALTERNATIVES",
                "value": {
                    "type": "ambiguity",
                    "value": [
                        {
                            "value": copy.deepcopy(alternative["value"]),
                            "probability": float(alternative["probability"]),
                            "evidence": alternative["evidence"],
                        }
                        for alternative in contextual["alternatives"]
                    ],
                },
            }
        )
    return arguments


def masc_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
    interaction_annotation: dict[str, Any] | None = None,
    contextual_frame_ambiguity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    annotation = copy.deepcopy(row["annotation"])
    if contextual_frame_ambiguity is not None:
        alternatives = contextual_frame_ambiguity.get("alternatives") or []
        if annotation["frame_name"] not in {
            alternative.get("frame_name") for alternative in alternatives
        }:
            raise ValueError("selected MASC frame absent from contextual alternatives")
        annotation["contextual_frame_ambiguity"] = copy.deepcopy(
            contextual_frame_ambiguity
        )
    explicit_spans = [
        {
            "start": span["start"],
            "end": span["end"],
            "object_type": span["object_type"],
            "copy_policy": span["copy_policy"],
        }
        for span in annotation.get("protected_spans") or []
    ]
    protected = extract_protected_objects(
        annotation["sentence"],
        explicit_spans=explicit_spans,
    )
    arguments = masc_frame_arguments(
        annotation,
        protected_objects=protected["protected_objects"],
    )
    predicate = "FRAME_" + safe_symbol(annotation["frame_name"], prefix="UNKNOWN")
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": annotation["target_spans"],
                "arguments": copy.deepcopy(arguments),
            }
        ],
    }
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": copy.deepcopy(arguments),
            }
        ],
        "required_terms": [
            {
                "concept": "framenet."
                + re.sub(r"[^a-z0-9]+", "_", annotation["frame_name"].lower()).strip("_"),
                "surface_policy": "source_licensed_frame",
            }
        ],
        "required_caveats": [],
        "style": {"register": "source_authored"},
    }
    identity = row["selection_key"].split(":", 1)[1]
    segment_frame = {
        "frame_name": annotation["frame_name"],
        "lexical_unit": annotation["lexical_unit"],
        "target_spans": annotation["target_spans"],
        "frame_roles": sorted(
            {safe_symbol(element["role"], prefix="ROLE") for element in annotation["frame_elements"]}
        ),
    }
    token_tags = [
        {
            "tag": "FRAME_TARGET:" + safe_symbol(annotation["frame_name"], prefix="UNKNOWN"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for span in annotation["target_spans"]
    ]
    token_tags.extend(
        {
            "tag": "FRAME_ROLE:" + safe_symbol(element["role"], prefix="ROLE"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for element in annotation["frame_elements"]
        for span in element["spans"]
    )
    token_tags.extend(
        {
            "tag": "ENTITY:" + str(span["object_type"]),
            "source_span": [int(span["start"]), int(span["end"])],
            "authority": "licensed_manual_annotation",
        }
        for span in annotation.get("protected_spans") or []
    )
    return base_record(
        split=split,
        source_text=annotation["sentence"],
        surface_target=annotation["sentence"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="masc-frame:" + identity[:24],
        source_group="masc-document:" + annotation["document_id"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=True,
        explicit_spans=explicit_spans,
        segment_frame=segment_frame,
        token_tags=token_tags,
        interaction_annotation=interaction_annotation,
        interaction_entries=(
            [
                {
                    "segment_id": "previous_turn",
                    "key": "frame_name",
                    "value": str(interaction_annotation["frame_name"]),
                },
                {
                    "segment_id": "previous_turn",
                    "key": "lexical_unit",
                    "value": str(interaction_annotation["lexical_unit"]),
                },
            ]
            if interaction_annotation
            else []
        ),
        interaction_actor_id="masc_manual_framenet",
    )


def select_masc_composites(
    selected: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    *,
    minimum_frames: int,
    maximum_frames: int,
) -> dict[str, list[list[dict[str, Any]]]]:
    if minimum_frames < 2 or maximum_frames < minimum_frames:
        raise ValueError("MASC composite frame bounds are invalid")
    output: dict[str, list[list[dict[str, Any]]]] = {}
    for split, required in counts.items():
        by_sentence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected[split]:
            by_sentence[row["sentence_identity"]].append(row)
        groups = [
            sorted(rows, key=lambda item: item["selection_key"])
            for rows in by_sentence.values()
            if minimum_frames <= len(rows) <= maximum_frames
        ]
        groups.sort(
            key=lambda rows: stable_hash(
                [row["selection_key"] for row in rows]
            )
        )
        if len(groups) < int(required):
            raise ValueError(
                f"insufficient MASC composite rows: {split}:{len(groups)}:{required}"
            )
        output[split] = groups[: int(required)]
    return output


def masc_composite_record(
    rows: list[dict[str, Any]],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
    frame_priors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    annotations = [masc_annotation_with_prior(row, frame_priors) for row in rows]
    sentence = annotations[0]["sentence"]
    document_id = annotations[0]["document_id"]
    sentence_node = annotations[0]["sentence_node"]
    if any(
        annotation["sentence"] != sentence
        or annotation["document_id"] != document_id
        or annotation["sentence_node"] != sentence_node
        for annotation in annotations
    ):
        raise ValueError("MASC composite rows do not share one source sentence")
    protected_spans = annotations[0].get("protected_spans") or []
    if any((annotation.get("protected_spans") or []) != protected_spans for annotation in annotations[1:]):
        raise ValueError("MASC composite protected spans disagree")
    explicit_spans = [
        {
            "start": span["start"],
            "end": span["end"],
            "object_type": span["object_type"],
            "copy_policy": span["copy_policy"],
        }
        for span in protected_spans
    ]
    protected = extract_protected_objects(sentence, explicit_spans=explicit_spans)
    nodes: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    token_tags: list[dict[str, Any]] = []
    frame_receipts: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations):
        node_id = f"k{index}"
        claim_id = f"claim-{index + 1}"
        predicate = "FRAME_" + safe_symbol(annotation["frame_name"], prefix="UNKNOWN")
        arguments = masc_frame_arguments(
            annotation,
            protected_objects=protected["protected_objects"],
        )
        common = {
            "predicate": predicate,
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": copy.deepcopy(arguments),
        }
        nodes.append(
            {
                "node_id": node_id,
                "operator": common["predicate"],
                "modality": common["modality"],
                "polarity": common["polarity"],
                "quantifier": common["quantifier"],
                "confidence": common["confidence"],
                "derivation": "preserved",
                "source_spans": annotation["target_spans"],
                "arguments": common["arguments"],
            }
        )
        claims.append({"claim_id": claim_id, **common})
        frame_receipts.append(
            {
                "node_id": node_id,
                "claim_id": claim_id,
                "annotation_set_node": annotation["annotation_set_node"],
                "annotation_set_id": annotation["annotation_set_id"],
                "frame_name": annotation["frame_name"],
                "lexical_unit": annotation["lexical_unit"],
                "target_spans": annotation["target_spans"],
                "frame_roles": sorted(
                    {
                        safe_symbol(element["role"], prefix="ROLE")
                        for element in annotation["frame_elements"]
                    }
                ),
                "source_annotation_sha256": stable_hash(annotation),
            }
        )
        token_tags.extend(
            {
                "tag": "FRAME_TARGET:" + safe_symbol(annotation["frame_name"], prefix="UNKNOWN"),
                "source_span": list(span),
                "authority": "licensed_manual_annotation",
            }
            for span in annotation["target_spans"]
        )
        token_tags.extend(
            {
                "tag": "FRAME_ROLE:" + safe_symbol(element["role"], prefix="ROLE"),
                "source_span": list(span),
                "authority": "licensed_manual_annotation",
            }
            for element in annotation["frame_elements"]
            for span in element["spans"]
        )
    token_tags.extend(
        {
            "tag": "ENTITY:" + str(span["object_type"]),
            "source_span": [int(span["start"]), int(span["end"])],
            "authority": "licensed_manual_annotation",
        }
        for span in protected_spans
    )
    source_ids = ["masc-frame:" + row["selection_key"].split(":", 1)[1][:24] for row in rows]
    composite_identity = stable_hash(source_ids).split(":", 1)[1]
    source_annotation = {
        "source_kind": "masc_manual_framenet_composite",
        "document_id": document_id,
        "sentence_node": sentence_node,
        "sentence": sentence,
        "component_source_ids": source_ids,
        "frames": annotations,
        "frame_receipts": frame_receipts,
        "semantic_claim_scope": source["composite_semantic_claim_scope"],
        "complete_sentence_semantics_claimed": False,
        "inter_frame_discourse_edges_claimed": False,
    }
    record = base_record(
        split=split,
        source_text=sentence,
        surface_target=sentence,
        program={"roots": [node["node_id"] for node in nodes], "nodes": nodes},
        answer_packet={
            "claims": claims,
            "required_terms": [
                {
                    "concept": frame_concept(annotation["frame_name"]),
                    "surface_policy": "source_licensed_frame",
                }
                for annotation in annotations
            ],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=source,
        source_id="masc-composite:" + composite_identity[:24],
        source_group="masc-document:" + document_id,
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=source_annotation,
        exact_residual=True,
        explicit_spans=explicit_spans,
        segment_frame={
            "schema": "framenet_composite_v1",
            "frames": frame_receipts,
        },
        token_tags=token_tags,
    )
    record["semantic_supervision"]["unique_source_credit"] = int(
        source["composite_semantic_unique_source_credit"]
    )
    record["semantic_supervision"]["composite_semantic_authority"] = (
        "manual_framenet_multiple_annotations_same_source_sentence"
    )
    return record


def semantic_concept(namespace: str, value: str) -> dict[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".") or "unknown"
    return {"type": "concept", "value": f"{namespace}.{normalized}"}


def masc_decision_value(layer: str, field: str, value: str) -> dict[str, Any]:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return {"type": "boolean", "value": lowered == "true"}
    if field == "nested-source":
        return {
            "type": "list",
            "value": [semantic_concept("mpqa.source", part) for part in value.split(",") if part.strip()],
        }
    if field in {
        "annotation-uncertain", "attitude-type", "attitude-uncertain", "contrast",
        "es-uncertain", "expression-intensity", "implicit", "inferred", "insubstantial",
        "intensity", "polarity", "repetition", "sarcastic", "subjective-uncertain",
        "target-uncertain",
    }:
        return semantic_concept(f"{layer}.{safe_symbol(field, prefix='field').lower()}", value)
    return byte_literal(value)


def masc_decision_modality(annotation: dict[str, Any]) -> str:
    label = str(annotation["label"])
    if annotation["layer"] == "cb":
        return "ASSERTED" if label.startswith("Committed Belief") else "POSSIBLE"
    if any("uncertain" in key and value.lower() not in {"", "false", "no"} for key, value in annotation["fields"].items()):
        return "POSSIBLE"
    return "ASSERTED"


def masc_decision_polarity(annotation: dict[str, Any]) -> str:
    label = str(annotation["label"])
    explicit = str(annotation["fields"].get("polarity") or "").lower()
    if label.lower().startswith("not ") or explicit in {"negative", "neg", "both"}:
        return "NEGATED"
    if explicit in {"positive", "pos"}:
        return "AFFIRMED"
    return "UNKNOWN" if annotation["layer"] == "mpqa" else "AFFIRMED"


def masc_decision_arguments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    arguments = [
        {"role": "ANNOTATION_KIND", "value": semantic_concept(annotation["layer"], annotation["label"])},
        {"role": "SOURCE_EXPRESSION", "value": byte_literal(annotation["text"])},
    ]
    if annotation["layer"] == "cb":
        arguments.append(
            {
                "role": "TEMPORAL_ORIENTATION",
                "value": semantic_concept(
                    "time", "future" if annotation["label"].endswith("Future") else "non_future"
                ),
            }
        )
    for field, value in sorted(annotation["fields"].items()):
        if field == "id" or not value:
            continue
        arguments.append(
            {
                "role": safe_symbol(field, prefix="FIELD"),
                "value": masc_decision_value(annotation["layer"], field, value),
            }
        )
    return arguments


def select_masc_decision_semantics(
    candidates: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    *,
    minimum_annotations: int,
    maximum_annotations: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, required in counts.items():
        eligible = [
            row for row in candidates.get(split, ())
            if minimum_annotations <= len(row["annotation"]["annotations"]) <= maximum_annotations
        ]
        if len(eligible) < int(required):
            raise ValueError(
                f"insufficient MASC decision-semantic rows: {split}:{len(eligible)}:{required}"
            )
        output[split] = eligible[: int(required)]
    return output


def masc_decision_semantic_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    annotation = copy.deepcopy(row["annotation"])
    nodes: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    token_tags: list[dict[str, Any]] = []
    for index, item in enumerate(annotation["annotations"]):
        predicate = (
            "EPISTEMIC_STATUS" if item["layer"] == "cb" else
            "EVENT_" + safe_symbol(item["label"], prefix="UNKNOWN") if item["layer"] == "event" else
            "SUBJECTIVITY_" + safe_symbol(item["label"], prefix="UNKNOWN")
        )
        common = {
            "predicate": predicate,
            "modality": masc_decision_modality(item),
            "polarity": masc_decision_polarity(item),
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": masc_decision_arguments(item),
        }
        nodes.append(
            {
                "node_id": f"k{index}",
                "operator": predicate,
                "modality": common["modality"],
                "polarity": common["polarity"],
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[int(item["start"]), int(item["end"])]],
                "arguments": copy.deepcopy(common["arguments"]),
            }
        )
        claims.append({"claim_id": f"claim-{index + 1}", **common})
        token_tags.append(
            {
                "tag": f"{item['layer'].upper()}:{safe_symbol(item['label'], prefix='UNKNOWN')}",
                "source_span": [int(item["start"]), int(item["end"])],
                "authority": "licensed_manual_annotation",
            }
        )
    annotation.update(
        {
            "semantic_claim_scope": source["decision_semantic_claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "event_coreference_grouping_claimed": False,
            "source_declared_cross_annotation_links_resolved": False,
        }
    )
    identity = row["selection_key"].split(":", 1)[1]
    record = base_record(
        split=split,
        source_text=annotation["sentence"],
        surface_target=annotation["sentence"],
        program={"roots": [node["node_id"] for node in nodes], "nodes": nodes},
        answer_packet={
            "claims": claims,
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=source,
        source_id="masc-decision:" + identity[:24],
        source_group="masc-document:" + annotation["document_id"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=True,
        token_tags=token_tags,
    )
    record["semantic_supervision"]["unique_source_credit"] = int(
        source["decision_semantic_unique_source_credit"]
    )
    record["semantic_supervision"]["decision_semantic_authority"] = (
        "manual_masc_cb_mpqa_event_annotations"
    )
    return record


def event_type_concept(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return "masc.event_type." + (normalized or "unknown")


def masc_event_coreference_record(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    contract = source["event_coreference"]
    annotation = copy.deepcopy(row["annotation"])
    annotation["semantic_claim_scope"] = contract["claim_scope"]
    members: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    token_tags: list[dict[str, Any]] = []
    segment_mentions: list[dict[str, Any]] = []
    for index, mention in enumerate(annotation["mentions"]):
        node_id = f"k{index}"
        claim_id = f"claim-{index + 1}"
        event_type = event_type_concept(mention["event_type"])
        arguments = [
            {
                "role": "EVENT_TYPE",
                "value": {"type": "concept", "value": event_type},
            },
            {
                "role": "GROUP_ID",
                "value": {
                    "type": "concept",
                    "value": annotation["group_concept"],
                },
            },
        ]
        common = {
            "predicate": "EVENT_MENTION",
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": copy.deepcopy(arguments),
        }
        members.append(
            {
                "node_id": node_id,
                "operator": "EVENT_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": copy.deepcopy(mention["target_spans"]),
                "arguments": arguments,
            }
        )
        claims.append({"claim_id": claim_id, **common})
        segment_mentions.append(
            {
                "node_id": node_id,
                "claim_id": claim_id,
                "event_type": event_type,
                "target_spans": copy.deepcopy(mention["target_spans"]),
                "source_annotation_sha256": mention[
                    "source_annotation_sha256"
                ],
            }
        )
        for target_span in mention["target_spans"]:
            token_tags.append(
                {
                    "tag": "EVENT_COREFERENCE:"
                    + safe_symbol(annotation["annotation_set_name"], prefix="GROUP"),
                    "source_span": copy.deepcopy(target_span),
                    "authority": "licensed_manual_annotation",
                }
            )
    group_node_id = f"k{len(members)}"
    group_claim_id = f"claim-{len(claims) + 1}"
    all_spans = sorted(
        {
            tuple(span)
            for mention in annotation["mentions"]
            for span in mention["target_spans"]
        }
    )
    group_arguments = [
        {
            "role": "GROUP_ID",
            "value": {"type": "concept", "value": annotation["group_concept"]},
        },
        {
            "role": "MEMBERS",
            "value": {
                "type": "list",
                "value": [
                    {"type": "node_ref", "value": member["node_id"]}
                    for member in members
                ],
            },
        },
    ]
    members.append(
        {
            "node_id": group_node_id,
            "operator": "EVENT_COREFERENCE_GROUP",
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "derivation": "preserved",
            "source_spans": [list(span) for span in all_spans],
            "arguments": group_arguments,
        }
    )
    claims.append(
        {
            "claim_id": group_claim_id,
            "predicate": "EVENT_COREFERENCE_GROUP",
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": [
                {
                    "role": "GROUP_ID",
                    "value": {
                        "type": "concept",
                        "value": annotation["group_concept"],
                    },
                },
                {
                    "role": "MEMBER_EVENT_TYPES",
                    "value": {
                        "type": "list",
                        "value": [
                            {
                                "type": "concept",
                                "value": event_type_concept(mention["event_type"]),
                            }
                            for mention in annotation["mentions"]
                        ],
                    },
                },
                {
                    "role": "MENTION_COUNT",
                    "value": {
                        "type": "number",
                        "value": {
                            "value": len(annotation["mentions"]),
                            "unit": "event_mentions",
                            "precision": "exact",
                        },
                    },
                },
            ],
        }
    )
    record = base_record(
        split=row["split"],
        source_text=annotation["source_text"],
        surface_target=annotation["source_text"],
        program={"roots": [group_node_id], "nodes": members},
        answer_packet={
            "claims": claims,
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=source,
        source_id=row["source_id"],
        source_group="masc-document:" + annotation["document_id"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=True,
        segment_frame={
            "schema": "event_coreference_group_v1",
            "group_id": annotation["group_concept"],
            "group_node_id": group_node_id,
            "group_claim_id": group_claim_id,
            "mentions": segment_mentions,
        },
        token_tags=token_tags,
    )
    record["semantic_supervision"]["unique_source_credit"] = int(
        contract["unique_source_credit"]
    )
    record["semantic_supervision"]["event_coreference_authority"] = (
        "manual_named_gate_annotation_set_membership"
    )
    return record


def mpqa_member_concept(member_type: str, member: dict[str, Any]) -> str:
    identity = str(member.get("annotation_id") or member.get("annotation_line_id") or "")
    fragment = re.sub(r"[^a-z0-9]+", "_", identity.casefold()).strip("_")
    if member_type == "source" and identity == "w":
        return "mpqa.source.w"
    digest = str(member["source_annotation_sha256"]).split(":", 1)[1][:12]
    return f"mpqa.{member_type}.{(fragment or 'unnamed')[:48]}.{digest}"


def mpqa_member_span_status(member: dict[str, Any]) -> str:
    if member["target_spans"]:
        return "explicit"
    fields = member.get("fields") or {}
    if (
        str(fields.get("implicit") or "").casefold() == "true"
        or str(member.get("annotation_id") or "").casefold() in {"w", "implicit"}
        or member.get("node_type") == "implicit-writer"
    ):
        return "declared_implicit"
    return "zero_width_annotation"


def masc_mpqa_relation_record(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    contract = source["mpqa_relations"]
    annotation = copy.deepcopy(row["annotation"])
    annotation.update(
        {
            "semantic_claim_scope": contract["claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "causal_relation_claimed": False,
            "temporal_relation_claimed": False,
            "inferred_relation_count": 0,
        }
    )
    typed_members: list[tuple[str, dict[str, Any]]] = [
        ("expression", annotation["expression"]),
        *(("source", member) for member in annotation["source_chain"]),
    ]
    for attitude in annotation["attitudes"]:
        typed_members.append(("attitude", attitude))
        typed_members.extend(("target", target) for target in attitude["targets"])
    unique_members: list[tuple[str, dict[str, Any]]] = []
    seen_receipts: set[str] = set()
    for member_type, member in typed_members:
        receipt = str(member["source_annotation_sha256"])
        if receipt in seen_receipts:
            continue
        seen_receipts.add(receipt)
        unique_members.append((member_type, member))
    receipt_to_node = {
        str(member["source_annotation_sha256"]): f"k{index}"
        for index, (_, member) in enumerate(unique_members)
    }
    concepts = {
        str(member["source_annotation_sha256"]): mpqa_member_concept(
            member_type, member
        )
        for member_type, member in unique_members
    }
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    segment_edges: list[dict[str, Any]] = []
    for edge in annotation["edges"]:
        from_node = receipt_to_node[str(edge["from"])]
        to_node = receipt_to_node[str(edge["to"])]
        order = int(edge.get("order", -1))
        normalized = {
            "edge_type": str(edge["edge_type"]),
            "from_node_id": from_node,
            "to_node_id": to_node,
            "order": order,
            "source_field": str(edge["manual_field"]),
        }
        segment_edges.append(normalized)
        outgoing[from_node].append(normalized)
    nodes: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    segment_members: list[dict[str, Any]] = []
    token_tags: list[dict[str, Any]] = []
    for index, (member_type, member) in enumerate(unique_members):
        node_id = f"k{index}"
        claim_id = f"claim-{index + 1}"
        receipt = str(member["source_annotation_sha256"])
        concept_id = concepts[receipt]
        span_status = mpqa_member_span_status(member)
        arguments: list[dict[str, Any]] = [
            {
                "role": "MEMBER_CONCEPT",
                "value": {"type": "concept", "value": concept_id},
            },
            {
                "role": "RELATION_ID",
                "value": {
                    "type": "concept",
                    "value": annotation["relation_concept"],
                },
            },
            {
                "role": "SPAN_STATUS",
                "value": {
                    "type": "concept",
                    "value": "mpqa.span_status." + span_status,
                },
            },
        ]
        for edge in sorted(outgoing.get(node_id, []), key=canonical_json):
            role = "LINK_" + safe_symbol(edge["edge_type"], prefix="EDGE")
            if edge["edge_type"] == "nested_source_member":
                role += "_" + str(edge["order"])
            arguments.append(
                {
                    "role": role,
                    "value": {"type": "node_ref", "value": edge["to_node_id"]},
                }
            )
        predicate = "MPQA_" + member_type.upper()
        common = {
            "predicate": predicate,
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
        }
        nodes.append(
            {
                "node_id": node_id,
                "operator": predicate,
                **{key: value for key, value in common.items() if key != "predicate"},
                "derivation": "preserved",
                "source_spans": copy.deepcopy(member["target_spans"]),
                "arguments": arguments,
            }
        )
        claim_arguments = [
            {
                "role": "MEMBER_CONCEPT",
                "value": {"type": "concept", "value": concept_id},
            },
            {
                "role": "RELATION_ID",
                "value": {
                    "type": "concept",
                    "value": annotation["relation_concept"],
                },
            },
            {
                "role": "MEMBER_TYPE",
                "value": {
                    "type": "concept",
                    "value": "mpqa.member_type." + member_type,
                },
            },
            {
                "role": "SPAN_STATUS",
                "value": {
                    "type": "concept",
                    "value": "mpqa.span_status." + span_status,
                },
            },
        ]
        claims.append({"claim_id": claim_id, **common, "arguments": claim_arguments})
        segment_members.append(
            {
                "node_id": node_id,
                "claim_id": claim_id,
                "member_type": member_type,
                "concept_id": concept_id,
                "target_spans": copy.deepcopy(member["target_spans"]),
                "source_annotation_sha256": receipt,
                "implicit": span_status == "declared_implicit",
                "span_status": span_status,
            }
        )
        for target_span in member["target_spans"]:
            token_tags.append(
                {
                    "tag": "MPQA_RELATION_" + member_type.upper(),
                    "source_span": copy.deepcopy(target_span),
                    "authority": "licensed_manual_annotation",
                }
            )
    expression_receipt = str(annotation["expression"]["source_annotation_sha256"])
    expression_node = receipt_to_node[expression_receipt]
    record = base_record(
        split=row["split"],
        source_text=annotation["source_text"],
        surface_target=annotation["source_text"],
        program={"roots": [expression_node], "nodes": nodes},
        answer_packet={
            "claims": claims,
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=source,
        source_id=row["source_id"],
        source_group="masc-document:" + annotation["document_id"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=True,
        segment_frame={
            "schema": "mpqa_relation_chain_v1",
            "relation_id": annotation["relation_concept"],
            "members": segment_members,
            "edges": segment_edges,
        },
        token_tags=token_tags,
    )
    record["semantic_supervision"]["unique_source_credit"] = int(
        contract["unique_source_credit"]
    )
    record["semantic_supervision"]["mpqa_relation_authority"] = (
        "manual_complete_expression_attitude_target_source_links"
    )
    return record


def gum_source_grounded_scope_projection(
    annotation: dict[str, Any],
    *,
    document_concept: str,
) -> dict[str, Any]:
    """Project only unambiguous human eRST edges into the scoped ABI."""

    relation = str(annotation["primary_relation"])
    contract = GUM_SCOPED_RELATION_CONTRACT.get(relation)
    common = {
        "policy": GUM_SCOPED_SEMANTIC_SUPERVISION_POLICY,
        "scoped_graph_policy": KERC_SCOPED_SEMANTIC_POLICY,
        "source_relation": relation,
        "edge_count": len(annotation["edges"]),
        "authority": "human_erst_primary_relation_direction_and_endpoint_spans",
        "complete_sentence_semantics_claimed": False,
        "truth_claimed": False,
        "learned_competence_claimed": False,
        "derived_view_unique_source_credit": 0,
    }
    if contract is None:
        return {
            **common,
            "disposition": "EXCLUDED",
            "exclusion_reason": "relation_outside_conservative_scope_contract",
            "operator": None,
            "target_roles": [],
        }
    edges = annotation["edges"]
    if len(edges) != 1 or str(edges[0]["edge_kind"]) != "primary":
        return {
            **common,
            "disposition": "EXCLUDED",
            "exclusion_reason": "multi_edge_or_nonprimary_neighborhood_has_shared_endpoint_ownership",
            "operator": str(contract["operator"]),
            "target_roles": [],
        }

    units = sorted(annotation["units"], key=lambda unit: int(unit["edu_id"]))
    if len(units) != 2:
        return {
            **common,
            "disposition": "EXCLUDED",
            "exclusion_reason": "single_primary_edge_does_not_project_exactly_two_units",
            "operator": str(contract["operator"]),
            "target_roles": [],
        }
    edge = edges[0]
    unit_by_id = {int(unit["edu_id"]): unit for unit in units}
    child = unit_by_id.get(int(edge["child_edu_id"]))
    parent = unit_by_id.get(int(edge["parent_edu_id"]))
    if child is None or parent is None or child is parent:
        return {
            **common,
            "disposition": "EXCLUDED",
            "exclusion_reason": "primary_edge_endpoint_projection_invalid",
            "operator": str(contract["operator"]),
            "target_roles": [],
        }
    proposition_id = {
        int(unit["edu_id"]): f"p{index}" for index, unit in enumerate(units)
    }
    if "roles" in contract:
        role_units = [
            (str(contract["roles"]["child"]), child),
            (str(contract["roles"]["parent"]), parent),
        ]
    else:
        ordered = sorted(
            (child, parent),
            key=lambda unit: (int(unit["excerpt_span"][0]), int(unit["edu_id"])),
        )
        role_units = list(zip(contract["ordered_roles"], ordered))

    propositions = []
    for unit in units:
        propositions.append(
            {
                "proposition_id": proposition_id[int(unit["edu_id"])],
                "predicate": "DISCOURSE_UNIT",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [copy.deepcopy(unit["excerpt_span"])],
                "arguments": [
                    {
                        "role": "DOCUMENT",
                        "value": {"type": "concept", "value": document_concept},
                    },
                    {
                        "role": "EDU_ID",
                        "value": {
                            "type": "number",
                            "value": {
                                "value": int(unit["edu_id"]),
                                "unit": "identifier",
                            },
                        },
                    },
                ],
            }
        )
    source_arguments = [
        {
            "role": "SOURCE_RELATION",
            "value": {
                "type": "concept",
                "value": "erst.relation."
                + relation.removesuffix("_m").removesuffix("_r").replace("-", "."),
            },
        },
        {
            "role": "NUCLEARITY",
            "value": {
                "type": "concept",
                "value": "erst.nuclearity."
                + ("multinuclear" if relation.endswith("_m") else "satellite_nucleus"),
            },
        },
        {
            "role": "EDGE_KIND",
            "value": {"type": "concept", "value": "erst.edge_kind.primary"},
        },
    ]
    targets = [
        {"role": role, "target_id": proposition_id[int(unit["edu_id"])]}
        for role, unit in role_units
    ]
    def endpoint_concept(unit: dict[str, Any]) -> str:
        return f"erst.edu.{annotation['document_id'].lower()}.{int(unit['edu_id'])}"
    answer_arguments = [
        {"role": role, "value": {"type": "concept", "value": endpoint_concept(unit)}}
        for role, unit in role_units
    ] + copy.deepcopy(source_arguments)
    scope_spans = sorted(copy.deepcopy(unit["excerpt_span"]) for unit in units)
    operator = str(contract["operator"])
    return {
        **common,
        "disposition": "ADMITTED",
        "exclusion_reason": None,
        "operator": operator,
        "target_roles": [role for role, _unit in role_units],
        "graph": {
            "policy": KERC_SCOPED_SEMANTIC_POLICY,
            "roots": ["s0"],
            "scopes": [
                {
                    "scope_id": "s0",
                    "operator": operator,
                    "targets": targets,
                    "arguments": source_arguments,
                    "source_spans": scope_spans,
                }
            ],
            "propositions": propositions,
        },
        "answer_claims": [
            {
                "claim_id": "claim-1",
                "predicate": "SCOPE_" + operator,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": answer_arguments,
            }
        ],
    }


def gum_scoped_supervision_audit(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    relations: dict[str, Counter[str]] = defaultdict(Counter)
    operators: dict[str, Counter[str]] = defaultdict(Counter)
    genres: dict[str, Counter[str]] = defaultdict(Counter)
    depths: dict[str, Counter[str]] = defaultdict(Counter)
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)
    source_groups: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            annotation = row["annotation"]
            projection = gum_source_grounded_scope_projection(
                annotation,
                document_concept="gum.document.audit",
            )
            if projection["disposition"] != "ADMITTED":
                if projection["operator"] is not None:
                    exclusions[split][str(projection["exclusion_reason"])] += 1
                continue
            relation = str(annotation["primary_relation"])
            relations[split][relation] += 1
            operators[split][str(projection["operator"])] += 1
            genres[split][f"{relation}:{annotation['genre']}"] += 1
            units = {int(unit["edu_id"]): unit for unit in annotation["units"]}
            edge = annotation["edges"][0]
            depth = max(
                int(units[int(edge["child_edu_id"])]["tree_depth"]),
                int(units[int(edge["parent_edu_id"])]["tree_depth"]),
            )
            depths[split][f"{relation}:{min(depth, 5)}"] += 1
            source_groups[split].add(str(row["source_group"]))
    splits = ("private_train", "private_dev", "private_eval")
    relation_names = sorted(GUM_SCOPED_RELATION_CONTRACT)
    genre_names = ("academic", "bio", "court", "interview", "news", "voyage")
    missing_cells = {
        split: [
            f"{relation}:{genre}"
            for relation in relation_names
            for genre in genre_names
            if genres[split][f"{relation}:{genre}"] == 0
        ]
        for split in splits
    }
    groups = [source_groups[split] for split in splits]
    return {
        "policy": GUM_SCOPED_SEMANTIC_SUPERVISION_POLICY,
        "admitted_relation_contract": GUM_SCOPED_RELATION_DESCRIPTIONS,
        "record_counts_by_split_and_relation": {
            split: dict(sorted(relations[split].items())) for split in splits
        },
        "record_counts_by_split": {
            split: sum(relations[split].values()) for split in splits
        },
        "operator_counts_by_split": {
            split: dict(sorted(operators[split].items())) for split in splits
        },
        "relation_genre_counts_by_split": {
            split: dict(sorted(genres[split].items())) for split in splits
        },
        "relation_depth_counts_by_split": {
            split: dict(sorted(depths[split].items())) for split in splits
        },
        "missing_relation_genre_cells_by_split": missing_cells,
        "excluded_mapped_record_counts_by_split_and_reason": {
            split: dict(sorted(exclusions[split].items())) for split in splits
        },
        "excluded_mapped_record_count": sum(
            sum(exclusions[split].values()) for split in splits
        ),
        "source_group_counts_by_split": {
            split: len(source_groups[split]) for split in splits
        },
        "cross_split_source_group_overlap_count": sum(
            len(groups[left] & groups[right])
            for left in range(len(groups))
            for right in range(left + 1, len(groups))
        ),
        "minimum_relation_count_by_split": {
            split: min(relations[split].values()) for split in splits
        },
        "unique_source_credit": 0,
        "complete_sentence_semantics_claimed": False,
        "truth_claimed": False,
        "learned_competence_claimed": False,
    }


def gum_discourse_record(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    """Compile one human eRST edge neighborhood into typed Kernel topology."""

    annotation = copy.deepcopy(row["annotation"])
    units = sorted(annotation["units"], key=lambda unit: int(unit["edu_id"]))
    unit_node_by_id = {
        int(unit["edu_id"]): f"k{index}" for index, unit in enumerate(units)
    }
    nodes: list[dict[str, Any]] = []
    segment_units: list[dict[str, Any]] = []
    token_tags: list[dict[str, Any]] = []
    document_concept = "gum.document." + re.sub(
        r"[^a-z0-9]+", ".", annotation["document_id"].lower()
    ).strip(".")
    for index, unit in enumerate(units):
        node_id = f"k{index}"
        span = copy.deepcopy(unit["excerpt_span"])
        nodes.append(
            {
                "node_id": node_id,
                "operator": "DISCOURSE_UNIT",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [span],
                "arguments": [
                    {
                        "role": "DOCUMENT",
                        "value": {"type": "concept", "value": document_concept},
                    },
                    {
                        "role": "EDU_ID",
                        "value": {
                            "type": "number",
                            "value": {"value": int(unit["edu_id"]), "unit": "identifier"},
                        },
                    },
                ],
            }
        )
        segment_units.append(
            {
                "edu_id": int(unit["edu_id"]),
                "node_id": node_id,
                "target_spans": [span],
                "tree_depth": int(unit["tree_depth"]),
                "source_row_sha256": str(unit["source_row_sha256"]),
            }
        )
        token_tags.append(
            {
                "tag": "ERST_DISCOURSE_UNIT",
                "source_span": span,
                "authority": "licensed_manual_annotation",
            }
        )
    claims: list[dict[str, Any]] = []
    segment_edges: list[dict[str, Any]] = []
    roots: list[str] = []
    for index, edge in enumerate(annotation["edges"]):
        node_id = f"k{len(units) + index}"
        roots.append(node_id)
        child_node = unit_node_by_id[int(edge["child_edu_id"])]
        parent_node = unit_node_by_id[int(edge["parent_edu_id"])]
        relation = str(edge["relation"])
        relation_base = relation.removesuffix("_m").removesuffix("_r")
        nuclearity = (
            "multinuclear"
            if relation.endswith("_m")
            else "satellite_nucleus"
            if relation.endswith("_r")
            else "secondary_unspecified"
        )
        predicate = "ERST_" + safe_symbol(relation_base, prefix="RELATION")
        program_arguments = [
            {"role": "CHILD", "value": {"type": "node_ref", "value": child_node}},
            {"role": "PARENT", "value": {"type": "node_ref", "value": parent_node}},
            {
                "role": "RELATION",
                "value": {
                    "type": "concept",
                    "value": "erst.relation." + relation_base.replace("-", "."),
                },
            },
            {
                "role": "NUCLEARITY",
                "value": {"type": "concept", "value": "erst.nuclearity." + nuclearity},
            },
            {
                "role": "EDGE_KIND",
                "value": {
                    "type": "concept",
                    "value": "erst.edge_kind." + str(edge["edge_kind"]),
                },
            },
        ]
        answer_arguments = copy.deepcopy(program_arguments)
        endpoint_concepts = {
            "CHILD": (
                f"erst.edu.{annotation['document_id'].lower()}."
                f"{int(edge['child_edu_id'])}"
            ),
            "PARENT": (
                f"erst.edu.{annotation['document_id'].lower()}."
                f"{int(edge['parent_edu_id'])}"
            ),
        }
        for argument in answer_arguments:
            role = str(argument["role"])
            if role in endpoint_concepts:
                argument["value"] = {
                    "type": "concept",
                    "value": endpoint_concepts[role],
                }
        spans = sorted(
            [
                copy.deepcopy(
                    next(
                        unit["excerpt_span"]
                        for unit in units
                        if int(unit["edu_id"]) == int(edge["child_edu_id"])
                    )
                ),
                copy.deepcopy(
                    next(
                        unit["excerpt_span"]
                        for unit in units
                        if int(unit["edu_id"]) == int(edge["parent_edu_id"])
                    )
                ),
            ]
        )
        nodes.append(
            {
                "node_id": node_id,
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": spans,
                "arguments": copy.deepcopy(program_arguments),
            }
        )
        claims.append(
            {
                "claim_id": f"claim-{index + 1}",
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": copy.deepcopy(answer_arguments),
            }
        )
        payload = str(edge["raw_signal_payload"])
        signal_count = 0 if not payload else payload.count(";") + 1
        segment_edges.append(
            {
                "edge_id": f"edge-{index}",
                "edge_order": int(edge["edge_order"]),
                "node_id": node_id,
                "edge_kind": str(edge["edge_kind"]),
                "child_node_id": child_node,
                "parent_node_id": parent_node,
                "relation": relation,
                "nuclearity": nuclearity,
                "source_annotation_sha256": str(edge["source_annotation_sha256"]),
                "signal_count": signal_count,
            }
        )
        child_span = next(
            unit["excerpt_span"]
            for unit in units
            if int(unit["edu_id"]) == int(edge["child_edu_id"])
        )
        token_tags.append(
            {
                "tag": "ERST_RELATION:" + safe_symbol(relation_base, prefix="RELATION"),
                "source_span": copy.deepcopy(child_span),
                "authority": "licensed_manual_annotation",
            }
        )
    scoped_projection = gum_source_grounded_scope_projection(
        annotation,
        document_concept=document_concept,
    )
    if scoped_projection["disposition"] == "ADMITTED":
        compiled_scope = compile_scoped_semantic_graph(
            scoped_projection["graph"],
            protected_objects={},
            concept_capsules={},
            source_length=len(row["source_text"]),
        )
        nodes = compiled_scope["program"]["nodes"]
        roots = compiled_scope["program"]["roots"]
        claims = scoped_projection["answer_claims"]

    row_source = {**source, "license_spdx": row["license_spdx"]}
    record = base_record(
        split=row["split"],
        source_text=row["source_text"],
        surface_target=row["source_text"],
        program={"roots": roots, "nodes": nodes},
        answer_packet={
            "claims": claims,
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=row_source,
        source_id=row["source_id"],
        source_group=row["source_group"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=False,
        segment_frame={
            "schema": "erst_discourse_graph_v1",
            "document_id": annotation["document_id"],
            "anchor_edu_id": int(annotation["anchor_edu_id"]),
            "units": segment_units,
            "edges": segment_edges,
        },
        token_tags=token_tags,
    )
    record["semantic_supervision"]["source_credit_unit"] = "document"
    record["semantic_supervision"]["derived_view_unique_source_credit"] = 0
    record["semantic_supervision"]["erst_relation_authority"] = (
        "human_source_declared_primary_and_secondary_discourse_edges"
    )
    record["semantic_supervision"]["scoped_semantic_projection"] = {
        key: copy.deepcopy(value)
        for key, value in scoped_projection.items()
        if key not in {"graph", "answer_claims"}
    }
    return record


def gum_entity_coreference_record(
    row: dict[str, Any],
    *,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    """Compile one complete human entity/coreference neighborhood."""

    annotation = copy.deepcopy(row["annotation"])
    groups = annotation["groups"]
    mentions = annotation["mentions"]
    group_by_mention = {
        mention_id: group
        for group in groups
        for mention_id in group["mention_ids"]
    }
    handle_by_group = {
        group["group_id"]: f"@C{index}" for index, group in enumerate(groups)
    }
    capsules = {}
    for group in groups:
        group_mentions = [
            mention for mention in mentions if mention["mention_id"] in group["mention_ids"]
        ]
        capsules[handle_by_group[group["group_id"]]] = {
            "stable_identity": group["stable_identity"],
            "provenance": {
                "dataset_revision": source["dataset_revision"],
                "document_id": annotation["document_id"],
                "source_group_id": group["group_id"],
                "source_annotation_sha256": stable_hash(
                    [mention["source_annotation_sha256"] for mention in group_mentions]
                ),
            },
            "entity_types": sorted(
                {mention["attributes"]["entity_type"] for mention in group_mentions}
            ),
            "source_identity_values": sorted(
                {
                    mention["attributes"]["identity"]
                    for mention in group_mentions
                    if mention["attributes"]["identity"] != "_"
                }
            ),
            "mention_count": len(group_mentions),
        }
    nodes = []
    claims = []
    token_tags = []
    node_by_mention = {}
    for index, mention in enumerate(mentions):
        node_id = f"k{index}"
        node_by_mention[mention["mention_id"]] = node_id
        group = group_by_mention[mention["mention_id"]]
        handle = handle_by_group[group["group_id"]]
        attributes = mention["attributes"]
        arguments = [
            {"role": "IDENTITY", "value": {"type": "handle", "value": handle}},
            {
                "role": "ENTITY_TYPE",
                "value": {
                    "type": "concept",
                    "value": "gum.entity_type."
                    + re.sub(r"[^a-z0-9]+", ".", attributes["entity_type"].lower()).strip("."),
                },
            },
            {
                "role": "INFORMATION_STATUS",
                "value": {
                    "type": "concept",
                    "value": "gum.information_status."
                    + re.sub(r"[^a-z0-9]+", ".", attributes["information_status"].lower()).strip("."),
                },
            },
            {
                "role": "CENTERING",
                "value": {
                    "type": "concept",
                    "value": "gum.centering."
                    + re.sub(r"[^a-z0-9]+", ".", attributes["centering"].lower()).strip("."),
                },
            },
        ]
        common = {
            "predicate": "ENTITY_MENTION",
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": copy.deepcopy(arguments),
        }
        nodes.append(
            {
                "node_id": node_id,
                "operator": "ENTITY_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": copy.deepcopy(mention["excerpt_spans"]),
                "arguments": copy.deepcopy(arguments),
            }
        )
        claims.append({"claim_id": f"claim-{len(claims) + 1}", **common})
        for span in mention["excerpt_spans"]:
            token_tags.append(
                {
                    "tag": "ENTITY_MENTION:"
                    + safe_symbol(attributes["entity_type"], prefix="UNKNOWN"),
                    "source_span": copy.deepcopy(span),
                    "authority": "licensed_manual_annotation",
                }
            )
    roots = []
    for group in groups:
        node_id = f"k{len(nodes)}"
        roots.append(node_id)
        member_ids = group["mention_ids"]
        handle = handle_by_group[group["group_id"]]
        nodes.append(
            {
                "node_id": node_id,
                "operator": "ENTITY_IDENTITY_COMPONENT",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": sorted(
                    span
                    for mention in mentions
                    if mention["mention_id"] in member_ids
                    for span in mention["excerpt_spans"]
                ),
                "arguments": [
                    {"role": "IDENTITY", "value": {"type": "handle", "value": handle}},
                    {
                        "role": "MEMBERS",
                        "value": {
                            "type": "list",
                            "value": [
                                {
                                    "type": "node_ref",
                                    "value": node_by_mention[mention_id],
                                }
                                for mention_id in member_ids
                            ],
                        },
                    },
                ],
            }
        )
        claims.append(
            {
                "claim_id": f"claim-{len(claims) + 1}",
                "predicate": "ENTITY_IDENTITY_COMPONENT",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
                    {"role": "IDENTITY", "value": {"type": "handle", "value": handle}},
                    {
                        "role": "MENTION_COUNT",
                        "value": {
                            "type": "number",
                            "value": {
                                "value": len(member_ids),
                                "unit": "entity_mentions",
                                "precision": "exact",
                            },
                        },
                    },
                ],
            }
        )
    for relation in annotation["relations"]:
        node_id = f"k{len(nodes)}"
        roots.append(node_id)
        relation_name = safe_symbol(relation["relation_type"], prefix="RELATION")
        predicate = "ENTITY_RELATION_" + relation_name
        source_mention = relation["source_mention_id"]
        target_mention = relation["target_mention_id"]
        spans = sorted(
            copy.deepcopy(span)
            for identity in (source_mention, target_mention)
            for span in next(
                mention["excerpt_spans"]
                for mention in mentions
                if mention["mention_id"] == identity
            )
        )
        program_arguments = [
            {
                "role": "SOURCE",
                "value": {"type": "node_ref", "value": node_by_mention[source_mention]},
            },
            {
                "role": "TARGET",
                "value": {"type": "node_ref", "value": node_by_mention[target_mention]},
            },
            {
                "role": "RELATION_TYPE",
                "value": {
                    "type": "concept",
                    "value": "gum.entity_relation."
                    + re.sub(r"[^a-z0-9]+", ".", relation["relation_type"].lower()).strip("."),
                },
            },
        ]
        nodes.append(
            {
                "node_id": node_id,
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": spans,
                "arguments": program_arguments,
            }
        )
        claims.append(
            {
                "claim_id": f"claim-{len(claims) + 1}",
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
                    {
                        "role": "SOURCE_IDENTITY",
                        "value": {
                            "type": "handle",
                            "value": handle_by_group[
                                group_by_mention[source_mention]["group_id"]
                            ],
                        },
                    },
                    {
                        "role": "TARGET_IDENTITY",
                        "value": {
                            "type": "handle",
                            "value": handle_by_group[
                                group_by_mention[target_mention]["group_id"]
                            ],
                        },
                    },
                    copy.deepcopy(program_arguments[2]),
                ],
            }
        )
    row_source = {**source, "license_spdx": row["license_spdx"]}
    record = base_record(
        split=row["split"],
        source_text=row["source_text"],
        surface_target=row["source_text"],
        program={"roots": roots, "nodes": nodes},
        answer_packet={
            "claims": claims,
            "required_terms": [],
            "required_caveats": [],
            "style": {"register": "source_authored"},
        },
        source=row_source,
        source_id=row["source_id"],
        source_group=row["source_group"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=False,
        token_tags=token_tags,
        concept_capsules=capsules,
    )
    record["semantic_supervision"].update(
        {
            "source_credit_unit": "document",
            "derived_view_unique_source_credit": 0,
            "entity_coreference_authority": (
                "human_source_declared_complete_identity_components_and_bridging_edges"
            ),
            "concept_capsule_identity_authority": "human_source_declared_component",
        }
    )
    return record


def interaction_predecessors(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind a frame to the preceding distinct sentence in the same document."""

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_document[row["annotation"]["document_id"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for document_rows in by_document.values():
        by_sentence: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in document_rows:
            annotation = row["annotation"]
            by_sentence[
                (int(annotation["sentence_start"]), int(annotation["sentence_end"]))
            ].append(row)
        ordered = sorted(by_sentence.items())
        for index in range(1, len(ordered)):
            previous = min(ordered[index - 1][1], key=lambda row: row["selection_key"])[
                "annotation"
            ]
            descriptor = {
                "document_id": previous["document_id"],
                "sentence_node": previous["sentence_node"],
                "annotation_set_node": previous["annotation_set_node"],
                "frame_name": previous["frame_name"],
                "lexical_unit": previous["lexical_unit"],
                "source_annotation_sha256": stable_hash(previous),
            }
            for current in ordered[index][1]:
                output[current["selection_key"]] = descriptor
    return output


def partition_dolly(
    rows: list[dict[str, Any]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    needed = sum(int(value) for value in counts.values())
    if len(rows) < needed:
        raise ValueError(f"insufficient Dolly rows: {len(rows)} < {needed}")
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split, count in counts.items():
        output[split] = rows[offset : offset + int(count)]
        offset += int(count)
    return output


def select_masc(
    rows: dict[str, list[dict[str, Any]]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        available = rows.get(split, [])
        if len(available) < int(count):
            raise ValueError(f"insufficient MASC rows: {split}:{len(available)}:{count}")
        output[split] = available[: int(count)]
    return output


def frame_concept(frame_name: str) -> str:
    return "framenet." + re.sub(
        r"[^a-z0-9]+", "_", frame_name.lower()
    ).strip("_")


def masc_contextual_frame_priors(
    selected_by_split: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Fit lexical-frame alternatives from selected private-train rows only."""

    if (
        contract.get("policy") != MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY
        or contract.get("fit_split") != "private_train"
    ):
        raise ValueError("MASC contextual-frame ambiguity contract mismatch")
    minimum_frames = int(contract.get("minimum_distinct_frames") or 0)
    minimum_total = int(contract.get("minimum_total_occurrences") or 0)
    if minimum_frames < 2 or minimum_total < minimum_frames:
        raise ValueError("MASC contextual-frame ambiguity floors are invalid")
    counts_by_lexical_unit: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected_by_split.get("private_train", []):
        annotation = row["annotation"]
        lexical_unit = str(annotation.get("lexical_unit") or "").strip().lower()
        frame_name = str(annotation.get("frame_name") or "").strip()
        if lexical_unit and frame_name:
            counts_by_lexical_unit[lexical_unit][frame_name] += 1
    priors: dict[str, dict[str, Any]] = {}
    for lexical_unit, frame_counts in sorted(counts_by_lexical_unit.items()):
        total = sum(frame_counts.values())
        if len(frame_counts) < minimum_frames or total < minimum_total:
            continue
        ordered = sorted(frame_counts.items(), key=lambda row: (-row[1], row[0]))
        alternatives = []
        assigned_mass = 0.0
        for index, (frame_name, count) in enumerate(ordered):
            probability = (
                1.0 - assigned_mass
                if index == len(ordered) - 1
                else count / total
            )
            assigned_mass += probability
            alternatives.append(
                {
                    "frame_name": frame_name,
                    "value": {"type": "concept", "value": frame_concept(frame_name)},
                    "count": count,
                    "probability": probability,
                    "evidence": "selected_private_train_manual_framenet_frequency",
                }
            )
        payload = {
            "policy": MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY,
            "fit_split": "private_train",
            "lexical_unit": lexical_unit,
            "total_occurrences": total,
            "alternatives": alternatives,
            "claim_scope": str(contract["claim_scope"]),
        }
        payload["content_sha256"] = stable_hash(payload)
        priors[lexical_unit] = payload
    return priors


def select_oasst(
    rows: dict[str, list[dict[str, Any]]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        available = rows.get(split, [])
        if len(available) < int(count):
            raise ValueError(f"insufficient OASST2 rows: {split}:{len(available)}:{count}")
        output[split] = available[: int(count)]
    return output


def produce(
    config_path: Path,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
    bypass_run_cache: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    phase_started = run_started
    phase_runtime_ms: dict[str, int] = {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage = validate_kernel_english_config(config)
    corpus = stage["semantic_corpus_materialization"]
    if corpus["policy"] != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("semantic corpus contract mismatch")
    producer_sha256 = sha256_file(Path(__file__).resolve())
    cache_cfg = corpus["content_cache"]
    cache_enabled = bool(use_cache and cache_cfg["enabled"])
    cache_root = resolve(cache_cfg["root"])
    cache_dependencies = dependency_bindings(
        producer_cache_dependency_paths(config_path, corpus)
    )
    output_path = resolve(corpus["candidate_records_jsonl"])
    manifest_path = resolve(corpus["producer_manifest_json"])
    cache_outputs = {
        "candidate_records": output_path,
        "producer_manifest": manifest_path,
    }
    if cache_enabled and not refresh_cache and not bypass_run_cache:
        cached = load_receipt(
            cache_root,
            role=str(cache_cfg["producer_role"]),
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="producer_manifest",
        )
        if cached is not None and cached.get("trigger_state") == "GREEN":
            write_json_atomic(
                cache_root / "telemetry" / "producer_last.json",
                {
                    "policy": "project_theseus_kerc_incremental_cache_telemetry_v1",
                    "run_cache_hit": True,
                    "structural_economics": {"hits": 0, "misses": 0},
                    "candidate_records": {
                        "hits_by_family": {},
                        "misses_by_family": {},
                        "hits": 0,
                        "misses": 0,
                        "entry_count": 0,
                    },
                    "candidate_finalization": {
                        "hits": 0,
                        "misses": 0,
                        "entry_count": 0,
                    },
                    "candidate_records_sha256": cached["candidate_records"]["sha256"],
                    "storage": cache_storage_telemetry(
                        cache_root / "producer_objects.sqlite3"
                    ),
                    "claim_scope": "cache execution telemetry only; not semantic or capability evidence",
                },
            )
            return cached
    phase_runtime_ms["configuration_and_run_cache"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    dolly_rows, dolly_rejects = load_dolly_candidates(
        corpus["dolly"], maximum_characters=int(corpus["maximum_source_characters"])
    )
    dolly_grounded_rows, dolly_grounded_rejects = (
        load_dolly_grounded_question_candidates(
            corpus["dolly"],
            maximum_characters=int(corpus["maximum_source_characters"]),
        )
    )
    dolly_grounded_selected = select_dolly_grounded_questions(
        dolly_grounded_rows,
        corpus["dolly"]["grounded_question_records_by_split"],
        required_question_forms=list(
            corpus["dolly"]["grounded_question_required_forms"]
        ),
    )
    grounded_source_hashes = {
        row["annotation"]["source_row_sha256"]
        for rows in dolly_grounded_selected.values()
        for row in rows
    }
    dolly_rows = [
        row
        for row in dolly_rows
        if row["annotation"]["source_row_sha256"] not in grounded_source_hashes
    ]
    dolly_selected = partition_dolly(
        dolly_rows, corpus["dolly"]["records_by_split"]
    )
    masc_rows, masc_rejects = load_masc_candidates(
        corpus["masc"], maximum_characters=int(corpus["maximum_source_characters"])
    )
    masc_selected = select_masc(masc_rows, corpus["masc"]["records_by_split"])
    masc_frame_priors = masc_contextual_frame_priors(
        masc_selected, corpus["masc"]["contextual_frame_ambiguity"]
    )
    masc_composite_selected = select_masc_composites(
        masc_selected,
        corpus["masc"]["composite_semantic_records_by_split"],
        minimum_frames=int(corpus["masc"]["composite_semantic_minimum_frames"]),
        maximum_frames=int(corpus["masc"]["composite_semantic_maximum_frames"]),
    )
    masc_decision_candidates = load_masc_decision_semantic_candidates(
        corpus["masc"], maximum_characters=int(corpus["maximum_source_characters"])
    )
    masc_decision_selected = select_masc_decision_semantics(
        masc_decision_candidates,
        corpus["masc"]["decision_semantic_records_by_split"],
        minimum_annotations=int(corpus["masc"]["decision_semantic_minimum_annotations"]),
        maximum_annotations=int(corpus["masc"]["decision_semantic_maximum_annotations"]),
    )
    event_coreference_contract = corpus["masc"]["event_coreference"]
    masc_event_coreference_selected, masc_event_coreference_audit = (
        reconstruct_event_coreference_groups(
            original_event_root=resolve(
                event_coreference_contract["original_event_root"]
            ),
            data_root=resolve(corpus["masc"]["extracted_root"]) / "data",
            document_map=event_coreference_contract["document_map"],
            private_dev_documents=set(
                corpus["masc"]["document_groups"]["private_dev"]
            ),
            private_eval_documents=set(
                corpus["masc"]["document_groups"]["private_eval"]
            ),
            maximum_characters=int(corpus["maximum_source_characters"]),
        )
    )
    expected_rejected_event_groups = {
        (row["document_id"], row["annotation_set_name"])
        for row in event_coreference_contract["expected_rejected_groups"]
    }
    observed_rejected_event_groups = {
        (row["document_id"], row["annotation_set_name"])
        for row in masc_event_coreference_audit["rejected_groups"]
    }
    if (
        masc_event_coreference_audit["policy"] != MASC_EVENT_COREFERENCE_POLICY
        or event_coreference_contract["alignment_contract"]
        != MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT
        or event_coreference_contract["source_compaction_contract"]
        != MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT
        or masc_event_coreference_audit["alignment_implementation"]
        != "producer_global_sequence_matcher_v1"
        or masc_event_coreference_audit["observed_group_count"]
        != int(event_coreference_contract["expected_observed_group_count"])
        or masc_event_coreference_audit["observed_mention_count"]
        != int(event_coreference_contract["expected_observed_mention_count"])
        or masc_event_coreference_audit["admitted_group_count"]
        != int(event_coreference_contract["expected_admitted_group_count"])
        or masc_event_coreference_audit["admitted_mention_count"]
        != int(event_coreference_contract["expected_admitted_mention_count"])
        or masc_event_coreference_audit["record_count_by_split"]
        != event_coreference_contract["records_by_split"]
        or masc_event_coreference_audit["mention_count_by_split"]
        != event_coreference_contract["mentions_by_split"]
        or masc_event_coreference_audit["rejected_group_count"]
        != int(event_coreference_contract["expected_rejected_group_count"])
        or observed_rejected_event_groups != expected_rejected_event_groups
        or masc_event_coreference_audit["partial_group_admission_count"] != 0
        or masc_event_coreference_audit["cooccurrence_inferred_relation_count"] != 0
    ):
        raise ValueError("MASC event-coreference producer reconstruction mismatch")
    mpqa_relation_contract = corpus["masc"]["mpqa_relations"]
    masc_mpqa_relation_selected, masc_mpqa_relation_audit = (
        reconstruct_mpqa_relation_chains(
            original_mpqa_root=resolve(
                mpqa_relation_contract["original_mpqa_root"]
            ),
            private_dev_documents=set(
                mpqa_relation_contract["private_dev_documents"]
            ),
            private_eval_documents=set(
                mpqa_relation_contract["private_eval_documents"]
            ),
            maximum_characters=int(corpus["maximum_source_characters"]),
        )
    )
    if (
        masc_mpqa_relation_audit["policy"] != MASC_MPQA_RELATION_POLICY
        or mpqa_relation_contract["relation_contract"]
        != MASC_MPQA_RELATION_CONTRACT
        or mpqa_relation_contract["source_compaction_contract"]
        != MASC_MPQA_RELATION_COMPACTION_CONTRACT
        or masc_mpqa_relation_audit["parser_implementation"]
        != "producer_regex_attribute_parser_v1"
        or masc_mpqa_relation_audit["observed_linked_expression_count"]
        != int(mpqa_relation_contract["expected_observed_linked_expression_count"])
        or masc_mpqa_relation_audit["admitted_relation_count"]
        != int(mpqa_relation_contract["expected_admitted_relation_count"])
        or masc_mpqa_relation_audit["admitted_source_member_count"]
        != int(mpqa_relation_contract["expected_admitted_source_member_count"])
        or masc_mpqa_relation_audit["admitted_attitude_count"]
        != int(mpqa_relation_contract["expected_admitted_attitude_count"])
        or masc_mpqa_relation_audit["admitted_target_count"]
        != int(mpqa_relation_contract["expected_admitted_target_count"])
        or masc_mpqa_relation_audit["record_count_by_split"]
        != mpqa_relation_contract["records_by_split"]
        or masc_mpqa_relation_audit["rejection_reason_counts"]
        != mpqa_relation_contract["expected_rejection_reason_counts"]
        or masc_mpqa_relation_audit["partial_relation_admission_count"] != 0
        or masc_mpqa_relation_audit["inferred_relation_count"] != 0
    ):
        raise ValueError("MASC MPQA-relation producer reconstruction mismatch")
    gum_contract = corpus["gum"]
    gum_selected, gum_audit = reconstruct_gum_discourse_relations(
        source_root=resolve(gum_contract["source_root"]),
        allowed_genre_licenses=dict(gum_contract["allowed_genre_licenses"]),
        private_dev_documents=set(gum_contract["private_dev_documents"]),
        private_eval_documents=set(gum_contract["private_eval_documents"]),
        expected_selected_source_sha256=gum_contract["content_sha256"],
        maximum_characters=int(corpus["maximum_source_characters"]),
    )
    if (
        gum_audit["policy"] != GUM_DISCOURSE_POLICY
        or gum_audit["relation_contract"] != GUM_DISCOURSE_RELATION_CONTRACT
        or gum_audit["split_contract"] != GUM_DISCOURSE_SPLIT_CONTRACT
        or gum_audit["projection_contract"] != GUM_DISCOURSE_PROJECTION_CONTRACT
        or gum_audit["parser_implementation"]
        != "producer_elementtree_tab_parser_v1"
        or gum_audit["selected_source_sha256"] != gum_contract["content_sha256"]
        or gum_audit["selected_document_count"]
        != int(gum_contract["expected_selected_document_count"])
        or gum_audit["document_count_by_split"]
        != gum_contract["documents_by_split"]
        or gum_audit["record_count_by_split"] != gum_contract["records_by_split"]
        or gum_audit["secondary_edge_count_by_split"]
        != gum_contract["secondary_edges_by_split"]
        or any(
            int(value) < int(gum_contract["minimum_relation_types_per_split"])
            for value in gum_audit["relation_type_count_by_split"].values()
        )
        or any(
            int(value) < int(gum_contract["minimum_weak_tail_count_per_split"])
            for value in gum_audit["minimum_relation_count_by_split"].values()
        )
        or gum_audit["official_nontrain_document_admission_count"] != 0
        or gum_audit["partial_relation_admission_count"] != 0
        or gum_audit["inferred_relation_count"] != 0
    ):
        raise ValueError("GUM eRST producer reconstruction mismatch")
    gum_scoped_contract = gum_contract["scoped_semantic_supervision"]
    gum_scoped_audit = gum_scoped_supervision_audit(gum_selected)
    if (
        gum_scoped_audit["policy"] != gum_scoped_contract["policy"]
        or gum_scoped_audit["admitted_relation_contract"]
        != gum_scoped_contract["admitted_relation_contract"]
        or gum_scoped_audit["record_counts_by_split_and_relation"]
        != gum_scoped_contract["record_counts_by_split_and_relation"]
        or gum_scoped_audit["record_counts_by_split"]
        != gum_scoped_contract["record_counts_by_split"]
        or gum_scoped_audit["excluded_mapped_record_count"]
        != int(gum_scoped_contract["excluded_mapped_multi_edge_record_count"])
        or gum_scoped_audit["minimum_relation_count_by_split"]
        != gum_scoped_contract["minimum_relation_count_by_split"]
        or gum_scoped_audit["cross_split_source_group_overlap_count"] != 0
    ):
        raise ValueError("GUM scoped-semantic producer reconstruction mismatch")
    gum_entity_contract = gum_contract["entity_coreference"]
    gum_entity_selected, gum_entity_audit = reconstruct_gum_entity_coreference(
        source_root=resolve(gum_contract["source_root"]),
        allowed_genre_licenses=dict(gum_contract["allowed_genre_licenses"]),
        private_dev_documents=set(gum_contract["private_dev_documents"]),
        private_eval_documents=set(gum_contract["private_eval_documents"]),
        expected_selected_source_sha256=gum_entity_contract["content_sha256"],
        maximum_characters=int(corpus["maximum_source_characters"]),
    )
    if (
        gum_entity_audit["policy"] != GUM_ENTITY_COREFERENCE_POLICY
        or gum_entity_audit["relation_contract"]
        != GUM_ENTITY_COREFERENCE_RELATION_CONTRACT
        or gum_entity_audit["split_contract"]
        != GUM_ENTITY_COREFERENCE_SPLIT_CONTRACT
        or gum_entity_audit["source_compaction_contract"]
        != GUM_ENTITY_COREFERENCE_COMPACTION_CONTRACT
        or gum_entity_audit["parser_implementation"]
        != "producer_webanno_tsv_state_machine_v1"
        or gum_entity_audit["selected_source_sha256"]
        != gum_entity_contract["content_sha256"]
        or gum_entity_audit["selected_document_count"]
        != int(gum_entity_contract["expected_selected_document_count"])
        or {
            split: int(values["records"])
            for split, values in gum_entity_audit["counts_by_split"].items()
        }
        != gum_entity_contract["records_by_split"]
        or {
            split: int(values["identity_records"])
            for split, values in gum_entity_audit["counts_by_split"].items()
        }
        != gum_entity_contract["identity_records_by_split"]
        or {
            split: int(values["bridge_records"])
            for split, values in gum_entity_audit["counts_by_split"].items()
        }
        != gum_entity_contract["bridge_records_by_split"]
        or {
            split: int(values["mentions"])
            for split, values in gum_entity_audit["counts_by_split"].items()
        }
        != gum_entity_contract["mentions_by_split"]
        or {
            split: int(values["components"])
            for split, values in gum_entity_audit["counts_by_split"].items()
        }
        != gum_entity_contract["components_by_split"]
        or gum_entity_audit["rejected_record_count"] != 0
        or gum_entity_audit["official_nontrain_document_admission_count"] != 0
        or gum_entity_audit["partial_component_admission_count"] != 0
        or gum_entity_audit["inferred_relation_count"] != 0
    ):
        raise ValueError("GUM entity/coreference producer reconstruction mismatch")
    oasst_rows, oasst_rejects = load_oasst_candidates(corpus["oasst2"])
    oasst_behavior_rows, oasst_behavior_rejects = load_oasst_behavior_candidates(
        corpus["oasst2"]
    )
    oasst_behavior_selected = select_oasst_behavior(
        oasst_behavior_rows,
        corpus["oasst2"]["explicit_behavior_records_by_split"],
    )
    reserved_behavior_groups = {
        row["source_group"]
        for rows in oasst_behavior_selected.values()
        for row in rows
    }
    oasst_rows = {
        split: [
            row for row in rows if row["source_group"] not in reserved_behavior_groups
        ]
        for split, rows in oasst_rows.items()
    }
    oasst_selected = select_oasst(
        oasst_rows, corpus["oasst2"]["records_by_split"]
    )
    masc_interactions = {
        split: interaction_predecessors(rows)
        for split, rows in masc_selected.items()
    }
    phase_runtime_ms["source_reconstruction_and_selection"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    producer_family_receipts = producer_family_identity_receipts()
    producer_family_identities = {
        family: receipt["identity_sha256"]
        for family, receipt in producer_family_receipts.items()
    }
    candidate_store = (
        ContentObjectCache(
            cache_root / "producer_objects.sqlite3",
            namespace=str(cache_cfg["producer_role"])
            + ":"
            + str(cache_cfg["producer_candidate_layer"]),
        )
        if cache_enabled
        else None
    )
    candidate_cache_hits: Counter[str] = Counter()
    candidate_cache_misses: Counter[str] = Counter()

    def materialize_candidate(
        *,
        family: str,
        inputs: dict[str, Any],
        expected_source_id: str,
        build: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        record, hit = cached_candidate_record(
            store=candidate_store,
            role=str(cache_cfg["producer_role"]),
            layer=str(cache_cfg["producer_candidate_layer"]),
            family=family,
            family_identity=producer_family_identities[family],
            inputs=inputs,
            expected_source_id=expected_source_id,
            refresh_cache=refresh_cache,
            build=build,
        )
        (candidate_cache_hits if hit else candidate_cache_misses)[family] += 1
        return record

    candidates: list[dict[str, Any]] = []
    source_groups_by_split: dict[str, set[str]] = defaultdict(set)
    source_sentences_by_split: dict[str, set[str]] = defaultdict(set)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    frame_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("private_train", "private_dev", "private_eval"):
        for row in dolly_selected[split]:
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="dolly_direct",
                inputs={"row": row, "split": split, "source": corpus["dolly"]},
                expected_source_id="dolly:" + identity[:24],
                build=lambda row=row, split=split: dolly_record(
                    row,
                    split=split,
                    source=corpus["dolly"],
                    producer_sha256=producer_family_identities["dolly_direct"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
            category_counts[split][row["category"]] += 1
        for row in dolly_grounded_selected[split]:
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="dolly_grounded",
                inputs={"row": row, "split": split, "source": corpus["dolly"]},
                expected_source_id="dolly-grounded:" + identity[:24],
                build=lambda row=row, split=split: dolly_grounded_question_record(
                    row,
                    split=split,
                    source=corpus["dolly"],
                    producer_sha256=producer_family_identities["dolly_grounded"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
            category_counts[split]["grounded_" + row["question_form"]] += 1
        for row in masc_selected[split]:
            lexical_unit = str(row["annotation"]["lexical_unit"]).strip().lower()
            frame_prior = masc_frame_priors.get(lexical_unit)
            if frame_prior is not None and row["annotation"]["frame_name"] not in {
                alternative["frame_name"]
                for alternative in frame_prior["alternatives"]
            }:
                frame_prior = None
            interaction = masc_interactions[split].get(row["selection_key"])
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="masc_frame",
                inputs={
                    "row": row,
                    "split": split,
                    "source": corpus["masc"],
                    "interaction_annotation": interaction,
                    "contextual_frame_ambiguity": frame_prior,
                },
                expected_source_id="masc-frame:" + identity[:24],
                build=lambda row=row, split=split, interaction=interaction, frame_prior=frame_prior: masc_record(
                    row,
                    split=split,
                    source=corpus["masc"],
                    producer_sha256=producer_family_identities["masc_frame"],
                    interaction_annotation=interaction,
                    contextual_frame_ambiguity=frame_prior,
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(row["sentence_identity"])
            frame_counts[split][row["annotation"]["frame_name"]] += 1
        for rows in masc_composite_selected[split]:
            component_ids = [
                "masc-frame:" + row["selection_key"].split(":", 1)[1][:24]
                for row in rows
            ]
            composite_identity = stable_hash(component_ids).split(":", 1)[1]
            record = materialize_candidate(
                family="masc_composite",
                inputs={
                    "rows": rows,
                    "split": split,
                    "source": corpus["masc"],
                    "frame_priors": masc_frame_priors,
                },
                expected_source_id="masc-composite:" + composite_identity[:24],
                build=lambda rows=rows, split=split: masc_composite_record(
                    rows,
                    split=split,
                    source=corpus["masc"],
                    producer_sha256=producer_family_identities["masc_composite"],
                    frame_priors=masc_frame_priors,
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(rows[0]["sentence_identity"])
        for row in masc_decision_selected[split]:
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="masc_decision",
                inputs={"row": row, "split": split, "source": corpus["masc"]},
                expected_source_id="masc-decision:" + identity[:24],
                build=lambda row=row, split=split: masc_decision_semantic_record(
                    row,
                    split=split,
                    source=corpus["masc"],
                    producer_sha256=producer_family_identities["masc_decision"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in masc_event_coreference_selected[split]:
            record = materialize_candidate(
                family="masc_event_coreference",
                inputs={"row": row, "source": corpus["masc"]},
                expected_source_id=row["source_id"],
                build=lambda row=row: masc_event_coreference_record(
                    row,
                    source=corpus["masc"],
                    producer_sha256=producer_family_identities[
                        "masc_event_coreference"
                    ],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in masc_mpqa_relation_selected[split]:
            record = materialize_candidate(
                family="masc_mpqa_relation",
                inputs={"row": row, "source": corpus["masc"]},
                expected_source_id=row["source_id"],
                build=lambda row=row: masc_mpqa_relation_record(
                    row,
                    source=corpus["masc"],
                    producer_sha256=producer_family_identities[
                        "masc_mpqa_relation"
                    ],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in gum_selected[split]:
            record = materialize_candidate(
                family="gum_discourse",
                inputs={"row": row, "source": corpus["gum"]},
                expected_source_id=row["source_id"],
                build=lambda row=row: gum_discourse_record(
                    row,
                    source=corpus["gum"],
                    producer_sha256=producer_family_identities["gum_discourse"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in gum_entity_selected[split]:
            record = materialize_candidate(
                family="gum_entity_coreference",
                inputs={
                    "row": row,
                    "source": {
                        key: value
                        for key, value in corpus["gum"].items()
                        if key != "scoped_semantic_supervision"
                    },
                },
                expected_source_id=row["source_id"],
                build=lambda row=row: gum_entity_coreference_record(
                    row,
                    source=corpus["gum"],
                    producer_sha256=producer_family_identities[
                        "gum_entity_coreference"
                    ],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in oasst_selected[split]:
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="oasst_dialogue",
                inputs={"row": row, "split": split, "source": corpus["oasst2"]},
                expected_source_id="oasst2:" + identity[:24],
                build=lambda row=row, split=split: oasst_record(
                    row,
                    split=split,
                    source=corpus["oasst2"],
                    producer_sha256=producer_family_identities["oasst_dialogue"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in oasst_behavior_selected[split]:
            identity = row["selection_key"].split(":", 1)[1]
            record = materialize_candidate(
                family="oasst_behavior",
                inputs={"row": row, "split": split, "source": corpus["oasst2"]},
                expected_source_id="oasst2-behavior:" + identity[:24],
                build=lambda row=row, split=split: oasst_behavior_record(
                    row,
                    split=split,
                    source=corpus["oasst2"],
                    producer_sha256=producer_family_identities["oasst_behavior"],
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
    if candidate_store is not None:
        candidate_store.close()
    phase_runtime_ms["candidate_record_materialization"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    candidates.sort(
        key=lambda row: (
            ("private_train", "private_dev", "private_eval").index(row["split"]),
            row["provenance"]["source_id"],
        )
    )
    economics = corpus["residual_economics"]
    importance_policy = fit_importance_policy(candidates)
    provisional: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    initial_lambda = float(economics["allocation_lambda_grid_bits"][0])
    economics_cache_hits = 0
    economics_cache_misses = 0
    importance_implementation_sha256 = sha256_file(
        ROOT / "scripts" / "kerc_importance_policy.py"
    )
    economics_implementation_sha256 = sha256_file(
        ROOT / "scripts" / "kerc_residual_economics.py"
    )
    economics_store = (
        ContentObjectCache(
            cache_root / "producer_objects.sqlite3",
            namespace=str(cache_cfg["producer_role"]) + ":structural_economics_v1",
        )
        if cache_enabled
        else None
    )
    try:
        for record in candidates:
            importance = predict_importance(record, importance_policy)
            packet = record["kernel_packet"]
            residual = packet["residual"]
            allocation_dependencies = {
                "kernel_program": packet["program"],
                "global_state": record["hrl_state"]["global"],
                "segment_residual": residual["segment_frame"],
                "token_residuals": residual["token_tags"],
                "exact_objects": packet["protected_objects"],
                "exact_codec": residual["codec"],
                "importance": importance,
                "initial_lambda": initial_lambda,
                "importance_policy_sha256": importance_policy["policy_sha256"],
                "importance_implementation_sha256": importance_implementation_sha256,
                "economics_implementation_sha256": economics_implementation_sha256,
            }
            allocation_key = object_key(
                role=str(cache_cfg["producer_role"]),
                layer="structural_economics_v1",
                dependencies=allocation_dependencies,
            )
            cached_allocation = (
                economics_store.get(allocation_key)
                if economics_store is not None and not refresh_cache
                else None
            )
            if isinstance(cached_allocation, dict):
                allocation = cached_allocation
                economics_cache_hits += 1
            else:
                allocation = build_structural_rate_distortion_allocation(
                    kernel_program=packet["program"],
                    global_state=record["hrl_state"]["global"],
                    segment_residual=residual["segment_frame"],
                    token_residuals=residual["token_tags"],
                    exact_objects=packet["protected_objects"],
                    importance=float(importance["allocation_importance"]),
                    lambda_value=initial_lambda,
                    exact_codec=residual["codec"],
                )
                economics_cache_misses += 1
                if economics_store is not None:
                    economics_store.put(allocation_key, allocation)
            provisional.append((record, importance, allocation))
    finally:
        if economics_store is not None:
            economics_store.close()
    phase_runtime_ms["importance_and_structural_economics"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    lambda_calibration = calibrate_allocation_lambda(
        [
            allocation
            for record, _importance, allocation in provisional
            if record["split"] == "private_dev"
        ],
        lambda_grid=economics["allocation_lambda_grid_bits"],
        maximum_importance_weighted_distortion=float(
            economics["maximum_dev_importance_weighted_structural_distortion"]
        ),
    )
    finalization_receipt = producer_finalization_identity_receipt()
    finalization_store = (
        ContentObjectCache(
            cache_root / "producer_objects.sqlite3",
            namespace=str(cache_cfg["producer_role"])
            + ":"
            + str(cache_cfg["producer_finalization_layer"]),
        )
        if cache_enabled
        else None
    )
    finalization_cache_hits = 0
    finalization_cache_misses = 0
    allocation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    finalized_candidates: list[dict[str, Any]] = []
    try:
        for record, importance, provisional_allocation in provisional:
            record, allocation, hit = cached_finalized_candidate(
                store=finalization_store,
                role=str(cache_cfg["producer_role"]),
                layer=str(cache_cfg["producer_finalization_layer"]),
                finalization_identity=finalization_receipt["identity_sha256"],
                record=record,
                importance=importance,
                provisional_allocation=provisional_allocation,
                lambda_value=float(lambda_calibration["selected_lambda"]),
                refresh_cache=refresh_cache,
            )
            if hit:
                finalization_cache_hits += 1
            else:
                finalization_cache_misses += 1
            finalized_candidates.append(record)
            allocation_counts[str(record["split"])][
                allocation["selected_fidelity"]
            ] += 1
    finally:
        if finalization_store is not None:
            finalization_store.close()
    candidates = finalized_candidates
    codec_bits: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_bits: dict[str, list[int]] = defaultdict(list)
    kernel_bits: dict[str, list[int]] = defaultdict(list)
    reasoning_wire_bits: dict[str, list[int]] = defaultdict(list)
    encoded_storage_bytes: dict[str, list[int]] = defaultdict(list)
    cleartext_residual_storage_bytes: dict[str, list[int]] = defaultdict(list)
    packet_audit_storage_bytes: dict[str, list[int]] = defaultdict(list)
    for record in candidates:
        split = str(record["split"])
        source_bits[split].append(len(record["source_text"].encode("utf-8")) * 8)
        codec = record["kernel_packet"]["residual"]["codec"]
        packet = record["kernel_packet"]
        codec_bits[split]["total"].append(int(codec["total_encoded_bits"]))
        kernel = len(residual_wire_bytes(packet["program"])) * 8
        kernel_bits[split].append(kernel)
        reasoning_wire_bits[split].append(kernel + int(codec["total_encoded_bits"]))
        encoded_storage_bytes[split].append(int(codec["encoded_storage_bytes"]))
        cleartext_residual_storage_bytes[split].append(
            int(codec["cleartext_abi_storage_bytes"])
        )
        packet_audit_storage_bytes[split].append(len(canonical_json(packet).encode("utf-8")))
        for channel in ("interaction", "segment", "token", "exact"):
            codec_bits[split][channel].append(
                int(codec["channels"][channel]["encoded_bits"])
            )
    hard_gaps: list[str] = []
    for split, floor in corpus["minimum_source_groups_by_split"].items():
        if len(source_groups_by_split[split]) < int(floor):
            hard_gaps.append(f"insufficient_source_groups:{split}")
    for split, floor in corpus["minimum_source_sentences_by_split"].items():
        if len(source_sentences_by_split[split]) < int(floor):
            hard_gaps.append(f"insufficient_source_sentences:{split}")
    groups = list(source_groups_by_split.values())
    if any(groups[left] & groups[right] for left in range(len(groups)) for right in range(left + 1, len(groups))):
        hard_gaps.append("cross_split_source_group_overlap")
    phase_runtime_ms["allocation_finalization_and_accounting"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    row_count, output_sha256 = write_jsonl_atomic(output_path, candidates)
    phase_runtime_ms["candidate_serialization"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    report = {
        "policy": PRODUCER_POLICY,
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "config": relative(config_path),
        "producer_sha256": producer_sha256,
        "producer_family_identities": producer_family_receipts,
        "producer_finalization_identity": finalization_receipt,
        "runtime_ms": round((time.perf_counter() - run_started) * 1000),
        "phase_runtime_ms": phase_runtime_ms,
        "candidate_records": {
            "path": relative(output_path),
            "sha256": output_sha256,
            "row_count": row_count,
        },
        "rows_by_split_and_source": {
            split: {
                "dolly": len(dolly_selected[split]),
                "dolly_grounded_question": len(dolly_grounded_selected[split]),
                "masc": len(masc_selected[split]),
                "masc_composite": len(masc_composite_selected[split]),
                "masc_decision_semantics": len(masc_decision_selected[split]),
                "masc_event_coreference": len(
                    masc_event_coreference_selected[split]
                ),
                "masc_mpqa_relations": len(masc_mpqa_relation_selected[split]),
                "gum_erst_discourse": len(gum_selected[split]),
                "gum_entity_coreference": len(gum_entity_selected[split]),
                "oasst2": len(oasst_selected[split]),
                "oasst2_explicit_behavior": len(oasst_behavior_selected[split]),
            }
            for split in ("private_train", "private_dev", "private_eval")
        },
        "source_group_count_by_split": {
            split: len(values) for split, values in source_groups_by_split.items()
        },
        "source_sentence_count_by_split": {
            split: len(values) for split, values in source_sentences_by_split.items()
        },
        "dolly_category_counts": {
            split: dict(values) for split, values in category_counts.items()
        },
        "dolly_grounded_question_form_counts_by_split": {
            split: dict(Counter(row["question_form"] for row in rows))
            for split, rows in dolly_grounded_selected.items()
        },
        "masc_frame_counts": {
            split: dict(values) for split, values in frame_counts.items()
        },
        "masc_composite_semantics": {
            "policy": "project_theseus_kerc_masc_composite_semantics_v1",
            "record_count_by_split": {
                split: len(groups)
                for split, groups in masc_composite_selected.items()
            },
            "frame_count_distribution_by_split": {
                split: dict(Counter(str(len(rows)) for rows in groups))
                for split, groups in masc_composite_selected.items()
            },
            "multi_node_program_count": sum(
                len(groups) for groups in masc_composite_selected.values()
            ),
            "multi_root_program_count": sum(
                len(groups) for groups in masc_composite_selected.values()
            ),
            "multi_claim_answer_count": sum(
                len(groups) for groups in masc_composite_selected.values()
            ),
            "unique_source_credit": int(
                corpus["masc"]["composite_semantic_unique_source_credit"]
            ),
            "claim_scope": corpus["masc"]["composite_semantic_claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "inter_frame_discourse_edges_claimed": False,
        },
        "masc_decision_semantics": {
            "policy": MASC_DECISION_SEMANTICS_POLICY,
            "record_count_by_split": {
                split: len(rows) for split, rows in masc_decision_selected.items()
            },
            "annotation_count_by_split_and_layer": {
                split: dict(
                    Counter(
                        item["layer"]
                        for row in rows
                        for item in row["annotation"]["annotations"]
                    )
                )
                for split, rows in masc_decision_selected.items()
            },
            "missing_layer_record_count_by_split": {
                split: {
                    layer: sum(
                        1 for row in rows if row["annotation"]["missingness"][layer]
                    )
                    for layer in ("cb", "event", "mpqa")
                }
                for split, rows in masc_decision_selected.items()
            },
            "typed_nonliteral_argument_count": sum(
                1
                for rows in masc_decision_selected.values()
                for row in rows
                for item in row["annotation"]["annotations"]
                for argument in masc_decision_arguments(item)
                if argument["value"]["type"] != "byte_literal"
            ),
            "unique_source_credit": int(
                corpus["masc"]["decision_semantic_unique_source_credit"]
            ),
            "claim_scope": corpus["masc"]["decision_semantic_claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "event_coreference_grouping_claimed": False,
            "source_declared_cross_annotation_links_resolved": False,
        },
        "masc_event_coreference": {
            "policy": MASC_EVENT_COREFERENCE_POLICY,
            "alignment_contract": MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT,
            "source_compaction_contract": MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT,
            "producer_alignment_implementation": masc_event_coreference_audit[
                "alignment_implementation"
            ],
            "observed_group_count": masc_event_coreference_audit[
                "observed_group_count"
            ],
            "observed_mention_count": masc_event_coreference_audit[
                "observed_mention_count"
            ],
            "admitted_group_count": masc_event_coreference_audit[
                "admitted_group_count"
            ],
            "admitted_mention_count": masc_event_coreference_audit[
                "admitted_mention_count"
            ],
            "record_count_by_split": masc_event_coreference_audit[
                "record_count_by_split"
            ],
            "mention_count_by_split": masc_event_coreference_audit[
                "mention_count_by_split"
            ],
            "admitted_source_ids_by_split": {
                split: [row["source_id"] for row in rows]
                for split, rows in masc_event_coreference_selected.items()
            },
            "rejected_group_count": masc_event_coreference_audit[
                "rejected_group_count"
            ],
            "rejected_groups": sorted(
                [
                    {
                        "document_id": row["document_id"],
                        "annotation_set_name": row["annotation_set_name"],
                    }
                    for row in masc_event_coreference_audit["rejected_groups"]
                ],
                key=lambda row: (row["document_id"], row["annotation_set_name"]),
            ),
            "partial_group_admission_count": 0,
            "cooccurrence_inferred_relation_count": 0,
            "unique_source_credit": int(
                event_coreference_contract["unique_source_credit"]
            ),
            "claim_scope": event_coreference_contract["claim_scope"],
            "complete_group_alignment_required": True,
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "causal_relation_claimed": False,
            "temporal_relation_claimed": False,
        },
        "masc_mpqa_relations": {
            "policy": MASC_MPQA_RELATION_POLICY,
            "relation_contract": MASC_MPQA_RELATION_CONTRACT,
            "source_compaction_contract": MASC_MPQA_RELATION_COMPACTION_CONTRACT,
            "producer_parser_implementation": masc_mpqa_relation_audit[
                "parser_implementation"
            ],
            "observed_linked_expression_count": masc_mpqa_relation_audit[
                "observed_linked_expression_count"
            ],
            "admitted_relation_count": masc_mpqa_relation_audit[
                "admitted_relation_count"
            ],
            "admitted_source_member_count": masc_mpqa_relation_audit[
                "admitted_source_member_count"
            ],
            "admitted_attitude_count": masc_mpqa_relation_audit[
                "admitted_attitude_count"
            ],
            "admitted_target_count": masc_mpqa_relation_audit[
                "admitted_target_count"
            ],
            "record_count_by_split": masc_mpqa_relation_audit[
                "record_count_by_split"
            ],
            "rejection_reason_counts": masc_mpqa_relation_audit[
                "rejection_reason_counts"
            ],
            "partial_relation_admission_count": 0,
            "inferred_relation_count": 0,
            "admitted_source_ids_by_split": {
                split: [row["source_id"] for row in rows]
                for split, rows in masc_mpqa_relation_selected.items()
            },
            "span_status_count": dict(
                Counter(
                    mpqa_member_span_status(member)
                    for rows in masc_mpqa_relation_selected.values()
                    for row in rows
                    for member in [
                        row["annotation"]["expression"],
                        *row["annotation"]["source_chain"],
                        *row["annotation"]["attitudes"],
                        *[
                            target
                            for attitude in row["annotation"]["attitudes"]
                            for target in attitude["targets"]
                        ],
                    ]
                )
            ),
            "unique_source_credit": int(
                mpqa_relation_contract["unique_source_credit"]
            ),
            "claim_scope": mpqa_relation_contract["claim_scope"],
            "complete_relation_alignment_required": True,
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "causal_relation_claimed": False,
            "temporal_relation_claimed": False,
        },
        "gum_erst_discourse": {
            **gum_audit,
            "scoped_semantic_supervision": {
                **gum_scoped_audit,
                "admission_contract": gum_scoped_contract["admission_contract"],
                "claim_scope": gum_scoped_contract["claim_scope"],
            },
            "claim_scope": gum_contract["claim_scope"],
            "source_credit_unit": "document",
            "derived_view_unique_source_credit": 0,
            "official_dev_test_quarantined": True,
            "public_gum_or_disrpt_score_claimed": False,
            "learned_competence_claimed": False,
        },
        "gum_entity_coreference": {
            **gum_entity_audit,
            "claim_scope": gum_entity_contract["claim_scope"],
            "source_credit_unit": "document",
            "derived_view_unique_source_credit": 0,
            "official_dev_test_quarantined": True,
            "cross_format_topology_required": True,
            "public_coreference_score_claimed": False,
            "cross_document_identity_claimed": False,
            "learned_competence_claimed": False,
        },
        "masc_interaction_record_count_by_split": {
            split: len(values) for split, values in masc_interactions.items()
        },
        "masc_contextual_frame_ambiguity": {
            "policy": MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY,
            "fit_split": "private_train",
            "eligible_lexical_unit_count": len(masc_frame_priors),
            "bound_record_count_by_split": {
                split: sum(
                    1
                    for row in rows
                    if str(row["annotation"]["lexical_unit"]).strip().lower()
                    in masc_frame_priors
                    and row["annotation"]["frame_name"]
                    in {
                        alternative["frame_name"]
                        for alternative in masc_frame_priors[
                            str(row["annotation"]["lexical_unit"])
                            .strip()
                            .lower()
                        ]["alternatives"]
                    }
                )
                for split, rows in masc_selected.items()
            },
            "prior_sha256_by_lexical_unit": {
                lexical_unit: prior["content_sha256"]
                for lexical_unit, prior in masc_frame_priors.items()
            },
            "unresolved_ambiguity_record_count": 0,
            "calibrated_probability_claimed": False,
            "claim_scope": corpus["masc"]["contextual_frame_ambiguity"][
                "claim_scope"
            ],
        },
        "oasst2_context_bound_record_count_by_split": {
            split: len(values) for split, values in oasst_selected.items()
        },
        "oasst2_human_valid_realization_count_by_split": {
            split: sum(len(row["targets"]) for row in values)
            for split, values in oasst_selected.items()
        },
        "oasst2_explicit_behavior_counts_by_split": {
            split: dict(Counter(row["disposition"] for row in rows))
            for split, rows in oasst_behavior_selected.items()
        },
        "candidate_pool_counts": {
            "dolly": len(dolly_rows),
            "dolly_grounded_question": len(dolly_grounded_rows),
            "masc_by_split": {split: len(rows) for split, rows in masc_rows.items()},
            "oasst2_by_split": {split: len(rows) for split, rows in oasst_rows.items()},
            "oasst2_explicit_behavior": len(oasst_behavior_rows),
        },
        "residual_codec_accounting": {
            "policy": "project_theseus_kerc_corpus_residual_bit_accounting_v1",
            "codec_policy": "project_theseus_kerc_conditional_residual_codec_v1",
            "by_split": {
                split: {
                    "source_bits": bit_distribution(source_bits[split]),
                    "residual_bits": {
                        channel: bit_distribution(codec_bits[split][channel])
                        for channel in ("interaction", "segment", "token", "exact", "total")
                    },
                    "aggregate_residual_to_source_bit_ratio": (
                        sum(codec_bits[split]["total"])
                        / max(1, sum(source_bits[split]))
                    ),
                    "kernel_wire_bits": bit_distribution(kernel_bits[split]),
                    "total_reasoning_wire_bits": bit_distribution(
                        reasoning_wire_bits[split]
                    ),
                    "aggregate_total_reasoning_wire_to_source_bit_ratio": (
                        sum(reasoning_wire_bits[split])
                        / max(1, sum(source_bits[split]))
                    ),
                    "encoded_residual_storage_bytes": bit_distribution(
                        encoded_storage_bytes[split]
                    ),
                    "cleartext_residual_abi_storage_bytes": bit_distribution(
                        cleartext_residual_storage_bytes[split]
                    ),
                    "full_packet_audit_storage_bytes": bit_distribution(
                        packet_audit_storage_bytes[split]
                    ),
                }
                for split in ("private_train", "private_dev", "private_eval")
            },
            "cleartext_abi_copy_charged_to_wire_bits": False,
            "cleartext_abi_copy_charged_to_storage": True,
            "capability_or_efficiency_claim": False,
        },
        "importance_policy": importance_policy,
        "rate_distortion_allocation": {
            "policy": "project_theseus_kerc_corpus_rate_distortion_allocation_v1",
            "lambda_calibration": lambda_calibration,
            "lambda_bits": float(lambda_calibration["selected_lambda"]),
            "selected_fidelity_counts_by_split": {
                split: dict(allocation_counts[split])
                for split in ("private_train", "private_dev", "private_eval")
            },
            "target_authority": "source_bound_structural_omission_not_semantic_utility",
            "semantic_utility_claim": False,
        },
        "rejection_counts": {
            "dolly": dict(dolly_rejects),
            "dolly_grounded_question": dict(dolly_grounded_rejects),
            "masc": dict(masc_rejects),
            "oasst2": dict(oasst_rejects),
            "oasst2_explicit_behavior": dict(oasst_behavior_rejects),
        },
        "independent_verification_required": True,
        "canonical_training_rows_written": 0,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "score_semantics": "licensed semantic candidate production; not admission or capability",
        "hard_gaps": hard_gaps,
    }
    write_json_atomic(manifest_path, report)
    if cache_enabled and report["trigger_state"] == "GREEN":
        publish_receipt(
            cache_root,
            role=str(cache_cfg["producer_role"]),
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="producer_manifest",
        )
    write_json_atomic(
        cache_root / "telemetry" / "producer_last.json",
        {
            "policy": "project_theseus_kerc_incremental_cache_telemetry_v1",
            "run_cache_hit": False,
            "structural_economics": {
                "hits": economics_cache_hits,
                "misses": economics_cache_misses,
                "entry_count": economics_cache_hits + economics_cache_misses,
            },
            "candidate_records": {
                "hits_by_family": dict(sorted(candidate_cache_hits.items())),
                "misses_by_family": dict(sorted(candidate_cache_misses.items())),
                "hits": sum(candidate_cache_hits.values()),
                "misses": sum(candidate_cache_misses.values()),
                "entry_count": sum(candidate_cache_hits.values())
                + sum(candidate_cache_misses.values()),
            },
            "candidate_finalization": {
                "hits": finalization_cache_hits,
                "misses": finalization_cache_misses,
                "entry_count": finalization_cache_hits
                + finalization_cache_misses,
            },
            "candidate_records_sha256": report["candidate_records"]["sha256"],
            "storage": cache_storage_telemetry(
                cache_root / "producer_objects.sqlite3"
            ),
            "claim_scope": "cache execution telemetry only; not semantic or capability evidence",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--bypass-run-cache", action="store_true")
    args = parser.parse_args()
    report = produce(
        resolve(args.config),
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        bypass_run_cache=args.bypass_run_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
