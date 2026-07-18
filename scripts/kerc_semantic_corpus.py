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
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from kernel_english_protocol import (
    ANSWER_DECISION_POLICY,
    SEMANTIC_SUPERVISION_POLICY,
    TRAINING_OBJECTIVES,
    TRAINING_RECORD_POLICY,
    build_kernel_packet,
    extract_protected_objects,
    revise_kernel_packet_fidelity,
    stable_hash,
)
from kerc_importance_policy import fit_importance_policy, predict_importance
from kerc_content_cache import (
    ContentObjectCache,
    dependency_bindings,
    load_receipt,
    object_key,
    publish_receipt,
)
from kerc_residual_economics import (
    build_structural_rate_distortion_allocation,
    calibrate_allocation_lambda,
    reallocate_structural_receipt,
    residual_wire_bytes,
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
        "kernel_protocol": scripts / "kernel_english_protocol.py",
        "importance_policy": scripts / "kerc_importance_policy.py",
        "residual_economics": scripts / "kerc_residual_economics.py",
        "semantic_config_validator": scripts / "moecot_source_conditioned_pretraining.py",
        "vcm_residual_lifecycle": scripts / "vcm_semantic_memory.py",
        "dolly_source": resolve(corpus["dolly"]["path"]),
        "masc_archive": resolve(corpus["masc"]["archive_path"]),
        "masc_extracted_tree": resolve(corpus["masc"]["extracted_root"]),
    }
    for split, row in sorted(corpus["oasst2"]["files"].items()):
        paths[f"oasst2_{split}_source"] = resolve(row["path"])
    return paths


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
    return {
        "policy": "project_theseus_kerc_residual_supervision_v1",
        "labels_by_channel": labels,
        "record_fidelity_label": fidelity_labels[residual["fidelity"]],
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
) -> dict[str, Any]:
    identity = stable_hash(
        {
            "split": split,
            "source_id": source_id,
            "source_group": source_group,
            "source_text": source_text,
            "surface_target": surface_target,
            "source_annotation": source_annotation,
            "interaction_annotation": interaction_annotation,
            "valid_realizations": valid_realizations,
        }
    ).split(":", 1)[1]
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
            identity, packet=packet, hrl_state=state
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
                    "candidate_records_sha256": cached["candidate_records"]["sha256"],
                    "claim_scope": "cache execution telemetry only; not semantic or capability evidence",
                },
            )
            return cached
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
    candidates: list[dict[str, Any]] = []
    source_groups_by_split: dict[str, set[str]] = defaultdict(set)
    source_sentences_by_split: dict[str, set[str]] = defaultdict(set)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    frame_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("private_train", "private_dev", "private_eval"):
        for row in dolly_selected[split]:
            record = dolly_record(
                row,
                split=split,
                source=corpus["dolly"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
            category_counts[split][row["category"]] += 1
        for row in dolly_grounded_selected[split]:
            record = dolly_grounded_question_record(
                row,
                split=split,
                source=corpus["dolly"],
                producer_sha256=producer_sha256,
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
            record = masc_record(
                row,
                split=split,
                source=corpus["masc"],
                producer_sha256=producer_sha256,
                interaction_annotation=masc_interactions[split].get(
                    row["selection_key"]
                ),
                contextual_frame_ambiguity=frame_prior,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(row["sentence_identity"])
            frame_counts[split][row["annotation"]["frame_name"]] += 1
        for rows in masc_composite_selected[split]:
            record = masc_composite_record(
                rows,
                split=split,
                source=corpus["masc"],
                producer_sha256=producer_sha256,
                frame_priors=masc_frame_priors,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(rows[0]["sentence_identity"])
        for row in masc_decision_selected[split]:
            record = masc_decision_semantic_record(
                row,
                split=split,
                source=corpus["masc"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in oasst_selected[split]:
            record = oasst_record(
                row,
                split=split,
                source=corpus["oasst2"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
        for row in oasst_behavior_selected[split]:
            record = oasst_behavior_record(
                row,
                split=split,
                source=corpus["oasst2"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
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
    allocation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record, importance, provisional_allocation in provisional:
        packet = record["kernel_packet"]
        allocation = reallocate_structural_receipt(
            provisional_allocation,
            lambda_value=float(lambda_calibration["selected_lambda"]),
        )
        packet = revise_kernel_packet_fidelity(
            packet,
            allocation["selected_fidelity"],
            local_hrl_state=record["hrl_state"],
        )
        record["kernel_packet"] = packet
        record["residual_supervision"] = residual_supervision(
            str(record["provenance"]["source_id"]),
            packet=packet,
            hrl_state=record["hrl_state"],
            importance=importance,
            allocation=allocation,
        )
        allocation_counts[str(record["split"])][
            allocation["selected_fidelity"]
        ] += 1
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
    row_count, output_sha256 = write_jsonl_atomic(output_path, candidates)
    report = {
        "policy": PRODUCER_POLICY,
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "config": relative(config_path),
        "producer_sha256": producer_sha256,
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
            "candidate_records_sha256": report["candidate_records"]["sha256"],
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
