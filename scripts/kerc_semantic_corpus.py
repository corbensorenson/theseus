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
    stable_hash,
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
DOLLY_QUESTION_FORM_RE = re.compile(
    r"^(who|what|when|where|which|how(?:\s+(?:many|much|long|old|far))?)\b",
    re.IGNORECASE,
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def residual_supervision(
    identity: str, *, packet: dict[str, Any], hrl_state: dict[str, Any]
) -> dict[str, Any]:
    residual = packet["residual"]
    fidelity_labels = {"semantic": 0, "faithful": 1, "lexical": 2, "exact": 3}
    labels = {
        "interaction": 1 if (hrl_state.get("segments") or {}) else 0,
        "segment": 1 if residual["segment_frame"] else 0,
        "token": 2 if residual["token_tags"] else 0,
        "exact": 3 if residual["exact_object_handles"] else 0,
    }
    return {
        "policy": "project_theseus_kerc_residual_supervision_v1",
        "labels_by_channel": labels,
        "record_fidelity_label": fidelity_labels[residual["fidelity"]],
        "annotator_independent_of_model": True,
        "evidence_sha256": stable_hash(
            {"identity": identity, "labels_by_channel": labels, "rule": "source_fidelity_v1"}
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


def masc_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
    interaction_annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    annotation = row["annotation"]
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
    handles_by_span = {
        (
            value["source_span"]["character_start"],
            value["source_span"]["character_end"],
        ): handle
        for handle, value in protected["protected_objects"].items()
        if value["protection_source"] == "explicit_user_or_caller_span"
    }

    def semantic_value(element: dict[str, Any]) -> dict[str, Any]:
        spans = element["spans"]
        if spans:
            bounds = (min(value[0] for value in spans), max(value[1] for value in spans))
            handle = handles_by_span.get(bounds)
            if handle is not None and annotation["sentence"][bounds[0] : bounds[1]] == element["text"]:
                return {"type": "handle", "value": handle}
        return byte_literal(element["text"])

    arguments = [
        {
            "role": safe_symbol(element["role"], prefix="ROLE"),
            "value": semantic_value(element),
        }
        for element in annotation["frame_elements"]
    ]
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


def produce(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage = validate_kernel_english_config(config)
    corpus = stage["semantic_corpus_materialization"]
    if corpus["policy"] != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("semantic corpus contract mismatch")
    producer_sha256 = sha256_file(Path(__file__).resolve())
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
            record = masc_record(
                row,
                split=split,
                source=corpus["masc"],
                producer_sha256=producer_sha256,
                interaction_annotation=masc_interactions[split].get(
                    row["selection_key"]
                ),
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(row["sentence_identity"])
            frame_counts[split][row["annotation"]["frame_name"]] += 1
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
    output_path = resolve(corpus["candidate_records_jsonl"])
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
        "masc_interaction_record_count_by_split": {
            split: len(values) for split, values in masc_interactions.items()
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
    write_json_atomic(resolve(corpus["producer_manifest_json"]), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = produce(resolve(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
