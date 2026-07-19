#!/usr/bin/env python3
"""Independently replay and admit the licensed KERC semantic corpus.

The verifier intentionally does not import the producer. It reparses the pinned
raw datasets, reconstructs split membership and semantic annotations, checks the
candidate packet against that evidence, and only then writes canonical rows.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from kerc_content_cache import (
    ContentObjectCache,
    cache_storage_telemetry,
    dependency_bindings,
    load_receipt,
    object_key,
    publish_receipt,
)
from kernel_english_protocol import (
    ANSWER_DECISION_POLICY,
    TRAINING_OBJECTIVES,
    TRAINING_VERIFICATION_POLICY,
    canonical_json,
    stable_hash,
    training_semantic_payload_sha256,
    validate_training_record,
)
from kerc_importance_policy import fit_importance_policy, predict_importance
from kerc_masc_event_coreference_verify import (
    ALIGNMENT_CONTRACT as MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT,
    COMPACTION_CONTRACT as MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT,
    POLICY as MASC_EVENT_COREFERENCE_POLICY,
    independently_reconstruct_event_coreference_groups,
)
from kerc_masc_mpqa_relations_verify import (
    COMPACTION_CONTRACT as MASC_MPQA_RELATION_COMPACTION_CONTRACT,
    POLICY as MASC_MPQA_RELATION_POLICY,
    RELATION_CONTRACT as MASC_MPQA_RELATION_CONTRACT,
    independently_reconstruct_mpqa_relation_chains,
)
from kerc_gum_discourse_relations_verify import (
    POLICY as GUM_DISCOURSE_POLICY,
    PROJECTION_CONTRACT as GUM_DISCOURSE_PROJECTION_CONTRACT,
    RELATION_CONTRACT as GUM_DISCOURSE_RELATION_CONTRACT,
    SPLIT_CONTRACT as GUM_DISCOURSE_SPLIT_CONTRACT,
    independently_reconstruct_gum_discourse_relations,
)
from kerc_gum_entity_coreference_verify import (
    COMPACTION_CONTRACT as GUM_ENTITY_COREFERENCE_COMPACTION_CONTRACT,
    POLICY as GUM_ENTITY_COREFERENCE_POLICY,
    RELATION_CONTRACT as GUM_ENTITY_COREFERENCE_RELATION_CONTRACT,
    SPLIT_CONTRACT as GUM_ENTITY_COREFERENCE_SPLIT_CONTRACT,
    independently_reconstruct_gum_entity_coreference,
)
from kerc_residual_economics import (
    calibrate_allocation_lambda,
    residual_wire_bytes,
    validate_structural_rate_distortion_allocation,
)
from kerc_source_family_identity import (
    PRODUCER_FAMILY_ROOTS,
    VERIFIER_FAMILY_ROOTS,
    family_identity_receipts,
    source_closure_receipt,
    source_family,
)
from moecot_source_conditioned_pretraining import (
    KERC_SEMANTIC_CORPUS_POLICY,
    KERC_SOURCE_CATALOG_POLICY,
    validate_kernel_english_config,
)
from vcm_semantic_memory import (
    apply_hierarchical_residual_delta,
    create_hierarchical_residual_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
DEFAULT_RUNTIME_CONFIG = ROOT / "configs" / "kerc_semantic_verifier_runtime.json"
GRAF = "{http://www.xces.org/ns/GrAF/1.0/}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
VERIFIER_POLICY = "project_theseus_kerc_semantic_corpus_verifier_v1"
VERIFIER_ID = "kerc_semantic_corpus_source_replay_v1"
SPLITS = ("private_train", "private_dev", "private_eval")
MASC_ENTITY_TYPES = {
    "person": "PERSON",
    "location": "PLACE",
    "org": "ORGANIZATION",
    "date": "DATE_TIME",
}
MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY = (
    "project_theseus_kerc_masc_train_only_contextual_frame_ambiguity_v1"
)
MASC_DECISION_SEMANTICS_POLICY = "project_theseus_kerc_masc_decision_semantics_v1"
MASC_MPQA_SEMANTIC_LABELS = {
    "agent", "attitude", "direct-subjective", "expressive-subjectivity", "target"
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def verifier_cache_dependency_paths(
    config_path: Path,
    runtime_config_path: Path,
    corpus: dict[str, Any],
    *,
    candidate_path: Path,
    manifest_path: Path,
) -> dict[str, Path]:
    scripts = ROOT / "scripts"
    paths = {
        "config": config_path.resolve(),
        "runtime_config": runtime_config_path.resolve(),
        "verifier": Path(__file__).resolve(),
        "producer": scripts / "kerc_semantic_corpus.py",
        "cache_integrity": scripts / "kerc_content_cache.py",
        "source_family_identity": scripts / "kerc_source_family_identity.py",
        "kernel_protocol": scripts / "kernel_english_protocol.py",
        "importance_policy": scripts / "kerc_importance_policy.py",
        "residual_economics": scripts / "kerc_residual_economics.py",
        "masc_event_coreference_verifier": scripts
        / "kerc_masc_event_coreference_verify.py",
        "masc_mpqa_relation_verifier": scripts
        / "kerc_masc_mpqa_relations_verify.py",
        "gum_entity_coreference_verifier": scripts
        / "kerc_gum_entity_coreference_verify.py",
        "semantic_config_validator": scripts / "moecot_source_conditioned_pretraining.py",
        "vcm_residual_lifecycle": scripts / "vcm_semantic_memory.py",
        "candidate_records": candidate_path,
        "producer_manifest": manifest_path,
        "dolly_source": resolve(corpus["dolly"]["path"]),
        "masc_archive": resolve(corpus["masc"]["archive_path"]),
        "masc_extracted_tree": resolve(corpus["masc"]["extracted_root"]),
        "gum_source_tree": resolve(corpus["gum"]["source_root"]),
    }
    for split, row in sorted(corpus["oasst2"]["files"].items()):
        paths[f"oasst2_{split}_source"] = resolve(row["path"])
    return paths


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def host_total_memory_bytes() -> int | None:
    """Return physical memory without adding an optional runtime dependency."""

    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
            return None
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def resolve_semantic_worker_count(
    parallel_cfg: dict[str, Any],
    resource_cfg: dict[str, Any],
    *,
    requested_workers: int | None,
    logical_cpu_count: int | None,
    total_memory_bytes: int | None,
) -> tuple[int, dict[str, Any]]:
    enabled = bool(parallel_cfg.get("enabled", False))
    maximum_workers = int(parallel_cfg.get("maximum_workers", 1))
    minimum_cpu_count = int(resource_cfg.get("minimum_logical_cpu_count", 2))
    parent_reserve = int(resource_cfg.get("parent_reserve_bytes", 0))
    memory_per_worker = int(resource_cfg.get("minimum_memory_bytes_per_worker", 1))
    if maximum_workers < 1 or minimum_cpu_count < 1 or memory_per_worker < 1:
        raise ValueError("invalid semantic verifier resource contract")
    cpu_count = max(1, int(logical_cpu_count or 1))
    cpu_safe_workers = max(1, cpu_count // minimum_cpu_count)
    memory_safe_workers = maximum_workers
    if total_memory_bytes is not None:
        memory_safe_workers = max(
            1,
            (int(total_memory_bytes) - parent_reserve) // memory_per_worker,
        )
    safe_workers = max(
        1, min(maximum_workers, cpu_safe_workers, memory_safe_workers)
    )
    configured = parallel_cfg.get("default_workers", 1)
    if requested_workers is None:
        worker_count = (
            safe_workers if configured == "auto" else int(configured)
        )
    else:
        worker_count = int(requested_workers)
    if not enabled:
        worker_count = 1
    if worker_count < 1 or worker_count > maximum_workers:
        raise ValueError(
            f"semantic admission workers must be between 1 and {maximum_workers}"
        )
    if worker_count > safe_workers:
        raise ValueError(
            "semantic admission worker request exceeds host resource contract"
        )
    return worker_count, {
        "policy": "project_theseus_kerc_semantic_worker_resource_receipt_v1",
        "parallelism_enabled": enabled,
        "configured_default": configured,
        "requested_workers": requested_workers,
        "selected_workers": worker_count,
        "maximum_workers": maximum_workers,
        "safe_workers": safe_workers,
        "logical_cpu_count": cpu_count,
        "total_memory_bytes": total_memory_bytes,
        "parent_reserve_bytes": parent_reserve,
        "minimum_memory_bytes_per_worker": memory_per_worker,
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


def residual_codec_accounting(records: list[dict[str, Any]]) -> dict[str, Any]:
    codec_bits: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_bits: dict[str, list[int]] = defaultdict(list)
    kernel_bits: dict[str, list[int]] = defaultdict(list)
    reasoning_wire_bits: dict[str, list[int]] = defaultdict(list)
    encoded_storage_bytes: dict[str, list[int]] = defaultdict(list)
    cleartext_residual_storage_bytes: dict[str, list[int]] = defaultdict(list)
    packet_audit_storage_bytes: dict[str, list[int]] = defaultdict(list)
    for record in records:
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
        packet_audit_storage_bytes[split].append(
            len(canonical_json(packet).encode("utf-8"))
        )
        for channel in ("interaction", "segment", "token", "exact"):
            codec_bits[split][channel].append(
                int(codec["channels"][channel]["encoded_bits"])
            )
    return {
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
            for split in SPLITS
        },
        "cleartext_abi_copy_charged_to_wire_bits": False,
        "cleartext_abi_copy_charged_to_storage": True,
        "capability_or_efficiency_claim": False,
    }


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


def dolly_prompt(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or "").strip()


def independent_dolly_assignments(
    source: dict[str, Any],
    maximum_characters: int,
    *,
    reserved_source_hashes: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    path = resolve(source["path"])
    if sha256_file(path) != source["content_sha256"]:
        raise ValueError("Dolly source hash mismatch")
    eligible: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        prompt = dolly_prompt(row)
        target = str(row.get("response") or "").strip()
        context = str(row.get("context") or "").strip()
        if (
            not 12 <= len(prompt) <= maximum_characters
            or not 2 <= len(target) <= maximum_characters
            or len(context) > maximum_characters
        ):
            continue
        prompt_hash = stable_hash(prompt.encode("utf-8"))
        target_hash = stable_hash(target.encode("utf-8"))
        if prompt_hash in seen_prompts or target_hash in seen_targets:
            continue
        seen_prompts.add(prompt_hash)
        seen_targets.add(target_hash)
        annotation = {
            "source_kind": "dolly_human_instruction_response",
            "line_number": line_number,
            "instruction": str(row.get("instruction") or ""),
            "context": str(row.get("context") or ""),
            "response": str(row.get("response") or ""),
            "category": str(row.get("category") or ""),
            "source_row_sha256": stable_hash(row),
        }
        if annotation["source_row_sha256"] in (reserved_source_hashes or set()):
            continue
        selection_key = stable_hash({"dataset": source["dataset_id"], "annotation": annotation})
        eligible.append(
            {
                "selection_key": selection_key,
                "source_id": "dolly:" + selection_key.split(":", 1)[1][:24],
                "prompt": prompt,
                "target": target,
                "annotation": annotation,
            }
        )
    eligible.sort(key=lambda row: row["selection_key"])
    output: dict[str, dict[str, Any]] = {}
    offset = 0
    for split, count in source["records_by_split"].items():
        for row in eligible[offset : offset + int(count)]:
            output[row["source_id"]] = {**row, "split": split}
        offset += int(count)
    if len(output) != sum(int(value) for value in source["records_by_split"].values()):
        raise ValueError("Dolly independent split reconstruction is incomplete")
    return output


def independent_dolly_grounded_assignments(
    source: dict[str, Any], maximum_characters: int
) -> dict[str, dict[str, Any]]:
    """Replay unique extractive QA eligibility and balanced split assignment."""

    path = resolve(source["path"])
    if sha256_file(path) != source["content_sha256"]:
        raise ValueError("Dolly grounded-question source hash mismatch")
    question_re = re.compile(
        r"^(who|what|when|where|which|how(?:\s+(?:many|much|long|old|far))?)\b",
        re.IGNORECASE,
    )
    eligible: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        prompt = dolly_prompt(row)
        target = str(row.get("response") or "").strip()
        context = str(row.get("context") or "").strip()
        question_match = question_re.match(prompt)
        if (
            str(row.get("category") or "") != "closed_qa"
            or question_match is None
            or not prompt.endswith("?")
            or not 12 <= len(prompt) <= maximum_characters
            or not 2 <= len(target) <= min(512, maximum_characters)
            or not 16 <= len(context) <= maximum_characters
            or context.count(target) != 1
            or len(target) / len(context) > 0.5
        ):
            continue
        prompt_hash = stable_hash(prompt.encode("utf-8"))
        target_hash = stable_hash(target.encode("utf-8"))
        if prompt_hash in seen_prompts or target_hash in seen_targets:
            continue
        seen_prompts.add(prompt_hash)
        seen_targets.add(target_hash)
        answer_start = context.index(target)
        question_form = question_match.group(1).lower().replace(" ", "_")
        annotation = {
            "source_kind": "dolly_human_unique_extractive_question_answer",
            "line_number": line_number,
            "instruction": str(row.get("instruction") or ""),
            "context": str(row.get("context") or ""),
            "response": str(row.get("response") or ""),
            "category": "closed_qa",
            "question_form": question_form,
            "answer_span": [answer_start, answer_start + len(target)],
            "support_relation": "unique_contiguous_exact_span",
            "support_claim_scope": "extractive_source_support_only",
            "broad_entailment_or_truth_claimed": False,
            "source_row_sha256": stable_hash(row),
        }
        selection_key = stable_hash(
            {"dataset": source["dataset_id"], "annotation": annotation}
        )
        eligible.append(
            {
                "selection_key": selection_key,
                "source_id": "dolly-grounded:"
                + selection_key.split(":", 1)[1][:24],
                "source_group": "dolly-grounded-row:"
                + annotation["source_row_sha256"].split(":", 1)[1],
                "prompt": prompt,
                "target": target,
                "context": context,
                "question_form": question_form,
                "annotation": annotation,
            }
        )
    available = sorted(eligible, key=lambda row: row["selection_key"])
    output: dict[str, dict[str, Any]] = {}
    required_forms = list(source["grounded_question_required_forms"])
    split_rows = list(source["grounded_question_records_by_split"].items())
    for split_index, (split, raw_count) in enumerate(split_rows):
        count = int(raw_count)
        future_split_count = len(split_rows) - split_index - 1
        selected: list[dict[str, Any]] = []
        for form in required_forms:
            match = next(
                (row for row in available if row["question_form"] == form), None
            )
            if match is None:
                raise ValueError(
                    f"Dolly grounded-question form reconstruction incomplete: {split}:{form}"
                )
            selected.append(match)
            available.remove(match)
        form_counts = Counter(row["question_form"] for row in selected)
        while len(selected) < count:
            available_form_counts = Counter(row["question_form"] for row in available)
            fillable = [
                row
                for row in available
                if row["question_form"] not in required_forms
                or available_form_counts[row["question_form"]] > future_split_count
            ]
            if not fillable:
                raise ValueError("Dolly grounded-question reconstruction incomplete")
            match = min(
                fillable,
                key=lambda row: (form_counts[row["question_form"]], row["selection_key"]),
            )
            selected.append(match)
            available.remove(match)
            form_counts[match["question_form"]] += 1
        for row in selected:
            output[row["source_id"]] = {**row, "split": split}
    return output


def independent_source_labels(row: dict[str, Any]) -> dict[str, float]:
    payload = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    names = payload.get("name") if isinstance(payload.get("name"), list) else []
    values = payload.get("value") if isinstance(payload.get("value"), list) else []
    return {str(name): float(value) for name, value in zip(names, values)}


def independent_oasst_eligible(row: dict[str, Any], source: dict[str, Any]) -> bool:
    labels = independent_source_labels(row)
    return bool(
        str(row.get("lang") or "").lower() == "en"
        and row.get("review_result") is True
        and row.get("deleted") is not True
        and row.get("synthetic") is not True
        and str(row.get("tree_state") or "") == "ready_for_export"
        and str(row.get("text") or "").strip()
        and float(labels.get("quality", -1.0)) >= float(source["minimum_quality"])
        and all(
            float(labels.get(name, 0.0)) <= float(maximum)
            for name, maximum in source["maximum_label_values"].items()
        )
    )


def independent_explicit_answer_behavior(text: str) -> tuple[str, str] | None:
    """Reconstruct the narrow surface behavior without producer code."""

    normalized = " ".join(str(text).strip().lower().split())[:400]
    clarification_patterns = (
        ("request_more_detail", r"^(?:i(?:'m| am) sorry[^?!.]{0,120}[,.]?\s*)?(?:could|can|would|will) you (?:please )?(?:clarify|provide|specify|explain|elaborate|tell me|give me)\b"),
        ("imperative_more_detail", r"^please (?:clarify|provide|specify|explain|elaborate)\b"),
        ("meaning_question", r"^(?:what do you mean|which .{0,100} do you mean|do you mean)\b"),
        ("explicit_choice_question", r"^there (?:are|is) .{0,120}(?:which|what).{0,120}\?"),
    )
    if "?" in normalized:
        for rule_id, pattern in clarification_patterns:
            if re.search(pattern, normalized):
                return "CLARIFY", rule_id
    abstention_patterns = (
        ("explicit_unknown", r"^(?:i(?:'m| am) sorry,? (?:but )?)?i (?:do not|don't) know\b"),
        ("explicit_cannot_determine", r"^(?:as [^.!?]{1,120}[,.]\s*)?[^.!?]{0,120}\bi (?:cannot|can't) (?:determine|answer|know|tell)\b"),
        ("explicit_insufficient", r"^(?:there is|there's) (?:insufficient|not enough) (?:information|context|details)\b"),
        ("explicit_missing_context", r"^i (?:do not|don't) have (?:enough |sufficient )?(?:information|context|details)\b"),
    )
    for rule_id, pattern in abstention_patterns:
        if re.search(pattern, normalized):
            return "ABSTAIN", rule_id
    return None


def independent_oasst_behavior_assignments(
    source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    contract = source["files"]["train"]
    path = resolve(contract["path"])
    if sha256_file(path) != contract["content_sha256"]:
        raise ValueError("OASST2 behavior source hash mismatch")
    raw = pq.read_table(path).to_pylist()
    nodes = {str(row["message_id"]): row for row in raw if row.get("message_id")}
    candidates: list[dict[str, Any]] = []
    for response in nodes.values():
        if not (
            independent_oasst_eligible(response, source)
            and response.get("role") == "assistant"
        ):
            continue
        behavior = independent_explicit_answer_behavior(str(response.get("text") or ""))
        if behavior is None:
            continue
        parent_id = str(response.get("parent_id") or "")
        parent = nodes.get(parent_id)
        if not (
            parent
            and independent_oasst_eligible(parent, source)
            and parent.get("role") == "prompter"
        ):
            continue
        chain: list[dict[str, Any]] = []
        current = parent
        seen: set[str] = set()
        while current and str(current["message_id"]) not in seen and len(chain) < 16:
            if not independent_oasst_eligible(current, source):
                chain = []
                break
            seen.add(str(current["message_id"]))
            chain.append(current)
            current = nodes.get(str(current.get("parent_id") or ""))
        chain.reverse()
        if current is not None or not chain:
            continue
        roles = ["user" if row.get("role") == "prompter" else "assistant" for row in chain]
        if roles != ["user" if index % 2 == 0 else "assistant" for index in range(len(chain))]:
            continue
        prompt = str(parent["text"]).strip()
        target = str(response["text"]).strip()
        prior = chain[:-1][-int(source["maximum_prior_turns"]) :]
        turns = [
            {
                "turn_index": index,
                "role": "user" if row.get("role") == "prompter" else "assistant",
                "content": str(row["text"]).strip(),
                "source_message_sha256": stable_hash(str(row["message_id"]).encode("utf-8")),
            }
            for index, row in enumerate(prior)
        ]
        compiled = [
            [f"turn_{turn['turn_index']:03d}", key, turn[key]]
            for turn in turns
            for key in ("content", "role")
        ]
        if (
            len(prompt) > int(source["maximum_current_characters"])
            or len(target) > int(source["maximum_response_characters"])
            or sum(len(turn["content"]) for turn in turns) > int(source["maximum_context_characters"])
            or len(canonical_json(compiled).encode("utf-8")) > int(source["maximum_compiled_context_bytes"])
        ):
            continue
        disposition, rule_id = behavior
        tree_hash = stable_hash(str(response["message_tree_id"]).encode("utf-8"))
        annotation = {
            "source_kind": "oasst2_reviewed_explicit_answer_behavior",
            "official_split": "train",
            "message_tree_sha256": tree_hash,
            "parent_message_sha256": stable_hash(parent_id.encode("utf-8")),
            "response_message_sha256": stable_hash(str(response["message_id"]).encode("utf-8")),
            "prompt": prompt,
            "target": target,
            "interaction_turns": turns,
            "behavior_policy": "project_theseus_oasst_explicit_answer_behavior_v1",
            "disposition": disposition,
            "matched_rule_id": rule_id,
            "behavior_claim_scope": "explicit_human_surface_behavior_only",
            "optimality_or_truth_verified": False,
        }
        key = stable_hash({"dataset": source["dataset_id"], "annotation": annotation})
        candidates.append(
            {
                "selection_key": key,
                "source_id": "oasst2-behavior:" + key.split(":", 1)[1][:24],
                "source_group": "oasst2-tree:" + tree_hash.split(":", 1)[1],
                "prompt": prompt,
                "target": target,
                "interaction_turns": turns,
                "annotation": annotation,
                "disposition": disposition,
            }
        )
    by_tree: dict[str, dict[str, Any]] = {}
    for row in sorted(candidates, key=lambda value: value["selection_key"]):
        by_tree.setdefault(row["source_group"], row)
    pools = {
        disposition: sorted(
            (row for row in by_tree.values() if row["disposition"] == disposition),
            key=lambda row: row["selection_key"],
        )
        for disposition in ("CLARIFY", "ABSTAIN")
    }
    cursor = {disposition: 0 for disposition in pools}
    output: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for disposition in ("CLARIFY", "ABSTAIN"):
            count = int(source["explicit_behavior_records_by_split"][split][disposition])
            start = cursor[disposition]
            chosen = pools[disposition][start : start + count]
            if len(chosen) != count:
                raise ValueError(f"OASST2 behavior reconstruction incomplete: {split}:{disposition}")
            for row in chosen:
                output[row["source_id"]] = {**row, "split": split}
            cursor[disposition] += count
    return output


def independent_oasst_assignments(
    source: dict[str, Any], *, reserved_groups: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_file_hashes: dict[str, str] = {}
    for official_split, contract in source["files"].items():
        path = resolve(contract["path"])
        observed_file_hashes[official_split] = sha256_file(path)
        if observed_file_hashes[official_split] != contract["content_sha256"]:
            raise ValueError(f"OASST2 source hash mismatch: {official_split}")
        rows = pq.read_table(path).to_pylist()
        nodes = {str(row["message_id"]): row for row in rows if row.get("message_id")}
        children: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if (
                independent_oasst_eligible(row, source)
                and row.get("role") == "assistant"
                and row.get("rank") in (0, 1)
            ):
                children[str(row.get("parent_id") or "")].append(row)
        for parent_id, replies in children.items():
            if sorted(int(row["rank"]) for row in replies) != [0, 1]:
                continue
            chain: list[dict[str, Any]] = []
            current = nodes.get(parent_id)
            seen: set[str] = set()
            while current and str(current["message_id"]) not in seen and len(chain) < 8:
                if not independent_oasst_eligible(current, source):
                    chain = []
                    break
                seen.add(str(current["message_id"]))
                chain.append(current)
                current = nodes.get(str(current.get("parent_id") or ""))
            chain.reverse()
            if current is not None or len(chain) < 3:
                continue
            roles = [
                "user" if row.get("role") == "prompter" else "assistant"
                for row in chain
            ]
            if roles != ["user" if index % 2 == 0 else "assistant" for index in range(len(chain))]:
                continue
            prompt = str(chain[-1]["text"]).strip()
            replies = sorted(replies, key=lambda row: int(row["rank"]))
            targets = [str(row["text"]).strip() for row in replies]
            if len(set(targets)) != len(targets):
                continue
            prior = chain[:-1]
            compiled_context = [
                [f"turn_{index:03d}", key, value]
                for index, ancestor in enumerate(prior)
                for key, value in (
                    ("content", str(ancestor["text"]).strip()),
                    (
                        "role",
                        "user"
                        if str(ancestor.get("role") or "") == "prompter"
                        else "assistant",
                    ),
                )
            ]
            if (
                len(prior) < int(source["minimum_prior_turns"])
                or len(prior) > int(source["maximum_prior_turns"])
                or len(prompt) > int(source["maximum_current_characters"])
                or any(len(target) > int(source["maximum_response_characters"]) for target in targets)
                or sum(len(str(row["text"])) for row in prior)
                > int(source["maximum_context_characters"])
                or len(canonical_json(compiled_context).encode("utf-8"))
                > int(source["maximum_compiled_context_bytes"])
            ):
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
                for index, row in enumerate(prior)
            ]
            annotation = {
                "source_kind": "oasst2_reviewed_conversation_tree",
                "official_split": official_split,
                "message_tree_sha256": tree_hash,
                "parent_message_sha256": stable_hash(parent_id.encode("utf-8")),
                "current_prompt": prompt,
                "interaction_turns": interaction_turns,
                "responses": [
                    {
                        "rank": int(row["rank"]),
                        "text": str(row["text"]).strip(),
                        "source_message_sha256": stable_hash(
                            str(row["message_id"]).encode("utf-8")
                        ),
                        "quality": independent_source_labels(row).get("quality"),
                    }
                    for row in replies
                ],
            }
            selection_key = stable_hash(
                {"dataset": source["dataset_id"], "annotation": annotation}
            )
            candidates[split].append(
                {
                    "selection_key": selection_key,
                    "source_id": "oasst2:" + selection_key.split(":", 1)[1][:24],
                    "source_group": "oasst2-tree:" + tree_hash.split(":", 1)[1],
                    "prompt": prompt,
                    "interaction_turns": interaction_turns,
                    "targets": targets,
                    "annotation": annotation,
                    "split": split,
                }
            )
    if stable_hash(observed_file_hashes) != source["content_sha256"]:
        raise ValueError("OASST2 aggregate source identity mismatch")
    output: dict[str, dict[str, Any]] = {}
    for split, count in source["records_by_split"].items():
        rows = sorted(
            (
                row
                for row in candidates.get(split, [])
                if row["source_group"] not in (reserved_groups or set())
            ),
            key=lambda row: row["selection_key"],
        )
        if len(rows) < int(count):
            raise ValueError(f"OASST2 independent split reconstruction is incomplete: {split}")
        for row in rows[: int(count)]:
            output[row["source_id"]] = row
    return output


def parse_graf(path: Path) -> tuple[dict[str, list[tuple[str, dict[str, str]]]], dict[str, list[str]], dict[str, list[str]]]:
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
        fs = annotation.find(GRAF + "fs")
        if fs is not None:
            for field in fs.findall(GRAF + "f"):
                fields[str(field.get("name"))] = str(field.get("value") or "")
        annotations[str(annotation.get("ref"))].append((str(annotation.get("label") or ""), fields))
    for edge in root.findall(GRAF + "edge"):
        edges[str(edge.get("from"))].append(str(edge.get("to")))
    return dict(annotations), dict(edges), links


def independent_direct_annotations(path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(path).getroot()
    regions: dict[str, tuple[int, int]] = {}
    for region in tree.findall(GRAF + "region"):
        values = str(region.get("anchors") or "").split()
        if len(values) == 2:
            regions[str(region.get(XML_ID))] = (int(values[0]), int(values[1]))
    node_regions: dict[str, list[str]] = {}
    for node in tree.findall(GRAF + "node"):
        link = node.find(GRAF + "link")
        node_regions[str(node.get(XML_ID))] = (
            str(link.get("targets") or "").split() if link is not None else []
        )
    rows: list[dict[str, Any]] = []
    for annotation in tree.findall(GRAF + "a"):
        spans = [
            regions[region]
            for region in node_regions.get(str(annotation.get("ref")), [])
            if region in regions
        ]
        if not spans:
            continue
        fields: dict[str, str] = {}
        fs = annotation.find(GRAF + "fs")
        for field in fs.findall(GRAF + "f") if fs is not None else ():
            fields[str(field.get("name") or "")] = str(field.get("value") or "")
        rows.append(
            {
                "annotation_id": str(annotation.get(XML_ID) or ""),
                "node_id": str(annotation.get("ref") or ""),
                "label": str(annotation.get("label") or ""),
                "fields": dict(sorted(fields.items())),
                "start": min(start for start, _end in spans),
                "end": max(end for _start, end in spans),
            }
        )
    return rows


def independent_masc_decision_rows(base: Path, root: Path) -> list[dict[str, Any]]:
    text = Path(str(base) + ".txt").read_text(encoding="utf-8", errors="replace")
    document_id = str(base.relative_to(root)).replace(os.sep, "/")
    all_annotations: list[dict[str, Any]] = []
    for layer in ("cb", "event", "mpqa"):
        path = Path(str(base) + f"-{layer}.xml")
        if not path.exists():
            continue
        for row in independent_direct_annotations(path):
            if layer == "cb" and row["label"] == "Not Applicable":
                continue
            if layer == "mpqa" and row["label"] not in MASC_MPQA_SEMANTIC_LABELS:
                continue
            all_annotations.append({"layer": layer, **row})
    output: list[dict[str, Any]] = []
    for sentence in independent_direct_annotations(Path(str(base) + "-s.xml")):
        start, end = int(sentence["start"]), int(sentence["end"])
        members = []
        for raw in all_annotations:
            if start <= int(raw["start"]) < int(raw["end"]) <= end:
                members.append(
                    {
                        **raw,
                        "start": int(raw["start"]) - start,
                        "end": int(raw["end"]) - start,
                        "text": text[int(raw["start"]):int(raw["end"])],
                    }
                )
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
            "sentence": text[start:end],
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


def independent_masc_decision_assignments(
    source: dict[str, Any], maximum_characters: int
) -> dict[str, dict[str, Any]]:
    root = resolve(source["extracted_root"]) / "data"
    dev = set(source["document_groups"]["private_dev"])
    evaluation = set(source["document_groups"]["private_eval"])
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for text_path in sorted(root.rglob("*.txt")):
        base = Path(str(text_path)[:-4])
        if not Path(str(base) + "-s.xml").exists() or not any(
            Path(str(base) + f"-{layer}.xml").exists() for layer in ("cb", "event", "mpqa")
        ):
            continue
        document_id = str(base.relative_to(root)).replace(os.sep, "/")
        split = "private_dev" if document_id in dev else "private_eval" if document_id in evaluation else "private_train"
        for row in independent_masc_decision_rows(base, root):
            count = len(row["annotation"]["annotations"])
            if (
                0 < len(row["annotation"]["sentence"]) <= maximum_characters
                and int(source["decision_semantic_minimum_annotations"])
                <= count <= int(source["decision_semantic_maximum_annotations"])
            ):
                by_split[split].append(row)
    output: dict[str, dict[str, Any]] = {}
    for split, required in source["decision_semantic_records_by_split"].items():
        rows = sorted(by_split.get(split, []), key=lambda row: row["selection_key"])
        if len(rows) < int(required):
            raise ValueError(f"MASC decision-semantic independent split is incomplete: {split}")
        for row in rows[: int(required)]:
            identity = row["selection_key"].split(":", 1)[1]
            output["masc-decision:" + identity[:24]] = {"split": split, **row}
    return output


def descendants(root: str, edges: dict[str, list[str]]) -> set[str]:
    observed: set[str] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if node in observed:
            continue
        observed.add(node)
        pending.extend(edges.get(node, ()))
    return observed


def fields_for(annotations: dict[str, list[tuple[str, dict[str, str]]]], node: str, label: str) -> list[dict[str, str]]:
    return [fields for observed, fields in annotations.get(node, ()) if observed == label]


def spans_for(node: str, edges: dict[str, list[str]], token_links: dict[str, list[str]], anchors: dict[str, tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(
        {
            anchors[region]
            for descendant in descendants(node, edges)
            for region in token_links.get(descendant, ())
            if region in anchors
        }
    )


def independent_masc_named_entities(
    base: Path,
    *,
    text: str,
    anchors: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    entity_path = Path(str(base) + "-ne.xml")
    penn_path = Path(str(base) + "-penn.xml")
    if not entity_path.exists() or not penn_path.exists():
        return []
    annotations, edges, _ = parse_graf(entity_path)
    _, _, penn_links = parse_graf(penn_path)
    candidates: list[dict[str, Any]] = []
    for node, rows in annotations.items():
        for label, fields in rows:
            object_type = MASC_ENTITY_TYPES.get(label)
            if object_type is None:
                continue
            spans = spans_for(node, edges, penn_links, anchors)
            if not spans:
                continue
            start = min(value[0] for value in spans)
            end = max(value[1] for value in spans)
            if not 0 <= start < end <= len(text):
                continue
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "object_type": object_type,
                    "copy_policy": "EXACT",
                    "source_label": label,
                    "source_features": dict(sorted(fields.items())),
                    "text": text[start:end],
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


def independent_masc_document(path: Path, root: Path) -> list[dict[str, Any]]:
    base = Path(str(path)[: -len("-fn.xml")])
    document_id = str(base.relative_to(root)).replace(os.sep, "/")
    text = Path(str(base) + ".txt").read_text(encoding="utf-8", errors="replace")
    annotations, edges, _ = parse_graf(path)
    _, _, token_links = parse_graf(Path(str(base) + "-fntok.xml"))
    segment_root = ET.parse(Path(str(base) + "-seg.xml")).getroot()
    anchors = {
        str(region.get(XML_ID)): tuple(int(value) for value in str(region.get("anchors") or "").split())
        for region in segment_root.findall(GRAF + "region")
    }
    named_entities = independent_masc_named_entities(
        base,
        text=text,
        anchors=anchors,
    )
    output: list[dict[str, Any]] = []
    for sentence_node, annotation_rows in annotations.items():
        if not any(label == "sentence" for label, _fields in annotation_rows):
            continue
        sentence_spans = spans_for(sentence_node, edges, token_links, anchors)
        if not sentence_spans:
            continue
        sentence_start = min(start for start, _end in sentence_spans)
        sentence_end = max(end for _start, end in sentence_spans)
        sentence = text[sentence_start:sentence_end]
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
            sets = fields_for(annotations, annotation_node, "annotationSet")
            if not sets or sets[0].get("status") != "MANUAL" or not sets[0].get("frameName"):
                continue
            target_spans: list[list[int]] = []
            frame_elements: list[dict[str, Any]] = []
            for child in edges.get(annotation_node, ()):
                absolute = spans_for(child, edges, token_links, anchors)
                relative_spans = [
                    [start - sentence_start, end - sentence_start]
                    for start, end in absolute
                    if sentence_start <= start < end <= sentence_end
                ]
                if fields_for(annotations, child, "Target"):
                    target_spans.extend(relative_spans)
                for fields in fields_for(annotations, child, "FE"):
                    if relative_spans:
                        frame_elements.append(
                            {
                                "role": str(fields.get("FE") or "ROLE"),
                                "rank": str(fields.get("rank") or ""),
                                "gf": str(fields.get("GF") or ""),
                                "pt": str(fields.get("PT") or ""),
                                "spans": relative_spans,
                                "text": " ".join(sentence[start:end] for start, end in relative_spans),
                            }
                        )
            if not target_spans or not frame_elements:
                continue
            annotation = {
                "source_kind": "masc_manual_framenet",
                "document_id": document_id,
                "sentence_node": sentence_node,
                "annotation_set_node": annotation_node,
                "annotation_set_id": str(sets[0].get("ID") or ""),
                "frame_name": str(sets[0]["frameName"]),
                "lexical_unit": str(sets[0].get("luName") or ""),
                "status": str(sets[0].get("status") or ""),
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "sentence": sentence,
                "target_spans": sorted({tuple(value) for value in target_spans}),
                "frame_elements": sorted(frame_elements, key=lambda row: (row["role"], row["spans"], row["text"])),
                "protected_spans": protected_spans,
            }
            annotation["target_spans"] = [list(value) for value in annotation["target_spans"]]
            selection_key = stable_hash(annotation)
            output.append(
                {
                    "selection_key": selection_key,
                    "source_id": "masc-frame:" + selection_key.split(":", 1)[1][:24],
                    "annotation": annotation,
                }
            )
    return output


def independent_frame_concept(frame_name: str) -> str:
    return "framenet." + re.sub(
        r"[^a-z0-9]+", "_", frame_name.lower()
    ).strip("_")


def independent_masc_contextual_frame_priors(
    selected_by_split: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
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
                    "value": {
                        "type": "concept",
                        "value": independent_frame_concept(frame_name),
                    },
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


def independent_masc_assignments(source: dict[str, Any], maximum_characters: int) -> dict[str, dict[str, Any]]:
    archive = resolve(source["archive_path"])
    if sha256_file(archive) != source["content_sha256"]:
        raise ValueError("MASC source hash mismatch")
    root = resolve(source["extracted_root"]) / "data"
    dev = set(source["document_groups"]["private_dev"])
    evaluation = set(source["document_groups"]["private_eval"])
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*-fn.xml")):
        document_id = str(Path(str(path)[: -len("-fn.xml")]).relative_to(root)).replace(os.sep, "/")
        split = "private_dev" if document_id in dev else "private_eval" if document_id in evaluation else "private_train"
        for row in independent_masc_document(path, root):
            if 12 <= len(row["annotation"]["sentence"]) <= maximum_characters:
                by_split[split].append(row)
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, count in source["records_by_split"].items():
        selected = sorted(by_split[split], key=lambda row: row["selection_key"])[: int(count)]
        if len(selected) != int(count):
            raise ValueError(f"MASC independent split reconstruction is incomplete: {split}")
        selected_by_split[split] = selected
    priors = independent_masc_contextual_frame_priors(
        selected_by_split, source["contextual_frame_ambiguity"]
    )
    output: dict[str, dict[str, Any]] = {}
    for split, selected in selected_by_split.items():
        interactions = independent_interaction_predecessors(selected)
        for row in selected:
            annotation = json.loads(json.dumps(row["annotation"]))
            lexical_unit = str(annotation["lexical_unit"]).strip().lower()
            prior = priors.get(lexical_unit)
            if prior is not None and annotation["frame_name"] in {
                alternative["frame_name"] for alternative in prior["alternatives"]
            }:
                annotation["contextual_frame_ambiguity"] = prior
            output[row["source_id"]] = {
                **row,
                "annotation": annotation,
                "split": split,
                "interaction_annotation": interactions.get(row["selection_key"]),
            }
    return output


def independent_masc_composite_assignments(
    masc_rows: dict[str, dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    minimum_frames = int(source["composite_semantic_minimum_frames"])
    maximum_frames = int(source["composite_semantic_maximum_frames"])
    if minimum_frames < 2 or maximum_frames < minimum_frames:
        raise ValueError("MASC composite frame bounds are invalid")
    output: dict[str, dict[str, Any]] = {}
    for split, required in source["composite_semantic_records_by_split"].items():
        by_sentence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in masc_rows.values():
            if row["split"] != split:
                continue
            annotation = row["annotation"]
            by_sentence[
                (annotation["document_id"], annotation["sentence_node"])
            ].append(row)
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
                f"MASC independent composite reconstruction is incomplete: {split}"
            )
        for rows in groups[: int(required)]:
            annotations = [row["annotation"] for row in rows]
            source_ids = [row["source_id"] for row in rows]
            identity = stable_hash(source_ids).split(":", 1)[1]
            source_id = "masc-composite:" + identity[:24]
            frame_receipts = [
                {
                    "node_id": f"k{index}",
                    "claim_id": f"claim-{index + 1}",
                    "annotation_set_node": annotation["annotation_set_node"],
                    "annotation_set_id": annotation["annotation_set_id"],
                    "frame_name": annotation["frame_name"],
                    "lexical_unit": annotation["lexical_unit"],
                    "target_spans": annotation["target_spans"],
                    "frame_roles": sorted(
                        {
                            safe_symbol(element["role"], "ROLE")
                            for element in annotation["frame_elements"]
                        }
                    ),
                    "source_annotation_sha256": stable_hash(annotation),
                }
                for index, annotation in enumerate(annotations)
            ]
            output[source_id] = {
                "source_id": source_id,
                "split": split,
                "component_rows": rows,
                "annotation": {
                    "source_kind": "masc_manual_framenet_composite",
                    "document_id": annotations[0]["document_id"],
                    "sentence_node": annotations[0]["sentence_node"],
                    "sentence": annotations[0]["sentence"],
                    "component_source_ids": source_ids,
                    "frames": annotations,
                    "frame_receipts": frame_receipts,
                    "semantic_claim_scope": source[
                        "composite_semantic_claim_scope"
                    ],
                    "complete_sentence_semantics_claimed": False,
                    "inter_frame_discourse_edges_claimed": False,
                },
            }
    return output


def independent_interaction_predecessors(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reconstruct adjacent-document interaction bindings from raw GrAF rows."""

    documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        documents[row["annotation"]["document_id"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for document_rows in documents.values():
        sentences: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in document_rows:
            annotation = row["annotation"]
            sentences[
                (int(annotation["sentence_start"]), int(annotation["sentence_end"]))
            ].append(row)
        ordered = sorted(sentences.items())
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


def decode_literal(value: Any) -> str:
    if not isinstance(value, dict) or value.get("type") != "byte_literal":
        raise ValueError("expected byte literal")
    return base64.b64decode(str(value.get("value") or ""), validate=True).decode("utf-8")


def expected_masc_value(
    element: dict[str, Any],
    *,
    sentence: str,
    protected_objects: dict[str, dict[str, Any]],
) -> dict[str, str]:
    spans = element["spans"]
    if spans:
        start = min(value[0] for value in spans)
        end = max(value[1] for value in spans)
        if sentence[start:end] == element["text"]:
            for handle, value in protected_objects.items():
                source_span = value.get("source_span") or {}
                if (
                    source_span.get("character_start") == start
                    and source_span.get("character_end") == end
                    and value.get("protection_source") == "explicit_user_or_caller_span"
                ):
                    return {"type": "handle", "value": handle}
    return {"type": "byte_literal", "text": element["text"]}


def observed_masc_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("type") == "handle":
        return {"type": "handle", "value": str(value.get("value") or "")}
    if isinstance(value, dict) and value.get("type") == "concept":
        return {"type": "concept", "value": str(value.get("value") or "")}
    if isinstance(value, dict) and value.get("type") == "ambiguity":
        alternatives = value.get("value")
        if not isinstance(alternatives, list):
            raise ValueError("expected ambiguity alternatives")
        return {
            "type": "ambiguity",
            "value": [
                {
                    "value": observed_masc_value(alternative.get("value")),
                    "probability": float(alternative.get("probability") or 0.0),
                    "evidence": alternative.get("evidence"),
                }
                for alternative in alternatives
            ],
        }
    return {"type": "byte_literal", "text": decode_literal(value)}


def encoded_literal(text: str) -> dict[str, str]:
    return {
        "type": "byte_literal",
        "value": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def expected_dialogue_answer(text: str) -> dict[str, Any]:
    return {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "DIALOGUE_RESPONSE",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [{"role": "CONTENT", "value": encoded_literal(text)}],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored_conversation"},
    }


def expected_behavior_answer(
    text: str, *, disposition: str, ambiguity_id: str
) -> dict[str, Any]:
    return {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "REQUEST_CLARIFICATION" if disposition == "CLARIFY" else "ABSTAIN",
                "modality": "REQUIRED" if disposition == "CLARIFY" else "UNKNOWN",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [{"role": "CONTENT", "value": encoded_literal(text)}],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored_conversation"},
        "decision": {
            "policy": ANSWER_DECISION_POLICY,
            "disposition": disposition,
            "evidence_status": "AMBIGUOUS" if disposition == "CLARIFY" else "INSUFFICIENT_CONTEXT",
            "uncertainty_state": "AMBIGUOUS" if disposition == "CLARIFY" else "INSUFFICIENT_CONTEXT",
            "confidence": 1.0,
            "controlling_claim_ids": ["claim-1"],
            "unresolved_ambiguity_ids": [ambiguity_id] if disposition == "CLARIFY" else [],
        },
    }


def independent_hrl_replay(
    *,
    split: str,
    source_id: str,
    source_group: str,
    source_text: str,
    surface_target: str,
    source_annotation: dict[str, Any],
    interaction_annotation: dict[str, Any] | None,
    interaction_entries: list[dict[str, Any]],
    actor_id: str,
    source: dict[str, Any],
    valid_realizations: list[dict[str, Any]] | None,
    concept_capsules: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "kerc-corpus-" + identity[:24],
        scope={
            "user": "project-theseus-corpus",
            "project": "theseus",
            "conversation": identity[:24],
            "privacy": "private_local",
        },
    )
    deltas: list[dict[str, Any]] = []
    if interaction_entries:
        operations = [
            {
                "op": "OVERRIDE",
                "segment_id": str(entry["segment_id"]),
                "key": str(entry["key"]),
                "value": entry["value"],
                "privacy": "interaction_private",
            }
            for entry in interaction_entries
        ]
        state, delta = apply_hierarchical_residual_delta(
            state,
            operations,
            expected_state_hash=state["state_hash"],
            actor_authority="document",
            actor_id=actor_id,
            provenance={
                "source": source["dataset_id"],
                "interaction_annotation_sha256": stable_hash(interaction_annotation),
            },
        )
        deltas.append(delta)
    return state, deltas


def safe_symbol(value: str, prefix: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    if not symbol or not symbol[0].isalpha():
        symbol = prefix + "_" + symbol
    return symbol[:96]


def independent_semantic_concept(namespace: str, value: str) -> dict[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", ".", value.lower()).strip(".") or "unknown"
    return {"type": "concept", "value": f"{namespace}.{normalized}"}


def independent_decision_value(layer: str, field: str, value: str) -> dict[str, Any]:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return {"type": "boolean", "value": lowered == "true"}
    if field == "nested-source":
        return {
            "type": "list",
            "value": [
                independent_semantic_concept("mpqa.source", part)
                for part in value.split(",") if part.strip()
            ],
        }
    categorical = {
        "annotation-uncertain", "attitude-type", "attitude-uncertain", "contrast",
        "es-uncertain", "expression-intensity", "implicit", "inferred", "insubstantial",
        "intensity", "polarity", "repetition", "sarcastic", "subjective-uncertain",
        "target-uncertain",
    }
    if field in categorical:
        return independent_semantic_concept(
            f"{layer}.{safe_symbol(field, 'field').lower()}", value
        )
    return encoded_literal(value)


def independent_decision_modality(annotation: dict[str, Any]) -> str:
    if annotation["layer"] == "cb":
        return "ASSERTED" if annotation["label"].startswith("Committed Belief") else "POSSIBLE"
    uncertain = any(
        "uncertain" in key and value.lower() not in {"", "false", "no"}
        for key, value in annotation["fields"].items()
    )
    return "POSSIBLE" if uncertain else "ASSERTED"


def independent_decision_polarity(annotation: dict[str, Any]) -> str:
    explicit = str(annotation["fields"].get("polarity") or "").lower()
    if annotation["label"].lower().startswith("not ") or explicit in {"negative", "neg", "both"}:
        return "NEGATED"
    if explicit in {"positive", "pos"}:
        return "AFFIRMED"
    return "UNKNOWN" if annotation["layer"] == "mpqa" else "AFFIRMED"


def independent_decision_arguments(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    output = [
        {
            "role": "ANNOTATION_KIND",
            "value": independent_semantic_concept(annotation["layer"], annotation["label"]),
        },
        {"role": "SOURCE_EXPRESSION", "value": encoded_literal(annotation["text"])},
    ]
    if annotation["layer"] == "cb":
        output.append(
            {
                "role": "TEMPORAL_ORIENTATION",
                "value": independent_semantic_concept(
                    "time", "future" if annotation["label"].endswith("Future") else "non_future"
                ),
            }
        )
    for field, value in sorted(annotation["fields"].items()):
        if field != "id" and value:
            output.append(
                {
                    "role": safe_symbol(field, "FIELD"),
                    "value": independent_decision_value(annotation["layer"], field, value),
                }
            )
    return output


def verify_dolly_record(record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    annotation = record.get("source_annotation")
    if annotation != expected["annotation"]:
        raise ValueError("Dolly source annotation replay mismatch")
    if record.get("split") != expected["split"] or record.get("source_text") != expected["prompt"] or record.get("surface_target") != expected["target"]:
        raise ValueError("Dolly split or surface replay mismatch")
    authority = record.get("semantic_supervision", {}).get("objective_authority")
    expected_authority = {objective: objective == "surface_direct_control_v1" for objective in TRAINING_OBJECTIVES}
    if authority != expected_authority:
        raise ValueError("Dolly objective authority exceeds source evidence")
    node = record["kernel_packet"]["program"]["nodes"][0]
    claim = record["answer_packet"]["claims"][0]
    if node.get("operator") != "RESPOND" or decode_literal(node["arguments"][0]["value"]) != expected["prompt"]:
        raise ValueError("Dolly packet task binding mismatch")
    if claim.get("predicate") != "RESPOND" or decode_literal(claim["arguments"][0]["value"]) != expected["target"]:
        raise ValueError("Dolly answer binding mismatch")
    residual = record["kernel_packet"].get("residual") or {}
    if residual.get("segment_frame") or residual.get("token_tags"):
        raise ValueError("Dolly record received unsupported semantic residual annotations")
    context = str(expected["annotation"].get("context") or "").strip()
    interaction = (
        {
            "kind": "licensed_document_context",
            "source_row_sha256": expected["annotation"]["source_row_sha256"],
            "context_sha256": stable_hash(context.encode("utf-8")),
            "context": context,
            "source_category": expected["annotation"]["category"],
        }
        if context
        else None
    )
    entries = (
        [{"segment_id": "document_context", "key": "content", "value": context}]
        if context
        else []
    )
    if record.get("interaction_annotation") != interaction:
        raise ValueError("Dolly document-context annotation replay mismatch")
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=record["provenance"]["source_id"],
        source_group=record["provenance"]["source_group"],
        source_text=expected["prompt"],
        surface_target=expected["target"],
        source_annotation=expected["annotation"],
        interaction_annotation=interaction,
        interaction_entries=entries,
        actor_id="databricks_dolly_human_context",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("Dolly VCM document-context replay mismatch")
    expected_label = 1 if context else 0
    if (record.get("residual_supervision") or {}).get("labels_by_channel", {}).get(
        "interaction"
    ) != expected_label:
        raise ValueError("Dolly interaction residual label mismatch")
    return {
        "source_row_sha256": expected["annotation"]["source_row_sha256"],
        "document_context_bound": bool(context),
        "source_category": expected["annotation"]["category"],
    }


def verify_dolly_grounded_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Verify extractive support and KERC bindings directly from the pinned source."""

    annotation = expected["annotation"]
    if record.get("source_annotation") != annotation:
        raise ValueError("Dolly grounded-question annotation replay mismatch")
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != expected["prompt"]
        or record.get("surface_target") != expected["target"]
        or record.get("provenance", {}).get("source_group")
        != expected["source_group"]
    ):
        raise ValueError("Dolly grounded-question split or source replay mismatch")
    authority = record.get("semantic_supervision", {}).get("objective_authority")
    if authority != {objective: True for objective in TRAINING_OBJECTIVES}:
        raise ValueError("Dolly grounded-question objective authority mismatch")
    context = expected["context"]
    start, end = (int(value) for value in annotation["answer_span"])
    if (
        annotation.get("support_relation") != "unique_contiguous_exact_span"
        or annotation.get("support_claim_scope") != "extractive_source_support_only"
        or annotation.get("broad_entailment_or_truth_claimed") is not False
        or context.count(expected["target"]) != 1
        or context[start:end] != expected["target"]
    ):
        raise ValueError("Dolly grounded-question support relation mismatch")
    context_sha256 = stable_hash(context.encode("utf-8"))
    node = record["kernel_packet"]["program"]["nodes"][0]
    arguments = {
        str(argument.get("role") or ""): decode_literal(argument.get("value"))
        for argument in node.get("arguments") or []
    }
    if (
        node.get("operator") != "ANSWER_FROM_CONTEXT"
        or arguments
        != {
            "QUESTION": expected["prompt"],
            "QUESTION_FORM": expected["question_form"],
            "CONTEXT_SHA256": context_sha256,
        }
    ):
        raise ValueError("Dolly grounded-question Kernel program replay mismatch")
    answer = record.get("answer_packet") or {}
    claims = answer.get("claims") if isinstance(answer.get("claims"), list) else []
    claim = claims[0] if len(claims) == 1 else {}
    claim_arguments = {
        str(argument.get("role") or ""): decode_literal(argument.get("value"))
        for argument in claim.get("arguments") or []
    }
    if (
        claim.get("claim_id") != "claim-1"
        or claim.get("predicate") != "SUPPORTED_ANSWER"
        or claim_arguments
        != {
            "ANSWER_SPAN": expected["target"],
            "CONTEXT_SHA256": context_sha256,
        }
        or answer.get("decision")
        != {
            "policy": "project_theseus_kernel_answer_decision_v1",
            "disposition": "ANSWER",
            "evidence_status": "SUPPORTED",
            "uncertainty_state": "RESOLVED",
            "confidence": 1.0,
            "controlling_claim_ids": ["claim-1"],
            "unresolved_ambiguity_ids": [],
        }
    ):
        raise ValueError("Dolly grounded-question answer decision replay mismatch")
    interaction = {
        "kind": "licensed_grounded_question_context",
        "policy": "project_theseus_dolly_unique_extractive_question_support_v1",
        "source_row_sha256": annotation["source_row_sha256"],
        "context_sha256": context_sha256,
        "answer_span": [start, end],
        "support_relation": "unique_contiguous_exact_span",
        "support_claim_scope": "extractive_source_support_only",
    }
    entries = [
        {"segment_id": "document_context", "key": "content", "value": context},
        {
            "segment_id": "question_contract",
            "key": "context_sha256",
            "value": context_sha256,
        },
    ]
    if record.get("interaction_annotation") != interaction:
        raise ValueError("Dolly grounded-question interaction replay mismatch")
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=record["provenance"]["source_id"],
        source_group=expected["source_group"],
        source_text=expected["prompt"],
        surface_target=expected["target"],
        source_annotation=annotation,
        interaction_annotation=interaction,
        interaction_entries=entries,
        actor_id="databricks_dolly_grounded_question",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("Dolly grounded-question VCM replay mismatch")
    if (record.get("residual_supervision") or {}).get("labels_by_channel", {}).get(
        "interaction"
    ) != 1:
        raise ValueError("Dolly grounded-question interaction label mismatch")
    return {
        "source_row_sha256": annotation["source_row_sha256"],
        "question_form": expected["question_form"],
        "answer_span": [start, end],
        "support_relation": "unique_contiguous_exact_span",
        "claim_scope": "extractive_source_support_only",
    }


def verify_oasst_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    if record.get("source_annotation") != expected["annotation"]:
        raise ValueError("OASST2 source-tree annotation replay mismatch")
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != expected["prompt"]
        or record.get("surface_target") != expected["targets"][0]
        or record.get("provenance", {}).get("source_group") != expected["source_group"]
    ):
        raise ValueError("OASST2 split, source, or canonical target replay mismatch")
    allowed = set(source["allowed_objectives"])
    authority = record.get("semantic_supervision", {}).get("objective_authority")
    if authority != {objective: objective in allowed for objective in TRAINING_OBJECTIVES}:
        raise ValueError("OASST2 objective authority exceeds reviewed tree evidence")
    node = record["kernel_packet"]["program"]["nodes"][0]
    if (
        node.get("operator") != "DIALOGUE_RESPOND"
        or decode_literal(node["arguments"][0]["value"]) != expected["prompt"]
    ):
        raise ValueError("OASST2 dialogue Kernel program replay mismatch")
    valid_realizations = [
        {
            "realization_id": f"oasst-rank-{rank}",
            "surface_target": target,
            "answer_packet": expected_dialogue_answer(target),
            "source_rank": rank,
            "human_source_bound": True,
            "source_message_sha256": expected["annotation"]["responses"][rank][
                "source_message_sha256"
            ],
        }
        for rank, target in enumerate(expected["targets"])
    ]
    if record.get("valid_realizations") != valid_realizations:
        raise ValueError("OASST2 valid-realization set replay mismatch")
    if record.get("answer_packet") != valid_realizations[0]["answer_packet"]:
        raise ValueError("OASST2 canonical answer packet replay mismatch")
    interaction = {
        "kind": "reviewed_conversation_ancestry",
        "turns": expected["interaction_turns"],
    }
    entries = [
        {
            "segment_id": f"turn_{turn['turn_index']:03d}",
            "key": key,
            "value": turn[key],
        }
        for turn in expected["interaction_turns"]
        for key in ("role", "content")
    ]
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=record["provenance"]["source_id"],
        source_group=expected["source_group"],
        source_text=expected["prompt"],
        surface_target=expected["targets"][0],
        source_annotation=expected["annotation"],
        interaction_annotation=interaction,
        interaction_entries=entries,
        actor_id="openassistant_oasst2_reviewed_tree",
        source=source,
        valid_realizations=valid_realizations,
    )
    if (
        record.get("interaction_annotation") != interaction
        or record.get("hrl_state") != state
        or record.get("hrl_deltas") != deltas
    ):
        raise ValueError("OASST2 VCM conversation-state replay mismatch")
    if (record.get("residual_supervision") or {}).get("labels_by_channel", {}).get(
        "interaction"
    ) != 1:
        raise ValueError("OASST2 interaction residual label mismatch")
    return {
        "message_tree_sha256": expected["annotation"]["message_tree_sha256"],
        "context_turn_count": len(expected["interaction_turns"]),
        "human_valid_realization_count": len(valid_realizations),
    }


def verify_oasst_behavior_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    if record.get("source_annotation") != expected["annotation"]:
        raise ValueError("OASST2 behavior annotation replay mismatch")
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != expected["prompt"]
        or record.get("surface_target") != expected["target"]
        or record.get("provenance", {}).get("source_group") != expected["source_group"]
    ):
        raise ValueError("OASST2 behavior split or source replay mismatch")
    allowed = set(source["allowed_objectives"])
    authority = record.get("semantic_supervision", {}).get("objective_authority")
    if authority != {objective: objective in allowed for objective in TRAINING_OBJECTIVES}:
        raise ValueError("OASST2 behavior objective authority mismatch")
    node = record["kernel_packet"]["program"]["nodes"][0]
    if (
        node.get("operator") != "DIALOGUE_RESPOND"
        or decode_literal(node["arguments"][0]["value"]) != expected["prompt"]
    ):
        raise ValueError("OASST2 behavior Kernel program replay mismatch")
    ambiguity_id = "amb-" + expected["selection_key"].split(":", 1)[1][:16]
    answer = expected_behavior_answer(
        expected["target"],
        disposition=expected["disposition"],
        ambiguity_id=ambiguity_id,
    )
    realization = {
        "realization_id": "oasst-explicit-behavior",
        "surface_target": expected["target"],
        "answer_packet": answer,
        "source_rank": 0,
        "human_source_bound": True,
        "source_message_sha256": expected["annotation"]["response_message_sha256"],
    }
    if record.get("answer_packet") != answer or record.get("valid_realizations") != [realization]:
        raise ValueError("OASST2 behavior answer replay mismatch")
    interaction = {
        "kind": "reviewed_conversation_ancestry",
        "turns": expected["interaction_turns"],
    }
    entries = [
        {
            "segment_id": f"turn_{turn['turn_index']:03d}",
            "key": key,
            "value": turn[key],
        }
        for turn in expected["interaction_turns"]
        for key in ("role", "content")
    ]
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=record["provenance"]["source_id"],
        source_group=expected["source_group"],
        source_text=expected["prompt"],
        surface_target=expected["target"],
        source_annotation=expected["annotation"],
        interaction_annotation=interaction,
        interaction_entries=entries,
        actor_id="openassistant_oasst2_reviewed_tree",
        source=source,
        valid_realizations=[realization],
    )
    if (
        record.get("interaction_annotation") != interaction
        or record.get("hrl_state") != state
        or record.get("hrl_deltas") != deltas
    ):
        raise ValueError("OASST2 behavior VCM replay mismatch")
    return {
        "message_tree_sha256": expected["annotation"]["message_tree_sha256"],
        "disposition": expected["disposition"],
        "matched_rule_id": expected["annotation"]["matched_rule_id"],
        "claim_scope": "explicit_human_surface_behavior_only",
    }


def expected_masc_arguments(
    annotation: dict[str, Any],
    protected_objects: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    arguments = [
        {
            "role": safe_symbol(element["role"], "ROLE"),
            "value": expected_masc_value(
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
                            "value": alternative["value"],
                            "probability": float(alternative["probability"]),
                            "evidence": alternative["evidence"],
                        }
                        for alternative in contextual["alternatives"]
                    ],
                },
            }
        )
    return arguments


def verify_masc_decision_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    annotation = {
        **expected["annotation"],
        "semantic_claim_scope": source["decision_semantic_claim_scope"],
        "complete_sentence_semantics_claimed": False,
        "truth_claimed": False,
        "event_coreference_grouping_claimed": False,
        "source_declared_cross_annotation_links_resolved": False,
    }
    if record.get("source_annotation") != annotation:
        raise ValueError("MASC decision-semantic raw annotation replay mismatch")
    sentence = annotation["sentence"]
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != sentence
        or record.get("surface_target") != sentence
        or record.get("provenance", {}).get("source_group")
        != "masc-document:" + annotation["document_id"]
    ):
        raise ValueError("MASC decision-semantic split or source replay mismatch")
    semantic = record.get("semantic_supervision") or {}
    allowed = set(source["allowed_objectives"])
    if semantic.get("objective_authority") != {
        objective: objective in allowed for objective in TRAINING_OBJECTIVES
    }:
        raise ValueError("MASC decision-semantic objective authority mismatch")
    if (
        semantic.get("unique_source_credit")
        != int(source["decision_semantic_unique_source_credit"])
        or semantic.get("decision_semantic_authority")
        != "manual_masc_cb_mpqa_event_annotations"
    ):
        raise ValueError("MASC decision-semantic source-credit mismatch")
    nodes = record["kernel_packet"]["program"]["nodes"]
    claims = record["answer_packet"]["claims"]
    if len(nodes) != len(annotation["annotations"]) or len(claims) != len(nodes):
        raise ValueError("MASC decision-semantic cardinality mismatch")
    expected_tags = []
    typed_nonliteral = 0
    for index, item in enumerate(annotation["annotations"]):
        predicate = (
            "EPISTEMIC_STATUS" if item["layer"] == "cb" else
            "EVENT_" + safe_symbol(item["label"], "UNKNOWN") if item["layer"] == "event" else
            "SUBJECTIVITY_" + safe_symbol(item["label"], "UNKNOWN")
        )
        arguments = independent_decision_arguments(item)
        typed_nonliteral += sum(arg["value"]["type"] != "byte_literal" for arg in arguments)
        common = {
            "predicate": predicate,
            "modality": independent_decision_modality(item),
            "polarity": independent_decision_polarity(item),
            "quantifier": "NONE",
            "confidence": 1.0,
            "arguments": arguments,
        }
        expected_node = {
            "node_id": f"k{index}",
            "operator": predicate,
            "modality": common["modality"],
            "polarity": common["polarity"],
            "quantifier": "NONE",
            "confidence": 1.0,
            "derivation": "preserved",
            "source_spans": [[int(item["start"]), int(item["end"])]],
            "arguments": arguments,
        }
        if nodes[index] != expected_node:
            raise ValueError("MASC decision-semantic Kernel node replay mismatch")
        if claims[index] != {"claim_id": f"claim-{index + 1}", **common}:
            raise ValueError("MASC decision-semantic answer claim replay mismatch")
        expected_tags.append(
            {
                "tag": f"{item['layer'].upper()}:{safe_symbol(item['label'], 'UNKNOWN')}",
                "source_span": [int(item["start"]), int(item["end"])],
                "authority": "licensed_manual_annotation",
            }
        )
    if record["kernel_packet"]["program"]["roots"] != [f"k{i}" for i in range(len(nodes))]:
        raise ValueError("MASC decision-semantic root replay mismatch")
    if record["kernel_packet"]["residual"]["token_tags"] != sorted(
        expected_tags, key=lambda row: (row["source_span"], row["tag"], row["authority"])
    ):
        raise ValueError("MASC decision-semantic token-tag replay mismatch")
    source_id = record["provenance"]["source_id"]
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=source_id,
        source_group="masc-document:" + annotation["document_id"],
        source_text=sentence,
        surface_target=sentence,
        source_annotation=annotation,
        interaction_annotation=None,
        interaction_entries=[],
        actor_id="licensed_source_context",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("MASC decision-semantic VCM replay mismatch")
    return {
        "policy": MASC_DECISION_SEMANTICS_POLICY,
        "annotation_count": len(nodes),
        "typed_nonliteral_argument_count": typed_nonliteral,
        "layers": sorted({item["layer"] for item in annotation["annotations"]}),
        "claim_scope": source["decision_semantic_claim_scope"],
        "complete_sentence_semantics_claimed": False,
        "truth_claimed": False,
    }


def independent_event_type_concept(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return "masc.event_type." + (normalized or "unknown")


def verify_masc_event_coreference_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    contract = source["event_coreference"]
    annotation = json.loads(json.dumps(expected["annotation"]))
    annotation["semantic_claim_scope"] = contract["claim_scope"]
    if record.get("source_annotation") != annotation:
        raise ValueError("MASC event-coreference raw annotation replay mismatch")
    source_text = annotation["source_text"]
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != source_text
        or record.get("surface_target") != source_text
        or record.get("provenance", {}).get("source_id") != expected["source_id"]
        or record.get("provenance", {}).get("source_group")
        != "masc-document:" + annotation["document_id"]
    ):
        raise ValueError("MASC event-coreference source or split mismatch")
    semantic = record.get("semantic_supervision") or {}
    allowed = set(source["allowed_objectives"])
    if semantic.get("objective_authority") != {
        objective: objective in allowed for objective in TRAINING_OBJECTIVES
    }:
        raise ValueError("MASC event-coreference objective authority mismatch")
    if (
        semantic.get("unique_source_credit") != int(contract["unique_source_credit"])
        or semantic.get("event_coreference_authority")
        != "manual_named_gate_annotation_set_membership"
    ):
        raise ValueError("MASC event-coreference authority mismatch")
    expected_nodes: list[dict[str, Any]] = []
    expected_claims: list[dict[str, Any]] = []
    expected_mentions: list[dict[str, Any]] = []
    expected_tags: list[dict[str, Any]] = []
    for index, mention in enumerate(annotation["mentions"]):
        node_id = f"k{index}"
        claim_id = f"claim-{index + 1}"
        event_type = independent_event_type_concept(mention["event_type"])
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
        expected_nodes.append(
            {
                "node_id": node_id,
                "operator": "EVENT_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": mention["target_spans"],
                "arguments": arguments,
            }
        )
        expected_claims.append(
            {
                "claim_id": claim_id,
                "predicate": "EVENT_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": json.loads(json.dumps(arguments)),
            }
        )
        expected_mentions.append(
            {
                "node_id": node_id,
                "claim_id": claim_id,
                "event_type": event_type,
                "target_spans": mention["target_spans"],
                "source_annotation_sha256": mention[
                    "source_annotation_sha256"
                ],
            }
        )
        for target_span in mention["target_spans"]:
            expected_tags.append(
                {
                    "tag": "EVENT_COREFERENCE:"
                    + safe_symbol(annotation["annotation_set_name"], "GROUP"),
                    "source_span": target_span,
                    "authority": "licensed_manual_annotation",
                }
            )
    group_node_id = f"k{len(expected_nodes)}"
    group_claim_id = f"claim-{len(expected_claims) + 1}"
    all_spans = sorted(
        {
            tuple(span)
            for mention in annotation["mentions"]
            for span in mention["target_spans"]
        }
    )
    expected_nodes.append(
        {
            "node_id": group_node_id,
            "operator": "EVENT_COREFERENCE_GROUP",
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "derivation": "preserved",
            "source_spans": [list(span) for span in all_spans],
            "arguments": [
                {
                    "role": "GROUP_ID",
                    "value": {
                        "type": "concept",
                        "value": annotation["group_concept"],
                    },
                },
                {
                    "role": "MEMBERS",
                    "value": {
                        "type": "list",
                        "value": [
                            {"type": "node_ref", "value": f"k{index}"}
                            for index in range(len(annotation["mentions"]))
                        ],
                    },
                },
            ],
        }
    )
    expected_claims.append(
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
                                "value": independent_event_type_concept(
                                    mention["event_type"]
                                ),
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
    program = record["kernel_packet"]["program"]
    if program["roots"] != [group_node_id] or program["nodes"] != expected_nodes:
        raise ValueError("MASC event-coreference Kernel graph replay mismatch")
    if record["answer_packet"]["claims"] != expected_claims:
        raise ValueError("MASC event-coreference answer graph replay mismatch")
    expected_segment = {
        "schema": "event_coreference_group_v1",
        "group_id": annotation["group_concept"],
        "group_node_id": group_node_id,
        "group_claim_id": group_claim_id,
        "mentions": expected_mentions,
    }
    if record["kernel_packet"]["residual"]["segment_frame"] != expected_segment:
        raise ValueError("MASC event-coreference residual group replay mismatch")
    expected_tags.sort(key=lambda row: (row["source_span"], row["tag"]))
    if record["kernel_packet"]["residual"]["token_tags"] != expected_tags:
        raise ValueError("MASC event-coreference mention tag replay mismatch")
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=expected["source_id"],
        source_group="masc-document:" + annotation["document_id"],
        source_text=source_text,
        surface_target=source_text,
        source_annotation=annotation,
        interaction_annotation=None,
        interaction_entries=[],
        actor_id="licensed_source_context",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("MASC event-coreference VCM replay mismatch")
    return {
        "policy": MASC_EVENT_COREFERENCE_POLICY,
        "group_concept": annotation["group_concept"],
        "mention_count": len(annotation["mentions"]),
        "complete_group_alignment": True,
        "cooccurrence_inferred_relation_count": 0,
        "claim_scope": contract["claim_scope"],
    }


def independent_mpqa_member_concept(
    member_type: str, member: dict[str, Any]
) -> str:
    identity = str(member.get("annotation_id") or member.get("annotation_line_id") or "")
    fragment = re.sub(r"[^a-z0-9]+", "_", identity.lower()).strip("_")
    if member_type == "source" and identity == "w":
        return "mpqa.source.w"
    receipt_suffix = str(member["source_annotation_sha256"]).partition(":")[2][:12]
    return f"mpqa.{member_type}.{(fragment or 'unnamed')[:48]}.{receipt_suffix}"


def independent_mpqa_span_status(member: dict[str, Any]) -> str:
    if list(member.get("target_spans") or []):
        return "explicit"
    fields = member.get("fields") if isinstance(member.get("fields"), dict) else {}
    identity = str(member.get("annotation_id") or "").lower()
    if (
        str(fields.get("implicit") or "").lower() == "true"
        or identity in {"w", "implicit"}
        or member.get("node_type") == "implicit-writer"
    ):
        return "declared_implicit"
    return "zero_width_annotation"


def verify_masc_mpqa_relation_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    contract = source["mpqa_relations"]
    annotation = json.loads(json.dumps(expected["annotation"]))
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
    if record.get("source_annotation") != annotation:
        raise ValueError("MASC MPQA-relation raw annotation replay mismatch")
    source_text = annotation["source_text"]
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != source_text
        or record.get("surface_target") != source_text
        or record.get("provenance", {}).get("source_id") != expected["source_id"]
        or record.get("provenance", {}).get("source_group")
        != "masc-document:" + annotation["document_id"]
    ):
        raise ValueError("MASC MPQA-relation source or split mismatch")
    semantic = record.get("semantic_supervision") or {}
    allowed = {
        "surface_to_kernel_program_v1",
        "kernel_program_to_answer_packet_v1",
        "answer_packet_to_surface_v1",
    }
    if semantic.get("objective_authority") != {
        objective: objective in allowed for objective in TRAINING_OBJECTIVES
    }:
        raise ValueError("MASC MPQA-relation objective authority mismatch")
    if (
        semantic.get("unique_source_credit") != int(contract["unique_source_credit"])
        or semantic.get("mpqa_relation_authority")
        != "manual_complete_expression_attitude_target_source_links"
    ):
        raise ValueError("MASC MPQA-relation authority mismatch")

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
        if receipt not in seen_receipts:
            seen_receipts.add(receipt)
            unique_members.append((member_type, member))
    receipt_to_node = {
        str(member["source_annotation_sha256"]): f"k{index}"
        for index, (_, member) in enumerate(unique_members)
    }
    concepts = {
        str(member["source_annotation_sha256"]): independent_mpqa_member_concept(
            member_type, member
        )
        for member_type, member in unique_members
    }
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_edges = []
    for edge in annotation["edges"]:
        normalized = {
            "edge_type": str(edge["edge_type"]),
            "from_node_id": receipt_to_node[str(edge["from"])],
            "to_node_id": receipt_to_node[str(edge["to"])],
            "order": int(edge.get("order", -1)),
            "source_field": str(edge["manual_field"]),
        }
        outgoing[normalized["from_node_id"]].append(normalized)
        expected_edges.append(normalized)
    expected_nodes = []
    expected_claims = []
    expected_members = []
    expected_tags = []
    for index, (member_type, member) in enumerate(unique_members):
        node_id = f"k{index}"
        claim_id = f"claim-{index + 1}"
        receipt = str(member["source_annotation_sha256"])
        concept_id = concepts[receipt]
        span_status = independent_mpqa_span_status(member)
        arguments = [
            {
                "role": "MEMBER_CONCEPT",
                "value": {"type": "concept", "value": concept_id},
            },
            {
                "role": "RELATION_ID",
                "value": {"type": "concept", "value": annotation["relation_concept"]},
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
            role = "LINK_" + safe_symbol(edge["edge_type"], "EDGE")
            if edge["edge_type"] == "nested_source_member":
                role += "_" + str(edge["order"])
            arguments.append(
                {
                    "role": role,
                    "value": {"type": "node_ref", "value": edge["to_node_id"]},
                }
            )
        predicate = "MPQA_" + member_type.upper()
        expected_nodes.append(
            {
                "node_id": node_id,
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": member["target_spans"],
                "arguments": arguments,
            }
        )
        expected_claims.append(
            {
                "claim_id": claim_id,
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
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
                ],
            }
        )
        expected_members.append(
            {
                "node_id": node_id,
                "claim_id": claim_id,
                "member_type": member_type,
                "concept_id": concept_id,
                "target_spans": member["target_spans"],
                "source_annotation_sha256": receipt,
                "implicit": span_status == "declared_implicit",
                "span_status": span_status,
            }
        )
        for span in member["target_spans"]:
            expected_tags.append(
                {
                    "tag": "MPQA_RELATION_" + member_type.upper(),
                    "source_span": span,
                    "authority": "licensed_manual_annotation",
                }
            )
    expression_node = receipt_to_node[
        str(annotation["expression"]["source_annotation_sha256"])
    ]
    program = record["kernel_packet"]["program"]
    if program["roots"] != [expression_node] or program["nodes"] != expected_nodes:
        raise ValueError("MASC MPQA-relation Kernel graph replay mismatch")
    if record["answer_packet"]["claims"] != expected_claims:
        raise ValueError("MASC MPQA-relation answer graph replay mismatch")
    expected_members.sort(key=lambda row: int(row["node_id"][1:]))
    expected_edges.sort(
        key=lambda row: (
            row["edge_type"],
            int(row["from_node_id"][1:]),
            row["order"],
            int(row["to_node_id"][1:]),
        )
    )
    expected_segment = {
        "schema": "mpqa_relation_chain_v1",
        "relation_id": annotation["relation_concept"],
        "members": expected_members,
        "edges": expected_edges,
    }
    if record["kernel_packet"]["residual"]["segment_frame"] != expected_segment:
        raise ValueError("MASC MPQA-relation residual graph replay mismatch")
    expected_tags.sort(key=lambda row: (row["source_span"], row["tag"], row["authority"]))
    if record["kernel_packet"]["residual"]["token_tags"] != expected_tags:
        raise ValueError("MASC MPQA-relation token-tag replay mismatch")
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=expected["source_id"],
        source_group="masc-document:" + annotation["document_id"],
        source_text=source_text,
        surface_target=source_text,
        source_annotation=annotation,
        interaction_annotation=None,
        interaction_entries=[],
        actor_id="licensed_source_context",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("MASC MPQA-relation VCM replay mismatch")
    return {
        "policy": MASC_MPQA_RELATION_POLICY,
        "relation_id": annotation["relation_concept"],
        "member_count": len(expected_members),
        "edge_count": len(expected_edges),
        "complete_relation_alignment": True,
        "inferred_relation_count": 0,
        "claim_scope": contract["claim_scope"],
    }


def verify_masc_composite_record(
    record: dict[str, Any],
    source: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    annotation = record.get("source_annotation")
    if annotation != expected["annotation"]:
        raise ValueError("MASC composite raw-annotation replay mismatch")
    sentence = annotation["sentence"]
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != sentence
        or record.get("surface_target") != sentence
    ):
        raise ValueError("MASC composite split or sentence replay mismatch")
    semantic = record.get("semantic_supervision") or {}
    allowed = set(source["allowed_objectives"])
    if semantic.get("objective_authority") != {
        objective: objective in allowed for objective in TRAINING_OBJECTIVES
    }:
        raise ValueError("MASC composite objective authority mismatch")
    if (
        semantic.get("unique_source_credit")
        != int(source["composite_semantic_unique_source_credit"])
        or semantic.get("composite_semantic_authority")
        != "manual_framenet_multiple_annotations_same_source_sentence"
    ):
        raise ValueError("MASC composite source-credit contract mismatch")

    frames = annotation["frames"]
    protected_objects = record["kernel_packet"].get("protected_objects") or {}
    expected_explicit = {
        (span["start"], span["end"], span["object_type"], span["copy_policy"])
        for span in (frames[0].get("protected_spans") or [])
    }
    observed_explicit = {
        (
            value["source_span"]["character_start"],
            value["source_span"]["character_end"],
            value["object_type"],
            value["copy_policy"],
        )
        for value in protected_objects.values()
        if value.get("protection_source") == "explicit_user_or_caller_span"
    }
    if observed_explicit != expected_explicit:
        raise ValueError("MASC composite protected-object replay mismatch")

    program = record["kernel_packet"]["program"]
    claims = record["answer_packet"]["claims"]
    expected_roots = [f"k{index}" for index in range(len(frames))]
    if program.get("roots") != expected_roots:
        raise ValueError("MASC composite root replay mismatch")
    if len(program.get("nodes") or []) != len(frames) or len(claims) != len(frames):
        raise ValueError("MASC composite frame cardinality mismatch")
    for index, annotation_frame in enumerate(frames):
        node = program["nodes"][index]
        claim = claims[index]
        predicate = "FRAME_" + safe_symbol(
            annotation_frame["frame_name"], "UNKNOWN"
        )
        arguments = expected_masc_arguments(annotation_frame, protected_objects)
        observed_node_arguments = [
            {"role": value.get("role"), "value": observed_masc_value(value.get("value"))}
            for value in node.get("arguments") or []
        ]
        observed_claim_arguments = [
            {"role": value.get("role"), "value": observed_masc_value(value.get("value"))}
            for value in claim.get("arguments") or []
        ]
        if (
            node.get("node_id") != f"k{index}"
            or node.get("operator") != predicate
            or node.get("source_spans") != annotation_frame["target_spans"]
            or node.get("derivation") != "preserved"
            or observed_node_arguments != arguments
        ):
            raise ValueError("MASC composite kernel-node replay mismatch")
        if (
            claim.get("claim_id") != f"claim-{index + 1}"
            or claim.get("predicate") != predicate
            or observed_claim_arguments != arguments
        ):
            raise ValueError("MASC composite answer-claim replay mismatch")
    expected_terms = [
        {
            "concept": independent_frame_concept(frame["frame_name"]),
            "surface_policy": "source_licensed_frame",
        }
        for frame in frames
    ]
    if record["answer_packet"].get("required_terms") != expected_terms:
        raise ValueError("MASC composite required-term replay mismatch")

    expected_segment = {
        "schema": "framenet_composite_v1",
        "frames": annotation["frame_receipts"],
    }
    expected_tags = [
        {
            "tag": "FRAME_TARGET:" + safe_symbol(frame["frame_name"], "UNKNOWN"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for frame in frames
        for span in frame["target_spans"]
    ]
    expected_tags.extend(
        {
            "tag": "FRAME_ROLE:" + safe_symbol(element["role"], "ROLE"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for frame in frames
        for element in frame["frame_elements"]
        for span in element["spans"]
    )
    expected_tags.extend(
        {
            "tag": "ENTITY:" + str(span["object_type"]),
            "source_span": [int(span["start"]), int(span["end"])],
            "authority": "licensed_manual_annotation",
        }
        for span in (frames[0].get("protected_spans") or [])
    )
    expected_tags.sort(key=lambda value: (value["source_span"], value["tag"]))
    residual = record["kernel_packet"].get("residual") or {}
    if residual.get("segment_frame") != expected_segment:
        raise ValueError("MASC composite segment replay mismatch")
    if residual.get("token_tags") != expected_tags:
        raise ValueError("MASC composite token replay mismatch")
    labels = (record.get("residual_supervision") or {}).get("labels_by_channel") or {}
    if labels.get("interaction") != 0 or labels.get("segment") != 1 or labels.get("token") != 2:
        raise ValueError("MASC composite residual labels mismatch")
    if record.get("interaction_annotation") is not None:
        raise ValueError("MASC composite received an invented interaction")
    expected_state, expected_deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=record["provenance"]["source_id"],
        source_group=record["provenance"]["source_group"],
        source_text=sentence,
        surface_target=sentence,
        source_annotation=annotation,
        interaction_annotation=None,
        interaction_entries=[],
        actor_id="masc_manual_framenet",
        source=source,
        valid_realizations=None,
    )
    if record.get("hrl_state") != expected_state or record.get("hrl_deltas") != expected_deltas:
        raise ValueError("MASC composite VCM replay mismatch")
    return {
        "document_id": annotation["document_id"],
        "sentence_node": annotation["sentence_node"],
        "component_source_ids": annotation["component_source_ids"],
        "frame_count": len(frames),
        "unique_source_credit": 0,
        "complete_sentence_semantics_claimed": False,
        "inter_frame_discourse_edges_claimed": False,
    }


def verify_masc_record(record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    annotation = record.get("source_annotation")
    if annotation != expected["annotation"]:
        raise ValueError("MASC GrAF annotation replay mismatch")
    if record.get("split") != expected["split"] or record.get("source_text") != annotation["sentence"] or record.get("surface_target") != annotation["sentence"]:
        raise ValueError("MASC split or sentence replay mismatch")
    authority = record.get("semantic_supervision", {}).get("objective_authority")
    allowed = set(source["allowed_objectives"])
    if authority != {objective: objective in allowed for objective in TRAINING_OBJECTIVES}:
        raise ValueError("MASC objective authority exceeds manual annotation evidence")
    predicate = "FRAME_" + safe_symbol(annotation["frame_name"], "UNKNOWN")
    protected_objects = record["kernel_packet"].get("protected_objects") or {}
    expected_explicit = {
        (span["start"], span["end"], span["object_type"], span["copy_policy"])
        for span in annotation.get("protected_spans") or []
    }
    observed_explicit = {
        (
            value["source_span"]["character_start"],
            value["source_span"]["character_end"],
            value["object_type"],
            value["copy_policy"],
        )
        for value in protected_objects.values()
        if value.get("protection_source") == "explicit_user_or_caller_span"
    }
    if observed_explicit != expected_explicit:
        raise ValueError("MASC protected-object replay mismatch")
    expected_arguments = [
        {
            "role": safe_symbol(element["role"], "ROLE"),
            "value": expected_masc_value(
                element,
                sentence=annotation["sentence"],
                protected_objects=protected_objects,
            ),
        }
        for element in annotation["frame_elements"]
    ]
    contextual_frame_ambiguity = annotation.get("contextual_frame_ambiguity")
    if contextual_frame_ambiguity is not None:
        if (
            contextual_frame_ambiguity.get("policy")
            != MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY
            or contextual_frame_ambiguity.get("fit_split") != "private_train"
            or annotation["frame_name"]
            not in {
                alternative.get("frame_name")
                for alternative in contextual_frame_ambiguity.get("alternatives") or []
            }
        ):
            raise ValueError("MASC contextual-frame ambiguity replay mismatch")
        expected_arguments.append(
            {
                "role": "CONTEXTUAL_FRAME_ALTERNATIVES",
                "value": {
                    "type": "ambiguity",
                    "value": [
                        {
                            "value": alternative["value"],
                            "probability": float(alternative["probability"]),
                            "evidence": alternative["evidence"],
                        }
                        for alternative in contextual_frame_ambiguity["alternatives"]
                    ],
                },
            }
        )
    node = record["kernel_packet"]["program"]["nodes"][0]
    claim = record["answer_packet"]["claims"][0]
    observed_node_arguments = [
        {"role": row.get("role"), "value": observed_masc_value(row.get("value"))}
        for row in node.get("arguments") or []
    ]
    observed_claim_arguments = [
        {"role": row.get("role"), "value": observed_masc_value(row.get("value"))}
        for row in claim.get("arguments") or []
    ]
    if node.get("operator") != predicate or node.get("source_spans") != annotation["target_spans"] or observed_node_arguments != expected_arguments:
        raise ValueError("MASC kernel program replay mismatch")
    if claim.get("predicate") != predicate or observed_claim_arguments != expected_arguments:
        raise ValueError("MASC answer packet replay mismatch")
    expected_segment = {
        "frame_name": annotation["frame_name"],
        "lexical_unit": annotation["lexical_unit"],
        "target_spans": annotation["target_spans"],
        "frame_roles": sorted(
            {safe_symbol(element["role"], "ROLE") for element in annotation["frame_elements"]}
        ),
    }
    expected_tags = [
        {
            "tag": "FRAME_TARGET:" + safe_symbol(annotation["frame_name"], "UNKNOWN"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for span in annotation["target_spans"]
    ]
    expected_tags.extend(
        {
            "tag": "FRAME_ROLE:" + safe_symbol(element["role"], "ROLE"),
            "source_span": list(span),
            "authority": "licensed_manual_annotation",
        }
        for element in annotation["frame_elements"]
        for span in element["spans"]
    )
    expected_tags.extend(
        {
            "tag": "ENTITY:" + str(span["object_type"]),
            "source_span": [int(span["start"]), int(span["end"])],
            "authority": "licensed_manual_annotation",
        }
        for span in annotation.get("protected_spans") or []
    )
    expected_tags.sort(key=lambda row: (row["source_span"], row["tag"]))
    residual = record["kernel_packet"].get("residual") or {}
    if residual.get("segment_frame") != expected_segment:
        raise ValueError("MASC segment residual replay mismatch")
    if residual.get("token_tags") != expected_tags:
        raise ValueError("MASC token residual replay mismatch")
    labels = (record.get("residual_supervision") or {}).get("labels_by_channel") or {}
    if labels.get("segment") != 1 or labels.get("token") != 2:
        raise ValueError("MASC residual supervision does not reflect manual annotations")
    interaction = expected.get("interaction_annotation")
    if record.get("interaction_annotation") != interaction:
        raise ValueError("MASC interaction predecessor replay mismatch")
    identity = stable_hash(
        {
            "split": expected["split"],
            "source_id": record["provenance"]["source_id"],
            "source_group": record["provenance"]["source_group"],
            "source_text": annotation["sentence"],
            "surface_target": annotation["sentence"],
            "source_annotation": annotation,
            "interaction_annotation": interaction,
            "valid_realizations": None,
        }
    ).split(":", 1)[1]
    expected_state = create_hierarchical_residual_state(
        "kerc-corpus-" + identity[:24],
        scope={
            "user": "project-theseus-corpus",
            "project": "theseus",
            "conversation": identity[:24],
            "privacy": "private_local",
        },
    )
    expected_deltas: list[dict[str, Any]] = []
    if interaction:
        operations = [
            {
                "op": "OVERRIDE",
                "segment_id": "previous_turn",
                "key": "frame_name",
                "value": str(interaction["frame_name"]),
                "privacy": "interaction_private",
            },
            {
                "op": "OVERRIDE",
                "segment_id": "previous_turn",
                "key": "lexical_unit",
                "value": str(interaction["lexical_unit"]),
                "privacy": "interaction_private",
            },
        ]
        expected_state, delta = apply_hierarchical_residual_delta(
            expected_state,
            operations,
            expected_state_hash=expected_state["state_hash"],
            actor_authority="document",
            actor_id="masc_manual_framenet",
            provenance={
                "source": source["dataset_id"],
                "interaction_annotation_sha256": stable_hash(interaction),
            },
        )
        expected_deltas.append(delta)
    if record.get("hrl_state") != expected_state or record.get("hrl_deltas") != expected_deltas:
        raise ValueError("MASC VCM interaction state replay mismatch")
    expected_interaction_label = 1 if interaction else 0
    if labels.get("interaction") != expected_interaction_label:
        raise ValueError("MASC interaction residual label mismatch")
    return {
        "document_id": annotation["document_id"],
        "annotation_set_node": annotation["annotation_set_node"],
        "frame_name": annotation["frame_name"],
        "interaction_bound": bool(interaction),
        "contextual_frame_ambiguity_bound": bool(contextual_frame_ambiguity),
    }


def verify_gum_discourse_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild typed eRST topology from independently parsed source evidence."""

    annotation = expected["annotation"]
    if record.get("source_annotation") != annotation:
        raise ValueError("GUM source annotation replay mismatch")
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != expected["source_text"]
        or record.get("surface_target") != expected["source_text"]
        or record.get("provenance", {}).get("source_id") != expected["source_id"]
        or record.get("provenance", {}).get("source_group")
        != expected["source_group"]
        or record.get("provenance", {}).get("license_spdx")
        != expected["license_spdx"]
    ):
        raise ValueError("GUM record identity or license mismatch")
    supervision = record.get("semantic_supervision") or {}
    expected_objectives = {
        "surface_direct_control_v1": False,
        "surface_to_kernel_program_v1": True,
        "kernel_program_to_answer_packet_v1": True,
        "answer_packet_to_surface_v1": False,
    }
    if (
        supervision.get("objective_authority") != expected_objectives
        or supervision.get("source_credit_unit") != "document"
        or supervision.get("derived_view_unique_source_credit") != 0
        or supervision.get("erst_relation_authority")
        != "human_source_declared_primary_and_secondary_discourse_edges"
        or supervision.get("public_calibration_surface") is not False
        or supervision.get("benchmark_payload_used") is not False
    ):
        raise ValueError("GUM semantic authority mismatch")

    units = sorted(annotation["units"], key=lambda row: int(row["edu_id"]))
    unit_nodes = {
        int(unit["edu_id"]): f"k{index}" for index, unit in enumerate(units)
    }
    document_concept = "gum.document." + re.sub(
        r"[^a-z0-9]+", ".", annotation["document_id"].lower()
    ).strip(".")
    expected_nodes: list[dict[str, Any]] = []
    expected_segment_units: list[dict[str, Any]] = []
    expected_tags: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        node_id = f"k{index}"
        span = list(unit["excerpt_span"])
        expected_nodes.append(
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
        expected_segment_units.append(
            {
                "edu_id": int(unit["edu_id"]),
                "node_id": node_id,
                "target_spans": [span],
                "tree_depth": int(unit["tree_depth"]),
                "source_row_sha256": str(unit["source_row_sha256"]),
            }
        )
        expected_tags.append(
            {
                "tag": "ERST_DISCOURSE_UNIT",
                "source_span": span,
                "authority": "licensed_manual_annotation",
            }
        )
    expected_claims: list[dict[str, Any]] = []
    expected_segment_edges: list[dict[str, Any]] = []
    roots: list[str] = []
    for index, edge in enumerate(annotation["edges"]):
        node_id = f"k{len(units) + index}"
        roots.append(node_id)
        child_node = unit_nodes[int(edge["child_edu_id"])]
        parent_node = unit_nodes[int(edge["parent_edu_id"])]
        relation = str(edge["relation"])
        relation_base = relation.removesuffix("_m").removesuffix("_r")
        nuclearity = (
            "multinuclear"
            if relation.endswith("_m")
            else "satellite_nucleus"
            if relation.endswith("_r")
            else "secondary_unspecified"
        )
        predicate = "ERST_" + safe_symbol(relation_base, "RELATION")
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
        answer_arguments = json.loads(json.dumps(program_arguments))
        endpoint_concepts = {
            "CHILD": (
                f"erst.edu.{str(annotation['document_id']).lower()}."
                f"{int(edge['child_edu_id'])}"
            ),
            "PARENT": (
                f"erst.edu.{str(annotation['document_id']).lower()}."
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
                list(
                    next(
                        unit["excerpt_span"]
                        for unit in units
                        if int(unit["edu_id"]) == int(edge["child_edu_id"])
                    )
                ),
                list(
                    next(
                        unit["excerpt_span"]
                        for unit in units
                        if int(unit["edu_id"]) == int(edge["parent_edu_id"])
                    )
                ),
            ]
        )
        expected_nodes.append(
            {
                "node_id": node_id,
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": spans,
                "arguments": json.loads(json.dumps(program_arguments)),
            }
        )
        expected_claims.append(
            {
                "claim_id": f"claim-{index + 1}",
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": json.loads(json.dumps(answer_arguments)),
            }
        )
        signal_payload = str(edge["raw_signal_payload"])
        expected_segment_edges.append(
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
                "signal_count": 0
                if not signal_payload
                else signal_payload.count(";") + 1,
            }
        )
        child_span = next(
            unit["excerpt_span"]
            for unit in units
            if int(unit["edu_id"]) == int(edge["child_edu_id"])
        )
        expected_tags.append(
            {
                "tag": "ERST_RELATION:" + safe_symbol(relation_base, "RELATION"),
                "source_span": list(child_span),
                "authority": "licensed_manual_annotation",
            }
        )
    program = record["kernel_packet"]["program"]
    if program["roots"] != roots or program["nodes"] != expected_nodes:
        raise ValueError("GUM Kernel graph topology mismatch")
    if record["answer_packet"]["claims"] != expected_claims:
        raise ValueError("GUM answer-packet relation topology mismatch")
    expected_segment = {
        "schema": "erst_discourse_graph_v1",
        "document_id": annotation["document_id"],
        "anchor_edu_id": int(annotation["anchor_edu_id"]),
        "units": expected_segment_units,
        "edges": expected_segment_edges,
    }
    if record["kernel_packet"]["residual"]["segment_frame"] != expected_segment:
        raise ValueError("GUM eRST residual graph mismatch")
    expected_tags.sort(
        key=lambda row: (row["source_span"], row["tag"], row["authority"])
    )
    if record["kernel_packet"]["residual"]["token_tags"] != expected_tags:
        raise ValueError("GUM eRST source alignment mismatch")
    return {
        "document_id": annotation["document_id"],
        "anchor_edu_id": int(annotation["anchor_edu_id"]),
        "genre": annotation["genre"],
        "primary_relation": annotation["primary_relation"],
        "edge_count": len(annotation["edges"]),
        "source_declared_only": True,
        "common_source_binding": {
            "dataset_revision": str(source["dataset_revision"]),
            "license_spdx": str(expected["license_spdx"]),
            "content_sha256": str(source["content_sha256"]),
        },
    }


def verify_gum_entity_coreference_record(
    record: dict[str, Any], source: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Independently bind a packet to complete human coreference topology."""

    annotation = expected["annotation"]
    if record.get("source_annotation") != annotation:
        raise ValueError("GUM entity/coreference source annotation replay mismatch")
    if (
        record.get("split") != expected["split"]
        or record.get("source_text") != expected["source_text"]
        or record.get("surface_target") != expected["source_text"]
        or record.get("provenance", {}).get("source_id") != expected["source_id"]
        or record.get("provenance", {}).get("source_group") != expected["source_group"]
        or record.get("provenance", {}).get("license_spdx")
        != expected["license_spdx"]
    ):
        raise ValueError("GUM entity/coreference identity or license mismatch")
    semantic = record.get("semantic_supervision") or {}
    if (
        semantic.get("objective_authority")
        != {
            "surface_direct_control_v1": False,
            "surface_to_kernel_program_v1": True,
            "kernel_program_to_answer_packet_v1": True,
            "answer_packet_to_surface_v1": False,
        }
        or semantic.get("source_credit_unit") != "document"
        or semantic.get("derived_view_unique_source_credit") != 0
        or semantic.get("entity_coreference_authority")
        != "human_source_declared_complete_identity_components_and_bridging_edges"
        or semantic.get("concept_capsule_identity_authority")
        != "human_source_declared_component"
        or semantic.get("public_calibration_surface") is not False
        or semantic.get("benchmark_payload_used") is not False
    ):
        raise ValueError("GUM entity/coreference semantic authority mismatch")

    groups = annotation["groups"]
    mentions = annotation["mentions"]
    handle_by_group = {
        group["group_id"]: f"@C{index}" for index, group in enumerate(groups)
    }
    group_by_mention = {
        identity: group
        for group in groups
        for identity in group["mention_ids"]
    }
    expected_capsules = {}
    for group in groups:
        members = [
            mention for mention in mentions if mention["mention_id"] in group["mention_ids"]
        ]
        expected_capsules[handle_by_group[group["group_id"]]] = {
            "stable_identity": group["stable_identity"],
            "provenance": {
                "dataset_revision": source["dataset_revision"],
                "document_id": annotation["document_id"],
                "source_group_id": group["group_id"],
                "source_annotation_sha256": stable_hash(
                    [mention["source_annotation_sha256"] for mention in members]
                ),
            },
            "entity_types": sorted(
                {mention["attributes"]["entity_type"] for mention in members}
            ),
            "source_identity_values": sorted(
                {
                    mention["attributes"]["identity"]
                    for mention in members
                    if mention["attributes"]["identity"] != "_"
                }
            ),
            "mention_count": len(members),
        }
    packet = record["kernel_packet"]
    if packet.get("concept_capsules") != expected_capsules:
        raise ValueError("GUM entity/coreference concept capsule mismatch")
    program = packet["program"]
    nodes = program["nodes"]
    claims = record["answer_packet"]["claims"]

    def node_header_matches(node: dict[str, Any], node_id: str, predicate: str) -> bool:
        return {
            key: node.get(key)
            for key in (
                "node_id",
                "operator",
                "modality",
                "polarity",
                "quantifier",
                "confidence",
                "derivation",
            )
        } == {
            "node_id": node_id,
            "operator": predicate,
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
            "derivation": "preserved",
        }

    def claim_header_matches(
        claim: dict[str, Any], claim_id: str, predicate: str
    ) -> bool:
        return {
            key: claim.get(key)
            for key in (
                "claim_id",
                "predicate",
                "modality",
                "polarity",
                "quantifier",
                "confidence",
            )
        } == {
            "claim_id": claim_id,
            "predicate": predicate,
            "modality": "ASSERTED",
            "polarity": "AFFIRMED",
            "quantifier": "NONE",
            "confidence": 1.0,
        }
    expected_nodes = len(mentions) + len(groups) + len(annotation["relations"])
    expected_claims = expected_nodes
    if len(nodes) != expected_nodes or len(claims) != expected_claims:
        raise ValueError("GUM entity/coreference graph cardinality mismatch")
    node_by_mention = {
        mention["mention_id"]: f"k{index}" for index, mention in enumerate(mentions)
    }
    expected_tags = []
    for index, mention in enumerate(mentions):
        node = nodes[index]
        claim = claims[index]
        group = group_by_mention[mention["mention_id"]]
        handle = handle_by_group[group["group_id"]]
        attributes = mention["attributes"]
        expected_arguments = [
            {"role": "IDENTITY", "value": {"type": "handle", "value": handle}},
            {
                "role": "ENTITY_TYPE",
                "value": independent_semantic_concept(
                    "gum.entity_type", attributes["entity_type"]
                ),
            },
            {
                "role": "INFORMATION_STATUS",
                "value": independent_semantic_concept(
                    "gum.information_status", attributes["information_status"]
                ),
            },
            {
                "role": "CENTERING",
                "value": independent_semantic_concept(
                    "gum.centering", attributes["centering"]
                ),
            },
        ]
        if (
            node
            != {
                "node_id": f"k{index}",
                "operator": "ENTITY_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": mention["excerpt_spans"],
                "arguments": expected_arguments,
            }
            or claim
            != {
                "claim_id": f"claim-{index + 1}",
                "predicate": "ENTITY_MENTION",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": expected_arguments,
            }
        ):
            raise ValueError("GUM entity mention graph mismatch")
        for span in mention["excerpt_spans"]:
            expected_tags.append(
                {
                    "tag": "ENTITY_MENTION:"
                    + safe_symbol(attributes["entity_type"], "UNKNOWN"),
                    "source_span": span,
                    "authority": "licensed_manual_annotation",
                }
            )
    roots = []
    cursor = len(mentions)
    claim_cursor = len(mentions)
    for group in groups:
        node = nodes[cursor]
        claim = claims[claim_cursor]
        roots.append(f"k{cursor}")
        member_ids = group["mention_ids"]
        handle = handle_by_group[group["group_id"]]
        expected_spans = sorted(
            span
            for mention in mentions
            if mention["mention_id"] in member_ids
            for span in mention["excerpt_spans"]
        )
        if (
            not node_header_matches(
                node, f"k{cursor}", "ENTITY_IDENTITY_COMPONENT"
            )
            or node.get("source_spans") != expected_spans
            or node.get("arguments")
            != [
                {"role": "IDENTITY", "value": {"type": "handle", "value": handle}},
                {
                    "role": "MEMBERS",
                    "value": {
                        "type": "list",
                        "value": [
                            {"type": "node_ref", "value": node_by_mention[identity]}
                            for identity in member_ids
                        ],
                    },
                },
            ]
            or not claim_header_matches(
                claim, f"claim-{claim_cursor + 1}", "ENTITY_IDENTITY_COMPONENT"
            )
            or claim.get("arguments")
            != [
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
            ]
        ):
            raise ValueError("GUM identity component graph mismatch")
        cursor += 1
        claim_cursor += 1
    for relation in annotation["relations"]:
        node = nodes[cursor]
        claim = claims[claim_cursor]
        roots.append(f"k{cursor}")
        source_mention = relation["source_mention_id"]
        target_mention = relation["target_mention_id"]
        relation_concept = independent_semantic_concept(
            "gum.entity_relation", relation["relation_type"]
        )
        spans = sorted(
            span
            for identity in (source_mention, target_mention)
            for span in next(
                mention["excerpt_spans"]
                for mention in mentions
                if mention["mention_id"] == identity
            )
        )
        predicate = "ENTITY_RELATION_" + safe_symbol(
            relation["relation_type"], "RELATION"
        )
        if (
            not node_header_matches(node, f"k{cursor}", predicate)
            or node.get("source_spans") != spans
            or node.get("arguments")
            != [
                {
                    "role": "SOURCE",
                    "value": {
                        "type": "node_ref",
                        "value": node_by_mention[source_mention],
                    },
                },
                {
                    "role": "TARGET",
                    "value": {
                        "type": "node_ref",
                        "value": node_by_mention[target_mention],
                    },
                },
                {"role": "RELATION_TYPE", "value": relation_concept},
            ]
            or not claim_header_matches(
                claim, f"claim-{claim_cursor + 1}", predicate
            )
            or claim.get("arguments")
            != [
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
                {"role": "RELATION_TYPE", "value": relation_concept},
            ]
        ):
            raise ValueError("GUM source-declared relation graph mismatch")
        cursor += 1
        claim_cursor += 1
    if program["roots"] != roots:
        raise ValueError("GUM entity/coreference roots mismatch")
    expected_tags.sort(
        key=lambda row: (row["source_span"], row["tag"], row["authority"])
    )
    if packet["residual"]["token_tags"] != expected_tags:
        raise ValueError("GUM entity/coreference source alignment mismatch")
    state, deltas = independent_hrl_replay(
        split=expected["split"],
        source_id=expected["source_id"],
        source_group=expected["source_group"],
        source_text=expected["source_text"],
        surface_target=expected["source_text"],
        source_annotation=annotation,
        interaction_annotation=None,
        interaction_entries=[],
        actor_id="licensed_source_context",
        source={**source, "license_spdx": expected["license_spdx"]},
        valid_realizations=None,
        concept_capsules=expected_capsules,
    )
    if record.get("hrl_state") != state or record.get("hrl_deltas") != deltas:
        raise ValueError("GUM entity/coreference residual identity mismatch")
    return {
        "document_id": annotation["document_id"],
        "record_kind": annotation["record_kind"],
        "mention_count": len(mentions),
        "group_count": len(groups),
        "relation_count": len(annotation["relations"]),
        "source_declared_only": True,
        "common_source_binding": {
            "dataset_revision": str(source["dataset_revision"]),
            "license_spdx": str(expected["license_spdx"]),
            "content_sha256": str(source["content_sha256"]),
        },
    }


def producer_family_identity_receipts_from_source() -> dict[str, dict[str, Any]]:
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
            "raw_relation_producer": scripts / "kerc_gum_discourse_relations.py"
        },
        "gum_entity_coreference": {
            "raw_relation_producer": scripts / "kerc_gum_entity_coreference.py"
        },
    }
    return family_identity_receipts(
        source_path=scripts / "kerc_semantic_corpus.py",
        source_label="scripts/kerc_semantic_corpus.py",
        role="candidate_record_producer",
        family_roots=PRODUCER_FAMILY_ROOTS,
        external_paths=common_external,
        family_external_paths=family_external,
    )


def verifier_family_identity_receipts() -> dict[str, dict[str, Any]]:
    scripts = ROOT / "scripts"
    family_external = {
        "masc_event_coreference": {
            "raw_relation_verifier": scripts / "kerc_masc_event_coreference_verify.py"
        },
        "masc_mpqa_relation": {
            "raw_relation_verifier": scripts / "kerc_masc_mpqa_relations_verify.py"
        },
        "gum_discourse": {
            "raw_relation_verifier": scripts
            / "kerc_gum_discourse_relations_verify.py"
        },
        "gum_entity_coreference": {
            "raw_relation_verifier": scripts
            / "kerc_gum_entity_coreference_verify.py"
        },
    }
    return family_identity_receipts(
        source_path=Path(__file__).resolve(),
        source_label="scripts/kerc_semantic_corpus_verify.py",
        role="independent_record_verifier",
        family_roots=VERIFIER_FAMILY_ROOTS,
        external_paths={},
        family_external_paths=family_external,
    )


def producer_finalization_identity_from_source() -> dict[str, Any]:
    return source_closure_receipt(
        source_path=ROOT / "scripts" / "kerc_semantic_corpus.py",
        source_label="scripts/kerc_semantic_corpus.py",
        role="candidate_record_finalization",
        family="all_source_families",
        root_function="finalize_candidate_record",
        external_paths={
            "kernel_protocol": ROOT / "scripts" / "kernel_english_protocol.py",
            "residual_economics": ROOT / "scripts" / "kerc_residual_economics.py",
        },
    )


def verify_candidate_common(
    *,
    record: dict[str, Any],
    source: dict[str, Any],
    expected_importance: dict[str, Any],
    replay: dict[str, Any],
    family: str,
    producer_family_identity: str,
    verifier_route_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supervision = record.get("residual_supervision") or {}
    if supervision.get("importance") != expected_importance:
        raise ValueError("candidate importance receipt replay mismatch")
    packet = record.get("kernel_packet") or {}
    residual = packet.get("residual") or {}
    expected_allocation = validate_structural_rate_distortion_allocation(
        supervision.get("rate_distortion_allocation") or {},
        kernel_program=packet.get("program") or {},
        global_state=(record.get("hrl_state") or {}).get("global") or {},
        segment_residual=residual.get("segment_frame") or {},
        token_residuals=residual.get("token_tags") or [],
        exact_objects=packet.get("protected_objects") or {},
        exact_codec=residual.get("codec") or None,
    )
    if expected_allocation["selected_fidelity"] != residual.get("fidelity"):
        raise ValueError("candidate allocation fidelity mismatch")
    provenance = record.get("provenance") or {}
    if (
        provenance.get("dataset_revision") != source["dataset_revision"]
        or provenance.get("license_spdx") != source["license_spdx"]
    ):
        raise ValueError("candidate provenance mismatch")
    semantic = record.get("semantic_supervision") or {}
    if (
        semantic.get("annotation_source_sha256") != source["content_sha256"]
        or semantic.get("producer_artifact_sha256") != producer_family_identity
    ):
        raise ValueError("candidate semantic source binding mismatch")
    source_id = str(provenance.get("source_id") or "")
    evidence_sha256 = stable_hash(
        {
            "policy": VERIFIER_POLICY,
            "verifier_route_identity": verifier_route_identity,
            "source_content_sha256": source["content_sha256"],
            "source_id": source_id,
            "split": record["split"],
            "family": family,
            "replay": replay,
        }
    )
    semantic_payload_sha256 = training_semantic_payload_sha256(record)
    receipt = {
        "policy": TRAINING_VERIFICATION_POLICY,
        "receipt_id": "kerc-source-replay:" + evidence_sha256.split(":", 1)[1][:32],
        "accepted": True,
        "verifier_id": VERIFIER_ID,
        "reviewer_independent_of_record_producer": True,
        "method": "licensed_semantic_dataset_plus_independent_schema_review",
        "evidence_sha256": evidence_sha256,
        "semantic_payload_sha256": semantic_payload_sha256,
    }
    record["verification_receipt"] = receipt
    return validate_training_record(record), receipt


def verify_semantic_admission_job(job: dict[str, Any]) -> dict[str, Any]:
    """Verify one cache-miss record without touching shared process state.

    The worker returns data only. SQLite publication, aggregate accounting, and
    canonical serialization remain single-writer responsibilities in the parent
    process, which makes completion order irrelevant to the authoritative bytes.
    """

    line_number = int(job["line_number"])
    family = str(job["family"])
    try:
        family_verifier = globals().get(VERIFIER_FAMILY_ROOTS[family])
        if not callable(family_verifier):
            raise ValueError(f"source-family verifier unavailable: {family}")
        record = copy.deepcopy(job["record"])
        replay = family_verifier(record, job["source"], job["expected_row"])
        verification_source = replay.get("common_source_binding", job["source"])
        canonical, receipt = verify_candidate_common(
            record=record,
            source=verification_source,
            expected_importance=job["expected_importance"],
            replay=replay,
            family=family,
            producer_family_identity=str(job["producer_family_identity"]),
            verifier_route_identity=str(job["verifier_route_identity"]),
        )
        return {
            "line_number": line_number,
            "family": family,
            "source_id": str(job["source_id"]),
            "semantic_key": str(job["semantic_key"]),
            "accepted": True,
            "canonical": canonical,
            "receipt": receipt,
        }
    except Exception as exc:
        return {
            "line_number": line_number,
            "family": family,
            "source_id": str(job.get("source_id") or ""),
            "semantic_key": str(job.get("semantic_key") or ""),
            "accepted": False,
            "failure_code": str(getattr(exc, "code", type(exc).__name__)),
            "failure_message": str(exc)[:160],
        }


def _verify_semantic_admission_batch(
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [verify_semantic_admission_job(job) for job in batch]


def execute_semantic_admission_jobs(
    jobs: list[dict[str, Any]],
    *,
    worker_count: int,
    batch_size: int,
    max_in_flight_batches_per_worker: int = 2,
) -> Iterable[dict[str, Any]]:
    """Run admission jobs with bounded memory and deterministic authority.

    Results may arrive out of order, but every result carries its original line
    number. The caller sorts authoritative rows and failures before writing.
    Spawned workers never inherit an open SQLite connection.
    """

    if worker_count < 1:
        raise ValueError("semantic admission worker_count must be positive")
    if batch_size < 1:
        raise ValueError("semantic admission batch_size must be positive")
    if max_in_flight_batches_per_worker < 1:
        raise ValueError(
            "semantic admission max_in_flight_batches_per_worker must be positive"
        )
    if worker_count == 1 or len(jobs) <= 1:
        for job in jobs:
            yield verify_semantic_admission_job(job)
        return

    batches = [
        jobs[offset : offset + batch_size]
        for offset in range(0, len(jobs), batch_size)
    ]
    pending: dict[concurrent.futures.Future[list[dict[str, Any]]], int] = {}
    next_batch = 0
    maximum_pending = worker_count * max_in_flight_batches_per_worker
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        while next_batch < len(batches) or pending:
            while next_batch < len(batches) and len(pending) < maximum_pending:
                future = executor.submit(
                    _verify_semantic_admission_batch, batches[next_batch]
                )
                pending[future] = next_batch
                next_batch += 1
            completed, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in completed:
                pending.pop(future)
                for result in future.result():
                    yield result


def order_semantic_admission_authority(
    accepted_rows: list[tuple[int, dict[str, Any], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restore candidate order before any order-sensitive aggregate replay."""

    ordered = sorted(accepted_rows, key=lambda row: row[0])
    line_numbers = [row[0] for row in ordered]
    if len(line_numbers) != len(set(line_numbers)):
        raise ValueError("duplicate accepted semantic admission line number")
    return [row[1] for row in ordered], [row[2] for row in ordered]


def source_catalog(corpus: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for key in ("dolly", "masc", "gum", "oasst2"):
        source = corpus[key]
        sources.append(
            {
                "dataset_id": source["dataset_id"],
                "dataset_revision": source["dataset_revision"],
                "source_url": source["source_url"],
                "license_evidence_url": source["license_evidence_url"],
                "content_sha256": source["content_sha256"],
                "license_spdx": source["license_spdx"],
                "permitted_use": "model_training",
                "training_allowed": True,
                "public_benchmark_surface": False,
                "public_benchmark_payload": False,
                "allowed_evidence_tiers": ["licensed_human_task_gold"],
                "allowed_objectives": sorted(
                    set(source["allowed_objectives"])
                    | set(source.get("grounded_question_allowed_objectives") or [])
                ),
                **(
                    {
                        "per_record_license_matrix": dict(
                            source["allowed_genre_licenses"]
                        ),
                        "official_partitions_admitted": ["train"],
                        "official_partitions_quarantined": ["dev", "test", "test2"],
                    }
                    if key == "gum"
                    else {}
                ),
            }
        )
    return {"policy": KERC_SOURCE_CATALOG_POLICY, "sources": sources}


def verify(
    config_path: Path,
    *,
    runtime_config_path: Path = DEFAULT_RUNTIME_CONFIG,
    use_cache: bool = True,
    refresh_cache: bool = False,
    bypass_run_cache: bool = False,
    semantic_workers: int | None = None,
    semantic_batch_size: int | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    phase_started = run_started
    phase_runtime_ms: dict[str, int] = {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    if runtime_config.get("policy") != "project_theseus_kerc_semantic_verifier_runtime_v1":
        raise ValueError("KERC semantic verifier runtime policy mismatch")
    stage = validate_kernel_english_config(config)
    corpus = stage["semantic_corpus_materialization"]
    if corpus.get("policy") != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("semantic corpus policy mismatch")
    candidate_path = resolve(corpus["candidate_records_jsonl"])
    manifest_path = resolve(corpus["producer_manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("trigger_state") != "GREEN" or manifest.get("candidate_records", {}).get("sha256") != sha256_file(candidate_path):
        raise ValueError("producer manifest does not bind the candidate artifact")
    producer_path = ROOT / "scripts" / "kerc_semantic_corpus.py"
    if manifest.get("producer_sha256") != sha256_file(producer_path):
        raise ValueError("producer changed after candidate materialization")
    producer_family_receipts = producer_family_identity_receipts_from_source()
    if manifest.get("producer_family_identities") != producer_family_receipts:
        raise ValueError("producer family identities do not replay from source")
    producer_finalization_receipt = producer_finalization_identity_from_source()
    if manifest.get("producer_finalization_identity") != producer_finalization_receipt:
        raise ValueError("producer finalization identity does not replay from source")
    output_root = resolve(corpus["output_root"])
    records_path = output_root / "records.jsonl"
    ledger_path = output_root / "verification_ledger.jsonl"
    catalog_path = output_root / "semantic_source_catalog.json"
    report_path = ROOT / "reports" / "runtime" / "kerc_semantic_corpus_verification.json"
    cache_cfg = corpus["content_cache"]
    cache_enabled = bool(use_cache and cache_cfg["enabled"])
    cache_root = resolve(cache_cfg["root"])
    parallel_cfg = runtime_config.get("parallelism") or {}
    resource_cfg = runtime_config.get("resource_contract") or {}
    worker_count, worker_resource_receipt = resolve_semantic_worker_count(
        parallel_cfg,
        resource_cfg,
        requested_workers=semantic_workers,
        logical_cpu_count=os.cpu_count(),
        total_memory_bytes=host_total_memory_bytes(),
    )
    batch_size = int(
        parallel_cfg.get("batch_size", 4)
        if semantic_batch_size is None
        else semantic_batch_size
    )
    in_flight_batches = int(
        parallel_cfg.get("max_in_flight_batches_per_worker", 2)
    )
    if batch_size < 1:
        raise ValueError("semantic admission batch size must be positive")
    cache_dependencies = dependency_bindings(
        verifier_cache_dependency_paths(
            config_path,
            runtime_config_path,
            corpus,
            candidate_path=candidate_path,
            manifest_path=manifest_path,
        )
    )
    cache_outputs = {
        "canonical_records": records_path,
        "semantic_source_catalog": catalog_path,
        "verification_ledger": ledger_path,
        "verification_report": report_path,
    }
    if cache_enabled and not refresh_cache and not bypass_run_cache:
        cached = load_receipt(
            cache_root,
            role=str(cache_cfg["verifier_role"]),
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="verification_report",
        )
        if cached is not None and cached.get("trigger_state") == "GREEN":
            write_json_atomic(
                cache_root / "telemetry" / "verifier_last.json",
                {
                    "policy": "project_theseus_kerc_incremental_cache_telemetry_v1",
                    "run_cache_hit": True,
                    "semantic_admission": {
                        "hits": 0,
                        "misses": 0,
                        "hits_by_family": {},
                        "misses_by_family": {},
                    },
                    "candidate_records_sha256": cached["candidate_records_sha256"],
                    "storage": cache_storage_telemetry(
                        cache_root / "verifier_objects.sqlite3"
                    ),
                    "claim_scope": "cache execution telemetry only; not semantic or capability evidence",
                },
            )
            return cached

    phase_runtime_ms["configuration_and_run_cache"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()

    maximum_characters = int(corpus["maximum_source_characters"])
    behavior_expected = independent_oasst_behavior_assignments(corpus["oasst2"])
    grounded_expected = independent_dolly_grounded_assignments(
        corpus["dolly"], maximum_characters
    )
    grounded_source_hashes = {
        row["annotation"]["source_row_sha256"]
        for row in grounded_expected.values()
    }
    masc_expected = independent_masc_assignments(corpus["masc"], maximum_characters)
    masc_composite_expected = independent_masc_composite_assignments(
        masc_expected, corpus["masc"]
    )
    masc_decision_expected = independent_masc_decision_assignments(
        corpus["masc"], maximum_characters
    )
    event_coreference_contract = corpus["masc"]["event_coreference"]
    masc_event_rows, masc_event_audit = (
        independently_reconstruct_event_coreference_groups(
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
            maximum_characters=maximum_characters,
        )
    )
    masc_event_expected = {
        row["source_id"]: row
        for rows in masc_event_rows.values()
        for row in rows
    }
    expected_event_rejections = {
        (row["document_id"], row["annotation_set_name"])
        for row in event_coreference_contract["expected_rejected_groups"]
    }
    observed_event_rejections = {
        (row["document_id"], row["annotation_set_name"])
        for row in masc_event_audit["rejected_groups"]
    }
    if (
        masc_event_audit["policy"] != MASC_EVENT_COREFERENCE_POLICY
        or event_coreference_contract["alignment_contract"]
        != MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT
        or event_coreference_contract["source_compaction_contract"]
        != MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT
        or masc_event_audit["alignment_implementation"]
        != "verifier_local_context_margin_v1"
        or masc_event_audit["observed_group_count"]
        != int(event_coreference_contract["expected_observed_group_count"])
        or masc_event_audit["observed_mention_count"]
        != int(event_coreference_contract["expected_observed_mention_count"])
        or masc_event_audit["admitted_group_count"]
        != int(event_coreference_contract["expected_admitted_group_count"])
        or masc_event_audit["admitted_mention_count"]
        != int(event_coreference_contract["expected_admitted_mention_count"])
        or masc_event_audit["record_count_by_split"]
        != event_coreference_contract["records_by_split"]
        or masc_event_audit["mention_count_by_split"]
        != event_coreference_contract["mentions_by_split"]
        or masc_event_audit["rejected_group_count"]
        != int(event_coreference_contract["expected_rejected_group_count"])
        or observed_event_rejections != expected_event_rejections
        or masc_event_audit["partial_group_admission_count"] != 0
        or masc_event_audit["cooccurrence_inferred_relation_count"] != 0
    ):
        raise ValueError("independent MASC event-coreference reconstruction mismatch")
    mpqa_relation_contract = corpus["masc"]["mpqa_relations"]
    masc_mpqa_rows, masc_mpqa_audit = independently_reconstruct_mpqa_relation_chains(
        original_mpqa_root=resolve(mpqa_relation_contract["original_mpqa_root"]),
        private_dev_documents=set(mpqa_relation_contract["private_dev_documents"]),
        private_eval_documents=set(mpqa_relation_contract["private_eval_documents"]),
        maximum_characters=maximum_characters,
    )
    masc_mpqa_expected = {
        row["source_id"]: row
        for rows in masc_mpqa_rows.values()
        for row in rows
    }
    if (
        masc_mpqa_audit["policy"] != MASC_MPQA_RELATION_POLICY
        or mpqa_relation_contract["relation_contract"]
        != MASC_MPQA_RELATION_CONTRACT
        or mpqa_relation_contract["source_compaction_contract"]
        != MASC_MPQA_RELATION_COMPACTION_CONTRACT
        or masc_mpqa_audit["parser_implementation"]
        != "verifier_state_machine_attribute_parser_v1"
        or masc_mpqa_audit["observed_linked_expression_count"]
        != int(mpqa_relation_contract["expected_observed_linked_expression_count"])
        or masc_mpqa_audit["admitted_relation_count"]
        != int(mpqa_relation_contract["expected_admitted_relation_count"])
        or masc_mpqa_audit["admitted_source_member_count"]
        != int(mpqa_relation_contract["expected_admitted_source_member_count"])
        or masc_mpqa_audit["admitted_attitude_count"]
        != int(mpqa_relation_contract["expected_admitted_attitude_count"])
        or masc_mpqa_audit["admitted_target_count"]
        != int(mpqa_relation_contract["expected_admitted_target_count"])
        or masc_mpqa_audit["record_count_by_split"]
        != mpqa_relation_contract["records_by_split"]
        or masc_mpqa_audit["rejection_reason_counts"]
        != mpqa_relation_contract["expected_rejection_reason_counts"]
        or masc_mpqa_audit["partial_relation_admission_count"] != 0
        or masc_mpqa_audit["inferred_relation_count"] != 0
    ):
        raise ValueError("independent MASC MPQA-relation reconstruction mismatch")
    gum_contract = corpus["gum"]
    gum_rows, gum_audit = independently_reconstruct_gum_discourse_relations(
        source_root=resolve(gum_contract["source_root"]),
        allowed_genre_licenses=dict(gum_contract["allowed_genre_licenses"]),
        private_dev_documents=set(gum_contract["private_dev_documents"]),
        private_eval_documents=set(gum_contract["private_eval_documents"]),
        expected_selected_source_sha256=gum_contract["content_sha256"],
        maximum_characters=maximum_characters,
    )
    gum_expected = {
        row["source_id"]: row for rows in gum_rows.values() for row in rows
    }
    if (
        gum_audit["policy"] != GUM_DISCOURSE_POLICY
        or gum_audit["relation_contract"] != GUM_DISCOURSE_RELATION_CONTRACT
        or gum_audit["split_contract"] != GUM_DISCOURSE_SPLIT_CONTRACT
        or gum_audit["projection_contract"] != GUM_DISCOURSE_PROJECTION_CONTRACT
        or gum_audit["parser_implementation"]
        != "verifier_expat_csv_state_machine_v1"
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
        raise ValueError("independent GUM eRST reconstruction mismatch")
    gum_entity_contract = gum_contract["entity_coreference"]
    gum_entity_rows, gum_entity_audit = (
        independently_reconstruct_gum_entity_coreference(
            source_root=resolve(gum_contract["source_root"]),
            allowed_genre_licenses=dict(gum_contract["allowed_genre_licenses"]),
            private_dev_documents=set(gum_contract["private_dev_documents"]),
            private_eval_documents=set(gum_contract["private_eval_documents"]),
            expected_selected_source_sha256=gum_entity_contract["content_sha256"],
            maximum_characters=maximum_characters,
        )
    )
    gum_entity_expected = {
        row["source_id"]: row for rows in gum_entity_rows.values() for row in rows
    }
    if (
        gum_entity_audit["policy"] != GUM_ENTITY_COREFERENCE_POLICY
        or gum_entity_audit["relation_contract"]
        != GUM_ENTITY_COREFERENCE_RELATION_CONTRACT
        or gum_entity_audit["split_contract"]
        != GUM_ENTITY_COREFERENCE_SPLIT_CONTRACT
        or gum_entity_audit["source_compaction_contract"]
        != GUM_ENTITY_COREFERENCE_COMPACTION_CONTRACT
        or gum_entity_audit["parser_implementation"]
        != "verifier_csv_expat_union_find_v1"
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
        or any(
            int(gum_entity_audit["cross_format_topology_by_split"][split]["mentions"])
            != int(gum_entity_contract["mentions_by_split"][split])
            or int(
                gum_entity_audit["cross_format_topology_by_split"][split][
                    "components"
                ]
            )
            != int(gum_entity_contract["components_by_split"][split])
            for split in SPLITS
        )
        or any(
            int(
                gum_entity_audit["cross_format_topology_by_split"][split][
                    "component_membership_documents_agreeing"
                ]
            )
            != int(gum_contract["documents_by_split"][split])
            for split in SPLITS
        )
        or gum_entity_audit["rejected_record_count"] != 0
        or gum_entity_audit["official_nontrain_document_admission_count"] != 0
        or gum_entity_audit["partial_component_admission_count"] != 0
        or gum_entity_audit["inferred_relation_count"] != 0
    ):
        raise ValueError("independent GUM entity/coreference reconstruction mismatch")
    expected = {
        **independent_dolly_assignments(
            corpus["dolly"],
            maximum_characters,
            reserved_source_hashes=grounded_source_hashes,
        ),
        **grounded_expected,
        **masc_expected,
        **masc_composite_expected,
        **masc_decision_expected,
        **masc_event_expected,
        **masc_mpqa_expected,
        **gum_expected,
        **gum_entity_expected,
        **independent_oasst_assignments(
            corpus["oasst2"],
            reserved_groups={row["source_group"] for row in behavior_expected.values()},
        ),
        **behavior_expected,
    }
    expected_frame_priors = {
        row["annotation"]["contextual_frame_ambiguity"]["lexical_unit"]: row[
            "annotation"
        ]["contextual_frame_ambiguity"]["content_sha256"]
        for row in masc_expected.values()
        if "contextual_frame_ambiguity" in row["annotation"]
    }
    expected_frame_ambiguity_counts = {
        split: sum(
            1
            for row in masc_expected.values()
            if row["split"] == split
            and "contextual_frame_ambiguity" in row["annotation"]
        )
        for split in SPLITS
    }
    producer_frame_ambiguity = manifest.get("masc_contextual_frame_ambiguity") or {}
    if (
        producer_frame_ambiguity.get("policy")
        != MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY
        or producer_frame_ambiguity.get("fit_split") != "private_train"
        or producer_frame_ambiguity.get("prior_sha256_by_lexical_unit")
        != expected_frame_priors
        or producer_frame_ambiguity.get("bound_record_count_by_split")
        != expected_frame_ambiguity_counts
        or producer_frame_ambiguity.get("unresolved_ambiguity_record_count") != 0
        or producer_frame_ambiguity.get("calibrated_probability_claimed") is not False
    ):
        raise ValueError("producer MASC contextual-frame ambiguity telemetry mismatch")
    expected_composite_counts = {
        split: sum(
            1 for row in masc_composite_expected.values() if row["split"] == split
        )
        for split in SPLITS
    }
    expected_composite_frame_counts = {
        split: dict(
            Counter(
                str(len(row["annotation"]["frames"]))
                for row in masc_composite_expected.values()
                if row["split"] == split
            )
        )
        for split in SPLITS
    }
    producer_composites = manifest.get("masc_composite_semantics") or {}
    expected_composite_total = sum(expected_composite_counts.values())
    if (
        producer_composites.get("policy")
        != "project_theseus_kerc_masc_composite_semantics_v1"
        or producer_composites.get("record_count_by_split")
        != expected_composite_counts
        or producer_composites.get("frame_count_distribution_by_split")
        != expected_composite_frame_counts
        or producer_composites.get("multi_node_program_count")
        != expected_composite_total
        or producer_composites.get("multi_root_program_count")
        != expected_composite_total
        or producer_composites.get("multi_claim_answer_count")
        != expected_composite_total
        or producer_composites.get("unique_source_credit") != 0
        or producer_composites.get("complete_sentence_semantics_claimed") is not False
        or producer_composites.get("inter_frame_discourse_edges_claimed") is not False
        or producer_composites.get("claim_scope")
        != corpus["masc"]["composite_semantic_claim_scope"]
    ):
        raise ValueError("producer MASC composite telemetry mismatch")
    expected_decision_counts = {
        split: sum(1 for row in masc_decision_expected.values() if row["split"] == split)
        for split in SPLITS
    }
    expected_decision_layers = {
        split: dict(
            Counter(
                item["layer"]
                for row in masc_decision_expected.values()
                if row["split"] == split
                for item in row["annotation"]["annotations"]
            )
        )
        for split in SPLITS
    }
    expected_decision_missing = {
        split: {
            layer: sum(
                1 for row in masc_decision_expected.values()
                if row["split"] == split and row["annotation"]["missingness"][layer]
            )
            for layer in ("cb", "event", "mpqa")
        }
        for split in SPLITS
    }
    expected_typed_count = sum(
        1
        for row in masc_decision_expected.values()
        for item in row["annotation"]["annotations"]
        for argument in independent_decision_arguments(item)
        if argument["value"]["type"] != "byte_literal"
    )
    producer_decisions = manifest.get("masc_decision_semantics") or {}
    if (
        producer_decisions.get("policy") != MASC_DECISION_SEMANTICS_POLICY
        or producer_decisions.get("record_count_by_split") != expected_decision_counts
        or producer_decisions.get("annotation_count_by_split_and_layer") != expected_decision_layers
        or producer_decisions.get("missing_layer_record_count_by_split") != expected_decision_missing
        or producer_decisions.get("typed_nonliteral_argument_count") != expected_typed_count
        or producer_decisions.get("unique_source_credit") != 0
        or producer_decisions.get("claim_scope") != corpus["masc"]["decision_semantic_claim_scope"]
        or producer_decisions.get("complete_sentence_semantics_claimed") is not False
        or producer_decisions.get("truth_claimed") is not False
        or producer_decisions.get("event_coreference_grouping_claimed") is not False
        or producer_decisions.get("source_declared_cross_annotation_links_resolved") is not False
    ):
        raise ValueError("producer MASC decision-semantic telemetry mismatch")
    producer_event_coreference = manifest.get("masc_event_coreference") or {}
    expected_event_source_ids = {
        split: [row["source_id"] for row in rows]
        for split, rows in masc_event_rows.items()
    }
    expected_event_rejected_rows = sorted(
        [
            {
                "document_id": document_id,
                "annotation_set_name": annotation_set_name,
            }
            for document_id, annotation_set_name in observed_event_rejections
        ],
        key=lambda row: (row["document_id"], row["annotation_set_name"]),
    )
    if (
        producer_event_coreference.get("policy")
        != MASC_EVENT_COREFERENCE_POLICY
        or producer_event_coreference.get("alignment_contract")
        != MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT
        or producer_event_coreference.get("source_compaction_contract")
        != MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT
        or producer_event_coreference.get("producer_alignment_implementation")
        != "producer_global_sequence_matcher_v1"
        or producer_event_coreference.get("observed_group_count")
        != masc_event_audit["observed_group_count"]
        or producer_event_coreference.get("observed_mention_count")
        != masc_event_audit["observed_mention_count"]
        or producer_event_coreference.get("admitted_group_count")
        != masc_event_audit["admitted_group_count"]
        or producer_event_coreference.get("admitted_mention_count")
        != masc_event_audit["admitted_mention_count"]
        or producer_event_coreference.get("record_count_by_split")
        != masc_event_audit["record_count_by_split"]
        or producer_event_coreference.get("mention_count_by_split")
        != masc_event_audit["mention_count_by_split"]
        or producer_event_coreference.get("admitted_source_ids_by_split")
        != expected_event_source_ids
        or producer_event_coreference.get("rejected_group_count")
        != masc_event_audit["rejected_group_count"]
        or producer_event_coreference.get("rejected_groups")
        != expected_event_rejected_rows
        or producer_event_coreference.get("partial_group_admission_count") != 0
        or producer_event_coreference.get("cooccurrence_inferred_relation_count") != 0
        or producer_event_coreference.get("unique_source_credit")
        != int(event_coreference_contract["unique_source_credit"])
        or producer_event_coreference.get("claim_scope")
        != event_coreference_contract["claim_scope"]
        or producer_event_coreference.get("complete_group_alignment_required") is not True
        or producer_event_coreference.get("complete_sentence_semantics_claimed") is not False
        or producer_event_coreference.get("truth_claimed") is not False
        or producer_event_coreference.get("causal_relation_claimed") is not False
        or producer_event_coreference.get("temporal_relation_claimed") is not False
    ):
        raise ValueError("producer MASC event-coreference telemetry mismatch")
    producer_mpqa_relations = manifest.get("masc_mpqa_relations") or {}
    expected_mpqa_source_ids = {
        split: [row["source_id"] for row in rows]
        for split, rows in masc_mpqa_rows.items()
    }
    expected_mpqa_span_status = dict(
        Counter(
            independent_mpqa_span_status(member)
            for rows in masc_mpqa_rows.values()
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
    )
    if (
        producer_mpqa_relations.get("policy") != MASC_MPQA_RELATION_POLICY
        or producer_mpqa_relations.get("relation_contract")
        != MASC_MPQA_RELATION_CONTRACT
        or producer_mpqa_relations.get("source_compaction_contract")
        != MASC_MPQA_RELATION_COMPACTION_CONTRACT
        or producer_mpqa_relations.get("producer_parser_implementation")
        != "producer_regex_attribute_parser_v1"
        or producer_mpqa_relations.get("observed_linked_expression_count")
        != masc_mpqa_audit["observed_linked_expression_count"]
        or producer_mpqa_relations.get("admitted_relation_count")
        != masc_mpqa_audit["admitted_relation_count"]
        or producer_mpqa_relations.get("admitted_source_member_count")
        != masc_mpqa_audit["admitted_source_member_count"]
        or producer_mpqa_relations.get("admitted_attitude_count")
        != masc_mpqa_audit["admitted_attitude_count"]
        or producer_mpqa_relations.get("admitted_target_count")
        != masc_mpqa_audit["admitted_target_count"]
        or producer_mpqa_relations.get("record_count_by_split")
        != masc_mpqa_audit["record_count_by_split"]
        or producer_mpqa_relations.get("rejection_reason_counts")
        != masc_mpqa_audit["rejection_reason_counts"]
        or producer_mpqa_relations.get("admitted_source_ids_by_split")
        != expected_mpqa_source_ids
        or producer_mpqa_relations.get("span_status_count")
        != expected_mpqa_span_status
        or producer_mpqa_relations.get("partial_relation_admission_count") != 0
        or producer_mpqa_relations.get("inferred_relation_count") != 0
        or producer_mpqa_relations.get("unique_source_credit")
        != int(mpqa_relation_contract["unique_source_credit"])
        or producer_mpqa_relations.get("claim_scope")
        != mpqa_relation_contract["claim_scope"]
        or producer_mpqa_relations.get("complete_relation_alignment_required")
        is not True
        or producer_mpqa_relations.get("complete_sentence_semantics_claimed")
        is not False
        or producer_mpqa_relations.get("truth_claimed") is not False
        or producer_mpqa_relations.get("causal_relation_claimed") is not False
        or producer_mpqa_relations.get("temporal_relation_claimed") is not False
    ):
        raise ValueError("producer MASC MPQA-relation telemetry mismatch")
    producer_gum = manifest.get("gum_erst_discourse") or {}
    if (
        producer_gum.get("policy") != GUM_DISCOURSE_POLICY
        or producer_gum.get("relation_contract") != GUM_DISCOURSE_RELATION_CONTRACT
        or producer_gum.get("split_contract") != GUM_DISCOURSE_SPLIT_CONTRACT
        or producer_gum.get("projection_contract")
        != GUM_DISCOURSE_PROJECTION_CONTRACT
        or producer_gum.get("producer_parser_implementation")
        not in {None, "producer_elementtree_tab_parser_v1"}
        or producer_gum.get("parser_implementation")
        != "producer_elementtree_tab_parser_v1"
        or producer_gum.get("selected_source_sha256")
        != gum_audit["selected_source_sha256"]
        or producer_gum.get("selected_document_count")
        != gum_audit["selected_document_count"]
        or producer_gum.get("document_count_by_split")
        != gum_audit["document_count_by_split"]
        or producer_gum.get("record_count_by_split")
        != gum_audit["record_count_by_split"]
        or producer_gum.get("genre_count_by_split")
        != gum_audit["genre_count_by_split"]
        or producer_gum.get("relation_count_by_split")
        != gum_audit["relation_count_by_split"]
        or producer_gum.get("secondary_edge_count_by_split")
        != gum_audit["secondary_edge_count_by_split"]
        or producer_gum.get("relation_type_count_by_split")
        != gum_audit["relation_type_count_by_split"]
        or producer_gum.get("minimum_relation_count_by_split")
        != gum_audit["minimum_relation_count_by_split"]
        or producer_gum.get("official_nontrain_document_admission_count") != 0
        or producer_gum.get("partial_relation_admission_count") != 0
        or producer_gum.get("inferred_relation_count") != 0
        or producer_gum.get("claim_scope") != gum_contract["claim_scope"]
        or producer_gum.get("source_credit_unit") != "document"
        or producer_gum.get("derived_view_unique_source_credit") != 0
        or producer_gum.get("official_dev_test_quarantined") is not True
        or producer_gum.get("public_gum_or_disrpt_score_claimed") is not False
        or producer_gum.get("learned_competence_claimed") is not False
        or producer_gum.get("complete_sentence_semantics_claimed") is not False
        or producer_gum.get("truth_claimed") is not False
    ):
        raise ValueError("producer GUM eRST telemetry mismatch")
    producer_gum_entity = manifest.get("gum_entity_coreference") or {}
    if (
        producer_gum_entity.get("policy") != GUM_ENTITY_COREFERENCE_POLICY
        or producer_gum_entity.get("relation_contract")
        != GUM_ENTITY_COREFERENCE_RELATION_CONTRACT
        or producer_gum_entity.get("split_contract")
        != GUM_ENTITY_COREFERENCE_SPLIT_CONTRACT
        or producer_gum_entity.get("source_compaction_contract")
        != GUM_ENTITY_COREFERENCE_COMPACTION_CONTRACT
        or producer_gum_entity.get("parser_implementation")
        != "producer_webanno_tsv_state_machine_v1"
        or producer_gum_entity.get("selected_source_sha256")
        != gum_entity_contract["content_sha256"]
        or producer_gum_entity.get("selected_document_count")
        != int(gum_entity_contract["expected_selected_document_count"])
        or producer_gum_entity.get("counts_by_split")
        != gum_entity_audit["counts_by_split"]
        or producer_gum_entity.get("relation_type_count_by_split")
        != gum_entity_audit["relation_type_count_by_split"]
        or producer_gum_entity.get("component_size_distribution_by_split")
        != gum_entity_audit["component_size_distribution_by_split"]
        or producer_gum_entity.get("rejected_record_count") != 0
        or producer_gum_entity.get("official_nontrain_document_admission_count") != 0
        or producer_gum_entity.get("partial_component_admission_count") != 0
        or producer_gum_entity.get("inferred_relation_count") != 0
        or producer_gum_entity.get("claim_scope")
        != gum_entity_contract["claim_scope"]
        or producer_gum_entity.get("source_credit_unit") != "document"
        or producer_gum_entity.get("derived_view_unique_source_credit") != 0
        or producer_gum_entity.get("official_dev_test_quarantined") is not True
        or producer_gum_entity.get("cross_format_topology_required") is not True
        or producer_gum_entity.get("public_coreference_score_claimed") is not False
        or producer_gum_entity.get("cross_document_identity_claimed") is not False
        or producer_gum_entity.get("learned_competence_claimed") is not False
    ):
        raise ValueError("producer GUM entity/coreference telemetry mismatch")
    phase_runtime_ms["independent_source_reconstruction"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    raw_records = [
        json.loads(raw)
        for raw in candidate_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    importance_policy = fit_importance_policy(raw_records)
    if importance_policy != manifest.get("importance_policy"):
        raise ValueError("producer importance policy replay mismatch")
    verifier_sha256 = sha256_file(Path(__file__).resolve())
    verifier_family_receipts = verifier_family_identity_receipts()
    verifier_common_receipt = source_closure_receipt(
        source_path=Path(__file__).resolve(),
        source_label="scripts/kerc_semantic_corpus_verify.py",
        role="independent_semantic_admission_common",
        family="all_source_families",
        root_function="verify_candidate_common",
        external_paths={
            "kernel_protocol": ROOT / "scripts" / "kernel_english_protocol.py",
            "importance_policy": ROOT / "scripts" / "kerc_importance_policy.py",
            "residual_economics": ROOT / "scripts" / "kerc_residual_economics.py",
            "vcm_residual_lifecycle": ROOT / "scripts" / "vcm_semantic_memory.py",
        },
    )
    verifier_route_identities = {
        family: stable_hash(
            {
                "common": verifier_common_receipt["identity_sha256"],
                "family": receipt["identity_sha256"],
            }
        )
        for family, receipt in verifier_family_receipts.items()
    }
    phase_runtime_ms["candidate_load_and_identity_replay"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()
    accepted_rows: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    failures: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    counts_by_split_and_objective: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    semantic_cache_hits = 0
    semantic_cache_misses = 0
    semantic_cache_hits_by_family: Counter[str] = Counter()
    semantic_cache_misses_by_family: Counter[str] = Counter()
    semantic_store = (
        ContentObjectCache(
            cache_root / "verifier_objects.sqlite3",
            namespace=str(cache_cfg["verifier_role"])
            + ":"
            + str(cache_cfg["verifier_semantic_layer"]),
        )
        if cache_enabled
        else None
    )
    semantic_jobs: list[dict[str, Any]] = []
    ordered_failures: list[dict[str, Any]] = []
    for line_number, record in enumerate(raw_records, 1):
        try:
            source_id = str(record.get("provenance", {}).get("source_id") or "")
            if source_id in seen_source_ids:
                raise ValueError("duplicate candidate source id")
            expected_row = expected.get(source_id)
            if expected_row is None:
                raise ValueError("candidate absent from independent split reconstruction")
            seen_source_ids.add(source_id)
            dataset_id = str(record.get("provenance", {}).get("dataset_id") or "")
            source_key = (
                "dolly"
                if dataset_id == corpus["dolly"]["dataset_id"]
                else "masc"
                if dataset_id == corpus["masc"]["dataset_id"]
                else "gum"
                if dataset_id == corpus["gum"]["dataset_id"]
                else "oasst2"
                if dataset_id == corpus["oasst2"]["dataset_id"]
                else ""
            )
            if not source_key:
                raise ValueError("candidate dataset absent from frozen source contract")
            family = source_family(dataset_key=source_key, source_id=source_id)
            source = corpus[source_key]
            producer_family_identity = producer_family_receipts[family][
                "identity_sha256"
            ]
            verifier_route_identity = verifier_route_identities[family]
            semantic_key = object_key(
                role=str(cache_cfg["verifier_role"]),
                layer=str(cache_cfg["verifier_semantic_layer"]),
                dependencies={
                    "producer_family_identity": producer_family_identity,
                    "verifier_route_identity": verifier_route_identity,
                    "candidate": record,
                    "independent_expected_row": expected_row,
                },
            )
            cached_semantic = (
                semantic_store.get(semantic_key)
                if semantic_store is not None and not refresh_cache
                else None
            )
            if (
                isinstance(cached_semantic, dict)
                and isinstance(cached_semantic.get("canonical"), dict)
                and isinstance(cached_semantic.get("receipt"), dict)
                and cached_semantic["receipt"].get("accepted") is True
                and cached_semantic["canonical"].get("verification_receipt")
                == cached_semantic["receipt"]
                and cached_semantic["canonical"].get("provenance", {}).get(
                    "source_id"
                )
                == source_id
            ):
                canonical = cached_semantic["canonical"]
                receipt = cached_semantic["receipt"]
                accepted_rows.append((line_number, canonical, receipt))
                for objective, authorized in canonical["semantic_supervision"][
                    "objective_authority"
                ].items():
                    if authorized:
                        counts_by_split_and_objective[canonical["split"]][
                            objective
                        ] += 1
                semantic_cache_hits += 1
                semantic_cache_hits_by_family[family] += 1
                continue
            semantic_cache_misses += 1
            semantic_cache_misses_by_family[family] += 1
            expected_importance = predict_importance(record, importance_policy)
            semantic_jobs.append(
                {
                    "line_number": line_number,
                    "source_id": source_id,
                    "family": family,
                    "semantic_key": semantic_key,
                    "record": record,
                    "source": source,
                    "expected_row": expected_row,
                    "expected_importance": expected_importance,
                    "producer_family_identity": producer_family_identity,
                    "verifier_route_identity": verifier_route_identity,
                }
            )
        except Exception as exc:
            ordered_failures.append(
                {
                    "line_number": line_number,
                    "failure_code": str(
                        getattr(exc, "code", type(exc).__name__)
                    ),
                    "failure_message": str(exc)[:160],
                }
            )

    for result in execute_semantic_admission_jobs(
        semantic_jobs,
        worker_count=worker_count,
        batch_size=batch_size,
        max_in_flight_batches_per_worker=in_flight_batches,
    ):
        if not result["accepted"]:
            ordered_failures.append(result)
            continue
        canonical = result["canonical"]
        receipt = result["receipt"]
        accepted_rows.append(
            (int(result["line_number"]), canonical, receipt)
        )
        if semantic_store is not None:
            semantic_store.put(
                result["semantic_key"],
                {"canonical": canonical, "receipt": receipt},
            )
        for objective, authorized in canonical["semantic_supervision"][
            "objective_authority"
        ].items():
            if authorized:
                counts_by_split_and_objective[canonical["split"]][objective] += 1

    for index, failure in enumerate(
        sorted(ordered_failures, key=lambda row: int(row["line_number"]))
    ):
        failures[str(failure["failure_code"])] += 1
        if index < 10:
            failures[
                f"sample:{failure['line_number']}:{failure['failure_message']}"
            ] += 0
    canonical_records, receipts = order_semantic_admission_authority(accepted_rows)
    if semantic_store is not None:
        semantic_store.close()

    phase_runtime_ms["semantic_admission"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    semantic_execution_mode = (
        "cache_only_no_worker_jobs"
        if not semantic_jobs
        else "bounded_spawn_process_pool"
        if worker_count > 1
        else "serial"
    )
    phase_started = time.perf_counter()

    hard_gaps: list[str] = []
    if failures:
        hard_gaps.append("candidate_verification_failures")
    if set(expected) != seen_source_ids:
        hard_gaps.append("independent_expected_candidate_set_mismatch")
    if len(canonical_records) != int(manifest["candidate_records"]["row_count"]):
        hard_gaps.append("canonical_record_count_mismatch")
    floors = stage["semantic_supervision"]["minimum_decision_grade_records_by_split_and_objective"]
    for split, objective_floors in floors.items():
        for objective, floor in objective_floors.items():
            if counts_by_split_and_objective[split][objective] < int(floor):
                hard_gaps.append(f"decision_grade_floor_unmet:{split}:{objective}")
    codec_accounting = residual_codec_accounting(canonical_records)
    if codec_accounting != manifest.get("residual_codec_accounting"):
        hard_gaps.append("residual_codec_accounting_replay_mismatch")
    economics = corpus["residual_economics"]
    lambda_calibration = calibrate_allocation_lambda(
        [
            record["residual_supervision"]["rate_distortion_allocation"]
            for record in canonical_records
            if record["split"] == "private_dev"
        ],
        lambda_grid=economics["allocation_lambda_grid_bits"],
        maximum_importance_weighted_distortion=float(
            economics["maximum_dev_importance_weighted_structural_distortion"]
        ),
    )
    allocation_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for record in canonical_records:
        allocation_counts[record["split"]][
            record["residual_supervision"]["rate_distortion_allocation"][
                "selected_fidelity"
            ]
        ] += 1
    allocation_report = {
        "policy": "project_theseus_kerc_corpus_rate_distortion_allocation_v1",
        "lambda_calibration": lambda_calibration,
        "lambda_bits": float(lambda_calibration["selected_lambda"]),
        "selected_fidelity_counts_by_split": {
            split: dict(allocation_counts[split]) for split in SPLITS
        },
        "target_authority": "source_bound_structural_omission_not_semantic_utility",
        "semantic_utility_claim": False,
    }
    if allocation_report != manifest.get("rate_distortion_allocation"):
        hard_gaps.append("rate_distortion_allocation_replay_mismatch")

    phase_runtime_ms["aggregate_replay_and_gates"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    phase_started = time.perf_counter()

    if hard_gaps:
        canonical_written = 0
        records_sha256 = ""
        ledger_sha256 = ""
        catalog_sha256 = ""
    else:
        canonical_records.sort(key=lambda row: (SPLITS.index(row["split"]), row["record_sha256"]))
        receipts.sort(key=lambda row: row["receipt_id"])
        canonical_written, records_sha256 = write_jsonl_atomic(records_path, canonical_records)
        _, ledger_sha256 = write_jsonl_atomic(ledger_path, receipts)
        catalog = source_catalog(corpus)
        write_json_atomic(catalog_path, catalog)
        catalog_sha256 = sha256_file(catalog_path)
    phase_runtime_ms["canonical_serialization"] = round(
        (time.perf_counter() - phase_started) * 1000
    )
    report = {
        "policy": VERIFIER_POLICY,
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "config": relative(config_path),
        "runtime_config": relative(runtime_config_path),
        "producer_manifest_sha256": sha256_file(manifest_path),
        "candidate_records_sha256": sha256_file(candidate_path),
        "verifier_sha256": verifier_sha256,
        "producer_family_identities": producer_family_receipts,
        "producer_finalization_identity": producer_finalization_receipt,
        "verifier_common_identity": verifier_common_receipt,
        "verifier_family_identities": verifier_family_receipts,
        "verifier_route_identities": verifier_route_identities,
        "runtime_ms": round((time.perf_counter() - run_started) * 1000),
        "phase_runtime_ms": phase_runtime_ms,
        "semantic_admission_execution": {
            "mode": semantic_execution_mode,
            "worker_count": worker_count,
            "batch_size": batch_size,
            "max_in_flight_batches_per_worker": in_flight_batches,
            "cache_hit_count": semantic_cache_hits,
            "worker_job_count": len(semantic_jobs),
            "cache_publication": "single_writer_parent_process",
            "authoritative_order": "split_then_record_sha256_and_receipt_id",
            "resource_receipt": worker_resource_receipt,
        },
        "independent_expected_count": len(expected),
        "canonical_training_rows_written": canonical_written,
        "records": {"path": relative(records_path), "sha256": records_sha256},
        "verification_ledger": {"path": relative(ledger_path), "sha256": ledger_sha256},
        "semantic_source_catalog": {"path": relative(catalog_path), "sha256": catalog_sha256},
        "decision_grade_counts_by_split_and_objective": {
            split: dict(counts_by_split_and_objective[split]) for split in SPLITS
        },
        "residual_codec_accounting": codec_accounting,
        "importance_policy": importance_policy,
        "rate_distortion_allocation": allocation_report,
        "masc_contextual_frame_ambiguity": {
            "policy": MASC_CONTEXTUAL_FRAME_AMBIGUITY_POLICY,
            "fit_split": "private_train",
            "eligible_lexical_unit_count": len(expected_frame_priors),
            "bound_record_count_by_split": expected_frame_ambiguity_counts,
            "prior_sha256_by_lexical_unit": expected_frame_priors,
            "unresolved_ambiguity_record_count": 0,
            "calibrated_probability_claimed": False,
            "claim_scope": corpus["masc"]["contextual_frame_ambiguity"][
                "claim_scope"
            ],
        },
        "masc_composite_semantics": {
            "policy": "project_theseus_kerc_masc_composite_semantics_v1",
            "record_count_by_split": expected_composite_counts,
            "frame_count_distribution_by_split": expected_composite_frame_counts,
            "multi_node_program_count": expected_composite_total,
            "multi_root_program_count": expected_composite_total,
            "multi_claim_answer_count": expected_composite_total,
            "unique_source_credit": 0,
            "claim_scope": corpus["masc"]["composite_semantic_claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "inter_frame_discourse_edges_claimed": False,
            "independently_reconstructed_from_raw_graf": True,
        },
        "masc_decision_semantics": {
            "policy": MASC_DECISION_SEMANTICS_POLICY,
            "record_count_by_split": expected_decision_counts,
            "annotation_count_by_split_and_layer": expected_decision_layers,
            "missing_layer_record_count_by_split": expected_decision_missing,
            "typed_nonliteral_argument_count": expected_typed_count,
            "unique_source_credit": 0,
            "claim_scope": corpus["masc"]["decision_semantic_claim_scope"],
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "event_coreference_grouping_claimed": False,
            "source_declared_cross_annotation_links_resolved": False,
            "independently_reconstructed_from_raw_graf": True,
        },
        "masc_event_coreference": {
            "policy": MASC_EVENT_COREFERENCE_POLICY,
            "alignment_contract": MASC_EVENT_COREFERENCE_ALIGNMENT_CONTRACT,
            "source_compaction_contract": MASC_EVENT_COREFERENCE_COMPACTION_CONTRACT,
            "producer_alignment_implementation": producer_event_coreference[
                "producer_alignment_implementation"
            ],
            "verifier_alignment_implementation": masc_event_audit[
                "alignment_implementation"
            ],
            "alignment_implementations_independent": True,
            "observed_group_count": masc_event_audit["observed_group_count"],
            "observed_mention_count": masc_event_audit["observed_mention_count"],
            "admitted_group_count": masc_event_audit["admitted_group_count"],
            "admitted_mention_count": masc_event_audit["admitted_mention_count"],
            "record_count_by_split": masc_event_audit["record_count_by_split"],
            "mention_count_by_split": masc_event_audit["mention_count_by_split"],
            "admitted_source_ids_by_split": expected_event_source_ids,
            "rejected_group_count": masc_event_audit["rejected_group_count"],
            "rejected_groups": expected_event_rejected_rows,
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
            "producer_parser_implementation": producer_mpqa_relations[
                "producer_parser_implementation"
            ],
            "verifier_parser_implementation": masc_mpqa_audit[
                "parser_implementation"
            ],
            "parser_implementations_independent": True,
            "observed_linked_expression_count": masc_mpqa_audit[
                "observed_linked_expression_count"
            ],
            "admitted_relation_count": masc_mpqa_audit[
                "admitted_relation_count"
            ],
            "admitted_source_member_count": masc_mpqa_audit[
                "admitted_source_member_count"
            ],
            "admitted_attitude_count": masc_mpqa_audit[
                "admitted_attitude_count"
            ],
            "admitted_target_count": masc_mpqa_audit[
                "admitted_target_count"
            ],
            "record_count_by_split": masc_mpqa_audit["record_count_by_split"],
            "rejection_reason_counts": masc_mpqa_audit[
                "rejection_reason_counts"
            ],
            "admitted_source_ids_by_split": expected_mpqa_source_ids,
            "span_status_count": expected_mpqa_span_status,
            "partial_relation_admission_count": 0,
            "inferred_relation_count": 0,
            "unique_source_credit": int(
                mpqa_relation_contract["unique_source_credit"]
            ),
            "claim_scope": mpqa_relation_contract["claim_scope"],
            "complete_relation_alignment_required": True,
            "complete_sentence_semantics_claimed": False,
            "truth_claimed": False,
            "causal_relation_claimed": False,
            "temporal_relation_claimed": False,
            "independently_reconstructed_from_raw_mpqa": True,
        },
        "gum_erst_discourse": {
            **gum_audit,
            "producer_parser_implementation": producer_gum[
                "parser_implementation"
            ],
            "verifier_parser_implementation": gum_audit[
                "parser_implementation"
            ],
            "parser_implementations_independent": True,
            "claim_scope": gum_contract["claim_scope"],
            "source_credit_unit": "document",
            "derived_view_unique_source_credit": 0,
            "official_dev_test_quarantined": True,
            "public_gum_or_disrpt_score_claimed": False,
            "learned_competence_claimed": False,
            "independently_reconstructed_from_raw_rsd_and_xml": True,
        },
        "gum_entity_coreference": {
            **gum_entity_audit,
            "producer_parser_implementation": producer_gum_entity[
                "parser_implementation"
            ],
            "verifier_parser_implementation": gum_entity_audit[
                "parser_implementation"
            ],
            "parser_implementations_independent": True,
            "cross_format_topology_required": True,
            "claim_scope": gum_entity_contract["claim_scope"],
            "source_credit_unit": "document",
            "derived_view_unique_source_credit": 0,
            "official_dev_test_quarantined": True,
            "public_coreference_score_claimed": False,
            "cross_document_identity_claimed": False,
            "learned_competence_claimed": False,
            "independently_reconstructed_from_raw_tsv_and_conllu": True,
        },
        "verification_failures": dict(failures),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "score_semantics": "independent source replay and schema admission; not model capability",
        "hard_gaps": sorted(set(hard_gaps)),
    }
    write_json_atomic(report_path, report)
    if cache_enabled and report["trigger_state"] == "GREEN":
        publish_receipt(
            cache_root,
            role=str(cache_cfg["verifier_role"]),
            dependencies=cache_dependencies,
            outputs=cache_outputs,
            result_output_id="verification_report",
        )
    write_json_atomic(
        cache_root / "telemetry" / "verifier_last.json",
        {
            "policy": "project_theseus_kerc_incremental_cache_telemetry_v1",
            "run_cache_hit": False,
            "semantic_admission": {
                "hits": semantic_cache_hits,
                "misses": semantic_cache_misses,
                "entry_count": semantic_cache_hits + semantic_cache_misses,
                "hits_by_family": dict(sorted(semantic_cache_hits_by_family.items())),
                "misses_by_family": dict(
                    sorted(semantic_cache_misses_by_family.items())
                ),
                "producer_authority_reused": False,
                "mode": semantic_execution_mode,
                "worker_count": worker_count,
                "batch_size": batch_size,
                "cache_publication": "single_writer_parent_process",
            },
            "candidate_records_sha256": report["candidate_records_sha256"],
            "storage": cache_storage_telemetry(
                cache_root / "verifier_objects.sqlite3"
            ),
            "claim_scope": "cache execution telemetry only; not semantic or capability evidence",
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--bypass-run-cache", action="store_true")
    parser.add_argument("--semantic-workers", type=int)
    parser.add_argument("--semantic-batch-size", type=int)
    args = parser.parse_args()
    report = verify(
        resolve(args.config),
        runtime_config_path=resolve(args.runtime_config),
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        bypass_run_cache=args.bypass_run_cache,
        semantic_workers=args.semantic_workers,
        semantic_batch_size=args.semantic_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
