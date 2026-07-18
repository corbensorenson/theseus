#!/usr/bin/env python3
"""Independently replay and admit the licensed KERC semantic corpus.

The verifier intentionally does not import the producer. It reparses the pinned
raw datasets, reconstructs split membership and semantic annotations, checks the
candidate packet against that evidence, and only then writes canonical rows.
"""

from __future__ import annotations

import argparse
import base64
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
    TRAINING_OBJECTIVES,
    TRAINING_VERIFICATION_POLICY,
    canonical_json,
    stable_hash,
    training_semantic_payload_sha256,
    validate_training_record,
)
from kerc_importance_policy import fit_importance_policy, predict_importance
from kerc_residual_economics import (
    calibrate_allocation_lambda,
    residual_wire_bytes,
    validate_structural_rate_distortion_allocation,
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


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def source_catalog(corpus: dict[str, Any]) -> dict[str, Any]:
    sources = []
    for key in ("dolly", "masc", "oasst2"):
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
            }
        )
    return {"policy": KERC_SOURCE_CATALOG_POLICY, "sources": sources}


def verify(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    raw_records = [
        json.loads(raw)
        for raw in candidate_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    importance_policy = fit_importance_policy(raw_records)
    if importance_policy != manifest.get("importance_policy"):
        raise ValueError("producer importance policy replay mismatch")
    verifier_sha256 = sha256_file(Path(__file__).resolve())
    canonical_records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    counts_by_split_and_objective: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for line_number, record in enumerate(raw_records, 1):
        try:
            expected_importance = predict_importance(record, importance_policy)
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
                else "oasst2"
                if dataset_id == corpus["oasst2"]["dataset_id"]
                else ""
            )
            if not source_key:
                raise ValueError("candidate dataset absent from frozen source contract")
            source = corpus[source_key]
            provenance = record["provenance"]
            if provenance.get("dataset_revision") != source["dataset_revision"] or provenance.get("license_spdx") != source["license_spdx"]:
                raise ValueError("candidate provenance mismatch")
            semantic = record.get("semantic_supervision") or {}
            if semantic.get("annotation_source_sha256") != source["content_sha256"] or semantic.get("producer_artifact_sha256") != manifest["producer_sha256"]:
                raise ValueError("candidate semantic source binding mismatch")
            replay = (
                verify_dolly_grounded_record(record, source, expected_row)
                if source_id.startswith("dolly-grounded:")
                else verify_dolly_record(record, source, expected_row)
                if source_key == "dolly"
                else verify_masc_composite_record(record, source, expected_row)
                if source_id.startswith("masc-composite:")
                else verify_masc_decision_record(record, source, expected_row)
                if source_id.startswith("masc-decision:")
                else verify_masc_record(record, source, expected_row)
                if source_key == "masc"
                else verify_oasst_behavior_record(record, source, expected_row)
                if source_id.startswith("oasst2-behavior:")
                else verify_oasst_record(record, source, expected_row)
            )
            evidence_sha256 = stable_hash(
                {
                    "policy": VERIFIER_POLICY,
                    "verifier_sha256": verifier_sha256,
                    "source_content_sha256": source["content_sha256"],
                    "source_id": source_id,
                    "split": record["split"],
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
            canonical = validate_training_record(record)
            canonical_records.append(canonical)
            receipts.append(receipt)
            for objective, authorized in canonical["semantic_supervision"]["objective_authority"].items():
                if authorized:
                    counts_by_split_and_objective[canonical["split"]][objective] += 1
        except Exception as exc:
            failures[str(getattr(exc, "code", type(exc).__name__))] += 1
            if sum(failures.values()) <= 10:
                failures[f"sample:{line_number}:{str(exc)[:160]}"] += 0

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

    output_root = resolve(corpus["output_root"])
    records_path = output_root / "records.jsonl"
    ledger_path = output_root / "verification_ledger.jsonl"
    catalog_path = output_root / "semantic_source_catalog.json"
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
    report = {
        "policy": VERIFIER_POLICY,
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "config": relative(config_path),
        "producer_manifest_sha256": sha256_file(manifest_path),
        "candidate_records_sha256": sha256_file(candidate_path),
        "verifier_sha256": verifier_sha256,
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
        "verification_failures": dict(failures),
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "score_semantics": "independent source replay and schema admission; not model capability",
        "hard_gaps": sorted(set(hard_gaps)),
    }
    report_path = ROOT / "reports" / "runtime" / "kerc_semantic_corpus_verification.json"
    write_json_atomic(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = verify(resolve(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
