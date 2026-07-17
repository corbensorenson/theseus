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


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
    output: dict[str, dict[str, Any]] = {}
    for split, count in source["records_by_split"].items():
        selected = sorted(by_split[split], key=lambda row: row["selection_key"])[: int(count)]
        if len(selected) != int(count):
            raise ValueError(f"MASC independent split reconstruction is incomplete: {split}")
        interactions = independent_interaction_predecessors(selected)
        for row in selected:
            output[row["source_id"]] = {
                **row,
                "split": split,
                "interaction_annotation": interactions.get(row["selection_key"]),
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


def observed_masc_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict) and value.get("type") == "handle":
        return {"type": "handle", "value": str(value.get("value") or "")}
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
    expected = {
        **independent_dolly_assignments(
            corpus["dolly"],
            maximum_characters,
            reserved_source_hashes=grounded_source_hashes,
        ),
        **grounded_expected,
        **independent_masc_assignments(corpus["masc"], maximum_characters),
        **independent_oasst_assignments(
            corpus["oasst2"],
            reserved_groups={row["source_group"] for row in behavior_expected.values()},
        ),
        **behavior_expected,
    }
    verifier_sha256 = sha256_file(Path(__file__).resolve())
    canonical_records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    seen_source_ids: set[str] = set()
    counts_by_split_and_objective: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for line_number, raw in enumerate(candidate_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(raw)
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
