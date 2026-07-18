#!/usr/bin/env python3
"""Reconstruct complete manual MASC event-coreference groups for KERC.

The distributed GrAF event layer preserves event mentions but drops the named
GATE annotation-set grouping. This producer-side owner recovers only complete
manual groups using global token-sequence alignment. It never infers a relation
from co-occurrence, event type, or document context.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


POLICY = "project_theseus_kerc_masc_manual_event_coreference_v1"
ALIGNMENT_CONTRACT = "complete_named_gate_group_dual_independent_token_alignment_v1"
COMPACTION_CONTRACT = "uniform_radius_mention_centered_source_windows_v1"
EXCLUDED_ANNOTATION_SETS = {"", "Other Events", "Original markups"}
GRAF = "{http://www.xces.org/ns/GrAF/1.0/}"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)*")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold().replace("\u2019", "'"), match.start(), match.end())
        for match in TOKEN_RE.finditer(text)
    ]


def _gate_document(path: Path) -> tuple[str, list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    text_with_nodes = root.find("TextWithNodes")
    if text_with_nodes is None:
        raise ValueError(f"GATE document has no TextWithNodes: {path}")
    text = text_with_nodes.text or ""
    node_offsets: dict[str, int] = {}
    for node in text_with_nodes:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in node_offsets:
            raise ValueError(f"GATE node identity invalid: {path}:{node_id}")
        node_offsets[node_id] = len(text)
        text += node.tail or ""
    groups: list[dict[str, Any]] = []
    for annotation_set in root.findall("AnnotationSet"):
        name = str(annotation_set.get("Name") or "").strip()
        if name in EXCLUDED_ANNOTATION_SETS:
            continue
        mentions: list[dict[str, Any]] = []
        for annotation in annotation_set.findall("Annotation"):
            start_node = str(annotation.get("StartNode") or "")
            end_node = str(annotation.get("EndNode") or "")
            if start_node not in node_offsets or end_node not in node_offsets:
                raise ValueError(f"GATE annotation references an unknown node: {path}")
            start, end = node_offsets[start_node], node_offsets[end_node]
            features: dict[str, str] = {}
            for feature in annotation.findall("Feature"):
                feature_name = feature.find("Name")
                feature_value = feature.find("Value")
                key = "" if feature_name is None else "".join(feature_name.itertext()).strip()
                value = "" if feature_value is None else "".join(feature_value.itertext()).strip()
                if key:
                    features[key] = value
            raw = {
                "annotation_id": str(annotation.get("Id") or ""),
                "event_type": str(annotation.get("Type") or "").strip(),
                "start_node": start_node,
                "end_node": end_node,
                "gate_start": start,
                "gate_end": end,
                "features": dict(sorted(features.items())),
            }
            raw["source_annotation_sha256"] = _stable_hash(raw)
            mentions.append(raw)
        if len(mentions) >= 2:
            mentions.sort(key=lambda row: (row["gate_start"], row["gate_end"], row["annotation_id"]))
            groups.append({"annotation_set_name": name, "mentions": mentions})
    return text, groups


def _sentence_spans(path: Path) -> list[tuple[int, int]]:
    root = ET.parse(path).getroot()
    spans = {
        tuple(int(value) for value in str(region.get("anchors") or "").split())
        for region in root.findall(GRAF + "region")
    }
    if any(len(span) != 2 or span[0] >= span[1] for span in spans):
        raise ValueError(f"invalid MASC sentence regions: {path}")
    return sorted(spans)


def _global_alignment(
    gate_text: str, target_text: str, mentions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    gate_tokens = _tokens(gate_text)
    target_tokens = _tokens(target_text)
    matcher = difflib.SequenceMatcher(
        a=[row[0] for row in gate_tokens],
        b=[row[0] for row in target_tokens],
        autojunk=False,
    )
    token_map = {
        left + offset: right + offset
        for left, right, size in matcher.get_matching_blocks()
        for offset in range(size)
    }
    aligned: list[dict[str, Any]] = []
    failure_count = 0
    for mention in mentions:
        indices = [
            index
            for index, (_token, start, end) in enumerate(gate_tokens)
            if start < int(mention["gate_end"]) and end > int(mention["gate_start"])
        ]
        mapped = [token_map.get(index) for index in indices]
        if (
            not indices
            or any(index is None for index in mapped)
            or mapped != list(range(int(mapped[0]), int(mapped[0]) + len(mapped)))
        ):
            failure_count += 1
            continue
        target_start = target_tokens[int(mapped[0])][1]
        target_end = target_tokens[int(mapped[-1])][2]
        expected = [gate_tokens[index][0] for index in indices]
        observed = [row[0] for row in _tokens(target_text[target_start:target_end])]
        if expected != observed:
            failure_count += 1
            continue
        aligned.append(
            {
                **mention,
                "document_start": target_start,
                "document_end": target_end,
                "source_text": target_text[target_start:target_end],
            }
        )
    return aligned, failure_count


def _build_excerpt(
    target_text: str,
    sentence_spans: list[tuple[int, int]],
    mentions: list[dict[str, Any]],
    *,
    maximum_characters: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], int]:
    sentence_for_mention: list[tuple[int, int]] = []
    for mention in mentions:
        containing = [
            span
            for span in sentence_spans
            if span[0] <= int(mention["document_start"])
            and int(mention["document_end"]) <= span[1]
        ]
        if len(containing) != 1:
            raise ValueError("event mention does not map to exactly one source sentence")
        sentence_for_mention.append(containing[0])
    def windows(radius: int) -> list[tuple[int, int]]:
        candidates = sorted(
            (
                max(sentence[0], int(mention["document_start"]) - radius),
                min(sentence[1], int(mention["document_end"]) + radius),
            )
            for mention, sentence in zip(mentions, sentence_for_mention)
        )
        merged: list[tuple[int, int]] = []
        for start, end in candidates:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def rendered_length(selected: list[tuple[int, int]]) -> int:
        return sum(end - start for start, end in selected) + max(0, len(selected) - 1)

    minimum_windows = windows(0)
    if not minimum_windows or rendered_length(minimum_windows) > maximum_characters:
        raise ValueError("event group mentions exceed the source-length contract")
    low, high = 0, max(end - start for start, end in sentence_spans)
    while low < high:
        midpoint = (low + high + 1) // 2
        if rendered_length(windows(midpoint)) <= maximum_characters:
            low = midpoint
        else:
            high = midpoint - 1
    context_radius = low
    selected_windows = windows(context_radius)
    parts: list[str] = []
    excerpt_windows: list[dict[str, Any]] = []
    excerpt_offsets: dict[tuple[int, int], int] = {}
    cursor = 0
    for start, end in selected_windows:
        if parts:
            parts.append("\n")
            cursor += 1
        text = target_text[start:end]
        excerpt_offsets[(start, end)] = cursor
        excerpt_windows.append(
            {
                "document_span": [start, end],
                "excerpt_span": [cursor, cursor + len(text)],
                "source_window_sha256": _stable_hash(text),
            }
        )
        parts.append(text)
        cursor += len(text)
    excerpt = "".join(parts)
    if not excerpt or len(excerpt) > maximum_characters:
        raise ValueError("event group excerpt is outside the source-length contract")
    normalized_mentions: list[dict[str, Any]] = []
    for mention in mentions:
        containing_windows = [
            window
            for window in selected_windows
            if window[0] <= int(mention["document_start"])
            and int(mention["document_end"]) <= window[1]
        ]
        if len(containing_windows) != 1:
            raise ValueError("event mention does not map to exactly one excerpt window")
        window = containing_windows[0]
        excerpt_start = excerpt_offsets[window] + int(mention["document_start"]) - window[0]
        excerpt_end = excerpt_offsets[window] + int(mention["document_end"]) - window[0]
        normalized_mentions.append(
            {
                "annotation_id": mention["annotation_id"],
                "event_type": mention["event_type"],
                "source_text": mention["source_text"],
                "target_spans": [[excerpt_start, excerpt_end]],
                "source_annotation_sha256": mention["source_annotation_sha256"],
            }
        )
    normalized_mentions.sort(key=lambda row: (row["target_spans"], row["annotation_id"]))
    return excerpt, excerpt_windows, normalized_mentions, context_radius


def reconstruct_event_coreference_groups(
    *,
    original_event_root: Path,
    data_root: Path,
    document_map: Mapping[str, str],
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    observed_mentions = 0
    admitted_mentions = 0
    rejected: list[dict[str, Any]] = []
    observed_groups = 0
    for filename, document_id in sorted(document_map.items()):
        event_path = original_event_root / filename
        text_path = data_root / f"{document_id}.txt"
        sentence_path = data_root / f"{document_id}-s.xml"
        if not event_path.is_file() or not text_path.is_file() or not sentence_path.is_file():
            raise ValueError(f"MASC event-coreference source mapping is incomplete: {filename}")
        gate_text, groups = _gate_document(event_path)
        target_text = text_path.read_text(encoding="utf-8", errors="replace")
        sentences = _sentence_spans(sentence_path)
        split = (
            "private_dev"
            if document_id in private_dev_documents
            else "private_eval"
            if document_id in private_eval_documents
            else "private_train"
        )
        for group in groups:
            observed_groups += 1
            observed_mentions += len(group["mentions"])
            aligned, failure_count = _global_alignment(
                gate_text, target_text, group["mentions"]
            )
            if failure_count or len(aligned) != len(group["mentions"]):
                rejected.append(
                    {
                        "document_id": document_id,
                        "annotation_set_name": group["annotation_set_name"],
                        "mention_count": len(group["mentions"]),
                        "failure_count": failure_count,
                        "reason": "incomplete_global_token_alignment",
                    }
                )
                continue
            try:
                excerpt, excerpt_windows, mentions, context_radius = _build_excerpt(
                    target_text,
                    sentences,
                    aligned,
                    maximum_characters=maximum_characters,
                )
            except ValueError as exc:
                rejected.append(
                    {
                        "document_id": document_id,
                        "annotation_set_name": group["annotation_set_name"],
                        "mention_count": len(group["mentions"]),
                        "failure_count": len(group["mentions"]),
                        "reason": str(exc),
                    }
                )
                continue
            set_receipt = {
                "original_annotation_file_sha256": _file_hash(event_path),
                "annotation_set_name": group["annotation_set_name"],
                "mention_receipts": [
                    mention["source_annotation_sha256"] for mention in mentions
                ],
            }
            group_identity = _stable_hash(
                {
                    "document_id": document_id,
                    "annotation_set": set_receipt,
                }
            )
            group_concept = "masc.event_coreference." + group_identity.split(":", 1)[1][:24]
            annotation = {
                "policy": POLICY,
                "alignment_contract": ALIGNMENT_CONTRACT,
                "document_id": document_id,
                "original_annotation_filename": filename,
                "original_annotation_file_sha256": set_receipt[
                    "original_annotation_file_sha256"
                ],
                "distributed_document_sha256": _file_hash(text_path),
                "sentence_graph_sha256": _file_hash(sentence_path),
                "annotation_set_name": group["annotation_set_name"],
                "group_concept": group_concept,
                "source_text": excerpt,
                "excerpt_windows": excerpt_windows,
                "source_compaction_contract": COMPACTION_CONTRACT,
                "maximum_source_characters": maximum_characters,
                "uniform_context_radius_characters": context_radius,
                "mentions": mentions,
                "complete_group_alignment": True,
                "missingness": {
                    "event_coreference_grouping": False,
                    "complete_sentence_semantics": True,
                    "truth": True,
                    "causal_relation": True,
                    "temporal_relation": True,
                },
            }
            selection_key = _stable_hash(annotation)
            by_split[split].append(
                {
                    "source_id": "masc-event-coref:" + selection_key.split(":", 1)[1][:24],
                    "split": split,
                    "selection_key": selection_key,
                    "annotation": annotation,
                }
            )
            admitted_mentions += len(mentions)
    for rows in by_split.values():
        rows.sort(key=lambda row: row["selection_key"])
    audit = {
        "policy": POLICY,
        "alignment_implementation": "producer_global_sequence_matcher_v1",
        "observed_group_count": observed_groups,
        "observed_mention_count": observed_mentions,
        "admitted_group_count": sum(len(rows) for rows in by_split.values()),
        "admitted_mention_count": admitted_mentions,
        "record_count_by_split": {
            split: len(rows) for split, rows in by_split.items()
        },
        "mention_count_by_split": {
            split: sum(len(row["annotation"]["mentions"]) for row in rows)
            for split, rows in by_split.items()
        },
        "rejected_group_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected_groups": rejected,
        "partial_group_admission_count": 0,
        "cooccurrence_inferred_relation_count": 0,
    }
    return by_split, audit
