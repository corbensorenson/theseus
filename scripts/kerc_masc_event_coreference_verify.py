#!/usr/bin/env python3
"""Independent MASC event-coreference reconstruction for KERC admission.

This verifier-side owner deliberately avoids the producer's global sequence
alignment. Each manual mention is located by exact normalized token sequence,
then disambiguated with local left/right context and a strict score margin.
"""

from __future__ import annotations

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
CONTEXT_WINDOW = 16
MINIMUM_CONTEXT_MARGIN = 1


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _lex(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in TOKEN_RE.finditer(text):
        rows.append(
            {
                "token": match.group(0).casefold().replace("\u2019", "'"),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return rows


def _read_gate(path: Path) -> tuple[str, list[dict[str, Any]]]:
    document = ET.parse(path).getroot()
    content = document.find("TextWithNodes")
    if content is None:
        raise ValueError(f"missing serialized GATE content: {path}")
    text = content.text or ""
    offsets: dict[str, int] = {}
    for marker in list(content):
        marker_id = str(marker.attrib.get("id") or "")
        if not marker_id or marker_id in offsets:
            raise ValueError(f"invalid serialized GATE marker: {path}:{marker_id}")
        offsets[marker_id] = len(text)
        text = text + (marker.tail or "")
    groups: list[dict[str, Any]] = []
    for annotation_set in document.findall("AnnotationSet"):
        set_name = str(annotation_set.attrib.get("Name") or "").strip()
        if set_name in EXCLUDED_ANNOTATION_SETS:
            continue
        members: list[dict[str, Any]] = []
        for annotation in annotation_set.findall("Annotation"):
            start_marker = str(annotation.attrib.get("StartNode") or "")
            end_marker = str(annotation.attrib.get("EndNode") or "")
            if start_marker not in offsets or end_marker not in offsets:
                raise ValueError(f"unresolved GATE annotation marker: {path}")
            feature_map: dict[str, str] = {}
            for feature in annotation.findall("Feature"):
                name_element = feature.find("Name")
                value_element = feature.find("Value")
                name = (
                    ""
                    if name_element is None
                    else "".join(name_element.itertext()).strip()
                )
                value = (
                    ""
                    if value_element is None
                    else "".join(value_element.itertext()).strip()
                )
                if name:
                    feature_map[name] = value
            member = {
                "annotation_id": str(annotation.attrib.get("Id") or ""),
                "event_type": str(annotation.attrib.get("Type") or "").strip(),
                "start_node": start_marker,
                "end_node": end_marker,
                "gate_start": offsets[start_marker],
                "gate_end": offsets[end_marker],
                "features": dict(sorted(feature_map.items())),
            }
            member["source_annotation_sha256"] = _digest(member)
            members.append(member)
        if len(members) > 1:
            members.sort(
                key=lambda row: (
                    row["gate_start"], row["gate_end"], row["annotation_id"]
                )
            )
            groups.append({"annotation_set_name": set_name, "mentions": members})
    return text, groups


def _read_sentence_boundaries(path: Path) -> list[tuple[int, int]]:
    graph = ET.parse(path).getroot()
    boundaries: set[tuple[int, int]] = set()
    for region in graph.findall(GRAF + "region"):
        anchors = str(region.attrib.get("anchors") or "").split()
        if len(anchors) != 2:
            raise ValueError(f"invalid sentence-region anchors: {path}")
        start, end = int(anchors[0]), int(anchors[1])
        if start >= end:
            raise ValueError(f"empty sentence-region anchors: {path}")
        boundaries.add((start, end))
    return sorted(boundaries)


def _context_score(
    gate_tokens: list[dict[str, Any]],
    target_tokens: list[dict[str, Any]],
    gate_indices: list[int],
    target_start_index: int,
) -> int:
    score = 0
    target_end_index = target_start_index + len(gate_indices) - 1
    for distance in range(1, CONTEXT_WINDOW + 1):
        gate_left = gate_indices[0] - distance
        target_left = target_start_index - distance
        if gate_left >= 0 and target_left >= 0:
            score += int(
                gate_tokens[gate_left]["token"] == target_tokens[target_left]["token"]
            )
        gate_right = gate_indices[-1] + distance
        target_right = target_end_index + distance
        if gate_right < len(gate_tokens) and target_right < len(target_tokens):
            score += int(
                gate_tokens[gate_right]["token"] == target_tokens[target_right]["token"]
            )
    return score


def _locate_mentions_locally(
    gate_text: str, target_text: str, mentions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    gate_tokens = _lex(gate_text)
    target_tokens = _lex(target_text)
    target_values = [row["token"] for row in target_tokens]
    aligned: list[dict[str, Any]] = []
    failures = 0
    used_spans: set[tuple[int, int]] = set()
    for mention in mentions:
        gate_indices = [
            index
            for index, token in enumerate(gate_tokens)
            if int(token["start"]) < int(mention["gate_end"])
            and int(token["end"]) > int(mention["gate_start"])
        ]
        sequence = [gate_tokens[index]["token"] for index in gate_indices]
        candidates: list[tuple[int, int]] = []
        if sequence:
            for start_index in range(len(target_tokens) - len(sequence) + 1):
                if target_values[start_index : start_index + len(sequence)] == sequence:
                    candidates.append(
                        (
                            _context_score(
                                gate_tokens,
                                target_tokens,
                                gate_indices,
                                start_index,
                            ),
                            start_index,
                        )
                    )
        candidates.sort(key=lambda row: (-row[0], row[1]))
        margin = (
            candidates[0][0] - candidates[1][0]
            if len(candidates) > 1
            else candidates[0][0] + 1
            if candidates
            else 0
        )
        if not candidates or margin < MINIMUM_CONTEXT_MARGIN:
            failures += 1
            continue
        start_index = candidates[0][1]
        end_index = start_index + len(sequence) - 1
        document_span = (
            int(target_tokens[start_index]["start"]),
            int(target_tokens[end_index]["end"]),
        )
        if document_span in used_spans:
            failures += 1
            continue
        used_spans.add(document_span)
        aligned.append(
            {
                **mention,
                "document_start": document_span[0],
                "document_end": document_span[1],
                "source_text": target_text[document_span[0] : document_span[1]],
            }
        )
    return aligned, failures


def _excerpt_from_sentences(
    document_text: str,
    sentence_boundaries: list[tuple[int, int]],
    mentions: list[dict[str, Any]],
    maximum_characters: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], int]:
    containers: list[tuple[int, int]] = []
    for mention in mentions:
        matches = [
            boundary
            for boundary in sentence_boundaries
            if boundary[0] <= int(mention["document_start"])
            and int(mention["document_end"]) <= boundary[1]
        ]
        if len(matches) != 1:
            raise ValueError("event mention lacks one unambiguous source sentence")
        containers.append(matches[0])
    def assemble(radius: int) -> list[tuple[int, int]]:
        spans = sorted(
            (
                max(container[0], int(mention["document_start"]) - radius),
                min(container[1], int(mention["document_end"]) + radius),
            )
            for mention, container in zip(mentions, containers)
        )
        output: list[list[int]] = []
        for start, end in spans:
            if output and start <= output[-1][1]:
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        return [(start, end) for start, end in output]

    def size(spans: list[tuple[int, int]]) -> int:
        return sum(end - start for start, end in spans) + max(len(spans) - 1, 0)

    exact_spans = assemble(0)
    if not exact_spans or size(exact_spans) > maximum_characters:
        raise ValueError("event group mention surfaces violate source-length bound")
    floor = 0
    ceiling = max(end - start for start, end in sentence_boundaries)
    while floor < ceiling:
        probe = floor + (ceiling - floor + 1) // 2
        if size(assemble(probe)) <= maximum_characters:
            floor = probe
        else:
            ceiling = probe - 1
    context_radius = floor
    ordered_boundaries = assemble(context_radius)
    excerpt_chunks: list[str] = []
    window_records: list[dict[str, Any]] = []
    starts: dict[tuple[int, int], int] = {}
    cursor = 0
    for boundary in ordered_boundaries:
        if excerpt_chunks:
            excerpt_chunks.append("\n")
            cursor += 1
        sentence = document_text[boundary[0] : boundary[1]]
        starts[boundary] = cursor
        window_records.append(
            {
                "document_span": [boundary[0], boundary[1]],
                "excerpt_span": [cursor, cursor + len(sentence)],
                "source_window_sha256": _digest(sentence),
            }
        )
        excerpt_chunks.append(sentence)
        cursor += len(sentence)
    excerpt = "".join(excerpt_chunks)
    if not excerpt or len(excerpt) > maximum_characters:
        raise ValueError("event group excerpt violates source-length bound")
    normalized: list[dict[str, Any]] = []
    for mention in mentions:
        matches = [
            boundary
            for boundary in ordered_boundaries
            if boundary[0] <= int(mention["document_start"])
            and int(mention["document_end"]) <= boundary[1]
        ]
        if len(matches) != 1:
            raise ValueError("event mention lacks one unambiguous excerpt window")
        boundary = matches[0]
        relative_start = starts[boundary] + int(mention["document_start"]) - boundary[0]
        relative_end = starts[boundary] + int(mention["document_end"]) - boundary[0]
        normalized.append(
            {
                "annotation_id": mention["annotation_id"],
                "event_type": mention["event_type"],
                "source_text": mention["source_text"],
                "target_spans": [[relative_start, relative_end]],
                "source_annotation_sha256": mention["source_annotation_sha256"],
            }
        )
    normalized.sort(key=lambda row: (row["target_spans"], row["annotation_id"]))
    return excerpt, window_records, normalized, context_radius


def independently_reconstruct_event_coreference_groups(
    *,
    original_event_root: Path,
    data_root: Path,
    document_map: Mapping[str, str],
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    output: dict[str, list[dict[str, Any]]] = {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    observed_groups = 0
    observed_mentions = 0
    admitted_mentions = 0
    rejected: list[dict[str, Any]] = []
    for filename, document_id in sorted(document_map.items()):
        original_path = original_event_root / filename
        distributed_text_path = data_root / f"{document_id}.txt"
        sentence_graph_path = data_root / f"{document_id}-s.xml"
        if not (
            original_path.is_file()
            and distributed_text_path.is_file()
            and sentence_graph_path.is_file()
        ):
            raise ValueError(f"independent event source mapping is incomplete: {filename}")
        gate_text, groups = _read_gate(original_path)
        distributed_text = distributed_text_path.read_text(
            encoding="utf-8", errors="replace"
        )
        sentence_boundaries = _read_sentence_boundaries(sentence_graph_path)
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
            aligned, failures = _locate_mentions_locally(
                gate_text, distributed_text, group["mentions"]
            )
            if failures or len(aligned) != len(group["mentions"]):
                rejected.append(
                    {
                        "document_id": document_id,
                        "annotation_set_name": group["annotation_set_name"],
                        "mention_count": len(group["mentions"]),
                        "failure_count": failures,
                        "reason": "incomplete_local_context_alignment",
                    }
                )
                continue
            try:
                excerpt, window_records, mentions, context_radius = _excerpt_from_sentences(
                    distributed_text,
                    sentence_boundaries,
                    aligned,
                    maximum_characters,
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
                "original_annotation_file_sha256": _file_digest(original_path),
                "annotation_set_name": group["annotation_set_name"],
                "mention_receipts": [
                    mention["source_annotation_sha256"] for mention in mentions
                ],
            }
            identity = _digest(
                {"document_id": document_id, "annotation_set": set_receipt}
            )
            concept = "masc.event_coreference." + identity.split(":", 1)[1][:24]
            annotation = {
                "policy": POLICY,
                "alignment_contract": ALIGNMENT_CONTRACT,
                "document_id": document_id,
                "original_annotation_filename": filename,
                "original_annotation_file_sha256": set_receipt[
                    "original_annotation_file_sha256"
                ],
                "distributed_document_sha256": _file_digest(distributed_text_path),
                "sentence_graph_sha256": _file_digest(sentence_graph_path),
                "annotation_set_name": group["annotation_set_name"],
                "group_concept": concept,
                "source_text": excerpt,
                "excerpt_windows": window_records,
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
            selection_key = _digest(annotation)
            output[split].append(
                {
                    "source_id": "masc-event-coref:" + selection_key.split(":", 1)[1][:24],
                    "split": split,
                    "selection_key": selection_key,
                    "annotation": annotation,
                }
            )
            admitted_mentions += len(mentions)
    for rows in output.values():
        rows.sort(key=lambda row: row["selection_key"])
    audit = {
        "policy": POLICY,
        "alignment_implementation": "verifier_local_context_margin_v1",
        "observed_group_count": observed_groups,
        "observed_mention_count": observed_mentions,
        "admitted_group_count": sum(len(rows) for rows in output.values()),
        "admitted_mention_count": admitted_mentions,
        "record_count_by_split": {split: len(rows) for split, rows in output.items()},
        "mention_count_by_split": {
            split: sum(len(row["annotation"]["mentions"]) for row in rows)
            for split, rows in output.items()
        },
        "rejected_group_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected_groups": rejected,
        "partial_group_admission_count": 0,
        "cooccurrence_inferred_relation_count": 0,
    }
    return output, audit
