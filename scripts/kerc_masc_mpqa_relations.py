#!/usr/bin/env python3
"""Reconstruct complete human MPQA attribution chains for KERC.

The MASC GrAF conversion preserves MPQA fields but loses the original ID-based
expression -> attitude -> target links. This producer reads the licensed raw
MPQA files and admits only complete, uniquely resolved chains. It does not
infer attribution, attitude, target, scope, truth, or relations from text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_masc_manual_mpqa_relation_v1"
RELATION_CONTRACT = "complete_manual_mpqa_expression_attitude_target_source_chain_v1"
COMPACTION_CONTRACT = "uniform_radius_relation_member_source_windows_v1"
ATTRIBUTE_RE = re.compile(r'([A-Za-z0-9_-]+)="([^"]*)"')
SPAN_RE = re.compile(r"^(\d+),(\d+)$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _references(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_annotations(path: Path, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) < 4 or not parts[3].strip().startswith("GATE_"):
            continue
        span = SPAN_RE.fullmatch(parts[1].strip())
        if span is None:
            raise ValueError(f"invalid MPQA span at {path}:{line_number}")
        start, end = int(span.group(1)), int(span.group(2))
        if not 0 <= start <= end <= len(source):
            raise ValueError(f"out-of-range MPQA span at {path}:{line_number}")
        fields = dict(ATTRIBUTE_RE.findall(parts[4] if len(parts) > 4 else ""))
        receipt = {
            "line_id": parts[0].strip(),
            "line_number": line_number,
            "label": parts[3].strip().removeprefix("GATE_"),
            "start": start,
            "end": end,
            "fields": dict(sorted(fields.items())),
        }
        rows.append({**receipt, "source_annotation_sha256": _digest(receipt)})
    return rows


def _index_annotations(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        identity = str(row["fields"].get("id") or "").strip()
        if identity:
            index[identity][str(row["label"])].append(row)
    return index


def _resolve_one(
    index: dict[str, dict[str, list[dict[str, Any]]]],
    identity: str,
    label: str,
) -> dict[str, Any] | None:
    candidates = index.get(identity, {}).get(label, [])
    return candidates[0] if len(candidates) == 1 else None


def _compact_source(
    source: str,
    members: list[dict[str, Any]],
    maximum_characters: int,
) -> tuple[str, list[dict[str, Any]], dict[str, list[list[int]]], int]:
    explicit = [row for row in members if int(row["end"]) > int(row["start"])]
    if not explicit:
        raise ValueError("MPQA relation contains no explicit source span")

    def windows(radius: int) -> list[tuple[int, int]]:
        candidates = sorted(
            (
                max(0, int(row["start"]) - radius),
                min(len(source), int(row["end"]) + radius),
            )
            for row in explicit
        )
        merged: list[list[int]] = []
        for start, end in candidates:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def length(spans: list[tuple[int, int]]) -> int:
        return sum(end - start for start, end in spans) + max(0, len(spans) - 1)

    minimum = windows(0)
    if length(minimum) > maximum_characters:
        raise ValueError("MPQA relation members exceed the source-length contract")
    low, high = 0, len(source)
    while low < high:
        probe = (low + high + 1) // 2
        if length(windows(probe)) <= maximum_characters:
            low = probe
        else:
            high = probe - 1
    selected = windows(low)
    chunks: list[str] = []
    window_rows: list[dict[str, Any]] = []
    offsets: dict[tuple[int, int], int] = {}
    cursor = 0
    for start, end in selected:
        if chunks:
            chunks.append("\n")
            cursor += 1
        text = source[start:end]
        offsets[(start, end)] = cursor
        window_rows.append(
            {
                "document_span": [start, end],
                "excerpt_span": [cursor, cursor + len(text)],
                "source_window_sha256": _digest(text),
            }
        )
        chunks.append(text)
        cursor += len(text)
    excerpt = "".join(chunks)
    spans_by_receipt: dict[str, list[list[int]]] = {}
    for row in members:
        receipt = str(row["source_annotation_sha256"])
        if int(row["start"]) == int(row["end"]):
            spans_by_receipt[receipt] = []
            continue
        containers = [
            span
            for span in selected
            if span[0] <= int(row["start"]) and int(row["end"]) <= span[1]
        ]
        if len(containers) != 1:
            raise ValueError("MPQA relation member lacks one excerpt window")
        container = containers[0]
        relative_start = offsets[container] + int(row["start"]) - container[0]
        relative_end = offsets[container] + int(row["end"]) - container[0]
        if excerpt[relative_start:relative_end] != source[int(row["start"]):int(row["end"])]:
            raise ValueError("MPQA relation source projection mismatch")
        spans_by_receipt[receipt] = [[relative_start, relative_end]]
    return excerpt, window_rows, spans_by_receipt, low


def _normalized_member(
    row: dict[str, Any], source: str, spans: dict[str, list[list[int]]]
) -> dict[str, Any]:
    receipt = str(row["source_annotation_sha256"])
    return {
        "annotation_id": str(row["fields"].get("id") or row["line_id"]),
        "annotation_line_id": str(row["line_id"]),
        "node_type": str(row["label"]),
        "source_text": source[int(row["start"]):int(row["end"])],
        "target_spans": spans[receipt],
        "fields": dict(row["fields"]),
        "source_annotation_sha256": receipt,
    }


def _writer_member() -> dict[str, Any]:
    receipt = _digest({"special_mpqa_source": "w", "meaning": "implicit_writer"})
    return {
        "annotation_id": "w",
        "annotation_line_id": "",
        "node_type": "implicit-writer",
        "source_text": "",
        "target_spans": [],
        "fields": {"id": "w", "implicit": "true"},
        "source_annotation_sha256": receipt,
    }


def reconstruct_mpqa_relation_chains(
    *,
    original_mpqa_root: Path,
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    output: dict[str, list[dict[str, Any]]] = {
        "private_train": [],
        "private_dev": [],
        "private_eval": [],
    }
    rejections: Counter[str] = Counter()
    observed_expressions = 0
    admitted_attitudes = 0
    admitted_targets = 0
    admitted_sources = 0
    annotation_files = sorted(original_mpqa_root.glob("*/gateman.mpqa.lre.2.0"))
    if not annotation_files:
        raise ValueError("MASC MPQA annotation root is empty")
    for annotation_path in annotation_files:
        document_name = annotation_path.parent.name
        source_path = original_mpqa_root / "texts" / document_name
        if not source_path.is_file():
            raise ValueError(f"missing original MPQA source text: {document_name}")
        source = source_path.read_text(encoding="utf-8", errors="replace")
        rows = _parse_annotations(annotation_path, source)
        index = _index_annotations(rows)
        split = (
            "private_dev"
            if document_name in private_dev_documents
            else "private_eval"
            if document_name in private_eval_documents
            else "private_train"
        )
        for expression in rows:
            if expression["label"] != "direct-subjective":
                continue
            attitude_ids = _references(str(expression["fields"].get("attitude-link") or ""))
            if not attitude_ids:
                continue
            observed_expressions += 1
            attitudes: list[dict[str, Any]] = []
            for attitude_id in attitude_ids:
                attitude = _resolve_one(index, attitude_id, "attitude")
                if attitude is None:
                    attitudes = []
                    rejections["ambiguous_or_missing_attitude"] += 1
                    break
                attitudes.append(attitude)
            if not attitudes:
                continue
            target_rows: dict[str, dict[str, Any]] = {}
            target_ids_by_attitude: dict[str, list[str]] = {}
            relation_complete = True
            for attitude in attitudes:
                attitude_id = str(attitude["fields"]["id"])
                target_ids = _references(str(attitude["fields"].get("target-link") or ""))
                if not target_ids:
                    relation_complete = False
                    rejections["missing_target_link"] += 1
                    break
                target_ids_by_attitude[attitude_id] = target_ids
                for target_id in target_ids:
                    target = _resolve_one(index, target_id, "target")
                    if target is None:
                        relation_complete = False
                        rejections["ambiguous_or_missing_target"] += 1
                        break
                    target_rows[target_id] = target
                if not relation_complete:
                    break
            if not relation_complete:
                continue
            source_ids = _references(str(expression["fields"].get("nested-source") or ""))
            if not source_ids:
                rejections["missing_source_chain"] += 1
                continue
            source_rows: dict[str, dict[str, Any]] = {}
            for source_id in source_ids:
                if source_id == "w":
                    continue
                agent = _resolve_one(index, source_id, "agent")
                if agent is None:
                    source_rows = {}
                    relation_complete = False
                    rejections["ambiguous_or_missing_source_agent"] += 1
                    break
                source_rows[source_id] = agent
            if not relation_complete:
                continue
            members = [
                expression,
                *attitudes,
                *target_rows.values(),
                *source_rows.values(),
            ]
            try:
                excerpt, windows, spans, context_radius = _compact_source(
                    source, members, maximum_characters
                )
            except ValueError as exc:
                rejections[str(exc)] += 1
                continue
            expression_node = _normalized_member(expression, source, spans)
            source_nodes = [
                _writer_member()
                if source_id == "w"
                else _normalized_member(source_rows[source_id], source, spans)
                for source_id in source_ids
            ]
            attitude_nodes = []
            edges = []
            expression_ref = expression_node["source_annotation_sha256"]
            for index_in_chain, source_node in enumerate(source_nodes):
                edges.append(
                    {
                        "edge_type": "nested_source_member",
                        "from": expression_ref,
                        "to": source_node["source_annotation_sha256"],
                        "order": index_in_chain,
                        "manual_field": "nested-source",
                    }
                )
            for attitude in attitudes:
                attitude_id = str(attitude["fields"]["id"])
                attitude_node = _normalized_member(attitude, source, spans)
                targets = [
                    _normalized_member(target_rows[target_id], source, spans)
                    for target_id in target_ids_by_attitude[attitude_id]
                ]
                attitude_nodes.append({**attitude_node, "targets": targets})
                edges.append(
                    {
                        "edge_type": "attitude_link",
                        "from": expression_ref,
                        "to": attitude_node["source_annotation_sha256"],
                        "manual_field": "attitude-link",
                    }
                )
                edges.extend(
                    {
                        "edge_type": "target_link",
                        "from": attitude_node["source_annotation_sha256"],
                        "to": target["source_annotation_sha256"],
                        "manual_field": "target-link",
                    }
                    for target in targets
                )
            relation_receipt = {
                "document_name": document_name,
                "expression": expression_node["source_annotation_sha256"],
                "source_chain": [row["source_annotation_sha256"] for row in source_nodes],
                "attitudes": [
                    {
                        "attitude": row["source_annotation_sha256"],
                        "targets": [target["source_annotation_sha256"] for target in row["targets"]],
                    }
                    for row in attitude_nodes
                ],
            }
            relation_identity = _digest(relation_receipt)
            annotation = {
                "policy": POLICY,
                "relation_contract": RELATION_CONTRACT,
                "document_id": "original-mpqa/" + document_name,
                "original_annotation_file_sha256": _file_digest(annotation_path),
                "original_source_file_sha256": _file_digest(source_path),
                "relation_concept": "masc.mpqa_relation." + relation_identity.split(":", 1)[1][:24],
                "source_text": excerpt,
                "excerpt_windows": windows,
                "source_compaction_contract": COMPACTION_CONTRACT,
                "maximum_source_characters": maximum_characters,
                "uniform_context_radius_characters": context_radius,
                "expression": expression_node,
                "source_chain": source_nodes,
                "attitudes": attitude_nodes,
                "edges": sorted(edges, key=_canonical),
                "complete_relation_alignment": True,
                "missingness": {
                    "attribution_chain": False,
                    "attitude_target_relation": False,
                    "scope": True,
                    "truth": True,
                    "causal_relation": True,
                    "temporal_relation": True,
                    "complete_sentence_semantics": True,
                },
            }
            selection_key = _digest(annotation)
            output[split].append(
                {
                    "source_id": "masc-mpqa-relation:" + selection_key.split(":", 1)[1][:24],
                    "split": split,
                    "selection_key": selection_key,
                    "annotation": annotation,
                }
            )
            admitted_sources += len(source_nodes)
            admitted_attitudes += len(attitude_nodes)
            admitted_targets += sum(len(row["targets"]) for row in attitude_nodes)
    for rows in output.values():
        rows.sort(key=lambda row: row["selection_key"])
    audit = {
        "policy": POLICY,
        "parser_implementation": "producer_regex_attribute_parser_v1",
        "observed_linked_expression_count": observed_expressions,
        "admitted_relation_count": sum(len(rows) for rows in output.values()),
        "admitted_source_member_count": admitted_sources,
        "admitted_attitude_count": admitted_attitudes,
        "admitted_target_count": admitted_targets,
        "record_count_by_split": {split: len(rows) for split, rows in output.items()},
        "rejection_reason_counts": dict(sorted(rejections.items())),
        "partial_relation_admission_count": 0,
        "inferred_relation_count": 0,
    }
    return output, audit
