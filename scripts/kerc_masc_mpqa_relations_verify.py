#!/usr/bin/env python3
"""Independent raw-source replay for MASC MPQA relation admission.

This verifier deliberately does not import the producer. It parses quoted GATE
attributes with a state machine, resolves typed ID references independently,
and reconstructs source projections before comparing canonical rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_masc_manual_mpqa_relation_v1"
RELATION_CONTRACT = "complete_manual_mpqa_expression_attitude_target_source_chain_v1"
COMPACTION_CONTRACT = "uniform_radius_relation_member_source_windows_v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _split_refs(raw: str) -> list[str]:
    output: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            output.append(item)
    return output


def _quoted_fields(raw: str, *, location: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = 0
    while cursor < len(raw):
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor >= len(raw):
            break
        key_start = cursor
        while cursor < len(raw) and (raw[cursor].isalnum() or raw[cursor] in "_-"):
            cursor += 1
        key = raw[key_start:cursor]
        if not key or cursor >= len(raw) or raw[cursor] != "=":
            raise ValueError(f"malformed MPQA attribute key at {location}")
        cursor += 1
        if cursor >= len(raw) or raw[cursor] != '"':
            raise ValueError(f"unquoted MPQA attribute at {location}:{key}")
        cursor += 1
        value_start = cursor
        while cursor < len(raw) and raw[cursor] != '"':
            cursor += 1
        if cursor >= len(raw):
            raise ValueError(f"unterminated MPQA attribute at {location}:{key}")
        if key in fields:
            raise ValueError(f"duplicate MPQA attribute at {location}:{key}")
        fields[key] = raw[value_start:cursor]
        cursor += 1
    return fields


def _independent_rows(path: Path, source: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) < 4 or not parts[3].strip().startswith("GATE_"):
            continue
        span_parts = parts[1].strip().split(",")
        if len(span_parts) != 2 or not all(item.isdigit() for item in span_parts):
            raise ValueError(f"invalid independent MPQA span at {path}:{line_number}")
        start, end = (int(item) for item in span_parts)
        if start < 0 or end < start or end > len(source):
            raise ValueError(f"out-of-bounds independent MPQA span at {path}:{line_number}")
        fields = _quoted_fields(
            parts[4] if len(parts) > 4 else "", location=f"{path}:{line_number}"
        )
        receipt = {
            "line_id": parts[0].strip(),
            "line_number": line_number,
            "label": parts[3].strip()[len("GATE_") :],
            "start": start,
            "end": end,
            "fields": dict(sorted(fields.items())),
        }
        output.append({**receipt, "source_annotation_sha256": _hash(receipt)})
    return output


def _typed_index(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        identity = str(row["fields"].get("id") or "").strip()
        if identity:
            output[identity][str(row["label"])].append(row)
    return output


def _unique(
    index: dict[str, dict[str, list[dict[str, Any]]]], identity: str, kind: str
) -> dict[str, Any] | None:
    observed = index.get(identity, {}).get(kind, [])
    if len(observed) != 1:
        return None
    return observed[0]


def _source_projection(
    source: str, members: list[dict[str, Any]], maximum_characters: int
) -> tuple[str, list[dict[str, Any]], dict[str, list[list[int]]], int]:
    explicit = [member for member in members if member["end"] > member["start"]]
    if not explicit:
        raise ValueError("MPQA relation contains no explicit source span")

    def expanded(radius: int) -> list[tuple[int, int]]:
        intervals = sorted(
            (
                max(0, int(member["start"]) - radius),
                min(len(source), int(member["end"]) + radius),
            )
            for member in explicit
        )
        reduced: list[tuple[int, int]] = []
        for interval in intervals:
            if reduced and interval[0] <= reduced[-1][1]:
                reduced[-1] = (reduced[-1][0], max(reduced[-1][1], interval[1]))
            else:
                reduced.append(interval)
        return reduced

    def rendered_size(intervals: list[tuple[int, int]]) -> int:
        return sum(end - start for start, end in intervals) + max(len(intervals) - 1, 0)

    if rendered_size(expanded(0)) > maximum_characters:
        raise ValueError("MPQA relation members exceed the source-length contract")
    accepted_radius = 0
    rejected_radius = len(source) + 1
    while accepted_radius + 1 < rejected_radius:
        candidate = (accepted_radius + rejected_radius) // 2
        if rendered_size(expanded(candidate)) <= maximum_characters:
            accepted_radius = candidate
        else:
            rejected_radius = candidate
    intervals = expanded(accepted_radius)
    segments: list[str] = []
    receipts: list[dict[str, Any]] = []
    excerpt_start_by_interval: dict[tuple[int, int], int] = {}
    offset = 0
    for interval in intervals:
        if segments:
            segments.append("\n")
            offset += 1
        segment = source[interval[0] : interval[1]]
        excerpt_start_by_interval[interval] = offset
        receipts.append(
            {
                "document_span": [interval[0], interval[1]],
                "excerpt_span": [offset, offset + len(segment)],
                "source_window_sha256": _hash(segment),
            }
        )
        segments.append(segment)
        offset += len(segment)
    excerpt = "".join(segments)
    projected: dict[str, list[list[int]]] = {}
    for member in members:
        identity = str(member["source_annotation_sha256"])
        if member["start"] == member["end"]:
            projected[identity] = []
            continue
        containers = [
            interval
            for interval in intervals
            if interval[0] <= member["start"] and member["end"] <= interval[1]
        ]
        if len(containers) != 1:
            raise ValueError("MPQA relation member lacks one excerpt window")
        interval = containers[0]
        begin = excerpt_start_by_interval[interval] + member["start"] - interval[0]
        finish = excerpt_start_by_interval[interval] + member["end"] - interval[0]
        if excerpt[begin:finish] != source[member["start"] : member["end"]]:
            raise ValueError("MPQA relation source projection mismatch")
        projected[identity] = [[begin, finish]]
    return excerpt, receipts, projected, accepted_radius


def _member(
    raw: dict[str, Any], source: str, projected: dict[str, list[list[int]]]
) -> dict[str, Any]:
    identity = str(raw["source_annotation_sha256"])
    return {
        "annotation_id": str(raw["fields"].get("id") or raw["line_id"]),
        "annotation_line_id": str(raw["line_id"]),
        "node_type": str(raw["label"]),
        "source_text": source[raw["start"] : raw["end"]],
        "target_spans": projected[identity],
        "fields": dict(raw["fields"]),
        "source_annotation_sha256": identity,
    }


def _implicit_writer() -> dict[str, Any]:
    identity = _hash({"special_mpqa_source": "w", "meaning": "implicit_writer"})
    return {
        "annotation_id": "w",
        "annotation_line_id": "",
        "node_type": "implicit-writer",
        "source_text": "",
        "target_spans": [],
        "fields": {"id": "w", "implicit": "true"},
        "source_annotation_sha256": identity,
    }


def independently_reconstruct_mpqa_relation_chains(
    *,
    original_mpqa_root: Path,
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    output = {"private_train": [], "private_dev": [], "private_eval": []}
    rejected: Counter[str] = Counter()
    observed = 0
    source_member_count = 0
    attitude_count = 0
    target_count = 0
    paths = sorted(original_mpqa_root.glob("*/gateman.mpqa.lre.2.0"))
    if not paths:
        raise ValueError("independent MASC MPQA source is empty")
    for annotation_path in paths:
        document_name = annotation_path.parent.name
        source_path = original_mpqa_root / "texts" / document_name
        if not source_path.is_file():
            raise ValueError(f"independent MPQA source text missing: {document_name}")
        source = source_path.read_text(encoding="utf-8", errors="replace")
        rows = _independent_rows(annotation_path, source)
        typed = _typed_index(rows)
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
            attitude_ids = _split_refs(str(expression["fields"].get("attitude-link") or ""))
            if not attitude_ids:
                continue
            observed += 1
            attitudes = [_unique(typed, identity, "attitude") for identity in attitude_ids]
            if any(row is None for row in attitudes):
                rejected["ambiguous_or_missing_attitude"] += 1
                continue
            resolved_attitudes = [row for row in attitudes if row is not None]
            targets: dict[str, dict[str, Any]] = {}
            target_ids_by_attitude: dict[str, list[str]] = {}
            valid = True
            for attitude in resolved_attitudes:
                attitude_id = str(attitude["fields"]["id"])
                target_ids = _split_refs(str(attitude["fields"].get("target-link") or ""))
                if not target_ids:
                    rejected["missing_target_link"] += 1
                    valid = False
                    break
                target_ids_by_attitude[attitude_id] = target_ids
                for target_id in target_ids:
                    target = _unique(typed, target_id, "target")
                    if target is None:
                        rejected["ambiguous_or_missing_target"] += 1
                        valid = False
                        break
                    targets[target_id] = target
                if not valid:
                    break
            if not valid:
                continue
            source_ids = _split_refs(str(expression["fields"].get("nested-source") or ""))
            if not source_ids:
                rejected["missing_source_chain"] += 1
                continue
            agents: dict[str, dict[str, Any]] = {}
            for source_id in source_ids:
                if source_id == "w":
                    continue
                agent = _unique(typed, source_id, "agent")
                if agent is None:
                    rejected["ambiguous_or_missing_source_agent"] += 1
                    valid = False
                    break
                agents[source_id] = agent
            if not valid:
                continue
            members = [expression, *resolved_attitudes, *targets.values(), *agents.values()]
            try:
                excerpt, windows, projected, radius = _source_projection(
                    source, members, maximum_characters
                )
            except ValueError as exc:
                rejected[str(exc)] += 1
                continue
            expression_node = _member(expression, source, projected)
            source_nodes = [
                _implicit_writer()
                if source_id == "w"
                else _member(agents[source_id], source, projected)
                for source_id in source_ids
            ]
            expression_ref = expression_node["source_annotation_sha256"]
            edges = [
                {
                    "edge_type": "nested_source_member",
                    "from": expression_ref,
                    "to": source_node["source_annotation_sha256"],
                    "order": order,
                    "manual_field": "nested-source",
                }
                for order, source_node in enumerate(source_nodes)
            ]
            attitude_nodes = []
            for attitude in resolved_attitudes:
                attitude_id = str(attitude["fields"]["id"])
                attitude_node = _member(attitude, source, projected)
                target_nodes = [
                    _member(targets[target_id], source, projected)
                    for target_id in target_ids_by_attitude[attitude_id]
                ]
                attitude_nodes.append({**attitude_node, "targets": target_nodes})
                edges.append(
                    {
                        "edge_type": "attitude_link",
                        "from": expression_ref,
                        "to": attitude_node["source_annotation_sha256"],
                        "manual_field": "attitude-link",
                    }
                )
                for target_node in target_nodes:
                    edges.append(
                        {
                            "edge_type": "target_link",
                            "from": attitude_node["source_annotation_sha256"],
                            "to": target_node["source_annotation_sha256"],
                            "manual_field": "target-link",
                        }
                    )
            relation_receipt = {
                "document_name": document_name,
                "expression": expression_node["source_annotation_sha256"],
                "source_chain": [node["source_annotation_sha256"] for node in source_nodes],
                "attitudes": [
                    {
                        "attitude": node["source_annotation_sha256"],
                        "targets": [target["source_annotation_sha256"] for target in node["targets"]],
                    }
                    for node in attitude_nodes
                ],
            }
            relation_identity = _hash(relation_receipt)
            annotation = {
                "policy": POLICY,
                "relation_contract": RELATION_CONTRACT,
                "document_id": "original-mpqa/" + document_name,
                "original_annotation_file_sha256": _hash_file(annotation_path),
                "original_source_file_sha256": _hash_file(source_path),
                "relation_concept": "masc.mpqa_relation." + relation_identity.split(":", 1)[1][:24],
                "source_text": excerpt,
                "excerpt_windows": windows,
                "source_compaction_contract": COMPACTION_CONTRACT,
                "maximum_source_characters": maximum_characters,
                "uniform_context_radius_characters": radius,
                "expression": expression_node,
                "source_chain": source_nodes,
                "attitudes": attitude_nodes,
                "edges": sorted(edges, key=_json),
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
            selection_key = _hash(annotation)
            output[split].append(
                {
                    "source_id": "masc-mpqa-relation:" + selection_key.split(":", 1)[1][:24],
                    "split": split,
                    "selection_key": selection_key,
                    "annotation": annotation,
                }
            )
            source_member_count += len(source_nodes)
            attitude_count += len(attitude_nodes)
            target_count += sum(len(node["targets"]) for node in attitude_nodes)
    for rows in output.values():
        rows.sort(key=lambda row: row["selection_key"])
    audit = {
        "policy": POLICY,
        "parser_implementation": "verifier_state_machine_attribute_parser_v1",
        "observed_linked_expression_count": observed,
        "admitted_relation_count": sum(len(rows) for rows in output.values()),
        "admitted_source_member_count": source_member_count,
        "admitted_attitude_count": attitude_count,
        "admitted_target_count": target_count,
        "record_count_by_split": {split: len(rows) for split, rows in output.items()},
        "rejection_reason_counts": dict(sorted(rejected.items())),
        "partial_relation_admission_count": 0,
        "inferred_relation_count": 0,
    }
    return output, audit
