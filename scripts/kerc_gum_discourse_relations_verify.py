#!/usr/bin/env python3
"""Independently reconstruct GUM eRST discourse graphs for KERC admission.

This verifier deliberately shares no parsing or graph-construction code with
the producer.  It uses Expat for XML metadata, Python's CSV state machine for
RSD, and a compiled full-match parser for secondary edges.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.parsers.expat
from collections import Counter
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_gum_human_erst_discourse_v1"
RELATION_CONTRACT = "complete_source_declared_primary_plus_secondary_erst_edges_v1"
SPLIT_CONTRACT = "explicit_document_disjoint_internal_split_from_official_train_v1"
PROJECTION_CONTRACT = "ordered_complete_edu_projection_v1"
PARSER_IMPLEMENTATION = "verifier_expat_csv_state_machine_v1"
SPLITS = ("private_train", "private_dev", "private_eval")
RELATION_RE = re.compile(r"[a-z][a-z0-9-]*(?:_[rm])?")
SECONDARY_RE = re.compile(
    r"(?P<parent>[0-9]+):(?P<relation>[a-z][a-z0-9-]*):"
    r"(?P<left>[0-9]+):(?P<right>[0-9]+):(?P<signals>.*)"
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _hash_object(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(payload)
    return "sha256:" + state.hexdigest()


def _source_set_hash(root: Path, document_ids: list[str]) -> str:
    state = hashlib.sha256()
    files = [root / "LICENSE.md"]
    for identity in sorted(document_ids):
        files.append(root / "xml" / f"{identity}.xml")
        files.append(root / "rst" / "dependencies" / f"{identity}.rsd")
    for path in files:
        item_hash = _hash_file(path).partition(":")[2]
        state.update(f"{path.relative_to(root)}\0{item_hash}\n".encode("utf-8"))
    return "sha256:" + state.hexdigest()


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    first_element_seen = False
    parser = xml.parsers.expat.ParserCreate()

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal first_element_seen, metadata
        if first_element_seen:
            return
        first_element_seen = True
        if name != "text":
            raise ValueError(f"GUM XML root is not text: {path.name}")
        metadata = {str(key): str(value) for key, value in sorted(attrs.items())}

    parser.StartElementHandler = start
    with path.open("rb") as stream:
        parser.ParseFile(stream)
    if not {"id", "type"} <= set(metadata):
        raise ValueError(f"GUM XML metadata incomplete: {path.name}")
    return metadata


def _parse_secondaries(
    field: str, *, source_name: str, line_number: int
) -> list[dict[str, Any]]:
    if field == "_":
        return []
    rows = []
    cursor = 0
    for raw in field.split("|"):
        match = SECONDARY_RE.fullmatch(raw)
        if match is None:
            raise ValueError(
                f"malformed GUM secondary edge at {source_name}:{line_number}"
            )
        relation = match.group("relation")
        if RELATION_RE.fullmatch(relation) is None:
            raise ValueError(
                f"invalid GUM secondary relation at {source_name}:{line_number}"
            )
        rows.append(
            {
                "edge_kind": "secondary",
                "edge_order": cursor,
                "parent_edu_id": int(match.group("parent")),
                "relation": relation,
                "raw_depth_fields": [
                    int(match.group("left")),
                    int(match.group("right")),
                ],
                "raw_signal_payload": match.group("signals"),
                "source_annotation_sha256": _hash_object(
                    {
                        "path": source_name,
                        "line": line_number,
                        "raw": raw,
                    }
                ),
            }
        )
        cursor += 1
    return rows


def _read_rsd(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(
            stream, delimiter="\t", quoting=csv.QUOTE_NONE, strict=True
        )
        for line_number, fields in enumerate(reader, 1):
            if not fields:
                continue
            if len(fields) != 10:
                raise ValueError(f"malformed GUM RSD row at {path}:{line_number}")
            try:
                edu_id = int(fields[0])
                depth = int(fields[2])
                parent = int(fields[6])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric GUM RSD field at {path}:{line_number}"
                ) from exc
            if edu_id in rows or edu_id != len(rows) + 1 or not fields[1].strip():
                raise ValueError(f"invalid GUM EDU at {path}:{line_number}")
            relation = fields[7]
            if (parent == 0) != (relation == "ROOT"):
                raise ValueError(f"GUM root relation mismatch at {path}:{line_number}")
            if relation != "ROOT" and RELATION_RE.fullmatch(relation) is None:
                raise ValueError(f"invalid GUM relation at {path}:{line_number}")
            features: list[dict[str, str]] = []
            scanner = fields[5]
            position = 0
            while position < len(scanner):
                boundary = scanner.find("|", position)
                if boundary < 0:
                    boundary = len(scanner)
                item = scanner[position:boundary]
                equals = item.find("=")
                if not item:
                    raise ValueError(f"malformed GUM feature at {path}:{line_number}")
                if equals < 0:
                    key, value = item, "true"
                else:
                    key, value = item[:equals], item[equals + 1 :]
                if not key:
                    raise ValueError(f"malformed GUM feature at {path}:{line_number}")
                features.append({"key": key, "value": value})
                position = boundary + 1
            raw = "\t".join(fields)
            rows[edu_id] = {
                "edu_id": edu_id,
                "text": fields[1],
                "tree_depth": depth,
                "features": features,
                "parent_edu_id": parent,
                "relation": relation,
                "secondary_edges": _parse_secondaries(
                    fields[8], source_name=path.name, line_number=line_number
                ),
                "signals": [] if fields[9] == "_" else fields[9].split(";"),
                "source_row_sha256": _hash_object(
                    {"path": path.name, "line_number": line_number, "raw": raw}
                ),
            }
    if not rows:
        raise ValueError(f"empty GUM RSD document: {path}")
    root_count = 0
    for row in rows.values():
        if row["relation"] == "ROOT":
            root_count += 1
        parent = int(row["parent_edu_id"])
        if parent and parent not in rows:
            raise ValueError(f"unknown GUM primary parent in {path.name}")
        if any(int(edge["parent_edu_id"]) not in rows for edge in row["secondary_edges"]):
            raise ValueError(f"unknown GUM secondary parent in {path.name}")
    if root_count != 1:
        raise ValueError(f"GUM document must have one discourse root: {path.name}")
    return rows


def _reconstruct_record(
    document_id: str,
    genre: str,
    source_url: str,
    license_spdx: str,
    split: str,
    rows: dict[int, dict[str, Any]],
    anchor: dict[str, Any],
    xml_sha256: str,
    rsd_sha256: str,
) -> dict[str, Any]:
    unit_ids = {int(anchor["edu_id"]), int(anchor["parent_edu_id"])}
    unit_ids.update(int(edge["parent_edu_id"]) for edge in anchor["secondary_edges"])
    source_parts: list[str] = []
    units: list[dict[str, Any]] = []
    offset = 0
    for edu_id in sorted(unit_ids):
        if source_parts:
            source_parts.append("\n")
            offset += 1
        text = str(rows[edu_id]["text"])
        begin = offset
        source_parts.append(text)
        offset += len(text)
        units.append(
            {
                "edu_id": edu_id,
                "text": text,
                "excerpt_span": [begin, offset],
                "tree_depth": int(rows[edu_id]["tree_depth"]),
                "features": rows[edu_id]["features"],
                "source_row_sha256": rows[edu_id]["source_row_sha256"],
            }
        )
    source_text = "".join(source_parts)
    edges = [
        {
            "edge_kind": "primary",
            "edge_order": 0,
            "child_edu_id": int(anchor["edu_id"]),
            "parent_edu_id": int(anchor["parent_edu_id"]),
            "relation": str(anchor["relation"]),
            "raw_depth_fields": [],
            "raw_signal_payload": ";".join(anchor["signals"]),
            "source_annotation_sha256": anchor["source_row_sha256"],
        }
    ]
    for secondary in anchor["secondary_edges"]:
        edges.append({**secondary, "child_edu_id": int(anchor["edu_id"])})
    relation = str(anchor["relation"])
    if relation.endswith("_m"):
        nuclearity = "multinuclear"
    elif relation.endswith("_r"):
        nuclearity = "satellite_nucleus"
    else:
        nuclearity = "secondary_unspecified"
    return {
        "source_id": f"gum-erst:{document_id}:{int(anchor['edu_id'])}",
        "source_group": f"gum-document:{document_id}",
        "split": split,
        "license_spdx": license_spdx,
        "source_text": source_text,
        "annotation": {
            "policy": POLICY,
            "relation_contract": RELATION_CONTRACT,
            "projection_contract": PROJECTION_CONTRACT,
            "document_id": document_id,
            "official_partition": "train",
            "genre": genre,
            "source_url": source_url,
            "license_spdx": license_spdx,
            "anchor_edu_id": int(anchor["edu_id"]),
            "primary_relation": relation,
            "primary_relation_base": relation.removesuffix("_m").removesuffix("_r"),
            "primary_nuclearity": nuclearity,
            "source_text": source_text,
            "units": units,
            "edges": edges,
            "xml_sha256": xml_sha256,
            "rsd_sha256": rsd_sha256,
            "complete_source_declared_edges": True,
            "inferred_relation_count": 0,
            "truth_claimed": False,
            "complete_sentence_semantics_claimed": False,
        },
    }


def independently_reconstruct_gum_discourse_relations(
    *,
    source_root: Path,
    allowed_genre_licenses: dict[str, str],
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    expected_selected_source_sha256: str,
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if private_dev_documents.intersection(private_eval_documents):
        raise ValueError("GUM private document splits overlap")
    output = {split: [] for split in SPLITS}
    relation_counts = {split: Counter() for split in SPLITS}
    genre_counts = {split: Counter() for split in SPLITS}
    secondary_counts = Counter()
    document_counts = Counter()
    selected: list[str] = []
    for xml_path in sorted((source_root / "xml").glob("*.xml")):
        metadata = _read_metadata(xml_path)
        if metadata.get("partition") != "train" or metadata["type"] not in allowed_genre_licenses:
            continue
        if not metadata.get("sourceURL"):
            raise ValueError(f"GUM admitted document lacks source URL: {xml_path.name}")
        document_id = metadata["id"]
        if document_id != xml_path.stem:
            raise ValueError(f"GUM document identity mismatch: {xml_path.name}")
        if document_id in private_dev_documents:
            split = "private_dev"
        elif document_id in private_eval_documents:
            split = "private_eval"
        else:
            split = "private_train"
        rsd_path = source_root / "rst" / "dependencies" / f"{document_id}.rsd"
        if not rsd_path.is_file():
            raise ValueError(f"missing GUM RSD source: {document_id}")
        rows = _read_rsd(rsd_path)
        xml_sha256, rsd_sha256 = _hash_file(xml_path), _hash_file(rsd_path)
        selected.append(document_id)
        document_counts[split] += 1
        genre_counts[split][metadata["type"]] += 1
        for anchor in rows.values():
            if anchor["relation"] == "ROOT":
                continue
            record = _reconstruct_record(
                document_id,
                metadata["type"],
                metadata["sourceURL"],
                allowed_genre_licenses[metadata["type"]],
                split,
                rows,
                anchor,
                xml_sha256,
                rsd_sha256,
            )
            if len(record["source_text"]) > maximum_characters:
                raise ValueError(f"GUM complete EDU projection exceeds cap: {record['source_id']}")
            output[split].append(record)
            relation_counts[split][str(anchor["relation"])] += 1
            secondary_counts[split] += len(anchor["secondary_edges"])
    missing = private_dev_documents.union(private_eval_documents).difference(selected)
    if missing:
        raise ValueError("GUM configured heldout document is absent from admitted source")
    source_hash = _source_set_hash(source_root, selected)
    if source_hash != expected_selected_source_sha256:
        raise ValueError("GUM selected source content mismatch")
    for split in SPLITS:
        output[split].sort(key=lambda row: row["source_id"])
    audit = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "split_contract": SPLIT_CONTRACT,
        "projection_contract": PROJECTION_CONTRACT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "selected_source_sha256": source_hash,
        "selected_document_count": len(selected),
        "document_count_by_split": dict(document_counts),
        "record_count_by_split": {split: len(output[split]) for split in SPLITS},
        "genre_count_by_split": {
            split: dict(sorted(genre_counts[split].items())) for split in SPLITS
        },
        "relation_count_by_split": {
            split: dict(sorted(relation_counts[split].items())) for split in SPLITS
        },
        "secondary_edge_count_by_split": dict(secondary_counts),
        "relation_type_count_by_split": {
            split: len(relation_counts[split]) for split in SPLITS
        },
        "minimum_relation_count_by_split": {
            split: min(relation_counts[split].values()) for split in SPLITS
        },
        "official_nontrain_document_admission_count": 0,
        "partial_relation_admission_count": 0,
        "inferred_relation_count": 0,
        "complete_sentence_semantics_claimed": False,
        "truth_claimed": False,
    }
    return output, audit
