#!/usr/bin/env python3
"""Reconstruct licensed human eRST discourse graphs for KERC.

This producer reads only the pinned GUM train partition.  It admits a
license-filtered document subset, projects each primary discourse edge with
all source-declared secondary edges, and keeps document-disjoint private
train/development/evaluation partitions.  It never infers a relation from
surface text.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_gum_human_erst_discourse_v1"
RELATION_CONTRACT = "complete_source_declared_primary_plus_secondary_erst_edges_v1"
SPLIT_CONTRACT = "explicit_document_disjoint_internal_split_from_official_train_v1"
PROJECTION_CONTRACT = "ordered_complete_edu_projection_v1"
PARSER_IMPLEMENTATION = "producer_elementtree_tab_parser_v1"
SPLITS = ("private_train", "private_dev", "private_eval")
RELATION_RE = re.compile(r"^[a-z][a-z0-9-]*(?:_[rm])?$")


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


def _selected_source_digest(root: Path, document_ids: list[str]) -> str:
    digest = hashlib.sha256()
    paths = [root / "LICENSE.md"]
    for document_id in sorted(document_ids):
        paths.extend(
            (
                root / "xml" / f"{document_id}.xml",
                root / "rst" / "dependencies" / f"{document_id}.rsd",
            )
        )
    for path in paths:
        relative = str(path.relative_to(root))
        digest.update(f"{relative}\0{_file_digest(path).split(':', 1)[1]}\n".encode())
    return "sha256:" + digest.hexdigest()


def _metadata(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    if root.tag != "text":
        raise ValueError(f"GUM XML root is not text: {path.name}")
    required = {"id", "type"}
    if not required <= set(root.attrib):
        raise ValueError(f"GUM XML metadata incomplete: {path.name}")
    return {key: str(value) for key, value in sorted(root.attrib.items())}


def _secondary_edges(value: str, *, path: Path, line_number: int) -> list[dict[str, Any]]:
    if value == "_":
        return []
    output = []
    for index, raw in enumerate(value.split("|")):
        fields = raw.split(":", 4)
        if len(fields) != 5 or not fields[0].isdigit() or not RELATION_RE.fullmatch(fields[1]):
            raise ValueError(f"malformed GUM secondary edge at {path}:{line_number}")
        if not fields[2].isdigit() or not fields[3].isdigit():
            raise ValueError(f"malformed GUM secondary edge depth at {path}:{line_number}")
        output.append(
            {
                "edge_kind": "secondary",
                "edge_order": index,
                "parent_edu_id": int(fields[0]),
                "relation": fields[1],
                "raw_depth_fields": [int(fields[2]), int(fields[3])],
                "raw_signal_payload": fields[4],
                "source_annotation_sha256": _digest(
                    {"path": path.name, "line": line_number, "raw": raw}
                ),
            }
        )
    return output


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        fields = raw.split("\t")
        if len(fields) != 10 or not fields[0].isdigit() or not fields[2].isdigit():
            raise ValueError(f"malformed GUM RSD row at {path}:{line_number}")
        edu_id = int(fields[0])
        if edu_id in output or edu_id != len(output) + 1:
            raise ValueError(f"non-sequential GUM EDU identity at {path}:{line_number}")
        if not fields[1].strip() or not fields[6].isdigit():
            raise ValueError(f"incomplete GUM RSD row at {path}:{line_number}")
        parent = int(fields[6])
        relation = fields[7]
        if (parent == 0) != (relation == "ROOT"):
            raise ValueError(f"GUM root relation mismatch at {path}:{line_number}")
        if relation != "ROOT" and not RELATION_RE.fullmatch(relation):
            raise ValueError(f"invalid GUM relation at {path}:{line_number}")
        features: list[dict[str, str]] = []
        for item in fields[5].split("|"):
            key, value = item.split("=", 1) if "=" in item else (item, "true")
            if not key:
                raise ValueError(f"malformed GUM feature at {path}:{line_number}")
            features.append({"key": key, "value": value})
        row = {
            "edu_id": edu_id,
            "text": fields[1],
            "tree_depth": int(fields[2]),
            "features": features,
            "parent_edu_id": parent,
            "relation": relation,
            "secondary_edges": _secondary_edges(
                fields[8], path=path, line_number=line_number
            ),
            "signals": [] if fields[9] == "_" else fields[9].split(";"),
            "source_row_sha256": _digest(
                {"path": path.name, "line_number": line_number, "raw": raw}
            ),
        }
        output[edu_id] = row
    if not output:
        raise ValueError(f"empty GUM RSD document: {path}")
    for row in output.values():
        parent = int(row["parent_edu_id"])
        if parent and parent not in output:
            raise ValueError(f"unknown GUM primary parent in {path.name}")
        for edge in row["secondary_edges"]:
            if int(edge["parent_edu_id"]) not in output:
                raise ValueError(f"unknown GUM secondary parent in {path.name}")
    roots = [row for row in output.values() if row["relation"] == "ROOT"]
    if len(roots) != 1:
        raise ValueError(f"GUM document must have one discourse root: {path.name}")
    return output


def _project(
    *,
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
    involved = {
        int(anchor["edu_id"]),
        int(anchor["parent_edu_id"]),
        *(int(edge["parent_edu_id"]) for edge in anchor["secondary_edges"]),
    }
    chunks: list[str] = []
    units = []
    cursor = 0
    for edu_id in sorted(involved):
        if chunks:
            chunks.append("\n")
            cursor += 1
        text = str(rows[edu_id]["text"])
        start = cursor
        chunks.append(text)
        cursor += len(text)
        units.append(
            {
                "edu_id": edu_id,
                "text": text,
                "excerpt_span": [start, cursor],
                "tree_depth": int(rows[edu_id]["tree_depth"]),
                "features": rows[edu_id]["features"],
                "source_row_sha256": rows[edu_id]["source_row_sha256"],
            }
        )
    source_text = "".join(chunks)
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
    edges.extend(
        {
            **edge,
            "child_edu_id": int(anchor["edu_id"]),
        }
        for edge in anchor["secondary_edges"]
    )
    relation = str(anchor["relation"])
    nuclearity = (
        "multinuclear"
        if relation.endswith("_m")
        else "satellite_nucleus"
        if relation.endswith("_r")
        else "secondary_unspecified"
    )
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


def reconstruct_gum_discourse_relations(
    *,
    source_root: Path,
    allowed_genre_licenses: dict[str, str],
    private_dev_documents: set[str],
    private_eval_documents: set[str],
    expected_selected_source_sha256: str,
    maximum_characters: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if private_dev_documents & private_eval_documents:
        raise ValueError("GUM private document splits overlap")
    output = {split: [] for split in SPLITS}
    relation_counts = {split: Counter() for split in SPLITS}
    genre_counts = {split: Counter() for split in SPLITS}
    secondary_counts = Counter()
    document_counts = Counter()
    selected_documents: list[str] = []
    for xml_path in sorted((source_root / "xml").glob("*.xml")):
        metadata = _metadata(xml_path)
        if metadata.get("partition") != "train":
            continue
        genre = metadata["type"]
        if genre not in allowed_genre_licenses:
            continue
        if not metadata.get("sourceURL"):
            raise ValueError(f"GUM admitted document lacks source URL: {xml_path.name}")
        document_id = metadata["id"]
        if document_id != xml_path.stem:
            raise ValueError(f"GUM document identity mismatch: {xml_path.name}")
        split = (
            "private_dev"
            if document_id in private_dev_documents
            else "private_eval"
            if document_id in private_eval_documents
            else "private_train"
        )
        rsd_path = source_root / "rst" / "dependencies" / f"{document_id}.rsd"
        if not rsd_path.is_file():
            raise ValueError(f"missing GUM RSD source: {document_id}")
        rows = _rows(rsd_path)
        xml_sha256, rsd_sha256 = _file_digest(xml_path), _file_digest(rsd_path)
        selected_documents.append(document_id)
        document_counts[split] += 1
        genre_counts[split][genre] += 1
        for anchor in rows.values():
            if anchor["relation"] == "ROOT":
                continue
            record = _project(
                document_id=document_id,
                genre=genre,
                source_url=metadata["sourceURL"],
                license_spdx=allowed_genre_licenses[genre],
                split=split,
                rows=rows,
                anchor=anchor,
                xml_sha256=xml_sha256,
                rsd_sha256=rsd_sha256,
            )
            if len(record["source_text"]) > maximum_characters:
                raise ValueError(f"GUM complete EDU projection exceeds cap: {record['source_id']}")
            output[split].append(record)
            relation_counts[split][str(anchor["relation"])] += 1
            secondary_counts[split] += len(anchor["secondary_edges"])
    if (private_dev_documents | private_eval_documents) - set(selected_documents):
        raise ValueError("GUM configured heldout document is absent from admitted source")
    observed_source_sha256 = _selected_source_digest(source_root, selected_documents)
    if observed_source_sha256 != expected_selected_source_sha256:
        raise ValueError("GUM selected source content mismatch")
    for split in SPLITS:
        output[split].sort(key=lambda row: row["source_id"])
    audit = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "split_contract": SPLIT_CONTRACT,
        "projection_contract": PROJECTION_CONTRACT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "selected_source_sha256": observed_source_sha256,
        "selected_document_count": len(selected_documents),
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
