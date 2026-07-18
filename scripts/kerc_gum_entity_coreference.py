#!/usr/bin/env python3
"""Reconstruct complete human GUM entity/coreference neighborhoods for KERC.

This producer reads the pinned WebAnno TSV export. It preserves complete
source-declared identity components and complete endpoint components for every
bridging edge. Oversize or malformed neighborhoods are rejected as a whole;
relations are never inferred from co-occurrence or surface similarity.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_gum_human_entity_coreference_v1"
RELATION_CONTRACT = "complete_source_declared_identity_component_or_bridge_endpoint_graph_v1"
SPLIT_CONTRACT = "explicit_document_disjoint_internal_split_from_official_train_v1"
COMPACTION_CONTRACT = "uniform_sentence_bounded_mention_window_v1"
PARSER_IMPLEMENTATION = "producer_webanno_tsv_state_machine_v1"
SPLITS = ("private_train", "private_dev", "private_eval")
LAYER_RE = re.compile(r"^(.*?)(?:\[([0-9]+)\])?$")
LINK_RE = re.compile(r"^([^[]+)\[([0-9]+)_([0-9]+)\]$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def selected_source_digest(root: Path, document_ids: list[str]) -> str:
    digest = hashlib.sha256()
    paths = [root / "LICENSE.md"]
    for document_id in sorted(document_ids):
        paths.extend(
            (
                root / "xml" / f"{document_id}.xml",
                root / "coref" / "gum" / "tsv" / f"{document_id}.tsv",
                root / "coref" / "gum" / "conll" / f"{document_id}.conll",
                root / "dep" / f"{document_id}.conllu",
            )
        )
    for path in paths:
        digest.update(
            f"{path.relative_to(root)}\0{_file_digest(path).split(':', 1)[1]}\n".encode()
        )
    return "sha256:" + digest.hexdigest()


def _layer(value: str) -> list[tuple[str, str]]:
    if value == "_":
        return []
    output = []
    for item in value.split("|"):
        match = LAYER_RE.fullmatch(item)
        if match is None or match.group(2) is None:
            raise ValueError(f"GUM layer item lacks stable identity: {item}")
        output.append((match.group(1), match.group(2)))
    return output


def _attribute(value: str, mention_id: str) -> str:
    if value == "_":
        return "_"
    matches = [label for label, identity in _layer(value) if identity == mention_id]
    if len(matches) > 1:
        raise ValueError(f"GUM attribute identity is ambiguous: {mention_id}")
    return matches[0] if matches else "_"


def _parse_document(path: Path) -> dict[str, Any]:
    mentions: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    sentence_spans: dict[str, list[int]] = {}
    max_end = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 10 or any(value for value in fields[10:]):
            raise ValueError(f"malformed GUM TSV row at {path}:{line_number}")
        fields = fields[:10]
        token_id = fields[0]
        sentence_id = token_id.split("-", 1)[0]
        offsets = fields[1].split("-", 1)
        if len(offsets) != 2 or not all(value.isdigit() for value in offsets):
            raise ValueError(f"malformed GUM token offsets at {path}:{line_number}")
        start, end = map(int, offsets)
        if start >= end or end - start != len(fields[2]):
            raise ValueError(f"GUM token extent mismatch at {path}:{line_number}")
        max_end = max(max_end, end)
        sentence_spans.setdefault(sentence_id, [start, end])
        sentence_spans[sentence_id][0] = min(sentence_spans[sentence_id][0], start)
        sentence_spans[sentence_id][1] = max(sentence_spans[sentence_id][1], end)
        token = {"token_id": token_id, "sentence_id": sentence_id, "start": start, "end": end, "text": fields[2]}
        tokens.append(token)
        entity_items = _layer(fields[3])
        for entity_type, mention_id in entity_items:
            attributes = {
                "entity_type": entity_type,
                "information_status": _attribute(fields[4], mention_id),
                "salience": _attribute(fields[5], mention_id),
                "identity": _attribute(fields[6], mention_id),
                "centering": _attribute(fields[7], mention_id),
            }
            mention = mentions.setdefault(
                mention_id,
                {"mention_id": mention_id, "attributes": attributes, "tokens": [], "source_rows": []},
            )
            if mention["attributes"] != attributes:
                raise ValueError(f"GUM mention attributes drift within span: {path}:{mention_id}")
            mention["tokens"].append(token)
            mention["source_rows"].append(_digest({"line": line_number, "raw": raw}))
        relation_types = [] if fields[8] == "_" else fields[8].split("|")
        relation_links = [] if fields[9] == "_" else fields[9].split("|")
        if len(relation_types) != len(relation_links):
            raise ValueError(f"GUM relation arity mismatch at {path}:{line_number}")
        current_ids = {identity for _label, identity in entity_items}
        for order, (relation_type, link) in enumerate(zip(relation_types, relation_links)):
            match = LINK_RE.fullmatch(link)
            if match is None:
                raise ValueError(f"malformed GUM relation link at {path}:{line_number}")
            target_id, source_id = match.group(2), match.group(3)
            if source_id not in current_ids or source_id == target_id:
                raise ValueError(f"GUM relation endpoint identity is ambiguous at {path}:{line_number}")
            relations.append(
                {
                    "relation_type": relation_type,
                    "source_mention_id": source_id,
                    "target_mention_id": target_id,
                    "target_token_id": match.group(1),
                    "edge_order": order,
                    "source_annotation_sha256": _digest({"line": line_number, "raw": raw, "order": order}),
                }
            )
    if not mentions:
        raise ValueError(f"empty GUM entity document: {path}")
    canvas = [" "] * max_end
    for token in tokens:
        observed = "".join(canvas[token["start"] : token["end"]])
        if observed.strip() and observed != token["text"]:
            raise ValueError(f"overlapping GUM token text mismatch: {path}:{token['token_id']}")
        canvas[token["start"] : token["end"]] = token["text"]
    document_text = "".join(canvas)
    for mention in mentions.values():
        ordered = sorted(mention.pop("tokens"), key=lambda row: (row["start"], row["end"]))
        spans: list[list[int]] = []
        sentence_ids: set[str] = set()
        for token in ordered:
            sentence_ids.add(token["sentence_id"])
            if spans and token["start"] <= spans[-1][1] + 1:
                spans[-1][1] = token["end"]
            else:
                spans.append([token["start"], token["end"]])
        mention["document_spans"] = spans
        mention["sentence_ids"] = sorted(sentence_ids, key=int)
        mention["source_text"] = " ... ".join(document_text[start:end] for start, end in spans)
        mention["source_annotation_sha256"] = _digest(
            {"mention_id": mention["mention_id"], "attributes": mention["attributes"], "rows": mention.pop("source_rows"), "spans": spans}
        )
    if any(edge["source_mention_id"] not in mentions or edge["target_mention_id"] not in mentions for edge in relations):
        raise ValueError(f"GUM relation references an unknown mention: {path}")
    return {
        "document_text": document_text,
        "sentence_spans": {key: value for key, value in sorted(sentence_spans.items(), key=lambda row: int(row[0]))},
        "mentions": mentions,
        "relations": relations,
    }


def _components(mentions: dict[str, Any], relations: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    parent = {identity: identity for identity in mentions}

    def find(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    for edge in relations:
        if str(edge["relation_type"]).startswith("bridge:"):
            continue
        left, right = find(edge["source_mention_id"]), find(edge["target_mention_id"])
        if left != right:
            parent[max(left, right, key=int)] = min(left, right, key=int)
    groups: dict[str, set[str]] = defaultdict(set)
    mention_to_group = {}
    for mention_id in mentions:
        root = find(mention_id)
        groups[root].add(mention_id)
        mention_to_group[mention_id] = root
    return dict(groups), mention_to_group


def _excerpt(document: dict[str, Any], mention_ids: set[str], maximum_characters: int) -> tuple[str, list[dict[str, Any]], int]:
    mentions = document["mentions"]
    sentence_spans = document["sentence_spans"]
    source_spans = [span for identity in mention_ids for span in mentions[identity]["document_spans"]]
    sentence_for_span = []
    for span in source_spans:
        candidates = [bounds for bounds in sentence_spans.values() if bounds[0] <= span[0] and span[1] <= bounds[1]]
        if len(candidates) != 1:
            raise ValueError("GUM mention does not map to one sentence")
        sentence_for_span.append(candidates[0])

    def windows(radius: int) -> list[list[int]]:
        values = sorted(
            [max(sentence[0], span[0] - radius), min(span[1] + radius, sentence[1])]
            for span, sentence in zip(source_spans, sentence_for_span)
        )
        merged: list[list[int]] = []
        for start, end in values:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    def rendered_length(values: list[list[int]]) -> int:
        return sum(end - start for start, end in values) + max(0, len(values) - 1)

    if rendered_length(windows(0)) > maximum_characters:
        raise ValueError("complete GUM component exceeds source-length contract")
    low, high = 0, max((end - start for start, end in sentence_spans.values()), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if rendered_length(windows(middle)) <= maximum_characters:
            low = middle
        else:
            high = middle - 1
    selected = windows(low)
    chunks: list[str] = []
    projected_windows = []
    cursor = 0
    for start, end in selected:
        if chunks:
            chunks.append("\n")
            cursor += 1
        text = document["document_text"][start:end]
        projected_windows.append({"document_span": [start, end], "excerpt_span": [cursor, cursor + len(text)]})
        chunks.append(text)
        cursor += len(text)
    excerpt = "".join(chunks)
    projected_mentions = []
    for mention_id in sorted(mention_ids, key=int):
        mention = mentions[mention_id]
        excerpt_spans = []
        for span in mention["document_spans"]:
            containing = [window for window in projected_windows if window["document_span"][0] <= span[0] and span[1] <= window["document_span"][1]]
            if len(containing) != 1:
                raise ValueError("GUM mention is not fully preserved by source compaction")
            window = containing[0]
            excerpt_spans.append([
                window["excerpt_span"][0] + span[0] - window["document_span"][0],
                window["excerpt_span"][0] + span[1] - window["document_span"][0],
            ])
        projected_mentions.append({**mention, "excerpt_spans": excerpt_spans})
    return excerpt, projected_mentions, low


def _project(document_id: str, genre: str, license_spdx: str, split: str, document: dict[str, Any], group_ids: list[str], anchor_bridge: dict[str, Any] | None, maximum_characters: int) -> dict[str, Any]:
    components, mention_to_group = _components(document["mentions"], document["relations"])
    mention_ids = set().union(*(components[group_id] for group_id in group_ids))
    source_text, mentions, context_radius = _excerpt(document, mention_ids, maximum_characters)
    admitted_edges = [
        edge for edge in document["relations"]
        if not str(edge["relation_type"]).startswith("bridge:")
        and edge["source_mention_id"] in mention_ids
        and edge["target_mention_id"] in mention_ids
    ]
    if anchor_bridge is not None:
        admitted_edges.append(anchor_bridge)
    admitted_edges.sort(key=lambda row: (int(row["source_mention_id"]), int(row["target_mention_id"]), row["relation_type"], row["source_annotation_sha256"]))
    kind = "bridge_relation" if anchor_bridge is not None else "identity_component"
    identity = _digest({"document_id": document_id, "kind": kind, "groups": group_ids, "bridge": anchor_bridge})
    groups = []
    for group_id in group_ids:
        component_mentions = sorted(components[group_id], key=int)
        groups.append({
            "group_id": group_id,
            "stable_identity": f"gum.entity.{document_id.lower()}.{group_id}",
            "mention_ids": component_mentions,
            "complete_component": True,
        })
    annotation = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "source_compaction_contract": COMPACTION_CONTRACT,
        "document_id": document_id,
        "official_partition": "train",
        "genre": genre,
        "license_spdx": license_spdx,
        "record_kind": kind,
        "source_text": source_text,
        "uniform_context_radius_characters": context_radius,
        "groups": groups,
        "mentions": mentions,
        "relations": admitted_edges,
        "anchor_bridge_sha256": None if anchor_bridge is None else anchor_bridge["source_annotation_sha256"],
        "complete_source_declared_components": True,
        "partial_component_admission_count": 0,
        "inferred_relation_count": 0,
        "truth_claimed": False,
        "complete_sentence_semantics_claimed": False,
    }
    return {
        "source_id": f"gum-entity:{kind}:{identity.split(':', 1)[1][:24]}",
        "source_group": f"gum-document:{document_id}",
        "split": split,
        "license_spdx": license_spdx,
        "source_text": source_text,
        "annotation": annotation,
    }


def reconstruct_gum_entity_coreference(*, source_root: Path, allowed_genre_licenses: dict[str, str], private_dev_documents: set[str], private_eval_documents: set[str], expected_selected_source_sha256: str, maximum_characters: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    overlap = private_dev_documents & private_eval_documents
    if overlap:
        raise ValueError(f"GUM entity/coreference private splits overlap: {sorted(overlap)}")
    selected_metadata = []
    official_nontrain = 0
    for xml_path in sorted((source_root / "xml").glob("GUM_*.xml")):
        metadata = ET.parse(xml_path).getroot().attrib
        if metadata.get("partition") != "train":
            official_nontrain += 1
            continue
        genre = str(metadata.get("type") or "")
        if genre in allowed_genre_licenses:
            selected_metadata.append((str(metadata["id"]), genre))
    document_ids = [row[0] for row in selected_metadata]
    observed_digest = selected_source_digest(source_root, document_ids)
    if observed_digest != expected_selected_source_sha256:
        raise ValueError("GUM entity/coreference selected source content mismatch")
    if not private_dev_documents | private_eval_documents <= set(document_ids):
        raise ValueError("GUM entity/coreference split references excluded documents")
    by_split = {split: [] for split in SPLITS}
    rejected = []
    totals = Counter()
    counts_by_split = {split: Counter() for split in SPLITS}
    relation_types = {split: Counter() for split in SPLITS}
    component_sizes = {split: Counter() for split in SPLITS}
    for document_id, genre in selected_metadata:
        split = "private_dev" if document_id in private_dev_documents else "private_eval" if document_id in private_eval_documents else "private_train"
        document = _parse_document(source_root / "coref" / "gum" / "tsv" / f"{document_id}.tsv")
        components, mention_to_group = _components(document["mentions"], document["relations"])
        counts_by_split[split]["documents"] += 1
        counts_by_split[split]["mentions"] += len(document["mentions"])
        counts_by_split[split]["components"] += len(components)
        for component in components.values():
            component_sizes[split][str(len(component))] += 1
        for edge in document["relations"]:
            relation_types[split][edge["relation_type"]] += 1
        for group_id, component in sorted(components.items(), key=lambda row: int(row[0])):
            if len(component) < 2:
                continue
            try:
                by_split[split].append(_project(document_id, genre, allowed_genre_licenses[genre], split, document, [group_id], None, maximum_characters))
            except ValueError as exc:
                rejected.append({"document_id": document_id, "record_kind": "identity_component", "group_ids": [group_id], "reason": str(exc)})
        for edge in document["relations"]:
            if not str(edge["relation_type"]).startswith("bridge:"):
                continue
            group_ids = sorted({mention_to_group[edge["source_mention_id"]], mention_to_group[edge["target_mention_id"]]}, key=int)
            if len(group_ids) != 2:
                rejected.append({"document_id": document_id, "record_kind": "bridge_relation", "group_ids": group_ids, "reason": "bridge endpoints collapse into one identity component"})
                continue
            try:
                by_split[split].append(_project(document_id, genre, allowed_genre_licenses[genre], split, document, group_ids, edge, maximum_characters))
            except ValueError as exc:
                rejected.append({"document_id": document_id, "record_kind": "bridge_relation", "group_ids": group_ids, "reason": str(exc)})
    for split, rows in by_split.items():
        rows.sort(key=lambda row: row["source_id"])
        counts_by_split[split]["records"] = len(rows)
        counts_by_split[split]["identity_records"] = sum(row["annotation"]["record_kind"] == "identity_component" for row in rows)
        counts_by_split[split]["bridge_records"] = sum(row["annotation"]["record_kind"] == "bridge_relation" for row in rows)
        totals.update(counts_by_split[split])
    audit = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "split_contract": SPLIT_CONTRACT,
        "source_compaction_contract": COMPACTION_CONTRACT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "selected_source_sha256": observed_digest,
        "selected_document_count": len(selected_metadata),
        "counts_by_split": {split: dict(values) for split, values in counts_by_split.items()},
        "relation_type_count_by_split": {split: dict(values) for split, values in relation_types.items()},
        "component_size_distribution_by_split": {split: dict(values) for split, values in component_sizes.items()},
        "rejected_record_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected_records": rejected,
        "official_nontrain_document_admission_count": 0,
        "official_nontrain_document_seen_count": official_nontrain,
        "partial_component_admission_count": 0,
        "inferred_relation_count": 0,
    }
    return by_split, audit
