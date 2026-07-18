#!/usr/bin/env python3
"""Independent GUM entity/coreference replay for KERC admission.

This verifier uses CSV field decoding, Expat metadata parsing, an independent
union-find implementation, and a separate CoNLL-U topology audit. It does not
import or execute the producer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.parsers.expat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY = "project_theseus_kerc_gum_human_entity_coreference_v1"
RELATION_CONTRACT = "complete_source_declared_identity_component_or_bridge_endpoint_graph_v1"
SPLIT_CONTRACT = "explicit_document_disjoint_internal_split_from_official_train_v1"
COMPACTION_CONTRACT = "uniform_sentence_bounded_mention_window_v1"
PARSER_IMPLEMENTATION = "verifier_csv_expat_union_find_v1"
SPLITS = ("private_train", "private_dev", "private_eval")
ANNOTATED = re.compile(r"^(.*)\[([0-9]+)\]$")
SLOT = re.compile(r"^([^[]+)\[([0-9]+)_([0-9]+)\]$")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _source_hash(root: Path, ids: list[str]) -> str:
    value = hashlib.sha256()
    files = [root / "LICENSE.md"]
    for document_id in sorted(ids):
        files += [
            root / "xml" / f"{document_id}.xml",
            root / "coref" / "gum" / "tsv" / f"{document_id}.tsv",
            root / "coref" / "gum" / "conll" / f"{document_id}.conll",
            root / "dep" / f"{document_id}.conllu",
        ]
    for path in files:
        value.update(f"{path.relative_to(root)}\0{_file_hash(path)[7:]}\n".encode())
    return "sha256:" + value.hexdigest()


def _xml_attributes(path: Path) -> dict[str, str]:
    attributes: dict[str, str] = {}
    parser = xml.parsers.expat.ParserCreate()

    def start(name: str, values: dict[str, str]) -> None:
        if not attributes:
            if name != "text":
                raise ValueError(f"GUM XML root is not text: {path.name}")
            attributes.update({str(key): str(value) for key, value in values.items()})

    parser.StartElementHandler = start
    with path.open("rb") as stream:
        parser.ParseFile(stream)
    return attributes


def _items(value: str) -> list[tuple[str, str]]:
    if value == "_":
        return []
    result = []
    for raw in value.split("|"):
        match = ANNOTATED.fullmatch(raw)
        if match is None:
            raise ValueError(f"GUM annotation lacks identity: {raw}")
        result.append((match.group(1), match.group(2)))
    return result


def _property(raw: str, identity: str) -> str:
    values = [name for name, item_id in _items(raw) if item_id == identity]
    if len(values) > 1:
        raise ValueError(f"ambiguous GUM property: {identity}")
    return values[0] if values else "_"


def _independent_document(path: Path) -> dict[str, Any]:
    entities: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    sentences: dict[int, tuple[int, int]] = {}
    extent = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for line_number, fields in enumerate(
            csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE), 1
        ):
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 10 or any(fields[10:]):
                raise ValueError(f"malformed GUM TSV row at {path}:{line_number}")
            fields = fields[:10]
            left, separator, right = fields[1].partition("-")
            if not separator or not left.isdecimal() or not right.isdecimal():
                raise ValueError(f"malformed GUM offset at {path}:{line_number}")
            start, end = int(left), int(right)
            if start >= end or end - start != len(fields[2]):
                raise ValueError(f"GUM token extent mismatch at {path}:{line_number}")
            sentence_id = int(fields[0].split("-", 1)[0])
            if sentence_id in sentences:
                old_start, old_end = sentences[sentence_id]
                sentences[sentence_id] = (min(old_start, start), max(old_end, end))
            else:
                sentences[sentence_id] = (start, end)
            token = {
                "token_id": fields[0],
                "sentence_id": str(sentence_id),
                "start": start,
                "end": end,
                "text": fields[2],
                "_document_token_ordinal": len(token_rows),
            }
            token_rows.append(token)
            extent = max(extent, end)
            current = _items(fields[3])
            current_ids = {identity for _kind, identity in current}
            for kind, identity in current:
                attributes = {
                    "entity_type": kind,
                    "information_status": _property(fields[4], identity),
                    "salience": _property(fields[5], identity),
                    "identity": _property(fields[6], identity),
                    "centering": _property(fields[7], identity),
                }
                entity = entities.setdefault(identity, {"mention_id": identity, "attributes": attributes, "tokens": [], "receipts": []})
                if entity["attributes"] != attributes:
                    raise ValueError(f"GUM mention attribute drift: {path}:{identity}")
                entity["tokens"].append(token)
                entity["receipts"].append(_hash({"line": line_number, "raw": "\t".join(fields) + "\t"}))
            kinds = [] if fields[8] == "_" else fields[8].split("|")
            targets = [] if fields[9] == "_" else fields[9].split("|")
            if len(kinds) != len(targets):
                raise ValueError(f"GUM relation arity mismatch: {path}:{line_number}")
            raw_line = "\t".join(fields) + "\t"
            for order, (kind, target) in enumerate(zip(kinds, targets)):
                match = SLOT.fullmatch(target)
                if match is None or match.group(3) not in current_ids or match.group(2) == match.group(3):
                    raise ValueError(f"GUM relation endpoint mismatch: {path}:{line_number}")
                links.append({
                    "relation_type": kind,
                    "source_mention_id": match.group(3),
                    "target_mention_id": match.group(2),
                    "target_token_id": match.group(1),
                    "edge_order": order,
                    "source_annotation_sha256": _hash({"line": line_number, "raw": raw_line, "order": order}),
                })
    document = [" "] * extent
    for token in token_rows:
        existing = "".join(document[token["start"] : token["end"]])
        if existing.strip() and existing != token["text"]:
            raise ValueError(f"GUM token overlap mismatch: {path}:{token['token_id']}")
        document[token["start"] : token["end"]] = token["text"]
    text = "".join(document)
    for entity in entities.values():
        parts = sorted(entity.pop("tokens"), key=lambda token: (token["start"], token["end"]))
        spans: list[list[int]] = []
        sentence_ids = set()
        for token in parts:
            sentence_ids.add(token["sentence_id"])
            if spans and token["start"] <= spans[-1][1] + 1:
                spans[-1][1] = token["end"]
            else:
                spans.append([token["start"], token["end"]])
        entity["document_spans"] = spans
        entity["sentence_ids"] = sorted(sentence_ids, key=int)
        entity["source_text"] = " ... ".join(text[a:b] for a, b in spans)
        entity["_document_token_ordinals"] = sorted(
            token["_document_token_ordinal"] for token in parts
        )
        entity["source_annotation_sha256"] = _hash({
            "mention_id": entity["mention_id"],
            "attributes": entity["attributes"],
            "rows": entity.pop("receipts"),
            "spans": spans,
        })
    if any(link["source_mention_id"] not in entities or link["target_mention_id"] not in entities for link in links):
        raise ValueError(f"unknown GUM relation endpoint: {path}")
    return {
        "document_text": text,
        "sentence_spans": {str(key): list(value) for key, value in sorted(sentences.items())},
        "mentions": entities,
        "relations": links,
        "_document_tokens": [token["text"] for token in token_rows],
    }


class _UnionFind:
    def __init__(self, identities: set[str]) -> None:
        self.parent = {identity: identity for identity in identities}

    def root(self, identity: str) -> str:
        path = []
        while self.parent[identity] != identity:
            path.append(identity)
            identity = self.parent[identity]
        for member in path:
            self.parent[member] = identity
        return identity

    def merge(self, left: str, right: str) -> None:
        left, right = self.root(left), self.root(right)
        if left != right:
            self.parent[max(left, right, key=int)] = min(left, right, key=int)


def _groups(document: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, str]]:
    union = _UnionFind(set(document["mentions"]))
    for relation in document["relations"]:
        if not relation["relation_type"].startswith("bridge:"):
            union.merge(relation["source_mention_id"], relation["target_mention_id"])
    result: dict[str, set[str]] = defaultdict(set)
    membership = {}
    for identity in document["mentions"]:
        group = union.root(identity)
        result[group].add(identity)
        membership[identity] = group
    return dict(result), membership


def _compact(document: dict[str, Any], identities: set[str], limit: int) -> tuple[str, list[dict[str, Any]], int]:
    spans = [span for identity in identities for span in document["mentions"][identity]["document_spans"]]
    sentence_bounds = []
    for span in spans:
        candidates = [bounds for bounds in document["sentence_spans"].values() if bounds[0] <= span[0] and span[1] <= bounds[1]]
        if len(candidates) != 1:
            raise ValueError("GUM mention does not map to one sentence")
        sentence_bounds.append(candidates[0])

    def choose(radius: int) -> list[list[int]]:
        raw = sorted([max(span[0] - radius, bounds[0]), min(span[1] + radius, bounds[1])] for span, bounds in zip(spans, sentence_bounds))
        result: list[list[int]] = []
        for start, end in raw:
            if result and start <= result[-1][1] + 1:
                result[-1][1] = max(result[-1][1], end)
            else:
                result.append([start, end])
        return result

    def size(windows: list[list[int]]) -> int:
        return sum(end - start for start, end in windows) + max(0, len(windows) - 1)

    if size(choose(0)) > limit:
        raise ValueError("complete GUM component exceeds source-length contract")
    low = 0
    high = max((end - start for start, end in document["sentence_spans"].values()), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if size(choose(middle)) <= limit:
            low = middle
        else:
            high = middle - 1
    windows = choose(low)
    pieces = []
    mappings = []
    cursor = 0
    for start, end in windows:
        if pieces:
            pieces.append("\n")
            cursor += 1
        value = document["document_text"][start:end]
        mappings.append(([start, end], [cursor, cursor + len(value)]))
        pieces.append(value)
        cursor += len(value)
    projected = []
    for identity in sorted(identities, key=int):
        mention = document["mentions"][identity]
        local_spans = []
        for span in mention["document_spans"]:
            matches = [(source, target) for source, target in mappings if source[0] <= span[0] and span[1] <= source[1]]
            if len(matches) != 1:
                raise ValueError("GUM mention is not fully preserved by source compaction")
            source, target = matches[0]
            local_spans.append([target[0] + span[0] - source[0], target[0] + span[1] - source[0]])
        projected.append(
            {
                **{
                    key: value
                    for key, value in mention.items()
                    if not key.startswith("_document_token_")
                },
                "excerpt_spans": local_spans,
            }
        )
    return "".join(pieces), projected, low


def _record(document_id: str, genre: str, license_spdx: str, split: str, document: dict[str, Any], group_ids: list[str], bridge: dict[str, Any] | None, limit: int) -> dict[str, Any]:
    groups, _membership = _groups(document)
    mention_ids = set().union(*(groups[group] for group in group_ids))
    text, mentions, radius = _compact(document, mention_ids, limit)
    edges = [edge for edge in document["relations"] if not edge["relation_type"].startswith("bridge:") and edge["source_mention_id"] in mention_ids and edge["target_mention_id"] in mention_ids]
    if bridge is not None:
        edges.append(bridge)
    edges.sort(key=lambda row: (int(row["source_mention_id"]), int(row["target_mention_id"]), row["relation_type"], row["source_annotation_sha256"]))
    kind = "bridge_relation" if bridge is not None else "identity_component"
    identity = _hash({"document_id": document_id, "kind": kind, "groups": group_ids, "bridge": bridge})
    annotation = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "source_compaction_contract": COMPACTION_CONTRACT,
        "document_id": document_id,
        "official_partition": "train",
        "genre": genre,
        "license_spdx": license_spdx,
        "record_kind": kind,
        "source_text": text,
        "uniform_context_radius_characters": radius,
        "groups": [{
            "group_id": group,
            "stable_identity": f"gum.entity.{document_id.lower()}.{group}",
            "mention_ids": sorted(groups[group], key=int),
            "complete_component": True,
        } for group in group_ids],
        "mentions": mentions,
        "relations": edges,
        "anchor_bridge_sha256": None if bridge is None else bridge["source_annotation_sha256"],
        "complete_source_declared_components": True,
        "partial_component_admission_count": 0,
        "inferred_relation_count": 0,
        "truth_claimed": False,
        "complete_sentence_semantics_claimed": False,
    }
    return {
        "source_id": f"gum-entity:{kind}:{identity[7:31]}",
        "source_group": f"gum-document:{document_id}",
        "split": split,
        "license_spdx": license_spdx,
        "source_text": text,
        "annotation": annotation,
    }


def _component_id(label: str) -> str:
    match = re.match(r"^[^-()]+-([0-9]+)(?:-|$)", label)
    if match is None:
        raise ValueError(f"malformed GUM CoNLL component label: {label}")
    return match.group(1)


def _conll_component_topology(path: Path) -> tuple[list[str], tuple[tuple[tuple[int, ...], ...], ...]]:
    tokens: list[str] = []
    active: dict[str, list[int]] = defaultdict(list)
    mentions: dict[str, list[tuple[int, ...]]] = defaultdict(list)

    def close(component_id: str, ordinal: int) -> None:
        if not active.get(component_id):
            raise ValueError(
                f"GUM CoNLL component closure mismatch: {path.name}:{component_id}"
            )
        start = active[component_id].pop()
        if not active[component_id]:
            del active[component_id]
        mentions[component_id].append(tuple(range(start, ordinal + 1)))

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 3 or not fields[0].isdigit():
            raise ValueError(f"malformed GUM CoNLL row: {path}:{line_number}")
        ordinal = len(tokens)
        tokens.append(fields[1])
        markup = fields[2]
        if markup == "_":
            continue
        cursor = 0
        while cursor < len(markup):
            if markup[cursor] == "(":
                cursor += 1
                end = cursor
                while end < len(markup) and markup[end] not in "()":
                    end += 1
                label = markup[cursor:end]
                component_id = _component_id(label)
                active[component_id].append(ordinal)
                cursor = end
                if cursor < len(markup) and markup[cursor] == ")":
                    close(component_id, ordinal)
                    cursor += 1
                continue
            end = cursor
            while end < len(markup) and markup[end] not in "()":
                end += 1
            label = markup[cursor:end]
            if end >= len(markup) or markup[end] != ")":
                raise ValueError(f"malformed GUM CoNLL closure: {path}:{line_number}")
            close(_component_id(label), ordinal)
            cursor = end + 1
    if active:
        raise ValueError(
            f"unclosed GUM CoNLL component: {path.name}:{sorted(active)[0]}"
        )
    topology = tuple(
        sorted(tuple(sorted(component_mentions)) for component_mentions in mentions.values())
    )
    return tokens, topology


def _tsv_component_topology(
    document: dict[str, Any], groups: dict[str, set[str]]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    tuple(document["mentions"][mention_id]["_document_token_ordinals"])
                    for mention_id in members
                )
            )
            for members in groups.values()
        )
    )


def _conllu_topology(path: Path) -> tuple[int, int]:
    mention_count = 0
    component_ids = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 10 or "-" in fields[0] or "." in fields[0]:
            continue
        match = re.search(r"(?:^|\|)Entity=([^|]+)", fields[9])
        if match is None:
            continue
        opened = re.findall(r"\(([0-9]+)-", match.group(1))
        mention_count += len(opened)
        component_ids.update(opened)
    return mention_count, len(component_ids)


def independently_reconstruct_gum_entity_coreference(*, source_root: Path, allowed_genre_licenses: dict[str, str], private_dev_documents: set[str], private_eval_documents: set[str], expected_selected_source_sha256: str, maximum_characters: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if private_dev_documents & private_eval_documents:
        raise ValueError("GUM entity/coreference private splits overlap")
    selected = []
    official_nontrain = 0
    for path in sorted((source_root / "xml").glob("GUM_*.xml")):
        metadata = _xml_attributes(path)
        if metadata.get("partition") != "train":
            official_nontrain += 1
        elif metadata.get("type") in allowed_genre_licenses:
            selected.append((metadata["id"], metadata["type"]))
    document_ids = [row[0] for row in selected]
    source_hash = _source_hash(source_root, document_ids)
    if source_hash != expected_selected_source_sha256:
        raise ValueError("GUM entity/coreference selected source content mismatch")
    if not private_dev_documents | private_eval_documents <= set(document_ids):
        raise ValueError("GUM entity/coreference split references excluded documents")
    output = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    relation_types = {split: Counter() for split in SPLITS}
    component_sizes = {split: Counter() for split in SPLITS}
    rejected = []
    cross_format = {split: Counter() for split in SPLITS}
    for document_id, genre in selected:
        split = "private_dev" if document_id in private_dev_documents else "private_eval" if document_id in private_eval_documents else "private_train"
        document = _independent_document(source_root / "coref" / "gum" / "tsv" / f"{document_id}.tsv")
        groups, membership = _groups(document)
        conll_tokens, conll_components = _conll_component_topology(
            source_root / "coref" / "gum" / "conll" / f"{document_id}.conll"
        )
        if conll_tokens != document["_document_tokens"]:
            raise ValueError(f"GUM TSV/CoNLL token alignment disagreement: {document_id}")
        if conll_components != _tsv_component_topology(document, groups):
            raise ValueError(
                f"GUM TSV/CoNLL component membership disagreement: {document_id}"
            )
        conllu_mentions, conllu_components = _conllu_topology(source_root / "dep" / f"{document_id}.conllu")
        if (conllu_mentions, conllu_components) != (len(document["mentions"]), len(groups)):
            raise ValueError(f"GUM TSV/CoNLL-U topology disagreement: {document_id}")
        cross_format[split]["documents_agreeing"] += 1
        cross_format[split]["component_membership_documents_agreeing"] += 1
        cross_format[split]["mentions"] += conllu_mentions
        cross_format[split]["components"] += conllu_components
        counts[split].update(documents=1, mentions=len(document["mentions"]), components=len(groups))
        for group in groups.values():
            component_sizes[split][str(len(group))] += 1
        for edge in document["relations"]:
            relation_types[split][edge["relation_type"]] += 1
        for group_id, members in sorted(groups.items(), key=lambda row: int(row[0])):
            if len(members) < 2:
                continue
            try:
                output[split].append(_record(document_id, genre, allowed_genre_licenses[genre], split, document, [group_id], None, maximum_characters))
            except ValueError as error:
                rejected.append({"document_id": document_id, "record_kind": "identity_component", "group_ids": [group_id], "reason": str(error)})
        for edge in document["relations"]:
            if not edge["relation_type"].startswith("bridge:"):
                continue
            group_ids = sorted({membership[edge["source_mention_id"]], membership[edge["target_mention_id"]]}, key=int)
            if len(group_ids) != 2:
                rejected.append({"document_id": document_id, "record_kind": "bridge_relation", "group_ids": group_ids, "reason": "bridge endpoints collapse into one identity component"})
                continue
            try:
                output[split].append(_record(document_id, genre, allowed_genre_licenses[genre], split, document, group_ids, edge, maximum_characters))
            except ValueError as error:
                rejected.append({"document_id": document_id, "record_kind": "bridge_relation", "group_ids": group_ids, "reason": str(error)})
    for split in SPLITS:
        output[split].sort(key=lambda row: row["source_id"])
        counts[split]["records"] = len(output[split])
        counts[split]["identity_records"] = sum(row["annotation"]["record_kind"] == "identity_component" for row in output[split])
        counts[split]["bridge_records"] = sum(row["annotation"]["record_kind"] == "bridge_relation" for row in output[split])
    audit = {
        "policy": POLICY,
        "relation_contract": RELATION_CONTRACT,
        "split_contract": SPLIT_CONTRACT,
        "source_compaction_contract": COMPACTION_CONTRACT,
        "parser_implementation": PARSER_IMPLEMENTATION,
        "selected_source_sha256": source_hash,
        "selected_document_count": len(selected),
        "counts_by_split": {split: dict(values) for split, values in counts.items()},
        "relation_type_count_by_split": {split: dict(values) for split, values in relation_types.items()},
        "component_size_distribution_by_split": {split: dict(values) for split, values in component_sizes.items()},
        "cross_format_topology_by_split": {split: dict(values) for split, values in cross_format.items()},
        "rejected_record_count": len(rejected),
        "rejection_reason_counts": dict(Counter(row["reason"] for row in rejected)),
        "rejected_records": rejected,
        "official_nontrain_document_admission_count": 0,
        "official_nontrain_document_seen_count": official_nontrain,
        "partial_component_admission_count": 0,
        "inferred_relation_count": 0,
    }
    return output, audit
