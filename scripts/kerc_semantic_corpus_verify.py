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

from kernel_english_protocol import (
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
    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    return instruction + (("\n\nContext:\n" + context) if context else "")


def independent_dolly_assignments(source: dict[str, Any], maximum_characters: int) -> dict[str, dict[str, Any]]:
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
        if not 12 <= len(prompt) <= maximum_characters or not 2 <= len(target) <= maximum_characters:
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
    if record.get("interaction_annotation") is not None or record.get("hrl_deltas"):
        raise ValueError("Dolly record received unsupported interaction state")
    if (record.get("residual_supervision") or {}).get("labels_by_channel", {}).get(
        "interaction"
    ) != 0:
        raise ValueError("Dolly interaction residual label must be zero")
    return {"source_row_sha256": expected["annotation"]["source_row_sha256"]}


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
    for key in ("dolly", "masc"):
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
                "allowed_objectives": list(source["allowed_objectives"]),
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
    expected = {
        **independent_dolly_assignments(corpus["dolly"], maximum_characters),
        **independent_masc_assignments(corpus["masc"], maximum_characters),
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
            source_key = "dolly" if dataset_id == corpus["dolly"]["dataset_id"] else "masc" if dataset_id == corpus["masc"]["dataset_id"] else ""
            if not source_key:
                raise ValueError("candidate dataset absent from frozen source contract")
            source = corpus[source_key]
            provenance = record["provenance"]
            if provenance.get("dataset_revision") != source["dataset_revision"] or provenance.get("license_spdx") != source["license_spdx"]:
                raise ValueError("candidate provenance mismatch")
            semantic = record.get("semantic_supervision") or {}
            if semantic.get("annotation_source_sha256") != source["content_sha256"] or semantic.get("producer_artifact_sha256") != manifest["producer_sha256"]:
                raise ValueError("candidate semantic source binding mismatch")
            replay = verify_dolly_record(record, source, expected_row) if source_key == "dolly" else verify_masc_record(record, source, expected_row)
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
