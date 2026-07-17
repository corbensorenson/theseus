#!/usr/bin/env python3
"""Produce source-bound KERC candidates from licensed human task data.

This producer is intentionally not an admission authority. It parses pinned
source artifacts and emits candidate records; ``kerc_semantic_corpus_verify``
must independently replay the raw sources before canonical rows exist.
"""

from __future__ import annotations

import argparse
import base64
import copy
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
    SEMANTIC_SUPERVISION_POLICY,
    TRAINING_OBJECTIVES,
    TRAINING_RECORD_POLICY,
    build_kernel_packet,
    stable_hash,
)
from moecot_source_conditioned_pretraining import (
    KERC_SEMANTIC_CORPUS_POLICY,
    validate_kernel_english_config,
)
from vcm_semantic_memory import create_hierarchical_residual_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moecot_language_arm_training.json"
GRAF = "{http://www.xces.org/ns/GrAF/1.0/}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
PRODUCER_POLICY = "project_theseus_kerc_semantic_corpus_producer_v1"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def byte_literal(text: str) -> dict[str, Any]:
    return {
        "type": "byte_literal",
        "value": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def safe_symbol(value: str, *, prefix: str) -> str:
    symbol = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    if not symbol or not symbol[0].isalpha():
        symbol = prefix + "_" + symbol
    return symbol[:96]


def semantic_evidence(
    *,
    source: dict[str, Any],
    source_id: str,
    objectives: set[str],
    producer_sha256: str,
) -> dict[str, Any]:
    return {
        "policy": SEMANTIC_SUPERVISION_POLICY,
        "evidence_tier": "licensed_human_task_gold",
        "producer_kind": "licensed_semantic_dataset",
        "producer_id": PRODUCER_POLICY,
        "producer_artifact_sha256": producer_sha256,
        "annotation_source_id": source_id,
        "annotation_source_sha256": source["content_sha256"],
        "claim_authority": "decision_grade_reference",
        "model_derived": False,
        "public_calibration_surface": False,
        "benchmark_payload_used": False,
        "objective_authority": {
            objective: objective in objectives for objective in TRAINING_OBJECTIVES
        },
        "optimizer_sampling_weight": 1.0,
    }


def scope(identity: str) -> dict[str, Any]:
    return {
        "user": "project-theseus-corpus",
        "project": "theseus",
        "conversation": identity,
        "privacy": "private_local",
    }


def residual_supervision(identity: str, *, packet: dict[str, Any]) -> dict[str, Any]:
    residual = packet["residual"]
    fidelity_labels = {"semantic": 0, "faithful": 1, "lexical": 2, "exact": 3}
    labels = {
        "interaction": 0,
        "segment": 1 if residual["segment_frame"] else 0,
        "token": 2 if residual["token_tags"] else 0,
        "exact": 3 if residual["exact_object_handles"] else 0,
    }
    return {
        "policy": "project_theseus_kerc_residual_supervision_v1",
        "labels_by_channel": labels,
        "record_fidelity_label": fidelity_labels[residual["fidelity"]],
        "annotator_independent_of_model": True,
        "evidence_sha256": stable_hash(
            {"identity": identity, "labels_by_channel": labels, "rule": "source_fidelity_v1"}
        ),
    }


def provisional_receipt(identity: str) -> dict[str, Any]:
    return {
        "policy": "project_theseus_kernel_english_verification_receipt_v1",
        "receipt_id": "pending:" + identity,
        "accepted": False,
        "verifier_id": "independent_verifier_required",
        "reviewer_independent_of_record_producer": False,
        "method": "licensed_semantic_dataset_plus_independent_schema_review",
        "evidence_sha256": "sha256:" + "0" * 64,
        "semantic_payload_sha256": "sha256:" + "0" * 64,
    }


def base_record(
    *,
    split: str,
    source_text: str,
    surface_target: str,
    program: dict[str, Any],
    answer_packet: dict[str, Any],
    source: dict[str, Any],
    source_id: str,
    source_group: str,
    objectives: set[str],
    producer_sha256: str,
    source_annotation: dict[str, Any],
    exact_residual: bool,
) -> dict[str, Any]:
    identity = stable_hash(
        {
            "split": split,
            "source_id": source_id,
            "source_group": source_group,
            "source_text": source_text,
            "surface_target": surface_target,
            "source_annotation": source_annotation,
        }
    ).split(":", 1)[1]
    state = create_hierarchical_residual_state(
        "kerc-corpus-" + identity[:24], scope=scope(identity[:24])
    )
    packet = build_kernel_packet(
        source_text,
        program,
        hrl_state=state,
        provenance={
            "source": source["dataset_id"],
            "source_id": source_id,
            "source_annotation_sha256": stable_hash(source_annotation),
        },
        fidelity="exact" if exact_residual else "faithful",
    )
    return {
        "policy": TRAINING_RECORD_POLICY,
        "split": split,
        "language": "en",
        "source_text": source_text,
        "kernel_packet": packet,
        "hrl_state": state,
        "answer_packet": answer_packet,
        "surface_target": surface_target,
        "provenance": {
            "source_id": source_id,
            "source_group": source_group,
            "license_spdx": source["license_spdx"],
            "permitted_use": "model_training",
            "dataset_id": source["dataset_id"],
            "dataset_revision": source["dataset_revision"],
        },
        "semantic_supervision": semantic_evidence(
            source=source,
            source_id=source_id,
            objectives=objectives,
            producer_sha256=producer_sha256,
        ),
        "residual_supervision": residual_supervision(identity, packet=packet),
        "verification_receipt": provisional_receipt(identity[:24]),
        "source_annotation": source_annotation,
        "public_benchmark": False,
        "public_tests_included": False,
        "public_benchmark_solutions_included": False,
        "external_inference": False,
        "fallback_return_count": 0,
        "template_credit": 0,
        "deterministic_renderer_credit": 0,
        "candidate_generation_credit": 0,
    }


def dolly_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction") or "").strip()
    context = str(row.get("context") or "").strip()
    return instruction + (("\n\nContext:\n" + context) if context else "")


def load_dolly_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    path = resolve(source["path"])
    if sha256_file(path) != source["content_sha256"]:
        raise ValueError("Dolly source hash mismatch")
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    seen_targets: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        source_row = json.loads(raw)
        prompt = dolly_prompt(source_row)
        target = str(source_row.get("response") or "").strip()
        if not 12 <= len(prompt) <= maximum_characters or not 2 <= len(target) <= maximum_characters:
            rejects["length"] += 1
            continue
        prompt_hash = stable_hash(prompt.encode("utf-8"))
        target_hash = stable_hash(target.encode("utf-8"))
        if prompt_hash in seen_prompts or target_hash in seen_targets:
            rejects["duplicate_prompt_or_target"] += 1
            continue
        seen_prompts.add(prompt_hash)
        seen_targets.add(target_hash)
        annotation = {
            "source_kind": "dolly_human_instruction_response",
            "line_number": line_number,
            "instruction": str(source_row.get("instruction") or ""),
            "context": str(source_row.get("context") or ""),
            "response": str(source_row.get("response") or ""),
            "category": str(source_row.get("category") or ""),
            "source_row_sha256": stable_hash(source_row),
        }
        rows.append(
            {
                "selection_key": stable_hash(
                    {"dataset": source["dataset_id"], "annotation": annotation}
                ),
                "prompt": prompt,
                "target": target,
                "annotation": annotation,
                "category": annotation["category"],
            }
        )
    return sorted(rows, key=lambda row: row["selection_key"]), rejects


def dolly_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": "RESPOND",
                "modality": "REQUIRED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": [[0, len(row["prompt"])]],
                "arguments": [
                    {"role": "TASK", "value": byte_literal(row["prompt"])},
                ],
            }
        ],
    }
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": "RESPOND",
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": [
                    {"role": "OUTPUT", "value": byte_literal(row["target"])},
                ],
            }
        ],
        "required_terms": [],
        "required_caveats": [],
        "style": {"register": "source_authored"},
    }
    identity = row["selection_key"].split(":", 1)[1]
    return base_record(
        split=split,
        source_text=row["prompt"],
        surface_target=row["target"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="dolly:" + identity[:24],
        source_group="dolly-row:" + identity,
        objectives={"surface_direct_control_v1"},
        producer_sha256=producer_sha256,
        source_annotation=row["annotation"],
        exact_residual=False,
    )


def parse_graf(path: Path) -> tuple[
    dict[str, list[tuple[str, dict[str, str]]]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
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
        feature_structure = annotation.find(GRAF + "fs")
        if feature_structure is not None:
            for field in feature_structure.findall(GRAF + "f"):
                fields[str(field.get("name"))] = str(field.get("value") or "")
        annotations[str(annotation.get("ref"))].append(
            (str(annotation.get("label") or ""), fields)
        )
    for edge in root.findall(GRAF + "edge"):
        edges[str(edge.get("from"))].append(str(edge.get("to")))
    return dict(annotations), dict(edges), links


def descendants(node: str, edges: dict[str, list[str]]) -> set[str]:
    output: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in output:
            continue
        output.add(current)
        stack.extend(edges.get(current, ()))
    return output


def field_rows(
    annotations: dict[str, list[tuple[str, dict[str, str]]]], node: str, label: str
) -> list[dict[str, str]]:
    return [fields for observed, fields in annotations.get(node, ()) if observed == label]


def token_spans(
    node: str,
    *,
    edges: dict[str, list[str]],
    token_links: dict[str, list[str]],
    anchors: dict[str, tuple[int, int]],
) -> list[tuple[int, int]]:
    spans = {
        anchors[region]
        for descendant in descendants(node, edges)
        for region in token_links.get(descendant, ())
        if region in anchors
    }
    return sorted(spans)


def masc_document_instances(path: Path, root: Path) -> list[dict[str, Any]]:
    base = Path(str(path)[: -len("-fn.xml")])
    document_id = str(base.relative_to(root)).replace(os.sep, "/")
    source_text = Path(str(base) + ".txt").read_text(
        encoding="utf-8", errors="replace"
    )
    annotations, edges, _ = parse_graf(path)
    _, _, token_links = parse_graf(Path(str(base) + "-fntok.xml"))
    segment_root = ET.parse(Path(str(base) + "-seg.xml")).getroot()
    anchors = {
        str(region.get(XML_ID)): tuple(
            int(value) for value in str(region.get("anchors") or "").split()
        )
        for region in segment_root.findall(GRAF + "region")
    }
    output: list[dict[str, Any]] = []
    sentence_nodes = [
        node
        for node, rows in annotations.items()
        if any(label == "sentence" for label, _fields in rows)
    ]
    for sentence_node in sentence_nodes:
        sentence_token_spans = token_spans(
            sentence_node, edges=edges, token_links=token_links, anchors=anchors
        )
        if not sentence_token_spans:
            continue
        sentence_start = min(start for start, _end in sentence_token_spans)
        sentence_end = max(end for _start, end in sentence_token_spans)
        sentence = source_text[sentence_start:sentence_end]
        for annotation_node in edges.get(sentence_node, ()):
            sets = field_rows(annotations, annotation_node, "annotationSet")
            if not sets:
                continue
            frame = sets[0]
            if frame.get("status") != "MANUAL" or not frame.get("frameName"):
                continue
            target_spans: list[tuple[int, int]] = []
            frame_elements: list[dict[str, Any]] = []
            for child in edges.get(annotation_node, ()):
                spans = token_spans(child, edges=edges, token_links=token_links, anchors=anchors)
                relative_spans = [
                    [start - sentence_start, end - sentence_start]
                    for start, end in spans
                    if sentence_start <= start < end <= sentence_end
                ]
                if field_rows(annotations, child, "Target"):
                    target_spans.extend((int(a), int(b)) for a, b in relative_spans)
                for fields in field_rows(annotations, child, "FE"):
                    if relative_spans:
                        frame_elements.append(
                            {
                                "role": str(fields.get("FE") or "ROLE"),
                                "rank": str(fields.get("rank") or ""),
                                "gf": str(fields.get("GF") or ""),
                                "pt": str(fields.get("PT") or ""),
                                "spans": relative_spans,
                                "text": " ".join(
                                    sentence[start:end] for start, end in relative_spans
                                ),
                            }
                        )
            if not target_spans or not frame_elements:
                continue
            annotation = {
                "source_kind": "masc_manual_framenet",
                "document_id": document_id,
                "sentence_node": sentence_node,
                "annotation_set_node": annotation_node,
                "annotation_set_id": str(frame.get("ID") or ""),
                "frame_name": str(frame["frameName"]),
                "lexical_unit": str(frame.get("luName") or ""),
                "status": str(frame.get("status") or ""),
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "sentence": sentence,
                "target_spans": [list(value) for value in sorted(set(target_spans))],
                "frame_elements": sorted(
                    frame_elements,
                    key=lambda row: (row["role"], row["spans"], row["text"]),
                ),
            }
            output.append(
                {
                    "selection_key": stable_hash(annotation),
                    "document_id": document_id,
                    "sentence_identity": stable_hash(
                        {"document_id": document_id, "sentence_node": sentence_node}
                    ),
                    "annotation": annotation,
                }
            )
    return output


def load_masc_candidates(
    source: dict[str, Any], *, maximum_characters: int
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    archive = resolve(source["archive_path"])
    if sha256_file(archive) != source["content_sha256"]:
        raise ValueError("MASC source hash mismatch")
    root = resolve(source["extracted_root"]) / "data"
    dev_groups = set(source["document_groups"]["private_dev"])
    eval_groups = set(source["document_groups"]["private_eval"])
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejects: Counter[str] = Counter()
    for path in sorted(root.rglob("*-fn.xml")):
        document_id = str(Path(str(path)[: -len("-fn.xml")]).relative_to(root)).replace(
            os.sep, "/"
        )
        split = (
            "private_dev"
            if document_id in dev_groups
            else "private_eval"
            if document_id in eval_groups
            else "private_train"
        )
        for row in masc_document_instances(path, root):
            sentence = row["annotation"]["sentence"]
            if not 12 <= len(sentence) <= maximum_characters:
                rejects["sentence_length"] += 1
                continue
            selected[split].append(row)
    return {
        split: sorted(rows, key=lambda row: row["selection_key"])
        for split, rows in selected.items()
    }, rejects


def masc_record(
    row: dict[str, Any],
    *,
    split: str,
    source: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    annotation = row["annotation"]
    arguments = [
        {
            "role": safe_symbol(element["role"], prefix="ROLE"),
            "value": byte_literal(element["text"]),
        }
        for element in annotation["frame_elements"]
    ]
    predicate = "FRAME_" + safe_symbol(annotation["frame_name"], prefix="UNKNOWN")
    program = {
        "roots": ["k0"],
        "nodes": [
            {
                "node_id": "k0",
                "operator": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "derivation": "preserved",
                "source_spans": annotation["target_spans"],
                "arguments": copy.deepcopy(arguments),
            }
        ],
    }
    answer = {
        "claims": [
            {
                "claim_id": "claim-1",
                "predicate": predicate,
                "modality": "ASSERTED",
                "polarity": "AFFIRMED",
                "quantifier": "NONE",
                "confidence": 1.0,
                "arguments": copy.deepcopy(arguments),
            }
        ],
        "required_terms": [
            {
                "concept": "framenet."
                + re.sub(r"[^a-z0-9]+", "_", annotation["frame_name"].lower()).strip("_"),
                "surface_policy": "source_licensed_frame",
            }
        ],
        "required_caveats": [],
        "style": {"register": "source_authored"},
    }
    identity = row["selection_key"].split(":", 1)[1]
    return base_record(
        split=split,
        source_text=annotation["sentence"],
        surface_target=annotation["sentence"],
        program=program,
        answer_packet=answer,
        source=source,
        source_id="masc-frame:" + identity[:24],
        source_group="masc-document:" + annotation["document_id"],
        objectives={
            "surface_to_kernel_program_v1",
            "kernel_program_to_answer_packet_v1",
            "answer_packet_to_surface_v1",
        },
        producer_sha256=producer_sha256,
        source_annotation=annotation,
        exact_residual=True,
    )


def partition_dolly(
    rows: list[dict[str, Any]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    needed = sum(int(value) for value in counts.values())
    if len(rows) < needed:
        raise ValueError(f"insufficient Dolly rows: {len(rows)} < {needed}")
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split, count in counts.items():
        output[split] = rows[offset : offset + int(count)]
        offset += int(count)
    return output


def select_masc(
    rows: dict[str, list[dict[str, Any]]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        available = rows.get(split, [])
        if len(available) < int(count):
            raise ValueError(f"insufficient MASC rows: {split}:{len(available)}:{count}")
        output[split] = available[: int(count)]
    return output


def produce(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage = validate_kernel_english_config(config)
    corpus = stage["semantic_corpus_materialization"]
    if corpus["policy"] != KERC_SEMANTIC_CORPUS_POLICY:
        raise ValueError("semantic corpus contract mismatch")
    producer_sha256 = sha256_file(Path(__file__).resolve())
    dolly_rows, dolly_rejects = load_dolly_candidates(
        corpus["dolly"], maximum_characters=int(corpus["maximum_source_characters"])
    )
    dolly_selected = partition_dolly(
        dolly_rows, corpus["dolly"]["records_by_split"]
    )
    masc_rows, masc_rejects = load_masc_candidates(
        corpus["masc"], maximum_characters=int(corpus["maximum_source_characters"])
    )
    masc_selected = select_masc(masc_rows, corpus["masc"]["records_by_split"])
    candidates: list[dict[str, Any]] = []
    source_groups_by_split: dict[str, set[str]] = defaultdict(set)
    source_sentences_by_split: dict[str, set[str]] = defaultdict(set)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    frame_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("private_train", "private_dev", "private_eval"):
        for row in dolly_selected[split]:
            record = dolly_record(
                row,
                split=split,
                source=corpus["dolly"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(stable_hash(record["source_text"]))
            category_counts[split][row["category"]] += 1
        for row in masc_selected[split]:
            record = masc_record(
                row,
                split=split,
                source=corpus["masc"],
                producer_sha256=producer_sha256,
            )
            candidates.append(record)
            source_groups_by_split[split].add(record["provenance"]["source_group"])
            source_sentences_by_split[split].add(row["sentence_identity"])
            frame_counts[split][row["annotation"]["frame_name"]] += 1
    candidates.sort(
        key=lambda row: (
            ("private_train", "private_dev", "private_eval").index(row["split"]),
            row["provenance"]["source_id"],
        )
    )
    hard_gaps: list[str] = []
    for split, floor in corpus["minimum_source_groups_by_split"].items():
        if len(source_groups_by_split[split]) < int(floor):
            hard_gaps.append(f"insufficient_source_groups:{split}")
    for split, floor in corpus["minimum_source_sentences_by_split"].items():
        if len(source_sentences_by_split[split]) < int(floor):
            hard_gaps.append(f"insufficient_source_sentences:{split}")
    groups = list(source_groups_by_split.values())
    if any(groups[left] & groups[right] for left in range(len(groups)) for right in range(left + 1, len(groups))):
        hard_gaps.append("cross_split_source_group_overlap")
    output_path = resolve(corpus["candidate_records_jsonl"])
    row_count, output_sha256 = write_jsonl_atomic(output_path, candidates)
    report = {
        "policy": PRODUCER_POLICY,
        "trigger_state": "RED" if hard_gaps else "GREEN",
        "config": relative(config_path),
        "producer_sha256": producer_sha256,
        "candidate_records": {
            "path": relative(output_path),
            "sha256": output_sha256,
            "row_count": row_count,
        },
        "rows_by_split_and_source": {
            split: {
                "dolly": len(dolly_selected[split]),
                "masc": len(masc_selected[split]),
            }
            for split in ("private_train", "private_dev", "private_eval")
        },
        "source_group_count_by_split": {
            split: len(values) for split, values in source_groups_by_split.items()
        },
        "source_sentence_count_by_split": {
            split: len(values) for split, values in source_sentences_by_split.items()
        },
        "dolly_category_counts": {
            split: dict(values) for split, values in category_counts.items()
        },
        "masc_frame_counts": {
            split: dict(values) for split, values in frame_counts.items()
        },
        "candidate_pool_counts": {
            "dolly": len(dolly_rows),
            "masc_by_split": {split: len(rows) for split, rows in masc_rows.items()},
        },
        "rejection_counts": {
            "dolly": dict(dolly_rejects),
            "masc": dict(masc_rejects),
        },
        "independent_verification_required": True,
        "canonical_training_rows_written": 0,
        "public_training_rows_written": 0,
        "public_benchmark_payload_count": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "template_credit": 0,
        "score_semantics": "licensed semantic candidate production; not admission or capability",
        "hard_gaps": hard_gaps,
    }
    write_json_atomic(resolve(corpus["producer_manifest_json"]), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    report = produce(resolve(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
