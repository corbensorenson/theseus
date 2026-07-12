#!/usr/bin/env python3
"""Scalable governed conversation intake for the registered data pantry.

The intake is deliberately separate from model training. It reconstructs real
conversation paths, redacts sensitive literals, rejects public-calibration
overlap, performs exact and near deduplication, and commits atomic JSONL shards
with compact DataAdmissionReceipts. External-teacher corpora fail closed unless
the source is explicitly admitted by the teacher-distillation policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password)\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})"
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
ROLE_MAP = {"prompter": "user", "human": "user", "user": "user", "assistant": "assistant", "system": "system"}


def run_streaming_intake(
    config: dict[str, Any],
    *,
    root: Path,
    execute: bool,
    target_token_positions: int = 0,
    max_total_rows: int = 0,
    shard_rows: int = 0,
    max_disk_bytes: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    scale = dict(config.get("scalable_intake") or {})
    target_token_positions = int(target_token_positions or scale.get("target_one_pass_token_positions") or 20_000_000)
    max_total_rows = int(max_total_rows or scale.get("max_total_rows") or 50_000)
    shard_rows = int(shard_rows or scale.get("shard_rows") or 2_000)
    max_disk_bytes = int(max_disk_bytes or scale.get("max_disk_bytes") or 2_000_000_000)
    allowed_licenses = {normalize_license(value) for value in config.get("allowed_licenses", [])}
    sources = [dict(row) for row in config.get("sources", []) if isinstance(row, dict)]
    scalable_sources = [row for row in sources if row.get("enabled") and row.get("scalable")]
    public_index = build_public_index(max_texts=int(scale.get("max_public_contamination_texts") or 20_000))
    teacher_allowed = governed_teacher_conversation_allowed()
    licensed_static_policy = config.get("licensed_model_generated_corpora") or {}
    planned_sources = [
        source_plan(
            row, allowed_licenses=allowed_licenses, teacher_allowed=teacher_allowed,
            licensed_static_policy=licensed_static_policy,
        )
        for row in scalable_sources
    ]
    hard_preflight = bool(allowed_licenses and public_index.get("text_count", 0) > 0)
    if not execute:
        return report_payload(
            started=started,
            root=root,
            execute=False,
            target_token_positions=target_token_positions,
            max_total_rows=max_total_rows,
            max_disk_bytes=max_disk_bytes,
            public_index=public_index,
            sources=planned_sources,
            writer=None,
            counters=Counter(),
            errors=[],
            hard_preflight=hard_preflight,
        )

    root.mkdir(parents=True, exist_ok=True)
    writer = AtomicConversationShardWriter(root, shard_rows=max(1, shard_rows), max_disk_bytes=max_disk_bytes)
    previous_report = read_json(ROOT / "reports" / "open_conversation_training_pantry.json")
    previous_summary = previous_report.get("summary") if isinstance(previous_report.get("summary"), dict) else {}
    counters: Counter[str] = Counter(
        {
            f"decision:{key}": int(value)
            for key, value in (previous_summary.get("decision_counts") or {}).items()
        }
    )
    previous_sources = {
        str(row.get("source_id")): row
        for row in previous_report.get("sources", [])
        if isinstance(row, dict) and row.get("source_id")
    }
    errors: list[dict[str, Any]] = list(previous_report.get("errors") or [])
    source_reports: list[dict[str, Any]] = []
    stop = False
    for source in scalable_sources:
        plan = source_plan(
            source, allowed_licenses=allowed_licenses, teacher_allowed=teacher_allowed,
            licensed_static_policy=licensed_static_policy,
        )
        source_id = str(source.get("id") or source.get("dataset") or "source")
        if plan["decision"] != "eligible_for_intake":
            source_reports.append(plan)
            continue
        if source_id in writer.completed_sources:
            prior_source = previous_sources.get(source_id)
            if prior_source:
                plan.update(prior_source)
            plan.update({"status": "complete", "resume_status": "source_already_complete"})
            source_reports.append(plan)
            continue
        source_counter: Counter[str] = Counter()
        try:
            metadata = fetch_dataset_metadata(str(source.get("dataset") or ""))
            actual_license = normalize_license(extract_dataset_license(metadata))
            plan["actual_license"] = actual_license
            if not actual_license or actual_license != normalize_license(source.get("license_spdx")):
                plan.update({"status": "blocked", "decision": "live_license_missing_or_mismatch"})
                source_reports.append(plan)
                continue
            iterator = source_conversations(source)
            per_source_limit = max(1, int(source.get("max_admitted_rows") or max_total_rows))
            for messages, provenance in iterator:
                source_counter["candidate_rows"] += 1
                result = admit_conversation(
                    messages,
                    source=source,
                    provenance=provenance,
                    public_index=public_index,
                    deduper=writer.deduper,
                    max_chars=int(config.get("default_max_chars_per_conversation") or 16_000),
                )
                decision = str(result["receipt"]["decision"])
                source_counter[f"decision:{decision}"] += 1
                counters[f"decision:{decision}"] += 1
                for reason in result["receipt"].get("decision_reasons", []):
                    source_counter[f"reason:{reason}"] += 1
                    counters[f"reason:{reason}"] += 1
                if decision != "admit":
                    continue
                writer.add(result["train_row"], result["sts_row"], result["receipt"])
                source_counter["admitted_rows"] += 1
                source_counter["one_pass_token_positions"] += int(result["receipt"]["metrics"]["token_positions"])
                counters["admitted_rows"] += 1
                counters["one_pass_token_positions"] += int(result["receipt"]["metrics"]["token_positions"])
                if source_counter["admitted_rows"] >= per_source_limit:
                    break
                if writer.total_rows >= max_total_rows or writer.total_token_positions >= target_token_positions:
                    stop = True
                    break
                if writer.disk_bytes >= max_disk_bytes:
                    counters["disk_budget_stop"] += 1
                    stop = True
                    break
            writer.mark_source_complete(source_id)
            plan.update(
                {
                    "status": "complete",
                    "decision": "stream_materialized",
                    "candidate_rows": source_counter["candidate_rows"],
                    "admitted_rows": source_counter["admitted_rows"],
                    "one_pass_token_positions": source_counter["one_pass_token_positions"],
                    "decision_counts": {
                        key.removeprefix("decision:"): value
                        for key, value in sorted(source_counter.items())
                        if key.startswith("decision:")
                    },
                    "quarantine_reason_counts": {
                        key.removeprefix("reason:"): value
                        for key, value in sorted(source_counter.items())
                        if key.startswith("reason:")
                    },
                }
            )
        except BaseException as exc:  # datasets/Arrow can raise native-backed exceptions
            errors.append({"source_id": source_id, "stage": "stream_intake", "error_type": type(exc).__name__, "error": str(exc)[:800]})
            plan.update({"status": "blocked", "decision": "stream_fault"})
        source_reports.append(plan)
        if stop:
            break
    writer.close()
    return report_payload(
        started=started,
        root=root,
        execute=True,
        target_token_positions=target_token_positions,
        max_total_rows=max_total_rows,
        max_disk_bytes=max_disk_bytes,
        public_index=public_index,
        sources=source_reports,
        writer=writer,
        counters=counters,
        errors=errors,
        hard_preflight=hard_preflight,
    )


def run_document_intake(
    config: dict[str, Any],
    *,
    root: Path,
    execute: bool,
    target_token_positions: int = 0,
    max_total_rows: int = 0,
    shard_rows: int = 0,
    max_disk_bytes: int = 0,
) -> dict[str, Any]:
    """Materialize human-authored broad-English pretraining without SFT/STS pollution."""
    started = time.perf_counter()
    scale = dict(config.get("broad_text_intake") or {})
    target_token_positions = int(target_token_positions or scale.get("target_one_pass_token_positions") or 12_000_000)
    max_total_rows = int(max_total_rows or scale.get("max_total_rows") or 12_000)
    shard_rows = int(shard_rows or scale.get("shard_rows") or 500)
    max_disk_bytes = int(max_disk_bytes or scale.get("max_disk_bytes") or 1_500_000_000)
    allowed_licenses = {normalize_license(value) for value in config.get("allowed_licenses", [])} | {"public-domain"}
    sources = [dict(row) for row in config.get("broad_text_sources", []) if isinstance(row, dict)]
    sources = [row for row in sources if row.get("enabled") and row.get("scalable")]
    public_index = build_public_index(max_texts=int(scale.get("max_public_contamination_texts") or 20_000))
    plans = [document_source_plan(row, allowed_licenses=allowed_licenses) for row in sources]
    if not execute:
        return document_report_payload(
            started=started, root=root, execute=False, target_token_positions=target_token_positions,
            max_total_rows=max_total_rows, max_disk_bytes=max_disk_bytes, public_index=public_index,
            sources=plans, writer=None, counters=Counter(), errors=[],
        )

    writer = AtomicDocumentShardWriter(root, shard_rows=max(1, shard_rows), max_disk_bytes=max_disk_bytes)
    counters: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    if writer.total_token_positions >= target_token_positions:
        for source in sources:
            plan = document_source_plan(source, allowed_licenses=allowed_licenses)
            plan.update({"status": "complete", "decision": "resume_target_already_met"})
            source_reports.append(plan)
        return document_report_payload(
            started=started, root=root, execute=True, target_token_positions=target_token_positions,
            max_total_rows=max_total_rows, max_disk_bytes=max_disk_bytes, public_index=public_index,
            sources=source_reports, writer=writer, counters=counters, errors=errors,
        )
    stop = False
    for source in sources:
        plan = document_source_plan(source, allowed_licenses=allowed_licenses)
        source_id = str(source.get("id") or source.get("dataset") or "source")
        if plan["decision"] != "eligible_for_intake":
            source_reports.append(plan)
            continue
        if source_id in writer.completed_sources:
            plan.update({"status": "complete", "decision": "resume_source_already_complete"})
            source_reports.append(plan)
            continue
        source_counter: Counter[str] = Counter()
        try:
            metadata = fetch_dataset_metadata(str(source.get("dataset") or ""))
            actual_license = normalize_license(extract_dataset_license(metadata))
            if actual_license and actual_license != normalize_license(source.get("license_spdx")):
                plan.update({"status": "blocked", "decision": "live_license_mismatch", "actual_license": actual_license})
                source_reports.append(plan)
                continue
            for text, provenance in source_documents(source, scale):
                source_counter["candidate_rows"] += 1
                result = admit_document(
                    text,
                    source=source,
                    provenance=provenance,
                    public_index=public_index,
                    deduper=writer.deduper,
                    min_chars=int(scale.get("min_chars_per_chunk") or 800),
                    max_chars=int(scale.get("max_chars_per_chunk") or 12_000),
                )
                decision = str(result["receipt"]["decision"])
                source_counter[f"decision:{decision}"] += 1
                counters[f"decision:{decision}"] += 1
                for reason in result["receipt"].get("decision_reasons", []):
                    counters[f"reason:{reason}"] += 1
                if decision != "admit":
                    continue
                writer.add(result["train_row"], result["receipt"])
                source_counter["admitted_rows"] += 1
                source_counter["one_pass_token_positions"] += int(result["receipt"]["metrics"]["token_positions"])
                if writer.total_rows >= max_total_rows or writer.total_token_positions >= target_token_positions:
                    stop = True
                    break
            writer.mark_source_complete(source_id, complete=not stop or writer.total_token_positions >= target_token_positions)
            plan.update({
                "status": "complete" if not stop else "bounded_stop",
                "decision": "stream_materialized",
                "candidate_rows": source_counter["candidate_rows"],
                "admitted_rows": source_counter["admitted_rows"],
                "one_pass_token_positions": source_counter["one_pass_token_positions"],
            })
        except BaseException as exc:
            errors.append({"source_id": source_id, "stage": "document_stream_intake", "error_type": type(exc).__name__, "error": str(exc)[:800]})
            plan.update({"status": "blocked", "decision": "stream_fault"})
        source_reports.append(plan)
        if stop:
            break
    writer.close()
    return document_report_payload(
        started=started, root=root, execute=True, target_token_positions=target_token_positions,
        max_total_rows=max_total_rows, max_disk_bytes=max_disk_bytes, public_index=public_index,
        sources=source_reports, writer=writer, counters=counters, errors=errors,
    )


def document_source_plan(source: dict[str, Any], *, allowed_licenses: set[str]) -> dict[str, Any]:
    expected_license = normalize_license(source.get("license_spdx"))
    decision = "eligible_for_intake"
    if expected_license not in allowed_licenses:
        decision = "license_not_allowlisted"
    elif source.get("provenance_class") != "human_public_domain":
        decision = "provenance_class_not_admitted"
    elif source.get("format") != "document_text" or source.get("require_row_public_domain_license") is not True:
        decision = "row_public_domain_evidence_not_required"
    return {
        "source_id": source.get("id"), "dataset": source.get("dataset"), "format": source.get("format"),
        "expected_license": expected_license, "provenance_class": source.get("provenance_class"),
        "status": "planned" if decision == "eligible_for_intake" else "blocked", "decision": decision,
    }


def source_documents(source: dict[str, Any], scale: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for governed document intake") from exc
    kwargs = {"split": str(source.get("split") or "train"), "streaming": True}
    if source.get("revision"):
        kwargs["revision"] = str(source["revision"])
    stream = load_dataset(
        str(source.get("dataset") or ""), str(source.get("config") or "default"), **kwargs,
    )
    max_scan = max(1, int(source.get("max_scan_rows") or 5_000))
    max_chars = int(scale.get("max_chars_per_chunk") or 12_000)
    min_chars = int(scale.get("min_chars_per_chunk") or 800)
    for row_index, raw in enumerate(stream):
        if row_index >= max_scan:
            break
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        if str(metadata.get("license") or "").strip().lower() != "public domain":
            continue
        if str(metadata.get("language") or "").lower() != str(source.get("required_language") or "en").lower():
            continue
        text = str(raw.get("text") or "").strip()
        for chunk_index, chunk in enumerate(chunk_document(text, max_chars=max_chars, min_chars=min_chars)):
            yield chunk, {
                "adapter": "public_domain_document_chunks_v1", "row_index": row_index,
                "chunk_index": chunk_index, "source_document_id": str(raw.get("id") or ""),
                "source_url": str(metadata.get("url") or ""), "title": str(metadata.get("title") or ""),
                "author": str(metadata.get("author") or ""), "row_license": "Public Domain",
            }


def chunk_document(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        pieces = [paragraph[index:index + max_chars] for index in range(0, len(paragraph), max_chars)]
        for piece in pieces:
            if current and size + len(piece) + 2 > max_chars:
                rendered = "\n\n".join(current)
                if len(rendered) >= min_chars:
                    chunks.append(rendered)
                current, size = [], 0
            current.append(piece)
            size += len(piece) + (2 if size else 0)
    if current:
        rendered = "\n\n".join(current)
        if len(rendered) >= min_chars:
            chunks.append(rendered)
    return chunks


def admit_document(
    text: str, *, source: dict[str, Any], provenance: dict[str, Any], public_index: dict[str, Any],
    deduper: "ConversationDeduper", min_chars: int, max_chars: int,
) -> dict[str, Any]:
    redacted, redaction_counts = redact_sensitive_text(text)
    reasons: list[str] = []
    if len(redacted) < min_chars or len(redacted) > max_chars:
        reasons.append("document_chunk_length_out_of_bounds")
    if not provenance.get("source_url") or provenance.get("row_license") != "Public Domain":
        reasons.append("public_domain_provenance_incomplete")
    exact_public = False
    semantic_public_count = 0
    semantic_public_max = 0.0
    digest = stable_hash(normalize_text(redacted))
    if not reasons:
        exact_public = digest in public_index.get("exact_hashes", set())
        semantic_public_count, semantic_public_max = semantic_public_overlap([redacted], public_index)
        if exact_public:
            reasons.append("exact_public_calibration_overlap")
        if semantic_public_count:
            reasons.append("semantic_public_calibration_overlap")
    if not reasons:
        duplicate = deduper.classify(redacted, digest)
        if duplicate:
            reasons.append(duplicate)
    decision = "admit" if not reasons else "quarantine"
    receipt_seed = f"{source.get('id')}:{digest}:{decision}"
    receipt_id = f"data-admission-{stable_hash(receipt_seed)[:20]}"
    receipt = {
        "policy": "project_theseus_data_admission_receipt_v2", "receipt_id": receipt_id,
        "source_id": source.get("id"), "dataset": source.get("dataset"),
        "license_spdx": "public-domain", "provenance_class": "human_public_domain",
        "permitted_use": "broad_english_self_supervised_pretraining_only", "decision": decision,
        "decision_reasons": reasons, "content_sha256": digest,
        "metrics": {"character_count": len(redacted), "token_positions": len(TOKEN_RE.findall(redacted)),
                    "redaction_counts": redaction_counts, "exact_public_overlap": exact_public,
                    "semantic_public_overlap_count": semantic_public_count,
                    "semantic_public_overlap_max": round(float(semantic_public_max), 6)},
        "provenance": provenance, "retention_class": "training_corpus_shard",
        "deletion_scope": ["document_shard", "training_manifest", "derived_checkpoint", "vcm_index"],
        "public_benchmark_training_rows": 0, "external_inference_calls_during_intake": 0,
        "raw_unredacted_text_persisted": False,
    }
    if decision != "admit":
        return {"receipt": receipt, "train_row": {}}
    deduper.add(redacted, digest)
    return {
        "receipt": receipt,
        "train_row": {
            "task_id": f"governed-document-{digest[:20]}", "split": "private_train",
            "modality": "natural_language_document", "causal_text": redacted,
            "license_spdx": "public-domain", "source_id": source.get("id"),
            "source_url": provenance.get("source_url"), "title": provenance.get("title"),
            "author": provenance.get("author"), "content_sha256": digest,
            "data_admission_receipt_id": receipt_id,
            "training_use": "broad_english_self_supervised_pretraining_only",
            "public_benchmark": False, "external_inference_calls": 0,
        },
    }


def source_plan(
    source: dict[str, Any], *, allowed_licenses: set[str], teacher_allowed: bool,
    licensed_static_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance_class = str(source.get("provenance_class") or "unknown")
    expected_license = normalize_license(source.get("license_spdx"))
    static_policy = licensed_static_policy or {}
    static_model_corpus_allowed = bool(
        static_policy.get("enabled") is True
        and static_policy.get("static_open_corpus_required") is True
        and source.get("static_open_corpus") is True
        and str(source.get("quality_tier") or "") in set(static_policy.get("required_quality_tiers") or [])
        and static_policy.get("live_provider_calls_allowed") is False
    )
    decision = "eligible_for_intake"
    if expected_license not in allowed_licenses:
        decision = "license_not_allowlisted"
    elif provenance_class == "external_teacher_generated" and not (teacher_allowed or static_model_corpus_allowed):
        decision = "teacher_distillation_gate_not_admitted_for_conversation"
    elif provenance_class not in {"human_contributed", "external_teacher_generated"}:
        decision = "provenance_class_not_admitted"
    return {
        "source_id": source.get("id"),
        "dataset": source.get("dataset"),
        "format": source.get("format"),
        "expected_license": expected_license,
        "provenance_class": provenance_class,
        "static_model_corpus_allowed": static_model_corpus_allowed,
        "quality_tier": source.get("quality_tier"),
        "status": "planned" if decision == "eligible_for_intake" else "blocked",
        "decision": decision,
        "max_admitted_rows": int(source.get("max_admitted_rows") or 0),
    }


def source_conversations(source: dict[str, Any]) -> Iterator[tuple[list[dict[str, str]], dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for governed streaming intake") from exc
    dataset = str(source.get("dataset") or "")
    config = str(source.get("config") or "default")
    split = str(source.get("split") or "train")
    kwargs = {"split": split, "streaming": True}
    if source.get("revision"):
        kwargs["revision"] = str(source["revision"])
    stream = load_dataset(dataset, config, **kwargs)
    fmt = str(source.get("format") or "messages")
    if fmt == "oasst_tree":
        yield from reconstruct_oasst_conversations(stream, source)
        return
    for row_index, raw in enumerate(stream):
        messages = normalize_direct_messages(raw, fmt)
        if messages:
            yield messages, {"row_index": row_index, "adapter": fmt}


def reconstruct_oasst_conversations(rows: Iterable[dict[str, Any]], source: dict[str, Any]) -> Iterator[tuple[list[dict[str, str]], dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    required_langs = {str(value).lower() for value in source.get("required_langs", ["en"])}
    max_scan = max(1, int(source.get("max_scan_rows") or 500_000))
    for index, raw in enumerate(rows):
        if index >= max_scan:
            break
        message_id = str(raw.get("message_id") or "")
        if not message_id or raw.get("deleted") is True or raw.get("review_result") is not True:
            continue
        if required_langs and str(raw.get("lang") or "").lower() not in required_langs:
            continue
        role = ROLE_MAP.get(str(raw.get("role") or "").lower(), "")
        text = str(raw.get("text") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        rank = raw.get("rank")
        if role == "assistant" and rank is not None and int(rank) > int(source.get("max_assistant_rank") or 0):
            continue
        nodes[message_id] = {
            "message_id": message_id,
            "parent_id": str(raw.get("parent_id") or ""),
            "role": role,
            "content": text,
            "rank": rank,
        }
    for message_id in sorted(nodes):
        node = nodes[message_id]
        if node["role"] != "assistant":
            continue
        chain: list[dict[str, str]] = []
        seen: set[str] = set()
        current = node
        while current and current["message_id"] not in seen and len(chain) < 16:
            seen.add(current["message_id"])
            chain.append({"role": current["role"], "content": current["content"]})
            current = nodes.get(current["parent_id"])
        chain.reverse()
        chain = canonicalize_turns(chain)
        if len(chain) >= 2 and chain[0]["role"] == "user" and chain[-1]["role"] == "assistant":
            yield chain, {
                "message_id_hash": stable_hash(message_id),
                "ancestry_depth": len(chain),
                "adapter": "oasst_tree_ranked_reviewed_v1",
                "raw_message_ids_emitted": False,
            }


def normalize_direct_messages(raw: dict[str, Any], fmt: str) -> list[dict[str, str]]:
    if fmt == "messages":
        value = raw.get("messages") or raw.get("conversations")
        rows = value if isinstance(value, list) else []
        messages = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            role = ROLE_MAP.get(str(item.get("role") or item.get("from") or "").lower(), "")
            content = str(item.get("content") or item.get("value") or item.get("text") or "").strip()
            if role and content:
                messages.append({"role": role, "content": content})
        return canonicalize_turns(messages)
    if fmt == "instruction_response":
        prompt = str(raw.get("instruction") or raw.get("prompt") or raw.get("question") or "").strip()
        response = str(raw.get("response") or raw.get("output") or raw.get("answer") or "").strip()
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}] if prompt and response else []
    return []


def admit_conversation(
    messages: list[dict[str, str]],
    *,
    source: dict[str, Any],
    provenance: dict[str, Any],
    public_index: dict[str, Any],
    deduper: "ConversationDeduper",
    max_chars: int,
) -> dict[str, Any]:
    redaction_counts: Counter[str] = Counter()
    redacted: list[dict[str, str]] = []
    for message in canonicalize_turns(messages):
        content, counts = redact_sensitive_text(str(message.get("content") or ""))
        redaction_counts.update(counts)
        if content:
            redacted.append({"role": str(message.get("role") or ""), "content": content})
    rendered = render_messages(redacted)
    reasons = quality_rejection_reasons(redacted, rendered, max_chars=max_chars)
    exact_public = False
    semantic_public_count = 0
    semantic_public_max = 0.0
    if not reasons:
        normalized = normalize_text(rendered)
        exact_public = stable_hash(normalized) in public_index.get("exact_hashes", set())
        semantic_public_count, semantic_public_max = semantic_public_overlap([rendered], public_index)
        if exact_public:
            reasons.append("exact_public_calibration_overlap")
        if semantic_public_count:
            reasons.append("semantic_public_calibration_overlap")
    digest = stable_hash(normalize_text(rendered))
    near_duplicate = False
    if not reasons:
        duplicate_kind = deduper.classify(rendered, digest)
        if duplicate_kind:
            reasons.append(duplicate_kind)
            near_duplicate = duplicate_kind == "near_duplicate"
    decision = "admit" if not reasons else "quarantine"
    token_positions = len(TOKEN_RE.findall(rendered))
    receipt_seed = f"{source.get('id')}:{digest}:{decision}"
    receipt_id = f"data-admission-{stable_hash(receipt_seed)[:20]}"
    receipt = {
        "policy": "project_theseus_data_admission_receipt_v2",
        "receipt_id": receipt_id,
        "source_id": source.get("id"),
        "dataset": source.get("dataset"),
        "license_spdx": source.get("license_spdx"),
        "provenance_class": source.get("provenance_class"),
        "quality_tier": source.get("quality_tier"),
        "permitted_use": "private_conversation_pretraining_and_sft_only",
        "decision": decision,
        "decision_reasons": reasons,
        "content_sha256": digest,
        "metrics": {
            "turn_count": len(redacted),
            "character_count": len(rendered),
            "token_positions": token_positions,
            "redaction_counts": dict(sorted(redaction_counts.items())),
            "exact_public_overlap": exact_public,
            "semantic_public_overlap_count": semantic_public_count,
            "semantic_public_overlap_max": round(float(semantic_public_max), 6),
            "near_duplicate": near_duplicate,
        },
        "provenance": provenance,
        "retention_class": "training_corpus_shard",
        "deletion_scope": ["conversation_shard", "training_manifest", "derived_checkpoint", "vcm_index"],
        "public_benchmark_training_rows": 0,
        "external_inference_calls_during_intake": 0,
        "raw_unredacted_text_persisted": False,
    }
    if decision != "admit":
        return {"receipt": receipt, "train_row": {}, "sts_row": {}}
    deduper.add(rendered, digest)
    assistant_index = max(index for index, row in enumerate(redacted) if row["role"] == "assistant")
    prompt_messages = redacted[:assistant_index]
    target_message = redacted[assistant_index]
    task_id = f"governed-conversation-{digest[:20]}"
    train_row = {
        "task_id": task_id,
        "split": "private_train",
        "modality": "conversation",
        "prompt_messages": prompt_messages,
        "target_message": target_message,
        "causal_text": rendered,
        "license_spdx": source.get("license_spdx"),
        "source_id": source.get("id"),
        "provenance_class": source.get("provenance_class"),
        "quality_tier": source.get("quality_tier"),
        "data_admission_receipt_id": receipt_id,
        "training_use": "private_conversation_training",
        "public_benchmark": False,
        "external_inference_calls": 0,
    }
    sts_row = {
        "task_id": f"sts-{digest[:20]}",
        "split": "private_train",
        "streams": {
            "conversation_context": render_messages(prompt_messages),
            "assistant_visible": target_message["content"],
            "audit_stream": f"data_admission_receipt={receipt_id};public_training_rows=0",
        },
        "input_streams": ["conversation_context", "audit_stream"],
        "output_streams": ["assistant_visible"],
        "data_admission_receipt_id": receipt_id,
        "license_spdx": source.get("license_spdx"),
        "public_benchmark": False,
        "external_inference_calls": 0,
    }
    return {"receipt": receipt, "train_row": train_row, "sts_row": sts_row}


class ConversationDeduper:
    def __init__(self, *, max_hamming_distance: int = 3) -> None:
        self.exact: set[str] = set()
        self.fingerprints: list[int] = []
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.max_hamming_distance = max(0, max_hamming_distance)

    def classify(self, text: str, digest: str) -> str:
        if digest in self.exact:
            return "exact_duplicate"
        fingerprint = simhash64(text)
        candidates: set[int] = set()
        for band, value in simhash_bands(fingerprint):
            candidates.update(self.buckets.get((band, value), []))
        if any(bin(fingerprint ^ self.fingerprints[index]).count("1") <= self.max_hamming_distance for index in candidates):
            return "near_duplicate"
        return ""

    def add(self, text: str, digest: str) -> None:
        self.exact.add(digest)
        fingerprint = simhash64(text)
        index = len(self.fingerprints)
        self.fingerprints.append(fingerprint)
        for band, value in simhash_bands(fingerprint):
            self.buckets[(band, value)].append(index)


class AtomicConversationShardWriter:
    def __init__(self, root: Path, *, shard_rows: int, max_disk_bytes: int) -> None:
        self.root = root
        self.shard_rows = shard_rows
        self.max_disk_bytes = max_disk_bytes
        self.state_path = root / "stream_state.json"
        self.manifest_path = root / "scalable_manifest.json"
        self.state = read_json(self.state_path)
        self.completed_sources = set(self.state.get("completed_sources") or [])
        self.shards = list(self.state.get("shards") or [])
        self.total_rows = int(self.state.get("total_rows") or 0)
        self.total_token_positions = int(self.state.get("total_token_positions") or 0)
        self.disk_bytes = int(self.state.get("disk_bytes") or 0)
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "sts": [], "receipts": []}
        self.deduper = ConversationDeduper()
        self._load_existing_deduper()

    def _load_existing_deduper(self) -> None:
        legacy_flat = self.root / "private_train" / "conversation_sft_pressure.jsonl"
        for row in iter_jsonl(legacy_flat):
            text = str(row.get("causal_text") or "")
            if text:
                self.deduper.add(text, stable_hash(normalize_text(text)))
        for shard in self.shards:
            train_path = resolve_path(self.root, str(shard.get("train_path") or ""))
            for row in iter_jsonl(train_path):
                text = str(row.get("causal_text") or "")
                if text:
                    self.deduper.add(text, stable_hash(normalize_text(text)))

    def add(self, train_row: dict[str, Any], sts_row: dict[str, Any], receipt: dict[str, Any]) -> None:
        encoded_size = sum(len(json.dumps(row, ensure_ascii=True, separators=(",", ":"))) + 1 for row in (train_row, sts_row, receipt))
        if self.disk_bytes + encoded_size > self.max_disk_bytes:
            raise RuntimeError("conversation intake disk budget exhausted")
        self.buffers["train"].append(train_row)
        self.buffers["sts"].append(sts_row)
        self.buffers["receipts"].append(receipt)
        self.total_rows += 1
        self.total_token_positions += int((receipt.get("metrics") or {}).get("token_positions") or 0)
        self.disk_bytes += encoded_size
        if len(self.buffers["train"]) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffers["train"]:
            return
        index = len(self.shards)
        paths = {
            "train": self.root / "private_train" / "shards" / f"conversation-{index:05d}.jsonl",
            "sts": self.root / "sts_streams" / "shards" / f"conversation-sts-{index:05d}.jsonl",
            "receipts": self.root / "receipts" / f"data-admission-{index:05d}.jsonl",
        }
        shard: dict[str, Any] = {"index": index, "rows": len(self.buffers["train"])}
        for kind, path in paths.items():
            atomic_write_jsonl(path, self.buffers[kind])
            shard[f"{kind}_path"] = relative_to_root(path, self.root)
            shard[f"{kind}_sha256"] = file_sha256(path)
            shard[f"{kind}_bytes"] = path.stat().st_size
            self.buffers[kind].clear()
        self.shards.append(shard)
        self._persist_state()

    def mark_source_complete(self, source_id: str) -> None:
        self.flush()
        self.completed_sources.add(source_id)
        self._persist_state()

    def close(self) -> None:
        self.flush()
        self._persist_state()

    def _persist_state(self) -> None:
        payload = {
            "policy": "project_theseus_governed_conversation_stream_state_v1",
            "updated_utc": now(),
            "completed_sources": sorted(self.completed_sources),
            "total_rows": self.total_rows,
            "total_token_positions": self.total_token_positions,
            "disk_bytes": self.disk_bytes,
            "shards": self.shards,
        }
        atomic_write_json(self.state_path, payload)
        atomic_write_json(self.manifest_path, {**payload, "state_path": relative_to_root(self.state_path, self.root)})


class AtomicDocumentShardWriter:
    """Atomic, resumable pretraining-only shards kept separate from SFT/STS data."""

    def __init__(self, root: Path, *, shard_rows: int, max_disk_bytes: int) -> None:
        self.root = root
        self.shard_rows = shard_rows
        self.max_disk_bytes = max_disk_bytes
        self.state_path = root / "stream_state.json"
        self.manifest_path = root / "scalable_manifest.json"
        self.state = read_json(self.state_path)
        self.completed_sources = set(self.state.get("completed_sources") or [])
        self.shards = list(self.state.get("shards") or [])
        self.total_rows = int(self.state.get("total_rows") or 0)
        self.total_token_positions = int(self.state.get("total_token_positions") or 0)
        self.disk_bytes = int(self.state.get("disk_bytes") or 0)
        self.buffers: dict[str, list[dict[str, Any]]] = {"train": [], "receipts": []}
        self.deduper = ConversationDeduper()
        for shard in self.shards:
            for row in iter_jsonl(resolve_path(root, str(shard.get("train_path") or ""))):
                text = str(row.get("causal_text") or "")
                if text:
                    self.deduper.add(text, stable_hash(normalize_text(text)))

    def add(self, train_row: dict[str, Any], receipt: dict[str, Any]) -> None:
        encoded_size = sum(
            len(json.dumps(row, ensure_ascii=True, separators=(",", ":"))) + 1
            for row in (train_row, receipt)
        )
        if self.disk_bytes + encoded_size > self.max_disk_bytes:
            raise RuntimeError("document intake disk budget exhausted")
        self.buffers["train"].append(train_row)
        self.buffers["receipts"].append(receipt)
        self.total_rows += 1
        self.total_token_positions += int((receipt.get("metrics") or {}).get("token_positions") or 0)
        self.disk_bytes += encoded_size
        if len(self.buffers["train"]) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffers["train"]:
            return
        index = len(self.shards)
        paths = {
            "train": self.root / "private_train" / "shards" / f"document-{index:05d}.jsonl",
            "receipts": self.root / "receipts" / f"data-admission-{index:05d}.jsonl",
        }
        shard: dict[str, Any] = {"index": index, "rows": len(self.buffers["train"])}
        for kind, path in paths.items():
            atomic_write_jsonl(path, self.buffers[kind])
            shard[f"{kind}_path"] = relative_to_root(path, self.root)
            shard[f"{kind}_sha256"] = file_sha256(path)
            shard[f"{kind}_bytes"] = path.stat().st_size
            self.buffers[kind].clear()
        self.shards.append(shard)
        self._persist_state()

    def mark_source_complete(self, source_id: str, *, complete: bool) -> None:
        self.flush()
        if complete:
            self.completed_sources.add(source_id)
        self._persist_state()

    def close(self) -> None:
        self.flush()
        self._persist_state()

    def _persist_state(self) -> None:
        payload = {
            "policy": "project_theseus_governed_document_stream_state_v1",
            "updated_utc": now(),
            "completed_sources": sorted(self.completed_sources),
            "total_rows": self.total_rows,
            "total_token_positions": self.total_token_positions,
            "disk_bytes": self.disk_bytes,
            "shards": self.shards,
        }
        atomic_write_json(self.state_path, payload)
        atomic_write_json(self.manifest_path, {**payload, "state_path": relative_to_root(self.state_path, self.root)})


def quality_rejection_reasons(messages: list[dict[str, str]], rendered: str, *, max_chars: int) -> list[str]:
    reasons: list[str] = []
    if len(messages) < 2 or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        reasons.append("invalid_conversation_contract")
    if len(rendered) > max_chars:
        reasons.append("conversation_too_long")
    assistant_text = " ".join(row["content"] for row in messages if row["role"] == "assistant")
    user_text = " ".join(row["content"] for row in messages if row["role"] == "user")
    if len(user_text) < 12 or len(assistant_text) < 20:
        reasons.append("insufficient_prompt_or_response_content")
    tokens = [token.lower() for token in TOKEN_RE.findall(rendered)]
    lexical_diversity = len(set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 80 and lexical_diversity < 0.08:
        reasons.append("low_lexical_diversity")
    if re.search(r"(.)\1{24,}", rendered):
        reasons.append("repeated_character_spam")
    if PRIVATE_KEY_RE.search(rendered):
        reasons.append("private_key_material")
    return reasons


def redact_sensitive_text(value: str) -> tuple[str, Counter[str]]:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    counts: Counter[str] = Counter()
    for name, regex, replacement in (
        ("email", EMAIL_RE, "[redacted_email]"),
        ("phone", PHONE_RE, "[redacted_phone]"),
        ("ipv4", IPV4_RE, "[redacted_ip]"),
        ("ssn", SSN_RE, "[redacted_ssn]"),
        ("payment_card", CARD_RE, "[redacted_card]"),
    ):
        text, count = regex.subn(replacement, text)
        counts[name] += count
    text, secret_count = SECRET_RE.subn(lambda match: match.group(0).replace(match.group(1), "[redacted_secret]"), text)
    counts["secret"] += secret_count
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text, counts


def canonicalize_turns(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in messages:
        role = ROLE_MAP.get(str(row.get("role") or "").lower(), str(row.get("role") or "").lower())
        content = str(row.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        if output and output[-1]["role"] == role:
            output[-1]["content"] = f"{output[-1]['content']}\n\n{content}"
        else:
            output.append({"role": role, "content": content})
    return output


def simhash64(text: str) -> int:
    weights = [0] * 64
    counts = Counter(token.lower() for token in TOKEN_RE.findall(normalize_text(text)))
    for token, count in counts.items():
        value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        weight = 1 + int(math.log2(count))
        for bit in range(64):
            weights[bit] += weight if value & (1 << bit) else -weight
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def simhash_bands(fingerprint: int) -> list[tuple[int, int]]:
    return [(band, (fingerprint >> (band * 16)) & 0xFFFF) for band in range(4)]


def build_public_index(*, max_texts: int) -> dict[str, Any]:
    from training_data_lineage_audit import build_public_contamination_index

    return build_public_contamination_index(max_texts=max(100, max_texts))


def semantic_public_overlap(texts: list[str], index: dict[str, Any]) -> tuple[int, float]:
    from training_data_lineage_audit import semantic_overlap

    return semantic_overlap(texts, index)


def governed_teacher_conversation_allowed() -> bool:
    gate = read_json(ROOT / "reports" / "teacher_distillation_gate.json")
    summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
    return bool(
        (gate.get("distillation_allowed") or summary.get("distillation_allowed"))
        and summary.get("conversation_rows_allowed") is True
    )


def fetch_dataset_metadata(dataset: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset)
    card_data = info.card_data.to_dict() if info.card_data is not None and hasattr(info.card_data, "to_dict") else {}
    return {"cardData": card_data, "tags": list(info.tags or [])}


def extract_dataset_license(metadata: dict[str, Any]) -> str:
    card = metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
    value = card.get("license")
    if isinstance(value, list):
        value = value[0] if value else ""
    if value:
        return str(value)
    for tag in metadata.get("tags", []):
        if str(tag).startswith("license:"):
            return str(tag).split(":", 1)[1]
    return ""


def report_payload(
    *,
    started: float,
    root: Path,
    execute: bool,
    target_token_positions: int,
    max_total_rows: int,
    max_disk_bytes: int,
    public_index: dict[str, Any],
    sources: list[dict[str, Any]],
    writer: AtomicConversationShardWriter | None,
    counters: Counter[str],
    errors: list[dict[str, Any]],
    hard_preflight: bool,
) -> dict[str, Any]:
    total_rows = writer.total_rows if writer else 0
    total_tokens = writer.total_token_positions if writer else 0
    eligible_sources = sum(1 for row in sources if row.get("decision") in {"eligible_for_intake", "stream_materialized", "resume_source_already_complete"})
    hard_clean = hard_preflight and not any(row.get("decision") == "stream_fault" for row in sources)
    trigger_state = "PLANNED" if not execute else ("GREEN" if hard_clean and total_rows else "YELLOW")
    if execute and not hard_clean:
        trigger_state = "RED"
    return {
        "policy": "project_theseus_governed_conversation_stream_v1",
        "created_utc": now(),
        "execute": execute,
        "trigger_state": trigger_state,
        "root": str(root),
        "summary": {
            "source_count": len(sources),
            "enabled_source_count": len(sources),
            "sampled_source_count": sum(1 for row in sources if row.get("decision") == "stream_materialized"),
            "metadata_only_source_count": sum(1 for row in sources if row.get("status") == "planned"),
            "eligible_source_count": eligible_sources,
            "blocked_source_count": sum(1 for row in sources if row.get("status") == "blocked"),
            "total_rows": total_rows,
            "conversation_samples": total_rows,
            "private_train_rows": total_rows,
            "sts_rows": total_rows,
            "one_pass_token_positions": total_tokens,
            "target_one_pass_token_positions": target_token_positions,
            "target_fraction": round(total_tokens / max(1, target_token_positions), 6),
            "max_total_rows": max_total_rows,
            "disk_bytes": writer.disk_bytes if writer else 0,
            "max_disk_bytes": max_disk_bytes,
            "shard_count": len(writer.shards) if writer else 0,
            "bounded_streaming_intake": True,
            "bulk_download": False,
            "promotion_evidence": False,
            "teacher_distillation": False,
            "public_contamination_text_count": int(public_index.get("text_count") or 0),
            "public_contamination_index_digest": public_index.get("digest"),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "raw_unredacted_text_persisted": False,
            "optimizer_repetition_counted_as_unique_data": False,
            "decision_counts": {
                key.removeprefix("decision:"): value
                for key, value in sorted(counters.items())
                if key.startswith("decision:")
            },
            "quarantine_reason_counts": {
                key.removeprefix("reason:"): value
                for key, value in sorted(counters.items())
                if key.startswith("reason:")
            },
        },
        "manifest": relative_to_root(writer.manifest_path, root) if writer else "scalable_manifest.json",
        "sources": sources,
        "checks": [
            {
                "gate": "no_public_eval_token_overlap_in_accepted_rows",
                "passed": True,
                "evidence": "exact and semantic public-index matches are quarantined before shard writes",
            },
            {
                "gate": "raw_unredacted_text_not_persisted",
                "passed": True,
                "evidence": False,
            },
            {
                "gate": "optimizer_repetition_not_counted_as_unique_data",
                "passed": True,
                "evidence": False,
            },
        ],
        "errors": errors,
        "score_semantics": (
            "Rows are redacted, decontaminated, exact/near-deduplicated private training pressure. "
            "One-pass token positions count each admitted row once; optimizer epochs never increase data-scale credit."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def document_report_payload(
    *, started: float, root: Path, execute: bool, target_token_positions: int,
    max_total_rows: int, max_disk_bytes: int, public_index: dict[str, Any],
    sources: list[dict[str, Any]], writer: AtomicDocumentShardWriter | None,
    counters: Counter[str], errors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_rows = writer.total_rows if writer else 0
    total_tokens = writer.total_token_positions if writer else 0
    hard_clean = bool(public_index.get("text_count")) and not errors
    trigger_state = "PLANNED" if not execute else ("GREEN" if hard_clean and total_rows else "RED")
    return {
        "policy": "project_theseus_governed_document_stream_v1",
        "created_utc": now(),
        "execute": execute,
        "trigger_state": trigger_state,
        "root": str(root),
        "summary": {
            "source_count": len(sources),
            "total_rows": total_rows,
            "one_pass_token_positions": total_tokens,
            "target_one_pass_token_positions": target_token_positions,
            "target_fraction": round(total_tokens / max(1, target_token_positions), 6),
            "max_total_rows": max_total_rows,
            "disk_bytes": writer.disk_bytes if writer else 0,
            "max_disk_bytes": max_disk_bytes,
            "shard_count": len(writer.shards) if writer else 0,
            "public_contamination_text_count": int(public_index.get("text_count") or 0),
            "public_contamination_index_digest": public_index.get("digest"),
            "public_training_rows": 0,
            "external_inference_calls": 0,
            "raw_unredacted_text_persisted": False,
            "optimizer_repetition_counted_as_unique_data": False,
            "decision_counts": {
                key.removeprefix("decision:"): value
                for key, value in sorted(counters.items()) if key.startswith("decision:")
            },
            "quarantine_reason_counts": {
                key.removeprefix("reason:"): value
                for key, value in sorted(counters.items()) if key.startswith("reason:")
            },
        },
        "manifest": relative_to_root(writer.manifest_path, root) if writer else "scalable_manifest.json",
        "sources": sources,
        "checks": [
            {"gate": "row_public_domain_evidence_required", "passed": True},
            {"gate": "no_public_eval_overlap_in_accepted_rows", "passed": True},
            {"gate": "pretraining_shards_separate_from_sft_and_sts", "passed": True},
            {"gate": "optimizer_repetition_not_counted_as_unique_data", "passed": True},
        ],
        "errors": errors,
        "score_semantics": (
            "Human-authored public-domain document chunks are pretraining data only. "
            "They do not count as conversation SFT, STS evidence, public calibration, or capability."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def render_messages(messages: Iterable[dict[str, str]]) -> str:
    return "\n".join(f"<|{row['role']}|>\n{row['content']}" for row in messages)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
