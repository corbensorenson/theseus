"""Governed open conversational data pantry for Theseus training pressure.

This is deliberately a pressure-data intake lane, not a benchmark or promotion
lane. It fetches only tiny allowlisted Hugging Face dataset slices, stores them
under an approved local training-data root, writes source cards and provenance,
and emits train/STS rows that remain private-training support only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "open_conversation_training_pantry.json"
DEFAULT_OUT = ROOT / "reports" / "open_conversation_training_pantry.json"
DEFAULT_MARKDOWN_OUT = ROOT / "reports" / "open_conversation_training_pantry.md"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"
HF_API_DATASET_URL = "https://huggingface.co/api/datasets"
ROLE_MAP = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "prompter": "user",
    "human": "user",
    "user": "user",
    "system": "system",
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--root", default="")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-chars-per-conversation", type=int, default=0)
    parser.add_argument(
        "--scalable",
        action="store_true",
        help="Use the resumable receipt-bound streaming intake instead of legacy bounded row samples.",
    )
    parser.add_argument(
        "--scalable-documents",
        action="store_true",
        help="Materialize governed human-authored broad-English pretraining shards.",
    )
    parser.add_argument("--target-one-pass-token-positions", type=int, default=0)
    parser.add_argument("--max-total-rows", type=int, default=0)
    parser.add_argument("--shard-rows", type=int, default=0)
    parser.add_argument("--max-disk-bytes", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    config = read_json(resolve(args.config))
    if args.scalable or args.scalable_documents:
        from governed_conversation_stream import run_document_intake, run_streaming_intake

        root = Path(args.root or config.get("root") or "data/training_data/open_conversation_pantry")
        if not root.is_absolute():
            root = ROOT / root
        intake = run_document_intake if args.scalable_documents else run_streaming_intake
        intake_root = root / "broad_text" if args.scalable_documents else root
        report = intake(
            config,
            root=intake_root,
            execute=bool(args.allow_network_fetch),
            target_token_positions=max(0, int(args.target_one_pass_token_positions or 0)),
            max_total_rows=max(0, int(args.max_total_rows or 0)),
            shard_rows=max(0, int(args.shard_rows or 0)),
            max_disk_bytes=max(0, int(args.max_disk_bytes or 0)),
        )
        report["pantry_policy"] = "project_theseus_open_conversation_training_pantry_v2"
    else:
        report = build_pantry(config, args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW", "PLANNED"} else 1


def build_pantry(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(args.root or config.get("root") or "D:/ProjectTheseus/training_data/open_conversation_pantry")
    root_policy = approved_training_root(root)
    max_rows_per_source = args.max_rows_per_source or int(config.get("default_max_rows_per_source") or 24)
    max_chars = args.max_chars_per_conversation or int(config.get("default_max_chars_per_conversation") or 8000)
    allowed_licenses = {normal_license(item) for item in config.get("allowed_licenses", [])}
    exclusion_tokens = {str(item).lower() for item in config.get("public_eval_exclusion_tokens", [])}
    sample_path = root / "samples" / "open_conversation_samples.jsonl"
    train_path = root / "private_train" / "conversation_sft_pressure.jsonl"
    sts_path = root / "sts_streams" / "conversation_sts_streams.jsonl"
    card_dir = root / "dataset_cards"

    checks: list[dict[str, Any]] = [
        check("root_is_approved_training_data_root", root_policy["approved"], root_policy["evidence"]),
        check("bulk_downloads_disabled", not bool(get_path(config, ["safety", "bulk_downloads"], False)), "tiny rows only"),
        check("candidate_promotion_evidence_disabled", not bool(get_path(config, ["safety", "candidate_promotion_evidence"], True)), "training pressure only"),
    ]
    if not root_policy["approved"]:
        return {
            "policy": "project_theseus_open_conversation_training_pantry_v1",
            "created_utc": now(),
            "trigger_state": "RED",
            "root": normalize_path(root),
            "summary": {"private_train_rows": 0, "conversation_samples": 0, "sts_rows": 0},
            "checks": checks,
            "errors": [{"stage": "root_validation", "error": root_policy["error"]}],
            "external_inference_calls": 0,
        }

    root.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    sts_path.parent.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    sts_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for source in expand_sources(config.get("sources", [])):
        if not isinstance(source, dict):
            continue
        row = process_source(
            source,
            allowed_licenses=allowed_licenses,
            exclusion_tokens=exclusion_tokens,
            max_rows_per_source=max_rows_per_source,
            max_chars=max_chars,
            network_allowed=bool(args.allow_network_fetch),
        )
        sources.append(row["source_report"])
        errors.extend(row["errors"])
        samples.extend(row["samples"])
        train_rows.extend(row["train_rows"])
        sts_rows.extend(row["sts_rows"])
        write_json(card_dir / f"{safe_name(source.get('id') or source.get('dataset') or 'source')}.json", row["source_report"])

    previous_counts = {
        "conversation_samples": 0,
        "private_train_rows": 0,
        "sts_rows": 0,
    }
    if args.refresh:
        previous_samples = filter_retained_rows(
            read_jsonl(sample_path),
            allowed_licenses=allowed_licenses,
            exclusion_tokens=exclusion_tokens,
            row_kind="sample",
        )
        previous_train_rows = filter_retained_rows(
            read_jsonl(train_path),
            allowed_licenses=allowed_licenses,
            exclusion_tokens=exclusion_tokens,
            row_kind="train",
        )
        previous_sts_rows = filter_retained_rows(
            read_jsonl(sts_path),
            allowed_licenses=allowed_licenses,
            exclusion_tokens=exclusion_tokens,
            row_kind="sts",
        )
        previous_counts = {
            "conversation_samples": len(previous_samples),
            "private_train_rows": len(previous_train_rows),
            "sts_rows": len(previous_sts_rows),
        }
        samples = merge_json_rows(previous_samples, samples, key_fields=("record_id", "text_sha256"))
        train_rows = merge_json_rows(previous_train_rows, train_rows, key_fields=("task_id", "source_record_id"))
        sts_rows = merge_json_rows(previous_sts_rows, sts_rows, key_fields=("task_id", "source_record_id"))

    if samples or args.refresh or not sample_path.exists():
        write_jsonl(sample_path, samples)
    if train_rows or args.refresh or not train_path.exists():
        write_jsonl(train_path, train_rows)
    if sts_rows or args.refresh or not sts_path.exists():
        write_jsonl(sts_path, sts_rows)

    manifest = {
        "policy": "project_theseus_open_conversation_pantry_manifest_v1",
        "created_utc": now(),
        "root": normalize_path(root),
        "sample_jsonl": normalize_path(sample_path),
        "private_train_jsonl": normalize_path(train_path),
        "sts_stream_jsonl": normalize_path(sts_path),
        "source_cards": normalize_path(card_dir),
        "source_count": len(sources),
        "conversation_samples": len(samples),
        "private_train_rows": len(train_rows),
        "sts_rows": len(sts_rows),
        "previous_counts_on_refresh": previous_counts,
        "bulk_download": False,
        "external_inference_calls": 0,
    }
    write_json(root / "manifest.json", manifest)

    enabled_sources = [item for item in sources if item.get("enabled")]
    admitted_sources = [item for item in sources if item.get("status") in {"sampled", "metadata_verified"}]
    trigger_state = "GREEN" if train_rows and sts_rows else "YELLOW"
    if enabled_sources and not admitted_sources:
        trigger_state = "RED"
    checks.extend(
        [
            check("license_allowlist_present", bool(allowed_licenses), ",".join(sorted(allowed_licenses))),
            check("enabled_sources_checked", bool(enabled_sources), f"enabled={len(enabled_sources)}"),
            check("no_public_eval_token_overlap_in_accepted_rows", True, "rows containing public code eval tokens are rejected"),
            check("refresh_preserves_existing_admitted_rows", True, json.dumps(previous_counts, sort_keys=True)),
            check("private_train_rows_present", bool(train_rows), f"rows={len(train_rows)}"),
            check("sts_rows_present", bool(sts_rows), f"rows={len(sts_rows)}"),
        ]
    )
    return {
        "policy": "project_theseus_open_conversation_training_pantry_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "root": normalize_path(root),
        "sample_jsonl": normalize_path(sample_path),
        "private_train_jsonl": normalize_path(train_path),
        "sts_stream_jsonl": normalize_path(sts_path),
        "manifest": normalize_path(root / "manifest.json"),
        "summary": {
            "source_count": len(sources),
            "enabled_source_count": len(enabled_sources),
            "sampled_source_count": len([item for item in sources if item.get("status") == "sampled"]),
            "metadata_only_source_count": len([item for item in sources if item.get("status") == "metadata_only"]),
            "blocked_source_count": len([item for item in sources if item.get("status") == "blocked"]),
            "conversation_samples": len(samples),
            "private_train_rows": len(train_rows),
            "sts_rows": len(sts_rows),
            "previous_counts_on_refresh": previous_counts,
            "refresh_merge_policy": "preserve_prior_allowlisted_rows_when_current_fetch_is_rate_limited",
            "allowed_licenses": sorted(allowed_licenses),
            "bulk_download": False,
            "promotion_evidence": False,
            "public_benchmark_solutions_included": False,
            "teacher_distillation": False,
            "external_inference_calls": 0,
        },
        "sources": sources,
        "checks": checks,
        "errors": errors,
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def process_source(
    source: dict[str, Any],
    *,
    allowed_licenses: set[str],
    exclusion_tokens: set[str],
    max_rows_per_source: int,
    max_chars: int,
    network_allowed: bool,
) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("dataset") or "unknown_source")
    expected_license = normal_license(source.get("license_spdx"))
    enabled = bool(source.get("enabled", False))
    checks = [
        check("source_enabled", enabled, str(source.get("enabled", False))),
        check("source_kind_huggingface_dataset", source.get("source_kind") == "huggingface_dataset", str(source.get("source_kind"))),
        check("license_expected_allowlisted", expected_license in allowed_licenses, str(source.get("license_spdx"))),
    ]
    errors: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {
        "source_id": source_id,
        "dataset": source.get("dataset"),
        "url": source.get("url"),
        "license_spdx": source.get("license_spdx"),
        "expected_license_normalized": expected_license,
        "enabled": enabled,
        "format": source.get("format"),
        "config": source.get("config"),
        "split": source.get("split"),
        "offset": source.get("offset", 0),
        "length": source.get("length", 0),
        "why": source.get("why"),
        "use": source.get("use"),
        "checks": checks,
        "status": "metadata_only",
        "conversation_samples": 0,
        "private_train_rows": 0,
        "sts_rows": 0,
        "external_inference_calls": 0,
    }

    if not enabled:
        source_report["status"] = "metadata_only"
        source_report["decision"] = "disabled_by_policy"
        return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": []}
    if expected_license not in allowed_licenses:
        source_report["status"] = "blocked"
        source_report["decision"] = "license_not_allowlisted"
        return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": []}

    dataset = str(source.get("dataset") or "")
    try:
        metadata = fetch_hf_dataset_metadata(dataset) if network_allowed else {}
    except Exception as exc:  # noqa: BLE001
        metadata = {}
        errors.append({"source_id": source_id, "stage": "metadata_fetch", "error": str(exc)[:500]})
    actual_license = normal_license(extract_hf_license(metadata)) if metadata else ""
    if actual_license:
        checks.append(check("hf_license_matches_expected", actual_license == expected_license, actual_license))
        source_report["hf_license_normalized"] = actual_license
        if actual_license != expected_license:
            source_report["status"] = "blocked"
            source_report["decision"] = "hf_license_mismatch"
            return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": errors}
    else:
        checks.append(check("hf_license_verified", not network_allowed, "metadata unavailable; expected license only"))

    if not network_allowed:
        checks.append(check("network_fetch_allowed", False, "run with --allow-network-fetch for tiny samples"))
        source_report["status"] = "metadata_verified"
        source_report["decision"] = "metadata_only_network_disabled"
        return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": errors}

    length = int(source.get("length") or max_rows_per_source)
    if length <= 0:
        source_report["status"] = "metadata_only"
        source_report["decision"] = "length_zero"
        return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": errors}
    length = max(1, min(length, max_rows_per_source, 100))
    config = str(source.get("config") or "default")
    split = str(source.get("split") or "train")
    offset = int(source.get("offset") or 0)
    try:
        payload = fetch_hf_rows(dataset=dataset, config=config, split=split, offset=offset, length=length)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)[:500]
        checks.append(check("hf_rows_fetch", False, detail))
        if "HTTP 429" in detail:
            source_report["status"] = "metadata_verified"
            source_report["decision"] = "rows_fetch_rate_limited_deferred"
            source_report["checks"] = checks
            return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": errors}
        errors.append({"source_id": source_id, "stage": "rows_fetch", "error": detail})
        source_report["status"] = "blocked"
        source_report["decision"] = "rows_fetch_failed"
        return {"source_report": source_report, "samples": [], "train_rows": [], "sts_rows": [], "errors": errors}

    samples: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    sts_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("rows", []):
        raw = item.get("row") if isinstance(item, dict) else {}
        if not isinstance(raw, dict):
            continue
        if bool(raw.get("deleted", False)):
            continue
        if source.get("require_review_result") and raw.get("review_result") is not True:
            continue
        if not language_allowed(raw, source.get("required_langs")):
            continue
        messages = normalize_messages(raw, str(source.get("format") or "messages"), max_chars=max_chars)
        if not messages:
            continue
        rendered = render_messages(messages)
        if contains_exclusion_token(rendered, exclusion_tokens):
            continue
        content_hash = stable_hash(rendered)
        if content_hash in seen:
            continue
        seen.add(content_hash)
        sample = {
            "record_id": f"open_conv_{source_id}_{content_hash[:16]}",
            "source_id": source_id,
            "dataset": dataset,
            "source_url": source.get("url"),
            "license_spdx": source.get("license_spdx"),
            "dataset_config": config,
            "dataset_split": split,
            "row_idx": item.get("row_idx"),
            "messages": messages,
            "turn_count": len(messages),
            "char_count": len(rendered),
            "text_sha256": content_hash,
            "fetched_utc": now(),
            "governance": {
                "training_use": "allowed_private_conversation_pressure_only",
                "bulk_download": False,
                "promotion_evidence": False,
                "public_benchmark_solution": False,
                "external_inference_calls": 0,
            },
            "provenance": {
                "fetch_api": "huggingface_datasets_server_rows",
                "fetch_url": hf_rows_url(dataset, config, split, offset, length),
                "row_fields": sorted(raw.keys()),
            },
        }
        samples.append(sample)
        train = build_train_row(sample)
        if train:
            train_rows.append(train)
        sts = build_sts_row(sample)
        if sts:
            sts_rows.append(sts)

    requires_train_rows = str(source.get("format") or "") != "oasst_message_rows"
    checks.extend(
        [
            check("sample_rows_present", bool(samples), f"samples={len(samples)}"),
            check("bounded_sample_rows", len(samples) <= length, f"samples={len(samples)} length={length}"),
            check("train_rows_present_or_not_required", (not requires_train_rows) or bool(train_rows), f"train={len(train_rows)}"),
            check("sts_rows_present_or_not_required", (not requires_train_rows) or bool(sts_rows), f"sts={len(sts_rows)}"),
        ]
    )
    source_report.update(
        {
            "status": "sampled" if samples else "metadata_verified",
            "decision": "tiny_sample_materialized" if samples else "no_usable_rows_after_filters",
            "conversation_samples": len(samples),
            "private_train_rows": len(train_rows),
            "sts_rows": len(sts_rows),
            "num_rows_total": payload.get("num_rows_total"),
            "checks": checks,
        }
    )
    return {"source_report": source_report, "samples": samples, "train_rows": train_rows, "sts_rows": sts_rows, "errors": errors}


def normalize_messages(raw: dict[str, Any], fmt: str, *, max_chars: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if fmt == "messages":
        value = raw.get("messages") or raw.get("conversations")
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                role = normalize_role(item.get("role") or item.get("from") or "")
                content = clean_text(str(item.get("content") or item.get("value") or item.get("text") or ""), max_chars)
                if role and content:
                    messages.append({"role": role, "content": content})
    elif fmt == "instruction_response":
        instruction = clean_text(str(raw.get("instruction") or raw.get("prompt") or raw.get("question") or raw.get("query") or ""), max_chars)
        context = clean_text(str(raw.get("context") or raw.get("system_prompt") or ""), max_chars)
        response = clean_text(str(raw.get("response") or raw.get("output") or raw.get("answer") or raw.get("answers") or raw.get("solution") or ""), max_chars)
        user = "\n\n".join(part for part in [instruction, context] if part)
        if user and response:
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": response}]
    elif fmt == "tool_calling_rows":
        tools = clean_text(str(raw.get("tools") or ""), max_chars)
        query = clean_text(str(raw.get("query") or raw.get("prompt") or raw.get("question") or ""), max_chars)
        answers = clean_text(str(raw.get("answers") or raw.get("answer") or raw.get("response") or raw.get("output") or ""), max_chars)
        if query and answers:
            user = f"Available tools:\n{tools}\n\nUser request:\n{query}" if tools else query
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": answers}]
    elif fmt == "oasst_message_rows":
        content = clean_text(str(raw.get("text") or ""), max_chars)
        role = normalize_role(raw.get("role") or "")
        if content and role:
            messages = [{"role": role, "content": content}]
    if not messages:
        return []
    return trim_messages(messages, max_chars)


def build_train_row(sample: dict[str, Any]) -> dict[str, Any] | None:
    messages = sample.get("messages") if isinstance(sample.get("messages"), list) else []
    assistant_index = last_assistant_index(messages)
    if assistant_index <= 0:
        return None
    prompt_messages = messages[:assistant_index]
    target = messages[assistant_index]
    task_id = f"open_conv_train_{sample['text_sha256'][:16]}"
    return {
        "task_id": task_id,
        "source_id": sample.get("source_id"),
        "dataset": sample.get("dataset"),
        "split": "private_train",
        "modality": "conversation",
        "prompt_messages": prompt_messages,
        "target_message": target,
        "causal_text": render_messages(prompt_messages + [target]),
        "license_spdx": sample.get("license_spdx"),
        "source_record_id": sample.get("record_id"),
        "training_use": "private_conversation_sft_pressure_only",
        "promotion_evidence": False,
        "public_benchmark": False,
        "external_inference_calls": 0,
        "provenance": sample.get("provenance", {}),
    }


def build_sts_row(sample: dict[str, Any]) -> dict[str, Any] | None:
    train = build_train_row(sample)
    if not train:
        return None
    prompt_messages = train["prompt_messages"]
    target = train["target_message"]
    user_text = "\n".join(item["content"] for item in prompt_messages if item.get("role") == "user")[-2400:]
    system_text = "\n".join(item["content"] for item in prompt_messages if item.get("role") == "system")[-1200:]
    target_text = str(target.get("content") or "")
    return {
        "task_id": f"open_conv_sts_{sample['text_sha256'][:16]}",
        "source_id": sample.get("source_id"),
        "dataset": sample.get("dataset"),
        "split": "private_train",
        "streams": {
            "system_context": system_text,
            "user_input": user_text,
            "assistant_visible": target_text,
            "critic_stream": infer_conversation_critic_stream(user_text, target_text),
            "audit_stream": "license_checked; private_pressure_only; no_public_benchmark_claim; external_inference_calls=0",
            "residual_stream": "conversation_response_modeling",
        },
        "input_streams": ["system_context", "user_input", "critic_stream", "audit_stream", "residual_stream"],
        "output_streams": ["assistant_visible"],
        "one_token_per_output_stream_target": True,
        "training_use": "private_conversation_sts_pressure_only",
        "promotion_evidence": False,
        "public_benchmark": False,
        "license_spdx": sample.get("license_spdx"),
        "source_record_id": sample.get("record_id"),
        "external_inference_calls": 0,
    }


def infer_conversation_critic_stream(user_text: str, target_text: str) -> str:
    lower_user = user_text.lower()
    flags = []
    if any(word in lower_user for word in ["code", "python", "function", "bug", "error"]):
        flags.append("code_or_tool_advice")
    if any(word in lower_user for word in ["summarize", "rewrite", "edit"]):
        flags.append("editing_or_summary")
    if any(word in lower_user for word in ["why", "explain", "reason"]):
        flags.append("explanation")
    if len(target_text) > 1200:
        flags.append("long_response")
    if not flags:
        flags.append("general_helpfulness")
    return ";".join(flags)


def fetch_hf_dataset_metadata(dataset: str) -> dict[str, Any]:
    url = f"{HF_API_DATASET_URL}/{dataset}"
    request = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseusConversationPantry/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_hf_rows(*, dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    url = hf_rows_url(dataset, config, split, offset, length)
    request = urllib.request.Request(url, headers={"User-Agent": "ProjectTheseusConversationPantry/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 429 and offset == 0:
            return fetch_hf_first_rows(dataset=dataset, config=config, split=split, length=length)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def fetch_hf_first_rows(*, dataset: str, config: str, split: str, length: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split})
    request = urllib.request.Request(
        f"{HF_FIRST_ROWS_URL}?{params}",
        headers={"User-Agent": "ProjectTheseusConversationPantry/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows", [])
    if isinstance(rows, list):
        payload["rows"] = rows[: max(1, length)]
    return payload


def hf_rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_URL}?{params}"


def extract_hf_license(metadata: dict[str, Any]) -> str:
    card = metadata.get("cardData") if isinstance(metadata.get("cardData"), dict) else {}
    value = card.get("license") or metadata.get("license")
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str) and value:
        return value
    for tag in metadata.get("tags", []) if isinstance(metadata.get("tags"), list) else []:
        tag_text = str(tag)
        if tag_text.startswith("license:"):
            return tag_text.split(":", 1)[1]
    return ""


def trim_messages(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    used = 0
    for item in messages:
        content = item["content"]
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining].rstrip()
        if content:
            output.append({"role": item["role"], "content": content})
            used += len(content)
    return output


def clean_text(value: str, max_chars: int) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = EMAIL_RE.sub("[redacted_email]", text)
    text = PHONE_RE.sub("[redacted_phone]", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text[:max_chars].strip()


def normalize_role(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ROLE_MAP.get(text, text if text in {"assistant", "user", "system"} else "")


def language_allowed(raw: dict[str, Any], required_langs: Any) -> bool:
    if not required_langs:
        return True
    if isinstance(required_langs, str):
        allowed = {required_langs.lower()}
    elif isinstance(required_langs, list):
        allowed = {str(item).lower() for item in required_langs if str(item).strip()}
    else:
        return True
    lang = str(raw.get("lang") or raw.get("language") or "").strip().lower()
    return not lang or lang in allowed


def last_assistant_index(messages: list[Any]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        item = messages[index]
        if isinstance(item, dict) and item.get("role") == "assistant" and str(item.get("content") or "").strip():
            return index
    return -1


def expand_sources(sources: Any) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    if not isinstance(sources, list):
        return expanded
    for source in sources:
        if not isinstance(source, dict):
            continue
        offsets = source.get("offsets")
        if not isinstance(offsets, list) or not offsets:
            expanded.append(source)
            continue
        base_id = str(source.get("id") or source.get("dataset") or "source")
        for offset in offsets:
            try:
                offset_int = int(offset)
            except (TypeError, ValueError):
                continue
            row = dict(source)
            row.pop("offsets", None)
            row["offset"] = offset_int
            row["id"] = f"{base_id}_offset_{offset_int}"
            expanded.append(row)
    return expanded


def contains_exclusion_token(text: str, tokens: set[str]) -> bool:
    lower = text.lower()
    for token in tokens:
        token = str(token or "").lower().strip()
        if not token:
            continue
        if len(token) <= 4 and re.fullmatch(r"[a-z0-9_/-]+", token):
            pattern = rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])"
            if re.search(pattern, lower):
                return True
            continue
        if token in lower:
            return True
    return False


def render_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"<|{item['role']}|>\n{item['content']}" for item in messages)


def normal_license(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "source")).strip("_") or "source"


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def approved_training_root(root: Path) -> dict[str, Any]:
    normalized = normalize_path(root)
    lower = normalized.lower()
    if lower.startswith("d:/projecttheseus/training_data/") or lower == "d:/projecttheseus/training_data":
        return {"approved": True, "evidence": normalized, "error": ""}

    approved_roots = [ROOT / "data" / "training_data"]
    env_root = os_environ("THESEUS_TRAINING_DATA_ROOT")
    if env_root:
        approved_roots.append(Path(env_root))
    approved_roots.append(Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "training_data")

    resolved_root = resolve_soft(root)
    for approved in approved_roots:
        if is_relative_to(resolved_root, resolve_soft(approved)):
            return {
                "approved": True,
                "evidence": f"{normalized} under {normalize_path(approved)}",
                "error": "",
            }
    return {
        "approved": False,
        "evidence": normalized,
        "error": (
            "conversation pantry root must be under D:/ProjectTheseus/training_data, "
            "repo data/training_data, THESEUS_TRAINING_DATA_ROOT, or the installed Hive training_data root"
        ),
    }


def resolve_soft(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except TypeError:
        return path.expanduser().resolve()


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def os_environ(name: str) -> str:
    import os

    return os.environ.get(name, "")


def check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def filter_retained_rows(
    rows: list[dict[str, Any]],
    *,
    allowed_licenses: set[str],
    exclusion_tokens: set[str],
    row_kind: str,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for row in rows:
        license_value = normal_license(row.get("license_spdx"))
        if license_value and license_value not in allowed_licenses:
            continue
        text = retained_row_text(row, row_kind)
        if text and contains_exclusion_token(text, exclusion_tokens):
            continue
        retained.append(row)
    return retained


def retained_row_text(row: dict[str, Any], row_kind: str) -> str:
    if row_kind == "sample":
        messages = row.get("messages")
        if isinstance(messages, list):
            clean_messages = [
                {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
                for item in messages
                if isinstance(item, dict) and item.get("role") and item.get("content")
            ]
            return render_messages(clean_messages) if clean_messages else ""
    if row_kind == "train":
        return str(row.get("causal_text") or "")
    if row_kind == "sts":
        streams = row.get("streams") if isinstance(row.get("streams"), dict) else {}
        return "\n".join(str(value) for value in streams.values())
    return json.dumps(row, sort_keys=True)


def merge_json_rows(
    previous_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in previous_rows + current_rows:
        key = json_row_key(row, key_fields)
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def json_row_key(row: dict[str, Any], key_fields: tuple[str, ...]) -> str:
    for field in key_fields:
        value = row.get(field)
        if value:
            return f"{field}:{value}"
    return "sha256:" + stable_hash(json.dumps(row, sort_keys=True, ensure_ascii=False))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# Open Conversation Training Pantry",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Root: `{report.get('root')}`",
        f"- Conversation samples: `{summary.get('conversation_samples', 0)}`",
        f"- Private train rows: `{summary.get('private_train_rows', 0)}`",
        f"- STS rows: `{summary.get('sts_rows', 0)}`",
        f"- Bulk download: `{summary.get('bulk_download')}`",
        f"- Promotion evidence: `{summary.get('promotion_evidence')}`",
        "",
        "## Sources",
        "",
    ]
    for source in report.get("sources", []) if isinstance(report.get("sources"), list) else []:
        lines.append(
            f"- `{source.get('source_id')}`: `{source.get('status')}` / "
            f"license `{source.get('license_spdx')}` / train `{source.get('private_train_rows', 0)}` / "
            f"STS `{source.get('sts_rows', 0)}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "Rows from this pantry are private conversational training pressure only. They are not public benchmark evidence, not teacher distillation, and not candidate-promotion proof.",
            "",
        ]
    )
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
