#!/usr/bin/env python3
"""Replace only the six VCM source slots rejected by English-scope audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5
import theseus_vcm_source_acquisition_v6 as v6
import theseus_vcm_source_materialization as source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_language_replacements.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_language_replacements.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_language_replacements_checkpoint.json"
POLICY = "project_theseus_vcm_language_replacements_v1"
FORBIDDEN_SCRIPT_NAMES = (
    "ARABIC", "ARMENIAN", "BENGALI", "CJK", "CYRILLIC", "DEVANAGARI",
    "GEORGIAN", "GREEK", "GUJARATI", "GURMUKHI", "HANGUL", "HEBREW",
    "HIRAGANA", "KANNADA", "KATAKANA", "KHMER", "LAO", "MALAYALAM",
    "MYANMAR", "ORIYA", "SINHALA", "TAMIL", "TELUGU", "THAI",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint", default=p2a.rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        config = p2a.read_json(config_path)
        retry_policy = p2a.mapping(config.get("transport_retry_policy"))
        ledger = source.SourceLedger(p2a.resolve(args.checkpoint), config_path, retry_policy)
        client = source.SourceClient(ledger, retry_policy)
        try:
            report = acquire(config_path, ledger, client, retry_policy)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "LANGUAGE_REPLACEMENT_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "replacement_set_admitted": False,
            }
        report = source.finalize_receipt(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"source_binding_invalid:{binding.get('id')}")
    audit_path = p2a.resolve(str(config.get("source_panel_audit_report") or ""))
    audit = p2a.read_json(audit_path) if audit_path.is_file() else {}
    slots = p2a.dicts(config.get("replacement_slots"))
    observed_slots = p2a.dicts(audit.get("replacement_slots_required"))
    slot_shape = [(integer(row.get("index")), row.get("panel"), row.get("query_language")) for row in slots]
    observed_shape = [(integer(row.get("index")), row.get("panel"), row.get("query_language")) for row in observed_slots]
    if audit.get("trigger_state") != "RED" or audit.get("state") != "SOURCE_PANEL_LANGUAGE_REPLACEMENTS_REQUIRED" or audit.get("archive_integrity_green") is not True or slot_shape != observed_shape:
        faults.append("language_replacement_trigger_invalid")
    if len(slots) != 6 or len({integer(row.get("index")) for row in slots}) != 6:
        faults.append("replacement_slot_cardinality_invalid")
    language_policy = p2a.mapping(config.get("english_language_policy"))
    classifier_path = p2a.resolve(str(language_policy.get("classifier_source") or ""))
    if not classifier_path.is_file() or p2a.sha256_file(classifier_path) != str(language_policy.get("classifier_source_sha256") or ""):
        faults.append("language_classifier_binding_invalid")
    if language_policy.get("required_dominant_language") != "en" or language_policy.get("forbid_non_latin_scripts") is not True:
        faults.append("english_language_policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {"public_metadata_queries_authorized", "public_source_file_retrieval_authorized", "public_pr_title_metadata_retrieval_authorized", "local_language_scope_classification_authorized"}
    if any(authority.get(key) is not True for key in allowed) or any(value is not False for key, value in authority.items() if key not in allowed):
        faults.append("authority_boundary_invalid")
    counters = source.zero_counters()
    counters.update({"public_metadata_selection_requests": 0, "local_language_scope_classification_calls": 0})
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "SIX_LANGUAGE_REPLACEMENT_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT",
        "faults": sorted(set(faults)),
        "config": v1.artifact(config_path),
        "source_panel_audit": v1.artifact(audit_path),
        "replacement_set_admitted": False,
        "selected_repository_count": 0,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "counters": counters,
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(
    config_path: Path,
    ledger: source.SourceLedger,
    client: source.SourceClient,
    retry_policy: dict[str, Any],
    classifier: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    scientific = p2a.read_json(p2a.resolve(str(config.get("scientific_selection_config") or "")))
    selection = p2a.read_json(p2a.resolve(str(config.get("metadata_selection_report") or "")))
    replacement_28 = p2a.read_json(p2a.resolve(str(config.get("replacement_28_report") or "")))
    slots = p2a.dicts(config.get("replacement_slots"))
    slots_by_language: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        slots_by_language.setdefault(str(slot.get("query_language") or ""), []).append(slot)
    denylist = set(v1.tracked_prior_repositories(config_path))
    denylist.update(str(row.get("repository") or "") for row in p2a.dicts(selection.get("selected_source_identities")))
    denylist.add(str(p2a.mapping(replacement_28.get("replacement_materialization")).get("repository") or ""))
    rest = v5.RetryingClient(v1.api_json, ledger, retry_policy)
    graphql = v5.RetryingClient(v6.graphql_api, ledger, retry_policy)
    chronology = p2a.mapping(scientific.get("chronology"))
    search = p2a.mapping(scientific.get("search"))
    selection_policy = p2a.mapping(scientific.get("selection"))
    graph_policy = p2a.mapping(scientific.get("graphql_transport"))
    interval = float(graph_policy.get("rest_search_minimum_interval_seconds") or 0.0)
    batch_size = integer(graph_policy.get("node_batch_size"))
    classify = classifier or classify_english
    language_policy = p2a.mapping(config.get("english_language_policy"))
    metadata_requests = 0
    classifier_calls = 0
    rejections: Counter[str] = Counter()
    selected_rows: list[dict[str, Any]] = []
    language_receipts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="theseus_vcm_language_replacements_", dir="/private/tmp") as temporary:
        staging = Path(temporary)
        for language, language_slots in slots_by_language.items():
            candidates: dict[tuple[str, int], dict[str, Any]] = {}
            last_search_started: float | None = None
            for page in range(1, integer(search.get("pages_per_language")) + 1):
                now = time.monotonic()
                if last_search_started is not None:
                    time.sleep(max(0.0, last_search_started + interval - now))
                last_search_started = time.monotonic()
                payload, _digest = rest.call("search/issues", {
                    "q": v1.search_query(language, chronology),
                    "sort": search.get("sort"), "order": search.get("order"),
                    "per_page": search.get("items_per_page"), "page": page,
                })
                metadata_requests += 1
                for item in p2a.dicts(p2a.mapping(payload).get("items")):
                    repository = v1.repository_from_url(str(item.get("repository_url") or ""))
                    number = integer(item.get("number"))
                    node_id = str(item.get("node_id") or "")
                    if repository and number and node_id:
                        candidates[(repository, number)] = {
                            "repository": repository, "pull_request": number,
                            "query_language": language, "node_id": node_id,
                            "title_sha256": hashlib.sha256(str(item.get("title") or "").encode()).hexdigest(),
                            "rank": v1.rank(selection_policy.get("selection_seed"), repository, number),
                        }
            ranked = sorted(candidates.values(), key=lambda row: row["rank"])
            offset = 0
            accepted_for_language = 0
            while offset < len(ranked) and accepted_for_language < len(language_slots):
                batch: list[dict[str, Any]] = []
                batch_repositories: set[str] = set()
                while offset < len(ranked) and len(batch) < batch_size:
                    candidate = ranked[offset]
                    offset += 1
                    repository = candidate["repository"]
                    if repository in denylist or repository in batch_repositories:
                        rejections["prior_or_duplicate_repository"] += 1
                        continue
                    batch.append(candidate)
                    batch_repositories.add(repository)
                if not batch:
                    continue
                payload, _digest = graphql.call("graphql:nodes", {"ids": [row["node_id"] for row in batch]})
                metadata_requests += 1
                nodes = {str(node.get("id") or ""): node for node in p2a.dicts(p2a.mapping(p2a.mapping(payload).get("data")).get("nodes"))}
                for candidate in batch:
                    qualified, reasons = v6.qualify_node(nodes.get(candidate["node_id"]), candidate, scientific)
                    if reasons:
                        rejections.update(reasons)
                        continue
                    title = client.title(qualified["repository"], integer(qualified.get("pull_request")))
                    if hashlib.sha256(title.encode()).hexdigest() != qualified.get("title_sha256"):
                        raise RuntimeError("pull_request_title_digest_changed")
                    english, language_receipt = classify(title, language_policy)
                    classifier_calls += 1
                    language_receipt = {**language_receipt, "title_sha256": qualified.get("title_sha256")}
                    if not english:
                        rejections["natural_language_out_of_scope"] += 1
                        continue
                    slot = language_slots[accepted_for_language]
                    qualified["panel"] = slot["panel"]
                    try:
                        materialized, content_faults, _row_bytes = source.materialize_row(
                            qualified, integer(slot.get("index")), staging, client,
                            integer(p2a.mapping(config.get("archive_policy")).get("maximum_single_file_bytes")),
                        )
                    except v5.CandidateMetadataUnavailable:
                        rejections["candidate_source_or_license_unavailable"] += 1
                        continue
                    if content_faults:
                        rejections.update(content_faults)
                        continue
                    materialized["replacement_for_title_sha256"] = slot.get("rejected_title_sha256")
                    selected_rows.append(materialized)
                    language_receipts.append({**language_receipt, "index": integer(slot.get("index")), "repository": qualified["repository"]})
                    denylist.add(qualified["repository"])
                    accepted_for_language += 1
                    if accepted_for_language == len(language_slots):
                        break
            if accepted_for_language != len(language_slots):
                return failure(before, metadata_requests, classifier_calls, rejections, selected_rows)
        selected_rows.sort(key=lambda row: integer(row.get("index")))
        output_directory = p2a.resolve(str(config.get("output_directory") or ""))
        output_directory.mkdir(parents=True, exist_ok=True)
        for row in selected_rows:
            for receipt in p2a.mapping(row.get("archives")).values():
                record = p2a.mapping(receipt)
                staged = p2a.resolve(str(record.get("path") or ""))
                final = output_directory / staged.name
                staged.replace(final)
                record["path"] = p2a.rel(final)
    counters = source.zero_counters()
    counters.update({
        "public_metadata_selection_requests": metadata_requests,
        "public_metadata_title_requests": client.title_requests,
        "public_source_content_requests": client.source_requests,
        "source_archives_materialized": 4 * len(selected_rows),
        "source_bytes_materialized": sum(member["bytes"] for row in selected_rows for receipt in p2a.mapping(row.get("archives")).values() for member in p2a.dicts(p2a.mapping(receipt).get("members"))),
        "local_language_scope_classification_calls": classifier_calls,
    })
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN",
        "state": "SIX_ENGLISH_SOURCE_REPLACEMENTS_BOUND",
        "faults": [],
        "replacement_set_admitted": True,
        "selected_repository_count": len(selected_rows),
        "source_content_retrieval_opened": True,
        "replacement_rows": selected_rows,
        "language_classification_receipts": language_receipts,
        "rejection_counts": dict(sorted(rejections.items())),
        "counters": counters,
    }


def failure(before: dict[str, Any], metadata_requests: int, classifier_calls: int, rejections: Counter[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = dict(p2a.mapping(before.get("counters")))
    counters["public_metadata_selection_requests"] = metadata_requests
    counters["local_language_scope_classification_calls"] = classifier_calls
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "RED",
        "state": "SIX_LANGUAGE_REPLACEMENT_POOL_EXHAUSTED_NO_PARTIAL_ADMISSION",
        "faults": ["ranked_same_language_replacement_pool_exhausted"],
        "replacement_set_admitted": False,
        "selected_repository_count": 0,
        "staged_replacement_count_discarded": len(rows),
        "source_content_retrieval_opened": True,
        "rejection_counts": dict(sorted(rejections.items())),
        "counters": counters,
    }


def classify_english(title: str, policy: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    forbidden = sorted({name for character in title for name in FORBIDDEN_SCRIPT_NAMES if name in unicodedata.name(character, "")})
    command = [
        str(policy.get("swift_executable")), "-module-cache-path",
        str(policy.get("module_cache_path")), str(p2a.resolve(str(policy.get("classifier_source")))), title,
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = p2a.mapping(json.loads(completed.stdout))
    dominant = str(payload.get("dominant_language") or "unknown")
    accepted = dominant == str(policy.get("required_dominant_language")) and not forbidden
    return accepted, {
        "dominant_language": dominant,
        "forbidden_unicode_scripts": forbidden,
        "accepted_english": accepted,
        "classifier_output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "replacement_set_admitted", "selected_repository_count",
        "source_content_retrieval_opened", "candidate_packet_materialization_opened",
        "hidden_evaluation_opened", "faults", "rejection_counts", "counters",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
