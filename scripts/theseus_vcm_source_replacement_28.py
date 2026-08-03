#!/usr/bin/env python3
"""Select and materialize one frozen-rule Python claim replacement for Task 28."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5
import theseus_vcm_source_acquisition_v6 as v6
import theseus_vcm_source_materialization as source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_replacement_28.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_replacement_28.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_source_replacement_28_checkpoint.json"
POLICY = "project_theseus_vcm_source_replacement_28_v1"


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
        policy = p2a.mapping(config.get("transport_retry_policy"))
        ledger = source.SourceLedger(
            p2a.resolve(args.checkpoint), config_path, policy
        )
        client = source.SourceClient(ledger, policy)
        try:
            report = acquire(config_path, ledger, client, policy)
        except (
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "TASK_28_REPLACEMENT_TRANSPORT_PAUSED_NO_ADMISSION",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "replacement_admitted": False,
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
        if not path.is_file() or p2a.sha256_file(path) != str(
            binding.get("sha256") or ""
        ):
            faults.append(f"source_binding_invalid:{binding.get('id')}")
    selection_path = p2a.resolve(str(config.get("metadata_selection_report") or ""))
    failed_path = p2a.resolve(str(config.get("failed_materialization_report") or ""))
    selection = p2a.read_json(selection_path) if selection_path.is_file() else {}
    failed = p2a.read_json(failed_path) if failed_path.is_file() else {}
    rows = p2a.dicts(selection.get("selected_source_identities"))
    failed_rows = p2a.dicts(failed.get("rows"))
    failed_row = next(
        (row for row in failed_rows if int(row.get("index") or 0) == 28), {}
    )
    if (
        selection.get("trigger_state") != "GREEN"
        or len(rows) != 62
        or failed.get("trigger_state") != "RED"
        or failed.get("faults") != ["task_28:selected_verifier_bytes_unchanged"]
        or failed_row.get("panel") != "claim"
        or failed_row.get("query_language") != "Python"
        or failed_row.get("faults") != ["selected_verifier_bytes_unchanged"]
    ):
        faults.append("replacement_trigger_invalid")
    replacement = p2a.mapping(config.get("replacement_policy"))
    if (
        int(replacement.get("task_index") or 0) != 28
        or replacement.get("panel") != "claim"
        or replacement.get("language") != "Python"
        or replacement.get("same_selection_seed_required") is not True
        or replacement.get("all_v7_repositories_excluded") is not True
        or replacement.get("selected_source_bytes_must_change") is not True
        or replacement.get("selected_verifier_bytes_must_change") is not True
        or replacement.get("first_ranked_content_qualified_candidate_required")
        is not True
    ):
        faults.append("replacement_policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    allowed = {
        "public_metadata_queries_authorized",
        "public_source_file_retrieval_authorized",
        "public_pr_title_metadata_retrieval_authorized",
    }
    if any(authority.get(key) is not True for key in allowed):
        faults.append("required_retrieval_authority_missing")
    if any(
        value is not False
        for key, value in authority.items()
        if key not in allowed
    ):
        faults.append("authority_boundary_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": (
            "TASK_28_PYTHON_CLAIM_REPLACEMENT_PREFLIGHT_GREEN"
            if not faults
            else "INVALID_PREFLIGHT"
        ),
        "faults": sorted(set(faults)),
        "config": v1.artifact(config_path),
        "metadata_selection": v1.artifact(selection_path),
        "failed_materialization": v1.artifact(failed_path),
        "replacement_admitted": False,
        "selected_repository_count": 0,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "counters": source.zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(
    config_path: Path,
    ledger: source.SourceLedger,
    client: source.SourceClient,
    retry_policy: dict[str, Any],
) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    scientific = p2a.read_json(
        p2a.resolve(str(config.get("scientific_selection_config") or ""))
    )
    selection = p2a.read_json(
        p2a.resolve(str(config.get("metadata_selection_report") or ""))
    )
    selected_repositories = {
        str(row.get("repository") or "")
        for row in p2a.dicts(selection.get("selected_source_identities"))
    }
    denylist = set(v1.tracked_prior_repositories(config_path)) | selected_repositories
    rest = v5.RetryingClient(v1.api_json, ledger, retry_policy)
    graphql = v5.RetryingClient(v6.graphql_api, ledger, retry_policy)
    chronology = p2a.mapping(scientific.get("chronology"))
    search = p2a.mapping(scientific.get("search"))
    selection_policy = p2a.mapping(scientific.get("selection"))
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    last_search_started: float | None = None
    interval = float(
        p2a.mapping(scientific.get("graphql_transport")).get(
            "rest_search_minimum_interval_seconds"
        )
        or 0.0
    )
    metadata_requests = 0
    for page in range(1, int(search.get("pages_per_language") or 0) + 1):
        now = time.monotonic()
        if last_search_started is not None:
            time.sleep(max(0.0, last_search_started + interval - now))
        last_search_started = time.monotonic()
        payload, _digest = rest.call(
            "search/issues",
            {
                "q": v1.search_query("Python", chronology),
                "sort": search.get("sort"),
                "order": search.get("order"),
                "per_page": search.get("items_per_page"),
                "page": page,
            },
        )
        metadata_requests += 1
        for item in p2a.dicts(p2a.mapping(payload).get("items")):
            repository = v1.repository_from_url(
                str(item.get("repository_url") or "")
            )
            number = int(item.get("number") or 0)
            node_id = str(item.get("node_id") or "")
            if repository and number and node_id:
                candidates[(repository, number)] = {
                    "repository": repository,
                    "pull_request": number,
                    "query_language": "Python",
                    "node_id": node_id,
                    "title_sha256": hashlib.sha256(
                        str(item.get("title") or "").encode()
                    ).hexdigest(),
                    "rank": v1.rank(
                        selection_policy.get("selection_seed"), repository, number
                    ),
                }
    ranked = sorted(candidates.values(), key=lambda row: row["rank"])
    rejections: dict[str, int] = {}
    output_directory = p2a.resolve(str(config.get("output_directory") or ""))
    maximum_file = int(
        p2a.mapping(config.get("archive_policy")).get("maximum_single_file_bytes")
        or 0
    )
    batch_size = int(
        p2a.mapping(scientific.get("graphql_transport")).get("node_batch_size")
        or 0
    )
    offset = 0
    while offset < len(ranked):
        batch: list[dict[str, Any]] = []
        batch_repositories: set[str] = set()
        while offset < len(ranked) and len(batch) < batch_size:
            candidate = ranked[offset]
            offset += 1
            repository = candidate["repository"]
            if repository in denylist or repository in batch_repositories:
                v1.bump(rejections, "prior_or_duplicate_repository")
                continue
            batch.append(candidate)
            batch_repositories.add(repository)
        if not batch:
            continue
        payload, _digest = graphql.call(
            "graphql:nodes", {"ids": [row["node_id"] for row in batch]}
        )
        metadata_requests += 1
        nodes = {
            str(node.get("id") or ""): node
            for node in p2a.dicts(
                p2a.mapping(p2a.mapping(payload).get("data")).get("nodes")
            )
        }
        for candidate in batch:
            qualified, reasons = v6.qualify_node(
                nodes.get(candidate["node_id"]), candidate, scientific
            )
            if reasons:
                for reason in reasons:
                    v1.bump(rejections, reason)
                continue
            qualified["panel"] = "claim"
            try:
                materialized, content_faults, row_bytes = source.materialize_row(
                    qualified, 28, output_directory, client, maximum_file
                )
            except v5.CandidateMetadataUnavailable:
                v1.bump(rejections, "candidate_source_or_license_unavailable")
                continue
            if content_faults:
                for reason in content_faults:
                    v1.bump(rejections, reason)
                continue
            counters = source.zero_counters()
            counters["public_metadata_selection_requests"] = metadata_requests
            counters["public_metadata_title_requests"] = client.title_requests
            counters["public_source_content_requests"] = client.source_requests
            counters["source_archives_materialized"] = 4
            counters["source_bytes_materialized"] = row_bytes
            return {
                **before,
                "created_utc": p2a.now(),
                "trigger_state": "GREEN",
                "state": "TASK_28_PYTHON_CLAIM_REPLACEMENT_SOURCE_BOUND",
                "faults": [],
                "replacement_admitted": True,
                "selected_repository_count": 1,
                "source_content_retrieval_opened": True,
                "replacement_selection_rank_sha256": qualified[
                    "selection_rank_sha256"
                ],
                "replacement_metadata": qualified,
                "replacement_materialization": materialized,
                "rejection_counts": dict(sorted(rejections.items())),
                "counters": counters,
            }
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "RED",
        "state": "TASK_28_REPLACEMENT_NOT_FOUND",
        "faults": ["ranked_python_replacement_pool_exhausted"],
        "replacement_admitted": False,
        "source_content_retrieval_opened": True,
        "rejection_counts": dict(sorted(rejections.items())),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state", "state", "replacement_admitted",
            "selected_repository_count", "source_content_retrieval_opened",
            "candidate_packet_materialization_opened", "hidden_evaluation_opened",
            "faults", "counters",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
