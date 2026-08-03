#!/usr/bin/env python3
"""Expanded-pool, bounded-concurrency metadata selector for VCM source v3."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition_v3.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition_v3.json"
POLICY = "project_theseus_vcm_source_acquisition_v1"
REPORT_POLICY = "project_theseus_vcm_source_acquisition_report_v1"
PAGES_PER_LANGUAGE = 10
MAXIMUM_SEARCH_ROWS = 4000
WORKERS = 8
QUALIFICATION_BATCH_SIZE = 40


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        try:
            report = acquire(config_path)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "PUBLIC_METADATA_TRANSPORT_PAUSED_NO_SOURCE_EXPOSURE",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
            }
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    started = time.perf_counter()
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if (
        config.get("policy") != POLICY
        or config.get("state") != "PROSPECTIVE_METADATA_SELECTION_ZERO_CANDIDATE_CONTENT"
    ):
        faults.append("config_identity_invalid")
    for row in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(row.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != row.get("sha256"):
            faults.append(f"source_binding_invalid:{row.get('id') or p2a.rel(path)}")
    chronology = p2a.mapping(config.get("chronology"))
    try:
        snapshot = v1.parse_time(chronology.get("frozen_model_snapshot_created_utc"))
        start = v1.parse_time(chronology.get("search_window_start_utc"))
        end = v1.parse_time(chronology.get("search_window_end_utc"))
        if not snapshot < start < end:
            faults.append("chronology_invalid")
    except (TypeError, ValueError):
        faults.append("chronology_invalid")
    search = p2a.mapping(config.get("search"))
    languages = p2a.strings(search.get("languages"))
    if (
        languages != list(v1.LANGUAGE_SUFFIXES)
        or int(search.get("pages_per_language") or 0) != PAGES_PER_LANGUAGE
        or int(search.get("items_per_page") or 0) != 100
        or int(search.get("maximum_search_metadata_rows") or 0) != MAXIMUM_SEARCH_ROWS
        or int(search.get("qualification_workers") or 0) != WORKERS
        or int(search.get("qualification_batch_size") or 0) != QUALIFICATION_BATCH_SIZE
    ):
        faults.append("search_boundary_invalid")
    panels = p2a.mapping(config.get("panels"))
    control = p2a.mapping(panels.get("control_qualification"))
    claim = p2a.mapping(panels.get("claim"))
    if (
        int(control.get("task_count") or 0) != 9
        or int(claim.get("task_count") or 0) != 53
        or int(panels.get("total_task_count") or 0) != 62
        or sum(int(v) for v in p2a.mapping(control.get("language_quotas")).values()) != 9
        or sum(int(v) for v in p2a.mapping(claim.get("language_quotas")).values()) != 53
        or set(p2a.mapping(control.get("language_quotas"))) != set(languages)
        or set(p2a.mapping(claim.get("language_quotas"))) != set(languages)
        or panels.get("source_disjoint") is not True
        or panels.get("reference_outputs_may_select_or_assign_tasks") is not False
    ):
        faults.append("panel_contract_invalid")
    selection = p2a.mapping(config.get("selection"))
    denylist = v1.tracked_prior_repositories(config_path)
    if (
        len(denylist) != int(selection.get("prior_repository_denylist_count") or 0)
        or v1.stable_list_hash(denylist) != selection.get("prior_repository_denylist_sha256")
        or selection.get("one_pull_request_per_repository") is not True
        or selection.get("eligibility_is_computed_without_model_outputs") is not True
        or selection.get("candidate_visible_packet_materialization_authorized") is not False
        or selection.get("target_source_content_retrieval_authorized") is not False
        or int(selection.get("minimum_repository_stars") or 0) != 1
        or not p2a.strings(selection.get("license_spdx_allowlist"))
    ):
        faults.append("selection_contract_invalid")
    authority = p2a.mapping(config.get("authority"))
    if (
        authority.get("public_metadata_queries_authorized_after_green_preflight") is not True
        or authority.get("public_source_content_retrieval_authorized") is not False
        or authority.get("candidate_packet_materialization_authorized") is not False
        or authority.get("local_model_calls_authorized") != 0
        or authority.get("external_reference_calls_authorized") != 0
        or authority.get("hidden_evaluation_authorized") is not False
        or authority.get("teacher_calls_authorized") is not False
        or authority.get("training_rows_authorized") is not False
        or authority.get("D1_authorized") is not False
        or authority.get("D2_authorized") is not False
        or authority.get("book_support_promotion_authorized") is not False
        or authority.get("user_or_operator_gate") is not False
    ):
        faults.append("authority_boundary_invalid")
    return {
        "policy": REPORT_POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "METADATA_SELECTION_V3_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT",
        "config": v1.artifact(config_path),
        "prior_repository_denylist": {
            "count": len(denylist), "sha256": v1.stable_list_hash(denylist),
            "identities_retained": False,
        },
        "metadata_selection_opened": False,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "selected_repository_count": 0,
        "faults": sorted(set(faults)),
        "counters": v1.zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
        "runtime_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def acquire(config_path: Path) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    search = p2a.mapping(config.get("search"))
    selection = p2a.mapping(config.get("selection"))
    chronology = p2a.mapping(config.get("chronology"))
    panels = p2a.mapping(config.get("panels"))
    denylist = set(v1.tracked_prior_repositories(config_path))
    response_digests: list[str] = []
    request_count = 0
    search_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for language in p2a.strings(search.get("languages")):
        for page in range(1, PAGES_PER_LANGUAGE + 1):
            payload, digest = v1.api_json(
                "search/issues",
                {"q": v1.search_query(language, chronology), "sort": search.get("sort"),
                 "order": search.get("order"), "per_page": 100, "page": page},
            )
            request_count += 1
            response_digests.append(digest)
            for item in p2a.dicts(p2a.mapping(payload).get("items")):
                repository = v1.repository_from_url(str(item.get("repository_url") or ""))
                number = int(item.get("number") or 0)
                if repository and number:
                    search_rows[(repository, number)] = {
                        "repository": repository,
                        "pull_request": number,
                        "query_language": language,
                        "rank": v1.rank(selection.get("selection_seed"), repository, number),
                    }
    selected_by_language: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for language in p2a.strings(search.get("languages")):
            control_quota = int(p2a.mapping(p2a.mapping(panels.get("control_qualification")).get("language_quotas")).get(language) or 0)
            claim_quota = int(p2a.mapping(p2a.mapping(panels.get("claim")).get("language_quotas")).get(language) or 0)
            total_quota = control_quota + claim_quota
            candidates = sorted(
                (row for row in search_rows.values() if row["query_language"] == language),
                key=lambda row: row["rank"],
            )
            admitted: list[dict[str, Any]] = []
            used_repositories: set[str] = set()
            for offset in range(0, len(candidates), QUALIFICATION_BATCH_SIZE):
                if len(admitted) >= total_quota:
                    break
                batch = [row for row in candidates[offset:offset + QUALIFICATION_BATCH_SIZE]
                         if row["repository"] not in denylist and row["repository"] not in used_repositories]
                results = list(executor.map(lambda row: v1.qualify_metadata(row, config), batch))
                for candidate, (row, reasons, calls, digests) in zip(batch, results):
                    request_count += calls
                    response_digests.extend(digests)
                    if reasons:
                        for reason in reasons:
                            v1.bump(rejection_counts, reason)
                        continue
                    if candidate["repository"] in used_repositories:
                        v1.bump(rejection_counts, "duplicate_repository")
                        continue
                    used_repositories.add(candidate["repository"])
                    admitted.append(row)
                    if len(admitted) >= total_quota:
                        break
            if len(admitted) != total_quota:
                return v1.terminal_report(
                    before, config, request_count, response_digests, [], rejection_counts,
                    [f"language_quota_unfilled:{language}:{len(admitted)}/{total_quota}"],
                )
            panel_ranked = sorted(admitted, key=lambda row: v1.rank(
                panels.get("assignment_seed"), row["repository"], row["pull_request"]
            ))
            controls = {(row["repository"], row["pull_request"]) for row in panel_ranked[:control_quota]}
            for row in admitted:
                row["panel"] = "control_qualification" if (row["repository"], row["pull_request"]) in controls else "claim"
                row["panel_assignment_sha256"] = v1.rank(
                    panels.get("assignment_seed"), row["repository"], row["pull_request"]
                )
            selected_by_language[language] = admitted
    rows = sorted(
        [row for group in selected_by_language.values() for row in group],
        key=lambda row: (row["panel"], row["query_language"], row["selection_rank_sha256"]),
    )
    repositories = [row["repository"] for row in rows]
    faults = []
    if len(rows) != 62 or len(set(repositories)) != 62 or set(repositories) & denylist:
        faults.append("terminal_source_disjointness_invalid")
    return v1.terminal_report(
        before, config, request_count, response_digests, rows, rejection_counts, faults
    )


if __name__ == "__main__":
    raise SystemExit(main())
