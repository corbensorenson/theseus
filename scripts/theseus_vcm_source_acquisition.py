#!/usr/bin/env python3
"""Prospectively bound, model-free source metadata selection for the VCM claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition.json"
POLICY = "project_theseus_vcm_source_acquisition_v1"
REPORT_POLICY = "project_theseus_vcm_source_acquisition_report_v1"
LANGUAGE_SUFFIXES = {
    "Python": (".py",),
    "TypeScript": (".ts", ".tsx"),
    "JavaScript": (".js", ".jsx", ".mjs", ".cjs"),
    "Rust": (".rs",),
}
TEST_PARTS = {"test", "tests", "testing", "spec", "specs", "__tests__"}
EXCLUDED_PARTS = {
    "vendor", "vendors", "generated", "dist", "build", "node_modules",
    "fixtures", "snapshots", "__snapshots__",
}
LOCK_SUFFIXES = (".lock", "package-lock.json", "npm-shrinkwrap.json")


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
    print(json.dumps(summary(report), indent=2, sort_keys=True))
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
        snapshot = parse_time(chronology.get("frozen_model_snapshot_created_utc"))
        start = parse_time(chronology.get("search_window_start_utc"))
        end = parse_time(chronology.get("search_window_end_utc"))
        if not snapshot < start < end:
            faults.append("chronology_invalid")
    except (TypeError, ValueError):
        faults.append("chronology_invalid")
    search = p2a.mapping(config.get("search"))
    languages = p2a.strings(search.get("languages"))
    if (
        languages != list(LANGUAGE_SUFFIXES)
        or int(search.get("pages_per_language") or 0) != 3
        or int(search.get("items_per_page") or 0) != 100
        or int(search.get("maximum_search_metadata_rows") or 0) != 1200
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
    denylist = tracked_prior_repositories(config_path)
    if (
        len(denylist) != int(selection.get("prior_repository_denylist_count") or 0)
        or stable_list_hash(denylist) != selection.get("prior_repository_denylist_sha256")
        or selection.get("one_pull_request_per_repository") is not True
        or selection.get("eligibility_is_computed_without_model_outputs") is not True
        or selection.get("candidate_visible_packet_materialization_authorized") is not False
        or selection.get("target_source_content_retrieval_authorized") is not False
        or int(selection.get("minimum_repository_stars") or 0) < 1
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
        "state": "METADATA_SELECTION_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT",
        "config": artifact(config_path),
        "prior_repository_denylist": {
            "count": len(denylist),
            "sha256": stable_list_hash(denylist),
            "identities_retained": False,
        },
        "metadata_selection_opened": False,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "selected_repository_count": 0,
        "faults": sorted(set(faults)),
        "counters": zero_counters(),
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
    denylist = set(tracked_prior_repositories(config_path))
    response_digests: list[str] = []
    request_count = 0
    search_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for language in p2a.strings(search.get("languages")):
        for page in range(1, int(search.get("pages_per_language") or 0) + 1):
            query = search_query(language, chronology)
            payload, digest = api_json(
                "search/issues",
                {"q": query, "sort": search.get("sort"), "order": search.get("order"),
                 "per_page": search.get("items_per_page"), "page": page},
            )
            request_count += 1
            response_digests.append(digest)
            for item in p2a.dicts(p2a.mapping(payload).get("items")):
                repository = repository_from_url(str(item.get("repository_url") or ""))
                number = int(item.get("number") or 0)
                if repository and number:
                    search_rows[(repository, number)] = {
                        "repository": repository,
                        "pull_request": number,
                        "query_language": language,
                        "rank": rank(selection.get("selection_seed"), repository, number),
                    }
    selected_by_language: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: dict[str, int] = {}
    for language in p2a.strings(search.get("languages")):
        control_quota = int(p2a.mapping(p2a.mapping(panels.get("control_qualification")).get("language_quotas")).get(language) or 0)
        claim_quota = int(p2a.mapping(p2a.mapping(panels.get("claim")).get("language_quotas")).get(language) or 0)
        total_quota = control_quota + claim_quota
        admitted: list[dict[str, Any]] = []
        used_repositories: set[str] = set()
        candidates = sorted(
            (row for row in search_rows.values() if row["query_language"] == language),
            key=lambda row: row["rank"],
        )
        for candidate in candidates:
            if len(admitted) >= total_quota:
                break
            repository = candidate["repository"]
            if repository in denylist or repository in used_repositories:
                bump(rejection_counts, "prior_or_duplicate_repository")
                continue
            row, reasons, calls, digests = qualify_metadata(candidate, config)
            request_count += calls
            response_digests.extend(digests)
            if reasons:
                for reason in reasons:
                    bump(rejection_counts, reason)
                continue
            used_repositories.add(repository)
            admitted.append(row)
        if len(admitted) != total_quota:
            return terminal_report(
                before, config, request_count, response_digests, [], rejection_counts,
                [f"language_quota_unfilled:{language}:{len(admitted)}/{total_quota}"],
            )
        panel_ranked = sorted(
            admitted,
            key=lambda row: rank(panels.get("assignment_seed"), row["repository"], row["pull_request"]),
        )
        controls = {(row["repository"], row["pull_request"]) for row in panel_ranked[:control_quota]}
        for row in admitted:
            row["panel"] = "control_qualification" if (row["repository"], row["pull_request"]) in controls else "claim"
            row["panel_assignment_sha256"] = rank(
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
    return terminal_report(
        before, config, request_count, response_digests, rows, rejection_counts, faults
    )


def qualify_metadata(candidate: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[str], int, list[str]]:
    repository = candidate["repository"]
    number = candidate["pull_request"]
    digests: list[str] = []
    pr, digest = api_json(f"repos/{repository}/pulls/{number}", {})
    digests.append(digest)
    repo, digest = api_json(f"repos/{repository}", {})
    digests.append(digest)
    files, digest = api_json(f"repos/{repository}/pulls/{number}/files", {"per_page": 100})
    digests.append(digest)
    head_sha = str(p2a.mapping(pr).get("head", {}).get("sha") or "")
    head_commit, digest = api_json(f"repos/{repository}/commits/{head_sha}", {})
    digests.append(digest)
    reasons = metadata_rejection_reasons(candidate, pr, repo, p2a.dicts(files), head_commit, config)
    row = {
        "opaque_source_id": hashlib.sha256(f"vcm-source:{repository}#{number}".encode()).hexdigest(),
        "repository": repository,
        "pull_request": number,
        "pull_request_url": pr.get("html_url"),
        "query_language": candidate["query_language"],
        "title_sha256": hashlib.sha256(str(pr.get("title") or "").encode()).hexdigest(),
        "created_utc": pr.get("created_at"),
        "merged_utc": pr.get("merged_at"),
        "base_revision": p2a.mapping(pr.get("base")).get("sha"),
        "head_revision": head_sha,
        "merge_revision": pr.get("merge_commit_sha"),
        "head_commit_utc": p2a.mapping(p2a.mapping(head_commit.get("commit")).get("committer")).get("date"),
        "license_spdx": p2a.mapping(repo.get("license")).get("spdx_id"),
        "repository_stars": repo.get("stargazers_count"),
        "changed_file_count": pr.get("changed_files"),
        "changed_line_count": int(pr.get("additions") or 0) + int(pr.get("deletions") or 0),
        "source_paths": sorted(path for path in file_paths(p2a.dicts(files)) if is_source(path, candidate["query_language"]) and not is_test(path)),
        "verifier_paths": sorted(path for path in file_paths(p2a.dicts(files)) if is_test(path)),
        "selection_rank_sha256": candidate["rank"],
        "metadata_qualified": not reasons,
        "candidate_content_retrieved": False,
        "candidate_packet_materialized": False,
    }
    return row, reasons, 4, digests


def metadata_rejection_reasons(candidate: dict[str, Any], pr: dict[str, Any], repo: dict[str, Any], files: list[dict[str, Any]], head_commit: dict[str, Any], config: dict[str, Any]) -> list[str]:
    selection = p2a.mapping(config.get("selection"))
    chronology = p2a.mapping(config.get("chronology"))
    reasons: list[str] = []
    if pr.get("state") != "closed" or not pr.get("merged_at") or pr.get("draft") is True:
        reasons.append("not_merged_closed_nondraft")
    author = str(p2a.mapping(pr.get("user")).get("login") or "").lower()
    if author.endswith("[bot]") or author.endswith("-bot") or author == "dependabot":
        reasons.append("bot_author")
    if repo.get("fork") is True or repo.get("archived") is True or repo.get("disabled") is True:
        reasons.append("repository_state")
    if repo.get("language") != candidate["query_language"]:
        reasons.append("primary_language_mismatch")
    if int(repo.get("stargazers_count") or 0) < int(selection.get("minimum_repository_stars") or 0):
        reasons.append("repository_below_star_floor")
    if p2a.mapping(repo.get("license")).get("spdx_id") not in set(p2a.strings(selection.get("license_spdx_allowlist"))):
        reasons.append("license_not_allowlisted")
    try:
        snapshot = parse_time(chronology.get("frozen_model_snapshot_created_utc"))
        start = parse_time(chronology.get("search_window_start_utc"))
        end = parse_time(chronology.get("search_window_end_utc"))
        created = parse_time(pr.get("created_at"))
        merged = parse_time(pr.get("merged_at"))
        head_time = parse_time(p2a.mapping(p2a.mapping(head_commit.get("commit")).get("committer")).get("date"))
        if created <= snapshot or head_time <= snapshot or not start <= merged <= end:
            reasons.append("chronology")
    except (TypeError, ValueError):
        reasons.append("chronology")
    file_count = int(pr.get("changed_files") or 0)
    line_count = int(pr.get("additions") or 0) + int(pr.get("deletions") or 0)
    if not int(selection.get("changed_file_count_min") or 0) <= file_count <= int(selection.get("changed_file_count_max") or 0):
        reasons.append("changed_file_count")
    if not int(selection.get("changed_line_count_min") or 0) <= line_count <= int(selection.get("changed_line_count_max") or 0):
        reasons.append("changed_line_count")
    paths = file_paths(files)
    sources = [path for path in paths if is_source(path, candidate["query_language"]) and not is_test(path)]
    verifiers = [path for path in paths if is_test(path)]
    if not sources:
        reasons.append("no_non_test_source_change")
    if not verifiers:
        reasons.append("no_machine_verifier_change")
    if sources and all(p2a.mapping(next((row for row in files if row.get("filename") == path), {})).get("status") == "removed" for path in sources):
        reasons.append("deleted_only_source")
    if paths and all(is_excluded_or_lock(path) for path in paths):
        reasons.append("generated_vendor_lock_only")
    return sorted(set(reasons))


def terminal_report(before: dict[str, Any], config: dict[str, Any], request_count: int, digests: list[str], rows: list[dict[str, Any]], rejection_counts: dict[str, int], faults: list[str]) -> dict[str, Any]:
    green = not faults and len(rows) == 62
    counters = zero_counters()
    counters["public_metadata_requests"] = request_count
    counters["public_metadata_rows_selected"] = len(rows)
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if green else "RED",
        "state": "SIXTY_TWO_SOURCE_IDENTITIES_FROZEN_BEFORE_CONTENT_RETRIEVAL" if green else "METADATA_SELECTION_INCOMPLETE",
        "metadata_selection_opened": True,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "selected_repository_count": len(rows),
        "selected_source_identities": rows,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "transport": {
            "provider": "GitHub_public_REST_API_via_gh_cli",
            "request_count": request_count,
            "response_digest_chain_sha256": stable_list_hash(digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "faults": sorted(set(faults)),
        "counters": counters,
        "maximum_inference": config.get("maximum_inference"),
    }


def api_json(resource: str, fields: dict[str, Any]) -> tuple[Any, str]:
    command = ["gh", "api", "-X", "GET", resource]
    for key, value in fields.items():
        command.extend(["-f", f"{key}={value}"])
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    raw = completed.stdout.encode()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def tracked_prior_repositories(config_path: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "configs/*.json"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    repositories: set[str] = set()
    for name in completed.stdout.splitlines():
        path = ROOT / name
        if path.resolve() == config_path.resolve():
            continue
        try:
            collect_repositories(json.loads(path.read_text(encoding="utf-8")), repositories)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(repositories)


def collect_repositories(value: Any, repositories: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "repository" and isinstance(child, str) and re.fullmatch(r"[^/\s]+/[^/\s]+", child):
                repositories.add(child)
            collect_repositories(child, repositories)
    elif isinstance(value, list):
        for child in value:
            collect_repositories(child, repositories)


def search_query(language: str, chronology: dict[str, Any]) -> str:
    return " ".join([
        "is:pr", "is:merged", "archived:false", f"language:{language}",
        f"created:{chronology.get('search_window_start_utc')}..{chronology.get('search_window_end_utc')}",
        f"merged:{chronology.get('search_window_start_utc')}..{chronology.get('search_window_end_utc')}",
    ])


def repository_from_url(url: str) -> str:
    prefix = "https://api.github.com/repos/"
    return url[len(prefix):] if url.startswith(prefix) else ""


def rank(seed: Any, repository: str, number: int) -> str:
    return hashlib.sha256(f"{seed}:{repository.lower()}#{number}".encode()).hexdigest()


def parse_time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def file_paths(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("filename") or "") for row in rows if row.get("filename")]


def is_source(path: str, language: str) -> bool:
    return path.lower().endswith(LANGUAGE_SUFFIXES[language]) and not is_excluded_or_lock(path)


def is_test(path: str) -> bool:
    lowered = path.lower()
    parts = set(filter(None, re.split(r"[/_.-]+", lowered)))
    return bool(parts & TEST_PARTS) or ".test." in lowered or ".spec." in lowered


def is_excluded_or_lock(path: str) -> bool:
    lowered = path.lower()
    parts = set(filter(None, lowered.split("/")))
    return bool(parts & EXCLUDED_PARTS) or lowered.endswith(LOCK_SUFFIXES)


def bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def stable_list_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def zero_counters() -> dict[str, int]:
    return {
        "public_metadata_requests": 0,
        "public_metadata_rows_selected": 0,
        "public_source_content_requests": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_inference_calls": 0,
        "hidden_evaluator_executions": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in (
        "trigger_state", "state", "metadata_selection_opened",
        "source_content_retrieval_opened", "candidate_packet_materialization_opened",
        "selected_repository_count", "faults", "counters",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
