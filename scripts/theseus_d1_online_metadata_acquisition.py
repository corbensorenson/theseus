#!/usr/bin/env python3
"""Acquire a complete post-snapshot GitHub metadata frame for fresh D1.

The transport is fail-closed and cannot make a network call before a green P4-v2r2
survivor.  It fetches public metadata only; source archives, tests, evaluators,
oracles, models, candidates, and controls remain outside this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_fresh_qualification_instrument as d1  # noqa: E402
import theseus_d1_source_selection as selection  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_source_selection.json"
REPORT = ROOT / "reports" / "theseus_d1_online_metadata_acquisition.json"
POLICY = "project_theseus_d1_online_metadata_acquisition_v1"
API_ROOT = "https://api.github.com"
PER_PAGE = 100


class GitHubPublicMetadataClient:
    def __init__(self) -> None:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Project-Theseus-D1-Metadata/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.request_count = 0
        self.response_digests: list[str] = []

    def get(self, path: str, parameters: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode(parameters or {})
        url = f"{API_ROOT}{path}{'?' + query if query else ''}"
        request = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 -- fixed GitHub API root.
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"github_metadata_http_error:{exc.code}:{path}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"github_metadata_transport_error:{path}") from exc
        self.request_count += 1
        self.response_digests.append(hashlib.sha256(body).hexdigest())
        return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=d1.relative(DEFAULT_CONFIG))
    parser.add_argument("--out", default=d1.relative(REPORT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = d1.resolve(args.config)
    report = preflight(config_path)
    if args.execute and report.get("network_acquisition_authorized") is True:
        try:
            report = acquire(config_path, client=GitHubPublicMetadataClient())
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "network_acquisition_authorized": False,
                "error": f"{type(exc).__name__}:{exc}"[:4000],
                "partial_metadata_frame_written": False,
            }
    d1.write_json(d1.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def preflight(
    config_path: Path = DEFAULT_CONFIG,
    *,
    disposition_override: dict[str, Any] | None = None,
    now_override: datetime | None = None,
) -> dict[str, Any]:
    config = d1.read_json(config_path)
    faults = selection.validate_config(config)
    instrument_path = d1.resolve(str(config.get("instrument") or ""))
    instrument_report = d1.build_report(
        instrument_path, disposition_override=disposition_override
    )
    temporal_guard, temporal_faults = selection.audit_temporal_guard(config)
    faults.extend(temporal_faults)
    activation_ready = instrument_report.get("source_acquisition_authorized") is True
    current = now_override or datetime.now(timezone.utc)
    observed = selection.parse_utc(
        str(temporal_guard.get("model_snapshot_observed_utc") or "")
    )
    intervals = complete_utc_intervals(observed, current) if observed else []
    if activation_ready and not intervals:
        faults.append("no_complete_post_snapshot_UTC_interval_available")
    return {
        "policy": POLICY,
        "created_utc": utc_now(current),
        "trigger_state": "GREEN" if activation_ready and not faults else "PAUSED",
        "faults": sorted(set(faults)),
        "config": d1.artifact(config_path),
        "instrument_audit": instrument_report,
        "temporal_contamination_guard": temporal_guard,
        "activation_state": instrument_report.get("activation_state"),
        "complete_interval_count": len(intervals),
        "complete_intervals": intervals,
        "network_acquisition_authorized": activation_ready and not faults,
        "network_calls": 0,
        "archive_fetches": 0,
        "parent_target_oracle_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "user_or_operator_gate": False,
        "maximum_inference": (
            "Preflight authorizes public GitHub metadata transport only after a "
            "green P4-v2r2 survivor. It does not authorize archives, execution, "
            "evaluators, candidates, controls, D1 consumption, training, serving, "
            "D2, or book promotion."
        ),
    }


def acquire(
    config_path: Path,
    *,
    client: GitHubPublicMetadataClient,
    disposition_override: dict[str, Any] | None = None,
    now_override: datetime | None = None,
) -> dict[str, Any]:
    before = preflight(
        config_path,
        disposition_override=disposition_override,
        now_override=now_override,
    )
    if before.get("network_acquisition_authorized") is not True:
        return before
    config = d1.read_json(config_path)
    instrument = d1.read_json(d1.resolve(str(config["instrument"])))
    activation = d1.mapping(instrument.get("activation"))
    disposition_path = d1.resolve(str(activation["p4_terminal_disposition"]))
    disposition = (
        disposition_override
        if disposition_override is not None
        else d1.read_json(disposition_path)
    )
    disposition_sha = (
        d1.stable_hash(disposition)
        if disposition_override is not None
        else hashlib.sha256(disposition_path.read_bytes()).hexdigest()
    )
    languages = sorted(
        selection.normalized_languages(
            d1.mapping(instrument.get("source_surface")).get(
                "programming_language_scope"
            )
        )
    )
    terms = [
        str(value).strip().lower()
        for value in d1.mapping(config.get("discovery_frame")).get(
            "maintenance_title_terms"
        )
        or []
        if str(value).strip()
    ]
    if not terms:
        raise ValueError("maintenance_title_terms_empty")
    rows_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    partitions: list[dict[str, Any]] = []
    pull_urls: set[str] = set()
    transport_exclusions: list[dict[str, str]] = []
    retrieved_utc = utc_now(now_override or datetime.now(timezone.utc))
    for interval in before["complete_intervals"]:
        for language in languages:
            for term in terms:
                partition, issues = fetch_search_partition(
                    client,
                    language=language,
                    title_term=term,
                    start_date=str(interval["start_date"]),
                    end_date=str(interval["end_date"]),
                )
                partitions.append(partition)
                for issue in issues:
                    pull_url = str(d1.mapping(issue.get("pull_request")).get("url") or "")
                    if pull_url.startswith(f"{API_ROOT}/repos/"):
                        pull_urls.add(pull_url)
    for pull_url in sorted(pull_urls):
        pull = client.get(pull_url.removeprefix(API_ROOT))
        row = fetch_candidate_row(
            client,
            pull,
            metadata_retrieved_utc=retrieved_utc,
        )
        if row:
            rows_by_identity[
                (str(row["repository"]).lower(), int(row["pull_request"]))
            ] = row
        else:
            transport_exclusions.append({
                "pull_request_api_url_sha256": hashlib.sha256(
                    pull_url.encode()
                ).hexdigest(),
                "reason": "required_public_metadata_incomplete",
            })
    ledger = {
        "policy": selection.LEDGER_POLICY,
        "state": "COMPLETE_QUERY_PARTITIONS_SEALED",
        "created_utc": retrieved_utc,
        "activation_disposition_sha256": disposition_sha,
        "acquisition_opened_utc": before["created_utc"],
        "frame_start_utc": min(
            (str(row["start_utc"]) for row in before["complete_intervals"]),
            default="",
        ),
        "frame_end_utc": max(
            (str(row["end_utc"]) for row in before["complete_intervals"]),
            default="",
        ),
        "query_partitions": partitions,
        "rows": sorted(
            rows_by_identity.values(),
            key=lambda row: (str(row["repository"]).lower(), int(row["pull_request"])),
        ),
        "transport_denominators": {
            "search_issue_occurrences": sum(
                int(row["retrieved_issue_count"]) for row in partitions
            ),
            "unique_pull_request_urls": len(pull_urls),
            "normalized_rows": len(rows_by_identity),
            "metadata_incomplete_exclusions": len(transport_exclusions),
        },
        "transport_exclusions": transport_exclusions,
        "transport": {
            "provider": "GitHub_public_REST_API",
            "request_count": client.request_count,
            "response_digest_chain_sha256": d1.stable_hash(client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "boundaries": {
            "archive_fetches": 0,
            "parent_target_oracle_or_evaluator_executions": 0,
            "candidate_or_control_calls": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "training_rows_written": 0,
        },
        "maximum_inference": (
            "Complete metadata transport over the declared query partitions only. "
            "No archive, test, evaluator, model, candidate, control, D1, training, "
            "serving, D2, or book evidence."
        ),
    }
    ledger_faults = selection.audit_ledger(
        ledger,
        config,
        disposition=disposition,
        disposition_sha256=disposition_sha,
    )
    if ledger_faults:
        raise RuntimeError(f"acquired_metadata_ledger_invalid:{sorted(ledger_faults)}")
    ledger_path = d1.resolve(str(config["candidate_metadata_ledger"]))
    write_json_atomic(ledger_path, ledger)
    selection_report = selection.build_report(
        config_path,
        disposition_override=disposition_override,
        ledger_override=ledger,
    )
    return {
        **before,
        "trigger_state": "GREEN",
        "network_acquisition_authorized": False,
        "network_calls": client.request_count,
        "metadata_ledger": d1.artifact(ledger_path),
        "metadata_row_count": len(ledger["rows"]),
        "query_partition_count": len(partitions),
        "selection_preflight": {
            "trigger_state": selection_report.get("trigger_state"),
            "registry_ready": selection_report.get("registry_ready"),
            "faults": selection_report.get("faults"),
        },
        "partial_metadata_frame_written": False,
    }


def complete_utc_intervals(
    snapshot_observed: datetime,
    current: datetime,
) -> list[dict[str, str]]:
    if snapshot_observed.tzinfo is None or current.tzinfo is None:
        raise ValueError("UTC interval inputs must be timezone-aware")
    start_date = snapshot_observed.astimezone(timezone.utc).date()
    last_complete_date = current.astimezone(timezone.utc).date() - timedelta(days=1)
    if last_complete_date < start_date:
        return []
    intervals: list[dict[str, str]] = []
    cursor = start_date
    while cursor <= last_complete_date:
        day_start = datetime.combine(cursor, datetime.min.time(), tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        effective_start = max(snapshot_observed.astimezone(timezone.utc), day_start)
        intervals.append({
            "start_date": cursor.isoformat(),
            "end_date": cursor.isoformat(),
            "start_utc": utc_now(effective_start),
            "end_utc": utc_now(day_end),
        })
        cursor += timedelta(days=1)
    return intervals


def fetch_search_partition(
    client: GitHubPublicMetadataClient,
    *,
    language: str,
    title_term: str,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = (
        f"is:pr is:merged {title_term} in:title "
        f"language:{language} merged:{start_date}..{end_date}"
    )
    first = client.get(
        "/search/issues",
        {"q": query, "sort": "created", "order": "asc", "per_page": PER_PAGE, "page": 1},
    )
    total = int(d1.mapping(first).get("total_count") or 0)
    if d1.mapping(first).get("incomplete_results") is True:
        raise RuntimeError(
            f"github_search_partition_reported_incomplete:{language}:{title_term}:{start_date}"
        )
    if total > 1000:
        raise RuntimeError(
            f"github_search_partition_exceeds_complete_API_window:{language}:{title_term}:{start_date}:{total}"
        )
    issues = selection.dictionaries(d1.mapping(first).get("items"))
    pages = max(1, math.ceil(total / PER_PAGE))
    response_projection: list[Any] = [first]
    for page in range(2, pages + 1):
        payload = client.get(
            "/search/issues",
            {"q": query, "sort": "created", "order": "asc", "per_page": PER_PAGE, "page": page},
        )
        response_projection.append(payload)
        if d1.mapping(payload).get("incomplete_results") is True:
            raise RuntimeError(
                f"github_search_partition_page_incomplete:{language}:{title_term}:{start_date}:{page}"
            )
        issues.extend(selection.dictionaries(d1.mapping(payload).get("items")))
    if len(issues) != total:
        raise RuntimeError(
            f"github_search_partition_count_mismatch:{language}:{title_term}:{len(issues)}/{total}"
        )
    return {
        "id": f"{language}:{title_term}:{start_date}:{end_date}",
        "language": language,
        "title_term": title_term,
        "start_date": start_date,
        "end_date": end_date,
        "reported_total_count": total,
        "retrieved_issue_count": len(issues),
        "page_count": pages,
        "complete": True,
        "raw_response_sha256": d1.stable_hash(response_projection),
    }, issues


def fetch_candidate_row(
    client: GitHubPublicMetadataClient,
    pull: dict[str, Any],
    *,
    metadata_retrieved_utc: str,
) -> dict[str, Any] | None:
    if not pull.get("merged_at") or not pull.get("merge_commit_sha"):
        return None
    base = d1.mapping(pull.get("base"))
    repo = d1.mapping(base.get("repo"))
    repository = str(repo.get("full_name") or "")
    number = int(pull.get("number") or 0)
    if not repository or number < 1:
        return None
    repository_payload = client.get(f"/repos/{repository}")
    license_info = d1.mapping(repository_payload.get("license"))
    merge_revision = str(pull.get("merge_commit_sha") or "").lower()
    commit = client.get(f"/repos/{repository}/commits/{merge_revision}")
    parents = selection.dictionaries(commit.get("parents"))
    if not parents:
        return None
    parent_revision = str(parents[0].get("sha") or "").lower()
    changed_files = int(pull.get("changed_files") or 0)
    paths: list[str] = []
    for page in range(1, max(1, math.ceil(changed_files / PER_PAGE)) + 1):
        payload = client.get(
            f"/repos/{repository}/pulls/{number}/files",
            {"per_page": PER_PAGE, "page": page},
        )
        paths.extend(
            str(row.get("filename") or "")
            for row in selection.dictionaries(payload)
            if str(row.get("filename") or "")
        )
    if len(paths) != changed_files:
        return None
    return {
        "repository": repository,
        "repository_url": f"https://github.com/{repository}",
        "license_spdx": str(license_info.get("spdx_id") or ""),
        "primary_language": str(repository_payload.get("language") or ""),
        "pull_request": number,
        "pull_request_url": f"https://github.com/{repository}/pull/{number}",
        "pull_request_title": str(pull.get("title") or ""),
        "merged_utc": str(pull.get("merged_at") or ""),
        "parent_revision": parent_revision,
        "target_revision": merge_revision,
        "merge_revision": merge_revision,
        "changed_paths": sorted(set(paths)),
        "metadata_retrieved_utc": metadata_retrieved_utc,
        "metadata_only_selection": True,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def utc_now(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "activation_state": report.get("activation_state"),
        "network_acquisition_authorized": report.get(
            "network_acquisition_authorized"
        ),
        "network_calls": report.get("network_calls"),
        "metadata_row_count": report.get("metadata_row_count"),
        "query_partition_count": report.get("query_partition_count"),
        "faults": report.get("faults"),
        "error": report.get("error", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
