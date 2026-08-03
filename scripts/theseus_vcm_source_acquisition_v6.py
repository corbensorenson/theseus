#!/usr/bin/env python3
"""GraphQL-batched metadata transport for the frozen VCM source selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_acquisition_v6.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_acquisition_v6.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_source_acquisition_v6_checkpoint.json"
GRAPHQL_QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    __typename
    ... on PullRequest {
      id
      number
      url
      state
      isDraft
      createdAt
      mergedAt
      additions
      deletions
      changedFiles
      baseRefOid
      headRefOid
      mergeCommit { oid }
      author { login }
      repository {
        nameWithOwner
        isFork
        isArchived
        isDisabled
        stargazerCount
        primaryLanguage { name }
        licenseInfo { spdxId }
      }
      files(first: 13) { nodes { path changeType } }
      commits(last: 1) { nodes { commit { oid committedDate } } }
    }
  }
}
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=p2a.rel(DEFAULT_OUT))
    parser.add_argument("--checkpoint", default=p2a.rel(DEFAULT_CHECKPOINT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    report = preflight(config_path)
    config = p2a.read_json(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        not_before = v1.parse_time(
            p2a.mapping(config.get("graphql_transport")).get("execution_not_before_utc")
        )
        if datetime.now(timezone.utc) < not_before:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "GITHUB_RATE_WINDOW_NOT_RECOVERED_ZERO_REQUESTS",
                "faults": ["execution_not_before_utc_not_reached"],
            }
        else:
            retry_policy = p2a.mapping(config.get("transport_retry_policy"))
            ledger = v5.RequestLedger(
                p2a.resolve(args.checkpoint), config_path, retry_policy
            )
            try:
                report = acquire(config_path, ledger, retry_policy)
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
                    "state": "GRAPHQL_METADATA_TRANSPORT_PAUSED_NO_SOURCE_EXPOSURE",
                    "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                }
            report = v5.attach_accounting(report, ledger)
            ledger.finalize(
                str(report.get("state") or "UNKNOWN"),
                int(report.get("selected_repository_count") or 0),
            )
            report["transport_retry_accounting"] = ledger.summary()
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    report = v5.preflight(config_path)
    config = p2a.read_json(config_path)
    transport = p2a.mapping(config.get("graphql_transport"))
    qualification = p2a.mapping(transport.get("live_schema_qualification"))
    faults = p2a.strings(report.get("faults"))
    if (
        transport.get("rest_search_population_unchanged") is not True
        or int(transport.get("node_batch_size") or 0) != 40
        or int(transport.get("maximum_parallel_graphql_requests") or 0) != 1
        or transport.get("rank_order_consumption_required") is not True
        or transport.get("query_may_request_body_patch_or_review_content") is not False
        or transport.get("candidate_identities_in_checkpoint") is not False
        or transport.get("source_content_retrieval_authorized") is not False
        or qualification.get("required_schema_fields_missing") != []
        or qualification.get("consumed_pr_query_shape_green") is not True
        or int(qualification.get("fresh_candidate_queries") or 0) != 0
        or "body" in GRAPHQL_QUERY.lower()
        or "patch" in GRAPHQL_QUERY.lower()
        or "reviews" in GRAPHQL_QUERY.lower()
    ):
        faults.append("graphql_transport_contract_invalid")
    try:
        if v1.parse_time(transport.get("execution_not_before_utc")) <= v1.parse_time(
            p2a.mapping(config.get("chronology")).get("search_window_end_utc")
        ):
            faults.append("rate_window_boundary_invalid")
    except (TypeError, ValueError):
        faults.append("rate_window_boundary_invalid")
    report["faults"] = sorted(set(faults))
    report["trigger_state"] = "GREEN" if not faults else "RED"
    report["state"] = (
        "METADATA_SELECTION_V6_GRAPHQL_BATCH_LIVE_SCHEMA_QUALIFIED"
        if not faults
        else "INVALID_PREFLIGHT"
    )
    report["live_schema_qualification"] = qualification
    return report


def acquire(
    config_path: Path,
    ledger: v5.RequestLedger,
    retry_policy: dict[str, Any],
) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    search = p2a.mapping(config.get("search"))
    selection = p2a.mapping(config.get("selection"))
    chronology = p2a.mapping(config.get("chronology"))
    panels = p2a.mapping(config.get("panels"))
    transport = p2a.mapping(config.get("graphql_transport"))
    denylist = set(v1.tracked_prior_repositories(config_path))
    rest_client = v5.RetryingClient(v1.api_json, ledger, retry_policy)
    graphql_client = v5.RetryingClient(graphql_api, ledger, retry_policy)
    response_digests: list[str] = []
    search_rows: dict[tuple[str, int], dict[str, Any]] = {}
    pages = int(search.get("pages_per_language") or 0)
    for language in p2a.strings(search.get("languages")):
        for page in range(1, pages + 1):
            payload, digest = rest_client.call(
                "search/issues",
                {
                    "q": v1.search_query(language, chronology),
                    "sort": search.get("sort"),
                    "order": search.get("order"),
                    "per_page": search.get("items_per_page"),
                    "page": page,
                },
            )
            response_digests.append(digest)
            for item in p2a.dicts(p2a.mapping(payload).get("items")):
                repository = v1.repository_from_url(
                    str(item.get("repository_url") or "")
                )
                number = int(item.get("number") or 0)
                node_id = str(item.get("node_id") or "")
                if repository and number and node_id:
                    search_rows[(repository, number)] = {
                        "repository": repository,
                        "pull_request": number,
                        "query_language": language,
                        "node_id": node_id,
                        "title_sha256": hashlib.sha256(
                            str(item.get("title") or "").encode()
                        ).hexdigest(),
                        "rank": v1.rank(
                            selection.get("selection_seed"), repository, number
                        ),
                    }
    selected_by_language: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: dict[str, int] = {}
    batch_size = int(transport.get("node_batch_size") or 0)
    for language in p2a.strings(search.get("languages")):
        control_quota = int(
            p2a.mapping(
                p2a.mapping(panels.get("control_qualification")).get(
                    "language_quotas"
                )
            ).get(language)
            or 0
        )
        claim_quota = int(
            p2a.mapping(
                p2a.mapping(panels.get("claim")).get("language_quotas")
            ).get(language)
            or 0
        )
        total_quota = control_quota + claim_quota
        candidates = sorted(
            (
                row
                for row in search_rows.values()
                if row["query_language"] == language
            ),
            key=lambda row: row["rank"],
        )
        admitted: list[dict[str, Any]] = []
        used_repositories: set[str] = set()
        offset = 0
        while offset < len(candidates) and len(admitted) < total_quota:
            batch: list[dict[str, Any]] = []
            batch_repositories: set[str] = set()
            while offset < len(candidates) and len(batch) < batch_size:
                candidate = candidates[offset]
                offset += 1
                repository = candidate["repository"]
                if (
                    repository in denylist
                    or repository in used_repositories
                    or repository in batch_repositories
                ):
                    v1.bump(rejection_counts, "prior_or_duplicate_repository")
                    continue
                batch.append(candidate)
                batch_repositories.add(repository)
            if not batch:
                continue
            payload, digest = graphql_client.call(
                "graphql:nodes", {"ids": [row["node_id"] for row in batch]}
            )
            response_digests.append(digest)
            nodes = {
                str(node.get("id") or ""): node
                for node in p2a.dicts(
                    p2a.mapping(p2a.mapping(payload).get("data")).get("nodes")
                )
            }
            for candidate in batch:
                row, reasons = qualify_node(nodes.get(candidate["node_id"]), candidate, config)
                if reasons:
                    for reason in reasons:
                        v1.bump(rejection_counts, reason)
                    continue
                used_repositories.add(candidate["repository"])
                admitted.append(row)
                if len(admitted) >= total_quota:
                    break
        if len(admitted) != total_quota:
            return v1.terminal_report(
                before,
                config,
                int(ledger.summary()["logical_request_count"] or 0),
                response_digests,
                [],
                rejection_counts,
                [f"language_quota_unfilled:{language}:{len(admitted)}/{total_quota}"],
            )
        panel_ranked = sorted(
            admitted,
            key=lambda row: v1.rank(
                panels.get("assignment_seed"),
                row["repository"],
                row["pull_request"],
            ),
        )
        controls = {
            (row["repository"], row["pull_request"])
            for row in panel_ranked[:control_quota]
        }
        for row in admitted:
            identity = (row["repository"], row["pull_request"])
            row["panel"] = (
                "control_qualification" if identity in controls else "claim"
            )
            row["panel_assignment_sha256"] = v1.rank(
                panels.get("assignment_seed"),
                row["repository"],
                row["pull_request"],
            )
        selected_by_language[language] = admitted
    rows = sorted(
        [row for group in selected_by_language.values() for row in group],
        key=lambda row: (
            row["panel"],
            row["query_language"],
            row["selection_rank_sha256"],
        ),
    )
    repositories = [row["repository"] for row in rows]
    faults: list[str] = []
    if len(rows) != 62 or len(set(repositories)) != 62 or set(repositories) & denylist:
        faults.append("terminal_source_disjointness_invalid")
    report = v1.terminal_report(
        before,
        config,
        int(ledger.summary()["logical_request_count"] or 0),
        response_digests,
        rows,
        rejection_counts,
        faults,
    )
    report["metadata_transport"] = {
        "rest_search_requests": pages * len(p2a.strings(search.get("languages"))),
        "graphql_node_batching": True,
        "graphql_node_batch_size": batch_size,
        "maximum_parallel_graphql_requests": 1,
        "body_patch_or_review_content_requested": False,
    }
    return report


def graphql_api(resource: str, fields: dict[str, Any]) -> tuple[Any, str]:
    if resource != "graphql:nodes":
        raise ValueError("graphql_resource_invalid")
    ids = p2a.strings(fields.get("ids"))
    if not ids or len(ids) > 40:
        raise ValueError("graphql_node_batch_invalid")
    command = ["gh", "api", "graphql", "-f", f"query={GRAPHQL_QUERY}"]
    for node_id in ids:
        command.extend(["-F", f"ids[]={node_id}"])
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    raw = completed.stdout.encode()
    payload = json.loads(raw)
    if p2a.dicts(p2a.mapping(payload).get("errors")):
        raise RuntimeError("graphql_response_errors")
    return payload, hashlib.sha256(raw).hexdigest()


def qualify_node(
    node: dict[str, Any] | None,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    repository = str(candidate.get("repository") or "")
    number = int(candidate.get("pull_request") or 0)
    base_row = {
        "opaque_source_id": hashlib.sha256(
            f"vcm-source:{repository}#{number}".encode()
        ).hexdigest(),
        "repository": repository,
        "pull_request": number,
        "query_language": candidate.get("query_language"),
        "selection_rank_sha256": candidate.get("rank"),
        "title_sha256": candidate.get("title_sha256"),
        "candidate_content_retrieved": False,
        "candidate_packet_materialized": False,
    }
    if not isinstance(node, dict) or node.get("__typename") != "PullRequest":
        return {**base_row, "metadata_qualified": False}, [
            "candidate_graphql_node_unavailable"
        ]
    repository_node = p2a.mapping(node.get("repository"))
    file_nodes = p2a.dicts(p2a.mapping(node.get("files")).get("nodes"))
    commit_nodes = p2a.dicts(p2a.mapping(node.get("commits")).get("nodes"))
    commit_node = (
        p2a.mapping(p2a.mapping(commit_nodes[-1]).get("commit"))
        if commit_nodes
        else {}
    )
    files = [
        {
            "filename": row.get("path"),
            "status": graphql_change_status(row.get("changeType")),
        }
        for row in file_nodes
    ]
    pr = {
        "state": "closed" if node.get("mergedAt") else str(node.get("state") or "").lower(),
        "merged_at": node.get("mergedAt"),
        "created_at": node.get("createdAt"),
        "draft": node.get("isDraft"),
        "changed_files": node.get("changedFiles"),
        "additions": node.get("additions"),
        "deletions": node.get("deletions"),
        "user": {"login": p2a.mapping(node.get("author")).get("login")},
        "base": {"sha": node.get("baseRefOid")},
        "head": {"sha": node.get("headRefOid")},
        "merge_commit_sha": p2a.mapping(node.get("mergeCommit")).get("oid"),
    }
    repo = {
        "fork": repository_node.get("isFork"),
        "archived": repository_node.get("isArchived"),
        "disabled": repository_node.get("isDisabled"),
        "language": p2a.mapping(repository_node.get("primaryLanguage")).get("name"),
        "stargazers_count": repository_node.get("stargazerCount"),
        "license": {
            "spdx_id": p2a.mapping(repository_node.get("licenseInfo")).get("spdxId")
        },
    }
    head_commit = {
        "commit": {"committer": {"date": commit_node.get("committedDate")}}
    }
    reasons = v1.metadata_rejection_reasons(
        candidate, pr, repo, files, head_commit, config
    )
    if repository_node.get("nameWithOwner") != repository or int(node.get("number") or 0) != number:
        reasons = sorted(set([*reasons, "graphql_identity_mismatch"]))
    if not commit_nodes:
        reasons = sorted(set([*reasons, "head_commit_metadata_unavailable"]))
    elif commit_node.get("oid") != node.get("headRefOid"):
        reasons = sorted(set([*reasons, "head_commit_identity_mismatch"]))
    paths = v1.file_paths(files)
    row = {
        **base_row,
        "pull_request_url": node.get("url"),
        "created_utc": node.get("createdAt"),
        "merged_utc": node.get("mergedAt"),
        "base_revision": node.get("baseRefOid"),
        "head_revision": node.get("headRefOid"),
        "merge_revision": p2a.mapping(node.get("mergeCommit")).get("oid"),
        "head_commit_utc": commit_node.get("committedDate"),
        "head_chronology_source": "graphql_pull_request_commit_connection",
        "license_spdx": p2a.mapping(repository_node.get("licenseInfo")).get("spdxId"),
        "repository_stars": repository_node.get("stargazerCount"),
        "changed_file_count": node.get("changedFiles"),
        "changed_line_count": int(node.get("additions") or 0)
        + int(node.get("deletions") or 0),
        "source_paths": sorted(
            path
            for path in paths
            if v1.is_source(path, str(candidate.get("query_language") or ""))
            and not v1.is_test(path)
        ),
        "verifier_paths": sorted(path for path in paths if v1.is_test(path)),
        "metadata_qualified": not reasons,
    }
    return row, reasons


def graphql_change_status(value: Any) -> str:
    mapping = {
        "ADDED": "added",
        "DELETED": "removed",
        "RENAMED": "renamed",
        "COPIED": "copied",
    }
    return mapping.get(str(value or "").upper(), "modified")


if __name__ == "__main__":
    raise SystemExit(main())
