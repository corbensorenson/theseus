#!/usr/bin/env python3
"""Recompute public metadata and license eligibility for the S3 adequacy panel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_d1_online_metadata_acquisition as github_metadata  # noqa: E402
import theseus_semantic_ir_production_adequacy as adequacy  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_source_candidates_v2.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_metadata.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_acquisition_v1"
CANDIDATE_POLICY_V1 = "project_theseus_semantic_ir_production_adequacy_source_candidates_v1"
CANDIDATE_POLICY_V2 = "project_theseus_semantic_ir_production_adequacy_source_candidates_v2"
HEX40 = set("0123456789abcdef")


class Client(Protocol):
    request_count: int
    response_digests: list[str]

    def get(self, path: str, parameters: dict[str, Any] | None = None) -> Any: ...


class GitHubCliMetadataClient:
    """Read public GitHub JSON through the authenticated local gh transport."""

    def __init__(self) -> None:
        self.request_count = 0
        self.response_digests: list[str] = []

    def get(self, path: str, parameters: dict[str, Any] | None = None) -> Any:
        command = ["gh", "api", "-X", "GET", path]
        for key, value in sorted((parameters or {}).items()):
            command.extend(["-f", f"{key}={value}"])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=False,
            timeout=120,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"github_cli_metadata_error:{path}:{detail}")
        self.request_count += 1
        self.response_digests.append(hashlib.sha256(completed.stdout).hexdigest())
        return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=relative(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--transport", choices=("gh_cli", "urllib"), default="gh_cli")
    args = parser.parse_args()
    candidates_path = resolve(args.candidates)
    report = preflight(candidates_path)
    if args.execute and report["trigger_state"] == "GREEN":
        try:
            client: Client = (
                GitHubCliMetadataClient()
                if args.transport == "gh_cli"
                else github_metadata.GitHubPublicMetadataClient()
            )
            report = acquire(candidates_path, client=client)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "faults": sorted(set(report["faults"] + [f"{type(exc).__name__}:{exc}"[:4000]])),
                "partial_selection_admitted": False,
            }
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(path: Path = DEFAULT_CANDIDATES) -> dict[str, Any]:
    candidates = read_json(path)
    adequacy_path = resolve(str(candidates.get("adequacy_preregistration") or ""))
    config = read_json(adequacy_path) if adequacy_path.is_file() else {}
    faults = audit_candidate_registry(candidates, config)
    if not adequacy_path.is_file() or sha256_file(adequacy_path) != str(
        candidates.get("adequacy_preregistration_sha256") or ""
    ):
        faults.append("adequacy_preregistration_binding_invalid")
    adequacy_report = adequacy.audit(adequacy_path) if adequacy_path.is_file() else {}
    if adequacy_report.get("trigger_state") != "GREEN":
        faults.append("adequacy_preregistration_not_green")
    return {
        "policy": POLICY,
        "stage": "metadata_preflight",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "candidate_registry": artifact(path),
        "adequacy_preregistration_audit": {
            "trigger_state": adequacy_report.get("trigger_state"),
            "config_sha256": adequacy_report.get("config_sha256"),
            "auditor_sha256": adequacy_report.get("auditor_sha256"),
        },
        "selection_admitted": False,
        "partial_selection_admitted": False,
        "counters": zero_counters(),
        "maximum_inference": "A GREEN preflight permits public GitHub metadata and license retrieval only. It does not permit source archives, evaluator execution, candidate generation, controls, model calls, D1, D2, training, serving, or book support.",
    }


def acquire(path: Path, *, client: Client) -> dict[str, Any]:
    before = preflight(path)
    if before["trigger_state"] != "GREEN":
        return before
    candidates = read_json(path)
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for selected in dictionaries(candidates.get("candidates")):
        row, row_faults = fetch_and_audit_candidate(client, selected)
        rows.append(row)
        faults.extend(f"candidate_{selected.get('index')}:{fault}" for fault in row_faults)
    admitted = not faults and len(rows) == 18
    counters = zero_counters()
    counters["network_metadata_calls"] = client.request_count
    return {
        **before,
        "stage": "metadata_acquisition_complete",
        "trigger_state": "GREEN" if admitted else "RED",
        "faults": sorted(set(faults)),
        "selection_admitted": admitted,
        "partial_selection_admitted": False,
        "rows": rows,
        "transport": {
            "provider": "GitHub_public_REST_API",
            "request_count": client.request_count,
            "response_digest_chain_sha256": adequacy.stable_hash(client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "counters": counters,
        "maximum_inference": "A GREEN report proves that the 18 preselected public PR identities, first-parent revisions, changed-source membership, post-snapshot timing, distinct-repository rule, and license files were independently recomputed before source materialization. It is metadata eligibility only, not evaluator, mechanics, capability, claim, D1, D2, or book evidence.",
    }


def fetch_and_audit_candidate(
    client: Client, selected: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    repository = str(selected.get("repository") or "")
    number = integer(selected.get("pull_request"))
    pull = mapping(client.get(f"/repos/{repository}/pulls/{number}"))
    repo = mapping(client.get(f"/repos/{repository}"))
    merge_revision = str(pull.get("merge_commit_sha") or "").lower()
    commit = mapping(client.get(f"/repos/{repository}/commits/{merge_revision}"))
    parents = dictionaries(commit.get("parents"))
    changed_count = integer(pull.get("changed_files"))
    files: list[dict[str, Any]] = []
    for page in range(1, max(1, math.ceil(changed_count / 100)) + 1):
        files.extend(dictionaries(client.get(
            f"/repos/{repository}/pulls/{number}/files",
            {"per_page": 100, "page": page},
        )))
    license_payload = mapping(client.get(f"/repos/{repository}/license"))
    encoded = str(license_payload.get("content") or "").replace("\n", "")
    try:
        license_bytes = base64.b64decode(encoded, validate=True)
    except ValueError:
        license_bytes = b""
    file_paths = {str(row.get("filename") or "") for row in files}
    selected_paths = [str(value) for value in selected.get("selected_source_paths") or []]
    declared_license = str(selected.get("declared_license_spdx") or "")
    api_license = str(mapping(repo.get("license")).get("spdx_id") or "")
    api_license_path = str(license_payload.get("path") or "")
    allowed_license_paths = {str(value) for value in selected.get("license_paths") or []}
    faults: list[str] = []
    checks = (
        (str(mapping(mapping(pull.get("base")).get("repo")).get("full_name") or "").lower() == repository.lower(), "repository_identity_mismatch"),
        (integer(pull.get("number")) == number, "pull_request_number_mismatch"),
        (str(pull.get("title") or "") == str(selected.get("title") or ""), "title_mismatch"),
        (str(pull.get("merged_at") or "") == str(selected.get("merged_utc") or ""), "merged_utc_mismatch"),
        (merge_revision == str(selected.get("merge_revision") or ""), "merge_revision_mismatch"),
        (len(parents) >= 1 and valid_sha(str(parents[0].get("sha") or "")), "first_parent_missing"),
        (len(files) == changed_count and changed_count >= 1, "changed_file_inventory_incomplete"),
        (set(selected_paths).issubset(file_paths), "selected_source_path_not_changed"),
        (api_license_path in allowed_license_paths, "license_path_mismatch"),
        (len(license_bytes) >= 100, "license_content_missing"),
        (api_license in {declared_license, "NOASSERTION"}, "declared_license_conflicts_with_api"),
    )
    faults.extend(name for passed, name in checks if not passed)
    return {
        "index": selected.get("index"),
        "repository": repository,
        "pull_request": number,
        "pull_request_url": str(pull.get("html_url") or ""),
        "title": str(pull.get("title") or ""),
        "merged_utc": str(pull.get("merged_at") or ""),
        "parent_revision": str(parents[0].get("sha") or "").lower() if parents else "",
        "target_revision": merge_revision,
        "changed_file_count": changed_count,
        "changed_file_inventory_sha256": adequacy.stable_hash([
            {
                "filename": str(row.get("filename") or ""),
                "status": str(row.get("status") or ""),
                "additions": integer(row.get("additions")),
                "deletions": integer(row.get("deletions")),
                "changes": integer(row.get("changes")),
                "patch_sha256": hashlib.sha256(str(row.get("patch") or "").encode()).hexdigest(),
            }
            for row in files
        ]),
        "selected_source_paths": selected_paths,
        "stratum": selected.get("stratum"),
        "license": {
            "declared_spdx": declared_license,
            "api_spdx": api_license,
            "path": api_license_path,
            "content_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "content_bytes": len(license_bytes),
            "verified": len(license_bytes) >= 100 and api_license_path in allowed_license_paths,
        },
        "metadata_faults": faults,
    }, faults


def audit_candidate_registry(
    registry: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    faults: list[str] = []
    policy = registry.get("policy")
    if policy not in {CANDIDATE_POLICY_V1, CANDIDATE_POLICY_V2}:
        faults.append("candidate_registry_policy_invalid")
    expected_state = (
        "FIXED_BEFORE_SOURCE_MATERIALIZATION_OR_EVALUATOR_EXECUTION"
        if policy == CANDIDATE_POLICY_V1
        else "AMENDED_AFTER_SOURCE_ONLY_FAILURE_BEFORE_EVALUATOR_OR_MODEL"
    )
    if registry.get("state") != expected_state:
        faults.append("candidate_registry_state_invalid")
    rows = dictionaries(registry.get("candidates"))
    expected = integer(mapping(config.get("competence_design")).get("panel_size"))
    if len(rows) != expected or integer(registry.get("task_count")) != expected:
        faults.append("candidate_count_mismatch")
    if [integer(row.get("index")) for row in rows] != list(range(1, expected + 1)):
        faults.append("candidate_indices_invalid")
    repositories = [str(row.get("repository") or "").lower() for row in rows]
    if len(set(repositories)) != len(repositories):
        faults.append("candidate_repositories_not_distinct")
    excluded = {str(value).lower() for value in config.get("excluded_repositories") or []}
    if excluded.intersection(repositories):
        faults.append("candidate_repository_overlaps_prior_theseus")
    snapshot = str(mapping(config.get("temporal_guard")).get("model_snapshot_observed_utc") or "")
    allowed = set(mapping(config.get("source_eligibility")).get("license_allowlist") or [])
    expected_strata = {
        str(row.get("id")): integer(row.get("task_count"))
        for row in dictionaries(mapping(config.get("competence_design")).get("strata"))
    }
    observed = {key: 0 for key in expected_strata}
    for row in rows:
        if str(row.get("merged_utc") or "") <= snapshot:
            faults.append("candidate_not_strictly_post_snapshot")
        if not valid_sha(str(row.get("merge_revision") or "")):
            faults.append("candidate_merge_revision_invalid")
        if str(row.get("declared_license_spdx") or "") not in allowed:
            faults.append("candidate_license_not_allowlisted")
        if not row.get("selected_source_paths"):
            faults.append("candidate_source_paths_empty")
        stratum = str(row.get("stratum") or "")
        if stratum not in observed:
            faults.append("candidate_stratum_invalid")
        else:
            observed[stratum] += 1
    if observed != expected_strata:
        faults.append("candidate_strata_not_balanced")
    if policy == CANDIDATE_POLICY_V2:
        faults.extend(audit_source_only_amendment(registry))
    else:
        boundaries = mapping(registry.get("boundaries"))
        for key, value in boundaries.items():
            if key == "user_or_operator_gate":
                if value is not False:
                    faults.append("user_gate_present")
            elif integer(value) != 0:
                faults.append(f"preselection_counter_nonzero:{key}")
    return faults


def audit_source_only_amendment(registry: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    amendment = mapping(registry.get("source_only_amendment"))
    predecessor_path = resolve(str(amendment.get("predecessor_registry") or ""))
    failure_path = resolve(str(amendment.get("trigger_failure_report") or ""))
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != str(amendment.get("predecessor_registry_sha256") or "")
    ):
        faults.append("amendment_predecessor_binding_invalid")
        predecessor = {}
    else:
        predecessor = read_json(predecessor_path)
    if (
        not failure_path.is_file()
        or sha256_file(failure_path)
        != str(amendment.get("trigger_failure_report_sha256") or "")
    ):
        faults.append("amendment_failure_binding_invalid")
        failure = {}
    else:
        failure = read_json(failure_path)
    if failure.get("trigger_state") != "RED" or failure.get("faults") != [
        "task_4:selected_source_bytes_unchanged"
    ]:
        faults.append("amendment_trigger_failure_invalid")
    failure_counters = mapping(failure.get("counters"))
    prohibited = (
        "parent_target_evaluator_executions",
        "candidate_or_control_calls",
        "local_model_calls",
        "external_inference_calls",
        "teacher_calls",
        "training_rows_written",
        "D1_cases_consumed",
        "D2_cases_consumed",
    )
    if any(integer(failure_counters.get(key)) != 0 for key in prohibited):
        faults.append("amendment_trigger_crossed_prohibited_boundary")
    predecessor_rows = dictionaries(predecessor.get("candidates"))
    current_rows = dictionaries(registry.get("candidates"))
    changes: list[dict[str, Any]] = []
    for before, after in zip(predecessor_rows, current_rows, strict=False):
        if before != after:
            changes.append({"before": before, "after": after})
    expected_change = mapping(amendment.get("selected_path_change"))
    if len(predecessor_rows) != len(current_rows) or len(changes) != 1:
        faults.append("amendment_scope_not_single_candidate")
    elif (
        integer(changes[0]["before"].get("index")) != integer(expected_change.get("index"))
        or changes[0]["before"].get("selected_source_paths") != expected_change.get("from")
        or changes[0]["after"].get("selected_source_paths") != expected_change.get("to")
        or {
            key: value
            for key, value in changes[0]["before"].items()
            if key != "selected_source_paths"
        }
        != {
            key: value
            for key, value in changes[0]["after"].items()
            if key != "selected_source_paths"
        }
    ):
        faults.append("amendment_scope_mismatch")
    boundaries = mapping(registry.get("boundaries"))
    if boundaries.get("user_or_operator_gate") is not False:
        faults.append("user_gate_present")
    if integer(boundaries.get("network_source_calls")) != 78:
        faults.append("amendment_source_call_count_invalid")
    if integer(boundaries.get("source_archives_materialized")) != 34:
        faults.append("amendment_archive_receipt_count_invalid")
    if any(integer(boundaries.get(key)) != 0 for key in prohibited):
        faults.append("amendment_boundary_invalid")
    return faults


def zero_counters() -> dict[str, int]:
    return {
        "network_metadata_calls": 0,
        "source_archives_fetched": 0,
        "parent_target_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def valid_sha(value: str) -> bool:
    return len(value) == 40 and set(value.lower()) <= HEX40


def artifact(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value or [] if isinstance(row, dict)]


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": report.get("stage"),
        "trigger_state": report.get("trigger_state"),
        "faults": report.get("faults"),
        "selection_admitted": report.get("selection_admitted"),
        "row_count": len(dictionaries(report.get("rows"))),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
