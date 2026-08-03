#!/usr/bin/env python3
"""Materialize the frozen VCM source panels without executing untrusted code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import theseus_assistant_p2a as p2a
import theseus_semantic_ir_production_adequacy_materialization as archives
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_source_materialization.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_source_materialization.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_source_materialization_checkpoint.json"
POLICY = "project_theseus_vcm_source_materialization_v1"
TITLE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { title }
  }
}
""".strip()


class SourceLedger(v5.RequestLedger):
    def __init__(self, path: Path, config_path: Path, policy: dict[str, Any]) -> None:
        super().__init__(path, config_path, policy)
        with self.lock:
            self.data["policy"] = "project_theseus_vcm_source_materialization_checkpoint_v1"
            self.data["state"] = "RUNNING_PUBLIC_SOURCE_MATERIALIZATION"
            self.data["source_content_retrieval_opened"] = True
            self.data["candidate_packet_materialization_opened"] = False
            self.data["hidden_evaluation_opened"] = False
            self._write_locked()


class SourceClient:
    def __init__(self, ledger: SourceLedger, policy: dict[str, Any]) -> None:
        self.rest = v5.RetryingClient(v1.api_json, ledger, policy)
        self.graphql = v5.RetryingClient(graphql_title_api, ledger, policy)
        self.title_requests = 0
        self.source_requests = 0

    def title(self, repository: str, number: int) -> str:
        owner, name = repository.split("/", 1)
        payload, _digest = self.graphql.call(
            "graphql:title", {"owner": owner, "name": name, "number": number}
        )
        self.title_requests += 1
        title = str(
            p2a.mapping(
                p2a.mapping(
                    p2a.mapping(p2a.mapping(payload).get("data")).get("repository")
                ).get("pullRequest")
            ).get("title")
            or ""
        )
        if not title:
            raise RuntimeError("pull_request_title_unavailable")
        return title

    def file(self, repository: str, revision: str, path: str) -> bytes | None:
        resource = f"repos/{repository}/contents/{quote(path, safe='/')}"
        try:
            payload, _digest = self.rest.call(resource, {"ref": revision})
        except v5.CandidateMetadataUnavailable:
            self.source_requests += 1
            return None
        self.source_requests += 1
        return decode_content(payload, expected_path=path)

    def license(self, repository: str, revision: str) -> tuple[str, bytes]:
        payload, _digest = self.rest.call(
            f"repos/{repository}/license", {"ref": revision}
        )
        self.source_requests += 1
        row = p2a.mapping(payload)
        path = str(row.get("path") or "")
        return path, decode_content(row, expected_path=path)


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
        ledger = SourceLedger(p2a.resolve(args.checkpoint), config_path, policy)
        client = SourceClient(ledger, policy)
        try:
            report = materialize(config_path, client)
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
                "state": "PUBLIC_SOURCE_MATERIALIZATION_PAUSED_NO_PARTIAL_ADMISSION",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "archive_set_admitted": False,
                "partial_archive_set_admitted": False,
            }
        report = finalize_receipt(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    bindings = p2a.dicts(config.get("source_bindings"))
    for binding in bindings:
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(
            binding.get("sha256") or ""
        ):
            faults.append(f"source_binding_invalid:{binding.get('id')}")
    selection_path = p2a.resolve(str(config.get("metadata_selection_report") or ""))
    selection = p2a.read_json(selection_path) if selection_path.is_file() else {}
    rows = p2a.dicts(selection.get("selected_source_identities"))
    if (
        selection.get("trigger_state") != "GREEN"
        or selection.get("state")
        != "SIXTY_TWO_SOURCE_IDENTITIES_FROZEN_BEFORE_CONTENT_RETRIEVAL"
        or len(rows) != 62
        or len({str(row.get("repository") or "") for row in rows}) != 62
        or sum(row.get("panel") == "control_qualification" for row in rows) != 9
        or sum(row.get("panel") == "claim" for row in rows) != 53
    ):
        faults.append("metadata_selection_invalid")
    archive = p2a.mapping(config.get("archive_policy"))
    if (
        archive.get("deterministic_tar_gzip") is not True
        or archive.get("selected_paths_and_license_only") is not True
        or archive.get("symlinks_allowed") is not False
        or archive.get("untrusted_code_execution_authorized") is not False
        or int(archive.get("maximum_single_file_bytes") or 0) != 2 * 1024 * 1024
        or int(archive.get("maximum_total_materialized_bytes") or 0)
        != 512 * 1024 * 1024
    ):
        faults.append("archive_policy_invalid")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("public_source_file_retrieval_authorized") is not True:
        faults.append("source_retrieval_not_authorized")
    if authority.get("public_pr_title_metadata_retrieval_authorized") is not True:
        faults.append("title_metadata_retrieval_not_authorized")
    if any(
        value is not False
        for key, value in authority.items()
        if key
        not in {
            "public_source_file_retrieval_authorized",
            "public_pr_title_metadata_retrieval_authorized",
        }
    ):
        faults.append("authority_boundary_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": (
            "VCM_SOURCE_MATERIALIZATION_PREFLIGHT_GREEN"
            if not faults
            else "INVALID_PREFLIGHT"
        ),
        "faults": sorted(set(faults)),
        "config": v1.artifact(config_path),
        "metadata_selection": v1.artifact(selection_path),
        "archive_set_admitted": False,
        "partial_archive_set_admitted": False,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "selected_repository_count": len(rows),
        "counters": zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
    }


def materialize(config_path: Path, client: SourceClient) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    selection = p2a.read_json(
        p2a.resolve(str(config.get("metadata_selection_report") or ""))
    )
    output_directory = p2a.resolve(str(config.get("output_directory") or ""))
    output_directory.mkdir(parents=True, exist_ok=True)
    maximum_file = int(
        p2a.mapping(config.get("archive_policy")).get("maximum_single_file_bytes")
        or 0
    )
    maximum_total = int(
        p2a.mapping(config.get("archive_policy")).get(
            "maximum_total_materialized_bytes"
        )
        or 0
    )
    total_bytes = 0
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for index, selected in enumerate(
        p2a.dicts(selection.get("selected_source_identities")), 1
    ):
        row, row_faults, row_bytes = materialize_row(
            selected, index, output_directory, client, maximum_file
        )
        total_bytes += row_bytes
        rows.append(row)
        faults.extend(f"task_{index}:{fault}" for fault in row_faults)
        if total_bytes > maximum_total:
            faults.append("host_total_materialized_bytes_boundary_hit")
            break
    admitted = not faults and len(rows) == 62
    counters = zero_counters()
    counters["public_metadata_title_requests"] = client.title_requests
    counters["public_source_content_requests"] = client.source_requests
    counters["source_archives_materialized"] = sum(
        len(p2a.mapping(row.get("archives"))) for row in rows
    )
    counters["source_bytes_materialized"] = total_bytes
    return {
        **before,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if admitted else "RED",
        "state": (
            "SIXTY_TWO_SOURCE_PAIRS_MATERIALIZED_BEFORE_EVALUATION"
            if admitted
            else "SOURCE_MATERIALIZATION_INCOMPLETE"
        ),
        "faults": sorted(set(faults)),
        "archive_set_admitted": admitted,
        "partial_archive_set_admitted": False,
        "source_content_retrieval_opened": True,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "rows": rows,
        "counters": counters,
    }


def materialize_row(
    selected: dict[str, Any],
    index: int,
    output_directory: Path,
    client: SourceClient,
    maximum_file: int,
) -> tuple[dict[str, Any], list[str], int]:
    repository = str(selected.get("repository") or "")
    number = int(selected.get("pull_request") or 0)
    base_revision = str(selected.get("base_revision") or "")
    head_revision = str(selected.get("head_revision") or "")
    source_paths = p2a.strings(selected.get("source_paths"))
    verifier_paths = p2a.strings(selected.get("verifier_paths"))
    title = client.title(repository, number)
    if hashlib.sha256(title.encode()).hexdigest() != selected.get("title_sha256"):
        raise RuntimeError("pull_request_title_digest_changed")
    parent_license_path, parent_license_bytes = client.license(
        repository, base_revision
    )
    target_license_path, target_license_bytes = client.license(
        repository, head_revision
    )
    validate_file(parent_license_path, parent_license_bytes, maximum_file)
    validate_file(target_license_path, target_license_bytes, maximum_file)
    groups: dict[str, dict[str, bytes]] = {
        "parent_source": {parent_license_path: parent_license_bytes},
        "target_source": {target_license_path: target_license_bytes},
        "parent_verifier": {parent_license_path: parent_license_bytes},
        "target_verifier": {target_license_path: target_license_bytes},
    }
    faults: list[str] = []
    total_bytes = 2 * (len(parent_license_bytes) + len(target_license_bytes))
    for kind, paths in (("source", source_paths), ("verifier", verifier_paths)):
        changed = False
        for path in paths:
            parent = client.file(repository, base_revision, path)
            target = client.file(repository, head_revision, path)
            if parent is not None:
                validate_file(path, parent, maximum_file)
                groups[f"parent_{kind}"][path] = parent
                total_bytes += len(parent)
            if target is not None:
                validate_file(path, target, maximum_file)
                groups[f"target_{kind}"][path] = target
                total_bytes += len(target)
            if parent is None and target is None:
                faults.append(f"selected_{kind}_path_missing_both_revisions:{path}")
            if parent != target:
                changed = True
        if not changed:
            faults.append(f"selected_{kind}_bytes_unchanged")
        if len(groups[f"parent_{kind}"]) == 1 and len(groups[f"target_{kind}"]) == 1:
            faults.append(f"selected_{kind}_bytes_missing_both_revisions")
    archive_rows: dict[str, Any] = {}
    if not faults:
        root = f"vcm-claim-{index:02d}"
        for role, files in groups.items():
            path = output_directory / f"{root}-{role}.tar.gz"
            archives.write_deterministic_archive(path, root, files)
            archive_rows[role] = archive_receipt(path, root, files)
    return {
        "index": index,
        "opaque_source_id": selected.get("opaque_source_id"),
        "repository": repository,
        "pull_request": number,
        "panel": selected.get("panel"),
        "query_language": selected.get("query_language"),
        "natural_language_request": title,
        "natural_language_request_sha256": hashlib.sha256(title.encode()).hexdigest(),
        "base_revision": base_revision,
        "head_revision": head_revision,
        "license_spdx": selected.get("license_spdx"),
        "parent_license_path": parent_license_path,
        "target_license_path": target_license_path,
        "selected_source_paths": source_paths,
        "selected_verifier_paths": verifier_paths,
        "archives": archive_rows,
        "faults": faults,
    }, faults, total_bytes


def graphql_title_api(resource: str, fields: dict[str, Any]) -> tuple[Any, str]:
    if resource != "graphql:title":
        raise ValueError("graphql_title_resource_invalid")
    command = [
        "gh", "api", "graphql", "-f", f"query={TITLE_QUERY}",
        "-F", f"owner={fields['owner']}", "-F", f"name={fields['name']}",
        "-F", f"number={int(fields['number'])}",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    raw = completed.stdout.encode()
    payload = json.loads(raw)
    if p2a.dicts(p2a.mapping(payload).get("errors")):
        raise RuntimeError("graphql_title_response_errors")
    return payload, hashlib.sha256(raw).hexdigest()


def decode_content(value: Any, *, expected_path: str) -> bytes:
    row = p2a.mapping(value)
    if row.get("type") != "file" or str(row.get("path") or "") != expected_path:
        raise RuntimeError("github_content_identity_invalid")
    encoded = str(row.get("content") or "").replace("\n", "")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RuntimeError("github_content_base64_invalid") from exc


def validate_file(path: str, content: bytes, maximum: int) -> None:
    archives.validate_member_path(path)
    if len(content) > maximum:
        raise RuntimeError("host_single_file_bytes_boundary_hit")


def archive_receipt(path: Path, root: str, files: dict[str, bytes]) -> dict[str, Any]:
    return {
        "path": p2a.rel(path),
        "sha256": p2a.sha256_file(path),
        "root": root,
        "members": [
            {
                "path": member,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            for member, content in sorted(files.items())
        ],
    }


def finalize_receipt(
    report: dict[str, Any], ledger: SourceLedger, client: SourceClient
) -> dict[str, Any]:
    ledger.finalize(
        str(report.get("state") or "UNKNOWN"),
        int(report.get("selected_repository_count") or 0),
    )
    report["transport_retry_accounting"] = ledger.summary()
    report["transport"] = {
        "provider": "GitHub_public_GraphQL_title_and_REST_contents_via_gh_cli",
        "logical_request_count": ledger.summary()["logical_request_count"],
        "physical_attempt_count": ledger.summary()["physical_attempt_count"],
        "retry_attempt_count": ledger.summary()["retry_attempt_count"],
        "title_requests": client.title_requests,
        "source_content_requests": client.source_requests,
        "credentials_retained": False,
        "raw_response_bodies_retained": False,
    }
    report["checkpoint"] = v1.artifact(ledger.path)
    if report["checkpoint"]["sha256"] != p2a.sha256_file(ledger.path):
        raise RuntimeError("final_checkpoint_artifact_hash_mismatch")
    report["checkpoint_artifact_hash_verified_final"] = True
    return report


def zero_counters() -> dict[str, int]:
    return {
        "public_metadata_title_requests": 0,
        "public_source_content_requests": 0,
        "source_archives_materialized": 0,
        "source_bytes_materialized": 0,
        "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "D1_cases_consumed": 0,
        "D2_cases_consumed": 0,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state", "state", "archive_set_admitted",
            "partial_archive_set_admitted", "source_content_retrieval_opened",
            "candidate_packet_materialization_opened", "hidden_evaluation_opened",
            "selected_repository_count", "faults", "counters",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
