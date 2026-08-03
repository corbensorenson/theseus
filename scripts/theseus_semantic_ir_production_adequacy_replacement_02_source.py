#!/usr/bin/env python3
"""Freeze a fresh same-stratum source replacement for consumed adequacy Task 2."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import theseus_semantic_ir_production_adequacy_materialization as materialize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_replacement_02_source.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_replacement_source_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = resolve(args.config)
    report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        try:
            report = acquire(config_path)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "SOURCE_TRANSPORT_PAUSED_NO_CANDIDATE_EXPOSURE",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "source_pair_admitted": False,
            }
    materialize.write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = materialize.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("config_policy_invalid")
    if config.get("state") != "FIXED_BEFORE_REPLACEMENT_SOURCE_RETRIEVAL":
        faults.append("config_state_invalid")
    authority = mapping(config.get("authority"))
    if authority.get("public_source_file_retrieval_authorized") is not True or any(
        value is not False
        for key, value in authority.items()
        if key != "public_source_file_retrieval_authorized"
    ):
        faults.append("authority_boundary_invalid")
    consumed = mapping(config.get("consumed_observation"))
    if consumed.get("rerun_authorized") is not False:
        faults.append("consumed_task_rerun_not_fail_closed")
    for path_key, hash_key in (
        ("interruption_receipt", "interruption_receipt_sha256"),
        ("original_task_manifest", "original_task_manifest_sha256"),
        ("candidate_packet", "candidate_packet_sha256"),
    ):
        path = resolve(str(consumed.get(path_key) or ""))
        if not path.is_file() or materialize.sha256_file(path) != consumed.get(hash_key):
            faults.append(f"consumed_binding_invalid:{path_key}")
    prior = resolve(str(config.get("prior_materialization") or ""))
    if (
        not prior.is_file()
        or materialize.sha256_file(prior) != config.get("prior_materialization_sha256")
    ):
        faults.append("prior_materialization_binding_invalid")
        prior_report: dict[str, Any] = {}
    else:
        prior_report = materialize.read_json(prior)
    replacement = mapping(config.get("replacement"))
    prior_rows = dictionaries(prior_report.get("rows"))
    prior_repositories = {str(row.get("repository") or "") for row in prior_rows}
    prior_task = next((row for row in prior_rows if int(row.get("index") or 0) == 2), {})
    if replacement.get("repository") in prior_repositories:
        faults.append("replacement_repository_not_source_disjoint")
    if replacement.get("stratum") != prior_task.get("stratum"):
        faults.append("replacement_stratum_mismatch")
    if int(replacement.get("index") or 0) != 2:
        faults.append("replacement_index_invalid")
    try:
        merged = datetime.fromisoformat(str(replacement.get("merged_utc")).replace("Z", "+00:00"))
        snapshot = datetime.fromisoformat(
            str(config.get("frozen_model_snapshot_created_utc")).replace("Z", "+00:00")
        )
        if merged <= snapshot:
            faults.append("replacement_not_post_snapshot")
    except (TypeError, ValueError):
        faults.append("replacement_chronology_invalid")
    paths = strings(mapping(config.get("expected_change")).get("selected_source_paths"))
    if paths != ["business_logic_test.py"]:
        faults.append("replacement_path_contract_invalid")
    return {
        "policy": POLICY,
        "state": "SOURCE_PREFLIGHT_GREEN" if not faults else "INVALID_SOURCE_PREFLIGHT",
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "config": materialize.artifact(config_path),
        "source_pair_admitted": False,
        "candidate_packet_materialized": False,
        "source_disjoint_from_prior_panel": replacement.get("repository") not in prior_repositories,
        "same_stratum_as_consumed_task": replacement.get("stratum") == prior_task.get("stratum"),
        "counters": materialize.zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config_path: Path) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = materialize.read_json(config_path)
    replacement = mapping(config["replacement"])
    expected = mapping(config["expected_change"])
    license_contract = mapping(config["license"])
    repository = str(replacement["repository"])
    base = str(replacement["base_revision"])
    head = str(replacement["head_revision"])
    merge = str(replacement["merge_revision"])
    pr_number = int(replacement["pull_request"])
    network_digests: list[str] = []
    pr = gh_json(f"repos/{repository}/pulls/{pr_number}", network_digests)
    files = gh_json(f"repos/{repository}/pulls/{pr_number}/files?per_page=100", network_digests)
    merge_commit = gh_json(f"repos/{repository}/git/commits/{merge}", network_digests)
    client = materialize.GitHubCliContentClient()
    selected_paths = strings(expected.get("selected_source_paths"))
    license_path = str(license_contract["path"])
    pair: dict[str, dict[str, bytes]] = {"parent": {}, "target": {}}
    faults = metadata_faults(config, pr, files, merge_commit)
    for role, revision in (("parent", base), ("target", head)):
        for path in [*selected_paths, license_path]:
            materialize.validate_member_path(path)
            content = client.get_file(repository, revision, path)
            if content is None:
                faults.append(f"{role}_source_missing:{path}")
            else:
                pair[role][path] = content
    if pair["parent"].get(license_path) != pair["target"].get(license_path):
        faults.append("license_bytes_changed")
    license_bytes = pair["parent"].get(license_path, b"")
    if str(license_contract.get("required_header") or "").encode() not in license_bytes:
        faults.append("apache_2_license_header_missing")
    parent_source = pair["parent"].get(selected_paths[0], b"") if selected_paths else b""
    target_source = pair["target"].get(selected_paths[0], b"") if selected_paths else b""
    faults.extend(source_change_faults(parent_source, target_source, expected))
    archive_rows: dict[str, Any] = {}
    if not faults:
        output = resolve(str(config["output_directory"]))
        root = "semantic-adequacy-02r1"
        for role in ("parent", "target"):
            archive = output / f"{root}-{role}.tar.gz"
            materialize.write_deterministic_archive(archive, root, pair[role])
            archive_rows[role] = {
                "path": rel(archive),
                "sha256": materialize.sha256_file(archive),
                "root": root,
                "members": [
                    {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                    for path, content in sorted(pair[role].items())
                ],
            }
    admitted = not faults and len(archive_rows) == 2
    counters = materialize.zero_counters()
    counters["network_source_calls"] = len(network_digests) + client.request_count
    counters["source_archives_materialized"] = len(archive_rows)
    return {
        **before,
        "state": "SOURCE_PAIR_FROZEN_BEFORE_EVALUATOR_QUALIFICATION" if admitted else "INVALID_SOURCE_PAIR",
        "trigger_state": "GREEN" if admitted else "RED",
        "faults": sorted(set(faults)),
        "source_pair_admitted": admitted,
        "metadata": {
            "repository": repository,
            "pull_request": pr_number,
            "pull_request_url": pr.get("html_url"),
            "title": pr.get("title"),
            "merged_utc": pr.get("merged_at"),
            "parent_revision": base,
            "target_revision": head,
            "merge_revision": merge,
            "merge_first_parent_revision": (merge_commit.get("parents") or [{}])[0].get("sha"),
            "changed_files": [row.get("filename") for row in files if isinstance(row, dict)],
            "stratum": replacement.get("stratum"),
            "license_spdx": license_contract.get("spdx"),
        },
        "archives": archive_rows,
        "transport": {
            "provider": "GitHub_public_API_via_gh_cli",
            "request_count": len(network_digests) + client.request_count,
            "response_digest_chain_sha256": stable_hash(network_digests + client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "counters": counters,
    }


def metadata_faults(
    config: dict[str, Any],
    pr: dict[str, Any],
    files: list[Any],
    merge_commit: dict[str, Any],
) -> list[str]:
    expected = mapping(config["replacement"])
    faults: list[str] = []
    observed = {
        "base_revision": get_path(pr, "base", "sha"),
        "head_revision": get_path(pr, "head", "sha"),
        "merge_revision": pr.get("merge_commit_sha"),
        "merged_utc": pr.get("merged_at"),
        "title": pr.get("title"),
    }
    for key, value in observed.items():
        if value != expected.get(key):
            faults.append(f"metadata_mismatch:{key}")
    if pr.get("state") != "closed" or pr.get("merged_at") is None:
        faults.append("pull_request_not_merged")
    changed = [row.get("filename") for row in files if isinstance(row, dict)]
    if changed != strings(mapping(config["expected_change"]).get("selected_source_paths")):
        faults.append("changed_file_inventory_mismatch")
    if merge_commit.get("sha") != expected.get("merge_revision"):
        faults.append("merge_commit_identity_mismatch")
    parents = merge_commit.get("parents") if isinstance(merge_commit.get("parents"), list) else []
    if not parents or mapping(parents[0]).get("sha") != expected.get("base_revision"):
        faults.append("merge_first_parent_mismatch")
    return faults


def source_change_faults(parent: bytes, target: bytes, expected: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    try:
        parent_text = parent.decode("utf-8")
        target_text = target.decode("utf-8")
        ast.parse(parent_text)
        ast.parse(target_text)
    except (UnicodeDecodeError, SyntaxError):
        return ["selected_source_not_valid_utf8_python"]
    if parent_text == target_text:
        faults.append("selected_source_bytes_unchanged")
    diff = list(difflib.ndiff(parent_text.splitlines(), target_text.splitlines()))
    deleted = [line[2:].strip() for line in diff if line.startswith("- ")]
    added = [line[2:].strip() for line in diff if line.startswith("+ ")]
    if deleted != strings(expected.get("deleted_fragments")):
        faults.append("deleted_fragment_inventory_mismatch")
    if added != strings(expected.get("added_fragments")):
        faults.append("added_fragment_inventory_mismatch")
    return faults


def gh_json(endpoint: str, digests: list[str]) -> Any:
    completed = subprocess.run(
        ["gh", "api", "-X", "GET", endpoint],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=False,
        timeout=120,
    )
    digests.append(hashlib.sha256(completed.stdout + b"\0" + completed.stderr).hexdigest())
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-1000:])
    return json.loads(completed.stdout)


def get_path(value: dict[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "source_pair_admitted": report.get("source_pair_admitted"),
        "candidate_packet_materialized": report.get("candidate_packet_materialized"),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dictionaries(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(row) for row in value] if isinstance(value, list) else []


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
