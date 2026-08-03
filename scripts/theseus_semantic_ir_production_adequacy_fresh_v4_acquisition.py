#!/usr/bin/env python3
"""Freeze four fresh source pairs for the statement-granular adequacy denominator."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_semantic_ir_production_adequacy_materialization as materialize
import theseus_semantic_ir_production_adequacy_replacement_02_source as source02


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v4_sources.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v4_sources.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v4_sources_v1"


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
                "source_pairs_admitted": False,
            }
    materialize.write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = materialize.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY or config.get("state") != "FIXED_BEFORE_FRESH_SOURCE_RETRIEVAL":
        faults.append("config_identity_invalid")
    authority = mapping(config.get("authority"))
    if authority.get("public_source_file_retrieval_authorized") is not True or any(
        value is not False for key, value in authority.items()
        if key != "public_source_file_retrieval_authorized"
    ):
        faults.append("authority_boundary_invalid")
    for path_key, hash_key, owner in (
        ("prior_materialization", "prior_materialization_sha256", config),
        ("statement_granularity_audit", "statement_granularity_audit_sha256", config),
        ("candidate_report", "candidate_report_sha256", mapping(config.get("consumed_evidence"))),
        ("interruption_receipt", "interruption_receipt_sha256", mapping(config.get("consumed_evidence"))),
    ):
        path = resolve(str(owner.get(path_key) or ""))
        if not path.is_file() or materialize.sha256_file(path) != owner.get(hash_key):
            faults.append(f"binding_invalid:{path_key}")
    consumed = mapping(config.get("consumed_evidence"))
    if consumed.get("consumed_indices") != [1, 2, 3, 4] or consumed.get("rerun_authorized") is not False:
        faults.append("consumed_surface_boundary_invalid")
    prior_path = resolve(str(config.get("prior_materialization") or ""))
    prior = materialize.read_json(prior_path) if prior_path.is_file() else {}
    prior_rows = dictionaries(prior.get("rows"))
    prior_repositories = {str(row.get("repository") or "") for row in prior_rows}
    sources = dictionaries(config.get("sources"))
    if [integer(row.get("index")) for row in sources] != [1, 2, 3, 4]:
        faults.append("replacement_indices_invalid")
    repositories = [str(row.get("repository") or "") for row in sources]
    if len(set(repositories)) != 4 or any(repo in prior_repositories for repo in repositories):
        faults.append("replacement_repositories_not_source_disjoint")
    expected_strata = ["single_expression_replacement"] * 3 + ["branch_or_predicate_replacement"]
    if [str(row.get("stratum") or "") for row in sources] != expected_strata:
        faults.append("replacement_strata_invalid")
    try:
        snapshot = datetime.fromisoformat(str(config.get("frozen_model_snapshot_created_utc")).replace("Z", "+00:00"))
        if any(datetime.fromisoformat(str(row.get("merged_utc")).replace("Z", "+00:00")) <= snapshot for row in sources):
            faults.append("replacement_not_post_snapshot")
    except (TypeError, ValueError):
        faults.append("replacement_chronology_invalid")
    for row in sources:
        selected = strings(row.get("selected_source_paths"))
        if len(selected) != 1 or not selected[0].endswith(".py"):
            faults.append(f"selected_source_contract_invalid:{integer(row.get('index')):02d}")
        if str(row.get("license", {}).get("spdx") or "") not in {"BSD-3-Clause", "MIT"}:
            faults.append(f"license_contract_invalid:{integer(row.get('index')):02d}")
    return {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "SOURCE_PREFLIGHT_GREEN" if not faults else "INVALID_SOURCE_PREFLIGHT",
        "config": artifact(config_path),
        "source_pairs_admitted": False,
        "candidate_packet_materialized": False,
        "faults": sorted(set(faults)),
        "counters": materialize.zero_counters(),
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config_path: Path) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = materialize.read_json(config_path)
    output = resolve(str(config.get("output_directory") or ""))
    client = materialize.GitHubCliContentClient()
    network_digests: list[str] = []
    rows: list[dict[str, Any]] = []
    faults: list[str] = []
    for spec in dictionaries(config.get("sources")):
        row, row_faults = acquire_row(spec, output, client, network_digests)
        rows.append(row)
        faults.extend(f"task_{integer(spec.get('index')):02d}:{fault}" for fault in row_faults)
    admitted = not faults and len(rows) == 4 and all(row.get("trigger_state") == "GREEN" for row in rows)
    counters = materialize.zero_counters()
    counters["network_source_calls"] = len(network_digests) + client.request_count
    counters["source_archives_materialized"] = sum(len(mapping(row.get("archives"))) for row in rows)
    return {
        **before,
        "trigger_state": "GREEN" if admitted else "RED",
        "state": "FOUR_SOURCE_PAIRS_FROZEN_BEFORE_EVALUATOR_QUALIFICATION" if admitted else "INVALID_SOURCE_PAIRS",
        "source_pairs_admitted": admitted,
        "rows": rows,
        "faults": sorted(set(faults)),
        "transport": {
            "provider": "GitHub_public_API_via_gh_cli",
            "request_count": len(network_digests) + client.request_count,
            "response_digest_chain_sha256": source02.stable_hash(network_digests + client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "counters": counters,
    }


def acquire_row(spec: dict[str, Any], output: Path, client: Any, digests: list[str]) -> tuple[dict[str, Any], list[str]]:
    repository = str(spec.get("repository") or "")
    number = integer(spec.get("pull_request"))
    base = str(spec.get("base_revision") or "")
    head = str(spec.get("head_revision") or "")
    merge = str(spec.get("merge_revision") or "")
    pr = source02.gh_json(f"repos/{repository}/pulls/{number}", digests)
    files = source02.gh_json(f"repos/{repository}/pulls/{number}/files?per_page=100", digests)
    commit = source02.gh_json(f"repos/{repository}/git/commits/{merge}", digests)
    observed = {
        "title": pr.get("title"),
        "merged_utc": pr.get("merged_at"),
        "base_revision": source02.get_path(pr, "base", "sha"),
        "head_revision": source02.get_path(pr, "head", "sha"),
        "merge_revision": pr.get("merge_commit_sha"),
    }
    faults = [f"metadata_mismatch:{key}" for key, value in observed.items() if value != spec.get(key)]
    if pr.get("state") != "closed" or not pr.get("merged_at"):
        faults.append("pull_request_not_merged")
    changed = [row.get("filename") for row in files if isinstance(row, dict)]
    if changed != strings(spec.get("changed_paths")):
        faults.append("changed_path_inventory_mismatch")
    parents = commit.get("parents") if isinstance(commit.get("parents"), list) else []
    if commit.get("sha") != merge or not parents or mapping(parents[0]).get("sha") != base:
        faults.append("merge_first_parent_binding_invalid")
    selected = strings(spec.get("selected_source_paths"))
    license_contract = mapping(spec.get("license"))
    license_path = str(license_contract.get("path") or "")
    pair: dict[str, dict[str, bytes]] = {"parent": {}, "target": {}}
    for role, revision in (("parent", base), ("target", head)):
        for path in [*selected, license_path]:
            materialize.validate_member_path(path)
            content = client.get_file(repository, revision, path)
            if content is None:
                faults.append(f"{role}_source_missing:{path}")
            else:
                pair[role][path] = content
    if pair["parent"].get(license_path) != pair["target"].get(license_path):
        faults.append("license_bytes_changed")
    notice = str(license_contract.get("required_notice") or "").encode()
    if notice not in pair["parent"].get(license_path, b""):
        faults.append("license_notice_missing")
    if selected:
        try:
            parent_text = pair["parent"].get(selected[0], b"").decode("utf-8")
            target_text = pair["target"].get(selected[0], b"").decode("utf-8")
            ast.parse(parent_text)
            ast.parse(target_text)
            if parent_text == target_text:
                faults.append("selected_source_unchanged")
            for fragment in strings(spec.get("parent_fragments")):
                if fragment not in parent_text:
                    faults.append("parent_fragment_missing:" + hashlib.sha256(fragment.encode()).hexdigest()[:12])
            for fragment in strings(spec.get("target_fragments")):
                if fragment not in target_text:
                    faults.append("target_fragment_missing:" + hashlib.sha256(fragment.encode()).hexdigest()[:12])
        except (UnicodeDecodeError, SyntaxError):
            faults.append("selected_source_not_valid_utf8_python")
    archives: dict[str, Any] = {}
    index = integer(spec.get("index"))
    root_name = f"semantic-adequacy-{index:02d}r2"
    if not faults:
        for role in ("parent", "target"):
            path = output / f"{root_name}-{role}.tar.gz"
            materialize.write_deterministic_archive(path, root_name, pair[role])
            archives[role] = {
                "path": rel(path),
                "sha256": materialize.sha256_file(path),
                "root": root_name,
                "members": [
                    {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                    for name, content in sorted(pair[role].items())
                ],
            }
    return ({
        "index": index,
        "opaque_task_id": spec.get("opaque_task_id"),
        "trigger_state": "GREEN" if not faults else "RED",
        "repository": repository,
        "pull_request": number,
        "pull_request_url": pr.get("html_url"),
        "title": pr.get("title"),
        "merged_utc": pr.get("merged_at"),
        "parent_revision": base,
        "target_revision": head,
        "merge_revision": merge,
        "merge_first_parent_revision": mapping(parents[0]).get("sha") if parents else None,
        "stratum": spec.get("stratum"),
        "selected_source_paths": selected,
        "changed_paths": changed,
        "license_spdx": license_contract.get("spdx"),
        "archives": archives,
        "faults": sorted(set(faults)),
    }, faults)


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "source_pairs_admitted": report.get("source_pairs_admitted"),
        "green_rows": sum(row.get("trigger_state") == "GREEN" for row in dictionaries(report.get("rows"))),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


mapping = source02.mapping
dictionaries = source02.dictionaries
strings = source02.strings
resolve = source02.resolve
rel = source02.rel


def integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def artifact(path: Path) -> dict[str, str]:
    return {"path": rel(path), "sha256": materialize.sha256_file(path)}


if __name__ == "__main__":
    raise SystemExit(main())
