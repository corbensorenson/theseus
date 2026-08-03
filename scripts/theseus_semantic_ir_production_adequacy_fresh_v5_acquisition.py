#!/usr/bin/env python3
"""Freeze the one fresh source pair required after the v4 Task 1 host wall."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import theseus_semantic_ir_production_adequacy_fresh_v4_acquisition as v4
import theseus_semantic_ir_production_adequacy_materialization as materialize
import theseus_semantic_ir_production_adequacy_replacement_02_source as source02


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_semantic_ir_production_adequacy_fresh_v5_sources.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_semantic_ir_production_adequacy_fresh_v5_sources.json"
POLICY = "project_theseus_semantic_ir_production_adequacy_fresh_v5_sources_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=source02.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=source02.rel(DEFAULT_OUT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = source02.resolve(args.config)
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
    materialize.write_json(source02.resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = materialize.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY or config.get("state") != "FIXED_BEFORE_V5_SOURCE_RETRIEVAL":
        faults.append("config_identity_invalid")
    authority = source02.mapping(config.get("authority"))
    if authority.get("public_source_file_retrieval_authorized") is not True or any(
        value is not False
        for key, value in authority.items()
        if key != "public_source_file_retrieval_authorized"
    ):
        faults.append("authority_boundary_invalid")
    for path_key, hash_key, owner in (
        ("prior_task_pool", "prior_task_pool_sha256", config),
        ("candidate_report", "candidate_report_sha256", source02.mapping(config.get("consumed_evidence"))),
        ("journal", "journal_sha256", source02.mapping(config.get("consumed_evidence"))),
        ("runtime_receipt", "runtime_receipt_sha256", source02.mapping(config.get("consumed_evidence"))),
    ):
        path = source02.resolve(str(owner.get(path_key) or ""))
        if not path.is_file() or materialize.sha256_file(path) != owner.get(hash_key):
            faults.append(f"binding_invalid:{path_key}")
    consumed = source02.mapping(config.get("consumed_evidence"))
    if consumed.get("consumed_indices") != [1] or consumed.get("rerun_authorized") is not False:
        faults.append("consumed_surface_boundary_invalid")
    candidate = materialize.read_json(source02.resolve(str(consumed.get("candidate_report") or "")))
    journal = materialize.read_json(source02.resolve(str(consumed.get("journal") or "")))
    runtime = materialize.read_json(source02.resolve(str(consumed.get("runtime_receipt") or "")))
    if candidate.get("state") != "BLOCKED_INFRASTRUCTURE_REPLACEMENT_REQUIRED" or candidate.get("rows") != []:
        faults.append("v4_candidate_custody_invalid")
    if journal.get("state") != "MODEL_CALL_INFRASTRUCTURE_INVALID" or journal.get("task_index") != 1:
        faults.append("v4_journal_custody_invalid")
    backend_path = source02.resolve(str(source02.mapping(runtime.get("generation_backend")).get("out") or ""))
    backend = materialize.read_json(backend_path) if backend_path.is_file() else {}
    backend_metrics = source02.mapping(backend.get("metrics"))
    if (
        backend_metrics.get("termination_reason") != "host_safety_wall_time"
        or backend_metrics.get("generated_tokens") != 0
        or backend_metrics.get("physical_context_boundary_hit") is not False
        or backend_metrics.get("project_selected_quality_token_cap") is not None
    ):
        faults.append("v4_infrastructure_diagnosis_invalid")
    prior = materialize.read_json(source02.resolve(str(config.get("prior_task_pool") or "")))
    prior_repositories = {str(row.get("repository") or "") for row in source02.dictionaries(prior.get("source_denominator"))}
    sources = source02.dictionaries(config.get("sources"))
    if len(sources) != 1 or int(sources[0].get("index") or 0) != 1:
        faults.append("replacement_index_invalid")
    if sources:
        spec = sources[0]
        if str(spec.get("repository") or "") in prior_repositories:
            faults.append("replacement_repository_not_source_disjoint")
        if spec.get("stratum") != "single_expression_replacement":
            faults.append("replacement_stratum_invalid")
        if source02.strings(spec.get("selected_source_paths")) != ["pytboss/http.py"]:
            faults.append("selected_source_contract_invalid")
        if source02.mapping(spec.get("license")).get("spdx") != "Apache-2.0":
            faults.append("license_contract_invalid")
        try:
            snapshot = datetime.fromisoformat(str(config.get("frozen_model_snapshot_created_utc")).replace("Z", "+00:00"))
            merged = datetime.fromisoformat(str(spec.get("merged_utc")).replace("Z", "+00:00"))
            if merged <= snapshot:
                faults.append("replacement_not_post_snapshot")
        except (TypeError, ValueError):
            faults.append("replacement_chronology_invalid")
    return {
        "policy": POLICY,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "SOURCE_PREFLIGHT_GREEN" if not faults else "INVALID_SOURCE_PREFLIGHT",
        "config": v4.artifact(config_path),
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
    output = source02.resolve(str(config.get("output_directory") or ""))
    client = materialize.GitHubCliContentClient()
    digests: list[str] = []
    spec = source02.dictionaries(config.get("sources"))[0]
    row, faults = v4.acquire_row(spec, output, client, digests)
    admitted = not faults and row.get("trigger_state") == "GREEN"
    counters = materialize.zero_counters()
    counters["network_source_calls"] = len(digests) + client.request_count
    counters["source_archives_materialized"] = len(source02.mapping(row.get("archives")))
    return {
        **before,
        "trigger_state": "GREEN" if admitted else "RED",
        "state": "ONE_SOURCE_PAIR_FROZEN_BEFORE_V5_EVALUATOR_QUALIFICATION" if admitted else "INVALID_SOURCE_PAIR",
        "source_pairs_admitted": admitted,
        "rows": [row],
        "faults": sorted(set(faults)),
        "transport": {
            "provider": "GitHub_public_API_via_gh_cli",
            "request_count": len(digests) + client.request_count,
            "response_digest_chain_sha256": source02.stable_hash(digests + client.response_digests),
            "credentials_retained": False,
            "raw_response_bodies_retained": False,
        },
        "counters": counters,
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report.get("trigger_state"),
        "state": report.get("state"),
        "source_pairs_admitted": report.get("source_pairs_admitted"),
        "green_rows": sum(row.get("trigger_state") == "GREEN" for row in source02.dictionaries(report.get("rows"))),
        "faults": report.get("faults"),
        "counters": report.get("counters"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
