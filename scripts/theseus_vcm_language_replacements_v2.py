#!/usr/bin/env python3
"""Transport-only successor for the six VCM English-scope replacements."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import theseus_assistant_p2a as p2a
import theseus_vcm_language_replacements as v1r
import theseus_vcm_source_acquisition as v1
import theseus_vcm_source_acquisition_v5 as v5
import theseus_vcm_source_materialization as source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_language_replacements_v2.json"
DEFAULT_OUT = ROOT / "reports" / "theseus_vcm_language_replacements_v2.json"
DEFAULT_CHECKPOINT = ROOT / "reports" / "theseus_vcm_language_replacements_v2_checkpoint.json"
POLICY = "project_theseus_vcm_language_replacements_v2"


class InstrumentedSourceClient(source.SourceClient):
    def __init__(self, ledger: source.SourceLedger, policy: dict[str, Any]) -> None:
        super().__init__(ledger, policy)
        self.title_logical_attempts = 0
        self.source_logical_attempts = 0

    def title(self, repository: str, number: int) -> str:
        self.title_logical_attempts += 1
        return super().title(repository, number)

    def file(self, repository: str, revision: str, path: str) -> bytes | None:
        self.source_logical_attempts += 1
        return super().file(repository, revision, path)

    def license(self, repository: str, revision: str) -> tuple[str, bytes]:
        self.source_logical_attempts += 1
        return super().license(repository, revision)


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
        retry_policy = p2a.mapping(config.get("transport_retry_policy"))
        ledger = source.SourceLedger(p2a.resolve(args.checkpoint), config_path, retry_policy)
        client = InstrumentedSourceClient(ledger, retry_policy)
        try:
            report = acquire(config_path, ledger, client, retry_policy)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {
                **report,
                "trigger_state": "PAUSED",
                "state": "LANGUAGE_REPLACEMENT_V2_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION",
                "faults": [f"{type(exc).__name__}:{exc}"[:4000]],
                "replacement_set_admitted": False,
            }
        report = source.finalize_receipt(report, ledger, client)
        report = attach_attempted_role_accounting(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out), report)
    print(json.dumps(v1r.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path)
    faults: list[str] = []
    if config.get("policy") != POLICY:
        faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != str(binding.get("sha256") or ""):
            faults.append(f"source_binding_invalid:{binding.get('id')}")
    predecessor_path = p2a.resolve(str(config.get("predecessor_report") or ""))
    predecessor = p2a.read_json(predecessor_path) if predecessor_path.is_file() else {}
    checkpoint = p2a.mapping(predecessor.get("checkpoint"))
    checkpoint_path = p2a.resolve(str(checkpoint.get("path") or ""))
    if predecessor.get("trigger_state") != "PAUSED" or predecessor.get("state") != "LANGUAGE_REPLACEMENT_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION" or predecessor.get("replacement_set_admitted") is not False or predecessor.get("counters", {}).get("source_archives_materialized") != 0:
        faults.append("predecessor_pause_invalid")
    if not checkpoint_path.is_file() or p2a.sha256_file(checkpoint_path) != str(checkpoint.get("sha256") or ""):
        faults.append("predecessor_checkpoint_invalid")
    repair = p2a.mapping(config.get("transport_repair"))
    if int(repair.get("predecessor_graphql_node_batch_size") or 0) != 40 or int(repair.get("successor_graphql_node_batch_size") or 0) != 20 or repair.get("scientific_selection_rules_changed") is not False or repair.get("attempted_role_accounting_required") is not True:
        faults.append("transport_repair_invalid")
    base_config_path = p2a.resolve(str(config.get("predecessor_config") or ""))
    base_config = p2a.read_json(base_config_path) if base_config_path.is_file() else {}
    if replacement_scientific_view(config) != replacement_scientific_view(base_config):
        faults.append("scientific_replacement_contract_changed")
    counters = source.zero_counters()
    counters.update({"public_metadata_selection_requests": 0, "local_language_scope_classification_calls": 0})
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "state": "SIX_LANGUAGE_REPLACEMENT_V2_TRANSPORT_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT",
        "faults": sorted(set(faults)),
        "config": v1.artifact(config_path),
        "predecessor_report": v1.artifact(predecessor_path),
        "replacement_set_admitted": False,
        "selected_repository_count": 0,
        "source_content_retrieval_opened": False,
        "candidate_packet_materialization_opened": False,
        "hidden_evaluation_opened": False,
        "transport_repair": repair,
        "counters": counters,
        "maximum_inference": config.get("maximum_inference"),
    }


def acquire(config_path: Path, ledger: source.SourceLedger, client: InstrumentedSourceClient, retry_policy: dict[str, Any]) -> dict[str, Any]:
    before = preflight(config_path)
    if before["trigger_state"] != "GREEN":
        return before
    config = p2a.read_json(config_path)
    scientific_path = p2a.resolve(str(config.get("scientific_selection_config") or ""))
    scientific = p2a.read_json(scientific_path)
    modified_scientific = json.loads(json.dumps(scientific))
    modified_scientific["graphql_transport"]["node_batch_size"] = int(p2a.mapping(config.get("transport_repair")).get("successor_graphql_node_batch_size"))
    predecessor_config = p2a.read_json(p2a.resolve(str(config.get("predecessor_config") or "")))
    with tempfile.TemporaryDirectory(prefix="theseus_vcm_language_v2_", dir="/private/tmp") as temporary:
        temporary_root = Path(temporary)
        scientific_temp = temporary_root / "scientific.json"
        config_temp = temporary_root / "config.json"
        p2a.write_json(scientific_temp, modified_scientific)
        effective = {**predecessor_config, "scientific_selection_config": str(scientific_temp)}
        p2a.write_json(config_temp, effective)
        report = v1r.acquire(config_temp, ledger, client, retry_policy)
    return {
        **report,
        "policy": POLICY,
        "config": v1.artifact(config_path),
        "predecessor_report": before["predecessor_report"],
        "transport_repair": before["transport_repair"],
        "maximum_inference": config.get("maximum_inference"),
    }


def attach_attempted_role_accounting(report: dict[str, Any], ledger: source.SourceLedger, client: InstrumentedSourceClient) -> dict[str, Any]:
    counters = dict(p2a.mapping(report.get("counters")))
    logical = int(ledger.summary().get("logical_request_count") or 0)
    counters["public_metadata_selection_requests"] = logical - client.title_logical_attempts - client.source_logical_attempts
    counters["public_metadata_title_requests"] = client.title_logical_attempts
    counters["public_source_content_requests"] = client.source_logical_attempts
    report["counters"] = counters
    report["attempted_request_role_accounting"] = {
        "metadata": counters["public_metadata_selection_requests"],
        "title": client.title_logical_attempts,
        "source_content": client.source_logical_attempts,
        "sum_equals_checkpoint_logical_requests": sum((counters["public_metadata_selection_requests"], client.title_logical_attempts, client.source_logical_attempts)) == logical,
    }
    return report


def replacement_scientific_view(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config.get(key) for key in (
        "scientific_selection_config", "metadata_selection_report", "replacement_28_report",
        "source_panel_audit_report", "replacement_slots", "replacement_policy",
        "english_language_policy", "archive_policy", "output_directory", "authority",
    )}


if __name__ == "__main__":
    raise SystemExit(main())
