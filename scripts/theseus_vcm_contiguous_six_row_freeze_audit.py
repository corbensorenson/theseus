#!/usr/bin/env python3
"""Role-separated audit of the contiguous six-row VCM instrument freeze."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_contiguous_six_row_freeze as producer  # noqa: E402
import theseus_vcm_parent_only_materializer_audit as parent_audit  # noqa: E402

POLICY = "project_theseus_vcm_contiguous_six_row_freeze_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_contiguous_six_row_freeze.json"
QUALIFIED = "QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS"
EXPECTED_PREDECESSOR = {16, 25, 56}
EXPECTED_REPLACEMENTS = {12, 13, 35}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(
    path: Path = DEFAULT_CONFIG,
    *,
    producer_report: dict[str, Any] | None = None,
    store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    producer.validate_binding(cfg, "audit_owner", "audit_owner_sha256", Path(__file__).resolve(), faults)
    producer.validate_binding(
        cfg,
        "parent_only_audit_owner",
        "parent_only_audit_owner_sha256",
        Path(parent_audit.__file__).resolve(),
        faults,
    )
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    evidence = load_bound_evidence(cfg, faults)
    claim_audit = evidence.get("claim_instrument_audit", {})
    if claim_audit.get("trigger_state") != "GREEN" or claim_audit.get("faults") != []:
        faults.append("claim_instrument_audit_not_green")
    qualification = audit_qualification(cfg, evidence, faults)
    audit_source_bindings(cfg, evidence, faults)

    actual_report = producer_report or p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    actual_store = store or p2a.read_json(p2a.resolve(str(cfg.get("store_out") or "")))
    replay_cfg = {
        "audit_policy": parent_audit.POLICY,
        "audit_owner": p2a.rel(Path(parent_audit.__file__).resolve()),
        "audit_owner_sha256": p2a.sha256_file(Path(parent_audit.__file__).resolve()),
        "owner": str(cfg.get("parent_only_owner")),
        "owner_sha256": str(cfg.get("parent_only_owner_sha256")),
        "broad_parent_effect_root": "repository",
        "expected_row_count": 6,
        "rows": cfg.get("rows"),
        "report": str(cfg.get("report")),
        "store_out": str(cfg.get("store_out")),
        "audit_maximum_inference": cfg.get("audit_maximum_inference"),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(replay_cfg, handle, sort_keys=True)
        handle.flush()
        parent_replay = parent_audit.audit(Path(handle.name), producer=actual_report, store=actual_store)
    faults.extend(p2a.strings(parent_replay.get("faults")))

    claim = evidence.get("claim_instrument", {})
    contract_faults: list[str] = []
    expected_contract = producer.freeze_contract(cfg, claim, contract_faults)
    faults.extend(contract_faults)
    if actual_report.get("frozen_contract") != expected_contract:
        faults.append("frozen_contract_rederivation_failed")
    if (
        actual_report.get("trigger_state") != "GREEN"
        or actual_report.get("state") != "K2_05_CONTIGUOUS_SIX_ROW_INSTRUMENT_FROZEN"
        or actual_report.get("panel_admitted") is not False
        or actual_report.get("project_selected_quality_token_cap") is not None
    ):
        faults.append("producer_state_or_boundary_invalid")
    for key in (
        "candidate_or_control_calls",
        "local_model_calls",
        "external_reference_calls",
        "teacher_calls",
        "parent_target_or_evaluator_executions",
        "network_calls",
    ):
        if actual_report.get(key) != 0:
            faults.append(f"producer_counter_invalid:{key}")
    flow = p2a.mapping(actual_report.get("information_flow"))
    if any(
        flow.get(key) is not False
        for key in (
            "producer_parsed_hidden_qualification_evidence",
            "producer_read_target_archive",
            "producer_read_target_diff",
            "producer_read_hidden_evaluator",
            "producer_read_answer_identifying_metadata",
            "target_derived_effect_paths_present",
        )
    ):
        faults.append("producer_information_flow_invalid")
    ready = (
        not faults
        and parent_replay.get("trigger_state") == "GREEN"
        and qualification.get("qualified_task_count") == 6
    )
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": (
            "K2_05_CONTIGUOUS_SIX_ROW_INSTRUMENT_ROLE_SEPARATELY_REDERIVED"
            if ready
            else "K2_05_CONTIGUOUS_SIX_ROW_INSTRUMENT_AUDIT_FAILED"
        ),
        "faults": sorted(set(faults)),
        "config": producer.identity(path),
        "producer_report": producer.identity(p2a.resolve(str(cfg.get("report") or ""))) if producer_report is None else {"in_memory": True},
        "store_artifact": producer.identity(p2a.resolve(str(cfg.get("store_out") or ""))) if store is None else {"in_memory": True},
        "qualification": qualification,
        "audited_row_count": parent_replay.get("audited_row_count"),
        "audited_candidate_visible_field_count": parent_replay.get("audited_candidate_visible_field_count"),
        "parent_only_rederivation_conclusions": parent_replay.get("conclusions"),
        "frozen_contract_sha256": parent_audit.digest(parent_audit.canonical(expected_contract)),
        "audit_kind": "role-separated rederivation",
        "panel_admitted": False,
        "candidate_or_control_calls": 0,
        "local_model_calls": 0,
        "external_reference_calls": 0,
        "teacher_calls": 0,
        "parent_target_or_evaluator_executions": 0,
        "network_calls": 0,
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


def load_bound_evidence(cfg: dict[str, Any], faults: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    bindings = [p2a.mapping(cfg.get("claim_instrument")), *p2a.dicts(cfg.get("opaque_prerequisite_bindings"))]
    for binding in bindings:
        name = str(binding.get("id") or "claim_instrument")
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"evidence_binding_invalid:{name}")
            result[name] = {}
        else:
            result[name] = p2a.read_json(source)
    return result


def audit_qualification(
    cfg: dict[str, Any], evidence: dict[str, dict[str, Any]], faults: list[str]
) -> dict[str, Any]:
    predecessor = evidence.get("predecessor_matched_verifier", {})
    predecessor_audit = evidence.get("predecessor_matched_verifier_audit", {})
    replacement = evidence.get("replacement_matched_verifier", {})
    replacement_audit = evidence.get("replacement_matched_verifier_audit", {})
    if predecessor.get("trigger_state") != "GREEN" or predecessor_audit.get("trigger_state") != "GREEN":
        faults.append("predecessor_evidence_not_green")
    if replacement.get("trigger_state") != "GREEN" or replacement_audit.get("trigger_state") != "GREEN":
        faults.append("replacement_evidence_not_green")
    for report, label in ((predecessor, "predecessor"), (replacement, "replacement")):
        for key in ("candidate_or_control_calls", "external_reference_calls"):
            if report.get(key) != 0:
                faults.append(f"{label}_qualification_counter_invalid:{key}")
    for audit_report, label in (
        (predecessor_audit, "predecessor"),
        (replacement_audit, "replacement"),
    ):
        if audit_report.get("parent_target_or_evaluator_executions") != 0:
            faults.append(f"{label}_audit_execution_counter_invalid")
    pred_rows = {int(row.get("index") or 0): row for row in p2a.dicts(predecessor_audit.get("rows"))}
    repl_rows = {int(row.get("index") or 0): row for row in p2a.dicts(replacement_audit.get("rows"))}
    pred_qualified = {index for index, row in pred_rows.items() if row.get("disposition") == QUALIFIED and row.get("parent_returncode") != 0 and row.get("target_returncode") == 0}
    repl_qualified = {index for index, row in repl_rows.items() if row.get("disposition") == QUALIFIED and row.get("parent_returncode") != 0 and row.get("target_returncode") == 0}
    if pred_qualified != EXPECTED_PREDECESSOR:
        faults.append("predecessor_qualified_set_invalid")
    if repl_qualified != EXPECTED_REPLACEMENTS:
        faults.append("replacement_qualified_set_invalid")
    if any(pred_rows.get(index, {}).get("disposition") == QUALIFIED for index in EXPECTED_REPLACEMENTS):
        faults.append("inadequate_predecessor_rows_not_superseded")
    bindings = {str(row.get("id")): row for row in p2a.dicts(cfg.get("opaque_prerequisite_bindings"))}
    for report, audit_report, label in (
        (predecessor, predecessor_audit, "predecessor"),
        (replacement, replacement_audit, "replacement"),
    ):
        bound = p2a.mapping(audit_report.get("producer_report"))
        expected_binding = p2a.mapping(bindings.get(f"{label}_matched_verifier"))
        if bound.get("sha256") != expected_binding.get("sha256"):
            faults.append(f"{label}_producer_audit_binding_invalid")
    qualified = sorted(pred_qualified | repl_qualified)
    return {
        "qualified_task_count": len(qualified),
        "qualified_task_indices": qualified,
        "predecessor_qualified_task_indices": sorted(pred_qualified),
        "replacement_qualified_task_indices": sorted(repl_qualified),
        "exact_evaluator_reruns_performed": 0,
        "qualification_evidence_reused_by_receipt": True,
    }

def audit_source_bindings(
    cfg: dict[str, Any], evidence: dict[str, dict[str, Any]], faults: list[str]
) -> None:
    panel = evidence.get("source_panel", {})
    closures = evidence.get("repository_closures", {})
    panel_rows = {int(row.get("index") or 0): row for row in p2a.dicts(panel.get("assembled_rows"))}
    closure_rows = {int(row.get("campaign_index") or 0): row for row in p2a.dicts(closures.get("tasks"))}
    expected_indices = [12, 13, 16, 25, 35, 56]
    if panel.get("trigger_state") != "GREEN" or closures.get("trigger_state") != "GREEN":
        faults.append("source_or_closure_evidence_not_green")
    configured_indices = [int(value) for value in cfg.get("qualified_task_indices", [])]
    if configured_indices != expected_indices:
        faults.append("configured_task_indices_invalid")
    for index, binding in zip(expected_indices, p2a.dicts(cfg.get("rows")), strict=False):
        source = p2a.mapping(panel_rows.get(index))
        closure = p2a.mapping(closure_rows.get(index))
        parent = next((row for row in p2a.dicts(closure.get("artifacts")) if row.get("label") == "parent"), {})
        checks = (
            binding.get("natural_language_request") == source.get("natural_language_request"),
            binding.get("natural_language_request_sha256") == source.get("natural_language_request_sha256"),
            binding.get("parent_revision") == source.get("base_revision") == parent.get("revision"),
            binding.get("license_spdx") == source.get("license_spdx"),
            binding.get("parent_archive") == parent.get("normalized"),
            binding.get("parent_archive_sha256") == parent.get("normalized_sha256"),
            binding.get("parent_archive_root") == parent.get("source_archive_root"),
            binding.get("sanitization_report") == parent.get("sanitization_report"),
            binding.get("sanitization_report_sha256") == parent.get("sanitization_report_sha256"),
        )
        if not all(checks):
            faults.append(f"parent_only_source_binding_invalid:{index}")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state",
            "state",
            "faults",
            "audited_row_count",
            "audited_candidate_visible_field_count",
            "local_model_calls",
            "external_reference_calls",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
