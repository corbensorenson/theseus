#!/usr/bin/env python3
"""Role-separated receipt audit for the six matched common evaluators."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_six_row_matched_verifier as producer  # noqa: E402

POLICY = "project_theseus_vcm_six_row_matched_verifier_audit_v1"
DEFAULT_CONFIG = ROOT / "configs/theseus_vcm_six_row_matched_verifier.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps({key: report.get(key) for key in (
        "trigger_state", "state", "faults", "audited_task_count", "qualified_task_count",
        "inconclusive_task_count", "candidate_or_control_calls", "external_reference_calls",
    )}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, execution: dict[str, Any] | None = None, verify_store: bool = True) -> dict[str, Any]:
    cfg, bound, faults = producer.preflight(path, verify_store=verify_store)
    audit_owner = p2a.resolve(str(cfg.get("audit_owner") or ""))
    if cfg.get("audit_policy") != POLICY or audit_owner != Path(__file__).resolve() or not audit_owner.is_file() or p2a.sha256_file(audit_owner) != cfg.get("audit_owner_sha256"):
        faults.append("audit_policy_or_owner_binding_invalid")
    report = execution if execution is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    if report.get("trigger_state") != "GREEN" or report.get("state") != "K2_05_SIX_ROW_MATCHED_VERIFIERS_EXECUTED_WITH_SCOPED_DISPOSITIONS":
        faults.append("producer_state_invalid")
    if report.get("panel_admitted") is not False or report.get("partial_panel_admission_forbidden") is not True:
        faults.append("producer_panel_policy_invalid")
    if report.get("target_production_transplant_count") != 0 or report.get("network_enabled_calls") != 0:
        faults.append("producer_information_flow_or_network_policy_invalid")
    if report.get("candidate_or_control_calls") != 0 or report.get("external_reference_calls") != 0 or report.get("teacher_calls") != 0:
        faults.append("producer_model_call_policy_invalid")
    if report.get("project_selected_output_cap") is not None:
        faults.append("producer_output_cap_invalid")

    configured = {int(row.get("index") or 0): row for row in p2a.dicts(cfg.get("rows"))}
    audited: list[dict[str, Any]] = []
    observed_executions = 0
    observed_installs = 0
    reused_count = 0
    predecessor_rows = {int(row.get("index") or 0): row for row in p2a.dicts(p2a.mapping(bound.get("sources")).get("matched_verifier_predecessor", {}).get("rows"))}
    for actual in p2a.dicts(report.get("rows")):
        index = int(actual.get("index") or 0)
        expected = configured.get(index, {})
        if actual.get("repository") != expected.get("repository") or actual.get("manager") != expected.get("manager"):
            faults.append(f"row_identity_invalid:{index}")
        if actual.get("reused_from_predecessor") is True:
            reused_count += 1
            prior = predecessor_rows.get(index, {})
            predecessor_receipt = p2a.mapping(actual.get("predecessor"))
            predecessor_binding = p2a.mapping(cfg.get("sources")).get("matched_verifier_predecessor", {})
            if index not in bound.get("reuse_indices", set()) or predecessor_receipt.get("path") != predecessor_binding.get("path") or predecessor_receipt.get("sha256") != predecessor_binding.get("sha256"):
                faults.append(f"predecessor_reuse_binding_invalid:{index}")
            for key in ("disposition", "faults"):
                expected_value = prior.get(key)
                observed_value = actual.get(key)
                if key == "faults":
                    expected_value, observed_value = p2a.strings(expected_value), p2a.strings(observed_value)
                if observed_value != expected_value:
                    faults.append(f"predecessor_reuse_receipt_invalid:{index}:{key}")
            for side in ("parent", "target"):
                for key in ("returncode", "boundary_hit", "boundary_reason"):
                    if p2a.mapping(actual.get(side)).get(key) != p2a.mapping(prior.get(side)).get(key):
                        faults.append(f"predecessor_side_reuse_invalid:{index}:{side}:{key}")
            audited.append({
                "index": index, "parent_returncode": p2a.mapping(actual.get("parent")).get("returncode"),
                "target_returncode": p2a.mapping(actual.get("target")).get("returncode"),
                "disposition": actual.get("disposition"), "common_evaluator_count": len(p2a.mapping(expected.get("common_evaluator_sha256"))),
                "reused_from_predecessor": True,
            })
            continue
        lock = p2a.resolve(str(p2a.mapping(actual.get("lock")).get("path") or ""))
        if not lock.is_file() or p2a.sha256_file(lock) != expected.get("lock_sha256") or p2a.mapping(actual.get("lock")).get("sha256") != expected.get("lock_sha256"):
            faults.append(f"lock_receipt_invalid:{index}")
        for side in ("parent", "target"):
            archive_receipt = p2a.mapping(p2a.mapping(actual.get("archives")).get(side))
            archive_binding = p2a.mapping(expected.get(f"{side}_archive"))
            archive = p2a.resolve(str(archive_receipt.get("path") or ""))
            if not archive.is_file() or p2a.sha256_file(archive) != archive_binding.get("sha256") or archive_receipt.get("sha256") != archive_binding.get("sha256"):
                faults.append(f"archive_receipt_invalid:{index}:{side}")

        expected_evaluators = p2a.mapping(expected.get("common_evaluator_sha256"))
        evaluator_receipts = p2a.dicts(actual.get("common_evaluator_receipts"))
        expected_pairs = {(side, path) for side in ("parent", "target") for path in p2a.strings(expected.get("common_evaluator_paths"))}
        observed_pairs: set[tuple[str, str]] = set()
        for receipt in evaluator_receipts:
            pair = (str(receipt.get("side") or ""), str(receipt.get("path") or ""))
            observed_pairs.add(pair)
            expected_sha = expected_evaluators.get(pair[1])
            if receipt.get("sha256") != expected_sha or receipt.get("expected_sha256") != expected_sha:
                faults.append(f"common_evaluator_receipt_invalid:{index}:{pair[0]}:{pair[1]}")
        if observed_pairs != expected_pairs:
            faults.append(f"common_evaluator_denominator_invalid:{index}")
        if set(p2a.strings(actual.get("common_evaluator_paths"))) & set(p2a.strings(expected.get("forbidden_transplant_paths"))):
            faults.append(f"forbidden_transplant_observed:{index}")

        sides = {side: p2a.mapping(actual.get(side)) for side in ("parent", "target")}
        for side, receipt in sides.items():
            if receipt:
                observed_executions += 1
                if receipt.get("declared_arguments") != expected.get("arguments") or receipt.get("working_directory") != expected.get("working_directory"):
                    faults.append(f"command_receipt_invalid:{index}:{side}")
                if receipt.get("python_path_roots") != expected.get("python_path_roots", []):
                    faults.append(f"python_path_receipt_invalid:{index}:{side}")
                if receipt.get("network_denied") is not True or receipt.get("project_selected_output_cap") is not None:
                    faults.append(f"runtime_policy_receipt_invalid:{index}:{side}")
                if receipt.get("stdout_complete") is not True or receipt.get("stderr_complete") is not True:
                    faults.append(f"diagnostic_completeness_invalid:{index}:{side}")
                for stream in ("stdout", "stderr"):
                    payload = str(receipt.get(stream) or "").encode("utf-8")
                    if len(payload) != receipt.get(f"{stream}_bytes") or hashlib.sha256(payload).hexdigest() != receipt.get(f"{stream}_sha256"):
                        faults.append(f"diagnostic_identity_invalid:{index}:{side}:{stream}")
        environment = p2a.mapping(actual.get("environment"))
        if expected.get("manager") == "uv" and environment.get("sync"):
            observed_installs += 1
            for key in ("venv_command", "sync"):
                receipt = p2a.mapping(environment.get(key))
                if receipt.get("network_denied") is not True or receipt.get("project_selected_output_cap") is not None:
                    faults.append(f"environment_policy_receipt_invalid:{index}:{key}")
            if environment.get("project_installation") is not False:
                faults.append(f"project_installation_invalid:{index}")
        disposition = producer.derive_disposition(sides, p2a.strings(actual.get("faults")))
        if disposition != actual.get("disposition"):
            faults.append(f"disposition_rederivation_failed:{index}")
        audited.append({
            "index": index, "parent_returncode": sides["parent"].get("returncode"),
            "target_returncode": sides["target"].get("returncode"), "disposition": disposition,
            "common_evaluator_count": len(expected_evaluators),
        })

    if len(audited) != int(cfg.get("expected_task_count") or 0) or set(configured) != {row["index"] for row in audited}:
        faults.append("audited_denominator_invalid")
    qualified = sum(row["disposition"] == "QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS" for row in audited)
    if report.get("qualified_task_count") != qualified or report.get("inconclusive_task_count") != len(audited) - qualified:
        faults.append("producer_count_rederivation_failed")
    if report.get("parent_target_or_evaluator_executions") != observed_executions or report.get("repository_runner_executions") != observed_executions:
        faults.append("execution_count_rederivation_failed")
    if report.get("package_installations") != observed_installs:
        faults.append("package_installation_count_rederivation_failed")
    clone_receipts = p2a.mapping(report.get("cache_clone_receipts"))
    for manager in ("uv", "cargo"):
        clone = p2a.mapping(clone_receipts.get(manager))
        if clone.get("returncode") != 0 or clone.get("network_denied_by_absence_of_network_operation") is not True or clone.get("project_selected_output_cap") is not None:
            faults.append(f"cache_clone_receipt_invalid:{manager}")
    cargo_executions = sum(1 for actual in p2a.dicts(report.get("rows")) if actual.get("manager") == "cargo" for side in ("parent", "target") if p2a.mapping(actual.get(side)))
    if report.get("source_build_executions") != cargo_executions or report.get("project_installations") != 0:
        faults.append("build_or_project_installation_count_invalid")
    before = p2a.mapping(report.get("retained_store_before"))
    after = p2a.mapping(report.get("retained_store_after"))
    expected_store = p2a.mapping(p2a.mapping(bound.get("sources")).get("environment")).get("retained_shared_store")
    if before != after or before != expected_store or report.get("retained_store_unchanged") is not True:
        faults.append("retained_store_identity_invalid")

    return {
        "policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if not faults else "RED",
        "state": "K2_05_SIX_ROW_MATCHED_VERIFIERS_ROLE_SEPARATELY_REDERIVED" if not faults else "K2_05_SIX_ROW_MATCHED_VERIFIER_AUDIT_FAILED",
        "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "producer_report": {"path": cfg.get("report"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or ""))) if execution is None else None},
        "audited_task_count": len(audited), "qualified_task_count": qualified,
        "inconclusive_task_count": len(audited) - qualified, "reused_predecessor_task_count": reused_count, "rows": audited,
        "retained_store_identity_rederived": before == after == expected_store,
        "panel_admitted": False, "audit_kind": "role-separated rederivation",
        "network_or_dependency_execution_performed": False, "parent_target_or_evaluator_executions": 0,
        "candidate_or_control_calls": 0, "external_reference_calls": 0, "teacher_calls": 0,
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
