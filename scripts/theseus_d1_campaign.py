#!/usr/bin/env python3
"""Run the sealed 44-task D1 campaign exactly once after a P4 survivor."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_d1_cognitive_compilation as candidate  # noqa: E402
import theseus_d1_evaluator as evaluator  # noqa: E402
import theseus_d1_evaluator_seal as seal  # noqa: E402
import theseus_p4v2r2r2_campaign as p4_campaign  # noqa: E402


POLICY = "project_theseus_d1_blind_matched_campaign_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_d1_campaign.json"
LEARNED_ARMS = {
    "typed_semantic_ir_treatment",
    "direct_target_generation",
    "natural_language_plan_control",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = p2a.resolve(args.config)
    config = p2a.read_json(config_path)
    report = audit_campaign(config, config_path=config_path)
    if args.execute and not args.audit_only and report.get("execution_authorized") is True:
        report = execute_once(config, config_path=config_path)
    p2a.write_json(p2a.resolve(str(config["progress_report"])), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") in {"GREEN", "PAUSED"} else 2


def audit_campaign(
    config: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    disposition_override: dict[str, Any] | None = None,
    pool_override: dict[str, Any] | None = None,
    consumption_override: list[dict[str, Any]] | None = None,
    lease_exists_override: bool | None = None,
    extra_faults: list[str] | None = None,
) -> dict[str, Any]:
    faults = validate_config(config) + list(extra_faults or [])
    disposition_path = p2a.resolve(str(config.get("p4_terminal_disposition") or ""))
    disposition = (
        disposition_override
        if disposition_override is not None
        else p2a.read_json(disposition_path) if disposition_path.is_file() else {}
    )
    terminal_p4 = (
        disposition.get("policy") == config.get("required_p4_policy")
        and disposition.get("trigger_state") == "GREEN"
    )
    survivor = terminal_p4 and disposition.get("scientific_status") == config.get(
        "required_p4_status"
    )
    pool_path = p2a.resolve(str(config.get("task_pool") or ""))
    pool = pool_override if pool_override is not None else (
        p2a.read_json(pool_path) if pool_path.is_file() else {}
    )
    pool_audit = audit_pool(config, pool, pool_path, pool_override)
    control = select_primary_control(disposition, config) if survivor else ""
    instrument_path = p2a.resolve(str(config.get("causal_instrument") or ""))
    instrument_audit = candidate.audit_instrument(instrument_path)
    if instrument_audit.get("trigger_state") != "GREEN":
        faults.append("causal_instrument_audit_red")
    consumption = (
        consumption_override
        if consumption_override is not None
        else read_jsonl(p2a.resolve(str(config.get("consumption_registry") or "")))
    )
    pool_sha = stable_input_hash(pool, pool_path, pool_override)
    consumed = [row for row in consumption if row.get("task_pool_sha256") == pool_sha]
    if len(consumed) > 1:
        faults.append("task_pool_consumption_duplicated")
    result_rows = audit_results(config, pool) if pool_audit["passed"] else []
    complete = sum(row.get("complete") is True for row in result_rows)
    lease_path = p2a.resolve(str(config.get("active_lease") or ""))
    lease_exists = lease_path.exists() if lease_exists_override is None else lease_exists_override
    activation = "WAITING_FOR_TERMINAL_P4V2R2R3"
    if terminal_p4 and not survivor:
        activation = "D1_CLOSED_P4V2R2R3_NON_SURVIVOR"
    elif survivor and not pool_audit["passed"]:
        activation = "WAITING_FOR_GREEN_SEALED_D1_TASK_POOL"
    elif survivor and pool_audit["passed"] and not control:
        activation = "PRIMARY_CONTROL_SELECTION_INVALID"
        faults.append("primary_control_selection_invalid")
    elif survivor and pool_audit["passed"] and consumed:
        activation = "D1_TASK_POOL_ALREADY_CONSUMED_RERUN_FORBIDDEN"
    elif survivor and pool_audit["passed"] and complete == 44:
        activation = "D1_CAMPAIGN_COMPLETE"
    elif survivor and pool_audit["passed"] and lease_exists:
        activation = "WAITING_FOR_EXCLUSIVE_D1_CAMPAIGN_LEASE"
    elif survivor and pool_audit["passed"]:
        activation = "D1_CAMPAIGN_EXECUTION_READY"
    execution_authorized = (
        not faults
        and survivor
        and pool_audit["passed"]
        and bool(control)
        and complete < 44
        and not consumed
        and not lease_exists
    )
    trigger = "RED" if faults else "GREEN" if (
        activation in {
            "D1_CLOSED_P4V2R2R3_NON_SURVIVOR",
            "D1_CAMPAIGN_COMPLETE",
            "D1_TASK_POOL_ALREADY_CONSUMED_RERUN_FORBIDDEN",
        }
        or execution_authorized
    ) else "PAUSED"
    return {
        "policy": POLICY,
        "created_utc": seal.now(),
        "trigger_state": trigger,
        "activation_state": activation,
        "execution_authorized": execution_authorized,
        "faults": sorted(set(faults)),
        "config": source_identity(config_path),
        "p4_terminal": terminal_p4,
        "p4_survivor": survivor,
        "p4_terminal_disposition": input_identity(disposition_path, disposition, disposition_override),
        "task_pool_audit": pool_audit,
        "task_pool_sha256": pool_sha,
        "primary_treatment": "typed_semantic_ir_treatment",
        "primary_control": control,
        "primary_control_selected_before_D1_outcomes": bool(control),
        "causal_instrument_audit": instrument_audit,
        "complete_tasks": complete,
        "pending_tasks": max(0, 44 - complete) if pool_audit["passed"] else 44,
        "tasks": result_rows,
        "consumption_matches": len(consumed),
        "model_calls_retained": sum(int(row.get("model_calls") or 0) for row in result_rows),
        "physical_context_boundary_hits": sum(
            int(row.get("physical_context_boundary_hits") or 0) for row in result_rows
        ),
        "candidate_or_control_calls_before_final_pool_seal": 0,
        "external_inference_calls": 0,
        "teacher_calls": 0,
        "training_rows_written": 0,
        "project_selected_quality_token_cap": None,
        "maximum_inference": str(config.get("maximum_inference") or ""),
    }


def execute_once(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    before = audit_campaign(config, config_path=config_path)
    if before.get("execution_authorized") is not True:
        return before
    lease_path = p2a.resolve(str(config["active_lease"]))
    lease_id = uuid.uuid4().hex
    lease = {
        "policy": POLICY,
        "lease_id": lease_id,
        "state": "RUNNING",
        "created_utc": seal.now(),
        "task_pool_sha256": before["task_pool_sha256"],
    }
    try:
        write_json_exclusive(lease_path, lease)
    except FileExistsError:
        raced = audit_campaign(config, config_path=config_path)
        raced["trigger_state"] = "PAUSED"
        raced["execution_authorized"] = False
        raced["activation_state"] = "LEASE_ACQUISITION_RACE"
        return raced
    try:
        report = run_campaign(config, config_path=config_path)
        if report.get("complete_tasks") == 44 and report.get("trigger_state") == "GREEN":
            append_consumption(config, report)
            report = audit_campaign(config, config_path=config_path, lease_exists_override=True)
    finally:
        lease["state"] = "COMPLETED"
        lease["completed_utc"] = seal.now()
        p2a.write_json(lease_path, lease)
        archive = p2a.resolve(str(config["lease_archive_directory"]))
        archive.mkdir(parents=True, exist_ok=True)
        os.replace(lease_path, archive / f"{lease_id}.json")
    report["execution_authorized"] = False
    return report


def run_campaign(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    pool = p2a.read_json(p2a.resolve(str(config["task_pool"])))
    instrument = p2a.resolve(str(config["causal_instrument"]))
    result_root = p2a.resolve(str(config["result_root"]))
    result_root.mkdir(parents=True, exist_ok=True)
    for row in p2a.dicts(pool.get("tasks")):
        paths = result_paths(config, row)
        if not paths["run"].is_file():
            if runtime_reports(config, row):
                return audit_campaign(
                    config,
                    config_path=config_path,
                    extra_faults=[f"partial_unsealed_runtime_receipts:{row.get('campaign_index')}"],
                )
            run = candidate.run_experiment(
                instrument,
                p2a.resolve(str(row["task"])),
                p2a.resolve(str(row["evaluator"])),
            )
            receipts = runtime_reports(config, row)
            custody = p4_campaign.route_custody(receipts)
            if len(receipts) != 6 or custody.get("passed") is not True:
                return audit_campaign(
                    config,
                    config_path=config_path,
                    extra_faults=[f"route_custody_red:{row.get('campaign_index')}"],
                )
            p2a.write_json(paths["run"], run)
            if run.get("trigger_state") not in {"GREEN", "YELLOW"}:
                return audit_campaign(
                    config,
                    config_path=config_path,
                    extra_faults=[f"candidate_run_red:{row.get('campaign_index')}"],
                )
        if not paths["evaluation"].is_file():
            scored = evaluator.evaluate_report(
                paths["run"], p2a.resolve(str(row["evaluator"]))
            )
            p2a.write_json(paths["evaluation"], scored)
            if scored.get("trigger_state") != "GREEN":
                return audit_campaign(
                    config,
                    config_path=config_path,
                    extra_faults=[f"blind_evaluation_red:{row.get('campaign_index')}"],
                )
        progress = audit_campaign(config, config_path=config_path, lease_exists_override=True)
        p2a.write_json(p2a.resolve(str(config["progress_report"])), progress)
    return audit_campaign(config, config_path=config_path, lease_exists_override=True)


def audit_results(config: dict[str, Any], pool: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in p2a.dicts(pool.get("tasks")):
        paths = result_paths(config, task)
        run_exists = paths["run"].is_file()
        evaluation_exists = paths["evaluation"].is_file()
        calls = 0
        boundaries = 0
        faults: list[str] = []
        receipts = runtime_reports(config, task)
        if receipts and not run_exists:
            faults.append("partial_unsealed_runtime_receipts")
        if run_exists:
            run = p2a.read_json(paths["run"])
            calls = int(p2a.mapping(run.get("denominators")).get("model_calls") or 0)
            if run.get("policy") != candidate.POLICY or calls != 6 or len(receipts) != 6:
                faults.append("candidate_run_or_call_denominator_invalid")
            custody = p4_campaign.route_custody(receipts)
            if custody.get("passed") is not True:
                faults.extend(p2a.strings(custody.get("faults")))
            boundaries = sum(
                int(row.get("physical_context_boundary_hit") is True)
                for row in p2a.dicts(custody.get("terminations"))
            )
        if evaluation_exists:
            scored = p2a.read_json(paths["evaluation"])
            if scored.get("policy") != evaluator.POLICY or scored.get("trigger_state") != "GREEN":
                faults.append("evaluation_invalid")
            if not run_exists or scored.get("candidate_report_sha256") != p2a.sha256_file(paths["run"]):
                faults.append("evaluation_run_binding_invalid")
        rows.append(
            {
                "campaign_index": task.get("campaign_index"),
                "repository": task.get("repository"),
                "run": p2a.rel(paths["run"]) if run_exists else "",
                "run_sha256": p2a.sha256_file(paths["run"]),
                "evaluation": p2a.rel(paths["evaluation"]) if evaluation_exists else "",
                "evaluation_sha256": p2a.sha256_file(paths["evaluation"]),
                "runtime_receipts": len(receipts),
                "model_calls": calls,
                "physical_context_boundary_hits": boundaries,
                "faults": sorted(set(faults)),
                "complete": run_exists and evaluation_exists and not faults,
            }
        )
    return rows


def audit_pool(
    config: dict[str, Any],
    pool: dict[str, Any],
    path: Path,
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    faults: list[str] = []
    if not pool:
        return {"passed": False, "faults": ["task_pool_missing"], "task_count": 0}
    if pool.get("policy") != config.get("required_task_pool_policy") or pool.get("state") != (
        "SEALED_BEFORE_CANDIDATE_OR_CONTROL_GENERATION"
    ):
        faults.append("task_pool_policy_or_state_invalid")
    tasks = p2a.dicts(pool.get("tasks"))
    if int(pool.get("task_count") or 0) != 44 or len(tasks) != 44:
        faults.append("task_pool_count_invalid")
    if len({str(row.get("repository") or "").lower() for row in tasks}) != 44:
        faults.append("task_pool_repository_count_invalid")
    for row in tasks:
        for path_key, digest_key in (("task", "task_sha256"), ("evaluator", "evaluator_sha256")):
            owner = p2a.resolve(str(row.get(path_key) or ""))
            if not owner.is_file() or p2a.sha256_file(owner) != row.get(digest_key):
                faults.append(f"task_pool_binding_invalid:{path_key}")
    if int(pool.get("candidate_or_control_calls") or 0) != 0:
        faults.append("task_pool_preseal_candidate_calls_nonzero")
    if pool.get("post_candidate_task_replacement_allowed") is not False:
        faults.append("task_pool_replacement_allowed")
    if pool.get("project_selected_quality_token_cap") is not None:
        faults.append("task_pool_quality_token_cap_present")
    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "task_count": len(tasks),
        "path": p2a.rel(path),
        "sha256": stable_input_hash(pool, path, override),
    }


def select_primary_control(disposition: dict[str, Any], config: dict[str, Any]) -> str:
    policy = p2a.mapping(config.get("primary_arm_policy"))
    controls = p2a.strings(policy.get("control_candidates"))
    totals = p2a.mapping(disposition.get("arm_totals"))
    if set(controls) != {"direct_target_generation", "natural_language_plan_control"}:
        return ""
    scores = {
        arm: int(p2a.mapping(totals.get(arm)).get("useful_candidates") or 0)
        for arm in controls
    }
    return sorted(
        controls,
        key=lambda arm: (-scores[arm], 0 if arm == "direct_target_generation" else 1),
    )[0]


def result_paths(config: dict[str, Any], row: dict[str, Any]) -> dict[str, Path]:
    index = int(row.get("campaign_index") or 0)
    digest = str(row.get("selection_digest") or "")[:16]
    root = p2a.resolve(str(config["result_root"]))
    stem = f"d1_{index:02d}_{digest}"
    return {"run": root / f"{stem}_run.json", "evaluation": root / f"{stem}_evaluation.json"}


def runtime_reports(config: dict[str, Any], row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(p2a.resolve(str(row.get("task") or "")))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    namespace = str(config.get("runtime_attempt_namespace") or "")
    return sorted((ROOT / "runtime" / "p2a").glob(f"*{task_id}*{namespace}*.json"))


def append_consumption(config: dict[str, Any], report: dict[str, Any]) -> None:
    path = p2a.resolve(str(config["consumption_registry"]))
    rows = read_jsonl(path)
    pool_sha = str(report.get("task_pool_sha256") or "")
    if any(row.get("task_pool_sha256") == pool_sha for row in rows):
        raise ValueError("D1_task_pool_already_consumed")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "policy": "project_theseus_d1_consumption_registry_v1",
        "created_utc": seal.now(),
        "task_pool_sha256": pool_sha,
        "complete_tasks": 44,
        "model_calls_retained": report.get("model_calls_retained"),
        "project_selected_quality_token_cap": None,
        "rerun_allowed": False,
        "training_eligible": False,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def validate_config(config: dict[str, Any]) -> list[str]:
    faults: list[str] = []
    if config.get("policy") != POLICY or config.get("state") != (
        "PROSPECTIVELY_BOUND_BEFORE_D1_FINAL_POOL_OR_CANDIDATE_GENERATION"
    ):
        faults.append("config_policy_or_state_invalid")
    instrument = p2a.resolve(str(config.get("causal_instrument") or ""))
    if not instrument.is_file() or p2a.sha256_file(instrument) != config.get("causal_instrument_sha256"):
        faults.append("causal_instrument_binding_invalid")
    matching = p2a.mapping(config.get("matching"))
    required_matching = {
        "same_frozen_weights",
        "same_candidate_visible_information",
        "same_two_calls_and_visible_repair_opportunity",
        "same_natural_completion_policy",
        "same_context_residual_rule",
        "same_effect_sandbox",
        "same_visible_and_hidden_verifier_budget",
        "same_total_task_denominator",
        "joined_cost_reported_per_arm",
        "complete_visible_verifier_feedback_visible_to_second_call",
    }
    if any(matching.get(key) is not True for key in required_matching):
        faults.append("matched_contract_invalid")
    if matching.get("project_selected_quality_token_cap") is not None:
        faults.append("quality_token_cap_present")
    if matching.get("project_selected_verifier_feedback_character_cap") is not None:
        faults.append("verifier_feedback_character_cap_present")
    authority = p2a.mapping(config.get("authority"))
    if authority.get("user_or_operator_approval_required") is not False:
        faults.append("user_gate_present")
    if authority.get("post_candidate_task_replacement_allowed") is not False:
        faults.append("post_candidate_replacement_allowed")
    if any(int(authority.get(key) or 0) != 0 for key in (
        "external_inference_calls", "teacher_calls", "training_rows_written"
    )):
        faults.append("external_or_training_authority_present")
    return faults


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_input_hash(value: dict[str, Any], path: Path, override: dict[str, Any] | None) -> str:
    return p2a.stable_hash(value) if override is not None and value else p2a.sha256_file(path)


def input_identity(path: Path, value: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    return {"path": p2a.rel(path), "present": bool(value), "sha256": stable_input_hash(value, path, override)}


def source_identity(path: Path) -> dict[str, str]:
    return {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "trigger_state", "activation_state", "execution_authorized",
            "p4_survivor", "primary_control", "complete_tasks", "pending_tasks",
            "model_calls_retained", "physical_context_boundary_hits",
            "project_selected_quality_token_cap", "faults",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
