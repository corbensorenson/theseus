#!/usr/bin/env python3
"""Run the sealed mechanics-qualified P4-v2r2 local campaign exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_p4_cognitive_compilation_evaluator as p4_evaluator  # noqa: E402
import theseus_p4v2r2_cognitive_compilation as p4v2r2  # noqa: E402


POLICY = "project_theseus_p4v2r2_cognitive_compilation_campaign_v1"
POOL = ROOT / "configs" / "theseus_p4v2r2_task_pool.json"
INSTRUMENT = ROOT / "configs" / "theseus_p4v2r2_cognitive_compilation_instrument.json"
POOL_SEAL_COMMIT = "9cd965514ea4babbd58317e5d9908563f0b440ba"
POOL_SHA256 = "64269ba60e3798a5af91367c4bed98cdfdc70ae6bb7c5c3e885e91aa12f70ffd"
INSTRUMENT_SHA256 = "046f989e7eaa444a64e38d96fae8def10dabc4a6c3b1b2b2de3118ba4fe5d6ff"
RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2_attempt1"
PROGRESS = ROOT / "reports" / "theseus_p4v2r2_campaign_attempt2_progress.json"
NORMAL_TERMINATIONS = {"parser_complete", "model_eos"}
BOOTSTRAP_FAILURE_COMMIT = "209f2f2ed22a11835323fc56ff4805c088080fed"
BOOTSTRAP_FAILURES = {
    "candidate_run": (
        ROOT / "reports" / "theseus_p4v2r2_attempt1_01_flask_5917_run.json",
        "b40a957528e1b47b0e4354250199a401c869a74be709771358ca0d0ddde9f6b6",
    ),
    "campaign_progress": (
        ROOT / "reports" / "theseus_p4v2r2_campaign_attempt1_progress.json",
        "a246708f88053bfc13141de100ec5575f3d2b17997a3239626ea6fd2b0a54fe8",
    ),
    "terminal_disposition": (
        ROOT / "reports" / "theseus_p4v2r2_terminal_disposition.json",
        "c2543ea2419444b4f1046ff74014057bb2f2e66b4b40e92af85dd299376e09b1",
    ),
    "autonomous_launch": (
        ROOT / "reports" / "theseus_p4v2r2_autonomous_launch.json",
        "622690097dd4c204655b298a286141dd1489bb7f2b763a88b787ab724c883561",
    ),
    "lease": (
        ROOT
        / "reports"
        / "theseus_p4v2r2_campaign_leases"
        / "7847d0a1444f4447bf513e5a0a983ab7.json",
        "bff06225ffa81e7062c1e50d8ebc9ef925ab017b2ff12166927514de4af01698",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    report = audit_campaign()
    if not args.audit_only and report["trigger_state"] == "GREEN":
        report = run_campaign()
    p2a.write_json(PROGRESS, report)
    print(
        json.dumps(
            {
                "trigger_state": report["trigger_state"],
                "complete_tasks": report["complete_tasks"],
                "pending_tasks": report["pending_tasks"],
                "model_calls_retained": report["model_calls_retained"],
                "safety_ceiling_hits": report["safety_ceiling_hits"],
                "faults": report["faults"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["trigger_state"] == "GREEN" else 2


def run_campaign() -> dict[str, Any]:
    pool = p2a.read_json(POOL)
    for row in p2a.dicts(pool.get("tasks")):
        paths = result_paths(row)
        if not paths["run"].is_file():
            partial = runtime_reports(row)
            if partial:
                return audit_campaign(
                    [f"partial_unsealed_runtime_receipts:{row['stem']}"]
                )
            run = p4v2r2.run_experiment(INSTRUMENT, ROOT / str(row["task"]))
            p2a.write_json(paths["run"], run)
            if run.get("trigger_state") not in {"GREEN", "YELLOW"}:
                return audit_campaign([f"candidate_run_red:{row['stem']}"])
        if not paths["evaluation"].is_file():
            evaluation = p4_evaluator.evaluate_report(
                paths["run"], ROOT / str(row["evaluator"])
            )
            p2a.write_json(paths["evaluation"], evaluation)
            if evaluation.get("trigger_state") != "GREEN":
                return audit_campaign([f"blind_evaluation_red:{row['stem']}"])
        p2a.write_json(PROGRESS, audit_campaign())
    return audit_campaign()


def audit_campaign(extra_faults: list[str] | None = None) -> dict[str, Any]:
    faults = list(extra_faults or [])
    bootstrap_repair = audit_preconsumption_bootstrap_failure()
    if bootstrap_repair["passed"] is not True:
        faults.extend(bootstrap_repair["faults"])
    if p2a.sha256_file(POOL) != POOL_SHA256:
        faults.append("task_pool_digest_mismatch")
    pool = p2a.read_json(POOL)
    if pool.get("state") != "SEALED_BEFORE_CANDIDATE_GENERATION":
        faults.append("task_pool_not_sealed")
    if int(pool.get("green_evaluator_audits") or 0) != 10:
        faults.append("evaluator_adequacy_floor_invalid")
    if int(pool.get("v2r2_oracle_replays_green") or 0) != 10:
        faults.append("v2r2_transport_oracle_floor_invalid")
    if int(pool.get("dependency_corruptions_rejected") or 0) != 10:
        faults.append("dependency_corruption_floor_invalid")
    counters = p2a.mapping(pool.get("counters"))
    for key in (
        "local_model_calls",
        "hosted_model_calls",
        "teacher_calls",
        "deterministic_request_compiler_calls",
        "public_calibration_cases_consumed",
        "D1_cases_consumed",
        "D2_cases_consumed",
        "training_rows_written",
    ):
        if int(counters.get(key) or 0) != 0:
            faults.append(f"pre_campaign_counter_nonzero:{key}")
    if str(pool.get("instrument_sha256") or "") != INSTRUMENT_SHA256:
        faults.append("pool_instrument_binding_mismatch")
    if p2a.sha256_file(INSTRUMENT) != INSTRUMENT_SHA256:
        faults.append("instrument_digest_mismatch")
    instrument = p2a.read_json(INSTRUMENT)
    if instrument.get("runtime_attempt_namespace") != RUNTIME_ATTEMPT_NAMESPACE:
        faults.append("runtime_attempt_namespace_invalid")
    generation = p2a.mapping(instrument.get("generation_budget"))
    budgets = p2a.mapping(instrument.get("budgets"))
    if generation.get("project_selected_quality_token_cap") is not None:
        faults.append("project_selected_quality_token_cap_present")
    if int(budgets.get("maximum_generation_tokens_per_call") or 0) != int(
        generation.get("model_declared_context_window_tokens") or 0
    ):
        faults.append("physical_context_transport_binding_invalid")
    if generation.get("ceiling_hit_invalidates_observation") is not True:
        faults.append("context_boundary_disposition_invalid")
    instrument_audit = p4v2r2.audit_instrument(INSTRUMENT)
    if instrument_audit.get("trigger_state") != "GREEN":
        faults.append("instrument_audit_red")
    rows = p2a.dicts(pool.get("tasks"))
    if len(rows) != 10:
        faults.append("task_count_invalid")

    status_rows: list[dict[str, Any]] = []
    total_calls = 0
    total_ceiling_hits = 0
    total_parser_complete = 0
    total_model_eos = 0
    for expected, row in enumerate(rows, 1):
        stem = str(row.get("stem") or "")
        if int(row.get("campaign_index") or 0) != expected:
            faults.append(f"campaign_index_invalid:{stem}")
        task_path = ROOT / str(row.get("task") or "")
        evaluator_path = ROOT / str(row.get("evaluator") or "")
        if p2a.sha256_file(task_path) != str(row.get("task_sha256") or ""):
            faults.append(f"task_binding_invalid:{stem}")
        if p2a.sha256_file(evaluator_path) != str(
            row.get("evaluator_sha256") or ""
        ):
            faults.append(f"evaluator_binding_invalid:{stem}")
        paths = result_paths(row)
        run_exists = paths["run"].is_file()
        evaluation_exists = paths["evaluation"].is_file()
        receipts = runtime_reports(row)
        if evaluation_exists and not run_exists:
            faults.append(f"evaluation_without_run:{stem}")
        if receipts and not run_exists:
            faults.append(f"partial_unsealed_runtime_receipts:{stem}")
        if run_exists:
            run = p2a.read_json(paths["run"])
            if run.get("policy") != p4v2r2.POLICY:
                faults.append(f"run_policy_invalid:{stem}")
            if run.get("instrument_sha256") != p2a.sha256_file(INSTRUMENT):
                faults.append(f"run_instrument_binding_invalid:{stem}")
            if run.get("task_sha256") != p2a.sha256_file(task_path):
                faults.append(f"run_task_binding_invalid:{stem}")
            matched = p2a.mapping(run.get("matched_set"))
            if matched.get("ready") is not True:
                faults.append(f"matched_set_invalid:{stem}")
            denominators = p2a.mapping(run.get("denominators"))
            calls = int(denominators.get("model_calls") or 0)
            total_calls += calls
            if calls != 6:
                faults.append(f"model_call_count_invalid:{stem}")
            if int(denominators.get("model_loads") or 0) != 1:
                faults.append(f"model_load_count_invalid:{stem}")
            if denominators.get("project_selected_quality_token_cap") is not None:
                faults.append(f"run_quality_token_cap_present:{stem}")
            retained = sum(
                len(p2a.dicts(attempt.get("runtime_calls")))
                for attempt in p2a.dicts(run.get("attempts"))
            )
            telemetry = p2a.dicts(run.get("generation_termination_telemetry"))
            if retained != 6 or len(receipts) != 6 or len(telemetry) != 6:
                faults.append(f"runtime_receipt_count_invalid:{stem}")
            for call_index, termination in enumerate(telemetry, 1):
                reason = str(termination.get("termination_reason") or "")
                if reason not in NORMAL_TERMINATIONS:
                    faults.append(f"termination_invalid:{stem}:{call_index}")
                if termination.get("completion_predicate_enabled") is not True:
                    faults.append(f"completion_predicate_disabled:{stem}:{call_index}")
                if termination.get("safety_ceiling_hit") is True:
                    faults.append(
                        f"physical_context_boundary_hit:{stem}:{call_index}"
                    )
                    total_ceiling_hits += 1
                prompt_tokens = int(termination.get("prompt_tokens") or 0)
                generated_tokens = int(termination.get("generated_tokens") or 0)
                context_tokens = int(
                    termination.get("model_context_window_tokens") or 0
                )
                effective = int(termination.get("effective_maximum_tokens") or 0)
                if min(prompt_tokens, generated_tokens, context_tokens, effective) < 1:
                    faults.append(
                        f"termination_token_custody_invalid:{stem}:{call_index}"
                    )
                if effective > max(0, context_tokens - prompt_tokens):
                    faults.append(f"context_residual_exceeded:{stem}:{call_index}")
                total_parser_complete += int(reason == "parser_complete")
                total_model_eos += int(reason == "model_eos")
        if evaluation_exists:
            evaluation = p2a.read_json(paths["evaluation"])
            if evaluation.get("candidate_report_sha256") != p2a.sha256_file(
                paths["run"]
            ):
                faults.append(f"evaluation_run_binding_invalid:{stem}")
            if evaluation.get("evaluator_sha256") != p2a.sha256_file(
                evaluator_path
            ):
                faults.append(f"evaluation_evaluator_binding_invalid:{stem}")
            if evaluation.get("trigger_state") != "GREEN":
                faults.append(f"blind_evaluation_invalid:{stem}")
        status_rows.append(
            {
                "campaign_index": expected,
                "stem": stem,
                "run": relative(paths["run"]) if run_exists else "",
                "run_sha256": p2a.sha256_file(paths["run"]),
                "evaluation": (
                    relative(paths["evaluation"]) if evaluation_exists else ""
                ),
                "evaluation_sha256": p2a.sha256_file(paths["evaluation"]),
                "runtime_receipts": len(receipts),
                "complete": run_exists and evaluation_exists,
            }
        )
    complete = sum(row["complete"] for row in status_rows)
    pending = len(status_rows) - complete
    if total_calls != complete * 6:
        faults.append("campaign_retained_call_count_invalid")
    return {
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "scope": (
            "Exact sealed completion-based P4-v2r2 local decision-development "
            "campaign; no hosted, D1, D2, serving, training, or automatic "
            "book-support authority."
        ),
        "pool": relative(POOL),
        "pool_sha256": p2a.sha256_file(POOL),
        "pool_seal_commit": POOL_SEAL_COMMIT,
        "instrument": relative(INSTRUMENT),
        "instrument_sha256": p2a.sha256_file(INSTRUMENT),
        "runtime_attempt_namespace": RUNTIME_ATTEMPT_NAMESPACE,
        "preconsumption_bootstrap_repair": bootstrap_repair,
        "instrument_audit": instrument_audit,
        "complete_tasks": complete,
        "pending_tasks": pending,
        "model_calls_retained": total_calls,
        "parser_complete_calls": total_parser_complete,
        "model_eos_calls": total_model_eos,
        "safety_ceiling_hits": total_ceiling_hits,
        "project_selected_quality_token_cap": None,
        "tasks": status_rows,
        "hosted_reference": {
            "model": "gpt-5.6-luna",
            "effort": "xhigh",
            "state": "DEFINED_TRANSPORT_NOT_BOUND",
            "calls": 0,
            "P4V2R2_blocking": False,
        },
        "maximum_inference": (
            "Campaign execution custody only; terminal scientific status is "
            "computed separately from sealed blind evaluations."
        ),
    }


def result_paths(row: dict[str, Any]) -> dict[str, Path]:
    suffix = str(row.get("stem") or "").removeprefix("p4v2r2_")
    return {
        "run": ROOT / "reports" / f"theseus_p4v2r2_attempt2_{suffix}_run.json",
        "evaluation": (
            ROOT / "reports" / f"theseus_p4v2r2_attempt2_{suffix}_evaluation.json"
        ),
    }


def audit_preconsumption_bootstrap_failure() -> dict[str, Any]:
    faults: list[str] = []
    identities: dict[str, dict[str, Any]] = {}
    for name, (path, expected) in BOOTSTRAP_FAILURES.items():
        observed = p2a.sha256_file(path)
        if observed != expected:
            faults.append(f"bootstrap_failure_binding_invalid:{name}")
        identities[name] = {
            "path": relative(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
        }

    run = p2a.read_json(BOOTSTRAP_FAILURES["candidate_run"][0])
    if run.get("trigger_state") != "RED" or sorted(
        p2a.strings(run.get("faults"))
    ) != ["persistent_backend_not_ready", "qualified_mlx_runtime_missing"]:
        faults.append("bootstrap_failure_signature_invalid")
    if any(int(value or 0) != 0 for value in p2a.mapping(run.get("counters")).values()):
        faults.append("bootstrap_failure_counter_nonzero")

    progress = p2a.read_json(BOOTSTRAP_FAILURES["campaign_progress"][0])
    if (
        int(progress.get("complete_tasks") or 0) != 0
        or int(progress.get("model_calls_retained") or 0) != 0
        or int(progress.get("safety_ceiling_hits") or 0) != 0
    ):
        faults.append("bootstrap_failure_consumption_nonzero")

    launch = p2a.read_json(BOOTSTRAP_FAILURES["autonomous_launch"][0])
    final_audit = p2a.mapping(launch.get("final_campaign_audit"))
    if (
        launch.get("child_returncode") != 2
        or int(final_audit.get("model_calls_retained") or 0) != 0
        or int(final_audit.get("complete_tasks") or 0) != 0
    ):
        faults.append("bootstrap_launch_receipt_invalid")

    disposition = p2a.read_json(BOOTSTRAP_FAILURES["terminal_disposition"][0])
    denominators = p2a.mapping(disposition.get("denominators"))
    if (
        disposition.get("scientific_status") != "P4V2R2_REVIEW_REQUIRED"
        or int(denominators.get("learned_model_calls") or 0) != 0
        or int(denominators.get("tasks") or 0) != 0
    ):
        faults.append("bootstrap_disposition_invalid")

    return {
        "passed": not faults,
        "faults": sorted(set(faults)),
        "incident_commit": BOOTSTRAP_FAILURE_COMMIT,
        "failure_class": "qualified_runtime_executable_not_bound",
        "candidate_or_control_calls": 0,
        "tasks_consumed": 0,
        "candidate_outputs": 0,
        "surface_reuse_authorized": not faults,
        "runtime_attempt_namespace_reused_because_no_runtime_receipt_exists": True,
        "identities": identities,
        "maximum_inference": (
            "The preserved first launch proves only a pre-generation interpreter "
            "binding defect with zero model calls. It is not a model, mechanism, "
            "task, or token-boundary result."
        ),
    }


def runtime_reports(row: dict[str, Any]) -> list[Path]:
    task = p2a.read_json(ROOT / str(row.get("task") or ""))
    task_id = p2a.safe_slug(str(task.get("opaque_task_id") or ""))
    return sorted(
        (ROOT / "runtime" / "p2a").glob(
            f"*{task_id}*{RUNTIME_ATTEMPT_NAMESPACE}*.json"
        )
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
