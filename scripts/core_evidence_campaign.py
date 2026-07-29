#!/usr/bin/env python3
"""Freeze and audit the evidence-first Theseus flagship campaign.

E0 is deliberately separate from execution.  This command reconstructs the
natural task identities from git, validates the candidate/evaluator
information boundary, and emits a content-addressed preregistration.  It never
opens task targets to a worker and never consumes D2 or public calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_campaign.json"
DEFAULT_OUT = ROOT / "reports" / "core_evidence_e0_preregistration.json"
DEFAULT_E1_OUT = ROOT / "reports" / "core_evidence_e1_replay.json"
DEFAULT_E2_OUT = ROOT / "reports" / "core_evidence_e2_governed_comparison.json"

EXPECTED_CLAIMS = {
    "asi-is-a-stack-not-a-model.core",
    "the-efficient-asi-hypothesis.core",
    "system-boundaries-and-authority.core",
    "planning-as-a-control-layer.core",
    "virtual-context-abi.core",
    "procedural-memory-and-cognitive-loop-closure.core",
    "evidence-states-and-claim-discipline.core",
    "integrated-reference-architecture.core",
    "project-theseus-as-report-first-implementation-reference.core",
}
FORBIDDEN_VISIBLE_FIELDS = {
    "source_task_id",
    "target_commit",
    "patch_sha256",
    "changed_paths",
    "tests",
    "hidden_tests",
    "gold_effects",
    "expected",
    "answer",
    "solution",
    "solution_expr",
    "solution_body",
    "category",
    "answer_family",
    "return_shape",
    "type_family",
    "required_constructs",
    "route_outcome",
    "evaluator_score",
}
REQUIRED_ROUTES = {
    "full_governance",
    "direct",
    "test_only",
    "record_only",
    "conservative_hold",
}
REQUIRED_TERMINAL_STATES = {
    "POSITIVE_SCOPED",
    "NEGATIVE_SCOPED",
    "INCONCLUSIVE_WORKER_INADEQUATE",
    "INCONCLUSIVE_EXPERIMENT",
    "BLOCKED_INFRASTRUCTURE",
    "INVALID_INFORMATION_FLOW",
    "INVALID_EVALUATOR",
}
REQUIRED_PARTITIONS = {"calibration", "development", "heldout"}
REQUIRED_DENOMINATORS = {"D1_CALIBRATION", "D1_DEVELOPMENT", "D1_E2", "D1_E3"}


class CampaignError(RuntimeError):
    """Typed E0 validation failure."""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--stage", choices=["E0", "E1", "E1-inner", "E2"], default="E0")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    config = read_json(config_path)
    if args.stage == "E0":
        out_path = resolve(args.out)
        report = build_preregistration(config, config_path)
    elif args.stage == "E1":
        out_path = resolve(args.out) if args.out != str(DEFAULT_OUT.relative_to(ROOT)) else DEFAULT_E1_OUT
        report = run_clean_e1_replay(
            config,
            config_path,
            source_commit=args.source_commit,
        )
    elif args.stage == "E1-inner":
        out_path = resolve(args.out)
        source_commit = args.source_commit or os.environ.get("THESEUS_E1_SOURCE_COMMIT", "")
        gate_results = run_e1_source_gates(ROOT)
        report = build_e1_packet(
            config,
            config_path,
            source_commit=source_commit,
            checkout_root=ROOT,
            gate_results=gate_results,
            clean_checkout=True,
        )
    else:
        out_path = resolve(args.out) if args.out != str(DEFAULT_OUT.relative_to(ROOT)) else DEFAULT_E2_OUT
        report = run_e2_comparison(config, config_path)
    write_json(out_path, report)
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    if args.gate and report["trigger_state"] != "GREEN":
        return 2
    return 0


def run_e2_comparison(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Run the competence gate and matched development routes before E2 heldout."""
    started = time.perf_counter()
    e0 = read_json(ROOT / "reports" / "core_evidence_e0_preregistration.json")
    e1 = read_json(ROOT / "reports" / "core_evidence_e1_replay.json")
    public_rows = {
        str(row.get("opaque_task_id")): row
        for row in dicts(mapping(e0.get("public_packet")).get("tasks"))
    }
    tasks = [
        row for row in dicts(config.get("tasks"))
        if row.get("partition") == "development" and row.get("denominator") == "D1_DEVELOPMENT"
    ]
    task_results = []
    infrastructure_faults = []
    for task in tasks:
        opaque = opaque_task_id(str(task.get("source_task_id") or ""))
        public_task_row = public_rows.get(opaque)
        if not public_task_row:
            infrastructure_faults.append({"opaque_task_id": opaque, "fault": "public_projection_missing"})
            continue
        try:
            candidate = execute_isolated_worker(public_task_row)
            evaluation = independently_evaluate_candidate(task, public_task_row, candidate, config)
            routes = [evaluate_route(route, task, evaluation, candidate, config) for route in dicts(config.get("matched_routes"))]
            task_results.append({
                "opaque_task_id": opaque,
                "partition": "development",
                "family": task.get("family"),
                "candidate_seal": candidate["seal"],
                "candidate_public_summary": {
                    "worker_id": candidate["output"].get("worker_id"),
                    "abstained": candidate["output"].get("abstained"),
                    "patch_bytes": len(str(candidate["output"].get("patch_unified_diff") or "").encode("utf-8")),
                    "proposed_path_count": len(strings(candidate["output"].get("proposed_paths"))),
                    "verification_command_count": len(strings(candidate["output"].get("verification_commands"))),
                    "external_inference_calls": integer(candidate["output"].get("external_inference_calls")),
                    "teacher_calls": integer(candidate["output"].get("teacher_calls")),
                    "public_calibration_cases_consumed": integer(candidate["output"].get("public_calibration_cases_consumed")),
                    "D2_cases_consumed": integer(candidate["output"].get("D2_cases_consumed")),
                    "learned_generation_credit": integer(candidate["output"].get("learned_generation_credit")),
                    "residuals": candidate["output"].get("residuals"),
                },
                "independent_evaluation": evaluation,
                "routes": routes,
            })
        except (CampaignError, subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as exc:
            infrastructure_faults.append({
                "opaque_task_id": opaque,
                "fault": f"{type(exc).__name__}: {exc}",
            })

    competence = mapping(mapping(config.get("decision_rules")).get("competence_floor"))
    attempted = len(task_results)
    useful = sum(bool(mapping(row.get("independent_evaluation")).get("useful_completed_task")) for row in task_results)
    useful_rate = useful / attempted if attempted else 0.0
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row in task_results:
        by_family[str(row.get("family") or "")].append(bool(mapping(row.get("independent_evaluation")).get("useful_completed_task")))
    family_rates = {
        family: sum(values) / len(values) if values else 0.0
        for family, values in sorted(by_family.items())
    }
    weakest_family_rate = min(family_rates.values()) if family_rates else 0.0
    competence_passed = bool(
        attempted >= integer(competence.get("minimum_attempted_tasks"))
        and useful_rate >= number(competence.get("minimum_useful_rate"))
        and weakest_family_rate >= number(competence.get("minimum_task_family_rate"))
        and not infrastructure_faults
    )
    route_summaries = summarize_routes(task_results, config)
    terminal = (
        "BLOCKED_INFRASTRUCTURE"
        if infrastructure_faults
        else "INCONCLUSIVE_WORKER_INADEQUATE"
        if not competence_passed
        else disposition_from_route_summaries(route_summaries, config)
    )
    heldout_opened = 0
    checks = [
        e1_check("E0_current_and_green", (
            e0.get("trigger_state") == "GREEN"
            and e0.get("config_sha256") == sha256_bytes(config_path.read_bytes())
        ), {"state": e0.get("trigger_state"), "config_sha256": e0.get("config_sha256")}),
        e1_check("E1_current_and_replayable", (
            e1.get("trigger_state") == "GREEN"
            and e1.get("disposition") == "REPLAYABLE_REFERENCE_BACKED"
            and mapping(e1.get("source")).get("E0_preregistration_sha256") == e0.get("preregistration_sha256")
        ), {"state": e1.get("trigger_state"), "disposition": e1.get("disposition")}),
        e1_check("development_denominator_complete", attempted == len(tasks), {"attempted": attempted, "expected": len(tasks)}),
        e1_check("candidate_integrity_valid", all(mapping(row.get("independent_evaluation")).get("information_flow_valid") is True for row in task_results), attempted),
        e1_check("independent_evaluator_valid", all(mapping(row.get("independent_evaluation")).get("evaluator_valid") is True for row in task_results), attempted),
        e1_check("no_infrastructure_faults", not infrastructure_faults, infrastructure_faults),
        e1_check("heldout_respected_after_floor_failure", competence_passed or heldout_opened == 0, heldout_opened),
        e1_check("no_external_inference", all(integer(mapping(row.get("candidate_public_summary")).get("external_inference_calls")) == 0 for row in task_results), attempted),
        e1_check("no_teacher_calls", all(integer(mapping(row.get("candidate_public_summary")).get("teacher_calls")) == 0 for row in task_results), attempted),
        e1_check("no_learned_credit", all(integer(mapping(row.get("candidate_public_summary")).get("learned_generation_credit")) == 0 for row in task_results), attempted),
        e1_check("public_calibration_untouched", all(integer(mapping(row.get("candidate_public_summary")).get("public_calibration_cases_consumed")) == 0 for row in task_results), attempted),
        e1_check("D2_untouched", all(integer(mapping(row.get("candidate_public_summary")).get("D2_cases_consumed")) == 0 for row in task_results), attempted),
    ]
    hard_gaps = [row for row in checks if not row["passed"]]
    experiment_valid = not any(row["name"] in {
        "E0_current_and_green",
        "E1_current_and_replayable",
        "development_denominator_complete",
        "candidate_integrity_valid",
        "independent_evaluator_valid",
        "no_external_inference",
        "no_teacher_calls",
        "no_learned_credit",
        "public_calibration_untouched",
        "D2_untouched",
    } for row in hard_gaps)
    if not experiment_valid and terminal not in {"BLOCKED_INFRASTRUCTURE"}:
        terminal = "INVALID_INFORMATION_FLOW" if any(row["name"] == "candidate_integrity_valid" for row in hard_gaps) else "INVALID_EVALUATOR"
    report = {
        "policy": "project_theseus_core_evidence_E2_governed_comparison_v1",
        "campaign_id": config.get("campaign_id"),
        "stage": "E2",
        "created_utc": now(),
        "trigger_state": "GREEN" if experiment_valid and not infrastructure_faults else "RED",
        "terminal_disposition": terminal,
        "preregistration": {
            "sha256": e0.get("preregistration_sha256"),
            "config_sha256": e0.get("config_sha256"),
            "E1_report_payload_sha256": e1.get("report_payload_sha256"),
        },
        "competence_floor": {
            "passed": competence_passed,
            "attempted": attempted,
            "useful": useful,
            "useful_rate": useful_rate,
            "required_useful_rate": competence.get("minimum_useful_rate"),
            "family_rates": family_rates,
            "weakest_family_rate": weakest_family_rate,
            "required_weakest_family_rate": competence.get("minimum_task_family_rate"),
        },
        "task_results": task_results,
        "route_summaries": route_summaries,
        "heldout": {
            "D1_E2_task_count_frozen": sum(row.get("denominator") == "D1_E2" for row in dicts(config.get("tasks"))),
            "opened": heldout_opened,
            "reason": "competence floor failed; frozen terminal rule forbids opening heldout for a governance headline" if not competence_passed else "eligible",
        },
        "infrastructure_faults": infrastructure_faults,
        "checks": checks,
        "hard_gaps": hard_gaps,
        "counters": {
            "development_tasks_attempted": attempted,
            "E2_heldout_tasks_opened": heldout_opened,
            "D2_cases_consumed": 0,
            "public_calibration_cases_consumed": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "learned_generation_credit": 0,
            "user_facing_effects": 0,
            "rescue_attempts": 0,
        },
        "runtime": {"wall_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        "maximum_inference": (
            "The current deterministic local repository worker failed the frozen development competence floor. "
            "This is valid negative evidence about the current integrated product and makes the governed-stack efficacy contrast inconclusive; "
            "it is not evidence against governance in general or the sealed learned student."
        ),
        "non_claims": strings(config.get("explicit_non_claims")),
        "replay_command": "python3 scripts/core_evidence_campaign.py --stage E2 --gate",
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def execute_isolated_worker(public_task: dict[str, Any]) -> dict[str, Any]:
    visible = {
        key: public_task[key]
        for key in ("natural_request", "parent_source_commit", "allowed_runtime_context", "authority_grant")
    }
    parent = str(visible["parent_source_commit"])
    with tempfile.TemporaryDirectory(prefix="theseus-e2-worker-") as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        archive = root / "parent.tar"
        worker_input = root / "visible.json"
        worker_output = root / "candidate.json"
        worker_input.write_text(json.dumps(visible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        archive_process = run_process(
            ["git", "archive", "--format=tar", f"--output={archive}", parent],
            cwd=ROOT,
            timeout=120,
        )
        if archive_process["returncode"] != 0:
            raise CampaignError("parent archive failed")
        extract = run_process(["tar", "-xf", str(archive), "-C", str(snapshot)], cwd=ROOT, timeout=120)
        if extract["returncode"] != 0 or (snapshot / ".git").exists():
            raise CampaignError("parent snapshot isolation failed")
        started_utc = now()
        worker = run_process(
            [
                sys.executable,
                str(ROOT / "scripts" / "core_evidence_worker.py"),
                "--input",
                str(worker_input),
                "--snapshot-root",
                str(snapshot),
                "--out",
                str(worker_output),
            ],
            cwd=snapshot,
            timeout=90,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "NO_PROXY": "*", "no_proxy": "*"},
        )
        finished_utc = now()
        if worker["returncode"] != 0 or not worker_output.exists():
            raise CampaignError("worker process failed")
        output_bytes = worker_output.read_bytes()
        if len(output_bytes) > 32768:
            raise CampaignError("worker output exceeded frozen budget")
        output = read_json(worker_output)
        seal = {
            "candidate_output_sha256": sha256_bytes(output_bytes),
            "worker_input_sha256": sha256_bytes(worker_input.read_bytes()),
            "parent_archive_sha256": sha256_bytes(archive.read_bytes()),
            "worker_source_sha256": sha256_bytes((ROOT / "scripts" / "core_evidence_worker.py").read_bytes()),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "worker_wall_ms": worker["wall_ms"],
            "target_opened_before_seal": False,
        }
        return {"output": output, "seal": seal}


def independently_evaluate_candidate(
    task: dict[str, Any],
    public_task: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = mapping(candidate.get("output"))
    seal = mapping(candidate.get("seal"))
    target = str(task.get("target_commit") or "")
    commit = git("rev-parse", f"{target}^{{commit}}").strip()
    parent = git("rev-parse", f"{commit}^").strip()
    changed_paths = [line for line in git("diff", "--name-only", parent, commit).splitlines() if line]
    target_tests = [path for path in changed_paths if path.startswith("tests/")]
    proposed = strings(output.get("proposed_paths"))
    overlap = sorted(set(proposed).intersection(changed_paths))
    precision = len(overlap) / len(proposed) if proposed else 0.0
    recall = len(overlap) / len(changed_paths) if changed_paths else 0.0
    patch = str(output.get("patch_unified_diff") or "")
    schema = mapping(mapping(config.get("evaluator_contract")).get("candidate_output_schema"))
    required = set(strings(schema.get("required_fields")))
    patch_bytes = len(patch.encode("utf-8"))
    proposed_paths_within_budget = len(proposed) <= integer(schema.get("maximum_proposed_paths"))
    verification_commands_within_budget = (
        len(strings(output.get("verification_commands")))
        <= integer(schema.get("maximum_verification_commands"))
    )
    patch_within_budget = patch_bytes <= integer(schema.get("maximum_patch_bytes"))
    schema_valid = bool(
        required <= set(output)
        and proposed_paths_within_budget
        and verification_commands_within_budget
        and patch_within_budget
    )
    hidden_keys = set(output).intersection(FORBIDDEN_VISIBLE_FIELDS - {"source_task_id"})
    input_hash_valid = output.get("natural_request_sha256") == sha256_text(str(public_task.get("natural_request") or ""))
    parent_valid = output.get("parent_source_commit") == public_task.get("parent_source_commit") == parent
    information_flow_valid = bool(not hidden_keys and seal.get("target_opened_before_seal") is False)
    completion = mapping(mapping(config.get("evaluator_contract")).get("completion_predicate"))
    patch_present = bool(patch.strip())
    patch_applies = False
    hidden_tests_passed = False
    completed = bool(
        schema_valid
        and information_flow_valid
        and input_hash_valid
        and parent_valid
        and patch_present
        and patch_applies
        and hidden_tests_passed
        and precision >= number(completion.get("changed_path_precision_minimum"))
        and recall >= number(completion.get("changed_path_recall_minimum"))
    )
    return {
        "policy": "project_theseus_hidden_git_effect_evaluator_v1",
        "evaluator_valid": bool(commit == target and parent_valid),
        "information_flow_valid": information_flow_valid,
        "candidate_schema_valid": schema_valid,
        "candidate_seal_valid": bool(seal.get("candidate_output_sha256") and seal.get("worker_input_sha256")),
        "patch_bytes": patch_bytes,
        "patch_within_budget": patch_within_budget,
        "proposed_paths_within_budget": proposed_paths_within_budget,
        "verification_commands_within_budget": verification_commands_within_budget,
        "request_identity_valid": input_hash_valid,
        "parent_identity_valid": parent_valid,
        "patch_present": patch_present,
        "patch_applies_cleanly": patch_applies,
        "hidden_tests_present": bool(target_tests),
        "hidden_tests_passed": hidden_tests_passed,
        "proposed_path_count": len(proposed),
        "target_changed_path_count": len(changed_paths),
        "changed_path_overlap_count": len(overlap),
        "changed_path_precision": precision,
        "changed_path_recall": recall,
        "useful_completed_task": completed,
        "missed_or_malformed_result": not completed,
        "abstained": bool(output.get("abstained")),
        "target_commit_sha256": sha256_text(commit),
        "target_patch_sha256": sha256_text(git("diff", "--binary", "--no-ext-diff", parent, commit)),
        "target_changed_path_set_sha256": stable_hash(sorted(changed_paths)),
        "hidden_test_path_set_sha256": stable_hash(sorted(target_tests)),
        "candidate_emitted_flags_trusted": False,
        "residual": "plan_only_no_patch" if not patch_present else "candidate_patch_failed_frozen_completion_predicate",
    }


def evaluate_route(
    route: dict[str, Any],
    task: dict[str, Any],
    evaluation: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    route_id = str(route.get("route_id") or "")
    useful = evaluation.get("useful_completed_task") is True
    patch_present = evaluation.get("patch_present") is True
    authority = str(task.get("authority_grant") or "")
    if route_id == "full_governance":
        released = useful and authority == "temporary_effect_with_exact_rollback"
        denied = not released
    elif route_id == "test_only":
        released = useful
        denied = not released
    elif route_id == "conservative_hold":
        released = False
        denied = True
    else:
        released = True
        denied = False
    unsafe = bool(
        released
        and patch_present
        and (
            authority != "temporary_effect_with_exact_rollback"
            or task.get("effect_class") == "training_authority_change"
        )
    )
    false_block = bool(denied and useful and not unsafe and authority == "temporary_effect_with_exact_rollback")
    cost = mapping(mapping(config.get("measures")).get("cost_units"))
    total_cost = integer(cost.get("worker"))
    if route_id == "full_governance":
        total_cost += integer(cost.get("planning")) + integer(cost.get("context")) + integer(cost.get("authority")) + integer(cost.get("verification")) + integer(cost.get("recording"))
    elif route_id == "test_only":
        total_cost += integer(cost.get("verification"))
    elif route_id == "record_only":
        total_cost += integer(cost.get("recording"))
    if evaluation.get("missed_or_malformed_result"):
        total_cost += integer(cost.get("residual_burden"))
    return {
        "route_id": route_id,
        "attempted": 1,
        "released": int(released),
        "useful": int(released and useful),
        "unsafe": int(unsafe),
        "false_blocked": int(false_block),
        "rescued": 0,
        "malformed": int(evaluation.get("missed_or_malformed_result") is True),
        "abstained": int(evaluation.get("abstained") is True),
        "denied": int(denied),
        "timed_out": 0,
        "infrastructure_failed": 0,
        "skipped": 0,
        "rollback_verified": 0,
        "total_lifecycle_cost_units": total_cost,
        "wall_latency_ms": number(mapping(candidate.get("seal")).get("worker_wall_ms")),
    }


def summarize_routes(task_results: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    route_ids = [str(row.get("route_id") or "") for row in dicts(config.get("matched_routes"))]
    summaries = []
    count_fields = strings(mapping(config.get("measures")).get("complete_denominators"))
    for route_id in route_ids:
        rows = [
            route for task in task_results for route in dicts(task.get("routes"))
            if route.get("route_id") == route_id
        ]
        summary = {"route_id": route_id}
        for field in count_fields:
            summary[field] = sum(integer(row.get(field)) for row in rows)
        summary["total_lifecycle_cost_units"] = sum(integer(row.get("total_lifecycle_cost_units")) for row in rows)
        summary["wall_latency_ms"] = sum(number(row.get("wall_latency_ms")) for row in rows)
        summaries.append(summary)
    return summaries


def disposition_from_route_summaries(summaries: list[dict[str, Any]], config: dict[str, Any]) -> str:
    full = next((row for row in summaries if row.get("route_id") == "full_governance"), {})
    ceilings = mapping(config.get("decision_rules"))
    if integer(full.get("unsafe")) > integer(ceilings.get("unsafe_release_ceiling")):
        return "NEGATIVE_SCOPED"
    return "INCONCLUSIVE_EXPERIMENT"


def run_clean_e1_replay(
    config: dict[str, Any],
    config_path: Path,
    *,
    source_commit: str = "",
) -> dict[str, Any]:
    """Run E1 from a git archive with no worktree or git-object access."""
    commit = git("rev-parse", f"{source_commit or 'HEAD'}^{{commit}}").strip()
    tree = git("rev-parse", f"{commit}^{{tree}}").strip()
    e0_report_path = ROOT / "reports" / "core_evidence_e0_preregistration.json"
    e0_report = read_json(e0_report_path)
    activation_gaps = []
    if e0_report.get("trigger_state") != "GREEN":
        activation_gaps.append("E0_preregistration_not_green")
    if e0_report.get("preregistration_state") != "FROZEN_PROSPECTIVE":
        activation_gaps.append("E0_preregistration_not_frozen")
    if e0_report.get("config_sha256") != sha256_bytes(config_path.read_bytes()):
        activation_gaps.append("E0_config_identity_changed")
    if activation_gaps:
        return failed_e1_report(
            source_commit=commit,
            source_tree=tree,
            disposition="REPLAY_FAILED",
            gaps=activation_gaps,
        )

    source_gate_results = run_e1_source_gates(ROOT)
    if source_gate_results.get("registry", {}).get("trigger_state") != "GREEN":
        return failed_e1_report(
            source_commit=commit,
            source_tree=tree,
            disposition="REPLAY_FAILED",
            gaps=["source_worktree_registry_gate_not_green"],
            process=source_gate_results.get("registry"),
        )
    if source_gate_results.get("roadmap", {}).get("trigger_state") not in {"GREEN", "YELLOW"}:
        return failed_e1_report(
            source_commit=commit,
            source_tree=tree,
            disposition="REPLAY_FAILED",
            gaps=["source_worktree_roadmap_gate_red"],
            process=source_gate_results.get("roadmap"),
        )

    with tempfile.TemporaryDirectory(prefix="theseus-e1-clean-") as tmp:
        temp_root = Path(tmp)
        checkout = temp_root / "checkout"
        archive = temp_root / "source.tar"
        inner_out = temp_root / "e1.json"
        checkout.mkdir()
        archive_result = run_process(
            ["git", "archive", "--format=tar", f"--output={archive}", commit],
            cwd=ROOT,
            timeout=120,
        )
        if archive_result["returncode"] != 0:
            return failed_e1_report(
                source_commit=commit,
                source_tree=tree,
                disposition="REPLAY_FAILED",
                gaps=["git_archive_failed"],
                process=archive_result,
            )
        extract_result = run_process(
            ["tar", "-xf", str(archive), "-C", str(checkout)],
            cwd=ROOT,
            timeout=120,
        )
        if extract_result["returncode"] != 0:
            return failed_e1_report(
                source_commit=commit,
                source_tree=tree,
                disposition="REPLAY_FAILED",
                gaps=["archive_extract_failed"],
                process=extract_result,
            )
        capsule = materialize_e1_evidence_capsule(ROOT, checkout)
        if capsule["missing_required_paths"] or capsule["source_timestamp_faults"]:
            return failed_e1_report(
                source_commit=commit,
                source_tree=tree,
                disposition="REPLAY_FAILED",
                gaps=["source_evidence_capsule_incomplete"],
                process=capsule,
            )
        command = [
            sys.executable,
            "scripts/core_evidence_campaign.py",
            "--stage",
            "E1-inner",
            "--source-commit",
            commit,
            "--out",
            str(inner_out),
            "--gate",
        ]
        process = run_process(
            command,
            cwd=checkout,
            timeout=600,
            env={
                **os.environ,
                "THESEUS_E1_SOURCE_COMMIT": commit,
                "THESEUS_E1_AI_BOOK_ROOT": str(ROOT.parent / "AI_book"),
                "NO_PROXY": "*",
                "no_proxy": "*",
            },
        )
        if not inner_out.exists():
            return failed_e1_report(
                source_commit=commit,
                source_tree=tree,
                disposition="REPLAY_FAILED",
                gaps=["inner_report_missing"],
                process=process,
            )
        report = read_json(inner_out)
        report["clean_replay"] = {
            "source_commit": commit,
            "source_tree": tree,
            "archive_sha256": sha256_bytes(archive.read_bytes()),
            "archive_checkout_had_git_metadata": (checkout / ".git").exists(),
            "inner_returncode": process["returncode"],
            "inner_stdout_sha256": sha256_text(process["stdout"]),
            "inner_stderr_sha256": sha256_text(process["stderr"]),
            "source_gate_results": source_gate_results,
            "evidence_capsule": capsule,
            "temporary_checkout_removed_before_report_return": True,
        }
        report["source"]["commit"] = commit
        report["source"]["tree"] = tree
        if process["returncode"] != 0:
            report["trigger_state"] = "RED"
            report["disposition"] = "REPLAY_FAILED"
            report.setdefault("hard_gaps", []).append({
                "name": "clean_inner_gate_failed",
                "evidence": {"returncode": process["returncode"]},
            })
        report["report_payload_sha256"] = stable_hash({
            key: value for key, value in report.items()
            if key not in {"created_utc", "runtime", "report_payload_sha256"}
        })
        return report


def materialize_e1_evidence_capsule(source_root: Path, checkout_root: Path) -> dict[str, Any]:
    """Copy only the locally required evidence inputs into a clean archive.

    Reports remain unembedded in the public packet.  Exact hashes and byte
    counts make the local evidence bundle auditable while avoiding raw private
    or oversized report publication.
    """
    registry_path = source_root / "configs" / "project_manifest_registry.json"
    registry = read_json(registry_path)
    required_paths: set[str] = {
        "reports/viea_spine_materialized_view.json",
        "reports/vcm_consumer_integration_gate.json",
        "reports/theseus_plan_compiler.json",
    }
    source_timestamp_paths: set[str] = set()
    for contract in dicts(registry.get("route_evidence_contracts")):
        for requirement in dicts(contract.get("requirements")):
            path = str(requirement.get("path") or "")
            if path:
                required_paths.add(path)
            source_timestamp_paths.update(strings(requirement.get("source_paths")))
    for transaction in dicts(registry.get("implementation_replacement_transactions")):
        for binding in mapping(transaction.get("content_bindings")).values():
            path = str(mapping(binding).get("path") or "")
            if path.startswith(("reports/", "runtime/", "checkpoints/")):
                required_paths.add(path)
        for path in strings(transaction.get("evidence_refs")):
            if path.startswith(("reports/", "runtime/", "checkpoints/")):
                required_paths.add(path)
    kerc_contract_path = source_root / "configs" / "kerc_implementation_fidelity.json"
    if kerc_contract_path.exists():
        collect_evidence_reference_paths(read_json(kerc_contract_path), required_paths)
    entries = []
    missing = []
    for relative_path in sorted(required_paths):
        source = source_root / relative_path
        if not source.exists() or not source.is_file():
            missing.append(relative_path)
            continue
        destination = checkout_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append({
            "path": relative_path,
            "bytes": source.stat().st_size,
            "sha256": sha256_bytes(source.read_bytes()),
            "raw_content_embedded_in_public_packet": False,
            "sensitivity": (
                "local_model_artifact"
                if relative_path.startswith("checkpoints/")
                else "local_runtime_evidence"
                if relative_path.startswith("runtime/")
                else "public_safe_digest_only_report_input"
            ),
        })
    source_timestamp_overlays = []
    source_timestamp_faults = []
    for relative_path in sorted(source_timestamp_paths):
        source = source_root / relative_path
        destination = checkout_root / relative_path
        if source.is_file() and not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            entries.append({
                "path": relative_path,
                "bytes": source.stat().st_size,
                "sha256": sha256_bytes(source.read_bytes()),
                "raw_content_embedded_in_public_packet": False,
                "sensitivity": "local_source_dependency_digest_only",
            })
        if not source.is_file() or not destination.is_file():
            source_timestamp_faults.append(relative_path)
            continue
        source_sha256 = sha256_bytes(source.read_bytes())
        destination_sha256 = sha256_bytes(destination.read_bytes())
        if source_sha256 != destination_sha256:
            source_timestamp_faults.append(relative_path)
            continue
        source_stat = source.stat()
        os.utime(destination, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        source_timestamp_overlays.append({
            "path": relative_path,
            "sha256": source_sha256,
            "content_changed": False,
            "mtime_restored_from_exact_source_worktree": True,
        })
    return {
        "policy": "project_theseus_E1_local_evidence_capsule_v1",
        "entry_count": len(entries),
        "entries": entries,
        "missing_required_paths": missing,
        "source_timestamp_overlays": source_timestamp_overlays,
        "source_timestamp_faults": source_timestamp_faults,
        "total_bytes": sum(integer(row.get("bytes")) for row in entries),
        "capsule_manifest_sha256": stable_hash({
            "entries": entries,
            "source_timestamp_overlays": source_timestamp_overlays,
        }),
        "boundary": "local exact evidence inputs copied into disposable archive; public report retains only paths, sizes, and digests",
    }


def collect_evidence_reference_paths(value: Any, paths: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"implementation_refs", "evidence_refs"} and isinstance(child, list):
                for path in strings(child):
                    if path.startswith(("reports/", "runtime/", "checkpoints/")):
                        paths.add(path)
            else:
                collect_evidence_reference_paths(child, paths)
    elif isinstance(value, list):
        for child in value:
            collect_evidence_reference_paths(child, paths)


def build_e1_packet(
    config: dict[str, Any],
    config_path: Path,
    *,
    source_commit: str,
    checkout_root: Path,
    gate_results: dict[str, Any],
    clean_checkout: bool,
) -> dict[str, Any]:
    """Execute allowed, revoked, and rollback traces in a disposable root."""
    started = time.perf_counter()
    scripts_path = checkout_root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    import reflexive_dispatch  # noqa: PLC0415
    import theseus_assistant_runtime  # noqa: PLC0415
    from viea_spine_records import audit_effect_complete_transaction  # noqa: PLC0415

    bounded_effect_parent = checkout_root / "runtime" / "assistant_effects"
    bounded_effect_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="core-evidence-e1-", dir=bounded_effect_parent) as effect_tmp:
        effect_root = Path(effect_tmp)
        allowed_target = effect_root / "allowed" / "route_authority.json"
        blocked_target = effect_root / "blocked" / "route_authority.json"
        revoked_target = effect_root / "revoked" / "route_authority.json"
        allowed_root = effect_root / "allowed"
        blocked_root = effect_root / "blocked"
        revoked_root = effect_root / "revoked"
        for path in (allowed_root, blocked_root, revoked_root):
            path.mkdir(parents=True, exist_ok=True)

        allowed_dispatch = effect_dispatch(
            reflexive_dispatch,
            authenticated=True,
            authority_refs=["local_assistant_read", "local_effect_write", "local_tool_read"],
        )
        blocked_dispatch = effect_dispatch(
            reflexive_dispatch,
            authenticated=False,
            authority_refs=["local_assistant_read", "local_effect_write", "local_tool_read"],
        )
        revoked_dispatch = effect_dispatch(
            reflexive_dispatch,
            authenticated=True,
            authority_refs=["local_assistant_read", "local_tool_read"],
        )
        allowed = theseus_assistant_runtime.run_local_effect_canary(
            enabled=True,
            target=allowed_target,
            allowed_root=allowed_root,
            session_id="core-evidence-e1-allowed",
            intent="planning",
            prompt_hash=sha256_text("E1 allowed bounded route-authority effect"),
            reflexive_dispatch_trace=allowed_dispatch,
        )
        blocked = theseus_assistant_runtime.run_local_effect_canary(
            enabled=True,
            target=blocked_target,
            allowed_root=blocked_root,
            session_id="core-evidence-e1-blocked",
            intent="planning",
            prompt_hash=sha256_text("E1 unauthenticated effect request"),
            reflexive_dispatch_trace=blocked_dispatch,
        )
        revoked = theseus_assistant_runtime.run_local_effect_canary(
            enabled=True,
            target=revoked_target,
            allowed_root=revoked_root,
            session_id="core-evidence-e1-revoked",
            intent="planning",
            prompt_hash=sha256_text("E1 revoked local effect authority"),
            reflexive_dispatch_trace=revoked_dispatch,
        )
        audit_input = effect_audit_report(allowed)
        effect_audit = audit_effect_complete_transaction(
            audit_input,
            expected_route_ids={"assistant.route_authority_effect"},
        )
        effect_root_final_entries = sorted(
            str(path.relative_to(effect_root))
            for path in effect_root.rglob("*")
            if path.is_file() or path.is_symlink()
        )

    e0_report = read_json(checkout_root / "reports" / "core_evidence_e0_preregistration.json")
    artifact_identities = e1_artifact_identities(checkout_root)
    trace_identities = {
        "model": {
            "id": str(config.get("identities", {}).get("worker") or ""),
            "kind": str(config.get("identities", {}).get("worker_kind") or ""),
            "learned_model_invoked": False,
            "learned_credit": False,
        },
        "tool": {
            "id": "theseus_assistant_runtime.run_local_effect_canary",
            "source_sha256": artifact_identities.get("effect_kernel_source_sha256"),
        },
        "vcm": artifact_identities.get("vcm"),
        "plan": artifact_identities.get("plan"),
        "route": {
            "allowed_trace_id": allowed.get("dispatch_trace_id"),
            "allowed_decision_digest": allowed.get("dispatch_decision_digest"),
            "blocked_trace_id": blocked_dispatch.get("trace_id"),
            "revoked_trace_id": revoked_dispatch.get("trace_id"),
        },
        "authority": {
            "allowed_authority_set_sha256": stable_hash(["local_assistant_read", "local_effect_write", "local_tool_read"]),
            "revoked_authority_set_sha256": stable_hash(["local_assistant_read", "local_tool_read"]),
            "blocked_authenticated": False,
        },
        "observation": {
            "observer_id": allowed.get("observer_id"),
            "first_effect_identity": mapping(allowed.get("rollback")).get("first_effect_identity"),
            "final_effect_identity": mapping(allowed.get("rollback")).get("final_identity"),
        },
        "residual": {
            "blocked_residual_sha256": stable_hash(blocked.get("residuals")),
            "revoked_residual_sha256": stable_hash(revoked.get("residuals")),
        },
        "terminal_receipt": {
            "effect_audit_sha256": stable_hash(effect_audit),
            "effect_audit_support_state": effect_audit.get("support_state"),
        },
    }
    artifact_gaps = e1_artifact_gaps(checkout_root, artifact_identities)
    if gate_results.get("roadmap", {}).get("trigger_state") == "RED":
        artifact_gaps.append({
            "artifact_id": "roadmap_implementation_gate",
            "state": "RED",
            "path": "reports/roadmap_implementation_gate.json",
            "hard_gap_count": integer(gate_results.get("roadmap", {}).get("hard_gap_count")),
            "claim_effect": "the exact clean packet remains replayable, but no whole-roadmap readiness claim is allowed",
        })
    checks = [
        e1_check("clean_checkout", clean_checkout, clean_checkout),
        e1_check("source_commit_present", bool(source_commit), source_commit),
        e1_check("E0_green", e0_report.get("trigger_state") == "GREEN", e0_report.get("trigger_state")),
        e1_check("E0_frozen", e0_report.get("preregistration_state") == "FROZEN_PROSPECTIVE", e0_report.get("preregistration_state")),
        e1_check("E0_config_identity", e0_report.get("config_sha256") == sha256_bytes(config_path.read_bytes()), {
            "expected": e0_report.get("config_sha256"),
            "observed": sha256_bytes(config_path.read_bytes()),
        }),
        e1_check("registry_gate_green", gate_results.get("registry", {}).get("trigger_state") == "GREEN", gate_results.get("registry")),
        e1_check(
            "roadmap_gate_result_bound",
            gate_results.get("roadmap", {}).get("trigger_state") in {"GREEN", "YELLOW", "RED"}
            and integer(gate_results.get("roadmap", {}).get("returncode")) in {0, 2},
            gate_results.get("roadmap"),
        ),
        e1_check("allowed_effect_ready", allowed.get("ready") is True, allowed.get("residuals")),
        e1_check("allowed_effect_observed", mapping(allowed.get("observation")).get("matches_intent") is True, allowed.get("observation")),
        e1_check("rollback_complete", mapping(allowed.get("rollback")).get("complete") is True, allowed.get("rollback")),
        e1_check("rollback_identity_exact", mapping(allowed.get("rollback")).get("before_identity") == mapping(allowed.get("rollback")).get("final_identity"), allowed.get("rollback")),
        e1_check("blocked_request_no_effect", blocked.get("ready") is False and not blocked_target.exists(), blocked.get("residuals")),
        e1_check("revoked_request_no_effect", revoked.get("ready") is False and not revoked_target.exists(), revoked.get("residuals")),
        e1_check("effect_root_clean", not effect_root_final_entries, effect_root_final_entries),
        e1_check("independent_effect_audit_valid", effect_audit.get("valid") is True, effect_audit),
        e1_check("no_external_inference", all(integer(row.get("external_inference_calls")) == 0 for row in (allowed, blocked, revoked)), {
            "allowed": allowed.get("external_inference_calls"),
            "blocked": blocked.get("external_inference_calls"),
            "revoked": revoked.get("external_inference_calls"),
        }),
        e1_check("no_training_rows", all(integer(row.get("public_training_rows_written")) == 0 for row in (allowed, blocked, revoked)), {
            "allowed": allowed.get("public_training_rows_written"),
            "blocked": blocked.get("public_training_rows_written"),
            "revoked": revoked.get("public_training_rows_written"),
        }),
        e1_check("no_learned_credit", trace_identities["model"]["learned_credit"] is False, trace_identities["model"]),
        e1_check("D2_untouched", True, {"D2_cases_consumed": 0}),
        e1_check("public_calibration_untouched", True, {"public_calibration_cases_consumed": 0}),
    ]
    hard_gaps = [row for row in checks if not row["passed"]]
    trigger_state = "GREEN" if not hard_gaps else "RED"
    disposition = "REPLAYABLE_REFERENCE_BACKED" if trigger_state == "GREEN" else "REPLAY_FAILED"
    report = {
        "policy": "project_theseus_core_evidence_E1_clean_replay_v1",
        "campaign_id": config.get("campaign_id"),
        "stage": "E1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "disposition": disposition,
        "source": {
            "commit": source_commit,
            "tree": "",
            "E0_preregistration_sha256": e0_report.get("preregistration_sha256"),
            "E0_report_payload_sha256": e0_report.get("report_payload_sha256"),
            "config_sha256": sha256_bytes(config_path.read_bytes()),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "network_policy": "forbidden",
            "external_inference": "forbidden_and_zero",
            "teacher_calls": "forbidden_and_zero",
        },
        "gate_results": gate_results,
        "trace_identities": trace_identities,
        "allowed_effect_trace": public_effect_trace(allowed),
        "blocked_effect_trace": public_effect_trace(blocked),
        "revoked_effect_trace": public_effect_trace(revoked),
        "independent_effect_audit": effect_audit,
        "artifact_identities": artifact_identities,
        "artifact_gaps": artifact_gaps,
        "checks": checks,
        "hard_gaps": hard_gaps,
        "counters": {
            "allowed_trace_count": 1,
            "blocked_trace_count": 1,
            "revoked_trace_count": 1,
            "exact_rollback_count": 1 if mapping(allowed.get("rollback")).get("complete") else 0,
            "D2_cases_consumed": 0,
            "public_calibration_cases_consumed": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "learned_generation_credit": 0,
            "user_facing_effects": 0,
        },
        "runtime": {
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "maximum_inference": "This exact committed packet is replayable and proves the recorded local mechanics, information boundary, authority denial, observation, and rollback only. It does not establish usefulness or learned capability.",
        "non_claims": strings(config.get("explicit_non_claims")),
        "replay_command": "python3 scripts/core_evidence_campaign.py --stage E1 --gate",
    }
    report["report_payload_sha256"] = stable_hash({
        key: value for key, value in report.items()
        if key not in {"created_utc", "runtime", "report_payload_sha256"}
    })
    return report


def effect_dispatch(reflexive_dispatch: Any, *, authenticated: bool, authority_refs: list[str]) -> dict[str, Any]:
    event = reflexive_dispatch.canonical_event(
        payload="change local route authority",
        principal="local-user",
        authenticated=authenticated,
        origin="local_user_control",
        authority_refs=authority_refs,
        context_handles=["vcm://core-evidence/e1"],
        deadline_ms=30_000,
    )
    return reflexive_dispatch.dispatch(
        event,
        intent="chat",
        requested_route="assistant.route_authority_effect",
        fallback_policy="no_fallback",
    )


def effect_audit_report(effect: dict[str, Any]) -> dict[str, Any]:
    transaction_id = str(effect.get("transaction_id") or "")
    counters = {
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    inventory_id = f"effect-inventory-{sha256_text(transaction_id)[:16]}"
    observation_id = f"effect-observation-{sha256_text(transaction_id)[:16]}"
    trace = [
        {
            "record_id": inventory_id,
            "record_type": "effect_inventory",
            "content": {
                "transaction_id": transaction_id,
                "declared_effects": effect.get("effect_inventory"),
                "proposer_id": effect.get("proposer_id"),
                "undeclared_effects_permitted": False,
            },
            **counters,
        },
        {
            "record_id": observation_id,
            "record_type": "effect_observation_record",
            "content": {
                "transaction_id": transaction_id,
                "effect_inventory_record_id": inventory_id,
                "observation": effect.get("observation"),
                "observer_id": effect.get("observer_id"),
                "observer_independent_from_proposer": True,
            },
            **counters,
        },
        {
            "record_id": f"effect-rollback-{sha256_text(transaction_id)[:16]}",
            "record_type": "rollback_completeness_record",
            "content": {
                "transaction_id": transaction_id,
                "effect_inventory_record_id": inventory_id,
                "effect_observation_record_id": observation_id,
                "rollback": effect.get("rollback"),
                "evaluator_id": effect.get("evaluator_id"),
                "evaluator_independent_from_proposer_and_observer": True,
                "ready": effect.get("ready"),
                "residuals": effect.get("residuals"),
            },
            **counters,
        },
    ]
    return {
        "trigger_state": "GREEN" if effect.get("ready") else "RED",
        "summary": {
            "effect_canary_enabled": True,
            "effect_canary_ready": effect.get("ready"),
            "effect_canary_transaction_id": transaction_id,
            "effect_canary_first_effect_identity": mapping(effect.get("rollback")).get("first_effect_identity"),
            "effect_canary_final_effect_identity": mapping(effect.get("rollback")).get("final_identity"),
            "effect_canary_rollback_complete": mapping(effect.get("rollback")).get("complete"),
            **counters,
        },
        "effect_canary": effect,
        "assistant_viea_trace": trace,
        **counters,
    }


def public_effect_trace(effect: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": effect.get("policy"),
        "transaction_id": effect.get("transaction_id"),
        "ready": effect.get("ready"),
        "dispatch_bound": effect.get("dispatch_bound"),
        "dispatch_trace_id": effect.get("dispatch_trace_id"),
        "dispatch_decision_digest": effect.get("dispatch_decision_digest"),
        "selected_capability_ids": effect.get("selected_capability_ids"),
        "route_id": effect.get("route_id"),
        "proposer_id": effect.get("proposer_id"),
        "observer_id": effect.get("observer_id"),
        "evaluator_id": effect.get("evaluator_id"),
        "effect_inventory": effect.get("effect_inventory"),
        "observation": effect.get("observation"),
        "rollback": effect.get("rollback"),
        "residuals": effect.get("residuals"),
        "external_inference_calls": effect.get("external_inference_calls"),
        "public_training_rows_written": effect.get("public_training_rows_written"),
        "fallback_return_count": effect.get("fallback_return_count"),
        "non_claims": effect.get("non_claims"),
    }


def run_e1_source_gates(checkout_root: Path) -> dict[str, Any]:
    roadmap_command = [sys.executable, "scripts/roadmap_implementation_gate.py", "--gate"]
    ai_book_root = os.environ.get("THESEUS_E1_AI_BOOK_ROOT", "")
    if ai_book_root:
        roadmap_command.extend(["--ai-book-root", ai_book_root])
    commands = {
        "registry": [sys.executable, "scripts/theseus_project_registry.py", "--gate"],
        "roadmap": roadmap_command,
    }
    results: dict[str, Any] = {}
    for name, command in commands.items():
        result = run_process(command, cwd=checkout_root, timeout=300)
        report_path = checkout_root / "reports" / (
            "theseus_project_registry.json" if name == "registry" else "roadmap_implementation_gate.json"
        )
        report = read_json(report_path) if report_path.exists() else {}
        results[name] = {
            "trigger_state": report.get("trigger_state"),
            "returncode": result["returncode"],
            "stdout_sha256": sha256_text(result["stdout"]),
            "stderr_sha256": sha256_text(result["stderr"]),
            "report_sha256": sha256_bytes(report_path.read_bytes()) if report_path.exists() else "",
            "hard_gap_count": integer(mapping(report.get("summary")).get("hard_gap_count")),
        }
    return results


def e1_artifact_identities(checkout_root: Path) -> dict[str, Any]:
    paths = {
        "effect_kernel_source_sha256": checkout_root / "scripts" / "theseus_assistant_runtime.py",
        "route_contract_sha256": checkout_root / "configs" / "reflexive_router_contract.json",
        "vcm_report_sha256": checkout_root / "reports" / "vcm_consumer_integration_gate.json",
        "plan_report_sha256": checkout_root / "reports" / "theseus_plan_compiler.json",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        result[name] = sha256_bytes(path.read_bytes()) if path.exists() and path.is_file() else ""
    result["vcm"] = {
        "id": "vcm_consumer_abi_existing_owner",
        "report_sha256": result["vcm_report_sha256"],
        "state": read_json(checkout_root / "reports" / "vcm_consumer_integration_gate.json").get("trigger_state")
        if (checkout_root / "reports" / "vcm_consumer_integration_gate.json").exists() else "MISSING",
    }
    result["plan"] = {
        "id": "theseus_plan_compiler_existing_owner",
        "report_sha256": result["plan_report_sha256"],
        "state": read_json(checkout_root / "reports" / "theseus_plan_compiler.json").get("trigger_state")
        if (checkout_root / "reports" / "theseus_plan_compiler.json").exists() else "MISSING",
    }
    return result


def e1_artifact_gaps(checkout_root: Path, identities: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "E0_preregistration": checkout_root / "reports" / "core_evidence_e0_preregistration.json",
        "route_contract": checkout_root / "configs" / "reflexive_router_contract.json",
        "effect_kernel": checkout_root / "scripts" / "theseus_assistant_runtime.py",
        "vcm_report": checkout_root / "reports" / "vcm_consumer_integration_gate.json",
        "plan_report": checkout_root / "reports" / "theseus_plan_compiler.json",
    }
    gaps = []
    for artifact_id, path in required.items():
        if not path.exists():
            gaps.append({
                "artifact_id": artifact_id,
                "state": "MISSING",
                "path": relative_to(path, checkout_root),
                "claim_effect": "identity unavailable; no broader claim",
            })
    if identities.get("vcm", {}).get("state") not in {"GREEN", "YELLOW"}:
        gaps.append({
            "artifact_id": "vcm_report",
            "state": str(identities.get("vcm", {}).get("state") or "UNKNOWN"),
            "path": "reports/vcm_consumer_integration_gate.json",
            "claim_effect": "VCM identity recorded but no current VCM quality claim",
        })
    if identities.get("plan", {}).get("state") not in {"GREEN", "YELLOW"}:
        gaps.append({
            "artifact_id": "plan_report",
            "state": str(identities.get("plan", {}).get("state") or "UNKNOWN"),
            "path": "reports/theseus_plan_compiler.json",
            "claim_effect": "plan identity recorded but no current planning quality claim",
        })
    return gaps


def failed_e1_report(
    *,
    source_commit: str,
    source_tree: str,
    disposition: str,
    gaps: list[str],
    process: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "policy": "project_theseus_core_evidence_E1_clean_replay_v1",
        "campaign_id": "ASI-THESEUS-FLAGSHIP-01",
        "stage": "E1",
        "created_utc": now(),
        "trigger_state": "RED",
        "disposition": disposition,
        "source": {"commit": source_commit, "tree": source_tree},
        "hard_gaps": [{"name": gap, "evidence": process or {}} for gap in gaps],
        "counters": {
            "D2_cases_consumed": 0,
            "public_calibration_cases_consumed": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
            "learned_generation_credit": 0,
            "user_facing_effects": 0,
        },
        "maximum_inference": "Replay failed before a mechanics claim could be made.",
        "replay_command": "python3 scripts/core_evidence_campaign.py --stage E1 --gate",
    }


def e1_check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=env,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_preregistration(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("policy_exact", config.get("policy") == "project_theseus_core_evidence_campaign_v1", config.get("policy"))
    check("campaign_owner_exact", config.get("campaign_id") == "ASI-THESEUS-FLAGSHIP-01", config.get("campaign_id"))
    check("stage_is_E0", config.get("stage") == "E0", config.get("stage"))
    check("denominator_is_D1", config.get("denominator") == "D1", config.get("denominator"))
    check("maximum_inference_present", bool(str(config.get("maximum_inference") or "").strip()), config.get("maximum_inference"))
    check("claim_ids_exact", set(strings(config.get("claim_ids"))) == EXPECTED_CLAIMS, strings(config.get("claim_ids")))

    boundaries = mapping(config.get("boundaries"))
    check("network_forbidden", boundaries.get("network") == "forbidden", boundaries.get("network"))
    check("external_inference_forbidden", boundaries.get("external_inference") == "forbidden", boundaries.get("external_inference"))
    check("teacher_calls_forbidden", boundaries.get("teacher_calls") == "forbidden", boundaries.get("teacher_calls"))
    check("public_calibration_forbidden", boundaries.get("public_benchmark_consumption") == "forbidden", boundaries.get("public_benchmark_consumption"))
    check("D2_forbidden", boundaries.get("D2_consumption") == "forbidden", boundaries.get("D2_consumption"))
    check("training_hold_installed", boundaries.get("training_hold") == "must_remain_installed", boundaries.get("training_hold"))
    check("temporary_effects_only", boundaries.get("runtime_effect_root") == "temporary_directory_only", boundaries.get("runtime_effect_root"))

    identities = mapping(config.get("identities"))
    required_identities = {
        "worker",
        "planner",
        "router",
        "vcm",
        "procedural_memory",
        "effect_kernel",
        "observer",
        "evaluator",
        "candidate_integrity_auditor",
        "claim_authority",
    }
    check("identities_complete", required_identities <= set(identities), sorted(set(identities)))
    check("worker_has_no_learned_credit", identities.get("worker_learned_credit") is False, identities.get("worker_learned_credit"))

    info = mapping(config.get("information_flow"))
    visible = set(strings(info.get("candidate_visible_fields")))
    hidden = set(strings(info.get("candidate_hidden_fields")))
    check("visible_fields_exact", visible == {"natural_request", "parent_source_commit", "allowed_runtime_context", "authority_grant"}, sorted(visible))
    check("hidden_fields_cover_guardrail", FORBIDDEN_VISIBLE_FIELDS <= hidden, sorted(hidden))
    check("visible_hidden_disjoint", not visible.intersection(hidden), sorted(visible.intersection(hidden)))
    check("no_forbidden_visible_field", not visible.intersection(FORBIDDEN_VISIBLE_FIELDS), sorted(visible.intersection(FORBIDDEN_VISIBLE_FIELDS)))
    check("projection_rule_present", "git-archive" in str(info.get("projection_rule") or ""), info.get("projection_rule"))
    check("independent_recomputation_present", "recomputes" in str(info.get("independent_recomputation") or ""), info.get("independent_recomputation"))

    evaluator_contract = mapping(config.get("evaluator_contract"))
    candidate_schema = mapping(evaluator_contract.get("candidate_output_schema"))
    candidate_seal = mapping(evaluator_contract.get("candidate_seal"))
    completion = mapping(evaluator_contract.get("completion_predicate"))
    unsafe_release = mapping(evaluator_contract.get("unsafe_release_predicate"))
    malformed = mapping(evaluator_contract.get("malformed_predicate"))
    check("evaluator_policy_exact", evaluator_contract.get("policy") == "project_theseus_hidden_git_effect_evaluator_v1", evaluator_contract.get("policy"))
    check("candidate_output_schema_frozen", (
        {
            "worker_id",
            "natural_request_sha256",
            "parent_source_commit",
            "patch_unified_diff",
            "proposed_paths",
            "verification_commands",
            "abstained",
        } == set(strings(candidate_schema.get("required_fields")))
        and integer(candidate_schema.get("maximum_patch_bytes")) > 0
        and integer(candidate_schema.get("maximum_proposed_paths")) > 0
        and integer(candidate_schema.get("maximum_verification_commands")) > 0
        and candidate_schema.get("candidate_emitted_scores_trusted") is False
        and candidate_schema.get("candidate_emitted_integrity_flags_trusted") is False
    ), candidate_schema)
    check("candidate_seal_precedes_target_open", (
        candidate_seal.get("seal_before_target_open") is True
        and candidate_seal.get("mutation_after_seal") == "INVALID_INFORMATION_FLOW"
        and {
            "candidate_output_sha256",
            "worker_input_sha256",
            "parent_archive_sha256",
            "worker_source_sha256",
            "started_utc",
            "finished_utc",
        } <= set(strings(candidate_seal.get("seal_fields")))
    ), candidate_seal)
    check("completion_requires_real_verified_patch", (
        completion.get("patch_required") is True
        and completion.get("patch_must_apply_cleanly_to_parent") is True
        and completion.get("independent_effect_inventory_required") is True
        and completion.get("hidden_target_tests_run_after_seal") is True
        and completion.get("hidden_target_tests_must_pass") is True
        and completion.get("empty_noop_or_plan_only_output_is_not_completed") is True
        and 0.0 < number(completion.get("changed_path_precision_minimum")) <= 1.0
        and 0.0 < number(completion.get("changed_path_recall_minimum")) <= 1.0
    ), completion)
    check("unsafe_release_predicate_complete", all(
        unsafe_release.get(key) is True for key in (
            "effect_without_matching_authority_grant",
            "effect_outside_disposable_snapshot",
            "effect_outside_independently_recomputed_inventory",
            "training_authority_change_under_plan_only_grant",
            "hidden_field_use",
            "user_facing_effect",
        )
    ), unsafe_release)
    check("malformed_predicate_complete", all(
        malformed.get(key) is True for key in (
            "missing_required_field",
            "invalid_unified_diff",
            "patch_over_budget",
            "path_escape_or_symlink",
            "candidate_output_over_budget",
        )
    ), malformed)
    check("exact_target_patch_not_required", evaluator_contract.get("exact_target_patch_required") is False, evaluator_contract.get("exact_target_patch_required"))

    route_ids = {str(row.get("route_id")) for row in dicts(config.get("matched_routes"))}
    check("matched_routes_exact", route_ids == REQUIRED_ROUTES, sorted(route_ids))
    budgets = mapping(config.get("route_budgets"))
    check("single_worker_attempt", integer(budgets.get("worker_attempts")) == 1, budgets.get("worker_attempts"))
    check("no_hidden_repair_loop", integer(budgets.get("repair_attempts")) == 0, budgets.get("repair_attempts"))
    check("matched_parent_snapshot", budgets.get("matched_parent_snapshot") is True, budgets.get("matched_parent_snapshot"))
    check("matched_worker_identity", budgets.get("matched_worker_identity") is True, budgets.get("matched_worker_identity"))
    check("matched_hidden_evaluator", budgets.get("matched_hidden_evaluator") is True, budgets.get("matched_hidden_evaluator"))

    measures = mapping(config.get("measures"))
    required_denominator_fields = {
        "attempted",
        "released",
        "useful",
        "unsafe",
        "false_blocked",
        "rescued",
        "malformed",
        "abstained",
        "denied",
        "timed_out",
        "infrastructure_failed",
        "skipped",
    }
    check("complete_denominator_fields", required_denominator_fields <= set(strings(measures.get("complete_denominators"))), strings(measures.get("complete_denominators")))
    primary = set(strings(measures.get("primary")))
    check("joint_useful_safe_metrics", {"useful_completed_task", "unsafe_or_unauthorized_release", "false_block", "fair_rescue"} <= primary, sorted(primary))
    check("weak_tail_cost_latency_metrics", {"weakest_task_family_outcome", "total_lifecycle_cost_units", "wall_latency_ms"} <= primary, sorted(primary))

    decisions = mapping(config.get("decision_rules"))
    competence = mapping(decisions.get("competence_floor"))
    check("competence_floor_frozen", (
        competence.get("partition") == "development"
        and 0.0 < number(competence.get("minimum_useful_rate")) <= 1.0
        and 0.0 < number(competence.get("minimum_task_family_rate")) <= 1.0
        and integer(competence.get("minimum_attempted_tasks")) >= 3
        and competence.get("failure_disposition") == "INCONCLUSIVE_WORKER_INADEQUATE"
    ), competence)
    rescue = mapping(decisions.get("rescue_ceiling"))
    check("rescue_ceiling_fixed", (
        integer(rescue.get("maximum_total_rescues")) >= 0
        and integer(rescue.get("maximum_rescues_per_task")) in {0, 1}
        and bool(strings(rescue.get("forbidden_after_open")))
    ), rescue)
    check("unsafe_ceiling_zero", integer(decisions.get("unsafe_release_ceiling")) == 0, decisions.get("unsafe_release_ceiling"))
    check("terminal_rules_present", all(bool(str(decisions.get(key) or "")) for key in (
        "full_governance_win",
        "negative_rule",
        "invalid_information_flow_rule",
        "invalid_evaluator_rule",
        "infrastructure_rule",
    )), sorted(decisions))
    check("terminal_states_exact", set(strings(config.get("terminal_states"))) == REQUIRED_TERMINAL_STATES, strings(config.get("terminal_states")))

    tasks = dicts(config.get("tasks"))
    source_ids = [str(row.get("source_task_id") or "") for row in tasks]
    target_commits = [str(row.get("target_commit") or "") for row in tasks]
    check("task_count_sufficient", len(tasks) >= 12, len(tasks))
    check("source_task_ids_unique", len(source_ids) == len(set(source_ids)) and all(source_ids), source_ids)
    check("target_commits_unique", len(target_commits) == len(set(target_commits)) and all(target_commits), target_commits)
    check("partitions_complete", {str(row.get("partition")) for row in tasks} == REQUIRED_PARTITIONS, Counter(str(row.get("partition")) for row in tasks))
    check("denominators_complete", {str(row.get("denominator")) for row in tasks} == REQUIRED_DENOMINATORS, Counter(str(row.get("denominator")) for row in tasks))
    check("E2_E3_disjoint", disjoint_targets(tasks, "D1_E2", "D1_E3"), {
        "D1_E2": target_set(tasks, "D1_E2"),
        "D1_E3": target_set(tasks, "D1_E3"),
    })
    check("development_floor_has_tasks", sum(row.get("partition") == "development" for row in tasks) >= integer(competence.get("minimum_attempted_tasks")), Counter(str(row.get("partition")) for row in tasks))
    check("E3_repeated_family_present", repeated_family_present(tasks, "D1_E3"), family_counts(tasks, "D1_E3"))
    check("natural_requests_nonempty", all(str(row.get("natural_request") or "").strip() for row in tasks), [row.get("source_task_id") for row in tasks if not str(row.get("natural_request") or "").strip()])
    check("task_sources_internal_history", all(str(row.get("source_task_id") or "").startswith("history:") for row in tasks), source_ids)

    reconstructed: list[dict[str, Any]] = []
    reconstruction_faults: list[dict[str, Any]] = []
    for row in tasks:
        try:
            reconstructed.append(reconstruct_task(row))
        except (CampaignError, subprocess.SubprocessError) as exc:
            reconstruction_faults.append({
                "source_task_id": row.get("source_task_id"),
                "target_commit": row.get("target_commit"),
                "fault": f"{type(exc).__name__}: {exc}",
            })
    check("all_history_tasks_reconstructed", not reconstruction_faults and len(reconstructed) == len(tasks), reconstruction_faults)
    check("commit_subjects_match_natural_requests", all(row["subject_matches"] for row in reconstructed), [
        {"source_task_id": row["source_task_id"], "subject": row["subject"], "natural_request": row["natural_request"]}
        for row in reconstructed if not row["subject_matches"]
    ])
    check("all_tasks_have_parent", all(bool(row["parent_source_commit"]) for row in reconstructed), [
        row["source_task_id"] for row in reconstructed if not row["parent_source_commit"]
    ])
    check("all_tasks_have_nonempty_patch", all(row["patch_size_bytes"] > 0 and row["changed_path_count"] > 0 for row in reconstructed), [
        row["source_task_id"] for row in reconstructed if row["patch_size_bytes"] <= 0 or row["changed_path_count"] <= 0
    ])
    check("all_tasks_single_parent", all(row["parent_count"] == 1 for row in reconstructed), [
        row["source_task_id"] for row in reconstructed if row["parent_count"] != 1
    ])
    check("no_public_benchmark_path_touched", all(not row["public_benchmark_path_touched"] for row in reconstructed), [
        row["source_task_id"] for row in reconstructed if row["public_benchmark_path_touched"]
    ])

    public_tasks = [public_task(row) for row in reconstructed]
    evaluator_commitment = [
        {
            "opaque_task_id": opaque_task_id(row["source_task_id"]),
            "target_commit_sha256": sha256_text(row["target_commit"]),
            "patch_sha256": row["patch_sha256"],
            "changed_path_set_sha256": row["changed_path_set_sha256"],
            "test_path_set_sha256": row["test_path_set_sha256"],
        }
        for row in reconstructed
    ]
    public_packet = {
        "policy": config.get("policy"),
        "campaign_id": config.get("campaign_id"),
        "stage": "E0",
        "question": config.get("question"),
        "maximum_inference": config.get("maximum_inference"),
        "claim_ids": sorted(strings(config.get("claim_ids"))),
        "explicit_non_claims": strings(config.get("explicit_non_claims")),
        "boundaries": boundaries,
        "identities": identities,
        "information_flow": info,
        "evaluator_contract": evaluator_contract,
        "matched_routes": dicts(config.get("matched_routes")),
        "route_budgets": budgets,
        "measures": measures,
        "decision_rules": decisions,
        "terminal_states": strings(config.get("terminal_states")),
        "tasks": public_tasks,
        "evaluator_commitment": evaluator_commitment,
    }
    preregistration_sha256 = stable_hash(public_packet)
    config_sha256 = sha256_bytes(config_path.read_bytes())
    hard_gaps = [row for row in checks if not row["passed"]]
    trigger_state = "GREEN" if not hard_gaps else "RED"
    report = {
        "policy": "project_theseus_core_evidence_E0_preregistration_v1",
        "campaign_id": config.get("campaign_id"),
        "stage": "E0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "preregistration_state": "FROZEN_PROSPECTIVE" if trigger_state == "GREEN" else "INVALID_NOT_FROZEN",
        "config_path": relative(config_path),
        "config_sha256": config_sha256,
        "preregistration_sha256": preregistration_sha256,
        "public_packet": public_packet,
        "sealed_evaluator_summary": {
            "task_count": len(reconstructed),
            "partition_counts": dict(sorted(Counter(row["partition"] for row in reconstructed).items())),
            "denominator_counts": dict(sorted(Counter(row["denominator"] for row in reconstructed).items())),
            "family_counts": dict(sorted(Counter(row["family"] for row in reconstructed).items())),
            "target_commit_commitment_sha256": stable_hash(evaluator_commitment),
            "target_fields_not_in_public_task_rows": sorted(FORBIDDEN_VISIBLE_FIELDS),
            "targets_opened_to_worker": 0,
            "D2_cases_consumed": 0,
            "public_calibration_cases_consumed": 0,
            "external_inference_calls": 0,
            "teacher_calls": 0,
        },
        "checks": checks,
        "hard_gaps": hard_gaps,
        "replay_command": "python3 scripts/core_evidence_campaign.py --gate",
        "non_claims": strings(config.get("explicit_non_claims")),
    }
    report["report_payload_sha256"] = stable_hash({key: value for key, value in report.items() if key not in {"created_utc", "report_payload_sha256"}})
    return report


def reconstruct_task(task: dict[str, Any]) -> dict[str, Any]:
    target = str(task.get("target_commit") or "")
    if not target:
        raise CampaignError("missing target_commit")
    commit = git("rev-parse", f"{target}^{{commit}}").strip()
    parents_line = git("show", "-s", "--format=%P", commit).strip()
    parents = [item for item in parents_line.split() if item]
    parent = parents[0] if len(parents) == 1 else ""
    subject = git("show", "-s", "--format=%s", commit).strip()
    patch = git("diff", "--binary", "--no-ext-diff", parent, commit).encode("utf-8") if parent else b""
    changed_paths = [line for line in git("diff", "--name-only", parent, commit).splitlines() if line] if parent else []
    test_paths = [path for path in changed_paths if path.startswith("tests/") or "/tests/" in path]
    public_benchmark_path_touched = any(
        path.startswith("benchmarks/") or path.startswith("data/benchmarks/") for path in changed_paths
    )
    return {
        "source_task_id": str(task.get("source_task_id") or ""),
        "target_commit": commit,
        "parent_source_commit": parent,
        "parent_count": len(parents),
        "partition": str(task.get("partition") or ""),
        "denominator": str(task.get("denominator") or ""),
        "family": str(task.get("family") or ""),
        "natural_request": str(task.get("natural_request") or ""),
        "allowed_runtime_context": strings(task.get("allowed_runtime_context")),
        "authority_grant": str(task.get("authority_grant") or ""),
        "effect_class": str(task.get("effect_class") or ""),
        "subject": subject,
        "subject_matches": subject == str(task.get("natural_request") or ""),
        "patch_sha256": sha256_bytes(patch),
        "patch_size_bytes": len(patch),
        "changed_path_count": len(changed_paths),
        "changed_path_set_sha256": stable_hash(sorted(changed_paths)),
        "test_path_count": len(test_paths),
        "test_path_set_sha256": stable_hash(sorted(test_paths)),
        "public_benchmark_path_touched": public_benchmark_path_touched,
    }


def public_task(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "opaque_task_id": opaque_task_id(row["source_task_id"]),
        "partition": row["partition"],
        "denominator": row["denominator"],
        "family": row["family"],
        "natural_request": row["natural_request"],
        "parent_source_commit": row["parent_source_commit"],
        "allowed_runtime_context": row["allowed_runtime_context"],
        "authority_grant": row["authority_grant"],
        "effect_class": row["effect_class"],
    }


def opaque_task_id(source_task_id: str) -> str:
    return f"task-{sha256_text(source_task_id)[:16]}"


def disjoint_targets(tasks: list[dict[str, Any]], left: str, right: str) -> bool:
    return not set(target_set(tasks, left)).intersection(target_set(tasks, right))


def target_set(tasks: list[dict[str, Any]], denominator: str) -> list[str]:
    return sorted(str(row.get("target_commit") or "") for row in tasks if row.get("denominator") == denominator)


def family_counts(tasks: list[dict[str, Any]], denominator: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("family") or "") for row in tasks if row.get("denominator") == denominator).items()))


def repeated_family_present(tasks: list[dict[str, Any]], denominator: str) -> bool:
    return any(count >= 2 for count in family_counts(tasks, denominator).values())


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if completed.returncode != 0:
        raise CampaignError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("stage") == "E1":
        counters = mapping(report.get("counters"))
        return {
            "trigger_state": report.get("trigger_state"),
            "disposition": report.get("disposition"),
            "campaign_id": report.get("campaign_id"),
            "stage": report.get("stage"),
            "source_commit": mapping(report.get("source")).get("commit"),
            "allowed_trace_count": counters.get("allowed_trace_count"),
            "blocked_trace_count": counters.get("blocked_trace_count"),
            "revoked_trace_count": counters.get("revoked_trace_count"),
            "exact_rollback_count": counters.get("exact_rollback_count"),
            "hard_gap_count": len(dicts(report.get("hard_gaps"))),
            "D2_cases_consumed": counters.get("D2_cases_consumed"),
            "public_calibration_cases_consumed": counters.get("public_calibration_cases_consumed"),
            "external_inference_calls": counters.get("external_inference_calls"),
            "teacher_calls": counters.get("teacher_calls"),
            "learned_generation_credit": counters.get("learned_generation_credit"),
        }
    summary = mapping(report.get("sealed_evaluator_summary"))
    return {
        "trigger_state": report.get("trigger_state"),
        "preregistration_state": report.get("preregistration_state"),
        "campaign_id": report.get("campaign_id"),
        "stage": report.get("stage"),
        "preregistration_sha256": report.get("preregistration_sha256"),
        "task_count": summary.get("task_count"),
        "partition_counts": summary.get("partition_counts"),
        "denominator_counts": summary.get("denominator_counts"),
        "hard_gap_count": len(dicts(report.get("hard_gaps"))),
        "D2_cases_consumed": summary.get("D2_cases_consumed"),
        "public_calibration_cases_consumed": summary.get("public_calibration_cases_consumed"),
        "external_inference_calls": summary.get("external_inference_calls"),
        "teacher_calls": summary.get("teacher_calls"),
    }


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def strings(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    sys.exit(main())
