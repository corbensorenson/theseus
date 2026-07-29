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
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "core_evidence_campaign.json"
DEFAULT_OUT = ROOT / "reports" / "core_evidence_e0_preregistration.json"

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
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    config_path = resolve(args.config)
    out_path = resolve(args.out)
    config = read_json(config_path)
    report = build_preregistration(config, config_path)
    write_json(out_path, report)
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    if args.gate and report["trigger_state"] != "GREEN":
        return 2
    return 0


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
