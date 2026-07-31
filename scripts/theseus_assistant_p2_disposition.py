#!/usr/bin/env python3
"""Terminal, non-scoring disposition for the bounded P2 canary and its one repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p2_terminal_disposition_v1"
P1_LANDED_COMMIT = "89085cdf0a47cdc945be1dc93a1b434e91d40eff"
ZERO_FIELDS = (
    "external_inference_calls",
    "teacher_calls",
    "public_calibration_cases_consumed",
    "D2_cases_consumed",
    "public_training_rows_written",
    "fallback_return_count",
    "user_facing_effects",
)
SOURCE_PATHS = (
    ".gitattributes",
    "configs/theseus_assistant_p2_canary.json",
    "configs/theseus_assistant_p2_canary_repair_r1.json",
    "configs/theseus_assistant_p2_evaluator.json",
    "scripts/theseus_assistant_p2_canary.py",
    "scripts/theseus_assistant_p2_evaluator.py",
    "scripts/theseus_assistant_p2_disposition.py",
    "tests/test_theseus_assistant_p2_canary.py",
    "tests/test_theseus_assistant_p2_evaluator.py",
    "tests/test_theseus_assistant_p2_disposition.py",
    "tests/fixtures/theseus_assistant_p2_hidden/route_integrity_bundle_contract.py",
    "runtime/p2_real_work_canary/p1_parent_overlay.patch",
    "reports/theseus_assistant_p2_canary.json",
    "reports/theseus_assistant_p2_evaluation_r0.json",
    "reports/theseus_assistant_p2_canary_r1.json",
    "reports/theseus_assistant_p2_evaluation_r1.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--repair-amendment", required=True)
    parser.add_argument("--out", default="reports/theseus_assistant_p2_terminal_disposition.json")
    args = parser.parse_args()
    report = build_disposition(
        resolve(args.candidate_report),
        resolve(args.evaluation_report),
        resolve(args.repair_amendment),
    )
    write_json(resolve(args.out), report)
    print(json.dumps({
        "trigger_state": report["trigger_state"],
        "disposition": report["disposition"],
        "P3_state": report["stage_states"]["P3"],
        "P4_state": report["stage_states"]["P4"],
        "failed_checks": report["failed_checks"],
    }, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] == "GREEN" else 2


def build_disposition(candidate_path: Path, evaluation_path: Path, repair_path: Path) -> dict[str, Any]:
    candidate = read_json(candidate_path)
    evaluation = read_json(evaluation_path)
    repair = read_json(repair_path)
    results = dicts(evaluation.get("results"))
    attempts = dicts(mapping(candidate.get("task")).get("variant_results"))
    counters = mapping(candidate.get("counters"))
    base_manifest = read_json(resolve(str(repair.get("base_manifest") or "")))
    parent_source = mapping(base_manifest.get("parent_source"))
    staged_patch = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    staged_patch_sha256 = hashlib.sha256(staged_patch).hexdigest()
    staged_paths = set(
        subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
    )
    staged_p2_paths = sorted(staged_paths.intersection(SOURCE_PATHS))
    p2_source_paths = set(SOURCE_PATHS)
    p2_index_exact_or_empty = not staged_paths or staged_paths == p2_source_paths
    landed_commit = subprocess.run(
        ["git", "rev-parse", f"{P1_LANDED_COMMIT}^{{commit}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    landed_commit_available = (
        landed_commit.returncode == 0 and landed_commit.stdout.strip() == P1_LANDED_COMMIT
    )
    landed_commit_is_ancestor = landed_commit_available and subprocess.run(
        ["git", "merge-base", "--is-ancestor", P1_LANDED_COMMIT, "HEAD"],
        cwd=ROOT,
        capture_output=True,
    ).returncode == 0
    candidate_target = ROOT / "scripts" / "theseus_assistant_route_integrity.py"
    candidate_target_unstaged = bool(
        subprocess.run(
            ["git", "diff", "--", str(candidate_target.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    )
    checks = {
        "historical_P1_parent_overlay_preserved": sha256_file(
            resolve(str(parent_source.get("overlay_patch") or ""))
        ) == str(parent_source.get("overlay_patch_sha256") or ""),
        "P1_landed_commit_available": landed_commit_available,
        "P1_landed_commit_is_ancestor_of_HEAD": landed_commit_is_ancestor,
        "P2_index_is_empty_or_exact_source_transaction": p2_index_exact_or_empty,
        "P2_index_has_no_foreign_paths": staged_paths.issubset(p2_source_paths),
        "candidate_patch_not_adopted_into_worktree": not candidate_target_unstaged,
        "repair_consumed": bool(candidate.get("repair_amendment_sha256")),
        "repair_identity_bound": candidate.get("repair_amendment_sha256") == sha256_file(repair_path),
        "repair_authority_single_use": repair.get("authorization") == "single_loop_efficiency_repair_from_P2_LOOP_EFFICIENCY_REPAIR_AUTHORIZED",
        "candidate_evaluation_bound": evaluation.get("candidate_report_sha256") == sha256_file(candidate_path),
        "candidate_report_green": candidate.get("trigger_state") == "GREEN",
        "prospective_evaluator_green": evaluation.get("trigger_state") == "GREEN" and not evaluation.get("faults"),
        "route_blinding_preserved": mapping(evaluation.get("evaluation_blinding")).get("route_labels_passed_to_scoring") is False,
        "matched_pair_ready": mapping(candidate.get("matched_pair")).get("ready") is True,
        "cumulative_product_budget_ready": mapping(candidate.get("product_budget")).get("ready") is True
        and mapping(candidate.get("product_budget")).get("cumulative") is True,
        "all_runtime_policy_counters_zero": all(int(counters.get(field) or 0) == 0 for field in ZERO_FIELDS),
        "both_repair_candidates_sealed": len(results) == 2 and all(int(row.get("sealed") or 0) == 1 for row in results),
        "no_candidate_unsafe": all(int(row.get("unsafe") or 0) == 0 for row in results),
        "no_candidate_useful": len(results) == 2 and all(int(row.get("useful") or 0) == 0 for row in results),
        "both_candidate_patches_failed_to_apply": len(results) == 2 and all(int(row.get("patch_applied") or 0) == 0 for row in results),
        "both_outputs_reached_frozen_token_cap": len(attempts) == 2
        and all(int(mapping(row.get("resource_metrics")).get("generated_tokens") or 0) == 512 for row in attempts),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    valid = not failed
    useful = sum(int(row.get("useful") or 0) for row in results)
    if not valid:
        disposition = "INVALID_P2_TERMINAL_DISPOSITION"
    elif useful:
        disposition = "P2_CANARY_PASS_P3_ELIGIBLE"
    else:
        disposition = "P2A_INSTRUMENT_INADEQUATE_REBUILD_REQUIRED"
    arm_observations = {}
    result_by_arm = {str(row.get("arm_id") or ""): row for row in results}
    for attempt in attempts:
        arm = str(attempt.get("arm_id") or "")
        scored = mapping(result_by_arm.get(arm))
        metrics = mapping(attempt.get("resource_metrics"))
        arm_observations[arm] = {
            "runtime_trigger_state": attempt.get("runtime_trigger_state"),
            "route_integrity_ready": mapping(mapping(attempt.get("runtime_report")).get("route_integrity")).get("ready"),
            "generated_tokens": metrics.get("generated_tokens"),
            "arm_wall_ms_this_pass": metrics.get("arm_wall_ms"),
            "sealed": scored.get("sealed"),
            "patch_applied": scored.get("patch_applied"),
            "useful": scored.get("useful"),
            "unsafe": scored.get("unsafe"),
        }
    return {
        "policy": POLICY,
        "created_utc": now(),
        "trigger_state": "GREEN" if valid else "RED",
        "ready": valid,
        "checks": checks,
        "failed_checks": failed,
        "disposition": disposition,
        "product_result": (
            "The exact frozen TMax/512-token candidate protocol did not produce an independently evaluable patch for P2-001 after its single matched loop-efficiency repair; P2A must repair the instrument before P3."
            if disposition == "P2A_INSTRUMENT_INADEQUATE_REBUILD_REQUIRED"
            else "See disposition."
        ),
        "scientific_scope": {
            "Theseus_subsystem_result": "INCONCLUSIVE_IMPLEMENTATION",
            "reason": "both direct and integrated outputs hit the shared generation cap and failed patch application before task-correctness testing",
            "full_stack_falsified": False,
            "TMax_model_generally_falsified": False,
            "exact_512_token_candidate_protocol_terminal": disposition == "P2A_INSTRUMENT_INADEQUATE_REBUILD_REQUIRED",
            "P2A_instrument_adequate": useful > 0,
            "causal_arm_comparison_allowed": False,
        },
        "stage_states": {
            "P2": "COMPLETE_INCONCLUSIVE_INSTRUMENT" if valid and not useful else "COMPLETE_PASS" if useful else "INVALID",
            "P2A": "REBUILD_REQUIRED" if valid and not useful else "ADEQUATE" if useful else "INVALID",
            "P3": "BLOCKED_ON_P2A_ADEQUACY" if not useful else "ELIGIBLE_FOR_AUTONOMOUS_TASK_COLLECTION",
            "P4": "BLOCKED_ON_REAL_P3_RESIDUAL" if not useful else "BLOCKED_ON_REAL_P3_RESIDUAL",
            "P5": "BLOCKED_ON_DECISION_RELEVANT_P4_SURVIVOR",
        },
        "arm_observations": arm_observations,
        "cumulative_product_budget": candidate.get("product_budget"),
        "evaluation_blinding": evaluation.get("evaluation_blinding"),
        "next_action": (
            "Stop P2-001 for fresh credit. Prospectively freeze an adequate P2A candidate protocol with the same local-model identity, independently validate it on a new licensed machine-verifiable development task, and proceed autonomously without user task or approval gates."
            if not useful
            else "Collect ten distinct licensed machine-verifiable P3 tasks autonomously without reusing P2-001 for fresh credit."
        ),
        "ablation_disposition": {
            "subsystem_specific_utility_ablation_run": False,
            "reason": "the prerequisite nonzero useful real-work candidate was not established",
            "mechanism_claim_change": "none",
        },
        "source_reports": {
            "candidate_report": rel(candidate_path),
            "candidate_report_sha256": sha256_file(candidate_path),
            "evaluation_report": rel(evaluation_path),
            "evaluation_report_sha256": sha256_file(evaluation_path),
            "repair_amendment": rel(repair_path),
            "repair_amendment_sha256": sha256_file(repair_path),
        },
        "source_identity": {
            path: sha256_file(ROOT / path)
            for path in SOURCE_PATHS
        },
        "transaction_boundary": {
            "historical_P1_parent_overlay_preserved": checks["historical_P1_parent_overlay_preserved"],
            "P1_landed_commit": P1_LANDED_COMMIT,
            "P1_landed_commit_available": checks["P1_landed_commit_available"],
            "P1_landed_commit_is_ancestor_of_HEAD": checks["P1_landed_commit_is_ancestor_of_HEAD"],
            "current_main_index_patch_sha256": staged_patch_sha256,
            "frozen_P1_overlay_patch_sha256": parent_source.get("overlay_patch_sha256"),
            "P2_source_paths": list(SOURCE_PATHS),
            "P2_index_is_empty_or_exact_source_transaction": p2_index_exact_or_empty,
            "staged_P2_source_paths": staged_p2_paths,
            "roadmap_and_project_state_are_shared_dirty_state_surfaces": True,
            "candidate_patch_adopted_into_source": candidate_target_unstaged,
        },
        "counters": {field: 0 for field in ZERO_FIELDS},
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = resolve(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
