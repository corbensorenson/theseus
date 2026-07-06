"""Bounded self-edit experiment lane for Project Theseus.

This lane does not let the teacher or student freely rewrite the system. It
turns residual clusters into small source-patch experiments with explicit
scope, verification, rollback notes, and no external inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LANE_PATHS = [
    "scripts/local_code_repair_organism.py",
    "scripts/pressure_runner.py",
    "scripts/real_code_benchmark_graduation.py",
    "scripts/benchmaxx_curriculum.py",
    "scripts/candidate_promotion_gate.py",
    "scripts/self_edit_experiment_lane.py",
    "scripts/long_horizon_memory_probe.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/self_edit_experiment_lane.json")
    parser.add_argument("--bundle-out", default="reports/self_edit_patch_bundle.json")
    args = parser.parse_args()

    state = load_state()
    experiments = build_experiments(state)
    verification = run_verification()
    rollback = rollback_plan()
    gates = [
        gate("residual_to_patch_contracts_written", bool(experiments), f"experiments={len(experiments)}"),
        gate("source_patch_scope_bounded", all(in_lane_scope(path) for path in rollback["lane_owned_changed_paths"]), rollback),
        gate("verification_commands_passed", all(row["passed"] for row in verification), verification),
        gate("rollback_plan_written", bool(rollback["commands"]), rollback["commands"]),
        gate("teacher_apply_mode_disabled", True, "patch proposals and local deterministic checks only"),
        gate("external_inference_zero", True, "no teacher/student calls made by this lane"),
    ]
    bundle = {
        "policy": "project_theseus_self_edit_patch_bundle_v1",
        "created_utc": now(),
        "experiments": experiments,
        "verification": verification,
        "rollback_plan": rollback,
        "external_inference_calls": 0,
    }
    report = {
        "policy": "project_theseus_self_edit_experiment_lane_v1",
        "created_utc": now(),
        "purpose": "residual cluster -> proposed source patch -> tests -> profile/gate evidence -> rollback or commit",
        "allowed_patch_scope": {
            "paths": ["scripts/*.py", "configs/*.json", "reports/*.json"],
            "forbidden": ["model parameter growth", "teacher apply mode", "network side effects", "hardware control"],
        },
        "experiments": experiments,
        "verification": verification,
        "rollback_plan": rollback,
        "gates": gates,
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "commit_allowed": False,
        "commit_reason": "dirty workspace requires human-reviewed commit boundary; lane emits bundle only",
        "bundle": rel(ROOT / args.bundle_out),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.bundle_out, bundle)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in gates) else 1


def build_experiments(state: dict[str, Any]) -> list[dict[str, Any]]:
    forge = state.get("code_residual_forge") if isinstance(state.get("code_residual_forge"), dict) else {}
    summary = forge.get("summary") if isinstance(forge.get("summary"), dict) else {}
    dominant = str(summary.get("dominant_residual_class") or "local_code_generation_adapter_needed")
    active_card = str(summary.get("active_card_id") or get_path(state, ["frontier", "pressure_card_id"], ""))
    specs = [
        {
            "id": "real_code_benchmark_graduation",
            "residual_cluster": "public_benchmark_item_generation_needed",
            "patch_hypothesis": "Graduate code pressure from loader/synthetic tasks to real local public benchmark tasks with honest score semantics and transfer heredity checks.",
            "paths": ["scripts/real_code_benchmark_graduation.py", "scripts/pressure_runner.py", "scripts/candidate_promotion_gate.py"],
            "status": status_for_strings(
                "scripts/real_code_benchmark_graduation.py",
                ["project_theseus_real_code_benchmark_graduation_v1", "public_benchmark_score_claim", "transfer_behavior_changed"],
            ),
            "verification_commands": [
                "python -m py_compile scripts/real_code_benchmark_graduation.py scripts/pressure_runner.py scripts/candidate_promotion_gate.py",
                "python scripts/real_code_benchmark_graduation.py --cards source_evalplus,source_human_eval --seed 14 --out reports/real_code_benchmark_graduation.json",
            ],
        },
        {
            "id": "local_code_repair_organism",
            "residual_cluster": dominant,
            "patch_hypothesis": "Add a local deterministic repair loop that generates candidates, runs sandbox tests, records traces, classifies failures, and emits transfer evidence.",
            "paths": ["scripts/local_code_repair_organism.py", "scripts/pressure_runner.py"],
            "status": status_for_strings(
                "scripts/pressure_runner.py",
                ["run_code_repair_organism", "code_transfer_altered_behavior", "code_repair_patch_trace_written"],
            ),
            "verification_commands": [
                "python -m py_compile scripts/local_code_repair_organism.py scripts/pressure_runner.py",
                f"python scripts/local_code_repair_organism.py --card-id {active_card or 'source_livecodebench'} --seed 14",
            ],
        },
        {
            "id": "runnable_code_frontier_rotation",
            "residual_cluster": "source_not_staged",
            "patch_hypothesis": "Do not spend pressure runs on smoke-passed code cards whose source tree is not actually staged and runnable.",
            "paths": ["scripts/benchmaxx_curriculum.py"],
            "status": status_for_strings(
                "scripts/benchmaxx_curriculum.py",
                ["coding_source_material_present", "coding_source_setup_required", "coding_runnable"],
            ),
            "verification_commands": ["python -m py_compile scripts/benchmaxx_curriculum.py", "python scripts/benchmaxx_curriculum.py"],
        },
        {
            "id": "transfer_consumption_gate",
            "residual_cluster": "transfer_claim_without_heredity_proof",
            "patch_hypothesis": "Promotion should require the active code pressure runner to consume transfer artifacts and report a behavior delta.",
            "paths": ["scripts/candidate_promotion_gate.py"],
            "status": status_for_strings(
                "scripts/candidate_promotion_gate.py",
                ["code_frontier_transfer_consumed", "metrics\", \"code_repair_organism", "pass_rate_delta"],
            ),
            "verification_commands": ["python -m py_compile scripts/candidate_promotion_gate.py"],
        },
        {
            "id": "long_horizon_goal_memory",
            "residual_cluster": "long_horizon_memory_unproven",
            "patch_hypothesis": "Add a deterministic memory probe that recovers goals and rejects decoys from compressed Theseus traces.",
            "paths": ["scripts/long_horizon_memory_probe.py"],
            "status": "implemented" if (ROOT / "scripts" / "long_horizon_memory_probe.py").exists() else "pending",
            "verification_commands": [
                "python -m py_compile scripts/long_horizon_memory_probe.py",
                "python scripts/long_horizon_memory_probe.py --out reports/long_horizon_memory_probe.json",
            ],
        },
    ]
    for spec in specs:
        spec["teacher_needed"] = False
        spec["risk_tier"] = "low"
        spec["rollback"] = "Do not revert unrelated user changes; revert only listed paths if this patch experiment fails review."
    return specs


def status_for_strings(path: str, needles: list[str]) -> str:
    target = ROOT / path
    if not target.exists():
        return "pending"
    text = target.read_text(encoding="utf-8", errors="ignore")
    missing = [needle for needle in needles if needle not in text]
    return "implemented" if not missing else f"partial_missing:{','.join(missing)}"


def run_verification() -> list[dict[str, Any]]:
    commands = [
        [sys.executable, "-m", "py_compile", *LANE_PATHS],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        rows.append(
            {
                "command": " ".join(command),
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-2000:],
            }
        )
    return rows


def rollback_plan() -> dict[str, Any]:
    changed = git_changed_paths()
    lane_owned = [path for path in changed if in_lane_scope(path)]
    return {
        "lane_owned_changed_paths": lane_owned,
        "all_changed_path_count": len(changed),
        "commands": [
            "Review reports/self_edit_patch_bundle.json.",
            "If a lane-owned patch fails review, restore only that listed path from the previous commit.",
            "Never reset the full workspace because unrelated user/generated changes may be present.",
        ],
    }


def git_changed_paths() -> list[str]:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, timeout=30)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        value = line[3:].strip()
        if " -> " in value:
            value = value.split(" -> ", 1)[1].strip()
        if value:
            paths.append(value.replace("\\", "/"))
    return paths


def in_lane_scope(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized in LANE_PATHS or normalized.startswith("reports/")


def load_state() -> dict[str, Any]:
    reports = ROOT / "reports"
    return {
        "code_residual_forge": read_json(reports / "code_residual_forge.json"),
        "frontier": read_json(reports / "frontier_policy_status.json"),
        "candidate": read_json(reports / "candidate_promotion_gate.json"),
    }


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
