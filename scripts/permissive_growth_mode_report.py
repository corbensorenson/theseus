#!/usr/bin/env python3
"""Summarize the governed permissive growth posture and latest loop evidence.

This report stays small enough to read, but it now includes both control-plane
readiness and the latest executed local growth cycle when available.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/permissive_growth_policy.json")
    parser.add_argument("--training-admission", default="reports/training_data_admission_v1.json")
    parser.add_argument("--teacher-gate", default="reports/teacher_distillation_gate.json")
    parser.add_argument("--teacher-distillation-smoke", default="reports/teacher_distillation_admission_smoke.json")
    parser.add_argument("--public-runner", default="reports/operator_bounded_public_calibration_dry_run.json")
    parser.add_argument("--public-contract", default="configs/public_benchmark_contract_v1.json")
    parser.add_argument("--growth-loop", default="reports/permissive_growth_loop_report.json")
    parser.add_argument("--out", default="reports/permissive_growth_mode_report.json")
    parser.add_argument("--markdown-out", default="reports/permissive_growth_mode_report.md")
    args = parser.parse_args()

    report = build_report(args)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    policy = read_json(resolve(args.policy))
    admission = read_json(resolve(args.training_admission))
    teacher = read_json(resolve(args.teacher_gate))
    teacher_smoke = read_json(resolve(args.teacher_distillation_smoke))
    public_runner = read_json(resolve(args.public_runner))
    public_contract = read_json(resolve(args.public_contract))
    growth_loop = read_json(resolve(args.growth_loop))

    admission_summary = as_dict(admission.get("summary"))
    teacher_summary = as_dict(teacher.get("summary"))
    teacher_smoke_summary = as_dict(teacher_smoke.get("summary"))
    public_summary = as_dict(public_runner.get("summary"))
    public_policy = as_dict(policy.get("public_benchmarks"))
    teacher_policy = as_dict(policy.get("teacher"))
    firewall = as_dict(policy.get("heldout_firewall"))
    loop_summary = as_dict(growth_loop.get("latest_summary"))
    loop_no_cheat = as_dict(loop_summary.get("no_cheat"))
    runtime_external_inference_calls = (
        int(admission_summary.get("external_inference_calls") or 0)
        + int(public_runner.get("external_inference_calls") or 0)
        + int(loop_no_cheat.get("runtime_external_inference_calls") or 0)
    )
    teacher_training_external_inference_calls = (
        max(
            int(teacher.get("external_inference_calls") or 0),
            int(loop_no_cheat.get("teacher_training_external_inference_calls") or 0),
        )
        + int(teacher_smoke.get("external_inference_calls") or 0)
    )

    no_cheat = {
        "runtime_external_inference_forbidden": policy.get("runtime_external_inference") == "forbidden"
        and teacher_summary.get("runtime_external_tokens_forbidden") is True,
        "fallback_returns_forbidden": policy.get("fallback_returns") == "forbidden_for_training_credit_and_public_claims",
        "public_benchmark_payload_admitted": bool(admission_summary.get("public_benchmark_payload_admitted")),
        "public_benchmark_training_allowed": bool(admission_summary.get("public_benchmark_training_allowed")),
        "runtime_external_inference_calls": runtime_external_inference_calls,
        "teacher_training_external_inference_calls": teacher_training_external_inference_calls,
        "external_inference_calls": runtime_external_inference_calls,
        "total_external_inference_calls": runtime_external_inference_calls + teacher_training_external_inference_calls,
        "growth_loop_public_eval_leakage": bool(loop_no_cheat.get("public_eval_leakage_violation")),
        "growth_loop_fallback_violation": bool(loop_no_cheat.get("fallback_return_violation")),
        "growth_loop_runtime_external_violation": bool(loop_no_cheat.get("runtime_external_inference_violation")),
    }

    gates = [
        gate("permissive_growth_policy_loaded", policy.get("policy") == "project_theseus_permissive_growth_policy_v1", args.policy, "hard"),
        gate("public_open_training_policy_enabled", get_path(policy, ["public_open_training_data", "default"]) == "allowed_when_governed", get_path(policy, ["public_open_training_data", "default"]), "hard"),
        gate("public_open_rows_admitted_when_clean", int(admission_summary.get("admitted_open_public_source_count") or 0) > 0, admission_summary, "warning"),
        gate("exact_public_benchmark_payloads_excluded", no_cheat["public_benchmark_payload_admitted"] is False and no_cheat["public_benchmark_training_allowed"] is False, admission_summary, "hard"),
        gate("fallback_returns_not_credited", no_cheat["fallback_returns_forbidden"] is True, policy.get("fallback_returns"), "hard"),
        gate("runtime_external_inference_forbidden", no_cheat["runtime_external_inference_forbidden"] is True, no_cheat, "hard"),
        gate("teacher_proposal_enabled_by_policy", teacher_policy.get("proposal_mode_default") == "enabled_governed", teacher_policy, "hard"),
        gate("teacher_distillation_not_operator_locked", teacher.get("operator_unlock_required") is False and teacher.get("governed_training_enabled") is True, {
            "operator_unlock_required": teacher.get("operator_unlock_required"),
            "governed_training_enabled": teacher.get("governed_training_enabled"),
            "default_state": teacher.get("default_state"),
        }, "hard"),
        gate("teacher_rows_still_require_clean_manifest", bool(teacher_summary.get("manifest_admission_safety_checks_clean")) or int(teacher_summary.get("manifest_row_count") or 0) == 0, teacher_summary, "hard"),
        gate("teacher_distillation_admission_mechanics_proven", not teacher_smoke or (
            teacher_smoke.get("trigger_state") == "GREEN"
            and teacher_smoke_summary.get("real_teacher_manifest_unchanged") is True
            and teacher_smoke_summary.get("real_teacher_ledger_unchanged") is True
        ), teacher_smoke_summary, "warning"),
        gate(
            "public_benchmark_run_registry_enabled",
            public_policy.get("execution_default")
            in {
                "governed_measurement_run_registry",
                "governed_run_registry",
            },
            public_policy,
            "hard",
        ),
        gate("public_runner_uses_run_registry", public_summary.get("authorization_mode") in {"run_registry", None}, public_summary, "warning"),
        gate("no_runtime_external_inference_calls_recorded", no_cheat["runtime_external_inference_calls"] == 0, no_cheat, "hard"),
        gate("growth_loop_executed_when_present", not growth_loop or growth_loop.get("execute") is True, {"present": bool(growth_loop), "execute": growth_loop.get("execute")}, "warning"),
        gate("growth_loop_ok_when_present", not growth_loop or growth_loop.get("ok") is True, {"present": bool(growth_loop), "ok": growth_loop.get("ok"), "hard_stop": growth_loop.get("hard_stop")}, "hard"),
        gate("growth_loop_no_cheat_clean", not any([
            no_cheat["growth_loop_public_eval_leakage"],
            no_cheat["growth_loop_fallback_violation"],
            no_cheat["growth_loop_runtime_external_violation"],
        ]), loop_no_cheat, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    warning_failed = [row for row in gates if row["severity"] != "hard" and not row["passed"]]
    trigger_state = "GREEN" if not hard_failed and not warning_failed else ("YELLOW" if not hard_failed else "RED")

    return {
        "policy": "project_theseus_permissive_growth_mode_report_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "growth_mode": policy.get("mode"),
            "public_open_training_allowed": admission_summary.get("public_open_training_allowed"),
            "admitted_open_public_source_count": admission_summary.get("admitted_open_public_source_count"),
            "admitted_open_public_row_count": admission_summary.get("admitted_open_public_row_count"),
            "public_benchmark_training_allowed": admission_summary.get("public_benchmark_training_allowed"),
            "public_benchmark_payload_admitted": admission_summary.get("public_benchmark_payload_admitted"),
            "teacher_proposal_mode_default": teacher_policy.get("proposal_mode_default"),
            "teacher_distillation_default": teacher_policy.get("distillation_default"),
            "teacher_gate_distillation_allowed": teacher.get("distillation_allowed"),
            "teacher_share_cap": get_path(teacher, ["teacher_share", "max_initial_training_ratio"]),
            "teacher_accepted_row_share": get_path(teacher, ["teacher_share", "teacher_accepted_row_share"]),
            "teacher_distillation_admission_smoke_state": teacher_smoke.get("trigger_state"),
            "teacher_distillation_admission_smoke_gate_state": teacher_smoke_summary.get("gate_trigger_state"),
            "teacher_distillation_admission_mechanics_proven": teacher_smoke.get("trigger_state") == "GREEN",
            "teacher_distillation_smoke_real_manifest_unchanged": teacher_smoke_summary.get("real_teacher_manifest_unchanged"),
            "teacher_distillation_smoke_real_ledger_unchanged": teacher_smoke_summary.get("real_teacher_ledger_unchanged"),
            "public_benchmark_execution_default": get_path(public_contract, ["global_rules", "public_execution_default"]),
            "public_runner_authorization_mode": public_summary.get("authorization_mode"),
            "public_runner_run_registry_allowed": public_summary.get("run_registry_allowed"),
            "public_runner_would_execute": public_summary.get("would_execute"),
            "growth_loop_present": bool(growth_loop),
            "growth_loop_ok": growth_loop.get("ok"),
            "growth_loop_cycles_completed": growth_loop.get("cycles_completed"),
            "growth_loop_broad_train_rows": loop_summary.get("broad_train_rows"),
            "growth_loop_broad_eval_rows": loop_summary.get("broad_eval_rows"),
            "growth_loop_architecture_winner": loop_summary.get("architecture_winner"),
            "growth_loop_transformer_sts_on_pass_rate": loop_summary.get("transformer_sts_on_pass_rate"),
            "growth_loop_symliquid_sts_on_pass_rate": loop_summary.get("symliquid_sts_on_pass_rate"),
            "growth_loop_symliquid_minus_transformer": loop_summary.get("symliquid_minus_transformer"),
            "growth_loop_comparator_state": loop_summary.get("broad_comparator_state"),
            "growth_loop_mlx_state": loop_summary.get("mlx_state"),
            "growth_loop_metal_state": loop_summary.get("metal_state"),
            "growth_loop_resource_execution_owner": loop_summary.get("resource_execution_owner"),
            "teacher_accepted_rows": loop_summary.get("teacher_accepted_rows", teacher_summary.get("teacher_accepted_rows")),
            "teacher_cost_external_calls": no_cheat["teacher_training_external_inference_calls"],
            "heldout_firewall": firewall,
            "no_cheat_counters": no_cheat,
        },
        "gates": gates,
        "inputs": {
            "policy": args.policy,
            "training_admission": args.training_admission,
            "teacher_gate": args.teacher_gate,
            "teacher_distillation_smoke": args.teacher_distillation_smoke,
            "public_runner": args.public_runner,
            "public_contract": args.public_contract,
            "growth_loop": args.growth_loop,
        },
        "next_actions": next_actions(hard_failed, warning_failed, admission_summary, teacher_summary, teacher_smoke_summary, public_summary, growth_loop),
        "score_semantics": "Governance plus latest local growth-loop evidence. Public benchmark scores remain separate and heldout; broad loop scores are private comparator evidence.",
        "external_inference_calls": 0,
    }


def next_actions(
    hard_failed: list[dict[str, Any]],
    warning_failed: list[dict[str, Any]],
    admission_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
    teacher_smoke_summary: dict[str, Any],
    public_summary: dict[str, Any],
    growth_loop: dict[str, Any],
) -> list[str]:
    if hard_failed:
        return ["Fix hard growth-mode gates before launching autonomous training: " + ", ".join(row["name"] for row in hard_failed) + "."]
    actions: list[str] = []
    if int(admission_summary.get("admitted_open_public_source_count") or 0) == 0:
        actions.append("Materialize or repair at least one open-license public source with license/provenance/hash/decontamination metadata.")
    if int(teacher_summary.get("manifest_row_count") or 0) == 0:
        if teacher_smoke_summary.get("gate_trigger_state") == "GREEN":
            actions.append("Request a live governed distillation teacher row when local evidence contains a verifier-accepted private row shape; the admission mechanics are already proven by smoke.")
        else:
            actions.append("Prove the governed teacher distillation admission path, then admit only verifier-accepted rows through the real manifest.")
    if public_summary.get("run_registry_allowed") is not True:
        actions.append("Refresh the public calibration readiness packet and run-registry dry-run before any public execute.")
    if not growth_loop:
        actions.append("Run python3 scripts/permissive_growth_loop.py --execute to produce actual training/evaluation evidence.")
    elif growth_loop.get("ok") is not True:
        actions.append("Fix the failed permissive growth loop before treating the policy as operational.")
    if not actions and warning_failed:
        actions.append("Resolve warning gates before claiming the permissive loop is fully ready.")
    if not actions:
        actions.append("Run a longer multi-cycle permissive growth loop, then decide whether the architecture winner holds under larger matched slices.")
    return actions


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = as_dict(report.get("summary"))
    lines = [
        "# Permissive Growth Mode",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- growth_mode: `{summary.get('growth_mode')}`",
        f"- public/open training allowed: `{summary.get('public_open_training_allowed')}`",
        f"- admitted public/open sources: `{summary.get('admitted_open_public_source_count')}`",
        f"- admitted public/open rows: `{summary.get('admitted_open_public_row_count')}`",
        f"- teacher proposal mode: `{summary.get('teacher_proposal_mode_default')}`",
        f"- teacher distillation allowed now: `{summary.get('teacher_gate_distillation_allowed')}`",
        f"- teacher accepted share/cap: `{summary.get('teacher_accepted_row_share')}` / `{summary.get('teacher_share_cap')}`",
        f"- teacher distillation admission smoke: `{summary.get('teacher_distillation_admission_smoke_state')}`",
        f"- public runner authorization: `{summary.get('public_runner_authorization_mode')}`",
        f"- public runner run registry allowed: `{summary.get('public_runner_run_registry_allowed')}`",
        f"- public runner would execute: `{summary.get('public_runner_would_execute')}`",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
