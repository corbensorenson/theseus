#!/usr/bin/env python3
"""Fail-closed readiness gate for the Mac MLX resource route.

This gate does not claim model quality, CUDA parity, or production scheduler
routing. It proves the route is wired enough to be audited: resource policy is
current, MLX runs in a usable Python, tight budgets fail closed, broader
resource budgets pass, behavior-zero candidates still block production, and
parity claims remain locked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_evidence_store  # noqa: E402


DEFAULT_RESOURCE_POLICY = ROOT / "reports" / "resource_aware_execution_policy.json"
DEFAULT_MLX_DIAGNOSIS = ROOT / "reports" / "macos_mlx_environment_diagnosis.json"
DEFAULT_TIGHT_PROBE = (
    ROOT
    / "reports"
    / "strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v1.json"
)
DEFAULT_BROAD_PROBE = (
    ROOT
    / "reports"
    / "strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v2.json"
)
DEFAULT_PARITY_AUDIT = ROOT / "reports" / "macos_mlx_parity_audit.json"
DEFAULT_OUT = ROOT / "reports" / "resource_mlx_route_readiness_gate.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "resource_mlx_route_readiness_gate.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-policy", default=rel(DEFAULT_RESOURCE_POLICY))
    parser.add_argument("--mlx-diagnosis", default=rel(DEFAULT_MLX_DIAGNOSIS))
    parser.add_argument("--tight-probe", default=rel(DEFAULT_TIGHT_PROBE))
    parser.add_argument("--broad-probe", default=rel(DEFAULT_BROAD_PROBE))
    parser.add_argument("--parity-audit", default=rel(DEFAULT_PARITY_AUDIT))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(
        resource_policy_path=resolve(args.resource_policy),
        mlx_diagnosis_path=resolve(args.mlx_diagnosis),
        tight_probe_path=resolve(args.tight_probe),
        broad_probe_path=resolve(args.broad_probe),
        parity_audit_path=resolve(args.parity_audit),
        started=started,
    )
    report_evidence_store.write_json_report(
        resolve(args.out),
        report,
        markdown_path=resolve(args.markdown_out),
        markdown_text=render_markdown(report),
    )
    view = gate_view(report) if args.gate else report
    print(json.dumps(view, indent=2, sort_keys=True))
    return 2 if report["trigger_state"] == "RED" else 0


def build_report(
    *,
    resource_policy_path: Path,
    mlx_diagnosis_path: Path,
    tight_probe_path: Path,
    broad_probe_path: Path,
    parity_audit_path: Path,
    started: float,
) -> dict[str, Any]:
    resource_policy = read_json(resource_policy_path)
    mlx_diagnosis = read_json(mlx_diagnosis_path)
    tight_probe = read_json(tight_probe_path)
    broad_probe = read_json(broad_probe_path)
    parity_audit = read_json(parity_audit_path)

    resource_summary = summary(resource_policy)
    mlx_summary = summary(mlx_diagnosis)
    tight_summary = summary(tight_probe)
    broad_summary = summary(broad_probe)
    parity_summary = summary(parity_audit)

    checks = [
        check_resource_policy(resource_summary),
        check_mlx_environment(mlx_summary),
        check_tight_budget_fails_closed(tight_summary),
        check_broad_budget_passes(broad_summary),
        check_behavior_zero_blocks_production(broad_summary),
        check_no_cheat_probe_counters("tight_probe", tight_summary),
        check_no_cheat_probe_counters("broad_probe", broad_summary),
        check_runtime_reuse_receipts(broad_summary),
        check_parity_claims_blocked(parity_summary),
    ]
    failed = [row for row in checks if not row["passed"]]
    expected_invalid_controls = [
        expected_invalid("no_mlx_runtime_blocks_route", check_no_mlx_runtime_blocks_route(mlx_summary)),
        expected_invalid("native_abort_blocks_route", check_native_abort_blocks_route(mlx_summary)),
        expected_invalid("public_external_fallback_fault_blocks_route", check_probe_fault_blocks_route(broad_summary)),
        expected_invalid("slow_child_budget_must_fail_closed", check_slow_child_budget_control(tight_summary)),
        expected_invalid("broad_resource_budget_must_pass", check_broad_budget_passes(broad_summary)),
        expected_invalid("quality_zero_must_block_production", check_behavior_zero_blocks_production(broad_summary)),
        expected_invalid("parity_claim_must_remain_blocked", check_parity_claims_blocked(parity_summary)),
        expected_invalid("loader_and_verifier_cache_receipts_required", check_runtime_reuse_receipts(broad_summary)),
    ]
    failed_expected_invalid = [row for row in expected_invalid_controls if not row["passed"]]
    trigger_state = "GREEN" if not failed and not failed_expected_invalid else "RED"

    broad_route = dict_value(broad_summary.get("route_eligibility"))
    broad_resource = dict_value(broad_summary.get("resource_budget"))
    broad_quality = dict_value(broad_route.get("quality_gate"))
    return {
        "policy": "project_theseus_resource_mlx_route_readiness_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "phase": 8,
            "phase_title": "Resource, Cost, And Mac Acceleration Routing",
            "phase8_resource_mlx_route_state": trigger_state,
            "phase8_resource_mlx_route_support_state": "synthetic-test-backed",
            "production_route_eligible": bool(broad_route.get("production_route_eligible")),
            "production_route_block_reason": broad_route.get("route_state"),
            "parity_claim_allowed": bool(parity_summary.get("native_hot_loop_parity_claim_allowed")),
            "accelerator_backend": resource_summary.get("accelerator_backend"),
            "recommended_python": mlx_summary.get("recommended_python"),
            "broad_eval_rows_per_second": broad_resource.get("eval_rows_per_second"),
            "broad_max_child_decode_eval_runtime_ms": broad_resource.get("max_child_decode_eval_runtime_ms"),
            "broad_total_eval_rows": broad_quality.get("total_eval_rows"),
            "broad_total_private_passes": broad_quality.get("total_private_passes"),
            "check_count": len(checks),
            "failed_check_count": len(failed),
            "expected_invalid_count": len(expected_invalid_controls),
            "failed_expected_invalid_count": len(failed_expected_invalid),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "inputs": {
            "resource_policy": rel(resource_policy_path),
            "mlx_diagnosis": rel(mlx_diagnosis_path),
            "tight_probe": rel(tight_probe_path),
            "broad_probe": rel(broad_probe_path),
            "parity_audit": rel(parity_audit_path),
        },
        "checks": checks,
        "expected_invalid_controls": expected_invalid_controls,
        "hard_gaps": failed + failed_expected_invalid,
        "non_claims": [
            "This gate proves Mac MLX route readiness and fail-closed resource policy, not model semantic quality.",
            "This gate does not enable production MLX routing while private behavior remains zero.",
            "This gate does not claim CUDA/MLX/Metal parity.",
            "This gate does not run training, public calibration, teacher inference, or external inference.",
            "Tool/template/router outputs remain non-learned-generation credit under AGENTS.md.",
        ],
        "next_wall": {
            "owner_phase": 10,
            "summary": "Semantic candidate quality remains the next blocker before production route eligibility.",
        },
    }


def check_resource_policy(resource_summary: dict[str, Any]) -> dict[str, Any]:
    return check(
        "resource_policy_mlx_available",
        bool(resource_summary.get("accelerator_available"))
        and resource_summary.get("accelerator_backend") == "mlx_apple"
        and resource_summary.get("run_public_calibration") is False,
        {
            "accelerator_available": resource_summary.get("accelerator_available"),
            "accelerator_backend": resource_summary.get("accelerator_backend"),
            "run_public_calibration": resource_summary.get("run_public_calibration"),
            "profile": resource_summary.get("profile"),
        },
    )


def check_mlx_environment(mlx_summary: dict[str, Any]) -> dict[str, Any]:
    return check(
        "mlx_environment_usable_without_native_abort",
        mlx_summary.get("route_action") == "route_mlx_to_usable_python"
        and int_value(mlx_summary.get("usable_mlx_runtime_count")) > 0
        and int_value(mlx_summary.get("native_abort_count")) == 0
        and int_value(mlx_summary.get("public_training_rows_written")) == 0
        and int_value(mlx_summary.get("external_inference_calls")) == 0
        and int_value(mlx_summary.get("fallback_return_count")) == 0,
        {
            "route_action": mlx_summary.get("route_action"),
            "recommended_python": mlx_summary.get("recommended_python"),
            "usable_mlx_runtime_count": mlx_summary.get("usable_mlx_runtime_count"),
            "native_abort_count": mlx_summary.get("native_abort_count"),
            "public_training_rows_written": mlx_summary.get("public_training_rows_written"),
            "external_inference_calls": mlx_summary.get("external_inference_calls"),
            "fallback_return_count": mlx_summary.get("fallback_return_count"),
        },
    )


def check_tight_budget_fails_closed(probe_summary: dict[str, Any]) -> dict[str, Any]:
    route = dict_value(probe_summary.get("route_eligibility"))
    resource = dict_value(probe_summary.get("resource_budget"))
    return check(
        "tight_child_budget_fails_closed_on_slow_rung",
        route.get("production_route_eligible") is False
        and route.get("route_state") == "fail_closed_resource_budget"
        and resource.get("enforced") is True
        and resource.get("budget_ok") is False
        and resource.get("child_budget_ok") is False
        and "rung_25000000" in list_value(resource.get("slow_child_decode_eval_ids"))
        and resource.get("throughput_ok") is True
        and resource.get("total_budget_ok") is True,
        {
            "route_state": route.get("route_state"),
            "production_route_eligible": route.get("production_route_eligible"),
            "budget_ok": resource.get("budget_ok"),
            "child_budget_ok": resource.get("child_budget_ok"),
            "slow_child_decode_eval_ids": resource.get("slow_child_decode_eval_ids"),
            "throughput_ok": resource.get("throughput_ok"),
            "total_budget_ok": resource.get("total_budget_ok"),
        },
    )


def check_broad_budget_passes(probe_summary: dict[str, Any]) -> dict[str, Any]:
    resource = dict_value(probe_summary.get("resource_budget"))
    performance = dict_value(probe_summary.get("performance"))
    return check(
        "broad_resource_budget_passes_with_minimum_private_throughput",
        resource.get("enforced") is True
        and resource.get("budget_ok") is True
        and resource.get("child_budget_ok") is True
        and resource.get("throughput_ok") is True
        and float_value(resource.get("eval_rows_per_second")) >= float_value(resource.get("min_eval_rows_per_second"), 0.15)
        and int_value(resource.get("max_child_decode_eval_runtime_ms")) <= int_value(resource.get("max_child_decode_eval_ms"), 25000)
        and int_value(performance.get("total_eval_rows")) >= 10
        and int_value(performance.get("total_generated_candidate_rows")) >= 10,
        {
            "budget_ok": resource.get("budget_ok"),
            "child_budget_ok": resource.get("child_budget_ok"),
            "throughput_ok": resource.get("throughput_ok"),
            "eval_rows_per_second": resource.get("eval_rows_per_second"),
            "min_eval_rows_per_second": resource.get("min_eval_rows_per_second"),
            "max_child_decode_eval_runtime_ms": resource.get("max_child_decode_eval_runtime_ms"),
            "max_child_decode_eval_ms": resource.get("max_child_decode_eval_ms"),
            "total_eval_rows": performance.get("total_eval_rows"),
            "total_generated_candidate_rows": performance.get("total_generated_candidate_rows"),
        },
    )


def check_behavior_zero_blocks_production(probe_summary: dict[str, Any]) -> dict[str, Any]:
    route = dict_value(probe_summary.get("route_eligibility"))
    quality = dict_value(route.get("quality_gate"))
    return check(
        "behavior_zero_blocks_production_route",
        route.get("production_route_eligible") is False
        and route.get("route_state") == "fail_closed_behavior_quality_zero"
        and int_value(quality.get("total_eval_rows")) >= 10
        and int_value(quality.get("total_private_passes")) == 0
        and quality.get("quality_ok") is False,
        {
            "production_route_eligible": route.get("production_route_eligible"),
            "route_state": route.get("route_state"),
            "quality_gate": quality,
        },
    )


def check_no_cheat_probe_counters(label: str, probe_summary: dict[str, Any]) -> dict[str, Any]:
    route = dict_value(probe_summary.get("route_eligibility"))
    integrity = dict_value(route.get("integrity_gate"))
    no_cheat = dict_value(route.get("no_cheat_gate"))
    return check(
        f"{label}_no_public_external_fallback_or_integrity_faults",
        int_value(route.get("public_training_rows")) == 0
        and int_value(route.get("external_inference_calls")) == 0
        and int_value(route.get("fallback_template_router_tool_credit_count")) == 0
        and no_cheat.get("no_cheat_ok") is True
        and integrity.get("integrity_ok") is True
        and int_value(integrity.get("integrity_mismatch_count")) == 0,
        {
            "public_training_rows": route.get("public_training_rows"),
            "external_inference_calls": route.get("external_inference_calls"),
            "fallback_template_router_tool_credit_count": route.get("fallback_template_router_tool_credit_count"),
            "no_cheat_gate": no_cheat,
            "integrity_gate": integrity,
        },
    )


def check_runtime_reuse_receipts(probe_summary: dict[str, Any]) -> dict[str, Any]:
    loader = dict_value(probe_summary.get("checkpoint_loader_reuse"))
    loader_stats = dict_value(loader.get("stats"))
    verifier = dict_value(probe_summary.get("verifier_cache_warmup"))
    return check(
        "loader_reuse_and_verifier_cache_receipts_present",
        loader.get("enabled") is True
        and int_value(loader_stats.get("vocab_read_count")) == 1
        and int_value(loader_stats.get("vocab_cache_hit_count")) >= 4
        and int_value(loader_stats.get("model_construct_count")) == 1
        and int_value(loader_stats.get("model_reuse_count")) >= 4
        and int_value(loader_stats.get("checkpoint_weight_load_count")) >= 5
        and verifier.get("all_test_harness_compile_cache_enabled") is True
        and int_value(verifier.get("total_test_harness_cache_hit_count")) >= 10,
        {
            "loader_stats": loader_stats,
            "verifier_total_test_harness_cache_hit_count": verifier.get("total_test_harness_cache_hit_count"),
            "all_test_harness_compile_cache_enabled": verifier.get("all_test_harness_compile_cache_enabled"),
        },
    )


def check_parity_claims_blocked(parity_summary: dict[str, Any]) -> dict[str, Any]:
    return check(
        "parity_claim_and_production_route_remain_blocked",
        parity_summary.get("native_hot_loop_parity_claim_allowed") is False
        and int_value(parity_summary.get("production_route_pending_count")) > 0
        and int_value(parity_summary.get("native_metal_production_route_ready_count")) == 0
        and int_value(parity_summary.get("kernel_parity_pending_count")) > 0,
        {
            "native_hot_loop_parity_claim_allowed": parity_summary.get("native_hot_loop_parity_claim_allowed"),
            "production_route_pending_count": parity_summary.get("production_route_pending_count"),
            "native_metal_production_route_ready_count": parity_summary.get("native_metal_production_route_ready_count"),
            "kernel_parity_pending_count": parity_summary.get("kernel_parity_pending_count"),
        },
    )


def check_no_mlx_runtime_blocks_route(mlx_summary: dict[str, Any]) -> dict[str, Any]:
    return check(
        "control_no_mlx_runtime_would_block_route",
        int_value(mlx_summary.get("usable_mlx_runtime_count")) > 0
        and mlx_summary.get("route_action") == "route_mlx_to_usable_python",
        {
            "usable_mlx_runtime_count": mlx_summary.get("usable_mlx_runtime_count"),
            "route_action": mlx_summary.get("route_action"),
        },
    )


def check_native_abort_blocks_route(mlx_summary: dict[str, Any]) -> dict[str, Any]:
    return check(
        "control_native_abort_would_block_route",
        int_value(mlx_summary.get("native_abort_count")) == 0,
        {"native_abort_count": mlx_summary.get("native_abort_count")},
    )


def check_probe_fault_blocks_route(probe_summary: dict[str, Any]) -> dict[str, Any]:
    route = dict_value(probe_summary.get("route_eligibility"))
    return check(
        "control_public_external_fallback_faults_absent",
        int_value(route.get("public_training_rows")) == 0
        and int_value(route.get("external_inference_calls")) == 0
        and int_value(route.get("fallback_template_router_tool_credit_count")) == 0,
        {
            "public_training_rows": route.get("public_training_rows"),
            "external_inference_calls": route.get("external_inference_calls"),
            "fallback_template_router_tool_credit_count": route.get("fallback_template_router_tool_credit_count"),
        },
    )


def check_slow_child_budget_control(probe_summary: dict[str, Any]) -> dict[str, Any]:
    return check_tight_budget_fails_closed(probe_summary)


def check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def expected_invalid(name: str, check_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(check_row)
    row["name"] = name
    row["expected_invalid_control"] = True
    return row


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "non_claims": report["non_claims"],
        "next_wall": report["next_wall"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Resource MLX Route Readiness Gate",
        "",
        f"- state: `{report['trigger_state']}`",
        f"- support: `{report['summary']['phase8_resource_mlx_route_support_state']}`",
        f"- production route eligible: `{report['summary']['production_route_eligible']}`",
        f"- production route block: `{report['summary']['production_route_block_reason']}`",
        f"- parity claim allowed: `{report['summary']['parity_claim_allowed']}`",
        "",
        "## Checks",
    ]
    for row in report["checks"]:
        mark = "PASS" if row["passed"] else "FAIL"
        lines.append(f"- `{mark}` {row['name']}")
    lines.extend(["", "## Non-Claims"])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("summary")
    return value if isinstance(value, dict) else report


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
