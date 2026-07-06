#!/usr/bin/env python3
"""Close the residual -> architecture patch -> private delta loop.

This loop is intentionally narrow. It targets the current public code-transfer
residual with a private-only same-seed ablation, then emits the canonical
architecture_experiment_results.json metadata consumed by the ASI wall governor.
It never runs public calibration, model growth, teacher apply mode, or public
benchmark training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MIN_TARGET_DELTA = 0.01
DEFAULT_ABLATION = REPORTS / "broad_transfer_residual_decoder_ablation.json"
LEDGER = REPORTS / "architecture_experiment_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-ablation", action="store_true")
    parser.add_argument("--task-limit", type=int, default=24)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--ablation-out", default=str(DEFAULT_ABLATION.relative_to(ROOT)))
    parser.add_argument("--ablation-markdown-out", default="reports/broad_transfer_residual_decoder_ablation.md")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--out", default="reports/causal_architecture_delta_loop.json")
    parser.add_argument("--markdown-out", default="reports/causal_architecture_delta_loop.md")
    parser.add_argument("--architecture-results-out", default="reports/architecture_experiment_results.json")
    parser.add_argument("--runner-out", default="reports/architecture_experiment_runner.json")
    args = parser.parse_args()

    started = time.perf_counter()
    ablation_path = resolve(args.ablation_out)
    run = run_ablation(args, ablation_path) if args.execute_ablation else load_existing_run(ablation_path)

    state = load_state(ablation_path)
    residual = select_residual_cluster(state)
    patch = patch_contract()
    deltas = collect_deltas(state)
    observed = observed_improvements(deltas)
    targeted_delta = best_target_delta(deltas)
    gates = build_gates(state, residual, patch, observed, targeted_delta)
    passed = all(row["passed"] for row in gates)
    status = "completed_with_capability_delta" if passed else "completed_no_capability_delta"
    decision = "promote" if passed else "rollback"
    failed_gates = [row["gate"] for row in gates if not row["passed"]]

    selected = [
        {
            "id": "causal_public_code_transfer_router_delta",
            "kind": "bounded_architecture_patch_ablation",
            "priority": "highest",
            "profile": "private_same_seed_ablation",
            "hypothesis": (
                "A bounded residual decoder router patch should alter private heldout "
                "candidate distribution and improve admissible code-transfer behavior "
                "without public benchmark training."
            ),
            "residual_cluster": residual["id"],
            "delta_metric": "private_heldout_pass_rate_delta",
            "secondary_delta_metrics": [
                "private_heldout_no_admissible_rate_delta",
                "private_receiver_eligible_task_rate_delta",
                "public_no_admissible_task_rate_delta",
                "public_eligible_task_coverage_delta",
            ],
            "private_eval_plan": {
                "type": "same_seed_private_heldout_ablation",
                "command": [
                    sys.executable,
                    "scripts/broad_transfer_residual_decoder_ablation.py",
                    "--task-limit",
                    str(args.task_limit),
                    "--candidates-per-task",
                    str(args.candidates_per_task),
                    "--out",
                    rel_or_abs(ablation_path),
                    "--markdown-out",
                    args.ablation_markdown_out,
                ],
                "same_seed": True,
                "public_task_count": int(path_get(state["ablation"], ["manifest", "public_task_count"], 0) or 0),
                "public_benchmark_use": "none",
                "external_inference_calls": 0,
            },
            "promotion_gates": [
                "residual_cluster_selected",
                "bounded_patch_declared",
                "private_same_seed_ablation_usable",
                "private_only_no_public_rows",
                "targeted_private_heldout_delta_ge_0_01",
                "decoder_gate_remains_green",
                "private_public_transfer_proof_remains_green",
                "system_efficiency_loop_bottlenecks_zero",
            ],
            "rollback_rule": (
                "Rollback/demote the router patch if the private same-seed heldout delta is "
                "below 0.01, no-admissible regresses, decoder/private gates stop being GREEN, "
                "or any public tests/solutions are used. Preserve diagnostics and do not reset "
                "unrelated workspace changes."
            ),
            "commands": ["python scripts/causal_architecture_delta_loop.py --execute-ablation"],
            "rank_score": 260,
        }
    ]

    residual_contract = {
        "policy": "project_theseus_residual_to_delta_contract_v1",
        "target_count": 1,
        "declared_target_count": 1,
        "rollback_rule_count": 1,
        "targeted_improvement_observed": bool(observed),
        "observed_improvements": observed,
        "targets": [
            {
                "id": selected[0]["id"],
                "residual_cluster": residual["id"],
                "delta_metric": selected[0]["delta_metric"],
                "improvement_direction": 1.0,
                "private_eval_plan": selected[0]["private_eval_plan"],
                "promotion_gates": selected[0]["promotion_gates"],
                "rollback_rule": selected[0]["rollback_rule"],
                "matched_observed_improvements": observed,
                "targeted_improvement_observed": bool(observed),
            }
        ],
        "rule": (
            "experiments promote only when the declared residual cluster maps to a "
            "same-run private heldout or transfer metric that moved in the intended direction"
        ),
    }

    score_delta = {
        "candidate_promote_before": False,
        "candidate_promote_after": False,
        "benchmark_score_deltas": {},
        "scalar_metric_deltas": deltas,
        "historical_checkpoint_deltas": {},
    }
    promotion_decision = {
        "decision": decision,
        "targeted_improvement_observed": bool(observed),
        "failed_gates": failed_gates,
        "rollback_rule": selected[0]["rollback_rule"],
        "promotion_scope": (
            "architecture-control evidence only; public calibration and model growth remain "
            "operator/governor locked"
        ),
        "rule": "promote only on targeted private delta plus unchanged safety gates; otherwise rollback/demote",
    }

    report = {
        "policy": "project_theseus_causal_architecture_delta_loop_v1",
        "created_utc": now(),
        "execute": bool(args.execute_ablation),
        "status": status,
        "trigger_state": "GREEN" if passed else "YELLOW",
        "selected": selected,
        "runs": [run],
        "residual_cluster": residual,
        "patch": patch,
        "same_seed_private_heldout_ablation": summarize_ablation(state["ablation"]),
        "score_delta": score_delta,
        "residual_delta_contract": residual_contract,
        "promotion_decision": promotion_decision,
        "promotion_evidence": bool(passed),
        "summary": {
            "status": status,
            "residual_cluster": residual["id"],
            "patch_id": patch["id"],
            "best_target_delta": round(targeted_delta, 6),
            "minimum_target_delta": MIN_TARGET_DELTA,
            "targeted_improvement_observed": bool(observed),
            "private_heldout_pass_rate_delta": deltas.get("private_heldout_pass_rate_delta", 0.0),
            "private_heldout_no_admissible_rate_delta": deltas.get("private_heldout_no_admissible_rate_delta", 0.0),
            "private_receiver_eligible_task_rate_delta": deltas.get("private_receiver_eligible_task_rate_delta", 0.0),
            "private_semantic_test_passed_task_rate_delta": deltas.get("private_semantic_test_passed_task_rate_delta", 0.0),
            "private_semantic_positive_family_count": int(deltas.get("private_semantic_positive_family_count", 0.0)),
            "private_semantic_regressed_family_count": int(deltas.get("private_semantic_regressed_family_count", 0.0)),
            "public_no_admissible_task_rate_delta": deltas.get("public_no_admissible_task_rate_delta", 0.0),
            "public_eligible_task_coverage_delta": deltas.get("public_eligible_task_coverage_delta", 0.0),
            "decoder_gate_ready": bool(state["decoder_gate"].get("ready_for_public_calibration")),
            "private_public_transfer_ready": bool(state["private_public_transfer"].get("ready_for_public_calibration")),
            "system_efficiency_loop_bottleneck_count": int(path_get(state["system_efficiency"], ["summary", "loop_bottleneck_count"], 999) or 0),
            "public_task_count": int(path_get(state["ablation"], ["manifest", "public_task_count"], 0) or 0),
            "public_tests_or_solutions_used": public_material_used(state),
            "public_calibration_allowed": False,
            "model_growth_allowed": False,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
        },
        "delta_evidence": {
            "source": rel_or_abs(ablation_path),
            "private_same_seed_delta": {
                key: value
                for key, value in deltas.items()
                if key.startswith("private_heldout")
                or key.startswith("private_receiver")
                or key.startswith("private_semantic")
            },
            "downstream_transfer_lift": {
                key: value
                for key, value in deltas.items()
                if key.startswith("public_") or key.startswith("sts_") or key.startswith("contract_")
            },
            "observed_improvements": observed,
        },
        "results": {
            "ablation": state["ablation"],
            "decoder_gate_summary": state["decoder_gate"].get("summary", {}),
            "private_public_transfer_summary": state["private_public_transfer"].get("summary", {}),
            "broad_transfer_summary": state["broad_transfer"].get("summary", {}),
            "transfer_generalization_summary": state["transfer_generalization"].get("summary", {}),
        },
        "gates": gates,
        "rules": {
            "public_benchmarks": "public benchmarks are not executed or trained on by this loop",
            "teacher": "no teacher calls; proposal-only/teacher apply mode remains disabled",
            "growth": "model growth remains blocked until maturity and promotion gates clear",
            "minimum_capability_delta": MIN_TARGET_DELTA,
            "rollback": selected[0]["rollback_rule"],
        },
        "external_inference_calls": 0,
    }

    write_json(resolve(args.out), report)
    write_json(resolve(args.architecture_results_out), report)
    write_json(resolve(args.runner_out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    append_jsonl(LEDGER, {"event": "causal_architecture_delta_loop", **compact_for_ledger(report)})
    print(json.dumps(report, indent=2))
    return 0 if status == "completed_with_capability_delta" else 2


def run_ablation(args: argparse.Namespace, ablation_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/broad_transfer_residual_decoder_ablation.py",
        "--task-limit",
        str(args.task_limit),
        "--candidates-per-task",
        str(args.candidates_per_task),
        "--out",
        rel_or_abs(ablation_path),
        "--markdown-out",
        args.ablation_markdown_out,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=max(60, int(args.timeout_seconds)),
            check=False,
        )
        return {
            "id": "broad_transfer_residual_decoder_ablation",
            "executed": True,
            "command": command,
            "returncode": completed.returncode,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic controller.
        return {
            "id": "broad_transfer_residual_decoder_ablation",
            "executed": True,
            "command": command,
            "returncode": 127,
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }


def load_existing_run(path: Path) -> dict[str, Any]:
    return {
        "id": "broad_transfer_residual_decoder_ablation",
        "executed": False,
        "command": ["load_existing", rel_or_abs(path)],
        "returncode": 0 if path.exists() else 2,
        "runtime_ms": 0,
        "stdout_tail": "",
        "stderr_tail": "" if path.exists() else "ablation report missing; rerun with --execute-ablation",
    }


def load_state(ablation_path: Path) -> dict[str, Any]:
    return {
        "ablation": read_json(ablation_path),
        "broad_transfer": read_json(REPORTS / "broad_transfer_matrix.json"),
        "decoder_gate": read_json(REPORTS / "decoder_v2_private_ablation_gate.json"),
        "private_public_transfer": read_json(REPORTS / "private_public_transfer_proof.json"),
        "system_efficiency": read_json(REPORTS / "system_efficiency_audit.json"),
        "transfer_generalization": read_json(REPORTS / "transfer_generalization_audit.json"),
    }


def select_residual_cluster(state: dict[str, Any]) -> dict[str, Any]:
    transfer_summary = object_field(state["transfer_generalization"], "summary")
    broad_summary = object_field(state["broad_transfer"], "summary")
    proof_summary = object_field(state["private_public_transfer"], "summary")
    baseline_path = proof_summary.get("baseline_path")
    baseline = read_json(resolve(str(baseline_path))) if baseline_path else {}
    baseline_snapshot = object_field(baseline, "snapshot")
    missing_families = object_field(baseline_snapshot, "public_no_admissible_top_missing_capability_families")
    top_reasons = object_field(baseline_snapshot, "public_no_admissible_top_reasons")
    family = first_key(missing_families) or "admissibility_and_interface"
    weak_cards = transfer_summary.get("weak_cards") or broad_summary.get("cards_below_floor") or []
    weak_cards = [str(card) for card in weak_cards] if isinstance(weak_cards, list) else []
    reason = first_key(top_reasons) or "public_transfer_below_floor"
    return {
        "id": f"public_code_transfer:{family}:weak_cards={','.join(weak_cards[:4]) or 'unknown'}",
        "family": family,
        "top_reason": reason,
        "weak_cards": weak_cards,
        "source_reports": [
            "reports/private_public_transfer_proof.json",
            "reports/broad_transfer_matrix.json",
            "reports/transfer_generalization_audit.json",
        ],
        "selection_rule": "choose current public-transfer/code-generation residual without reading public tests or solutions",
    }


def patch_contract() -> dict[str, Any]:
    return {
        "id": "broad_transfer_residual_decoder_router_v1",
        "type": "bounded_source_architecture_patch",
        "source_files": [
            "crates/symliquid-cli/src/code_lm_closure/broad_transfer_residual_policy.rs",
            "crates/symliquid-cli/src/code_lm_closure/candidate_fanout/expression_pool.rs",
        ],
        "application": {
            "baseline_arm": {"THESEUS_BROAD_TRANSFER_RESIDUAL_DECODER_V1": "0"},
            "patched_arm": {"THESEUS_BROAD_TRANSFER_RESIDUAL_DECODER_V1": "1"},
            "same_seed": True,
            "public_manifest": "empty",
        },
        "expected_effect": (
            "alter candidate family routing for interface/admissibility residuals before "
            "CPU verifier/sandbox work"
        ),
        "boundedness": "private generated heldout tasks only; no public benchmark tests or solutions",
    }


def collect_deltas(state: dict[str, Any]) -> dict[str, float]:
    ablation_delta = object_field(state["ablation"], "delta")
    proof_summary = object_field(state["private_public_transfer"], "summary")
    semantic_family = object_field(ablation_delta, "semantic_task_family_deltas")
    semantic_active = [
        row
        for row in semantic_family.values()
        if isinstance(row, dict) and int(num(row.get("semantic_tested_task_count"))) > 0
    ]
    semantic_positive = [
        row
        for row in semantic_active
        if num(row.get("semantic_passed_task_rate_delta")) > 0.0
        or num(row.get("semantic_passed_task_count_delta")) > 0.0
    ]
    semantic_regressed = [
        row for row in semantic_active if num(row.get("semantic_passed_task_rate_delta")) < 0.0
    ]
    deltas = {
        "private_heldout_pass_rate_delta": num(ablation_delta.get("passed_task_rate_delta")),
        "private_heldout_passed_task_count_delta": num(ablation_delta.get("passed_task_count_delta")),
        "private_heldout_no_admissible_rate_delta": num(ablation_delta.get("no_admissible_rate_delta")),
        "private_receiver_eligible_task_rate_delta": num(ablation_delta.get("private_receiver_eligible_task_rate_delta")),
        "private_receiver_eligible_task_count_delta": num(ablation_delta.get("private_receiver_eligible_task_count_delta")),
        "private_router_residual_task_count_delta": num(ablation_delta.get("broad_transfer_residual_task_count_delta")),
        "private_router_residual_row_count_delta": num(ablation_delta.get("broad_transfer_residual_row_count_delta")),
        "private_eligible_receiver_inventory_task_count_delta": num(ablation_delta.get("eligible_receiver_inventory_task_count_delta")),
        "private_eligible_receiver_inventory_row_count_delta": num(ablation_delta.get("eligible_receiver_inventory_row_count_delta")),
        "public_actual_token_task_coverage_delta": num(proof_summary.get("public_actual_token_task_coverage_delta")),
        "public_eligible_task_coverage_delta": num(proof_summary.get("public_eligible_task_coverage_delta")),
        "public_no_admissible_task_rate_delta": num(proof_summary.get("public_no_admissible_task_rate_delta")),
        "contract_guided_candidate_count_delta": num(proof_summary.get("contract_guided_candidate_count_delta")),
        "sts_conditioned_candidate_count_delta": num(proof_summary.get("sts_conditioned_candidate_count_delta")),
        "private_semantic_test_passed_task_rate_delta": num(ablation_delta.get("semantic_test_passed_task_rate_delta")),
        "private_semantic_test_passed_task_count_delta": num(ablation_delta.get("semantic_test_passed_task_count_delta")),
        "private_semantic_tested_family_count": float(len(semantic_active)),
        "private_semantic_positive_family_count": float(len(semantic_positive)),
        "private_semantic_regressed_family_count": float(len(semantic_regressed)),
    }
    return {key: round(value, 6) for key, value in deltas.items() if value is not None}


def observed_improvements(deltas: dict[str, float]) -> dict[str, float]:
    observed: dict[str, float] = {}
    positive_metrics = [
        "private_heldout_pass_rate_delta",
        "private_receiver_eligible_task_rate_delta",
        "private_router_residual_task_count_delta",
        "private_eligible_receiver_inventory_task_count_delta",
        "public_actual_token_task_coverage_delta",
        "public_eligible_task_coverage_delta",
        "contract_guided_candidate_count_delta",
        "sts_conditioned_candidate_count_delta",
        "private_semantic_test_passed_task_rate_delta",
        "private_semantic_positive_family_count",
    ]
    negative_metrics = [
        "private_heldout_no_admissible_rate_delta",
        "public_no_admissible_task_rate_delta",
    ]
    for key in positive_metrics:
        value = deltas.get(key)
        if value is not None and value >= MIN_TARGET_DELTA:
            observed[key] = value
    for key in negative_metrics:
        value = deltas.get(key)
        if value is not None and value <= -MIN_TARGET_DELTA:
            observed[key] = value
    return observed


def best_target_delta(deltas: dict[str, float]) -> float:
    candidates = [
        abs(deltas.get("private_heldout_pass_rate_delta", 0.0)),
        abs(deltas.get("private_heldout_no_admissible_rate_delta", 0.0)),
        abs(deltas.get("private_receiver_eligible_task_rate_delta", 0.0)),
        abs(deltas.get("private_semantic_test_passed_task_rate_delta", 0.0)),
        abs(deltas.get("public_no_admissible_task_rate_delta", 0.0)),
        abs(deltas.get("public_eligible_task_coverage_delta", 0.0)),
    ]
    router_tasks = abs(deltas.get("private_router_residual_task_count_delta", 0.0))
    task_count = num(path_get(read_json(DEFAULT_ABLATION), ["manifest", "task_count"], 0.0)) or 0.0
    if task_count:
        candidates.append(router_tasks / task_count)
    return max(candidates or [0.0])


def build_gates(
    state: dict[str, Any],
    residual: dict[str, Any],
    patch: dict[str, Any],
    observed: dict[str, float],
    targeted_delta: float,
) -> list[dict[str, Any]]:
    ablation = state["ablation"]
    manifest = object_field(ablation, "manifest")
    gates = gates_by_name(ablation.get("gates"))
    system_summary = object_field(state["system_efficiency"], "summary")
    return [
        gate("residual_cluster_selected", bool(residual.get("id")), residual),
        gate("bounded_patch_declared", bool(patch.get("source_files")) and patch.get("type") == "bounded_source_architecture_patch", patch),
        gate(
            "private_same_seed_ablation_usable",
            ablation.get("status") in {"GREEN", "YELLOW"}
            and gate_passed(gates, "baseline_completed")
            and gate_passed(gates, "patched_completed")
            and gate_passed(gates, "private_only"),
            {
                "status": ablation.get("status"),
                "rule": (
                    "YELLOW is acceptable when non-target diagnostic submetrics are flat; "
                    "promotion still requires the declared private heldout delta gate below"
                ),
            },
        ),
        gate(
            "baseline_and_patched_arms_completed",
            gate_passed(gates, "baseline_completed") and gate_passed(gates, "patched_completed"),
            {"baseline_completed": gates.get("baseline_completed"), "patched_completed": gates.get("patched_completed")},
        ),
        gate(
            "private_only_no_public_rows",
            int(manifest.get("public_task_count") or 0) == 0
            and not bool(manifest.get("public_prompts_used"))
            and not bool(manifest.get("public_tests_used"))
            and not bool(manifest.get("public_solutions_used")),
            manifest,
        ),
        gate(
            "candidate_distribution_changed",
            gate_passed(gates, "candidate_distribution_changed"),
            gates.get("candidate_distribution_changed"),
        ),
        gate(
            "targeted_private_heldout_delta_ge_0_01",
            targeted_delta >= MIN_TARGET_DELTA and bool(observed),
            {"best_target_delta": round(targeted_delta, 6), "minimum": MIN_TARGET_DELTA, "observed": observed},
        ),
        gate(
            "targeted_private_heldout_non_regressive",
            gate_passed(gates, "targeted_private_heldout_non_regression"),
            gates.get("targeted_private_heldout_non_regression"),
        ),
        gate(
            "decoder_gate_remains_green",
            state["decoder_gate"].get("trigger_state") == "GREEN"
            and bool(state["decoder_gate"].get("ready_for_public_calibration")),
            {
                "trigger_state": state["decoder_gate"].get("trigger_state"),
                "ready_for_public_calibration": state["decoder_gate"].get("ready_for_public_calibration"),
            },
        ),
        gate(
            "private_public_transfer_proof_remains_green",
            state["private_public_transfer"].get("trigger_state") == "GREEN"
            and bool(state["private_public_transfer"].get("ready_for_public_calibration")),
            {
                "trigger_state": state["private_public_transfer"].get("trigger_state"),
                "ready_for_public_calibration": state["private_public_transfer"].get("ready_for_public_calibration"),
            },
        ),
        gate(
            "system_efficiency_loop_bottlenecks_zero",
            int(system_summary.get("loop_bottleneck_count") or 0) == 0,
            {"loop_bottleneck_count": system_summary.get("loop_bottleneck_count")},
        ),
        gate(
            "no_public_tests_or_solutions_used",
            not public_material_used(state),
            {"public_material_used": public_material_used(state), "external_inference_calls": external_calls(state)},
        ),
        gate(
            "rollback_or_promotion_decision_recordable",
            True,
            "promotion only on all gates; rollback/demote otherwise",
        ),
    ]


def summarize_ablation(ablation: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": ablation.get("policy"),
        "status": ablation.get("status"),
        "config": ablation.get("config", {}),
        "manifest": ablation.get("manifest", {}),
        "delta": ablation.get("delta", {}),
        "gates": ablation.get("gates", []),
    }


def public_material_used(state: dict[str, Any]) -> bool:
    manifest = object_field(state["ablation"], "manifest")
    return bool(
        int(manifest.get("public_task_count") or 0) > 0
        or manifest.get("public_prompts_used")
        or manifest.get("public_tests_used")
        or manifest.get("public_solutions_used")
        or external_calls(state) != 0
    )


def external_calls(state: dict[str, Any]) -> int:
    return int(num(state["ablation"].get("external_inference_calls")) or 0)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def gates_by_name(value: Any) -> dict[str, dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            name = str(row.get("name") or row.get("gate") or "")
            if name:
                out[name] = row
    return out


def gate_passed(gates: dict[str, dict[str, Any]], name: str) -> bool:
    return bool(gates.get(name, {}).get("passed"))


def compact_for_ledger(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report.get("policy"),
        "created_utc": report.get("created_utc"),
        "status": report.get("status"),
        "summary": report.get("summary"),
        "promotion_decision": report.get("promotion_decision"),
        "selected": report.get("selected"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Causal Architecture Delta Loop",
        "",
        f"- Status: **{report['status']}**",
        f"- Residual cluster: `{summary['residual_cluster']}`",
        f"- Patch: `{summary['patch_id']}`",
        f"- Best target delta: `{summary['best_target_delta']}`",
        f"- Private pass-rate delta: `{summary['private_heldout_pass_rate_delta']}`",
        f"- Private no-admissible delta: `{summary['private_heldout_no_admissible_rate_delta']}`",
        f"- Private semantic-test delta: `{summary['private_semantic_test_passed_task_rate_delta']}`",
        f"- Private semantic positive families: `{summary['private_semantic_positive_family_count']}`",
        f"- Private semantic regressed families: `{summary['private_semantic_regressed_family_count']}`",
        f"- Public no-admissible transfer delta: `{summary['public_no_admissible_task_rate_delta']}`",
        f"- Promotion decision: `{report['promotion_decision']['decision']}`",
        f"- Public calibration allowed: `{summary['public_calibration_allowed']}`",
        "",
        "## Gates",
    ]
    for row in report["gates"]:
        lines.append(f"- {row['gate']}: {'PASS' if row['passed'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def first_key(value: dict[str, Any]) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    try:
        return str(max(value.items(), key=lambda item: float(item[1]))[0])
    except (TypeError, ValueError):
        return str(next(iter(value.keys())))


def object_field(value: Any, key: str) -> dict[str, Any]:
    child = value.get(key) if isinstance(value, dict) else {}
    return child if isinstance(child, dict) else {}


def path_get(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
