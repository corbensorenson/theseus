#!/usr/bin/env python3
"""Named smoke workloads for Circle Calculus AI contracts.

These are deterministic structural workloads, not model benchmarks. They attach
names, metrics, and ordinary baseline fields to the six Circle contract
families so later private Theseus-Hive experiments have stable report slots.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT.parent / "circle math" / "site" / "data" / "generated" / "theseus_hive_ai_contracts.json"
DEFAULT_OUT = ROOT / "reports" / "circle_ai_private_workload_smoke.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "circle_ai_private_workload_smoke.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", default=str(DEFAULT_CONTRACTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    source = resolve(args.contracts)
    pack = read_json(source)
    report = build_report(pack, source)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(pack: dict[str, Any], source: Path) -> dict[str, Any]:
    contracts = [row for row in pack.get("contracts", []) if isinstance(row, dict)]
    workloads = [workload_for_contract(row) for row in contracts]
    failed = [row["workload_id"] for row in workloads if not row["passed"]]
    unknown = [row["workload_id"] for row in workloads if row["kind"] == "unknown"]
    hard_failures = failed + unknown
    return {
        "policy": "theseus_hive_circle_ai_private_workload_smoke_v0",
        "created_utc": now(),
        "trigger_state": "RED" if hard_failures else "YELLOW",
        "source_contract_path": rel(source),
        "circle_contract_schema_id": pack.get("schema_id", ""),
        "claim_boundary": (
            "These are deterministic structural smoke workloads. They are not "
            "model-quality, reasoning, speed, context-length, public-transfer, "
            "promotion, or ASI evidence."
        ),
        "score_semantics": "named private-smoke structural checks only; no learned model is evaluated",
        "public_calibration_used": False,
        "external_inference_calls": 0,
        "training_mutation": False,
        "promotion_evidence": False,
        "private_data_exported": False,
        "summary": {
            "workload_count": len(workloads),
            "passed_workload_count": sum(1 for row in workloads if row["passed"]),
            "failed_workloads": failed,
            "unknown_workloads": unknown,
            "model_quality_metrics_present": False,
            "named_workloads_present": bool(workloads),
        },
        "governance_gates": [
            gate("external_inference_calls_zero", True, "No model/API/local inference is called."),
            gate("training_mutation_absent", True, "This script writes reports only."),
            gate("promotion_evidence_absent", True, "Smoke metrics are non-promotion evidence."),
            gate("public_calibration_absent", True, "No public benchmark calibration is used."),
            gate("model_quality_absent", True, "Metrics are structural smoke checks only."),
            gate("known_workload_kinds", not unknown, unknown or "all loaded kinds have smoke workloads"),
            gate("smoke_workloads_passed", not failed, failed or "all smoke workloads passed"),
        ],
        "workloads": workloads,
        "next_actions": next_actions(workloads, hard_failures),
    }


def workload_for_contract(contract: dict[str, Any]) -> dict[str, Any]:
    kind = str(contract.get("kind") or "")
    if kind == "recurrence_schedule":
        return recurrence_workload(contract)
    if kind == "strided_candidate_fanout":
        return fanout_workload(contract)
    if kind == "cyclic_memory_residue_winding":
        return memory_workload(contract)
    if kind == "multicoil_phase_feature":
        return phase_workload(contract)
    if kind == "circulant_block_cyclic_mixer":
        return mixer_workload(contract)
    if kind == "seed_rule_exact_regeneration":
        return seed_rule_workload(contract)
    return base_workload(contract, "unknown", "circle_unknown_contract_smoke") | {
        "passed": False,
        "metrics": {},
        "ordinary_baselines": list(contract.get("ordinary_baselines") or []),
        "next_private_benchmark": "Add a smoke workload evaluator for this contract kind.",
    }


def recurrence_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    tokens = [row for row in fields.get("tokens", []) if isinstance(row, dict)]
    max_loops = as_int(fields.get("max_loops"))
    active_steps = sum(len(row.get("active_steps") or []) for row in tokens)
    fixed_depth_steps = len(tokens) * max_loops
    exit_matches_required = as_int(fields.get("exit_step")) == as_int(fields.get("required_steps"))
    budgets_in_range = all(1 <= as_int(row.get("budget")) <= as_int(fields.get("loop_period")) for row in tokens)
    passed = bool(contract.get("contract_passed")) and exit_matches_required and budgets_in_range
    return base_workload(contract, "recurrence_schedule", "circle_recurrence_budget_trace_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "token_count": len(tokens),
            "fixed_depth_steps": fixed_depth_steps,
            "circle_active_steps": active_steps,
            "deterministic_step_delta_vs_fixed_depth": fixed_depth_steps - active_steps,
            "exit_matches_required": exit_matches_required,
            "budgets_in_range": budgets_in_range,
            "model_quality_metric": None,
        },
        "ordinary_baselines": [
            {"name": "fixed_depth", "metric": "steps", "value": fixed_depth_steps},
            {"name": "circle_active_schedule", "metric": "steps", "value": active_steps},
            {"name": "no_recurrence", "metric": "phase_trace_available", "value": False},
        ],
        "next_private_benchmark": "Run the same fields beside a private looped-model/work-budget task with quality and runtime metrics.",
    }


def fanout_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    context_length = as_int(fields.get("context_length"))
    predicted_reach = as_int(fields.get("predicted_reach"))
    candidate_budget = as_int(fields.get("candidate_budget"))
    duplicate_count = as_int(fields.get("duplicate_count"))
    coverage_ratio = safe_ratio(predicted_reach, context_length)
    passed = bool(contract.get("contract_passed")) and duplicate_count == 0 and coverage_ratio == 1.0
    return base_workload(contract, "strided_candidate_fanout", "circle_strided_candidate_coverage_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "context_length": context_length,
            "candidate_budget": candidate_budget,
            "predicted_reach": predicted_reach,
            "coverage_ratio": coverage_ratio,
            "duplicate_count": duplicate_count,
            "semantic_coverage_metric": None,
        },
        "ordinary_baselines": [
            {"name": "sequential_fanout", "metric": "coverage_order", "value": "sequential"},
            {"name": "round_robin_fanout", "metric": "coverage_order", "value": "cyclic"},
            {"name": "local_window", "metric": "coverage_scope", "value": "local_only"},
        ],
        "next_private_benchmark": "Attach to a candidate generation workload and report residual quality plus duplicate/rejection rates.",
    }


def memory_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    same_residue_events = list(fields.get("same_residue_events") or [])
    same_residue_windings = list(fields.get("same_residue_windings") or [])
    alias_count = max(0, len(same_residue_events) - 1)
    winding_disambiguates = len(set(same_residue_windings)) == len(same_residue_windings)
    passed = bool(contract.get("contract_passed")) and alias_count > 0 and winding_disambiguates
    return base_workload(contract, "cyclic_memory_residue_winding", "circle_memory_alias_visibility_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "residue_slot": fields.get("residue_slot"),
            "same_residue_event_count": len(same_residue_events),
            "alias_count_exposed": alias_count,
            "winding_disambiguates_same_residue_events": winding_disambiguates,
            "retrieval_quality_metric": None,
        },
        "ordinary_baselines": [
            {"name": "slot_only_memory", "metric": "alias_count_hidden", "value": alias_count},
            {"name": "score_based_retention", "metric": "score_signal_used", "value": False},
            {"name": "fifo_lru", "metric": "residue_winding_available", "value": False},
        ],
        "next_private_benchmark": "Run on context-packet traces and compare retrieval/retention outcomes against FIFO, LRU, and score retention.",
    }


def phase_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    phase_invariant = fields.get("phase_tuple") == fields.get("shifted_phase_tuple")
    relative_invariant = fields.get("relative_phase") == fields.get("shifted_relative_phase")
    passed = bool(contract.get("contract_passed")) and phase_invariant and relative_invariant
    return base_workload(contract, "multicoil_phase_feature", "circle_phase_feature_invariance_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "phase_invariant_after_joint_repeat": phase_invariant,
            "relative_phase_invariant_after_common_shift": relative_invariant,
            "joint_repeat_horizon": fields.get("joint_repeat_horizon"),
            "feature_ablation_quality_metric": None,
        },
        "ordinary_baselines": [
            {"name": "existing_position_bucket", "metric": "phase_tuple_visible", "value": False},
            {"name": "wrong_period", "metric": "negative_control_required", "value": True},
            {"name": "no_phase_feature", "metric": "relative_phase_visible", "value": False},
        ],
        "next_private_benchmark": "Run a private Code LM feature ablation with existing-position, learned-position, wrong-period, and no-phase controls.",
    }


def mixer_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    dense_parameters = as_int(fields.get("dense_parameters"))
    circulant_parameters = as_int(fields.get("circulant_parameters"))
    dense_adapter_parameters = as_int(fields.get("dense_adapter_parameters"))
    block_cyclic_parameters = as_int(fields.get("block_cyclic_parameters"))
    parity_delta = as_float(fields.get("max_abs_dense_delta"))
    passed = bool(contract.get("contract_passed")) and parity_delta == 0.0
    return base_workload(contract, "circulant_block_cyclic_mixer", "circle_mixer_parameter_accounting_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "max_abs_dense_delta": parity_delta,
            "circulant_vs_dense_ratio": safe_ratio(circulant_parameters, dense_parameters),
            "block_vs_dense_adapter_ratio": safe_ratio(block_cyclic_parameters, dense_adapter_parameters),
            "route_quality_metric": None,
            "runtime_metric": None,
        },
        "ordinary_baselines": [
            {"name": "dense_mixer", "metric": "parameters", "value": dense_parameters},
            {"name": "circulant_mixer", "metric": "parameters", "value": circulant_parameters},
            {"name": "lora_adapter", "metric": "parameters", "value": fields.get("lora_parameters")},
            {"name": "block_cyclic_adapter", "metric": "parameters", "value": block_cyclic_parameters},
        ],
        "next_private_benchmark": "Attach to a route/ranker microbench and report quality, runtime, memory, and parameter count separately.",
    }


def seed_rule_workload(contract: dict[str, Any]) -> dict[str, Any]:
    fields = contract.get("fields") if isinstance(contract.get("fields"), dict) else {}
    exact = bool(fields.get("exact_regeneration"))
    generator_shorter = bool(fields.get("generator_shorter"))
    passed = bool(contract.get("contract_passed")) and exact
    return base_workload(contract, "seed_rule_exact_regeneration", "circle_seed_rule_regeneration_smoke") | {
        "passed": passed,
        "metrics": {
            "structural_pass": passed,
            "exact_regeneration": exact,
            "explicit_length": fields.get("explicit_length"),
            "generator_length": fields.get("generator_length"),
            "generator_shorter": generator_shorter,
            "workflow_success_metric": None,
        },
        "ordinary_baselines": [
            {"name": "object_only_storage", "metric": "exact_regeneration_rule_visible", "value": False},
            {"name": "seed_rule_storage", "metric": "exact_regeneration", "value": exact},
            {"name": "unverified_template", "metric": "verification_required", "value": True},
        ],
        "next_private_benchmark": "Use on a repeated workflow/tool-card artifact and measure exact regeneration, edit distance, verification cost, and residual rate.",
    }


def base_workload(contract: dict[str, Any], kind: str, workload_id: str) -> dict[str, Any]:
    return {
        "workload_id": workload_id,
        "contract_id": contract.get("id", ""),
        "kind": kind,
        "workload_scope": "deterministic_private_smoke",
        "model_inference_used": False,
        "training_data_mutated": False,
        "promotion_evidence": False,
        "theorem_ids": list(contract.get("theorem_ids") or []),
        "dictionary_ids": list(contract.get("dictionary_ids") or []),
    }


def next_actions(workloads: list[dict[str, Any]], hard_failures: list[str]) -> list[str]:
    if hard_failures:
        return [
            "Fix failed or unknown smoke workloads before attaching private benchmark tasks.",
            "Rerun the Circle contract exporter and both Theseus Circle consumers.",
        ]
    return [
        "Promote recurrence smoke to a private looped-model/work-budget benchmark with quality and runtime metrics.",
        "Promote fanout smoke to candidate-generation residual testing with duplicate/rejection metrics.",
        "Promote memory smoke to context-packet retention traces with retrieval-quality controls.",
        "Promote phase-feature smoke to Code LM state-sequence ablations with wrong-period and no-phase controls.",
        "Promote mixer smoke to route/ranker microbenchmarks with dense, low-rank, LoRA, and no-mixer baselines.",
        "Promote seed-rule smoke to repeated workflow/tool-card regeneration benchmarks.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Circle AI Private Workload Smoke",
        "",
        f"State: **{report.get('trigger_state')}**",
        "",
        f"- Workloads: {summary.get('workload_count')}",
        f"- Passed workloads: {summary.get('passed_workload_count')}",
        f"- Model-quality metrics present: {summary.get('model_quality_metrics_present')}",
        f"- External inference calls: {report.get('external_inference_calls')}",
        f"- Training mutation: {report.get('training_mutation')}",
        f"- Promotion evidence: {report.get('promotion_evidence')}",
        "",
        "These are deterministic structural smoke workloads only.",
        "",
        "## Workloads",
    ]
    for row in report.get("workloads", []):
        lines.extend(
            [
                "",
                f"### {row.get('workload_id')}",
                "",
                f"- Contract: `{row.get('contract_id')}`",
                f"- Kind: `{row.get('kind')}`",
                f"- Passed: `{row.get('passed')}`",
                f"- Next private benchmark: {row.get('next_private_benchmark')}",
            ]
        )
    lines.extend(["", "## Next Actions"])
    for action in report.get("next_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    pack = {
        "schema_id": "circle_calculus.theseus_hive_ai_contracts.self_test",
        "contracts": [
            {
                "id": "SELF-REC",
                "kind": "recurrence_schedule",
                "contract_passed": True,
                "fields": {
                    "max_loops": 4,
                    "loop_period": 4,
                    "required_steps": 2,
                    "exit_step": 2,
                    "tokens": [{"budget": 1, "active_steps": [1]}, {"budget": 2, "active_steps": [1, 2]}],
                },
            },
            {
                "id": "SELF-FAN",
                "kind": "strided_candidate_fanout",
                "contract_passed": True,
                "fields": {"context_length": 5, "predicted_reach": 5, "candidate_budget": 5, "duplicate_count": 0},
            },
            {
                "id": "SELF-MEM",
                "kind": "cyclic_memory_residue_winding",
                "contract_passed": True,
                "fields": {"same_residue_events": [1, 5], "same_residue_windings": [0, 1]},
            },
            {
                "id": "SELF-PHASE",
                "kind": "multicoil_phase_feature",
                "contract_passed": True,
                "fields": {"phase_tuple": [1], "shifted_phase_tuple": [1], "relative_phase": 0, "shifted_relative_phase": 0},
            },
            {
                "id": "SELF-MIX",
                "kind": "circulant_block_cyclic_mixer",
                "contract_passed": True,
                "fields": {"dense_parameters": 16, "circulant_parameters": 4, "dense_adapter_parameters": 64, "block_cyclic_parameters": 8, "max_abs_dense_delta": 0.0},
            },
            {
                "id": "SELF-SEED",
                "kind": "seed_rule_exact_regeneration",
                "contract_passed": True,
                "fields": {"exact_regeneration": True, "explicit_length": 4, "generator_length": 9, "generator_shorter": False},
            },
        ],
    }
    report = build_report(pack, ROOT / "self-test-contracts.json")
    assert report["external_inference_calls"] == 0
    assert report["training_mutation"] is False
    assert report["promotion_evidence"] is False
    assert report["summary"]["workload_count"] == 6
    assert report["summary"]["passed_workload_count"] == 6
    assert report["summary"]["model_quality_metrics_present"] is False
    print(json.dumps({"self_test": "passed", "workload_count": 6}, indent=2))


def gate(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"gate": name, "ok": bool(ok), "detail": detail}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
