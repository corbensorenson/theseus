#!/usr/bin/env python3
"""Ablation summary for the integrated structural-action decoder family.

This report is intentionally evidence-only. It does not train, fetch data,
call a teacher, run public calibration, or promote a model. It reads the main
private token-decoder comparator report and separates the structural-action
candidate family into auditable axes:

1. sequence-class selection;
2. line-action compilation;
3. finer AST synthesis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTEGRATED = ROOT / "reports" / "neural_seed_token_decoder_structural_integrated_smoke.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "neural_seed_token_decoder_structural_integrated_smoke_candidates.jsonl"
DEFAULT_STANDALONE = ROOT / "reports" / "neural_seed_structural_action_decoder_96eval_multiseed.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_structural_action_ablation_report.json"
DEFAULT_MD = ROOT / "reports" / "neural_seed_structural_action_ablation_report.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated-report", default=str(DEFAULT_INTEGRATED.relative_to(ROOT)))
    parser.add_argument("--candidate-manifest", default=str(DEFAULT_CANDIDATES.relative_to(ROOT)))
    parser.add_argument("--standalone-report", default=str(DEFAULT_STANDALONE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MD.relative_to(ROOT)))
    args = parser.parse_args()

    integrated = read_json(resolve(args.integrated_report))
    candidates = read_jsonl(resolve(args.candidate_manifest))
    standalone = read_json(resolve(args.standalone_report)) if resolve(args.standalone_report).exists() else {}
    report = build_report(args, integrated, candidates, standalone)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(
    args: argparse.Namespace,
    integrated: dict[str, Any],
    candidates: list[dict[str, Any]],
    standalone: dict[str, Any],
) -> dict[str, Any]:
    structural_rows = [
        row for row in candidates if row.get("candidate_generation_mode") == "private_train_structural_action_sequence_decoder"
    ]
    token_rows = [row for row in candidates if row.get("candidate_generation_mode") == "token_level_code_decoder"]
    modes = Counter(str(row.get("candidate_generation_mode") or "") for row in candidates if row.get("phase") != "private_baseline")
    arms = {
        arm_id: arm_axis_summary(integrated, arm_id)
        for arm_id in ["symliquid_style", "transformer_control"]
    }
    structural_family = dict_or_empty(get_path(integrated, ["body_structure_decoder", "structural_action_family"], {}))
    sequence_axis = {
        "axis": "sequence_class_selection",
        "implemented": bool(structural_family.get("active")),
        "target": structural_family.get("target_source"),
        "source_view": structural_family.get("source_view"),
        "class_count": structural_family.get("structural_action_class_count"),
        "token_count": structural_family.get("structural_action_token_count"),
        "fanout_top_k": structural_family.get("fanout_top_k"),
        "matched_for_both_arms": True,
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
        "evidence_by_arm": {
            arm_id: {
                "substrate": get_path(integrated, ["arms", arm_id, "views", "structural_action", "substrate"], ""),
                "parameter_count": get_path(integrated, ["arms", arm_id, "views", "structural_action", "parameter_count"], 0),
                "loss_curve": get_path(integrated, ["arms", arm_id, "views", "structural_action", "train", "loss_curve"], []),
                "candidate_rows": get_path(integrated, ["arms", arm_id, "views", "structural_action", "candidate_rows"], 0),
            }
            for arm_id in arms
        },
    }
    compiler_axis = {
        "axis": "line_action_compilation",
        "implemented": bool(structural_family.get("active")),
        "compiler": structural_family.get("compiler"),
        "structural_rows": len(structural_rows),
        "unique_structural_sequences": len({str(row.get("structural_sequence_id") or "") for row in structural_rows}),
        "syntax_pass_rate_min": min(
            [float(arms[arm]["structural_action_syntax_pass_rate"] or 0.0) for arm in arms] or [0.0]
        ),
        "fallback_return_rate_max": max(
            [float(arms[arm]["structural_action_fallback_rate"] or 0.0) for arm in arms] or [0.0]
        ),
        "strategy_counts": dict(
            Counter(str(get_path(row, ["grammar_repair", "strategy"], "")) for row in structural_rows).most_common(8)
        ),
        "uses_eval_tests_or_solutions": False,
        "uses_public_data": False,
        "teacher_used": False,
    }
    ast_axis = {
        "axis": "finer_ast_synthesis",
        "implemented": False,
        "status": "separated_but_not_yet_implemented",
        "reason": (
            "The integrated family currently selects complete private-train line-action sequences and compiles them. "
            "It does not yet synthesize finer AST edits or novel action compositions."
        ),
        "next_evidence_needed": [
            "Predict action fragments independently of full train-body sequence classes.",
            "Compile fragments through an AST validator instead of replaying complete line sequences.",
            "Report verifier pass, syntax pass, fallback rate, and residual deltas against the same private split.",
        ],
    }
    gates = [
        gate("integrated_comparator_green", integrated.get("trigger_state") == "GREEN", integrated.get("trigger_state"), "hard"),
        gate("structural_family_active", bool(structural_family.get("active")), structural_family, "hard"),
        gate("structural_rows_present", len(structural_rows) > 0, {"structural_rows": len(structural_rows)}, "hard"),
        gate("token_rows_still_present", len(token_rows) > 0, {"token_rows": len(token_rows)}, "hard"),
        gate("fallback_return_rate_zero", float(compiler_axis["fallback_return_rate_max"]) == 0.0, compiler_axis, "hard"),
        gate("compiler_syntax_validity_nonzero", float(compiler_axis["syntax_pass_rate_min"]) > 0.0, compiler_axis, "hard"),
        gate("no_public_or_teacher_use", no_public_or_teacher_use(integrated, candidates), 0, "hard"),
        gate("finer_ast_axis_separated", ast_axis["status"] == "separated_but_not_yet_implemented", ast_axis, "soft"),
    ]
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger_state = "GREEN" if hard_pass else "RED"
    if trigger_state == "GREEN" and any(not row["passed"] for row in gates):
        trigger_state = "YELLOW"
    return {
        "policy": "project_theseus_neural_seed_structural_action_ablation_report_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "inputs": {
            "integrated_report": args.integrated_report,
            "candidate_manifest": args.candidate_manifest,
            "standalone_report": args.standalone_report if standalone else "",
        },
        "summary": {
            "integrated_trigger_state": integrated.get("trigger_state"),
            "train_rows": get_path(integrated, ["summary", "train_rows"], 0),
            "eval_rows": get_path(integrated, ["summary", "eval_rows"], 0),
            "generated_mode_counts": dict(sorted(modes.items())),
            "structural_rows": len(structural_rows),
            "token_rows": len(token_rows),
            "symliquid_sts_on_pass_rate": get_path(integrated, ["arms", "symliquid_style", "summary", "sts_on_verifier_pass_rate"], 0.0),
            "transformer_sts_on_pass_rate": get_path(integrated, ["arms", "transformer_control", "summary", "sts_on_verifier_pass_rate"], 0.0),
            "standalone_96eval_multiseed_available": bool(standalone),
            "standalone_96eval_summary": get_path(standalone, ["summary"], {}),
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "axes": {
            "sequence_class_selection": sequence_axis,
            "line_action_compilation": compiler_axis,
            "finer_ast_synthesis": ast_axis,
        },
        "arms": arms,
        "gates": gates,
        "score_semantics": (
            "Ablation report only. It separates selection, compilation, and finer-AST axes from already-generated "
            "private comparator evidence. Verifier gain is not attributed to finer AST synthesis because that axis is "
            "not implemented. No public calibration, public training rows, teacher call, external inference, or "
            "promotion occurred."
        ),
        "external_inference_calls": 0,
    }


def arm_axis_summary(report: dict[str, Any], arm_id: str) -> dict[str, Any]:
    summary = dict_or_empty(get_path(report, ["arms", arm_id, "summary"], {}))
    structural = dict_or_empty(get_path(report, ["arms", arm_id, "views", "structural_action"], {}))
    return {
        "sts_on_verifier_pass_rate": summary.get("sts_on_verifier_pass_rate"),
        "sts_off_verifier_pass_rate": summary.get("sts_off_verifier_pass_rate"),
        "structural_action_candidate_rows": summary.get("structural_action_candidate_rows"),
        "structural_action_syntax_pass_rate": summary.get("structural_action_syntax_pass_rate"),
        "structural_action_fallback_rate": summary.get("structural_action_fallback_rate"),
        "sequence_selector_substrate": structural.get("substrate"),
        "sequence_selector_train_loss_curve": get_path(structural, ["train", "loss_curve"], []),
        "line_action_compiler": get_path(structural, ["decoder_constraints", "compiler"], ""),
        "finer_ast_synthesis_enabled": bool(get_path(structural, ["decoder_constraints", "finer_ast_synthesis_enabled"], False)),
    }


def no_public_or_teacher_use(report: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    if int(report.get("external_inference_calls") or 0) != 0:
        return False
    if bool(get_path(report, ["summary", "teacher_used"], False)):
        return False
    if int(get_path(report, ["summary", "public_training_rows"], 0) or 0) != 0:
        return False
    for row in candidates:
        if int(row.get("external_inference_calls") or 0) != 0:
            return False
        if bool(row.get("public_tests_visible_to_generator") or row.get("public_solutions_visible_to_generator")):
            return False
        if bool(row.get("eval_tests_visible_to_generator") or row.get("eval_solution_visible_to_generator")):
            return False
    return True


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    axes = dict_or_empty(report.get("axes"))
    compiler = dict_or_empty(axes.get("line_action_compilation"))
    ast_axis = dict_or_empty(axes.get("finer_ast_synthesis"))
    lines = [
        "# Structural Action Ablation Report",
        "",
        f"- Trigger state: {report.get('trigger_state')}",
        f"- Integrated comparator: {summary.get('integrated_trigger_state')}",
        f"- Rows: train={summary.get('train_rows')} eval={summary.get('eval_rows')}",
        f"- Generated modes: {json.dumps(summary.get('generated_mode_counts', {}), sort_keys=True)}",
        f"- SymLiquid pass rate: {summary.get('symliquid_sts_on_pass_rate')}",
        f"- Transformer pass rate: {summary.get('transformer_sts_on_pass_rate')}",
        "",
        "## Axes",
        "",
        f"- Sequence-class selection: {bool(get_path(axes, ['sequence_class_selection', 'implemented'], False))}",
        f"- Line-action compilation: syntax_min={compiler.get('syntax_pass_rate_min')} fallback_max={compiler.get('fallback_return_rate_max')}",
        f"- Finer AST synthesis: {ast_axis.get('status')}",
        "",
        "## Gates",
    ]
    for row in report.get("gates", []):
        lines.append(f"- {row['name']}: {'PASS' if row['passed'] else 'FAIL'} ({row['severity']})")
    lines.extend(["", report.get("score_semantics", "")])
    return "\n".join(lines) + "\n"


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(data: Any, path: list[Any], default: Any = None) -> Any:
    cur = data
    for item in path:
        if isinstance(cur, dict):
            cur = cur.get(item, default)
        elif isinstance(cur, list) and isinstance(item, int) and 0 <= item < len(cur):
            cur = cur[item]
        else:
            return default
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
