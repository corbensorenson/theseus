#!/usr/bin/env python3
"""Blind information-flow audit for Theseus candidate generation.

This audit exists to catch the failure mode where answer-identifying metadata
reaches generation or ranking and then a report presents the result as learned
capability. It is intentionally conservative: action catalogs and fixed
renderers may be useful tools/baselines, but they are not learned generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    ROOT / "scripts" / "trainable_transformer_hybrid_code_generator_v1.py",
    ROOT / "scripts" / "neural_seed_token_decoder_comparator.py",
    ROOT / "scripts" / "neural_seed_token_decoder_support.py",
    ROOT / "scripts" / "neural_seed_token_decoder_rendering.py",
    ROOT / "scripts" / "neural_seed_visible_source.py",
    ROOT / "scripts" / "neural_seed_static_coherence.py",
    ROOT / "scripts" / "neural_seed_decode_static_guard.py",
    ROOT / "scripts" / "neural_seed_expression_value_guard.py",
    ROOT / "scripts" / "neural_seed_candidate_evidence_summary.py",
    ROOT / "scripts" / "neural_seed_teacher_distillation_rows.py",
    ROOT / "scripts" / "neural_seed_full_state_pretraining.py",
    ROOT / "scripts" / "neural_seed_token_model_backend.py",
    ROOT / "scripts" / "neural_seed_candidate_generation.py",
    ROOT / "scripts" / "neural_seed_report_io.py",
    ROOT / "scripts" / "neural_seed_route_memory.py",
    ROOT / "scripts" / "strict_generator_mlx_decode_eval.py",
    ROOT / "scripts" / "strict_generator_mlx_decode_guards.py",
    ROOT / "scripts" / "strict_generator_mlx_decode_plans.py",
    ROOT / "scripts" / "strict_generator_mlx_source_text.py",
    ROOT / "scripts" / "strict_generator_mlx_decode_reporting.py",
    ROOT / "scripts" / "strict_generator_mlx_specialist_routing.py",
    ROOT / "scripts" / "strict_generator_mlx_private_adaptation.py",
    ROOT / "scripts" / "strict_generator_mlx_adaptation_weights.py",
    ROOT / "scripts" / "strict_generator_mlx_adaptation_selection.py",
    ROOT / "scripts" / "strict_generator_mlx_replay_selection.py",
    ROOT / "scripts" / "strict_generator_mlx_pretraining_probe.py",
    ROOT / "scripts" / "strict_generator_mlx_rung_decode_sweep.py",
    ROOT / "scripts" / "strict_generator_pretraining_spine.py",
]
DEFAULT_CONFIGS = [
    ROOT / "configs" / "neural_seed_token_decoder_comparator.json",
]
DEFAULT_CANDIDATES = [
    ROOT / "reports" / "transformer_hybrid_code_candidates_clean64_v1.jsonl",
    ROOT / "reports" / "neural_seed_token_decoder_candidates_strict_body_tokens.jsonl",
]
DEFAULT_REPORTS = [
    ROOT / "reports" / "transformer_hybrid_code_generator_clean64_v1.json",
    ROOT / "reports" / "neural_seed_token_decoder_comparator_strict_body_tokens.json",
]
DEFAULT_OUT = ROOT / "reports" / "blind_information_flow_audit.json"
DEFAULT_MD = ROOT / "reports" / "blind_information_flow_audit.md"

AUDITED_INFERENCE_FUNCTIONS = {
    "row_to_text",
    "prompt_contract_score",
    "prompt_signature_score",
    "rank_actions",
    "sanitize_task",
}

FORBIDDEN_INFERENCE_FIELDS = {
    "category",
    "source_task_id",
    "solution",
    "solution_expr",
    "solution_body",
    "tests",
    "hidden_tests",
    "expected",
    "expected_output",
    "answer",
    "answers",
    "canonical_solution",
    "return_shape",
    "type_family",
    "required_constructs",
    "action_id",
    "family",
    "benchmark_card",
}

FORBIDDEN_CONFIG_FIELD_PARTS = {
    "category",
    "concept_residual_label",
    "decoder_contract.return_shape",
    "decoder_contract.return_contract.shape",
    "decoder_contract.type_family",
    "decoder_contract.required_constructs",
    "decoder_contract.semantic_family",
    "decoder_contract.residual_label_hint",
    "hidden_tests",
    "solution",
    "solution_body",
    "solution_expr",
    "source_task_id",
    "tests",
}

ANSWER_LABEL_PATTERNS = [
    re.compile(r"category\s*==\s*action\.action_id"),
    re.compile(r"action\.action_id\s*==\s*category"),
    re.compile(r"task\.get\([\"']category[\"']\)"),
]

ACTION_SELECTOR_TOKENS = (
    "action_selector",
    "action_generator",
    "fixed_renderer",
    "action_renderer",
    "grammar_safe_action_renderer",
    "action=",
)

LEARNED_OVERCLAIM_FIELDS = (
    "benchmark_promotion_eligible",
    "token_level_code_generation_learned",
    "full_body_token_candidate",
    "grammar_masked_learned_token_candidate",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[], help="Source file to statically audit. May repeat.")
    parser.add_argument("--config", action="append", default=[], help="Comparator/generator config JSON to audit. May repeat.")
    parser.add_argument("--candidates", action="append", default=[], help="Candidate JSONL manifest to audit. May repeat.")
    parser.add_argument("--report", action="append", default=[], help="Report JSON to audit. May repeat.")
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MD))
    args = parser.parse_args()

    sources = [resolve(path) for path in args.source] if args.source else DEFAULT_SOURCES
    configs = [resolve(path) for path in args.config] if args.config else DEFAULT_CONFIGS
    candidates = [resolve(path) for path in args.candidates] if args.candidates else DEFAULT_CANDIDATES
    reports = [resolve(path) for path in args.report] if args.report else discover_default_reports()

    source_results = [audit_source(path) for path in sources]
    config_results = [audit_config(path) for path in configs]
    candidate_results = [audit_candidates(path) for path in candidates]
    report_results = [audit_report(path) for path in reports]

    static_violations = sum(item["violation_count"] for item in source_results)
    config_violations = sum(item["violation_count"] for item in config_results)
    candidate_overclaims = sum(item["overclaim_count"] for item in candidate_results)
    report_overclaims = sum(item["overclaim_count"] for item in report_results)
    invalid_claim_count = static_violations + config_violations + candidate_overclaims + report_overclaims
    missing_inputs = sum(1 for item in [*source_results, *config_results, *candidate_results, *report_results] if item.get("missing"))

    trigger_state = "GREEN" if invalid_claim_count == 0 and missing_inputs == 0 else "RED"
    report = {
        "policy": "project_theseus_blind_information_flow_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            "source_file_count": len(source_results),
            "config_file_count": len(config_results),
            "candidate_manifest_count": len(candidate_results),
            "report_count": len(report_results),
            "static_information_flow_violation_count": static_violations,
            "config_information_flow_violation_count": config_violations,
            "candidate_overclaim_count": candidate_overclaims,
            "report_overclaim_count": report_overclaims,
            "invalid_claim_count": invalid_claim_count,
            "missing_input_count": missing_inputs,
        },
        "rules": {
            "allowed_inference_input": "natural-language prompt plus function signature only",
            "forbidden_inference_fields": sorted(FORBIDDEN_INFERENCE_FIELDS),
            "action_selector_boundary": "fixed action catalogs/renderers are baselines or tools, never learned code generation",
            "self_declared_integrity_sufficient": False,
        },
        "source_results": source_results,
        "config_results": config_results,
        "candidate_results": candidate_results,
        "report_results": report_results,
        "external_inference_calls": 0,
    }
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if trigger_state == "GREEN" else 2


def audit_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "missing": True, "violation_count": 1, "violations": ["missing_source_file"]}
    text = path.read_text(encoding="utf-8")
    violations: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "path": rel(path),
            "missing": False,
            "violation_count": 1,
            "violations": [{"kind": "source_syntax_error", "line": exc.lineno, "detail": str(exc)}],
        }

    function_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> Any:
            current = function_stack[-1] if function_stack else ""
            if current in AUDITED_INFERENCE_FUNCTIONS and is_get_call(node):
                key = get_call_key(node)
                if key in FORBIDDEN_INFERENCE_FIELDS:
                    violations.append(
                        {
                            "kind": "forbidden_field_in_inference_path",
                            "function": current,
                            "field": key,
                            "line": getattr(node, "lineno", None),
                        }
                    )
            self.generic_visit(node)

    Visitor().visit(tree)

    for pattern in ANSWER_LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                {
                    "kind": "answer_label_scoring_or_text_pattern",
                    "pattern": pattern.pattern,
                    "line": line_for_offset(text, match.start()),
                }
            )

    return {
        "path": rel(path),
        "missing": False,
        "violation_count": len(violations),
        "violations": violations,
    }


def audit_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "missing": True, "violation_count": 1, "violations": ["missing_config_file"]}
    payload = read_json(path)
    violations: list[dict[str, Any]] = []
    if not payload:
        return {"path": rel(path), "missing": False, "violation_count": 1, "violations": ["invalid_or_empty_config"]}

    text_views = payload.get("text_views") if isinstance(payload.get("text_views"), dict) else {}
    for view_name, fields in text_views.items():
        if str(view_name) in {"withheld_from_text", "withheld", "denylist"}:
            continue
        if not isinstance(fields, list):
            continue
        for field in fields:
            field_text = str(field)
            if config_field_forbidden(field_text):
                violations.append(
                    {
                        "kind": "forbidden_field_in_text_view",
                        "view": str(view_name),
                        "field": field_text,
                    }
                )

    structure_cfg = payload.get("body_structure_decoder") if isinstance(payload.get("body_structure_decoder"), dict) else {}
    beam_cfg = structure_cfg.get("visible_contract_semantic_beam") if isinstance(structure_cfg.get("visible_contract_semantic_beam"), dict) else {}
    if beam_cfg.get("enabled"):
        fields = beam_cfg.get("fields") if isinstance(beam_cfg.get("fields"), list) else []
        forbidden_fields = [str(field) for field in fields if config_field_forbidden(str(field))]
        violations.append(
            {
                "kind": "visible_contract_semantic_beam_enabled",
                "detail": "contract-derived semantic beams are diagnostic only and cannot be on in the default promotion path",
                "forbidden_fields": forbidden_fields,
            }
        )

    routing_cfg = structure_cfg.get("internal_semantic_routing") if isinstance(structure_cfg.get("internal_semantic_routing"), dict) else {}
    if routing_cfg.get("enabled"):
        violations.append(
            {
                "kind": "internal_semantic_routing_enabled",
                "detail": "semantic route memories use private solution-derived plans and must stay diagnostic outside a separate non-promotion ablation",
            }
        )

    structural_cfg = structure_cfg.get("structural_action_family") if isinstance(structure_cfg.get("structural_action_family"), dict) else {}
    if structural_cfg.get("enabled") and not structural_cfg.get("diagnostic_only"):
        violations.append(
            {
                "kind": "structural_action_family_enabled_without_diagnostic_boundary",
                "detail": "line-action renderers are adapters/baselines, not free learned token generation",
            }
        )

    data_cfg = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("train_jsonl", "eval_jsonl"):
        value = str(data_cfg.get(key) or "")
        if "private_contract_blind_transfer" in value:
            violations.append(
                {
                    "kind": "contract_blind_split_in_default_generator_config",
                    "field": key,
                    "value": value,
                    "detail": "contract-blind rows hide semantics in decoder_contract; use natural-prompt private/licensed rows for promotion-grade generation",
                }
            )

    return {
        "path": rel(path),
        "missing": False,
        "violation_count": len(violations),
        "violations": violations,
    }


def config_field_forbidden(field_text: str) -> bool:
    normalized = str(field_text).strip()
    if normalized in FORBIDDEN_CONFIG_FIELD_PARTS:
        return True
    return any(part in normalized for part in FORBIDDEN_CONFIG_FIELD_PARTS if "." in part)


def audit_candidates(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "missing": True, "overclaim_count": 1, "samples": ["missing_candidate_manifest"]}
    rows = read_jsonl(path)
    counts: Counter[str] = Counter()
    samples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = candidate_text(row)
        if not is_action_selector_text(text):
            continue
        for field in LEARNED_OVERCLAIM_FIELDS:
            if truthy(row.get(field)):
                counts[field] += 1
                if len(samples) < 20:
                    samples.append(
                        {
                            "index": index,
                            "field": field,
                            "candidate_generation_mode": row.get("candidate_generation_mode"),
                            "origin": str(row.get("origin") or "")[:180],
                        }
                    )
    return {
        "path": rel(path),
        "missing": False,
        "row_count": len(rows),
        "overclaim_count": sum(counts.values()),
        "overclaim_counts": dict(sorted(counts.items())),
        "samples": samples,
    }


def audit_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "missing": True, "overclaim_count": 1, "violations": ["missing_report"]}
    payload = read_json(path)
    text = json.dumps(payload, sort_keys=True)
    lower = text.lower()
    violations = []
    stale_clean64_scope = any(
        token in str(path).replace("\\", "/")
        for token in (
            "transformer_hybrid_clean64",
            "canonical_transformer_hybrid_clean64",
            "canonical_transformer_hybrid_survival_lane_ablation",
        )
    )
    action_selector_scope = is_action_selector_text(lower) or stale_clean64_scope
    if action_selector_scope:
        if truthy(find_key(payload, "token_level_code_generation_learned")):
            violations.append("action_selector_report_claims_token_level_code_generation_learned")
        if truthy(find_key(payload, "benchmark_promotion_eligible")):
            violations.append("action_selector_report_claims_benchmark_promotion_eligible")
        if "selected functional transfer" in lower and "64/64" in lower:
            violations.append("action_selector_report_claims_perfect_transfer")
        if "functional promotion" in lower and "64/64" in lower:
            violations.append("action_selector_report_claims_perfect_functional_promotion")
        if "64/64" in lower and not truthy(find_key(payload, "not_learned_code_generation")):
            violations.append("stale_clean64_report_missing_invalid_or_non_learned_boundary")
        rules = payload.get("rules") if isinstance(payload.get("rules"), dict) else {}
        if rules and rules.get("not_learned_code_generation") is not True:
            violations.append("action_selector_report_missing_not_learned_generation_rule")
    return {
        "path": rel(path),
        "missing": False,
        "policy": payload.get("policy") if isinstance(payload, dict) else None,
        "trigger_state": payload.get("trigger_state") if isinstance(payload, dict) else None,
        "overclaim_count": len(violations),
        "violations": violations,
    }


def discover_default_reports() -> list[Path]:
    reports = list(DEFAULT_REPORTS)
    stale_patterns = [
        "private_heldout_transfer_baseline_v1_canonical_transformer_hybrid_clean64_v1.json",
        "private_candidate_replay_contract_audit_canonical_transformer_hybrid_clean64_v1.json",
        "canonical_transformer_hybrid_survival_lane_ablation_v1.json",
    ]
    for pattern in stale_patterns:
        candidate = ROOT / "reports" / pattern
        if candidate.exists():
            reports.append(candidate)
    return reports


def is_get_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Attribute) and node.func.attr == "get" and bool(node.args)


def get_call_key(node: ast.Call) -> str:
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return ""


def candidate_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("candidate_generation_mode"),
        row.get("candidate_generation_contract"),
        row.get("candidate_source"),
        row.get("origin"),
        row.get("candidate_program_scope"),
    ]
    for key in ("transformer_hybrid_v1", "neural_action_selector_v1"):
        value = row.get(key)
        if isinstance(value, dict):
            parts.append(json.dumps(value, sort_keys=True))
    return " ".join(str(part or "") for part in parts).lower()


def is_action_selector_text(text: str) -> bool:
    lower = str(text).lower()
    return any(token in lower for token in ACTION_SELECTOR_TOKENS)


def find_key(obj: Any, target: str) -> Any:
    if isinstance(obj, dict):
        if target in obj:
            return obj[target]
        for value in obj.values():
            found = find_key(value, target)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, target)
            if found is not None:
                return found
    return None


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Blind Information-Flow Audit",
        "",
        f"- State: `{report.get('trigger_state')}`",
        f"- Invalid claim count: `{summary.get('invalid_claim_count')}`",
        f"- Static information-flow violations: `{summary.get('static_information_flow_violation_count')}`",
        f"- Config information-flow violations: `{summary.get('config_information_flow_violation_count')}`",
        f"- Candidate overclaims: `{summary.get('candidate_overclaim_count')}`",
        f"- Report overclaims: `{summary.get('report_overclaim_count')}`",
        "",
        "## Rule",
        "",
        "Generation and ranking may see only the natural-language prompt plus the function signature. "
        "Fixed action catalogs/renderers are baselines or tools, never learned code generation.",
        "",
    ]
    for section in ("source_results", "config_results", "candidate_results", "report_results"):
        lines.extend(["", f"## {section}", ""])
        for item in report.get(section, []):
            count = item.get("violation_count", item.get("overclaim_count", 0))
            lines.append(f"- `{item.get('path')}` count=`{count}`")
            details = item.get("violations") or item.get("samples") or []
            for detail in details[:5]:
                lines.append(f"  - `{detail}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
