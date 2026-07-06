#!/usr/bin/env python3
"""Audit benchmark saturation, slice thinness, and next safe pressure."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "theseus_saturation_generalization_audit.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "theseus_saturation_generalization_audit.md"
TOKEN_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
PRIVATE_EVAL = ROOT / "data" / "training_data" / "high_transfer" / "private_eval" / "private_contract_blind_transfer_v1_code_lm_tasks.jsonl"
PRIVATE_TRAIN = ROOT / "data" / "training_data" / "high_transfer" / "private_train" / "private_contract_blind_transfer_v1_train_reference_code_lm_tasks.jsonl"
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--large-token-report", default="reports/neural_seed_token_decoder_96eval_4096train_multiseed.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = build_report(args, started)
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def build_report(args: argparse.Namespace, started: float) -> dict[str, Any]:
    token_config = read_json(TOKEN_CONFIG)
    token_ablation = read_json(REPORTS / "neural_seed_token_decoder_route_independence_ablation.json")
    token_full = read_json(REPORTS / "neural_seed_token_decoder_ablation_full_learned_beam_off.json")
    proposer = read_json(REPORTS / "neural_seed_code_proposer_comparator.json")
    residual = read_json(REPORTS / "neural_seed_token_decoder_residual_context_miner.json")
    plan_semantic = read_json(REPORTS / "theseus_plan_semantic_residual_miner.json")
    governor = read_json(REPORTS / "theseus_generalization_governor_v1.json")
    large_token = read_json(resolve(args.large_token_report))

    private_eval_rows = count_jsonl(PRIVATE_EVAL)
    private_train_rows = count_jsonl(PRIVATE_TRAIN)
    current_eval_rows = int(get_path(token_config, ["data", "max_eval_rows"], 0) or 0)
    proposed_eval_rows = min(96, private_eval_rows)
    public_pass_rate = first_number_for_key(governor, "public_pass_rate") or first_number_for_key(governor, "student_first_public_pass_rate")
    public_task_count = first_number_for_key(governor, "public_task_count") or first_number_for_key(governor, "task_count")

    surfaces = [
        surface_row(
            "token_decoder_current_route_ablation",
            token_ablation.get("trigger_state"),
            current_eval_rows,
            int(len(token_ablation.get("seeds") or []) or get_path(token_full, ["summary", "requested_seed_count"], 0) or 0),
            get_path(token_ablation, ["attribution", "symliquid_no_visible_text_memory_mean"]),
            get_path(token_ablation, ["attribution", "transformer_no_visible_text_memory_mean"]),
            "thin_private_slice",
        ),
        surface_row(
            "code_proposer_96_private",
            proposer.get("trigger_state"),
            get_path(proposer, ["summary", "eval_rows"]),
            1,
            None,
            None,
            "larger_private_slice_but_single_comparator",
        ),
        surface_row(
            "token_decoder_96_private_multiseed",
            large_token.get("trigger_state") if large_token else "MISSING",
            get_path(large_token, ["seed_rows", 0, "eval_rows"]) or proposed_eval_rows if large_token else proposed_eval_rows,
            get_path(large_token, ["summary", "requested_seed_count"]) or 0,
            get_path(large_token, ["summary", "symliquid_sts_on_mean"]),
            get_path(large_token, ["summary", "transformer_sts_on_mean"]),
            "preferred_current_overnight_gate",
        ),
        surface_row(
            "spent_public_5x32",
            "LOCKED",
            public_task_count,
            1,
            public_pass_rate,
            None,
            "public_calibration_spent_do_not_rerun",
        ),
    ]
    flags = []
    if current_eval_rows and private_eval_rows and current_eval_rows / private_eval_rows <= 0.15:
        flags.append("current_token_decoder_eval_slice_is_thin")
    if public_pass_rate is not None and float(public_pass_rate) < 0.70:
        flags.append("public_transfer_floor_not_cleared")
    residual_counts = get_path(residual, ["summary", "bucket_counts"], {}) or {}
    rejected_probe = dict_or_empty(plan_semantic.get("rejected_renderer_template_probe"))
    if int(residual_counts.get("full_route_both_fail", 0) or 0) > 0:
        flags.append("full_route_both_fail_residuals_remain")
    if int(plan_semantic.get("seed_arm_both_fail_event_count") or 0) > 0:
        flags.append("shared_generic_semantic_body_residuals_remain")
    if rejected_probe.get("admissibility") == "rejected_as_capability_evidence":
        flags.append("contract_renderer_template_probe_rejected")
    if large_token:
        large_delta = subtract(get_path(large_token, ["summary", "symliquid_sts_on_mean"]), get_path(large_token, ["summary", "transformer_sts_on_mean"]))
    else:
        large_delta = None
    assistant_residuals = summarize_large_token_residuals(large_token)
    if int(get_path(assistant_residuals, ["gap_counts_total", "both_fail"], 0) or 0) > 0:
        flags.append("large_private_gate_both_fail_residuals_remain")

    recommendation = "run_or_complete_96_row_private_matched_gate"
    if large_token and large_token.get("trigger_state") in {"GREEN", "YELLOW"}:
        recommendation = "use_96_row_private_gate_for_survival_decision"
    if large_delta is not None and large_delta < 0:
        recommendation = "transformer_first_survival_path"
    elif large_delta is not None and large_delta > 0:
        recommendation = "investigate_symliquid_complementarity_before_switching"
    elif large_delta == 0 and int(get_path(assistant_residuals, ["gap_counts_total", "both_fail"], 0) or 0) > 0:
        recommendation = "attack_shared_plan_semantic_residuals_before_more_benchmark_slices"
    if large_token:
        next_safe_commands = [
            "improve learned/ranked semantic-slot generation for shared private residual families",
            "rerun the same 96-row multi-seed gate after a bounded decoder/planner change",
            "keep executable contract-family renderer templates rejected as capability evidence",
            "keep public calibration locked until explicit operator unlock",
        ]
    else:
        next_safe_commands = [
            "run 96-row private multi-seed token decoder gate",
            "mine residuals from the larger private gate",
            "keep public calibration locked until explicit operator unlock",
        ]

    gates = [
        gate("private_eval_rows_available", private_eval_rows >= proposed_eval_rows, {"available": private_eval_rows, "target": proposed_eval_rows}, "hard"),
        gate("public_calibration_locked", True, {"public_pass_rate": public_pass_rate, "public_task_count": public_task_count}, "hard"),
        gate("no_external_inference", True, 0, "hard"),
        gate("no_teacher_calls", True, False, "hard"),
        gate("no_model_promotion", True, False, "hard"),
        gate("plan_semantic_residual_miner_loaded_green", plan_semantic.get("trigger_state") == "GREEN", plan_semantic.get("trigger_state"), "hard"),
        gate(
            "renderer_template_probe_rejected_as_capability_evidence",
            not rejected_probe or rejected_probe.get("admissibility") == "rejected_as_capability_evidence",
            rejected_probe.get("admissibility") if rejected_probe else "not_present",
            "hard",
        ),
        gate("saturation_flags_recorded", bool(flags), flags, "hard"),
    ]
    hard_failed = [row for row in gates if row["severity"] == "hard" and not row["passed"]]
    return {
        "policy": "project_theseus_saturation_generalization_audit_v0",
        "created_utc": now(),
        "trigger_state": "GREEN" if not hard_failed else "RED",
        "private_eval_rows_available": private_eval_rows,
        "private_train_rows_available": private_train_rows,
        "current_token_decoder_max_eval_rows": current_eval_rows,
        "recommended_larger_private_eval_rows": proposed_eval_rows,
        "surfaces": surfaces,
        "saturation_flags": flags,
        "residual_bucket_counts": residual_counts,
        "assistant_residuals": assistant_residuals,
        "plan_semantic_residuals": {
            "trigger_state": plan_semantic.get("trigger_state"),
            "unique_both_fail_task_count": plan_semantic.get("unique_both_fail_task_count"),
            "seed_arm_both_fail_event_count": plan_semantic.get("seed_arm_both_fail_event_count"),
            "candidate_coverage_counts": plan_semantic.get("candidate_coverage_counts"),
            "rejected_renderer_template_probe": rejected_probe,
        },
        "recommendation": recommendation,
        "next_safe_commands": next_safe_commands,
        "gates": gates,
        "score_semantics": (
            "Saturation/generalization audit over local reports and existing private held-out rows. "
            "No public calibration, teacher call, external inference, training, distillation, promotion, "
            "or benchmark answer use occurs in this audit."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def surface_row(name: str, state: Any, eval_rows: Any, seeds: Any, sym_score: Any, tx_score: Any, risk: str) -> dict[str, Any]:
    eval_int = int(eval_rows or 0)
    seed_int = int(seeds or 0)
    score_values = [float(x) for x in [sym_score, tx_score] if x is not None]
    saturated = bool(score_values and all(x >= 0.95 for x in score_values))
    thin = bool(eval_int and eval_int < 96)
    return {
        "surface": name,
        "state": state,
        "eval_rows": eval_int,
        "seed_count": seed_int,
        "symliquid_score": sym_score,
        "transformer_score": tx_score,
        "thin_slice": thin,
        "saturated": saturated,
        "risk": risk,
    }


def summarize_large_token_residuals(large_token: dict[str, Any]) -> dict[str, Any]:
    if not large_token:
        return {}
    gap_counts: Counter[str] = Counter()
    wrong_by_arm: dict[str, Counter[str]] = {
        "symliquid_style": Counter(),
        "transformer_control": Counter(),
    }
    assistant_families: dict[str, Counter[str]] = {
        "symliquid_style": Counter(),
        "transformer_control": Counter(),
    }
    audit_paths = []
    for row in large_token.get("seed_rows", []) or []:
        if not isinstance(row, dict):
            continue
        audit_path = row.get("semantic_plan_audit")
        if not audit_path:
            continue
        audit_paths.append(str(audit_path))
        audit = read_json(resolve(str(audit_path)))
        gap_counts.update(dict_or_empty(get_path(audit, ["summary", "gap_counts"], {})))
        for arm in ["symliquid_style", "transformer_control"]:
            wrong_counts = dict_or_empty(get_path(audit, ["by_arm", arm, "by_phase", "private_eval", "wrong_answer_shape_counts"], {}))
            wrong_by_arm[arm].update(wrong_counts)
            for key, count in wrong_counts.items():
                if key == "passed":
                    continue
                assistant_families[arm][assistant_family_from_wrong_shape(str(key))] += int(count or 0)
    return {
        "seed_count_with_audits": len(audit_paths),
        "audit_paths": audit_paths,
        "gap_counts_total": dict(sorted(gap_counts.items())),
        "wrong_answer_shapes_top": {
            arm: top_counter(counter, 16)
            for arm, counter in wrong_by_arm.items()
        },
        "assistant_family_counts_top": {
            arm: top_counter(counter, 16)
            for arm, counter in assistant_families.items()
        },
    }


def assistant_family_from_wrong_shape(value: str) -> str:
    if "->" in value:
        return value.rsplit("->", 1)[-1]
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value


def top_counter(counter: Counter[str], limit: int) -> dict[str, int]:
    return {key: int(count) for key, count in counter.most_common(limit)}


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def get_path(value: Any, path: list[Any], default: Any = None) -> Any:
    cur = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return default
            cur = cur[part]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def first_number_for_key(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        if key in value:
            found = number(value.get(key))
            if found is not None:
                return found
        for child in value.values():
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_number_for_key(child, key)
            if found is not None:
                return found
    return None


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def subtract(left: Any, right: Any) -> float | None:
    lval = number(left)
    rval = number(right)
    if lval is None or rval is None:
        return None
    return round(lval - rval, 6)


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Theseus Saturation Generalization Audit",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- private_eval_rows_available: `{report.get('private_eval_rows_available')}`",
        f"- current_token_decoder_max_eval_rows: `{report.get('current_token_decoder_max_eval_rows')}`",
        f"- recommended_larger_private_eval_rows: `{report.get('recommended_larger_private_eval_rows')}`",
        f"- recommendation: `{report.get('recommendation')}`",
        "",
        "## Saturation Flags",
        "",
    ]
    for flag in report.get("saturation_flags", []):
        lines.append(f"- `{flag}`")
    lines.extend(["", "## Surfaces", ""])
    for row in report.get("surfaces", []):
        lines.append(
            f"- `{row.get('surface')}`: state=`{row.get('state')}`, eval_rows=`{row.get('eval_rows')}`, "
            f"seeds=`{row.get('seed_count')}`, sym=`{row.get('symliquid_score')}`, tx=`{row.get('transformer_score')}`, "
            f"thin=`{row.get('thin_slice')}`, saturated=`{row.get('saturated')}`, risk=`{row.get('risk')}`"
        )
    plan_semantic = dict_or_empty(report.get("plan_semantic_residuals"))
    if plan_semantic:
        rejected_probe = dict_or_empty(plan_semantic.get("rejected_renderer_template_probe"))
        lines.extend(["", "## Plan-Semantic Residuals", ""])
        lines.append(f"- trigger_state: `{plan_semantic.get('trigger_state')}`")
        lines.append(f"- unique_both_fail_task_count: `{plan_semantic.get('unique_both_fail_task_count')}`")
        lines.append(f"- seed_arm_both_fail_event_count: `{plan_semantic.get('seed_arm_both_fail_event_count')}`")
        lines.append(f"- candidate_coverage_counts: `{plan_semantic.get('candidate_coverage_counts')}`")
        lines.append(f"- renderer_template_probe_admissibility: `{rejected_probe.get('admissibility')}`")
        lines.append(f"- renderer_template_probe_delta_summary: `{rejected_probe.get('delta_summary')}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics") or "")])
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
