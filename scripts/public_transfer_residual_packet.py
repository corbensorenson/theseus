#!/usr/bin/env python3
"""Build a teacher-ready residual packet after a public transfer check.

This script does not call the teacher and does not train on public benchmark
content. It compresses public calibration evidence into aggregate residual
families, hashed failing-task identifiers, and private-gate context so a later
teacher call can ask for architecture experiments without exposing benchmark
answers, tests, or prompts as training material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUT = REPORTS / "public_transfer_residual_packet.json"
DEFAULT_MARKDOWN = REPORTS / "public_transfer_residual_packet.md"
DEFAULT_PROMPT = REPORTS / "teacher_public_transfer_residual_prompt.md"
PUBLIC_CODE_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", default="reports/broad_transfer_matrix.json")
    parser.add_argument(
        "--real-code-verdict",
        default="",
        help="Optional exact real_code_benchmark_graduation verdict to summarize instead of broad-matrix aggregate evidence.",
    )
    parser.add_argument("--private-closure", default="reports/code_lm_closure_private_pressure_private.json")
    parser.add_argument("--ablation-gate", default="reports/decoder_v2_private_ablation_gate.json")
    parser.add_argument("--edge-gate", default="reports/edge_obligation_decode_gate_v1_private_pressure_private.json")
    parser.add_argument("--plan-ir", default="reports/decoder_plan_ir_private_pressure.json")
    parser.add_argument("--scheduler", default="reports/broad_code_calibration_scheduler.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--prompt-out", default=str(DEFAULT_PROMPT.relative_to(ROOT)))
    args = parser.parse_args()

    broad = read_json(resolve(args.broad), {})
    real_code_verdict = read_json(resolve(args.real_code_verdict), {}) if args.real_code_verdict else {}
    private_closure = read_json(resolve(args.private_closure), {})
    ablation = read_json(resolve(args.ablation_gate), {})
    edge_gate = read_json(resolve(args.edge_gate), {})
    plan_ir = read_json(resolve(args.plan_ir), {})
    scheduler = read_json(resolve(args.scheduler), {})

    broad_packet = (
        summarize_real_code_verdict(real_code_verdict, verdict_path=args.real_code_verdict)
        if real_code_verdict
        else summarize_broad_transfer(broad)
    )
    private_packet = summarize_private_context(private_closure, ablation, edge_gate, plan_ir, scheduler)
    reason = classify_reason(broad_packet, private_packet)
    prompt_text = render_teacher_prompt(reason, broad_packet, private_packet)
    local_evidence_paths = [rel(resolve(args.out)), rel(resolve(args.private_closure)), rel(resolve(args.ablation_gate)), rel(resolve(args.plan_ir))]
    if not real_code_verdict:
        local_evidence_paths.insert(1, rel(resolve(args.broad)))
    teacher_command = [
        sys.executable,
        "scripts/teacher_oracle.py",
        "--reason",
        "architecture_wall",
        "--mode",
        "proposal",
        "--prompt-file",
        rel(resolve(args.prompt_out)),
        "--local-evidence",
        *local_evidence_paths,
        "--queue-only",
        "--out",
        "reports/teacher_public_transfer_residual_last.json",
    ]
    gates = [
        gate(
            "public_transfer_loaded",
            bool(broad) or bool(real_code_verdict),
            {
                "broad_matrix_created_utc": broad.get("created_utc"),
                "real_code_verdict_created_utc": real_code_verdict.get("created_utc"),
                "real_code_verdict_path": args.real_code_verdict or None,
            },
        ),
        gate("public_residuals_are_aggregate_or_hashed", not public_content_embedded(broad_packet), "no prompts/tests/solutions/code"),
        gate("private_context_loaded", bool(private_closure) or bool(plan_ir), {"private_closure": bool(private_closure), "plan_ir": bool(plan_ir)}),
        gate("teacher_prompt_specific_reason", bool(reason.get("reason_for_call")), reason),
        gate("proposal_only_command", "--mode" in teacher_command and "proposal" in teacher_command and "--queue-only" in teacher_command, teacher_command),
    ]
    payload = {
        "policy": "project_theseus_public_transfer_residual_packet_v1",
        "created_utc": now(),
        "trigger_state": "GREEN" if all(row["passed"] for row in gates) else "YELLOW",
        "summary": {
            "reason_for_teacher": reason.get("reason_for_call"),
            "aggregate_public_pass_rate": broad_packet.get("aggregate_public_pass_rate"),
            "calibration_source": broad_packet.get("calibration_source"),
            "fresh_verdict_path": broad_packet.get("fresh_verdict_path"),
            "cards_below_floor": broad_packet.get("cards_below_floor"),
            "dominant_residuals": broad_packet.get("dominant_residuals"),
            "next_source_patch": broad_packet.get("next_source_patch"),
            "private_plan_ir_rows": private_packet.get("decoder_plan_ir", {}).get("plan_ir_row_count"),
            "external_inference_calls": 0,
        },
        "teacher_packet": {
            "reason": reason,
            "public_calibration": broad_packet,
            "private_context": private_packet,
            "allowed_experiment_types": [
                "private verifier-guided skeleton planning",
                "private STS-conditioned skeleton-choice ablation",
                "private local adapter generation for execution-shaped tasks",
                "private edge-case execution repair",
                "private repo-repair transfer bridge",
                "SymLiquid route/retry bias experiment",
            ],
            "forbidden_actions": [
                "public benchmark answers",
                "public hidden tests",
                "public solution distillation",
                "training on public prompts/tests/solutions",
                "benchmark-specific wrappers/templates",
                "teacher apply mode",
                "unreviewed destructive side effects",
            ],
            "expected_teacher_output_schema": {
                "diagnosis": "one paragraph naming the architectural wall",
                "experiment_specs": [
                    {
                        "id": "short_snake_case",
                        "hypothesis": "mechanism-level claim",
                        "private_eval_plan": "exact local private gate",
                        "promotion_rule": "one public 4-card calibration only after private gate",
                        "rollback_rule": "what to demote if flat/regressive",
                        "forbidden_ack": "explicit no public answers/no apply mode",
                    }
                ],
            },
        },
        "teacher_prompt_file": rel(resolve(args.prompt_out)),
        "teacher_command_queue_only": teacher_command,
        "gates": gates,
        "rules": {
            "public_benchmarks": "calibration-only; only aggregate residual families and hashed task identifiers are included",
            "teacher": "proposal-only architecture experiments",
            "promotion": "private held-out gate first, then one public 4-card calibration",
        },
        "external_inference_calls": 0,
    }
    write_text(resolve(args.prompt_out), prompt_text)
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    ingest_self(resolve(args.out), payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] == "GREEN" else 1


def summarize_broad_transfer(broad: dict[str, Any]) -> dict[str, Any]:
    summary = object_field(broad, "summary")
    rows = broad.get("rows") if isinstance(broad.get("rows"), list) else []
    residuals: Counter[str] = Counter()
    below_floor = []
    cards = []
    hashed_failures = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        card = str(row.get("card_id") or row.get("card") or "")
        rate = first_number(row.get("multi_stream_pass_rate"), row.get("real_public_pass_rate"), row.get("pass_rate")) or 0.0
        residual_counts = row.get("residual_family_counts") if isinstance(row.get("residual_family_counts"), dict) else {}
        for family, count in residual_counts.items():
            residuals[str(family)] += int(number(count))
        task_ids = row.get("residual_task_ids") if isinstance(row.get("residual_task_ids"), list) else []
        for task_id in task_ids[:32]:
            hashed_failures.append({"card_id": card, "task_hash": stable_hash(f"{card}:{task_id}")[:16]})
        card_packet = {
            "card_id": card,
            "public_task_count": int(number(row.get("public_task_count") or row.get("case_count"))),
            "multi_stream_pass_rate": rate,
            "single_stream_pass_rate": first_number(row.get("single_stream_pass_rate")),
            "sts_delta": first_number(row.get("pass_rate_delta")),
            "residual_family_counts": residual_counts,
            "clean_evidence": bool(row.get("clean_student_evidence_available")),
            "no_cheat_valid": bool(row.get("no_cheat_valid")),
            "selected_report": row.get("selected_report"),
        }
        cards.append(card_packet)
        if rate < PUBLIC_CODE_FLOOR and card:
            below_floor.append(card_packet)
    return {
        "created_utc": broad.get("created_utc"),
        "trigger_state": broad.get("trigger_state"),
        "aggregate_public_pass_rate": first_number(summary.get("real_public_pass_rate"), summary.get("aggregate_pass_rate")),
        "real_public_task_count": int(number(summary.get("real_public_task_count"))),
        "floor": PUBLIC_CODE_FLOOR,
        "cards_below_floor": [row["card_id"] for row in below_floor],
        "card_summaries": cards,
        "dominant_residuals": residuals.most_common(10),
        "hashed_public_residual_task_ids": hashed_failures[:80],
        "no_cheat_violation_count": int(number(summary.get("no_cheat_violation_count"))),
        "public_content_policy": "No public prompts, tests, answers, or generated candidate code are embedded.",
    }


def summarize_real_code_verdict(report: dict[str, Any], *, verdict_path: str = "") -> dict[str, Any]:
    summary = object_field(report, "summary")
    suites = report.get("suites") if isinstance(report.get("suites"), list) else []
    residual_rows = report.get("residuals") if isinstance(report.get("residuals"), list) else []
    if not residual_rows:
        residual_rows = [
            item
            for suite in suites
            for item in (suite.get("residuals") if isinstance(suite, dict) and isinstance(suite.get("residuals"), list) else [])
            if isinstance(item, dict)
        ]
    residuals: Counter[str] = Counter()
    card_residuals: dict[str, Counter[str]] = {}
    stage_counts: Counter[str] = Counter()
    hashed_failures = []
    for item in residual_rows:
        if not isinstance(item, dict):
            continue
        detail = str(item.get("detail") or "")
        family = residual_family(str(item.get("type") or "unknown"), detail)
        card = str(item.get("card_id") or "unknown")
        task_id = str(item.get("task_id") or "")
        stage = residual_stage(detail)
        residuals[family] += 1
        stage_counts[stage] += 1
        card_residuals.setdefault(card, Counter())[family] += 1
        if task_id:
            hashed_failures.append(
                {
                    "card_id": card,
                    "residual_type": family,
                    "stage": stage,
                    "task_hash": stable_hash(f"{card}:{task_id}")[:16],
                }
            )
    cards = []
    below_floor = []
    for suite in suites:
        if not isinstance(suite, dict):
            continue
        card = str(suite.get("card_id") or suite.get("source_id") or "")
        rate = first_number(suite.get("multi_stream_pass_rate"), suite.get("pass_rate")) or 0.0
        card_packet = {
            "card_id": card,
            "public_task_count": int(number(suite.get("case_count"))),
            "multi_stream_pass_rate": rate,
            "single_stream_pass_rate": first_number(suite.get("single_stream_pass_rate")),
            "sts_delta": first_number(suite.get("pass_rate_delta")),
            "residual_family_counts": dict(card_residuals.get(card, Counter()).most_common()),
            "residual_count": int(number(suite.get("residual_count"))),
            "clean_evidence": bool(suite.get("student_candidate_benchmark_integrity_valid", True)),
            "no_cheat_valid": bool(suite.get("student_candidate_benchmark_integrity_valid", True)),
            "selected_report": verdict_path,
        }
        cards.append(card_packet)
        if card and rate < PUBLIC_CODE_FLOOR:
            below_floor.append(card_packet)
    return {
        "created_utc": report.get("created_utc"),
        "trigger_state": report.get("trigger_state"),
        "calibration_source": "exact_real_code_verdict",
        "fresh_verdict_path": verdict_path,
        "aggregate_public_pass_rate": first_number(
            summary.get("real_public_task_pass_rate"),
            report.get("score"),
            summary.get("aggregate_pass_rate"),
        ),
        "real_public_task_count": int(number(summary.get("public_task_count"))),
        "floor": PUBLIC_CODE_FLOOR,
        "cards_below_floor": [row["card_id"] for row in below_floor],
        "card_summaries": cards,
        "dominant_residuals": residuals.most_common(10),
        "next_source_patch": source_patch_hint(residuals),
        "residual_stage_counts": dict(stage_counts.most_common()),
        "hashed_public_residual_task_ids": hashed_failures[:120],
        "template_like_candidate_count": int(number(summary.get("template_like_candidate_count"))),
        "loop_closure_candidate_count": int(number(summary.get("loop_closure_candidate_count"))),
        "external_inference_calls": int(number(report.get("external_inference_calls"))),
        "student_candidate_benchmark_integrity_valid": bool(summary.get("student_candidate_benchmark_integrity_valid")),
        "public_content_policy": "No public prompts, tests, answers, raw task ids, or generated candidate code are embedded.",
    }


def summarize_private_context(
    private_closure: dict[str, Any],
    ablation: dict[str, Any],
    edge_gate: dict[str, Any],
    plan_ir: dict[str, Any],
    scheduler: dict[str, Any],
) -> dict[str, Any]:
    closure_summary = object_field(private_closure, "summary")
    plan_summary = object_field(plan_ir, "summary")
    return {
        "private_closure": {
            "created_utc": private_closure.get("created_utc"),
            "trigger_state": private_closure.get("trigger_state"),
            "private_train_rows": first_number(closure_summary.get("private_train_rows"), closure_summary.get("outer_private_train_rows")),
            "private_eval_pass_rate": first_number(closure_summary.get("private_eval_pass_rate"), closure_summary.get("eval_pass_rate")),
            "sts_repair_delta": first_number(closure_summary.get("sts_repair_delta"), closure_summary.get("real_public_sts_delta")),
        },
        "ablation_gate": {
            "created_utc": ablation.get("created_utc"),
            "trigger_state": ablation.get("trigger_state"),
            "ready_for_public_calibration": bool(ablation.get("ready_for_public_calibration")),
            "candidate_groups": compact_candidate_groups(ablation.get("candidate_groups")),
        },
        "edge_obligation_gate": {
            "created_utc": edge_gate.get("created_utc"),
            "trigger_state": edge_gate.get("trigger_state"),
            "ready_for_public_calibration": bool(edge_gate.get("ready_for_public_calibration")),
            "summary": object_field(edge_gate, "summary"),
        },
        "decoder_plan_ir": {
            "created_utc": plan_ir.get("created_utc"),
            "trigger_state": plan_ir.get("trigger_state"),
            "plan_ir_row_count": int(number(plan_summary.get("plan_ir_row_count"))),
            "coverage": object_field(plan_summary, "coverage"),
            "semantic_family_counts": plan_summary.get("semantic_family_counts") if isinstance(plan_summary.get("semantic_family_counts"), dict) else {},
            "skeleton_kind_counts": plan_summary.get("skeleton_kind_counts") if isinstance(plan_summary.get("skeleton_kind_counts"), dict) else {},
            "repair_signal_counts": plan_summary.get("repair_signal_counts") if isinstance(plan_summary.get("repair_signal_counts"), dict) else {},
        },
        "scheduler_gate": {
            "selected": object_field(scheduler, "selected"),
            "private_receiver_gate": object_field(scheduler, "private_receiver_gate"),
        },
        "private_public_split": "private rows and private held-out gates may train/evaluate; public cards calibrate only.",
    }


def classify_reason(broad_packet: dict[str, Any], private_packet: dict[str, Any]) -> dict[str, Any]:
    rate = first_number(broad_packet.get("aggregate_public_pass_rate")) or 0.0
    below = broad_packet.get("cards_below_floor") or []
    plan_rows = int(number(private_packet.get("decoder_plan_ir", {}).get("plan_ir_row_count")))
    if below and plan_rows:
        reason = "public_transfer_wall_after_private_decoder_pressure"
        intent = "Ask teacher for one architecture experiment that makes private contract/skeleton pressure transfer broadly."
    elif below:
        reason = "public_transfer_wall"
        intent = "Ask teacher for one bounded private experiment targeting the residual cluster."
    else:
        reason = "public_transfer_review"
        intent = "Ask teacher to preserve the mechanism that transferred and propose regression protection only."
    return {
        "reason_for_call": reason,
        "intent": intent,
        "aggregate_public_pass_rate": rate,
        "cards_below_floor": below,
        "dominant_residuals": broad_packet.get("dominant_residuals"),
        "next_source_patch": broad_packet.get("next_source_patch"),
    }


def render_teacher_prompt(reason: dict[str, Any], broad_packet: dict[str, Any], private_packet: dict[str, Any]) -> str:
    prompt = {
        "role": "Teacher-as-architect for Project Theseus",
        "task": "Propose architecture experiments only. Do not provide benchmark answers, hidden tests, public solutions, or direct patches.",
        "reason": reason,
        "public_calibration_summary": broad_packet,
        "private_context_summary": private_packet,
        "requested_output": {
            "diagnosis": "Name the specific architectural bottleneck.",
            "experiment_specs": "Return 1-3 experiments max. Each must have private eval, promotion rule, rollback rule, and forbidden-action acknowledgement.",
        },
        "hard_rules": [
            "No public benchmark answers, prompts, tests, or solution distillation.",
            "No apply mode. No direct code edits.",
            "Use private generated/local data and private held-out gates first.",
            "Allow exactly one public 4-card calibration only after a private gate passes.",
        ],
    }
    return "# Teacher Public Transfer Residual Prompt\n\n```json\n" + json.dumps(prompt, indent=2, sort_keys=True) + "\n```\n"


def compact_candidate_groups(value: Any) -> dict[str, Any]:
    groups = value if isinstance(value, dict) else {}
    out = {}
    for name, row in groups.items():
        if not isinstance(row, dict):
            continue
        out[str(name)] = {
            "task_count": row.get("task_count"),
            "candidate_count": row.get("candidate_count"),
            "verifier_pass_rate": row.get("verifier_pass_rate"),
            "top_fail_reasons": row.get("top_fail_reasons"),
            "top_modes": row.get("top_modes"),
        }
    return out


def public_content_embedded(packet: dict[str, Any]) -> bool:
    text = json.dumps(packet, sort_keys=True).lower()
    banned = ["canonical_solution", "hidden_test", "assert ", "def solution", "candidate_code", "prompt_text"]
    return any(token in text for token in banned)


def residual_stage(detail: str) -> str:
    text = detail.lower()
    if (
        "d:/projecttheseus/tmp" in text
        or "d:\\projecttheseus\\tmp" in text
        or ("can't open file" in text and "theseus_real_code_grad" in text and "projecttheseus/tmp" in text)
    ):
        return "platform_path_resolution_failed"
    if "__theseus_stage__:" in text:
        return text.split("__theseus_stage__:", 1)[1].splitlines()[0].strip() or "stage_unknown"
    if "candidate_dependency_unavailable" in text:
        return "candidate_dependency_unavailable"
    if "unavailable_external_import" in text:
        return "unavailable_external_import"
    if "beautiful_code_lint_failed" in text:
        return "beautiful_code_lint_failed"
    if "missing local theseus student checkpoint candidate" in text:
        return "no_candidate"
    if "typeerror" in text:
        return "type_error"
    if "assertionerror" in text:
        return "assertion_failed"
    return "stage_unknown"


def residual_family(raw_type: str, detail: str) -> str:
    text = f"{raw_type} {detail}".lower()
    if (
        "d:/projecttheseus/tmp" in text
        or "d:\\projecttheseus\\tmp" in text
        or ("can't open file" in text and "theseus_real_code_grad" in text and "projecttheseus/tmp" in text)
    ):
        return "platform_path_normalization"
    return raw_type or "unknown"


def source_patch_hint(residuals: Counter[str]) -> str:
    if residuals.get("platform_path_normalization", 0):
        return "scripts/real_code_benchmark_support.py: runtime_tmp_dir must stay platform-native on macOS/Linux"
    if residuals.get("local_code_generation_adapter_needed", 0):
        return "student candidate adapter must materialize candidates for every gated public task before calibration"
    if residuals.get("algorithm_choice", 0):
        return "private algorithm-choice pressure must improve before the next bounded public calibration"
    return ""


def ingest_self(path: Path, payload: dict[str, Any]) -> None:
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import report_evidence_store  # type: ignore

        report_evidence_store.ingest_report_path(report_evidence_store.DEFAULT_DB, path, payload=payload)
    except Exception:
        return


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    packet = payload.get("teacher_packet", {})
    public = packet.get("public_calibration", {}) if isinstance(packet, dict) else {}
    private = packet.get("private_context", {}) if isinstance(packet, dict) else {}
    lines = [
        "# Public Transfer Residual Packet",
        "",
        f"- State: `{payload.get('trigger_state')}`",
        f"- Teacher reason: `{summary.get('reason_for_teacher')}`",
        f"- Aggregate public pass rate: `{summary.get('aggregate_public_pass_rate')}`",
        f"- Cards below floor: `{', '.join(summary.get('cards_below_floor') or [])}`",
        f"- Private Plan IR rows: `{summary.get('private_plan_ir_rows')}`",
        "",
        "## Dominant Residuals",
        "",
    ]
    for family, count in public.get("dominant_residuals", [])[:10]:
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Private Context", ""])
    lines.append(f"- Decoder Plan IR: `{private.get('decoder_plan_ir', {}).get('trigger_state')}` rows `{private.get('decoder_plan_ir', {}).get('plan_ir_row_count')}`")
    lines.append(f"- Ablation gate: `{private.get('ablation_gate', {}).get('trigger_state')}` ready `{private.get('ablation_gate', {}).get('ready_for_public_calibration')}`")
    lines.append(f"- Edge gate: `{private.get('edge_obligation_gate', {}).get('trigger_state')}` ready `{private.get('edge_obligation_gate', {}).get('ready_for_public_calibration')}`")
    lines.extend(["", "Public benchmark content remains calibration-only; the packet uses aggregate counts and hashed failing identifiers only.", ""])
    return "\n".join(lines)


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def object_field(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else {}
    return item if isinstance(item, dict) else {}


def first_number(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        # Use absolute() instead of resolve() so report-directory junctions stay
        # anchored to the active workspace path in generated teacher commands.
        return str(path.absolute().relative_to(ROOT.absolute())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
