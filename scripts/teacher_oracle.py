"""Sparse teacher/architect wrapper for SparkStream.

This script lets the local ratchet ask Codex CLI for high-level architectural
guidance only when policy allows it. By default it is proposal-only and uses a
read-only sandbox. Governed distillation mode may request candidate training
rows, but those rows are only trainable after the manifest/gate admits them.
All calls are logged so teacher dependence can be measured and driven down over
time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from teacher_provider_policy import teacher_launch_decision  # noqa: E402

DEFAULT_POLICY = ROOT / "configs" / "teacher_policy.json"
DEFAULT_CODEX_MIRROR = Path.home() / ".codex" / "external-bin" / "codex.exe"
MAX_EVIDENCE_CHARS = 3200
MAX_INLINE_EVIDENCE_CHARS = 900

REASON_INTENTS = {
    "architecture_wall": "Diagnose the current measured architecture wall and propose one smallest local experiment that can move the wall.",
    "bigcodebench_execution_wall": "Diagnose the BigCodeBench receiver hard-zero wall from residual evidence and propose one private-first execution-shaped decoder experiment.",
    "benchmark_frontier_design": "Design the next transferable benchmark frontier without adding public answers, hidden tests, or benchmark-specific hacks.",
    "frontier_exhausted": "Explain why the current frontier is saturated and propose the next harder private pressure or receiver calibration.",
    "residual_conflict": "Resolve conflicting residual signals by naming the likely common mechanism and one disambiguating experiment.",
    "promotion_gate_blocked": "Identify the exact blocked promotion gate and the smallest verification-first intervention to unblock or confirm it.",
    "user_requested_benchmark": "Help frame the requested benchmark as calibration-only evidence with private training pressure kept separate.",
    "safety_or_governance_uncertainty": "Classify the governance risk and propose the safest reversible next step.",
    "attd_maintenance": "Diagnose unattended-runtime maintenance failure and propose one bounded repair with clear verification.",
    "checkpoint_chat_escalation": "Help route a conversational/checkpoint ambiguity into a safe local plan without overriding the personality core or evidence gates.",
}

DEFAULT_REASON_EVIDENCE = {
    "architecture_wall": [
        "reports/architecture_guidance_loop.json",
        "reports/broad_transfer_matrix.json",
        "reports/transfer_generalization_audit.json",
        "reports/learning_scoreboard.json",
        "reports/code_lm_closure_edge_contract_4card_private.json",
        "reports/autonomy_watchdog.json",
        "reports/candidate_promotion_gate.json",
    ],
    "bigcodebench_execution_wall": [
        "reports/architecture_guidance_loop_bigcodebench.json",
        "reports/real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.json",
        "reports/real_code_benchmark_traces_source_mbpp_source_evalplus_source_bigcodebench_source_livecodebench_seed14_32.jsonl",
        "reports/broad_transfer_matrix.json",
        "reports/transfer_generalization_audit.json",
        "reports/learning_scoreboard.json",
        "reports/autonomy_watchdog.json",
    ],
    "benchmark_frontier_design": [
        "reports/high_transfer_curriculum_scheduler.json",
        "reports/broad_transfer_matrix.json",
        "reports/transfer_generalization_audit.json",
        "reports/learning_scoreboard.json",
    ],
    "frontier_exhausted": [
        "reports/high_transfer_curriculum_scheduler.json",
        "reports/broad_transfer_matrix.json",
        "reports/learning_scoreboard.json",
        "reports/autonomy_watchdog.json",
    ],
    "residual_conflict": [
        "reports/architecture_guidance_loop.json",
        "reports/real_code_benchmark_traces.jsonl",
        "reports/learning_scoreboard.json",
    ],
    "promotion_gate_blocked": [
        "reports/candidate_promotion_gate.json",
        "reports/learning_scoreboard.json",
        "reports/broad_transfer_matrix.json",
    ],
    "attd_maintenance": [
        "reports/autonomy_watchdog.json",
        "reports/vacation_mode_supervisor_overnight.json",
        "reports/hive_work_board_executor.json",
        "reports/hive_node_registry.json",
        "reports/overnight_learning_readiness.json",
    ],
    "safety_or_governance_uncertainty": [
        "reports/autonomy_watchdog.json",
        "reports/candidate_promotion_gate.json",
        "reports/teacher_budget_last.json",
    ],
    "checkpoint_chat_escalation": [
        "reports/checkpoint_chat_last.json",
        "reports/personality_core_validation.json",
        "reports/autonomy_watchdog.json",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--reason", default="promotion_gate_blocked")
    parser.add_argument("--mode", choices=["proposal", "distillation", "apply"], default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--local-evidence", nargs="*", default=[])
    parser.add_argument("--allow-teacher", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument("--out", default="reports/teacher_oracle_last.json")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    provider_policy_path = resolve_path(
        policy.get("provider_policy_path", "configs/teacher_distillation_policy.json")
    )
    provider_policy = read_json(provider_policy_path)
    launch_decision = teacher_launch_decision(provider_policy, policy)
    if not launch_decision["accepted"]:
        blocked = {
            "status": "blocked_by_teacher_policy",
            "blocked_reason": "teacher_provider_or_executable_forbidden",
            "provider_policy_path": rel(provider_policy_path),
            "provider_decision": launch_decision,
            "external_inference_calls": 0,
        }
        write_json(ROOT / args.out, blocked)
        print(json.dumps(blocked, indent=2))
        return 1
    mode = args.mode or policy.get("default_mode", "proposal")
    request = build_request(policy, args, mode)
    if mode == "apply" and not bool((policy.get("budget") or {}).get("apply_mode_enabled", False)):
        request["status"] = "blocked_by_teacher_policy"
        request["blocked_reason"] = "teacher_apply_mode_forbidden"
        write_json(ROOT / args.out, request)
        print(json.dumps(request, indent=2))
        return 0
    if mode == "distillation" and not bool((policy.get("budget") or {}).get("distillation_training_enabled", False)):
        request["status"] = "blocked_by_teacher_policy"
        request["blocked_reason"] = "teacher_distillation_mode_disabled"
        write_json(ROOT / args.out, request)
        print(json.dumps(request, indent=2))
        return 0
    if bool((policy.get("budget") or {}).get("proposal_only_no_distillation", True)) and mode != "proposal":
        request["status"] = "blocked_by_teacher_policy"
        request["blocked_reason"] = "teacher_must_remain_proposal_only"
        write_json(ROOT / args.out, request)
        print(json.dumps(request, indent=2))
        return 0
    allowed_reasons = {str(item) for item in policy.get("allowed_reasons", [])}
    if allowed_reasons and args.reason not in allowed_reasons:
        request["status"] = "blocked_by_teacher_policy"
        request["blocked_reason"] = "reason_not_allowed"
        request["allowed_reasons"] = sorted(allowed_reasons)
        write_json(ROOT / args.out, request)
        print(json.dumps(request, indent=2))
        return 0

    if args.queue_only or not args.allow_teacher:
        request["status"] = "queued_not_executed"
        request["blocked_reason"] = (
            "queue_only" if args.queue_only else "teacher_requires_allow_teacher_flag"
        )
        append_jsonl(ROOT / policy.get("request_queue", "reports/teacher_request_queue.jsonl"), request)
        write_json(ROOT / policy.get("queue_last_path", "reports/teacher_queue_last.json"), request)
        out_path = ROOT / args.out
        preserve_last = bool(policy.get("preserve_last_completed_on_queue", True))
        if not preserve_last or not out_path.exists():
            write_json(out_path, request)
        print(json.dumps(request, indent=2))
        return 0

    evidence_decision = local_wall_evidence_decision(policy, args.reason, args.local_evidence)
    request["local_wall_evidence_decision"] = evidence_decision
    if not evidence_decision.get("allowed", True):
        request["status"] = "blocked_by_teacher_policy"
        request["blocked_reason"] = evidence_decision.get("reason", "local_wall_evidence_missing")
        write_json(ROOT / args.out, request)
        print(json.dumps(request, indent=2))
        return 0

    budget_decision = call_budget_decision(policy, args.reason)
    request["budget_decision"] = budget_decision
    if not budget_decision.get("allowed", False):
        request["status"] = "blocked_by_teacher_budget"
        request["blocked_reason"] = budget_decision.get("reason", "budget_gate")
        if should_log_budget_block(policy, request):
            append_jsonl(ROOT / policy.get("log_path", "reports/teacher_calls.jsonl"), request)
        write_json(ROOT / policy.get("budget_block_path", "reports/teacher_budget_last.json"), request)
        out_path = ROOT / args.out
        preserve_last = bool(policy.get("preserve_last_completed_on_budget_block", True))
        if not preserve_last or not out_path.exists():
            write_json(out_path, request)
        print(json.dumps(request, indent=2))
        return 0

    result = run_teacher(policy, request, mode)
    append_jsonl(ROOT / policy.get("log_path", "reports/teacher_calls.jsonl"), result)
    write_budget_status(policy, result)
    write_json(ROOT / args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "completed" else 1


def build_request(policy: dict[str, Any], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    prompt_source = "inline" if args.prompt else ("prompt_file" if args.prompt_file else "default")
    raw_prompt = args.prompt
    prompt_file_missing = False
    if args.prompt_file:
        prompt_path = resolve_path(args.prompt_file)
        if prompt_path.exists():
            raw_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt_file_missing = True
            raw_prompt = ""
    if not raw_prompt:
        raw_prompt = default_task_prompt(args.reason)
    wall_packet = build_wall_packet(
        policy,
        args.reason,
        args.local_evidence,
        mode=mode,
        prompt_source=prompt_source,
        prompt_file=args.prompt_file,
        prompt_file_missing=prompt_file_missing,
    )
    prompt = compose_teacher_prompt(policy, args.reason, raw_prompt, wall_packet, mode)
    return {
        "request_id": f"teacher_{int(time.time() * 1000)}",
        "created_utc": now(),
        "provider": policy.get("provider"),
        "model": policy.get("model"),
        "reasoning_effort": policy.get("reasoning_effort"),
        "reason_for_call": args.reason,
        "mode": mode,
        "local_evidence": args.local_evidence,
        "evidence_paths": wall_packet.get("evidence_paths", []),
        "inline_evidence": wall_packet.get("inline_evidence", []),
        "wall_packet": wall_packet,
        "prompt_source": prompt_source,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
        "policy": "sparse_teacher_governed_distillation" if mode == "distillation" else "sparse_teacher_only",
    }


def default_task_prompt(reason: str) -> str:
    return REASON_INTENTS.get(
        reason,
        "Diagnose the supplied local evidence and propose one bounded verification-first next step.",
    )


def compose_teacher_prompt(
    policy: dict[str, Any],
    reason: str,
    raw_prompt: str,
    wall_packet: dict[str, Any],
    mode: str,
) -> str:
    required = [
        "reason_for_call",
        "wall_type",
        "blocked_gate",
        "residual_family",
        "local_evidence_used",
        "diagnosis",
        "recommended_intervention",
        "implementation_plan",
        "local_executor_inputs",
        "private_eval_plan",
        "public_calibration_plan",
        "verification_steps",
        "promotion_gates",
        "promotion_rollback_rule",
        "risks",
        "evidence_gaps",
        "anti_goals_acknowledged",
        "forbidden_actions_acknowledged",
        "experiment_spec",
        "confidence",
        "distill_into_local_rules",
        "distillation_training_row",
    ]
    contract = policy.get("teacher_prompt_contract") if isinstance(policy.get("teacher_prompt_contract"), dict) else {}
    role = "sparse proposal-only architecture teacher"
    decision_request = [
        "Return one concrete architecture/training/verifier experiment spec that a local executor can run.",
        "Optimize for making the local executor do less interpretation: name the residual, source files/reports, command shape, private gate, and rollback rule.",
        "Do not brainstorm a menu unless evidence is insufficient; if insufficient, name the missing evidence precisely.",
    ]
    distillation_contract: list[str] = []
    if mode == "distillation":
        role = "governed training-time distillation teacher"
        decision_request = [
            "Return one concrete local training intervention plus at most one candidate distillation_training_row.",
            "The row must be private/open-licensed training pressure only, never a public benchmark answer, hidden test, solution, trace, answer template, or benchmark-specific wrapper.",
            "If you cannot produce a safe row from the supplied local evidence, set distillation_training_row=null and explain the missing evidence.",
        ]
        distillation_contract = [
            "DISTILLATION ROW CONTRACT:",
            "distillation_training_row may be an object only in mode=distillation.",
            "It must include row_id, source_kind='teacher_distillation', task_family, input_text, target_text, target_hash, license_spdx, provenance, public_benchmark=false, public_prompt=false, public_overlap_hits=0, holdout_overlap_hits=0, runtime_serving='forbidden', and admission_checks.",
            "For code-generation training pressure, include code_lm_task either as a top-level field on distillation_training_row or inside target_text JSON. code_lm_task must include task_id, split='train', category, concept_residual_label, prompt, entry_point, solution_body, tests, decoder_contract.visible_arg_count_hint, public_benchmark=false, and public_prompt=false.",
            "code_lm_task.solution_body must be only the indented body that belongs inside the function. Do not include a def line, decorators, imports, markdown fences, explanatory text, or wrapper code.",
            "The local renderer wraps solution_body as def entry_point(data): for one visible argument, def entry_point(data, other): for two visible arguments, or def entry_point(data, other=None, *extra): for more. The body must use these internal parameter names, not prompt-specific names such as records/items/numbers.",
            "code_lm_task.tests must be one newline-delimited Python assert source string. Do not encode tests as a JSON array; the local verifier executes the tests source directly.",
            "The code_lm_task must be a new private/project-internal task, must not target the currently held-out family-disjoint eval families, and must not contain public benchmark prompts/tests/solutions/traces/answer templates.",
            "The local manifest builder will execute solution_body against tests before the row can become training data; a row that only looks plausible is not admissible.",
            "provenance must include source='governed_teacher_distillation_call', created_utc, evidence_paths, policy='sparse_teacher_governed_distillation', and local_verifier='pending_local_manifest_gate'.",
            "Use target_hash='sha256:auto' unless the caller supplied the exact hash; the local verifier computes and retains the real hash.",
            "admission_checks must include provenance_retained, license_checked, leakage_audited, verifier_accepted, runtime_serving_forbidden, and public_benchmark_excluded.",
            "verifier_accepted=true means the teacher is proposing a concrete private executable row for the local verifier, not that the teacher's claim is trusted. The local manifest builder is the authority and will execute solution_body against tests before admission.",
            "If you cannot provide a concrete private executable row with solution_body and tests for local verification, set distillation_training_row=null.",
            "The local manifest/gate will re-check every field and may reject the row. Rejection is acceptable; unsafe rows are not.",
            "",
        ]
    else:
        distillation_contract = [
            "DISTILLATION ROW CONTRACT:",
            "Set distillation_training_row=null outside distillation mode.",
            "",
        ]
    return "\n".join(
        [
            "# Project Theseus Teacher Call Contract",
            "",
            "ROLE:",
            f"You are a {role} for Project Theseus / VIEA / SymLiquid.",
            "You are called only when local loops have hit a measured wall or need governed experiment design.",
            f"CALL_MODE: {mode}",
            "",
            f"REASON_FOR_CALL: {reason}",
            f"REASON_INTENT: {REASON_INTENTS.get(reason, 'Use the local wall packet to diagnose the exact next experiment.')}",
            "",
            "DECISION REQUEST:",
            *decision_request,
            "",
            "ANTI-GOALS:",
            "- Do not solve benchmark tasks.",
            "- Do not provide public benchmark answers, hidden tests, canonical solutions, wrappers, or templates.",
            "- Do not recommend training on public benchmark solutions or public hidden-test lookalikes.",
            "- Do not claim that you ran training, inspected files not in the packet, or verified results yourself.",
            "- Do not ask for teacher apply mode or direct code edits.",
            "- Do not emit fallback returns, template shortcuts, or wrapper-only answers as training credit.",
            "- Do not use web/network facts unless the user explicitly supplied them in the evidence packet.",
            "- Do not optimize a single benchmark at the cost of broad transfer.",
            "- In proposal mode, do not emit training rows. In distillation mode, emit at most one governed candidate row and only when the row contract is satisfied.",
            "",
            "LOCAL WALL PACKET JSON:",
            json.dumps(wall_packet, indent=2, sort_keys=True),
            "",
            "CALLER PROMPT:",
            raw_prompt.strip(),
            "",
            "OUTPUT CONTRACT:",
            "Return JSON only. Required schema fields are: " + ", ".join(required) + ".",
            "Use null for nullable fields that are genuinely not applicable; use [] for empty evidence/anti-goal arrays.",
            "If experiment_spec is not applicable, set it to null. If it is present, include id, hypothesis, target_files, private_eval, public_calibration, rollback_plan, and success_metric.",
            "The recommended_intervention must be the smallest useful local change.",
            "local_executor_inputs must list the exact reports, residual counts, code areas, and config switches the local system should inspect or change.",
            "private_eval_plan must be sufficient to decide whether the experiment worked before any public benchmark rerun.",
            "public_calibration_plan must be one bounded calibration only, and only after the private gate passes.",
            "The implementation_plan must name likely source/report/config areas, not public answers.",
            "The verification_steps must start with private eval or local diagnostic evidence, then honest public calibration only.",
            "The promotion_gates must include leakage/no-template/no-wrapper checks when benchmark evidence is involved.",
            "promotion_rollback_rule must state exactly what gets preserved, rolled back, or demoted if transfer is flat.",
            "forbidden_actions_acknowledged must explicitly confirm: no public answers, no public solutions in training, no teacher apply mode, no wrapper/template shortcuts, no benchmark-specific score games.",
            "Set distill_into_local_rules=true only for durable routing, prompt, verifier, or governance lessons.",
            "",
            *distillation_contract,
            "CURRENT TEACHER PROMPT POLICY JSON:",
            json.dumps(contract, indent=2, sort_keys=True),
            "",
        ]
    )


def build_wall_packet(
    policy: dict[str, Any],
    reason: str,
    explicit_evidence: list[str],
    *,
    mode: str,
    prompt_source: str,
    prompt_file: str,
    prompt_file_missing: bool,
) -> dict[str, Any]:
    evidence_paths, inline_evidence, missing_evidence_paths = split_evidence_items(reason, explicit_evidence)
    evidence_reports = [summarize_evidence_path(path) for path in evidence_paths]
    evidence_reports = [row for row in evidence_reports if row.get("exists")]
    wall_summary = infer_wall_summary(reason, evidence_reports, inline_evidence)
    return {
        "packet_policy": "theseus_teacher_wall_packet_v1",
        "created_utc": now(),
        "reason_for_call": reason,
        "reason_intent": REASON_INTENTS.get(reason, ""),
        "prompt_source": prompt_source,
        "prompt_file": prompt_file,
        "prompt_file_missing": prompt_file_missing,
        "teacher_mode": mode,
        "evidence_paths": [row.get("path") for row in evidence_reports],
        "missing_evidence_paths": missing_evidence_paths,
        "inline_evidence": inline_evidence,
        "evidence_reports": evidence_reports,
        "wall_summary": wall_summary,
        "forbidden_teacher_roles": [
            "benchmark answer generator",
            "hidden-test author",
            "public-solution distiller",
            "apply-mode patcher",
            "unbounded architecture brainstormer",
        ],
        "expected_teacher_output": {
            "shape": "one experiment spec plus verification gates",
            "private_first": True,
            "public_calibration_only": True,
            "promotion_decided_locally": True,
        },
    }


def split_evidence_items(reason: str, explicit_evidence: list[str]) -> tuple[list[Path], list[str], list[str]]:
    paths: list[Path] = []
    inline: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for item in DEFAULT_REASON_EVIDENCE.get(reason, []):
        text = str(item or "").strip()
        if not text:
            continue
        candidate = resolve_path(text)
        if candidate.exists():
            key = str(candidate).lower()
            if key not in seen:
                paths.append(candidate)
                seen.add(key)
            continue
        missing.append(text)
    for item in explicit_evidence:
        text = str(item or "").strip()
        if not text:
            continue
        candidate = resolve_path(text)
        if candidate.exists():
            key = str(candidate).lower()
            if key not in seen:
                paths.append(candidate)
                seen.add(key)
            continue
        if "=" in text or len(text) > 0:
            inline.append(text[:MAX_INLINE_EVIDENCE_CHARS])
    return paths, inline[:16], missing[:16]


def summarize_evidence_path(path: Path) -> dict[str, Any]:
    rel_path = rel(path)
    if not path.exists():
        return {"path": rel_path, "exists": False}
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_json_report(rel_path, value)
        elif suffix == ".jsonl":
            rows = read_jsonl(path)
            summary = summarize_jsonl_report(rel_path, rows)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            summary = {
                "kind": "text",
                "line_count": len(text.splitlines()),
                "snippet": text[:MAX_EVIDENCE_CHARS],
            }
        return {
            "path": rel_path,
            "exists": True,
            "sha256": file_sha256(path),
            "summary": summary,
        }
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "path": rel_path,
            "exists": True,
            "error": str(exc),
        }


def summarize_json_report(rel_path: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"kind": "json", "value_type": type(value).__name__, "snippet": compact(value)}
    summary: dict[str, Any] = {
        "kind": "json",
        "policy": value.get("policy"),
        "trigger_state": value.get("trigger_state"),
        "status": value.get("status"),
        "created_utc": value.get("created_utc"),
        "blocked_reason": value.get("blocked_reason"),
    }
    for key in (
        "summary",
        "diagnosis",
        "promotion",
        "public_transfer",
        "broad_transfer_matrix",
        "wall_summary",
        "teacher",
        "artifacts",
    ):
        if key in value:
            summary[key] = bounded(value[key])
    gates = value.get("gates") or value.get("checks")
    if isinstance(gates, list):
        failed = [
            {
                "gate": row.get("gate") or row.get("check"),
                "severity": row.get("severity"),
                "evidence": bounded(row.get("evidence")),
            }
            for row in gates
            if isinstance(row, dict) and row.get("passed") is False
        ]
        summary["failed_gates"] = failed[:10]
        summary["gate_count"] = len(gates)
    if rel_path.endswith("broad_transfer_matrix.json"):
        summary["broad_cards_below_floor"] = get_path(value, ["summary", "cards_below_floor"], [])
        summary["real_public_pass_rate"] = get_path(value, ["summary", "real_public_pass_rate"], None)
        summary["real_public_task_count"] = get_path(value, ["summary", "real_public_task_count"], None)
    return prune_empty(summary)


def summarize_jsonl_report(rel_path: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample = []
    for row in rows[-8:]:
        if not isinstance(row, dict):
            continue
        sample.append(
            {
                "event": row.get("event"),
                "task_hash": short_hash(str(row.get("task_id") or row.get("source_task_id") or row.get("id") or "")),
                "passed": row.get("passed"),
                "residual_class": row.get("residual_class"),
                "candidate_origin": row.get("candidate_origin"),
                "status": row.get("status"),
            }
        )
    counts: dict[str, int] = {}
    residuals: dict[str, int] = {}
    for row in rows:
        event = str(row.get("event") or row.get("status") or "row")
        counts[event] = counts.get(event, 0) + 1
        residual = str(row.get("residual_class") or "")
        if residual:
            residuals[residual] = residuals.get(residual, 0) + 1
    return {
        "kind": "jsonl",
        "row_count": len(rows),
        "event_counts": dict(sorted(counts.items())[:12]),
        "residual_counts": dict(sorted(residuals.items(), key=lambda item: item[1], reverse=True)[:12]),
        "recent_sample": sample,
        "path": rel_path,
    }


def infer_wall_summary(reason: str, reports: list[dict[str, Any]], inline_evidence: list[str]) -> dict[str, Any]:
    failed_gates: list[str] = []
    trigger_states: dict[str, str] = {}
    wall_cards: set[str] = set()
    residual_counts: dict[str, int] = {}
    for report in reports:
        path = str(report.get("path") or "")
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        state = summary.get("trigger_state") or summary.get("status")
        if state:
            trigger_states[path] = str(state)
        for gate in summary.get("failed_gates", []) if isinstance(summary.get("failed_gates"), list) else []:
            if isinstance(gate, dict) and gate.get("gate"):
                failed_gates.append(str(gate.get("gate")))
        for card in summary.get("broad_cards_below_floor", []) if isinstance(summary.get("broad_cards_below_floor"), list) else []:
            wall_cards.add(str(card))
        nested_diag = summary.get("diagnosis") if isinstance(summary.get("diagnosis"), dict) else {}
        for card in nested_diag.get("broad_wall_cards", []) if isinstance(nested_diag.get("broad_wall_cards"), list) else []:
            wall_cards.add(str(card))
        if isinstance(summary.get("residual_counts"), dict):
            for key, value in summary["residual_counts"].items():
                residual_counts[str(key)] = residual_counts.get(str(key), 0) + int(value or 0)
        report_residual = dominant_residual_from_report(summary)
        if report_residual:
            residual_counts[report_residual] = max(residual_counts.get(report_residual, 0), 1)
    dominant_residual = ""
    if residual_counts:
        dominant_residual = max(residual_counts.items(), key=lambda item: item[1])[0]
    return {
        "reason_for_call": reason,
        "intent": REASON_INTENTS.get(reason, ""),
        "trigger_states": trigger_states,
        "failed_gates": failed_gates[:12],
        "wall_cards": sorted(wall_cards),
        "dominant_residual": dominant_residual,
        "inline_evidence_count": len(inline_evidence),
        "evidence_report_count": len(reports),
    }


def dominant_residual_from_report(summary: dict[str, Any]) -> str:
    diagnosis = summary.get("diagnosis") if isinstance(summary.get("diagnosis"), dict) else {}
    residual = str(diagnosis.get("dominant_residual") or "")
    if residual:
        return residual
    nested = summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
    return str(nested.get("dominant_residual") or "")


def bounded(value: Any, max_chars: int = MAX_EVIDENCE_CHARS) -> Any:
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return value
    return {"truncated_json": text[:max_chars], "original_chars": len(text)}


def compact(value: Any, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text[:max_chars]


def prune_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_teacher(policy: dict[str, Any], request: dict[str, Any], mode: str) -> dict[str, Any]:
    last_message = ROOT / policy.get("last_message_path", "reports/teacher_last_message.md")
    last_message.parent.mkdir(parents=True, exist_ok=True)
    sandbox = (
        policy.get("apply_sandbox", "workspace-write")
        if mode == "apply"
        else policy.get("proposal_sandbox", "read-only")
    )
    command = [
        resolve_codex_command(policy),
        "exec",
        "-m",
        policy.get("model", "gpt-5.6-sol"),
        "-c",
        f'model_reasoning_effort="{policy.get("reasoning_effort", "high")}"',
        "-c",
        f'approval_policy="{policy.get("approval_policy", "never")}"',
        "-s",
        sandbox,
        "-C",
        str(ROOT),
        "-o",
        str(last_message),
    ]
    schema = ROOT / str(policy.get("output_schema", ""))
    if schema.exists():
        command.extend(["--output-schema", str(schema)])
    command.append("-")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            input=request["prompt"],
            text=True,
            capture_output=True,
            timeout=int(policy.get("timeout_seconds", 1800)),
        )
        combined_output = f"{proc.stdout}\n{proc.stderr}"
        usage_limit = codex_usage_limit_info(combined_output)
        if proc.returncode != 0 and usage_limit:
            return {
                **request,
                "status": "blocked_by_codex_usage_limit",
                "blocked_reason": usage_limit["blocked_reason"],
                "retry_after_hint": usage_limit.get("retry_after_hint", ""),
                "completed_utc": now(),
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "command": command_without_prompt(command),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "response_text": "",
                "response_json": None,
            }
        output = last_message.read_text(encoding="utf-8") if proc.returncode == 0 and last_message.exists() else proc.stdout
        parsed = parse_json_object(output) if proc.returncode == 0 else None
        status = "completed" if proc.returncode == 0 else "failed"
        return {
            **request,
            "status": status,
            "external_inference_calls": 1 if status == "completed" else 0,
            "completed_utc": now(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "command": command_without_prompt(command),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "response_text": output,
            "response_json": parsed,
        }
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            queued = {
                **request,
                "status": "queued_not_executed",
                "blocked_reason": "codex_cli_not_found",
                "completed_utc": now(),
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "command": command_without_prompt(command),
                "error": str(exc),
                "fallback": "queued_for_manual_or_app_level_teacher_execution",
            }
            append_jsonl(ROOT / policy.get("request_queue", "reports/teacher_request_queue.jsonl"), queued)
            write_json(ROOT / policy.get("queue_last_path", "reports/teacher_queue_last.json"), queued)
            return queued
        if isinstance(exc, PermissionError) or "Access is denied" in str(exc):
            queued = {
                **request,
                "status": "queued_not_executed",
                "blocked_reason": "codex_cli_access_denied",
                "completed_utc": now(),
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "command": command_without_prompt(command),
                "error": str(exc),
                "fallback": "queued_for_manual_or_app_level_teacher_execution",
            }
            append_jsonl(ROOT / policy.get("request_queue", "reports/teacher_request_queue.jsonl"), queued)
            write_json(ROOT / policy.get("queue_last_path", "reports/teacher_queue_last.json"), queued)
            return queued
        return {
            **request,
            "status": "failed",
            "completed_utc": now(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "command": command_without_prompt(command),
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **request,
            "status": "failed",
            "completed_utc": now(),
            "runtime_ms": int((time.perf_counter() - started) * 1000),
            "command": command_without_prompt(command),
            "error": str(exc),
        }


def call_budget_available(policy: dict[str, Any], reason: str) -> bool:
    return bool(call_budget_decision(policy, reason).get("allowed", False))


def local_wall_evidence_decision(policy: dict[str, Any], reason: str, local_evidence: list[str]) -> dict[str, Any]:
    budget = policy.get("budget") or {}
    if not bool(budget.get("requires_local_wall_evidence", False)):
        return {"allowed": True, "reason": "local_wall_evidence_not_required"}
    evidence_required_reasons = {
        "architecture_wall",
        "bigcodebench_execution_wall",
        "benchmark_frontier_design",
        "frontier_exhausted",
        "promotion_gate_blocked",
        "residual_conflict",
    }
    if reason not in evidence_required_reasons:
        return {"allowed": True, "reason": "reason_does_not_require_wall_evidence"}

    learning = read_json(ROOT / "reports" / "learning_scoreboard.json")
    watchdog = read_json(ROOT / "reports" / "autonomy_watchdog.json")
    architecture = read_json(ROOT / "reports" / "architecture_guidance_loop.json")
    candidate = read_json(ROOT / "reports" / "candidate_promotion_gate.json")

    blockers = get_path(learning, ["promotion", "honest_blockers"], []) or []
    public_gap = float(get_path(learning, ["public_transfer", "floor_gap"], 0.0) or 0.0)
    architecture_gap = float(get_path(architecture, ["diagnosis", "floor_gap"], 0.0) or 0.0)
    watchdog_wall = bool(get_path(watchdog, ["summary", "active_frontier_wall"], False))
    candidate_blocked = candidate and not bool(candidate.get("promote"))
    explicit_evidence = bool(local_evidence)
    wall_present = bool(
        explicit_evidence
        or "public_code_pass_rate_below_floor" in blockers
        or public_gap > 0.0
        or architecture_gap > 0.0
        or watchdog_wall
        or candidate_blocked
    )
    return {
        "allowed": wall_present,
        "reason": "local_wall_evidence_present" if wall_present else "local_wall_evidence_missing",
        "public_floor_gap": public_gap,
        "architecture_floor_gap": architecture_gap,
        "watchdog_active_frontier_wall": watchdog_wall,
        "candidate_blocked": candidate_blocked,
        "explicit_local_evidence": explicit_evidence,
        "honest_blockers": blockers,
    }


def call_budget_decision(policy: dict[str, Any], reason: str) -> dict[str, Any]:
    log_path = ROOT / policy.get("log_path", "reports/teacher_calls.jsonl")
    budget = policy.get("budget") or {}
    if policy.get("default_mode") == "apply" and not bool(budget.get("apply_mode_enabled", False)):
        return {
            "allowed": False,
            "reason": "teacher_apply_mode_forbidden",
        }
    if bool(budget.get("proposal_only_no_distillation", True)) and policy.get("default_mode") != "proposal":
        return {
            "allowed": False,
            "reason": "teacher_must_remain_proposal_only",
        }
    reason_overrides = budget.get("reason_overrides") if isinstance(budget.get("reason_overrides"), dict) else {}
    reason_override = reason_overrides.get(reason) if isinstance(reason_overrides.get(reason), dict) else {}
    max_per_day = optional_positive_int(reason_override.get("max_calls_per_day", budget.get("max_calls_per_day", 8)), 8)
    cooldown = int(budget.get("cooldown_seconds", 900))
    critical_cooldown = int(reason_override.get("critical_cooldown_seconds", budget.get("critical_cooldown_seconds", cooldown)))
    critical_reasons = set(str(item) for item in budget.get("critical_reasons_bypass_cooldown", []))
    max_critical = optional_positive_int(
        reason_override.get("max_critical_calls_per_day", budget.get("max_critical_calls_per_day", 0)),
        None,
    )
    calls = read_jsonl(log_path)
    completed = [call for call in calls if call.get("status") == "completed"]
    if not completed:
        return {
            "allowed": True,
            "reason": "first_teacher_call",
            "completed_today": 0,
            "max_calls_per_day": max_per_day,
        }
    now_ts = time.time()
    today = datetime.now(timezone.utc).date().isoformat()
    calls_today = [
        call
        for call in completed
        if str(call.get("created_utc", "")).startswith(today)
    ]
    if max_per_day is not None and len(calls_today) >= max_per_day:
        return {
            "allowed": False,
            "reason": "daily_call_budget_exhausted",
            "completed_today": len(calls_today),
            "max_calls_per_day": max_per_day,
        }
    if reason in critical_reasons:
        critical_today = [
            call
            for call in calls_today
            if call.get("reason_for_call") in critical_reasons
        ]
        if max_critical is not None and len(critical_today) >= max_critical:
            return {
                "allowed": False,
                "reason": "critical_call_budget_exhausted",
                "critical_completed_today": len(critical_today),
                "max_critical_calls_per_day": max_critical,
            }
        same_reason = [
            call
            for call in calls_today
            if call.get("reason_for_call") == reason
        ]
        if same_reason:
            latest_same = max(timestamp(call.get("created_utc")) for call in same_reason)
            elapsed = now_ts - latest_same
            if elapsed < critical_cooldown:
                return {
                    "allowed": False,
                    "reason": "critical_reason_cooldown",
                    "reason_for_call": reason,
                    "seconds_until_allowed": int(critical_cooldown - elapsed),
                    "critical_cooldown_seconds": critical_cooldown,
                    "completed_today": len(calls_today),
                    "critical_completed_today": len(critical_today),
                }
        return {
            "allowed": True,
            "reason": "critical_reason_budget_available",
            "completed_today": len(calls_today),
            "critical_completed_today": len(critical_today),
            "max_calls_per_day": "unlimited" if max_per_day is None else max_per_day,
            "max_critical_calls_per_day": "unlimited" if max_critical is None else max_critical,
            "critical_cooldown_seconds": critical_cooldown,
        }
    latest = max(timestamp(call.get("created_utc")) for call in completed)
    elapsed = now_ts - latest
    if elapsed < cooldown:
        return {
            "allowed": False,
            "reason": "standard_cooldown",
            "seconds_until_allowed": int(cooldown - elapsed),
            "cooldown_seconds": cooldown,
            "completed_today": len(calls_today),
        }
    return {
        "allowed": True,
        "reason": "standard_budget_available",
        "completed_today": len(calls_today),
        "max_calls_per_day": "unlimited" if max_per_day is None else max_per_day,
        "cooldown_seconds": cooldown,
    }


def optional_positive_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null", "unlimited", "off", "false"}:
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else None


def should_log_budget_block(policy: dict[str, Any], request: dict[str, Any]) -> bool:
    log_path = ROOT / policy.get("log_path", "reports/teacher_calls.jsonl")
    budget = policy.get("budget") or {}
    cooldown = int(budget.get("blocked_log_cooldown_seconds", 300))
    rows = read_jsonl(log_path)
    now_ts = timestamp(request.get("created_utc"))
    same_reason_blocks = [
        row
        for row in rows
        if row.get("status") == "blocked_by_teacher_budget"
        and row.get("reason_for_call") == request.get("reason_for_call")
    ]
    if not same_reason_blocks:
        return True
    latest = max(timestamp(row.get("created_utc")) for row in same_reason_blocks)
    return (now_ts - latest) >= cooldown


def write_budget_status(policy: dict[str, Any], result: dict[str, Any]) -> None:
    path = ROOT / policy.get("budget_block_path", "reports/teacher_budget_last.json")
    status = {
        "request_id": result.get("request_id"),
        "created_utc": result.get("created_utc"),
        "completed_utc": result.get("completed_utc"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "reason_for_call": result.get("reason_for_call"),
        "mode": result.get("mode"),
        "policy": result.get("policy"),
        "status": result.get("status"),
        "blocked_reason": result.get("blocked_reason"),
        "budget_decision": result.get("budget_decision"),
        "local_wall_evidence_decision": result.get("local_wall_evidence_decision"),
        "external_inference_calls": 1 if result.get("status") == "completed" else 0,
    }
    write_json(path, status)


def timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def parse_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def codex_usage_limit_info(text: str) -> dict[str, str]:
    lower = text.lower()
    if "usage limit" not in lower and "try again at" not in lower:
        return {}
    marker = "try again at "
    retry_after = ""
    index = lower.find(marker)
    if index >= 0:
        retry_after = text[index + len(marker) :].splitlines()[0].strip().rstrip(".")
    return {
        "blocked_reason": "codex_cli_usage_limit",
        "retry_after_hint": retry_after,
    }


def get_path(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def command_without_prompt(command: list[str]) -> list[str]:
    return ["-" if item == "-" else item for item in command]


def resolve_codex_command(policy: dict[str, Any]) -> str:
    configured = str(policy.get("codex_command", "codex") or "codex")
    mirror = Path(str(policy.get("codex_command_mirror", "") or DEFAULT_CODEX_MIRROR)).expanduser()
    if configured.lower() in {"codex", "codex.exe"}:
        resolved = shutil.which("codex.exe") or shutil.which("codex")
        if resolved:
            return mirror_windowsapps_codex(resolved, policy)
        if mirror.exists():
            return str(mirror)
        return configured
    configured_path = Path(configured).expanduser()
    if configured_path.exists():
        return mirror_windowsapps_codex(str(configured_path), policy)
    resolved = shutil.which(configured)
    if resolved:
        return mirror_windowsapps_codex(resolved, policy)
    return mirror_windowsapps_codex(configured, policy)


def mirror_windowsapps_codex(command: str, policy: dict[str, Any]) -> str:
    """Mirror packaged Codex Desktop CLI binaries out of WindowsApps.

    Windows exposes packaged app binaries on PATH, but child processes can hit
    WinError 5 when launching them directly from C:\Program Files\WindowsApps.
    Copying the signed CLI binary to a normal user-owned directory preserves the
    user's Codex auth/config while making subprocess execution reliable.
    """

    if not command:
        return command
    path = Path(command)
    command_text = str(path)
    if "WindowsApps" not in command_text or path.name.lower() != "codex.exe":
        return command
    mirror = Path(str(policy.get("codex_command_mirror", "") or DEFAULT_CODEX_MIRROR)).expanduser()
    try:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        needs_copy = (
            not mirror.exists()
            or mirror.stat().st_size != path.stat().st_size
            or int(mirror.stat().st_mtime) < int(path.stat().st_mtime)
        )
        if needs_copy:
            shutil.copy2(path, mirror)
        return str(mirror)
    except OSError:
        return command


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
