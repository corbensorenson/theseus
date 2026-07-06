#!/usr/bin/env python3
"""Governed teacher/self-generated code-LM row admission helpers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

from neural_seed_code_proposer_comparator import (  # noqa: E402
    dict_or_empty,
    get_path,
    rel,
    resolve,
    stable_hash,
)
from neural_seed_visible_source import strict_disjoint_family_key  # noqa: E402


def load_governed_teacher_code_lm_training_rows(config: dict[str, Any]) -> dict[str, Any]:
    safety = dict_or_empty(config.get("safety"))
    teacher_cfg = dict_or_empty(config.get("teacher_distillation"))
    enabled = bool(teacher_cfg.get("enabled", safety.get("teacher_distillation_allowed", False)))
    manifest_path = resolve(str(teacher_cfg.get("manifest") or "reports/teacher_distillation_manifest.json"))
    gate_path = resolve(str(teacher_cfg.get("gate") or "reports/teacher_distillation_gate.json"))
    max_rows = int(teacher_cfg.get("max_code_lm_rows") or 64)
    gate_report = read_json(gate_path)
    manifest = read_json(manifest_path)
    manifest_summary = dict_or_empty(manifest.get("summary"))
    strict_holdout = strict_report_holdout_families()
    accepted_rows = manifest.get("rows") if isinstance(manifest.get("rows"), list) else []
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for admitted in accepted_rows:
        if not isinstance(admitted, dict):
            continue
        training_row = admitted.get("training_row") if isinstance(admitted.get("training_row"), dict) else {}
        code_task = training_row.get("code_lm_task") if isinstance(training_row.get("code_lm_task"), dict) else None
        verifier = training_row.get("code_lm_task_verifier") if isinstance(training_row.get("code_lm_task_verifier"), dict) else {}
        if not isinstance(code_task, dict):
            continue
        utility_decision = teacher_code_lm_utility_decision(code_task, admitted, teacher_cfg)
        if not utility_decision["accepted"]:
            rejected.append(utility_decision)
            continue
        decision = teacher_code_lm_training_row_decision(code_task, admitted, strict_holdout)
        if not decision["accepted"]:
            rejected.append(decision)
            continue
        row = dict(code_task)
        row["split"] = "train"
        row["source_kind"] = "teacher_distillation"
        row["teacher_generated"] = True
        row["teacher_manifest_row_id"] = admitted.get("row_id")
        row["teacher_request_id"] = admitted.get("request_id")
        row["teacher_candidate_sha256"] = admitted.get("candidate_sha256")
        row["external_inference_calls"] = int(admitted.get("external_inference_calls") or 0)
        row["public_benchmark"] = False
        row["public_prompt"] = False
        row["public_training_rows"] = 0
        row["teacher_code_lm_task_verifier"] = verifier
        rows.append(row)
        if len(rows) >= max_rows:
            break
    holdout_rows = [
        row
        for row in rows
        if strict_disjoint_family_key(row, "concept_residual_label") in strict_holdout
    ]
    external_calls = sum(int(row.get("external_inference_calls") or 0) for row in rows)
    summary = {
        "enabled": enabled,
        "manifest": rel(manifest_path),
        "gate": rel(gate_path),
        "gate_green": gate_report.get("trigger_state") == "GREEN" and bool(gate_report.get("distillation_allowed", False)),
        "manifest_present": bool(manifest),
        "manifest_row_count": int(manifest_summary.get("row_count") or len(accepted_rows) or 0),
        "manifest_safety_clean": bool(manifest_summary.get("admission_safety_checks_clean")),
        "manifest_verifier_pass_rate": float(manifest_summary.get("verifier_pass_rate") or 0.0),
        "public_overlap_hits": int(manifest_summary.get("public_overlap_hits") or manifest.get("public_overlap_hits") or 0),
        "holdout_overlap_hits": int(manifest_summary.get("holdout_overlap_hits") or manifest.get("holdout_overlap_hits") or 0),
        "accepted_code_lm_training_rows": len(rows) if enabled else 0,
        "available_code_lm_training_rows": len(rows),
        "rejected_code_lm_training_rows": len(rejected),
        "rejected_examples": rejected[:8],
        "utility_quarantine": teacher_code_lm_utility_quarantine_summary(teacher_cfg),
        "strict_holdout_families": sorted(strict_holdout),
        "holdout_family_code_lm_training_rows": len(holdout_rows),
        "external_inference_calls": external_calls if enabled else 0,
        "public_training_rows": 0,
        "runtime_serving_external_tokens": "forbidden",
        "visible_to_generation": "prompt_and_entry_point_only_after_row_admission",
        "score_semantics": (
            "Only manifest-admitted private code_lm_task rows are appended to training. "
            "Teacher provenance, target tests, and solution metadata are not visible generation features; "
            "teacher rows are rejected if they target the strict family-disjoint holdout families."
        ),
    }
    return {
        "rows": rows if enabled else [],
        "summary": summary,
    }


def teacher_code_lm_utility_quarantine_summary(teacher_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict_or_empty(teacher_cfg.get("utility_quarantine"))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "blocked_task_ids": sorted(str(item) for item in list_items(cfg.get("blocked_task_ids"))),
        "blocked_manifest_row_ids": sorted(str(item) for item in list_items(cfg.get("blocked_manifest_row_ids"))),
        "blocked_request_ids": sorted(str(item) for item in list_items(cfg.get("blocked_request_ids"))),
        "reason": str(cfg.get("reason") or ""),
        "score_semantics": (
            "Utility quarantine retains verifier-accepted teacher rows in the manifest/ledger but excludes "
            "rows with negative private flywheel evidence from future student training."
        ),
    }


def teacher_code_lm_utility_decision(
    row: dict[str, Any],
    admitted: dict[str, Any],
    teacher_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict_or_empty(teacher_cfg.get("utility_quarantine"))
    if not bool(cfg.get("enabled", False)):
        return {"accepted": True, "row_id": row.get("task_id"), "reasons": []}
    blocked_task_ids = {str(item) for item in list_items(cfg.get("blocked_task_ids"))}
    blocked_manifest_ids = {str(item) for item in list_items(cfg.get("blocked_manifest_row_ids"))}
    blocked_request_ids = {str(item) for item in list_items(cfg.get("blocked_request_ids"))}
    reasons: list[str] = []
    if str(row.get("task_id") or "") in blocked_task_ids:
        reasons.append("teacher_row_utility_quarantined_task_id")
    if str(admitted.get("row_id") or "") in blocked_manifest_ids:
        reasons.append("teacher_row_utility_quarantined_manifest_row_id")
    if str(admitted.get("request_id") or "") in blocked_request_ids:
        reasons.append("teacher_row_utility_quarantined_request_id")
    return {
        "accepted": not reasons,
        "row_id": row.get("task_id"),
        "teacher_manifest_row_id": admitted.get("row_id"),
        "family": row.get("concept_residual_label") or row.get("category"),
        "reasons": sorted(set(reasons)),
        "quarantine_reason": str(cfg.get("reason") or ""),
    }


def list_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, "", {}, ()):
        return []
    return [value]


def record_verified_self_generated_rows(
    config: dict[str, Any],
    no_cheat: dict[str, Any],
    *,
    candidate_manifest_out: str,
) -> dict[str, Any]:
    teacher_cfg = dict_or_empty(config.get("teacher_distillation"))
    self_cfg = dict_or_empty(teacher_cfg.get("verified_self_generation"))
    enabled = bool(
        teacher_cfg.get("record_verified_self_generated_rows")
        or self_cfg.get("enabled")
    )
    if not enabled:
        return {
            "enabled": False,
            "appended_rows": 0,
            "reason": "verified_self_generation_recording_disabled",
        }

    gate_path = resolve(str(teacher_cfg.get("gate") or "reports/teacher_distillation_gate.json"))
    gate_report = read_json(gate_path)
    if gate_report.get("trigger_state") != "GREEN":
        return {
            "enabled": True,
            "appended_rows": 0,
            "reason": "teacher_gate_not_green",
            "gate": rel(gate_path),
            "gate_trigger_state": gate_report.get("trigger_state"),
        }

    policy_path = resolve(str(teacher_cfg.get("policy") or "configs/teacher_distillation_policy.json"))
    policy = read_json(policy_path)
    ledger_path = resolve(
        str(
            self_cfg.get("ledger")
            or teacher_cfg.get("ledger")
            or policy.get("ledger_path")
            or "reports/teacher_distillation_ledger.jsonl"
        )
    )
    existing = read_jsonl(ledger_path)
    existing_ids = {str(row.get("ledger_event_id") or "") for row in existing if isinstance(row, dict)}
    raw_traces = get_path(no_cheat, ["private_verifier", "passed_verification_traces"], [])
    traces = raw_traces if isinstance(raw_traces, list) else []
    max_rows = max(0, int(self_cfg.get("max_rows_per_run") or teacher_cfg.get("max_verified_self_rows_per_run") or 32))
    accepted: list[dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        decision = verified_self_trace_decision(trace)
        if not decision["accepted"]:
            for reason in decision["reject_reasons"]:
                rejected_reasons[str(reason)] += 1
            continue
        event_id = "verified_self_generated_" + stable_hash(
            json.dumps(
                {
                    "candidate_sha256": trace.get("candidate_sha256"),
                    "task_id": trace.get("task_id"),
                    "phase": trace.get("phase"),
                    "substrate_arm": trace.get("substrate_arm"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )[:24]
        if event_id in existing_ids:
            rejected_reasons["duplicate_ledger_event"] += 1
            continue
        existing_ids.add(event_id)
        accepted.append(
            {
                "ledger_event_id": event_id,
                "created_utc": now(),
                "source_kind": "verified_self_generated",
                "accepted": True,
                "training_admission_status": "accepted_by_private_execution_verifier",
                "runtime_serving": "local_only",
                "public_training_rows_written": 0,
                "external_inference_calls": 0,
                "candidate_sha256": trace.get("candidate_sha256"),
                "code_sha256": trace.get("code_sha256"),
                "task_id": trace.get("task_id"),
                "source_task_id": trace.get("source_task_id"),
                "entry_point": trace.get("entry_point"),
                "phase": trace.get("phase"),
                "substrate_arm": trace.get("substrate_arm"),
                "candidate_generation_mode": trace.get("candidate_generation_mode"),
                "candidate_source": trace.get("candidate_source"),
                "verification_stage": trace.get("verification_stage"),
                "verification_reward": trace.get("verification_reward"),
                "verifier": "scripts/code_lm_private_verifier.py::evaluate_private_candidates",
                "candidate_manifest": rel(resolve(candidate_manifest_out)),
                "score_semantics": (
                    "Verified self-generated ledger row. This records a no-cheat eligible student "
                    "candidate that passed the private execution verifier; it is accounting evidence "
                    "for teacher-share trend only and is not a public calibration or promotion claim."
                ),
            }
        )
        if len(accepted) >= max_rows:
            break

    if accepted:
        write_jsonl(ledger_path, [*existing, *accepted])
    return {
        "enabled": True,
        "gate": rel(gate_path),
        "gate_trigger_state": gate_report.get("trigger_state"),
        "ledger": rel(ledger_path),
        "passed_trace_count": len(traces),
        "appended_rows": len(accepted),
        "max_rows_per_run": max_rows,
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
    }


def verified_self_trace_decision(trace: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if trace.get("passed") is not True or trace.get("intended_behavior_passed") is not True:
        reasons.append("private_execution_verifier_not_passed")
    if str(trace.get("phase") or "") != "private_eval":
        reasons.append("not_trained_candidate_phase")
    if str(trace.get("candidate_generation_mode") or "") != "token_level_code_decoder":
        reasons.append("not_token_level_code_decoder")
    if not trace.get("candidate_sha256"):
        reasons.append("missing_candidate_sha256")
    if int(trace.get("external_inference_calls") or 0) != 0:
        reasons.append("external_inference_in_candidate")
    for key in [
        "public_tests_visible_to_generator",
        "public_solutions_visible_to_generator",
        "eval_tests_visible_to_generator",
        "eval_solution_visible_to_generator",
    ]:
        if bool(trace.get(key)):
            reasons.append(f"{key}_true")
    return {"accepted": not reasons, "reject_reasons": sorted(set(reasons))}


def teacher_code_lm_training_row_decision(
    row: dict[str, Any],
    admitted: dict[str, Any],
    strict_holdout: set[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    required = ["task_id", "split", "category", "concept_residual_label", "prompt", "entry_point", "solution_body", "tests"]
    for key in required:
        if not str(row.get(key) or "").strip():
            reasons.append(f"missing_or_empty:{key}")
    if row.get("public_benchmark") is not False:
        reasons.append("public_benchmark_not_false")
    if row.get("public_prompt") not in {False, None}:
        reasons.append("public_prompt_not_false_or_null")
    if int(admitted.get("external_inference_calls") or 0) < 0:
        reasons.append("invalid_external_inference_call_count")
    family = strict_disjoint_family_key(row, "concept_residual_label")
    if family in strict_holdout:
        reasons.append("strict_family_disjoint_holdout_family")
    verifier = dict_or_empty(dict_or_empty(admitted.get("training_row")).get("code_lm_task_verifier"))
    if verifier and verifier.get("accepted") is not True:
        reasons.append("manifest_code_lm_verifier_not_accepted")
    return {
        "accepted": not reasons,
        "row_id": row.get("task_id"),
        "teacher_manifest_row_id": admitted.get("row_id"),
        "family": family,
        "reasons": sorted(set(reasons)),
    }


def strict_report_holdout_families() -> set[str]:
    report = read_json(resolve("reports/neural_seed_token_decoder_comparator_strict_body_tokens.json"))
    family_report = report.get("family_disjoint_eval") if isinstance(report.get("family_disjoint_eval"), dict) else {}
    summary = family_report.get("summary") if isinstance(family_report.get("summary"), dict) else {}
    holdout = summary.get("holdout_families") if isinstance(summary.get("holdout_families"), list) else []
    return {str(item) for item in holdout if str(item)}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [row for row in rows if isinstance(row, dict)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
