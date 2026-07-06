#!/usr/bin/env python3
"""Build the governed teacher-distillation manifest.

This script is intentionally conservative. It never calls a teacher and it
never converts proposal-mode teacher output into training rows. Its job is to
make the distillation boundary auditable: retained teacher calls are hashed,
candidate training rows are admitted only when every provenance/license/leakage
/verifier/runtime-serving check is already present, and rejected/proposal rows
remain visible as non-training evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_verifier import evaluate_private_candidates  # noqa: E402
from neural_seed_code_proposer_comparator import render_private_function  # noqa: E402

REQUIRED_ADMISSION_KEYS = [
    "provenance_retained",
    "license_checked",
    "leakage_audited",
    "verifier_accepted",
    "runtime_serving_forbidden",
    "public_benchmark_excluded",
]
FORBIDDEN_TRAINING_SOURCES = {
    "public_benchmark",
    "public_prompt",
    "public_test",
    "public_solution",
    "hidden_test",
    "benchmark_trace",
    "answer_template",
}
ALLOWED_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "mit",
    "project-internal",
}
FORBIDDEN_TEXT_MARKERS = [
    "hidden test",
    "public benchmark answer",
    "public benchmark solution",
    "canonical solution",
    "answer template",
    "benchmark-specific wrapper",
]
CODE_LM_TASK_REQUIRED_KEYS = [
    "task_id",
    "split",
    "category",
    "concept_residual_label",
    "prompt",
    "entry_point",
    "solution_body",
    "tests",
]
NEGATED_MARKER_CONTEXT = [
    "do not",
    "must not",
    "never",
    "forbid",
    "forbids",
    "forbidden",
    "exclude",
    "excluded",
    "without",
    "zero",
    "no ",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/teacher_distillation_policy.json")
    parser.add_argument("--teacher-calls", default="reports/teacher_calls.jsonl")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--ledger-out", default="")
    parser.add_argument("--audit-out", default="reports/teacher_distillation_manifest_audit.json")
    parser.add_argument("--markdown-out", default="reports/teacher_distillation_manifest_audit.md")
    args = parser.parse_args()

    policy_path = resolve(args.policy)
    policy = read_json(policy_path, {})
    manifest_path = resolve(args.manifest_out or str(policy.get("manifest_path", "reports/teacher_distillation_manifest.json")))
    ledger_path = resolve(args.ledger_out or str(policy.get("ledger_path", "reports/teacher_distillation_ledger.jsonl")))
    teacher_calls_path = resolve(args.teacher_calls)
    teacher_calls = read_jsonl(teacher_calls_path)
    report = build_manifest(
        policy=policy,
        policy_path=policy_path,
        teacher_calls_path=teacher_calls_path,
        teacher_calls=teacher_calls,
    )
    preserved_self_rows = preserved_verified_self_rows(read_jsonl(ledger_path), report["ledger_rows"])
    if preserved_self_rows:
        report["ledger_rows"].extend(preserved_self_rows)
        report["summary"]["preserved_verified_self_generated_ledger_rows"] = len(preserved_self_rows)
        report["summary"]["ledger_row_count"] = len(report["ledger_rows"])
        report["manifest"]["summary"]["preserved_verified_self_generated_ledger_rows"] = len(preserved_self_rows)
        report["manifest"]["summary"]["ledger_row_count"] = len(report["ledger_rows"])
    write_json(manifest_path, report["manifest"])
    write_jsonl(ledger_path, report["ledger_rows"])
    write_json(resolve(args.audit_out), report)
    write_text(resolve(args.markdown_out), render_markdown(report, manifest_path))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def preserved_verified_self_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Carry verified-self ledger rows through teacher manifest rebuilds.

    The manifest is rebuilt from retained teacher calls, but the teacher-share
    ledger is also the durable accounting surface for verified student outputs.
    Rebuilding the teacher manifest must not erase those rows.
    """

    existing_ids = {str(row.get("ledger_event_id") or "") for row in new_rows if isinstance(row, dict)}
    preserved: list[dict[str, Any]] = []
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        if not str(row.get("source_kind") or row.get("source") or "").startswith("verified_self"):
            continue
        event_id = str(row.get("ledger_event_id") or "")
        if not event_id or event_id in existing_ids:
            continue
        existing_ids.add(event_id)
        preserved.append(row)
    return preserved


def build_manifest(
    *,
    policy: dict[str, Any],
    policy_path: Path,
    teacher_calls_path: Path,
    teacher_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    retained_proposals: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    admitted_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    blocking_reason_counts: Counter[str] = Counter()
    public_overlap_hits = 0
    holdout_overlap_hits = 0

    for index, row in enumerate(teacher_calls):
        if not isinstance(row, dict):
            continue
        retained = retained_teacher_call(row, index)
        response_json = row.get("response_json") if isinstance(row.get("response_json"), dict) else {}
        candidate = (
            row.get("distillation_training_row")
            or row.get("teacher_distillation_row")
            or response_json.get("distillation_training_row")
            or response_json.get("teacher_distillation_row")
        )
        mode = str(row.get("mode") or "")
        if not isinstance(candidate, dict):
            proposal = {**retained, "admission_status": "proposal_only_not_training"}
            retained_proposals.append(proposal)
            ledger_rows.append(ledger_event(proposal, accepted=False, source_kind="teacher_proposal"))
            blocking_reason_counts["proposal_mode_without_distillation_training_row"] += 1
            continue
        decision = candidate_admission_decision(candidate, row)
        public_overlap_hits += int(decision["public_overlap_hits"])
        holdout_overlap_hits += int(decision["holdout_overlap_hits"])
        if decision["accepted"]:
            admitted = {
                "row_id": stable_id(
                    [
                        str(row.get("request_id") or f"teacher_call_{index}"),
                        str(candidate.get("row_id") or ""),
                        str(candidate.get("target_hash") or ""),
                    ]
                ),
                "source_kind": "teacher_distillation",
                "request_id": row.get("request_id"),
                "teacher_mode": mode,
                "prompt_sha256": row.get("prompt_sha256") or sha256_text(str(row.get("prompt") or "")),
                "response_sha256": sha256_text(str(row.get("response_text") or row.get("stdout_tail") or "")),
                "candidate_sha256": stable_hash(candidate),
                "task_family": candidate.get("task_family"),
                "license_spdx": candidate.get("license_spdx"),
                "external_inference_calls": teacher_call_external_inference_calls(row),
                "training_row": retained_candidate_payload(candidate, decision["local_verifier"]),
                "local_verifier": decision["local_verifier"],
                "provenance": {
                    "teacher_calls": rel(teacher_calls_path),
                    "source_index": index,
                    "policy": rel(policy_path),
                },
                "admission_checks": {key: True for key in REQUIRED_ADMISSION_KEYS},
                "runtime_serving": "forbidden",
            }
            admitted_rows.append(admitted)
            ledger_rows.append(ledger_event(admitted, accepted=True, source_kind="teacher_distillation"))
        else:
            rejected = {
                **retained,
                "admission_status": "rejected_not_training",
                "reject_reasons": decision["reject_reasons"],
                "candidate_sha256": stable_hash(candidate),
                "local_verifier": decision["local_verifier"],
            }
            rejected_candidates.append(rejected)
            ledger_rows.append(ledger_event(rejected, accepted=False, source_kind="teacher_distillation_rejected"))
            for reason in decision["reject_reasons"]:
                blocking_reason_counts[str(reason)] += 1

    accepted_count = len(admitted_rows)
    verifier_pass_count = sum(
        1
        for row in admitted_rows
        if isinstance(row.get("admission_checks"), dict) and row["admission_checks"].get("verifier_accepted") is True
    )
    verifier_pass_rate_applicable = accepted_count > 0
    verifier_pass_rate = (verifier_pass_count / accepted_count) if accepted_count else 0.0
    external_inference_calls = sum(teacher_call_external_inference_calls(row) for row in teacher_calls)
    admission_safety_checks = {
        "provenance_retained": True,
        "license_checked": True,
        "leakage_audited": True,
        "runtime_serving_forbidden": True,
        "public_benchmark_excluded": public_overlap_hits == 0 and all(
            "public_benchmark_content" not in item.get("reject_reasons", []) for item in rejected_candidates
        ),
    }
    admission_checks = {
        **admission_safety_checks,
        "verifier_accepted": accepted_count > 0 and verifier_pass_count == accepted_count,
    }
    admission_safety_checks_clean = all(admission_safety_checks.values())
    summary = {
        "row_count": accepted_count,
        "candidate_source_count": len(teacher_calls),
        "retained_proposal_count": len(retained_proposals),
        "rejected_candidate_count": len(rejected_candidates),
        "distillation_candidate_row_count": len(admitted_rows) + len(rejected_candidates),
        "proposal_without_distillation_row_count": len(retained_proposals),
        "blocking_reason_counts": dict(sorted(blocking_reason_counts.items())),
        "provenance_retained": True,
        "rows_retained": True,
        "license_check": {"ok": True, "allowed": True, "scope": "admitted_rows_only"},
        "public_overlap_hits": public_overlap_hits,
        "holdout_overlap_hits": holdout_overlap_hits,
        "verifier_pass_count": verifier_pass_count,
        "verifier_pass_rate": verifier_pass_rate,
        "verifier_pass_rate_applicable": verifier_pass_rate_applicable,
        "admission_checks": admission_checks,
        "admission_safety_checks": admission_safety_checks,
        "admission_safety_checks_clean": admission_safety_checks_clean,
        "proposal_only_fail_closed": accepted_count == 0 and len(retained_proposals) > 0,
        "ledger_row_count": len(ledger_rows),
        "accepted_ledger_row_count": sum(1 for row in ledger_rows if row.get("accepted") is True),
        "proposal_ledger_row_count": sum(1 for row in ledger_rows if row.get("source_kind") == "teacher_proposal"),
        "rejected_ledger_row_count": sum(1 for row in ledger_rows if row.get("source_kind") == "teacher_distillation_rejected"),
        "teacher_share_metric_ready": True,
        "external_inference_calls": external_inference_calls,
        "public_training_rows_written": 0,
        "next_required_input": (
            "A distillation-mode teacher call containing distillation_training_row with every required "
            "admission check true, verifier acceptance, zero public/holdout overlap, retained provenance, "
            "license evidence, and runtime_serving_forbidden. Proposal-only rows remain non-training evidence."
        ),
    }
    manifest = {
        "policy": "project_theseus_teacher_distillation_manifest_v0",
        "created_utc": now(),
        "source_policy": rel(policy_path),
        "source_teacher_calls": rel(teacher_calls_path),
        "provenance_retained": True,
        "rows_retained": True,
        "license_check": summary["license_check"],
        "public_overlap_hits": public_overlap_hits,
        "holdout_overlap_hits": holdout_overlap_hits,
        "verifier_pass_rate": verifier_pass_rate,
        "verifier_pass_rate_applicable": verifier_pass_rate_applicable,
        "admission_checks": admission_checks,
        "admission_safety_checks": admission_safety_checks,
        "admission_safety_checks_clean": admission_safety_checks_clean,
        "summary": summary,
        "rows": admitted_rows,
        "retained_teacher_proposals": retained_proposals,
        "rejected_candidates": rejected_candidates,
        "boundary": {
            "runtime_serving_external_tokens": "forbidden",
            "proposal_mode_is_not_training": True,
            "public_benchmarks_training_excluded": True,
            "teacher_rows_require_gate": True,
        },
        "score_semantics": (
            "Retained teacher-distillation manifest only. Proposal-mode teacher calls are "
            "hashed and retained but not admitted as training rows. This script never calls "
            "a teacher, never trains, never runs public calibration, and never serves external tokens."
        ),
        "external_inference_calls": external_inference_calls,
    }
    trigger_state = "GREEN" if accepted_count > 0 and verifier_pass_rate >= 0.95 else "YELLOW"
    return {
        "policy": "project_theseus_teacher_distillation_manifest_builder_v0",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": {
            **summary,
            "manifest_ready_for_distillation": trigger_state == "GREEN",
            "proposal_rows_retained_not_training": len(retained_proposals),
            "teacher_rows_admitted": accepted_count,
        },
        "manifest": manifest,
        "ledger_rows": ledger_rows,
        "next_action": next_action(accepted_count, verifier_pass_rate),
        "external_inference_calls": external_inference_calls,
    }


def retained_teacher_call(row: dict[str, Any], index: int) -> dict[str, Any]:
    response_json = row.get("response_json") if isinstance(row.get("response_json"), dict) else {}
    response_text = row.get("response_text") or row.get("stdout_tail") or (
        json.dumps(response_json, sort_keys=True) if response_json else ""
    )
    return {
        "source_index": index,
        "request_id": row.get("request_id"),
        "created_utc": row.get("created_utc"),
        "status": row.get("status"),
        "mode": row.get("mode"),
        "reason_for_call": row.get("reason_for_call") or response_json.get("reason_for_call"),
        "prompt_sha256": row.get("prompt_sha256") or sha256_text(str(row.get("prompt") or "")),
        "response_sha256": sha256_text(str(response_text)),
        "external_inference_calls": teacher_call_external_inference_calls(row),
    }


def teacher_call_external_inference_calls(row: dict[str, Any]) -> int:
    recorded = row.get("external_inference_calls")
    if recorded is not None:
        try:
            return int(recorded)
        except (TypeError, ValueError):
            return 0
    if row.get("status") == "completed" and str(row.get("provider") or "") == "codex_cli":
        return 1
    return 0


def ledger_event(row: dict[str, Any], *, accepted: bool, source_kind: str) -> dict[str, Any]:
    event = {
        "ledger_event_id": stable_id(
            [
                source_kind,
                str(row.get("request_id") or row.get("row_id") or row.get("source_index") or ""),
                str(row.get("candidate_sha256") or row.get("response_sha256") or ""),
            ]
        ),
        "created_utc": now(),
        "source_kind": source_kind,
        "accepted": bool(accepted),
        "training_admission_status": "accepted_by_manifest_pending_gate" if accepted else row.get("admission_status", "not_training"),
        "request_id": row.get("request_id"),
        "prompt_sha256": row.get("prompt_sha256"),
        "response_sha256": row.get("response_sha256"),
        "candidate_sha256": row.get("candidate_sha256"),
        "runtime_serving": "forbidden",
        "public_training_rows_written": 0,
        "external_inference_calls": int(row.get("external_inference_calls") or 0),
        "teacher_call_record_only": not accepted,
    }
    if row.get("reject_reasons"):
        event["reject_reasons"] = row.get("reject_reasons")
    if row.get("admission_checks"):
        event["admission_checks"] = row.get("admission_checks")
    if row.get("local_verifier"):
        event["local_verifier"] = row.get("local_verifier")
    return event


def candidate_admission_decision(candidate: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    checks = candidate.get("admission_checks") if isinstance(candidate.get("admission_checks"), dict) else {}
    reject_reasons: list[str] = []
    mode = str(row.get("mode") or "")
    local_verifier = local_candidate_verifier(candidate, row)
    if mode != "distillation":
        reject_reasons.append("teacher_call_not_distillation_mode")
    for key in REQUIRED_ADMISSION_KEYS:
        if checks.get(key) is not True:
            reject_reasons.append(f"missing_or_false:{key}")
    reject_reasons.extend(local_verifier["reject_reasons"])
    source_kind = str(candidate.get("source_kind") or candidate.get("source") or "")
    if source_kind in FORBIDDEN_TRAINING_SOURCES:
        reject_reasons.append("public_benchmark_content")
    if candidate.get("public_benchmark") is True or candidate.get("public_prompt") is True:
        reject_reasons.append("public_benchmark_content")
    public_hits = int(candidate.get("public_overlap_hits") or 0)
    holdout_hits = int(candidate.get("holdout_overlap_hits") or 0)
    if public_hits:
        reject_reasons.append("public_overlap_hits")
    if holdout_hits:
        reject_reasons.append("holdout_overlap_hits")
    return {
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "public_overlap_hits": public_hits,
        "holdout_overlap_hits": holdout_hits,
        "local_verifier": local_verifier,
    }


def local_candidate_verifier(candidate: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    reject_reasons: list[str] = []
    required_text = ["row_id", "task_family", "input_text", "target_text", "target_hash", "license_spdx"]
    for key in required_text:
        if not str(candidate.get(key) or "").strip():
            reject_reasons.append(f"missing_or_empty:{key}")
    if str(row.get("status") or "") != "completed":
        reject_reasons.append("teacher_call_status_not_completed")
    if str(candidate.get("source_kind") or "") != "teacher_distillation":
        reject_reasons.append("source_kind_not_teacher_distillation")
    target_text = str(candidate.get("target_text") or "")
    target_hash = str(candidate.get("target_hash") or "")
    computed_target_hash = sha256_text(target_text)
    auto_hash_requested = target_hash.strip().lower() in {"sha256:auto", "local_verifier_computes"}
    if target_text and not auto_hash_requested and target_hash != computed_target_hash:
        reject_reasons.append("target_hash_mismatch")
    license_spdx = str(candidate.get("license_spdx") or "").strip().lower()
    if license_spdx not in ALLOWED_LICENSES:
        reject_reasons.append("license_not_allowed")
    if not isinstance(candidate.get("provenance"), dict) or not candidate.get("provenance"):
        reject_reasons.append("missing_provenance")
    if candidate.get("runtime_serving") != "forbidden":
        reject_reasons.append("runtime_serving_not_forbidden")
    if candidate.get("public_benchmark") is not False:
        reject_reasons.append("public_benchmark_flag_not_false")
    if candidate.get("public_prompt") is not False:
        reject_reasons.append("public_prompt_flag_not_false")
    combined_text = "\n".join(
        [
            str(candidate.get("input_text") or ""),
            str(candidate.get("target_text") or ""),
            json.dumps(candidate.get("provenance") or {}, sort_keys=True),
        ]
    ).lower()
    for marker in FORBIDDEN_TEXT_MARKERS:
        if forbidden_marker_is_payload(combined_text, marker):
            reject_reasons.append(f"forbidden_text_marker:{marker.replace(' ', '_')}")
    code_lm = verify_code_lm_task(candidate)
    reject_reasons.extend(code_lm["reject_reasons"])
    return {
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "computed_target_hash": computed_target_hash,
        "claimed_target_hash": target_hash,
        "auto_hash_requested": auto_hash_requested,
        "code_lm_task": code_lm,
        "checks": {
            "row_shape_present": not any(reason.startswith("missing_or_empty:") for reason in reject_reasons),
            "target_hash_matches": "target_hash_mismatch" not in reject_reasons,
            "license_allowed": "license_not_allowed" not in reject_reasons,
            "provenance_present": "missing_provenance" not in reject_reasons,
            "runtime_serving_forbidden": "runtime_serving_not_forbidden" not in reject_reasons,
            "public_flags_clean": "public_benchmark_flag_not_false" not in reject_reasons
            and "public_prompt_flag_not_false" not in reject_reasons,
            "forbidden_markers_absent": not any(reason.startswith("forbidden_text_marker:") for reason in reject_reasons),
            "code_lm_task_execution_verified": code_lm["accepted"],
            "code_lm_task_present": code_lm["present"],
        },
}


def verify_code_lm_task(candidate: dict[str, Any]) -> dict[str, Any]:
    """Verify an optional private code-LM task carried by a teacher row.

    Distillation rows are only admitted as teacher training rows when they carry
    a directly executable ``code_lm_task`` payload. Generic teacher output stays
    retained as non-training evidence; it cannot satisfy the flywheel goal's
    verifier-accepted training-row requirement.
    """

    task = normalize_code_lm_task(extract_code_lm_task(candidate))
    if not isinstance(task, dict):
        return {
            "present": False,
            "accepted": False,
            "reject_reasons": ["code_lm_task_missing"],
            "score_semantics": "generic distillation row retained as non-training evidence only",
        }

    reject_reasons: list[str] = []
    for key in CODE_LM_TASK_REQUIRED_KEYS:
        if not str(task.get(key) or "").strip():
            reject_reasons.append(f"code_lm_task_missing_or_empty:{key}")
    if solution_body_contains_function_def(str(task.get("solution_body") or "")):
        reject_reasons.append("code_lm_task_solution_body_must_be_function_body_not_def")
    if str(task.get("split") or "") not in {"train", "private_train"}:
        reject_reasons.append("code_lm_task_split_not_train")
    if task.get("public_benchmark") is not False:
        reject_reasons.append("code_lm_task_public_benchmark_flag_not_false")
    if task.get("public_prompt") not in {False, None}:
        reject_reasons.append("code_lm_task_public_prompt_flag_not_false")
    if task.get("public_tests_included") or task.get("public_benchmark_solutions_included"):
        reject_reasons.append("code_lm_task_public_payload_flagged")

    family = str(task.get("concept_residual_label") or task.get("category") or "")
    holdout = current_strict_holdout_families()
    if family and family in holdout:
        reject_reasons.append("code_lm_task_family_in_strict_holdout")

    verifier_summary: dict[str, Any] = {
        "attempted": False,
        "trained_passed": 0,
        "trained_pass_rate": 0.0,
        "residual_count": 1,
    }
    if not reject_reasons:
        task_for_eval = dict(task)
        task_for_eval["split"] = "eval"
        code = render_private_function(task_for_eval, str(task.get("solution_body") or ""))
        candidate_row = {
            "task_id": task_for_eval.get("task_id"),
            "phase": "private_eval",
            "rank": 1,
            "rank_score": 1.0,
            "candidate_generation_mode": "teacher_distillation_code_lm_solution",
            "code": code,
            "candidate_sha256": sha256_text(code),
            "substrate_arm": "teacher_distillation",
            "benchmark_promotion_eligible": False,
            "external_inference_calls": 0,
        }
        verifier_summary = evaluate_private_candidates([task_for_eval], [candidate_row])
        verifier_summary["attempted"] = True
        if int(verifier_summary.get("trained_passed") or 0) < 1:
            reject_reasons.append("code_lm_solution_failed_private_execution_verifier")

    return {
        "present": True,
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "task_id": task.get("task_id"),
        "family": family,
        "holdout_families": sorted(holdout),
        "strict_holdout_excluded": family not in holdout,
        "private_execution_verifier": bounded_verifier(verifier_summary),
        "score_semantics": (
            "Optional teacher-distillation code-LM payload. It is private train "
            "pressure only and must pass the local execution verifier; held-out "
            "family payloads are rejected so family-disjoint eval remains blind."
        ),
    }


def extract_code_lm_task(candidate: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("code_lm_task", "private_code_lm_task"):
        value = candidate.get(key)
        if isinstance(value, dict):
            return value
    target_text = str(candidate.get("target_text") or "").strip()
    if not target_text:
        return None
    try:
        parsed = json.loads(target_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    for key in ("code_lm_task", "private_code_lm_task"):
        value = parsed.get(key)
        if isinstance(value, dict):
            return value
    if all(key in parsed for key in ("prompt", "entry_point", "solution_body", "tests")):
        return parsed
    return None


def normalize_code_lm_task(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(task, dict):
        return None
    out = dict(task)
    if isinstance(out.get("solution_body"), str):
        out["solution_body"] = "\n".join(
            line.rstrip()
            for line in textwrap.dedent(str(out.get("solution_body") or "")).strip().splitlines()
        )
    tests = out.get("tests")
    if isinstance(tests, list):
        out["tests"] = "\n".join(str(item) for item in tests if str(item).strip()) + "\n"
    return out


def solution_body_contains_function_def(body: str) -> bool:
    try:
        parsed = ast.parse(str(body or ""))
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) for node in parsed.body)


def current_strict_holdout_families() -> set[str]:
    report = read_json(ROOT / "reports" / "neural_seed_token_decoder_comparator_strict_body_tokens.json", {})
    summary = {}
    if isinstance(report.get("family_disjoint_eval"), dict):
        summary = report["family_disjoint_eval"].get("summary") if isinstance(report["family_disjoint_eval"].get("summary"), dict) else {}
    holdout = summary.get("holdout_families") if isinstance(summary.get("holdout_families"), list) else []
    return {str(item) for item in holdout if str(item)}


def bounded_verifier(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempted": bool(value.get("attempted")),
        "eval_task_count": value.get("eval_task_count"),
        "trained_passed": value.get("trained_passed"),
        "trained_pass_rate": value.get("trained_pass_rate"),
        "trained_rank1_passed": value.get("trained_rank1_passed"),
        "trained_pass_if_any_passed": value.get("trained_pass_if_any_passed"),
        "residual_count": value.get("residual_count"),
        "private_verification": value.get("private_verification"),
        "residuals": value.get("residuals", [])[:2] if isinstance(value.get("residuals"), list) else [],
    }


def forbidden_marker_is_payload(text: str, marker: str) -> bool:
    """Return true only when a marker appears as payload, not as a guardrail."""
    start = 0
    found = False
    while True:
        index = text.find(marker, start)
        if index < 0:
            return found
        found = True
        before = text[max(0, index - 96) : index]
        if any(token in before for token in NEGATED_MARKER_CONTEXT):
            start = index + len(marker)
            found = False
            continue
        return True


def retained_candidate_payload(candidate: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    retained = {
        "row_id": candidate.get("row_id"),
        "source_kind": candidate.get("source_kind"),
        "task_family": candidate.get("task_family"),
        "input_text": candidate.get("input_text"),
        "target_text": candidate.get("target_text"),
        "target_hash": verifier.get("computed_target_hash") or candidate.get("target_hash"),
        "license_spdx": candidate.get("license_spdx"),
        "provenance": candidate.get("provenance"),
        "runtime_serving": candidate.get("runtime_serving"),
        "public_benchmark": candidate.get("public_benchmark"),
        "public_prompt": candidate.get("public_prompt"),
        "public_overlap_hits": candidate.get("public_overlap_hits"),
        "holdout_overlap_hits": candidate.get("holdout_overlap_hits"),
    }
    code_lm_task = normalize_code_lm_task(extract_code_lm_task(candidate))
    code_lm_verifier = verifier.get("code_lm_task") if isinstance(verifier.get("code_lm_task"), dict) else {}
    if isinstance(code_lm_task, dict) and code_lm_verifier.get("accepted") is True:
        retained["code_lm_task"] = code_lm_task
        retained["code_lm_task_verifier"] = code_lm_verifier
    return retained


def next_action(accepted_count: int, verifier_pass_rate: float) -> str:
    if accepted_count <= 0:
        return (
            "Teacher proposal calls are retained as non-training evidence. Keep distillation "
            "locked until a distillation-mode row has provenance, license, leakage, verifier, "
            "runtime-serving, and public-exclusion checks."
        )
    if verifier_pass_rate < 0.95:
        return "Keep distillation locked until retained candidate rows clear the verifier pass-rate floor."
    return "Manifest rows are ready for the separate teacher distillation gate; runtime serving remains forbidden."


def render_markdown(report: dict[str, Any], manifest_path: Path) -> str:
    summary = report.get("summary", {})
    return "\n".join(
        [
            "# Teacher Distillation Manifest Audit",
            "",
            f"- trigger_state: `{report.get('trigger_state')}`",
            f"- manifest: `{rel(manifest_path)}`",
            f"- teacher_rows_admitted: `{summary.get('teacher_rows_admitted')}`",
            f"- proposal_rows_retained_not_training: `{summary.get('proposal_rows_retained_not_training')}`",
            f"- distillation_candidate_row_count: `{summary.get('distillation_candidate_row_count')}`",
            f"- blocking_reason_counts: `{summary.get('blocking_reason_counts')}`",
            f"- verifier_pass_rate: `{summary.get('verifier_pass_rate')}`",
            f"- public_overlap_hits: `{summary.get('public_overlap_hits')}`",
            f"- external_inference_calls: `{report.get('external_inference_calls')}`",
            f"- next_action: {report.get('next_action')}",
            "",
        ]
    )


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def stable_id(parts: list[str]) -> str:
    return "teacher_distill_" + sha256_text("\n".join(parts))[:24]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
