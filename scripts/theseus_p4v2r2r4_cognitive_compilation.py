#!/usr/bin/env python3
"""Run P4-v2r2 with complete, information-matched repair artifacts."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import theseus_assistant_p2a as p2a
import theseus_p4_cognitive_compilation as p4
import theseus_p4v2r2_cognitive_compilation as causal
import theseus_p4v2r2r3_cognitive_compilation as predecessor


ROOT = Path(__file__).resolve().parents[1]
POLICY = "project_theseus_p4v2r2r4_cognitive_compilation_run_v1"
INSTRUMENT_POLICY = "project_theseus_p4v2r2r3_prompt_continuity_repair_v1"
INSTRUMENT_STATE = "PROSPECTIVELY_BOUND_AFTER_ZERO_CALL_INFORMATION_CAP_DISCOVERY"
RUNTIME_ATTEMPT_NAMESPACE = "p4v2r2r4_attempt1"
DEFAULT_INSTRUMENT = (
    ROOT / "configs" / "theseus_p4v2r2r3_prompt_continuity_repair.json"
)


def audit_instrument(path: Path) -> dict[str, Any]:
    value = p2a.read_json(path)
    faults: list[str] = []
    if value.get("policy") != INSTRUMENT_POLICY:
        faults.append("instrument_policy_invalid")
    if value.get("state") != INSTRUMENT_STATE:
        faults.append("instrument_state_invalid")
    if value.get("runtime_attempt_namespace") != RUNTIME_ATTEMPT_NAMESPACE:
        faults.append("runtime_attempt_namespace_invalid")
    for path_key, hash_key in (
        ("base_instrument", "base_instrument_sha256"),
        ("zero_call_disposition", "zero_call_disposition_sha256"),
    ):
        owner = p2a.resolve(str(value.get(path_key) or ""))
        if not owner.is_file() or p2a.sha256_file(owner) != str(
            value.get(hash_key) or ""
        ):
            faults.append(f"binding_invalid:{path_key}")
    base_path = p2a.resolve(str(value.get("base_instrument") or ""))
    base_audit = predecessor.audit_instrument(base_path) if base_path.is_file() else {}
    if base_audit.get("trigger_state") != "GREEN":
        faults.append("base_instrument_audit_red")
    disposition_path = p2a.resolve(str(value.get("zero_call_disposition") or ""))
    disposition = (
        p2a.read_json(disposition_path) if disposition_path.is_file() else {}
    )
    custody = p2a.mapping(disposition.get("custody"))
    if (
        disposition.get("scientific_status") != "NO_OBSERVATION"
        or custody.get("candidate_generation_opened") is not False
        or int(custody.get("local_model_calls") or 0) != 0
        or int(custody.get("consumed_tasks") or 0) != 0
        or custody.get("task_pool_reuse_on_repaired_attempt_authorized") is not True
    ):
        faults.append("zero_call_reuse_authority_invalid")
    repair = p2a.mapping(value.get("prompt_continuity_repair"))
    required_true = (
        "complete_first_call_artifact_visible_to_second_call",
        "same_rule_all_learned_arms",
        "exact_complete_repair_prompt_counted_by_pinned_tokenizer",
        "repair_prompt_must_fit_model_declared_context",
    )
    required_false = (
        "causal_mechanism_changed",
        "frozen_model_changed",
        "task_pool_changed",
        "evaluator_changed",
        "control_definition_changed",
        "prior_outputs_reused_or_rescored",
    )
    if any(repair.get(key) is not True for key in required_true):
        faults.append("complete_prompt_continuity_not_required")
    if any(repair.get(key) is not False for key in required_false):
        faults.append("single_repair_boundary_invalid")
    if (
        repair.get("project_selected_first_artifact_character_cap") is not None
        or repair.get("project_selected_first_artifact_token_cap") is not None
        or p2a.mapping(value.get("generation_budget")).get(
            "project_selected_quality_token_cap"
        )
        is not None
    ):
        faults.append("project_selected_quality_or_artifact_cap_present")
    authority = p2a.mapping(value.get("authority"))
    if authority.get("user_or_operator_approval_required") is not False:
        faults.append("user_gate_present")
    if any(
        authority.get(key) is not False
        for key in (
            "external_inference_authorized",
            "teacher_calls_authorized",
            "training_rows_authorized",
            "serving_authorized",
            "D1_authorized",
            "D2_authorized",
            "book_support_promotion_authorized",
        )
    ):
        faults.append("cross_stage_authority_present")
    return {
        "policy": "project_theseus_p4v2r2r3_prompt_continuity_audit_v1",
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if not faults else "RED",
        "faults": sorted(set(faults)),
        "instrument_sha256": p2a.sha256_file(path),
        "runtime_attempt_namespace": RUNTIME_ATTEMPT_NAMESPACE,
        "base_instrument_audit": base_audit,
        "complete_first_call_artifact_retained": True,
        "project_selected_quality_token_cap": None,
    }


def render_full_final_prompt(
    original: str,
    first_output: str,
    first_candidate: dict[str, Any],
    verification: dict[str, Any],
    implicated: set[str],
) -> str:
    feedback = {
        "parse_or_lower_faults": p2a.strings(first_candidate.get("faults")),
        "apply_faults": p2a.strings(verification.get("apply_faults")),
        "visible_verifier_returncode": p2a.mapping(
            verification.get("visible_verifier")
        ).get("returncode"),
        "visible_verifier_stdout_tail": str(
            p2a.mapping(verification.get("visible_verifier")).get("stdout_tail")
            or ""
        )[-1000:],
        "visible_verifier_stderr_tail": str(
            p2a.mapping(verification.get("visible_verifier")).get("stderr_tail")
            or ""
        )[-1000:],
        "dependency_local_repair_obligation_ids": sorted(implicated),
    }
    if original.startswith(causal.SEMANTIC_PROMPT_MARKER):
        return (
            original
            + "\n\n[PROVISIONAL_LABELED_SEMANTIC_IR]\n"
            + str(first_output or "")
            + "\n\n[ACTUAL_TARGET_VALIDATION_FEEDBACK]\n"
            + json.dumps(feedback, sort_keys=True)
            + "\n\nReturn only one complete corrected labeled Semantic IR artifact against the "
            "ORIGINAL snapshot. Repair only units whose obligation references are "
            "within the dependency-local repair set; copy every unrelated unit "
            "byte-for-byte. Do not emit a delta, plan, JSON, Markdown, or commentary."
        )
    return (
        original
        + "\n\n[PROVISIONAL_OUTPUT]\n"
        + str(first_output or "")
        + "\n\n[ACTUAL_TARGET_VALIDATION_FEEDBACK]\n"
        + json.dumps(feedback, sort_keys=True)
        + "\n\nReturn one complete final candidate against the ORIGINAL snapshot in the "
        "same arm grammar. Do not emit a delta against the provisional candidate."
    )


def projected_instrument(overlay: dict[str, Any]) -> dict[str, Any]:
    base = p2a.read_json(p2a.resolve(str(overlay.get("base_instrument") or "")))
    base["runtime_attempt_namespace"] = RUNTIME_ATTEMPT_NAMESPACE
    base["state"] = INSTRUMENT_STATE
    base["prompt_continuity_repair"] = p2a.mapping(
        overlay.get("prompt_continuity_repair")
    )
    base["pre_generation_prompt_cap_disposition"] = {
        "path": str(overlay.get("zero_call_disposition") or ""),
        "sha256": str(overlay.get("zero_call_disposition_sha256") or ""),
    }
    return base


def run_experiment(
    instrument_path: Path,
    task_path: Path,
    *,
    session_factory: Callable[..., Any] = predecessor.persistent_v2_session,
) -> dict[str, Any]:
    audit = audit_instrument(instrument_path)
    if audit.get("trigger_state") != "GREEN":
        return {
            "policy": POLICY,
            "trigger_state": "RED",
            "faults": ["prompt_continuity_instrument_audit_red"],
        }
    overlay = p2a.read_json(instrument_path)
    projected = projected_instrument(overlay)
    original_namespace = predecessor.RUNTIME_ATTEMPT_NAMESPACE
    original_state = predecessor.INSTRUMENT_STATE
    original_render = causal.render_final_prompt
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=ROOT / "runtime" / "control",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(projected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        predecessor.RUNTIME_ATTEMPT_NAMESPACE = RUNTIME_ATTEMPT_NAMESPACE
        predecessor.INSTRUMENT_STATE = INSTRUMENT_STATE
        causal.render_final_prompt = render_full_final_prompt
        report = predecessor.run_experiment(
            temporary,
            task_path,
            session_factory=session_factory,
        )
    finally:
        predecessor.RUNTIME_ATTEMPT_NAMESPACE = original_namespace
        predecessor.INSTRUMENT_STATE = original_state
        causal.render_final_prompt = original_render
        temporary.unlink(missing_ok=True)
    report["policy"] = POLICY
    report["instrument_overlay_sha256"] = p2a.sha256_file(instrument_path)
    report["runtime_attempt_namespace"] = RUNTIME_ATTEMPT_NAMESPACE
    report["prompt_continuity"] = {
        "complete_first_call_artifact_retained": True,
        "same_rule_all_learned_arms": True,
        "project_selected_first_artifact_character_cap": None,
        "project_selected_first_artifact_token_cap": None,
        "physical_context_boundary_disposition": (
            "INVALID_OBSERVATION_INSTRUMENT_INADEQUATE"
        ),
    }
    report["scope"] = (
        "Unchanged ten-task P4 development surface with complete first-call artifact "
        "continuity only; no D1, D2, serving, training, hosted inference, or automatic "
        "book-support authority."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default=p2a.rel(DEFAULT_INSTRUMENT))
    parser.add_argument("--task", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    instrument = p2a.resolve(args.instrument)
    report = (
        audit_instrument(instrument)
        if args.audit_only
        else run_experiment(instrument, p2a.resolve(args.task))
    )
    p2a.write_json(p2a.resolve(args.out), report)
    print(
        json.dumps(
            {
                "trigger_state": report.get("trigger_state"),
                "faults": report.get("faults"),
                "denominators": report.get("denominators"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.get("trigger_state") in {"GREEN", "YELLOW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
