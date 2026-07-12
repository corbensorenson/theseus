#!/usr/bin/env python3
"""Independently qualify the standard causal-transformer survival evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from standard_causal_transformer_survival import build_data_model_scaling_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "standard_causal_transformer_survival.json"
DEFAULT_CONFIG = ROOT / "configs" / "standard_causal_transformer_survival.json"
DEFAULT_CANDIDATES = ROOT / "reports" / "standard_causal_transformer_survival_candidates.jsonl"
DEFAULT_OUT = ROOT / "reports" / "standard_causal_transformer_survival_gate.json"
DEFAULT_TARGET_MODE_COMPARISON = (
    ROOT / "runtime" / "standard_causal_transformer_plan_body_canary" / "matched_comparison.json"
)
DEFAULT_SFT_CONTRACT_REPORT = (
    ROOT / "runtime" / "standard_causal_transformer_contract_clean_body_canary" / "report.json"
)
DEFAULT_SFT_CONTRACT_INTEGRITY = (
    ROOT / "runtime" / "standard_causal_transformer_contract_clean_body_canary" / "integrity.json"
)
DEFAULT_SFT_CONTRACT_BLIND_AUDIT = (
    ROOT / "runtime" / "standard_causal_transformer_contract_clean_body_canary" / "blind_audit.json"
)
DEFAULT_SFT_CONTRACT_CONTROL_REPORT = (
    ROOT / "runtime" / "standard_causal_transformer_body_only_matched_canary" / "report.json"
)
DEFAULT_STATE_MEMORY_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_body_only_current_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_state_semantic_canary",
    "hash_control": ROOT / "runtime" / "standard_causal_transformer_state_hash_control_canary",
    "zero": ROOT / "runtime" / "standard_causal_transformer_state_zero_replay",
    "shuffle": ROOT / "runtime" / "standard_causal_transformer_state_shuffle_replay",
}
DEFAULT_STATE_ROLE_READ_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_body_only_current_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_state_role_read_semantic_canary",
    "hash_control": ROOT / "runtime" / "standard_causal_transformer_state_role_read_hash_control_canary",
    "zero": ROOT / "runtime" / "standard_causal_transformer_state_role_read_zero_replay",
    "shuffle": ROOT / "runtime" / "standard_causal_transformer_state_role_read_shuffle_replay",
}
DEFAULT_STATE_CONTINUATION_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_body_only_continuation",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_state_semantic_continuation",
    "hash_control": ROOT / "runtime" / "standard_causal_transformer_state_hash_control_continuation",
}
DEFAULT_TEACHER_RESIDUAL_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_teacher_body_canary",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_teacher_state_canary",
}
DEFAULT_SEMANTIC_PLAN_HEAD_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_plan_head_body_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_plan_head_semantic",
    "shuffled": ROOT / "runtime" / "standard_causal_transformer_plan_head_shuffled",
}
DEFAULT_ORDERED_PLAN_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_ordered_plan_body_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_ordered_plan_semantic",
    "shuffled": ROOT / "runtime" / "standard_causal_transformer_ordered_plan_shuffled",
    "dropout": ROOT / "runtime" / "standard_causal_transformer_ordered_plan_dropout",
}
DEFAULT_LATENT_ORDERED_PLAN_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_latent_plan_body_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_latent_plan_semantic",
    "shuffled": ROOT / "runtime" / "standard_causal_transformer_latent_plan_shuffled",
    "dropout": ROOT / "runtime" / "standard_causal_transformer_latent_plan_dropout",
}
DEFAULT_SLOT_ORDERED_PLAN_ARM_DIRS = {
    "body_only": ROOT / "runtime" / "standard_causal_transformer_latent_plan_body_control",
    "semantic": ROOT / "runtime" / "standard_causal_transformer_slot_factorized_plan_semantic",
    "shuffled": ROOT / "runtime" / "standard_causal_transformer_slot_factorized_plan_shuffled",
    "dropout": ROOT / "runtime" / "standard_causal_transformer_slot_factorized_plan_dropout",
}
ALLOWED_READ_SET = {"prompt", "entry_point", "callable_signature"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=rel(DEFAULT_REPORT))
    parser.add_argument("--candidates", default=rel(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=rel(DEFAULT_OUT))
    parser.add_argument("--target-mode-comparison", default=rel(DEFAULT_TARGET_MODE_COMPARISON))
    parser.add_argument("--sft-contract-report", default=rel(DEFAULT_SFT_CONTRACT_REPORT))
    parser.add_argument("--sft-contract-integrity", default=rel(DEFAULT_SFT_CONTRACT_INTEGRITY))
    parser.add_argument("--sft-contract-blind-audit", default=rel(DEFAULT_SFT_CONTRACT_BLIND_AUDIT))
    parser.add_argument("--sft-contract-control-report", default=rel(DEFAULT_SFT_CONTRACT_CONTROL_REPORT))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    report_path = resolve(args.report)
    candidates_path = resolve(args.candidates)
    report = read_json(report_path)
    candidates = read_jsonl(candidates_path)
    scaling_contract = build_data_model_scaling_contract(read_json(DEFAULT_CONFIG))
    gate = build_gate(
        report_path,
        candidates_path,
        report,
        candidates,
        target_mode_comparison_path=resolve(args.target_mode_comparison),
        sft_contract_report_path=resolve(args.sft_contract_report),
        sft_contract_integrity_path=resolve(args.sft_contract_integrity),
        sft_contract_blind_audit_path=resolve(args.sft_contract_blind_audit),
        sft_contract_control_report_path=resolve(args.sft_contract_control_report),
        scaling_contract=scaling_contract,
    )
    write_json(resolve(args.out), gate)
    view = {
        "trigger_state": gate["trigger_state"],
        "summary": gate["summary"],
        "hard_gaps": gate["hard_gaps"],
        "adoption_gaps": gate["adoption_gaps"],
    }
    print(json.dumps(view if args.gate else gate, indent=2, sort_keys=True))
    return 2 if gate["trigger_state"] == "RED" else 0


def build_gate(
    report_path: Path,
    candidates_path: Path,
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    target_mode_comparison_path: Path = DEFAULT_TARGET_MODE_COMPARISON,
    sft_contract_report_path: Path = DEFAULT_SFT_CONTRACT_REPORT,
    sft_contract_integrity_path: Path = DEFAULT_SFT_CONTRACT_INTEGRITY,
    sft_contract_blind_audit_path: Path = DEFAULT_SFT_CONTRACT_BLIND_AUDIT,
    sft_contract_control_report_path: Path = DEFAULT_SFT_CONTRACT_CONTROL_REPORT,
    scaling_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    adoption_gaps: list[dict[str, Any]] = []
    scaling_contract = scaling_contract or {
        "training_authorized": False,
        "hard_gaps": ["data_model_scaling_contract_missing"],
    }
    scaling_infrastructure_gaps = [
        gap for gap in scaling_contract.get("hard_gaps") or []
        if gap != "canonical_mixed_corpus_receipt_missing"
    ]
    if scaling_infrastructure_gaps:
        hard_gaps.append(gap(
            "data_model_scaling_contract_invalid",
            {"hard_gaps": scaling_infrastructure_gaps},
        ))
    if scaling_contract.get("training_authorized") is not True:
        adoption_gaps.append(gap(
            "data_model_scaling_contract_not_ready",
            {
                "hard_gaps": scaling_contract.get("hard_gaps") or [],
                "planning_estimate_shortfall_positions": scaling_contract.get("planning_estimate_shortfall_positions"),
            },
            severity="adoption_gap",
        ))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    stage = report.get("stage") if isinstance(report.get("stage"), dict) else {}
    training = report.get("training") if isinstance(report.get("training"), dict) else {}
    architecture = report.get("architecture") if isinstance(report.get("architecture"), dict) else {}
    checkpoint = resolve(str((report.get("artifacts") or {}).get("checkpoint") or ""))
    preference_audit = audit_preference_canary(report.get("preference_canary"), checkpoint)
    hard_gaps.extend(preference_audit["hard_gaps"])
    generation_mode_audit = audit_generation_mode_canary(report.get("generation_mode_canary"))
    hard_gaps.extend(generation_mode_audit["hard_gaps"])
    target_mode_audit = audit_target_mode_comparison(target_mode_comparison_path)
    hard_gaps.extend(target_mode_audit["hard_gaps"])
    sft_contract_audit = audit_sft_contract_admission(
        report_path=sft_contract_report_path,
        integrity_path=sft_contract_integrity_path,
        blind_audit_path=sft_contract_blind_audit_path,
        control_report_path=sft_contract_control_report_path,
    )
    hard_gaps.extend(sft_contract_audit["hard_gaps"])
    state_memory_audit = audit_state_memory_ablation(DEFAULT_STATE_MEMORY_ARM_DIRS)
    hard_gaps.extend(state_memory_audit["hard_gaps"])
    state_role_read_audit = audit_state_memory_ablation(DEFAULT_STATE_ROLE_READ_ARM_DIRS)
    hard_gaps.extend(state_role_read_audit["hard_gaps"])
    state_continuation_audit = audit_state_memory_continuation(DEFAULT_STATE_CONTINUATION_ARM_DIRS)
    hard_gaps.extend(state_continuation_audit["hard_gaps"])
    teacher_residual_audit = audit_teacher_residual_ablation(DEFAULT_TEACHER_RESIDUAL_ARM_DIRS)
    hard_gaps.extend(teacher_residual_audit["hard_gaps"])
    semantic_plan_head_audit = audit_semantic_plan_head_ablation(
        DEFAULT_SEMANTIC_PLAN_HEAD_ARM_DIRS
    )
    hard_gaps.extend(semantic_plan_head_audit["hard_gaps"])
    ordered_plan_audit = audit_ordered_plan_ablation(DEFAULT_ORDERED_PLAN_ARM_DIRS)
    hard_gaps.extend(ordered_plan_audit["hard_gaps"])
    latent_ordered_plan_audit = audit_latent_ordered_plan_ablation(
        DEFAULT_LATENT_ORDERED_PLAN_ARM_DIRS
    )
    hard_gaps.extend(latent_ordered_plan_audit["hard_gaps"])
    slot_ordered_plan_audit = audit_slot_ordered_plan_ablation(
        DEFAULT_SLOT_ORDERED_PLAN_ARM_DIRS
    )
    hard_gaps.extend(slot_ordered_plan_audit["hard_gaps"])

    if not report_path.exists():
        hard_gaps.append(gap("report_missing", {"path": rel(report_path)}))
    if not candidates_path.exists():
        hard_gaps.append(gap("candidate_manifest_missing", {"path": rel(candidates_path)}))
    if report.get("policy") != "project_theseus_standard_causal_transformer_survival_v1":
        hard_gaps.append(gap("report_policy_mismatch", {"policy": report.get("policy")}))
    if architecture.get("family") != "standard_decoder_only_causal_transformer":
        hard_gaps.append(gap("architecture_family_mismatch", {"architecture": architecture}))
    reported_attention_policy = str(
        architecture.get("attention_policy")
        or (architecture.get("config") or {}).get("attention_policy")
        or ""
    )
    if not reported_attention_policy and architecture.get("attention") == "RoPE_grouped_query_causal_attention":
        reported_attention_policy = "causal"
        adoption_gaps.append(
            gap(
                "legacy_report_attention_policy_inferred",
                {"inferred_policy": reported_attention_policy},
                severity="adoption_gap",
            )
        )
    if reported_attention_policy not in {"causal", "prefix_lm"}:
        hard_gaps.append(gap("attention_policy_missing_or_invalid", {"architecture": architecture}))
    if int(architecture.get("parameter_count") or 0) <= 0:
        hard_gaps.append(gap("parameter_count_missing", {"architecture": architecture}))
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        hard_gaps.append(gap("checkpoint_missing_or_empty", {"path": rel(checkpoint)}))
    if not training.get("complete"):
        adoption_gaps.append(gap("training_tranche_incomplete", {"training": training}, severity="adoption_gap"))
    if not training.get("eval_loss_improved"):
        adoption_gaps.append(gap("heldout_lm_loss_not_improved", {"training": training}, severity="adoption_gap"))

    for key in (
        "train_holdout_family_overlap_count",
        "train_eval_prompt_overlap_count",
        "train_eval_body_overlap_count",
        "licensed_pretrain_eval_body_overlap_source_surviving_count",
        "private_hidden_derived_signature_count",
        "eval_hidden_derived_signature_count",
        "public_training_rows",
        "external_inference_calls",
    ):
        if int(stage.get(key) or 0) != 0:
            hard_gaps.append(gap(f"stage_{key}_nonzero", {"observed": stage.get(key)}))
    sequence_partition = (
        stage.get("sequence_partition_audit")
        if isinstance(stage.get("sequence_partition_audit"), dict)
        else {}
    )
    if not sequence_partition:
        adoption_gaps.append(
            gap(
                "legacy_report_sequence_partition_receipt_missing",
                {
                    "required_on_next_report": ["pretrain", "sft", "eval"],
                    "current_code_tests_fail_closed": True,
                },
                severity="adoption_gap",
            )
        )
    else:
        for partition_name in ("pretrain", "sft", "eval"):
            receipt = (
                sequence_partition.get(partition_name)
                if isinstance(sequence_partition.get(partition_name), dict)
                else {}
            )
            if receipt.get("valid") is not True:
                hard_gaps.append(
                    gap(
                        f"{partition_name}_sequence_partition_invalid",
                        {"receipt": receipt},
                    )
                )
    if int(stage.get("unique_semantic_eval_task_count") or 0) != int(
        stage.get("family_disjoint_eval_task_count") or 0
    ):
        hard_gaps.append(
            gap(
                "repeated_semantic_eval_tasks",
                {
                    "unique": stage.get("unique_semantic_eval_task_count"),
                    "reported": stage.get("family_disjoint_eval_task_count"),
                },
            )
        )
    if int(stage.get("unique_semantic_eval_task_count") or 0) < 24:
        hard_gaps.append(
            gap(
                "insufficient_distinct_eval_tasks",
                {"observed": stage.get("unique_semantic_eval_task_count"), "required": 24},
            )
        )
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(report.get(key) or 0) != 0:
            hard_gaps.append(gap(f"report_{key}_nonzero", {"observed": report.get(key)}))

    parse_valid = 0
    task_ids: set[str] = set()
    duplicate_within_task: set[tuple[str, str]] = set()
    cross_task_duplicate_hashes: set[str] = set()
    observed_hash_tasks: dict[str, set[str]] = {}
    observed_body_hash_tasks: dict[str, set[str]] = {}
    family_counts: dict[str, int] = {}
    from candidate_integrity import recompute_candidate_integrity

    for index, row in enumerate(candidates):
        code = str(row.get("code") or "")
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if str(row.get("candidate_sha256") or "") != digest:
            hard_gaps.append(gap("candidate_hash_mismatch", {"index": index}))
        task_id = str(row.get("task_id") or "")
        prior_tasks = observed_hash_tasks.setdefault(digest, set())
        if task_id in prior_tasks:
            duplicate_within_task.add((task_id, digest))
        elif prior_tasks:
            cross_task_duplicate_hashes.add(digest)
        prior_tasks.add(task_id)
        try:
            parsed = ast.parse(code)
            parse_valid += 1
            body_digest = normalized_function_body_hash(parsed)
            body_tasks = observed_body_hash_tasks.setdefault(body_digest, set())
            if task_id not in body_tasks and body_tasks:
                cross_task_duplicate_hashes.add(body_digest)
            body_tasks.add(task_id)
        except SyntaxError as exc:
            hard_gaps.append(gap("candidate_syntax_invalid", {"index": index, "error": str(exc)}))
        if set(row.get("generation_read_set") or []) != ALLOWED_READ_SET:
            hard_gaps.append(
                gap(
                    "generation_read_set_mismatch",
                    {"index": index, "observed": row.get("generation_read_set"), "allowed": sorted(ALLOWED_READ_SET)},
                )
            )
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
            if int(row.get(key) or 0) != 0:
                hard_gaps.append(gap(f"candidate_{key}_nonzero", {"index": index, "observed": row.get(key)}))
        if row.get("fallback_return_used") is not False or row.get("body_template_selected") is not False:
            hard_gaps.append(gap("candidate_fallback_or_template_boundary_fault", {"index": index}))
        integrity = recompute_candidate_integrity(row)
        family = str(integrity.get("recomputed_candidate_family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        if family not in {"transformer_hybrid", "learned_full_body_token"}:
            adoption_gaps.append(
                gap(
                    "candidate_not_independently_promotion_grade",
                    {"index": index, "integrity": integrity},
                    severity="adoption_gap",
                )
            )
        task_ids.add(task_id)

    if duplicate_within_task:
        hard_gaps.append(gap("duplicate_candidate_code_within_task", {"count": len(duplicate_within_task)}))
    if cross_task_duplicate_hashes:
        adoption_gaps.append(
            gap(
                "cross_task_candidate_collapse",
                {"duplicate_code_hash_count": len(cross_task_duplicate_hashes)},
                severity="adoption_gap",
            )
        )
    if int(summary.get("candidate_count") or 0) != len(candidates):
        hard_gaps.append(
            gap("candidate_count_mismatch", {"reported": summary.get("candidate_count"), "actual": len(candidates)})
        )
    if int(summary.get("syntax_valid_candidate_count") or 0) != parse_valid:
        hard_gaps.append(
            gap("syntax_count_mismatch", {"reported": summary.get("syntax_valid_candidate_count"), "actual": parse_valid})
        )
    passed = int(summary.get("model_only_passed_task_count") or 0)
    if passed <= 0:
        adoption_gaps.append(
            gap(
                "model_only_private_behavior_not_above_zero",
                {"passed": passed, "eval_tasks": summary.get("family_disjoint_eval_task_count")},
                severity="adoption_gap",
            )
        )
    adoption_state = "QUALIFIED" if not adoption_gaps and not hard_gaps else "NOT_ADOPTED"
    return {
        "policy": "project_theseus_standard_causal_transformer_survival_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trigger_state": "GREEN" if not hard_gaps else "RED",
        "summary": {
            "candidate_count": len(candidates),
            "candidate_task_count": len(task_ids - {""}),
            "syntax_valid_candidate_count": parse_valid,
            "independent_family_counts": family_counts,
            "model_only_passed_task_count": passed,
            "hard_gap_count": len(hard_gaps),
            "adoption_gap_count": len(adoption_gaps),
            "adoption_state": adoption_state,
            "checkpoint": rel(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else "",
            "preference_canary_state": preference_audit["state"],
            "preference_adoption_state": preference_audit["adoption_state"],
            "preference_reward_behavior_delta": preference_audit["reward_behavior_delta"],
            "generation_mode_canary_state": generation_mode_audit["state"],
            "generation_mode_adoption_state": generation_mode_audit["adoption_state"],
            "generation_mode_speedup": generation_mode_audit["speedup"],
            "target_mode_comparison_state": target_mode_audit["state"],
            "target_mode_adoption_state": target_mode_audit["adoption_state"],
            "target_mode_comparison_sha256": target_mode_audit["comparison_sha256"],
            "target_mode_adoption_rejection_reasons": target_mode_audit["adoption_rejection_reasons"],
            "target_mode_deltas": target_mode_audit["deltas"],
            "sft_contract_admission_state": sft_contract_audit["state"],
            "sft_contract_adoption_state": sft_contract_audit["adoption_state"],
            "sft_contract_deltas": sft_contract_audit["deltas"],
            "state_memory_ablation_state": state_memory_audit["state"],
            "state_memory_adoption_state": state_memory_audit["adoption_state"],
            "state_memory_rejection_reasons": state_memory_audit["adoption_rejection_reasons"],
            "state_memory_deltas": state_memory_audit["deltas"],
            "state_role_read_ablation_state": state_role_read_audit["state"],
            "state_role_read_adoption_state": state_role_read_audit["adoption_state"],
            "state_role_read_rejection_reasons": state_role_read_audit["adoption_rejection_reasons"],
            "state_role_read_deltas": state_role_read_audit["deltas"],
            "state_continuation_state": state_continuation_audit["state"],
            "state_continuation_adoption_state": state_continuation_audit["adoption_state"],
            "state_continuation_rejection_reasons": state_continuation_audit["adoption_rejection_reasons"],
            "state_continuation_deltas": state_continuation_audit["deltas"],
            "teacher_residual_state": teacher_residual_audit["state"],
            "teacher_residual_adoption_state": teacher_residual_audit["adoption_state"],
            "teacher_residual_rejection_reasons": teacher_residual_audit[
                "adoption_rejection_reasons"
            ],
            "teacher_residual_deltas": teacher_residual_audit["deltas"],
            "semantic_plan_head_state": semantic_plan_head_audit["state"],
            "semantic_plan_head_adoption_state": semantic_plan_head_audit[
                "adoption_state"
            ],
            "semantic_plan_head_rejection_reasons": semantic_plan_head_audit[
                "adoption_rejection_reasons"
            ],
            "semantic_plan_head_deltas": semantic_plan_head_audit["deltas"],
            "ordered_plan_state": ordered_plan_audit["state"],
            "ordered_plan_adoption_state": ordered_plan_audit["adoption_state"],
            "ordered_plan_rejection_reasons": ordered_plan_audit[
                "adoption_rejection_reasons"
            ],
            "ordered_plan_deltas": ordered_plan_audit["deltas"],
            "latent_ordered_plan_state": latent_ordered_plan_audit["state"],
            "latent_ordered_plan_adoption_state": latent_ordered_plan_audit[
                "adoption_state"
            ],
            "latent_ordered_plan_rejection_reasons": latent_ordered_plan_audit[
                "adoption_rejection_reasons"
            ],
            "latent_ordered_plan_deltas": latent_ordered_plan_audit["deltas"],
            "slot_ordered_plan_state": slot_ordered_plan_audit["state"],
            "slot_ordered_plan_adoption_state": slot_ordered_plan_audit[
                "adoption_state"
            ],
            "slot_ordered_plan_rejection_reasons": slot_ordered_plan_audit[
                "adoption_rejection_reasons"
            ],
            "slot_ordered_plan_deltas": slot_ordered_plan_audit["deltas"],
            "data_model_scaling_contract_state": scaling_contract.get("state"),
            "data_model_scaling_training_authorized": scaling_contract.get("training_authorized") is True,
            "data_model_scaling_shortfall_positions": scaling_contract.get("planning_estimate_shortfall_positions"),
        },
        "hard_gaps": hard_gaps,
        "adoption_gaps": adoption_gaps,
        "rules": {
            "integrity": "candidate family and syntax are independently recomputed; candidate flags are not trusted",
            "adoption": "integrity can be GREEN for a valid negative, but route adoption requires complete training, improved heldout loss, and model-only family-disjoint behavior above zero",
            "no_credit": "templates, renderers, routers, tools, repairs, and fallback returns cannot satisfy this gate",
        },
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
        "target_mode_comparison": target_mode_audit["receipt"],
        "sft_contract_admission_ablation": sft_contract_audit["receipt"],
        "state_memory_ablation": state_memory_audit["receipt"],
        "state_role_read_ablation": state_role_read_audit["receipt"],
        "state_continuation": state_continuation_audit["receipt"],
        "teacher_residual_ablation": teacher_residual_audit["receipt"],
        "semantic_plan_head_ablation": semantic_plan_head_audit["receipt"],
        "ordered_plan_ablation": ordered_plan_audit["receipt"],
        "latent_ordered_plan_ablation": latent_ordered_plan_audit["receipt"],
        "slot_ordered_plan_ablation": slot_ordered_plan_audit["receipt"],
        "data_model_scaling_contract": scaling_contract,
    }


def audit_semantic_plan_head_ablation(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    """Qualify true-label planning against shuffled labels and a body-only model."""

    required_arms = {"body_only", "semantic", "shuffled"}
    required_files = (
        "report.json",
        "config.json",
        "candidates.jsonl",
        "integrity.json",
        "blind_audit.json",
    )
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED"},
            "hard_gaps": [gap("semantic_plan_head_arm_set_mismatch", {"observed": sorted(arm_dirs)})],
        }
    missing = [
        rel(directory / filename)
        for directory in arm_dirs.values()
        for filename in required_files
        if not (directory / filename).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }

    reports = {name: read_json(directory / "report.json") for name, directory in arm_dirs.items()}
    configs = {name: read_json(directory / "config.json") for name, directory in arm_dirs.items()}
    integrities = {name: read_json(directory / "integrity.json") for name, directory in arm_dirs.items()}
    blind = {name: read_json(directory / "blind_audit.json") for name, directory in arm_dirs.items()}
    hard_gaps: list[dict[str, Any]] = []

    def matched_config(value: dict[str, Any]) -> dict[str, Any]:
        copied = json.loads(json.dumps(value))
        model = copied.get("model") if isinstance(copied.get("model"), dict) else {}
        model.pop("semantic_plan_feature_count", None)
        model.pop("semantic_plan_separator_token_id", None)
        copied.pop("semantic_plan_training", None)
        return copied

    configs_equal = len(
        {json.dumps(matched_config(value), sort_keys=True) for value in configs.values()}
    ) == 1
    expected_modes = {"body_only": "none", "semantic": "semantic", "shuffled": "shuffled"}
    modes_correct = all(
        str((configs[name].get("semantic_plan_training") or {}).get("label_mode") or "none")
        == mode
        for name, mode in expected_modes.items()
    )
    feature_counts = {
        name: int((config.get("model") or {}).get("semantic_plan_feature_count") or 0)
        for name, config in configs.items()
    }
    feature_contract_correct = (
        feature_counts["body_only"] == 0
        and feature_counts["semantic"] == feature_counts["shuffled"] > 0
    )
    stage_signatures = {
        str((report.get("stage") or {}).get("stage_signature") or "")
        for report in reports.values()
    }

    def body_positions(report: dict[str, Any]) -> int:
        return sum(
            int(row.get("optimizer_body_positions_consumed") or 0)
            for row in (report.get("training") or {}).get("phases", [])
            if isinstance(row, dict)
        )

    def plan_phase(report: dict[str, Any]) -> dict[str, Any]:
        return next(
            (
                row
                for row in (report.get("training") or {}).get("phases", [])
                if isinstance(row, dict)
                and str(row.get("phase") or "").startswith("prompt_signature_body_sft")
            ),
            {},
        )

    exposures = {name: body_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(exposures.values())) == 1 and next(iter(exposures.values())) > 0
    parameters = {
        name: int((report.get("architecture") or {}).get("parameter_count") or 0)
        for name, report in reports.items()
    }
    plan_parameter_match = (
        parameters["semantic"] == parameters["shuffled"] > parameters["body_only"] > 0
    )
    plan_receipts = {
        name: (plan_phase(reports[name]).get("semantic_plan_labels") or {})
        for name in required_arms
    }
    positive_weight_receipts = {
        name: (plan_phase(reports[name]).get("semantic_plan_positive_weights") or {})
        for name in required_arms
    }
    labels_mass_matched = (
        int(plan_receipts["semantic"].get("positive_label_count") or 0)
        == int(plan_receipts["shuffled"].get("positive_label_count") or -1)
        > 0
        and int(plan_receipts["semantic"].get("fixed_point_count") or 0)
        == int(plan_receipts["semantic"].get("row_count") or -1)
        and int(plan_receipts["shuffled"].get("fixed_point_count", -1)) == 0
        and str(plan_receipts["semantic"].get("label_sha256") or "")
        != str(plan_receipts["shuffled"].get("label_sha256") or "")
        and str(positive_weight_receipts["semantic"].get("weight_sha256") or "")
        == str(positive_weight_receipts["shuffled"].get("weight_sha256") or "")
        != ""
    )
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        and resolve(str(value.get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((value.get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name, value in integrities.items()
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    source_contract_clean = all(
        ((reports[name].get("training") or {}).get("semantic_plan_eval_after") or {}).get(
            "source_contract"
        )
        == "prompt_signature_only_before_separator"
        and ((reports[name].get("training") or {}).get("semantic_plan_eval_after") or {}).get(
            "target_labels_visible_at_inference"
        )
        is False
        for name in ("semantic", "shuffled")
    )
    checks = {
        "configs_equal_except_plan_policy": configs_equal,
        "label_modes_expected": modes_correct,
        "feature_contract_expected": feature_contract_correct,
        "stage_signature_equal_and_present": len(stage_signatures) == 1 and "" not in stage_signatures,
        "optimizer_body_exposure_equal": exposure_equal,
        "plan_parameter_counts_equal": plan_parameter_match,
        "semantic_and_shuffled_label_mass_equal": labels_mass_matched,
        "source_only_inference_contract": source_contract_clean,
        "integrity_clean_and_bound": integrity_clean,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"semantic_plan_head_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        plan_eval = (report.get("training") or {}).get("semantic_plan_eval_after") or {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "eval_loss_after": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "plan_micro_f1": float(plan_eval.get("micro_f1") or 0.0),
            "plan_binary_cross_entropy": float(plan_eval.get("binary_cross_entropy") or 0.0),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    arm_metrics = {name: metrics(report) for name, report in reports.items()}
    semantic = arm_metrics["semantic"]
    shuffled = arm_metrics["shuffled"]
    body_only = arm_metrics["body_only"]
    plan_signal_causal = (
        semantic["plan_micro_f1"] > shuffled["plan_micro_f1"]
        and semantic["plan_binary_cross_entropy"] < shuffled["plan_binary_cross_entropy"]
    )
    behavior_gain = semantic["passed_task_count"] > max(
        body_only["passed_task_count"], shuffled["passed_task_count"]
    )
    coverage_non_regressed = semantic["candidate_task_count"] >= max(
        body_only["candidate_task_count"], shuffled["candidate_task_count"]
    )
    reward_non_regressed = semantic["mean_verification_reward"] >= max(
        body_only["mean_verification_reward"], shuffled["mean_verification_reward"]
    )
    adopted = (
        not hard_gaps
        and plan_signal_causal
        and behavior_gain
        and coverage_non_regressed
        and reward_non_regressed
    )
    rejection_reasons: list[str] = []
    if not plan_signal_causal:
        rejection_reasons.append("semantic_labels_did_not_beat_shuffled_labels")
    if not behavior_gain:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if not coverage_non_regressed:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if not reward_non_regressed:
        rejection_reasons.append("mean_verification_reward_regressed")
    deltas = {
        comparator: {
            key: round(float(semantic[key]) - float(arm_metrics[comparator][key]), 8)
            for key in semantic
        }
        for comparator in ("body_only", "shuffled")
    }
    artifacts = {
        arm: {
            filename: {
                "path": rel(directory / filename),
                "sha256": file_sha256(directory / filename),
                "bytes": (directory / filename).stat().st_size,
            }
            for filename in required_files
        }
        for arm, directory in arm_dirs.items()
    }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": "ADOPTED" if adopted else "NOT_ADOPTED",
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "optimizer_body_positions": exposures,
        "parameter_counts": parameters,
        "plan_label_receipts": plan_receipts,
        "plan_positive_weight_receipts": positive_weight_receipts,
        "metrics": arm_metrics,
        "deltas": deltas,
        "semantic_label_causal_gain": plan_signal_causal,
        "artifacts": artifacts,
        "non_claims": [
            "plan-label prediction without verifier behavior gain is not capability promotion",
            "training-body-derived labels are never available during heldout generation",
            "the plan head does not render, compile, repair, route, or provide fallback code",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": receipt["adoption_state"],
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_ordered_plan_ablation(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    """Qualify ordered Semantic-IR planning against invalid matched controls."""

    required_arms = {"body_only", "semantic", "shuffled", "dropout"}
    required_files = (
        "report.json",
        "config.json",
        "candidates.jsonl",
        "integrity.json",
        "blind_audit.json",
    )
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED", "observed_arms": sorted(arm_dirs)},
            "hard_gaps": [gap("ordered_plan_arm_set_mismatch", {"observed": sorted(arm_dirs)})],
        }
    missing = [
        rel(directory / filename)
        for directory in arm_dirs.values()
        for filename in required_files
        if not (directory / filename).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }

    reports = {name: read_json(directory / "report.json") for name, directory in arm_dirs.items()}
    configs = {name: read_json(directory / "config.json") for name, directory in arm_dirs.items()}
    integrities = {name: read_json(directory / "integrity.json") for name, directory in arm_dirs.items()}
    blind = {name: read_json(directory / "blind_audit.json") for name, directory in arm_dirs.items()}
    hard_gaps: list[dict[str, Any]] = []

    def normalized_config(value: dict[str, Any]) -> dict[str, Any]:
        copied = json.loads(json.dumps(value))
        tokenization = copied.get("tokenization") if isinstance(copied.get("tokenization"), dict) else {}
        tokenization.pop("target_mode", None)
        tokenization.pop("semantic_plan_max_tokens", None)
        copied.pop("ordered_plan_training", None)
        return copied

    configs_equal = len(
        {json.dumps(normalized_config(value), sort_keys=True) for value in configs.values()}
    ) == 1
    expected_modes = {
        "body_only": ("body_tokens", "none"),
        "semantic": ("typed_semantic_ir_plan_body_tokens_v1", "semantic"),
        "shuffled": ("typed_semantic_ir_plan_body_tokens_v1", "shuffled"),
        "dropout": ("typed_semantic_ir_plan_body_tokens_v1", "dropout"),
    }
    modes_correct = all(
        (
            str((configs[name].get("tokenization") or {}).get("target_mode") or "body_tokens"),
            str((configs[name].get("ordered_plan_training") or {}).get("label_mode") or "none"),
        )
        == expected
        for name, expected in expected_modes.items()
    )

    def body_positions(report: dict[str, Any]) -> int:
        return sum(
            int(row.get("optimizer_body_positions_consumed") or 0)
            for row in (report.get("training") or {}).get("phases", [])
            if isinstance(row, dict)
        )

    exposures = {name: body_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(exposures.values())) == 1 and next(iter(exposures.values())) > 0
    lineage_fields = (
        "family_disjoint_eval_task_count",
        "unique_body_target_positions",
        "unique_sft_body_count",
        "train_holdout_family_overlap_count",
        "train_eval_prompt_overlap_count",
        "train_eval_body_overlap_count",
    )
    lineage_equal = all(
        len({json.dumps((report.get("stage") or {}).get(field), sort_keys=True) for report in reports.values()})
        == 1
        for field in lineage_fields
    )
    holdout_equal = len(
        {
            json.dumps((report.get("stage") or {}).get("holdout_families") or [], sort_keys=True)
            for report in reports.values()
        }
    ) == 1
    parameters = {
        name: int((report.get("architecture") or {}).get("parameter_count") or 0)
        for name, report in reports.items()
    }
    parameter_contract = (
        parameters["semantic"] == parameters["shuffled"] == parameters["dropout"]
        and parameters["semantic"] >= parameters["body_only"] > 0
    )
    receipts = {
        name: (report.get("stage") or {}).get("ordered_plan_label_receipt") or {}
        for name, report in reports.items()
    }
    semantic_receipt = receipts["semantic"]
    shuffled_receipt = receipts["shuffled"]
    dropout_receipt = receipts["dropout"]
    plan_mass_equal = (
        int(semantic_receipt.get("encoded_plan_position_count") or 0)
        == int(shuffled_receipt.get("encoded_plan_position_count") or -1)
        == int(dropout_receipt.get("encoded_plan_position_count") or -2)
        > 0
        and int(semantic_receipt.get("encoded_body_position_count") or 0)
        == int(shuffled_receipt.get("encoded_body_position_count") or -1)
        == int(dropout_receipt.get("encoded_body_position_count") or -2)
        and int(semantic_receipt.get("fixed_point_count") or 0)
        == int(semantic_receipt.get("row_count") or -1)
        and int(shuffled_receipt.get("fixed_point_count", -1)) == 0
        and len(
            {
                str(semantic_receipt.get("plan_sha256") or ""),
                str(shuffled_receipt.get("plan_sha256") or ""),
                str(dropout_receipt.get("plan_sha256") or ""),
            }
        )
        == 3
    )
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        and resolve(str(value.get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((value.get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name, value in integrities.items()
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    evaluation_replay_bound = all(
        evaluation_replay_is_content_bound(report, arm_dirs[name] / "report.json")
        for name, report in reports.items()
    )
    checks = {
        "configs_equal_except_ordered_plan_policy": configs_equal,
        "modes_expected": modes_correct,
        "optimizer_body_exposure_equal": exposure_equal,
        "training_and_holdout_lineage_equal": lineage_equal and holdout_equal,
        "plan_parameter_counts_equal": parameter_contract,
        "plan_and_body_token_mass_matched": plan_mass_equal,
        "integrity_clean_and_bound": integrity_clean,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
        "evaluation_replay_content_bound": evaluation_replay_bound,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"ordered_plan_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        plan_eval = (report.get("training") or {}).get("ordered_plan_eval_after") or {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "body_eval_loss": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "ordered_plan_eval_loss": (
                float(plan_eval.get("teacher_forced_loss"))
                if plan_eval.get("state") == "MEASURED"
                else None
            ),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    arm_metrics = {name: metrics(report) for name, report in reports.items()}
    semantic = arm_metrics["semantic"]
    controls = [arm_metrics[name] for name in ("body_only", "shuffled", "dropout")]
    semantic_plan_loss = semantic["ordered_plan_eval_loss"]
    plan_signal_causal = (
        semantic_plan_loss is not None
        and arm_metrics["shuffled"]["ordered_plan_eval_loss"] is not None
        and arm_metrics["dropout"]["ordered_plan_eval_loss"] is not None
        and semantic_plan_loss < arm_metrics["shuffled"]["ordered_plan_eval_loss"]
        and semantic_plan_loss < arm_metrics["dropout"]["ordered_plan_eval_loss"]
    )
    behavior_gain = semantic["passed_task_count"] > max(row["passed_task_count"] for row in controls)
    coverage_non_regressed = semantic["candidate_task_count"] >= max(
        row["candidate_task_count"] for row in controls
    )
    reward_non_regressed = semantic["mean_verification_reward"] >= max(
        row["mean_verification_reward"] for row in controls
    )
    adopted = (
        not hard_gaps
        and plan_signal_causal
        and behavior_gain
        and coverage_non_regressed
        and reward_non_regressed
    )
    rejection_reasons: list[str] = []
    if not plan_signal_causal:
        rejection_reasons.append("semantic_plan_did_not_beat_invalid_plan_controls")
    if not behavior_gain:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if not coverage_non_regressed:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if not reward_non_regressed:
        rejection_reasons.append("mean_verification_reward_regressed")
    deltas = {
        comparator: {
            key: (
                round(float(semantic[key]) - float(arm_metrics[comparator][key]), 8)
                if semantic[key] is not None and arm_metrics[comparator][key] is not None
                else None
            )
            for key in semantic
        }
        for comparator in ("body_only", "shuffled", "dropout")
    }
    artifacts = {
        arm: {
            filename: {
                "path": rel(directory / filename),
                "sha256": file_sha256(directory / filename),
                "bytes": (directory / filename).stat().st_size,
            }
            for filename in required_files
        }
        for arm, directory in arm_dirs.items()
    }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": "ADOPTED" if adopted else "NOT_ADOPTED",
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "optimizer_body_positions": exposures,
        "parameter_counts": parameters,
        "ordered_plan_receipts": receipts,
        "metrics": arm_metrics,
        "deltas": deltas,
        "semantic_plan_causal_gain": plan_signal_causal,
        "artifacts": artifacts,
        "non_claims": [
            "ordered-plan loss or syntax without exact verifier gain is not capability promotion",
            "heldout plan labels are measurement-only and never enter generation",
            "no plan is deterministically rendered, repaired, routed, or counted as body generation",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": receipt["adoption_state"],
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_latent_ordered_plan_ablation(
    arm_dirs: dict[str, Path],
    *,
    conditioning_mode: str = "global_additive",
    loss_mode: str = "binary_multilabel",
    probability_mode: str = "independent_sigmoid",
    target_contract: str = "ordered_plan_slot_token_field",
    gap_prefix: str = "latent_ordered_plan",
    policy: str = "project_theseus_latent_ordered_plan_ablation_v1",
) -> dict[str, Any]:
    """Qualify the low-rank ordered plan field against invalid matched controls."""

    required_arms = {"body_only", "semantic", "shuffled", "dropout"}
    required_files = (
        "report.json",
        "config.json",
        "candidates.jsonl",
        "integrity.json",
        "blind_audit.json",
    )
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED", "observed_arms": sorted(arm_dirs)},
            "hard_gaps": [
                gap(f"{gap_prefix}_arm_set_mismatch", {"observed": sorted(arm_dirs)})
            ],
        }
    missing = [
        rel(directory / filename)
        for directory in arm_dirs.values()
        for filename in required_files
        if not (directory / filename).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }

    reports = {name: read_json(directory / "report.json") for name, directory in arm_dirs.items()}
    configs = {name: read_json(directory / "config.json") for name, directory in arm_dirs.items()}
    integrities = {name: read_json(directory / "integrity.json") for name, directory in arm_dirs.items()}
    blind = {name: read_json(directory / "blind_audit.json") for name, directory in arm_dirs.items()}
    hard_gaps: list[dict[str, Any]] = []

    def normalized_config(value: dict[str, Any]) -> dict[str, Any]:
        copied = json.loads(json.dumps(value))
        model = copied.get("model") if isinstance(copied.get("model"), dict) else {}
        for key in (
            "semantic_plan_feature_count",
            "semantic_plan_separator_token_id",
            "semantic_plan_bottleneck_dim",
            "semantic_plan_slot_count",
            "semantic_plan_conditioning_mode",
            "semantic_plan_probability_mode",
            "semantic_plan_factor_group_sizes",
        ):
            model.pop(key, None)
        copied.pop("semantic_plan_training", None)
        return copied

    configs_equal = len(
        {json.dumps(normalized_config(value), sort_keys=True) for value in configs.values()}
    ) == 1
    expected_modes = {
        "body_only": "none",
        "semantic": "semantic",
        "shuffled": "shuffled",
        "dropout": "dropout",
    }
    modes_correct = all(
        str(
            ((configs[name].get("semantic_plan_training") or {}).get("label_mode"))
            or "none"
        )
        == expected
        for name, expected in expected_modes.items()
    )
    target_contract_clean = all(
        str((configs[name].get("tokenization") or {}).get("target_mode") or "")
        == "body_tokens"
        for name in required_arms
    ) and all(
        (configs[name].get("semantic_plan_training") or {}).get("target")
        == target_contract
        and int((configs[name].get("semantic_plan_training") or {}).get("ordered_slot_count") or 0)
        > 0
        for name in ("semantic", "shuffled", "dropout")
    )
    conditioning_contract_clean = all(
        str(
            (configs[name].get("model") or {}).get("semantic_plan_conditioning_mode")
            or "global_additive"
        )
        == conditioning_mode
        for name in ("semantic", "shuffled", "dropout")
    )
    objective_contract_clean = all(
        str(
            (configs[name].get("semantic_plan_training") or {}).get("loss_mode")
            or "binary_multilabel"
        )
        == loss_mode
        and str(
            (configs[name].get("model") or {}).get("semantic_plan_probability_mode")
            or "independent_sigmoid"
        )
        == probability_mode
        for name in ("semantic", "shuffled", "dropout")
    )
    factor_group_contract_clean = True
    factor_groups: dict[str, tuple[int, ...]] = {}
    if probability_mode == "factorized_step":
        factor_groups = {
            name: tuple(
                int(value)
                for value in (
                    (configs[name].get("model") or {}).get(
                        "semantic_plan_factor_group_sizes"
                    )
                    or []
                )
            )
            for name in ("semantic", "shuffled", "dropout")
        }
        factor_group_contract_clean = (
            len(set(factor_groups.values())) == 1
            and all(groups and groups[0] == 1 for groups in factor_groups.values())
            and all(
                sum(factor_groups[name])
                * int(
                    (configs[name].get("model") or {}).get(
                        "semantic_plan_slot_count"
                    )
                    or 0
                )
                == int(
                    (configs[name].get("model") or {}).get(
                        "semantic_plan_feature_count"
                    )
                    or 0
                )
                for name in factor_groups
            )
        )
    representation_audit: dict[str, Any] = {
        "state": "NOT_APPLICABLE",
        "reason": "non_factorized_plan_contract",
    }
    representation_collision_free = True
    if probability_mode == "factorized_step":
        arrays_path = arm_dirs["semantic"] / "stage" / "stage_arrays_v1.npz"
        if arrays_path.exists():
            with np.load(arrays_path) as arrays:
                labels = np.asarray(arrays["eval_plan_labels"], dtype=np.float32)
            slots = int(
                (configs["semantic"].get("model") or {}).get(
                    "semantic_plan_slot_count"
                )
                or 0
            )
            groups = factor_groups.get("semantic", ())
            width = sum(groups)
            shape_valid = (
                labels.ndim == 2
                and slots > 0
                and width > 0
                and labels.shape[1] == slots * width
            )
            group_closed = shape_valid
            if shape_valid:
                slot_labels = labels.reshape(len(labels), slots, width)
                presence = slot_labels[:, :, 0]
                group_closed = bool(
                    np.all((presence == 0.0) | (presence == 1.0))
                )
                offset = 1
                for group_width in groups[1:]:
                    group_closed = group_closed and bool(
                        np.all(
                            slot_labels[:, :, offset : offset + group_width].sum(
                                axis=-1
                            )
                            == presence
                        )
                    )
                    offset += group_width
            unique_count = (
                int(np.unique(labels, axis=0).shape[0]) if shape_valid else 0
            )
            row_count = int(labels.shape[0]) if labels.ndim == 2 else 0
            expected_rows = int(
                (reports["semantic"].get("stage") or {}).get(
                    "unique_semantic_eval_task_count"
                )
                or 0
            )
            representation_collision_free = (
                shape_valid
                and group_closed
                and row_count == expected_rows
                and unique_count == row_count
            )
            representation_audit = {
                "state": "GREEN" if representation_collision_free else "RED",
                "path": rel(arrays_path),
                "sha256": file_sha256(arrays_path),
                "row_count": row_count,
                "expected_semantic_task_count": expected_rows,
                "unique_plan_count": unique_count,
                "collision_row_count": row_count - unique_count,
                "feature_count": int(labels.shape[1]) if labels.ndim == 2 else 0,
                "slot_count": slots,
                "factor_group_sizes": list(groups),
                "group_closed": group_closed,
                "measurement_only": True,
            }
        else:
            representation_collision_free = False
            representation_audit = {
                "state": "RED",
                "reason": "factorized_eval_plan_labels_missing",
                "path": rel(arrays_path),
            }
    if conditioning_mode == "slot_attention":
        conditioning_contract_clean = conditioning_contract_clean and all(
            int((configs[name].get("model") or {}).get("semantic_plan_slot_count") or 0)
            == int(
                (configs[name].get("semantic_plan_training") or {}).get(
                    "ordered_slot_count"
                )
                or 0
            )
            > 0
            for name in ("semantic", "shuffled", "dropout")
        )
    else:
        conditioning_contract_clean = conditioning_contract_clean and all(
            int((configs[name].get("model") or {}).get("semantic_plan_slot_count") or 0)
            == 0
            for name in ("semantic", "shuffled", "dropout")
        )

    def body_positions(report: dict[str, Any]) -> int:
        return sum(
            int(row.get("optimizer_body_positions_consumed") or 0)
            for row in (report.get("training") or {}).get("phases", [])
            if isinstance(row, dict)
        )

    exposures = {name: body_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(exposures.values())) == 1 and next(iter(exposures.values())) > 0
    lineage_fields = (
        "sft_example_count",
        "unique_body_target_positions",
        "unique_sft_body_count",
        "train_holdout_family_overlap_count",
        "train_eval_prompt_overlap_count",
        "train_eval_body_overlap_count",
        "holdout_families",
    )
    lineage_equal = all(
        len(
            {
                json.dumps((report.get("stage") or {}).get(field), sort_keys=True)
                for report in reports.values()
            }
        )
        == 1
        for field in lineage_fields
    )
    parameters = {
        name: int((report.get("architecture") or {}).get("parameter_count") or 0)
        for name, report in reports.items()
    }
    parameter_contract = (
        parameters["semantic"] == parameters["shuffled"] == parameters["dropout"]
        and parameters["semantic"] > parameters["body_only"] > 0
    )
    feature_contracts = {
        name: (report.get("architecture") or {}).get("semantic_plan_head") or {}
        for name, report in reports.items()
    }
    feature_contract_clean = (
        len(
            {
                str(feature_contracts[name].get("feature_contract_sha256") or "")
                for name in ("semantic", "shuffled", "dropout")
            }
        )
        == 1
        and "" not in {
            str(feature_contracts[name].get("feature_contract_sha256") or "")
            for name in ("semantic", "shuffled", "dropout")
        }
        and len(
            {
                int(feature_contracts[name].get("feature_count") or 0)
                for name in ("semantic", "shuffled", "dropout")
            }
        )
        == 1
    )
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        and resolve(str(value.get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((value.get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name, value in integrities.items()
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    replay_bound = all(
        evaluation_replay_is_content_bound(report, arm_dirs[name] / "report.json")
        for name, report in reports.items()
    )
    checks = {
        "configs_equal_except_latent_plan_policy": configs_equal,
        "modes_expected": modes_correct,
        "body_target_stream_unchanged": target_contract_clean,
        "conditioning_contract_matches": conditioning_contract_clean,
        "objective_contract_matches": objective_contract_clean,
        "factor_group_contract_matches": factor_group_contract_clean,
        "factorized_representation_collision_free": representation_collision_free,
        "optimizer_body_exposure_equal": exposure_equal,
        "training_and_holdout_lineage_equal": lineage_equal,
        "plan_parameter_counts_equal": parameter_contract,
        "ordered_feature_contract_equal": feature_contract_clean,
        "integrity_clean_and_bound": integrity_clean,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
        "evaluation_replay_content_bound": replay_bound,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"{gap_prefix}_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        plan_eval = (report.get("training") or {}).get("semantic_plan_eval_after") or {}
        passed_task_count = int(summary.get("model_only_passed_task_count") or 0)
        decode_runtime_ms = int((report.get("decode") or {}).get("runtime_ms") or 0)
        return {
            "passed_task_count": passed_task_count,
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "body_eval_loss": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "plan_micro_f1": float(plan_eval.get("micro_f1") or 0.0),
            "plan_objective_loss": float(
                next(
                    (
                        plan_eval.get(key)
                        for key in (
                            "factorized_cross_entropy",
                            "categorical_cross_entropy",
                            "binary_cross_entropy",
                        )
                        if plan_eval.get(key) is not None
                    ),
                    0.0,
                )
            ),
            "decode_runtime_ms": decode_runtime_ms,
            "accepted_verified_output_per_second": round(
                passed_task_count / max(decode_runtime_ms / 1000.0, 1e-9), 8
            ),
        }

    arm_metrics = {name: metrics(report) for name, report in reports.items()}
    semantic = arm_metrics["semantic"]
    controls = [arm_metrics[name] for name in ("body_only", "shuffled", "dropout")]
    plan_signal_learned = semantic["plan_micro_f1"] > max(
        arm_metrics["shuffled"]["plan_micro_f1"],
        arm_metrics["dropout"]["plan_micro_f1"],
    )
    behavior_gain = semantic["passed_task_count"] > max(
        row["passed_task_count"] for row in controls
    )
    coverage_non_regressed = semantic["candidate_task_count"] >= max(
        row["candidate_task_count"] for row in controls
    )
    reward_non_regressed = semantic["mean_verification_reward"] >= max(
        row["mean_verification_reward"] for row in controls
    )
    rejection_reasons: list[str] = []
    if not plan_signal_learned:
        rejection_reasons.append("ordered_plan_signal_not_learned")
    if not behavior_gain:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if not coverage_non_regressed:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if not reward_non_regressed:
        rejection_reasons.append("mean_verification_reward_regressed")
    adoption_state = "ADOPTED" if not hard_gaps and not rejection_reasons else "NOT_ADOPTED"
    deltas = {
        name: {
            key: round(float(semantic[key]) - float(values[key]), 8)
            for key in semantic
        }
        for name, values in arm_metrics.items()
        if name != "semantic"
    }
    artifacts = {
        name: {
            filename: {
                "path": rel(directory / filename),
                "sha256": file_sha256(directory / filename),
            }
            for filename in required_files
        }
        for name, directory in arm_dirs.items()
    }
    receipt = {
        "policy": policy,
        "state": "RED" if hard_gaps else "GREEN",
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "optimizer_body_positions": exposures,
        "parameter_counts": parameters,
        "metrics": arm_metrics,
        "deltas": deltas,
        "ordered_plan_signal_learned": plan_signal_learned,
        "conditioning_mode": conditioning_mode,
        "loss_mode": loss_mode,
        "probability_mode": probability_mode,
        "representation_audit": representation_audit,
        "artifacts": artifacts,
        "non_claims": [
            "ordered plan classification quality is not body-generation capability",
            "generic extra parameters or latent context are not semantic-plan credit",
            "no route is enabled without exact verifier gain and non-regressed coverage/reward",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_slot_ordered_plan_ablation(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    """Qualify factorized per-layer plan memory without generic capacity credit."""

    return audit_latent_ordered_plan_ablation(
        arm_dirs,
        conditioning_mode="slot_attention",
        loss_mode="factorized_step_categorical",
        probability_mode="factorized_step",
        target_contract="ordered_plan_step_factor_field",
        gap_prefix="slot_ordered_plan",
        policy="project_theseus_slot_ordered_plan_ablation_v1",
    )


def audit_teacher_residual_ablation(
    arm_dirs: dict[str, Path],
    *,
    teacher_gate_path: Path = ROOT / "reports" / "teacher_distillation_gate.json",
    provider_audit_path: Path = ROOT / "reports" / "external_inference_audit.json",
) -> dict[str, Any]:
    required_arms = {"body_only", "semantic"}
    required_files = ("report.json", "config.json", "candidates.jsonl", "integrity.json", "blind_audit.json")
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED"},
            "hard_gaps": [gap("teacher_residual_arm_set_mismatch", {"observed": sorted(arm_dirs)})],
        }
    missing = [
        rel(directory / filename)
        for directory in arm_dirs.values()
        for filename in required_files
        if not (directory / filename).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }

    reports = {name: read_json(path / "report.json") for name, path in arm_dirs.items()}
    configs = {name: read_json(path / "config.json") for name, path in arm_dirs.items()}
    integrities = {name: read_json(path / "integrity.json") for name, path in arm_dirs.items()}
    blind = {name: read_json(path / "blind_audit.json") for name, path in arm_dirs.items()}
    prior_paths = {
        name: resolve(str((report.get("artifacts") or {}).get("prior_training_receipt") or ""))
        for name, report in reports.items()
    }
    priors = {name: read_json(path) for name, path in prior_paths.items()}
    teacher_gate = read_json(teacher_gate_path)
    provider_audit = read_json(provider_audit_path)
    hard_gaps: list[dict[str, Any]] = []
    state_keys = {
        "state_memory_slots", "state_memory_chunk_size", "state_memory_local_window",
        "state_memory_mode", "state_memory_ablation", "state_memory_read_policy",
    }

    def normalized_config(value: dict[str, Any]) -> dict[str, Any]:
        copy = json.loads(json.dumps(value))
        for key in state_keys:
            (copy.get("model") or {}).pop(key, None)
        return copy

    configs_equal = len({json.dumps(normalized_config(value), sort_keys=True) for value in configs.values()}) == 1
    modes_correct = (
        str((configs["body_only"].get("model") or {}).get("state_memory_mode") or "none") == "none"
        and str((configs["semantic"].get("model") or {}).get("state_memory_mode") or "") == "semantic_roles"
        and str((configs["semantic"].get("model") or {}).get("state_memory_read_policy") or "") == "unrestricted"
    )
    stage_signatures = {str((report.get("stage") or {}).get("stage_signature") or "") for report in reports.values()}
    teacher_cfg = configs["body_only"].get("teacher_distillation") or {}
    minimum_rows = int(teacher_cfg.get("minimum_code_lm_rows_for_sampling") or 0)
    target_probability = float(teacher_cfg.get("teacher_sampling_probability_target") or 0.0)
    stages = {name: report.get("stage") or {} for name, report in reports.items()}
    tranche_bound = all(
        int(stage.get("governed_teacher_prompt_pair_count") or 0) >= minimum_rows > 0
        and int(stage.get("governed_teacher_unique_body_count") or 0)
        == int(stage.get("governed_teacher_prompt_pair_count") or 0)
        and abs(float(stage.get("teacher_sampling_probability") or 0.0) - target_probability) <= 1e-9
        and bool((stage.get("governed_teacher_source_summary") or {}).get("gate_green"))
        and bool((stage.get("governed_teacher_source_summary") or {}).get("tranche_ready"))
        for stage in stages.values()
    )
    overlaps_clean = all(
        int(stage.get(key) or 0) == 0
        for stage in stages.values()
        for key in (
            "train_holdout_family_overlap_count",
            "train_eval_prompt_overlap_count",
            "train_eval_body_overlap_count",
            "governed_teacher_current_holdout_rejected_count",
            "governed_teacher_eval_overlap_rejected_count",
        )
    )

    def phase_positions(report: dict[str, Any]) -> tuple[int, int]:
        rows = [row for row in (report.get("training") or {}).get("phases", []) if isinstance(row, dict)]
        total = sum(int(row.get("optimizer_body_positions_consumed") or 0) for row in rows)
        latest = int(rows[-1].get("optimizer_body_positions_consumed") or 0) if rows else 0
        return total, latest

    positions = {name: phase_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(positions.values())) == 1 and next(iter(positions.values()))[1] > 0
    prior_bound = all(path.exists() and bool(priors[name]) for name, path in prior_paths.items())
    resume_hash_bound = all(
        (
            lambda checkpoint, expected: checkpoint.exists() and file_sha256(checkpoint) == expected
        )(
            resolve(str((priors[name].get("artifacts") or {}).get("checkpoint") or "")),
            str((reports[name].get("conditioning") or {}).get("resume_base_checkpoint_sha256") or ""),
        )
        for name in required_arms
    )
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        and resolve(str(value.get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((value.get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name, value in integrities.items()
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    provider_counts = (provider_audit.get("summary") or {}).get("teacher_provider_counts") or {}
    provider_clean = (
        provider_audit.get("ok") is True
        and int((provider_audit.get("summary") or {}).get("teacher_receipt_violations") or 0) == 0
        and bool(provider_counts)
        and all(str(identity).startswith("codex_cli/gpt-") for identity in provider_counts)
    )
    checks = {
        "configs_equal_except_state_policy": configs_equal,
        "modes_expected": modes_correct,
        "stage_signature_equal": len(stage_signatures) == 1 and "" not in stage_signatures,
        "teacher_tranche_bound": tranche_bound,
        "teacher_gate_green": teacher_gate.get("trigger_state") == "GREEN" and bool(teacher_gate.get("distillation_allowed")),
        "openai_provider_receipts_clean": provider_clean,
        "holdout_and_eval_overlap_clean": overlaps_clean,
        "body_exposure_equal": exposure_equal,
        "prior_receipts_bound": prior_bound,
        "resume_checkpoint_hashes_present": resume_hash_bound,
        "integrity_clean": integrity_clean,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"teacher_residual_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "eval_loss_after": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    current = {name: metrics(report) for name, report in reports.items()}
    previous = {name: metrics(report) for name, report in priors.items()}
    deltas = {
        name: {key: round(float(current[name][key]) - float(previous[name][key]), 6) for key in current[name]}
        for name in required_arms
    }
    body_gain = current["body_only"]["passed_task_count"] > previous["body_only"]["passed_task_count"]
    state_gain = (
        current["semantic"]["passed_task_count"] > previous["semantic"]["passed_task_count"]
        and current["semantic"]["passed_task_count"] > current["body_only"]["passed_task_count"]
    )
    if not hard_gaps and state_gain:
        adoption_state = "ADOPTED_STATE_SHADOW"
    elif not hard_gaps and body_gain:
        adoption_state = "ADOPTED_BODY_ONLY_SHADOW"
    else:
        adoption_state = "NOT_ADOPTED"
    rejection_reasons: list[str] = []
    if not body_gain and not state_gain:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if current["body_only"]["eval_loss_after"] >= previous["body_only"]["eval_loss_after"]:
        rejection_reasons.append("body_only_heldout_loss_worsened")
    if current["semantic"]["eval_loss_after"] >= previous["semantic"]["eval_loss_after"]:
        rejection_reasons.append("semantic_heldout_loss_worsened")
    if current["body_only"]["candidate_task_count"] < previous["body_only"]["candidate_task_count"]:
        rejection_reasons.append("body_only_candidate_coverage_regressed")
    artifacts = {
        arm: {
            filename: {
                "path": rel(directory / filename),
                "sha256": file_sha256(directory / filename),
                "bytes": (directory / filename).stat().st_size,
            }
            for filename in required_files
        }
        for arm, directory in arm_dirs.items()
    }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "teacher_row_count": int(stages["body_only"].get("governed_teacher_prompt_pair_count") or 0),
        "teacher_sampling_probability": float(stages["body_only"].get("teacher_sampling_probability") or 0.0),
        "optimizer_body_positions": positions,
        "prior_metrics": previous,
        "continuation_metrics": current,
        "deltas": deltas,
        "artifacts": artifacts,
        "decision": "quarantine_teacher_tranche" if adoption_state == "NOT_ADOPTED" else "retain_shadow_only",
        "non_claims": [
            "teacher-row verifier acceptance is not student capability",
            "syntax and candidate coverage without an exact pass receive no promotion credit",
            "this private result does not authorize preference/RL or public calibration",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_state_memory_continuation(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    required_arms = {"body_only", "semantic", "hash_control"}
    required_files = ("report.json", "config.json", "candidates.jsonl", "integrity.json", "blind_audit.json")
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED"},
            "hard_gaps": [gap("state_continuation_arm_set_mismatch", {"observed": sorted(arm_dirs)})],
        }
    missing = [
        rel(directory / filename)
        for directory in arm_dirs.values()
        for filename in required_files
        if not (directory / filename).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }
    reports = {name: read_json(path / "report.json") for name, path in arm_dirs.items()}
    configs = {name: read_json(path / "config.json") for name, path in arm_dirs.items()}
    integrities = {name: read_json(path / "integrity.json") for name, path in arm_dirs.items()}
    blind = {name: read_json(path / "blind_audit.json") for name, path in arm_dirs.items()}
    prior_paths = {
        name: resolve(str((report.get("artifacts") or {}).get("prior_training_receipt") or ""))
        for name, report in reports.items()
    }
    priors = {name: read_json(path) for name, path in prior_paths.items()}
    hard_gaps: list[dict[str, Any]] = []
    state_keys = {
        "state_memory_slots", "state_memory_chunk_size", "state_memory_local_window",
        "state_memory_mode", "state_memory_ablation", "state_memory_read_policy",
    }

    def normalized_config(value: dict[str, Any]) -> dict[str, Any]:
        copy = json.loads(json.dumps(value))
        for key in state_keys:
            (copy.get("model") or {}).pop(key, None)
        return copy

    configs_equal = len({json.dumps(normalized_config(value), sort_keys=True) for value in configs.values()}) == 1
    expected_modes = {"body_only": "none", "semantic": "semantic_roles", "hash_control": "hash_control"}
    modes_correct = all(
        str((configs[name].get("model") or {}).get("state_memory_mode") or "none") == mode
        for name, mode in expected_modes.items()
    )
    stages_equal = len({str((report.get("stage") or {}).get("stage_signature") or "") for report in reports.values()}) == 1
    parameter_counts = {
        name: int((report.get("architecture") or {}).get("parameter_count") or 0)
        for name, report in reports.items()
    }
    state_parameters_equal = parameter_counts["semantic"] == parameter_counts["hash_control"] > 0

    def phase_positions(report: dict[str, Any]) -> tuple[int, int]:
        rows = [row for row in (report.get("training") or {}).get("phases", []) if isinstance(row, dict)]
        total = sum(int(row.get("optimizer_body_positions_consumed") or 0) for row in rows)
        continuation = sum(
            int(row.get("optimizer_body_positions_consumed") or 0)
            for row in rows
            if str(row.get("phase") or "").endswith("_continuation")
        )
        return total, continuation

    positions = {name: phase_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(positions.values())) == 1 and next(iter(positions.values()))[1] > 0
    prior_bound = all(path.exists() and bool(priors[name]) for name, path in prior_paths.items())
    resume_hash_bound = all(
        (
            lambda checkpoint, expected: checkpoint.exists()
            and file_sha256(checkpoint) == expected
        )(
            resolve(str((priors[name].get("artifacts") or {}).get("checkpoint") or "")),
            str((reports[name].get("conditioning") or {}).get("resume_base_checkpoint_sha256") or ""),
        )
        for name in required_arms
    )
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        and resolve(str(value.get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((value.get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name, value in integrities.items()
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    checks = {
        "configs_equal_except_state_policy": configs_equal,
        "modes_expected": modes_correct,
        "stage_signature_equal": stages_equal,
        "state_parameter_counts_equal": state_parameters_equal,
        "body_exposure_equal": exposure_equal,
        "prior_receipts_bound": prior_bound,
        "resume_checkpoint_hashes_present": resume_hash_bound,
        "integrity_clean": integrity_clean,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"state_continuation_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "eval_loss_after": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    current = {name: metrics(report) for name, report in reports.items()}
    previous = {name: metrics(report) for name, report in priors.items()}
    deltas = {
        name: {key: round(float(current[name][key]) - float(previous[name][key]), 6) for key in current[name]}
        for name in required_arms
    }
    semantic = current["semantic"]
    exact_gain = semantic["passed_task_count"] > previous["semantic"]["passed_task_count"]
    control_gain = semantic["passed_task_count"] > max(
        current["body_only"]["passed_task_count"], current["hash_control"]["passed_task_count"]
    )
    loss_improved = semantic["eval_loss_after"] < previous["semantic"]["eval_loss_after"]
    adoption_state = "ADOPTED" if not hard_gaps and exact_gain and control_gain and loss_improved else "NOT_ADOPTED"
    rejection_reasons = []
    if not exact_gain:
        rejection_reasons.append("no_continuation_verifier_pass_gain")
    if not control_gain:
        rejection_reasons.append("no_exact_gain_over_matched_controls")
    if not loss_improved:
        rejection_reasons.append("semantic_heldout_loss_worsened")
    artifacts = {
        arm: {
            filename: {
                "path": rel(directory / filename),
                "sha256": file_sha256(directory / filename),
                "bytes": (directory / filename).stat().st_size,
            }
            for filename in required_files
        }
        for arm, directory in arm_dirs.items()
    }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "optimizer_body_positions": positions,
        "parameter_counts": parameter_counts,
        "prior_metrics": previous,
        "continuation_metrics": current,
        "deltas": deltas,
        "artifacts": artifacts,
        "decision": "stop_state_scaling_branch" if adoption_state == "NOT_ADOPTED" else "qualify_shadow_only",
        "non_claims": [
            "partial reward movement without an exact pass is not capability promotion",
            "continuation evidence does not authorize preference/RL or public calibration",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_state_memory_ablation(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    required_arms = {"body_only", "semantic", "hash_control", "zero", "shuffle"}
    if set(arm_dirs) != required_arms:
        return {
            "state": "RED",
            "adoption_state": "NOT_ADOPTED",
            "adoption_rejection_reasons": ["arm_set_mismatch"],
            "deltas": {},
            "receipt": {"state": "RED", "observed_arms": sorted(arm_dirs)},
            "hard_gaps": [gap("state_memory_arm_set_mismatch", {"observed": sorted(arm_dirs)})],
        }
    required_files = ("report.json", "config.json", "candidates.jsonl", "integrity.json", "blind_audit.json")
    missing = [
        rel(directory / name)
        for directory in arm_dirs.values()
        for name in required_files
        if not (directory / name).exists()
    ]
    if missing:
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "missing": missing},
            "hard_gaps": [],
        }

    reports = {name: read_json(directory / "report.json") for name, directory in arm_dirs.items()}
    configs = {name: read_json(directory / "config.json") for name, directory in arm_dirs.items()}
    integrities = {name: read_json(directory / "integrity.json") for name, directory in arm_dirs.items()}
    blind = {name: read_json(directory / "blind_audit.json") for name, directory in arm_dirs.items()}
    hard_gaps: list[dict[str, Any]] = []

    state_keys = {
        "state_memory_slots",
        "state_memory_chunk_size",
        "state_memory_local_window",
        "state_memory_mode",
        "state_memory_ablation",
        "state_memory_read_policy",
    }

    def matched_config(value: dict[str, Any]) -> dict[str, Any]:
        copy = json.loads(json.dumps(value))
        model = copy.get("model") if isinstance(copy.get("model"), dict) else {}
        for key in state_keys:
            model.pop(key, None)
        return copy

    configs_equal = len({json.dumps(matched_config(value), sort_keys=True) for value in configs.values()}) == 1
    expected_modes = {
        "body_only": ("none", "none"),
        "semantic": ("semantic_roles", "none"),
        "hash_control": ("hash_control", "none"),
        "zero": ("semantic_roles", "zero"),
        "shuffle": ("semantic_roles", "shuffle"),
    }
    modes_correct = all(
        (
            str((configs[name].get("model") or {}).get("state_memory_mode") or "none"),
            str((configs[name].get("model") or {}).get("state_memory_ablation") or "none"),
        )
        == expected
        for name, expected in expected_modes.items()
    )
    stage_signatures = {
        str((report.get("stage") or {}).get("stage_signature") or "")
        for report in reports.values()
    }

    def body_positions(report: dict[str, Any]) -> int:
        return sum(
            int(row.get("optimizer_body_positions_consumed") or 0)
            for row in (report.get("training") or {}).get("phases", [])
            if isinstance(row, dict)
        )

    exposures = {name: body_positions(report) for name, report in reports.items()}
    exposure_equal = len(set(exposures.values())) == 1 and next(iter(exposures.values())) > 0
    parameters = {
        name: int((report.get("architecture") or {}).get("parameter_count") or 0)
        for name, report in reports.items()
    }
    state_parameters_equal = len({parameters[name] for name in ("semantic", "hash_control", "zero", "shuffle")}) == 1
    integrity_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
        for value in integrities.values()
    )
    integrity_bound = all(
        resolve(str(integrities[name].get("source") or "")) == arm_dirs[name] / "candidates.jsonl"
        and int((integrities[name].get("summary") or {}).get("candidate_count") or 0)
        == len(read_jsonl(arm_dirs[name] / "candidates.jsonl"))
        for name in required_arms
    )
    blind_clean = all(
        value.get("trigger_state") == "GREEN"
        and int((value.get("summary") or {}).get("invalid_claim_count") or 0) == 0
        for value in blind.values()
    )
    boundaries_clean = all(
        int(report.get(key) or 0) == 0
        for report in reports.values()
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    semantic_checkpoint = resolve(str((reports["semantic"].get("artifacts") or {}).get("checkpoint") or ""))
    replay_checkpoints = [
        resolve(str((reports[name].get("artifacts") or {}).get("checkpoint") or ""))
        for name in ("zero", "shuffle")
    ]
    replay_checkpoint_bound = semantic_checkpoint.exists() and all(path == semantic_checkpoint for path in replay_checkpoints)
    checks = {
        "configs_equal_except_state_policy": configs_equal,
        "modes_and_ablations_expected": modes_correct,
        "stage_signature_equal_and_present": len(stage_signatures) == 1 and "" not in stage_signatures,
        "optimizer_body_exposure_equal": exposure_equal,
        "state_parameter_counts_equal": state_parameters_equal,
        "integrity_clean": integrity_clean,
        "integrity_candidate_binding_clean": integrity_bound,
        "blind_information_flow_clean": blind_clean,
        "boundaries_clean": boundaries_clean,
        "replays_use_semantic_checkpoint": replay_checkpoint_bound,
    }
    for name, passed in checks.items():
        if not passed:
            hard_gaps.append(gap(f"state_memory_{name}_failed", {}))

    def metrics(report: dict[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        private = (report.get("private_verifier") or {}).get("private_verification") or {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "eval_loss_after": float((report.get("training") or {}).get("eval_loss_after") or 0.0),
            "decode_runtime_ms": int((report.get("decode") or {}).get("runtime_ms") or 0),
        }

    arm_metrics = {name: metrics(report) for name, report in reports.items()}
    semantic = arm_metrics["semantic"]
    zero = arm_metrics["zero"]
    shuffle = arm_metrics["shuffle"]
    zero_degrades = (
        semantic["mean_verification_reward"] > zero["mean_verification_reward"]
        and semantic["eval_loss_after"] < zero["eval_loss_after"]
    )
    shuffle_degrades = (
        semantic["mean_verification_reward"] > shuffle["mean_verification_reward"]
        and semantic["eval_loss_after"] < shuffle["eval_loss_after"]
    )
    comparators = [arm_metrics["body_only"], arm_metrics["hash_control"]]
    behavior_gain = semantic["passed_task_count"] > max(row["passed_task_count"] for row in comparators)
    coverage_non_regressed = all(
        semantic["candidate_task_count"] >= row["candidate_task_count"] for row in comparators
    )
    reward_non_regressed = all(
        semantic["mean_verification_reward"] >= row["mean_verification_reward"] for row in comparators
    )
    adoption_state = (
        "ADOPTED"
        if not hard_gaps
        and behavior_gain
        and coverage_non_regressed
        and reward_non_regressed
        and zero_degrades
        and shuffle_degrades
        else "NOT_ADOPTED"
    )
    rejection_reasons = []
    if not behavior_gain:
        rejection_reasons.append("no_family_disjoint_verifier_pass_gain")
    if not zero_degrades:
        rejection_reasons.append("zero_memory_did_not_causally_degrade")
    if not shuffle_degrades:
        rejection_reasons.append("role_shuffle_did_not_causally_degrade")
    if not coverage_non_regressed:
        rejection_reasons.append("candidate_task_coverage_regressed")
    if not reward_non_regressed:
        rejection_reasons.append("mean_verification_reward_regressed")

    deltas = {}
    for comparator_name in ("body_only", "hash_control", "zero", "shuffle"):
        deltas[comparator_name] = {
            key: round(float(semantic[key]) - float(arm_metrics[comparator_name][key]), 6)
            for key in semantic
        }
    artifacts: dict[str, Any] = {}
    for arm, directory in arm_dirs.items():
        artifacts[arm] = {}
        for filename in required_files:
            path = directory / filename
            artifacts[arm][filename] = {
                "path": rel(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "matched_checks": checks,
        "optimizer_body_positions": exposures,
        "parameter_counts": parameters,
        "metrics": arm_metrics,
        "deltas": deltas,
        "zero_memory_causal_degradation": zero_degrades,
        "role_shuffle_causal_degradation": shuffle_degrades,
        "artifacts": artifacts,
        "non_claims": [
            "lower loss and higher partial verifier reward are not exact behavior",
            "recurrent-memory use does not prove semantic role specificity",
            "the state route remains disabled without a verifier-pass gain and both causal ablations",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": adoption_state,
        "adoption_rejection_reasons": rejection_reasons,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_sft_contract_admission(
    *,
    report_path: Path,
    integrity_path: Path,
    blind_audit_path: Path,
    control_report_path: Path,
) -> dict[str, Any]:
    paths = (report_path, integrity_path, blind_audit_path, control_report_path)
    if not all(path.exists() for path in paths):
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "deltas": {},
            "receipt": {
                "state": "NOT_RUN",
                "paths": [rel(path) for path in paths],
                "reason": "one or more local matched-ablation artifacts are absent",
            },
            "hard_gaps": [],
        }

    filtered = read_json(report_path)
    control = read_json(control_report_path)
    integrity = read_json(integrity_path)
    blind = read_json(blind_audit_path)
    hard_gaps: list[dict[str, Any]] = []
    filtered_config_path = resolve(str((filtered.get("artifacts") or {}).get("config") or ""))
    control_config_path = resolve(str((control.get("artifacts") or {}).get("config") or ""))
    filtered_config = read_json(filtered_config_path)
    control_config = read_json(control_config_path)

    def without_contract(value: dict[str, Any]) -> dict[str, Any]:
        copy = json.loads(json.dumps(value))
        copy.pop("sft_contract_admission", None)
        return copy

    matched_config = bool(filtered_config and control_config) and without_contract(
        filtered_config
    ) == without_contract(control_config)
    contract = filtered_config.get("sft_contract_admission")
    contract_enabled = isinstance(contract, dict) and contract.get("require_self_contained_body") is True
    filtered_stage = filtered.get("stage") if isinstance(filtered.get("stage"), dict) else {}
    contract_receipt = (
        filtered_stage.get("sft_contract_admission")
        if isinstance(filtered_stage.get("sft_contract_admission"), dict)
        else {}
    )
    boundaries_clean = all(
        int(payload.get(key) or 0) == 0
        for payload in (filtered, control)
        for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count")
    )
    integrity_clean = (
        integrity.get("trigger_state") == "GREEN"
        and int((integrity.get("summary") or {}).get("integrity_mismatch_count") or 0) == 0
    )
    blind_clean = blind.get("trigger_state") == "GREEN" and int(
        (blind.get("summary") or {}).get("invalid_claim_count") or 0
    ) == 0
    for name, passed in (
        ("sft_contract_configs_unmatched", matched_config),
        ("sft_contract_filter_not_enabled", contract_enabled),
        ("sft_contract_boundaries_unclean", boundaries_clean),
        ("sft_contract_integrity_unclean", integrity_clean),
        ("sft_contract_blind_audit_unclean", blind_clean),
        (
            "sft_contract_target_derived_source_field_nonzero",
            int(contract_receipt.get("target_body_fields_added_to_model_source") or 0) == 0,
        ),
        (
            "sft_contract_heldout_rows_read_nonzero",
            int(contract_receipt.get("heldout_rows_read_by_filter") or 0) == 0,
        ),
    ):
        if not passed:
            hard_gaps.append(gap(name, {}))

    def metrics(value: dict[str, Any]) -> dict[str, Any]:
        summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
        verifier = value.get("private_verifier") if isinstance(value.get("private_verifier"), dict) else {}
        private = verifier.get("private_verification") if isinstance(verifier.get("private_verification"), dict) else {}
        training = value.get("training") if isinstance(value.get("training"), dict) else {}
        decode = value.get("decode") if isinstance(value.get("decode"), dict) else {}
        return {
            "passed_task_count": int(summary.get("model_only_passed_task_count") or 0),
            "candidate_task_count": int(summary.get("candidate_task_count") or 0),
            "candidate_count": int(summary.get("candidate_count") or 0),
            "mean_verification_reward": float(private.get("mean_verification_reward") or 0.0),
            "eval_loss_after": float(training.get("eval_loss_after") or 0.0),
            "generation_runtime_ms": int(decode.get("runtime_ms") or 0),
        }

    filtered_metrics = metrics(filtered)
    control_metrics = metrics(control)
    deltas = {
        key: round(float(filtered_metrics[key]) - float(control_metrics[key]), 6)
        for key in filtered_metrics
    }
    behavior_improved = (
        filtered_metrics["passed_task_count"] > control_metrics["passed_task_count"]
        and filtered_metrics["candidate_task_count"] >= control_metrics["candidate_task_count"]
        and filtered_metrics["mean_verification_reward"] >= control_metrics["mean_verification_reward"]
    )
    adoption_state = (
        "ADOPTED"
        if not hard_gaps and behavior_improved
        else "NOT_ADOPTED"
    )
    artifacts = {}
    for name, path in (
        ("filtered_report", report_path),
        ("filtered_config", filtered_config_path),
        ("filtered_integrity", integrity_path),
        ("filtered_blind_audit", blind_audit_path),
        ("control_report", control_report_path),
        ("control_config", control_config_path),
    ):
        artifacts[name] = {
            "path": rel(path),
            "sha256": file_sha256(path) if path.exists() else "",
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "adoption_state": adoption_state,
        "matched_config_except_contract_admission": matched_config,
        "boundaries_clean": boundaries_clean,
        "integrity_clean": integrity_clean,
        "blind_audit_clean": blind_clean,
        "contract_receipt": contract_receipt,
        "filtered": filtered_metrics,
        "control": control_metrics,
        "deltas": deltas,
        "artifacts": artifacts,
        "non_claims": [
            "a cleaner SFT admission set is not a capability improvement",
            "the filter remains disabled unless matched behavior improves",
        ],
    }
    return {
        "state": receipt["state"],
        "adoption_state": adoption_state,
        "deltas": deltas,
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_target_mode_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "state": "NOT_RUN",
            "adoption_state": "NOT_RUN",
            "comparison_sha256": "",
            "adoption_rejection_reasons": [],
            "deltas": {},
            "receipt": {"state": "NOT_RUN", "path": rel(path)},
            "hard_gaps": [],
        }
    comparison = read_json(path)
    hard_gaps: list[dict[str, Any]] = []
    if comparison.get("policy") != "project_theseus_standard_causal_target_mode_matched_comparison_v1":
        hard_gaps.append(gap("target_mode_comparison_policy_mismatch", {"path": rel(path)}))
    if comparison.get("trigger_state") != "GREEN":
        hard_gaps.append(
            gap("target_mode_comparison_not_green", {"state": comparison.get("trigger_state")})
        )
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(comparison.get(key) or 0) != 0:
            hard_gaps.append(gap(f"target_mode_comparison_{key}_nonzero", {"observed": comparison.get(key)}))
    matched_checks = comparison.get("matched_checks") if isinstance(comparison.get("matched_checks"), dict) else {}
    plan = comparison.get("plan") if isinstance(comparison.get("plan"), dict) else {}
    control = comparison.get("control") if isinstance(comparison.get("control"), dict) else {}
    integrity_clean = int(plan.get("integrity_mismatch_count") or 0) == int(
        control.get("integrity_mismatch_count") or 0
    ) == 0
    expected_adoption = (
        "ADOPTED"
        if matched_checks
        and all(value is True for value in matched_checks.values())
        and comparison.get("boundaries_clean") is True
        and integrity_clean
        and int(plan.get("passed_task_count") or 0) > int(control.get("passed_task_count") or 0)
        and int(plan.get("candidate_task_count") or 0) >= int(control.get("candidate_task_count") or 0)
        and float(plan.get("mean_verification_reward") or 0.0)
        >= float(control.get("mean_verification_reward") or 0.0)
        else "NOT_ADOPTED"
    )
    if comparison.get("adoption_state") != expected_adoption:
        hard_gaps.append(
            gap(
                "target_mode_adoption_decision_mismatch",
                {"observed": comparison.get("adoption_state"), "expected": expected_adoption},
            )
        )
    artifacts = comparison.get("artifacts") if isinstance(comparison.get("artifacts"), dict) else {}
    required_artifacts = {
        "plan_report",
        "plan_config",
        "plan_candidates",
        "plan_checkpoint",
        "plan_integrity",
        "control_report",
        "control_config",
        "control_candidates",
        "control_checkpoint",
        "control_integrity",
    }
    if set(artifacts) != required_artifacts:
        hard_gaps.append(
            gap(
                "target_mode_artifact_set_mismatch",
                {"observed": sorted(artifacts), "required": sorted(required_artifacts)},
            )
        )
    artifact_receipts: dict[str, Any] = {}
    for name, item in artifacts.items():
        record = item if isinstance(item, dict) else {}
        artifact_path = resolve(str(record.get("path") or ""))
        observed_sha256 = file_sha256(artifact_path) if artifact_path.exists() else ""
        observed_bytes = artifact_path.stat().st_size if artifact_path.exists() else 0
        matches = (
            artifact_path.exists()
            and observed_sha256 == str(record.get("sha256") or "")
            and observed_bytes == int(record.get("bytes") or 0)
        )
        artifact_receipts[name] = {
            "path": rel(artifact_path),
            "sha256": observed_sha256,
            "bytes": observed_bytes,
            "matches": matches,
        }
        if not matches:
            hard_gaps.append(gap("target_mode_artifact_binding_mismatch", {"artifact": name}))
    receipt = {
        "state": "GREEN" if not hard_gaps else "RED",
        "path": rel(path),
        "sha256": file_sha256(path),
        "adoption_state": comparison.get("adoption_state"),
        "adoption_rejection_reasons": list(comparison.get("adoption_rejection_reasons") or []),
        "matched_checks": matched_checks,
        "plan": plan,
        "control": control,
        "deltas": comparison.get("deltas") or {},
        "artifact_receipts": artifact_receipts,
        "non_claims": list(comparison.get("non_claims") or []),
    }
    return {
        "state": receipt["state"],
        "adoption_state": str(comparison.get("adoption_state") or ""),
        "comparison_sha256": receipt["sha256"],
        "adoption_rejection_reasons": receipt["adoption_rejection_reasons"],
        "deltas": receipt["deltas"],
        "receipt": receipt,
        "hard_gaps": hard_gaps,
    }


def audit_preference_canary(value: Any, canonical_checkpoint: Path) -> dict[str, Any]:
    canary = value if isinstance(value, dict) else {}
    state = str(canary.get("state") or "NOT_RUN")
    adoption_state = str(canary.get("adoption_state") or "NOT_RUN")
    hard_gaps: list[dict[str, Any]] = []
    if state in {"NOT_RUN", "TYPED_NO_REWARD_PAIRS"}:
        return {
            "state": state,
            "adoption_state": adoption_state,
            "reward_behavior_delta": 0,
            "hard_gaps": [],
        }
    if state != "GREEN":
        hard_gaps.append(gap("preference_canary_state_invalid", {"state": state}))
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(canary.get(key) or 0) != 0:
            hard_gaps.append(gap(f"preference_canary_{key}_nonzero", {"observed": canary.get(key)}))
    base = canary.get("base_heldout") if isinstance(canary.get("base_heldout"), dict) else {}
    reward = canary.get("reward_present_heldout") if isinstance(canary.get("reward_present_heldout"), dict) else {}
    control = canary.get("reward_removed_heldout") if isinstance(canary.get("reward_removed_heldout"), dict) else {}
    reward_training = canary.get("reward_present_training") if isinstance(canary.get("reward_present_training"), dict) else {}
    control_training = canary.get("reward_removed_training") if isinstance(canary.get("reward_removed_training"), dict) else {}
    pair_summary = canary.get("preference_pair_summary") if isinstance(canary.get("preference_pair_summary"), dict) else {}
    reward_passes = int(reward.get("passed_task_count") or 0)
    base_passes = int(base.get("passed_task_count") or 0)
    control_passes = int(control.get("passed_task_count") or 0)
    reward_rank1 = int(reward.get("rank1_passed_task_count") or 0)
    base_rank1 = int(base.get("rank1_passed_task_count") or 0)
    control_rank1 = int(control.get("rank1_passed_task_count") or 0)
    independently_improves = (reward_passes > base_passes and reward_passes > control_passes) or (
        reward_passes >= base_passes
        and reward_passes >= control_passes
        and reward_rank1 > max(base_rank1, control_rank1)
    )
    if bool(canary.get("reward_improves_behavior")) != independently_improves:
        hard_gaps.append(
            gap(
                "preference_behavior_decision_mismatch",
                {
                    "claimed": canary.get("reward_improves_behavior"),
                    "recomputed": independently_improves,
                },
            )
        )
    expected_adoption = "QUALIFIED_SHADOW" if independently_improves else "NOT_ADOPTED"
    if adoption_state != expected_adoption:
        hard_gaps.append(
            gap(
                "preference_adoption_state_mismatch",
                {"observed": adoption_state, "expected": expected_adoption},
            )
        )
    if int(pair_summary.get("selected_pair_count") or 0) <= 0:
        hard_gaps.append(gap("preference_pair_evidence_missing", {"pair_summary": pair_summary}))
    if float(reward_training.get("preference_margin_delta") or 0.0) <= 0:
        hard_gaps.append(gap("reward_present_margin_not_improved", {"training": reward_training}))
    if abs(float(control_training.get("preference_margin_delta") or 0.0)) > 1e-8:
        hard_gaps.append(gap("reward_removed_control_margin_changed", {"training": control_training}))
    for label, heldout in (("reward", reward), ("control", control)):
        if int(heldout.get("integrity_mismatch_count") or 0) != 0:
            hard_gaps.append(gap(f"preference_{label}_integrity_mismatch", {"heldout": heldout}))
    artifacts = canary.get("artifacts") if isinstance(canary.get("artifacts"), dict) else {}
    for key in ("reward_checkpoint", "control_checkpoint"):
        path = resolve(str(artifacts.get(key) or ""))
        if path == canonical_checkpoint:
            hard_gaps.append(gap("preference_shadow_overwrote_canonical_checkpoint", {"artifact": key}))
    return {
        "state": state,
        "adoption_state": adoption_state,
        "reward_behavior_delta": reward_passes - base_passes,
        "hard_gaps": hard_gaps,
    }


def audit_generation_mode_canary(value: Any) -> dict[str, Any]:
    canary = value if isinstance(value, dict) else {}
    state = str(canary.get("state") or "NOT_RUN")
    adoption_state = str(canary.get("adoption_state") or "NOT_RUN")
    speedup = float(canary.get("generation_speedup") or 0.0)
    if state == "NOT_RUN":
        return {"state": state, "adoption_state": adoption_state, "speedup": speedup, "hard_gaps": []}
    hard_gaps: list[dict[str, Any]] = []
    if state != "GREEN":
        hard_gaps.append(gap("generation_mode_canary_state_invalid", {"state": state}))
    for key in ("public_training_rows_written", "external_inference_calls", "fallback_return_count"):
        if int(canary.get(key) or 0) != 0:
            hard_gaps.append(gap(f"generation_mode_{key}_nonzero", {"observed": canary.get(key)}))
    serial = canary.get("serial") if isinstance(canary.get("serial"), dict) else {}
    batched = canary.get("batched") if isinstance(canary.get("batched"), dict) else {}
    recomputed_behavior = (
        int(batched.get("passed_task_count") or 0) >= int(serial.get("passed_task_count") or 0)
        and int(batched.get("rank1_passed_task_count") or 0)
        >= int(serial.get("rank1_passed_task_count") or 0)
    )
    recomputed_integrity = int(batched.get("integrity_mismatch_count") or 0) <= int(
        serial.get("integrity_mismatch_count") or 0
    )
    serial_runtime = int(serial.get("generation_runtime_ms") or 0)
    batched_runtime = int(batched.get("generation_runtime_ms") or 0)
    recomputed_speedup = serial_runtime / max(1, batched_runtime)
    runtime_qualified = (
        canary.get("candidate_manifest_equal") is True
        and recomputed_behavior
        and recomputed_integrity
        and recomputed_speedup > 1.0
    )
    behavior_qualified = runtime_qualified and int(batched.get("passed_task_count") or 0) > 0
    expected_adoption = (
        "BATCHED_DEFAULT"
        if behavior_qualified
        else "BATCHED_RUNTIME_ONLY"
        if runtime_qualified
        else "NOT_ADOPTED"
    )
    if bool(canary.get("behavior_non_regression")) != recomputed_behavior:
        hard_gaps.append(gap("generation_mode_behavior_decision_mismatch", {}))
    if bool(canary.get("integrity_non_regression")) != recomputed_integrity:
        hard_gaps.append(gap("generation_mode_integrity_decision_mismatch", {}))
    if abs(speedup - recomputed_speedup) > 1e-5:
        hard_gaps.append(
            gap(
                "generation_mode_speedup_mismatch",
                {"reported": speedup, "recomputed": recomputed_speedup},
            )
        )
    if adoption_state != expected_adoption:
        hard_gaps.append(
            gap(
                "generation_mode_adoption_state_mismatch",
                {"observed": adoption_state, "expected": expected_adoption},
            )
        )
    return {
        "state": state,
        "adoption_state": adoption_state,
        "speedup": round(recomputed_speedup, 6),
        "hard_gaps": hard_gaps,
    }


def gap(kind: str, detail: dict[str, Any], *, severity: str = "hard_gap") -> dict[str, Any]:
    return {"kind": kind, "severity": severity, "detail": detail}


def normalized_function_body_hash(tree: ast.Module) -> str:
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    body = function.body if function is not None else tree.body
    payload = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_replay_is_content_bound(
    report: dict[str, Any],
    live_report_path: Path,
) -> bool:
    """Verify that evaluation-only evidence binds a distinct training receipt."""

    training = report.get("training") if isinstance(report.get("training"), dict) else {}
    if training.get("evaluation_only_replay") is not True:
        return True
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    conditioning = (
        report.get("conditioning") if isinstance(report.get("conditioning"), dict) else {}
    )
    prior = resolve(str(artifacts.get("prior_training_receipt") or ""))
    checkpoint = resolve(str(artifacts.get("checkpoint") or ""))
    return (
        prior.is_file()
        and checkpoint.is_file()
        and prior.resolve() != live_report_path.resolve()
        and str(conditioning.get("prior_training_receipt_sha256") or "")
        == file_sha256(prior)
        and str(conditioning.get("evaluation_base_checkpoint_sha256") or "")
        == file_sha256(checkpoint)
        and conditioning.get("evaluation_replay_contract")
        == "content_bound_checkpoint_and_training_receipt_v1"
    )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
