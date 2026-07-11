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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "standard_causal_transformer_survival.json"
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
) -> dict[str, Any]:
    hard_gaps: list[dict[str, Any]] = []
    adoption_gaps: list[dict[str, Any]] = []
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

    if not report_path.exists():
        hard_gaps.append(gap("report_missing", {"path": rel(report_path)}))
    if not candidates_path.exists():
        hard_gaps.append(gap("candidate_manifest_missing", {"path": rel(candidates_path)}))
    if report.get("policy") != "project_theseus_standard_causal_transformer_survival_v1":
        hard_gaps.append(gap("report_policy_mismatch", {"policy": report.get("policy")}))
    if architecture.get("family") != "standard_decoder_only_causal_transformer":
        hard_gaps.append(gap("architecture_family_mismatch", {"architecture": architecture}))
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
