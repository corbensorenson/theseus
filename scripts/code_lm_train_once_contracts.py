"""Contracts and static policy for train-once Code LM fanout."""

from __future__ import annotations

from typing import Any

from code_lm_private_rows import high_transfer_private_rows_string


BROAD_FLOOR_PUBLIC_CARDS = "source_mbpp,source_evalplus,source_bigcodebench,source_livecodebench"
BROAD_FLOOR_PUBLIC_CASES_PER_CARD = 32
PRIVATE_ROWS = high_transfer_private_rows_string(include_broad_floor_recovery=True)

PHASE_CONTRACTS: dict[str, dict[str, Any]] = {
    "release_cuda_binary": {
        "kind": "build_release_hot_path",
        "target_max_seconds": 1800,
        "gpu_expected": "cuda_when_supported_else_native_cpu",
        "consumer": "code_lm_train_once_fanout",
        "evidence_semantics": "runtime_envelope_only",
    },
    "train_once_checkpoint": {
        "kind": "release_readout_training",
        "target_max_seconds": 7200,
        "gpu_expected": "cuda_when_supported_else_native_cpu",
        "consumer": "decoder_v2_private_ablation_gate",
        "evidence_semantics": "private_training_checkpoint_only_not_public_calibration",
    },
    "checkpoint_fanout_candidate_generation": {
        "kind": "checkpoint_fanout",
        "target_max_seconds": 21600,
        "gpu_expected": "ranker_or_readout_when_supported",
        "consumer": "private_public_transfer_proof",
        "evidence_semantics": "candidate_manifest_evidence_not_public_score",
    },
    "staged_verification_contract": {
        "kind": "verification_policy",
        "target_max_seconds": 0,
        "gpu_expected": False,
        "consumer": "decoder_v2_private_ablation_gate",
        "evidence_semantics": "verification_order_contract",
    },
}

STAGED_VERIFICATION_CONTRACT = [
    {
        "stage": "lint_parse",
        "pass_signal": "python_ast_parse_or_language_parser_ok",
        "fail_fast": True,
        "reward": "syntax_valid_candidate",
        "consumer": "candidate_ranker_prefilter",
    },
    {
        "stage": "compile_or_import",
        "pass_signal": "module_loads_and_entry_point_resolves",
        "fail_fast": True,
        "reward": "interface_load_candidate",
        "consumer": "private_decoder_ablation_gate",
    },
    {
        "stage": "cheap_behavior",
        "pass_signal": "contract_shape_required_constructs_and_smoke_behavior",
        "fail_fast": True,
        "reward": "contract_admissible_candidate",
        "consumer": "private_public_transfer_proof",
    },
    {
        "stage": "sandbox_full_tests",
        "pass_signal": "bounded_private_or_calibration_test_pass",
        "fail_fast": False,
        "reward": "behavioral_success_candidate",
        "consumer": "promotion_gate_after_private_public_transfer",
    },
]

CONTROL_SIGNAL_CONTRACT = {
    "policy": "project_theseus_report_control_signal_contract_v1",
    "report": "code_lm_train_once_fanout",
    "semantics": "control_signal",
    "diagnostic_only_until": [
        "decoder_v2_private_ablation_gate.ready_for_public_calibration",
        "private_public_transfer_proof.ready_for_public_calibration",
    ],
    "consumers": [
        "system_efficiency_audit",
        "asi_wall_breaker_governor",
        "decoder_v2_private_ablation_gate",
        "private_public_transfer_proof",
        "hive_work_board_executor",
    ],
    "measured_effects_required": [
        "phase_timing_delta",
        "candidate_manifest_coverage_delta",
        "public_no_admissible_rate_delta",
        "sts_conditioned_candidate_delta",
    ],
    "public_boundary": (
        "public benchmark tasks may appear only as prompt/signature calibration metadata; "
        "no public tests or solutions become training rows"
    ),
}
