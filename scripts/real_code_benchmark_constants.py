"""Shared constants for the real-code benchmark graduation lane."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARDS = [
    "source_evalplus",
    "source_human_eval",
    "source_mbpp",
    "source_bigcodebench",
    "source_livecodebench",
]
STREAMS = [
    "context_stream",
    "solver_stream",
    "tool_test_stream",
    "critic_stream",
    "patch_stream",
    "residual_stream",
]
STUDENT_CANDIDATE_SOURCES = {
    "local_theseus_student_checkpoint",
    "student_learning_checkpoint_v1",
    "student_neural_checkpoint_v1",
    "student_token_generator_checkpoint_v1",
    "student_code_lm_checkpoint_v1",
}
STUDENT_PUBLIC_SCORE_CLAIMS = {
    "local_theseus_student_checkpoint": "student_checkpoint_public_task_calibration_only",
    "student_learning_checkpoint_v1": "student_learning_checkpoint_public_task_calibration_only",
    "student_neural_checkpoint_v1": "student_neural_checkpoint_public_task_calibration_only",
    "student_token_generator_checkpoint_v1": "student_token_generator_checkpoint_public_task_calibration_only",
    "student_code_lm_checkpoint_v1": "student_code_lm_checkpoint_public_task_calibration_only",
}
DEFAULT_STUDENT_CANDIDATE_MANIFEST = "reports/student_code_candidates.jsonl"
FORBIDDEN_NON_LEARNED_SCORE_CLAIM = "forbidden_non_token_level_code_generation"
NON_PROMOTABLE_CANDIDATE_MODES = {
    "prompt_program_induction_prior",
    "trace_ranker_over_parent_candidates",
    "neural_ranker_over_parent_candidates",
    "local_template_prior",
    "repair_template",
    "deterministic_program_prior",
    "rust_code_lm_causal_contract_skeleton_decoder",
    "rust_code_lm_causal_contract_skeleton_decoder_sts_conditioned",
    "rust_code_lm_contract_guided_skeleton_decoder",
    "rust_code_lm_contract_guided_skeleton_decoder_sts_conditioned",
    "rust_code_lm_execution_shape_skeleton_decoder",
    "rust_code_lm_execution_shape_skeleton_decoder_sts_conditioned",
    "rust_code_lm_local_adapter_edge_skeleton_decoder",
    "rust_code_lm_local_adapter_edge_skeleton_decoder_sts_conditioned",
    "rust_code_lm_sts_causal_skeleton_decoder_sts_conditioned",
    "rust_code_lm_semantic_plan_v2_token_decoder",
    "rust_code_lm_semantic_plan_v2_token_decoder_sts_conditioned",
    "rust_code_lm_private_body_prototype_token_decoder",
    "rust_code_lm_private_body_prototype_token_decoder_sts_conditioned",
    "rust_code_lm_seeded_body_ngram_token_decoder",
    "rust_code_lm_seeded_body_ngram_token_decoder_sts_conditioned",
    "rust_code_lm_sparse_state_sequence_seeded_decoder",
    "rust_code_lm_sparse_state_sequence_seeded_decoder_sts_conditioned",
    "rust_code_lm_native_sts_stream_expression",
    "rust_code_lm_contract_guided_prompt_program_decoder",
    "rust_code_lm_contract_guided_prompt_program_decoder_sts_conditioned",
    "rust_code_lm_contract_guided_prompt_program_decoder_parser_ast_completion",
    "rust_code_lm_contract_guided_prompt_program_decoder_sts_conditioned_parser_ast_completion",
    "rust_code_lm_contract_guided_token_decoder_same_seed_non_sts_comparator",
}
LEARNED_PRIVATE_BODY_NGRAM_CANDIDATE_MODES = {
    "rust_code_lm_private_multistatement_body_ngram",
}
LEARNED_TOKEN_CANDIDATE_MODE_TOKENS = [
    "token_decoder",
    "contract_transduced_token_decoder",
    "full_body_token_beam",
    "greedy_body_token_decoder",
]
NON_LEARNED_TOKEN_CANDIDATE_MODE_TOKENS = [
    "prompt_program_decoder",
    "same_seed_non_sts_comparator",
    "skeleton",
    "prototype",
    "semantic_plan",
    "native_sts_stream_expression",
]
TEMPLATE_OR_RULE_TOKENS = [
    "program_induction_prior",
    "prompt_program_induction",
    "prompt_program_decoder",
    "heuristic",
    "pattern_",
    "template_",
    "repair_template",
    "canonical",
    "hardcoded",
    "public_task_pattern",
    "baseline_prompt_stub",
    "synthesize_from_contract",
    "skeleton_decoder",
    "semantic_plan_v2",
    "edge_exec_repair",
    "local_adapter_edge",
    "contract_guided_skeleton",
    "private_body_prototype",
    "seeded_body_ngram",
    "sparse_state_sequence_seeded",
]
MAX_PUBLIC_LITERAL_REPR_CHARS = 12000
PUBLIC_TEST_RUNTIME_PRELUDE = "import math\nimport itertools\nimport functools\nimport collections\n\n"
VERIFICATION_STAGE_CONTRACT = [
    {
        "stage": "lint_parse",
        "reward": 0.10,
        "continues_to": "beautiful_code_lint",
        "cpu_role": "fast_static_gate",
        "failure_semantics": "candidate_syntax_training_residual",
    },
    {
        "stage": "beautiful_code_lint",
        "reward": 0.15,
        "continues_to": "candidate_compile",
        "cpu_role": "fast_static_gate",
        "failure_semantics": "style_contract_or_scaffold_junk_residual",
    },
    {
        "stage": "candidate_compile",
        "reward": 0.20,
        "continues_to": "test_harness_compile",
        "cpu_role": "fast_static_gate",
        "failure_semantics": "compiler_training_residual",
    },
    {
        "stage": "test_harness_compile",
        "reward": 0.05,
        "continues_to": "sandbox_runtime_load",
        "cpu_role": "fast_static_gate",
        "failure_semantics": "benchmark_adapter_residual",
    },
    {
        "stage": "sandbox_runtime_load",
        "reward": 0.20,
        "continues_to": "intended_behavior",
        "cpu_role": "bounded_parallel_judge",
        "failure_semantics": "import_or_runtime_load_residual",
    },
    {
        "stage": "intended_behavior",
        "reward": 0.30,
        "continues_to": "pass",
        "cpu_role": "bounded_parallel_judge",
        "failure_semantics": "semantic_behavior_residual",
    },
]
PARALLEL_VERIFICATION_POLICY = {
    "policy": "project_theseus_staged_parallel_code_verification_v1",
    "parallel_unit": "task",
    "early_exit": "candidate stops before sandbox when parse, beautiful-code, candidate compile, or harness compile fails",
    "task_static_cache": "visible prompt prelude, test harness compile result, and runtime env are computed once per task",
    "candidate_quality_cache": "ranker quality computed during candidate ordering is reused by verification",
    "default_worker_env": "THESEUS_CODE_VERIFY_WORKERS",
    "default_worker_rule": "max(2, min(12, os.cpu_count())) unless explicitly bounded by --verification-workers",
    "cpu_role": "judge_only_after_gpu_or_student_candidate_generation",
    "promotion_semantics": "verification_speed_contract_only_not_capability_evidence",
}
