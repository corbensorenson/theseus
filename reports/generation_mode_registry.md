# Generation Mode Registry

- trigger_state: `YELLOW`
- modes: `20`
- comparisons: `14`
- promotable comparisons: `0`
- hard gaps: `0` warnings: `18`

## Modes

| id | status | accepted/s | useful/s | pass | fallback | integrity mismatches |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_mlx_ar_beam_strict_generator_v1 | baseline | 0.05745 | 0.0 | 0/10 | 0 | 0 |
| semantic_head_prefix_guided_mlx_decode_v1 | candidate | 0.028422 | 0.0 | 0/9 | 0 | 0 |
| plan_aux_clean_no_head_mlx_decode_broad4_v1 | diagnostic_baseline | 0.0 | 0.0 | 0/4 | 0 | 0 |
| plan_aux_clean_semantic_head_mlx_decode_broad4_v1 | diagnostic_candidate | 0.08186 | 0.0 | 0/6 | 0 | 0 |
| plan_aux_body_semantics_head_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.042786 | 0.0 | 0/6 | 0 | 0 |
| plan_semantic_slots_dataflow_mlx_decode_broad4_v5 | negative_evidence_not_promotable | 0.0 | 0.0 | 0/4 | 0 | 0 |
| plan_semantic_slots_update_contract_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.128771 | 0.0 | 0/8 | 0 | 0 |
| plan_semantic_slots_plan_subspace_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.23737 | 0.0 | 0/25 | 0 | 0 |
| plan_semantic_slots_body_action_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.093666 | 0.0 | 0/8 | 0 | 0 |
| plan_semantic_slots_algorithmic_replay_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.47221 | 0.0 | 0/65 | 0 | 0 |
| plan_semantic_slots_algorithmic_escape_replay_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.121003 | 0.0 | 0/30 | 0 | 0 |
| plan_semantic_stmt_slots_algorithmic_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.099378 | 0.0 | 0/19 | 0 | 0 |
| plan_semantic_stmt_slots_action_trace_pairwise_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.797497 | 0.0 | 0/69 | 0 | 0 |
| train_once_rust_cpu_sparse_fanout_v1 | baseline_or_infrastructure | 1.308928 | 0.0 | 0/0 | 0 | 0 |
| vcm_mlx_tensor_prefix_descriptor_v1 | runtime_metadata_ready_not_native_kv_claim | 0.0 | 0.0 | 0/0 | 0 | 0 |
| standard_causal_transformer_serial_beam_v1 | diagnostic_baseline | None | None | 0/0 | 0 | 0 |
| standard_causal_transformer_batched_beam_v1 | shadow_candidate | None | None | 0/0 | 0 | 0 |
| speculative_draft_decode_research_v1 | planned | None | None | 0/0 | 0 | 0 |
| plan_semantic_stmt_slots_source_slot_head_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.063754 | 0.0 | 0/47 | 0 | 0 |
| plan_semantic_stmt_expression_slots_vocab_guard_mlx_decode_broad4_v1 | negative_evidence_not_promotable | 0.039802 | 0.0 | 0/96 | 0 | 0 |

## Comparisons

- `broad8_no_plan_vs_semantic_head_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_until_behavior_nonregression_and_speed_lift`
- `broad4_plan_aux_no_head_vs_semantic_head_v1` promotable=`False` accepted_speed_lift=`True` useful_speed_lift=`False` pass_non_regression=`True` decision=`coverage_lift_observed_but_not_promotable_until_functional_pass_moves_above_zero`
- `broad4_plan_aux_head_vs_body_semantics_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_lm_and_plan_loss_improved_but_functional_pass_still_zero`
- `broad4_body_semantics_vs_plan_semantic_slots_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_richer_prefix_regressed_to_zero_verified_learned_candidates`
- `broad4_plan_semantic_slots_vs_update_contract_v1` promotable=`False` accepted_speed_lift=`True` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_candidate_emission_and_runtime_load_recovered_but_task_pass_count_remains_zero`
- `broad4_update_contract_vs_plan_subspace_v1` promotable=`False` accepted_speed_lift=`True` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_plan_prefix_diversity_and_plan_loss_improved_but_functional_pass_remains_zero`
- `broad4_plan_subspace_vs_body_action_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_body_action_weighting_improved_plan_accuracy_but_regressed_decode_coverage_and_kept_zero_pass`
- `broad4_plan_subspace_vs_algorithmic_replay_v1` promotable=`False` accepted_speed_lift=`True` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_private_replay_lowered_lm_loss_but_broad_functional_pass_stayed_zero`
- `broad4_algorithmic_replay_vs_escape_replay_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_escape_replay_shifted_failure_shape_to_decode_starvation_and_zero_pass`
- `broad4_escape_replay_vs_statement_slots_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_statement_slots_recovered_candidate_coverage_but_broad_functional_pass_stayed_zero`
- `broad4_statement_slots_vs_action_trace_pairwise_v1` promotable=`False` accepted_speed_lift=`True` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_action_trace_pairwise_reduced_candidate_count_and_kept_broad_functional_pass_zero`
- `broad4_action_trace_pairwise_vs_source_slot_head_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_source_slot_head_improved_emitted_loadable_candidate_coverage_but_kept_functional_pass_zero`
- `broad4_source_slot_head_vs_expression_vocab_guard_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`not_promotable_expression_slot_canonicalization_improved_auxiliary_accuracy_but_kept_functional_pass_zero_and_body_semantics_collapsed`
- `standard_causal_transformer_serial_vs_batched_beam_v1` promotable=`False` accepted_speed_lift=`False` useful_speed_lift=`False` pass_non_regression=`True` decision=`batched_default_after_exact_candidate_parity_behavior_integrity_and_useful_verified_output_speed_nonregression`

## Warnings

- `standard_causal_transformer_serial_beam_v1` `missing_metric_path`: `{"metric_path": "generation_mode_canary.serial", "missing": ["reports/standard_causal_transformer_survival.json"]}`
- `standard_causal_transformer_serial_beam_v1` `runtime_ms_missing`: `{"refs": ["reports/standard_causal_transformer_survival.json"]}`
- `standard_causal_transformer_batched_beam_v1` `missing_metric_path`: `{"metric_path": "generation_mode_canary.batched", "missing": ["reports/standard_causal_transformer_survival.json"]}`
- `standard_causal_transformer_batched_beam_v1` `runtime_ms_missing`: `{"refs": ["reports/standard_causal_transformer_survival.json"]}`
- `broad8_no_plan_vs_semantic_head_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_plan_aux_no_head_vs_semantic_head_v1` `comparison_not_promotable`: `{"accepted_speed_lift": true, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_plan_aux_head_vs_body_semantics_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_body_semantics_vs_plan_semantic_slots_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_plan_semantic_slots_vs_update_contract_v1` `comparison_not_promotable`: `{"accepted_speed_lift": true, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_update_contract_vs_plan_subspace_v1` `comparison_not_promotable`: `{"accepted_speed_lift": true, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_plan_subspace_vs_body_action_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_plan_subspace_vs_algorithmic_replay_v1` `comparison_not_promotable`: `{"accepted_speed_lift": true, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_algorithmic_replay_vs_escape_replay_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_escape_replay_vs_statement_slots_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_statement_slots_vs_action_trace_pairwise_v1` `comparison_not_promotable`: `{"accepted_speed_lift": true, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_action_trace_pairwise_vs_source_slot_head_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `broad4_source_slot_head_vs_expression_vocab_guard_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`
- `standard_causal_transformer_serial_vs_batched_beam_v1` `comparison_not_promotable`: `{"accepted_speed_lift": false, "candidate_manifest_equal": null, "candidate_task_pass_count": 0, "fallback_non_regression": true, "integrity_non_regression": true, "pass_non_regression": true, "useful_speed_lift": false}`

## Rules

- `speed_claim`: Speed claims must report accepted spans per second and useful verified solutions per second separately.
- `promotion`: A faster mode is not promotable if verifier pass, integrity, context adequacy, no-cheat counters, or fallback burden regresses.
- `kv_claim`: VCM descriptor reuse is not native model KV/prefix-cache parity unless a model-runtime lifecycle test passes.
- `learned_generation`: Generation-mode acceleration does not bypass candidate-integrity or learned-generation claim rules.
- `first_campaign`: AR is the frozen base; MTP is the sole included checkpoint-shaping auxiliary at zero initial weight; speculative decoding is post-hoc and disabled; Medusa, EAGLE, LayerSkip, and sketch-first/LLaDA are retired from the first campaign with explicit re-entry conditions.
