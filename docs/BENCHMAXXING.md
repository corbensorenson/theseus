# Benchmaxxing in SymLiquid

Benchmaxxing is now the benchmark ratchet inside the broader Capability
Ratchet. It owns benchmark lifecycle, saturation, wall diagnosis, and frontier
pressure. The Capability Ratchet adds procedural loop closure and tool-registry
state on top of this benchmark layer.

See `docs/CAPABILITY_RATCHET.md` for the combined workflow.

Benchmaxxing is implemented as a local-only performance ratchet for SymLiquid:

```text
evaluate -> classify benchmark lifecycle -> lock saturated regressions
         -> expose active frontier -> diagnose wall -> recommend next intervention
```

The operational rule is:

```text
frontier gain required, regression loss forbidden, external inference forbidden
```

## Current Artifacts

Run:

```powershell
python scripts\benchmark_treadmill.py --reports reports --out reports\benchmark_treadmill_status.json --benchmark-ledger-out reports\benchmark_ledger.json --model-ledger-out reports\model_ledger.json --public-comparator-ledger-out reports\public_comparator_ledger.json
```

The treadmill writes:

| Artifact | Purpose |
| --- | --- |
| `reports/benchmark_treadmill_status.json` | Current ratchet state, active frontier, saturated regressions, next commands. |
| `reports/benchmark_ledger.json` | Benchmark lifecycle ledger: capability, type, contamination risk, label quality, wall type, retirement criteria. |
| `reports/model_ledger.json` | Model ledger: architecture, data, training, inference process, scores, residual map, next wall. |
| `reports/public_comparator_ledger.json` | Public apples-to-apples comparator surfaces kept visible without letting them replace private mutation pressure. |
| `reports/babylm_residual_analysis.json` | Current BabyLM/BLIMP residual clusters by field, linguistic term, and rule. |
| `reports/ratcheting_generative_system_report.json` | Top-level audit of benchmark ratcheting, residual escrow, procedural memory, public calibration, and remaining framework gaps. |

## Lifecycle Mapping

| Benchmaxxing concept | SymLiquid implementation |
| --- | --- |
| Frontier benchmark | Any tracked family with score below the saturation threshold and above broken threshold. |
| Regression benchmark | Saturated family; preserved in the treadmill as a capability lock. |
| Diagnostic benchmark | Broken or immature family that needs audit before model decisions. |
| Saturation | Starts at `score >= 0.90`; after the patience window, ordinary benchmarks decay by `0.01` per attempt toward a `0.70` floor. |
| Residual escrow | Failed tails from graduated benchmarks stay in `reports/residual_escrow.json` for spaced reattempts and recurrence promotion. |
| Wall diagnosis | Per-family wall labels such as `architecture_training_wall`, `state_or_rollout_wall`, or `evaluation_frontier_wall`. |
| Anti-Goodhart | Public comparator tracking, generated/mutated holdout pressure, external inference violation rejection. |
| Generalization guard | `reports/transfer_generalization_audit.json` checks cross-card spread, shared residual concepts, per-card STS causality, and benchmark-name-specific private pressure. |
| Ratchet action | The next benchmark or architecture step emitted by the treadmill. |

## Current Ratchet State

As of the latest local run:

```text
active_family=coding_local_sandbox
best_public_calibration_card=source_human_eval_wide_32_tasks_pass_rate_0.78125
active_public_calibration_card=source_evalplus_source_bigcodebench_source_livecodebench_below_floor_after_execution_shape_receiver
latest_4_card_receiver_calibration=MBPP_32_tasks_0.6875_EvalPlus_32_tasks_0.59375_BigCodeBench_32_tasks_0.125_LiveCodeBench_32_tasks_0.125_no_leakage_template_wrapper_violations_calibration_only
latest_private_execution_shape_ablation=GREEN_learned_token_decoder_v1_1.0_on_32_execution_shaped_private_heldout_tasks_no_admissible_0.0_template_candidates_0
latest_receiver_candidate_coverage_gate=YELLOW_stale_receiver_manifest_public_eligible_task_coverage_0.101562_no_admissible_task_rate_0.898438_public_calibration_locked
latest_edge_and_type_private_recalibrations=clean_no_lift_128_public_receiver_tasks_pass_rate_0.398438
latest_decoder_v2_execution_shape_diagnostic=non_humaneval_receiver_53_of_128_0.414062_BigCodeBench_8_of_32_clean_gates
latest_algorithmic_planning_private_pressure=960_clean_private_rows_0_private_solution_failures_interval_window_frequency_graph_state_twopointer
latest_sts_skeleton_patch=STS_affects_skeleton_admission_and_ranking_public_positive_typed_interface_private_repair_delta_positive
latest_typed_interface_skeleton_private_pressure=960_clean_private_rows_0_private_solution_failures_signature_type_family_branch_loop_local_return_shape
latest_typed_interface_private_closure=private_only_code_lm_closure_GREEN_private_pass_rate_0.083682_to_0.39749_next_token_delta_0.082302_STS_repair_delta_plus_0.05021_12_improvements_0_regressions
latest_source_agnostic_private_pressure_closure=private_only_GREEN_private_pass_rate_0.062762_to_0.485356_delta_0.422594_next_token_delta_0.084484_STS_repair_delta_plus_0.05021_12_improvements_0_regressions_public_calibration_skipped
latest_decoder_contract_verifier_v1=active_candidate_gate_signature_argument_use_return_shape_ast_branch_loop_local_semantic_family_scaffold_bogus_return_attribute_rejection
latest_algorithmic_full_receiver_attempt=diagnostic_quarantined_48_of_128_0.375_public_receiver_regression_after_stratified_admission
latest_execution_shape_receiver_result=board_executed_128_task_public_calibration_0.382812_clean_MBPP_0.6875_EvalPlus_0.59375_BigCodeBench_0.125_LiveCodeBench_0.125_broad_matrix_0.50625
next_private_decoder_wall=remaining_zero_pass_private_families_list_difference_list_tail_replace_split_list_at_index_word_count_nonempty_substring_count_same_chars_is_anagram_if_receiver_calibration_stays_flat
next_code_rotation_card=fresh_private_pressure_private_closure_then_decoder_v2_private_ablation_gate_then_single_public_receiver_measurement_only_if_gate_green
same_family_rotation=enabled
transfer_interleave=enabled; same-family code first, broader transfer if wall persists
hard_conversation_regression=GREEN_96_cases_217_turns_accuracy_0.9772569444444442_personality_ready_217_of_217
broad_transfer_matrix=YELLOW_160_public_calibration_tasks_best_clean_per_card_aggregate_pass_rate_0.50625_sts_delta_0.26875
public_code_pass_rate=0.50625
required_public_code_floor=0.70
candidate_promote=false
token_level_student_generation_valid=true
template_like_candidate_count=0
loop_closure_candidate_count=0
external_inference_violations=0
rust_work_budget_admission_default=legacy_sequential_work_budget_admission_v1
rust_work_budget_admission_stratified=opt_in_only_THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION
```

The saturated regression suite still includes prior BabyLM/Ocean/CGS/RAG
surfaces and the 96-case hard conversation lane, but the active pressure is
currently programming/code. The next
treadmill action is to close the code promotion with residual escrow and rotate
within the code family; if that still cannot clear the public-transfer floor,
the autonomy policy interleaves broader transfer surfaces and returns with
artifacts loaded rather than grinding one stuck benchmark forever.

Benchmaxxing must optimize transferable concepts, not benchmark identities. The
current generalization audit is `YELLOW`: only `source_human_eval` is above
floor, best-clean-per-card aggregate broad pass rate is
`0.50625`, cross-card spread is `0.5625`, and shared residual concepts are
`type_and_return_shape`, `edge_conditions`, and `admissibility_and_interface`, with
`algorithmic_planning` concentrated on BigCodeBench. Private curriculum should
target these concepts source-agnostically, then use donor/receiver calibration
before claiming progress.

The broad matrix is now explicitly a best-clean-per-card calibration view. It
does not let a newer lower-scoring run overwrite stronger clean evidence, and
it still cannot promote a model because promotion requires one clean checkpoint
report with sufficient coverage and no regressions.

The current BigCodeBench wall is treated as an execution-shaped program wall,
not as a reason to keep rerunning the same four-card calibration. The private
lane now trains only on generated/local CC0 rows for transferable file/path,
CSV/archive, JSON/payload, system/library-API, multi-step local-state, edge, and
return-shape behavior. Public BigCodeBench remains receiver calibration only.
A 4-task BigCodeBench smoke of this lane reached `1/4` with clean no-leakage and
no-template gates, which was integration evidence. The 32-task receiver
calibration first moved BigCodeBench to `1/32` without cross-card regression.
After the execution-shape decoder fixes and private ablation gate, the next
same-seed 32-task receiver calibration moved BigCodeBench to `5/32` while
keeping task-level regressions, template-like candidates, loop-closure
benchmark candidates, external inference calls, and no-cheat violations at
`0`. A follow-up Decoder V2 diagnostic added transferable execution skeleton
fixes for Windows `/tmp` calibration paths, process-control mocks, failed shell
command outputs, and prompt-derived JSON schema/email validation. BigCodeBench
then reached `8/32`, and the four non-HumanEval receiver cards reached
`53/128 = 0.414062` with clean gates. This is useful architecture evidence but
still far below the `0.70` floor.

The follow-up `execution_shape_skeleton_decoder_private_v1` lane is now guarded
by a stricter private ablation gate before any public rerun. The gate compares
`semantic_plan_v2`, `edge_exec_repair_v1`, and
`execution_shape_skeleton_decoder_private_v1` on the same private held-out
execution-shaped tasks, seed, and candidate pool. The current gate is `GREEN`:
skeleton decoding passes `64/64 = 1.0`, `edge_exec_repair_v1` passes
`56/64 = 0.875`, and `semantic_plan_v2` remains `0/64`. The stricter public
gate now clears because skeleton decoding has zero no-admissible-candidate
residuals and at least one passing task in every private execution-shape
category.

That gate was consumed by the board-selected
`execution_shaped_four_card_calibration` task. The same-seed 128-task public
receiver calibration completed cleanly at `49/128 = 0.382812`, with MBPP
`22/32 = 0.6875`, EvalPlus `19/32 = 0.59375`, BigCodeBench `4/32 = 0.125`,
and LiveCodeBench `4/32 = 0.125` in that run. The broad best-clean-per-card
matrix is now `0.50625`, with only `source_human_eval` above floor and no
no-cheat/template/wrapper/external-inference violations. This is useful
calibration evidence, but it is not promotion evidence.

This is useful evidence, not promotion. The remaining broad-transfer wall is
edge and type generalization across cards: `source_evalplus`,
`source_bigcodebench`, and `source_livecodebench` remain below the `0.70`
public floor. The board now uses `reports/transfer_generalization_audit.json`
to rank concept pressure dynamically: `edge_conditions` remains the largest
shared cross-card residual, `type_and_return_shape` and
`admissibility_and_interface` remain shared transfer targets, and
`algorithmic_planning` is the BigCodeBench-heavy receiver wall. The latest
edge-condition private pressure produced 960 clean private rows, then a fresh
same-seed 128-task receiver calibration completed cleanly at `0.398438` with
no leakage/template/wrapper/external-inference violations but no broad/public
lift. `type_and_return_shape` then regenerated 960 clean private rows and the
next same-seed receiver calibration also stayed at `0.398438`.

The follow-up `typed_interface_skeleton` lane attacks the interaction between
type/return-shape and admissibility/interface instead of treating them as
separate walls. It generated 960 private rows with zero private solution
failures. The board then ran `typed_interface_private_closure` with public
calibration skipped. That closure produced useful private learning signal:
private pass rate moved from `0.083682` to `0.39749`, next-token accuracy
delta was `0.082302`, private STS repair delta was `+0.05021`, and the Rust
closure report was `GREEN`. The scheduler therefore records the closure as
complete, but public four-card recalibration should still wait for a
decoder/generator source change or stronger private gate evidence instead of
rerunning merely because fresh private rows exist. The next work should make
the type/interface/edge/algorithmic pressure causal in generation, then retest
transfer.

The current source-agnostic private closure is stronger than the older
typed-interface-only closure. `reports/code_lm_closure_private_pressure_private.json`
consumed the private type/return-shape, type-contract feedback,
typed-interface skeleton, admissibility/interface, edge-condition, and
algorithmic-planning rows with public calibration skipped. It produced a
private pass-rate lift from `0.062762` to `0.485356`, next-token accuracy delta
`0.084484`, private STS repair delta `+0.05021`, 12 STS task-level
improvements, and 0 STS regressions. The private hygiene families
`add_numbers`, `common_elements`, `median_list`, `median_odd`,
`dict_merge_three`, and `title_case_words` are now at `1.0`; the remaining
zero-pass private families are `list_difference`, `list_tail_replace`,
`split_list_at_index`, `word_count`, `nonempty_substring_count`, `same_chars`,
and `is_anagram`. This gate unblocks one receiver calibration, but it is not a
license to churn public cards if transfer remains flat.

The teacher-proposed stratified Rust work-budget admission policy is a useful
diagnostic but not a default training policy. It balanced private high-transfer
row admission under max-work-step trimming, then regressed the public receiver
calibration (`0.296875` 4-card pass rate; broad matrix `0.39375`). It is now
demoted to opt-in through `THESEUS_STRATIFIED_WORK_BUDGET_ADMISSION`; default
Rust closure uses `legacy_sequential_work_budget_admission_v1`, and
`rust_work_budget_admission` in reports is diagnostic-only, not promotion
evidence. This is an explicit anti-Goodhart rule: better-looking private
coverage must be rolled back when broad receiver transfer gets worse.

The high-transfer row loader is now balanced under caps. Before the fix,
`--max-high-transfer-private-train 2000` consumed only
`type_and_return_shape` and `type_contract_decoder_feedback`, starving
`admissibility_and_interface`, `edge_conditions`, `algorithmic_planning`, and
`execution_shaped_programs`. `load_extra_private_train_many` now round-robins
across source files, and the latest diagnostic loaded approximately one sixth
from each source. This prevents accidental benchmark-specific or concept-family
overfitting before a receiver calibration.

The 2026-05-20 algorithmic-planning patch is useful private pressure but not a
public breakthrough. It generated 960 private rows with zero private solution
test failures for interval merge, fixed-window sums, top-k frequency, graph
reachability, alternating-run state, and minimum-subarray-length families.
Decoder V2 now lets STS stream content affect skeleton admission and transfer
ranking directly. Bounded smokes stayed clean and showed positive public STS
delta, but the first full 32-per-card receiver run under stratified admission
regressed from the best `53/128` non-HumanEval diagnostic to `48/128`. That run
is therefore quarantined as diagnostic-only. The next ratchet step is candidate
hygiene and uniform type/interface/return-shape verification before any new full
public calibration.

Threshold policy:

```text
initial_mastery_threshold=0.90
ordinary_floor_threshold=0.70
patience_cycles=3
decay_rate_per_attempt_after_patience=0.01
critical_failure_veto=true
frontier_momentum=graduate when current threshold clears, then escrow the tail
```

## Diagnostic Ladder

Before changing architecture, the treadmill now encodes the Benchmaxxing ladder:

1. Benchmark audit
2. Data improvement
3. Training improvement
4. Inference improvement
5. Architecture change

For the current mutated BabyLM track, the recommended interpretation is:

```text
wall_type=architecture_training_wall
action=continue residual-guided grammar-state work against seed55, while
       preserving seed49 plus public BLIMP as regressions
```

Latest seed49 mutated BabyLM residual pressure:

```text
score=0.9814583
worst_field=syntax residual=0.0389
worst_term=ellipsis residual=0.0491
second_term=filler_gap_dependency residual=0.0400
worst_rule=wh_vs_that_with_gap residual=0.5000
```

Residual escrow currently tracks 46 clusters and 50 sampled cases. Recurring or
high-residual clusters are reactivated as diagnostics, while ordinary tail items
are reattempted on a spaced cadence so the frontier keeps moving.

## No External Inference

Any report with `external_inference_calls > 0` is marked invalid for SymLiquid competition tracking. Public model scores can be used as reference numbers from published leaderboards, but SymLiquid benchmark runs must be local and standalone.
