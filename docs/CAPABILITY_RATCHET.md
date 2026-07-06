# The Capability Ratchet in SymLiquid

The Capability Ratchet is the top-level SymLiquid development loop. It combines
three mechanisms already present in the repo:

1. Benchmark pressure from the Benchmaxxing treadmill.
2. Loop closure through verified procedural tools.
3. Compact verified capability through CGS-style ledgers and regression guards.

The operational rule is:

```text
frontier gain required
regression loss forbidden
external inference forbidden
initial mastery threshold = 0.90
ordinary floor threshold = 0.70
threshold decay = 0.01 per attempt after patience
graduated residuals enter escrow
```

## Run

Generate the full ratchet state from local artifacts:

```powershell
python scripts\run_capability_ratchet.py --out reports\capability_ratchet_run.json
```

## Artifacts

| Artifact | Role |
| --- | --- |
| `reports/benchmark_treadmill_status.json` | Benchmark ratchet state: frontier, regressions, anti-Goodhart warnings. |
| `reports/benchmark_ledger.json` | Benchmark lifecycle ledger with capability, wall type, contamination risk, and retirement criteria. |
| `reports/model_ledger.json` | Model ledger with scores, residual map, regression status, and next wall. |
| `reports/public_comparator_ledger.json` | Public apples-to-apples comparator ledger. |
| `reports/babylm_residual_analysis.json` | BabyLM/BLIMP residual clusters by field, term, and rule. |
| `data/babylm_mutated_holdout_seed49.jsonl` | Latest local mutated BabyLM/BLIMP holdout generated from residual pressure families. |
| `reports/babylm_mutated_residual_analysis.json` | Residual clusters on the mutated holdout. |
| `reports/residual_escrow.json` | Active backlog of graduated benchmark tails, recurrence-promoted diagnostics, and reattempt schedules. |
| `reports/capability_ratchet_run.json` | Compiled workflow run log proving the ledgers and registries were refreshed together. |
| `reports/tool_registry.json` | Procedural tool cards for repeated local workflows. |
| `reports/capability_ratchet_report.json` | Combined benchmark/procedural/structural ratchet report. |
| `reports/ratcheting_generative_system_report.json` | Top-level audit showing how completely SymLiquid implements the Ratcheting Generative Systems framework. |
| `reports/ratcheting_modular_intelligence_report.json` | Unified RMI audit for compact structure, active compression, loop closure, benchmark ratcheting, octopus routing, routing memory, and arm lifecycle governance. |
| `reports/octopus_router_report.json` | System-level modular routing report with arm registry, router eval, dynamic loading, safety, and bridge metrics. |
| `reports/arm_registry.json` | ORA arm cards with scope, schemas, permissions, local benchmarks, residuals, lifecycle, and dynamic loading policy. |
| `reports/octopus_router_eval.json` | Local routing benchmark for the resident head/router. |
| `reports/routing_memory.json` | Task-to-arm routing memory and per-arm route outcome memory. |
| `reports/arm_lifecycle_ledger.json` | Specialist add/split/merge/retire lifecycle ledger. |
| `reports/octopus_router_head_report.json` | Local sparse router-head training report and promotion gate. |
| `reports/octopus_router_head_eval.json` | Learned router holdout metrics. |
| `reports/safety_benchmark_ledger.json` | Safety and quarantine checks for high-risk routing, approvals, runtime tiers, and least privilege. |
| `reports/bridge_benchmark_ledger.json` | Bridge benchmark ledger generated from recurring residual escrow. |
| `reports/architecture_gate_report.json` | Pre-training gate that blocks heavy training unless ratchet, ORA, safety, public calibration, residual escrow, and learned routing are green. |

## Current State

Latest local report:

```text
framework=capability_ratchet
active_family=coding_local_sandbox
best_public_calibration_card=source_human_eval_wide_32_tasks_pass_rate_0.78125
active_public_calibration_card=source_evalplus_source_bigcodebench_source_livecodebench_below_floor
latest_mbpp_evalplus_cards=MBPP_32_tasks_0.71875_above_floor_EvalPlus_32_tasks_0.59375_below_floor
next_code_rotation_card=source_agnostic_type_edge_interface_algorithmic_pressure
transfer_interleave=same_family_code_first_then_broader_transfer_if_wall_persists
broad_transfer_matrix=YELLOW_160_public_calibration_tasks_aggregate_pass_rate_0.5125_sts_delta_0.28125
public_code_pass_rate=0.5125
required_public_code_floor=0.70
candidate_promote=false
token_level_student_generation_valid=true
stale_ranker_lane=superseded_by_token_level_code_lm
tool_registry_entries=24
rgs_implementation_score=1.0
rmi_implementation_score=1.0
rgs_implemented_components=12
rgs_partial_components=0
rgs_missing_components=0
ora_implementation_score=1.0
ora_implemented_components=12
ora_arm_count=12
ora_router_selection_accuracy=1.0
routing_memory_entries=10
arm_lifecycle_arms=12
learned_router_exact_set_accuracy=1.0
architecture_gate_ready=true
architecture_gate_passed=14/14
residual_escrow_clusters=50
residual_escrow_cases=50
external_inference_violations=0
```

The ratchet starts ordinary benchmarks at a 90% mastery expectation, but it no
longer treats that number as a permanent hostage condition. After 3 attempts on
the same benchmark family, the graduation threshold decays by 1 percentage
point per attempt toward a 70% floor. Critical failures do not decay. Any
unsolved tail after graduation enters residual escrow, where recurring clusters
are promoted back into active diagnostics.

Code-family rotation now has a transfer-interleave escape hatch. Same-family
public/source code cards are tried first. If public transfer remains below the
floor long enough, the autonomy policy can temporarily interleave broader local
learning pressure, currently local RL memory/control, then return to the queued
code card with transfer artifacts loaded. This does not weaken promotion gates:
public code promotion still requires token-level learned student generation,
clean anti-cheat evidence, no regressions, and honest public/held-out score
semantics.

The seed49 mutated BabyLM/BLIMP anti-Goodhart holdout passed the mastery gate
and is locked as regression. Seed127 is also regression after clearing the
decayed ordinary threshold. The public BLIMP split remains an apples-to-apples
comparator/regression surface with raw scores reported.

## Ratcheting Generative Systems Audit

The RGS auditor now checks whether the paper is represented by executable
SymLiquid artifacts rather than just prose:

```text
report=reports/ratcheting_generative_system_report.json
score=1.0
implemented=benchmark_ledger, model_ledger, time_decayed_mastery_thresholds,
            residual_escrow, public_calibration_track,
            procedural_tool_registry, execution_modes,
            active_compression_substrate, high_bandwidth_embodied_logging,
            safety_and_reflex_layer, bridge_benchmark_protocol,
            octopus_router_architecture
partial=none
missing=none
```

Puffer/Ocean now writes a bounded eventized rollout log with raw windows, event
logs, semantic traces, skill traces, and residual logs:

```text
event_log=reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json
sampled_raw_windows=256
event_count=384
semantic_events=256
skill_events=256
residual_events=0
external_inference_calls=0
```

## Octopus Router Architecture

The local ORA layer is now active:

```text
report=reports/octopus_router_report.json
arm_registry=reports/arm_registry.json
router_eval=reports/octopus_router_eval.json
routing_memory=reports/routing_memory.json
arm_lifecycle_ledger=reports/arm_lifecycle_ledger.json
safety_ledger=reports/safety_benchmark_ledger.json
bridge_ledger=reports/bridge_benchmark_ledger.json
bridge_benchmark=benchmarks/bridges/babylm_wh_gap_bridge.jsonl
status=active_system_level_router_v0
implementation_score=1.0
implemented=12
arms=12
resident_head=1
router_cases=10
router_selection_accuracy=1.0
risk_routing_accuracy=1.0
routing_memory_entries=10
arm_lifecycle_arms=12
estimated_memory_savings=0.6187
learned_router_training=implemented
```

The head/router now uses a deterministic rule router as bootloader/fallback and
a local sparse centroid router head trained from ORA traces:

```text
router_head_report=reports/octopus_router_head_report.json
source_cases=10
augmented_examples=50
holdout_examples=10
exact_set_accuracy=1.0
arm_micro_f1=1.0
risk_routing_accuracy=1.0
promotion_gate_passed=true
```

The pre-training architecture gate is also green:

```text
report=reports/architecture_gate_report.json
ready_for_heavy_training=true
passed=14/14
```

## Three Ratchets

### Benchmark Ratchet

Current frontier expansion:

```text
active_family=coding_local_sandbox
best_public_calibration_card=source_human_eval_wide_32_tasks_pass_rate_0.78125
active_public_calibration_card=source_evalplus_source_bigcodebench_source_livecodebench_below_floor
latest_mbpp_evalplus_cards=MBPP_32_tasks_0.71875_above_floor_EvalPlus_32_tasks_0.59375_below_floor
next_code_rotation_card=source_agnostic_type_edge_interface_algorithmic_pressure
transfer_interleave=same_family_code_first_then_broader_transfer_if_wall_persists
reason=programming/code pressure is the active growth lane; transfer interleave prevents single-wall lock-in
broad_transfer_matrix=YELLOW_160_public_calibration_tasks_aggregate_pass_rate_0.5125_sts_delta_0.28125
public_code_pass_rate=0.5125
next_action=continue source-agnostic semantic residual pressure; BigCodeBench and LiveCodeBench both have 32+ clean tasks and are now below-floor receiver cards
```

Current regression suite:

```text
ocean-noisy-tmaze
ocean-cartpole
ocean-slot-tmaze
ocean-noisy-memory
cgs_hard_governance
ocean-chain
ocean-memory
ocean-tmaze
cgs_frontier_governance
unseen_adversarial_rag
babylm_local_probe
babylm_mutated_holdout
```

Public comparator ledger:

```text
babylm_local_probe
score=0.9243361
residual=0.0756639
status=curriculum_passed_promote_to_regression
rule=report public scores regularly, but promote only when public gains transfer to private/mutated holdouts
```

Residual escrow:

```text
attention_budget=frontier 60%, regression 20%, escrow 10%, public_calibration 10%
active_targets=wh_vs_that_with_gap, high-residual public BLIMP morphology/agreement clusters
rule=recurring escrow clusters or max residual >= 0.10 become active diagnostics
ledger=reports/residual_escrow.json
```

### Procedural Ratchet

The generated tool registry currently includes:

```text
active (22 total, selected examples):
  benchmark_treadmill_runner
  capability_ratchet_orchestrator
  babylm_residual_analyzer
  babylm_mutated_holdout_factory
  unseen_adversarial_rag_mutator
  rust_ffi_puffer_rollout_trainer
  puffer_ocean_eventized_rollout_logger
  babylm_mutated_residual_analyzer
  residual_escrow_builder
  ratcheting_generative_system_auditor
  octopus_router_architecture_builder
  octopus_router_head_trainer
  architecture_gate_runner
  ratcheting_modular_intelligence_auditor
  real_training_preflight_gate
  candidate_promotion_gate
  rtx2060super_ablation_matrix_runner
  one_command_training_ratchet_profile_runner

proposed:
  babylm_frontier_trainer
  regression_guard_runner
```

These are not model-provider tools. They are local workflow tools for preserving
and accelerating SymLiquid development.

### Structural Ratchet

Current architecture hypothesis:

```text
The BabyLM wall remains because sequence-state formation is too shallow for
agreement, binding, ellipsis, and argument-structure residuals.
```

Missing mechanism:

```text
learned liquid/reservoir/VSA grammar state with role, number, animacy,
and dependency slots
```

Public BLIMP residual targets:

```text
determiner_noun_agreement_with_adj_irregular_2
irregular_plural_subject_verb_agreement_1
irregular_plural_subject_verb_agreement_2
determiner_noun_agreement_with_adj_irregular_1
principle_A_reconstruction
wh_vs_that_with_gap_long_distance
```

Mutated holdout residual targets:

```text
irregular_plural_subject_verb_agreement_1
principle_A_domain_1
distractor_agreement_relative_clause
transitive
wh_vs_that_with_gap
animate_subject_trans
```

## Promotion Gate

A new candidate may advance only if it:

1. improves the active frontier;
2. preserves the saturated regression suite;
3. reports residual changes by capability family;
4. validates public benchmark gains on mutated or private holdouts;
5. has `external_inference_calls=0`.

## Next Action

The ratchet currently recommends:

```text
Treat the active mutated holdout as the current anti-Goodhart frontier.
Train grammar-state candidates against public BLIMP plus the active mutated
frontier, but promote only if the frontier clears policy gates and residual
deltas stay bounded.
After promotion, regenerate a new mutated holdout seed before further tuning.
```
