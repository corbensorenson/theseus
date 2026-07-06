# Ratcheting Generative Systems in SymLiquid

Ratcheting Generative Systems is now an executable audit layer around
SymLiquid's development loop. It verifies that the system is not just training
models, but turning pressure into structure:

```text
benchmark frontier -> attempt -> residual map -> loop closure -> verification
                   -> regression lock-in -> frontier expansion
```

Run the full local workflow:

```powershell
python scripts\run_capability_ratchet.py --out reports\capability_ratchet_run.json
```

That command refreshes the benchmark ledger, model ledger, residual analyses,
residual escrow, tool registry, ORA router artifacts, learned router head,
architecture gate, capability report, and the RGS audit report.

## Current Audit

Latest local RGS report:

```text
report=reports/ratcheting_generative_system_report.json
policy=local_only_no_external_inference
implementation_score=1.0
implemented=12
partial=0
missing=0
```

Implemented:

```text
benchmark_ledger
model_ledger
time_decayed_mastery_thresholds
residual_escrow
public_calibration_track
procedural_tool_registry
execution_modes
active_compression_substrate
high_bandwidth_embodied_logging
safety_and_reflex_layer
bridge_benchmark_protocol
octopus_router_architecture
```

No RGS framework component is currently marked partial or missing. The ORA
subsystem is also complete for the current pre-training architecture gate.

ORA subsystem:

```text
report=reports/octopus_router_report.json
implementation_score=1.0
implemented=12
arms=12
router_selection_accuracy=1.0
risk_routing_accuracy=1.0
routing_memory_entries=10
arm_lifecycle_arms=12
estimated_memory_savings=0.6187
learned_router_training=implemented
learned_router_exact_set_accuracy=1.0
```

RMI subsystem:

```text
report=reports/ratcheting_modular_intelligence_report.json
implementation_score=1.0
implemented=12
partial=0
missing=0
```

## Threshold And Momentum Policy

Ordinary capability benchmarks start with a high mastery target and then avoid
tail obsession:

```text
initial_mastery_threshold=0.90
ordinary_floor_threshold=0.70
patience_cycles=3
decay_mode=per_attempt_after_patience
decay_rate_per_cycle=0.01
stall_epsilon=0.005
stall_window=3
residual_escrow_budget=0.10
critical_failure_veto=true
```

A graduated benchmark becomes regression pressure, not a claim of perfect
mastery. Remaining failures enter residual escrow and recur as diagnostics if
they keep showing up.

## Public Calibration

Public benchmarks stay visible for apples-to-apples comparison, but they do not
replace private mutation pressure:

```text
public_comparator=babylm_local_probe
score=0.9243361
residual=0.0756639
role=public calibration and regression
```

The seed49 mutated BabyLM holdout and later BabyLM mutated frontiers are now
regression/background evidence. The active promotion-facing frontier is code:

```text
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
score_semantics=student_code_lm_checkpoint_public_task_calibration_only
```

## Procedural Memory

The tool registry is the procedural-memory surface for repeated local
development workflows:

```text
active_tools=22
proposed_tools=2
registry=reports/tool_registry.json
```

Active tools include the benchmark treadmill runner, capability ratchet
orchestrator, residual analyzers, mutated holdout factory, Rust FFI Puffer
rollout trainer, eventized rollout logger, residual escrow builder, and the RGS
auditor, the ORA builder, the learned router-head trainer, and the architecture
gate runner, plus the RMI auditor.

## Embodied Logging

Puffer/Ocean now has a high-bandwidth eventized logging sidecar:

```text
event_log=reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json
env=ocean-slot-tmaze
feature_set=slot_tmaze_recurrent_linear_v1
sampled_raw_windows=256
event_count=384
semantic_events=256
skill_events=256
residual_events=0
external_inference_calls=0
```

The log contains bounded raw windows, event logs, semantic traces, skill traces,
and residual logs. It gives the embodied ratchet enough structure to discover
future loops and enough detail to debug failures without storing unbounded
streams.

## Honest Gaps

The current implementation pressure should be:

```text
1. Improve the seed55 mutated BabyLM frontier.
2. Convert wh_vs_that_with_gap into a bridge diagnostic or architecture test.
3. Append real task-to-arm traces and retrain the learned router head before each gate.
4. Keep ORA safety/quarantine tests as candidate-promotion blockers.
5. Keep public calibration periodic while private/live frontiers drive training.
```

This is the current SymLiquid path: keep the floor locked, keep the frontier
moving, and require `reports/architecture_gate_report.json` to stay green
before heavier local training starts.
