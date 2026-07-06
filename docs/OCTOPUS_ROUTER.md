# Octopus Router Architecture in SymLiquid

SymLiquid now has a system-level modular router layer. The resident head/router
selects bounded specialist arms, grants scoped memory and permissions, measures
dynamic loading, and returns structured routing decisions for verification.

Run it through the compiled ratchet:

```powershell
python scripts\run_capability_ratchet.py --out reports\capability_ratchet_run.json
```

Or directly:

```powershell
python scripts\octopus_router.py --out reports\octopus_router_report.json
python scripts\train_octopus_router_head.py --router-eval reports\octopus_router_eval.json --arm-registry reports\arm_registry.json --out reports\octopus_router_head_report.json
python scripts\octopus_router_head_gate.py --gate
python scripts\octopus_router.py --router-head-report reports\octopus_router_head_report.json --router-head-eval reports\octopus_router_head_eval.json --out reports\octopus_router_report.json
```

## Artifacts

| Artifact | Role |
| --- | --- |
| `reports/octopus_router_report.json` | Top-level ORA report. |
| `reports/arm_registry.json` | Arm cards with scope, schemas, tools, memory, permissions, benchmarks, residuals, lifecycle, and dynamic loading policy. |
| `reports/octopus_router_eval.json` | Router benchmark decisions, permission envelopes, composition plans, and dynamic loading metrics. |
| `reports/routing_memory.json` | Task signatures, selected arms, route outcomes, permission summaries, and per-arm routing memory. |
| `reports/routing_memory_real_traces.jsonl` | Real dashboard/daemon goal routes appended by the autonomous goal runner. |
| `reports/arm_lifecycle_ledger.json` | Add/split/merge/retire governance ledger for specialist arms. |
| `reports/arm_lifecycle_governance.json` | Operational lifecycle report with arm validation, real usage telemetry, and split/merge/update/deprecation proposals. |
| `configs/arm_lifecycle_policy.json` | Protected arms, thresholds, and review requirements for lifecycle actions. |
| `configs/arm_sucker_policy.json` | High-transfer arm and low-transfer sucker policy. |
| `reports/arm_sucker_registry.json` | Ready/blocked arm-attached suckers, routing contracts, transfer matrix, and maintenance packets. |
| `reports/arm_sucker_registry.md` | Human-readable arm-sucker transfer hierarchy summary. |
| `reports/octopus_router_trace_dataset.json` | Augmented local task-to-arm traces used to train the sparse router head. |
| `reports/octopus_router_head_model.json` | Dependency-free sparse centroid router-head model. |
| `reports/octopus_router_head_eval.json` | Holdout evaluation for learned arm-set routing, risk routing, and contrastive wrong-labelset margins. |
| `reports/octopus_router_head_report.json` | Router-head training report, VIEA records, and promotion gate. |
| `reports/octopus_router_head_gate.json` | Independent gate for real schema-bound traces, contrastive negatives, no-cheat counters, and no generation-credit leakage. |
| `reports/safety_benchmark_ledger.json` | Safety/quarantine tests for high-risk routing and least privilege. |
| `reports/bridge_benchmark_ledger.json` | Bridge benchmark ledger created from residual escrow. |
| `benchmarks/bridges/babylm_wh_gap_bridge.jsonl` | Generated bridge benchmark for the recurring `wh_vs_that_with_gap` residual. |

## Current State

Latest ORA report:

```text
status=active_system_level_router_v0
implementation_score=1.0
implemented=14
partial=0
missing=0
possible=14
arms=16
resident_head=1
router_cases=14
router_selection_accuracy=1.0
risk_routing_accuracy=1.0
routing_memory_entries=14
arm_lifecycle_arms=16
arm_lifecycle_governance_ready=true
schema_bound_real_routing_traces=10
learned_router_exact_set_accuracy=1.0
learned_router_risk_routing_accuracy=1.0
learned_router_contrastive_negatives=240
learned_router_holdout_contrastive_negatives=42
learned_router_contrastive_accuracy=1.0
candidate_generation_credit=0
estimated_memory_savings=0.739
external_inference_calls=0
```

The head now has two layers: a deterministic rule router that acts as the
bootloader/fallback, and a local sparse centroid head trained from ORA routing
traces. The trainer defaults to `reports/routing_memory_real_traces.jsonl` when
it exists, emits contrastive wrong-labelset negatives, and fails promotion if
the head cannot beat those negatives on holdout cases. The learned head is
deliberately small and inspectable so routing can be gated before heavy model
training begins, and router-head selections remain non-generative evidence.

## Arms And Suckers

Arms hold high-transfer capability. Suckers are loadable, low-transfer
specializations attached to an arm. For example, `video_game_play_arm` owns
generic game-control skills such as observation normalization, action mapping,
reward/done normalization, replay traces, and controller priors. It can then
load `minecraft_open_world_sucker`, `crafter_bridge_sucker`,
`minecraft_java_local_sucker`, or `emulator_gba_sucker` for a specific
environment without bloating the top-level arm set.

The same pattern applies to drone control: `drone_racing_control_arm` keeps the
general sim/control safety contract, while `gym_pybullet_hover_sucker`,
`pyflyt_waypoint_sucker`, and `ai_grand_prix_sitl_sucker` attach environment
details. A sucker becomes a new arm only after it repeatedly transfers across
siblings or needs distinct permissions/runtime/safety boundaries. See
`docs/ARM_SUCKER_HIERARCHY.md` for the full contract.

## Arms

The active registry contains:

```text
head_router
benchmark_ratchet_arm
babylm_grammar_arm
bridge_benchmark_arm
residual_governance_arm
puffer_ocean_control_arm
puffer_ocean_logging_arm
drone_racing_control_arm
python_runtime_compliance_arm
adversarial_rag_arm
rust_cuda_systems_arm
loop_closure_tool_arm
public_calibration_arm
safety_reflex_arm
```

Each arm has:

```text
capability_scope
input_schema
output_schema
local_tools
local_memory
permission_boundary
runtime_tier
benchmark_frontier
regression_suite
residual_escrow
reliability_score
lifecycle_status
dynamic_loading policy
retirement_criteria
```

## Safety And Quarantine

The safety ledger currently passes:

```text
high_risk_routes_include_safety_arm=true
critical_routes_require_human_approval=true
no_external_inference_in_permission_envelopes=true
runtime_tiers_present=true
quarantine_domains_present=true
dynamic_loading_manifest_present=true
```

High-risk and critical routes get the safety/reflex arm. Critical routes also
receive explicit human-approval side effects.

## Dynamic Loading

Router eval measures system-level sparse activation:

```text
cache_capacity_non_head_arms=4
cold_loads=26
warm_hits=7
evictions=22
avg_loaded_memory_mb=652.8
monolith_memory_mb=1712
estimated_memory_savings=0.6187
```

This is a first-pass local estimate. The next step is to replace the estimate
with actual process/module memory telemetry as arms become executable services.

## Routing Memory And Lifecycle

ORA now writes the modular memory and lifecycle artifacts expected by RMI:

```text
routing_memory=reports/routing_memory.json
routing_memory_entries=10
arm_memories=14
passed_routes=12

arm_lifecycle_ledger=reports/arm_lifecycle_ledger.json
arms=14
split_candidates=1
merge_inspections=1
retire_candidates=0
spawn_recommendations=0

arm_lifecycle_governance=reports/arm_lifecycle_governance.json
ready_for_long_autonomy=true
schema_errors=0
unknown_selected_arm_count=0
proposal_count=15
```

Routing memory is the seed for future learned routing beyond the hand-built
benchmark cases. The lifecycle ledger keeps arm growth disciplined: split when
bloated, merge when redundant, retire when stale, and spawn only when recurring
pressure justifies a new specialist.

The dashboard and daemon now append real task-to-arm traces through
`scripts/autonomous_goal_runner.py`. These traces should be used to retrain the
learned head after enough real goals accumulate, so the router moves from
synthetic ORA coverage toward lived project behavior.

The current governance proposal is to inspect `loop_closure_tool_arm` for a
split into trajectory logging, tool synthesis, tool verification, and tool
retirement sub-arms. It also flags a low-priority merge inspection between
`benchmark_ratchet_arm` and `bridge_benchmark_arm`, and watches
`babylm_grammar_arm` because residual surface plus bloat is rising.

## Bridge Benchmark

The bridge benchmark factory generated 12 `wh_vs_that_with_gap` cases:

```text
bridge_benchmark=benchmarks/bridges/babylm_wh_gap_bridge.jsonl
case_count=12
source_residual=wh_vs_that_with_gap
```

This turns the top residual escrow target into an active diagnostic before the
next grammar-state architecture change.

## Learned Router Head

The current learned head is trained from local routing traces only:

```text
dataset=reports/octopus_router_trace_dataset.json
model=reports/octopus_router_head_model.json
eval=reports/octopus_router_head_eval.json
source_cases=10
augmented_examples=50
train_examples=40
holdout_examples=10
exact_set_accuracy=1.0
arm_micro_f1=1.0
risk_routing_accuracy=1.0
promotion_gate_passed=true
```

Next router work is to append real task-to-arm traces, add contrastive negative
routes as arms multiply, and keep the learned head behind the architecture
gate before every major training run.
