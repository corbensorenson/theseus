# Ratcheting Modular Intelligence in SymLiquid

Ratcheting Modular Intelligence is now an executable audit layer over the full
SymLiquid architecture. It unifies the benchmark ratchet, loop closure, ORA
router, routing memory, arm lifecycle governance, safety layer, and public
calibration track.

Run it through the compiled workflow:

```powershell
python scripts\run_capability_ratchet.py --out reports\capability_ratchet_run.json
```

Or directly:

```powershell
python scripts\ratcheting_modular_intelligence.py --out reports\ratcheting_modular_intelligence_report.json
```

## Current Audit

```text
report=reports/ratcheting_modular_intelligence_report.json
framework=ratcheting_modular_intelligence
implementation_score=1.0
implemented=12
partial=0
missing=0
external_inference_calls=0
```

Implemented RMI components:

```text
compact_generative_structure
active_compression
cognitive_loop_closure
benchmark_ratcheting
octopus_routing
learned_router_head
routing_memory
arm_lifecycle_governance
safety_runtime_tiers
bridge_benchmark_protocol
high_bandwidth_embodied_logging
external_inference_forbidden
```

## Formal State Mapping

The RMI report maps the paper notation into concrete artifacts:

```text
H_t=head_router + learned sparse router head
A_t=12 specialist arms
R_t=rule router, learned router, routing memory
M_t=global, arm-local, shared-task, safety, and residual memory layers
T_t=reports/tool_registry.json
B_t=reports/benchmark_ledger.json
G_t=regression entries in reports/benchmark_ledger.json
E_t=reports/residual_escrow.json
V_t=safety ledger, bridge ledger, architecture gate
```

SparkStream adds the current autonomy/control surfaces:

```text
resource_governor=reports/resource_governor.json
autonomous_goal_runner=reports/autonomous_goal_last.json
real_routing_traces=reports/routing_memory_real_traces.jsonl
checkpoint_chain=reports/checkpoint_registry.json
launch_readiness=reports/autonomy_launch_readiness.json
dashboard=http://127.0.0.1:8787
```

## Modular Memory And Lifecycle

```text
routing_memory=reports/routing_memory.json
entries=10
arm_memories=12
passed_routes=10

arm_lifecycle=reports/arm_lifecycle_ledger.json
arms=12
split_candidates=1
merge_inspections=1
retire_candidates=0
spawn_recommendations=0

arm_lifecycle_governance=reports/arm_lifecycle_governance.json
ready_for_long_autonomy=true
real_trace_count=7
proposal_count=13
```

Routing memory keeps task-to-arm outcomes available for future learned routing.
The lifecycle ledger keeps the arm ecosystem from becoming a quiet monolith:
split, merge, retire, or spawn only when the ledger shows pressure.

Synthetic ORA cases are no longer the only routing memory source. Real
autonomous goals now append traces to `reports/routing_memory_real_traces.jsonl`
and should be folded into future router-head training.

## Training Policy

RMI is now part of the pre-training architecture gate:

```text
architecture_gate=reports/architecture_gate_report.json
ready_for_heavy_training=true
passed=14/14
```

Heavy local training should only proceed through the compiled ratchet while
this gate stays green, the resource governor allows the requested profile, and
the candidate gate does not report promotion blockers.
