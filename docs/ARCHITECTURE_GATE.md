# SymLiquid Architecture Gate

The architecture gate is the pre-training promotion check. It keeps heavy local
training from starting until the ratchet, RMI audit, ORA router, learned head,
safety ledger, residual escrow, bridge benchmarks, public calibration, routing
memory, arm lifecycle governance, and procedural tooling are all coherent.

Run it directly:

```powershell
py -3.13 scripts\architecture_gate.py --out reports\architecture_gate_report.json
```

It also runs at the end of the compiled ratchet:

```powershell
py -3.13 scripts\run_capability_ratchet.py --out reports\capability_ratchet_run.json
```

## Current Gate

```text
report=reports/architecture_gate_report.json
status=ready_for_heavy_training
ready_for_heavy_training=true
passed=14/14
external_inference_calls=0
```

The gate currently checks:

```text
rgs_complete
rmi_complete
ora_complete
rule_router_eval_passed
learned_router_head_promoted
safety_ledger_passed
regression_suite_present
public_calibration_present
residual_escrow_present
bridge_benchmark_present
procedural_tools_registered
routing_memory_present
arm_lifecycle_governed
external_inference_zero
```

Training policy:

```text
Do not start heavy training unless this gate is green.
Failed gates become ratchet residuals.
Re-run after every architecture change, frontier update, or arm lifecycle change.
```

This gate being green is necessary, not sufficient. Longer runs also need the
training preflight, candidate promotion gate, and resource governor to agree.
See `docs/PROJECT_STATE.md` for the current combined status.
