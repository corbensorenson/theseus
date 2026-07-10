# Theseus Plan Compiler

The Theseus Plan Compiler is the canonical planning layer for Project Theseus.
It converts a goal into a typed contract, semantic IR DAG, VCM context slices,
executor routes, claim/evidence targets, and replay traces.

By default it compiles and routes only. With `--execute-private`, it invokes the
bounded local VIEA execution spine for private deterministic tool packets. The
broader execution/control surfaces remain:

- `scripts/theseus_control_plane.py`
- `scripts/hive_work_board_executor.py`
- `scripts/autonomy_watchdog.py`
- `scripts/vcm_task_context_bridge.py`

## Contract

Each compiled goal includes:

- objective, non-goals, owner surface, risk, priority, outputs, and acceptance
  tests;
- hard constraint capsules for no public benchmark training, no arbitrary
  remote execution, no fallback returns, registry-first ownership, VCM context,
  and teacher proposal-only behavior;
- a contract hash that all node traces reference.

## Plan IR

Plan nodes are typed atoms with:

- inputs, outputs, dependencies, preconditions, effects, and semantic hash;
- VCM context slice and context hash;
- executor backend and existing surface route;
- schedule layer, critical-path status, slack, and estimated duration;
- claim objects, evidence refs, and localized repair policy.

## Typed Program IR

`scripts/semantic_ir.py` is the code-facing form of the same semantic contract.
It losslessly lowers a generated Python function body into generic typed AST
nodes, dependencies, inputs/outputs, constraints, authority, verifier
obligations, VCM refs, and localized repair scopes. Direct generator candidates
carry compact receipts; the private verifier recomputes their program hash
independently. `typed_semantic_ir_tokens_v1` is available as a trainable target,
but any body reconstructed by the deterministic compiler remains assisted,
zero-credit output rather than learned generation. Malformed, truncated, or
unknown IR returns a typed fault and no fallback body.

## Benchmark Boundary

PlanBench, TravelPlanner, APB, OSWorld, and WebArena-style public planning
benchmarks are calibration-only. The compiler may prepare dry-run adapter
contracts, but it does not fetch public benchmark payloads, train on public
prompts or answers, or spend public calibration.

## Verification

Refresh the compiler in dry-run mode:

```sh
python3 scripts/theseus_plan_compiler.py
```

Run the private execute-mode proof:

```sh
python3 scripts/theseus_plan_compiler.py --execute-private
```

Expected outputs:

- `reports/theseus_plan_compiler.json`
- `reports/theseus_plan_compiled_dags.json`
- `reports/theseus_plan_trace_bundle.jsonl`
- `reports/theseus_plan_compiler_ablation.json`
- `reports/theseus_plan_compiler.md`
- `reports/viea_execution_spine.json`
- `reports/viea_execution_spine_trace.jsonl`
- `reports/viea_execution_spine_vcm_artifacts.jsonl`
- `reports/viea_tool_use_learning_traces.jsonl`
- `reports/viea_tool_use_training_evidence.jsonl`
- `reports/viea_execution_loop_closure_candidates.json`
- `reports/viea_verified_procedural_tools.json`

Current verified execute-mode proof: `reports/viea_execution_spine.json` is
`GREEN` with `14` old-direct baseline cases, `14` compiled-spine cases, `14`
leases, `14` checkpoints, verifier pass rate `1.0`, duplicate work `0`,
retries `0`, residuals `0`, training evidence rows `14`, verified procedural
tools `2`, and no public training rows, external inference calls, or fallback
returns.
