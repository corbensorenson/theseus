# VIEA Execution Spine And Deterministic Tool Substrate

This document is the implementation checklist for wiring the planning, memory,
evidence, tool, and feedback organs into one execution spine.

Canonical path:

`CommandContract -> ArtifactGraph -> ClaimLedger -> PlanCompilerDAG -> VCMContextPacket -> DeterministicToolCall -> WorkBoardExecution -> Verification -> ResidualOrDogfoodTrace -> LoopClosureCandidate`

## Scope

This work directly serves the current priorities in `AGENTS.md`:

- make planning the default private-work route instead of a dry-run side report;
- make VCM the context substrate for planned work;
- make exact deterministic tools available to learned systems instead of asking
  the model to memorize arithmetic, search, or proof checking;
- emit durable evidence and feedback from every meaningful execution;
- avoid new loose lanes, public benchmark training, external runtime inference,
  arbitrary remote execution, or fallback-return cheating.

## Checklist

- [x] Add a registry-owned deterministic tool substrate surface.
- [x] Define deterministic tool cards with schema, trust tier, cost, replay
  checksum, VCM bindings, and structured failure behavior.
- [x] Register initial local tools:
  - `math.sympy_exact`
  - `math.numeric_interval`
  - `math.linear_algebra`
  - `math.numeric_verify`
  - `math.mpmath_verify`
  - `logic.lean_check`
  - `logic.z3_smt`
  - `rewrite.egraph_minimal`
  - `rewrite.equality_saturation`
  - `search.local_bm25`
  - `search.local_hybrid`
  - `search.vcm_hybrid`
  - `tool.trace_replay`
- [x] Emit tool results as evidence artifacts, not chat text.
- [x] Attach a VCM address/context packet to every tool result.
- [x] Emit local dogfood metadata events without raw private text.
- [x] Emit loop-closure candidates from repeated successful tool traces.
- [x] Add tool-on/tool-off private A/B metrics.
- [x] Teach the plan compiler to route `local_deterministic_tool` nodes to the
  deterministic tool substrate instead of pointing back at itself.
- [x] Include deterministic tool packets in compiled DAG nodes.
- [x] Add private execute mode for bounded local deterministic packets.
- [x] Run old-direct vs compiled-spine private A/B on the same task set.
- [x] Emit durable run ids, node leases, retry policy, checkpoints, VCM
  artifacts, learning traces, training evidence, loop-closure candidates,
  verified procedural tool records, and residuals.
- [x] Ingest deterministic substrate reports into the report evidence store.
- [x] Ingest VIEA execution-spine reports into the report evidence store.
- [x] Ingest deterministic tool cards/results into the VIEA artifact kernel.
- [x] Ingest VIEA execution-spine runs, VCM artifacts, feedback traces,
  residuals, and loop candidates into the VIEA artifact kernel.
- [x] Run syntax checks and private smoke/ablation verification.
- [x] Refresh registry/evidence/artifact reports.

## Verified 2026-06-21

- `reports/deterministic_tool_substrate.json`: `GREEN`, `13` tool cards,
  `15/15` verified private/replay results, exact solve rate `1.0`, tool-on
  solve rate `1.0`, tool-off solve rate `0.0`, external inference calls `0`,
  public training rows `0`, fallback returns `0`.
- `reports/theseus_plan_compiler.json`: `GREEN`, `7` compiled goals, `19`
  compiled nodes, `7` local deterministic tool packets, `13` deterministic
  tool requirements, average VCM pages per node `6.0`, hard gate failures `0`,
  and execute-mode trigger `GREEN`.
- `reports/viea_execution_spine.json`: `GREEN`, old-direct baseline `14`
  cases, compiled-spine execution `14` cases, compiled useful completion rate
  `1.0`, `14` leases, `14` checkpoints, duplicate work `0`, retries `0`,
  residuals `0`, training evidence rows `14`, verified procedural tools `2`,
  public training rows `0`, external inference calls `0`, and fallback returns
  `0`.
- `reports/report_evidence_store.json`: `GREEN`, deterministic tool,
  plan-compiler, VIEA execution-spine, procedural-tool, and research-matrix
  families ingested, current unstored reports `0`.
- `reports/viea_artifact_kernel.json`: `GREEN`, deterministic tool cards,
  tool results, compiled plan nodes, execution-spine run objects, VCM artifacts,
  learning traces, training evidence, procedural tools, residuals, and claims
  indexed; `347` objects and `265`
  relationships.
- `reports/theseus_project_registry.json`: `YELLOW`, with active source
  coverage `1.0`, unregistered active source files `0`, generated source
  artifacts `0`, and hard registry governance violations `0`. Remaining
  pressure is existing duplicate generated report families.

## No-Cheat Boundary

- Public benchmark rows are not used here.
- External inference calls are zero.
- Runtime serving does not use teacher output.
- Fallback returns are disallowed. A tool that cannot solve emits `UNKNOWN`,
  `UNSOLVED`, `TOOL_UNAVAILABLE`, or `TOOL_FAULT` with evidence.
- Tool-assisted and model-only scores must remain separate.

## Boundaries

- Public-like planning/tool benchmark adapters use private local dry-run
  scorers here; no public payloads are fetched or trained on.
- Tool-use training evidence is emitted for governed admission, but this smoke
  does not silently write durable student-training rows.
- The equality-saturation tool is bounded and local; it is not a claim of a
  production-grade e-graph optimizer.
