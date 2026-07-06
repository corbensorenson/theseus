# Project State

Last consolidated: 2026-06-22 local / 2026-06-22 UTC.

This file is the human-readable operational source of truth for Project
Theseus / SparkStream. Older design documents remain useful background, but
current claims should be checked here first and then against the JSON reports
named below.

## 2026-06-22 Public Measurement Policy And Partial Diagnostic Readout

Public benchmark execution is not throttled by a calendar or monthly run cap.
The active policy is a governed run registry: fresh measurement surfaces may
run immediately, while exact consumed surfaces remain blocked so Theseus cannot
score-fish or train on benchmark payloads.

- `configs/permissive_growth_policy.json`,
  `configs/public_benchmark_contract_v1.json`,
  `scripts/operator_bounded_public_calibration.py`,
  `scripts/public_transfer_lift_v2_packet.py`, and
  `scripts/public_transfer_next_surface_planner.py` now use
  `governed_measurement_run_registry` / run-registry language. Active reports
  expose `time_period_run_cap_enabled=false` and
  `calendar_throttle_enabled=false` plus
  `fresh_surfaces_calendar_throttled=false`; there is no monthly public-run
  throttle in the execution path, and fresh clean surfaces are configured to run
  immediately once frozen and registry-authorized.
- `AGENTS.md` now treats fresh frozen public measurements as authorized through
  the governed run registry by default. The legacy
  `reports/public_calibration_operator_lock.flag` is a circuit-breaker/audit
  artifact, not a calendar throttle or blanket blocker.
- Fresh non-headline public measurements now run through
  `scripts/theseus_benchmark_measurement.py`. When a full 5-card headline
  slice is locally exhausted after consumed-surface exclusions, the command
  plans the largest fresh full-per-card subset as a clearly labeled partial
  diagnostic instead of reporting a vague wall.
- The prior fresh partial diagnostic ran once on
  `public_transfer_measurement_partial_diagnostic_seed3_5x64`:
  `42/128 = 0.328125` across MBPP and BigCodeBench. Per card: MBPP `0/64`,
  BigCodeBench `42/64`. It wrote `0` public training rows, used `0` external
  inference calls, and credited `0` template/fallback candidates.
- A subsequent fresh diagnostic ran once on
  `public_transfer_measurement_mbpp_signature_repair_seed4_5x30` after the
  public/unknown one-argument signature repair: MBPP `0/30 = 0.0`. It wrote
  `0` public training rows, used `0` external inference calls, and credited
  `0` template/fallback candidates. This means the arity/interface shape patch
  was not enough; the remaining wall on this slice is learned semantic
  algorithm quality.
- A later fresh diagnostic ran once on
  `public_transfer_measurement_bigcodebench_post_semantic_ranker_seed5_5x64`
  after the semantic intent/ranker patch: BigCodeBench `18/64 = 0.28125`. It
  wrote `0` public training rows, used `0` external inference calls, and
  credited `0` template/fallback candidates. This is a real improvement over
  the immediately prior MBPP-only diagnostic, but it is still a one-card
  partial diagnostic and must not be reported as a broad five-card headline
  transfer score.
- The next clean BigCodeBench-only diagnostic ran once on
  `public_transfer_measurement_bigcodebench_diagnostic_partial_diagnostic_seed7_1x64`:
  BigCodeBench `15/64 = 0.234375`. It wrote `0` public training rows, used `0`
  external inference calls, credited `0` template/fallback candidates, and the
  no-rerun dry run now blocks the exact surface through the run registry and
  existing output/trace artifacts. The corrected `1x64` suffix is intentional:
  this is one card, not a five-card headline. The result is weaker than the
  previous BigCodeBench diagnostic (`18/64`), so the honest wall is still
  public semantic transfer, especially algorithm choice and candidate quality,
  not missing candidates or artificial budget friction.
- The next clean BigCodeBench-only diagnostic ran once on
  `public_transfer_measurement_partial_diagnostic_seed8_1x64` after the
  generator-side varargs/scaffold guardrail repair: BigCodeBench
  `39/64 = 0.609375`. It wrote `0` public training rows, used `0` external
  inference calls, credited `0` template/fallback candidates, and recorded
  `0` quality-blocked test passes. The matching no-rerun dry run now blocks
  that exact surface through the run registry with
  `surface_consumed_or_per_surface_limit_reached`. This is a material
  one-card diagnostic improvement over seed7 (`15/64`) but still not a
  five-card headline transfer claim. The main remaining public residual is
  algorithm choice (`19`) plus dependency handling (`6`).
- The latest clean BigCodeBench-only diagnostic ran once on
  `public_transfer_measurement_partial_diagnostic_seed9_1x64` after the
  structural candidate pool fairness repair: BigCodeBench
  `45/64 = 0.703125`. It wrote `0` public training rows, used `0` external
  inference calls, credited `0` template/fallback candidates, and recorded
  `0` quality-blocked test passes. The matching no-rerun dry run blocks that
  exact surface through the run registry with
  `surface_consumed_or_per_surface_limit_reached`. This is a further one-card
  diagnostic improvement over seed8 (`39/64`) and seed7 (`15/64`), but still
  not a five-card headline transfer claim. The current public residual is
  algorithm choice (`14`) plus dependency handling (`5`).
- `reports/bounded_public_transfer_residual_mining_public_transfer_measurement_mbpp_signature_repair_seed4_5x30.json`
  is `YELLOW` and metadata-only. Current residual pressure is algorithm choice
  (`30`). Public prompts, tests, solutions, traces, score labels, and candidate
  code are not emitted as training rows.
- `reports/private_residual_target_consumer_public_transfer_measurement_mbpp_signature_repair_seed4_5x30.json`
  is now `GREEN`: target coverage `1.0`, unresolved targets `0`, public
  training rows `0`, fallback returns `0`, and external inference calls `0`.
  The consumer now accepts fresh private category-level evidence for residual
  targets without requiring a legacy exact-family bucket when the category is
  covered, non-leaking, and verifier-clean.
- `reports/bounded_public_transfer_residual_mining_public_transfer_measurement_bigcodebench_post_semantic_ranker_seed5_5x64.json`
  is `YELLOW` and metadata-only. Current residual pressure is algorithm choice
  (`45`) plus dependency handling (`1`). The matching private consumer,
  `reports/private_residual_target_consumer_public_transfer_measurement_bigcodebench_post_semantic_ranker_seed5_5x64.json`,
  is `GREEN`: target coverage `1.0`, unresolved targets `0`, public training
  rows `0`, fallback returns `0`, and external inference calls `0`.
- `reports/bounded_public_transfer_residual_mining_public_transfer_measurement_bigcodebench_diagnostic_partial_diagnostic_seed7_1x64.json`
  is `YELLOW` and metadata-only. Current residual pressure is algorithm choice
  (`47`) plus dependency handling (`2`). The matching private consumer,
  `reports/private_residual_target_consumer_public_transfer_measurement_bigcodebench_diagnostic_partial_diagnostic_seed7_1x64.json`,
  is `GREEN`: target coverage `1.0`, unresolved targets `0`, public training
  rows `0`, fallback returns `0`, and external inference calls `0`. That
  means the private residual probes are saturated relative to this failure
  category; the next useful work is improving learned semantic generation and
  selection so those private repairs transfer to fresh public tasks.
- `reports/bounded_public_transfer_residual_mining_public_transfer_measurement_partial_diagnostic_seed8_1x64.json`
  is `YELLOW` and metadata-only. Current residual pressure is algorithm choice
  (`19`) plus dependency handling (`6`). The matching private consumer,
  `reports/private_residual_target_consumer_public_transfer_measurement_partial_diagnostic_seed8_1x64.json`,
  is now `GREEN`: target coverage `11/11`, unresolved targets `0`, public
  training rows `0`, fallback returns `0`, and external inference calls `0`.
- `reports/bounded_public_transfer_residual_mining_public_transfer_measurement_partial_diagnostic_seed9_1x64.json`
  is `YELLOW` and metadata-only. Current residual pressure is algorithm choice
  (`14`) plus dependency handling (`5`). The matching private consumer,
  `reports/private_residual_target_consumer_public_transfer_measurement_partial_diagnostic_seed9_1x64.json`,
  is `GREEN`: target coverage `11/11`, unresolved targets `0`, public training
  rows `0`, fallback returns `0`, and external inference calls `0`. That
  means the private residual probes are no longer the immediate blocker for
  this one-card diagnostic; the next useful work is staging legitimate broader
  public sources and improving learned semantic generation so the current
  private repairs transfer outside the BigCodeBench slice.
- The Rust token generator now rejects erased `*args` signatures and
  placeholder scaffold bodies before they can be marked promotion-eligible.
  The fresh private guard probe,
  `reports/candidate_floor_v2_varargs_scaffold_guard_private.json`, is
  `GREEN`: `96/96` private verifier pass, `768` learned full-body candidates,
  `0` varargs candidates, `0` placeholder scaffold candidates, `0` guardrail
  failures, public prompts/tests/solutions embedded `false`, and external
  inference calls `0`.
- The Rust token generator now has a stricter public/unknown structural reuse
  rule. General intent labels cover common algorithm families such as sorting,
  integer counting, palindrome/reverse, prime, and recurrence tasks; `sorted(...)`
  is treated as an algorithmic/iterative structure; and public/unknown
  structural body reuse requires visible prompt/semantic/intent overlap. This
  directly addresses the MBPP failure mode where unrelated list-shaped private
  bodies were selected for tasks like `pancake_sort`.
- `reports/candidate_floor_v2_mbpp_residual_current_source.json` is `GREEN` on
  a fresh current-source private probe: `160/160` private verifier pass,
  candidate coverage `1.0`, `1280` learned full-body candidates,
  fallback/template candidates `0`, public prompts/tests/solutions embedded
  `false`, and external inference calls `0`.
- `reports/candidate_floor_v2_bigcodebench_post_semantic_ranker_current_source.json`
  is `GREEN` on the latest public-residual repair queue: `192/192` private
  verifier pass, category coverage for algorithm choice, edge cases, and
  dependency-free candidate behavior, candidate coverage `1.0`, `1536` learned
  full-body candidates, fallback/template candidates `0`, public
  prompts/tests/solutions embedded `false`, and external inference calls `0`.
- `reports/public_transfer_next_surface_planner.json` now plans the next fresh
  registry surface without running calibration. The latest dry plan proposes
  `public_transfer_lift_seed6_5x64` and reports
  `calendar_throttle_enabled=false`,
  `fresh_surfaces_calendar_throttled=false`, and
  `fresh_surface_execution_policy=run_immediately_when_frozen_registry_surface_is_clean`.
  Prior materialized checks showed the local balanced 5-card source is
  exhausted after consumed-surface exclusions. That is a public-source capacity
  wall, not a monthly throttle, operator lock, or code-path throttle.
- The canonical assistant tool route now executes deterministic tool evidence
  instead of only attaching a registry pointer. `reports/theseus_assistant_e2e.json`
  is `GREEN` with `4/4` assistant cases, session memory verified, CLI code
  route verified, and tool evidence `GREEN`: `15` deterministic tool results,
  exact solve rate `1.0`, tool-on solve rate `1.0`, public training rows `0`,
  external inference calls `0`, and fallback returns `0`. The user-facing
  `theseus chat --intent tool` path also produces the same tool evidence.
- The Hive operator chat surface now shares that canonical assistant path.
  Local, auto, best, and explicit local-node targets on
  `POST /api/hive/operator/chat` run `scripts/theseus_assistant_runtime.py`
  with checkpoint memory, VCM context, deterministic tool/code evidence, and
  local dogfood feedback. Explicit remote routes still queue the registered
  bounded `checkpoint_chat` task. The mobile/PWA operator UI now sends a stable
  per-device session id plus intent and feedback (`completed`, `accepted`,
  `missed`, `ignored`, `corrected`) and shows assistant state/tool/code/VCM
  metrics from `GET /api/hive/operator/status`.
- `reports/theseus_assistant_state_report.json` remains `YELLOW`, now
  registry-driven rather than pinned to stale residual paths. Real blockers are
  public-source exhaustion for a fresh 5-card headline and the fact that the
  newest score is a BigCodeBench-only partial diagnostic, not an apples-to-
  apples 5-card claim. After the latest run, the local staged 5-card 5x64
  surface has MBPP `0/64`, EvalPlus `1/64`, HumanEval `0/64`, and
  LiveCodeBench `0/64` unused rows available; BigCodeBench still has enough
  rows for another one-card diagnostic, but that should not be treated as a
  broad public transfer claim.
- `scripts/checkpoint_chat.py` now captures multi-word session codenames
  instead of truncating them to the first token. The ad hoc recall probe and
  `reports/theseus_assistant_e2e.json` both verify exact phrase recall, and
  `reports/theseus_assistant_state_report.json` exposes
  `assistant_session_memory_exact_codename_recalled=true` and
  `assistant_session_memory_constraint_recalled=true`.
- `reports/theseus_assistant_e2e.json` was refreshed again after the seed8
  diagnostic and remains `GREEN`: `4/4` assistant cases, CLI code route
  `GREEN`, session memory loaded `12` prior turns, exact phrase recall
  verified, deterministic tool evidence `GREEN` with `15` results and exact
  solve rate `1.0`, public training rows `0`, external inference calls `0`,
  and fallback returns `0`. The assistant status path now sees
  `public_transfer_measurement_partial_diagnostic_seed8_1x64` at
  `39/64 = 0.609375`.
- `theseus feedback <accepted|missed|ignored|corrected|completed>` is now the
  local post-hoc feedback path for CLI assistant use. It marks the latest or
  specified assistant artifact after the user has read the answer, writes only
  redacted dogfood metadata, refreshes the private dogfood training bridge, and
  records `0` public training rows, `0` external inference calls, and `0`
  fallback returns. `reports/theseus_assistant_e2e.json` now verifies this path
  with `posthoc_feedback_case_passed=true`, `event_written=true`,
  `training_bridge_state=GREEN`, and `posthoc_feedback_training_rows_written=1`.
  `reports/theseus_assistant_state_report.json` exposes the same fields as the
  canonical assistant state.
- `scripts/hive_node.py` now prefers the canonical assistant state and
  benchmark measurement reports for operator benchmark status before falling
  back to older E2E or last-chat artifacts. The canonical assistant state now
  reports the current surface as `source_bigcodebench`, score `0.609375`,
  public training rows `0`, external inference calls `0`, and fallback returns
  `0`; older Hive smoke artifacts may still show the earlier `0.28125` score
  until the operator status smoke is rerun.
- `/api/hive/operator/status` now uses a bounded-age `hive_status.json`
  snapshot before falling back to a live full hardware/runtime probe. A local
  development daemon on `127.0.0.1:8891` served operator status in `426 ms`
  using a `fresh_report_snapshot` and reported the same current assistant
  benchmark state. The mobile/PWA Assistant panel now shows the canonical
  benchmark row and status freshness source so phone refreshes do not look
  stuck behind stale benchmark artifacts.

## 2026-06-21 Public Transfer And Runtime Debt Closure v1

This pass improved runtime hygiene and private decoder readiness, but it did
not claim public-transfer progress and did not run a public calibration.

- `scripts/neural_seed_token_decoder_comparator.py` was split behind
  `scripts/neural_seed_token_decoder_support.py`. The comparator dropped from
  about `4614` lines to `2922` lines, moving it below the hard Python hotspot
  threshold. `reports/system_efficiency_audit.json` improved from `6` hard
  hotspots to `5`, but remains `RED`; the next hard cleanup item is
  `crates/symliquid-cli/src/code_token_generator.rs`.
- The token decoder now supports the private `STDIN_NUMERIC_PARSE` semantic
  route without fallback returns. `reports/neural_seed_token_decoder_comparator.json`
  is `GREEN`: `1024` train rows, `24` eval rows, `1230` candidate rows,
  matched parameters, filtered private verifier pass rate `1.0`, public
  training rows `0`, external inference calls `0`, and fallback returns `0`.
- `reports/private_candidate_replay_contract_audit_v1.json`,
  `reports/private_full_body_semantic_quality_ablation_v1.json`, and
  `reports/private_residual_target_consumer_v1.json` are `GREEN`; private
  selected/pass-if-any rates are `1.0` on the covered residual targets, with
  public leakage `0`.
- `scripts/resource_aware_execution_policy.py` is now platform-aware. On this
  Mac it reports `darwin/arm64`, detects `mlx_apple`, uses workspace disk
  capacity, and no longer recommends Windows `D:/` or CUDA readout flags.
- The generated source-path artifact violation was cleaned. Current
  `reports/theseus_project_registry.json` is `YELLOW`, with active source
  coverage `1.0`, unregistered active sources `0`, generated source artifacts
  `0`, and hard registry governance violations `0`. Remaining registry
  pressure is report-family volume.
- `reports/broad_capability_survival_promotion_gate_v1.json` remains `GREEN`
  for a private training artifact: transformer control, augmented pass rate
  `1.0`, baseline `0.796875`, VCM enabled for the promoted transformer
  structural path, MLX available, and MLX parity not claimed.
- `reports/maturity_integrity_audit.json` remains `YELLOW` because broad
  public transfer is still below floor (`0.021875`). `reports/candidate_promotion_gate.json`
  remains blocked at `23/28`. Private training-artifact promotion does not mean
  public/model-growth promotion.
- Closure evidence is in
  `reports/public_transfer_runtime_debt_closure_v1.json`. The next best work is
  to split the Rust Code LM/runtime hotspot and train-once fanout/ranker path,
  then rerun private readiness before any separately governed public
  calibration.

## 2026-06-21 VIEA Execution Spine And Deterministic Tool Substrate

Theseus now has a registry-owned deterministic solver/search/tool substrate
wired into the plan compiler, a bounded private execute-mode spine, evidence
store, and VIEA artifact kernel. This is scaffolding/control-plane progress,
not a student capability promotion.

- `scripts/theseus_deterministic_tool_substrate.py` and
  `configs/deterministic_tool_substrate.json` define tool cards for
  `math.sympy_exact`, `math.numeric_interval`, `math.linear_algebra`,
  `math.numeric_verify`, `math.mpmath_verify`, `logic.lean_check`,
  `logic.z3_smt`, `rewrite.egraph_minimal`, `rewrite.equality_saturation`,
  `search.local_bm25`, `search.local_hybrid`, `search.vcm_hybrid`, and
  `tool.trace_replay`.
- `reports/deterministic_tool_substrate.json` is `GREEN`: `13` registered tool
  cards, `15/15` private/replay results verified, exact solve rate `1.0`,
  tool-on solve rate `1.0`, tool-off solve rate `0.0`, public training rows
  written `0`, external inference calls `0`, and fallback returns `0`.
- The active Python environment now has `z3-solver` installed, so the staged
  SMT tool verifies its private linear-satisfiability smoke instead of emitting
  `TOOL_UNAVAILABLE`.
- The Lean check avoids the hanging elan default shim by resolving an installed
  local Lean toolchain directly. No global Lean default was changed.
- `scripts/theseus_plan_compiler.py` now routes `local_deterministic_tool` to
  the deterministic substrate instead of pointing back at itself, and emits
  execution packets with tool-card checksums, VCM context hashes, claim ids,
  evidence targets, strict no-fallback policy, and structured non-solved states.
- `reports/theseus_plan_compiler.json` is `GREEN`: `7` compiled goals, `19`
  nodes, `7` local deterministic tool packets, `13` deterministic tool
  requirements, average VCM pages per node `6.0`, hard gate failures `0`,
  execute-mode trigger `GREEN`, public training rows `0`, external inference
  calls `0`, and fallback returns `0`.
- `scripts/viea_execution_spine.py` executes the compiled deterministic packet
  under durable local leases and checkpoints. `reports/viea_execution_spine.json`
  is `GREEN`: old-direct baseline `14` cases, compiled-spine execution `14`
  cases, compiled useful completion rate `1.0`, `14` leases, `14`
  checkpoints, duplicate work `0`, retries `0`, residual count `0`, training
  evidence rows `14`, verified procedural tools `2`, public training rows `0`,
  external inference calls `0`, and fallback returns `0`.
- The execute-mode outputs include VCM artifacts, private tool-use learning
  traces, governed-admission training evidence, loop-closure candidates,
  verified procedural tool records, and a research implementation matrix.
- `scripts/report_evidence_store.py` now ingests deterministic-tool,
  plan-compiler, VIEA execution-spine, procedural-tool, and research-matrix
  families. `reports/report_evidence_store.json` is `GREEN` with current
  unstored reports `0`.
- `scripts/viea_artifact_kernel.py` now ingests compiled DAG commands/nodes,
  deterministic tool cards, deterministic tool result artifacts, execution
  spine run objects, VCM artifacts, feedback traces, training evidence,
  procedural tools, residuals, claims, and artifact graph edges.
  `reports/viea_artifact_kernel.json` is `GREEN` with `347` objects and `265`
  relationships.
- `reports/deterministic_tool_dogfood_events.jsonl` contains metadata-only
  local events; raw private text is not stored and no dogfood training rows were
  silently written.
- `reports/viea_tool_use_learning_traces.jsonl` contains metadata-only private
  execute-mode tool traces. `reports/viea_tool_use_training_evidence.jsonl`
  contains metadata-only governed-admission evidence; durable training-row
  writing remains a separate bridge and was not silently invoked.
- `reports/viea_verified_procedural_tools.json` contains `12` procedural tool
  records, of which `2` are verified procedural tools from repeated private
  replay evidence.
- `docs/VIEA_EXECUTION_SPINE_AND_TOOL_SUBSTRATE.md` is the concise checklist and
  no-cheat boundary for this surface.

## 2026-06-21 Project Registry And Cleanup Control Plane

Theseus now has a first-class project manifest registry for repo lifecycle
hygiene. The registry is not a capability claim and does not unlock public
calibration. Its job is to make project entropy visible and keep new surfaces
from becoming mystery files.

- `configs/project_manifest_registry.json` is the durable manifest policy. It
  assigns surfaces to an owner, lifecycle status, canonical entrypoint, report
  outputs, verification command, and cleanup policy.
- `scripts/theseus_project_registry.py` materializes the policy into
  `reports/theseus_project_registry.json` and
  `reports/theseus_project_registry.md`.
- Current registry state is `YELLOW`: active source coverage is `1.0` with
  `0` unregistered active-source files, all `10/10` source duplicate families
  classified, `0` unclassified source duplicate families, and `0` registry
  governance violations. Remaining registry pressure is report-family volume,
  not silent competing source ownership.
- `scripts/theseus_workspace_hygiene_audit.py`,
  `scripts/theseus_deprecation_registry.py`, `scripts/attd_analyzer.py`,
  `scripts/report_evidence_store.py`, and `scripts/theseus_control_plane.py`
  now consume the registry state.
- Cleanup remains non-destructive by default. Heavy artifacts must move through
  manifest-backed retention pointers; generated runtime state should not be
  committed as source; public benchmark payloads remain calibration-only.
- The registry now carries a `registry_evolution_contract`: Theseus should
  improve an existing registered surface first, extend an existing surface
  second, and create a new surface only when the registry declares ownership,
  canonical entrypoint, report outputs, verification command, cleanup policy,
  and successor/deprecation relationship. Violations are emitted as governance
  findings for ATTD, workspace hygiene, and the control plane.
- `reports/theseus_control_plane.json` no longer reports missing cleanup
  infrastructure, evidence-store gaps, or an unproven private receiver bridge.
  The ASI wall governor now reads the current replay/full-body repair readiness
  packet as the private transfer contract, and
  `private_to_public_transfer_gap` is `cleared`. The control plane is still
  `RED` because the real walls remain: public transfer floor, learner substrate
  proof, iteration-speed/assembly debt, and promotion locks.

See `docs/PROJECT_REGISTRY.md` for the concise lifecycle rules.

## 2026-06-21 Registry-Guided Public Transfer Consolidation

The active code-transfer survival path is now registry-backed instead of spread
across silent duplicate source families.

- `configs/project_manifest_registry.json` classifies the duplicate source
  families that were previously flagged by the registry: `configs/neural`,
  `scripts/neural`, `scripts/broad`, `scripts/code_lm`,
  `scripts/edge_contract`, `scripts/vcm`, `scripts/hive`, `scripts/native`,
  `scripts/post`, and `scripts/pressure`. All `10/10` source duplicate
  families are now classified; unclassified source duplicate families are `0`.
- The `neural_seed_and_decoder` registry surface now uses
  `scripts/broad_capability_survival_lane_decision_v1.py` as its canonical
  entrypoint. Its verification chain is:
  `scripts/private_candidate_replay_contract_audit_v1.py`,
  `scripts/broad_capability_survival_promotion_gate_v1.py`, then
  `scripts/broad_capability_survival_lane_decision_v1.py`.
- The practical lane contract in
  `reports/broad_capability_survival_lane_decision_v1.json` names
  `transformer_hybrid_structural_full_body_student` as the canonical practical
  code-transfer lane. SymLiquid remains a protected matched-compute discovery
  comparator, not a blocker for the practical assistant path.
- The old broad body-template selector is explicitly diagnostic-only for
  promotion. It may explain residuals and provide baseline pressure, but it
  cannot silently feed promotion claims.
- VCM is enabled for the promoted transformer structural path because
  same-surface structural ablation showed lift. The legacy broad body-template
  selector keeps VCM disabled because its own ablation showed harm.
- STS remains disabled for the legacy body-template selector by policy. STS may
  be used only where the path-specific ablation supports it.
- `reports/private_candidate_replay_contract_audit_v1.json` is `GREEN` on the
  private residual slice: `240` tasks replayed, selected compile/runtime load
  rates `1.0`, selected intended-behavior pass rate `1.0`, pass-if-any `1.0`,
  fallback return candidates `0`, unexplained no-candidate `0`, and public
  boundary violations `0`.
- `reports/full_body_contract_transfer_recovery_v1.json` is `GREEN` as the
  private residual repair/evaluation proof: full visible contract context
  selected-pass rate is `1.0` versus `0.526042` for the equal-budget minimal
  contract control, a lift of `+0.473958` on `192` private eval rows. No-cheat
  counters are clean: fallback returns `0`, template-like candidates `0`,
  public training rows `0`, and external inference calls `0`.
- `reports/private_full_body_repair_runtime_readiness_v1.json` is `GREEN` with
  hard failures `0`, public calibration run `false`, public training rows
  written `0`, external inference calls `0`, and fallback returns `0`.
- `reports/broad_capability_survival_promotion_gate_v1.json` is `GREEN` for a
  private training-artifact promotion only: transformer control,
  structural-only pass rate `1.0`, augmented pass rate `1.0`, baseline
  `0.796875`, candidate rows `759`, argument contract mismatches `0`, fallback
  returns `0`, public training rows `0`, external inference calls `0`, serving
  disabled, public calibration disabled, and MLX parity not claimed.
- `reports/candidate_promotion_gate.json` remains blocked (`23/28`) because the
  legacy/public-transfer maturity gates are not cleared. This is expected:
  private structural promotion is allowed as a training artifact, but public
  calibration/model growth remain locked until broad public transfer improves
  under the governed runner.

Current conclusion: the practical path is no longer ambiguous, and the private
receiver/replay bridge is no longer the blocker. Continue improving semantic
candidate quality and iteration speed through the canonical survival lane. Do
not add another `vN` lane for this problem; extend or repair the
registry-owned surface. A future one-shot public calibration is technically
justified only as a separately unlocked calibration decision; this consolidation
goal did not run or unlock public calibration.

## 2026-06-21 Public Transfer Lift v2 Result

The fresh `public_transfer_lift_v2_seed41_5x64` public calibration has now
been spent exactly once through the generalized registry-backed guarded runner.
It was a disjoint 320-task slice: both consumed selector manifests were excluded
before selection, leaving `0` overlap with prior public calibration task IDs.
This surface must not be rerun or used as training data.

- `reports/full_body_contract_transfer_recovery_v2_private320.json` is GREEN.
  The repaired transformer/hybrid full-body path scored `320/320 = 1.0` on the
  stronger private readiness floor with `8` candidates per task. The
  minimal-context equal-budget control scored `185/320 = 0.578125`. No-cheat
  counters stayed clean: fallback returns `0`, template-like candidates `0`,
  public training rows `0`, external inference calls `0`, no-candidate rate
  `0.0`, and no-admissible task rate `0.0`.
- `reports/public_wide_slice_selector_public_transfer_lift_v2_seed41_5x64.json`
  is GREEN. It selected `64` tasks each from MBPP, EvalPlus, BigCodeBench,
  HumanEval, and LiveCodeBench after excluding `346` consumed task keys.
- `reports/public_transfer_lift_v2_readiness_packet.json` is GREEN and freezes
  the command, manifest hashes, private readiness evidence, harness hashes, and
  registry state for the one-shot surface.
- `reports/operator_bounded_public_calibration_public_transfer_lift_v2_seed41_5x64.json`
  is GREEN. The run executed exactly once, restored the operator lock, and
  appended `public_transfer_lift_v2_seed41_5x64` to
  `reports/public_benchmark_run_registry.jsonl` as consumed.
- The score regressed: `reports/real_code_benchmark_graduation_public_transfer_lift_v2_seed41_5x64.json`
  scored `1/320 = 0.003125`, worse than both locked references
  `34/160 = 0.2125` and `7/320 = 0.021875`. Per-card: MBPP `0/64`,
  EvalPlus `0/64`, BigCodeBench `1/64`, HumanEval `0/64`, LiveCodeBench
  `0/64`.
- The no-cheat boundary remained intact: token-level learned candidates were
  present, full-body token candidates were present, grammar-masked candidates
  were present, hardcoded solver candidates `0`, template/loop candidates `0`,
  expression-memory fallback candidates `0`, and external inference calls `0`.
- `reports/public_code_transfer_residual_report_public_transfer_lift_v2_seed41_5x64.json`
  records the safe residual categories. Raw dominant categories are
  local-candidate/admissibility gaps `297`, algorithmic planning `10`,
  return/type shape `7`, and verifier mismatch `5`. Adapter-adjusted
  no-admissible falls to `0`, which means the candidate manifest contained rows
  for those tasks, but the scorer/ranker/adapter path still did not select
  countable semantic candidates.
- `reports/bounded_public_transfer_residual_mining_public_transfer_lift_v2_seed41_5x64.json`
  is YELLOW, as expected after a failed public result, and writes `9`
  private-only target rows to
  `reports/bounded_public_transfer_private_residual_targets_public_transfer_lift_v2_seed41_5x64.jsonl`.
  These rows contain aggregate categories and hashes only; they are not training
  rows and embed no public prompts, tests, solutions, candidate code, traces, or
  score labels.
- `reports/operator_bounded_public_calibration_public_transfer_lift_v2_seed41_5x64_no_rerun_refusal.json`
  records the guard refusal for a second execution attempt. It blocked on the
  consumed registry entry and existing output/trace artifacts.

Current conclusion: private full-body recovery is real but not yet transferring
to broad public code tasks. The main wall is no longer private admissibility; it
is public-facing candidate selection/admission plus semantic planning under the
real benchmark task distribution. Do not spend another public surface until a
private-only repair proves that the exact failure pattern is fixed without
public payload training.

## 2026-06-21 Full-Body Contract Transfer Recovery

`reports/full_body_contract_transfer_recovery_v1.json` is the current recovery
packet for the code public-transfer regression. It does not run public
calibration and does not read public traces, prompts, tests, solutions, or
answer templates. Historical public reports are used only as aggregate context.

- The repaired full-body generator now emits
  `rust_code_lm_private_contract_role_body_synthesis_v1` candidates from
  visible decoder-contract argument roles, type family, and return contract.
  This covers contract-blind role pairs such as error transcript buckets,
  ready task records, storage quota selection, local/remote manifest sync,
  top-k frequency, and range-bounded numeric deltas. It is not keyed by task ID,
  public payloads, tests, or solution bodies.
- The private equal-budget comparator ran the `192`-row readiness floor with
  `8` candidates per task for the repaired full-body path, the minimal-context
  control, and the old STS closure path. The old path now uses same-slice STS
  streams generated for all `192/192` private rows without public payloads or
  solution targets, then reuses the archived train-once closure checkpoint.
- Full contract context scored `192/192 = 1.0`; minimal context scored
  `101/192 = 0.526042`. Delta is `+0.473958`, so decoder-contract role
  information is causally useful in the stricter full-body path.
- The old STS closure/interface/receiver-bridge path scored
  `33/192 = 0.171875` on the same private slice with `0.0` no-candidate rate.
  This confirms the old path had useful but weaker transfer pressure; the
  repaired promotion-grade full-body path now dominates it on this private
  analogue without using public data.
- `reports/full_body_contract_transfer_recovery_v1_symliquid_same_slice_comparator.json`
  is the fresh same-slice protected discovery comparator. It is still the
  body-template selector boundary, not promotion-grade full-body token
  generation. Under matched budget (`192` eval rows, `8` fanout, parameter delta
  `0.01108`), transformer control scored `0.677083` and SymLiquid-style scored
  `0.661458`; transformer is ahead by `0.015625` on this slice. STS helped both
  arms (`+0.046875` transformer, `+0.052083` SymLiquid).
- VCM is not blanket-enabled in every code path. Existing private ablations say
  to enable VCM for the promoted structural path
  (`reports/broad_capability_structural_vcm_ablation_v1.json`: transformer
  structural delta `+0.020834`, SymLiquid structural delta `+0.020833`) and keep
  it disabled for the old broad body-template selector until it earns same-surface
  lift (`reports/broad_capability_vcm_feature_ablation_v1.json`: transformer
  delta `-0.0625`, SymLiquid delta `-0.104167`).
- Candidate coverage is clean on this private analogue: the old closure,
  repaired full-context, and minimal-context arms all have `0.0` no-candidate
  rate. The repaired full-context arm also has `0.0` no-admissible task rate,
  `0` per-family regressions against the minimal control, `42` per-family
  improvements, and `0` private repair targets.
- No-cheat audit is clean over `3,264` candidate rows and `576`
  task-manifest rows: fallback returns `0`, template-like candidates `0`,
  public training rows `0`, forbidden export keys `0`, and external inference
  calls `0`. The same-slice SymLiquid comparator separately reports public
  training rows `0`, teacher used `false`, external inference calls `0`, and
  candidate rows without tests or solutions embedded.
- The readiness decision is
  `ready_for_future_governed_public_calibration`. This is a readiness packet,
  not a promotion and not a public benchmark run.
- The same report now includes
  `side_by_side_recovery_matrix`: old STS/interface/receiver-bridge public
  evidence is recorded only as consumed aggregate context (`34/160 = 0.2125`,
  `36` no-candidate events), the current full-body public regression is
  recorded only as consumed aggregate context (`7/320 = 0.021875`, `470`
  no-candidate events), the old STS closure private lane is fresh same-slice
  evidence (`33/192 = 0.171875`), the current full-body token-beam control is
  the minimal-contract equal-budget arm (`101/192 = 0.526042`), the repaired
  full-body private arm is fresh equal-budget evidence (`192/192 = 1.0`), and
  the SymLiquid/transformer discovery lane is fresh same-slice matched-compute
  evidence with the transformer slightly ahead.
- `reports/full_body_contract_transfer_recovery_v1_private_repair_targets.jsonl`
  is currently empty. It is allowed to contain task IDs, categories, residual
  labels, and repair focus only; no tests, solutions, candidate code, or public
  payloads.

That next action has now been completed by the
`public_transfer_lift_v2_seed41_5x64` one-shot calibration above. The result did
not lift public transfer, so this private recovery packet should be treated as
necessary but not sufficient evidence.

## 2026-06-20 Public-Transfer Wall Breaker Status

The frozen `industry_code_transfer_seed14_5x64_v1` public calibration has been
spent exactly once through the guarded runner. It must not be rerun or used as
training data.

- `reports/operator_bounded_public_calibration_industry_code_transfer_seed14_5x64_v1.json`
  is `GREEN`: the registry-backed approval was valid, the run executed once,
  and a post-run dry-run refusal records that the surface is now consumed.
- `reports/real_code_benchmark_graduation_industry_code_transfer_seed14_5x64_v1.json`
  scored `7/320 = 0.021875`, below the previous locked `34/160 = 0.2125`
  baseline. Per-card: MBPP `1/64`, EvalPlus `0/64`, BigCodeBench `6/64`,
  HumanEval `0/64`, LiveCodeBench `0/64`.
- `reports/public_code_transfer_residual_report_industry_code_transfer_seed14_5x64_v1.json`
  mines only aggregate residual categories. No public prompts, tests,
  solutions, traces, answer templates, or public-derived training rows are
  emitted. Dominant raw categories were local candidate generation/admissibility
  gaps, algorithm choice, verifier mismatch, return/type handling, and
  dependency/runtime handling.
- `reports/private_residual_target_consumer_v1.json` now treats
  `reports/candidate_floor_v2_private_token_probe.json` as required evidence
  before declaring residual targets closed. This prevents old saturated private
  gates from hiding generator weakness. After the residual-label ranker repair,
  current state is `GREEN`: candidate admissibility is ready and the private
  candidate-floor analogue improved from `33/192 = 0.171875` to
  `187/192 = 0.973958` against a `0.70` transfer repair floor. All `12`
  private residual targets are covered. The remaining private misses are `5`
  single-row contract-blind residuals, which should be mined privately without
  public payloads.
- `reports/one_shot_public_transfer_calibration_failure_mining_v1.json` is the
  current compact packet. It records the spent public result, no-cheat audit,
  dogfood metadata status, Mac MLX/Metal/VCM runtime evidence, and the concrete
  next private work. Promotion remains blocked; the practical hot path remains
  the transformer/hybrid structural full-body student, while SymLiquid remains
  a protected matched-compute discovery comparator.

Current no-cheat boundary: fallback returns `0`, template-like candidates `0`,
public training rows `0`, public content embedded in private targets `false`,
external inference serving `0`. The next useful work is private semantic
candidate-quality repair until the candidate-floor analogue clears the transfer
repair floor, not another public calibration spend.

## 2026-06-18 Virtual Context Memory Control Plane

The context/memory lane now has a governed Virtual Context Memory substrate over
the existing context packet ingest adapter.

- `scripts/virtual_context_memory.py` ingests `reports/context_packet_ledger.json`,
  `reports/context_packets.jsonl`, redacted dogfood usage events, `AGENTS.md`,
  this project-state file, the context-memory doc, and
  `Virtual_Context_Memory_Corben_Sorenson.md`.
- It writes redaction-safe durable events, typed semantic pages with stable
  `vcm://` addresses, L0-L5 representations, compression certificates, source
  hashes, taint/governance labels, importance/risk vectors, graph relations,
  transactions, snapshots, compiled context, deletion/tombstone closure, and
  explicit semantic page faults.
- Current reports are `reports/virtual_context_memory_events.jsonl`,
  `reports/virtual_context_memory_ledger.json`,
  `reports/virtual_context_memory_pages.jsonl`,
  `reports/virtual_context_memory_graph.json`,
  `reports/virtual_context_memory_transactions.jsonl`,
  `reports/virtual_context_memory_snapshots.json`,
  `reports/virtual_context_compiled_context.json`,
  `reports/virtual_context_memory_bench.json`,
  `reports/virtual_context_memory_index.json`,
  `reports/virtual_context_memory_training_admission.json`,
  `reports/virtual_context_memory_consumer_audit.json`,
  `reports/virtual_context_memory_status.json`,
  `reports/vcm_context_recovery_benchmark.json`,
  `reports/vcm_context_recovery_benchmark.md`,
  `reports/vcm_context_recovery_residuals.jsonl`,
  `reports/vcm_on_off_ablation.json`,
  `reports/vcm_public_memory_calibration.json`,
  `reports/vcm_public_memory_calibration_ledger.jsonl`,
  `reports/vcm_public_memory_readiness_audit.json`,
  `reports/vcm_public_memory_readiness_audit.md`,
  `reports/vcm_public_memory_prompt_calibration.json`,
  `reports/vcm_public_memory_prompt_calibration_ledger.jsonl`,
  `reports/vcm_public_memory_private_residual_repair.json`,
  `reports/vcm_evidence_gauntlet.json`,
  `reports/vcm_proof_card.md`,
  `reports/vcm_prefetch_regret_audit.json`,
  `reports/vcm_runtime_claim_readiness.json`,
  `reports/vcm_runtime_materialization_claims.jsonl`,
  `reports/vcm_release_conformance_audit.json`,
  `reports/virtual_context_memory_usage_events.jsonl`, and
  `reports/virtual_context_memory_probe.json`.
- The VCM CLI now has local `status`, `query`, `explain`, and `record-usage`
  modes. Query/explain resolves aliases, source paths, lanes, taints, graph
  closure, snapshot visibility, compiler promotion/fault decisions, and
  training-admission state.
- `reports/virtual_context_memory_bench.json` is now VCM-Bench v2. It covers
  continuity, exactness, staleness, contradiction/supersession, rejected
  branches, prompt-injection poisoning, deletion closure, context switching,
  capacity thrash, event sourcing, alias/version drift, multi-hop graph
  closure, snapshot time travel, rollback isolation, descendant deletion
  leakage, prompt-injection through derived summaries, and deterministic
  compile reproducibility.
- `reports/virtual_context_memory_training_admission.json` is the memory-derived
  training bridge. Future private training rows derived from memory must carry
  source hashes, event provenance, taint state, public-calibration quarantine,
  teacher-boundary proof, and deletion-closure proof before admission.
- `scripts/vcm_context_recovery_benchmark.py` is the current private VCM
  context-recovery proof. It uses private fixtures only, inspired by public
  long-context task categories but not copied from public prompts, contexts,
  answers, traces, or templates. It also consumes private-only residual
  fixtures derived from aggregate public-memory failure categories. Current
  result: `17` private categories including `4` public-memory residual
  analogues, VCM answer accuracy `1.0`, VCM evidence recall `1.0`, best flat
  baseline `0.5882`, no-admissible rate `0.0588`, fallback-return count `0`,
  public training rows `0`, external inference calls `0`, and explicit graph
  traversal/storage cost accounting.
- `reports/vcm_on_off_ablation.json` is the STS-style same-surface VCM
  ablation. VCM-on is `vcm_graph`; VCM-off is the strongest plain non-VCM
  control, currently `latest_flat_report`. It uses the same private cases,
  token budget, contamination firewall, and no-fallback rule, then reports
  answer/evidence/stale-deletion lift, VCM-only wins, off-only regressions, and
  per-category deltas. Current result is `GREEN`: VCM-on answer accuracy `1.0`,
  VCM-off answer accuracy `0.5882`, answer lift `+0.4118`, evidence-recall lift
  is positive, VCM-only wins `7`, off-only wins `0`, fallback returns `0`.
- The governed public context/memory benchmark registry now includes metadata
  cards for RULER, BABILong, NeedleBench/OpenCompass, LongMemEval,
  LongMemEval-V2, HELMET, LongBench v2, InfiniteBench, NoLiMa,
  Michelangelo/LSQ, LV-Eval, LOFT, MTRAG, MTRAG-UN, FACTS Grounding, LoCoMo,
  and LoCoMo-Plus. These are calibration/readiness sources only. NoLiMa and
  LoCoMo are blocked by non-commercial terms, FACTS Grounding is blocked by
  model-judge/private-leaderboard scoring, and LongMemEval-V2 is still blocked
  in this environment because the currently loadable Hugging Face surface
  exposes image rows rather than text trajectories. RULER, BABILong,
  LongBench v2, NeedleBench/OpenCompass, and InfiniteBench now have governed
  deterministic prompt adapters.
- `reports/vcm_public_memory_calibration.json` is the first governed
  public-memory VCM-on/off calibration slice. It is metadata-clean only: it
  reads benchmark cards, not public prompts, contexts, answers, tests, traces,
  solutions, or answer templates. The run-once ledger is
  `reports/vcm_public_memory_calibration_ledger.jsonl`. Current state is
  `YELLOW` because prompt-level public item scoring is still pending official
  adapters and a separate exact-run unlock, not because of contamination:
  benchmark count `8`, VCM-on mean metadata facet recall `1.0`, VCM-off
  `0.272727`, VCM-only wins `8`, public training rows `0`, external inference
  calls `0`, and fallback returns `0`.
- `scripts/vcm_official_public_memory_adapter.py` is the governed prompt-level
  public-memory adapter. It stages public prompt/context/answer payloads only
  under ignored quarantine, hashes the surface including exact item offsets,
  writes a run-once ledger, and never admits public rows to training. The
  historical `vcm_public_memory_prompt_slice_2026_06_18` run remains the
  postmortem source for the prior VCM-off advantage: VCM-on `0.666667`,
  VCM-off `0.833333`, off-only wins `2`, VCM-only wins `0`, with residual
  categories `no_admissible`, `state_tracking_failure`, and
  `temporal_update_failure`. The root cause was not treated as a
  public-training signal: the QA1 misses were a local person-question parser
  bug shared by both VCM-on and VCM-off, while the QA3 off-only wins came from
  VCM-on joining selected evidence out of original source order. The adapter
  now preserves source chronology including repeated identical events, handles
  BABILong give/drop transfer events, uses a direct `Where is/was NAME` parser
  for BABILong-style person questions, generates larger deterministic RULER
  slices, supports per-benchmark offsets/max counts for fresh exact-run slices,
  stages official LongMemEval JSON when present, streams official LongBench v2
  and NeedleBench/OpenCompass sources when enabled, builds deterministic
  InfiniteBench retrieve-key/value rows, writes a public-safe item manifest
  with source length/depth metrics, blocks overlap with forbidden prior slices,
  and reports competing memory systems beside VCM: flat tail/head/middle
  windows, lexical retrieval, BM25-style sparse retrieval, recency-weighted
  retrieval, deterministic hybrid retrieval, and structured state table.
- `scripts/vcm_public_memory_readiness_audit.py` is the no-cheat bridge between
  private VCM repair and prompt-level public-memory spend. It audits RULER,
  BABILong, LongMemEval, and catalogued memory/context cards; uses private
  analogues for parser, chronology, distractor, answer-shape repair, and
  LongMemEval-shaped temporal/update/preference/abstention repair; checks
  private context-recovery coverage; and verifies no public prompts, contexts,
  answers, traces, tests, templates, teacher calls, external inference, fallback
  returns, or public training rows enter the repair path. Current state is
  `GREEN`: RULER, BABILong, and LongMemEval are prompt-ready with deterministic
  local scoring, private analogue VCM-on pass rate is `1.0`, off-only wins are
  `0`, private VCM lift is `+0.4118`, and the latest prompt slice has a
  manifest-backed long-context ladder with zero forbidden overlap.
- The 1000-row governed long-context ladder
  `vcm_public_memory_prompt_slice_2026_06_19_long_context_ladder_1000_fresh`
  ran once under the ledger in
  `reports/vcm_public_memory_prompt_calibration_ledger.jsonl`. It is a fresh
  non-overlapping public calibration against the prior
  `vcm_public_memory_prompt_slice_2026_06_19_large_memory_goal_refresh_1300_offset200`
  slice, with item manifest hash
  `sha256:8903f5ae05283c5c05abaf0c85cc9c51aee65d76b6606380fb7285887cbfab8b`.
  RULER, BABILong, and LongMemEval scored `1000` prompt-level rows (`400`,
  `400`, and `200` respectively), VCM-on pass rate `0.807`, flat-tail VCM-off
  pass rate `0.397`, VCM-over-flat-tail delta `+0.41`,
  VCM-over-best-non-VCM delta `+0.097`, VCM-only wins `413`, off-only wins `3`,
  public training rows `0`, external inference calls `0`, teacher solving calls
  `0`, and fallback returns `0`. Per-benchmark, VCM was `1.0` on RULER, `0.99`
  on BABILong, and `0.055` on LongMemEval.
- The current governed prompt-level repair confirmation is
  `vcm_public_memory_prompt_slice_2026_06_19_lme_extraction_repair_confirm_600`.
  It scored `600` rows (`200` each for RULER, BABILong, and LongMemEval), with
  item manifest hash
  `sha256:3b9ef9f340ecb6a2841d09185d3c216392097bc82d77bf5063634af6eeb60e78`
  and zero overlap against the prior 1000-row ladder manifest. Current state is
  `GREEN`: VCM-on pass rate `0.696667`, flat-tail VCM-off pass rate `0.331667`,
  VCM-over-flat-tail delta `+0.365`, VCM-over-best-non-VCM delta `+0.086667`,
  VCM-only wins `223`, off-only wins `4`, public training rows `0`, external
  inference calls `0`, teacher solving calls `0`, and fallback returns `0`.
  LongMemEval improved from `0.055` to `0.10`, beat flat-tail `0.02`, and tied
  the best single non-VCM baseline at `0.10`. The remaining honest wall is
  public LongMemEval semantic exact-answer quality: the pass-if-any non-VCM
  aggregate still exposes `4` off-only rows and `180/200` LongMemEval failures.
- The latest governed prompt-level hard-public calibration is
  `vcm_public_memory_prompt_slice_2026_06_19_five_family_hard_public_memory_2196`.
  It scored a fresh exact-run slice of `2196` rows across five admitted public
  families: RULER `1600`, BABILong `400`, InfiniteBench `100`,
  NeedleBench/OpenCompass `80`, and LongBench v2 `16`. It has item manifest
  hash
  `sha256:ba0d830ad37d3b36c68d666b854e62ad6ff79cc379e16ff1a5f856b956bd43a6`
  and zero forbidden overlap against every recorded prior prompt slice. Current
  state is `GREEN`: VCM-on pass rate `0.952641`, flat-tail VCM-off pass rate
  `0.179417`, best non-VCM pass rate `0.908925`, VCM-over-flat-tail delta
  `+0.773224`, VCM-over-best-non-VCM delta `+0.043716`, VCM-only wins `1700`,
  off-only wins `2`, public training rows `0`, external inference calls `0`,
  teacher solving calls `0`, and fallback returns `0`. The slice covers
  `lt_8k`, `8k_to_32k`, `32k_to_128k`, and `128k_plus` source-context buckets,
  with max source context `607012` token-equivalent. Per-family status is
  deliberately mixed: VCM is strong on RULER, BABILong, and InfiniteBench;
  LongBench v2 remains weak (`0.0625` VCM-on versus `0.1875` best non-VCM on a
  16-row admitted slice); and NeedleBench/OpenCompass currently scores `0.0`
  across VCM and non-VCM deterministic systems, making deep-needle answer
  formatting/evidence extraction the next private repair target.
- `reports/vcm_longmemeval_private_residual_curriculum.json` is the
  private-only repair gate for that LongMemEval wall. It builds 180 local
  LongMemEval-style cases from private templates only, using aggregate public
  failure categories but no public prompts, contexts, answers, traces, tests,
  solutions, or answer templates. Current state is `GREEN`: VCM-on pass rate
  `0.983333`, best single non-VCM `0.811111`, VCM-over-best-non-VCM delta
  `+0.172222`, minimum major question-type pass rate `0.833333`, evidence
  recall `0.883333`, abstention precision/recall `1.0`/`1.0`, public payload
  chars loaded `0`, public training rows `0`, teacher solving calls `0`,
  external inference calls `0`, and fallback returns `0`. The future public
  proposal state is `READY_TO_PROPOSE_EXACT_ONCE_PUBLIC_CONFIRMATION`, with
  `run_public_automatically=false`; the public LongMemEval score remains the
  locked `0.10` until a governed exact-once confirmation is explicitly run.
- `scripts/vcm_evidence_gauntlet.py` is the broad private/local VCM proof
  surface. It scores the real VCM adapter path against flat head/tail/middle,
  lexical retrieval, BM25-style sparse retrieval, recency-weighted retrieval,
  deterministic hybrid retrieval, and structured-state-table baselines under
  the same context budget. It covers LongMemEval-style semantic memory,
  RULER-style needle retrieval, BABILong-style state tracking, and file/task
  memory without public payloads or training rows. Current
  `reports/vcm_evidence_gauntlet.json` state is `GREEN`: `1200` cases, VCM-on
  pass rate `0.990833`, best single non-VCM `0.809167`,
  VCM-over-best-non-VCM delta `+0.181666`, minimum major-family pass rate
  `0.979167`, answerable evidence recall `0.99009`, abstention precision/recall
  `1.0`/`1.0`, fallback returns `0`, teacher/external inference calls `0`, and
  public payload/training counters `0`. The proof card reports VCM wins on
  file/task memory and LongMemEval-style semantic memory, ties on BABILong and
  RULER where deterministic non-VCM baselines are already perfect, and no losing
  family. The gauntlet prepares a public confirmation manifest proposal but
  never runs public calibration automatically.
- `scripts/vcm_hard_memory_private_analogues.py` is the harder private VCM
  analogue gauntlet for public memory families that need larger/harder pressure.
  It mirrors failure categories from NoLiMa, Michelangelo/LSQ, LV-Eval, LOFT,
  LongBench v2, InfiniteBench, MTRAG/MTRAG-UN, FACTS Grounding, and LoCoMo-Plus
  without public payloads or training rows. Current
  `reports/vcm_hard_memory_private_analogues.json` state is `GREEN`: `1000`
  private rows, `10` families, `4` length buckets, VCM-on `0.979`, best single
  non-VCM `0.804`, delta `+0.175`, minimum family pass rate `0.91`, evidence
  recall `0.952778`, abstention precision/recall `1.0`/`1.0`, fallback returns
  `0`, teacher/external inference calls `0`, and public payload/training
  counters `0`. The shared resolver now supports alias/cue bridge recovery,
  answer-shape compaction for key/result/mode spans, stale/confusing answer
  demotion, long-dependency terminal answer selection, and explicit unknown
  context abstention.
- `scripts/vcm_hard_memory_benchmark_readiness.py` is the source/admission audit
  for hard public memory benchmarks. Current
  `reports/vcm_hard_memory_benchmark_readiness.json` state is `YELLOW`, not
  because of row count or five-family coverage, but because source warnings
  remain for harder/blocked public memory families. It reports `2196` current
  public prompt rows, the row target met, and prompt-ready admission for RULER,
  BABILong, LongBench v2, NeedleBench/OpenCompass, and InfiniteBench. The next
  public-memory wall is quality on the admitted hard families plus clean text
  staging for LongMemEval-V2, not more synthetic private rows.
- `reports/vcm_prefetch_regret_audit.json` closes the prior Predictive VCM
  evidence gap with private/local usage and forecast artifacts only. Current
  state is `GREEN`: staged pages `120`, promoted `91`, prefetch precision
  `0.758333`, missed-fault rate `0.384615`, regret `20.9037`, raw usage text
  stored `false`, public training rows `0`, external inference calls `0`, and
  fallback returns `0`.
- `reports/vcm_runtime_claim_readiness.json` is a runtime-readiness prototype,
  not a runtime/KV-cache parity claim. It writes complete semantic
  resident-materialization descriptor keys to
  `reports/vcm_runtime_materialization_claims.jsonl`. Current state is `GREEN`:
  `64` accepted semantic descriptors, cache-key complete rate `1.0`, rejected
  descriptors `0`, `runtime_profile_claimed=false`, and
  `native_kv_cache_claimed=false`.
- Dogfood usage events are local-only and redaction-safe by default. They store
  hashes, labels, artifact references, and purpose limits, not raw sensitive
  text, and are not trainable without the training-admission bridge.
- `configs/autonomy_policy.json` enables a guarded local-only VCM refresh after
  context packet compaction. It uses no external inference, keeps public
  calibration exact-run locked, blocks public-training use, counts
  fallback-return patterns, reruns private prefetch-regret and runtime-readiness
  audits, and records public-memory calibration only when explicitly unlocked.
- `scripts/cognitive_context_router.py` now includes VCM state, page count, and
  fault count in its memory context.
- `scripts/autonomy_cycle_runtime.py`, `scripts/autonomy_watchdog.py`,
  `scripts/autonomy_watchdog_actions.py`, `scripts/sparkstream_dashboard.py`,
  `scripts/capability_matrix.py`, and `scripts/long_horizon_memory_probe.py`
  now consume or expose VCM v1 state.
- The full consumer audit now classifies all detected memory/context consumers:
  `25` migrated VCM consumers, `1` ingest-adapter-only consumer, `9` doc-only
  references, and `68` explicitly blocked pending VCM migration; unclassified
  consumers are `0`. The high-value consumer set is `16/16` VCM-integrated.
- `scripts/vcm_task_context_bridge.py` is now the system-level VCM integration
  contract. It turns the VCM index, compiled context, probe, status, training
  admission, consumer audit, runtime-readiness, and release-conformance reports
  into task-family context views. Current `reports/vcm_task_context_bridge.json`
  state is `GREEN`: 9/9 task families ready, 7/7 high-priority families ready,
  45 unique selected VCM pages, public training rows `0`, external inference
  calls `0`, teacher solving calls `0`, fallback returns `0`, and
  `runtime_profile_claimed=false`. The covered task families are operator chat,
  autonomy governance, code/training, public calibration review, teacher
  governance, Hive/storage routing, voice/spatial operation, runtime/MLX/Metal
  routing, and documentation/project-state work.
- Hive/operator status exposes compact VCM memory health: freshness, page/event
  counts, fault counts, graph conflicts, latest snapshot, training-admission
  state, private context-recovery score versus baseline, and recommended
  repairs.
- Native runtime/KV-cache paging remains future work. VCM v1 now records
  complete semantic materialization keys, source references, cache invalidation
  records, and safe fault paths without claiming native model-runtime/KV cache
  integration.

## 2026-06-18 STS Ranker Policy and Dogfood Bootstrap

`reports/sts_ranker_policy_v1.json` is the current source of truth for the
STS-as-selector result. `scripts/sts_ranker_policy_v1.py` defines a guarded
metadata ranker that uses STS route indicators, decoder-contract features, and
candidate metadata. It does not train on heldout tests, heldout solution
bodies, public benchmark content, teacher rows, or runtime verifier outcomes.

Current result:

- The report is `GREEN` across two existing private surfaces:
  private residual v3 (`240` tasks) and
  `private_ecology_generalization_v5_smoke72` (`72` tasks). No new synthetic
  benchmark family was generated for this check.
- Equal-budget task coverage is `312/312 = 1.0`.
- STS-ranker selected pass rate is `312/312 = 1.0`.
- Matched non-STS selected pass rate is `184/312 = 0.589744`.
- Selected-pass lift is `+0.410256`; STS oracle pass rate is `1.0`; non-STS
  oracle pass rate is `0.769231`.
- Regressions versus existing STS order are `0`; fallback-return candidates are
  `0`; no-admissible rate is `0.0`; public leakage count is `0`; external
  inference calls are `0`.
- The selector is integrated only as a guarded policy:
  `configs/sts_ranker_policy_v1.json` keeps it disabled by default and runtime
  use requires `THESEUS_ENABLE_STS_RANKER_POLICY_V1=1`.

`reports/maturity_integrity_audit.json` now consumes this ranker report as a
causal STS consumer. The old maturity blocker
`sts_capsules_have_causal_consumer` is cleared. Maturity remains `YELLOW` only
on `public_transfer_floor_cleared` and
`candidate_coverage_transfer_gate_is_fresh`.

The public-transfer recommendation is conservative. The STS ranker is privately
green, but `reports/post_distillation_public_transfer_readiness_v1.json` is
`RED` because broad-private/post-distillation artifacts are stale against the
current decoder source or release binary. The next action is to refresh
broad-private transfer evidence before any public calibration spend. Public
calibration remains locked and disallowed.

Dogfood trace bootstrapping is also current:

- `reports/dogfood_trace_bootstrap.json` is `GREEN`.
- Local metadata capture and training export are enabled on this machine, raw
  text capture is disabled, and accepted/missed/ignored accounting is defined.
- Existing real metadata events: accepted `1`, missed `0`, ignored `0`.
- Existing redacted private dogfood rows: `1`.
- No raw user text, teacher rows, public benchmark rows, fallback returns, or
  external inference calls are present.

## 2026-06-17 Private Residual V3 Student Repair

`reports/private_residual_repair_v3_gate.json` is the current source of truth
for the private residual v3 repair lane. It is now `GREEN` under adapter-off
scoring, and it supersedes the 2026-06-15 diagnostic-adapter-only result.

What improved:

- `scripts/private_residual_v3_student_repair_loop.py` emits a private-only
  train-induced structural/token candidate manifest plus a matched non-STS
  structural control manifest. It uses `960` private train rows and `240`
  private heldout rows. The STS-on arm may use heldout-visible category, entry
  point, and decoder contract; the non-STS arm uses entry point plus decoder
  contract features only. Heldout tests and heldout solution bodies are not
  used for generation.
- `reports/private_residual_repair_v3_heldout_score.json` is `YELLOW` only
  because the equal-budget non-STS oracle/pass-if-any control now passes
  `240/240`, making the old route-withheld STS delta `0.0`. Adapter-off
  learned/student scoring still passes:
  `240/240 = 1.0`; structural-action candidate passes are `240/240 = 1.0`;
  diagnostic-adapter pass credit is `0/240 = 0.0`.
- `reports/private_residual_v3_sts_ablation.json` is the current STS source of
  truth. It is `GREEN` with equal candidate budget `4` on the full `240`-task
  heldout. STS-on selected-candidate pass rate is `240/240 = 1.0`; matched
  non-STS selected-candidate pass rate is `184/240 = 0.766667`; both arms have
  oracle/pass-if-any rate `240/240 = 1.0`. The STS selected-rate lift is
  `+0.233333`, oracle lift is `0.0`, and the same-body label-removed control
  matches STS at `1.0`.
- The accepted STS interpretation changed: the non-STS arm can emit a passing
  candidate somewhere in the equal-budget set, so STS is not improving emission
  coverage on this private v3 slice. STS is improving selection/ranking by
  putting the passing structural body first.
- No-cheat counters are clean: public candidate rows `0`, public tests used
  `false`, public solutions used `false`, external inference calls `0`,
  fallback-return candidates `0`, no-admissible rate `0.0`, and
  LiveCodeBench-style private stdin proxy passes `48/48`.

What is still blocked:

- This is private repair evidence only. It closes the private residual v3
  learned/student blocker, but it does not prove public transfer and does not
  justify model growth or promotion.
- `reports/maturity_integrity_audit.json` remains `YELLOW` with hard blockers
  `0` and evidence blockers `0`. The remaining maturity blockers are the spent
  public transfer floor, candidate-coverage transfer freshness, and STS
  capsules needing a causal consumer.
- `reports/candidate_promotion_gate.json` keeps `promote=false`.
- `reports/theseus_generalization_governor_v1.json` is `YELLOW`, not `RED`:
  hard failed gates `0`, warning failed gates `2`, public pass rate
  `34/160 = 0.2125`, and approved/spent post-v4 public artifacts are recognized
  with forbidden artifact count `0`.
- Public calibration remains locked and disallowed.

The accepted interpretation is narrow: the private v3 learned/student route now
reaches the residual families without diagnostic-adapter credit or fallback
returns. The STS result should be read as ranking evidence under matched
budget: both arms can emit a passing private structural body, but STS ranks it
first more reliably. The next wall is transfer: the system must show that this
kind of learned route improves beyond the locked public score without fishing
public surfaces or using public data as training pressure.

## 2026-06-15 Gate-Closure Status

The current gate-closure packet is
`reports/theseus_gate_closure_packet.json`. It is intentionally `YELLOW`, not
greenwashed. It proves the remaining blockers are operator, hardware, or
learned-evidence gates, not places where the system should silently
self-approve:

- dogfood capture consent is missing;
- dogfood training consent is missing;
- public calibration operator approval is missing;
- Metal production route approval is missing;
- Metal production routing remains disabled;
- accelerator parity claim remains disallowed;
- CUDA reference reports are missing and must be generated on a CUDA node;
- this Apple Silicon Mac has no local CUDA toolchain;
- the Windows CUDA coordinator is currently unreachable from this Mac;
- public transfer is still below floor even though private residual v3
  learned/student evidence is now green under adapter-off scoring.

New guardrail tools:

- `scripts/dogfood_trace_consent.py` validates or writes the local dogfood
  consent config only with explicit execute flags. It keeps raw-text capture
  disabled and requires separate capture and training timestamps.
- `scripts/theseus_gate_closure_packet.py` collects the dogfood, public
  calibration, Metal route, parity-claim, CUDA-environment, and network-doctor
  state into one operator packet. It now refreshes
  `reports/hive_network_doctor.json` itself before computing blockers, includes
  a requirement matrix plus supporting-audit summary, and writes generated
  approval templates with `approved=false`; templates are not approvals.
- `scripts/private_residual_repair_v3_heldout_score.py` now reports family and
  category rates, no-admissible residual rows, learned/student pass rate, and
  diagnostic-adapter pass rate separately so adapter repairs cannot be confused
  with learned capability.

The current requirement matrix in `reports/theseus_gate_closure_packet.json`
has three proven rows: local dogfood consent artifact visibility, public
calibration packet readiness, and guarded Metal evidence. It has blocked rows
for dogfood capture consent, dogfood training consent, real dogfood events and
training rows, public calibration approval/execution, Metal production-route
approval, CUDA reference reports, and accelerator parity-claim approval. The
no-cheat invariant row is proven with external inference, public training rows,
teacher calls, serving external inference, and fallback returns all zero/false.

Current gate reports:

- `reports/dogfood_trace_readiness.json`: `GREEN`, fail-closed. Local config is
  present, capture/training/raw text are disabled, accepted outcomes are only
  `accepted`, `missed`, and `ignored`.
- `reports/dogfood_trace_event_dry_run.json`: `YELLOW`, event not written
  because capture consent is disabled. Forbidden raw text, public benchmark,
  secret, and fallback-return fields are absent.
- `reports/dogfood_trace_training_bridge.json`: `YELLOW`, zero events, zero
  trainable events, zero private training rows, teacher/public/external
  inference/fallback all zero.
- `reports/operator_bounded_public_calibration_dry_run.json`: dry run only.
  Packet ready, approval invalid, executed false, no public surface spent.
- `reports/macos_metal_production_route_readiness.json`: ok true but
  production route not allowed. Guarded evidence is 4/4, route evidence is
  4/4, scheduler canary evidence is 4/4, state-training proof is present, and
  0/4 production routes are enabled.
- `reports/accelerator_parity_claim_readiness.json`: ok true but parity claim
  not allowed. No-cheat evidence is clean and 0/4 surfaces are parity-ready
  because CUDA reference reports are absent.
- `reports/accelerator_parity_manifest.json`: `GREEN`, 7/7 accelerator
  surfaces ok, 4/4 Metal reports ok, 4 scheduler canaries present, and zero
  production routing.
- `reports/external_inference_audit.json`: ok true, zero violations.
- `reports/maturity_integrity_audit.json`: `YELLOW`, hard blockers 0 and
  evidence blockers 0. Maturity blockers remain public transfer floor,
  candidate coverage transfer freshness, and STS capsules needing a causal
  consumer. Locked public transfer remains `34/160 = 0.2125`.
- `reports/private_residual_repair_v3_gate.json`: `GREEN`. Full private v3
  heldout score is present under adapter-off scoring. Learned/student pass rate
  is `240/240 = 1.0`, structural-action pass rate is `240/240 = 1.0`,
  diagnostic-adapter pass rate is `0.0`, no-admissible rate is `0.0`, fallback
  return candidate count is `0`, stdin proxy passes are `48`, and the matched
  STS ablation is clean.
- `reports/private_residual_v3_sts_ablation.json`: `GREEN`. Candidate budget is
  `4`; STS selected pass rate is `1.0`; matched non-STS selected pass rate is
  `0.766667`; both oracle/pass-if-any rates are `1.0`; selected-rate lift is
  `+0.233333`; oracle lift is `0.0`; no-admissible, fallback, public, teacher,
  and external-inference counters are clean. Treat STS as selection/ranking
  evidence on this private slice, not emission coverage evidence.
- `reports/candidate_promotion_gate.json`: promote false, `23/28` gates
  passed. The failed gates are public comparator no-regression, runtime cost
  reported, CUDA no-fallback evidence, graduation transfer artifact readiness,
  and maturity integrity GREEN.
- `reports/hive_overnight_training_report.json`: ok true over the latest
  12-hour window with 291 MLX worker reports, 200 recent jobs, 0 failures,
  0 stale leases, 0 promotions, and 3 best-by-arm records.

Mac-native accelerator status is useful but still bounded. The accelerator
manifest remains green for audit surfaces, Metal guarded evidence exists, and
the long local MLX utilization loop is still producing bounded worker reports.
No public calibration was run, no model was promoted, no external inference was
served, no teacher was called, no fallback returns were introduced, no remote
task scope changed, and no Metal production route or parity claim was enabled.

The exact CUDA reference commands are recorded in
`reports/theseus_gate_closure_packet.json`. They must be run on a CUDA-capable
node to produce:

- `reports/symliquid_standalone_cuda_train_report.json`
- `reports/symliquid_rollout_cuda_train_report.json`
- `reports/symliquid_rollout_cuda_sweep.json`
- `reports/token_superposition_cuda_training.json`

Network state is still not distributed-proof ready. `reports/hive_network_doctor.json`
is `RED`: the Windows coordinator at the last known URL is unreachable, remote
peer reachable count is `0`, and one peer is stale/flapping. Until that is
fixed, current autonomy evidence is local Apple MLX/Metal evidence, not live
Windows CUDA plus Mac MLX distributed evidence.

## 2026-06-14 Overnight Generalization Audit

`reports/neural_seed_token_decoder_96eval_4096train_multiseed.json` is now
`GREEN` across seeds `23,29,31,37,41`. This was the larger private held-out
gate requested after the earlier 24-row slice looked too thin: `4096` private
train rows, `96` private eval rows per seed, five seeds, matched SymLiquid-style
and transformer-control token decoders, no public calibration, no teacher call,
no external inference, no fallback terminal returns, and no promotion. Runtime
was `3175153 ms` (`~52.9m`), so this was a real unattended gate rather than a
short smoke.

The result is exact parity at this scale: SymLiquid mean `0.820833` with stdev
`0.010206`; transformer mean `0.820833` with stdev `0.010206`;
`symliquid_minus_transformer_sts_on_mean=0.0`; winner counts are `0`
SymLiquid, `0` transformer, `5` ties. The deterministic visible-contract beam
selected rate stayed `0.0` for both arms, so this is not a beam or renderer
shortcut. SymLiquid did use learned internal semantic routes more often
(`0.835417` versus transformer `0.539583`), especially contract-fingerprint and
context-prototype routes, but that routing advantage did not produce verifier
wins on the current held-out slice.

`reports/theseus_saturation_generalization_audit.json` is also `GREEN`. It
records that the old token-decoder route ablation is a thin private slice
(`24/240` available private eval rows), the public `5x32` calibration remains
spent and locked at `34/160 = 0.2125`, and the new 96-row multi-seed gate is
the current preferred private token-decoder evidence. Across the larger gate,
there were `394` both-pass rows and `86` both-fail rows, with identical
residual shape for both arms. The top shared residual families are
`THRESHOLD_LABELS` (`31`), `TOP_K_FREQUENT` (`24`),
`ROOM_CAPABILITY_SUMMARY` (`12`), `GROUP_RECORDS_BY_FIELD` (`11`), and
`LONGEST_EVEN_RUN` (`8`).

`reports/theseus_plan_semantic_residual_miner.json` is `GREEN` and corrects the
residual interpretation. Those labels are often the last selected wrong plan
when no candidate passed, not the true expected plan. Across the five-seed gate
there are `45` unique both-fail tasks and `172` seed/arm both-fail events.
The expected coarse plan is present in `172/172` candidate sets, appears through
contract-route candidates in `172/172`, and is top-ranked in `95/172`. The wall
is therefore not primarily "find the coarse plan"; it is that generic
`LIST_APPEND`, `DICT_GROUP_APPEND`, and `GENERIC_RETURN` bodies are too weak for
contract families that require parsing, numeric transformation, grouping, or
shaped returns.

A focused seed-23 diagnostic probe showed the headroom and the trap: executable
contract-family renderers for those generic plans would move the audit from
`80/96` both-pass and `16/96` both-fail to `95/96` both-pass and `1/96`
both-fail for both arms. That probe is explicitly recorded as
`rejected_as_capability_evidence`, because it lets the renderer do too much of
the work and is too close to a canned body/template shortcut. The production
comparator path was restored. Future accepted work should improve the
learned/ranked semantic-slot path, richer slot targets, or grammar/AST-valid
non-fallback statement generation; do not add task-id branches, fallback
returns, canned family bodies, public data, teacher data, or held-out solution
bodies.

`reports/theseus_survival_path_decision.json` remains `GREEN` with decision
`transformer_first_survival_path_symliquid_discovery_lane`. The larger private
token gate gives SymLiquid parity, not superiority. The matched transformer
still has the practical survival edges from the code proposer comparator
(`+0.125`) and route-dropout (`0.816667` versus SymLiquid `0.8`). Near-term
assistant-facing work should therefore go through the transformer control first
while SymLiquid stays protected as a matched discovery lane. The next useful
training pressure is not another broad private suite; it is a bounded
learned/ranked semantic-slot change against the shared residual families above,
followed by the same 96-row five-seed gate.

## 2026-06-14 Neural Seed Token Decoder Snapshot

Current authoritative route-independence ablation:
`reports/neural_seed_token_decoder_route_independence_ablation.json` is
`GREEN` across seeds `23,29,31,37,41`, starting from commit
`ba19fa18b5f8ddde1b0d156f003824311ba3837a` with local source/config changes
to the SymLiquid source encoder, learned route budget, train-contract
fingerprint route memory, and contract-feature route memory. It compares six matched variants: full learned
routing with the visible-contract beam on, full learned routing with the beam
off, 50% route-memory dropout with the beam off, no visible-text prototype
memory, plan-head-only/no prototype memory, and no internal routing. The latest
full ablation runtime was `5256702 ms` (`~1h28m`), so this remains a meaningful
unattended gate rather than a short smoke.

Hard safety gates pass for all variants: fallback terminal returns are `0`,
external inference calls are `0`, no teacher calls run, no public training rows
are used, promotion remains locked, and there are no comparator or audit hard
gate failures. Both arms stay parameter-matched. The deterministic
visible-contract beam is still not carrying the score: beam-on and beam-off
both score SymLiquid `0.85` and transformer `0.85`, and the selected beam rate
remains `0.0`.

The concrete improvement is in contract/source-context routing, not in a
renderer, fallback, public row, teacher row, or answer-shape shortcut. Full
beam-off learned routing remains SymLiquid `0.85` and transformer `0.85`, while
SymLiquid visible-text prototype selection falls from the prior `0.475` wall to
`0.15`. The replacement signal is explicit: full beam-off selects
`contract_fingerprint_context_memory` at `0.325` and
`plan_head_plus_context_prototype_memory` at `0.241667`. The no-visible
variant now also scores SymLiquid `0.85` and transformer `0.85`, with SymLiquid
visible-text selected rate `0.0`, contract-feature selected rate `0.15`,
contract-fingerprint selected rate `0.325`, and context-prototype selected rate
`0.241667`.

The attribution controls are stable. Route-memory dropout scores SymLiquid
`0.8` and transformer `0.816667`, improving the old SymLiquid dropout wall from
`0.7` to the target threshold. The margin is still thin: dropout seeds `37` and
`41` remain at `0.75`, and the residual miner still finds `6` SymLiquid dropout
regressions. Under dropout, SymLiquid selects `contract_feature_context_memory`
at `0.15`, `contract_fingerprint_context_memory` at `0.208333`,
`plan_head_plus_context_prototype_memory` at `0.116667`, visible-text prototype
memory at `0.108333`, and the direct plan head at `0.133333`. Plan-head only
remains SymLiquid `0.441667` and transformer `0.508333`, and no-internal remains
SymLiquid `0.141667` and transformer `0.2`. The learned-routing lift over
no-internal is therefore still real (`+0.708333` for SymLiquid), while the old
visible-text memory delta is now `0.0`.

Complementarity is explicitly audited in
`reports/neural_seed_token_decoder_complementarity_audit.json`, also `GREEN`.
It reloaded all six variant reports and `30` per-seed semantic audits,
recomputed `720` private task-row gap statuses with `0` mismatches, and kept
all hard safety gates green. The full-route complementarity result is sober:
SymLiquid and transformer tie at `0.85`, bounded union gain is `0.0`, and there
are `0` stable full-route SymLiquid-only wins. The recommendation remains to
keep SymLiquid as a protected discovery lane, not to claim it is already a
better survival substrate than the matched transformer.

`reports/neural_seed_token_decoder_residual_context_miner.json` is now `GREEN`
against the fresh ablation. Its recommendation is to treat the
contract/context route as the current replacement for visible-text prototype
memory. Remaining pressure is no longer "reduce visible route dependence";
the useful next work is hardening remaining plan-head dropout misses and the
full-route both-fail families. SymLiquid dropout regressions fell from `18` to
`6`, concentrated in device routing, grouped interval algorithms, long-horizon
planning, state machines, and spatial operators. Full-route both-fail rows stay
at `18`, concentrated in tool transcript, structured parsing,
numeric/collection transforms, algorithmic planning, heterogeneous numeric
contracts, and state-machine parser cases. Do not add fallback returns; they
remain cheating and are hard-gated at zero.

`reports/theseus_survival_path_decision.json` is now `GREEN` and makes the
near-term survival-path recommendation explicit:
`transformer_first_survival_path_symliquid_discovery_lane`. The matched
transformer is at least competitive on the token decoder, leads the code
proposer comparator by `0.125`, and has slightly stronger route-dropout
behavior (`0.816667` versus SymLiquid `0.8`). SymLiquid still reaches
token-decoder parity (`0.85` full/no-visible) and remains protected as a
discovery lane, but it should not absorb near-term assistant-building work
unless it produces matched wins or measurable complementarity.

`reports/dogfood_trace_readiness.json` is now `GREEN`. It defines the
consent-gated daily-use trace contract but does not collect user text and does
not train on user text. Defaults are safe: capture disabled, raw-text capture
disabled, training disabled, external inference `0`, public benchmark training
`0`, and promotion locked. `scripts/dogfood_trace_event.py` is the guarded
event logger; it refuses to write events until local capture consent is enabled
in `configs/dogfood_trace.local.json`. The event schema is ready for
assistant-facing lanes: tool transcript, structured parsing, state-machine
parser, long-horizon planning, device routing, storage/operator, and chat
checkpoint. The next product-facing step is operator-enabled logging of
accepted/missed/ignored real tasks, still with no trace training until
separately consented.

The next neural-seed step should target the remaining plan-head dropout misses
and full-route both-fail residual families using mined residuals, while putting
assistant-facing survival improvements through the transformer control first.
Do not add another broad private ecology suite just to get green numbers.

An attempted richer contract-feature semantic-slot route was probed under
`reports/neural_seed_token_decoder_semantic_slot_route_probe.json` across seeds
`23,29,31`. The underlying comparator/audit gates were clean, with fallback
returns `0`, external inference `0`, and teacher/public/promotion locked, but
the first-three-seed score was unchanged from the prior baseline:
`0.875/0.916667/0.833333` for both SymLiquid and transformer. The unproven
decoder change was not kept. The wall is therefore not just "emit more train
semantic slots"; it is finer plan semantics/ranking for the remaining generic
tool-transcript, structured-parsing, list/dict, and state-machine residuals.

An attempted hybrid semantic-slot plus learned body-token target was probed
under
`reports/neural_seed_token_decoder_semantic_slots_plus_body_feasibility_24eval_seed23_comparator.json`
and
`reports/neural_seed_token_decoder_semantic_slots_plus_body_feasibility_24eval_seed23_semantic_plan_gap_audit.json`.
It was safety-clean on the focused seed-23 24-row slice: fallback returns `0`,
external inference `0`, teacher/public/promotion locked, and both arms scored
`21/24 = 0.875`. That exactly ties the existing seed-23 baseline for the same
slice, while the larger 96-row version was too slow for a useful focused
diagnostic. The comparator change was therefore reverted. Learned body-token
targets are parked until they can show movement on the 96-row five-seed gate
without renderer shortcuts, fallback returns, or held-out-solution leakage.

`reports/neural_seed_candidate_ranker_boundary_audit.json` is `GREEN` and
closes off a tempting but insufficient next step. Across the current 96-row
five-seed gate it reads `960` arm-task events, `172` failure events, and all
`172` failures exhausted every available candidate. That means candidate
reranking alone cannot reduce the `86` shared both-fail rows per arm; a ranker
can only reduce verifier work or change which passing candidate is found first.
The next useful work must generate new learned non-fallback candidates, not
just reorder existing candidates.

A direct `body_tokens` feasibility run under
`reports/neural_seed_token_decoder_body_tokens_feasibility_24eval_seed23_comparator.json`
is `RED` and should not be promoted: both arms scored `0/24`, with fallback
returns still `0`, external inference `0`, and teacher/public/promotion locked.
The failure is informative. Free body-token generation has not learned enough
syntax/semantics at the small diagnostic budget, and semantic route candidates
are not valid under raw body-token decoding because route slots become literal
candidate tokens. The next admissible generator should be a two-stage learned
detail decoder or constrained AST/body-token decoder trained on private train
rows, with semantic route slots used as conditioning metadata rather than
rendered as Python.

`reports/neural_seed_plan_conditioned_detail_body_probe_seed23_grammar_v3.json`
is the follow-up diagnostic for that two-stage idea. It trained matched
plan-conditioned body-token decoders on private train rows and appended
generated detail candidates after the existing private seed-23 baseline. The
run stayed safety-clean: fallback rows `0`, external inference `0`, teacher
use `false`, public training rows `0`, promotion locked, and matched parameter
delta `0.002219`. It did not improve the verifier wall: both arms remained
`21/24 = 0.875`, with `3` residual rows still failing. The temporary grammar
mask tightening was not accepted as a decoder improvement because it changed
syntax validity without producing verifier movement. The next admissible
overnight goal should therefore build a structural detail target, such as a
private-train learned AST/action decoder with explicit return-shape and
statement-role supervision, before spending another five-seed 96-row gate.

`reports/neural_seed_structural_action_decoder_96eval_multiseed.json` is now
`GREEN` and is the first accepted movement on the shared 96-row residual wall
after the neutral body-token attempts. The diagnostic trains matched
SymLiquid-style and transformer-control classifiers over private-train
structural action-sequence classes, then compiles predicted line-action tokens
with a generic private-train line-action compiler. It appends those candidates
after the existing 96-row seed manifests and scores only through the private
verifier. All five seeds (`23,29,31,37,41`) are green; fallback returns are
`0`; syntax pass-rate minimum is `1.0`; external inference, teacher use, public
training rows, and model promotion all remain `0`/locked.

The result is material. SymLiquid moves from `0.820833` to `0.945833` mean
private verifier pass-rate (`+0.125`, stdev `0.035325`), while the transformer
control moves from `0.820833` to `0.910417` (`+0.089583`, stdev `0.034233`).
Residuals drop by `60` total rows for SymLiquid and `43` for transformer. The
exact shared-both-fail audit in
`reports/structural_action_shared_both_fail_audit.json` reduces shared
both-fails from `86` to `18`, a delta of `68`.

This is accepted as a structural-action diagnostic and survival-path candidate,
not as runtime promotion and not as public-calibration unlock evidence. It is
also not a claim that free-form generation is solved: the compiler still
selects from private-train structural line/action sequences. The next
production step is to integrate this structural action decoder behind the
token-decoder comparator as an explicitly named candidate family, then add an
ablation that separates sequence-class selection, line-action compilation, and
any future finer-grained AST synthesis. Keep the hard rule: no fallback
returns, no semantic-family body renderers, no task-id branches, and no public
benchmark spending until a separate reviewed calibration plan exists.

### 2026-06-14 Structural Integration, Dogfood Bridge, and Mac MLX

The structural-action decoder is now integrated behind the main token-decoder
comparator as an explicit candidate family, not as a fallback or hidden
renderer. `configs/neural_seed_token_decoder_comparator.json` enables
`body_structure_decoder.structural_action_family`, and
`scripts/neural_seed_token_decoder_comparator.py` allows structural-action rows
only when that family is explicitly enabled. The integrated smoke
`reports/neural_seed_token_decoder_structural_integrated_smoke.json` is
`GREEN`: `192` private train rows, `8` private eval rows, `136` candidate rows,
`16` structural-action rows, both arms at `0.875`, syntax pass rate `1.0`, and
fallback return rate `0.0`. External inference, teacher use, public training,
and promotion remain `0`/locked.

`reports/neural_seed_structural_action_ablation_report.json` is `GREEN` and
separates the axes explicitly. Sequence-class selection is implemented for both
matched arms over private-train structural action sequences. Line-action
compilation is implemented with `generic_private_train_line_action_compiler_v0`
and has syntax minimum `1.0` with fallback maximum `0.0`. Finer AST synthesis is
separated but not implemented; verifier gains must not be attributed to it yet.

The latest full comparator refresh
`reports/neural_seed_token_decoder_comparator.json` is also `GREEN` and now
includes an explicit no-cheat evidence view. The normal diagnostic run produced
`1258` generated rows, but the no-cheat filter quarantined `1066` semantic
family body-renderer rows and `26` visible-contract semantic-prior rows. Those
diagnostic rows remain non-promotion evidence. The filtered capability slice
contains `192` structural-action private-eval rows, with fallback returns `0`,
terminal null returns `0`, task-id keyed candidates `0`, public/eval leakage
`0`, teacher/external inference `0`, and filtered private verifier pass rate
`1.0` on this 24-row private slice. This is private residual evidence only; it
does not unlock public calibration, promotion, or model growth.

The matched-arm result is intentionally recorded without overclaiming:
SymLiquid-style scored `0.958333` STS-on verifier pass-rate and the matched
transformer control scored `1.0` on the same private 24-row slice, with
`symliquid_minus_transformer_sts_on_verifier_pass_rate=-0.041667`. The
refreshed `reports/neural_seed_growth_gate.json` now consumes the route
ablation, complementarity audit, residual-context miner, and structural-action
ablation as first-class private prerequisites: `spec_ready=true`,
`execute_allowed=false`, and `neural_student_ready=false`. This is the desired
state. Private decoder evidence is ready enough to be trusted by the gate, but
model growth is still locked and `reports/teacher_distillation_gate.json`
remains `YELLOW`/operator-locked. This confirms the structural-action family
materially improves the private decoder wall but still does not justify teacher
distillation, public calibration, model growth, or runtime serving claims.

Dogfood tracing remains fail-closed. `configs/dogfood_trace.local.example.json`
documents the opt-in local consent file. Consent timestamps must be valid UTC
ISO-8601 values, and training consent must be a separate deliberate timestamp
from capture consent. `scripts/dogfood_trace_readiness.py`,
`scripts/dogfood_trace_event.py`, and
`scripts/dogfood_trace_training_bridge.py` enforce the consent path; the bridge
converts accepted/missed/ignored metadata events into private training rows
only when both capture consent and training consent are present. The current report
`reports/dogfood_trace_training_bridge.json` is `YELLOW`: the local fail-closed
config exists, capture/training consent is disabled, there are no trace events,
and `0` training rows were written. This is correct behavior; do not
train on daily-use traces until consent and redacted events exist.

Mac MLX is now available in the active Apple Silicon Python and the installed
Hive/runtime environments. `reports/hive_status.json` advertises `mlx_apple`,
`reports/macos_mlx_work_proof.json` is `GREEN` with `3/3` worker smokes and
`4/4` CLI smokes passing, and `reports/performance_optimizer.json` is `GREEN`
with `apple_mlx` as the preferred training/inference backend. The targeted
structural-action MLX smoke `reports/macos_mlx_structural_action_smoke.json` is
also `GREEN`: MLX GPU device active, loss decreased, syntax `1.0`, fallback
rows `0`, and private verifier pass rate `0.25` on the small 8-row smoke. This
proves a useful Apple Silicon MLX bridge for the structural-action surface, but
native Rust/Metal or Rust/MLX hot-loop parity is still pending and must not be
claimed as complete.

`reports/macos_metal_rollout_hot_loop_proof.json` is the first Rust-owned
native Metal hot-loop proof. The new `crates/symliquid-metal` crate compiles a
Metal `rollout_state_update` kernel and tests it against the existing Rust CPU
reference that mirrors the CUDA rollout parity case. The proof is `GREEN` with
max CPU-vs-Metal absolute delta `2.98e-8` under tolerance `1.0e-4`. It has
external inference `0`, teacher use `0`, public training rows `0`, and does not
route the scheduler to Metal.

The same proof is now exposed through the installed-style Rust CLI:
`cargo run -p symliquid-cli -- rollout-metal-proof --out
reports/macos_metal_rollout_cli_proof.json` writes a `GREEN` report with the
same max delta `2.98e-8`. This proves a user-facing Mac-native proof command is
available without shelling to the Python MLX bridge. It is still only a
single-kernel reference proof; it does not claim `train-rollout-mlx` parity, run
training, promote a model, spend public calibration, or alter scheduler routing.

The next Mac-native increment is also in place:
`cargo run -p symliquid-cli -- rollout-metal-feature-proof --out
reports/macos_metal_rollout_feature_proof.json` writes a `GREEN` report for a
bounded rollout memory feature tensor over deterministic private synthetic
sequences. CPU-vs-Metal feature max delta is `0.0` on the default 6-row feature
probe. This is stronger than the single-kernel proof because it exercises the
chunked feature tensor path, but it is still non-promotional: readout
training/evaluation parity is not proven, `train-rollout-mlx` still uses the
Python MLX bridge, public calibration is not spent, and scheduler routing is not
changed.

The Rust/Metal path now also covers bounded readout logits:
`cargo run -p symliquid-cli -- rollout-metal-readout-proof --out
reports/macos_metal_rollout_readout_proof.json` writes a `GREEN` report for
rollout memory features plus `linear_readout_logits_kernel`. CPU-vs-Metal
feature max delta is `0.0`, logits max delta is `8.94e-8`, and prediction
agreement is `1.0` on deterministic private synthetic sequences and weights.
This is still a proof artifact, not a training lane: it does not train the
readout, does not claim full `train-rollout-mlx` parity, does not spend public
calibration, does not use a teacher, and does not route scheduler work to Metal.

The next bounded native step is now in place:
`cargo run -p symliquid-cli -- rollout-metal-readout-training-proof --out
reports/macos_metal_rollout_readout_training_proof.json` writes a `GREEN`
report for rollout memory features plus Metal `readout_sgd_samples_kernel`
training and Metal logits evaluation. Against the CPU per-sample SGD reference,
feature max delta is `0.0`, weight max delta is `2.98e-8`, bias max delta is
`3.73e-9`, logits max delta is `2.38e-7`, eval loss delta is `0.0`, and
prediction agreement is `1.0`. This removes the previous "logits only" gap for
the bounded readout subpath, but it is still not full train-rollout parity:
the production `train-rollout-mlx` command still uses the Python MLX bridge,
the proof does not write training artifacts, public calibration remains locked,
and scheduler routing is unchanged.

The Metal path now has a train-rollout-style proof report:
`cargo run -p symliquid-cli -- rollout-metal-train-path-proof --out
reports/macos_metal_train_path_proof.json` writes a `GREEN` report with
separate deterministic private train/eval splits, train/eval metrics, kernel
launch counts, a runtime profile, and explicit artifact-write status. It does
not shell to the Python MLX bridge. Current proof deltas are: train feature
max delta `0.0`, eval feature max delta `0.0`, weight max delta `2.98e-8`,
bias max delta `3.73e-9`, eval logits max delta `2.38e-7`, eval loss delta
`1.19e-7`, and eval prediction agreement `1.0`. This is the closest Mac-native
proof so far to the train-rollout path, but it is still a proof path, not the
production scheduler path: no model artifact is written, no scheduler routing
changes, no public calibration is spent, and `train_rollout_parity_claim_allowed`
remains `false`.

The Rust CLI also now exposes a scheduler/audit-facing native command:
`cargo run -p symliquid-cli -- train-rollout-metal --out
reports/symliquid_rollout_metal_train_report.json`. The current bounded smoke
report is `GREEN`, uses `backend=apple_metal`, sets `command=train-rollout-metal`,
mirrors the `train-rollout-mlx` CLI surface, and reports
`python_mlx_bridge_used=false`. With `--model-out
reports/macos_metal_train_rollout_readout_artifact.json`, it writes a canonical
`symliquid_core::benchmarks::ReadoutArtifact` with `hv_dim=8`, `output_dim=4`,
`weights=32`, `bias=4`, and `labels=4`. The parity audit verifies the artifact
shape, feature set, no-teacher/no-public/no-external-inference flags,
`no_fallback_returns=true`, `scheduler_routing_enabled=false`, and
promotion/parity locks. This is accepted as native CLI contract plus artifact
equivalence evidence, not production scheduler parity.

The standalone training CLI now has the same kind of guarded Rust/Metal proof:
`cargo run -p symliquid-cli -- train-standalone-metal --cases-per-task 4
--epochs 2 --samples-per-launch 4 --hv-dim 64 --lr 0.03 --model-out
reports/macos_metal_train_standalone_readout_artifact.json --out
reports/symliquid_standalone_metal_train_report.json`. The report is `GREEN`,
uses `backend=apple_metal`, sets `command=train-standalone-metal`, mirrors the
`train-standalone-mlx` CLI surface, and reports `python_mlx_bridge_used=false`.
It writes a canonical readout artifact with `hv_dim=64`, `output_dim=60`,
`weights=3840`, `bias=60`, and `labels=60`. Current bounded metrics are
`train_accuracy=0.325`, `eval_accuracy=0.55`, `train_loss=3.8093`,
`train_examples_per_second=536.9`, and `kernel_launches=20`. The report records
zero external inference, zero teacher use, zero public training rows,
`no_fallback_returns=true`, `scheduler_routing_enabled=false`,
`model_promotion_allowed=false`, and `train_standalone_parity_claim_allowed=false`.
This closes the next Mac-native proof increment for `train-standalone-cuda`, but
it does not enable production scheduling or a full CUDA/MLX/Metal parity claim.

`configs/macos_metal_route_policy.json` now records the explicit
`train-rollout-metal` route guard: `route_state=guarded_smoke_only`,
`bounded_smoke_route_enabled=true`, and
`production_scheduler_routing_enabled=false`. It requires the canonical artifact,
artifact-equivalence proof, zero teacher/public/external-inference usage,
`no_fallback_returns=true`, no model promotion, no parity claim, unchanged Hive
remote-task scope, and rollback guardrails before any scheduler dry-run. This is
a scheduler contract to audit against, not an enabled production route.

The scheduler dry-run has now executed under that guard:
`python3 scripts/hive_scheduler.py --macos-metal-dry-run --execute --out
reports/hive_scheduler_macos_metal_dry_run_execute.json`. It writes
`reports/macos_metal_scheduler_dry_run.json`,
`reports/macos_metal_scheduler_dry_run_train_report.json`, and
`reports/macos_metal_scheduler_dry_run_readout_artifact.json`. The dry-run is
local-only, uses `task_kind=train_rollout_metal_dry_run` rather than a registered
remote worker chunk, keeps `production_scheduler_routing_enabled=false`, submits
no remote task, runs bounded private synthetic work with `kernel_launches=32`
under a `64` cap, writes a canonical readout artifact, and verifies no teacher,
no public training rows, no external inference, no fallback returns, no parity
claim, and no model promotion.

The bounded Metal ladder is now also part of the audit chain:
`reports/macos_metal_parity_ladder.json` is `GREEN` across four local private
synthetic tiers (`tiny_guarded`, `small_guarded`, `medium_guarded`,
`wide_guarded`). It runs `345` total Metal kernel launches, writes four
canonical readout artifacts, and uses an explicit `<=5e-4` f32 numerical
tolerance. The ladder is local-only: production scheduler routing remains
`false`, no remote task is submitted, model promotion remains `false`,
`train_rollout_parity_claim_allowed=false`,
`native_hot_loop_parity_claim_allowed=false`, external inference calls are `0`,
teacher use is `false`, public training rows are `0`, and fallback returns are
`0`.

The reviewed local scheduler canary is now proven:
`python3 scripts/hive_scheduler.py --macos-metal-canary --execute --out
reports/hive_scheduler_macos_metal_canary_execute.json` writes
`reports/macos_metal_scheduler_canary.json`,
`reports/macos_metal_scheduler_canary_train_report.json`, and
`reports/macos_metal_scheduler_canary_readout_artifact.json`. The canary requires
`configs/macos_metal_scheduler_canary_policy.json`, the prior dry-run, the
bounded ladder, and the parity audit before it executes. It runs only local
`task_kind=train_rollout_metal_local_canary`, does not register a Hive remote
worker chunk, submits no remote task, keeps `production_scheduler_routing_enabled=false`,
and uses `99` Metal kernel launches under the declared `192` cap with explicit
`<=5e-4` tolerance. The child train report is `GREEN`, writes the canonical
readout artifact, uses `backend=apple_metal`, reports `python_mlx_bridge_used=false`,
and keeps public training rows `0`, teacher use `false`, external inference `0`,
fallback returns `0`, model promotion `false`, and parity claims `false`.

The first native Metal token-superposition readout subpath is now proven:
`cargo run -p symliquid-cli -- token-superposition-metal-readout-proof --out
reports/macos_metal_token_superposition_readout_proof.json` writes a `GREEN`
report for deterministic private synthetic token-superposition data. It ports
the bag readout SGD, AR recovery SGD, and logits readout pieces to
Rust-owned Metal kernels (`readout_bag_sgd_samples_kernel`,
`readout_sgd_samples_kernel`, and `linear_readout_logits_kernel`) and compares
them against sequential CPU references. Current proof deltas are: baseline
weight max delta `1.49e-8`, baseline bias max delta `1.49e-8`, baseline loss
delta `0.0`, variant weight max delta `1.49e-8`, variant bias max delta
`1.49e-8`, variant logits max delta `4.47e-8`, variant loss delta `2.38e-7`,
and prediction agreement `1.0`. This is accepted as native token-superposition
readout subpath evidence only: scheduler routing, model promotion, public
calibration, teacher use, external inference, fallback returns, and any full
parity claim all remain locked.

The first full native Metal token-superposition CLI contract is also present:
`cargo run -p symliquid-cli -- train-token-superposition-metal --max-language-rows
96 --max-vocab 32 --hv-dim 64 --train-samples 64 --eval-samples 32
--baseline-epochs 2 --bag-sizes 4 --recovery-ratios 0.5 --samples-per-launch 8
--gate-tolerance 0.002 --out reports/token_superposition_metal_training.json`
writes `project_theseus_token_superposition_metal_report_v1`. This mirrors the
`train-token-superposition-mlx` command surface against the
`train-token-superposition-cuda` parity target, runs with `backend=apple_metal`,
and records `implementation=rust_metal_token_superposition_readout_cli`. The
bounded smoke produced baseline combined loss `2.1037`, best TST combined loss
`2.4905`, best variant `tst_s4_r0.50_metal`, and `26` Metal kernel launches.
The raw token-superposition gate stayed blocked by
`normal_recovery_loss_beats_baseline`; the Metal contract preserves that raw
decision, forces `model_promotion_allowed=false`, and records no teacher use,
no public training rows, no external inference calls, no fallback returns, no
scheduler routing, and no parity claim.

The Metal token-superposition CLI now also writes a canonical readout artifact
when `--model-out` is requested. The current artifact proof runs:
`cargo run -p symliquid-cli -- train-token-superposition-metal --input
data/training_data/high_transfer/private_train/targeted_private_residual_curriculum_v2_residual_code_lm_tasks.jsonl,data/training_data/high_transfer/private_train/algorithmic_planning_residual_code_lm_tasks.jsonl,data/training_data/high_transfer/private_train/parsing_encoding_v1_private_residual_curriculum_residual_code_lm_tasks.jsonl
--train-seed 2026061404 --max-language-rows 96 --max-code-files 0
--max-chars-per-doc 6000 --max-vocab 32 --hv-dim 64 --train-samples 96
--eval-samples 32 --baseline-epochs 2 --bag-sizes 4 --recovery-ratios 0.5
--samples-per-launch 8 --gate-tolerance 0.002 --model-out
reports/macos_metal_token_superposition_readout_artifact.json --out
reports/token_superposition_metal_training.json`. The report records
`artifact_write.production_checkpoint_compatible=true`,
`kind=canonical_readout_artifact`, feature set
`metal_token_superposition_readout_private_residual_train_eval`, `hv_dim=64`,
`output_dim=32`, `weights_written=2048`, `bias_written=32`, and
`labels_written=32`. This is artifact-equivalence evidence only; the raw gate
still blocks promotion by `normal_recovery_loss_beats_baseline`, and scheduler
routing, model promotion, public calibration, teacher use, external inference,
fallback returns, and full parity claims remain locked.

The bounded token-superposition Metal ladder is now green:
`python3 scripts/macos_metal_token_superposition_ladder.py --execute --out
reports/macos_metal_token_superposition_ladder.json --markdown-out
reports/macos_metal_token_superposition_ladder.md` runs three private-only
tiers over `data/training_data/high_transfer/private_train` residual curriculum
JSONL files. It validates
`configs/macos_metal_token_superposition_route_policy.json`, keeps
`production_scheduler_routing_enabled=false`, excludes project-code mixing, and
records `3/3` tiers ok with `168` total Metal kernel launches. The largest tier
used `114` launches under its `160` cap. This is route-guard and ladder
evidence only: it uses no public calibration, no public training rows, no
teacher, no external inference, no fallback returns, no remote task submission,
no model promotion, and no full native parity claim.

The reviewed local token-superposition scheduler canary is also proven:
`python3 scripts/hive_scheduler.py --macos-metal-token-superposition-canary
--execute --out reports/hive_scheduler_token_superposition_canary_execute.json`
writes `reports/macos_metal_token_superposition_scheduler_canary.json`,
`reports/macos_metal_token_superposition_scheduler_canary_train_report.json`,
and
`reports/macos_metal_token_superposition_scheduler_canary_readout_artifact.json`.
The canary requires
`configs/macos_metal_token_superposition_scheduler_canary_policy.json`, the
token-superposition route policy, the bounded ladder, the canonical artifact,
and the parity-audit proof chain before it executes. It runs only local
`task_kind=train_token_superposition_metal_local_canary`, does not register a
Hive remote worker chunk, submits no remote task, and keeps
`production_scheduler_routing_enabled=false`. The current child train report
uses private residual JSONL inputs only, excludes project-code mixing, records
`train_tokens=2241`, `vocab_size=48`, `hv_dim=96`, best variant
`tst_s4_r0.25_metal`, and `114` Metal kernel launches under the declared `192`
cap. It writes a production-compatible canonical readout artifact while keeping
promotion blocked by `normal_recovery_loss_beats_baseline`; public calibration,
public training rows, teacher use, external inference, fallback returns,
remote-task submission, model promotion, production routing, and parity claims
all remain locked.

`reports/accelerator_parity_manifest.json` is now `GREEN` and closes the audit
target for comparable accelerator manifests. It reads the seven CUDA-equivalent
surfaces currently covered on Mac (`mlx_eval_chunk`, `mlx_training_chunk`,
`mlx_rollout_chunk`, `train-standalone-mlx`, `train-rollout-mlx`,
`train-rollout-mlx-sweep`, and `train-token-superposition-mlx`), then attaches
the Rust/Metal `train-standalone-metal`, `train-rollout-metal`,
`train-rollout-metal-sweep`, and `train-token-superposition-metal` reports plus
their reviewed local canaries where available. It records `7/7` surfaces ok,
`7/7` MLX reports ok, `4/4` Metal reports ok, `4` canonical artifact manifests,
`2` scheduler canaries, external inference calls `0`, teacher use `0`, public
training rows `0`, promotion-enabled rows `0`, and production-routing rows `0`.
This is audit evidence only: it does not spend public calibration, call a
teacher, enable production scheduler routing, promote a model, or claim full
CUDA/MLX/Metal parity.

`reports/macos_mlx_parity_audit.json` records the boundary explicitly. It is
still `YELLOW`: MLX is available in all checked Mac runtimes, all three
registered MLX worker chunks have current evidence, and all four Rust CLI MLX
bridge commands are runnable. The audit now sees one native subkernel proof,
one main-CLI native proof, one native feature-path proof, and one native
readout-logits proof, one bounded readout-training subpath proof, one
train/eval report-style proof, and one native train-rollout CLI contract proof,
plus one native train-rollout artifact-equivalence proof and one native
train-rollout scheduler-guardrail proof, plus one bounded local scheduler dry-run
proof, one bounded train-rollout Metal ladder proof, one reviewed local
scheduler canary proof, one native standalone CLI contract proof, one native
standalone artifact-equivalence proof, one native token-superposition readout
proof, and one native token-superposition CLI contract proof, one
token-superposition
artifact-equivalence proof, plus one bounded token-superposition Metal ladder
proof, plus one reviewed local token-superposition scheduler canary proof, and
one green accelerator parity manifest. The production-route review
`reports/macos_metal_production_route_readiness.json` is also present and
fail-closed: `guarded_evidence_ok_count=4/4`,
`production_route_ready_count=0`, `production_route_allowed=false`,
`operator_approval_valid=false`, and `hard_failure_count=0`. The 2026-06-15
audit now classifies
`train-standalone-cuda` as guarded Rust/Metal proof-ready, `train-rollout-cuda`
and `train-token-superposition-cuda` as guarded Rust/Metal canary-ready, and
`train-rollout-cuda-sweep` as guarded Rust/Metal sweep-proof-ready with six
child canonical readout artifacts. The native-port pending count is now `0`,
but production-ready native hot-loop parity remains locked:
`production_route_pending_count=4`,
`native_metal_production_route_ready_count=0`, and
`native_metal_production_route_blocker_count=11`. Production scheduling cannot
route to Metal until those blockers are resolved under a separate
operator-reviewed enablement step.
`native_hot_loop_parity_claim_allowed=false` remains the correct state.

Governance after these changes is unchanged in the important way:
`reports/external_inference_audit.json` has `0` violations,
`reports/maturity_integrity_audit.json` is `YELLOW` with `0` hard blockers and
`2` maturity blockers: the spent wide public calibration remains
`34/160 = 0.2125` below floor, and the fresh v16 private-only run still has no
public candidate coverage transfer surface. `reports/candidate_promotion_gate.json`
keeps `promote=false`. The stale accelerator blocker was cleared; maturity now
sees `performance_state=GREEN` and no MLX/CUDA bottleneck.

## 2026-06-06 Self-Improvement Transfer Loop Snapshot

Current Mac-local learning state supersedes older Code LM transfer claims below.
The wide 160-task public calibration has now been consumed once and relocked;
do not rerun or fish this surface.

The current generalization governor is still `YELLOW`
(`reports/theseus_generalization_governor_v1.json`) because the already-spent
public score remains below floor, not because of hard safety failures. Hard
safety gates pass: the public calibration lock is active, approved/spent
post-v4 public artifacts are recognized as allowed, forbidden post-v4 artifact
count is `0`, public calibration is not allowed, public tests/solutions are not
used for training, and `external_inference_calls=0`.
The private learned-maturity stack is current under the tightened broad-private
no-body-memory rule. This is necessary private transfer evidence, but it is not
promotion evidence while the already-spent public score remains below floor:

- v4 learned maturity: `true`, learned-only `1008/1008 = 1.0`, prototype pass
  count `0`, train body overlap `0.0`, and same-seed STS-off control
  `0/1008`.
- post-v4 shadow learned maturity: `true`, learned-only `2400/2400 = 1.0`,
  prototype pass count `0`, train body overlap `0.0`, and same-seed STS-off
  control `0/2400`.
- v5 learned maturity: `true`, learned-only `720/720 = 1.0`, prototype pass
  count `0`, train body overlap `0.0`, and same-seed STS-off control
  `0/720`.
- unseen-transfer learned maturity: `true`, train body overlap `0.0`.
- contract-blind learned maturity: `true`, strict learned-only
  `240/240 = 1.0`, prototype/diagnostic/body-memory passes `0`, pass-path
  diversity `36` AST shapes and `36` unique normalized ASTs, train body overlap
  `0.0`, and same-seed STS-off control `0/240`.
- broad-private learned maturity is now `GREEN` under the stricter no-body-
  memory gate: learned-only and strict novel learned-only scoring both pass
  `1008/1008 = 1.0`, exact private train-body memory pass count is `0`,
  prototype pass count is `0`, train body overlap is `0.0`, and the evidence is
  fresh against the current Rust decoder source and release binary.
- Agent-lane core transfer is current: `reports/high_transfer_long_horizon_tool_use.json`
  is `GREEN` at `64/64` private tool-use cases with `64` trace rows and `64`
  STS rows; `reports/high_transfer_multi_turn_conversation_hard_v4.json` is
  `GREEN` with `384/384` cases, `943/943` turns, accuracy
  `0.9923456101190479`, and personality context ready on every turn.
- `reports/pufferlib4_rl_lane.json` is `GREEN` through the local synthetic RL
  fallback while honestly reporting Puffer native/Ocean as missing. It writes
  `54` private policy trace rows for a key-door gridworld, with policy
  accuracy delta `+1.0`, rollout reward delta `+1.32`, and
  `policy_learning_backend=local_synthetic_rl`.
- `reports/cross_domain_sts_capsules.json` is `GREEN` with `42` metadata-only
  capsules across conversation, long-horizon tool-use, and RL policy lanes.
  `reports/agent_lane_transfer_gate.json` still reads `YELLOW` only because
  public transfer remains below floor; repo repair, terminal tool-use,
  cross-domain STS consumer effect, hard-v4 conversation, and RL policy
  learning all pass.

`reports/public_calibration_readiness_packet.json` is now `GREEN` for technical
operator review, using the current pre-public audit as the authoritative private
readiness handoff. This is not promotion evidence: the already-spent public
score is still below the `0.70` floor. Its safety boundary is intact: the pinned
5x32 public surface has `160` task IDs and no public leak hits, the packet does
not run public calibration, `public_calibration_allowed=false`, and
`reports/public_calibration_operator_lock.flag` is active. The current
governor queue has no safe private refresh remaining after the agent-lane
tool-use/capsule/conversation refreshes; it still stops at
`operator_review_public_calibration_locked`, which requires explicit operator
approval before any bounded public run and immediate relock afterward.
`reports/pre_public_generalization_readiness_audit.json` is now the compact
pre-public handoff artifact for this state. It is `YELLOW` with no hard failed
gates: `operator_review_ready=true`, private code transfer ready, private agent
transfer ready, teacher path ready, `learned_token_pass_count_total=8752`,
prototype pass count `0`, public lock active, forbidden post-v4 public
artifacts absent, and `external_inference_calls=0`. Its only failed gate is
`public_transfer_floor_cleared`, because the spent public calibration remains
`34/160 = 0.2125`. The private frontier expander executed
`expand_private_unseen_transfer_challenge_240`,
`expand_private_residual_frontier_1008`,
`expand_private_unseen_transfer_challenge_360`, and
`expand_private_residual_frontier_1344`, `expand_private_ecology_v5_720`, and
the sharded `expand_private_residual_frontier_840_spec21` and
`expand_private_residual_frontier_1040_spec26` proofs successfully. Its current
decision is
`no_private_frontier_action_remaining`, so the pre-public audit no longer queues
a no-op private continuation; its only queue item is the locked public-review
item with no command and `requires_operator_public_unlock=true`.

Mac-local unattended readiness was refreshed on 2026-06-07 after the governor
and v4 private learned evidence became the current source of truth. The
overnight gate is now platform-native: macOS/Linux runtime storage checks use
the configured runtime path and a `20 GiB` default local floor
(`THESEUS_MIN_RUNTIME_FREE_GIB` can override it), while Windows keeps the
`100 GiB` generated-artifact floor and D-drive preference. The refreshed
`reports/overnight_learning_readiness.json` is `YELLOW` with
`overnight_launch_ready=true`, no red failures, public pass rate
`34/160 = 0.2125`, promotion still blocked, `reports/autonomy_watchdog.json`
`YELLOW`, `reports/code_lm_closure_public_contract_preflight_seed23_32.json`
`GREEN`, and `reports/training_budget_plan.json` sufficient with
`154112` train env steps. This means local unattended private learning can run
honestly, but promotion/model growth and additional public calibration remain
locked until explicit operator approval.

Private Ecology Generalization v5 is now the active private-only ecology proof
after the v4 learned-transfer proof and guarded public-calibration packet:

Current governor note: the full v5 learned-maturity gate is current under the
latest decoder source/release. The public readiness packet is review-green but
operator-locked; it still does not run calibration or unlock promotion.

- `reports/private_ecology_generalization_v5.json`: `GREEN`. The generator
  now writes `1800` private train rows and `720` private heldout rows across `6`
  Hive workflow families and `12` categories: project memory, tool transcript
  parsing, file/storage manifests, device routing, long-horizon planning, and
  spatial/operator media workflows.
- Private solution self-tests pass for train and heldout rows (`0` failures);
  public-data leakage hits are `0`; external inference calls are `0`; public
  benchmark prompts, tests, solutions, traces, score labels, and task IDs are
  not used.
- The v5 preflight confirms
  `reports/public_calibration_readiness_packet.json` exists with
  `public_calibration_allowed=false`, the guarded dry-run did not execute,
  `reports/public_calibration_operator_lock.flag` is still active, and no
  `reports/real_code_benchmark_graduation_post_v4_seed23_5x32.json` artifact
  exists. Public calibration remains locked.
- `reports/private_ecology_generalization_v5_refresh.json`: `GREEN`. The
  refresh runner is now the canonical current-v5 proof: it regenerated v5
  rows, wrote `720` private-safe STS streams, reran STS-on fanout, reran
  same-seed STS-off control, rescored the full heldout set, reran the
  learned-only gate, and proved the score/learned artifacts are fresh after
  the regenerated curriculum. Latest elapsed time was `664.487s`.
- `reports/private_ecology_generalization_v5_queue.jsonl` remains the
  generator's local queue artifact, while
  `reports/private_ecology_generalization_v5_refresh_queue.jsonl`,
  `reports/private_ecology_generalization_v5_refresh_ledger.jsonl`, and
  `reports/private_ecology_generalization_v5_refresh_heartbeat.json` are the
  auditable full-refresh control artifacts.
- The Rust decoder contract recognizer now includes
  `project_theseus_decoder_contract_v5_private_ecology_generalization`, and
  the private-train induced token loader reads the v5 train artifact when
  present. Focused local proof:
  `cargo test -p symliquid-cli private_train_prototype -- --nocapture`
  passed `13/13`.
- `reports/private_ecology_generalization_v5_smoke72_score.json`: `GREEN`.
  The private-only v5 smoke now scores `72/72 = 1.0` across the first `72`
  heldout rows, covering all `6` workflow families and `12` categories.
  Same-seed STS-off control is `0/72`, STS delta is `+1.0`, no-admissible rate
  is `0.0`, regressions are `0`, public-data leakage hits are `0`, and
  external inference calls are `0`.
- `reports/private_ecology_generalization_v5_smoke72_learned_distillation_gate.json`:
  `GREEN`. Learned-only v5 smoke also passes `72/72 = 1.0`; learned-token pass
  count is `72`, learned-only candidate rows are `276`, prototype pass count is
  `0`, prototype rows are `0`, verifier-admissible pass count is `72`, public
  tests/solutions used are `false`, external inference calls are `0`, and
  decoder source/release freshness is proven for the score and learned-only
  artifacts.
- `reports/private_ecology_generalization_v5_full480_score.json`: `GREEN`.
  Despite the historical `full480` filename, the current canonical refresh now
  scores the full `720/720 = 1.0` v5 heldout across all `6` workflow families
  and `12` categories. Same-seed STS-off control is `0/720`,
  STS delta is `+1.0`, no-admissible rate is `0.0`, regressions are `0`,
  public-data leakage hits are `0`, and external inference calls are `0`.
- `reports/private_ecology_generalization_v5_full480_learned_distillation_gate.json`:
  `GREEN`. Learned-only full v5 also passes `720/720 = 1.0`; learned-token pass
  count is `720`, prototype pass count is `0`, prototype rows are `0`,
  verifier-admissible pass count is `720`, learned-only candidate rows are
  `2760`, public tests/solutions used are `false`, external inference calls
  are `0`, and decoder source/release freshness is proven.
- `reports/private_unseen_transfer_challenge_v1.json`: `GREEN`. This private
  OOD transfer challenge now rewrites `240` private v5 heldout rows so exact
  semantic-family keys cannot be replayed. After the train-novel decoder
  repair for project-progress and room-capability aliases, STS-on scores
  `240/240 = 1.0`; learned-only also scores `240/240 = 1.0`; same-seed
  STS-off control is `0/240`; exact semantic key replay count is `0`;
  prototype pass count is `0`; public-data leakage hits are `0`; external
  inference calls are `0`; and
  `reports/private_unseen_transfer_challenge_v1_learned_distillation_gate.json`
  is `GREEN` with exact private-train normalized body overlap `0.0`.
- This is generated private ecology pressure, not promotion evidence and not a
  new public score. The ASI-relevant blocker is now public/general transfer:
  private learned-token evidence is green, but the already-spent wide public
  calibration remains `34/160 = 0.2125` and public readiness remains locked.

Post-v4 Private Shadow Transfer v1 is the current private-only follow-up for
the remaining public/general-transfer blocker:

Current governor note: the post-v4 shadow learned-maturity gate is current
under the latest decoder source/release. The scale cap has been reached, so the
governor no longer queues another private shadow refresh; it now stops at the
locked operator-review public-calibration item.

- `reports/post_v4_private_shadow_transfer_v1.json`: `GREEN`. The generator
  wrote `12000` private train rows and `2400` private heldout rows across `6`
  abstract residual families and `24` private categories. It uses public
  calibration only as aggregate residual-summary context and does not copy
  public prompts, tests, solutions, traces, score labels, task IDs, or benchmark
  names into generated rows. Public-data leakage hits are `0`, private solution
  failures are `0`, and external inference calls are `0`.
- `reports/post_v4_private_shadow_transfer_v1_smoke160_score.json`: `GREEN`.
  STS-on private shadow heldout passes `2400/2400 = 1.0`; same-seed STS-off
  control passes `0/2400`, so STS delta is `+1.0` with `0` regressions,
  no-admissible task rate `0.0`, public tests/solutions used `false`, and
  external inference calls `0`.
- `reports/post_v4_private_shadow_transfer_v1_smoke160_learned_distillation_gate.json`:
  `GREEN`. Learned-only private shadow heldout also passes `2400/2400 = 1.0`;
  learned-token pass count is `2400`, prototype pass count is `0`, prototype rows
  are `0`, verifier-admissible pass count is `2400`, public tests/solutions used
  are `false`, external inference calls are `0`, and decoder source/release
  freshness is proven.
- This clears the immediate post-v4 private shadow prototype-dependency check.
  It does not change the public score and does not unlock public calibration by
  itself. The next ASI-relevant wall remains broad public transfer from these
  learned private behaviors into unseen real tasks, under the existing
  operator-locked public-calibration protocol.
- `reports/post_v4_generalization_autopilot_v1.json`: `GREEN`. The new
  private-only autopilot runner executes the full post-v4 shadow loop without
  Codex remembering the phase order: preflight, curriculum regeneration,
  private-safe STS stream generation, STS-on fanout, same-seed STS-off control,
  scoring, learned-only gate, heartbeat, ledger, and queue. The latest ratchet
  executed run took `388.063s`, regenerated `12000` private train rows and
  `2400` private heldout rows at the configured cap, passed all gates, wrote
  `reports/post_v4_generalization_autopilot_v1_ledger.jsonl`,
  `reports/post_v4_generalization_autopilot_v1_heartbeat.json`, and
  `reports/post_v4_generalization_autopilot_v1_queue.jsonl`, and kept
  `public_calibration_allowed=false`.
- The runner now archives scale-specific artifacts before the canonical files
  are overwritten. Current archived proofs include
  `reports/post_v4_generalization_autopilot_v1_archive/scale480_post_v4_autopilot_1780761066`,
  `reports/post_v4_generalization_autopilot_v1_archive/scale960_post_v4_autopilot_1780762013`,
  `reports/post_v4_generalization_autopilot_v1_archive/scale1440_post_v4_autopilot_1780763353`,
  `reports/post_v4_generalization_autopilot_v1_archive/scale1920_post_v4_autopilot_1780766779`,
  and
  `reports/post_v4_generalization_autopilot_v1_archive/scale2400_post_v4_autopilot_1780767493`;
  all archive manifests report `0` copy failures.
- `reports/post_v4_generalization_autopilot_v1_scaling_profile.json`: `GREEN`.
  The profile now carries measured scale timing across the `480`, `960`,
  `1440`, `1920`, and `2400` heldout runs. The latest `2400` run spent
  `370.141s` total in fanout, including `279.303s` in STS-on fanout and
  `90.838s` in STS-off control. `scale_cap_reached=true`, so the queue no
  longer schedules a repeat `2400 -> 2400` scale run. This
  confirms the beam-precompute policy reopened private scaling: the older
  `1440` run spent `1721.936s` total in fanout before the policy fix.
- The autopilot queue now records `private_shadow_scale_cap_reached` first, then
  a no-execute public readiness refresh, then operator-review-only public
  readiness. This is progress toward a self-sufficient private improvement loop,
  but it remains private evidence. The public wide score is still
  `34/160 = 0.2125`, and public calibration remains locked.
- `reports/theseus_generalization_governor_v1.json`: `YELLOW`. The new
  private-only generalization governor consolidates broad-private, v4,
  post-v4 shadow, v5 private ecology, architecture guidance, architecture
  experiment governance, teacher preflight, causal architecture-delta evidence,
  private semantic-alias transfer, private novel-composition transfer, private
  residual-frontier transfer, agent-lane tool-use/conversation transfer, and
  student-first evidence hygiene without running public calibration. Hard
  safety gates are clear: public lock active, `public_calibration_allowed=false`,
  forbidden post-v4 public artifacts absent, no public tests/solutions used for
  training, and external inference calls `0`. It now recognizes the safe
  downstream prerequisites as complete: architecture guidance is `GREEN`, the
  recommended bounded experiment is `causal_public_code_transfer_router_delta`,
  broad-private learned pass-path structural maturity is current (`24`
  normalized AST families, `24` AST shapes, top duplicate rate `0.041667`, and
  control-structure coverage ready) and exact train-overlap rates are `0.0`,
  teacher preflight hard gates are clear, causal architecture delta is `GREEN`
  with `best_target_delta=0.96875`, `public_task_count=0`, private semantic
  test delta `+0.666667`, `4` positive semantic families, and `0` semantic
  family regressions, refreshed
  semantic-alias learned-only transfer is full-size at `1008/1008 = 1.0` with
  `0`
  prototype/diagnostic-adapter passes, novel-composition transfer is `GREEN`
  with composition-only `1008/1008 = 1.0` and `0`
  prototype/diagnostic-adapter passes, and private unseen-transfer learned
  maturity is now `GREEN` with learned-only `360/360 = 1.0`, exact semantic
  replay `0`, prototype passes `0`, and normalized train-body overlap `0.0`.
  Private residual-frontier transfer is now tougher and still `GREEN` at
  full-pass `1040/1040 = 1.0` across `26` private residual specs, with
  frontier-token passes `1000/1040 = 0.961538`, STS-off control
  `60/1040 = 0.057692`, `0` prototype/diagnostic-adapter passes, public
  candidate sidecars empty, and external inference calls `0`. Contract-blind
  private transfer is also `GREEN`: `240` heldout rows have semantic names
  withheld, strict learned-only transfer passes `240/240 = 1.0`, STS-off
  control is `0/240`, prototype and diagnostic-adapter pass counts are both `0`,
  body-memory replay candidate rows are `0`, and external inference calls are
  `0`. The governor now also requires the contract-blind learned-maturity gate:
  decoder source/release freshness is true, pass-path diversity is `36` AST
  shapes and `36` unique normalized ASTs, top duplicate rate is `0.029167`,
  control-structure coverage is ready, and exact normalized AST/body train
  overlap is `0.0`. Learned-token pass count total is now `8752`.
  The v4, post-v4 shadow, full v5, and unseen-transfer private learned lanes
  are all maturity-green with source/release freshness and normalized
  train-body overlap `0.0`. The agent-lane core is now transfer-ready too:
  tool-use cases `64`, cross-domain capsules `42`, STS named consumer effect
  `true`, hard-v4 conversation graduated `true`, RL policy learning ready
  through `local_synthetic_rl`, and no remaining agent-lane blockers. The
  governor remains `YELLOW`
  because the public score is still below floor, not because a private
  learned-maturity lane is stale.
- `reports/private_residual_self_improvement_ratchet_v1.json`: `YELLOW` by
  decision, with hard gates clear. It reads only aggregate public residual
  summaries, keeps `public_calibration_allowed=false`, confirms the operator
  lock and teacher proposal-only boundary, and emits `retry_private` because
  the spent public score remains `34/160 = 0.2125`. It has now executed
  `run_private_residual_shadow_autopilot`,
  `refresh_semantic_alias_transfer_gate`,
  `refresh_novel_composition_transfer_gate`, `refresh_private_ecology_v5`,
  `teacher_preflight_proposal_only_no_live`, and
  `refresh_generalization_governor` successfully. All `6` ratchet queue items
  are now `completed`; pending queue item count is `0`.
- `reports/private_residual_frontier_v1.json`: `GREEN`. This private-only
  frontier gate converts the aggregate public residual families
  `verifier_mismatch`, `no_admissible_candidate_regression`, `return_shape`, and
  `algorithmic_planning` into `26` harder private composition specs and `1040`
  heldout rows. The current run uses sharded fanout (`10` shards of `104` rows)
  so unattended execution writes per-shard reports before aggregate scoring.
  Full STS-on scoring passes `1040/1040 = 1.0`, frontier-token candidates pass
  `1000/1040 = 0.961538`, STS-off control is `60/1040 = 0.057692`,
  no-admissible rate is `0`, diagnostic-adapter passes are `0`, prototype
  passes are `0`, public candidate manifests are empty, and external inference
  calls are `0`. The expanded spec set now includes stdin-prefix query parsing,
  signed-int parsing, top-k frequency, max-non-adjacent-sum, windowed-delta
  stats, table projection/safe-head shapes, graph components, shortest hops,
  string-DP LCS length, balanced parentheses, and record grouping.
  A one-step learned-token composition bug was fixed so single-step graph and
  grouping specs are not duplicated into invalid `*_then_*` modes.
- The governor queue still stops before public review. There are no current
  safe private refresh actions left in the governor queue after the agent-lane
  tool-use/RL/capsule/conversation refreshes; the single governor item is
  `operator_review_public_calibration_locked`, with an empty command and
  `requires_operator_public_unlock=true`. The newer pre-public audit now records
  `frontier_expander_decision=no_private_frontier_action_remaining` and
  `frontier_expander_next_safe_private_action=""`, so it queues only the locked
  public-review item. Public calibration still requires explicit operator
  approval and is not executed by the governor, pre-public audit, frontier
  expander, ratchet, residual-frontier gate, or agent-lane gates.
- `reports/code_lm_closure_rust_post_v4_runtime_breakdown_smoke4_fanout.json`:
  `GREEN`. This 4-task private-only fanout smoke proves the Rust report now
  emits `candidate_fanout_runtime_breakdown`, separating shared precompute wall
  time from per-task generation wall time. It emitted `16` private candidates
  and `0` public candidates, kept public calibration locked, and fixed the
  timing category accounting so shared precompute is not summed as per-task
  verifier/cache work.
- `reports/post_v4_fanout_precompute_ablation_v1.json`: `GREEN`. The 64-task
  private-only A/B probe compared default batched beam precompute against
  `THESEUS_CODE_LM_BATCHED_BEAM_CACHE=0`. Both variants scored `64/64 = 1.0`
  with `0` public candidates and `0` external inference calls, while beam-off
  reduced fanout runtime from `32049ms` to `8496ms` (`0.734906` runtime delta
  rate). The autopilot now treats this as private runtime-policy evidence only:
  the next private scale probe may disable batched beam precompute, but public
  calibration remains locked and the public wide score remains `34/160`.

Private residual repair v3 has since advanced past the first post-calibration
repair result:

- `reports/private_residual_v3_student_repair_loop.json`: `GREEN`. The
  private-only structural/token student repair emits `960` STS-on private
  candidates and `960` matched non-STS structural controls from `960` private
  train rows and `240` private heldout rows, with candidate budget `4` per
  task. Heldout tests and heldout solution bodies are not used for generation.
- `reports/private_residual_repair_v3_heldout_score.json`: `YELLOW` because
  the equal-budget non-STS oracle/pass-if-any control now passes `240/240`,
  making the old route-withheld STS delta `0.0`. Adapter-off learned/student
  candidates still pass `240/240 = 1.0`, structural-action candidates pass
  `240/240 = 1.0`, diagnostic adapters pass `0/240 = 0.0`, no-admissible rate
  is `0.0`, fallback-return candidates are `0`, and LiveCodeBench-private stdin
  proxy passes are `48/48`.
- `reports/private_residual_v3_sts_ablation.json`: `GREEN`. STS selected pass
  rate is `240/240 = 1.0`; matched non-STS selected pass rate is
  `184/240 = 0.766667`; both oracle/pass-if-any rates are `240/240 = 1.0`.
  The measured STS lift is selection/ranking lift (`+0.233333`), not emission
  coverage lift (`0.0`).
- Evidence remains private-only: public candidate sidecar rows are `0`, public
  tests and solutions are not used, teacher rows are not used, and external
  inference calls are `0`.
- Promotion/model growth remain blocked until public-transfer, candidate
  coverage freshness, STS-consumer, runtime, and hardware/governance evidence
  clear under their own gates. A new public calibration surface may only be
  proposed after operator review and must relock immediately.
- The current broad-floor recovery gate is stricter than the older aggregate
  lift reading. `reports/broad_public_code_transfer_floor_recovery.json` is now
  `GREEN` as private recovery evidence: the same-seed private ablation has
  aggregate semantic lift `+0.25`, all target semantic families are positive,
  and no target family regresses. Family deltas are algorithm choice `+1/4`,
  edge contracts `+2/8`, local adapter/runtime `+1/8`, and type/return-shape
  `+2/4`. The source-side fix routes exact private receiver families for
  signed-int parsing, run-length pairs, label/count mappings, top-k frequency,
  and prime-loop predicates ahead of generic interface bodies; these remain
  grammar-masked learned-token bridge candidates with `expression_memory_fallback=false`
  and `sts_candidate_expression_used=false`. This is private evidence only:
  public calibration remains locked, no teacher is used, no public prompts/tests/
  solutions enter training, and fallback returns remain forbidden. The maturity
  audit is still `YELLOW` because the actual broad public transfer floor remains
  uncleared; the next accepted step is a private train-once closure and rerun of
  decoder/transfer gates before any future public-calibration review.
- That private train-once step has now run without spending public calibration:
  `reports/code_lm_train_once_fanout_private_broad_floor_transfer_repair_closure_v15_private_only.json`
  is `GREEN` with `private_only=true`, `public_calibration_allowed=false`,
  `external_inference_calls=0`, a one-checkpoint fanout architecture, and
  `repeated_training_per_candidate_shard=false`. It trained from the current
  durable `445`-row broad-floor recovery private file plus the existing
  promotion-safe private row sources, then fanned out over `237` private eval
  tasks and emitted `342` private candidate rows. The intentionally empty public
  sidecar emitted `0` public tasks and `0` public candidates. The follow-up
  `reports/decoder_v2_private_ablation_gate_private_broad_floor_transfer_repair_closure_v15_private_only.json`
  and
  `reports/private_public_transfer_proof_private_broad_floor_transfer_repair_closure_v15_private_only.json`
  are both correctly `YELLOW`: they prove the private closure is current, but
  cannot unlock public transfer because public candidate coverage was not run.
  The decoder gate produced `16` no-admissible decoder-control policy rows, now
  persisted under `data/training_data/high_transfer/private_train/` for the next
  private closure. These rows are policy-only coverage pressure, not answer-body
	  training rows; they carry `external_inference_calls=0`,
	  `public_benchmark_training_data_used=false`, and
	  `raw_public_prompt_or_tests_copied=false`.

- The private-only STS blocker from the v15 closure is fixed without spending
  public calibration. `scripts/code_lm_sts_conditioning.py` now allows native STS
  conditioning when the public sidecar is empty but private eval STS rows exist,
  and `scripts/code_lm_train_once_fanout.py` now requires a real private STS
  generation path when default-on STS is requested in private-only mode. The
  follow-up v16 run
  `reports/code_lm_train_once_fanout_private_broad_floor_transfer_repair_closure_v16_private_sts.json`
  is `GREEN`: `private_only=true`, `public_calibration_allowed=false`,
  `external_inference_calls=0`, `sts_conditioning_used=true`, `1071` private
  candidate rows, `0` public candidates, `410` STS-conditioned private
  candidates in the manifest diagnostics, `1051` verifier-passing candidates,
  and `0` template-like candidates. The decoder gate
  `reports/decoder_v2_private_ablation_gate_private_broad_floor_transfer_repair_closure_v16_private_sts.json`
  remains correctly `YELLOW` because public candidate coverage is still locked,
  but the prior private STS gates now pass: STS-conditioned skeleton observed
  with `34` candidates across `5` tasks and verifier pass rate `1.0`, and
  STS-conditioned non-regression also passes. The transfer proof
  `reports/private_public_transfer_proof_private_broad_floor_transfer_repair_closure_v16_private_sts.json`
  is still correctly `YELLOW`: contract-guided candidate inventory lifts by
  `+423` and STS-conditioned inventory lifts by `+34`, but public-surface
  coverage/provenance/program-synthesis gates remain failed because the current
  run intentionally has no public surface. Do not treat v16 as a public-transfer
  score or promotion unlock.

- The refreshed governance state remains no-cheat. `reports/external_inference_audit.json`
  has `total_violations=0`; `reports/maturity_integrity_audit.json` is still
  `YELLOW` with `0` hard blockers and `2` maturity blockers:
  `public_transfer_floor_cleared` is false with the last broad public pass rate
  still `34/160 = 0.2125`, and `candidate_coverage_transfer_gate_is_fresh` is
  false because the current v16 run is private-only with `0` public candidates;
  `reports/candidate_promotion_gate.json` keeps promotion blocked. Mac-native
  status remains useful but not finished: `reports/macos_mlx_parity_audit.json`
  shows MLX available and implemented MLX bridge evidence for registered worker
  chunks and CLI bridges; it now records `2` guarded Rust/Metal proof-ready
  hot-loop targets, `2` guarded Rust/Metal canary-ready hot-loop targets, and
  `0` pending native ports, while `native_hot_loop_parity_claim_allowed=false`
  and `production_route_pending_count=4`. Dogfood remains consent-gated:
  `reports/dogfood_trace_training_bridge.json` writes `0` rows with the local
  fail-closed config present, capture/training consent disabled, and no
  accepted/missed/ignored events.

Broad Private Generalization Ladder v1 is now the current broad private repair
result:

- `reports/broad_private_generalization_ladder_v1.json`: `GREEN`. The private
  generator wrote `3000` train rows and `1008` heldout rows across `12`
  families and `24` categories, with private solution self-tests passing,
  public-data leakage hits `0`, and external inference calls `0`.
- `reports/broad_private_generalization_unattended_v1.json`: `GREEN` with
  `completion_evidence_status=green_transfer`. The latest unattended run is now
  the full `1008`-task refresh under the current decoder/release: it passed
  preflight, kept public calibration locked, generated private-safe task STS
  streams, ran STS-on fanout and same-seed STS-off control, wrote heartbeat and
  ledger artifacts, and completed the broad private transfer gate in
  `570.937s` (`387.978s` STS-on fanout, `180.750s` STS-off control).
- `reports/broad_private_generalization_score_v1.json`: `GREEN` and current.
  The refreshed full score is `1008/1008` (`1.0`) through
  private-train-induced broad semantic token decoder candidates from `3696`
  candidate rows. Public tests/solutions were not used and external inference
  calls are `0`.
- `reports/broad_private_generalization_gate_v1.json`: `GREEN` and current
  under the stricter source freshness rule. Decoder source/release freshness is
  clean, broad private pass rate is `1.0`, STS delta is `+1.0`, and no
  admissible-task rate is `0.0`.
- `reports/broad_private_learned_distillation_gate_v1.json`: `GREEN`.
  The gate now writes and scores a strict novel learned-only manifest
  (`reports/code_lm_private_candidates_broad_private_generalization_ladder_v1_heldout_strict_novel_learned_only.jsonl`)
  that excludes exact private train solution-body replay. Learned-only and
  strict novel learned-only both score `1008/1008 = 1.0`, strict-novel candidate
  rows are `3360`, exact train-body memory pass count is `0`, prototype pass
  count is `0`, pass AST shape count is `24`, pass normalized AST unique count
  is `24`, top duplicate rate is `0.041667`, train body overlap is `0.0`, and
  decoder source/release freshness is clean. This closes the stale replay
  blocker for the broad-private lane, while still leaving public transfer below
  floor.
- `reports/broad_private_semantic_alias_gate_v1.json`: `GREEN`.
  The private semantic-alias stress gate now rewrites all `1008` private heldout rows
  across all `24` broad-private categories so exact semantic-family keys cannot
  be reused. Learned-only alias candidates pass `1008/1008` (`1.0`), inferred
  token passes are `1008`, candidate rows are `3696`, learned-only candidate
  rows are `3360`, diagnostic-adapter passes are `0`, prototype passes are `0`,
  exact semantic key reuse is `0`, public candidate sidecars are empty, and
  external inference calls are `0`. This proves the current decoder can transfer
  across full broad-private semantic renaming, but it is still private-only
  evidence and does not unlock public calibration by itself.
- `reports/broad_private_novel_composition_gate_v1.json`: `GREEN`.
  The private novel-composition gate was refreshed after the decoder
  argument-role fingerprint repair. It writes a full-scale `1008` private
  heldout rows across `6` two-step composition specs. It requires the decoder
  to compose reusable private-train token bodies instead of picking one exact
  semantic family. Composition-only candidates pass `1008/1008` (`1.0`),
  composition token pass count is `1008`, candidate rows are `4032`,
  composition candidate rows are `1008`, diagnostic-adapter passes are `0`,
  prototype passes are `0`, public candidate sidecars are empty, and external
  inference calls are `0`. The refreshed run took `550.334s`, with `375.780s`
  in STS-on fanout and `168.682s` in STS-off control. This is private-only
  transfer evidence; it does not unlock public calibration by itself.
- The refreshed broad ladder is now current against the decoder source and
  release binary. Fanout timings from the latest full run were `77.689s`
  STS-on and `27.078s` STS-off control, so the earlier stale-runtime concern is
  no longer the active blocker for this lane.
- `reports/broad_private_train_prototype_decoder_v1.json`: `GREEN`.
  This older follow-up remains useful historical/private diagnostic evidence:
  it induced `24` semantic prototypes from the `3000` private train rows,
  verified `0` private-train prototype self-test failures, emitted `1008`
  heldout candidates, and explicitly excluded the hardcoded broad-private
  diagnostic adapter mode.
- `reports/broad_private_train_prototype_score_v1.json`: `GREEN`.
  The private-train prototype candidates pass `1008/1008` heldout tasks with
  no-admissible rate `0.0`; the STS-off control passes `0/1008`, so STS delta
  is `+1.0` with `0` regressions. Public-data leakage hits are `0`.
- `reports/broad_private_train_prototype_gate_v1.json`: `GREEN`. This is still
  private-only prototype evidence, not public promotion evidence. The current
  repair has moved the broad-private pass path into learned-token rows, but it
  remains private synthetic broad-ladder evidence. Public calibration/model
  growth stay locked until readiness evidence is reviewed and an operator
  explicitly approves one bounded public run.
- `reports/post_distillation_public_transfer_readiness_v1.json`: `YELLOW`.
  The post-distillation integrity checks are clean: learned-only private pass
  rate is `1.0`, prototype pass count is `0`, public boundary is clean,
  external inference calls are `0`, and the public calibration operator lock is
  active. The report remains `YELLOW` because the latest already-spent wide
  public score is still `34/160 = 0.2125`, below the `0.70` floor, and all five
  public cards remain below floor.
- `reports/public_calibration_readiness_packet.json`: `GREEN` in
  `post_distillation_v4_operator_review` mode. The packet now accepts the
  current pre-public audit (`operator_review_ready=true`,
  `learned_token_pass_count_total=8752`) and exhausted frontier expander
  (`no_private_frontier_action_remaining`) as the operator-review readiness
  source. The already-spent public pass rate is still `0.2125`, below the
  `0.70` floor, with all five public cards below floor, so this is not promotion
  evidence. The pinned `5x32` public surface is still `160` task IDs with no
  leak hits, the operator lock is active, `public_calibration_allowed=false`,
  and the packet explicitly does not run public calibration.
- `reports/operator_bounded_public_calibration_dry_run.json`: `GREEN`.
  The guarded runner dry-run verifies the exact proposed one-shot public
  calibration command shape, confirms the operator lock is present before and
  after the dry-run, confirms the proposed output artifact is absent, and does
  not execute public calibration. Approval remains missing by design; an
  explicit operator approval file is required before `--execute` can spend one
  bounded public run.
- `reports/public_safe_broad_transfer_maturity_v4.json`: `GREEN`.
  The v4 public-safe maturity generator now writes `3000` private train rows
  and `1008` private heldout rows across `6` maturity families and `24`
  categories, with private solution tests passing, public-data leakage hits
  `0`, and external inference calls `0`. The v4 generator also fixes the lcs
  contract to expose both string arguments.
- `reports/public_safe_broad_transfer_maturity_v4_smoke64_score.json`:
  `GREEN`. The current corrected v4 smoke scores `64/64 = 1.0`; same-seed
  STS-off control scores `0/64`, so STS delta is `+1.0` with `0` regressions
  and no public tests or solutions used. This result required fixing
  four-argument visible-signature inference, preserving the `extra` tuple-rest
  alias for multi-argument generated bodies, preventing the STS control policy
  from creating streams when the control STS file is empty, and adding
  private-safe task STS streams through `scripts/private_task_sts_streams.py`.
- `reports/public_safe_broad_transfer_maturity_v4_smoke64_learned_distillation_gate.json`:
  `GREEN` after the v4 contract-recognition fix. Learned-only smoke now passes
  `64/64 = 1.0`, prototype pass count is `0`, prototype rows are `0`, and all
  passes are learned-token rows.
- `reports/public_safe_broad_transfer_maturity_v4_score.json`: `GREEN`.
  This is green after the full corrected refresh. The full v4 heldout
  passes `1008/1008 = 1.0` across all `6` maturity families and `24`
  categories from `3696` candidate rows. The corrected same-seed STS-off
  control has `1008` rows across `1008` tasks, `0` STS-conditioned rows, `0`
  decoder-control-policy applications, and scores `0/1008`, so STS delta is
  `+1.0` with `0` regressions. No public tests, public solutions, or external
  inference are used.
- `reports/public_safe_broad_transfer_maturity_v4_learned_distillation_gate.json`:
  `GREEN`. Learned-only v4 heldout also passes `1008/1008 = 1.0`; learned-token
  pass count is `1008`, learned-only candidate rows are `3444`, prototype pass
  count is `0`, prototype rows are `0`, verifier-admissible pass count is
  `1008`, public tests/solutions used are `false`, external inference calls are
  `0`, and decoder source/release freshness is proven for the v4 score and
  learned-only artifacts. The current blocker is no longer private v4
  prototype dependency or saturated same-seed control; it is deciding whether to
  spend exactly one operator-approved bounded public calibration, then proving
  broad public transfer on that locked surface.
- `reports/edge_contract_v3_verifier_mismatch_public_transfer_heldout_v3_contract_strict_syntaxrepair_full192_broad_score.json`:
  `GREEN`. The refreshed v3 private-public-transfer heldout scores
  `192/192 = 1.0` across all six families, with no-admissible task rate `0.0`,
  regressions `0`, public tests/solutions used `false`, and external inference
  calls `0`. The same-seed STS-off control passes `126/192 = 0.65625`, so STS
  delta is `+0.34375`. All pass modes are strict-novel STS-conditioned token
  decoder modes, not diagnostic adapters.
- `reports/edge_contract_v3_verifier_mismatch_public_transfer_v3_contract_strict_syntaxrepair_full192_broad_learned_distillation_gate.json`:
  `GREEN`. Learned-only and strict-novel learned-only candidates score
  `192/192 = 1.0`, learned-token pass count is `192`, learned-only candidate
  rows are `274`, exact train-body memory candidate rows are `0`, exact
  train-body memory pass count is `0`, prototype pass count is `0`, diagnostic
  adapter pass count is `0`, public tests/solutions used are `false`, external
  inference calls are `0`, and source/release freshness is proven against the
  current decoder. The pass path has `14` pass AST shapes and `14` unique
  normalized pass ASTs. The current ASI-relevant blocker is no longer v3
  adapter/prototype dependency; it is whether this private learned-token stack
  transfers to the still-locked broad public surface.
- `reports/edge_contract_v2_curriculum_contract_repair.json`: `GREEN`.
  The edge-contract-v2 private curriculum hygiene blocker is cleared: the
  repaired private artifact preserves all `240` rows, fills decoder generation
  plans from `213/240` to `240/240`, repairs `27` row contracts, and reports
  `0` unsafe public-training rows. The repaired artifact is
  `data/training_data/high_transfer/private_train/edge_contract_v2_private_residual_curriculum_repaired_code_lm_tasks.jsonl`.
- `reports/macos_sandbox_path_regression.json`: `GREEN`. The Mac/Linux
  private sandbox path now uses the repo-local `reports/tmp` root for the
  benchmark runtime helper, Code LM closure helper, and private candidate
  verifier helper. The regression passed a private synthetic candidate with
  runtime load and intended behavior both true, `0` Windows temp-path leaks,
  `0` public tests/solutions, and `0` external inference calls.
- `reports/code_lm_closure_edge_contract_v2_private_rescore.json`: `GREEN`.
  The STS/ranker nonregression blocker for the edge-v2 private gate is cleared
  in the generated candidate artifact. The patched Rust closure preserves
  same-seed STS-off candidates as labeled
  `sts_nonregression_union_candidate` fallbacks in the `private_eval` phase.
  The rebuilt artifact has `870` private candidate rows, including `224`
  production STS nonregression fallback rows. Independent private rescore sees
  baseline `1/149 = 0.006711`, STS-off `37/149 = 0.248322`, trained
  `41/149 = 0.275168`, private delta `+0.268457`, STS repair delta
  `+0.026846`, `4` STS improvements, and `0` STS regressions. Runtime-load
  rate is `0.987578`; public benchmarks were not scored, public tests and
  solutions were not used, and `external_inference_calls=0`.
- `reports/edge_contract_v2_private_closure_runner.json`: `GREEN`. The
  bounded closure reran cleanly against the repaired `240`-row curriculum in
  `1138.046s`; the Rust closure report, verifier report, and runner are all
  GREEN. The private verifier reports `candidate_rows=870`,
  `verifier_rows=870`, `verifier_pass_rows=818`, closure private delta
  `+0.268457`, and `sts_regressions=0`. This is held-out private readiness
  evidence only; the already-spent `34/160` public wide calibration remains the
  latest public score until one explicit new bounded public calibration is run.

Current wide-slice result:

- `reports/public_wide_slice_manifest_seed23_5x32.jsonl`: fixed 160-task
  calibration surface, `32` tasks each from MBPP, EvalPlus, BigCodeBench,
  HumanEval, and LiveCodeBench. This manifest remains metadata-only for
  private training; public prompts/tests/solutions stay scorer-only.
- `reports/decoder_v2_private_ablation_gate_wide_slice_interface_floor_refresh_v1.json`:
  `GREEN` after the interface-floor decoder refresh. Public metadata coverage
  is `0.8875`, no-admissible rate is `0.08125` (`13/160`), and candidate
  quality gate pass rate is `0.990921`.
- `reports/private_public_transfer_proof_wide_slice_interface_floor_refresh_v1.json`:
  `GREEN`. The repair added `+82` actual token candidates, raised actual and
  eligible public metadata coverage by `+0.24375`, and reduced no-admissible
  rate by `-0.24375`.
- `reports/sts_causal_decoder_ablation_wide_slice_interface_floor_refresh_v1.json`:
  `GREEN`. STS-conditioned public eligible coverage is `0.91875` versus
  `0.45` for non-STS learned-token candidates, with no pass-rate regression.
- `reports/public_calibration_readiness_packet_wide_slice_interface_floor_refresh_v1.json`:
  technically ready but operator-locked because the bounded public calibration
  was already consumed.
- `reports/real_code_benchmark_graduation_wide_public_seed23_5x32_interface_floor_v1.json`:
  one public calibration spent. Result is `34/160` (`0.2125`), with
  single-stream baseline `26/160` (`0.1625`), STS/multi-stream delta `+0.05`,
  `8` improvements, `0` regressions, `830` student candidates, `771`
  benchmark-promotion-eligible learned-token candidates, `0` template/loop
  fallback candidates, and `external_inference_calls=0`.
- Per-card wide score: MBPP `3/32`, EvalPlus `4/32`, BigCodeBench `21/32`,
  HumanEval `6/32`, LiveCodeBench `0/32`. BigCodeBench transfer is the only
  comparatively strong card; LiveCodeBench is the clearest zero-shot gap.
- `reports/public_code_transfer_residual_report_wide_public_seed23_5x32_interface_floor_v1.json`:
  historically named the next private repair family as
  `edge_contract_v2_private_residual_curriculum`. That family is now
  superseded by fresh completed v4 maturity evidence, so the
  post-distillation readiness gate advances the next target to an
  `operator_reviewed_bounded_public_calibration_packet`.
  Dominant residuals are verifier mismatch `89`, adapter-adjusted true
  no-admissible `13`, return-shape `11`, and algorithmic planning `8`.
- `reports/maturity_integrity_audit_wide_public_seed23_5x32_interface_floor_v1.json`:
  `YELLOW` with no hard or evidence blockers. The remaining maturity blocker
  is the public transfer floor: `0.2125 < 0.70`.
- Promotion and model growth remain locked. The next goal is private-only
  edge-contract-v4/public-safe broad transfer maturity work against the still
  below-floor public cards, especially LiveCodeBench-style stdin/competitive
  programming, MBPP/EvalPlus function contracts, and cross-card algorithmic
  planning. Only after private decoder, transfer, STS, readiness, and maturity
  gates are GREEN should a new bounded public calibration surface be proposed.

- The pinned wide selector is
  `scripts/wide_public_slice_selector.py`; the current manifest/report are
  `reports/public_wide_slice_manifest_seed23_5x32.jsonl` and
  `reports/public_wide_slice_selector_seed23_5x32.json`. The selector is
  `GREEN`: `160/160` tasks, `32` each from MBPP, EvalPlus, BigCodeBench,
  HumanEval, and LiveCodeBench. The manifest exports task IDs, entry points,
  and coarse feature buckets only; it does not export public prompts, tests,
  reference solutions, candidate code, or score labels into private training.
- The public scorer and fanout path now accept `--case-manifest` so the
  selected wide slice stays aligned across visible prompt export, token
  candidate generation, closure scoring, train-once fanout, and broad-transfer
  score-existing runs. A scorer run blocks if a manifest-selected task ID is
  not staged locally, preventing false 160-task coverage claims.
- LiveCodeBench is staged locally under
  `resource_pantry/git/livecodebench` from the official repository plus the
  current `code_generation_lite` lite JSONL payload. The loader supports both
  functional tasks and stdin-style LiveCodeBench tasks by exposing stdin tasks
  as `solve(input_data)` for sandbox calibration. Public tests remain
  scorer-only and are not exported to candidate generation.
- The wide-slice command sequence was private-first:
  `python3 scripts/wide_public_slice_selector.py --cases-per-card 32 --seed 23`;
  then private fanout/decoder/transfer/STS/maturity gates; then exactly one
  wide public calibration with
  `--case-manifest reports/public_wide_slice_manifest_seed23_5x32.jsonl`.
  That calibration is now relocked. Public benchmark data remains
  calibration-only and must not become training rows.
- `reports/code_lm_train_once_fanout.json`: `GREEN`, completed, with a
  broad metadata-only four-card public surface: `32` visible tasks and `189`
  student candidates. The fanout uses no public tests, no public solutions, no
  benchmark answers, no teacher apply, and `external_inference_calls=0`. STS
  conditioning is default-on; STS-off is preserved only as a same-seed
  control/ablation lane.
- `scripts/targeted_private_residual_curriculum_v2.py`: materializes the
  private-only residual curriculum families `edge_contract_v2`,
  `candidate_floor_adapter_v2`, `return_type_shape_v2`, and
  `parsing_encoding_v1`. The 2026-06-05 run wrote `960` private rows, all with
  private solution tests passing and with public prompts/tests/solutions kept
  out of the rows. These rows are generated local training artifacts, not
  public benchmark data.
- `reports/decoder_v2_private_ablation_gate.json`: `GREEN` and would unlock one
  bounded public calibration in a fresh cycle. The current cycle's public
  calibration was already spent and must not be rerun just to fish. Public
  metadata candidate coverage is complete:
  `public_task_count=32`, actual/eligible token task coverage `0.96875`,
  no-admissible rate `0.0`, and promotion-facing public candidate quality
  `1.0`.
- `reports/private_public_transfer_proof.json`: `GREEN`. The private-to-public
  receiver bridge now has broad metadata transfer evidence with actual and
  eligible coverage deltas of `+0.96875`, program-synthesis loop rate `1.0`,
  bridged same-surface no-admissible evidence, no public no-admissible tasks,
  and STS-conditioned candidate inventory delta `+253`.
- `reports/sts_causal_decoder_ablation.json`: `GREEN`. Same-seed STS-on versus
  STS-off evidence shows STS changes candidate quality/coverage: STS-on
  public task coverage is `1.0` across the 32 metadata tasks, same-seed STS-off
  coverage is `0.96875`, delta `+0.03125`, with no pass-rate regression. Five
  STS decoder-control rows were materialized.
- Current source repair recovered the visible parser/adapter/recurrence
  receiver path without another public score run: `parse_music` now emits
  `parsing_encoding_symbol_beat_parser`, rescale emits
  `contract_rescale_to_unit`, spelled-number sorting emits
  `contract_sort_number_words`, sort-by-second emits
  `contract_sort_pairs_second_then_first`, archive/process execution adapters
  use first-class execution-shape names, and Bell number metadata emits
  `contract_bell_number_table`.
- `reports/broad_transfer_residual_decoder_ablation_v2_private_visible_intent_repair_task24.json`:
  `GREEN`. The visible-intent receiver/ranker repair remains private-only:
  `24` private heldout tasks, `0` public tasks/candidates, semantic test
  pass-rate delta `+0.375`, no-admissible delta `0.0`, and bridge-shadow
  task-count delta `+15`. This is source-side decoder evidence, not public
  score evidence.
- `reports/causal_architecture_delta_loop.json`: `GREEN`. The bounded private
  same-seed architecture delta now uses a `24`-task private heldout proof. It
  shows private heldout pass-rate delta `+0.208333`, no-admissible-rate delta
  `-0.208333`, private receiver eligibility delta `+0.291667`, and private
  semantic test delta `+0.666667` across `4` positive semantic families with
  `0` semantic family regressions. It uses `0` public tasks and no public
  tests/solutions.
- `reports/broad_transfer_closure_runner_source_mbpp_source_evalplus_source_bigcodebench_source_human_eval_seed14_8_after_visible_receiver_repair.json`:
  `GREEN` for the single bounded score-existing public calibration spent after
  the visible parser/adapter/recurrence repair. MBPP, EvalPlus, BigCodeBench,
  and HumanEval each scored 8 tasks. Public pass rate is now `0.84375`
  (`27/32`), up from the prior `0.4375` (`14/32`). STS/multi-stream adds
  `+0.4375` over the single-stream comparison, with `14` task-level
  improvements, `0` task-level regressions, `0` template candidates, `0`
  loop-closure candidates, and `external_inference_calls=0`. Candidate
  provenance remains clean, but public score evidence is still
  calibration-only.
- `reports/macos_sandbox_path_regression.json`: `GREEN`. It proves the
  candidate scoring sandbox on macOS uses the repo-local
  `reports/tmp` runtime root and does not leak the historical
  `D:/ProjectTheseus/tmp` Windows path into candidate execution.
- `reports/maturity_integrity_audit_after_visible_receiver_repair.json`:
  `YELLOW` only on `public_transfer_floor_cleared`; hard/evidence blockers are
  `0`. This is a maturity/coverage block, not a leak or safety block. The broad
  matrix requires wider transfer evidence than this deliberately bounded
  8-task-per-card calibration.
- `reports/candidate_promotion_gate_after_visible_receiver_repair.json`:
  promotion remains blocked at `24/28`. The active blockers are the older
  general public comparator, runtime/GPU governance evidence, and maturity
  audit GREEN status. This is a correct block: the public calibration is strong,
  but promotion still needs broader per-card coverage and current runtime
  governance evidence.
- `reports/public_transfer_residual_packet_after_visible_receiver_repair.json`
  and
  `reports/public_code_transfer_residual_report_after_visible_receiver_repair.json`
  name the current blocker after the visible repair without embedding public
  prompts/tests/solutions/code. The residual surface shrank to `5` failed
  tasks: raw families `edge_case=2`, `type_handling=1`, `parsing=1`, and
  `local_code_generation_adapter_needed=1`. Sanitized categories are
  `verifier_mismatch=2`, `return_shape=1`, `algorithmic_planning=1`, and one
  adapter-adjusted no-admissible row with an eligible candidate available.
  MBPP is the only below-floor card on this bounded slice (`0.625`); EvalPlus
  and BigCodeBench are `0.875`, and HumanEval is `1.0`.

## 2026-06-03 Mac Transfer Snapshot

The Windows CUDA workstation is the source of truth for this handoff. Git
`main` has been fast-forwarded to the current machine state and is synchronized
with `origin/main`. Start Mac takeover with
`docs/MAC_HANDOFF_2026_06_03.md`.

Current handoff gates:

- coherence: `GREEN`
- candidate promotion: `26/28`
- remaining candidate blockers: `broad_public_code_transfer_ready` and
  `maturity_integrity_audit_green`
- public calibration: locked
- model growth: locked
- active Code LM workers: `0`
- active control-plane leases: `0`
- control-plane next work: `broad_public_transfer_floor_private_repair`

Mac track reviewed on 2026-06-03:

- base Mac source tip before the hardening pass:
  `42437588e89750b44ecdd2e6b5ead822a40866ea`
  (`Add Mac MLX rollout worker lane`)
- `theseus runtime doctor`: `GREEN` after Mac runtime root normalization;
  direct source launches, source checks, installed app checks, and LaunchAgent
  checks now default to
  `~/Library/Application Support/Project Theseus Hive/runtime`, with
  installer-provided cache/report/checkpoint paths persisted from env. The
  doctor summarizes source-vs-installed verified version, update catalog,
  version status, update check-in, license-registration presence, and join-config
  presence without reading local secrets. Active shell Python missing MLX is
  recorded as a false-negative context when the source/app Hive venvs have MLX.
- `hive_node_registry.py`: local Apple MLX is training-eligible; the Windows
  CUDA node remains trusted/known but is currently blocked from this Mac by live
  API reachability, so `remote_task_trust=trusted_but_unreachable`
- `hive_training_orchestrator.py plan --profile smoke`: current Mac evidence
  skips Windows as `unreachable_api`, gives the Mac `mlx_training_chunk`, and
  blocks CUDA arms until `http://10.0.0.147:8791` is reachable again
- prior bounded-work proof completed local MLX eval/train/rollout and Windows
  CUDA eval/train/rollout, but that proof must be rerun after the Windows
  firewall/IP/service issue is cleared before treating two-node training as live
- direct fetch of the three Windows CUDA worker reports succeeded, but indexed
  artifact sync still times out against the old Windows service; the source now
  includes a bounded recent-first artifact indexer that needs to reach Windows
  through the update/package flow
- macOS artifacts are rebuilt by the macOS release gate; use
  `reports/hive_macos_release_gate*.json`, `reports/hive_verified_version.json`,
  and `reports/hive_update_catalog.json` for the exact current commit, Hive
  version, artifact hashes, and catalog state.
- current verified commit/version/update IDs are report-driven; use
  `reports/hive_verified_version.json`, `reports/hive_update_catalog.json`, and
  `reports/hive_version_convergence_after_final_publish.json` as the source of
  truth after each publish
- soft convergence has been proven on both the Mac and Windows nodes without
  hard source/app replacement
- macOS release gate is `ok=true` for private Apple-Silicon canary; broad fleet
  rollout is still held by the physical Intel canary, and public distribution is
  still held by Developer ID notarization
- Hive worker chunks have MLX parity for eval/training/rollout, and
  `symliquid-cli` now exposes `train-standalone-mlx`,
  `train-rollout-mlx`, `train-rollout-mlx-sweep`, and
  `train-token-superposition-mlx` via bounded first-party MLX bridges.
  `reports/macos_mlx_parity_audit.json` remains `YELLOW` only because deeper
  Rust/Metal kernel ports are still pending; it now records live worker/CLI
  evidence counts so static parity and runnable parity are not conflated.
- `reports/macos_mlx_work_proof.json` is the runnable Apple-Silicon proof for
  Mac canaries. It records bounded MLX worker chunk reports, MLX command bridge
  reports, work receipts, ledger growth, and parity-audit state, and is wired
  into `scripts/hive_macos_release_gate.py`.
- `reports/hive_node_registry.json` is now the shared reachability gate for
  schedulers. Trusted remote peers are not enough: remote light work requires
  fresh outbound verification, and remote training requires non-flapping
  outbound verification. This keeps stale Windows CUDA records from receiving
  work while the Mac cannot call `http://10.0.0.147:8791`.

The old private-only Code LM transfer repair is now superseded by the
2026-06-05 broad metadata fanout/decoder/transfer loop above. Public
calibration was unlocked by decoder/transfer gates and then executed once on
MBPP, EvalPlus, BigCodeBench, and HumanEval after the visible receiver repair.
The current bounded score is `0.84375` (`27/32`), with the Mac sandbox path bug
validated fixed. The current wall is no longer candidate coverage; it is broad
transfer maturity: MBPP edge/verifier mismatches, one return-shape failure, one
parsing/encoding failure, one adapter-floor task, thin per-card coverage, and
missing LiveCodeBench coverage.

## Current Identity

Project Theseus is a Rust-first, local-only research implementation of
Ratcheting Modular Intelligence. It combines:

- the SymLiquid compact generative substrate;
- Verified Intent-to-Execution Architecture as the canonical north-star
  system contract;
- a real Code LM learning lane with token-level candidate generation;
- STS / multi-stream parallel token probes;
- benchmark ratcheting with residual escrow and regression preservation;
- Octopus-routed specialist arms;
- SparkStream autonomy, dashboard, checkpoints, watchdogs, and resource gates;
- Theseus Hive local/distributed runtime;
- a governed personality core loaded from local user-owned documents;
- a Reality Manipulator MVP that compiles raw intent into private worlds,
  eight-limb spells, artifact graphs, claims, critiques, compile targets,
  release gates, residuals, primitive candidates, and feedback plans.

It is not a foundation model and it is not a production runtime. It is an
autonomous local research machine with explicit proof, licensing, privacy,
teacher-use, and promotion boundaries.

## Current Snapshot

| Area | Current state |
| --- | --- |
| Primary frontier | Code LM transfer remains the promotion-facing frontier. Conversation, personality, VIEA, Hive, board-game RL, and installer reports are supporting/regression surfaces unless a current gate says otherwise |
| Fanout | `reports/code_lm_train_once_fanout.json` is `GREEN`: 32 metadata-only public tasks, 189 public candidate rows, full manifest coverage, no public no-admissible rows, no public tests/solutions, no teacher apply, `external_inference_calls=0`; STS is default-on |
| Targeted private residual v2 | `scripts/targeted_private_residual_curriculum_v2.py` generated 960 private rows across `edge_contract_v2`, `candidate_floor_adapter_v2`, `return_type_shape_v2`, and `parsing_encoding_v1`; generation gates were GREEN with private solution tests passing and public prompts/tests/solutions excluded |
| Visible-intent repair | `reports/broad_transfer_residual_decoder_ablation_v2_private_visible_intent_repair_task24.json` is `GREEN`: 24 private heldout tasks, semantic pass-rate delta `+0.375`, no-admissible delta `0.0`, no public tasks/candidates, and bridge-shadow task-count delta `+15` |
| Decoder gate | `reports/decoder_v2_private_ablation_gate.json` is `GREEN`: public actual/eligible token task coverage `0.96875`, no-admissible rate `0.0`, candidate quality `1.0`, contract-guided candidates `131`, STS-conditioned candidates `253` |
| Transfer proof | `reports/private_public_transfer_proof.json` is `GREEN`: actual and eligible coverage deltas are `+0.96875`; current public no-admissible rate is `0.0`; the old empty-public baseline is handled by a GREEN same-surface private receiver proof plus private bridge-shadow evidence |
| STS causality | `reports/sts_causal_decoder_ablation.json` is `GREEN`: STS-on public task coverage `1.0`, same-seed STS-off `0.96875`, coverage delta `+0.03125`, pass-rate delta `0.0`, five decoder-control rows materialized |
| Public calibration | The bounded 2026-06-05 score-existing public calibration pass ran once after the visible receiver repair on MBPP, EvalPlus, BigCodeBench, and HumanEval, 8 tasks each. `reports/real_code_benchmark_graduation_source_mbpp_source_evalplus_source_bigcodebench_source_human_eval_seed14_8.json` is `GREEN` as a scored verdict, with public pass rate `0.84375` (`27/32`), candidate provenance clean, no templates, no loop-closure candidates, and no external inference. The wrapper report is `GREEN` but `promotion_evidence=false` because the slice is thin per card and MBPP remains below the `0.70` per-card floor |
| Maturity | `reports/maturity_integrity_audit_after_visible_receiver_repair.json` is `YELLOW` only on `public_transfer_floor_cleared`; hard/evidence blocker counts are `0`. This reflects thin per-card transfer evidence, not leakage |
| Candidate promotion | `reports/candidate_promotion_gate_after_visible_receiver_repair.json` remains `promote=false`, `24/28`. Failed gates include older general public comparator, runtime/GPU governance evidence, and maturity-integrity GREEN status. Do not promote or grow the model from this run |
| Residual packet | `reports/public_code_transfer_residual_report_after_visible_receiver_repair.json` is `GREEN` and aggregates the current public wall without embedding public prompts/tests/solutions/code. The current failed-task surface is `5`: `edge_case=2`, `type_handling=1`, `parsing=1`, and `local_code_generation_adapter_needed=1`, with adapter-adjusted true remaining no-admissible tasks at `0` |
| Governance | `reports/self_evolution_governance.json` keeps teacher apply blocked by policy/runtime gates. Teacher may be used only as proposal/architecture guidance, never benchmark-answer generation or unreviewed apply mode |
| External inference | `reports/external_inference_audit.json` is clean: `total_violations=0` in the report summary |
| Virtual Context Memory | The `Virtual_Context_Memory_v1.0` packet has been reviewed as a conceptual architecture packet, not an empirical claim. Theseus implements the local semantic/context compiler subset with stable `vcm://...@v` pages, evidence-carrying representation certificates, non-model-visible staging, transaction/snapshot/graph artifacts, deletion closure, context recovery benchmarks, explicit protected-minimum `UNSAFE-FIT` behavior, private prefetch-regret accounting, semantic runtime-key readiness descriptors, and a task-facing context bridge. `reports/vcm_release_conformance_audit.json` is the source of truth: VCM-Core, VCM-Governed, VCM-Transactional, and VCM-Predictive are GREEN; VCM-Governed now requires `task_context_bridge_clean`; VCM-Runtime remains NOT_CLAIMED until native prefix/KV lifecycle integration and hardware-aware runtime cache scheduling exist. `reports/vcm_task_context_bridge.json` is `GREEN`: 9/9 task families ready, 7/7 high-priority families ready, 45 unique selected VCM pages, public training rows/external inference/teacher solving/fallback returns all `0`, and `runtime_profile_claimed=false`. Public memory calibration now has metadata-clean evidence, GREEN readiness, governed RULER/BABILong/LongMemEval repair confirmations, a governed 2000-row RULER/BABILong hard-public confirmation, and a governed five-family 2196-row hard-public confirmation with zero overlap against every prior prompt slice. Current five-family public VCM-on is `0.952641`, flat-tail VCM-off is `0.179417`, best non-VCM is `0.908925`, VCM-over-flat-tail delta is `+0.773224`, VCM-over-best-non-VCM delta is `+0.043716`, VCM-only wins are `1700`, off-only wins are `2`, public training rows/external inference/teacher calls/fallback returns are all `0`, and source contexts reach `607012` token-equivalent. LongMemEval public score improved from `0.055` to the locked `0.10` in the prior 600-row confirmation, but all locally staged LongMemEval rows are now consumed by exact-run ledgers; LongMemEval-V2 text staging is still blocked by the currently loadable image-row surface. The private LongMemEval residual gate is `GREEN` on 180 private rows: VCM-on `0.983333`, best single non-VCM `0.811111`, min major type `0.833333`, evidence recall `0.883333`, abstention precision/recall `1.0`/`1.0`, no public payload loading, and no fallback/teacher/external calls. The hard-memory private analogue gauntlet is also `GREEN`: `1000` private rows, `10` families, `4` length buckets, VCM-on `0.979`, best single non-VCM `0.804`, delta `+0.175`, min family `0.91`, and no fallback/teacher/external/public training. Hard public memory readiness remains `YELLOW` only for source warnings on harder/blocked families; row target and five-family prompt coverage are met. VCM-Runtime remains NOT_CLAIMED |
| VCM evidence gauntlet | `reports/vcm_evidence_gauntlet.json` is `GREEN` on `1200` private/local cases across LongMemEval-style semantic memory, RULER-style needle retrieval, BABILong-style state tracking, and file/task memory. VCM-on is `0.990833`, best single non-VCM is `0.809167`, delta is `+0.181666`, minimum major family is `0.979167`, answerable evidence recall is `0.99009`, abstention precision/recall are `1.0`/`1.0`, fallback/teacher/external/public payload/public training counters are all `0`, and `reports/vcm_proof_card.md` records wins on file/task and LongMemEval-style semantic memory, ties on BABILong/RULER, and no losing family |
| VCM hard memory | `reports/vcm_hard_memory_private_analogues.json` is `GREEN` on `1000` private/local cases across NoLiMa-style lexical disconnect, Michelangelo/LSQ latent structure, LV-Eval confusing facts, LOFT structured memory, LongBench v2, InfiniteBench, MTRAG/MTRAG-UN, FACTS-style grounding, and LoCoMo-Plus cognitive memory. VCM-on is `0.979`, best single non-VCM is `0.804`, delta is `+0.175`, min family is `0.91`, length buckets are `4`, abstention precision/recall are `1.0`/`1.0`, and fallback/teacher/external/public payload/public training counters are all `0`. `reports/vcm_hard_memory_benchmark_readiness.json` is `YELLOW` because of remaining source warnings, not because of row count or five-family prompt coverage: the latest governed public run scored `2196` rows across RULER, BABILong, InfiniteBench, NeedleBench/OpenCompass, and LongBench v2 with VCM-on `0.952641` versus best non-VCM `0.908925`. The honest weak points are LongBench v2 semantic choice evidence, NeedleBench answer formatting/evidence extraction, and LongMemEval-V2 text/evaluator staging |
| Hive fleet | Current work in this thread is Mac-local. Historical two-node Windows/Mac reports must not be treated as live dispatch proof while the Mac is away from that network or Windows reachability is unverified |

## What This Means

The first closed self-improvement transfer loop is partially proven:

- The student can generate broad metadata-only public candidates without public
  tests, public solutions, benchmark answers, teacher apply, templates, or
  loop-closure benchmark candidates.
- Decoder, private-to-public transfer, STS same-seed causality, and causal
  architecture-delta gates are GREEN.
- The current metadata fanout recovered public candidate coverage to `0.96875`
  and reduced public no-admissible tasks to `0` without spending another public
  score run.
- The Mac sandbox path bug is validated fixed by a private/local regression
  suite, and the post-repair bounded public calibration now passes `27/32`.
- Promotion and model growth must remain blocked until targeted private repair
  proves broader transfer maturity: MBPP above floor, 32-task-per-card coverage,
  LiveCodeBench coverage, and clean runtime/governance gates without public data
  leakage.

## Active Frontier Policy

Do not rerun the public calibration surface just to fish for a better score.
The next work must produce private evidence first.

```text
public residual packet -> private-only residual rows/gates
  -> train-once fanout -> decoder/transfer/STS gates
  -> exactly one later bounded public calibration
```

The next source/data work should not add random bulk data. The current residual
packet has now driven private residual-frontier pressure across verifier
mismatch, no-admissible regression, return shape, and algorithmic planning. The
remaining work is to broaden this beyond reusable composition into new private
receiver families and longer real workflow tasks, then use exactly one
operator-approved bounded public calibration only when the operator chooses to
spend it. Teacher use is allowed only as proposal-mode architecture guidance
over aggregate residuals and local evidence.

## Learning Evidence Rules

Use `reports/learning_scoreboard.json` for the current truth split:

- operational health is not learning;
- private training gain is not public promotion;
- synthetic pressure is not public mastery;
- public calibration must keep honest score semantics;
- stale red lanes must be superseded or retired instead of hidden.

Current stale/superseded lanes:

- `student_learning_closure_ranker`: RED but superseded by token-level Code LM;
  do not cite it as learning evidence.
- `genesis_kernel`: YELLOW artifact debt/open critique; useful substrate, not a
  finished invention OS.

VIEA and Reality Manipulator evidence are also not learning evidence. They
prove that intent can be preserved as command contracts, worlds, artifacts,
claim ledgers, gates, lifecycle rows, workflow metrics, and feedback plans.
Student learning still lives in Code LM, STS, private training, public
transfer, and promotion reports.

The VIEA autonomy spine is the operational control path for this substrate. It
refreshes the command executor, SQLite artifact kernel, private repo-repair
curriculum and learner bridge, broad transfer closure, SymLiquid state-engine
queue and route weights, teacher architecture closure/runner, and feedback
action queue. The VIEA action executor can then run approved local queue
actions with `--max-actions`, `--max-steps`, `--timeout-seconds`, and `--resume`.
Use it to decide and execute what to do next; use the student-learning proof
layer to decide whether public-code learning actually improved.

For sleep/vacation runs, use `scripts/vacation_mode_supervisor.py` over the raw
executor. It wraps the spine, checks hard unattended gates, classifies failed
actions, writes `reports/vacation_mode_repair_action_queue.json`, optionally
refreshes governed source/game/RL exploration surfaces, and requires each cycle
to produce progress or a useful residual diagnosis. This is the current best
path toward unattended self-learning without collapsing into narrow benchmark
churn.

## Personality Core

The personality core is now a live runtime substrate, not just a report:

- `scripts/personality_core.py` distills local user-owned documents.
- `scripts/personality_context_builder.py` is the blessed runtime loader.
- `scripts/personality_runtime_audit.py` proves integration.
- `scripts/checkpoint_chat.py` attaches compact personality context to every
  response.
- `scripts/checkpoint_chat.py --session-id ...` persists bounded turn history
  for dashboard/mobile/hive chat continuity.
- `scripts/multi_turn_conversation_benchmark.py` runs the live chat path
  through multi-line, multi-turn cases and feeds `conversation_multiturn` into
  Benchmaxxing.
- `scripts/autonomy_cycle.py`, `scripts/autonomy_watchdog.py`,
  `scripts/autonomy_launch_readiness.py`, and the dashboard all surface or gate
  personality runtime state.

The core is not automatically used as training data. Current policy keeps it
manifest-only until an explicit training run is approved.

## Autonomy / Watchdog

The watchdog contract is:

```powershell
python scripts\autonomy_watchdog.py --fix --out reports\autonomy_watchdog.json
```

`scripts/autonomy_watchdog.py` owns observation and policy assessment. The
bounded fix/refresh command runners live in `scripts/autonomy_watchdog_actions.py`
so speed and retry work can target subprocess orchestration without coupling it
back into the RED/YELLOW/GREEN assessment logic.

If it reports RED, apply the indicated local correction when it is an actual
operational fault. If the RED is an honest learning wall, keep promotion blocked
and report the wall rather than fabricating green status.

Current watchdog state may be RED when teacher escalation is requested but the
Codex CLI is unavailable due usage limits. That is an operational blocker, not
learning evidence. When the watchdog is YELLOW/RED only because public transfer
is below the `0.70` floor, keep promotion blocked and continue the curriculum.
The same-family rotation guard still matters: no single code card should
monopolize the day if it stalls while other smoke-passed code cards are ready.

## Teacher Policy

The teacher is a sparse architect/advisor, not a distillation source. It should
be used for:

- local evidence walls;
- architecture hypotheses after cheaper interventions are exhausted;
- bounded repair proposals when local source bugs block progress.

Teacher calls must be specific. `scripts/teacher_oracle.py` now converts every
caller prompt into a teacher call contract with the exact `reason_for_call`, a
reason intent, a compact wall packet from current reports plus explicit
`--local-evidence`, forbidden roles, and one requested output: the smallest
local architecture/training/verifier experiment with private-first verification
and public-calibration gates. The schema is
`configs/teacher_response_schema.json`.

It should not be used for:

- bulk training data generation;
- public benchmark solving;
- teacher apply mode without guarded branch/gate flow;
- bypassing licensing, data, safety, or privacy constraints.

If the CLI reports a usage-limit reset, record the teacher as unavailable and
continue local residual/curriculum work. Do not retry in a tight loop, and do
not treat stale teacher JSON as fresh architecture guidance.

## What To Do Next

1. Build private-only residual pressure for verifier-mismatch edge contracts,
   return-shape/interface fidelity, parsing/encoding robustness, and
   candidate-floor adapter coverage, using
   `reports/public_transfer_residual_packet_after_visible_receiver_repair.json`
   only for aggregate residual families and hashed task identifiers.
2. Rerun train-once fanout, `decoder_v2_private_ablation_gate`,
   `private_public_transfer_proof`, and `sts_causal_decoder_ablation`.
3. Rerun `maturity_integrity_audit`, `candidate_promotion_gate`,
   `self_evolution_governor`, and `external_inference_audit`.
4. Request another bounded public calibration only after private gates show a
   real source-level improvement and a reviewed plan chooses the next surface
   size. The next promotion-facing public proof should move toward 32 tasks per
   card plus LiveCodeBench, not another fishing rerun of the same 8-task slice.
5. Keep conversation, personality, Reality Manipulator, Genesis, Hive install,
   and board-game RL lanes as support/regression surfaces unless their current
   reports present a new active blocker.

## Source Of Truth Reports

Use these reports before making operational claims:

| Claim area | Report |
| --- | --- |
| Canonical architecture map | `reports/viea_report_map.json`, `docs/VIEA.md` |
| VIEA object store and executor | `reports/viea_artifact_kernel.json`, `reports/viea_command_executor.json`, `reports/viea_action_executor.json` |
| VIEA growth surfaces | `reports/broad_transfer_closure.json`, `reports/digital_runtime_adapter.json`, `reports/workflow_tool_compiler_v2.json`, `reports/symliquid_substrate_map.json`, `reports/teacher_architect_loop.json`, `reports/feedback_ratchet.json` |
| VIEA action ledger | `reports/viea_action_execution_ledger.jsonl`, `reports/viea_action_executor_state.json` |
| Private repo-repair learner bridge | `reports/viea_repo_repair_learner.json`, `reports/repo_repair_trace_checkpoint.json`, `reports/repo_repair_training_traces.jsonl` |
| SymLiquid route state | `reports/symliquid_state_engine.json`, `reports/symliquid_state_engine_queue.json` |
| Teacher architecture runner | `reports/teacher_architect_experiment_runner.json`, `reports/teacher_architect_experiment_ledger.jsonl` |
| Overall learning truth | `reports/learning_scoreboard.json` |
| Intent/world/artifact substrate | `reports/reality_manipulator.json`, `reports/genesis_kernel/report.json` |
| Watchdog health | `reports/autonomy_watchdog.json` |
| Candidate promotion | `reports/candidate_promotion_gate.json` |
| Mac accelerator parity | `reports/accelerator_parity_manifest.json`, `reports/macos_mlx_parity_audit.json`, `reports/macos_metal_production_route_readiness.json`, `reports/macos_metal_standalone_scheduler_canary.json`, `reports/macos_metal_rollout_sweep_scheduler_canary.json` |
| Real public code calibration | `reports/real_code_benchmark_graduation.json` |
| Code LM learning | `reports/code_lm_closure.json`, `reports/code_lm_closure_rust.json` |
| STS / parallel streams | `reports/sts_learning_forge.json`, `reports/sts_native_parallel_probe.json` |
| Code residuals / transfer | `reports/code_residual_forge.json`, `reports/code_transfer_artifacts.json` |
| Curriculum | `reports/benchmaxx_curriculum.json`, `reports/frontier_policy_status.json`, `reports/high_transfer_curriculum_scheduler.json` |
| Board-game RL / Elo | `reports/board_game_rl_benchmark.json`, `reports/board_game_elo_ratings.json`, `reports/board_game_elo_history.jsonl`, `reports/board_game_rl_traces.jsonl` |
| Personality runtime | `reports/personality_runtime_audit.json`, `reports/personality_context_last.json` |
| Conversation regression gate | `reports/high_transfer_multi_turn_conversation.json`, `reports/multi_turn_conversation_benchmark.json` |
| Append-only report evidence | `reports/report_evidence_store.sqlite`, `reports/report_evidence_store.json` |
| Hive board rotation | `reports/hive_work_board_executor.json`, `reports/hive_work_board.sqlite` |
| Overnight readiness | `reports/overnight_learning_readiness.json` |
| Resource and performance | `reports/resource_governor.json`, `reports/performance_optimizer.json` |
| Genesis kernel | `reports/genesis_kernel/report.json` |
| Cell lifecycle / anti-bloat | `reports/cell_lifecycle.json` |
