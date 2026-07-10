# Project Registry

The project registry is the governance kernel for Theseus' active identity. It
exists to prevent new scripts, reports, generated payloads, compatibility
wrappers, model routes, and tool paths from accumulating without a stable
contract and owner.

It is also Theseus' current self-model for maintenance and autonomy decisions.
If a change is not represented here, Theseus should not treat it as part of its
active identity.

Authoritative files:

- `configs/project_manifest_registry.json` defines surfaces, abstractions,
  implementations, lifecycle state, routing eligibility, ATT&D hooks, canonical
  entrypoints, report outputs, verification commands, report-output freshness
  policies, and cleanup policy. Mutable latest-view reports remain
  freshness-gated; historical experiment reports are retained separately so
  old evidence remains auditable without pretending to be current route health.
  The old strict-generator MLX sweep/adaptation/decode family and Circle
  transfer diagnostics are now retained evidence. Current neural-seed routing
  health is represented by `reports/private_verifier_spine_smoke.json`,
  `reports/private_verifier_spine_smoke.md`,
  `reports/neural_seed_strict_generator_fanout_receipt.json`, and
  `reports/neural_seed_strict_generator_fanout_receipt.md`.
- `scripts/theseus_project_registry.py` materializes the current state into
  `reports/theseus_project_registry.json` and
  `reports/theseus_project_registry.md`.
- `configs/roadmap_implementation_matrix.json` is the machine-readable
  implementation state for the AI-book-derived roadmap. It preserves phases
  0-19, binds each phase to registry surfaces/abstractions/implementations,
  and records the smallest next patch before any phase can be called done.
- `scripts/roadmap_implementation_gate.py` validates that roadmap phases are
  registry-owned, SCF-bound where appropriate, execution-spine-wired, and not
  credited from public benchmark training, router/template/tool fallbacks, or
  prose-only evidence. It also writes `reports/book_to_theseus_crosswalk.json`
  so AI_book source ideas, roadmap phase state, registry ownership, evidence,
  and smallest next patches stay machine-readable. The crosswalk includes an
  AI_book source inventory with SHA-256 checksums, excludes generated book
  builds/archives/Lean build outputs, records per-phase source-sync state, and
  emits sticky stale-source `roadmap_backlog_item` rows when matched source
  hashes change. Those backlog rows carry forward until steward review clears
  them, and they are routed by `scripts/module_definition_of_done_gate.py` into
  module DoD work cards plus steward-decision candidates.
- `configs/project_steward.json` carries active cleanup steward decisions. The
  current registry cleanup queue is still visible as pressure, but `16/16`
  queue items are covered by active decisions that say whether to consolidate,
  retain under artifact retention, or reject new side lanes.
- `configs/viea_spine_record_contracts.json` defines the shared VIEA record
  families and aliases used by assistant, planner, tool, Hive scheduler route
  and execution-receipt records, candidate-integrity, private verifier traces,
  train-once fanout supervisor records, direct generator boundary records, VCM
  native-runtime readiness records, Context ABI fixture records, governance
  rights receipt records, simulation fidelity receipt records, Octopus route
  records, learned Octopus router-head training records, proof-carrying
  contract receipts, substrate adoption records, strict generator fanout replay
  receipts, strict compressed-artifact records, compression receipts, and
  compact-generative records. Producers may
  keep local payloads, but they must normalize into these canonical record
  families before roadmap completion or routing claims can use them.
- `scripts/viea_spine_record_gate.py` validates current assistant flat traces,
  planner nested ASI-stack records, deterministic tool evidence/artifact
  graphs, execution-spine runtime records, Hive scheduler route records, Hive
  task-submission receipt smoke records, candidate-integrity audit records,
  private verifier spine-smoke records, train-once fanout records, direct
  generator context-boundary records, semantic IR obligation records, VCM
  runtime-cache boundary records,
  Context ABI fixture records, governance rights receipt records, live operator
  governance audit/export records, simulation fidelity receipt records,
  Octopus route records, learned Octopus router-head training records,
  proof-carrying contract receipts, substrate adoption records,
  public-calibration proposal receipts, strict generator fanout replay records,
  compressed-artifact records, compression receipts, and compact-generative
  records against that shared contract. It also
  writes
  `reports/viea_spine_materialized_view.json`, the shared
  claim/evidence/semantic/simulation/governance/authority/route/failure record
  view that other runtime surfaces should consume. Artifact graph and context
  transaction, compressed-artifact, compression-receipt, and
  compact-generative materialized payloads are normalized to the current
  AI_book schema-required fields, and `schema_payload_gap_count` must stay zero
  before those records can support roadmap or route-readiness claims.
- `scripts/viea_spine_records.py` is the shared helper for canonical record
  aliases and materialized-view consumer receipts. Candidate integrity, private
  verifier summaries, the assistant runtime, and the Hive policy-first
  scheduler use it to bind their local reports to the same VIEA view without
  turning tool/router/verifier behavior into learned-generation evidence.
- `reports/semantic_ir_obligation_gate.json` is the current Phase 13
  obligation-binding proof. It binds candidate-integrity, private-verifier, and
  direct-generator report obligations to the materialized semantic atom/node
  view and emits semantic-obligation, dependency-edge, and evidence-binding
  records without executing generation or public calibration.
- The project registry SCF route validator also consumes that materialized VIEA
  view. Route approval now requires governance, failure-boundary, authority,
  and resource-route groups to be present before routable implementations are
  trusted.
- The Hive scheduler route-validator bootstrap cycle is explicit in
  `configs/viea_spine_record_contracts.json`: bootstrap is pass-only when
  required groups are present and no no-cheat counters fault, then the scheduler
  must rerun against the green materialized view and record a ready receipt.
- Assistant product traces and Hive scheduler route records attach those
  route-validator consumer receipts to their route-decision records. This keeps
  runtime routing tied to the same governance/failure view the registry uses.
- `configs/assistant_trace_schema.json` is part of the
  `theseus_assistant_runtime` surface. Runtime reports, conversation events,
  and VIEA `policy_optimization_record` rows must cite this schema before
  accepted/missed/ignored/corrected/completed dogfood outcomes count as local
  metadata training pressure.
- Hive installer, artifact index, artifact sync, and artifact merge reports
  attach VIEA artifact-citation receipts from the same materialized view. This
  ties artifact manifests to claim-ledger, artifact-record, and
  evidence-transition refs without treating package/sync metadata as a
  learned-generation claim. The shared citation helper ranks refs by artifact
  family relevance, support state, route-validator readiness, and no-cheat
  cleanliness so Hive manifests prefer Hive-native support records when they
  exist.
- The report evidence store and artifact-retention service now emit generic
  compression records, strict compressed-artifact records, compression
  receipts, and defeater records. Large snapshot-backed reports can be traced
  back to exact payload hashes, and mutable latest-view supersessions are
  recorded without deleting or invalidating historical runs. RMI also emits a
  compact-generative record for its operating map with generation,
  verification, fallback, residual-burden, promotion-blocker, source-ref,
  evidence-ref, and non-claim fields required by the current AI_book schema;
  these are evidence/shape records, not compression-ratio, downstream-utility,
  or learned-generation claims.
- `reports/public_calibration_proposal_gate.json` is part of the
  `public_calibration_registry` surface. It binds public calibration proposals
  to candidate integrity, training-data firewall state, alignment preflight,
  and exact run-registry consumed/fresh-surface state. The current default
  surface is refused because it is already consumed; clean fresh surfaces can
  still run without calendar throttles.
- `reports/theseus_assistant_product_spine_smoke.json` is the current
  product-facing assistant integration proof. It exercises VCM context,
  deterministic tool evidence, private verifier receipt, materialized-view
  receipt, authority/resource/failure/artifact/claim trace records, and
  metadata-only dogfood feedback in one local assistant call.
- `scripts/theseus_weekly_focus_20260706.py` is the current public-safe book
  evidence bridge for the weekly focus. It refreshes the registered assistant
  product-spine route, exports
  `reports/theseus_public_safe_reference_trace_20260706.json`, exports
  `reports/theseus_book_importable_evidence_packs_20260706.json`, audits
  receipt faithfulness, residual conservation, verifier capacity,
  governance-tax accounting, capability-claim dispositions, and book-schema
  conformance, and preregisters exactly one bounded correctness-in-the-loop
  generator experiment. This is implementation-reference evidence for
  AI_book import/review, not a model-quality, public-benchmark,
  learned-generation, deployed-readiness, or ASI claim.
- `reports/theseus_assistant_vcm_governor_smoke.json` is the current
  product-facing VCM adequacy proof. It exercises the VCM governor inside the
  assistant runtime, emits `context_transaction` and `context_adequacy` trace
  records, and hard-gates on mission-brief, taint, deletion-closure, and
  no-cheat receipts. This is context-governance evidence, not native
  KV/prefix-cache parity and not learned capability evidence.
- `reports/vcm_context_governor.json` is also the current Context ABI fixture
  proof. The gate validates leased materialization, mandatory miss typed fault,
  verification-inadequate rejection, mount-policy denial, and expired-lease
  reuse blocking, then emits `viea_context_abi_records` for the shared
  `vcm_context_abi_fixture_spine_v1` profile. This is resolver/fault protocol
  evidence, not a public-memory benchmark, native KV/prefix-cache parity, or
  learned-generation claim. The same report now also emits
  `viea_context_resolver_records` for
  `vcm_context_resolver_conformance_spine_v1`: `7/7` real semantic-address
  requests pass, `3` local artifact refs are materialized, and `4` typed
  faults are emitted for blocked/missing/stale/unsafe context requests without
  leaking raw payloads.
- `reports/deterministic_tool_substrate.json` is the deterministic-tool VCM
  context proof. The local exact-tool lane now requires the VCM governor
  receipt, emits `viea_tool_context_records` for
  `deterministic_tool_context_spine_v1`, and keeps tool evidence separate from
  learned-generation evidence. The latest smoke is `GREEN` with `13` local
  tools, `15/15` verified private results, `7` VCM context records, and zero
  public training rows/runtime external inference/fallback returns.
- `reports/training_data_admission_v1.json` is the training-admission VCM
  context proof. It consumes `reports/vcm_context_governor.json`, emits
  `viea_training_data_context_records` for
  `training_data_admission_context_spine_v1`, and fails closed if the resolver
  is not ready. This is metadata admission governance, not training execution
  or public calibration evidence.
- The same registered `teacher_and_data_governance` field owns
  `scripts/training_data_lineage_audit.py` and the content-bound receipt ledger
  at `runtime/data_governance/data_admission_receipts_v1.jsonl.gz`.
  `training_data_candidate_lifecycle_spine_v1` requires candidate admission,
  lineage, license, leakage, verifier, lifecycle-policy, deletion-closure, and
  failure-boundary records. Curriculum and survival-lane materialization prove
  every selected row hash is admitted; receipt completeness does not imply
  data quality, model capability, forgetting, or physical unlearning.
- `reports/theseus_plan_compiler.json` is the current planner-facing VCM
  adequacy proof. The registered planner now requires
  `reports/vcm_context_governor.json`, emits governed `context_transaction`
  and `context_adequacy` records for every compiled node, and hard-gates if the
  mission brief, SCIF, deletion closure, or no-cheat receipt is not ready. The
  same planner report now also consumes `reports/deterministic_tool_substrate.json`
  and `reports/deterministic_tool_registry.json`: every compiled node declares
  tool eligibility, every node carries tool receipts, and the shared VIEA
  `plan_compiler_nested_spine_v1` profile requires `tool_call_receipt` records.
  These receipts are evidence refs for local exact tools and must not support
  learned-generation claims or model-only benchmark scores.
- `reports/private_verifier_spine_smoke.json` is the current verifier-facing
  VCM adequacy proof. The private verifier spine smoke now consumes
  `reports/vcm_context_governor.json`, emits governed `context_transaction`
  and `context_adequacy` records, and hard-gates the smoke if context
  governance is not ready. This is verifier-label/context evidence, not
  learned-generation credit.
- `reports/code_lm_train_once_fanout.json` is the current fanout-facing VCM
  adequacy proof. The canonical train-once fanout supervisor now consumes
  `reports/vcm_context_governor.json` and emits governed
  `context_transaction` and `context_adequacy` records in its existing report
  without starting a training run. This is candidate-manifest traceability
  evidence, not public calibration or learned-generation credit.
- `reports/neural_seed_token_decoder_comparator.json` is the current direct
  generator VCM boundary proof. In planned mode it records that
  `neural_seed_candidate_generation.generate_candidates` consumes
  `reports/vcm_context_governor.json`, fails closed before model decode if the
  governed context receipt is missing, and emits VIEA direct-generator records
  for the shared spine gate. This is boundary/traceability evidence, not a
  decoded-code quality claim.
- `reports/candidate_promotion_gate.json` is the current promotion-integrity
  guard. It must cite `reports/candidate_integrity_audit.json`, require that
  audit's VIEA consumer receipt to be ready, and expose recomputed family
  counts before promotion can pass. `--allow-blocked` validates the report
  shape while preserving `promote=false` when unrelated capability gates block.
- `reports/maturity_integrity_audit.json` and
  `reports/public_transfer_readiness_refresh_v1.json` inherit the same
  promotion-integrity receipt before promotion/public-transfer readiness can be
  treated as current. This prevents downstream reports from bypassing
  independent candidate-family recomputation.
- `reports/vcm_native_runtime_probe.json` is the current VCM runtime-cache
  boundary proof. It emits VIEA runtime records for authority, context
  transaction/adequacy, runtime adapter, resource, costed route, generation
  mode, failure, artifact, claim, and evidence transition families. It proves
  backend-scoped CPU Transformers DynamicCache prefix/KV lifecycle and MLX
  resident tensor descriptor lifecycle, while keeping scheduler native KV
  routing fail-closed for the recommended MLX backend until model-native MLX
  KV/prefix lifecycle is proven. This is runtime-boundary evidence, not a
  CUDA/MLX/Metal parity claim.
- `reports/governance_rights_receipt_suite.json` is the current
  governance-rights material-usability fixture proof. It covers complete audit
  response, justified redaction with appeal, portable exit export, and fork
  denial when safety obligations cannot transfer, then emits
  `governance_right`, `failure_boundary`, artifact, claim, and evidence records
  through `governance_rights_receipt_spine_v1`. This is protocol evidence, not
  institutional governance or learned capability.
- `reports/hive_operator_governance_audit.json` is the current live local
  operator audit/export receipt. It is produced by
  `python3 scripts/hive_node.py operator-governance-audit --out reports/hive_operator_governance_audit.json`
  or `/api/hive/operator/governance-audit`, exposes refs/hashes/citations
  rather than raw private text or secrets, and emits governance-right,
  authority-use, failure-boundary, artifact, claim, and evidence-transition
  records through `hive_operator_governance_audit_spine_v1`.
- `reports/simulation_fidelity_receipt_suite.json` is the current simulation
  claim-boundary fixture proof. It covers unit invariant, benchmark-with-limits,
  scenario-only, blocked-transfer, and downgraded-claim cases, then emits
  simulation contract, fidelity, counterfactual, world-adapter, failure,
  artifact/claim/evidence records through `simulation_fidelity_receipt_spine_v1`.
  This is fidelity-accounting evidence, not physical feasibility, public
  benchmark transfer, live simulator evidence, or learned generation.
- `reports/book_to_theseus_backlog_work_cards.jsonl` is the generated module
  work-card view over unresolved AI_book source-change backlog rows. It is
  quality/governance evidence only; it cannot support learned-model capability
  or public-transfer claims.
- `reports/teacher_share_ledger_summary.json` is the durable teacher-share
  accounting view. `scripts/teacher_distillation_gate.py` writes it from
  `reports/teacher_distillation_ledger.jsonl`; the control plane consumes it
  through `teacher_share_ledger_ready`, and the assistant runtime consumes it
  through the `teacher_share_ledger_metric_ready` hard gate. It records
  training-time teacher calls separately from runtime external inference, which
  must remain zero.
- `scripts/theseus_plan_compiler.py` is the registered planning compiler. It
  turns goals into typed, governed VCM-backed execution DAGs and routes them to
  existing executor surfaces; it is not a separate executor. It now consumes
  `reports/procedural_memory_toolification.json` as a canary-only input: a
  replay-passed assistant-trace procedural candidate can become a
  registry-gated planner route packet.
- `scripts/procedural_memory_canary_executor.py` is the bounded executor for
  those canary route packets. It validates the replay fixture against current
  metadata-only assistant events, checks the compiled planner DAG, emits a
  local route packet, records duplicate-work and verification-cost deltas, and
  writes `reports/procedural_memory_canary_execution.json`. It may prove
  canary execution, but it may not adopt a default procedural route; default
  adoption still requires a registry transaction, no-cheat counters, replay
  cleanliness, and rollback criteria.
- `scripts/procedural_memory_route_adoption_gate.py` performs that registry
  transaction. It consumes the procedural-memory gate, canary execution report,
  route-validator-ready registry view, and project steward contract, then emits
  `reports/procedural_memory_route_adoption.json`. A default route can be
  adopted only as local metadata workflow compression with an armed regression
  guard; it remains forbidden to count this as learned generation or public
  transfer. The adoption transaction now carries a `route_binding_contract`
  that binds the route to `local_assistant`, `planning`, `planning_assistant`,
  and `autonomy_governance`, with selection keys `surface`, `intent`,
  `assistant_lane`, and `vcm_task_family`.
- `scripts/theseus_assistant_runtime.py` is now a consumer of that adoption
  report for planning intent. The runtime hard-gates on a GREEN adoption
  report, armed regression guard, learned-generation claims disabled, and zero
  public/external/fallback counters before it attaches the guarded metadata
  route to an assistant planning trace. It must also prove that the route
  binding contract matched the runtime request and that
  `theseus_assistant_runtime` is an authorized consumer; otherwise the
  procedural default route remains unavailable.

Registry layers:

- **Surface**: files/reports/docs/configs owned by a lifecycle entry.
- **Abstraction**: stable contract for a capability, including I/O contract,
  forbidden shortcuts, allowed variability, required evidence, related
  surfaces, and canonical implementation.
- **Stable Capability Field**: the semantic ABI for an abstraction. It records
  the contract version, exact content identity, authority/effect ceiling, state
  continuity policy, scoped qualification evidence, caller-bound route
  validation, lease behavior, observability receipts, adaptation/migration
  policy, lifecycle/recovery rules, and governance controls. A field does not
  perform domain work; it mediates whether replaceable implementations may do
  so.
- **Implementation**: concrete backend/entrypoint that satisfies one
  abstraction, with role, trust tier, evidence outputs, ATT&D hooks,
  replacement/retirement policy, router eligibility, and a stable capability
  binding to the abstraction field.

Top-level layout:

- `bin/`: human-facing CLI/setup wrappers only. Keep root launchers out of the
  repository root.
- `docs/reference/`: source papers, architecture packets, and externalized
  design references that are still useful to active audits.
- `deprecated/`: historical or accidental material retained for mining ideas.
  Active routes, routers, and training paths should not depend on files under
  `deprecated/`.
- `runtime/`, `reports/`, `archive/`, `checkpoints/`, `target/`, and `tmp/`:
  generated/build/runtime state. Keep source claims out of these roots and use
  retention or ignore policy for large artifacts.

Router rule:

Routers, Octopus arms, promotion gates, autonomy loops, teacher gates, and
runtime paths may select only implementations that are registered,
routing-eligible, evidence-current, and allowed for the requested role.
Templates, adapters, deterministic tools, and routers are not learned code
generation unless the implementation contract and independent integrity
evidence explicitly allow that claim.

Octopus/MoECOT route records:

- `scripts/octopus_router.py` is the canonical Octopus router implementation;
  do not create a parallel `octopus_registry_gated_router` surface to paper
  over routing gaps.
- `reports/octopus_router_report.json` must emit
  `viea_moecot_route_records` and pass the
  `octopus_moecot_route_spine_v1` profile in
  `configs/viea_spine_record_contracts.json`.
- A valid route decision records the selected specialist arms, rejected or
  missing arms, SCF/registry route-validator receipt, VCM context transaction
  and adequacy, authority receipt, runtime adapter, resource budget, costed
  route, generation-mode boundary, failure boundary, artifact, claim,
  evidence-transition, and residual.
- Octopus routing may improve execution and scheduling, but it contributes
  `0` learned-generation credit. It must keep public training rows, runtime
  external inference calls, and fallback returns at `0`.

SCF rule:

The registry must answer a scoped routing question, not return a flat global
implementation pointer. The current SCF source is
`docs/Stable_Capability_Fields_Public_Release_v1.0_Corben_Sorenson.docx`, with
the earlier reference retained under `docs/reference/`. Each routable
capability needs:

- semantic contract: inputs, outputs, pre/postconditions, invariants, typed
  failures, and uncertainty semantics;
- identity policy: exact content binding for artifact, manifest, contract,
  profile, dependency closure, evaluator policy, state relation, and evidence
  bundle identities; human-readable names are aliases only;
- evidence registry policy: append-only source events, deterministic
  materialized views, scoped/defeasible claims, defeaters, waivers, and
  transfer witnesses;
- authority/effect policy: allowed effect classes, authority ceiling,
  delegation depth, and forbidden authority compositions;
- state policy: state classes, ownership, migration policy, and rollback or
  containment policy;
- qualification policy: deployment profiles, required evaluators, hard vetoes,
  and lifecycle statuses;
- resolution and route-validation policy: route dimensions,
  deterministic/replayable selection semantics, caller binding, validator
  receipts, role constraints, and expiring fail-closed leases;
- observability policy: required events, evidence receipts, and privacy
  boundary;
- adaptation/migration policy: sealed adaptation epochs, pinned updaters,
  approved data receipts, sentinels, journals, qualified baselines, migration
  solvency class, and rollback or compensation path;
- composition/governance policy: bundled dependency cycles, toxic composition
  controls, change classification, trust-root/threshold/timelock policy,
  federation/appeals/incident propagation, and emergency powers that can
  narrow or recover but not silently expand authority.

`scripts/theseus_project_registry.py --gate` fails if a live/retained field or
implementation binding is missing these SCF dimensions. ATT&D and the control
plane consume the same SCF health output, so self-improvement work cannot route
through a field that lacks exact identity, state, authority, evidence, route
validation, lease, adaptation, migration, or recovery semantics.

Lifecycle statuses:

- `live`: active surface with a canonical owner and verification command.
- `experimental`: registered but not canonical or route-eligible unless a
  bounded experiment says so.
- `retained`: tracked evidence/support path; not selected by routers unless
  explicitly eligible.
- `generated`: runtime/build/report state. Keep it out of source paths and use
  retention or GC manifests for cleanup.
- `deprecated`: compatibility or legacy material that needs a replacement,
  pointer, or retained reason.
- `retired`: intentionally inactive and no longer considered part of current
  identity.
- `blocked`: known but not usable until named evidence gaps clear.
- `platform_not_applicable`: valid on another platform but not active here.

Cleanup rules:

- Improve an existing registered surface before creating a new one.
- Improve or replace an implementation under an existing abstraction before
  creating a new abstraction.
- A new surface is allowed only when the registry declares its owner, role,
  canonical entrypoint, report outputs, verification command, cleanup policy,
  and relationship to the surface it supersedes or coexists with.
- A new abstraction is allowed only when no existing abstraction can honestly
  own the behavior.
- Repeated `vN`, `seed`, `current`, `after`, `smoke`, and one-off lanes are
  registry violations unless they are explicit compatibility wrappers,
  retained evidence, or archived generated artifacts.
- Do not delete evidence blindly. Heavy artifacts move through
  `scripts/theseus_artifact_retention.py` with manifest pointers.
- Do not commit generated runtime payloads unless they are intentionally small
  manifests or curated evidence.
- Public benchmark payloads remain calibration-only and never become training
  data.
- New lanes must be registered as a surface or classified as retained support;
  repeated vN/seed/current/after families should be consolidated instead of
  cloned again.
- Standalone design packets belong under `docs/reference/`, not the repo root.
- Historical launchers belong under `bin/` if still useful, or
  `deprecated/legacy-launchers/` if they are no longer supported.
- ATT&D consumes registry gaps and must emit concrete maintenance packets for
  abstraction/implementation gaps, routing blockers, unregistered active
  sources, duplicate families, stale report outputs, and generated artifacts in
  source paths.

Evolution contract:

1. Improve the canonical registered surface.
2. Extend the existing surface with an explicit role if the boundary is still
   the same.
3. Replace an implementation under the same abstraction when machinery changes.
4. Create a new surface only with a complete registry entry.
5. Create a new abstraction only with a complete contract and justification.
6. Deprecate, archive, retire, or explicitly retain the superseded surface.

Hard registry violations:

- Active source/config/doc files not matched by a registry surface.
- Live major surfaces without an abstraction binding.
- Live abstractions without a canonical live implementation or split-route
  policy.
- Implementations whose abstraction, surface, evidence, or routing eligibility
  cannot be verified.
- Generated cache/build/runtime files inside source paths.
- Missing required surface fields.
- Stale or missing declared latest-view reports for live surfaces, unless the
  report is platform-not-applicable.

Roadmap implementation rule:

- `roadmap.md` is the human roadmap; `configs/roadmap_implementation_matrix.json`
  is the audited state. A roadmap item is not complete until the matrix names
  a registered surface, abstraction, implementation, execution-spine hook,
  required records, required gates, docs, current evidence, and an integration
  smoke. Phases 13-19 from the AI book are preserved even when sequenced behind
  more urgent phases; they may not be removed to make the gate green.
- Current roadmap gate state: `YELLOW` with `0` hard gaps, `11` implemented or
  wired phases, `8` partial phases, and `1` frozen phase. Phase `0`
  Repository Self-Model/Registry Discipline is implemented. Phases `3`, `5`,
  `6`, `7`, `11`, `12`, `13`, `14`, `16`, and `17` are wired. Remaining partial phases must keep explicit
  `missing_items` and `smallest_next_patch` entries until their own registered
  gates and integration smokes justify promotion.

Current expected command sequence:

```sh
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
python3 scripts/viea_spine_record_gate.py --gate
python3 scripts/theseus_workspace_hygiene_audit.py
python3 scripts/theseus_deprecation_registry.py
python3 scripts/attd_analyzer.py
python3 scripts/report_evidence_store.py
python3 scripts/theseus_control_plane.py
```
