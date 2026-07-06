# Project Theseus Roadmap

Created: 2026-06-25

This roadmap mines `/Users/corbensorenson/Documents/AI_book` for the ASI Stack
architecture that Theseus should implement. It also checks the current Theseus
repo state before naming work. The goal is not to create another sidecar plan.
The goal is to make Theseus the working implementation reference for the book:
stable interfaces, governed replacement, real evidence, useful daily operation,
and a local model that improves without cheating.

## Source Basis

Book sources reviewed:

- `/Users/corbensorenson/Documents/AI_book/README.md`
- `/Users/corbensorenson/Documents/AI_book/book_structure.json`
- `/Users/corbensorenson/Documents/AI_book/docs/book_outline.md`
- `/Users/corbensorenson/Documents/AI_book/docs/source_mining_synthesis.md`
- `/Users/corbensorenson/Documents/AI_book/docs/local_project_mining_theseus_circle.md`
- AI_book schemas for command contracts, typed jobs, artifact graphs, claim
  records, evidence transitions, stable capability fields, replacement
  transactions, self-improvement transitions, PlanForge DAGs, VCM records,
  semantic page certificates, runtime adapter invocations, procedural tool
  records, readiness gates, benchmark ratchets, policy optimization records,
  routing decisions, MoECOT orchestration records, and proof-carrying claims.
- AI_book chapters covering stable capability fields, replacement and rollback,
  recursive self-improvement boundaries, intent-to-execution contracts,
  planning, PlanForge DAGs, cognitive compilation, VCM, semantic pages,
  verification bandwidth, claim ledgers, Spinoza verification, tribunals,
  typed jobs, artifact graphs, loop closure, routing heads, readiness gates,
  MoECOT runtime, generate-verify-repair, deterministic substrates, policy
  optimization, integrated reference architecture, and Theseus as the
  report-first implementation reference.
- Second-pass AI_book chapters covering system boundaries and authority,
  ordinary/catastrophic failure boundaries, claim discipline, human intent,
  constitutional predicates, agency/dignity/corrigibility, value conflicts,
  governance rights, Digital SCIFs, runtime adapter permissions, resource
  budgets, simulation fidelity, proof-carrying AI contracts, cyclic memory,
  cyclic mixers, compact generative systems, fast generation modes, and the
  research backlog.
- Third-pass AI_book mining covering artifact steward agents, personal compute
  hives, policy optimization, fast generation, cognitive compilation, procedural
  memory, prototype-roadmap gates, source notes for context engineering,
  verification bandwidth, Black Hole Context Manager, GenesisCode, TreeLLM,
  fast-generation literature, and feedback-learning literature.
- Fourth-pass Claude/AI_book mining covering the book's own `Planned Codex
  test` lists as the concrete technique backlog: DPO/IPO/ORPO/KTO/SimPO,
  GRPO/RLOO/ReMax/RLVR, MTP, Medusa/EAGLE/speculative/LayerSkip generation,
  diffusion/LLaDA sketch-first repair, GVR state transitions, semantic-IR
  localized repair, verifier-capacity routing, VCM transaction certificates,
  Circle/Coil cyclic primitives, belief revision, and procedural-memory
  lookahead.

Theseus sources checked:

- `AGENTS.md`
- `docs/PROJECT_STATE.md`
- `docs/PROJECT_REGISTRY.md`
- `configs/project_manifest_registry.json`
- `configs/theseus_plan_compiler.json`
- `configs/theseus_assistant_runtime.json`
- `configs/permissive_growth_policy.json`
- `scripts/theseus_plan_compiler.py`
- `scripts/theseus_assistant_runtime.py`
- `reports/strict_generator_mlx_rung_findings_v1.md`
- `scripts/theseus_project_registry.py --gate`
- `scripts/theseus_workspace_hygiene_audit.py`

This roadmap is a planning artifact. It is not evidence that any item below is
complete.

## Execution Progress

Updated: 2026-07-06

## 2026-07-06 Claude Review Reconciliation

Claude's latest review surfaced three criticisms with enough evidence to change
the roadmap:

1. `B1_assisted_verified_assistant_product_lane` was over-labeled. The current
   events prove the assistant product lane wiring, VCM/tool/verifier receipts,
   metadata-only fixture/e2e dogfood outcomes, and expected-invalid controls.
   They do **not** yet prove real daily usefulness. B1 is therefore
   `synthetic-test-backed` until real multi-day user dogfood traces exist.
2. Report sprawl has regrown and must be governed as a first-class blocker, not
   a style preference. The hard report-family budget is now implemented through
   `configs/artifact_retention_budget_policy.json` and
   `python3 scripts/theseus_artifact_retention.py --budget-gate`: generated
   report/checkpoint families need registry ownership, retention class,
   archive/replay policy, and a reason they cannot update an existing flagship
   claim record. The current budget gate is `GREEN` after archiving and
   replay-verifying `386` report snapshots; hot reports are under the `1 GiB`
   cap. The remaining retention wall is warning-level checkpoint bulk, which
   needs current-reference-aware checkpoint compaction.
3. The roadmap underweighted concrete ASI Stack Part III capability techniques.
   The next C1/Phase 10 capability work must test real from-scratch techniques
   under substrate-adoption discipline: multi-token prediction (MTP) versus a
   matched single-token control, generate-verify-repair versus one-shot decode
   with tool-assisted/model-only scores separated, and a dense-vs-sparse-MoE or
   50-150M scale ablation with matched active compute. These are not new side
   lanes; they are registered implementation candidates for the practical
   transformer/hybrid survival lane.
4. The deeper ASI Stack mining pass adds a stronger correction: the book's
   Policy Optimization and Learning from Feedback chapter is not optional
   machinery. Theseus has verifier labels, accepted/rejected private
   candidates, STS/VCM/router policies, and dogfood metadata. The live
   generator now has a first governed DPO shadow update with a frozen reference
   checkpoint, but it has not yet proven heldout decode/verifier behavior or
   earned any default route. The roadmap must therefore treat policy
   optimization as the next behavior-changing lever, not as a completed
   fixture.
5. Fast generation is broader than the current MLX/beam work. MTP,
   lookahead/trie continuation from verified procedural memory,
   sketch-first/diffusion-style repair, early-exit/self-speculative modes, and
   KV/prefix-cache accounting must be generation-mode records judged by useful
   verified output per second, not proposed-token throughput.
6. VCM, claim ledgers, and verification bandwidth need harder mechanics:
   representation certificates, authority ceilings, copy-on-write context
   branches, taint/deletion closure, receipt-faithfulness audits, an explicit
   epistemic trusted-computing-base record, claim-belief revision transitions,
   verifier-capacity budgets, and governance-tax accounting. These are not
   product polish; they are the control surfaces that prevent self-improvement
   from becoming uninspectable.
7. Claude's later book-mining pass does not add another lane; it sharpens the
   execution order. The ASI Stack chapter-level `Planned Codex test` lists are
   the implementation backlog, and Chapter 38 remains a necessary capability
   lever. The first DPO shadow update exists now and the required private replay
   has run; it failed behaviorally with zero learned candidate rows and zero
   private heldout passes. That means policy-gap movement is not enough. Phase
   10 must repair direct prompt/signature body-token emission before broader
   RLVR/GRPO, MTP, diffusion, or scale work. Accepted output quality must move
   without reward-hacking, public rows, fallback rows, renderer credit, router
   credit, or tool credit. The latest strict decode hygiene pass removes
   malformed `isinstance` first-argument chains, bare builtin type values used
   as runtime values, and constant-only control-flow conditions, but the broad
   private canary still emits zero learned rows and falls back to noncredit
   `return None` baselines; this confirms the remaining wall is semantic/action
   body construction, not another narrow malformed-token family.

## 2026-07-06 Claude Book-Mining Delta Review

Claude's deeper AI_book mining packet is accepted as a roadmap-quality review,
not as evidence. Its main value is that it turns the book's `Planned Codex
test` lists into an implementation backlog. Those planned tests are now treated
as missing technique obligations unless a registered Theseus implementation has
run a matched control, negative control, no-cheat audit, and retained residuals.
The review does **not** authorize a new side lane, a public-training exception,
or learned-generation credit for routers, tools, renderers, action catalogs,
templates, or fallback bodies.

Current reconciliation:

| Claude-mined requirement | Current Theseus state | Roadmap action |
| --- | --- | --- |
| DPO / IPO / ORPO / KTO / SimPO as offline preference levers | A first DPO shadow update exists and moved a private preference gap, but strict replay emitted `0` learned candidate rows and `0/16` behavior passes. | Keep DPO quarantined as loss/policy-gap evidence only. Repair direct prompt/signature body-token emission before any larger offline-preference sweep. When resumed, compare DPO/IPO/ORPO/KTO/SimPO on the same private pair split with the same reference checkpoint and heldout verifier. |
| GRPO / RLOO / ReMax / RLVR | Policy-optimization records and reward-hacking probes exist, but no verifier-reward update has improved heldout behavior. | Queue only after non-fallback learned candidates exist. Require reward-present/reward-removed ablation, reward-hacking probes, drift bounds, rollback plan, rollout budget, verifier-cost ledger, and no authority expansion. |
| Multi-target policy optimization | Current policy-update evidence is generator-centered. The book explicitly treats planners, routers, VCM selectors, verifier policies, execution policies, and generation-mode selectors as policy targets. | Add multi-target policy updates only through behavior-change leases with target policy, admissible feedback, authority effect, rollback, monitor window, and task-specific heldouts. Do not let a policy update expand authority or erase verification cost. |
| Fast-generation modes: MTP, Medusa, EAGLE, speculative, early-exit/LayerSkip, lookahead/trie, Mamba/PagedAttention cache policy, diffusion/LLaDA sketch-first repair | Generation-mode registry and runtime accounting exist; current comparisons are negative for accepted verified output. | Treat each mode as a registered generation-mode candidate measured by useful verified output per second, not proposed-token speed. Do not run MTP/diffusion before the direct learned emission wall moves. |
| GVR as typed state machine | The generator has verifier receipts and repair hooks, but GVR is not yet the universal transition contract. | Implement candidate -> verified-exact -> verified-lossy -> repaired-exact -> literal-fallback/noncredit -> quarantined as a typed state machine with S/R/Q/G/V/E consistency checks, verifier receipts, repair cost, fallback count, residuals, and explicit model-only versus assisted scoring. |
| Semantic IR / localized repair | Semantic atom and semantic patch gates exist, but generator failures are not yet lowered into atom-level repair obligations. | Connect strict-generator decode failures to semantic atoms: failed atom -> localized repair -> dependent obligation replay -> residual ledger. |
| Verification bandwidth | VCM/context-governor and verifier-spine records exist, but verifier capacity is not yet a scheduled resource in all fanout/ranking paths. | Add verifier-capacity budgets, residual-obligation ledgers, decomposition contracts, and governance-tax accounting to Phase 16 before route-policy changes. |
| VCM certificates and context transactions | Context ABI fixtures, mission briefs, deletion closure, and materialized VIEA context records exist. | Promote fixture-level semantics into a deployed resolver/compiler conformance gate, then add model-native MLX KV/prefix-cache lifecycle proof before runtime parity claims. |
| Circle / Coil / cyclic substrate primitives | Circle/proof-carrying bridge and substrate-adoption records exist as proof-boundary machinery. | Keep cyclic/SymLiquid ideas protected as matched-compute discovery candidates only. Test cyclic memory, KV-cache ring buffers, recurrence schedules, sparse-attention coverage, circulant/block-cyclic mixers, and MultiCoil RoPE phase features against dense/LoRA/RoPE/Mamba controls with matched compute before any route claim. |
| Planning control layer and intelligence-arbitrage ledger | VIEA/PlanForge-style records exist; live peer execution remains externally frozen and some arbitrage costs are not yet measured. | Keep local traces canonical; add adequacy contracts, displaced verification/repair/human-cleanup cost, and local subgraph repair before changing default routing. |
| Claim ledger belief revision | Support states and evidence packs exist, but revision transitions are not universal. | Add downgrade/split/merge/contradiction/retire transitions with immutable revision history under Phase 14. |
| Procedural memory / loop closure | Procedural-memory gates exist and canary route selection is available; useful real traces are still too thin. | Convert repeated real assistant/repo/generator traces into monitored procedural candidates; use verified procedures as lookahead/trie sources without learned-generation credit. |

The active dependency order after this review is:

1. Preserve the book-derived control substrate and registry gates already
   present.
2. Repair direct learned body emission under the strict prompt/signature-only
   decoder profile.
3. Only after non-fallback top-level-return candidates exist, rerun bounded
   offline preference comparisons across DPO/IPO/ORPO/KTO/SimPO and then one
   verifier-reward RLVR/GRPO/RLOO/ReMax update with ablations.
4. Then test MTP, GVR/localized semantic repair, lookahead/trie, and
   sketch-first/diffusion as generation-mode candidates under accepted-output
   accounting.
5. Keep the product assistant useful through VCM, deterministic tools,
   planning, retrieval, and verifier receipts, while labeling that behavior as
   assisted product capability rather than learned model generation.

This reconciliation changes claims, not the no-cheat charter. It does not
authorize public benchmark training rows, runtime external inference,
fallback/template/router/tool credit as learned generation, or another loose
marker/slot/guard family. Each capability technique needs a substrate-adoption
record with baseline, negative control, matched data/compute, falsification
condition, residuals, and explicit non-claims before it can affect a default
route.

## 2026-07-06 Planned Codex Test Backlog Hardening

Claude's latest book-mining pass is now represented as machine-readable
roadmap debt, not only prose. `configs/roadmap_implementation_matrix.json`
contains `planned_codex_test_backlog`, and
`scripts/roadmap_implementation_gate.py` now validates that each item has an
owner phase, implementation track, dependency, acceptance gate, support-state
target, and no-claim boundary.

This matters because the ASI book's `Planned Codex test` lists are the real
technique backlog. They are not permission to spawn more lanes. They are
obligations that must be routed through the existing stack:

- Chapter 38 policy optimization: DPO/IPO/ORPO/KTO/SimPO first as bounded
  offline preference comparisons, then GRPO/RLOO/ReMax/RLVR only after the
  direct learned generator emits non-fallback candidates with behavior signal.
- Multi-target policy updates: generator, planner, router, VCM selector,
  verifier, executor, and generation-mode policies can change only through
  behavior-change leases with feedback source, drift bounds, rollback, monitor
  window, and task-specific heldouts.
- Fast generation: MTP, Medusa/EAGLE, speculative/self-speculative,
  early-exit/LayerSkip, lookahead/trie, Mamba/PagedAttention cache policy, and
  diffusion/LLaDA are generation-mode candidates measured by accepted verified
  output per second, not raw token speed.
- GVR and semantic IR: generated candidates must move through typed
  generated/verified/repaired/fallback/quarantine states, and failing verifier
  obligations should become localized semantic-atom repair targets rather than
  whole-body rerolls.
- VCM and verification bandwidth: context packets need representation
  certificates, authority ceilings, typed mandatory-miss faults, taint/deletion
  closure, copy-on-write lineage, verifier-capacity ledgers, and governance-tax
  accounting.
- Belief revision and procedural memory: claim ledgers must be able to
  downgrade/split/merge/contradict/retire claims immutably, and verified
  repeated trajectories must become monitored procedural tools and optional
  lookahead/trie sources without learned-generation credit.
- Circle/Coil/cyclic substrate work stays protected but falsifiable:
  cyclic-memory, KV-ring-buffer, recurrence, sparse-attention,
  circulant/block-cyclic mixer, and MultiCoil/RoPE features need matched dense,
  LoRA, RoPE, and Mamba baselines before any route or parity claim.

The active dependency remains strict: direct prompt/signature body-token
generation must produce non-fallback, nontrivial, loadable candidates with real
private behavior before larger DPO/GRPO/MTP/diffusion/scale work can be
interpreted as capability movement. Assisted tools, routers, semantic
renderers, deterministic repairs, templates, and fallback bodies remain useful
product machinery, but they cannot become learned-generation evidence.

## Book Complete Implementation Contract

This roadmap now treats `book_structure.json` as the authoritative inventory of
what Theseus must implement from the ASI Stack book. The current contract is
machine-readable in `configs/roadmap_implementation_matrix.json` and checked by
`scripts/roadmap_implementation_gate.py`.

"Beyond state of the art" does not mean optimistic prose, more report volume,
or a green private fixture. For this roadmap it means:

- the implementation is registry-owned and attached to a stable capability
  field;
- every important claim carries a support state: `argument`,
  `prototype-backed`, `synthetic-test-backed`, `empirical-test-backed`, or
  `replayable-reference-backed`;
- the implementation has valid and expected-invalid fixtures, independent
  replay/evaluator receipts, explicit non-claims, rollback or no-rollback
  behavior, and retained negative evidence;
- side effects cross an authority membrane with runtime-adapter receipts;
- tool-assisted/product behavior is useful but never counted as learned model
  generation;
- public benchmarks remain calibration-only and never become training rows.

The 44 authored chapters in `/Users/corbensorenson/Documents/AI_book/book_structure.json`
are now represented by `book_chapter_implementation_crosswalk`. Each row names
the book chapter, its minimal implementation, its beyond-SOTA endpoint, the
Theseus phases that own it, the support-state target, the required fixture or
gate, and the no-claim boundary. This is planning coverage, not a claim that
every chapter is implemented.

The active flagship lane is intentionally narrow:
`A1_claim_ledger_trace_kernel`. The next implementation focus should make one
normal Theseus task replayable through command contract, VCM packet, typed job,
runtime receipt, artifact graph, claim ledger, evidence transition, residual,
and update boundary. Other work may repair dependencies or maintain existing
surfaces, but it should not spawn another flagship lane.

Before long training or score-chasing becomes the primary roadmap focus again,
the book-reference core needs these slices to be prototype-backed or better:

1. `A1_claim_ledger_trace_kernel`: one support-stated, replayable,
   public-safe task trace with negative controls and no duplicate report
   family.
2. `A2_replacement_transaction_kernel`: one ordinary route or implementation
   swap through precheck, independent evaluator, regression, residual escrow,
   and rollback/no-rollback receipt.
3. `E1_authority_scif_runtime_adapter_kernel`: one side-effecting task with
   authority transition, adapter invocation, effect receipt, rollback residual,
   confused-deputy denial, and fake-secret handle proof.
4. `B1_assisted_verified_assistant_product_lane`: the local assistant path is
   useful under honest labels, with metadata-only dogfood outcomes and
   verifier receipts.
5. `C1_correctness_rl_and_generator_survival_lane`: one bounded private
   verifier-driven learned body-token experiment improves behavior or records
   a falsifying wall; no templates, routers, tools, public data, or fallback
   returns count as learned generation.

This preserves every late-book concept instead of deleting it. Authority
receipts, constitutional predicates, value conflicts, governance rights, VCM
transactions, resource routes, simulation contracts, semantic atoms,
compression/proof records, substrate-adoption records, and research backlog
items remain core Theseus obligations. The sequencing is strict only to prevent
new sidecars and report churn from replacing implementation.

## 2026-07-06 Weekly Focus Decision

Recommendation: spend the remaining weekly token budget primarily on Project
Theseus, while keeping the ASI book open as the specification, evidence ledger,
and publication surface. The right split is approximately 80% Theseus
implementation and 20% ASI book synchronization. The book is now strong enough
to tell Theseus what to build; Theseus is where the next unit of work can turn
the book's architecture into executable evidence.

This is not a shift away from the book. It is the fastest path to improve the
book without churning prose: make Theseus produce hard-to-fake, public-safe
evidence that the book can cite honestly. New ASI book work this week should be
limited to importing Theseus evidence packs, updating claim/evidence states,
recording non-claims, and tightening chapter claims that change because Theseus
produced a real result. Do not spend the week on broad chapter rewrites,
outline churn, or another review/report pass unless an implementation result
forces the change.

Theseus work should attack the implementation surfaces that make the book true
in practice:

1. Export one public-safe reference trace from a real Theseus run:
   `intent -> contract -> plan -> VCM packet -> route -> verifier -> execution -> artifact -> claim -> evidence -> residual -> improvement gate`.
2. Make the product-facing assistant path useful under honest labels: VCM,
   deterministic tools, retrieval, planning, verifier receipts, artifact refs,
   claim states, and dogfood outcomes should flow through one trace.
3. Standardize book-importable evidence packs for GREEN gates worth citing:
   command, digest, fixture/live-run boundary, negative controls, residuals,
   support state, and explicit non-claims.
4. Harden the evidence substrate before new model churn: receipt-faithfulness
   traps, residual conservation, verifier-capacity accounting, governance-tax
   measurement, capability-claim dispositions, and book-schema conformance.
5. Keep strict-generator work bounded to the next preregistered
   correctness-in-the-loop experiment. Do not add another marker/slot/guard
   family or public benchmark run until the implementation substrate above is
   current.

End-of-week success is not "more reports." It is at least one stronger
implementation trace, one stronger evidence/export path back to AI_book, and a
clearer product path Corben can actually use while learned-generation claims
remain honestly bounded.

- Weekly-focus implementation pass is now represented by
  `reports/theseus_weekly_focus_20260706.json` (`GREEN`): it refreshes the
  registered assistant product-spine run, exports
  `reports/theseus_public_safe_reference_trace_20260706.json`, exports
  `reports/theseus_book_importable_evidence_packs_20260706.json`, rejects `7/7`
  expected-invalid receipt controls, records residual-conservation,
  verifier-capacity, governance-tax, capability-claim-disposition, and
  book-schema-conformance audits, proves the active
  `A1_claim_ledger_trace_kernel` as `synthetic-test-backed`, and preregisters
  exactly one bounded correctness-in-the-loop generator experiment in
  `configs/correctness_in_loop_generator_experiments.json`. This is
  implementation-reference evidence, not a model-quality or ASI claim.
- Machine-readable roadmap state is governed by
  `configs/roadmap_implementation_matrix.json`. The current roadmap gate is
  `YELLOW` with `0` hard gaps: phase `0` is `implemented`; phases `4`, `5`,
  `6`, `7`, `8`, `11`, `12`, `17`, and `19` are `wired`; phases `3`, `10`,
  `13`, `14`, `15`, and `16` are `partial`; and phases `1`, `2`, `9`, and
  `18` are externally frozen until trusted peers are reachable. The active
  flagship core slice is `A1_claim_ledger_trace_kernel`, and the gate now
  enforces its current support state as `synthetic-test-backed`. The core-slice
  support state summary now records
  `A1_claim_ledger_trace_kernel=synthetic-test-backed`,
  `A2_replacement_transaction_kernel=synthetic-test-backed`,
  `E1_authority_scif_runtime_adapter_kernel=synthetic-test-backed`,
  `B1_assisted_verified_assistant_product_lane=synthetic-test-backed`, and
  `C1_correctness_rl_and_generator_survival_lane=synthetic-test-backed`.
- `scripts/roadmap_implementation_gate.py` now exposes a strict
  pre-training architecture readiness mode:
  `python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready`.
  The normal roadmap gate remains `YELLOW` with no hard gaps for implementation
  work, while strict readiness is currently `RED` because six book-derived
  phases remain partial: VCM transactional ABI/runtime parity (`3`), practical
  neural seed survival/policy optimization/generation modes (`10`), semantic
  IR localized repair (`13`), evidence hygiene/receipt faithfulness/claim
  revision (`14`), procedural-memory-to-lookahead/tool lifecycle (`15`), and
  verification-bandwidth/governance-tax routing (`16`). Phases `1`, `2`, `9`,
  and `18` are tracked as external-frozen with current network-doctor evidence:
  `coordinator_unreachable`, `registered_peers_unreachable`,
  `peer_inbound_only_outbound_blocked`, and `No route to host` for the trusted
  Windows coordinator. All five pre-training book-reference core slices still
  meet their current target support state, but the latest book-mining pass
  found real missing mechanics that must be implemented or falsified before
  the architecture is called ready for general training focus.
- Post-readiness execution is now governed by
  `configs/training_inference_execution_roadmap.json` and checked by
  `python3 scripts/training_inference_execution_plan_gate.py --gate`. This is
  the canonical bridge from "architecture ready" to actual training and
  inference work. It marks governed private training focus and a private MLX
  training smoke as ready; marks local assisted inference canaries as ready
  under honest labels; keeps bounded longer training planned until the smoke
  produces a clean checkpoint; keeps model-only general chat serving, public
  calibration, production MLX routing, and Hive fleet training blocked until
  their evidence conditions are met. The latest gate result is `RED` because
  strict architecture readiness is still `RED`. The plan explicitly rejects public
  benchmark training, runtime external inference, exact consumed public-surface
  reruns, fallback/template/router/tool credit as learned generation, raw
  private user text by default, and arbitrary remote execution.
  Current result: `reports/training_inference_execution_plan_gate.json` is
  `GREEN` with `0` failed checks and `0` failed expected-invalid controls.
- The E1 authority/SCIF runtime-adapter kernel is synthetic-test-backed:
  `reports/governance_rights_receipt_suite.json` proves one side-effecting
  assistant/tool fixture through runtime adapter invocation, authority
  transition/use receipts, effect receipt, rollback/no-rollback boundary,
  confused-deputy denial, Digital SCIF handle proof, expected-invalid controls,
  and clean no-cheat counters. This is a reference authority-membrane fixture,
  not a claim that every runtime route has universal deployed E1 enforcement.
- The B1 assisted verified assistant product lane is synthetic-test-backed:
  `reports/theseus_assistant_product_lane_gate.json` verifies the existing
  assistant runtime surface across `4/4` route cases, CLI/memory/feedback
  receipts, VCM readiness, deterministic tool evidence, private verifier
  receipts, `80` recent metadata-only dogfood events, `66`
  completed-or-accepted outcomes, `8/8` expected-invalid controls rejected,
  VIEA product trace records, and zero public-training/runtime-external/fallback
  counters. The gate preserves the strict code-generator semantic wall as C1
  negative evidence rather than laundering product usefulness into
  learned-generation capability. It does not claim real daily usefulness or
  empirical support until real multi-day user dogfood traces exist with raw
  private text off, verifier receipts retained, and accepted/missed/ignored/
  corrected/completed outcomes spread across real use days.
- The C1 correctness/RL/generator survival lane is synthetic-test-backed:
  `reports/correctness_generator_survival_lane_gate.json` verifies one bounded
  private verifier-driven learned body-token experiment under the preregistered
  correctness-in-the-loop contract. The fixture has `36` eligible
  transformer/hybrid candidates over `8` private tasks, independent integrity
  recomputation, selected compile/runtime-load rates of `0.375`, zero
  selected/pass-if-any functional behavior, zero promotion, and zero
  public-training/runtime-external/fallback/boundary counters. The gate now
  requires replay, integrity, blind-flow, generation-mode, policy, and `9/9`
  expected-invalid controls. This records a clean falsifying semantic wall, not
  a learned-generation win.
- The AI_book crosswalk is intentionally sticky: it now indexes `1703` AI_book
  source files, has `38` active backlog items, `0` stale-source phase
  candidates, `57` public-safe evidence pointers, and `136` active source-sync
  review decisions. This keeps book-to-Theseus follow-up visible instead of
  clearing it with superficial steward decisions.
- Phase 0 is implemented: the project registry gate is `GREEN` with no
  abstraction gaps, no SCF gaps, no implementation routing blockers, no hard
  registry governance violations, `12` routing-eligible implementations, and
  `16/16` cleanup queue items covered by active steward decisions. A bounded
  cleanup canary quarantined five old tmp/smoke artifacts with gzip checksums
  and no deletion.
- Registry freshness now separates current evidence from retained historical
  evidence. The old strict-generator MLX sweep/adaptation/decode reports and
  old Circle transfer diagnostics are retained for audit/mining, but no longer
  freshness-gate live routing. Current neural-seed route health is represented
  by the fresh private verifier spine smoke and strict fanout replay receipt.
- Phase 1 now has a working vertical slice and the active A1 reference-trace
  kernel is prototype-backed: the plan compiler emits ASI Stack records,
  `reports/viea_execution_spine.json` is `GREEN` on a bounded private execute
  path, assistant code/tool/planning calls require prompt-scoped VIEA trace
  records before passing, and `reports/theseus_weekly_focus_20260706.json`
  now hard-gates the weekly-focus reference trace on required trace records,
  support-state transitions, digest replay, source-to-verifier continuity,
  expected-invalid controls, duplicate-family avoidance, and clean no-cheat
  counters. Phase 1 is externally frozen only because the live reachable-peer
  Hive submission proof is separate and still requires a trusted peer.
- The VIEA route-validator view is `GREEN` across `31/31` producer profiles
  with `2227` materialized records, including `184` claim/proof entries,
  `8` compression records, and `1` defeater record. The Hive scheduler
  route-validator bootstrap cycle is now explicit and resolved: bootstrap is
  allowed only when required groups are present and no no-cheat counters fault,
  then the scheduler reruns against the green view and records a ready receipt.
- Phase 2 is externally frozen: typed-job records exist in plan/compiler
  and assistant traces, but the full local work-board and all private work do
  not yet use the artifact graph as the default execution queue. The A2
  replacement-transaction kernel is now prototype-backed through
  `reports/procedural_memory_route_adoption.json`: one guarded default-route
  adoption passes prechecks, independent toolification/canary/registry/steward
  evaluators, regression guard, residual escrow, rollback criteria,
  support-state transition, expected-invalid controls, and clean no-cheat
  counters. The live reachable-peer registered task receipt proof is still
  separate and requires a trusted peer.
- Phase 3 is materially improved: `reports/vcm_task_context_bridge.json` is
  `GREEN` with all `9` task families ready, and
  `reports/theseus_assistant_vcm_governor_smoke.json` is `GREEN` with a
  hard `vcm_context_governor_ready` gate. Product assistant traces now emit
  `context_transaction` and `context_adequacy` records with mission-brief,
  taint, deletion-closure, and no-cheat receipts. The standalone plan compiler
  also consumes `reports/vcm_context_governor.json`; `reports/theseus_plan_compiler.json`
  is `GREEN`, has zero failed gates, and records governed VCM adequacy for all
  `19/19` compiled nodes. The private verifier spine smoke now emits governed
  context records too: `reports/private_verifier_spine_smoke.json` is `GREEN`
  with `vcm_context_adequacy_state=governed_sufficient_for_verification`.
  Fanout/generator paths now emit the same governed context records. Phase 3 is
  wired in the matrix because the deployed VCM resolver now handles real
  semantic addresses, not only fixtures: `reports/vcm_context_governor.json`
  is `GREEN`, has `7/7` resolver requests passing, materializes `3` local
  artifact refs, emits `4` typed faults for blocked/missing/stale/unsafe
  routes, and contributes `56` resolver VIEA records. The deterministic tool
  substrate now carries its own VCM context receipt too:
  `reports/deterministic_tool_substrate.json` is `GREEN` with `13` local tools,
  `15/15` verified private smoke results, and `7` VCM tool context records.
  Training-data admission also consumes the same VCM governor receipt through
  `reports/training_data_admission_v1.json`, while keeping public benchmark
  payloads, raw user text, runtime external inference, and fallback rows out of
  admitted training paths. Native runtime evidence remains on the shared spine:
  `reports/vcm_native_runtime_probe.json` is `GREEN`, proves a backend-scoped
  CPU Transformers DynamicCache prefix/KV lifecycle, proves MLX resident tensor
  descriptor reuse/invalidation, allows scheduler VCM descriptor routing for
  the recommended MLX backend, and keeps scheduler native KV routing
  fail-closed until model-native MLX KV/prefix lifecycle is proven in Phase 8.
- Phase 10 now has a canonical strict-generator replay receipt:
  `reports/neural_seed_strict_generator_fanout_receipt.json` is `GREEN` after
  replaying the current strict body-token candidate manifest through
  independent candidate-integrity recomputation and the private verifier. It
  measures `93` integrity-verified learned full-body candidates across `22`
  private eval tasks, syntax-valid rate `0.684211`, runtime-load task rate
  `1.0`, and intended behavior pass rate `0.227273`. This narrows the wall to
  semantic behavior quality. A larger private candidate cap (`8` per task)
  preserved the same `0.227273` behavior pass rate, so the wall is semantic
  candidate quality rather than fanout width. The receipt now carries a blind
  residual diagnosis with `17` runtime-loaded failed tasks, `0` no-load tasks,
  and dominant failed issue labels `blind_static_shape_plausible`,
  `prompt_implies_branching_but_no_branch`,
  `prompt_implies_structured_output_but_no_collection_construction`, and
  `prompt_implies_string_processing_but_no_string_ops`. The selector ablation
  now compares baseline rank, prompt/AST-only blind rank, and full-pool
  pass-if-any oracle; all stay at `5/22` behavior passes, with
  `baseline_to_oracle_pass_delta=0`, so the next wall is candidate-pool
  semantic construction rather than fanout width or selector quality. No public data, teacher calls, fallback returns,
  fixed renderers, routers, tools, or templates are credited as learned
  generation.
- C1 is now explicitly bound to that wall by
  `reports/correctness_generator_survival_lane_gate.json`. The stricter
  transformer/hybrid replay fixture verifies `36` integrity-clean candidates
  across `8` private tasks, selected compile/runtime-load rates of `0.375`, and
  selected/pass-if-any behavior `0.0`. This supports the book-reference C1 slice
  as a falsification fixture and keeps learned-generation promotion blocked
  until the generator improves semantic candidate construction without
  templates, routers, deterministic tools, fallback returns, public data, or
  hidden answer-derived fields.
- Phase 10 also now has a named MLX private repair profile:
  `strict_full_body_semantic_construction_v1` in
  `scripts/strict_generator_mlx_private_adaptation.py`. The profile composes
  existing private-only source-contrastive, semantic-plan, semantic-slot,
  loop-update, expression-synthesis, plan-conditioned body, update-contract,
  return-expression, and primary-dataflow losses. It fails closed on the old
  plain body-token checkpoint because semantic-plan/slot targets are absent,
  and it passes GREEN on the semantic-slot MLX checkpoint in
  `reports/strict_generator_mlx_private_adaptation_semantic_construction_profile_semantic_slot_private_smoke_v1.json`.
  The GREEN smoke activates all required components: semantic-plan loss and
  source-contrastive gap both improve, visible-operation plan rows boost (`37`),
  and semantic-slot/loop/expression/plan-conditioned/update-contract token
  weighting all match private target spans. The follow-up broad-private decode
  smoke
  `reports/strict_generator_mlx_decode_eval_semantic_construction_profile_broad4_v1.json`
  is still RED: no accepted strict candidate rows, `0/4` behavior pass,
  nontrivial-return rate `0.0`, and top learned beams starve inside repeated
  loop-update prefixes with `inside_loop_without_update` and
  `missing_local_return`.
- Phase 10 now also has the narrower direct-body emission profile
  `strict_direct_body_emission_path_v1`. It adds private-only supervised
  weighting for exact AST spans on the reachable body path: top-level state
  bindings, branch guards, loop headers, loop body state transitions,
  local-state returns, and nontrivial return expressions. The bounded smoke
  `reports/strict_generator_mlx_private_adaptation_direct_body_emission_path_smoke_20260706.json`
  is `YELLOW`: `128/128` private rows matched, `8054` token positions were
  weighted, heldout LM loss improved from `1.766813` to `1.467117`, semantic
  plan and source-contrastive metrics improved, and no public/external/fallback
  counters moved. The required strict replay
  `reports/strict_generator_mlx_decode_eval_direct_body_emission_path_broad4_replay8_20260706.json`
  remains `RED`: `0` accepted learned candidate rows and `0/8` behavior pass.
  This is useful negative evidence. It shows target-side direct body-emission
  weighting is active, but decode still needs a trainable local-return/
  finalizer continuation mechanism that admits non-fallback candidates; another
  scalar CE-weighting profile is not enough by itself.
- The first body-transition guard follow-up,
  `reports/strict_generator_mlx_decode_eval_body_transition_guard_broad4_v1.json`,
  removes that starvation mode on the same broad-private slice without public
  data, teacher calls, templates, tools, fallback returns, or renderer credit:
  `8` generated transformer/hybrid candidates are integrity-clean, verifier
  labels attach, and zero-candidate tasks fall from `4/4` to `0/4`. This is
  still negative capability evidence, not promotion evidence: behavior remains
  `0/4`, nontrivial-return rate remains `0.0`, and the residual labels are now
  `loop_without_decision_or_state_update` plus `missing_semantic_update_value`.
  The next patch is learned semantic update choice and nontrivial local-return
  synthesis, not another selector, fanout-width, or generic semantic-weighting
  pass.
- Phase 10 negative replay now consumes independent candidate-integrity
  failures as private repair pressure. The patch extends the existing
  strict-generator MLX adaptation path rather than adding a new lane:
  `reports/strict_generator_mlx_private_adaptation_source_condition_operation_integrity_negative_replay_smoke_v1.json`
  is `GREEN` with `16` failed private-train replay rows, `4` recomputed
  no-function/syntax-invalid integrity-negative rows, `12` integrity-verified
  failed rows, `0` public training rows, `0` external inference, and `0`
  candidate-generation credit. The paired broad-private heldout smoke
  `reports/strict_generator_mlx_decode_eval_source_condition_operation_integrity_negative_replay_broad4_v1.json`
  remains `YELLOW`: `6/7` integrity-verified transformer/hybrid candidates,
  one no-function/syntax mismatch, nontrivial-return rate `0.857143`, and
  behavior `0/4`. That keeps the honest wall semantic candidate quality, while
  making malformed full-body outputs first-class private negative evidence.
- `reports/neural_seed_survival_readiness_gate.json` now wires Phase 10 as
  architecture readiness for the next governed private training/adaptation
  pass. It is `GREEN` with `93` eligible full-body learned candidates,
  runtime-load task rate `1.0`, current behavior pass rate `0.227273`,
  top-8 behavior pass rate `0.227273`, C1 selected/pass-if-any behavior `0.0`,
  `11/11` readiness checks passing, and `8/8` expected-invalid controls
  passing. This is not a model-quality, public-transfer, or promotion claim, and
  it does not run training, teacher inference, or public calibration. The next
  governed step is private semantic update/final-return adaptation on the
  existing transformer/hybrid survival lane.
- Phase 12 is wired: `reports/public_calibration_proposal_gate.json` is a
  VIEA-gated public-calibration proposal receipt. It requires candidate
  integrity, training-data firewall, alignment preflight, and exact public
  run-registry state before a calibration result can count as evidence. The
  current default surface is refused because it is already consumed; fresh
  frozen surfaces remain eligible without calendar throttles when clean.
- Phase 13 is wired: `reports/semantic_ir_obligation_gate.json` binds
  candidate-integrity, private-verifier, and direct-generator obligations to
  the materialized semantic IR view, with `25` semantic atoms, `93` semantic
  nodes, `3/3` ready consumers, `3` semantic-obligation records, `3`
  dependency edges, and `3` evidence bindings.
- Phase 17 and Phase 18 now have materiality beyond schema shape:
  `reports/governance_rights_receipt_suite.json` is `GREEN` with `4/4`
  governance-right fixtures, `4/4` constitutional-predicate fixtures, and `4`
  VIEA `constitutional_predicate` records for least sufficient power, predicate
  conflict routing, constitutional migration, and self-modification weakening
  rejection. `reports/simulation_fidelity_receipt_suite.json` is `GREEN` with
  `5/5` simulation/fidelity fixtures plus one bounded planning-world adapter
  over the current compiled plan DAG. These emit canonical VIEA governance,
  constitutional, failure, simulation, fidelity, counterfactual, world-adapter,
  claim, artifact, and evidence records. They are not institutional governance,
  moral correctness, physical feasibility, benchmark transfer, live simulator,
  deployment, or learned-generation claims.
- Phase 18 now also has a live local operator audit/export receipt:
  `python3 scripts/hive_node.py operator-governance-audit --out reports/hive_operator_governance_audit.json`
  is `GREEN` with `13` audit refs, `6/6` applicable local artifact payload
  citations passing, `1` governance-right record, `1` authority-use receipt,
  `1` failure-boundary record, `13` artifact-graph records, `1` claim record,
  `13` evidence-transition records, and zero public training rows/runtime
  external inference/fallback returns. `/api/hive/operator/governance-audit`
  and the mobile operator UI expose the same request path. The remaining Phase
  18 wall is trusted-peer artifact endpoint proof, not local operator rights
  receipt materialization.
- The daily-use assistant/dogfood trace loop is wired in the machine matrix:
  `reports/theseus_assistant_runtime.json` is `GREEN` with VCM ready,
  assistant VIEA trace complete, dogfood bridge `GREEN`, and zero public
  training rows/runtime external inference/fallback returns. The latest
  assistant e2e run has `4/4` cases passing, `80` recent trainable events,
  `66` completed-or-accepted recent events, `397` local dogfood events, and
  `396` governed dogfood trace training rows. The remaining warning is the
  honest code-generator wall: private replay is safe but selected/pass-if-any
  semantic pass is still `0.0`.
- Phase 6 is wired in the machine matrix: deterministic tools/search are
  registry-backed, VCM-bound, replayable, and separately accounted from learned
  generation. `reports/deterministic_tool_substrate.json` is `GREEN` with `13`
  local tools and `15/15` verified private smoke results. The normal plan
  compiler now declares tool eligibility for `25/25` compiled nodes, persists
  tool receipts for `25/25` nodes, emits `71` tool-call receipts, and the
  shared VIEA profile requires `tool_call_receipt` records on planner nodes.
- Teacher/data governance now has a durable teacher-share view:
  `reports/teacher_share_ledger_summary.json` is `GREEN`, metric-ready, and
  consumed by the control plane, assistant runtime, and Hive operator/mobile
  status payload. Current share is
  `3/64152 = 0.00004676393565282454` accepted teacher rows over accepted
  training rows, with `7` verified self-generated rows, `10` training-time
  teacher calls recorded, `0` runtime external inference calls, `0` public
  training rows, and `0` fallback returns. `reports/hive_operator_status.json`
  exposes the same `teacher_governance` state with governance-right fixture
  counts and no-cheat counters. This is governance/accounting evidence, not
  model capability evidence.
- Strict generator promotion evidence now has a fail-closed source contract:
  `prompt_signature_only_v1` emits only prompt, entry point, visible
  argument-count signature, identifier pieces, and tokenizer subword repair.
  `reports/strict_generator_mlx_decode_eval_prompt_signature_broad8_v1.json`
  is `YELLOW`: source audit clean, no public training, no runtime external
  inference, no fallback returns, and zero candidate-integrity mismatches, but
  `0/8` intended-behavior pass. This makes the next wall semantic full-body
  generation, not source leakage or syntax/admissibility.
- Strict MLX generator follow-through has now separated three problems:
  source/rehearsal hygiene, candidate replay/loadability, and semantic
  correctness. The prompt/signature source builder now allows only
  prompt-derived intent/type-shape tags beyond the original visible fields;
  rehearsal rows are rebuilt through that same builder; and explicit `0.0`
  guard-loss weights are honored. The invalid rehearsal report
  `reports/strict_generator_mlx_private_adaptation_prompt_signature_tags_v1.json`
  remains a useful RED artifact because it caught legacy source contamination.
  The clean guard report
  `reports/strict_generator_mlx_private_adaptation_prompt_signature_tags_guard_clean_v2.json`
  is `GREEN`, and the broad decode report
  `reports/strict_generator_mlx_decode_eval_prompt_signature_tags_guard_clean_v2_search_semantic_head_broad8_labelled_v2.json`
  emits `34` integrity-clean candidates and attaches private verifier
  correctness labels to all `34` generated candidates. Those labels show
  `34` runtime-loaded generated attempts and `0` generated intended-behavior
  passes, with residual classes concentrated in runtime, type-handling, and
  wrong-answer failures. It still scores `0/8` behavior, so Phase 4 should now
  consume those verifier labels in a preference/rejection objective rather than
  adding more source/syntax scaffolding.
- Phase 4 now also has a private-train correctness replay hook. The
  `private_train_replay` split copies private train rows into a verifier-only
  replay view, excludes configured family-disjoint holdout families, and marks
  the evidence as training-signal replay rather than heldout transfer. The
  strict-guard smoke
  `reports/strict_generator_mlx_decode_eval_private_train_replay_labelled_smoke_v1.json`
  failed closed with no admissible candidates. The no-guard replay smoke
  `reports/strict_generator_mlx_decode_eval_private_train_replay_labelled_no_guard_smoke_v1.json`
  emitted `16` private-train replay candidates, attached labels to every
  generated row, and showed `8` runtime-loaded attempts, `8` lint/parse
  failures, `0` intended-behavior passes, and integrity mismatches. This is the
  correct negative-signal source for a guarded rejection/unlikelihood training
  objective; it must not be reported as heldout performance.
- Phase 4 now has that first guarded negative-signal objective. The high-weight
  smoke
  `reports/strict_generator_mlx_private_adaptation_negative_replay_smoke_v1.json`
  failed correctly because heldout LM loss worsened. The lower-weight smoke
  `reports/strict_generator_mlx_private_adaptation_negative_replay_smoke_low_weight_v1.json`
  is `GREEN`: it used `16` failed private replay candidates as bounded
  unlikelihood pressure, improved heldout LM loss from `2.552511` to
  `1.715164`, and preserved `0` public training rows / `0` external inference.
  Decode from that checkpoint in
  `reports/strict_generator_mlx_decode_eval_negative_replay_smoke_low_weight_private_train_replay_v1.json`
  is `YELLOW`: `64` generated candidates, integrity verified on `48`, syntax
  pass `0.8125`, runtime load rate `0.708333`, inert-stub rate `0.359375`, but
  still `0` intended-behavior passes. The current wall is semantic/dataflow
  correctness after loadability, not negative-label plumbing.
- A verifier-stage-weighted replay ablation has now been tested. The adaptation
  report
  `reports/strict_generator_mlx_private_adaptation_negative_replay_stage_weighted_smoke_v1.json`
  is `GREEN`, but its decode report
  `reports/strict_generator_mlx_decode_eval_negative_replay_stage_weighted_private_train_replay_v1.json`
  regressed syntax, runtime-load, integrity, and nontrivial-return rates while
  keeping semantic pass at `0`. This means naive reward-inverse weighting is
  not the next route. The next correctness-in-the-loop objective should use
  richer positive/negative structure, not merely punish low-stage failures
  harder.
- The existing private train set now has an explicit replay tier selector
  inside `scripts/strict_generator_mlx_decode_eval.py`; this is a selection
  view over existing private rows, not a new synthetic benchmark family. After
  family-disjoint holdout exclusion the inventory is `125` simple-return rows,
  `1250` loop-accumulate rows, and `1625` algorithmic-small rows. The current
  uniform negative-replay checkpoint on the simple-return tier
  (`reports/strict_generator_mlx_decode_eval_negative_replay_low_weight_simple_return_replay_v1.json`)
  produced `64/64` integrity-verified candidates, syntax pass `1.0`, runtime
  load `0.833333`, and still `0` intended-behavior passes, with failures
  concentrated in type/return handling. The next Phase 4 target should be
  return/type semantics on this tier before another broad replay run.
- The first return-expression-targeted simple-return adaptation is complete.
  `reports/strict_generator_mlx_private_adaptation_simple_return_return_expr_v1.json`
  is `GREEN` on existing private simple-return rows only, with heldout LM loss
  improving from `2.289181` to `0.095394` and no public/teacher/runtime
  inference use. Its decode report,
  `reports/strict_generator_mlx_decode_eval_simple_return_return_expr_v1.json`,
  keeps `64/64` integrity verification, syntax pass `1.0`, and runtime load
  `0.833333`; it moves nontrivial-return rate from `0.0` to `1.0`, but still
  has `0/64` intended-behavior pass. This is real shape progress, not a
  promotion: the current blocker is wrong-answer/value semantics from prompt
  and signature, not fallback returns, syntax, or loadability.
- Follow-up value-semantics hooks have been implemented and tested. The
  same-source private pairwise replay objective is GREEN as a training
  mechanism on `64` failed private replay candidates, but
  `reports/strict_generator_mlx_decode_eval_simple_return_pairwise_value_v1.json`
  remains `0/64`, so naive accepted-vs-rejected margin pressure is not enough.
  The visible-default return weighting hook also trains cleanly, and the
  top-level-return decode guard makes candidates include `return other`, but
  the model still fails to emit the `and data` guard required by the visible
  empty/default prompt. The combined default/truthiness weighted run is GREEN
  but
  `reports/strict_generator_mlx_decode_eval_simple_return_default_truthiness_top_level_guard_v1.json`
  remains `0/64`. The next Phase 4 target is condition construction and
  efficient guarded beam/search for prompt-visible empty/default behavior; do
  not treat any of these weighted private smokes as promotion evidence.
- Phase 4 has a narrow constrained-control breakthrough, with the boundary
  explicitly recorded. `scripts/strict_generator_mlx_decode_eval.py` now has
  a prompt/signature-only source-condition expectation and stricter adequacy
  audit for visible empty/default sequence contracts. The first polarity run
  (`reports/strict_generator_mlx_decode_eval_simple_return_source_condition_polarity_prefer_v1.json`)
  remains negative evidence because the old adequacy check accepted tuple-only
  and overlong boolean-return candidates while behavior stayed `0/64`. The
  ordered sequence run
  (`reports/strict_generator_mlx_decode_eval_simple_return_source_condition_sequence_ordered_prefer_v1.json`)
  failed closed until mandatory indentation was restored before
  source-condition preference and verifier-label gates were aligned to
  evaluated traces. The current bounded report
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_sequence_indent_strict_v1.json`
  is `GREEN`: `16/16` rank-1 and pass-if-any on existing private
  simple-return replay rows, `32` integrity-verified transformer/hybrid
  candidates, source audit clean, no public training, no runtime external
  inference, no fallback returns, and `candidate_generation_credit=0` for the
  constraint path. This is useful as a control and scaffolding target, not as
  learned free-generation evidence. The next Phase 4 goal should distill this
  branch/value behavior into the learned token probabilities/internal plan head
  and then verify it on disjoint private families without deterministic
  source-condition assistance.
- The first source-condition internalization attempt is complete and negative
  in the way that matters. The private-only adaptation
  `reports/strict_generator_mlx_private_adaptation_simple_return_source_condition_internalize_v1.json`
  is `GREEN`: it boosted `2,600` prompt-visible condition target-token
  positions across `100` admitted private simple-return rows, improved heldout
  LM loss, and kept source audits clean with no public data, no external
  inference, and no candidate-generation credit. The no-assist decode
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_internalize_no_assist_v1.json`
  remains `0/16` behavior. It improves some visible sequence/indexing shape
  but still emits no `and data` truthiness guard and no adequate
  source-condition candidate without deterministic assistance. The next Phase
  4 work should focus on learned planning/decision pressure for guard
  placement and branch condition choice, not just heavier CE weights on the
  same tokens.
- The plan-head portion is now separated from the body-token wall. The
  private-only plan-auxiliary adaptation
  `reports/strict_generator_mlx_private_adaptation_simple_return_source_condition_plan_aux_v1.json`
  is `GREEN`: heldout semantic-plan accuracy moves from `0.0` to `1.0`, with
  all train/eval rows targeting `SLOT:PLAN_SAFE_HEAD_DEFAULT`. The no-assist
  metadata decode
  `reports/strict_generator_mlx_decode_eval_simple_return_source_condition_plan_aux_no_assist_planmeta_v1.json`
  emits a complete `SLOT:PLAN_SAFE_HEAD_DEFAULT` prefix for `64/64`
  candidates, but remains `0/16` behavior with `0/64` adequate
  source-condition candidates and `0` `and data` guards. The current Phase 4
  target is therefore plan-conditioned body generation: operand choice,
  truthiness guard placement, and default-return expression learning under the
  strict no-fallback/no-template/no-tool-credit boundary.
- The exact learned decision-prefix rung is now implemented under that same
  contract. The mode `plan_semantic_slots_body_tokens_v1` emits learned
  decision slots before body tokens, strips them before Python compilation,
  and records `candidate_generation_credit=0` for the prefix/ranker
  constraint. The exact prefix checkpoint/adaptation reports are
  `reports/strict_generator_mlx_pretraining_probe_plan_semantic_slots_body_exact_smoke_v1.json`
  and
  `reports/strict_generator_mlx_private_adaptation_simple_return_plan_semantic_slots_body_exact_decision_v1.json`.
  The strict no-assist smoke remains `0/4`, but the opt-in prefix-guided and
  prefix-ranked diagnostic
  `reports/strict_generator_mlx_decode_eval_simple_return_plan_semantic_slots_body_exact_decision_prefix_guided_ranked_v2.json`
  reaches `16/16` rank-1 and pass-if-any on private simple-return replay with
  no public data, no external inference, and no fallback returns. This is a
  practical transformer/hybrid survival-lane improvement, but not an
  unconstrained learned body-generation claim. The next Phase 4 wall is to
  generalize the learned-prefix/ranker result beyond `simple_return` and to
  make the body decoder consume those decisions with less constrained beam
  pressure.
- The multi-tier follow-through has now isolated that wall more sharply.
  `scripts/strict_generator_mlx_private_adaptation.py` supports deterministic
  tier-balanced private adaptation sampling, and the strict decoder now records
  max-target overrides while preventing incomplete exact-prefix body starts,
  invalid loop targets, and mismatched closing brackets. The balanced private
  checkpoint
  `reports/strict_generator_mlx_private_adaptation_multi_tier_balanced_plan_semantic_slots_body_exact_decision_v1.json`
  is `GREEN`, and the stricter simple-return replay
  `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_prefixrequired_v1.json`
  is `GREEN` with `16/16` rank-1 and pass-if-any, syntax `1.0`, no public
  data, no external inference, and no fallback returns. The same path remains
  `YELLOW` on loop/algorithmic private replay: loop-focused source-contrastive
  adaptation trains cleanly on `900` loop rows and improves heldout source
  contrast, but
  `reports/strict_generator_mlx_decode_eval_loop_accumulate_loop_focused_source_contrast_plan_semantic_slots_body_exact_decision_cap160_v1.json`
  still has `0/8` behavioral pass and slow decode. The next Phase 4 target is
  learned loop/action body semantics and faster candidate-expression search,
  not another simple-return proof or public calibration.
- Pairwise replay has been upgraded from a narrow plain-loss ablation into a
  composable private objective: it can now run together with semantic-plan,
  source-contrastive, and negative-replay losses instead of reporting
  requested-but-inactive. The composite report
  `reports/strict_generator_mlx_private_adaptation_loop_focused_composite_pairwise_plan_semantic_slots_body_exact_decision_v1.json`
  confirms pairwise, source-contrastive, semantic-plan, and negative replay
  all active under no-public/no-eval/no-credit rules, but its decode report
  `reports/strict_generator_mlx_decode_eval_loop_accumulate_loop_focused_composite_pairwise_plan_semantic_slots_body_exact_decision_cap160_v1.json`
  is RED with `0` accepted candidates. A lighter pairwise-only composition,
  `reports/strict_generator_mlx_private_adaptation_loop_focused_pairwise_only_plan_semantic_slots_body_exact_decision_v1.json`,
  stays `GREEN` and keeps pairwise active without negative-unlikelihood, but
  `reports/strict_generator_mlx_decode_eval_loop_accumulate_loop_focused_pairwise_only_plan_semantic_slots_body_exact_decision_cap160_v1.json`
  remains `0/8`, producing only one valid wrong candidate. The simple-return
  regression
  `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_regression_v2.json`
  remains `GREEN` at `8/8`. Conclusion: the current wall is not inactive
  pairwise plumbing; it is learned loop/action body construction and search.
- ATT-D hard source cap is improved: the neural token decoder comparator was
  split into `scripts/neural_seed_visible_source.py` and
  `scripts/neural_seed_static_coherence.py`, while preserving the old
  comparator import surface. The strict static decode guard is now in
  `scripts/neural_seed_decode_static_guard.py`; shared candidate schema,
  no-cheat evidence, grammar/static/body summaries, and token-decoder gate
  predicates are in `scripts/neural_seed_candidate_evidence_summary.py`;
  expression-value hygiene is in `scripts/neural_seed_expression_value_guard.py`;
  governed teacher/self code-LM row admission helpers are now in
  `scripts/neural_seed_teacher_distillation_rows.py`; semantic route-memory
  helpers for learned plan prototypes, contract fingerprints, contract
  features, and visible text prototypes are in
  `scripts/neural_seed_route_memory.py`; private target-body guard weighting
  and existing-row tier selection for strict MLX adaptation are in
  `scripts/strict_generator_mlx_adaptation_selection.py`; private replay and
  family/broad heldout row selection are in
  `scripts/strict_generator_mlx_replay_selection.py`; prompt/signature-only
  strict MLX source-text construction and leakage audits are in
  `scripts/strict_generator_mlx_source_text.py`; strict MLX decode row
  stamping, inline integrity summaries, gate construction, checkpoint path
  resolution, and report IO are in
  `scripts/strict_generator_mlx_decode_reporting.py`; specialist-head
  route/profile records for private replay checkpoints are in
  `scripts/strict_generator_mlx_specialist_routing.py`; task-blind strict
  body-token decode hygiene and call/method guards are in
  `scripts/strict_generator_mlx_decode_guards.py`; local corpus admission,
  full-state pretraining row construction, quality filtering, and vocab
  extension summaries are in `scripts/neural_seed_full_state_pretraining.py`.
  Matched token decoder backends, checkpoint/vocab initialization, supervised
  token training, grammar auxiliary loss, semantic token weighting, and
  parameter-update summaries are in `scripts/neural_seed_token_model_backend.py`.
  Task-blind token beam expansion/sorting is in
  `scripts/neural_seed_candidate_generation.py`; report/latest-view IO and
  markdown rendering are in `scripts/neural_seed_report_io.py`. Prompt-visible
  source-condition expectations, learned prefix adequacy, loop-plan adequacy,
  expression closure, and body action-trace summaries are now in
  `scripts/strict_generator_mlx_decode_plans.py`. The comparator is now
  `3,196` lines and the strict MLX decoder is now `2,510` lines, both below
  the `3,200` system-efficiency hard limit. Private strict-generator target
  weighting and AST span extraction are now in
  `scripts/strict_generator_mlx_adaptation_weights.py`, which moves
  `scripts/strict_generator_mlx_private_adaptation.py` down to `3,051` lines.
  Token-decoder rendering, strict-action rendering, semantic-slot rendering,
  deterministic semantic bodies, and body-token decoding are now in
  `scripts/neural_seed_token_decoder_rendering.py`, moving
  `scripts/neural_seed_token_decoder_support.py` down to `2,832` lines. There
  are no remaining RED source-size blockers. The strict MLX generator files are
  registry-owned by the existing
  `neural_seed_and_decoder` surface and ATT-D classifies the decode path as
  `neural_seed`; `scripts/attd_analyzer.py` is `YELLOW` with hard caps passed
  and no violations. `reports/system_efficiency_audit.json` is now `YELLOW`
  with `0` hard maintainability hotspots; the remaining cleanup target is soft
  source-size pressure, led by `crates/symliquid-cli/src/main.rs`, plus rolling
  residue and duplicate family pressure.
- The refactored strict MLX decode path has an executed private smoke:
  `reports/strict_generator_mlx_decode_eval_refactor_smoke_v1.json` is
  `YELLOW` after running one family-disjoint private row on MLX. The smoke
  loads the checkpoint, emits a candidate, keeps hard no-cheat gates passing
  (`0` external inference calls, `0` public training rows, no fallback
  returns), and correctly marks the candidate as integrity-failing/inert rather
  than laundering it into learned-generation evidence.
- Blind information-flow coverage now includes the extracted neural-seed
  modules (`visible_source`, `static_coherence`, `decode_static_guard`,
  `expression_value_guard`, `candidate_evidence_summary`, and
  `teacher_distillation_rows`) in addition to the comparator/support/generator
  entrypoints and strict MLX decode/adaptation/pretraining entrypoints. The
  expanded 28-file default audit remains `GREEN`, so the split and strict MLX
  path do not hide forbidden-field or learned-generation overclaim risks from
  the default no-cheat audit.

Current control-plane wall: stale evidence is no longer the issue. The real
blockers are public transfer, prompt/signature-only learned-generator semantic
quality, promotion coherence, iteration speed, and assembly debt.

## Current Diagnosis

Theseus has the right ingredients, but too many are not yet forced through one
canonical execution spine.

The live strong pieces are:

- A serious operating charter in `AGENTS.md`.
- A registry and SCF direction in `configs/project_manifest_registry.json` and
  `docs/PROJECT_REGISTRY.md`.
- VCM context machinery.
- A plan compiler that can build typed, VCM-backed DAGs.
- A deterministic tool substrate direction.
- Candidate integrity and blind information-flow guardrails.
- A governed teacher path and dogfood metadata path.
- A real Apple Silicon MLX training lane.
- A local assistant runtime that composes checkpoint chat, VCM, tools, planning,
  code-route metadata, and dogfood feedback.

The current walls are:

- `scripts/theseus_project_registry.py --gate` is GREEN: abstraction gaps,
  SCF gaps, implementation routing blockers, and hard registry governance
  violations are all currently `0`. Registry hygiene should stay active, but it
  is no longer the primary blocker.
- `scripts/theseus_workspace_hygiene_audit.py` can still report workspace
  churn, duplicate/generated families, or cleanup queue pressure. Treat that as
  iteration drag, not the central capability wall.
- The strict generator wall is semantic correctness, not throughput. The MLX
  lane trains quickly, can emit syntax-valid non-stub candidates, records zero
  integrity mismatches, and now reaches useful candidate runtime-load on the
  clean guard path, but the corrected prompt/signature-only broad smoke is
  still `0/8` intended-behavior pass and the latest larger private replay
  evidence remains at zero semantic pass.
- The model training objective is still too close to language modeling. It does
  not yet put verifier correctness, role selection, return semantics, and
  algorithm choice into the learning loop strongly enough.
- The assistant runtime exists, but it is not yet the daily-use product quality
  bar. Theseus needs to be useful in conversation and coding while it improves.
- The book's ASI Stack concepts exist in pieces, but Theseus does not yet emit
  one end-to-end trace for every meaningful task:
  `intent -> command -> plan -> context -> job -> adapter -> artifact -> claim -> evidence -> residual -> policy update`.
- Reports are still abundant enough that evidence can become noisy. The repo
  needs fewer, stronger, registry-owned reports.

## Non-Negotiable Rules

These rules are the implementation interpretation of the book plus `AGENTS.md`.

- Public benchmarks are calibration only. Do not train on public benchmark
  prompts, tests, hidden tests, solutions, traces, score labels, or answer
  templates.
- Teacher rows are allowed only when governed, provenance-tagged,
  license-checked, verifier-accepted, leakage-audited, and forbidden from
  runtime serving.
- External inference is never served to a user. It is only teacher-side
  training support.
- No fallback returns for capability credit.
- Routers, templates, deterministic tools, semantic renderers, and action
  catalogs are useful tools and baselines, but they are not learned generation.
- Candidate family and integrity must be recomputed independently.
- Every routeable capability must have an abstraction, implementation, stable
  capability field, evidence output, rollback rule, and cleanup policy.
- Prefer improving a registered surface before adding a new one.
- If a result is negative, keep it. Negative evidence is part of the system.

## The ASI Stack Crosswalk

| Book concept | Theseus should do | Current gap | Roadmap phase |
|---|---|---|---|
| Stable Capability Fields | Make each capability a stable field with semantic boundary, interface contract, authority ceiling, evidence, route validity, migration, and rollback. | Registry/SCF exists, but evidence for live implementations is stale and routing eligibility is mostly blocked. | Phase 0, Phase 1 |
| Replacement and rollback | Promote implementations only through replacement transactions with regression checks, residual escrow, and rollback. | Registry has lifecycle language, but replacement transactions are not yet the ordinary route for model/router/tool changes. | Phase 1, Phase 6 |
| Recursive self-improvement boundaries | Treat improvement as bounded transitions over fields with evaluator independence. | Improvement runs exist, but the canonical self-improvement transition record is not yet the universal trace shape. | Phase 1, Phase 6 |
| Intent-to-execution contracts | Compile accepted user intent into command contracts before work. | Plan compiler has contracts, but the assistant/runtime path does not yet require them for every meaningful task. | Phase 1, Phase 2 |
| PlanForge DAGs | Decompose goals into typed DAGs with MVI routing, context requests, verification burden, and cost-quality ledger. | Plan compiler exists and can smoke execute, but it is not the default execution spine for all private work. | Phase 2 |
| Cognitive compilation | Use semantic IR so failures repair localized plan/code/context nodes. | Partial plan/code structures exist, but model failures are still reported more than repaired through IR-level feedback. | Phase 3 |
| VCM ABI | Make context addressable, versioned, mounted, snapshotted, tainted, and adequacy-scored. | VCM is useful, but it is not yet mandatory across assistant, training, verifier repair, and artifact routing. | Phase 2, Phase 3 |
| Semantic pages and certificates | Store context as certified semantic cells with provenance, omissions, validity, permitted uses, and risks. | Context packets exist, but certificate discipline is not yet universal. | Phase 3 |
| Verification bandwidth | Route work based on how hard it is to verify, not just how hard it is to generate. | Verifiers exist, but training still optimizes fluency before correctness enough to stall. | Phase 4 |
| Claim ledgers and belief revision | Every claim has a support state, evidence bundle, contradictions, and transition record. | Reports contain evidence, but claims and state transitions are not yet first-class for every run. | Phase 1, Phase 5 |
| Proof-carrying claims | Strong claims should attach proof, executable checks, or verifier receipts. | Circle proof-carrying contract receipts are now schema-shaped and VIEA-materialized for structural/proof-boundary claims; downstream utility, model-quality, speed, memory, and transfer claims still require separate Theseus task evidence. | Phase 5 |
| Tribunal/adversarial review | Important promotions get independent checks, dissent, residuals, and contradiction handling. | Candidate integrity is strong, but broader tribunal records are not yet standard for architecture changes. | Phase 5, Phase 6 |
| Labor OS typed jobs | Execution runs as typed jobs with permissions, outputs, audit events, failure behavior, and replay status. | Hive/job surfaces exist, but typed job records are not the universal unit. | Phase 2 |
| Artifact graphs and replay | Every artifact links to parent job, source refs, context refs, tools, claims, tests, environment, and replay limits. | Reports are abundant but not always unified into an artifact graph. | Phase 1, Phase 5 |
| Procedural memory | Repeated successful trajectories become verified procedural tools. | One low-risk assistant trace now has replay, canary execution, registry adoption, guarded planner default route, and planning-assistant consumption; other candidates still need the same adoption discipline. | Phase 7 |
| Routing heads and specialist cores | Use route decisions with readiness, authority, cost-quality reason, fallback route, and residuals. | Octopus/router language exists; routing eligibility is mostly blocked by stale evidence. | Phase 6, Phase 7 |
| Readiness gates and residual escrow | Modules and implementations move through quarantine, canary, qualified, default, deprecated, retired. | Gates exist, but stale outputs and duplicate families reduce trust. | Phase 0, Phase 6 |
| MoECOT runtime | Bind compact orchestrator, route head, specialists, gates, ledgers, replay, handoffs, and residuals. | The parts exist, but not one canonical MoECOT orchestration record for live tasks. | Phase 7 |
| Generate-verify-repair | Treat generation as propose, verify, repair, record residual, repeat. | The strict generator has replay, but correctness is eval-only more than training signal. | Phase 4 |
| Deterministic math/search substrates | Use exact tools for math/search/verification and record them as evidence, not model skill. | Tool substrate exists directionally; it needs to be attached to planning, VCM, claim ledgers, and assistant workflows. | Phase 2, Phase 8 |
| Benchmark ratchets | Public benchmarks calibrate; residuals become private repair targets; no Goodhart loops. | Policy is mostly right, but the current public transfer story is stale and must be remeasured only after private correctness improves. | Phase 9 |
| Policy optimization | Use feedback to update planners, routers, context selectors, verifiers, executors, and generators with drift bounds and rollback. | Dogfood metadata exists, but correctness-in-the-loop policy optimization is not yet the generator's central learning mechanism. | Phase 4, Phase 10 |
| Integrated reference architecture | Trace intent through governance, planning, VCM, routing, verification, execution, evidence, and improvement. | This is the missing canonical spine. | Phase 1 |
| Theseus as implementation reference | Theseus should be mined as the book's report-first example only where reports are reproducible and public-safe. | The book has the chapter, but Theseus needs a public-safe crosswalk bundle. | Phase 11 |
| Layer boundaries and authority | Every layer crossing should emit authority transition records and authority-use receipts. | Theseus has authority language in registry/VCM docs, but no universal authority transition ledger or Digital SCIF receipt layer. | Phase 13 |
| Failure boundary maps | Every major failure should map to a layer, trigger, protected invariant, detection route, containment action, evidence record, and downstream owner. | Gates report failures, but failures are not consistently normalized as boundary maps. | Phase 13 |
| Constitutional predicates | Alignment commitments should compile into protected operational predicates with translation status and self-modification rules. | `AGENTS.md` and personality policies carry commitments, but they are not first-class constitutional predicate records consumed by planning and self-improvement. | Phase 14 |
| Agency/dignity/corrigibility | High-impact plans should carry affected parties, delegation scope, manipulation risk, review/appeal channel, rollback path, and accountable principal. | Theseus mentions agency/corrigibility, but assistant, Hive, and autonomy plans do not consistently emit agency rights checklists. | Phase 14 |
| Value conflicts | Conflicts should become explicit unresolved obligations, not hidden prompt tradeoffs. | There is no visible value-conflict record for contested goals, risky autonomy, or irreversible changes. | Phase 14 |
| Governance rights | Fork, exit, audit, dissent, contestability, and preservation rules should be technical interfaces. | Repo audit exists, but user-visible governance rights are not yet a structured interface. | Phase 14 |
| Runtime adapter permissions | Tool calls and side effects should emit adapter invocation records with sandbox, approval, authority handle, effect receipt, rollback handle, and residuals. | Some tools run through scripts, but no universal runtime adapter invocation envelope exists across assistant/planner/Hive. | Phase 13 |
| Context transactions | VCM should emit immutable memory events, snapshots, mounts, taint, deletion obligations, contradiction refs, closure state, and faults. | VCM has strong design material and context packets, but transactional memory semantics are not yet universal. | Phase 15 |
| Context adequacy | Context should declare semantic units, compression path, verification mode, adequacy state, residual risks, and escalation. | VCM adequacy exists in concept, but assistant/model runs do not consistently fail closed on inadequate context. | Phase 15 |
| Resource budgets and costed routes | Routes should record value hypothesis, risk class, capacity pool, cost estimate, verification tax, rejected cheaper routes, and residual obligations. | Resource governor exists, but costed route records are not the normal routing/evaluation artifact. | Phase 16 |
| Generation mode records | Fast generation should be selected by risk, budget, context, draft source, verifier, accepted-output accounting, and quality result. | Candidate generation modes exist as row strings, but not as governed generation-mode records with accepted-output accounting. | Phase 16 |
| Simulation contracts | Simulation/fidelity claims should name scope, fidelity standard, time semantics, demand, bottlenecks, approximation liberties, and supported claim boundary. | Theseus uses simulations/fixtures in places, but no simulation contract separates map from territory. | Phase 17 |
| Semantic atoms and nodes | Semantic IR should use atom/node records with provenance, hierarchy, tokenization contract, grounding state, version, supersession, and permitted uses. | Plan/compiler artifacts exist, but semantic atom/node records are not the common substrate for planner, VCM, and generator repair. | Phase 18 |
| Compact/compressed artifacts | Compression claims should record reconstruction contract, residual coding, fallback, decode determinism, utility tests, and non-claims. | Report-evidence and retention now emit VIEA-gated `compressed_artifact_record` and `compression_receipt` rows, RMI emits a conservative AI_book-shaped `compact_generative_record` with generation/verification/fallback/residual-burden/promotion-blocker/source fields, and the Phase 14 retention canary now proves archive-pointer replay by exact hash. | Phase 18 |
| Substrate adoption | Circle, Coil, cyclic mixers, SymLiquid, transformers, and any novel substrate should have adoption records with baselines, negative controls, proof boundary, falsification condition, residuals, and evidence refs. | The Circle proof lane now emits three schema-shaped substrate adoption records, but adoption discipline still needs to become universal across SymLiquid, transformer/hybrid, STS, VCM, MLX/Metal, and future substrates before elegant ideas can bypass practical baselines. | Phase 18 |
| Research backlog | Every source/gap should have access state, source-note state, claim mapping state, proof/test backlog, insertion decision, residuals, and next action. | Theseus has many docs/reports, but no unified research backlog tying AI_book source gaps to Theseus implementation gaps. | Phase 19 |

## Phase 0: Stabilize the Current Truth

Purpose: make the repo trustworthy before adding capability work.

Deliverables:

- Update `docs/PROJECT_STATE.md` so it reflects the latest strict MLX generator
  findings from `reports/strict_generator_mlx_rung_findings_v1.md`.
- Refresh or explicitly retire stale registry evidence outputs so
  `scripts/theseus_project_registry.py --gate` is no longer RED from stale
  evidence.
- Resolve the hard registry governance violations.
- Keep generated/runtime/checkpoint/report bulk out of source claims.
- Add a short `docs/THESEUS_ASI_STACK_CROSSWALK.md` generated from this roadmap
  that maps each book layer to current Theseus surfaces, evidence, and gaps.

Acceptance gates:

- `python3 scripts/theseus_project_registry.py --gate` is GREEN or YELLOW with
  only explicit non-capability cleanup items.
- `python3 scripts/theseus_workspace_hygiene_audit.py` is GREEN or YELLOW with
  no hard registry violations.
- `docs/PROJECT_STATE.md` contains the current MLX strict generator wall:
  syntax/non-stub is no longer the main issue; semantic correctness is.
- No capability claim is promoted from stale reports.

Do not:

- Create another vN report family just to turn a gate green.
- Hide negative MLX results.
- Treat registry cleanup as capability progress.

## Phase 1: Canonical VIEA Execution Spine

Purpose: force all serious work through the book's reference architecture.

Canonical trace:

`IntentContract -> CommandContract -> PlanForgeDAG -> ContextABIRecord -> TypedJob -> RuntimeAdapterInvocation -> ArtifactGraphRecord -> ClaimRecord -> EvidenceTransitionRecord -> Residual -> PolicyOptimizationRecord`

Deliverables:

- Extend `scripts/theseus_plan_compiler.py` so every compiled goal emits the
  above records, not only a Theseus-specific plan packet.
- Make `scripts/theseus_assistant_runtime.py` call the spine for all non-trivial
  code, tool, planning, and repo-maintenance tasks.
- Emit one `reports/viea_execution_spine.jsonl` row per meaningful task.
- Add artifact graph IDs to generated reports, dogfood events, model training
  runs, verifier runs, and assistant outputs.
- Add claim IDs and evidence transition IDs for every reported capability
  statement.
- Add a missing-contract residual when a task cannot pass through the spine.

Acceptance gates:

- A local assistant coding task produces an intent contract, command contract,
  DAG, context packet, typed job, runtime invocation, artifact graph record,
  claim record, and evidence transition.
- The trace is replayable enough to identify source refs, context refs, tools,
  outputs, verifier commands, residuals, and limits.
- Side-effecting tasks have an authority ceiling and failure behavior.
- The system records `UNKNOWN`, `UNSOLVED`, or `TOOL_FAULT` instead of inventing
  success when evidence is missing.

Do not:

- Let the assistant answer as if work completed when no artifact was produced.
- Let planning and execution collapse into prose.
- Add a second planner.

## Phase 2: PlanForge and Labor OS Execution

Purpose: make planning operational instead of descriptive.

Deliverables:

- Convert representative goals in `configs/theseus_plan_compiler.json` into
  executable typed jobs.
- Add typed job records with lifecycle state, runtime adapter, inputs, outputs,
  permissions, approval state, failure behavior, audit events, and replay
  status.
- Route jobs through registered implementations only.
- Attach VCM context requests, deterministic tools, verifier burden, and
  resource class to each job.
- Add a minimal local work-board view that reads typed jobs from the artifact
  graph instead of inventing parallel status.

Acceptance gates:

- A private repo-maintenance goal runs as a DAG with at least three typed jobs.
- Failed nodes produce residuals and replanning suggestions, not silent retry
  churn.
- Job routing refuses unregistered or evidence-stale implementations.
- Cost, latency, verifier burden, and context size are recorded per job.

Do not:

- Create a dashboard that is not backed by typed job records.
- Let routers select deprecated or stale-evidence implementations.

## Phase 3: VCM as the Default Context ABI

Purpose: make VCM the memory layer of Theseus, not a benchmark side system.

Deliverables:

- Define stable semantic addresses for project state, registry state, model
  training state, assistant sessions, dogfood traces, codebase surfaces, and
  benchmark residuals.
- Emit `ContextABIRecord` and `SemanticPageCertificate` records for assistant,
  training, verifier, and planner contexts.
- Add context adequacy labels: `sufficient`, `partial`, `stale`, `tainted`,
  `missing`, and `overspecified`.
- Add VCM taint rules so public calibration artifacts cannot flow into training
  context.
- Add VCM ablations for the assistant and generator: context-on versus
  context-off, same task, same budget, same verifier.

Acceptance gates:

- The assistant can answer "what is the current wall?" from VCM-backed current
  state, with source report refs.
- The strict generator training/eval path records exactly what context was
  allowed.
- Public benchmark rows are tainted as calibration-only and rejected from
  training context.
- Context-on improves at least one practical internal task without regressions
  on leakage or runtime.

Do not:

- Keep proving VCM on saturated tiny benchmarks instead of using it in live
  Theseus workflows.
- Store raw private user text in training rows by default.

## Phase 4: Correctness-In-The-Loop Generator Training

Purpose: move from fluent wrong code to verified useful code.

The current strict MLX result says more raw token positions are not the next
lever. Theseus needs the verifier to shape learning.

Deliverables:

- Build a private, license-clean, contamination-audited curriculum ladder with
  tiers where the current model sometimes passes. The ladder must include very
  small tasks, role-selection tasks, return-shape tasks, and algorithm-family
  tasks.
- Add correctness labels from the private verifier to candidate rows:
  executable, syntax-valid, role-correct, return-contract-correct,
  algorithm-family-correct, verifier-pass, and failure family.
- Train or fine-tune the practical transformer/hybrid MLX lane with a
  correctness-aware objective:
  supervised body tokens plus verifier-labeled preference/ranking or
  rejection-sampling updates.
- Run the first actual policy-optimization update on private data:
  DPO/IPO/ORPO/KTO/SimPO offline preference learning over accepted/rejected
  verifier-labeled candidate pairs. Each family must share the same private
  pair split, reference checkpoint, optimizer budget, and heldout verifier. It
  must report pre/post heldout verifier pass, accepted-output quality, drift
  bound, rollback plan, reward-hacking probes, and no-cheat counters. It is not
  complete if it only improves LM loss or policy-gap metrics.
- Add a bounded GRPO/RLOO/ReMax/RLVR verifier-reward lane for exact private
  tasks after the offline preference path is reproducible. Reward is functional
  verifier evidence plus context/evidence adequacy, not style. Authority
  expansion by training side effect is forbidden.
- Treat policy optimization as a multi-target behavior-change mechanism, not
  only generator fine-tuning. Planner, router, VCM-selector, verifier-policy,
  execution-policy, and generation-mode-selector updates require explicit
  leases naming target policy, admissible feedback, reward boundary,
  authority effect, drift bound, rollback, monitor window, and task-specific
  heldouts.
- Extend the negative-replay objective beyond candidate-level CE into
  verifier-stage-aware preference learning: prefer private candidates that
  advance from syntax -> runtime load -> intended behavior, and reject repeated
  wrong-answer/dataflow bodies without lowering the base supervised target.
  The first reward-inverse ablation regressed decode quality, so do not promote
  it; use it as evidence that stage preference needs pairwise or accepted-
  candidate structure instead of scalar negative weights alone.
- Run prompt/signature sensitivity ablations: prompt only, signature only,
  prompt plus signature, VCM-on, VCM-off, STS-on, STS-off.
- Keep SymLiquid as a matched discovery comparator, not as the blocker for the
  practical assistant lane.
- Report model-only generation separately from deterministic tool-assisted
  completion.
- Add multi-token prediction as the first concrete Part III capability
  technique: next-2/next-4 auxiliary heads on the transformer/hybrid survival
  lane, verifier-stage-aware accepted-span accounting, and a matched
  single-token control under the same data, optimizer, active-parameter, and
  wall-clock budget. MTP is not a speed claim or a promotion claim until it
  improves heldout private verifier behavior or useful accepted-span rate.
- Add bounded generate-verify-repair as the first product/capability bridge:
  one-shot model-only decode, verifier-guided repair, repair cost, fallback
  count, verifier wall time, and useful solution/sec are reported separately.
  The verifier may guide repair as a tool-assisted route, but repaired output
  cannot be credited as unassisted learned generation.
- Implement generate-verify-repair as a typed state machine:
  candidate -> verified-exact -> verified-lossy -> repaired-exact ->
  literal-fallback/noncredit -> quarantined. Every transition needs a verifier
  receipt, repair cost, residual, and non-claim.
- Add lookahead/trie retrieval from verified procedural memory plus one
  sketch-first/diffusion/LLaDA-style repair experiment as governed generation
  modes. Medusa, EAGLE, speculative/self-speculative, early-exit/LayerSkip,
  Mamba-state, and PagedAttention/KV-cache routes are allowed only as
  generation-mode candidates with accepted-output accounting. They are judged
  by useful verified output and are noncredit for learned generation unless the
  same model-only checkpoint improves one-shot behavior.
- Add semantic-IR localized repair to the generator loop: compile
  prompt/signature/candidate body into semantic atoms, map verifier failures to
  failed atoms, repair only affected atoms, and replay dependents. Requirement
  changes must emit a scope-change ledger entry instead of silently rewriting
  the task.
- Add one capacity ablation before more marker/slot churn: either a 50-150M
  dense transformer/hybrid scale step or a sparse MoE with comparable active
  compute. The run must compare against the current dense baseline and record
  router receipts, active parameters, memory, throughput, verifier behavior,
  residuals, and a falsification condition.
- Wrap MTP, generate-verify-repair, MoE/scale, STS, VCM, MLX/Metal, and any
  future substrate change in a substrate-adoption record: baseline, negative
  control, matched data/compute, consumer axis, fallback/retirement rule,
  residuals, and non-claims.

Acceptance gates:

- The trained lane beats the current strict private baseline on verifier pass,
  not just syntax, nontrivial return, or lower LM loss.
- The first DPO/IPO update beats its pre-update checkpoint on private heldout
  verifier pass or accepted-output quality, and reward-hacking probes pass.
- GRPO/RLVR does not become a default route until verifier reward improves
  correctness without increasing shallow identity loops, inert stubs, early
  returns, latency abuse, context ignoring, or authority creep.
- At least one held-out private tier has non-zero semantic verifier pass.
- Family-disjoint behavior improves without answer-identifying metadata.
- Candidate integrity is GREEN with zero mismatches.
- Blind information-flow audit is GREEN.
- No fallback returns, templates, fixed action routers, or deterministic tools
  count as learned generation.
- MTP only advances if matched-control private verifier behavior, accepted-span
  quality, or useful solution/sec improves without worse no-cheat, integrity,
  or fallback counters.
- Generate-verify-repair only advances as an assisted route unless the same
  checkpoint also improves one-shot model-only behavior.
- MoE/scale only advances if matched active-compute private verifier behavior
  improves or the result is recorded as a falsifying substrate-adoption record.

Do not:

- Spend the next major cycle only on more MLX scale.
- Count generic but syntactically valid bodies as progress.
- Use public benchmarks as training data.
- Let semantic renderers or structural adapters support learned-generation
  claims.
- Start another target-label, slot, vocab, marker, guard, token-bias, or
  contrast family unless it is inside the registered MTP/GVR/MoE/scale
  substrate-adoption sequence and has a falsification stop.

## Phase 5: Claim Ledger, Artifact Graph, and Evidence State

Purpose: make evidence easy to audit and hard to overclaim.

Deliverables:

- Add first-class claim records for current project claims:
  MLX throughput, generator semantic correctness, VCM integration, registry
  health, assistant usefulness, public transfer, teacher share, and dogfood use.
- Add evidence transition records whenever a claim changes support state.
- Add contradiction links for negative or regressed results.
- Add artifact graph records for model checkpoints, reports, configs,
  curricula, verifier outputs, and assistant dogfood events.
- Make `docs/PROJECT_STATE.md` a human view over the claim ledger, not a manual
  historical dump.

Acceptance gates:

- A reader can trace any active claim to report refs, configs, commands,
  negative results, and non-claims.
- Stale evidence blocks route eligibility or claim promotion automatically.
- Regressions remain attached to the claim they contradict.

Do not:

- Rewrite negative evidence into vague language.
- Let a report be both the evidence and the claim without a support state.

## Phase 6: SCF Replacement Transactions and Rollback

Purpose: make self-improvement controlled without becoming slow or fake.

Deliverables:

- For each routable abstraction, define lifecycle states:
  `experimental`, `shadow`, `canary`, `qualified`, `default`, `deprecated`,
  `retired`, `blocked`.
- Require replacement transactions for changing default implementations of:
  candidate generation, VCM context selection, STS/ranking, plan compiler,
  assistant runtime, deterministic tools, teacher distillation, resource
  routing, and Octopus routing.
- Add rollback metadata before default promotion.
- Add evaluator-independence checks so candidates cannot be their own sole
  judge.
- Add residual escrow for known weaknesses.

Acceptance gates:

- One non-trivial implementation replacement runs through transaction records
  with precheck, qualification evidence, regression results, residual escrow,
  rollback, monitor window, and decision.
- Regression failure blocks promotion.
- Rollback metadata exists before default promotion.

Do not:

- Let "works once" become default route.
- Let an implementation expand authority by being more capable.

## Phase 7: MoECOT, Octopus Routing, and Procedural Memory

Purpose: make Theseus route work through specialists and turn repeated wins into
tools.

Deliverables:

- Emit `RoutingDecisionRecord` for every specialist/tool/model route.
- Emit `MoECOTOrchestrationRecord` for multi-core runs.
- Add route reasons: capability request, candidate specialists, readiness
  check, authority check, cost-quality reason, selected specialist, fallback
  route, residuals, ledger refs.
- Normalize Octopus route decisions into the shared VIEA route spine: routing
  decision, specialist semantic node, authority receipt, VCM context
  transaction/adequacy, runtime adapter, resource budget, costed route,
  generation-mode boundary, failure boundary, artifact, claim, evidence
  transition, and residual.
- Promote repeated successful traces into procedural tool candidates only when
  preconditions, postconditions, verifier result, monitoring plan, regressions,
  and retirement criteria are present.
- Tie Octopus arms to registry implementations and SCF route validity.

Acceptance gates:

- A repeated assistant/repo workflow becomes a procedural tool candidate with
  verification and residuals.
- A routing decision can be replayed and explains why another specialist was
  not chosen.
- Stale or unqualified specialists are not route-eligible.
- Router selections carry `0` learned-generation credit and cannot launder
  template, fallback, tool, or structural-adapter evidence into learned model
  claims.

Do not:

- Treat procedural tools as learned model capability.
- Route around registry evidence because a path is convenient.

## Phase 8: Deterministic Tool and Search Substrate Integration

Purpose: stop asking the model to learn what exact tools can compute or check.

Deliverables:

- Register deterministic tools as fields and implementations:
  exact math, numeric intervals, local code search, VCM hybrid retrieval, Lean
  checking, private verifier replay, artifact search, and trace replay.
- Store tool outputs as evidence artifacts with replay checksums.
- Add tool selection into PlanForge nodes and assistant runtime.
- Add tool-on versus tool-off evaluations for private math, code-contract,
  repo-search, and function-calling tasks.
- Report tool-assisted scores separately from model-only scores.

Acceptance gates:

- Tool calls produce evidence records, not chat-only text.
- Wrong tool, failed tool, and unverifiable tool outputs produce typed failure
  states.
- Tool use improves at least one practical private assistant workflow.

Do not:

- Hide tool use inside model claims.
- Treat deterministic solver output as learned generation.

## Phase 9: Benchmark Ratchets and Public Transfer

Purpose: measure broad transfer honestly without Goodhart loops.

Deliverables:

- Keep public calibration surfaces frozen, registered, decontaminated, and
  consumed after execution.
- Before the next public calibration, require a private readiness packet from
  Phase 4 showing semantic correctness improvement, candidate integrity, blind
  information-flow, and replayability.
- After public calibration, mine failures into private residual categories:
  algorithm choice, return shape, IO contract, verifier mismatch, timeout,
  no-admissible, prompt understanding, role selection, tool-selection error.
- Do not rerun the same consumed public surface for score fishing.

Acceptance gates:

- Public benchmark reports separate model-only, tool-assisted, deterministic
  tool, and baseline paths.
- No public payload becomes training data.
- Failures become private repair targets, not another immediate public rerun.

Do not:

- Lock public benchmarks for arbitrary calendar reasons.
- Run public surfaces repeatedly to hunt a lucky score.
- Train on public benchmark answers or traces.

## Phase 10: Daily-Use Assistant and Dogfood Learning

Purpose: make Theseus useful to Corben now while creating real learning
pressure.

Deliverables:

- Make the local assistant the primary daily-use lane.
- Add conversation-specific training only through local metadata and governed
  accepted/missed/ignored/corrected/completed events.
- Add VCM-backed memory of project state, user preferences, current walls,
  pending work, and tool outcomes.
- Add code assistance tasks that produce patches, tests, evidence records, and
  dogfood outcome events.
- Add an "answer with uncertainty" mode where Theseus can explicitly say what
  it knows, what it inferred, what it checked, and what remains unsupported.
- Add a private assistant quality suite from real dogfood events, not public
  benchmark payloads.

Acceptance gates:

- The assistant can help with one real Theseus repo task end to end, with
  artifacts and verification.
- Dogfood events are recorded without raw private text by default.
- Accepted/missed/ignored/corrected/completed outcomes feed policy optimization
  or curriculum records.
- The assistant does not claim external inference output as local model output.

Do not:

- Build new mobile/spatial/product surfaces before the local assistant is useful.
- Store raw private conversation as training data by default.

## Phase 11: Use Theseus as the Book's Implementation Reference

Purpose: after Theseus is improved, feed public-safe implementation evidence
back into AI_book.

Deliverables:

- Produce a public-safe Theseus crosswalk bundle for the book:
  stack layer, source reports, configs, gates, residuals, public claim boundary,
  verification status, missing artifacts, non-claims, and evidence refs.
- Export one or more anonymized, reproducible vertical slices:
  intent-to-execution trace, replacement transaction, VCM context transaction,
  model-training policy optimization record, and benchmark ratchet record.
- Update AI_book source notes only with evidence that is actually public-safe
  and reproducible.
- Keep Theseus private capability claims out of the book unless they are
  independently supported.

Acceptance gates:

- Book artifacts clearly separate design argument, source-reported result,
  local prototype result, and reproduced evidence.
- Theseus examples can be rendered, audited, and understood without exposing
  private data.
- Negative results are included where relevant.

Do not:

- Use Theseus as marketing proof of ASI capability.
- Let book prose inflate a Theseus report beyond its evidence state.

## Phase 12: Mac and Hive Runtime After the Core Spine Is Stable

Purpose: make the distributed runtime efficient only after the single-node
control spine is honest.

Deliverables:

- Keep MLX as the first-class Apple Silicon training route.
- Fix or quarantine stale cross-machine assumptions in resource profiles.
- Make Intel Mac routes CPU/storage/operator by default.
- Make Hive typed jobs consume the same command/plan/artifact/evidence records
  as local jobs.
- Add report/checkpoint retention and compaction so training artifacts do not
  choke iteration.
- Add a hard live-report budget gate: no new report family without a registry
  owner, retention class, archive/replay policy, and a reason it cannot update
  an existing flagship claim/evidence record. Current live generated state
  should be driven below `1G` unless a larger bound is justified by active
  checkpoint artifacts.

Acceptance gates:

- Mac local training uses MLX where available and reports backend/device,
  throughput, checkpoint refs, and verifier results.
- Non-MLX Macs do not claim MLX.
- Hive jobs are auditable through the same typed job and artifact graph records
  as local jobs.
- The report/checkpoint retention gate reports live byte size, file count,
  largest families, registry owner, retention class, archive pointer, replay
  hash, and deletion/quarantine decision for every generated family above the
  threshold.
- Repeated metric variants update a registered flagship claim record instead
  of spawning unregistered `_vN` report families.

Do not:

- Spend engineering time on installer/product polish while the generator,
  assistant, registry, and spine remain unhealthy.
- Treat generated report volume as progress.

## Phase 13: Authority Kernel, Failure Boundaries, and Runtime Adapter Receipts

Purpose: make authority, security, and side effects enforceable instead of
merely described.

Deliverables:

- Add an authority transition record for every layer crossing that can read,
  transform, disclose, write, execute, approve, train, route, or publish.
- Add authority-use receipts for secret handles, privileged paths, network
  operations, external teacher calls, file-system writes, Hive jobs, and update
  operations.
- Add a Digital SCIF-style pattern for sensitive context: the model sees a
  handle and permitted use, while the executor substitutes the secret or
  privileged value outside model-visible text.
- Add failure boundary maps for the main failure classes:
  evaluator capture, memory poisoning, prompt injection through context,
  tool/action overreach, hidden public-data contamination, stale registry
  routing, and residual hiding.
- Wrap tool and script calls used by the assistant/planner/Hive in runtime
  adapter invocation records with permission, sandbox, approval, authority
  handle, effect receipt, rollback handle, and residuals.

Acceptance gates:

- A side-effecting local assistant task emits an authority transition record,
  authority-use receipt, runtime adapter invocation, effect receipt, and
  rollback or explicit no-rollback residual.
- A simulated confused-deputy request is denied before tool execution.
- A fake-secret fixture proves the model never receives the raw secret while
  an allowed executor can use the handle for a permitted destination/action.
- Failure boundary maps are attached to at least five existing gate failures.

Do not:

- Put secrets, tokens, or private raw text into model-visible prompts to make a
  demo easier.
- Treat read access as write/execute authority.
- Let "local machine" imply unlimited authority.

## Phase 14: Constitutional Predicates, Agency Rights, Value Conflicts, and Governance Rights

Purpose: turn Theseus's values into operational constraints without pretending
the system has solved moral philosophy.

Deliverables:

- Compile `AGENTS.md`, `docs/PERSONALITY_CORE.md`, and active governance
  policies into constitutional predicate records with normative source,
  commitment, operational test, protected scope, translation status,
  uncertainty, review route, self-modification rule, and non-claims.
- Add agency rights checklists for high-impact plans: affected parties,
  delegation scope, manipulation risk, reversibility, review channel, appeal
  channel, shutdown/rollback path, accountable principal, residual dependency
  risk, and approval requirement.
- Add value conflict records for ambiguous or contested tasks, especially
  autonomy, external teacher use, public calibration, data retention, private
  memory, and irreversible filesystem/network actions.
- Add governance right records for audit, exit, fork, dissent, contestability,
  data export, deletion, and route refusal.
- Make self-improvement transitions prove they did not weaken protected
  predicates.

Acceptance gates:

- A self-improvement proposal cites protected constitutional predicates and
  cannot proceed if it weakens them.
- A high-impact assistant/Hive/autonomy plan emits an agency rights checklist.
- A value conflict produces an explicit decision, residual uncertainty, and
  revisit condition.
- The user can inspect what Theseus believes their audit/export/delete/fork
  rights are from a structured report.

Do not:

- Use worldview or values language as a substitute for enforceable predicates.
- Let convenience optimize away user agency, review, or rollback.
- Treat unresolved value conflicts as if the model silently solved them.

## Phase 15: VCM Transactions, Snapshot Semantics, Taint, and Context Adequacy

Purpose: make VCM safe enough to become the default memory/context layer for
the assistant, planner, verifier, and training loop.

Deliverables:

- Add context transaction records for VCM reads, writes, derivations, branches,
  summary creation, public-calibration imports, dogfood metadata, and deletion
  operations.
- Track snapshot IDs, mounts, read sets, write sets, branch policy, taint
  labels, deletion obligations, contradiction refs, closure state, faults, and
  audit refs.
- Add context adequacy records to assistant answers, planner nodes, verifier
  runs, teacher proposals, and model-training/eval packets.
- Add fail-closed behavior when context is `tainted`, `stale`, `missing`, or
  inadequate for a verification-backed claim.
- Add deletion-closure tests showing that deleted or revoked material does not
  reappear through derived summaries, embeddings, caches, reports, or training
  rows unless policy explicitly permits a retained derivative.

Acceptance gates:

- A VCM-assisted assistant answer includes transaction and adequacy records.
- A public benchmark artifact is tainted calibration-only and cannot flow into
  training context.
- Each materialized packet has a representation certificate: source refs,
  omissions, loss contract, permitted uses, authority ceiling, lease, and
  consumer policy. A summary can never raise the authority ceiling of the
  source it summarizes.
- Copy-on-write branches, mounts, read/write sets, taint propagation,
  contradiction refs, and deletion obligations are replayable. Mandatory misses
  emit typed faults instead of best-effort context.
- A stale or contradictory memory read produces a context fault or residual,
  not a confident answer.
- A deletion fixture can prove closure or emit a clear closure fault.

Do not:

- Treat "more context" as proof.
- Let context compression elevate authority or erase taint.
- Allow deleted data to survive through summaries without an explicit policy.
- Claim model-native MLX/Metal KV or prefix-cache parity from semantic
   descriptor-cache evidence. Runtime parity needs a backend-specific replay
   proof.

## Phase 16: Resource Budgets, Costed Routes, and Generation Mode Accounting

Purpose: make speed and efficiency meaningful by measuring accepted useful work,
not raw motion.

Deliverables:

- Add resource budget records to plan nodes, training runs, verifier passes,
  public calibrations, teacher calls, Hive jobs, and assistant tasks.
- Add costed route records to every major route decision: candidate routes,
  selected route, rejected lower-cost routes, verification result, cost
  accounting, residual obligations, fallback route, promotion candidate, and
  non-claims.
- Add generation mode records for model decoding, speculative generation,
  draft-verify-accept paths, deterministic tool assistance, MLX generation,
  Rust generation, STS-conditioned generation, and VCM-assisted generation.
- Name concrete fast-generation families in the generation-mode registry:
  AR baseline, MTP, Medusa, EAGLE, speculative/self-speculative, early-exit or
  LayerSkip, lookahead/trie retrieval, Mamba-state decoding, PagedAttention/KV
  cache policy, and diffusion/LLaDA sketch-first repair.
- Measure proposed output and accepted output separately.
- Report verification tax explicitly: time spent proving/validating versus
  time spent generating.

Acceptance gates:

- A code-generation run reports accepted-output accounting, verifier cost,
  wall-clock time, quality/pass result, repair/fallback state, and promotion
  decision for each generation mode.
- A cheaper route rejected by the planner includes the quality/authority reason
  it was rejected.
- A fast path cannot be promoted on throughput if accepted verified output gets
  worse.

Do not:

- Optimize tokens/sec while verifier-passing outputs remain flat or worse.
- Hide human repair burden outside the cost model.
- Let low-value tasks consume scarce verification capacity without a budget
  decision.

## Phase 17: Simulation and Fidelity Contracts

Purpose: separate useful simulations, toy fixtures, and structural proxies from
claims about the real system.

Deliverables:

- Add simulation contract records for toy environments, private fixtures,
  proxy benchmarks, Circle-derived finite fixtures, resource simulations,
  synthetic curricula, and model-training smoke tests.
- Each contract must name scope, fidelity standard, temporal semantics, demand
  estimate, capacity bottlenecks, approximation liberties, supported claim
  boundary, residual risks, and evidence refs.
- Add map-territory labels to reports: `toy_fixture`, `proxy`, `private_heldout`,
  `public_calibration`, `runtime_dogfood`, `production_route`.
- Require public or runtime claims to cite why a toy/proxy result transfers, or
  explicitly state that it does not.

Acceptance gates:

- Existing private/synthetic reports can be classified by fidelity level.
- A toy/simulation result cannot support a public-transfer or production-route
  claim without an explicit transfer witness.
- Negative proxy results remain visible.

Do not:

- Treat private synthetic 1.0s as broad capability.
- Treat structural proofs or fixtures as model-quality evidence.

## Phase 18: Semantic IR, Compression, and Substrate Adoption Discipline

Purpose: let Theseus explore strong ideas without letting elegant structures
outrun evidence.

Deliverables:

- Add semantic atom records to plan compiler nodes and code-generation repair
  paths: intent, inputs, outputs, constraints, dependencies, authority,
  validator, target, and repair scope.
- Add semantic node records to VCM pages, claim graphs, code concepts, and
  training residual categories with provenance, hierarchy, relation refs,
  tokenization contract, grounding state, version, supersession, uncertainty,
  permitted uses, and evaluation refs.
- Add compact generative and compressed artifact records for checkpoints,
  reports, VCM pages, and code artifacts before claiming compression wins.
  Current implementation emits conservative report/retention
  `compressed_artifact_record` and `compression_receipt` rows plus one RMI
  `compact_generative_record`; these are schema/evidence contracts only, not
  compression-ratio, downstream-utility, or learned-generation claims.
- Add substrate adoption records for SymLiquid, transformer/hybrid, STS, VCM,
  Circle, Coil, cyclic mixers, RankFold/NeuralFold, and deterministic tools.
- Add cyclic memory and cyclic mixer evaluation records only as optional
  substrate experiments with baselines, negative controls, proof boundary,
  falsification condition, hardware notes, adoption state, residuals, and
  non-claims.

Acceptance gates:

- A generator repair can point to a failed semantic atom rather than only a
  failed task row.
- A substrate cannot become default without a registered adoption record and
  baseline comparison.
- Compression claims include metadata, parameters, residuals, fallback, decode
  determinism, utility tests, and non-claims.
- SymLiquid and any cyclic/Coil substrate remain protected experiments unless
  they beat practical baselines under matched evidence.

Do not:

- Let mathematical elegance replace verifier results.
- Claim parameter reduction, structural invariance, or compression ratio as
  capability without task utility.
- Keep inventing new substrate lanes without falsification criteria.

## Phase 19: Research Backlog and Book-to-Theseus Synchronization

Purpose: make AI_book and Theseus co-evolve without turning source mining into
untracked idea sprawl.

Deliverables:

- Add a research backlog record for every mined AI_book gap that Theseus has
  not implemented.
- Track source or gap, access state, assigned book chapters, source-note state,
  claim mapping state, external literature need, proof/test backlog, insertion
  decision, residuals, and next action.
- Add a `book_to_theseus_gap` view that maps AI_book schemas/chapters to
  Theseus surfaces, fields, reports, and missing implementation records.
- Refresh the `book_to_theseus_gap` view from authored AI_book source
  checksums, excluding generated book builds, archives, and build outputs.
- Emit stale-source backlog rows when matched AI_book source hashes change, so
  implementation drift becomes work cards instead of manual prose drift.
- Carry unresolved stale-source rows forward until steward review clears them,
  even after the checksum baseline catches up.
- Add a `theseus_to_book_evidence` view for public-safe implementation examples
  that can later be fed back to the book.
- Update this roadmap only through registry-owned changes, not one-off notes.

Acceptance gates:

- Every roadmap item has either an implementation surface, backlog record, or
  explicit deferral reason.
- AI_book source gaps do not become Theseus claims until implemented/tested.
- Theseus evidence exported to AI_book is public-safe and support-state labeled.
- Current implementation evidence: `reports/book_to_theseus_crosswalk.json`
  reports `38` active roadmap backlog items, `0` stale phases, `136` active
  source-sync review decisions, and `57` public-safe Theseus-to-book evidence
  pointers. The chapter-level implementation contract is now carried in
  `configs/roadmap_implementation_matrix.json` with `44/44` authored chapters
  mapped to Theseus phases, support-state targets, gates, and no-claim
  boundaries. The July 6 weekly-focus public-safe reference trace and
  book-importable evidence packs are imported into AI_book through the existing
  reference-trace harness and validated by `python3 scripts/validate_reference_trace.py`,
  `python3 scripts/validate_reference_trace_replay.py`,
  `python3 scripts/validate_receipt_repository_audit.py`, and
  `python3 scripts/validate_book.py`; the import remains a record-shape
  implementation reference, not a model-quality or ASI claim.

Do not:

- Duplicate book concepts into new Theseus docs without registry ownership.
- Cite unread sources as support.
- Let roadmap milestones become evidence by themselves.

## Immediate Next Goal Candidate

The next goal should not run another training loop, public benchmark, or score
chase. The next goal should make Theseus a faithful implementation of the best
systems from AI_book, with each architecture idea owned by a registry surface
and wired into the canonical execution spine. Claude's recommendation to remove
late roadmap phases is useful as a warning against capability theater, but it
is not accepted as scope deletion: authority, governance, VCM transactions,
resource accounting, semantic IR, substrate adoption, and research backlog are
core to Theseus becoming the book's implementation reference.

**AI Book Systems Integration and Roadmap Implementation v1**

Goal statement:

Implement the roadmap as a cohesive system, not a pile of reports. For every
AI_book-derived phase, create or update the registered abstraction,
implementation, schema/record, gate, and execution-spine integration needed for
Theseus to use it. The assistant/product path, VCM, tools, verifier, claim
ledger, authority receipts, resource accounting, semantic IR, Hive routing, and
research backlog must become one governed architecture. Training and benchmark
runs are explicitly out of scope except for lightweight syntax/schema/gate
smokes needed to verify that the implementation is wired.

Acceptance criteria:

- Every roadmap phase has a registry-owned module card, abstraction/SCF
  binding, implementation record, owner surface, lifecycle state, and explicit
  integration point into the VIEA execution spine.
- Phases 13-19 are preserved and implemented as core governance/runtime
  substrates, not deleted: authority receipts, constitutional predicates, VCM
  transactions, resource budgets, simulation contracts, semantic IR,
  compression/substrate adoption records, and book-to-Theseus backlog records.
- The assistant path is wired as the product-facing integration surface:
  VCM context packets, deterministic tools, retrieval, verifier gates, claim
  ledger entries, artifact refs, and accepted/missed/corrected/ignored dogfood
  events all flow through one execution trace.
- Product usefulness and learned-generation capability claims are separated in
  the claim ledger. Tool-assisted output can ship as product behavior, but it
  never supports a learned-generation claim.
- Strict generator work is capped to implementation readiness: no new
  target-label/slot/marker/guard family and no training run until the
  architecture surfaces above are complete and the next experiment contract is
  preregistered.
- The roadmap contains a current implementation matrix that marks each phase as
  `implemented`, `wired`, `partial`, or `missing`, with a smallest next patch
  for every missing item.
- Public benchmarks remain calibration-only and are not training data.
- No fallback/template/router/tool path receives learned-generation credit.

Recommended first commands for that goal:

```sh
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
python3 scripts/viea_spine_record_gate.py --gate
python3 scripts/theseus_workspace_hygiene_audit.py
```

The actual implementation should patch registered architecture surfaces,
schemas, configs, docs, and gates. It should not create loose lanes, score
reports, new benchmark families, or another target-label family.

### Implementation Priority For The Next Goal

The next long-running goal should implement the roadmap, not train the model.
The correct order is:

1. Build the phase implementation matrix for all roadmap phases and AI_book
   crosswalk rows.
2. Fill missing schemas/records for authority transitions, adapter receipts,
   constitutional predicates, VCM transactions, resource routes, simulation
   contracts, semantic IR atoms, compression/substrate adoption, and
   book-to-Theseus backlog records.
3. Wire those records into the canonical VIEA execution spine and assistant
   runtime so real work emits them automatically.
4. Make the registry gate verify that new Theseus work either improves an
   existing registered implementation or creates a justified registry-owned
   abstraction/implementation pair.
5. Update docs so the roadmap, registry, PROJECT_STATE, and AI_book crosswalk
   agree on what Theseus is right now.

### Phase Implementation Matrix Seed

The authoritative machine-readable matrix now lives at
`configs/roadmap_implementation_matrix.json` and is validated by
`scripts/roadmap_implementation_gate.py`. The table below is the human summary;
the config is the working contract. This matrix is a planning target, not
completion evidence. A phase is complete only when the matrix names the
registered surface, abstraction, implementation, execution-spine hook, required
records, gates, docs, evidence, and smoke trace.

The shared execution-spine record contract now lives at
`configs/viea_spine_record_contracts.json` and is validated by
`scripts/viea_spine_record_gate.py`. It normalizes assistant flat traces,
planner nested ASI-stack records, deterministic tool evidence, and
execution-spine runtime records into one record family vocabulary before those
traces can support roadmap completion. The direct generator boundary and
train-once fanout supervisor now contribute the same vocabulary through
`direct_generator_context_spine_v1` and `train_once_fanout_spine_v1`,
including governed `context_transaction` and `context_adequacy` records from
the shared VCM governor receipt without starting a training run. The Hive
scheduler contributes the same
vocabulary directly through `hive_scheduler_route_records_v1`, so route plans
and non-executing task-submission receipt smokes carry authority, adapter,
budget, costed-route, failure-boundary, artifact, evidence-transition, and
generation-mode records before live peer execution is trusted. Candidate
integrity contributes `candidate_integrity_producer_v1`, so independent family
recompute and promotion-claim guarding enter the same claim/proof/evidence
view instead of living only as report prose. The private verifier contributes
`private_verifier_spine_v1`, so lint/compile/runtime/intended-behavior cascade
labels enter the same claim/proof/authority/failure/artifact vocabulary without
becoming generation credit. The product assistant smoke at
`reports/theseus_assistant_product_spine_smoke.json` now exercises the chain
from VCM packet to deterministic tool evidence, private verifier receipt,
materialized-view receipt, authority/resource records, artifact refs, claim
ledger entry, and dogfood event. The gate also writes
`reports/viea_spine_materialized_view.json`, a compact shared view over
claim/proof/evidence, semantic IR, simulation-contract, governance-right,
constitutional-predicate, artifact, authority, route, and failure-boundary
records.

The roadmap gate writes `reports/book_to_theseus_crosswalk.json`, mapping every
phase to its AI_book source basis, registry surface, abstraction,
implementation, evidence, missing items, smallest next patch, and source-sync
support state. It inventories authored AI_book source files and keeps the exact
source count plus manifest hash gate-owned in
`reports/book_to_theseus_crosswalk.json`, so the human roadmap does not churn
when the living book changes. Generated
book builds, archives, and Lean build outputs are excluded. It also emits
`57` public-safe Theseus-to-book evidence pointers, carries `38` active sticky
roadmap backlog rows in the crosswalk with `136` source-sync review decisions,
`0` newly cleared backlog rows against the latest source inventory, and `0`
currently stale phases. The module
DoD gate keeps raw source-drift review routed into `38` module work cards and
`38` steward-decision candidates through
`reports/book_to_theseus_backlog_work_cards.jsonl`.
This is the current machine-readable book-to-implementation sync point, not a
completion claim.

The registry gate now has a manifest-owned per-report freshness policy so
immutable hash-addressed MLX checkpoint provenance and the 30M-rung strict
generator replay family can be weekly-lived while fast canaries and mutable
latest-health reports remain tightly freshness-gated. A full private MLX rung
replay on this Mac refreshed three child artifacts before exposing a real
`mlx.core.eval`/verifier hot-loop bottleneck. The bounded canary
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_canary_v1.json`
is now `GREEN`: it selected `1/5` hash-addressed checkpoints, ran one
family-disjoint private row through MLX replay in `3532` ms total / `2497` ms
child decode-eval time, reused the private split selection through a hash
receipt, and kept public training rows, external inference,
fallback/template/router/tool credit, and integrity mismatches at zero. The
child decode is still `YELLOW` with `0/1` passes, so this is route-health and
runtime-economics evidence only. The two-checkpoint receipt below supersedes
this one-checkpoint bottleneck note with split, top-k, checkpoint-loader, and
verifier-cache reuse evidence. This keeps expensive proof artifacts from
forcing daily reruns without letting stale current-health evidence authorize
routing.

The follow-up two-checkpoint canary
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_two_checkpoint_canary_v1.json`
is also `GREEN`: it selected `2/5` checkpoints, reused the same private
family-disjoint split across both replays, and wrote child decode receipts that
name `argpartition_bounded_topk_v1` with `full_vocabulary_sort_in_hot_loop =
false`. The same canary now proves same-vocab checkpoint-loader reuse without
skipping weight reloads: `vocab_read_count=1`, `vocab_cache_hit_count=1`,
`model_construct_count=1`, `model_reuse_count=1`, and
`checkpoint_weight_load_count=2` with the hard
`checkpoint_loader_reloads_each_checkpoint` gate passing. The canary now also
emits the verifier sandbox warmup rollup
`private_verifier_sandbox_warmup_accounting_rollup_v1`: `receipt_count=2`,
test-harness compile caching enabled, and `total_test_harness_cache_hit_count=2`.
It still records `0/2` private passes, so the evidence improves Mac runtime
economics and route observability, not learned code capability.

The bounded all-rung follow-through
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_all_rung_bounded_v1.json`
is `GREEN`: it selected all `5/5` available rungs, passed the hard
`all_available_checkpoints_selected_when_unbounded` gate, reused one
family-disjoint private split across every checkpoint, reused one same-vocab
model construction, reloaded all five checkpoint weight files, and emitted five
verifier-cache warmup receipts. Loader stats are `vocab_read_count=1`,
`vocab_cache_hit_count=4`, `model_construct_count=1`, `model_reuse_count=4`,
and `checkpoint_weight_load_count=5`; verifier warmup reports
`receipt_count=5` and `total_test_harness_cache_hit_count=5`. The all-rung
smoke also emits `project_theseus_mlx_rung_replay_resource_budget_v1`; the
no-threshold baseline passes the `resource_budget_thresholds_respected` gate.
It now also emits
`project_theseus_mlx_rung_replay_route_eligibility_v1`: the baseline is
`production_route_eligible=false` with
`route_state=observation_only_no_resource_thresholds`. It still records `0/5`
private behavior passes and one final rung with nontrivial-return rate `0.0`,
so this closes a route-health/runtime proof but does not move learned code
capability.

The strict resource-budget probe
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_resource_budget_probe_v1.json`
is `GREEN` on route-resource mechanics under a 30s total budget, 10s max child
budget, and 0.2 eval rows/sec floor: `budget_ok=true`,
`max_child_decode_eval_runtime_ms=6451`, and `eval_rows_per_second=0.300084`.
The new route-eligibility receipt still fail-closes production routing with
`production_route_eligible=false` and
`route_state=fail_closed_behavior_quality_zero` because the replay remains
`0/5` on private behavior. The next resource-route step was the broader replay
and readiness gate below; semantic candidate-quality repair is now Phase 10
work before any production MLX route claim.

The broader private MLX replay
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v2.json`
is now `GREEN` under a 65s total budget, 25s max child budget, and 0.15 eval
rows/sec floor: `total_eval_rows=10`, `total_generated_candidate_rows=10`,
`eval_rows_per_second=0.1906`, `max_child_decode_eval_runtime_ms=16961`,
`checkpoint_weight_load_count=5`, `model_reuse_count=4`,
`total_test_harness_cache_hit_count=10`, and all no-cheat/integrity counters at
zero. The stricter sibling
`reports/strict_generator_mlx_rung_decode_sweep_plan_prefix_plan_aux_30m_rungs_broader_resource_probe_v1.json`
correctly failed closed at a 20s child budget because `rung_25000000` took
`21274` ms. The larger replay therefore proves budget enforcement and route
fail-closed behavior over more private rows, not learned capability: behavior
is still `0/10`, `production_route_eligible=false`, and
`route_state=fail_closed_behavior_quality_zero`.
`reports/resource_mlx_route_readiness_gate.json` is now `GREEN` with `0` failed
checks and `0` failed expected-invalid controls, so Phase 8 is wired as a
resource-route readiness surface. This does not claim model quality, production
MLX routing, or CUDA/MLX/Metal parity.

| Phase | Surface | Current state | Next implementation target |
| --- | --- | --- | --- |
| 0 | Stabilize current truth | Implemented/green registry gate/registered roadmap matrix; cleanup queue `16/16` steward-covered; bounded tmp/smoke cleanup canary complete | Continue governed retention/consolidation in bounded passes for steward-covered cleanup families; do not treat generated evidence history as active source lanes. |
| 1 | VIEA Execution Spine | Frozen | When a trusted peer is reachable, run one bounded registered Hive task submission and verify the emitted execution receipt records match the dry-plan authority/runtime-adapter/failure/resource contract. |
| 2 | Stable Capability Fields And Route Authority | Frozen | When a trusted peer is reachable, run one bounded registered Hive task submission and verify the emitted execution receipt records plus route-validator receipt against the shared VIEA contract. |
| 3 | Virtual Context Memory As Default Context Substrate | Partial | Complete VCM as a transactional context ABI: representation certificates, authority ceilings, copy-on-write snapshots, taint/deletion propagation, mandatory typed faults, and native KV/prefix-cache replay proofs. |
| 4 | Candidate Integrity And Learned Generation Accounting | Wired | Keep direct learned full-body quality receipts flowing through independent candidate integrity; semantic behavior repair stays in Phase 10. |
| 5 | Daily-Use Assistant Runtime And Dogfood Trace Loop | Wired | Turn repeated successful real assistant traces into guarded procedural-memory candidates and continue improving the code-assistant generator wall through Phase 10 rather than creating a parallel assistant lane. |
| 6 | Deterministic Tool And Search Substrate | Wired | Keep tool-assisted public/tool-use measurement ledgers separate from model-only scores as future public adapters are added; do not allow deterministic tool receipts to support learned-generation claims. |
| 7 | Teacher And Data Governance | Wired | Once additional governed teacher/self-generated cycles exist, compute and display the multi-cycle trend delta in the existing operator-visible `teacher_governance` surface. |
| 8 | Resource, Cost, And Mac Acceleration Routing | Wired | Keep the resource/MLX route gate current; production routing stays disabled until Phase 10 behavior is positive and parity remains separately proven. |
| 9 | Hive Policy-First Distributed Operation | Frozen | When peers are reachable, run one bounded registered Hive task submission and verify live execution receipts against the scheduler route-local VIEA contract. |
| 10 | Practical Neural Seed Survival Lane | Partial | T2/T3 private MLX evidence proves trainability/loss improvement, but T4 direct decode is still not broadly behavior-ready. A first DPO shadow update moved the private policy/reference preference gap, then its private replay failed behaviorally: `0` learned candidate rows and `0/16` heldout passes, with candidate integrity classifying the JSONL rows as fallback/template baselines. The new `strict_direct_body_emission_path_v1` profile activates target-side direct body-emission supervision (`128/128` matched rows, `8054` weighted positions, clean no-cheat counters). Local-return continuation plus a narrow guarded/default static-guard correction now gets simple-return private replay to `GREEN` with `2` emitted transformer/hybrid rows, `2/2` intended-behavior passes, and clean no-cheat counters. Follow-up strict decode hygiene blocks malformed `isinstance` first-argument chains, bare builtin type values used as runtime values, and constant-only control-flow conditions, but broad/private replay remains `RED` with `0` generated/admitted learned rows and `0/4` behavior passes, falling back to noncredit `return None` baselines. The next behavior-changing sequence is broad semantic/action body construction first, then bounded DPO/IPO scale, RLVR/GRPO, MTP, generate-verify-repair, lookahead/diffusion, and dense-vs-MoE/scale ablation only after non-fallback nontrivial candidates exist beyond the narrow simple-return replay. |
| 11 | SymLiquid Discovery Lane Verdict | Wired | Refresh this verdict only after a new matched-compute comparator run; keep the practical transformer/hybrid route separate from protected SymLiquid discovery evidence. |
| 12 | Public Calibration And Residual Mining Discipline | Wired | Public calibration remains measurement-only. The execution plan keeps it blocked until private semantic behavior improves and a fresh, non-consumed surface passes the proposal gate; exact consumed-surface reruns stay refused. |
| 13 | Semantic IR And Substrate-Neutral Reasoning Atoms | Partial | Connect semantic IR to real generator failures: failed atom -> localized repair -> dependent obligation replay, with scope-change ledgers when requirements change. |
| 14 | Compression, Proof, And Claim Evidence Records | Partial | Hard report-family budgets are now green with archive-pointer replay verification; continue receipt-faithfulness trap/replay audits, epistemic TCB/auditor records, claim-belief revision transitions, public-safe evidence-pack exports, and current-reference-aware checkpoint compaction. |
| 15 | Procedural Memory And Toolification | Partial | Verified trajectories must become durable procedural tools and trie/lookahead sources with monitoring, retirement, drift checks, and no learned-generation credit. |
| 16 | MoECOT And Octopus Router Integration | Partial | Add verifier-capacity budgets, residual-obligation ledgers, governance-tax accounting, and more diverse real task-to-arm traces before major route-policy changes. |
| 17 | Simulation, Fidelity, And World-Model Contracts | Wired | Expand the same fidelity/counterfactual/world-adapter/failure-boundary semantics from the current bounded planning adapter to future real simulators and resource adapters without turning simulation into deployment evidence. |
| 18 | Governance Rights, Constitutional Predicates, And Failure Boundaries | Frozen | When a trusted peer is reachable, validate remote Hive artifact endpoint citations through the same operator governance audit, VIEA route-validator, and claim-ledger path. |
| 19 | Book-To-Theseus Backlog And Evidence Synchronization | Wired | Keep future AI_book source drift sticky until exact source-sync review decisions clear or update the affected phase contracts. |

Out of scope for that goal:

- Long training runs.
- Public benchmark runs.
- New benchmark families.
- New strict-generator target modes, marker families, or decode guards.
- Claims that a roadmap item is complete without a registered surface and a
  runnable integration path.

## Training And Inference Execution Roadmap

The repository has enough clean smoke evidence to preserve the training
sequence, but it is **not** ready to make governed training the main focus
again until the strict pre-training architecture gate is green. It is also not
ready to claim model-only production inference. The execution contract is:

1. `T0_preflight_freeze`: refresh roadmap, registry, data-admission, and
   execution-plan gates.
2. `T1_data_and_context_manifest`: materialize the exact admitted
   private/licensed rows, governed teacher rows if available, dogfood metadata,
   and VCM context packets. Public benchmark taint is excluded from training.
3. `T2_private_training_smoke`: run a short private MLX transformer/hybrid
   smoke that writes a digest-bound checkpoint, reports throughput, and keeps
   all no-cheat counters at zero.
4. `T3_bounded_private_training_rung`: only after T2 passes, run the longer
   private stateful-update/finalizer semantic repair rung against the current
   wall: `missing_semantic_update_value`, `type_handling`, `wrong_answer`, and
   `missing_finalizer`.
5. `T4_private_eval_and_candidate_integrity`: replay the checkpoint through
   independent candidate integrity, blind-flow audit, private verifier,
   selector ablation, VCM-on/off, and STS-on/off under equal candidate budget.
6. `T5_local_assisted_inference_canary`: use the assistant runtime locally with
   VCM, deterministic tools, planning, uncertainty, evidence refs, and
   accepted/missed/ignored/corrected/completed feedback. This is assisted local
   inference, not a model-only ChatGPT-grade claim.
7. `T6_dogfood_feedback_to_training_rows`: convert real outcomes into
   metadata-first training pressure and procedural-memory candidates with raw
   private text disabled by default.
8. `T7_fresh_public_calibration_proposal`: only after private semantic
   behavior improves, propose a fresh frozen public calibration surface. The
   proposal gate must pass; consumed exact surfaces must not rerun; public
   artifacts never become training rows.
9. `T8_hive_fleet_training_scaleout`: keep blocked while trusted peers are
   unreachable. When unblocked, use only registered bounded Hive tasks with
   input/output artifacts, backend, owner node, score, merge result, and stale
   lease recovery.

Current gate interpretation:

- Current `reports/training_inference_execution_plan_gate.json` state is `RED`
  because strict pre-training architecture readiness is `RED`. This is not a
  no-cheat failure; it is the roadmap correctly saying that the book-derived
  partial phases should be implemented or falsified before training/public
  calibration becomes the main focus again.
- Completed evidence: `T2` private MLX smoke is `GREEN`; `T3` bounded private
  adaptation is `GREEN`; `T4` replay ran and is `RED` because direct decode
  emitted `0` candidate rows.
- Ready now: roadmap/architecture implementation work on phases `3`, `10`,
  `13`, `14`, `15`, and `16`; targeted T4 repair analysis belongs under Phase
  10, but not as a claim that training focus is globally unblocked.
- Planned but not yet ready: the next full `T3`/`T4` rerun after the T4
  no-candidate decode fault is repaired, plus `T6`.
- Correctly blocked: `T7` public calibration and `T8` Hive scaleout.
- Explicit non-claims: no model-only general chat serving, no public transfer
  win, no production MLX route, no CUDA/MLX/Metal parity claim, and no ASI
  capability claim.

## Current Generator Wall

The strict MLX transformer/hybrid lane now has a clearer boundary:

- Simple safe-head/default-return body-token replay is healthy:
  `reports/strict_generator_mlx_decode_eval_simple_return_multi_tier_balanced_plan_semantic_slots_body_exact_decision_prefix_guided_cap80_action_trace_replay_regression_v1.json`
  is `GREEN` with `8/8` rank-1/pass-if-any.
- Private loop-operation weighting now exists and is trainable:
  `reports/strict_generator_mlx_private_adaptation_loop_operation_from_balanced_v1.json`
  is `GREEN` and improves heldout LM loss from `1.160228` to `0.780920` on
  admitted private loop rows.
- Loop/action replay is still not promotion-grade. With the old forced-update
  scaffold,
  `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_v1.json`
  emits `15` syntactically valid candidates, but verifier pass remains `0.0`
  and all candidates are `shallow_identity_accumulation`.
- With the hand-coded update injection removed,
  `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_no_forced_update_v3.json`
  exposes the deeper wall: the learned path tends toward
  `early_return_inside_loop` and emits only `2` loadable candidates on cap-8.
- `reports/strict_generator_mlx_decode_eval_loop_operation_from_balanced_cap8_action_trace_v1.json`
  now carries `body_action_trace` mismatch labels for residual mining:
  `loop_without_decision_or_state_update`, `missing_numeric_accumulation`,
  `missing_list_construction`, and `missing_windowed_finalizer`.
- `reports/strict_generator_mlx_private_adaptation_action_trace_replay_v1.json`
  proves action-trace-aware private replay is wired: it is `GREEN`, improves
  heldout LM loss from `0.769207` to `0.669650`, and reports weighted
  accepted/rejected token positions from the failed candidates' trace labels.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_v1.json`
  remains `YELLOW`; the new training signal has not yet moved loop verifier
  pass above `0.0`.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_summary_probe_v1.json`
  adds the report-level `body_action_trace` aggregate that the next loop can
  consume directly: both traced loop candidates still return inside the loop,
  both miss real decision/state-update structure, both miss list/windowed
  finalization, and the summary records zero public/test/solution use and zero
  candidate-generation credit.
- `reports/strict_generator_mlx_decode_eval_action_trace_replay_cap8_loop_return_blocked_v1.json`
  is `RED` with `0` emitted loop candidates after nested loop returns are
  blocked in the loop-plan exploration path. This is useful negative evidence:
  the checkpoint currently depends on the bad nested-return escape instead of
  learning a valid update/finalizer path.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_action_v1.json`
  adds exact target-span weighting for AST loop-body decisions, loop-body
  updates, and top-level finalizers. It is `GREEN`, matches `192/192` private
  loop rows, weights `7,624` target positions, and improves heldout LM loss
  from `0.669650` to `0.619043` without public rows, teacher/runtime
  inference, or candidate-generation credit.
- `reports/strict_generator_mlx_decode_eval_loop_statement_action_trace_label_cap8_v1.json`
  is `YELLOW`: early loop returns are gone and `2` loop candidates emit, but
  the current failure is now `continue` before `out.append(value)`. The
  report-level trace adds `unreachable_loop_update_after_control_flow` so the
  next private replay can target the actual reachable-update wall.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_action_unreachable_v1.json`
  proves that the new unreachable-update label can be consumed in private
  replay and improves LM loss again, but
  `reports/strict_generator_mlx_decode_eval_loop_statement_action_unreachable_cap8_v1.json`
  is `RED` with `0` loop candidates. That overcorrection is negative evidence:
  penalizing the bad control-flow token alone does not yet teach a replacement
  update/finalizer action.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_reachable_v1.json`
  refines that objective so bare `continue`/`break` terminals are kept only as
  span-matching context and excluded from positive weighting. It is `GREEN`,
  improves heldout LM loss from `0.669650` to `0.618266`, and reports
  `NAME:continue` as `103` excluded positive positions rather than a boosted
  update token.
- `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_cap8_v1.json`
  and the wider `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_cap8_wide_v1.json`
  are both `RED` with `0` loop candidates. The simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_statement_reachable_regression_v1.json`
  remains `GREEN` with `8/8`, so the new wall is isolated to learned loop
  update/finalizer synthesis.
- `reports/strict_generator_mlx_decode_eval_loop_statement_reachable_starvation_cap4_v1.json`
  now exposes top-level `split_decode_starvation`: failed loop beams are still
  inside the loop without an update or local return, with runaway nested
  expression/condition prefixes. This makes zero-candidate loop failures
  auditable from the report itself.
- `reports/strict_generator_mlx_private_adaptation_loop_statement_update_finalizer_v1.json`
  role-filters the private span objective to only `loop_body_update` and
  `top_level_finalizer`. It is `GREEN`, improves heldout LM loss
  `0.669650 -> 0.610516`, filters out `177` decision spans, and gives no
  public/teacher/runtime/template credit.
- `reports/strict_generator_mlx_decode_eval_loop_statement_update_finalizer_cap8_v1.json`
  is `YELLOW`: the zero-candidate wall moves to `2` emitted loop candidates,
  including one reachable `out.append(value)` followed by `return out`. Verifier
  pass remains `0` because the body is still shallow identity and misses the
  required numeric/windowed/list finalizer semantics. The simple-return
  regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_statement_update_finalizer_regression_v1.json`
  stays `GREEN` with `8/8`.
- `reports/strict_generator_mlx_private_adaptation_loop_semantic_operation_v1.json`
  adds private semantic-operation span weighting over admitted loop targets. It
  is `YELLOW`, but the hard training gates pass: `192/192` loop rows match,
  `4,533` operation/finalizer token positions are weighted, heldout LM loss
  improves `0.610516 -> 0.565156`, public rows remain `0`, external inference
  remains `0`, and candidate-generation credit remains `0`. The soft warning
  is that the already-perfect auxiliary plan head lost a small amount of loss
  margin.
- `reports/strict_generator_mlx_decode_eval_loop_semantic_operation_default_after_flag_cap8_v1.json`
  shows the honest result: semantic weighting alone still emits only `1` loop
  candidate on cap-8, verifier pass remains `0`, and the residual is still
  shallow identity plus missing list/windowed finalizer behavior. The
  simple-return regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_semantic_operation_regression_v1.json`
  remains `GREEN` with `8/8`.
- `reports/strict_generator_mlx_decode_eval_loop_semantic_operation_no_shallow_append_cap8_v1.json`
  is a negative decode-hygiene experiment. The opt-in
  `--block-shallow-loop-identity-update` flag blocks the known-bad direct
  `accumulator.append(loop_var)` continuation for non-identity plans, but the
  checkpoint then emits `0` candidates. This flag is explicit/off-by-default;
  it proves the model has not yet learned an alternate semantic loop-update
  expression path.
- `reports/strict_generator_mlx_decode_eval_loop_operation_hints_source_only_cap8_v1.json`
  is `RED`: prompt-only operation hints can be emitted under the strict source
  audit with no forbidden marker hits or solution/test leakage, but the current
  checkpoint emits `0` loop candidates without adaptation.
- `reports/strict_generator_mlx_private_adaptation_loop_operation_hints_v1.json`
  is `GREEN`: adapting on the prompt-operation-hint source contract improves
  heldout LM loss `0.601345 -> 0.507321` and source-contrastive gap
  `0.488449 -> 0.565484` while preserving clean no-cheat counters. Its loop
  decode `reports/strict_generator_mlx_decode_eval_loop_operation_hints_cap8_v1.json`
  remains `YELLOW` with `1` emitted candidate and `0` pass; the candidate is
  still shallow identity with missing numeric/list/windowed semantics.
- `reports/strict_generator_mlx_private_adaptation_loop_slot_prefix_v1.json`
  and
  `reports/strict_generator_mlx_private_adaptation_loop_slot_prefix_low_v1.json`
  prove a semantic-slot prefix training hook now exists and is audited:
  `1,525` private target prefix positions are weighted across
  update/finalizer/guard/init/return-shape/loop-source roles, LM loss improves,
  and no public/teacher/runtime/template credit is used. Their loop decodes are
  negative: `reports/strict_generator_mlx_decode_eval_loop_slot_prefix_cap8_v1.json`
  and
  `reports/strict_generator_mlx_decode_eval_loop_slot_prefix_low_cap8_v1.json`
  are both `RED` with `0` loop candidates, while simple-return regressions stay
  `GREEN`. Slot-prefix CE alone is therefore not the next winning route.
- Decode-starvation replay is now a registered private repair signal, not a
  new benchmark lane. `reports/strict_generator_mlx_private_adaptation_loop_starvation_replay_v1.json`
  is `GREEN` after selecting `1` failed private replay candidate plus `8`
  private decode-starvation beam previews as rejected bodies. It keeps
  `public_training_rows=0`, `external_inference_calls=0`, clean source audits,
  and `candidate_generation_credit=0`. The corresponding unguarded loop decode
  is still `RED` with `0` emitted loop candidates, so the replay signal alone
  does not solve block closure.
- The opt-in `--enable-loop-progress-guard` separates loadability from
  semantics. With shallow-identity blocking also enabled,
  `reports/strict_generator_mlx_decode_eval_loop_progress_guard_cap8_v1.json`
  remains `RED`: preventing direct `append(loop_var)` before the model learns a
  replacement update expression collapses search into malformed expressions.
  With the progress guard but no identity block,
  `reports/strict_generator_mlx_decode_eval_loop_progress_guard_no_identity_block_cap8_v1.json`
  is `YELLOW` and emits `12` integrity-verified, loadable loop candidates.
  Behavior stays `0/8`; all candidates are shallow identity loops.
- The follow-up private behavior replay
  `reports/strict_generator_mlx_private_adaptation_loop_behavior_replay_v1.json`
  consumes those `12` shallow identity failures plus `4` remaining starvation
  prefixes. Its decode
  `reports/strict_generator_mlx_decode_eval_loop_behavior_replay_progress_guard_cap8_v1.json`
  remains `YELLOW` with `12` loadable candidates and `0` behavior pass, while
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_behavior_replay_regression_v1.json`
  stays `GREEN` with `8/8`. This proves the current wall is semantic
  update-expression and block synthesis, not loadability, candidate replay
  plumbing, or simple-return regression.
- The first explicit expression-synthesis objective is implemented in
  `scripts/strict_generator_mlx_private_adaptation.py`. It extracts
  expression-level AST spans from admitted private targets, including
  non-identity loop update RHS/arguments and semantic top-level finalizer
  expressions, then weights those target tokens without rendering code or
  granting learned-generation credit. The report
  `reports/strict_generator_mlx_private_adaptation_loop_expression_synthesis_v1.json`
  is `YELLOW` only on the already-saturated auxiliary-plan-loss soft gate; it
  matches all `192` private loop training rows and weights `3,479` expression
  token positions while preserving `0` public rows and `0` external inference.
- The decode result is honest negative evidence. With progress guarding and no
  identity block,
  `reports/strict_generator_mlx_decode_eval_loop_expression_synthesis_progress_guard_cap8_v1.json`
  remains `YELLOW`: `12` loadable/integrity-verified loop candidates, `0/8`
  behavior pass, and unchanged shallow-identity residual labels. The
  simple-return regression stays `GREEN` at `8/8`. With identity blocking
  enabled,
  `reports/strict_generator_mlx_decode_eval_loop_expression_synthesis_progress_guard_block_identity_cap8_v1.json`
  is `RED` with `0` emitted candidates, though starvation traces now show some
  beams reaching update calls/local returns before failing expression/block
  closure. This narrows the next wall to block-level decode state or model
  architecture, not another scalar CE boost.
- `scripts/strict_generator_mlx_decode_eval.py` now has two additional
  opt-in, no-credit loop decode guards: `--enable-expression-closure-guard`
  and `--enable-expression-value-guard`. The closure guard uses generated
  prefix syntax only to close delimiters, terminate statements, dedent finished
  loop blocks, and start top-level local returns. The value guard rejects
  generated-prefix pathologies such as empty update-call arguments, direct set
  literals in append/update calls, and bare builtin objects closed as values.
  Both report `candidate_generation_credit=0`, use no tests/solutions/public
  artifacts, and must not be counted as learned generation.
- The closure guard separates loadability from behavior. With identity blocking
  enabled, the previous `0`-candidate loop run becomes
  `reports/strict_generator_mlx_decode_eval_loop_expression_closure_guard_cap8_v1.json`:
  `23` loadable/integrity-clean loop candidates, `0/8` behavior pass, and
  shallow identity reduced from `12` to `2`. The easy-lane regression
  `reports/strict_generator_mlx_decode_eval_simple_return_expression_closure_guard_regression_v1.json`
  stays `GREEN` with `8/8`.
- A closure-replay adaptation
  `reports/strict_generator_mlx_private_adaptation_loop_expression_closure_replay_v1.json`
  is `YELLOW` only on soft gates and keeps hard no-cheat counters clean:
  private rows only, `0` public training rows, `0` external inference, source
  audit clean, and heldout LM improved. Its decode
  `reports/strict_generator_mlx_decode_eval_loop_expression_closure_replay_cap8_v1.json`
  emits `26` loadable loop candidates but still scores `0/8`. The
  simple-return regression remains `GREEN` with `8/8`.
- The expression-value guard improves expression hygiene but exposes the deeper
  wall. `reports/strict_generator_mlx_decode_eval_loop_expression_value_guard_cap8_v2.json`
  reduces emitted loop candidates to `11`, removes several empty-tuple/bare-set
  failures, and produces more plausible calls such as `abs(value)`, but still
  scores `0/8` and starves several loop tasks. The paired simple-return report
  `reports/strict_generator_mlx_decode_eval_simple_return_expression_value_guard_regression_v1.json`
  stays `GREEN` with `8/8`.
- A stronger loop state-transition adaptation
  `reports/strict_generator_mlx_private_adaptation_loop_state_transition_v1.json`
  is `GREEN`: `256` private loop rows, `64` heldout replay rows, `0` public
  training rows, `0` external inference, no fallback/template/router credit,
  heldout LM improved, and the span hooks weighted `180` loop-body decisions,
  `199` loop-body updates, `399` semantic loop updates, and `787` expression
  spans. Its decode
  `reports/strict_generator_mlx_decode_eval_loop_state_transition_cap8_v1.json`
  remains `YELLOW` with `14` loadable candidates and `0/8` behavior pass; it
  even introduces some worse collection-mutation sequences. The simple-return
  regression
  `reports/strict_generator_mlx_decode_eval_simple_return_loop_state_transition_regression_v1.json`
  stays `GREEN` with `8/8`.
- The learned state-transition prefix representation is now real target/vocab
  machinery rather than a report-only idea. `scripts/neural_seed_token_decoder_support.py`
  emits `SLOT:STATE_*` target tokens for loop branch/update/finalizer shape
  before `SLOT:BODY_START`; `scripts/strict_generator_mlx_decode_eval.py`
  parses those slots into loop expectations and scores branch/update
  consistency without rendering code. A fresh MLX smoke checkpoint,
  `reports/strict_generator_mlx_pretraining_state_slots_smoke_v1.json`, is
  `GREEN`: target mode `plan_semantic_slots_body_tokens_v1`, `14`
  state-slot vocabulary entries, `906,437` optimizer token positions,
  heldout LM loss improved `8.192360 -> 4.241084`, `0` public training rows,
  and `0` external inference calls.
- State-slot private adaptation is also clean:
  `reports/strict_generator_mlx_private_adaptation_state_slots_loop_v1.json`
  is `GREEN` with `256` private loop rows, heldout LM improvement, and
  explicit prefix weighting over `1,882` `SLOT:STATE_*` positions plus the
  existing loop decision/update/expression spans. Decode evidence is still
  negative on behavior. Before prefix coverage tightening,
  `reports/strict_generator_mlx_decode_eval_state_slots_loop_cap8_v1.json`
  emitted `26` loadable candidates with `0/8` behavior and no state slots in
  the final prefixes. After requiring state-slot emission,
  `reports/strict_generator_mlx_decode_eval_state_slots_required_loop_cap8_v1.json`
  emitted state slots in every inspected prefix but still scored `0/8`.
  After requiring loop-source/init/update/state category coverage,
  `reports/strict_generator_mlx_decode_eval_state_slots_category_loop_cap8_v1.json`
  kept `26` candidates and `0/8` behavior while reducing residual spread to
  mostly wrong plan/operand failures (`missing_rle_branch_or_update`,
  `loop_without_decision_or_state_update`). Simple-return regression remains
  `GREEN` in
  `reports/strict_generator_mlx_decode_eval_state_slots_category_simple_return_regression_v1.json`
  with `8/8`.
- A first causal operand-binding prefix representation now exists. The strict
  target builder emits `SLOT:BIND_*` tokens for loop/branch operand use,
  update operand use, and accumulator/finalizer binding. These are private
  target tokens only: they are stripped before compilation, never render code,
  use no tests/solutions/public payloads, and grant zero learned-generation
  credit. The decoder now parses them into loop adequacy metadata, filters
  impossible loop-source slots against the visible callable signature, and
  exposes `--require-binding-prefix-groups` as an explicit ablation rather
  than default behavior.
- The fresh binding-slot MLX smoke
  `reports/strict_generator_mlx_pretraining_operand_bindings_smoke_v1.json`
  is `GREEN`: target vocab `2889`, `14` `SLOT:BIND_*` entries, `14`
  `SLOT:STATE_*` entries, `901,017` optimizer token positions, heldout LM loss
  improved `8.125375 -> 4.351802`, `0` public training rows, and `0` external
  inference calls.
- The loop-only binding adaptation
  `reports/strict_generator_mlx_private_adaptation_operand_bindings_loop_v1.json`
  is `GREEN`: `256` private loop rows, `6,046` semantic-prefix weighted
  positions, `1,909` binding-slot positions, `1,882` state-slot positions,
  hard no-cheat counters clean, and no fallback/template/router/tool credit.
  Its default loop decode
  `reports/strict_generator_mlx_decode_eval_operand_bindings_default_loop_cap8_v2.json`
  improves loadability over the fresh checkpoint (`6/8` integrity-verified and
  runtime-loaded attempts versus `0/8` before adaptation) but remains `0/8`
  behavior with `0.0` nontrivial-return rate. The explicit
  `--require-binding-prefix-groups` ablation
  `reports/strict_generator_mlx_decode_eval_operand_bindings_required_groups_loop_cap8_v2.json`
  is negative evidence: richer forced binding coverage regresses integrity to
  `4/8` and still scores `0/8`.
- The easy-lane tradeoff is not solved. The loop-only binding checkpoint
  `reports/strict_generator_mlx_decode_eval_operand_bindings_simple_return_regression_v1.json`
  keeps `8/8` integrity/runtime load but scores `0/8` and emits inert loop
  bodies for simple-return rows. A balanced private adaptation
  `reports/strict_generator_mlx_private_adaptation_operand_bindings_balanced_v1.json`
  is `GREEN` on training mechanics (`384` rows, `2,122` binding positions,
  `2,160` state positions) but its simple-return decode
  `reports/strict_generator_mlx_decode_eval_operand_bindings_balanced_simple_return_v1.json`
  regresses to `0/8` integrity due syntax failures. This is negative evidence
  against more scalar prefix weighting or naive tier mixing as the next fix.
- Specialist-head routing now exists inside the canonical strict MLX decoder
  surface rather than as a new lane. `scripts/strict_generator_mlx_decode_eval.py`
  accepts `--specialist-checkpoint-report tier=report.json` for existing
  private replay tiers and records `strict_generator_private_replay_specialist_head_route_v1`
  route receipts with zero candidate-generation credit. The route chooses only
  among learned checkpoints; generation still receives strict prompt/signature
  source text and the selected checkpoint, not tests, solutions, verifier
  labels, public payloads, answer templates, renderers, tools, or fallbacks.
- The first specialist-head proof,
  `reports/strict_generator_mlx_decode_eval_specialist_heads_profiled_v1.json`,
  routes `simple_return` to the state-slot checkpoint that preserved the
  safe-head/default lane and `loop_accumulate` to the operand-binding loop
  checkpoint. It is `YELLOW` with route profiles enabled: simple-return stays
  `4/4` intended-behavior passed, loop remains `0/4` behavior with `3`
  runtime-loaded attempts and `1` parse failure, overall candidate integrity
  is `7/8`, and all route records report `0` public training rows, `0`
  external inference, and `0` fallback/template/router/tool credit. This is a
  real architecture improvement because it prevents the loop head from erasing
  the easy lane, but it is not a loop semantic promotion.
- The newest loop-value hygiene pass is implemented and sharper. The
  expression-value guard now rejects generated bodies that use bare builtin/type
  objects as runtime values or obvious invalid call argument types such as
  `round(round, int)` and `range(int, int)`, while the static-coherence helper
  now correctly preserves valid `max(x, y)` / `min(max(x, lo), hi)` patterns.
  `reports/strict_generator_mlx_private_adaptation_value_guard_loop_v2.json`
  is the clean adaptation evidence: private rows only, no public training rows,
  no external inference, heldout LM loss improved `4.288703 -> 0.85482`, and
  corrected target static-guard pass rates rose to about `0.80` on train and
  `0.79` on heldout. The earlier
  `reports/strict_generator_mlx_private_adaptation_value_guard_loop_v1.json`
  should be treated only as historical negative/diagnostic evidence because it
  used the overbroad `min`/`max` guard.
- Decode from the clean value-guard checkpoint still does not solve behavior.
  `reports/strict_generator_mlx_decode_eval_value_guard_adapted_loop_v1.json`
  emits `6` integrity-clean transformer/hybrid loop candidates with `0/8`
  pass, but they are shallow identity/self-append candidates such as
  `out.append(ch)` or `out.append(out)`. With the stricter direct-identity
  blocker and corrected semantic-update audit,
  `reports/strict_generator_mlx_decode_eval_value_guard_no_identity_loop_v2.json`
  emits only `3` integrity-clean candidates, still `0/8` pass. The remaining
  bodies are constants or accumulator self-appends, with residual counts
  `missing_semantic_update_value=3`, `append_accumulator_to_itself=1`,
  `loop_without_decision_or_state_update=1`, and `missing_gcd_call=1`.
  This is honest progress in failure localization, not a capability promotion.
- The semantic-plan label-space bug has been fixed. The plan auxiliary loss
  and evaluation now restrict classification to `SLOT:PLAN_*` target-vocab IDs,
  matching the decode-time plan-prefix choice space. The clean checkpoint
  `reports/strict_generator_mlx_private_adaptation_semantic_plan_subspace_private_smoke_v1.json`
  is `GREEN`, improves heldout LM loss `5.539204 -> 4.138901`, and improves
  heldout plan loss `6.731428 -> 3.111023` with all no-cheat counters clean.
  Its broad4 decode
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_broad4_v1.json`
  emits integrity-clean/loadable candidates and recovers plan-prefix diversity
  (`RLE_ENCODE`, `INTERVAL_COVERAGE`, `WINDOWED_DELTAS`), but remains `0/4`
  behavior. This means plan classification is no longer the primary wall.
- Two private replay repairs were tested after the plan-subspace fix and both
  are negative evidence for promotion. First,
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_algorithmic_replay_v1.json`
  generated fresh failed `private_train_replay` candidates from the current
  checkpoint. Then
  `reports/strict_generator_mlx_private_adaptation_semantic_plan_subspace_algorithmic_replay_private_v1.json`
  used those failed private candidates as bounded negative/pairwise replay and
  improved heldout LM loss `4.199799 -> 1.845056`. The broad4 follow-up
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_algorithmic_replay_broad4_v1.json`
  still scored `0/4` and regressed to shallow `out.append(value)` loops.
  Blocking shallow identity updates in
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_algorithmic_replay_block_identity_broad4_v1.json`
  reduced that exact failure family but exposed weak substitutions such as
  `max(value)` and `int(value)`, still `0/4`.
- The stricter escape replay also failed honestly. The guarded private replay
  report
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_algorithmic_replay_block_identity_train_v1.json`
  exposed decode starvation and degenerate repeated-return/parenthesis beams
  under the identity guard. The follow-up adaptation
  `reports/strict_generator_mlx_private_adaptation_semantic_plan_subspace_algorithmic_replay_escape_private_v1.json`
  improved private heldout LM loss `1.840928 -> 1.358781`, but broad4 decode
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_algorithmic_replay_escape_block_identity_broad4_v1.json`
  remained `0/4` and produced only one heldout task with completed candidates.
  The current wall is therefore learned body continuation and operand/finalizer
  binding under a real sequence model, not syntax, loadability, plan-head
  classification, or private replay plumbing.
- Statement-sequence prefix metadata was tested as a stricter learned target,
  not as a renderer. The new target mode
  `plan_semantic_stmt_slots_body_tokens_v1` adds private target-side
  `SLOT:STMT_*` prefix tokens before `SLOT:BODY_START`; those tokens are
  stripped before compilation and are not code-generation credit by themselves.
  `reports/strict_generator_mlx_pretraining_statement_slots_private_smoke_v1.json`
  is `GREEN`: MLX GPU, about `5.2M` parameters, `300000` token positions,
  heldout LM loss `8.385571 -> 4.152093`, heldout plan accuracy `0.0 ->
  0.1875`, and no public/external/template counters. The first broad4 decode,
  `reports/strict_generator_mlx_decode_eval_statement_slots_broad4_v1.json`,
  emitted `4` syntax-valid, integrity-clean transformer/hybrid candidates but
  stayed `0/4` functional pass.
- One algorithmic private adaptation on the statement-slot checkpoint also
  failed to move behavior. `reports/strict_generator_mlx_private_adaptation_statement_slots_algorithmic_private_v1.json`
  is `GREEN` and improved private heldout LM loss `4.411463 -> 3.052563`
  while weighting statement, update, finalizer, binding, and expression spans
  from admitted private targets only. The follow-up
  `reports/strict_generator_mlx_decode_eval_statement_slots_algorithmic_broad4_v1.json`
  emitted `7` loadable/integrity-clean candidates, but broad4 remained `0/4`
  and the summaries still report zero nontrivial-return evidence. This is
  negative evidence for prefix/weight tweaks as the main repair path: they
  improve candidate coverage and shape, not semantic task solving.
- A private-train replay check showed that the current statement-slot
  checkpoint has no self-generated positive pool yet.
  `reports/strict_generator_mlx_decode_eval_statement_slots_algorithmic_train_replay16_v1.json`
  emitted `28` integrity-clean, runtime-loaded candidates on private-train
  replay rows, but `0` verifier-passed intended behavior. The dominant
  post-generation labels were missing windowed finalizers and one missing GCD
  call, so the replay data is useful as rejected-candidate pressure but not as
  verified self-generated training data.
- Action-trace pairwise replay was then tested against those failed
  private-train candidates. `reports/strict_generator_mlx_private_adaptation_statement_slots_action_trace_pairwise_private_v1.json`
  is `GREEN`, selected `28` runtime-loaded wrong private replay candidates,
  applied bounded unlikelihood plus accepted-over-failed pairwise pressure, and
  improved private heldout LM loss `3.010078 -> 2.434338`. The broad4 replay
  `reports/strict_generator_mlx_decode_eval_statement_slots_action_trace_pairwise_broad4_v1.json`
  remained `0/4`; it emitted `5` loadable, integrity-clean candidates and
  shifted the residual shape toward shallow/incorrect update semantics
  (`loop_without_decision_or_state_update`, `shallow_identity_accumulation`,
  `missing_semantic_update_value`). This is negative evidence for action-trace
  pairwise replay as a standalone fix.
- A simpler private replay tier did not provide an escape hatch.
  `reports/strict_generator_mlx_decode_eval_statement_slots_pairwise_simple_replay16_v1.json`
  emitted `32` integrity-clean, runtime-loaded candidates on the
  `simple_return` replay tier, but still scored `0` verifier-passed intended
  behavior and over-routed all candidates into `WINDOWED_DELTAS` loop bodies.
  That means the current checkpoint cannot yet supply verifier-positive
  self-generated rows even on the easiest replay probe.

Roadmap implication: the next generator work should not add another renderer,
template, router, or benchmark lane. It should add verifier-shaped private
operation learning for loop bodies that is not implemented as a fixed update
renderer. The immediate repair target has moved from candidate emission to
stateful operation-expression synthesis and block closure: projected/windowed
list construction, numeric accumulation, branch-conditioned accumulator updates,
statement sequencing, and finalizer selection from prompt/signature evidence.
Prompt operation hints, semantic-slot prefix CE, starvation replay, the
loop-progress guard, expression-synthesis weighting, closure/value hygiene,
state/binding slots, and specialist-head routing are implemented but
insufficient. They make the wall easier to see; they do not yet produce loop
behavior passes. The next non-naive repair should keep the routed survival
architecture and improve the loop specialist itself with a larger/fresher
transformer-hybrid checkpoint, verifier-positive operand-span/state-transition
objectives, or a genuinely learned loop-specific AST/state-transition head
whose supervision is tied to verifier-positive behavior. It should not keep
turning the same shallow loops through more scalar loss boosts or richer prefix
metadata and then call lower LM loss progress. Because the private replay pass
found no self-generated verifier positives, including on a simpler replay
tier, the next viable repair is not another pairwise replay loop on this
checkpoint. It is a stronger trainable state-transition/AST head or a
larger/fresher transformer-hybrid checkpoint that can learn executable
update/finalizer structure before replay mining can provide positive
self-generated rows. No fallback returns, no public benchmark training, and no
tool/search/guard/renderer credit as learned generation.

## Third-Pass AI Book Addendum

This pass focused on what the first roadmap still under-specified after the
obvious VIEA, SCF, VCM, PlanForge, and registry work was already captured. The
new work items below come from the AI_book implementation chapters and source
notes, but they are still roadmap items. They are not evidence that Theseus has
implemented them.

### A. Artifact Steward Layer

Book basis:

- `artifact-steward-agents-and-living-project-governance`
- `schemas/artifact_steward_charter.schema.json`
- `schemas/project_work_contract.schema.json`
- `schemas/contribution_ledger_entry.schema.json`
- `schemas/treasury_policy_record.schema.json`
- `schemas/event_taint_record.schema.json`
- `schemas/steward_action_decision.schema.json`
- `schemas/sunset_review_record.schema.json`

Theseus gap:

- Theseus has a project registry and roadmap, but no first-class steward
  charter that binds mission, non-goals, authority ceiling, evidence policy,
  allowed work, forbidden work, contribution ledger, and sunset criteria.
- Repo events, issue/PR text, generated reports, worker outputs, benchmark
  artifacts, and external prompts are not consistently taint-classified before
  they influence planning or training.
- Roadmap work is still mostly prose. It should compile into bounded project
  work contracts.

Add to Theseus:

- `PROJECT_STEWARD.yml` or an equivalent registry-owned steward record for
  Theseus itself.
- `ProjectWorkContract` records for roadmap tasks before autonomous execution.
- `EventTaintRecord` intake for untrusted issue text, PR text, benchmark
  payloads, teacher proposals, copied browser notes, and external model output.
- `ContributionLedgerEntry` records for human, model, teacher, tool, and Hive
  contributions, separating authorship, review, evidence credit, economic
  credit, and governance effect.
- `SunsetReviewRecord` or merge/deprecate records for stale scripts, report
  families, dead branches, and duplicated implementations.
- ATT-D integration so new scripts/docs/configs without steward/registry
  ownership are warnings or blockers depending on authority/risk.

Acceptance gates:

- A real Theseus cleanup or model-training task is represented as a project
  work contract before execution.
- The registry gate can report work-contract coverage and event-taint coverage.
- At least five existing generated/report families receive steward decisions:
  keep, merge, deprecate, archive, or delete after evidence review.
- No steward action can merge, spend, publish, delete training data, or promote
  a capability without an explicit authority basis.

Do not:

- Build a general autonomous project manager.
- Add treasury or funding automation before the project steward works for
  local code, reports, and evidence.
- Let the steward become a new sidecar doc that the registry ignores.

Implementation status:

- Initial registry-owned steward implementation is complete as of 2026-06-25.
- Canonical steward record: `configs/project_steward.json`.
- Registry integration: `configs/project_manifest_registry.json` now declares
  `project_steward_config`, the hard `project_steward_boundary` rule, and the
  live `project_steward_records` surface.
- ATT-D integration: `configs/attd_policy.json` assigns
  `configs/project_steward.json` to the `project_registry` role.
- Registry report evidence:
  `reports/theseus_project_registry.json` and
  `reports/theseus_project_registry.md` report `project_steward_status=GREEN`,
  `project_steward_hard_gap_count=0`, `project_steward_warning_count=0`,
  seven event-taint records, five steward decisions, and full major-surface
  module-card coverage.
- Validation commands run:
  `python3 -m py_compile scripts/theseus_project_registry.py`;
  `python3 -m json.tool configs/project_manifest_registry.json`;
  `python3 -m json.tool configs/project_steward.json`;
  `python3 -m json.tool configs/attd_policy.json`;
  `python3 scripts/theseus_project_registry.py --gate`;
  `python3 scripts/attd_analyzer.py`.
- Remaining related work: contribution ledger entries and sunset review
  records should be added when the roadmap reaches retention/cleanup execution.
  They are not required for the first steward gate because the current boundary
  is charter, work contract, event-taint, module-card, and steward-decision
  coverage.

### B. Policy Optimization Program

Book basis:

- `policy-optimization-and-learning-from-feedback`
- Source notes for DPO, IPO, ORPO, KTO, SimPO, ReMax, REINFORCE-style RLHF,
  DeepSeek-R1, DAPO, GSPO, S-GRPO, LongRLVR, and RLHF limitations.
- `schemas/policy_optimization_record.schema.json`

Theseus gap:

- Theseus has verifier labels, private replay, STS/VCM/router knobs, and
  dogfood events, but it does not yet treat planner, VCM, router, verifier,
  generation-mode, and generator updates as governed policy updates with a
  common record shape.
- Current strict generator work has many CE/weighting attempts that improve LM
  loss but not behavior. The book points to preference and reward design as the
  next controlled learning surface, not another scalar token-weight tweak.

Add to Theseus:

- A `PolicyOptimizationRecord` writer for every generator/ranker/router/VCM
  policy update.
- A local offline preference baseline for private candidate pairs:
  DPO-style or IPO-style first because private accepted/rejected candidate
  pairs already exist. Keep ORPO/KTO/SimPO as follow-up baselines only after
  the simplest preference path is reproducible.
- A verifier-reward/RLVR toy lane for exact private tasks where reward is
  functional verifier pass plus context/evidence adequacy, not style.
- Reward-hacking probes for each reward source: shallow identity loops,
  early returns, inert stubs, prompt-label shortcuts, verifier-only hacks,
  excessive reasoning/latency, and context-ignoring answers.
- Reasoning-budget and generation-mode policy experiments only when correctness
  and verification adequacy are preserved.

Acceptance gates:

- A policy update names target layer, feedback source, update constraint,
  drift bound, evaluation refs, governance gate, rollback plan, residuals, and
  non-claims.
- A private preference update beats its pre-update checkpoint on verifier pass
  or accepted-output quality, not merely LM loss.
- Reward-hacking probes run and are recorded before any policy update becomes
  default.
- Policy updates cannot expand authority or weaken no-cheat boundaries.

Do not:

- Treat reward as evidence.
- Optimize preference style while semantic verifier pass stays flat.
- Let public benchmark outcomes become training rows.

Implementation status:

- Initial governed policy optimization record/gate scaffolding is complete as
  of 2026-06-25. This is a lease/evidence gate, not implementation of the
  chapter's core behavior-change techniques.
- The first real DPO shadow update ran on 2026-07-06 through
  `scripts/strict_generator_mlx_private_adaptation.py` using a frozen
  reference checkpoint and private accepted/rejected replay pairs. Report
  `reports/strict_generator_mlx_private_adaptation_dpo_pairwise_smoke_20260706.json`
  is `YELLOW`: heldout LM loss improved from `1.419164` to `0.459953`, the
  policy-minus-reference accepted-vs-rejected gap moved by `+0.628971`, and
  no-cheat counters stayed clean (`public_training_rows=0`,
  `external_inference_calls=0`, fallback/template/router/tool credit `0`).
  This is policy-update evidence only. It is not a behavior-lift claim, not a
  default route, not public transfer, and not learned-generation promotion.
- The required DPO private replay then ran through
  `reports/strict_generator_mlx_decode_eval_dpo_pairwise_smoke_broad8_replay16_20260706.json`
  with the same bounded `8+8` private replay profile used against the
  pre-update checkpoint. It is `RED`: `generated_candidate_rows=0`,
  `split_passes` are `0` for both `broad_private_heldout` and
  `family_disjoint`, and the standalone candidate-integrity audit classifies
  all `16` candidate JSONL rows as `fallback_or_template` baselines with
  `integrity_verified_candidate_count=0`. The blind information-flow audit is
  `GREEN`, so the result is clean negative evidence rather than a tainted run.
- The actual behavior-changing policy updates are **not** complete yet: no
  governed DPO/IPO update or GRPO/RLVR verifier-reward update has beaten its
  pre-update checkpoint on private heldout decode/verifier behavior or been
  accepted as a default generator improvement.
- Canonical config: `configs/policy_optimization_program.json`.
- Gate implementation: `scripts/policy_optimization_gate.py`.
- Registry integration: the existing `model_governance_gates` surface now
  declares `reports/policy_optimization_program.json` and
  `reports/policy_optimization_program.md`, and its verification command
  includes `python3 scripts/policy_optimization_gate.py`.
- Steward integration: `configs/project_steward.json` now explicitly allows
  the policy optimization config/gate under
  `roadmap.generator_policy_optimization_v1`.
- Current policy records cover strict-generator private pairwise preference,
  STS ranker policy, VCM context policy, and Octopus/router policy. None is
  default-enabled. The gate requires behavior evidence, no authority expansion,
  no loss-only promotion, no deterministic renderer credit, and full
  reward-hacking probe coverage before default review.
- Report evidence:
  `reports/policy_optimization_program.json` reports
  `trigger_state=GREEN`, four records, nine required reward-hacking probes,
  zero hard gaps, zero warnings, zero default policies, and one record with
  behavior-lift evidence.
- Validation commands run:
  `python3 -m py_compile scripts/policy_optimization_gate.py`;
  `python3 -m json.tool configs/policy_optimization_program.json`;
  `python3 scripts/policy_optimization_gate.py`;
  `python3 scripts/theseus_project_registry.py --gate`;
  `python3 scripts/attd_analyzer.py`.
- Remaining related work: keep the DPO checkpoint quarantined and repair direct
  learned body emission first. The next patch should make strict
  prompt/signature decode emit non-fallback, non-template, top-level-return
  learned candidates under the same private replay profile. If that works,
  rerun the DPO/IPO update at a bounded scale with reward-hacking probes; only
  after behavior improves should a bounded GRPO/RLVR exact-private
  verifier-reward update run. The policy program is the gate for that work; it
  is not itself a behavior-lift claim.
- Follow-up syntax-pathology cleanup is now partially implemented in the
  existing strict body-token legality policy, not as a new lane: malformed
  comparison/membership chains, augmented assignment in expressions, uncalled
  method-attribute chains, excessive same-line subscripts, long flat boolean
  chains, comments, and builtin type objects in returns are blocked before they
  dominate beam search. Bounded DPO-checkpoint canaries still emit `0` accepted
  learned candidate rows and score `0/4`, while runtime drops from `74283` ms
  to roughly `21` seconds. The honest next action is therefore not another
  scalar DPO/GRPO run and not more guard accumulation; it is a stronger trained
  state-transition/AST/body-token emission objective that can create valid
  update/finalizer/top-level-return structure under the same no-cheat replay.
- A first version of that objective now exists as
  `strict_direct_body_emission_path_v1`. It proves the target-side AST span
  weighting path is live and private-only. The first local-return continuation
  decoder patch proves syntax emission is no longer completely starved under a
  weak private train-replay canary (`8` emitted rows, `8/8` independently
  integrity verified, no fallback returns), but strict nontrivial/top-level
  replay still rejects all learned rows before promotion-grade admission. The
  next patch should therefore improve semantic top-level-return quality and
  verifier-aligned body behavior, not another report-only profile and not
  broader RL/fast generation.
- The newest strict decode hygiene pass blocks runaway `isinstance((data) and
  ...` first-argument chains, bare builtin type values such as `max(list)`, and
  constant-only branch conditions such as `if -1:`. The narrow train-replay
  canary stays `GREEN` with `2/2` behavior passes, but the paired broad canary
  remains `RED` with `0` generated learned rows, `0/4` behavior passes, and
  only noncredit `return None` baseline JSONL rows. This is useful
  pathology-removal evidence, not a behavior claim; Phase 10 should now move
  toward stronger semantic/action body construction rather than accumulating
  more one-off decode guards.

### C. Fast Generation and Runtime Accounting

Book basis:

- `fast-generation-architectures`
- Source notes for speculative decoding, multi-token prediction, Medusa,
  EAGLE, lookahead decoding, LayerSkip, PagedAttention/vLLM, Mamba, LLaDA, and
  diffusion LLM scaling.
- `schemas/generation_mode_record.schema.json`

Theseus gap:

- Theseus records many `candidate_generation_mode` strings, but not enough
  governed generation-mode records with proposed output, accepted output,
  verifier cost, memory pressure, fallback, and useful-solution-per-second.
- Speed work risks optimizing decode throughput while verifier-passing output
  remains unchanged.

Add to Theseus:

- A generation-mode registry with modes for baseline AR, MLX transformer/hybrid,
  STS-conditioned decode, VCM-assisted decode, verifier-guided repair,
  speculative draft, multi-token/future-token heads, early-exit/self-speculative
  research, KV/prefix-cache reuse, and deterministic tool-assisted completion.
- Accepted-output accounting:
  `proposed_tokens_or_spans`, `accepted_tokens_or_spans`, verifier wall time,
  repair/fallback count, runtime memory, and task pass result.
- A first executable speed-quality comparison: current AR/beam baseline versus
  exactly one verifier-preserving acceleration route on a small private suite.
- KV/prefix-cache and VCM-runtime accounting that separates aggregate serving
  throughput from single-request verified-output latency.
- A Mac MLX profile for these measurements so Apple Silicon speed work targets
  useful verified output, not raw kernel demos.

Acceptance gates:

- A generation-mode report can compute effective verified tokens/spans per
  second and useful solution per second.
- A faster mode is not promotable if verifier pass, integrity, context
  adequacy, or fallback burden regresses.
- Speculative, MTP, Medusa-like, EAGLE-like, diffusion, Mamba, and early-exit
  entries stay experimental until local tests exist.

Do not:

- Claim speedup from proposed tokens.
- Mix raw throughput, accepted output, and task success into one metric.
- Use fast-generation experiments to bypass the learned-generation integrity
  rules.

Implementation status:

- Initial governed generation-mode/runtime accounting is complete as of
  2026-06-25.
- Canonical config: `configs/generation_mode_registry.json`.
- Gate implementation: `scripts/generation_mode_gate.py`.
- Registry integration: the existing `model_governance_gates` surface now owns
  `configs/generation_mode_registry.json`, `reports/generation_mode_registry.json`,
  and `reports/generation_mode_registry.md`, and its verification command
  includes `python3 scripts/generation_mode_gate.py`.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.generation_mode_runtime_accounting_v1`.
- Current mode records cover baseline MLX AR/beam strict-generator decode,
  semantic-head prefix-guided MLX decode, train-once Rust CPU sparse fanout,
  VCM MLX tensor prefix descriptor runtime metadata, and speculative draft
  decode as a planned research mode.
- The first same-suite comparison is intentionally negative:
  `broad8_no_plan_vs_semantic_head_v1` is not promotable because the candidate
  route did not improve accepted-output speed or useful verified
  solutions/second and still had `0` task-pass evidence. This is the desired
  gate behavior: speed experiments cannot promote on proposed throughput or
  prefix mechanics when verified usefulness remains flat.
- Report evidence:
  `reports/generation_mode_registry.json` reports no hard gaps, zero missing
  report refs, five modes, one comparison, no fallback burden, no public
  training rows, no runtime external inference, and
  `promotable_comparison_count=0`.
- Validation commands run:
  `python3 -m py_compile scripts/generation_mode_gate.py`;
  `python3 -m json.tool configs/generation_mode_registry.json`;
  `python3 scripts/generation_mode_gate.py`;
  `python3 scripts/theseus_project_registry.py --gate`.
- Remaining related work: future fast-generation work should start by
  producing task-pass evidence under this accounting contract, then compare
  MLX/CUDA/Metal useful-solution throughput once real verified output exists.

### D. Cognitive Compilation, Semantic Patches, and GenesisCode Discipline

Book basis:

- `cognitive-compilation-and-semantic-ir`
- Source notes for Cognitive Compilation, PlanForge Compiler Architecture,
  GenesisCode, TreeLLM, and Software Magic Grimoire.
- `schemas/semantic_atom.schema.json`
- `schemas/semantic_node_record.schema.json`

Theseus gap:

- Plan/compiler artifacts exist, but code generation and repo work still often
  fail as whole-task failures instead of localized semantic-atom failures.
- Generated code changes are not consistently represented as semantic patches
  with provenance hashes, obligation lists, and effect boundaries.

Add to Theseus:

- A source-plan to semantic-IR compiler for one concrete artifact class:
  small Python script edit, roadmap edit, or code-generator repair.
- Semantic atom records for prompt intent, callable signature, loop operation,
  branch condition, update expression, finalizer, verifier obligation, and
  repair scope.
- Semantic patch records for AI-generated code changes with provenance hashes,
  affected obligations, tests/proofs, side effects, and rollback notes.
- Deterministic effect logs for tool/script calls used by generated patches.
- A localized repair fixture: intentionally break one atom-level requirement
  and prove repair stays inside the atom's repair scope or emits a scope-change
  ledger entry.

Acceptance gates:

- A generator failure can point to a failed semantic atom and failed obligation,
  not only a failed task row.
- A code edit can be reviewed as a semantic patch with validation obligations.
- Repair that changes earlier requirements updates the claim/artifact ledger.

Do not:

- Turn semantic IR into another prompt template.
- Treat syntactic patch success as discharged obligations.
- Grow the trusted computing base without replay/effect logs.

Implementation status:

- Initial semantic atom / semantic patch gate is complete as of 2026-06-25.
- Canonical config: `configs/semantic_patch_registry.json`.
- Gate implementation: `scripts/semantic_patch_gate.py`.
- Registry integration: the `theseus_plan_compiler` surface now owns
  `scripts/semantic_patch_gate.py`, `configs/semantic_patch_registry.json`,
  `reports/semantic_patch_gate.json`, `reports/semantic_patch_gate.md`,
  `reports/semantic_atoms.jsonl`, `reports/semantic_patches.jsonl`, and
  `reports/semantic_effect_logs.jsonl`.
- ATT-D integration: `configs/attd_policy.json` assigns the semantic patch
  gate/config to the `control_plane` role.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.semantic_patch_spine_v1`.
- Current semantic records cover concrete atoms in the policy optimization and
  generation-mode gates: raw-throughput promotion boundary, accepted-output
  metric accounting, promotion branch condition, loss-only policy boundary, and
  the generation-mode work contract.
- The localized repair fixture
  `fixture.raw_throughput_boundary_repair_v1` is GREEN: it simulates a broken
  raw-throughput promotion boundary and verifies the current repair stays
  inside the declared atom repair scope without a scope-change ledger.
- Report evidence:
  `reports/semantic_patch_gate.json` reports `trigger_state=GREEN`, five
  anchored atoms, two semantic patches, three effect logs, one localized repair
  fixture, zero hard gaps, and zero warnings.
- Validation commands run:
  `python3 -m py_compile scripts/semantic_patch_gate.py`;
  `python3 -m json.tool configs/semantic_patch_registry.json`;
  `python3 scripts/semantic_patch_gate.py`;
  `python3 scripts/theseus_project_registry.py --gate`.
- Remaining related work: connect generator candidate failures directly to
  semantic atoms from candidate manifests, so model-training residuals can
  target failed atoms rather than whole failed task rows.

### E. VCM Verification Bandwidth and Context Governor

Book basis:

- `verification-bandwidth-and-context-adequacy`
- Source notes for Verification Bandwidth, Context Engineer, Black Hole Context
  Manager, VCM, and context transactions.

Theseus gap:

- VCM is integrated directionally, but context adequacy still needs harsher
  measurement: contradiction rate, compression loss, summary fidelity,
  stale/tainted context refusal, and verification-workspace limits.
- Context staging/prefetch exists in design, but goal drift, entropy/mass,
  criticality, and deletion closure are not universal runtime mechanics.

Add to Theseus:

- A verification-bandwidth test suite measuring contradiction reduction,
  pairwise semantic-unit checking, summary loss, and context adequacy under
  bounded context.
- Context chunk scoring with relevance, entropy, criticality, goal similarity,
  taint, clearance, and eviction/freeze state.
- Mission-brief generation from VCM that records omitted material, authority
  limits, and adequacy state before a planner/model sees it.
- Digital SCIF fixture tied to VCM transactions: sensitive context enters as
  handles, raw data is zeroized or retained only by policy, and sanitized
  outputs record residual leak risk.
- Deletion-closure and taint-closure tests across summaries, embeddings,
  caches, reports, and training rows.

Acceptance gates:

- An assistant or generator run can fail closed because context is inadequate,
  stale, tainted, overspecified, or missing.
- A summary/mission brief records coverage and compression loss.
- Speculative prefetch cannot influence generation before promotion.
- Deleted/revoked material either closes cleanly or emits a closure fault.

Do not:

- Use larger context as a substitute for verified context.
- Let compressed summaries erase taint, omissions, or uncertainty.
- Let mission briefs become hidden authority escalations.

Implementation status:

- Initial VCM verification-bandwidth/context-governor gate is complete as of
  2026-06-25.
- Canonical config: `configs/vcm_context_governor.json`.
- Gate implementation: `scripts/vcm_context_governor_gate.py`.
- Registry integration: the `vcm_memory` surface now owns
  `configs/vcm_context_governor.json`, `reports/vcm_context_governor.json`,
  `reports/vcm_context_governor.md`, `reports/vcm_mission_brief.json`, and
  `reports/vcm_deletion_closure.json`.
- ATT-D integration: `configs/attd_policy.json` assigns the VCM governor config
  to the `vcm_memory` role.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.vcm_context_governor_v1`.
- Current gate covers context chunk scoring by relevance, entropy,
  criticality, goal similarity, taint, clearance, and eviction state; mission
  brief omission/authority-limit recording; private-handle SCIF behavior; and
  deletion closure over summaries, embeddings, caches, and training-row
  descendants.
- Report evidence:
  `reports/vcm_context_governor.json` reports `trigger_state=GREEN`, four
  chunks, two pinned chunks, mission brief `ready`, four recorded omissions,
  compression loss `0.15`, SCIF `ready`, deletion closure `closed`, zero
  closure faults, zero hard gaps, and zero warnings.
- Context ABI fixture evidence:
  `reports/vcm_context_governor.json` reports `context_abi_fixture_status=ready`,
  `context_abi_fixture_passed_count=5`, and `context_abi_viea_record_count=40`
  across valid leased materialization, mandatory miss typed fault,
  verification-inadequate rejection, mount-policy denial, and expired-lease
  reuse blocking. This proves fixture-level resolver/fault semantics only; it is
  not a deployed resolver, benchmark score, native KV-cache parity claim, or
  learned-generation claim.
- Planner integration evidence:
  `reports/theseus_plan_compiler.json` reports `trigger_state=GREEN`,
  `vcm_context_governor_ready=true`,
  `vcm_context_adequacy_governed_node_count=19`, zero failed gates, and zero
  public training rows, runtime external inference calls, or fallback returns.
- Verifier integration evidence:
  `reports/private_verifier_spine_smoke.json` reports `trigger_state=GREEN`,
  `vcm_context_governor_ready=true`,
  `vcm_context_adequacy_state=governed_sufficient_for_verification`, `12` VIEA
  verifier records including `context_transaction` and `context_adequacy`, and
  zero public training rows, runtime external inference calls, or fallback
  returns.
- Validation commands run:
  `python3 -m py_compile scripts/vcm_context_governor_gate.py`;
  `python3 -m py_compile scripts/theseus_plan_compiler.py`;
  `python3 -m json.tool configs/vcm_context_governor.json`;
  `python3 scripts/vcm_context_governor_gate.py`;
  `python3 scripts/theseus_plan_compiler.py`;
  `python3 scripts/code_lm_private_verifier.py --spine-smoke`;
  `python3 scripts/viea_spine_record_gate.py`;
  `python3 scripts/theseus_project_registry.py --gate`.
- Fanout integration evidence:
  `reports/code_lm_train_once_fanout.json` reports
  `vcm_context_governor_ready=true`,
  `vcm_context_adequacy_state=governed_sufficient_for_generation_fanout`, and
  `10` VIEA fanout records including `context_transaction` and
  `context_adequacy`. `reports/viea_spine_record_gate.json` now has
  `train_once_fanout_spine_v1` passing.
- Direct generator integration evidence:
  `reports/neural_seed_token_decoder_comparator.json` reports
  `direct_generator_vcm_context_ready=true`,
  `direct_generator_vcm_context_adequacy_state=governed_sufficient_for_direct_generation`,
  and `10` VIEA direct-generator records including `context_transaction` and
  `context_adequacy`. `reports/viea_spine_record_gate.json` now has
  `direct_generator_context_spine_v1` passing, with the whole shared spine
  gate `GREEN`.
- Native runtime boundary evidence:
  `reports/vcm_runtime_claim_readiness.json` is `GREEN` with `64` accepted
  semantic materialization descriptors and complete runtime keys;
  `reports/vcm_runtime_cache_lifecycle.json` is `GREEN` with full semantic
  descriptor reuse and invalidation; and `reports/vcm_native_runtime_probe.json`
  is `GREEN` with `11` VIEA runtime records. The current proof supports a
  backend-scoped CPU Transformers DynamicCache prefix/KV lifecycle and an MLX
  resident tensor descriptor lifecycle. It does not claim model-native MLX
  KV/prefix parity: scheduler native KV routing for the recommended MLX backend
  stays fail-closed until that exact backend has a lifecycle proof. The shared
  VIEA spine gate is now `GREEN` at `31/31` profiles.
- Remaining related work: promote Context ABI fixtures into a deployed
  resolver/context compiler conformance gate over real semantic addresses, then
  implement model-native MLX KV/prefix lifecycle and exact-backend
  CUDA/MLX/Metal parity evidence.

### F. Personal Hive Protocol and Policy-First Scheduling

Book basis:

- `personal-compute-hives-and-federated-edge-intelligence`
- `schemas/device_resource_card.schema.json`
- `schemas/portal_card.schema.json`
- `schemas/hive_job_contract.schema.json`
- `schemas/hive_job_bid.schema.json`
- `schemas/hive_scheduling_decision.schema.json`
- `schemas/hive_approval_receipt.schema.json`
- `schemas/hive_federation_lease.schema.json`

Theseus gap:

- Hive exists, but the book's current protocol makes the missing invariant
  explicit: reachability is not authority.
- Current resource/scheduling work should be re-centered on device cards, job
  contracts, bids after policy filtering, approvals, leases, and audit replay.

Add to Theseus:

- Device resource cards for Mac, Windows, Intel Mac, phones, NAS, old machines,
  rented nodes, and future workshop machines.
- Portal cards for phone, browser, Watch, Vision, desktop, and shared-room
  displays.
- Hive job contracts that lower from the same PlanForge/Talos typed-job shape
  used locally.
- Job bids only after data class, tool class, authority, network, physical-risk,
  battery, thermal, operator-presence, and trust policies have rejected
  forbidden nodes.
- Approval receipts for data/authority/physical-risk/spending thresholds.
- Federation leases disabled by default until sandbox and revocation tests
  pass.

Acceptance gates:

- A faster or more available node is rejected when its policy membrane forbids
  the job.
- Phone/operator approval is bound to one job, one permission, and one time
  window.
- Scheduler decisions explain selected and rejected nodes.
- Hive artifact output uses the same artifact/evidence records as local jobs.

Do not:

- Treat LAN reachability, Tailscale-style reachability, or relay reachability as
  authority.
- Send private memory/secrets to rented or public nodes without an explicit
  permitted data class and lease.
- Build public federation before the private scheduler is auditable.

Implementation status:

- Initial Personal Hive policy-first scheduler is complete as of 2026-06-25.
- Canonical config: `configs/hive_policy_first_scheduler.json`.
- Gate implementation: `scripts/hive_policy_first_scheduler_gate.py`.
- Registry integration: the `hive_install_and_apps` surface now owns
  `configs/hive_policy_first_scheduler.json`,
  `reports/hive_policy_first_scheduler.json`,
  `reports/hive_policy_first_scheduler.md`,
  `reports/hive_policy_first_bids.jsonl`, and
  `reports/hive_policy_first_decisions.jsonl`.
- ATT-D integration: `configs/attd_policy.json` assigns the Hive scheduler
  config to the `hive_install` role.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.hive_policy_first_scheduler_v1`.
- Current fixture records four devices: local Apple Silicon Mac, stale Windows
  coordinator, iPhone operator portal, and NAS storage extension. It records
  two portal cards, four job contracts, one approval receipt, and federation
  leases disabled by default.
- Report evidence:
  `reports/hive_policy_first_scheduler.json` reports no hard gaps, 16 bids,
  three scheduled decisions, nine reachable bids rejected by policy, one
  one-job/one-permission/one-window approval receipt, and federation leases
  disabled by default.
- `reports/hive_scheduler.json` now emits first-class
  `hive_scheduler_route_records_v1` VIEA producer evidence for the live Hive
  scheduler's dry-plan path and task-submission receipt schema smoke:
  authority transition/use receipts, runtime adapter invocation, resource
  budget, costed route, generation mode, failure boundary, artifact graph, and
  evidence transition for every placement plus the non-executing receipt
  smoke. This is route and submission-contract traceability, not a
  learned-generation claim or a live multi-node proof.
- `reports/candidate_integrity_audit.json` now emits first-class
  `candidate_integrity_producer_v1` VIEA producer evidence for independent
  candidate-family recomputation, promotion-claim guarding, family-level
  generation-mode accounting, failure boundaries, artifact graph, and
  evidence transition. This is integrity evidence, not semantic pass evidence.
- `reports/private_verifier_spine_smoke.json` now emits first-class
  `private_verifier_spine_v1` VIEA producer evidence from the existing private
  verifier cascade: claim/proof, authority transition/use, runtime adapter,
  resource budget, generation mode, failure boundary, governed
  `context_transaction` and `context_adequacy`, artifact graph, and evidence
  transition. This is verifier-label evidence and context-governance evidence,
  not generation credit.
- The gate is intentionally `YELLOW` because
  `job.remote_cuda_private_training` has no eligible reachable CUDA device on
  this Mac-only fixture. That is not a failure to route around; it is the
  policy-first scheduler refusing to pretend remote CUDA is available when it
  is not.
- Validation commands run:
  `python3 -m py_compile scripts/hive_policy_first_scheduler_gate.py`;
  `python3 -m json.tool configs/hive_policy_first_scheduler.json`;
  `python3 scripts/hive_policy_first_scheduler_gate.py`;
  `python3 scripts/theseus_project_registry.py --gate`.
- Remaining related work: replace fixture cards with live mDNS/Hive discovery
  cards and bind mobile approval receipts to the native iPhone/Watch clients
  after the local control-plane path is stable.

### G. Procedural Memory and Toolification From Real Traces

Book basis:

- `procedural-memory-and-cognitive-loop-closure`
- `schemas/procedural_tool_record.schema.json`

Theseus gap:

- Loop closure exists conceptually, but repeated assistant/repo/generator repair
  traces are not yet automatically converted into verified procedural tool
  candidates with preconditions, postconditions, regressions, monitoring, and
  retirement criteria.

Add to Theseus:

- Trace clustering for repeated repo tasks, generator repair loops, report
  mining, VCM context builds, and private verifier replay.
- Procedural tool candidates only after multiple comparable traces exist.
- Tool cards with source traces, invariant structure, parameters, preconditions,
  postconditions, verification result, risk tier, runtime tier, monitoring,
  residuals, regressions, lifecycle state, and retirement criteria.
- Route eligibility through SCF and registry only after regression gates pass.

Acceptance gates:

- At least one repeated Theseus maintenance workflow becomes a procedural tool
  candidate, not necessarily a default tool.
- A failed regression blocks routable promotion.
- Toolification never counts as learned model capability.

Do not:

- Turn a one-off successful trace into a default tool.
- Hide tool assumptions inside code comments.
- Route future work through a tool whose retirement criteria are absent.

Implementation status:

- Initial procedural memory/toolification gate is complete as of 2026-06-25.
- Canonical config: `configs/procedural_memory_toolification.json`.
- Gate implementation: `scripts/procedural_memory_toolification_gate.py`.
- Registry integration: the `theseus_plan_compiler` surface now owns
  `configs/procedural_memory_toolification.json`,
  `scripts/procedural_memory_toolification_gate.py`,
  `reports/procedural_memory_toolification.json`,
  `reports/procedural_memory_toolification.md`,
  `reports/procedural_tool_candidates.jsonl`, and
  `reports/procedural_tool_route_decisions.jsonl`.
- ATT-D integration: `configs/attd_policy.json` assigns the procedural-memory
  gate/config to the `control_plane` role.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.procedural_memory_toolification_v1`.
- Current gate consumes existing loop-closure reports rather than replacing
  them: `reports/loop_closure_harvester.json`,
  `reports/loop_closure_tool_promoter.json`,
  `reports/viea_verified_procedural_tools.json`, and
  `reports/deterministic_tool_loop_closure_candidates.json`. It now also
  consumes the assistant trace schema, metadata-only dogfood event streams, and
  assistant VIEA trace streams.
- Report evidence:
  `reports/procedural_memory_toolification.json` reports
  `trigger_state=GREEN`, `397` schema-bound assistant trace events, `10`
  assistant-trace procedural candidates, `17` total procedural tool candidates,
  `4` route-eligible legacy candidates, `14` blocked route decisions, `14`
  failed-regression blocks, one benchmark/eval blocked candidate, `1` replay
  fixture, `1` replay-passed fixture, `1` canary-eligible route, `124` VIEA
  procedural tool records, zero raw private text, zero public training rows,
  zero external inference calls, zero fallback returns, zero hard gaps, and zero
  warnings.
- Canary planner integration:
  `replay.local_planning_assistant_metadata_only_v1` verifies
  `procedural.assistant_trace.9b50fc9d7d977f46` with `55` repeated
  planning-assistant traces, `1.0` success/useful rates, low risk, `E1`
  runtime, VIEA binding, no residuals, no raw private text, and clean no-cheat
  counters. The canary route
  `canary.local_planning_assistant_metadata_only_v1` remains non-default and
  is not learned-generation evidence. `scripts/theseus_plan_compiler.py` now
  consumes the procedural-memory report and compiles one canary-only planner
  goal, so `reports/theseus_plan_compiler.json` is `GREEN` with `8` compiled
  goals, `22` nodes, `22` trace rows, `1` procedural-memory canary goal, zero
  hard gate failures, zero public training rows, zero external inference calls,
  and zero fallback returns.
- Canary execution:
  `scripts/procedural_memory_canary_executor.py` runs the compiled canary in
  bounded local metadata-replay mode. `reports/procedural_memory_canary_execution.json`
  is `GREEN` with `1` eligible canary route, `1` executed route, `55` matched
  schema-bound planning-assistant events, `1` emitted route packet, `0`
  default route adoptions, `0` learned-generation claims, `10` VIEA
  canary-execution records, actual duplicate-work delta `-54`, metadata
  verification-cost delta `-206`, zero public training rows, zero external
  inference calls, and zero fallback returns. This is canary execution
  evidence, not default route adoption.
- Registry adoption:
  `scripts/procedural_memory_route_adoption_gate.py` consumes the procedural
  memory gate, canary execution report, registry route-validator view, and
  steward contract. `reports/procedural_memory_route_adoption.json` is `GREEN`
  with `1` transaction, `1` guarded default route adopted, `1` regression guard
  armed, `12` VIEA adoption records, `0` learned-generation claims, zero public
  training rows, zero external inference calls, and zero fallback returns. The
  adopted default route is local metadata workflow compression only. It is not
  public-transfer evidence and cannot support learned-generation promotion.
- Planner default-route integration:
  `scripts/theseus_plan_compiler.py` now consumes
  `reports/procedural_memory_route_adoption.json` and compiles one default-route
  maintenance goal alongside the replay/canary goal. `reports/theseus_plan_compiler.json`
  is `GREEN` with `9` compiled goals, `25` nodes, `25` trace rows, `1`
  procedural-memory canary goal, `1` procedural-memory default-route goal, zero
  hard gate failures, zero public training rows, zero external inference calls,
  and zero fallback returns.
- Assistant default-route integration:
  `configs/theseus_assistant_runtime.json` now makes planning intent consult
  `reports/procedural_memory_route_adoption.json` as a required guarded
  metadata route. `scripts/theseus_assistant_runtime.py` reads the route,
  hard-gates on `GREEN` adoption, armed regression guard, no learned-generation
  claim, zero public/external/fallback counters, and an explicit
  `route_binding_contract` match. That contract binds selection to
  `surface=local_assistant`, `intent=planning`,
  `assistant_lane=planning_assistant`, `vcm_task_family=autonomy_governance`,
  and authorized runtime consumers
  `theseus_plan_compiler`/`theseus_assistant_runtime`; the hard
  `procedural_default_route_binding_contract_enforced` gate must pass before
  the route attaches. The runtime then emits a `procedural_tool_record` with
  route scope, binding contract, and selection evidence in the assistant VIEA
  trace. The canonical roadmap and trace-schema assistant smokes are `GREEN`
  with `procedural_default_route_ready=true`,
  `procedural_default_route_selection_matched=true`, route
  `default.local_planning_assistant_metadata_only_v1`, `19` assistant VIEA trace
  records, and zero public training rows, runtime external inference, or
  fallback returns. After those traces and later route/context/rights/fidelity
  producers were ingested, the VIEA spine gate is `GREEN` with `2227`
  materialized records and `184` claim/proof entries after the learned
  router-head spine profile and bounded planning-world adapter were added.
- Validation commands run:
  `python3 -m py_compile scripts/procedural_memory_toolification_gate.py`;
  `python3 -m py_compile scripts/procedural_memory_canary_executor.py`;
  `python3 -m py_compile scripts/procedural_memory_route_adoption_gate.py`;
  `python3 -m json.tool configs/procedural_memory_toolification.json`;
  `python3 scripts/procedural_memory_toolification_gate.py`;
  `python3 scripts/procedural_memory_canary_executor.py`;
  `python3 scripts/procedural_memory_route_adoption_gate.py`;
  `python3 -m py_compile scripts/theseus_plan_compiler.py`;
  `python3 scripts/theseus_plan_compiler.py`;
  `python3 -m py_compile scripts/theseus_assistant_runtime.py`;
  `python3 scripts/theseus_assistant_runtime.py --intent planning ...`;
  `python3 scripts/viea_spine_record_gate.py --gate`;
  `python3 scripts/theseus_project_registry.py --gate`.
- Remaining related work: generalize this adoption/rollback guard across future
  procedural candidates and eligible assistant intents beyond this first
  binding-scoped planning metadata route, using the same route-binding
  contract shape rather than adding a parallel shortcut.

### H. Circle and Proof-Carrying Contract Transfer

Book basis:

- `circle-calculus-and-proof-carrying-ai-contracts`
- `coil-attention-cyclic-memory-and-recurrence-contracts`
- `coilra-multicoil-rope-and-cyclic-mixers`
- `executable-specifications-and-lean-proof-envelope`
- Source notes for Circle contract suite, Coil attention/memory, CoilRA,
  RoPE position certifier, and proof-carrying circular computation.

Theseus gap:

- Circle-derived fixtures and cyclic ideas are referenced, but Theseus needs a
  stricter boundary between proof-carrying structural facts and model-quality
  claims.

Add to Theseus:

- A proof-carrying AI contract adapter that can import Circle theorem IDs,
  compiled Lean status, fingerprints, deterministic fields, validation
  commands, and explicit non-claims.
- Circle contract usage only as fixture/configuration/proof boundary unless a
  Theseus model/task result separately proves utility.
- Substrate adoption records for Coil/cyclic/RoPE/MultiCoil experiments with
  baselines, negative controls, falsification criteria, and residuals.

Acceptance gates:

- A Circle-derived fixture can support only the claim its theorem proves.
- Any model-quality, speed, memory, or transfer claim needs separate Theseus
  task evidence.
- Proof placeholders or missing Lean build status block proof-carrying claims.

Do not:

- Treat finite structural proof as evidence of general intelligence.
- Let cyclic substrate experiments bypass transformer/hybrid baselines.

Implementation status:

- Initial Circle/proof-carrying contract transfer gate is complete as of
  2026-06-25.
- Canonical config: `configs/proof_carrying_contracts.json`.
- Gate implementation: `scripts/proof_carrying_contract_gate.py`.
- Registry integration: the `deterministic_tool_substrate` surface now owns
  Circle bridge scripts, the proof-carrying config, and the reports
  `reports/proof_carrying_contract_gate.json`,
  `reports/proof_carrying_contract_gate.md`,
  `reports/proof_carrying_contract_records.jsonl`, and
  `reports/substrate_adoption_records.jsonl`.
- ATT-D integration: `configs/attd_policy.json` assigns the proof-carrying
  gate and config to control-plane/governance coverage.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.proof_carrying_contract_transfer_v1`.
- Current gate checks the newer Circle AI contract pack, theorem manifest,
  legacy Theseus Circle bridge reports, local Lean contract artifact freshness,
  claim requests, and substrate adoption records.
- Report evidence:
  `reports/proof_carrying_contract_gate.json` reports `trigger_state=GREEN`,
  nine schema-shaped `proof_contract_receipt_record` rows, nine
  fixture-ready/proof-ready contracts, a current local Lean contract artifact,
  two allowed fixture/proof-boundary claims, one blocked overbroad
  context-length/model-quality claim, three schema-shaped
  `substrate_adoption_record` rows, no production defaults, and zero hard gaps
  or warnings.
- VIEA materialization: `configs/viea_spine_record_contracts.json` includes
  `proof_contract_and_substrate_adoption_spine_v1`, and
  `reports/viea_spine_record_gate.json` is `GREEN` for that profile. These
  records now flow into `reports/viea_spine_materialized_view.json` as
  canonical `proof_contract_receipt` and `substrate_adoption` families.
- Shared record shape: the same materialized VIEA view now normalizes
  `artifact_graph_record`, `context_transaction`, `compressed_artifact_record`,
  `compression_receipt`, and `compact_generative_record` payloads to the current
  AI_book schema-required fields. The VIEA gate reports
  `summary.schema_payload_gap_count=0`, so skeletal record IDs cannot stand in
  for replay/evidence/non-claim payloads.
- Current non-claims: this lane proves structural/proof-boundary accounting,
  not model quality, runtime speed, context length, memory efficiency, broad
  public transfer, learned generation, or ASI capability.
- Remaining related work: attach one Circle-derived fixture to a real private
  Theseus task beside transformer/hybrid and no-fixture controls before making
  any utility claim.

### I. Book-Quality Definition of Done for Theseus Modules

Book basis:

- `docs/book_quality_standard.md`
- `docs/evidence_test_plan.md`
- `docs/proof_and_code_plan.md`
- `prototype-roadmap`

Theseus gap:

- Book chapters have a consistent DoD shape: problem, mechanism, interfaces,
  invariants, failure modes, minimal implementation, test plan, source
  crosswalk. Theseus modules/scripts do not consistently carry an equivalent
  module card or registry entry.

Add to Theseus:

- A registry module DoD for each major surface:
  problem, owner field, interface, invariants, failure modes, minimal
  implementation, validation commands, evidence refs, non-claims, deprecation
  route, and source crosswalk.
- An ATT-D rule that flags new major modules without a module card or registry
  mapping.
- A report-family cap and compaction policy tied to module cards so evidence
  does not become endless duplicate report families.

Acceptance gates:

- New major scripts/configs/docs cannot quietly appear without registry/module
  ownership.
- A stale module can be merged, deprecated, or retired through a visible
  steward decision.
- Negative and inconclusive results remain linked to the module claim they
  contradict.

Do not:

- Require ceremonial paperwork for tiny local edits.
- Let module cards become evidence of capability.
- Add new registries when the existing project registry can own the surface.

Implementation status:

- Initial book-quality module definition-of-done gate is complete as of
  2026-06-25.
- Canonical config: `configs/module_definition_of_done.json`.
- Gate implementation: `scripts/module_definition_of_done_gate.py`.
- Registry integration: the `project_manifest_registry` surface now owns the
  DoD config, gate, and reports
  `reports/module_definition_of_done.json`,
  `reports/module_definition_of_done.md`, and
  `reports/module_definition_cards.jsonl`.
- ATT-D integration: `configs/attd_policy.json` assigns the DoD gate and config
  to the project-registry role.
- Steward integration: `configs/project_steward.json` now declares
  `roadmap.module_definition_of_done_v1`; weak module evidence refs for Hive
  and Rust were replaced with concrete existing artifacts.
- Current gate checks 22 major surfaces against required module-card fields,
  book/charter source crosswalks, report-family caps, stale latest-view policy,
  and steward linkage for negative evidence.
- Report evidence:
  `reports/module_definition_of_done.json` reports `trigger_state=GREEN`, 22
  major surfaces, 22 ready module records, major-surface coverage `1.0`, seven
  present book/charter source refs, negative evidence linked, and zero hard gaps
  or warnings.
- Remaining related work: use this DoD gate as a preflight for new major
  modules so cleanup and capability work improves registered systems instead of
  creating successor sprawl.

### J. Revised Near-Term Ordering

The earlier roadmap phases remain valid, but this pass changes the practical
ordering for the next several goals:

1. Stabilize current truth and steward coverage:
   project steward record, work-contract coverage for the next task, event-taint
   intake, module DoD, and stale report-family steward decisions.
2. Make the generator wall a policy-optimization problem:
   private preference/RLVR baseline over verifier-labeled candidates, with
   reward-hacking probes and rollback records.
3. Add accepted-output generation-mode accounting:
   measure useful verified output per second before chasing faster decode
   machinery.
4. Add semantic-IR repair locality:
   semantic atoms and semantic patches for the current loop-generation wall.
5. Harden VCM adequacy:
   contradiction/summary-loss/context-taint tests and Digital SCIF fixture.
6. Only then return to Hive distribution:
   device cards, portal cards, job contracts, bids, approvals, and federation
   leases under the same authority/evidence spine.

This ordering is meant to reduce circular work. It directs the next serious
implementation pass toward the places where Theseus currently loses leverage:
unowned project lifecycle, loss-improves-but-behavior-does-not training,
speed metrics that do not count accepted useful work, whole-task failures that
should become semantic-atom failures, and distributed reachability without a
fully explicit authority membrane.

Implementation status:

- Revised-order slices A through I are implemented and validated as of
  2026-06-25.
- Current registry evidence:
  `python3 scripts/theseus_project_registry.py --gate` reports
  `trigger_state=GREEN`, zero registry governance violations, zero hard
  governance violations, project steward `GREEN`, nine active work contracts,
  22 module cards, and major-surface module-card coverage `1.0`.
- Current ATT-D evidence:
  `python3 scripts/attd_analyzer.py` reports `trigger_state=YELLOW`,
  `attd_score=0.561355`, hard caps passed, and zero hard-cap violations.
- Remaining roadmap wall: this closes the revised-order governance/spine
  scaffolding, not the full 20-phase capability program. The next material
  work should use the new gates to improve the practical generator, assistant,
  VCM runtime path, Mac acceleration path, and public-transfer calibration
  evidence without adding new side lanes.
- Highest current technical debt pressure from ATT-D: project registry shape,
  governance-gate breadth, training-runtime source shape, Mac acceleration
  source shape, and benchmark adapter breadth. These are maintenance targets,
  not capability claims.
- Capability-harness correction:
  `scripts/candidate_integrity.py` now auto-resolves the missing/empty legacy
  `reports/student_code_candidates.jsonl` default to the newest non-empty
  registered private/strict candidate source. The current canonical audit uses
  `reports/neural_seed_token_decoder_candidates_strict_body_tokens.jsonl` and
  reports 456 audited candidates, 360 learned full-body token candidates, 176
  independently integrity-verified learned candidates, zero integrity
  mismatches, and 144 syntax-invalid learned rows.
- Current maturity evidence after the candidate-integrity latest-view repair:
  `reports/maturity_integrity_audit.json` remains `YELLOW`; current broad
  public pass-rate view is `0.14375`, promotion/growth remain blocked, and
  public calibration remains not allowed by the current maturity gate.
- Current generation-mode wall:
  `reports/generation_mode_registry.json` remains `YELLOW`; accepted-span
  throughput is nonzero, but useful verified solution/sec remains `0.0`.
  The next generator work should reduce syntax-invalid learned rows and produce
  task-pass evidence, not just faster accepted spans.
- Strict body-token decode repair pass, 2026-06-25:
  `scripts/neural_seed_token_decoder_support.py` now has task-blind token-policy
  guards for incomplete `for ... in` iterables, bare constructor loop iterables,
  adjacent expression atoms without separators, comparison/membership assignment
  targets, overlong condition connector chains, and runaway condition subscripts.
  These guards reject impossible next tokens only; they do not render bodies,
  inspect tests/solutions, use public data, or add fallback returns.
- Fresh smoke evidence:
  `reports/strict_generator_mlx_decode_eval_token_policy_repair_smoke_v1.json`
  and `reports/strict_generator_mlx_decode_eval_token_policy_repair_smoke_v2.json`
  both ran against four broad-private heldout rows with
  `external_inference_calls=0`, `public_training_rows=0`, no fallback returns,
  clean strict source-text audit, and inline candidate integrity with
  three independently verified transformer/hybrid generated candidates and
  zero syntax-invalid generated rows. Explicit candidate integrity reports are
  `reports/candidate_integrity_token_policy_repair_smoke_v1.json` and
  `reports/candidate_integrity_token_policy_repair_smoke_v2.json`, both
  `GREEN` with zero integrity mismatches and zero syntax-invalid rows.
- Current learned-generation wall after the repair:
  syntax/loadability is no longer the immediate smoke blocker, but functional
  pass remains `0/4`. The fresh generated bodies are shallow and the decode
  starvation summary still reports three zero-candidate tasks. v1 starvation was
  dominated by `missing_local_return`; v2 moved some pressure into
  `current_line_starts_return`, `inside_loop`, and `inside_loop_without_update`.
  The next non-cheating generator work should target learned statement
  completion, return-expression closure, and semantic grounding from visible
  prompt/signature features rather than adding another template, renderer, or
  fallback path.
- Additional strict-token hygiene pass, 2026-06-25:
  expression-level `for` tokens are now blocked in direct body-token decoding
  so the model cannot enter broken comprehension-like tails inside `return`
  expressions, comparison operators are treated as unfinished expression tails
  by both the token policy and MLX closure helper, and same-line subscript
  chains are capped to prevent the decoder from exhausting the target budget on
  `...[1][1][1]...` continuations. These are prefix-only grammar/closure
  constraints; they do not render answers or add fallback code.
- Evidence from the follow-up smokes:
  `reports/strict_generator_mlx_decode_eval_expression_for_block_smoke_v1.json`,
  `reports/strict_generator_mlx_decode_eval_source_condition_expression_for_block_smoke_v1.json`,
  and `reports/strict_generator_mlx_decode_eval_subscript_cap_smoke_v1.json`
  all stayed no-cheat clean: no external inference, no public training rows, no
  fallback returns, clean source-text audit, three verified transformer/hybrid
  generated candidates, and zero syntax-invalid generated rows. Candidate
  integrity reports
  `reports/candidate_integrity_expression_for_block_smoke_v1.json`,
  `reports/candidate_integrity_source_condition_expression_for_block_smoke_v1.json`,
  and `reports/candidate_integrity_subscript_cap_smoke_v1.json` are `GREEN`.
  The negative result is also explicit: all three smokes still report
  functional pass `0/4`, candidate rows `3`, and three zero-candidate tasks.
  Source-condition priority did not improve this 4-row slice. The remaining
  wall is therefore learned semantic planning/candidate coverage, not syntax
  loadability. The next pass should change the training signal or learned
  prefix/return-state objective, then verify on the same smoke before spending
  public calibration.
- Correctness-in-the-loop replay follow-up, 2026-06-25:
  `scripts/strict_generator_mlx_private_adaptation.py` now maps current decode
  starvation labels into action-trace-aware pairwise replay pressure for
  missing local returns, unfinished return expressions, and loops that fail to
  update or exit. The mapping is private-train replay only and feeds existing
  unlikelihood/pairwise objectives; it does not render code, use public data,
  inspect tests/solutions, call a teacher, or grant candidate-generation
  credit.
- Two replay-weighting outcomes are now recorded:
  `reports/strict_generator_mlx_private_adaptation_starvation_pairwise_return_closure_v1.json`
  is `RED` because heldout LM worsened from `0.352950` to `0.437255`. Its
  decode smoke,
  `reports/strict_generator_mlx_decode_eval_starvation_pairwise_return_closure_broad4_v1.json`,
  stayed no-cheat clean but emitted only one verified transformer/hybrid
  candidate, remained `0/4`, and left three zero-candidate tasks. The explicit
  candidate integrity report
  `reports/candidate_integrity_starvation_pairwise_return_closure_broad4_v1.json`
  is `GREEN` with zero mismatches.
- The conservative replay variant,
  `reports/strict_generator_mlx_private_adaptation_starvation_pairwise_return_closure_conservative_v1.json`,
  is `GREEN` and improves heldout LM from `0.352950` to `0.236962` with zero
  public training rows, zero external inference calls, and zero fallback/router
  credit. Its decode smoke,
  `reports/strict_generator_mlx_decode_eval_starvation_pairwise_return_closure_conservative_broad4_v1.json`,
  is still only diagnostic: one verified transformer/hybrid candidate,
  functional pass `0/4`, nontrivial-return rate `1.0`, and three zero-candidate
  tasks. `reports/candidate_integrity_starvation_pairwise_return_closure_conservative_broad4_v1.json`
  is `GREEN`.
- Current strict-generator conclusion:
  conservative correctness replay can improve LM loss without cheating, but it
  still does not improve broad heldout behavior. The next material generator
  target is not more generic loss weighting; it is a learned plan/state
  objective that raises admissible candidate coverage and semantic correctness
  under prompt/signature-only source text. Specifically, target the repeated
  starvation states `missing_local_return`, `inside_loop_without_update`, and
  wrong `SAFE_HEAD_DEFAULT`/text-build plan choices with a measured ablation
  against the same broad-private smoke.
- Learned plan-head ablation, 2026-06-25:
  `reports/strict_generator_mlx_private_adaptation_plan_aux_starvation_conservative_clean_v1.json`
  is `GREEN`: heldout LM improved from `0.352950` to `0.236518`, heldout
  semantic-plan loss improved from `1.545287` to `1.164753`, and heldout
  semantic-plan accuracy improved from `0.5625` to `0.6875`. The incompatible
  semantic-slot-prefix boost from the prior `YELLOW` ablation was removed
  because the current checkpoint uses `plan_prefix_body_tokens_v1`, not the
  semantic-slots target mode.
- Decode with the learned semantic-plan head enabled:
  `reports/strict_generator_mlx_decode_eval_plan_aux_starvation_conservative_clean_head_broad4_v1.json`
  stayed no-cheat clean and improved verified transformer/hybrid candidate
  coverage from one candidate to two candidates on the same four-row broad
  private smoke. It still reports functional pass `0/4`, nontrivial-return rate
  `1.0`, three zero-candidate tasks, and wrong-answer verifier labels for both
  generated candidates. `reports/candidate_integrity_plan_aux_starvation_conservative_clean_head_broad4_v1.json`
  is `GREEN`, with two independently verified transformer/hybrid candidates,
  zero syntax-invalid rows, and zero integrity mismatches.
- Paired generation-mode accounting:
  `reports/strict_generator_mlx_decode_eval_plan_aux_starvation_conservative_clean_no_head_broad4_v1.json`
  is the same-checkpoint no-head control. It is `RED` with zero emitted learned
  candidates; its candidate-integrity report
  `reports/candidate_integrity_plan_aux_starvation_conservative_clean_no_head_broad4_v1.json`
  is `YELLOW` only because there are no verified learned rows. The generation
  mode registry now has an explicit paired comparison,
  `broad4_plan_aux_no_head_vs_semantic_head_v1`. The gate reports accepted
  speed/candidate-coverage lift for the semantic head, no hard gaps, and still
  no promotable comparison because `task_pass_count=0`.
- Updated generator wall:
  the learned plan head is a real coverage lever, but the generated bodies are
  still semantically wrong. The next strict-generator work should train body
  token semantics conditioned on the learned plan decision, especially value
  expression choice, accumulator/update choice, and final return expression,
  rather than only improving plan classification or global LM loss.
- Plan-conditioned body semantic weighting follow-through, 2026-06-25:
  `scripts/strict_generator_mlx_adaptation_weights.py` now includes
  `private_plan_conditioned_body_semantic_weighting_v1`, which boosts admitted
  private target spans selected from the existing semantic-plan taxonomy:
  guard expressions, loop conditions/sources/updates, plan-key calls, and final
  return expressions. The hook is wired through
  `scripts/strict_generator_mlx_private_adaptation.py` behind explicit
  `--plan-conditioned-body-*` flags. It does not render code, inspect eval
  tests/solutions, use public data, call a teacher, or grant candidate
  generation credit.
- Evidence from the conservative plan-conditioned run:
  `reports/strict_generator_mlx_private_adaptation_plan_aux_body_semantics_conservative_v1.json`
  is `GREEN`: heldout LM loss improved from `0.352950` to `0.247737`,
  heldout semantic-plan loss improved from `1.545287` to `1.154249`, heldout
  plan accuracy improved from `0.5625` to `0.6875`, and the new weighting
  matched `128` private rows with `3317` weighted token positions. The decode
  report
  `reports/strict_generator_mlx_decode_eval_plan_aux_body_semantics_conservative_head_broad4_v1.json`
  is still `YELLOW`: two verified transformer/hybrid candidates, zero syntax
  invalid rows, zero integrity mismatches, no public training, no external
  inference, no fallback returns, and functional pass still `0/4`.
  `reports/candidate_integrity_plan_aux_body_semantics_conservative_head_broad4_v1.json`
  is `GREEN` and independently classifies the six emitted rows as four
  fallback/template inventory rows plus two verified transformer/hybrid rows.
- Current strict-generator conclusion after plan-conditioned weighting:
  span weighting can improve private LM/plan-head metrics without cheating, but
  it did not move behavioral transfer on the broad-private smoke. The learned
  rows remain shallow expression guesses rather than full task algorithms. The
  next generator work should make the source-to-plan-to-body path produce
  executable dataflow: plan-conditioned statement sequencing, variable binding,
  loop/update semantics, and final return construction. Do not promote another
  loss-only or coverage-only route until verified task passes move above zero
  under independent candidate integrity.
- Richer learned-prefix/dataflow target-mode ablation, 2026-06-25:
  `reports/strict_generator_mlx_pretraining_probe_plan_semantic_slots_dataflow_smoke_v1.json`
  is `GREEN` using `plan_semantic_slots_body_tokens_v1`,
  `--semantic-plan-loss-weight 0.2`, and primary dataflow weighting. It trained
  on MLX with no public rows, no external inference, no fallback/template/tool
  credit, and improved heldout LM loss from `8.136910` to `5.256321` while
  consuming about `25,245` training token positions/sec. This is training-loss
  evidence only.
- Decode hygiene exposed the real richer-prefix wall:
  `scripts/strict_generator_mlx_decode_guards.py` now uses inferred local
  receiver types when blocking invalid method receivers, rejects operators
  immediately after `.`, and blocks dunder attribute filler. The guided loop
  plan path in `scripts/strict_generator_mlx_decode_plans.py` now runs the same
  strict/value guards before returning forced update tokens. The close-after
  top-level-return helper in `scripts/strict_generator_mlx_decode_eval.py`
  can close only prefixes that already pass the final static guard. These are
  task-blind grammar/dataflow hygiene checks; they do not synthesize code or
  grant learned-generation credit.
- Evidence after the hygiene chain:
  `reports/strict_generator_mlx_decode_eval_plan_semantic_slots_dataflow_head_broad4_v5.json`
  is `RED` with `0` learned candidates and `0/4` functional pass. Its
  starvation summary reports all four tasks stopped with
  `inside_loop_without_update` and `missing_local_return`. The best beams now
  stall at `result.` after a generated prefix such as
  `INIT_NUMBER + UPDATE_CALL`, which is an incompatible update contract.
  `reports/candidate_integrity_plan_semantic_slots_dataflow_head_broad4_v5.json`
  is `YELLOW`: four fallback/baseline rows, zero verified learned candidates,
  zero integrity mismatches.
- Updated strict-generator wall:
  the next material repair is not more decode hygiene. It is learned
  update-contract consistency: the prefix model must stop pairing numeric
  accumulators with method-call updates, and the body model must learn a legal
  update expression (`+=`, transformed assignment, list initializer before
  append, etc.) from visible prompt/signature context and private target
  supervision. The right next experiment is a private update-contract
  consistency objective/auxiliary head plus an ablation on this same smoke,
  not another public calibration or fallback renderer.
- Update-contract consistency follow-through, 2026-06-25:
  `scripts/strict_generator_mlx_decode_plans.py` no longer maps numeric
  accumulator contracts to method mutation syntax, and its task-blind update
  detector now recognizes `AugAssign` and transformed assignment updates inside
  loops. `scripts/strict_generator_mlx_decode_eval.py` now requires generated
  learned-prefix loop plans to choose a `SLOT:STATE_UPDATE_*` state before body
  start when the vocabulary supports it, with contract-only filtering that
  blocks incoherent numeric mutation-call states. This carries
  `candidate_generation_credit=0` and uses only generated prefix tokens,
  visible signature names, and task-blind grammar state.
- Private update-contract objective evidence:
  `scripts/strict_generator_mlx_adaptation_weights.py` now includes
  `private_update_contract_consistency_weighting_v1`, wired through
  `scripts/strict_generator_mlx_private_adaptation.py` behind
  `--update-contract-consistency-loss-boost`. The GREEN adaptation report
  `reports/strict_generator_mlx_private_adaptation_update_contract_consistency_private_smoke_v1.json`
  matched all `128` private train rows and weighted `1611` token positions,
  including `1098` learned prefix contract slots and `513` body update-token
  positions. It used `0` public rows, `0` external inference calls, and grants
  no fallback/template/router/tool credit as learned generation.
- Behavior after update-contract repair:
  `reports/strict_generator_mlx_decode_eval_update_contract_consistency_broad4_v1.json`
  is `YELLOW`: it emits `4` generated transformer/hybrid candidates on the
  same broad-private four-row smoke, all `4` compile/runtime-load, all `4`
  pass independent inline candidate integrity, and the action-trace summary no
  longer reports loop-update mismatch. The independent audit
  `reports/candidate_integrity_update_contract_consistency_broad4_v1.json` is
  `GREEN` with zero integrity mismatches. Functional pass remains `0/4`; the
  generated bodies collapse to simple sum-like loops and fail with
  type-handling/wrong-answer residuals. This is a real execution-contract
  repair, not a promotion.
- Current strict-generator wall after update-contract repair:
  the wall has moved from syntax/loadability/update-contract starvation to
  semantic algorithm choice. The next Phase 4 target should train and evaluate
  source-to-plan semantic selection and branch/operation choice under the same
  no-cheat boundary: choose clamp/round/window/type-handling/contract-specific
  operations from visible prompt/signature context instead of defaulting to
  `result = result + value`. Do not count deterministic plan guidance,
  fallback renderers, or verifier labels as learned generation; use them only
  as private supervision/evidence for a later learned selector/body model.
- Semantic-plan label-space repair, 2026-06-25:
  `scripts/strict_generator_mlx_pretraining_probe.py` now supports
  `semantic_plan_token_subspace_v1`, so semantic-plan auxiliary loss/eval are
  computed only over `SLOT:PLAN_*` target-vocab classes instead of the entire
  body-token vocabulary. `scripts/strict_generator_mlx_private_adaptation.py`
  now passes that plan-token subspace through private adaptation and gates it
  with `semantic_plan_label_space_active_when_enabled`. This is learned
  plan-head supervision only: no public rows, no teacher calls, no tool calls,
  no renderer, no fallback credit, and no learned-generation promotion by
  itself.
- Evidence for plan-subspace repair:
  `reports/strict_generator_mlx_private_adaptation_semantic_plan_subspace_private_smoke_v1.json`
  is `GREEN`: heldout LM loss improved from `5.539204` to `4.138901`,
  heldout semantic-plan loss improved from `6.731428` to `3.111023`, and
  heldout semantic-plan accuracy moved from `0.0` to `0.054688` across `71`
  plan classes. The decode report
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_broad4_v1.json`
  is still `YELLOW`: it emits `10` verified transformer/hybrid candidates with
  zero integrity mismatches and zero fallback/template/router/tool credit, but
  functional pass remains `0/4`. The important improvement is that plan-prefix
  collapse moved from `AST_BRANCH_RETURN_NONE` to a diverse learned prefix set
  (`RLE_ENCODE`, `INTERVAL_COVERAGE`, `WINDOWED_DELTAS`). The important wall is
  that body continuations remain shallow sum/list loops.
- Guarded decode ablation:
  `reports/strict_generator_mlx_decode_eval_semantic_plan_subspace_block_identity_broad4_v1.json`
  enabled the existing task-blind shallow-identity blocker. It reduced shallow
  identity accumulation but still produced `0/4` functional pass, shifting
  residuals toward `append_whole_source_argument`,
  `loop_without_decision_or_state_update`, `missing_semantic_update_value`, and
  `missing_rle_branch_or_update`. This confirms the wall is learned body
  continuation quality, not merely decode acceptance policy.
- Body-action weighting follow-up:
  `reports/strict_generator_mlx_private_adaptation_semantic_body_action_private_smoke_v1.json`
  is `GREEN` and improved heldout semantic-plan accuracy from `0.054688` to
  `0.65625` after adding guard, loop-source, loop-condition, update, finalizer,
  and plan-key-call span weighting. However
  `reports/strict_generator_mlx_decode_eval_semantic_body_action_broad4_v1.json`
  is `YELLOW` and regressed broad4 behavior: only `4` verified
  transformer/hybrid candidates, `0/4` functional pass, two zero-candidate
  tasks, and repeated shallow `AST_TEXT_SPLIT_TRANSFORM` continuations. The
  independent audit
  `reports/candidate_integrity_semantic_body_action_broad4_v1.json` remains
  `GREEN`, so this is honest negative capability evidence, not a contamination
  or integrity failure.
- Updated strict-generator wall after plan-subspace/body-action experiments:
  the plan-head label-space bug is fixed, and source-to-plan learning can now
  move under the no-cheat contract. The next material repair is the
  plan-conditioned body continuation path: train against failed private replay
  candidates and admitted private targets so generated bodies learn complete
  branch/update/finalizer statement sequences without private body inventory,
  deterministic renderers, or fallback returns. Do not promote the body-action
  checkpoint; use the plan-subspace checkpoint as the cleaner base unless a
  later replay objective proves better verifier behavior.
- Source-conditioned semantic slot head, 2026-06-25:
  `scripts/strict_generator_mlx_pretraining_probe.py` now has a role-aware
  source-pooled `semantic_slot_logits` auxiliary head over existing learned
  prefix roles: return shape, loop source, init, update, state update,
  bindings, finalizer, statement sequence, and simple guard/default-return
  slots. `scripts/strict_generator_mlx_decode_eval.py` can consume it behind
  `--use-semantic-slot-head-prefix`; it only chooses learned prefix tokens and
  does not render code, call tools, inspect tests/solutions/public payloads, or
  grant learned-generation promotion credit. Old checkpoints remain loadable
  with non-strict missing `slot_router` weights.
- Semantic slot-head evidence:
  `reports/strict_generator_mlx_pretraining_semantic_slot_smoke_v1.json` is
  `GREEN`: heldout LM loss improved from `8.383856` to `5.234303`, plan
  accuracy moved from `0.0` to `0.125`, and slot accuracy moved from `0.0` to
  `0.530233`. On the same checkpoint, plan-only broad4 decode
  (`reports/strict_generator_mlx_decode_eval_semantic_slot_checkpoint_plan_only_broad4_v1.json`)
  emitted zero candidates, while plan+slot broad4 decode
  (`reports/strict_generator_mlx_decode_eval_semantic_slot_head_broad4_v1.json`)
  emitted four integrity-clean runtime-loaded candidates. Functional pass
  remained `0/4`.
- Larger slot-head smoke:
  `reports/strict_generator_mlx_pretraining_semantic_slot_full_smoke_v1.json`
  trained the same private/licensed smoke corpus for `600000` token positions
  on MLX GPU with source-contrastive, plan, slot, and primary-dataflow losses.
  It is `GREEN`: source contrastive gap improved from `0.028191` to
  `0.115867`, plan accuracy moved to `0.1875`, and slot accuracy moved to
  `0.572093`. The decode report
  `reports/strict_generator_mlx_decode_eval_semantic_slot_full_smoke_broad4_v1.json`
  is still `YELLOW`: seven generated transformer/hybrid candidates, all
  syntax-valid, integrity-clean, and runtime-loaded, but zero verifier-passed
  tasks. The bodies still collapse to generic accumulator patterns such as
  `result = result + value`, choosing visible parameters but not prompt-specific
  transforms.
- Current strict-generator wall after source-slot heads:
  prefix selection and loadability are no longer the main blocker on this
  smoke. The next material repair is learned prompt-conditioned expression and
  body semantics: choose clamp/round/window/tolerance/contract-specific
  operations and correct return shapes from visible prompt/signature context.
  Do not add deterministic renderers or action routers and count them as
  learned generation. The next honest implementation should add a trainable
  expression/value head or richer body-continuation objective that predicts
  operation operands and final return expressions, then decode raw body tokens
  and prove verifier pass movement.
- Scalar body-weighting limit:
  `reports/strict_generator_mlx_private_adaptation_semantic_slot_body_expression_repair_private_v1.json`
  applied the existing private body-expression, loop semantic operation,
  update-contract, semantic-slot-prefix, plan-conditioned body, return
  expression, primary dataflow, and source-contrastive losses to the full
  slot-head checkpoint. It lowered heldout LM loss from `4.557769` to
  `2.148576`, but
  `reports/strict_generator_mlx_decode_eval_semantic_slot_body_expression_repair_broad4_v1.json`
  still scored `0/4`. It emitted four integrity-clean runtime-loaded
  candidates, all wrong-answer residuals, and showed malformed nested
  branch/update continuations. This is negative evidence against simply adding
  more scalar token weights to the current autoregressive path. The next repair
  should be a real trainable expression/value decision surface or body-token
  policy integrated into decode, with an A/B against this exact report.
- Expression-slot and vocabulary-closure follow-up:
  `scripts/neural_seed_token_decoder_support.py` now emits target-side
  expression intent slots for calls, binary operators, return expressions,
  loop/top-level updates, branch predicates, indexing, and comprehensions.
  `scripts/strict_generator_mlx_pretraining_probe.py` now closes the semantic
  slot-head ABI by appending all staged private/licensed semantic slot tokens to
  the target vocabulary when the slot head is enabled. This fixed the concrete
  missing-label issue in
  `reports/strict_generator_mlx_pretraining_expression_slot_vocab_mode_smoke_v1.json`:
  train and eval `missing_slot_token_count` are both `0`, `1472` slot tokens are
  accounted for, and the semantic-slot heldout accuracy moved from `0.003135`
  to `0.329154`. This is still not behavior evidence. Expression-call,
  expression-update, and expression-return heldout role accuracies remain `0.0`
  on the smoke, and the broad4 decode reports
  `reports/strict_generator_mlx_decode_eval_expression_slot_vocab_mode_smoke_broad4_v1.json`,
  `reports/strict_generator_mlx_decode_eval_expression_slot_vocab_mode_smoke_paren_guard_broad4_v1.json`,
  and
  `reports/strict_generator_mlx_decode_eval_expression_slot_vocab_mode_smoke_expression_guard_broad4_v1.json`
  all emit zero completed candidates. The latest task-blind expression guards
  block pathological open-parenthesis and incomplete-expression newlines, but
  the beam still starves around unfinished `append(...)` continuations and
  repeated standalone identifier lines.
- Role-prefix subspace and expression-role class balancing are now implemented
  in `scripts/strict_generator_mlx_pretraining_probe.py`. The smoke report
  `reports/strict_generator_mlx_pretraining_expression_slot_subspace_smoke_v1.json`
  improved heldout semantic-slot accuracy from `0.094044` to `0.376176`, but
  `reports/strict_generator_mlx_decode_eval_expression_slot_subspace_smoke_broad4_v1.json`
  still emitted zero completed candidates. The larger private/licensed MLX run
  `reports/strict_generator_mlx_pretraining_expression_slot_subspace_full_v1.json`
  consumed `606261` token positions on MLX GPU, improved heldout LM loss from
  `8.589429` to `4.621592`, and moved expression-update to `2/18` and
  expression-return to `3/18`; expression-call remained `0/31`, semantic-slot
  loss worsened, and candidate-generation credit remains `0` for the auxiliary
  head and task-blind guards. The paired decode
  `reports/strict_generator_mlx_decode_eval_expression_slot_subspace_full_broad4_v1.json`
  emitted `6` candidates, `0` verifier passes, and `0.0` nontrivial-return
  rate. The failure shape is now clearly expression-call/value construction:
  malformed `dict(not not ...)` loops, builtin/type loop sources, shallow
  update conflicts, and unfinished local returns.
- Two decode-side expression repairs were tested and did not clear the wall.
  `scripts/strict_generator_mlx_decode_plans.py` now blocks repeated unary
  `not` chains in expression value slots, and
  `scripts/strict_generator_mlx_decode_eval.py` has an opt-in
  `--enable-learned-expression-token-bias` that softly rescales body-token
  probabilities from model-generated expression slots without rendering code or
  taking learned-generation credit. On the same broad4 private split,
  `reports/strict_generator_mlx_decode_eval_expression_not_guard_full_broad4_v1.json`
  and
  `reports/strict_generator_mlx_decode_eval_expression_slot_token_bias_full_broad4_v1.json`
  each emitted only `2` candidates, `0` verifier passes, and `0.0`
  nontrivial-return rate. This is negative evidence for decode-side bias as the
  next route. The next repair needs training signal for loop-source selection,
  expression-call arguments, update semantics, and return-value construction,
  not another hard-coded expression continuation.
- Expression-call slot canonicalization is now implemented as auxiliary target
  labeling, not as rendering or tool credit. `scripts/neural_seed_token_decoder_support.py`
  collapses long-tail call names into task-blind call families such as
  common builtins, text transforms, mapping lookup, sequence mutation, and
  user/library calls. On the canonical smoke report
  `reports/strict_generator_mlx_pretraining_expression_slot_canonical_smoke_v1.json`,
  total semantic-slot labels fell from `1472` to `667`, expression-call labels
  fell from `635` to `32`, heldout semantic-slot accuracy moved from
  `0.081505` to `0.39185`, and expression-call role accuracy reached `15/31`.
  The behavior result stayed negative. The paired broad4 decode
  `reports/strict_generator_mlx_decode_eval_expression_slot_canonical_smoke_broad4_v1.json`
  emitted `4` integrity-clean runtime-loaded transformer/hybrid candidates
  with `0/4` verifier pass and `0.0` nontrivial-return rate. The opt-in
  call-family token-bias A/B
  `reports/strict_generator_mlx_decode_eval_expression_slot_canonical_smoke_token_bias_broad4_v1.json`
  produced the same outcome. The fresh failure shape is a semantic collapse to
  `items = items + value` inside generic loops, even when the slot metadata
  expects call-like updates. That makes the next real wall the learned
  expression/value-to-body transition, not vocabulary closure, parser guards,
  or token-level call bias.
- Expression-transition body markers are now implemented as a stripped learned
  target mode, not a renderer. The mode
  `plan_semantic_stmt_expression_transition_body_tokens_v1` inserts bounded
  `TRACE:EXPR_*` markers after `SLOT:BODY_START`; `decode_body_tokens` strips
  them before Python compilation, and the strict grammar caps them per line so
  they cannot become marker spam. The smoke report
  `reports/strict_generator_mlx_pretraining_expression_transition_smoke_v1.json`
  improved heldout LM loss from `8.246911` to `6.516979` and semantic-slot
  accuracy from `0.037618` to `0.432602`, but
  `reports/strict_generator_mlx_decode_eval_expression_transition_smoke_broad4_v1.json`
  failed closed with zero emitted candidates. A larger private/licensed medium
  rung,
  `reports/strict_generator_mlx_pretraining_expression_transition_medium_v1.json`,
  improved heldout LM loss to `5.521098` and recovered candidate emission in
  `reports/strict_generator_mlx_decode_eval_expression_transition_medium_broad4_v1.json`,
  but behavior stayed `0/4` and nontrivial-return rate stayed `0.0`. The medium
  bodies now collapse to `items = 0` followed by `items = items + value`; the
  residual mix improved only from pure type-handling to type-handling plus
  wrong-answer. This confirms transition markers alone are not enough. The
  next strict-generator repair needs source-conditioned update/call/return
  semantics, likely by strengthening correct-prompt versus mismatched-prompt
  pressure over body spans rather than adding another target-side label family.
- After-body-start source contrast is now implemented in
  `scripts/strict_generator_mlx_pretraining_probe.py` as an explicit
  `--source-contrastive-span-mode after_body_start` option. It compares
  correct-prompt versus deterministic in-batch mismatched-prompt loss only
  after `SLOT:BODY_START`; it still uses admitted private/licensed rows only
  and does not inspect tests, solutions, public data, answer metadata, teacher
  output, or verifier labels. The first smoke
  `reports/strict_generator_mlx_pretraining_expression_transition_body_contrast_smoke_v1.json`
  improved heldout LM loss from `8.246911` to `5.969187`, moved the
  after-body source gap from `-0.002314` to `0.000391`, and moved semantic-slot
  accuracy to `0.438871`. The behavior report
  `reports/strict_generator_mlx_decode_eval_expression_transition_body_contrast_smoke_broad4_v1.json`
  stayed `0/4` with `0.0` nontrivial-return rate. Generated bodies used
  `values = []` and `values = values + value`; no call updates appeared despite
  the prefix expectation. This proves the new loss plumbing works, but the
  weight/run was far too weak to create source-bound update semantics.
  The medium rung
  `reports/strict_generator_mlx_pretraining_expression_transition_body_contrast_medium_v1.json`
  improved heldout LM loss further to `5.49157` and ran at `4961.895` token
  positions/second, but the after-body source gap remained tiny at `0.000345`.
  Its broad4 decode
  `reports/strict_generator_mlx_decode_eval_expression_transition_body_contrast_medium_broad4_v1.json`
  again emitted `4` clean runtime-loaded candidates with `0/4` behavior and
  `0.0` nontrivial-return rate; generated bodies returned to `items = 0` and
  `items = items + value`. This closes the current body-span contrast attempt
  as negative capability evidence. The next repair should not be a longer run
  of the same objective unless it first changes the contrast design enough to
  create a materially larger source/body separation signal.
- Current strict-generator wall after expression-slot evidence:
  target-mode/vocabulary accounting is now cleaner, but learned expression-token
  distribution and completion are still not good enough. The next non-naive
  repair should train a role-conditioned expression/value policy or a
  body-token transition head that directly improves call arguments, loop
  updates, final return expressions, and EOS/completion behavior under the same
  private/licensed no-cheat contract. Do not spend public calibration or count
  deterministic expression guards as learned generation until completed
  candidate coverage and verifier pass move above zero on private heldouts.

## Post-Overnight Course Correction

Local review plus Claude Desktop's read-only review agree on the generator wall:
the strict learned generator cycle has been producing machinery around a
capability blocker. Roughly thirty target/objective/decoder variants have
improved proxies such as LM loss, semantic-slot accuracy, syntax, and
nontrivial-return rate, but the unassisted behavior result remains pinned at
zero on the meaningful private heldout checks. The latest medium MLX
body-contrast run is representative:
`reports/strict_generator_mlx_pretraining_expression_transition_body_contrast_medium_v1.json`
is a real GPU training run with clean gates, but its source/body separation is
still tiny (`0.000345`), and
`reports/strict_generator_mlx_decode_eval_expression_transition_body_contrast_medium_broad4_v1.json`
stays `0/4` with generic accumulator bodies.

The roadmap should now be read with this active ordering:

1. Implement the AI_book architecture as Theseus' operating substrate before
   more training or benchmarking. Every phase stays in scope, including the
   authority, constitutional, VCM transaction, resource, simulation, semantic
   IR, compression, substrate-adoption, and research-backlog phases.
2. Ship the assisted, verifier-gated daily assistant path as the product-facing
   vertical slice through that substrate. VCM, tools, retrieval, planning, and
   deterministic scaffolds are product machinery, not learned-generation
   evidence. They can deliver usefulness when every executable output is
   verifier-gated and honestly labeled.
3. Maintain the claim ledger and registry as enforcement surfaces. The current
   registry gate is GREEN, so the next work is not to invent more registry
   concepts; it is to make every AI_book-derived concept actually route through
   the registry and execution spine.
4. Freeze strict-generator experiment churn until the implementation substrate
   is complete. The next learned-generation experiment must be preregistered as
   one bounded correctness-in-the-loop run with a falsification stop, not
   another target-side marker or guard variant.
5. Keep SymLiquid protected as a matched comparator, but do not spend scarce
   MLX/GPU time scaling it unless it wins repeat matched evidence.

Stop doing until those gates move:

- Do not add another target-label, slot, vocab, marker, guard, token-bias, or
  contrast family to strict learned generation unless it is inside the single
  bounded correctness experiment above.
- Do not optimize auxiliary proxies as if they predict behavior. LM loss,
  semantic-slot accuracy, syntax, and nontrivial-return are diagnostics, not
  promotion evidence.
- Do not treat broad4-sized evals as capability evidence. Use them only as
  smoke tests.
- Do not withhold the assisted path from product use because it is not learned
  generation. The measurement rule is correct for capability claims; it is not
  a reason to avoid shipping a verifier-gated local assistant.
- Do not remove the late roadmap phases. Instead, implement them as quiet
  infrastructure and governance substrates. They should constrain and organize
  work, not become a new source of report churn.

Hard-to-fake evidence required next:

- A phase-by-phase implementation matrix that maps every AI_book-derived
  roadmap item to a registry surface, implementation record, execution-spine
  hook, gate, and smallest missing patch.
- Product-facing assistant trace evidence:
  `reports/theseus_assistant_product_spine_smoke.json` exercises the implemented
  substrate with VCM packet, authority receipt, resource route, deterministic
  tool evidence, private verifier receipt, claim ledger entry, artifact refs,
  and accepted metadata-only dogfood event.
- Candidate integrity and blind information-flow audits for every future
  learned capability claim.
- Later, after implementation completeness: one correctness-in-the-loop report
  with reward-present/reward-removed ablation and a written stop decision.

## Fourth-Pass AI Book Parity Addendum (2026-07-04)

Provenance: written against the AI_book tree at its 2026-07-04 state — 44
consolidated chapters, the Idea Depth Program era (receipt faithfulness,
epistemic trusted computing base, human oversight degradation, partition
governance, interpretability evidence, amendment legitimacy), the
verification-bandwidth capacity model, residual-honesty conservation, the
governance-tax trade-off lane, the contribution novelty ledger, per-chapter
core-claim dispositions, six accepted narrow evidence transitions (one
empirical), and the book's Logical Conclusion five-test definition of done.
The Third-Pass Addendum above predates all of this.

Standing intent from Corben: **Theseus is the exemplar implementation of the
book.** If the book establishes a mechanism as useful, Theseus implements the
best version of it; once everything is implemented and trained, Theseus goes
public as the reference repo for the book's practices. This addendum is the
work-card source for that parity, not evidence of it. Ingest it through the
Phase 19 machinery: refresh the crosswalk, register backlog rows, and update
the implementation matrix through registry-owned changes only. Nothing here
creates a capability claim, relaxes the Operating Charter, or bypasses the
Breadth Freeze — every item routes into an existing phase and is tranche-gated
below so current priorities keep absolute precedence.

### Step 0 - Refresh the crosswalk against the consolidated book

The crosswalk refresh has now run against the current 44-chapter AI_book tree:
`scripts/roadmap_implementation_gate.py --gate` reports `44/44` chapter-level
implementation rows in `configs/roadmap_implementation_matrix.json`, `0`
stale phases, and `41` active backlog rows. Keep the merged/folded chapter-ID
notes below as historical remapping guidance for future source-sync reviews,
not as a current stale-state claim:

- `planforge-dags-and-intelligence-arbitrage` -> `planning-as-a-control-layer`
- `command-contracts-and-semantic-interfaces` -> `intent-to-execution-contracts`
- `semantic-pages-context-cells-and-certificates` -> `virtual-context-abi`
- `unified-adaptive-tribunal-and-adversarial-review` -> `spinoza-verification-and-proof-carrying-claims`
- `agency-dignity-and-corrigibility` -> `constitutional-alignment-substrate`
- `governance-rights-fork-exit-and-audit` -> `moral-uncertainty-and-value-conflict`
- `generate-verify-repair-compression` and
  `semantic-representation-and-tree-structured-models` -> `compact-generative-systems-and-residual-honesty`
- `moecot-runtime-and-multi-core-orchestration` -> `routing-heads-and-specialist-cores`
- `simulation-fidelity-and-physical-constraints` -> `resource-economics-and-token-budgets`

### Book-parity invariants

Theseus should eventually implement every load-bearing ASI Stack mechanism, but
parity is not a prose claim. A book mechanism is considered represented in
Theseus only when all of the following are true:

- It has a registry-owned abstraction or explicit binding to an existing
  abstraction; no loose sidecar implementations.
- It has a stable implementation record, authority boundary, typed failure
  behavior, and replacement/rollback path where replacement is possible.
- It emits VIEA-compatible artifacts: command/intent contract when applicable,
  artifact graph refs, claim/support-state refs, receipts, residuals, and
  non-claims.
- It has a gate or fixture that can reject at least one malformed or
  overclaimed case, not just a happy-path smoke.
- It records whether the evidence is live-run, fixture-only, import-only,
  static digest, or blocked by public-safety/private-artifact constraints.
- If the book can cite it, Theseus emits a public-safe evidence pack with
  digests and exact non-claims; private payloads, benchmark payloads, secrets,
  and teacher-private material stay out.
- If it changes `AGENTS.md`, registry policy, capability semantics, authority,
  routing, training data admission, benchmark admission, or publication claims,
  it is treated as a governance/amendment event, not an ordinary refactor.

Phase routing below uses the human roadmap phase language. The
machine-readable source of truth remains
`configs/roadmap_implementation_matrix.json`; when the two disagree, Phase 19
must reconcile the matrix first instead of hiding the drift in narrative text.

### Fourth-pass parity table (new book concepts since the third pass)

| Book concept | Theseus should do | Current gap | Phase routing |
|---|---|---|---|
| Receipt faithfulness / record-reality gap (idea-depth keystone 1; `experiments/receipt_faithfulness_adversarial`, `experiments/receipt_repository_audit` in AI_book) | Generalize the anti-cheating/candidate-integrity discipline into universal receipt attestation: trap fixtures, randomized deep replay audits, and cross-component verification for effect receipts, audit logs, and gate reports — no record trusted because it exists. | Candidate integrity already embodies this for learned-generation claims; effect receipts, gate reports, and ledger writes elsewhere are not yet trap-tested or randomly re-audited. | Phase 5, Phase 13 |
| Epistemic trusted computing base (keystone 2) | Name Theseus's minimal trusted core explicitly (verifier harness, blind information-flow auditor, ledger/digest writers), record trust roots and the attestation chain outward, and add verifier-of-verifier rotation or independent audit for the core itself. | The trusted components exist and are strong, but nothing names the TCB, bounds it, or audits the auditors. | Phase 5, Phase 13 |
| Human oversight degradation (keystone 3) | Instrument the operator/approval surfaces: approval latency, rubber-stamp rate (approval-without-inspection proxies), queue depth, alert precision; add reviewer-load fields to high-impact approvals and fail closed when fatigue indicators cross thresholds. | Approval gates exist; the human side is uninstrumented, so approval quality is assumed rather than measured. | Phase 10, Phase 13 |
| Partition governance / revocation propagation (book: personal-compute-hives partition section; `experiments/partitioned_authority`, `experiments/authority_revocation_propagation`) | Define authority consistency under Hive partition: revocation propagation SLAs, grant/effect race rules, offline-node authority decay, and partition-state receipts. | Hive is policy-frozen; no partition-governance semantics exist. Record as backlog now; implement at Hive unfreeze. | Phase 12 (backlog until unfreeze) |
| Interpretability as an evidence class (keystone 5) | Add white-box/probe evidence as a distinct rung in promotion evidence: internal-state probes on the student/SymLiquid substrate, substrate-attribution probes for the discovery lane's matched-control verdicts. | Promotion evidence is entirely behavioral; the discovery lane's "did the substrate matter" question is exactly where mechanistic evidence pays. | Phase 4, discovery track |
| Constitutional amendment legitimacy (keystone 6) | Treat changes to `AGENTS.md`, the charter, personality documents, and constitutional predicates as amendment records: proposer, review, dissent, ratification, rollback, and effective date. | Charter changes are ordinary edits today; the governance layer that governs everything else does not govern its own amendment. | Phase 14 |
| Verification-bandwidth capacity model (book capacity model: units, pairwise obligations, verifier capacity, decomposition contracts) | Make verifier capacity a scheduled budget in the fanout/STS/verifier-cascade harness: obligations counted per claim, capacity per run, residual obligations ledgered, decomposition contracts named when clustering reduces obligations. | Verifiers exist and are respected, but verification is not yet budgeted as explicit capacity with residual-obligation accounting. | Phase 4, Phase 16 |
| Residual-honesty conservation law (`experiments/residual_honesty_conservation`, `experiments/residual_ledger_trace`) | Add a conservation audit over residual escrow: every residual is moved, priced, or discharged — never deleted; per-run escrow diff audit that fails on silent residual disappearance. | Residual escrow exists; conservation is convention, not a checked invariant. | Phase 6, Phase 9 |
| Governance-tax accounting (book: resource-economics governance-tax trade-off bridge) | Measure the governed-versus-raw overhead on real runs: per-gate latency/compute/review-load, tax-per-route records, and tax-versus-caught-failure accounting. This is also the book's missing empirical economics evidence — Theseus is where it can be measured for real. | Runtime load is tracked; governance overhead is not isolated as a measured, ledgered quantity. | Phase 16 |
| Pattern language / layer-contract local deltas (book pattern spine) | Extend VIEA spine records with per-subsystem layer-boundary records that name each layer's local delta (what it owns beyond the shared pattern), and keep one generated layer map. | Spine records normalize record families; explicit layer-contract ownership per subsystem is implicit. | Phase 1 |
| Contribution/novelty ledger (book `docs/contribution_novelty_ledger.md`) | Formalize the discovery-track rule as a ledger: for SymLiquid/CGS/VSA/STS and any novel mechanism — claimed contribution, closest baseline, matched-control delta, confidence label, next artifact. | The matched-control discipline exists in the charter; it is not yet a durable ledger surface. | Discovery track, Phase 11 |
| Capability-claim disposition ledger (book `claim_decisions/v1_x_core_claim_dispositions.json`) | For every standing Theseus capability claim: current support state plus exactly what evidence would move it — no silent stalls; render alongside the registry. | Promotion gates say what blocks; a per-claim disposition surface does not exist. | Phase 5, Phase 9 |
| Evidence-transition export packs (book non-core ledger + transition records) | Make "book-importable public-safe evidence pack" a standard output: digests, exact commands, baselines, negative controls, residuals, non-claims, support-state label — emitted for every GREEN gate worth citing. | 53 evidence pointers exist; packaging is ad hoc per import rather than a standard emit format. | Phase 11, Phase 19 |
| Reference-trace export — flagship cross-item (book integrated-reference chapter demands a live trace with real layer handoffs) | Produce one real end-to-end trace — intent -> contract -> plan -> VCM packet -> route -> verification -> execution -> evidence -> improvement gate — from an actual Theseus run, exported public-safe with digests, and keep it reproducible. This single artifact closes the book's largest open evidence ask and Theseus Phase 11 simultaneously. | The assistant product-spine smoke is close; it is not yet packaged as the book-shaped reference trace with all hops, authority deltas, and residuals. | Phase 1, Phase 11 |
| Run-discipline parity (book clean-handoff, burn-down truth, status-sprawl rules) | Adopt: never end a run with failing gates or an uncommitted increment; reconcile implementation-matrix status against artifacts each run in both directions; restructure any status surface whose enforced text exceeds ~100 words into count-plus-ledger-link form. | Theseus practice is close (walls are written plainly); the rules are not standing policy. | Phase 0 |
| Book-schema conformance (book `schemas/` — 76 record schemas with fixtures) | Where a Theseus record claims a book shape (spine records already do), validate against the book's schema in CI, pinned by digest to a book commit; divergences become explicit abstraction notes, not drift. | Spine materialization is book-shaped by construction but not schema-checked against the book's actual files. | Phase 1, Phase 19 |

### Tranche plan (Breadth Freeze compliant)

- **Tranche A — harness honesty on existing lanes (allowed now; serves
  Current Priorities 1-4 directly):** receipt-attestation generalization
  (trap fixtures + randomized re-audit), verifier capacity budgeting,
  residual conservation audit, governance-tax measurement, oversight-fatigue
  instrumentation on existing operator surfaces, run-discipline parity,
  capability-claim disposition ledger, and the reference-trace export. Every
  one of these makes the existing harness more honest or the existing
  evidence easier to audit — the charter's definition of progress.
- **Tranche B — governance completions (gated on the neural-seed verdict and
  repo-health priority; no new product surfaces):** epistemic TCB naming and
  verifier-of-verifier audit, amendment-legitimacy records, tribunal records
  for architecture changes, interpretability probes for the discovery-lane
  verdict, pattern-language layer map, novelty ledger, book-schema
  conformance in CI.
- **Tranche C — Hive unfreeze and public release (gated on "implemented and
  trained" per Corben's direction):** partition governance and revocation
  propagation, standard evidence-pack emission, and the Public Release
  Program below.

### Public Release Program (end-state, Tranche C)

When Corben calls the system implemented and trained, going public follows
the book's own release discipline:

1. Public-safe audit sweep: secrets, private data, teacher-derived rows,
   benchmark payloads, and personal material scrubbed or quarantined per
   `docs/DATA_AND_ARTIFACTS.md`.
2. Claims audit: every README/doc capability statement mapped to a support
   state with book-style non-claims; anything unsupported is demoted or
   deleted before publication, never softened.
3. Sixty-second trust surface: a cold reader must see within one minute what
   Theseus is, what is proven versus argued, how to replay the evidence, and
   where the book defines the practices it implements.
4. Reproducibility floor: pinned toolchains, one-command smokes for every
   headline gate, and digest-verified fixture packs.
5. Tagged release with an exact release record (artifacts that exist, checks
   that ran, residuals that remain), mirroring the book's v1.0.0 release-gate
   pattern.

### Success alignment

The Success Definition below stands unchanged. This addendum adds one
interpretation: Theseus is the book's proof that the practices work, and the
book is Theseus's specification of what "working" must mean. The two-way
contract — book schemas and tests consumed here, Theseus evidence packs and
the reference trace exported there — is how both projects reach their logical
conclusions without either one overclaiming. ASI remains the north star and
remains a non-claim until evidence says otherwise; the parity program's job is
to make every step toward it inspectable.

## Success Definition

Theseus is on track when all of this is true:

- The repo has one canonical control spine.
- The registry answers what Theseus is, what each capability means, what backs
  it, and what can replace it.
- The assistant is useful in daily work and logs real accepted/missed/ignored
  outcomes.
- The practical generator improves on verifier-passing private heldouts without
  cheating.
- Public benchmark calibration remains measurement-only and becomes more
  informative, not more gamed.
- Deterministic tools, VCM, STS, planning, and routers are integrated as parts
  of one system rather than separate proof islands.
- SymLiquid remains a protected research lane, but the practical assistant uses
  the best evidence-backed architecture.
- The book can point to Theseus as an implementation reference with honest
  support states, not inflated claims.
