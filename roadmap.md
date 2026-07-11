# Project Theseus Roadmap

Consolidated 2026-07-07; reconciled with the 2026-07-09 AI Stack manifest. This
roadmap lists only work that still needs doing. It is the forward plan; it is not
an audit trail. Historical execution logs, dated
book-mining/review passes, and per-experiment records were removed from this file
during consolidation and remain in git history. The machine-readable coverage
source of truth is `configs/roadmap_implementation_matrix.json`
(`scripts/roadmap_implementation_gate.py --gate`).

Goal: make Theseus the working implementation reference for the ASI Stack book
(`/Users/corbensorenson/Documents/AI_book`) - stable interfaces, governed
replacement, real evidence, useful daily operation, and a local model that
improves without cheating.

## How To Read This Roadmap

Precedence when sections disagree: the **Capability Plan v2** decides sequencing;
the **Phases** decide mechanism; the **Book-Parity Backlog** decides book coverage.
When this file and `configs/roadmap_implementation_matrix.json` disagree, reconcile
the matrix first (Phase 19), do not paper over drift in prose.

Read order: Status at a Glance -> Non-Negotiable Rules -> Capability Plan v2 ->
the Phase that owns the surface you are working on -> the Quality Bars a "done"
module must meet.

## Status at a Glance

| Area | Owner | State | Next concrete action |
|---|---|---|---|
| Data engine + curriculum | Track 0 / Phase 7 | governed scalable intake implemented; still below mature scale | 35,297 deduplicated code functions provide 5.58M encoded one-pass positions; 13,918 human-contributed conversations add 4.29M redacted/decontaminated positions |
| Sparse specialist core (100M) | Track 1 / Phases 10, 16 | implemented comparator; not adopted | dense wins matched 100K diagnostics and admissibility; revisit sparse only after a behavior win |
| Verifier-guided search | Track 2 / Phases 6, 10 | not started | propose->verify->repair search; beat one-shot on held-out |
| Correctness training (DPO->GRPO/RLVR) | Track 3 / Phase 10 | DPO shadow ran; behavior flat | rerun after proposer/search floor; then verifier-reward curriculum |
| Fast-gen modes (MTP/diffusion/self-draft) | Track 4 / Phases 8, 10 | not started | MTP first; ablate vs AR baseline on accepted output |
| Generator capability (held-out pass) | Phase 10 | RED - direct, plan-conditioned, and zero-unknown rungs all score 0/24 | retain body-only as the practical baseline; improve prompt-conditioned semantic behavior before preference/RL or scale |
| Self-improvement flywheel | Tracks 0, 3 / Phases 7, 10 | not started | generate->verify->admit->retrain->ratchet after a proposer floor |
| VCM ABI + transactions/certificates | Phase 3 | implemented; synthetic + exact-backend runtime evidence | use the ABI in Phase 7/10 model work; preserve equal-budget on/off and backend-scoped claims |
| Claim ledger + belief revision | Phase 14 | implemented; synthetic/replay infrastructure evidence | keep new citeable gates in the canonical pack/TCB/dependency audit; earn empirical support from real revisions |
| Replacement transactions | Phase 2 | synthetic-test-backed | one real default swap with rollback |
| Procedural memory + toolification | Phase 15 | implemented; three real metadata workflows guarded | keep lifecycle receipts live; stale/drifted routes retire and exact lookahead abstains on ambiguity |
| Authority kernel / SCIF | Phase 18 | synthetic-test-backed | one real side-effecting run |
| Assistant product | Phase 5 | synthetic-test-backed | earn empirical support from real multi-day use |
| Report hygiene | Phases 0, 8, 14 | budget GREEN: hot reports 1.02 GB, active index 211 MB, checkpoints 6.32 GB physical | keep current-reference compaction and exact replay mandatory as artifacts grow |
| Book crosswalk / parity | Phase 19 | 49 chapters mapped | keep matrix synced; import Theseus evidence into AI_book |
| Book futures intake | Phase 19 + routed owners | 19 chapter candidates + 2 section routes mapped | activate only at source/ownership and operational entry gates |

## Non-Negotiable Rules

- No open/base/pretrained model weights, ever, for the from-scratch student lanes.
- Family-disjoint held-out verifier pass is the only capability scoreboard.
  Syntax, nontrivial-return, LM loss, and preference-gap are diagnostics, never
  wins.
- Public benchmarks are calibration only. Never train on public benchmark
  prompts, tests, hidden tests, solutions, traces, score labels, or templates.
- External inference is only governed teacher-side training support; it is never
  served to a user at runtime.
- Teacher rows are allowed only when governed, provenance-tagged, license-checked,
  verifier-accepted, leakage-audited, and barred from runtime serving.
- No fallback returns for capability credit. Routers, templates, deterministic
  tools, semantic renderers, action catalogs, and search are tools/baselines,
  never learned generation. Always report model-only versus assisted separately.
- Candidate family and integrity are recomputed independently, not trusted from
  self-declared row flags.
- Every routeable capability has an abstraction, implementation, stable capability
  field, evidence output, rollback rule, and cleanup policy. Prefer improving a
  registered surface over adding a new one.
- Every capability change ships a substrate-adoption record: baseline, negative
  control, matched data/compute, falsification condition, residuals, non-claims.
- No new report family without a registry owner, retention class, and a reason it
  cannot update an existing flagship claim record.
- Negative results are kept. A refuted lever is progress.

## Implementation Standard (Beyond State of the Art)

Every mechanism on this roadmap is implemented as the strongest known version, not
a toy, demo, or MVP. Reaching the frontier is only feasible if each layer is built
to lead, so the standard for every phase, track, and surface is:

- **Name the SOTA baseline you match or beat.** Cite the best published or open
  method for the mechanism and implement to meet or exceed it, or record exactly why
  you cannot yet and what the gap is.
- **Name the beyond-SOTA / mature endpoint, not just the MVP.** Every surface
  declares two horizons: a minimal implementation that proves the interface, and a
  mature endpoint that is genuinely ahead of the field. An MVP is only ever an
  explicit, temporary stepping stone toward its named endpoint; shipping the MVP as
  the finished surface is not "done."
- **No placeholder passes as complete.** A happy-path fixture, stub, hardcoded
  shortcut, or single-case smoke is never the deliverable. Done means robust,
  adversarially tested, and ahead of the obvious approach.
- **Prefer the harder correct design over the easy fragile one.** Where the book
  names a mechanism, implement the full contract, not a narrowed subset that only
  covers the demo.
- **Efficiency and rigor are the edge, not scale alone.** Beyond-SOTA here means
  best capability-per-active-parameter and best-governed, not biggest. Elegance that
  skips baselines or verification is unproven, not beyond-SOTA.

Enforced by the Quality Bars (each module card names both horizons and the baseline
it beats) and the falsification gates (a lever is not "done" until it beats its
baseline on held-out evidence).

## ASI-Trajectory Capability Plan v2

The controlling capability plan. The Phases below are the mechanism; this section
is the sequencing.

### North Star (honest, ambitious, falsifiable)

Theseus demonstrates the **Efficient ASI Hypothesis**: a from-scratch,
sparsely-activated, verifier-and-search-governed cognitive architecture that
reaches **frontier-competitive capability on verifiable domains (English + code)
at 1-2 orders of magnitude fewer active parameters** than a dense model, running
fully local, with every claim falsifiable and no cheating.

"Over the edge" is defined concretely as **capability-per-active-parameter**
crossing published dense-model curves on a frozen held-out verifiable benchmark.
Not prose, not a green fixture. If that never happens after the critical path is
exhausted, the hypothesis is refuted at this scale and the roadmap says so. This
plan never claims general superintelligence from local compute; it claims a
demonstrable efficient-intelligence architecture on the domains where a governed
local system can actually reach the frontier.

### The Core Architectural Bet

The fuzzy weights do only the irreducible reasoning. Everything else is
deterministic architecture: verification, execution, retrieval, memory, syntax,
and search.

- **100M total parameters, sparsely activated** (target ~10-20M active/token) via
  MoE / Octopus routing / MoECOT. Each expert is a stable capability field
  accountable for an action family (control flow, data structures, arithmetic,
  string/IO, contract shaping), with readiness, authority ceiling, and residual
  accounting.
- **Verifier-guided search in the generation loop** - the missing multiplier. The
  deterministic verifier guides proposal, not just grades it.
- **Correctness trained into the weights** via the Chapter 38 RL/preference
  toolkit, sequenced only after a proposer that sometimes succeeds exists.
- **Data quality is co-equal with architecture.** The efficient-intelligence bet
  fails on a weak corpus; the training-data program (Track 0) is a first-class
  capability lever, not a preprocessing step.
- **100M is the first rung of a scaling ladder, not the ceiling.** The architecture,
  data pipeline, training loop, and search must be designed so active parameters,
  expert count, corpus size, search budget, and reasoning depth scale together as
  compute grows, with no rung-one shortcut that blocks rung two.

### Critical Path (ordered - this is what moves capability)

**Track 0 - Data and Curriculum Program** (foundational; runs ahead of and parallel
to Track 1).
- Make every learning/evaluation candidate a versioned `DataAdmissionReceipt`
  before use: origin, authority/license, provenance class, real/synthetic lineage,
  permitted use, split exclusions, exact/semantic contamination checks, retention,
  deletion scope, evaluation refs, residuals, decision, and non-claims.
- Build a large, curated, license-clean English + code corpus with measured
  deduplication, quality, domain balance, freshness, tail coverage, poisoning risk,
  semantic leakage, and benchmark contamination. Missing authority/provenance
  blocks; missing exclusions/contamination quarantines; incomplete lifecycle data
  remains experimental-only.
- Build a verified synthetic-data engine: the model and tools propose tasks/solutions,
  the private verifier admits only correct, non-leaking, provenance-clean rows, and
  admitted rows expand the curriculum. Preserve generator/version lineage and track
  recursive synthetic share, diversity, tail support, and model-collapse signals.
- Design a difficulty curriculum with tiers the current model can reach, growing
  toward harder algorithm families, so RL and search always have a non-zero reward
  signal to climb.
- Compare replacement, accumulation, targeted replay, quarantine, and retraining on
  a frozen workload. Measure utility, forgetting, coverage/tail retention,
  calibration, privacy/revocation risk, freshness, storage/compute, and deletion
  cost; do not assume one universal policy.
- Carry revocation/deletion closure through source rows, transforms, datasets,
  checkpoints, adapters, caches, VCM/retrieval indexes, distilled artifacts,
  reports, and publications. Report each descendant as removed, invalidated,
  retrained, withdrawn, retained-by-policy, or unverified; never call a request or
  source-file deletion proof of parametric forgetting.
- Acceptance: corpus and lifecycle metrics reported; contamination audit clean
  against every held-out set; verified synthetic rows carry provenance + verifier
  receipts; curriculum tiers have measured current-model pass rates; a controlled
  policy comparison and negative deletion-propagation fixture replay independently.
  A complete receipt proves bounded eligibility only, never quality, privacy,
  unlearning, training success, or capability. Beyond-SOTA endpoint: a stack-wide,
  descendant-aware, self-refreshing data governance plane, not a static dump.

**Track 1 - The Sparse Specialist Core** (owned by Phases 10 and 16).
- Scale to a 100M sparse MoE proposer: top-k expert routing, small active count;
  establish the held-out loss/perplexity curve vs active params.
- Octopus route head + live MoECOT: a learned router selects experts per
  token/task and emits real `RoutingDecisionRecord` and `MoECOTOrchestrationRecord`
  (why this expert, why not others, readiness, authority, residual).
- Expert accountability: experts specialize to action families, verified by
  activation attribution.
- Acceptance (hard-to-fake): held-out family-disjoint verifier pass moves off 0;
  active params much less than total; per-expert activation attributable;
  sparse-specialist beats a dense model with matched active-param count.

**Track 2 - Verifier-Guided Search** (owned by Phases 6 and 10).
- Propose -> verify -> repair loop with a real search budget: beam/tree over
  candidates, verifier feedback prunes and steers, repair localized over semantic
  IR rather than full regeneration.
- Search-budget -> pass-rate curve; planner selects budget by task risk.
- Extend search into deliberate reasoning / test-time compute: long chain-of-thought,
  self-reflection, and multi-step deliberation are search over reasoning states, not
  a bigger single forward pass. Beyond-SOTA endpoint: an o1/R1-class deliberate
  reasoner where extra test-time compute measurably raises held-out verifier pass,
  reasoning traces verified and reported separately from model-only decode.
- Acceptance: verifier-guided search materially beats one-shot decode on held-out;
  model-only vs search-guided reported separately. This is the decisive test of
  the efficiency thesis; validate it before scaling further.

**Track 3 - Correctness-in-the-Loop Training** (owned by Phase 10).
- DPO on existing accepted/rejected verifier pairs (rerun after Tracks 1-2 give a
  proposer floor), then GRPO/RLVR verifier-reward on an achievable curriculum, as
  governed policy-update leases with reward-hacking probes, drift bounds, rollback.
- Drive `teacher_share -> 0` as verified self-generated rows grow.
- Acceptance: RL/preference moves held-out pass with an ablation (reward removed,
  same compute, stays flat -> proves the signal caused it); probes clean;
  self-generated-verified rows rising.

**Track 4 - Fast Generation Modes** (owned by Phases 8 and 10).
- MTP (denser signal + lookahead), diffusion/sketch-first, self-drafting heads, as
  governed generation-mode routes scored on useful-solution-per-second.
- Acceptance: each mode ablated vs AR baseline on held-out; accepted output
  counted; no mode promoted on raw tokens-per-second.

### The Self-Improvement Flywheel (continuous engine)

The engine that makes reaching the frontier feasible over time rather than in one
shot: generate -> verify -> admit -> retrain -> measure -> ratchet.
- The model (with search and tools) proposes solutions; the private verifier admits
  only correct, provenance-clean, non-leaking rows; admitted rows expand the Track-0
  curriculum; Track-3 training folds them into the weights; the capability ratchet
  measures the lift; `teacher_share` trends to 0 as verified self-generated data
  takes over.
- Every turn is a governed policy-update lease with reward-hacking probes, drift
  bounds, rollback, and a held-out measurement, so self-improvement cannot drift,
  cheat, or expand its own authority.
- Acceptance: verified self-generated rows rise turn over turn; the capability
  ratchet shows monotonic held-out improvement (or a recorded plateau/falsification);
  teacher_share decreases; no reward-hacking or authority creep. Beyond-SOTA
  endpoint: a durable recursive-self-improvement loop with evaluator independence,
  not a one-off distillation.

### Scaling Trajectory

100M sparse is rung one. Define an explicit ladder (active params, expert count,
corpus size, search budget, reasoning depth) with a governed step function: each
rung requires a matched-baseline ablation, a capability-per-active-parameter
datapoint, and a falsification condition before the next rung is funded. This is
how "reach the frontier" becomes a measured trajectory instead of a single bet.

### Long-Horizon Autonomy (later capability stage)

After the core engine clears its capability gate, extend from single-task assistant
to governed long-horizon autonomous operation: multi-step goals executed through the
Labor OS typed-job spine (Phase 1), under authority ceilings and agency-rights checks
(Phases 2 and 18), driven by the self-improvement flywheel. Do not build autonomous
long-horizon operation before the assistant is genuinely useful and the generator
clears its gate; premature autonomy is sprawl, not capability.

### Supporting Layer - The Governed Cognitive System

The beyond-SOTA governance contribution: what makes even a small model
trustworthy. Promote each from fixture to one real run (support state reaches
empirical- or replayable-reference-backed): VCM ABI + certificates + transactions
(Phase 3), verification-bandwidth routing (Phase 8), claim ledger + belief revision
(Phase 14), semantic IR + localized repair (Phase 13), procedural memory -> tools
(Phase 15), planning adequacy + intelligence arbitrage (Phases 1 and 8), authority
kernel/SCIF + replacement transactions (Phases 18 and 2).

### Product - The Assistant (owned by Phase 5)

Ship the assisted+verified assistant over Corben's own codebase; verifier-gated;
honest labels. Real daily use logged (distinct days, real intents) feeds the
Track-3 curriculum and Phase-15 procedural tools. The assistant lane stays
synthetic-test-backed until real logged usage exists; synthetic dogfood batches
do not earn an empirical label.

### Deferred Modalities (Later Core Expansion)

The eventual core also includes native English **STT/TTS**, **Mac + Windows
computer use**, and **playing games well**. These are on the roadmap but sequenced
after the core engine, because each is a different modality needing its own
perception/action stack, not a data problem the engine solves for free.
- Native English STT/TTS: audio encoder + vocoder/synthesis, from scratch, own
  corpus and verifier (transcription accuracy, intelligibility).
- Mac + Windows computer use: vision (screen understanding) + UI grounding +
  action model + interaction traces. Hardest modality; gate strictly behind the
  core engine.
- Game-playing: turn-based/text games can ride the core engine + search sooner;
  real-time video games need vision + RL and are far-future.
Sequencing rule: no additional modality starts until the core engine clears its
capability gate (held-out verifier pass off 0 and rising). Each enters as its own
tranche with a baseline, a verifier, and a falsification condition.

### Falsification and Anti-Treadmill Gates

- Capability gate: if held-out family-disjoint pass is still 0 after Track 1 (100M
  sparse), Track 2 (search), and Track 3 (RL), with the Track-0 data program in
  place, the Efficient ASI Hypothesis is refuted at this scale. Record it, keep the
  negative, decide scale-more or narrow-the-domain. No more heads.
- No-head rule: no new value-guard/adequacy/slot/marker/token-bias/contrast variant
  without a preceding Track-1/2/3 result that motivates it, inside a registered
  substrate-adoption sequence with a falsification stop.
- One-flagship rule: exactly one active capability lane at a time; supporting work
  may only repair its dependencies.

### The Honest ASI Scoreboard (`docs/capability_ratchet.md`)

One page, measured not asserted: capability-per-active-parameter on a frozen
held-out verifiable benchmark plotted against published dense-model curves
(crossing the curve is the earned beyond-SOTA claim); held-out verifier pass
(model-only and search-guided); active-param count; useful-solution-per-second;
teacher-share; self-generated-verified-row count; and every falsification condition
with its current status.

## Governed-Surface Work (Phases)

Each phase lists remaining deliverables, acceptance gates, and prohibitions. A
phase is complete only when its acceptance gates hold on a real (not fixture-only)
run and it meets the Quality Bars below.

### Phase 0: Repository Self-Model and Registry Discipline
- Keep the abstraction/implementation registry, SCF bindings, module cards,
  deprecation routes, ownership, evidence refs, and `docs/PROJECT_STATE.md` aligned
  with the current honest generator wall.
- ATT-D rejects unregistered major modules, duplicate default implementations,
  unowned report families, stale route evidence, and capability changes that omit
  rollback or substrate-adoption records.
- Route evidence is lifecycle-aware: each routable implementation names a minimal
  source-bound, TTL-bound, or current-invocation receipt; supporting and retained
  reports remain audit history but neither authorize nor expire a route.
- Enforce live report/checkpoint retention, current-reference-aware compaction, and
  one flagship claim record per metric family; generated bulk is runtime state, not
  source or capability evidence.
- Acceptance: registry/hygiene/module-definition gates have no hard gaps, every
  active route resolves through one current implementation, and the repo can answer
  what Theseus is without scanning historical reports.
- Do not: turn a gate green with another `_vN` family; hide negatives; treat cleanup
  as capability progress.

### Phase 1: VIEA Execution Spine
- Canonical trace: `IntentContract -> CommandContract -> PlanForgeDAG ->
  ContextABIRecord -> TypedJob -> RuntimeAdapterInvocation -> ArtifactGraphRecord
  -> ClaimRecord -> EvidenceTransitionRecord -> Residual -> PolicyOptimizationRecord`.
- Extend `theseus_plan_compiler.py` so every compiled goal emits all these records.
- Make `theseus_assistant_runtime.py` call the spine for all non-trivial code,
  tool, planning, and repo-maintenance tasks; one `viea_execution_spine.jsonl` row
  per meaningful task; artifact-graph/claim/evidence IDs on every capability
  statement; a missing-contract residual when a task cannot pass through the spine.
- Make PlanForge/Labor OS execution ordinary: typed jobs carry lifecycle, adapter,
  inputs/outputs, authority, approvals, failure behavior, audit/replay state, VCM
  requests, verifier burden, resource class, and dependency-local replanning.
- Add planning-adequacy contracts and an intelligence-arbitrage ledger so cheap
  routes cannot hide displaced verification, repair, or human cleanup.
- Acceptance: one real assistant coding task produces the full replayable trace;
  a >=3-node private DAG replays with node-local residual/replan behavior;
  side-effecting tasks carry authority ceilings and effect receipts; the system
  records `UNKNOWN`/`UNSOLVED`/`TOOL_FAULT` instead of inventing success.
- Do not: answer as if work completed with no artifact; collapse planning into
  prose; add a second planner or dashboard detached from typed jobs.

### Phase 2: Stable Capability Fields and Route Authority
- Every routable abstraction has lifecycle state, evidence freshness, authority
  ceiling, residual escrow, monitor/rollback triggers, and content-bound route
  receipts. Default changes use replacement transactions: precheck -> independent
  evaluator -> regression -> residual escrow -> canary -> monitor -> decision.
- Normalize authority transitions and runtime-adapter receipts across assistant,
  planner, tools, training, verifier, Hive, model/router, VCM, and teacher paths.
- Acceptance: one non-trivial real default swap can fail, roll back, and preserve
  its negative evidence; stale/unqualified implementations are not route-eligible;
  a capability cannot expand its own authority during replacement.
- Do not: let `works once` become default; infer authority from locality; let a
  replacement implementation evaluate or authorize itself.

### Phase 3: VCM as the Default Context ABI
- Stable semantic addresses for project/registry/training/session/codebase/residual
  state; `ContextABIRecord` and `SemanticPageCertificate` for assistant, training,
  verifier, planner contexts; adequacy labels (sufficient/partial/stale/tainted/
  missing/overspecified); taint rules barring public-calibration artifacts from
  training context; context-on vs context-off ablations.
- Deployed resolver emits representation certificates (source refs, omissions, loss
  contract, permitted uses, authority ceiling, consumer policy, authority-nonwidening)
  and copy-on-write snapshot branch records (read/write sets, taint propagation,
  deletion obligations, contradiction refs, closure state, typed faults).
- Require transaction records for reads/writes/derivations/branches/summaries/
  public imports/dogfood/deletions. Deleted or revoked material must not reappear
  through summaries, embeddings, caches, retrieval, reports, or training rows
  without an explicit closure fault.
- Acceptance: assistant answers "what is the current wall?" from VCM-backed state
  with source refs; generator train/eval records exactly what context was allowed;
  public rows are tainted calibration-only; mandatory misses fail closed; context-on
  improves one real internal task with no leakage/runtime regression; the real
  MLX/Metal serving path separately proves KV/prefix-cache lifecycle before parity
  is claimed.
- Current implementation evidence: one fail-closed packet ABI is independently
  validated across assistant, planner, verifier, deterministic-tool,
  training-admission, candidate-generation, and fanout paths; `33/33` canonical
  consumers and `25/25` planner nodes pass, while authority-widening,
  source-mutation, taint-drop, best-effort, fallback, missing, stale,
  contradictory, over-compressed, revoked, and deleted controls fail closed. A
  real tiny MLX-LM Llama forward creates, reuses, appends, and invalidates an
  `mlx_lm.models.cache.KVCache`; this supports only an exact `mlx_apple`
  lifecycle claim, not CUDA, custom Metal-kernel, or cross-backend parity. The
  equal-budget private context ablation is `1.0` VCM-on versus `0.5625` VCM-off
  over 16 cases with zero fallback returns; it is integration evidence, not a
  broad public-capability claim. The separate `VCM-Governed` release profile
  remains RED because its quarantined public prompt calibration manifest is not
  currently available; Phase 3 implementation status does not override that
  capability/calibration result.
- Do not: keep proving VCM on tiny saturated benchmarks; store raw private user
  text in training rows by default.

### Phase 4: Candidate Integrity and Learned-Generation Accounting
- Independently recompute candidate family, provenance, blind read-set, loadability,
  fallback/template/tool/router use, public-data contact, and replay equivalence.
  Candidate-emitted labels never establish learned-generation status.
- Every score separates direct learned full-body generation, deterministic/tool
  assistance, structural renderers, search repair, and fallbacks. Public and private
  promotion reports consume the same integrity receipt.
- Acceptance: an adversarial fixture that lies about family/credit is rejected; a
  selected candidate loads, ranks, executes, and hashes identically across replay;
  old body-template inventory can be disabled without changing accounting semantics.
- Do not: treat syntax, nontrivial returns, adapters, action catalogs, templates,
  tools, or verifier repair as learned-generation capability.

### Phase 5: Daily-Use Assistant Runtime and Dogfood Trace Loop
- Make the local assistant the primary daily lane with conversation state in VCM,
  uncertainty-aware answers, code patches/tests/evidence, and accepted/missed/
  ignored/corrected/completed outcome records. Raw private text stays off by default.
- Route nontrivial tasks through the VIEA spine and record context, tools, verifier,
  authority, effect, and dogfood receipts. Feed only governed metadata/outcomes into
  curriculum or policy optimization.
- Acceptance: real multi-day use completes useful repo tasks end to end and earns an
  empirical support label; fixture/e2e metadata remains synthetic-test-backed.
- Do not: claim external teacher output as local inference; let product polish hide
  a weak model; store raw conversations as training data by default.

### Phase 6: Deterministic Tool and Search Substrate
- Register exact math, numeric intervals, code/artifact/BM25/VCM hybrid search, Lean
  checking, private verifier replay, and trace replay as typed tools with authority,
  schemas, cost, trust, checksums, evidence outputs, and typed faults.
- Build verifier-guided beam/tree search with semantic-IR-localized repair. The
  planner selects search/verification budget by task risk and reports model-only,
  search-guided, and tool-assisted behavior separately.
- Acceptance: one real private workflow improves with replayable tool evidence;
  search materially beats one-shot generation on frozen heldout work; failed tools
  produce `UNKNOWN`/`UNSOLVED`/`TOOL_FAULT`, never invented results.
- Do not: hide tool use inside model claims or let deterministic renderers receive
  learned-generation credit.

### Phase 7: Teacher and Data Governance
(Owns Track 0 and teacher-side learning authority.)

Implementation status (2026-07-10): `implemented`. The canonical admission path
now writes and replays `92,032` content-bound candidate receipts, admits
`91,725`, rejects `307` rows, and exposes zero detected exact/semantic
public overlap, fallback markers, or raw-user rows. All nine adversarial cases,
five frozen continual-policy simulations, and the 11-kind positive/negative
descendant-deletion fixture pass. Curriculum and survival-lane materialization
require admitted row hashes. The lifecycle surface remains `YELLOW`, not
`GREEN`, because recursive synthetic share is `0.824771`; that is a data-risk
warning, not proof of model collapse. The policy comparison is simulation, and
deletion closure is graph evidence, not physical unlearning.
- Replace file-level metadata admission with receipt-bound candidate lifecycle:
  provenance/authority, lineage, permitted use, split exclusions, exact + semantic
  contamination, retention, deletion scope, evaluation refs, residuals, bounded
  decision, and non-claims. Incomplete records block, quarantine, or remain
  experimental; admission never promotes capability.
- Govern teacher proposals/distillation with license, provenance, verifier,
  leakage, runtime-serving prohibition, and durable teacher-share ledgers. Track
  recursive synthetic share, coverage, diversity, tail loss, and provenance drift.
- Compare replacement, accumulation, replay, quarantine, and retraining on a frozen
  continual-learning workload. Propagate revocation/deletion obligations through
  every declared descendant and retain unverified closure states.
- Acceptance: the data-policy comparison and a negative descendant-deletion case
  replay independently; teacher rows enter only through their gate; public
  calibration payloads remain excluded; `teacher_share` trends down.
- Do not: call a complete receipt high-quality data, call deletion verified
  forgetting, or let the learner manufacture its own training authority.

### Phase 8: Resource, Cost, and Mac Acceleration Routing
- Resource/cost records cover plan nodes, training, verification, teacher calls,
  calibration, Hive, and assistant work. Costed routes include rejected alternatives,
  displaced verification/repair/human cost, authority, residuals, and rollback.
- Make MLX first-class on Apple Silicon, CPU/storage/operator explicit on Intel,
  and keep backend/device/throughput/memory/thermal/battery/checkpoint/load receipts.
  Move hot loops toward native Metal/MLX only with apples-to-apples correctness and
  useful-output-per-second evidence.
- Account for AR/MTP/speculative/self-draft/diffusion modes by accepted verified
  output and verification tax, not raw tokens/sec. Route scarce verification
  bandwidth by risk.
- Acceptance: Mac routes fail closed on unavailable backends; one hot-loop ablation
  improves accepted output per second without quality loss; report/checkpoint
  retention stays under policy.
- Do not: claim CUDA parity without reference reports; optimize throughput while
  verifier-passing output or human burden worsens.

### Phase 9: Hive Policy-First Distributed Operation
- Stable node identity, durable discovery/TTL/capability refresh, policy-first task
  routing, artifact sync, roaming profiles, and registered typed jobs share the same
  VIEA/SCF/authority/resource contracts as local work. No arbitrary remote shell.
- Scheduler decisions include backend/capability, memory, thermals, battery, disk,
  queue depth, latency, trust, stale/revoked grants, and partition behavior.
- Acceptance: a bounded task and artifact transfer replay across trusted nodes;
  revocation/partition races fail closed; local-only operation remains functional
  when peers are unavailable.
- Do not: expose raw port 8791 publicly, treat stale peers as live, or let product
  surfaces bypass task-kind and authority policy.

### Phase 10: Practical Neural Seed Survival Lane
(Owns Tracks 1, 2's learned proposer/repair loop, 3, and 4's model path.)
- Build the practical transformer/hybrid from-scratch generator first. The current
  wall is semantic transfer after correcting a severe data/capacity mismatch. The
  old 43.6M dense-active model saw only 0.73M target positions from 16,000 licensed
  functions before repeated training and remained 0/8 after adaptation. The
  replacement path now selects across the full admitted corpus (35,297 functions),
  removes exact source/body duplicates, exposes 5.58M encoded one-pass positions to
  a 3.70M-active-parameter tied-output model, and reports optimizer repetition
  separately. The governed conversation path adds 13,918 human-contributed,
  redacted, decontaminated rows (4.29M one-pass positions). Neither data scale nor
  LM loss clears the wall without direct verifier behavior. The completed one-pass
  contrastive run improved heldout LM loss `8.410877 -> 2.273732` and the matched
  versus mismatched source-loss gap `0.006521 -> 0.238046`. Frozen 1M/3M/5.5M
  replay emitted `57` integrity-verified, syntax-valid candidates over `24` private
  task-checkpoints but scored `0/24`; outputs collapsed onto shallow `return data`,
  `len(data)`, and repeated type-check patterns. This falsifies more scalar tuning
  or epochs on the same body-token target. A reversible byte-piece codec now makes
  source and target encoding open-vocabulary with zero `<unk>` positions and rejects
  incomplete targets before splitting. The matched semantic-plan-plus-direct-body
  arm improves heldout loss to `2.524909` but emits only one integrity-clean candidate
  and remains `0/24`; the zero-unknown direct-body arm reaches `2.809789`, emits 17
  integrity-clean candidates at its final rung, and also remains `0/24`. Its first
  replay exceeds the five-minute child budget. These results retain open-vocabulary
  encoding as correctness infrastructure but falsify both representations as the
  current survival model.
- The standard 6.6M-parameter MLX decoder follow-up exposed and removed a
  hidden-information leak: its earlier `1/24` result inferred callable arity
  from private tests. The corrected split-vocabulary route binds all admitted
  source content, uses a target-independent interface when no explicit
  signature exists, and encodes all `24/24` heldout targets with zero overflow.
  A bounded `2,001,802`-position continuation improves corrected heldout loss
  `2.212320 -> 1.667694`, emits `88` syntax-valid transformer-family candidates
  over `23` tasks with zero integrity mismatches, and still scores `0/24`.
  Independent integrity verifies `38` and rejects `50` as inert stubs. Wider beam emits `376`
  syntax-valid candidates at `0/24`; a shared-vocabulary arm also remains
  `0/24` while weakening prompt conditioning. The route is contained as an
  experimental successor. Search width, truncation, signature recovery,
  tokenizer sharing, and more same-distribution SFT are now falsified as the
  immediate repair; prompt-conditioned semantic learning and inert-body collapse
  are the active wall.
- A subsequent ID-space audit found that split source-vocabulary IDs were
  incorrectly shifted into the target embedding segment. The repaired mapping
  raises independently integrity-verified candidates from `38/88` to `57/87`
  and reduces inert stubs from `50` to `30`, but covers only `22/24` tasks and
  remains `0/24` on exact behavior. The completed training receipt was then
  overwritten by the old evaluation-only CLI behavior; the checkpoint and final
  heartbeat are retained, but the canonical gate records incomplete training
  provenance and no adoption. Evaluation-only now requires execute mode plus an
  explicit prior receipt, and candidate artifact paths are request-bound. This
  keeps the encoding correction while rejecting the checkpoint as capability.
- Exact-bound source-conditioning measurement across all `24` heldouts records
  matched loss `1.662704` versus deterministic wrong-source loss `1.766282`, a
  positive gap of `0.103578`. The checkpoint therefore uses source information,
  but still cannot turn it into correct behavior. The existing conditioning route
  now separates `preflight`, `measure`, and `train`; binds the exact base config,
  checkpoint hash, stage signature, arrays, metadata, and base report; preserves
  target-mask cardinality under source derangement; publishes checkpoints
  atomically; emits typed faults; and cannot overwrite canonical survival evidence.
  It correctly refuses training while the base completion receipt is incomplete.
  The pre-repair conditioned checkpoint is retained under an explicitly stale
  lineage path and is absent from the configured output location.
- The registered Semantic IR now has a closed, target-independent learned-plan
  protocol that the same causal model can emit before direct body tokens. The
  independent integrity path replays the raw token trace, validates every plan
  transition, reversibly decodes the body subsequence, normalizes only implicit
  terminal dedents, and demotes corrupt traces without trusting candidate flags.
  Under matched MLX data, model, seed, target-position, evaluation, and fanout
  budgets, plan-plus-body is a clean negative: `42` syntax-valid candidates over
  `15/24` tasks versus body-only `70` over `20/24`; mean verifier reward is
  `0.296970` versus `0.401639`; decode is `11.252` seconds slower; both score
  `0/24`. The plan mode is therefore `NOT_ADOPTED`, body-only remains canonical,
  and lower plan-arm LM loss receives no capability credit.
- Scale toward a 100M sparse specialist proposer with matched dense active-compute
  control, expert attribution, prompt/signature-only visibility, strict direct-body
  replay, and family-disjoint heldouts. Keep the old body-template inventory disabled
  for promotion claims.
- Once a proposer sometimes succeeds, run verifier-stage-aware DPO/IPO/ORPO/KTO/
  SimPO, then bounded GRPO/RLOO/ReMax/RLVR. Every update has a policy lease,
  evaluator independence, drift/authority bounds, rollback, reward-hacking probes,
  and pre/post behavior replay.
- Compare one-shot versus verifier-guided semantic-IR repair, then AR versus MTP,
  self-draft/speculative, and sketch/diffusion modes using accepted verified output
  per second. VCM/STS on/off ablations use equal candidate/context budgets.
- Acceptance: direct model-only verifier pass moves above zero and beats its matched
  baseline; preference/RL/search/fast-gen techniques advance only on heldout
  behavior or record falsification; active-parameter efficiency is reported.
- Do not: add another marker/head/slot variant without a preregistered falsification
  sequence; train on public benchmarks; credit renderer/tool/search/fallback output
  as learned generation; scale or repeatedly adapt a behavior-flat checkpoint; call
  repeated token positions unique data exposure.

### Phase 11: SymLiquid Discovery-Lane Verdict
- Keep SymLiquid/CGS/VSA protected as a discovery comparator, but make the practical
  transformer/hybrid lane the default unless SymLiquid wins repeated matched-data,
  matched-active-compute, matched-search/verifier experiments.
- Run across multiple seeds and at least two capacity rungs; report mean/spread,
  throughput, memory, verifier behavior, unique residuals, scaling trend, and exact
  regimes where either substrate wins. Freeze or narrow a losing lever rather than
  keeping it alive through bespoke metrics.
- Acceptance: a substrate-adoption record reaches a reproducible verdict or names
  the evidence still required; the practical assistant is never blocked by the
  discovery lane.
- Do not: bury the control result, move the goalposts, or treat architecture elegance
  as capability evidence.

### Phase 12: Public Calibration and Residual-Mining Discipline
- Keep public surfaces registered, decontaminated, frozen by exact surface/seed, and
  consumed after use. Fresh surfaces are available when measurement is useful; only
  exact reruns, contamination, and fishing are blocked.
- Public reports separate model-only, search-guided, tool-assisted, deterministic,
  and baseline paths. No public prompt/test/solution/trace/template or derived answer
  metadata enters training, retrieval-for-generation, teacher rows, or curriculum.
- Mine one calibration into private residual categories and a private-only repair
  manifest. Data-admission receipts record the public exclusion/taint boundary.
- Acceptance: per-card results, integrity receipts, contamination checks, consumed
  registry entry, negatives, and private residual plan replay; no public payload is
  admitted to training.
- Do not: lock measurement by calendar/budget, rerun an exact consumed surface,
  hunt lucky seeds, or train on benchmark artifacts.

### Phase 13: Semantic IR and Substrate-Neutral Reasoning Atoms
- Make semantic atoms the executable bridge among planner nodes, candidate
  generation, verifier feedback, VCM pages, code concepts, and localized repair.
  Each atom carries intent, typed inputs/outputs, constraints, dependencies,
  authority, validator, target, repair scope, and residual lineage.
- The strict generator must consume action-aware executable state objects/spans to
  construct traversal, update, finalizer, value, and return bodies; failed verifier
  obligations localize repair to dependent atoms instead of rerolling whole bodies.
- Acceptance: semantic-IR-localized repair or construction improves heldout direct
  model behavior under an ablation; missing/failed atoms produce typed residuals and
  dependent-node replay.
- Do not: count IR rendered by a deterministic compiler as learned generation or
  add another auxiliary target whose only win is its own loss.
- Current status: `partial`. The generic AST IR, obligations, localized assisted
  repair, and independently audited learned plan/body protocol are integrated,
  and the standard transformer has both an opt-in source-only obligation head and
  a compact ordered, alpha-renamed plan language covering operation, semantic
  role, control flow, abstract dataflow, value kind, and feature obligations.
  The ordered four-arm MLX ablation reserves identical plan space and trains all
  arms on the same `1,202,267` direct body positions. Semantic plans are causally
  learned: teacher-forced plan loss is `0.930240`, versus `1.558662` shuffled and
  `10.722825` dropout, and body loss is best at `1.787126` versus `1.808397`
  body-only. This does not transfer to behavior. Content-bound hierarchical
  replay yields `40` candidates over `17/24` tasks with reward `0.294737`, versus
  body-only `83` over `23/24` at `0.440152`; every arm remains `0/24` exact.
  Independent integrity classifies every emitted candidate as transformer/hybrid
  with zero mismatches, and the gate records `GREEN` evidence but `NOT_ADOPTED`.
  A follow-up rank-64 latent ordered field keeps the target stream body-only and
  consumes exactly `1,202,267` body positions in every arm. It also fails
  causally: semantic labels improve plan F1 (`0.449198` vs `0.253545` shuffled),
  but semantic and shuffled emit the same `76` candidates, cover `23/24`, score
  reward `0.425600`, and pass `0/24`; dropout exactly reproduces body-only's `83`,
  `23/24`, `0.440152`, and `0/24`. The gate records `GREEN` evidence and
  `NOT_ADOPTED`. Global additive plan superposition is therefore retired as the
  next lever. The slot-addressable follow-up adds a separate cross-attention read
  over `16` predicted plan slots at every decoder layer while preserving the same
  direct body target and `1,202,267` body positions in each arm. Semantic plan F1
  improves to `0.449339` versus `0.261993` shuffled and `0.0` dropout; semantic
  coverage/reward is `24/24` and `0.441538`, versus shuffled `23/24` and
  `0.436154`. It still passes `0/24`, fails to beat dropout reward `0.442105`, and
  costs `35,705 ms` versus body-only `23,325 ms`. Its integrity and blind-flow
  audits are GREEN, while the gate records `NOT_ADOPTED`. Phase completion now
  requires prompt-to-plan semantic grounding and independently measured
  obligation adequacy to improve exact family-disjoint behavior under the same
  controls; plan loss, deterministic rendering, repair, generic extra capacity,
  another attention mechanism, or behavior-flat scaling cannot substitute.

### Phase 14: Compression, Proof, and Claim-Evidence Records
- First-class claim/evidence transitions preserve contradiction, downgrade, split,
  merge, retirement, and replacement history. `docs/PROJECT_STATE.md` becomes a
  compact ledger view over checkpoints/reports/configs/curricula/verifier/dogfood.
- Build an epistemic trusted computing base: independent verifiers/auditors,
  bounded trust propagation, trap fixtures, randomized deep replay, and
  receipt-faithfulness checks so a report or receipt is not trusted because it
  exists.
- Compression receipts declare reconstruction, residual coding, determinism,
  supported use, authority/taint preservation, and exact-replay boundary. Proof and
  Lean artifacts support only their finite stated claims.
- Acceptance: one live claim undergoes evidence-backed belief revision; stale or
  contradicted evidence blocks route/promotion; a compressed artifact cannot replace
  source evidence without its declared-use replay gate.
- Do not: overwrite negatives, elevate authority through summaries/compression, or
  treat structural proof as model-quality evidence.
- Current implementation: the canonical evidence store emits 91/91 valid public-safe
  packs, an eight-root TCB with five rotated primary/shadow assignments, a 1,532-claim
  index spanning 43 emitting families and 9,166 dependent-surface edges, thirteen real
  evidence-driven downgrades, 64/64 randomized deep receipt replays, and 7/7 rejected
  adversarial traps. Exact snapshot migration reclaimed 866 MB from SQLite. Reference-
  aware checkpoint retention archived more than 13 GiB behind replayable pointers,
  hard-linked ten exact duplicate payloads, restored 36 missing historical pointers
  after archive-hash verification, archived 115 superseded hot-report views, replayed
  1,361/1,361 cumulative entries, and leaves
  the artifact budget GREEN with zero hard gaps and zero warnings. These are evidence-system results, not model-
  capability or checkpoint-quality claims.

### Phase 15: Procedural Memory and Toolification
- Cluster repeated verified trajectories into procedural candidates only when they
  carry source traces, pre/postconditions, authority, verifier receipts, monitoring,
  drift/retirement, rollback, and route-binding contracts.
- Promote through SCF replacement transactions. Feed verified procedures into
  assistant usefulness and optional trie/lookahead decode sources while preserving
  explicit noncredit for learned generation.
- Acceptance: more than one diverse real workflow becomes a monitored, replayable
  tool and survives postcondition drift; stale procedures retire automatically.
- Do not: toolify one-off traces, let procedure reuse widen authority, or call a
  stored trajectory parametric learning.
- Implemented evidence: three diverse schema-bound real workflows (planning,
  chat, and deterministic-tool assistance) pass replay and canary execution;
  each compiles to a digest-bound, noncredit lookahead asset. Exact lookup
  selects `3/3`, unknown/ambiguous routes abstain, and append-only lifecycle
  receipts automatically retire stale or postcondition-drifted procedures.
  Three SCF replacement transactions arm rollback guards, the plan compiler
  consumes exact asset/receipt IDs, and assistant runtime selects all three
  bindings without widening authority.

### Phase 16: MoECOT and Octopus Router Integration
- Emit `RoutingDecisionRecord` for each specialist/tool/model route and
  `MoECOTOrchestrationRecord` for each multi-core run: capability request, candidate
  fields, readiness, authority, active parameters, cost/quality, context adequacy,
  verification burden, selection/rejection, fallback, residuals, and ledger refs.
- Tie sparse experts and Octopus arms to registry implementations/SCF route validity;
  learn routing only from governed traces and attribute activation to action families.
- Acceptance: diverse real traces support a replayable route policy that explains
  why alternatives lost; stale/unqualified experts are ineligible; sparse routing
  beats a matched active-parameter dense control before default adoption.
- Do not: let routing hide weak candidate generation or count expert/tool selection
  as learned answer generation.
- Implemented evidence: the registry-gated deterministic bootloader now passes
  `14/14` source-defined routing cases with exact arm sets, token-boundary keyword
  matching, specialist-over-generic subsumption, fail-closed stale/unqualified
  routes, and an explicit live-hardware veto for simulation-only arms. Eighteen
  schema-bound completed-work traces are source-disjoint from holdout and carry
  permission/no-cheat receipts. The learned sparse head is a valid negative
  (`exact=0.375`, contrastive rejection `0.1667`) and remains `NOT_ADOPTED`; the
  matched sparse proposer also remains `NOT_ADOPTED` because dense control won.
  The independent router-head integrity gate is GREEN without converting those
  losses into route authorization. Fanout, verifier, STS, VCM, verification-
  bandwidth, governance-tax, VIEA, and plan-compiler contracts all replay; a live
  fanout refresh correctly emits a typed stale-checkpoint deferral rather than
  running an obsolete model. New learned/default routing evidence now belongs to
  the Phase-10 replacement proposer, not to more router scaffolding.

### Phase 17: Simulation, Fidelity, and World-Model Contracts
- Simulation contract records for toy environments/private fixtures/proxy
  benchmarks/Circle finite fixtures/resource sims/synthetic curricula/training
  smokes (scope, fidelity, temporal semantics, demand, bottlenecks, approximation
  liberties, supported claim boundary, residuals, evidence refs); map-territory
  labels on reports (toy_fixture/proxy/private_heldout/public_calibration/
  runtime_dogfood/production_route); public/runtime claims must cite why a toy/proxy
  transfers or state it does not.
- Acceptance: existing private/synthetic reports classify by fidelity; a toy/sim
  result cannot support a public-transfer/production claim without a transfer
  witness; negative proxy results stay visible.
- Do not: treat private synthetic 1.0s as broad capability; treat proofs/fixtures as
  model-quality evidence.

### Phase 18: Governance Rights, Constitutional Predicates, and Failure Boundaries
- Compile normative sources into testable constitutional predicates with protected
  scope, uncertainty, amendment rule, review route, self-modification boundary, and
  non-claims. High-impact plans emit agency-rights/value-conflict records and preserve
  audit, exit, fork, dissent, contestability, export, deletion, and route refusal.
- Record every read/transform/disclose/write/execute/approve/train/route/publish
  authority transition. Digital SCIF execution exposes handles and permitted use to
  the model while privileged adapters substitute secrets outside model-visible text.
- Model distributed revocation, stale grants, partitions, confused-deputy attacks,
  amendment legitimacy, and human oversight degradation/rubber-stamping.
- Acceptance: one real side-effecting task emits authority/adapter/effect/rollback
  receipts; secret and confused-deputy fixtures fail correctly; self-improvement
  cannot weaken protected predicates; user governance rights are inspectable.
- Do not: put secrets/raw private text in model-visible context, infer authority from
  locality, or use values prose instead of enforceable predicates.

### Phase 19: Book-to-Theseus Backlog and Evidence Synchronization
- Maintain a unified research backlog tying each AI_book source/gap to Theseus
  implementation state: access state, source-note state, claim-mapping state,
  proof/test backlog, insertion decision, residuals, next action.
- Keep `configs/roadmap_implementation_matrix.json` the source of truth for chapter
  coverage; reconcile it first when this file and the matrix disagree; refresh the
  crosswalk against the current book tree; register new book concepts as backlog
  rows routed to an existing phase (see Book-Parity Backlog).
- Acceptance: `roadmap_implementation_gate.py --gate` reports full chapter coverage
  with 0 stale phases; every book mechanism represented in Theseus meets the
  Book-Parity Invariants below; public-safe Theseus evidence can flow back into the
  book without importing private payloads or inflating support state.
- Do not: hide matrix drift in narrative; create a capability claim from a backlog
  row.

## Book-Parity Backlog (remaining coverage)

Every load-bearing ASI Stack mechanism should eventually be represented in Theseus
(see Book-Parity Invariants for what "represented" requires). Current remaining
gaps route into the phases above:

| Book mechanism | Remaining gap | Phase |
|---|---|---|
| Stable Capability Fields | evidence for live implementations stale; routing eligibility mostly blocked | 2 |
| Replacement + rollback | replacement transactions not yet the ordinary route for model/router/tool changes | 2 |
| Recursive self-improvement boundaries | canonical transition record not yet universal across policy, data, route, and model changes | 2, 14 |
| Intent-to-execution contracts | runtime path does not yet require contracts for every meaningful task | 1 |
| PlanForge DAGs + adequacy + arbitrage | not the default execution spine; adequacy contracts + arbitrage ledger missing | 1 |
| Cognitive compilation / semantic IR | failures reported more than repaired through IR-level localized feedback | 13 |
| Data engines + continual learning + unlearning | admission is file-level metadata; candidate receipts, policy comparison, semantic leakage, lineage closure, and descendant deletion proof are missing | 3, 7, 12, 14 |
| Verification bandwidth | verification not yet budgeted/routed as a scarce resource | 8 |
| Claim ledgers + belief revision | claim/transition records + contradiction links not first-class per run | 14 |
| Proof-carrying + tribunal/adversarial review | broader independent-review records not standard for architecture changes | 14, 18 |
| Labor OS typed jobs / artifact graphs | typed job + unified artifact graph not the universal unit | 1, 14 |
| Procedural memory | adoption discipline applied to only one trace so far | 15 |
| Routing heads + MoECOT + specialist cores | one canonical live MoECOT orchestration record missing; sparse specialist model unbuilt | 16 |
| Readiness gates + residual escrow | stale outputs/duplicate families reduce trust | 0, 2 |
| Generate-verify-repair | correctness still eval-only, not a training/search signal | 6, 10, 13 |
| Deterministic math/search substrates | not yet attached universally to planning/VCM/claim-ledger/assistant | 6 |
| Benchmark ratchets | public-transfer story stale; remeasure only after private correctness improves | 12 |
| Policy optimization (DPO/GRPO/RLVR) | not yet the generator's central learning mechanism | 7, 10 |
| Integrated reference architecture | the missing canonical spine | 1 |
| Authority kernel / SCIF / failure boundaries / adapter receipts | no universal authority-transition ledger or SCIF receipt layer | 18 |
| Constitutional predicates / agency / value conflicts / governance rights | not first-class records consumed by planning and self-improvement | 18 |
| Resource budgets + costed routes + generation-mode records | costed routes + accepted-output accounting not the normal artifact | 8 |
| Simulation contracts | no contract separating map from territory | 17 |
| Semantic atoms/nodes + compact artifacts + substrate adoption | not the common substrate; adoption discipline not universal | 11, 13, 14 |
| Research backlog | no unified source-gap-to-implementation backlog | 19 |
| Receipt faithfulness / record-reality gap | effect receipts, gate reports, ledger writes not yet trap-tested or randomly re-audited | 14, 18 |
| Epistemic trusted computing base | roots of trust, verifier independence, bounded trust propagation not yet explicit | 14, 18 |
| Human oversight degradation | no measure/guard for review-capacity capture and rubber-stamping | 8, 18 |
| Partition / distributed authority governance | stale-grant/revocation-race behavior across Hive nodes not modeled in runtime | 9, 18 |
| Interpretability evidence discipline | interpretability claims not held to the same support-state/non-claim bar | 13, 14 |
| Amendment legitimacy | governance/charter changes not treated as first-class amendment events with legitimacy checks | 18 |

## Book-Futures Intake (provisional, phase-routed)

The ASI Stack's `2026-07-09 Controlled ASI Completeness Intake` names future
chapter candidates and section-level obligations that are not all in the active
46-chapter manifest. Theseus tracks them now so the architecture does not discover
the omissions after deployment. They are **not new active lanes**: each is owned by
existing phases, remains planning-only until its book source/ownership gate passes,
and activates only at the entry condition below. Except for work that directly
governs the current data/update path, these rows do not block bounded private
training; they block the higher-authority operation they govern.

| Book candidate or section | Theseus owner phases | Entry condition / disposition |
|---|---|---|
| Governed Deliberation and Test-Time Scaling | 6, 8, 10, 13 | after direct model-only pass is nonzero; own think/search budgets, verifier selection, stopping, diminishing returns, and residuals |
| Data Engines, Continual Learning, and Unlearning | 3, 7, 12, 14 | admitted to the local 47-chapter manifest; current Phase-7 receipt/lifecycle work |
| Scalable Oversight and Adversarial AI Control | 4, 14, 18 | before high-impact or self-improving routes; weak-supervisor limits, trusted/untrusted monitors, collusion resistance, oversight recursion |
| Adversarial Evaluation, Sandbagging, and Training-Time Deception | 4, 10, 12, 14, 18 | admitted to the local 47-chapter manifest; before promotion/public claims, bind elicitation/selection/monitor context and cross-context discrepancies to existing evidence and promotion routes |
| Capability Thresholds and Deployment Commitments | 2, 8, 12, 18 | before production serving or authority growth; if-then safeguards, autonomy/time horizon, automated-R&D share, open-weight decisions |
| Reasoning-Trace Faithfulness (section route) | 1, 4, 14 | now as an evidence boundary; distinguish traces from receipts and test trace/action consistency without treating hidden reasoning as authoritative |
| Inter-Stack Protocols, Identity, and Economic Exchange | 1, 2, 9, 18 | before cross-Hive/external-agent federation; protocol, identity, credential, payment, authority, and receipt interoperability |
| World Models and Model-Based Cognition | 10, 13, 17 | after the core proposer floor; prediction/error ledgers, model-predictive control, imagination/search, causal limits, sim-to-real |
| Safety Cases and Structured Assurance | 14, 17, 18 | before production/high-impact routes; claims, arguments, assumptions, hazards, defeaters, context, and invalidation lifecycle |
| Model-Weight Custody and Hardware Roots of Trust | 0, 8, 18 | before distributing valuable weights or production loading; exfiltration, key release, attestation, TEE/GPU limits, open-weight custody |
| Multi-Agent Systemic Risk and Agent Economies | 9, 16, 18 | before multi-agent economic/autonomous operation; collusion, cascades, miscoordination, market behavior, gradual disempowerment |
| Persuasion, Epistemic Security, and Human Agency | 5, 14, 18 | source-contingent; require stronger empirical anchors before activation or claims |
| Open-Ended Improvement Engines | 7, 10, 12, 14, 15, 18 | admitted to the local 47-chapter manifest; implement the bounded campaign/admission contract now, then run empirical campaigns only after a verifier-positive proposer exists |
| CAIS, Embedded Agency, and Corrigibility Foundations (section route) | 1, 18, 19 | prior-art/foundations work only unless a distinct mechanism and proof program earns a chapter boundary |
| Perception, Grounding, and Observation Trust | 3, 8, 17, 18 | when a vision/audio/sensor modality starts; observation contracts, provenance/taint, fusion disagreement, uncertainty, refresh, spoofing/occlusion |
| Embodied Execution and Physical-World Safety | 1, 8, 17, 18 | before physical actuation; control periods, hazard envelopes, interlocks, fallback controllers, stop authority, degraded modes |
| Operational Resilience, Incident Response, and Decommissioning | 0, 8, 9, 14, 18 | before production/fleet dependence; criticality, impact tolerance, monitoring, containment, failover, recovery, disclosure, secure retirement |
| AI Supply-Chain Integrity and Lifecycle Provenance | 0, 7, 8, 14, 18 | before reproducible releases/training fleets; AIBOM, signed build/training attestations, dependency/data integrity, supplier trust, revocation |
| Privacy, Data Rights, and Multi-Agent Memory | 3, 5, 7, 9, 18 | before multi-user or shared-memory production; purpose limitation, consent, minimization, access/export/correction, privacy budgets, leakage, deletion closure |
| Autonomous Replication, Proliferation, and Containment | 2, 8, 9, 14, 18 | before any replication/provisioning authority; descendant identity, principal binding, resource acquisition, persistence, containment, shutdown tests |
| Scientific Discovery and Experimental Governance | 1, 6, 7, 14, 17, 18 | when Theseus begins autonomous experiments; hypothesis provenance, preregistration, instrument authority, statistics/causality limits, replication, dual-use review |

Cross-cutting Tier-2 obligations are folded into those owners rather than becoming
more systems: calibrated uncertainty/selective prediction/abstention and
trace-faithfulness (Phases 4/14); prompt-injection limits, autonomous-insider and
collusion probes, randomized monitoring, and operator-load residuals (Phase 18);
OOD/active perception plus causal/counterfactual/identifiability boundaries
(Phases 13/17); temporal-logic monitors, interlocks, degraded operation, and
sim-to-real residuals (Phases 17/18); AIBOM/reproducible-build/derivative-revocation
plus privacy budgets and cross-user leakage (Phases 0/3/7/14/18); and objective
fingerprints, periodic recontracting, time-horizon/automated-R&D measures,
dual-use profiles, provider concentration, portability, and shutdown handoff
(Phases 2/8/9/18).

## Quality Bars (definition of "done")

### Book-Parity Invariants
A book mechanism counts as represented in Theseus only when all hold:
- registry-owned abstraction or explicit binding to one; no loose sidecar
  implementations;
- stable implementation record, authority boundary, typed failure behavior, and
  replacement/rollback path where replacement is possible;
- emits VIEA-compatible artifacts (command/intent contract where applicable,
  artifact-graph refs, claim/support-state refs, receipts, residuals, non-claims);
- has a gate/fixture that rejects at least one malformed or overclaimed case, not
  just a happy-path smoke;
- records whether evidence is live-run, fixture-only, import-only, static-digest, or
  blocked by public-safety/private-artifact constraints;
- if the book can cite it, emits a public-safe evidence pack with digests and exact
  non-claims (private payloads, benchmark payloads, secrets, teacher-private
  material stay out);
- if it changes `AGENTS.md`, registry policy, capability semantics, authority,
  routing, training-data or benchmark admission, or publication claims, it is a
  governance/amendment event, not an ordinary refactor.

### Module Definition of Done
Each major surface carries a registry module card: problem, owner field, interface,
invariants, failure modes, minimal implementation, validation commands, evidence
refs, non-claims, deprecation route, and source crosswalk. An ATT-D rule flags new
major modules without a module card or registry mapping. A report-family cap and
compaction policy is tied to module cards so evidence does not become endless
duplicate report families.

Implementation horizons: each module card names both a minimal implementation and a
beyond-SOTA mature endpoint (per the Implementation Standard), cites the SOTA
baseline the surface meets or beats, and marks the surface "done" only at its mature
endpoint - never at the MVP.

## Success Definition

Theseus is on track when all of this is true:
- The repo has one canonical control spine and every serious task passes through it.
- The registry answers what Theseus is, what each capability means, what backs it,
  and what can replace it.
- The assistant is useful in daily work and logs real accepted/missed/ignored/
  corrected/completed outcomes.
- The practical generator improves on verifier-passing private family-disjoint
  heldouts without cheating, and the capability-per-active-parameter scoreboard
  shows a real, measured trajectory (or a recorded falsification).
- Public benchmark calibration stays measurement-only and becomes more informative,
  not more gamed.
- Deterministic tools, VCM, STS, planning, verifier-guided search, and routers are
  integrated as one system, not separate proof islands.
- SymLiquid remains a protected research comparator; the practical lane uses the
  best evidence-backed architecture.
- The book can point to Theseus as an implementation reference with honest support
  states, not inflated claims.
