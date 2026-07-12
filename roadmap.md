# Project Theseus Roadmap

Consolidated 2026-07-07; reconciled with the 2026-07-11 AI Stack manifest and
post-v2.1 evidence cycle. This
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
| Data engine + curriculum | Track 0 / Phase 7 | quality-filtered receipt, exact 132.46M-position stage, and seven-target MLX checkpoint smoke are `GREEN` | run bounded complete arm/control pretraining; licensed English + requested code families remain primary and teacher rows remain sparse residual pressure |
| Dense transformer control | Phase 10 | canonical matched falsification control; architecture clean, behavior flat | train from the same 132.46M-position view as the five-arm system and keep it non-default unless it wins the governed comparison |
| MoECOT language-specialist seed | Track 1 / Phases 10, 16 | five independent MLX arms plus total- and active-parameter dense controls pass checkpoint/optimizer/resume smoke; capability unevaluated | complete bounded pretraining, add arm-specific supervised/eval contracts, then run the frozen two-view architecture comparison before routing adoption |
| Verifier-guided search | Track 2 / Phases 6, 10 | architecture wired, amplifier waiting for signal | preserve the bounded kernel and replay contract; qualify it only after one-shot generation sometimes succeeds and search materially increases held-out pass |
| Correctness training (DPO->GRPO/RLVR) | Track 3 / Phase 10 | premature at the current zero-pass floor | build verifier-positive pair density from a behavior-positive proposer, then run matched reward-present/reward-removed ablations |
| Fast-gen modes (MTP/diffusion/self-draft) | Track 4 / Phases 8, 10 | deferred | optimize accepted verified output per second only after accepted output is non-zero |
| Generator capability (held-out pass) | Phase 10 | RED - current clean MLX route remains 0/24; prior monolithic interventions did not produce behavior | complete per-arm architecture and run the frozen MoECOT-versus-dense control; the next accepted intervention must improve direct model-only behavior, not routing, loss, syntax, candidate count, or assisted output alone |
| Self-improvement flywheel | Tracks 0, 3 / Phases 7, 10 | architecture available, capability loop not started | start generate->verify->admit->retrain only after the proposer produces independently verified successes |
| VCM ABI + transactions/certificates | Phase 3 | wired: ABI, stable semantic objects, typed temporal relations, hybrid retrieval, lifecycle transactions, compaction, and fresh-process ontology migration | consume lifecycle records in Phase 7/10; keep dense embedding, parametric unlearning, and public-memory capability claims separate |
| Claim ledger + belief revision | Phase 14 | ledger implemented; assurance/evaluation-integrity consumption partial | compile one live assurance graph and cross-context integrity record into route decisions |
| Replacement transactions | Phase 2 | replayable-reference-backed for one bounded local route-authority effect | keep the independent effect audit and mutation controls green; require equivalent receipts for each new effect class |
| Procedural memory + toolification | Phase 15 | implemented; three real metadata workflows guarded | keep lifecycle receipts live; stale/drifted routes retire and exact lookahead abstains on ambiguity |
| Authority kernel / SCIF | Phase 18 | replayable-reference-backed for the same bounded local effect; SCIF and wider authority controls remain synthetic | preserve exact effect/rollback auditing and expand support only when another real effect class earns it |
| Assistant product | Phase 5 | assisted runtime wired; model-only general chat unavailable | dogfood deterministic/verified assistance now, but earn model usefulness only from real multi-day use after the local model clears its behavior floor |
| Report hygiene | Phases 0, 8, 14 | budget gate GREEN but hot reports are near the 1 GiB ceiling and total local generated/archive state remains large | compact to <=80% hot-budget headroom after the active stage build; run retention before long training and preserve exact replay pointers |
| Book crosswalk / parity | Phase 19 | 54 chapters mapped in manifest order; no beyond-SOTA completion implied | keep book-owned fields, tests, and source identity checksum-bound |
| Book test obligations | Phase 19 + routed owners | 504 authored Codex tests; 102 remain planned or partial in the book | close by mechanism family with real controls, not checkbox fixtures |
| Book futures intake | Phase 19 + routed owners | requirements retained but capability-frozen; QCSA source is available and manifest pending | preserve the crosswalk without implementation expansion until a language arm or matched dense control clears non-zero heldout behavior, except concrete correctness/security defects |

Pre-training readiness uses the matrix-owned phase partition rather than every
unfinished roadmap item. Phases `0, 1, 2, 3, 4, 6, 8, 11, 14, 15, 18, 19` are
architecture prerequisites; phases `5, 7, 10, 12, 13, 16, 17` require training,
real-use time, public calibration, or behavior-positive candidates and therefore
cannot circularly block architecture readiness; Phase `9` remains an external
environment proof. The canonical gate now reports
`pre_training_architecture_ready=true` with zero blockers and warnings. The
architecture-first wave is closed. Architecture maintenance may fix a concrete
defect, but it is no longer allowed to displace data adequacy and direct learned
behavior as the flagship work.

Public-result interpretation is deliberately split by route. The historical
single-card `45/64` diagnostic is mostly a private n-gram body route (`44` of the
`45` passes) plus one full-body token pass, so it is not learned-model evidence.
The old five-card full-body route scored `7/320`, then a later surface scored
`1/320`; neither is the current MLX checkpoint. The current clean model-only wall
is `0/24` family-disjoint private behavior. These numbers are retained, but they
must never be plotted as one model's capability trend or used to hide the current
zero-pass floor.

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
- High-quality static open data is eligible regardless of which model/provider
  generated it when its license permits training and its provenance, quality tier,
  recursive-synthetic share, deduplication, contamination, and retention checks pass.
  Static corpus admission does not grant live-teacher authority.
- English is the only natural-language target in the current seed. The requested
  programming-language scope is Python, JS/TS, HTML/CSS, and Rust. Other human
  languages are excluded or quarantined rather than consuming model capacity.
- Licensed static corpora, verified self-generation, and dogfood outcomes are the
  primary learning pressure. Live OpenAI teacher data is residual-only: accepted-row
  share is capped at 10%, optimizer sampling at 2%, both trend down, and bulk teacher
  generation is forbidden.
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

Theseus tests the **Efficient ASI Hypothesis**: a from-scratch,
verifier-and-search-governed cognitive architecture that reaches
**frontier-competitive capability on verifiable domains (English + code) at 1-2
orders of magnitude fewer active parameters** than published dense baselines,
running fully local, with every claim falsifiable and no cheating. The intended
seed is MoECOT/Octopus modular intelligence: a lightweight governed head routes to
bounded language-specialist arms. A mixed dense transformer remains the mandatory
matched control, so modularity is still falsifiable rather than assumed superior.

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
and search. The target seed is a governed Octopus/MoECOT system with one English arm
and separate Python, JS/TS, HTML/CSS, and Rust arms. The head compiles visible user
intent, VCM context, authority, and resource constraints into a task-local route;
specialists exchange typed Semantic IR/artifacts rather than duplicating all domain
knowledge inside one monolith. A 6.6M mixed dense transformer is trained as the
matched control, not silently promoted as the intended product architecture.

- **Data-supported arms and matched control.** Predeclare unique admitted positions,
  optimizer exposure, heldouts, parameter count, and compute for each arm and for the
  mixed dense control. Reusing English/context rows across specialists is optimizer
  exposure, not additional unique-data credit. No arm is scaled beyond its data.
- **Language specialists are the intended seed, not an unmeasured assumption.** Each
  arm is a stable capability field with its own weights, data view, tokenizer/profile,
  checkpoint, readiness, authority ceiling, attribution, interference, residuals,
  split/merge/retire lifecycle, and independent rollback. The dense control must be
  capable of falsifying the modular thesis under matched total/active compute.
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

**Current execution order (binding for the next capability cycle):**
1. Freeze the shared ABI: visible-input boundary, byte-safe tokenizer base, VCM and
   Semantic-IR packet, arm registry/card schema, route lease, verifier, checkpoint,
   replay, composition, and no-credit contracts. Freeze five arms: English, Python,
   JS/TS, HTML/CSS, and Rust. Architecture changes then require a demonstrated defect.
2. Materialize one content-bound corpus into non-overlapping per-arm views plus the
   mixed dense-control view. Record arm-specific unique positions, shared-row reuse,
   tokenizer efficiency, contamination, license/provenance, and exact row ranges.
3. Select the smallest data-supported MLX model for each arm and a matched mixed
   dense control. Predeclare total parameters, active parameters per route, FLOPs,
   wall/energy budget, and heldouts. Train arms independently; do not allow one arm's
   optimizer state or residuals to mutate another except through an explicit update.
4. Evaluate arm behavior, router behavior, and composed behavior separately. Compare
   MoECOT versus dense at matched total parameters, matched active compute, matched
   data, and matched verifier/search budget. Route success never counts as answer
   success; dense wins remain valid evidence against the thesis.
5. After any direct arm behavior is positive, qualify STS/VCM conditioning, search,
   preference/RL, and fast generation in that order. Only after a materially changed
   current system clears private integrity/behavior gates, spend a fresh public
   calibration surface. SymLiquid remains a later protected comparator.

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
- Establish a data/model scaling contract before the next capability run. Count
  unique admitted model-visible positions separately from repeated optimizer
  exposure and choose a published compute-optimal planning band before training.
  For the current 6.6M-parameter dense model, the canonical audit now measures
  247.91M unique positions after file-level quality filtering under the actual
  tokenizer ABI: 77.35M English and 170.56M code across Python, JS/TS, HTML/CSS,
  Rust, and a small uncredited other-code tail. This clears the frozen 20:1 floor;
  the earlier 62.98M diagnostic accounting
  is superseded. The exact 132.46M-position training view remains selected and frozen
  before results rather than reverse-engineered afterward.
- Expand the corpus in the product's actual domains: English conversation and
  instruction following; Python first; then JS/TS/HTML/CSS and Rust. Measure
  executable completeness, dependency context, algorithm/contract coverage,
  natural-language quality, dialogue continuity, corrections/tool traces, and
  long-tail representation. Avoid bulk intake that merely increases tokens while
  leaving the current semantic residuals untouched.
- Natural-language scope is English. "Multilingual code" means Python, JS/TS,
  HTML/CSS, and Rust, not broad human-language intake. Every natural-language source
  carries an English filter/receipt; non-English rows quarantine.
- Prefer high-quality licensed human/open corpora and static openly licensed
  model-derived corpora over live teacher generation. Provider origin alone neither
  admits nor rejects a static dataset; license, provenance, quality, diversity,
  decontamination, and measured heldout utility decide.
- Code volume is quality-filtered after repository/license curation: generated,
  vendored, bundled/minified, decode-damaged, extreme-line, low-diversity,
  tokenizer-unrepresentable, and invalid-Python files receive no scaling or training
  credit. The frozen corpus floor must still pass after these exclusions.
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

**Track 1 - MoECOT Language Arms With Dense Falsification Control** (owned by Phases
10 and 16).
- Canonical seed topology: `english`, `python`, `javascript_typescript`, `html_css`,
  and `rust` arms. English owns conversation, instruction understanding, and
  user-facing synthesis. Code arms own direct generation/repair in their language.
  Cross-language tasks use typed sequential/parallel/adjudicated composition rather
  than a hidden generalist fallback.
- The head/router is a governed hybrid: deterministic capability/extension/signature
  evidence first, learned ambiguity ranking only after trace support, VCM for context,
  and route leases for authority/readiness/budget. It may abstain or compose arms.
- Each arm has separate weights, optimizer/checkpoint lineage, data receipt, context
  contract, verifier suite, capability scorecard, and lifecycle. A shared ABI is not
  shared weights. Arm updates are independent and cannot cause silent cross-arm drift.
- Train a mixed dense 6.6M control from the same canonical rows. Compare at matched
  total parameters, active parameters, data, optimizer positions, MLX wall/energy,
  and verifier/search budget. Report routing accuracy, answer utility, interference,
  transfer, and composition failure separately.
- Scale total specialist capacity only when every arm's own unique-data ratio and
  heldout curve support it. Shared/replayed English scaffolding is counted once as
  unique system data and once per optimizer exposure; it cannot inflate an arm's
  scaling receipt.
- Acceptance (hard to fake): at least one arm produces direct family-disjoint
  verifier-positive behavior; the router selects or composes the correct arm on
  source-disjoint ambiguous tasks; MoECOT beats or matches the dense control on
  useful verified output per active parameter/second without worsening tail arms;
  otherwise the dense result falsifies or narrows the modular thesis at that rung.

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

The first five-arm seed is rung one; 100M total sparse capacity is a later rung.
Define an explicit ladder (per-arm and total parameters, active params, expert count,
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

### Architecture Closure Outcome and Capability Pivot (2026-07-11)

The architecture-first wave has reached its stopping condition. The canonical
pre-training gate reports zero architecture blockers and warnings. Registry/SCF,
VIEA, VCM, candidate integrity, deterministic tools, bounded search, resource
routing, SymLiquid isolation, evidence/claim records, procedural memory,
authority/failure contracts, and book synchronization are all wired or implemented.
One local assistant canary also performed an actual route-authority file effect,
observed exact candidate bytes, and restored the prior identity with zero residuals;
A2/E1 still need independent support-state consumption of that receipt, but that is
evidence maintenance rather than a reason to postpone training.

The roadmap now pivots from architecture breadth to one capability flagship:
`C1_correctness_rl_and_generator_survival_lane`, beginning with Track-0 corpus
adequacy and the frozen dense transformer baseline. The following rules apply:

1. **No more architecture-by-analogy.** New book ideas map to existing fields and
   wait unless they expose a concrete missing contract or failure in the active run.
2. **Do not confuse route success with model success.** VCM, STS, tools, search,
   renderers, n-grams, and procedural routes can improve assisted usefulness, but
   direct learned behavior remains its own scoreboard.
3. **Do not optimize a zero numerator.** DPO/RL, search width, sparse routing, and
   fast generation wait until direct verifier-positive candidates exist.
4. **Treat data scale as a first-class experimental variable.** The next run freezes
   architecture and changes governed unique data/exposure, with a predeclared
   scaling contract and negative controls.
5. **Calibrate publicly after a material model change, not by calendar.** Fresh
   frozen surfaces remain available without arbitrary budgets, but an unchanged
   zero-pass model does not justify spending another public surface.

## Governed-Surface Work (Phases)

Each phase lists remaining deliverables, acceptance gates, and prohibitions. A
phase is complete only when its acceptance gates hold on a real (not fixture-only)
run and it meets the Quality Bars below.

### Phase 0: Repository Self-Model and Registry Discipline
- Keep the abstraction/implementation registry, SCF bindings, module cards,
  deprecation routes, ownership, evidence refs, and `docs/PROJECT_STATE.md` aligned
  with the current honest generator wall.
- Restore `docs/PROJECT_STATE.md` to a bounded current-wall document (target <=200
  lines). Move dated experiment narratives to the existing docs archive and keep
  only current architecture readiness, active flagship, current model/data metrics,
  latest comparable calibration, and next falsifiable action in the live page.
- ATT-D rejects unregistered major modules, duplicate default implementations,
  unowned report families, stale route evidence, and capability changes that omit
  rollback or substrate-adoption records.
- Route evidence is lifecycle-aware: each routable implementation names a minimal
  source-bound, TTL-bound, or current-invocation receipt; supporting and retained
  reports remain audit history but neither authorize nor expire a route.
- Enforce live report/checkpoint retention, current-reference-aware compaction, and
  one flagship claim record per metric family; generated bulk is runtime state, not
  source or capability evidence.
- Artifact consumers resolve both inline and retention-sidecar archive pointers,
  prefer live originals over stale sidecars, and verify reconstructed bytes against
  the owning receipt before a checkpoint can satisfy readiness. Missing or corrupt
  archive payloads fail closed rather than forcing needless retraining.
- Maintain an AI bill of materials spanning code, data, models, tokenizers,
  evaluators, tools, hardware/runtime profiles, build inputs, derived artifacts,
  signatures, advisories, and descendant invalidation. Requested, resolved, and
  observed identities remain distinct; reproducible build evidence and relocation
  tests are required before release claims.
- Acceptance: registry/hygiene/module-definition gates have no hard gaps, every
  active route resolves through one current implementation, and the repo can answer
  what Theseus is without scanning historical reports.
- Do not: turn a gate green with another `_vN` family; hide negatives; treat cleanup
  as capability progress.
- Current implementation evidence: the registry materializes a content-bound
  AIBOM with requested/resolved/observed identities, surface Merkle roots,
  derivative invalidation, relocation-stable build identity, and zero missing
  required local identities. Ed25519 attestations and advisory snapshots can
  gate Torch and MLX/NumPy checkpoint loads before deserialization; tamper,
  stale advisory, revoked key/artifact, generation rollback, bad ownership/mode,
  wrong purpose, and incomplete valuable-weight custody fail closed. The repo
  contains no private signing key. Real signed release/build/training/custody
  evidence remains unclaimed until a routeable trained artifact exists.

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
- Bind default model loads and valuable-weight movement to custody records: exact
  artifact/derivative identity, encrypted-storage reference, key-release policy,
  environment measurement, attestation freshness, access scope, revocation,
  anti-rollback, relocation, incident response, and open-weight irreversibility.
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
- Implement durable semantic memory inside VCM rather than as another sidecar:
  stable object identity; typed and temporal relations; ontology versions and
  migrations; provenance-preserving merge, supersession, retraction, and conflict;
  graph/vector hybrid navigation; hot/warm/cold consolidation; forgetting and
  compaction; restart consistency; and certified bounded subgraph snapshots.
- Acceptance: assistant answers "what is the current wall?" from VCM-backed state
  with source refs; generator train/eval records exactly what context was allowed;
  public rows are tainted calibration-only; mandatory misses fail closed; context-on
  improves one real internal task with no leakage/runtime regression; the real
  MLX/Metal serving path separately proves KV/prefix-cache lifecycle before parity
  is claimed.
- Current implementation evidence: one fail-closed packet ABI is independently
  validated across assistant, planner, verifier, deterministic-tool,
  training-admission, candidate-generation, and fanout paths; `45/45` canonical
  consumers and `37/37` planner nodes pass, while authority-widening,
  source-mutation, taint-drop, best-effort, fallback, missing, stale,
  contradictory, over-compressed, revoked, and deleted controls fail closed. A
  real tiny MLX-LM Llama forward creates, reuses, appends, and invalidates an
  `mlx_lm.models.cache.KVCache`; this supports only an exact `mlx_apple`
  lifecycle claim, not CUDA, custom Metal-kernel, or cross-backend parity. The
  canonical VCM graph now also embeds `256` stable semantic objects and `1,871`
  typed temporal relations under ontology `1.0.0`, with sparse-BM25 plus graph
  retrieval, explicit hot/warm/cold lifecycle records, retraction/quarantine
  suppression, certified bounded subgraph snapshots, and deterministic
  serialization/query replay. This is a durable semantic projection over VCM
  pages, not a dense-embedding, parametric-unlearning, or completed physical
  parametric-unlearning claim. Merge, supersession, retraction, and cold
  compaction now execute as typed transactions with rollback hashes and typed
  rejection; a fresh process migrates a persisted `0.9.0` ontology fixture to
  `1.0.0` and preserves state/query replay. The equal-budget private context
  ablation is `1.0` VCM-on versus `0.5625` VCM-off
  over 16 cases with zero fallback returns; it is integration evidence, not a
  broad public-capability claim. The separate `VCM-Governed` release profile
  remains RED because its quarantined public prompt calibration manifest is not
  currently available; Phase 3 implementation status does not override that
  capability/calibration result.
- Do not: keep proving VCM on tiny saturated benchmarks; store raw private user
  text in training rows by default.

#### Question-Compiled Semantic Addressing extension

The QCSA whitepaper (`qcsa_whitepaper`, v1.0, 2026-07-12) is available in the
AI_book source tree but is not yet an active `book_structure.json` chapter. Treat it
as a source-available, manifest-pending chapter candidate. It extends existing VCM,
SCF, PlanForge, VIEA, Semantic IR, tool, and Octopus contracts; it does not create a
new memory abstraction or bypass the active C1 capability flagship.

Implement it through the existing fields in this order:

1. **Identity/address/route type separation (Phases 0, 2, 3, 14).** Make an opaque
   stable Semantic Object ID independent of page revision, semantic address, label,
   embedding, atlas epoch, and physical location. Distinguish occurrence, type,
   instance, proposition, expression, memory, tool, policy, capability, and artifact
   obligation. Creation is provisional; merge, split, deprecation, alias, and
   revocation preserve lineage. Similarity, translation, or neighborhood never
   establishes identity by itself.
2. **Plural semantic address atlas and certificates (Phases 3, 13, 14).** Extend
   the existing VCM representation certificate into a `SemanticAddressCertificate`
   bound to SOID, occurrence/expression, context, task, consumer, atlas epoch,
   facet/version, weighted top-k variable-length paths, confidence/entropy/boundary,
   adequacy termination, cross-facet consistency, provenance, groundings,
   residuals, permitted/prohibited uses, authority ceiling, expiry/revalidation,
   migration refs, and digest/signature. Facets declare consumer, objective,
   source relations, arity/depth policy, stability budget, and prohibited uses.
3. **Question compiler (Phases 1, 3, 6, 8, 14).** Compile unresolved posteriors
   into internal discriminators, retrievals, verifier/tool calls, or human
   clarifications selected by decision-relevant information value minus compute,
   latency, privacy, interaction, and risk cost. Support soft candidates, batched
   questions, noisy answers, backtracking, `UNKNOWN`, and abstention. Persist the
   posterior change and evidence refs as a replayable question trace; never invent
   an intensional explanation for an operational-only latent split.
4. **Semantic-to-physical translation (Phases 1, 2, 3, 6, 8, 16, 18).** Compile
   SOID plus certificate into temporary VIEA/PlanForge route plans for VCM shards,
   retrieval indexes, specialists, tools, verifiers, compute tiers, decoders,
   approvals, fallback, and abstention. Physical remapping may not alter identity or
   atlas meaning. Semantic classification never grants authority; the translator
   must independently consume SCF, resource, privacy, and effect constraints.
5. **Atlas release and migration (Phases 0, 2, 3, 14, 18).** Publish atlas epochs
   with codebooks/topology, facet purposes, data/objective summaries, branch
   utilization/calibration, boundary examples, collisions, prohibited uses,
   migration maps, signatures, and rollback. Distinguish readdress from object
   merge/split. Shadow new epochs, dual-resolve supported old addresses, preserve
   SOID identity, reject silent misrouting, and retain explicit orphan/conflict
   faults and residual burden.
6. **Compositional/open-world semantics (Phases 3, 13, 14).** Represent novel
   compositions as typed expression DAGs/programs with roles, quantity, negation,
   modality, time, and causal operators before creating permanent leaves. Require
   semantic-to-surface-to-semantic round-trip checks for generation claims while
   keeping direct model-only behavior separate from certificate/tool assistance.

Testing must reuse existing VCM, planner, routing, evidence, and private capability
fixtures rather than create another benchmark lane. Preregister matched baselines:
flat IDs, random/frequency trees, one semantic tree, residual-quantized IDs,
multi-view IDs without stable identity, and full QCSA with active-question,
physical-translation, soft-routing, and migration-governance ablations. Cover:

- identity consistency, false merge/split, novel-object detection, correction
  persistence, and provenance completeness;
- atlas prefix utility, branch balance, entropy/path cost, boundary calibration,
  collision/ambiguity, perturbation robustness, and migration stability;
- question cost, decision error after each question, noisy-answer recovery,
  privacy/user burden, unnecessary clarification, and downstream utility;
- retrieval recall/ranking/latency/tail behavior, deletion, stale-address recovery,
  authority filtering, and stable-reference preservation after reindexing;
- expert/tool route quality at matched compute, paraphrase stability, permission
  violations, fallback, throughput, and replacement blast radius;
- semantic round-trip role/quantity/negation/time/citation accuracy, repair locality,
  latency, and direct-generation fallback frequency;
- old-address resolution, silent-misroute rate, migration coverage, rollback,
  unauthorized activation, residual preservation, and epoch operating cost.

Adopt only on a meaningful Pareto gain in task quality, matched-quality cost,
calibrated disambiguation, tail/generalization, migration blast radius, or governed
policy compliance. Falsify or narrow the mechanism if learned addresses do not beat
random/frequency controls, plural facets add only cost, questions do not beat direct
inference/simple clarification, identity indirection does not reduce migration
errors, certificates prevent no failures, or semantic-first generation loses
quality/latency without a compensating gain. Architecture coherence alone earns no
support-state promotion.

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
- Architecture status: the canonical bounded search kernel now enforces proposal,
  verifier-call, depth, branch, and wall-time budgets; independently recomputes
  candidate integrity; rejects hidden-test/solution/expected-answer feedback;
  records typed faults and deterministic replay hashes; and separates one-shot
  model, learned-repair search, deterministic repair, and tool-assisted claims.
  Plan compilation binds risk-tier budgets to every node (`37/37` contracts,
  `20` eligible in the representative compile), and the existing Semantic-IR
  repair runtime emits `2/2` replay-valid search receipts. Its private workload
  remains `0` behavior passes, so Track 2 is wired but not capability-qualified.
- Acceptance: one real private workflow improves with replayable tool evidence;
  search materially beats one-shot generation on frozen heldout work; failed tools
  produce `UNKNOWN`/`UNSOLVED`/`TOOL_FAULT`, never invented results.
- Do not: hide tool use inside model claims or let deterministic renderers receive
  learned-generation credit.

### Phase 7: Teacher and Data Governance
(Owns Track 0 and teacher-side learning authority.)

Admission-substrate status (2026-07-10): `implemented`; Phase 7 remains `partial`
under the 2026-07-11 book reconciliation until full-state update/unlearning
causality runs. The canonical admission path
now writes and replays `92,032` content-bound candidate receipts, admits
`91,725`, rejects `307` rows, and exposes zero detected exact/semantic
public overlap, fallback markers, or raw-user rows. All nine adversarial cases,
five frozen continual-policy simulations, and the 11-kind positive/negative
descendant-deletion fixture pass. Curriculum and survival-lane materialization
require admitted row hashes. The lifecycle surface remains `YELLOW`, not
`GREEN`, because recursive synthetic share is `0.824771`; that is a data-risk
warning, not proof of model collapse. The policy comparison is simulation, and
deletion closure is graph evidence, not physical unlearning.
The matched mixed dense control now owns a frozen pre-training scaling contract rather
than an informal token estimate. It selects the existing 6,623,232-active-parameter
MLX rung and requires 132,464,640 unique admitted model-visible positions (20:1),
with 60M English conversation/instruction, 60M code, a 12.46M tail reserve, and
explicit Python, JS/TS, HTML/CSS, and Rust minima. The canonical materializer now
measures 247,908,698 unique positions after file-level quality filtering under the
frozen tokenizer ABI: 77,351,958 English positions, including 18,947,648
conversation/instruction, and 170,556,740 code positions. Its receipt is `GREEN`
with zero integrity hard gaps and every total/domain/language/evidence minimum
satisfied. The stage now consumes the exact bound shards as 273,951 disk-backed
windows and exposes immutable arm views: 64.15M English, 38.08M Python, 14.08M
JS/TS, 8.08M HTML/CSS, and 8.08M Rust positions, plus the identical mixed
dense-control view. Independent arm trainer/checkpoint consumers remain pending.
The receipt proves all license/provenance, exact and
semantic deduplication, contamination, completeness, coverage/tail, synthetic-share,
lifecycle, tokenizer, domain, language, and repetition requirements.
- Replace file-level metadata admission with receipt-bound candidate lifecycle:
  provenance/authority, lineage, permitted use, split exclusions, exact + semantic
  contamination, retention, deletion scope, evaluation refs, residuals, bounded
  decision, and non-claims. Incomplete records block, quarantine, or remain
  experimental; admission never promotes capability.
- Govern teacher proposals/distillation with license, provenance, verifier,
  leakage, runtime-serving prohibition, and durable teacher-share ledgers. Track
  recursive synthetic share, coverage, diversity, tail loss, and provenance drift.
- Keep live teacher use sparse and causal: at most 10% accepted-row share and 2%
  optimizer sampling in bootstrap, decreasing toward zero as verified local data
  grows. Require a predeclared residual family and matched teacher-on/off utility
  test. The first Sol 5.6 code-row ablation was negative and remains quarantined;
  verifier acceptance alone never justifies reuse.
- Compare replacement, accumulation, replay, quarantine, and retraining on a frozen
  continual-learning workload. Propagate revocation/deletion obligations through
  every declared descendant and retain unverified closure states.
- Make update causality full-state: predeclare best-versus-final checkpoint
  authority and inventory model, optimizer, scheduler, RNG, caches, backups,
  indexes, checkpoints, descendants, and external effects. Measure target gain,
  forgetting, retained utility, lineage invalidation, deletion influence, privacy,
  and physical storage erasure as separate claims.
- Acceptance: the data-policy comparison and a negative descendant-deletion case
  replay independently; teacher rows enter only through their gate; public
  calibration payloads remain excluded; accepted-row and optimizer-sampling teacher
  shares both remain under their caps and trend down.
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
- **Current decision:** the practical seed is a five-arm MoECOT/Octopus topology,
  not one dense model containing all English and programming-language knowledge.
  Freeze English, Python, JS/TS, HTML/CSS, and Rust arm contracts with independent
  weights, optimizer/checkpoint lineage, data views, verifiers, and rollback. The
  existing clean 6.6M MLX causal transformer remains the mixed dense falsification
  control. No arm or control receives promotion without direct family-disjoint
  behavior; route correctness, loss, or syntax alone cannot adopt a checkpoint.
- **Trainer/checkpoint status:** `moecot_language_arm_training` now binds the five
  immutable arm ranges to independent 1,211,520-parameter MLX models and binds both
  a 6,623,232-parameter total-control and a 1,211,520-parameter active-control.
  All seven bounded smokes write distinct model and optimizer digests; Python also
  replays an exact content-bound resume. The five arms total 6,057,600 parameters,
  8.54% below the total-control. This is T2 training-substrate evidence only: every
  checkpoint is incomplete and carries `capability_claim=NOT_EVALUATED`.
- **Frozen comparison:** report equal-unique-position/total-parameter and
  equal-active-parameter/active-compute views together. MoECOT cannot be adopted by
  choosing the favorable accounting view after results are known; it must show
  nonzero direct behavior, avoid arm-floor regressions, and beat or Pareto-dominate
  both controls. A dense win is retained as falsification, not routed around.
- **Frozen scaling decision:** the selected rung is exactly 6,623,232 active
  parameters with a 132,464,640 unique-position floor and maximum 4x optimizer
  repetition. The canonical materializer credits 247,908,698 quality-filtered unique positions under
  the standard-transformer tokenizer/stage ABI and excludes stale source identities,
  tokenizer-unrepresentable documents, and exact/near duplicates from that credit.
  This measurement is corpus evidence, not training authority or capability.
  `standard_causal_transformer_survival` and its independent gate deny
  training/adoption until the content-bound
  canonical mixed-corpus receipt is GREEN. This prevents both epoch inflation and
  accidental training on an architecture whose data contract is still incomplete.
- The corpus is English natural language plus Python, JS/TS, HTML/CSS, and Rust.
  The earlier "multilingual" label refers to programming-language coverage, not a
  human-language expansion. High-quality appropriately licensed static datasets are
  admissible even when model-derived; origin is retained and synthetic share is
  measured rather than treated as an automatic ban.
- Static openly licensed model-derived corpora are eligible data sources when their
  quality tier, provenance, permitted use, contamination, retention, and recursive
  synthetic position share are explicit. They do not authorize a live teacher
  provider and cannot enter runtime serving as external tokens.
- Live teacher rows are not the bootstrap corpus. They are capped residual examples
  after ordinary licensed-data coverage is exhausted: <=10% of accepted rows and
  <=2% of optimizer sampling initially, with a downward schedule and matched utility
  ablation. The existing negative teacher tranche remains retained but quarantined.
- Training sequence is fixed by prerequisites: arm-specific causal pretraining plus
  a matched mixed dense control -> governed English instruction/conversation and
  language-specific direct-body SFT ->
  verifier-positive pair construction -> preference optimization -> verifier-reward
  RL -> search/fast-generation/sparse challengers. A later stage cannot be used to
  compensate for a missing earlier signal.
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
- A target-independent curriculum canary retains all `13,474` licensed rows but
  weights self-contained bodies `3.0x` and context-dependent bodies `0.5x` while
  freezing private sampling probability at `0.269068`. It improves heldout loss
  `1.808397 -> 1.706521` and emits `84` syntax-valid candidates over `23/24`
  tasks, but still passes `0/24`; mean verifier reward changes only
  `0.440152 -> 0.442105`. Integrity and blind-flow audits are GREEN with zero
  fallback/tool/template/router credit. This rejects standalone-body weighting as
  a capability fix while retaining the general sampling mechanism for future
  data-quality studies.
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
- The first architecture-qualified Sol 5.6 teacher curriculum ablation is also a
  clean negative. Matched MLX arms resumed the same frozen checkpoint with the
  same `1,024` train rows, `192` heldouts, seed, optimizer, four epochs, and
  direct-body auxiliary profile; the teacher-on arm reserved `14` governed rows
  to train only. Teacher-off reached heldout LM loss `0.352542`; teacher-on was
  worse at `0.371843`. On the same `24` family-disjoint plus `24` broad-private
  tasks, both arms scored `0/48`. Teacher-on emitted fewer independently verified
  learned candidates (`110` versus `135`), increased decode-starved tasks (`11`
  versus `2`), and lowered mean verifier reward on both splits. Candidate-integrity
  and blind-information-flow audits are GREEN with zero learned credit for the
  `48` baseline/fallback rows in each manifest. These rows remain admitted and
  provenance-bound, but this curriculum is `NOT_ADOPTED`; verifier acceptance is
  not a substitute for causal heldout utility.
- The same ablation exposed and removed a canonical decoder hot-loop defect.
  Operand and state-event bias recomputed generated-prefix context once per
  vocabulary token. Prefix context is now computed once per probability row and
  reused, preserving bit-identical probabilities; the focused benchmark improved
  `27.96x`. The ordinary MLX decoder now also uses checkpoint-compatible
  incremental source/cross/self-attention state with an explicit full-prefix
  control. Every materialized output head is numerically equivalent within
  `1e-5`, candidate/token identity is exact, and a real 48-token checkpoint
  canary improves pre-verifier decode from `832 ms` to `500 ms` (`1.664x`) while
  reducing source encodes from `47` to `1`. Decode progress is committed with
  fsync plus atomic replacement and bound to config, checkpoint, vocabulary,
  options, task order, and opaque model-visible-input hashes. A forced stop after
  two tasks resumed the third and reproduced all six candidate identities from
  a clean control. Corrupt, stale, symlinked, reordered, or hash-mismatched state
  fails closed. This is runtime/replay evidence only, not semantic capability or
  CUDA/MLX parity.
- A parameter-neutral prefix-LM control now tests whether full bidirectional source
  encoding helps the same direct body decoder. The canonical sequence contract has
  exactly one source/target separator on all `12,111` SFT rows and `24` eval rows,
  no separator on `17,643` raw-code windows, and no supervised position at or before
  the separator. Unit and MLX causal tests prove later-source influence, zero
  target-to-source leakage, cached/full-prefix equivalence, unchanged checkpoint
  schema, and fail-closed malformed partitions. Under matched seed `20260711`,
  parameters, `47` optimizer steps, `100,260` supervised positions, and `24`
  family-disjoint tasks, causal and prefix-LM both emit zero syntax-valid candidates
  and pass `0/24`; prefix-LM slightly worsens loss (`4.700036 -> 4.704584`), SFT
  throughput (`13,113.833 -> 12,843.229` tokens/s), and decode (`10,000 -> 10,065`
  ms). The comparison is `GREEN` but prefix-LM is `NOT_ADOPTED`; causal remains the
  canonical attention policy. This falsifies source-prefix masking as the next
  capability lever at the bounded diagnostic rung.
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
- Keep incompatible historical routes separate. The single-card `45/64` result is
  assisted mainly by private n-gram candidates (`44/45` passes) and is not a learned
  generator headline. The five-card `7/320` full-body result and later `1/320`
  result belong to older candidate generators, not the current clean MLX checkpoint.
  The next public report must name the exact current checkpoint, independent
  candidate family, model-only/search/tool channels, and private pre-calibration
  behavior; no cross-route trend line is permitted.
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
  audits are GREEN, while the gate records `NOT_ADOPTED`. A stricter follow-up
  adds a per-slot categorical objective with an implicit EMPTY class. Under the
  same `8,348,528` parameters and `1,202,267` body
  positions, semantic slot F1 beats shuffled (`0.422111` vs `0.344595`) and body
  loss improves (`1.781374` vs `1.790626`), but behavior is worse: semantic emits
  `71` candidates at reward `0.414167`, versus shuffled `78` at `0.429921`;
  dropout emits `70` at `0.411765`. All remain `0/24`, accepted verified output
  per second remains zero, and the canonical slot gate is `NOT_ADOPTED`. The
  next audit identifies a representation defect hidden by those aggregate
  metrics: the 16 token slots encode only `17` distinct plans for `24` heldout
  tasks and collapse seven unrelated algorithms onto the same truncated prefix.
  The replacement factorizes eight complete statement steps into presence,
  depth, kind, intent, flow, data roles, value kind, and feature groups. It cuts
  the field from `5,488` to `1,856` features, parameters from `8,348,528` to
  `7,880,000`, and staged arrays from `373,789,062` to `199,714,566` bytes. The
  independent gate verifies `24/24` unique, group-closed heldout plans with zero
  collisions. Behavior still regresses: semantic factorized plans emit `63`
  candidates over `21/24` tasks at reward `0.386842`, versus shuffled `74`,
  `23/24`, `0.421138`; dropout `82`, `23/24`, `0.438168`; and body-only `83`,
  `23/24`, `0.440152`. Every arm remains `0/24`, and the gate stays
  `NOT_ADOPTED`. Phase completion still requires a learned Semantic-IR route to
  improve exact family-disjoint behavior under the same controls. The immediate
  dependency is the Phase 10 direct proposer floor: another plan loss, renderer,
  deterministic repair, generic capacity increase, attention variant, or
  behavior-flat scale run cannot substitute.

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
- Compile structured assurance graphs from existing claims, evidence, assumptions,
  hazards, contexts, acceptance criteria, defeaters, mitigations, dependencies,
  owners, and review/expiry state. Changes invalidate affected nodes; a complete
  graph never becomes safety or deployment authority by itself.
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
- Make the first live specialist registry concrete: English, Python, JS/TS,
  HTML/CSS, and Rust. Every arm card binds its own model/config/tokenizer profile,
  corpus view, checkpoint/optimizer lineage, verifier families, VCM/context contract,
  readiness, authority ceiling, cost profile, residuals, and split/merge/retire rule.
  A shared ABI never implies shared weights or shared promotion.
- Support single-arm, sequential, parallel, verification, and adjudicated routes for
  cross-language work. Composition emits typed artifacts and verifier receipts;
  there is no hidden dense/generalist fallback and no route success credited as
  answer success.
- Emit `RoutingDecisionRecord` for each specialist/tool/model route and
  `MoECOTOrchestrationRecord` for each multi-core run: capability request, candidate
  fields, readiness, authority, active parameters, cost/quality, context adequacy,
  verification burden, selection/rejection, fallback, residuals, and ledger refs.
- Tie sparse experts and Octopus arms to registry implementations/SCF route validity;
  learn routing only from governed traces and attribute activation to action families.
- Calibrate on naturally ambiguous requests where specialist, generalist,
  clarification, fallback, and abstention can all be correct. Preserve request-
  feature provenance, persistent budget ownership, rejected alternatives,
  interference counterfactuals, outcome calibration, and the distinction between
  correct routing action and correct generated answer.
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
- Add a scalable-oversight protocol that separates proposer, trusted monitor,
  untrusted monitor, effect observer, evaluator, and promotion authority; measures
  correlation/collusion, weak-supervisor limits, random audit coverage, oversight
  latency, operator load, dissent, escalation, and abstention; and prevents review
  output from silently becoming training or execution authority.
- Add versioned capability-threshold commitments and safety-case consumption:
  comparable assessment identity, context, threshold, safeguard bundle, exception
  expiry, compensating controls, residual owner, and fail-closed route effect.
- Before cross-stack federation, require protocol/identity/exchange records for
  principal, credential, delegation chain, expiry/revocation, task/artifact schema,
  reserved resource/value budget, payment or accounting receipt, dispute path,
  residual, and shutdown handoff.
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
- Check exact chapter identity/order and every book-owned title, part, file, claim
  label, evidence level, minimal implementation, mature endpoint, interface,
  invariant, failure mode, and Codex-test inventory. Count equality alone is not
  synchronization. Preserve the book commit and manifest digest used for review.
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
| Effect-complete governed execution | useful release, first/final effect identity, observer separation, exact rollback inventory, and governance tax are not one ordinary assistant transaction | 1, 2, 5, 14, 18 |
| Recursive self-improvement boundaries | canonical transition record not yet universal across policy, data, route, and model changes | 2, 14 |
| Open-ended improvement campaigns | no canonical campaign admission, generator/evaluator separation, single-axis champion/challenger frontier, negative-knowledge archive, debt ceiling, or stop-authority transaction | 7, 10, 12, 14, 15, 18 |
| Intent-to-execution contracts | runtime path does not yet require contracts for every meaningful task | 1 |
| PlanForge DAGs + adequacy + arbitrage | not the default execution spine; adequacy contracts + arbitrage ledger missing | 1 |
| Cognitive compilation / semantic IR | failures reported more than repaired through IR-level localized feedback | 13 |
| Data engines + continual learning + unlearning | admission is file-level metadata; candidate receipts, policy comparison, semantic leakage, lineage closure, and descendant deletion proof are missing | 3, 7, 12, 14 |
| Full-state update causality | model/optimizer/scheduler/RNG/cache/backup/index/descendant inventory, prospective best/final authority, forgetting, influence, storage erasure, and exact rollback are not unified | 0, 2, 7, 10, 14 |
| Durable semantic memory inside VCM | wired: stable objects, typed/temporal relations, additive ontology migration, transactional merge/supersession/retraction/compaction, sparse-vector/graph retrieval, bounded snapshots, and fresh-process replay; dense embedding and parametric unlearning remain explicit non-claims | 3, 14, 15 |
| Question-Compiled Semantic Addressing | source available, chapter manifest pending; current VCM derives semantic object identity from address roots and lacks explicit SOID/address/physical-route indirection, plural atlas epochs, SACs, information-value question compilation, and matched migration/routing falsification | 0, 1, 2, 3, 6, 8, 13, 14, 16, 18, 19 |
| Verification bandwidth | verification not yet budgeted/routed as a scarce resource | 8 |
| Claim ledgers + belief revision | claim/transition records + contradiction links not first-class per run | 14 |
| Proof-carrying + tribunal/adversarial review | broader independent-review records not standard for architecture changes | 14, 18 |
| Labor OS typed jobs / artifact graphs | typed job + unified artifact graph not the universal unit | 1, 14 |
| Procedural memory | adoption discipline applied to only one trace so far | 15 |
| Routing heads + MoECOT + specialist cores | one canonical live MoECOT orchestration record missing; sparse specialist model unbuilt | 16 |
| Ambiguous routing + adaptive deliberation | naturally ambiguous route corpus, fallback/abstention calibration, first-hit/last-correct, overthinking harm, branch credit, and verifier-disagreement accounting are missing on useful model outputs | 4, 6, 8, 10, 14, 16, 18 |
| Readiness gates + residual escrow | stale outputs/duplicate families reduce trust | 0, 2 |
| Generate-verify-repair | correctness still eval-only, not a training/search signal | 6, 10, 13 |
| Deterministic math/search substrates | not yet attached universally to planning/VCM/claim-ledger/assistant | 6 |
| Benchmark ratchets | public-transfer story stale; remeasure only after private correctness improves | 12 |
| Policy optimization (DPO/GRPO/RLVR) | not yet the generator's central learning mechanism | 7, 10 |
| Integrated reference architecture | the missing canonical spine | 1 |
| Authority kernel / SCIF / failure boundaries / adapter receipts | no universal authority-transition ledger or SCIF receipt layer | 18 |
| Scalable oversight | no trusted/untrusted monitor separation, correlation/collusion probe, weak-supervisor boundary, randomized audit policy, operator-load ledger, or recursion-bottom contract | 4, 14, 18 |
| Capability commitments + safety cases | threshold/safeguard/exception commitments and dependency-invalidating assurance graphs are not consumed by release/default-route decisions | 2, 12, 14, 18 |
| Weight custody + AI supply chain | local AIBOM now distinguishes more than 500 requested/resolved/observed identities across 19 domains, surface Merkle roots, runtime identity, and derivative invalidation; signed advisories, reproducible training/build attestation, attestation-gated load, weight custody, and key lifecycle remain | 0, 2, 7, 8, 14, 18 |
| Inter-stack identity and exchange | federation lacks credential/delegation expiry, reserved budgets, value/accounting receipts, dispute, revocation, and shutdown-handoff interoperability | 1, 2, 9, 16, 18 |
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

The ASI Stack now has 54 active chapters. Ten items from the earlier completeness
intake have been admitted and are owned by the chapter crosswalk rather than this
future queue. The 12 remaining candidates/section routes below are **not new active
lanes**: each is owned by existing phases, remains planning-only until its source,
ownership, and operational entry gates pass, and blocks only the higher-authority
operation it governs. This queue is frozen while current model-only heldout behavior
is zero: it remains the future design contract and is not deleted, but it cannot
spawn implementation/report work unless a concrete correctness, security, or
training-readiness defect requires it.

| Book candidate or section | Theseus owner phases | Entry condition / disposition |
|---|---|---|
| Reasoning-Trace Faithfulness (section route) | 1, 4, 14 | now as an evidence boundary; distinguish traces from receipts and test trace/action consistency without treating hidden reasoning as authoritative |
| Question-Compiled Semantic Addressing | 0, 1, 2, 3, 6, 8, 13, 14, 16, 18, 19 | source available and chapter manifest pending; extend VCM/SCF/VIEA with stable SOID-address-route indirection, plural address atlases, active question traces, SACs, physical translation, migration receipts, and matched falsification without creating a parallel memory lane |
| World Models and Model-Based Cognition | 10, 13, 17 | after the core proposer floor; prediction/error ledgers, model-predictive control, imagination/search, causal limits, sim-to-real |
| Multi-Agent Systemic Risk and Agent Economies | 9, 16, 18 | before multi-agent economic/autonomous operation; collusion, cascades, miscoordination, market behavior, gradual disempowerment |
| Persuasion, Epistemic Security, and Human Agency | 5, 14, 18 | source-contingent; require stronger empirical anchors before activation or claims |
| CAIS, Embedded Agency, and Corrigibility Foundations (section route) | 1, 18, 19 | prior-art/foundations work only unless a distinct mechanism and proof program earns a chapter boundary |
| Perception, Grounding, and Observation Trust | 3, 8, 17, 18 | when a vision/audio/sensor modality starts; observation contracts, provenance/taint, fusion disagreement, uncertainty, refresh, spoofing/occlusion |
| Embodied Execution and Physical-World Safety | 1, 8, 17, 18 | before physical actuation; control periods, hazard envelopes, interlocks, fallback controllers, stop authority, degraded modes |
| Operational Resilience, Incident Response, and Decommissioning | 0, 8, 9, 14, 18 | before production/fleet dependence; criticality, impact tolerance, monitoring, containment, failover, recovery, disclosure, secure retirement |
| Privacy, Data Rights, and Multi-Agent Memory | 3, 5, 7, 9, 18 | before multi-user or shared-memory production; purpose limitation, consent, minimization, access/export/correction, privacy budgets, leakage, deletion closure |
| Autonomous Replication, Proliferation, and Containment | 2, 8, 9, 14, 18 | before any replication/provisioning authority; descendant identity, principal binding, resource acquisition, persistence, containment, shutdown tests |
| Scientific Discovery and Experimental Governance | 1, 6, 7, 14, 17, 18 | when Theseus begins autonomous experiments; hypothesis provenance, preregistration, instrument authority, statistics/causality limits, replication, dual-use review |

Cross-cutting Tier-2 obligations are folded into those owners rather than becoming
more systems: calibrated uncertainty/selective prediction/abstention and
trace-faithfulness (Phases 4/14); prompt-injection limits, autonomous-insider and
collusion probes, randomized monitoring, and operator-load residuals (Phase 18);
stable semantic identity/address/physical-route separation, consumer-relative
address adequacy, active-question value/cost/privacy/risk traces, atlas epoch
migration, and semantic-route authority non-inheritance (Phases 0/1/2/3/6/8/13/
14/16/18);
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
