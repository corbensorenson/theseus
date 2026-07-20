# Project Theseus Roadmap

Consolidated 2026-07-16 and reorganized 2026-07-18; bound to AI Stack commit
`32635eb94ded42a5f54e528302685cab343993b7`. This
roadmap is the forward plan. It retains only the current baseline and completion
receipts needed to prevent repeated or contradictory work; it is not a chronological
audit trail. Historical execution logs, dated
book-mining/review passes, and per-experiment records were removed from this file
during consolidation and remain in git history. The machine-readable coverage
source of truth is `configs/roadmap_implementation_matrix.json`
(`scripts/roadmap_implementation_gate.py --gate`).

Goal: make Theseus the working implementation reference for the ASI Stack book
(`/Users/corbensorenson/Documents/AI_book`) - stable interfaces, governed
replacement, real evidence, useful daily operation, and a local model that
improves without cheating.

## How To Read This Roadmap

Precedence when sections disagree: the **Critical Path and Shared Flagship** decides
what may consume primary effort; the **Capability Plan v2** decides experiment
sequencing; the **Phases** decide mechanism ownership; the **ASI Stack-to-Theseus
Completion Program** decides book-derived coverage.
When this file and `configs/roadmap_implementation_matrix.json` disagree, reconcile
the matrix first (Phase 19), do not paper over drift in prose.

Read order: Critical Path and Shared Flagship -> Status at a Glance ->
Non-Negotiable Rules -> Capability Plan v2 -> the Phase that owns the surface you
are working on -> the Quality Bars a "done" module must meet.

## Critical Path and Shared Flagship

Theseus has a strong governance and evidence denominator but no useful learned
behavior numerator yet. The current model-only functional result is `0/160`, the
assistant product lane is still synthetic-test-backed, and the ASI Stack has no
competence-qualified natural, non-authored flagship result. The next quality gain
therefore comes from one causal chain, not another architecture family, dashboard,
report family, or private benchmark.

The shared program ID is **`ASI-THESEUS-FLAGSHIP-01`**. Its research question is:

> Can a locally trained, source-disjoint Theseus student complete useful natural
> repository work through the governed stack, and does full governed admission
> improve the joint useful-safe-release frontier over simpler matched controls at
> an acceptable total lifecycle cost?

The book owns the claim identity, competence standard, preregistration, evidence
transition, argument-exit language, and publication boundary. Theseus owns the local
student, natural dogfood path, runtime/effect implementation, joined traces, matched
controls, and public-safe evidence pack. The repositories may share stable IDs and
digests; they may not share hidden answers, held-out outcomes, private payloads, or
support states by implication.

### Binding execution board

| Gate | State | Exit condition | Work unlocked |
|---|---|---|---|
| `T0` — Finite architecture closure | `complete` | KERC is explicitly deferred as `INCONCLUSIVE_IMPLEMENTATION` with K0-K3 banked and zero first-campaign exposure; the 53-artifact package passes all eight independent replays and binds its generated effect/governance receipts. | Long optimizer spend. |
| `T1` — Frozen neural-seed campaign | `ready_to_launch` | Five arms and the two matched dense controls train from the exact eight-target frozen package with data, optimizer, checkpoint, resource, cleanup, and no-cheat lineage. | Frozen functional evaluation. |
| `T2` — Honest behavioral numerator | `blocked_by_T1` | At least one lineage-bound arm produces nonzero, model-only, source-disjoint functional behavior; a zero result receives only the exact decision-grade scope earned. | Behavior-dependent search, routing, preference/RL, and fast-generation qualification. |
| `T3` — Real daily-use lane | `ready_for_assisted_use`; learned credit waits for `T2` | At least one week or five distinct days of real low-risk use records accepted, missed, ignored, corrected, completed, failed, or abstained outcomes with time, verifier/human cost, effect, rollback, and component attribution. | Empirical product-usefulness and procedural-memory evidence. |
| `T4` — Joined governed vertical | `blocked_by_T2_and_T3` | One natural happy path and one blocked or rollback path join intent, VCM, plan, route, candidate, verification, authority, effect observation, terminal acknowledgement, residual, and dogfood outcome without orphan or stale projection. | Book-importable implementation evidence. |
| `T5` — Matched causal flagship | `blocked_by_T4` | Full governance, test-only, record-only, and appropriate ablations are compared on a frozen natural denominator with matched tuning, inference, verification, and lifecycle budgets, independent evaluation, uncertainty, weak tails, and a fair rescue ladder. | A bounded positive, exact negative, or inconclusive book transition. |
| `T6` — Book handoff and independent challenge | `blocked_by_T5` | A public-safe pack binds exact commits, artifacts, estimand, effects, costs, residuals, non-claims, and maximum inference; the book updates the affected atoms and narrative case, then any broadened claim faces separate reproduction and transfer. | Broader architecture claims or an honest narrowing. |

`T3` may collect assisted-product evidence before `T2`, but deterministic tools,
retrieval, authored workflows, routers, teachers, and human corrections never become
learned-generation credit. `T4` and `T5` require a behavior-positive student because a
perfect governance trace around a zero-capability proposer cannot establish useful
governed intelligence. All other roadmap work is either a dependency of these gates,
continuous integrity maintenance, or deferred under the breadth freeze.

## Current Operating Decision: Architecture Before Long Training

The current priority is **finite architecture closure**, not corpus-scale optimizer
work. Theseus will not spend a long training run on an architecture that is already
known to be missing accepted representation, objective, context, routing, generation,
update, checkpoint, or self-improvement contracts. The 13-item matrix-owned docket
briefly reached 13 recorded dispositions, but the 2026-07-17 evidence-adequacy audit
reopened KERC: its retirement relied on a hand-coded/linear proxy that did not exercise
the proposed learned architecture. K3 now supplies decision-grade bounded allocator
evidence, while K4-K8 remain incomplete. Rather than convert that incompleteness into a
false negative or let it postpone the practical lane indefinitely, the full KERC
candidate is deferred from the first run as `INCONCLUSIVE_IMPLEMENTATION`: it receives
zero topology, objective, target, matched-control, and optimizer exposure. Training
authority now depends on rebuilding and independently replaying the resulting
eight-target package.

This is not a demand to prove learned efficacy without learned weights. Pre-training
closure proves that selected mechanisms are real, integrated, checkpointable,
migratable, resource-accounted, replayable, negatively tested, and represented in the
frozen campaign. Prospectively bounded mechanics and representative-subset learnability
canaries remain allowed when their data, step ceiling, stop rule, and non-claim are fixed
before execution. Full-corpus training, architecture tuning from interim losses, public
calibration, and capability claims remain after the freeze.

The closure is finite. New ideas discovered after the freeze target the successor
campaign unless they expose a concrete correctness, security, information-flow,
checkpoint, migration, or replay defect that invalidates the frozen system. "Make the
architecture excellent" therefore means close every accepted training-invalidating
idea now, not postpone training forever in pursuit of unspecified perfection.

### Binding architecture-completion contract

Long training is the **last** step of the architecture program, not a convenient way
to discover contracts we already know are missing. An accepted idea cannot sit behind
the first long run when adopting it later would change any model weights, topology,
tokenizer or representation, optimizer objective, routing semantics, memory identity,
generation mode, verifier state, update causality, checkpoint format, or autonomous
campaign behavior. It must instead receive one of three binding outcomes now:

1. **Include:** implement it through the canonical owner and freeze its exact campaign
   role, state schema, objective exposure, resource accounting, migration, rollback,
   replay, and negative controls.
2. **Exclude:** falsify or retire it with a content-bound reason and enforce zero
   topology/objective/optimizer exposure in the first campaign.
3. **Wire but defer efficacy:** complete every interface needed to train and preserve
   it without retrofit, while leaving only evidence that intrinsically requires learned
   behavior for after training. `wired` is not a capability claim.

An architecture item is not complete because a schema exists, a fixture passes, or a
sidecar can be called. Its pre-training boundary requires canonical execution-path
integration, registry ownership, source/configuration identity, checkpoint round trip,
fresh-process replay, migration and rollback, resource bounds, positive/negative/
adversarial/mutation tests, independent integrity checks, cleanup, and an explicit
campaign disposition. When a mechanism is intentionally disabled, the trainer must
prove it receives zero optimizer exposure and cannot silently activate.

The only optimizer work allowed before architecture freeze is a prospectively bounded
mechanics or learnability canary whose purpose is to test tensor flow, representative-
subset overfit, intervention response, state capture, checkpoint/reload, resume
equivalence, cleanup, or a falsification condition. The step ceiling must be justified
by the mechanism and fixed before the run; an arbitrary tiny ceiling must not manufacture
a false negative. Canary metrics cannot select campaign hyperparameters, tune architecture
from heldout outcomes, support a capability claim, or expand into corpus training. Data
inspection and data-contract replay may continue,
but corpus identities stay frozen unless an architecture decision genuinely changes
tokenization, ownership, supervision, or position accounting; in that case the affected
receipts must be recomputed before authority can become GREEN.

Training authority is therefore sequential and machine checked:

`accepted idea intake -> finite docket disposition -> canonical integration -> final`
`architecture freeze package -> unchanged mechanics canaries -> joint campaign`
`preregistration -> long training -> frozen utility evaluation -> causal amplifiers`.

No calendar deadline, sunk training cost, available compute, or desire for a score may
skip a step. Conversely, architecture closure cannot be extended by vague novelty: a
new pre-freeze item needs a concrete campaign-invalidating effect and an existing owner.
Everything else is recorded for a successor campaign.

### Decision-grade evidence and false-negative control

Theseus distinguishes **mechanics evidence**, **proxy evidence**, **regime evidence**,
and **mechanism evidence**. A failed fixture proves a fixture failed. A failed linear
probe proves that representation/learner pair failed. A failed capacity rung constrains
that rung. None of these alone falsifies the complete architecture or theory.

Before a negative result may retire or broadly falsify a mechanism, an independent
adequacy audit must prove:

1. **Implementation fidelity:** every causal stage named by the hypothesis is present,
   trained or executed as intended, instrumented for actual use, and compared with its
   claimed design. Conceptual approximations are labeled and cannot support broad claims.
2. **Learnability and optimization:** gradients reach every learned module; a tiny
   source-disjoint subset can be deliberately overfit; loss and intervention responses
   move in the expected direction; checkpoints reload; module removal changes behavior;
   and no hidden deterministic path supplies the result.
3. **Matched opportunity:** strong baselines and the candidate receive matched raw
   data, total training FLOPs, tuning budget, wall/resource envelope, inference/search/
   verifier budget, and total-system cost accounting. Architecture-specific tuning is
   allowed only through equal prospectively frozen opportunity, not identical settings
   that handicap one design.
4. **Construct-valid evaluation:** tasks exercise the intended advantage, heldouts and
   evaluators are source-disjoint and independently built, task families and weak tails
   are broad enough, and known harness faults cannot dominate the result.
5. **Statistical adequacy:** multiple seeds, paired effects, confidence intervals,
   preregistered meaningful-effect thresholds, and prospective power/sensitivity
   analysis accompany the result. Replication is required before a broad verdict.

The only permitted negative states are `INCONCLUSIVE_IMPLEMENTATION`,
`INCONCLUSIVE_EXPERIMENT`, `NEGATIVE_FOR_EXACT_REGIME`, and
`REPLICATED_MECHANISM_NEGATIVE`. Campaign exclusion for sequencing or cost is recorded
separately and is not scientific falsification. Existing negative evidence is retained,
but claims are narrowed when this audit shows the test did not faithfully instantiate
the idea.

### Finite pre-training disposition ledger

This table is a readable projection of
`pre_training_architecture_contract.required_backlog_ids`; the matrix remains
authoritative. A row may leave `required` only with integrated evidence or a binding
retirement/falsification receipt.

| Order | Contract | Current disposition | Required before long training |
|---:|---|---|---|
| 1 | Kernel English + hierarchical residual compiler | `pretraining_wired_behavior_qualification_pending` / K0-K3 banked; full candidate deferred from the first run as `INCONCLUSIVE_IMPLEMENTATION` | Preserve the K3 receipt unchanged. The first campaign has exactly eight targets and gives KERC zero topology/objective/optimizer exposure. Complete K4-K7 later under a separately frozen K8 successor campaign; do not translate this sequencing deferral into a scientific negative. |
| 2 | Question-compiled semantic addressing | `pretraining_wired_behavior_qualification_pending` | preserve VCM identities/certificates/migration; keep retired objective and adaptive questions at zero exposure |
| 3 | Reflexive Router | `pretraining_wired_behavior_qualification_pending` | pre-training mechanics are wired and replayed; learned routing quality, route regret, and trace-to-reflex value remain post-training evidence |
| 4 | Full-state data/update causality | `pretraining_wired_behavior_qualification_pending` | canonical admission owns a 13-kind content-addressed state inventory, distinct best/final authority, package replay, bounded update, deletion receipts, and exact rollback; learned forgetting/unlearning efficacy waits for training |
| 5 | Multi-target policy-update leases | `pretraining_wired_behavior_qualification_pending` | one registered lease now covers planner/router/VCM/verifier/executor/generator/generation-mode updates with feedback, heldout, full-cost, journal, sentinel, conflict, and exact rollback controls |
| 6 | GVR state machine and localized repair | `pretraining_wired_behavior_qualification_pending` | six-state transition kernel, immutable source identity, independent verifier and bounded repair receipts, tamper-evident history, exact rollback, and learned/assisted/fallback accounting are canonical; behavior lift remains post-training |
| 7 | Open-ended improvement campaign | `pretraining_wired_behavior_qualification_pending` | registered disabled controller now owns independent generation/evaluation/promotion/stop/rollback, single-axis matched challengers, immutable holdouts, novelty/coverage/tail scoring, negative knowledge, six-field debt ceilings, best/final state, exact rollback, and shutdown handoff; activation and efficacy wait for independently verified positive behavior |
| 8 | Preference objectives (DPO/IPO/ORPO/KTO/SimPO) | `pretraining_wired_behavior_qualification_pending` | five interchangeable adapters now share one provenance schema, frozen reference, checkpoint/optimizer round trip, migration, generator lease, exact rollback, MLX parity, and no-cheat mutations; all remain disabled pending honest behavior-positive preference signal |
| 9 | Verifier-reward objectives (GRPO/RLOO/ReMax/RLVR) | `pretraining_wired_behavior_qualification_pending` | four adapters now share one rollout/reward ABI with independent verifier provenance, bounded capacity, reference/baseline identity, clipped ratios, KL accounting, lease, exact rollback, and adversarial controls; all remain disabled pending verifier-positive learned behavior |
| 10 | MTP generation mode | `pretraining_wired_behavior_qualification_pending` | canonical MLX model/trainer now own shared-rank optional heads, masked future-token loss, parameter ceiling, NPZ round trip, and the frozen MoECOT/control campaign binding; initial campaign loss scale is zero and activation/speed wait for preregistered behavior-positive evidence |
| 11 | Medusa/EAGLE/speculative/LayerSkip modes | `pretraining_wired_behavior_qualification_pending` | all modes have explicit topology/objective/checkpoint/cache records: Medusa, EAGLE, and LayerSkip are retired from the first campaign; speculative decode is target-checkpoint-compatible, revision-bound, accepted-prefix-only, disabled, and post-training |
| 12 | Diffusion/LLaDA/sketch-first repair | `retired_by_pretraining_verdict` | the incompatible iterative checkpoint and full-sequence state are excluded from the first campaign under the frozen complexity/resource verdict; re-entry requires a useful AR base and prospectively frozen Semantic-IR-localized matched-total-cost campaign |
| 13 | OneCell-RWM candidate kernel | `retired_by_pretraining_verdict` | excluded from the first language campaign by a content-bound mechanical verdict: the separate recurrent/exact-substrate curriculum would invalidate the frozen assistant comparison; preserve the substrate-neutral ABI and re-enter only through its own preregistered cognitive-kernel campaign |

All thirteen dispositions are mechanically closed for the first practical campaign.
KERC closes by explicit zero-exposure deferral, not by a broad retirement claim. The
content-addressed cross-owner package is GREEN with all eight replays passing; learned
efficacy now requires the unchanged frozen campaign.

## Status at a Glance

| Area | Owner | State | Next concrete action |
|---|---|---|---|
| Data engine + curriculum | Track 0 / Phase 7 | GREEN for the frozen 57.315M proposal: 1,293,454,903 broad unique positions versus 1,146,808,520 required; every task-complete floor is GREEN; HTML/CSS now has 71,416,629 unique positions versus 50,083,860 required; source/vocabulary/supervision identities replay exactly | preserve the content-bound corpus and receipts unchanged while the finite architecture docket closes; any architecture change that alters ownership or tokenization must recompute, never assume, these floors |
| Dense transformer control | Phase 10 | both v8 controls completed unchanged; each is 0/544 exact, with Python syntax 46/128 (active) and 48/128 (total) | retain as immutable 10.8M-rung falsification evidence; do not spend confirmation or patch this rung |
| MoECOT language-specialist seed | Track 1 / Phases 10, 16 | v8 trunk/specialists are complete; exact recovery is English 1/128 and code 0, while frozen functional utility is 0/160 | retain as negative modular evidence; do not let routing hide the behavior-zero result |
| Verifier-guided search | Track 2 / Phases 6, 10 | architecture wired, amplifier waiting for signal | preserve the bounded kernel and replay contract; qualify it only after one-shot generation sometimes succeeds and search materially increases held-out pass |
| Correctness training (DPO->GRPO/RLVR) | Track 3 / Phase 10 | learned optimization remains premature at the zero-pass floor, but all five offline-preference and four verifier-reward objective adapters now execute through a shared frozen schema, checkpoint/reference identity, MLX parity probe, capacity ledger, update lease, exact rollback, and twelve rejecting controls | preserve zero exposure and run/select an objective only after a behavior-positive proposer creates honest private verifier signal |
| Fast-gen modes (MTP/diffusion/self-draft) | Track 4 / Phases 8, 10 | pre-training topology is closed: AR is canonical; checkpointed low-rank MTP is campaign-bound at zero initial loss scale; Medusa/EAGLE/LayerSkip/sketch-first are explicitly retired; speculative decode is disabled and post-hoc compatible. Canonical MLX forward/loss/save/reload passes at 21.875% output-head overhead with zero reload drift | preserve the frozen disposition; set any nonzero MTP schedule prospectively and adopt any fast mode only from accepted verified output per second under matched quality/cost accounting |
| Generator capability (held-out utility) | Phase 10 | the 10.8M scale regime is closed; the practical 57.315M eight-target package is frozen with KERC deferred at zero exposure. The 53-artifact package and all eight replays are GREEN; no long optimizer run or new capability evidence exists yet | begin the frozen joint campaign, preserving exact package identity and per-target lineage |
| Cognitive-kernel discovery | Phase 11 | OneCell-RWM is non-routeable and retired from the first language campaign with zero optimizer/checkpoint exposure; its ABI, exact/latent boundary, objective/checkpoint groups, owner reuse, and separate successor-campaign prerequisites are content-bound, but no OneCell substrate or learned capability is claimed | preserve the successor experiment without appending it to first-campaign weights; re-enter only after exact substrate and a separately preregistered matched cognitive-kernel campaign exist |
| Kernel English + hierarchical residual compiler | Phases 3, 10, 13, 14, 16 | K0-K3 are banked; the incomplete full candidate is deferred from the first practical run as `INCONCLUSIVE_IMPLEMENTATION`, with zero topology/objective/optimizer exposure. | After the practical capability run, complete K4 real coding/amortization, K5 renderer/verifier calibration, K6 lifecycle/security stress, and K7 total-cost instrumentation, then prospectively freeze K8. Never treat the deferral as a KERC failure. |
| Self-improvement flywheel | Tracks 0, 3 / Phases 7, 10 | the disabled campaign controller is now canonical through governance and the existing overnight entrypoint: it separates authorities, freezes holdouts/budgets, preserves rejected families, accounts debt, stops transactionally, and restores exact state; behavior remains at the zero-pass floor | preserve the disabled mechanics and enable generate->verify->admit->retrain only after an independently verified behavior-positive proposer receipt exists |
| VCM ABI + transactions/certificates | Phase 3 | wired: ABI, stable semantic objects, typed temporal relations, hybrid retrieval, lifecycle transactions, compaction, and fresh-process ontology migration | consume lifecycle records in Phase 7/10; keep dense embedding, parametric unlearning, and public-memory capability claims separate |
| Claim ledger + belief revision | Phase 14 | ledger implemented; assurance/evaluation-integrity consumption partial | compile one live assurance graph and cross-context integrity record into route decisions |
| Replacement transactions | Phase 2 | replayable-reference-backed for one bounded local route-authority effect | keep the independent effect audit and mutation controls green; require equivalent receipts for each new effect class |
| Procedural memory + toolification | Phase 15 | implemented for the three guarded exact assets; the exact Reflexive Router/trace-to-reflex contract is now part of pre-training architecture closure, while learned promotion still requires real traces | wire qualification, dependency invalidation, differential/shadow/canary, quarantine, decompilation, and rollback interfaces before training; defer only empirical promotion and route-regret claims |
| Authority kernel / SCIF | Phase 18 | replayable-reference-backed for the same bounded local effect; SCIF and wider authority controls remain synthetic | preserve exact effect/rollback auditing and expand support only when another real effect class earns it |
| Assistant product | Phase 5 | assisted runtime wired; model-only general chat unavailable | dogfood deterministic/verified assistance now, but earn model usefulness only from real multi-day use after the local model clears its behavior floor |
| Report/repository hygiene | Phases 0, 8, 14 | live reports are about 3.2 GB, runtime fell from 46 GB to 25 GB after removing 20 GB of reproducible deferred-KERC cache, archive remains 14 GB, and the superseded strict-generator/code-lm-closure family still occupies 17 scripts and about 50,865 Rust LOC | keep canonical evidence and training inputs intact; after campaign launch, execute the registered retirement proof and compact this roadmap below its 1,100-line target without dropping matrix obligations |
| Book crosswalk / parity | Phase 19 | 54 chapters are bound to immutable AI_book commit `32635eb...`; live worktree changes are intake drift, not an architecture regression | reconcile and advance the pin only in a reviewed change after book edits are committed |
| Book test obligations | Phase 19 + routed owners | 511 authored Codex tests; 109 remain planned or partial in the book | close by mechanism family with real controls, not checkbox fixtures |
| Book futures intake | Phase 19 + routed owners | every accepted item is classified by training-invalidating effect; KERC, QCSA, Reflexive Router, OneCell disposition, update/data lifecycle, and generation-objective interfaces are on the finite pre-training docket | implement or explicitly retire every checkpoint-shaping item before training; retain genuinely modality-, peer-, real-use-, or behavior-dependent evidence as post-training qualification |

Pre-training readiness uses the matrix-owned phase partition rather than every
unfinished roadmap item. Phases `0, 1, 2, 3, 4, 6, 8, 11, 14, 15, 18, 19` are
architecture prerequisites; phases `5, 7, 10, 12, 13, 16, 17` contain empirical
acceptance that requires training, real-use time, public calibration, or
behavior-positive candidates and therefore cannot circularly block architecture
readiness; Phase `9` remains an external-environment proof. A matrix-owned finite
pre-training backlog cuts across that partition for known mechanisms that could
change model topology, representation, tokenization, objectives, routing, memory
addressing, update causality, checkpoint lineage, or generation semantics.

Every item on that docket must reach one of four honest dispositions before long
training: `pretraining_wired_behavior_qualification_pending`, `implemented`,
`falsified_pretraining`, or `retired_by_pretraining_verdict`. "Wired" means the full
registered interface, migration, lifecycle, security, resource, checkpoint, replay,
negative-test, and campaign contract exists; it does not mean learned utility has
been proved. Items whose only missing evidence intrinsically requires learned output
retain that evidence as a post-training obligation. Items that are merely fashionable,
duplicate an existing owner, or cannot justify their total cost are retired before
the run rather than half-added afterward. Once the finite docket closes, architecture
freezes for the campaign; later ideas target a successor campaign unless a concrete
correctness, security, replay, migration, or checkpoint-invalidating defect is found.

The canonical MoECOT trainer currently enforces this sequencing with a fixed eight-step
canary ceiling. Replace that coarse ceiling with a signed canary manifest that binds the
mechanics question, representative subset, maximum steps/tokens/FLOPs, stop rule,
forbidden heldout use, and exact non-claims before execution. The gate may authorize the
smallest prospectively bounded run capable of testing gradient flow, representative-
subset overfit, intervention response, checkpoint/reload, or resume equivalence; it may
not authorize corpus-scale training, architecture selection from interim outcomes, or
capability claims. Unbounded runs still fail closed unless
`roadmap_implementation_gate.py --gate --require-pre-training-ready` is GREEN. This is
an optimizer-spend boundary, not a mechanism for forcing inconclusive canaries through an
arbitrary step count.

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
- Capability scoreboards are task-appropriate and source-disjoint. English uses
  frozen blind rubric/pairwise judgments; Python, JS/TS, HTML/CSS, and Rust use
  compile/test/execute or structural/render checks. Exact recovery, syntax,
  nontrivial-return, LM loss, and preference-gap remain diagnostics unless the task
  explicitly requires byte-exact output.
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
- Negative results are kept with exact implementation/regime scope. A result cannot
  retire a broader mechanism until the decision-grade adequacy audit passes; otherwise
  it is explicitly inconclusive and the missing implementation or experiment work is
  returned to its canonical owner.
- Close the finite pre-training architecture docket, then freeze control-plane and
  architecture expansion for at least three capability cycles. A change during the
  freeze requires a concrete failure receipt showing that the frozen contract cannot
  represent, train, checkpoint, migrate, or execute the required behavior.

## Implementation and Evidence Standard

Every mechanism selected for the active critical path must be strong enough to test
the causal idea it represents. The roadmap still records a minimal honest slice and a
mature or beyond-SOTA endpoint for every surface, but the mature endpoint is a research
horizon, not a universal prerequisite that can block the neural seed forever.

- **Name the relevant strong comparator.** Use the best reproducible published or
  open method appropriate to the exact claim. If it cannot be run under comparable
  conditions, record the gap instead of implying a win or making the whole roadmap
  wait indefinitely.
- **Separate the current envelope from the mature endpoint.** A surface may be done
  for a bounded interface, safety, or evidence envelope while its broader performance
  endpoint remains pending. The completed state must say exactly what it authorizes
  and what it does not.
- **No placeholder passes as mechanism evidence.** A schema, happy-path fixture,
  stub, hardcoded shortcut, or single-case smoke may prove plumbing only. It cannot
  establish usefulness, mechanism efficacy, governance efficacy, or a scientific
  negative.
- **Prefer the smallest faithful design.** Implement every causal stage needed by the
  claim and omit ornamental machinery. Added complexity must earn its maintenance and
  lifecycle cost against a simpler control.
- **Measure the whole system.** Capability, safety, latency, compute, storage,
  verification, human effort, migration, rollback, and residual debt stay in one
  lifecycle accounting envelope; no route wins by moving cost off-ledger.
- **A negative claim is harder to earn than a green smoke.** Each adoption or
  retirement card records fidelity, learnability checks, matched opportunity,
  evaluator independence, power/sensitivity, uncertainty, weak tails, replication,
  and maximum claim scope. Toy, undertrained, inactive, or construct-invalid tests
  are diagnostic or inconclusive, not architecture verdicts.

The Quality Bars enforce the declared current envelope. The capability-per-active-
parameter and governed-system frontiers remain ambitious mature targets that must be
earned by held-out evidence, not definitions that make all intermediate work
permanently incomplete.

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
knowledge inside one monolith. The current frozen controls are a 10.780M-active
dense transformer and a 12.500M-total dense transformer, matched to the v8 sparse
system under two preregistered accounting views. Neither control is silently
promoted; their purpose is to falsify or select the practical architecture.

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
- **50M-100M is the next candidate band, not an automatic entitlement or ceiling.** The architecture,
  data pipeline, training loop, and search must be designed so active parameters,
  expert count, corpus size, search budget, and reasoning depth scale together as
  compute grows, with no rung-one shortcut that blocks rung two.

### Critical Path (ordered - this is what moves capability)

**Current execution order (binding for the next capability cycle):**
1. Close the finite architecture-decision docket before optimizer spend. For every
   accepted book-derived mechanism, record whether it changes model/checkpoint shape,
   training objectives, representation/tokenization, memory/routing semantics, or is
   strictly checkpoint-compatible. Implement selected checkpoint-shaping mechanisms
   through existing owners; explicitly falsify or retire rejected ones. No item may
   remain merely `planned` if adopting it later would invalidate the long run.
2. Correct and complete KERC's **trainable contract** before long optimizer spend.
   Preserve the exact packet, object, codec, and VCM lifecycle substrate, but treat the
   current single-node/opaque-literal corpus and channel-presence classifier as mechanics
   fixtures only. Execute K0-K8 in the KERC section: independently audit fidelity;
   build licensed source-disjoint multi-proposition semantic supervision; represent and
   train per-unit rate-distortion decisions; exercise real correction, concept, macro,
   ambiguity, copy, and interaction state; make the compiler/core/answer/renderer losses
   coordinated; harden an independent semantic recompiler/verifier; and freeze strong
   conventional, byte/dynamic-chunking, semantic-representation, and KERC controls.
   Canaries may establish gradient flow, replay, and intervention response, never utility.
   Do not manufacture easy templates, opaque whole-answer semantic targets, stage-token
   shortcuts, or deterministic rendering credit. If KERC cannot reach this finite
   campaign-ready boundary, exclude it from the first practical run as
   `INCONCLUSIVE_IMPLEMENTATION` rather than either falsely retiring it or indefinitely
   blocking the conventional surface-English survival lane.
3. Preserve the completed QCSA disposition: VCM owns stable SOID/address/route
   indirection, plural facets, Semantic Address Certificates, authority separation,
   and exact migration/rollback; the full QCSA objective and adaptive-question policy
   remain outside the first long run. Preserve the completed Reflexive Router mechanics
   and the completed OneCell sequencing, full-state update/rollback, policy-update,
   GVR, autonomous-campaign, preference/RL, MTP, speculative/LayerSkip, and sketch-first
   dispositions. These remain closed unless their own adequacy audit identifies a
   concrete implementation defect; learned promotion remains post-training.
4. Preserve the content-bound corpus and source-disjoint conventional surface-English,
   unchanged Python/JS-TS/HTML-CSS/Rust arm, and mixed dense-control views. Record
   unique raw bytes and model-visible positions, shared-row reuse, tokenizer efficiency,
   contamination, license/provenance, and exact row ranges. Derived views of one source
   are not new unique data.
5. Re-preregister one joint MLX campaign before training: conventional surface-English,
   a faithful separately accounted KERC English candidate, unchanged code arms, and
   matched dense controls. The prior KERC proxy receipt remains attached as
   `INCONCLUSIVE_IMPLEMENTATION`, not as a retirement disposition.
   Predeclare total/active parameters, raw bytes, total train FLOPs, complete
   end-to-end inference cost, wall/energy budget, heldouts, weak-tail floors, and
   falsification. Replay finite-update, checkpoint/reload, optimizer-resume,
   migration, cleanup, memory, and resource canaries. The original 57.315M canary is
   retained as mechanical evidence for the surface baseline but cannot authorize the
   revised campaign by itself.
6. Replay registry, integrity, finite-update, checkpoint/reload, optimizer-resume,
   migration, cleanup, memory, resource, and exact rollback canaries against the
   final architecture package. Publish one content-addressed architecture-freeze
   manifest binding source, data views, tokenizers/compilers, topology, objectives,
   routes, VCM/QCSA schemas, KERC/OneCell dispositions, verifier versions, and AIBOM.
7. Train each frozen alternative once. Do not let one arm's optimizer state or
   residuals mutate another except through an explicit governed update. Do not tune
   architecture, representation, renderer, or thresholds from heldout outcomes.
8. Evaluate arm behavior, router behavior, and composed behavior separately. Preserve
   the v8 exact-recovery result as a diagnostic, then compare unchanged checkpoints on
   a separately frozen functional surface: blind rubric/pairwise English and
   compile/test/execute or structural/render checks for every code arm. Route success
   never counts as answer success; dense wins remain valid evidence against the thesis.
9. After any direct arm behavior is positive, qualify STS/VCM conditioning, search,
   preference/RL, and fast generation in that order. Only after a materially changed
   current system clears private integrity/behavior gates, spend a fresh public
   calibration surface. SymLiquid remains a later protected comparator.

**Current completion ledger and binding next sequence (2026-07-17):**
- The historical v8 ABI and five arm contracts are frozen; the
  215.55M-position training view is immutable; the 10.347M shared trunk and all five
  430,849-parameter specialists completed with distinct checkpoint/optimizer lineage.
- Arm-level v8 development evaluation is complete. English produced one exact target
  recovery (`1/128`); all code arms remain `0` exact and Python is `41/128`
  syntax-valid. Because these rows contain prompt and target but no executable test or
  functional verifier, this is an **exact-recovery diagnostic**, not a utility verdict.
- Both dense controls completed unchanged. The active checkpoint `71d3e099...` and
  total checkpoint `e56aaa9b...` each consumed about 227.184M optimizer positions and
  scored `0/544` exact; Python syntax was `46/128` and `48/128`, respectively. These
  remain exact-recovery/syntax diagnostics, not utility.
- The architecture-neutral 160-case functional contract was frozen before total-
  control completion and consumed exactly once. Eleven append-only identities each
  have one reservation and one completion, with no failed lifecycle. MoECOT,
  dense-active, and dense-total each scored `0/160`, including `0/32` in English,
  Python, JS/TS, HTML/CSS, and Rust. Model-only and assisted channels stayed separate;
  public rows, runtime external inference, fallbacks, templates, tools, and routers
  received zero generation credit.
- Local English scoring used two pinned 4-bit MLX primaries plus a conditional local
  adjudicator, retained no raw responses, and admitted no judgments to training.
  Quadratic weighted agreement was low (`0.0862`, `0.0355`, and `0.1489` by model),
  so English rubric precision remains a harness limitation; it cannot rescue or hide
  the independently exact code result of zero passes in every arm.
- The preregistered verdict is `NEGATIVE_FOR_EXACT_REGIME:10_8M_ACTIVE_SCALE_RUNG`.
  No architecture was
  selected, no Pareto relation was claimed, confirmation remains untouched, and route
  replacement is unauthorized. Do not rerun, reweight, or patch this consumed rung.
- The KERC retirement has been withdrawn after an implementation-adequacy audit. Its
  192-row proxy used a hand-coded keyword compiler/renderer and TF-IDF logistic
  regression, omitted every major learned KERC stage, and trained on 128 authored
  templates. Preserve it as proxy-negative evidence, but do not use it to exclude KERC.
  This is the one concrete architecture defect that reopens the freeze.
- The practical model lane is now data-supported at 57.315M active parameters. The
  canonical corpus has 1,293,454,903 unique positions, every specialist clears its
  20:1 owned-parameter floor, every task-complete floor is GREEN, and all three MLX
  mechanics canaries pass with temporary artifacts removed. It is **not the current
  primary execution lane**: first close the finite architecture docket and freeze the
  joint campaign, then replay these canaries against the final package before optimizer
  spend. Learned qualification of STS, VCM conditioning, search, preference/RL, and
  fast generation still waits for a nonzero model-only numerator; their
  training-invalidating interfaces do not.
- Public calibration follows a material, confirmation-qualified model change. It is
  measurement only and never a source of prompts, tests, solutions, traces, labels,
  templates, or residual training rows.
- OneCell-RWM remains a separately scoped successor cognitive-kernel candidate, not a
  v8 rescue patch or a replacement for Theseus. Its first-campaign exclusion is an
  engineering sequencing decision, not a negative scientific result. It may not use
  v8 outputs as tuning data or receive route authority before its own matched
  qualification, and it must extend existing VIEA, VCM, Octopus, SCF, Loop Closure,
  SparkStream, artifact, residual, and evidence owners.

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
  The current content-bound audit measures `1,293,454,903` unique positions versus
  `1,146,808,520` required (`22.557469` per active parameter). Every arm clears its
  `50,083,860`-position floor: English `546,946,708`, Python `186,888,952`, JS/TS
  `283,990,104`, HTML/CSS `71,416,629`, and Rust `188,216,352`. Every task-complete
  unit/position floor and the training-admission TCB are GREEN. Preserve these exact
  receipts while KERC adds a separately accounted English view; derived KERC views do
  not create new unique source credit.
- Keep the three data budgets mechanically separate in every scale decision:
  `broad_pretraining_unique_positions`, `task_complete_unique_target_positions`, and
  `optimizer_exposure_positions`. Task-complete targets measure product supervision;
  they do not close the broad pretraining floor unless their complete model-visible
  source rows also pass canonical corpus admission and cross-source deduplication.
  Repetition can increase optimizer exposure only. The 57.315M rung is authorized
  only when its 1.146B broad unique-position floor and every per-arm task-complete
  unit/position floor pass simultaneously; no subtotal may be relabeled or counted
  twice to clear either gate.
- Expand the corpus in the product's actual domains: English conversation and
  instruction following; Python first; then JS/TS/HTML/CSS and Rust. Measure
  executable completeness, dependency context, algorithm/contract coverage,
  natural-language quality, dialogue continuity, corrections/tool traces, and
  long-tail representation. Avoid bulk intake that merely increases tokens while
  leaving the current semantic residuals untouched.
- Natural-language scope is English. "Multilingual code" means Python, JS/TS,
  HTML/CSS, and Rust, not broad human-language intake. Every natural-language source
  carries an English filter/receipt; non-English rows quarantine.
- Rust is part of the frozen five-arm seed and cannot be removed after observing its
  coverage or model result merely to clear a data or architecture gate. If a future
  seed changes language scope, that change requires a new prospective contract,
  remapped heldouts and matched controls, and an explicit product tradeoff; it cannot
  retroactively authorize the current 57.315M rung.
- Prefer high-quality licensed human/open corpora and static openly licensed
  model-derived corpora over live teacher generation. Provider origin alone neither
  admits nor rejects a static dataset; license, provenance, quality, diversity,
  decontamination, and measured heldout utility decide.
- Operator policy ratification (2026-07-16): already-published static corpora may
  contain Claude/Anthropic-derived rows and remain eligible under the ordinary
  third-party corpus gates. This does not authorize live Anthropic generation or use
  of Corben's Claude subscription, credentials, CLI, API, or desktop app.
- Treat published agent traces as structured trajectory data, not automatically as
  instruction targets. A trace source enters quarantine before any training credit:
  freeze the upstream revision and byte hash; preserve model/provider provenance;
  resolve dataset and embedded repository/content rights; remove secrets, personal
  paths, credentials, and private payloads; enforce English plus Python, JS/TS,
  HTML/CSS, and Rust scope; and deduplicate by source task, repository, patch,
  message, tool sequence, and semantic near-match. Mirrors and repackagings receive
  no new unique-data credit.
- Admit only outcome-grounded trace views. Successful final answers, patches, and
  tool calls require replayable compile/test/render or task-outcome evidence;
  failed trajectories may supply explicitly rejected preference/search examples but
  never positive SFT targets. Hidden chain-of-thought is not required by the product
  and receives no privileged quality assumption. Preserve final-answer, action,
  tool-call, result, and verifier boundaries so the model learns when and how to use
  tools rather than imitating an unverified transcript.
- The original [`Glint-Research/Fable-5-traces`](https://huggingface.co/datasets/Glint-Research/Fable-5-traces)
  release is an eligible **quarantine candidate**, not an admitted corpus. Its
  AGPL-3.0 dataset license, Claude/Fable provenance, 4,665-row scale, embedded
  code/tool payloads, and many public mirrors
  require a frozen-source rights audit, exact cross-mirror deduplication, PII/secret
  scan, benchmark decontamination, outcome reconstruction, verifier replay, and a
  small heldout causal pilot before any row or token counts toward the 57.315M rung.
  The 2.01M-row repackaging must not be treated as 2.01M independent high-quality
  traces: it is a normalized-row-deduplicated compilation of 17 source datasets whose
  rows include events and intermediate completions. Preserve `row_hash`, `seen_count`,
  and first-source provenance, then reconstruct and deduplicate source/session/task/
  repository trajectories before counting quality units or model-visible positions.
- Keep trace purposes separate. Content-bearing, rights-clean, verifier-positive
  rows may compete for a capped synthetic SFT/preference tranche. Sanitized
  metadata-only traces such as Codex-filtered TraceLab may calibrate context length,
  prefix-cache, tool latency, scheduling, and runtime replay, but cannot count as
  conversational/code supervision or model capability. Report source-specific
  unique positions, optimizer exposure, acceptance yield, verifier yield, duplicate
  collapse, contamination rejects, and heldout lift; static trace intake never counts
  as live teacher usage and never authorizes that provider for serving or generation.
  [TraceLab](https://github.com/uw-syfi/TraceLab) is the reference metadata-workload
  source; pin its release and retain only the provider subset authorized by the
  particular experiment.
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
- Each arm owns separate expert weights, optimizer/checkpoint lineage, data receipt,
  context contract, verifier suite, capability scorecard, and lifecycle. A versioned
  shared trunk is a common immutable dependency, counted once; changing it invalidates
  every dependent arm for coordinated requalification. Expert-only updates remain
  independent and cannot cause silent cross-arm drift.
- Train matched dense controls from the same canonical rows. Compare at matched
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

The 10.8M-active five-arm seed is a systems-canary rung. If it is functionally zero,
the next candidate band is 50M-100M active only after task-complete data and MLX
canaries support it. Define an explicit ladder (per-arm and total parameters, active
params, expert count, unique/task-complete data, search budget, reasoning depth) with
a governed step function: each rung requires a matched-baseline ablation, functional
capability-per-active-parameter datapoint, and falsification condition before the next
rung is funded. Scale is earned by data and evidence, not architecture preference.

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

- Evidence-scope gate: no experiment may claim beyond the most specific state its
  adequacy audit supports. Missing learned stages or failed sanity checks produce
  `INCONCLUSIVE_IMPLEMENTATION`; weak power, narrow tasks, confounded evaluators, or
  unmatched opportunity produce `INCONCLUSIVE_EXPERIMENT`; a sound single campaign
  produces `NEGATIVE_FOR_EXACT_REGIME`; only independently repeated decision-grade
  evidence may produce `REPLICATED_MECHANISM_NEGATIVE`.
- Capability gate: if model-only functional utility is zero at 10.8M across sparse
  and both dense controls, that exact scale/data/training regime is closed before
  search or RL. This does not falsify transformers, MoECOT, or sparse specialization.
  If a later
  task-complete, data-supported 50M-100M rung also fails against matched dense
  controls, stop that rung and do not authorize a third size increase. First pivot
  the objective, task-complete data mix, tokenizer, decode/search interface, and
  verifier-grounded supervision under a new preregistered hypothesis. The current
  57.315M contract already stops on two consecutive reviews with no model-only
  functional gain; interim results cannot relax that rule. Search and RL never rescue
  a zero candidate numerator or retroactively turn the failed rung into a success.
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

### Finite Architecture Closure Before First Long Run (2026-07-16)

The July 11 architecture closure was valid for the source set known then: Registry/
SCF, VIEA, VCM, candidate integrity, deterministic tools, bounded search, resource
routing, SymLiquid isolation, evidence/claim records, procedural memory,
authority/failure contracts, and book synchronization were wired or implemented.
The subsequent KERC/QCSA disposition work plus Reflexive Router, OneCell, and
update-causality intake exposed concrete missing representation, routing, memory-address, lifecycle, training, and
evaluation contracts, and no 57.315M long optimizer run has started. It is cheaper
and more honest to close these known gaps now than to train and retrofit them. Existing
corpus receipts and original MLX canaries remain evidence for their exact inputs; they
are not discarded or narrated as capability, but the final architecture must replay
them before gaining training authority.

The roadmap therefore keeps one finite pre-training program open, not an endless
research pause. Its machine-owned docket is
`pre_training_architecture_contract.required_backlog_ids`. It ends when every item is
fully wired, decision-grade negative, or scope-limited retired; the joint campaign and architecture-freeze manifest
are content-bound; and all mechanical, lifecycle, security, resource, migration,
checkpoint, integrity, and no-cheat canaries pass. Learned efficacy remains
post-training evidence and cannot circularly block the run. The following rules apply:

1. **No architecture-by-analogy or perpetual perfection gate.** An item enters the
   docket only if it is already accepted in this roadmap and adopting it later could
   invalidate model weights, training state, representation, routing/memory semantics,
   or the campaign comparison. New speculative ideas wait for the successor campaign
   unless they reveal a concrete correctness or training-invalidating defect.
2. **Do not confuse proxy failure with mechanism failure.** The KERC adequacy audit
   invalidates the old retirement because the experiment omitted the learned compiler,
   core, residual allocator, and renderer. Any future negative disposition must pass
   the decision-grade evidence contract before it can narrow more than the exact proxy.
3. **Do not confuse route success with model success.** VCM, STS, tools, search,
   renderers, n-grams, and procedural routes can improve assisted usefulness, but
   direct learned behavior remains its own scoreboard.
4. **Implement interfaces before efficacy experiments.** Preference/RL, GVR, search,
   sparse routing, fast-generation, continual-update, and open-ended-improvement
   contracts must exist before training when they affect objectives, state, or
   checkpoints. Their learned qualification still waits for a non-zero numerator.
5. **Treat architecture and data as jointly frozen experimental variables.** The next
   run preserves conventional surface-English, the governed corpus, code arms, matched
   controls, source-disjoint heldouts, and the faithful KERC alternative.
   Derived views receive no unique-data credit, and all runtime cost counts.
6. **Calibrate publicly after a material model change, not by calendar.** Fresh
   frozen surfaces remain available without arbitrary budgets, but an unchanged
   zero-pass model does not justify spending another public surface.

### Architecture-Freeze Package

The architecture docket is not closed by prose or by changing a matrix status. One
content-addressed freeze package must bind all of the following before long training:

1. **Model topology and ownership:** trunk, each Octopus/MoECOT arm, dense controls,
   parameter accounting, route boundaries, OneCell disposition, and the exact
   conventional-English plus faithful KERC English alternatives.
2. **Representation and context:** tokenizers, protected-object and scoped-glossary
   mechanisms, the constrained learned compiler/core/renderer representation,
   Semantic IR, VCM/QCSA schemas and atlas epochs, context
   budgets, residual lifecycle, and migration compatibility.
3. **Objectives and update causality:** causal LM, separately accounted KERC compiler,
   Kernel reasoning, residual-allocation, answer-packet, renderer, and verification objectives; preference/RLVR
   interfaces, policy leases, GVR transitions, optimizer ownership, best/final
   checkpoint authority, rollback, deletion, unlearning, and invalidation semantics.
4. **Generation and deliberation:** autoregressive baseline plus explicit include,
   checkpoint-compatible defer, falsify, or retire decisions for MTP, speculative/
   LayerSkip, diffusion/sketch-first, search, STS, reflex compilation, and tool use.
   Assisted mechanisms remain separately scored and receive no learned-generation
   credit.
5. **Execution and autonomy:** VIEA plan/effect spine, Reflexive Router lifecycle,
   campaign stop authority, resource/thermal limits, structured failures, exact replay,
   negative knowledge, and sole effect-commit ownership.
6. **Evaluation contract:** source-disjoint development and heldout identities,
   English rubric policy, per-language functional verifiers, weak-tail floors, matched
   data/compute/cost views, no-cheat checks, and public-calibration separation.
7. **State portability:** checkpoint schemas, loader and migration matrix, optimizer
   resume, deterministic replay tolerance, cleanup, storage retention, AIBOM/source
   identity, and failure recovery on Apple Silicon MLX.

Every binding records its registry abstraction and implementation IDs, schema/version,
source digest, configuration digest, evidence refs, support state, rollback route, and
explicit non-claims. The final gate independently recomputes those bindings and runs
negative mutations; a self-declared GREEN report cannot grant training authority.

Only three classes of work may remain after this package freezes: evidence that
intrinsically requires learned behavior, evidence that requires real elapsed use or
external peers, and checkpoint-compatible additions explicitly shown not to alter the
campaign comparison. Everything else is either implemented now or deliberately
retired now.

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
- Formally retire superseded capability lanes instead of leaving them adjacent to
  the active route. The first audit covers the strict-generator family and
  `crates/symliquid-cli/src/code_lm_closure`: trace active imports, registry edges,
  verification commands, report consumers, and reusable mechanisms; extract live
  dependencies into canonical owners; mark retained comparators non-routeable;
  archive superseded reports through verified retention pointers; and delete or move
  reference-only source under `deprecated/`. No active gate may depend on a retired
  implementation; git history remains the recovery path.
- Keep this roadmap forward-only and bounded. Phase prose contains requirements,
  dependencies, acceptance, and prohibitions; dated run narratives, report catalogs,
  and implementation inventories belong in the matrix, registry, bounded state page,
  or git history. Add a roadmap-entropy check and consolidate the live roadmap below
  1,100 lines without dropping unresolved book obligations from the matrix.
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
  what Theseus is without scanning historical reports. No active route or gate
  imports a retired strict-generator implementation, and roadmap-entropy checks
  reject implementation-history accretion.
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

The bounded pre-training disposition is complete and content-bound by
`configs/vcm_semantic_addressing.json`. The source campaign covered 12 implementation
lanes, 60 heldout tasks, 13 systems, three seeds, 2,340 predictions, a 13-stage
vertical path, and ten vertical adversarial paths. QCSA and the best task baseline
both scored `1.0` task accuracy; QCSA scored `1.0` object accuracy versus `0.633333`,
but consumed `4.05` mean operations versus `2.116667` (`1.913386x`). The active-
question ablation also retained `1.0` object/task accuracy. Therefore the complete
QCSA objective and adaptive active-question policy do not enter the first long run.

The mechanisms with attributable value are retained through the existing VCM field:

1. **Stable identity and plural addresses.** Opaque SOIDs are independent of VCM
   page revisions, contextual semantic addresses, atlas facets, and physical routes.
   Context-packet and usage-event occurrences remain distinct even when they share a
   backing report. Similarity and neighborhood never establish identity.
2. **Semantic Address Certificates.** Every governed semantic object receives a
   task-, consumer-, epoch-, facet-, provenance-, residual-, permitted-use-, and
   authority-bound certificate. Tampering, stale epochs, consumer laundering, and
   use/authority widening fail closed.
3. **Authority-safe translation.** A valid certificate may propose a temporary
   physical route but cannot authorize an effect; SCF/VIEA effect authority remains a
   separate mandatory decision. Superseded, quarantined, and retracted objects are
   not routeable.
4. **Atlas lifecycle.** Readdressing preserves SOID identity and explicit lineage;
   migrations require descendant/cache/backup/receipt inventory and a passed shadow,
   emit typed failures, reject silent retargeting, and support exact rollback.

The complete learned address/question objective, adaptive question selection,
soft-routing efficacy, and semantic-to-surface generation remain outside the frozen
campaign. They may return only through a prospectively frozen, source-disjoint,
matched-total-cost trial that beats direct/simple controls; deterministic schemas and
certificates receive no learned-generation credit. The observed reference replay is
also recorded honestly: the implementation validator passed, but byte-identical
evaluation replay differed by `0.000001` on one Brier value while leaving every gate
and disposition unchanged; the vertical validator was not rerun after that fault.

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

Admission-substrate status (2026-07-16): `implemented`; full-state update causality
mechanics are now `pretraining_wired_behavior_qualification_pending`. Learned forgetting,
influence, and unlearning efficacy remain post-training measurements. The canonical admission path
now writes and replays `115,407` content-bound candidate receipts, admits
`115,103`, rejects `304` rows, and exposes zero detected exact/semantic
public overlap, fallback markers, or raw-user rows. All nine adversarial cases,
five frozen continual-policy simulations, and the 11-kind positive/negative
descendant-deletion fixture pass. Curriculum and survival-lane materialization
require admitted row hashes. The lifecycle surface remains `YELLOW`, not
`GREEN`, because recursive synthetic share is `0.459053`; that is a data-risk
warning, not proof of model collapse. The policy comparison is simulation, and
deletion closure is graph evidence, not physical unlearning.
The historical v8 materializer measured 683.25M content-bound positions and froze
215.55M for that completed matched comparison. The current canonical broad receipt
  is `1,293,454,903` positions for the preregistered 57.315M rung. These receipts prove quantity,
lineage, and tokenizer integrity, not product suitability. Most v8 development rows carry prompt and target
without executable tests or task-level utility contracts; no later score may relabel
them as functionally verified examples.
- The task-complete contract is now implemented inside canonical admission. Its full
  pinned-source pass admits `30,962/35,887` units: English `25,566`, Python `2,022`,
  JS/TS `1,180`, HTML/CSS `690`, and Rust `1,504`. Python is GREEN at `1,129,825`
  verified target positions after pinned Click, MarkupSafe, Pluggy, Rich, and Jinja
  function-hole campaigns. JS/TS is GREEN at `1,612,835` after complete Vite and
  content-bound 600-candidate Tailwind test-kill campaigns; the Tailwind selection
  receipt binds the ordered inventory and cannot use verifier outcomes. Pinned npm,
  pnpm, TypeScript, project lockfiles, canonical order-independent package inventories,
  and build outputs replay offline before verification. HTML/CSS is GREEN at
  `1,906,926` target positions: `517` semantic HTML components pass DOM,
  accessibility, and dual-viewport render deltas, while `173` CSS rules pass a
  JavaScript-disabled dual-viewport render-kill verifier and contribute `58,178`
  positions. The contract binds target behavior, verifier,
  source-disjoint split, license/provenance, contamination, and toolchain/sandbox
  identity; zero public overlap, teacher calls, external inference, and fallback
  returns were observed. Rust is GREEN at `1,162,003` verified target positions, all
  five arm floors pass, and the complete task corpus contains `36,552,344` target
  positions. The separately governed broad corpus is now GREEN at `1,293,454,903`
  unique positions; task targets and optimizer exposure remain separate budgets.
- Add a registered `task_complete_training_unit` to the existing data contract, not a
  new lane. Every supervised unit binds visible context, complete target, task family,
  language/toolchain, verifier or rubric artifact, expected observable behavior,
  source-disjoint split, license/provenance, contamination result, difficulty, and
  outcome class. Units missing their claimed verifier remain language-model data and
  cannot count toward product-task coverage.
- Build the next corpus around the intended product: natural English multi-turn
  conversations, corrections, clarification, and instruction following; complete
  Python tasks and repository edits with tests; JS/TS parse/typecheck/build/runtime
  tasks; HTML/CSS DOM, accessibility, layout, and deterministic render assertions;
  Rust `cargo check`, test, and clippy tasks. Record per-family counts and verified
  position coverage so bulk code or single-turn prose cannot hide a weak product arm.
- Data support for a larger rung requires both unique admitted positions and minimum
  task-complete verifier coverage. More repeated tokens, teacher paraphrases, or
  target-only snippets cannot satisfy the scale gate.
- Before 57.315M training authorization, qualify the admission path as a risk-ranked
  epistemic trusted-computing-base component. Bind source hashes and independent
  golden-invalid oracles for target-pass/starter-fail, contamination, split
  isolation, source-only selection, cache identity, timeout/process cleanup, source
  restoration, final baseline, ledger serialization, and scale-budget separation.
  Run mutation/fault-injection testing over decision branches and report surviving
  mutants and correlated dependencies; raw script-to-test-file ratio is not a valid
  coverage metric.
- The Phase-7 admission TCB is now qualified. It independently replays all `35,887`
  task ledger rows and `9,739` verifier-cache receipts, checks `17` source-only
  selection receipts and `1,407` admitted Rust final baselines, verifies zero cleanup
  residue, and kills `17/17` golden-invalid mutations. It declares three correlated
  dependencies rather than claiming full independence. Canonical admission and the
  57.315M preregistration bind its current task/report/ledger digests; a stale or
  self-declared green report cannot authorize training.
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
- Replace assigned capacity units and hardcoded gate-latency constants with observed
  wall/CPU/GPU/memory/storage, queue delay, human-review minutes, false accepts/
  rejects, failures caught, repair/rework, migration/recovery, and residual cost on
  the daily-use and architecture campaigns. Reconcile route-level ledgers to system
  totals and estimate the marginal value of another check before allocating it.
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
(Owns Tracks 1, 3, and 4. The practical lane is transformer/hybrid unless matched
evidence earns a different route.)

- **Current frozen rung:** v8 uses a 10,347,009-parameter shared transformer trunk
  plus five independently checkpointed 430,849-parameter English, Python, JS/TS,
  HTML/CSS, and Rust specialists. Active sparse parameters are 10,777,601 and total
  sparse parameters are 12,501,254. The dense controls are mechanically matched at
  10,779,648 active and 12,499,968 total parameters. The sparse trunk and specialists
  and both dense controls completed transactionally under the unchanged data, seed,
  tokenizer, decoding, split, and two-view accounting contract. The rung is now
  immutable negative evidence and may not be restarted or patched.

- **Correct interpretation of v8:** the 128-row development records per arm contain
  prompt and target but no executable tests or complete functional verifier. English
  recovered one target exactly; every code arm recovered zero, with Python 41/128
  syntax-valid. These are exact-recovery and syntax diagnostics only. They are not a
  complete utility verdict, cannot prove a practical architecture win, and cannot be
  blended with older assisted/public routes.

- **Frozen diagnostic publication:** complete. Both dense controls are `0/544` exact;
  total-control Python syntax is `48/128` versus active-control `46/128`, while MoECOT
  English alone has one exact recovery. Per-language serialization/syntax, similarity,
  throughput, active/total parameters, optimizer positions, checkpoint identity, and
  cost are published under both preregistered views. This is a systems-canary result,
  not functional capability.

- **Frozen functional utility qualification:** the 160-case contract was frozen on
  2026-07-14 before dense-control completion, with 32 source-disjoint cases per arm,
  a model packet containing only case ID, arm route, and prompt, and hashes for the
  case compiler, independent verifier, toolchains, v8 plan, and stage. After both
  controls finish unchanged, run every
  unchanged sparse/dense checkpoint once. It must bind task IDs, model-visible input,
  toolchain/container identity, timeout, scoring, failure taxonomy, candidate
  integrity, adjudication, and consumed status. No model may influence task inclusion,
  weighting, rubric, or verifier after its output is observed.
  Generation, code verification, blind local scoring, final qualification, and the
  aggregate verdict reserve append-only content identities before execution. Failed
  reservations remain consumed; final qualification reuses the hash-bound preliminary
  code evidence rather than executing code twice.
  - English: multi-turn instruction following, correction, clarification, factual
    grounding from supplied context, calibrated abstention, and conversation
    continuity scored through blinded pairwise/rubric judgments with inter-rater
    agreement. Exact match remains a separate diagnostic.
  - Python: parse/compile plus isolated unit/property tests, return/type and exception
    contracts, repository-edit patch application, and regression tests.
  - JS/TS: parser, typecheck, build, bounded runtime tests, and package/repository edit
    checks under pinned toolchains.
  - HTML/CSS: parse/DOM assertions, accessibility checks, responsive/layout invariants,
    and deterministic screenshot or render-tree comparison where stable.
  - Rust: parse/format diagnostics, `cargo check`, clippy, isolated unit/integration
    tests, and repository-edit regression checks under a pinned toolchain.
  - Report exact recovery and functional utility separately; report every language,
    task family, tail floor, no-output/timeout/invalid rates, accepted output per
    second, and model-only versus assisted channels. English cannot hide code failure.
  - Current state is consumed and complete. Every model scored `0/160` overall and
    `0/32` in each arm. The append-only ledger contains 11 clean reserved/completed
    identity pairs and no failure. Any rerun, reweighting, task replacement, or
    in-place contract edit after these outputs is forbidden.

- **Architecture verdict:** issue the practical verdict only after both frozen
  diagnostics and the functional qualification exist.
  - If a dense control Pareto-dominates MoECOT under either preregistered view without
    a hidden cost regression, route the dense/hybrid successor and retain MoECOT as
    negative/discovery evidence.
  - Retain MoECOT only if it has a repeatable Pareto advantage and no weak-arm floor
    regression; route accuracy cannot substitute for answer utility.
  - If results trade off without a preregistered winner, record unresolved and use one
    untouched confirmation surface, not narrative preference.
  - Confirmation is spent once on the selected unchanged implementation. Route
    replacement then uses one SCF/registry transaction with exact rollback.
  - **Recorded outcome:** `FALSIFY_10_8M_ACTIVE_SCALE_RUNG`; no architecture selected,
    no Pareto winner, confirmation unspent, and route replacement unauthorized.

- **Scale-floor exit:** 10.8M active parameters are a systems canary, not a credible
  ChatGPT-like assistant target. If sparse and both dense controls remain functionally
  zero across code, mark the rung falsified and stop same-scale architecture repair.
  That condition is now met and the rung is closed.
  The next rung must be the smallest data-supported model expected to expose scaling,
  ordinarily 50M-100M active parameters, and must predeclare:
  - scaling-law estimate and falsification condition;
  - unique admitted positions and task-complete verified-unit coverage per arm;
  - matched active/total dense controls and sparse attribution;
  - MLX memory, throughput, checkpoint, resume, and wall/energy canaries;
  - fixed optimizer exposure, development/confirmation splits, and stop criteria.
  Repeated positions, teacher paraphrases, target-only snippets, or optimizer epochs
  do not fund scale. A larger rung is denied if task-complete data support is absent.

- **Frozen post-v8 proposal:** the smallest mechanically matched candidate is now
  preregistered at `57,315,329` active parameters with a `54,811,649`-parameter
  shared trunk and five `2,504,193`-parameter specialist deltas. Dense controls are
  `57,323,520` active and `67,332,096` total, each within one indivisible FF-width
  increment of its sparse accounting view. Earlier mechanics canaries produced finite
  updates, exact checkpoint replay, optimizer resume, and temporary-artifact cleanup;
  the pointer-generator scatter was made deterministic after a canary exposed an
  invalid-index collision. The current hash-bound preregistration is data-GREEN at
  `1,293,454,903` unique positions (`22.557469` per active parameter), every task arm is
  GREEN. The previous package no longer grants training authority because it binds the
  inadequate KERC retirement. Rebuild it only after the faithful KERC alternative and
  its matched conventional-English control are fully implemented and all unchanged
  canaries replay against that contract. This remains mechanics evidence only.

- **Product-aligned supervision:** consume Phase 7 `task_complete_training_unit`
  records. English emphasizes natural multi-turn dialogue, corrections, and useful
  instruction following. Code emphasizes complete tasks and repository edits with the
  same pinned compile/test/render contracts used for private qualification. Licensed
  open data is primary; local dogfood outcomes and verified self-generation follow;
  governed OpenAI teacher rows remain capped residual pressure. Public benchmark
  artifacts remain calibration-only.

- **Amplifier order:** do not optimize a zero numerator.
  1. Establish nonzero model-only functional behavior.
  2. Qualify STS and VCM independently under equal visible context and candidate
     budgets, then jointly only if each causal contribution survives.
  3. Qualify verifier-guided search against one-shot generation, reporting displaced
     verification/repair cost and never crediting deterministic repairs as learned.
  4. Build real accepted/rejected density before DPO/IPO/ORPO/KTO/SimPO.
  5. Run GRPO/RLOO/ReMax/RLVR only with independent verifiers, reward-hacking probes,
     policy leases, drift bounds, and rollback.
  6. Optimize AR, MTP, speculative/self-draft, sketch/diffusion, or other fast modes
     by accepted verified output per second, never proposed-token throughput alone.

- **Architecture freeze:** for the next three capability cycles, do not add a model
  head, marker, router, lane, dashboard, report family, or governance layer unless a
  concrete training/inference failure receipt proves the current registered contract
  cannot express the required behavior. Correctness, replay, security, and measured
  hot-loop fixes are allowed. Every proposed architecture change names the defect,
  matched control, expected utility delta, rollback, and retirement target.
  The KERC correction is the single current exception because the audit proves that
  the frozen package excluded a mechanism using evidence from a different, much weaker
  implementation. This exception closes when the faithful KERC package refreezes.

- **Acceptance:** a practical checkpoint is independently loadable, model-only,
  candidate-integrity clean, functionally positive on source-disjoint English and
  relevant code tasks, confirmation-qualified, and better than its matched baseline
  without hiding weak arms or human/verifier cleanup cost.

- **Do not:** train on public benchmarks; credit templates, n-grams, renderers, tools,
  routers, fallbacks, hidden target metadata, or private-test-derived interfaces as
  learned generation; select an accounting view after results; repeatedly patch a
  falsified scale rung; or call loss, syntax, exact recovery, route selection, or
  infrastructure readiness a utility win.
### Phase 11: Cognitive-Kernel Discovery-Lane Verdicts
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

#### Future Candidate: OneCell-RWM

Canonical source: the vendored
[`docs/research/ONECELL_RWM_CANONICAL_HANDOFF.md`](docs/research/ONECELL_RWM_CANONICAL_HANDOFF.md),
content-bound in the project registry. The original
[shared conversation](https://chatgpt.com/share/6a57781e-17a4-83ea-aaae-fa263a9ac2fd)
is provenance only, not durable implementation authority. This handoff supersedes
the earlier standalone OneCell-RSM packet.

OneCell is a candidate implementation of the learned cognitive-kernel slot, not a
replacement for Theseus. Theseus continues to own intent and authority (VIEA),
durable context and memory (VCM), capability-provider resolution (Octopus),
procedure compilation (Cognitive Loop Closure), qualification/replacement/rollback
(SCF), improvement scheduling (SparkStream), evidence/residual ledgers, and backend
routing (Hive). The frozen v8 comparison now has a recorded scale-falsification
disposition, so OneCell architecture-contract and exact-substrate work is eligible.
It remains non-routeable and may not block the practical transformer/hybrid assistant,
consume v8 outputs as tuning evidence, or become a same-scale patch after the
falsified rung.

**Preregistered hypothesis.** A compact, weight-tied recurrent neural microkernel,
coupled to exact external state, typed query/program execution, explicit
branch-preserving search, predictive world modeling, and verified abstraction
learning, can broaden held-out capability and reduce future verified search cost
without proportional growth of the central neural core. The claim fails if gains
come primarily from a hidden generator, large language adapter, privileged
retrieval, hand-authored skill inventory, deterministic answer-producing tool,
benchmark identity, or unreported search/verification cost.

**Cognitive Kernel ABI.** Add one registered substrate-neutral abstraction only
after the v8 disposition. Sparse Transformer/MoECOT, matched dense Transformer,
SymLiquid, and OneCell must implement the same initialize/propose/accept-receipt/
checkpoint/parameter-accounting contract. Kernel output is a typed proposal, never
a fact or authorized effect. Exact identity, values, hashes, permissions, stack and
program state, provenance, observations, and receipts remain outside latent state.
Beliefs, predictions, relevance, uncertainty, and heuristic value remain explicitly
non-authoritative. Observation, belief, prediction, world state, confidence, and
authority must never collapse into one field.

**Reuse rather than sidecars.** Extend the registered VCM/context ABI with stable
workspace handles, copy-on-write branch overlays, neural-state checkpoint handles,
belief-particle references, query-plan execution, typed memory-fault escalation, and
deletion/supersession closure. Extend existing Semantic IR and deterministic runtime
with typed Query IR (`LOOKUP/FILTER/EXPAND/JOIN/RANK/PROJECT/AGGREGATE/LIMIT`) and
Program IR (typed values, control flow, recursion, exceptions, contracts, resource
limits, capability requests, and effect receipts). Extend existing search,
transaction, replay, rollback, artifact, verifier, and Octopus capability-request
contracts; do not build a second VCM, graph database, tool registry, artifact store,
SCF, benchmark ledger, or general execution spine.

**Research machine.** The initial candidate uses a small fixed workspace with exact
and latent channels; typed perception, working-cognition, planning, world-model,
metacognition, and consolidation lanes; one shared gated sparse-message transition;
low-rank relation/lane conditioning; and operation, pointer, workspace-patch,
prediction, value, uncertainty, progress, value-of-compute, and halt/escalation
heads. The control instruction set is `ACQUIRE/APPLY/EXPLORE/CHECK/COMMIT/STOP`.
Impossible operations are masked by type, authority, resource, and effect policy.
Inner recurrence performs bounded latent refinement; the outer runtime owns exact
search nodes, branch snapshots, backtracking, counterfactuals, verifier-guided
pruning, interruption, replay, and best-supported partial or typed-abstention returns.
The first canary may preregister 32 slots, latent width 384, relation rank 32,
4-64 recurrent training steps, and a 6M-15M core, but these are experimental starting
values rather than theory or adoption constraints. Liquid cells, reservoirs, VSA,
FEP features, and KAN-like transforms remain isolated SymLiquid-derived ablations,
not a mandatory bundle hidden inside the default cell.

**Receipts and learning timescales.** Every proposal resolves to an existing typed
receipt state (`OK/FAILED/BLOCKED/UNKNOWN/TIMEOUT/PARTIAL/CONFLICTING/UNVERIFIABLE`)
with authority, resources, cell steps, search nodes, memory reads, runtime operations,
verifiers, evidence, state deltas, and residuals. Learned critics remain advisory;
durable promotion requires exact, empirical, or appropriate human/domain authority.
Immediate recurrence makes no durable weight changes. Fast adaptive memory is
disposable and non-authoritative. Core/skill/concept updates occur only during
quiescent consolidation from a frozen evidence snapshot through replay, regression,
shadowing, SCF promotion, and rollback.

**Memory and verification hierarchy.** Preserve four separately accounted stores:
the tiny differentiable workspace, exact content-addressed artifact memory,
disposable adaptive neural memory, and the qualified skill/model library. Only the
exact artifact layer and scoped execution/proof receipts may establish durable facts;
adaptive memory and learned critics may propose or prioritize but never silently
authorize, verify, or overwrite provenance. Verification remains tiered from hard
schema/type/permission/resource vetoes, through exact execution/proof, empirical
simulation/property/differential tests, advisory learned critics, and finally
consequence-appropriate human/domain authority. Unknown or unverifiable outcomes
remain typed results rather than fallback answers.

**Training and curriculum contract.** Do not begin with open-domain language or use
the practical assistant corpus as an opaque shortcut. First prove exact substrate
behavior with no neural cell, then oracle-supervised operation/pointer/workspace
updates, train-small/test-large depth and size generalization, costly active
acquisition, recurrence-plus-search, qualified skill use, a frozen-core ratchet, and
world-model planning; English and code adapters come last. The preregistered loss
family must separately account for operation, pointer, workspace patch, value,
uncertainty calibration, transition prediction, halting, recurrence consistency, and
resource prediction. Invalid-action negatives and shorter/longer trajectory
consistency are required. Longer reasoning receives no intrinsic credit, and exact
runtime computation, retrieved answers, adapter capacity, or verifier work cannot be
laundered into neural-core loss or capability.

**Dependency-ordered milestones.** These are one experiment series, not independent
lanes or permission to start open-domain training early:

0. Freeze architecture, ABI, matched-baseline, channel, budget, failure, and
   non-claim contracts after v8 disposition; feature-flag and register the candidate.
1. Implement the exact substrate: typed workspace, Query IR, Program IR, snapshots,
   transactions, receipts, replay/rollback, VCM bridge, and abstract Octopus calls;
   require property/mutation tests and exact artifact identity preservation.
2. Implement the shared typed recurrent cell and serialization; prove operation and
   pointer imitation, stable variable-depth recurrence, and zero exact-state
   corruption before broader training.
3. Preregister train-small/test-large algorithmic generalization on lookup/binding,
   graph traversal, nested expressions, sorting, dynamic programming, symbolic
   execution, stack/queue operations, and active acquisition, with recurrent and
   Transformer controls and no benchmark-identity features.
4. Add anytime evidence-guided search only after recurrence-only evidence exists;
   report value-of-compute calibration, nodes, memory reads, verifier calls, latency,
   energy where available, privacy/risk cost, and the retained recurrence-only arm.
5. Extend existing Loop Closure with trace normalization, typed anti-unification,
   behavioral equivalence, program synthesis/simplification, counterexample-guided
   refinement, shadow routing, SCF qualification, monitoring, and retirement.
6. Run a frozen-core cumulative-intelligence ratchet: allow verified external
   artifacts, queries, skills, and concepts to accumulate while core weights remain
   fixed; require held-out cross-family success or lower future total verified work,
   not near-duplicate replay or renamed trace compression.
7. Add observation/belief separation, action-conditioned prediction, calibrated
   belief particles, hierarchical world-model errors, and model-predictive planning;
   retain model-free search and deterministic-simulator controls.
8. Add English/code adapters last. A Transformer adapter is allowed, but adapter
   size, compute, accessible state, and ablations must prove it is not secretly the
   cognitive kernel. Keep generated language/code model-only evidence separate from
   rendered, retrieved, searched, skilled, or tool-produced output.
9. Integrate checkpoint/candidate gates, SparkStream profile, residual escrow,
   Octopus routing, and SCF replacement only after matched qualification. Possible
   dispositions are primary, narrow-domain, research-only, selected-mechanism merge,
   or rejection; failure is an acceptable and durable result.

**Execution-order boundary.** OneCell milestone 0 and the reusable exact-substrate
parts of milestone 1 may proceed after the recorded v8 disposition, but they are
dependency work, not the flagship capability lane. They may not delay licensed
task-complete corpus growth, the preregistered 57.315M-active transformer/hybrid rung,
or delivery of a useful local assistant. Neural OneCell training begins only after
its exact substrate and matched experiment contract pass; language/code adaptation
begins only after structured generalization, search, frozen-core abstraction, and
world-model evidence exist. Any reusable ABI/runtime improvement remains owned by
its existing SCF abstraction even if the OneCell candidate is later rejected.

**RATCHET-100 contract.** Build generated hidden sequential task families with exact
verifiers, increasing size, related but non-identical reusable structure, and a fixed
core during the main sequence. Compare core-only, transcript memory, exact artifact
memory, matched search, executable skills, abstraction compiler, and full Theseus
channels. Measure success by sequence position, out-of-distribution size/depth,
cell steps, search nodes, query/memory/program/verifier cost, permanent bytes, skill
reuse and false activation, rollback, maintenance/migration burden, and net future
verified work avoided. No answer replay, benchmark-name routing, or hand-authored
task-family solution catalog is admissible.

**Matched evidence and total-cost rule.** Report `CORE_ONLY`, `CORE_PLUS_MEMORY`,
`CORE_PLUS_SEARCH`, and `FULL_SYSTEM` independently. Match governed data, tokenizer
where applicable, active and total parameters, optimizer positions, adapter size,
wall/FLOP budget, memory, tool/search/verifier access, and public-calibration policy.
Publish both preregistered active- and total-parameter views. Total cost includes
training, inference, search, memory, verification, skill compilation, routing,
maintenance, migration, governance, and human cleanup. OneCell is retained or
promoted only for repeatable Pareto value without weak-tail regression; it is revised
or rejected when depth generalization, pointers, exact-state integrity, query recall,
search economics, shared-weight interference, frozen-core gains, world-model planning
utility, or matched total task cost fails.

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

#### Kernel English and hierarchical residual compiler extension

Author-source intake: *Kernel English with Hierarchical, Interaction-Amortized
Residuals* (KERC, July 2026), content SHA-256
`f560c61196cb2a114475ebd455f8643536e78c82dbbf6ec8dd712d993f2b6519`.
The canonical source is currently under
`AI_book/sources/raw/kernel_english_residual_compiler/`; until the book chapter
and source note are committed, this is an argument-backed intake contract rather
than evidence that the architecture works. KERC extends the existing English arm,
Semantic IR, VCM, SCF/registry, evidence/compression, and Octopus owners. It must
not become a second memory system, tokenizer lane, renderer shortcut, or hidden
deterministic answer path.

**Corrected pre-training disposition (2026-07-17):** KERC is `partial` and the prior
retirement is withdrawn. `p4-m8-kerc-runtime-001` replayed its own frozen design, but
that design was not a faithful KERC implementation: it used 128 authored templated
training rows, a hand-coded keyword compiler and renderer, and a TF-IDF/L2 logistic-
regression intent-plus-polarity classifier. It did not implement or train the
constrained Surface-to-Kernel transducer, Kernel-native reasoner, learned four-level
residual allocator, structured answer generator, copy-aware learned renderer, or
neural round-trip verifier. Its `0.5` accuracy, `714`-byte packet cost, and one attack
escape are retained as negative evidence for that exact toy proxy and as useful
instrument defects. They are `INCONCLUSIVE_IMPLEMENTATION` for KERC and cannot retire
the architecture. The previous freeze package is stale until the work below is real,
integrated, mechanically replayed, and prospectively frozen.

**First faithful neural mechanics tranche (2026-07-17):** the canonical MLX model and
MoECOT trainer now expose `english_kerc` as a separately checkpointed candidate and
`english_surface_control` as a conventional encoder-decoder control. KERC alone is
charged for fixed-capacity V_K/V_P code spaces, producing a `13,315`-symbol vocabulary
versus the control's `8,195`; independent FF-width matching leaves a 2,603-parameter
delta (`19,531,487` versus `19,534,090`). KERC has trusted source-only stage
selection, stage adapters, separate Kernel/surface output paths, a four-channel
residual classifier, an independently parameterized raw-token verifier, copy access,
content-bound mechanics labels, one targeted semantic/object/value/fidelity/decision corruption
per positive view with zero generator loss, joint residual/verifier losses, cached
decode, and strict checkpoint reload. The stage checks both positive and corruption
representability before publishing. V_K/V_P are fit only on positive private-train
compiler/core targets; development/evaluation rows and verifier corruptions contribute
zero vocabulary-fit signal. Compiler/core logits are restricted to V_K/V_P plus EOS,
while direct/renderer logits are restricted to V_S plus EOS. Vocabulary identity and
size are checkpoint/resume invariants. The KERC evaluation route now executes the
stage-routed shared-trunk compiler, Kernel reasoner, renderer, surface recompiler, and
second reasoner in sequence. These are executable trainable paths, not yet independently
capable learned modules. The route parses and validates every packet,
requires answer handles to resolve against packet-owned objects/capsules, and rejects
the candidate unless structured round-trip constraints match; the direct-surface head,
templates, literals, tools, and fallback returns cannot satisfy this route. The live
plan has ten explicit targets and the governed stage now replays GREEN.
This is architecture mechanics, not a learned KERC system or capability result. The
semantic admission owner separates audited/licensed human gold from local-parser
silver and governed OpenAI residual evidence. Only gold can occupy private-dev/eval or
satisfy decision-grade floors. Silver and teacher residual rows are train-only, carry
maximum sampling weights of `0.25` and `0.02`, cannot support semantic claims, and are
subject to record-share and optimizer-probability caps. Every record must bind a
content-addressed dataset revision/license/source catalog; public semantic benchmark
surfaces are rejected even when permissively licensed. The MLX stage consumes these
weights for positive and verifier-only pairs. The source qualification ledger keeps
UCCA/TUPA, SNLI, and PMB calibration-only, with PMB 5.1 annotation and raw-text
license policy reviewed but every PMB split forbidden for model training; LDC AMR remains excluded. A separate
producer and independently implemented raw-source replayer now admit 6,144 Dolly
instruction/response records for the direct-surface objective only and 4,096 manual
MASC FrameNet records for compiler/reasoner/renderer objectives only. Another 1,064
bounded OASST2 records carry multi-turn dialogue context and exactly two distinct,
reviewed rank-0/rank-1 human answers per prompt across all four objectives. The same
raw-source replayer reserves 24 source-disjoint records whose reviewed human surface
explicitly clarifies or abstains. This supports narrow behavior supervision only; it
does not establish optimality, truth, semantic completeness, or calibration. Adjacent
human-authored document sentences supply journaled, independently replayed VCM
interaction state to 1,997 train, 998 development, and 1,015 evaluation MASC records.
Compiler, reasoner, and renderer consume a bounded interaction ABI; cross-user reuse,
privacy widening, unjournaled or tampered deltas, and context overflow fail closed.
The same independently replayed Dolly source contributes a separate 96-record stratum
(`64/16/16` train/development/evaluation) in which a natural question's answer is a
unique bounded contiguous span of its source context. Every split covers `what`, `who`,
`where`, `when`, `how many`, and `which`; rare forms are reserved before larger-split
balancing so train selection cannot starve heldouts. These records authorize all four
learned objectives, bind context through the VCM interaction ABI, and carry explicit
`ANSWER`/`SUPPORTED`/`RESOLVED` decisions. The compiler input sees the question, form,
context identity, and governed context but never the evaluator-owned answer span.
Independent raw-source replay reconstructs eligibility, split assignment, support span,
Kernel program, VCM state, answer claim, and decision. This is narrow extractive-source
support only; it does not establish broad entailment, truth, completeness, reasoning,
calibration, or model quality.
The selected private-train MASC rows also fit a source-bound contextual FrameNet prior
for 59 lexical units with at least two manually annotated frames. Typed ambiguity values
carry those alternatives through the Kernel program and answer packet for 411 train,
178 development, and 157 evaluation records. The independent verifier reparses the raw
GrAF archive, reconstructs selection, fits the prior from private-train rows only, and
rejects changed alternatives, weights, or content identities. Development/evaluation
labels cannot influence the prior. These empirical weights are source-train frequencies,
not calibrated probabilities; the stratum proves contextual sense supervision and
replay only, not unresolved ambiguity, WordNet equivalence, or learned disambiguation.
K1b adds a second separately accounted 256-row MASC stratum (`128/64/64`) selected
deterministically from manual committed-belief, MPQA subjectivity, and event layers.
The producer aligns raw GrAF spans to source sentences and emits typed
`EPISTEMIC_STATUS`, `EVENT_*`, and `SUBJECTIVITY_*` nodes. Observed categorical,
boolean, list, polarity, intensity, and temporal-orientation fields contribute `3,666`
nonliteral typed arguments; source expressions and unclassified textual fields remain
byte literals. The independent implementation reparses the raw annotation graphs,
reconstructs selection and every expected K1b record, and admitted the then-current `11,936` canonical
rows with the same `sha256:c8502fbd5c2d4628aee41a1ad380d7b9f8797993ba2feb8a8ce318655a94ac49`
candidate identity and zero verification failures. Its first full replay correctly
rejected 28 multi-claim rows because default decision hashes were not idempotent above
nine claims; claim-id canonicalization and a regression test now close that protocol
fault. This evidence covers only the observed labels and spans. K1b did not recover
event-coreference grouping, resolve source-declared cross-annotation links, establish
scope or truth, provide complete sentence semantics, or demonstrate learned competence.
Full materialization took about 21 minutes and independent replay about 18 minutes on
this Mac, making content-addressed producer/verifier layer caches an immediate K1
performance requirement rather than optional cleanup. K1c now adds separate exact-run
cache receipts whose keys bind actual source files and the complete extracted MASC tree,
configuration, owner code, protocol/economics/VCM dependencies, and every output hash.
Cold/warm measurements are `1321.33s/1.86s` for production and `1100.62s/1.93s` for
independent verification, roughly `710x` and `570x` repeated-run speedups. Output or
receipt mutation and dependency changes fail closed. This does not complete the
incremental requirement: any changed dependency still rebuilds the whole owner path;
raw parse, split, packet, and economics layers still need separate producer/verifier
keys and selective invalidation. K1d now closes two of those selective layers with
namespace-separated canonical SQLite object caches and transaction-batched durable
writes. Producer structural-economics keys bind the complete program, VCM/global state,
all residual inputs, exact objects, codec, calibrated importance policy, lambda, and
implementation identities. Independent-verifier semantic-admission keys bind the full
candidate, independently reconstructed expected row, and verifier/protocol/economics
implementations; producer authority is never reused. Full-corpus cold versus forced
selective-hit replay preserved byte-identical outputs: production improved from
`1151.89s` to `802.88s` with `11,936/11,936` hits and zero misses, while verification
improved from `997.70s` to `71.47s` with `11,936/11,936` hits, zero misses, and zero
producer authority reuse. Exact-run replay remains `1.70s/1.84s`. Corrupt and
cross-namespace objects fail closed, and a dependency mutation retains two unaffected
hits while recomputing exactly one changed object. This is not complete incremental
replay: raw parsing, split reconstruction, packet compilation, aggregate calibration,
and serialization remain measured open owners.
K1e now reconstructs 13 complete manual named MASC event-coreference groups containing
215 mentions (`8/3/2` groups and `133/24/58` mentions across train/development/evaluation).
The producer uses global token-sequence alignment while the independent verifier uses
exact local sequence matching with a context-margin requirement. They produce identical
admitted rows. Two incomplete groups are rejected in full; partial-group admission and
co-occurrence inference remain zero. A deterministic uniform-radius, mention-centered
source-window contract retains every admitted mention under the unchanged 2,048-character
input bound, including a 75-mention group, with a minimum observed context radius of nine
characters. Derived group records receive zero additional source credit and explicitly
deny truth, causal, temporal, and complete-sentence-semantic authority. The resulting
11,949-record corpus has 525 multi-node and multi-claim records, of which 512 are
multi-root. Full materialization and independent verification pass with zero failures;
unchanged exact replay takes `2.16s/1.87s`. This is source-grounded relation and replay
evidence only, not general coreference, learned semantic competence, utility, or KERC
thesis evidence.
K1f now reconstructs 846 complete manually linked MASC MPQA attribution graphs from
1,459 direct-subjective expressions (`713/93/40` train/development/evaluation). The
producer uses a regex attribute parser while the independent verifier uses a separate
state-machine quoted-field parser; both re-read the original annotation files and exact
source texts and agree on every admitted row. Each graph retains one expression, its
ordered nested-source memberships, one or more linked attitudes, and every linked target.
The ABI distinguishes explicit spans, declared implicit members, and source-authored
zero-width annotations; repeated source-chain positions remain ordered edges rather than
being silently deduplicated. Exactly 613 incomplete or ambiguous chains are rejected as
whole relations: 67 attitude, 304 source-agent, 51 target, one missing source-chain, and
190 missing target-link failures. No link is inferred, no partial relation is admitted,
and derived rows receive zero additional unique-source credit. The resulting 12,795-row
producer and verifier runs are GREEN with zero verification failures, zero public payload,
zero external inference, and zero fallback/template credit. This establishes bounded
manual attribution-link replay only. It does not establish proposition scope, truth,
causality, temporality, complete sentence semantics, learned semantic competence, utility,
or the KERC thesis. Changed full-corpus runs still take roughly 23 minutes per side, so
packet compilation, aggregate calibration, serialization, and finer selective reuse remain
real performance owners.
Source-disjoint heldouts, per-objective authority, archive/revision/license hashes,
and a verification ledger are mandatory. Duplicate reviewed replies, answer leakage,
compiled-context overflow, and producer-only identity claims fail closed. The prior canonical
stage was GREEN with 26,360 authorized views and 26,360 verifier corruptions over 11,424
source-bound records; derived
views receive zero unique-data credit. A bounded
Apple-MLX KERC-only step consumed KERC target positions with nonzero residual and verifier
auxiliary weights and published a reloadable checkpoint. Dynamic staging and batch
cropping raised that path from `2.648` to `39.552` target tokens/second. The subsequent
content-bound adequacy run used exact-source real-row pairs across all four objectives
and 64 full-size MLX updates. MASC manual named-entity graphs are now independently
replayed alongside FrameNet: 3,070 records carry 7,220 typed person/place/organization/
date protections, and 395 exactly aligned semantic arguments use packet-owned handles
rather than byte literals. Learned stages now consume a compact least-privilege object
view while full hashes, authority, provenance, and access policy stay in the packet and
evidence plane. The genuinely rematerialized stage retains all 26,360 positive views
and 26,360 zero-generator-loss negatives without unknown-token substitution or
truncation; the maximum staged sequence is 4,243 tokens under an 8,192-token contract.
Lossless UTF-8-safe splitting handles oversized structured atoms while preserving the
codec's per-span safety bound. Manual FrameNet targets, roles, and named entities now
provide genuine source-bound segment and token annotations rather than an all-zero
proxy. The current neural target still reduces them to channel-presence classes and
therefore is not per-unit fidelity supervision. The first balanced audit caught that raw
verifier bit accuracy rewarded an
all-positive classifier; the canonical trainer now derives content-bound inverse-
frequency weights per verifier dimension and fails closed when either class is absent.
The canonical answer packet now carries explicit answer/partial/clarify/abstain,
evidence, uncertainty, confidence, controlling-claim, and unresolved-ambiguity fields.
Missing or inconsistent decisions and unresolved-correction certainty laundering fail
closed. The independently parameterized verifier now has a fifth learned output for
answer-decision consistency rather than leaving that contract in deterministic validation.
The source-grounded question stratum now also produces `384` verifier-only context
interventions: context withheld and source-disjoint context shuffled for the direct and
Kernel-to-answer objectives in every split. Donor contexts exclude the original answer;
shuffled program and answer-packet hashes are rebuilt so stale checksums cannot become a
shortcut. These rows fail only semantic and answer-decision consistency, receive zero
generator loss, unique-source credit, or candidate-generation credit, and establish only
counterfactual support sensitivity. Under this expanded objective, the current cold-start
Apple-MLX canary's deterministic 16-row subset
covers all `21` required objective, answer/clarify/abstain, interaction, residual,
five-dimension verifier, and withheld/shuffled-context groups. Token accuracy reaches
`0.338462`; verifier macro balanced accuracy is `0.812261` with `1.0` minimum negative
recall; all four informative residual channels remain at `0.5` macro balanced accuracy
after 64 mixed-objective updates. This cold score is diagnostic only. Verifier-only and
context-counterfactual rows now have an explicit zero residual-authority mask. A
same-model channel-presence overfit rung
records nonzero residual-head and source-representation gradients and reaches `1.0`
minimum channel balanced accuracy at step 128 of its 256-step bound. This proves that
the head can memorize the mechanics labels, not that it learned rate-distortion allocation.
Explicit trusted-stage
conditioning prevents long sources from washing out the stage signal, and all four
channels reach `1.0` after 64 additional canonical joint updates. A separately seeded
source-conditioned disposition probe reaches and retains `1.0` balanced accuracy over
observed ANSWER/CLARIFY/ABSTAIN by 64 updates; PARTIAL is absent and unclaimed. The complete mechanics
replay is GREEN. Checkpoint and
optimizer reload are exact; resumed logits differ by at most `2.05040e-05` under an
absolute-plus-relative float32 equivalence contract with unchanged discrete outcomes;
real NPZ-to-safetensors migration and rollback produce zero logit drift; unknown schema
and stale codebook identities are rejected; partial artifacts are absent; and trusted-
stage, full-residual, interaction-only, verifier, and trained leave-one-mechanism-out
controls execute. This still has no utility or negative-verdict authority. The prior
adjacent-document target plus the new counterfactuals prove interaction representation,
training-path consumption, replay, and causal testability, not that prior context improves
answers. The new OASST2 rows establish source-bound
multi-turn, multiple-valid, and explicit-decision supervision, while the Dolly question
  stratum establishes unique extractive support, while the train-only FrameNet stratum
  establishes contextual alternatives for resolved human labels; none proves semantic
  equivalence, context benefit, unresolved ambiguity, broad entailment, truth, calibrated
  confidence, or learned question-answer quality. Unresolved ambiguity/broad-entailment,
protected-span abstention, calibrated verifier behavior, full-size multi-seed
resource evidence, and the matched utility campaign remain open.

**Residual-economics adequacy correction (2026-07-17):** the prior four-channel
labels exercised hierarchy and gradient flow, but did not implement the paper's
rate objective. Kernel packets now bind an exact adaptive order-1 arithmetic codec
conditioned on the Kernel plus higher-level residual state. Its four channel receipts
carry encoded payloads, condition/content hashes, exact replay, encoded bits, and
uncompressed bits. VCM promotion requires a replayed strict break-even receipt for
`b_def + m*b_ref < m*b_direct`; an unauthenticated or premature promotion aborts the
transaction. The corpus now measures actual semantic/faithful/lexical/exact structural
candidates, computes `argmin(bits + lambda*w*distortion)`, and enforces exact fidelity
for protected content. Lambda is selected only from a frozen grid on private-development;
private-evaluation and public data cannot influence it. The source-visible importance
policy is fit on private-train preservation labels, calibrated on private-development,
and independently replayed on source-group-disjoint private-evaluation records. The
canonical validator rejects `source_fidelity_v1` as record-level allocation authority.
Its `labels_by_channel` training target remains a separate structural-presence heuristic
and must also be retired from allocator claims.

This closes the missing exact-codec and promotion-economics baseline and adds an
independently replayed record-level structural controller, but it does not close the
learned allocator or prove the architecture thesis. Importance
currently predicts source-bound structural preservation, not downstream answer utility or
per-unit entailment sensitivity. Allocation is packet-record-level, its distortion is
omitted structural mass, and one protected object hard-promotes every residual channel in
the record to exact. The neural head is still trained on channel presence, not these
selected fidelities. On private evaluation, selected fidelities are semantic `992`, faithful
`2`, lexical `236`, and exact `858`; semantic/surface/identity weak-tail recalls are
`0.997338`/`1.0`/`0.975163`. Exact replay passes, but residual bits are `5.114718x` source
bits and total Kernel-plus-residual wire is `9.998350x` source bits. The canonical stage
build takes `1365.17` seconds. These negative cost measurements remain visible. The next
faithful step is per-unit intervention-sensitive allocation and total-cost optimization,
followed by the frozen matched campaign. Any learned/neural entropy-codec challenger must
beat the exact finite-state baseline under identical payloads with parameters, latency,
memory, energy, state synchronization, and recovery charged. No codec or allocation
receipt receives learned-language capability credit.

**Construct-validity review (binding current boundary, corrected 2026-07-18):** a
fresh source-and-artifact audit found that the protocol surface remains substantially
ahead of the evidence surface. Across `27,941` canonical records, `16,517` programs are
multi-node, `6,202` are multi-root, and `7,061` answer packets are multi-claim. Thirteen
manual event-coreference groups preserve 215 independently aligned mentions; 846 manual
MPQA graphs preserve complete expression/attitude/target/ordered-source links; 9,785
GUM eRST records preserve human discourse topology, including 339 source-declared
secondary edges and all 32 admitted relation types in each private split; and 5,361 GUM
entity/coreference records preserve 3,542 complete non-singleton components plus 1,819
typed bridging links. The latter exercise `7,180` document-local concept capsules behind
a learned-output ABI that excludes source identities, annotation hashes, stable registry
identity, and provenance. These improve bounded relation and reference coverage but do
not establish ontological or cross-document identity, proposition scope, truth,
quantification, compositional negation, temporal truth, or complete semantics. There are
`18,759` opaque `byte_literal` values, zero correction alternatives, zero nonempty macro
registries, zero nonempty interaction-global dictionaries, zero `PARTIAL` decisions,
and zero per-unit allocation receipts. `6,809` rows still receive a nonzero
interaction-channel label while the coded global dictionary is empty. The hard
round-trip probe does not reject byte-literal substitution. These facts do not invalidate
packet, codec, replay, or lifecycle mechanics, and the human MASC/GUM strata are real
construct-validity improvements. They do invalidate claims of complete semantic
compilation, compositional Kernel reasoning, learned per-unit allocation, true
interaction amortization, broad semantic round-trip fidelity, or KERC utility. High
importance-policy scores remain label-recovery diagnostics because the targets are
source-structure preservation classes closely exposed by packet features; they are not
downstream entailment-sensitivity evidence. The last staged MLX package still contains
`11,424` records and is stale relative to this `27,941`-record corpus; it must be rebuilt
after K1 closes, not silently treated as evidence for the newer semantic surface.

**Better-than-paper rule:** the paper defines the hypothesis and minimum faithful
baseline, not an implementation ceiling. A stronger constrained optimizer, packed ABI,
neural hyperprior, graph-equivalence objective, uncertainty policy, or modular/shared
hybrid may replace a paper mechanism when it is registered as a distinct implementation,
preserves the same information and hard constraints, and wins a matched ablation. Do not
silently change the mechanism under the KERC name or handicap the paper-faithful control.

**2026-07-18 review conclusion:** the present implementation is neither a throwaway
fake nor a decision-grade KERC system. The protocol, exact-object path, arithmetic-code
baseline, VCM transaction mechanics, split discipline, independent raw-source replay,
and checkpoint/migration path are real. The current learned path is still an
underpowered shared-trunk mechanics candidate whose allocator target, semantic breadth,
interaction state, renderer evidence, and verifier independence are insufficient for a
utility verdict. Future work must improve these owners in place through the registered
abstraction/implementation records. Do not add a `v2` side lane, and do not translate
failure of the present head, corpus slice, or teacher-forced route into failure of the
paper mechanism.

**Binding K0-K8 closure sequence (complete in order):**

1. **K0 - independent fidelity map and claim reset (`GREEN`, 2026-07-17).**
   `configs/kerc_implementation_fidelity.json` now maps all Sections 5-10 mechanisms
   and H1-H8 to `faithful`, `stronger_registered_alternative`, `approximate`,
   `fixture_only`, `inactive`, or `absent`, with source-bound code/test/evidence refs
   and explicit claim ceilings. `scripts/kerc_implementation_fidelity_gate.py` independently
   scans all 27,941 corpus records and directly probes the channel-label and byte-literal
   verifier behavior; the canonical roadmap gate consumes its result. The current map
   retains only three narrow faithful-mechanics classifications, records ten
   approximate/fixture mechanisms and three inactive/absent mechanisms, and keeps H1-H8
   inactive. Mutation tests reject relabeling the missing per-unit allocator, empty
   interaction dictionary, absent macro path, single-node compiler, or untested
   hypothesis as faithful. The present route is a shared-trunk KERC mechanics candidate,
   the arithmetic codec is an exact finite-state baseline, the current allocator is
   record-level structural allocation, and the neural residual head is channel-presence
   prediction. Historical content-bound reports remain immutable evidence for their exact
   runs but are subordinated to this later adequacy audit; no current registry, roadmap,
   gate, freeze, promotion, or summary report may exceed the K0 ceiling. This GREEN
   status means the claim boundary is honest, not that KERC is faithful, useful,
   efficient, or scientifically falsified.

2. **K1 - decision-grade semantic representation and corpus.** Replace wrapper-shaped
   supervision with licensed, independently reviewed examples that actually populate
   multi-node/multi-clause Kernel graphs, scope, conjunction/conditionals, compositional negation,
   modality, quantification, temporal and causal graph relations, attribution links, coreference, unresolved
   ambiguity, correction alternatives, concept capsules, source alignment, typed values
   and units, and grammar-aware macros. Include adversarial near-equivalents and multiple
   valid graph/answer realizations. Opaque whole-answer byte literals may train only the
   direct-surface or exact-copy path; they receive zero compiler, core-reasoning,
   semantic-verifier, or answer-packet authority. Semantic answer packets must expose
   propositions and typed arguments rather than hide the answer in one literal. Fit
   train-only transforms and vocabularies on source-group-disjoint data; development and
   evaluation targets come from independent human or otherwise decision-grade owners.
   **K1a evidence (`GREEN`, 2026-07-17):** the canonical corpus now includes 128 train,
   64 development, and 64 evaluation records selected from source-disjoint MASC
   documents where one licensed sentence has 2-8 manual FrameNet annotations. The
   producer emits one root/node/claim per manual frame through the registered
   `framenet_composite_v1` residual ABI. A separate verifier reparses raw GrAF, rebuilds
   split membership and frame grouping, and admitted the then-current 11,680 corpus rows with zero
   failures. Composite rows earn zero additional unique-source credit and explicitly
   deny inter-frame discourse-edge, truth, reasoning-utility, and complete-sentence-
   semantics claims. This closes the old all-single-node/all-single-claim corpus defect;
   it does not close K1 because FrameNet co-presence alone supplies none of the broader
   relations or learned semantic competence listed above. Tampered frame claims and
   source credit fail focused tests.
   **K1b evidence (`GREEN`, 2026-07-18):** another source-disjoint `128/64/64`
   MASC stratum binds manual committed-belief, MPQA subjectivity, and event annotations
   to typed Kernel nodes and multi-claim answer packets. It contributes `3,666` typed
   nonliteral arguments for observed modality, polarity, temporal orientation, event,
   and subjectivity fields. The independent verifier reparses raw GrAF and reconstructs
   selection, layer missingness, arguments, packet identities, and the then-current `11,936` rows.
   The first full replay rejected 28 records because default decision hashes were not
   idempotent for ten or more claims; canonical claim ordering and a regression test
   fixed the protocol defect, and the final replay is GREEN with zero failures. K1b supplied no
   cross-annotation links, event-coreference groups, scope, truth, complete semantics,
   or learned competence are claimed. The corpus still has `18,759` byte literals.
   **K1c exact-run cache evidence (`GREEN`, 2026-07-18):** producer and verifier use
   separate role receipts and retain independent semantic code. Exact unchanged runs
   improve from `1321.33s` to `1.86s` and from `1100.62s` to `1.93s`; source, config,
   code, candidate, output, or receipt mutations invalidate reuse. Changed inputs still
   cause monolithic work. Extend the cache into pinned raw-source parsing, split reconstruction, packet
   compilation, and economics, with dependency hashes and mutation tests proving that
   changed source/config/protocol/verifier inputs invalidate only affected layers. A
   cached run must preserve byte-identical outputs and independent replay; only a
   previously completed independent verification transaction with the exact same bound
   identities may be reused, and producer-owned artifacts cannot grant admission.
   **K1d selective replay-cache evidence (`GREEN`, 2026-07-18):** namespace-separated
   canonical SQLite caches now bind producer structural-economics objects and verifier
   semantic-admission objects to their complete role-specific dependencies. Full-corpus
   cold/selective-hit production improved from `1151.89s` to `802.88s`; independent
   verification improved from `997.70s` to `71.47s`; both paths recorded
   `11,936/11,936` hits, zero misses, byte-identical outputs, and zero producer-authority
   reuse. Exact-run replay remains `1.70s/1.84s`; corruption, cross-namespace reuse, and
   targeted dependency mutation fail closed. This is performance/replay evidence only.
   Pinned raw parsing, split reconstruction, packet compilation, aggregate calibration,
   and serialization still require independently keyed selective invalidation.
   **K1e source-grounded relation reconstruction (`GREEN`, 2026-07-18):** 13 complete
   manual MASC named-event coreference groups containing 215 mentions are reconstructed
   from the original GATE annotation sets that the converted GrAF layer discarded.
   Producer global alignment and independent verifier local-context alignment agree
   byte-for-byte. Every mention must align or the complete group is rejected; two
   incomplete groups are wholly excluded. Deterministic mention-centered source windows
   preserve every admitted mention under 2,048 characters, including a 75-mention group,
   while retaining at least nine source-context characters around each mention in the
   observed corpus. Partial groups, co-occurrence-inferred edges, truth, causality,
   temporality, and complete-sentence semantics receive no authority. Derived group
   records receive zero additional source credit and carry explicit missingness. The
   11,949-record producer and independent verifier runs are GREEN with zero failures;
   unchanged exact replay takes `2.16s/1.87s`. Whole-group rejection, long-group
   compaction, duplicate identities, changed spans, altered membership, and source-credit
   inflation have focused negative tests. This closes only the first K1 relation family,
   not general coreference or learned competence. Follow with licensed human gold
   for scope, quantification, conditionals, compositional negation, causal/temporal
   relations, attribution, unresolved ambiguity, correction lattices, concept capsules,
   quantities/units, and grammar-aware macros; do not synthesize these labels from the
   same rules that will evaluate them.
   **K1f source-grounded MPQA attribution reconstruction (`GREEN`, 2026-07-18):**
   846 complete relations are reconstructed from 1,459 manually linked direct-subjective
   expressions in the original MASC MPQA files. Independent regex and state-machine
   parsers agree on all rows and on the `713/93/40` split. Every admitted record preserves
   expression-to-attitude, attitude-to-target, and ordered nested-source links. Repeated
   source positions, declared implicit members, and zero-width annotations are represented
   explicitly. Exactly 613 incomplete or ambiguous relations are wholly rejected with a
   frozen reason ledger; partial admission, inferred links, and additional source credit are
   zero. Full production and independent verification admit all 12,795 canonical rows with
   zero failures. Corrupted edges, annotation receipts, source credit, missing members,
   duplicate identities, malformed spans, parser disagreements, and count drift fail tests
   or hard gates. This closes one bounded attribution family only; scope, truth,
   causal/temporal relations, complete semantics, and learned competence remain unclaimed.
   **K1g source-family selective replay and finalization cache (`GREEN`, 2026-07-18):**
   producer and verifier identities are independently derived from transitive top-level
   function/constant source closures for nine source families. The verifier reparses the
   producer source without importing or executing it. Candidate objects bind the complete
   family identity and provisional state; finalization objects bind candidate hash,
   source id, calibrated importance, allocation, selected lambda, and the finalization
   source closure; semantic-admission objects bind the independent expected row plus
   verifier-common and family closures. On all `12,795` rows, warm production takes
   `65.089s` instead of the prior `802.88s` selective-hit path (`12.335x`) with full
   candidate/finalization/economics hits and an unchanged candidate digest. Independent
   verification takes `71.059s` after a `1,354.142s` cold pass (`19.057x`) with full
   semantic hits, zero misses, zero failures, and zero producer-authority reuse. An
   interrupted fill resumes from `3,456` valid candidate objects. A telemetry-only source
   edit outside every semantic closure preserves all objects; family-local and shared-helper
   mutation tests invalidate only the declared dependent families. Cache corruption and
   namespace crossing fail closed. The measured cost is `2.369 GiB` of SQLite cache and
   about `2.64 GB` maximum resident memory. This is replay/performance evidence only and
   grants no semantic, learned, utility, compression, or KERC-thesis claim. Raw parsing,
   split reconstruction, aggregate calibration, and serialization remain open incremental
   owners. Evidence: `reports/kerc_selective_replay_k1g.json`.
   **K1h licensed human discourse topology (`GREEN`, 2026-07-18):** the pinned
   GUM V12.1.0 source at commit `22fdf87f9c71c96bcc771461d06e689b1f90020d`
   contributes 9,785 eRST edge-neighborhood records from 89 permissively licensed,
   official-train documents. Academic, biography, court, interview, news, and voyage
   genres retain their exact underlying-text licenses; noncommercial and restricted
   genres are excluded. The official dev, test, and test2 partitions are quarantined,
   then 65/12/12 admitted documents are frozen into source-group-disjoint private
   train/development/evaluation splits. Producer ElementTree/tab parsing and independent
   verifier Expat/CSV state-machine parsing reconstruct identical source-declared primary
   dependencies and 339 secondary edges. Every split contains all 32 admitted relation
   types; minimum per-type counts are 4/5/5, so this establishes breadth and weak-tail
   visibility rather than statistical adequacy. The typed ABI preserves EDU spans,
   direction, relation, nuclearity where declared, edge kind, source hashes, and signal
   counts. It never infers a relation from text, gives derived views zero unique-source
   credit, and authorizes only compiler/core objectives. The current corpus therefore has
   22,580 records, 11,156 multi-node programs, 841 multi-root programs, and 1,700
   multi-claim answer packets. This closes bounded source-grounded causal/result/cause,
   condition, contrast, attribution, evidence, elaboration, purpose, sequence, and other
   discourse-topology supervision. It does not establish proposition scope, truth,
   quantification, compositional negation, general coreference, temporal truth, complete
   sentence semantics, public GUM/DISRPT performance, learned competence, or KERC utility.
   Focused malformed-edge, content-digest, official-heldout, split-overlap, topology,
   source-family, false-nuclearity, license-substitution, and answer-node-reference
   controls pass. Kernel-program edges use internal node references, while answer packets
   use stable source-derived EDU concepts so they remain self-contained under the shared
   anti-leakage validator. The first independent full replay rejected all GUM rows when
   aggregate license authority was incorrectly substituted for per-record licenses; the
   second rejected all GUM rows when answer claims pointed into the internal Kernel graph.
   Both failures remained RED until family-local source binding and the self-contained
   answer ABI were repaired. The resulting full independent replay admits all 22,580 rows,
   reproduces the canonical digest, and has zero verification, allocation, or residual-
   accounting gaps. A selective producer pass rebuilding 9,785 GUM rows takes 1,305.450s;
   a selective verifier pass independently admitting those rows takes 2,197.806s. Warm
   producer/verifier replay takes 152.365s/176.954s with 22,580/22,580 keyed hits, zero
   misses, zero verifier failures, and zero producer-authority reuse. The cache cost is
   4,983,504,896 bytes (4.641 GiB). Full independent corpus replay is the admission
   authority; producer telemetry alone cannot admit rows.
   **K1i licensed human entity/coreference topology (`GREEN`, 2026-07-18):**
   the same pinned GUM revision contributes 5,361 records: 3,542 complete
   non-singleton source-declared coreference components and 1,819 typed bridging links
   with complete endpoint components. Independent TSV reconstruction agrees byte-for-
   byte. Native CoNLL independently confirms exact mention-to-component membership in
   all 65/12/12 private-split documents, while CoNLL-U independently confirms 24,162
   mentions and 13,436 components. The four-format source digest is
   `sha256:d23b29d2dc73320fb14ab13fb6e53e4fde3d3ef6004d4b94c199e7932d4a9b96`.
   These rows exercise 7,180 provenance-bound, document-local concept capsules and add
   typed mention, component, relation, and answer structures. Source-derived capsule
   labels are absent from compiler prompts. Learned targets contain only actionable
   capsule semantics; source IDs, annotation hashes, stable registry identities, and
   provenance remain in the evidence plane. Learned output is forbidden from minting
   those authority fields, and deterministic validation materializes packet-local,
   non-promotable identities before checking every handle. The model therefore learns
   grouping and semantic binding rather than impossible bookkeeping. Derived views earn
   zero unique-source credit and partial admission, inferred links, public-score claims,
   fallback/template credit, truth claims, and learned-competence claims remain zero.
   This closes one bounded document-local source-declared reference family only. It does
   not establish ontological identity, cross-document registry resolution, aliases,
   open-world definition quality, capsule migration/promotion, complete semantics,
   truth, or learned linking competence. The pre-K1k cold producer and independent-verifier
   replay took 3,917.120s/6,341.007s, admitted all 27,941 rows with zero verification
   failures and zero producer-authority reuse, and reproduces candidate digest
   `sha256:5213e35a2e4697e7b2296c40534a776671e03ec13f129c860d7198a7cdf13bd8`.
   Warm producer/verifier replay takes 3.470s/12.350s with exact top-level run-cache
   hits and zero misses; per-object work is bypassed rather than counted as 27,941 hits.
   The two pre-K1k SQLite stores cost 11,365,433,344 bytes (10.585 GiB). Raw parsing, split reconstruction,
   aggregate calibration, and serialization remain uncached owners; the replay is GREEN
   for integrity and reuse, not for semantic utility, compression, or runtime efficiency.
   **K1j deterministic cross-document concept registry (`GREEN`, 2026-07-18):**
   the existing ConceptNet 5.7.0 source is pinned by compressed-content hash and scanned
   in full. Only the 218,061 English `/d/wordnet/3.1` edges with admitted source licenses,
   relation types, and weight enter a 352,411,648-byte read-only SQLite registry. The
   artifact contains 174,480 stable URI-derived identities and 410,638 normalized aliases
   over 12 typed relations. The full source build took `155.254s`; independent replay
   took `76.655s` while the canonical replay ran concurrently. The producer uses CSV framing; an independent verifier uses
   manual tab framing and a separate URI/alias reconstruction, compares every admitted
   relation, concept, and alias in order, checks SQLite integrity/foreign keys, and
   reproduces all counts and three content digests without importing producer authority.
   Learned compiler output can emit only a bounded surface request with optional
   non-authoritative POS/sense hints. It cannot mint or choose a stable identity or
   provenance. Deterministic validation attaches a cross-document identity only when
   the normalized surface maps to exactly one registry identity globally; hints never
   collapse ambiguity. Ambiguous and unknown requests retain packet-local nonpromotable identities with
   explicit `AMBIGUOUS` or `UNRESOLVED` receipts, and registry/tool faults fail closed.
   Registry evidence and relation neighborhoods are removed from the learned capsule view.
   The canonical learned compiler pipeline injects the resolver, while the route writes
   zero training rows, invokes no external inference, and receives deterministic-tool
   rather than learned capability credit. Corruption, direct and hint-mediated authority
   injection, external-inference receipt, absent-tool, ambiguous-sense, unknown-concept,
   invalid-POS, and source-hash controls pass. This
   establishes exact registered lookup and ambiguity preservation only. ConceptNet is not
   a truth oracle: graph completeness, assertion truth, natural-context learned linking,
   definition quality, migration, causal utility, and KERC promotion remain unclaimed.
   **K1k scoped-semantic ABI and exact program serialization (`GREEN`, 2026-07-18):**
   a source-independent graph contract now gives every proposition and nested scope
   exactly one owner. It supports assertion, negation, possibility, necessity, question,
   conjunction, alternation, condition, consequence, contrast, continuation,
   explanation, attribution, and quotation without flattening operator order. A
   separately implemented verifier reconstructs expected Kernel topology without
   importing the producer. All `14/14` operator routes replay; `5/5` modality/negation,
   condition-direction, and quantifier interventions produce distinct canonical program
   identities; `4/4` operator, target-role, root, and identity-map mutations reject.
   Strict quantity, temporal, text, symbol, and evidence-bearing ambiguity values fail
   closed. `KE-SERIALIZATION-2.0` repairs a prior construct-validity gap by encoding and
   decoding roots, every typed argument, confidence, derivation, and source alignment;
   packet validation now compares the decoded canonical program with its authoritative
   program instead of checking only macro expansion. The focused shared-contract suite
   passes `50/50`; the canonical gate is `GREEN` in
   `reports/kerc_scoped_semantics_k1k.json`. This is authored-fixture ABI, causal-
   intervention, and exact-replay mechanics evidence only. It writes zero training rows
   and establishes no semantic parsing, truth, completeness, learned competence, public
   benchmark, utility, efficiency, SOTA, AGI, or ASI claim. PMB remains calibration-only.
   The complete `27,941`-record corpus is now migrated under this serialization revision.
   Cold production took `3,642.746s`. The original serial independent admission took
   `4,982.793s`, of which `4,878.178s` was semantic admission. K1k now uses a bounded
   spawned process pool with parent-only SQLite publication, candidate-line ordering
   restored before aggregate replay, and a host resource receipt. On this 8-logical-CPU,
   16-GiB Mac the exact current-source four-worker cold replay took `1,503.943s`, with
   `1,418.771s` in semantic admission: `3.31x` and `3.44x` faster respectively. It wrote
   all `27,941` canonical rows and receipts with zero failures and reproduced the exact
   candidate, canonical, ledger, aggregate, and gate identities. Maximum resident size
   was `4,786,143,232` bytes, peak footprint was `10,342,161,792` bytes, and the process
   reported zero swaps. The policy selects four workers on this host and two on the
   tested 8-GiB/4-logical-CPU profile; unsafe explicit oversubscription rejects. A first
   parallel attempt exposed an order-sensitive rate-distortion calibration mismatch and
   correctly went RED; candidate-line authority ordering plus duplicate-line rejection
   now has a regression test. The producer/verifier stores cost `14,234,451,968` bytes
   (`13.257 GiB`). K1 still needs licensed non-
   benchmark proposition-level supervision and source-disjoint weak-tail evidence before
   any learned scope/compiler result can be interpreted; an absent source is an open
   evidence cell, not permission to substitute a toy proxy or issue a negative verdict.
   **K1l source-grounded scoped supervision (`GREEN` source evidence; full stage `RED`,
   bounded claim, 2026-07-19):**
   the GUM producer now projects seven direction-sensitive human eRST relation forms into
   six existing scope operators: condition, consequence (cause and result directions),
   explanation/evidence, contrast, alternation, and continuation. Admission requires one
   human primary edge and exactly two complete endpoint units. Any mapped neighborhood
   with shared secondary-edge ownership is excluded as a whole while its original eRST
   graph remains available; no arbitrary ownership choice or flattened substitute is
   permitted. This yields `1,525` admitted records (`1,127/204/194` train/dev/eval) and
   `90` explicit multi-edge exclusions (`66/10/14`). Minimum per-relation counts are
   `34/6/9`, source-group overlap is zero, train covers every relation-by-genre cell, and
   dev/eval each retain five explicit missing cells. A separately implemented verifier
   rebuilds direction, endpoint spans, operator, target roles, roots, and program identity
   from raw RSD/XML evidence without importing producer authority. Direction swaps, span
   shifts, forged authority, and shared-edge admission reject. Focused tests pass `43/43`
   and additionally prove the exact scoped program enters the source-only compiler target;
   the canonical reasoner objective consumes the same packet program. The full corpus
   remains `27,941` rows with candidate digest
   `sha256:b9f0fc0098d2088c902c7622d1d63d3c4b8ec79410f0dbdcf418700713ed694a`.
   A corrected warm producer replay takes `291.794s` with all candidate, finalization, and
   economics objects reused. Independent verification takes `658.590s`: `18,156` semantic
   admissions are reused and all `9,785` GUM discourse records are recomputed by the
   bounded four-worker verifier, producing canonical digest
   `sha256:87db1c759082e48806f9788c409f150e3900c3e4f1df90e2a0d83d6dc9884475`
   with zero failures. The producer warm path peaks near `2.9 GB` resident memory, so
   streaming cached packet finalization remains an efficiency follow-up. The learned-view
   ABI now explicitly admits every licensed semantic tag family, rejects unregistered tags,
   and preserves all `256` composite FrameNet records with their complete frame inventory
   and deterministic union role vocabulary. A `6.498s` raw preflight covers all `27,941`
   rows. The repaired full canonical stage validates all `27,941` candidates and
   deterministically selects `14,238/3,621/3,350` train/dev/eval records after excluding
   five raw sources that cross fixed splits. It meets every frozen decision-grade objective
   floor, retains all `96` grounded question records, and exposes neither answers nor model
   outcomes to selection. The stage writes `53,414` learned views, `53,414` verifier-only
   compact-ABI corruptions, and `384` generator-loss-disabled context counterfactuals from
   `18,033` unique raw sources / `4,721,570` raw bytes. Exact typed compact program and
   answer transports invert without loss. The compiler is iterative in eight-target-node
   chunks over visible source plus prior generated state; the reasoner is topologically
   scheduled in eight-node chunks over direct predecessor stubs plus exact prior generated
   claims; merged outputs are exactly revalidated before advancing. Encoded-length-only
   ragged `8K/16K` buckets retain all examples with zero truncation or drop; only `347` of
   `107,212` materialized rows enter the single-row long bucket, and that mechanism receives
   zero capability credit. The cold run takes `1,636.573s`; unchanged dependencies and
   outputs exact-replay in `1.58s` under cache key
   `89435e019dd7b5d75d195828ec8a6900e6627b51d1e70d82677c8f39f0f51004`.
   This is GREEN architecture/data materialization evidence, not learned competence,
   complete sentence semantics, truth, temporal truth, utility, SOTA, AGI, or ASI. The next
   gate is bounded learnability, gradients, overfit, checkpoint/reload/resume, causal
   intervention, and no-fallback replay on this exact stage before K2 or long training.
   **K1m production-route adequacy result (`SUPERSEDED_PARTIAL`, 2026-07-19):** two independently
   selected, source-bound Apple-MLX profiles now pass stage-gradient, exact teacher-forced
   overfit, learned protected-span declaration plus independent byte reconstruction, real
   no-fallback autoregressive execution, and exact checkpoint-reload gates. The stateful
   profile contains two compiler chunks, two topologically scheduled core chunks, eight
   prior-claim dependencies, and a renderer; its forward/recompile path executes all nine
   expected stages and both removal interventions measurably change target logits. This
   establishes production-shaped mechanics only. The enclosing canary remains `RED`:
   continuous and resumed training preserve every discrete outcome and differ by at most
   `2.98e-7` in parameters, but one token logit differs by `5.75e-5`, beyond the frozen
   tolerance; after joint-objective training, the seven-row decision probe retains ANSWER
   and ABSTAIN but forgets the sole CLARIFY row (`0.6667` macro balanced accuracy versus
   the `0.75` floor). Do not relax either threshold after inspection. The next owner is a
   preregistered repeated fresh-process MLX numerical-equivalence characterization plus
   gradient-conflict/rehearsal repair for joint decision retention. This result grants no
   source-disjoint competence, utility, efficiency, KERC success, or negative verdict.
   **K1n resume-and-retention closure (`GREEN`, 2026-07-19):** the canonical canary now
   branches continuous and fresh-process MLX runs from the same content-addressed step-48
   checkpoint and optimizer state. Two preregistered calibration seeds derive numerical
   envelopes with a frozen `4x` safety factor and absolute hard ceilings; two untouched
   validation seeds preserve every discrete outcome. Validation maxima are `2.8313e-7`
   parameter delta, `9.8226e-9` optimizer delta, `1.0968e-5` token-logit delta, and
   `5.9605e-8` verifier-logit delta. The protected-single and hierarchical profiles reach
   exact teacher-forced recovery in `936` and `1,536` updates, then pass real no-fallback
   autoregressive execution, exact checkpoint replay, and causal state interventions. Plain
   `80`-update joint training retains observed ANSWER/CLARIFY/ABSTAIN mechanics at `0.9333`
   macro balanced accuracy. A deterministic class-balanced rehearsal candidate reaches
   `1.0` decision macro accuracy under the same update count but is not adopted because its
   `0.1806` token-accuracy regression violates the frozen `0.05` preservation bound. The
   enclosing report is GREEN with no hard or inconclusive gaps. This closes bounded K1
   mechanics only; PARTIAL behavior, source-disjoint semantics, utility, efficiency,
   interaction amortization, and the KERC thesis remain unclaimed. K2 is now closed by
   the evidence below; K3 is the next owner.
   **K1 closure gate:** maintain a construct-by-domain-by-composition-depth coverage
   matrix, not one aggregate record count. Every claimed construct needs source-group-
   disjoint train/development/evaluation examples, adversarial minimal pairs, multiple
   valid realizations where appropriate, weak-tail counts, mutation tests, and an
   explicit missingness mask. The existing `1,024/1,024/1,024` human-gold floor is a
   minimum, not evidence of adequacy by itself: preregister learning curves and a power
   analysis, and expand only the underpowered cells until the intended effect is
   detectable. No count of derived views is presented as unique semantic data.

3. **K2 - per-unit packet and rate-distortion contract.** Replace the single record-wide
   fidelity control with stable residual-unit identities and independently selectable
   `q_i` values for interaction entries, each segment/frame, token-local residue, concept
   realization, and exact object. Preserve source residual and output render plan as
   different typed objects. Hard minimum fidelity is determined per unit from type,
   copy policy, user instruction, authority, and risk; one number or quote must not force
   unrelated units to exact. Every candidate level must serialize a real payload and
   record actual conditional bits, a multidimensional distortion vector, hard-constraint
   results, provenance, and replay identity. The packet-wide fidelity is at most a
   derived summary and cannot drive training. Migrations from KPP-1.1 must be exact or
   explicitly reject unsupported legacy rows.

   **K2 closure evidence (2026-07-19):** KPP-1.2 now materializes 229,852 stable
   residual units over all 27,941 canonical records: 27,941 interaction entries,
   49,478 segment/frame units, 115,180 token residues, 7,180 concept realizations,
   and 30,073 exact objects. Each unit owns four real typed payload candidates, an
   exact dictionary-conditioned codec receipt, a compact 13-dimension structural
   distortion vector, a hard minimum and ceiling, and a separate source-residual and
   render-plan identity. The independent verifier reconstructs the inventory,
   candidate projections, codec costs, constraints, and selected action from source.
   Codec conditioning excludes the priced unit and includes only strictly prior
   residual state; dedicated causal-isolation tests prevent self-conditioning.
   KPP-1.1 migration preserves every legacy payload field and fails closed on tampered
   or unknown versions. The cold producer and four-worker verifier are GREEN with zero
   hard gaps; exact warm replays preserve the corpus and governed-record hashes in
   3.78 and 12.14 seconds. Cold production remains expensive at 3,585.081 seconds
   (2,101.849 materialization and 1,400.937 finalization), so this is faithful K2
   mechanics, not an efficiency win. The allocator remains a measured structural
   baseline, not learned, intervention-sensitive, semantic, or utility evidence. K3
   is the next owner.

4. **K3 - intervention-sensitive learned allocation.** Build target labels by actually
   downgrading one unit at a time and measuring independently recomputed proposition,
   entity, value/unit/precision, scope, polarity, modality, temporal, causal,
   attribution, quote, terminology, style, byte, and downstream task effects. Use hard
   infinity constraints where loss is forbidden. Train a source-visible per-unit policy
   to predict fidelity and calibrated confidence from Kernel, higher-level residual
   state, unit type/content, discourse centrality, recurrence, normalization uncertainty,
   task/risk context, and verifier sensitivity; evaluator-only answers and hidden targets
   never enter its features. Allocation-target production and final evaluation must be
   cross-fitted and organizationally separate: no verifier, corruption generator,
   semantic parser, or human adjudication rubric may both manufacture a target and judge
   the same heldout decision. Use an evaluator panel combining hard typed checks,
   source-disjoint semantic models, executable downstream tasks where available, and
   adjudicated human gold; disagreement remains uncertainty rather than being collapsed
   into a convenient label. Prefer constrained optimization (`min rate/cost` subject to
   distortion and hard-safety bounds) over one global scalar lambda when it is measurably
   more stable; calibrate dual variables/thresholds on development only. Retire
   `labels_by_channel` from allocator loss and feed the actual unit decisions, estimated
   bits, and confidence into the model/checkpoint. Acceptance: gradient, overfit,
   calibration, intervention monotonicity, leave-signal-out, missing-signal, and
   source-disjoint generalization checks pass; a presence classifier cannot pass the
   allocator gate.

   **K3 bounded qualification (`GREEN`, 2026-07-20):** the canonical producer now
   performs four real one-unit downgrade interventions across thirteen typed fidelity
   dimensions for `160,139` units over `21,209` governed records. It grants allocator
   authority to `69,979` decisions, withholds `90,160` uncertain units, and exposes no
   evaluator effect, public payload, hidden answer, external inference, fallback, or
   template credit to the model. A source-family-disjoint MLX qualifier uses exact ragged
   byte packing rather than maximum-length padding, preserves every byte without
   truncation, and passes nonzero gradients, exact representative overfit, hard masks,
   and strict checkpoint reload over five seeds. It adds a source-only unit-to-task
   relation channel and multiplicative source-candidate scoring; answer text and evaluator
   effects remain hidden. Heldout contested accuracy is `0.9697-0.9710`, versus `0.8638`
   for the strongest source-blind candidate-schedule control. Five matched controls
   retrained with the relation channel absent are worse on every seed by
   `0.0094-0.1025`, with a `0.0326` mean loss. The independently implemented human panel
   covers `3,061` adjudicated units across ten annotation policies and reaches `0.9971`
   decision agreement, `0.9983` fact recall, `0.00294` semantic violations, zero hard
   violations, and zero adjudicated rate regret. The content-bound receipt authorizes
   canonical allocator loss. This closes bounded K3 only; end-to-end KERC utility,
   compression, amortization, public transfer, architecture advantage, and the KERC
   thesis remain unclaimed. K4 is the next KERC discovery owner after the practical
   eight-target capability campaign; it no longer blocks that campaign.

5. **K4 - real coding and interaction amortization.** Keep the exact adaptive arithmetic
   coder as a replay baseline, then compare compact static/dictionary, rANS/arithmetic,
   and reproducible learned conditional/hyperprior codecs on identical unit payloads.
   Charge model parameters/state, initialization, empty-channel framing, packed metadata,
   synchronization, checkpoints, lookup, recovery, latency, memory, and energy. Encode
   deltas/references rather than repeatedly charging or shipping a full global state.
   Build coherent 1/2/4/8/16/32/64-turn interactions with repeated terminology, aliases,
   styles, local exceptions, scope changes, resets, and adversarial reuse. Promotion must
   optimize measured definition/reference savings plus expected future use, renderer
   consistency, state-management and privacy risk; strict observed break-even remains a
   hard lower bound, not the whole policy. Acceptance: independently replayed curves show
   where global sharing beats local tags, including confidence intervals and negative
   cases; empty or synthetic state cannot satisfy H5/H6.

6. **K5 - coordinated learned architecture.** Complete both the declared shared-trunk
   candidate and a faithful modular compiler/core/renderer candidate unless a prospective
   resource study selects one without a scientific verdict. Train calibrated protected-
   span detection and lexical abstention; constrained surface-to-Kernel compilation;
   Kernel language modeling/reasoning and structured answer generation; copy/pointer-
   aware rendering; and independent recompilation/verification. Add direct parallel,
   paraphrase-convergence, minimal-pair, cycle, bidirectional-entailment, question-
   preservation, source-alignment, uncertainty, macro expansion, per-unit rate,
   interaction-promotion, and task losses under one versioned objective ledger. The
   current four stage adapters and task tokens are a shared-trunk baseline, not proof of
   logically separable learned stages. Stage tokens may route authority but may not leak
   labels or become the only predictive feature. Compare stagewise warm-start, joint,
   and alternating optimization under the same total tuning allowance; instrument
   gradient conflict, loss-scale domination, dead modules, and capacity starvation
   instead of assuming one weighted sum is adequate. Close the train/inference exposure
   gap with scheduled self-generated intermediate packets and report teacher-forced,
   mixed, and fully autoregressive results separately. Build an oracle-substitution
   ladder (`gold compiler`, `gold residual allocation`, `gold core answer`, `gold
   renderer`, and combinations) so the achievable ceiling and each propagated error are
   measured before blaming the whole architecture. Sweep compiler/core/renderer/verifier
   capacity allocations within a matched total-parameter/FLOP envelope; a deliberately
   tiny component may not establish a negative. Acceptance: each module has nonzero
   gradients, can overfit a genuinely representative subset, survives checkpoint/resume,
   changes the intended behavior when intervened on, retains competence when upstream
   inputs are model-generated, and has no deterministic answer bypass or uncredited
   generalist fallback.

7. **K6 - construct-valid verification and decisions.** Expand the hard verifier to
   compare byte literals when they are legitimate exact values; numbers with units,
   ranges, precision and approximation; identity/aliases; scope; modality; time;
   causality; attribution; quotations; terminology; caveats; evidence and uncertainty.
   Pair it with independently implemented semantic entailment and recompiler evaluators
   whose data producers, features, tokenization where practical, checkpoints, and
   corruption families are disjoint from the generator and allocator-target producer.
   Use executable task oracles and blinded human adjudication for a frozen subset to
   estimate every automated evaluator's false-positive/false-negative rate and
   correlated blind spots. Add
   PARTIAL, unresolved ambiguity, insufficient/conflicting evidence, protected-span
   detector misses, and calibrated ANSWER/CLARIFY/ABSTAIN behavior. Test correlated
   compiler/recompiler errors and evaluator blind spots with human-audited and
   adversarial counterfactuals. Bounded local regeneration is reported separately with
   retry cost; literal/template repair remains assisted noncredit. Acceptance: per-
   dimension positive/negative recall, calibration, selective risk, weak tails, and
   independent mutation coverage clear frozen floors; changing an opaque answer while
   preserving its wrapper must fail.

8. **K7 - false-negative and mechanics gate.** Before campaign freeze, pass learnability,
   gradient-flow, representative-subset overfit, checkpoint/reload, optimizer-resume,
   migration/rollback, intervention, leave-one-mechanism-out, resource, security,
   information-flow, and fresh-process replay checks. Canaries test mechanics only and
   must include nontrivial semantic structures; memorizing seven presence-label rows or
   passing a schema fixture cannot clear adequacy. Require a decision-grade adequacy
   packet with five independent sections before any negative verdict:
   (a) specification traceability from each paper mechanism to code, data, objective,
   checkpoint tensor, intervention, and metric; (b) component oracle ceilings and
   end-to-end error propagation; (c) data/capacity/optimization learning curves showing
   the run is above random, trivial-copy, and shared floor effects; (d) matched strong-
   baseline and architecture-specific tuning receipts; and (e) statistical power for
   the preregistered minimum useful effect with multiple seeds, paired items, confidence
   intervals, and weak-tail/domain results. Independently audit that the runtime path is
   the tested path and that every claimed causal mechanism is active. If all learned
   systems are near floor, an oracle route fails, tuning is unstable, the evaluator is
   underpowered, or model-generated interfaces collapse relative to teacher forcing,
   repair that owner and report `INCONCLUSIVE_*`; do not call KERC negative. Missing
   fidelity yields `INCONCLUSIVE_IMPLEMENTATION`; insufficient power, baseline strength,
   optimization adequacy, or construct validity yields `INCONCLUSIVE_EXPERIMENT`.
   Neither is translated into a KERC failure or a nearby green proxy.

9. **K8 - matched campaign, adoption, and integration.** Prospectively freeze a
   multi-seed, source-disjoint 25M-75M-active comparison covering the conventional
   surface model, full KERC, KERC without residuals, semantic-graph/renderer,
   compact-reasoning without a compiler, byte/dynamic-chunking, and reproducible
   compression controls. Match raw data, total FLOPs, tuning opportunity, active and
   total parameters, inference/search/verifier budget, hardware, and full lifecycle
   cost while allowing equal-budget architecture-specific tuning. Freeze a minimum
   useful effect, noninferiority margins for fidelity/weak tails, stopping rule, seed
   count, and capacity/data scaling ladder before outcome inspection. Run representation,
   component-oracle, core-modeling, and end-to-end tracks separately: a representation
   win cannot become a utility claim, and an end-to-end loss cannot identify a substrate
   failure without the component evidence. The 25M-75M tier is a debugging tier; a
   scientific negative requires both KERC and strong controls to demonstrate nontrivial
   learning and enough power in the tested regime. Measure H1-H8 on
   short and long contexts, model-only English utility, fidelity, calibration, accepted
   verified output per second, KV/memory/energy, and weak tails. STS, VCM retrieval,
   tools, and Octopus routing are held equal or separately ablated and cannot rescue a
   KERC model-only claim. Publish total-system Pareto fronts and per-stage failure
   decomposition, not one aggregate winner. The practical English arm adopts the Pareto
   winner; useful KERC components may survive independently. One decision-grade negative closes only
   the exact regime; broad retirement requires replication. Failure to reach K7 excludes
   KERC from this first practical campaign as `INCONCLUSIVE_IMPLEMENTATION` without
   blocking conventional-model training indefinitely.

**Cross-cutting KERC requirements:**

- Define one versioned `KernelPacket` protocol spanning immutable/content-addressed
  source identity, protected objects, correction lattice, expanded semantic Kernel,
  compact serialization, grammar-aware macro tokens, entity/concept handles,
  source alignment, uncertainty, provenance, authority, and residual references.
  Protection precedes correction; canonicalization targets contextual senses rather
  than strings; unresolved sense, coreference, scope, and correction ambiguity stay
  explicit rather than being compressed into false certainty.
- Separate stable concept identity from runtime spelling and maintain three typed
  code spaces: surface English, Kernel concepts/operators, and pointer/object/control
  symbols. Grammar-aware Kernel macros must have deterministic typed expansion and
  may not cross entity, exact-object, clause/scope, negation, quantifier, quotation,
  code, value/unit, provenance, or authority boundaries. Byte fallback is complete;
  open-world concepts use provenance-bound local capsules or opaque handles, and
  promotion into the shared concept registry requires an SCF replacement transaction.
- Extend VCM rather than adding a residual side store. The Hierarchical Residual
  Ledger has interaction-global, segment, token-local, and exact-object levels;
  distinguishes source reconstruction from output render policy; and assigns
  semantic, faithful, lexical, or exact fidelity by calibrated rate-distortion plus
  hard policy. Every entry carries scope, origin, authority, confidence, privacy,
  expiry, version, dependency, and deletion state. State hashes, append-only deltas,
  checkpoints, deterministic replay, migration, bounded eviction, and descendant
  invalidation are mandatory. Missing or mismatched state produces a typed recovery
  or rejection outcome, never approximate interpretation or silent cross-user reuse.
- Make the English reasoner consume Kernel/VCM packets and emit a structured answer
  packet with claims, modality, confidence, entities, values, qualifiers, provenance,
  required terms, and rendering constraints. A separately accounted surface renderer
  may realize prose and copy authorized exact objects, but generation credit belongs
  only to the learned model channel. Recompile output and verify entity identity,
  numbers/units/precision, negation/quantifier scope, modality, time, causal direction,
  attribution, quotations, terminology locks, caveats, and uncertainty. Failed
  verification remains failure or bounded repair; a literal/template renderer can be
  reported only as an assisted noncredit baseline and never rescue model-only utility.
- Bind Kernel versions, concept-registry hashes, codebook hashes, macro vocabulary,
  residual schema, object types, compiler/core/renderer/verifier compatibility, and
  migrations into checkpoint and AIBOM lineage. VCM memory, Semantic IR, tool calls,
  planner nodes, Reflexive Router events, and Octopus composition may exchange Kernel
  packets only through registered adapters with loss/use/authority certificates; the
  representation itself grants no truth, evidence, execution authority, or capability.
- Train each declared KERC candidate as one coordinated architecture, not disconnected
  prototypes: protected-span
  detection and calibrated lexical abstention; surface-to-Kernel parallel, paraphrase-
  convergence, hard-negative, cycle, entailment/question-preservation, structured-
  validity, and uncertainty objectives; Kernel language modeling and answer-packet
  generation; residual allocation and interaction promotion; copy-aware rendering;
  and symbolic plus neural round-trip verification. Data remains licensed, English-
  scoped, provenance-bound, source-disjoint, and contamination-audited. Public
  benchmark payloads remain calibration-only, and live teacher use remains governed
  OpenAI-only residual pressure. Version migration is deterministic or the row stays
  explicitly versioned; no compiled row silently changes meaning across revisions.
- Treat every learned interface as a distribution-shift boundary. Report performance
  under gold upstream state, corrupted upstream state, mixed scheduled state, and fully
  model-generated state; measure recovery, calibration, and error amplification at each
  boundary. Teacher forcing, deterministic packet construction, and oracle rendering are
  training/debugging instruments only and cannot support end-to-end learned utility.
- Build the semantic corpus as a tiered evidence program, not a bulk generated proxy.
  Require at least `1,024` decision-grade train records and all `1,024` private-dev plus
  all `1,024` private-eval records to be audited/licensed human semantic gold. Qualified
  local-parser output may supply at most 80% of train records at weight <= `0.25`; it
  remains silver and cannot satisfy heldout or claim floors. Governed OpenAI residual
  rows remain capped at 10% accepted records and 2% optimizer probability. Candidate
  sources must pass license, revision, content identity, public-surface, contamination,
  producer fidelity, and independent-review checks. Do not convert a syntax parse,
  deterministic wrapper, self-reconstruction target, VIEA metadata trace, or public
  semantic benchmark corpus into purported Kernel semantic gold.
- Pre-training closure requires a faithful English-arm architecture alternative. Implement the
  exact substrate, dataset transforms, objectives, checkpoint/migration path,
  instrumentation, lifecycle, and security contracts before optimizer spend, then
  preregister conventional surface-English and a 25M-75M-active KERC candidate in one
  joint campaign. The code arms remain unchanged. Scale KERC only if it wins. Compare standard BPE,
  SentencePiece/unigram, spelling-normalized BPE, controlled-English BPE, multiword or
  over-tokenized input, byte-level, faithful BLT/H-Net-style dynamic chunking,
  reproducible neural text compression, compact-reasoning without a compiler,
  semantic-graph/renderer, KERC without residuals, and full KERC. Label conceptual
  approximations separately from faithful reproductions.
- Match three budgets independently: underlying raw bytes, total training FLOPs, and
  end-to-end inference latency/energy. Charge compiler, core, renderer, verifier,
  residual codec, object/concept tables, registries, exact-source access, cache/KV,
  retries, failure recovery, storage, and human review. Report original bytes, surface
  and Kernel tokens, expanded/compact lengths, residual bits by level, object bytes,
  parameters by module, per-stage P50/P95/P99 latency, FLOPs, energy, peak memory,
  accepted verified output per second, and total lifecycle cost. Token reduction or
  core-only speed is never an efficiency win by itself.
- Freeze representation, core-model, end-to-end, fidelity, interaction, robustness,
  and security tracks before results. Measure loss per original byte, task utility,
  sample efficiency, domain transfer, exact reconstruction, bidirectional entailment,
  QA preservation, semantic-atom recall, entity/name/value/scope/modality/attribution/
  quote fidelity, terminology consistency, calibration, and conversations of
  1/2/4/8/16/32/64 turns. English dialect/noise, Unicode/confusable names, technical
  terminology, exact-form tasks, long documents, and held-out domains are required;
  non-English natural-language training remains quarantined for this seed.
- Run causal ablations over all four residual levels; protection-before-correction;
  correction lattice versus forced correction; handles; concept capsules; regular
  morphology; compact orthography; Kernel BPE versus bytes; grammar constraints;
  separate versus tied vocabularies; verifier strength; fidelity policy; byte
  fallback; local macros; exact-source access; and interaction-state sharing. Use
  multiple seeds, paired items, confidence intervals, frozen thresholds, published
  failures, and independent integrity/replay audits.
- Trap-test overcorrection, ambiguity collapse, semantic drift, dropped negation or
  modality, swapped identities, unit/precision changes, quote corruption, Unicode
  confusables, residual injection, untrusted-content state mutation, object-store
  exfiltration, prompt injection across representation layers, stale/desynchronized
  state, cache poisoning, macro/concept poisoning, unauthorized registry promotion,
  incompatible migration, cross-user leakage, and deletion failure. Source access is
  least-privilege and auditable; users can inspect/reset/delete residual state.
- A full-architecture negative verdict is allowed only after the decision-grade
  adequacy audit passes and an independent replication confirms that no realistic
  long-context regime improves total
  quality/compute/memory, residual overhead erases savings, compiler error cannot be
  contained, shared state adds no value or unacceptable risk, exact identity/value
  fidelity regresses, or a simpler byte/dynamic-chunking/latent system Pareto-dominates
  it. Preserve useful components such as protected handles or terminology state if
  their own ablations win. Until then, report the exact weak implementation or
  experiment boundary rather than preserving or rejecting KERC by label.

Pre-training acceptance: K0-K7 are complete against nontrivial semantic structures;
the canonical packet/compiler/core/renderer/verifier and VCM residual lifecycle exist
behind registered versioned interfaces; actual per-unit allocation targets replace
presence labels; the corpus exercises every claimed mechanism; and transforms,
objectives, checkpoints, migrations, cost accounting, adversarial tests, matched
controls, and the K8 campaign are content-bound and frozen. Finite-update,
reload/resume, migration, cleanup, memory, resource, replay, integrity, information-flow,
and no-cheat canaries are GREEN. This authorizes the matched training campaign, not a
capability, efficiency, or KERC-success claim. If K0-K7 cannot close within the frozen
candidate scope, record `INCONCLUSIVE_IMPLEMENTATION`, preserve the research branch as a
registered alternative, and allow the conventional surface-English lane to proceed.

Post-training acceptance: one content-bound, source-disjoint, multi-seed campaign
either shows a preregistered Pareto improvement in learned English utility and
total-system cost without fidelity/weak-tail regression or records a decision-grade
negative for the exact regime. Broad retirement additionally requires independent
replication. A schema, compressed sequence, readable trace,
exact copy, green round trip, assisted renderer, or reduced core token count is not
learned reasoning, broad transfer, runtime authority, AGI, or ASI evidence.

### Phase 14: Compression, Proof, and Claim-Evidence Records
- First-class claim/evidence transitions preserve contradiction, downgrade, split,
  merge, retirement, and replacement history. `docs/PROJECT_STATE.md` becomes a
  compact ledger view over checkpoints/reports/configs/curricula/verifier/dogfood.
- Build an epistemic trusted computing base: independent verifiers/auditors,
  bounded trust propagation, trap fixtures, randomized deep replay, and
  receipt-faithfulness checks so a report or receipt is not trusted because it
  exists.
- Run a retrospective adequacy audit over every negative result currently used to
  close a rung, retire an objective, or constrain a route. Start with KERC, then the
  10.8M regime, QCSA objective, prefix-LM, STS/VCM ablations, SymLiquid comparators,
  and generation-mode dispositions. Reclassify each as inconclusive, exact-regime
  negative, or replicated mechanism negative; narrowing a claim does not authorize a
  rerun or erase the original result. No negative may affect a future route until its
  implementation fidelity and experiment adequacy are machine-readable.
- Materialize an `EpistemicTCBManifest` for every component that can admit training
  data, authorize execution, score promotion, load weights, mutate routes, or publish
  a claim. Each entry binds implementation identity, authority, assumptions,
  independent checker, shared/correlated dependencies, golden traps, mutation or
  fault-injection coverage, replay sampling, blind spots, expiry, and rollback.
  Training admission requires the Phase 7 subset to qualify; later route/release
  claims require their own subsets rather than inheriting trust from a green report.
- Compression receipts declare reconstruction, residual coding, determinism,
  supported use, authority/taint preservation, and exact-replay boundary. Proof and
  Lean artifacts support only their finite stated claims.
- Keep generated state out of the source tree by default. After the active v8 run
  closes, migrate heavyweight runtime/checkpoint/report payloads to the configured
  external state root; retain only content-bound manifests, current route pointers,
  active/confirmation checkpoints, compact negative-result summaries, and exact
  replay locators in the repository. Per-family quotas apply to superseded canaries,
  candidates, optimizer snapshots, and duplicate reports; retention always runs
  dry-run plus reference-closure verification before moving bytes.
- Compile structured assurance graphs from existing claims, evidence, assumptions,
  hazards, contexts, acceptance criteria, defeaters, mitigations, dependencies,
  owners, and review/expiry state. Changes invalidate affected nodes; a complete
  graph never becomes safety or deployment authority by itself.
- Acceptance: one live claim undergoes evidence-backed belief revision; every active
  negative route constraint has a decision-grade or campaign-scope-only disposition;
  stale or
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
  the artifact budget GREEN with zero hard gaps and zero warnings. Local
  `archive/`, `runtime/`, `data/`, `checkpoints/`, and `reports/` are still large, so
  this is replay correctness rather than completion of physical source-tree cleanup.
  These are evidence-system results, not model-capability or checkpoint-quality claims.

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
- Implemented evidence: three diverse schema-bound metadata workflows (planning,
  chat, and deterministic-tool assistance) pass replay and canary execution;
  each compiles to a digest-bound, noncredit lookahead asset. Exact lookup
  selects `3/3`, unknown/ambiguous routes abstain, and append-only lifecycle
  receipts automatically retire stale or postcondition-drifted procedures.
  Three SCF replacement transactions arm rollback guards, the plan compiler
  consumes exact asset/receipt IDs, and assistant runtime selects all three
  bindings without widening authority. These are authored/synthetic workflow
  compression mechanics, not evidence that repeated natural work was discovered,
  made useful, or adopted by Corben in daily operation.

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

#### Reflexive Router integration extension

Author-source intake: *The Reflexive Router* v1.2 (2026-07-16), content SHA-256
`003a693741c40ca96ec3aece5b76ee90ec95a1d6c27ec81a970cff175f509068`.
The source is currently uncommitted in the AI_book worktree, so this is an
argument-backed intake contract until it is rebound to a committed book revision.
It extends the existing assistant, SCF, Octopus, VIEA, VCM, deterministic-tool,
effect, and Loop-Closure owners; it must not create another router or ledger lane.

Current pre-training state: the canonical assistant now imports one qualification-first
dispatch contract before checkpoint inference and before code, tool, planning, dogfood,
or effect execution. Authenticated direct commands, literal isolation, authority/profile/
freshness qualification, finite terminal outcomes, bounded DAGs, trace integrity,
bitemporal Chronicle records, dependency-complete cache identities, and governed
compile/decompile helpers, three bounded effort profiles, true multi-capability workflow
selection, per-capability qualification, and malformed learned-plan rejection pass 24
foundational focused tests. Typed parameter schemas and bounded dynamic defaults, preview/apply/rollback
registry mutation, structured cancellation and immutable partial-result semantics, retry
idempotency, late-result suppression, compensation, and exact registry/effect rollback are
also implemented; the complete focused suite is now 34 tests. The existing assistant
E2E now passes all hard gates across six cases: chat, code, tool, planning, a tool-to-plan
workflow, and a reversible effect route. The workflow executes 15 independently verified
deterministic-tool cases, then executes the actual `tool-evidence -> compile-plan` DAG
through the structured reference kernel before exposing only verified references and
digests to checkpoint synthesis. The effect canary is bound to the
verified dispatch trace and decision digest, materially observed, and restored to the
exact prior filesystem identity. Direct effort rejects the over-budget code route with
`resource_exceeded`; a forced unknown route produces `unsupported`, skips downstream
inference, and grants no effect authority. The frozen private ReflexBench mechanics
profile executes 32 cases across eight tracks against ten policies (320 result rows).
The full reflexive route resolves 32/32 mechanics cases; no-Chronicle and no-compiler
ablations each resolve 28/32. Non-oracle policies cannot read held fields, held-field
mutation does not change their route decisions, and all literal-isolation, authority,
unsafe-effect, mutation, and replay counters remain zero. Because profile intent is a
visible fixture field and the learned-router arm is an untrained proxy, these are
interface/mechanics results, not learned language-routing evidence.

Pre-training disposition: `pretraining_wired_behavior_qualification_pending`. The
architecture, ownership, negative controls, state transitions, checkpoint compatibility,
and exact replay boundary are complete enough to freeze. Learned-router calibration,
selective-risk curves, real route regret and end-to-end latency, trace-to-reflex promotion
from diverse governed traces, dogfood drift detection, and answer-quality improvement
remain post-training qualifications. No learned routing, reflex-promotion, answer-quality,
public-transfer, or capability-gain claim is made from the mechanics profile.

- Put one canonical pre-deliberative path before ordinary assistant inference:
  canonical event and authenticated command plane -> deterministic and learned
  route proposals -> SCF qualification -> minimum-sufficient-compute DAG -> VIEA
  execution/effect commit -> verification -> typed result/VCM Chronicle update.
- Support typed user route directives, direct capability commands, workflows, and
  requested-versus-realized deliberation profiles. A forced route fails with a
  typed unqualified outcome unless the user explicitly permits fallback. The user
  may bypass inference, never authentication, authority, verification, or rollback.
- Extend the implementation registry in two explicit, linked roles rather than
  inventing a parallel catalog: a declarative reflex registry for bounded automatic
  fast paths and a scoped User Command Registry for authenticated route directives,
  typed capability bindings, parameterized commands, and workflow DAGs. Contracts
  carry owner, version, precedence, schemas, preconditions, capability dependencies,
  effect class, resource budget, verifier, operating envelope, fallback, telemetry,
  expiry, and rollback. Command bindings are capabilities, never trusted shell,
  SQL, URL, or prompt-text macros.
- Make command ingestion structurally origin-aware. Slash-like text from retrieved
  pages, emails, documents, code blocks, tools, or model output remains untrusted
  data and cannot activate the command plane. Registration and mutation require
  namespace/alias conflict checks, workflow-cycle checks, capability-diff and
  authority-expansion review, typed dynamic-default validation, preview, provenance,
  signature/trust-tier checks for imported packages, and immediate disable/rollback.
- Make the learned router a calibrated, abstaining proposal mechanism with atomicity,
  composite-task, OOD, selective-risk, and deadline-aware cascade outputs. Route
  selection minimizes total qualified cost, including verification, repair, human
  cleanup, privacy, energy, and route-regret burden; ranking never grants authority.
- Keep proposal, qualification, authorization, execution, verification, rendering,
  and promotion as separately owned decisions. Qualification must check schema,
  entity resolution, context adequacy, freshness, implementation health and tested
  distribution, risk-conditioned quality, verifier availability, effect bounds, and
  explicit fallback before cost optimization. Preserve a finite terminal outcome
  vocabulary including resolved, prepared, partial, ambiguous, insufficient-context,
  insufficient-evidence, conflicting, stale, unauthorized, unsupported, OOD,
  resource-exceeded, execution-failed, verification-failed, escalate, and rejected;
  model prose may explain but never rewrite one outcome into success.
- Compile deterministic reflex contracts into an indexed decision structure rather
  than a linear ad-hoc rule chain. Precedence is explicit over authority, effect risk,
  specificity, verified confidence, freshness, owner priority, and a stable tie-break;
  registration rejects unresolved overlap, shadowed/unreachable rules, recursion or
  workflow cycles, missing failure edges, hidden effects, and privilege amplification.
- Extend VCM with a small rebuildable Reflex Context Frame and bitemporal Chronicle
  records that keep entity/event/state/claim/plan/prediction/counterfactual types,
  valid time, transaction time, epistemic status, provenance, and corrections
  separate. Every executor returns one typed result packet with input identity,
  dependencies, dispatch provenance, evidence, verification, effects, warnings, and
  context handles; rendering is a tested consumer. The frame is a projection that can
  be invalidated and rebuilt from authoritative ledgers, never a second source of truth.
- Treat semantic caches and Chronicle writes as governed reflexes. Cache identity must
  close over task semantics, principal/tenant, authority, entity/time scope, privacy,
  source/schema/capability/model/policy versions, and freshness; dependency changes
  invalidate descendants. Chronicle writes require record-class authority, immutable
  source identity, claim/state separation, contradiction/correction handling, and
  poisoning controls. Typed structure remains traceability, not evidence of truth.
- Make the existing effect transaction the sole external-write kernel: prepare,
  authorize, commit, observe, verify, record, and compensate or roll back. Retries
  require idempotency identity, and verification failure remains a typed failure.
- Require structured concurrency for composite routes: inherited authority and
  deadlines, bounded node/depth/fanout/retry/aggregate-cost limits, cancellation of
  dependent branches, immutable partial results, late-result suppression, explicit
  all-or-nothing/best-effort/compensate/stop policies, and plan provenance. Dispatch
  overhead and executor latency are measured separately at every stage.
- Extend Phase 15 from exact procedural lookup to governed reflex compilation:
  diverse verified traces, negative-space guard synthesis, static overlap/cycle and
  privilege analysis, historical replay, differential tests, live shadowing, canary
  activation, signed SCF replacement, monitoring, narrowing, quarantine, revocation,
  and decompilation back to deliberation.
- Admit compilation only when expected verified reuse clears its complete lifecycle
  cost: deliberative executions avoided must outweigh compilation, replay, shadow,
  verification, monitoring, correction, rollback, and human-review cost without
  weakening quality or risk. Parameterized shared reflexes and expiry prevent rule
  explosion; holdouts, source diversity, correction channels, and descendant
  invalidation prevent self-reinforcing trace feedback.
- Evaluate through one frozen private ReflexBench profile over existing assistant,
  router, tool, temporal, composite-DAG, effect, and procedural fixtures rather than
  a new benchmark family. Under matched event information, model access, authority,
  retries, verification, and total resources, compare monolithic model-only,
  LLM-first tool agent, hard-rule-only, learned model router, semantic cache plus model,
  modular LLM-routed tools, full route without Chronicle, full route without compiler,
  complete Reflexive Router, and oracle selection. Ablate qualification, typed results,
  hot context, Chronicle, command plane, effect kernel, and shadow-governed compilation.
- Report complete denominators and joint metrics: useful task outcome, qualified and
  fast-path coverage, wrong-fast-path rate, selective-risk curves, OOD abstention,
  route regret, P50/P95/P99 dispatch and end-to-end latency, quality/cost/energy/human
  cleanup, override and effort-profile fidelity, silent fallback, parameter binding,
  stale binding, literal isolation, registry-mutation safety, DAG validity, temporal
  accuracy, provenance/context continuity, unauthorized effects, verification escape,
  postcondition/rollback completeness, reflex transfer, drift detection/quarantine,
  and Useful Reflex Efficiency. No summary efficiency score may hide errors or effects.
- Trap-test the full lifecycle with untrusted command text, ambiguous expressions,
  stale bindings/context, OOD/adversarial paraphrases, rule and namespace collisions,
  cache collisions, Chronicle poisoning, decomposition/cost bombs, verifier monoculture,
  retry duplication, partial effects, explanation mismatch, premature compilation,
  dependency-version breakage, drift, quarantine, rollback, and decompilation. Breaking
  capability/schema/policy changes automatically disqualify dependent reflexes until
  replayed and requalified.
- Sequence: exact command/qualification/result/effect contracts may mature as
  maintenance, but learned-router and reflex-promotion claims wait for the selected
  behavior-positive local model. Default adoption requires a preregistered Pareto
  gain with no weak-tail, authority, privacy, or verification regression. Fast tool
  success, route accuracy, typed records, and compiled procedures remain separate
  from learned answer capability and public-transfer claims.

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
- Promote the existing synthetic scalable-oversight scaffold into a measured protocol:
  retain proposer/trusted-monitor/untrusted-monitor/observer/evaluator/promotion-
  authority separation, but replace `*_measured: true` declarations with referenced
  direct-review baseline, weak-supervisor outcomes, reviewer correlation/collusion,
  persuasion, randomized-audit coverage, latency, operator load/fatigue, dissent,
  escalation, abstention, and recursion-bottom evidence.
- Correct the existing synthetic capability commitment before it gains route authority:
  its current observed metric is declared local effect count, not capability. Rename it
  as an effect-exposure safeguard commitment or consume a real domain-scoped comparable
  assessment with uncertainty, prospective threshold, safeguard bundle, exception
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
  coverage; reconcile it first when this file and the matrix disagree. Bind the
  crosswalk to an immutable committed AI_book revision. Report later live-worktree
  edits as intake drift, but do not let an external dirty checkout invalidate model
  training readiness. Advance the pin only through a reviewed reconciliation and
  route new concepts to existing phases (see ASI Stack-to-Theseus Completion Program).
- Check exact chapter identity/order and every book-owned title, part, file, claim
  label, evidence level, minimal implementation, mature endpoint, interface,
  invariant, failure mode, and Codex-test inventory. Count equality alone is not
  synchronization. Preserve the book commit and manifest digest used for review.
- Acceptance: `roadmap_implementation_gate.py --gate` reports full pinned chapter
  coverage and separately identifies live drift; every represented book mechanism meets the
  ASI Stack Completion Invariants below; public-safe Theseus evidence can flow back into the
  book without importing private payloads or inflating support state.
- Do not: hide matrix drift in narrative; create a capability claim from a backlog
  row.

## ASI Stack-to-Theseus Completion Program

This is the exhaustive implementation intake for the current ASI Stack architecture.
It replaces "book parity" as a count-matching exercise with a finite, dependency-
ordered completion program. The pinned machine crosswalk remains authoritative for
the exact book revision it names; the live book currently has 55 chapters, including
`replaceable-cognitive-substrates-beyond-transformer-monoculture`, while the pinned
Theseus crosswalk remains on the reviewed 54-chapter snapshot until a new explicit
reconciliation transaction advances it.

This coverage program is subordinate to `ASI-THESEUS-FLAGSHIP-01`. A package may
protect or unblock gates `T0`–`T6`; it may not create a second capability flagship or
claim completion merely because every chapter has an owner.

The book is a design source, not an automatic implementation mandate. Every live
chapter core is still `Design rationale` at `argument` support. A concept enters the
Theseus execution path only when it serves the neural seed, governed teacher/data
loop, daily-use dogfood lane, or repository/evidence health. Everything else remains
visible as a deferred obligation rather than becoming a new lane, dashboard, product
surface, benchmark family, or perpetual pre-training blocker.

### Completion-state vocabulary

Every work package below must use one of these states. A prose claim such as
"implemented" cannot substitute for the corresponding evidence state.

| State | Exact meaning |
|---|---|
| `required_now` | Repository truth, safety, or contract coherence is currently broken; repair may run beside the active flagship because it protects that lane. |
| `pretraining_contract` | Adopting the mechanism later would invalidate topology, representation, objective, checkpoint, state, or matched-comparison identity; implement or explicitly exclude it before the next long run. |
| `real_use_qualification` | Mechanics may exist, but natural multi-day use and outcome evidence are required before the mechanism can claim usefulness or become a default. |
| `post_behavior_qualification` | The experiment requires a behavior-positive student; schema or zero-numerator canaries cannot satisfy it and do not block the first valid run. |
| `external_dependency` | Valid work requires another trusted node, hardware, organization, network, or real operating environment not currently available. |
| `deferred_by_breadth_freeze` | The concept is retained with an entry condition but cannot create a current lane or surface. |
| `decision_grade_retired` | A faithful, adequately powered, matched experiment earned a scoped retirement. Proxy or construct-invalid failures never use this state. |

### Dependency waves

| Wave | When | Required outcome | Work packages |
|---:|---|---|---|
| 0 | now, beside faithful KERC repair | the roadmap, matrix, registered verifiers, retained reports, and their projections agree with live reality | `ASI-00` through `ASI-04`, `ASI-29`, `ASI-30` |
| 1 | before the next architecture freeze | every checkpoint-shaping interface is executable or explicitly excluded, with exact state and matched-comparison identity frozen | `ASI-05`, `ASI-06`, `ASI-13`, `ASI-16`, `ASI-17`, `ASI-19` |
| 2 | now through the first useful local lane | one natural assistant task traverses the existing stack and produces empirical dogfood/procedural evidence without learned-credit laundering | `ASI-07` through `ASI-12` |
| 3 | after one complete natural vertical exists | governance, assurance, failure detection, verification, and resource decisions consume observed evidence rather than declared booleans or fixture-only units | `ASI-14`, `ASI-15`, `ASI-18`, `ASI-20` through `ASI-23` |
| 4 | only after nonzero clean student behavior | measure routing, search, deliberation, memory, feedback learning, fast generation, compression, unlearning, and open-ended improvement causally | `ASI-24` through `ASI-27` |
| 5 | only when entry conditions become true | multi-node, network, embodied, world-model, public-operation, and hardware-root work may start without changing the current flagship | `ASI-28` |

### Governed completion work packages

#### `ASI-00` - Reconcile the live 55-chapter book without moving evidence by fiat

- **State:** `required_now`; owner Phases 0 and 19.
- Review the live 55-chapter manifest against the pinned 54-chapter snapshot, including
  changed core claims, interfaces, invariants, failure modes, minimal implementations,
  mature endpoints, tests, and proof targets. Add the replaceable-substrates chapter to
  the next pinned crosswalk only in the same transaction that records its disposition.
- Classify every delta as already satisfied, stronger existing obligation, new
  pre-training contract, real-use/post-behavior qualification, external dependency,
  deferred by the breadth freeze, or rejected as duplicate/out of scope.
- **Done:** immutable book commit and manifest digest, exact 55-row order, independently
  recomputed source-field parity, reviewed disposition for every changed chapter, and
  zero automatic support-state movement.

#### `ASI-01` - Canonical/projection integrity and stale-derived-state invalidation

- **State:** `required_now`; owner Phases 0, 14, and 19.
- Add one shared projection receipt used by roadmap summaries, project-state summaries,
  evidence-store views, module status, book crosswalk counts, and readiness reports.
  Bind canonical input identities/digests, compiler version, derived counts, freshness,
  and reverse-dependency closure. Derived counts must be recomputed, never maintained by
  hand beside their canonical rows.
- A changed source, schema, registered verifier, API, config, checkpoint, or canonical
  record invalidates every dependent projection and green report until regenerated.
- Preserve the current chapter-state histogram from canonical rows and its generated
  summary: 38 partial, 11 wired, two pretraining-pending, two frozen-dependency, and
  one mapped-missing state.
- **Done:** mutation tests catch changed rows, stale API/verifier identity, omitted state,
  wrong counts, and a projection that claims fresher authority than its canonical input.

#### `ASI-02` - Registered verifier migration and report lineage

- **State:** `required_now`; owner Phases 0, 1, 2, and 14.
- Bind every verification command to an API/schema version and successor/retirement
  record. When an interface changes, migrate or explicitly retire the old tests and
  invalidate their descendant reports; compatibility shims carry an owner and expiry.
- First repair: reconcile the legacy procedural-route effect transaction suite with the
  current dispatch-digest-bound reflexive effect contract. The successor tests pass;
  the six old callers still target a removed keyword and cannot remain an active
  verification surface.
- **Done:** no registered canonical verifier fails because it targets a superseded API,
  and no report remains current solely because an older verifier once passed.

#### `ASI-03` - Cross-layer trace join and terminal reconciliation

- **State:** `required_now` for the contract and `real_use_qualification` for natural
  evidence; owner Phases 1, 2, 3, 5, 14, 15, 16, and 18.
- Extend the existing VIEA/assistant path with one Reference Trace Join Contract that
  binds root request, interpretation version, authority, VCM snapshot and cells read,
  plan nodes, kernel/tool proposals, verifier decisions, route, costs, effects,
  observations, rollback/commit, claims, residuals, feedback, and terminal child
  acknowledgements under one lineage.
- Require one natural happy trace and one natural blocked trace. Blocked, partial,
  cancelled, timed-out, compensated, rolled-back, and escalated paths must be as visible
  as success. Every dispatched child reaches one terminal state; every residual has an
  owner; effect inventory and final observed state reconcile exactly.
- **Done:** independent join audit rejects orphan nodes, request/effect parent mismatch,
  missing terminal acknowledgement, stale projections, residual loss, unobserved effects,
  and incomplete rollback.

#### `ASI-04` - Consolidation, retirement, and one-owner enforcement

- **State:** `required_now`; owner Phases 0 and 19.
- Build a bounded cleanup queue from duplicate scripts, reports, schemas, old chapter
  summaries, superseded verification surfaces, and sidecars that duplicate VIEA, VCM,
  SCF, Octopus, registry, artifact, proof, or evidence ownership.
- Preserve negative evidence and immutable run artifacts; retire execution authority,
  not history. Every retained generated report must have a registered consumer or a
  retention/GC disposition.
- **Done:** one canonical owner per abstraction, no ambiguous active successor, compact
  current-state docs, and a decreasing unowned/stale/duplicate surface count.

#### `ASI-05` - Executable Cognitive Kernel ABI

- **State:** `pretraining_contract`; owner Phases 2, 3, 10, 11, 13, 14, and 16.
- Replace the config-only ABI declaration with an executable protocol/trait and schema:
  `initialize`, `propose`, `accept_receipt`, `checkpoint`, `restore`,
  `parameter_accounting`, and `resource_accounting`. Requests bind task/consumer,
  context and exact-state handles, model/state version, authority ceiling, resource and
  stop policies, evaluator, evidence obligation, and fallback. Proposals bind kernel,
  checkpoint/state schema, mutable-state lineage, assistance, uncertainty, requested
  changes/effects, evidence handles, cost, and non-claims.
- Implement real adapters first for the faithful MoECOT/sparse candidate and matched
  dense control. Add SymLiquid only when its actual state satisfies the contract.
  OneCell remains non-routeable until a real implementation exists.
- **Done:** malicious/incompatible adapter tests reject forged authority, omitted mutable
  state, unaccounted assistance, checkpoint mismatch, incompatible migration, hidden
  effect commits, and incomplete restore. No fake adapter may satisfy conformance.

#### `ASI-06` - Exact/latent state custody and full checkpoint semantics

- **State:** `pretraining_contract`; owner Phases 2, 3, 7, 10, 11, and 14.
- Standardize identity, authority, protected predicates, typed values, program state,
  effects, receipts, provenance, revocation, resource leases, durable memory, online
  adaptation, and checkpoint parentage as exact external state. Keep belief, prediction,
  relevance, similarity, uncertainty, heuristic value, and learned representations
  explicitly non-authoritative.
- Any recurrent state, prefix/KV cache, test-time learning, neural memory, retrieval
  write, optimizer-like online state, or mutable adapter becomes part of provenance,
  privacy, poisoning, deletion, backup, restore, and descendant invalidation.
- **Done:** save/reload/migrate/rollback restores every declared governed surface, not
  weights alone; exact state is never embedded only inside an opaque latent checkpoint.

#### `ASI-07` - Versioned intent interpretation and re-contracting

- **State:** `required_now` contract plus `real_use_qualification`; owner Phases 1, 5,
  and 18.
- Upgrade the live assistant intent record to preserve a privacy-safe root-request
  reference separately from desired outcome; allowed/forbidden means; authority basis
  and ceiling; affected parties; source/privacy/retention/training/publication limits;
  acceptance/evidence requirements; field provenance; confirmed assumptions, bounded
  defaults, contested/open ambiguities; permitted consumers; expiry, revocation,
  correction, appeal, stop, and re-contract conditions.
- Material changes to means, authority, tools, affected parties, sources, publication,
  evidence, support-state effect, or stop conditions create a new contract version and
  cannot silently pass through planning.
- **Done:** natural underspecified/urgent/trust/publication/private-source cases measure
  unauthorized-action rate, clarification burden, useful bounded help, re-contract
  precision/recall, correction/appeal success, and user satisfaction.

#### `ASI-08` - Planning, typed jobs, semantic lowering, and runtime replanning

- **State:** `real_use_qualification`; owner Phases 1, 3, 5, 13, and 15.
- Make the accepted intent contract, typed job, dependency DAG, adequacy contract,
  capability tier, verifier, cost envelope, authority, residual behavior, merge rule,
  and terminal state the ordinary execution unit. Preserve requirement/constraint
  provenance through Semantic IR and artifact lowering.
- Compare direct execution, schema-only, fixed workflow, current planner, and human
  baseline under matched information and resources. Measure decomposition correctness,
  dependency completeness, plan churn, invalid cycles, requirement loss, dispatch
  success, useful completion, replanning quality, and total cost.
- **Done:** a natural task can replan after stale context, failed dependency, changed
  evidence, cancellation, or partial result without widening authority or losing work.

#### `ASI-09` - One empirical daily-use assistant lane

- **State:** `real_use_qualification`; owner Phase 5 with existing Phase 1/3/6/14/15
  dependencies.
- Select one existing low-risk task Corben actually needs and run it repeatedly through
  the joined trace. Do not create a new app or dashboard. Tool-assisted usefulness may
  count as product value but remains separate from learned-generation credit.
- Record invoked/ignored, accepted/missed/corrected/completed, time saved, downstream
  action, abstention, verifier/human cost, effect/rollback result, and whether learned,
  retrieval, tool, workflow, or human components caused the outcome.
- **Done:** multi-day real-use evidence with at least one correction and one failure or
  abstention path; the product gate may move to empirical only from real user events.

#### `ASI-10` - Empirical procedural-memory foundry

- **State:** `real_use_qualification`; owner Phases 5 and 15.
- Mine repeated source-disjoint real traces for reusable procedure structure without
  copying answer-bearing content. Compare against direct execution; require heldout
  positive/negative triggers, shadow execution, cost/usefulness improvement, drift,
  decompilation, rollback, retirement, and descendant invalidation.
- Preserve the existing three adopted metadata workflows as mechanics evidence, not as
  proof of general procedure discovery or learned capability.
- **Done:** at least one procedure discovered from real repeated work earns guarded
  adoption and later survives a natural heldout use; false triggers remain below a
  preregistered ceiling.

#### `ASI-11` - Runtime adapters, approvals, secrets, and effect classes

- **State:** `real_use_qualification` for local effects and `external_dependency` for
  production isolation; owner Phases 1, 2, 5, 6, and 18.
- Keep proposal, interpretation, approval, execution, observation, and evaluation under
  distinct ownership. Expand beyond the bounded route-authority file only when a daily
  task needs a new effect class. Each class requires least privilege, secret substitution
  outside model-visible context, exact inventory, independent observation, idempotency,
  compensation, rollback, denial, confused-deputy, path/symlink, and partial-failure tests.
- **Done:** every enabled effect class has one natural successful transaction and one
  denied/rolled-back transaction; unregistered effects fail closed.

#### `ASI-12` - VCM/context and Semantic IR natural-use campaign

- **State:** `real_use_qualification`; owner Phases 3, 5, 13, and 14.
- Compare full context, ordinary retrieval, schema-only packets, current VCM/QCSA
  identity-address-route packets, and no-memory baselines on natural tasks. Instrument
  exact objects/cells read, omissions, taint, freshness, contradiction, deletion,
  consumer use, and whether the model actually used the supplied context.
- Stress concurrent writers/readers, crash recovery, stale snapshots, poisoned mounts,
  copy-on-write branches, supersession/retraction, compaction, migration, and deletion
  closure. Separately test natural source-plan to Semantic IR to concrete artifact
  preservation with independent source-target and artifact evaluators.
- **Done:** a measured task or cost advantage with no authority/privacy regression, or a
  scoped negative result for the exact implementation; native KV/prefix parity remains a
  separate claim.

#### `ASI-13` - Architecture-freeze package and common comparator identity

- **State:** `pretraining_contract`; owner Phases 0, 7, 8, 10, 11, 14, and 16.
- Extend the existing freeze package with Cognitive Kernel ABI/schema versions, exact
  and latent state inventory, mutable online-state custody, adapter conformance,
  proposal/effect separation, verifier identity, assistance disclosure, complete cost
  schema, migration/rollback, and explicit OneCell/SymLiquid dispositions.
- **Done:** faithful KERC closure plus a content-addressed package for conventional
  surface-English, KERC English candidate, code arms, MoECOT, dense controls, data,
  objectives, evaluators, and total-cost views. No long run begins from an older green
  package.

#### `ASI-14` - Verification-bandwidth allocator and context adequacy economics

- **State:** `real_use_qualification`; owner Phases 3, 8, 14, and 16.
- Replace fixture-derived capacity units and hardcoded latency constants with observed
  wall/CPU/GPU/memory/storage, queue delay, human-review minutes, checks run, failures
  caught, false accepts/rejects, repairs induced, downstream rework avoided, residual
  risk, and marginal value of another check.
- Route among verify now, stronger verifier, cheaper verifier, more context, abstain,
  partial return, escalate, or stop. High-risk work cannot spend nonexistent verifier
  capacity or replace independent checks with self-score.
- **Done:** prospective budgets alter real route decisions and beat a fixed/all-checks
  baseline on verified utility per total cost without raising the no-cheat or safety wall.

#### `ASI-15` - Total lifecycle Resource OS

- **State:** `real_use_qualification`; owner Phase 8 and every producing phase.
- Make total lifecycle accounting ordinary per run: data acquisition, training,
  inference, active/total parameters, memory, storage, retrieval, routing, search,
  verification, repair, human cleanup, migration, recovery, idle/queue time, energy,
  governance, rollback, maintenance, and residual failure cost.
- Preserve equal-active-parameter, equal-total-parameter, equal-train-compute, and
  equal-total-lifecycle-cost views separately. No route may displace cost outside its
  ledger and then claim efficiency.
- **Done:** measured rather than assigned timings for the daily-use and architecture
  campaigns, uncertainty intervals, and reconciliation to system-level resource totals.

#### `ASI-16` - Security kernel, supply chain, weight custody, and epistemic trust roots

- **State:** `pretraining_contract` for loader/admission invariants;
  `external_dependency` for real valuable-weight/hardware-root claims; owner Phases 0,
  2, 7, 8, 14, and 18.
- Preserve requested/resolved/observed identity, Ed25519 admission, advisory freshness,
  anti-rollback generation, revocation, relocation, ownership/mode, purpose, encrypted-
  storage, key-release, and derivative-invalidation gates. Add reproducible second build
  or explicit residual, signed training/build/advisory/custody records, copy/recipient
  inventory, key rotation/revocation rehearsal, and hardware/firmware/operator trust
  assumptions when a valuable routeable checkpoint exists.
- Record the epistemic TCB for every promotion: which producer, verifier, monitor,
  evaluator, human, toolchain, source, and hardware facts are trusted and where
  independence is only nominal.
- **Done:** current loaders fail closed before deserialization; later custody/release
  claims require real keys, encrypted storage, and available hardware evidence rather
  than fixture signatures.

#### `ASI-17` - Load-bearing proof-contract resolver

- **State:** `pretraining_contract` for selected invariants; owner Phases 0, 2, 14,
  and 18.
- Do not port the book's entire Lean inventory. Register only load-bearing predicates:
  non-increasing authority, proposal/effect separation, stale/revoked lease rejection,
  public-benchmark training exclusion, hidden-target information-flow exclusion,
  canonical/projection equality, complete rollback/residual conservation, and complete
  governed checkpoint state.
- Each target binds predicate, source claim, code/schema version, artifact lane,
  verifier command/result, semantic adequacy review, consumer, support-state effect,
  limitations, and non-claims. Finite-model proof does not imply runtime fidelity.
- **Done:** at least one live consumer blocks on each selected invariant and mutation
  controls show the monitor and proof artifact cannot be self-attested by the producer.

#### `ASI-18` - Measured scalable oversight and human-review degradation

- **State:** `real_use_qualification`; owner Phases 4, 8, 14, and 18.
- Replace `*_measured: true` declarations with evidence references and observed metrics.
  Freeze a scoped task cohort and compare direct review, assisted consultation,
  adversarial review, and abstention using an independent outcome adjudicator.
- Measure supervisor envelope, accuracy/calibration by risk, reviewer outcome
  correlation, common model/tool/prompt/information access, persuasion versus
  correctness, disagreement/dissent retention, random-audit coverage, false/missed
  escalation, latency, operator minutes, queue load, fatigue/rubber-stamping, recursion
  bottom, and stop criterion.
- **Done:** one natural oversight campaign beats or fails against a strong direct-review
  baseline under matched access/cost; the exact result remains scoped to that protocol.

#### `ASI-19` - Capability commitments that measure capability

- **State:** `pretraining_contract` for schema semantics and
  `post_behavior_qualification` for real thresholds; owner Phases 2, 12, 14, and 18.
- Rename the current effect-inventory-count record as an effect-exposure safeguard
  commitment or replace its metric with a real domain-scoped capability assessment.
  Bind model/checkpoint, task family, evaluator/instrument, uncertainty, threshold and
  rationale, safeguard obligations, reassessment triggers, exception/expiry,
  compensating controls, residual owner, and route/release effect.
- Non-crossing never authorizes release; crossing never proves safeguards effective.
- **Done:** the first real threshold is frozen prospectively and consumes an actual
  assessment rather than record shape or declared effect count.

#### `ASI-20` - Assurance graphs and evaluation-observation integrity

- **State:** `real_use_qualification`; owner Phases 12, 14, and 18.
- Move beyond selection of the first supported claim and an empty defeater list. Pick a
  consequential Theseus claim prospectively; model top claim, subclaims, hazards,
  evidence, assumptions, counterevidence, defeaters, monitor provenance, evaluator
  independence, context perturbations, discrepancies, freshness, residuals, reviewer,
  decision authority, and rereview triggers as separate graph nodes.
- Require at least one real countercase and one intentionally unresolved or explicitly
  resolved defeater. An empty generated list cannot satisfy a counterevidence search.
- **Done:** independent review may reject or narrow the case; context discrepancy or
  stale dependency invalidates its route effect and descendants.

#### `ASI-21` - Empirical failure-boundary radar and incident lifecycle

- **State:** `real_use_qualification`; owner Phases 0, 8, 14, and 18.
- For stale evidence, hidden target leakage, evaluator common-mode error, unaccounted
  test-time state, construct-invalid proxy substitution, contamination, teacher-cap
  breach, unobserved effect, rollback failure, and authority widening, record detector,
  observable, threshold, false-positive/negative estimate, time to detect/contain,
  affected closure, recurrence, escalation, recovery, disclosure, and retirement.
- **Done:** fault-injection and natural incidents exercise detection through terminal
  containment; dashboards or static taxonomies alone do not satisfy the package.

#### `ASI-22` - Constitutional predicates, value conflict, rights, appeal, and amendment

- **State:** `real_use_qualification`; owner Phases 1, 5, 14, and 18.
- Make least-sufficient-power, dignity/agency, consent, privacy, correction, refusal,
  exit, export, deletion, dissent, fork, appeal, self-modification, and affected-party
  constraints executable consumers of intent, plans, jobs, effects, policy updates,
  and steward actions. Preserve uncertainty and minority reports rather than reducing
  conflicts to one score.
- Treat charter/policy changes as versioned amendment events with proposer authority,
  affected scope, notice/review, conflict checks, expiry/rollback, and legitimacy
  residuals. No steward or model may amend its own authority unilaterally.
- **Done:** natural conflict, correction, appeal, and amendment cases alter real route
  decisions and remain visible through the joined trace.

#### `ASI-23` - Evaluation integrity, sandbagging, reward hacking, and interpretability

- **State:** `post_behavior_qualification` except for pre-training information-flow and
  provenance contracts; owner Phases 4, 7, 10, 12, 14, and 18.
- Freeze elicitation, selection, reward, monitor, and evaluator provenance. Use blinded
  holdouts where feasible; trusted/untrusted monitor separation; cross-context,
  paraphrase, pressure, tool-access, and incentive perturbations; capability-elicitation
  controls; discrepancy quarantine; and independent replay.
- Treat interpretability features, probes, circuits, traces, and monitor scores as
  scoped evidence with stability, causal intervention, false-positive, and evaluator
  limits. Hidden reasoning is neither required evidence nor execution authority.
- **Done:** only after a capable model exists, measure strategic underperformance,
  reward hacking, monitor interference, and trace/action divergence against strong
  baselines; until then, preserve contract-only support.

#### `ASI-24` - Routing, deliberation, search, and fast-generation qualification

- **State:** `post_behavior_qualification`; owner Phases 6, 8, 10, 12, 14, and 16.
- After nonzero direct behavior, freeze naturally ambiguous requests and compare direct,
  specialist/generalist, learned/rule routing, abstention/fallback, bounded revision,
  candidate search, tool-assisted, and deliberative routes. Measure route versus answer
  success separately, calibration/selective risk, first-hit/last-correct, overthinking
  harm, verifier disagreement, branch credit, accepted verified output per second,
  latency/memory/energy, and total lifecycle cost.
- MTP, speculative decoding, self-draft, diffusion, LayerSkip, and other fast modes earn
  adoption only at matched verified quality. Extra tokens, self-scores, and branches are
  not evidence.
- **Done:** causal ablations show which route/mode improved verified output and the
  stopping policy beats fixed-compute baselines without weak-tail regression.

#### `ASI-25` - Policy learning, feedback, and open-ended improvement efficacy

- **State:** `post_behavior_qualification`; owner Phases 5, 7, 10, 14, 15, and 18.
- Preserve the seven-target update lease, GVR kernel, objective contracts, disabled
  autonomous campaign, full-state journal, negative knowledge, best/final identity,
  debt ceilings, stop authority, and rollback. Activate only after a behavior-positive
  proposer and independent verifier create a real numerator.
- Compare no-update, supervised, preference, verifier-reward, and other selected
  objectives under matched data/compute/tuning and full verifier/repair/human cost.
  Track teacher share, self-generated verified share, forgetting, reward hacking,
  distribution shift, weak tails, and rollback triggers.
- **Done:** useful heldout behavior improves beyond matched controls and survives monitor
  window, reproduction, rollback rehearsal, and no-cheat audit.

#### `ASI-26` - Behavioral continual learning, unlearning, privacy, and descendant erasure

- **State:** `post_behavior_qualification`; owner Phases 3, 7, 10, 12, and 14.
- Extend the strong 13-kind full-state mechanics to measured acquisition/retention,
  forgetting, source influence, deletion efficacy, membership/privacy leakage,
  retained-capability damage, backups/caches/indexes/adapters, external-effect receipts,
  retraining cost, and exact rollback.
- Preserve source purpose, consent, license, synthetic ancestry/share, teacher caps,
  contamination, retention, export/correction/deletion rights, and multi-user isolation.
- **Done:** a learned datum or governed group can be removed with independently measured
  influence/privacy reduction and bounded collateral damage; package deletion alone is
  not behavioral unlearning.

#### `ASI-27` - Compression, mathematical/search, cyclic, and alternative-substrate discovery

- **State:** `post_behavior_qualification` for adoption; protected discovery lane under
  Phases 6, 11, 13, and 14.
- Preserve RankFold/NeuralFold, CGS, Circle, Coil Attention, CoilRA, SymLiquid, VSA,
  liquid/KAN-like, program-search, and exact mathematical substrates as separately
  identified candidates. Every campaign declares the causal mechanism, faithful
  implementation, learnability/gradient/checkpoint/intervention sanity, strong ordinary
  baselines, matched raw data/compute/tuning/inference/verifier/total cost, multiple
  seeds, uncertainty, weak tails, source-disjoint evaluators, fallback, residuals, and
  transfer.
- Compressed artifacts are consumer-scoped candidates and keep the full artifact plus
  residual/fallback until utility, decode fidelity, and downstream probes admit them.
- **Done:** adopt or retire only the exact candidate/regime tested; discovery may not be
  absorbed into the transformer survival lane or broadly falsified by a toy proxy.

#### `ASI-28` - Deferred and external operations without roadmap loss

- **State:** `external_dependency` or `deferred_by_breadth_freeze`; owner Phases 9, 17,
  18, and the relevant existing phase only after entry.
- Retain explicit entry conditions for: real multi-node Hive/federation; inter-stack
  credentials/delegation/value/dispute/shutdown; public gateway/market operation;
  confidential hardware roots; multi-user memory/privacy; vision/audio/sensor grounding;
  world models; embodied actuation; autonomous scientific experiments; replication/
  proliferation controls; incident response/decommissioning; mobile/Watch/spatial/voice;
  and broader multi-agent economies.
- **Done now:** no implementation lane, dashboard, or training blocker exists. **Entry:**
  a current priority needs the capability, the required environment exists, authority is
  explicit, and the new work reuses existing owners rather than duplicating the stack.

#### `ASI-29` - Source-gap-owner-evidence graph and roadmap continuity

- **State:** `required_now`; owner Phase 19.
- Extend the existing matrix/registry rather than creating a new doc family. Every
  source/chapter atom and accepted Theseus obligation links to owner, dependency,
  implementation, verifier, current evidence, support state, residual, next falsifiable
  action, defer/retire condition, and public-safe return path to the book.
- Closing any active roadmap requires one exact successor in the same transaction;
  completed programs remain immutable history and do not reacquire authority.
- **Done:** the machine view can answer what is missing, why, who owns it, what blocks
  it, what evidence would move it, and whether it is required now without reading a
  pile of reports.

#### `ASI-30` - Shared governed-lease envelope

- **State:** `required_now` for schema consolidation and `pretraining_contract` where a
  lease governs checkpoint/update authority; owner Phases 0, 2, 3, 7, 14, 16, and 18.
- Normalize the recurring SCF substitution, VCM context, node/job, policy-update,
  benchmark-instrument, capability-commitment, artifact-use, data/source, kernel-route,
  and steward-action lease semantics without creating another execution engine. The
  shared envelope binds lease/root identity, exact subject and consumer, use/scope,
  authority ceiling, version/schema, evidence prerequisite, start/expiry, revocation,
  correction/appeal, re-contract triggers, monitor/requalification, rollback,
  descendant invalidation, residual owner, and non-claims; domain payloads remain owned
  by their existing modules.
- Adapters may translate legacy records during a finite migration, but cannot discard
  authority, expiry, revocation, consumer, or rollback fields. Wrong-consumer, stale,
  superseded, widened-authority, missing-evidence, and unacknowledged-revocation cases
  fail closed across every adopting lease family.
- **Done:** at least policy-update, VCM context, SCF substitution, and one runtime/effect
  route consume the same envelope and pass cross-family replay/mutation tests; no two
  lease families disagree on expiry, revocation, or authority-order semantics.

### Mechanism-level coverage inventory

Every load-bearing ASI Stack mechanism should eventually be represented in Theseus
(see ASI Stack Completion Invariants for what "represented" requires). This detailed inventory
is subordinate to the work packages and dependency waves above; it cannot override
their sequencing or create a new flagship lane.

| Book mechanism | Remaining gap | Phase |
|---|---|---|
| Stable Capability Fields | evidence for live implementations stale; routing eligibility mostly blocked | 2 |
| Replacement + rollback | replacement transactions not yet the ordinary route for model/router/tool changes | 2 |
| Effect-complete governed execution | useful release, first/final effect identity, observer separation, exact rollback inventory, and governance tax are not one ordinary assistant transaction | 1, 2, 5, 14, 18 |
| Recursive self-improvement boundaries | canonical transition record not yet universal across policy, data, route, and model changes | 2, 14 |
| Open-ended improvement campaigns | pre-training mechanics are canonical: distinct generator/evaluator/promoter/stop/rollback authorities, single-axis matched-budget challengers, frozen holdout, novelty/coverage/weak-tail decisions, negative-knowledge retention, six-field debt ceilings, explicit shutdown handoff, hash-chained journal, full-state rollback, and fourteen rejecting mutations. Runtime activation and learned improvement remain unproven and disabled | 7, 10, 12, 14, 15, 18 |
| Intent-to-execution contracts | structured runtime records exist, but the live assistant contract is thinner than the plan compiler and lacks versioned ambiguity, field provenance, affected-party, expiry/revocation/appeal, and material re-contract semantics on natural tasks | 1, 5, 18 |
| PlanForge DAGs + adequacy + arbitrage | typed DAG/reflexive mechanics are wired; natural decomposition quality, requirement preservation, live replanning, and matched intelligence-arbitrage evidence remain unproved | 1, 5, 13 |
| Cognitive compilation / semantic IR | failures reported more than repaired through IR-level localized feedback | 13 |
| Kernel English + Hierarchical Residual Ledger | reopened after adequacy audit: exact packet/residual/codec interfaces and shared-trunk mechanics exist, and K1a now independently replays 256 bounded multi-frame/multi-claim MASC records without inflating source credit. The old campaign still tested a keyword/linear proxy, while broader semantics, typed nonliteral values, per-unit allocation, real global interaction dictionaries, coordinated learned stages, semantic verification, and utility remain absent or unproven. Complete K1-K8 before refreezing | 0, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19 |
| Data engines + continual learning + unlearning | canonical admission now carries 115,407 candidate receipts, five frozen policy simulations, contamination controls, 125,702 lineage edges, a 13-kind full-state inventory, descendant deletion, and bounded storage-erasure receipts. Real forgetting, influence, privacy, and unlearning efficacy remain post-training measurements | 3, 7, 12, 14 |
| Full-state update causality | pre-training mechanics are unified: model/optimizer/scheduler/RNG/cache/index/checkpoint/backup/effect state is content-addressed, best and final authority are distinct, packages replay exactly, update exposure is lineage-bound and bounded, deletion reaches descendants, and rollback restores the exact pre-state digest. Behavioral unlearning and undeclared external erasure are explicit non-claims | 0, 2, 7, 10, 14 |
| Durable semantic memory inside VCM | wired: stable objects, typed/temporal relations, additive ontology migration, transactional merge/supersession/retraction/compaction, sparse-vector/graph retrieval, bounded snapshots, and fresh-process replay; dense embedding and parametric unlearning remain explicit non-claims | 3, 14, 15 |
| Question-Compiled Semantic Addressing | bounded pre-training disposition complete: VCM now owns stable SOID/address/route indirection, three authoritative facets, Semantic Address Certificates, authority-safe translation, and exact migration/rollback; the matched source campaign retired the full objective and adaptive-question policy from the first long run because they added cost without task advantage | 0, 1, 2, 3, 6, 8, 13, 14, 16, 18, 19 |
| Verification bandwidth | route/verifier capacity and governance-tax records exist, but current units and latency are synthetic formulas/constants rather than observed marginal value, false-accept/reject, human-load, and displaced-cost evidence | 3, 8, 14, 16 |
| Claim ledgers + belief revision | claim/transition/contradiction records are widely materialized; canonical/projection freshness, natural-run causal use, and independent downgrade/defeater behavior remain incomplete | 0, 14, 19 |
| Proof-carrying + tribunal/adversarial review | broader independent-review records not standard for architecture changes | 14, 18 |
| Labor OS typed jobs / artifact graphs | typed job + unified artifact graph not the universal unit | 1, 14 |
| Procedural memory | three metadata workflows have guarded adoption mechanics; real repeated-work discovery, direct-execution baselines, natural heldout reuse, false-trigger measurement, drift, and retirement remain missing | 5, 15 |
| Routing heads + MoECOT + specialist cores | v8 shared trunk and five specialist deltas are trained and arm-evaluated; matched dense verdict, confirmation, live arm-card binding, typed composition, and route-versus-answer qualification remain | 10, 16 |
| Reflexive Router / compiled reflexes | pre-training mechanics are wired through the canonical assistant: typed command/reflex registries, finite outcomes, precedence and qualification, Chronicle/cache invalidation, structured concurrency, sole-kernel reversible effects, compilation/decompilation interfaces, and a frozen 32-case/eight-track/ten-policy mechanics profile. Learned calibration, real route regret, diverse trace promotion, drift response, and answer-quality benefit remain post-training evidence | 0, 1, 2, 3, 5, 6, 8, 14, 15, 16, 18, 19 |
| Ambiguous routing + adaptive deliberation | naturally ambiguous route corpus, fallback/abstention calibration, first-hit/last-correct, overthinking harm, branch credit, and verifier-disagreement accounting are missing on useful model outputs | 4, 6, 8, 10, 14, 16, 18 |
| Readiness gates + residual escrow | stale outputs/duplicate families reduce trust | 0, 2 |
| Generate-verify-repair | canonical repair now consumes a six-state GVR kernel with immutable candidate identity, independent verifier binding, localized repair authority, tamper-evident transitions, rollback, and zero learned credit for assisted/fallback output. Current repaired behavior remains 0/2, so efficacy is still unproven | 6, 10, 13 |
| Deterministic math/search substrates | not yet attached universally to planning/VCM/claim-ledger/assistant | 6 |
| Benchmark ratchets | public-transfer story stale; remeasure only after private correctness improves | 12 |
| Policy optimization (DPO/GRPO/RLVR) | shared update authority covers seven policy targets, and DPO/IPO/ORPO/KTO/SimPO plus GRPO/RLOO/ReMax/RLVR now execute through one disabled, provenance-bound, checkpointed, MLX-parity-tested objective contract with verifier-capacity and reward-hacking controls. Learned efficacy and objective selection remain post-training evidence | 7, 10 |
| Integrated reference architecture | many canonical component records exist, but one natural request has not yet produced an independently joined happy trace plus blocked trace with complete parentage, terminal acknowledgement, effect/residual conservation, and dogfood outcome | 1, 2, 3, 5, 14, 15, 16, 18 |
| Authority kernel / SCIF / failure boundaries / adapter receipts | no universal authority-transition ledger or SCIF receipt layer | 18 |
| Scalable oversight | role separation, private-state partitions, risk routing, random-audit policy, and rejecting fixtures are wired synthetically; `*_measured` fields are not backed by outcome references, and direct-review baselines, weak-supervisor results, real correlation, persuasion, fatigue, operator load, and recursion-bottom evidence remain missing | 4, 8, 14, 18 |
| Capability commitments + safety cases | a synthetic commitment and one assurance/evaluation-integrity case exist, but the commitment metric is declared local effect count rather than capability and the selected-claim assurance case has no real counterevidence/defeater campaign; rebuild or rename before route authority | 2, 12, 14, 18 |
| Weight custody + AI supply chain | local AIBOM, descendant invalidation, fail-closed Ed25519 artifact admission, advisory/revocation/anti-rollback, relocation, and valuable-weight custody requirements exist; the current AIBOM still honestly reports unsigned supply chain, no reproducible second build, no real weight custody, and no hardware-root evidence | 0, 2, 7, 8, 14, 18 |
| Inter-stack identity and exchange | federation lacks credential/delegation expiry, reserved budgets, value/accounting receipts, dispute, revocation, and shutdown-handoff interoperability | 1, 2, 9, 16, 18 |
| Constitutional predicates / agency / value conflicts / governance rights | not first-class records consumed by planning and self-improvement | 18 |
| Resource budgets + costed routes + generation-mode records | many surfaces emit cost records, but actual wall/CPU/GPU/memory/storage/human/migration/recovery/residual totals are fragmented and not reconciled as the ordinary total-lifecycle artifact | 8, 14 |
| Simulation contracts | no contract separating map from territory | 17 |
| Semantic atoms/nodes + compact artifacts + substrate adoption | not the common substrate; adoption discipline not universal | 11, 13, 14 |
| Research backlog | chapter crosswalk and source-sync machinery exist, but no canonical atom-to-owner/dependency/implementation/verifier/evidence/residual/next-action graph spans the live 55-chapter book | 0, 19 |
| Receipt faithfulness / record-reality gap | extensive mutation controls exist, but registered API/verifier migrations, stale descendant reports, declared measurement booleans, generated empty-defeater lists, and canonical/projection disagreement still permit record/reality gaps | 0, 14, 18, 19 |
| Epistemic trusted computing base | roots of trust, verifier independence, bounded trust propagation not yet explicit | 14, 18 |
| Human oversight degradation | no measure/guard for review-capacity capture and rubber-stamping | 8, 18 |
| Partition / distributed authority governance | stale-grant/revocation-race behavior across Hive nodes not modeled in runtime | 9, 18 |
| Interpretability evidence discipline | interpretability claims not held to the same support-state/non-claim bar | 13, 14 |
| Amendment legitimacy | governance/charter changes not treated as first-class amendment events with legitimacy checks | 18 |

### Complete 55-chapter ownership index

This index prevents a chapter from disappearing merely because its implementation is
grouped into a cross-cutting package. `strong_mechanics` means preserve and qualify,
not that the chapter core is demonstrated. `deferred` means the entry condition is
retained under `ASI-28`; it is not permission to start the lane.

| ASI Stack chapter | Primary package(s) | Current roadmap disposition |
|---|---|---|
| ASI Is a Stack, Not a Model | `ASI-03`, `ASI-29` | `required_now`: prove composition with joined traces rather than more component records |
| The Efficient ASI Hypothesis | `ASI-15`, `ASI-27` | `post_behavior_qualification`: total lifecycle efficiency and matched architecture evidence |
| System Boundaries and Authority | `ASI-03`, `ASI-11`, `ASI-22` | `strong_mechanics`; broaden only through required natural/effect cases |
| Failure Modes of Ungoverned Intelligence | `ASI-21` | `real_use_qualification`: empirical detection/containment radar |
| Evidence States and Claim Discipline | `ASI-01`, `ASI-20` | `strong_mechanics`; close projection freshness and defeater use |
| Scalable Oversight and Adversarial AI Control | `ASI-18` | synthetic scaffold exists; measured natural campaign pending |
| Human Intent as a Formal Input | `ASI-07` | thin runtime contract; versioned interpretation and re-contracting required |
| Constitutional Alignment: Agency, Dignity, and Corrigibility | `ASI-22` | predicates exist; natural correction/refusal/appeal consumption pending |
| Moral Uncertainty, Value Conflict, and Contestable Governance | `ASI-22` | preserve dissent/contestability; no universal value learner |
| Stable Capability Fields | `ASI-01`, `ASI-13`, `ASI-30` | `strong_mechanics`; bind live evidence freshness and kernel qualification |
| Capability Replacement and Rollback | `ASI-03`, `ASI-11`, `ASI-25` | `strong_mechanics`; make replacement ordinary for each admitted change class |
| Security Kernel and Digital SCIFs | `ASI-11`, `ASI-16` | structural boundary exists; real isolation/secrets evidence is partial/external |
| Model-Weight Custody and Hardware Roots of Trust | `ASI-16` | loader contract now; valuable-weight/hardware proof later |
| AI Supply-Chain Integrity and Lifecycle Provenance | `ASI-01`, `ASI-16` | AIBOM/admission strong; signing/reproducibility/custody incomplete |
| Recursive Self-Improvement Boundaries | `ASI-25` | `strong_mechanics`; learned efficacy remains disabled |
| Open-Ended Improvement Engines | `ASI-25` | disabled controller preserved; behavior-positive entry required |
| Command Contracts: From Intent to Executable Work | `ASI-03`, `ASI-07`, `ASI-08` | natural end-to-end preservation pending |
| Planning as a Control Layer | `ASI-08` | mechanics wired; decomposition, replanning, and arbitrage evidence pending |
| Cognitive Compilation and Semantic IR | `ASI-08`, `ASI-12` | natural obligation-preservation campaign pending |
| The Virtual Context ABI | `ASI-06`, `ASI-12`, `ASI-30` | `strong_mechanics`; natural use, model-use attribution, and native parity separate |
| Context Transactions, Snapshots, Mounts, and Taint | `ASI-12` | stress concurrency/crash/poison/stale/deletion paths |
| Verification Bandwidth and Context Adequacy | `ASI-14` | synthetic accounting exists; observed allocation/economics pending |
| Claim Ledgers and Belief Revision | `ASI-01`, `ASI-20` | `strong_mechanics`; natural causal downgrade and projection closure pending |
| Proof-Carrying Claims and Adversarial Review | `ASI-17`, `ASI-20` | proof receipts exist; live adequacy/countercase consumption pending |
| Labor OS and Typed Jobs | `ASI-08`, `ASI-28` | local typed jobs now; real worker/fleet economics later |
| Artifact Graphs, Audit Logs, and Replay | `ASI-01`, `ASI-03` | `strong_mechanics`; stale projection and joined terminal reconciliation pending |
| Runtime Adapters, Tool Permissions, and Human Approval | `ASI-11` | one effect class real; add only task-required classes with full controls |
| Inter-Stack Protocols, Identity, and Economic Exchange | `ASI-28` | local fixtures retained; remote identity/value/network operation deferred |
| Procedural Memory and Cognitive Loop Closure | `ASI-09`, `ASI-10` | three guarded routes; empirical discovery/reuse pending |
| Routing Heads and Specialist Cores | `ASI-05`, `ASI-24` | mechanics wired; answer-quality numerator required before qualification |
| Replaceable Cognitive Substrates: Beyond Transformer Monoculture | `ASI-05`, `ASI-06`, `ASI-13`, `ASI-27` | new live-book intake; executable common ABI required, tournament later |
| Readiness Gates, Residual Escrow, and Quarantine | `ASI-01`, `ASI-13`, `ASI-30` | `strong_mechanics`; keep freshness and residual ownership exact |
| Personal Compute Hives and Federated Edge Intelligence | `ASI-28` | trusted second-node evidence external; no current blocker |
| Compact Generative Systems: Generate, Verify, Repair, and Residual Honesty | `ASI-24`, `ASI-27` | scoped mechanics/negatives retained; useful behavior pending |
| Fast Generation Architectures | `ASI-24` | topology disposition preserved; matched accepted-output qualification later |
| Governed Deliberation and Test-Time Scaling | `ASI-14`, `ASI-24` | contract belongs in existing route; empirical campaign waits for behavior |
| RankFold, NeuralFold, and Artifact Compression | `ASI-27` | consumer-scoped admission with residual/fallback; no broad adoption yet |
| Resource Economics and Token Budgets | `ASI-14`, `ASI-15` | fragmented/synthetic accounting must become observed lifecycle economics |
| Mathematical and Search Substrates | `ASI-24`, `ASI-27` | deterministic tools useful; causal contribution and total cost pending |
| Circle Calculus and Proof-Carrying AI Contracts | `ASI-17`, `ASI-27` | synthetic proof-contract substrate; no architectural privilege |
| Coil Attention, Cyclic Memory, and Recurrence Contracts | `ASI-27` | protected discovery candidate; matched causal qualification required |
| CoilRA, MultiCoil RoPE, and Cyclic Mixers | `ASI-27` | protected discovery candidate; matched causal qualification required |
| Executable Specifications and Lean Proof Envelope | `ASI-17` | select load-bearing invariants; do not port the full book proof inventory |
| Benchmark Ratchets and Anti-Goodhart Evidence | `ASI-01`, `ASI-23`, `ASI-24`, `ASI-30` | `strong_mechanics`; fresh public use only after material private improvement |
| Capability Thresholds and Deployment Commitments | `ASI-19`, `ASI-30` | current effect-count scaffold must be renamed or rebuilt around capability |
| Adversarial Evaluation, Sandbagging, and Training-Time Deception | `ASI-23` | pre-training provenance now; real capable-model tests later |
| Safety Cases and Structured Assurance | `ASI-20` | one synthetic case exists; real hazards/counterevidence/defeaters pending |
| Policy Optimization and Learning from Feedback | `ASI-25`, `ASI-30` | leases/objectives strong; matched learned improvement pending |
| Data Engines, Continual Learning, and Unlearning | `ASI-06`, `ASI-26`, `ASI-30` | state mechanics strong; behavioral forgetting/privacy evidence later |
| Artifact Steward Agents and Living Project Governance | `ASI-04`, `ASI-22`, `ASI-29`, `ASI-30` | governance strong; do not grant ambient autonomous management authority |
| Integrated Reference Architecture | `ASI-03` | major missing natural joined happy/blocked vertical |
| Project Theseus as Report-First Implementation Reference | `ASI-01`, `ASI-02`, `ASI-29` | preserve report discipline; reports lose authority on source/verifier drift |
| Prototype Roadmap | `ASI-29` | phase machinery strong; successor continuity and generated summaries required |
| Living Book Methodology | `ASI-00`, `ASI-29` | immutable pin process strong; current 55-chapter reconciliation due |
| Open Research Agenda and Bibliography Plan | `ASI-29` | unify source-gap-owner-evidence graph in existing machine contract |

## Book-Futures Intake (classified, phase-routed)

The pinned Theseus crosswalk has 54 chapters; the live ASI Stack now has 55. The
new replaceable-substrates chapter is owned by `ASI-05`, `ASI-06`, `ASI-13`, and
`ASI-27` and enters the next reviewed pin transaction rather than a parallel lane.
Ten items from the earlier completeness
intake have been admitted and are owned by the chapter crosswalk rather than this
future queue. The 13 remaining candidates/section routes below are **not new parallel
lanes**. Each is owned by existing phases and classified by adoption cost. A candidate
that would invalidate the first long-run architecture joins the finite pre-training
docket; a checkpoint-compatible or modality/peer/real-use-dependent candidate remains
post-training and blocks only the higher-authority operation it governs. Schemas alone
never satisfy the docket: selected mechanisms require registered runtime ownership,
negative tests, migration/rollback, resource accounting, and a frozen experiment or
explicit retirement decision.

| Book candidate or section | Theseus owner phases | Entry condition / disposition |
|---|---|---|
| Reasoning-Trace Faithfulness (section route) | 1, 4, 14, 18 | source queue now includes information-flow faithfulness and MonitorBench comparators; distinguish traces from receipts, test trace/action and monitorability under pressure, and never treat hidden reasoning as authoritative |
| Question-Compiled Semantic Addressing | 0, 1, 2, 3, 6, 8, 13, 14, 16, 18, 19 | pre-training disposition complete: retain stable SOID-address-route indirection, plural atlas facets, SACs, authority-safe translation, and exact migration/rollback under VCM/SCF/VIEA; retire the full objective and adaptive questions from the first run after the matched campaign found no task advantage and higher cost; learned semantic-routing efficacy remains post-training |
| Kernel English with Hierarchical Residuals | 0, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19 | pre-training implementation open: preserve exact-object and VCM residual work, replace the toy proxy with all learned stages, prove mechanism use and learnability, freeze matched strong baselines and total-system accounting, and run no long campaign until the rebuilt architecture package passes |
| World Models and Model-Based Cognition | 10, 13, 17 | V-JEPA 2 is now a source comparator; activation still waits for the core proposer floor and requires prediction/error ledgers, model-predictive control, imagination/search, causal limits, and sim-to-real controls |
| Multi-Agent Systemic Risk and Agent Economies | 9, 16, 18 | before multi-agent economic/autonomous operation; collusion, cascades, miscoordination, market behavior, gradual disempowerment |
| Persuasion, Epistemic Security, and Human Agency | 5, 14, 18 | source-contingent; require stronger empirical anchors before activation or claims |
| CAIS, Embedded Agency, and Corrigibility Foundations (section route) | 1, 2, 14, 18, 19 | Embedded Agency source is now mapped; use it to constrain self-model, delegation, subsystem-alignment, and authority claims, but keep it foundations-only unless a distinct mechanism and proof program earns a chapter boundary |
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
14/16/18); immutable-source/protected-object capture, uncertainty-preserving
surface-to-Kernel compilation, separate semantic identity/runtime code, hierarchical
interaction residuals, dual vocabularies, grammar-safe macros, round-trip fidelity,
and total-system rate/compute/fidelity accounting (Phases 0/2/3/4/5/7/8/10/11/13/
14/16/18);
OOD/active perception plus causal/counterfactual/identifiability boundaries
(Phases 13/17); temporal-logic monitors, interlocks, degraded operation, and
sim-to-real residuals (Phases 17/18); AIBOM/reproducible-build/derivative-revocation
plus privacy budgets and cross-user leakage (Phases 0/3/7/14/18); and objective
fingerprints, periodic recontracting, time-horizon/automated-R&D measures,
dual-use profiles, provider concentration, portability, and shutdown handoff
(Phases 2/8/9/18).

## Quality Bars (definition of "done")

### ASI Stack Completion Invariants
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
beyond-SOTA mature endpoint (per the Implementation Standard), cites the relevant
strong comparator, and states which bounded envelope—plumbing, mechanics, executable,
empirical, deployed, or transferred—is actually complete. A bounded completed slice
does not complete the mature endpoint, and an unearned mature endpoint does not block
unrelated critical-path work.

### Decision-Grade Negative Result
A negative result may change a route or retire a mechanism only when its evidence pack
contains all of the following and an independent audit recomputes them:
- a source-to-implementation fidelity matrix with every claimed causal stage marked
  faithful, approximate, absent, or inactive;
- learnability, gradient-flow, tiny-subset overfit, checkpoint/reload, intervention,
  ablation, and module-use receipts;
- strong current baselines with matched data, compute, tuning opportunity, inference,
  verifier/search budget, and total-system cost;
- frozen source-disjoint tasks and independent evaluators that exercise the claimed
  advantage, including weak tails and adversarial cases;
- multiple seeds, paired effect sizes, confidence intervals, meaningful-effect floor,
  prospective power/sensitivity, and replication state;
- exact claim scope, residuals, implementation faults, harness faults, non-claims, and
  the next discriminating experiment.

Failure to satisfy this definition is not evidence of success, but it is also not a
mechanism-level failure. It produces an inconclusive state and returns the missing work
to the existing owner. This applies symmetrically to favored and conventional ideas.

## Success Definition

Theseus is on track when all of this is true:
- The repo has one canonical control spine and every serious task passes through it.
- The registry answers what Theseus is, what each capability means, what backs it,
  and what can replace it.
- The assistant is useful in daily work and logs real accepted/missed/ignored/
  corrected/completed outcomes.
- The practical generator improves on verifier-passing private family-disjoint
  heldouts without cheating, and the capability-per-active-parameter scoreboard
  shows a real, measured trajectory (or a decision-grade, correctly scoped negative).
- Public benchmark calibration stays measurement-only and becomes more informative,
  not more gamed.
- Deterministic tools, VCM, STS, planning, verifier-guided search, and routers are
  integrated as one system, not separate proof islands.
- SymLiquid remains a protected research comparator; the practical lane uses the
  best evidence-backed architecture.
- The book can point to Theseus as an implementation reference with honest support
  states, not inflated claims.
