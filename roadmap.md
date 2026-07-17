# Project Theseus Roadmap

Consolidated 2026-07-16; bound to AI Stack commit
`32635eb94ded42a5f54e528302685cab343993b7`. This
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

## Current Operating Decision: Architecture Before Long Training

The current priority is **finite architecture closure**, not corpus-scale optimizer
work. Theseus will not spend a long training run on an architecture that is already
known to be missing accepted representation, objective, context, routing, generation,
update, checkpoint, or self-improvement contracts. The 13-item matrix-owned docket
briefly reached 13 recorded dispositions, but the 2026-07-17 evidence-adequacy audit
reopened KERC: its retirement relied on a hand-coded/linear proxy that did not exercise
the proposed learned architecture. The previous freeze package is therefore stale and
training authority is denied until KERC reaches a faithful pre-training boundary and a
new package replays.

This is not a demand to prove learned efficacy without learned weights. Pre-training
closure proves that selected mechanisms are real, integrated, checkpointable,
migratable, resource-accounted, replayable, negatively tested, and represented in the
frozen campaign. Tiny finite-update canaries of at most eight optimizer steps remain
allowed for mechanics. Full-corpus training, architecture tuning from interim losses,
public calibration, and capability claims remain after the freeze.

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

The only optimizer work allowed before architecture freeze is a mechanics canary of at
most eight steps whose purpose is to test tensor flow, state capture, checkpoint/reload,
resume equivalence, cleanup, or a falsification condition. Canary metrics cannot select
hyperparameters, tune architecture from held-out outcomes, support a capability claim,
or expand into corpus training. Data inspection and data-contract replay may continue,
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
| 1 | Kernel English + hierarchical residual compiler | `partial` / `INCONCLUSIVE_IMPLEMENTATION` | implement the learned compiler, Kernel reasoner, residual allocator, renderer, verifier, and a decision-grade matched campaign; the prior linear proxy cannot retire KERC |
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

Twelve dispositions remain mechanically closed. KERC is the sole reopened
checkpoint-shaping obligation because its prior negative receipt lacked implementation
fidelity. This does not reopen unrelated architecture or authorize novelty-by-analogy.
After faithful KERC integration, the final content-addressed cross-owner package must be
rebuilt; learned efficacy still requires the prospectively frozen joint campaign.

## Status at a Glance

| Area | Owner | State | Next concrete action |
|---|---|---|---|
| Data engine + curriculum | Track 0 / Phase 7 | GREEN for the frozen 57.315M proposal: 1,293,454,903 broad unique positions versus 1,146,808,520 required; every task-complete floor is GREEN; HTML/CSS now has 71,416,629 unique positions versus 50,083,860 required; source/vocabulary/supervision identities replay exactly | preserve the content-bound corpus and receipts unchanged while the finite architecture docket closes; any architecture change that alters ownership or tokenization must recompute, never assume, these floors |
| Dense transformer control | Phase 10 | both v8 controls completed unchanged; each is 0/544 exact, with Python syntax 46/128 (active) and 48/128 (total) | retain as immutable 10.8M-rung falsification evidence; do not spend confirmation or patch this rung |
| MoECOT language-specialist seed | Track 1 / Phases 10, 16 | v8 trunk/specialists are complete; exact recovery is English 1/128 and code 0, while frozen functional utility is 0/160 | retain as negative modular evidence; do not let routing hide the behavior-zero result |
| Verifier-guided search | Track 2 / Phases 6, 10 | architecture wired, amplifier waiting for signal | preserve the bounded kernel and replay contract; qualify it only after one-shot generation sometimes succeeds and search materially increases held-out pass |
| Correctness training (DPO->GRPO/RLVR) | Track 3 / Phase 10 | learned optimization remains premature at the zero-pass floor, but all five offline-preference and four verifier-reward objective adapters now execute through a shared frozen schema, checkpoint/reference identity, MLX parity probe, capacity ledger, update lease, exact rollback, and twelve rejecting controls | preserve zero exposure and run/select an objective only after a behavior-positive proposer creates honest private verifier signal |
| Fast-gen modes (MTP/diffusion/self-draft) | Track 4 / Phases 8, 10 | pre-training topology is closed: AR is canonical; checkpointed low-rank MTP is campaign-bound at zero initial loss scale; Medusa/EAGLE/LayerSkip/sketch-first are explicitly retired; speculative decode is disabled and post-hoc compatible. Canonical MLX forward/loss/save/reload passes at 21.875% output-head overhead with zero reload drift | preserve the frozen disposition; set any nonzero MTP schedule prospectively and adopt any fast mode only from accepted verified output per second under matched quality/cost accounting |
| Generator capability (held-out utility) | Phase 10 | the 10.8M scale regime is closed; the 57.315M package previously replayed, but its authority is invalidated by the KERC adequacy correction. No long optimizer run or new capability evidence exists yet | complete faithful KERC architecture/data/objective/checkpoint integration, add its matched English-arm alternative, rebuild the freeze package, then begin the unchanged joint campaign |
| Cognitive-kernel discovery | Phase 11 | OneCell-RWM is non-routeable and retired from the first language campaign with zero optimizer/checkpoint exposure; its ABI, exact/latent boundary, objective/checkpoint groups, owner reuse, and separate successor-campaign prerequisites are content-bound, but no OneCell substrate or learned capability is claimed | preserve the successor experiment without appending it to first-campaign weights; re-enter only after exact substrate and a separately preregistered matched cognitive-kernel campaign exist |
| Kernel English + hierarchical residual compiler | Phases 3, 10, 13, 14, 16 | `partial`: exact packets/VCM residual mechanics now feed an explicit MLX KERC model with trusted stage routing, train-only content-bound V_K/V_P codebooks, disjoint Kernel/pointer/surface output support, learned four-level residual allocation, an independent verifier, joint auxiliary losses, strict reload, and a parameter-matched surface-English target. The prior keyword/linear result remains proxy-negative only | build the governed source-disjoint stage corpus/codebook artifact, calibrate residual/verifier labels and VCM lifecycle rewards, pass overfit/intervention/ablation/resume/resource canaries, then refreeze; no KERC utility or failure claim before the matched campaign |
| Self-improvement flywheel | Tracks 0, 3 / Phases 7, 10 | the disabled campaign controller is now canonical through governance and the existing overnight entrypoint: it separates authorities, freezes holdouts/budgets, preserves rejected families, accounts debt, stops transactionally, and restores exact state; behavior remains at the zero-pass floor | preserve the disabled mechanics and enable generate->verify->admit->retrain only after an independently verified behavior-positive proposer receipt exists |
| VCM ABI + transactions/certificates | Phase 3 | wired: ABI, stable semantic objects, typed temporal relations, hybrid retrieval, lifecycle transactions, compaction, and fresh-process ontology migration | consume lifecycle records in Phase 7/10; keep dense embedding, parametric unlearning, and public-memory capability claims separate |
| Claim ledger + belief revision | Phase 14 | ledger implemented; assurance/evaluation-integrity consumption partial | compile one live assurance graph and cross-context integrity record into route decisions |
| Replacement transactions | Phase 2 | replayable-reference-backed for one bounded local route-authority effect | keep the independent effect audit and mutation controls green; require equivalent receipts for each new effect class |
| Procedural memory + toolification | Phase 15 | implemented for the three guarded exact assets; the exact Reflexive Router/trace-to-reflex contract is now part of pre-training architecture closure, while learned promotion still requires real traces | wire qualification, dependency invalidation, differential/shadow/canary, quarantine, decompilation, and rollback interfaces before training; defer only empirical promotion and route-regret claims |
| Authority kernel / SCIF | Phase 18 | replayable-reference-backed for the same bounded local effect; SCIF and wider authority controls remain synthetic | preserve exact effect/rollback auditing and expand support only when another real effect class earns it |
| Assistant product | Phase 5 | assisted runtime wired; model-only general chat unavailable | dogfood deterministic/verified assistance now, but earn model usefulness only from real multi-day use after the local model clears its behavior floor |
| Report/repository hygiene | Phases 0, 8, 14 | budget gate reports GREEN, but live reports are about 2.3 GB, runtime 18 GB, archive 14 GB, and the superseded strict-generator/code-lm-closure family still occupies 17 scripts and about 50,865 Rust LOC | run the already-specified retirement proof in parallel with KERC architecture work: extract live dependencies, make retained controls non-routeable, archive superseded reports, and remove reference-only source from active paths; compact this roadmap below its 1,100-line target without dropping matrix obligations |
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

This sequencing is machine-enforced in the canonical MoECOT trainer. Finite-update
architecture canaries are allowed for at most eight optimizer steps so contracts can
be tested without circularity. Unbounded or longer runs fail closed unless
`roadmap_implementation_gate.py --gate --require-pre-training-ready` is GREEN. This
is an optimizer-spend boundary, not a ban on implementing, integrating, replaying,
or negatively testing architecture before training.

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
- **A negative claim must be harder to earn than a green smoke.** Each adoption or
  retirement card records the exact implementation fidelity matrix, sanity tests,
  matched-opportunity design, evaluator independence, power/sensitivity analysis,
  confidence intervals, weak-tail effects, replication state, and maximum claim scope.
  A toy, undertrained, or construct-invalid test is useful diagnostic evidence but
  cannot close the idea it approximates.

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
2. Correct and complete KERC before optimizer spend. Preserve the exact packet and VCM
   residual substrate, but replace the inadequate hand-coded/linear proxy with the
   paper's actual constrained neural Surface-to-Kernel compiler, Kernel reasoner,
   learned hierarchical residual allocator, structured answer-packet head, copy-aware
   learned renderer, and independently implemented recompiler/verifier. Build licensed,
   source-disjoint English supervision for every stage; pass learnability, gradient,
   overfit, intervention, ablation, migration, security, and MLX canaries; and freeze a
   matched conventional-English versus KERC campaign. Do not manufacture easy templates
   or let deterministic rendering earn learned credit.
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
versus the control's `8,195`; independent FF-width matching leaves a 2,860-parameter
delta (`19,531,230` versus `19,534,090`). KERC has trusted source-only stage
selection, stage adapters, separate Kernel/surface output paths, a learned four-level
residual allocator, an independently parameterized raw-token verifier, copy access,
content-bound residual labels, one targeted semantic/object/value/fidelity corruption
per positive view with zero generator loss, joint residual/verifier losses, cached
decode, and strict checkpoint reload. The stage checks both positive and corruption
representability before publishing. V_K/V_P are fit only on positive private-train
compiler/core targets; development/evaluation rows and verifier corruptions contribute
zero vocabulary-fit signal. Compiler/core logits are restricted to V_K/V_P plus EOS,
while direct/renderer logits are restricted to V_S plus EOS. Vocabulary identity and
size are checkpoint/resume invariants. The live
plan has ten explicit targets and is RED only on `kernel_english_manifest_missing`.
This is architecture mechanics, not a learned KERC system or capability result: the
governed stage corpus/codebook artifact, distribution-calibrated labels, VCM interaction
reward, executable recompile verifier, full resume/migration/resource proofs, overfit,
intervention, ablation, and matched multi-seed campaign remain open.

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
- Train one coordinated KERC architecture, not disconnected prototypes: protected-span
  detection and calibrated lexical abstention; surface-to-Kernel parallel, paraphrase-
  convergence, hard-negative, cycle, entailment/question-preservation, structured-
  validity, and uncertainty objectives; Kernel language modeling and answer-packet
  generation; residual allocation and interaction promotion; copy-aware rendering;
  and symbolic plus neural round-trip verification. Data remains licensed, English-
  scoped, provenance-bound, source-disjoint, and contamination-audited. Public
  benchmark payloads remain calibration-only, and live teacher use remains governed
  OpenAI-only residual pressure. Version migration is deterministic or the row stays
  explicitly versioned; no compiled row silently changes meaning across revisions.
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

Pre-training acceptance: the canonical packet/compiler/core/renderer/verifier and
VCM residual lifecycle exist behind registered versioned interfaces; corpus transforms,
objectives, checkpoints, migrations, cost accounting, adversarial tests, matched
controls, and the joint campaign are content-bound and frozen; finite-update,
reload/resume, migration, cleanup, memory, resource, replay, integrity, and no-cheat
canaries are GREEN. This authorizes training, not a capability claim.

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
  coverage; reconcile it first when this file and the matrix disagree. Bind the
  crosswalk to an immutable committed AI_book revision. Report later live-worktree
  edits as intake drift, but do not let an external dirty checkout invalidate model
  training readiness. Advance the pin only through a reviewed reconciliation and
  route new concepts to existing phases (see Book-Parity Backlog).
- Check exact chapter identity/order and every book-owned title, part, file, claim
  label, evidence level, minimal implementation, mature endpoint, interface,
  invariant, failure mode, and Codex-test inventory. Count equality alone is not
  synchronization. Preserve the book commit and manifest digest used for review.
- Acceptance: `roadmap_implementation_gate.py --gate` reports full pinned chapter
  coverage and separately identifies live drift; every represented book mechanism meets the
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
| Open-ended improvement campaigns | pre-training mechanics are canonical: distinct generator/evaluator/promoter/stop/rollback authorities, single-axis matched-budget challengers, frozen holdout, novelty/coverage/weak-tail decisions, negative-knowledge retention, six-field debt ceilings, explicit shutdown handoff, hash-chained journal, full-state rollback, and fourteen rejecting mutations. Runtime activation and learned improvement remain unproven and disabled | 7, 10, 12, 14, 15, 18 |
| Intent-to-execution contracts | runtime path does not yet require contracts for every meaningful task | 1 |
| PlanForge DAGs + adequacy + arbitrage | not the default execution spine; adequacy contracts + arbitrage ledger missing | 1 |
| Cognitive compilation / semantic IR | failures reported more than repaired through IR-level localized feedback | 13 |
| Kernel English + Hierarchical Residual Ledger | reopened after adequacy audit: exact packet/residual interfaces exist, but the old campaign tested a keyword/linear proxy rather than the learned KERC pipeline. Implement the constrained compiler, Kernel reasoner, learned residual allocator, renderer, verifier, MLX state path, governed supervision, and decision-grade matched campaign before refreezing | 0, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 18, 19 |
| Data engines + continual learning + unlearning | canonical admission now carries 115,407 candidate receipts, five frozen policy simulations, contamination controls, 125,702 lineage edges, a 13-kind full-state inventory, descendant deletion, and bounded storage-erasure receipts. Real forgetting, influence, privacy, and unlearning efficacy remain post-training measurements | 3, 7, 12, 14 |
| Full-state update causality | pre-training mechanics are unified: model/optimizer/scheduler/RNG/cache/index/checkpoint/backup/effect state is content-addressed, best and final authority are distinct, packages replay exactly, update exposure is lineage-bound and bounded, deletion reaches descendants, and rollback restores the exact pre-state digest. Behavioral unlearning and undeclared external erasure are explicit non-claims | 0, 2, 7, 10, 14 |
| Durable semantic memory inside VCM | wired: stable objects, typed/temporal relations, additive ontology migration, transactional merge/supersession/retraction/compaction, sparse-vector/graph retrieval, bounded snapshots, and fresh-process replay; dense embedding and parametric unlearning remain explicit non-claims | 3, 14, 15 |
| Question-Compiled Semantic Addressing | bounded pre-training disposition complete: VCM now owns stable SOID/address/route indirection, three authoritative facets, Semantic Address Certificates, authority-safe translation, and exact migration/rollback; the matched source campaign retired the full objective and adaptive-question policy from the first long run because they added cost without task advantage | 0, 1, 2, 3, 6, 8, 13, 14, 16, 18, 19 |
| Verification bandwidth | verification not yet budgeted/routed as a scarce resource | 8 |
| Claim ledgers + belief revision | claim/transition records + contradiction links not first-class per run | 14 |
| Proof-carrying + tribunal/adversarial review | broader independent-review records not standard for architecture changes | 14, 18 |
| Labor OS typed jobs / artifact graphs | typed job + unified artifact graph not the universal unit | 1, 14 |
| Procedural memory | adoption discipline applied to only one trace so far | 15 |
| Routing heads + MoECOT + specialist cores | v8 shared trunk and five specialist deltas are trained and arm-evaluated; matched dense verdict, confirmation, live arm-card binding, typed composition, and route-versus-answer qualification remain | 10, 16 |
| Reflexive Router / compiled reflexes | pre-training mechanics are wired through the canonical assistant: typed command/reflex registries, finite outcomes, precedence and qualification, Chronicle/cache invalidation, structured concurrency, sole-kernel reversible effects, compilation/decompilation interfaces, and a frozen 32-case/eight-track/ten-policy mechanics profile. Learned calibration, real route regret, diverse trace promotion, drift response, and answer-quality benefit remain post-training evidence | 0, 1, 2, 3, 5, 6, 8, 14, 15, 16, 18, 19 |
| Ambiguous routing + adaptive deliberation | naturally ambiguous route corpus, fallback/abstention calibration, first-hit/last-correct, overthinking harm, branch credit, and verifier-disagreement accounting are missing on useful model outputs | 4, 6, 8, 10, 14, 16, 18 |
| Readiness gates + residual escrow | stale outputs/duplicate families reduce trust | 0, 2 |
| Generate-verify-repair | canonical repair now consumes a six-state GVR kernel with immutable candidate identity, independent verifier binding, localized repair authority, tamper-evident transitions, rollback, and zero learned credit for assisted/fallback output. Current repaired behavior remains 0/2, so efficacy is still unproven | 6, 10, 13 |
| Deterministic math/search substrates | not yet attached universally to planning/VCM/claim-ledger/assistant | 6 |
| Benchmark ratchets | public-transfer story stale; remeasure only after private correctness improves | 12 |
| Policy optimization (DPO/GRPO/RLVR) | shared update authority covers seven policy targets, and DPO/IPO/ORPO/KTO/SimPO plus GRPO/RLOO/ReMax/RLVR now execute through one disabled, provenance-bound, checkpointed, MLX-parity-tested objective contract with verifier-capacity and reward-hacking controls. Learned efficacy and objective selection remain post-training evidence | 7, 10 |
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

## Book-Futures Intake (classified, phase-routed)

The ASI Stack now has 54 active chapters. Ten items from the earlier completeness
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
