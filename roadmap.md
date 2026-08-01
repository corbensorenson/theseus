# Project Theseus Roadmap

Last corrected: 2026-07-31 UTC, to restore ASI Stack hypothesis proof as the
primary goal and remove the user-dependent assistant loop.

This is the forward human execution map. Current facts belong in
`docs/PROJECT_STATE.md`; operating rules belong in `AGENTS.md`; detailed
machine obligations belong in `configs/roadmap_implementation_matrix.json`;
canonical implementations and route authority belong in
`configs/project_manifest_registry.json`.

## North Star

The primary goal of Theseus is to rigorously test and earn evidence for the
largest, highest-leverage ideas in *The ASI Stack*. Theseus is the experimental
implementation: it must turn book mechanisms into real causal interventions,
compare them with strong matched controls, and preserve positive, negative, and
inconclusive evidence at exactly the scope earned.

The resulting system must remain private and locally served, use zero external
inference at serving time, and drive accepted live-teacher share toward zero.
Autonomous machine-verifiable work is the measurement substrate; it is not a
personal-assistant product goal and is not the reason the project exists.

The immediate objective is narrower:

> Qualify a fixed, genuinely runnable local model as an adequate measurement
> instrument; use it on autonomously acquired, license-compatible,
> machine-verifiable work; then test one major ASI Stack mechanism at a time
> through matched direct, integrated, intervention, and ablation evidence.

No forward phase may depend on Corben providing tasks, preference labels,
acceptance decisions, routine approvals, or a convenient time window. Bounded
authority, resource ceilings, task selection, stop conditions, and promotion
rules must be encoded in machine-readable policy. Work outside that policy
fails closed with a recorded wall; it does not become an interactive gate.

Progress is a stronger claim-bearing experiment, a measured causal result, a
more adequate implementation, or an honest terminal finding—not another
implemented subsystem or report. A useful completed task is necessary
instrument evidence, not the north star by itself. Positive, negative,
inconclusive, blocked, and invalid results count only at the exact scope their
evidence earned.

### ASI Stack proof contract

Each selected book idea must move through the same evidence ladder:

1. bind an exact book claim, causal mechanism, maximum inference, and failure
   semantics;
2. prove implementation and instrument adequacy before interpreting outcomes;
3. run matched L0 development comparisons with independent integrity audits;
4. qualify a survivor once on a fresh, source-disjoint D1 surface; and
5. update book support only after claim-level review of the full evidence.

No mechanics canary, useful demonstration, green report, or failed proxy can
promote or retire a major idea. The proof program tests one decision-relevant
mechanism at a time and preserves the strong baseline, matched budget,
weak-tail, cost, and rollback requirements in the operating charter.

### Bound high-leverage claim portfolio

The exact claim identities are pinned to AI_book commit `17c6ece…` and its
manifest digest in the implementation matrix. They are deliberately split by
what kind of experiment can decide them:

| Role | Exact chapter-core claims | Decision boundary |
| --- | --- | --- |
| Integrity prerequisites | `integrated-reference-architecture.core`; `evidence-states-and-claim-discipline.core` | Must pass before evidence is interpreted; they establish trace/claim mechanics, not usefulness |
| P4 causal candidates | `virtual-context-abi.core`; `planning-as-a-control-layer.core`; `cognitive-compilation-and-semantic-ir.core`; `verification-bandwidth-and-context-adequacy.core`; `routing-heads-and-specialist-cores.core`; `procedural-memory-and-cognitive-loop-closure.core`; `system-boundaries-and-authority.core`; `capability-replacement-and-rollback.core` | P3 residuals activate exactly one faithful intervention; order only breaks ties |
| Independent D2 claim | `replaceable-cognitive-substrates-beyond-transformer-monoculture.core` | Modular language arms versus dense-active and dense-total under the sealed neural contract |
| Synthesis claims | `the-efficient-asi-hypothesis.core`; `asi-is-a-stack-not-a-model.core` | Require multiple independently causal survivors and fresh matched composition evidence; no single P4 ablation can decide them |

`project-theseus-as-report-first-implementation-reference.core` remains an
evidence-handoff obligation rather than a headline scientific claim. A clean
report packet is necessary for book review but is not evidence that an ASI
Stack mechanism works.

## Direction Recovery

The previous local-model plan had the right controlled-variable idea but joined
the wrong implementations. The detailed review found five concrete faults:

1. `scripts/theseus_assistant_runtime.py` is registered as the canonical local
   assistant, but its generation backend is `scripts/checkpoint_chat.py`, a
   deterministic report/status responder rather than the frozen TMax model.
2. The TMax repository experiment bypasses that assistant runtime and invokes
   MLX through `scripts/core_evidence_worker_v2.py`.
3. The experiment's `full_stack` adapter compiles and records planning, VCM,
   routing, governance, and reuse state, then appends a context object to the
   same worker prompt. It does not execute the compiled plan or the registered
   subsystem routes. It can test a context wrapper, not the integrated stack.
4. L0-003-R1 required roughly 28-30 minutes per arm and more than 185,000
   prompt tokens per arm. Repeatedly tuning that 18-turn protocol is not a
   practical autonomous work loop.
5. The roadmap made neural work wait on this harness even though the matched
   neural-seed verdict is an independent top-priority experiment.

The correction is therefore architectural, not cosmetic: join the frozen local
model to the canonical runtime, prove that the registered subsystems cause live
execution differences, then use autonomously sourced A/B pairs. The neural-seed
campaign proceeds in parallel under an automated resource and evidence policy,
not a user-presence gate.

### Evidence surfaces remain separate

1. **L0 — autonomous development:** reusable machine-selected work pairs. L0
   may choose a reversible local default but cannot qualify a claim or student.
2. **D1 — stack efficacy:** a later prospectively frozen, source-disjoint
   denominator. D1 can support only scoped subsystem claims.
3. **D2 — student competence:** the modular student and matched dense controls.
   D2 remains sealed and independent of L0/D1.

No L0 prompt, outcome, repair, or route choice enters D1 or D2. D1 does not
authorize a student claim, and D2 does not inherit stack efficacy.

## Active Workstreams

### P — ASI Stack subsystem causal-proof track

#### P0 — Stop the invalid critical path

State: `COMPLETE_DIRECTION_DECISION`.

- Do not resume L0-003-R2. Its interrupted trace contains five read actions,
  no mutation, no candidate, no verification, and no arm comparison.
- Retain the two-turn verify/finish controller repair as ordinary regression
  maintenance; it is not the active scientific hypothesis.
- Scope L0-003-R1 as one positive context-wrapper observation: the wrapped arm
  produced a patch that passed independent acceptance while direct did not,
  but it never verified or finished and the wrapper was not the live Theseus
  runtime. No subsystem or default won.
- Keep all prior negative and interrupted evidence. Do not rerun the same
  workspace-hygiene task for credit.

#### P1 — Join TMax to the canonical Theseus runtime

State: `COMPLETE_LIVE_ROUTE_INTEGRITY`.

Implement one local inference backend contract owned by the existing
`theseus_assistant_runtime` surface. The first backend is frozen to:

- model: `mlx-community/Tmax-9B-MLX-8bit`;
- revision: `33812d6cf04f88856f25eb828de4f3144a194560`;
- runtime: `runtime/venvs/mlx-0.32.0-py312/bin/python`;
- generation policy: the exact prospectively bound configuration used by both
  arms in a pair; and
- serving boundary: local only, zero external inference, zero teacher calls.

The canonical execution path must be real:

```text
request
  -> intent and authority
  -> VCM selection
  -> executable plan
  -> route and allowed tools
  -> frozen local model
  -> independent verification
  -> candidate effect / hold / rollback
  -> outcome and cost ledger
```

The direct baseline uses the same local backend, decoder, snapshot, effect
sandbox, and independent evaluator, while bypassing optional Theseus context,
planning, routing, and reuse. Safety containment is common infrastructure, not
an advantage removed from the baseline.

P1 exits only when a route-integrity test proves all of the following:

- every enabled subsystem emits a receipt consumed by a downstream live step;
- the compiled plan is executed or explicitly held, not merely rendered;
- selected VCM content reaches the model and stale/denied content does not;
- routing changes a callable capability, budget, tool, or hold decision;
- verification controls release and repair rather than supplying a label; and
- deleting the `full` label or context decoration cannot preserve a false
  full-stack pass.

Until that gate exists, use `context_wrapped_worker`, not `full_theseus`, in
evidence language.

P1 now has a source-bound live route receipt for the exact frozen TMax backend.
Direct and integrated requests use the same model, decoder, snapshot, sandbox,
structural verifier, and one-call budget; the integrated receipt proves actual
VCM content and executed route state reached the model. This establishes route
mechanics only, not usefulness or subsystem efficacy.

#### P2 — Establish a practical paired canary

State: `COMPLETE_TERMINAL_INSTRUMENT_UNSUITABLE`.

Run one new, useful, low-risk repository request in independent disposable
snapshots. Freeze the natural request, visible acceptance criteria, parent
source, model, decoding, information boundary, tool set, verifier, effect
scope, and arm order before execution. The evaluator is route-blind until both
candidates seal.

The canary may use at most six model calls per arm, 12 wall-clock minutes per
arm, and 25 minutes for the pair including verification. These are product
budgets, not scientific adequacy claims. If the frozen TMax backend cannot
produce an independently evaluable candidate after one bounded loop-efficiency
repair, record it as unsuitable for this coding instrument. Do not repeatedly
expand the budget or tune on the same task. A different model starts a new
instrument version and cannot be mixed into the same denominator.

P2-001 froze a new maintenance request discovered during live P1 operation:
make the route-integrity CLI independently re-audit its canonical evidence
bundle. The first matched pair and the one prospectively frozen
loop-efficiency repair both stayed inside cumulative product budgets, preserved
route blinding, and used four total local-model calls. Both repaired candidates
sealed safely but reached the exact 512-token cap, ended mid-patch, and failed
independent patch application before hidden correctness tests. The exact
TMax/512-token instrument is therefore terminally unsuitable for this coding
lane. This is `INCONCLUSIVE_IMPLEMENTATION` for Theseus subsystems and
does not falsify TMax generally, the integrated architecture, or any ASI Stack
mechanism. P2-001 may not be replayed for fresh credit.

#### P2A — Establish an adequate autonomous coding instrument

State: `TERMINAL_INCONCLUSIVE_INSTRUMENT_AND_TASK_NAMESPACE`.

P2-001 ended an exact instrument version; it did not end the fixed-model
strategy. P2A is now implemented as a prospectively named successor instrument
that changes the candidate protocol before changing the model:

- keep the exact frozen TMax weights and revision for the first successor;
- keep a persistent local backend loaded across the matched pair;
- give both arms the same bounded repository search, file-read, typed-edit,
  patch-application, and visible verification capabilities;
- represent edits as concise typed operations rather than a JSON-wrapped long
  diff;
- allow 1,024-1,536 generated tokens, at most two model calls per arm, and one
  verifier-fed repair;
- keep the source snapshot, task, decoder, tools, effect sandbox, visible
  verifier, hidden evaluator, order policy, and total budgets identical; and
- vary only the named Theseus mechanism. Safety containment and necessary
  action tools are common infrastructure, not advantages removed from direct.

The first task comes from the autonomous task-source contract below. It must be
fresh relative to all consumed Theseus surfaces and must not be selected or
tuned after either arm opens. A positive-control pass establishes only that the
instrument can produce an applicable candidate; it does not promote a
subsystem claim.

P2A exits `INSTRUMENT_ADEQUATE` only when at least one arm produces an
applicable candidate that reaches independent correctness evaluation inside the
frozen call, token, latency, and effect budgets. If exact frozen TMax fails this
successor, mark it unsuitable for the coding denominator and autonomously
qualify one stronger locally installed model under a new instrument identity.
Do not enlarge the same denominator repeatedly.

The exact source-bound run consumed the PSF-licensed
`python/typing_extensions` PR 677 task once. Pair integrity stayed GREEN: one
persistent TMax load served four calls, both routes used the same frozen model
and budgets, every call had a valid route-integrity receipt, external inference
and user-facing effects remained zero, and the route-blind evaluator opened no
route label. Neither arm produced a parseable candidate, so zero candidates
reached correctness evaluation. Both first calls ignored the edit envelope;
the direct repair used `src/typing_extensions.py`, which matched the natural
request but was rejected because the archive exposed only
`typing_extensions_677_parent/src/typing_extensions.py`; the integrated repair
again ignored the envelope. The integrated runtime also reported its unrelated
product-trace gate RED while its route-integrity receipt stayed GREEN.

The terminal disposition is `INCONCLUSIVE_IMPLEMENTATION`, not a direct versus
integrated result and not negative evidence about any ASI Stack mechanism. The
task is consumed and may not be rerun, trained on, or reused for D1/D2. P2B must
freeze a new instrument identity, use one canonical repository-relative path
namespace, separate runtime-product health from experiment route integrity,
prospectively select a stronger installed local candidate, and consume one
fresh licensed source-disjoint development task.

#### P2B — Qualify the repaired instrument with the strongest retained local candidate

State: `TERMINAL_INCONCLUSIVE_LITERAL_GRAMMAR_TRANSPORT`.

The retained local-model bakeoff qualified no model, so P2B does not pretend
otherwise. It prospectively selects Qwen3.5-9B only as the strongest diagnostic
candidate: it was runtime- and exact-action-preflight GREEN and produced the
only useful retained result (one of one completed task), while two other tasks
timed out; Qwen2.5-Coder-7B and Qwen3-8B completed three tasks each but produced
zero useful outcomes. The old worker's adequacy floor remains failed for every
candidate.

P2B freezes the exact Qwen3.5 revision and snapshot, one persistent load, the
same 1,536-token/two-call-per-arm caps, and the same direct/integrated route
contract. Its new archive-root adapter presents one repository-relative path
namespace consistently in the natural request, reads/searches, edit grammar,
patch application, visible verifier, and hidden evaluator. Runtime-product
health and route-integrity validity are both retained and may not substitute
for one another. The model-selection report, runtime overlay, candidate runner,
assistant runtime, evaluator, and their hashes are frozen before task
acquisition. P2B remains instrument infrastructure: it cannot support a
Theseus mechanism or book claim.

The first P2B task is now sealed from Apache-2.0-licensed `psf/requests`
PR 7502 after the instrument-freeze commit. The exact upstream parent and
target archives are retained; deterministic regular-file-only derivatives
omit two unrelated certificate-directory links that the sandbox rejects. A
network-free evaluator confirms the parent fails on a dynamically proxied
`read` method and the exact target passes. The model sees only the natural
request and repository-relative `src/requests/models.py` context; the later
patch, hidden test, PR identity, and evaluator remain sealed. Candidate model
calls were zero at seal time; this exact task was eligible for one run only.

The one allowed run is now complete. One persistent Qwen3.5 load served four
matched calls, every route-integrity receipt was GREEN, and external inference
and user-facing effects remained zero. However, the frozen JSON grammar
contained literal `\\n` characters while the parser required real newline
characters. All four outputs reproduced literal `\\n`; three contained the
`REPLACE` token and all four used the authorized repository-relative path.
Thus zero candidates parsed and zero reached correctness evaluation. The model
followed the displayed transport closely enough that this is decisively
`INCONCLUSIVE_IMPLEMENTATION`, not evidence of model incompetence or a Theseus
effect. The Requests task is consumed and may not be rerun.

P2C must change only the grammar transport: actual newline characters in the
rendered prompt plus an audit proving that the exact displayed example
round-trips through the exact parser. Qwen3.5, decoder, persistence, arm order,
budgets, runtime overlay, path namespace, evaluator boundary, and effect
authority stay fixed. P2C receives a fresh licensed source-disjoint task under
a new pre-task instrument identity.

#### P2C — Repair and qualify the exact grammar transport

State: `TERMINAL_INSTRUMENT_ADEQUATE_ZERO_USEFUL`.

P2C is now implemented with the exact P2B Qwen3.5 model, snapshot, decoder,
persistent load, arm order, call/token budgets, assistant runtime overlay,
repository-relative namespace, patch application, visible verifier, blind
evaluator contract, runtime-health interpretation, and zero-effect boundary.
The sole experimental change is that the JSON grammar decodes to actual newline
characters. Its independent instrument audit renders the exact configured
grammar, substitutes one concrete authorized action, and proves the result
parses through the exact candidate parser with one action and zero faults.
This audit was GREEN before task acquisition. P2C then autonomously selected
the BSD-3-Clause `pallets/click` PR 3578 parent at
`8929d392781c8113bc569f388c15c47b94f86581` and upstream target at
`762c97eef7c1b3779678992f26a553a2a8c80793`. Deterministically normalized
archives contain only regular files and directories. The network-free blind
oracle fails the exact parent on double-bracketed optional Choice/DateTime
metavars, passes the exact target, and checks required, deprecated, and
repeated-argument behavior. Its evaluator audit is GREEN. The source snapshot,
task, hidden evaluator, and target are sealed before candidate generation;
candidate calls were zero at seal time and the task was eligible for exactly one
P2C run.

That run is complete. One persistent Qwen3.5 load served three calls across the
matched pair, all route-integrity receipts were GREEN, and the actual-newline
grammar produced one safe, authorized, parseable integrated candidate. The
direct arm remained malformed after its one repair. The integrated edit reached
the independent hidden evaluator but preserved the double-bracketing defect, so
the task produced zero useful candidates. Rollback passed and no unsafe,
external-inference, training, or user-facing effect occurred. P2C therefore
qualifies this exact harness as an adequate development instrument while the
exact task remains unsolved. One parseable integrated edit on one task is not a
direct-versus-integrated effect, a model-competence claim, a subsystem result, or
a book-support change. The Click task is consumed and may not be rerun.

##### Autonomous task-source contract

Task acquisition is an offline-serving data-ingestion process, not external
inference. It may read from allowlisted online sources and must cache every
admitted artifact locally before an experiment begins.

- Prefer permissively licensed active repositories in the in-scope languages.
  Reconstruct maintenance work from a parent revision plus natural issue or
  change request; retain the later patch and tests as evaluator-only material.
- Prefer tasks published after the frozen model artifact or otherwise passing
  a declared memorization-risk screen. Record repository, revision, timestamps,
  license, acquisition digest, and contamination result.
- Candidate-visible data is limited to the natural request, callable surface,
  parent snapshot, and explicitly allowed runtime context. Later patches,
  hidden tests, commit identity, source task ID, answer labels, and metadata
  derived from the target remain sealed.
- Use deterministic eligibility filtering and sampling from a sealed pool.
  The model, wrapper, evaluator, and implementation author may not choose a
  favorable task after outcomes are visible.
- Keep development, D1 qualification, D2, public calibration, and training
  pools disjoint. A consumed evaluation task never becomes training data or a
  fresh claim-bearing surface.
- Public benchmark prompts, tests, solutions, traces, and answer templates
  remain calibration-only and are excluded from this task pool.
- No task may mutate its upstream repository, open a pull request, contact a
  maintainer, or produce an external effect. All execution occurs in disposable
  local snapshots with exact rollback.

This contract replaces personal-assistant data collection. The system obtains
its own work and its own independent correctness signal; Corben is not a task,
label, approval, or scheduling dependency.

#### P3 — Autonomous paired development

State: `INSTRUMENT_AND_TEN_TASK_POOL_SEALED_LOCAL_RUN_NOT_OPENED_LUNA_TRANSPORT_UNBOUND`.

The local P3 instrument is now frozen before task-pool acquisition. It retains
the exact P2C Qwen3.5 revision, snapshot, decoder, runtime overlay, 1,536-token
and two-call-per-arm caps, actual-newline typed-edit protocol, repository-root
path namespace, disposable effects, blind evaluator, and route-integrity/product
health separation. Qwen3.5 is the best retained development denominator because
it is the only installed candidate with any prior useful retained outcome and it
exercised the complete parse/apply/seal/evaluate path in P2C; it is not promoted
as capability-qualified. The ten-task count and source-disjointness rules are
fixed. Odd campaign indices run direct first and even indices run integrated
first, eliminating a systematic arm-order assignment while keeping each task's
source, request, context, model, decoder, budgets, verifier, and evaluator
matched. The instrument audit is GREEN. The autonomous source selection is now
sealed as ten distinct recent maintenance changes from ten MIT or BSD-3-Clause
repositories. Every normalized parent archive, exact revision, license, natural
request, candidate-visible context, allowed path, target archive, hidden oracle,
and evaluator digest is bound. All ten parents fail and all ten exact upstream
targets pass their network-free evaluators under Python 3.12. Candidate-visible
manifests contain no later patch, hidden test, source-task identifier, or answer
label. No P3 candidate call has opened. The Luna-xhigh reference remains
separately defined but transport-unbound; the local campaign may proceed without
inventing hosted results, and Luna must use this identical sealed pool if a
governed callable transport becomes available. Repeated tasks are regression
checks only.

For each pair:

- run direct and integrated Theseus from the same source snapshot with
  randomized order where practical;
- keep candidates inside disposable snapshots until independent checks and the
  preregistered machine decision rule complete;
- record verifier-accepted, missed, corrected, completed, failed, abstained,
  unsafe, false-blocked, and rollback outcomes;
- measure wall time, first useful artifact time, model calls, prompt/generated
  tokens, tool and verifier work, autonomous repair work, and total cost; and
- preserve every failure and residual without creating a new report family.

The decision order is: reject unauthorized effects or failed rollback; maximize
useful completion; minimize total cost among quality-eligible candidates; then
prefer the simpler route when practically tied. A local default stays shadowed
and reversible until at least five distinct real tasks show no safety,
rollback, or weakest-task-type regression. L0 remains development evidence.

##### P3 hosted reference control

Run the same sealed task pool as a separately labeled 2×2 reference after task
selection is immutable:

| Model | Direct | Theseus-integrated |
| --- | --- | --- |
| Best locally qualified model | Primary local baseline | Primary Theseus causal contrast |
| `gpt-5.6-luna` at `xhigh` | Hosted reference baseline | Hosted Theseus robustness contrast |

The Luna arms use the same candidate-visible source snapshot, natural request,
typed-edit protocol, two-call cap, visible verifier, hidden evaluator, effect
sandbox, and route-blind scoring. Luna outputs are measurement-only: never
served, never admitted as training rows, never used to choose tasks, and never
mixed into the local denominator. The within-local and within-Luna Theseus
deltas are interpretable separately. The cross-model ranking and interaction
remain exploratory because the hosted Codex wrapper, tokenizer, reasoning
tokens, latency, cost, and runtime differ. Bind a callable transport and exact
model/effort receipt before running this row; current official guidance names
Luna for clear, repeatable, high-volume or focused coding work and permits
`xhigh` when supported.

#### P4 — Test one ASI Stack mechanism at a time

State: `BLOCKED_ON_NONZERO_AUTONOMOUS_P3_USEFULNESS`.

Use the observed instrument and task residuals to select exactly one
intervention. Do not run a factorial or make record production the dependent
variable. P4 is the causal development stage for the book's major subsystem
claims; P2A and P3 exist to make these comparisons interpretable.

P3 residuals select the first matching eligible mechanism; the order below is
only a deterministic tie-break, not permission to run every row.

| Order | Exact claim | Causal variable | Strongest required contrasts |
| --- | --- | --- | --- |
| 0 | Integrated architecture + evidence-state prerequisites | Downstream-consumed trace joins and independent claim transitions | Direct, record-only, context-decoration-only, producer self-score |
| 1 | `virtual-context-abi.core` | Source-bound admitted model-visible context | No context, information-matched plain context, maximal ungoverned context, stale/shuffled/tainted/revoked interventions |
| 2 | `planning-as-a-control-layer.core` | Executed typed dependencies and feedback-driven replanning | Information-matched direct, non-executing plan text, strong static workflow |
| 3 | `cognitive-compilation-and-semantic-ir.core` | Typed lowering, stable identities, target validation, and dependency-local repair | Natural-language plan, direct target generation, deterministic compiler-only |
| 4 | `verification-bandwidth-and-context-adequacy.core` | Risk/adequacy-routed real verification work | Fixed minimal, fixed maximal, random/cost-only allocation with held candidates |
| 5 | `routing-heads-and-specialist-cores.core` | Least-sufficient eligible route changing real capability/tool/budget access | Always-direct, always-maximal, random/cost-only, oracle route ceiling |
| 6 | `procedural-memory-and-cognitive-loop-closure.core` | Trace-derived verified parameterized procedure | Fresh slow path, retrieval-only, checklist reuse, hand-authored script |
| 7 | `system-boundaries-and-authority.core` | Live revocable authority tuple at dispatch/effect time | Coarse allowlist, ambient authority, conservative hold |
| 8 | `capability-replacement-and-rollback.core` | Prospective shadow/canary/monitor/commit/recovery transaction | Ad-hoc swap, pointer rollback, retain-prior control |

A failed proxy, underpowered local model, or incomplete runtime receives
`INCONCLUSIVE_IMPLEMENTATION` or `INCONCLUSIVE_EXPERIMENT`; it cannot retire the
book mechanism.

#### P5 — Fresh D1 qualification

State: `BLOCKED_ON_A_DECISION_RELEVANT_P4_SURVIVOR`.

Only after L0 selects a mechanism, deterministically freeze a fresh
source-disjoint cohort from the autonomous online task pool, with competent
positive controls, model/runtime identity, evaluator, budgets, minimum
worthwhile effect, uncertainty method, weak-tail rules, and terminal states.
Consume it once. D1 never uses the ten P3 development tasks.

### N — Matched neural-seed verdict

The neural campaign is parallel to P, not blocked by it. N0 source/custody
readiness can run while the training hold stays installed. N1 begins only when
an automated launch controller proves the frozen source, custody, resource,
rollback, and stop-policy predicates. N2 completes the modular, dense-active,
and dense-total runs transactionally; N3 consumes the sealed D2 surface once
through an automated one-shot authority; N4 composes D1 and D2 without
denominator or support leakage. No neural stage waits for user presence or an
interactive timing choice.

### M — Teacher, repository, and evidence maintenance

Maintenance serves P or N and does not become another research program:

- keep `teacher_share_of_accepted_training_rows` durable and drive it toward
  zero; no live teacher is part of runtime A/B testing;
- split the current 88-file, multi-domain dirty transaction before any source
  binding or release claim;
- inventory the roughly 41 GiB / 249,000-file runtime tree and 1.9 GiB report
  tree under existing retention authority before deletion;
- consolidate or retire superseded L0 configs, scripts, and generated reports
  only after their negative evidence and replay locators remain durable; and
- keep one roadmap, one project-state page, one registry, and existing ledgers.

## Current Program State

| Track | State | Meaning |
| --- | --- | --- |
| Documentation | `DIRECTION_RECOVERY_SOURCE_BOUND` | P1 and P2 are isolated source transactions; the roadmap/book reconciliation is an independently validated bounded transaction |
| Primary scientific objective | `ASI_STACK_CAUSAL_PROOF_ACTIVE` | P2A/P3 establish the instrument and residuals; P4 develops one book mechanism at a time; P5 provides fresh D1 qualification; N tests the modular neural claim on D2 |
| Frozen local model | `GREEN_STANDALONE_INSTRUMENT` | Exact TMax revision loads through pinned MLX; this proves runtime compatibility only |
| Canonical assistant backend | `FROZEN_TMAX_PLUS_STATUS_COMPATIBILITY` | Direct and integrated learned-generation modes use the exact offline TMax snapshot; the status shim remains a maintenance-compatible mode |
| Integrated TMax + Theseus runtime | `P1_GREEN_ROUTE_INTEGRITY` | Live VCM and route state are model-consumed and independently receipt-bound; no usefulness claim follows |
| Paired instrument A/B | `P2_TERMINAL_INSTRUMENT_UNSUITABLE` | P2-001 used the only allowed repair; both 512-token candidates failed patch application, so its exact denominator is closed |
| Autonomous instrument successor | `P2A_TERMINAL_INCONCLUSIVE_INSTRUMENT_AND_TASK_NAMESPACE` | The exact matched TMax run held its route/persistence budgets but produced zero parseable candidates; a repo-relative/archive-prefix mismatch also made the task packet ambiguous, so no subsystem comparison is valid |
| P2B repaired instrument | `TERMINAL_INCONCLUSIVE_LITERAL_GRAMMAR_TRANSPORT` | Qwen3.5 reproduced the prompt's literal backslash-n transport on all four matched calls, but the parser required actual newlines; zero candidates were evaluable, so the exact result is a harness implementation failure only |
| P2C grammar-transport instrument | `TERMINAL_INSTRUMENT_ADEQUATE_ZERO_USEFUL` | One persistent Qwen3.5 load served three matched calls; one safe integrated edit parsed and reached the blind evaluator but failed correctness, direct remained malformed, rollback passed, and the consumed task cannot support a subsystem effect |
| L0-003-R2 | `INTERRUPTED_DIRECTION_CANCELLED` | Five reads, no mutation, candidate, verification, or comparison; do not resume |
| Autonomous usefulness | `P3_INSTRUMENT_AND_TEN_TASK_POOL_SEALED_RUN_PENDING` | The exact best-retained Qwen3.5 denominator and counterbalanced ten-task contract are audit-GREEN; ten distinct licensed source tasks now have parent-fail/target-pass blind evaluators sealed before any candidate call |
| Hosted reference control | `DEFINED_TRANSPORT_NOT_BOUND` | `gpt-5.6-luna` at `xhigh` is prospectively scoped as a measurement-only 2×2 reference; no callable experiment adapter is yet source-bound |
| D1 stack efficacy | `TERMINAL_PRIOR_INCONCLUSIVE_NEW_D1_SEALED` | Prior worker was inadequate; a future fresh D1 waits for a faithful P4 survivor |
| Neural checkpoint custody | `CUSTODY_GREEN` | Exact step-11,416 model, AdamW, RNG, cursor, and prospective lineage |
| Long training | `TRAINING_HELD_PENDING_AUTONOMOUS_POLICY` | N0 may refresh readiness; N1/N2 require a machine-readable launch/resource/rollback policy, not a user decision |
| D2 student capability | `NOT_EVALUATED_SEALED` | The 57M candidate and dense controls have no verdict |
| Runtime exposure | `LOCAL_ONLY` | No LAN or public exposure qualification |

## Immediate Execution Order

1. Keep the isolated source-bound P1 implementation and terminal P2-001
   evidence immutable; do not mix neural, release, cleanup, or Rust work into
   their transactions.
2. Preserve the source-bound 84-chapter pin and 13-claim causal portfolio. Its
   exact identities, causal variables, adequacy requirements, strongest
   controls, decisions, and maximum inferences are now the experiment
   contract; the binding changes no book support state.
3. Preserve the terminal P2A run, blind evaluation, runtime receipts, and
   independent disposition as immutable negative evidence. Do not rerun its
   consumed `typing_extensions` task or translate its instrument failure into a
   subsystem result.
4. Preserve P2B's run, blind evaluation, runtime receipts, and terminal
   disposition. Do not rerun its consumed Requests task or infer model or
   subsystem failure from the literal-newline transport mismatch.
5. Preserve the terminal P2C run, blind evaluation, runtime receipts, and
   disposition. Do not rerun the consumed Click task or infer a subsystem effect
   from one parseable but incorrect integrated candidate.
6. Preserve the frozen P3 instrument and sealed ten-task pool. Run every task
   exactly once through its counterbalanced direct/integrated local pair, then
   score route-blind. Keep malformed, incorrect, useful, unsafe, rollback,
   latency, and weak-tail outcomes explicit; never replace a hard task.
7. Bind the measurement-only Luna-xhigh adapter when a callable governed
   transport exists. Run both
   models direct and integrated on the same ten tasks without mixing
   denominators or allowing hosted outputs into serving or training.
8. In parallel, refresh N0 with the training hold installed and implement the
   automated N1 launch/resource/rollback controller. Start no long run until
   that controller proves every frozen predicate.
9. Select no P4 subsystem ablation until P3 establishes nonzero useful work and
   the autonomous residual ledger identifies one decision-relevant defect.

## Historical Rapid Evidence Campaign

E0-E5 and the consumed worker-successor attempts are immutable evidence
history, not the active work queue. They define the information-flow,
competence, rollback, and negative-scope requirements inherited by L0 and any
future D1 qualification.

### E0 — Freeze Questions, Tasks, And Decision Rules

State: `COMPLETE` — frozen at preregistration digest
`97d28226f39c62a11c81143fc31d44c9637466244b54088815102f37e2aced72`.

Create one prospectively frozen campaign packet under the existing
`ASI-THESEUS-FLAGSHIP-01` owner. Do not create a new benchmark family or
dashboard. The active implementation obligation is the existing
`planned.governed_usefulness_effect_complete_rollback_v1` backlog item.

The packet must bind:

- exact claim IDs and maximum inference;
- worker, model, route, tool, memory, and verifier identities;
- one natural repository-replay cohort for D1;
- disjoint calibration, development, and held-out partitions;
- task-source, license, privacy, and contamination boundaries;
- allowed candidate-visible fields;
- hidden tests, gold effects, and evaluator-only fields;
- matched route budgets and retry rules;
- competence, usefulness, unsafe-release, false-block, weak-tail, cost, and
  latency measures;
- all failure, timeout, abstention, denial, and infrastructure denominators;
- independent evaluator and candidate-integrity recomputation;
- terminal decision rules and a fixed rescue ceiling.

Preferred tasks are real repository changes reconstructed from committed
history or existing low-risk work records: parent source state plus the natural
request is candidate-visible; later patches, tests, outcomes, commit identity,
answer-family labels, and derived target metadata are not.

If a task cannot be reconstructed without answer leakage or an authored
success path, exclude it before the held-out partition is opened.

Exit: an immutable preregistration or an exact wall showing that a competent
natural denominator cannot yet be assembled.

### E1 — Clean Live Theseus Replay

State: `COMPLETE_REPLAYABLE_REFERENCE_BACKED`.

Reproduce the current public-safe stack from a clean checkout and bind:

- source commit and environment;
- registry and roadmap gate results;
- one allowed-effect trace;
- one blocked or revoked trace;
- one exact rollback trace;
- candidate/evaluator information-flow audit;
- model, tool, VCM, plan, route, authority, observation, residual, and terminal
  receipt identities;
- every missing, stale, skipped, or private artifact.

This closes the “latest report versus reproducible reality” gap. It does not
establish usefulness or student capability.

Exit: `REPLAYABLE_REFERENCE_BACKED` for the exact packet, or an honest
`REPLAY_FAILED` disposition with the causal defect.

### E2 — D1 Natural Governed-Stack Comparison

State: `TERMINAL_INCONCLUSIVE_WORKER_INADEQUATE`.

Observed result: the frozen local worker completed 0/3 development tasks. The
preregistered stop rule preserved all four E2 heldout tasks unopened. No
efficacy comparison is justified.

Run the same frozen natural tasks through:

1. full governed stack;
2. direct local worker;
3. test-only route;
4. record-only route;
5. conservative-hold route.

The worker must be local. No external inference, teacher call, hidden answer,
current-repository escape, or user-facing effect is allowed.

Primary outcomes:

- useful completed task;
- unsafe or unauthorized release;
- false block and fair rescue;
- missed or malformed result;
- verified rollback or compensation;
- weakest task-family outcome;
- total lifecycle cost and latency.

Interpretation gate:

- if the frozen worker does not meet the preregistered competence floor, the
  governance contrast is `INCONCLUSIVE_WORKER_INADEQUATE`;
- the same result is still valid negative evidence about the current
  integrated product;
- the worker or floor may not be changed after held-out outcomes open;
- full governance wins only if it improves the joint useful-safe frontier
without unacceptable false blocking or hidden lifecycle cost.

### E3 — Efficiency, Planning, Memory, And Reuse

State: `TERMINAL_INCONCLUSIVE_WORKER_INADEQUATE`.

Observed result: nine existing-owner mechanics controls passed, all 42
candidate variants were sealed before their corresponding targets opened, and
no maximal, cheapest, or least-sufficient policy completed a useful task.

Use real repeated repository-work families to test three route policies:

- always-maximal eligible route;
- always-cheapest eligible route;
- least-cost route satisfying the frozen quality predicate.

Within the adaptive route, predeclare causal comparisons for:

- full planning versus direct execution;
- VCM packet versus no VCM;
- correct packet versus stale, shuffled, or omission-bearing packet;
- verified reusable procedure versus fresh execution;
- false-trigger, drift, quarantine, decompilation, rollback, and retirement.

Reuse the existing reflexive-router, VCM ABI, verification-bandwidth,
procedural-memory, claim-ledger, and ambiguous-routing owners. This is behavior
qualification for already-wired mechanisms, not permission to implement a new
router, memory system, planner, or evidence store.

Measure accepted useful work per total contract cost, not raw token count or
isolated latency. Cost includes planning, retrieval, verification, repair,
retries, human-equivalent intervention, storage, rollback, and residual burden.

If no route achieves the quality predicate, the experiment does not support
the efficient-ASI claim. It records the competence or route wall instead.

### E4 — Joined Evidence And Claim Disposition

State: `COMPLETE` — nine claim-scoped dispositions are recorded in
`reports/core_evidence_e4_disposition.json`.

Update the existing evidence owner with:

- complete method and preregistration;
- all arms and denominators;
- uncertainty and weak-tail results;
- one sanitized success trace when one exists;
- one failure trace;
- one blocked/revoked/rollback trace;
- resource and lifecycle-cost table;
- exact negative, null, skipped, and infrastructure outcomes;
- claim-by-claim disposition;
- maximum justified inference and explicit non-claims;
- public-safe replay command and artifact digests.

Each result receives exactly one terminal state:

- `POSITIVE_SCOPED`;
- `NEGATIVE_SCOPED`;
- `INCONCLUSIVE_WORKER_INADEQUATE`;
- `INCONCLUSIVE_EXPERIMENT`;
- `BLOCKED_INFRASTRUCTURE`;
- `INVALID_INFORMATION_FLOW`;
- `INVALID_EVALUATOR`.

No roadmap, report, chart, test count, or polished demonstration changes an ASI
Stack support state automatically. The book must review any proposed transition
against its own claim identity and evidence-quality rules.

### E5 — Showable Evidence Brief

State: `COMPLETE_PUBLIC_SAFE` — the shareable result is
`docs/CORE_EVIDENCE_BRIEF.md`.

Produce a public-safe brief from the existing flagship evidence packet:

- the question in plain English;
- a small architecture and experiment diagram;
- matched-arm result table;
- useful-safe frontier plot;
- lifecycle-cost breakdown;
- strongest success, failure, and rollback examples;
- what the evidence changes in Theseus;
- what it changes in the ASI Stack;
- what remains unproved.

The brief may be shared after review. It must remain interesting even when the
result is negative.

### Consumed local-worker successor dispositions

The detailed worker history is retained in the registered configs, reports,
event traces, and `docs/PROJECT_STATE.md`. Its forward constraints are:

- the original worker campaign is
  `TERMINAL_INCONCLUSIVE_WORKER_INADEQUATE`;
- Qwen3/Qwen3.5 successor attempts remain scoped negative or invalid evidence;
- exact TMax-9B/Worker-v3 historically passed 2/3 prospective repository tasks
  with zero unsafe effects and exact rollback, but that competence does not
  transfer to changed worker code;
- exact TMax-9B/Worker-v4 is
  `TERMINAL_FAIL_TMAX9B_WORKER_V4_EDIT_COMMITMENT` on its consumed development
  task after adequate inspection and planning;
- that task may not be tuned, replayed for credit, or relabeled as an
  infrastructure failure;
- the original E2 heldouts remain sealed; and
- L0 may use the runnable local model on new real development work only after
  P1 route integrity and the P2 canary pass, without reopening those
  denominators or claiming qualification.

A stronger model may still be compared when it is locally installed and
preflighted, but model shopping no longer blocks autonomous work or subsystem
development. The active estimand is the delta between matched direct and
Theseus-wrapped routes for the same fixed model.

## Neural Verdict Campaign

This is the only long-training program that remains on the roadmap.

### N0 — Source-Bound Readiness Refresh

State: `may run now; leave the hold installed`.

1. Recompute the architecture freeze and independent readiness package from
   committed source.
2. Verify the exact step-11,416 model, AdamW, RNG, cursor, plan migration, and
   prospective lineage.
3. Separate selected-Mac-route readiness from nonblocking cross-platform work.
4. Preserve the frozen corpus, architecture, objective, schedule, and
   evaluation identity.
5. Reach `TRAINING_READY_BUT_HELD`.

### N1 — Bounded Fresh-Process Qualification

State: `BLOCKED_ON_AUTOMATED_LAUNCH_POLICY_AND_N0_GREEN`.

Run one exact transactional resume segment with:

- a machine-readable resource, checkpoint, rollback, and terminal-stop budget;
- external resource watchdog;
- two measured checkpoint transactions of disk headroom;
- no arbitrary memory floor or time-of-day rule;
- checkpoint/reload and next-update replay;
- no evaluation consumption;
- no topology, optimizer, data, KERC, or ANE change.

### N2 — Complete The Matched Training

State: `BLOCKED_ON_N1_AND_AUTOMATED_RESOURCE_ENVELOPE`.

Order:

1. modular shared-trunk candidate;
2. dense active-parameter control;
3. dense total-parameter control.

All candidates receive matched raw data, compute, tuning opportunity,
inference/verifier budget, and total-system-cost accounting. Runs are
transactional and resumable; intermediate results do not alter the frozen
experiment.

### N3 — Consume D2 Once

State: `blocked on N2, clean source binding, independent custody, and automated
one-shot evaluation authority`.

The independent readiness audit rematerializes the current 160 cases, verifies
the content-addressed consumed v8 packet and registry history, and finds zero
exact or whitespace/case-normalized model-visible prompt overlap. This supports
only `VALID_FRESH_PRIVATE_SURFACE` at that stated scope. It does not establish
semantic-family independence, evaluate the 57M candidate, authorize D2, or
authorize training. Preserve the current and historical records, keep both
holds installed, and do not create another benchmark family.

Consume the frozen 160-case private functional surface exactly once. Report:

- model-only and separately assisted outcomes;
- weakest-arm utility;
- verifier, malformed, empty, injection, and forbidden-field outcomes;
- modular versus both dense paired effects;
- latency, memory, and total lifecycle cost;
- uncertainty and complete failure denominators.

Decision:

- retain MoECOT only if it earns a meaningful useful-safe advantage without an
  unacceptable weak-arm or cost failure;
- simplify to dense if a dense control wins;
- if all candidates fail, use the observed residual—not another speculative
  mechanism list—to define the smallest successor;
- preserve the exact negative scope.

### N4 — Compose D1 And D2

State: `blocked on E2-E4 and N3`.

The final flagship conclusion separately states:

- whether the governed stack helped a competent local worker;
- whether the Theseus student was competent;
- whether modular specialization helped versus dense controls;
- whether the integrated local student plus stack produced useful natural work;
- total lifecycle cost and limiting residuals.

No conclusion inherits support from another axis.

## Work That Is Explicitly Sidelined

The following work is frozen because it does not shorten P1-P5 or N0-N4:

- new architectures, attention variants, optimizers, tokenizers, objectives,
  curricula, speculative decoding modes, or same-scale rescue patches;
- KERC/RDC K4-K8, OneCell, SymLiquid expansion, CGS mechanism expansion,
  Coil/RankFold/NeuralFold, and alternative-substrate implementation;
- additional ANE, CPU/GPU/NPU partitioning, Rust rewrites, or generic
  acceleration searches without a measured selected-route defect;
- broad corpus growth, bulk teacher generation, public calibration, preference
  optimization, RL, self-training, continual learning, or unlearning;
- new benchmark, private-suite, report, dashboard, document, registry, or
  product-surface families;
- Hive networking, trusted-peer federation, public gateways, compute markets,
  licensing expansion, mobile, spatial, voice, multimodal, embodied, and
  multi-user work;
- LAN/public exposure, packaging, release engineering, and production serving;
- cross-platform parity, CUDA work, continuous batching, and system-energy
  studies not required by the selected Mac experiment;
- book chapter growth, broad prose rewrites, theorem-count expansion, reader
  derivatives, publication packaging, and reconciliation of future live-book
  drift beyond the current immutable 84-chapter pin;
- cleanup that does not remove a blocker, reduce evidence ambiguity, recover
  material disk, or eliminate an active duplicate owner.

Failures and incomplete ideas remain catalogued with their exact evidence
scope. Sidelined means preserved and nonblocking, not refuted or deleted.

## Re-entry Rule

A sidelined item may re-enter only when a terminal flagship result identifies
it as the smallest causal intervention for a measured defect and the proposed
experiment states:

- exact defect and evidence;
- why existing owners cannot repair it;
- matched baseline and negative control;
- expected decision value;
- resource and opportunity cost;
- rollback and retirement conditions;
- maximum inference.

Curiosity, utilization, theoretical elegance, an unread paper, a green
mechanics canary, or a desire to avoid a negative result is not a re-entry
condition.

## Autonomous Execution Contract

Work inside this envelope proceeds without task supply, labeling, review,
routine approval, scheduling, or continued presence from Corben:

- read-only audits and source reconciliation;
- read-only acquisition from allowlisted online task/data sources with local
  caching, provenance, license, retention, and contamination receipts;
- P2A, P4, and P5 local, no-teacher experiments after their machine gates pass;
- P3 measurement-only Luna reference calls after the exact transport, task,
  no-serving, no-training, cost, and evaluator gates pass;
- N0 source/custody readiness and implementation of the automated N1 launch
  controller with the training hold installed;
- bounded N1-N2 execution after the automated controller proves exact source,
  custody, disk, watchdog, checkpoint, rollback, and stop predicates;
- one-shot D1/D2 consumption only through the sealed run registry after every
  predecessor and independence predicate passes;
- disposable-worktree or temporary-directory effects with exact rollback;
- tests, validators, independent replays, and public-safe evidence assembly;
- source changes that repair a discovered flagship defect and preserve the
  frozen experiment;
- commits that contain one coherent, reviewed, validated transaction.

The autonomous lane must not:

- remove a hold except through its exact machine-readable controller;
- exceed the configured compute, storage, call, token, or time envelope;
- rerun a consumed D1, D2, or public-calibration surface;
- change architecture, optimizer, data, objective, schedule, or evaluator;
- invoke external inference, transmit private data, or use online content at
  runtime serving;
- delete checkpoints, corpora, user files, or meaningful negative evidence;
- expose a service beyond loopback;
- publish, push, or mutate an external system without separate authority;
- convert assisted behavior into learned credit;
- retry a semantic failure as if it were infrastructure noise.

Jobs stop on their declared terminal condition or a measured causal resource
wall. They do not use clock-of-day restrictions, arbitrary available-memory
floors, or percentage-improvement hurdles.

Missing preauthorization is a terminal machine-policy wall. External
publication, destructive deletion, account spending outside a predeclared
budget, or mutation of a third-party system stays out of scope; the program
records the wall and continues with the next eligible local action instead of
asking Corben to unblock ordinary research execution.

## Twenty-Phase Disposition

The detailed matrix retains every obligation, but only the following phase
work is active.

| Phase | Disposition during proof campaign |
| --- | --- |
| 0 Repository self-model | Split the dirty transaction; maintain only what P1/N0 or evidence custody requires |
| 1 VIEA spine | P1 route integrity and later L0/D1 joined traces |
| 2 Stable capability fields | Maintain authority and replacement identity |
| 3 VCM | P1 live integration, then L0/D1 natural causal comparison |
| 4 Candidate integrity | Mandatory for every experiment |
| 5 Autonomous local agent | P1-P3 canonical runtime and machine-selected paired work |
| 6 Deterministic tools | P1 integration and later L0/D1 attribution; no learned credit |
| 7 Teacher/data governance | Freeze corpus and preserve accounting |
| 8 Resource/acceleration | N0/N1 selected-route maintenance only |
| 9 Hive | Sidelined |
| 10 Practical neural seed | N0-N4 |
| 11 Discovery substrates | Sidelined |
| 12 Public calibration | Locked |
| 13 Semantic IR/KERC | Sidelined except existing Semantic-IR control |
| 14 Claims/proof records | D1 claim disposition only |
| 15 Procedural memory | P1 live integration, then L0/D1 natural reuse comparison |
| 16 MoECOT/Octopus | N2-N4 verdict |
| 17 Simulation/fidelity | Sidelined |
| 18 Governance/security | P1 live effect boundary, then L0/D1 authority/rollback evidence; local-only |
| 19 Book synchronization | Selected flagship claim crosswalk only |

## Definition Of Done

The reoriented program is complete when:

1. the selected highest-leverage ASI Stack ideas have exact claim identities,
   faithful causal implementations, matched controls, and maximum-inference
   boundaries;
2. the fixed local model is served through the canonical runtime and
   every claimed subsystem is causally live rather than decorative;
3. machine-selected eligible work has matched direct and integrated candidates
   with honest verifier-accepted/missed/failed accounting and practical
   latency;
4. each selected subsystem idea has a terminal L0 development disposition and
   any claim-bearing idea advances only through fresh D1 qualification;
5. D1 has a terminal natural governed-stack result;
6. repeated-work routing, planning, VCM, verification, and reuse have scoped causal
   dispositions;
7. a clean live Theseus replay and public-safe evidence packet exist;
8. the exact modular candidate and both dense controls complete matched
   training;
9. a fresh, private, source-disjoint D2 surface passes an independent
   no-reuse audit, is consumed once, and produces a modular-versus-dense
   verdict;
10. D1 and D2 are composed without denominator or support leakage;
11. each affected ASI Stack claim receives a reviewed transition or explicit
   no-promotion decision;
12. a concise public-safe evidence brief communicates the strongest result and
   its limits;
13. every promoted local default remains reversible with regression and
    rollback evidence; and
14. runtime external inference and public-benchmark training remain zero.

## Canonical Verification Commands

```bash
python3 scripts/theseus_doc_link_audit.py
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
```

For the held step-11416 campaign, refresh the current readiness chain with:

```bash
python3 scripts/pre_long_run_acceleration_residual_audit.py
python3 scripts/pre_long_run_independent_readiness_audit.py
python3 scripts/pre_long_run_replacement_freeze.py
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
```

`reports/pretraining_architecture_freeze_package.json` is the sealed historical
step-3480 architecture transaction. Do not regenerate it against the evolved
step-11416 plan; current authority flows through the replacement-freeze chain
above.

These commands produce evidence. They do not remove the training hold, consume
an evaluation surface, or promote a claim.
