# Project Theseus Roadmap

Last recentered: 2026-08-02 UTC.

This is the canonical forward execution map. Current facts belong in
`docs/PROJECT_STATE.md`; durable operating rules belong in `AGENTS.md`; exact
machine obligations and the book crosswalk belong in
`configs/roadmap_implementation_matrix.json`; implementation ownership belongs
in `configs/project_manifest_registry.json`. Historical reports are evidence,
not an alternate roadmap.

## North Star

Project Theseus exists to implement and rigorously test the largest,
highest-leverage ideas in *The ASI Stack*.

The near-term product is not a personal assistant and it is not a newly trained
Theseus model. The near-term product is a trustworthy proving system that can:

1. convert an exact book claim into a causally active subsystem intervention;
2. hold the underlying model fixed;
3. compare the intervention with strong information-, tool-, verifier-, and
   cost-matched controls on autonomous machine-verifiable work;
4. distinguish a mechanism result from an inadequate implementation;
5. qualify a survivor once on fresh source-disjoint evidence; and
6. return a claim-level packet to the living book without automatically moving
   support.

Autonomous repository work is the experimental substrate. It is not the reason
Theseus exists. Corben is not a task source, labeler, acceptance gate,
scheduler, or routine approval dependency.

## The Program Is Deliberately Bifurcated

| Track | State | Job | Boundary |
| --- | --- | --- | --- |
| A — ASI Stack subsystem proof | **ACTIVE FLAGSHIP** | Test one architecture-shaping book mechanism at a time with a frozen local model and a separately denominated OpenAI reference | The only track allowed to open new claim-development work |
| B — Theseus neural seed | **HOLD: SUBSYSTEM PROOF FIRST** | Preserve the current modular/dense experiment, checkpoint, corpus, controls, and sealed D2 identity | No optimizer steps, architecture changes, or D2 consumption until the subsystem architecture freeze below |
| M — maintenance | **SERVICE ONLY** | Keep source binding, custody, storage, tests, and evidence replay honest enough for A and B | May remove a real blocker or material storage wall; may not become a research program |

The previous roadmap treated neural training as parallel progress. That was the
wrong strategic selection. Training a model while the interfaces, control
layers, and causal value of its intended subsystems remain unresolved risks
baking untested assumptions into an expensive experiment. The neural state is
therefore preserved but inactive.

## Current Evidence Boundary

| Surface | Current state | What it means |
| --- | --- | --- |
| ASI Stack source | `84_CHAPTER_PIN_STABLE_LIVING_BOOK_MOVING` | The experiment claim identities remain pinned to AI_book commit `17c6ece80f771d3bce5f89c6b85c99ca9b6c2ea0`. The concurrently advancing living book was observed at `1280fdfa457735ae39289d42a82173f9485434fc` during the final audit; committed and working manifests still had 84 chapters, and its uncommitted work remains intake-only. The selected core-claim vector file was unchanged from the pin at that observation. Freeze experiment semantics, observe book drift, and return evidence by stable claim ID. |
| Local controlled variable | `FROZEN_TMAX_BLOCK` | `mlx-community/Tmax-9B-MLX-8bit@33812d6cf04f88856f25eb828de4f3144a194560` is the static local base for the current mechanism block. A later local model creates a new instrument version and cannot be mixed into this denominator. |
| Canonical runtime | `P1_ROUTE_MECHANICS_GREEN` | Direct and integrated routes use the same frozen model and effect sandbox; enabled subsystem receipts reach downstream execution. This proves route mechanics, not usefulness. |
| Historical P3 | `TERMINAL_BOUNDED_NULL` | Ten matched tasks yielded direct 1/10 useful and integrated 1/10 useful, with parseability 5/10 versus 9/10 and zero unsafe outcomes. Its project-selected 1,536-token ceiling makes it historical bounded evidence, not current claim proof. |
| Cognitive-compilation P4 | `TERMINAL_INCONCLUSIVE_IMPLEMENTATION` | The source-locked run completed 10 tasks, 60 learned calls, 10 evaluator replays, and zero physical-boundary hits. Direct was useful on 3/10, plan control on 1/10, and Semantic IR on 0/10; Semantic IR parsed/lowered only 2/10 against the frozen 8/10 mechanics floor. The implementation is inadequate, so no treatment-effect or broad negative inference is authorized. |
| D1 | `CLOSED_NO_SURVIVOR` | No fresh qualification is allowed for the failed Semantic-IR implementation. The prepared D1 machinery remains unconsumed. |
| OpenAI reference | `TRANSPORT_SOURCE_BOUND_OFFLINE_QUALIFIED_ZERO_CALLS` | The exact `gpt-5.6-luna` Responses API adapter at fixed `xhigh` effort is source-bound and passes offline positive and negative controls. Calls remain disabled; it has made zero project calls and cannot open until a future four-cell claim pool is prospectively sealed. |
| Neural checkpoint | `HELD_STEP_11992_NOT_EVALUATED` | Nine transactional 64-step segments advanced the preserved shared trunk from step 11,416 to 11,992 and from 87,441,996 to 91,869,446 optimizer positions. D2 remains sealed and capability remains `NOT_EVALUATED`. |
| Repository | `SOURCE_BOUND_REGISTRY_GREEN` | The earlier 88-file mixed transaction is historical. The recenter and registry repair are committed, and the complete registry gate reports zero abstraction gaps, route blockers, hard governance failures, or missing AIBOM identities with 11 eligible routes source-bound. The large generated surface remains real: about 41 GiB/250,030 runtime files, 2.1 GiB/10,790 report files, and 9.0 GiB of checkpoints. |

## Scientific Contract

### Hold the model fixed

The current local model, exact revision, tokenizer, chat template, decoder,
completion policy, and runtime stay fixed across a mechanism comparison.
Model-shopping after outcomes are visible is forbidden. If the local model is
replaced, every baseline needed for the new claim block is re-established under
a new identity.

TMax is the current qualified local instrument, not a timeless claim that it is
the best model the host can ever run. A successor block may select a stronger
local model only through a prospectively frozen, claim-task-disjoint competence,
runtime-stability, context, and total-cost bakeoff conducted before that block's
tasks or subsystem outcomes are visible. Selection never uses Luna outputs. The
winner is then frozen across the whole block; old and new denominators are not
pooled.

The model is a controlled instrument, not the treatment. Theseus subsystems
must change real execution: admitted context, executable dependencies, tool or
route eligibility, verification allocation, repair scope, authority, or
rollback. A label, prompt decoration, report, or rendered plan is not an active
mechanism.

### Test one causal variable

Every claim packet must bind before task selection:

- exact book claim and immutable claim hash;
- causal variable and faithful implementation;
- primary outcome and minimum worthwhile effect;
- competence and mechanics requirements;
- strongest matched controls and ablations;
- candidate-visible information and hidden evaluator information;
- task population, source-disjointness, and power method;
- safety, rollback, weak-tail, latency, verifier, token, and total-cost
  accounting;
- terminal states and maximum positive and negative inference; and
- the next action for a positive, adequate null, implementation failure,
  experiment failure, or invalid observation.

Do not test an omnibus “full stack” treatment when the result cannot be
attributed. Common safety containment remains in every arm; optional mechanisms
change one at a time.

### Separate mechanics from claim evidence

A mechanism first passes a non-claim mechanics bench. That bench may use
synthetic or reconstructed fixtures, known-positive artifacts, corruption
tests, overfit/learnability checks, and production-path round trips. It receives
no usefulness or book-support credit.

No fresh claim denominator opens until the exact production renderer,
tokenizer, model-visible protocol, returned transport, canonicalizer, parser,
lowerer, applier, verifier, intervention, and repair path pass together. A green
test-only renderer is insufficient.

If a mechanics owner fails its prospectively declared adequacy contract, retain
`INCONCLUSIVE_IMPLEMENTATION`, freeze that implementation, and move the
portfolio to the next decision-relevant residual. Do not consume another fresh
task pool to rediscover the same parser or transport defect.

### Preserve blind information flow

Generation and ranking see only the natural request, callable signature,
parent/source snapshot, and explicitly allowed runtime context. Later patches,
hidden tests, source-task identities, answers, labels, target-derived families,
and answer-identifying decoder fields remain hidden. An independent audit
recomputes candidate integrity and route blindness.

### Do not manufacture quality with token scissors

Quality-bearing generation ends on a prospectively defined complete artifact
or model EOS. There is no project-selected generated-token quality cap. The
exact residual of the pinned model context after exact prompt tokenization is a
physical addressability boundary; touching it invalidates the observation and
cannot count as a model, mechanism, arm, or architecture failure. Host-safety
stops are handled the same way.

Calls, verifier work, effects, time, and money may be bounded because they are
causal or safety resources. Record actual prompt, reasoning, generated tokens,
wall time, verifier work, retries, and cost. Do not force models to emit equal
token counts.

### Scope negative evidence

Only a faithful, adequate, powered implementation with strong matched controls
can earn a scoped negative. Failed mechanics, underpowered pools, weak models,
missing interventions, evaluator defects, or boundary hits produce
`INCONCLUSIVE_IMPLEMENTATION`, `INCONCLUSIVE_EXPERIMENT`, or an invalid
observation. They never falsify the broader book mechanism.

## Bound Claim Portfolio

The existing 13-claim portfolio remains the decision boundary; the 84-chapter
crosswalk is planning coverage, not a mandate to implement 84 simultaneous
lanes.

| Role | Claims | Program use |
| --- | --- | --- |
| Integrity prerequisites | `integrated-reference-architecture.core`; `evidence-states-and-claim-discipline.core` | Common evidence requirements; never counted as a usefulness win |
| Architecture-shaping candidates | `cognitive-compilation-and-semantic-ir.core`; `planning-as-a-control-layer.core`; `virtual-context-abi.core`; `verification-bandwidth-and-context-adequacy.core`; `routing-heads-and-specialist-cores.core` | Must receive terminal interface dispositions before the neural architecture freeze |
| Runtime amplifiers | `procedural-memory-and-cognitive-loop-closure.core`; `system-boundaries-and-authority.core`; `capability-replacement-and-rollback.core` | Enter when an observed residual activates them; authority and rollback remain common constraints |
| Independent neural claim | `replaceable-cognitive-substrates-beyond-transformer-monoculture.core` | Held D2 modular-versus-dense experiment after subsystem architecture freeze |
| Synthesis | `the-efficient-asi-hypothesis.core`; `asi-is-a-stack-not-a-model.core` | Require multiple independently qualified mechanisms and a later composition experiment |

The queue is a dependency graph, not a checklist. Exactly one candidate is
active. After each terminal result, the observed residual and architecture
decision value choose the next candidate; claim order never licenses a batch of
campaigns.

## Active Proof Cycle

### 1. Bind the current claim

Active claim: `cognitive-compilation-and-semantic-ir.core`.

Current maximum inference: the exact frozen TMax plus Semantic-IR v2r2
implementation did not meet its own mechanics floor on the consumed P4 surface.
That result says nothing decisive about cognitive compilation generally.

### 2. Qualify the repaired mechanics owner independently

The observed identity/coverage/target/unit transport wall now has one canonical
role-aware production owner. It passes ten deterministic production-path
conformance fixtures, exact identity round trips, seven corruption classes, and
one frozen-TMax model-produced non-claim canary through the real route. The
canary made two naturally completed local calls and zero external, hidden,
teacher, training, D1, or D2 calls.

That closes the bounded repair stage only. It does not establish distributional
competence or a treatment effect. The independent adequacy owner must now:

1. verify mechanism fidelity and known-positive reachability independently;
2. acquire a licensed, source-disjoint, model-produced non-claim panel without
   user task supply or labels;
3. cover multiple edit shapes, targets, dependency structures, and difficulty
   strata with weak-tail reporting;
4. prove stable identities, corruption rejection, intervention sensitivity,
   dependency-local repair, and unrelated-unit stability; and
5. derive the competence floor and panel size from the planned observable
   effect rather than copying a ten-task tradition.

The prospective design is now frozen before source materialization or model
calls: 18 distinct repositories, six edit strata with three tasks each, at
least 13/18 overall and 2/3 in every stratum, all identity/corruption and
known-positive controls, and zero unsafe or physical-context-boundary events.
For the one-sided exact binomial design, the false-positive probability under
an inadequate 0.50 mechanics rate is 0.048126220703125 and power at an adequate
0.80 rate is 0.867083665757176. Source acquisition is restricted to licensed
public changes merged strictly after the frozen model was observed; no model or
reference call is authorized until the complete panel is sealed.

Metadata selection is complete and source-bound: all 18 repository identities,
PR titles, merge revisions, first-parent revisions, selected changed paths, and
license files were independently recomputed through 90 public GitHub metadata
requests with a retained response-digest chain. The first urllib transport
attempt failed closed with zero admitted rows and is preserved. Source archives,
parent/target evaluators, and model calls remain unopened; the next boundary is
source materialization and independent evaluator qualification.

Exit A: the exact production implementation passes the prospectively frozen
mechanics and intervention contract and may open one new claim-development
denominator.

Exit B: it fails; retain the implementation failure, freeze this candidate,
leave the book claim unresolved, and activate the next architecture-shaping
residual. No nearby reseal is allowed.

### 3. Freeze one claim-development experiment

Only after Exit A, autonomously acquire a new licensed, source-disjoint task
cohort. Pool size and task diversity come from the predeclared effect and power
analysis. Every parent must fail and exact target must pass an independent
network-free evaluator before candidate generation.

The primary model/subsystem factorial is:

| Model block | Mechanism absent | Mechanism present |
| --- | --- | --- |
| Frozen TMax local | Primary local baseline | Primary local causal treatment |
| Frozen Luna reference | Hosted reference baseline | Hosted reference robustness treatment |

Mechanism-specific controls—such as an information-matched natural-language
plan or deterministic compiler—remain required when they are stronger than the
plain direct baseline. They do not change the primary within-model contrast.

The independent evaluator, task packet, callable surface, allowed context,
effect sandbox, verifier opportunity, and prospectively declared call/resource
budget are matched within each model. The local within-model effect is primary;
the Luna within-model effect is a separate reference. Cross-model ranking and
the difference of those effects are descriptive because wrappers, tokenizers,
reasoning tokens, latency, cost, and training differ.

### P3 hosted reference control

The current requested reference is `gpt-5.6-luna` at fixed `xhigh` reasoning.
It is a cost-efficient hosted reference, not a frontier ceiling and not a
serving candidate. Its exact callable surface, provider identity, model alias or
snapshot, effort, wrapper, tool policy, completion policy, price basis, and cost
authority must be source-bound before any arm in that future claim pool opens.

If the transport is unavailable at seal time, the local experiment may proceed
with an explicit missing-reference state. The Luna cells may not be backfilled
after local outcomes are inspected. Luna outputs are never served, admitted as
training rows, used to select or tune tasks, used to tune the subsystem or pick
the next claim, granted source effects, or mixed into the local denominator.

Normal Luna completion is the complete artifact or model EOS. A provider output
or context ceiling is a physical transport boundary; touching it invalidates
the observation for capability inference. Fixed `xhigh` is a quality-oriented
reference choice, not a claim that `xhigh` is the most efficient effort.

### 4. Run and disposition once

Open all arms only after source, task, mechanism, model, prompt, evaluator,
budget, cost, and analysis identities are sealed. Counterbalance order. Preserve
every failure. Score useful-safe completion first, then weak tails and total
system cost. Parsing, plan validity, verifier acceptance, and route receipts are
diagnostics unless the claim names them as the primary outcome.

Terminal outcomes:

- `DEVELOPMENT_SURVIVOR`: adequate mechanism improves the predeclared primary
  estimand and passes safety, rollback, weak-tail, and cost vetoes;
- `ADEQUATE_NO_SURVIVOR`: faithful experiment completed but the exact candidate
  did not earn advancement;
- `INCONCLUSIVE_IMPLEMENTATION`;
- `INCONCLUSIVE_EXPERIMENT`;
- `INVALID_INFORMATION_FLOW`;
- `INVALID_EVALUATOR`; or
- `INVALID_OBSERVATION_CONTEXT_OR_HOST_BOUNDARY`.

Do not rerun, rescore, rename, or reseal a consumed denominator to obtain a more
convenient state.

### 5. Qualify only a survivor on fresh D1

A development survivor receives exactly one fresh, source-disjoint D1
qualification after its model, mechanism, competence floor, minimum worthwhile
effect, uncertainty, weak-tail, cost, evaluator, and terminal rules are frozen.
D1 never contains development tasks and never borrows Luna or neural credit.

### 6. Return evidence to the book

Every terminal implementation result may return a public-safe claim packet,
including null and inconclusive results. The packet binds exact evidence,
denominators, costs, residuals, limitations, and maximum inference. It proposes
no automatic support transition and grants no publication or release authority.

The living book may accept, reject, narrow, or block a separate claim-level
transition. Book prose drift never silently changes an already opened
experiment.

## Autonomous Task Contract

Theseus acquires experimental work without user supply:

- allowlisted, license-compatible online repositories;
- locally cached parent and target artifacts with provenance and digests;
- natural requests reconstructed without answer leakage;
- deterministic eligibility, contamination, and memorization-risk screens;
- candidate-visible parent state and hidden target/evaluator state;
- independent parent-fail/target-pass qualification;
- disposable local effects and exact rollback; and
- strict separation among mechanics, development, D1, D2, training, and public
  calibration.

Public benchmark prompts, tests, solutions, traces, and answer templates remain
calibration-only. A consumed task never becomes training data or a fresh
claim-bearing surface. No experiment opens pull requests, contacts maintainers,
or mutates upstream repositories.

## Neural Hold And Re-entry

The shared trunk is preserved at step 11,992. The source-controlled availability
policy and the runtime yield signal must both deny new segments. The hold is
strategic, not a statement that the checkpoint or architecture failed.

Training may re-enter only through a reviewed machine-readable **Subsystem
Architecture Freeze** proving all of the following:

1. every architecture-shaping claim above has either a qualified interface and
   scoped evidence or an explicit first-model exclusion with a terminal scoped
   disposition;
2. no unresolved subsystem decision would require changing the proposed model
   topology, training target, data representation, routing contract, verifier
   interface, or checkpoint semantics after training begins;
3. at least one production-equivalent end-to-end composition canary proves the
   selected interfaces join without label-only or prompt-decoration shortcuts;
4. the exact modular, dense-active, and dense-total question is still
   decision-relevant after the subsystem results;
5. the current checkpoint, optimizer, RNG, cursor, corpus, controls, and D2
   identity are rebound from clean source; and
6. autonomous resource, transaction, rollback, and one-shot D2 policies remain
   green without a user-presence or routine approval gate.

Until then:

- no optimizer steps;
- no architecture, optimizer, tokenizer, objective, data, KERC, ANE, or generic
  acceleration work;
- no D2 consumption;
- no capability inference from loss or training progress; and
- checkpoint custody and replay maintenance only.

## Maintenance That Serves the Proof Program

Use existing owners; create no new cleanup or report family.

1. Keep every claim transaction source-clean and commit-coherent. A mixed
   product/neural/Rust/release/cleanup transaction cannot open an experiment.
2. Run the existing reference-aware retention controller against the whole
   generated surface. Preserve terminal reports, negative evidence, active
   pointers, irreproducible acquisitions, checkpoint lineage, and independent
   recovery. Archive only manifest-bound, replay-safe candidates.
3. Treat 828 scripts, 416 top-level configs, 261 top-level tests, and thousands
   of report variants as an ownership smell. When an active family is touched,
   consolidate superseded wrappers and redirect callers through the registered
   owner; do not launch a repository-wide rewrite.
4. Add no dashboard, product surface, benchmark family, private suite, document
   family, or subsystem lane unless it is required by the active claim.
5. Measure storage, wall time, and report growth. A report is evidence, not
   progress.

## Immediate Execution Order

1. **Complete:** install the source-controlled neural hold and synchronize the
   roadmap, project state, matrix, README, and active flagship identity.
2. **Complete:** bind and offline-qualify the Luna measurement adapter, exact
   model/effort receipt, no-serving/no-training boundary, completion telemetry,
   evaluator interface, and campaign-derived cost authority with zero calls.
3. **Complete:** repair the exact Semantic-IR identity/coverage/target/unit
   mechanics on bounded non-claim evidence through the production path.
4. **Active:** complete the prospectively frozen selected-file source
   materialization and independent parent-negative/target-positive evaluator
   qualification, then run the independent adequacy audit. If it fails, freeze
   the implementation and select the next architecture-shaping residual; do not
   open fresh claim tasks.
5. If it passes, freeze and run one new source-disjoint local-plus-Luna claim
   campaign, with Luna omitted rather than backfilled if its transport was not
   sealed before the first arm opened.
6. Advance a development survivor once to fresh D1; otherwise retain the exact
   terminal state and move the portfolio forward.
7. Return the claim packet to governed book review with support unchanged.
8. Repeat the single-claim cycle until the Subsystem Architecture Freeze can be
   decided. Only then reassess neural training and D2.

## Explicitly Sidelined

- model training and D2;
- new model architectures, optimizers, tokenizers, objectives, curricula, or
  corpus expansion;
- KERC/RDC, OneCell, SymLiquid, CGS, VSA, Coil, RankFold, NeuralFold, ANE, and
  generic accelerator campaigns;
- preference optimization, RL, self-training, continual learning, or
  unlearning;
- public benchmark consumption beyond governed calibration;
- personal-assistant data collection;
- Hive networking, public gateways, arbitrary remote execution, compute
  markets, mobile, spatial, voice, multimodal, embodied, or multi-user work;
- LAN/public serving and release engineering; and
- broad ASI Stack rewriting or publication packaging unrelated to a returned
  claim packet.

A sidelined item re-enters only when a terminal active result identifies it as
the smallest causal response to a measured defect. Curiosity, available compute,
theoretical elegance, or a green mechanics report is not an entry condition.

## Definition of Progress

Progress means one of:

- a high-leverage claim receives a more faithful implementation or stronger
  matched causal evidence;
- an inadequate approximation is repaired before fresh evidence is consumed;
- a claim receives an honest scoped terminal disposition;
- a survivor qualifies once on fresh D1;
- a claim-level evidence packet reaches the book without support laundering; or
- source/custody/retention work removes a concrete threat to those outcomes.

More files, reports, routes, mechanisms, optimizer steps, or green mechanics
checks are not progress by themselves.

## Definition of Done for This Roadmap Era

This subsystem-first era is complete when:

1. the architecture-shaping claim set has terminal interface dispositions;
2. every positive claim has source-disjoint qualification and every negative is
   adequacy-bounded;
3. the chosen subsystem interfaces compose in the live local runtime;
4. local TMax and the separately denominated Luna reference have complete
   within-model evidence wherever the reference transport was prospectively
   available;
5. claim packets have returned to the book with no automatic promotion; and
6. the Subsystem Architecture Freeze makes an explicit evidence-based decision
   about whether and how to resume the modular-versus-dense neural experiment.

## Canonical Verification

```bash
python3 scripts/theseus_doc_link_audit.py
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
python3 scripts/theseus_asi_stack_claim_handoff.py
python3 -m pytest -q tests/test_roadmap_book_sync.py \
  tests/test_roadmap_pretraining_gate.py \
  tests/test_theseus_external_reference_control.py \
  tests/test_neural_seed_training_campaign.py \
  tests/test_neural_seed_autonomous_launch_controller.py \
  tests/test_theseus_asi_stack_claim_handoff.py
```
