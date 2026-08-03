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

Active claim: `virtual-context-abi.core`.

Prior-claim terminal disposition: the exact frozen TMax plus compact Semantic-IR
v6 campaign sealed one candidate, then a fresh 45,113-token full-source prompt
generated zero tokens before the prospectively declared 600-second host wall.
The exact implementation is `INCONCLUSIVE_EXPERIMENT` and frozen for this
TMax/host block. That result says nothing decisive about cognitive compilation
or Semantic IR generally. The residual directly activates governed model-visible
context materialization, so VCM is next by causal relevance rather than chapter
order or implementation convenience.

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

Metadata selection remains source-bound for all 18 repository identities, PR
titles, selected changed paths, and license files. Source-only materialization
exposed that GitHub's `merge_commit_sha` is merge-method dependent: for one
rebased PR it names a docs-only final commit whose first-parent delta omits the
PR-head source repair. Both RED materialization receipts are preserved. Before
any evaluator or model call, the task revision policy is therefore frozen to
the public PR `base.sha` and `head.sha`; merge commit and first parent remain
lineage receipts only. Renewed metadata acquisition and source materialization
must pass under that policy before independent evaluator qualification opens.
That source boundary is now GREEN: 18/18 task pairs produced 36 deterministic
minimal archives, and an independent network-free audit verified all archive
and member receipts, normalization, path safety, exact membership, and selected
source differences. Parent-negative/target-positive evaluator qualification is
the active boundary; no model or reference call is yet authorized.

Evaluator-custody construct review then found that task 11's selected source
diff is only one blank line. The archive layer remains valid, but the task panel
is RED because a byte difference is not a causal guard/setup mechanism. A
source-only search selected MIT-licensed `stn1slv/md-fetch#27` as the replacement:
its production delta removes an eager optional import and inserts a bounded
ImportError setup guard before client construction. The other 17 tasks, stratum
balance, and statistical design are unchanged. Replay all 18 metadata and source
pairs before evaluator admission or any model call. The full amended metadata
replay is now GREEN for 18/18 rows in 90 public calls; its initial zero-row
sandbox transport pause is retained. Source-byte replay and independent archive
and construct audits are the active boundary. The amended source contract is now
hash-bound, preserves the prior v3 archive set, targets a separate v4 directory,
and passes zero-call preflight before retrieval. The v4 replay is GREEN for
18/18 pairs and 36 archives; an explicit-hash network-free audit verified 36
archive receipts, 76 members, exact safe membership, and 18 selected-source
differences. All 17 unreplaced archive pairs are byte-identical to v3. Renewed
construct review names one causal slice and evaluator obligation per task, so
parent-negative/target-positive evaluator qualification is active. The pool is
not sealed and model or reference execution remains closed. Independent
dependency-stubbed evaluator qualification is now GREEN for all 18 causal slices:
36 parent/target observations plus 72 benign-equivalence, mechanism-removal,
missing-path, and unauthorized-path controls pass. Candidate-visible task packet
materialization and a recursive anti-cheating audit are the remaining pre-model
boundary.

Those boundaries subsequently closed and the balanced 18-task pool was sealed.
The first execution attempt preserved Task 1 but consumed Task 2 through an
abnormally long unbounded generation, so Task 2 was replaced—never rerun—with a
fresh post-snapshot task from a different licensed repository. The call-free v2
resume audit then passed. During that sealed resume, replacement Task 2 and Task
3 completed normally; Task 4 call 1 completed normally; and Task 4 call 2 reached
the prospectively declared 600-second host watchdog after 2,288 generated
tokens. The downstream route correctly held the partial response, no hidden
evaluator ran, and no model, mechanism, or claim failure may be inferred.

Candidates 1–3 and every v2 receipt remain immutable. The prospectively bound
v3 replacement then reached the same 600-second infrastructure wall on Task 4r1
call 1 after 1,949 generated tokens, without touching the context boundary or
an answer-length cap. Inspection localized the representation defect: a
one-line repair was addressable only as a 433-line top-level `FunctionDef`.
That watchdog observation is retained as invalid infrastructure evidence, not a
model, mechanism, or claim failure.

The v4 successor exposes every complete candidate-visible Python statement and
forbids silent address-inventory truncation. It reduces that demonstrated
433-line mutation span to one line while preserving the exact production
renderer, transport, parser, lowerer, applier, verifier, and repair route. Four
licensed repair sources from scikit-learn, Django, NetworkX, and Black were
merged after the frozen TMax snapshot and were frozen before evaluator
qualification. Their independent evaluators pass parent-negative,
target/benign-positive, mechanism-removal, missing-path, and unauthorized-path
controls. Those four replacements plus fourteen previously unexposed sources
form a fresh 18-repository, six-stratum denominator.

The fresh pool is sealed and the v4 campaign is GREEN under a call-free audit.
Exact frozen-tokenizer measurement shows every prompt fits the 262,144-token
physical context window; the largest prompt is 124,138 tokens and the smallest
residual is 138,006 tokens. The campaign authorizes exactly 36 local calls,
normal completion by complete artifact or model EOS, no project-selected
quality token cap, and no hidden evaluation until all 18 candidates seal. Luna,
teacher, training, claim-development, D1, and D2 remain closed.

The first v4 execution then consumed Task 1 call 1 without sealing a candidate.
Its 124,138-token prompt remained physically addressable, but prompt ingestion
occupied the full 600-second host wall and the backend generated zero tokens.
The context boundary and project-selected quality cap were not touched. The
campaign stopped with zero admitted candidates and zero hidden evaluation. This
is retained infrastructure evidence: exact context fit is necessary but not
sufficient for a usable instrument on this host. Task 1 may not be rerun.
Acquire one smaller licensed post-snapshot source in the same stratum, qualify
its evaluator before packet creation, rebind unexposed Tasks 2-18 unchanged,
and seal a fresh v5 campaign before any further model call.

That v5 replacement transaction is now sealed. Apache-2.0
`dknowles2/pytboss#546` supplies the new Task 1 exception-translation repair;
its exact source and license bytes were frozen before the independent async
behavior evaluator ran. Parent is negative, target and benign controls are
positive, and mechanism-removal, missing-path, and unauthorized-path controls
are negative. The exact Task 1 prompt falls from 124,138 to 9,165 tokens. Tasks
2-18 retain byte-identical serialized prompts under new v5 custody paths, the
18 repositories remain unique, and every stratum still contains three tasks.
The v5 campaign audit is GREEN and call-free, prospectively authorizing exactly
36 frozen-local calls. Hidden evaluation remains closed until all candidates
seal.

The v5 execution then sealed Tasks 1-3 through six admitted frozen-local calls.
Task 4 call 1 received a 74,626-token prompt and reached the 600-second host wall
with zero generated tokens. It did not touch the physical context boundary or a
project-selected answer-length limit. The campaign stopped before candidate
admission for Task 4 and before every hidden evaluator. This reproduces the v4
prefill-scale wall on a different source and preserves three unscored candidates
plus one invalid infrastructure receipt; it is not evidence against TMax,
Semantic IR, or the book claim.

The next repair is representation-wide rather than another one-off task swap.
The complete statement inventory currently repeats a full path, 64-hex digest,
and label for every node even though the full parent source is also visible.
Bind a compact statement-address ABI that keeps every address and resolves
integrity independently, prove collision resistance and exact parse/lower/apply
behavior, measure exact frozen-tokenizer prompt size without truncation, and
then seal one uniform fresh denominator. Every v5-exposed source is consumed;
unexposed sources may be rebound only under the same compact protocol. No model
call may precede the call-free audit.

That compact ABI is now mechanics-GREEN. Every statement receives a unique,
collision-checked 128-bit handle derived from path, type, exact coordinates, and
the full node digest. Candidate context groups handles by path and omits repeated
labels and full digests; the parser resolves the full digest independently before
the unchanged lowerer runs. All 18 prior sources retain identical statement
cardinality, complete parent context, and zero inventory truncation. Exact frozen-
tokenizer prompts remain physically addressable; the largest is 44,325 tokens,
and consumed Task 4 falls from 74,626 to 35,917 tokens. The audit used zero model,
hidden-evaluator, external, teacher, training, D1, or D2 calls. This is only
representation-mechanics evidence. Acquire and evaluator-qualify four new
post-snapshot sources for consumed indices 1-4, uniformly rebind unexposed 5-18,
and seal the fresh campaign before testing host operability.

The source and pool portion of that boundary is now GREEN. Exact PR base/head
pairs from Apache-2.0 LightLLM, GPL-3.0 translation-finder, BSD-3-Clause feu,
and BSD-3-Clause statsmodels replace consumed indices 1-4 and are repository-
disjoint from the full prior denominator. Independent evaluators distinguish
parent, target, benign, required-mechanism mutation, missing-path, and
unauthorized-path controls before packet creation. Unexposed indices 5-18 were
regenerated—not byte-copied—under the same compact ABI. The resulting pool has
18 unique repositories, three tasks in each of six strata, zero candidate/model
calls, and a 45,113-token maximum exact prompt. The v6 campaign binding is also
now GREEN under a call-free audit. It hash-binds the exact compact runtime,
frozen TMax identity, independent fresh/base evaluators, candidate runner, blind
scorer, complete-artifact/EOS termination, 600-second infrastructure watchdog,
and exactly 36 local calls. Its versioned receipt namespace and journal cannot
collide with or resume from v5. Candidate generation is now the active step;
hidden scoring remains closed until all 18 candidates and 36 receipts seal.

The v6 execution then sealed Task 1 through two normal local calls. Task 2 call
1 received the panel's largest 45,113-token prompt and reached the host wall
with zero generated tokens after 609.5 seconds. The physical context boundary
was not touched, no project-selected answer-length cap existed, the route held
the empty response, and no hidden evaluator ran. The terminal independent
disposition is `INCONCLUSIVE_EXPERIMENT`: preserve Task 1, consume Task 2 without
rerun, freeze this exact Semantic-IR implementation for the current model/host
block, and do not spend another fresh denominator on representation repair.

### 3. Bind the VCM claim instrument

Use the existing VCM owner rather than creating a new context lane. First prove
that correct, omitted, stale, shuffled, wrong-scope, tainted, revoked, and
declared-insufficient packets traverse distinct production paths and that the
actual selected packet reaches the frozen model. Then prospectively bind a
source-disjoint natural-work experiment comparing correct VCM context with no
added context, information-matched plain context, and maximal ungoverned context.
Measure useful completion, requirement preservation, unsafe release, false
block, latency, exact prompt/generated tokens, retrieval/materialization work,
and total cost. Context size and retrieval work are causal resources and may be
prospectively matched; generated answer length may not be capped for quality.
No model or Luna call opens until the complete instrument, tasks, evaluators,
controls, contamination checks, and inference bounds are sealed call-free.

That call-free instrument audit is now GREEN. It replays the existing governor,
resolver, representation certificates, snapshot branches, 45 consumer routes,
and fail-closed missing, stale, tainted, revoked, wrong-scope, compressed, and
route-materialization controls. It also proves that correct and shuffled VCM
packets create distinct bound model prompts while the exact information-matched
plain context bypasses VCM without changing candidate-visible content.

The powered design uses a nine-task source-disjoint control-qualification panel
to freeze the strongest eligible local non-VCM control without Luna input. The
claim panel then uses 53 new source-disjoint tasks and one exact one-sided paired
McNemar primary comparison. A 35-point absolute effect is the minimum worth the
selection/governance/lifecycle complexity; 53 is the first task count with at
least 0.80 numerically minimized power over the closed feasible discordance
interval from 0.35 through 1.0 (worst-case power 0.8173). Local VCM and the
frozen local control form the claim denominator; Luna runs the same two packets
as a separately denominated measurement reference. Source acquisition and
evaluator qualification are now active, but every model and Luna call remains
closed.

The first acquisition boundary is sealed and GREEN before candidate inspection.
It authorizes a bounded public-metadata query only, excludes 87 repositories
already named in tracked experiment configs, requires fresh PR and head-commit
chronology plus a licensed in-scope source change and machine-verifier change,
and uses independent hashes for selection and control/claim assignment. Source
contents, task packets, hidden evaluation, TMax, and Luna remain closed.

Metadata attempt 1 terminated RED after 1,212 requests because only 4 of 20
required Python repositories passed; it retained zero task identities and made
zero source-content, evaluator, TMax, or Luna calls. Its rejection distribution
identified the 50-star floor as an arbitrary popularity proxy and the metadata
license allowlist as needlessly narrow relative to the charter. Preserve this
attempt and prospectively repair those policy defects without weakening
freshness, source-plus-test structure, source disjointness, or inference closure.

The v2 repair is prospectively sealed and GREEN before its first query. It uses
a one-star existence check rather than a popularity proxy and recognizes common
OSI licenses. The query window, hash ranking, quotas, freshness rules,
source-plus-test requirement, source disjointness, and every downstream closure
are unchanged. Its run improved Python eligibility from 4 to 11 of 20 but still
failed closed after 1,208 requests with zero retained identities and zero
downstream calls. V3 may expand only the deterministic candidate head from 300
to the GitHub 1,000-result boundary per language and batch the same metadata
fields; eligibility may not be relaxed.

V3 is now prospectively sealed and GREEN at zero queries. It expands each fixed
language head to the GitHub 1,000-result boundary and evaluates bounded batches
concurrently while consuming results in the original hash rank. This is a
transport and pool-size repair only: every eligibility, chronology,
randomization, panel, and authority rule is unchanged. One v3 run is active.

The v3 run was interrupted before any identity sealed because a fork-origin PR
head commit was not addressable through the base repository and the concurrent
owner leaked the 404 as an uncaught exception. The interruption receipt records
zero source, evaluator, or model exposure. V4 may only use the PR commit-list
endpoint for chronology and fail paused on transport errors; the pool,
eligibility, randomization, panels, and authority must remain unchanged.

V4 is prospectively sealed and GREEN at zero queries. It resolves chronology
through the base PR's commit list, including fork-origin PRs, and converts any
transport exception into a PAUSED receipt. Pool, eligibility, rank order,
panels, and authority are unchanged. One v4 run is active.

V4 then paused correctly on a transient public-metadata failure, and the same
endpoint succeeded on immediate recheck. It retained zero identities and opened
no source, evaluator, or model path. Its inherited zero request counter is
superseded by exact-unknown accounting with a 40-request lower bound. The next
repair may add only bounded retries and checkpointed accounting; selection
science remains frozen.

V5 is prospectively sealed and GREEN at zero queries. It permits at most four
physical attempts for transient metadata failures, treats stable candidate
404/410 gaps as rejections, and checkpoints logical and physical counts after
every attempt without storing repository identities. The pool, eligibility,
chronology, rank order, panels, and every downstream closure remain frozen. One
v5 metadata run is active.

V5 then paused after exhausting bounded retries under GitHub secondary
throttling. Its checkpoint records 239 logical requests, 271 physical attempts,
231 successes, 32 retries, and 39 HTTP 403 attempts; core quota still had 4,663
of 5,000 requests. It retained zero identities and opened no downstream path.
V6 may replace only the per-candidate REST fan-out with GraphQL node batches for
the same metadata fields; the REST search population, rank order, filters,
panels, and authority remain frozen.

V6 is prospectively sealed at zero fresh-candidate queries. GitHub's live
schema exposes every required field, the REST search `node_id` resolves as a
GraphQL pull request, and the exact `nodes(ids:)` query returned the expected
PR, repository, file-path, and commit metadata for an already
consumed denylisted pull request without requesting body, patch, or review
content. It batches at most 40 nodes through one GraphQL request at a time,
keeps the 40-call REST search population unchanged, spaces those calls by at
least 2.1 seconds against the observed 30-per-minute search quota, and rejects a last-commit
identity that does not match the PR head. The full metadata run is bound not to
start before `2026-08-03T10:43:25Z`; source retrieval, evaluators, local and
Luna inference, training, D1/D2, serving, and book promotion remain closed. The
search pacing is a host-transport condition, never a task or mechanism outcome.

The v6 run then failed closed after 58 logical requests and 62 physical
attempts: 57 requests succeeded, one HTTP 502 recovered, and four rapid
unknown-network attempts exhausted the inherited 1.75-second backoff horizon.
GitHub immediately reported search 30/30 and GraphQL 4,956/5,000, so quota
exhaustion is excluded. Zero identities, source bytes, evaluator calls, model
calls, or downstream authority sealed. The terminal audit also found that the
inherited owner hashed its checkpoint before finalization, leaving a stale
embedded checkpoint hash. The next repair may change only the bounded transport
backoff and finalize the checkpoint before report hashing; selection science
and every downstream closure remain frozen.

V7 is now prospectively sealed and GREEN at zero queries. It invokes the exact
v6 REST-search, GraphQL query, pacing, qualification, ranking, and panel owner;
only the transient/unknown retry horizon changes to eight attempts across at
most 108 seconds. It finalizes the checkpoint before hashing it into the report
and fails if the embedded and final file hashes differ. All source-content,
evaluator, local/Luna, training, D1/D2, serving, and book authority remains
closed. One unchanged metadata run is active.

Exit A: the exact production implementation passes the prospectively frozen
mechanics and intervention contract and may open one new claim-development
denominator.

Exit B: the exact implementation falls below the preregistered adequacy floor;
record `INCONCLUSIVE_IMPLEMENTATION`, retain the evidence, leave the book claim
unresolved, and repair or exclude the implementation owner. Do not translate
that scoped result into a broad negative for cognitive compilation or the book
claim, and do not manufacture a nearby green reseal.

### 4. Freeze one claim-development experiment

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
4. **Complete:** run the prospectively audited v6 compact adequacy denominator;
   preserve its one candidate and zero-token Task 2 watchdog; issue
   `INCONCLUSIVE_EXPERIMENT`; freeze this exact implementation without a broad
   negative or another Semantic-IR reseal in the current block.
5. **Complete:** bind and call-free audit the existing VCM owner, production-
   path controls, nine-task control-qualification panel, 53-task powered claim
   panel, separate Luna denominator, and zero-inference authority.
6. **Active:** autonomously acquire and independently evaluator-qualify 62
   licensed repositories split into source-disjoint 9-task control and 53-task
   claim panels. Attempt 1 failed closed with an insufficient Python pool;
   v2 improved eligibility but still failed closed. Preserve both; seal and run
   the pool-expansion v3 exposed a fork-head transport bug. Preserve it, repair
   only that owner; v4 then paused on transient transport with zero identities.
   V5 paused on secondary throttling with exact checkpoint accounting. V6
   passed live schema, node-bridge, quota pacing, and full selector rehearsal,
   then failed closed on a rapid unknown-network exhaustion after 57 successful
   requests. Preserve it. V7 now seals the bounded 108-second recovery horizon
   and final checkpoint hash verification at zero queries; run the unchanged
   batched metadata selector. If it seals, bind exact sources and
   packets,
   host-operability canaries,
   runners, blind scorers, calls, spend, and stop conditions before inference.
7. If it passes, freeze and run one new source-disjoint local-plus-Luna claim
   campaign, with Luna omitted rather than backfilled if its transport was not
   sealed before the first arm opened.
8. Advance a development survivor once to fresh D1; otherwise retain the exact
   terminal state and move the portfolio forward.
9. Return the claim packet to governed book review with support unchanged.
10. Repeat the single-claim cycle until the Subsystem Architecture Freeze can be
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
