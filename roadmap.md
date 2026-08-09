# Project Theseus Roadmap

Last deeply reviewed and recentered: 2026-08-08 UTC.

This is the canonical forward execution map. Current facts belong in
`docs/PROJECT_STATE.md`; durable rules belong in `AGENTS.md`; exact machine
obligations and the book crosswalk belong in
`configs/roadmap_implementation_matrix.json`; implementation ownership belongs
in `configs/project_manifest_registry.json`. Historical reports are evidence,
not an alternate roadmap.

## North Star

Project Theseus exists to implement and rigorously test the largest,
highest-leverage ideas in *The ASI Stack*.

The current product is a trustworthy proving system, not a personal assistant
and not yet a newly trained Theseus model. It must:

1. bind an exact book claim to a causally active subsystem;
2. hold the underlying model fixed while the mechanism changes;
3. compare the mechanism with strong, simple, matched controls on autonomous,
   machine-verifiable work;
4. distinguish mechanism failure from implementation or experiment failure;
5. qualify a survivor once on fresh source-disjoint evidence; and
6. return a claim-level packet to the book without automatically changing its
   support state.

Autonomous repository work is the experimental substrate, not the primary
goal. Corben is not a task source, labeler, acceptance gate, scheduler, or
routine approval dependency.

## Program Selection

| Track | State | Purpose | Boundary |
| --- | --- | --- | --- |
| ASI Stack subsystem proof | **ACTIVE FLAGSHIP** | Test one architecture-shaping book mechanism at a time with a frozen local model and separately denominated OpenAI reference | The only track allowed to open new claim-development work |
| Theseus neural seed | **HOLD AT STEP 11,992** | Preserve the modular-versus-dense experiment, checkpoint, corpus, controls, and sealed D2 identity | No optimizer steps, architecture changes, or D2 consumption before the subsystem architecture freeze |
| Maintenance | **SERVICE ONLY** | Keep source binding, custody, storage, tests, and evidence replay honest | May remove a real proof-track blocker; may not become a separate research program |

Training the model before resolving the interfaces and causal value of its
intended subsystems would bake untested assumptions into an expensive
experiment. The neural state therefore remains preserved but inactive.

## Single Active-State Contract

The machine-readable authority is
`configs/roadmap_implementation_matrix.json` at
`research_program_recenter.active_claim`. Human-facing documents summarize
that record and may not independently redefine it.

| Field | Current value |
| --- | --- |
| Claim | `virtual-context-abi.core` |
| Subsystem | `virtual_context_abi` |
| Phase | `K2_EVALUATOR_INSTRUMENT_QUALIFICATION` |
| State | `VCM_V3_K2_05_REPLACEMENT_EVALUATOR_QUALIFICATION_ACTIVE` |
| Selected task | 26 |
| Active attempt | `k2_05_three_row_replacement_evaluator_qualification_v1` |
| Current wall | `host_adequate_replacement_sources_and_full_closures_green_but_tasks_12_13_35_dependency_locks_environments_and_common_evaluators_not_yet_qualified` |
| Last closed task | 26 |
| Next legal action | `prospectively_bind_one_generic_three_row_resolution_transaction_for_tasks_12_13_35_then_materialize_environments_and_common_evaluators_before_reusing_qualified_16_25_56_without_rerun` |

The exact Semantic-IR implementation is terminal
`INCONCLUSIVE_EXPERIMENT` and frozen for the current TMax/host block. Its
prompt-ingest residual selected governed model-visible context materialization
as the next causal target. That is the reason VCM is active.

The frozen VCM source panel contains exactly 62 tasks: nine control-
qualification tasks and 53 claim tasks. No task, repository slot, task family,
or evaluator semantic expansion is authorized. A slot may be replaced only
after a predeclared invalidation, preserving language, source-disjointness, and
the frozen rank rule.

## Deep-Review Findings That Change Execution

The subsystem-first direction is correct, but the implementation path had begun
to optimize for producing evidence artifacts rather than reaching a causal
decision. The review therefore preserves the VCM panel and completed closure
evidence while changing how all forward work is organized.

| Finding | Evidence observed at the review boundary | Roadmap correction |
| --- | --- | --- |
| Book semantics remain stable | The experiment is pinned to ASI Stack commit `17c6ece80f771d3bce5f89c6b85c99ca9b6c2ea0`; the clean live book was observed at `61c74d65a2f47aeb34b25ddfb4d5348ef608a303`. The VCM core claim and `argument` support state are unchanged; live changes are formal refinement and proof growth | Keep the exact claim identity. Do not invent a broader or easier VCM claim, and do not promote book support automatically |
| Instrument work is dominating decision work | From the VCM claim-binding boundary through commit `5e4b2b38`, the effort spans 82 commits, 509 changed files, roughly 137,000 insertions, and 59 transaction reports before a claim-bearing VCM model call | Task 26 ends bespoke per-task owners. One generic builder, one row schema, one compact report, and one test family own the rest |
| Host resources are a scientific feasibility variable | At review, `reports/` was about 3.7 GiB, `runtime/` about 43 GiB, `runtime/vcm_evaluator/` about 2.3 GiB with 11,115 files, checkpoints about 9 GiB, and the host volume was 97% used. The refreshed resource policy saw about 4.4 GiB available memory, selected `conservative`, and allows one heavy closure at a time. The frozen schedule contains 32,290 locked package entries | Require shared content-addressed package stores, disposable installed environments, projected peak-space/memory/host-reserve preflight, serialized heavy work, and a bounded stop before bulk closure work |
| The current candidate protocol contains an answer-leak risk | The frozen v1 claim instrument lists `allowed_effect_paths` as candidate-visible even though those paths originate in target-diff source selection | Remove target-derived path restrictions before packet materialization. Give every arm the same broad disposable parent write root and audit every visible byte independently |
| The task panel is source-bound but no natural-task VCM materializer exists yet | The repository has VCM schema, synthetic mechanics, and a call-free claim contract, but no owner currently turns each external parent archive into the real task-specific VCM store and governed packet | K2 now requires a parent-only store and production VCM materializer before any campaign freeze |
| Existing baseline names are not implementations | Ordinary retrieval, summary/compression, full-context, graph, persistent-memory, and human-curated comparators are not all production-qualified campaign routes | Implement only the minimum causal screen now; treat missing mature book comparators as a cap on inference, then qualify extended routes only for a survivor |
| A target-derived oracle would violate blindness | A “source-bound single-memory oracle” selected from the hidden target would reveal answer location | Forbid target-derived oracle controls. Use parent-only no-context, flat, ordinary-retrieval, summary/compression, full-parent, and VCM routes |
| The nine-task panel cannot establish the claim | It was sized for control qualification, not a VCM treatment estimate | Use it only to test model/task/evaluator/host adequacy and freeze the strongest eligible control; reserve causal inference for the powered 53-task panel |
| The reference-control contract is bound to the wrong claim and transport | `configs/theseus_external_reference_control.json` still names `cognitive-compilation-and-semantic-ir.core`, describes an API adapter, and authorizes zero calls | Rebind it prospectively to the exact VCM task/evaluator/arm contract through a demonstrably Codex-subscription-backed route with zero billable API inference, or omit Luna permanently for this campaign |
| “Accuracy gain or retire” is too coarse | The book claim includes typed governance, lineage, omission, authority, freshness, revocation, fault, cost, and residual custody—not just retrieval usefulness | Issue separate conformance, integrity, model-use, economics, and transfer findings; allow a narrow governed-transport survivor when governance value is real but retrieval superiority is not |

These observations are an audit snapshot, not a standing resource allowance.
Every execution owner must remeasure actual host capacity and bind its own
resource ceiling before mutation.

## Exact Book Claim And Evidence Layers

The VCM claim is not “retrieval usually helps.” It is that context should cross
a static, typed, consumer- and purpose-relative request-to-materialization
boundary that accounts for source and field lineage, transformations, omissions
and loss, provenance and taint, authority and permitted use, freshness and
revocation, selection and unresolved frontier, adequacy, cost, faults, and
residual custody.

Admission across that boundary does **not** by itself imply that the material is
true, sufficient, useful, safe, used by the model, or supportive of the book
claim. The campaign must therefore report five layers independently:

| Layer | Question | Minimum evidence | What it cannot prove alone |
| --- | --- | --- | --- |
| L0 — Conformance | Did the exact request produce the exact packet with complete lineage, transformations, omissions, authority, lease, cost, faults, and residuals? | Deterministic source-to-packet replay plus producer-independent audit | Integrity under attack or model usefulness |
| L1 — Integrity | Do stale, revoked, tainted, wrong-scope, conflicting, poisoned, cache-invalid, mandatory-miss, and omitted-frontier cases produce the declared typed behavior without silent authority widening? | Mutation tests, hard-fault gates, and independent intervention audit | That a correct packet helps the model |
| L2 — Model use and utility | Does frozen TMax use the governed packet and improve useful-safe completion beyond a strong simple route, with the predicted ablation signature? | Blind paired local campaign and mechanism interventions | Transfer or acceptable economics |
| L3 — Economics | Is any governance or utility benefit worth tokens, retrieval, transformation, latency, memory, storage, verifier, repair, privacy, and maintenance cost? | Complete total-system ledger and frozen break-even rule | Generalization to another model or backend |
| L4 — Transfer | Does a survivor reproduce on fresh sources and transfer across a second memory backend and separately denominated second model? | One fresh D1 qualification, alternate-backend replay, prospective OpenAI reference, and eventually external reproduction | Automatic book-support promotion |

K3 can establish at most L0-L3 for the exact TMax, task population, VCM
implementation, controls, evaluator, host, and cost regime. Full interface
selection requires L4. Book support remains `argument` until the book's own
claim-review workflow changes it.

## Experimental Information-Flow Boundary

For every local or reference arm, the candidate-visible world is limited to:

- the natural-language request;
- a callable signature only when it is intrinsic to that request;
- a disposable copy of the exact parent snapshot with the same broad write
  root; and
- the arm-specific context produced from that parent snapshot under a frozen
  parent-only route policy.

Generation, retrieval, ranking, compression, and packet selection may not use
the target snapshot, target patch, changed-file list, selected source paths,
selected verifier paths, hidden tests, repository or pull-request identity,
source-task ID, expected answer, or reference output. In particular,
K2.04 removed target-derived `allowed_effect_paths` before any natural-task
candidate was materialized; the frozen successor contract requires the common
broad root `repository` instead.

Every arm receives the same request, parent store, output patch ABI, tools,
effect root, and verifier opportunity. Semantic IR and every other optional
subsystem remain disabled. A separate audit recomputes the visible packet and
selector inputs; it does not trust candidate-emitted flags. Any violation is
`INVALID_INFORMATION_FLOW` and cannot be repaired by relabeling the report.

## Scientific Contract

### Hold the model fixed

The current local instrument is
`mlx-community/Tmax-9B-MLX-8bit@33812d6cf04f88856f25eb828de4f3144a194560`.
Its tokenizer, chat template, decoder, completion policy, and runtime remain
fixed within the VCM block. A successor model requires a prospective,
claim-task-disjoint bakeoff and a new instrument identity; denominators may not
be pooled.

`gpt-5.6-luna` at fixed `xhigh` effort is a separately denominated,
measurement-only OpenAI reference. The existing transport contract is still
bound to Semantic IR and describes an API adapter, so it grants VCM zero-call
authority. A VCM-specific reference must bind the exact tasks, routes,
candidate-visible packet, independent evaluator, call count, custody, and a
demonstrably Codex-subscription-backed access receipt before any local claim
outcome exists, or be omitted permanently for this campaign. Billable API
inference is forbidden; an unverifiable subscription route means omission.
Execution may occur only after the corresponding local candidate cells are sealed. Luna
outputs may not select tasks or controls, tune the local system, train the
local model, serve a user, or enter the local denominator. Public-task
contamination remains unknown and is reported as a model non-equivalence.

The model is the controlled instrument, not the treatment. A subsystem must
change real execution: admitted context, dependency availability, tool or route
eligibility, verification allocation, repair scope, authority, or rollback.
Labels, prompt decoration, rendered plans, and reports are not mechanisms.

### Test one causal variable

Before any claim-bearing call, bind:

- exact book claim and immutable claim hash;
- causal mechanism and faithful production implementation;
- primary outcome and minimum worthwhile effect;
- competence, mechanics, and intervention requirements;
- strongest matched controls and ablations;
- candidate-visible and hidden-evaluator information;
- frozen task population, source-disjointness, and power method;
- safety, rollback, weak-tail, latency, verifier, token, and total-cost
  accounting;
- terminal states and maximum positive and negative inference; and
- actions for a positive, adequate null, implementation failure, experiment
  failure, or invalid observation.

Do not test an omnibus “full stack” treatment when the result cannot be
attributed. Common safety containment remains fixed in every arm; optional
mechanisms change one at a time.

### Separate mechanics, adequacy, and causal evidence

A mechanism first passes a non-claim production-path mechanics bench. It then
passes an evaluator and instrument adequacy stage. Neither earns usefulness or
book-support credit.

No claim denominator opens until the production renderer, tokenizer,
model-visible protocol, transport, parser, applier, dependency environment,
verifier, interventions, and repair path pass together. A failed mechanics
floor remains `INCONCLUSIVE_IMPLEMENTATION`. An invalid, underpowered, or
operationally blocked campaign remains `INCONCLUSIVE_EXPERIMENT` or invalid.
Neither falsifies the broader book mechanism.

### Preserve blind information flow

Generation and ranking see only the natural request, callable signature,
parent/source snapshot, and explicitly admitted runtime context. They may not
see later patches, hidden tests, source-task identities, answers, labels,
target-derived families, or answer-identifying decoder fields. A separate
audit recomputes candidate integrity and route blindness.

### Do not manufacture quality with token scissors

Quality-bearing generation ends on a prospectively declared complete artifact
or model EOS. There is no project-selected generated-token quality cap. The
exact residual of the pinned model context is a physical addressability
boundary; touching it invalidates capability inference. Host-safety stops are
reported separately and cannot be scored as mechanism failure.

Record actual prompt, reasoning, and generated tokens, wall time, verifier
work, retries, memory, storage, and money. Equality is obtained through matched
opportunity and information, not forced output truncation.

### Use precise independence claims

Use only these evidence labels:

| Label | Meaning |
| --- | --- |
| Role-separated rederivation | A separate owner or path recomputes the result inside this repository |
| Producer-independent replay | A closed bundle is replayed without mutable producer state |
| External reproduction | A different operator uses a separately provisioned environment |

Do not call repository-local role separation “fully independent.” Public
claims of full independence require external reproduction.

## Bound Claim Portfolio

The existing 13-claim portfolio is the decision boundary. The 84-chapter
crosswalk is coverage, not a mandate to build 84 lanes.

| Role | Claims | Program use |
| --- | --- | --- |
| Integrity prerequisites | `integrated-reference-architecture.core`; `evidence-states-and-claim-discipline.core` | Common evidence requirements, never a usefulness win |
| Architecture-shaping candidates | `cognitive-compilation-and-semantic-ir.core`; `planning-as-a-control-layer.core`; `virtual-context-abi.core`; `verification-bandwidth-and-context-adequacy.core`; `routing-heads-and-specialist-cores.core` | Must receive terminal interface dispositions before neural architecture freeze |
| Runtime amplifiers | `procedural-memory-and-cognitive-loop-closure.core`; `system-boundaries-and-authority.core`; `capability-replacement-and-rollback.core` | Enter only when an observed residual activates them |
| Independent neural claim | `replaceable-cognitive-substrates-beyond-transformer-monoculture.core` | Held D2 modular-versus-dense experiment |
| Synthesis | `the-efficient-asi-hypothesis.core`; `asi-is-a-stack-not-a-model.core` | Require multiple qualified mechanisms and later composition evidence |

Exactly one candidate is active. Each terminal result selects the next claim by
observed residual and architecture decision value, never by chapter order or
implementation convenience.

The architecture freeze does not require every scientific question to be
solved forever. It requires one seed-facing engineering disposition for each
architecture-shaping interface:

| Disposition | Meaning for the neural seed | Scientific meaning |
| --- | --- | --- |
| `SELECT` | Include the qualified full interface | Exact implementation survived its required evidence layers |
| `SELECT_NARROW` | Include only the bounded sub-interface that survived | Broader claim remains unresolved |
| `EXCLUDE_CURRENT_SEED_INCONCLUSIVE` | Do not build the current approximation into this seed | Cost, host, implementation, or experimental adequacy was insufficient; mechanism is not falsified |
| `RETIRE_EXACT_IMPLEMENTATION` | Exclude this exact implementation and regime | An adequate negative applies only to the tested implementation and regime |

After VCM, the default dependency order is verification bandwidth/context
adequacy, planning control, then routing/specialist cores. Procedural memory
enters only if an observed residual activates it. Semantic IR remains excluded
from the current seed unless a genuinely adequate successor receives a new
identity. An observed VCM result may change this order, but implementation
convenience may not.

## Phase Ladder

### K0 — Claim contract

Bind the claim, causal mechanism, information flow, outcomes, controls,
adequacy requirements, power, cost, terminal states, and maximum inference.

Exit: one immutable, machine-readable contract with no model calls.

### K1 — Production mechanics

Exercise the real route on non-claim fixtures. Prove identity round trips,
corruption rejection, intervention reachability, containment, replay, and
failure receipts.

Exit: the exact production path is mechanically operable; no usefulness
inference is authorized.

### K2 — Evaluator and experimental-instrument qualification — **CURRENT**

Finish the frozen 62-task VCM instrument without turning every task into a new
code family.

Task 26 was the final bespoke per-task dependency canary. Its exact sealed run
failed closed because locked `proxy-tools==0.1.0` has an sdist and no wheel
while source builds are forbidden. A separate static audit rederived the lock,
command, source identity, zero downstream authority, and scoped
`INCONCLUSIVE_INSTRUMENT_DEPENDENCY_POLICY_RISK_CLASS` disposition. This is not
task or VCM evidence. All remaining dependency closures now use one generic,
manifest-driven batch owner. Only four risk-class canaries remain eligible for
special handling: Bun, Yarn, TypeScript transpilation, and untrusted Rust
compilation. They extend the generic owner; they do not create per-task
script/config/test/report families.

The exact forward denominator is:

| Class | Frozen count | Current boundary |
| --- | ---: | --- |
| Tasks with project locks | 48 | 58 distinct parent/target closures because ten parent and target locks diverge |
| Closures GREEN before Task 26 | 6 | npm, pnpm, Cargo, and uv mechanics are represented |
| Task 26 uv closure | 1 | Terminal scoped instrument wall: sdist-only `proxy-tools==0.1.0` cannot satisfy the frozen wheel-only policy; role-separated audit GREEN |
| Locked closures after Task 26 | 51 | Must use the generic owner |
| Immutable-resolution tasks | 6 | Must receive a source-bound static/immutable execution classification |
| No-project-lock static tasks | 8 | Must receive an explicit static evaluator path; no fabricated dependency lock |

The generic owner must:

1. consume one frozen row manifest rather than task-specific constants;
2. verify source, parent, target, license, lock, toolchain, and evaluator hashes;
3. use shared content-addressed npm, pnpm, Yarn, Bun, Cargo, and uv stores while
   treating installed parent and target environments as disposable;
4. run installs, builds, and verifiers inside the existing bounded sandbox with
   network and untrusted-build behavior explicit;
5. emit one row-oriented report with content-bound stdout, stderr, exit,
   duration, storage, and failure receipts;
6. resume idempotently without treating partial success as panel admission; and
7. replay completed npm, pnpm, Cargo, and uv evidence before claiming the
   generic path is faithful.

K2.02 is now complete for this bounded boundary: the owner statically replays
the six already qualified closures as six content-bound rows across npm, pnpm,
Cargo, and uv, and freezes the manager-shared/disposable forward topology.
K2.03 then qualified the four named ecosystem risk classes through the generic
owner and role-separated audit. K2.04 is complete and K2.05 is current.

The first K2.03 transaction binds, at zero execution, the exact parent-only
representatives and order: Task 61 Bun install, Task 4 Yarn install,
Task 61 TypeScript `tsc --noEmit`, and Task 36 Rust 1.97.1 `cargo test --no-run`.
The rows execute serially under a 10 GiB reserve; their conservative temporary,
time, and RSS ceilings are host-safety boundaries rather than capability limits.
The v1 launch exposed a missing generic sandbox binding before any external
command and remains `INCONCLUSIVE_IMPLEMENTATION`. V2 bound that tool and ran:
Task 61 Bun acquisition and network-denied replay passed, one content-bound
shared Bun store was retained, and the 10 GiB reserve held. Task 4 Yarn then
failed before installation because transitive `jsdom@30.0.1` requires Node
`^22.22.2 || ^24.15.0 || >=26.0.0`, while the sealed runtime was 22.15.0.
TypeScript and Rust did not execute. This is a scoped toolchain-coverage gap,
not task or VCM evidence. Official Node 22.22.2 for Darwin ARM64 is now
project-local, verified against the published archive SHA-256, version-probed,
and bound by its extracted binary hash. V3 is prospectively sealed to verify
and replay the qualified Bun store offline without reacquisition, then run
Yarn, TypeScript, and Rust serially under the same reserve.

V3 stopped before any external command or network request because its generic
tree receipt included a macOS `.DS_Store` file written after V2 had recorded
the 8,320 package files. The dependency payload itself is unchanged. Preserve
V3 as `INCONCLUSIVE_IMPLEMENTATION`. V4 is prospectively sealed to exclude
only declared host metadata from package-content identity while still reporting
its presence, replay Bun offline without reacquisition, and then run the three
remaining risk classes.

V4 passed regular-file identity but failed before external execution because
the retained Bun cache contains 216 absolute symlinks to V2's deleted temporary
root; ordinary `copytree` followed them and failed. This makes the retained
topology itself non-replayable even though its package files are intact. V5 may
derive a disposable cache by preserving files and rebasing only links that are
proven to target an existing object inside that retained package payload. It
must leave retained evidence untouched and report every transformed link. V5
is now sealed with those exact checks and a unit-tested disposable-copy path.

V5 qualified that disposable Bun replay and exact Yarn online/offline replay,
retaining one shared Yarn store. The Task 61 TypeScript command returned 2
without a safety boundary, but the recorder preserved only its stdout hash, not
the 135-byte diagnostic, so its cause is unresolved. Rust did not run. V6 must
use only retained stores with zero network, retain the TypeScript stdout
diagnostic, and continue to the independent Rust canary even if TypeScript is a
non-safety failure.

V6 is now prospectively sealed with zero network and zero acquisition authority.
It verifies both retained stores, replays them offline, retains the TypeScript
stdout diagnostic, and continues to Rust after a non-safety TypeScript failure.

V6 requalified both retained stores offline, retained the exact TypeScript
diagnostic, and qualified Task 36 Rust compilation. The TypeScript failure is a
parent prerequisite gap: `framework/src/index-octane.ts` imports absent
`framework/src/styles.generated.ts`. A new seal may compile one real parent
TypeScript source that does not require generated content for mechanics-only
coverage, then K2.03 must undergo role-separated audit.

V7 is sealed to compile exact real parent file `framework/src/styles.ts` with
strict, no-emit TypeScript after zero-network Bun replay. It cannot qualify the
full repository. K2.03 remains open until this receipt and all prior risk
receipts are role-separately rederived.

V7 passed in 3.0 seconds at 195 MiB RSS with zero output, no boundary, unchanged
source, and unchanged retained Bun content. Bun, Yarn, narrow TypeScript, and
Rust therefore have scoped GREEN mechanics. Task 61 full-project typecheck
remains inconclusive. K2.03 now needs one role-separated rederivation of the
committed V2/V5/V6/V7 chain before it can close.

That audit is now GREEN. It rederived four committed attempts, all four scoped
risk classes, reserve and source/store invariants, zero downstream calls, and
the full-project TypeScript caveat. K2.03 is complete.

K2.04 is also complete. One archive-backed generic owner indexed four already
qualified parent snapshots without reading target archives, target diffs,
changed paths, verifier paths, source identities, hidden tests, or reference
outputs. It retained all 2,478 UTF-8 pages across 2,816 regular files as an
uncapped retrieval frontier, routed every request through the production VCM
consumer ABI, emitted information-identical governed, plain, maximal, and
ordinary-retrieval projections, and gave every arm the common broad disposable
write root `repository`. A separate owner rederived every archive inventory,
selector frontier, matched information identity, and all 16 candidate-visible
field byte receipts. Both reports are GREEN with zero candidate, model,
reference, evaluator, or runner calls. This is mechanics evidence only; context
adequacy and model usefulness remain untested. K2.05 is now active for the
remaining frozen closures, static execution classes, stores, packets, and
parent-fail/target-pass evaluator receipts through the one generic owner.

The first K2.05 call-free batch preflight validates the frozen coverage but
closes full-batch execution on current host storage. The 48 locked rows contain
32,290 package entries. Manager-specific upper coefficients derived from the
qualified npm, pnpm, Cargo, uv, Bun, and Yarn stores project 35.6 GiB of new
downloads, an 11.5 GiB largest serial disposable install, and 4 GiB temporary
space: 51,413,161,160 incremental bytes in total. Current free space minus the
10 GiB reserve provides 10,798,305,280 safe bytes, leaving a 40,614,855,880-byte
deficit. This deliberately conservative no-cross-lock-deduplication upper bound
is not expected spend and does not falsify a task, evaluator, VCM, or model.
No fetch, install, build, runner, evaluator, packet, or inference call occurred.
K2.05 now has a GREEN role-separated segment-plan audit. The compiler rejected
the historically misaligned v1 materialization rows as a new manifest source
and instead binds the authoritative v3 source panel, v2 parent closures, v3
runner inventory, dependency classes, and dependency schedule. It rederived
one target-free 62-row parent manifest and an 8 static / 6 immutable-resolution
/ 48 locked schedule with broad effect root `repository`, zero target-derived
selector fields, and panel admission withheld. No dependency, repository,
evaluator, packet, local-model, or reference call occurred. The next bounded
transaction has now materialized and role-separately rederived all 62
archive-backed parent stores: 133,048 regular files, 130,968 UTF-8 pages, and
248 candidate-visible field receipts, with no duplicated payload and zero
downstream calls. The static segment then qualified exact parent-fail/target-pass
behavior for Tasks 10, 22, and 27. Tasks 1, 20, 23, 57, and 58 are scoped
`INCONCLUSIVE_EXPERIMENT_STATIC_EVALUATOR_CONSTRUCT`; the uniform code-71 nested
sandbox attempt is separately preserved as an invalid launch wall. The five
constructs exposed the need for one common hidden evaluator. V3 now transplants
the exact target verifier into both parent and target, preserves archive file
modes, isolates Task 23's scoped test from unrelated conftest imports, and
retains complete diagnostics. All eight exact parents fail and targets pass;
the separate role audit is GREEN. The six immutable-resolution closures are now
6/6 GREEN under a separate role audit. Five sealed predecessor locks were
reused byte-for-byte; Task 13 resolved under Python 3.14.2 after exact static
sdist qualification and network-denied wheel builds for its sdist frontier.
The resolution phase installed no packages and ran no repository evaluator or
model. One generic, serial, reserve-safe owner has now materialized and
role-separately rederived all six common evaluator environments through a
1.75 GB shared uv/Cargo store with network-denied replay. The predecessor
matched-verifier campaign qualified Tasks 16, 25, and 56 and scoped Tasks 12,
13, and 35 to host-platform or implementation-toolchain walls. A sealed,
all-or-none replacement transaction then selected `plwp/chief-wiggum`,
`frame-consulting/QuantLibXlOil`, and `knoguchi/marsdb` for exactly those slots
under the original rank while preserving panel, programming language, and
source disjointness. Full manifests then exposed that the first Task 13
replacement was explicitly Windows/xloil-bound, so it was invalidated before
dependency execution. A sealed generic host-feasibility successor replaced
only that slot with `paulomtts/pyjinhx`. Source-panel v5 and all 124 exact full
parent/head closures are role-audited GREEN, with 61 closure pairs replayed
unchanged. No repository, dependency, evaluator, model, Luna, teacher, or
reference call occurred. Replacement source and host-static adequacy are
repaired, but locks, environments, and common-evaluator receipts remain open.
Tasks 16, 25, and 56 may not rerun, and the full locked batch remains closed by
the existing storage wall.

Before bulk materialization, the owner must measure projected download,
installed, temporary, and deduplicated-store bytes; projected wall time; host
free space and protected reserve; memory; and untrusted-build risk. If the
generic instrument cannot fit safely or qualify the risk classes in bounded
transactions, stop as `INCONCLUSIVE_IMPLEMENTATION` when the generic owner is
unfaithful or mechanically broken, or `INCONCLUSIVE_EXPERIMENT` when host,
storage, cost, time, or evaluator adequacy blocks a valid instrument. Preserve
the 62-task panel and design a simpler instrument under a new identity. Do not
fall back to another chain of per-task caches and owners.

K2 must establish for all 62 tasks:

- exact source, parent, target, license, and contamination identity;
- reproducible parent and target dependency closures;
- parent-fail/target-pass hidden-evaluator receipts;
- a VCM store constructed from the exact parent snapshot only;
- a production-path VCM request-to-packet materializer rather than a synthetic
  fixture or report renderer;
- candidate packet blindness and independent forbidden-field and selector-input
  audit;
- compatible sandbox, verifier, and effect boundaries;
- hard-fault rejection before model invocation and model-use intervention
  reachability;
- one common output patch ABI and broad disposable parent write root; and
- one contiguous frozen evaluator/instrument campaign identity.

After K2 freezes, evaluator semantics may change only for a predeclared
invalidation class and under a new campaign identity. Cosmetic report changes
do not reopen the experiment. Exact consumed surfaces may not be rerun.

### K3 — Matched causal VCM campaign

K3 has a call-free freeze, a nine-task adequacy screen, an exact reference
rebind-or-omit checkpoint, a powered local claim campaign, an optional
separately denominated reference execution, and a layered disposition.

#### K3.1 — Implement and freeze the routes before model calls

All current-campaign routes use the exact natural request and parent-only store:

| Route | Experimental role | Matching rule |
| --- | --- | --- |
| No added context | Competence floor | Same model, output ABI, tools, sandbox, verifier opportunity, and parent write root |
| Information-matched flat direct context | Mandatory mechanism ablation | Same parent-only information items and visible context opportunity as VCM, without typed governance or lifecycle machinery |
| Ordinary direct retrieval | Strong simple control candidate | Same parent store, request-derived query, retrieval opportunity, and visible context boundary |
| Hierarchical summary or prompt compression | Strong simple control candidate | Same parent store and visible context opportunity; transformation and loss are recorded |
| Maximal full-parent context | Addressability upper bound | Runs only if synthetic size-bucket canaries establish physical and host operability; a boundary hit is invalid/host evidence, not model failure |
| Governed VCM | Treatment | Real typed request, materialization, lineage, omission, authority, freshness, fault, cost, and residual path |

A target-derived oracle is forbidden. Graph retrieval, persistent memory, and
human-curated or external-context reproduction remain required by the book
before a full general interface claim, but building all three before the first
causal VCM decision would recreate the breadth problem. Their absence caps K3's
maximum inference; a K3 survivor qualifies those comparators in K4 under the
same claim identity. “Human-curated” means a prospectively frozen,
license-compatible independent public artifact or external reproduction—not a
request for Corben to provide context or judge an outcome.

Stale mandatory memory, wrong-scope memory, tainted or unauthorized memory,
revoked or expired memory, and mandatory-field/frontier miss must fail before a
model call. Irrelevant/random, shuffled, conflicting/poisoned,
retrieval-disabled, and retrieval-degraded interventions test whether the model
actually uses packet content. Their denominator and whether each is powered or
descriptive are frozen before outcomes.

The call-free freeze also binds candidate custody, exact tokenizer accounting,
complete-artifact/EOS completion, model and host watchdogs, counterbalanced arm
order, exact paired analysis, weak tails, false-context veto, total-cost ledger,
the flat-ablation attribution threshold and power, the narrow-interface
noninferiority margin and power, valid fixed-sequence or other multiplicity
control, the material family-harm boundary, and all terminal outcomes.
Programming-language tails are heterogeneity and harm checks, not separately
powered family claims. Synthetic size-bucket host canaries use no claim task. A
physical context or host-safety boundary makes that observation invalid for
capability inference.

The prospective core call shape is explicit:

| Stage | Unique core-route calls | Notes |
| --- | ---: | --- |
| Nine-task local screen | At most 54 | Nine tasks by six routes; fewer only when a route is prospectively declared physically inoperable, never because an outcome is inconvenient |
| 53-task local claim panel | 106 to 159 | Two unique arms when flat context is the selected strongest control; otherwise three |
| Optional Luna reference | 106 | VCM and frozen-control cells only, separately denominated |
| Hard-fault checks | Zero model calls | Must stop before inference |
| Model-use interventions | Prospectively frozen before outcomes | Additional denominator must state whether it is powered or descriptive |

If Luna is desired, this stage freezes a VCM reference envelope: common packet,
evaluator, custody, two-cell shape, maximum-cell rule, and the requirement to
bind or omit immediately after the local screen selects its control. Two
reference cells across 53 tasks define 106 sealed observations. The route must
prove that inference is covered by Corben's existing Codex subscription and
that billable API spend is zero. If that provenance cannot be established, the
reference is prospectively omitted. This is an access and information-flow
boundary, not an answer-length or quality cap.

#### K3.2 — Nine-task local adequacy and control screen

The nine tasks are not claim evidence. They answer whether the frozen model,
task construction, evaluator, route implementation, interventions, and host can
support the predeclared 0.35 minimum worthwhile effect. Luna is forbidden.

The screen must predeclare competence and futility from task opportunity,
known-positive evaluator reachability, exact paired power, and host operation;
it may not invent a conventional pass-count threshold. If the model cannot
complete enough tasks, the contexts do not contain usable parent-only evidence,
the evaluator is insensitive, or the host cannot run the routes, stop before
the 53-task panel as
`INCONCLUSIVE_EXPERIMENT_MODEL_TASK_HOST_OR_EVALUATOR_ADEQUACY`.

Among eligible non-VCM routes, select the strongest control by:

1. highest useful-safe completion;
2. then lower total-system cost; and
3. then greater parent-only information opportunity.

The rule and tie breaks freeze before the screen runs. Reference output, hidden
target structure, and implementation preference never enter the selection.

Immediately after the control is frozen—and before the first 53-task candidate
call—the OpenAI contract must either bind the exact VCM claim, task and evaluator
hashes, VCM and selected-control routes, 106-cell maximum, custody, and a
Codex-subscription-backed zero-API-spend receipt, or record
`MISSING_PROSPECTIVE_REFERENCE`. This checkpoint exists after the
screen because the control identity was previously unknown, and before the
claim panel because outcome-dependent backfill is forbidden.

#### K3.3 — Powered 53-task local claim campaign

Run up to three unique local arms per task:

1. governed VCM;
2. the frozen strongest simple control; and
3. information-matched flat direct context as the mandatory mechanism ablation.

If the strongest control is already the flat arm, do not duplicate it. Seal
every candidate artifact before the independent hidden evaluator runs.
Temperature-zero TMax makes the task the statistical unit; repeated deterministic
seeds are not independent observations. Counterbalance route order across tasks
to expose cache, thermal, and host-order effects. Do not rerun an exact consumed
task surface to repair a loss.

The confirmatory order is frozen. First test VCM superiority by the 0.35
minimum worthwhile effect against the strongest control. Only under the sealed
multiplicity procedure may the VCM-versus-flat mechanism-attribution hypothesis
and, if full selection is not established, narrow noninferiority be interpreted.
If flat context is the strongest control, deduplicate the hypotheses as well as
the calls. The fixed 53-task panel may not be expanded when a secondary or
narrow gate is underpowered; the maximum inference is reduced instead.

VCM minus information-matched flat context attributes any supported difference
to typed governance and selection rather than merely receiving useful parent
information. Report exact paired uncertainty, all four programming-language
tails, unsafe or invalid outcomes, prompt/reasoning/generated tokens, latency,
retrieval and transformation work, verifier and repair work, memory, storage,
and total cost.

#### K3.4 — Optional Luna measurement reference

If and only if its VCM-specific contract was sealed before local claim outcomes,
run the corresponding VCM and frozen-control cells after the local candidates
are immutable. Use the same candidate-visible packet and independent evaluator.
The route must be demonstrably covered by Corben's Codex subscription; billable
API inference and API-price fallback are forbidden. Report provider/model/effort,
access provenance, wrapper, tokenization and reasoning differences, public-task
contamination uncertainty, actual usage, latency, and zero API spend.

Luna is neither a local denominator nor an oracle. Its outputs cannot select or
tune tasks, controls, prompts, packets, routes, evaluators, or the local model;
cannot become training rows; cannot receive source-effect credit; and cannot be
backfilled if it was not prospectively bound.

#### K3.5 — Layered decision

The prospective VCM decision rule is:

- minimum useful absolute effect: 0.35;
- one-sided alpha: 0.05;
- minimum power: 0.80;
- false-context acceptance: zero;
- positive direction in at least three of four programming-language families,
  with no material family harm;
- unsafe release or material weak-tail harm vetoes promotion; and
- VCM must remain nondominated after actual tokens, retrieval work, latency,
  memory, storage, verifier/repair work, and total cost.

Task-deadline and cost-break-even boundaries are derived from the control panel
and frozen before claim-panel outcomes are visible. L0-L3 are reported
separately, then K3 issues exactly one disposition:

| Disposition | Required evidence | Maximum inference |
| --- | --- | --- |
| `ADVANCE_FULL_INTERFACE_CANDIDATE_TO_K4` | L0/L1 GREEN; powered L2 effect at or above 0.35 against the strongest control; predicted flat-ablation signature; acceptable L3 cost; no safety or weak-tail veto | Full VCM candidate for transfer, not yet selected and not book support |
| `ADVANCE_NARROW_GOVERNED_TRANSPORT_CANDIDATE_TO_K4` | L0/L1 GREEN; prospectively powered noninferiority to information-matched flat context; acceptable L3 cost | Typed governance/transport candidate only; no retrieval-superiority claim |
| `REVISE_NEW_CAMPAIGN_IDENTITY` | A specific repair is justified but changes claim implementation, evaluator, population, or causal contract | Old campaign remains closed; no pooled denominator |
| `RETIRE_EXACT_IMPLEMENTATION_AND_REGIME` | Faithful implementation and every adequacy requirement pass, followed by an adequate negative | Only the exact implementation, data, model, controls, evaluator, budget, host, and regime are retired |
| `INCONCLUSIVE_IMPLEMENTATION` | Faithfulness, learning/mechanics, intervention, or production-path adequacy fails | Repair or exclude this approximation; no mechanism negative |
| `INCONCLUSIVE_EXPERIMENT` | Power, task opportunity, evaluator validity, source independence, host, or cost adequacy fails | Preserve evidence and repair the experiment; no mechanism negative |
| `INVALID_INFORMATION_FLOW` | Candidate or selector sees answer-identifying information | No claim inference and no relabeling repair |

### K4 — Fresh D1 survivor qualification

Only a K3 survivor may consume D1, once, on fresh source-disjoint tasks. D1 is
qualification, not another tuning loop. Failure returns the exact interface to
its predeclared disposition; it does not trigger repeated fresh-pool search.

K4 completes L4 by qualifying the survivor on a second memory backend and,
when prospectively bound, the separately denominated OpenAI model. A full-
interface candidate must also close the graph, persistent-memory, and
human-curated/external comparator obligations required for the broader book
interface claim. A narrow governed-transport candidate transfers only its
declared narrow contract and makes no broader retrieval claim. A comparator
that is not mature caps the inference rather than being represented by a toy
stand-in.

K4 ends with exactly one neural-seed engineering disposition: `SELECT`,
`SELECT_NARROW`, `EXCLUDE_CURRENT_SEED_INCONCLUSIVE`, or
`RETIRE_EXACT_IMPLEMENTATION`.

### K5 — Claim-level book handoff

Produce one compact subsystem disposition memo containing:

- exact claim, implementation, model, tasks, evaluator, controls, and costs;
- causal result, uncertainty, language-family tails, safety, and residuals;
- mechanics and adequacy status;
- maximum positive and negative inference;
- integration, maintenance, latency, memory, storage, and failure-mode cost;
- terminal interface decision and next residual; and
- stable links to replayable evidence.

The handoff leaves the book support state unchanged. Book-side claim review is
the only authority that may promote support.

## Evidence Proportionality And Transaction Discipline

Evidence cost is a first-class experimental outcome, not progress by itself.
Every bounded closure records:

- source files changed;
- generated artifact count and bytes per sample;
- evaluator wall time;
- local and reference cost;
- manual or agent interventions;
- commits per closure;
- role-separated evidence fraction;
- shared-store deduplicated bytes;
- projected and actual peak host bytes plus protected reserve; and
- evidence work per claim-bearing model observation.

The default transaction limit is one prospective-seal commit and one final
audited-closure commit per bounded closure. Do not create per-seed,
per-condition, or per-task commits. Do not create a new dashboard or report
family when the canonical matrix, generic owner, and compact disposition can
carry the evidence. Building the generic K2 owner is itself limited to one
prospective builder transaction and one audited qualification transaction.

Keep only the evidence needed to establish identity, replay the causal path,
audit information flow, reconstruct the decision, or satisfy legal custody.
Deduplicate immutable packages and generated environments through governed
retention owners; do not ad hoc-delete negative or superseded evidence. If a
new artifact does not alter a predeclared gate, disposition, cost estimate, or
maximum inference, it is not authorized merely because it is easy to report.

A report is evidence, not progress. Progress means a book claim receives
stronger causal evidence, an inadequate owner is repaired, a subsystem survives
a fair comparison, or the harness becomes materially more honest or cheaper.

## Neural Hold And Re-entry

The neural seed remains at step 11,992, `NOT_EVALUATED`, with D2 sealed. It may
resume only after all of the following are machine-readable and green:

1. architecture-shaping subsystem interfaces have terminal seed-facing
   dispositions—selection, narrow selection, current-seed exclusion as
   inconclusive, or exact-implementation retirement—without pretending every
   scientific question is permanently solved;
2. selected interfaces and common safety containment are frozen;
3. autonomous source, custody, contamination, and teacher-share gates pass;
4. modular, dense-active, and dense-total controls receive matched raw data,
   optimizer exposure, tuning opportunity, and total-system accounting;
5. required pretraining and checkpoint migration/rebind work is complete; and
6. D2 remains unconsumed until the matched candidates are ready.

The modular-versus-dense result is independent D2 evidence. Subsystem proof
cannot substitute for neural training, and neural training cannot retroactively
prove a subsystem.

## Maintenance Boundary

Maintenance may consolidate scripts and documents, repair registry or custody
gaps, reduce generated storage safely, or restore reproducibility required by
the active proof. It may not create new product surfaces, benchmark families,
private task ecologies, dashboards, or research lanes.

Do not delete historical evidence merely because it is negative, superseded,
or bulky. Keep authoritative summaries and replay-critical artifacts; handle
retention through governed custody rather than ad hoc cleanup.

## Immediate Execution Order

1. Preserve the completed Task 26 result: its exact wheel-only canary failed on
   sdist-only `proxy-tools==0.1.0`, and the role-separated audit scoped it to
   `INCONCLUSIVE_INSTRUMENT`. Do not weaken the policy or add another per-task owner.
2. Preserve the GREEN generic static replay of six existing closures; it freezes
   the shared-store/disposable-environment schema but grants no execution authority.
   npm, pnpm, Cargo, and uv closure evidence through its common row schema.
3. Preserve the role-separated K2.03 qualification of Bun, Yarn, narrow real-parent
   TypeScript mechanics, and untrusted Rust, including the full-project
   TypeScript caveat.
4. **COMPLETE:** Build and independently audit the real parent-only VCM store and
   request-to-packet materializer on already qualified rows. Remove all
   target-derived effect paths and selector inputs before any task packet exists.
5. **CURRENT — THREE-ROW ADEQUACY REPAIR:** Preserve the GREEN eight-row
   static segment, six immutable locks, six common environments, and qualified
   Tasks 16, 25, and 56. Prospectively bind one new campaign for only the Task
   12 host-platform, Task 13 dependency, and Task 35 Ruby-toolchain walls;
   repair faithfully or invalidation-replace under the frozen rank rule. Never
   rerun the three qualified surfaces or admit a partial panel. The 48-row
   locked segment remains separately closed by the 40.6 GiB reserve-safe
   storage deficit until a bounded acquisition plan fits.
6. Freeze one contiguous K2 source/evaluator/store/packet/sandbox/output/
   intervention/cost identity and complete producer-independent replay plus the
   role-separated blindness audit.
7. Implement and call-free audit the six current local routes, host canaries,
   competence/futility rules, exact analysis, counterbalancing, cost ledger, and
   terminal decisions. Freeze the optional VCM reference envelope.
8. Run the nine-task local-only adequacy/control screen. Stop as scoped
   inconclusive if the model, tasks, evaluator, interventions, or host cannot
   expose the useful effect; otherwise freeze the strongest eligible control,
   then bind Luna through a demonstrably Codex-subscription-backed zero-API-spend
   route to that exact VCM/control campaign or prospectively omit it.
9. Run the 53-task local VCM, strongest-control, and information-matched-flat
   campaign, deduplicating the flat arm if it is already the selected control.
   Seal candidates before hidden scoring and never rerun a consumed surface.
10. If prospectively authorized through the sealed Codex-subscription route,
    run the separately denominated Luna VCM and
    control cells. Then issue separate L0, L1, L2, and L3 findings and one K3
    terminal disposition.
11. Let one survivor consume fresh D1 once, qualify alternate-backend and
    remaining mature-comparator transfer obligations, and issue `SELECT`,
    `SELECT_NARROW`, `EXCLUDE_CURRENT_SEED_INCONCLUSIVE`, or
    `RETIRE_EXACT_IMPLEMENTATION`.
12. Return one compact L0-L4 claim packet to the book with support unchanged,
    freeze the VCM seed interface, and activate exactly one next subsystem from
    the observed residual and dependency order.

No item in this sequence requires Corben to supply tasks, labels, routine
approval, or timing. Actions outside machine-readable bounded authority fail
closed and record the wall.

## Explicitly Sidelined

- personal-assistant product work;
- user-supplied task collection;
- omnibus full-stack A/B claims;
- exact consumed-surface replay or public benchmark training;
- candidate-visible target-derived effect paths or target-selected oracles;
- arbitrary generated-token quality caps;
- per-task dependency, evaluator, packet, test, cache, report, or commit families
  after Task 26;
- new private task-suite families and evidence dashboards;
- neural optimizer steps or D2 consumption before re-entry;
- externally served reference outputs;
- automated book-support promotion; and
- broad architecture rejection from inadequate proxies.

## Definition Of Done For This Roadmap Era

This era is complete when:

1. every architecture-shaping interface has a terminal, evidence-scoped
   seed-facing disposition, including explicit current-seed exclusion when the
   scientific result remains inconclusive;
2. survivors have one fresh D1 qualification and compact costed memo;
3. the subsystem architecture freeze is machine-readable;
4. the neural seed re-entry gates are either green or have plain unresolved
   walls; and
5. claim-level evidence has returned to the book without automatic promotion.

## Canonical Verification

```bash
python3 -m json.tool configs/roadmap_implementation_matrix.json
python3 scripts/theseus_doc_link_audit.py
python3 scripts/theseus_project_registry.py --gate
python3 scripts/roadmap_implementation_gate.py --gate
python3 scripts/theseus_asi_stack_claim_handoff.py
python3 -m pytest -q tests/test_active_status_sync.py \
  tests/test_roadmap_book_sync.py \
  tests/test_roadmap_pretraining_gate.py \
  tests/test_theseus_external_reference_control.py \
  tests/test_theseus_vcm_claim_instrument.py \
  tests/test_theseus_vcm_task26_dependency_canary.py
```
