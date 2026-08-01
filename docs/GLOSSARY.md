# Project Theseus Glossary

This glossary gives plain-English meanings for project terms. When a document
uses a term differently, this page and the owning machine contract should be
reconciled.

## Project And Model

**Project Theseus**
The whole research system: model training, evaluation, data governance,
runtime, tools, memory, evidence, and local product surfaces.

**Theseus student**
The locally trained learned model. Tools, retrieval, rules, templates, and
teacher calls are not part of its model-only capability.

**Neural seed**
The first bounded model intended to establish useful learned behavior in
English and the supported programming-language arms. It is a research seed,
not a foundation-model claim.

**MoECOT**
The modular candidate with a shared transformer trunk and independently
trained specialist language arms. Its value must be established against
matched dense controls.

**Dense control**
A conventional transformer comparison receiving matched data, compute, tuning
opportunity, evaluation, and cost accounting. Theseus has active-parameter and
total-parameter matched controls.

**Arm**
A specialist learned output path, currently scoped to English, Python,
JavaScript/TypeScript, HTML/CSS, or Rust.

**Shared trunk**
The model layers shared by all specialist arms.

**Octopus router**
The governed route-selection layer that can select one or more registered
specialists. Routing success and answer success are measured separately.

## Architecture

**VIEA — Verified Intent-To-Execution Architecture**
The contract that turns an intent into a bounded plan, route, execution,
observation, verification result, and feedback record.

**SCF — Stable Capability Field**
A stable interface for a capability whose implementation may be replaced only
with explicit authority, evidence, migration, rollback, and lifecycle records.

**VCM — Virtual Context Memory**
Governed context and memory packets with source references, leases, taints,
copy-on-write behavior, and deletion/lifecycle evidence.

**CGS — Compact Generative Systems**
The project’s framing for compact state, rules, memory, residuals, verification
cost, governance cost, and generative leverage.

**SymLiquid**
The protected discovery substrate combining liquid/recurrent state,
vector-symbolic memory, compact generative accounting, and related mechanisms.
It is not the selected practical model by default.

**KERC**
The Kernel English / relational compiler discovery path for learned structured
representation and generation. It is currently frozen out of campaign one as
`INCONCLUSIVE_EXPERIMENT`.

**RDC — Relational Dimension Compiler**
The typed relational intermediate representation associated with the KERC
research path. It may inform shared semantic contracts but cannot become an
unreviewed parallel ontology.

**Cognitive Loop Closure**
The process of turning repeated, verified trajectories into bounded
parameterized tools while retaining provenance, negative space, and rollback.

**Procedural memory**
Verified reusable procedures derived from repeated traces. Deterministic
procedure success is assisted behavior, not learned-generation credit.

## Training And Data

**Optimizer step**
One parameter update.

**Optimizer position**
A supervised token/position contribution used for exact campaign accounting.
Positions are more comparable than nominal steps when batch widths vary.

**Checkpoint transaction**
An atomic model, optimizer, RNG, cursor, receipt, and lineage update. Disk
launch safety reserves space for two complete measured transactions.

**Lineage**
The append-only chain linking checkpoint state, configuration, plan, data
cursor, execution receipt, and predecessor identity.

**Prospective lineage anchor**
The first state from which a complete forward chain is retained. It does not
reconstruct missing history before that point.

**Candidate lease**
A bounded authorization naming the exact implementation, target, budget,
namespace, seed, data, checks, resource limits, and claim boundaries for one
candidate run.

**Frozen corpus**
The admitted training corpus is held unchanged so architecture and control
comparisons remain interpretable. It changes only after a measured defect and a
new governed admission transaction.

**Teacher row**
A retained, provenance-tagged training row generated through the governed
OpenAI teacher path. Teacher rows are capped residual pressure and are never
served directly.

**Static third-party corpus**
Published training data admitted through license, provenance, quality,
deduplication, contamination, privacy, retention, and synthetic-share checks.
It does not authorize live use of the originating provider.

**Teacher share of accepted training rows**
Teacher-accepted rows divided by all accepted training rows. The durable ledger
tracks this value and its content-bound inputs.

## Evaluation And Claims

**Source-disjoint**
Evaluation sources are separated from training sources so results do not rely
on prompt or source reuse.

**Blind information flow**
Generation and ranking see only the natural-language prompt, callable
signature, and explicitly allowed runtime context—not hidden tests, answers,
labels, or answer-derived metadata.

**Candidate integrity**
Independent recomputation of whether an output is learned, admissible,
non-template, contamination-clean, and eligible for a claim. Candidate-emitted
integrity flags are not trusted.

**Model-only**
The learned model produced the result without tool, template, retrieval,
router, fallback, or deterministic-repair credit.

**Assisted**
One or more non-model mechanisms contributed. Assisted results can be useful
but must be reported separately.

**Functional utility**
Whether the model completes source-disjoint natural-language or programming
tasks under the frozen verifier and autonomous independent-machine-audit
contract.

**Weakest-arm utility**
The least successful supported language arm. An average cannot hide a failed
arm.

**Consumption surface**
A frozen evaluation set that may be used only through its registry. Exact
consumed surfaces are not rerun.

**Public calibration**
Governed measurement on a public benchmark. Public payloads never become
training data.

**Mechanics evidence**
Evidence that code runs, gradients flow, state reloads, or parity holds. It is
not evidence of useful behavior.

**Capability evidence**
Source-disjoint behavior under an integrity-valid evaluator with appropriate
controls and attribution.

**Adequacy**
Whether an experiment faithfully implements the mechanism, learns, uses strong
matched controls, has sufficient diversity/seeds/power, and uses independent
heldouts and evaluators.

## State Labels

**`CUSTODY_GREEN`**
Exact state and lineage are preserved and replayable. No capability is implied.

**`TRAINING_HELD`**
No new long training segment may launch.

**`TRAINING_READY`**
Every current launch gate passes for one exact route. Operator approval is
still required.

**`NOT_EVALUATED`**
The capability surface has not been consumed.

**`INCONCLUSIVE_IMPLEMENTATION`**
The proxy or implementation does not faithfully exercise the claimed
mechanism.

**`INCONCLUSIVE_EXPERIMENT`**
The experiment is underpowered, undertrained, poorly controlled, or otherwise
insufficient for the broader claim.

**`LOCAL_ONLY`**
The surface is qualified only for local loopback use.

**`ASSISTED_ONLY`**
The result includes non-model mechanisms and cannot support learned-model
credit.

**`EMPIRICAL_SUPPORT_INSUFFICIENT`**
Available observations are synthetic, sparse, or not representative enough to
establish real usefulness.

**`FROZEN`**
The lane or artifact is preserved but inactive until a named re-entry
condition is met.

**`GREEN`**
Allowed only with a named dimension, such as registry GREEN, custody GREEN, or
security-test GREEN. Bare GREEN is ambiguous and should be repaired.

**`YELLOW`**
A warning or partial state requiring interpretation. It is neither failure nor
permission.

**`RED`**
A named gate failed. Repair the owning condition; do not relabel or create an
adjacent green artifact.

## Runtime And Governance

**SparkStream**
The local automation and observability layer for bounded jobs, reports, goals,
and resource-aware scheduling.

**Theseus Hive**
The registered distributed-work runtime. It is not a remote shell and remains
loopback/private by default.

**Authority surface**
Any API, dashboard, CLI, scheduler, or worker route that can mutate state,
invoke a teacher/network, execute work, or expose data.

**Request-local authority**
Teacher, network, mutation, or execution permission explicitly attached to one
request. Missing authority defaults false and cannot be inherited silently.

**Effect-complete receipt**
A record that distinguishes intended action, attempted effect, observed
effect, verification, failure, and rollback.

**Project registry**
The machine-readable inventory of canonical surfaces, implementations,
ownership, lifecycle, route evidence, and replacement relationships.

**Roadmap implementation matrix**
The machine-readable obligation set for every project phase.

**Breadth freeze**
The rule against creating new lanes, dashboards, products, benchmark families,
or document families unless they directly serve the current priorities.

**External inference**
Model inference outside the local Theseus serving model. It is allowed only as
a governed training teacher and never at runtime serving.
