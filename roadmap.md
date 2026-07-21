# Project Theseus Roadmap

Last consolidated: 2026-07-20 UTC.

This is the forward-only human execution map. The complete machine-readable obligation
set is `configs/roadmap_implementation_matrix.json`; implementation identity and route
authority live in `configs/project_manifest_registry.json`; the bounded current wall is
`docs/PROJECT_STATE.md`; operating rules live in `AGENTS.md`. Historical roadmap prose
remains in Git. A detail omitted here is not retired when it remains open in the matrix.

## North Star

Build a private, locally trained assistant that Corben uses daily, serves with zero
external inference, improves through governed evidence, and drives accepted live-teacher
share toward zero. The seed specializes in English, Python, JavaScript/TypeScript,
HTML/CSS, and Rust. It must become useful before it becomes broad.

The current falsifiable question is:

> Can a locally trained, source-disjoint Theseus student complete useful natural
> conversation and repository work through the governed stack, and does modular
> MoECOT admission improve the useful-safe-release frontier over matched dense controls
> at acceptable total lifecycle cost?

## Controlling Boundaries

1. Public benchmarks are calibration only. Never train on their prompts, tests, hidden
   tests, solutions, traces, labels, or answer templates.
2. Externally generated tokens are never served. Live teachers are OpenAI-only,
   provenance-bound, verifier-accepted, leakage-audited residual pressure.
3. Published third-party corpora require explicit training rights, provenance, quality,
   scope, deduplication, privacy, contamination, retention, and synthetic-share records.
4. Generation and ranking may see only the prompt, callable signature, and authorized
   runtime context. Hidden targets and answer-derived metadata are forbidden.
5. Templates, deterministic renderers, action catalogs, tools, routers, retrieval, and
   fallbacks are useful assisted mechanisms but never learned-generation credit.
6. A negative result applies only to the exact implementation, data, scale, objective,
   optimizer, budget, evaluator, and regime tested. Missing adequacy is
   `INCONCLUSIVE_IMPLEMENTATION` or `INCONCLUSIVE_EXPERIMENT`, never broad falsification.
7. New architecture is frozen unless a concrete correctness, security, information-flow,
   checkpoint, migration, replay, or measured hot-loop failure proves the current contract
   cannot express the needed behavior.
8. One canonical owner per abstraction. Improve or replace it transactionally; do not
   create adjacent versions, sidecars, loose lanes, or nearby green reports.

## Binding Execution Board

| Gate | State | Exit condition |
|---|---|---|
| `T0` Finite architecture closure | complete | The 70-artifact freeze package passes 10 independent replays. KERC K0-K3 are banked; K4-K8 are explicitly deferred with zero first-campaign exposure. OneCell and optional generation modes are likewise content-bound and excluded. |
| `T1` Frozen neural-seed campaign | checkpoint/resume qualified; sustained speed target open | The exact 57.340M-active campaign has a durable step-3,000 checkpoint at 22,999,779 optimizer positions. Compiled training preserves bounded full-parameter parity and cuts peak MLX memory about 58%, but the adopted route's repeated pooled speedup is 1.66x against a 2x gate. Direct decode is 9.44x faster with exact parity. Improve the hot path, then continue the same weights through the evidence-efficient rung ladder. |
| `T2` Honest behavioral numerator | waits for `T1` | At least one lineage-bound checkpoint produces nonzero direct model-only behavior on the frozen source-disjoint functional surface. Zero earns only its exact scoped verdict. |
| `T3` Real daily-use lane | assisted use ready; learned credit waits for `T2` | At least five distinct days of accepted, missed, ignored, corrected, completed, failed, or abstained real outcomes with effect and governance-cost records. |
| `T4` Joined governed vertical | waits for `T2` and `T3` | A natural success and a blocked/rollback path join intent, VCM, plan, route, generation, verification, authority, effect observation, residual, and dogfood outcome without orphan state. |
| `T5` Matched causal flagship | waits for `T4` | Full governance, test-only, record-only, and appropriate ablations run on one frozen natural denominator with matched budgets, independent evaluation, uncertainty, weak tails, and a fair rescue ladder. |
| `T6` Book handoff and challenge | waits for `T5` | A public-safe pack binds commits, artifacts, estimand, effects, costs, residuals, non-claims, and maximum inference; broadened claims face separate reproduction and transfer. |

`T3` may gather assisted-product evidence before `T2`, but assistance cannot become model
credit. A perfect governance trace around a behavior-zero proposer is not useful governed
intelligence.

## Current Operating Decision

The first practical architecture is frozen:

- MoECOT: a shared encoder-decoder transformer trunk plus independently owned low-rank
  English, Python, JS/TS, HTML/CSS, and Rust arms; 57,340,426 active parameters and
  67,357,711 total parameters.
- Controls: decoder-only dense active-parameter and total-parameter controls at
  57,348,617 active and 67,357,193 total parameters.
- Data: 422,334,331 unique model-visible positions, 7.365 per active parameter; frozen
  optimizer target 1,146,808,520 positions over 2.715 planned passes.
- Evaluation: one unconsumed, source-disjoint 160-case functional surface with 32 cases
  each for English, Python, JS/TS, HTML/CSS, and Rust.
- Runtime: native MLX grouped-query attention and compiled width-bucketed microbatches of
  four, accumulated into one token-mass-weighted batch-16 clip/update. A same-state
  three-pair, 24-update qualification measured 1,949-2,425 eager versus 3,567-3,584 compiled
  positions/second with bounded full-parameter equivalence. The adopted pooled speedup is
  1.66x, below the 2x gate, while peak MLX memory fell from about 8.10 GB to 3.40 GB. The
  real 500-update continuation sustained 2,914.2 positions/second. Full-batch compilation
  was slower and reached 8.57 GB. A stricter microbatch-eight rerun checked all 54,836,746
  parameters after every paired route; drift remained below `2.38e-7` absolute/about `8e-8`
  relative L2, but speed was only 1.52x median/1.62x pooled and peak memory was 5.00 GB.
  A real-checkpoint bf16-compute/fp32-master rerun was numerically bounded but reached only
  0.976x median/0.978x pooled speed and increased peak memory to 5.21 GB. Neither alternative
  is adopted on this M1.
- Current run: shared trunk checkpoint 3,000, 22,999,779 optimizer positions, exact model
  SHA `3a9b04ad...05a7`, and optimizer SHA `62e1b52b...5f96`. Private-development loss
  improved 0.57% from step 2,500 to 3,000; English regressed 0.43% while the four code arms
  improved. Direct beam decoding is 9.44x faster with 8/8 exact output-and-receipt parity,
  but seven of eight qualification outputs still fail closed on serialization.

At the measured canary rate, the remaining shared-trunk raw-position budget is about four
days of ideal continuous compute rather than more than eight; the complete 4.05B-position
campaign is still about 14.5 ideal compute-days on this M1. Preserve the resumable
step-3,000 lineage while qualifying faster numerics and evidence-efficient stopping.
Never mutate weights in flight or tune from the frozen confirmation heldout.

## Decision Rule

1. Train the shared trunk, five arms, and both dense controls under the frozen contract.
2. Evaluate direct arm behavior, routing behavior, and composed behavior separately.
3. Select the dense/hybrid survival lane if it wins matched practical utility or if all
   modular advantages disappear once route success is separated from answer success.
4. Retain MoECOT only for a repeatable Pareto advantage without a weak-arm regression.
5. Preserve SymLiquid as a nonblocking discovery comparator. It cannot delay the useful
   assistant unless repeated matched-compute evidence earns that role.
6. Preserve KERC as an inconclusive successor candidate until K4-K8 are faithful and
   prospectively frozen. Its K0-K3 evidence is neither end-to-end success nor failure.
7. If every 57M alternative remains functionally zero, falsify only this scale/data/
   training regime and advance once to a preregistered data-supported larger rung.

## Capability Tracks

### Track 0: Data And Curriculum

Owns rights, provenance, source-disjoint splits, contamination, quality, deduplication,
tokenizer efficiency, task-complete units, optimizer exposure, continual update, and
deletion causality. Licensed product-aligned data is primary. Teacher rows are capped
residual pressure, not the bootstrap corpus. Repeated positions never count as unique.

Acceptance after training: every checkpoint traces to admitted rows and exact optimizer
exposure; data deletion can identify descendants and prove model/storage consequences
without claiming parametric unlearning from storage erasure alone.

### Track 1: Neural Seed

Owns the shared trunk, language arms, dense controls, checkpoints, optimizer lineage,
resume, direct generation, and architecture verdict. English and every programming
language retain separate arm-level evidence even when a shared trunk is used.

Acceptance: independently loadable checkpoints produce useful model-only behavior on
source-disjoint functional tasks. Loss, syntax, exact recovery, route accuracy, and
assisted repair are diagnostics, not substitutes.

### Track 2: Verifier-Guided Search

Owns bounded candidate search, verifier bandwidth, abstention, localized repair, and
accepted-solution-per-second accounting. It activates only after one-shot generation has
a nonzero numerator.

Acceptance: equal-budget A/B evidence shows search increases independently verified
functional passes after all verifier, repair, latency, and fallback costs are charged.

### Track 3: Correctness Learning

Owns preference optimization, verifier-reward learning, policy-update leases, reward
hacking probes, rollback, and continual improvement. DPO/IPO/ORPO/KTO/SimPO and
GRPO/RLOO/ReMax/RLVR share frozen schemas and matched comparisons.

Acceptance: a selected objective improves source-disjoint functional behavior without
authority expansion, evaluator leakage, reward hacking, or unacceptable weak-tail drift.

### Track 4: Fast Generation

AR remains canonical. MTP is checkpointed but starts with zero loss scale. Speculative,
self-draft, Medusa/EAGLE, LayerSkip, state-space, KV-cache, and sketch/diffusion routes
remain disabled, retired, or post-hoc until independently qualified.

Acceptance: useful verified output per second improves under matched quality, memory,
latency, and lifecycle cost. Proposed-token throughput alone never qualifies a route.

### Product Track: Daily Assistant

Owns local chat, repository work, tool use, VCM continuity, effects, rollback, and
dogfood outcomes. Zero external inference is mandatory at serving time.

Acceptance: real multi-day use establishes useful completion and correction rates, cost,
latency, failure behavior, and rollback. Fixture traces provide mechanics only.

## Phase Program

### Phase 0: Repository Self-Model And Registry Discipline

State: `wired`. The abstraction/implementation registry, SCF bindings, project steward,
cleanup queue, AIBOM, ownership, and replacement contracts are canonical.

Next: inventory and remove unrelated live dependency edges to the superseded
strict-generator/code-LM-closure family, archive its bulky generated state through
content-addressed pointers, and preserve independently useful verifier/guard code under
its owning abstraction. Retire the predecessor executable route only after the registered
57M successor passes the same functional contract and regression floor, or after an
explicit evidence-preserving demotion; checkpoint durability alone is not behavioral
replacement. Prove each change with import, CLI, registry, and evidence replay receipts.
Keep this roadmap below 1,100 lines; keep complete detail in the matrix, registry, state,
and Git history.

### Phase 1: VIEA Execution Spine

State: `wired`. Intent, artifact graph, claim/evidence records, plan, VCM packet, route,
verification, authority, effect, residual, and outcome share stable joined identities.

Next: require the effect-complete profile for every new side-effect class and prove
independent observation plus rollback before route authority.

### Phase 2: Stable Capability Fields And Route Authority

State: `wired`. Abstractions are stable fields; implementations are content-bound,
attested, leased, observable, replaceable, migratable, and revocable.

Next: every replacement must preserve exact identity, state migration, authority,
negative tests, observation, rollback, and stale-reference rejection.

### Phase 3: VCM As Default Context Substrate

State: `wired`. VCM owns stable semantic objects, typed temporal relations, hybrid
retrieval, context compilation, transactions, certificates, compaction, deletion, and
ontology migration.

Next: consume lifecycle records in continual-update and model-context paths. Qualify VCM,
STS, and their interaction only after a behavior-positive checkpoint, under equal visible
context and candidate budgets. Native KV/prefix-cache claims remain separate.

### Phase 4: Candidate Integrity And Learned Accounting

State: `wired`. Independent audits classify provenance from source and execution rather
than trusting candidate flags. Hidden-answer information flow fails closed.

Next: keep direct full-body learned candidates and every assisted mechanism separately
accounted through functional evaluation and promotion.

### Phase 5: Daily Assistant And Dogfood

State: `partial`. The local VIEA/effect-complete assisted runtime exists; learned general
chat is not yet useful.

Next: use the selected local checkpoint on real low-risk work and collect consented local
accepted/missed/ignored/corrected/completed outcomes, with raw private text off by default.

### Phase 6: Deterministic Tools And Search

State: `wired`. Typed math, logic, local search, retrieval, replay, evidence, and tool
cards fail with structured `UNKNOWN`, `UNSOLVED`, or `TOOL_FAULT`, never fabricated output.

Next: after direct behavior is positive, measure tool selection and verifier-guided repair
against equal-budget model-only controls. Tool-assisted scores remain separate.

### Phase 7: Teacher And Data Governance

State: `partial` for behavior-dependent continual-update proof; first-campaign corpus is
frozen and training-authorized. Static model-derived corpora do not authorize live use.

Next: preserve the frozen corpus during training. Later, qualify continual update and
deletion causality. Published-agent traces require rights, privacy, deduplication,
outcome reconstruction, verifier replay, and heldout causal lift before training credit.

### Phase 8: Resource, Cost, And Mac Acceleration

State: `active`. Native GQA, semantics-qualified compiled microbatch training, prompt-only batched beam
decoding, exact checkpoint-format comparison, and content-bound assistant refresh reuse
are qualified. The latest repeated compiled/eager pair is 1.66x pooled, so the 2x training
target remains open even though compiled execution is stable and uses about 58% less peak
MLX memory. The 500-update route sustained 2,914.2 positions/second; uncached decode improved
9.49x with 8/8 exact parity; unchanged governed assistant refresh improved 578x under exact
content identity. Scheduler records
CPU/GPU/MLX/CUDA, memory, thermals, battery, disk, queue, latency, and lifecycle cost.

Next: execute the acceleration program below. Preserve optimizer semantics and the frozen
data/order/schedule unless a separately preregistered experiment changes them. Mixed
precision, quantization, compilation, caching, and faster hardware require finite-loss,
gradient, update, reload/resume, divergence, memory, throughput, and heldout-equivalence
evidence before adoption.

#### Capability-Critical Acceleration Program

The primary optimization metric is **time to trustworthy useful evidence**, followed by
interactive p50/p95 latency and accepted verified output per second. Raw tokens/second is
diagnostic. The 100x target is a stretch target for end-to-end time-to-decision through a
portfolio of kernel speed, work avoidance, reuse, and parallelism; it is not a claim that
software can create 100x more M1 arithmetic throughput.

1. **Measurement closure.** Establish reproducible cold/warm baselines for stage loading,
   training step, checkpoint save/load, private-development loss, direct generation,
   verification, VCM compilation/query, routing, and local assistant completion. Record
   useful positions or accepted outputs, p50/p95, peak unified memory, disk I/O, thermal
   state, and energy when observable. Profile only canonical routes.
   Current evidence covers training, checkpoint load/format, private-development loss,
   direct generation, and unchanged joined VCM/tool/planner/verifier refresh. Checkpoint
   publication, task-specific assistant completion, resident serving, and peak-memory/
   energy evidence remain open.
2. **Sustained MLX training qualification.** Run the compiled microbatch route for at
   least 500 real-data optimizer updates across a checkpoint boundary and exact resume.
   Require the same batch-16 token mass, one clip/update per logical batch, matching data
   order, finite gradients, bounded parameter delta, non-regressed private-development
   loss, and at least 2x sustained useful positions/second over the eager baseline.
   Autotune only safe microbatch and sequence-width buckets for each Mac memory tier.
   Semantics-qualified on M1: the adopted microbatch-four route measured 1.66x pooled over
   eager with about 58% lower peak MLX memory, and sustained 2,914.2 positions/second over
   500 real updates. Microbatch eight was then checked over all 54,836,746 final parameters
   across three alternating 24-update pairs; maximum drift was `2.38e-7` absolute/about
   `8e-8` relative L2, but speed was only 1.52x median/1.62x pooled and varied from 1.36x to
   1.97x. Reject it for performance despite semantic parity. The 2x gate remains uncleared.
   Aggregate dev loss improved 0.57%, with an explicit English weak-tail regression.
3. **Precision and optimizer memory.** Compare float32, bfloat16, and mixed-precision
   master-weight policies on the same checkpoint/data. The initial pure-bfloat16 canary
   produced only about 1.09x throughput and changed loss. The stricter compiled
   bf16-compute/fp32-master route preserved finite fp32 authority and bounded loss drift;
   its latest microbatch-eight rerun measured 0.976x median/0.978x pooled speed and 5.21 GB
   peak MLX memory against 5.00 GB for fp32. Reject both on this M1. Reopen precision only on a different Apple
   GPU generation or after a backend change, using the same qualification contract.
4. **Evidence-efficient rung control.** Make the declared pilot/review contract executable.
   Emit immutable learning-curve checkpoints and private-development measurements at
   short logarithmic intervals, then evaluate direct model-only behavior at preregistered
   review points. Use successive halving across MoECOT and dense controls so clearly
   dominated candidates stop consuming compute. Never inspect or tune from confirmation
   or public surfaces. A stopped run remains scoped evidence, not broad falsification.
5. **Checkpoint and storage path.** The current 209 MB model plus 418 MB optimizer state is
   measured. Safetensors preserved the exact 197-tensor manifest and loaded 4.58x faster in
   the latest alternating three-load comparison; size and save time did not materially
   improve. Migrate only through registered lineage.
   Next qualify publication cadence and peak memory; do not background-copy a 57M model on
   a 16 GB unified-memory host without immutable-snapshot and memory evidence. Retention
   keeps the canonical/latest generation plus explicit pins. Recovery integrity outranks
   write speed, and free disk must remain high enough to prevent invisible swap cost.
6. **Canonical inference hot path.** Replace per-case/per-beam Python stepping in the
   MoECOT direct generator with resident model loading, batched prompt prefill, batched
   beam advance or greedy mode, incremental KV cache, prefix-cache reuse, and continuous
   batching. Remove host conversions and full-vocabulary Python sorts from token loops.
   Evaluate quantized serving and MTP/speculation only under identical visible context,
   output contract, and functional quality. Report time to first token, decode rate,
   end-to-end p50/p95, memory, and accepted verified outputs/second.
   Batched beam advance, device-side admissible ranking, and exact pre-forward pruning are
   qualified at 9.44x aggregate uncached speed and 8/8 parity. The deferred KERC decoder now
   uses the same machinery and has serial/optimized token-path parity; a full KERC pipeline
   throughput claim remains pending. An indexed shared-cache gather preserved exact output
   but measured only 1.004x pooled against the simpler per-branch assembly and was removed.
   Bounded sequence-axis preallocation reached only 1.009x on a 512-token stress run and was
   also removed. Resident serving, batched
   prefill, prefix reuse, continuous batching, and bounded KV-cache growth remain open.
   A separate external speed-audit hypothesis that wide rows were silently forced to batch
   one was checked against executed receipts: the trunk is width 512 and the KERC canary
   peaked at width 2,580, so neither crossed the 8K batch-two boundary. Do not change that
   safety bucket as a proxy fix. Profile and compile the actual KERC auxiliary-objective path,
   currently about 23 positions/second, before changing sequence policy.
7. **Context, routing, and tools.** Profile the joined assistant route rather than isolated
   fixtures. Make VCM indexes persistent and incremental, cache content-bound compiled
   packets and denial decisions, prefetch only from observable plans, avoid repeated JSON
   scans, and batch independent verifier/tool calls. Cache hits must bind model, tokenizer,
   policy, capability, snapshot, and source identities; stale or unauthorized hits fail
   closed. Qualified: the five-step VCM governor, task-context bridge, deterministic-tool
   registry, plan compiler, and private-verifier refresh measured 578x faster in the latest
   cold/warm comparison. Reuse binds exact command/input/output hashes and a
   300-second freshness window; mutation and stale controls force a rerun. Task-specific
   tool execution and model generation remain uncached and must be measured separately.
8. **Native and distributed escalation.** Move only measured residual Python hot loops to
   Rust/Metal/MLX kernels. When another node is available, parallelize independent arms,
   controls, evaluation shards, and verification, not tightly synchronized dense steps
   over a high-latency link. Record transfer and merge cost in the speed denominator.

Milestones: at least 2x sustained same-semantics M1 training throughput; at least 10x
shorter first architecture-decision wall time; at least 5x repeated-prompt end-to-end
latency or 2x uncached decode throughput at non-regressed quality; and a stretch 100x
time-to-feedback improvement on reuse-heavy workloads. Reject any optimization that wins
only by skipping required work, reducing candidate/evaluation budgets without disclosure,
changing the learning objective, weakening verification, or losing replayability.

### Phase 9: Hive Distributed Operation

State: `frozen` by the current single-node environment. Stable identity, discovery,
policy-first registered tasks, stale-peer cleanup, capability refresh, artifact routing,
and bounded authority exist.

Next: when peers are reachable, run one real registered task and verify scheduler,
authority, failure, resource, artifact, and update receipts. No arbitrary shell or public
gateway authority.

### Phase 10: Practical Neural Seed

State: `partial`, with the first 57M shared trunk actively training. The 10.8M rung is
closed at 0/160 across MoECOT and both dense controls; that result does not transfer to
the current rung.

Next: reach a durable checkpoint, prove resume, complete all frozen alternatives, and
consume the fresh functional surface once. Only then qualify preference/RL, GVR, search,
MTP, other generation modes, and substrate adoption from real behavior.

### Phase 11: Cognitive Kernel Discovery

State: `wired` as non-routeable discovery contracts. OneCell-RWM has a content-bound
handoff, exact/latent boundary, ABI, objective/checkpoint groups, and successor campaign
requirements. SymLiquid remains protected but nonblocking.

Next: require separately preregistered matched total-cost campaigns before either can
earn route authority. Absence from the first campaign is not falsification.

### Phase 12: Public Calibration

State: `wired`. Fresh surfaces use exact-once reservations, contamination checks, and
payload firewalls.

Next: after a confirmation-qualified private functional improvement, run one genuinely
fresh public calibration. Report per benchmark and failure family; mine only abstract
failure categories into private curriculum, never public payloads.

### Phase 13: Semantic IR And KERC

State: `partial`. KERC K0-K3 are banked: exact packet/object codecs, rate-distortion
economics, calibrated source-visible per-unit allocation, real linguistic supervision,
independent evaluation, causal intervention, MLX, and content-addressed replay.

Next after the practical campaign: K4 real interaction amortization and codec challengers;
K5 coordinated learned compiler/reasoner/renderer plus independent semantic verifier;
K6 lifecycle/security/checkpoint/migration/rollback stress; K7 total-cost accounting;
K8 prospective matched multi-seed campaign. Until then: `INCONCLUSIVE_IMPLEMENTATION`.

### Phase 14: Compression, Proof, And Claim Evidence

State: `wired`. Evidence records, claims, defeaters, compression receipts, archive
pointers, assurance graphs, and evaluation-integrity cases are canonical. Exact replay is
GREEN for 2,114/2,114 retained artifacts. Digest-bound aggregate records reduced the live
replay report from 28.5 MB to 22.6 KB without dropping failed details or fail-closed behavior;
the hot-report budget is GREEN at about 1.01 GB.

Next: reduce the 9.70 GB checkpoint warning through reference-aware retirement after the
active training lineage is settled, retire duplicate families, and never let a
producer-authored green report self-authorize trust.

### Phase 15: Procedural Memory And Toolification

State: `implemented` for three guarded exact assets. Reflex compilation, qualification,
dependency invalidation, quarantine, canary replacement, and rollback share canonical
owners.

Next after real traces: learn negative-space guards; run differential plus live-shadow
qualification; measure route regret; support decompilation and exact retirement.

### Phase 16: MoECOT And Octopus

State: `partial` pending trained checkpoints and natural route evaluation. Stable language
arm contracts and composition semantics exist; route accuracy cannot imply answer utility.

Next: bind trained arm cards to exact model/data/checkpoint identities. Evaluate learned,
rule, generalist, specialist, clarification, abstention, and single/sequential/parallel/
adjudicated composition on frozen naturally ambiguous requests.

### Phase 17: Simulation And World Models

State: `wired` for bounded planning adapters. Simulation fidelity, counterfactual scope,
world-adapter identity, and failure boundaries remain distinct from deployment evidence.

Next: extend only when a real simulator or resource adapter is required; require fidelity
receipts and preserve the simulation-to-reality claim boundary.

### Phase 18: Governance And Failure Boundaries

State: `wired`. The governance kernel owns roles, predicates, thresholds, authority,
budgets, leases, revocation, dispute, shutdown, replay protection, and assurance use.

Next: every new effect or inter-stack exchange must preserve these semantics and prove
blocked, revoked, expired, replayed, conflicting, and rollback paths.

### Phase 19: Book-To-Theseus Synchronization

State: `implemented`. The project binds a committed ASI-book manifest and source-owned
chapter fields; live dirty-book edits are intake drift, not architecture regression.

Next: reconcile only reviewed committed book revisions. Every accepted mechanism routes
to an existing abstraction or receives an explicit disposition, acceptance gate, test
obligation, and claim boundary.

## Registered Cross-Phase Contracts

The matrix is authoritative for exact evidence, boundaries, and statuses. Current compact
inventory:

| Contract | State |
|---|---|
| Reflexive Router integration | pretraining wired; behavioral qualification pending |
| Kernel English / KERC | K0-K3 banked; K4-K8 successor work pending |
| Private functional utility | 10.8M rung falsified; 57M evaluation pending |
| Task-complete corpus and scale floor | first-campaign data frozen; behavioral result pending |
| OneCell-RWM | excluded from first campaign; protected successor |
| DPO/IPO/ORPO/KTO/SimPO | wired; behavior-dependent selection pending |
| GRPO/RLOO/ReMax/RLVR | wired; behavior-dependent selection pending |
| Multi-target policy leases | wired; learned effects pending |
| Generate-verify-repair | wired; behavior-dependent qualification pending |
| MTP | wired at zero initial loss; matched qualification pending |
| Medusa/EAGLE/speculative/LayerSkip | explicit disabled/retired dispositions |
| Sketch-first/diffusion/LLaDA | retired from first campaign; separate re-entry contract |
| VCM transaction certificate ABI | implemented |
| Verification bandwidth/governance tax | implemented |
| Claim and belief revision | wired |
| Procedural memory/lookahead trie | wired |
| Circle/COIL/cyclic substrate matrix | protected discovery lane |
| Continual learning and deletion causality | wired; empirical campaign pending |
| Durable semantic memory | planned after direct behavior |
| Governed usefulness/effect rollback | planned through real dogfood |
| Ambiguous routing/adaptive deliberation | waits for nonfallback candidate quality |
| Scalable oversight | wired; empirical use pending |
| Cross-context evaluation integrity | wired |
| Capability commitment/assurance graph | wired |
| Supply-chain and weight custody | wired; real signing/custody evidence pending |
| Inter-stack identity exchange | wired |
| Open-ended improvement campaign | wired and disabled until behavior-positive proposer |
| QCSA | pretraining wired; adaptive learned qualification pending |

## Evaluation Program

### Private Functional Utility

- English: blinded local rubric/pairwise assessment with agreement and adjudication
  reported; no external runtime tokens and no raw response retention by default.
- Python: compile plus isolated tests and runtime behavior.
- JS/TS: parse/type-check/build plus isolated runtime behavior.
- HTML/CSS: structural validity, DOM/style contract, and deterministic rendering checks.
- Rust: format/check/compile/test with bounded execution.
- Repository work: before/after context, patch application, tests, effect observation,
  rollback, and human outcome.

Report direct model-only, model plus VCM, plus STS, plus search, plus tools, plus repair,
and fully assisted outcomes separately. Exact recovery remains diagnostic. Selection is
predeclared and cannot use heldout outcomes as architecture tuning data.

### Adequacy Before Verdict

Every promotion, retirement, or broad negative requires:

1. faithful causal mechanism coverage;
2. learnability, gradient, overfit, intervention, ablation, checkpoint/reload/resume,
   migration, and rollback sanity checks;
3. strong baselines matched on raw data, train compute, tuning opportunity, inference,
   verifier budget, and total lifecycle cost;
4. source-disjoint independent evaluation, multiple seeds, uncertainty, power, weak
   tails, and attack cases;
5. independent candidate integrity and evaluator provenance.

Missing conditions produce an inconclusive result and an owner repair task.

### Public Calibration

Public calibration follows a material frozen model change and private functional evidence.
Fresh industry surfaces may run without arbitrary monthly budgets, but exact consumed
surfaces cannot be rerun and public payloads cannot enter training. Compare with industry
benchmarks only under their standard, disclosed rules. Tool-assisted and model-only
scores are distinct.

## Repository And Runtime Discipline

- `main` is the single working branch and should reflect this device's intended state.
- Generated reports, checkpoints, runtime caches, and distribution artifacts stay out of
  source control unless they are small canonical manifests or public-safe evidence.
- Hot reports stay below policy through reference-aware archive pointers; current routes,
  mutable ledgers, promoted artifacts, and explicit pins remain hot.
- Superseded active code is retired only after import/replay proves no live route depends
  on it. Git history is the idea archive; deprecated code cannot be imported by active
  paths.
- Repeated scans, tokenization, copies, oversized JSON, serial fanout, and Python hot loops
  must become cached, incremental, batched, streaming, Rust, Metal, or MLX implementations
  when measured profiles justify the change.
- Performance claims require sustained workloads, not short warmup canaries. Report peak
  memory, thermals where available, checkpoint cost, useful throughput, and weak tails.

## Success Definition

Theseus reaches the current roadmap milestone only when:

1. the frozen campaign and matched controls complete with exact lineage;
2. at least one checkpoint is independently loadable and useful on source-disjoint direct
   English and code tasks;
3. architecture selection follows the preregistered matched decision, with no hidden
   assistance or weak-arm suppression;
4. VCM, STS, search, tools, preference/RL, routing, and fast generation earn route roles
   through separate causal ablations;
5. fresh public calibration measures transfer without contaminating training;
6. a local assistant serves canonically with zero external inference and accumulates real
   outcome pressure;
7. updates, effects, memory, authority, evidence, cleanup, replay, migration, security,
   rollback, and retirement remain green under the selected implementation;
8. every matrix row is mature and exercised or honestly marked external, time-dependent,
   protected discovery, or inconclusive with a precise re-entry condition.

Mechanics canaries, schemas, report volume, private saturation, training loss, route
accuracy, and governance completeness alone are not completion.

## Canonical Commands

```bash
python3 scripts/theseus_project_registry.py --gate
python3 scripts/pretraining_architecture_freeze.py --execute-replays
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
python3 scripts/theseus_artifact_retention.py --budget-gate
```
