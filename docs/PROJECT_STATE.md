# Project Theseus: Current State

Last consolidated: 2026-07-29 UTC. Implementation evidence is current through
`54aa6171`; this page is part of the following documentation transaction.

This is the canonical human-readable current-state page. It describes what is
true now; it is not the historical evidence ledger and it does not authorize a
training or serving action by itself.

## One-Sentence Verdict

Theseus is a well-governed local AI research system with an exact resumable
57.3M-parameter training lineage and a substantial assisted runtime, but it is
not yet a demonstrated useful learned assistant: training is deliberately
held, the current checkpoint is `NOT_EVALUATED`, matched dense controls have
not been trained, and genuine dogfood evidence is insufficient.

## How Truth Is Resolved

Different sources own different kinds of truth:

| Question | Authority |
| --- | --- |
| What is allowed? | `AGENTS.md` and the relevant policy/configuration |
| What implementation or route is canonical? | `configs/project_manifest_registry.json` |
| What work remains? | `configs/roadmap_implementation_matrix.json` |
| What was observed? | Content-bound `reports/`, checkpoints, and append-only ledgers |
| What is true in plain English now? | This file |
| What should happen next? | `roadmap.md` |
| How is the system designed? | `docs/TOP_TO_BOTTOM_ARCHITECTURE.md` and `docs/VIEA.md` |

If a human summary conflicts with a current machine artifact, the machine
artifact wins and the summary must be repaired. A GREEN mechanics, security,
or registry result is not model capability or training authority.

## State Vocabulary

| State | Meaning |
| --- | --- |
| `CUSTODY_GREEN` | Exact state and lineage replay; no capability implication |
| `TRAINING_HELD` | No new long training segment may launch |
| `TRAINING_READY` | Every current launch gate passes; still requires an explicit operator decision |
| `NOT_EVALUATED` | The frozen capability surface has not been consumed |
| `INCONCLUSIVE_EXPERIMENT` | The tested result cannot support the broader scientific claim |
| `LOCAL_ONLY` | Suitable only for loopback/local use; no LAN or public exposure claim |
| `ASSISTED_ONLY` | Tools, retrieval, rules, or scaffolding contributed; no learned-model credit |
| `EMPIRICAL_SUPPORT_INSUFFICIENT` | Synthetic or sparse observations do not establish daily usefulness |
| `FROZEN` | Preserved and intentionally inactive until a named re-entry condition is met |

The glossary in `docs/GLOSSARY.md` defines project-specific terms.

## Current Scorecard

| Dimension | Current state | Evidence boundary |
| --- | --- | --- |
| Repository | `LOCAL_ONLY_NOT_REMOTE_VERIFIED` | Implementation baseline `54aa6171`; inspect Git for the documentation commit and current divergence from `origin/main` |
| Project registry | `GREEN` | Zero routing blockers and zero hard governance violations |
| Human roadmap gate | `YELLOW` | Zero hard gaps; two pre-training readiness blockers and one book-pin warning |
| Active model custody | `CUSTODY_GREEN` | Exact step 11,416 model, AdamW, MLX RNG, cursor, and 37-manifest prospective lineage |
| Training | `TRAINING_HELD` | `runtime/control/neural_seed_yield_after_segment` is present |
| Learned capability | `NOT_EVALUATED` | Frozen 160-case private functional surface has consumed 0 cases |
| Matched controls | `NOT_TRAINED` | Both dense controls remain at 0 optimizer steps |
| Product usefulness | `EMPIRICAL_SUPPORT_INSUFFICIENT` | Existing support is mostly synthetic; the first explicit travel-mode task was honestly recorded as `missed` |
| Runtime exposure | `LOCAL_ONLY` | Authority surfaces pass local adversarial tests; LAN/public exposure remains unauthorized |
| Teacher accounting | `GREEN_ACCOUNTING` | 15 teacher-accepted rows among 115,429 accepted rows, about 0.013% |
| Storage | `GREEN_MAINTENANCE` | About 40.8 GB reclaimed; roughly 40 GiB free after cleanup |
| Public CI | `DEFINED_NOT_REMOTE_VERIFIED` | Linux and guarded Mac workflows exist locally but the 13 commits have not been pushed |
| ASI Stack synchronization | `PIN_WARNING` | The live book manifest differs from the reviewed Theseus pin |

## The Active Neural Experiment

The practical experiment is deliberately narrow:

> Does the modular MoECOT candidate improve useful, safe, source-disjoint
> behavior over matched dense controls at comparable parameter and training
> cost?

The active candidate uses a shared encoder-decoder transformer trunk with
independent English, Python, JavaScript/TypeScript, HTML/CSS, and Rust arms.
The matched controls hold active or total parameter count comparable.

The selected campaign recipe is:

- compiled FP32 MLX;
- compiled microbatch 4;
- width quantum 64;
- AdamW;
- separate Q/K/V projections;
- SwiGLU;
- sequential unscaled residuals;
- fixed open-vocabulary autoregressive generation;
- canonical KV handling;
- Semantic-IR as a control surface, not a capability claim;
- independent verification and zero credit for templates, tools, retrieval,
  routers, deterministic renderers, or fallbacks.

Per-head Muon, AttnRes, SiTU-GLU, fused QKV, full-shape SOAP, native ANE block
training, target-window projection, and MLX fast synchronization were tested in
bounded matched regimes and were not selected. Their negative evidence remains
scoped to those implementations and regimes.

KERC/RDC remains a protected successor discovery lane with
`INCONCLUSIVE_EXPERIMENT` status and zero campaign-one optimizer exposure. It
may not block the current practical experiment and may re-enter only through
its recorded K4-K8 adequacy and resource conditions.

## Checkpoint And Lineage

The shared trunk is paused at:

- optimizer step: `11,416`;
- optimizer positions: `87,441,996`;
- checkpoint root:
  `checkpoints/moecot_mlx_57m_active_preregistered_v1`;
- model SHA-256:
  `d2a485a59add5eccaf9388ae6bb7ae0972037c6e182602b0f91bec764b632506`;
- AdamW SHA-256:
  `4d85f93999c5b030728e550b9bbe8aeda476945d36bee0b96cdb3fe98da071eb`;
- MLX RNG SHA-256:
  `9209383f5b4eb599c89462f922957b13e60f03eb6fac3b73f6af9003c023850f`.

The prospective append-only lineage begins at step 9,048 and contains 37
manifests through step 11,416. The complete predecessor chain from the earlier
step-3,480 state to step 9,048 was not retained. This gap is explicit and may
not be laundered into a full-chain replay claim.

The frozen pretraining target is `1,096,734,920` optimizer positions. The
shared trunk has completed about 7.97% of that target. This is progress, not
capability.

## Why Training Is Held

The finite acceleration and independent readiness audits are GREEN, and the
content-addressed step-11,416 replacement package was GREEN before the latest
maintenance transaction. However:

1. the operator hold is intentionally installed;
2. the architecture-freeze package is source-bound and now reports stale
   identities after legitimate evaluator, security, registry, and
   documentation changes;
3. Phase 0 remains partial because superseded families and reproduction debt
   are not fully consolidated;
4. Phase 8 is globally partial even though the selected M1/MLX campaign route
   is closed; its remaining cross-platform and production-serving work should
   not be confused with a campaign blocker.

Therefore the honest state is `TRAINING_HELD`, not `TRAINING_READY`. After this
documentation transaction is banked, the source-bound freeze and independent
readiness package must be regenerated and reviewed before the hold could be
removed. Regenerating a GREEN report is not itself an instruction to train.

## Evaluation State

The private functional utility contract is frozen before evaluation:

- 160 cases;
- 32 cases for each of five arms;
- source-disjoint;
- candidate output treated as inert data;
- runtime forbidden-field capability enforcement;
- independent candidate-integrity recomputation;
- prompt-injection, rubric-copy, self-score, empty, and malformed-output
  rejection;
- a separately bound prospective human-audit receipt;
- 0 consumed cases;
- capability claim `NOT_EVALUATED`.

The surface should be consumed only after the selected candidate and both
matched controls complete their preregistered training and custody checks.

## Product And Dogfood State

The assisted local CLI, VCM, deterministic tools, VIEA records, and local
runtime are real mechanisms. They do not demonstrate learned capability.

The first explicit travel-mode dogfood request asked the assistant to identify
the training hold and one safe next action. The response missed both
requirements. It was recorded as:

- outcome `missed`;
- error family `request_fidelity_miss`;
- learned-model credit `false`;
- capability credit `none_assisted_or_tool_mediated`;
- training rows written `0`;
- external inference calls `0`.

This is useful evidence: the product loop is operational, while actual
request-following utility remains unproven. The next dogfood work should use
real low-risk tasks and record accepted, missed, ignored, corrected, completed,
failed, and abstained outcomes without manufacturing success.

## Security And Exposure

The local authority package now enforces:

- authenticated mutations;
- strict JSON, content-type, and body-size checks;
- exact Origin and CSRF policy;
- rate, concurrency, job, and SSE limits;
- request-local teacher/network authority that defaults false;
- tokened, non-wildcard-CORS OpenAI-compatible defaults;
- loopback Hive defaults;
- separate coordinator, worker, and discovery credentials;
- signed discovery;
- header-only credentials;
- fail-closed remote execution when sandbox qualification is absent;
- sanitized errors and random identifiers.

This is local code and adversarial-test evidence. It is not an
internet-security audit. Keep dashboard, OpenAI shim, and Hive loopback-only
until a separate exposure qualification is reviewed.

## Data And Teacher State

The frozen corpus passed the current provenance, license, exact and semantic
deduplication, contamination, retention, tokenizer, and recursive synthetic
share checks. It should not be expanded merely because training has not yet
run.

Teacher accounting is content-bound through an append-only hash chain:

- total accepted rows: `115,429`;
- teacher-accepted rows: `15`;
- teacher share: `0.0001299500` (about `0.013%`);
- configured cap: `10%`;
- accounting replay: valid.

Live teacher data remains targeted residual pressure, never bootstrap data or
runtime serving.

## Resource And Reproducibility State

The selected training route is compiled FP32 MLX. Python remains orchestration;
Rust is used where correctness, parsing, data movement, or profiled CPU work
justifies it. Rewriting orchestration in Rust without a measured bottleneck is
not an optimization plan.

Travel-mode cleanup permanently removed 93 unreferenced closed canary runs and
126 superseded checkpoint files. The deleted payloads were regenerable but are
not directly recoverable. Active checkpoint custody and meaningful evidence
were preserved and rehashed.

Linux CI now defines:

- scoped Python format, lint, and security/evaluator tests;
- Rust format, clippy, and tests excluding Mac-only crates;
- license and public-release checks;
- dependency review;
- a deterministic public-safe evidence-protocol capsule.

Mac MLX qualification remains a guarded manual workflow on the qualified local
runner. Because local `main` is ahead of `origin/main`, hosted CI has not yet
validated the latest transaction.

## Current Gates

| Gate | State | Interpretation |
| --- | --- | --- |
| Project registry | `GREEN` | Canonical routes have current minimal evidence |
| Roadmap implementation | `YELLOW` | Zero hard gaps; partial phases and book-pin warning remain |
| Pre-training architecture | `NOT_READY` | Two blockers: Phase 0 consolidation/currentness and Phase 8 status semantics/currentness |
| Replacement freeze | Historical `GREEN` | Exact step-11,416 package before later source changes |
| Independent pre-long-run audit | Historical `GREEN` | Readiness evidence only; explicitly did not authorize training |
| Functional utility freeze | `NOT_EVALUATED` | 0/160 cases consumed |
| Runtime exposure | `LOCAL_ONLY` | No LAN/public authorization |

## Immediate Work

In order:

1. Finish this documentation consolidation and bank it.
2. Reconcile the Phase 0/Phase 8 readiness semantics so nonblocking
   cross-platform work cannot make a Mac campaign appear both authorized and
   not ready.
3. Push the 13 local commits and observe the new Linux CI before relying on it.
4. Use the assisted CLI on genuine low-risk work and collect honest empirical
   outcomes.
5. When the laptop can be dedicated, regenerate the exact source-bound
   architecture freeze and independent readiness audit, then run one bounded
   fresh-process selected-route qualification.
6. Only after explicit operator approval, resume the exact shared-trunk
   lineage, train both matched dense controls, and consume the frozen
   functional evaluation once.

## Explicitly Not Immediate

- new architecture families;
- more KERC or ANE work;
- generic acceleration searches;
- more private benchmark families;
- broad corpus expansion;
- public benchmark consumption;
- LAN or public dashboard exposure;
- remote arbitrary execution;
- mobile, spatial, compute-market, or product-surface expansion;
- claims that assisted behavior is learned capability.

## Definition Of Project Progress

Progress means at least one of:

- the learned student improves on source-disjoint functional behavior;
- the comparison becomes more scientifically honest;
- the runtime becomes safer or materially faster by measurement;
- Corben genuinely uses the system and the outcome is recorded;
- duplicated machinery is removed without losing evidence.

A new report, mechanism, dashboard, or document is not progress by itself.
