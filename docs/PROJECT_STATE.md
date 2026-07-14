# Project State

Last consolidated: 2026-07-14 UTC.

This file is the bounded current-wall page for Project Theseus. The machine-readable
implementation contract is `configs/roadmap_implementation_matrix.json`; the
forward narrative is `roadmap.md`; historical detail remains in Git and
`docs/archive/`.

## North Star

Build a private, locally trained assistant that Corben uses daily, serves with zero
external inference, improves through governed evidence, and drives accepted teacher
share toward zero. Public benchmarks are calibration only.

## Current Verdict

- **Architecture readiness:** locally actionable pre-training architecture is wired.
  This authorizes governed training; it does not prove model quality.
- **Practical model lane:** the seed is a shared transformer trunk plus separately
  checkpointed English, Python, JS/TS, HTML/CSS, and Rust experts. Dense decoder-only
  models remain matched falsification controls. Search, preference/RL, and fast
  generation remain amplifiers, not substitutes for a working proposer.
- **Capability wall:** v8 English reproduces one target exactly (`1/128`). Python,
  JS/TS, HTML/CSS, and Rust remain `0` exact; Python is `41/128` syntax-valid. The
  v8 development rows contain prompt and target but no complete executable verifier,
  so this is an exact-recovery diagnostic, not functional utility or an architecture
  win. Confirmation is untouched and no checkpoint is routeable.
- **Data gate:** the content-bound audit credits 683,254,465 unique governed
  positions. The frozen comparison uses 215,552,020 positions for a 10,777,601-active
  sparse model and mechanically matched 10,779,648-active/12,499,968-total dense
  controls. The data, tokenizer, seed, supervision, development, confirmation, and
  verifier contracts are immutable through the verdict.
- **Immediate dependency:** finish both dense controls unchanged and publish the v8
  exact-recovery diagnostic. Before completed control outputs are inspected, freeze a
  source-disjoint functional utility manifest: blind rubric/pairwise English and
  pinned compile/test/render checks for all code arms. Use that result, not exact
  target reproduction alone, for practical route selection. Do not patch from interim
  results.

## Evidence Boundaries

- Learned generation is scored separately from templates, structural renderers,
  n-grams, deterministic tools, retrieval, VCM, STS, and procedural routes.
- Candidate family and integrity are independently recomputed; producer-declared
  flags cannot support promotion.
- No fallback return receives learned-generation credit.
- Public prompts, tests, solutions, traces, and answer templates never become
  training rows. Fresh frozen public surfaces may be used for calibration after a
  material model change; consumed surfaces are not rerun.
- Live external models are training teachers only through the governed distillation
  path. Runtime external tokens are forbidden. OpenAI is the only authorized live
  teacher provider; Claude/Anthropic are forbidden for teacher use and project review.
- Static openly licensed corpora may include model-derived rows when quality tier,
  provenance, permitted use, contamination, retention, and recursive synthetic share
  are measured. This is data admission, not live teacher authority.
- Teacher rows require provenance, license, contamination, verifier, and retention
  receipts. Live teacher use is residual-only: accepted rows are capped at 10% and
  optimizer sampling at 2%, both trend down, and usefulness must beat a matched
  teacher-off control. The first teacher tranche was negative and remains quarantined.
- Natural-language training is English-only. Python, JS/TS, HTML/CSS, and Rust are
  separate programming-language arms; other human languages are quarantined.

## Current Architecture

- **Registry/SCF:** one abstraction-to-implementation source of truth, content-bound
  identities, route authority, deprecation, derivative invalidation, and module
  ownership are enforced by the project registry.
- **VIEA execution spine:** intent, command, context, job, plan, authority, artifact,
  claim, evidence transition, residual, and failure-boundary records are materialized
  and validated.
- **VCM:** stable semantic objects, typed temporal relations, hybrid sparse/graph
  retrieval, lifecycle transactions, compaction, deletion closure, and deterministic
  migration/replay are wired. Native KV/prefix-cache parity is not claimed.
- **Verifier/search:** bounded propose/verify/repair search is registered with typed
  budgets, independent integrity, sanitized feedback, replay, and separate learned,
  deterministic, and tool-assisted channels. It has not yet produced a behavior win
  on the clean model-only wall.
- **Octopus/MoECOT v8:** the 10,347,009-parameter shared trunk completed 227,189,939
  optimizer positions in 34,303 steps on MLX. Checkpoint `3819f966...` and optimizer
  `1d45dadd...` verify. All five 430,849-parameter specialists then completed against
  that immutable trunk with independent checkpoint/optimizer lineage and no cross-arm
  mutation. English is `1/128` exact; all code arms are `0` exact. Earlier standalone,
  v6, and v7 rungs remain negative evidence in Git/reports rather than active plans.
- **Dense falsification controls:** the active and total controls are fixed at
  10,779,648 and 12,499,968 parameters. The active control is currently training with
  atomic step-specific model/optimizer receipts; the total control follows. Both use
  the same frozen stage and direct per-language exact-recovery contract as MoECOT.
  Their v8 result remains diagnostic until unchanged checkpoints run through the
  separately frozen functional utility qualification.
- **Tokenizer/data correctness:** all corpus categories use exact reversible text
  streams. The frozen stage contains 452,782 non-overlapping windows, zero admitted
  unknown positions, zero public training rows, zero external inference calls, and
  zero fallback/template/router/tool generation credit.
- **Deterministic tools:** exact tools are evidence-producing instruments, not learned
  generation. `UNKNOWN`, `UNSOLVED`, and typed faults are preferred to fabricated
  answers.
- **Procedural memory:** three metadata workflows are adopted with lifecycle and
  ambiguity guards; they cannot claim model capability.
- **Authority/effects:** A2 and E1 are replayable-reference-backed for one bounded
  local route-authority filesystem effect. An independent audit checks declared
  inventory, distinct proposer/observer/evaluator roles, content and route identity,
  material observation, exact rollback, VIEA linkage, no-cheat counters, and eight
  adversarial mutations. This is not general tool-safety or deployed-security proof;
  wider SCIF/authority fixtures remain synthetic.
- **Artifact admission:** checkpoint deserialization is guarded by content identity,
  attestation/advisory policy, anti-rollback, revocation, and custody checks.

## Training State

- The active-parameter dense control is in full training and remains developmental,
  non-routeable, and usable only as frozen falsification evidence. The total-parameter
  control has not started its full run.
- The frozen comparison reports equal-unique-position/total-parameter and
  equal-active-parameter/active-compute views together. Neither accounting view may
  be selected after results are known.
- A corrected causal-transformer training/evaluation contract enforces transitive
  family-disjoint lineage, target-independent interfaces, source/target vocabulary
  separation, content binding, and decode-ABI checks.
- Prefix-LM was tested under a parameter-neutral control and was not adopted.
- A matched teacher-on/off ablation did not improve behavior and was not adopted.
- Preference/DPO improved its objective while worsening candidate emission; it was
  not adopted. RL/DPO must wait for a nonzero verifier-positive numerator.
- Incremental MLX decode reuses source encoding and beam KV state with numerical and
  candidate-identity regression tests. This is a local speed improvement, not
  CUDA/MLX parity evidence.

## Public Calibration

- Historical broad public results include `7/320` and `1/320`; neither measures the
  current clean MLX checkpoint.
- A `45/64` single-card diagnostic is not a learned-model headline: 44 passes came
  from a private n-gram body route and one from the full-body token route.
- The current clean v8 development result is English `1/128` and all code arms `0`
  exact. It is not comparable to the historical public surfaces and cannot be blended
  with assisted routes.
- The next public calibration should occur once after a material, frozen model/data
  change and must report model-only and assisted channels separately.

## Roadmap Order

1. Keep repository/registry/effect evidence coherent and compact.
2. Keep the frozen data/model scaling contract and heldout/stop criteria fail-closed.
3. Finish both dense controls under the unchanged v8 evidence contract and publish
   per-language exact-recovery diagnostics under both accounting views.
4. Freeze and run source-disjoint functional qualification on every unchanged
   checkpoint. Select and confirm an architecture only from task-appropriate utility;
   exact recovery remains separate.
5. If sparse and both dense controls are functionally zero across code, falsify the
   10.8M rung and require task-complete data plus a preregistered 50M-100M active MLX
   rung. Do not spend another cycle on a same-scale architecture patch.
6. Require nonzero clean behavior before enabling STS/VCM/search/preference/RL as
   causal amplifiers; ablate every amplifier under equal budgets.
7. Integrate Question-Compiled Semantic Addressing only through the registered VCM
   and planning abstractions, with leakage and route-authority controls.
8. Calibrate once on fresh frozen public surfaces after material model improvement.
9. Route the winning model into assistant dogfood; use accepted, missed, ignored,
   corrected, and completed outcomes to drive the governed improvement flywheel.

## External and Deferred Work

- Real multi-node Hive proof remains Phase 9 external-environment evidence and waits
  for another trusted reachable node. It does not block local model work.
- Mobile, Watch, spatial, voice-following, NAS, and broader product surfaces remain
  deferred by the breadth freeze unless they directly serve the daily-use lane.
- CUDA/Metal parity cannot be claimed on this Mac without comparable external
  hardware evidence. MLX performance can and should be improved locally.

## Canonical Checks

```bash
python3 scripts/procedural_memory_route_adoption_gate.py
python3 scripts/governance_rights_receipt_suite.py
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
python3 scripts/theseus_project_registry.py --gate
python3 scripts/theseus_control_plane.py
```

## Next Falsifiable Action

Finish both dense controls without changing v8 and publish their exact-recovery
diagnostics. Freeze the architecture-neutral functional manifest before completed
control outputs are inspected, then run unchanged sparse/dense checkpoints through
blind English utility and per-language code execution/toolchain checks. Select a
practical route only from that evidence. If every code path is functionally zero,
close the 10.8M rung and prepare a task-complete, data-supported 50M-100M active MLX
rung with matched controls rather than adding search, routing, or another tiny patch.
