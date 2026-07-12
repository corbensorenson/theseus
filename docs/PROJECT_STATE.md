# Project State

Last consolidated: 2026-07-12 UTC.

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
- **Practical model lane:** the intended seed is five independently trained
  Octopus/MoECOT arms: English, Python, JS/TS, HTML/CSS, and Rust. The existing dense
  6.6M-parameter MLX transformer is the matched mixed-control arm, not the default
  product architecture. Search, preference/RL, and fast generation remain amplifiers.
- **Capability wall:** the current clean family-disjoint model-only diagnostic is
  `0/24`. The model can emit integrity-clean candidates but does not yet reliably
  learn prompt/signature-conditioned state transitions, operands, algorithms, and
  final-return semantics.
- **Data gate:** the frozen contract selects 6,623,232 active parameters and a
  132,464,640 unique-position floor (20:1). The canonical mixed-corpus receipt is
  `GREEN` after file-level quality filtering at 247,908,698 unique positions, or
  37.43 per dense-control parameter, with zero integrity hard gaps. It contains
  77,351,958 English positions, including an 18,947,648 conversation/instruction
  subset, and 170,556,740 code positions: 43,105,570 Python, 52,685,801 JS/TS,
  13,257,091 HTML/CSS, 57,626,478 Rust, and 3,881,800 other code. Generated,
  vendored, minified, decode-damaged, low-diversity, and invalid sources receive no
  credit. Every total/domain/language/evidence minimum still passes.
- **Immediate dependency:** freeze language-appropriate supervised and direct
  family-disjoint evaluation contracts before spending the full training budget;
  then complete bounded pretraining for the five MLX arms and two dense controls.

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
- **Octopus/MoECOT:** the seed contract now defines separate English, Python, JS/TS,
  HTML/CSS, and Rust weights/data/checkpoint lifecycles and typed single, sequential,
  parallel, verification, and adjudicated routes. All five 1.211M-parameter arms,
  the 6.623M total-control, and the 1.211M active-control have distinct bounded MLX
  model/optimizer receipts; Python resume is content-bound and replayed. All remain
  incomplete and `NOT_EVALUATED`. Route success cannot count as answer success and
  hidden generalist fallback is forbidden.
- **Tokenizer correction:** the canonical pretraining stage no longer routes every
  language through Python body tokenization. All six corpus categories use exact
  reversible text streams; 18,004 selected documents prove their category/profile
  binding, zero round-trip failures occurred, and no unknown token position entered
  training. The prior seven smokes were rejected by changed plan/stage/range identity
  and replaced against stage `2fc4d283...`. This is data-path correctness only.
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

- The current dense checkpoint is developmental, non-routeable, and retained only as
  the mixed falsification control.
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
- The current comparable model-only wall is therefore the private family-disjoint
  `0/24`, not a narrated blend of assisted routes.
- The next public calibration should occur once after a material, frozen model/data
  change and must report model-only and assisted channels separately.

## Roadmap Order

1. Keep repository/registry/effect evidence coherent and compact.
2. Keep the frozen data/model scaling contract and heldout/stop criteria fail-closed.
3. Consume the licensed, quality-filtered canonical corpus through separate English,
   Python, JS/TS, HTML/CSS, and Rust MLX arms, with independent optimizer/checkpoint
   lineage, while training the mixed dense control under the same evidence contract.
4. Require nonzero clean behavior before enabling STS/VCM/search/preference/RL as
   causal amplifiers; ablate every amplifier under equal budgets.
5. Integrate Question-Compiled Semantic Addressing only through the registered VCM
   and planning abstractions, with leakage and route-authority controls.
6. Compare the five-arm MoECOT seed with the dense control under matched total/active
   parameters, data, compute, and verifier budget. Keep the winning evidence-backed
   topology; SymLiquid remains a later protected comparator.
7. Calibrate once on fresh frozen public surfaces after material model improvement.
8. Route the winning model into assistant dogfood; use accepted, missed, ignored,
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

Freeze arm-specific supervised and direct family-disjoint evaluation contracts,
including generator-visible field boundaries and independent language validators.
Then complete bounded English, Python, JS/TS, HTML/CSS, Rust, total-control, and
active-control pretraining under the frozen plan. Success requires durable
provenance, heldout lineage, nonzero independently verified arm behavior, and an
honest two-view modular-versus-dense verdict. If those conditions fail, retain the
falsification; do not hide it with routing accuracy, loss, another auxiliary head,
or a nearby green report.
