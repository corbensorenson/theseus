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
- **Practical model lane:** the seed is a shared transformer trunk plus separately
  checkpointed English, Python, JS/TS, HTML/CSS, and Rust experts. Dense decoder-only
  models remain matched falsification controls. Search, preference/RL, and fast
  generation remain amplifiers, not substitutes for a working proposer.
- **Capability wall:** the completed 3.716M shared trunk plus 995,841-parameter Rust
  source-specialist delta remains `0/60` exact on private development. It improves
  target/source similarity to 0.358/0.382 and emits 60/60 valid outputs, but this is
  diagnostic only. Confirmation is untouched and no checkpoint is routeable.
- **Data gate:** the frozen contract selects 6,623,232 active parameters and a
  132,464,640 unique-position floor (20:1). The canonical mixed-corpus receipt is
  `GREEN` after file-level quality filtering at 247,908,698 unique positions, or
  37.43 per dense-control parameter, with zero integrity hard gaps. It contains
  77,351,958 English positions, including an 18,947,648 conversation/instruction
  subset, and 170,556,740 code positions: 43,105,570 Python, 52,685,801 JS/TS,
  13,257,091 HTML/CSS, 57,626,478 Rust, and 3,881,800 other code. Generated,
  vendored, minified, decode-damaged, low-diversity, and invalid sources receive no
  credit. Every total/domain/language/evidence minimum still passes.
- **Immediate dependency:** preregister a larger active model from the full admitted
  licensed-data budget. The source-specialist rung improved similarity but stayed at
  zero exact, so same-scale objective and expert-scope patches are stopped. Expand
  canonical packing only from admitted source identities and keep confirmation/public
  surfaces out of architecture sizing.

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
- **Octopus/MoECOT:** standalone full-model arms, extra SFT repetition, prefix-LM,
  plain cross-attention, pointer-only repair, and source-conditioned v5 all remained
  0/60 and are retained as negatives. Their successor shares general syntax and
  reasoning in one 3,716,001-parameter encoder/decoder trunk while each language owns
  an independently checkpointed expert. The trunk completed 144,101,816 optimizer
  positions on MLX with exact resume lineage. The first 52,000-parameter Rust expert
  completed 1,164,892 positions against frozen trunk `f727ae56...`. Direct dev is
  still 0/60 exact, 59/60 serialization-valid, 0.309 mean target similarity, 0.341
  source similarity, and 0.840 target-length ratio. A private-dev beam ablation fixed
  short-output bias but not semantics. The v7 successor stores a 995,841-parameter
  source-specialist delta while keeping the trunk byte-identical. It improves target
  similarity to 0.358, source similarity to 0.382, length ratio to 0.884, and
  serialization to 60/60, but remains 0/60 exact. These are diagnostics, not answer
  success. Same-scale repair is stopped; the next action is a licensed-data-bound
  capacity rung.
- **Data-bounded v8 rung:** a fresh content-bound audit measures 683,254,465 unique
  governed positions. The frozen view selects 215,552,020 positions for a
  10,777,601-active-parameter shared encoder/decoder plus one independently
  checkpointed 430,849-parameter low-rank source specialist per language
  (12,501,254 total parameters). All specialists clear the 20:1 owned-data floor;
  HTML/CSS is tightest at 20.536. Matched decoder-only controls differ by +2,047
  active and -1,286 total parameters. Stage `1b3b9c9e...` is GREEN with 452,782
  non-overlapping windows, exact reversible tokenization, no admitted unknown
  positions, and zero public rows/external inference/fallback credit. One-step MLX
  canaries are resumable and measure 3,982 positions/s for the trunk, 4,583 for the
  active control, and 2,933 for the total control. None is complete; these figures
  authorize the long frozen comparison but provide no capability evidence. A
  controlled step-500 interruption then verified atomic generation-based resume:
  exact step-specific model/optimizer hashes committed through one receipt, restart
  advanced to step 501 with 3,799,137 positions retained, and superseded generation
  files were removed. Dry-run plans no longer publish candidate artifacts. The full
  trunk then completed 215,558,499 causal, 2,732,624 source-conditioned, and
  8,898,816 supervised positions (227,189,939 total; 34,303 steps) in 44,573
  seconds. Final losses were 1.282, 1.453, and 1.506. Canonical checkpoint
  `3819f966...` and optimizer `1d45dadd...` verify with no progress generations
  retained. This is completed training provenance, not capability; direct evaluation
  waits for the independently trained language experts.
- **v8 specialist result:** all five 430,849-parameter specialists completed with
  independent checkpoints bound to trunk `3819f966...`. English reaches 1/128 exact
  (0.0078125) with 128/128 valid serialization. Python, JS/TS, HTML/CSS, and Rust
  remain 0 exact; Python is 41/128 syntax-valid, and code-arm target similarities
  range from 0.266 to 0.337. This is the first narrow direct v8 success but not broad
  utility, promotion, or an architecture win. Confirmation is untouched, no arm is
  routeable, and matched dense active/total controls remain required.
- **Tokenizer correction:** the canonical pretraining stage no longer routes every
  language through Python body tokenization. All six corpus categories use exact
  reversible text streams; 38,443 selected documents prove their category/profile
  binding, zero round-trip failures occurred, and no unknown token position entered
  training. The current seven one-step receipts bind stage `45f26ca5...`; data
  identities are scoped to vocabulary/supervision contracts so evaluation-only edits
  do not force corpus rebuilds. This is data-path correctness only.
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

Retain both shared-trunk Rust rungs as `0/60`. Measure how many unique optimizer
positions can be packed from the already admitted licensed corpus without changing
source admission, then preregister one larger shared-trunk/source-specialist rung and
its dense active/total controls at a minimum 16-20 unique positions per active
parameter. Bind model shape, MLX peak-memory/time canaries, checkpoint migration or
fresh-init policy, exact data identities, stop rules, and consumed-development
diagnostics before training. Do not spend confirmation, add synthetic benchmark
analogues, or tune from public results. If the larger rung still has zero direct
behavior, retain the result and reconsider from-scratch bootstrap economics rather
than producing another nearby architecture patch.
