# Project State

Last consolidated: 2026-07-20 UTC.

This is the bounded current-wall page. The machine contract is
`configs/roadmap_implementation_matrix.json`; the forward plan is `roadmap.md`; history
remains in Git, immutable reports, and `docs/archive/`.

## North Star

Build a private, locally trained assistant that Corben uses daily, serves with zero
external inference, improves through governed evidence, and drives accepted live-teacher
share toward zero. Public benchmarks are calibration only.

## Current Verdict

- **First-campaign architecture:** frozen shared encoder-decoder transformer trunk plus
  independent low-rank English, Python, JavaScript/TypeScript, HTML/CSS, and Rust arms.
  Decoder-only dense controls are matched at 57,348,617 active parameters and 67,357,193
  total parameters. The MoECOT path has 57,340,426 active and 67,357,711 total parameters.
- **Training authority:** GREEN. The exact corpus stage, tokenizer/index ABI, model shape,
  checkpoint namespace, optimizer exposure, controls, functional evaluation identity,
  registry, and MLX mechanics canaries agree. KERC and OneCell have machine-checked zero
  first-campaign exposure and cannot silently activate.
- **Data:** 422,334,331 unique model-visible positions, or 7.365 per active parameter.
  The optimizer target is 1,146,808,520 positions over 2.715 planned passes, below the
  frozen four-pass ceiling. Repetition is not counted as unique data. Family totals are
  English 160,196,476; Python 116,242,500; JS/TS 82,769,680; HTML/CSS 17,577,020; and
  Rust 45,548,655.
- **MLX mechanics:** GREEN for the MoECOT active arm and both dense controls. Each full
  model produced finite loss, changed parameters, round-tripped weights and optimizer
  state, and resumed training. Earlier bounded canaries used about 2.3-2.8 GB. The exact
  57M shared-trunk shape peaks near 9.1 GB on this M1. Native MLX grouped-query attention
  is mathematically equivalent to explicit KV tiling and raised the exact eight-step
  trunk canary from 786 to 1,108 target positions/second while loss fell from 10.424 to
  6.592. MLX compilation and batch 24 were rejected for this host after measured
  regressions; eager float32 batch 16 remains the qualified route. The relaunched full
  campaign reached step 50 at 741 positions/second overall (about 792 over steps 26-50),
  so the short canary is not used as the sustained-runtime estimate.
- **Capability:** still unmeasured at the 57M rung. The historical 10.8M rung remains
  falsified for practical utility: MoECOT, dense-active, and dense-total each scored
  0/160 on the frozen functional surface. That result cannot be relabeled as 57M evidence.
- **Evaluation:** a new source-disjoint, exact-once 160-case private functional contract
  is frozen with 32 cases each for English, Python, JS/TS, HTML/CSS, and Rust. It has zero
  overlap with the v8 packet and zero public payloads, hidden verifier access, fallback
  credit, or authored-template/tool/router credit. Capability remains `NOT_EVALUATED`
  until trained checkpoints are scored.
- **KERC:** K0-K3 are banked as high-quality bounded discovery evidence. K4-K8 remain
  incomplete, so the full candidate is `INCONCLUSIVE_IMPLEMENTATION` and deferred to a
  successor campaign. This is not a scientific failure. Its source snapshot and live
  content-addressed evidence replay independently, while the first campaign proves zero
  topology, objective, target, control, and optimizer exposure.
- **Registry:** GREEN with no routing blockers, missing identities, or governance
  violations. The canonical project state is the registry plus the frozen package, not
  generated report volume.
- **Repository hygiene:** the generated-artifact budget is GREEN after reference-aware
  compaction replaced old, unreferenced registry snapshots with verified archive pointers.
  Checkpoint volume remains above its warning target but below its hard ceiling. The
  forward roadmap is still too long; compact its narrative after relaunch without dropping
  matrix obligations or mutating the frozen training contract.

## Binding Next Action

1. Preserve the independently replayed 70-artifact architecture package unchanged.
2. Relaunch the shared-trunk MLX run under sleep prevention with resumable checkpoints.
   The first launch reached step 75 and was stopped before any checkpoint because measured
   throughput implied an infeasible host runtime; it has zero capability credit.
3. Train the five language arms from that exact trunk and both preregistered dense controls.
4. Evaluate all candidates once on the frozen 160-case functional contract without tuning
   from heldout outcomes.
5. Select the practical architecture from direct model-only utility, weak-arm behavior,
   matched compute/data, resource cost, and uncertainty. Routing success, tools, renderers,
   templates, or verifier repair cannot count as model generation.
6. Only after a behavior-positive proposer exists, qualify VCM, STS, search,
   preference/RL, and fast-generation modes through matched causal ablations.
7. Run fresh public calibration only after a material frozen model change; never train on
   public benchmark prompts, tests, solutions, traces, or answer templates.

## Non-Claims

- Mechanics readiness is not learned capability.
- Training loss is not functional utility.
- KERC deferral is not KERC falsification.
- MoECOT routing or arm selection is not answer correctness.
- Externally generated tokens are never served to users.
- Public benchmarks and frozen heldouts are not training data.

## Canonical Evidence

- `reports/neural_seed_50m_scale_preregistration.json`
- `runtime/standard_causal_transformer_scale_v2/canonical_pretrain_capacity_v2.json`
- `runtime/standard_causal_transformer_scale_v2/stage_metadata_v1.json`
- `reports/standard_causal_transformer_57m_stage_materialization.json`
- `reports/moecot_language_arm_training.json`
- `configs/neural_seed_57m_functional_utility_freeze.json`
- `configs/evaluation_history/neural_seed_v8_candidate_packet.json`
- `reports/pretraining_architecture_freeze_package.json`
- `reports/theseus_artifact_budget_gate_current.json`

## Canonical Checks

```bash
python3 scripts/theseus_project_registry.py --gate
python3 scripts/pretraining_architecture_freeze.py --execute-replays
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
```

## Current Wall

There is no remaining architecture rationale for postponing the campaign. The current
wall is empirical and computational: the relaunched native-GQA run is healthy through
step 50, with finite declining loss and about 741 target positions/second overall. The
frozen 1.097-billion-position shared trunk therefore projects to roughly 16-17 days on
this M1 if the later rate holds. Preserve the run through its first checkpoint/resume
proof while qualifying a materially faster numerical or hardware route separately; do
not change the frozen weights in flight. Then train the matched alternatives and decide
whether any candidate produces useful direct behavior. Do not reopen architecture work,
create another benchmark family, or generate a nearby GREEN report in place of that run.
