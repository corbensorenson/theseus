# Project State

Last consolidated: 2026-07-21 UTC.

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
- **MLX mechanics:** GREEN for the MoECOT active arm and both dense controls. Native MLX
  grouped-query attention is equivalent to explicit KV tiling. The durable shared-trunk
  lineage is now step 3,000 and 22,999,779 optimizer positions, with model SHA
  `3a9b04ad...05a7` and AdamW SHA `62e1b52b...5f96`. A same-state 16-update qualification
  measured 1,397.8 eager versus 3,063.2 compiled positions/second, a 2.19x gain, with
  identical mean and final loss. The real 500-update continuation sustained 2,914.2
  positions/second across the checkpoint boundary. Full-batch compilation and
  shape-polymorphic/source-conditioned compilation remain disabled on this 16 GB M1
  because they crossed the efficient memory regime or faulted in native MLX.
- **Inference mechanics:** GREEN for the prompt-only direct decoder acceleration. Batched
  beam advance, device-side admissible-logit ranking, and exact pre-forward pruning
  preserved output and normalized receipt identity on 8/8 arm-covered private prompts.
  The latest canonical run reduced aggregate uncached latency from 223.71 to 23.10 seconds,
  a 9.69x speedup; the slower repeated qualification still measured 8.45x. Seven of eight
  current outputs still failed closed on byte serialization, so this is a mechanics win,
  not a capability claim.
- **Checkpoint and joined-runtime mechanics:** exact tensor-level qualification over all
  197 shared-trunk tensors found safetensors and the current NPZ checkpoint identical.
  Safetensors materialized 3.74x faster in the latest alternating three-load comparison
  while size and save time were effectively unchanged, so it is qualified for a controlled future
  migration rather than silently replacing the durable step-3,000 artifact. The assistant's
  unchanged governed refresh path now reuses command-, input-, output-, and TTL-bound
  receipts: repeated cold/warm comparisons measured 325x-550x, with the latest canonical
  run at 3.69 seconds versus 9.7 milliseconds (about 380x). Missing, changed, expired, or
  failed evidence still reruns fail-closed.
- **Learning signal:** the source-disjoint private-development audit now compares step
  2,500 with step 3,000 over 62,743 target positions. Aggregate teacher-forced loss fell
  from 4.198748 to 4.174738 (0.57%). Python, JS/TS, HTML/CSS, and Rust improved; English
  regressed from 4.299828 to 4.318473. This is training feedback, not direct utility.
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
- **Repository hygiene:** the forward roadmap is now a compact execution map backed by
  the complete machine-readable matrix, down from 3,756 lines without deleting an open
  obligation. Reference-aware retention replaces old, unreferenced registry snapshots
  with verified archive pointers. Checkpoint volume remains above its warning target but
  below its hard ceiling. The superseded strict-generator/code-LM route is being decoupled,
  but cannot be honestly retired until its registered successor passes the same functional
  contract or an explicit evidence-preserving demotion is recorded.

## Binding Next Action

1. Preserve the independently replayed 70-artifact architecture package unchanged.
2. Preserve the exact step-3,000 shared-trunk lineage. The compiled microbatch and direct
   decode accelerations are qualified; migrate model snapshots to safetensors only through
   the registered lineage path, and continue measuring publication and residual KV-cache
   cost without changing model math or objective.
3. Make the declared pilot/review ladder executable, then train the five language arms
   and both preregistered dense controls through matched successive-halving reviews.
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
- `reports/resource_acceleration_qualification.json`
- `reports/theseus_artifact_budget_gate_current.json`

## Canonical Checks

```bash
python3 scripts/theseus_project_registry.py --gate
python3 scripts/pretraining_architecture_freeze.py --execute-replays
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
```

## Current Wall

There is no remaining architecture rationale for postponing empirical learning. Training
is at least 2.19x faster under matched same-state canaries, direct decode is 8.45x-9.69x
faster with 8/8 exact parity, and unchanged governed assistant refresh is 325x-550x faster,
but direct
functional utility is still unmeasured and current step-3,000 generation usually fails
closed on byte serialization. The current wall is therefore semantic/serialization
learning quality plus time to the first defensible architecture decision. Make the
pilot/review ladder executable, preserve weak-arm evidence, and finish resident/KV/runtime
measurement while continuing the exact lineage. The 100x aspiration applies to
end-to-end time-to-feedback through kernels, early stopping, caching, reuse, and
independent parallelism, not an unsupported raw-compute claim.
