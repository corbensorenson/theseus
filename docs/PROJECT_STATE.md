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
  registry, and MLX semantic-mechanics canaries agree. KERC and OneCell have machine-checked zero
  first-campaign exposure and cannot silently activate.
- **Data:** 422,334,331 unique model-visible positions, or 7.365 per active parameter.
  The optimizer target is 1,146,808,520 positions over 2.715 planned passes, below the
  frozen four-pass ceiling. Repetition is not counted as unique data. Family totals are
  English 160,196,476; Python 116,242,500; JS/TS 82,769,680; HTML/CSS 17,577,020; and
  Rust 45,548,655.
- **MLX mechanics:** GREEN for the MoECOT active arm and both dense controls. Native MLX
  grouped-query attention is equivalent to explicit KV tiling. The durable shared-trunk
  lineage is now step 3,000 and 22,999,779 optimizer positions, with safetensors model SHA
  `606640cd...5c0e` and AdamW SHA `62e1b52b...5f96`. The adopted microbatch-four route's
  latest same-state three-pair qualification measured 1.86x pooled over eager while reducing
  peak MLX memory from about 8.10 GB to 3.40 GB. The real 500-update continuation sustained
  2,914.2 positions/second across the checkpoint boundary. A stricter microbatch-eight
  rerun compared all 54,836,746 final parameters after each 24-update route: maximum
  absolute drift was at most `2.38e-7`, relative L2 drift was about `8e-8`, and loss stayed
  within `2e-6`, but speed varied from 1.36x to 1.97x and measured only 1.52x median/1.62x
  pooled with 5.00 GB peak MLX memory. It is semantics-qualified but performance-rejected.
  Full-batch compilation was slower and reached 8.57 GB. The latest
  bf16-compute/fp32-master rerun remained finite with fp32 authority but measured 0.955x
  median/0.957x pooled and did not clear its peak-memory gate. It remains
  unadopted on this M1. Deferring all four compiled microbatch synchronizations to the final
  update also preserved all 54,836,746 parameters and exact reported loss, but regressed to
  1.12x median/1.16x pooled and raised compiled peak memory to 7.86 GB. Keep per-microbatch
  synchronization; graph deferral is rejected for this backend/model.
  A bounded size sweep found microbatch six best (`1.86x` median, `1.85x` pooled),
  while sizes five and seven measured `1.64x` and `1.31x`; none cleared `2x`.
  Full-model trials also rejected fused MLX RoPE (`1.24x`), parameter-preserving
  fused QKV/SwiGLU projections (`1.57x`), a reusable 512-position RoPE basis
  (`1.64x`), and one monolithic accumulation/update graph (`1.53x`). Each retained
  bounded update integrity but lost end-to-end throughput, so the original
  implementations remain canonical.
- **Inference mechanics:** GREEN for the prompt-only direct decoder acceleration. Batched
  beam advance, device-side admissible-logit ranking, and exact pre-forward pruning
  preserved output and normalized receipt identity on 8/8 arm-covered private prompts.
  The latest canonical run reduced aggregate uncached latency by 9.51x. Seven of eight
  current outputs still failed closed on byte serialization, so this is a mechanics win,
  not a capability claim. The deferred KERC decoder now shares the optimized beam path
  and passes serial/optimized token-path parity; full KERC pipeline throughput is not claimed.
  A shared-cache indexed-gather variant was exact but only 1.004x pooled. Bounded
  sequence-axis preallocation was only 1.009x on a 512-token stress run. Both were removed.
  The governed resident MLX runtime now loads the registered model once, keeps bounded
  model-local KV prefill and deterministic completion caches, and is wired behind the
  existing OpenAI-compatible service. On one private prompt it preserved exact output/token
  identity, measured 1,895.40x prompt-prefill reuse and 5,482.24x repeated completion reuse,
  and failed closed when production serving was requested. Production remains disabled
  until a direct model-only capability result grants runtime authority.
- **Checkpoint and joined-runtime mechanics:** exact tensor-level qualification over all
  197 shared-trunk tensors found safetensors and the prior NPZ checkpoint identical.
  The registered migration is now committed: the canonical checkpoint is
  `weights.safetensors` (`606640cd...c0e`), its tensor manifest remains
  `5ad2ec6b...1bfa`, all 22,999,779 optimizer positions and the exact AdamW state are
  unchanged, and the old NPZ was removed only after replay and resume validation.
  Safetensors materialized 4.76x faster in the latest alternating three-load comparison; size
  and save time were effectively unchanged. The assistant's
  unchanged governed refresh path now reuses command-, input-, output-, and TTL-bound
  receipts: the latest canonical comparison measured about 372x. Missing, changed, expired, or
  failed evidence still reruns fail-closed.
- **External speed-audit disposition:** the suggested beam batching, device-side admissible
  ranking, compiled train step, bf16 trial, and bounded KV preallocation were already present
  or already measured above. The alleged wide-sequence batch-1 collapse does not affect the
  measured shared trunk (fixed width 512); the cited KERC evidence also remained below the
  8K batch-two boundary, with an observed maximum width of 4,242. KERC's roughly 23
  positions/second remains a real separate
  hot path caused by long source-conditioned computation plus learned residual, verifier, and
  decision objectives, but KERC is deferred from the first executable campaign and is not the
  practical training blocker. Isolated kernel wins were tested at full-model scale and did not
  predict end-to-end training wins.
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
- **System gates:** the independently replayed architecture package is GREEN over 70
  artifacts, 13/13 architecture contracts, and 10/10 replay commands. Roadmap pretraining
  readiness has zero blockers; its only warning is that the live AI-book worktree differs
  from the pinned committed snapshot. ATTD is RED only on source shape:
  `standard_causal_transformer_survival.py` is 7,374 lines against the 7,000-line cap.
  The control plane remains RED from stale historical control reports and unresolved model
  capability/promotion state, not from the acceleration integration.
- **Repository hygiene:** the forward roadmap is now a compact execution map backed by
  the complete machine-readable matrix, down from 3,756 lines without deleting an open
  obligation. Reference-aware retention replaces old, unreferenced registry snapshots
  with verified archive pointers. The cumulative retention gate independently decodes and
  rehashes 2,114/2,114 archived payloads. Its canonical report was compacted from 28.5 MB
  to 22.6 KB by committing all outcomes under one digest plus bounded samples and aggregate
  VIEA records; the hot-report budget is GREEN at about 1.01 GB. Checkpoint volume remains
  above its warning target at about 9.70 GB but below its hard ceiling. The superseded
  strict-generator/code-LM route is being decoupled,
  but cannot be honestly retired until its registered successor passes the same functional
  contract or an explicit evidence-preserving demotion is recorded.

## Binding Next Action

1. Preserve the independently replayed 70-artifact architecture package unchanged.
2. Preserve the exact step-3,000 shared-trunk lineage. The compiled microbatch and direct
   decode routes preserve exact reference behavior; direct decode clears its speed gate,
   while compiled training still needs a robust 2x result. Safetensors migration and
   resident prompt/completion reuse are now qualified and registered. Continue measuring
   publication cost, uncached decode, and multi-request scheduling without changing model
   math or objective; do not enable production model serving before direct utility is positive.
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
is 1.86x faster under the latest pooled matched same-state canary, direct decode is 9.51x
faster with 8/8 exact parity, and unchanged governed assistant refresh is 372x faster,
but direct
functional utility is still unmeasured and current step-3,000 generation usually fails
closed on byte serialization. The current wall is therefore semantic/serialization
learning quality plus time to the first defensible architecture decision. Make the
pilot/review ladder executable, preserve weak-arm evidence, and finish resident/KV/runtime
measurement while continuing the exact lineage. The 100x aspiration applies to
end-to-end time-to-feedback through kernels, early stopping, caching, reuse, and
independent parallelism, not an unsupported raw-compute claim.
