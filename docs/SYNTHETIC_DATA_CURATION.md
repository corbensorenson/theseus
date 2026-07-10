# Synthetic Data Curation

Last updated: 2026-07-10.

## Candidate Lifecycle Contract

Synthetic curation no longer receives training authority from file-level source
admission alone. `scripts/training_data_lineage_audit.py` writes a compressed,
content-bound candidate receipt ledger at
`runtime/data_governance/data_admission_receipts_v1.jsonl.gz`; the curriculum
and survival-lane materializer accept a row only when its canonical SHA-256 is
present with decision `admit`. Receipts contain hashes and policy metadata, not
raw payload text. Exact/semantic public overlap, heldout splits, fallback
markers, raw-user text, missing licenses, and unverified teacher rows cannot be
admitted.

Current evidence covers `64,196` receipts: `63,892` admitted and `304` heldout
rejections. Recursive synthetic share is `0.824771`, so the lifecycle report is
intentionally `YELLOW`. This warning does not diagnose model collapse. The
five-policy continual-learning comparison is metadata simulation, and the
11-kind deletion closure is graph-propagation evidence rather than physical
model unlearning.

SparkStream now has a governed synthetic-data path. The goal is not to make
bulk synthetic text. The goal is to fold existing data and residuals into a
smaller, higher-signal training blend while keeping provenance, leakage checks,
and promotion gates intact.

## Current Implementation

Primary script:

```powershell
py -3.13 scripts\synthetic_data_curator.py `
  --policy configs\synthetic_data_policy.json `
  --out reports\synthetic_data_curator.json
```

Default outputs:

- `data/synthetic/babylm_residual_targeted_current.jsonl`
- `data/synthetic/babylm_train_plus_synthetic_current.jsonl`
- `data/synthetic/babylm_residual_targeted_current.dataset_card.json`
- `reports/synthetic_data_curator.json`

`data/synthetic/` is ignored by git because it is generated runtime data.

The one-command ratchet runner now prepares the synthetic blend before the
BabyLM frontier run. The train input becomes the blended file, but the eval
input remains the active holdout/frontier selected by policy.

If `reports/training_data_sampler.json` is green, the curator can also include
a tiny low-ratio external pairwise distillation artifact created from
open-license source samples. This is not a bulk corpus path and is not an eval
source.

## Policy

Source of truth:

- `configs/synthetic_data_policy.json`

Defaults:

- local-only generation;
- zero external inference calls;
- no teacher-generated samples for training by default;
- synthetic-only training forbidden;
- synthetic blend capped at 12%;
- governed external pairwise rows capped at 2%;
- exact pair and exact sentence overlap with eval/holdouts must be zero;
- dataset card and provenance required;
- candidate promotion must still pass public comparator, seed49 regression,
  seed55 frontier, residual delta, runtime, CUDA, and synthetic governance gates.

## Data Shape

The first supported family is BabyLM/BLIMP-style minimal pairs:

```json
{
  "sentence_good": "The careful children near the window are ready.",
  "sentence_bad": "The careful children near the window is ready.",
  "rule": "irregular_plural_subject_verb_agreement_1",
  "field": "morphology",
  "linguistics_term": "subject_verb_agreement",
  "source": "symliquid_synthetic_data_curator",
  "generation_policy": "local_only_no_external_inference",
  "training_origin": "synthetic_verified_residual_targeted"
}
```

Governed external rows use the same minimal-pair shape, but carry provenance:

```json
{
  "sentence_good": "The children are ready for the lesson.",
  "sentence_bad": "The children is ready for the lesson.",
  "source": "external_open_sample_pairwise_distill",
  "source_id": "fineweb_edu",
  "license_spdx": "odc-by",
  "generation_policy": "local_rule_corruption_no_external_inference",
  "training_origin": "external_open_sample_pairwise_governed"
}
```

Generation is residual-targeted:

1. Read `reports/residual_escrow.json`.
2. Select high-residual rules/terms.
3. Generate local template and feature-preserving mutation candidates.
4. Reject overlaps with train/eval/holdout/bridge exclusions.
5. Score quality and diversity.
6. Cap per-rule share.
7. Load a green external sampler artifact if present.
8. Cap external rows separately.
9. Write a capped real+synthetic+external training blend.

## Why This Shape

The design follows the current synthetic-data playbook:

- Microsoft Phi/Textbooks showed that high-quality textbook/exercise-style data
  can make small models surprisingly capable:
  https://www.microsoft.com/en-us/research/publication/textbooks-are-all-you-need/
- Cosmopedia emphasizes prompt/topic diversity and duplicate control:
  https://huggingface.co/blog/cosmopedia
- Self-Instruct and LLM2LLM show the value of generated data that is filtered
  and targeted around capability gaps:
  https://huggingface.co/papers/2212.10560
  https://huggingface.co/papers/2403.15042
- Distilling step-by-step shows that rationales/additional supervision can be
  data-efficient for smaller models:
  https://research.google/pubs/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/
- SPIN shows an iterative self-play path, but only after careful grounding in
  seed data:
  https://proceedings.mlr.press/v235/chen24j.html
- Model-collapse work warns against recursive, synthetic-only training:
  https://www.nature.com/articles/s41586-024-07566-y

## RMI Placement

Synthetic curation is an intervention-ladder step between data improvement and
training improvement:

```text
residual escrow
    -> targeted synthetic candidate generation
    -> governed tiny external sample distillation
    -> leakage/diversity/quality gates
    -> capped blend
    -> active frontier run
    -> public/private/regression candidate gates
    -> residual delta audit
```

It is not a replacement for benchmarks, teacher review, or architecture work.
If synthetic data improves the training score but worsens public calibration,
seed49 regression, or residual escrow, the candidate gate blocks promotion.

## Future Extensions

Allowed next extensions:

- teacher-proposed samples stored as proposals, not training rows;
- verifier-generated rationales for accepted rows;
- bridge-benchmark-neighbor examples for specific residual clusters;
- synthetic data ablations: public-only vs public+mutated vs public+synthetic;
- external-sample ablations: no external vs tiny governed pairwise rows;
- reward-model or classifier-based filtering once a local verifier exists.

Still gated:

- bulk web distillation;
- training on copied lookup pages;
- bulk training-data downloads;
- any uncertain-license source;
- candidate-profile use of external samples without teacher/human review;
- teacher outputs used directly as training data;
- synthetic ratio above policy.
