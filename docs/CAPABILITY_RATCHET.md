# Project Theseus Capability Ratchet

This document defines how capability may advance. It intentionally contains no
live score snapshot; use [Project State](PROJECT_STATE.md) for current facts.

## Governing Rule

```text
architecture readiness != model capability
training progress != model capability
loss or syntax != functional utility
assisted success != learned generation
public calibration != training data
mechanics failure != architecture falsification
GREEN requires a named dimension
```

## Evidence Channels

Report these separately:

| Channel | Includes | May support learned-model credit? |
| --- | --- | --- |
| Model-only | Learned generation from allowed visible inputs | Yes, after integrity and evaluation gates |
| Assisted | Tools, VCM, retrieval, rules, routers, templates, or deterministic repair | No |
| Teacher | Governed external training proposal or row | No runtime credit; training provenance only |
| Public calibration | Frozen public benchmark measurement | Transfer evidence only |
| Mechanics | Gradient, parity, reload, throughput, resource behavior | No capability credit |
| Human audit | Prospectively bound review | Supports interpretation, not hidden model input |

## Candidate Requirements

A candidate can enter capability evaluation only when:

- exact implementation, model, data, objective, optimizer, budget, seed, and
  checkpoint identities are frozen;
- candidate output sees no forbidden answer-identifying field;
- independent integrity recomputation passes;
- templates, fallbacks, tools, retrieval, and deterministic repairs receive
  zero learned credit;
- checkpoint/reload and next-update replay pass;
- data provenance, rights, deduplication, contamination, privacy, retention,
  and synthetic-share gates pass;
- the evaluation surface is source-disjoint, frozen, and unconsumed;
- matched controls receive comparable opportunity and cost accounting.

## Adequacy Before Negative Claims

Before retiring or broadly falsifying a mechanism, independently verify:

1. faithful implementation of every causal mechanism in the claim;
2. learnability, gradient flow, overfit, checkpoint/reload, intervention, and
   ablation sanity;
3. strong baselines with matched data, compute, tuning, inference, verifier,
   and total-system cost;
4. task diversity, multiple seeds, uncertainty, weak tails, and adequate
   detection power;
5. source-disjoint heldouts and evaluator independence.

Missing implementation adequacy is `INCONCLUSIVE_IMPLEMENTATION`. Missing
experiment adequacy is `INCONCLUSIVE_EXPERIMENT`.

## Private Functional Gate

The private gate should report:

- model-only task completion;
- weakest-arm utility;
- verifier pass rate;
- empty, malformed, injection, and forbidden-field rejection;
- paired candidate-control effects;
- latency, memory, and total cost;
- uncertainty;
- prospective human-audit receipt;
- exact consumption-ledger state.

Average success cannot hide a failed language arm.

## Public Calibration

Public benchmarks:

- remain calibration-only;
- never provide training prompts, tests, hidden tests, solutions, traces,
  labels, or answer templates;
- use fresh governed surfaces;
- are not rerun after exact consumption;
- occur only after a materially changed candidate has positive private
  functional evidence;
- produce residual categories, not copied training payloads.

## Product Ratchet

Real local use records one outcome:

- accepted;
- missed;
- ignored;
- corrected;
- completed;
- failed;
- abstained.

Product progress requires genuine tasks. Fixture metadata and private 1.0
scores cannot substitute for daily-use evidence. Model-only and assisted
outcomes remain separate.

## Data Ratchet

The corpus changes only in response to a measured functional or coverage
defect. Any new source must pass the corpus gate. Teacher rows remain targeted
residual pressure with a durable accepted-row and optimizer-sampling share.

More rows, reports, or benchmark families are not automatically progress.

## Architecture Decision

For the current MoECOT experiment:

- train the modular candidate and both matched dense controls;
- consume the same frozen private evaluation;
- compare paired functional utility, weakest-arm behavior, latency, memory,
  verification cost, and total lifecycle cost;
- retain MoECOT only if modularity earns a meaningful frontier improvement;
- simplify to the winning dense control if it performs better;
- if all fail, design the smallest successor from observed residuals.

Discovery mechanisms remain protected but cannot delay this verdict.

## Promotion

Promotion requires:

- a materially improved exact checkpoint;
- source-disjoint model-only behavior;
- no integrity, contamination, or authority fault;
- no unacceptable weakest-arm regression;
- acceptable resource and serving behavior;
- exact rollback;
- explicit operator approval.

Promotion never follows merely from:

- a GREEN registry;
- training completion;
- lower training loss;
- syntax/loadability;
- successful deterministic repair;
- public score alone;
- architecture novelty.

## Current State

The current project is `TRAINING_HELD`, `NOT_EVALUATED`, and
`EMPIRICAL_SUPPORT_INSUFFICIENT`. That state is maintained only in
[Project State](PROJECT_STATE.md), not duplicated here.
