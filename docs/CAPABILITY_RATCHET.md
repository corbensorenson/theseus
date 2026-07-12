# Project Theseus Capability Ratchet

This is the concise capability scoreboard for Project Theseus. It reports the
current comparable system, preserves negative results, and separates direct learned
generation from search, tools, renderers, private n-grams, and other assisted routes.
Implementation coverage belongs in `roadmap.md` and
`configs/roadmap_implementation_matrix.json`; historical scores remain in git and
retained reports rather than this live page.

## Governing Rule

```text
architecture readiness != model capability
loss/syntax/candidate count != verifier behavior
assisted success != learned generation
fresh public calibration != training data
negative interventions remain recorded
```

A checkpoint advances only when an independent integrity audit binds its exact
lineage and direct model-only behavior improves on a frozen family-disjoint private
surface. Search, tools, VCM, STS, deterministic repair, and fast-generation modes
have separate score channels. Public benchmarks measure transfer but never provide
training rows, targets, traces, or answer-derived metadata.

## Current Scoreboard

| Measure | Current state | Evidence meaning |
|---|---|---|
| Architecture readiness | GREEN: zero blockers and warnings | governed training is authorized; no capability implied |
| Active flagship | `C1_correctness_rl_and_generator_survival_lane` | data adequacy plus frozen dense MLX baseline |
| Practical model | 6.6M-parameter dense causal transformer on MLX | survival baseline, not promoted serving model |
| Current direct private behavior | `0/24` family-disjoint | main capability wall |
| Clean semantic adaptation | 14 integrity-clean candidates, `0/24` | not adopted |
| Starvation DPO | candidate emission `14 -> 0`, behavior `0/24` | not adopted; preference objective was misleading |
| Standard causal replay | 87 candidates over 22 tasks, 57 independently accepted, `0/24` | syntax/loadability improved; semantics did not |
| Search | architecture wired, current repair replay `0` behavior | wait for a non-zero one-shot proposer |
| Model-only general chat | unavailable | assisted local runtime is not a chat-model claim |
| Production MLX model route | blocked by zero behavior | fail-closed quality decision, not an architecture gap |
| Public transfer for current checkpoint | not yet measured | no cross-route substitution allowed |

## Data Adequacy

Current governed one-pass inventory:

- 35,297 deduplicated code functions;
- 5.58M encoded code positions;
- 13,918 redacted/decontaminated human-contributed conversations;
- 4.29M conversation positions;
- at most 9.87M combined positions before accounting for task/model-path
  differences.

For a 6.6M-parameter from-scratch model, this is at most about 1.5 unique combined
positions per parameter and less than one code position per parameter. It is a
diagnostic corpus, not enough evidence for a strong capability verdict. Repeated
optimizer exposure is reported separately and never relabeled as unique data.

The next training run requires a frozen scaling receipt containing:

- exact active and total parameters;
- unique admitted tokens by source/domain/language;
- optimizer token exposure and epoch/repetition count;
- source and semantic deduplication;
- license/authority and descendant lifecycle;
- private/public/holdout contamination checks;
- Python-first code balance, then JS/TS/HTML/CSS and Rust;
- English conversation, correction, instruction, and tool-use coverage;
- frozen tokenizer, visible-input contract, heldouts, verifier, seed, and compute;
- predeclared stop/adoption/falsification conditions.

Do not scale to a 100M sparse model until available unique data supports that rung
under the same predeclared scaling rule.

## Public Evidence

Historical public results are retained but are not one comparable trajectory:

- `45/64` on one BigCodeBench card: 44 passing tasks came from the private n-gram
  body route and one from the full-body token route. This is not a learned-model
  headline.
- `7/320` on five cards: older full-body candidate route; not the current MLX
  checkpoint.
- `1/320` on a later five-card surface: older candidate route; not the current MLX
  checkpoint.
- old `34/160` and other earlier scores belong to still older assisted/selector
  contracts and cannot establish current model capability.

Fresh frozen public surfaces are available when a materially changed model makes
measurement useful. Exact consumed surfaces, contamination, and result-driven seed
fishing remain prohibited. There is no calendar or monthly-spend lock.

## Architecture And Runtime

The following are architecture/runtime evidence, not model capability:

- registry and stable capability fields are GREEN with zero route blockers;
- VIEA materializes the claim/evidence/execution spine;
- VCM is the context ABI across canonical consumers;
- deterministic tools and bounded verifier-guided search are wired;
- a local assistant canary executed and exactly rolled back one route-authority
  file effect with zero residuals;
- incremental MLX decoding is cache-equivalent and restart-safe;
- the real checkpoint decode canary improved `832 ms -> 500 ms` (`1.664x`);
- a focused generated-prefix hot-loop defect improved `27.96x` with identical
  probabilities;
- the registry reports about 30.8 GiB of generated/build state, which remains a
  retention concern but not the capability flagship.

## Teacher Boundary

The canonical governed teacher is `codex_cli/gpt-5.6-sol`; `high` is the default
reasoning effort and only `medium` or `high` are allowed. Teacher output is
training-time input only through provenance, license, verifier, leakage, holdout,
and teacher-share gates. It is never served at runtime.

The latest matched teacher curriculum was negative: teacher-on and teacher-off both
scored `0/48`, while teacher-on worsened heldout loss, candidate count, starvation,
and verifier reward. Those rows remain retained and governed but are not adopted by
default. More teacher rows are not a substitute for adequate base data or a model
that can use the signal.

## Dependency Order

1. Freeze the current dense MLX architecture and all evaluation contracts.
2. Build the governed data/model scaling receipt and expand unique licensed data.
3. Complete the smallest data-supported dense training rung.
4. Require direct model-only family-disjoint behavior above zero.
5. Qualify VCM/STS conditioning, search, preference learning, RLVR, and fast
   generation with matched ablations, in that order.
6. Test sparse/Octopus and SymLiquid only as challengers against the positive dense
   baseline.
7. Run a fresh public calibration for the materially changed current checkpoint.
8. Promote only on replayable behavior, regression, cost, and no-cheat evidence.

## Canonical Checks

```bash
python3 scripts/roadmap_implementation_gate.py --gate --require-pre-training-ready
python3 scripts/theseus_project_registry.py --gate
python3 scripts/standard_causal_transformer_survival_gate.py --gate
python3 scripts/neural_seed_survival_readiness_gate.py --gate
python3 scripts/training_inference_execution_plan_gate.py --gate
```

Canonical evidence:

- `reports/roadmap_pre_training_architecture_readiness_gate.json`
- `reports/theseus_project_registry.json`
- `reports/standard_causal_transformer_survival_gate.json`
- `reports/neural_seed_survival_readiness_gate.json`
- `reports/training_inference_execution_plan_gate.json`
- `reports/teacher_distillation_gate.json`
- `reports/theseus_assistant_effect_complete_canary.json`

## Next Falsifiable Action

Produce the data/model scaling contract from the current admitted corpus, select the
smallest supportable dense MLX rung, expand only the missing governed data domains,
and train it to the predeclared stop. The result must answer one question: does a
properly data-supported current transformer produce direct verifier-positive
family-disjoint behavior? If yes, amplify it. If no, retain the result and diagnose
data, objective, representation, or capacity before changing architecture.
