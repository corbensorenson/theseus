# BabyLM And Parameter Golf Transfer

Status: background transfer strategy. Use `docs/PROJECT_STATE.md` for the
current BabyLM frontier, seed55 gate state, and promotion blockers.

This machine already has two relevant external workspaces:

```text
C:/Users/corbe/Documents/babylm-candidate
C:/Users/corbe/Documents/golf
```

These are not side quests. They are the best immediate pressure tests for SymLiquid because they are compact-language-model competitions with real evaluation harnesses, strict accounting, and a record of negative results.

## Why These Matter

SymLiquid's CGS claim is:

```text
compact state -> expansion rules -> symbolic/residual memory -> verified governance
```

BabyLM tests whether that idea improves language learning under severe data limits. Parameter Golf tested whether compact architectures can compress language under strict artifact and training-time limits.

The shared theme is not "build a giant foundation model." It is:

```text
win by using compact structure better
```

## Local BabyLM Snapshot

The BabyLM workspace already has:

- official BabyLM 2026 eval cloned at `external/babylm-eval`;
- local eval commit pinned to `467793f`;
- official data and cached public eval pieces;
- HF-compatible custom model families;
- candidate cards through `C177`;
- ledgers for experiment and candidate history;
- a stable decoder recipe;
- many structured-denoising, token-interface, entity-memory, and grammar-auxiliary experiments.

Important current local results from the workspace docs:

| Family | Status |
| --- | --- |
| Decoder control | strong baseline, full-available zero-shot around `51.98` |
| C120/C130/C135 family | robust local floor around fast average `53.54` and full-available zero-shot `52.48` |
| C133 / C153 entity-slot branches | interesting single-seed gains but seed fragile |
| C170 grammar-core auxiliary | useful 2026-task signal but not promoted because entity balance regressed |
| WordPiece scratch lane | promising architecture evidence, not yet robust enough |

The crucial lesson: late overlays often move the model into a BLIMP/EWoK-friendly but supplement/entity-weak basin. New SymLiquid ideas should be trained early or introduced as tightly bounded modules, not casually grafted onto a saturated checkpoint.

## Local Parameter Golf Snapshot

The Parameter Golf archive contains:

- official control repo and records;
- final under-16MB artifacts;
- run logs and promotion reports;
- route-aware recurrence work;
- MirrorLoop/LexLoRE/HRC notes;
- quantization and export-honesty lessons.

The official challenge optimized bits-per-byte on FineWeb under a 16MB artifact and 10-minute 8xH100 training constraint. The local review concluded that its strongest transfer to BabyLM is methodological:

- build a strong boring baseline first;
- trust official-shaped evals over proxy loss;
- use staged promotion ladders;
- preserve negative results;
- treat recurrence as route/control geometry;
- spend capacity at the token interface before broad trunk widening;
- keep artifact, tokenizer, and data accounting exact.

## SymLiquid Transfer Hypotheses

The right move is not to drop the current Rust prototype into BabyLM as-is. BabyLM needs Hugging Face-compatible language-model checkpoints and official eval outputs.

Translate SymLiquid into BabyLM as narrow controlled deltas:

| SymLiquid component | BabyLM translation |
| --- | --- |
| VSA memory | exact or soft role/entity slot memory, repeated-token cleanup, symbolic trace probes |
| liquid state | route-aware recurrence with learned update gates, not generic extra recurrence |
| reservoir expansion | cheap hidden expansion or depth reuse before symbolic/entity recompression |
| KAN-lite | train-only morphology/grammar feature auxiliaries or compact edge-function adapters |
| FEP/residual accounting | adaptive hard masking, residual-driven curricula, abstention/inspection diagnostics |
| CGS verification | official BabyLM eval, seed validation, candidate ledgers, promotion gates |

## Immediate Plan

1. Keep `babylm-candidate` as the BabyLM competition workspace.
2. Keep this Rust SymLiquid repo as the architecture/kernel/CGS research workspace.
3. Add a thin bridge: a BabyLM candidate card for "SymLiquid-BabyLM" that maps VSA/liquid/reservoir/KAN/FEP to HF-compatible modules.
4. Run the first SymLiquid-derived BabyLM candidate against the stable C120/C130/C135 floor, not only the weaker decoder control.
5. Use public official eval first: fast zero-shot, full available zero-shot, then finetune.
6. Require seed validation before promotion.

## First Candidate To Try

The first serious candidate should be:

```text
C178_symliquid_entity_residual_core
```

Design:

- start from the stable C120/C130-style substrate;
- add an inference-visible but zero-neutral entity/role memory lane;
- train it earlier than the failed late overlays;
- use tokenizer-local repeated-entity and grammar features only;
- keep a clean causal scoring path;
- add a residual gate that only writes when confidence/evidence is high;
- include a no-memory ablation and a train-only-memory ablation.

Promotion gate:

```text
beat C120/C130/C135 on fast average,
preserve or improve Entity Tracking,
do not regress supplement score,
pass at least 3 seeds,
then run full available zero-shot and finetune.
```

## Second Candidate To Try

```text
C179_symliquid_route_recurrence
```

Design:

- translate liquid/reservoir dynamics into route-conditioned recurrent transformer blocks;
- use pass/phase embeddings from the golf MirrorLoop lessons;
- compare equal-parameter and equal-compute controls;
- keep attention concentrated at shell/route-entry blocks;
- measure whether virtual depth helps data efficiency.

Promotion gate:

```text
must beat matched non-recurrent control under official fast eval,
must not create entity/supplement collapse,
must show a clear ablation effect for route/pass conditioning.
```

## What Would Count As "Architecture Is Good"

For BabyLM, SymLiquid is good only if it clears one of these bars:

1. Better official aggregate than the C120/C130/C135 floor.
2. Same aggregate with better Entity Tracking and no supplement collapse.
3. Same score with fewer parameters or less training.
4. Better seed stability than current fragile high-signal branches.
5. A clear task-family lift that is explainable through CGS residual accounting.

Anything else is just an interesting architecture note.

## Next Engineering Step

Add a bridge script in `babylm-candidate` that can:

```text
read a candidate card
run fast official zero-shot
run full available zero-shot when promoted
run finetune when promoted
append candidate_registry.csv
append experiment_ledger.csv
emit a promotion decision
```

That script matters more than another model idea. It turns the search into a ratchet.
