# Standalone Parity Plan

Status: retired historical parity and command reference, moved to
`deprecated/docs/background/` on 2026-06-03. Use
`../../../docs/PROJECT_STATE.md` for current frontier, gates, and
resource-governor state before running or quoting results from this file.

SymLiquid should stand on its own. We do not run proprietary or third-party
models through our harness. If public scores exist, we cite them as external
baselines and compare only when our task contract matches theirs.

## Policy

```text
No external inference.
No provider API calls.
No paid model judging.
No hidden teacher model.
No benchmark contamination.
```

Allowed:

- local SymLiquid training and evaluation;
- local classical baselines;
- local open-weight baselines only if weights are explicitly part of the experiment;
- public leaderboard scores as citations or metadata;
- BabyLM/Parameter Golf style benchmark contracts when run locally.

## Immediate Goal

Train and evaluate SymLiquid on held-out generated suites:

```bash
cargo run --release -p symliquid-cli -- train-standalone --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --batch-size 1 --hv-dim 4096 --model-out reports/symliquid_policy_500x30_hv4096.json --out reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
```

GPU readout path:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 100 --epochs 10 --samples-per-launch 32 --hv-dim 4096 --lr 0.05 --out reports/symliquid_cuda_sgd_100x10_hv4096.json
```

The current standalone path uses:

```text
text observation
  -> hash/VSA-style sparse hypervector features
  -> structured CGS/VSA answer, evidence, pairwise, and verifier features
  -> local linear readout
  -> exact verifier scoring
```

Symbolic fallback is disabled by default for the learned-transfer path. This is
a first serious local model, not a final architecture. The separate
`benchmark-symliquid` command remains a symbolic-governance sanity check.

Current measured synthetic-contract result:

```text
Train seed: 0
Eval seed: 10000
Cases per task: 500
Held-out cases: 5000
Accuracy: 0.997
Residual: 0.003
Invalid action rate: 0.000
External inference calls: 0
Local text-only baseline: accuracy=0.774 residual=0.226
Lift: +0.222 accuracy, -0.222 residual
Report: reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
Comparison: reports/compare_text_baseline_vs_symliquid_cached_sgd_500x30_runtime.json
Policy artifact: reports/symliquid_policy_500x30_hv4096.json
```

Current measured CUDA readout result:

```text
Train seed: 0
Eval seed: 10000
Cases per task: 100
Held-out cases: 1000
Accuracy: 0.787
Residual: 0.213
Invalid action rate: 0.000
External inference calls: 0
Train examples/sec: 2312.7
CPU same config: accuracy=0.787 residual=0.213 train_examples/sec=1981.4
Local text-only baseline: accuracy=0.631 residual=0.369
Lift: +0.156 accuracy, -0.156 residual
Report: reports/symliquid_cuda_sgd_100x10_hv4096.json
Comparison: reports/compare_baseline_vs_symliquid_cuda_100x10_hv4096.json
```

## Benchmark Families

The local hard suite includes:

```text
role_filler
long_context_role_filler
active_classification
gridworld
missing_evidence_rag
code_repair_verifier
babylm_minimal_pair
blimp_acceptability
long_context_retrieval
adversarial_rag
```

## Public Score Comparison

Public scores belong in:

```text
configs/public_baselines.toml
```

A public score is comparable only if:

1. the benchmark is the same;
2. the split is the same;
3. the evaluation script is the same;
4. the data budget is the same;
5. the metric is the same;
6. no external inference is used by our run.

## Local Competition Artifacts

Found on this machine:

```text
C:\Users\corbe\Documents\babylm-candidate
C:\Users\corbe\Documents\golf
```

Use these as local benchmark/evaluation sources, not as external model
providers.

## Commands Added

Local baseline reports:

```bash
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline bag_of_words --out reports/local_bow_baseline_report.json
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline hash_readout --epochs 10 --hv-dim 2048 --out reports/local_hash_readout_baseline_report.json
cargo run --release -p symliquid-cli -- train-baseline --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --hv-dim 4096 --out reports/local_text_hash_transfer_seed0_eval10000_500x30_hv4096.json
cargo run --release -p symliquid-cli -- benchmark-compare --baseline reports/local_text_hash_cpu_100x10_hv4096.json --candidate reports/symliquid_cuda_sgd_100x10_hv4096.json --out reports/compare_baseline_vs_symliquid_cuda_100x10_hv4096.json
```

Seed sweep:

```bash
cargo run -p symliquid-cli -- seed-sweep --train-seeds 0,1,2 --eval-seed-base 10000 --cases-per-task 20 --epochs 10 --hv-dim 2048 --out reports/symliquid_seed_sweep.json
```

BabyLM/BLIMP local probe:

```bash
cargo run -p symliquid-cli -- babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_50k_words.txt" --seed 0 --limit 50 --out-suite benchmarks/snapshots/babylm_local_probe_smoke.json --out-report reports/babylm_local_probe_smoke.json
```

Extended BabyLM probe training:

```bash
cargo run -p symliquid-cli -- train-babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_500k_words.txt" --train-seed 0 --eval-seed 10000 --train-limit 5000 --eval-limit 1000 --steps 5000 --hv-dim 8192 --lr 0.05 --out reports/babylm_probe_train_5k.json
```

Local cached BabyLM BLIMP filtered split:

```bash
python scripts/export_blimp_filtered.py --cache-root "C:\Users\corbe\Documents\babylm-candidate\.cache\huggingface\datasets\BabyLM-community___baby_lm-blimp-filtered" --out-train data/babylm_blimp_filtered_train.jsonl --out-eval data/babylm_blimp_filtered_eval.jsonl --eval-fraction 0.1 --seed 0
cargo run -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_blimp_filtered_eval.jsonl --train-limit 53888 --eval-limit 5987 --steps 800000 --hv-dim 16384 --lr 0.2 --out reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json
```

## Latest Smoke Results

```text
cgs_hard_governance learned transfer, 5000 held-out generated cases:
  local text-only baseline:       accuracy=0.774 residual=0.226
  SymLiquid structured CGS/VSA:   accuracy=0.997 residual=0.003
  lift:                           +0.222 accuracy, -0.222 residual

cuda_readout_sgd, 1000 held-out generated cases:
  local text-only baseline:       accuracy=0.631 residual=0.369
  SymLiquid CPU readout:          accuracy=0.787 residual=0.213 train_examples/sec=1981.4
  SymLiquid CUDA readout:         accuracy=0.787 residual=0.213 train_examples/sec=2312.7
  lift vs text-only baseline:     +0.156 accuracy, -0.156 residual

cgs_hard_smoke, 100 cases:
  SymLiquid symbolic governance: accuracy=1.000 residual=0.000
  local bag_of_words baseline:   accuracy=0.650 residual=0.350
  local hash_readout baseline:   accuracy=0.480 residual=0.520

babylm_local_probe_smoke, 50 corpus-corruption cases:
  SymLiquid current scorer:      accuracy=0.400 residual=0.600

babylm_local_probe, 1000 held-out cases from 500k-word local sample:
  first_allowed baseline:        accuracy=0.492 residual=0.508
  bag_of_words baseline:         accuracy=0.492 residual=0.508
  hash_readout baseline:         accuracy=0.593 residual=0.407
  SymLiquid sequence scorer 1k:  accuracy=0.816 residual=0.184
  SymLiquid sequence scorer 5k:  accuracy=0.814 residual=0.186

local BabyLM BLIMP filtered split, 53,888 train / 5,987 eval:
  contract symbolic scorer:      accuracy=0.500 residual=0.500
  SymLiquid 50k steps:           accuracy=0.746 residual=0.254
  SymLiquid 100k steps + state:  accuracy=0.810 residual=0.190
  SymLiquid 400k + binding/island: accuracy=0.890 residual=0.110
  SymLiquid 800k + binding/island/SVA: accuracy=0.924 residual=0.076
```

Interpretation:

```text
SymLiquid's current symbolic governance layer is strong on explicit contracts.
The learned local sequence scorer is a real improvement over local baselines.
The latest jump came from adding trainable state features for specific BLIMP
residual families rather than from simply widening the vector or balancing
rules. The biggest gains came from reflexive binding, filler-gap/island
structure, and subject-verb head binding.
```

## Next Engineering Steps

1. Wire the CUDA rollout state-update primitive into training for liquid, reservoir, VSA memory, and readout together.
2. Add seed sweeps and confidence intervals for the CUDA readout path.
3. Add reservoir-only, VSA-only, and tiny GRU/RNN local baselines.
4. Replace contract-specific symbolic rules with learned liquid/reservoir/VSA sequence updates.
5. Extend the BLIMP importer/reporting into seed sweeps and per-rule confidence intervals.
6. Build the compiled Puffer/Ocean backend or a Rust FFI bridge for the local Ocean CartPole-style adapter, then move policy/state scoring onto Rust/CUDA.
7. Add Parameter Golf style BPB evaluation only if SymLiquid becomes a language model artifact.
