# Project Theseus / SymLiquid RMI

Rust-first reference prototype for **Ratcheting Modular Intelligence**:
continuous-state SymLiquid substrate, Octopus-routed specialist arms,
SparkStream autonomy, Theseus Hive distributed runtime, and governed
self-improvement loops.

This repository is a research implementation, not a foundation model and not a
production runtime. It is designed to make the architecture testable through
clean module boundaries, synthetic tasks, ablations, residual accounting,
verification reports, resource-aware CPU/CUDA/MLX contracts, and explicit
governance around autonomy, teacher use, data ingress, licensing, and rented
Hive compute.

## Current Project State

Use `docs/PROJECT_STATE.md` as the live operational source of truth. The
README stays high-level so transfer work does not create two competing status
snapshots.

As of the 2026-06-03 Mac handoff:

- the Windows CUDA workstation state is the source of truth and `main` is
  synchronized with `origin/main`;
- coherence is `GREEN`, candidate promotion is `26/28`, active Code LM workers
  are `0`, and active control-plane leases are `0`;
- public calibration and model growth remain locked while the control plane
  points to `broad_public_transfer_floor_private_repair`;
- the remaining candidate blockers are `broad_public_code_transfer_ready` and
  `maturity_integrity_audit_green`;
- the latest private-only Code LM transfer repair completed `GREEN` with `373`
  private candidates and `0` public candidates. This is private/source-level
  repair evidence only, not public calibration unlock evidence.

Theseus is currently a local Ratcheting Modular Intelligence prototype with a
real Code LM learning lane, SymLiquid substrate, SparkStream autonomy,
Octopus-routed specialist arms, Hive distributed runtime, governed personality
context, report-backed control plane, and explicit CUDA/MLX/CPU execution
contracts. It is not a foundation model or production runtime yet; it is a
research machine with proof, privacy, licensing, teacher-use, data-ingress, and
promotion boundaries.

Read these first:

- `docs/VIEA.md` for the canonical north-star architecture: structured command
  contracts, artifact graph, specialist routing, workflow-to-tool compiler,
  evaluation ratchet, runtime adapters, feedback, and the separate
  student-learning proof layer.
- `docs/PROJECT_THESEUS_WHITEPAPER.md` for a standalone top-to-bottom
  whitepaper on the complete Project Theseus system.
- `docs/TOP_TO_BOTTOM_ARCHITECTURE.md` for the complete operational map of how
  the dashboard, daemon, autonomy cycle, arms, ledgers, teacher, checkpoints,
  data ingress, and Rust/CUDA hot loops connect.
- `docs/PROJECT_STATE.md` for the current live state and next actions.
- `docs/MAC_HANDOFF_2026_06_03.md` before transferring the current Windows
  state to Apple Silicon Codex/MLX.
- `docs/THESEUS_TRAVEL_PARENT_DEMO.md` for the calm MacBook demo path.
- `docs/REPLICATION_GUIDE.md` for a practical setup/runbook that should let a
  new operator rebuild the current local system from the repository plus
  governed data/artifacts.
- `docs/README.md` for the consolidated documentation map.
- `docs/SPARKSTREAM_AUTONOMY.md` for the dashboard, daemon, goals, teacher, and checkpoints.
- `docs/THESEUS_HIVE.md` for the no-terminal setup wizard, the `theseus` CLI,
  phone QR joining, Hive profiles, and the Windows/macOS/Linux Hive app layer.
- `docs/LICENSE_SYSTEM.md` for local registration, free community-use limits,
  signed paid licenses, feature checks, and release caveats.
- `docs/THESEUS_UPDATES.md` for accepted-candidate update offers, soft/hard
  installs, protected arms, dashboard controls, CLI, and Hive propagation.
- `docs/THESEUS_COMPUTE_MARKET.md` for internal work-credit accounting, gas
  quotes, receipt settlement, rented compute, and the guarded public token path.
- `docs/BENCHMAXX_CURRICULUM.md` for the planned capability course from
  SymLiquid/BabyLM through RL, emulator tasks, coding, web/desktop agents,
  native voice, and end-to-end user-agent behavior.
- `docs/CAPABILITY_MATRIX.md` for the current feature/capability matrix and
  market comparison policy.
- `docs/SYNTHETIC_DATA_CURATION.md` for residual-targeted synthetic data and
  model-collapse guardrails.
- `docs/ROM_RL_DATA_GROWTH_LANES.md` for local ROM inventory, emulator RL
  wrappers, and benchmark/data expansion lanes.
- `docs/REAL_TRAINING_PREFLIGHT.md` for the gate before longer training.
- `docs/DATA_AND_ARTIFACTS.md` for what is tracked, ignored, and covered by
  the repository license.

## Architecture

```text
Observation
  -> KAN-lite encoder
  -> liquid continuous-state cell
  -> reservoir expansion
  -> VSA symbolic memory
  -> belief / expected-free-energy layer
  -> task readout or action selection
```

The CGS loop is:

```text
observe -> compress -> expand -> bind -> predict -> act -> correct -> recompress
```

Every task report now includes a CGS accounting block:

| Field | Meaning |
| --- | --- |
| Seed cost | compact state or seed size |
| Rule cost | transition/kernel/model cost |
| Memory cost | persistent state cost |
| Residual cost | error, uncertainty, failed retrieval, or failure rate |
| Verification cost | number or cost of checks |
| Governance cost | action, query, or control cost |
| Generative leverage | target scale divided by compact structure cost |
| CGS quality score | provisional combined diagnostic |

The current prototype includes:

| Component | File |
| --- | --- |
| KAN-lite RBF edge functions | `crates/symliquid-core/src/modules/kan_lite.rs` |
| Liquid continuous-state cell | `crates/symliquid-core/src/modules/liquid.rs` |
| Fixed reservoir memory | `crates/symliquid-core/src/modules/reservoir.rs` |
| Bipolar VSA memory | `crates/symliquid-core/src/modules/vsa.rs` |
| Expected-free-energy layer | `crates/symliquid-core/src/modules/fep.rs` |
| Integrated model | `crates/symliquid-core/src/modules/model.rs` |
| CGS accounting | `crates/symliquid-core/src/cgs.rs` |
| Cognitive Loop Closure | `crates/symliquid-core/src/loop_closure.rs` |
| Backend trait | `crates/symliquid-core/src/backend.rs` |

Concept docs:

- `docs/README.md`
- `docs/PROJECT_STATE.md`
- `docs/CGS.md`
- `docs/PROJECT_THESEUS_WHITEPAPER.md`
- `docs/TRAINING_EVALS_BENCHMARKS.md`
- `docs/BABYLM_PARAMETER_GOLF_TRANSFER.md`
- `docs/COGNITIVE_LOOP_CLOSURE.md`
- `docs/GENESIS_KERNEL.md`
- `docs/REALITY_MANIPULATOR.md`
- `docs/RATCHETING_GENERATIVE_SYSTEMS.md`
- `docs/RATCHETING_MODULAR_INTELLIGENCE.md`
- `docs/OCTOPUS_ROUTER.md`
- `docs/CAPABILITY_RATCHET.md`
- `docs/SPARKSTREAM_AUTONOMY.md`
- `docs/BENCHMAXX_CURRICULUM.md`
- `docs/PUFFERLIB4_RL_LANE.md`
- `docs/CAPABILITY_MATRIX.md`
- `docs/SYNTHETIC_DATA_CURATION.md`

## SparkStream Autonomy Dashboard

SparkStream is the local automation and observability layer for ratcheting runs. It watches the project ledgers, runs bounded training profiles, queues or calls the Codex teacher only when policy allows, checkpoints report state, and serves a live dashboard:

```powershell
.\scripts\start_sparkstream.ps1
```

Then open:

```text
http://127.0.0.1:8787
```

For one-shot cycles:

```powershell
py -3.13 scripts\autonomy_cycle.py --profile smoke
py -3.13 scripts\autonomy_cycle.py --profile inner_loop --execute
py -3.13 scripts\autonomy_launch_readiness.py --profile inner_loop
```

Use the dashboard "Autonomous Goals" panel or:

```powershell
py -3.13 scripts\autonomous_goal_runner.py --goal "Refresh resources and keep the system efficient." --profile smoke
```

See `docs/SPARKSTREAM_AUTONOMY.md` for the full operator guide, teacher policy, benchmark source queue, checkpoints, resource governor, autonomous goals, and safety defaults.

The autonomous loop also maintains a governed resource pantry for public
benchmark/RL/eval source repositories. It uses the spare drive when available,
keeps datasets metadata-only unless the sampler policy approves tiny governed
samples, and feeds adapter/card work without bulk-downloading training corpora.

Start the integrated Project Theseus Hive runtime:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_theseus_hive.ps1
```

On Windows, install the resident tray operator so Theseus is reachable from the
icon area beside the clock:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_theseus_hive.ps1 -InstallTray -StartTray
```

The tray opens the Hive operator/chat UI, dashboard, setup wizard, project
folder, reports folder, and Windows/CUDA doctor, and can start/restart/stop the
local services. The Windows EXE/USB installer installs this tray surface by
default unless launched with `-NoTray`. It uses the generated Theseus icon under
`assets/windows/theseus-hive.ico` and surfaces local notifications for training
improvement, blocked actions, teacher-needed states, promotion readiness, and
CUDA/resource pressure.
Release builds can be Authenticode-signed with
`scripts/package_theseus_windows.ps1 -BuildExe -Sign` once a local signing cert
is configured.

Install the terminal/server CLI, then use one command for status, setup, Hive
joining, device invites, checkpoint chat, and safe task submission:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_theseus_cli.ps1
theseus status
theseus license register --name "Your Name" --usage personal_homelab --seats 1 --accept-terms
theseus setup
theseus hive create --name Home --tier private --mode lan --start
theseus chat "summarize current benchmark status"
theseus openai start
theseus market status
theseus market quote --task-kind cuda_eval_chunk --payload-json "{\"profile\":\"smoke\"}"
```

The Hive node discovers trusted peers and reports CUDA/MLX/CPU capacity through
`reports/hive_status.json`, `reports/hive_peers.json`, and
`reports/hive_scheduler.json`. Register the local install before creating
hives or scheduling worker chunks. Private Hive workers can run bounded real
CUDA and MLX chunks through `scripts/hive_worker_chunk.py`; use
`theseus schedule --worker-chunks` to plan them and
`theseus schedule --execute --worker-chunks` to submit them to authorized
nodes. Scheduler placements include internal gas estimates from the Theseus
Compute Market, and accepted worker receipts settle into local work-credit
accounting under `reports/compute_market_ledger.jsonl`. Remote work is limited to registered task kinds in
`configs/hive_policy.json` and requires `THESEUS_HIVE_SECRET` outside loopback.
`scripts/performance_optimizer.py` refreshes
`reports/performance_optimizer.json` each autonomy cycle so the system can keep
Rust/CUDA on the Windows hot path, prefer MLX on Apple Silicon, and escalate to
the teacher only after local throughput evidence shows a real wall.
Cross-network home/workshop/friend setups use
`scripts/hive_invite.py` plus `scripts/hive_relay.py` or a private VPN. Phones
join first as PWA/operator clients through the relay or dashboard URL.
Company hives, commercial use, and public gateway operation require an imported
signed paid license.

For agent harnesses that expect an OpenAI-compatible local server, use:

```powershell
theseus openai start
```

Then configure the client with base URL `http://127.0.0.1:8789/v1`, model
`theseus-live`, and any placeholder API key unless you enabled a local token.
The shim is local-only: it routes to `scripts/checkpoint_chat.py` and reports
`external_inference_calls=0`.

Benchmark target config:

- `configs/external_benchmarks.toml`
- `configs/local_benchmark_paths.toml`
- `configs/public_baselines.toml`

## Workspace

```text
crates/
  symliquid-core/   CPU reference modules, toy tasks, ablations
  symliquid-cuda/   optional CUDA backend surface and kernel sources
  symliquid-cli/    command-line task runner
  symliquid-bench/  small CPU benchmark harness
examples/           runnable package examples
tests/              root smoke tests
```

## License

First-party project code and documentation are licensed under Apache-2.0. See
`LICENSE` and `NOTICE`.

Third-party datasets, benchmark clones, generated reports, generated
checkpoints, caches, and model weights are not automatically covered by this
license. See `docs/DATA_AND_ARTIFACTS.md`.

## Install

Install Rust with Cargo, then run:

```bash
cargo test
```

CPU-only systems are supported. The core implementation has no external dataset requirements.

## Run Toy Tasks

```bash
cargo run -p symliquid-cli -- role-filler --steps 200
cargo run -p symliquid-cli -- delayed-recall --steps 200
cargo run -p symliquid-cli -- active-classification --episodes 100
cargo run -p symliquid-cli -- gridworld --episodes 100
```

Run ablations:

```bash
cargo run -p symliquid-cli -- ablations --task role_filler --steps 200
cargo run -p symliquid-cli -- ablations --task delayed_recall --steps 200
cargo run -p symliquid-cli -- ablations --task active_classification --steps 100
cargo run -p symliquid-cli -- ablations --task gridworld --steps 100
```

Run examples:

```bash
cargo run --example role_filler
cargo run --example delayed_recall
cargo run --example active_classification
cargo run --example gridworld
```

Run the simple benchmark harness:

```bash
cargo run -p symliquid-bench
```

## Standalone Training And Benchmarking

Train SymLiquid locally on generated hard benchmark suites and evaluate on a held-out seed:

```bash
cargo run --release -p symliquid-cli -- train-standalone --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --batch-size 1 --hv-dim 4096 --model-out reports/symliquid_policy_500x30_hv4096.json --out reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
```

This uses only local synthetic data and a local trained readout over structured CGS/VSA features. Symbolic fallback is off by default for this command. It does not call any external model.

Train the comparable text-only transfer baseline:

```bash
cargo run --release -p symliquid-cli -- train-baseline --train-seed 0 --eval-seed 10000 --cases-per-task 500 --epochs 30 --hv-dim 4096 --out reports/local_text_hash_transfer_seed0_eval10000_500x30_hv4096.json
```

Run the persistent CUDA readout trainer on the same local benchmark family:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 100 --epochs 10 --samples-per-launch 32 --hv-dim 4096 --lr 0.05 --out reports/symliquid_cuda_sgd_100x10_hv4096.json
```

Run CUDA rollout-backed training, where liquid/reservoir/VSA state evolves on
GPU before the readout trains:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 50 --epochs 5 --state-epochs 6 --state-lr 0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_50x5_hv1024_readout_gated_lr002_e6.json
```

Run a CUDA rollout state-training sweep instead of trusting one hyperparameter
point:

```bash
cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda-sweep --train-seeds 0,1,2 --eval-seed-base 10000 --cases-per-task 50 --epochs 5 --state-epochs 0,2,6 --state-lrs 0.0,0.005,0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_sweep.json
```

Latest learned transfer run:

```text
cases=5000
accuracy=0.997
residual=0.003
invalid_action_rate=0.000
external_inference_calls=0
text_only_baseline_accuracy=0.774
accuracy_lift=+0.222
report=reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
comparison=reports/compare_text_baseline_vs_symliquid_cached_sgd_500x30_runtime.json
policy_artifact=reports/symliquid_policy_500x30_hv4096.json
train_examples_per_second=2227.8
```

## Benchmark Harness

Generate frozen benchmark cases:

```bash
cargo run -p symliquid-cli -- benchmark-snapshot --seed 0 --cases-per-task 20 --out benchmarks/snapshots/cgs_hard_seed0.json
```

Run SymLiquid against that frozen suite:

```bash
cargo run -p symliquid-cli -- benchmark-symliquid --suite benchmarks/snapshots/cgs_hard_seed0.json --out reports/symliquid_reference_report.json
```

Run local-only baselines against the same suite:

```bash
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline bag_of_words --out reports/local_bow_baseline_report.json
cargo run -p symliquid-cli -- benchmark-baseline --suite benchmarks/snapshots/cgs_hard_seed0.json --baseline hash_readout --epochs 10 --hv-dim 2048 --out reports/local_hash_readout_baseline_report.json
cargo run -p symliquid-cli -- benchmark-breakdown --suite benchmarks/snapshots/cgs_hard_seed10000_500.json --report reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json --group-by task --out reports/symliquid_standalone_500x30_task_breakdown.csv
```

Experimental larger batch training is available with `--batch-size N`. The
current best-quality local run uses `--batch-size 1`, but features are cached
once per run, so it is faster than the earlier repeated-feature SGD path.

Run a seed sweep:

```bash
cargo run -p symliquid-cli -- seed-sweep --train-seeds 0,1,2 --eval-seed-base 10000 --cases-per-task 20 --epochs 10 --hv-dim 2048 --out reports/symliquid_seed_sweep.json
```

Build and score a local BabyLM/BLIMP-style corruption probe from the BabyLM sample text:

```bash
cargo run -p symliquid-cli -- babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_50k_words.txt" --seed 0 --limit 50 --out-suite benchmarks/snapshots/babylm_local_probe_smoke.json --out-report reports/babylm_local_probe_smoke.json
```

Train the local BabyLM sequence scorer for real feedback:

```bash
cargo run -p symliquid-cli -- train-babylm-probe --input "C:\Users\corbe\Documents\babylm-candidate\data\samples\strict_small_500k_words.txt" --train-seed 0 --eval-seed 10000 --train-limit 5000 --eval-limit 1000 --steps 5000 --hv-dim 8192 --lr 0.05 --out reports/babylm_probe_train_5k.json
```

Export the local cached BabyLM BLIMP filtered Arrow files into disjoint JSONL splits:

```bash
python scripts/export_blimp_filtered.py --cache-root "C:\Users\corbe\Documents\babylm-candidate\.cache\huggingface\datasets\BabyLM-community___baby_lm-blimp-filtered" --out-train data/babylm_blimp_filtered_train.jsonl --out-eval data/babylm_blimp_filtered_eval.jsonl --eval-fraction 0.1 --seed 0
```

Train and evaluate on the exported local BLIMP split:

```bash
cargo run -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_blimp_filtered_eval.jsonl --train-limit 53888 --eval-limit 5987 --steps 800000 --hv-dim 16384 --lr 0.2 --out reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json
```

The scorer reports accuracy, residual, invalid actions, rough token count, runtime, memory bytes, and verified success per cost. Experimental switches such as `--stateful`, `--pairwise-contrast`, and `--balance-rules` are available but are not defaults because current measurements did not improve the best held-out score.

Published scores from other teams or model providers should be entered as public baseline metadata, not rerun through external inference.

Local related benchmark work was found at:

```text
C:\Users\corbe\Documents\babylm-candidate
C:\Users\corbe\Documents\golf
```

The default frozen suite currently includes:

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

Recent smoke measurements:

```text
cgs_hard_governance learned transfer, 5000 held-out generated cases:
  local text-only baseline:       accuracy=0.774 residual=0.226
  SymLiquid structured CGS/VSA:   accuracy=0.997 residual=0.003
  lift:                           +0.222 accuracy, -0.222 residual
  report: reports/symliquid_cached_sgd_500x30_bs1_lr005_runtime.json
  policy artifact: reports/symliquid_policy_500x30_hv4096.json

cuda_readout_sgd, 1000 held-out generated cases:
  local text-only baseline:       accuracy=0.631 residual=0.369
  SymLiquid CPU readout:          accuracy=0.787 residual=0.213 train_examples/sec=1981.4
  SymLiquid CUDA readout:         accuracy=0.787 residual=0.213 train_examples/sec=2312.7
  lift vs text-only baseline:     +0.156 accuracy, -0.156 residual
  report: reports/symliquid_cuda_sgd_100x10_hv4096.json

cuda_rollout_state, 500 held-out generated cases:
  local text-only baseline:       accuracy=0.504 residual=0.496
  SymLiquid CUDA rollout:         accuracy=0.582 residual=0.418 train_examples/sec=131.9
  readout router:                 selected shared_head
  readout probe:                  shared=0.602 task_heads=0.523 worst_task_delta=-0.570
  state update candidate:         rejected by probe gate
  probe metric:                   masked accuracy on 1000 independent synthetic cases
  probe gate:                     base=0.609 candidate=0.592 worst_task_delta=-0.350
  task-gated candidate:           rejected; probe-positive task route did not pass full gate
  lift vs text-only baseline:     +0.078 accuracy, -0.078 residual
  report: reports/symliquid_rollout_cuda_50x5_hv1024_readout_gated_lr002_e6.json

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
  SymLiquid sequence scorer 50k: accuracy=0.854 residual=0.146

local BabyLM BLIMP filtered split, 53,888 train / 5,987 eval:
  contract symbolic scorer:      accuracy=0.500 residual=0.500
  SymLiquid 50k steps:           accuracy=0.746 residual=0.254
  SymLiquid 100k steps + state:  accuracy=0.810 residual=0.190
  SymLiquid 400k + binding/island: accuracy=0.890 residual=0.110
  SymLiquid 800k + binding/island/SVA: accuracy=0.924 residual=0.076
```

The BabyLM/BLIMP bridge is local-only and diagnostic. It does not call outside models, and its exported BLIMP split is not claimed as an official BabyLM leaderboard score unless the same split and metric are used.

## CUDA Backend

The `symliquid-cuda` crate defines the optional backend boundary and includes first-pass CUDA C kernel sources for:

| Kernel | Purpose |
| --- | --- |
| `vsa_bind_kernel` | elementwise VSA binding |
| `vsa_bundle_kernel` | decayed memory accumulation |
| `vsa_permute_kernel` | circular shift permutation |
| `cleanup_similarity_kernel` | symbol cleanup dot products |
| `reservoir_update_kernel` | recurrent reservoir update |
| `kan_rbf_expand_kernel` | KAN-lite RBF expansion |
| `liquid_elementwise_update_kernel` | liquid state update primitive |
| `efe_score_kernel` | expected-free-energy score composition |
| `readout_sgd_samples_kernel` | persistent in-device linear readout SGD over static feature/target/weight buffers |
| `rollout_state_update_kernel` | batched liquid/reservoir/VSA state update over persistent rollout buffers |

At this milestone, the CUDA feature uses `cudarc`/NVRTC to launch real CUDA kernels for VSA binding, bundling, permutation, persistent readout SGD, and a first batched rollout state update. The readout trainer copies features, targets, weights, and bias to the device once, reuses the compiled kernel/module/stream, runs chunked SGD on GPU, then copies the trained readout back once for local verifier evaluation. The rollout primitive copies observations, state, and parameters once, runs multi-step liquid/reservoir/VSA updates on device buffers, and parity-tests against the CPU implementation. The rollout trainer can attempt a supervised local update of the liquid input/recurrent weights, reservoir input/recurrent weights, biases, and hypervector projection, then accepts the candidate only if a larger independent masked probe improves aggregate accuracy, keeps loss stable, and avoids a serious per-task regression. It also trains task-specialized readout heads beside the shared readout, but selects them only if an independent readout probe beats the shared head without serious per-task damage. Otherwise it keeps the shared head and reports the rejected alternatives, state-alignment score, and task-family residuals. Other accelerated primitives still have kernel sources and CPU fallbacks while their launch paths are filled in. Speedups should not be claimed until each kernel is benchmarked against the CPU reference on the target hardware.

Feature check:

```bash
cargo test --features cuda
cargo run -p symliquid-bench --features cuda
cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --cases-per-task 20 --epochs 2 --samples-per-launch 16 --hv-dim 512
```

## Tests

The test suite covers:

- KAN-lite shape and regularization behavior
- liquid-cell finite outputs
- reservoir spectral-radius sanity and forward shape
- VSA binding, unbinding, cleanup retrieval
- FEP policy-score shape and belief update
- integrated model forward pass and readout update
- CGS accounting and backend trait behavior
- toy task smoke tests
- CUDA surface parity against CPU VSA binding
- CUDA readout SGD parity against CPU per-sample SGD
- CUDA rollout state-update parity against CPU liquid/reservoir/VSA state updates

## Limitations

This is a correctness-first reference prototype. It does not implement a general autograd engine, transformer-scale training, mixed precision, multi-GPU execution, or production CUDA dispatch. The current CUDA training paths accelerate the local linear readout over cached SymLiquid features and a rollout-backed liquid/reservoir/VSA feature path with guarded state-parameter updates, but not full recurrent backpropagation. The Puffer adapter currently exercises local vectorized toy and Ocean CartPole-style rollout boundaries; the high-throughput policy/state hot path still needs a Rust/CUDA FFI loop. The delayed-recall task trains only a small readout over fixed reservoir features. The active-classification and gridworld tasks use discrete toy expected-free-energy approximations.

No AGI, transformer replacement, guaranteed lifelong learning, exact symbolic extraction, or fixed efficiency multiplier is claimed here. Report only results measured by running the included commands on your hardware.

## Measured Results

Use [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) and the generated
`reports/*.json` files as the source of truth for current local results. The
numbers above are measured local development snapshots, not public leaderboard
claims. Generate fresh toy results with:

```bash
cargo run -p symliquid-cli -- ablations --task role_filler --steps 200
cargo run -p symliquid-cli -- ablations --task delayed_recall --steps 200
cargo run -p symliquid-cli -- ablations --task active_classification --steps 100
cargo run -p symliquid-cli -- ablations --task gridworld --steps 100
```

Use the resulting CLI tables in papers or reports only as measured toy-prototype results.
