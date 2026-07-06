# SymLiquid + PufferLib Adapter

This folder is the bridge for testing SymLiquid policies in PufferLib/Ocean
environments without using any external model inference.

PufferLib 4.0 is a high-throughput RL stack built around CUDA, static memory,
vectorized rollouts, and environment suites. SymLiquid should use it as an
environment and rollout harness, not as a teacher model.

## Contract

- PufferLib supplies environments, observations, rewards, and rollout speed.
- SymLiquid supplies the policy, state, memory, and action selection.
- No provider APIs or external model inference are allowed.
- Published third-party scores are metadata only.

## Intended Integration

```text
PufferLib vectorized env
  -> observation batch
  -> SymLiquid observation encoder
  -> liquid/reservoir/VSA state update
  -> policy/readout action logits
  -> PufferLib action batch
  -> rewards/dones/residuals
```

The adapter is dependency-optional and now has two policy backends:

- Python reference scorer for parity/debugging.
- Rust FFI scorer via `crates/symliquid-ffi` for local dense and recurrent
  policy/state updates.

The serious version should continue moving the hot path into Rust/CUDA and keep
Python only as the PufferLib environment boundary. Discrete CEM training for
local Ocean-style chain, memory, noisy-memory, noisy-T-maze, slot-T-maze, and T-maze tasks can now run
with a Rust-owned rollout/training loop through the same FFI DLL.

Current GPU status:

```text
cargo run --release -p symliquid-cli --features cuda -- train-standalone-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 100 --epochs 10 --samples-per-launch 32 --hv-dim 4096 --lr 0.05 --out reports/symliquid_cuda_sgd_100x10_hv4096.json
cargo run --release -p symliquid-cli --features cuda -- train-rollout-cuda --train-seed 0 --eval-seed 10000 --cases-per-task 50 --epochs 5 --state-epochs 6 --state-lr 0.02 --samples-per-launch 32 --rollout-batch 200 --obs-dim 64 --hidden-dim 96 --reservoir-dim 128 --hv-dim 1024 --seq-len 64 --lr 0.03 --out reports/symliquid_rollout_cuda_50x5_hv1024_readout_gated_lr002_e6.json
```

Those commands use persistent CUDA buffers for the local readout trainer and a
rollout-backed liquid/reservoir/VSA feature path. The rollout command can try a
supervised local update of the liquid input/recurrent weights, reservoir
input/recurrent weights, biases, and hypervector projection, but it keeps the
frozen state parameters unless a larger independent masked probe improves
aggregate accuracy, keeps loss stable, and avoids serious per-task regression.
It also trains task-specialized readout heads beside the shared head, then uses
an independent readout probe to reject task-head routing when it damages any
task family.
The CUDA crate also includes a
parity-tested `rollout_state_update_kernel` for batched liquid/reservoir/VSA
state updates over persistent rollout buffers. The current Rust FFI bridge moves
dense, memory-recurrent, T-maze recurrent, and noisy-evidence recurrent policy
scoring out of Python. It also owns the local discrete CEM rollout/training
loop for the non-CartPole Ocean-style tasks. The next PufferLib-relevant step is
CUDA-backed batched env stepping, rewards, dones, and optimizer state.

## Smoke Check

```bash
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --check
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --smoke-actions 8
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --rollout-smoke-steps 128 --num-envs 64 --obs-dim 8 --action-modulo 4 --out reports/puffer_style_smoke_128x64.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_policy_500x30_hv4096.json --env ocean-cartpole --rollout-smoke-steps 512 --num-envs 64 --out reports/puffer_ocean_cartpole_smoke_512x64.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-cartpole-policy --iterations 24 --population 32 --elite-count 6 --num-envs 64 --train-steps 256 --eval-steps 1024 --seed 0 --policy-out reports\symliquid_ocean_cartpole_policy_cem_seed0.json --out reports\symliquid_ocean_cartpole_policy_cem_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_cartpole_policy_cem_seed0.json --env ocean-cartpole --rollout-smoke-steps 1024 --num-envs 128 --out reports\puffer_ocean_cartpole_learned_cem_seed0_1024x128.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-chain --iterations 4 --population 12 --elite-count 4 --num-envs 64 --train-steps 128 --eval-steps 512 --seed 0 --policy-out reports\symliquid_ocean_chain_policy_cem_prior_seed0.json --out reports\symliquid_ocean_chain_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-memory --iterations 4 --population 12 --elite-count 4 --num-envs 64 --train-steps 128 --eval-steps 512 --seed 0 --policy-out reports\symliquid_ocean_memory_policy_cem_prior_seed0.json --out reports\symliquid_ocean_memory_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-tmaze --iterations 4 --population 12 --elite-count 4 --num-envs 32 --train-steps 96 --eval-steps 256 --seed 0 --policy-out reports\symliquid_ocean_tmaze_policy_cem_prior_seed0.json --out reports\symliquid_ocean_tmaze_policy_cem_prior_seed0_train.json
cargo build --release -p symliquid-ffi
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_tmaze_policy_cem_prior_seed0.json --env ocean-tmaze --rollout-smoke-steps 512 --num-envs 128 --use-rust-ffi --out reports\puffer_ocean_tmaze_rust_ffi_128x512.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-memory --iterations 8 --population 20 --elite-count 5 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 0 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json --out reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --artifact reports\symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json --env ocean-noisy-memory --rollout-smoke-steps 1024 --num-envs 128 --use-rust-ffi --out reports\puffer_ocean_noisy_memory_rust_ffi_cem_seed0_1024x128.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-memory --iterations 8 --population 20 --elite-count 5 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 0 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_memory_policy_rust_trainer_seed0.json --out reports\symliquid_ocean_noisy_memory_policy_rust_trainer_seed0_train.json
.\.venv-puffer\Scripts\python.exe adapters\pufferlib\symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-tmaze --iterations 12 --population 24 --elite-count 6 --num-envs 64 --train-steps 192 --eval-steps 768 --seed 2 --use-rust-ffi --policy-out reports\symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed2.json --out reports\symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed2_train.json
python scripts\benchmark_treadmill.py --reports reports --out reports\benchmark_treadmill_status.json --limit 320
```

Latest rollout smoke:

```text
transitions=8192
transitions_per_second=9866.5
mean_reward=0.246
report=reports/puffer_style_smoke_128x64.json
```

Latest Ocean CartPole-style smoke:

```text
env=ocean-cartpole
transitions=32768
transitions_per_second=17506.2
mean_reward=0.816
external_inference_calls=0
report=reports/puffer_ocean_cartpole_smoke_512x64.json
```

Latest learned Ocean CartPole policy:

```text
algorithm=local_cross_entropy_search
feature_set=cartpole_linear_v1
train_eval_mean_reward=0.995
rollout_mean_reward=0.995
rollout_transitions=131072
rollout_transitions_per_second=36694.1
rollout_dones=44
rollout_truncations=596
external_inference_calls=0
policy=reports/symliquid_ocean_cartpole_policy_cem_seed0.json
train_report=reports/symliquid_ocean_cartpole_policy_cem_seed0_train.json
rollout_report=reports/puffer_ocean_cartpole_learned_cem_seed0_1024x128.json
```

Light three-run learned-policy sweep:

```text
mean_reward=0.99471
std_reward=0.00070
mean_transitions_per_second=36746.1
external_inference_calls=0
report=reports/puffer_ocean_cartpole_learned_cem_seed_sweep_1024x128.json
```

Local Ocean breadth smoke:

```text
envs=cartpole,chain,memory,tmaze
policy_modes=cartpole_linear_v1,dense_linear_v1,memory_recurrent_linear_v1,tmaze_recurrent_linear_v1
initialization=cgs_governance_prior + local CEM tuning
mean_normalized_perf=0.9988
mean_transitions_per_second=51336.7
external_inference_calls=0
report=reports/puffer_ocean_breadth_smoke_cartpole_chain_memory_tmaze.json
```

Rust FFI policy/backend parity:

```text
crate=crates/symliquid-ffi
library=target/release/symliquid_ffi.dll
envs=cartpole,chain,memory,tmaze
all_reward_parity=true
speedups=6.06x,5.32x,8.12x,9.90x
summary=reports/puffer_ocean_rust_ffi_parity_summary.json
```

Noisy delayed-memory pressure test:

```text
env=ocean-noisy-memory
feature_set=evidence_recurrent_linear_v1
policy_backend=rust_ffi
normalized_perf=0.8641
last_cue_baseline_normalized_perf=0.7445
rollout_transitions_per_second=204968.7
external_inference_calls=0
policy=reports/symliquid_ocean_noisy_memory_policy_cem_prior_seed0.json
train_report=reports/symliquid_ocean_noisy_memory_policy_cem_prior_seed0_train.json
rollout_report=reports/puffer_ocean_noisy_memory_rust_ffi_cem_seed0_1024x128.json
baseline_report=reports/puffer_ocean_noisy_memory_last_cue_baseline_rust_ffi_1024x128.json
```

Rust FFI breadth smoke:

```text
envs=cartpole,chain,memory,tmaze,noisy-memory,noisy-tmaze,slot-tmaze
mean_normalized_perf=0.9720
min_normalized_perf=0.8641
mean_transitions_per_second=278999.4
external_inference_calls=0
report=reports/puffer_ocean_rust_ffi_breadth_cartpole_chain_memory_tmaze_noisy.json
```

Rust-owned rollout/trainer smoke:

```text
trainer_backend=rust_ffi_rollout_trainer
envs=chain,memory,tmaze,noisy-memory,noisy-tmaze,slot-tmaze
mean_normalized_perf=0.9658
min_normalized_perf=0.8631
mean_rollout_transitions_per_second=305117.8
noisy_memory_train_wall_seconds_observed=2.31
external_inference_calls=0
report=reports/puffer_ocean_rust_rollout_trainer_breadth_summary.json
```

Current treadmill frontier:

```text
policy=local_only_no_external_inference
methodology=capability_ratchet / benchmaxxing_performance_ratchet
families=12
initial_mastery_threshold=0.90
ordinary_floor_threshold=0.70
threshold_decay=stalled_effort_only
saturated=12
open=0
latest_mutated_babylm=babylm_mutated_holdout eval_accuracy=0.9814583 residual=0.0185417
public_babylm_comparator=babylm_local_probe eval_accuracy=0.9243361 residual=0.0756639
ratchet_warning=seed49 mutated BabyLM is now regression; rotate to seed55 as the next anti-Goodhart frontier
treadmill_report=reports/benchmark_treadmill_status.json
benchmark_ledger=reports/benchmark_ledger.json
model_ledger=reports/model_ledger.json
public_comparator_ledger=reports/public_comparator_ledger.json
babylm_residual_analysis=reports/babylm_residual_analysis.json
mutated_babylm_residual_analysis=reports/babylm_mutated_residual_analysis.json
residual_escrow=reports/residual_escrow.json
capability_ratchet=reports/capability_ratchet_report.json
ratcheting_generative_system=reports/ratcheting_generative_system_report.json
octopus_router=reports/octopus_router_report.json
tool_registry=reports/tool_registry.json
```

RGS audit note: high-bandwidth embodied logging is now implemented for the
local slot T-maze rollout. The eventized sidecar records bounded raw windows,
events, semantic phases, skill traces, and residuals:

```text
event_log=reports/puffer_ocean_slot_tmaze_eventized_rollout_log.json
sampled_raw_windows=256
event_count=384
semantic_events=256
skill_events=256
residual_events=0
external_inference_calls=0
```

Noisy T-maze architecture note:

```text
env=ocean-noisy-tmaze
feature_set=evidence_sum_tmaze_recurrent_linear_v1
change=equal-weight cue accumulation replaced decayed evidence memory
previous_normalized_perf=0.8070
current_normalized_perf=0.8322
ceiling_adjusted_normalized_perf=0.9943
report=reports/puffer_ocean_noisy_tmaze_sum_rust_trainer_seed2_rollout_2048x128.json
```

Slot-memory T-maze result:

```text
env=ocean-slot-tmaze
feature_set=slot_tmaze_recurrent_linear_v1
capability=two role-filler slots with delayed query at the branch
normalized_perf=0.9961
external_inference_calls=0
report=reports/puffer_ocean_slot_tmaze_rust_trainer_seed3_rollout_2048x128.json
```

Noisy-memory ceiling result:

```text
env=ocean-noisy-memory
feature_set=evidence_sum_recurrent_linear_v1
ceiling_adjusted_normalized_eval_reward=0.9975
external_inference_calls=0
report=reports/symliquid_ocean_noisy_memory_policy_rust_trainer_seed4_long_train.json
```

Local setup on this machine:

```text
vendor/pufferlib at commit 69fcbcff
.venv-puffer uses Python 3.11.9
pufferlib imports successfully
torch installed as 2.11.0+cpu
pufferlib._C is not built in the Windows editable install
```

The adapter intentionally keeps imports optional. If `pufferlib` is not
installed, the check reports that cleanly and exits without failing the Rust
build. The scorer supports sparse hashed policy features, learned
`cartpole_linear_v1`, dense discrete policies, recurrent cue-memory policies,
T-maze recurrent policies, and evidence-accumulation policies. This lets the
smoke loop distinguish untrained rollouts, learned rollouts, compact
governance-prior policies, explicit reflex-governance overlays, and recurrent
state tasks where accumulation beats a last-cue baseline. It is still not the
final GPU policy hot path.
