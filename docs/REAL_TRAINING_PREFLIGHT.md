# Real Training Preflight

SymLiquid should not start long training until the real-training gate is green.
The architecture gate proves the RMI/ORA ratchet is coherent; the preflight gate
proves the local training path is fast, measured, and safe enough to spend time on.

## Required Gate

Run:

```powershell
py -3.13 scripts\training_preflight.py --run-build-check --run-smokes --run-split-check --run-candidate-gate --out reports\training_preflight_report.json
```

For a hard CI-style block, add `--strict`.

Current status from the latest report:

```text
heavy_training_allowed=true
passed=23/24
blockers=0
warnings=1
warning=split_strict_quality strict_ok=False sentence_overlaps=9
```

This means training is allowed by the preflight gate, but seed55 should be
treated honestly as a mutated frontier with a split-quality warning, not as a
pristine private holdout.

## Hardware Policy

The RTX 2060 Super profile lives in:

```text
configs/training_profiles_rtx2060super.json
```

The profile defines:

- `smoke`
- `inner_loop`
- `candidate`
- `seed_sweep`

Each profile fixes `hv_dim`, rollout batch size, samples per launch, sequence
length, max VRAM, expected runtime, and promotion gates. Do not tune long runs by
guesswork; change the profile and let the preflight report record why.

The resource governor is part of the current training policy:

```powershell
py -3.13 scripts\resource_governor.py --profile inner_loop --out reports\resource_governor.json
```

It currently detects the RTX 2060 Super and allows the configured profile with
no throttle reasons. If the governor throttles a requested profile, run a
smaller profile or fix the resource condition before continuing.

## Current Backend Policy

Puffer/Ocean serious training is Rust/CUDA-owned for now.

Reason:

- the local `.venv-puffer` torch stack is CPU-only;
- the Puffer compiled backend is not available in this shell;
- Rust/CUDA can own env stepping, rollout state, scoring, rewards, and reporting.

Python should orchestrate and analyze. Hot loops belong in Rust/CUDA.

If native Python or Puffer extensions need MSVC, load the installed Visual
Studio Developer environment in the current PowerShell process with:

```powershell
.\scripts\use_msvc_dev_shell.ps1
```

## Promotion Policy

A real candidate promotes only if:

- architecture gate is green;
- RMI score is `1.0`;
- public BLIMP/BabyLM comparator does not regress;
- seed49 mutated holdout remains regression;
- seed55 mutated holdout exists and clears the frontier floor;
- residual escrow is active and not hiding critical regressions;
- CUDA runtime, timing, launch, and GPU telemetry are reported;
- CUDA fallback did not occur.

The candidate gate is:

```powershell
py -3.13 scripts\candidate_promotion_gate.py --runtime-report reports\preflight_cuda_rollout_smoke.json --out reports\candidate_promotion_gate.json
```

Current candidate status:

```text
promote=false
active_family=coding_local_sandbox
best_public_calibration_card=source_human_eval_wide_32_tasks_pass_rate_0.78125
active_public_calibration_card=source_evalplus_source_bigcodebench_source_livecodebench_below_floor
latest_mbpp_evalplus_cards=MBPP_32_tasks_0.71875_above_floor_EvalPlus_32_tasks_0.59375_below_floor
next_code_rotation_card=source_agnostic_type_edge_interface_algorithmic_pressure
transfer_interleave=same_family_code_first_then_broader_transfer_if_wall_persists
broad_transfer_matrix=YELLOW_160_public_calibration_tasks_aggregate_pass_rate_0.5125_sts_delta_0.28125
public_task_pass_rate=0.5125
required_public_task_floor=0.70
token_level_student_generation_valid=true
template_like_candidate_count=0
loop_closure_candidate_count=0
score_semantics=student_code_lm_checkpoint_public_task_calibration_only
```

This means the current candidate gate is intentionally not promoting. The
remaining failed gate is public code transfer below the floor. Promotion claims
still must preserve score semantics:
the public code number is calibration evidence for the learned student
checkpoint, not a broad claim of public-code mastery.

## Leakage Policy

Before BabyLM candidate training, run:

```powershell
py -3.13 scripts\check_babylm_splits.py --out reports\babylm_split_leakage_report.json
```

Exact minimal-pair overlap blocks training. Sentence overlap is reported as a
quality warning because mutated examples can intentionally reuse some grammar
shape while still preserving private pair separation.

## BabyLM Cache

The Rust BabyLM JSONL loader caches parsed minimal-pair cases under:

```text
data/.cache/babylm
```

The cache key includes source path, file size, modified time, and limit, so
changing a source file or limit naturally creates a new cache entry. Override the
cache directory with `SYMLIQUID_BABYLM_CACHE_DIR` when running larger sweeps on a
faster disk.

## Ablations

The matched ablation matrix lives in:

```text
configs/ablation_matrix_rtx2060super.json
```

It defines matched comparisons for CPU baseline, CUDA readout, frozen rollout
state, learned state, task heads, residual adapters, bridge benchmarks, and
public-only versus public-plus-mutated evaluation.

Run the whole matrix with:

```powershell
py -3.13 scripts\run_ablation_matrix.py --out reports\ablation_matrix_rtx2060super_report.json
```

## One-Command Profile Runner

After smoke is green, use the profile runner for repeatable local ratchets:

```powershell
py -3.13 scripts\run_training_ratchet_profile.py --profile inner_loop --out reports\training_ratchet_profile_run.json
```

The runner snapshots residual escrow, prepares the governed synthetic-data
blend, trains the seed55 frontier report, runs matched ablations, runs VRAM
stress probes, refreshes the RMI/ORA ledgers, reruns promotion gates, and
appends real workflow routing traces.

## Synthetic Data Gate

The local curator is:

```powershell
py -3.13 scripts\synthetic_data_curator.py --policy configs\synthetic_data_policy.json --out reports\synthetic_data_curator.json
```

It is allowed to feed BabyLM profile runs only when:

- exact pair and sentence overlap with eval/holdouts is zero;
- mean quality clears policy;
- per-rule concentration stays bounded;
- synthetic share remains capped;
- public comparator, seed49 regression, seed55 frontier, and residual delta
  gates still decide promotion.

## VRAM Stress

Before long runs, verify the configured dimensions fit the RTX 2060 Super:

```powershell
py -3.13 scripts\profile_vram_stress.py --profile inner_loop --profile candidate --out reports\profile_vram_stress_report.json
```

The stress probe uses each profile's rollout dimensions with tiny case counts so
it tests memory shape without becoming a long training run.

## Residual Delta Gate

Candidate promotion compares the current escrow ledger against:

```text
reports/residual_escrow_pre_candidate_baseline.json
```

Create or refresh the snapshot before a candidate/frontier run with:

```powershell
py -3.13 scripts\snapshot_residual_escrow.py
```

The promotion gate blocks candidates that add too many residual clusters,
reactivated diagnostics, critical clusters, or large max-residual increases.

## Hard Rule

Do not start long training while `reports/training_preflight_report.json` says:

```json
{"heavy_training_allowed": false}
```

The frontier moves only when the floor holds and the hot path is measured.
