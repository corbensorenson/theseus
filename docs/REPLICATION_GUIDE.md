# Project Theseus Replication Guide

Last consolidated: 2026-05-20.

This guide is the practical runbook for recreating the current local Project
Theseus / SymLiquid / SparkStream system from the repository. It is written for
a new operator who has the repo but not the chat history.

The system is a local research harness, not a finished foundation model. Its
goal is to make learning, routing, verification, residual pressure, teacher
guidance, and promotion gates explicit enough that progress can be reproduced
and audited.

## 1. What You Are Rebuilding

Project Theseus currently has these major planes:

| Plane | Purpose | Main entrypoints |
| --- | --- | --- |
| SparkStream control plane | Dashboard, daemon, autonomy cycles, watchdogs, launch readiness, reports, checkpoints, sparse teacher queue. | `scripts/start_sparkstream.ps1`, `scripts/sparkstream_dashboard.py`, `scripts/sparkstream_daemon.py`, `scripts/autonomy_cycle.py`, `scripts/autonomy_watchdog.py` |
| SymLiquid / Code LM learning plane | Rust-first learning and evaluation lanes, including full-body code generation, STS conditioning, residual training, and private/public calibration separation. | `crates/symliquid-cli`, `scripts/code_lm_closure.py`, `reports/code_lm_closure*.json`, `reports/real_code_benchmark_graduation.json` |
| Benchmark ratchet plane | Keeps benchmark surfaces staged, rotates frontiers, records residuals, preserves mastered surfaces as regressions, and blocks promotion when evidence is insufficient. | `scripts/benchmaxx_curriculum.py`, `scripts/candidate_promotion_gate.py`, `scripts/real_code_benchmark_graduation.py`, `reports/benchmark_ledger.json`, `reports/residual_escrow.json` |
| Deterministic taming plane | Linters, grammar suckers, schema validators, provenance checks, AST/syntax checks, and other rule systems that constrain outputs without providing benchmark answers. | `configs/grammar_suckers.json`, `scripts/grammar_suckers.py`, `scripts/deterministic_taming_stack.py`, `scripts/student_first_evidence_audit.py` |
| Octopus / Hive plane | Specialist arms, local/distributed device runtime, peer discovery, safe task routing, CLI, and future app shell. | `scripts/start_theseus_hive.ps1`, `scripts/theseus_cli.py`, `scripts/hive_node.py`, `reports/hive_*.json` |
| Governance plane | Anti-cheat rules, teacher budget, licensing, resources, cell lifecycle, arm/sucker expiry, update offers, and source-gated self-editing. | `configs/*.json`, `scripts/teacher_budget_audit.py`, `scripts/resource_governor.py`, `scripts/cell_lifecycle.py`, `scripts/arm_sucker_registry.py`, `scripts/license_manager.py` |
| Personality / context plane | User-owned personality cards, runtime context, VCM semantic memory, long-horizon context packets, drift checks, and checkpoint chat context loading. | `scripts/personality_context_builder.py`, `scripts/personality_runtime_audit.py`, `scripts/context_packet_ledger.py`, `scripts/virtual_context_memory.py`, `scripts/checkpoint_chat.py` |

The current promotion-facing learning wall is broad public code transfer. The
latest report state is:

```text
best clean single card: source_human_eval, 32 tasks, pass rate 0.78125
currently selected receiver pressures: source-agnostic type/return-shape, edge-condition, admissibility/interface, and BigCodeBench-heavy algorithmic-planning residuals, with cooled concepts revisited after rotation
below-floor receiver cards: EvalPlus 0.59375, BigCodeBench 0.25, LiveCodeBench 0.21875; MBPP is above floor at 0.71875
broad matrix: 160 public calibration tasks across HumanEval/MBPP/EvalPlus/BigCodeBench/LiveCodeBench, aggregate pass rate 0.5125
promotion floor: 0.70
candidate gate: promote=false
failed gate family: broad_public_code_transfer_ready
candidate source: student_code_lm_checkpoint_v1
token-level learned generation: true
template-like benchmark candidates: 0
loop-closure benchmark candidates: 0
BigCodeBench/LiveCodeBench: real D:-staged public tasks with clean task-matched candidates; both have 32-task clean slices and remain below floor
STS public ablation: positive but still below maturity; aggregate broad delta 0.28125,
latest board-run 4-card 128-task high-transfer closure delta 0.242187, no regressions
teacher status: proposal-only architecture diagnosis, no public answers
VIEA action executor: approved local queue actions run with step budgets,
pause/resume/block state, and `reports/viea_action_execution_ledger.jsonl`
resume evidence
```

Treat these numbers as a snapshot. Before quoting current status, refresh or
read the JSON reports listed in section 8.

## 2. Requirements

Primary supported environment:

- Windows 10/11 with PowerShell.
- Rust stable MSVC toolchain.
- Python 3.11+ or 3.13. Existing scripts prefer `.venv-puffer\Scripts\python.exe`
  when present, then `python` or `py`.
- Git.
- Optional CUDA-capable GPU. The local reference machine uses an RTX 2060 Super,
  but CPU smoke paths are still useful.
- Optional `D:` drive for large governed data, private curricula, and ignored
  training artifacts.

Recommended storage layout:

```text
D:\ProjectTheseus\repo                      preferred tracked repository on Windows
C:\Users\<you>\Documents\New project        optional compatibility junction to D:\ProjectTheseus\repo
D:\ProjectTheseus                           ignored large data/artifact root
D:\ProjectTheseus\training_data             governed private/open data
D:\ProjectTheseus\resource_pantry           cloned/staged external resources
D:\ProjectTheseus\runs                      longer run outputs when needed
```

Do not place bulk downloaded training data in the Git repo. Keep generated
reports under `reports/` and large datasets/checkpoints under ignored storage.

On this Windows workstation, `C:` is space-constrained and `D:` is the safer
canonical home. To migrate an existing checkout without breaking old launchers,
use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_project_home_to_d.ps1
```

The default dry run writes `reports/project_home_migration_plan.json`, estimates
the source footprint excluding existing junctions, and previews the `robocopy`
plus junction plan. The real migration should be run only after committing or
allowing intentional dirty files and closing dashboards, daemons, editors, and
terminals pointed at the old path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_project_home_to_d.ps1 -Execute -CreateCompatibilityJunction
```

The default execution keeps a timestamped `New project.pre-d-migration-*`
backup on C:. After verifying the D: checkout, free the old C: copy by deleting
that backup manually. Use `-RemoveBackupAfterJunction` during the first
execution only if you intentionally want immediate cleanup after the junction is
created.

## 3. First-Time Setup

From the repository root:

```powershell
git status --short
rustup show
cargo check -p symliquid-cli
powershell -ExecutionPolicy Bypass -File scripts\setup_coding_runtime.ps1
powershell -ExecutionPolicy Bypass -File scripts\install_theseus_cli.ps1
```

If `cargo check` fails because a previous Rust executable is still running,
inspect and stop only the stale Theseus process:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like '*symliquid-cli*' } |
  Select-Object ProcessId, CommandLine
```

Avoid broad process killing. Long-running training may be legitimate.

## 4. Start The Local System

Start Hive and SparkStream:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_theseus_hive.ps1 -Restart -NoDashboard
powershell -ExecutionPolicy Bypass -File scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute -AllowTeacher -AllowNetworkFetch -DurationHours 10 -Port 8787 -Restart
```

Dashboard:

```text
http://127.0.0.1:8787
```

Optional Hive status:

```text
http://127.0.0.1:8791/api/hive/status
```

Optional local OpenAI-compatible endpoint, when enabled from dashboard/CLI:

```text
http://127.0.0.1:8789/v1
```

## 5. Refresh The Truth Layer

Run these before making claims about health, learning, or promotion:

```powershell
python scripts\learning_scoreboard.py --out reports\learning_scoreboard.json --markdown-out reports\learning_scoreboard.md
python scripts\candidate_promotion_gate.py --out reports\candidate_promotion_gate.json
python scripts\autonomy_watchdog.py --fix --out reports\autonomy_watchdog.json
```

`candidate_promotion_gate.py` may exit nonzero when `promote=false`. That is
expected when the student is below floor. A blocked gate is not a failed script
if the JSON explains the honest blocker.

For overnight readiness:

```powershell
python scripts\overnight_learning_readiness.py --out reports\overnight_learning_readiness.json
```

## 6. Run The Current Code Learning Lane

The current canonical Code LM closure smoke path is step-budgeted. Wall-clock
timeouts are safety fuses; `--max-rust-work-steps` is the primary bound, and
the wrapper writes partial progress plus reusable Rust artifacts so interrupted
runs can resume without orphan locks:

```powershell
python scripts\code_lm_closure.py `
  --seed 211 `
  --private-count 220 `
  --public-cards source_human_eval `
  --max-public-cases-per-card 8 `
  --epochs 3 `
  --lr 0.08 `
  --candidates-per-task 6 `
  --max-extra-private-train 0 `
  --max-residual-private-train 180 `
  --sts-timeout-seconds 7200 `
  --rust-timeout-seconds 0 `
  --public-timeout-seconds 7200 `
  --max-rust-work-steps 240000 `
  --hv-dim 256 `
  --max-vocab 384 `
  --out reports\code_lm_closure.json `
  --rust-report-out reports\code_lm_closure_rust.json `
  --public-report-out reports\real_code_benchmark_graduation.json `
  --public-trace-out reports\real_code_benchmark_traces.jsonl `
  --public-transfer-artifact-out reports\transfer_artifacts\code\real_code_benchmark_graduation_transfer_artifact.json
```

Then refresh the public STS and evidence audits:

```powershell
python scripts\sts_repair_ablation.py --out reports\sts_repair_ablation.json
python scripts\student_first_evidence_audit.py --out reports\student_first_evidence_audit.json
python scripts\learning_scoreboard.py --out reports\learning_scoreboard.json --markdown-out reports\learning_scoreboard.md
python scripts\candidate_promotion_gate.py --out reports\candidate_promotion_gate.json
python scripts\autonomy_watchdog.py --fix --out reports\autonomy_watchdog.json
```

The student is allowed to learn from private hidden-test tasks, local approved
code, old approved Corben data, and governed permissive corpora. Public
HumanEval/MBPP/EvalPlus solutions and hidden public tests are evaluation-only
and must not enter training.

To stage public BigCodeBench/LiveCodeBench calibration payloads on `D:` without
admitting them to training, run:

```powershell
python scripts\stage_public_code_benchmark_data.py --live-shards 1 --out reports\public_code_benchmark_data_stage.json
```

Expected eval-only staging roots:

```text
D:\ProjectTheseus\resource_pantry\datasets\bigcodebench
D:\ProjectTheseus\resource_pantry\datasets\livecodebench
```

## 7. Anti-Cheat Contract

Promotion-facing code evidence must satisfy all of these:

- Candidate source is an actual student checkpoint.
- The student emits token-level code candidates.
- Public benchmark solutions are not in the training corpus.
- Hidden public tests are not used for training or repair.
- Template-like candidates are not counted as learning evidence.
- Loop-closure tools do not solve benchmark tasks.
- Task-id lookup, exact-answer memory, and benchmark-specific shortcuts are
  forbidden.
- External proprietary inference is not counted as local model capability.
- Deterministic grammar suckers may reject invalid form, but may not provide
  benchmark answers.
- Teacher output may diagnose architecture or propose experiments, but may not
  solve public benchmark tasks or become distillation data.

Private synthetic and private public-shaped curricula are allowed only as
training pressure. Public scores are calibration and promotion evidence only
when the anti-cheat reports stay clean.

## 8. Source Of Truth Reports

| Question | Read this first |
| --- | --- |
| Is the system operationally healthy? | `reports/autonomy_watchdog.json`, `reports/sparkstream_status.json` |
| Did the student actually learn? | `reports/learning_scoreboard.json`, `reports/code_lm_closure.json`, `reports/code_lm_closure_rust.json` |
| Is public code transfer good enough? | `reports/broad_transfer_matrix.json`, `reports/real_code_benchmark_graduation.json`, `reports/sts_repair_ablation.json` |
| Can the candidate promote? | `reports/candidate_promotion_gate.json` |
| What should run next? | `reports/benchmaxx_curriculum.json`, `reports/frontier_policy_status.json`, `reports/learning_scoreboard.json` |
| Are teacher calls allowed and scoped correctly? | `reports/teacher_budget_last.json`, `reports/teacher_oracle_last.json` |
| Are rule/verifier layers healthy? | `reports/deterministic_taming_stack.json`, `reports/grammar_suckers.json`, `reports/student_first_evidence_audit.json` |
| Are arms/suckers/tools bloating or expiring? | `reports/cell_lifecycle.json`, `reports/arm_sucker_registry.json` |
| Is personality context wired? | `reports/personality_runtime_audit.json`, `reports/personality_context_last.json` |
| Is large data governed? | `reports/training_data_inventory.json`, `reports/resource_pantry.json` |

## 9. Overnight Run Checklist

Before leaving the system unattended:

1. Commit or stash intentional source/config changes.
2. Confirm dirty files are reports/data you are comfortable regenerating.
3. Refresh the truth layer from section 5.
4. Confirm `reports/overnight_learning_readiness.json` has
   `overnight_launch_ready=true`.
5. Confirm the candidate gate is blocked only by honest learning evidence, not
   stale profiles or missing reports.
6. Start Hive and SparkStream with the commands in section 4.
7. Leave the dashboard open or inspect `reports/autonomy_ledger.jsonl` and
   `reports/sparkstream_daemon_ledger.jsonl` later.

If the watchdog is RED because an operational service is down, fix or restart
it. If the watchdog is YELLOW/RED because public transfer is below floor, keep
promotion blocked and continue the curriculum instead of changing the gate.

## 10. Teacher Policy

The teacher exists to save time on architecture diagnosis, not to replace
learning. Acceptable teacher tasks:

- diagnose residual clusters;
- propose decoder, routing, memory, STS, or verifier experiments;
- identify likely missing capabilities such as recursion, loop planning,
  syntax structure, type handling, or long-horizon repo repair;
- review local source bugs after the candidate-bottleneck reducer has exhausted
  safe local fixes.

Forbidden teacher tasks:

- answer public benchmark problems;
- generate public benchmark training solutions;
- use hidden public tests;
- apply broad source changes without a guarded branch/gate flow;
- bypass license, safety, data, or privacy constraints.

## 11. Current Best Next Work

As of this consolidation, the strongest replication target is not more
scaffolding. It is better learned broad public transfer:

1. Continue the selected source-agnostic high-transfer pressure, currently
   `admissibility_and_interface` after clean edge-condition and
   type/return-shape recalibrations produced no broad/public lift. Keep public
   benchmark solutions and hidden tests out of training.
2. Keep EvalPlus in same-family rotation pressure after it remained below floor
   at `0.59375`, without public EvalPlus solutions or hidden tests.
3. Improve clean MBPP/EvalPlus transfer toward the promotion floor while
   keeping public solutions and hidden tests eval-only. Repeated MBPP reruns
   after fresh residual-private curriculum held at `0.59375`, so the next
   useful work is decoder/architecture improvement rather than only row
   weighting.
4. Keep BigCodeBench and LiveCodeBench at 32+ clean tasks; both are now clean
   32-task receiver cards below floor, so failures should drive semantic
   decoder work rather than adapter expansion work.
5. Train the full-body SymLiquid/state decoder only on private or approved data.
6. Require STS-on to beat STS-off on the same seed/tasks across wider suites
   and improve private repair delta before calling STS repair mature.
7. Promote only if the learned checkpoint clears the floor without leakage or
   regressions.
