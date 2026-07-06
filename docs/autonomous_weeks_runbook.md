# SparkStream Autonomous Weeks Runbook

This project can run in a long-lived local autonomy loop while keeping the user
in control of expensive or risky boundaries.

For the current complete status snapshot, read `docs/PROJECT_STATE.md` first.

## Start Dashboard

```powershell
.\scripts\start_sparkstream.ps1
```

Dashboard:

```text
http://127.0.0.1:8787
```

When this node is acting as a Hive dashboard for a Mac, phone, or another PC,
`start_sparkstream.ps1` binds the dashboard to `0.0.0.0` by default and prints
the LAN URL. To force a local-only dashboard, pass `-DashboardHost 127.0.0.1`.

## Start Autonomous Mode

Refresh the readiness reports before a long run:

```powershell
py -3.13 scripts\arm_lifecycle_manager.py --out reports\arm_lifecycle_governance.json
py -3.13 scripts\autonomy_launch_readiness.py --profile inner_loop --out reports\autonomy_launch_readiness.json
py -3.13 scripts\hive_fleet_readiness.py --out reports\hive_fleet_readiness.json --markdown-out reports\hive_fleet_readiness.md
```

Maintenance-only mode refreshes ledgers, benchmark discovery, data inventory,
RL inventory, checkpoints, and history without running heavy training:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop
```

Training mode runs the selected profile each cycle:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute
```

Teacher and network expansion remain explicit:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute -AllowTeacher -AllowNetworkFetch
```

Prefer leaving network fetch off for ordinary week-long inner-loop training.
Use `-AllowNetworkFetch` only when you specifically want audited benchmark/RL
source discovery or import.

## Unattended VIEA Supervisor

Vacation Mode Supervisor V3 is the preferred unattended runner. It consumes the
Hive work board first, then wraps the older VIEA supervisor, restarts stale local
services when asked, checks hard resource/personality/public-data gates, triages
failed actions, writes a repair queue, and requires every cycle to produce
progress or a useful residual diagnosis.

One non-destructive smoke cycle:

```powershell
py -3.13 scripts\vacation_mode_supervisor.py --cycles 1 --start-services --explore --out reports\vacation_mode_supervisor.json --markdown-out reports\vacation_mode_supervisor.md
```

Overnight local execution:

```powershell
py -3.13 scripts\vacation_mode_supervisor.py --cycles 0 --sleep-seconds 300 --execute --allow-teacher --start-services
```

Exploratory mode can refresh the governed resource pantry and source catalog.
It is still bounded: public benchmarks stay calibration-only, uncertain-license
sources stay queued/metadata-only, commercial ROM/game asset downloads remain
forbidden, and large training-data downloads are not automatic.

```powershell
py -3.13 scripts\vacation_mode_supervisor.py --cycles 0 --execute --allow-teacher --allow-network-fetch --explore --start-services
```

Install it as a Windows scheduled task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_vacation_mode_task.ps1 -Execute -AllowTeacher -Explore -StartServices -RunAtStartup
```

If Windows denies Task Scheduler registration from the current process, the
installer falls back to a per-user Startup loop at
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Project Theseus Vacation Mode.cmd`.
That fallback runs `scripts\vacation_mode_startup_loop.ps1` every 30 minutes
after logon and still writes the same Vacation Mode, morning, overnight proof,
and long-run governor reports.

The reports to read after sleeping are:

```text
reports/hive_long_run_governor.md
reports/hive_morning_report.md
reports/hive_overnight_proof.md
reports/hive_node_registry.json
reports/hive_fleet_readiness.md
reports/hive_artifact_sync.json
reports/hive_work_board_executor.json
reports/hive_unattended_improvement_ledger.jsonl
reports/hive_teacher_auto_escalation_ledger.jsonl
reports/hive_no_progress_families.jsonl
reports/vacation_mode_supervisor.json
reports/vacation_mode_failure_triage.json
reports/vacation_mode_repair_action_queue.json
reports/vacation_mode_exploration.json
reports/unattended_autonomy_supervisor.json
reports/viea_action_executor.json
reports/hive_utilization_manager.json
reports/hive_rented_compute_status.json
reports/hive_rented_compute_plan.json
reports/project_home_migration_plan.json
```

One-night proof before trusting a longer unattended run:

```powershell
py -3.13 scripts\hive_node_registry.py --out reports\hive_node_registry.json
py -3.13 scripts\hive_fleet_readiness.py --out reports\hive_fleet_readiness.json --markdown-out reports\hive_fleet_readiness.md
py -3.13 scripts\hive_artifact_sync.py --out reports\hive_artifact_sync.json --limit 50
py -3.13 scripts\hive_overnight_proof.py --out reports\hive_overnight_proof.json --markdown-out reports\hive_overnight_proof.md
```

`hive_overnight_proof` should be green before you scale from one night to a
longer vacation run. It is an operational proof, not a learning-score claim. It
checks that all trusted capable nodes are fed, remote artifacts return and
merge, at least one improvement or useful residual signal exists, public-data
leak checks are clean, and no repeated task family is stuck.

The long-run governor is the every-cycle operator cockpit:

```powershell
py -3.13 scripts\hive_long_run_governor.py --out reports\hive_long_run_governor.json --markdown-out reports\hive_long_run_governor.md
```

It answers what each node is doing, what improved, what failed, whether
artifacts synced, whether teacher was used, whether any task family was demoted,
and the next bounded action. Vacation Mode writes this report every cycle, so a
30-minute scheduled Vacation Mode task is also the 30-minute governor cadence.

Local service probes use lightweight health endpoints so liveness checks do not
block on the full dashboard payload:

```text
http://127.0.0.1:8787/api/health
http://127.0.0.1:8791/api/hive/health
```

For overnight or vacation-style local autonomy, prefer the VIEA supervisor over
hand-starting heavy commands. It runs one approved VIEA feedback action per
cycle, keeps public benchmarks calibration-only, refreshes the watchdog and
Hive scheduler, reads residuals before teacher escalation, and honors the same
pause/stop files as SparkStream.

One cycle:

```powershell
py -3.13 scripts\unattended_autonomy_supervisor.py --execute --cycles 1 --allow-teacher --out reports\unattended_autonomy_supervisor.json --markdown-out reports\unattended_autonomy_supervisor.md
```

Continuous until stopped:

```powershell
py -3.13 scripts\unattended_autonomy_supervisor.py --execute --cycles 0 --sleep-seconds 300 --allow-teacher
```

Safety posture:

```text
one_approved_action_per_cycle=true
public_benchmarks=calibration_only_never_training
teacher=proposal_only_architect_loop_no_apply_mode
pause_flags=reports/sparkstream_pause.flag,reports/viea_action_executor_pause.flag
stop_flags=reports/sparkstream_stop.flag,reports/unattended_autonomy_stop.flag
```

If a run fails before public evidence is written, inspect:

```powershell
py -3.13 scripts\broad_transfer_residual_reader.py --closure-report reports\broad_transfer_closure_runner_source_evalplus.json
```

Operational faults such as stale locks, missing public reports, or in-progress
Rust reports should be fixed locally before asking the teacher for architecture
guidance. Teacher calls are for measured model walls, not broken plumbing.

## Stop, Pause, Resume

The dashboard has Stop, Pause, and Resume buttons.

The same controls are files:

```powershell
New-Item reports\sparkstream_pause.flag -ItemType File -Force
Remove-Item reports\sparkstream_pause.flag
New-Item reports\sparkstream_stop.flag -ItemType File -Force
```

## Safety Boundaries

Network discovery can queue public benchmark and RL sources. Open-license GitHub
sources can be staged as archives when network mode is explicitly enabled.

Commercial game ROM downloads are forbidden unless the user provides explicit
rights. ROM-like assets detected locally are quarantined until license audit.

Vacation Mode's exploration path is intentionally aimed at transfer, not narrow
benchmaxxing: it prefers source-agnostic coding concepts, open RL/source
adapters, and donor/receiver public calibration. If it finds a new game-like
environment, it should first become a source card, adapter smoke, and residual
surface before it becomes training pressure.

Vacation Mode now enforces the unattended improvement contract for board work.
Each completed task must produce at least one of:

```text
private residual shrank
public transfer improved
new clean evidence produced
adapter repaired
teacher experiment spec produced
stale tool retired
useful failure residual captured
```

If a task family produces no signal, it is demoted once and blocked after a
second no-progress result. If the same residual family fails twice, a
teacher-as-architect task is queued for diagnosis and experiment design, still
with no public benchmark answers and no public solution distillation.

## Watch Progress

The dashboard shows:

- benchmark score bars;
- metric history lines;
- resource governor status and efficiency score;
- launch readiness and arm lifecycle governance;
- autonomous goal routing through the arm registry;
- benchmark and RL registries;
- training data inventory;
- active daemon jobs and daemon ledger;
- self-improvement queue;
- checkpoints and checkpoint materialization;
- live/checkpoint state chat.

The append-only artifacts are:

```text
reports/autonomy_ledger.jsonl
reports/sparkstream_daemon_ledger.jsonl
reports/sparkstream_metrics.jsonl
reports/resource_governor.json
reports/arm_lifecycle_governance.json
reports/autonomy_launch_readiness.json
reports/autonomous_goal_ledger.jsonl
reports/routing_memory_real_traces.jsonl
reports/checkpoint_registry.json
```

Current live snapshot:

```text
active_family=coding_local_sandbox
best_public_calibration_card=source_human_eval_wide_32_tasks_pass_rate_0.78125
active_public_calibration_card=source_evalplus_source_bigcodebench_source_livecodebench_below_floor
latest_mbpp_evalplus_cards=MBPP_32_tasks_0.71875_above_floor_EvalPlus_32_tasks_0.59375_below_floor
next_code_rotation_card=source_agnostic_type_edge_interface_algorithmic_pressure
transfer_interleave=same_family_code_first_then_broader_transfer_if_wall_persists
broad_transfer_matrix=YELLOW_160_public_calibration_tasks_aggregate_pass_rate_0.5125_sts_delta_0.28125
required_public_code_floor=0.70
candidate_promote=false
learning_scoreboard=YELLOW_due_to_broad_public_transfer_below_floor
watchdog=YELLOW_or_RED_only_for_operational_or_stale_lane_faults
preflight_heavy_training_allowed=true
launch_ready=true
arm_lifecycle_ready=true
resource_governor_efficiency=1.0
compute_market=internal_accounting_only, exchange_off
```

## Windows Home On D:

Long unattended runs should not depend on a nearly full `C:` drive. On this
workstation, prefer `D:\ProjectTheseus\repo` as the canonical checkout and keep
the old `C:\Users\<you>\Documents\New project` path only as a compatibility
junction.

Plan the migration:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_project_home_to_d.ps1
```

Run the migration after closing SparkStream, Hive, editors, terminals, and
Codex sessions that are using the old path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\migrate_project_home_to_d.ps1 -Execute -CreateCompatibilityJunction
```

Only remove the old C: backup after `D:\ProjectTheseus\repo` passes `git
status`, `theseus status`, and a smoke command.

## Multi-Machine Hive

The detailed Hive setup, user access, Mac packaging, phone roaming, storage,
remote-control, voice-following, utilization, rented compute, and update flow
now lives in [Project Theseus Hive](THESEUS_HIVE.md). Keep this runbook focused
on unattended operation.

Before leaving the fleet unattended, check the Hive at a high level:

```powershell
theseus status
theseus hive users
theseus update verify-hive
theseus update publish-hive
theseus update converge-hive --execute
```

Minimum unattended requirements:

- every worker is joined by machine invite or active Hive profile;
- people/phones use per-user operator tokens instead of sharing the machine
  join token;
- remote paths use same-LAN, hotspot, self-hosted WireGuard/private tunnel, or
  authenticated HTTPS relay;
- public contribution remains opt-in and worker-only;
- remote tasks stay restricted to registered task kinds;
- hard source/app replacement remains package-based or explicit.

Before claiming multi-node overnight work is real, run the network doctor:

```powershell
theseus hive network-doctor
```

Fix RED coordinator/peer findings first. The doctor checks local API, LAN URL,
coordinator URL, stale peers, firewall symptoms, flapping/retry-recovered
reachability, and roaming posture. It writes `reports/hive_network_doctor.json`
and `.md`.

Probe each machine when anything changes:

```powershell
py -3.13 scripts\hive_fleet_readiness.py
```

The authoritative fleet view is:

```powershell
py -3.13 scripts\hive_node_registry.py --out reports\hive_node_registry.json
```

Board assignment, the Hive scheduler, fleet readiness, the training
orchestrator, and the utilization manager all read this same registry. A node
can be `light_task_allowed` while `training_allowed` is false; for example, a
Mac with MLX and low free disk should still serve chat/inference/status work
but should not receive heavy checkpoint-producing training until its runtime
paths are moved to roomy storage.

Before walking away from a healthy private Hive, run one utilization dry run so
you can see whether safe idle slots will be fed:

```powershell
theseus utilize status
```

Vacation Mode Supervisor runs the same utilization sweep each cycle. For a
manual queue-fill pass:

```powershell
theseus utilize sweep --execute
```

For a foreground overnight run, use the loop form. It keeps filling only safe
private Hive slots, writes `reports/hive_utilization_manager.json` each pass,
and exits when `reports/hive_utilization_stop.flag` is written:

```powershell
theseus utilize loop --execute --sleep-seconds 60 --max-new-jobs 2
```

When the Mac is traveling or intentionally isolated from the rest of the Hive,
use the solo wrapper instead:

```powershell
theseus solo loop --execute --allow-battery --keep-awake --sleep-seconds 120 --max-new-jobs 1
```

The solo wrapper forces offline/local-only utilization, records a local
training ledger, activates only best-by-arm improvements, writes rollback
metadata, and produces `reports/hive_solo_overnight_report.json` / `.md`.

On macOS, append `--keep-awake` for a long local run that should prevent normal
idle sleep while the utilization process is alive. A truly sleeping Mac cannot
train; MacBook closed-lid work still depends on normal macOS clamshell
conditions such as AC power and external display/input.

The queue-filler priority is user/operator work first, then bounded CUDA/MLX
training, then CPU smoke training and maintenance, then grounded checkpointing.
It does not grant arbitrary shell, teacher apply, public benchmark training, or
unreviewed rented compute.

The phone operator view exposes the same state: coverage, blocked nodes,
recent sweeps, recent training jobs, network doctor state, and
pause/resume/stop/sweep controls. A YELLOW overnight state is acceptable only
when the blocker is understood, such as a sleeping or unreachable node.

After the run, summarize what happened:

```powershell
theseus train overnight
theseus solo overnight
```

This report lists worker inputs/outputs, arm id, owner node, backend, score,
merge result, promotions, failures, and stale leases that the next deterministic
round can recover.

Rented compute/storage remains reviewed and dry-run-first. Use it only when the
local queue is pressure-heavy, broad transfer is still below floor, and the
profile budget/time-window policy allows the session:

```powershell
theseus rent status
theseus rent plan --profile aws-gpu-nightly --task-kind cuda_training_chunk --hours 4
```

## Goal Routing

Use the dashboard "Autonomous Goals" panel to hand the system a goal. The head
routes the goal to specialist arms, applies permission envelopes, checks the
resource governor, and then either plans or executes bounded local commands.

Teacher escalation is sparse by design. It is used only when the local route has
low confidence, no bounded local tool exists, or repeated evidence says the
architecture is the wall.

To route a goal without starting training:

```powershell
py -3.13 scripts\autonomous_goal_runner.py --goal "Refresh resources and keep the system efficient." --profile smoke
```

To let the routed goal execute its bounded local commands, add `--execute`.
Teacher and network access still require `--allow-teacher` and
`--allow-network-fetch`.
