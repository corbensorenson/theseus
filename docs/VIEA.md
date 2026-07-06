# Verified Intent-To-Execution Architecture

VIEA is the canonical north-star architecture for Project Theseus. It is the
systems-language version of the Reality Manipulator / Grimoire / Genesis /
Octopus / Capability Ratchet ideas:

```text
Intent
  -> Command Contract
  -> Artifact Graph
  -> Specialist Execution
  -> Runtime Target
  -> Verification
  -> Feedback
  -> Improved System
```

The core rule is:

```text
nothing important should remain only as a chat response
```

If it matters, it becomes an artifact. If it claims something, it gets a
support state. If it repeats, it can become a verified tool. If it fails, it
becomes a residual. If it is mastered, it becomes regression coverage. If it
executes into shared reality, it must pass the right gate.

## Kernel, Control, Expansion

The latest VIEA paper sharpens the build order into three layers:

| Layer | Role in Theseus |
| --- | --- |
| VIEA Kernel | artifact store, claim ledger, critique log, release manifest, feedback record, and resource log. This is the first useful substrate and should stay low-friction. |
| VIEA Control Layer | command contracts, provenance, retention, evidence hierarchy, permissions, routing policy, integration checks, side-effect classes, and resource/latency budgets. |
| VIEA Expansion Layer | specialist arms, workflow-to-tool compilation, evaluation ratchets, runtime adapters, spatial workspaces, hardware/fabrication/robotic handoffs, and autonomous feedback loops. |

Named primitives carry maturity labels. A concept starts at `M0`, becomes a
schema at `M1`, policy at `M2`, instrumented mechanism at `M3`, validated
mechanism at `M4`, and governed infrastructure at `M5`. Runtime adapters use
`R0` contract-only through `R5` closed-loop runtime. Do not treat a named
primitive as solved just because the architecture has a word for it.

## Theseus Mapping

| VIEA subsystem | Theseus implementation |
| --- | --- |
| Structured Command Layer | `scripts/reality_manipulator.py`, `reports/reality_manipulator/latest_world/command_contract.json` |
| Artifact Graph | Reality Manipulator world bundle plus Genesis kernel reports |
| Claim and Verification Ledger | `claim_ledger.json`, `critique_log.json`, deterministic taming, student-first audits |
| Orchestrator / Router | Octopus router, routing memory, SparkStream autonomy cycle |
| Specialist Modules | Arm lifecycle governance, cell lifecycle, arm/sucker registries |
| Workflow-To-Tool Compiler | loop-closure harvester/promoter, tool registry, workflow traces |
| Evaluation Ratchet | learning scoreboard, Benchmaxx, broad transfer matrix, candidate gate, residual escrow |
| Runtime Adapters | Reality Manipulator compile targets, benchmark adapter factory, Hive scheduler, resource pantry |
| Feedback Loop | autonomy ledgers, daemon ledgers, checkpoint registry, learning scoreboard |
| Student Learning Proof Layer | broad public transfer matrix, real code graduation, student-first evidence audit |
| Artifact Kernel | `reports/viea_artifact_kernel.sqlite`, `reports/viea_artifact_kernel.json` |
| Command Contract Executor | `reports/viea_command_executor.json` |
| VIEA Action Executor | `scripts/viea_action_executor.py`, `reports/viea_action_executor.json`, `reports/viea_action_execution_ledger.jsonl` |
| Private Repo-Repair Curriculum | `reports/private_repo_repair_curriculum.json` |
| Repo-Repair Learner Bridge | `scripts/viea_repo_repair_learner.py`, `reports/viea_repo_repair_learner.json`, `reports/repo_repair_trace_checkpoint.json` |
| SymLiquid State Engine | `scripts/symliquid_state_engine.py`, `reports/symliquid_state_engine.json` |
| Teacher-As-Architect Loop | `reports/teacher_architect_loop.json`, `reports/teacher_architect_experiment_runner.json` |
| Digital Runtime Adapter | `reports/digital_runtime_adapter.json` |
| Feedback Ratchet | `reports/feedback_ratchet.json` |
| Transfer Generalization Guard | `scripts/transfer_generalization_audit.py`, `reports/transfer_generalization_audit.json` |
| VIEA Autonomy Spine | `reports/viea_autonomy_spine.json`, `reports/feedback_action_queue.json` |
| Vacation Mode Supervisor | `scripts/vacation_mode_supervisor.py`, `reports/vacation_mode_supervisor.json`, `reports/vacation_mode_failure_triage.json`, `reports/vacation_mode_repair_action_queue.json` |

The generated machine-readable map is `reports/viea_report_map.json`; the
human-readable report is `reports/viea_report_map.md`.

Refresh it with:

```powershell
python scripts\viea_report_map.py --out reports\viea_report_map.json --markdown-out reports\viea_report_map.md
```

## MVP Object Bundle

The Reality Manipulator MVP must emit this bundle:

| File | VIEA role |
| --- | --- |
| `world.json` | project world and raw goal |
| `command_contract.json` | first-class structured command contract |
| `artifacts.json` | typed artifacts and relationships |
| `claim_ledger.json` | claim support states and risk |
| `critique_log.json` | critiques, blockers, and recommendations |
| `structured_output.md` | human-readable implementation handoff/spec |
| `release_manifest.json` | internal release snapshot and gates |
| `primitive_registry.json` | reusable primitive candidates |
| `feedback_plan.md` | reality feedback plan |
| `specialist_lifecycle.json` | lifecycle governance for selected arms |
| `workflow_tool_metrics.json` | loop-closure/tool-compiler metrics |
| `resource_log.jsonl` | resource accounting events for the sidecar kernel |

The bundle proves artifact preservation and governance. It does not prove model
learning.

Refresh the bundle with:

```powershell
python scripts\reality_manipulator.py --out reports\reality_manipulator.json --markdown-out reports\reality_manipulator.md --bundle-dir reports\reality_manipulator\latest_world
```

## Learning Proof Boundary

VIEA scaffold health is necessary but not sufficient for student improvement.
Theseus can only claim learned public-code progress when the student-first
proof layer is clean:

- token-level learned student code generation;
- full-body candidates, not wrappers/templates/rankers/loop tools;
- public benchmark solutions and hidden tests absent from training;
- broad public transfer matrix above the promotion floor;
- no regressions and no no-cheat violations.

Current public benchmark scores are calibration-only. Use
`reports/learning_scoreboard.json`, `reports/broad_transfer_matrix.json`, and
`reports/student_first_evidence_audit.json` before quoting learning progress.

The practical consequence is simple: VIEA can be GREEN while candidate
promotion remains blocked. That is expected. VIEA proves the system is
preserving, routing, gating, and learning from work; the student-learning proof
layer proves the small learner is actually transferring across broad public
calibration without leakage.

## Generalization Guardrail

Benchmaxxing is useful only when it pressures transferable concepts. The system
must not optimize one benchmark card until it looks good while adjacent cards
stay weak. The guardrail report is:

```powershell
python scripts\transfer_generalization_audit.py --out reports\transfer_generalization_audit.json --markdown-out reports\transfer_generalization_audit.md
```

It reads only public calibration metrics, residual labels, and task ids. It
does not read public answers or hidden tests. It measures:

- above-floor transfer card count;
- aggregate pass rate and cross-card pass-rate spread;
- shared residual concepts across cards;
- per-card STS causality;
- benchmark-name-specific private curriculum concentration.

Current shared transfer targets are `type_and_return_shape`,
`admissibility_and_interface`, and `edge_conditions`. Private pressure should
prefer source-agnostic concept families and donor/receiver evaluation over
benchmark-name families.

## VIEA Safety Controls Adopted From The Paper

Theseus should treat auto-drafted commands as proposals. Medium/high-impact
commands need an intent checksum and assumption diff. Model consensus alone
cannot verify a claim; verification needs source evidence, executable checks,
independent tools, expert review, or field feedback. Any output that can mutate
external state needs a side-effect class, reversibility flag, and compensation
plan. Human review is a resource, not a magic stamp, and resource events are
stored in the VIEA kernel as `ResourceEvent` objects.

## Implementation Order

1. Preserve intent as `command_contract.json`.
2. Preserve work as artifacts, claims, critiques, release manifests, and
   feedback plans.
3. Route work through lifecycle-governed specialist arms.
4. Compile repeated successful workflows into measured tool candidates.
5. Evaluate capability through the ratchet and residual escrow.
6. Target runtimes only after the proper gates exist.
7. Feed execution feedback back into artifacts, tools, benchmarks, and arms.

## Growth Loop Commands

The current VIEA control path is refreshed by the autonomy spine:

```powershell
python scripts\viea_autonomy_spine.py --max-steps 64 --timeout-seconds 7200 --out reports\viea_autonomy_spine.json --markdown-out reports\viea_autonomy_spine.md
```

For sleep/vacation autonomy, use the V2 supervisor:

```powershell
python scripts\vacation_mode_supervisor.py --cycles 0 --execute --allow-teacher --start-services --sleep-seconds 300 --out reports\vacation_mode_supervisor.json --markdown-out reports\vacation_mode_supervisor.md
```

Its operating contract is stricter than the spine alone:

- hard gates for stop/pause flags, disk, spillover storage, GPU reserve,
  public-data guard, and personality core;
- soft service probes for dashboard and Hive APIs;
- failed action -> triage class -> repair queue -> retry once -> teacher
  architecture diagnosis if still stuck;
- progress contract requiring each cycle to produce improvement, new clean
  evidence, a useful residual diagnosis, a repaired adapter, exploration
  output, or a teacher experiment step;
- optional exploration through governed source/catalog refresh, never bulk
  public-data ingestion or commercial game asset downloads.

That single runner executes the local VIEA spine:

```text
goal/command
  -> VIEA executor
  -> artifact kernel write
  -> runtime packet
  -> verification
  -> feedback ratchet
  -> next training/tool/residual action
```

It also emits:

| Report | Purpose |
| --- | --- |
| `reports/feedback_action_queue.json` | Concrete next actions from feedback ratchet, broad transfer, repo repair, tools, SymLiquid, and teacher architecture closure. |
| `reports/viea_action_executor.json` | Bounded executor status for approved local queue actions, with step budgets, pause/resume/block controls, and resume ledger. |
| `reports/broad_transfer_action_queue.json` | Per-card public-calibration blockers and allowed private-training/calibration actions. |
| `reports/repo_repair_main_curriculum.json` | Private SWE-style repo-repair loop status and next training pressure. |
| `reports/viea_repo_repair_learner.json` | Validated private repo-repair traces and governed Code LM rows for the learner; not promotion evidence. |
| `reports/symliquid_state_engine_queue.json` | SymLiquid state slots for command routes, residuals, tools, STS, repo repair, autonomy, and control policies. |
| `reports/symliquid_state_engine.json` | Live SymLiquid route weights consumed by the action executor and teacher runner. |
| `reports/teacher_architect_closure.json` | Residual -> diagnosis -> experiment -> private eval -> public calibration -> promote/rollback loop. |
| `reports/teacher_architect_experiment_runner.json` | Bounded teacher-as-architect stage runner; proposal-only teacher, private eval, public calibration only. |
| `reports/teacher_architect_experiment_runner_status.json` | Non-destructive queued/status view used by the autonomy spine so executed runner evidence is not overwritten. |

The spine is wired into `scripts/run_training_ratchet_profile.py`,
`scripts/autonomy_cycle.py`, the watchdog, the SQLite artifact kernel, and the
dashboard VIEA panels.

The underlying manual growth layer still runs in this order:

```powershell
python scripts\reality_manipulator.py --out reports\reality_manipulator.json --markdown-out reports\reality_manipulator.md --bundle-dir reports\reality_manipulator\latest_world
python scripts\viea_artifact_kernel.py --reset --db reports\viea_artifact_kernel.sqlite --out reports\viea_artifact_kernel.json --markdown-out reports\viea_artifact_kernel.md
python scripts\viea_command_executor.py --db reports\viea_artifact_kernel.sqlite --out reports\viea_command_executor.json --markdown-out reports\viea_command_executor.md
python scripts\long_horizon_programming_curriculum.py --out reports\private_repo_repair_curriculum.json --markdown-out reports\private_repo_repair_curriculum.md --repetitions 8
python scripts\viea_repo_repair_learner.py --out reports\viea_repo_repair_learner.json --markdown-out reports\viea_repo_repair_learner.md
python scripts\viea_growth_surfaces.py --out-dir reports
python scripts\symliquid_state_engine.py --out reports\symliquid_state_engine.json --markdown-out reports\symliquid_state_engine.md
python scripts\teacher_architect_experiment_runner.py --max-experiments 1 --max-steps 0 --out reports\teacher_architect_experiment_runner_status.json --markdown-out reports\teacher_architect_experiment_runner_status.md
python scripts\viea_action_executor.py --status --out reports\viea_action_executor.json --markdown-out reports\viea_action_executor.md
python scripts\viea_artifact_kernel.py --reset --db reports\viea_artifact_kernel.sqlite --out reports\viea_artifact_kernel.json --markdown-out reports\viea_artifact_kernel.md
python scripts\viea_report_map.py --out reports\viea_report_map.json --markdown-out reports\viea_report_map.md
```

This gives Theseus a real local object store, runnable command contracts,
digital runtime packets, private repo-repair pressure, a learner bridge,
stricter tool scoring, SymLiquid route-state, teacher-as-architect experiment
staging, and a feedback ratchet.

To execute approved queue actions:

```powershell
python scripts\viea_action_executor.py --execute --resume --max-actions 3 --max-steps 8 --timeout-seconds 7200 --out reports\viea_action_executor.json --markdown-out reports\viea_action_executor.md
```

The executor normalizes Windows Python paths to the current interpreter, never
uses shell interpretation, records `reports/viea_action_execution_ledger.jsonl`,
and refuses public benchmark training paths. Public benchmark commands are
calibration-only.

## Non-Claims

VIEA does not claim the full system already exists, that verification is
absolute, that every repeated workflow should become a tool, that specialist
modules always beat monoliths, or that AI should execute all user intent
automatically. It is an architecture for making intent durable, inspectable,
bounded, and progressively more executable.
