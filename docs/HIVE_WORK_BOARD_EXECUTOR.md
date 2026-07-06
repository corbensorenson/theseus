# Hive Work Board Executor

The Hive Work Board Executor makes the SQLite work board the source of truth for
unattended work.

```text
task -> node assignment -> run ledger -> evidence -> retry/block/done -> feedback
```

Run a status refresh:

```powershell
python scripts\hive_work_board_executor.py --status --out reports\hive_work_board_executor.json --markdown-out reports\hive_work_board_executor.md
```

Run one bounded board step:

```powershell
python scripts\hive_work_board_executor.py --execute --resume --max-tasks 1 --max-steps 1 --timeout-seconds 21600
```

Submit the first live command-channel proof:

```powershell
python scripts\hive_work_board_executor.py --command-text "/background run broad transfer status" --execute --max-tasks 1
```

## Reports

| Report | Purpose |
| --- | --- |
| `reports/hive_work_board.sqlite` | Durable task state. |
| `reports/hive_work_board_executor.json` | Latest executor status and selected task. |
| `reports/hive_work_board_execution_ledger.jsonl` | Append-only run ledger. |
| `reports/hive_live_command_ledger.jsonl` | Live dashboard/mobile/tray command intake. |
| `reports/hive_tool_hook_ledger.jsonl` | Before/after hooks for board/background/VIEA work. |
| `reports/hive_work_board_feedback.jsonl` | Feedback/residual routing from completed work. |
| `reports/hive_unattended_improvement_ledger.jsonl` | Improvement-contract outcomes for unattended tasks. |
| `reports/hive_teacher_auto_escalation_ledger.jsonl` | Residual-family auto-escalations into teacher architecture diagnosis. |
| `reports/report_evidence_store.sqlite` | Append-only report-run evidence DB used to avoid overwritten latest-report paths. |
| `reports/report_evidence_store.json` | Human-readable view of stored runs and best evidence by family. |
| `reports/hive_morning_report.json` | Human-readable overnight summary source. |
| `reports/hive_node_registry.json` | Authoritative Windows/Mac/Linux node view used by board placement, scheduler, readiness, and utilization. |
| `reports/hive_artifact_sync.json` | Remote evidence/checkpoint sync and merge status. |
| `reports/hive_overnight_proof.json` | One-night unattended proof gates: no unfed capable nodes, synced artifacts, progress signal, clean public-data checks. |

## Supported V1 Tasks

- VIEA feedback actions from `reports/feedback_action_queue.json`.
- Live command task: `/background run broad transfer status`.
- Live status tasks for Operator OS and Vacation Mode.
- High-transfer curriculum scheduler tasks:
  - `open_conversation_pantry`
  - `type_contract_diagnostic`
  - `type_contract_four_card_calibration`
  - `edge_exec_repair_four_card_calibration`
  - `execution_shaped_programs`
  - `execution_shape_private_ablation`
  - `execution_shaped_four_card_calibration`
  - `type_and_return_shape`
  - `typed_interface_skeleton`
  - `typed_interface_private_closure`
  - `private_pressure_private_closure`
  - `private_pressure_four_card_recalibration`
  - `admissibility_and_interface`
  - `edge_conditions`
  - `algorithmic_planning`
  - `multi_turn_conversation`
  - `multi_turn_conversation_hard`
  - `repo_repair`
  - `long_horizon_tool_use`
- Teacher architecture escalation tasks generated after repeated residual-family
  failures.

Unknown task kinds are blocked, not guessed.

## Improvement Contract

Every completed unattended board task is classified against the progress
contract. At least one signal must be produced:

- `private_residual_shrank`
- `public_transfer_improved`
- `new_clean_evidence_produced`
- `adapter_repaired`
- `teacher_experiment_spec_produced`
- `stale_tool_retired`
- `useful_failure_residual_captured`

If a task produces no signal, Vacation Mode records the family in
`reports/hive_no_progress_families.jsonl`. The first miss demotes that task
family; the second miss blocks it until the plan changes. If the same residual
family fails twice, the executor queues a teacher-as-architect task:

```text
residual cluster -> teacher diagnosis -> experiment spec -> private eval -> public calibration
```

Teacher tasks remain architecture-diagnosis only. Public benchmark answers and
public solution distillation are still forbidden.

## Vacation Mode

Vacation Mode Supervisor V3 consumes the board before running older loose-report
autonomy:

```powershell
python scripts\vacation_mode_supervisor.py --cycles 1 --execute --allow-teacher --max-actions-per-cycle 1
```

This keeps overnight/autonomous work tied to durable task status, retry counts,
evidence, and feedback routing.

## High-Transfer Lane

The high-transfer scheduler prevents benchmark-specific overfitting by queuing
concept pressure:

- `open_conversation_pantry`
- `type_contract_diagnostic`
- `type_contract_four_card_calibration`
- `edge_exec_repair_four_card_calibration`
- `execution_shaped_programs`
- `execution_shape_private_ablation`
- `execution_shaped_four_card_calibration`
- `type_and_return_shape`
- `typed_interface_skeleton`
- `typed_interface_private_closure`
- `private_pressure_private_closure`
- `private_pressure_four_card_recalibration`
- `admissibility_and_interface`
- `edge_conditions`
- `algorithmic_planning`
- `multi_turn_conversation`
- `multi_turn_conversation_hard`
- `repo_repair`
- `long_horizon_tool_use`

Run it directly:

```powershell
python scripts\high_transfer_curriculum_scheduler.py --out reports\high_transfer_curriculum_scheduler.json --markdown-out reports\high_transfer_curriculum_scheduler.md --tasks-out reports\high_transfer_curriculum_tasks.jsonl
```

Public benchmark cards remain receiver calibration surfaces only. Private/local
concept pressure is the training surface.

While `configs/autonomy_policy.json` declares
`english_multi_turn_conversation_before_returning_to_code_temporarily`, the
board ranks `open_conversation_pantry` and `multi_turn_conversation` ahead of
code-transfer work until the large conversation gate passes. The board runs
`multi_turn_conversation` with `--suite-mode large --case-limit 72`; smoke
success alone is not enough. After a 64+ case large suite clears the accuracy
floor and has no case-floor failures, the scheduler marks
`multi_turn_conversation` and the conversation pantry as `regression_only`.
Frontier slots then rotate back to transferable code semantics. The first code
slot, `type_contract_diagnostic`, writes private Decoder V2 feedback rows to
`D:/ProjectTheseus/training_data/high_transfer/private_train/type_contract_decoder_feedback.jsonl`.
After it writes a full feedback set, the scheduler marks it `regression_only`.
The first receiver check, `type_contract_four_card_calibration`, is a same-seed
32-case-per-card run across MBPP, EvalPlus, BigCodeBench, and LiveCodeBench.
After that run exposes a fresh teacher experiment, it also becomes
`regression_only`.

The current code-transfer frontier has split into two layers. The
`edge_exec_repair_four_card_calibration` lane verified the teacher-proposed
`edge_exec_repair_v1_private_first` stage: bounded private edge-condition
execution-feedback repair after Decoder V2 semantic planning, followed by the
same 32-case-per-card receiver calibration. May 19 runs produced clean
128-task evidence, moved LiveCodeBench from `1/32` to `7/32`, and kept
HumanEval, STS, leakage, template, wrapper, and task-regression gates clean.
That was useful but insufficient because BigCodeBench remained `0/32`.

The newer `execution_shaped_programs` lane targets the BigCodeBench hard zero
without training on public answers or tests. It writes private/generated
file/path/string, CSV/archive, JSON/payload, system/library-API, multi-step
state, edge-condition, and return-shape tasks, then exposes those concepts to
Decoder V2 through `type_contract_decoder_feedback.jsonl`. The same-seed
32-case-per-card receiver run first moved BigCodeBench to `1/32`, broad
aggregate to `0.45`, and LiveCodeBench/HumanEval/MBPP/EvalPlus, STS, leakage,
template, wrapper, and task-regression gates stayed clean. After the private
execution-shape ablation and skeleton decoder fixes, the next same-seed
32-case-per-card receiver run moved BigCodeBench to `5/32`, broad aggregate to
`0.475`, and still kept task regressions and no-cheat violations at `0`.

The 2026-05-20 Decoder V2 execution-shape patch added Windows `/tmp` sandbox
compatibility, stronger shell-command-output failure text, process-control
mock compatibility, and prompt-derived JSON schema/email validation. A
BigCodeBench 18-case diagnostic improved from `6/18` to `8/18`. A full
non-HumanEval 4-card diagnostic then reached MBPP `20/32`, EvalPlus `19/32`,
BigCodeBench `8/32`, and LiveCodeBench `6/32` (`53/128 = 0.414062`) with
clean no-cheat gates. This is board evidence of useful progress, not
promotion.

The board should treat high-transfer private-row balance as a hard improvement
precondition. `scripts/code_lm_closure.py` now round-robins capped
high-transfer rows across `type_and_return_shape`,
`type_contract_decoder_feedback`, `typed_interface_skeleton`,
`admissibility_and_interface`, `edge_conditions`, `algorithmic_planning`, and
`execution_shaped_programs`.
Before this fix, capped runs could load only the first two files and starve the
actual transfer concepts the board intended to test.

The board now has a private-only `typed_interface_private_closure` step between
fresh typed/interface pressure and any public four-card receiver run. On
2026-05-21 the selected tasks `transfer_1814815e34692322` and then
`transfer_994cc387a5212511` regenerated `typed_interface_skeleton` rows and
consumed them with public calibration skipped. It produced new clean Rust
closure evidence and improved private pass rate `0.083682 -> 0.39749`, with
private STS repair delta `+0.05021` across 12 task-level improvements and 0
regressions. The scheduler marks that closure complete while gating public
recalibration on a decoder/generator source change or stronger private gate
evidence, not merely fresh private rows.

The newer `private_pressure_private_closure` step is the current source-agnostic
private gate. It consumes private type/return-shape, type-contract feedback,
typed-interface skeleton, admissibility/interface, edge-condition, and
algorithmic-planning rows with public calibration skipped. The board executed
`transfer_d0325e3a88548e0e` and produced
`reports/code_lm_closure_private_pressure_private.json`: private pass rate
`0.062762 -> 0.485356`, private pass-rate delta `+0.422594`, next-token
accuracy delta `0.084484`, private STS repair delta `+0.05021`, 12 STS
task-level improvements, and 0 STS regressions. The scheduler now marks this
private closure current/regression-only and selects
`private_pressure_four_card_recalibration` as the next single receiver
measurement. That public task remains calibration-only and must not train on
public answers or hidden tests.

The `algorithmic_planning` lane now has a clean private pressure set:
`reports/high_transfer_algorithmic_planning_code_residual_curriculum.json`
generated 960 private rows with zero private solution failures for interval
merge, fixed-window sums, top-k frequency, graph reachability,
alternating-run-state, and minimum-subarray-length families. Decoder V2 also
uses STS streams during skeleton admission/ranking. The first full public
receiver attempt after enabling stratified admission regressed to `48/128`,
so that run is quarantined and the board should not rerun full public
calibration until the candidate hygiene wall is patched. The bounded
`patch9_hygiene_smoke8` remained clean at `16/32`, but private STS repair is
still not causal.

The scheduler now counts all completed `*4card` calibration reports when
deciding whether fresh private pressure has already been tested. This prevents
the board from re-queuing another full four-card public run immediately after a
diagnostic calibration. The broad matrix also selects best clean per-card
evidence instead of newest-file evidence, so lower-scoring diagnostic runs do
not overwrite stronger clean calibration rows.

The public receiver run is gated by private candidate coverage, not by hope.
The current `execution_shape_candidate_coverage_v10_current` private gate is
`GREEN`: learned-token decoding passes execution-shaped private held-out tasks
at `1.0`, has `0.0` no-admissible rate, covers archive/CSV/JSON/log/URL/system
families, and emits no diagnostic template candidates. The broader
`decoder_v2_private_ablation_gate` is still `YELLOW` because the last completed
receiver candidate manifest is stale and sparse: only about `0.101562` of public
receiver tasks have eligible learned-token candidates, with no-admissible around
`0.898438`. The gate now records no-admissible diagnostics with rejection
reasons, task contracts, required constructs, and sample bodies so the next
decoder patch can target the exact missing candidate families.

The board then executed the canonical `execution_shaped_four_card_calibration`
task (`transfer_32e7378cde10820f`). It ran the same-seed 32-case-per-card
receiver calibration across MBPP, EvalPlus, BigCodeBench, and LiveCodeBench.
The run completed as useful clean evidence (`49/128 = 0.382812`) with no
leakage, template, wrapper, external-inference, or task-regression violations:
MBPP `22/32`, EvalPlus `19/32`, BigCodeBench `4/32`, and LiveCodeBench `4/32`.
The best-clean-per-card broad matrix is now `0.50625`; only HumanEval is above
floor, so the next board pressure should target shared type/return-shape,
edge-condition, admissibility/interface, and algorithmic planning residuals
rather than blindly rerunning the same public calibration.

After that, the scheduler continues through source-agnostic private closure,
the decoder/private ablation gate, one receiver measurement only if the gate is
GREEN, repo repair, and long-horizon tool use unless conversation regresses.
The scheduler now ranks code-transfer concepts from
`reports/transfer_generalization_audit.json` instead of a fixed list: the
largest shared cross-card residual becomes critical, other shared residual
concepts become high priority, and a fresh private concept report can create a
`private_pressure_four_card_recalibration` task only when the public rerun gate
is justified by decoder/generator source change or stronger private gate
evidence. This prevents the board from both stockpiling private rows and
churning unchanged public receiver runs. Public calibration remains locked until
the fresh private closure refreshes the receiver candidate manifest and the
decoder gate proves broad candidate coverage.

The board
executor also retires older queued
high-transfer tasks for concepts that have since become `regression_only`, so
stale critical tasks from before graduation cannot keep being selected.
It also supersedes older scheduler task IDs for the same high-transfer concept
when `reports/high_transfer_curriculum_tasks.jsonl` emits a newer canonical
task. Same-priority selection prefers canonical high-transfer scheduler tasks
over loose feedback-action duplicates.

After the checkpoint-chat session-memory patch, `multi_turn_conversation_hard`
graduated to regression-only at `0.9772569444444442` across 96 cases / 217
turns with personality context ready on every turn. The board retires its
previous ready task as `graduated_to_regression_only` and returns selection to
the transfer/code or long-horizon lanes unless fresh conversation residuals
appear.

The board preserves terminal `done`/`blocked` state for a canonical scheduler
task ID, so refreshing the scheduler cannot reopen the same completed task.
New pressure still appears normally when the scheduler emits a new task ID
through a changed rotation epoch.

Private ablation gates use decoder-relevant source fingerprints instead of the
whole Rust file timestamp. This prevents unrelated bookkeeping changes, such as
work-budget admission policy updates, from reopening expensive decoder ablation
tasks. A fresh ablation is required only when new execution-shaped private
pressure appears or decoder-relevant source lines change.

High-transfer frontier scripts may return non-zero when the surface remains
YELLOW below graduation. The executor treats a non-zero return as completed
useful work when a valid GREEN/YELLOW frontier report was written and the
unattended improvement contract produced evidence. True RED reports, missing
reports, timeouts, and unsupported commands still fail or block. This keeps
Vacation Mode from wasting teacher calls on honest frontier evidence.

Feedback-action tasks are preflighted before selection. If a public card still
needs a 32+ clean calibration slice, the board blocks training/STS actions for
that card and leaves only the adapter-expansion action eligible. This prevents
overnight runs from burning cycles on STS ablations against two-task evidence.

The high-transfer scheduler uses `reports/report_evidence_store.sqlite` to pick
the strongest conversation report by graduation and case count, not the newest
smoke report, so dashboard/operator refreshes cannot accidentally demote the
large-suite graduation.

## Node-Aware Placement

The executor assigns work from `reports/hive_node_registry.json`, the same
authoritative view consumed by the Hive scheduler, utilization manager, and
fleet readiness check. That prevents the board from seeing one node while the
utilization layer sees another.

The executor assigns work by capability, trust, resource state, and version
state:

- CUDA-heavy semantic/code training goes to the best fresh Windows/CUDA node.
- MLX/inference/chat work can go to a Mac or inference-capable node when
  available. The current conversation-first board lane runs locally because it
  writes D:-based pantry artifacts and local checkpoint-chat reports.
- Storage, indexing, status, and light evaluation can run on CPU/coordinator
  nodes.
- Nodes with version drift are not selected for new work.
- Nodes below the heavy-training disk floor can still receive light/inference
  work, but they are not selected for checkpoint-producing training jobs until
  their runtime/data/cache paths are confirmed on roomy storage.

If the selected node is remote, the local executor keeps the task queued with
assignment evidence instead of pretending it ran locally. Remote nodes can then
pull the same board contract.

## One-Night Proof

Before trusting a long unattended run, use the diagnostic proof:

```powershell
python scripts\hive_overnight_proof.py --out reports\hive_overnight_proof.json --markdown-out reports\hive_overnight_proof.md
```

This is not promotion evidence. It is an operations gate that verifies the
fleet is visible, capable nodes are fed by board/scheduler/utilization work,
remote artifacts return and merge, at least one useful progress or residual
signal exists, public benchmark leakage checks are clean, and no task family is
stuck repeating without progress.
