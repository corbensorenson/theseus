# SparkStream Autonomy

SparkStream is the local control layer for Ratcheting Modular Intelligence. It keeps the system observable while it learns: cycles are logged, benchmarks are inventoried, checkpoints are recorded, teacher calls are sparse and auditable, and long runs stay behind explicit operator switches.

The implementation follows the RMI loop:

```text
observe current reports
  -> refresh resource governor
  -> refresh Project Theseus registration/license gates
  -> refresh benchmark/data inventory
  -> refresh personality core, runtime context, runtime audit, and drift gates
  -> refresh governed online source catalog
  -> sample approved external training data under tiny, license-gated caps
  -> curate governed synthetic training blend
  -> route the current goal through Octopus arms
  -> refresh Project Theseus Hive node/peer/scheduler state
  -> refresh arm lifecycle governance
  -> run a selected training profile
  -> refresh ratchet ledgers and gates
  -> refresh launch readiness
  -> refresh capability matrix and market posture
  -> refresh benchmark adapter cards, architecture experiments, loop closure,
     Autoresearch audit, and self-evolution governance
  -> refresh the VIEA autonomy spine
  -> materialize feedback, broad-transfer, repo-repair, SymLiquid, and
     teacher-architecture action queues
  -> run the VIEA action executor with step budgets when execute mode is enabled
  -> reduce candidate/runtime bottlenecks with policy-approved local setup
  -> run guarded teacher self-edit only if policy and local evidence allow it
  -> queue or call the teacher only when policy triggers
  -> checkpoint the cycle
  -> create/install accepted-candidate update offer if promotion passed
  -> update dashboard status
```

It also borrows the useful PlanForge discipline from your Drive notes: compile intent into bounded commands first, then run them through a visible watchdog instead of improvising a hidden agent loop.

`scripts/candidate_bottleneck_reducer.py` is the pre-teacher cleanup lane. In
execute mode it may create isolated local runtimes, refresh smoke reports, and
turn repeated setup failures into explicit manual/native blockers. It never
downloads bulk data, starts live hardware, or calls external inference. The
teacher should only see unresolved source bugs or architecture walls after this
lane has exhausted safe local fixes.

## Dashboard

Start the local dashboard:

```powershell
.\scripts\start_sparkstream.ps1
```

Open:

```text
http://127.0.0.1:8787
```

The dashboard shows:

- live SparkStream phase and recent report state;
- benchmark scores, thresholds, residuals, and wall type;
- the self-improvement queue;
- resource-governor status, GPU/VRAM budget, and efficiency score;
- Project Theseus Hive node/peer/scheduler status for distributed CUDA/MLX/CPU
  task routing;
- Project Theseus registration/license status for private Hive, worker chunk,
  company Hive, and public gateway gates;
- a per-device OpenAI-compatible endpoint toggle for local agent harnesses
  such as OpenCode or Hermes-style clients;
- launch readiness for autonomous training, teacher-enabled runs, and candidate
  promotion;
- personality runtime status, drift score, belief-governance status, and
  whether checkpoint/live chat is consuming the blessed personality context;
- arm lifecycle governance proposals for split, merge, update, registration,
  and deprecation review;
- autonomous goal routing through the Octopus arm registry;
- training-data inventory and RL benchmark registry;
- online source catalog for staged RL/benchmark/data candidates;
- governed external training samples and low-ratio pairwise distillation rows;
- synthetic-data curation quality, leakage, ratio, and artifacts;
- capability matrix maturity, market position, key gaps, and next actions;
- self-evolution lanes, guarded teacher self-edit status, architecture
  experiment queue, Autoresearch audit, benchmark adapter cards, and
  loop-closure candidates;
- VIEA OS status: SQLite artifact-kernel counts, latest command execution,
  feedback action queue, broad-transfer closure, repo-repair curriculum,
  action-executor state, broad-transfer closure, repo-repair curriculum and
  learner bridge, SymLiquid state-engine slots/route weights, and teacher
  architecture closure/runner;
- VIEA command-center controls: run/pause/resume/block the feedback action
  queue, request teacher architecture diagnosis, rerun broad calibration,
  refresh repo-repair learner traces, and refresh SymLiquid state;
- active background jobs;
- checkpoints, checkpoint materialization, and A/B checkpoint comparison;
- accepted-candidate checkpoint backup status;
- accepted-candidate update status, soft install controls, and hard-update
  restart guidance;
- live/checkpoint state chat;
- a benchmark/data source queue;
- a sparse teacher prompt box.

By default, dashboard actions queue or dry-run. The `execute` switch is required for real training actions. Candidate and seed-sweep profiles ask for long-run confirmation. Network source fetching is license-gated and bounded by policy. Hive creation, distributed worker chunks, company hives, and public gateway operation also pass through `reports/license_status.json`. Bulk training-data downloads, uncertain licenses, and commercial game assets remain blocked without explicit approval.

Current watchdog/readiness state, 2026-05-20:

```text
watchdog: YELLOW/RED only when an operational fault, stale promotion-facing governance report, or honest learning wall is present
promotion-facing frontier family: coding_local_sandbox
best clean public calibration card: source_human_eval, 32 tasks, pass rate 0.78125
below-floor receiver cards: EvalPlus 0.59375, BigCodeBench 0.25, LiveCodeBench 0.21875; MBPP is above floor at 0.71875
active transfer pressure: source-agnostic type/return-shape, edge-condition, admissibility/interface, and algorithmic-planning private concept rows after the execution-shape gate cleared
broad matrix: 160 public calibration tasks, aggregate pass rate 0.5125, STS delta 0.28125
BigCodeBench/LiveCodeBench: both have 32 clean tasks and both remain below floor; BigCodeBench is 0.25 and LiveCodeBench is 0.21875
stalled/next skipped card on this pass: no single benchmark card owns the frontier; scheduler ranks source-agnostic residual concepts
candidate gate: promote=false
learning scoreboard: YELLOW because broad public code transfer is below floor
personality runtime audit: GREEN
conversation lane: large 72-case surface graduated to regression; hard 96-case conversation stress lane is ready
Code LM closure: high-transfer private rows are now loaded by default up to 4,800 rows; latest board-run public receiver calibration cleared clean evidence gates but remains below promotion floor, so private type/interface/edge/algorithmic rows are training pressure while public cards stay calibration-only
Decoder V2: semantic skeletons and admissibility gates now target return shape, interface admissibility, edge conditions, branch/loop/local structure, and scalar/list/string family separation
VIEA autonomy spine: command -> artifact kernel -> runtime packet -> feedback action queue -> bounded action executor
overnight readiness: GREEN short proof with Windows CUDA + Mac MLX live, promotion blocked honestly
hive utilization: inner_loop queue filling targets depth 2, up to 4 jobs per sweep, and 900s accelerator leases for less bursty CUDA/MLX use
```

If the watchdog reports RED, apply the indicated local correction when it is an
operational fault. If the RED is an honest learning wall, keep promotion
blocked and report the wall.

Refresh the VIEA spine directly:

```powershell
py -3.13 scripts\viea_autonomy_spine.py --max-steps 64 --timeout-seconds 7200 --out reports\viea_autonomy_spine.json --markdown-out reports\viea_autonomy_spine.md
```

The spine is local-only and emits action queues. It does not turn VIEA scaffold
health into student promotion evidence; broad public transfer still decides
whether the learned student is improving.

Execute the approved feedback action queue directly:

```powershell
py -3.13 scripts\viea_action_executor.py --execute --resume --max-actions 3 --max-steps 8 --timeout-seconds 7200 --out reports\viea_action_executor.json --markdown-out reports\viea_action_executor.md
```

The executor uses action IDs, `reports/viea_action_execution_ledger.jsonl`,
pause/resume/block controls, and an explicit allowlist. It normalizes Windows
Python paths to the active interpreter to avoid path-specific access issues and
rejects shell constructs, public benchmark training outputs, teacher apply
mode, and any command outside the allowed local scripts.

Refresh the long-run readiness reports directly:

```powershell
py -3.13 scripts\arm_lifecycle_manager.py --out reports\arm_lifecycle_governance.json
py -3.13 scripts\autonomy_launch_readiness.py --profile inner_loop --out reports\autonomy_launch_readiness.json
```

## Context Packet Memory

SparkStream now treats long-horizon context as scored packets instead of one
ever-growing transcript. Reports, daemon events, routing traces, benchmark
outcomes, resource state, teacher results, and command outcomes are converted
into packets with importance scores.

Refresh the compact context view directly:

```powershell
py -3.13 scripts\context_packet_ledger.py --ingest-reports --compact --out reports\context_packet_ledger.json
```

The compactor keeps high-value conclusions active, merges related important
packets into summaries, reranks the active view, and marks low-value raw output
as drop candidates. The raw source remains referenced by path, so cleanup does
not destroy evidence.

## Synthetic Data Curation

SparkStream includes a governed local synthetic-data stage:

```powershell
py -3.13 scripts\synthetic_data_curator.py --policy configs\synthetic_data_policy.json --out reports\synthetic_data_curator.json
```

The curator reads residual escrow, generates BabyLM-style minimal-pair
candidates from local templates and feature-preserving mutations, rejects exact
overlaps with train/eval/holdout exclusions, scores quality, caps per-rule
share, and writes a real+synthetic blend capped by policy. If
`reports/training_data_sampler.json` is green, it may also include a tiny
low-ratio external pairwise distillation slice. It makes zero external
inference calls.

The one-command ratchet runner prepares this blend before the seed55 frontier
run. Candidate promotion checks synthetic governance only when the seed55 run
actually used the synthetic blend.

Details are in [Synthetic Data Curation](SYNTHETIC_DATA_CURATION.md).

## Governed External Training Samples

Discovery alone is not training. The autonomy loop now has a separate tiny
sample stage:

```powershell
py -3.13 scripts\training_data_sampler.py --policy configs\autonomy_policy.json --catalog configs\online_source_catalog.json --catalog-report reports\online_source_catalog_report.json --allow-network-fetch --sample-root D:/ProjectTheseus/training_data/governed_samples --out reports\training_data_sampler.json
```

The sampler only touches sources that are already in
`configs/online_source_catalog.json`, have an allowlisted data license, are
marked `metadata_only`, and provide an explicit `sampling` plan. It fetches
bounded Hugging Face dataset-viewer rows, writes ignored artifacts under
`D:/ProjectTheseus/training_data/governed_samples/`, rejects exact eval/holdout
overlap, deduplicates rows, records provenance, and derives a small local
minimal-pair artifact for low-ratio training use.

Current policy allows these governed samples for `smoke` and `inner_loop`
profiles only. Candidate-profile or bulk use still requires teacher/human
review. Knowledge sources such as Grokipedia remain lookup-only until their
license, terms, robots, and provenance gates are explicitly cleared.

## Capability Matrix

SparkStream maintains a live feature/capability matrix:

```powershell
py -3.13 scripts\capability_matrix.py --out reports\capability_matrix.json
```

The report reads local ledgers and compares them against the source-backed
market baseline in `configs/capability_market_baselines.json`. The dashboard
shows the lowest-maturity gaps first, and the autonomy cycle adds those gaps to
the self-improvement queue as `capability_gap` items.

Details are in [Capability Matrix](CAPABILITY_MATRIX.md).

## Project Theseus Hive

Start the integrated dashboard plus Hive node:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_theseus_hive.ps1
```

For server or terminal operation, install and use the `theseus` CLI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_theseus_cli.ps1
theseus status
theseus setup
theseus hive invite --out reports\hive_invite_private.json
theseus chat "summarize current benchmark status"
```

The Hive runtime is the app layer that other Windows/macOS machines run. It
advertises local CPU, RAM, disk, NVIDIA/CUDA, Rust, and optional Apple MLX
capacity; discovers trusted peers over LAN multicast; and writes:

```text
reports/hive_status.json
reports/hive_peers.json
reports/hive_scheduler.json
reports/hive_worker_chunk_ledger.jsonl
```

Remote work is not arbitrary shell access. The Hive worker accepts only task
kinds registered in `configs/hive_policy.json`, and off-loopback task
submission requires `THESEUS_HIVE_SECRET`. For home plus workshop plus phone,
use either a private VPN/tunnel or:

```powershell
py scripts\hive_invite.py create --tier private --relay-url "http://YOUR-RELAY-HOST:8793" --out reports\hive_invite_private.json
py scripts\hive_relay.py --port 8793
```

Phones can use the relay mobile page from the invite URL as a PWA/operator
client. Desktop/server machines apply the invite and run the Hive app.

Private Hive workers can now run real bounded training/eval chunks:

```powershell
theseus schedule --worker-chunks
theseus schedule --execute --worker-chunks
```

The registered chunk kinds are `cuda_eval_chunk`, `cuda_training_chunk`,
`cuda_rollout_chunk`, `mlx_eval_chunk`, `mlx_training_chunk`, and
`mlx_rollout_chunk`. They are
owned by `scripts/hive_worker_chunk.py`, which clamps resource parameters,
records CUDA/MLX telemetry, forbids teacher/external inference during worker
execution, and writes artifacts under `reports/hive_chunks/`.

Details are in [Project Theseus Hive](THESEUS_HIVE.md).

## OpenAI-Compatible Local Endpoint

For local agent harnesses that expect the OpenAI API shape, toggle the endpoint
from the dashboard or run:

```powershell
theseus openai start
```

Point the harness at:

```text
base_url=http://127.0.0.1:8789/v1
model=theseus-live
api_key=any-placeholder-value
```

The endpoint is a local compatibility shim. It speaks `/v1/models`,
`/v1/chat/completions`, and `/v1/completions`, but answers through the selected
Theseus checkpoint/live chat state and keeps `external_inference_calls=0`.
Teacher use is disabled by policy inside this shim.

## One-Cycle Runs

Dry-run one cycle:

```powershell
py -3.13 scripts\autonomy_cycle.py --profile smoke --out reports\autonomy_cycle_last.json
```

Execute a short local profile:

```powershell
py -3.13 scripts\autonomy_cycle.py --profile inner_loop --execute --out reports\autonomy_cycle_last.json
```

Allow a teacher call only when the policy triggers:

```powershell
py -3.13 scripts\autonomy_cycle.py --profile inner_loop --execute --allow-teacher --out reports\autonomy_cycle_last.json
```

Teacher calls use `configs/teacher_policy.json`. The default teacher is Codex CLI with `gpt-5.5` and `xhigh` reasoning. Ordinary teacher calls remain proposal-oriented. `scripts/teacher_oracle.py` wraps every request with a reason-specific intent, compact wall packet, local evidence summaries, anti-goals, and the JSON experiment-spec output contract, so the teacher is asked for one bounded diagnosis rather than a vague brainstorm. Source edits are only allowed through the guarded branch-and-gate lane in `scripts/teacher_self_edit_runner.py`, governed by `configs/self_evolution_policy.json`. Without `--allow-teacher`, requests are written to `reports/teacher_request_queue.jsonl` instead of being executed.

Verify the full-training teacher path before trusting an online run:

```bash
python3 scripts/theseus_cli.py train teacher-preflight --require-teacher-cli
python3 scripts/theseus_cli.py train teacher-preflight --require-teacher-cli --allow-teacher-live --require-live-teacher
```

The gate writes `reports/full_training_teacher_preflight.json` and `.md`. It
checks the Codex CLI, proposal-only policy, output schema, teacher budget audit,
queue-only behavior, apply-mode blocking, external-inference invariants, worker
ledgers, and the static handoff from `autonomy_cycle.py` to
`run_training_ratchet_profile.py` to `architecture_guidance_loop.py` to
`teacher_oracle.py`. A live run makes one bounded proposal-only teacher call.
`RED` means the teacher/control-plane contract is broken. `YELLOW` can be
acceptable when no current local report proves an architecture wall or when
general launch readiness is blocked by non-teacher prerequisites such as stale
candidate, CUDA, personality, or arm-governance reports.

On macOS, verify the local MLX/CPU lane separately before any full launch:

```bash
python3 scripts/theseus_cli.py mac training-preflight --execute --offline --allow-battery-smoke
python3 scripts/coherence_delirium_metric.py --out reports/coherence_delirium_report.json
python3 scripts/coherence_delirium_gate.py --out reports/coherence_delirium_gate.json
python3 scripts/autonomy_launch_readiness.py --profile smoke --require-teacher-cli --out reports/autonomy_launch_readiness.json
```

`ready_for_local_macos_smoke_training=true` means one bounded local Mac worker
can run. It does not mean `ready_for_autonomous_training=true`, and it does not
unlock candidate promotion, legacy-port runtime promotion, teacher apply mode,
or long battery training. The mobile/operator API exposes the result under
`training.mac_local`; the teacher gate remains under `training.teacher`.

## Self-Evolution Lane

Each autonomy cycle refreshes five self-evolution reports:

```powershell
py -3.13 scripts\attd_analyzer.py --out reports\attd_report.json --packets-out reports\attd_maintenance_packets.json
py -3.13 scripts\benchmark_adapter_factory.py --write-cards --out reports\benchmark_adapter_factory.json
py -3.13 scripts\architecture_experiment_governor.py --out reports\architecture_experiment_governance.json
py -3.13 scripts\autoresearch_gap_audit.py --out reports\autoresearch_gap_audit.json
py -3.13 scripts\loop_closure_harvester.py --out reports\loop_closure_harvester.json
py -3.13 scripts\self_evolution_governor.py --out reports\self_evolution_governance.json
```

ATTD runs first because it is the repo-health gate for autonomous source growth.
`GREEN` means normal operation, `YELLOW` means maintenance pressure with bounded
packets, and `RED` blocks long autonomy, ordinary teacher self-edit,
architecture change, and adapter-card writes except for ATTD maintenance work.
When packets exist under `YELLOW` or `RED`, self-evolution governance can trigger
the guarded teacher lane with `reason=attd_maintenance`.

If `reports/self_evolution_governance.json` says teacher apply is allowed and
the worktree is clean, the cycle can run:

```powershell
py -3.13 scripts\teacher_self_edit_runner.py --execute --allow-teacher --reason attd_maintenance --out reports\teacher_self_edit_last.json
```

The runner creates a `codex/self-evolution/<timestamp>` branch, asks the teacher
for the smallest patch, runs local checks, and leaves the branch for gate-based
review. It never performs destructive rollback. Dirty worktrees block automatic
apply mode so user changes are not mixed with autonomous source edits.

Guarded repairs are also treated as training evidence for future self-repair:
the runner appends compact before/after traces to
`reports/teacher_self_edit_traces.jsonl`, and loop closure can promote repeated
successful ATTD repairs into a local verified maintenance tool.

The Autoresearch audit imports the useful invariants from
`karpathy/autoresearch`: fixed-budget experiments, fixed metric surfaces, small
edit scopes, compact keep/discard/crash ledgers, crash repair limits, log
hygiene, and a simplicity preference. SparkStream remains multi-benchmark and
multi-arm, so the primary metric is the candidate/frontier gate portfolio rather
than one scalar `val_bpb`.

Append a compact experiment row after a profile run:

```powershell
py -3.13 scripts\autoresearch_experiment_ledger.py --profile inner_loop --append
```

Details are in [Self-Evolution System](SELF_EVOLUTION_SYSTEM.md).

## Resource Governor

Before a cycle or goal executes training work, SparkStream refreshes:

```text
reports/resource_governor.json
```

The governor checks the RTX 2060 Super profile, free VRAM, GPU utilization,
disk budget, active jobs, and configured profile limits. It records whether the
requested profile can run, which profile is recommended, and whether the hot
loop owner should remain Rust/CUDA.

Run it directly with:

```powershell
py -3.13 scripts\resource_governor.py --profile inner_loop --out reports\resource_governor.json
```

Current policy: Python orchestrates; Rust/CUDA owns rollout, scoring, optimizer
state, repeated eval kernels, and other hot loops.

## Performance Optimizer

Every autonomy cycle also refreshes:

```text
reports/performance_optimizer.json
reports/performance_optimizer.md
```

The optimizer reads the resource governor, Hive scheduler, CUDA/MLX worker
chunk ledger, and recent training reports. It records the preferred training
and inference backend, recent examples/sec, CUDA fallback state, MLX feature
cache status, bottlenecks, and next actions. This keeps speed work grounded in
local evidence: Windows/NVIDIA runs stay on Rust/CUDA, Apple Silicon machines
use MLX chunks, and teacher escalation is reserved for measured architecture
walls rather than ordinary profile tuning.

## Autonomous Goals

The dashboard "Autonomous Goals" panel sends a natural-language goal to:

```text
scripts/autonomous_goal_runner.py
```

The runner:

1. selects Octopus arms from `reports/arm_registry.json`;
2. grants permission envelopes;
3. checks `reports/resource_governor.json`;
4. plans bounded local commands;
5. executes only when `--execute` or the dashboard execute toggle is enabled;
6. queues/calls the teacher only when local confidence is too low and teacher
   use is explicitly allowed;
7. appends real traces to `reports/routing_memory_real_traces.jsonl`.

Those traces are consumed by `scripts/arm_lifecycle_manager.py` each autonomy
cycle so the arm registry can be governed from actual route usage, not only the
synthetic ORA benchmark cases.

Lifecycle thresholds and protected arms live in:

```text
configs/arm_lifecycle_policy.json
```

The current policy is report-only: metric refreshes can be automated, while
splits, merges, deprecations, permission changes, and new risky capabilities
remain teacher/human reviewed.

Example:

```powershell
py -3.13 scripts\autonomous_goal_runner.py --goal "Improve the frontier efficiently while preserving regressions." --profile smoke --out reports\autonomous_goal_last.json
```

## Daemon

Run repeated local cycles:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute
```

Run repeated local cycles with the sparse teacher enabled:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute -AllowTeacher
```

Restart the live dashboard and daemon on the latest checked-out code without
creating duplicate processes:

```powershell
.\scripts\start_sparkstream.ps1 -StartDaemon -Profile inner_loop -Execute -AllowTeacher -AllowNetworkFetch -Restart
```

The teacher remains proposal-oriented and governed by `configs/teacher_policy.json`.
Allowed teacher reasons currently have no daily call cap or cooldown so real
architecture walls are not starved; safety still comes from proposal-only mode,
local evidence requirements, no benchmark-answer distillation, and local
verification.
Current launch readiness says autonomous training and teacher-enabled operation
are ready, while candidate promotion is still blocked until a full `candidate`
profile supplies matched evidence for the promotion gate.

Stop the daemon by using the dashboard `Stop` button or creating:

```text
reports/sparkstream_stop.flag
```

## Benchmarks And Data Sources

Refresh the local benchmark inventory:

```powershell
py -3.13 scripts\benchmark_seeker.py --refresh-local --out reports\benchmark_seeker_registry.json
```

Queue a benchmark or dataset URL:

```powershell
py -3.13 scripts\benchmark_seeker.py --add-url "https://example.com/benchmark" --name "example" --notes "candidate source" --out reports\benchmark_seeker_registry.json
```

Fetch a source only when you explicitly want local network ingestion:

```powershell
py -3.13 scripts\benchmark_seeker.py --add-url "https://example.com/benchmark" --fetch-url "https://example.com/benchmark" --allow-network-fetch --out reports\benchmark_seeker_registry.json
```

Fetched artifacts are staged under `data/external_benchmark_candidates/` and should still go through benchmark audit before becoming a frontier or public calibration benchmark.

Discover public dataset candidates:

```powershell
py -3.13 scripts\benchmark_seeker.py --discover-query "grammar benchmark" --discover-limit 10 --allow-network-discovery --out reports\benchmark_seeker_registry.json
```

Discovery currently queries public dataset metadata and records candidates as `discovered_pending_audit`. This is intentionally not training ingestion. A discovered source still needs a benchmark card, contamination check, license/data review, split policy, and promotion into the benchmark ledger before it can pressure the ratchet.

Refresh the governed online source catalog:

```powershell
py -3.13 scripts\online_source_catalog.py --out reports\online_source_catalog_report.json
```

Stage approved source archives and dataset metadata under ignored folders:

```powershell
py -3.13 scripts\online_source_catalog.py --allow-network-fetch --import-sources --max-imports 12 --out reports\online_source_catalog_report.json
```

Stage approved open-source code repositories into the spillover-backed resource pantry:

```powershell
py -3.13 scripts\resource_pantry.py --execute --max-clones 24 --out reports\resource_pantry.json --markdown-out reports\resource_pantry.md
```

The pantry uses `configs/resource_pantry.json`, prefers the large spare drive,
and still keeps datasets metadata-only. The autonomy loop runs this with a small
per-cycle clone budget so it can keep adding ready benchmark/RL/eval source
material without blocking training for long.

The catalog currently prioritizes lightweight RL and benchmark sources such as
Gymnasium, Minigrid, bsuite, Craftax, Procgen, PettingZoo, Meta-World, and
Jumanji, Brax, dm_control, Crafter, `lm-evaluation-harness`, simple-evals,
LiveCodeBench, BigCodeBench, and inspect-evals, plus metadata-only
training-data candidates such as FineWeb-Edu, FineWeb, SmolLM Corpus, Dolma,
Cosmopedia, MMLU-Pro, and GPQA. These are staged candidates, not automatic
training inputs.

## Knowledge Sources

SparkStream keeps external knowledge lookup separate from training ingestion.
Refresh the registered knowledge-source gates with:

```powershell
py -3.13 scripts\knowledge_source_lookup.py --list --out reports\knowledge_source_registry.json
```

Plan a targeted lookup without fetching or ingesting content:

```powershell
py -3.13 scripts\knowledge_source_lookup.py --source grokipedia --query "topic to check" --out reports\knowledge_source_registry.json
```

Grokipedia is registered as `lookup_only_pending_terms_robots_license_audit`.
It may be used as a topic or claim-checking lead with provenance notes, but it
is blocked from autonomous bulk scraping, copied-page training, and model-output
distillation until the configured gates pass.

## Checkpoints

Create a checkpoint from current ledgers and reports:

```powershell
py -3.13 scripts\checkpoint_registry.py create --label manual_snapshot --reason operator_snapshot --profile inner_loop --status recorded --out reports\checkpoint_last.json
```

SparkStream checkpoints now use a major/minor chain:

- `major` checkpoints store a full workspace baseline for the configured code, docs, configs, benchmark snapshots, and selected local data files.
- `minor` checkpoints store a tamper-evident delta from the previous checkpoint: additions, deletions, text line transforms, and binary/full replacements only when a text transform is not safe.
- `auto` mode creates a major checkpoint when there is no valid parent or the minor chain gets too deep; otherwise it creates a minor checkpoint.
- every checkpoint stores a `state_hash` and `chain_hash`, so the chain behaves like a lightweight local ledger.

Force a full baseline:

```powershell
py -3.13 scripts\checkpoint_registry.py create --kind major --label major_baseline --reason full_baseline --profile inner_loop --status recorded
```

Force a delta from the latest checkpoint:

```powershell
py -3.13 scripts\checkpoint_registry.py create --kind minor --label minor_delta --reason ratchet_delta --profile inner_loop --status recorded
```

List checkpoints:

```powershell
py -3.13 scripts\checkpoint_registry.py list
```

Compare two checkpoints:

```powershell
py -3.13 scripts\checkpoint_registry.py compare --a CHECKPOINT_A --b CHECKPOINT_B --out reports\checkpoint_compare_last.json
```

Materialize a checkpoint into a separate folder:

```powershell
py -3.13 scripts\checkpoint_registry.py materialize --id CHECKPOINT_ID --out checkpoints\materialized\CHECKPOINT_ID
```

Materialization reconstructs the major baseline and then applies every minor transform in order. It writes into a separate directory; it does not restore over the active workspace.

Checkpoint creation also copies report artifacts up to the size limit in `configs/autonomy_policy.json`. Large reports and external datasets are referenced by hash/path or skipped by policy unless explicitly included.

Accepted-candidate backup is separate from ordinary checkpointing:

```powershell
py -3.13 scripts\checkpoint_backup_manager.py --if-promoted --execute --provider all
```

This command is safe to run every cycle. If `reports/candidate_promotion_gate.json`
has `promote=false`, it writes `skipped_not_promoted` and performs no off-machine
backup. If a candidate is actually accepted, it writes a small manifest under
`backup_manifests/accepted_candidates/` and can push the current git branch to
GitHub when the worktree is clean. Google Drive is queue-only by default through
`reports/google_drive_backup_queue.jsonl` until a connector/uploader worker is
configured. ROMs, datasets, generated reports, model binaries, and materialized
checkpoints are excluded by policy.

## Safety Defaults

SparkStream is intentionally local and conservative by default:

- resource governance runs before training profiles;
- arm lifecycle governance can block training if arm cards become invalid or
  routing selects an unregistered arm;
- launch readiness reports whether autonomous training, teacher-enabled runs,
  and candidate promotion are separately ready;
- teacher calls are queued unless `--allow-teacher` is set;
- external benchmark/data fetching is queued unless `--allow-network-fetch` is set;
- candidate and seed-sweep profiles require deliberate launch;
- autonomous git mutation is forbidden by policy;
- candidate promotion remains gate-only;
- checkpoint restore over the active workspace is not automatic.

This gives the project the useful kind of aliveness: it can keep observing, testing, queuing, checkpointing, and improving, while high-impact boundary crossings remain visible.
