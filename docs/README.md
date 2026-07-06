# Documentation Index

Last consolidated: 2026-06-05.

Start here:

- [Verified Intent-To-Execution Architecture](VIEA.md): canonical north-star
  architecture for the whole project. It maps intent, command contracts,
  artifact graphs, specialist routing, workflow-to-tool compilation,
  evaluation ratchets, runtime adapters, feedback, and student-learning proof
  into the concrete Theseus reports.
- [Project Theseus Whitepaper](PROJECT_THESEUS_WHITEPAPER.md): standalone
  top-to-bottom explanation of the project, including SymLiquid, SparkStream,
  Octopus routing, RMI ledgers, self-evolution, checkpoints, Hive, licensing,
  updates, performance policy, and safety boundaries.
- [Top-To-Bottom System Architecture](TOP_TO_BOTTOM_ARCHITECTURE.md): the
  operational map of the whole system from dashboard and daemon through
  autonomy cycle, arms, ledgers, teacher, checkpoints, data ingress, and
  Rust/CUDA training paths.
- [Project State](PROJECT_STATE.md): current operational truth, live benchmark
  state, gates, resource policy, autonomy controls, and next work.
- [Replication Guide](REPLICATION_GUIDE.md): practical setup, report refresh,
  learning-run, anti-cheat, and troubleshooting runbook for reproducing the
  current local system.
- [Mac Handoff 2026-06-03](MAC_HANDOFF_2026_06_03.md): transfer note for
  moving the current Windows CUDA state to Apple Silicon Codex/MLX.
- [Travel Parent Demo](THESEUS_TRAVEL_PARENT_DEMO.md): calm MacBook demo path
  for non-specialist family/friend audiences.
- Current snapshot: code-family pressure is still the promotion-facing wall.
  STS is default-on for the current Code LM fanout path, decoder/transfer/STS
  private gates are `GREEN`, and candidate promotion/model growth remain
  locked. The Mac sandbox path issue is fixed by a private/local regression.
  The single bounded June 5 public calibration now scores `14/32` (`0.4375`)
  with STS delta `+0.15625`, still below the `0.70` promotion floor. The v2
  private residual curriculum generator now covers edge contracts,
  candidate-floor adapters, return/type shape, and parsing/encoding, but the
  public score did not improve, so the next cycle must stay private until
  verifier-mismatch behavior changes. Use [Project State](PROJECT_STATE.md)
  before quoting any benchmark number.
- Benchmaxxing now has an explicit generalization guard:
  `reports/transfer_generalization_audit.json` is `YELLOW` because only
  HumanEval and MBPP are above floor and cross-card spread is high. The current
  shared transfer targets are type/return-shape, edge conditions, and
  admissibility/interface, with algorithmic planning concentrated on
  BigCodeBench. Private pressure should target those source-agnostic concepts,
  not benchmark names.
- VIEA is now an autonomy spine, not just a report family:
  `reports/viea_autonomy_spine.json` runs command contracts through executor,
  artifact kernel, runtime packets, verification, feedback ratchet, and queued
  next actions; `reports/viea_action_executor.json` can run approved local
  queue actions with step budgets and a resume ledger; the dashboard exposes
  VIEA OS, feedback actions, broad transfer closure, repo repair, SymLiquid
  state slots/route weights, and teacher architecture closure/runner controls.
- Vacation Mode Supervisor V3 is the preferred long-unattended runner:
  `reports/vacation_mode_supervisor.json` consumes the Hive work board, wraps
  the VIEA spine with hard unattended gates, failed-action triage, a repair
  queue, service restart, optional governed exploration, and a per-cycle
  progress contract.
- Full-training teacher readiness now has its own gate:
  `python3 scripts/theseus_cli.py train teacher-preflight --require-teacher-cli`
  verifies proposal-only policy, queueing, apply-mode blocking, worker
  teacher-free invariants, and the autonomy-to-teacher handoff. Add
  `--allow-teacher-live --require-live-teacher` for one bounded live Codex
  proposal smoke.
- [SparkStream Autonomy](SPARKSTREAM_AUTONOMY.md): dashboard, daemon, goal
  routing, launch readiness, teacher policy, checkpoints, and safety defaults.
- [Project Theseus Hive](THESEUS_HIVE.md): click-friendly setup wizard,
  server-friendly `theseus` CLI, cross-platform signed LAN/private-tunnel node
  discovery with durable peer registry, no-cost WireGuard/relay remote-access
  setup, phone QR joining, Hive
  profiles, opt-in public idle-compute contribution, CUDA/MLX/CPU capability
  advertisement, Intel Mac CPU/storage/operator nodes, resource-aware task
  routing, per-user family/operator access tokens, decentralized arm-level
  CUDA/MLX training orchestration, always-busy utilization sweeps and loops,
  reviewed rented compute/storage planning, verified Hive-version convergence, any-node
  phone and Apple Watch operator control, audited remote-control handoffs,
  bounded storage shares, room-aware voice following,
  and cross-platform installer/package handoff.
- [Hive Operator OS](HIVE_OPERATOR_OS.md): one-agent-many-channels command
  vocabulary, durable Hive work board, background jobs, persistent goals,
  Hive Skills, dynamic skill loading/hygiene, tool hooks, rollback/worktree
  safety, feedback routing, and the manifest-driven app surface for dashboard,
  mobile, CLI, tray/menu bar, relay, and future chat-channel adapters.
- [Hive Work Board Executor](HIVE_WORK_BOARD_EXECUTOR.md): board-driven
  unattended work loop, live `/background` command channel, high-transfer lane,
  node assignment/version drift checks, retries, evidence, and feedback routing.
- [Project Theseus Licensing System](LICENSE_SYSTEM.md): local registration,
  community-use limits, signed paid-license checks, feature gates, and release
  caveats.
- [Self-Evolution System](SELF_EVOLUTION_SYSTEM.md): guarded teacher
  self-editing, benchmark adapter factory, architecture experiment governance,
  Autoresearch-style experiment outcome governance, ATTD repo-health
  governance, and loop-closure harvesting.
- [Benchmaxx Curriculum](BENCHMAXX_CURRICULUM.md): predefined capability
  course from substrate and BabyLM through RL, emulator environments, coding,
  web/desktop agents, native voice, and end-to-end autonomous user-agent
  behavior.
- [Reality Manipulator](REALITY_MANIPULATOR.md): deterministic
  intent-to-artifact world compiler and current VIEA vertical MVP.
- [Genesis Kernel](GENESIS_KERNEL.md): artifact-substrate layer that compiles
  live Theseus evidence into durable artifacts, claims, critiques, primitive
  candidates, release manifests, and feedback plans.
- [Context Packet Memory](CONTEXT_PACKET_MEMORY.md): scored packetized context,
  compaction, summaries, and active-context retention policy.
- [Theseus Personality Core](PERSONALITY_CORE.md): local user-owned personality
  document distillation, best-self memory, privacy posture, retrieval cards,
  runtime audit, launch-readiness gate, and future adapter/steering lanes.
- [Checkpoint Backups](CHECKPOINT_BACKUPS.md): accepted-candidate-only GitHub
  manifest/push backup and Google Drive queue policy.
- [Project Theseus Candidate Updates](THESEUS_UPDATES.md): accepted-candidate
  update offers, soft versus hard installs, protected local/company arms,
  dashboard controls, CLI, and Hive propagation.
- [Theseus Compute Market](THESEUS_COMPUTE_MARKET.md): internal work-credit
  gas quotes, verified work receipts, provider payouts, rental plans, and the
  disabled-until-reviewed public token/exchange path.
- [Synthetic Data Curation](SYNTHETIC_DATA_CURATION.md): residual-targeted
  synthetic data, quality/leakage gates, blend policy, and model-collapse
  guardrails.
- [Old Projects Transfer Audit](OLD_PROJECTS_TRANSFER_AUDIT.md): canonical
  predecessor-project audit for `D:\old_projects`, consolidating concept
  coverage, live legacy mechanisms, old-registry training/RL imports, completion
  state, activation blockers, and next actions.
- [Capability Matrix](CAPABILITY_MATRIX.md): live feature/capability matrix,
  local evidence, market baselines, maturity, gaps, and next actions.
- [Online Source Catalog](ONLINE_SOURCE_CATALOG.md): governed staging for
  online RL environments, benchmark frameworks, and training-data metadata.
- [ROM/RL/Data Growth Lanes](ROM_RL_DATA_GROWTH_LANES.md): local user ROM
  inventory, emulator RL wrappers, open RL frontiers, and benchmark/data source
  expansion policy.
- [PufferLib 4 RL Lane](PUFFERLIB4_RL_LANE.md): governed fast RL pressure lane
  and native-backend gate.
- [Autonomous Weeks Runbook](autonomous_weeks_runbook.md): concise operator
  guide for running the system for long periods.
- [Real Training Preflight](REAL_TRAINING_PREFLIGHT.md): gates that must stay
  green before longer training or ratcheting runs.
- [Data And Artifacts](DATA_AND_ARTIFACTS.md): tracked source, ignored runtime
  artifacts, third-party data boundaries, and license scope.
- Native voice data is governed by `configs/native_voice_training_policy.json`
  and `scripts/native_voice_training_manifest.py`; STT/TTS remains native
  Theseus learning, not installed/provider speech inference. Hive voice
  following uses score-only room presence and does not relay raw audio.

## Current Architecture Docs

| Document | Use |
| --- | --- |
| [Verified Intent-To-Execution Architecture](VIEA.md) | Canonical north-star architecture and report/dashboard subsystem map. |
| [Ratcheting Modular Intelligence](RATCHETING_MODULAR_INTELLIGENCE.md) | How RMI maps into the concrete local artifacts. |
| [Ratcheting Generative Systems](RATCHETING_GENERATIVE_SYSTEMS.md) | Benchmark pressure, residual escrow, loop closure, and regression lock-in. |
| [Capability Ratchet](CAPABILITY_RATCHET.md) | Combined benchmark/procedural/structural ratchet. |
| [Octopus Router](OCTOPUS_ROUTER.md) | Head/router, arms, learned router head, permissions, safety, and dynamic loading. |
| [Arm-Sucker Transfer Hierarchy](ARM_SUCKER_HIERARCHY.md) | Parent arms, task suckers, deterministic grammar/verifier suckers, and lifecycle transfer rules. |
| [Cognitive Loop Closure](COGNITIVE_LOOP_CLOSURE.md) | Tool formation from repeated trajectories. |
| [Benchmaxxing](BENCHMAXXING.md) | Benchmark lifecycle and anti-Goodhart policy. |
| [Reality Manipulator](REALITY_MANIPULATOR.md) | Intent-to-artifact world compiler and VIEA vertical MVP. |
| [Genesis Kernel](GENESIS_KERNEL.md) | Durable artifact substrate for claims, critiques, release manifests, primitive candidates, and feedback. |

## Substrate And Background Docs

| Document | Status |
| --- | --- |
| [CGS](CGS.md) | Compact Generative Systems framing for the SymLiquid substrate. |
| [Training, Evals, Benchmarks](TRAINING_EVALS_BENCHMARKS.md) | Broad technical reference, including the current train-once Code LM fanout/provenance contract. Use [Project State](PROJECT_STATE.md) for current scores before quoting numbers. |
| [BabyLM And Parameter Golf Transfer](BABYLM_PARAMETER_GOLF_TRANSFER.md) | Background transfer strategy from related local work. |
| [Architecture Gate](ARCHITECTURE_GATE.md) | Short gate summary. Current detailed gate status is in [Project State](PROJECT_STATE.md). |
| [PufferLib 4 RL Lane](PUFFERLIB4_RL_LANE.md) | Governed fast RL pressure lane and native-backend readiness policy. |

Retired background drafts:

- `deprecated/docs/background/WHITEPAPER.md`: old SymLiquid FEP-Net draft.
  Use [Project Theseus Whitepaper](PROJECT_THESEUS_WHITEPAPER.md), [CGS](CGS.md),
  and [Top-To-Bottom System Architecture](TOP_TO_BOTTOM_ARCHITECTURE.md)
  instead.
- `deprecated/docs/background/STANDALONE_PARITY_PLAN.md`: old standalone
  parity command log. Use [Project State](PROJECT_STATE.md),
  [Training, Evals, Benchmarks](TRAINING_EVALS_BENCHMARKS.md), and
  [Real Training Preflight](REAL_TRAINING_PREFLIGHT.md) instead.

## Consolidation Policy

This docs folder keeps one active truth layer and several background design
papers. To reduce clutter:

- current state belongs in [Project State](PROJECT_STATE.md);
- operator flow belongs in [SparkStream Autonomy](SPARKSTREAM_AUTONOMY.md) and
  [Autonomous Weeks Runbook](autonomous_weeks_runbook.md);
- broad architecture belongs in
  [Verified Intent-To-Execution Architecture](VIEA.md) for the north-star
  system contract and [Top-To-Bottom System Architecture](TOP_TO_BOTTOM_ARCHITECTURE.md)
  for the operational implementation map;
- older conceptual papers are kept as background and should not be quoted for
  live scores unless they agree with [Project State](PROJECT_STATE.md) and the
  generated JSON reports.
- duplicate old-project redirect stubs live under
  `deprecated/docs/legacy-transfer/`; update
  [Old Projects Transfer Audit](OLD_PROJECTS_TRANSFER_AUDIT.md) instead of
  reviving those filenames in active docs.
- retired background drafts live under `deprecated/docs/background/`; do not
  revive them into active docs unless they are rewritten against current
  reports and linked from this index.

## Source Of Truth Rules

1. Use [Project State](PROJECT_STATE.md) for current live status.
2. Use `reports/learning_scoreboard.json` to separate operational health,
   private training gain, public transfer, promotion eligibility, and
   stale/superseded lanes.
3. Use report JSON files under `reports/` for machine-readable truth.
4. Use `reports/viea_report_map.json` to map reports and dashboard surfaces to
   VIEA subsystems. VIEA scaffold health is not student-learning proof.
5. Use `reports/viea_artifact_kernel.json` and
   `reports/viea_command_executor.json` to verify that VIEA command contracts
   are indexed in the SQLite artifact store and can execute into route,
   digital-runtime, gate, residual, and feedback packets.
6. Use `configs/control_plane_ownership.json` to resolve overlapping
   supervisor, resource, and promotion-gate scripts. Those scripts are report
   producers; `scripts/theseus_control_plane.py` owns unattended decisions,
   leases, and next-work selection.
7. Use `reports/theseus_doc_link_audit.json` before transfer or major doc
   cleanup. It validates local Markdown links across `README.md`, `docs/`, and
   `deprecated/`.
8. Use `reports/viea_autonomy_spine.json` and
   `reports/feedback_action_queue.json` to see the executable VIEA control
   loop and its next training/tool/residual actions.
9. Use `reports/viea_action_executor.json` and
   `reports/viea_action_execution_ledger.jsonl` to verify which VIEA feedback
   actions actually ran, which were blocked, and whether resume/pause state is
   clean.
10. Use `reports/viea_repo_repair_learner.json` and
   `reports/repo_repair_trace_checkpoint.json` to verify private repo-repair
   traces and learner bridge rows.
11. Use `reports/symliquid_state_engine.json` to see which compact recurrent
   state slots are influencing action routing, STS conditioning, repo repair,
   tool selection, and autonomy control.
12. Use `reports/teacher_architect_experiment_runner.json` to verify the
    teacher-as-architect loop: diagnosis/proposal only, private eval, public
    calibration, promote/rollback decision.
13. Use `reports/feedback_ratchet.json` after VIEA or learning runs to answer
   what improved, regressed, became a tool, became a residual, should expire,
   and should train next.
14. Use `reports/hive_training_state.json`,
   `reports/hive_training_orchestrator.json`, and
   `reports/hive_artifact_merge_summary.json` to answer which Hive node owns
   each active training arm, which arms are blocked by capacity, and which
   per-arm artifact is active. Use `theseus train overnight` for the concise
   audit of what ran, what improved, what failed, stale lease recovery, and
   promoted artifacts.
15. When the Mac is away from the Hive LAN, use `theseus solo status`,
   `theseus solo sweep --execute`, `theseus solo loop --execute`, and
   `theseus solo overnight`. These write `reports/hive_solo_learning_status.json`,
   `reports/hive_solo_learning_ledger.jsonl`, `reports/hive_solo_best_by_arm.json`,
   and `reports/hive_solo_overnight_report.json` without requiring Windows,
   another Mac, internet, teacher calls, or artifact sync.
16. Treat older "next step" prose as historical unless it agrees with
   [Project State](PROJECT_STATE.md).
17. Do not claim candidate promotion while
   `reports/candidate_promotion_gate.json` says `promote=false`; if it says
   `promote=true`, still preserve score semantics from
   `reports/real_code_benchmark_graduation.json`.
17. Do not start long training while `reports/training_preflight_report.json`
   reports blockers or while the resource governor throttles the requested
   profile.
18. Use `reports/arm_lifecycle_governance.json` and
   `reports/autonomy_launch_readiness.json` before teacher-enabled autonomous
   runs.
19. Use `configs/arm_lifecycle_policy.json` for protected arms and lifecycle
   review thresholds.
20. Use `reports/context_packet_ledger.json` as the context ingest view and
   `reports/virtual_context_memory_probe.json`,
   `reports/virtual_context_memory_graph.json`,
   `reports/virtual_context_memory_bench.json`, and
   `reports/virtual_context_compiled_context.json` as the governed VCM working
   context and memory-health view during long autonomous runs. Use
   `reports/vcm_public_memory_prompt_calibration.json`,
   `reports/vcm_public_memory_readiness_audit.json`, and
   `reports/vcm_release_conformance_audit.json` before making public
   memory/context transfer claims.
21. Use `reports/capability_matrix.json` for the current capability/market
   comparison, and refresh `configs/capability_market_baselines.json` before
   public market claims.
22. Use `reports/local_rom_registry.json` for local user-supplied ROM inventory;
   ROM files themselves must remain ignored and are never autonomously
   downloaded.
23. Use `reports/benchmaxx_curriculum.json` to decide which capability lane is
    active, which frontier family is next, and whether the teacher should audit
    a wall instead of inventing the curriculum.
24. Use `reports/self_evolution_governance.json` to decide whether teacher
    apply mode is allowed. Teacher source edits must go through
    `scripts/teacher_self_edit_runner.py` on a guarded branch.
25. Use `reports/benchmark_adapter_factory.json` and `benchmarks/cards/*.json`
    before treating a newly discovered source or local ROM profile as a real
    benchmark frontier.
26. Use `reports/attd_report.json` and
     `reports/attd_maintenance_packets.json` before long autonomy, teacher
     self-edit, architecture change, or adapter proliferation. `RED` means
     maintenance first; `YELLOW` means growth may continue with packets queued.
     ATTD packets can trigger guarded teacher repair with
     `reason=attd_maintenance`; repair traces are saved in
     `reports/teacher_self_edit_traces.jsonl` for future loop-closure
     distillation.
27. Use `reports/autoresearch_gap_audit.json` and
    `reports/autoresearch_experiment_ledger.jsonl` to keep autonomous
    experiments comparable in the Karpathy Autoresearch sense: fixed budget,
    fixed metric surface, compact outcome row, and explicit keep/discard/crash
    status.
28. Use `reports/checkpoint_backup_last.json` to verify accepted-candidate
    backups. Backup is automatic only when the candidate gate promotes; ordinary
    checkpoints remain local.
29. Use `reports/update_status.json` and `reports/update_offer_current.json`
    for accepted-candidate updates. Soft updates can install automatically when
    licensed; hard app/source updates require an explicit guarded install and
    restart. Local/company arms, local configs, data, ROMs, reports, and
    checkpoints are protected from update replacement.
30. Use `reports/hive_verified_version.json`, `reports/hive_update_catalog.json`,
    `reports/hive_version_status.json`, and
    `reports/hive_version_convergence.json` for Hive fleet version alignment.
    The coordinator publishes the most recent verified Hive catalog; workers
    installed with auto-update-soft can converge on safe metadata/checkpoint
    updates automatically. Source/app replacement stays package-based or an
    explicit hard-update flow, not arbitrary remote shell.
    For macOS DMG/pkg rollout, `reports/hive_macos_release_gate.json` and
    `reports/hive_macos_release_gate.md` are the release-candidate gate
    artifacts. Run `theseus hive macos-release-gate --execute` before using a
    DMG broadly; one Apple-Silicon canary and one Intel canary must pass before
    spare Macs are treated as fleet-ready.
31. Use `scripts/theseus_cli.py` or the installed `theseus` wrapper for
    terminal/server control of Hive install, status, setup, device invites,
    profile switching, checkpoint chat, scheduler refresh, and bounded task
    submission. The CLI delegates to the same governed modules as the
    dashboard and does not create a separate remote-task trust surface.
32. Use `reports/hive_status.json`, `reports/hive_peers.json`,
    `reports/hive_peer_registry.json`, `reports/hive_scheduler.json`,
    `reports/hive_remote_access_status.json`,
    `reports/hive_network_doctor.json`,
    `reports/hive_mobile_roaming_profile.json`, and
    `reports/hive_storage_status.json`,
    `reports/hive_voice_following_status.json`, and
    `reports/hive_voice_following_route.json` for distributed Hive state. Remote
    Hive work must use registered task kinds from `configs/hive_policy.json`;
    off-loopback task submission requires `THESEUS_HIVE_SECRET`.
    `hive_peers.json` keeps `peers` to online trusted nodes; stale or
    untrusted discovery history lives under `known_peers`, `stale_peers`, and
    `untrusted_peers` so schedulers do not route work to dead machines.
    Cross-network joining should use same-LAN/hotspot, self-hosted
    WireGuard/private tunnel, or the authenticated relay; paid mesh services
    are not required. Run `theseus hive network-doctor` or
    `theseus remote doctor` before assuming distributed work is live. The
    mobile roaming profile is a private token-bearing
    iPhone import profile and should be handled like an invite. Hive storage
    uses explicit read-only shares in ignored `configs/hive_storage.local.json`;
    do not expose whole disks or private credential directories. Hive voice
    following uses ignored `configs/hive_voice_following.local.json` for
    per-room mic/speaker opt-in.
33. Use `reports/hive_utilization_manager.json` for the always-busy queue
    filler. It may enqueue only registered bounded Hive task kinds, respects
    stop/pause/resource gates, and is allowed to keep safe idle private slots
    fed with training, inference, or maintenance work. It is not a remote-shell
    or teacher-call mechanism.
34. Use `reports/hive_rented_compute_status.json`,
    `reports/hive_rented_compute_plan.json`,
    `reports/hive_rented_compute_ledger.jsonl`,
    `configs/hive_policy.json`, and ignored
    `configs/hive_rented_compute.local.json` for cloud expansion. Rented
    compute/storage is dry-run-first, budget/time-window/queue-pressure gated,
    and provider-specific. Do not treat a rent plan as launched until the
    reviewed plan has been executed and recorded.
35. Use `reports/project_home_migration_plan.json` and
    `scripts/migrate_project_home_to_d.ps1` when this Windows workstation needs
    to move from the low-space C: checkout to `D:\ProjectTheseus\repo`. The
    old C: path should become a compatibility junction, not a second active
    source of truth.
36. Use `reports/hive_remote_control_status.json` and
    `reports/hive_remote_control_request.json` for screen/keyboard/mouse
    takeover handoffs. The Hive brokers audited connection metadata for RDP,
    VNC, RustDesk, Sunshine/Moonlight, or platform-native clients; it does not
    turn arbitrary remote desktop into a benchmark/training authority.
37. Use `reports/license_status.json` and `configs/license_policy.json` before
    creating hives, joining company/friends-family hives, scheduling distributed
    worker chunks, or enabling public contribution. Free community registration
    is non-commercial and capped; company/public-gateway use requires a signed
    paid license.
38. Use `reports/openai_compat_status.json`,
    `configs/openai_compat_policy.json`, and `scripts/openai_compat_server.py`
    for the local OpenAI-compatible endpoint. This shim is allowed because it
    is an inbound local compatibility adapter; it must route to local
    checkpoint/live chat, keep teacher use disabled, and report
    `external_inference_calls=0`.
39. Use `reports/compute_market_status.json`,
    `reports/compute_market_ledger.jsonl`, and
    `configs/compute_market_policy.json` for rented-compute accounting. The
    current unit is an internal work credit, not a tradable public token; public
    exchange, custody, fiat rails, and public-chain token bridging remain
    disabled until a separate reviewed release.
40. Use `reports/legacy_project_concept_audit.json` and
    `configs/legacy_concept_port_map.json` before claiming a predecessor idea is
    ported. Legacy concepts must map to a concrete Theseus surface, acceptance
    gate, and current bottleneck before teacher patches target them.
41. Use `reports/legacy_port_mechanisms.json` before long autonomous runs or
    teacher self-edits. RED mechanism reports are concrete local patch targets,
    not vague architecture inspiration.
42. Use `reports/personality_runtime_audit.json` and
    `reports/personality_context_last.json` as the launch-facing runtime
    personality packet. It is produced by `scripts/personality_context_builder.py`
    from `reports/personality_core.json` and is the blessed way to load the
    user's worldview into chat, Hive, teacher/self-edit, autonomy routing,
    context packets, and launch readiness. Personality documents are not bulk
    training data unless the user explicitly approves a training run.
43. Use `reports/real_code_benchmark_graduation.json` for code-public
    calibration claims. It must show token-level learned generation,
    benchmark-promotion-eligible candidates, zero template-like candidates, zero
    loop-closure benchmark candidates, no regressions, and honest score
    semantics before any code-frontier promotion claim.
44. Use `reports/cell_lifecycle.json` and
    `configs/cell_lifecycle_policy.json` for anti-bloat lifecycle decisions on
    arms, suckers, tools, verifiers, and mastered data. Lifecycle reports are
    non-destructive by default; retirement or data archival requires explicit
    safe policy and human-visible prune plans.
