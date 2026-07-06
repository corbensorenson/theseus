# Old Projects Transfer Audit

Updated: 2026-05-29

This is the canonical documentation page for the six predecessor projects in
`D:\old_projects`. It consolidates the former legacy concept audit, deep
transfer audit, mechanism summary, completion audit, and old-project review.

Use this page for human orientation. Use the generated JSON reports under
`reports/` for machine-readable truth.

## Bottom Line

- The six intended predecessor projects are present.
- The declared concept-port map is complete at the discovery layer: 42/42
  candidates are marked `done`, including all 22 P0 candidates.
- The old projects should not be copied wholesale into Theseus.
- Port coverage is now GREEN: the inherited concepts have current Theseus
  control/report surfaces and no live mechanism is RED.
- The remaining problem is operational maturation, not another broad concept
  import pass.
- Bounded and long autonomy are now allowed by the legacy runtime enforcement
  layer. Candidate promotion and self-evolution remain intentionally blocked
  by frontier evidence, coherence/promotion cleanliness, maturity, and WhiteCell
  constraints.

## Source Projects

| Project | Source role | Theseus inheritance | Current gap |
| --- | --- | --- | --- |
| `BeastBrain` | Multimodal platform, persistent knowledge, SparkStream background cognition, PlanForge scheduling, WhiteCell/Aletheia safety, device-first access. | Salience scheduler, tiered memory, PlanForge critical-path scheduling, Aletheia/Advocate gate, temporal replay, permission decay, WhiteCell threat memory. | Keep PlanForge as the autonomy scheduler. WhiteCell still blocks self-evolution while the active teacher-apply pattern is unresolved. |
| `BugBrain` | Bare-metal neuro-symbolic runtime with spreading activation, grammatical weaver, active inference, thermal/resource health, hot-path CI, work stealing. | Coherence/delirium metric, hot-path quality gates, synaptic work stealing, active-inference world model, HIL/emulator gate, zero-copy context prefetch. | Carry the same fail-closed hot-path discipline into current Code LM fanout/ranker/verifier work. |
| `cca` | Compiler-first, governance-first autonomy with self-mod contracts, proxy truth, apples-to-apples benchmark discipline, drone/sim2real boundaries. | Self-mod proof bundle, drone blackbox parity, proxy truth audit, formal runtime coupling, Veritas novelty, anti-expert tribunal routing. | Proxy-truth is GREEN for the active frontier. Keep promotion blocked until the public/code transfer and candidate evidence gates are clean. |
| `corbens best model possible` | Compact local model family, train/eval contract discipline, architecture A/B lifecycle, lane-safe low-rank adapter banks, TaskSpell-style locks. | TaskSpell contract lock, low-rank lane adapter bank, macro counterexample gate, architecture motif library, first-party speech contract, route burst budgets. | Zero-parameter adapter dry-run readiness is now true. Do not activate shared adapter weights without lane-slice/interference proof. |
| `corbens-trainer` | Contract-first trainer and registry layer for plans, datasets, benchmarks, holdouts, run bundles, evidence graphs, campaign DAGs. | Campaign DAGs, evidence graph/research ledger, runtime resolution boundary, dataset recipe scaffolder, RLDS/Minari trace export, local operator advisors. | Use only the one serious-ready source as training-ready. Keep public benchmark assets metadata/calibration-only until fresh Theseus evidence passes gates. |
| `moecot-manifest` | Compiler-era system root with semantic compiler, model compiler, trace-fabric training exchange, world adapters, compute routing, bridge promotion, benchmark bounty. | Trace-fabric exchange, semantic intent repair, eval-track contract library, world adapter job runtime, compute mode acceptance, emulator/game trace gateway, benchmark bounty, ORCP compression, USB/device endpoint, pretraining readiness integrity. | Native bridge promotion and pretraining readiness are GREEN. Asset-gated ROM cards and host-runtime dependencies are tracked as deferrals, not architecture-port blockers. |

## Live Report State

Refresh commands:

```powershell
python scripts\old_project_registry_port.py --policy configs\old_project_registry_port_policy.json --out reports\old_project_registry_port.json --markdown-out reports\old_project_registry_port.md
python scripts\legacy_project_concept_audit.py --map configs\legacy_concept_port_map.json --out reports\legacy_project_concept_audit.json --markdown-out reports\legacy_project_concept_audit.md
python scripts\legacy_port_mechanisms.py --policy configs\legacy_port_policy.json --out reports\legacy_port_mechanisms.json --markdown-out reports\legacy_port_mechanisms.md
python scripts\legacy_port_runtime_enforcer.py --out reports\legacy_port_runtime_enforcement.json
python scripts\legacy_training_source_audit.py --out reports\legacy_training_source_audit.json --markdown-out reports\legacy_training_source_audit.md --admissions-out reports\legacy_training_source_admissions.json
python scripts\legacy_rl_environment_admission.py --out reports\legacy_rl_environment_admission.json --markdown-out reports\legacy_rl_environment_admission.md
python scripts\legacy_rl_smoke_plan.py --admission reports\legacy_rl_environment_admission.json --out reports\legacy_rl_smoke_plan.json --plan-out reports\legacy_rl_smoke_plan.md
python scripts\legacy_port_completion_audit.py --out reports\legacy_port_completion_audit.json --markdown-out reports\legacy_port_completion_audit.md
```

Current machine-readable state:

| Surface | State | Important signal |
| --- | --- | --- |
| `reports/legacy_project_concept_audit.json` | GREEN | 6/6 projects present, 42 port candidates, 0 P0 open, 0 missing evidence. |
| `reports/legacy_port_mechanisms.json` | no RED | 43 live mechanisms: 41 green/ready/locked, 2 yellow/degraded, 0 red. Top blocker: none. |
| `reports/legacy_port_runtime_enforcement.json` | GREEN | Bounded and long autonomy are ready. Candidate promotion and self-evolution remain blocked by candidate/WhiteCell policy, not by missing legacy-port enforcement. |
| `reports/old_project_registry_port.json` | GREEN | 39 benchmark cards, 42 datasets, 2283 reference answers redacted, 9 ready training sources, 0 external inference calls. |
| `reports/legacy_training_source_audit.json` | YELLOW | 9 local verified sources, 1 serious training-ready source, 0 hash mismatches, 0 unsafe ready sources. |
| `reports/legacy_rl_environment_admission.json` | GREEN | 62 environments normalized, 11 P0 smoke candidates, hardware drone lanes blocked from autopromotion. |
| `reports/legacy_port_completion_audit.json` | YELLOW | Port coverage is GREEN. Operational maturity is still YELLOW because candidate promotion/self-evolution are intentionally not clean. |

The completion audit uses YELLOW because candidate-promotion and self-evolution
readiness are stricter than concept-port coverage. Treat that YELLOW as an
operational maturity blocker, not as a signal to import more old-project code.

## Declared Concept Coverage

| Project | Done concepts |
| --- | --- |
| BeastBrain | `sparkstream_salience_scheduler`, `ssd_tiered_memory_consolidation`, `aletheia_advocate_gate`, `beastbrain_planforge_critical_path_scheduler`, `beastbrain_synaptic_permission_decay`, `beastbrain_temporal_replay_assertions`, `beastbrain_whitecell_local_threat_memory` |
| BugBrain | `bugbrain_hotpath_quality_gates`, `bugbrain_synaptic_work_stealing`, `bugbrain_coherence_delirium_metric`, `bugbrain_active_inference_world_model`, `bugbrain_zero_copy_context_prefetch`, `bugbrain_hil_emulator_gate` |
| CCA | `cca_self_mod_proof_bundle`, `cca_drone_blackbox_parity`, `cca_proxy_truth_audit`, `cca_formal_runtime_coupling`, `cca_veritas_discovery_novelty`, `cca_anti_expert_tribunal_router` |
| Corben-1 | `corben_architecture_motif_library`, `corben_first_party_speech_contract`, `corben_taskspell_contract_lock`, `corben_macro_store_counterexample_gate`, `corben_low_rank_lane_adapter_bank`, `corben_probe_router_burst_budget` |
| corbens-trainer | `trainer_contract_first_campaign_dag`, `trainer_evidence_graph_research_ledger`, `trainer_runtime_resolution_boundary`, `trainer_dataset_recipe_scaffolder`, `trainer_rlds_minari_trace_export`, `trainer_live_operator_advisors` |
| moecot-manifest | `moecot_trace_fabric_training_exchange`, `moecot_semantic_intent_repair`, `moecot_eval_track_contract_library`, `moecot_world_adapter_job_runtime`, `moecot_compute_mode_acceptance`, `moecot_bridge_adapter_native_promotion`, `moecot_emulator_game_trace_gateway`, `moecot_benchmark_bounty_registry`, `moecot_orcp_compression_frontier`, `moecot_usb_serial_device_endpoint`, `moecot_pretraining_readiness_integrity_gate` |

This table means "there is a Theseus control/report surface for the concept."
It does not mean the concept is fully mature. Maturity is governed by the live
reports.

## Live Mechanism Status

The old-project concepts currently materialize through 43 live mechanisms. The
latest refresh reports:

- GREEN/READY/LOCKED/CONTRACT_READY mechanisms: 41
- YELLOW/degraded mechanisms: 2
- RED mechanisms: 0
- Top blocker: none

Important report surfaces include:

- `reports/planforge_schedule.json`
- `reports/coherence_delirium_report.json`
- `reports/proxy_truth_audit.json`
- `reports/taskspell_contracts.json`
- `reports/low_rank_adapter_bank.json`
- `reports/world_adapter_job_runtime.json`
- `reports/emulator_game_trace_gateway.json`
- `reports/compute_mode_acceptance.json`
- `reports/hotpath_quality_gates.json`
- `reports/drone_blackbox_parity.json`
- `reports/self_mod_proof_bundle.json`
- `reports/first_party_speech_contract.json`
- `reports/trace_fabric_training_exchange.json`

Do not interpret implemented mechanisms as promotion readiness. A mechanism is
useful only when its live report is GREEN, READY, or intentionally LOCKED.

## Remaining Work

1. Keep proxy truth hardened at the producer and audit layers.

   CCA's proxy-truth lineage requires runtime path, checkpoint lineage,
   tokenizer, decode config, prompt template, verifier pointer, raw output
   pointer/hash, and contamination fields.

   Current Theseus state: `reports/proxy_truth_audit.json` is GREEN for the
   active BabyLM frontier. Older nested eval reports are recognized when they
   contain local runtime identity and raw per-case outputs, and future BabyLM
   probe reports emit explicit local identity fields.

2. Convert adapter-bank dry-run readiness into measured ablation evidence.

   Corben-1 proved that shared adapters can improve one lane while damaging
   others, so lane-slice and interference gates are mandatory.

   Current Theseus state: `adapter_bank_zero_param_dry_run_ready` is true. The
   planned adapters remain manifest-only; zero-parameter lanes are control
   baselines until ablations and regression floors exist.

3. Keep public benchmark data calibration-only.

   corbens-trainer separates benchmark assets from benchmark rules and uses
   redacted case manifests.

   Current Theseus state: old registry import is GREEN with 2283/2283 reference
   answers redacted and no training data copied. Public score claims require
   fresh local Theseus runs, not old registry metadata.

4. Promote world/game/device lanes by native smoke, not bridge convenience.

   moecot-manifest world adapters require explicit bindings, replay IDs,
   coverage matrices, unsupported-feature reports, and fail-closed behavior.

   Current Theseus state: world jobs are present, RL environment admission is
   GREEN, and bridge adapter native promotion is GREEN. Hardware-gated drone
   lanes and user-supplied ROM lanes remain blocked from autopromotion.

5. Carry BugBrain's performance discipline into active hot paths.

   BugBrain repeatedly split large modules, added hot-path microbench gates,
   removed panic/unwrap paths, reduced unsafe concentration, and ratcheted
   warning/performance budgets.

   Current Theseus state: hotpath quality gates are GREEN and classify duplicate
   work prevention as self-protection, not a failure. The current Code LM
   candidate expansion/ranking/verifier path still needs first-class bottleneck
   governance and faster batched/resident execution.

6. Clear operational maturity blockers without weakening safety.

   Current Theseus state: `reports/legacy_port_completion_audit.json` is YELLOW
   because candidate promotion, self-evolution, coherence cleanliness,
   maturity-integrity, and frontier transfer evidence are still not promotion
   clean. These are not old-project port gaps; they are the next Theseus
   capability gates.

## Recommended Next Actions

1. Keep the old-project registry as metadata and governed tiny samples only.
   Do not bulk copy datasets, answer keys, public solutions, or old artifacts.

2. Run adapter-bank zero-parameter dry-run/ablation evidence before any adapter
   weight activation.

3. Clear candidate-promotion-only gates through fresh frontier evidence,
   transfer proof, coherence cleanliness, and maturity integrity.

4. Keep using `configs/legacy_concept_port_map.json` as the declared concept
   inventory. Add to it only when a new source-backed concept has a concrete
   Theseus surface and acceptance gate.

## Consolidated Legacy Pages

These older filenames have been retired from active docs and moved under
`deprecated/docs/legacy-transfer/` as link-history/audit material:

- `deprecated/docs/legacy-transfer/LEGACY_PROJECTS_CONCEPT_AUDIT.md`
- `deprecated/docs/legacy-transfer/LEGACY_PORT_MECHANISMS.md`
- `deprecated/docs/legacy-transfer/OLD_PROJECTS_DEEP_TRANSFER_AUDIT.md`
- `deprecated/docs/legacy-transfer/OLD_PROJECTS_PORT_COMPLETION_AUDIT.md`
- `deprecated/docs/legacy-transfer/OLD_PROJECTS_PORT_REVIEW.md`

They should not grow new duplicate content. Update this page instead.
