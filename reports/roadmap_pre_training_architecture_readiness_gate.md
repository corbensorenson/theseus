# Roadmap Implementation Gate

- Trigger state: `RED`
- Matrix: `configs/roadmap_implementation_matrix.json`
- Phases: `20/20`
- Implemented/wired: `12`
- Qualified: `0`
- Partial: `7`
- Missing: `0`
- Frozen: `1`
- Book crosswalk items: `20`
- AI_book authored source files: `4155`
- Stale book-linked phases: `0`
- Unresolved book-to-roadmap backlog items: `343`
- Public-safe Theseus-to-book evidence pointers: `87`
- Source-sync smoke passed: `True`
- Public-safe evidence smoke passed: `True`
- Book implementation tracks: `6`
- Book chapter implementation crosswalk: `54/54`
- Book manifest order match: `True`
- Book manifest digest match: `True`
- Book source-field drift: `0` fields across `0` chapters
- Book Codex tests: `511` total; `109` pending/partial
- Book future candidates/sections: `14`
- Book future candidate missing fields: `0`
- Book future candidate invalid refs: `0`
- Book future candidate dispositions: `{'route_into_existing_owners': 1, 'provisional_source_gated': 8, 'source_contingent': 1, 'prior_art_and_foundations_first': 1, 'source_available_manifest_pending_theseus_disposition_complete': 1, 'source_available_manifest_pending': 2}`
- Planned Codex test backlog: `30`
- Planned Codex backlog missing fields: `0`
- Planned Codex backlog invalid refs: `0`
- Planned Codex backlog blocked/queued: `1`
- Planned Codex backlog status counts: `{'pretraining_wired_behavior_qualification_pending': 11, 'retired_by_pretraining_verdict': 3, 'implemented_negative_result_rung_falsified': 1, 'partial': 1, 'implemented': 3, 'wired': 7, 'protected_discovery_lane': 1, 'planned': 2, 'queued_after_nonfallback_candidate_quality': 1}`
- Planned Codex backlog technique families: `{'qualification_first_pre_deliberative_dispatch_and_reflex_compilation': 1, 'dual_vocabulary_cognitive_compiler_and_hierarchical_residual_state': 1, 'functional_model_qualification': 1, 'data_supported_scale_decision': 1, 'cognitive_kernel_discovery_and_verified_abstraction_scaling': 1, 'offline_preference_optimization': 1, 'verifier_reward_policy_optimization': 1, 'policy_update_lease': 1, 'generate_verify_repair': 1, 'generation_mode': 2, 'sketch_first_generation': 1, 'context_memory_abi': 1, 'verification_resource_routing': 1, 'claim_ledger_revision': 1, 'procedural_memory': 1, 'substrate_adoption': 1, 'data_lifecycle_governance': 1, 'durable_semantic_memory': 1, 'governed_effect_transaction': 1, 'routing_and_deliberation': 1, 'scalable_oversight': 1, 'evaluation_integrity': 1, 'capability_commitment_and_assurance': 1, 'supply_chain_and_weight_custody': 1, 'inter_stack_protocol': 1, 'open_ended_improvement_campaign': 1, 'question_compiled_semantic_addressing': 1, 'independent_token_corpus_and_kerc_acceleration': 1, 'finite_pretraining_architecture_and_efficiency_closure': 1}`
- Active flagship lane: `C1_correctness_rl_and_generator_survival_lane`
- Active core slices: `1`
- Active core slice support states: `{'C1_correctness_rl_and_generator_survival_lane': 'synthetic-test-backed'}`
- Core slice support states: `{'A1_claim_ledger_trace_kernel': 'synthetic-test-backed', 'A2_replacement_transaction_kernel': 'replayable-reference-backed', 'E1_authority_scif_runtime_adapter_kernel': 'replayable-reference-backed', 'B1_assisted_verified_assistant_product_lane': 'synthetic-test-backed', 'C1_correctness_rl_and_generator_survival_lane': 'synthetic-test-backed'}`
- Support-state ladder ready: `True`
- Pre-training architecture ready: `False`
- Pre-training architecture blockers: `2`
- Book crosswalk missing fields: `0`
- Book crosswalk invalid phase refs: `0`
- Hard gaps: `1`
- Warnings: `1`
- Phases 13-19 preserved: `True`

## Phase Matrix

| Phase | Status | Priority | Surface | Missing | Hard Gaps |
| --- | --- | --- | --- | ---: | ---: |
| 0 Repository Self-Model And Registry Discipline | partial | critical | `project_manifest_registry` | 2 | 0 |
| 1 VIEA Execution Spine | wired | critical | `theseus_plan_compiler` | 0 | 0 |
| 2 Stable Capability Fields And Route Authority | wired | critical | `project_manifest_registry` | 0 | 0 |
| 3 Virtual Context Memory As Default Context Substrate | wired | critical | `vcm_memory` | 0 | 0 |
| 4 Candidate Integrity And Learned Generation Accounting | wired | critical | `candidate_integrity_harness` | 0 | 0 |
| 5 Daily-Use Assistant Runtime And Dogfood Trace Loop | partial | critical | `theseus_assistant_runtime` | 1 | 0 |
| 6 Deterministic Tool And Search Substrate | wired | high | `deterministic_tool_substrate` | 0 | 0 |
| 7 Teacher And Data Governance | partial | critical | `teacher_and_data_governance` | 2 | 0 |
| 8 Resource, Cost, And Mac Acceleration Routing | partial | critical | `resource_and_acceleration` | 3 | 0 |
| 9 Hive Policy-First Distributed Operation | frozen | medium | `hive_install_and_apps` | 1 | 0 |
| 10 Practical Neural Seed Survival Lane | partial | critical | `neural_seed_and_decoder` | 7 | 0 |
| 11 Cognitive Kernel Discovery Lane Verdicts | wired | high | `neural_seed_and_decoder` | 0 | 0 |
| 12 Public Calibration And Residual Mining Discipline | wired | critical | `public_calibration_registry` | 0 | 0 |
| 13 Semantic IR And Substrate-Neutral Reasoning Atoms | partial | critical | `theseus_plan_compiler` | 3 | 0 |
| 14 Compression, Proof, And Claim Evidence Records | wired | high | `evidence_store` | 0 | 0 |
| 15 Procedural Memory And Toolification | implemented | high | `theseus_plan_compiler` | 0 | 0 |
| 16 MoECOT And Octopus Router Integration | partial | high | `theseus_plan_compiler` | 2 | 0 |
| 17 Simulation, Fidelity, And World-Model Contracts | wired | medium | `theseus_plan_compiler` | 0 | 0 |
| 18 Governance Rights, Constitutional Predicates, And Failure Boundaries | wired | critical | `core_control_plane` | 0 | 0 |
| 19 Book-To-Theseus Backlog And Evidence Synchronization | implemented | high | `active_docs` | 0 | 0 |

## Hard Gaps

- `pre_training_architecture_readiness`: architecture_not_ready_for_training {"blocker_count": 2, "ready": false, "rule": "complete book-derived architecture slices before training, public calibration, or score-chasing"}

## Pre-Training Architecture Readiness

- Ready: `False`
- Blockers: `2`
- architecture_freeze_package_not_ready: {"evidence": {"acceptance_observed": {}, "declared": true, "disposition": "architecture_frozen_training_not_started", "faults": ["source_artifacts_stale:ROADMAP.md,configs/moecot_language_arm_training.json,configs/roadmap_implementation_matrix.json,scripts/fresh_process_pretraining_qualification.py,scripts/moecot_language_arm_training.py,scripts/standard_causal_transformer_survival.py,tests/test_moecot_language_arm_training.py,tests/test_standard_causal_transformer_survival.py"], "path": "reports/pretraining_architecture_freeze_package.json", "policy": "project_theseus_pretraining_architecture_freeze_v1", "ready": false, "source_artifact_count": 123, "trigger_state": "GREEN"}, "kind": "architecture_freeze_package_not_ready", "required_action": "Build and independently replay the exact content-addressed architecture package before authorizing long training."}
- unfinished_architecture_prerequisite_phases: {"count": 2, "kind": "unfinished_architecture_prerequisite_phases", "phases": [{"missing_item_count": 2, "missing_source_artifact_paths": [], "phase": 0, "smallest_next_patch": "Bank the independently replayed T0A acceleration transaction as a coherent reviewed checkpoint with bound source/config/report/replay identities, leaving unrelated user changes outside it; committing is custody, not qualification. Then complete Phase 0 reproduction and audit and retire the superseded strict-generator/code_lm_closure family, extract live mechanisms into canonical owners, remove retired imports from active gates, archive reports through verified retention pointers, and compact roadmap evidence prose. Real signed release/custody evidence remains a separate post-checkpoint task.", "status": "partial", "title": "Repository Self-Model And Registry Discipline"}, {"missing_item_count": 3, "missing_source_artifact_paths": ["reports/t0a_canary_storage_retention.json", "reports/training_acceleration_audit_2026_07_26.json", "scripts/t0a_canary_storage_retention.py", "tests/test_t0a_canary_storage_retention.py"], "phase": 8, "smallest_next_patch": "Keep the 31.791 GiB reclaimed disk reserve, extend the budget/reference graph to the remaining runtime canary checkpoints, and do not perform a blind second purge. When real host headroom is available, complete the 64-step fresh-process state/cursor/publication replay without weakening the live reserve and regenerate the replacement freeze. In parallel, implement FP16 compute with FP32 master state as a diagnostic-only challenger with correct recasting/loss scaling and run its matched full-state/reload/replay gates; this outranks persistent-process folklore because measured non-checkpoint process overhead is below 2.5%. Qualify MLX fast synchronization, use bounded station timing for at most one custom kernel, implement user-presence-aware segment scheduling, and fan independent controls/arms to additional local Macs when available.", "status": "partial", "title": "Resource, Cost, And Mac Acceleration Routing"}], "required_action": "Complete or explicitly external-freeze every architecture prerequisite before making training the primary roadmap focus."}

## Warnings

- `book_chapter_implementation_crosswalk`: live_book_worktree_differs_from_pinned_snapshot {"book_commit": "32635eb94ded42a5f54e528302685cab343993b7", "live_manifest_sha256": "4e66f0977ead70e116f08ca65aaf6ed775213bbcfc085e7850df154ae7249ea3", "pinned_manifest_sha256": "8ed8864d7a14673d75d81620f81a4f3740bf826fbb3862ee41e0e20ad56c485e", "rule": "Live book edits are intake work, not an architecture-readiness regression. Reconcile and advance the pin in a separate reviewed change."}
