"""Late-stage legacy port mechanism builders.

These are kept separate from the CLI/orchestration module so the active legacy
port runner stays below AI-maintainability hotspot limits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legacy_port_support import *

def build_synaptic_work_stealing(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    scheduler = state.get("hive_scheduler") if isinstance(state.get("hive_scheduler"), dict) else {}
    chunks = state.get("hive_worker_chunk_tail") if isinstance(state.get("hive_worker_chunk_tail"), list) else []
    task_kinds = get_path(state, ["hive_status", "task_kinds"], [])
    shard_rows = []
    for idx, kind in enumerate(task_kinds[:16] if isinstance(task_kinds, list) else []):
        shard_rows.append(
            {
                "shard_id": f"shard_{safe(kind)}",
                "task_kind": kind,
                "ownership": "claim_by_worker_then_merge",
                "stealable_when_idle": True,
                "merge_lock": "content_hash_plus_role",
                "priority": "high" if "cuda" in str(kind) or "training" in str(kind) else "normal",
            }
        )
    completed = sum(1 for row in chunks if isinstance(row, dict) and str(row.get("status", "")).lower() in {"ok", "completed", "success"})
    steal_rate = round(completed / max(1, len(chunks)), 4)
    return {
        "policy": "bugbrain_synaptic_work_stealing_v0",
        "created_utc": now(),
        "status": "READY" if shard_rows else "PLANNED",
        "scheduler_summary": scheduler.get("summary", {}),
        "shards": shard_rows,
        "worker_tail_count": len(chunks),
        "telemetry": {
            "completed_tail_chunks": completed,
            "steal_rate_proxy": steal_rate,
            "global_lock_required": False,
            "idle_workers_request_bounded_chunks": True,
        },
        "external_inference_calls": 0,
    }


def build_architecture_motif_library(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    search = state.get("architecture_search_space") if isinstance(state.get("architecture_search_space"), dict) else {}
    experiments = search.get("experiments") if isinstance(search.get("experiments"), list) else []
    declared = [
        ("prime_cycle", "periodic sparse recurrence/readout probe"),
        ("recursive_core", "small recurrent controller around SymLiquid state"),
        ("hemispheric_readout", "separate symbolic/control readouts with shared trunk"),
        ("control_placement", "move controller before/after world model state"),
        ("shadow_variant", "shadow runner with identical data and lower blast radius"),
        ("low_rank_lane_adapter", "rank-limited arm transfer without trunk bloat"),
    ]
    motifs = []
    for motif_id, description in declared:
        local = [row for row in experiments if motif_id in str(row).lower()]
        motifs.append(
            {
                "motif_id": motif_id,
                "description": description,
                "single_axis": True,
                "blast_radius": "low",
                "declared_in_search_space": bool(local),
                "retirement_rule": "retire after two matched failures unless residual class recurs",
            }
        )
    return {
        "policy": "corben_architecture_motif_library_v0",
        "created_utc": now(),
        "status": "READY",
        "motifs": motifs,
        "teacher_rule": "teacher may select named motifs only; unnamed architecture families require human review",
        "single_axis_enforced": True,
        "external_inference_calls": 0,
    }


def build_semantic_intent_repair(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    task = state.get("task_goal") if isinstance(state.get("task_goal"), dict) else {}
    teacher = state.get("teacher_self_edit") if isinstance(state.get("teacher_self_edit"), dict) else {}
    failed = failed_gates(state.get("candidate", {}))
    intents = [
        {
            "intent_id": "candidate_gate_repair",
            "verbs": ["diagnose", "repair", "verify"],
            "objects": failed or ["active_frontier"],
            "side_effects": ["reports", "governed_local_artifacts"],
            "source_hash": stable_hash({"failed": failed})[:16],
        },
        {
            "intent_id": "teacher_patch_review",
            "verbs": ["propose", "localize", "prove"],
            "objects": [teacher.get("status") or "teacher_self_edit"],
            "side_effects": ["patch_plan", "proof_bundle"],
            "source_hash": stable_hash(teacher)[:16],
        },
    ]
    observed_effects = [
        "candidate_promotion_gate",
        "self_mod_proof_bundle",
        "legacy_port_mechanisms",
    ]
    return {
        "policy": "moecot_semantic_intent_repair_v0",
        "created_utc": now(),
        "status": "READY",
        "intent_graphs": intents,
        "effect_graph": {
            "allowed_effects": ["write_report", "write_governed_artifact", "propose_patch"],
            "forbidden_effects": ["teacher_apply_without_gate", "bulk_fetch", "live_hardware"],
            "observed_effect_surfaces": observed_effects,
        },
        "verification": "compare intended side-effects with report deltas before promotion",
        "external_inference_calls": 0,
    }


def build_eval_track_contract_library(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    residual_wall = get_path(state, ["self_evolution_governance", "state", "frontier_wall_type"], "")
    tracks = [
        ("rlm_verifier_reasoning", "reasoning/verifier wall", ["local verifier smoke", "regression floor"]),
        ("multi_token_prediction", "sequence modeling throughput wall", ["loss proxy", "decode compatibility"]),
        ("kv_policy", "long context memory wall", ["context replay", "latency delta"]),
        ("differential_attention", "attention selectivity wall", ["ablation", "proxy truth audit"]),
        ("bitnet_quantized_lane", "compute/memory wall", ["cuda/mlx compatibility", "quality floor"]),
        ("context_freeze_thaw", "long-horizon recovery wall", ["temporal replay", "packet salience"]),
    ]
    rows = [
        {
            "track_id": track_id,
            "residual_wall_match": wall,
            "cheap_feasibility_probes": probes,
            "active_match": bool(residual_wall and residual_wall in wall),
            "teacher_can_select": True,
            "retirement_rule": "retire if matched probe fails twice without transfer evidence",
        }
        for track_id, wall, probes in tracks
    ]
    return {
        "policy": "moecot_eval_track_contract_library_v0",
        "created_utc": now(),
        "status": "READY",
        "tracks": rows,
        "unnamed_architecture_families_allowed": False,
        "external_inference_calls": 0,
    }


def build_synaptic_permission_decay(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    arms = maybe_rows(state.get("arm_registry"), ["arms", "active_arms", "registry"])
    tools = maybe_rows(state.get("tool_registry"), ["tools", "registry", "entries"])
    rows = []
    for idx, item in enumerate((arms + tools)[:80]):
        item_id = str(item.get("id") or item.get("name") or item.get("tool_name") or f"capability_{idx}")
        uses = int(item.get("usage_count") or item.get("uses") or item.get("success_count") or 0)
        failures = int(item.get("failure_count") or item.get("failures") or 0)
        success = clamp01(item.get("success_rate", 1.0 if uses and not failures else 0.65))
        age_penalty = 0.04 if uses == 0 else 0.0
        trust = round(clamp01(0.50 + 0.35 * success + 0.02 * min(5, uses) - 0.08 * failures - age_penalty), 4)
        rows.append(
            {
                "capability_id": item_id,
                "kind": "tool" if "tool" in item_id.lower() or item.get("tool_name") else "arm",
                "trust_score": trust,
                "permission_state": "trusted" if trust >= 0.80 else ("probationary" if trust >= 0.55 else "decayed"),
                "recent_proof_required_for_high_risk": trust < 0.85,
            }
        )
    return {
        "policy": "beastbrain_synaptic_permission_decay_v0",
        "created_utc": now(),
        "status": "READY" if rows else "PLANNED",
        "capabilities": rows,
        "summary": count_values(rows, "permission_state"),
        "decay_rule": "disuse and failures reduce envelope; verified use reinforces bounded permissions",
        "external_inference_calls": 0,
    }


def build_temporal_replay_assertions(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    checkpoints = maybe_rows(state.get("checkpoint_registry"), ["checkpoints", "snapshots", "items"])
    pressure = sorted(
        {
            *REPORTS.glob("pressure_source_*seed*.json"),
            *REPORTS.glob("pressure_*seed*.json"),
        },
        key=lambda path: path.stat().st_mtime,
    )[-16:]
    teacher_proof = state.get("teacher_self_edit_proof") if isinstance(state.get("teacher_self_edit_proof"), dict) else {}
    teacher_apply_block = read_json(REPORTS / "teacher_apply_block_smoke.json")
    teacher_cases = []
    if teacher_proof:
        teacher_cases.append(
            {
                "case_id": "teacher_self_edit_proof",
                "surface": "teacher_self_edit",
                "source": "reports/teacher_self_edit_proof.json",
                "status": teacher_proof.get("status"),
                "rollback_policy": "teacher patches remain proposal-only unless local gates pass",
            }
        )
    if isinstance(teacher_apply_block, dict) and teacher_apply_block.get("status") == "blocked_by_teacher_policy":
        teacher_cases.append(
            {
                "case_id": "teacher_apply_block_policy",
                "surface": "teacher_self_edit",
                "source": "reports/teacher_apply_block_smoke.json",
                "status": "blocked_by_teacher_policy",
                "rollback_policy": "apply-mode request is blocked before code mutation",
            }
        )
    replay_cases = {
        "checkpoint_chain_hash_replay": [
            {
                "case_id": str(row.get("checkpoint_id") or row.get("id") or f"checkpoint_{idx}"),
                "surface": "checkpoint",
                "source": "reports/checkpoint_registry.json",
                "created_utc": row.get("created_utc"),
                "status": row.get("status"),
                "sha256": row.get("sha256") or row.get("manifest_sha256"),
            }
            for idx, row in enumerate(checkpoints[:20])
            if isinstance(row, dict)
        ],
        "rl_seed_timeline_replay": [
            {
                "case_id": path.stem,
                "surface": "rl_pressure",
                "source": rel(path),
                "sha256": sha256_file(path),
            }
            for path in pressure
        ],
        "self_edit_rollback_replay": teacher_cases[:4],
    }
    assertions = [
        {
            "assertion_id": "checkpoint_chain_hash_replay",
            "surface": "checkpoint",
            "cases": len(replay_cases["checkpoint_chain_hash_replay"]),
            "required": True,
        },
        {
            "assertion_id": "rl_seed_timeline_replay",
            "surface": "rl_pressure",
            "cases": len(replay_cases["rl_seed_timeline_replay"]),
            "required": True,
        },
        {
            "assertion_id": "self_edit_rollback_replay",
            "surface": "teacher_self_edit",
            "cases": len(replay_cases["self_edit_rollback_replay"]),
            "required": True,
        },
    ]
    for row in assertions:
        row["status"] = "READY" if row["cases"] else "PLANNED"
        row["timeline_id"] = f"timeline_{stable_hash(row)[:16]}"
    all_required_ready = all(row["cases"] > 0 for row in assertions if row.get("required"))
    return {
        "policy": "beastbrain_temporal_replay_assertions_v0",
        "created_utc": now(),
        "status": "READY" if all_required_ready else "PLANNED",
        "assertions": assertions,
        "replay_cases": replay_cases,
        "failed_assertions_enter_residual_escrow": True,
        "external_inference_calls": 0,
    }


def build_whitecell_threat_memory(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    source_report = state.get("online_source_catalog_report") if isinstance(state.get("online_source_catalog_report"), dict) else {}
    teacher_calls = state.get("teacher_calls_tail") if isinstance(state.get("teacher_calls_tail"), list) else []
    patterns = [
        ("external_inference_boundary_violation", get_path(state, ["external_inference_audit", "ok"], True) is False),
        ("uncertain_license_training_source", bool(get_path(source_report, ["summary", "unknown_or_blocked"], 0))),
        ("teacher_apply_mode_request", any("apply" in json.dumps(row).lower() for row in teacher_calls)),
        ("live_hardware_without_sim", False),
        ("bulk_download_request", False),
    ]
    rows = [
        {
            "pattern_id": pattern,
            "active": bool(active),
            "action": "block_and_escalate" if active else "remember",
            "scope": "teacher_prompt_network_source_patch",
        }
        for pattern, active in patterns
    ]
    return {
        "policy": "beastbrain_whitecell_local_threat_memory_v0",
        "created_utc": now(),
        "trigger_state": "YELLOW" if any(row["active"] for row in rows) else "GREEN",
        "threat_patterns": rows,
        "local_only": True,
        "external_inference_calls": 0,
    }


def build_zero_copy_context_prefetch(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    targets = [
        "reports/candidate_promotion_gate.json",
        "reports/frontier_policy_status.json",
        "reports/residual_escrow.json",
        "reports/benchmark_ledger.json",
        "reports/legacy_port_mechanisms.json",
        "reports/capability_matrix.json",
    ]
    refs = []
    for target in targets:
        path = resolve(target)
        refs.append(
            {
                "ref": target,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "digest": sha256_file(path) if path.exists() else "",
                "prefetch_priority": "high" if "frontier" in target or "candidate" in target else "normal",
                "body_inline_allowed": path.exists() and path.stat().st_size < 8192,
            }
        )
    return {
        "policy": "bugbrain_zero_copy_context_prefetch_v0",
        "created_utc": now(),
        "status": "READY",
        "refs": refs,
        "stable_packet_reference": True,
        "dashboard_should_expose_refs": True,
        "external_inference_calls": 0,
    }


def build_hil_emulator_gate(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    world = state.get("world_adapter_jobs") if isinstance(state.get("world_adapter_jobs"), dict) else {}
    adapters = world.get("adapter_coverage_matrix") if isinstance(world.get("adapter_coverage_matrix"), list) else []
    rows = []
    for adapter_row in adapters:
        world_type = str(adapter_row.get("world_type"))
        live = world_type in {"robot_device"} or "live" in str(adapter_row.get("reason", "")).lower()
        sim_evidence = world_type in {"drone_rl", "emulator_rl", "web_agent_local", "coding_local_sandbox"}
        rows.append(
            {
                "world_type": world_type,
                "mode": "live" if live else "sim_or_emulator",
                "sim_smoke_evidence": sim_evidence,
                "live_mode_allowed": False if live else None,
                "promotion_gate": "blocked_until_sim_parity" if live and not sim_evidence else "ready_for_non_live_pressure",
            }
        )
    return {
        "policy": "bugbrain_hil_emulator_gate_v0",
        "created_utc": now(),
        "trigger_state": "GREEN",
        "rows": rows,
        "live_hardware_requires_explicit_user_approval": True,
        "external_inference_calls": 0,
    }


def build_formal_runtime_coupling(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    invariants = [
        "manifest_determinism",
        "provenance_completeness",
        "dag_acyclicity",
        "governance_immutability",
        "effect_isolation",
        "append_only_lineage",
        "tribunal_consensus",
        "self_mod_atomicity",
        "rollback_completeness",
        "human_corrigibility",
    ]
    refs = [
        "scripts/self_evolution_governor.py",
        "scripts/teacher_self_edit_runner.py",
        "scripts/promotion_closure.py",
        "reports/self_mod_proof_bundle.json",
    ]
    rows = [{"invariant": inv, "mapped": True, "runtime_refs": refs[:3], "proof_obligation": "hash_and_replay"} for inv in invariants]
    coupling_hash = stable_hash({"rows": rows, "refs": {ref: sha256_file(resolve(ref)) for ref in refs if resolve(ref).exists()}})
    return {
        "policy": "cca_formal_runtime_coupling_v0",
        "created_utc": now(),
        "status": "READY",
        "coupling_hash": coupling_hash,
        "invariants": rows,
        "rollback_completeness_required": True,
        "external_inference_calls": 0,
    }


def build_veritas_discovery_novelty(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    pantry = state.get("resource_pantry") if isinstance(state.get("resource_pantry"), dict) else {}
    entries = maybe_rows(pantry, ["items", "entries", "resources", "candidates"])
    factory_cards = maybe_rows(state.get("benchmark_adapter_factory"), ["cards"])
    seen = set()
    rows = []
    for item in (entries + factory_cards)[:80]:
        source_id = str(item.get("id") or item.get("card_id") or item.get("source_id") or item.get("name") or "unknown")
        fingerprint = stable_hash({"id": source_id, "url": item.get("url"), "command": item.get("command")})[:16]
        duplicate = fingerprint in seen
        seen.add(fingerprint)
        novelty = 0.35
        if "drone" in source_id or "voice" in source_id or "web" in source_id:
            novelty += 0.25
        if not duplicate:
            novelty += 0.25
        rows.append(
            {
                "source_id": source_id,
                "fingerprint": fingerprint,
                "duplicate": duplicate,
                "novelty_score": round(clamp01(novelty), 4),
                "claim_validation": "metadata_only_until_smoke",
                "falsification_probe_required": True,
            }
        )
    return {
        "policy": "cca_veritas_discovery_novelty_v0",
        "created_utc": now(),
        "status": "READY" if rows else "PLANNED",
        "rows": rows[:60],
        "summary": {
            "rows": len(rows),
            "duplicates": sum(1 for row in rows if row["duplicate"]),
            "mean_novelty": round(sum(row["novelty_score"] for row in rows) / max(1, len(rows)), 4),
        },
        "external_inference_calls": 0,
    }


def build_anti_expert_tribunal_router(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    residuals = maybe_rows(state.get("residual_escrow"), ["clusters", "residuals", "items"])
    traces = state.get("routing_trace_tail") if isinstance(state.get("routing_trace_tail"), list) else []
    rows = []
    for idx, residual in enumerate(residuals[:24]):
        cluster = str(residual.get("cluster") or residual.get("id") or residual.get("type") or f"cluster_{idx}")
        severity = str(residual.get("severity") or residual.get("status") or "medium")
        rows.append(
            {
                "residual_cluster": cluster,
                "anti_expert_rule": "avoid_arm_without_recent_success",
                "veto_strength": "hard" if severity == "critical" else "soft",
                "tribunal_required": severity in {"high", "critical"},
                "known_bad_route_source": "residual_escrow",
            }
        )
    return {
        "policy": "cca_anti_expert_tribunal_router_v0",
        "created_utc": now(),
        "status": "READY" if rows else "PLANNED",
        "negative_route_rules": rows,
        "routing_trace_tail": len(traces),
        "router_eval_additions": ["anti_route_accuracy", "abstention_quality_on_known_residuals", "dissent_preservation"],
        "external_inference_calls": 0,
    }


def build_probe_router_burst_budget(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("probe_router_burst_budget") or {}
    budget = int(cfg.get("deep_route_burst_budget", 3))
    bands = [
        {"band": "reflex", "max_cost": "tiny", "uses_tools": False},
        {"band": "grounded", "max_cost": "low", "uses_tools": True},
        {"band": "deep", "max_cost": "bounded", "uses_tools": True, "burst_budget": budget},
        {"band": "hard_stop", "max_cost": "none", "uses_tools": False, "requires_human": True},
    ]
    active_failed = failed_gates(state.get("candidate", {}))
    selected = "deep" if active_failed else "grounded"
    if any("safety" in gate for gate in active_failed):
        selected = "hard_stop"
    return {
        "policy": "corben_probe_router_burst_budget_v0",
        "created_utc": now(),
        "status": "READY",
        "selected_current_band": selected,
        "bands": bands,
        "spoken_turns_use_same_router": True,
        "openai_compatible_endpoint_uses_same_router": True,
        "external_inference_calls": 0,
    }


def build_rlds_minari_trace_export(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    pressure_reports = sorted(REPORTS.glob("pressure_source_*seed*.json"))[-40:]
    manifests = []
    for path in pressure_reports:
        payload = read_json(path)
        family = str(payload.get("frontier_family") or payload.get("runner_family") or "")
        if "rl" not in family and "drone" not in path.name and "game" not in path.name:
            continue
        manifests.append(
            {
                "episode_source": rel(path),
                "export_id": f"rlds_{stable_hash({'path': rel(path), 'sha': sha256_file(path)})[:16]}",
                "formats": ["theseus_episode_jsonl", "rlds_manifest", "minari_manifest"],
                "fields": ["observation_ref", "action", "reward", "done", "truncated", "info", "seed"],
                "license_metadata_required": True,
                "replay_smoke_required": True,
                "ready": payload.get("external_inference_calls", 0) in {0, None},
            }
        )
    return {
        "policy": "trainer_rlds_minari_trace_export_v0",
        "created_utc": now(),
        "status": "READY" if manifests else "PLANNED",
        "manifests": manifests,
        "summary": {"exports": len(manifests), "ready": sum(1 for row in manifests if row["ready"])},
        "external_inference_calls": 0,
    }


def build_live_operator_advisors(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    candidate = state.get("candidate") if isinstance(state.get("candidate"), dict) else {}
    failed = failed_gates(candidate)
    advisors = []
    if "active_frontier_clears_floor" in failed:
        advisors.append(
            {
                "advisor_id": "frontier_floor_blocker",
                "severity": "high",
                "local_fix_first": "run/inspect active frontier pressure runner and residual bridge before architecture growth",
                "teacher_needed": get_path(state, ["teacher_self_edit", "status"], "") not in {"completed", "ready"},
            }
        )
    if get_path(state, ["attd", "trigger_state"], "GREEN") == "YELLOW":
        advisors.append(
            {
                "advisor_id": "attd_maintenance",
                "severity": "medium",
                "local_fix_first": "apply bounded maintenance packet before broad self-edit expansion",
                "teacher_needed": False,
            }
        )
    if get_path(state, ["performance_optimizer", "trigger_state"], "GREEN") != "GREEN":
        advisors.append(
            {
                "advisor_id": "runtime_performance",
                "severity": "high",
                "local_fix_first": "resolve CUDA/MLX/CPU routing before long run",
                "teacher_needed": False,
            }
        )
    if not advisors:
        advisors.append(
            {
                "advisor_id": "continue_current_pressure",
                "severity": "low",
                "local_fix_first": "continue active frontier, preserve residuals, rotate on graduation or wall threshold",
                "teacher_needed": False,
            }
        )
    return {
        "policy": "trainer_live_operator_advisors_v0",
        "created_utc": now(),
        "status": "READY",
        "advisors": advisors,
        "top_recommendation": advisors[0],
        "external_inference_calls": 0,
    }


def build_benchmark_bounty_registry(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cards = maybe_rows(state.get("benchmark_adapter_factory"), ["cards"])
    seen = set()
    rows = []
    for card in cards[:120]:
        card_id = str(card.get("id") or card.get("card_id") or card.get("source_id") or "unknown")
        command = json.dumps(card.get("commands") or card.get("smoke_command") or card.get("runner_family") or "", sort_keys=True)
        fingerprint = stable_hash({"id": card_id, "command": command})[:16]
        duplicate = fingerprint in seen
        seen.add(fingerprint)
        required_fields = ["id", "category", "runner_family", "status"]
        present = sum(1 for field in required_fields if card.get(field) is not None)
        quality = round(clamp01(0.20 + 0.15 * present + (0.20 if not duplicate else 0.0) + (0.15 if "blocked" not in str(card.get("status", "")) else 0.0)), 4)
        rows.append(
            {
                "submission_id": card_id,
                "fingerprint": fingerprint,
                "duplicate": duplicate,
                "quality_score": quality,
                "accepted": quality >= 0.70 and not duplicate,
                "rejection_reasons": ([] if quality >= 0.70 and not duplicate else ["low_quality_or_duplicate"]),
            }
        )
    return {
        "policy": "moecot_benchmark_bounty_registry_v0",
        "created_utc": now(),
        "status": "READY" if rows else "PLANNED",
        "submissions": rows[:80],
        "summary": {
            "submissions": len(rows),
            "accepted": sum(1 for row in rows if row["accepted"]),
            "duplicates": sum(1 for row in rows if row["duplicate"]),
        },
        "external_inference_calls": 0,
    }


def build_legacy_fine_tooth_comb(policy: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(policy.get("legacy_source_root") or "D:/old_projects"))
    projects = []
    for project in ["BeastBrain", "BugBrain", "cca", "corbens best model possible", "corbens-trainer", "moecot-manifest"]:
        projects.append(project_inventory(root / project))
    candidates = maybe_rows(state.get("legacy_audit"), ["port_candidates"])
    open_items = [row for row in candidates if row.get("status") not in {"done", "retired"}]
    registry_counts = {
        "trainer_benchmarks": count_glob(root / "corbens-trainer" / "registry" / "benchmarks", "*.toml"),
        "trainer_datasets": count_glob(root / "corbens-trainer" / "registry" / "datasets", "*.toml"),
        "trainer_environments": count_glob(root / "corbens-trainer" / "registry" / "environments", "*.toml"),
        "trainer_holdouts": count_glob(root / "corbens-trainer" / "registry" / "holdouts", "*.toml"),
        "trainer_claims": count_glob(root / "corbens-trainer" / "registry" / "claims", "*.toml"),
    }
    return {
        "policy": "project_theseus_legacy_fine_tooth_comb_v0",
        "created_utc": now(),
        "status": "READY",
        "source_root": str(root),
        "projects": projects,
        "registry_counts": registry_counts,
        "candidate_coverage": {
            "declared": len(candidates),
            "open": len(open_items),
            "open_ids": [row.get("id") for row in open_items[:30]],
        },
        "notes": [
            "Zip archives and generated build/cache folders are treated as historical backups, not active source.",
            "The corbens-trainer registry is preserved as source metadata and should feed pantry/adapters through license gates.",
        ],
        "external_inference_calls": 0,
    }
