"""Local benchmark treadmill for SymLiquid.

The treadmill scans JSON reports, identifies saturated and unsaturated
capability surfaces, and emits concrete next local commands. It never calls
external inference providers; it only reads local artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Surface:
    path: str
    family: str
    metric: str
    value: float
    residual: float
    status: str
    external_inference_calls: int
    lifecycle: str
    capability: str
    benchmark_type: str
    contamination_risk: str
    label_quality: str
    transfer_evidence: str
    cost_class: str
    wall_type: str
    recommended_intervention: str
    score_ceiling: float | None
    capability_narrative: str
    curriculum_status: str
    comparator_class: str
    initial_threshold: float
    current_threshold: float
    floor_threshold: float
    subgroup_floor: float
    threshold_phase: str
    attempt_count: int
    recent_delta: float | None
    stalled_cycles: int
    decay_cycles: int
    threshold_decay_rate: float
    threshold_decay_mode: str
    critical_failure_count: int
    frontier_momentum_action: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--out", default="reports/benchmark_treadmill_status.json")
    parser.add_argument("--benchmark-ledger-out", default="reports/benchmark_ledger.json")
    parser.add_argument("--model-ledger-out", default="reports/model_ledger.json")
    parser.add_argument("--lifecycle-overrides", default="reports/benchmark_lifecycle_overrides.json")
    parser.add_argument("--saturation-threshold", type=float, default=0.90)
    parser.add_argument("--threshold-floor", type=float, default=0.70)
    parser.add_argument("--threshold-patience", type=int, default=3)
    parser.add_argument("--threshold-decay-rate", type=float, default=0.01)
    parser.add_argument(
        "--threshold-decay-mode",
        choices=("per_attempt_after_patience", "stalled_only"),
        default="per_attempt_after_patience",
    )
    parser.add_argument("--stall-epsilon", type=float, default=0.005)
    parser.add_argument("--stall-window", type=int, default=3)
    parser.add_argument("--residual-escrow-budget", type=float, default=0.10)
    parser.add_argument("--broken-threshold", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument(
        "--public-comparator-ledger-out",
        default="reports/public_comparator_ledger.json",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    surfaces = []
    report_paths = sorted(reports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime)
    if args.limit > 0:
        report_paths = report_paths[-args.limit :]
    for path in report_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if is_meta_report(payload):
            continue
        if not args.include_baselines and is_baseline_report(path, payload):
            continue
        surface = extract_surface(path, payload, args.saturation_threshold, args.broken_threshold)
        if surface is not None:
            surfaces.append(surface)

    best_by_family = {}
    for surface in surfaces:
        current = best_by_family.get(surface.family)
        if current is None or family_surface_replaces_current(surface, current):
            best_by_family[surface.family] = surface
    family_surfaces = sorted(best_by_family.values(), key=lambda item: item.value)
    apply_mastery_threshold_policy(family_surfaces, surfaces, args)
    apply_lifecycle_overrides(family_surfaces, args.lifecycle_overrides)
    open_surfaces = [surface for surface in family_surfaces if surface.status == "open"]
    saturated = [surface for surface in family_surfaces if surface.status == "saturated"]
    broken = [surface for surface in family_surfaces if surface.status == "broken"]
    invalid_external = [
        surface for surface in family_surfaces if surface.external_inference_calls > 0
    ]
    benchmark_ledger = [benchmark_ledger_entry(surface) for surface in family_surfaces]
    model_ledger = build_model_ledger(family_surfaces, open_surfaces, saturated, broken)
    public_comparator_ledger = build_public_comparator_ledger(benchmark_ledger)
    ratchet = build_ratchet_report(open_surfaces, saturated, broken, invalid_external)
    payload = {
        "policy": "local_only_no_external_inference",
        "methodology": "benchmaxxing_performance_ratchet",
        "curriculum_policy": {
            "initial_mastery_threshold": args.saturation_threshold,
            "floor_threshold": args.threshold_floor,
            "patience_cycles": args.threshold_patience,
            "decay_mode": args.threshold_decay_mode,
            "decay_rate_per_cycle": args.threshold_decay_rate,
            "legacy_decay_rate_per_stalled_cycle": args.threshold_decay_rate,
            "stall_epsilon": args.stall_epsilon,
            "stall_window": args.stall_window,
            "residual_escrow_budget": args.residual_escrow_budget,
            "meaning": "A graduated benchmark becomes regression pressure, not a claim of perfect mastery.",
            "reason": "Hold the system to high mastery first, then decay ordinary benchmark thresholds by 1 percentage point per attempt after patience until the 70% floor so the frontier cannot be held hostage by non-critical tails.",
            "critical_failure_veto": "Safety-critical or invalid external-inference failures block graduation regardless of aggregate score.",
        },
        "saturation_threshold": args.saturation_threshold,
        "broken_threshold": args.broken_threshold,
        "counts": {
            "surface_reports": len(surfaces),
            "families": len(family_surfaces),
            "open": len(open_surfaces),
            "saturated": len(saturated),
            "broken": len(broken),
            "external_inference_violations": len(invalid_external),
        },
        "ratchet": ratchet,
        "active_frontier": [asdict(surface) for surface in open_surfaces[:16]],
        "saturated_examples": [asdict(surface) for surface in saturated[:16]],
        "broken_or_smoke_only": [asdict(surface) for surface in broken[:16]],
        "external_inference_violations": [asdict(surface) for surface in invalid_external[:16]],
        "benchmark_ledger_out": args.benchmark_ledger_out,
        "model_ledger_out": args.model_ledger_out,
        "public_comparator_ledger_out": args.public_comparator_ledger_out,
        "next_commands": recommend_next_commands(open_surfaces, saturated, surfaces),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_json(args.benchmark_ledger_out, benchmark_ledger)
    write_json(args.model_ledger_out, model_ledger)
    write_json(args.public_comparator_ledger_out, public_comparator_ledger)
    print(json.dumps(payload, indent=2))
    return 0


def family_surface_replaces_current(candidate: Surface, current: Surface) -> bool:
    if candidate.family == "babylm_mutated_holdout":
        candidate_seed = seed_from_path(candidate.path)
        current_seed = seed_from_path(current.path)
        if candidate_seed is not None and current_seed is not None:
            return candidate_seed > current_seed
    if candidate.family.startswith("ocean-") and (
        "rl_frontier_" in candidate.path or "rl_frontier_" in current.path
    ):
        candidate_is_frontier = "rl_frontier_" in candidate.path
        current_is_frontier = "rl_frontier_" in current.path
        if candidate_is_frontier != current_is_frontier:
            return candidate_is_frontier
        candidate_seed = seed_from_path(candidate.path)
        current_seed = seed_from_path(current.path)
        if candidate_seed is not None and current_seed is not None:
            return candidate_seed > current_seed
        return surface_attempt_key(candidate) > surface_attempt_key(current)
    return candidate.value > current.value


def seed_from_path(path: str) -> int | None:
    match = re.search(r"seed(\d+)", path)
    if match is None:
        return None
    return int(match.group(1))


def apply_mastery_threshold_policy(
    family_surfaces: list[Surface],
    all_surfaces: list[Surface],
    args: argparse.Namespace,
) -> None:
    for surface in family_surfaces:
        history = [
            candidate
            for candidate in all_surfaces
            if candidate.family == surface.family and candidate.external_inference_calls == 0
        ]
        history.sort(key=surface_attempt_key)
        policy = threshold_policy_for_surface(surface, args)
        attempt_count = max(1, len(history))
        recent_delta = recent_improvement(surface, history, args.stall_window)
        score = clipped_score(surface.value)
        critical_failures = critical_failure_count(surface)
        eligible_for_decay = (
            score < policy["initial_threshold"]
            and attempt_count > args.threshold_patience
            and critical_failures == 0
            and not policy["no_decay"]
        )
        stalled = eligible_for_decay and recent_delta < args.stall_epsilon
        raw_decay_cycles = max(0, attempt_count - args.threshold_patience)
        if args.threshold_decay_mode == "stalled_only":
            decay_cycles = raw_decay_cycles if stalled else 0
        else:
            decay_cycles = raw_decay_cycles if eligible_for_decay else 0
        stalled_cycles = raw_decay_cycles if stalled else 0
        current_threshold = max(
            policy["floor_threshold"],
            policy["initial_threshold"] - args.threshold_decay_rate * decay_cycles,
        )
        threshold_phase = threshold_phase_for(
            score=score,
            initial_threshold=policy["initial_threshold"],
            current_threshold=current_threshold,
            floor_threshold=policy["floor_threshold"],
            stalled=stalled,
            decay_cycles=decay_cycles,
            critical_failures=critical_failures,
            attempt_count=attempt_count,
            patience=args.threshold_patience,
        )
        if surface.external_inference_calls > 0:
            status = "invalid_external_inference"
        elif critical_failures > 0:
            status = "open"
        elif score >= current_threshold:
            status = "saturated"
        elif score <= args.broken_threshold:
            status = "broken"
        else:
            status = "open"

        surface.status = status
        surface.lifecycle = lifecycle_for_status(status)
        surface.wall_type, surface.recommended_intervention = diagnose_wall(
            surface.family, status, surface.metric, score
        )
        surface.capability_narrative = capability_narrative_with_threshold(
            surface.family, status, score, current_threshold, threshold_phase
        )
        surface.curriculum_status = curriculum_status_for_surface(status, threshold_phase)
        surface.initial_threshold = policy["initial_threshold"]
        surface.current_threshold = current_threshold
        surface.floor_threshold = policy["floor_threshold"]
        surface.subgroup_floor = policy["subgroup_floor"]
        surface.threshold_phase = threshold_phase
        surface.attempt_count = attempt_count
        surface.recent_delta = recent_delta
        surface.stalled_cycles = stalled_cycles
        surface.decay_cycles = decay_cycles
        surface.threshold_decay_rate = args.threshold_decay_rate
        surface.threshold_decay_mode = args.threshold_decay_mode
        surface.critical_failure_count = critical_failures
        surface.frontier_momentum_action = frontier_momentum_action(surface)


def apply_lifecycle_overrides(family_surfaces: list[Surface], path: str) -> None:
    overrides = read_json(Path(path))
    promotions = overrides.get("promotions") if isinstance(overrides, dict) else []
    if not isinstance(promotions, list):
        return
    by_name = {str(item.get("benchmark_name")): item for item in promotions if isinstance(item, dict)}
    by_report = {
        normalize_report_path(item.get("report")): item
        for item in promotions
        if isinstance(item, dict) and item.get("report")
    }
    for surface in family_surfaces:
        override = by_report.get(normalize_report_path(surface.path)) or by_name.get(surface.family)
        if not override:
            continue
        if override.get("lifecycle") != "regression":
            continue
        surface.status = "saturated"
        surface.lifecycle = "regression"
        surface.threshold_phase = str(override.get("threshold_phase") or "candidate_promotion_override")
        surface.curriculum_status = "candidate_promoted_preserve_as_regression"
        surface.wall_type = "no_current_wall"
        surface.recommended_intervention = (
            "Candidate gate accepted this surface. Preserve it as regression, keep residuals in escrow, "
            "and rotate to a harder frontier."
        )
        surface.capability_narrative = capability_narrative_with_threshold(
            surface.family,
            surface.status,
            clipped_score(surface.value),
            surface.current_threshold,
            surface.threshold_phase,
        )
        surface.frontier_momentum_action = "candidate_promoted_rotate_to_harder_frontier"


def normalize_report_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def threshold_policy_for_surface(surface: Surface, args: argparse.Namespace) -> dict[str, Any]:
    benchmark_type = surface.benchmark_type.lower()
    family = surface.family.lower()
    safety_like = any(
        marker in benchmark_type or marker in family
        for marker in ("safety", "security", "finance", "deployment", "critical")
    )
    if safety_like:
        initial = max(0.95, args.saturation_threshold)
        return {
            "initial_threshold": initial,
            "floor_threshold": initial,
            "subgroup_floor": initial,
            "no_decay": True,
        }
    if "diagnostic" in benchmark_type:
        return {
            "initial_threshold": min(args.saturation_threshold, 0.80),
            "floor_threshold": args.threshold_floor,
            "subgroup_floor": 0.50,
            "no_decay": False,
        }
    return {
        "initial_threshold": args.saturation_threshold,
        "floor_threshold": args.threshold_floor,
        "subgroup_floor": 0.50,
        "no_decay": False,
    }


def surface_attempt_key(surface: Surface) -> tuple[float, str]:
    seed = seed_from_path(surface.path)
    if seed is not None:
        return (float(seed), surface.path)
    try:
        return (Path(surface.path).stat().st_mtime, surface.path)
    except OSError:
        return (0.0, surface.path)


def recent_improvement(surface: Surface, history: list[Surface], window: int) -> float:
    if not history:
        return 0.0
    current_index = next(
        (idx for idx, candidate in enumerate(history) if candidate.path == surface.path),
        len(history) - 1,
    )
    previous_index = max(0, current_index - max(1, window))
    return clipped_score(surface.value) - clipped_score(history[previous_index].value)


def clipped_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def critical_failure_count(surface: Surface) -> int:
    if surface.external_inference_calls > 0:
        return 1
    benchmark_type = surface.benchmark_type.lower()
    if "safety" in benchmark_type and surface.residual > 0.0:
        return 1
    return 0


def threshold_phase_for(
    *,
    score: float,
    initial_threshold: float,
    current_threshold: float,
    floor_threshold: float,
    stalled: bool,
    decay_cycles: int,
    critical_failures: int,
    attempt_count: int,
    patience: int,
) -> str:
    if critical_failures > 0:
        return "critical_failure_veto"
    if score >= initial_threshold:
        return "mastery_phase_graduated"
    if score >= current_threshold and current_threshold < initial_threshold:
        return "decayed_threshold_graduated"
    if current_threshold <= floor_threshold and decay_cycles > 0:
        return "floor_phase"
    if decay_cycles > 0:
        return "decay_phase"
    if stalled:
        return "stalled_waiting_for_decay"
    if attempt_count <= patience:
        return "patience_phase"
    return "active_frontier_phase"


def extract_surface(
    path: Path,
    payload: dict[str, Any],
    saturation_threshold: float,
    broken_threshold: float,
) -> Surface | None:
    metric = None
    value = None
    family = None
    external_calls = int(payload.get("external_inference_calls", 0) or 0)

    summary = payload.get("summary")
    if isinstance(summary, dict) and "accuracy" in summary:
        family = str(summary.get("suite") or payload.get("suite") or "benchmark")
        metric = "accuracy"
        value = safe_float(summary.get("accuracy"))
        external_calls = int(summary.get("total_tool_calls", external_calls) or external_calls)
    elif isinstance(payload.get("eval"), dict) and isinstance(payload["eval"].get("summary"), dict):
        summary = payload["eval"]["summary"]
        family = str(summary.get("suite") or payload.get("feature_set") or "train_eval")
        metric = "eval_accuracy"
        value = safe_float(summary.get("accuracy"))
        external_calls = int(summary.get("total_tool_calls", external_calls) or external_calls)
    elif "normalized_perf" in payload:
        family = str(payload.get("env") or "puffer_ocean")
        metric = "normalized_perf"
        value = safe_float(payload.get("normalized_perf"))
    elif "eval_mean_reward" in payload:
        family = str(payload.get("env") or "rl_train")
        metric = "normalized_eval_reward"
        raw_reward = safe_float(payload.get("eval_mean_reward"))
        value = normalize_sparse_reward(family, raw_reward)

    if family is None or metric is None or value is None or not (0.0 <= value <= 1.25):
        return None
    family = classify_benchmark_family(family, path, payload)

    ceiling = learnable_score_ceiling(family)
    if ceiling is not None:
        value = min(1.0, value / ceiling)
        metric = f"ceiling_adjusted_{metric}"

    clipped = max(0.0, min(1.0, value))
    if external_calls > 0:
        status = "invalid_external_inference"
    elif clipped >= saturation_threshold:
        status = "saturated"
    elif clipped <= broken_threshold:
        status = "broken"
    else:
        status = "open"
    profile = family_profile(family)
    wall_type, recommended_intervention = diagnose_wall(family, status, metric, clipped)
    return Surface(
        path=str(path),
        family=family,
        metric=metric,
        value=value,
        residual=max(0.0, 1.0 - clipped),
        status=status,
        external_inference_calls=external_calls,
        lifecycle=lifecycle_for_status(status),
        capability=profile["capability"],
        benchmark_type=profile["benchmark_type"],
        contamination_risk=profile["contamination_risk"],
        label_quality=profile["label_quality"],
        transfer_evidence=profile["transfer_evidence"],
        cost_class=profile["cost_class"],
        wall_type=wall_type,
        recommended_intervention=recommended_intervention,
        score_ceiling=ceiling,
        capability_narrative=capability_narrative(family, status, clipped),
        curriculum_status=curriculum_status_for_status(status),
        comparator_class=profile["comparator_class"],
        initial_threshold=saturation_threshold,
        current_threshold=saturation_threshold,
        floor_threshold=0.70,
        subgroup_floor=0.50,
        threshold_phase="initial",
        attempt_count=1,
        recent_delta=None,
        stalled_cycles=0,
        decay_cycles=0,
        threshold_decay_rate=0.01,
        threshold_decay_mode="per_attempt_after_patience",
        critical_failure_count=0,
        frontier_momentum_action="pending_threshold_policy",
    )


def is_baseline_report(path: Path, payload: dict[str, Any]) -> bool:
    name = path.name.lower()
    if any(marker in name for marker in ("baseline", "bag_of_words", "first_allowed", "bow")):
        return True
    summary = payload.get("summary")
    if isinstance(summary, dict) and str(summary.get("mode", "")).lower() == "local_baseline":
        return True
    return False


def is_meta_report(payload: dict[str, Any]) -> bool:
    methodology = str(payload.get("methodology", ""))
    if methodology.startswith("benchmaxxing_"):
        return True
    if "benchmark_scores" in payload and "model_version" in payload:
        return True
    if "ratchet" in payload and "active_frontier" in payload:
        return True
    return False


def classify_benchmark_family(family: str, path: Path, payload: dict[str, Any]) -> str:
    """Split public BabyLM probes from private/mutated holdout probes."""

    if family != "babylm_local_probe":
        return family
    report_name = path.name.lower()
    eval_input = str(payload.get("eval_input_path") or "").lower()
    input_path = str(payload.get("input_path") or "").lower()
    if any(marker in report_name for marker in ("mutated", "holdout", "private")):
        return "babylm_mutated_holdout"
    if any(marker in eval_input for marker in ("mutated", "holdout", "private")):
        return "babylm_mutated_holdout"
    if any(marker in input_path for marker in ("mutated", "holdout", "private")):
        return "babylm_mutated_holdout"
    return family


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def normalize_sparse_reward(family: str, value: float | None) -> float | None:
    if value is None:
        return None
    normalizers = {
        "ocean-chain": 0.46875,
        "ocean-memory": 0.125,
        "ocean-tmaze": 0.125,
        "ocean-noisy-memory": 1.0 / 12.0,
        "ocean-noisy-tmaze": 0.1,
        "ocean-slot-tmaze": 0.1,
    }
    denom = normalizers.get(family)
    if denom is None:
        return value
    return min(1.0, max(0.0, value / denom))


def learnable_score_ceiling(family: str) -> float | None:
    """Return the observable optimum when a benchmark contains irreducible noise."""

    ceilings = {
        "ocean-noisy-memory": majority_vote_ceiling(samples=5, evidence_accuracy=0.75),
        "ocean-noisy-tmaze": majority_vote_ceiling(samples=5, evidence_accuracy=0.70),
    }
    return ceilings.get(family)


def majority_vote_ceiling(samples: int, evidence_accuracy: float) -> float:
    threshold = samples // 2 + 1
    total = 0.0
    for correct in range(threshold, samples + 1):
        total += (
            combinations(samples, correct)
            * evidence_accuracy**correct
            * (1.0 - evidence_accuracy) ** (samples - correct)
        )
    return total


def combinations(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for idx in range(1, k + 1):
        result = result * (n - k + idx) // idx
    return result


def family_profile(family: str) -> dict[str, str]:
    if family.startswith("minecraft_rl_"):
        return {
            "capability": "licensed local Minecraft and Minecraft-like open-world RL, crafting, navigation, and player-coop pressure",
            "benchmark_type": "frontier_minecraft_rl",
            "contamination_risk": "low_local_seed_when_disposable_worlds_are_used",
            "label_quality": "environment_reward_inventory_goal_and_trace_contract",
            "transfer_evidence": "transfers_to_open_world_agent_memory_tool_use_and_player_instruction_following",
            "cost_class": "medium_to_high_runtime",
            "comparator_class": "local_open_world_control_frontier",
        }
    if family.startswith("drone_rl_"):
        return {
            "capability": "simulation-first drone control and racing policy pressure",
            "benchmark_type": "frontier_drone_rl",
            "contamination_risk": "low_local_simulator_seed",
            "label_quality": "simulator_reward_and_safety_contract",
            "transfer_evidence": "transfers_to_ai_grand_prix_sim_lanes_after_adapter_maturity",
            "cost_class": "medium_gpu_or_cpu_sim",
            "comparator_class": "local_sim_control_frontier",
        }
    if family.startswith("drone_control_"):
        return {
            "capability": "governed simulator-first drone command API control",
            "benchmark_type": "diagnostic_drone_control_api",
            "contamination_risk": "low_local_contract",
            "label_quality": "contract_and_import_probe",
            "transfer_evidence": "required_for_sitl_and_real_hardware_after_approval",
            "cost_class": "fast",
            "comparator_class": "local_safety_contract",
        }
    if family.startswith("coding_") or family.startswith("transfer_code_repair"):
        return {
            "capability": "local code repair and sandboxed unit-test reasoning",
            "benchmark_type": "frontier_coding_local",
            "contamination_risk": "medium_public_benchmark_when_using_public_tasks",
            "label_quality": "unit_tests",
            "transfer_evidence": "transfers_to_self_debugging_and_teacher_reduction",
            "cost_class": "fast_to_medium",
            "comparator_class": "public_calibration_plus_local_sandbox",
        }
    if family.startswith("web_agent_") or family.startswith("transfer_web_task"):
        return {
            "capability": "self-hosted web task planning without real-account side effects",
            "benchmark_type": "frontier_web_agent_local",
            "contamination_risk": "medium_public_benchmark",
            "label_quality": "grader_or_service_fixture",
            "transfer_evidence": "transfers_to_agentic_browser_workflows",
            "cost_class": "medium",
            "comparator_class": "self_hosted_web_agent",
        }
    if family.startswith("conversation_") or family.startswith("dialogue_") or family == "personality_dialogue_core":
        return {
            "capability": "multi-turn conversational continuity, correction handling, constraint carry, and personality-context attachment",
            "benchmark_type": "frontier_tool_dialogue_agent_local",
            "contamination_risk": "low_local_contract",
            "label_quality": "deterministic_session_contract_and_personality_runtime_audit",
            "transfer_evidence": "transfers_to_operator_chat_mobile_hive_control_and_long_horizon_user_agent_work",
            "cost_class": "fast",
            "comparator_class": "local_conversation_runtime_guard",
        }
    if family.startswith("transfer_") or family == "asi_transfer_suite":
        return {
            "capability": "cross-domain agentic transfer pressure",
            "benchmark_type": "frontier_transfer_suite",
            "contamination_risk": "low_local_fixture",
            "label_quality": "local_contract_and_smoke_fixtures",
            "transfer_evidence": "intended_to_prevent_single_benchmark_overfit",
            "cost_class": "fast",
            "comparator_class": "local_transfer_guard",
        }
    profiles = {
        "babylm_local_probe": {
            "capability": "BabyLM/BLIMP-style linguistic generalization",
            "benchmark_type": "frontier_or_regression_public",
            "contamination_risk": "medium_public_benchmark",
            "label_quality": "public_eval_split",
            "transfer_evidence": "needs_private_and_mutated_holdouts",
            "cost_class": "medium_gpu_or_long_cpu",
            "comparator_class": "public_apples_to_apples",
        },
        "babylm_mutated_holdout": {
            "capability": "mutated BabyLM/BLIMP-style linguistic generalization under anti-Goodhart pressure",
            "benchmark_type": "frontier_mutated_holdout",
            "contamination_risk": "low_locally_generated_holdout",
            "label_quality": "programmatic_and_mutated_minimal_pairs",
            "transfer_evidence": "mutation_guard_for_public_babylm",
            "cost_class": "medium_gpu_or_long_cpu",
            "comparator_class": "private_mutation_guard",
        },
        "ocean-cartpole": {
            "capability": "low-latency continuous control reflex boundary",
            "benchmark_type": "regression_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward",
            "transfer_evidence": "local_puffer_style_only",
            "cost_class": "fast",
            "comparator_class": "public_sim_family_local_run",
        },
        "ocean-chain": {
            "capability": "short-horizon sparse-reward control",
            "benchmark_type": "regression_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward",
            "transfer_evidence": "local_puffer_style_only",
            "cost_class": "fast",
            "comparator_class": "public_sim_family_local_run",
        },
        "ocean-memory": {
            "capability": "delayed one-bit memory under sparse reward",
            "benchmark_type": "regression_memory_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward",
            "transfer_evidence": "local_puffer_style_only",
            "cost_class": "fast",
            "comparator_class": "public_sim_family_local_run",
        },
        "ocean-noisy-memory": {
            "capability": "evidence accumulation with irreducible observation noise",
            "benchmark_type": "regression_memory_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward_with_known_bayes_ceiling",
            "transfer_evidence": "transfers_to_noisy_tmaze_memory",
            "cost_class": "fast",
            "comparator_class": "local_curriculum_variant",
        },
        "ocean-noisy-tmaze": {
            "capability": "delayed noisy evidence accumulation plus branch governance",
            "benchmark_type": "regression_memory_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward_with_known_bayes_ceiling",
            "transfer_evidence": "transfers_from_noisy_memory_to_navigation",
            "cost_class": "fast",
            "comparator_class": "local_curriculum_variant",
        },
        "ocean-slot-tmaze": {
            "capability": "role-filler slot binding with delayed queried control",
            "benchmark_type": "regression_memory_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward",
            "transfer_evidence": "local_slot_memory_control",
            "cost_class": "fast",
            "comparator_class": "local_curriculum_variant",
        },
        "ocean-tmaze": {
            "capability": "delayed branch-memory control",
            "benchmark_type": "regression_control",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "simulator_reward",
            "transfer_evidence": "local_puffer_style_only",
            "cost_class": "fast",
            "comparator_class": "public_sim_family_local_run",
        },
        "unseen_adversarial_rag": {
            "capability": "verified evidence governance and abstention under missing evidence",
            "benchmark_type": "regression_adversarial_rag",
            "contamination_risk": "low_locally_generated_holdout",
            "label_quality": "programmatic_generation_with_verifier_contract",
            "transfer_evidence": "seed_mutation_passed",
            "cost_class": "fast",
            "comparator_class": "local_mutation_guard",
        },
        "cgs_frontier_governance": {
            "capability": "CGS governance routing sanity suite",
            "benchmark_type": "regression_governance",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "programmatic_contract",
            "transfer_evidence": "sanity_only",
            "cost_class": "fast",
            "comparator_class": "local_sanity_regression",
        },
        "cgs_hard_governance": {
            "capability": "harder CGS governance routing sanity suite",
            "benchmark_type": "regression_governance",
            "contamination_risk": "low_local_synthetic",
            "label_quality": "programmatic_contract",
            "transfer_evidence": "sanity_only",
            "cost_class": "fast",
            "comparator_class": "local_sanity_regression",
        },
    }
    return profiles.get(
        family,
        {
            "capability": "unclassified capability surface",
            "benchmark_type": "diagnostic",
            "contamination_risk": "unknown",
            "label_quality": "unknown",
            "transfer_evidence": "unknown",
            "cost_class": "unknown",
            "comparator_class": "unclassified",
        },
    )


def lifecycle_for_status(status: str) -> str:
    if status == "saturated":
        return "regression"
    if status == "open":
        return "frontier"
    if status == "invalid_external_inference":
        return "invalid"
    return "diagnostic"


def curriculum_status_for_status(status: str) -> str:
    if status == "saturated":
        return "curriculum_passed_promote_to_regression"
    if status == "open":
        return "active_frontier_pressure"
    if status == "invalid_external_inference":
        return "invalid_reject"
    return "diagnostic_audit"


def curriculum_status_for_surface(status: str, threshold_phase: str) -> str:
    if status == "saturated" and threshold_phase == "decayed_threshold_graduated":
        return "graduated_with_residual_escrow"
    if status == "saturated":
        return "curriculum_passed_promote_to_regression"
    if threshold_phase == "decay_phase":
        return "decay_phase_frontier_pressure"
    if threshold_phase == "floor_phase":
        return "floor_phase_bridge_or_architecture_diagnosis"
    if threshold_phase == "critical_failure_veto":
        return "blocked_by_critical_failure_veto"
    if threshold_phase == "patience_phase":
        return "patience_phase_frontier_pressure"
    if status == "invalid_external_inference":
        return "invalid_reject"
    if status == "broken":
        return "diagnostic_audit_or_bridge_benchmark"
    return "active_frontier_pressure"


def diagnose_wall(family: str, status: str, metric: str, score: float) -> tuple[str, str]:
    if status == "invalid_external_inference":
        return (
            "invalid_external_inference",
            "Reject this report for SymLiquid competition tracking; rerun with external_inference_calls=0.",
        )
    if status == "saturated":
        return (
            "no_current_wall",
            "Lock this benchmark into the regression suite and escalate to a harder frontier.",
        )
    if status == "broken":
        return (
            "benchmark_or_implementation_wall",
            "Audit benchmark validity, labels, scoring, and implementation before changing architecture.",
        )
    family_lower = family.lower()
    if "blimp" in family_lower or "babylm" in family_lower:
        return (
            "architecture_training_wall",
            "Run residual family analysis, add private/mutated linguistic holdouts, then improve learned sequence state before adding complexity.",
        )
    if family.startswith("ocean-"):
        return (
            "state_or_rollout_wall",
            "Try seed sweeps and trainable state dynamics; if the wall persists, move rollout training into Rust/CUDA kernels.",
        )
    if family.startswith("drone_rl_"):
        return (
            "state_or_rollout_wall",
            "Train local hover/waypoint/racing controllers first; if score stalls below floor, ask the teacher for the smallest drone-control architecture upgrade.",
        )
    if family.startswith("drone_control_"):
        return (
            "evaluation_frontier_wall",
            "Complete simulator/SITL contract scoring before any live hardware lane; hardware remains approval-gated.",
        )
    if family.startswith("conversation_") or family.startswith("dialogue_"):
        return (
            "conversation_state_or_personality_integration_wall",
            "Run the multi-turn conversation benchmark, inspect missing terms and personality-context turns, then patch session continuity or personality-runtime attachment.",
        )
    if "rag" in family_lower:
        return (
            "evaluation_frontier_wall",
            "Generate a harder mutated holdout with new templates and verify transfer before tuning the scorer.",
        )
    return (
        "undifferentiated_wall",
        f"Run the diagnostic ladder: benchmark audit, data, training, inference, then architecture. Current score={score:.4f} metric={metric}.",
    )


def capability_narrative(family: str, status: str, score: float) -> str:
    return capability_narrative_with_threshold(
        family=family,
        status=status,
        score=score,
        current_threshold=0.90,
        threshold_phase="initial",
    )


def capability_narrative_with_threshold(
    family: str,
    status: str,
    score: float,
    current_threshold: float,
    threshold_phase: str,
) -> str:
    profile = family_profile(family)
    if status == "saturated":
        return (
            f"{profile['capability']} graduated at score={score:.4f} "
            f"against threshold={current_threshold:.4f} ({threshold_phase}); preserve it "
            "as regression and send remaining failures to residual escrow."
        )
    if status == "open":
        return (
            f"{profile['capability']} remains a frontier pressure surface at score={score:.4f}; "
            f"current graduation threshold={current_threshold:.4f} ({threshold_phase}). "
            "Use residuals to decide whether the wall is data, training, inference, evaluation, architecture, or bridge-benchmark need."
        )
    return (
        f"{profile['capability']} is not yet a reliable pressure surface; audit it before using it for model evolution."
    )


def frontier_momentum_action(surface: Surface) -> str:
    if surface.critical_failure_count > 0:
        return "blocked_by_critical_failure_veto"
    if surface.status == "saturated":
        return "graduate_to_regression_and_place_failures_in_residual_escrow"
    if surface.threshold_phase == "floor_phase":
        return "cannot_clear_floor_create_bridge_or_diagnose_architecture_wall"
    if surface.threshold_phase == "decay_phase":
        return "continue_frontier_pressure_at_decayed_threshold"
    if surface.status == "broken":
        return "audit_or_insert_bridge_benchmark"
    return "continue_mastery_pressure"


def benchmark_ledger_entry(surface: Surface) -> dict[str, Any]:
    return {
        "benchmark_name": surface.family,
        "best_report": surface.path,
        "capability_measured": surface.capability,
        "benchmark_type": surface.benchmark_type,
        "lifecycle": surface.lifecycle,
        "saturation_status": surface.status,
        "metric": surface.metric,
        "score": surface.value,
        "residual": surface.residual,
        "score_ceiling": surface.score_ceiling,
        "contamination_risk": surface.contamination_risk,
        "label_test_quality": surface.label_quality,
        "cost_class": surface.cost_class,
        "transfer_evidence": surface.transfer_evidence,
        "regression_value": "preserve" if surface.status == "saturated" else "pending",
        "wall_type": surface.wall_type,
        "recommended_intervention": surface.recommended_intervention,
        "retirement_criteria": retirement_criteria(surface),
        "capability_narrative": surface.capability_narrative,
        "curriculum_status": surface.curriculum_status,
        "graduation_policy": {
            "initial_threshold": surface.initial_threshold,
            "current_threshold": surface.current_threshold,
            "floor_threshold": surface.floor_threshold,
            "subgroup_floor": surface.subgroup_floor,
            "threshold_phase": surface.threshold_phase,
            "attempt_count": surface.attempt_count,
            "recent_delta": surface.recent_delta,
            "stalled_cycles": surface.stalled_cycles,
            "decay_cycles": surface.decay_cycles,
            "decay_mode": surface.threshold_decay_mode,
            "threshold_decay_rate": surface.threshold_decay_rate,
            "critical_failure_count": surface.critical_failure_count,
            "frontier_momentum_action": surface.frontier_momentum_action,
        },
        "comparator_class": surface.comparator_class,
        "external_inference_calls": surface.external_inference_calls,
    }


def build_public_comparator_ledger(
    benchmark_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    comparators = [
        entry
        for entry in benchmark_ledger
        if entry.get("comparator_class") == "public_apples_to_apples"
        or "public" in str(entry.get("benchmark_type", ""))
        or "public" in str(entry.get("contamination_risk", ""))
    ]
    return {
        "policy": "local_only_no_external_inference",
        "methodology": "public_comparator_ledger",
        "purpose": "Keep public, externally recognizable benchmarks visible for apples-to-apples capability comparison while private/mutated frontiers carry anti-Goodhart pressure.",
        "comparison_rule": "Report public benchmark scores regularly, but promote candidates only when public gains transfer to private/mutated or live local holdouts.",
        "minimum_public_cadence": "run_public_comparators_before_candidate_promotion",
        "comparators": [
            {
                "benchmark_name": entry["benchmark_name"],
                "capability_measured": entry["capability_measured"],
                "score": entry["score"],
                "residual": entry["residual"],
                "current_threshold": entry.get("graduation_policy", {}).get(
                    "current_threshold"
                ),
                "threshold_phase": entry.get("graduation_policy", {}).get(
                    "threshold_phase"
                ),
                "lifecycle": entry["lifecycle"],
                "curriculum_status": entry.get("curriculum_status"),
                "contamination_risk": entry["contamination_risk"],
                "best_report": entry["best_report"],
                "apples_to_apples_note": public_comparator_note(entry),
            }
            for entry in comparators
        ],
    }


def public_comparator_note(entry: dict[str, Any]) -> str:
    name = entry.get("benchmark_name", "")
    if "babylm" in name or "blimp" in name:
        return "Use as the public BLIMP/BabyLM comparator; pair with mutated BabyLM holdouts before promotion."
    return "Use as a public/local-comparable benchmark surface; do not let it replace private anti-Goodhart pressure."


def retirement_criteria(surface: Surface) -> str:
    if surface.contamination_risk.startswith("medium") or surface.contamination_risk.startswith("high"):
        return "Retire or demote if private/mutated holdouts diverge from public score."
    if surface.label_quality == "simulator_reward_with_known_bayes_ceiling":
        return "Retire as frontier when ceiling-adjusted score is saturated; preserve as noise-regression check."
    if surface.status == "saturated":
        return "Keep as regression until a harder benchmark fully dominates this capability."
    return "Retire if residuals stop being diagnostic or labels/tests prove unstable."


def build_model_ledger(
    family_surfaces: list[Surface],
    open_surfaces: list[Surface],
    saturated: list[Surface],
    broken: list[Surface],
) -> dict[str, Any]:
    active_frontier = open_surfaces[0] if open_surfaces else None
    return {
        "model_version": "symliquid-local-current",
        "architecture": "SymLiquid CGS: liquid/reservoir/VSA state, verifier-governed outputs, Rust/CUDA local training surfaces",
        "training_data": "local public benchmark snapshots, generated synthetic diagnostics, Puffer/Ocean-style simulators; no external inference",
        "training_process": "local readout/state training, Rust FFI rollout CEM, CUDA parity kernels, seed sweeps where available",
        "inference_process": "standalone local SymLiquid scoring and Rust FFI policy execution; provider calls forbidden",
        "benchmark_scores": {
            surface.family: {
                "metric": surface.metric,
                "score": surface.value,
                "residual": surface.residual,
                "lifecycle": surface.lifecycle,
                "report": surface.path,
                "current_threshold": surface.current_threshold,
                "threshold_phase": surface.threshold_phase,
                "frontier_momentum_action": surface.frontier_momentum_action,
            }
            for surface in family_surfaces
        },
        "residual_map": [
            {
                "family": surface.family,
                "residual": surface.residual,
                "wall_type": surface.wall_type,
                "recommended_intervention": surface.recommended_intervention,
                "current_threshold": surface.current_threshold,
                "threshold_phase": surface.threshold_phase,
                "frontier_momentum_action": surface.frontier_momentum_action,
            }
            for surface in open_surfaces
        ],
        "regression_status": {
            "locked_families": [surface.family for surface in saturated],
            "broken_families": [surface.family for surface in broken],
            "external_inference_violations": [
                surface.family for surface in family_surfaces if surface.external_inference_calls > 0
            ],
        },
        "cost_profile": {
            "fast_regressions": sum(1 for surface in family_surfaces if surface.cost_class == "fast"),
            "medium_or_slower_frontiers": [
                surface.family for surface in open_surfaces if surface.cost_class != "fast"
            ],
        },
        "safety_profile": {
            "external_inference_policy": "forbidden",
            "adversarial_rag_status": next(
                (
                    {
                        "score": surface.value,
                        "status": surface.status,
                        "report": surface.path,
                    }
                    for surface in family_surfaces
                    if surface.family == "unseen_adversarial_rag"
                ),
                None,
            ),
        },
        "next_wall": None
        if active_frontier is None
        else {
            "family": active_frontier.family,
            "capability": active_frontier.capability,
            "wall_type": active_frontier.wall_type,
            "recommended_intervention": active_frontier.recommended_intervention,
        },
    }


def build_ratchet_report(
    open_surfaces: list[Surface],
    saturated: list[Surface],
    broken: list[Surface],
    invalid_external: list[Surface],
) -> dict[str, Any]:
    return {
        "rule": "frontier_gain_required_without_regression_loss",
        "mastery_then_momentum_rule": {
            "policy": "start_high_decay_one_percent_per_attempt_after_patience_never_below_floor",
            "frontier_momentum": "If score clears the current threshold and no critical failures remain, graduate benchmark and move unsolved cases to residual escrow.",
            "floor_failure": "If a benchmark cannot clear the floor, create a bridge benchmark or diagnose architecture/evaluation quality.",
        },
        "stability_force": {
            "regression_suite": [surface.family for surface in saturated],
            "regression_count": len(saturated),
        },
        "pressure_force": {
            "active_frontier": [surface.family for surface in open_surfaces],
            "frontier_count": len(open_surfaces),
        },
        "diagnostic_ladder": [
            "benchmark_audit",
            "data_improvement",
            "training_improvement",
            "inference_improvement",
            "architecture_change",
        ],
        "anti_goodhart": {
            "public_frontier_count": sum(
                1 for surface in open_surfaces if "public" in surface.contamination_risk
            ),
            "mutation_or_generated_holdouts": [
                surface.family
                for surface in saturated + open_surfaces
                if "generated" in surface.label_quality or "mutation" in surface.transfer_evidence
            ],
            "external_inference_violations": [surface.family for surface in invalid_external],
            "public_comparator_surfaces": [
                surface.family
                for surface in open_surfaces + saturated
                if surface.comparator_class == "public_apples_to_apples"
            ],
            "warning": anti_goodhart_warning(open_surfaces, invalid_external),
        },
        "next_ratchet_action": next_ratchet_action(open_surfaces, broken, invalid_external),
    }


def anti_goodhart_warning(open_surfaces: list[Surface], invalid_external: list[Surface]) -> str:
    if invalid_external:
        return "Invalid run detected: provider/tool inference calls must be removed before comparing scores."
    if any("public" in surface.contamination_risk for surface in open_surfaces):
        return "Active frontier includes a public benchmark; pair improvements with private or mutated holdouts."
    return "No immediate Goodhart warning, but rotate frontiers after saturation."


def next_ratchet_action(
    open_surfaces: list[Surface],
    broken: list[Surface],
    invalid_external: list[Surface],
) -> str:
    if invalid_external:
        return "Reject invalid reports and rerun locally with external_inference_calls=0."
    if broken:
        return "Audit broken benchmark surfaces before using them for architecture decisions."
    if open_surfaces:
        frontier = open_surfaces[0]
        return f"Attack {frontier.family}: {frontier.recommended_intervention}"
    return "All tracked surfaces saturated; add harder unsaturated benchmark families."


def write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def recommend_next_commands(
    open_surfaces: list[Surface],
    saturated: list[Surface],
    all_surfaces: list[Surface],
) -> list[str]:
    commands = []
    families = {surface.family for surface in open_surfaces}
    if "ocean-slot-tmaze" in families:
        commands.append(
            "python adapters/pufferlib/symliquid_puffer_adapter.py --train-discrete-policy --env ocean-slot-tmaze --iterations 24 --population 40 --elite-count 8 --num-envs 128 --train-steps 384 --eval-steps 2048 --seed 3 --use-rust-ffi --policy-out reports/symliquid_ocean_slot_tmaze_policy_rust_trainer_seed3.json --out reports/symliquid_ocean_slot_tmaze_policy_rust_trainer_seed3_train.json"
        )
    if "ocean-noisy-tmaze" in families:
        commands.append(
            "python adapters/pufferlib/symliquid_puffer_adapter.py --train-discrete-policy --env ocean-noisy-tmaze --iterations 32 --population 48 --elite-count 8 --num-envs 128 --train-steps 512 --eval-steps 2048 --seed 1 --use-rust-ffi --policy-out reports/symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed1_long.json --out reports/symliquid_ocean_noisy_tmaze_policy_sum_rust_trainer_seed1_long_train.json"
        )
    if "drone_rl_source_pyflyt_waypoints" in families:
        next_seed = next_pressure_seed(all_surfaces, "source_pyflyt_waypoints")
        commands.append(
            f"python scripts/pressure_runner.py --card-id source_pyflyt_waypoints --frontier-family drone_rl --seed {next_seed} --episodes 2 --steps 128 --train-iterations 4 --train-population 12 --elite-count 4 --out reports/pressure_source_pyflyt_waypoints_seed{next_seed}.json"
        )
        commands.append(
            "Next benchmark frontier: keep PyFlyt waypoint pressure active until it clears the floor or triggers a bridge/architecture diagnosis; preserve hover and gym-pybullet as regression."
        )
    if "coding_source_bigcodebench" in families:
        next_seed = next_pressure_seed(all_surfaces, "source_bigcodebench")
        commands.append(
            f"python scripts/pressure_runner.py --card-id source_bigcodebench --frontier-family coding_local_sandbox --seed {next_seed} --out reports/pressure_source_bigcodebench_seed{next_seed}.json"
        )
    if "web_agent_source_webarena" in families:
        next_seed = next_pressure_seed(all_surfaces, "source_webarena")
        commands.append(
            f"python scripts/pressure_runner.py --card-id source_webarena --frontier-family web_agent_local --seed {next_seed} --out reports/pressure_source_webarena_seed{next_seed}.json"
        )
    if any(
        "blimp" in surface.family.lower() or "babylm" in surface.family.lower()
        for surface in open_surfaces
    ):
        commands.append(
            "python scripts/analyze_babylm_residuals.py --report reports/blimp_filtered_train_800k_evalfull_hv16k_lr02_complexnpfix.json --eval-input data/babylm_blimp_filtered_eval.jsonl --out reports/babylm_residual_analysis.json"
        )
        commands.append(
            "python scripts/generate_babylm_mutated_holdout.py --residual-analysis reports/babylm_residual_analysis.json --count 2400 --seed 31 --out data/babylm_mutated_holdout_seed31.jsonl --report-out reports/babylm_mutated_holdout_seed31_factory.json"
        )
        commands.append(
            "cargo run --release -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_mutated_holdout_seed31.jsonl --train-limit 53888 --eval-limit 2400 --steps 60000 --hv-dim 8192 --lr 0.08 --stateful --pairwise-contrast --balance-rules --prior-weight 1.0 --out reports/babylm_mutated_holdout_seed31_stateful_grammar_state_probe.json"
        )
        commands.append(
            "Next architecture step: learned liquid/reservoir/VSA grammar state for BLIMP residual families, then validate on private or mutated BabyLM-style holdouts."
        )
    saturated_families = {surface.family for surface in saturated}
    non_babylm_frontiers = [
        surface
        for surface in open_surfaces
        if "babylm" not in surface.family.lower() and "blimp" not in surface.family.lower()
    ]
    if (
        "babylm_mutated_holdout" in saturated_families
        and "babylm_mutated_holdout" not in families
        and not non_babylm_frontiers
    ):
        next_seed = next_mutated_babylm_seed(all_surfaces)
        next_count = 4800 if next_seed >= 43 else 3600
        commands.append(
            f"python scripts/generate_babylm_mutated_holdout.py --residual-analysis reports/babylm_mutated_residual_analysis.json --count {next_count} --seed {next_seed} --out data/babylm_mutated_holdout_seed{next_seed}.jsonl --report-out reports/babylm_mutated_holdout_seed{next_seed}_factory.json"
        )
        commands.append(
            f"cargo run --release -p symliquid-cli -- train-babylm-probe --input data/babylm_blimp_filtered_train.jsonl --eval-input data/babylm_mutated_holdout_seed{next_seed}.jsonl --train-limit 53888 --eval-limit {next_count} --steps 120000 --hv-dim 8192 --lr 0.08 --stateful --pairwise-contrast --balance-rules --prior-weight 1.0 --out reports/babylm_mutated_holdout_seed{next_seed}_stateful_grammar_state_frontier.json"
        )
        commands.append(
            f"Next benchmark frontier: rotate to fresh mutated BabyLM seed{next_seed} because the previous mutated holdout passed the current curriculum threshold."
        )
    if not commands and open_surfaces:
        commands.append(
            f"Run diagnostic ladder for {open_surfaces[0].family}: benchmark audit, data, training, inference, architecture."
        )
    if not commands and saturated:
        commands.append(
            "python scripts/generate_unseen_adversarial_rag.py --count 360 --seed 29 --out benchmarks/snapshots/unseen_adversarial_rag_seed29_harder.json"
        )
        commands.append(
            "cargo run --release -p symliquid-cli -- benchmark-symliquid --suite benchmarks/snapshots/unseen_adversarial_rag_seed29_harder.json --model-id symliquid-unseen-rag-governance --out reports/symliquid_unseen_adversarial_rag_seed29_harder.json"
        )
    if not any(command.startswith("Next architecture step:") for command in commands):
        commands.append(
            "Next architecture step: move rust_ffi_rollout_trainer env stepping/reward/done buffers into CUDA batched kernels."
        )
    return commands


def next_mutated_babylm_seed(saturated: list[Surface]) -> int:
    seeds = []
    for surface in saturated:
        if surface.family != "babylm_mutated_holdout":
            continue
        match = re.search(r"seed(\d+)", surface.path)
        if match:
            seeds.append(int(match.group(1)))
    return max(seeds, default=31) + 6


def next_pressure_seed(surfaces: list[Surface], card_id: str) -> int:
    seeds = []
    marker = f"pressure_{card_id}_seed"
    for surface in surfaces:
        normalized = str(surface.path).replace("\\", "/")
        if marker not in normalized:
            continue
        match = re.search(r"seed(\d+)", normalized)
        if match:
            seeds.append(int(match.group(1)))
    return max(seeds, default=0) + 1


if __name__ == "__main__":
    raise SystemExit(main())
