"""Run a deterministic active-inference/world-model pilot.

BugBrain's active-inference idea is useful only if it produces concrete local
signals: prediction error, expected-free-energy action rankings, and governed
belief updates. This script runs a tiny replayable line-world pilot against the
current active-inference and world-adapter reports. It does not train a model,
touch hardware, or use external inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVE = ROOT / "reports" / "active_inference_world_model.json"
DEFAULT_WORLD = ROOT / "reports" / "world_adapter_job_runtime.json"
DEFAULT_OUT = ROOT / "reports" / "legacy_active_inference_pilot.json"
DEFAULT_BELIEF_OUT = ROOT / "data" / "world_model" / "active_inference_belief_updates.jsonl"

ACTIONS = {
    "observe": {"delta": 0.0, "cost": 0.05, "epistemic_value": 0.28},
    "approach": {"delta": 1.0, "cost": 0.08, "epistemic_value": 0.10},
    "retreat": {"delta": -1.0, "cost": 0.08, "epistemic_value": 0.05},
    "stabilize": {"delta": 0.0, "cost": 0.03, "epistemic_value": 0.12},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-inference", default=str(DEFAULT_ACTIVE.relative_to(ROOT)))
    parser.add_argument("--world-runtime", default=str(DEFAULT_WORLD.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--belief-out", default=str(DEFAULT_BELIEF_OUT.relative_to(ROOT)))
    parser.add_argument("--steps", type=int, default=7)
    args = parser.parse_args()

    report = build_report(
        active_path=resolve(args.active_inference),
        world_path=resolve(args.world_runtime),
        belief_out=resolve(args.belief_out),
        steps=max(3, args.steps),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(*, active_path: Path, world_path: Path, belief_out: Path, steps: int) -> dict[str, Any]:
    active = read_json(active_path)
    world = read_json(world_path)
    trace, beliefs = run_line_world(steps=steps)
    write_jsonl(belief_out, beliefs)

    errors = [float(row["prediction_error"]) for row in trace]
    rankings = sum(len(row.get("expected_free_energy_ranking") or []) for row in trace)
    active_blockers = [
        row.get("name")
        for row in active.get("checks", [])
        if isinstance(row, dict) and not row.get("passed")
    ]
    world_jobs = [row for row in world.get("jobs", []) if isinstance(row, dict)]
    ready_jobs = [row for row in world_jobs if row.get("status") == "ready"]
    live_hardware = any(bool(row.get("live_hardware_allowed")) for row in world_jobs)
    accepted_updates = [row for row in beliefs if row.get("decision") == "accepted_local_world_model_update"]
    quarantined_updates = [row for row in beliefs if row.get("decision") == "quarantined"]

    gates = [
        gate("active_inference_report_ready", active.get("status") == "READY", active.get("status")),
        gate("active_inference_checks_pass", not active_blockers, active_blockers),
        gate("world_adapter_contract_green", world.get("trigger_state") in {"GREEN", "YELLOW"}, world.get("trigger_state")),
        gate("world_jobs_have_replay_ids", all(job.get("checkpoint_hash") for job in world_jobs), len(world_jobs)),
        gate("live_hardware_disabled", not live_hardware, live_hardware),
        gate("prediction_error_emitted", bool(errors), f"samples={len(errors)}"),
        gate("expected_free_energy_rankings_emitted", rankings >= steps * 3, f"ranked_actions={rankings}"),
        gate("belief_updates_governed", bool(beliefs) and not quarantined_updates, f"updates={len(beliefs)} quarantined={len(quarantined_updates)}"),
        gate("external_inference_zero", True, "deterministic local toy world only"),
    ]
    failed = [row["gate"] for row in gates if not row["passed"]]
    mean_error = round(sum(errors) / max(1, len(errors)), 6)
    trigger_state = "RED" if failed else ("GREEN" if mean_error <= 0.12 and accepted_updates else "YELLOW")
    replay_id = stable_id({"trace": trace, "beliefs": beliefs, "active": active.get("created_utc"), "world": world.get("created_utc")})

    return {
        "policy": "project_theseus_legacy_active_inference_pilot_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "ready_for_world_model_training_signal": trigger_state in {"GREEN", "YELLOW"},
        "replay_id": replay_id,
        "inputs": {
            "active_inference_world_model": rel_or_abs(active_path),
            "world_adapter_job_runtime": rel_or_abs(world_path),
        },
        "outputs": {
            "belief_updates": rel(belief_out),
        },
        "summary": {
            "steps": len(trace),
            "mean_prediction_error": mean_error,
            "max_prediction_error": round(max(errors) if errors else 0.0, 6),
            "final_state": trace[-1]["observed_state"] if trace else None,
            "target_state": 3,
            "action_rankings": rankings,
            "belief_updates": len(beliefs),
            "accepted_belief_updates": len(accepted_updates),
            "quarantined_belief_updates": len(quarantined_updates),
            "world_jobs": len(world_jobs),
            "ready_world_jobs": len(ready_jobs),
            "live_hardware_allowed": live_hardware,
            "external_inference_calls": 0,
        },
        "toy_world_contract": {
            "world": "bounded_line_world",
            "state_space": "integer positions 0..4",
            "target_state": 3,
            "episode_policy": "choose minimum expected free energy; no stochastic exploration",
            "hardware": "forbidden",
            "network": "forbidden",
        },
        "trace": trace,
        "belief_updates": beliefs,
        "gates": gates,
        "next_actions": [
            "Feed accepted local world-model updates into residual and trace-fabric governance before scaling to real RL adapters.",
            "Use expected-free-energy action rankings as an eval signal, not as public benchmark evidence.",
            "Keep the next pilot on ready non-hardware adapters such as coding_local_sandbox or web_agent_local.",
        ],
        "external_inference_calls": 0,
    }


def run_line_world(*, steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = 0
    target = 3
    transition_belief = {
        "observe": 0.0,
        "approach": 0.75,
        "retreat": -1.0,
        "stabilize": 0.0,
    }
    confidence = {
        "observe": 0.80,
        "approach": 0.72,
        "retreat": 0.78,
        "stabilize": 0.82,
    }
    trace = []
    beliefs = []
    for step in range(steps):
        ranking = rank_actions(state, target, transition_belief, confidence)
        action = ranking[0]["action"]
        predicted_state = clamp_state(state + transition_belief[action])
        observed_state = apply_action(state, action, target)
        error = abs(float(observed_state) - predicted_state)
        confidence[action] = round(min(0.98, max(0.55, confidence[action] + 0.08 * (1.0 - error))), 6)
        transition_belief[action] = round(transition_belief[action] + 0.5 * (float(observed_state - state) - transition_belief[action]), 6)
        trace_row = {
            "step": step,
            "prior_state": state,
            "selected_action": action,
            "predicted_state": round(predicted_state, 6),
            "observed_state": observed_state,
            "prediction_error": round(error, 6),
            "updated_transition_belief": transition_belief[action],
            "updated_confidence": confidence[action],
            "expected_free_energy_ranking": ranking,
        }
        trace.append(trace_row)
        if error > 0.001 or action in {"approach", "stabilize"}:
            beliefs.append(make_belief_update(trace_row, target))
        state = observed_state
    return trace, beliefs


def rank_actions(
    state: int,
    target: int,
    transition_belief: dict[str, float],
    confidence: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for action, spec in ACTIONS.items():
        predicted = clamp_state(state + transition_belief[action])
        risk = abs(float(target) - predicted) / 4.0
        ambiguity = 1.0 - float(confidence[action])
        cost = float(spec["cost"])
        epistemic_value = float(spec["epistemic_value"])
        efe = risk + 0.35 * ambiguity + cost - 0.22 * epistemic_value
        rows.append(
            {
                "action": action,
                "predicted_state": round(predicted, 6),
                "expected_free_energy": round(efe, 6),
                "risk": round(risk, 6),
                "ambiguity": round(ambiguity, 6),
                "cost": cost,
                "epistemic_value": epistemic_value,
            }
        )
    return sorted(rows, key=lambda row: (float(row["expected_free_energy"]), row["action"]))


def apply_action(state: int, action: str, target: int) -> int:
    if action == "approach" and state < target:
        return clamp_state_int(state + 1)
    if action == "retreat":
        return clamp_state_int(state - 1)
    return clamp_state_int(state)


def make_belief_update(trace_row: dict[str, Any], target: int) -> dict[str, Any]:
    action = str(trace_row["selected_action"])
    confidence = float(trace_row["updated_confidence"])
    decision = "accepted_local_world_model_update" if confidence >= 0.72 else "needs_review"
    inferred = (
        f"In the bounded line-world pilot, action '{action}' has transition delta "
        f"{trace_row['updated_transition_belief']} near target {target}."
    )
    return {
        "policy": "project_theseus_active_inference_belief_update_v1",
        "created_utc": now(),
        "belief_id": stable_id({"step": trace_row["step"], "action": action, "belief": inferred}),
        "source": "legacy_active_inference_pilot",
        "observation": (
            f"step={trace_row['step']} prior={trace_row['prior_state']} action={action} "
            f"predicted={trace_row['predicted_state']} observed={trace_row['observed_state']} "
            f"prediction_error={trace_row['prediction_error']}"
        ),
        "inferred_belief": inferred,
        "confidence": round(confidence, 6),
        "decision": decision,
        "conflict_with_inherited_core": False,
        "conflicts": [],
        "review_flags": [],
        "governance": {
            "toy_world_only": True,
            "not_public_benchmark_claim_evidence": True,
            "requires_real_adapter_replay_before_generalization": True,
            "external_inference_calls": 0,
        },
    }


def clamp_state(value: float) -> float:
    return min(4.0, max(0.0, value))


def clamp_state_int(value: int) -> int:
    return min(4, max(0, value))


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def stable_id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rel_or_abs(path: Path) -> str:
    try:
        return rel(path)
    except ValueError:
        return str(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
