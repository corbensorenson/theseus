"""Close the loop after a candidate promotion.

Candidate promotion should be an active transition, not just a green report.
This script records the accepted frontier, marks the mastered surface as a
regression through a lifecycle override, requests a fresh harder frontier, and
emits a small ledger record for checkpoint/update/backup systems.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CODING_PRESSURE_CARD_ORDER = [
    "source_bigcodebench",
    "source_evalplus",
    "source_human_eval",
    "source_mbpp",
    "source_livecodebench",
    "source_opencode",
    "source_swe_bench",
    "source_swe_agent",
    "source_mini_swe_agent",
    "source_codeclash",
    "source_swe_polybench",
    "source_swe_gen",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="reports/candidate_promotion_gate.json")
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--frontier-policy", default="reports/frontier_policy_status.json")
    parser.add_argument("--accepted-registry", default="reports/accepted_candidate_registry.json")
    parser.add_argument("--lifecycle-overrides", default="reports/benchmark_lifecycle_overrides.json")
    parser.add_argument("--rotation-request", default="reports/frontier_rotation_request.json")
    parser.add_argument("--ledger", default="reports/promotion_closure_ledger.jsonl")
    parser.add_argument("--out", default="reports/promotion_closure.json")
    args = parser.parse_args()

    candidate = read_json(ROOT / args.candidate)
    ledger = read_json(ROOT / args.benchmark_ledger)
    frontier_policy = read_json(ROOT / args.frontier_policy)
    report = close_promotion(args, candidate, ledger, frontier_policy)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", False) else 1


def close_promotion(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    ledger: Any,
    frontier_policy: dict[str, Any],
) -> dict[str, Any]:
    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    active_report = normalize_path(artifacts.get("active_frontier") or "")
    family_hint = str(artifacts.get("active_frontier_family") or "")
    score = safe_float((candidate.get("scores") or {}).get("active_frontier_accuracy"))

    if not candidate.get("promote"):
        report = {
            "policy": "project_theseus_promotion_closure_v0",
            "created_utc": now(),
            "ok": True,
            "status": "idle_candidate_not_promoted",
            "candidate_promote": candidate.get("promote"),
            "active_frontier": active_report,
            "external_inference_calls": 0,
        }
        return report

    row = match_frontier_row(ledger, active_report, family_hint)
    benchmark_name = str(row.get("benchmark_name") or suite_from_report(active_report) or family_hint or "unknown_frontier")
    accepted = {
        "candidate_id": f"accepted_{int(time.time() * 1000)}",
        "accepted_utc": now(),
        "benchmark_name": benchmark_name,
        "frontier_family": family_hint or family_from_benchmark(benchmark_name),
        "active_frontier_report": active_report,
        "score": score,
        "candidate_passed": candidate.get("passed"),
        "candidate_total": candidate.get("total"),
        "residual_delta": candidate.get("residual_delta"),
        "promotion_checks": compact_checks(candidate.get("checks", [])),
        "status": "accepted_for_checkpoint_and_regression",
    }

    registry_path = ROOT / args.accepted_registry
    registry = read_json(registry_path)
    registry = ensure_registry(registry)
    registry["accepted_candidates"] = upsert_by_report(registry["accepted_candidates"], accepted)
    registry["updated_utc"] = accepted["accepted_utc"]
    write_json(registry_path, registry)

    override_path = ROOT / args.lifecycle_overrides
    overrides = read_json(override_path)
    overrides = ensure_overrides(overrides)
    override = {
        "benchmark_name": benchmark_name,
        "report": active_report,
        "lifecycle": "regression",
        "saturation_status": "saturated",
        "threshold_phase": "candidate_promotion_override",
        "reason": "candidate_promoted",
        "score": score,
        "accepted_utc": accepted["accepted_utc"],
    }
    overrides["promotions"] = upsert_override(overrides.get("promotions", []), override)
    overrides["updated_utc"] = accepted["accepted_utc"]
    write_json(override_path, overrides)

    rotation = build_rotation_request(accepted, ledger, frontier_policy)
    write_json(ROOT / args.rotation_request, rotation)

    append_jsonl(ROOT / args.ledger, {"event": "promotion_closed", **accepted, "rotation": rotation})
    return {
        "policy": "project_theseus_promotion_closure_v0",
        "created_utc": accepted["accepted_utc"],
        "ok": True,
        "status": "promotion_closed",
        "accepted_candidate": accepted,
        "accepted_registry": args.accepted_registry,
        "lifecycle_overrides": args.lifecycle_overrides,
        "rotation_request": rotation,
        "external_inference_calls": 0,
    }


def build_rotation_request(
    accepted: dict[str, Any],
    ledger: Any,
    frontier_policy: dict[str, Any],
) -> dict[str, Any]:
    family = str(accepted.get("frontier_family") or "")
    accepted_name = str(accepted.get("benchmark_name") or "")
    next_card = next_pressure_card(family, accepted_name, ledger, frontier_policy)
    next_family = family if family in {"minecraft_rl", "drone_rl", "coding_local_sandbox", "web_agent_local", "transfer_eval"} else ""
    if not next_family:
        next_family = str(get_path(frontier_policy, ["frontier_pressure", "next_frontier_family"], "") or "rl_local")
    next_seed = int(get_path(frontier_policy, ["frontier_pressure", "next_rl_frontier_seed"], 1) or 1)
    if next_card and next_card != get_path(frontier_policy, ["pressure_card_id"], ""):
        next_seed = max(next_seed, seed_from_text(str(accepted.get("active_frontier_report") or "")) + 1)
    return {
        "policy": "project_theseus_frontier_rotation_request_v0",
        "request_id": f"rotation_{int(time.time() * 1000)}",
        "created_utc": now(),
        "expires_utc": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(),
        "consumed_utc": "",
        "reason": "candidate_promoted_rotate_to_harder_frontier",
        "accepted_benchmark": accepted_name,
        "accepted_report": accepted.get("active_frontier_report"),
        "frontier_family": next_family,
        "pressure_card_id": next_card,
        "rl_frontier_seed": next_seed,
        "profile": "inner_loop",
        "teacher_reason": "benchmark_frontier_design",
    }


def next_pressure_card(
    family: str,
    accepted_name: str,
    ledger: Any,
    frontier_policy: dict[str, Any],
) -> str:
    curriculum_card = str(get_path(frontier_policy, ["frontier_pressure", "curriculum_recommended_env"], "") or "")
    current_card = card_from_benchmark(accepted_name)
    if family == "drone_rl":
        for card in ["source_pyflyt_waypoints", "source_mavsdk_python", "source_pyflyt", "source_gym_pybullet_drones"]:
            if card != current_card and card_is_open(card, ledger):
                return card
        return "source_pyflyt_waypoints" if current_card != "source_pyflyt_waypoints" else "source_mavsdk_python"
    if family == "minecraft_rl":
        for card in ["source_craftax", "source_minedojo", "source_malmo", "source_crafter", "source_voyager_minecraft"]:
            if card != current_card and card_is_open(card, ledger):
                return card
        return "source_craftax" if current_card != "source_craftax" else "source_minedojo"
    if family == "coding_local_sandbox":
        for card in cards_after(CODING_PRESSURE_CARD_ORDER, current_card or "source_bigcodebench"):
            if card != current_card and card_is_open(card, ledger):
                return card
        return "source_bigcodebench" if current_card != "source_bigcodebench" else "source_evalplus"
    if family == "web_agent_local":
        return "source_webarena"
    if family == "transfer_eval":
        return "transfer_eval_suite"
    return curriculum_card


def card_is_open(card_id: str, ledger: Any) -> bool:
    if not isinstance(ledger, list):
        return True
    marker = card_id.replace("source_", "")
    for row in ledger:
        if not isinstance(row, dict):
            continue
        name = str(row.get("benchmark_name") or "")
        report = str(row.get("best_report") or "")
        if marker in name or marker in report:
            return row.get("lifecycle") != "regression"
    return True


def cards_after(cards: list[str], current_card: str) -> list[str]:
    if current_card not in cards:
        return list(cards)
    index = cards.index(current_card)
    return cards[index + 1 :] + cards[:index]


def match_frontier_row(ledger: Any, active_report: str, family_hint: str) -> dict[str, Any]:
    rows = ledger if isinstance(ledger, list) else []
    normalized = normalize_path(active_report)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if normalize_path(row.get("best_report") or "") == normalized:
            return row
    suite = suite_from_report(active_report)
    for row in rows:
        if isinstance(row, dict) and suite and row.get("benchmark_name") == suite:
            return row
    for row in rows:
        if isinstance(row, dict) and family_hint and str(row.get("benchmark_name", "")).startswith(family_hint):
            return row
    return {}


def suite_from_report(path_text: str) -> str:
    if not path_text:
        return ""
    path = ROOT / path_text
    payload = read_json(path)
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if isinstance(summary, dict):
        return str(summary.get("suite") or "")
    return ""


def family_from_benchmark(name: str) -> str:
    if name.startswith("minecraft_rl_") or name.startswith("minecraft_"):
        return "minecraft_rl"
    if name.startswith("drone_rl_"):
        return "drone_rl"
    if name.startswith("coding_"):
        return "coding_local_sandbox"
    if name.startswith("web_agent_"):
        return "web_agent_local"
    if name.startswith("transfer_") or name == "asi_transfer_suite":
        return "transfer_eval"
    if name.startswith("ocean-"):
        return "rl_local"
    if "babylm" in name:
        return "babylm_mutated"
    return "general"


def card_from_benchmark(name: str) -> str:
    if name.startswith("minecraft_rl_source_"):
        return name.removeprefix("minecraft_rl_")
    if name.startswith("drone_rl_source_"):
        return name.removeprefix("drone_rl_")
    if name.startswith("drone_control_source_"):
        return name.removeprefix("drone_control_")
    if name.startswith("coding_agent_source_"):
        return name.removeprefix("coding_agent_")
    if name.startswith("coding_source_"):
        return name.removeprefix("coding_")
    if name.startswith("web_agent_source_"):
        return name.removeprefix("web_agent_")
    return ""


def compact_checks(checks: Any) -> list[dict[str, Any]]:
    rows = []
    for item in checks or []:
        if not isinstance(item, dict):
            continue
        rows.append({"gate": item.get("gate"), "passed": item.get("passed"), "evidence": item.get("evidence")})
    return rows


def upsert_by_report(rows: list[dict[str, Any]], accepted: dict[str, Any]) -> list[dict[str, Any]]:
    report = accepted.get("active_frontier_report")
    output = [row for row in rows if row.get("active_frontier_report") != report]
    output.append(accepted)
    return output[-200:]


def upsert_override(rows: Any, override: dict[str, Any]) -> list[dict[str, Any]]:
    rows = rows if isinstance(rows, list) else []
    key = (override.get("benchmark_name"), override.get("report"))
    output = [row for row in rows if (row.get("benchmark_name"), row.get("report")) != key]
    output.append(override)
    return output[-500:]


def ensure_registry(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("policy", "project_theseus_accepted_candidate_registry_v0")
    payload.setdefault("accepted_candidates", [])
    return payload


def ensure_overrides(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("policy", "project_theseus_benchmark_lifecycle_overrides_v0")
    payload.setdefault("promotions", [])
    return payload


def seed_from_text(value: str) -> int:
    import re

    match = re.search(r"seed(\d+)", value)
    return int(match.group(1)) if match else 1


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
