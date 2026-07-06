"""Classify blocked benchmark adapter cards into safe next setup work."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", default="reports/benchmark_adapter_factory.json")
    parser.add_argument("--out", default="reports/benchmark_pantry_unblocker.json")
    args = parser.parse_args()

    factory = read_json(ROOT / args.factory)
    cards = factory.get("cards") if isinstance(factory.get("cards"), list) else []
    blocked = [card for card in cards if isinstance(card, dict) and str(card.get("status", "")).startswith("blocked")]
    actions = [classify(card) for card in blocked]
    safe_actions = [row for row in actions if row.get("autonomous_safe")]
    report = {
        "policy": "project_theseus_benchmark_pantry_unblocker_v0",
        "created_utc": now(),
        "summary": {
            "blocked_cards": len(blocked),
            "autonomous_safe_actions": len(safe_actions),
            "waiting_on_private_assets": sum(1 for row in actions if row.get("category") == "waiting_on_private_asset"),
            "license_or_terms_blocked": sum(1 for row in actions if row.get("category") == "license_or_terms_blocked"),
            "runtime_dependency_work": sum(1 for row in actions if row.get("category") == "runtime_dependency_work"),
        },
        "actions": actions[:80],
        "next_actions": [row["action"] for row in safe_actions[:8]],
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def classify(card: dict[str, Any]) -> dict[str, Any]:
    card_id = str(card.get("id") or "")
    status = str(card.get("status") or "")
    next_step = str(card.get("next_step") or "")
    license_allowed = bool(card.get("license_allowed"))
    category = "adapter_smoke_work"
    autonomous_safe = license_allowed
    action = next_step or f"Run adapter smoke for {card_id}."
    if not license_allowed and "awaiting_user_asset" in str(card.get("decision") or ""):
        category = "waiting_on_private_asset"
        autonomous_safe = False
        action = f"Wait for user-owned asset for {card_id}; do not download replacements."
    elif not license_allowed:
        category = "license_or_terms_blocked"
        autonomous_safe = False
        action = f"Keep {card_id} excluded from active use until license/terms are explicit."
    elif card_id in {"source_terminal_bench", "source_openhands", "source_swe_rex", "source_swe_smith", "source_swe_atlas"} and "dependency" in status:
        category = "runtime_dependency_work"
        autonomous_safe = True
        action = "After Windows reboots, run scripts/setup_podman_sandbox_windows.ps1, then rerun benchmark_adapter_smoke for sandbox-gated coding-agent cards."
    elif "dependency" in status or "Install" in next_step or "build" in next_step:
        category = "runtime_dependency_work"
        autonomous_safe = True
        action = next_step or f"Install or configure runtime dependency for {card_id}."
    return {
        "card_id": card_id,
        "status": status,
        "category": category,
        "autonomous_safe": autonomous_safe,
        "action": action,
        "runner_family": card.get("runner_family"),
        "risk_tier": card.get("risk_tier"),
    }


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
