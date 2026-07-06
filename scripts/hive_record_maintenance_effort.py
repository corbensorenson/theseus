#!/usr/bin/env python3
"""Record measured maintenance effort rows for Circle/ordinary comparisons.

This script is intentionally narrow: it writes aggregate-safe measurement rows
only when an operator supplies measured human-edit minutes or a measured time
window. It does not infer human time from runtime, logs, or model activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "reports" / "hive_maintenance_effort_ledger.jsonl"
MAINTENANCE_MODES = {"object_only", "circle_seed_rule_rebuild"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maintenance-mode", choices=sorted(MAINTENANCE_MODES), required=False)
    parser.add_argument("--human-edit-minutes", type=float)
    parser.add_argument("--started-utc", default="")
    parser.add_argument("--ended-utc", default="")
    parser.add_argument("--review-step-count", type=int, default=1)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--evidence-path", action="append", default=[])
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER.relative_to(ROOT)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    mode = normalize_maintenance_mode(args.maintenance_mode)
    if not mode:
        raise SystemExit("--maintenance-mode must be object_only or circle_seed_rule_rebuild")
    minutes, basis = measured_minutes(args.human_edit_minutes, args.started_utc, args.ended_utc)
    row = effort_row(
        maintenance_mode=mode,
        human_edit_minutes=minutes,
        human_edit_minutes_basis=basis,
        review_step_count=max(1, int(args.review_step_count)),
        task_id=args.task_id,
        workload_id=args.workload_id,
        summary=args.summary,
        evidence_paths=args.evidence_path,
    )
    if args.dry_run:
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0
    ledger = safe_ledger_path(Path(args.ledger))
    append_jsonl(ledger, row)
    print(json.dumps({"ok": True, "ledger": display_path(ledger), "maintenance_mode": mode}, indent=2))
    return 0


def effort_row(
    *,
    maintenance_mode: str,
    human_edit_minutes: float,
    human_edit_minutes_basis: str,
    review_step_count: int,
    task_id: str,
    workload_id: str,
    summary: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    summary_text = summary.strip()
    evidence = [path for path in evidence_paths if path]
    return {
        "created_utc": now(),
        "policy": "project_theseus_hive_maintenance_effort_measurement_v0",
        "ok": True,
        "success": True,
        "task_id": task_id,
        "workload_id": workload_id,
        "maintenance_mode": maintenance_mode,
        "maintenance_mode_basis": "operator_supplied_explicit_label",
        "human_edit_minutes": round(float(human_edit_minutes), 6),
        "human_edit_minutes_measured": True,
        "human_edit_minutes_basis": human_edit_minutes_basis,
        "review_step_count": max(1, int(review_step_count)),
        "review_step_basis": "operator_counted_maintenance_review_steps",
        "summary_present": bool(summary_text),
        "summary_hash": stable_hash(summary_text)[:16] if summary_text else "",
        "evidence_count": len(evidence),
        "evidence_path_hashes": [stable_hash(path)[:16] for path in evidence],
        "runtime_ms": None,
        "external_inference_calls": 0,
        "training_mutation": False,
        "promotion_evidence": False,
        "private_content_exported": False,
    }


def measured_minutes(explicit_minutes: float | None, started_utc: str, ended_utc: str) -> tuple[float, str]:
    if explicit_minutes is not None:
        if explicit_minutes < 0:
            raise SystemExit("--human-edit-minutes must be nonnegative")
        return float(explicit_minutes), "operator_supplied_minutes"
    if bool(started_utc) != bool(ended_utc):
        raise SystemExit("--started-utc and --ended-utc must be provided together")
    if started_utc and ended_utc:
        started = parse_utc(started_utc)
        ended = parse_utc(ended_utc)
        seconds = (ended - started).total_seconds()
        if seconds < 0:
            raise SystemExit("--ended-utc must be after --started-utc")
        return seconds / 60.0, "operator_supplied_time_window"
    raise SystemExit("provide --human-edit-minutes or both --started-utc and --ended-utc")


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_maintenance_mode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ordinary": "object_only",
        "ordinary_current": "object_only",
        "baseline": "object_only",
        "object": "object_only",
        "object_only": "object_only",
        "circle": "circle_seed_rule_rebuild",
        "circle_seed_rule": "circle_seed_rule_rebuild",
        "circle_seed_rule_rebuild": "circle_seed_rule_rebuild",
        "seed_rule_rebuild": "circle_seed_rule_rebuild",
    }
    return aliases.get(text, "")


def safe_ledger_path(path: Path) -> Path:
    full = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    reports = (ROOT / "reports").resolve()
    try:
        full.relative_to(reports)
    except ValueError as exc:
        raise SystemExit("maintenance effort ledger must be written under reports/") from exc
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "reports" / "hive_maintenance_effort_ledger.jsonl"
        row = effort_row(
            maintenance_mode="circle_seed_rule_rebuild",
            human_edit_minutes=2.5,
            human_edit_minutes_basis="operator_supplied_minutes",
            review_step_count=3,
            task_id="self_test",
            workload_id="circle_seed_rule_update_cycle_effort_v1",
            summary="self-test measured maintenance row",
            evidence_paths=["reports/example.json"],
        )
        append_jsonl(ledger, row)
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["human_edit_minutes_measured"] is True
        assert rows[0]["maintenance_mode"] == "circle_seed_rule_rebuild"
        assert rows[0]["external_inference_calls"] == 0
        assert rows[0]["promotion_evidence"] is False
    print(json.dumps({"self_test": "passed"}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
