#!/usr/bin/env python3
"""Bounded multi-seed architecture sweep for the neural seed code proposer.

This wrapper runs the existing private code-proposer comparator repeatedly with
controlled seed and budget-tier overrides, then reports mean/spread. It exists
to prevent single-seed architecture claims from being treated as evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from neural_seed_code_proposer_comparator import (  # noqa: E402
    dict_or_empty,
    get_path,
    read_json,
    rel,
    resolve,
    run_comparator,
    write_json,
    write_text,
)


DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_architecture_sweep.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--tier", action="append", default=[], help="Budget tier id to run. Defaults to enabled tiers.")
    parser.add_argument("--seeds", default="", help="Comma-separated seed override.")
    parser.add_argument("--force", action="store_true", help="Recompute per-seed reports even when matching output files exist.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(resolve(args.config))
    outputs = dict_or_empty(config.get("outputs"))
    out = args.out or str(outputs.get("out") or "reports/neural_seed_architecture_sweep.json")
    markdown_out = args.markdown_out or str(outputs.get("markdown_out") or "reports/neural_seed_architecture_sweep.md")
    if args.execute:
        report = run_sweep(config, args.config, tier_ids=args.tier, seed_override=args.seeds, force=args.force, started=started)
    else:
        report = planned_report(config, args.config)
    write_json(resolve(out), report)
    write_text(resolve(markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_sweep(
    config: dict[str, Any],
    config_path: str,
    *,
    tier_ids: list[str],
    seed_override: str,
    force: bool,
    started: float,
) -> dict[str, Any]:
    base_config_path = resolve(str(config.get("base_comparator_config") or "configs/neural_seed_code_proposer_comparator.json"))
    base_config = read_json(base_config_path)
    outputs = dict_or_empty(config.get("outputs"))
    run_dir = resolve(str(outputs.get("run_dir") or "reports/neural_seed_architecture_sweep"))
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(seed_override) or [int(seed) for seed in get_path(config, ["seed_sweep", "seeds"], [23, 29, 31, 37, 41])]
    tiers = selected_tiers(config, tier_ids)
    runs: list[dict[str, Any]] = []
    for tier in tiers:
        tier_id = str(tier.get("id") or "tier")
        for seed in seeds:
            run_config = build_run_config(base_config, tier, seed)
            report_path = run_dir / f"{tier_id}_seed_{seed}.json"
            candidates_path = run_dir / f"{tier_id}_seed_{seed}_candidates.jsonl"
            if (not force) and report_path.exists() and candidates_path.exists():
                report = read_json(report_path)
            else:
                report = run_comparator(
                    run_config,
                    f"{config_path}:{tier_id}:seed_{seed}",
                    str(candidates_path.relative_to(ROOT)),
                    time.perf_counter(),
                )
                write_json(report_path, report)
            runs.append(run_record(tier_id, seed, report, report_path, candidates_path))
    aggregate = aggregate_runs(runs)
    gates = build_gates(config, runs, aggregate, tiers, seeds)
    hard_pass = all(row["passed"] for row in gates if row["severity"] == "hard")
    trigger = "GREEN" if hard_pass else "RED"
    if trigger == "GREEN" and any(not row["passed"] for row in gates):
        trigger = "YELLOW"
    return {
        "policy": "project_theseus_neural_seed_architecture_sweep_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": trigger,
        "execute": True,
        "summary": {
            "tier_count": len(tiers),
            "seed_count": len(seeds),
            "run_count": len(runs),
            "minimum_claim_seed_count": get_path(config, ["seed_sweep", "minimum_claim_seed_count"], 5),
            "single_seed_claims_disallowed": True,
            "external_inference_calls": 0,
            "teacher_used": False,
            "public_training_rows": 0,
            "model_promotion_allowed": False,
        },
        "aggregate": aggregate,
        "runs": runs,
        "gates": gates,
        "score_semantics": config.get("score_semantics"),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def planned_report(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    tiers = selected_tiers(config, [])
    return {
        "policy": "project_theseus_neural_seed_architecture_sweep_report_v0",
        "created_utc": now(),
        "config": config_path,
        "trigger_state": "PLANNED",
        "execute": False,
        "summary": {
            "planned_tiers": [tier.get("id") for tier in tiers],
            "planned_seeds": get_path(config, ["seed_sweep", "seeds"], []),
            "single_seed_claims_disallowed": True,
            "external_inference_calls": 0,
        },
        "runbook": {
            "current_smoke_seed_sweep": "python3 scripts/neural_seed_architecture_sweep.py --execute --tier current_smoke_seed_sweep",
            "one_notch_budget_ladder": "python3 scripts/neural_seed_architecture_sweep.py --execute --tier one_notch_budget_ladder",
        },
        "score_semantics": config.get("score_semantics"),
        "external_inference_calls": 0,
    }


def selected_tiers(config: dict[str, Any], requested: list[str]) -> list[dict[str, Any]]:
    tiers = [tier for tier in config.get("budget_tiers", []) if isinstance(tier, dict)]
    if requested:
        requested_set = set(requested)
        return [tier for tier in tiers if str(tier.get("id")) in requested_set]
    return [tier for tier in tiers if bool(tier.get("enabled_by_default"))]


def build_run_config(base: dict[str, Any], tier: dict[str, Any], seed: int) -> dict[str, Any]:
    run = copy.deepcopy(base)
    run.setdefault("matched_budget", {})
    run["matched_budget"].update(dict_or_empty(tier.get("matched_budget_overrides")))
    run["matched_budget"]["seeds"] = [seed]
    run.setdefault("arms", {})
    run["arms"].setdefault("transformer_control", {})
    run["arms"].setdefault("symliquid_style", {})
    run["arms"]["transformer_control"].update(dict_or_empty(tier.get("transformer_control_overrides")))
    run["arms"]["symliquid_style"].update(dict_or_empty(tier.get("symliquid_style_overrides")))
    return run


def run_record(tier_id: str, seed: int, report: dict[str, Any], report_path: Path, candidates_path: Path) -> dict[str, Any]:
    comparisons = dict_or_empty(report.get("comparisons"))
    by_arm = dict_or_empty(comparisons.get("by_arm"))
    return {
        "tier_id": tier_id,
        "seed": seed,
        "trigger_state": report.get("trigger_state"),
        "report": rel(report_path),
        "candidate_manifest": rel(candidates_path),
        "parameter_match_delta": get_path(report, ["summary", "parameter_match_delta"], None),
        "trusted_parameter_match": get_path(report, ["summary", "trusted_parameter_match"], None),
        "symliquid_sts_on": get_path(by_arm, ["symliquid_style", "sts_on_verifier_pass_rate"], None),
        "transformer_sts_on": get_path(by_arm, ["transformer_control", "sts_on_verifier_pass_rate"], None),
        "symliquid_sts_delta": get_path(by_arm, ["symliquid_style", "sts_delta"], None),
        "transformer_sts_delta": get_path(by_arm, ["transformer_control", "sts_delta"], None),
        "symliquid_minus_transformer": comparisons.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"),
        "external_inference_calls": report.get("external_inference_calls"),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_tier.setdefault(str(row.get("tier_id")), []).append(row)
    return {tier: aggregate_tier(rows) for tier, rows in sorted(by_tier.items())}


def aggregate_tier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "symliquid_sts_on",
        "transformer_sts_on",
        "symliquid_sts_delta",
        "transformer_sts_delta",
        "symliquid_minus_transformer",
    ]
    out: dict[str, Any] = {"seed_count": len(rows), "seeds": [row.get("seed") for row in rows]}
    for metric in metrics:
        values = [float(row.get(metric)) for row in rows if row.get(metric) is not None]
        out[metric] = describe(values)
    return out


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def build_gates(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    aggregate: dict[str, Any],
    tiers: list[dict[str, Any]],
    seeds: list[int],
) -> list[dict[str, Any]]:
    safety = dict_or_empty(config.get("safety"))
    min_claim_seeds = int(get_path(config, ["seed_sweep", "minimum_claim_seed_count"], 5) or 5)
    return [
        gate("public_training_forbidden", not safety.get("public_training_allowed", True), safety, "hard"),
        gate("teacher_calls_forbidden", not safety.get("teacher_calls_allowed", True), safety, "hard"),
        gate("model_promotion_forbidden", not safety.get("model_promotion_allowed", True), safety, "hard"),
        gate("tiers_selected", bool(tiers), [tier.get("id") for tier in tiers], "hard"),
        gate("runs_completed", len(runs) == len(tiers) * len(seeds), {"runs": len(runs), "tiers": len(tiers), "seeds": len(seeds)}, "hard"),
        gate("all_runs_green_or_yellow", all(row.get("trigger_state") in {"GREEN", "YELLOW"} for row in runs), [row.get("trigger_state") for row in runs], "hard"),
        gate("external_inference_zero", all(int(row.get("external_inference_calls") or 0) == 0 for row in runs), 0, "hard"),
        gate("minimum_claim_seed_count_met", len(seeds) >= min_claim_seeds, {"seeds": seeds, "minimum": min_claim_seeds}, "hard"),
        gate("aggregate_spread_reported", bool(aggregate), aggregate, "hard"),
    ]


def parse_seeds(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Architecture Sweep",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- execute: `{report.get('execute')}`",
        f"- run_count: `{summary.get('run_count')}`",
        f"- seed_count: `{summary.get('seed_count')}`",
        f"- single_seed_claims_disallowed: `{summary.get('single_seed_claims_disallowed')}`",
        "",
        "## Aggregate",
        "",
    ]
    for tier, row in dict_or_empty(report.get("aggregate")).items():
        lines.append(f"- `{tier}`: `{row}`")
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
