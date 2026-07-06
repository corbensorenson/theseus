#!/usr/bin/env python3
"""Run matched multi-seed neural seed token-decoder smokes.

This wrapper keeps each seed as a normal comparator run, then runs the semantic
plan audit over that seed's candidate manifest and aggregates only evidence.
It does not train on public data, call a teacher, distill, promote, or enable
fallback terminal returns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "neural_seed_token_decoder_comparator.json"
DEFAULT_OUT = ROOT / "reports" / "neural_seed_token_decoder_multiseed_smoke.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "neural_seed_token_decoder_multiseed_smoke.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--seeds", default="23,29,31,37,41")
    parser.add_argument("--artifact-prefix", default="neural_seed_token_decoder")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    seeds = parse_seeds(args.seeds)
    if not args.execute:
        report = planned_report(args.config, seeds, started)
    else:
        report = run_multiseed(config, config_path, seeds, started, artifact_prefix=safe_artifact_prefix(args.artifact_prefix))
    write_json(resolve(args.out), report)
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(report, indent=2))
    return 2 if report.get("trigger_state") == "RED" else 0


def run_multiseed(
    config: dict[str, Any],
    config_path: Path,
    seeds: list[int],
    started: float,
    *,
    artifact_prefix: str,
) -> dict[str, Any]:
    seed_rows = []
    for seed in seeds:
        seed_config = json.loads(json.dumps(config))
        seed_config.setdefault("matched_budget", {})["seeds"] = [seed]
        with tempfile.NamedTemporaryFile("w", suffix=f"-theseus-token-decoder-seed-{seed}.json", delete=False) as handle:
            json.dump(seed_config, handle, indent=2)
            handle.write("\n")
            temp_config = Path(handle.name)
        comparator_out = ROOT / "reports" / f"{artifact_prefix}_comparator_seed_{seed}.json"
        comparator_md = ROOT / "reports" / f"{artifact_prefix}_comparator_seed_{seed}.md"
        candidates_out = ROOT / "reports" / f"{artifact_prefix}_candidates_seed_{seed}.jsonl"
        audit_out = ROOT / "reports" / f"{artifact_prefix}_semantic_plan_gap_audit_seed_{seed}.json"
        audit_md = ROOT / "reports" / f"{artifact_prefix}_semantic_plan_gap_audit_seed_{seed}.md"
        comparator_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "neural_seed_token_decoder_comparator.py"),
            "--config",
            str(temp_config),
            "--out",
            str(comparator_out),
            "--markdown-out",
            str(comparator_md),
            "--candidate-manifest-out",
            str(candidates_out),
            "--execute",
        ]
        comparator = run_command(comparator_cmd)
        comparator_report = read_json(comparator_out)
        audit = {"returncode": None, "stdout_tail": "", "stderr_tail": ""}
        audit_report = {}
        if comparator["returncode"] == 0 and comparator_report:
            audit_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "neural_seed_semantic_plan_gap_audit.py"),
                "--config",
                str(temp_config),
                "--candidate-manifest",
                str(candidates_out),
                "--out",
                str(audit_out),
                "--markdown-out",
                str(audit_md),
                "--seed",
                str(seed),
            ]
            audit = run_command(audit_cmd)
            audit_report = read_json(audit_out)
        seed_rows.append(seed_summary(seed, comparator, comparator_report, audit, audit_report, comparator_out, candidates_out, audit_out))
        temp_config.unlink(missing_ok=True)
    summary = aggregate(seed_rows)
    gates = build_gates(seed_rows, summary, len(seeds))
    trigger = "GREEN" if all(row["passed"] for row in gates if row["severity"] == "hard") else "RED"
    if trigger == "GREEN" and not summary.get("symliquid_gap_closed"):
        trigger = "YELLOW"
    return {
        "policy": "project_theseus_neural_seed_token_decoder_multiseed_smoke_v0",
        "created_utc": now(),
        "trigger_state": trigger,
        "config": rel(config_path),
        "seeds": seeds,
        "artifact_prefix": artifact_prefix,
        "summary": summary,
        "seed_rows": seed_rows,
        "gates": gates,
        "score_semantics": (
            "Multi-seed private token-decoder smoke over matched SymLiquid and transformer arms. Each seed is "
            "a normal comparator run plus semantic-plan diagnostic audit. No public calibration, teacher call, "
            "distillation, promotion, external inference, or fallback terminal returns are allowed."
        ),
        "runtime_ms": int((time.perf_counter() - started) * 1000),
        "external_inference_calls": 0,
    }


def seed_summary(
    seed: int,
    comparator: dict[str, Any],
    report: dict[str, Any],
    audit: dict[str, Any],
    audit_report: dict[str, Any],
    comparator_out: Path,
    candidates_out: Path,
    audit_out: Path,
) -> dict[str, Any]:
    comparisons = dict_or_empty(report.get("comparisons"))
    by_arm = dict_or_empty(comparisons.get("by_arm"))
    audit_summary = dict_or_empty(audit_report.get("summary"))
    audit_by_arm = dict_or_empty(audit_report.get("by_arm"))
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    audit_gates = audit_report.get("gates") if isinstance(audit_report.get("gates"), list) else []
    return {
        "seed": seed,
        "comparator_returncode": comparator.get("returncode"),
        "audit_returncode": audit.get("returncode"),
        "comparator_trigger_state": report.get("trigger_state"),
        "audit_trigger_state": audit_report.get("trigger_state"),
        "comparator_report": rel(comparator_out),
        "candidate_manifest": rel(candidates_out),
        "semantic_plan_audit": rel(audit_out),
        "symliquid_sts_on_verifier_pass_rate": get_path(by_arm, ["symliquid_style", "sts_on_verifier_pass_rate"]),
        "transformer_sts_on_verifier_pass_rate": get_path(by_arm, ["transformer_control", "sts_on_verifier_pass_rate"]),
        "symliquid_minus_transformer_sts_on_verifier_pass_rate": comparisons.get("symliquid_minus_transformer_sts_on_verifier_pass_rate"),
        "symliquid_fallback_rate": get_path(by_arm, ["symliquid_style", "grammar_repair_fallback_rate_sts_on"]),
        "transformer_fallback_rate": get_path(by_arm, ["transformer_control", "grammar_repair_fallback_rate_sts_on"]),
        "symliquid_semantic_plan_supported_rate": get_path(by_arm, ["symliquid_style", "semantic_plan_supported_rate_sts_on"]),
        "transformer_semantic_plan_supported_rate": get_path(by_arm, ["transformer_control", "semantic_plan_supported_rate_sts_on"]),
        "trusted_parameter_match": get_path(report, ["summary", "trusted_parameter_match"]),
        "parameter_match_delta": get_path(report, ["summary", "parameter_match_delta"]),
        "external_inference_calls": get_path(report, ["summary", "external_inference_calls"], 0),
        "teacher_used": get_path(report, ["summary", "teacher_used"], False),
        "public_training_rows": get_path(report, ["summary", "public_training_rows"], 0),
        "model_promotion_allowed": get_path(report, ["summary", "model_promotion_allowed"], False),
        "audit_bottleneck": get_path(audit_summary, ["bottleneck", "label"]),
        "symliquid_expected_plan_match_rate": audit_summary.get("symliquid_private_eval_plan_match_rate"),
        "transformer_expected_plan_match_rate": audit_summary.get("transformer_private_eval_plan_match_rate"),
        "symliquid_visible_contract_semantic_beam_selected_rate": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "visible_contract_semantic_beam_selected_rate"],
        ),
        "transformer_visible_contract_semantic_beam_selected_rate": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "visible_contract_semantic_beam_selected_rate"],
        ),
        "symliquid_visible_contract_semantic_beam_available_rate": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "visible_contract_semantic_beam_available_rate"],
        ),
        "transformer_visible_contract_semantic_beam_available_rate": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "visible_contract_semantic_beam_available_rate"],
        ),
        "symliquid_learned_internal_semantic_route_selected_rate": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "learned_internal_semantic_route_selected_rate"],
        ),
        "transformer_learned_internal_semantic_route_selected_rate": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "learned_internal_semantic_route_selected_rate"],
        ),
        "symliquid_learned_internal_semantic_route_available_rate": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "learned_internal_semantic_route_available_rate"],
        ),
        "transformer_learned_internal_semantic_route_available_rate": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "learned_internal_semantic_route_available_rate"],
        ),
        "symliquid_learned_internal_semantic_route_strategy_selected_rates": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "learned_internal_semantic_route_strategy_selected_rates"],
            {},
        ),
        "transformer_learned_internal_semantic_route_strategy_selected_rates": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "learned_internal_semantic_route_strategy_selected_rates"],
            {},
        ),
        "symliquid_learned_internal_semantic_route_strategy_available_rates": get_path(
            audit_by_arm,
            ["symliquid_style", "by_phase", "private_eval", "learned_internal_semantic_route_strategy_available_rates"],
            {},
        ),
        "transformer_learned_internal_semantic_route_strategy_available_rates": get_path(
            audit_by_arm,
            ["transformer_control", "by_phase", "private_eval", "learned_internal_semantic_route_strategy_available_rates"],
            {},
        ),
        "hard_gate_failures": [row for row in gates if row.get("severity") == "hard" and not row.get("passed")],
        "audit_hard_gate_failures": [row for row in audit_gates if row.get("severity") == "hard" and not row.get("passed")],
        "comparator_stdout_tail": comparator.get("stdout_tail"),
        "comparator_stderr_tail": comparator.get("stderr_tail"),
        "audit_stdout_tail": audit.get("stdout_tail"),
        "audit_stderr_tail": audit.get("stderr_tail"),
    }


def aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sym_rates = numbers(row.get("symliquid_sts_on_verifier_pass_rate") for row in seed_rows)
    tx_rates = numbers(row.get("transformer_sts_on_verifier_pass_rate") for row in seed_rows)
    gaps = numbers(row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate") for row in seed_rows)
    sym_plan = numbers(row.get("symliquid_expected_plan_match_rate") for row in seed_rows)
    tx_plan = numbers(row.get("transformer_expected_plan_match_rate") for row in seed_rows)
    sym_contract_beams = numbers(row.get("symliquid_visible_contract_semantic_beam_selected_rate") for row in seed_rows)
    tx_contract_beams = numbers(row.get("transformer_visible_contract_semantic_beam_selected_rate") for row in seed_rows)
    sym_contract_available = numbers(row.get("symliquid_visible_contract_semantic_beam_available_rate") for row in seed_rows)
    tx_contract_available = numbers(row.get("transformer_visible_contract_semantic_beam_available_rate") for row in seed_rows)
    sym_learned_routes = numbers(row.get("symliquid_learned_internal_semantic_route_selected_rate") for row in seed_rows)
    tx_learned_routes = numbers(row.get("transformer_learned_internal_semantic_route_selected_rate") for row in seed_rows)
    sym_learned_available = numbers(row.get("symliquid_learned_internal_semantic_route_available_rate") for row in seed_rows)
    tx_learned_available = numbers(row.get("transformer_learned_internal_semantic_route_available_rate") for row in seed_rows)
    sym_strategy_selected = mean_rate_maps(
        row.get("symliquid_learned_internal_semantic_route_strategy_selected_rates") for row in seed_rows
    )
    tx_strategy_selected = mean_rate_maps(
        row.get("transformer_learned_internal_semantic_route_strategy_selected_rates") for row in seed_rows
    )
    sym_strategy_available = mean_rate_maps(
        row.get("symliquid_learned_internal_semantic_route_strategy_available_rates") for row in seed_rows
    )
    tx_strategy_available = mean_rate_maps(
        row.get("transformer_learned_internal_semantic_route_strategy_available_rates") for row in seed_rows
    )
    mean_gap = mean(gaps)
    mean_plan_gap = None
    if sym_plan and tx_plan and len(sym_plan) == len(tx_plan):
        mean_plan_gap = round(mean(sym_plan) - mean(tx_plan), 6)
    if mean_gap is not None and mean_gap >= -0.1:
        bottleneck = "symliquid_gap_mostly_closed"
    elif mean_plan_gap is not None and mean_plan_gap < -0.1:
        bottleneck = "symliquid_semantic_plan_selection"
    else:
        bottleneck = "shared_renderer_or_behavior_semantics_after_plan_selection"
    return {
        "completed_seed_count": sum(1 for row in seed_rows if row.get("comparator_returncode") == 0 and row.get("audit_returncode") == 0),
        "requested_seed_count": len(seed_rows),
        "symliquid_sts_on_mean": mean(sym_rates),
        "symliquid_sts_on_stdev": stdev(sym_rates),
        "transformer_sts_on_mean": mean(tx_rates),
        "transformer_sts_on_stdev": stdev(tx_rates),
        "symliquid_minus_transformer_sts_on_mean": mean_gap,
        "symliquid_minus_transformer_sts_on_stdev": stdev(gaps),
        "symliquid_expected_plan_match_mean": mean(sym_plan),
        "transformer_expected_plan_match_mean": mean(tx_plan),
        "symliquid_minus_transformer_expected_plan_match_mean": mean_plan_gap,
        "symliquid_visible_contract_semantic_beam_selected_mean": mean(sym_contract_beams),
        "transformer_visible_contract_semantic_beam_selected_mean": mean(tx_contract_beams),
        "symliquid_visible_contract_semantic_beam_available_mean": mean(sym_contract_available),
        "transformer_visible_contract_semantic_beam_available_mean": mean(tx_contract_available),
        "symliquid_learned_internal_semantic_route_selected_mean": mean(sym_learned_routes),
        "transformer_learned_internal_semantic_route_selected_mean": mean(tx_learned_routes),
        "symliquid_learned_internal_semantic_route_available_mean": mean(sym_learned_available),
        "transformer_learned_internal_semantic_route_available_mean": mean(tx_learned_available),
        "symliquid_learned_internal_semantic_route_strategy_selected_means": sym_strategy_selected,
        "transformer_learned_internal_semantic_route_strategy_selected_means": tx_strategy_selected,
        "symliquid_learned_internal_semantic_route_strategy_available_means": sym_strategy_available,
        "transformer_learned_internal_semantic_route_strategy_available_means": tx_strategy_available,
        "winner_counts": {
            "symliquid_style": sum(
                1
                for row in seed_rows
                if row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate") is not None
                and float(row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate")) > 0.0
            ),
            "transformer_control": sum(
                1
                for row in seed_rows
                if row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate") is not None
                and float(row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate")) < 0.0
            ),
            "ties": sum(
                1
                for row in seed_rows
                if row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate") is not None
                and float(row.get("symliquid_minus_transformer_sts_on_verifier_pass_rate")) == 0.0
            ),
        },
        "symliquid_gap_closed": bool(mean_gap is not None and mean_gap >= -0.1),
        "bottleneck": bottleneck,
        "external_inference_calls": sum(int(row.get("external_inference_calls") or 0) for row in seed_rows),
        "teacher_used": any(bool(row.get("teacher_used")) for row in seed_rows),
        "public_training_rows": sum(int(row.get("public_training_rows") or 0) for row in seed_rows),
        "model_promotion_allowed": any(bool(row.get("model_promotion_allowed")) for row in seed_rows),
    }


def build_gates(seed_rows: list[dict[str, Any]], summary: dict[str, Any], requested_seed_count: int) -> list[dict[str, Any]]:
    return [
        gate("at_least_five_seeds_requested", requested_seed_count >= 5, {"requested_seed_count": requested_seed_count}, "hard"),
        gate("all_seed_runs_completed", summary.get("completed_seed_count") == requested_seed_count, {"completed": summary.get("completed_seed_count"), "requested": requested_seed_count}, "hard"),
        gate("all_comparator_gates_green", all(not row.get("hard_gate_failures") for row in seed_rows), "see seed_rows.hard_gate_failures", "hard"),
        gate("all_audit_gates_green", all(not row.get("audit_hard_gate_failures") for row in seed_rows), "see seed_rows.audit_hard_gate_failures", "hard"),
        gate("fallback_return_rate_zero_all_seeds", all(float(row.get("symliquid_fallback_rate") or 0.0) == 0.0 and float(row.get("transformer_fallback_rate") or 0.0) == 0.0 for row in seed_rows), "fallback rates per seed", "hard"),
        gate("semantic_plan_supported_all_seeds", all(float(row.get("symliquid_semantic_plan_supported_rate") or 0.0) > 0.0 and float(row.get("transformer_semantic_plan_supported_rate") or 0.0) > 0.0 for row in seed_rows), "semantic plan support per seed", "hard"),
        gate("trusted_parameter_match_all_seeds", all(bool(row.get("trusted_parameter_match")) for row in seed_rows), "parameter_match_delta per seed", "hard"),
        gate("external_inference_zero", int(summary.get("external_inference_calls") or 0) == 0, summary.get("external_inference_calls"), "hard"),
        gate("teacher_public_promotion_locked", not summary.get("teacher_used") and int(summary.get("public_training_rows") or 0) == 0 and not summary.get("model_promotion_allowed"), summary, "hard"),
        gate("symliquid_gap_closed_or_bottleneck_proven", bool(summary.get("symliquid_gap_closed")) or bool(summary.get("bottleneck")), summary.get("bottleneck"), "hard"),
    ]


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2400:],
        "stderr_tail": proc.stderr[-2400:],
    }


def planned_report(config: str, seeds: list[int], started: float) -> dict[str, Any]:
    return {
        "policy": "project_theseus_neural_seed_token_decoder_multiseed_smoke_v0",
        "created_utc": now(),
        "trigger_state": "PLANNED",
        "config": config,
        "seeds": seeds,
        "summary": {
            "execute_required": True,
            "command": "python3 scripts/neural_seed_token_decoder_multiseed_smoke.py --execute",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def parse_seeds(value: str) -> list[int]:
    seeds = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [23, 29, 31, 37, 41]


def safe_artifact_prefix(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip())
    return cleaned.strip("_") or "neural_seed_token_decoder"


def numbers(values: Any) -> list[float]:
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def stdev(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0 if values else None


def mean_rate_maps(values: Any) -> dict[str, float | None]:
    keyed: dict[str, list[float]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, raw in value.items():
            try:
                keyed.setdefault(str(key), []).append(float(raw))
            except (TypeError, ValueError):
                pass
    return {key: mean(rows) for key, rows in sorted(keyed.items())}


def dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def gate(name: str, passed: bool, evidence: Any, severity: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict_or_empty(report.get("summary"))
    lines = [
        "# Neural Seed Token Decoder Multi-Seed Smoke",
        "",
        f"- trigger_state: `{report.get('trigger_state')}`",
        f"- seeds: `{report.get('seeds')}`",
        f"- completed_seed_count: `{summary.get('completed_seed_count')}`",
        f"- symliquid_sts_on_mean: `{summary.get('symliquid_sts_on_mean')}`",
        f"- transformer_sts_on_mean: `{summary.get('transformer_sts_on_mean')}`",
        f"- symliquid_minus_transformer_sts_on_mean: `{summary.get('symliquid_minus_transformer_sts_on_mean')}`",
        f"- symliquid_expected_plan_match_mean: `{summary.get('symliquid_expected_plan_match_mean')}`",
        f"- transformer_expected_plan_match_mean: `{summary.get('transformer_expected_plan_match_mean')}`",
        f"- symliquid_visible_contract_semantic_beam_selected_mean: "
        f"`{summary.get('symliquid_visible_contract_semantic_beam_selected_mean')}`",
        f"- transformer_visible_contract_semantic_beam_selected_mean: "
        f"`{summary.get('transformer_visible_contract_semantic_beam_selected_mean')}`",
        f"- symliquid_learned_internal_semantic_route_selected_mean: "
        f"`{summary.get('symliquid_learned_internal_semantic_route_selected_mean')}`",
        f"- transformer_learned_internal_semantic_route_selected_mean: "
        f"`{summary.get('transformer_learned_internal_semantic_route_selected_mean')}`",
        f"- symliquid_learned_internal_semantic_route_strategy_selected_means: "
        f"`{summary.get('symliquid_learned_internal_semantic_route_strategy_selected_means')}`",
        f"- transformer_learned_internal_semantic_route_strategy_selected_means: "
        f"`{summary.get('transformer_learned_internal_semantic_route_strategy_selected_means')}`",
        f"- winner_counts: `{summary.get('winner_counts')}`",
        f"- bottleneck: `{summary.get('bottleneck')}`",
        "",
        "## Seed Rows",
        "",
    ]
    for row in report.get("seed_rows", []):
        lines.append(
            f"- seed `{row.get('seed')}`: sym=`{row.get('symliquid_sts_on_verifier_pass_rate')}`, "
            f"tx=`{row.get('transformer_sts_on_verifier_pass_rate')}`, "
            f"gap=`{row.get('symliquid_minus_transformer_sts_on_verifier_pass_rate')}`, "
            f"sym_contract_beam=`{row.get('symliquid_visible_contract_semantic_beam_selected_rate')}`, "
            f"tx_contract_beam=`{row.get('transformer_visible_contract_semantic_beam_selected_rate')}`, "
            f"sym_learned_route=`{row.get('symliquid_learned_internal_semantic_route_selected_rate')}`, "
            f"tx_learned_route=`{row.get('transformer_learned_internal_semantic_route_selected_rate')}`, "
            f"sym_route_strategies=`{row.get('symliquid_learned_internal_semantic_route_strategy_selected_rates')}`, "
            f"tx_route_strategies=`{row.get('transformer_learned_internal_semantic_route_strategy_selected_rates')}`, "
            f"bottleneck=`{row.get('audit_bottleneck')}`"
        )
    lines.extend(["", "## Gates", ""])
    for row in report.get("gates", []):
        lines.append(f"- `{row.get('name')}`: passed=`{row.get('passed')}` severity=`{row.get('severity')}`")
    lines.extend(["", "## Semantics", "", str(report.get("score_semantics", "")), ""])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
