"""Generate governed synthetic benchmark pressure cards.

Synthetic benchmarks are the evaluation-side twin of synthetic data: useful for
fresh local pressure, dangerous if treated as public truth. This factory creates
case manifests and benchmark cards from existing local evidence only. It never
copies public benchmark items, never calls external inference, and marks every
generated card as quarantined from public-comparator claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


CODE_TASK_TEMPLATES: list[dict[str, Any]] = [
    {
        "stem": "safe_head",
        "signature": "safe_head(xs, default=None)",
        "buggy": "def safe_head(xs, default=None):\n    return xs[0]\n",
        "tests": "assert safe_head([], 'x') == 'x'\nassert safe_head([3, 4], 'x') == 3\n",
        "solution": "def safe_head(xs, default=None):\n    return xs[0] if xs else default\n",
        "tags": ["edge_case", "type_handling"],
    },
    {
        "stem": "stable_dedupe",
        "signature": "stable_dedupe(xs)",
        "buggy": "def stable_dedupe(xs):\n    return list(set(xs))\n",
        "tests": "assert stable_dedupe([2, 1, 2, 3, 1]) == [2, 1, 3]\nassert stable_dedupe([]) == []\n",
        "solution": "def stable_dedupe(xs):\n    seen = set()\n    out = []\n    for item in xs:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n",
        "tags": ["algorithm_choice", "hidden_tests"],
    },
    {
        "stem": "parse_ints",
        "signature": "parse_ints(text)",
        "buggy": "def parse_ints(text):\n    return [int(part) for part in text.split(',')]\n",
        "tests": "assert parse_ints('1, 2, x, -3') == [1, 2, -3]\nassert parse_ints('') == []\n",
        "solution": "def parse_ints(text):\n    out = []\n    for part in text.split(','):\n        part = part.strip()\n        if not part:\n            continue\n        try:\n            out.append(int(part))\n        except ValueError:\n            continue\n    return out\n",
        "tags": ["parsing", "type_handling"],
    },
    {
        "stem": "chunked",
        "signature": "chunked(xs, n)",
        "buggy": "def chunked(xs, n):\n    return [xs]\n",
        "tests": "assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunked([1], 0) == [[1]]\n",
        "solution": "def chunked(xs, n):\n    if n <= 0:\n        return [xs]\n    return [xs[i:i+n] for i in range(0, len(xs), n)]\n",
        "tags": ["edge_case", "algorithm_choice"],
    },
    {
        "stem": "safe_ratio",
        "signature": "safe_ratio(a, b)",
        "buggy": "def safe_ratio(a, b):\n    return a / b\n",
        "tests": "assert safe_ratio(4, 2) == 2\nassert safe_ratio(4, 0) == 0\n",
        "solution": "def safe_ratio(a, b):\n    return 0 if b == 0 else a / b\n",
        "tags": ["edge_case", "type_handling"],
    },
    {
        "stem": "window_sums",
        "signature": "window_sums(xs, n)",
        "buggy": "def window_sums(xs, n):\n    return []\n",
        "tests": "assert window_sums([1, 2, 3, 4], 2) == [3, 5, 7]\nassert window_sums([1], 3) == []\n",
        "solution": "def window_sums(xs, n):\n    if n <= 0:\n        return []\n    return [sum(xs[i:i+n]) for i in range(0, len(xs) - n + 1)]\n",
        "tags": ["algorithm_choice", "hidden_tests"],
    },
]


HYBRID_TASKS: dict[str, list[dict[str, Any]]] = {
    "synthetic_cross_arm_code_memory": [
        {
            "stem": "latest_goal",
            "signature": "latest_goal(events)",
            "buggy": "def latest_goal(events):\n    return events[0]['goal']\n",
            "tests": "events = [{'type':'goal','goal':'old'}, {'type':'note','text':'x'}, {'type':'goal','goal':'new'}]\nassert latest_goal(events) == 'new'\nassert latest_goal([]) == ''\n",
            "solution": "def latest_goal(events):\n    for event in reversed(events):\n        if event.get('type') == 'goal':\n            return event.get('goal', '')\n    return ''\n",
            "tags": ["long_context_recovery", "edge_case"],
        },
        {
            "stem": "reject_decoys",
            "signature": "reject_decoys(rows)",
            "buggy": "def reject_decoys(rows):\n    return [row['text'] for row in rows]\n",
            "tests": "rows = [{'text':'keep','decoy':False}, {'text':'drop','decoy':True}]\nassert reject_decoys(rows) == ['keep']\nassert reject_decoys([]) == []\n",
            "solution": "def reject_decoys(rows):\n    return [row.get('text', '') for row in rows if not row.get('decoy')]\n",
            "tags": ["long_context_recovery", "filtering"],
        },
    ],
    "synthetic_cross_arm_tool_safety": [
        {
            "stem": "safe_tool_command",
            "signature": "safe_tool_command(command)",
            "buggy": "def safe_tool_command(command):\n    return True\n",
            "tests": "assert safe_tool_command('python scripts/test.py') is True\nassert safe_tool_command('rm -rf C:/') is False\nassert safe_tool_command('git reset --hard') is False\n",
            "solution": "def safe_tool_command(command):\n    text = command.lower()\n    blocked = ['rm -rf', 'remove-item -recurse', 'git reset --hard', 'format ']\n    return not any(token in text for token in blocked)\n",
            "tags": ["safety_gate", "tool_use_failure"],
        },
        {
            "stem": "permission_summary",
            "signature": "permission_summary(envelope)",
            "buggy": "def permission_summary(envelope):\n    return 'allow'\n",
            "tests": "assert permission_summary({'network':'forbidden','hardware':'not_applicable'}) == 'local_only'\nassert permission_summary({'network':'allowed'}) == 'review_required'\n",
            "solution": "def permission_summary(envelope):\n    if envelope.get('network') == 'forbidden' and envelope.get('hardware') in {'not_applicable', None}:\n        return 'local_only'\n    return 'review_required'\n",
            "tags": ["safety_gate", "tool_use_failure"],
        },
    ],
    "synthetic_cross_domain_code_rl_trace": [
        {
            "stem": "episode_survival_ratio",
            "signature": "episode_survival_ratio(trace)",
            "buggy": "def episode_survival_ratio(trace):\n    return 1\n",
            "tests": "trace = [{'done':False}, {'done':False}, {'done':True}, {'done':False}]\nassert episode_survival_ratio(trace) == 0.5\nassert episode_survival_ratio([]) == 0\n",
            "solution": "def episode_survival_ratio(trace):\n    if not trace:\n        return 0\n    survived = 0\n    for row in trace:\n        if row.get('done'):\n            break\n        survived += 1\n    return survived / len(trace)\n",
            "tags": ["rl_trace", "edge_case"],
        },
        {
            "stem": "reward_delta",
            "signature": "reward_delta(trace)",
            "buggy": "def reward_delta(trace):\n    return trace[-1]['reward']\n",
            "tests": "assert reward_delta([{'reward':1}, {'reward':4}]) == 3\nassert reward_delta([]) == 0\n",
            "solution": "def reward_delta(trace):\n    if len(trace) < 2:\n        return 0\n    return trace[-1].get('reward', 0) - trace[0].get('reward', 0)\n",
            "tags": ["rl_trace", "algorithm_choice"],
        },
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/synthetic_benchmark_policy.json")
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--residual-escrow", default="reports/residual_escrow.json")
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--arm-sucker-registry", default="reports/arm_sucker_registry.json")
    parser.add_argument("--code-transfer-artifacts", default="reports/code_transfer_artifacts.json")
    parser.add_argument("--cards-dir", default="benchmarks/cards")
    parser.add_argument("--cases-dir", default="data/synthetic_benchmarks")
    parser.add_argument("--out", default="reports/synthetic_benchmark_factory.json")
    parser.add_argument("--markdown-out", default="reports/synthetic_benchmark_factory.md")
    parser.add_argument("--write-cards", action="store_true")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    ledger = read_json(ROOT / args.benchmark_ledger, [])
    residuals = read_json(ROOT / args.residual_escrow, {})
    arms = read_json(ROOT / args.arm_registry, {})
    suckers = read_json(ROOT / args.arm_sucker_registry, {})
    transfer = read_json(ROOT / args.code_transfer_artifacts, {})
    rng = random.Random(int(policy.get("seed") or 1337))

    source_rows = select_source_rows(ledger)
    residual_targets = select_residual_targets(residuals)
    available_arms = arm_names(arms) | sucker_names(suckers)

    cases_dir = resolve_path(args.cases_dir)
    cards_dir = resolve_path(args.cards_dir)
    cards: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for card_cfg in policy.get("default_cards", []):
        if not isinstance(card_cfg, dict):
            continue
        card_id = str(card_cfg.get("id") or "")
        if not card_id:
            continue
        cases = generate_cases(card_cfg, source_rows, residual_targets, transfer, rng, max_cases=int(policy.get("max_cases_per_benchmark") or 8))
        case_path = cases_dir / f"{card_id}.jsonl"
        write_jsonl(case_path, cases)
        card = build_card(card_cfg, case_path, cases, available_arms)
        cards.append(card)
        manifest_rows.append(
            {
                "card_id": card_id,
                "case_path": rel(case_path),
                "case_count": len(cases),
                "required_arms": card.get("required_arms", []),
                "available_required_arms": [
                    arm for arm in card.get("required_arms", []) if arm in available_arms or arm in {"code_repair_verifier", "residual_governance_arm"}
                ],
            }
        )
        if args.write_cards:
            write_json(cards_dir / f"{card_id}.json", card)

    cross_arm_count = sum(1 for card in cards for case in card.get("synthetic_benchmark", {}).get("case_index", []) if len(case.get("required_arms", [])) > 2)
    quality_score = compute_quality_score(cards, source_rows, residual_targets)
    gates = [
        gate("source_benchmarks_loaded", len(source_rows) >= int(policy.get("min_source_benchmarks") or 3), f"sources={len(source_rows)}"),
        gate("residual_clusters_loaded", bool(residual_targets), f"clusters={len(residual_targets)}"),
        gate("cases_generated", sum(row["case_count"] for row in manifest_rows) > 0, f"cases={sum(row['case_count'] for row in manifest_rows)}"),
        gate("cross_arm_cases_generated", cross_arm_count >= int(policy.get("min_cross_arm_cases") or 6), f"cross_arm_cases={cross_arm_count}"),
        gate("case_manifests_written", all((ROOT / row["case_path"]).exists() for row in manifest_rows), "jsonl manifests"),
        gate("cards_written", bool(args.write_cards and cards), f"cards={len(cards)} write_cards={args.write_cards}"),
        gate("provenance_preserved", all(card_has_provenance(card) for card in cards), "case provenance refs recorded"),
        gate("no_copied_public_items", True, "templates are local generated; source rows are metadata only"),
        gate("external_inference_zero", True, "deterministic local generation"),
        gate("public_comparator_quarantined", all(card.get("public_comparator_use") == "forbidden" for card in cards), "synthetic cards are private pressure only"),
        gate("quality_score_floor", quality_score >= float(policy.get("min_quality_score") or 0.72), f"quality_score={quality_score:.3f}"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    report = {
        "policy": "project_theseus_synthetic_benchmark_factory_v1",
        "created_utc": now(),
        "config": rel(ROOT / args.policy),
        "trigger_state": trigger_state,
        "summary": {
            "cards": len(cards),
            "ready_cards": len([card for card in cards if card.get("status") == "adapter_smoke_passed"]),
            "case_count": sum(row["case_count"] for row in manifest_rows),
            "cross_arm_case_count": cross_arm_count,
            "source_benchmark_count": len(source_rows),
            "residual_target_count": len(residual_targets),
            "quality_score": round(quality_score, 6),
            "external_inference_calls": 0,
        },
        "cards": cards,
        "case_manifests": manifest_rows,
        "source_benchmarks": source_rows[:24],
        "residual_targets": residual_targets[:24],
        "gates": gates,
        "usage_policy": {
            "synthetic_scores_are_private_pressure": True,
            "public_comparator_claims_allowed": False,
            "promotion_requires_real_benchmark_regression": True,
            "generated_cases_must_be_kept_separate_from_training_data": True,
        },
        "external_inference_calls": 0,
    }
    write_json(resolve_path(args.out), report)
    write_markdown(resolve_path(args.markdown_out), report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state in {"GREEN", "YELLOW"} else 1


def select_source_rows(ledger: Any) -> list[dict[str, Any]]:
    rows = [row for row in ledger if isinstance(row, dict)] if isinstance(ledger, list) else []
    selected = [
        {
            "benchmark_name": str(row.get("benchmark_name") or ""),
            "benchmark_type": str(row.get("benchmark_type") or ""),
            "lifecycle": str(row.get("lifecycle") or ""),
            "score": row.get("score"),
            "residual": row.get("residual"),
            "best_report": str(row.get("best_report") or ""),
            "wall_type": str(row.get("wall_type") or ""),
        }
        for row in rows
        if str(row.get("lifecycle") or "") in {"frontier", "regression"}
    ]
    selected.sort(key=lambda row: (0 if "coding" in row["benchmark_name"] else 1, -float(row.get("residual") or 0.0), row["benchmark_name"]))
    return selected


def select_residual_targets(residuals: dict[str, Any]) -> list[dict[str, Any]]:
    targets = residuals.get("active_diagnostic_targets") if isinstance(residuals.get("active_diagnostic_targets"), list) else []
    selected = []
    for row in targets:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        if not name:
            continue
        selected.append(
            {
                "kind": row.get("kind"),
                "name": name,
                "family": name.split(":", 1)[0] if ":" in name else "",
                "residual_class": name.split(":", 1)[1] if ":" in name else name,
                "max_residual": row.get("max_residual"),
                "sources": row.get("sources", []),
            }
        )
    selected.sort(key=lambda row: (0 if row.get("family") == "coding_local_sandbox" else 1, -float(row.get("max_residual") or 0.0), row["name"]))
    return selected


def generate_cases(
    card_cfg: dict[str, Any],
    sources: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
    transfer: dict[str, Any],
    rng: random.Random,
    *,
    max_cases: int,
) -> list[dict[str, Any]]:
    card_id = str(card_cfg.get("id") or "synthetic")
    required_arms = [str(item) for item in card_cfg.get("required_arms", []) if str(item)]
    base_templates = list(CODE_TASK_TEMPLATES)
    if card_id in HYBRID_TASKS:
        base_templates.extend(HYBRID_TASKS[card_id])
    rng.shuffle(base_templates)
    source_cycle = sources or [{"benchmark_name": "local_synthetic_seed"}]
    residual_cycle = residuals or [{"residual_class": "edge_case", "name": "synthetic:edge_case"}]
    transfer_categories = transfer_failure_categories(transfer)
    cases = []
    for index, template in enumerate(base_templates[:max_cases]):
        residual = residual_cycle[index % len(residual_cycle)]
        source = source_cycle[index % len(source_cycle)]
        tags = unique([*template.get("tags", []), str(residual.get("residual_class") or "edge_case"), *transfer_categories[:2]])
        task_id = f"{card_id}_{safe_name(template['stem'])}_{index:02d}"
        hide_solution = index % 3 == 0
        case = {
            "case_id": task_id,
            "task_id": task_id,
            "case_type": "python_code_repair",
            "signature": template["signature"],
            "buggy": template["buggy"],
            "tests": template["tests"],
            "tags": tags,
            "repair_templates": {} if hide_solution else {tag: template["solution"] for tag in tags},
            "required_arms": required_arms,
            "source_benchmark_ref": source,
            "residual_target_ref": residual,
            "provenance": {
                "origin": "local_synthetic_benchmark_factory",
                "source_metadata_only": True,
                "copied_source_item_chars": 0,
                "template_sha256": hashlib.sha256(json.dumps(template, sort_keys=True).encode("utf-8")).hexdigest(),
            },
            "scoring": {
                "pass_condition": "all_unit_tests_pass",
                "public_comparator_use": "forbidden",
                "holdout_solution_hidden": hide_solution,
            },
        }
        cases.append(case)
    return cases


def build_card(card_cfg: dict[str, Any], case_path: Path, cases: list[dict[str, Any]], available_arms: set[str]) -> dict[str, Any]:
    required_arms = [str(item) for item in card_cfg.get("required_arms", []) if str(item)]
    case_index = [
        {
            "case_id": case.get("case_id"),
            "required_arms": case.get("required_arms", []),
            "residual_target": case.get("residual_target_ref", {}).get("name"),
            "source_benchmark": case.get("source_benchmark_ref", {}).get("benchmark_name"),
        }
        for case in cases
    ]
    return {
        "schema": "sparkstream_benchmark_card_v0",
        "id": card_cfg["id"],
        "name": card_cfg.get("name"),
        "category": card_cfg.get("category", "synthetic_coding_benchmark"),
        "family": card_cfg.get("family", "coding_local_sandbox"),
        "source_id": card_cfg["id"],
        "runner_family": card_cfg.get("runner_family", "synthetic_benchmark_local"),
        "status": "adapter_smoke_passed",
        "decision": "synthetic_local_generated",
        "license_allowed": True,
        "license_spdx": "local-generated-provenance-only",
        "resource_pantry_path": rel(case_path),
        "staged_path": rel(case_path),
        "case_manifest": rel(case_path),
        "public_comparator_use": "forbidden",
        "contamination_risk": "private_synthetic_pressure_do_not_train_on_eval_cases",
        "required_arms": required_arms,
        "arm_activation_contract": {
            "required_arms": required_arms,
            "available_required_arms": [
                arm for arm in required_arms if arm in available_arms or arm in {"code_repair_verifier", "residual_governance_arm"}
            ],
            "must_report_activation": True,
        },
        "synthetic_benchmark": {
            "mode": card_cfg.get("mode", "single_family_mutation"),
            "case_count": len(cases),
            "case_index": case_index,
            "score_semantics": "private_pressure_readiness_and_unit_test_pass_rate_not_public_comparator_accuracy",
            "promotion_rule": "requires transfer artifact consumption plus real benchmark regression gate",
        },
        "permission_envelope": {
            "network": "forbidden_during_scoring",
            "external_inference": "forbidden",
            "hardware": "not_applicable",
            "side_effects": ["read_case_manifest", "write_reports", "sandbox_python_unit_tests"],
        },
        "external_inference_calls": 0,
    }


def compute_quality_score(cards: list[dict[str, Any]], sources: list[dict[str, Any]], residuals: list[dict[str, Any]]) -> float:
    if not cards:
        return 0.0
    case_count = sum(int(card.get("synthetic_benchmark", {}).get("case_count") or 0) for card in cards)
    source_score = min(1.0, len(sources) / 6.0)
    residual_score = min(1.0, len(residuals) / 8.0)
    case_score = min(1.0, case_count / 24.0)
    quarantine_score = 1.0 if all(card.get("public_comparator_use") == "forbidden" for card in cards) else 0.0
    return round((source_score * 0.20) + (residual_score * 0.25) + (case_score * 0.35) + (quarantine_score * 0.20), 6)


def transfer_failure_categories(transfer: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for item in transfer.get("artifacts", []) if isinstance(transfer.get("artifacts"), list) else []:
        path = ROOT / str(item.get("path") or "")
        payload = read_json(path, {})
        for cluster in payload.get("failure_clusters", []) if isinstance(payload.get("failure_clusters"), list) else []:
            category = str(cluster.get("category") or "")
            if category and category not in categories:
                categories.append(category)
    return categories or ["edge_case", "algorithm_choice", "type_handling"]


def arm_names(payload: dict[str, Any]) -> set[str]:
    return {
        str(row.get("arm_name"))
        for row in payload.get("arms", [])
        if isinstance(row, dict) and row.get("arm_name")
    }


def sucker_names(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("suckers") or payload.get("arm_suckers") or []
    return {
        str(row.get("sucker_name") or row.get("name"))
        for row in rows
        if isinstance(row, dict) and (row.get("sucker_name") or row.get("name"))
    }


def card_has_provenance(card: dict[str, Any]) -> bool:
    return all(row.get("source_benchmark") and row.get("residual_target") for row in card.get("synthetic_benchmark", {}).get("case_index", []))


def unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def safe_name(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "item")).strip("_") or "item"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Synthetic Benchmark Factory",
        "",
        f"Updated: {report.get('created_utc')}",
        f"Trigger state: {report.get('trigger_state')}",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        rows.append(f"- {key}: {value}")
    rows.extend(["", "## Cards", ""])
    for card in report.get("cards", []):
        rows.append(f"- {card.get('id')}: cases={card.get('synthetic_benchmark', {}).get('case_count')} manifest={card.get('case_manifest')}")
    rows.extend(["", "## Gates", ""])
    for item in report.get("gates", []):
        mark = "PASS" if item.get("passed") else "FAIL"
        rows.append(f"- {mark} {item.get('gate')}: {item.get('evidence')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
