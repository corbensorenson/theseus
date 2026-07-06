"""Audit whether benchmaxxing is producing broad transfer or narrow fitting.

Public benchmark scores are calibration signals, not training targets. This
report reads the broad transfer matrix and residual curriculum, then asks
whether the system is learning source-agnostic concepts that should transfer
across related cards. It deliberately avoids public answers, hidden tests, or
task-specific solutions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PUBLIC_FLOOR = 0.70
MIN_TRANSFER_CARDS = 3

CONCEPT_MAP = {
    "type_handling": "type_and_return_shape",
    "local_code_generation_adapter_needed": "admissibility_and_interface",
    "edge_case": "edge_conditions",
    "timeout": "termination_and_complexity",
    "algorithm_choice": "algorithmic_planning",
    "wrong_answer": "semantic_execution",
    "no_admissible_candidate": "admissibility_and_interface",
    "parsing": "parsing_and_io_shape",
}

SOURCE_TAGS = {
    "source_human_eval": "human_eval",
    "source_mbpp": "mbpp",
    "source_evalplus": "evalplus",
    "source_bigcodebench": "bigcodebench",
    "source_livecodebench": "livecodebench",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="reports/broad_transfer_matrix.json")
    parser.add_argument("--residual-curriculum", default="reports/code_residual_curriculum.json")
    parser.add_argument("--scheduler", default="reports/broad_code_calibration_scheduler.json")
    parser.add_argument("--min-transfer-cards", type=int, default=MIN_TRANSFER_CARDS)
    parser.add_argument("--floor", type=float, default=PUBLIC_FLOOR)
    parser.add_argument("--out", default="reports/transfer_generalization_audit.json")
    parser.add_argument("--markdown-out", default="reports/transfer_generalization_audit.md")
    args = parser.parse_args()

    matrix = read_json(resolve(args.matrix))
    curriculum_path = select_residual_curriculum_path(args.residual_curriculum)
    curriculum = read_json(curriculum_path)
    scheduler = read_json(resolve(args.scheduler))
    payload = build_payload(
        matrix,
        curriculum,
        scheduler,
        min_transfer_cards=max(1, int(args.min_transfer_cards)),
        floor=float(args.floor),
        matrix_path=args.matrix,
        curriculum_path=rel(curriculum_path),
        scheduler_path=args.scheduler,
    )
    write_json(resolve(args.out), payload)
    write_text(resolve(args.markdown_out), render_markdown(payload))
    print(json.dumps(payload, indent=2))
    return 0 if payload["trigger_state"] in {"GREEN", "YELLOW"} else 2


def select_residual_curriculum_path(configured: str) -> Path:
    """Prefer fresh source-agnostic high-transfer pressure over stale generic rows.

    The generic residual curriculum is useful as a fallback, but it can stay
    stale after a teacher-guided source-agnostic shard is generated. The audit's
    default should inspect the current private pressure that the scheduler is
    actually using.
    """

    configured_path = resolve(configured)
    if configured != "reports/code_residual_curriculum.json":
        return configured_path
    candidates = [configured_path, *REPORTS.glob("high_transfer_*_code_residual_curriculum.json")]
    clean: list[Path] = []
    fallback: list[Path] = []
    for path in candidates:
        if not path.exists():
            continue
        data = read_json(path)
        if data.get("policy") != "project_theseus_code_residual_curriculum_v1":
            continue
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        if int(summary.get("private_row_count") or 0) <= 0:
            continue
        fallback.append(path)
        benchmark_rows = int(summary.get("benchmark_named_private_rows") or 0)
        execution_share = safe_float(summary.get("execution_shaped_programs_share"), 1.0)
        if data.get("trigger_state") == "GREEN" and benchmark_rows == 0 and execution_share <= 0.25:
            clean.append(path)
    if clean:
        return max(clean, key=lambda item: item.stat().st_mtime)
    if fallback:
        return max(fallback, key=lambda item: item.stat().st_mtime)
    return configured_path


def build_payload(
    matrix: dict[str, Any],
    curriculum: dict[str, Any],
    scheduler: dict[str, Any],
    *,
    min_transfer_cards: int,
    floor: float,
    matrix_path: str,
    curriculum_path: str,
    scheduler_path: str,
) -> dict[str, Any]:
    rows = [row for row in matrix.get("rows", []) if isinstance(row, dict)]
    clean_rows = [
        row
        for row in rows
        if bool(row.get("no_cheat_valid")) and int(row.get("public_task_count") or 0) > 0
    ]
    rates = [float(row.get("multi_stream_pass_rate") or 0.0) for row in clean_rows]
    deltas = [float(row.get("pass_rate_delta") or 0.0) for row in clean_rows]
    concept_rows = concept_pressure(clean_rows)
    curriculum_summary = curriculum.get("summary") if isinstance(curriculum.get("summary"), dict) else {}
    curriculum_risk = curriculum_specificity(curriculum_summary)
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    spread = round(max(rates) - min(rates), 6) if rates else 0.0
    stddev = round(population_stddev(rates), 6)
    above_floor_cards = [
        str(row.get("card_id"))
        for row in clean_rows
        if float(row.get("multi_stream_pass_rate") or 0.0) >= floor
        and int(row.get("public_task_count") or 0) >= int(summary.get("min_public_tasks_per_promotion_card") or 32)
    ]
    weak_cards = [
        str(row.get("card_id"))
        for row in clean_rows
        if float(row.get("multi_stream_pass_rate") or 0.0) < floor
    ]
    thin_cards = [
        str(row.get("card_id"))
        for row in clean_rows
        if int(row.get("public_task_count") or 0) < int(summary.get("min_public_tasks_per_promotion_card") or 32)
    ]
    risks = overfit_risks(
        clean_rows,
        rates=rates,
        deltas=deltas,
        spread=spread,
        above_floor_cards=above_floor_cards,
        weak_cards=weak_cards,
        thin_cards=thin_cards,
        curriculum_risk=curriculum_risk,
        floor=floor,
        min_transfer_cards=min_transfer_cards,
        matrix=matrix,
    )
    shared_concepts = [
        row for row in concept_rows if int(row.get("card_count") or 0) >= 2
    ]
    recommended = recommended_actions(
        clean_rows,
        concept_rows=concept_rows,
        shared_concepts=shared_concepts,
        risks=risks,
        scheduler=scheduler,
        floor=floor,
    )
    gates = [
        gate("broad_matrix_loaded", matrix.get("policy") == "project_theseus_broad_transfer_matrix_v1", matrix_path),
        gate("public_answers_absent", True, "audit reads metrics, residual labels, and task ids only"),
        gate("minimum_transfer_card_count", len(above_floor_cards) >= min_transfer_cards, above_floor_cards),
        gate("aggregate_above_floor", float(summary.get("real_public_pass_rate") or 0.0) >= floor, summary.get("real_public_pass_rate")),
        gate("no_single_card_truth", len(clean_rows) >= min_transfer_cards, [row.get("card_id") for row in clean_rows]),
        gate("curriculum_not_benchmark_name_dominated", not curriculum_risk["benchmark_name_dominated"], curriculum_risk),
    ]
    transfer_ready = all(item["passed"] for item in gates)
    trigger_state = "GREEN" if transfer_ready else "YELLOW"
    if matrix.get("policy") != "project_theseus_broad_transfer_matrix_v1":
        trigger_state = "RED"
    return {
        "policy": "project_theseus_transfer_generalization_audit_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "thesis": "Benchmaxxing must reward transferable concepts, not benchmark-name specialization.",
        "inputs": {
            "matrix": matrix_path,
            "residual_curriculum": curriculum_path,
            "scheduler": scheduler_path,
        },
        "summary": {
            "clean_public_card_count": len(clean_rows),
            "above_floor_transfer_card_count": len(above_floor_cards),
            "above_floor_cards": above_floor_cards,
            "weak_cards": weak_cards,
            "thin_cards": thin_cards,
            "aggregate_pass_rate": summary.get("real_public_pass_rate"),
            "aggregate_sts_delta": summary.get("real_public_sts_delta"),
            "best_card_pass_rate": round(max(rates), 6) if rates else 0.0,
            "worst_card_pass_rate": round(min(rates), 6) if rates else 0.0,
            "card_pass_rate_spread": spread,
            "card_pass_rate_stddev": stddev,
            "transfer_ready": transfer_ready,
            "overfit_risk_count": len(risks),
        },
        "concept_transfer_pressure": concept_rows,
        "shared_concept_targets": shared_concepts,
        "curriculum_specificity": curriculum_risk,
        "overfit_risks": risks,
        "recommended_actions": recommended,
        "gates": gates,
        "rules": {
            "public_benchmarks": "calibration_only_never_training_solutions",
            "private_curriculum": "source_agnostic_concept_families_before_benchmark_name_families",
            "promotion": "requires broad transfer across cards, not one high card",
            "teacher": "architecture diagnosis from residual clusters only",
        },
        "promotion_evidence": False,
        "score_semantics": "generalization audit over public calibration; not a student score by itself",
        "external_inference_calls": 0,
    }


def concept_pressure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    concept_cards: dict[str, set[str]] = defaultdict(set)
    concept_counts: Counter[str] = Counter()
    concept_public_tasks: Counter[str] = Counter()
    concept_residual_examples: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        card = str(row.get("card_id") or "")
        residuals = row.get("residual_family_counts") if isinstance(row.get("residual_family_counts"), dict) else {}
        for residual, count in residuals.items():
            concept = concept_for(str(residual))
            n = int(count or 0)
            concept_counts[concept] += n
            concept_cards[concept].add(card)
            concept_public_tasks[concept] += int(row.get("public_task_count") or 0)
            concept_residual_examples[concept][str(residual)] += n
    rows_out = []
    for concept, count in concept_counts.most_common():
        cards = sorted(concept_cards[concept])
        examples = dict(concept_residual_examples[concept].most_common(6))
        rows_out.append(
            {
                "concept": concept,
                "residual_count": int(count),
                "card_count": len(cards),
                "cards": cards,
                "residual_examples": examples,
                "recommended_private_pressure": private_pressure_for_concept(concept),
            }
        )
    return rows_out


def concept_for(residual: str) -> str:
    if residual in CONCEPT_MAP:
        return CONCEPT_MAP[residual]
    if "type" in residual or "return" in residual or "shape" in residual:
        return "type_and_return_shape"
    if "adapter" in residual or "candidate" in residual or "interface" in residual:
        return "admissibility_and_interface"
    if "edge" in residual or "empty" in residual or "guard" in residual:
        return "edge_conditions"
    if "algorithm" in residual or "choice" in residual or "loop" in residual:
        return "algorithmic_planning"
    if "parse" in residual or "stdin" in residual or "io" in residual:
        return "parsing_and_io_shape"
    return "semantic_execution"


def private_pressure_for_concept(concept: str) -> str:
    mapping = {
        "type_and_return_shape": "private source-agnostic tasks varying signatures, nested containers, scalar/list/string/int return shapes, and conversion boundaries",
        "admissibility_and_interface": "AST-constrained full-body skeletons with varied entry points, imports absent, local variables, branch/loop requirements, and return-shape checks",
        "edge_conditions": "hidden-test private edge families: empty inputs, singleton inputs, leading zeros, punctuation, negative values, final-element exceptions",
        "algorithmic_planning": "private algorithm families requiring selecting recurrence, counting, sorting, search, dynamic state, or number-theory loops from prompt semantics",
        "termination_and_complexity": "bounded-loop and complexity guard families with timeouts, early exits, and nonrecursive fallbacks",
        "parsing_and_io_shape": "private parsing families covering text quantities, stdin-like payloads, function-call payloads, and malformed-but-valid inputs",
        "semantic_execution": "private execution-feedback traces with solver->test->critic->patch loops and no public benchmark solutions",
    }
    return mapping.get(concept, mapping["semantic_execution"])


def curriculum_specificity(summary: dict[str, Any]) -> dict[str, Any]:
    private_rows = max(1, int(summary.get("private_row_count") or 0))
    target_counts = summary.get("target_wall_family_counts") if isinstance(summary.get("target_wall_family_counts"), dict) else {}
    source_specific_count = 0
    source_specific_families: dict[str, int] = {}
    for family, count in target_counts.items():
        family_text = str(family).lower()
        if any(tag in family_text for tag in ("mbpp", "evalplus", "humaneval", "bigcodebench", "livecodebench")):
            n = int(count or 0)
            source_specific_count += n
            source_specific_families[str(family)] = n
    top_family, top_count = ("", 0)
    if target_counts:
        top_family, top_count = max(((str(k), int(v or 0)) for k, v in target_counts.items()), key=lambda item: item[1])
    source_share = source_specific_count / private_rows
    top_share = int(top_count or 0) / private_rows
    return {
        "private_row_count": private_rows if summary.get("private_row_count") else 0,
        "active_card": summary.get("broad_active_card"),
        "source_specific_private_row_share": round(source_share, 6),
        "source_specific_families": source_specific_families,
        "top_family": top_family,
        "top_family_share": round(top_share, 6),
        "edge_case_private_rows": summary.get("edge_case_private_rows"),
        "local_adapter_private_rows": summary.get("local_adapter_private_rows"),
        "benchmark_name_dominated": source_share > 0.20 or top_share > 0.45,
    }


def overfit_risks(
    rows: list[dict[str, Any]],
    *,
    rates: list[float],
    deltas: list[float],
    spread: float,
    above_floor_cards: list[str],
    weak_cards: list[str],
    thin_cards: list[str],
    curriculum_risk: dict[str, Any],
    floor: float,
    min_transfer_cards: int,
    matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    summary = matrix.get("summary") if isinstance(matrix.get("summary"), dict) else {}
    if len(above_floor_cards) < min_transfer_cards:
        risks.append(
            risk(
                "insufficient_cross_card_mastery",
                "Promotion would be benchmark-specific because too few clean cards are above floor.",
                {"above_floor_cards": above_floor_cards, "required": min_transfer_cards},
            )
        )
    if rates and max(rates) >= floor and float(summary.get("real_public_pass_rate") or 0.0) < floor:
        risks.append(
            risk(
                "single_card_strength_hides_broad_weakness",
                "At least one card is above floor, but aggregate broad transfer remains below floor.",
                {"best": max(rates), "aggregate": summary.get("real_public_pass_rate")},
            )
        )
    if spread > 0.35:
        risks.append(
            risk(
                "large_cross_card_spread",
                "The learner is uneven across related coding cards, a sign of weak transfer.",
                {"spread": spread, "rates": {str(row.get("card_id")): row.get("multi_stream_pass_rate") for row in rows}},
            )
        )
    flat_cards = [
        str(row.get("card_id"))
        for row in rows
        if float(row.get("pass_rate_delta") or 0.0) <= 0.0
    ]
    if flat_cards:
        risks.append(
            risk(
                "sts_not_causal_per_card",
                "STS must beat STS-off per card, not only in aggregate.",
                {"flat_or_negative_delta_cards": flat_cards},
            )
        )
    if thin_cards:
        risks.append(
            risk(
                "thin_calibration_slice",
                "Coverage is too thin to judge transfer for one or more cards.",
                {"thin_cards": thin_cards},
            )
        )
    if weak_cards:
        risks.append(
            risk(
                "below_floor_receiver_cards",
                "Below-floor receiver cards show that learned concepts are not transferring broadly enough.",
                {"weak_cards": weak_cards},
            )
        )
    if curriculum_risk.get("benchmark_name_dominated"):
        risks.append(
            risk(
                "benchmark_name_specific_curriculum",
                "Private curriculum is too source-name dominated; prefer concept families that transfer across cards.",
                curriculum_risk,
            )
        )
    if int(summary.get("no_cheat_violation_count") or 0) > 0:
        risks.append(
            risk(
                "no_cheat_violation",
                "No-cheat violations invalidate transfer claims.",
                {"count": summary.get("no_cheat_violation_count")},
                severity="hard",
            )
        )
    return risks


def recommended_actions(
    rows: list[dict[str, Any]],
    *,
    concept_rows: list[dict[str, Any]],
    shared_concepts: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    scheduler: dict[str, Any],
    floor: float,
) -> list[dict[str, Any]]:
    weak_rows = [
        row for row in rows if float(row.get("multi_stream_pass_rate") or 0.0) < floor
    ]
    selected = scheduler.get("selected") if isinstance(scheduler.get("selected"), dict) else {}
    actions = []
    top_shared = shared_concepts[:4] or concept_rows[:4]
    for concept in top_shared:
        actions.append(
            {
                "priority": "high",
                "kind": "source_agnostic_private_concept_pressure",
                "title": f"Train transferable private {concept['concept']} families",
                "reason": "Concept appears across multiple public calibration cards, so it is a transfer target rather than a benchmark target.",
                "concept": concept["concept"],
                "cards": concept.get("cards", []),
                "private_pressure": concept.get("recommended_private_pressure"),
            }
        )
    if weak_rows:
        actions.append(
            {
                "priority": "high",
                "kind": "donor_receiver_transfer_eval",
                "title": "Run donor/receiver transfer checks before claiming progress",
                "reason": "A private curriculum inspired by one card should be evaluated on at least two receiver cards with the same checkpoint and seed.",
                "donor_card": selected.get("card_id") or "current_active_card",
                "receiver_cards": [str(row.get("card_id")) for row in weak_rows[:3]],
            }
        )
    if any(item.get("id") == "sts_not_causal_per_card" for item in risks):
        actions.append(
            {
                "priority": "medium",
                "kind": "per_card_sts_causality",
                "title": "Make STS-on beat STS-off per card",
                "reason": "Aggregate STS deltas can hide cards where streams are decorative.",
            }
        )
    actions.append(
        {
            "priority": "medium",
            "kind": "decoder_architecture_pressure",
            "title": "Patch decoder by transferable concept, not benchmark name",
            "reason": "Repeated source-specific residual rows did not move MBPP; next work should change type/AST/semantic planning in the generator.",
        }
    )
    return actions


def risk(risk_id: str, reason: str, evidence: Any, *, severity: str = "medium") -> dict[str, Any]:
    return {"id": risk_id, "severity": severity, "reason": reason, "evidence": evidence}


def population_stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def gate(name: str, passed: bool, evidence: Any, *, severity: str = "hard") -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Transfer Generalization Audit",
        "",
        f"- trigger_state: `{payload.get('trigger_state')}`",
        f"- clean_public_card_count: `{summary.get('clean_public_card_count')}`",
        f"- above_floor_transfer_card_count: `{summary.get('above_floor_transfer_card_count')}`",
        f"- aggregate_pass_rate: `{summary.get('aggregate_pass_rate')}`",
        f"- card_pass_rate_spread: `{summary.get('card_pass_rate_spread')}`",
        f"- overfit_risk_count: `{summary.get('overfit_risk_count')}`",
        "",
        "## Shared Concepts",
        "",
        "| Concept | Residuals | Cards | Private Pressure |",
        "| --- | ---: | --- | --- |",
    ]
    for row in payload.get("shared_concept_targets", [])[:8]:
        lines.append(
            f"| {row.get('concept')} | {row.get('residual_count')} | {', '.join(row.get('cards') or [])} | {row.get('recommended_private_pressure')} |"
        )
    lines.extend(["", "## Risks", ""])
    for item in payload.get("overfit_risks", []):
        lines.append(f"- `{item.get('id')}` ({item.get('severity')}): {item.get('reason')}")
    lines.extend(["", "## Recommended Actions", ""])
    for item in payload.get("recommended_actions", []):
        lines.append(f"- `{item.get('priority')}` `{item.get('kind')}`: {item.get('title')}")
    lines.extend(
        [
            "",
            "Public benchmarks remain calibration-only. This report is a guardrail against narrow benchmark fitting, not promotion evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = resolve(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
