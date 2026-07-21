#!/usr/bin/env python3
"""Control evidence-efficient reviews for the frozen neural-seed campaign.

The controller never trains or scores models itself. It validates independently emitted
private-development review receipts, decides whether the next matched training rung is
worth buying, and keeps early engineering stops distinct from architecture falsification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALE_CONFIG = ROOT / "configs/neural_seed_50m_scale_preregistration.json"
DEFAULT_TRAINING_CONFIG = ROOT / "configs/moecot_language_arm_training.json"
DEFAULT_REVIEW_DIR = ROOT / "reports/neural_seed_57m_architecture_reviews"
DEFAULT_OUT = ROOT / "reports/neural_seed_campaign_controller.json"
SYSTEM_IDS = ("moecot_system", "dense_active_parameter", "dense_total_parameter")
ARM_IDS = ("english", "python", "javascript_typescript", "html_css", "rust")
REVIEW_POLICY = "project_theseus_architecture_review_receipt_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-config", default=relative(DEFAULT_SCALE_CONFIG))
    parser.add_argument("--training-config", default=relative(DEFAULT_TRAINING_CONFIG))
    parser.add_argument("--review-dir", default=relative(DEFAULT_REVIEW_DIR))
    parser.add_argument("--out", default=relative(DEFAULT_OUT))
    args = parser.parse_args()

    report = build_campaign_status(
        scale_config_path=resolve(args.scale_config),
        training_config_path=resolve(args.training_config),
        review_dir=resolve(args.review_dir),
    )
    write_json(resolve(args.out), report)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] != "RED" else 2


def build_campaign_status(
    *,
    scale_config_path: Path,
    training_config_path: Path,
    review_dir: Path,
) -> dict[str, Any]:
    scale = read_json(scale_config_path)
    training = read_json(training_config_path)
    hard_gaps = validate_contract(scale, training)
    stop = scale.get("training_stop_contract") or {}
    pilot = int(stop.get("pilot_optimizer_positions") or 0)
    maximum = int(stop.get("maximum_optimizer_positions") or 0)
    review_points = [
        pilot,
        *[int(value) for value in stop.get("review_optimizer_positions") or []],
        maximum,
    ]
    review_points = sorted(set(value for value in review_points if value > 0))
    receipt_rows, receipt_faults = load_review_receipts(review_dir)
    hard_gaps.extend(receipt_faults)
    decisions = []
    active = list(SYSTEM_IDS)
    prior_complete_reviews = 0
    for review_position in review_points:
        rows = [
            row
            for row in receipt_rows
            if int(row.get("review_optimizer_positions") or 0) == review_position
            and str(row.get("candidate_id") or "") in active
        ]
        result = decide_review(
            review_position=review_position,
            maximum_optimizer_positions=maximum,
            active_candidates=active,
            rows=rows,
            prior_complete_reviews=prior_complete_reviews,
        )
        decisions.append(result)
        if result["state"] == "INVALID":
            hard_gaps.extend(
                f"review_{review_position}:{gap}"
                for gap in result.get("hard_gaps") or ["invalid_review"]
            )
        if result["state"] != "COMPLETE":
            break
        prior_complete_reviews += 1
        if result["decision"] == "STOP_SCALE_RUNG":
            active = []
            break
        halted = set(result.get("halted_candidate_ids") or [])
        active = [candidate for candidate in active if candidate not in halted]

    first = review_points[0] if review_points else 0
    speedup_opportunity = maximum / max(1, first)
    current = decisions[-1] if decisions else {}
    state = "RED" if hard_gaps else "GREEN" if current.get("state") == "COMPLETE" else "READY"
    return {
        "policy": "project_theseus_evidence_efficient_campaign_controller_v1",
        "created_utc": now(),
        "trigger_state": state,
        "scale_config": artifact(scale_config_path),
        "training_config": artifact(training_config_path),
        "review_directory": relative(review_dir),
        "candidate_ids": list(SYSTEM_IDS),
        "arm_ids": list(ARM_IDS),
        "review_optimizer_positions": review_points,
        "first_review_optimizer_positions": first,
        "maximum_optimizer_positions": maximum,
        "first_review_budget_speedup_opportunity": round(speedup_opportunity, 6),
        "target_speedup_met_by_contract": speedup_opportunity >= 10.0,
        "target_speedup_empirically_proven": bool(
            decisions
            and decisions[0].get("state") == "COMPLETE"
            and decisions[0].get("decision") in {"STOP_SCALE_RUNG", "HALT_DOMINATED"}
        ),
        "active_candidate_ids": active,
        "review_receipt_count": len(receipt_rows),
        "reviews": decisions,
        "next_action": next_action(decisions, active, first),
        "hard_gaps": sorted(set(hard_gaps)),
        "boundaries": {
            "development_surface_only": True,
            "confirmation_surface_consumed": False,
            "public_surface_consumed": False,
            "public_training_rows_written": 0,
            "external_inference_calls": 0,
            "fallback_return_count": 0,
            "templates_renderers_routers_tools_credit": 0,
            "early_stop_is_architecture_falsification": False,
            "loss_or_routing_success_substitutes_for_direct_utility": False,
        },
        "claim_scope": (
            "Campaign scheduling and exact-rung engineering disposition only. A halt does "
            "not broadly falsify an architecture, substrate, or scaling law."
        ),
    }


def validate_contract(scale: dict[str, Any], training: dict[str, Any]) -> list[str]:
    gaps = []
    if scale.get("policy") != "project_theseus_neural_seed_50m_scale_preregistration_v1":
        gaps.append("scale_preregistration_policy_mismatch")
    stop = scale.get("training_stop_contract") or {}
    pilot = int(stop.get("pilot_optimizer_positions") or 0)
    reviews = [int(value) for value in stop.get("review_optimizer_positions") or []]
    maximum = int(stop.get("maximum_optimizer_positions") or 0)
    if not pilot or pilot >= maximum or reviews != sorted(set(reviews)):
        gaps.append("review_budget_order_invalid")
    if any(value <= pilot or value >= maximum for value in reviews):
        gaps.append("review_budget_outside_pilot_maximum")
    if stop.get("stop_on_no_model_only_functional_gain_at_two_consecutive_reviews") is not True:
        gaps.append("direct_utility_stop_contract_missing")
    utility = scale.get("heldout_utility_contract") or {}
    for key in (
        "source_disjoint",
        "confirmation_untouched_until_candidate_selected",
        "report_every_arm_before_aggregate",
        "model_only_and_assisted_channels_separate",
        "accepted_verified_output_per_second_reported",
    ):
        if utility.get(key) is not True:
            gaps.append(f"heldout_utility_contract_missing:{key}")
    comparison = training.get("comparison_contract") or {}
    expected = {"shared_trunk", *ARM_IDS, "dense_active_parameter", "dense_total_parameter"}
    if set(comparison.get("first_campaign_candidate_ids") or []) != expected:
        gaps.append("first_campaign_inventory_mismatch")
    boundaries = scale.get("boundaries") or {}
    for key in (
        "public_training_rows_written",
        "external_inference_calls",
        "teacher_calls",
        "fallback_return_count",
        "templates_renderers_routers_tools_generation_credit",
    ):
        if int(boundaries.get(key) or 0) != 0:
            gaps.append(f"nonzero_boundary:{key}")
    return gaps


def load_review_receipts(review_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not review_dir.exists():
        return [], []
    rows = []
    faults = []
    for path in sorted(review_dir.glob("*.json")):
        try:
            row = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            faults.append(f"review_receipt_unreadable:{path.name}:{type(exc).__name__}")
            continue
        row["_artifact"] = artifact(path)
        receipt_faults = validate_review_receipt(row)
        if receipt_faults:
            faults.extend(f"{path.name}:{fault}" for fault in receipt_faults)
        else:
            rows.append(row)
    identities = [
        (str(row["candidate_id"]), int(row["review_optimizer_positions"])) for row in rows
    ]
    if len(identities) != len(set(identities)):
        faults.append("duplicate_candidate_review_receipt")
    return rows, faults


def validate_review_receipt(row: dict[str, Any]) -> list[str]:
    gaps = []
    if row.get("policy") != REVIEW_POLICY:
        gaps.append("policy_mismatch")
    if row.get("candidate_id") not in SYSTEM_IDS:
        gaps.append("candidate_id_invalid")
    if int(row.get("review_optimizer_positions") or 0) <= 0:
        gaps.append("review_optimizer_positions_invalid")
    evidence = row.get("evidence") or {}
    if evidence.get("split") != "private_dev":
        gaps.append("non_development_evidence")
    if evidence.get("source_disjoint") is not True:
        gaps.append("source_disjoint_not_proven")
    if evidence.get("direct_model_only") is not True:
        gaps.append("direct_model_only_not_proven")
    if evidence.get("confirmation_surface_consumed") is not False:
        gaps.append("confirmation_surface_consumed")
    if evidence.get("public_surface_consumed") is not False:
        gaps.append("public_surface_consumed")
    if int(evidence.get("fallback_return_count") or 0) != 0:
        gaps.append("fallback_return_count_nonzero")
    if int(evidence.get("templates_renderers_routers_tools_credit") or 0) != 0:
        gaps.append("assisted_generation_credit_nonzero")
    total = int(evidence.get("case_count") or 0)
    passed = int(evidence.get("passed_count") or 0)
    if total <= 0 or not 0 <= passed <= total:
        gaps.append("aggregate_counts_invalid")
    by_arm = evidence.get("by_arm") or {}
    if set(by_arm) != set(ARM_IDS):
        gaps.append("arm_coverage_incomplete")
    elif any(
        int((value or {}).get("case_count") or 0) <= 0
        or not 0
        <= int((value or {}).get("passed_count") or 0)
        <= int((value or {}).get("case_count") or 0)
        for value in by_arm.values()
    ):
        gaps.append("arm_counts_invalid")
    required_identity = (
        "plan_sha256",
        "stage_signature",
        "checkpoint_sha256",
        "evaluator_sha256",
        "case_contract_sha256",
        "visible_case_ids_sha256",
        "verifier_budget_sha256",
    )
    if any(not valid_digest(evidence.get(key)) for key in required_identity):
        gaps.append("identity_binding_incomplete")
    if int(evidence.get("optimizer_positions") or 0) < int(
        row.get("review_optimizer_positions") or 0
    ):
        gaps.append("review_budget_not_reached")
    if float(evidence.get("accepted_verified_outputs_per_second") or 0.0) < 0.0:
        gaps.append("throughput_invalid")
    return gaps


def decide_review(
    *,
    review_position: int,
    maximum_optimizer_positions: int,
    active_candidates: list[str],
    rows: list[dict[str, Any]],
    prior_complete_reviews: int,
) -> dict[str, Any]:
    indexed = {str(row.get("candidate_id")): row for row in rows}
    missing = sorted(set(active_candidates) - set(indexed))
    if missing:
        return {
            "review_optimizer_positions": review_position,
            "state": "WAITING",
            "decision": "WAIT_FOR_MATCHED_RECEIPTS",
            "missing_candidate_ids": missing,
            "received_candidate_ids": sorted(indexed),
            "halted_candidate_ids": [],
        }
    comparability = comparable_review_rows([indexed[item] for item in active_candidates])
    if comparability:
        return {
            "review_optimizer_positions": review_position,
            "state": "INVALID",
            "decision": "REJECT_UNMATCHED_REVIEW",
            "hard_gaps": comparability,
            "halted_candidate_ids": [],
        }
    candidates = [candidate_summary(indexed[item]) for item in active_candidates]
    all_zero = all(int(row["passed_count"]) == 0 for row in candidates)
    if all_zero:
        decision = "STOP_SCALE_RUNG"
        halted = list(active_candidates)
        rationale = "all matched candidates produced zero direct model-only passes"
    else:
        halted = []
        if prior_complete_reviews >= 1:
            best = max(candidates, key=lambda row: (row["pass_rate"], row["candidate_id"]))
            for candidate in candidates:
                if candidate["candidate_id"] == best["candidate_id"]:
                    continue
                if clearly_dominated(candidate, best):
                    halted.append(candidate["candidate_id"])
        decision = "HALT_DOMINATED" if halted else (
            "FINAL_REVIEW_COMPLETE" if review_position >= maximum_optimizer_positions else "CONTINUE_MATCHED"
        )
        rationale = (
            "repeated direct-utility confidence intervals support a scoped engineering halt"
            if halted
            else "evidence does not support pruning an active candidate"
        )
    return {
        "review_optimizer_positions": review_position,
        "state": "COMPLETE",
        "decision": decision,
        "rationale": rationale,
        "candidate_summaries": candidates,
        "halted_candidate_ids": halted,
        "architecture_falsification_claimed": False,
        "confirmation_surface_consumed": False,
        "public_surface_consumed": False,
    }


def comparable_review_rows(rows: list[dict[str, Any]]) -> list[str]:
    keys = (
        "plan_sha256",
        "stage_signature",
        "evaluator_sha256",
        "case_contract_sha256",
        "visible_case_ids_sha256",
        "verifier_budget_sha256",
        "case_count",
    )
    gaps = []
    for key in keys:
        values = {json.dumps((row.get("evidence") or {}).get(key), sort_keys=True) for row in rows}
        if len(values) != 1:
            gaps.append(f"unmatched_review_field:{key}")
    positions = [int((row.get("evidence") or {}).get("optimizer_positions") or 0) for row in rows]
    if max(positions) > min(positions) * 1.01:
        gaps.append("optimizer_position_opportunity_mismatch")
    return gaps


def candidate_summary(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row["evidence"]
    passed = int(evidence["passed_count"])
    total = int(evidence["case_count"])
    lower, upper = wilson_interval(passed, total)
    arms = {}
    for arm_id in ARM_IDS:
        arm = evidence["by_arm"][arm_id]
        arm_passed = int(arm["passed_count"])
        arm_total = int(arm["case_count"])
        arm_lower, arm_upper = wilson_interval(arm_passed, arm_total)
        arms[arm_id] = {
            "passed_count": arm_passed,
            "case_count": arm_total,
            "pass_rate": round(arm_passed / arm_total, 8),
            "wilson_95": [round(arm_lower, 8), round(arm_upper, 8)],
        }
    return {
        "candidate_id": row["candidate_id"],
        "passed_count": passed,
        "case_count": total,
        "pass_rate": round(passed / total, 8),
        "wilson_95": [round(lower, 8), round(upper, 8)],
        "by_arm": arms,
        "weakest_arm_pass_rate": min(value["pass_rate"] for value in arms.values()),
        "accepted_verified_outputs_per_second": float(
            evidence.get("accepted_verified_outputs_per_second") or 0.0
        ),
        "optimizer_positions": int(evidence["optimizer_positions"]),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
    }


def clearly_dominated(candidate: dict[str, Any], best: dict[str, Any]) -> bool:
    margin = 0.01
    aggregate = candidate["wilson_95"][1] + margin < best["wilson_95"][0]
    no_unique_arm = all(
        candidate["by_arm"][arm]["wilson_95"][1] + margin
        < best["by_arm"][arm]["wilson_95"][0]
        for arm in ARM_IDS
    )
    return bool(aggregate and no_unique_arm)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def next_action(decisions: list[dict[str, Any]], active: list[str], pilot: int) -> dict[str, Any]:
    if not decisions:
        return {"kind": "materialize_review", "review_optimizer_positions": pilot, "candidate_ids": active}
    current = decisions[-1]
    if current.get("state") == "WAITING":
        return {
            "kind": "materialize_review",
            "review_optimizer_positions": current["review_optimizer_positions"],
            "candidate_ids": current["missing_candidate_ids"],
            "requirements": [
                "matched optimizer-position opportunity",
                "source-disjoint private-development surface",
                "direct model-only outputs",
                "per-arm functional verification",
                "exact checkpoint/evaluator/case/verifier identities",
            ],
        }
    if current.get("state") == "INVALID":
        return {
            "kind": "repair_matched_review_contract",
            "review_optimizer_positions": current["review_optimizer_positions"],
            "candidate_ids": active,
            "hard_gaps": list(current.get("hard_gaps") or []),
        }
    if current.get("decision") == "STOP_SCALE_RUNG":
        return {"kind": "stop_exact_scale_rung", "candidate_ids": current["halted_candidate_ids"]}
    return {"kind": "continue_next_review", "candidate_ids": active}


def summary(report: dict[str, Any]) -> dict[str, Any]:
    current = (report.get("reviews") or [{}])[-1]
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "first_review_budget_speedup_opportunity": report[
            "first_review_budget_speedup_opportunity"
        ],
        "target_speedup_empirically_proven": report[
            "target_speedup_empirically_proven"
        ],
        "current_review_state": current.get("state"),
        "current_decision": current.get("decision"),
        "next_action": report["next_action"],
        "hard_gaps": report["hard_gaps"],
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def valid_digest(value: Any) -> bool:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
