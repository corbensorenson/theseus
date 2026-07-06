"""Gate non-code lanes on transfer evidence instead of trace volume alone."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/agent_lane_transfer_gate.json")
    args = parser.parse_args()

    repo = best_repo_repair_report()
    tool = read_json(ROOT / "reports/high_transfer_long_horizon_tool_use.json")
    puffer = read_json(ROOT / "reports/pufferlib4_rl_lane.json")
    conversation = read_json(ROOT / "reports/high_transfer_multi_turn_conversation_hard_v4.json")
    capsules = read_json(ROOT / "reports/cross_domain_sts_capsules.json")
    sts_ablation = read_json(ROOT / "reports/sts_causal_decoder_ablation.json")
    sts_control = read_json(ROOT / "reports/sts_decoder_control_contract.json")
    transfer = read_json(ROOT / "reports/transfer_generalization_audit.json")

    summary = {
        "repo_repair": lane_repo_repair(repo),
        "terminal_tool_use": lane_tool_use(tool),
        "pufferlib_rl": lane_puffer(puffer),
        "conversation": lane_conversation(conversation),
        "sts_consumption": lane_sts(capsules, sts_ablation, sts_control),
        "public_transfer": lane_public_transfer(transfer),
    }
    gates = [
        gate("repo_repair_has_private_traces", summary["repo_repair"]["trace_ready"], summary["repo_repair"]),
        gate(
            "repo_repair_has_transfer_consumer_evidence",
            summary["repo_repair"]["transfer_consumer_ready"],
            summary["repo_repair"],
        ),
        gate("terminal_tool_use_has_64_cases", summary["terminal_tool_use"]["case_ready"], summary["terminal_tool_use"]),
        gate(
            "terminal_tool_use_has_transfer_consumer",
            summary["terminal_tool_use"]["transfer_consumer_ready"],
            summary["terminal_tool_use"],
        ),
        gate("pufferlib_native_policy_learning_ready", summary["pufferlib_rl"]["native_policy_ready"], summary["pufferlib_rl"]),
        gate(
            "pufferlib_transfer_consumer_ready",
            summary["pufferlib_rl"]["transfer_consumer_ready"],
            summary["pufferlib_rl"],
        ),
        gate("conversation_hard_v4_graduated", summary["conversation"]["graduated"], summary["conversation"]),
        gate("sts_control_contract_ready", summary["sts_consumption"]["control_contract_ready"], summary["sts_consumption"]),
        gate("sts_has_named_consumer_effect", summary["sts_consumption"]["named_consumer_effect"], summary["sts_consumption"]),
        gate("public_transfer_above_floor", summary["public_transfer"]["transfer_ready"], summary["public_transfer"]),
    ]
    trigger_state = "GREEN" if all(row["passed"] for row in gates) else "YELLOW"
    report = {
        "policy": "project_theseus_agent_lane_transfer_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "promotion_evidence": trigger_state == "GREEN",
        "summary": summary,
        "gates": gates,
        "rules": {
            "trace_volume_is_not_promotion": True,
            "consumer_contract": "a lane counts as frontier-grade only when it has private traces plus a named downstream consumer or A/B transfer effect",
            "weakest_lane_caps_breadth": True,
        },
        "next_actions": next_actions(summary),
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def lane_repo_repair(report: Any) -> dict[str, Any]:
    summary = summary_of(report)
    validated = intish(summary.get("validated_private_trace_count"))
    rows = intish(summary.get("code_lm_row_count"))
    contract = summary.get("transfer_consumer_contract") if isinstance(summary.get("transfer_consumer_contract"), dict) else {}
    consumers = contract.get("consumers") if isinstance(contract.get("consumers"), list) else []
    trace_ready = validated >= 128 and rows >= 128
    transfer_ready = trace_ready and bool(summary.get("transfer_evidence_ready")) and len(consumers) >= 3
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "validated_private_trace_count": validated,
        "code_lm_row_count": rows,
        "trace_ready": trace_ready,
        "promotion_evidence": bool(summary.get("promotion_evidence") or report.get("promotion_evidence")),
        "transfer_consumer_ready": transfer_ready,
        "transfer_evidence_ready": bool(summary.get("transfer_evidence_ready")),
        "consumer_count": len(consumers),
        "consumer_contract": contract,
        "consumer": "code_lm_private_rows -> decoder_private_ablation_gate -> private_public_transfer_proof",
    }


def best_repo_repair_report() -> dict[str, Any]:
    reports = [
        read_json(ROOT / "reports/high_transfer_repo_repair_learner.json"),
        read_json(ROOT / "reports/viea_repo_repair_learner.json"),
    ]
    usable = [row for row in reports if isinstance(row, dict) and row]
    if not usable:
        return {}

    def rank(report: dict[str, Any]) -> tuple[int, int, int, int]:
        summary = summary_of(report)
        contract = summary.get("transfer_consumer_contract") if isinstance(summary.get("transfer_consumer_contract"), dict) else {}
        consumers = contract.get("consumers") if isinstance(contract.get("consumers"), list) else []
        return (
            1 if bool(summary.get("transfer_evidence_ready")) else 0,
            len(consumers),
            intish(summary.get("validated_private_trace_count")),
            intish(summary.get("code_lm_row_count")),
        )

    return max(usable, key=rank)


def lane_tool_use(report: Any) -> dict[str, Any]:
    summary = summary_of(report)
    cases = intish(summary.get("case_count"))
    pass_rate = floatish(summary.get("pass_rate"))
    trace_rows = intish(summary.get("trace_rows"))
    sts_rows = intish(summary.get("sts_rows"))
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "case_count": cases,
        "pass_rate": pass_rate,
        "trace_rows": trace_rows,
        "sts_rows": sts_rows,
        "case_ready": cases >= 64 and (pass_rate or 0.0) >= 0.85 and trace_rows >= 64,
        "transfer_consumer_ready": bool(summary.get("promotion_evidence")) or sts_rows >= 64,
        "consumer": "tool-use STS rows -> route memory / retry policy / repo repair task planning",
    }


def lane_puffer(report: Any) -> dict[str, Any]:
    summary = summary_of(report)
    policy_accuracy_delta = floatish(summary.get("policy_accuracy_delta"))
    reward_delta = floatish(summary.get("policy_rollout_reward_delta"))
    score_delta = floatish(summary.get("policy_rollout_score_delta"))
    policy_learning_evidence = bool(
        summary.get("policy_learning_evidence")
        or summary.get("native_policy_learning_evidence")
        or summary.get("fallback_policy_learning_evidence")
    )
    policy_learning_ready = bool(
        policy_learning_evidence
        and intish(summary.get("policy_train_row_count")) >= 32
        and (
            (policy_accuracy_delta or 0.0) > 0.0
            or (reward_delta or 0.0) > 0.0
            or (score_delta or 0.0) > 0.0
        )
    )
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "native_backend_ready": bool(summary.get("native_backend_ready")),
        "native_policy_learning_evidence": bool(summary.get("native_policy_learning_evidence")),
        "fallback_policy_learning_evidence": bool(summary.get("fallback_policy_learning_evidence")),
        "policy_learning_evidence": policy_learning_evidence,
        "policy_learning_backend": summary.get("policy_learning_backend") or "",
        "policy_accuracy_delta": policy_accuracy_delta,
        "policy_rollout_reward_delta": reward_delta,
        "policy_rollout_score_delta": score_delta,
        "policy_train_row_count": intish(summary.get("policy_train_row_count")),
        "native_policy_ready": policy_learning_ready,
        "policy_learning_ready": policy_learning_ready,
        "transfer_consumer_ready": intish(summary.get("sts_row_count")) >= 8,
        "consumer": "policy traces -> legal-action masking / delayed reward capsules / route memory",
    }


def lane_conversation(report: Any) -> dict[str, Any]:
    summary = summary_of(report)
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "case_count": intish(summary.get("case_count")),
        "accuracy": floatish(summary.get("accuracy")),
        "passed_cases": intish(summary.get("passed_cases")),
        "graduated": bool(summary.get("graduated")),
        "consumer": "conversation traces -> working-state narration / correction memory / dashboard chat transfer",
    }


def lane_sts(report: Any, ablation: Any, control: Any) -> dict[str, Any]:
    summary = summary_of(report)
    causal = summary.get("causal_transfer") if isinstance(summary.get("causal_transfer"), dict) else {}
    ablation_summary = summary_of(ablation)
    control_consumers = (
        get_path(control, ["consumer_contract", "consumers"], [])
        if isinstance(control, dict)
        else []
    )
    control_effects = (
        get_path(control, ["consumer_contract", "effects"], [])
        if isinstance(control, dict)
        else []
    )
    control_rows = intish(control.get("control_rows_written")) if isinstance(control, dict) else 0
    control_rows_path = str(control.get("control_rows_path") or "") if isinstance(control, dict) else ""
    control_contract_ready = bool(
        isinstance(control, dict)
        and control.get("trigger_state") in {"GREEN", "YELLOW"}
        and control_rows > 0
        and len(control_consumers if isinstance(control_consumers, list) else []) >= 3
    )
    public_ablation = (
        ablation_summary.get("public") if isinstance(ablation_summary.get("public"), dict) else {}
    )
    public_groups = public_ablation.get("groups") if isinstance(public_ablation.get("groups"), dict) else {}
    ablation_sts = (
        public_groups.get("sts_conditioned")
        if isinstance(public_groups.get("sts_conditioned"), dict)
        else {}
    )
    decoder_count = max(
        intish(causal.get("decoder_sts_conditioned_candidate_count")),
        intish(ablation_sts.get("row_count")),
    )
    private_ablation = (
        ablation_summary.get("private") if isinstance(ablation_summary.get("private"), dict) else {}
    )
    private_groups = private_ablation.get("groups") if isinstance(private_ablation.get("groups"), dict) else {}
    private_sts = (
        private_groups.get("sts_conditioned")
        if isinstance(private_groups.get("sts_conditioned"), dict)
        else {}
    )
    private_decoder_count = intish(private_sts.get("row_count"))
    ablation_green = ablation.get("trigger_state") == "GREEN" if isinstance(ablation, dict) else False
    ablation_named_effect = bool(
        ablation_green
        and ablation_summary.get("decoder_gate_ready") is True
        and ablation_summary.get("same_seed_non_sts_comparator_present") is True
        and ablation_summary.get("sts_positive_same_seed_lift") is True
        and ablation_summary.get("sts_coverage_non_regressive") is True
        and ablation_summary.get("sts_conditioning_regressed_candidate_coverage") is not True
        and ablation_summary.get("sts_control_contract_ready") is True
        and control_contract_ready
        and private_decoder_count > 0
    )
    capsule_named_effect = bool(causal.get("measured_transfer_effect")) and decoder_count > 0 and ablation_green
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "sts_causal_ablation_trigger_state": ablation.get("trigger_state") if isinstance(ablation, dict) else None,
        "capsule_count": intish(summary.get("capsule_count")),
        "measured_transfer_effect": bool(causal.get("measured_transfer_effect")) or ablation_named_effect,
        "decoder_sts_conditioned_candidate_count": decoder_count,
        "private_decoder_sts_conditioned_candidate_count": private_decoder_count,
        "same_seed_causal_ablation_green": ablation_green,
        "sts_pass_rate_delta": floatish(causal.get("sts_pass_rate_delta")),
        "sts_candidate_distribution_delta": floatish(ablation_summary.get("sts_candidate_distribution_delta")),
        "sts_positive_same_seed_lift": bool(ablation_summary.get("sts_positive_same_seed_lift")),
        "sts_coverage_non_regressive": bool(ablation_summary.get("sts_coverage_non_regressive")),
        "control_contract_ready": control_contract_ready,
        "control_trigger_state": control.get("trigger_state") if isinstance(control, dict) else None,
        "control_rows_written": control_rows,
        "control_rows_path": control_rows_path,
        "control_consumer_count": len(control_consumers) if isinstance(control_consumers, list) else 0,
        "control_effect_count": len(control_effects) if isinstance(control_effects, list) else 0,
        "control_consumers": control_consumers if isinstance(control_consumers, list) else [],
        "named_consumer_effect": capsule_named_effect or ablation_named_effect,
        "named_consumer_effect_source": "cross_domain_capsules" if capsule_named_effect else ("sts_causal_decoder_ablation" if ablation_named_effect else ""),
        "consumer": "decoder candidate distribution / SymLiquid routing / retry policy",
    }


def lane_public_transfer(report: Any) -> dict[str, Any]:
    summary = summary_of(report)
    return {
        "trigger_state": report.get("trigger_state") if isinstance(report, dict) else None,
        "aggregate_pass_rate": floatish(summary.get("aggregate_pass_rate")),
        "transfer_ready": bool(summary.get("transfer_ready")),
        "weak_cards": summary.get("weak_cards") if isinstance(summary.get("weak_cards"), list) else [],
    }


def next_actions(summary: dict[str, Any]) -> list[str]:
    actions = []
    if not summary["repo_repair"]["transfer_consumer_ready"]:
        actions.append("Run repo repair as a private eval lane with a named downstream consumer and decoder/transfer-proof delta, not just row generation.")
    if not summary["sts_consumption"]["named_consumer_effect"]:
        actions.append("Run a fresh STS causal decoder ablation after the patched decoder closure completes.")
    if not summary["public_transfer"]["transfer_ready"]:
        actions.append("Keep public calibration locked until decoder_v2 gate and private_public_transfer_proof are both GREEN.")
    if not summary["conversation"]["graduated"]:
        actions.append("Escrow the one hard_v4 conversation failure and generate hard_v5 targeted cases.")
    return actions or ["Keep lanes in regression and rotate to the next unsolved transfer wall."]


def summary_of(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        return payload["summary"]
    return payload if isinstance(payload, dict) else {}


def get_path(payload: Any, path: list[str], default: Any = None) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def intish(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def floatish(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
