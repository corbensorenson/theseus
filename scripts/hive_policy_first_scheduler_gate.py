#!/usr/bin/env python3
"""Policy-first Hive scheduler gate.

Reachability is not authority. This gate evaluates fixture device cards, portal
cards, job contracts, bids, approvals, and federation lease policy without
requiring live peer connectivity.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import viea_spine_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hive_policy_first_scheduler.json"
DEFAULT_REPORT = ROOT / "reports" / "hive_policy_first_scheduler.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "hive_policy_first_scheduler.md"
DEFAULT_BIDS = ROOT / "reports" / "hive_policy_first_bids.jsonl"
DEFAULT_DECISIONS = ROOT / "reports" / "hive_policy_first_decisions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default=rel(DEFAULT_REPORT))
    parser.add_argument("--markdown-out", default=rel(DEFAULT_MARKDOWN))
    parser.add_argument("--bids-out", default=rel(DEFAULT_BIDS))
    parser.add_argument("--decisions-out", default=rel(DEFAULT_DECISIONS))
    args = parser.parse_args()

    started = time.perf_counter()
    config_path = resolve(args.config)
    config = read_json(config_path)
    report = build_report(config_path, config, started)
    write_json(resolve(args.out), report)
    write_jsonl(resolve(args.bids_out), report["bids"])
    write_jsonl(resolve(args.decisions_out), report["decisions"])
    write_text(resolve(args.markdown_out), render_markdown(report))
    print(json.dumps(gate_view(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_report(config_path: Path, config: dict[str, Any], started: float) -> dict[str, Any]:
    devices = list_dicts(config.get("device_resource_cards"))
    jobs = list_dicts(config.get("job_contracts"))
    portals = list_dicts(config.get("portal_cards"))
    approvals = list_dicts(config.get("approval_receipts"))
    spine_receipt = viea_spine_records.materialized_view_consumer_receipt(
        "hive_policy_first_scheduler",
        required_groups=[
            "governance_records",
            "failure_boundaries",
            "authority_records",
            "resource_route_records",
        ],
    )
    boundaries = audit_boundaries(dict_value(config.get("hard_boundaries")), dict_value(config.get("federation_lease_policy")))
    bids = []
    decisions = []
    for job in jobs:
        job_bids = [score_bid(job, device) for device in devices]
        bids.extend(job_bids)
        decisions.append(attach_spine_receipt_to_decision(decide_job(job, job_bids), spine_receipt))
    approval_audit = [audit_approval(row, portals, jobs) for row in approvals]
    hard_gaps = [gate for gate in boundaries if gate["severity"] == "hard" and not gate["passed"]]
    spine_gate = gate(
        "viea_spine_materialized_view_consumed",
        bool(spine_receipt["ready"]),
        "hard",
        spine_receipt,
    )
    if not spine_gate["passed"]:
        hard_gaps.append(spine_gate)
    hard_gaps.extend([row for row in approval_audit if row["severity"] == "hard" and not row["passed"]])
    warnings = [row for row in approval_audit if row["severity"] == "warning" and not row["passed"]]
    for decision in decisions:
        if decision["selected_device_id"] == "" and decision["required"]:
            warnings.append(item_gap(decision["job_id"], "no_eligible_device", decision, severity="warning"))
    trigger_state = "GREEN"
    if hard_gaps:
        trigger_state = "RED"
    elif warnings:
        trigger_state = "YELLOW"
    summary = {
        "config": rel(config_path),
        "device_count": len(devices),
        "job_count": len(jobs),
        "portal_count": len(portals),
        "approval_count": len(approvals),
        "bid_count": len(bids),
        "eligible_bid_count": sum(1 for bid in bids if bid["eligible"]),
        "rejected_bid_count": sum(1 for bid in bids if not bid["eligible"]),
        "decision_count": len(decisions),
        "scheduled_decision_count": sum(1 for row in decisions if row["selected_device_id"]),
        "policy_rejected_faster_or_available_count": sum(1 for bid in bids if (not bid["eligible"]) and bid["reachable"]),
        "federation_leases_enabled_by_default": bool(dict_value(config.get("federation_lease_policy")).get("enabled_by_default")),
        "viea_spine_view_ready": spine_receipt["ready"],
        "viea_spine_view_record_count": spine_receipt["record_count"],
        "viea_spine_governance_record_count": spine_receipt["governance_record_count"],
        "viea_spine_failure_boundary_count": spine_receipt["failure_boundary_count"],
        "viea_spine_authority_record_count": spine_receipt["authority_record_count"],
        "viea_spine_resource_route_record_count": spine_receipt["resource_route_record_count"],
        "hard_gap_count": len(hard_gaps),
        "warning_count": len(warnings),
    }
    return {
        "policy": "project_theseus_hive_policy_first_scheduler_gate_v1",
        "created_utc": now(),
        "trigger_state": trigger_state,
        "summary": summary,
        "boundary_gates": boundaries,
        "approval_audit": approval_audit,
        "viea_spine_consumer_receipt": spine_receipt,
        "bids": bids,
        "decisions": decisions,
        "hard_gaps": hard_gaps,
        "warnings": warnings,
        "rules": {
            "authority": "Reachability never implies authority; data/tool/physical-risk policy filters bids before scoring.",
            "mobile": "Phone/operator approval receipts bind one job, one permission, and one time window.",
            "roaming": "Roaming profiles may expose status/control paths, but raw public gateway and arbitrary remote shell remain forbidden.",
            "federation": "Federation leases stay disabled by default until sandbox and revocation tests pass.",
            "latency": "Low-latency chat/control prefers nearby operator routes; long independent training may tolerate higher latency only after policy acceptance.",
        },
        "runtime_ms": int((time.perf_counter() - started) * 1000),
    }


def audit_boundaries(boundaries: dict[str, Any], lease_policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        gate("reachability_not_authority", boundaries.get("reachability_implies_authority") is False, "hard", boundaries.get("reachability_implies_authority")),
        gate("arbitrary_remote_shell_forbidden", boundaries.get("arbitrary_remote_shell_allowed") is False, "hard", boundaries.get("arbitrary_remote_shell_allowed")),
        gate("public_gateway_forbidden", boundaries.get("public_gateway_allowed") is False, "hard", boundaries.get("public_gateway_allowed")),
        gate("raw_private_data_to_untrusted_forbidden", boundaries.get("raw_private_data_to_untrusted_node_allowed") is False, "hard", boundaries.get("raw_private_data_to_untrusted_node_allowed")),
        gate("spending_requires_receipt", boundaries.get("spending_without_receipt_allowed") is False, "hard", boundaries.get("spending_without_receipt_allowed")),
        gate("federation_leases_default_disabled", boundaries.get("federation_leases_default_enabled") is False and lease_policy.get("enabled_by_default") is False, "hard", {"boundary": boundaries.get("federation_leases_default_enabled"), "policy": lease_policy.get("enabled_by_default")}),
    ]


def score_bid(job: dict[str, Any], device: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    job_id = str(job.get("id") or "")
    device_id = str(device.get("id") or "")
    reachable = bool(device.get("reachable"))
    if not reachable:
        reasons.append("not_reachable_currently")
    if str(job.get("data_class") or "") not in list_values(device.get("data_classes_allowed")):
        reasons.append("data_class_not_allowed")
    if str(job.get("tool_class") or "") not in list_values(device.get("tool_classes_allowed")):
        reasons.append("tool_class_not_allowed")
    required_accel = str(job.get("requires_accelerator") or "")
    if required_accel and required_accel not in list_values(device.get("accelerators")):
        reasons.append("required_accelerator_missing")
    if bool(job.get("requires_operator_presence")) and not bool(device.get("operator_presence_required")) and "operator" not in list_values(device.get("roles")):
        reasons.append("operator_presence_missing")
    if bool(job.get("physical_risk")) and not bool(device.get("physical_risk_allowed")):
        reasons.append("physical_risk_forbidden")
    latency = int(device.get("latency_ms") or 9999)
    if latency > int(job.get("max_latency_ms") or 999999):
        reasons.append("latency_too_high")
    if str(device.get("battery_state") or "") == "mobile_battery" and str(job.get("latency_class") or "") == "batch":
        reasons.append("mobile_battery_batch_work_forbidden")
    if str(device.get("thermal_state") or "") in {"hot", "critical"}:
        reasons.append("thermal_state_blocks_work")
    eligible = not reasons
    score = 0.0
    if eligible:
        score = round(100.0 - min(95.0, latency / 10.0), 4)
        if required_accel:
            score += 20.0
        if "storage" in list_values(device.get("roles")):
            score += 5.0
        if str(device.get("battery_state") or "") in {"plugged_in", "mains"}:
            score += 5.0
    return {
        "job_id": job_id,
        "device_id": device_id,
        "reachable": reachable,
        "eligible": eligible,
        "reject_reasons": reasons,
        "score": round(score, 4),
        "latency_ms": latency,
        "trust_tier": str(device.get("trust_tier") or ""),
        "accelerators": list_values(device.get("accelerators")),
    }


def decide_job(job: dict[str, Any], bids: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sorted([bid for bid in bids if bid["eligible"]], key=lambda row: (-row["score"], row["latency_ms"], row["device_id"]))
    selected = eligible[0] if eligible else {}
    rejected_reachable = [bid for bid in bids if bid["reachable"] and not bid["eligible"]]
    return {
        "job_id": str(job.get("id") or ""),
        "required": True,
        "selected_device_id": str(selected.get("device_id") or ""),
        "selected_score": selected.get("score"),
        "eligible_bid_count": len(eligible),
        "policy_rejected_reachable_bid_count": len(rejected_reachable),
        "rejected_reachable_examples": rejected_reachable[:4],
        "approval_required": bool(job.get("approval_required")),
        "lease_scope": "single_job_registered_task_only" if selected else "",
    }


def attach_spine_receipt_to_decision(decision: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    out = dict(decision)
    out["viea_spine_policy_receipt_id"] = receipt.get("receipt_id")
    out["viea_spine_view_ready"] = bool(receipt.get("ready"))
    out["policy_evidence_groups"] = {
        "governance_records": receipt.get("governance_record_count"),
        "failure_boundaries": receipt.get("failure_boundary_count"),
        "authority_records": receipt.get("authority_record_count"),
        "resource_route_records": receipt.get("resource_route_record_count"),
    }
    return out


def audit_approval(row: dict[str, Any], portals: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    portal_ids = {str(portal.get("id") or ""): portal for portal in portals}
    job_ids = {str(job.get("id") or "") for job in jobs}
    approval_id = str(row.get("id") or "<missing-id>")
    portal = portal_ids.get(str(row.get("portal_id") or ""))
    passed = True
    reason = ""
    severity = "hard"
    if portal is None:
        passed = False
        reason = "portal_missing"
    elif str(row.get("job_id") or "") not in job_ids:
        passed = False
        reason = "job_missing"
    elif str(row.get("permission") or "") not in list_values(portal.get("allowed_permissions")):
        passed = False
        reason = "permission_not_allowed_by_portal"
    elif int(row.get("valid_seconds") or 0) > int(portal.get("max_approval_window_seconds") or 0):
        passed = False
        reason = "approval_window_too_long"
    elif row.get("bound_to_single_job") is not True or row.get("revocable") is not True:
        passed = False
        reason = "approval_not_single_job_or_not_revocable"
    return {
        "id": approval_id,
        "kind": "approval_receipt_bound",
        "passed": passed,
        "severity": severity,
        "evidence": {"reason": reason, "approval": row},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hive Policy-First Scheduler",
        "",
        f"- trigger_state: `{report['trigger_state']}`",
        f"- devices: `{report['summary']['device_count']}` jobs: `{report['summary']['job_count']}`",
        f"- eligible bids: `{report['summary']['eligible_bid_count']}` rejected bids: `{report['summary']['rejected_bid_count']}`",
        f"- scheduled decisions: `{report['summary']['scheduled_decision_count']}`",
        f"- policy-rejected reachable bids: `{report['summary']['policy_rejected_faster_or_available_count']}`",
        f"- VIEA spine view ready: `{report['summary']['viea_spine_view_ready']}` records `{report['summary']['viea_spine_view_record_count']}`",
        f"- hard gaps: `{report['summary']['hard_gap_count']}` warnings: `{report['summary']['warning_count']}`",
        "",
        "## Decisions",
        "",
    ]
    for decision in report["decisions"]:
        lines.append(f"- `{decision['job_id']}` -> `{decision['selected_device_id']}` eligible=`{decision['eligible_bid_count']}` rejected_reachable=`{decision['policy_rejected_reachable_bid_count']}`")
    lines.extend(["", "## Hard Gaps", ""])
    if report["hard_gaps"]:
        for item in report["hard_gaps"]:
            lines.append(f"- `{item['id']}` `{item['kind']}`: `{json.dumps(item['evidence'], sort_keys=True)}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Rules", ""])
    for key, value in report["rules"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def gate_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": report["policy"],
        "created_utc": report["created_utc"],
        "trigger_state": report["trigger_state"],
        "summary": report["summary"],
        "hard_gaps": report["hard_gaps"],
        "warnings": report["warnings"],
    }


def gate(name: str, passed: bool, severity: str, evidence: Any) -> dict[str, Any]:
    return {"id": name, "kind": name, "passed": bool(passed), "severity": severity, "evidence": evidence}


def item_gap(item_id: str, kind: str, evidence: dict[str, Any], severity: str = "hard") -> dict[str, Any]:
    return {"id": item_id, "kind": kind, "passed": False, "severity": severity, "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_dicts(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def list_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
