"""Build the arm->sucker transfer hierarchy report.

Arms keep reusable, high-transfer capability. Suckers are small loadable
specializations attached to arms or broader suckers for low-transfer game,
simulator, benchmark, or local asset skills.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="configs/arm_sucker_policy.json")
    parser.add_argument("--arm-registry", default="reports/arm_registry.json")
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--transfer-artifacts", default="reports/arm_transfer_artifacts.json")
    parser.add_argument("--out", default="reports/arm_sucker_registry.json")
    parser.add_argument("--markdown-out", default="reports/arm_sucker_registry.md")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    arm_registry = read_json(ROOT / args.arm_registry, {})
    benchmark_ledger = read_json(ROOT / args.benchmark_ledger, [])
    transfer_artifacts = read_json(ROOT / args.transfer_artifacts, {})

    arms = arm_registry.get("arms") if isinstance(arm_registry, dict) else []
    if not isinstance(arms, list):
        arms = []
    known_arms = {str(arm.get("arm_name")) for arm in arms if isinstance(arm, dict)}
    cores = build_cores(policy, known_arms)
    suckers = build_suckers(policy, known_arms, benchmark_ledger, transfer_artifacts)
    matrix = build_transfer_matrix(cores, suckers)
    report = {
        "policy": "project_theseus_arm_sucker_registry_v0",
        "created_utc": now(),
        "config": args.policy,
        "summary": build_summary(cores, suckers),
        "rules": policy.get("rules", {}),
        "cores": cores,
        "suckers": suckers,
        "transfer_matrix": matrix,
        "routing_contracts": [routing_contract(row, suckers) for row in suckers],
        "maintenance_packets": maintenance_packets(cores, suckers),
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_text(ROOT / args.markdown_out, render_markdown(report))
    print(json.dumps(report, indent=2))
    return 0


def build_cores(policy: dict[str, Any], known_arms: set[str]) -> list[dict[str, Any]]:
    rows = []
    for core in policy.get("cores", []) if isinstance(policy.get("cores"), list) else []:
        if not isinstance(core, dict):
            continue
        arm = str(core.get("arm_name") or "")
        rows.append(
            {
                "arm_name": arm,
                "status": "ready" if arm in known_arms else "missing_parent_arm",
                "transfer_role": core.get("transfer_role", "high"),
                "permission_domain": core.get("permission_domain"),
                "sucker_cap": int(core.get("sucker_cap") or 0),
                "shared_skills": core.get("shared_skills") if isinstance(core.get("shared_skills"), list) else [],
                "loads_with": core.get("loads_with") if isinstance(core.get("loads_with"), list) else [],
            }
        )
    return rows


def build_suckers(
    policy: dict[str, Any],
    known_arms: set[str],
    benchmark_ledger: Any,
    transfer_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    sucker_ids = {
        str(row.get("sucker_id"))
        for row in policy.get("suckers", [])
        if isinstance(row, dict) and row.get("sucker_id")
    }
    rows = []
    for spec in policy.get("suckers", []) if isinstance(policy.get("suckers"), list) else []:
        if not isinstance(spec, dict):
            continue
        parent = str(spec.get("parent_arm") or "")
        parent_sucker = str(spec.get("parent_sucker_id") or "")
        required = [str(item) for item in spec.get("required_reports", []) if item]
        artifacts = expand_artifacts([str(item) for item in spec.get("artifact_globs", []) if item])
        score = score_for_sucker(spec, benchmark_ledger)
        missing = [path for path in required if not (ROOT / path).exists()]
        status = "ready"
        if parent not in known_arms:
            status = "missing_parent_arm"
        elif parent_sucker and parent_sucker not in sucker_ids:
            status = "missing_parent_sucker"
        elif missing:
            status = "blocked_missing_reports"
        maturity = maturity_score(status, score, artifacts)
        rows.append(
            {
                "sucker_id": str(spec.get("sucker_id") or ""),
                "parent_arm": parent,
                "parent_sucker_id": parent_sucker,
                "scope": spec.get("scope", ""),
                "transfer_role": spec.get("transfer_role", "low"),
                "frontier_families": spec.get("frontier_families", []),
                "routing_keywords": spec.get("routing_keywords", []),
                "required_reports": required,
                "missing_reports": missing,
                "artifacts": artifacts,
                "latest_score": score,
                "maturity": maturity,
                "status": status,
                "load_policy": spec.get("load_policy", "on_demand"),
                "unload_policy": spec.get("unload_policy", "after_task_boundary"),
                "loads_into": load_chain(str(spec.get("sucker_id") or ""), parent, parent_sucker, sucker_ids),
                "external_inference_calls": 0,
            }
        )
    return rows


def load_chain(sucker_id: str, parent_arm: str, parent_sucker: str, known_suckers: set[str]) -> list[str]:
    chain = [parent_arm]
    if parent_sucker and parent_sucker in known_suckers:
        chain.append(parent_sucker)
    chain.append(sucker_id)
    return [item for item in chain if item]


def expand_artifacts(patterns: list[str]) -> list[str]:
    rows: list[str] = []
    for pattern in patterns:
        rows.extend(rel(Path(path)) for path in glob.glob(str(ROOT / pattern)))
    return sorted(set(rows))


def score_for_sucker(spec: dict[str, Any], ledger: Any) -> float | None:
    families = [str(item) for item in spec.get("frontier_families", []) if item]
    keywords = [str(item).lower().replace("source_", "") for item in spec.get("routing_keywords", []) if item]
    if not isinstance(ledger, list):
        return None
    best: float | None = None
    for row in ledger:
        if not isinstance(row, dict):
            continue
        name = str(row.get("benchmark_name") or "").lower()
        family_hit = any(family.lower() in name for family in families)
        keyword_hit = any(keyword and keyword.replace(" ", "_") in name for keyword in keywords)
        if not family_hit and not keyword_hit:
            continue
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        best = score if best is None else max(best, score)
    return best


def maturity_score(status: str, score: float | None, artifacts: list[str]) -> float:
    if status != "ready":
        return 0.15
    base = 0.35
    if score is not None:
        base += min(0.4, max(0.0, score) * 0.4)
    if artifacts:
        base += 0.15
    return round(min(0.95, base), 4)


def build_transfer_matrix(cores: list[dict[str, Any]], suckers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for sucker in suckers:
        by_parent.setdefault(str(sucker.get("parent_arm")), []).append(sucker)
    rows = []
    for core in cores:
        parent = str(core.get("arm_name") or "")
        children = by_parent.get(parent, [])
        rows.append(
            {
                "from": parent,
                "to": [child["sucker_id"] for child in children],
                "transfer_strength": "high_to_medium",
                "shared_skills": core.get("shared_skills", []),
                "rule": "parent arm skills load before sucker-specific policy/adapters",
            }
        )
        for child in children:
            siblings = [row["sucker_id"] for row in children if row["sucker_id"] != child["sucker_id"]]
            rows.append(
                {
                    "from": child["sucker_id"],
                    "to": siblings,
                    "transfer_strength": "sibling_low_to_medium",
                    "rule": "share traces and priors only through the parent arm schema",
                }
            )
    return rows


def routing_contract(sucker: dict[str, Any], suckers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sucker_id": sucker.get("sucker_id"),
        "parent_arm": sucker.get("parent_arm"),
        "load_sequence": sucker.get("loads_into", []),
        "routing_keywords": sucker.get("routing_keywords", []),
        "load_policy": sucker.get("load_policy"),
        "unload_policy": sucker.get("unload_policy"),
        "memory_namespace": f"arm/{sucker.get('parent_arm')}/sucker/{sucker.get('sucker_id')}",
        "writes_allowed": [
            "trace_summaries",
            "policy_priors",
            "residual_clusters",
            "adapter_smoke_results"
        ],
    }


def build_summary(cores: list[dict[str, Any]], suckers: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in suckers if row.get("status") == "ready"]
    blocked = [row for row in suckers if row.get("status") != "ready"]
    parents = sorted({str(row.get("parent_arm")) for row in suckers if row.get("parent_arm")})
    return {
        "core_count": len(cores),
        "sucker_count": len(suckers),
        "ready_suckers": len(ready),
        "blocked_suckers": len(blocked),
        "parent_arms_with_suckers": parents,
        "average_sucker_maturity": round(sum(float(row.get("maturity") or 0.0) for row in suckers) / max(1, len(suckers)), 4),
        "ready_for_transfer_routing": bool(cores and ready),
        "top_ready_suckers": [row["sucker_id"] for row in sorted(ready, key=lambda item: -float(item.get("maturity") or 0.0))[:6]],
    }


def maintenance_packets(cores: list[dict[str, Any]], suckers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packets = []
    for core in cores:
        if core.get("status") != "ready":
            packets.append(
                {
                    "priority": "high",
                    "kind": "missing_parent_arm",
                    "target": core.get("arm_name"),
                    "action": "register parent arm before relying on attached suckers",
                }
            )
    for sucker in suckers:
        if sucker.get("status") != "ready":
            packets.append(
                {
                    "priority": "medium",
                    "kind": sucker.get("status"),
                    "target": sucker.get("sucker_id"),
                    "action": "satisfy required reports/artifacts or keep sucker disabled",
                    "missing_reports": sucker.get("missing_reports", []),
                }
            )
    return packets[:12]


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Arm Sucker Registry",
        "",
        f"Generated: `{report.get('created_utc')}`",
        "",
        "## Summary",
        "",
        f"- Cores: `{summary.get('core_count')}`",
        f"- Suckers: `{summary.get('sucker_count')}`",
        f"- Ready suckers: `{summary.get('ready_suckers')}`",
        f"- Blocked suckers: `{summary.get('blocked_suckers')}`",
        f"- Average maturity: `{summary.get('average_sucker_maturity')}`",
        "",
        "## Ready Suckers",
        "",
    ]
    for row in report.get("suckers", []):
        if row.get("status") != "ready":
            continue
        lines.append(
            f"- `{row.get('sucker_id')}` on `{row.get('parent_arm')}` "
            f"maturity `{row.get('maturity')}` score `{row.get('latest_score')}`"
        )
    lines.extend(["", "## Maintenance Packets", ""])
    for packet in report.get("maintenance_packets", []):
        lines.append(f"- `{packet.get('priority')}` `{packet.get('target')}`: {packet.get('action')}")
    lines.append("")
    return "\n".join(lines)


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


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
