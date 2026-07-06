"""Code Residual Forge v1 for Project Theseus.

The forge converts code-frontier benchmark output into reusable learning
pressure: residual classes, repair traces, synthesized test sketches, transfer
packets, and same-family rotation hints. It does not call external inference
and it does not grow the student.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "code_residual_forge_policy.json"
DEFAULT_OUT = ROOT / "reports" / "code_residual_forge.json"
DEFAULT_TRANSFER_OUT = ROOT / "reports" / "code_transfer_artifacts.json"
DEFAULT_ROTATION_OUT = ROOT / "reports" / "code_frontier_rotation.json"
DEFAULT_TRACE_OUT = ROOT / "reports" / "code_repair_traces.jsonl"
DEFAULT_ARTIFACT_DIR = ROOT / "reports" / "transfer_artifacts" / "code"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--benchmark-ledger", default="reports/benchmark_ledger.json")
    parser.add_argument("--curriculum", default="reports/benchmaxx_curriculum.json")
    parser.add_argument("--frontier-policy", default="reports/frontier_policy_status.json")
    parser.add_argument("--profile-report", default="reports/training_ratchet_profile_run.json")
    parser.add_argument("--tool-registry", default="reports/tool_registry.json")
    parser.add_argument("--active-card-id", default="")
    parser.add_argument("--active-report", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--transfer-out", default=str(DEFAULT_TRANSFER_OUT.relative_to(ROOT)))
    parser.add_argument("--rotation-out", default=str(DEFAULT_ROTATION_OUT.relative_to(ROOT)))
    parser.add_argument("--trace-out", default=str(DEFAULT_TRACE_OUT.relative_to(ROOT)))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    ledger = read_json(ROOT / args.benchmark_ledger, [])
    curriculum = read_json(ROOT / args.curriculum, {})
    frontier_policy = read_json(ROOT / args.frontier_policy, {})
    profile = read_json(ROOT / args.profile_report, {})
    tool_registry = read_json(ROOT / args.tool_registry, {})

    reports = collect_code_reports(
        ledger=ledger,
        profile=profile,
        frontier_policy=frontier_policy,
        curriculum=curriculum,
        policy=policy,
        active_report=args.active_report,
    )
    active = active_context(
        reports=reports,
        ledger=ledger,
        profile=profile,
        frontier_policy=frontier_policy,
        curriculum=curriculum,
        policy=policy,
        active_card_id=args.active_card_id,
        active_report=args.active_report,
    )
    observations = build_observations(reports, active_card_id=str(active.get("card_id") or ""))
    clusters = cluster_observations(observations)
    traces = build_repair_traces(observations, clusters)
    synthesized_tests = synthesize_test_templates(clusters)
    prompt_program_sketches = build_prompt_program_sketches(clusters)
    rotation = rotation_decision(
        policy=policy,
        ledger=ledger,
        curriculum=curriculum,
        active=active,
        clusters=clusters,
    )
    transfer = write_transfer_artifacts(
        policy=policy,
        active=active,
        reports=reports,
        clusters=clusters,
        traces=traces,
        synthesized_tests=synthesized_tests,
        prompt_program_sketches=prompt_program_sketches,
        out_path=ROOT / args.transfer_out,
        artifact_dir=ROOT / args.artifact_dir,
    )
    write_trace_jsonl(ROOT / args.trace_out, traces)
    write_json(ROOT / args.rotation_out, rotation)

    gates = [
        gate("code_reports_loaded", bool(reports), f"reports={len(reports)}"),
        gate("residuals_classified", bool(observations) and bool(clusters), f"observations={len(observations)} clusters={len(clusters)}"),
        gate("repair_traces_written", (ROOT / args.trace_out).exists(), args.trace_out),
        gate("transfer_artifacts_written", bool(transfer.get("artifacts")), f"artifacts={len(transfer.get('artifacts', []))}"),
        gate("rotation_decision_written", bool(rotation.get("decision")), rotation.get("decision")),
        gate("external_inference_zero", True, "forge reads local reports only"),
        gate("student_growth_blocked", not bool(policy.get("model_growth_allowed")), "code forge is a cheaper intervention"),
    ]
    trigger_state = "GREEN" if all(item["passed"] for item in gates) else "RED"
    report = {
        "policy": "project_theseus_code_residual_forge_report_v1",
        "created_utc": now(),
        "config": args.policy,
        "trigger_state": trigger_state,
        "summary": {
            "family": "coding_local_sandbox",
            "report_count": len(reports),
            "observation_count": len(observations),
            "cluster_count": len(clusters),
            "active_card_id": active.get("card_id"),
            "active_report": active.get("report_path"),
            "active_score": active.get("score"),
            "active_floor": active.get("floor"),
            "active_attempt_count": active.get("attempt_count"),
            "dominant_residual_class": clusters[0]["category"] if clusters else None,
            "transfer_artifacts": len(transfer.get("artifacts", [])),
            "rotation_decision": rotation.get("decision"),
            "selected_card_id": rotation.get("selected_card_id"),
            "cheaper_interventions_exhausted": False,
        },
        "active_context": active,
        "failure_clusters": clusters,
        "synthesized_tests": synthesized_tests,
        "prompt_program_sketches": prompt_program_sketches,
        "rotation": rotation,
        "transfer_artifacts": transfer,
        "tool_registry_context": {
            "tool_count": len(tool_registry.get("tools", [])) if isinstance(tool_registry.get("tools"), list) else 0,
            "registry_health": tool_registry.get("registry_health") if isinstance(tool_registry, dict) else {},
        },
        "gates": gates,
        "artifacts": {
            "trace_jsonl": args.trace_out,
            "transfer_artifacts": args.transfer_out,
            "rotation": args.rotation_out,
        },
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state != "RED" else 1


def collect_code_reports(
    *,
    ledger: Any,
    profile: dict[str, Any],
    frontier_policy: dict[str, Any],
    curriculum: dict[str, Any],
    policy: dict[str, Any],
    active_report: str = "",
) -> list[dict[str, Any]]:
    ordered = [str(item) for item in policy.get("ordered_card_ids", []) if str(item)]
    paths: list[Path] = []
    if active_report:
        paths.append(ROOT / active_report)
    real_code_graduation = ROOT / "reports" / "real_code_benchmark_graduation.json"
    if real_code_graduation.exists():
        paths.append(real_code_graduation)
    profile_path = str(get_path(profile, ["artifacts", "pressure_runner"], "") or "")
    if profile_path:
        paths.append(ROOT / profile_path)
    for row in ledger if isinstance(ledger, list) else []:
        if not isinstance(row, dict):
            continue
        best = str(row.get("best_report") or row.get("report") or "")
        if best and is_code_row(row, best, ordered):
            paths.append(ROOT / best)
    active_best = str(get_path(frontier_policy, ["frontier", "best_report"], "") or "")
    if active_best:
        paths.append(ROOT / active_best)
    selected = str(get_path(curriculum, ["next_frontier", "same_family_rotation", "selected_card_id"], "") or "")
    if selected:
        paths.extend((ROOT / "reports").glob(f"pressure_{safe_card(selected)}_seed*.json"))
    for card_id in ordered:
        paths.extend((ROOT / "reports").glob(f"pressure_{safe_card(card_id)}_seed*.json"))

    deduped: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = read_json(path, {})
        if not isinstance(data, dict) or not data:
            continue
        if data.get("frontier_family") != "coding_local_sandbox":
            continue
        key = str(path.resolve()).lower()
        deduped[key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
            "mtime": path.stat().st_mtime if path.exists() else 0,
            "data": data,
        }
    return sorted(deduped.values(), key=lambda item: (str(item["data"].get("card_id") or ""), -float(item.get("mtime") or 0)))


def active_context(
    *,
    reports: list[dict[str, Any]],
    ledger: Any,
    profile: dict[str, Any],
    frontier_policy: dict[str, Any],
    curriculum: dict[str, Any],
    policy: dict[str, Any],
    active_card_id: str = "",
    active_report: str = "",
) -> dict[str, Any]:
    profile_report = str(active_report or get_path(profile, ["artifacts", "pressure_runner"], "") or "")
    profile_card = str(get_path(profile, ["pressure_card_id"], "") or "")
    if not profile_card and profile_report:
        match = re.search(r"pressure_(source_[^_]+(?:_[^_]+)*)_seed", profile_report.replace("\\", "/"))
        profile_card = match.group(1) if match else ""
    # The profile report can lag after a promotion rotates the active coding
    # frontier. Treat frontier_policy_status as the source of truth so the
    # forge does not keep exporting stale transfer artifacts for the last run.
    policy_card = str(
        frontier_policy.get("pressure_card_id")
        or get_path(frontier_policy, ["frontier_pressure", "next_pressure_card_id"], "")
        or ""
    )
    curriculum_card = str(get_path(curriculum, ["next_frontier", "same_family_rotation", "selected_card_id"], "") or "")
    card_id = str(
        active_card_id
        or policy_card
        or curriculum_card
        or profile_card
        or ""
    )
    preferred_report = profile_report if profile_card == card_id else ""
    report = report_for_card(reports, card_id, preferred_path=preferred_report)
    if not card_id and report:
        card_id = str(get_path(report, ["data", "card_id"], "") or "")
    row = ledger_row_for_card(ledger, card_id)
    report_score = number(get_path(report, ["data", "summary", "accuracy"], None), default=None)
    score = report_score if report_score is not None else (number(row.get("score"), default=None) if row else None)
    floor = number(get_path(row, ["graduation_policy", "floor_threshold"], None), default=number(policy.get("floor_threshold"), default=0.70))
    attempt_count = int(number(get_path(row, ["graduation_policy", "attempt_count"], 0), default=0))
    stalled_cycles = int(number(get_path(row, ["graduation_policy", "stalled_cycles"], 0), default=0))
    return {
        "family": "coding_local_sandbox",
        "card_id": card_id,
        "report_path": report.get("path") if report else profile_report,
        "benchmark_name": row.get("benchmark_name") if row else get_path(report, ["data", "summary", "suite"], ""),
        "score": round(score, 6) if score is not None else None,
        "floor": round(floor, 6) if floor is not None else None,
        "attempt_count": attempt_count,
        "stalled_cycles": stalled_cycles,
        "below_floor": bool(score is not None and floor is not None and score < floor),
        "above_floor": bool(score is not None and floor is not None and score >= floor),
    }


def build_observations(reports: list[dict[str, Any]], *, active_card_id: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for report in reports:
        path = str(report["path"])
        data = report["data"]
        card_id = str(data.get("card_id") or "")
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        for residual in data.get("residuals", []) if isinstance(data.get("residuals"), list) else []:
            if not isinstance(residual, dict):
                continue
            observations.append(
                observation(
                    source_report=path,
                    card_id=card_id,
                    category=classify_text(str(residual.get("type") or ""), str(residual.get("detail") or "")),
                    residual_type=str(residual.get("type") or "residual"),
                    detail=str(residual.get("detail") or ""),
                    active=card_id == active_card_id,
                )
            )
        for check in data.get("checks", []) if isinstance(data.get("checks"), list) else []:
            if not isinstance(check, dict) or check.get("passed") is not False:
                continue
            observations.append(
                observation(
                    source_report=path,
                    card_id=card_id,
                    category=classify_text(str(check.get("name") or ""), str(check.get("evidence") or "")),
                    residual_type=f"failed_check:{check.get('name') or 'unknown'}",
                    detail=str(check.get("evidence") or ""),
                    active=card_id == active_card_id,
                )
            )
        for item in metrics.get("details", []) if isinstance(metrics.get("details"), list) else []:
            if not isinstance(item, dict) or item.get("passed") is not False:
                continue
            observations.append(
                observation(
                    source_report=path,
                    card_id=card_id,
                    category=classify_text(str(item.get("task") or ""), f"{item.get('stderr') or ''} {item.get('error') or ''}"),
                    residual_type=f"failed_task:{item.get('task') or 'unknown'}",
                    detail=str(item.get("stderr") or item.get("error") or item),
                    active=card_id == active_card_id,
                )
            )
        if not data.get("residuals") and not observations_for_report(observations, path):
            status = str(data.get("status") or "")
            if status and status not in {"frontier_open", "passed", "ok"}:
                observations.append(
                    observation(
                        source_report=path,
                        card_id=card_id,
                        category=classify_text(status, json.dumps(data.get("summary", {}))),
                        residual_type=f"status:{status}",
                        detail=json.dumps(data.get("summary", {}), sort_keys=True),
                        active=card_id == active_card_id,
                    )
                )
    return observations


def cluster_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[str(item["category"])].append(item)
    clusters: list[dict[str, Any]] = []
    for category, rows in grouped.items():
        cards = sorted({str(row.get("card_id") or "") for row in rows if row.get("card_id")})
        residual_types = Counter(str(row.get("residual_type") or "") for row in rows)
        active_count = sum(1 for row in rows if row.get("active"))
        clusters.append(
            {
                "category": category,
                "count": len(rows),
                "active_count": active_count,
                "cards": cards,
                "residual_types": [
                    {"type": name, "count": count}
                    for name, count in residual_types.most_common(8)
                ],
                "priority": round(len(rows) + active_count * 1.5 + category_priority(category), 4),
                "suggested_intervention": suggested_intervention(category),
                "examples": rows[:5],
            }
        )
    clusters.sort(key=lambda row: (-float(row.get("priority") or 0), str(row.get("category") or "")))
    return clusters


def build_repair_traces(observations: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {str(row["category"]): index for index, row in enumerate(clusters)}
    traces: list[dict[str, Any]] = []
    for idx, item in enumerate(sorted(observations, key=lambda row: (priority.get(str(row["category"]), 99), str(row["card_id"]), str(row["residual_type"])))):
        traces.append(
            {
                "trace_id": f"code_trace_{idx + 1:04d}",
                "created_utc": now(),
                "source_report": item["source_report"],
                "card_id": item["card_id"],
                "category": item["category"],
                "residual_type": item["residual_type"],
                "detail": item["detail"][:1000],
                "suggested_test": suggested_test(item["category"]),
                "repair_pattern": repair_pattern(item["category"]),
                "transfer_hint": transfer_hint(item["category"], item["card_id"]),
                "loads_into": ["code_repair_arm", "pressure_runner", "benchmark_adapter_factory"],
            }
        )
    return traces


def synthesize_test_templates(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for cluster in clusters[:8]:
        category = str(cluster.get("category") or "")
        templates.append(
            {
                "category": category,
                "name": f"code_residual_{category}_template",
                "purpose": suggested_test(category),
                "template": test_template(category),
                "source_cards": cluster.get("cards", []),
                "risk": "low",
            }
        )
    return templates


def build_prompt_program_sketches(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sketches: list[dict[str, Any]] = []
    for cluster in clusters[:8]:
        category = str(cluster.get("category") or "")
        sketches.append(
            {
                "category": category,
                "sketch": prompt_program_sketch(category),
                "expected_effect": repair_pattern(category),
                "loads_into": ["code_repair_arm", "octopus_router"],
            }
        )
    return sketches


def rotation_decision(
    *,
    policy: dict[str, Any],
    ledger: Any,
    curriculum: dict[str, Any],
    active: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = ready_order(policy, curriculum)
    current_id = str(active.get("card_id") or "")
    if current_id not in ordered and ordered:
        current_id = ordered[0]
    attempts_before = int(number(policy.get("below_floor_attempts_before_rotate"), default=2))
    stalled_before = int(number(policy.get("stalled_cycles_before_rotate"), default=2))
    below_floor = bool(active.get("below_floor"))
    attempts = int(number(active.get("attempt_count"), default=0))
    stalled = int(number(active.get("stalled_cycles"), default=0))
    active_cluster_count = sum(int(row.get("active_count") or 0) for row in clusters)
    rotate_due = below_floor and (
        attempts >= attempts_before
        or stalled >= stalled_before
        or active_cluster_count >= 2
    )
    selected = current_id
    reason = "continue_current_card"
    if rotate_due and len(ordered) > 1:
        selected = next_open_card(ordered, current_id, ledger) or current_id
        reason = "code_residual_forge_rotate_below_floor"
    elif below_floor:
        reason = "continue_until_code_rotation_threshold"
    elif active.get("above_floor"):
        reason = "continue_current_card_above_floor"
    return {
        "policy": "project_theseus_code_frontier_rotation_hint_v1",
        "created_utc": now(),
        "family": "coding_local_sandbox",
        "decision": "rotate" if selected and selected != current_id else "continue",
        "reason": reason,
        "current_card_id": current_id,
        "selected_card_id": selected,
        "return_queue": rotation_queue_after(ordered, selected)[:8],
        "ready_order": ordered[:20],
        "below_floor_attempts_before_rotate": attempts_before,
        "stalled_cycles_before_rotate": stalled_before,
        "active_below_floor": below_floor,
        "active_attempt_count": attempts,
        "active_stalled_cycles": stalled,
        "active_cluster_count": active_cluster_count,
        "external_inference_calls": 0,
    }


def write_transfer_artifacts(
    *,
    policy: dict[str, Any],
    active: dict[str, Any],
    reports: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    synthesized_tests: list[dict[str, Any]],
    prompt_program_sketches: list[dict[str, Any]],
    out_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    loads_into = [str(item) for item in policy.get("loads_into", []) if str(item)]
    card_ids = sorted({str(report["data"].get("card_id") or "") for report in reports if report["data"].get("card_id")})
    artifacts: list[dict[str, Any]] = []
    for card_id in card_ids or [str(active.get("card_id") or "code_frontier")]:
        card_clusters = [cluster for cluster in clusters if card_id in (cluster.get("cards") or [])]
        card_traces = [trace for trace in traces if trace.get("card_id") == card_id][:30]
        payload = {
            "policy": "project_theseus_code_transfer_artifact_v1",
            "created_utc": now(),
            "family": "coding_local_sandbox",
            "card_id": card_id,
            "active_card": card_id == active.get("card_id"),
            "summary": {
                "cluster_count": len(card_clusters),
                "trace_count": len(card_traces),
                "dominant_residual_class": card_clusters[0]["category"] if card_clusters else None,
            },
            "failure_clusters": card_clusters,
            "repair_traces": card_traces,
            "synthesized_tests": [
                row for row in synthesized_tests if card_id in row.get("source_cards", [])
            ],
            "prompt_program_sketches": prompt_program_sketches,
            "loads_into": loads_into,
            "verification": {
                "external_inference_calls": 0,
                "student_growth": "forbidden",
                "source_reports": [
                    report["path"] for report in reports if report["data"].get("card_id") == card_id
                ],
            },
        }
        artifact_path = artifact_dir / f"{safe_card(card_id)}_transfer_artifact.json"
        write_json(artifact_path, payload)
        artifacts.append(
            {
                "name": f"code_residual_transfer_{card_id}",
                "family": "coding_local_sandbox",
                "card_id": card_id,
                "path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
                "loads_into": loads_into,
                "cluster_count": len(card_clusters),
                "trace_count": len(card_traces),
                "active_card": card_id == active.get("card_id"),
            }
        )
    artifacts = artifacts + preserved_external_transfer_artifacts(out_path, artifacts, loads_into)
    cluster_count = sum(int(item.get("cluster_count") or 0) for item in artifacts)
    trace_count = sum(int(item.get("trace_count") or 0) for item in artifacts)
    index = {
        "policy": "project_theseus_code_transfer_artifacts_index_v1",
        "created_utc": now(),
        "summary": {
            "frontier_family": "coding_local_sandbox",
            "active_card_id": active.get("card_id"),
            "artifact_count": len(artifacts),
            "cluster_count": cluster_count,
            "trace_count": trace_count,
            "loads_into": loads_into,
        },
        "artifacts": artifacts,
        "external_inference_calls": 0,
    }
    write_json(out_path, index)
    return index


def preserved_external_transfer_artifacts(
    out_path: Path,
    generated_artifacts: list[dict[str, Any]],
    default_loads_into: list[str],
) -> list[dict[str, Any]]:
    existing = read_json(out_path, {})
    generated_paths = {str(item.get("path") or "") for item in generated_artifacts}
    preserved: list[dict[str, Any]] = []
    for item in existing.get("artifacts", []) if isinstance(existing.get("artifacts"), list) else []:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or "")
        if not rel_path or rel_path in generated_paths:
            continue
        artifact_path = ROOT / rel_path
        payload = read_json(artifact_path, {})
        if not payload:
            continue
        if payload.get("policy") == "project_theseus_code_transfer_artifact_v1":
            continue
        failure_clusters = payload.get("failure_clusters") if isinstance(payload.get("failure_clusters"), list) else []
        repair_traces = payload.get("repair_traces") if isinstance(payload.get("repair_traces"), list) else []
        preserved.append(
            {
                "name": item.get("name") or f"external_code_transfer_{safe_card(str(payload.get('card_id') or artifact_path.stem))}",
                "family": item.get("family") or payload.get("family") or "coding_local_sandbox",
                "card_id": item.get("card_id") or payload.get("card_id"),
                "path": rel_path,
                "loads_into": item.get("loads_into") or payload.get("loads_into") or default_loads_into,
                "cluster_count": len(failure_clusters),
                "trace_count": len(repair_traces),
                "active_card": bool(item.get("active_card")) or bool(payload.get("active_card")),
                "preserved_external_artifact": True,
            }
        )
    return preserved


def observation(
    *,
    source_report: str,
    card_id: str,
    category: str,
    residual_type: str,
    detail: str,
    active: bool,
) -> dict[str, Any]:
    return {
        "source_report": source_report,
        "card_id": card_id,
        "category": category,
        "residual_type": residual_type,
        "detail": detail,
        "active": bool(active),
    }


def observations_for_report(observations: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    return [row for row in observations if row.get("source_report") == path]


def classify_text(name: str, detail: str) -> str:
    text = f"{name} {detail}".lower()
    if any(token in text for token in ["syntaxerror", "indentation", "parse", "manifest_locator", "json decode"]):
        return "parsing"
    if any(token in text for token in ["typeerror", "attributeerror", "none", "null", "schema", "typing"]):
        return "type_handling"
    if any(token in text for token in ["edge", "boundary", "corner", "empty", "single", "negative", "overflow"]):
        return "edge_case"
    if any(token in text for token in ["wrong answer", "assertion", "expected", "algorithm", "complexity"]):
        return "algorithm_choice"
    if any(token in text for token in ["hidden", "contamination", "public calibration", "private", "leakage"]):
        return "hidden_tests"
    if any(token in text for token in ["mastered", "increase heldout", "diversity"]):
        return "mastered_needs_diversity"
    if any(token in text for token in ["endpoint", "provider", "tool", "harness", "agent"]):
        return "tool_use_failure"
    if any(token in text for token in ["adapter_needed", "repair", "generated-code", "sandboxed generated", "local_code_generation"]):
        return "repair_loop"
    if any(token in text for token in ["timeout", "timed out", "hang"]):
        return "timeout"
    if any(token in text for token in ["docker", "podman", "bun", "module", "dependency", "runtime missing", "source_not_staged"]):
        return "dependency_issue"
    if any(token in text for token in ["license", "audit", "uncertain", "integrity"]):
        return "benchmark_integrity"
    return "repair_loop"


def suggested_intervention(category: str) -> str:
    return {
        "parsing": "Add manifest/task parser contracts and schema smoke tests before scoring.",
        "edge_case": "Synthesize boundary tests and feed them into residual-targeted repair traces.",
        "algorithm_choice": "Add algorithm-family hints and complexity checks to the code repair arm.",
        "type_handling": "Add type/null/shape guards to generated-code verification.",
        "hidden_tests": "Separate public calibration from private residual tests and preserve contamination audit.",
        "repair_loop": "Wire local student output into sandboxed repair/eval harness and save patches.",
        "tool_use_failure": "Build deterministic local endpoint adapter and provider-off task runner.",
        "timeout": "Chunk runner into resumable budgets with partial evidence and retry envelopes.",
        "dependency_issue": "Keep source staged but gate full harness until runtime dependency is present.",
        "benchmark_integrity": "Hold as audit-only pressure until license/source/task integrity is verified.",
        "mastered_needs_diversity": "Increase heldout task diversity and move mastered seed to regression.",
    }.get(category, "Classify as repair-loop residual and create a deterministic local test.")


def suggested_test(category: str) -> str:
    return {
        "parsing": "Given a source repo, locate tasks/manifests and reject ambiguous schema with a typed error.",
        "edge_case": "Run empty, singleton, duplicate, negative, and large-input cases against each generated repair.",
        "algorithm_choice": "Compare simple and optimized algorithms against correctness plus timeout budget.",
        "type_handling": "Exercise None/null, mixed numeric types, nested containers, and missing fields.",
        "hidden_tests": "Verify public examples and private residual cases are separated in artifacts.",
        "repair_loop": "Generate a patch, execute tests in a sandbox, and store stdout/stderr plus diff metadata.",
        "tool_use_failure": "Run provider-off local tasks through the endpoint adapter with deterministic fixtures.",
        "timeout": "Force a small timeout and require partial report, checkpoint, and resumable retry metadata.",
        "dependency_issue": "Probe required local runtime and emit blocked-but-staged contract without scoring.",
        "benchmark_integrity": "Verify license/source provenance and task split before enabling train/eval pressure.",
        "mastered_needs_diversity": "Add unseen task families to heldout pool and require regression preservation.",
    }.get(category, "Create a local deterministic smoke test for this residual class.")


def repair_pattern(category: str) -> str:
    return {
        "parsing": "schema-first task loading",
        "edge_case": "boundary-case augmentation",
        "algorithm_choice": "algorithm template selection",
        "type_handling": "typed guard synthesis",
        "hidden_tests": "split-aware eval discipline",
        "repair_loop": "patch-test-escrow loop",
        "tool_use_failure": "local harness adapter",
        "timeout": "resumable budget chunking",
        "dependency_issue": "runtime readiness contract",
        "benchmark_integrity": "source truth audit",
        "mastered_needs_diversity": "heldout diversity expansion",
    }.get(category, "local repair verification")


def transfer_hint(category: str, card_id: str) -> str:
    return f"Load {repair_pattern(category)} before the next {card_id or 'code'} run and compare residual reduction."


def test_template(category: str) -> str:
    return {
        "parsing": "assert locate_tasks(repo).status in {'ready', 'blocked_with_reason'}",
        "edge_case": "for case in boundary_cases: assert candidate(case.input) == case.expected",
        "algorithm_choice": "assert runtime_ms(candidate, stress_case) <= budget_ms and output_ok",
        "type_handling": "for case in null_and_type_cases: assert no_type_crash(candidate, case)",
        "hidden_tests": "assert public_ids.isdisjoint(private_residual_ids)",
        "repair_loop": "patch = propose_patch(task); assert sandbox_run(patch).report_written",
        "tool_use_failure": "assert local_provider_off_harness(task).external_calls == 0",
        "timeout": "assert timed_run(task).partial_report and timed_run(task).resume_token",
        "dependency_issue": "assert runtime_probe(tool).status in {'available', 'blocked_with_contract'}",
        "benchmark_integrity": "assert source.license_allowed and source.split_integrity_ok",
        "mastered_needs_diversity": "assert heldout_family_count >= previous_family_count + 1",
    }.get(category, "assert local_residual_replay(category).report_written")


def prompt_program_sketch(category: str) -> str:
    return {
        "parsing": "Read task schema first; if fields are missing, emit a typed blocked report instead of guessing.",
        "edge_case": "Before final answer, enumerate boundary cases and run them as residual probes.",
        "algorithm_choice": "Choose the simplest algorithm that clears both examples and generated stress cases.",
        "type_handling": "Guard container shape, nullability, and numeric/string coercions explicitly.",
        "hidden_tests": "Never train on public calibration labels; route unknown failures into private residual escrow.",
        "repair_loop": "Always produce patch, run tests, record trace, then reuse failures as future training cases.",
        "tool_use_failure": "Use only local provider-off fixtures and deterministic endpoint adapters.",
        "timeout": "Checkpoint after each task chunk and resume from saved evidence.",
        "dependency_issue": "Probe runtime dependency before execution and classify missing tools as blocked pressure.",
        "benchmark_integrity": "Audit license, source provenance, and split boundaries before enabling scoring.",
        "mastered_needs_diversity": "Retain mastered task as regression and add harder unseen families.",
    }.get(category, "Convert failure into a deterministic residual replay before retrying.")


def category_priority(category: str) -> float:
    return {
        "repair_loop": 5.0,
        "tool_use_failure": 4.5,
        "hidden_tests": 4.0,
        "benchmark_integrity": 3.8,
        "dependency_issue": 3.0,
        "timeout": 2.8,
        "algorithm_choice": 2.6,
        "edge_case": 2.4,
        "type_handling": 2.2,
        "parsing": 2.0,
        "mastered_needs_diversity": 1.8,
    }.get(category, 1.0)


def report_for_card(reports: list[dict[str, Any]], card_id: str, *, preferred_path: str = "") -> dict[str, Any]:
    if preferred_path:
        normalized = preferred_path.replace("\\", "/").lower()
        for report in reports:
            if str(report.get("path") or "").replace("\\", "/").lower() == normalized:
                return report
    matches = [report for report in reports if str(report["data"].get("card_id") or "") == card_id]
    if not matches:
        return reports[0] if reports else {}
    return max(matches, key=lambda item: float(item.get("mtime") or 0))


def ledger_row_for_card(ledger: Any, card_id: str) -> dict[str, Any]:
    rows = []
    for row in ledger if isinstance(ledger, list) else []:
        if not isinstance(row, dict):
            continue
        haystack = f"{row.get('benchmark_name') or ''} {row.get('best_report') or ''}"
        if card_id and card_id in haystack:
            rows.append(row)
    if not rows:
        return {}
    return max(rows, key=lambda row: int(number(get_path(row, ["graduation_policy", "attempt_count"], 0), default=0)))


def next_open_card(ordered: list[str], current_id: str, ledger: Any) -> str:
    for card_id in rotation_queue_after(ordered, current_id):
        row = ledger_row_for_card(ledger, card_id)
        if row.get("lifecycle") != "regression":
            return card_id
    return ""


def rotation_queue_after(ordered: list[str], current_id: str) -> list[str]:
    if current_id not in ordered:
        return [card_id for card_id in ordered if card_id != current_id]
    index = ordered.index(current_id)
    return ordered[index + 1 :] + ordered[:index]


def ready_order(policy: dict[str, Any], curriculum: dict[str, Any]) -> list[str]:
    ordered = [str(item) for item in policy.get("ordered_card_ids", []) if str(item)]
    curriculum_order = get_path(curriculum, ["next_frontier", "same_family_rotation", "ready_order"], [])
    if isinstance(curriculum_order, list):
        for item in curriculum_order:
            text = str(item)
            if text and text not in ordered:
                ordered.append(text)
    return ordered


def is_code_row(row: dict[str, Any], best_report: str, ordered: list[str]) -> bool:
    name = str(row.get("benchmark_name") or "")
    return name.startswith("coding_") or any(card_id in best_report for card_id in ordered)


def safe_card(card_id: str) -> str:
    return str(card_id or "code").replace("-", "_").replace("/", "_")


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def number(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def write_trace_jsonl(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in traces)
    path.write_text(text, encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
