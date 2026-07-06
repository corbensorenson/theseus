"""Deterministic Assembly-Theoretic Technical Debt analyzer.

ATTD is a repo-native governance layer for autonomous source evolution. It is
not a replacement for tests or benchmarks; it measures whether the construction
history and current structure are becoming too costly for safe autonomous
growth.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "attd_policy.json"


@dataclass(frozen=True)
class FileMetric:
    path: str
    role: str
    extension: str
    line_count: int
    nonblank_lines: int
    bytes: int
    function_count: int
    class_count: int
    import_count: int
    todo_count: int
    max_nesting: int
    avg_line_length: float
    signature_hash: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--project-registry", default="reports/theseus_project_registry.json")
    parser.add_argument("--out", default="reports/attd_report.json")
    parser.add_argument("--packets-out", default="reports/attd_maintenance_packets.json")
    parser.add_argument("--markdown-out", default="reports/attd_report.md")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    files = collect_files(policy)
    metrics, motifs = analyze_files(files, policy)
    history = analyze_history(policy)
    role_report = analyze_roles(metrics, policy)
    components = compute_components(metrics, motifs, role_report, history, policy)
    hard_caps = evaluate_hard_caps(metrics, motifs, role_report, components, history, policy)
    score = weighted_score(components, policy)
    trigger_state = trigger_state_for(score, hard_caps, history, policy)
    project_registry = read_json(resolve_path(args.project_registry))
    packets = build_maintenance_packets(metrics, motifs, role_report, history, components, hard_caps, trigger_state, policy)
    packets.extend(build_project_registry_packets(project_registry))

    report = {
        "policy": "sparkstream_attd_report_v0",
        "created_utc": now(),
        "policy_file": args.policy,
        "analyzer_version": policy.get("analyzer_version", "attd_analyzer_v0"),
        "determinism_contract": {
            "pure_passes": True,
            "stable_ordering": True,
            "learned_classifiers": 0,
            "seed": 0,
            "abstain_on_unknown_role": True,
        },
        "scope": {
            "files_analyzed": len(metrics),
            "tracked_like_files_seen": len(files),
            "include_paths": policy.get("include_paths", []),
            "include_extensions": policy.get("include_extensions", []),
        },
        "trigger_state": trigger_state,
        "attd_score": round(score, 6),
        "components": components,
        "hard_caps": hard_caps,
        "roles": role_report,
        "project_registry": compact_project_registry(project_registry, args.project_registry),
        "motifs": summarize_motifs(motifs),
        "history": history,
        "governance": governance_decision(trigger_state, policy),
        "top_hotspots": top_hotspots(metrics, role_report),
        "maintenance_packets_path": args.packets_out,
        "external_inference_calls": 0,
    }
    packets_report = {
        "policy": "sparkstream_attd_maintenance_packets_v0",
        "created_utc": report["created_utc"],
        "trigger_state": trigger_state,
        "attd_score": report["attd_score"],
        "packet_count": len(packets),
        "packets": packets,
        "external_inference_calls": 0,
    }
    write_json(ROOT / args.out, report)
    write_json(ROOT / args.packets_out, packets_report)
    write_markdown(ROOT / args.markdown_out, report, packets_report)
    print(json.dumps(report, indent=2))
    return 0 if trigger_state != "RED" else 2


def collect_files(policy: dict[str, Any]) -> list[Path]:
    tracked = git_lines(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    include_paths = tuple(normalize_path(str(item)).rstrip("/") for item in policy.get("include_paths", []))
    include_ext = set(str(item).lower() for item in policy.get("include_extensions", []))
    exclude = [normalize_path(str(item)) for item in policy.get("exclude_globs", [])]
    paths: list[Path] = []
    for raw in tracked:
        rel = normalize_path(raw)
        path = ROOT / rel
        if not path.is_file():
            continue
        if include_paths and not any(rel == prefix or rel.startswith(prefix + "/") for prefix in include_paths):
            continue
        if include_ext and path.suffix.lower() not in include_ext:
            continue
        if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: normalize_path(str(item.relative_to(ROOT))))


def analyze_files(paths: list[Path], policy: dict[str, Any]) -> tuple[list[FileMetric], dict[str, dict[str, Any]]]:
    metrics: list[FileMetric] = []
    motif_window = int(get_path(policy, ["normalization", "motif_window_lines"], 3))
    motifs: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "files": set(), "sample": ""})
    for path in paths:
        rel = normalize_path(str(path.relative_to(ROOT)))
        text = read_text(path)
        if text is None:
            continue
        lines = text.splitlines()
        normalized = [normalize_source_line(line) for line in lines]
        normalized_nonempty = [line for line in normalized if line]
        for idx in range(0, max(0, len(normalized_nonempty) - motif_window + 1)):
            window = "\n".join(normalized_nonempty[idx : idx + motif_window])
            digest = stable_hash(window)
            motifs[digest]["count"] += 1
            motifs[digest]["files"].add(rel)
            if not motifs[digest]["sample"]:
                motifs[digest]["sample"] = window[:240]
        metrics.append(file_metric(path, rel, text, lines, policy))
    finalized = {
        key: {"count": value["count"], "files": sorted(value["files"]), "sample": value["sample"]}
        for key, value in motifs.items()
    }
    return metrics, finalized


def file_metric(path: Path, rel: str, text: str, lines: list[str], policy: dict[str, Any]) -> FileMetric:
    nonblank = [line for line in lines if line.strip()]
    signatures = signature_lines(text, path.suffix.lower())
    return FileMetric(
        path=rel,
        role=classify_role(rel, policy),
        extension=path.suffix.lower(),
        line_count=len(lines),
        nonblank_lines=len(nonblank),
        bytes=len(text.encode("utf-8")),
        function_count=count_functions(text, path.suffix.lower()),
        class_count=count_classes(text, path.suffix.lower()),
        import_count=count_imports(text, path.suffix.lower()),
        todo_count=len(re.findall(r"\b(TODO|FIXME|HACK|XXX)\b", text, flags=re.IGNORECASE)),
        max_nesting=max_nesting(lines),
        avg_line_length=mean([len(line) for line in nonblank]) if nonblank else 0.0,
        signature_hash=stable_hash("\n".join(signatures[:100])),
    )


def classify_role(rel: str, policy: dict[str, Any]) -> str:
    for role in policy.get("roles", []):
        if not isinstance(role, dict):
            continue
        for pattern in role.get("patterns", []):
            if fnmatch.fnmatch(rel, normalize_path(str(pattern))):
                return str(role.get("id") or "unknown")
    return "unassigned"


def analyze_roles(metrics: list[FileMetric], policy: dict[str, Any]) -> dict[str, Any]:
    by_role: dict[str, list[FileMetric]] = defaultdict(list)
    for item in metrics:
        by_role[item.role].append(item)
    role_weights = {
        str(role.get("id")): float(role.get("criticality", 0.5))
        for role in policy.get("roles", [])
        if isinstance(role, dict)
    }
    rows = []
    weighted_total = 0.0
    weight_sum = 0.0
    for role, items in sorted(by_role.items()):
        entropy = role_entropy(items)
        weight = role_weights.get(role, 0.35)
        weighted_total += entropy * weight * len(items)
        weight_sum += weight * len(items)
        rows.append(
            {
                "role": role,
                "files": len(items),
                "criticality": round(weight, 3),
                "entropy": round(entropy, 6),
                "largest_files": [
                    {"path": item.path, "lines": item.line_count}
                    for item in sorted(items, key=lambda row: (-row.line_count, row.path))[:5]
                ],
                "extensions": dict(sorted(Counter(item.extension or "<none>" for item in items).items())),
            }
        )
    assigned = len([item for item in metrics if item.role != "unassigned"])
    return {
        "coverage_ratio": round(assigned / max(1, len(metrics)), 6),
        "global_pattern_entropy": round(weighted_total / max(1.0, weight_sum), 6),
        "roles": rows,
    }


def role_entropy(items: list[FileMetric]) -> float:
    if len(items) < 2:
        return 0.0
    vectors = [
        [
            math.log1p(item.nonblank_lines),
            math.log1p(item.function_count),
            math.log1p(item.class_count),
            math.log1p(item.import_count),
            item.max_nesting / 10.0,
            item.avg_line_length / 120.0,
        ]
        for item in items
    ]
    dispersion = mean(coefficient_of_variation([vector[idx] for vector in vectors]) for idx in range(len(vectors[0])))
    ext_entropy = normalized_entropy([item.extension for item in items])
    dir_entropy = normalized_entropy([item.path.split("/")[0] if "/" in item.path else item.path for item in items])
    signature_diversity = len({item.signature_hash for item in items}) / max(1, len(items))
    return clamp(0.5 * dispersion + 0.2 * ext_entropy + 0.15 * dir_entropy + 0.15 * signature_diversity)


def analyze_history(policy: dict[str, Any]) -> dict[str, Any]:
    window = int(get_path(policy, ["normalization", "history_commit_window"], 24))
    log_lines = git_lines(["git", "log", f"-n{window}", "--numstat", "--format=--COMMIT--%H"])
    additions = 0
    deletions = 0
    files_touched: set[str] = set()
    commits = 0
    for line in log_lines:
        if line.startswith("--COMMIT--"):
            commits += 1
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            additions += int(parts[0])
            deletions += int(parts[1])
            files_touched.add(normalize_path(parts[2]))
    dirty_lines = git_lines(["git", "diff", "--numstat"])
    dirty_additions = 0
    dirty_deletions = 0
    dirty_files: set[str] = set()
    for line in dirty_lines:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            dirty_additions += int(parts[0])
            dirty_deletions += int(parts[1])
            dirty_files.add(normalize_path(parts[2]))
    committed_net_growth = max(0, additions - deletions)
    committed_simplification_credit = max(0, min(deletions, (deletions - additions) * 0.8))
    committed_unreconciled_reductive = max(0.0, deletions - committed_simplification_credit)
    committed_event_load = len(files_touched) * 8
    committed_destabilizing_load = additions + committed_unreconciled_reductive + committed_event_load
    committed_residue = max(0.0, committed_destabilizing_load - (committed_net_growth * 0.65))
    constructive_load = additions + dirty_additions
    reductive_load = deletions + dirty_deletions
    net_growth = max(0, constructive_load - reductive_load)
    simplification_credit = max(0, min(reductive_load, (reductive_load - constructive_load) * 0.8))
    unreconciled_reductive = max(0.0, reductive_load - simplification_credit)
    event_load = len(files_touched | dirty_files) * 8 + len(dirty_files) * 12
    destabilizing_load = constructive_load + unreconciled_reductive + event_load
    residue = max(0.0, destabilizing_load - (net_growth * 0.65))
    dirty_constructive = dirty_additions
    dirty_reductive = dirty_deletions
    dirty_net_growth = max(0, dirty_constructive - dirty_reductive)
    dirty_event_load = len(dirty_files) * 20
    dirty_residue = max(0.0, dirty_constructive + dirty_reductive + dirty_event_load - (dirty_net_growth * 0.65))
    red_lines = float(get_path(policy, ["normalization", "rolling_residue_red_lines"], 9000))
    return {
        "commit_window": window,
        "commits_observed": commits,
        "additions": additions,
        "deletions": deletions,
        "files_touched": len(files_touched),
        "dirty_additions": dirty_additions,
        "dirty_deletions": dirty_deletions,
        "dirty_files": sorted(dirty_files),
        "committed_rolling_assembly_residue": round(committed_residue, 3),
        "committed_rolling_residue_score": round(clamp(committed_residue / max(1.0, red_lines)), 6),
        "constructive_load": round(constructive_load, 3),
        "reductive_load": round(reductive_load, 3),
        "verified_simplification_credit_proxy": round(simplification_credit, 3),
        "rolling_assembly_residue": round(residue, 3),
        "rolling_residue_score": round(clamp(residue / max(1.0, red_lines)), 6),
        "dirty_assembly_residue": round(dirty_residue, 3),
        "dirty_residue_score": round(clamp(dirty_residue / max(1.0, red_lines)), 6),
        "workspace_dirty_assembly_residue": round(dirty_residue, 3),
        "workspace_dirty_residue_score": round(clamp(dirty_residue / max(1.0, red_lines)), 6),
        "dirty_residue_governance": "technical_debt_gate"
        if get_path(policy, ["workspace_checkpoint", "dirty_residue_is_debt_gate"], False)
        else "auto_checkpoint_provenance_hygiene",
    }


def compute_components(
    metrics: list[FileMetric],
    motifs: dict[str, dict[str, Any]],
    role_report: dict[str, Any],
    history: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, float]:
    total_lines = sum(item.nonblank_lines for item in metrics)
    total_files = len(metrics)
    large_limit = int(get_path(policy, ["normalization", "large_file_lines"], 1800))
    preferred_limit = int(get_path(policy, ["normalization", "preferred_file_lines"], 900))
    large_ratio = len([item for item in metrics if item.line_count > preferred_limit]) / max(1, total_files)
    deep_nesting = mean([clamp(item.max_nesting / float(get_path(policy, ["normalization", "max_nesting_red"], 10))) for item in metrics]) if metrics else 0.0
    size_pressure = scaled(total_lines, get_path(policy, ["normalization", "green_effective_lines"], 25000), get_path(policy, ["normalization", "red_effective_lines"], 115000))
    file_pressure = scaled(total_files, get_path(policy, ["normalization", "green_file_count"], 80), get_path(policy, ["normalization", "red_file_count"], 420))
    intrinsic = clamp(0.42 * size_pressure + 0.25 * large_ratio + 0.2 * deep_nesting + 0.13 * file_pressure)

    total_windows = sum(int(item["count"]) for item in motifs.values())
    supported = [item for item in motifs.values() if int(item["count"]) >= int(get_path(policy, ["normalization", "motif_min_support"], 2))]
    repeated_windows = sum(int(item["count"]) for item in supported)
    repeated_coverage = repeated_windows / max(1, total_windows)
    duplicate_density = raw_duplicate_density(supported, total_windows)
    reuse_failure = clamp(0.7 * max(0.0, 0.52 - repeated_coverage) / 0.52 + 0.3 * duplicate_density)

    pattern_entropy = float(role_report.get("global_pattern_entropy") or 0.0)
    dirty_is_debt_gate = bool(get_path(policy, ["workspace_checkpoint", "dirty_residue_is_debt_gate"], False))
    rolling_key = "rolling_residue_score" if dirty_is_debt_gate else "committed_rolling_residue_score"
    rolling_residue = float(history.get(rolling_key) or 0.0)
    debt_pressure = compute_debt_pressure(metrics, policy)
    return {
        "intrinsic_assembly_burden": round(intrinsic, 6),
        "reuse_failure": round(reuse_failure, 6),
        "pattern_entropy": round(pattern_entropy, 6),
        "rolling_residue": round(rolling_residue, 6),
        "debt_pressure": round(debt_pressure, 6),
        "motif_repeated_coverage": round(repeated_coverage, 6),
        "duplicate_density": round(duplicate_density, 6),
    }


def compute_debt_pressure(metrics: list[FileMetric], policy: dict[str, Any]) -> float:
    source = [item for item in metrics if item.extension in {".py", ".rs", ".js", ".ps1"}]
    tests = [item for item in metrics if item.role == "tests" or "test" in item.path.lower()]
    todo_count = sum(item.todo_count for item in metrics)
    large_limit = int(get_path(policy, ["normalization", "large_file_lines"], 1800))
    large_source_ratio = len([item for item in source if item.line_count > large_limit]) / max(1, len(source))
    todo_pressure = clamp(todo_count / float(get_path(policy, ["normalization", "debt_todo_red_count"], 80)))
    test_ratio = len(tests) / max(1, len(source))
    test_pressure = clamp(max(0.0, float(get_path(policy, ["normalization", "test_ratio_green"], 0.18)) - test_ratio) / 0.18)
    import_pressure = mean([clamp(item.import_count / 45.0) for item in source]) if source else 0.0
    boundary_count = boundary_violations(metrics)
    boundary_pressure = clamp(boundary_count / max(1.0, float(get_path(policy, ["hard_caps", "max_boundary_violation_count"], 8))))
    return clamp(0.25 * large_source_ratio + 0.22 * todo_pressure + 0.22 * test_pressure + 0.16 * import_pressure + 0.15 * boundary_pressure)


def boundary_violations(metrics: list[FileMetric]) -> int:
    count = 0
    for item in metrics:
        if item.path.startswith("scripts/") and item.role not in {"data_governance", "benchmark_adapter"}:
            # Report writes are expected; direct reads from ignored private asset areas are more suspicious.
            text = read_text(ROOT / item.path) or ""
            if re.search(r"data/(local_roms|external_benchmark_candidates)|games/", text):
                count += 1
    return count


def evaluate_hard_caps(
    metrics: list[FileMetric],
    motifs: dict[str, dict[str, Any]],
    role_report: dict[str, Any],
    components: dict[str, float],
    history: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    caps = policy.get("hard_caps", {})
    source_ext = {".py", ".rs", ".js", ".ps1"}
    largest_source = max((item for item in metrics if item.extension in source_ext), key=lambda row: row.line_count, default=None)
    max_role = max((row for row in role_report.get("roles", []) if isinstance(row, dict)), key=lambda row: float(row.get("entropy") or 0.0), default={})
    violations = []
    dirty_is_debt_gate = bool(get_path(policy, ["workspace_checkpoint", "dirty_residue_is_debt_gate"], False))
    dirty_score = float(history.get("dirty_residue_score") or 0.0)
    dirty_cap = float(caps.get("max_dirty_residue_score", 0.92))
    checks = [
        (
            "max_source_file_lines",
            (largest_source.line_count if largest_source else 0) <= int(caps.get("max_source_file_lines", 3600)),
            {"path": largest_source.path if largest_source else "", "lines": largest_source.line_count if largest_source else 0, "cap": caps.get("max_source_file_lines")},
        ),
        (
            "max_role_entropy",
            float(max_role.get("entropy") or 0.0) <= float(caps.get("max_role_entropy", 0.92)),
            {"role": max_role.get("role"), "entropy": max_role.get("entropy"), "cap": caps.get("max_role_entropy")},
        ),
        (
            "max_dirty_residue_score",
            (dirty_score <= dirty_cap) if dirty_is_debt_gate else True,
            {
                "value": history.get("dirty_residue_score"),
                "cap": caps.get("max_dirty_residue_score"),
                "mode": "technical_debt_gate" if dirty_is_debt_gate else "auto_checkpoint_provenance_hygiene",
                "would_fail_if_gate": dirty_score > dirty_cap,
                "note": "Dirty workspace residue is checkpointed before autonomy gates; ATTD blocking is reserved for code shape debt.",
            },
        ),
        (
            "max_debt_pressure",
            components["debt_pressure"] <= float(caps.get("max_debt_pressure", 0.9)),
            {"value": components["debt_pressure"], "cap": caps.get("max_debt_pressure")},
        ),
        (
            "max_duplicate_density",
            components["duplicate_density"] <= float(caps.get("max_duplicate_density", 0.42)),
            {"value": components["duplicate_density"], "cap": caps.get("max_duplicate_density")},
        ),
    ]
    rows = []
    for gate, passed, evidence in checks:
        row = {"gate": gate, "passed": bool(passed), "evidence": evidence}
        rows.append(row)
        if not passed:
            violations.append(row)
    return {
        "passed": not violations,
        "violations": violations,
        "checks": rows,
    }


def trigger_state_for(score: float, hard_caps: dict[str, Any], history: dict[str, Any], policy: dict[str, Any]) -> str:
    thresholds = policy.get("thresholds", {})
    if not hard_caps.get("passed", True):
        return "RED"
    if score >= float(thresholds.get("yellow_max_score", 0.78)):
        return "RED"
    if score >= float(thresholds.get("green_max_score", 0.55)):
        return "YELLOW"
    if float(history.get("rolling_residue_score") or 0.0) >= float(thresholds.get("yellow_growth_rate", 0.12)):
        return "YELLOW"
    return "GREEN"


def weighted_score(components: dict[str, float], policy: dict[str, Any]) -> float:
    weights = policy.get("weights", {})
    total = 0.0
    weight_sum = 0.0
    for key in ["intrinsic_assembly_burden", "reuse_failure", "pattern_entropy", "rolling_residue", "debt_pressure"]:
        weight = float(weights.get(key, 0.2))
        total += float(components.get(key, 0.0)) * weight
        weight_sum += weight
    return clamp(total / max(0.0001, weight_sum))


def governance_decision(trigger_state: str, policy: dict[str, Any]) -> dict[str, Any]:
    cfg = policy.get("governance", {})
    checkpoint_cfg = policy.get("workspace_checkpoint") or {}
    red = trigger_state == "RED"
    yellow = trigger_state == "YELLOW"
    return {
        "allows_long_autonomy": not (red and cfg.get("red_blocks_long_autonomy", True)),
        "allows_teacher_self_edit": not (red and cfg.get("red_blocks_teacher_self_edit_except_attd_maintenance", True)),
        "teacher_self_edit_exception_reason": cfg.get("maintenance_reason", "attd_maintenance") if red else "",
        "allows_architecture_change": not (red and cfg.get("red_blocks_architecture_change", True)),
        "allows_adapter_card_writes": not (red and cfg.get("red_blocks_adapter_card_writes", True)),
        "requires_maintenance_packets": bool(red or (yellow and cfg.get("yellow_requires_maintenance_packets", True))),
        "dirty_workspace_blocks_long_autonomy": bool(checkpoint_cfg.get("dirty_residue_is_debt_gate", False)),
        "auto_commit_dirty_workspace": bool(checkpoint_cfg.get("auto_commit_dirty_workspace", False)),
        "dirty_workspace_mode": str(checkpoint_cfg.get("mode", "auto_checkpoint_provenance_hygiene")),
    }


def build_maintenance_packets(
    metrics: list[FileMetric],
    motifs: dict[str, dict[str, Any]],
    role_report: dict[str, Any],
    history: dict[str, Any],
    components: dict[str, float],
    hard_caps: dict[str, Any],
    trigger_state: str,
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = policy.get("maintenance_packets", {})
    threshold = float(cfg.get("component_warning_threshold", 0.55))
    packets: list[dict[str, Any]] = []
    source_ext = {".py", ".rs", ".js", ".ps1"}
    source_target = int(get_path(policy, ["normalization", "red_file_lines"], 3600))
    oversized_sources = [item for item in metrics if item.extension in source_ext and item.line_count > source_target]
    if oversized_sources:
        largest = sorted(oversized_sources, key=lambda item: (-item.line_count, item.path))[:5]
        packets.append(packet("attd_split_source_over_target", "intrinsic_assembly_burden", 0.78, [item.path for item in largest], f"Split source files above the ATTD target of {source_target} lines into bounded modules without changing public behavior."))
    if components["intrinsic_assembly_burden"] >= threshold:
        largest = sorted(metrics, key=lambda item: (-item.line_count, item.path))[:5]
        packets.append(packet("attd_split_large_or_dense_regions", "intrinsic_assembly_burden", components["intrinsic_assembly_burden"], [item.path for item in largest], "Split or summarize large source regions, preserving public interfaces and tests."))
    if components["reuse_failure"] >= threshold or components["duplicate_density"] >= float(get_path(policy, ["hard_caps", "max_duplicate_density"], 0.42)) * 0.75:
        top = top_supported_motifs(motifs, 5)
        packets.append(packet("attd_consolidate_repeated_motifs", "reuse_failure", components["reuse_failure"], [",".join(item["files"][:3]) for item in top], "Replace repeated local patterns with a shared helper, card template, or documented convention where that improves clarity."))
    if components["pattern_entropy"] >= threshold:
        roles = sorted(role_report.get("roles", []), key=lambda row: (-float(row.get("entropy") or 0.0), str(row.get("role"))))[:4]
        packets.append(packet("attd_converge_role_dialects", "pattern_entropy", components["pattern_entropy"], [str(row.get("role")) for row in roles], "Normalize artifacts serving the same role around one schema, output shape, naming convention, or file layout."))
    if components["rolling_residue"] >= threshold or trigger_state == "RED":
        packets.append(packet("attd_retire_recent_residue", "rolling_residue", components["rolling_residue"], history.get("dirty_files", []), "Audit recent churn, remove transitional wrappers, finish half-migrations, and keep only verified simplifications."))
    if components["debt_pressure"] >= threshold:
        largest = sorted(metrics, key=lambda item: (-(item.todo_count + int(item.line_count > 1200)), item.path))[:5]
        packets.append(packet("attd_reduce_future_change_pressure", "debt_pressure", components["debt_pressure"], [item.path for item in largest], "Reduce future change radius with tests, boundary restoration, smaller modules, or clearer ownership seams."))
    for violation in hard_caps.get("violations", []):
        packets.append(packet(f"attd_hard_cap_{violation.get('gate')}", "hard_cap", 1.0, [json.dumps(violation.get("evidence", {}), sort_keys=True)], "Resolve this hard cap before allowing autonomous growth."))
    max_packets = int(cfg.get("max_packets", 12))
    for index, item in enumerate(packets[:max_packets], start=1):
        item["packet_id"] = f"{item['packet_id']}_{index:02d}"
        item["priority"] = "critical" if trigger_state == "RED" or item["component"] == "hard_cap" else item["priority"]
    return packets[:max_packets]


def build_project_registry_packets(project_registry: dict[str, Any]) -> list[dict[str, Any]]:
    if not project_registry:
        return [
            packet(
                "attd_materialize_project_registry",
                "project_registry",
                0.9,
                ["configs/project_manifest_registry.json", "scripts/theseus_project_registry.py"],
                "Materialize the project manifest registry so ATTD can distinguish active, deprecated, generated, and unregistered surfaces.",
            )
        ]
    summary = project_registry.get("summary") if isinstance(project_registry.get("summary"), dict) else {}
    packets: list[dict[str, Any]] = []
    governance_violations = [row for row in project_registry.get("governance_violations", []) if isinstance(row, dict)]
    abstraction_gaps = [row for row in project_registry.get("abstraction_registry_gaps", []) if isinstance(row, dict)]
    stable_field_gaps = [row for row in project_registry.get("stable_capability_field_gaps", []) if isinstance(row, dict)]
    stable_field_red = [
        row
        for row in project_registry.get("stable_capability_field_health", [])
        if isinstance(row, dict) and row.get("health_state") == "RED"
    ]
    implementation_blockers = [
        row
        for row in project_registry.get("implementation_health", [])
        if isinstance(row, dict) and row.get("routing_required") and not row.get("routing_eligible")
    ]
    cleanup_queue = [row for row in project_registry.get("cleanup_queue", []) if isinstance(row, dict)]
    if governance_violations:
        hard = any(str(row.get("severity") or "") == "hard" for row in governance_violations)
        packets.append(
            packet(
                "attd_resolve_registry_evolution_contract",
                "project_registry",
                0.9 if hard else 0.7,
                [
                    item
                    for row in governance_violations[:8]
                    for item in (row.get("scope") if isinstance(row.get("scope"), list) else [])[:2]
                    if item
                ],
                "Resolve registry evolution contract violations before adding new lanes; improve canonical registered surfaces or declare complete successor/deprecation relationships.",
            )
        )
    if abstraction_gaps or implementation_blockers:
        packets.append(
            packet(
                "attd_registry_abstraction_implementation_health",
                "project_registry",
                0.86,
                [
                    str(row.get("scope") or row.get("implementation_id") or row.get("abstraction_id") or "")
                    for row in (abstraction_gaps[:4] + implementation_blockers[:4])
                    if isinstance(row, dict)
                ],
                "Repair abstraction/implementation contract or routing-health gaps before routers, promotion gates, or autonomy can select those implementations.",
            )
        )
    if stable_field_gaps or stable_field_red:
        packets.append(
            packet(
                "attd_stable_capability_field_health",
                "project_registry",
                0.88,
                [
                    str(row.get("scope") or row.get("abstraction_id") or row.get("implementation_id") or "")
                    for row in (stable_field_gaps[:4] + stable_field_red[:4])
                    if isinstance(row, dict)
                ],
                "Repair SCF semantic-contract, authority/effect, state, evidence, routing, observability, or lifecycle gaps before self-improvement can route through the affected field.",
            )
        )
    unregistered = [row for row in project_registry.get("unregistered", []) if isinstance(row, dict)]
    if unregistered:
        packets.append(
            packet(
                "attd_register_unassigned_surfaces",
                "project_registry",
                0.82 if len(unregistered) >= 25 else 0.62,
                [str(row.get("path") or "") for row in unregistered[:8]],
                "Register unassigned active source/config/doc files under a canonical project surface or move them to deprecated/generated lifecycle state.",
            )
        )
    duplicates = [row for row in project_registry.get("duplicate_families", []) if isinstance(row, dict)]
    if duplicates:
        packets.append(
            packet(
                "attd_consolidate_duplicate_registry_families",
                "project_registry",
                0.72,
                [f"{row.get('root')}/{row.get('family')}" for row in duplicates[:8]],
                "Consolidate duplicate vN/seed/current/after families or mark deliberate compatibility wrappers in the registry.",
            )
        )
    stale_or_missing = [
        row
        for row in project_registry.get("report_outputs", [])
        if isinstance(row, dict) and row.get("status") in {"stale", "missing"}
    ]
    if stale_or_missing:
        packets.append(
            packet(
                "attd_refresh_registry_report_outputs",
                "project_registry",
                0.67,
                [str(row.get("path") or "") for row in stale_or_missing[:8]],
                "Refresh or intentionally retire stale/missing report outputs declared by the project registry.",
            )
        )
    generated_source = [row for row in project_registry.get("generated_source_artifacts", []) if isinstance(row, dict)]
    if generated_source:
        packets.append(
            packet(
                "attd_quarantine_generated_source_artifacts",
                "project_registry",
                0.8,
                [str(row.get("path") or "") for row in generated_source[:8]],
                "Move generated cache/scratch files out of source paths and add ignore coverage when needed.",
            )
        )
    if cleanup_queue:
        packets.append(
            packet(
                "attd_registry_cleanup_queue",
                "project_registry",
                0.7,
                [
                    str(scope)
                    for item in cleanup_queue[:8]
                    for scope in (item.get("scope") if isinstance(item.get("scope"), list) else [])
                    if scope
                ],
                "Work down the registry cleanup queue using improve-existing, replace, retire, archive, or consolidate actions before adding new lanes.",
            )
        )
    if not packets and summary.get("coverage_ratio") is not None:
        return []
    return packets


def packet(packet_id: str, component: str, score: float, scope: list[str], action: str) -> dict[str, Any]:
    return {
        "packet_id": packet_id,
        "component": component,
        "score": round(float(score), 6),
        "priority": "high" if score >= 0.78 else "medium",
        "owner_arm": "rust_cuda_systems_arm" if any(str(path).endswith(".rs") for path in scope) else "loop_closure_tool_arm",
        "scope": scope[:8],
        "bounded_action": action,
        "verification": [
            "Run scripts/attd_analyzer.py and require the targeted component to fall or remain below cap.",
            "Run relevant compile/tests/gates for touched files.",
            "Preserve candidate and regression reports.",
        ],
        "runtime_tier": "E1",
        "risk_tier": "medium",
        "external_inference_calls": 0,
    }


def summarize_motifs(motifs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_windows = sum(int(item["count"]) for item in motifs.values())
    supported = top_supported_motifs(motifs, 10)
    return {
        "motif_count": len(motifs),
        "total_windows": total_windows,
        "supported_motifs": len([item for item in motifs.values() if int(item["count"]) >= 2]),
        "top_supported": supported,
    }


def top_supported_motifs(motifs: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows = [
        {"hash": key, "count": int(value["count"]), "files": value["files"], "sample": value["sample"]}
        for key, value in motifs.items()
        if int(value["count"]) >= 2 and len(value["files"]) >= 2
    ]
    rows.sort(key=lambda row: (-row["count"], row["hash"]))
    return rows[:limit]


def top_hotspots(metrics: list[FileMetric], role_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "largest_files": [
            {"path": item.path, "lines": item.line_count, "role": item.role}
            for item in sorted(metrics, key=lambda row: (-row.line_count, row.path))[:12]
        ],
        "highest_entropy_roles": sorted(
            role_report.get("roles", []),
            key=lambda row: (-float(row.get("entropy") or 0.0), str(row.get("role"))),
        )[:8],
    }


def write_markdown(path: Path, report: dict[str, Any], packets: dict[str, Any]) -> None:
    lines = [
        "# ATTD Governance Report",
        "",
        f"Updated: {report.get('created_utc')}",
        "",
        f"- Trigger state: `{report.get('trigger_state')}`",
        f"- ATTD score: `{report.get('attd_score')}`",
        f"- Maintenance packets: `{packets.get('packet_count')}`",
        "",
        "## Components",
        "",
    ]
    for key, value in (report.get("components") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Hard Caps", ""])
    for check in get_path(report, ["hard_caps", "checks"], []):
        lines.append(f"- {check.get('gate')}: `{check.get('passed')}` {check.get('evidence')}")
    lines.extend(["", "## Maintenance Packets", ""])
    for item in packets.get("packets", []):
        lines.append(f"- {item.get('packet_id')}: {item.get('bounded_action')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_project_registry(project_registry: dict[str, Any], path: str) -> dict[str, Any]:
    summary = project_registry.get("summary") if isinstance(project_registry.get("summary"), dict) else {}
    return {
        "path": path,
        "present": bool(project_registry),
        "trigger_state": project_registry.get("trigger_state") if project_registry else "",
        "surface_count": project_registry.get("surface_count", 0) if project_registry else 0,
        "abstraction_count": summary.get("abstraction_count", 0),
        "implementation_count": summary.get("implementation_count", 0),
        "abstraction_registry_gap_count": summary.get("abstraction_registry_gap_count", 0),
        "implementation_routing_blocker_count": summary.get("implementation_routing_blocker_count", 0),
        "routing_eligible_implementation_count": summary.get("routing_eligible_implementation_count", 0),
        "registry_cleanup_queue_count": summary.get("registry_cleanup_queue_count", 0),
        "learned_generation_claim_allowed_count": summary.get("learned_generation_claim_allowed_count", 0),
        "runtime_serving_allowed_count": summary.get("runtime_serving_allowed_count", 0),
        "entry_count": summary.get("entry_count", 0),
        "coverage_ratio": summary.get("coverage_ratio"),
        "unregistered_active_source_count": summary.get("unregistered_active_source_count", 0),
        "duplicate_family_count": summary.get("duplicate_family_count", 0),
        "source_duplicate_family_count": summary.get("source_duplicate_family_count", 0),
        "classified_source_duplicate_family_count": summary.get("classified_source_duplicate_family_count", 0),
        "unclassified_source_duplicate_family_count": summary.get("unclassified_source_duplicate_family_count", 0),
        "stale_report_output_count": summary.get("stale_report_output_count", 0),
        "missing_report_output_count": summary.get("missing_report_output_count", 0),
        "generated_source_artifact_count": summary.get("generated_source_artifact_count", 0),
        "registry_governance_violation_count": summary.get("registry_governance_violation_count", 0),
        "registry_hard_governance_violation_count": summary.get("registry_hard_governance_violation_count", 0),
    }


def signature_lines(text: str, ext: str) -> list[str]:
    if ext == ".py":
        return re.findall(r"^\s*(?:def|class)\s+[A-Za-z_][A-Za-z0-9_]*.*$", text, flags=re.MULTILINE)
    if ext == ".rs":
        return re.findall(r"^\s*(?:pub\s+)?(?:fn|struct|enum|impl|trait)\s+[A-Za-z_][A-Za-z0-9_]*.*$", text, flags=re.MULTILINE)
    if ext == ".js":
        return re.findall(r"^\s*(?:function|class|const|let)\s+[A-Za-z_][A-Za-z0-9_]*.*$", text, flags=re.MULTILINE)
    return [line for line in text.splitlines() if line.strip().startswith(("#", '"', "{", "["))][:100]


def count_functions(text: str, ext: str) -> int:
    patterns = {
        ".py": r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*",
        ".rs": r"^\s*(pub\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*",
        ".js": r"^\s*(function\s+[A-Za-z_][A-Za-z0-9_]*|const\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\()",
    }
    return len(re.findall(patterns.get(ext, r"$^"), text, flags=re.MULTILINE))


def count_classes(text: str, ext: str) -> int:
    patterns = {
        ".py": r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*",
        ".rs": r"^\s*(pub\s+)?(struct|enum|trait)\s+[A-Za-z_][A-Za-z0-9_]*",
        ".js": r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*",
    }
    return len(re.findall(patterns.get(ext, r"$^"), text, flags=re.MULTILINE))


def count_imports(text: str, ext: str) -> int:
    patterns = {
        ".py": r"^\s*(import|from)\s+",
        ".rs": r"^\s*use\s+",
        ".js": r"^\s*import\s+",
    }
    return len(re.findall(patterns.get(ext, r"$^"), text, flags=re.MULTILINE))


def max_nesting(lines: list[str]) -> int:
    levels = []
    for line in lines:
        stripped = line.lstrip(" ")
        if not stripped:
            continue
        levels.append((len(line) - len(stripped)) // 4)
    return max(levels) if levels else 0


def normalize_source_line(line: str) -> str:
    text = line.strip()
    if not text:
        return ""
    text = re.sub(r"#.*$", "", text)
    text = re.sub(r"//.*$", "", text)
    text = re.sub(r'"[^"]*"', '"S"', text)
    text = re.sub(r"'[^']*'", "'S'", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def raw_duplicate_density(supported: list[dict[str, Any]], total_windows: int) -> float:
    duplicate_excess = sum(max(0, int(item["count"]) - 1) for item in supported if len(item["files"]) >= 2)
    return clamp(duplicate_excess / max(1, total_windows))


def normalized_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return clamp(entropy / math.log(len(counts)))


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    if avg <= 0.000001:
        return 0.0
    return clamp(pstdev(values) / avg)


def scaled(value: float, green: float, red: float) -> float:
    green = float(green)
    red = max(green + 0.0001, float(red))
    return clamp((float(value) - green) / (red - green))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def git_lines(command: list[str]) -> list[str]:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
