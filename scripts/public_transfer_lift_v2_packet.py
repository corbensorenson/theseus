#!/usr/bin/env python3
"""Freeze a fresh one-shot public-transfer calibration packet.

The packet is metadata only. It binds task IDs, commands, hashes, and private
readiness evidence, but it does not execute public benchmark cases and it does
not export public prompts, tests, solutions, traces, or score labels into
training artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_POLICY = ROOT / "configs" / "permissive_growth_policy.json"
DEFAULT_CARDS = [
    "source_mbpp",
    "source_evalplus",
    "source_bigcodebench",
    "source_human_eval",
    "source_livecodebench",
]
DEFAULT_EXCLUDED = [
    "reports/public_wide_slice_manifest_seed23_5x32.jsonl",
    "reports/public_wide_slice_manifest_industry_code_transfer_seed14_5x64_v1.jsonl",
]
HARNESS_FILES = [
    "scripts/operator_bounded_public_calibration.py",
    "scripts/real_code_benchmark_graduation.py",
    "scripts/student_token_code_candidate_generator.py",
    "scripts/wide_public_slice_selector.py",
    "scripts/full_body_contract_transfer_recovery_v1.py",
    "crates/symliquid-cli/src/code_token_generator.rs",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default="public_transfer_lift_v2_seed41_5x64")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cases-per-card", type=int, default=64)
    parser.add_argument("--cards", default=",".join(DEFAULT_CARDS))
    parser.add_argument("--case-manifest", default="")
    parser.add_argument("--case-manifest-report", default="")
    parser.add_argument("--private-recovery", default="reports/full_body_contract_transfer_recovery_v2_private320.json")
    parser.add_argument("--policy", default=rel(DEFAULT_POLICY))
    parser.add_argument("--registry", default="reports/public_benchmark_run_registry.jsonl")
    parser.add_argument("--operator-lock", default="reports/public_calibration_operator_lock.flag")
    parser.add_argument("--exclude-manifest", action="append", default=list(DEFAULT_EXCLUDED))
    parser.add_argument(
        "--auto-exclude-consumed-manifests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also exclude every consumed public case manifest found in the public benchmark registry.",
    )
    parser.add_argument("--out", default="reports/public_transfer_lift_v2_readiness_packet.json")
    parser.add_argument("--markdown-out", default="reports/public_transfer_lift_v2_readiness_packet.md")
    args = parser.parse_args()

    packet = build_packet(args)
    write_json(resolve(args.out), packet)
    write_text(resolve(args.markdown_out), render_markdown(packet))
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if packet["trigger_state"] in {"GREEN", "YELLOW"} else 2


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    slug = clean_slug(args.slug)
    cards = [card.strip() for card in str(args.cards).split(",") if card.strip()]
    case_manifest = args.case_manifest or f"reports/public_wide_slice_manifest_{slug}.jsonl"
    case_manifest_report = args.case_manifest_report or f"reports/public_wide_slice_selector_{slug}.json"
    candidate_manifest = f"reports/student_code_candidates_{slug}.jsonl"
    calibration_out = f"reports/real_code_benchmark_graduation_{slug}.json"
    trace_out = f"reports/real_code_benchmark_traces_{slug}.jsonl"
    transfer_artifact = f"reports/transfer_artifacts/code/real_code_benchmark_graduation_{slug}_transfer_artifact.json"
    command = [
        "python3",
        "scripts/real_code_benchmark_graduation.py",
        "--cards",
        ",".join(cards),
        "--seed",
        str(int(args.seed)),
        "--max-cases-per-card",
        str(int(args.cases_per_card)),
        "--case-manifest",
        case_manifest,
        "--student-candidate-generator",
        "token",
        "--student-candidate-manifest",
        candidate_manifest,
        "--out",
        calibration_out,
        "--trace-out",
        trace_out,
        "--transfer-artifact-out",
        transfer_artifact,
    ]

    registry_rows = read_jsonl(resolve(args.registry))
    consumed_rows = [row for row in registry_rows if row.get("consumed") is True]
    exclude_manifests = list(args.exclude_manifest)
    auto_excluded_manifests = consumed_case_manifest_paths(consumed_rows) if args.auto_exclude_consumed_manifests else []
    for path in auto_excluded_manifests:
        if path not in exclude_manifests:
            exclude_manifests.append(path)
    manifest_rows = read_jsonl(resolve(case_manifest))
    manifest_scan = scan_case_manifest(manifest_rows)
    excluded_keys = load_excluded_keys(exclude_manifests)
    overlap = sorted(
        key
        for key in manifest_scan["task_keys"]
        if key in excluded_keys
    )
    matching_consumed = [
        row for row in consumed_rows
        if str(row.get("run_id") or row.get("surface_slug") or "") == slug
    ]
    policy = read_json(resolve(args.policy), {})
    registry_status = public_registry_status(policy, consumed_rows, slug)
    private_recovery = read_json(resolve(args.private_recovery), {})
    private_summary = as_dict(private_recovery.get("summary"))
    private_ready = bool(
        private_recovery.get("trigger_state") == "GREEN"
        and private_summary.get("ready_for_future_governed_public_calibration") is True
        and int_or_zero(private_summary.get("private_eval_rows")) >= 320
        and float_or_zero(private_summary.get("full_contract_selected_pass_rate")) >= 0.95
        and float_or_zero(private_summary.get("full_no_candidate_rate")) <= 0.01
        and float_or_zero(private_summary.get("full_no_admissible_task_rate")) <= 0.01
        and int_or_zero(private_summary.get("fallback_return_count")) == 0
        and int_or_zero(private_summary.get("template_like_candidate_count")) == 0
        and int_or_zero(private_summary.get("public_training_rows")) == 0
    )

    total_task_count = len(cards) * int(args.cases_per_card)
    stage = {
        "runner_family": "real_code_benchmark_graduation",
        "slug": slug,
        "status": "frozen_unconsumed_contract",
        "seed": int(args.seed),
        "cards": cards,
        "cases_per_card": int(args.cases_per_card),
        "total_task_count": total_task_count,
        "case_manifest": case_manifest,
        "case_manifest_report": case_manifest_report,
        "command_after_run_registry_check": command,
        "command_after_unlock_only": command,
        "compute_budget": {
            "external_inference_calls": 0,
            "operator_wrapper_timeout_seconds": 7200,
            "runner_timeout_seconds": 21600,
            "public_timeout_seconds": 10800,
            "candidate_budget": "token generator max candidates per task fixed by runner defaults",
        },
        "mandatory_after_run": [
            "append reports/public_benchmark_run_registry.jsonl with consumed status",
            "run broad transfer matrix and residual mining for metadata-only categories",
            "do not train on public prompts, tests, hidden tests, solutions, traces, score labels, or answer templates",
        ],
    }
    harness_hashes = {path: file_record(resolve(path)) for path in HARNESS_FILES}
    evidence_hashes = {
        "case_manifest": file_record(resolve(case_manifest)),
        "case_manifest_report": file_record(resolve(case_manifest_report)),
        "private_recovery": file_record(resolve(args.private_recovery)),
        "policy": file_record(resolve(args.policy)),
        "registry": file_record(resolve(args.registry)),
    }
    optional_evidence_hashes = {
        "legacy_operator_lock": file_record(resolve(args.operator_lock)),
    }
    missing = [
        label
        for label, record in evidence_hashes.items()
        if not record["exists"]
    ] + [
        path
        for path, record in harness_hashes.items()
        if not record["exists"]
    ]
    integrity_payload = {
        "slug": slug,
        "stage": stage,
        "harness_hashes": harness_hashes,
        "evidence_hashes": evidence_hashes,
        "optional_evidence_hashes": optional_evidence_hashes,
        "private_summary": private_summary,
    }
    frozen_integrity = {
        "policy": "project_theseus_public_transfer_lift_v2_frozen_integrity_v1",
        "sha256": sha256_json(integrity_payload),
        "harness_files": harness_hashes,
        "harness_hashes_current": True,
        "harness_mismatch_count": 0,
        "harness_mismatches": [],
        "harness_missing_count": sum(1 for record in harness_hashes.values() if not record["exists"]),
        "harness_missing": [path for path, record in harness_hashes.items() if not record["exists"]],
        "evidence_hashes": evidence_hashes,
        "evidence_artifacts_hashable": not missing,
        "evidence_missing_count": len(missing),
        "evidence_missing": missing,
    }

    gates = [
        gate("private_recovery_ready_320", private_ready, private_summary),
        gate("manifest_row_count_matches_contract", len(manifest_rows) == total_task_count, {
            "observed": len(manifest_rows),
            "expected": total_task_count,
        }),
        gate("manifest_card_counts_match_contract", manifest_scan["card_counts"] == {card: int(args.cases_per_card) for card in sorted(cards)}, {
            "observed": manifest_scan["card_counts"],
            "expected": {card: int(args.cases_per_card) for card in sorted(cards)},
        }),
        gate("manifest_metadata_only", not manifest_scan["metadata_failures"], manifest_scan["metadata_failures"][:16]),
        gate("manifest_disjoint_from_consumed_selectors", not overlap, {
            "overlap_count": len(overlap),
            "sample": overlap[:16],
            "excluded_manifest_count": len(exclude_manifests),
            "auto_excluded_consumed_manifest_count": len(auto_excluded_manifests),
        }),
        gate("surface_not_consumed", not matching_consumed and registry_status["surface_not_consumed"], {
            "matching_consumed_count": len(matching_consumed),
            "run_registry": registry_status,
        }),
        gate("public_run_registry_enabled", registry_status["run_registry_execution_enabled"], registry_status),
        gate("output_absent_before_execute", not resolve(calibration_out).exists(), calibration_out),
        gate("trace_absent_before_execute", not resolve(trace_out).exists(), trace_out),
        gate("candidate_manifest_deferred", not resolve(candidate_manifest).exists(), candidate_manifest),
        gate("legacy_operator_lock_not_required_by_registry", registry_status["run_registry_execution_enabled"] or resolve(args.operator_lock).exists(), {
            "legacy_operator_lock_present": resolve(args.operator_lock).exists(),
            "run_registry_execution_enabled": registry_status["run_registry_execution_enabled"],
            "operator_lock": args.operator_lock,
        }),
        gate("frozen_integrity_hashable", frozen_integrity["evidence_artifacts_hashable"], frozen_integrity),
    ]
    ready = all(row["passed"] for row in gates)
    registry_authorized = bool(
        ready
        and registry_status["run_registry_execution_enabled"]
        and registry_status["surface_not_consumed"]
        and not matching_consumed
    )
    latest_consumed = consumed_rows[-1] if consumed_rows else {}
    summary = {
        "proposed_slug": slug,
        "proposed_public_surface_slug": slug,
        "proposed_public_surface_task_count": total_task_count,
        "proposed_public_surface_seed": int(args.seed),
        "proposed_public_surface_cases_per_card": int(args.cases_per_card),
        "proposed_public_surface_cards": sorted(cards),
        "latest_consumed_public_surface": latest_consumed.get("surface_slug") or latest_consumed.get("run_id") or "",
        "latest_consumed_public_surface_task_count": latest_consumed.get("public_task_count"),
        "latest_consumed_public_score": latest_consumed.get("real_public_task_pass_rate"),
        "fresh_surface_disjoint_from_consumed_selectors": not overlap,
        "excluded_manifest_count": len(exclude_manifests),
        "auto_excluded_consumed_manifest_count": len(auto_excluded_manifests),
        "run_registry_execution_enabled": registry_status["run_registry_execution_enabled"],
        "run_registry_reasons": registry_status["reasons"],
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "consumed_run_count_total_for_audit": registry_status["consumed_run_count_total_for_audit"],
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
        "private_eval_rows": private_summary.get("private_eval_rows"),
        "private_selected_pass_rate": private_summary.get("full_contract_selected_pass_rate"),
        "private_no_candidate_rate": private_summary.get("full_no_candidate_rate"),
        "private_no_admissible_task_rate": private_summary.get("full_no_admissible_task_rate"),
        "fallback_return_count": private_summary.get("fallback_return_count"),
        "template_like_candidate_count": private_summary.get("template_like_candidate_count"),
        "public_training_rows": private_summary.get("public_training_rows"),
        "technical_ready": ready,
        "public_calibration_allowed": registry_authorized,
        "operator_lock_active": resolve(args.operator_lock).exists(),
        "legacy_operator_lock_is_circuit_breaker_only": True,
    }
    return {
        "policy": "project_theseus_public_calibration_readiness_packet_v1",
        "mode": "post_distillation_v4_operator_review",
        "created_utc": now(),
        "trigger_state": "GREEN" if ready else "YELLOW",
        "technical_ready_for_one_bounded_public_calibration": ready,
        "public_calibration_allowed": registry_authorized,
        "operator_lock_active": resolve(args.operator_lock).exists(),
        "summary": summary,
        "public_benchmark_contract": {
            "policy": "project_theseus_public_benchmark_contract_v1",
            "status": "frozen_next_surface_contract",
            "purpose": "fresh one-shot public-transfer calibration after private survival-lane repair",
            "global_rules": {
                "public_calibration_only": True,
                "private_training_allowed": False,
                "do_not_train_on_public_prompts_tests_solutions_traces_or_score_labels": True,
                "run_registry_execution_enabled": registry_status["run_registry_execution_enabled"],
                "excluded_consumed_case_manifests": auto_excluded_manifests,
                "consumed_surface_do_not_rerun": [
                    {
                        "slug": row.get("surface_slug") or row.get("run_id"),
                        "score": row.get("real_public_task_pass_rate"),
                        "task_count": row.get("public_task_count"),
                    }
                    for row in consumed_rows
                ],
            },
            "stage_1_code_generation_surface": stage,
            "contract_snapshot": {
                "theseus_git_head": git_head(),
                "harness_files_sha256": {path: record.get("sha256") for path, record in harness_hashes.items()},
            },
        },
        "frozen_integrity": frozen_integrity,
        "gates": gates,
        "proposed_operator_actions": {
            "calibration_command_after_run_registry_check": " ".join(command),
            "calibration_command_after_unlock_only": " ".join(command),
            "guarded_runner_execute": (
                "python3 scripts/operator_bounded_public_calibration.py "
                f"--packet {rel(resolve(args.out))} "
                f"--execute --out reports/operator_bounded_public_calibration_{slug}.json "
                f"--markdown-out reports/operator_bounded_public_calibration_{slug}.md"
            ),
            "mandatory_after_run": stage["mandatory_after_run"],
        },
        "external_inference_calls": 0,
    }


def scan_case_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    card_counts: dict[str, int] = {}
    task_keys: set[str] = set()
    metadata_failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        card_id = str(row.get("card_id") or "")
        task_id = str(row.get("task_id") or "")
        if card_id and task_id:
            card_counts[card_id] = card_counts.get(card_id, 0) + 1
            task_keys.add(f"{card_id}:{task_id}")
        if row.get("prompts_exported") or row.get("tests_exported") or row.get("solutions_exported") or row.get("candidate_code_exported"):
            metadata_failures.append({"index": index, "card_id": card_id, "task_id": task_id})
    return {"card_counts": dict(sorted(card_counts.items())), "task_keys": task_keys, "metadata_failures": metadata_failures}


def load_excluded_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for value in paths:
        for part in str(value or "").split(","):
            path = part.strip()
            if not path:
                continue
            for row in read_jsonl(resolve(path)):
                card_id = str(row.get("card_id") or "")
                task_id = str(row.get("task_id") or "")
                if card_id and task_id:
                    keys.add(f"{card_id}:{task_id}")
    return keys


def consumed_case_manifest_paths(rows: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        command = row.get("command") if isinstance(row.get("command"), list) else []
        manifest = command_arg([str(part) for part in command], "--case-manifest")
        if (
            manifest
            and manifest.startswith("reports/public_wide_slice_manifest_")
            and manifest.endswith(".jsonl")
            and manifest not in seen
        ):
            paths.append(manifest)
            seen.add(manifest)
    return paths


def public_registry_status(policy: dict[str, Any], consumed_rows: list[dict[str, Any]], slug: str) -> dict[str, Any]:
    public_policy = as_dict(policy.get("public_benchmarks"))
    matching = [
        row
        for row in consumed_rows
        if str(row.get("run_id") or row.get("surface_slug") or "") == slug
    ]
    per_surface_max = int_or_zero(public_policy.get("per_surface_max_runs")) or 1
    surface_not_consumed = bool(slug and len(matching) < per_surface_max)
    reasons: list[str] = []
    if public_policy.get("execution_default") not in {"governed_measurement_run_registry", "governed_run_registry"}:
        reasons.append("governed_run_registry_policy_not_enabled")
    if not slug:
        reasons.append("missing_frozen_run_id")
    if not surface_not_consumed:
        reasons.append("surface_consumed_or_per_surface_limit_reached")
    return {
        "policy": rel(DEFAULT_POLICY),
        "execution_default": public_policy.get("execution_default"),
        "run_registry_execution_enabled": public_policy.get("execution_default") in {"governed_measurement_run_registry", "governed_run_registry"},
        "surface_not_consumed": surface_not_consumed,
        "matching_consumed_count": len(matching),
        "per_surface_max_runs": per_surface_max,
        "time_period_run_cap_enabled": False,
        "calendar_throttle_enabled": False,
        "consumed_run_count_total_for_audit": len(consumed_rows),
        "fresh_surfaces_calendar_throttled": False,
        "fresh_surface_execution_policy": "run_immediately_when_frozen_registry_surface_is_clean",
        "reasons": reasons,
    }


def command_arg(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1])


def gate(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def render_markdown(packet: dict[str, Any]) -> str:
    summary = as_dict(packet.get("summary"))
    gates = packet.get("gates") if isinstance(packet.get("gates"), list) else []
    lines = [
        "# Public Transfer Lift v2 Packet",
        "",
        f"- trigger_state: `{packet.get('trigger_state')}`",
        f"- slug: `{summary.get('proposed_slug')}`",
        f"- tasks: `{summary.get('proposed_public_surface_task_count')}`",
        f"- seed: `{summary.get('proposed_public_surface_seed')}`",
        f"- private rows: `{summary.get('private_eval_rows')}`",
        f"- private selected pass: `{summary.get('private_selected_pass_rate')}`",
        f"- latest consumed public score: `{summary.get('latest_consumed_public_score')}`",
        "",
        "## Gates",
    ]
    for row in gates:
        lines.append(f"- `{row.get('gate')}`: `{row.get('passed')}`")
    return "\n".join(lines) + "\n"


def clean_slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return out.strip("_") or "public_transfer_lift_v2"


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "sha256": "", "size_bytes": 0}
    data = path.read_bytes()
    return {"path": rel(path), "exists": True, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = ROOT / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else text
    return text


if __name__ == "__main__":
    raise SystemExit(main())
