"""Training data and benchmark asset inventory for SparkStream.

The autonomous loop needs an always-current view of what data it is allowed to
use, what role each file appears to play, and which large external benchmark
trees are present locally. This script is intentionally read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "training_data_inventory.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--max-rows", type=int, default=800)
    args = parser.parse_args()

    report = build_inventory(args.max_rows)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def build_inventory(max_rows: int) -> dict[str, Any]:
    roots = [
        ROOT / "data",
        ROOT / "benchmarks",
        ROOT / "configs",
        Path("D:/ProjectTheseus/training_data/open_code_pantry"),
        Path("D:/ProjectTheseus/training_data/open_conversation_pantry"),
        Path("D:/ProjectTheseus/training_data/residual_code_curriculum"),
    ]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or should_skip(path):
                continue
            rows.append(file_row(path))
    rows.sort(key=lambda item: (item["role"], item["path"]))
    truncated = len(rows) > max_rows
    kept = rows[:max_rows]
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows:
        role = row["role"]
        bucket = by_role.setdefault(role, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += row["bytes"]

    public_benchmarks = public_benchmark_summary(ROOT / "data" / "public_benchmarks")
    external_candidates = external_candidate_summary(ROOT / "data" / "external_benchmark_candidates")
    old_project_sources = old_project_training_source_summary(
        ROOT / "data" / "training_sources" / "old_project_registry_training_sources.json"
    )
    legacy_training_sample = read_json(ROOT / "reports" / "legacy_training_source_sample.json", {})
    trace_capsule_materialization = read_json(ROOT / "reports" / "trace_fabric_capsule_materialization.json", {})
    legacy_adapter_plan = read_json(ROOT / "reports" / "legacy_adapter_bank_training_plan.json", {})
    active_inference_pilot = read_json(ROOT / "reports" / "legacy_active_inference_pilot.json", {})
    open_code_pantry = read_json(ROOT / "reports" / "open_code_training_pantry.json", {})
    open_conversation_pantry = read_json(ROOT / "reports" / "open_conversation_training_pantry.json", {})
    code_residual_curriculum = read_json(ROOT / "reports" / "code_residual_curriculum.json", {})
    cognitive_context_router = read_json(ROOT / "reports" / "cognitive_context_router.json", {})
    return {
        "policy": "sparkstream_training_data_inventory_v0",
        "updated_utc": now(),
        "workspace": str(ROOT),
        "summary": {
            "files": len(rows),
            "shown": len(kept),
            "truncated": truncated,
            "bytes": sum(int(row["bytes"]) for row in rows),
            "by_role": by_role,
            "public_benchmark_trees": public_benchmarks["count"],
            "external_candidate_trees": external_candidates["count"],
            "old_project_ready_training_sources": old_project_sources["ready_count"],
            "legacy_training_tiny_sample_rows": get_path(
                legacy_training_sample, ["summary", "sample_rows"], 0
            ),
            "trace_fabric_materialized_rows": get_path(
                trace_capsule_materialization, ["summary", "materialized_rows"], 0
            ),
            "legacy_adapter_bank_plan_rows": get_path(
                legacy_adapter_plan, ["summary", "plan_rows"], 0
            ),
            "active_inference_belief_updates": get_path(
                active_inference_pilot, ["summary", "accepted_belief_updates"], 0
            ),
            "open_code_pantry_train_expressions": get_path(open_code_pantry, ["summary", "private_train_expression_count"], 0),
            "open_code_pantry_train_full_body_functions": get_path(
                open_code_pantry, ["summary", "private_train_full_body_count"], 0
            ),
            "open_code_pantry_train_rows": get_path(open_code_pantry, ["summary", "private_train_row_count"], 0),
            "open_code_pantry_benchmark_excluded": get_path(open_code_pantry, ["summary", "benchmark_excluded"], False),
            "open_conversation_pantry_train_rows": get_path(
                open_conversation_pantry, ["summary", "private_train_rows"], 0
            ),
            "open_conversation_pantry_sts_rows": get_path(open_conversation_pantry, ["summary", "sts_rows"], 0),
            "residual_code_curriculum_private_rows": get_path(
                code_residual_curriculum, ["summary", "private_row_count"], 0
            ),
            "cognitive_context_rows": get_path(
                cognitive_context_router, ["summary", "context_row_count"], 0
            ),
        },
        "files": kept,
        "public_benchmarks": public_benchmarks,
        "external_candidates": external_candidates,
        "old_project_training_sources": old_project_sources,
        "legacy_training_sample": legacy_training_sample,
        "trace_capsule_materialization": trace_capsule_materialization,
        "legacy_adapter_bank_training_plan": legacy_adapter_plan,
        "legacy_active_inference_pilot": active_inference_pilot,
        "open_code_training_pantry": open_code_pantry,
        "open_conversation_training_pantry": open_conversation_pantry,
        "code_residual_curriculum": code_residual_curriculum,
        "cognitive_context_router": cognitive_context_router,
        "usage_policy": {
            "external_inference": "forbidden_by_default",
            "network_imports": "license_audited_before_training_use",
            "copyrighted_game_roms": "do_not_download_or_train_on_without_explicit_rights",
            "external_source_archives": "ignored_staging_only_until_adapter_and_license_audit",
            "open_conversation_data": "private_training_pressure_only_never_public_promotion_evidence",
            "residual_code_curriculum": "private_generated_training_pressure_only_public_failures_inform_categories_not_answers",
            "cognitive_context_spaces": "private_sts_context_training_pressure_only_not_public_promotion_evidence",
            "trace_fabric_capsules": "governed_metadata_only_rows_no_raw_traces_no_public_claim_evidence",
            "legacy_adapter_bank_plan": "dry_run_routing_plan_only_no_adapter_weight_activation",
            "active_inference_belief_updates": "toy_world_only_until_real_adapter_replay_confirms_generalization",
        },
    }


def file_row(path: Path) -> dict[str, Any]:
    stat = path.stat()
    row = {
        "path": rel(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "extension": path.suffix.lower(),
        "role": role_guess(path),
    }
    if path.suffix.lower() in {".jsonl", ".csv", ".txt", ".md"}:
        row["line_count"] = count_lines(path)
    if stat.st_size <= 64 * 1024 * 1024:
        row["sha256"] = sha256(path)
    return row


def public_benchmark_summary(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "count": 0, "trees": []}
    trees: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file() and not should_skip(p)]
        license_file = first_existing(
            [
                child / "LICENSE",
                child / "LICENSE.txt",
                child / "COPYING",
                child / "pyproject.toml",
                child / "setup.py",
            ]
        )
        trees.append(
            {
                "name": child.name,
                "path": rel(child),
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "license_signal": rel(license_file) if license_file else "",
                "role": "public_benchmark_tree",
            }
        )
    return {"exists": True, "count": len(trees), "trees": trees}


def external_candidate_summary(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "count": 0, "trees": []}
    trees: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file() and not should_skip(p)]
        metadata_files = [p for p in files if p.name.endswith(".metadata.json") or p.name == "metadata.json"]
        archives = [p for p in files if p.suffix.lower() in {".zip", ".tar", ".gz", ".xz", ".7z"}]
        trees.append(
            {
                "name": child.name,
                "path": rel(child),
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "metadata_files": len(metadata_files),
                "archives": len(archives),
                "role": external_role_guess(child),
                "training_use_allowed": training_use_allowed_for_external_tree(child),
            }
        )
    return {"exists": True, "count": len(trees), "trees": trees}


def role_guess(path: Path) -> str:
    text = rel(path).lower()
    suffix = path.suffix.lower()
    if "d:/projecttheseus/training_data/open_code_pantry" in text:
        if "private_train" in text:
            return "permissive_open_code_train_only"
        if "tarballs" in text:
            return "permissive_open_code_source_archive"
        return "permissive_open_code_source_sample"
    if "d:/projecttheseus/training_data/open_conversation_pantry" in text:
        if "private_train" in text:
            return "permissive_open_conversation_train_only"
        if "sts_streams" in text:
            return "permissive_open_conversation_sts_streams"
        if "dataset_cards" in text:
            return "permissive_open_conversation_source_card"
        return "permissive_open_conversation_source_sample"
    if "d:/projecttheseus/training_data/residual_code_curriculum" in text:
        if "private_train" in text:
            return "private_residual_code_train_only"
        return "private_residual_code_curriculum_asset"
    if "data/sts_learning" in text:
        if "cognitive_context" in text or "sts_code_context_spaces" in text:
            return "cognitive_context_sts_train_eval_data"
        return "sts_parallel_stream_train_eval_data"
    if "data/external_benchmark_candidates/training_data_samples" in text:
        return "training_data_governed_sample"
    if "data/external_benchmark_candidates/training_data_metadata" in text:
        return "training_data_candidate_metadata"
    if "data/external_benchmark_candidates/rl_envs" in text:
        return "rl_environment_source_archive"
    if "data/external_benchmark_candidates/language_benchmarks" in text:
        return "language_benchmark_source_archive"
    if "data/external_benchmark_candidates" in text:
        return "external_benchmark_candidate"
    if "data/training_sources/old_project_registry_training_sources.json" in text:
        return "old_project_training_source_manifest"
    if "data/training_sources/legacy_training_admissions.json" in text:
        return "old_project_training_source_admission"
    if "data/training_sources/legacy_tiny_dry_run_sample.jsonl" in text:
        return "old_project_legacy_tiny_train_sample"
    if "data/training_sources/trace_fabric_capsule_candidates.jsonl" in text:
        return "trace_fabric_capsule_candidates"
    if "data/training_sources/trace_fabric_materialized_training_rows.jsonl" in text:
        return "trace_fabric_governed_training_rows"
    if "data/training_sources/legacy_adapter_bank_dry_run_plan.jsonl" in text:
        return "legacy_adapter_bank_dry_run_plan"
    if "data/world_model/active_inference_belief_updates.jsonl" in text:
        return "active_inference_world_model_belief_updates"
    if "data/old_project_benchmarks" in text:
        return "old_project_redacted_benchmark_case_manifest"
    if "data/synthetic" in text:
        if "train" in text or suffix in {".jsonl", ".json"}:
            return "synthetic_training_data"
        return "synthetic_data_asset"
    if "mutated" in text or "holdout" in text:
        return "frontier_holdout"
    if "bridge" in text:
        return "bridge_benchmark"
    if "train" in text and suffix in {".jsonl", ".json", ".txt"}:
        return "training_data"
    if "eval" in text or "test" in text:
        return "evaluation_data"
    if "public_benchmarks" in text:
        return "public_benchmark_asset"
    if suffix in {".toml", ".json"} and "config" in text:
        return "configuration"
    if suffix in {".jsonl", ".json", ".csv", ".txt"}:
        return "data_asset"
    return "support_asset"


def external_role_guess(path: Path) -> str:
    text = rel(path).lower()
    if "training_data_samples" in text:
        return "training_data_governed_samples"
    if "training_data_metadata" in text:
        return "training_data_candidate_metadata"
    if "rl_envs" in text:
        return "rl_environment_source_archives"
    if "language_benchmarks" in text:
        return "language_benchmark_source_archives"
    return "external_candidates"


def training_use_allowed_for_external_tree(path: Path) -> bool:
    if "training_data_samples" not in rel(path).lower():
        return False
    sampler = ROOT / "reports" / "training_data_sampler.json"
    if not sampler.exists():
        return False
    try:
        report = json.loads(sampler.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(report.get("training_use_allowed", False))


def old_project_training_source_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "ready_count": 0, "sources": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "ready_count": 0, "sources": [], "error": "unreadable_manifest"}
    rows = payload.get("sources", []) if isinstance(payload.get("sources"), list) else []
    sources = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sources.append(
            {
                "dataset_id": row.get("dataset_id"),
                "training_use_state": row.get("training_use_state"),
                "sample_count": row.get("sample_count"),
                "family": row.get("family"),
                "modality": row.get("modality"),
                "local_exists": row.get("local_exists"),
                "sha256_verified": row.get("sha256_verified"),
                "not_public_benchmark_claim_evidence": "not_public_benchmark_claim_evidence"
                in (row.get("usage_restrictions") if isinstance(row.get("usage_restrictions"), list) else []),
            }
        )
    return {
        "exists": True,
        "manifest": rel(path),
        "source_count": len(sources),
        "ready_count": len([row for row in sources if row.get("training_use_state") == "ready_local_verified"]),
        "sources": sources[:64],
    }


def should_skip(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if "__pycache__" in parts or ".git" in parts:
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def count_lines(path: Path) -> int:
    count = 0
    try:
        with path.open("rb") as handle:
            for _ in handle:
                count += 1
    except OSError:
        return 0
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
