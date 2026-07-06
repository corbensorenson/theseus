"""Checkpoint registry for SparkStream/RMI runs.

Checkpoints are lightweight release bundles for local ratchet state. They copy
small report artifacts, store score summaries, and provide compare/list helpers
for the dashboard and CLI. They intentionally do not copy large training data or
target build artifacts unless a future policy explicitly allows it.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--label", default="sparkstream_checkpoint")
    create.add_argument("--reason", default="manual_checkpoint")
    create.add_argument("--profile", default="unknown")
    create.add_argument("--status", default="recorded")
    create.add_argument("--notes", default="")
    create.add_argument("--kind", choices=["auto", "major", "minor"], default="auto")
    create.add_argument("--parent", default="")
    create.add_argument("--base-major", default="")
    create.add_argument("--out", default="reports/checkpoint_last.json")

    sub.add_parser("list").add_argument("--out", default="")

    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--out", default="reports/checkpoint_compare.json")

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--id", required=True)
    materialize.add_argument("--out", required=True)
    materialize.add_argument("--force", action="store_true")
    materialize.add_argument("--report-out", default="reports/checkpoint_materialize.json")

    args = parser.parse_args()
    policy = read_json(Path(args.policy))
    if args.command == "create":
        manifest = create_checkpoint(policy, args)
        write_json(ROOT / args.out, manifest)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "list":
        registry = load_registry(policy)
        if args.out:
            write_json(ROOT / args.out, registry)
        print(json.dumps(registry, indent=2))
        return 0
    if args.command == "compare":
        report = compare_checkpoints(policy, args.a, args.b)
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    if args.command == "materialize":
        report = materialize_checkpoint(policy, args.id, ROOT / args.out, args.force)
        if args.report_out:
            write_json(ROOT / args.report_out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    return 1


def create_checkpoint(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    created = now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = slug(args.label)
    checkpoint_id = f"{stamp}_{safe_label}"
    checkpoint_root = ROOT / get_path(policy, ["checkpointing", "root"], "checkpoints") / checkpoint_id
    artifacts_dir = checkpoint_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(policy)
    parent_manifest = resolve_parent_manifest(policy, registry, args.parent)
    workspace_files, workspace_skipped = collect_workspace_snapshot_files(policy)
    snapshot_kind = choose_snapshot_kind(policy, args.kind, parent_manifest)

    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    max_artifact_bytes = int(get_path(policy, ["checkpointing", "max_artifact_bytes"], 8 * 1024 * 1024))
    for source in resolve_artifact_paths(policy):
        rel = source.relative_to(ROOT)
        size = source.stat().st_size
        if size > max_artifact_bytes:
            skipped.append(
                {
                    "source": str(rel).replace("\\", "/"),
                    "bytes": size,
                    "reason": f"exceeds_max_artifact_bytes:{max_artifact_bytes}",
                }
            )
            continue
        dest = artifacts_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(
            {
                "source": str(rel).replace("\\", "/"),
                "bytes": size,
                "sha256": sha256(dest),
            }
        )

    if snapshot_kind == "major":
        snapshot = create_major_snapshot(policy, checkpoint_id, checkpoint_root, workspace_files, workspace_skipped)
    else:
        snapshot = create_minor_snapshot(
            policy,
            checkpoint_id,
            checkpoint_root,
            workspace_files,
            workspace_skipped,
            parent_manifest,
            args.base_major,
        )

    manifest = {
        "checkpoint_id": checkpoint_id,
        "created_utc": created,
        "label": args.label,
        "reason": args.reason,
        "profile": args.profile,
        "status": args.status,
        "notes": args.notes,
        "workspace": str(ROOT),
        "git": git_state(),
        "scores": collect_scores(ROOT / "reports"),
        "promotion": collect_promotion(ROOT / "reports" / "candidate_promotion_gate.json"),
        "artifacts": copied,
        "skipped_artifacts": skipped,
        "snapshot": snapshot,
        "artifact_root": str(artifacts_dir.relative_to(ROOT)).replace("\\", "/"),
        "policy": "major_minor_checkpoint_chain_v0",
    }
    write_json(checkpoint_root / "manifest.json", manifest)
    registry.setdefault("checkpoints", [])
    registry["checkpoints"] = [
        item for item in registry["checkpoints"] if item.get("checkpoint_id") != checkpoint_id
    ]
    registry["checkpoints"].append(registry_entry(manifest))
    registry["checkpoints"].sort(key=lambda item: item.get("created_utc", ""))
    registry["updated_utc"] = created
    write_json(ROOT / registry_path(policy), registry)
    return manifest


def compare_checkpoints(policy: dict[str, Any], left_id: str, right_id: str) -> dict[str, Any]:
    left = load_checkpoint_manifest(policy, left_id)
    right = load_checkpoint_manifest(policy, right_id)
    if not left or not right:
        return {
            "ok": False,
            "error": "checkpoint_not_found",
            "a": left_id,
            "b": right_id,
        }
    return {
        "ok": True,
        "a": registry_entry(left),
        "b": registry_entry(right),
        "score_delta": score_delta(left.get("scores", {}), right.get("scores", {})),
        "snapshot_delta": snapshot_delta(left.get("snapshot", {}), right.get("snapshot", {})),
        "promotion_delta": {
            "a_promote": get_path(left, ["promotion", "promote"], None),
            "b_promote": get_path(right, ["promotion", "promote"], None),
            "a_passed": get_path(left, ["promotion", "passed"], None),
            "b_passed": get_path(right, ["promotion", "passed"], None),
        },
    }


def load_checkpoint_manifest(policy: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    root = ROOT / get_path(policy, ["checkpointing", "root"], "checkpoints")
    direct = root / checkpoint_id / "manifest.json"
    if direct.exists():
        return read_json(direct)
    registry = load_registry(policy)
    for item in registry.get("checkpoints", []):
        if item.get("checkpoint_id") == checkpoint_id or item.get("label") == checkpoint_id:
            path = root / item["checkpoint_id"] / "manifest.json"
            return read_json(path)
    return {}


def load_registry(policy: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / registry_path(policy)
    if path.exists():
        return read_json(path)
    return {
        "policy": "sparkstream_checkpoint_registry_v0",
        "updated_utc": now(),
        "checkpoints": [],
    }


def registry_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": manifest.get("checkpoint_id"),
        "created_utc": manifest.get("created_utc"),
        "label": manifest.get("label"),
        "reason": manifest.get("reason"),
        "profile": manifest.get("profile"),
        "status": manifest.get("status"),
        "scores": manifest.get("scores", {}),
        "promote": get_path(manifest, ["promotion", "promote"], None),
        "passed": get_path(manifest, ["promotion", "passed"], None),
        "total": get_path(manifest, ["promotion", "total"], None),
        "snapshot_kind": get_path(manifest, ["snapshot", "kind"], None),
        "parent_id": get_path(manifest, ["snapshot", "parent_id"], None),
        "major_id": get_path(manifest, ["snapshot", "major_id"], None),
        "chain_depth": get_path(manifest, ["snapshot", "chain_depth"], None),
        "state_hash": get_path(manifest, ["snapshot", "state_hash"], None),
        "chain_hash": get_path(manifest, ["snapshot", "chain_hash"], None),
    }


def collect_scores(reports: Path) -> dict[str, Any]:
    benchmark_ledger = read_json(reports / "benchmark_ledger.json")
    scores: dict[str, Any] = {}
    if isinstance(benchmark_ledger, list):
        for row in benchmark_ledger:
            name = row.get("benchmark_name")
            if name:
                scores[name] = {
                    "score": row.get("score"),
                    "residual": row.get("residual"),
                    "lifecycle": row.get("lifecycle"),
                    "status": row.get("saturation_status"),
                    "threshold": get_path(row, ["graduation_policy", "current_threshold"], None),
                    "floor": get_path(row, ["graduation_policy", "floor_threshold"], None),
                    "wall_type": row.get("wall_type"),
                }
    candidate = read_json(reports / "candidate_promotion_gate.json")
    if candidate:
        scores["candidate_gate"] = candidate.get("scores", {})
    preflight = read_json(reports / "training_preflight_report.json")
    if preflight:
        scores["training_preflight"] = {
            "heavy_training_allowed": preflight.get("heavy_training_allowed"),
            "passed": preflight.get("passed"),
            "total": preflight.get("total"),
            "blocker_count": preflight.get("blocker_count"),
            "warning_count": preflight.get("warning_count"),
        }
    return scores


def collect_promotion(path: Path) -> dict[str, Any]:
    gate = read_json(path)
    if not gate:
        return {}
    return {
        "promote": gate.get("promote"),
        "passed": gate.get("passed"),
        "total": gate.get("total"),
        "failed_gates": [
            item.get("gate")
            for item in gate.get("checks", [])
            if isinstance(item, dict) and not item.get("passed")
        ],
    }


def resolve_parent_manifest(policy: dict[str, Any], registry: dict[str, Any], explicit_parent: str) -> dict[str, Any]:
    if explicit_parent:
        return load_checkpoint_manifest(policy, explicit_parent)
    checkpoints = registry.get("checkpoints", [])
    if not checkpoints:
        return {}
    latest = checkpoints[-1]
    checkpoint_id = latest.get("checkpoint_id")
    return load_checkpoint_manifest(policy, checkpoint_id) if checkpoint_id else {}


def choose_snapshot_kind(policy: dict[str, Any], requested: str, parent_manifest: dict[str, Any]) -> str:
    if requested in {"major", "minor"}:
        if requested == "minor" and not parent_manifest:
            return "major"
        if requested == "minor" and not parent_manifest.get("snapshot"):
            return "major"
        return requested
    if not parent_manifest or not parent_manifest.get("snapshot"):
        return "major"
    max_depth = int(get_path(policy, ["checkpointing", "minor_chain_max_depth"], 25))
    parent_depth = int(get_path(parent_manifest, ["snapshot", "chain_depth"], 0) or 0)
    return "major" if parent_depth >= max_depth else "minor"


def collect_workspace_snapshot_files(policy: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    include_globs = get_path(policy, ["checkpointing", "workspace_include_globs"], default_workspace_include_globs())
    exclude_patterns = get_path(policy, ["checkpointing", "workspace_exclude_patterns"], default_workspace_exclude_patterns())
    max_file_bytes = int(get_path(policy, ["checkpointing", "workspace_max_file_bytes"], 16 * 1024 * 1024))
    max_total_bytes = int(get_path(policy, ["checkpointing", "workspace_max_total_bytes"], 256 * 1024 * 1024))
    paths: list[Path] = []
    for pattern in include_globs:
        for path in ROOT.glob(pattern):
            if path.is_file() and path not in paths:
                paths.append(path)

    files: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(paths):
        rel = rel_path(path)
        if should_exclude(rel, exclude_patterns):
            continue
        size = path.stat().st_size
        if size > max_file_bytes:
            skipped.append({"path": rel, "bytes": size, "reason": f"exceeds_workspace_max_file_bytes:{max_file_bytes}"})
            continue
        if total_bytes + size > max_total_bytes:
            skipped.append({"path": rel, "bytes": size, "reason": f"exceeds_workspace_max_total_bytes:{max_total_bytes}"})
            continue
        files[rel] = {
            "bytes": size,
            "sha256": sha256(path),
            "text": is_utf8_text(path),
        }
        total_bytes += size
    return files, skipped


def create_major_snapshot(
    policy: dict[str, Any],
    checkpoint_id: str,
    checkpoint_root: Path,
    files: dict[str, dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    workspace_root = checkpoint_root / "workspace"
    for rel in sorted(files):
        source = ROOT / rel
        dest = workspace_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    state_hash = state_hash_for(files)
    chain_hash = hash_text(f"major\n{checkpoint_id}\n{state_hash}\n")
    return {
        "kind": "major",
        "major_id": checkpoint_id,
        "parent_id": None,
        "chain_depth": 0,
        "state_hash": state_hash,
        "chain_hash": chain_hash,
        "files_count": len(files),
        "bytes": sum(int(meta.get("bytes") or 0) for meta in files.values()),
        "workspace_root": str(workspace_root.relative_to(ROOT)).replace("\\", "/"),
        "files": files,
        "skipped_workspace_files": skipped,
        "policy": "full_workspace_baseline",
    }


def create_minor_snapshot(
    policy: dict[str, Any],
    checkpoint_id: str,
    checkpoint_root: Path,
    files: dict[str, dict[str, Any]],
    skipped: list[dict[str, Any]],
    parent_manifest: dict[str, Any],
    explicit_base_major: str,
) -> dict[str, Any]:
    if not parent_manifest or not parent_manifest.get("snapshot"):
        return create_major_snapshot(policy, checkpoint_id, checkpoint_root, files, skipped)

    with tempfile.TemporaryDirectory(prefix="sparkstream_parent_") as tmp:
        parent_out = Path(tmp) / "parent"
        try:
            materialized = materialize_manifest(policy, parent_manifest, parent_out, force=False)
        except Exception as exc:  # noqa: BLE001 - checkpoint continuity must degrade safely.
            materialized = {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
        if not materialized.get("ok"):
            # Minor checkpoints are a storage optimization. If a prior minor
            # delta chain can no longer materialize, preserve the current
            # workspace as a fresh major snapshot instead of breaking the
            # autonomy loop.
            return create_major_snapshot(policy, checkpoint_id, checkpoint_root, files, skipped)
        delta = create_delta_from_parent(checkpoint_root, parent_manifest, parent_out, files)

    parent_snapshot = parent_manifest.get("snapshot", {})
    state_hash = state_hash_for(files)
    parent_chain_hash = parent_snapshot.get("chain_hash") or ""
    chain_hash = hash_text(f"minor\n{checkpoint_id}\n{parent_chain_hash}\n{state_hash}\n{json.dumps(delta.get('summary', {}), sort_keys=True)}\n")
    major_id = explicit_base_major or parent_snapshot.get("major_id") or parent_manifest.get("checkpoint_id")
    return {
        "kind": "minor",
        "major_id": major_id,
        "parent_id": parent_manifest.get("checkpoint_id"),
        "chain_depth": int(parent_snapshot.get("chain_depth") or 0) + 1,
        "state_hash": state_hash,
        "parent_state_hash": parent_snapshot.get("state_hash"),
        "parent_chain_hash": parent_chain_hash,
        "chain_hash": chain_hash,
        "files_count": len(files),
        "bytes": sum(int(meta.get("bytes") or 0) for meta in files.values()),
        "files": files,
        "delta": delta,
        "skipped_workspace_files": skipped,
        "policy": "delta_from_parent_workspace_state",
    }


def create_delta_from_parent(
    checkpoint_root: Path,
    parent_manifest: dict[str, Any],
    parent_out: Path,
    current_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent_files = get_path(parent_manifest, ["snapshot", "files"], {})
    delta_root = checkpoint_root / "delta"
    additions: list[dict[str, Any]] = []
    modifications: list[dict[str, Any]] = []
    deletions: list[dict[str, Any]] = []
    current_paths = set(current_files)
    parent_paths = set(parent_files) if isinstance(parent_files, dict) else set()

    for rel in sorted(current_paths - parent_paths):
        dest = delta_root / "added" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
        additions.append({"path": rel, "bytes": current_files[rel].get("bytes"), "sha256": current_files[rel].get("sha256"), "content": rel_path(dest)})

    for rel in sorted(parent_paths - current_paths):
        deletions.append({"path": rel, "old_sha256": get_path(parent_files, [rel, "sha256"], None)})

    for rel in sorted(current_paths & parent_paths):
        cur = current_files[rel]
        old = parent_files[rel]
        if cur.get("sha256") == old.get("sha256"):
            continue
        base_path = parent_out / rel
        current_path = ROOT / rel
        modification = build_modification_delta(delta_root, rel, base_path, current_path, old, cur)
        modifications.append(modification)

    return {
        "root": rel_path(delta_root),
        "summary": {
            "added": len(additions),
            "modified": len(modifications),
            "deleted": len(deletions),
            "text_transforms": sum(1 for item in modifications if item.get("mode") == "text_line_ops"),
            "replacements": sum(1 for item in modifications if item.get("mode") == "replacement"),
        },
        "additions": additions,
        "modifications": modifications,
        "deletions": deletions,
    }


def build_modification_delta(
    delta_root: Path,
    rel: str,
    base_path: Path,
    current_path: Path,
    old_meta: dict[str, Any],
    cur_meta: dict[str, Any],
) -> dict[str, Any]:
    base_text = read_utf8_lines(base_path)
    current_text = read_utf8_lines(current_path)
    if base_text is not None and current_text is not None:
        ops = text_line_ops(base_text, current_text)
        transform_path = storage_path(delta_root / "transforms", rel, ".ops.json")
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        transform_payload = {
            "path": rel,
            "old_sha256": old_meta.get("sha256"),
            "new_sha256": cur_meta.get("sha256"),
            "ops": ops,
        }
        write_json(transform_path, transform_payload)
        diff_path = storage_path(delta_root / "diffs", rel, ".diff")
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_text = "".join(
            difflib.unified_diff(
                base_text,
                current_text,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        diff_path.write_text(diff_text, encoding="utf-8")
        return {
            "path": rel,
            "mode": "text_line_ops",
            "old_sha256": old_meta.get("sha256"),
            "new_sha256": cur_meta.get("sha256"),
            "transform": rel_path(transform_path),
            "diff": rel_path(diff_path),
            "replacement": rel_path(write_replacement_delta(delta_root, rel, current_path)),
            "op_count": len(ops),
        }

    replacement_path = write_replacement_delta(delta_root, rel, current_path)
    return {
        "path": rel,
        "mode": "replacement",
        "old_sha256": old_meta.get("sha256"),
        "new_sha256": cur_meta.get("sha256"),
        "replacement": rel_path(replacement_path),
        "bytes": cur_meta.get("bytes"),
    }


def write_replacement_delta(delta_root: Path, rel: str, current_path: Path) -> Path:
    replacement_path = delta_root / "replacements" / rel
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_path, replacement_path)
    return replacement_path


def score_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(left.keys()) | set(right.keys()))
    delta: dict[str, Any] = {}
    for key in keys:
        left_score = get_path(left, [key, "score"], None)
        right_score = get_path(right, [key, "score"], None)
        if isinstance(left_score, (int, float)) and isinstance(right_score, (int, float)):
            delta[key] = {
                "a": left_score,
                "b": right_score,
                "delta": right_score - left_score,
            }
    return delta


def snapshot_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_kind": left.get("kind"),
        "b_kind": right.get("kind"),
        "a_chain_depth": left.get("chain_depth"),
        "b_chain_depth": right.get("chain_depth"),
        "a_state_hash": left.get("state_hash"),
        "b_state_hash": right.get("state_hash"),
        "same_state": bool(left.get("state_hash") and left.get("state_hash") == right.get("state_hash")),
        "file_count_delta": int(right.get("files_count") or 0) - int(left.get("files_count") or 0),
        "byte_delta": int(right.get("bytes") or 0) - int(left.get("bytes") or 0),
        "b_delta_summary": get_path(right, ["delta", "summary"], {}),
    }


def materialize_checkpoint(policy: dict[str, Any], checkpoint_id: str, out_dir: Path, force: bool) -> dict[str, Any]:
    manifest = load_checkpoint_manifest(policy, checkpoint_id)
    if not manifest:
        return {"ok": False, "error": "checkpoint_not_found", "checkpoint_id": checkpoint_id}
    return materialize_manifest(policy, manifest, out_dir, force)


def materialize_manifest(policy: dict[str, Any], manifest: dict[str, Any], out_dir: Path, force: bool) -> dict[str, Any]:
    resolved = out_dir.resolve()
    if resolved.exists():
        if not force:
            occupied = any(resolved.iterdir()) if resolved.is_dir() else True
            if occupied:
                return {"ok": False, "error": "output_exists", "out": str(resolved)}
        elif not is_safe_materialize_delete(resolved):
            return {"ok": False, "error": "unsafe_force_output_path", "out": str(resolved)}
        elif resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    resolved.mkdir(parents=True, exist_ok=True)

    chain = checkpoint_chain(policy, manifest)
    if not chain or get_path(chain[0], ["snapshot", "kind"], None) != "major":
        return {"ok": False, "error": "major_baseline_not_found", "checkpoint_id": manifest.get("checkpoint_id")}

    major_root = ROOT / str(get_path(chain[0], ["snapshot", "workspace_root"], ""))
    if not major_root.exists():
        return {"ok": False, "error": "major_workspace_missing", "major_id": chain[0].get("checkpoint_id")}
    copy_tree_contents(major_root, resolved)
    applied: list[str] = [str(chain[0].get("checkpoint_id"))]
    for item in chain[1:]:
        try:
            apply_delta(item, resolved)
        except Exception as exc:  # noqa: BLE001 - materialize should report, not crash.
            return {
                "ok": False,
                "error": "delta_apply_failed",
                "detail": str(exc),
                "checkpoint_id": manifest.get("checkpoint_id"),
                "failed_delta": item.get("checkpoint_id"),
                "out": str(resolved),
                "applied": applied,
            }
        applied.append(str(item.get("checkpoint_id")))

    expected_hash = get_path(manifest, ["snapshot", "state_hash"], None)
    actual_files = hash_materialized_files(resolved, get_path(manifest, ["snapshot", "files"], {}))
    actual_hash = state_hash_for(actual_files)
    return {
        "ok": actual_hash == expected_hash,
        "checkpoint_id": manifest.get("checkpoint_id"),
        "out": str(resolved),
        "applied": applied,
        "expected_state_hash": expected_hash,
        "actual_state_hash": actual_hash,
        "files_count": len(actual_files),
    }


def checkpoint_chain(policy: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    chain = [manifest]
    cur = manifest
    seen = {manifest.get("checkpoint_id")}
    while get_path(cur, ["snapshot", "kind"], None) == "minor":
        parent_id = get_path(cur, ["snapshot", "parent_id"], None)
        if not parent_id or parent_id in seen:
            return []
        parent = load_checkpoint_manifest(policy, str(parent_id))
        if not parent:
            return []
        chain.append(parent)
        seen.add(parent.get("checkpoint_id"))
        cur = parent
    return list(reversed(chain))


def apply_delta(manifest: dict[str, Any], out_dir: Path) -> None:
    delta = get_path(manifest, ["snapshot", "delta"], {})
    for item in delta.get("deletions", []):
        target = out_dir / item["path"]
        if target.exists() and target.is_file():
            target.unlink()
    for item in delta.get("additions", []):
        copy_delta_file(item.get("content"), out_dir / item["path"])
    for item in delta.get("modifications", []):
        target = out_dir / item["path"]
        if item.get("mode") == "replacement":
            copy_delta_file(item.get("replacement"), target)
        elif item.get("mode") == "text_line_ops":
            transform = read_json(ROOT / str(item.get("transform")))
            try:
                apply_text_transform(target, transform)
            except Exception:
                replacement = item.get("replacement")
                if not replacement:
                    raise
                copy_delta_file(replacement, target)


def copy_delta_file(source_rel: str | None, target: Path) -> None:
    if not source_rel:
        return
    source = ROOT / source_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def apply_text_transform(target: Path, transform: dict[str, Any]) -> None:
    base = read_utf8_lines(target)
    if base is None:
        raise ValueError(f"cannot apply text transform to non-utf8 file: {target}")
    result: list[str] = []
    cursor = 0
    for op in transform.get("ops", []):
        i1 = int(op["i1"])
        i2 = int(op["i2"])
        result.extend(base[cursor:i1])
        if op.get("tag") in {"replace", "insert"}:
            result.extend(op.get("lines", []))
        cursor = i2
    result.extend(base[cursor:])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(result))
    expected = transform.get("new_sha256")
    if expected and sha256(target) != expected:
        raise ValueError(f"text transform hash mismatch: {target}")


def hash_materialized_files(root: Path, expected_files: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for rel in sorted(expected_files):
        path = root / rel
        if path.exists() and path.is_file():
            rows[rel] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "text": is_utf8_text(path),
            }
    return rows


def copy_tree_contents(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def is_safe_materialize_delete(path: Path) -> bool:
    materialized_root = (ROOT / "checkpoints" / "materialized").resolve()
    try:
        path.relative_to(materialized_root)
        return True
    except ValueError:
        return False


def resolve_artifact_paths(policy: dict[str, Any]) -> list[Path]:
    patterns = get_path(policy, ["checkpointing", "artifact_globs"], [])
    paths: list[Path] = []
    for pattern in patterns:
        abs_pattern = str(ROOT / pattern)
        matched = [Path(path) for path in ROOT.glob(pattern)]
        if not matched:
            # Path.glob does not support absolute-style strings on all Windows
            # forms; keep the intent visible for future debugging.
            explicit = Path(abs_pattern)
            if explicit.exists():
                matched = [explicit]
        for path in matched:
            if path.is_file() and path not in paths:
                paths.append(path)
    return sorted(paths)


def default_workspace_include_globs() -> list[str]:
    return [
        ".gitignore",
        "Cargo.toml",
        "Cargo.lock",
        "README.md",
        "configs/**/*.json",
        "configs/**/*.toml",
        "scripts/**/*.py",
        "dashboard/**/*",
        "docs/**/*.md",
        "src/**/*",
        "crates/**/*.rs",
        "crates/**/Cargo.toml",
        "tests/**/*",
        "examples/**/*",
        "adapters/**/*.py",
        "adapters/**/*.md",
        "benchmarks/**/*.json",
        "benchmarks/**/*.jsonl",
        "data/*.json",
        "data/*.jsonl",
    ]


def default_workspace_exclude_patterns() -> list[str]:
    return [
        ".git/**",
        "target/**",
        "checkpoints/**",
        "reports/**",
        ".venv*/**",
        "data/public_benchmarks/**/.git/**",
        "data/public_benchmarks/**/raw_results/**",
        "**/__pycache__/**",
        "**/*.pyc",
    ]


def should_exclude(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def is_utf8_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def read_utf8_lines(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.readlines()
    except (UnicodeDecodeError, OSError):
        return None


def text_line_ops(base: list[str], current: list[str]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=base, b=current, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        op: dict[str, Any] = {"tag": tag, "i1": i1, "i2": i2}
        if tag in {"replace", "insert"}:
            op["lines"] = current[j1:j2]
        ops.append(op)
    return ops


def storage_path(root: Path, rel: str, suffix: str) -> Path:
    rel_path_obj = Path(rel)
    filename = rel_path_obj.name + suffix
    return root / rel_path_obj.parent / filename


def state_hash_for(files: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(files):
        meta = files[rel]
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(meta.get("bytes", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(meta.get("sha256", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def registry_path(policy: dict[str, Any]) -> str:
    return get_path(policy, ["checkpointing", "registry"], "reports/checkpoint_registry.json")


def git_state() -> dict[str, Any]:
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "dirty": bool(run_text(["git", "status", "--short"])),
    }


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned[:80] or "checkpoint"


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
