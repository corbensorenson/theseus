"""Accepted-candidate backup manager for Project Theseus.

This intentionally backs up *accepted* candidates, not every experiment. By
default it only writes reports. With --execute and a promoted candidate, it can
write a small tracked backup manifest and push the current git branch. Large
ignored artifacts, ROMs, datasets, model binaries, and generated reports stay
out of the backup scope unless a future policy explicitly allows them.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "checkpoint_backup_policy.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--if-promoted", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--provider", choices=["all", "github", "google_drive"], default="all")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    report_path = ROOT / (args.out or get_path(policy, ["reports", "last"], "reports/checkpoint_backup_last.json"))
    report = build_report(policy, args)
    write_json(report_path, report)
    append_jsonl(ROOT / get_path(policy, ["reports", "history"], "reports/checkpoint_backup_history.jsonl"), compact_history(report))
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") or report.get("status") in {"skipped_not_promoted", "dry_run_ready"} else 1


def build_report(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reports = ROOT / "reports"
    candidate = read_json(ROOT / get_path(policy, ["trigger", "candidate_gate"], "reports/candidate_promotion_gate.json"))
    checkpoint_last = read_json(ROOT / get_path(policy, ["trigger", "checkpoint_last"], "reports/checkpoint_last.json"))
    registry = read_json(ROOT / get_path(policy, ["trigger", "checkpoint_registry"], "reports/checkpoint_registry.json"))
    git = git_state()
    promoted = bool(candidate.get("promote"))
    checkpoint = resolve_checkpoint(policy, args.checkpoint_id, checkpoint_last, registry)
    checkpoint_id = checkpoint.get("checkpoint_id") or args.checkpoint_id or checkpoint_last.get("checkpoint_id")
    required_status = str(get_path(policy, ["trigger", "require_checkpoint_status"], "promoted"))
    checkpoint_status = checkpoint.get("status") or checkpoint_last.get("status")
    checkpoint_promoted = bool(required_status and checkpoint_status == required_status)
    base = {
        "policy": "project_theseus_checkpoint_backup_report_v0",
        "created_utc": now(),
        "policy_file": str(Path(args.policy)),
        "execute": bool(args.execute),
        "provider": args.provider,
        "candidate_promote": promoted,
        "candidate_gate": f"{candidate.get('passed')}/{candidate.get('total')}",
        "checkpoint_id": checkpoint_id,
        "checkpoint_status": checkpoint_status,
        "git": git,
        "backup_scope": policy.get("backup_scope", {}),
        "external_inference_calls": 0,
    }

    if args.if_promoted and not promoted and not checkpoint_promoted:
        return {
            **base,
            "ok": True,
            "status": "skipped_not_promoted",
            "reason": "Neither the current candidate gate nor the resolved checkpoint is promoted, so no accepted-candidate backup is created.",
        }
    if not checkpoint_id or not checkpoint:
        return {**base, "ok": False, "status": "blocked_missing_checkpoint", "reason": "No checkpoint manifest could be resolved."}
    if args.if_promoted and required_status and checkpoint_status != required_status:
        return {
            **base,
            "ok": False,
            "status": "blocked_checkpoint_not_promoted",
            "reason": f"Checkpoint status must be {required_status!r} for accepted-candidate backup.",
        }

    manifest = backup_manifest(policy, candidate, checkpoint, git)
    forbidden_hits = forbidden_manifest_hits(policy, manifest)
    if forbidden_hits:
        return {
            **base,
            "ok": False,
            "status": "blocked_forbidden_scope",
            "reason": "Backup manifest references forbidden paths.",
            "forbidden_hits": forbidden_hits,
            "manifest_preview": manifest,
        }

    provider_reports: list[dict[str, Any]] = []
    if args.provider in {"all", "github"}:
        provider_reports.append(handle_github(policy, manifest, git, args.execute))
    if args.provider in {"all", "google_drive"}:
        provider_reports.append(handle_google_drive(policy, manifest, args.execute))
    ok = all(item.get("ok", False) for item in provider_reports if item.get("enabled", True))
    status = "backup_complete" if ok and args.execute else "dry_run_ready"
    update_offer = create_update_offer(policy, checkpoint_id)
    return {
        **base,
        "ok": ok,
        "status": status,
        "manifest": manifest,
        "providers": provider_reports,
        "update_offer": update_offer,
    }


def create_update_offer(policy: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    if not get_path(policy, ["update_offer", "enabled"], True):
        return {"enabled": False, "ok": True, "status": "disabled"}
    out = str(get_path(policy, ["update_offer", "report"], "reports/update_offer_current.json"))
    command = [
        sys.executable,
        "scripts/update_manager.py",
        "create",
        "--if-promoted",
        "--checkpoint-id",
        checkpoint_id,
        "--out",
        out,
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"enabled": True, "ok": False, "status": "update_offer_failed", "error": str(exc)}
    payload = parse_json(result.stdout)
    return {
        "enabled": True,
        "ok": result.returncode == 0,
        "status": payload.get("status") or ("created" if result.returncode == 0 else "failed"),
        "update_id": payload.get("update_id"),
        "checkpoint_id": payload.get("checkpoint_id"),
        "report": out,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-2000:],
    }


def backup_manifest(policy: dict[str, Any], candidate: dict[str, Any], checkpoint: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    artifacts = checkpoint.get("artifacts") or []
    snapshot = checkpoint.get("snapshot") or {}
    allowed_artifacts, excluded_artifacts = partition_allowed_artifacts(policy, artifacts)
    return {
        "policy": "project_theseus_accepted_candidate_backup_manifest_v0",
        "created_utc": now(),
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "label": checkpoint.get("label"),
        "reason": checkpoint.get("reason"),
        "profile": checkpoint.get("profile"),
        "status": checkpoint.get("status"),
        "git": git,
        "promotion": checkpoint.get("promotion") or {
            "promote": candidate.get("promote"),
            "passed": candidate.get("passed"),
            "total": candidate.get("total"),
        },
        "scores": checkpoint.get("scores", {}),
        "snapshot": {
            "kind": snapshot.get("kind"),
            "major_id": snapshot.get("major_id"),
            "parent_id": snapshot.get("parent_id"),
            "chain_depth": snapshot.get("chain_depth"),
            "state_hash": snapshot.get("state_hash"),
            "chain_hash": snapshot.get("chain_hash"),
            "files_count": snapshot.get("files_count"),
            "bytes": snapshot.get("bytes"),
        },
        "artifacts": [
            {
                "source": item.get("source"),
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for item in allowed_artifacts
        ],
        "excluded_artifacts_by_policy": excluded_artifacts,
        "excluded_by_policy": policy.get("forbidden_paths", []),
        "backup_scope": policy.get("backup_scope", {}),
        "external_inference_calls": 0,
    }


def partition_allowed_artifacts(policy: dict[str, Any], artifacts: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patterns = policy.get("forbidden_paths") or []
    allowed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in artifacts if isinstance(artifacts, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").replace("\\", "/")
        row = {"source": source, "bytes": item.get("bytes"), "sha256": item.get("sha256")}
        if source and any(fnmatch.fnmatch(source, pattern) for pattern in patterns):
            row["reason"] = "forbidden_path_policy"
            excluded.append(row)
        else:
            allowed.append(row)
    return allowed, excluded


def handle_github(policy: dict[str, Any], manifest: dict[str, Any], git: dict[str, Any], execute: bool) -> dict[str, Any]:
    cfg = get_path(policy, ["providers", "github"], {})
    if not cfg.get("enabled", False):
        return {"provider": "github", "enabled": False, "ok": True, "status": "disabled"}
    remote = str(cfg.get("remote") or "origin")
    manifest_root = ROOT / str(cfg.get("manifest_root") or "backup_manifests/accepted_candidates")
    manifest_path = manifest_root / f"{slug(str(manifest.get('checkpoint_id') or 'checkpoint'))}.json"
    report = {
        "provider": "github",
        "enabled": True,
        "remote": remote,
        "branch": git.get("branch"),
        "manifest_path": rel_path(manifest_path),
        "execute": execute,
        "ok": False,
    }
    remote_url = git_command(["remote", "get-url", remote], check=False)
    if remote_url["returncode"] != 0:
        return {**report, "status": "blocked_missing_remote", "remote_check": remote_url}
    report["remote_url_present"] = True
    if git.get("dirty") and bool(get_path(policy, ["trigger", "require_clean_git_for_push"], True)):
        return {
            **report,
            "status": "blocked_dirty_worktree",
            "reason": "Policy requires a clean worktree before pushing accepted-candidate backup.",
            "porcelain_sample": git.get("porcelain", [])[:20],
        }
    if not execute:
        return {**report, "ok": True, "status": "dry_run_manifest_would_be_written_and_pushed"}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    add = git_command(["add", rel_path(manifest_path)], check=False)
    if add["returncode"] != 0:
        return {**report, "status": "git_add_failed", "git_add": add}
    commit_message = str(cfg.get("commit_message_template") or "backup: accepted candidate {checkpoint_id}").format(
        checkpoint_id=manifest.get("checkpoint_id") or "checkpoint"
    )
    commit = git_command(["commit", "-m", commit_message], check=False)
    if commit["returncode"] != 0 and "nothing to commit" not in (commit.get("stdout", "") + commit.get("stderr", "")).lower():
        return {**report, "status": "git_commit_failed", "git_commit": commit}
    push = git_command(["push", remote, str(git.get("branch") or "HEAD")], check=False)
    if push["returncode"] != 0:
        return {**report, "status": "git_push_failed", "git_commit": commit, "git_push": push}
    return {**report, "ok": True, "status": "pushed", "git_commit": commit, "git_push": push}


def handle_google_drive(policy: dict[str, Any], manifest: dict[str, Any], execute: bool) -> dict[str, Any]:
    cfg = get_path(policy, ["providers", "google_drive"], {})
    if not cfg.get("enabled", False):
        return {"provider": "google_drive", "enabled": False, "ok": True, "status": "disabled"}
    queue_path = ROOT / str(cfg.get("queue_path") or "reports/google_drive_backup_queue.jsonl")
    payload = {
        "created_utc": now(),
        "provider": "google_drive",
        "folder_name": cfg.get("folder_name"),
        "checkpoint_id": manifest.get("checkpoint_id"),
        "manifest": manifest,
        "status": "queued_for_connector_upload",
        "notes": cfg.get("notes"),
    }
    if not execute:
        return {
            "provider": "google_drive",
            "enabled": True,
            "ok": True,
            "status": "dry_run_queue_item_would_be_written",
            "queue_path": rel_path(queue_path),
        }
    if cfg.get("queue_only_by_default", True):
        append_jsonl(queue_path, payload)
        return {
            "provider": "google_drive",
            "enabled": True,
            "ok": True,
            "status": "queued_manifest_only",
            "queue_path": rel_path(queue_path),
        }
    append_jsonl(queue_path, payload)
    return {
        "provider": "google_drive",
        "enabled": True,
        "ok": True,
        "status": "queued_manifest_only_no_direct_connector_in_cli",
        "queue_path": rel_path(queue_path),
    }


def resolve_checkpoint(
    policy: dict[str, Any],
    checkpoint_id: str,
    checkpoint_last: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    root = ROOT / "checkpoints"
    if checkpoint_id:
        direct = root / checkpoint_id / "manifest.json"
        return read_json(direct)
    if checkpoint_last.get("checkpoint_id"):
        return checkpoint_last
    checkpoints = registry.get("checkpoints") if isinstance(registry, dict) else []
    if isinstance(checkpoints, list) and checkpoints:
        for item in reversed(checkpoints):
            if item.get("promote") or item.get("status") == "promoted":
                return read_json(root / str(item.get("checkpoint_id")) / "manifest.json")
        item = checkpoints[-1]
        return read_json(root / str(item.get("checkpoint_id")) / "manifest.json")
    return {}


def forbidden_manifest_hits(policy: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    patterns = policy.get("forbidden_paths") or []
    candidates: list[str] = []
    for item in manifest.get("artifacts", []):
        if item.get("source"):
            candidates.append(str(item["source"]).replace("\\", "/"))
    for value in manifest.get("excluded_by_policy", []):
        if isinstance(value, str):
            continue
    hits = []
    for path in candidates:
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            hits.append(path)
    return hits


def git_state() -> dict[str, Any]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        porcelain = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
        return {
            "available": True,
            "branch": branch,
            "commit": commit,
            "dirty": bool(porcelain),
            "porcelain_count": len(porcelain),
            "porcelain": porcelain,
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


def git_command(args: list[str], check: bool = False) -> dict[str, Any]:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return {
        "command": ["git", *args],
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def compact_history(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_utc": report.get("created_utc"),
        "status": report.get("status"),
        "ok": report.get("ok"),
        "execute": report.get("execute"),
        "provider": report.get("provider"),
        "checkpoint_id": report.get("checkpoint_id"),
        "candidate_promote": report.get("candidate_promote"),
        "providers": [
            {
                "provider": item.get("provider"),
                "status": item.get("status"),
                "ok": item.get("ok"),
            }
            for item in report.get("providers", [])
            if isinstance(item, dict)
        ],
        "update_offer": {
            "status": get_path(report, ["update_offer", "status"], None),
            "ok": get_path(report, ["update_offer", "ok"], None),
            "update_id": get_path(report, ["update_offer", "update_id"], None),
        },
    }


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")[:120] or "checkpoint"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
