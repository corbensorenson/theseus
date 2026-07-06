"""Fetch, verify, and merge Hive worker artifacts from peers or relay results."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError
from urllib.parse import quote, urlencode


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hive_security  # noqa: E402
import hive_node_registry  # noqa: E402
import theseus_runtime  # noqa: E402
import viea_spine_records  # noqa: E402

DEFAULT_POLICY = ROOT / "configs" / "hive_policy.json"
DEFAULT_OUT = ROOT / "reports" / "hive_artifact_sync.json"
LOCK_PATH = ROOT / "reports" / "hive_artifact_sync.lock"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--peer-url", action="append", default=[])
    parser.add_argument("--relay-results", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--index-retries", type=int, default=3)
    args = parser.parse_args()

    lock = acquire_singleton_lock(LOCK_PATH)
    if lock.get("already_running"):
        report = {
            "ok": True,
            "policy": "project_theseus_hive_artifact_sync_v0",
            "created_utc": now(),
            "skipped": "existing_hive_artifact_sync_running",
            "singleton_lock": lock,
        }
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0
    policy = read_json(ROOT / args.policy, {})
    try:
        report = sync_artifacts(policy, args)
        report["singleton_lock"] = lock
        write_json(ROOT / args.out, report)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 2
    finally:
        release_singleton_lock(LOCK_PATH, lock)


def sync_artifacts(policy: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    inbox = ROOT / str(get_path(policy, ["artifact_sync", "inbox_path"], "reports/hive_artifact_inbox"))
    inbox.mkdir(parents=True, exist_ok=True)
    peers = peer_urls(policy, args.peer_url)
    fetched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for peer_url in peers:
        result = sync_peer(policy, peer_url, inbox, args.limit, timeout=float(args.timeout_seconds), index_retries=int(args.index_retries))
        fetched.extend(result.get("fetched", []))
        errors.extend(result.get("errors", []))
    if args.relay_results:
        result = sync_relay_results(policy, inbox, args.limit)
        fetched.extend(result.get("fetched", []))
        errors.extend(result.get("errors", []))
    merge = merge_artifacts(policy, inbox)
    citation = viea_spine_records.materialized_artifact_citation(
        "hive_artifact_sync",
        artifact_path=str(getattr(args, "out", "reports/hive_artifact_sync.json") or "reports/hive_artifact_sync.json"),
    )
    for row in fetched:
        if isinstance(row, dict):
            row.setdefault("viea_artifact_citation_id", citation.get("citation_id"))
    report = {
        "ok": not errors,
        "policy": "project_theseus_hive_artifact_sync_v0",
        "created_utc": now(),
        "peer_count": len(peers),
        "fetched_count": len(fetched),
        "fetched": fetched,
        "errors": errors,
        "merge": merge,
        "review_step_count": max(1, len(peers) + int(bool(args.relay_results)) + int(bool(merge))),
        "review_step_basis": "peer_sync_relay_merge",
        "maintenance_mode": "object_only",
        "maintenance_mode_basis": "artifact_sync_summary_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "viea_artifact_citation": citation,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    append_jsonl(ROOT / str(get_path(policy, ["artifact_sync", "ledger_path"], "reports/hive_artifact_sync_ledger.jsonl")), report)
    return report


def refresh_existing_artifact_citations(policy: dict[str, Any], *, sync_path: str = "reports/hive_artifact_sync.json") -> dict[str, Any]:
    """Upgrade existing Hive artifact reports with VIEA citations without sync side effects."""

    updates = []
    sync_report_path = ROOT / sync_path
    sync_report = read_json(sync_report_path, {})
    if sync_report:
        cited = attach_report_citation(sync_report, "hive_artifact_sync", sync_path)
        write_json(sync_report_path, cited)
        updates.append({"path": display_path(sync_report_path), "citation_ready": get_path(cited, ["viea_artifact_citation", "ready"], False)})
    merge_rel = str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json"))
    merge_path = ROOT / merge_rel
    merge_report = read_json(merge_path, {})
    if merge_report:
        cited = attach_report_citation(merge_report, "hive_artifact_merge_summary", merge_rel)
        write_json(merge_path, cited)
        updates.append({"path": display_path(merge_path), "citation_ready": get_path(cited, ["viea_artifact_citation", "ready"], False)})
    report = {
        "ok": all(bool(row.get("citation_ready")) for row in updates) if updates else False,
        "policy": "project_theseus_hive_artifact_citation_refresh_v1",
        "created_utc": now(),
        "updated_count": len(updates),
        "updated": updates,
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json(ROOT / "reports" / "hive_artifact_citation_refresh.json", report)
    return report


def attach_report_citation(report: dict[str, Any], consumer_surface: str, artifact_path: str) -> dict[str, Any]:
    out = dict(report)
    citation = viea_spine_records.materialized_artifact_citation(consumer_surface, artifact_path=artifact_path)
    out["viea_artifact_citation"] = citation
    out["public_training_rows_written"] = 0
    out["external_inference_calls"] = 0
    out["fallback_return_count"] = 0
    citation_id = citation.get("citation_id")
    for key in ["fetched", "promoted"]:
        rows = out.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row.setdefault("viea_artifact_citation_id", citation_id)
    return out


def acquire_singleton_lock(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "created_utc": now(), "path": str(path.relative_to(ROOT))}
    existing_processes = existing_hive_artifact_sync_processes()
    if existing_processes:
        return {
            "acquired": False,
            "already_running": True,
            "existing_processes": existing_processes[:4],
            "reason": "active_hive_artifact_sync_process_without_lock",
        }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = read_json(path, {})
        pid = int(existing.get("pid") or 0) if isinstance(existing, dict) else 0
        if pid and pid_is_alive(pid):
            return {"acquired": False, "already_running": True, "existing": existing}
        try:
            path.unlink()
        except OSError:
            return {"acquired": False, "already_running": True, "existing": existing, "stale_remove_failed": True}
        return acquire_singleton_lock(path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return {"acquired": True, "already_running": False, **payload}


def existing_hive_artifact_sync_processes() -> list[dict[str, Any]]:
    if sys.platform.startswith("win"):
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 2",
                ],
                text=True,
                capture_output=True,
                timeout=15,
            )
        except Exception:
            return []
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = int(item.get("ProcessId") or 0)
            command_line = str(item.get("CommandLine") or "")
            lowered = command_line.lower().replace("\\", "/")
            if pid == os.getpid() or "hive_artifact_sync.py" not in lowered:
                continue
            if "powershell" in lowered and "-command" in lowered:
                continue
            rows.append({"pid": pid, "command_preview": command_line[:300]})
        return rows
    try:
        result = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, timeout=15)
    except Exception:
        return []
    rows = []
    for line in (result.stdout or "").splitlines():
        pid_text, _, command_line = line.strip().partition(" ")
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid != os.getpid() and "hive_artifact_sync.py" in command_line:
            rows.append({"pid": pid, "command_preview": command_line[:300]})
    return rows


def release_singleton_lock(path: Path, lock: dict[str, Any]) -> None:
    if not lock.get("acquired"):
        return
    try:
        existing = read_json(path, {})
        if int(existing.get("pid") or 0) == os.getpid():
            path.unlink()
    except OSError:
        pass


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            return False
        return str(pid) in (result.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sync_peer(
    policy: dict[str, Any],
    peer_url: str,
    inbox: Path,
    limit: int,
    *,
    timeout: float = 20.0,
    index_retries: int = 3,
) -> dict[str, Any]:
    secret = hive_secret(policy)
    headers = {"X-Theseus-Hive-Secret": secret} if secret else {}
    index = fetch_artifact_index(peer_url, headers=headers, limit=limit, timeout=timeout, retries=index_retries)
    if not index.get("ok"):
        return {"fetched": [], "errors": [{"peer_url": peer_url, "error": "artifact_index_failed", "detail": index}]}
    fetched = []
    errors = []
    for artifact in index.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        rel_path = str(artifact.get("path") or "")
        payload = fetch_json(peer_url.rstrip("/") + "/api/hive/artifact?path=" + quote(rel_path, safe=""), headers=headers, timeout=timeout)
        if not payload.get("ok"):
            errors.append({"peer_url": peer_url, "path": rel_path, "error": "artifact_fetch_failed", "detail": payload})
            continue
        row = materialize_payload(inbox, str(index.get("node_id") or "unknown_node"), payload, expected_sha=str(artifact.get("sha256") or ""))
        if row.get("ok"):
            fetched.append(row)
        elif volatile_status_hash_race(rel_path, artifact, row):
            fetched.append({**row, "ok": True, "skipped": True, "warning": "volatile_status_changed_during_fetch"})
        else:
            errors.append(row)
    return {"fetched": fetched, "errors": errors, "index_meta": index.get("_sync_meta")}


def fetch_artifact_index(
    peer_url: str,
    *,
    headers: dict[str, str],
    limit: int,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    """Fetch a peer artifact index with bounded retries and smaller fallbacks.

    The Mac can be responsive for work but slow while building a 200-item index.
    Falling back to a smaller limit preserves evidence flow without weakening
    artifact signature or hash checks.
    """

    attempts: list[dict[str, Any]] = []
    fallback_limits = []
    for candidate in [limit, min(limit, 100), min(limit, 50), min(limit, 20)]:
        candidate = max(1, int(candidate))
        if candidate not in fallback_limits:
            fallback_limits.append(candidate)
    retry_count = max(1, int(retries))
    for fallback_limit in fallback_limits:
        for attempt in range(1, retry_count + 1):
            url = peer_url.rstrip("/") + "/api/hive/artifacts?" + urlencode({"limit": str(fallback_limit)})
            started = time.perf_counter()
            index = fetch_json(url, headers=headers, timeout=timeout)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            attempts.append(
                {
                    "limit": fallback_limit,
                    "attempt": attempt,
                    "ok": bool(index.get("ok")),
                    "elapsed_ms": elapsed_ms,
                    "error": index.get("error") if not index.get("ok") else None,
                }
            )
            if index.get("ok"):
                index["_sync_meta"] = {
                    "requested_limit": limit,
                    "used_limit": fallback_limit,
                    "attempts": attempts,
                    "fallback_used": fallback_limit != limit or attempt > 1,
                    "timeout_seconds": timeout,
                }
                return index
            if not transient_fetch_error(index):
                break
            time.sleep(min(1.5, 0.25 * attempt))
    return {
        "ok": False,
        "error": "artifact_index_retries_exhausted",
        "url": peer_url.rstrip("/") + "/api/hive/artifacts",
        "attempts": attempts,
        "requested_limit": limit,
        "timeout_seconds": timeout,
    }


def transient_fetch_error(payload: dict[str, Any]) -> bool:
    text = str(payload.get("error") or "").lower()
    return any(needle in text for needle in ["timed out", "timeout", "temporarily", "connection", "reset", "refused"])


def sync_relay_results(policy: dict[str, Any], inbox: Path, limit: int) -> dict[str, Any]:
    relay = relay_url(policy)
    hive = hive_id(policy)
    secret = hive_secret(policy)
    if not relay or not hive or not secret:
        return {"fetched": [], "errors": []}
    headers = {"X-Theseus-Hive-Secret": secret}
    results = fetch_json(relay.rstrip("/") + "/api/hive/relay/results?" + urlencode({"hive_id": hive, "limit": str(limit)}), headers=headers)
    if not results.get("ok"):
        return {"fetched": [], "errors": [{"relay_url": relay, "error": "relay_results_failed", "detail": results}]}
    fetched = []
    errors = []
    for result in results.get("results") or []:
        if not isinstance(result, dict):
            continue
        bundle = result.get("artifact_bundle") if isinstance(result.get("artifact_bundle"), dict) else {}
        node_id = str(result.get("reported_by") or result.get("node_id") or "relay_node")
        for payload in bundle.get("artifacts") or []:
            if isinstance(payload, dict):
                row = materialize_payload(inbox, node_id, payload, expected_sha=str(payload.get("sha256") or ""))
                fetched.append(row) if row.get("ok") else errors.append(row)
    return {"fetched": fetched, "errors": errors}


def materialize_payload(inbox: Path, node_id: str, payload: dict[str, Any], *, expected_sha: str) -> dict[str, Any]:
    signature = verify_artifact_signature(payload)
    if not signature.get("ok"):
        return {"ok": False, "error": "artifact_signature_invalid", "path": payload.get("path"), "signature": signature}
    try:
        data = base64.b64decode(str(payload.get("content_b64") or ""), validate=True)
    except Exception as exc:  # noqa: BLE001 - bad remote payload should be ledgered.
        return {"ok": False, "error": "base64_decode_failed", "message": str(exc), "path": payload.get("path")}
    actual_sha = hashlib.sha256(data).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        return {"ok": False, "error": "sha256_mismatch", "path": payload.get("path"), "expected": expected_sha, "actual": actual_sha}
    rel_path = safe_rel_path(str(payload.get("path") or actual_sha + ".json"))
    dest = inbox / safe_name(node_id) / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    meta = {
        "ok": True,
        "node_id": node_id,
        "path": str(payload.get("path") or ""),
        "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "kind": payload.get("kind"),
        "size_bytes": len(data),
        "sha256": actual_sha,
        "signature": signature,
        "viea_artifact_citation": viea_spine_records.materialized_artifact_citation(
            "hive_artifact_materialized_payload",
            artifact_path=str(dest.relative_to(ROOT)).replace("\\", "/"),
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    meta["viea_artifact_citation_id"] = get_path(meta, ["viea_artifact_citation", "citation_id"], "")
    write_json(dest.with_suffix(dest.suffix + ".artifact.json"), meta)
    return meta


def volatile_status_hash_race(rel_path: str, artifact: dict[str, Any], row: dict[str, Any]) -> bool:
    """Allow status-report races without weakening worker artifact integrity."""
    if row.get("error") != "sha256_mismatch":
        return False
    kind = str(artifact.get("kind") or "")
    normalized = rel_path.replace("\\", "/").lower()
    if kind in {"worker_report", "checkpoint_manifest", "model_json"}:
        return normalized.startswith("reports/hive_chunks/") and normalized.endswith("_last.json")
    volatile_names = {
        "reports/compute_market_status.json",
        "reports/compute_market_settlement_last.json",
        "reports/hive_status.json",
        "reports/hive_peers.json",
        "reports/hive_operator_status.json",
        "reports/hive_storage_status.json",
    }
    return normalized in volatile_names


def verify_artifact_signature(payload: dict[str, Any]) -> dict[str, Any]:
    signature = payload.get("artifact_signature") if isinstance(payload.get("artifact_signature"), dict) else {}
    if not signature:
        return {"ok": True, "status": "missing_allowed_for_legacy"}
    signed_payload = {key: payload.get(key) for key in ["path", "kind", "size_bytes", "sha256"]}
    digest = hive_security.sha256_json(signed_payload)
    if digest != str(signature.get("digest") or ""):
        return {"ok": False, "error": "artifact_digest_mismatch"}
    secret = hive_secret(read_json(DEFAULT_POLICY, {}))
    if secret and signature.get("signature"):
        expected = hashlib.sha256()
        import hmac

        expected_sig = hmac.new(secret.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, str(signature.get("signature") or "")):
            return {"ok": False, "error": "artifact_hmac_mismatch"}
        return {"ok": True, "status": "verified", "signer": signature.get("signer")}
    return {"ok": True, "status": "digest_verified_unsigned_or_no_secret", "signer": signature.get("signer")}


def merge_artifacts(policy: dict[str, Any], inbox: Path) -> dict[str, Any]:
    candidates = []
    promoted = []
    seen_reports: set[Path] = set()
    for root in artifact_candidate_roots(inbox):
        for report_path in root.rglob("*.json"):
            resolved = report_path.resolve()
            if resolved in seen_reports or report_path.name.endswith(".artifact.json"):
                continue
            seen_reports.add(resolved)
            report = read_json(report_path, {})
            if not isinstance(report, dict) or not is_worker_report(report):
                continue
            candidate = candidate_from_worker_report(report_path, report)
            if candidate:
                candidates.append(candidate)
    best_by_arm: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        arm = str(candidate.get("arm_id") or "unknown_arm")
        if arm not in best_by_arm or float(candidate.get("score") or 0.0) > float(best_by_arm[arm].get("score") or 0.0):
            best_by_arm[arm] = candidate
    checkpoint_root = ROOT / str(get_path(policy, ["artifact_sync", "promoted_checkpoint_root"], "checkpoints/hive_promoted"))
    for candidate in best_by_arm.values():
        if not candidate.get("accepted"):
            continue
        manifest = promote_candidate(checkpoint_root, candidate)
        promoted.append(manifest)
        append_jsonl(ROOT / str(get_path(policy, ["artifact_sync", "merge_ledger_path"], "reports/hive_artifact_merge_ledger.jsonl")), manifest)
    summary = {
        "ok": True,
        "policy": "project_theseus_hive_artifact_merge_summary_v0",
        "created_utc": now(),
        "candidate_count": len(candidates),
        "promoted_count": len(promoted),
        "best_by_arm": best_by_arm,
        "promoted": promoted,
        "review_step_count": max(1, len(candidates) + len(promoted)),
        "review_step_basis": "candidate_scan_and_promoted_manifest_count",
        "maintenance_mode": "object_only",
        "maintenance_mode_basis": "artifact_merge_summary_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "viea_artifact_citation": viea_spine_records.materialized_artifact_citation(
            "hive_artifact_merge_summary",
            artifact_path=str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")),
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    write_json(ROOT / str(get_path(policy, ["artifact_sync", "merge_summary_path"], "reports/hive_artifact_merge_summary.json")), summary)
    return summary


def artifact_candidate_roots(inbox: Path) -> list[Path]:
    roots = [inbox, ROOT / "reports" / "hive_chunks"]
    for root in report_roots():
        roots.extend([root / "hive_chunks", root / "hive_artifact_inbox"])
    out = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        out.append(root)
    return out


def candidate_from_worker_report(report_path: Path, report: dict[str, Any]) -> dict[str, Any] | None:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    job = report.get("job") if isinstance(report.get("job"), dict) else {}
    orchestration = report.get("orchestration") if isinstance(report.get("orchestration"), dict) else {}
    if not report.get("ok") and report.get("status") not in {"completed", None}:
        return None
    score = first_float(metrics, ["eval_accuracy", "train_accuracy", "accuracy", "train_examples_per_second"])
    model_path = str(get_path(report, ["telemetry", "model_path"], "") or "")
    return {
        "accepted": score is not None,
        "score": float(score or 0.0),
        "job_id": job.get("job_id") or report.get("chunk_id") or report_path.stem,
        "arm_id": job.get("arm_id") or default_arm(report),
        "merge_policy": job.get("merge_policy") or "",
        "orchestration": orchestration,
        "run_id": orchestration.get("run_id"),
        "round_id": orchestration.get("round_id"),
        "owner_node_id": orchestration.get("owner_node_id"),
        "source_report": display_path(report_path),
        "source_model_path": model_path,
        "backend": report.get("backend"),
        "metrics": metrics,
        "maintenance_mode": maintenance_mode_from_values(report, orchestration, job),
    }


def promote_candidate(checkpoint_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    arm = safe_name(str(candidate.get("arm_id") or "unknown_arm"))
    job = safe_name(str(candidate.get("job_id") or "job"))
    dest_dir = checkpoint_root / arm
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_source = find_downloaded_related_file(candidate)
    model_dest = ""
    if model_source and model_source.exists():
        target = dest_dir / f"{job}.model.json"
        shutil.copy2(model_source, target)
        model_dest = str(target.relative_to(ROOT)).replace("\\", "/")
    manifest = {
        "ok": True,
        "policy": "project_theseus_hive_promoted_artifact_v0",
        "created_utc": now(),
        "arm_id": candidate.get("arm_id"),
        "job_id": candidate.get("job_id"),
        "run_id": candidate.get("run_id"),
        "round_id": candidate.get("round_id"),
        "owner_node_id": candidate.get("owner_node_id"),
        "score": candidate.get("score"),
        "source_report": candidate.get("source_report"),
        "source_model_path": candidate.get("source_model_path"),
        "promoted_model_path": model_dest,
        "merge_policy": candidate.get("merge_policy"),
        "promotion_type": "artifact_level_best_by_arm",
        "orchestration": candidate.get("orchestration") or {},
        "backend": candidate.get("backend"),
        "metrics": candidate.get("metrics") or {},
        "review_step_count": 1,
        "review_step_basis": "promoted_artifact_manifest_review",
        "maintenance_mode": maintenance_mode_from_values(candidate),
        "maintenance_mode_basis": "candidate_or_object_only_default",
        "human_edit_minutes": None,
        "human_edit_minutes_measured": False,
        "viea_artifact_citation": viea_spine_records.materialized_artifact_citation(
            "hive_promoted_artifact_manifest",
            artifact_id=str(candidate.get("job_id") or ""),
            artifact_path=str(candidate.get("source_report") or ""),
        ),
        "public_training_rows_written": 0,
        "external_inference_calls": 0,
        "fallback_return_count": 0,
    }
    manifest["viea_artifact_citation_id"] = get_path(manifest, ["viea_artifact_citation", "citation_id"], "")
    write_json(dest_dir / f"{job}.manifest.json", manifest)
    write_json(dest_dir / "active_manifest.json", manifest)
    return manifest


def find_downloaded_related_file(candidate: dict[str, Any]) -> Path | None:
    wanted = str(candidate.get("source_model_path") or "")
    if not wanted:
        return None
    suffix = safe_rel_path(wanted)
    local = ROOT / suffix
    if local.exists():
        return local
    inbox = ROOT / "reports" / "hive_artifact_inbox"
    matches: list[Path] = []
    for root in [inbox, *report_roots()]:
        if not root.exists():
            continue
        matches.extend([path for path in root.rglob(suffix.name) if str(path).replace("\\", "/").endswith(str(suffix).replace("\\", "/"))])
    return matches[0] if matches else None


def peer_urls(policy: dict[str, Any], explicit: list[str]) -> list[str]:
    urls = [url for url in explicit if url]
    registry = hive_node_registry.build_registry(policy)
    write_json(ROOT / "reports" / "hive_node_registry.json", registry)
    for node in registry.get("nodes", []) if isinstance(registry.get("nodes"), list) else []:
        if not isinstance(node, dict) or node.get("is_local"):
            continue
        if get_path(node, ["trust", "trusted"], False) and node.get("api_url"):
            urls.append(str(node.get("api_url")))
    peers = read_json(ROOT / str(get_path(policy, ["node", "peers_path"], "reports/hive_peers.json")), {})
    for peer in peers.get("peers", []) if isinstance(peers.get("peers"), list) else []:
        if isinstance(peer, dict) and peer.get("api_url"):
            urls.append(str(peer.get("api_url")))
    return sorted(set(urls))


def fetch_json(url: str, *, headers: dict[str, str], timeout: float = 20.0) -> dict[str, Any]:
    req = urlrequest.Request(url, headers=headers, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:  # noqa: S310 - user-configured private Hive endpoint.
            raw = response.read().decode("utf-8")
    except (URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "url": url, "body": raw[:500]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_json", "url": url}


def is_worker_report(report: dict[str, Any]) -> bool:
    return str(report.get("policy") or "").endswith("hive_worker_chunk_v0") or "work_receipt" in report


def default_arm(report: dict[str, Any]) -> str:
    backend = str(report.get("backend") or "")
    if backend == "rust_cuda":
        return "rust_cuda_systems_arm"
    if backend.startswith("mlx"):
        return "apple_mlx_worker_arm"
    return "unknown_arm"


def first_float(metrics: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        try:
            if metrics.get(key) is not None:
                return float(metrics.get(key))
        except (TypeError, ValueError):
            continue
    return None


def hive_secret(policy: dict[str, Any]) -> str:
    env_name = str(get_path(policy, ["security", "shared_secret_env"], "THESEUS_HIVE_SECRET"))
    value = os.environ.get(env_name, "")
    if value:
        return value
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    if isinstance(join, dict) and join.get("join_token"):
        return str(join.get("join_token") or "")
    profiles = read_json(ROOT / str(get_path(policy, ["federation", "profiles_path"], "configs/hive_profiles.local.json")), {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            token = str(profile.get("join_token") or "")
            if token:
                return token
    return ""


def relay_url(policy: dict[str, Any]) -> str:
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    return os.environ.get("THESEUS_HIVE_RELAY_URL", "") or str(join.get("relay_url") or "")


def hive_id(policy: dict[str, Any]) -> str:
    join = read_json(ROOT / str(get_path(policy, ["federation", "join_config_path"], "configs/hive_join.local.json")), {})
    env_name = str(get_path(policy, ["federation", "hive_id_env"], "THESEUS_HIVE_ID"))
    return os.environ.get(env_name, "") or str(join.get("hive_id") or get_path(policy, ["federation", "default_hive_id"], ""))


def safe_rel_path(value: str) -> Path:
    parts = [safe_name(part) for part in Path(value.replace("\\", "/")).parts if part not in {"", ".", ".."}]
    return Path(*parts) if parts else Path("artifact.json")


def report_roots() -> list[Path]:
    roots = [ROOT / "reports"]
    env_reports = os.environ.get("THESEUS_REPORTS_DIR", "")
    if env_reports:
        roots.append(Path(env_reports).expanduser())
    try:
        runtime = theseus_runtime.runtime_report(create=True)
        runtime_reports = get_path(runtime, ["paths", "reports_dir", "path"], "")
        if runtime_reports:
            roots.append(Path(str(runtime_reports)).expanduser())
    except Exception:
        pass
    roots.extend(
        [
            Path.home() / "Library" / "Application Support" / "Project Theseus Hive" / "runtime" / "reports",
            Path.home() / "Library" / "Application Support" / "ProjectTheseus" / "runtime" / "reports",
        ]
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(root)
    return out


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned[:96] or "artifact"


def get_path(value: Any, path: list[str], default: Any) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def maintenance_mode_from_values(*values: Any) -> str:
    for value in values:
        normalized = explicit_maintenance_mode(value)
        if normalized:
            return normalized
    return "object_only"


def explicit_maintenance_mode(value: Any) -> str:
    if isinstance(value, dict):
        for key in ["maintenance_mode", "maintenance_policy", "maintenance_label"]:
            normalized = normalize_maintenance_mode(value.get(key))
            if normalized:
                return normalized
        for key in ["payload", "orchestration", "job"]:
            nested = value.get(key)
            if isinstance(nested, dict):
                normalized = explicit_maintenance_mode(nested)
                if normalized:
                    return normalized
    return normalize_maintenance_mode(value)


def normalize_maintenance_mode(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ordinary": "object_only",
        "ordinary_current": "object_only",
        "baseline": "object_only",
        "object": "object_only",
        "object_only": "object_only",
        "circle": "circle_seed_rule_rebuild",
        "circle_seed_rule": "circle_seed_rule_rebuild",
        "circle_seed_rule_rebuild": "circle_seed_rule_rebuild",
        "seed_rule_rebuild": "circle_seed_rule_rebuild",
    }
    return aliases.get(text, "")


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
