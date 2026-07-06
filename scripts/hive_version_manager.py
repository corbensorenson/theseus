"""Hive version, installer, and fleet-convergence manager.

This is the coordination layer above the existing update manager. The update
manager knows how to check/apply an accepted-candidate update on one node; this
script gives the Hive a shared view of the blessed version, publishes a private
catalog, records installer artifacts, and asks peers to converge on that
catalog through the registered update endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DIST = ROOT / "dist" / "hive-release"
DEFAULT_OUT = REPORTS / "hive_version_status.json"
VERIFIED_VERSION = REPORTS / "hive_verified_version.json"
CATALOG = REPORTS / "hive_update_catalog.json"
INSTALLER_ARTIFACTS = REPORTS / "hive_installer_artifacts.json"
BUILD_VERSION = ROOT / "configs" / "hive_build_version.json"

sys.path.insert(0, str(ROOT / "scripts"))
import update_manager  # noqa: E402
import viea_spine_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Theseus Hive version and convergence manager.")
    sub = parser.add_subparsers(dest="command")

    status = sub.add_parser("status", help="Report local Hive version, verification state, installers, and peers.")
    status.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))

    verify = sub.add_parser("verify", help="Run lightweight local release checks and write the verified-version manifest.")
    verify.add_argument("--out", default=str(VERIFIED_VERSION.relative_to(ROOT)))
    verify.add_argument("--skip-checks", action="store_true")

    publish = sub.add_parser("publish", help="Publish the current verified version as the private Hive update catalog.")
    publish.add_argument("--out", default=str(CATALOG.relative_to(ROOT)), help="Catalog output path. The publish report is always written to reports/hive_version_publish.json.")
    publish.add_argument("--release-root", default="")
    publish.add_argument("--skip-checks", action="store_true")

    converge = sub.add_parser("converge", help="Ask local/peer Hive nodes to use the private catalog and apply soft updates.")
    converge.add_argument("--catalog-url", default="")
    converge.add_argument("--peer-url", action="append", default=[])
    converge.add_argument("--execute", action="store_true")
    converge.add_argument("--allow-hard", action="store_true")
    converge.add_argument("--timeout-seconds", type=int, default=15)
    converge.add_argument("--out", default="reports/hive_version_convergence.json")

    artifacts = sub.add_parser("installer-artifacts", help="Refresh installer package artifact manifest.")
    artifacts.add_argument("--out", default=str(INSTALLER_ARTIFACTS.relative_to(ROOT)))

    build_manifest = sub.add_parser("build-manifest", help="Write a bundled installed-version manifest for package payloads.")
    build_manifest.add_argument("--out", default=str(BUILD_VERSION.relative_to(ROOT)))

    args = parser.parse_args()
    if args.command == "verify":
        report = verify_current_version(skip_checks=bool(args.skip_checks))
        write_json(resolve(args.out), report)
    elif args.command == "publish":
        report = publish_catalog(skip_checks=bool(args.skip_checks), release_root=str(args.release_root or ""), catalog_out=str(args.out or ""))
    elif args.command == "converge":
        report = converge_fleet(args)
        write_json(resolve(args.out), report)
    elif args.command == "installer-artifacts":
        report = installer_artifacts_report()
        write_json(resolve(args.out), report)
    elif args.command == "build-manifest":
        report = build_version_manifest()
        write_json(resolve(args.out), report)
    else:
        report = status_report(write_report=True)
        if getattr(args, "out", ""):
            write_json(resolve(args.out), report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 2


def status_report(*, write_report: bool = False) -> dict[str, Any]:
    verified = read_json(VERIFIED_VERSION, {})
    artifacts = installer_artifacts_report(write_report=False)
    update_status = update_manager.status_report(write_report=False)
    peers = read_json(REPORTS / "hive_peers.json", {})
    local = local_version_summary()
    report = {
        "ok": True,
        "policy": "project_theseus_hive_version_status_v1",
        "created_utc": now(),
        "local": local,
        "verified_version": compact_verified(verified),
        "installers": compact_installers(artifacts),
        "updates": compact_update(update_status),
        "peers": compact_peers(peers),
        "convergence": convergence_summary(local, verified, peers),
        "rules": {
            "source_of_truth": "reports/hive_verified_version.json plus reports/hive_update_catalog.json",
            "soft_updates": "private Hive nodes may auto-install verified soft update metadata",
            "hard_updates": "source/app replacement must be staged by a verified package or explicit hard-update flow",
            "package_sync": "macOS/Windows/Linux installer artifacts are shared through dist/* manifests and private Hive artifact sync",
        },
        "external_inference_calls": 0,
    }
    if write_report:
        write_json(DEFAULT_OUT, report)
    return report


def verify_current_version(*, skip_checks: bool = False) -> dict[str, Any]:
    local = local_version_summary()
    checks = [] if skip_checks else release_checks()
    if skip_checks:
        checks.append(check("checks_skipped_by_operator", True, "skip_checks"))
    source_dirty = [row for row in local.get("git", {}).get("dirty_paths", []) if is_source_path(str(row))]
    checks.append(check("no_dirty_source_paths", not source_dirty, source_dirty[:40]))
    checks.append(check("app_version_present", bool(local.get("app_version")), local.get("app_version")))
    checks.append(check("git_commit_present", bool(local.get("git", {}).get("commit")), local.get("git", {}).get("commit")))
    ok = all(row.get("passed") for row in checks)
    manifest = {
        "ok": ok,
        "policy": "project_theseus_hive_verified_version_v1",
        "created_utc": now(),
        "version_id": local["version_id"],
        "app_version": local["app_version"],
        "git": local["git"],
        "platform": local["platform"],
        "checks": checks,
        "installer_artifacts": compact_installers(installer_artifacts_report(write_report=True)),
        "update_status": compact_update(update_manager.status_report(write_report=False)),
        "promotion_state": "verified" if ok else "blocked",
        "external_inference_calls": 0,
    }
    write_json(VERIFIED_VERSION, manifest)
    return manifest


def publish_catalog(*, skip_checks: bool = False, release_root: str = "", catalog_out: str = "") -> dict[str, Any]:
    verified = verify_current_version(skip_checks=skip_checks)
    release_dir = resolve(release_root) if release_root else DIST / str(verified.get("version_id") or "unverified")
    release_dir.mkdir(parents=True, exist_ok=True)
    write_json(release_dir / "hive-version-manifest.json", verified)
    offer = catalog_offer_from_verified(verified)
    catalog = {
        "ok": bool(verified.get("ok")),
        "policy": "project_theseus_hive_update_catalog_v1",
        "created_utc": now(),
        "catalog_id": "private-hive",
        "channel": "community",
        "track": "stable",
        "latest_app_version": update_manager.app_version_info(),
        "latest_hive_version": {
            "version_id": verified.get("version_id"),
            "app_version": verified.get("app_version"),
            "git_commit": get_path(verified, ["git", "commit"], ""),
        },
        "offers": [offer] if verified.get("ok") else [],
        "installer_artifacts": verified.get("installer_artifacts", {}),
        "communication": {
            "catalog_path": "/api/hive/update-catalog",
            "version_path": "/api/hive/version",
            "installer_artifacts_path": "/api/hive/installer-artifacts",
        },
        "external_inference_calls": 0,
    }
    catalog_path = resolve(catalog_out) if catalog_out else CATALOG
    write_json(CATALOG, catalog)
    if catalog_path.resolve() != CATALOG.resolve():
        write_json(catalog_path, catalog)
    write_json(release_dir / "update-catalog.json", catalog)
    installers = installer_artifacts_report(write_report=True)
    write_json(release_dir / "installer-artifacts.json", installers)
    mirrored_reports = mirror_private_catalog_reports(
        {
            "hive_update_catalog.json": catalog,
            "hive_verified_version.json": verified,
            "hive_installer_artifacts.json": installers,
        }
    )
    report = {
        "ok": bool(verified.get("ok")),
        "policy": "project_theseus_hive_catalog_publish_v1",
        "created_utc": now(),
        "release_dir": rel(release_dir),
        "catalog": rel(catalog_path),
        "canonical_catalog": rel(CATALOG),
        "version_id": verified.get("version_id"),
        "offer_count": len(catalog["offers"]),
        "mirrored_reports": mirrored_reports,
        "next_action": "Run hive_version_manager.py converge --execute on the coordinator, or let nodes check the private catalog on startup.",
        "external_inference_calls": 0,
    }
    write_json(REPORTS / "hive_version_publish.json", report)
    return report


def converge_fleet(args: argparse.Namespace) -> dict[str, Any]:
    catalog_url = str(args.catalog_url or default_catalog_url())
    peers = peer_urls(args.peer_url)
    if not peers:
        peers = ["http://127.0.0.1:8791"]
    rows = []
    for peer in peers:
        rows.append(converge_peer(peer, catalog_url=catalog_url, execute=bool(args.execute), allow_hard=bool(args.allow_hard), timeout=int(args.timeout_seconds or 15)))
    ok = all(row.get("ok") or row.get("status") in {"dry_run", "soft_update_sent", "current"} for row in rows)
    report = {
        "ok": ok,
        "policy": "project_theseus_hive_version_convergence_v1",
        "created_utc": now(),
        "execute": bool(args.execute),
        "allow_hard": bool(args.allow_hard),
        "catalog_url": catalog_url,
        "target_count": len(rows),
        "updated_or_current": sum(1 for row in rows if row.get("status") in {"soft_update_sent", "current"}),
        "requires_manual_hard": [row for row in rows if row.get("hard_update_available") and not args.allow_hard],
        "nodes": rows,
        "external_inference_calls": 0,
    }
    append_jsonl(REPORTS / "hive_version_convergence_ledger.jsonl", report)
    return report


def converge_peer(peer_url: str, *, catalog_url: str, execute: bool, allow_hard: bool, timeout: int) -> dict[str, Any]:
    status = fetch_json(peer_url.rstrip("/") + "/api/hive/status", timeout=timeout)
    before = fetch_json(peer_url.rstrip("/") + "/api/hive/update-status", timeout=timeout, auth=True)
    row = {
        "peer_url": peer_url,
        "node_id": status.get("node_id"),
        "node_name": status.get("node_name"),
        "platform": status.get("platform", {}),
        "before": compact_peer_update(before or status.get("updates", {})),
    }
    if not execute:
        return {**row, "ok": True, "status": "dry_run", "planned": ["configure_auto_soft", "check_catalog_apply", "apply_soft_if_available"]}
    configure = post_json(
        peer_url.rstrip("/") + "/api/hive/update/configure",
        {"mode": "auto_soft", "catalog_url": catalog_url, "check_on_start": True, "auto_install_soft": True, "no_auto_install_hard": True},
        timeout=timeout,
        auth=True,
    )
    checkin = post_json(
        peer_url.rstrip("/") + "/api/hive/update/check",
        {"catalog_url": catalog_url, "apply": True, "respect_interval": False},
        timeout=timeout,
        auth=True,
    )
    apply_soft = post_json(peer_url.rstrip("/") + "/api/hive/update/apply-soft", {"execute": True}, timeout=timeout, auth=True)
    after = fetch_json(peer_url.rstrip("/") + "/api/hive/update-status", timeout=timeout, auth=True)
    hard_available = bool(get_path(after, ["hard_update_available"], False) or get_path(checkin, ["current_offer", "hard_available"], False))
    status_value = "soft_update_sent"
    if not configure.get("ok") or not checkin.get("ok") or not apply_soft.get("ok"):
        status_value = "failed"
    elif not get_path(after, ["update_available"], False):
        status_value = "current"
    if hard_available and allow_hard:
        hard = post_json(peer_url.rstrip("/") + "/api/hive/update/apply", {"mode": "hard", "execute": True, "allow_hard": True, "restart": True}, timeout=timeout, auth=True)
    else:
        hard = {}
    return {
        **row,
        "ok": status_value != "failed",
        "status": status_value,
        "configure": compact_response(configure),
        "checkin": compact_response(checkin),
        "apply_soft": compact_response(apply_soft),
        "hard_update_available": hard_available,
        "hard_apply": compact_response(hard) if hard else {},
        "after": compact_peer_update(after),
    }


def local_version_summary() -> dict[str, Any]:
    app = update_manager.app_version_info()
    git = git_status()
    build = read_json(BUILD_VERSION, {})
    source = "git"
    if not git.get("commit") and isinstance(build, dict) and get_path(build, ["git", "commit"], ""):
        git = build.get("git") if isinstance(build.get("git"), dict) else git
        app = build.get("app") if isinstance(build.get("app"), dict) else app
        source = "build_manifest"
    version_seed = "|".join([str(app.get("version") or ""), str(git.get("commit") or ""), str(git.get("branch") or "")])
    version_id = "hive-" + hashlib.sha256(version_seed.encode("utf-8")).hexdigest()[:16]
    return {
        "version_id": version_id,
        "app_version": app.get("version"),
        "app": app,
        "git": git,
        "source": source,
        "build_manifest_path": rel(BUILD_VERSION) if BUILD_VERSION.exists() else "",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }


def build_version_manifest() -> dict[str, Any]:
    local = local_version_summary()
    return {
        "ok": bool(get_path(local, ["git", "commit"], "")),
        "policy": "project_theseus_hive_build_version_v1",
        "created_utc": now(),
        "version_id": local.get("version_id"),
        "app_version": local.get("app_version"),
        "app": local.get("app") or {},
        "git": local.get("git") or {},
        "source": local.get("source"),
        "external_inference_calls": 0,
    }


def git_status() -> dict[str, Any]:
    branch = run_text(["git", "branch", "--show-current"]).strip()
    commit = run_text(["git", "rev-parse", "HEAD"]).strip()
    dirty_raw = run_text(["git", "status", "--porcelain"])
    dirty_paths = []
    for line in dirty_raw.splitlines():
        if not line.strip():
            continue
        dirty_paths.append(line[3:].strip() if len(line) > 3 else line.strip())
    return {
        "branch": branch,
        "commit": commit,
        "short_commit": commit[:12],
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def release_checks() -> list[dict[str, Any]]:
    checks = []
    checks.append(run_check("python_core_scripts_compile", [sys.executable, "-m", "py_compile", "scripts/hive_version_manager.py", "scripts/hive_node.py", "scripts/update_manager.py", "scripts/hive_invite.py"]))
    if os.name == "nt":
        ps = (
            "$tokens=$null;$errors=$null;"
            "[void][System.Management.Automation.PSParser]::Tokenize((Get-Content -LiteralPath 'scripts\\install_theseus_hive.ps1' -Raw), [ref]$errors);"
            "if($errors -and $errors.Count -gt 0){$errors|ForEach-Object{$_.Message};exit 1}else{'ok'}"
        )
        checks.append(run_check("windows_installer_parse", ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]))
    shell = shutil.which("bash") or shutil.which("sh")
    if shell:
        checks.append(run_check("macos_installer_shell_parse", [shell, "-n", "scripts/install_theseus_hive_macos.sh"]))
        checks.append(run_check("linux_installer_shell_parse", [shell, "-n", "scripts/install_theseus_hive_linux.sh"]))
        checks.append(run_check("macos_package_shell_parse", [shell, "-n", "scripts/package_theseus_macos.sh"]))
    return checks


def run_check(name: str, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        return check(name, result.returncode == 0, {"returncode": result.returncode, "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:], "runtime_ms": int((time.perf_counter() - started) * 1000)})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return check(name, False, {"error": str(exc), "runtime_ms": int((time.perf_counter() - started) * 1000)})


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "evidence": evidence}


def catalog_offer_from_verified(verified: dict[str, Any]) -> dict[str, Any]:
    version_id = str(verified.get("version_id") or "")
    update_id = "theseus-hive-" + hashlib.sha256(json.dumps({"version_id": version_id, "git": verified.get("git", {})}, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "ok": True,
        "policy": "project_theseus_update_offer_report_v0",
        "status": "offer_ready",
        "update_id": update_id,
        "channel": "community",
        "track": "stable",
        "update_kind": "soft",
        "soft_available": True,
        "hard_available": False,
        "restart_required": False,
        "auto_install_soft": True,
        "auto_install_hard": False,
        "checkpoint_id": version_id,
        "checkpoint_status": "hive_version_verified",
        "app_version": verified.get("app_version"),
        "hive_version": {
            "version_id": version_id,
            "git_commit": get_path(verified, ["git", "commit"], ""),
            "short_commit": get_path(verified, ["git", "short_commit"], ""),
        },
        "improvement_summary": {
            "headline": f"Verified Hive version {version_id} is available.",
            "what_users_should_notice": [
                "Hive nodes converge on the same verified metadata and package manifest.",
                "Hard source/app replacement remains a package or explicit hard-update step.",
            ],
        },
        "installer_artifacts": verified.get("installer_artifacts", {}),
        "published_utc": now(),
        "external_inference_calls": 0,
    }


def installer_artifacts_report(*, write_report: bool = True) -> dict[str, Any]:
    roots = [ROOT / "dist" / "macos", ROOT / "dist" / "windows", ROOT / "dist" / "linux", ROOT / "dist" / "hive-release", ROOT / "dist" / "universal-usb"]
    package_suffixes = (".dmg", ".pkg", ".zip", ".exe", ".msi", ".AppImage", ".deb", ".rpm", ".tar.gz", ".sh", ".ps1", ".cmd")
    root_level_suffixes = (".json", ".mobileconfig", ".svg")
    artifacts = []
    skipped_missing = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            text = path.name
            package_artifact = any(text.endswith(suffix) for suffix in package_suffixes)
            root_level_artifact = path.parent == root and any(text.endswith(suffix) for suffix in root_level_suffixes)
            if not package_artifact and not root_level_artifact:
                continue
            try:
                artifacts.append(artifact_row(path))
            except FileNotFoundError:
                skipped_missing.append(rel(path))
    citation = viea_spine_records.materialized_artifact_citation(
        "hive_installer_artifacts",
        artifact_path=rel(INSTALLER_ARTIFACTS),
    )
    for row in artifacts:
        row["viea_artifact_citation_id"] = citation.get("citation_id")
    report = {
        "ok": True,
        "policy": "project_theseus_hive_installer_artifacts_v1",
        "created_utc": now(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "skipped_missing": skipped_missing,
        "platforms": {
            "windows": [row for row in artifacts if "/windows/" in row["path"] or row["path"].endswith(".exe") or row["path"].endswith(".msi")],
            "macos": [row for row in artifacts if "/macos/" in row["path"] or row["path"].endswith(".dmg") or row["path"].endswith(".pkg")],
            "linux": [row for row in artifacts if "/linux/" in row["path"] or row["path"].endswith(".AppImage") or row["path"].endswith(".deb") or row["path"].endswith(".rpm") or row["path"].endswith(".tar.gz")],
        },
        "viea_artifact_citation": citation,
        "external_inference_calls": 0,
        "public_training_rows_written": 0,
        "fallback_return_count": 0,
    }
    if write_report:
        write_json(INSTALLER_ARTIFACTS, report)
    return report


def mirror_private_catalog_reports(files: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_reports = REPORTS.resolve()
    for reports_dir in runtime_reports_dirs():
        resolved = reports_dir.resolve()
        if resolved == source_reports:
            continue
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            rows.append({"path": str(reports_dir), "ok": False, "error": str(exc)})
            continue
        written = []
        for name, payload in files.items():
            try:
                write_json(resolved / name, payload)
                written.append(name)
            except OSError as exc:
                rows.append({"path": str(resolved / name), "ok": False, "error": str(exc)})
        if written:
            rows.append({"path": str(resolved), "ok": True, "files": written})
    return rows


def runtime_reports_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_reports = os.environ.get("THESEUS_REPORTS_DIR", "")
    if env_reports:
        candidates.append(Path(env_reports).expanduser())
    local_runtime = read_json(ROOT / "configs" / "runtime_paths.local.json", {})
    if isinstance(local_runtime, dict) and local_runtime.get("reports_dir"):
        candidates.append(Path(str(local_runtime["reports_dir"])).expanduser())
    if platform.system() == "Darwin":
        home = Path.home()
        candidates.append(home / "Library" / "Application Support" / "Project Theseus Hive" / "runtime" / "reports")
        candidates.append(home / "Library" / "Application Support" / "ProjectTheseus" / "runtime" / "reports")
    elif platform.system() == "Windows":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            candidates.append(Path(local_appdata) / "ProjectTheseus" / "runtime" / "reports")
        candidates.append(Path("D:/ProjectTheseus/runtime/reports"))
    else:
        candidates.append(Path.home() / ".local" / "share" / "ProjectTheseus" / "runtime" / "reports")
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve())
        if key not in seen and (candidate.exists() or candidate.parent.exists()):
            seen.add(key)
            unique.append(candidate.expanduser())
    return unique


def artifact_row(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def default_catalog_url() -> str:
    status = read_json(REPORTS / "hive_status.json", {})
    api = str(status.get("api_url") or "")
    if api:
        return api.rstrip("/") + "/api/hive/update-catalog"
    return "http://127.0.0.1:8791/api/hive/update-catalog"


def peer_urls(explicit: list[str]) -> list[str]:
    urls = [url for url in explicit if url]
    peers = read_json(REPORTS / "hive_peers.json", {})
    for peer in peers.get("peers", []) if isinstance(peers.get("peers"), list) else []:
        if isinstance(peer, dict) and peer.get("api_url"):
            urls.append(str(peer.get("api_url")))
    return sorted(set(urls))


def fetch_json(url: str, *, timeout: int, auth: bool = False) -> dict[str, Any]:
    headers = {"User-Agent": "ProjectTheseusHiveVersion/0.1"}
    if auth and hive_secret():
        headers["X-Theseus-Hive-Secret"] = hive_secret()
    try:
        with urlrequest.urlopen(urlrequest.Request(url, headers=headers), timeout=timeout) as response:  # noqa: S310 - private Hive URL.
            raw = response.read(4 * 1024 * 1024).decode("utf-8")
    except (OSError, URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "url": url, "body": raw[:400]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_json", "url": url}


def post_json(url: str, payload: dict[str, Any], *, timeout: int, auth: bool = False) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "User-Agent": "ProjectTheseusHiveVersion/0.1"}
    if auth and hive_secret():
        headers["X-Theseus-Hive-Secret"] = hive_secret()
    data = json.dumps(payload).encode("utf-8")
    try:
        with urlrequest.urlopen(urlrequest.Request(url, data=data, headers=headers, method="POST"), timeout=timeout) as response:  # noqa: S310 - private Hive URL.
            raw = response.read(4 * 1024 * 1024).decode("utf-8")
    except (OSError, URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "url": url}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": False, "error": "non_json_response", "url": url, "body": raw[:400]}
    return value if isinstance(value, dict) else {"ok": False, "error": "unexpected_json", "url": url}


def hive_secret() -> str:
    value = os.environ.get("THESEUS_HIVE_SECRET", "")
    if value:
        return value
    join = read_json(ROOT / "configs" / "hive_join.local.json", {})
    if isinstance(join, dict) and join.get("join_token"):
        return str(join.get("join_token") or "")
    profiles = read_json(ROOT / "configs" / "hive_profiles.local.json", {})
    active = str(profiles.get("active_profile_id") or "") if isinstance(profiles, dict) else ""
    for profile in profiles.get("profiles", []) if isinstance(profiles.get("profiles"), list) else []:
        if isinstance(profile, dict) and (not active or profile.get("profile_id") == active):
            return str(profile.get("join_token") or "")
    return ""


def compact_verified(verified: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": verified.get("ok"),
        "version_id": verified.get("version_id"),
        "app_version": verified.get("app_version"),
        "git_commit": get_path(verified, ["git", "short_commit"], ""),
        "promotion_state": verified.get("promotion_state"),
        "failed_checks": [row.get("check") for row in verified.get("checks", []) if isinstance(row, dict) and not row.get("passed")],
    }


def compact_installers(report: dict[str, Any]) -> dict[str, Any]:
    citation = report.get("viea_artifact_citation") if isinstance(report.get("viea_artifact_citation"), dict) else {}
    return {
        "artifact_count": report.get("artifact_count", 0),
        "windows_count": len(get_path(report, ["platforms", "windows"], [])),
        "macos_count": len(get_path(report, ["platforms", "macos"], [])),
        "linux_count": len(get_path(report, ["platforms", "linux"], [])),
        "viea_artifact_citation_ready": citation.get("ready"),
        "viea_artifact_citation_id": citation.get("citation_id"),
        "viea_claim_ledger_entry_count": citation.get("claim_ledger_entry_count"),
        "latest": (report.get("artifacts") or [])[:8],
    }


def compact_update(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "update_available": status.get("update_available"),
        "soft_update_available": status.get("soft_update_available"),
        "hard_update_available": status.get("hard_update_available"),
        "restart_required": status.get("restart_required"),
        "offer_update_id": get_path(status, ["current_offer", "update_id"], ""),
        "installed_update_id": get_path(status, ["installed", "active_update_id"], ""),
        "mode": get_path(status, ["client", "mode"], ""),
    }


def compact_peers(peers: dict[str, Any]) -> dict[str, Any]:
    rows = peers.get("peers") if isinstance(peers.get("peers"), list) else []
    return {
        "peer_count": peers.get("peer_count", len(rows)),
        "peers": [
            {"node_id": row.get("node_id"), "node_name": row.get("node_name"), "api_url": row.get("api_url"), "platform": row.get("platform", {})}
            for row in rows[:20]
            if isinstance(row, dict)
        ],
    }


def convergence_summary(local: dict[str, Any], verified: dict[str, Any], peers: dict[str, Any]) -> dict[str, Any]:
    target = str(verified.get("version_id") or local.get("version_id") or "")
    rows = peers.get("peers") if isinstance(peers.get("peers"), list) else []
    return {
        "target_version_id": target,
        "local_matches_target": local.get("version_id") == target,
        "peer_count": len(rows),
        "remote_version_visibility": "peers advertise app/update status; run converge for active check/apply",
    }


def compact_peer_update(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": status.get("ok"),
        "update_available": status.get("update_available"),
        "soft_update_available": status.get("soft_update_available"),
        "hard_update_available": status.get("hard_update_available"),
        "restart_required": status.get("restart_required"),
        "offer_update_id": get_path(status, ["current_offer", "update_id"], status.get("offer_update_id")),
        "installed_update_id": get_path(status, ["installed", "active_update_id"], status.get("installed_update_id")),
        "next_action": status.get("next_action"),
    }


def compact_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "error": payload.get("error"),
        "update_id": payload.get("update_id"),
        "catalog_ok": payload.get("catalog_ok"),
        "next_action": payload.get("next_action"),
    }


def is_source_path(path: str) -> bool:
    path = path.replace("\\", "/")
    prefixes = ("scripts/", "docs/", "src/", "crates/", "dashboard/", "configs/", "adapters/", "benchmarks/", "tests/", "examples/")
    return path in {"Cargo.toml", "Cargo.lock", "README.md", ".gitignore"} or path.startswith(prefixes)


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
