"""Governed resource pantry for Project Theseus.

The online source catalog answers "is this source allowed and why?".
The resource pantry answers "is the source locally staged in a cheap,
adapter-ready form?".

It intentionally stages source code only, never commercial ROMs and never bulk
training corpora. Training datasets remain metadata/tiny-sample governed by
online_source_catalog.py and training_data_sampler.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "resource_pantry.json"
DEFAULT_CATALOG = ROOT / "configs" / "online_source_catalog.json"
DEFAULT_OUT = ROOT / "reports" / "resource_pantry.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "resource_pantry.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN.relative_to(ROOT)))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-clones", type=int, default=None)
    parser.add_argument("--source-id", action="append", default=[], help="Only stage these source ids, preserving the provided order.")
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy, {})
    catalog = read_json(ROOT / args.catalog, {})
    report = build_report(policy, catalog, execute=args.execute, max_clones=args.max_clones, source_ids=args.source_id)
    write_json(ROOT / args.out, report)
    if get_path(policy, ["readiness_checks", "write_markdown_report"], True):
        write_text(ROOT / args.markdown_out, markdown_report(report))
    print(json.dumps(report, indent=2))
    return 0 if not report.get("hard_errors") else 1


def build_report(
    policy: dict[str, Any],
    catalog: dict[str, Any],
    *,
    execute: bool,
    max_clones: int | None,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    storage = select_storage(policy)
    sources = catalog_sources(policy, catalog, source_ids=source_ids)
    allowed_licenses = {
        normalize_license(item)
        for item in get_path(policy, ["clone_policy", "allowed_code_licenses"], [])
    }
    allowed_data_licenses = {
        normalize_license(item)
        for item in get_path(catalog, ["license_policy", "allowed_data_licenses"], [])
    }
    blocked_categories = set(get_path(policy, ["clone_policy", "blocked_categories"], []))
    metadata_only_categories = set(get_path(policy, ["clone_policy", "metadata_only_categories"], []))
    max_per_run = int(max_clones if max_clones is not None else get_path(policy, ["clone_policy", "max_clones_per_run"], 12))
    clone_budget = max(0, max_per_run)
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    hard_errors: list[dict[str, Any]] = []

    for source in sources:
        row = source_status(source, policy, storage, allowed_licenses, allowed_data_licenses, blocked_categories, metadata_only_categories)
        if execute and row["decision"] == "clone_allowed" and not row["present"] and clone_budget > 0:
            action = clone_source(source, row, policy, storage)
            actions.append(action)
            clone_budget -= 1
            row = source_status(source, policy, storage, allowed_licenses, allowed_data_licenses, blocked_categories, metadata_only_categories)
            row["last_action"] = action
            if not action.get("ok"):
                hard_errors.append({"id": row["id"], "error": action.get("error", "clone_failed")})
        elif execute and row["decision"] == "clone_allowed" and not row["present"]:
            row["last_action"] = {"ok": False, "status": "skipped_clone_budget_exhausted"}
        elif execute and row["decision"] in {"metadata_only", "blocked"}:
            action = write_metadata(source, row, storage)
            actions.append(action)
            row = source_status(source, policy, storage, allowed_licenses, allowed_data_licenses, blocked_categories, metadata_only_categories)
            row["last_action"] = action
        rows.append(row)

    summary = summarize(rows, storage, actions)
    return {
        "policy": "project_theseus_resource_pantry_report_v0",
        "created_utc": now(),
        "execute": execute,
        "source_ids": source_ids or [],
        "storage": storage,
        "summary": summary,
        "sources": rows,
        "actions": actions,
        "hard_errors": hard_errors,
        "next_actions": next_actions(rows, summary),
        "external_inference_calls": 0,
        "safety": policy.get("safety", {}),
    }


def catalog_sources(policy: dict[str, Any], catalog: dict[str, Any], source_ids: list[str] | None = None) -> list[dict[str, Any]]:
    sources = [item for item in catalog.get("sources", []) if isinstance(item, dict)]
    requested_ids = [str(item).strip() for item in source_ids or [] if str(item).strip()]
    if requested_ids:
        by_id = {str(item.get("id")): item for item in sources}
        return [by_id[source_id] for source_id in requested_ids if source_id in by_id]
    priority_ids = [str(item) for item in policy.get("priority_source_ids", [])]
    by_id = {str(item.get("id")): item for item in sources}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_id in priority_ids:
        item = by_id.get(source_id)
        if item:
            ordered.append(item)
            seen.add(source_id)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    rest = [item for item in sources if str(item.get("id")) not in seen]
    rest.sort(key=lambda item: (priority_rank.get(str(item.get("priority") or "medium"), 1), str(item.get("id") or "")))
    return ordered + rest


def source_status(
    source: dict[str, Any],
    policy: dict[str, Any],
    storage: dict[str, Any],
    allowed_licenses: set[str],
    allowed_data_licenses: set[str],
    blocked_categories: set[str],
    metadata_only_categories: set[str],
) -> dict[str, Any]:
    source_id = safe_name(str(source.get("id") or source.get("name") or "source"))
    category = str(source.get("category") or "")
    source_kind = str(source.get("source_kind") or "")
    license_spdx = normalize_license(source.get("license_spdx"))
    clone_path = Path(storage["clone_root"]) / source_id
    metadata_path = Path(storage["metadata_root"]) / f"{source_id}.json"
    present = clone_path.exists() and (clone_path / ".git").exists()
    metadata_present = metadata_path.exists()
    risk = str(source.get("risk") or "")
    decision = "blocked"
    reason = "default_block"
    exclude_uncleared = bool(get_path(policy, ["clone_policy", "exclude_non_permissive_or_uncleared_sources"], False))
    license_pool = allowed_data_licenses if source_kind == "huggingface_dataset" or category.endswith("_data") else allowed_licenses
    license_allowed = license_spdx in license_pool
    if exclude_uncleared and (str(source.get("import_policy") or "") == "queue_only" or not license_allowed):
        decision = "excluded"
        if str(source.get("import_policy") or "") == "queue_only":
            reason = "catalog_policy_queue_only_excluded_from_active_runway"
        else:
            reason = f"license_not_in_allowlist_excluded:{license_spdx or 'unknown'}"
    elif category in metadata_only_categories or category in blocked_categories:
        decision = "metadata_only"
        reason = "category_is_metadata_or_training_data"
    elif source_kind != "github_repo":
        decision = "metadata_only"
        reason = f"no_clone_handler_for_source_kind:{source_kind}"
    elif str(source.get("import_policy") or "") in {"metadata_only", "queue_only"}:
        decision = "metadata_only"
        reason = f"catalog_policy_{source.get('import_policy')}"
    elif looks_rom_related(source):
        decision = "blocked"
        reason = "rom_or_commercial_game_asset_signal"
    elif not license_allowed:
        decision = "blocked"
        reason = f"license_not_in_allowlist:{license_spdx or 'unknown'}"
    else:
        decision = "clone_allowed"
        reason = "open_source_code_license_and_catalog_policy_allow"
    health = clone_health(clone_path) if present else {}
    adapter_ready = bool(
        present
        and source.get("adapter_plan")
        and (not get_path(policy, ["readiness_checks", "require_readme_or_docs_hint"], True) or health.get("readme_hint"))
    )
    return {
        "id": source.get("id"),
        "name": source.get("name"),
        "category": category,
        "priority": source.get("priority"),
        "source_kind": source_kind,
        "url": source.get("url"),
        "license_spdx": license_spdx or "unknown",
        "import_policy": source.get("import_policy"),
        "decision": decision,
        "decision_reason": reason,
        "risk": risk,
        "clone_path": rel_or_abs(clone_path),
        "present": present,
        "metadata_path": rel_or_abs(metadata_path),
        "metadata_present": metadata_present,
        "adapter_plan": source.get("adapter_plan"),
        "smoke_plan": source.get("smoke_plan", []),
        "adapter_ready": adapter_ready,
        "health": health,
    }


def clone_source(source: dict[str, Any], row: dict[str, Any], policy: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    clone_path = Path(storage["clone_root"]) / safe_name(str(source.get("id") or source.get("name") or "source"))
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    if clone_path.exists():
        return {"id": row["id"], "ok": True, "status": "already_present", "path": rel_or_abs(clone_path)}
    url = str(source.get("url") or "")
    clone_url = github_clone_url(url)
    if not clone_url:
        return {"id": row["id"], "ok": False, "status": "blocked", "error": "not_a_github_url", "url": url}
    command = ["git", "clone", "--depth", str(int(get_path(policy, ["clone_policy", "shallow_depth"], 1)))]
    if get_path(policy, ["clone_policy", "filter_blob_none"], True):
        command.extend(["--filter=blob:none"])
    command.extend([clone_url, str(clone_path)])
    started = datetime.now(timezone.utc)
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=900)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the autonomy loop.
        return {"id": row["id"], "ok": False, "status": "failed", "error": str(exc), "command": command}
    ok = result.returncode == 0
    if not ok and clone_path.exists() and not any(clone_path.iterdir()):
        shutil.rmtree(clone_path, ignore_errors=True)
    metadata_action = write_metadata(source, row, storage)
    return {
        "id": row["id"],
        "ok": ok,
        "status": "cloned" if ok else "failed",
        "path": rel_or_abs(clone_path),
        "command": command,
        "runtime_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
        "stdout_tail": result.stdout[-1200:],
        "stderr_tail": result.stderr[-1200:],
        "metadata": metadata_action,
    }


def write_metadata(source: dict[str, Any], row: dict[str, Any], storage: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(storage["metadata_root"]) / f"{safe_name(str(source.get('id') or 'source'))}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "project_theseus_resource_pantry_source_metadata_v0",
        "created_utc": now(),
        "id": source.get("id"),
        "name": source.get("name"),
        "category": source.get("category"),
        "source_kind": source.get("source_kind"),
        "url": source.get("url"),
        "license_spdx": normalize_license(source.get("license_spdx")) or "unknown",
        "import_policy": source.get("import_policy"),
        "decision": row.get("decision"),
        "decision_reason": row.get("decision_reason"),
        "why": source.get("why"),
        "adapter_plan": source.get("adapter_plan"),
        "smoke_plan": source.get("smoke_plan", []),
        "training_use_allowed": False,
        "benchmark_execution_allowed": bool(row.get("adapter_ready")),
        "external_inference_calls": 0,
    }
    write_json(metadata_path, payload)
    return {"id": source.get("id"), "ok": True, "status": "metadata_written", "path": rel_or_abs(metadata_path)}


def clone_health(path: Path) -> dict[str, Any]:
    files = list(path.iterdir()) if path.exists() else []
    readme = any(item.name.lower().startswith("readme") for item in files)
    license_hint = any(item.name.lower().startswith(("license", "copying")) for item in files)
    pyproject = (path / "pyproject.toml").exists()
    setup_py = (path / "setup.py").exists()
    package_json = (path / "package.json").exists()
    cargo = (path / "Cargo.toml").exists()
    return {
        "readme_hint": readme,
        "license_file_hint": license_hint,
        "python_project_hint": pyproject or setup_py,
        "node_project_hint": package_json,
        "rust_project_hint": cargo,
        "top_level_entries": len(files),
    }


def select_storage(policy: dict[str, Any]) -> dict[str, Any]:
    preferred = expand_path(str(get_path(policy, ["storage", "preferred_clone_root"], "")))
    fallback = expand_path(str(get_path(policy, ["storage", "fallback_clone_root"], "data/external_benchmark_candidates/git_clones")))
    metadata = expand_path(str(get_path(policy, ["storage", "metadata_root"], "data/external_benchmark_candidates/resource_pantry_metadata")))
    min_preferred = float(get_path(policy, ["storage", "min_free_gib_preferred"], 100))
    min_fallback = float(get_path(policy, ["storage", "min_free_gib_fallback"], 25))
    preferred_status = disk_status(preferred)
    fallback_status = disk_status(fallback)
    if preferred_status.get("available") and preferred_status.get("free_gib", 0) >= min_preferred:
        root = preferred
        selected = "preferred"
    elif fallback_status.get("available") and fallback_status.get("free_gib", 0) >= min_fallback:
        root = fallback
        selected = "fallback"
    else:
        root = fallback
        selected = "fallback_low_space"
    return {
        "selected": selected,
        "clone_root": str(root),
        "metadata_root": str(metadata),
        "preferred": preferred_status,
        "fallback": fallback_status,
    }


def summarize(rows: list[dict[str, Any]], storage: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sources": len(rows),
        "clone_allowed": len([row for row in rows if row.get("decision") == "clone_allowed"]),
        "present_clones": len([row for row in rows if row.get("present")]),
        "adapter_ready": len([row for row in rows if row.get("adapter_ready")]),
        "metadata_only": len([row for row in rows if row.get("decision") == "metadata_only"]),
        "blocked": len([row for row in rows if row.get("decision") == "blocked"]),
        "excluded": len([row for row in rows if row.get("decision") == "excluded"]),
        "actions": len(actions),
        "successful_actions": len([action for action in actions if action.get("ok")]),
        "failed_actions": len([action for action in actions if not action.get("ok")]),
        "storage_selected": storage.get("selected"),
        "clone_root": storage.get("clone_root"),
        "preferred_free_gib": get_path(storage, ["preferred", "free_gib"], None),
        "fallback_free_gib": get_path(storage, ["fallback", "free_gib"], None),
    }


def next_actions(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    actions = []
    missing = [row for row in rows if row.get("decision") == "clone_allowed" and not row.get("present")]
    if missing:
        actions.append(f"Clone {min(8, len(missing))} remaining approved source repos into {summary.get('clone_root')}.")
    smoke = [row for row in rows if row.get("present") and not row.get("adapter_ready")]
    if smoke:
        actions.append(f"Add adapter smoke tests for {', '.join(str(row.get('id')) for row in smoke[:5])}.")
    blocked = [row for row in rows if row.get("decision") == "blocked"]
    if blocked:
        actions.append(f"Keep {len(blocked)} blocked sources out of active use until license or runtime gates clear.")
    if not actions:
        actions.append("Resource pantry is fully staged for the current policy; continue adapter smoke and frontier promotion.")
    return actions


def disk_status(path: Path) -> dict[str, Any]:
    try:
        anchor = path.anchor or str(ROOT.anchor)
        usage = shutil.disk_usage(anchor)
        return {
            "available": True,
            "root": anchor,
            "path": str(path),
            "total_gib": round(usage.total / 1024**3, 2),
            "free_gib": round(usage.free / 1024**3, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": str(path), "error": str(exc)}


def github_clone_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "github.com":
        return ""
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return ""
    return f"https://github.com/{parts[0]}/{parts[1]}.git"


def looks_rom_related(source: dict[str, Any]) -> bool:
    if str(source.get("category") or "") in {"minecraft_rl_environment", "emulator_runtime_dependency"}:
        return False
    text = " ".join(
        str(source.get(key) or "")
        for key in ["id", "name", "category", "why", "adapter_plan", "risk"]
    ).lower()
    if any(token in text for token in ["commercial rom", "rom download", "rom asset"]):
        return True
    return "emulator" in text and "user-owned" not in text and "user_supplied" not in text


def expand_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    return ROOT / path


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)[:120] or "source"


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower()


def rel_or_abs(path: Path) -> str:
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
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Resource Pantry",
        "",
        f"Created: `{report.get('created_utc')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "sources",
        "clone_allowed",
        "present_clones",
        "adapter_ready",
        "metadata_only",
        "blocked",
        "excluded",
        "successful_actions",
        "failed_actions",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.extend(["", f"Clone root: `{summary.get('clone_root')}`", "", "## Next Actions", ""])
    for action in report.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Sources", "", "| ID | Category | Decision | Present | Adapter ready |", "| --- | --- | --- | ---: | ---: |"])
    for row in report.get("sources") or []:
        lines.append(
            f"| {row.get('id')} | {row.get('category')} | {row.get('decision')} | {row.get('present')} | {row.get('adapter_ready')} |"
        )
    lines.append("")
    return "\n".join(lines)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
