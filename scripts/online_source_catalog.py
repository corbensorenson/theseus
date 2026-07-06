"""Governed online benchmark and training-data source catalog.

This script turns configs/online_source_catalog.json into an executable
ingestion plan. It is intentionally conservative:

- source archives and metadata land under ignored data/external_benchmark_candidates;
- unknown or queued licenses are never imported automatically;
- large training corpora are metadata-only until a sampling plan is approved;
- ROM-like/game assets stay quarantined unless explicit rights are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "online_source_catalog.json"
DEFAULT_OUT = ROOT / "reports" / "online_source_catalog_report.json"
MAX_SOURCE_ARCHIVE_BYTES = 150 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--import-sources", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--source-id", action="append", dest="id")
    parser.add_argument("--max-imports", type=int, default=8)
    args = parser.parse_args()

    catalog_path = ROOT / args.catalog
    catalog = read_json(catalog_path)
    selected = select_sources(catalog.get("sources") or [], args.category, args.id)
    storage_root = ROOT / str(catalog.get("storage_root") or "data/external_benchmark_candidates")
    report = {
        "policy": "sparkstream_online_source_catalog_report_v0",
        "created_utc": now(),
        "catalog": rel(catalog_path),
        "storage_root": rel(storage_root),
        "allow_network_fetch": args.allow_network_fetch,
        "import_sources": args.import_sources,
        "max_imports": args.max_imports,
        "summary": {},
        "sources": [],
        "imports": [],
        "training_data_candidates": [],
        "benchmark_candidates": [],
        "blocked": [],
        "excluded": [],
        "errors": [],
    }

    allowed_code = set(str(x).lower() for x in get_path(catalog, ["license_policy", "allowed_code_licenses"], []))
    allowed_data = set(str(x).lower() for x in get_path(catalog, ["license_policy", "allowed_data_licenses"], []))
    exclude_uncleared = bool(get_path(catalog, ["license_policy", "exclude_non_permissive_or_uncleared_sources"], False))
    imported_count = 0
    for source in selected:
        row = source_row(source, allowed_code, allowed_data, storage_root, exclude_uncleared)
        report["sources"].append(row)
        category = str(source.get("category") or "")
        if category == "training_data" or category.endswith("_training_data") or category == "voice_training_data":
            report["training_data_candidates"].append(compact_candidate(row))
        else:
            report["benchmark_candidates"].append(compact_candidate(row))
        if str(row["decision"]).startswith("excluded_"):
            report["excluded"].append(
                {
                    "id": row["id"],
                    "decision": row["decision"],
                    "reason": row["decision_reason"],
                    "url": row["url"],
                }
            )
            continue
        if row["decision"] != "approved_for_catalog_import":
            report["blocked"].append(
                {
                    "id": row["id"],
                    "decision": row["decision"],
                    "reason": row["decision_reason"],
                    "url": row["url"],
                }
            )
            continue
        if not args.import_sources:
            continue
        if imported_count >= max(0, args.max_imports):
            continue
        if not args.allow_network_fetch:
            report["imports"].append(
                {
                    "id": row["id"],
                    "status": "blocked_policy_requires_allow_network_fetch",
                    "url": row["url"],
                    "created_utc": now(),
                }
            )
            continue
        result = import_source(source, row, storage_root, args.refresh_metadata)
        report["imports"].append(result)
        if result.get("status") in {"imported", "already_imported", "metadata_written", "metadata_already_present"}:
            imported_count += 1

    report["summary"] = summarize(report)
    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if not report["errors"] else 1


def select_sources(sources: list[Any], categories: list[str], ids: list[str]) -> list[dict[str, Any]]:
    category_set = {item.strip() for item in categories if item.strip()}
    id_set = {item.strip() for item in ids if item.strip()}
    rows = [item for item in sources if isinstance(item, dict)]
    if category_set:
        rows = [item for item in rows if str(item.get("category") or "") in category_set]
    if id_set:
        rows = [item for item in rows if str(item.get("id") or "") in id_set]
    priority = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (priority.get(str(item.get("priority") or "medium"), 1), str(item.get("id") or "")))
    return rows


def source_row(
    source: dict[str, Any],
    allowed_code: set[str],
    allowed_data: set[str],
    storage_root: Path,
    exclude_uncleared: bool,
) -> dict[str, Any]:
    license_spdx = normalize_license(source.get("license_spdx"))
    import_policy = str(source.get("import_policy") or "queue_only")
    category = str(source.get("category") or "")
    source_kind = str(source.get("source_kind") or "")
    license_pool = allowed_data if source_kind == "huggingface_dataset" or category.endswith("_data") else allowed_code
    license_allowed = license_spdx in license_pool
    decision = "approved_for_catalog_import"
    reason = "license_and_policy_allow_metadata_or_source_archive"
    if exclude_uncleared and (import_policy == "queue_only" or not license_allowed):
        decision = "excluded_non_permissive_or_uncleared"
        if import_policy == "queue_only":
            reason = "source_policy_queue_only_excluded_from_active_runway"
        else:
            reason = f"license_not_in_allowlist_excluded:{license_spdx or 'unknown'}"
    elif import_policy == "queue_only":
        decision = "queued_only"
        reason = "source_policy_queue_only"
    elif not license_allowed:
        decision = "blocked_pending_license_audit"
        reason = f"license_not_in_allowlist:{license_spdx or 'unknown'}"
    elif looks_rom_related(source):
        decision = "blocked_pending_explicit_game_asset_rights"
        reason = "rom_or_game_asset_signal_requires_explicit_rights"
    elif source_kind not in {"github_repo", "huggingface_dataset", "pypi_package", "direct_download"}:
        decision = "blocked_unknown_source_kind"
        reason = f"unknown_source_kind:{source_kind}"
    elif import_policy not in {"source_archive", "metadata_only"}:
        decision = "blocked_unknown_import_policy"
        reason = f"unknown_import_policy:{import_policy}"

    local_dir = storage_root / str(source.get("local_subdir") or "misc")
    staged = staged_artifact(source, local_dir)
    return {
        "id": source.get("id"),
        "name": source.get("name"),
        "category": category,
        "priority": source.get("priority"),
        "source_kind": source_kind,
        "url": source.get("url"),
        "license_spdx": license_spdx or "unknown",
        "license_allowed": license_allowed,
        "import_policy": import_policy,
        "local_dir": rel(local_dir),
        "staged": bool(staged.get("staged")),
        "staged_path": staged.get("path", ""),
        "staged_metadata_path": staged.get("metadata_path", ""),
        "decision": decision,
        "decision_reason": reason,
        "why": source.get("why"),
        "adapter_plan": source.get("adapter_plan"),
        "smoke_plan": source.get("smoke_plan", []),
        "risk": source.get("risk"),
    }


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "category": row.get("category"),
        "priority": row.get("priority"),
        "decision": row.get("decision"),
        "import_policy": row.get("import_policy"),
        "license_spdx": row.get("license_spdx"),
        "risk": row.get("risk"),
        "url": row.get("url"),
        "staged": row.get("staged", False),
    }


def import_source(
    source: dict[str, Any],
    row: dict[str, Any],
    storage_root: Path,
    refresh_metadata: bool,
) -> dict[str, Any]:
    policy = row["import_policy"]
    if source.get("source_kind") == "github_repo" and policy == "source_archive":
        return import_github_archive(source, row, storage_root)
    if source.get("source_kind") == "github_repo" and policy == "metadata_only":
        return import_github_metadata(source, row, storage_root, refresh_metadata)
    if source.get("source_kind") == "huggingface_dataset" and policy == "metadata_only":
        return import_huggingface_metadata(source, row, storage_root, refresh_metadata)
    if source.get("source_kind") == "pypi_package" and policy == "metadata_only":
        return import_pypi_metadata(source, row, storage_root, refresh_metadata)
    if source.get("source_kind") == "direct_download" and policy == "metadata_only":
        return import_direct_download_metadata(source, row, storage_root, refresh_metadata)
    return {
        "id": row["id"],
        "status": "blocked_no_import_handler",
        "source_kind": row["source_kind"],
        "import_policy": policy,
        "created_utc": now(),
    }


def import_github_archive(source: dict[str, Any], row: dict[str, Any], storage_root: Path) -> dict[str, Any]:
    repo = repo_name_from_source(source)
    if not repo:
        return {"id": row["id"], "status": "failed", "error": "github_repo_name_missing", "created_utc": now()}
    target_dir = storage_root / str(source.get("local_subdir") or "source_archives")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_filename(str(source.get('id') or repo))}.zip"
    metadata_path = target.with_suffix(".metadata.json")
    if target.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        return {
            "id": row["id"],
            "status": "already_imported",
            "path": rel(target),
            "metadata_path": rel(metadata_path),
            "bytes": target.stat().st_size,
            "sha256": metadata.get("sha256") or sha256_file(target),
            "created_utc": now(),
        }
    url = f"https://api.github.com/repos/{urllib.parse.quote(repo, safe='/')}/zipball"
    try:
        size, digest = download_capped(url, target, MAX_SOURCE_ARCHIVE_BYTES)
        metadata = provenance_metadata(source, row, "github_zipball")
        metadata.update(
            {
                "download_url": url,
                "path": rel(target),
                "bytes": size,
                "sha256": digest,
                "status": "source_archive_staged_pending_adapter_smoke",
            }
        )
        write_json(metadata_path, metadata)
        return {
            "id": row["id"],
            "status": "imported",
            "path": rel(target),
            "metadata_path": rel(metadata_path),
            "bytes": size,
            "sha256": digest,
            "created_utc": now(),
        }
    except Exception as exc:  # noqa: BLE001 - record failure for the daemon.
        target.unlink(missing_ok=True)
        return {
            "id": row["id"],
            "status": "failed",
            "url": url,
            "error": str(exc),
            "created_utc": now(),
        }


def import_github_metadata(
    source: dict[str, Any],
    row: dict[str, Any],
    storage_root: Path,
    refresh_metadata: bool,
) -> dict[str, Any]:
    repo = repo_name_from_source(source)
    if not repo:
        return {"id": row["id"], "status": "failed", "error": "github_repo_name_missing", "created_utc": now()}
    target_dir = storage_root / str(source.get("local_subdir") or "repo_metadata") / safe_filename(row["id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.json"
    if metadata_path.exists() and not refresh_metadata:
        return {
            "id": row["id"],
            "status": "metadata_already_present",
            "metadata_path": rel(metadata_path),
            "created_utc": now(),
        }
    api_url = f"https://api.github.com/repos/{urllib.parse.quote(repo, safe='/')}"
    try:
        api_payload = fetch_json(api_url, MAX_METADATA_BYTES)
    except Exception as exc:  # noqa: BLE001 - record failure for daemon.
        return {"id": row["id"], "status": "failed", "url": api_url, "error": str(exc), "created_utc": now()}
    metadata = provenance_metadata(source, row, "github_repo_metadata")
    license_info = api_payload.get("license") if isinstance(api_payload.get("license"), dict) else {}
    metadata.update(
        {
            "api_url": api_url,
            "repo": repo,
            "status": "metadata_staged_source_use_blocked_until_adapter_smoke",
            "github": {
                "full_name": api_payload.get("full_name") or repo,
                "default_branch": api_payload.get("default_branch"),
                "pushed_at": api_payload.get("pushed_at"),
                "updated_at": api_payload.get("updated_at"),
                "stargazers_count": api_payload.get("stargazers_count"),
                "forks_count": api_payload.get("forks_count"),
                "size_kb": api_payload.get("size"),
                "archived": api_payload.get("archived"),
                "license_spdx": license_info.get("spdx_id"),
                "topics": api_payload.get("topics", [])[:50] if isinstance(api_payload.get("topics"), list) else [],
            },
        }
    )
    write_json(metadata_path, metadata)
    return {
        "id": row["id"],
        "status": "metadata_written",
        "metadata_path": rel(metadata_path),
        "created_utc": now(),
    }


def import_huggingface_metadata(
    source: dict[str, Any],
    row: dict[str, Any],
    storage_root: Path,
    refresh_metadata: bool,
) -> dict[str, Any]:
    dataset = str(source.get("name") or "").strip()
    if "/" not in dataset:
        return {"id": row["id"], "status": "failed", "error": "huggingface_dataset_id_missing", "created_utc": now()}
    target_dir = storage_root / str(source.get("local_subdir") or "training_data_metadata") / safe_filename(row["id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.json"
    readme_path = target_dir / "README.md"
    if metadata_path.exists() and not refresh_metadata:
        return {
            "id": row["id"],
            "status": "metadata_already_present",
            "metadata_path": rel(metadata_path),
            "readme_path": rel(readme_path) if readme_path.exists() else "",
            "created_utc": now(),
        }
    api_url = f"https://huggingface.co/api/datasets/{urllib.parse.quote(dataset, safe='/')}"
    readme_url = f"https://huggingface.co/datasets/{dataset}/raw/main/README.md"
    api_payload: dict[str, Any] = {}
    api_error = ""
    readme = ""
    readme_error = ""
    try:
        api_payload = fetch_json(api_url, MAX_METADATA_BYTES)
    except Exception as exc:  # noqa: BLE001 - keep catalog report useful.
        api_error = str(exc)
    try:
        readme = fetch_text(readme_url, MAX_METADATA_BYTES)
    except Exception as exc:  # noqa: BLE001 - README is helpful but not required.
        readme_error = str(exc)
    if api_error and not readme:
        return {"id": row["id"], "status": "failed", "error": api_error, "created_utc": now()}
    metadata = provenance_metadata(source, row, "huggingface_dataset_metadata")
    metadata.update(
        {
            "api_url": api_url,
            "readme_url": readme_url,
            "dataset_id": dataset,
            "status": "metadata_staged_training_use_blocked_until_sampling_plan",
            "huggingface": {
                "id": api_payload.get("id") or dataset,
                "downloads": api_payload.get("downloads"),
                "likes": api_payload.get("likes"),
                "tags": api_payload.get("tags", [])[:50] if isinstance(api_payload.get("tags"), list) else [],
                "last_modified": api_payload.get("lastModified"),
                "card_data": api_payload.get("cardData") if isinstance(api_payload.get("cardData"), dict) else {},
            },
        }
    )
    if api_error:
        metadata["api_error"] = api_error
    if readme_error:
        metadata["readme_error"] = readme_error
    if readme:
        readme_path.write_text(readme, encoding="utf-8")
        metadata["readme_path"] = rel(readme_path)
        metadata["readme_sha256"] = sha256_file(readme_path)
    write_json(metadata_path, metadata)
    return {
        "id": row["id"],
        "status": "metadata_written",
        "metadata_path": rel(metadata_path),
        "readme_path": rel(readme_path) if readme_path.exists() else "",
        "created_utc": now(),
    }


def import_pypi_metadata(
    source: dict[str, Any],
    row: dict[str, Any],
    storage_root: Path,
    refresh_metadata: bool,
) -> dict[str, Any]:
    package = str(source.get("name") or source.get("id") or "").strip()
    if not package:
        return {"id": row["id"], "status": "failed", "error": "pypi_package_name_missing", "created_utc": now()}
    target_dir = storage_root / str(source.get("local_subdir") or "package_metadata") / safe_filename(row["id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.json"
    if metadata_path.exists() and not refresh_metadata:
        return {
            "id": row["id"],
            "status": "metadata_already_present",
            "metadata_path": rel(metadata_path),
            "created_utc": now(),
        }
    api_url = f"https://pypi.org/pypi/{urllib.parse.quote(package)}/json"
    try:
        api_payload = fetch_json(api_url, MAX_METADATA_BYTES)
    except Exception as exc:  # noqa: BLE001 - record failure for daemon.
        return {"id": row["id"], "status": "failed", "url": api_url, "error": str(exc), "created_utc": now()}
    info = api_payload.get("info") if isinstance(api_payload.get("info"), dict) else {}
    metadata = provenance_metadata(source, row, "pypi_package_metadata")
    metadata.update(
        {
            "api_url": api_url,
            "package": package,
            "status": "metadata_staged_training_use_blocked_until_adapter_smoke",
            "pypi": {
                "name": info.get("name") or package,
                "version": info.get("version"),
                "license": info.get("license"),
                "summary": info.get("summary"),
                "project_urls": info.get("project_urls") if isinstance(info.get("project_urls"), dict) else {},
                "requires_python": info.get("requires_python"),
                "classifiers": info.get("classifiers", [])[:80] if isinstance(info.get("classifiers"), list) else [],
            },
        }
    )
    write_json(metadata_path, metadata)
    return {
        "id": row["id"],
        "status": "metadata_written",
        "metadata_path": rel(metadata_path),
        "created_utc": now(),
    }


def import_direct_download_metadata(
    source: dict[str, Any],
    row: dict[str, Any],
    storage_root: Path,
    refresh_metadata: bool,
) -> dict[str, Any]:
    url = str(source.get("url") or "").strip()
    if not url:
        return {"id": row["id"], "status": "failed", "error": "direct_download_url_missing", "created_utc": now()}
    target_dir = storage_root / str(source.get("local_subdir") or "download_metadata") / safe_filename(row["id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "metadata.json"
    page_path = target_dir / "source_page.html"
    if metadata_path.exists() and not refresh_metadata:
        return {
            "id": row["id"],
            "status": "metadata_already_present",
            "metadata_path": rel(metadata_path),
            "page_path": rel(page_path) if page_path.exists() else "",
            "created_utc": now(),
        }
    page = ""
    page_error = ""
    try:
        page = fetch_text(url, MAX_METADATA_BYTES)
    except Exception as exc:  # noqa: BLE001 - some direct sources block bots; keep provenance row.
        page_error = str(exc)
    metadata = provenance_metadata(source, row, "direct_download_metadata")
    metadata.update(
        {
            "source_url": url,
            "status": "metadata_staged_download_blocked_until_shard_plan",
            "download_policy": "metadata_only_no_bulk_archive_download",
        }
    )
    if page:
        page_path.write_text(page, encoding="utf-8")
        metadata["page_path"] = rel(page_path)
        metadata["page_sha256"] = sha256_file(page_path)
    if page_error:
        metadata["page_error"] = page_error
    write_json(metadata_path, metadata)
    return {
        "id": row["id"],
        "status": "metadata_written" if page or not page_error else "metadata_written_with_fetch_error",
        "metadata_path": rel(metadata_path),
        "page_path": rel(page_path) if page_path.exists() else "",
        "created_utc": now(),
    }


def provenance_metadata(source: dict[str, Any], row: dict[str, Any], provenance_kind: str) -> dict[str, Any]:
    return {
        "policy": "sparkstream_external_source_provenance_v0",
        "created_utc": now(),
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "source_kind": row["source_kind"],
        "provenance_kind": provenance_kind,
        "url": row["url"],
        "license_spdx": row["license_spdx"],
        "import_policy": row["import_policy"],
        "training_use_allowed": False,
        "benchmark_use_allowed": row["category"] != "training_data",
        "adapter_plan": row.get("adapter_plan"),
        "smoke_plan": row.get("smoke_plan", []),
        "sample_plan": source.get("sample_plan", ""),
        "risk": row.get("risk"),
        "required_gates_before_training_use": [
            "license_verified",
            "provenance_recorded",
            "dedupe_and_leakage_check",
            "quality_filter_report",
            "small_sample_audit",
            "human_or_teacher_review_for_sampling_plan",
        ],
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    sources = report.get("sources") or []
    imports = report.get("imports") or []
    approved = [item for item in sources if item.get("decision") == "approved_for_catalog_import"]
    training = [item for item in sources if item.get("category") == "training_data"]
    benchmarks = [item for item in sources if item.get("category") != "training_data"]
    imported = [
        item
        for item in imports
        if item.get("status") in {"imported", "already_imported", "metadata_written", "metadata_already_present"}
    ]
    return {
        "sources": len(sources),
        "approved_for_catalog_import": len(approved),
        "blocked_or_queued": len(report.get("blocked") or []),
        "excluded_non_permissive_or_uncleared": len(report.get("excluded") or []),
        "training_data_candidates": len(training),
        "benchmark_candidates": len(benchmarks),
        "import_attempts": len(imports),
        "staged_or_present": len([item for item in sources if item.get("staged")]),
        "imported_or_present": max(len(imported), len([item for item in sources if item.get("staged")])),
        "failed_imports": len([item for item in imports if item.get("status") == "failed"]),
        "external_inference_calls": 0,
        "training_use_allowed": False,
    }


def staged_artifact(source: dict[str, Any], local_dir: Path) -> dict[str, Any]:
    explicit_path = str(source.get("staged_path") or source.get("local_staged_path") or "")
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = ROOT / path
        metadata_value = str(source.get("staged_metadata_path") or "")
        metadata_path = Path(metadata_value) if metadata_value else Path("")
        if metadata_value and not metadata_path.is_absolute():
            metadata_path = ROOT / metadata_path
        return {
            "staged": path.exists(),
            "path": rel(path) if path.exists() else "",
            "metadata_path": rel(metadata_path) if metadata_value and metadata_path.exists() else "",
        }
    source_id = safe_filename(str(source.get("id") or "source"))
    import_policy = str(source.get("import_policy") or "queue_only")
    if import_policy == "source_archive":
        archive = local_dir / f"{source_id}.zip"
        metadata = archive.with_suffix(".metadata.json")
        return {
            "staged": archive.exists() and metadata.exists(),
            "path": rel(archive) if archive.exists() else "",
            "metadata_path": rel(metadata) if metadata.exists() else "",
        }
    if import_policy == "metadata_only":
        metadata = local_dir / source_id / "metadata.json"
        readme = local_dir / source_id / "README.md"
        return {
            "staged": metadata.exists(),
            "path": rel(readme) if readme.exists() else "",
            "metadata_path": rel(metadata) if metadata.exists() else "",
        }
    return {"staged": False, "path": "", "metadata_path": ""}


def repo_name_from_source(source: dict[str, Any]) -> str:
    name = str(source.get("name") or "")
    if "/" in name and " " not in name:
        return name.strip()
    url = str(source.get("url") or "")
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return ""


def download_capped(url: str, target: Path, max_bytes: int) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamOnlineSourceCatalog/0.1"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=90) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"download_exceeded_cap:{max_bytes}")
            digest.update(chunk)
            handle.write(chunk)
    return total, digest.hexdigest()


def fetch_json(url: str, max_bytes: int) -> dict[str, Any]:
    text = fetch_text(url, max_bytes)
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {}


def fetch_text(url: str, max_bytes: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamOnlineSourceCatalog/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"metadata_exceeded_cap:{max_bytes}")
    return data.decode("utf-8", errors="replace")


def looks_rom_related(source: dict[str, Any]) -> bool:
    text = " ".join(
        str(source.get(key) or "")
        for key in ["id", "name", "url", "why", "adapter_plan", "risk"]
    ).lower()
    return any(token in text for token in ["commercial rom", "gb rom", "gameboy rom", "game boy rom", ".gbc", ".gba"])


def normalize_license(value: Any) -> str:
    return str(value or "").strip().lower()


def safe_filename(value: Any) -> str:
    text = str(value or "source")
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in text)
    return safe[:120] or "source"


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


def sha256_file(path: Path) -> str:
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
