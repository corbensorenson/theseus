"""Benchmark and training-data seeker for SparkStream.

This is deliberately conservative: by default it inventories local/public
benchmark assets, queues user-provided URLs for review, and writes structured
recommendations. Network fetching is opt-in so the autonomous loop does not
silently expand its data boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "reports" / "benchmark_seeker_registry.json"
ALLOWED_OPEN_LICENSES = {
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "mit",
    "mpl-2.0",
    "odc-by",
    "unlicense",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_REGISTRY.relative_to(ROOT)))
    parser.add_argument("--refresh-local", action="store_true")
    parser.add_argument("--add-url", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--fetch-url", default="")
    parser.add_argument("--allow-network-discovery", action="store_true")
    parser.add_argument("--discover-query", default="")
    parser.add_argument("--discover-limit", type=int, default=10)
    args = parser.parse_args()

    registry = read_json(ROOT / args.out) or base_registry()
    registry["updated_utc"] = now()

    if args.refresh_local:
        registry["local_inventory"] = local_inventory()
        registry["recommendations"] = recommendations(registry["local_inventory"])

    if args.add_url:
        registry.setdefault("queued_external_candidates", [])
        existing = next(
            (
                item
                for item in registry["queued_external_candidates"]
                if isinstance(item, dict) and item.get("url") == args.add_url
            ),
            None,
        )
        if existing is None:
            registry["queued_external_candidates"].append(
                {
                    "id": f"url_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
                    "name": args.name or args.add_url,
                    "url": args.add_url,
                    "notes": args.notes,
                    "status": "queued_for_teacher_or_manual_import",
                    "network_fetched": false(),
                    "created_utc": now(),
                }
            )
        else:
            existing["name"] = args.name or existing.get("name") or args.add_url
            existing["notes"] = args.notes or existing.get("notes", "")
            existing["updated_utc"] = now()

    if args.fetch_url:
        if not args.allow_network_fetch:
            registry.setdefault("network_fetch_attempts", []).append(
                {
                    "url": args.fetch_url,
                    "status": "blocked_policy_requires_allow_network_fetch",
                    "created_utc": now(),
                }
            )
        else:
            registry.setdefault("network_fetch_attempts", []).append(fetch_url(args.fetch_url))

    if args.discover_query:
        if not args.allow_network_discovery:
            registry.setdefault("network_discovery_attempts", []).append(
                {
                    "query": args.discover_query,
                    "status": "blocked_policy_requires_allow_network_discovery",
                    "created_utc": now(),
                }
            )
        else:
            discovery = discover_external_sources(
                args.discover_query,
                args.discover_limit,
                registry.get("discovered_external_candidates") or [],
            )
            candidates = discovery.pop("candidates", [])
            registry.setdefault("network_discovery_attempts", []).append(discovery)
            registry.setdefault("discovered_external_candidates", []).extend(candidates)

    write_json(ROOT / args.out, registry)
    print(json.dumps(registry, indent=2))
    return 0


def base_registry() -> dict[str, Any]:
    return {
        "policy": "sparkstream_benchmark_seeker_v0",
        "created_utc": now(),
        "updated_utc": now(),
        "network_policy": "queued_only_unless_allow_network_fetch",
        "local_inventory": {},
        "queued_external_candidates": [],
        "network_fetch_attempts": [],
        "network_discovery_attempts": [],
        "discovered_external_candidates": [],
        "recommendations": [],
    }


def local_inventory() -> dict[str, Any]:
    external_benchmarks = toml_summary(ROOT / "configs" / "external_benchmarks.toml")
    return {
        "benchmark_ledger": ledger_summary(ROOT / "reports" / "benchmark_ledger.json"),
        "public_manifest": read_json(ROOT / "data" / "public_benchmarks" / "manifest.json"),
        "local_benchmark_paths": toml_summary(ROOT / "configs" / "local_benchmark_paths.toml"),
        "external_benchmarks": external_benchmarks,
        "knowledge_sources": knowledge_source_summary(external_benchmarks),
        "jsonl_data": jsonl_inventory([ROOT / "data", ROOT / "benchmarks"]),
        "report_families": report_families(ROOT / "reports"),
    }


def ledger_summary(path: Path) -> dict[str, Any]:
    rows = read_json(path)
    if not isinstance(rows, list):
        return {"exists": path.exists(), "benchmarks": []}
    return {
        "exists": True,
        "count": len(rows),
        "benchmarks": [
            {
                "name": row.get("benchmark_name"),
                "lifecycle": row.get("lifecycle"),
                "score": row.get("score"),
                "residual": row.get("residual"),
                "wall_type": row.get("wall_type"),
                "current_threshold": get_path(row, ["graduation_policy", "current_threshold"], None),
                "floor_threshold": get_path(row, ["graduation_policy", "floor_threshold"], None),
                "recommended_intervention": row.get("recommended_intervention"),
            }
            for row in rows
            if isinstance(row, dict)
        ],
    }


def toml_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if tomllib is None:
        data = simple_toml(path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "parser": "simple_fallback",
            "path": str(path.relative_to(ROOT)),
            "sections": sorted(data.keys()),
            "data": data,
        }
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "sections": sorted(data.keys()),
        "data": data,
    }


def knowledge_source_summary(external_benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    data = external_benchmarks.get("data")
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, section in data.items():
        if not isinstance(section, dict):
            continue
        if section.get("source_type") != "knowledge_source":
            continue
        rows.append(
            {
                "name": name,
                "url": section.get("url"),
                "status": section.get("status"),
                "terms_url": section.get("terms_url"),
                "intended_uses": section.get("intended_uses", []),
                "blocked_uses_without_permission_or_license_audit": section.get(
                    "blocked_uses_without_permission_or_license_audit", []
                ),
                "required_gates": section.get("required_gates", []),
            }
        )
    return rows


def jsonl_inventory(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                line_count = count_lines(path)
            except OSError:
                line_count = None
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "line_count": line_count,
                    "role_guess": role_guess(path),
                }
            )
    rows.sort(key=lambda item: (item["role_guess"], item["path"]))
    return rows[:500]


def report_families(reports: Path) -> dict[str, Any]:
    families: dict[str, int] = {}
    if not reports.exists():
        return families
    for path in reports.glob("*.json"):
        prefix = path.stem.split("_seed")[0]
        prefix = prefix.split("_train")[0]
        families[prefix] = families.get(prefix, 0) + 1
    return dict(sorted(families.items(), key=lambda item: (-item[1], item[0]))[:50])


def recommendations(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    ledger_rows = get_path(inventory, ["benchmark_ledger", "benchmarks"], [])
    active_frontiers = [
        bench for bench in ledger_rows if isinstance(bench, dict) and bench.get("lifecycle") == "frontier"
    ]
    regressions = [
        bench for bench in ledger_rows if isinstance(bench, dict) and bench.get("lifecycle") == "regression"
    ]
    for bench in get_path(inventory, ["benchmark_ledger", "benchmarks"], []):
        score = bench.get("score")
        floor = bench.get("floor_threshold")
        wall = bench.get("wall_type")
        if isinstance(score, (int, float)) and isinstance(floor, (int, float)) and score < floor:
            recs.append(
                {
                    "kind": "frontier_floor_gap",
                    "priority": "high",
                    "benchmark": bench.get("name"),
                    "score": score,
                    "floor": floor,
                    "wall_type": wall,
                    "action": "generate bridge data, run targeted residual ablation, and consider teacher escalation if repeated",
                }
            )
        elif wall and wall != "no_current_wall":
            recs.append(
                {
                    "kind": "wall_diagnosis",
                    "priority": "medium",
                    "benchmark": bench.get("name"),
                    "wall_type": wall,
                    "action": bench.get("recommended_intervention"),
                }
            )
    jsonl = inventory.get("jsonl_data") or []
    if not any("seed55" in item.get("path", "") for item in jsonl):
        recs.append(
            {
                "kind": "missing_babylm_frontier_data",
                "priority": "high",
                "action": "generate or import the next mutated frontier holdout",
            }
        )
    if not active_frontiers and regressions:
        recs.insert(
            0,
            {
                "kind": "frontier_exhausted",
                "priority": "high",
                "action": (
                    "do not rerun saturated surfaces; rotate to a fresh mutated holdout, "
                    "import a licensed benchmark candidate, or create an RL adapter smoke frontier"
                ),
                "regression_count": len(regressions),
            },
        )
    for source in inventory.get("knowledge_sources", []):
        status = source.get("status", "")
        if "pending" in status or "lookup_only" in status:
            recs.append(
                {
                    "kind": "knowledge_source_gated",
                    "priority": "medium",
                    "benchmark": source.get("name"),
                    "status": status,
                    "action": (
                        "use for targeted lookup only; complete terms, robots, provenance, "
                        "and human approval gates before training ingestion"
                    ),
                }
            )
    if not recs:
        recs.append(
            {
                "kind": "frontier_required",
                "priority": "high",
                "action": "no actionable frontier pressure found; generate or import the next harder benchmark before running more training",
            }
        )
    return recs


def simple_toml(text: str) -> dict[str, Any]:
    """Parse the small local TOML shape used by configs/*.toml.

    This is not a general TOML parser. It handles section headers, strings,
    one-line arrays of strings, and multi-line arrays of strings.
    """

    data: dict[str, Any] = {}
    section = "root"
    data[section] = {}
    pending_key: str | None = None
    pending_values: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if pending_key is not None:
            if "]" in line:
                before = line.split("]", 1)[0]
                pending_values.extend(parse_array_items(before))
                data[section][pending_key] = pending_values
                pending_key = None
                pending_values = []
            else:
                pending_values.extend(parse_array_items(line))
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            data.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value == "[":
            pending_key = key
            pending_values = []
        elif value.startswith("[") and value.endswith("]"):
            data[section][key] = parse_array_items(value.strip("[]"))
        else:
            data[section][key] = parse_scalar(value)
    return data


def parse_array_items(text: str) -> list[str]:
    items: list[str] = []
    for part in text.split(","):
        value = parse_scalar(part.strip())
        if value != "":
            items.append(str(value))
    return items


def parse_scalar(value: str) -> Any:
    value = value.strip().rstrip(",")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def fetch_url(url: str) -> dict[str, Any]:
    target_dir = ROOT / "data" / "external_benchmark_candidates"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").split("/")[-1] or "downloaded_benchmark"
    safe_name = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in filename)[:120]
    target = target_dir / safe_name
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(50 * 1024 * 1024)
        target.write_bytes(data)
        return {
            "url": url,
            "status": "downloaded_pending_audit",
            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(data),
            "created_utc": now(),
        }
    except Exception as exc:  # noqa: BLE001 - report, do not crash dashboard flow.
        return {
            "url": url,
            "status": "download_failed",
            "error": str(exc),
            "created_utc": now(),
        }


def discover_external_sources(query: str, limit: int, existing_candidates: list[Any]) -> dict[str, Any]:
    capped_limit = max(1, min(limit, 50))
    attempt = {
        "query": query,
        "limit": capped_limit,
        "created_utc": now(),
        "sources": [],
        "status": "ok",
    }
    try:
        hf_limit = max(1, capped_limit // 2)
        github_limit = max(1, capped_limit - hf_limit)
        attempt["sources"] = discover_huggingface_datasets(query, hf_limit)
        attempt["sources"].extend(discover_github_repositories(query, github_limit))
    except Exception as exc:  # noqa: BLE001 - record discovery failure.
        attempt["status"] = "failed"
        attempt["error"] = str(exc)
    seen = {item.get("url") for item in existing_candidates if isinstance(item, dict)}
    new_items = []
    for source in attempt.get("sources", []):
        url = source.get("url")
        if url in seen:
            continue
        new_items.append(
            {
                "id": f"discovered_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{len(new_items)}",
                "name": source.get("name"),
                "url": url,
                "source_kind": source.get("source_kind"),
                "query": query,
                "status": source.get("audit_status") or "discovered_pending_audit",
                "license_spdx": source.get("license_spdx") or "unknown",
                "audit_status": source.get("audit_status") or "pending_license_audit",
                "network_fetched": False,
                "metadata": source.get("metadata", {}),
                "created_utc": now(),
            }
        )
        seen.add(url)
    attempt["queued_new_candidates"] = len(new_items)
    attempt["candidates"] = new_items
    return attempt


def discover_huggingface_datasets(query: str, limit: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"search": query, "limit": str(limit)})
    url = f"https://huggingface.co/api/datasets?{encoded}"
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamBenchmarkSeeker/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))
    rows = payload if isinstance(payload, list) else []
    sources = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        dataset_id = row.get("id") or row.get("_id")
        if not dataset_id:
            continue
        tags = row.get("tags", []) if isinstance(row.get("tags"), list) else []
        license_spdx = license_from_tags(tags)
        sources.append(
            {
                "source_kind": "huggingface_dataset",
                "name": dataset_id,
                "url": f"https://huggingface.co/datasets/{dataset_id}",
                "license_spdx": license_spdx or "unknown",
                "audit_status": audit_status_for_license(license_spdx),
                "metadata": {
                    "downloads": row.get("downloads"),
                    "likes": row.get("likes"),
                    "tags": tags[:20],
                    "last_modified": row.get("lastModified"),
                },
            }
        )
    return sources


def discover_github_repositories(query: str, limit: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"q": query, "per_page": str(max(1, min(limit, 25)))})
    url = f"https://api.github.com/search/repositories?{encoded}"
    request = urllib.request.Request(url, headers={"User-Agent": "SparkStreamBenchmarkSeeker/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))
    sources = []
    for row in payload.get("items", [])[:limit]:
        if not isinstance(row, dict):
            continue
        license_info = row.get("license") or {}
        license_spdx = normalize_license(license_info.get("spdx_id"))
        sources.append(
            {
                "source_kind": "github_repo",
                "name": row.get("full_name"),
                "url": row.get("html_url"),
                "license_spdx": license_spdx or "unknown",
                "audit_status": audit_status_for_license(license_spdx),
                "metadata": {
                    "description": row.get("description"),
                    "stars": row.get("stargazers_count"),
                    "updated_at": row.get("updated_at"),
                    "topics": row.get("topics", [])[:20] if isinstance(row.get("topics"), list) else [],
                },
            }
        )
    return sources


def license_from_tags(tags: list[Any]) -> str:
    for tag in tags:
        text = str(tag).lower()
        if text.startswith("license:"):
            return normalize_license(text.split(":", 1)[1])
    return ""


def normalize_license(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text and text != "other" else ""


def audit_status_for_license(license_spdx: str) -> str:
    if normalize_license(license_spdx) in ALLOWED_OPEN_LICENSES:
        return "approved_open_license_pending_import"
    return "pending_license_audit"


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def role_guess(path: Path) -> str:
    text = str(path).lower()
    if "mutated" in text or "holdout" in text:
        return "frontier_or_holdout"
    if "bridge" in text:
        return "bridge"
    if "train" in text:
        return "training"
    if "eval" in text:
        return "evaluation"
    return "dataset"


def false() -> bool:
    return False


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
