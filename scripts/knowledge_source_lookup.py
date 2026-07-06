"""Gated knowledge-source lookup planner for SparkStream.

Knowledge sources are useful for claim checks, topic discovery, and benchmark
ideas, but they are not automatically training data. This script turns a
requested lookup into an auditable report and blocks autonomous fetch/training
use unless the source policy explicitly allows it.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "autonomy_policy.json"
DEFAULT_EXTERNAL = ROOT / "configs" / "external_benchmarks.toml"
DEFAULT_OUT = ROOT / "reports" / "knowledge_source_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_POLICY.relative_to(ROOT)))
    parser.add_argument("--external-config", default=str(DEFAULT_EXTERNAL.relative_to(ROOT)))
    parser.add_argument("--source", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--allow-network-fetch", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    policy = read_json(ROOT / args.policy)
    external = read_toml(ROOT / args.external_config)
    sources = merge_sources(policy_sources(policy), external_sources(external))

    report: dict[str, Any] = {
        "policy": "sparkstream_knowledge_sources_v0",
        "updated_utc": now(),
        "lookup_only_by_default": bool(get_path(policy, ["knowledge_sources", "lookup_only_by_default"], True)),
        "autonomous_bulk_training_ingest": bool(
            get_path(policy, ["knowledge_sources", "autonomous_bulk_training_ingest"], False)
        ),
        "sources": sources,
        "lookup_request": {},
    }

    if args.list or (not args.source and not args.query and not args.url):
        report["lookup_request"] = {
            "status": "listed_sources",
            "message": "Knowledge sources are listed for dashboard and autonomy-cycle visibility.",
        }
    else:
        report["lookup_request"] = plan_lookup(
            sources=sources,
            source_name=args.source,
            query=args.query,
            url=args.url,
            allow_network_fetch=args.allow_network_fetch,
        )

    write_json(ROOT / args.out, report)
    print(json.dumps(report, indent=2))
    return 0


def policy_sources(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = get_path(policy, ["knowledge_sources", "sources"], [])
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(normalize_source(row, "autonomy_policy"))
    return normalized


def external_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, section in config.items():
        if not isinstance(section, dict):
            continue
        if section.get("source_type") != "knowledge_source":
            continue
        source = dict(section)
        source.setdefault("name", name)
        rows.append(normalize_source(source, "external_benchmarks"))
    return rows


def normalize_source(source: dict[str, Any], registry: str) -> dict[str, Any]:
    return {
        "name": str(source.get("name") or "unnamed_source"),
        "url": str(source.get("url") or ""),
        "source_type": str(source.get("source_type") or "knowledge_source"),
        "status": str(source.get("status") or "pending_audit"),
        "terms_url": str(source.get("terms_url") or ""),
        "license_notes": str(source.get("license_notes") or source.get("notes") or ""),
        "allowed_uses": listify(source.get("allowed_uses") or source.get("intended_uses")),
        "blocked_uses_without_permission_or_license_audit": listify(
            source.get("blocked_uses_without_permission_or_license_audit")
        ),
        "required_gates": listify(source.get("required_gates")),
        "autonomous_fetch_allowed": bool(source.get("autonomous_fetch_allowed", False)),
        "training_use_allowed": bool(source.get("training_use_allowed", False)),
        "registry": registry,
    }


def merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for source in group:
            key = source["name"].lower()
            if key not in merged:
                merged[key] = source
                continue
            current = merged[key]
            for field, value in source.items():
                if field in {"allowed_uses", "blocked_uses_without_permission_or_license_audit", "required_gates"}:
                    current[field] = sorted(set(current.get(field, []) + listify(value)))
                elif not current.get(field) and value:
                    current[field] = value
            current["registry"] = ",".join(sorted(set(str(current.get("registry", "")).split(",") + [source["registry"]])))
    return sorted(merged.values(), key=lambda item: item["name"])


def plan_lookup(
    *,
    sources: list[dict[str, Any]],
    source_name: str,
    query: str,
    url: str,
    allow_network_fetch: bool,
) -> dict[str, Any]:
    source = select_source(sources, source_name, url)
    if not source:
        return {
            "status": "blocked_unknown_source",
            "source": source_name,
            "query": query,
            "url": url,
            "message": "Register the source before lookup so terms, permissions, and provenance are explicit.",
        }

    lookup_url = url or source.get("url", "")
    query_hint = build_query_hint(source, query)
    training_allowed = bool(source.get("training_use_allowed"))
    fetch_allowed = bool(source.get("autonomous_fetch_allowed"))
    network_status = "not_requested"
    if allow_network_fetch and not fetch_allowed:
        network_status = "blocked_by_source_policy"
    elif allow_network_fetch and fetch_allowed:
        network_status = "allowed_by_source_policy_pending_fetch_adapter"

    return {
        "status": "lookup_planned",
        "source": source.get("name"),
        "source_url": source.get("url"),
        "lookup_url": lookup_url,
        "query": query,
        "query_hint": query_hint,
        "network_status": network_status,
        "training_use_allowed": training_allowed,
        "policy_decision": "training_ingest_allowed" if training_allowed else "training_ingest_blocked_pending_audit",
        "allowed_uses": source.get("allowed_uses", []),
        "blocked_uses_without_permission_or_license_audit": source.get(
            "blocked_uses_without_permission_or_license_audit", []
        ),
        "required_gates": source.get("required_gates", []),
        "terms_url": source.get("terms_url"),
        "message": (
            "Use this source for targeted lookup/provenance notes only. Do not bulk ingest or train on copied "
            "content until all gates pass."
        ),
    }


def select_source(sources: list[dict[str, Any]], source_name: str, url: str) -> dict[str, Any] | None:
    if source_name:
        wanted = source_name.lower()
        for source in sources:
            if source.get("name", "").lower() == wanted:
                return source
    if url:
        host = urllib.parse.urlparse(url).netloc.lower()
        for source in sources:
            source_host = urllib.parse.urlparse(str(source.get("url") or "")).netloc.lower()
            if host and host == source_host:
                return source
    return None


def build_query_hint(source: dict[str, Any], query: str) -> str:
    if not query:
        return "Open the source UI or provide a specific page URL for a scoped lookup."
    host = urllib.parse.urlparse(str(source.get("url") or "")).netloc
    if host:
        return f"Use a targeted search such as: site:{host} {query}"
    return query


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_toml(path: Path) -> dict[str, Any]:
    if not path.exists() or tomllib is None:
        return {}
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_path(value: Any, path: list[str], default: Any) -> Any:
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
