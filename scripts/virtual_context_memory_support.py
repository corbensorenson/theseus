"""Support helpers for Virtual Context Memory.

Pure rendering, IO, hashing, path, and small value helpers live here so the VCM
core file can stay focused on memory-page semantics, graph construction, gates,
and operator commands.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def render_bench_markdown(report: dict[str, Any]) -> str:
    summary = object_field(report, "summary")
    gates = report.get("gates") if isinstance(report.get("gates"), list) else []
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    lines = [
        "# Virtual Context Memory Bench",
        "",
        f"State: `{report.get('trigger_state')}`",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary.get('case_count')}`",
        f"- VCM score: `{summary.get('vcm_score')}`",
        f"- Packet baseline score: `{summary.get('packet_baseline_score')}`",
        f"- Lexical baseline score: `{summary.get('lexical_baseline_score')}`",
        f"- External inference calls: `{summary.get('external_inference_calls')}`",
        "",
        "## Gates",
        "",
    ]
    for row in gates:
        if isinstance(row, dict):
            lines.append(f"- `{'PASS' if row.get('passed') else 'FAIL'}` `{row.get('gate')}`: {row.get('evidence')}")
    lines.extend(["", "## Cases", ""])
    for case in cases:
        if not isinstance(case, dict):
            continue
        lines.append(
            f"- `{case.get('id')}`: VCM={case.get('vcm_passed')} "
            f"packet={case.get('packet_baseline_passed')} lexical={case.get('lexical_baseline_passed')}"
        )
    return "\n".join(lines) + "\n"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {} if default is None else default


def read_jsonl_all(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_hash_or_text_hash(source_path: str, text: str) -> str:
    path = resolve(source_path)
    if path.exists() and path.is_file():
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return f"sha256:{digest.hexdigest()}"
        except OSError:
            pass
    return f"sha256:{sha256_text(text)}"


def modified_utc(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return now()


def parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def object_field(value: Any, key: str) -> dict[str, Any]:
    item = value.get(key) if isinstance(value, dict) else None
    return item if isinstance(item, dict) else {}


def get_path(value: Any, path: list[str], default: Any = None) -> Any:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compact_json(value: Any, limit: int) -> str:
    text = canonical_json(value)
    return truncate(text, limit)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def stable_id(value: str) -> str:
    return sha256_text(value)[:24]


def snapshot_id(pages: list[dict[str, Any]], task: str) -> str:
    basis = task + "\n" + "\n".join(sorted(str(page.get("content_hash") or "") for page in pages))
    return f"snap:{stable_id(basis)}"


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "item"


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", value.lower())


def estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4)


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
