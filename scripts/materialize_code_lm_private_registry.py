#!/usr/bin/env python3
"""Materialize portable Code LM private-row registry files.

Windows workers historically write private training rows under
``D:/ProjectTheseus/training_data``.  macOS/Linux workers use the repo-local
``data/training_data`` root unless THESEUS_TRAINING_DATA_ROOT is set.  This
script reconstructs missing registry files from an existing private curriculum
that already records the source lane for each row.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from code_lm_private_rows import (  # noqa: E402
    BASE_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_ROWS,
    BROAD_FLOOR_RECOVERY_PRIVATE_ROWS,
    DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL,
    DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL,
    DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
)


DEFAULT_SOURCE = ROOT / "data" / "private_code_curriculum" / (
    "code_lm_closure_private_pressure_private_recovery_train_once_fanout_v1.jsonl"
)
DEFAULT_OUT = ROOT / "reports" / "code_lm_private_registry_materialization.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-curriculum", default=str(DEFAULT_SOURCE.relative_to(ROOT)))
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)))
    args = parser.parse_args()

    source = resolve(args.source_curriculum)
    out = resolve(args.out)
    rows = read_jsonl(source)
    targets = target_specs()
    target_reports: list[dict[str, Any]] = []
    for raw_target, label, fallback in targets:
        target = resolve(raw_target)
        selected = select_rows(rows, target, fallback)
        safe_rows = [normalize_row(row, source=source, label=label) for row in selected if row_is_safe(row)]
        existing_rows = count_jsonl_rows(target)
        write_allowed = bool(args.execute and (args.force or not target.exists()))
        if write_allowed:
            write_jsonl(target, safe_rows)
        target_reports.append(
            {
                "label": label,
                "path": rel(target),
                "exists_before": target.exists() and not write_allowed,
                "rows_before": existing_rows,
                "rows_selected": len(selected),
                "rows_safe": len(safe_rows),
                "written": write_allowed,
                "ready": bool(safe_rows and (write_allowed or target.exists())),
                "selection_rule": selection_rule(target, fallback),
            }
        )

    report = {
        "policy": "project_theseus_code_lm_private_registry_materialization_v1",
        "created_utc": now(),
        "execute": bool(args.execute),
        "force": bool(args.force),
        "source_curriculum": rel(source),
        "source_rows": len(rows),
        "target_count": len(target_reports),
        "ready_target_count": sum(1 for row in target_reports if row["ready"]),
        "missing_or_empty_targets": [
            row for row in target_reports if not row["ready"] or int(row["rows_safe"] or 0) <= 0
        ],
        "targets": target_reports,
        "public_boundary": {
            "public_benchmark_training": False,
            "public_tests_used": False,
            "public_solutions_used": False,
            "source": "existing private curriculum rows only",
        },
        "external_inference_calls": 0,
    }
    report["trigger_state"] = "GREEN" if not report["missing_or_empty_targets"] else "YELLOW"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["trigger_state"] in {"GREEN", "YELLOW"} else 2


def target_specs() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    return [
        (DEFAULT_EXTRA_PRIVATE_TRAIN_JSONL, "open_code_pantry", has_tag_or_card("open_code_permissive_pantry")),
        (
            DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL,
            "residual_code_curriculum",
            lambda row: row.get("source_id") == "local_generated_residual_code_curriculum",
        ),
        (
            DEFAULT_REPO_REPAIR_PRIVATE_TRAIN_JSONL,
            "long_horizon_repo_repair",
            lambda row: has_tag(row, "private_repo_repair")
            or str(row.get("source_id") or "").startswith("github:")
            or str(row.get("source_id") or "") == "private_repo_repair_hidden_tests",
        ),
        *broad_floor_specs(),
        *[
            (
                path,
                Path(path).name.removesuffix(".jsonl"),
                lambda row, source_name=Path(path).name: row_source_basename(row) == source_name,
            )
            for path in BASE_PROMOTION_SAFE_HIGH_TRANSFER_PRIVATE_ROWS
        ],
    ]


def broad_floor_specs() -> list[tuple[str, str, Callable[[dict[str, Any]], bool]]]:
    specs: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = []
    for path in BROAD_FLOOR_RECOVERY_PRIVATE_ROWS:
        if path == DEFAULT_RESIDUAL_PRIVATE_TRAIN_JSONL:
            continue
        name = Path(path).name
        if name == "broad_public_transfer_floor_ratchet_v2_private_rows.jsonl":
            fallback = lambda row: row.get("source_id") == "local_generated_broad_public_floor_recovery_private_pressure"
        else:
            fallback = lambda row, source_name=name: row_source_basename(row) == source_name
        specs.append((path, name.removesuffix(".jsonl"), fallback))
    return specs


def select_rows(rows: list[dict[str, Any]], target: Path, fallback: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    source_name = target.name
    exact = [row for row in rows if row_source_basename(row) == source_name]
    if exact:
        return exact
    return [row for row in rows if fallback(row)]


def row_source_basename(row: dict[str, Any]) -> str:
    direct = str(row.get("high_transfer_source_jsonl") or "")
    if direct:
        return path_basename(direct)
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("source_jsonl", "source_private_jsonl"):
        value = str(provenance.get(key) or "")
        if value:
            return path_basename(value)
    return ""


def has_tag_or_card(value: str) -> Callable[[dict[str, Any]], bool]:
    return lambda row: has_tag(row, value) or str(row.get("card_id") or "") == value


def has_tag(row: dict[str, Any], value: str) -> bool:
    tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    return value in {str(tag) for tag in tags}


def row_is_safe(row: dict[str, Any]) -> bool:
    if row.get("public_benchmark") is True:
        return False
    if not str(row.get("license_spdx") or "").strip():
        return False
    if not str(row.get("prompt") or "").strip():
        return False
    if not str(row.get("entry_point") or "").strip():
        return False
    if not (str(row.get("solution_expr") or "").strip() or str(row.get("solution_body") or "").strip()):
        return False
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    for key in ("public_tests_used", "public_solutions_used", "public_benchmark_answers_used"):
        if provenance.get(key) is True:
            return False
    return True


def normalize_row(row: dict[str, Any], *, source: Path, label: str) -> dict[str, Any]:
    out = dict(row)
    out["public_benchmark"] = False
    out.setdefault("benchmark_evidence_level", "private_generated_training_or_eval")
    provenance = dict(out.get("provenance") or {})
    provenance.update(
        {
            "materialized_registry_label": label,
            "materialized_from_private_curriculum": rel(source),
            "materialized_private_registry_policy": "project_theseus_code_lm_private_registry_materialization_v1",
            "public_tests_used": False,
            "public_solutions_used": False,
            "public_benchmark_answers_used": False,
        }
    )
    out["provenance"] = provenance
    return out


def selection_rule(target: Path, fallback: Callable[[dict[str, Any]], bool]) -> str:
    del fallback
    return (
        f"prefer rows whose high_transfer_source_jsonl/provenance source basename is {target.name}; "
        "fallback only for pantry/residual/repo/broad rows without exact source labels"
    )


def path_basename(raw: str) -> str:
    return Path(str(raw).replace("\\", "/")).name


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def count_jsonl_rows(path: Path) -> int:
    return len(read_jsonl(path))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
