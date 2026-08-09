#!/usr/bin/env python3
"""Prospective resource/risk preflight for the six immutable environments."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402

POLICY = "project_theseus_vcm_six_row_environment_preflight_v1"
STATE = "PROSPECTIVE_K2_05_SIX_ROW_ENVIRONMENT_RESOURCE_AND_RISK_PREFLIGHT"
DEFAULT_CONFIG = ROOT / "configs/theseus_vcm_six_row_environment_preflight.json"


def lock_count(manager: str, path: Path) -> int:
    if manager == "cargo":
        return len(tomllib.loads(path.read_text())["package"])
    pattern = re.compile(r"^[A-Za-z0-9_.-]+==[^\\\s]+", re.MULTILINE)
    return len(pattern.findall(path.read_text()))


def evaluate(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    if cfg.get("policy") != POLICY or cfg.get("state") != STATE:
        faults.append("policy_or_state_invalid")
    owner = p2a.resolve(str(cfg.get("owner") or ""))
    if owner != Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner) != cfg.get("owner_sha256"):
        faults.append("owner_binding_invalid")
    sources: dict[str, dict[str, Any]] = {}
    for name, raw in p2a.mapping(cfg.get("sources")).items():
        binding = p2a.mapping(raw)
        source = p2a.resolve(str(binding.get("path") or ""))
        if not source.is_file() or p2a.sha256_file(source) != binding.get("sha256"):
            faults.append(f"source_binding_invalid:{name}")
            sources[name] = {}
        else:
            sources[name] = p2a.read_json(source)
    resolution = sources.get("resolution", {})
    audit = sources.get("resolution_audit", {})
    builder = sources.get("instrument_builder", {})
    if resolution.get("qualified_task_count") != 6 or resolution.get("inconclusive_task_count") != 0 or audit.get("trigger_state") != "GREEN":
        faults.append("six_lock_predecessor_invalid")
    risk = sources.get("instrument_builder_audit", {})
    qualified_risks = p2a.strings(risk.get("qualified_risk_classes"))
    if risk.get("trigger_state") != "GREEN" or risk.get("state") != "K2_03_GENERIC_ECOSYSTEM_RISK_EVIDENCE_ROLE_SEPARATELY_REDERIVED" or len(qualified_risks) != 4:
        faults.append("untrusted_build_risk_preflight_invalid")
    coefficients = p2a.mapping(p2a.mapping(builder.get("batch_preflight")).get("manager_projection"))
    rows: list[dict[str, Any]] = []
    for row in p2a.dicts(cfg.get("rows")):
        manager = str(row.get("manager") or "")
        lock = p2a.resolve(str(row.get("lock") or ""))
        if not lock.is_file() or p2a.sha256_file(lock) != row.get("sha256"):
            faults.append(f"lock_binding_invalid:{row.get('index')}")
            continue
        count = lock_count(manager, lock)
        coefficient = int(p2a.mapping(coefficients.get(manager)).get("observed_upper_bytes_per_entry") or 0)
        if count != row.get("package_count") or coefficient <= 0:
            faults.append(f"projection_input_invalid:{row.get('index')}")
        rows.append({"index": row.get("index"), "manager": manager, "package_count": count, "upper_bytes_per_entry": coefficient, "projected_store_or_environment_bytes": count * coefficient})
    limits = p2a.mapping(cfg.get("limits"))
    shared = sum(int(row["projected_store_or_environment_bytes"]) for row in rows)
    largest = max((int(row["projected_store_or_environment_bytes"]) for row in rows), default=0)
    temporary = int(limits.get("projected_temporary_bytes") or 0)
    reserve = int(limits.get("minimum_free_bytes_after_execution") or 0)
    free = shutil.disk_usage(ROOT).free
    required = shared + largest + temporary
    ready = not faults and free - required >= reserve
    return {"policy": POLICY, "created_utc": p2a.now(), "trigger_state": "GREEN" if not faults else "RED", "state": "K2_05_SIX_ROW_ENVIRONMENT_MATERIALIZATION_PREFLIGHT_READY" if ready else ("K2_05_SIX_ROW_ENVIRONMENT_MATERIALIZATION_HOST_STORAGE_WALL" if not faults else "K2_05_SIX_ROW_ENVIRONMENT_PREFLIGHT_INVALID"), "faults": sorted(set(faults)), "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)}, "rows": rows, "task_count": len(rows), "projected_shared_store_upper_bytes": shared, "projected_largest_disposable_environment_bytes": largest, "projected_temporary_bytes": temporary, "required_incremental_peak_bytes": required, "host_free_bytes": free, "minimum_free_bytes_after_execution": reserve, "execution_ready": ready, "untrusted_build_risk_classes_qualified": len(qualified_risks) == 4, "qualified_risk_classes": qualified_risks, "network_or_dependency_execution_performed": False, "package_installations": 0, "repository_runner_executions": 0, "parent_target_or_evaluator_executions": 0, "candidate_or_control_calls": 0, "external_reference_calls": 0, "panel_admitted": False, "maximum_inference": cfg.get("maximum_inference")}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG)); parser.add_argument("--out", default="")
    args = parser.parse_args(); path = p2a.resolve(args.config); cfg = p2a.read_json(path); report = evaluate(path); p2a.write_json(p2a.resolve(args.out or cfg["report"]), report); print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["trigger_state"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
