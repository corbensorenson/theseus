#!/usr/bin/env python3
"""Role-separated audit of the all-62 parent-store batch."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_parent_only_materializer_audit as base_audit  # noqa: E402
import theseus_vcm_parent_store_batch as producer  # noqa: E402

POLICY = "project_theseus_vcm_parent_store_batch_audit_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_parent_store_batch.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = p2a.resolve(args.config)
    cfg = p2a.read_json(path)
    report = audit(path)
    p2a.write_json(p2a.resolve(args.out or str(cfg.get("audit_report") or "")), report)
    print(json.dumps({key: report.get(key) for key in ("trigger_state", "state", "faults", "audited_row_count", "audited_candidate_visible_field_count", "candidate_or_control_calls", "external_reference_calls")}, indent=2, sort_keys=True))
    return 0 if report.get("trigger_state") == "GREEN" else 2


def audit(path: Path = DEFAULT_CONFIG, *, producer_report: dict[str, Any] | None = None, store: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = p2a.read_json(path)
    faults: list[str] = []
    producer.validate_binding(cfg, "audit_owner", "audit_owner_sha256", Path(__file__).resolve(), faults)
    producer.validate_binding(cfg, "base_audit_owner", "base_audit_owner_sha256", Path(base_audit.__file__).resolve(), faults)
    if cfg.get("audit_policy") != POLICY:
        faults.append("audit_policy_invalid")
    manifest, rows = producer.load_manifest(cfg, faults)
    actual_report = producer_report if producer_report is not None else p2a.read_json(p2a.resolve(str(cfg.get("report") or "")))
    actual_store = store if store is not None else p2a.read_json(p2a.resolve(str(cfg.get("store_out") or "")))
    replay_cfg = {
        "audit_policy": base_audit.POLICY,
        "audit_owner": p2a.rel(Path(base_audit.__file__).resolve()),
        "audit_owner_sha256": p2a.sha256_file(Path(base_audit.__file__).resolve()),
        "owner": str(cfg.get("base_materializer")),
        "owner_sha256": str(cfg.get("base_materializer_sha256")),
        "broad_parent_effect_root": "repository",
        "expected_row_count": int(cfg.get("expected_row_count") or 0),
        "rows": rows,
        "report": str(cfg.get("report")),
        "store_out": str(cfg.get("store_out")),
        "audit_maximum_inference": cfg.get("audit_maximum_inference"),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(replay_cfg, handle, sort_keys=True)
        handle.flush()
        replay = base_audit.audit(Path(handle.name), producer=actual_report, store=actual_store)
    faults.extend(p2a.strings(replay.get("faults")))
    if actual_report.get("state") != "K2_05_ALL_62_PARENT_STORES_MATERIALIZED" or actual_report.get("panel_admitted") is not False:
        faults.append("producer_state_or_admission_invalid")
    ready = not faults and replay.get("audited_row_count") == int(cfg.get("expected_row_count") or 0)
    return {
        **replay,
        "policy": POLICY,
        "created_utc": p2a.now(),
        "trigger_state": "GREEN" if ready else "RED",
        "state": "K2_05_ALL_62_PARENT_STORES_ROLE_SEPARATELY_REDERIVED" if ready else "K2_05_PARENT_STORE_BATCH_AUDIT_FAILED",
        "faults": sorted(set(faults)),
        "config": {"path": p2a.rel(path), "sha256": p2a.sha256_file(path)},
        "manifest": {"path": p2a.mapping(cfg.get("manifest")).get("path"), "sha256": p2a.mapping(cfg.get("manifest")).get("sha256"), "policy": manifest.get("policy")},
        "producer_report": {"path": cfg.get("report"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("report") or "")))},
        "store_artifact": {"path": cfg.get("store_out"), "sha256": p2a.sha256_file(p2a.resolve(str(cfg.get("store_out") or "")))},
        "panel_admitted": False,
        "audit_kind": "role-separated rederivation",
        "maximum_inference": cfg.get("audit_maximum_inference"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
