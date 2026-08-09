#!/usr/bin/env python3
"""Replace VCM rows invalidated by explicit unsupported-host source manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_source_materialization as source  # noqa: E402
import theseus_vcm_three_row_adequacy_replacements as predecessor  # noqa: E402

POLICY = "project_theseus_vcm_host_adequacy_replacements_v1"
DEFAULT_CONFIG = ROOT / "configs" / "theseus_vcm_host_adequacy_replacements.json"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=p2a.rel(DEFAULT_CONFIG)); parser.add_argument("--out", default=""); parser.add_argument("--checkpoint", default=""); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); config_path = p2a.resolve(args.config); config = p2a.read_json(config_path); report = preflight(config_path)
    if args.execute and report["trigger_state"] == "GREEN":
        retry = p2a.mapping(config.get("transport_retry_policy")); ledger = source.SourceLedger(p2a.resolve(args.checkpoint or config["checkpoint"]), config_path, retry); client = source.SourceClient(ledger, retry)
        try: report = acquire(config_path, ledger, client, retry)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            report = {**report, "trigger_state":"PAUSED", "state":"HOST_ADEQUACY_REPLACEMENT_TRANSPORT_OR_CLASSIFIER_PAUSED_NO_ADMISSION", "faults":[f"{type(exc).__name__}:{exc}"[:4000]], "replacement_set_admitted":False}
        report = source.finalize_receipt(report, ledger, client)
    p2a.write_json(p2a.resolve(args.out or config["report"]), report); print(json.dumps(predecessor.summary(report), indent=2, sort_keys=True))
    return 0 if report["trigger_state"] in {"GREEN", "PAUSED"} else 2


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = p2a.read_json(config_path); faults: list[str] = []
    if config.get("policy") != POLICY: faults.append("policy_invalid")
    for binding in p2a.dicts(config.get("source_bindings")):
        path = p2a.resolve(str(binding.get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path) != binding.get("sha256"): faults.append(f"source_binding_invalid:{binding.get('id')}")
    panel_path = p2a.resolve(str(config.get("source_panel") or "")); closure_path = p2a.resolve(str(config.get("repository_closure_report") or "")); panel = p2a.read_json(panel_path) if panel_path.is_file() else {}; closures = p2a.read_json(closure_path) if closure_path.is_file() else {}
    if panel.get("trigger_state") != "GREEN" or panel.get("source_panel_admitted") is not True or panel.get("assembled_task_count") != 62: faults.append("source_panel_invalid")
    if closures.get("trigger_state") != "GREEN" or closures.get("archive_artifacts") != 124: faults.append("repository_closures_invalid")
    slots = p2a.dicts(config.get("replacement_slots")); indices = [integer(row.get("index")) for row in slots]
    if indices != [13]: faults.append("replacement_denominator_invalid")
    panel_rows = {integer(row.get("index")):row for row in p2a.dicts(panel.get("assembled_rows"))}; closure_rows = {integer(row.get("campaign_index")):row for row in p2a.dicts(closures.get("tasks"))}; trigger_receipts=[]
    for slot in slots:
        index=integer(slot.get("index")); old=panel_rows.get(index,{}); closure=closure_rows.get(index,{}); target=next((row for row in p2a.dicts(closure.get("artifacts")) if row.get("label")=="target"),{})
        if slot.get("panel")!=old.get("panel") or slot.get("query_language")!=old.get("query_language") or slot.get("rejected_repository")!=old.get("repository") or slot.get("rejected_title_sha256")!=old.get("natural_language_request_sha256"): faults.append(f"replacement_slot_binding_invalid:{index}")
        receipt=scan_archive(p2a.resolve(str(target.get("normalized") or "")),str(target.get("source_archive_root") or ""),config); receipt.update({"index":index,"repository":old.get("repository")}); trigger_receipts.append(receipt)
        if receipt.get("explicit_unsupported_host") is not True: faults.append(f"host_invalidation_trigger_invalid:{index}")
    authority=p2a.mapping(config.get("authority")); allowed={"public_metadata_queries_authorized","public_source_file_retrieval_authorized","public_pr_title_metadata_retrieval_authorized","local_language_scope_classification_authorized","selected_content_static_language_scan_authorized","static_host_manifest_scan_authorized"}
    if any(authority.get(key) is not True for key in allowed) or any(value is not False for key,value in authority.items() if key not in allowed): faults.append("authority_boundary_invalid")
    counters=source.zero_counters(); counters.update({"public_metadata_selection_requests":0,"local_language_scope_classification_calls":0,"static_host_manifest_scan_calls":0})
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"HOST_ADEQUACY_REPLACEMENT_PREFLIGHT_GREEN" if not faults else "INVALID_PREFLIGHT","faults":sorted(set(faults)),"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"source_panel":{"path":p2a.rel(panel_path),"sha256":p2a.sha256_file(panel_path) if panel_path.is_file() else ""},"host_invalidation_receipts":trigger_receipts,"replacement_set_admitted":False,"selected_repository_count":0,"source_content_retrieval_opened":False,"candidate_packet_materialization_opened":False,"hidden_evaluation_opened":False,"qualified_rows_rerun":False,"frozen_qualified_indices":[16,25,56],"counters":counters,"maximum_inference":config.get("maximum_inference")}


def acquire(config_path: Path, ledger: source.SourceLedger, client: source.SourceClient, retry: dict[str, Any]) -> dict[str, Any]:
    before=preflight(config_path)
    if before["trigger_state"]!="GREEN": return before
    config=p2a.read_json(config_path); original_preflight=predecessor.preflight; original_materialize=source.materialize_row; receipts: dict[str,dict[str,Any]]={}
    def materialize_with_host_scan(selected: dict[str,Any], index: int, output: Path, active_client: source.SourceClient, maximum: int):
        row,faults,size=original_materialize(selected,index,output,active_client,maximum)
        if not faults:
            receipt=scan_candidate(selected,active_client,config); receipts[str(row.get("repository") or "")]=receipt
            if receipt.get("explicit_unsupported_host") is True: faults=[*faults,"explicit_unsupported_host_manifest_or_import"]
        return row,faults,size
    predecessor.preflight=lambda _path: {**before,"trigger_state":"GREEN"}; source.materialize_row=materialize_with_host_scan
    try: report=predecessor.acquire(config_path,ledger,client,retry)
    finally: predecessor.preflight=original_preflight; source.materialize_row=original_materialize
    if report.get("trigger_state")=="GREEN":
        selected=p2a.dicts(report.get("replacement_rows")); selected_receipts=[receipts.get(str(row.get("repository") or ""),{}) for row in selected]
        if len(selected_receipts)!=len(selected) or any(row.get("explicit_unsupported_host") is not False for row in selected_receipts): raise RuntimeError("selected_host_adequacy_receipt_invalid")
        report["host_compatibility_receipts"]=selected_receipts; report["state"]="HOST_ADEQUATE_ENGLISH_SOURCE_REPLACEMENTS_BOUND"
    elif report.get("trigger_state")=="RED": report["state"]="HOST_ADEQUACY_REPLACEMENT_POOL_EXHAUSTED_NO_PARTIAL_ADMISSION"
    counters=dict(p2a.mapping(report.get("counters"))); counters["static_host_manifest_scan_calls"]=len(receipts); report["counters"]=counters
    return {**report,"policy":POLICY,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"host_invalidation_receipts":before["host_invalidation_receipts"],"qualified_rows_rerun":False,"frozen_qualified_indices":[16,25,56],"maximum_inference":config.get("maximum_inference")}


def scan_candidate(selected: dict[str,Any], client: source.SourceClient, config: dict[str,Any]) -> dict[str,Any]:
    repository=str(selected.get("repository") or ""); revision=str(selected.get("head_revision") or ""); payloads={}
    for path in config.get("host_manifest_paths",[]):
        payload=client.file(repository,revision,str(path))
        if payload is not None: payloads[str(path)]=payload
    for path in [*p2a.strings(selected.get("source_paths")),*p2a.strings(selected.get("verifier_paths"))]:
        if path not in payloads:
            payload=client.file(repository,revision,path)
            if payload is not None: payloads[path]=payload
    return scan_payloads(repository,payloads,config)


def scan_archive(path: Path, root: str, config: dict[str,Any]) -> dict[str,Any]:
    payloads={}
    if path.is_file():
        with tarfile.open(path,"r:gz") as handle:
            names={member.name:member for member in handle.getmembers() if member.isfile()}
            wanted=set(config.get("host_manifest_paths",[]))
            for relative in wanted:
                member=names.get(f"{root}/{relative}")
                if member:
                    extracted=handle.extractfile(member); payloads[relative]=extracted.read() if extracted else b""
            for name,member in names.items():
                relative=name[len(root)+1:] if name.startswith(root+"/") else name
                if relative.endswith((".py",".toml")) and any(token in relative for token in ("src/","tests/","pyproject","requirements")):
                    extracted=handle.extractfile(member); payloads.setdefault(relative,extracted.read() if extracted else b"")
    return scan_payloads("",payloads,config)


def scan_payloads(repository: str, payloads: dict[str,bytes], config: dict[str,Any]) -> dict[str,Any]:
    patterns=[re.compile(str(value)) for value in config.get("unsupported_host_patterns",[])]; hits=[]; manifests=[]
    for path,payload in sorted(payloads.items()):
        text=payload.decode("utf-8","replace"); manifests.append({"path":path,"sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload)})
        for pattern in patterns:
            if pattern.search(text): hits.append({"path":path,"pattern":pattern.pattern})
    return {"repository":repository,"host_platform":sys.platform,"scanned_file_count":len(payloads),"scanned_files":manifests,"explicit_unsupported_host":bool(hits),"hits":hits}


def integer(value: Any) -> int:
    return int(value) if isinstance(value,(int,float)) and not isinstance(value,bool) else 0


if __name__=="__main__": raise SystemExit(main())
