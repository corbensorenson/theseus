#!/usr/bin/env python3
"""Independently assemble and audit the VCM English-content source panel v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_source_panel_audit as v1audit  # noqa:E402
import theseus_vcm_content_language_replacements as replacement_owner  # noqa:E402

POLICY="project_theseus_vcm_source_panel_audit_v3"
DEFAULT_CONFIG=ROOT/"configs/theseus_vcm_source_panel_audit_v3.json"
ROLES={"parent_source","target_source","parent_verifier","target_verifier"}

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));parser.add_argument("--out",default="");args=parser.parse_args()
    path=p2a.resolve(args.config);report=audit(path);p2a.write_json(p2a.resolve(args.out or p2a.read_json(path)["report"]),report);print(json.dumps(summary(report),indent=2,sort_keys=True));return 0 if report["trigger_state"]=="GREEN" else 2

def audit(config_path:Path=DEFAULT_CONFIG)->dict[str,Any]:
    config=p2a.read_json(config_path);faults=[];loaded={}
    if config.get("policy")!=POLICY:faults.append("policy_invalid")
    for name,binding in p2a.mapping(config.get("reports")).items():
        path=p2a.resolve(str(p2a.mapping(binding).get("path") or ""))
        if not path.is_file() or p2a.sha256_file(path)!=p2a.mapping(binding).get("sha256"):faults.append(f"report_binding_invalid:{name}");loaded[name]={}
        else:loaded[name]=p2a.read_json(path)
    prior=loaded.get("prior_panel_v2",{});replacements=loaded.get("content_replacements",{});content_failure=loaded.get("content_language_failure",{})
    indices=[int(v) for v in config.get("replacement_indices",[])]
    if prior.get("trigger_state")!="GREEN" or prior.get("source_panel_admitted") is not True:faults.append("prior_panel_invalid")
    if replacements.get("trigger_state")!="GREEN" or replacements.get("replacement_set_admitted") is not True:faults.append("replacement_report_invalid")
    if content_failure.get("violation_indices")!=indices or content_failure.get("source_panel_admission_remains_valid") is not False:faults.append("content_failure_trigger_invalid")
    old_rows=p2a.dicts(prior.get("assembled_rows"));replacement_rows={int(r.get("index") or 0):r for r in p2a.dicts(replacements.get("replacement_rows"))}
    if len(old_rows)!=62 or set(replacement_rows)!=set(indices):faults.append("row_cardinality_invalid")
    rows=[]
    for index in range(1,63):
        old=old_rows[index-1] if len(old_rows)>=index else {};row=replacement_rows.get(index,old)
        if index in replacement_rows:
            if row.get("panel")!=old.get("panel") or row.get("query_language")!=old.get("query_language") or row.get("replacement_for_title_sha256")!=old.get("natural_language_request_sha256"):faults.append(f"task_{index:02d}:replacement_slot_binding_invalid")
        elif row!=old:faults.append(f"task_{index:02d}:unapproved_substitution")
        if int(row.get("index") or 0)!=index:faults.append(f"task_{index:02d}:index_invalid")
        title=str(row.get("natural_language_request") or "")
        if hashlib.sha256(title.encode()).hexdigest()!=row.get("natural_language_request_sha256"):faults.append(f"task_{index:02d}:title_hash_invalid")
        rows.append(row)
    repositories=[str(r.get("repository") or "") for r in rows]
    if len(set(repositories))!=62 or any(not r for r in repositories):faults.append("repository_uniqueness_invalid")
    old_repositories={str(r.get("repository") or "") for r in old_rows};new_repositories={str(replacement_rows[i].get("repository") or "") for i in indices}
    if new_repositories & old_repositories:faults.append("replacement_repository_not_disjoint")
    receipts={int(r.get("index") or 0):r for r in p2a.dicts(replacements.get("language_classification_receipts"))}
    content_receipts={int(r.get("index") or 0):r for r in p2a.dicts(replacements.get("content_language_receipts"))}
    if set(receipts)!=set(indices) or set(content_receipts)!=set(indices):faults.append("replacement_language_receipt_set_invalid")
    for index in indices:
        row=rows[index-1];title_receipt=receipts.get(index,{});content_receipt=content_receipts.get(index,{})
        if title_receipt.get("repository")!=row.get("repository") or title_receipt.get("title_sha256")!=row.get("natural_language_request_sha256") or title_receipt.get("dominant_language")!="en" or title_receipt.get("accepted_english") is not True or title_receipt.get("forbidden_unicode_scripts")!=[]:faults.append(f"task_{index:02d}:title_language_receipt_invalid")
        if content_receipt.get("repository")!=row.get("repository") or content_receipt.get("selected_content_english_scope_passed") is not True or content_receipt.get("violations")!=[]:faults.append(f"task_{index:02d}:content_language_receipt_invalid")
    ranges=[(r["name"],int(r["start"],16),int(r["end"],16)) for r in config["forbidden_unicode_scripts"]];binary=set(config["binary_extensions"])
    archive_paths=[];members=bytes_total=source_diff=verifier_diff=0;content_violations=[]
    for row in rows:
        index=int(row.get("index") or 0);archives=p2a.mapping(row.get("archives"));hashes={}
        if set(archives)!=ROLES or row.get("faults")!=[]:faults.append(f"task_{index:02d}:archive_set_invalid");continue
        for role in sorted(ROLES):
            receipt=p2a.mapping(archives[role]);path=p2a.resolve(str(receipt.get("path") or ""));archive_faults,member_hashes,member_bytes=v1audit.audit_archive(path,receipt)
            faults.extend(f"task_{index:02d}:{role}:{f}" for f in archive_faults);hashes[role]=member_hashes;archive_paths.append(p2a.rel(path));members+=len(member_hashes);bytes_total+=member_bytes
            license_path=str(row.get("parent_license_path") if role.startswith("parent_") else row.get("target_license_path") or "")
            if license_path not in member_hashes:faults.append(f"task_{index:02d}:{role}:license_missing")
        if v1audit.selected_paths_changed(row,hashes,"source"):source_diff+=1
        else:faults.append(f"task_{index:02d}:source_unchanged")
        if v1audit.selected_paths_changed(row,hashes,"verifier"):verifier_diff+=1
        else:faults.append(f"task_{index:02d}:verifier_unchanged")
        violations=replacement_owner.selected_content_violations(row,ranges,binary)
        if violations:content_violations.append({"index":index,"repository":row.get("repository"),"violations":violations});faults.append(f"task_{index:02d}:selected_content_language_invalid")
    quotas=Counter((str(r.get("panel") or ""),str(r.get("query_language") or "")) for r in rows);expected_quotas={(r["panel"],r["query_language"]):int(r["count"]) for r in config["expected_quotas"]}
    if dict(quotas)!=expected_quotas:faults.append("panel_language_quotas_invalid")
    if len(archive_paths)!=248 or len(set(archive_paths))!=248:faults.append("archive_cardinality_invalid")
    if source_diff!=62 or verifier_diff!=62:faults.append("difference_count_invalid")
    authority=p2a.mapping(config.get("authority"))
    if not authority or any(value is not False for value in authority.values()):faults.append("authority_boundary_invalid")
    admitted=not faults
    return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if admitted else "RED","state":"SIXTY_TWO_ENGLISH_TITLE_AND_SELECTED_CONTENT_TASKS_ADMITTED" if admitted else "SOURCE_PANEL_V3_AUDIT_FAILED","faults":sorted(set(faults)),"source_panel_admitted":admitted,"assembled_task_count":len(rows),"unique_repository_count":len(set(repositories)),"english_title_and_selected_content_task_count":62 if admitted else 0,"archive_receipt_count":len(archive_paths),"member_receipt_count":members,"total_member_bytes":bytes_total,"selected_source_difference_count":source_diff,"selected_verifier_difference_count":verifier_diff,"replacement_indices":indices,"superseded_archive_count_preserved_but_ignored":len(indices)*4,"content_violations":content_violations,"panel_language_quotas":[{"panel":p,"query_language":l,"count":c} for (p,l),c in sorted(quotas.items())],"assembled_rows":rows,"candidate_packet_materialization_opened":False,"parent_target_or_evaluator_executions":0,"local_model_calls":0,"external_reference_calls":0,"D1_cases_consumed":0,"D2_cases_consumed":0,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"reports":{name:{"path":p2a.mapping(binding)["path"],"sha256":p2a.mapping(binding)["sha256"]} for name,binding in p2a.mapping(config.get("reports")).items()},"maximum_inference":config.get("maximum_inference")}

def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","source_panel_admitted","assembled_task_count","unique_repository_count","english_title_and_selected_content_task_count","archive_receipt_count","member_receipt_count","selected_source_difference_count","selected_verifier_difference_count","replacement_indices","content_violations","parent_target_or_evaluator_executions","local_model_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
