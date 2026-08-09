#!/usr/bin/env python3
"""Generically assemble and role-audit a VCM source-panel replacement successor."""

from __future__ import annotations
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_content_language_replacements as content  # noqa:E402
import theseus_vcm_source_panel_audit as base  # noqa:E402
POLICY="project_theseus_vcm_source_panel_successor_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_source_panel_successor_audit.json";ROLES={"parent_source","target_source","parent_verifier","target_verifier"}

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2

def audit(config_path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(config_path);faults=[];loaded={}
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 for binding in p2a.dicts(cfg.get("owner_bindings")):
  path=p2a.resolve(str(binding.get("path") or ""))
  if not path.is_file() or p2a.sha256_file(path)!=binding.get("sha256"):faults.append(f"owner_binding_invalid:{binding.get('path')}")
 for name,b in p2a.mapping(cfg.get("reports")).items():
  rec=p2a.mapping(b);path=p2a.resolve(str(rec.get("path") or ""));loaded[name]=p2a.read_json(path) if path.is_file() and p2a.sha256_file(path)==rec.get("sha256") else {}
  if not loaded[name]:faults.append(f"report_binding_invalid:{name}")
 prior=loaded.get("prior_panel",{});replacement=loaded.get("replacement",{});replacement_audit=loaded.get("replacement_audit",{});indices=[int(v) for v in cfg.get("replacement_indices",[])]
 if prior.get("trigger_state")!="GREEN" or prior.get("source_panel_admitted") is not True or prior.get("assembled_task_count")!=62:faults.append("prior_panel_invalid")
 if replacement.get("trigger_state")!="GREEN" or replacement.get("replacement_set_admitted") is not True:faults.append("replacement_invalid")
 if replacement_audit.get("trigger_state")!="GREEN" or replacement_audit.get("replacement_set_admitted") is not True:faults.append("replacement_audit_invalid")
 old=p2a.dicts(prior.get("assembled_rows"));new={int(r.get("index") or 0):r for r in p2a.dicts(replacement.get("replacement_rows"))}
 if set(new)!=set(indices) or not indices or len(old)!=62:faults.append("replacement_denominator_invalid")
 rows=[]
 for index in range(1,63):
  before=old[index-1] if len(old)>=index else {};row=new.get(index,before)
  if index in new and (row.get("panel")!=before.get("panel") or row.get("query_language")!=before.get("query_language") or row.get("replacement_for_title_sha256")!=before.get("natural_language_request_sha256")):faults.append(f"task_{index}:replacement_binding_invalid")
  if index not in new and row!=before:faults.append(f"task_{index}:unapproved_substitution")
  title=str(row.get("natural_language_request") or "")
  if int(row.get("index") or 0)!=index:faults.append(f"task_{index}:index_invalid")
  if hashlib.sha256(title.encode()).hexdigest()!=row.get("natural_language_request_sha256"):faults.append(f"task_{index}:title_hash_invalid")
  rows.append(row)
 repositories=[str(r.get("repository") or "") for r in rows];old_repositories={str(r.get("repository") or "") for r in old}
 if len(set(repositories))!=62 or any(not r for r in repositories):faults.append("repository_uniqueness_invalid")
 if {str(new[i].get("repository") or "") for i in indices}&old_repositories:faults.append("replacement_repository_not_disjoint")
 title_receipts={int(r.get("index") or 0):r for r in p2a.dicts(replacement.get("language_classification_receipts"))};content_receipts={int(r.get("index") or 0):r for r in p2a.dicts(replacement.get("content_language_receipts"))};host_receipts={str(r.get("repository") or ""):r for r in p2a.dicts(replacement.get("host_compatibility_receipts"))}
 if set(title_receipts)!=set(indices) or set(content_receipts)!=set(indices):faults.append("replacement_language_receipts_invalid")
 for index in indices:
  row=rows[index-1];tr=title_receipts.get(index,{});cr=content_receipts.get(index,{});hr=host_receipts.get(str(row.get("repository") or ""),{})
  if tr.get("repository")!=row.get("repository") or tr.get("title_sha256")!=row.get("natural_language_request_sha256") or tr.get("dominant_language")!="en" or tr.get("accepted_english") is not True or tr.get("forbidden_unicode_scripts")!=[]:faults.append(f"task_{index}:title_receipt_invalid")
  if cr.get("repository")!=row.get("repository") or cr.get("selected_content_english_scope_passed") is not True or cr.get("violations")!=[]:faults.append(f"task_{index}:content_receipt_invalid")
  if hr.get("explicit_unsupported_host") is not False or hr.get("host_platform")!="darwin":faults.append(f"task_{index}:host_receipt_invalid")
 ranges=[(r["name"],int(r["start"],16),int(r["end"],16)) for r in cfg["forbidden_unicode_scripts"]];binary=set(cfg["binary_extensions"]);archive_paths=[];members=total=source_diff=verifier_diff=0;violations=[]
 for row in rows:
  index=int(row.get("index") or 0);archives=p2a.mapping(row.get("archives"));hashes={}
  if set(archives)!=ROLES or row.get("faults")!=[]:faults.append(f"task_{index}:archive_set_invalid");continue
  for role in sorted(ROLES):
   receipt=p2a.mapping(archives[role]);path=p2a.resolve(str(receipt.get("path") or ""));af,mh,mb=base.audit_archive(path,receipt);faults.extend(f"task_{index}:{role}:{f}" for f in af);hashes[role]=mh;archive_paths.append(p2a.rel(path));members+=len(mh);total+=mb
  sc=base.selected_paths_changed(row,hashes,"source");vc=base.selected_paths_changed(row,hashes,"verifier");source_diff+=int(sc);verifier_diff+=int(vc)
  if not sc:faults.append(f"task_{index}:source_unchanged")
  if not vc:faults.append(f"task_{index}:verifier_unchanged")
  found=content.selected_content_violations(row,ranges,binary)
  if found:violations.append({"index":index,"violations":found});faults.append(f"task_{index}:content_invalid")
 quotas=Counter((str(r.get("panel") or ""),str(r.get("query_language") or "")) for r in rows);expected={(r["panel"],r["query_language"]):int(r["count"]) for r in cfg["expected_quotas"]}
 if dict(quotas)!=expected:faults.append("quota_invalid")
 if len(archive_paths)!=248 or len(set(archive_paths))!=248:faults.append("archive_cardinality_invalid")
 if source_diff!=62 or verifier_diff!=62:faults.append("difference_count_invalid")
 authority=p2a.mapping(cfg.get("authority"))
 if not authority or any(v is not False for v in authority.values()):faults.append("authority_invalid")
 green=not faults
 return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"SIXTY_TWO_TASK_SOURCE_PANEL_SUCCESSOR_ADMITTED" if green else "SOURCE_PANEL_SUCCESSOR_AUDIT_FAILED","faults":sorted(set(faults)),"source_panel_admitted":green,"assembled_task_count":len(rows),"unique_repository_count":len(set(repositories)),"english_title_and_selected_content_task_count":62 if green else 0,"archive_receipt_count":len(archive_paths),"member_receipt_count":members,"total_member_bytes":total,"selected_source_difference_count":source_diff,"selected_verifier_difference_count":verifier_diff,"replacement_indices":indices,"preserved_row_count":62-len(indices),"content_violations":violations,"panel_language_quotas":[{"panel":p,"query_language":l,"count":c} for (p,l),c in sorted(quotas.items())],"assembled_rows":rows,"candidate_packet_materialization_opened":False,"parent_target_or_evaluator_executions":0,"local_model_calls":0,"external_reference_calls":0,"D1_cases_consumed":0,"D2_cases_consumed":0,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"reports":cfg.get("reports"),"maximum_inference":cfg.get("maximum_inference")}

def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","source_panel_admitted","assembled_task_count","unique_repository_count","archive_receipt_count","member_receipt_count","selected_source_difference_count","selected_verifier_difference_count","replacement_indices","preserved_row_count","content_violations","parent_target_or_evaluator_executions","local_model_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
