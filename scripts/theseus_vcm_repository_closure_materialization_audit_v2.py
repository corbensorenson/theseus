#!/usr/bin/env python3
"""Audit one replacement closure pair while reusing 61 audited closure pairs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_repository_closure_materialization as closure  # noqa:E402
POLICY="project_theseus_vcm_repository_closure_materialization_audit_v2";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_repository_closure_materialization_audit_v2.json"
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(config_path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(config_path);faults=[];loaded={}
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 for name,b in p2a.mapping(cfg.get("bindings")).items():
  rec=p2a.mapping(b);path=p2a.resolve(str(rec.get("path") or ""));valid=path.is_file() and p2a.sha256_file(path)==rec.get("sha256")
  if not valid:faults.append(f"binding_invalid:{name}")
  if name in {"panel","producer","predecessor","predecessor_audit"}:loaded[name]=p2a.read_json(path) if valid else {}
 panel=loaded.get("panel",{});producer=loaded.get("producer",{});predecessor=loaded.get("predecessor",{});predecessor_audit=loaded.get("predecessor_audit",{})
 if panel.get("trigger_state")!="GREEN" or panel.get("source_panel_admitted") is not True:faults.append("panel_invalid")
 if producer.get("trigger_state")!="GREEN" or producer.get("archive_artifacts")!=124 or producer.get("network_fetches")!=2:faults.append("producer_invalid")
 if predecessor.get("trigger_state")!="GREEN" or predecessor_audit.get("trigger_state")!="GREEN":faults.append("predecessor_evidence_invalid")
 current={int(r.get("campaign_index") or 0):r for r in p2a.dicts(producer.get("tasks"))};prior={int(r.get("campaign_index") or 0):r for r in p2a.dicts(predecessor.get("tasks"))};registry=closure.transform_panel(panel);tasks={int(r.get("campaign_index") or 0):r for r in registry.get("tasks",[])}
 if set(current)!=set(range(1,63)) or set(tasks)!=set(current):faults.append("denominator_invalid")
 replayed=0;replacement_artifacts=0;rows=[]
 for index in range(1,63):
  row=current.get(index,{});before=prior.get(index,{});task=tasks.get(index,{});row_faults=[];artifacts=p2a.dicts(row.get("artifacts"))
  if index!=13:
   old={a.get("label"):a.get("normalized_sha256") for a in p2a.dicts(before.get("artifacts"))};new={a.get("label"):a.get("normalized_sha256") for a in artifacts}
   if row.get("repository")!=before.get("repository") or new!=old:row_faults.append("unchanged_closure_replay_drift")
   else:replayed+=1
  else:
   if row.get("repository")!="paulomtts/pyjinhx" or row.get("repository")==before.get("repository"):row_faults.append("replacement_identity_invalid")
   if {a.get("label") for a in artifacts}!={"parent","target"}:row_faults.append("replacement_artifact_labels_invalid")
   for receipt in artifacts:
    path=p2a.resolve(str(receipt.get("normalized") or ""));san_path=p2a.resolve(str(receipt.get("sanitization_report") or ""));san=p2a.read_json(san_path) if san_path.is_file() else {}
    if not path.is_file() or p2a.sha256_file(path)!=receipt.get("normalized_sha256"):row_faults.append(f"{receipt.get('label')}:archive_identity_invalid")
    if not san_path.is_file() or p2a.sha256_file(san_path)!=receipt.get("sanitization_report_sha256"):row_faults.append(f"{receipt.get('label')}:sanitization_identity_invalid")
    row_faults.extend(f"{receipt.get('label')}:{f}" for f in closure.audit_closure_artifact(task,receipt,path,san));replacement_artifacts+=1
  faults.extend(f"task_{index}:{f}" for f in row_faults);rows.append({"index":index,"repository":row.get("repository"),"artifact_count":len(artifacts),"faults":row_faults})
 if replayed!=61 or replacement_artifacts!=2:faults.append("replay_or_replacement_count_invalid")
 authority=p2a.mapping(cfg.get("authority"))
 if not authority or any(v is not False for v in authority.values()):faults.append("authority_invalid")
 green=not faults
 return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"HOST_ADEQUATE_62_ROW_CLOSURES_ROLE_SEPARATELY_REDERIVED" if green else "CLOSURE_AUDIT_V2_FAILED","faults":sorted(set(faults)),"task_count":len(current),"archive_artifact_count":124,"replayed_unchanged_task_count":replayed,"replacement_task_count":1,"replacement_index":13,"replacement_archive_count":replacement_artifacts,"audited_rows":rows,"network_calls":0,"parent_target_or_evaluator_executions":0,"local_model_calls":0,"external_reference_calls":0,"config":{"path":p2a.rel(config_path),"sha256":p2a.sha256_file(config_path)},"maximum_inference":cfg.get("maximum_inference")}
def summary(r):return {k:r.get(k) for k in ("trigger_state","state","task_count","archive_artifact_count","replayed_unchanged_task_count","replacement_task_count","replacement_index","replacement_archive_count","network_calls","parent_target_or_evaluator_executions","local_model_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
