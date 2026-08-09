#!/usr/bin/env python3
"""Role-separately audit the three replacement immutable locks."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_immutable_resolution_segment as prior  # noqa:E402
POLICY="project_theseus_vcm_replacement_resolution_audit_v2";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_resolution.json"
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=audit(path);p2a.write_json(p2a.resolve(a.out or cfg["audit_report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","task_count","qualified_task_count","package_count","network_resolution_task_count","static_lock_task_count","parent_target_or_evaluator_executions","faults")},indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path=DEFAULT_CONFIG):
 cfg=p2a.read_json(path);faults=[]
 for b in p2a.dicts(cfg.get("bindings")):
  q=p2a.resolve(str(b.get("path") or ""))
  if not q.is_file() or p2a.sha256_file(q)!=b.get("sha256"):faults.append(f"binding_invalid:{b.get('id')}")
 report_path=p2a.resolve(cfg["report"]);producer=p2a.read_json(report_path) if report_path.is_file() else {}
 if producer.get("trigger_state")!="GREEN" or producer.get("qualified_task_count")!=3:faults.append("producer_not_qualified")
 rows=[];packages=network=static=0
 for row in p2a.dicts(producer.get("rows")):
  index=int(row.get("index") or 0);receipt=p2a.mapping(row.get("receipt"));lock=p2a.mapping(receipt.get("lock"));lock_path=p2a.resolve(str(lock.get("path") or ""));manager="cargo" if row.get("manager")=="cargo_static" else "uv";observed,validation=prior.validate_lock(manager,lock_path) if lock_path.is_file() else ({},["lock_missing"])
  if row.get("disposition")!="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK" or row.get("faults")!=[]:faults.append(f"task_{index}:disposition_invalid")
  if not lock_path.is_file() or p2a.sha256_file(lock_path)!=lock.get("sha256") or lock_path.stat().st_size!=lock.get("bytes"):faults.append(f"task_{index}:lock_identity_invalid")
  faults.extend(f"task_{index}:{f}" for f in validation);count=int(observed.get("package_count") or 0);packages+=count
  if receipt.get("source_before")!=receipt.get("source_after"):faults.append(f"task_{index}:source_mutation")
  if receipt.get("stdout_complete") is not True or receipt.get("stderr_complete") is not True or receipt.get("project_selected_output_cap") is not None:faults.append(f"task_{index}:diagnostic_policy_invalid")
  if row.get("manager")=="cargo_static":static+=1
  else:network+=1
  rows.append({"index":index,"repository":row.get("repository"),"manager":row.get("manager"),"package_count":count,"lock_sha256":lock.get("sha256"),"faults":[]})
 if [r["index"] for r in rows]!=[12,13,35] or network!=2 or static!=1:faults.append("denominator_invalid")
 authority=p2a.mapping(cfg.get("audit_authority"))
 if not authority or any(v is not False for v in authority.values()):faults.append("audit_authority_invalid")
 green=not faults
 return {"policy":POLICY,"audit_kind":"role-separated_rederivation","created_utc":p2a.now(),"trigger_state":"GREEN" if green else "RED","state":"THREE_REPLACEMENT_LOCKS_ROLE_SEPARATELY_REDERIVED" if green else "REPLACEMENT_RESOLUTION_AUDIT_FAILED","faults":sorted(set(faults)),"task_count":len(rows),"qualified_task_count":len(rows) if green else 0,"package_count":packages,"network_resolution_task_count":network,"static_lock_task_count":static,"rows":rows,"package_installations":0,"source_build_executions":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"local_model_calls":0,"external_reference_calls":0,"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"producer_report":{"path":p2a.rel(report_path),"sha256":p2a.sha256_file(report_path) if report_path.is_file() else ""},"maximum_inference":cfg.get("audit_maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
