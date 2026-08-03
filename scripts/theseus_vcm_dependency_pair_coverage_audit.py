#!/usr/bin/env python3
"""Audit parent/target dependency-lock coverage across all frozen VCM tasks."""
from __future__ import annotations
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_dependency_prefetch_canary as base  # noqa:E402
POLICY="project_theseus_vcm_dependency_pair_coverage_audit_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_dependency_pair_coverage_audit.json"
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));ap.add_argument("--out",default="");a=ap.parse_args();p=p2a.resolve(a.config);r=audit(p);p2a.write_json(p2a.resolve(a.out or p2a.read_json(p)["report"]),r);print(json.dumps(summary(r),indent=2,sort_keys=True));return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 owner=p2a.resolve(str(cfg.get("owner") or ""))
 if owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("owner_sha256"):faults.append("owner_binding_invalid")
 reports={};paths={}
 for name,raw in p2a.mapping(cfg.get("reports")).items():
  b=p2a.mapping(raw);p=p2a.resolve(str(b.get("path") or ""));paths[name]=p
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"):faults.append(f"report_binding_invalid:{name}");reports[name]={}
  else:reports[name]=p2a.read_json(p)
 plan=reports.get("prefetch_plan",{});inventory=reports.get("runner_inventory",{});closures=reports.get("closure_materialization",{})
 if plan.get("trigger_state")!="GREEN" or len(p2a.dicts(plan.get("schedule")))!=48:faults.append("prefetch_plan_invalid")
 if inventory.get("trigger_state")!="GREEN" or inventory.get("observations",{}).get("task_count")!=62:faults.append("inventory_invalid")
 if closures.get("trigger_state")!="GREEN" or closures.get("task_count")!=62:faults.append("closure_materialization_invalid")
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k=="static_parent_target_lock_audit_authorized"):faults.append(f"authority_invalid:{k}")
 inv={int(r["index"]):r for r in p2a.dicts(inventory.get("rows"))};closure={int(r["campaign_index"]):r for r in p2a.dicts(closures.get("tasks"))};rows=[]
 for planned in p2a.dicts(plan.get("schedule")):
  idx=int(planned["index"]);task=closure.get(idx,{});artifacts={str(a.get("label")):a for a in p2a.dicts(task.get("artifacts"))};target_inventory=p2a.mapping(p2a.mapping(inv.get(idx,{})).get("target_archive"));governing=p2a.mapping(planned.get("governing_lock"));manifest=p2a.mapping(governing.get("manifest"));pair={}
  for label in ("parent","target"):
   a=p2a.mapping(artifacts.get(label));archive=p2a.resolve(str(a.get("normalized") or ""))
   if not archive.is_file() or p2a.sha256_file(archive)!=a.get("normalized_sha256"):faults.append(f"task_{idx}:archive_binding_invalid:{label}");pair[label]={};continue
   pair[label]={"archive":base.identity(archive),"manifest":member_identity(archive,str(manifest.get("path") or "")),"lock":member_identity(archive,str(governing.get("path") or ""))}
  if pair.get("target",{}).get("archive",{}).get("path")!=target_inventory.get("path") or pair.get("target",{}).get("archive",{}).get("sha256")!=target_inventory.get("sha256"):faults.append(f"task_{idx}:target_inventory_mismatch")
  if pair.get("target",{}).get("manifest",{}).get("sha256")!=manifest.get("sha256") or pair.get("target",{}).get("lock",{}).get("sha256")!=governing.get("sha256"):faults.append(f"task_{idx}:target_plan_mismatch")
  missing=any(not pair.get(label,{}).get(kind,{}).get("sha256") for label in ("parent","target") for kind in ("manifest","lock"));same=not missing and pair["parent"]["manifest"]["sha256"]==pair["target"]["manifest"]["sha256"] and pair["parent"]["lock"]["sha256"]==pair["target"]["lock"]["sha256"]
  state="PAIR_MEMBER_MISSING" if missing else "IDENTICAL_PARENT_TARGET_DEPENDENCY_IDENTITY" if same else "DIVERGENT_PARENT_TARGET_DEPENDENCY_IDENTITY"
  rows.append({"index":idx,"schedule_ordinal":planned.get("schedule_ordinal"),"repository":planned.get("repository"),"manager":planned.get("manager"),"pair":pair,"state":state,"required_distinct_closure_count":1 if same else 2 if not missing else None,"network_or_dependency_execution_performed":False})
 obs={"locked_task_count":len(rows),"identical_pair_task_count":sum(r["state"]=="IDENTICAL_PARENT_TARGET_DEPENDENCY_IDENTITY" for r in rows),"divergent_pair_task_count":sum(r["state"]=="DIVERGENT_PARENT_TARGET_DEPENDENCY_IDENTITY" for r in rows),"missing_pair_member_task_count":sum(r["state"]=="PAIR_MEMBER_MISSING" for r in rows),"divergent_task_indices":sorted(r["index"] for r in rows if r["state"]=="DIVERGENT_PARENT_TARGET_DEPENDENCY_IDENTITY"),"required_distinct_dependency_closure_count":sum(int(r["required_distinct_closure_count"] or 0) for r in rows),"network_or_dependency_execution_performed":False,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0}
 if obs["missing_pair_member_task_count"]!=0 or obs["identical_pair_task_count"]+obs["divergent_pair_task_count"]!=48 or obs["required_distinct_dependency_closure_count"]!=48+obs["divergent_pair_task_count"]:faults.append("pair_denominator_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"FORTY_EIGHT_TASK_PARENT_TARGET_DEPENDENCY_PAIR_COVERAGE_CLASSIFIED" if not faults else "DEPENDENCY_PAIR_COVERAGE_AUDIT_FAILED","faults":sorted(set(faults)),"config":base.identity(path),"reports":{n:base.identity(p) for n,p in paths.items()},"observations":obs,"rows":rows,"static_audit_only":True,"network_or_dependency_execution_performed":False,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}
def member_identity(archive:Path,relative:str)->dict[str,Any]:
 with tarfile.open(archive,"r:gz") as h:
  members=[m for m in h.getmembers() if m.isfile() and "/" in m.name and m.name.split("/",1)[1]==relative]
  if len(members)!=1:return {"path":relative,"bytes":0,"sha256":"","match_count":len(members)}
  e=h.extractfile(members[0]);b=e.read() if e else b"";return {"path":relative,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),"match_count":1}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","observations","static_audit_only","network_or_dependency_execution_performed","repository_executions","candidate_or_control_calls","external_reference_calls","faults")}
if __name__=="__main__":raise SystemExit(main())
