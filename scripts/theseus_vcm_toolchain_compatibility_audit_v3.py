#!/usr/bin/env python3
"""Reclassify frozen VCM tools with exact Rust/Cargo 1.97.1."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa: E402
import theseus_vcm_toolchain_compatibility_audit as base  # noqa: E402
POLICY="project_theseus_vcm_toolchain_compatibility_audit_v3"; DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_toolchain_compatibility_audit_v3.json"
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG)); ap.add_argument("--out",default=""); a=ap.parse_args(); p=p2a.resolve(a.config); r=audit(p); p2a.write_json(p2a.resolve(a.out or p2a.read_json(p)["report"]),r); print(json.dumps(summary(r),indent=2,sort_keys=True)); return 0 if r["trigger_state"]=="GREEN" else 2
def audit(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg=p2a.read_json(path); faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 owner=p2a.resolve(str(cfg.get("owner") or ""))
 if owner!=Path(__file__).resolve() or not owner.is_file() or p2a.sha256_file(owner)!=cfg.get("owner_sha256"):faults.append("owner_binding_invalid")
 artifacts={}; paths={}
 for name,raw in p2a.mapping(cfg.get("reports")).items():
  b=p2a.mapping(raw); p=p2a.resolve(str(b.get("path") or "")); paths[name]=p
  if not p.is_file() or p2a.sha256_file(p)!=b.get("sha256"): faults.append(f"report_binding_invalid:{name}"); artifacts[name]={}
  else: artifacts[name]=p2a.read_json(p)
 pred=artifacts.get("predecessor_audit",{}); rust=artifacts.get("rust_toolchain",{})
 if pred.get("trigger_state")!="GREEN" or pred.get("observations",{}).get("locked_task_count")!=48:faults.append("predecessor_invalid")
 if rust.get("trigger_state")!="GREEN" or rust.get("state")!="EXACT_RUST_1_97_1_MINIMAL_TOOLCHAIN_MATERIALIZED_AND_VERSION_QUALIFIED":faults.append("rust_toolchain_invalid")
 profile=p2a.mapping(cfg.get("added_profile")); probes=p2a.mapping(p2a.mapping(rust.get("receipt")).get("toolchain")).get("probes",{})
 if profile.get("id")!="cargo1_97_1_rustc1_97_1" or profile.get("manager")!="cargo" or profile.get("manager_version")!="1.97.1" or p2a.mapping(profile.get("runtime_versions")).get("rustc")!="1.97.1" or p2a.mapping(p2a.mapping(probes).get("rustc")).get("binary_sha256")!=profile.get("rustc_sha256") or p2a.mapping(p2a.mapping(probes).get("cargo")).get("binary_sha256")!=profile.get("cargo_sha256"):faults.append("profile_identity_invalid")
 for k,v in p2a.mapping(cfg.get("authority")).items():
  if v is not (k=="source_bound_static_reclassification_authorized"):faults.append(f"authority_invalid:{k}")
 rows=copy.deepcopy(p2a.dicts(pred.get("rows"))); before={int(r["index"]):r.get("state") for r in rows}
 for row in rows:
  if row.get("manager")!="cargo":continue
  evaluations=p2a.dicts(row.get("profile_evaluations"))+[base.evaluate_profile(profile,p2a.dicts(row.get("requirements")))]; compatible=[e for e in evaluations if e.get("compatible") is True]; unknown=[e for e in evaluations if e.get("compatible") is None]; declared=len(p2a.dicts(row.get("requirements")))
  row["profile_evaluations"]=evaluations; row["compatible_profile_ids"]=[e["profile_id"] for e in compatible]
  row["state"]="COMPATIBLE_DECLARED_REQUIREMENTS" if compatible and declared else "NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE" if compatible else "COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX" if unknown or row.get("requirement_parse_faults") else "INCOMPATIBLE_DECLARED_REQUIREMENTS"
 changed=sorted(int(r["index"]) for r in rows if before[int(r["index"])]!=r.get("state"))
 obs={"locked_task_count":len(rows),"compatible_declared_requirement_task_count":sum(r["state"]=="COMPATIBLE_DECLARED_REQUIREMENTS" for r in rows),"no_declared_version_requirement_task_count":sum(r["state"]=="NO_DECLARED_VERSION_REQUIREMENTS_TOOL_AVAILABLE" for r in rows),"incompatible_declared_requirement_task_count":sum(r["state"]=="INCOMPATIBLE_DECLARED_REQUIREMENTS" for r in rows),"unresolved_requirement_syntax_task_count":sum(r["state"]=="COMPATIBILITY_UNRESOLVED_REQUIREMENT_SYNTAX" for r in rows),"changed_task_indices":changed,"dependency_prefetch_executions":0,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0}
 if changed!=[29,36]:faults.append("change_scope_invalid")
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if not faults else "RED","state":"FORTY_EIGHT_LOCKED_TASK_TOOLCHAIN_COMPATIBILITY_RECLASSIFIED_WITH_RUST_1_97_1" if not faults else "TOOLCHAIN_COMPATIBILITY_RECLASSIFICATION_FAILED","faults":sorted(set(faults)),"config":artifact(path),"reports":{n:artifact(p) for n,p in paths.items()},"added_profile":profile,"observations":obs,"rows":rows,"dependency_prefetch_executions":0,"repository_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"maximum_inference":cfg.get("maximum_inference")}
def artifact(p:Path)->dict[str,str]:return {"path":p2a.rel(p),"sha256":p2a.sha256_file(p) if p.is_file() else ""}
def summary(r:dict[str,Any])->dict[str,Any]:return {k:r.get(k) for k in ("trigger_state","state","faults","observations","dependency_prefetch_executions","repository_executions","candidate_or_control_calls","external_reference_calls")}
if __name__=="__main__":raise SystemExit(main())
