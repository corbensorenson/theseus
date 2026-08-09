#!/usr/bin/env python3
"""Execute sealed parent/target common evaluators for the three VCM replacements."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,tempfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_six_row_matched_verifier as base  # noqa:E402
POLICY="project_theseus_vcm_replacement_matched_verifier_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_matched_verifier.json";EXPECTED=[12,13,35]

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");p.add_argument("--execute",action="store_true");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","execution_performed","task_count","qualified_task_count","inconclusive_task_count","parent_target_or_evaluator_executions","package_installations","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2

def preflight(path:Path=DEFAULT_CONFIG,*,verify_store=True)->tuple[dict[str,Any],dict[str,Any],list[str]]:
 cfg=p2a.read_json(path);faults=[];sources={}
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 for row in p2a.dicts(cfg.get("bindings")):
  q=p2a.resolve(str(row.get("path") or ""))
  if not q.is_file() or p2a.sha256_file(q)!=row.get("sha256"):faults.append(f"binding_invalid:{row.get('id')}")
  if row.get("kind")=="json" and q.is_file():sources[str(row.get("id"))]=p2a.read_json(q)
 environment=sources.get("environment",{});environment_audit=sources.get("environment_audit",{})
 if environment.get("trigger_state")!="GREEN" or environment.get("qualified_task_count")!=3 or environment_audit.get("trigger_state")!="GREEN" or environment_audit.get("qualified_task_count")!=3:faults.append("environment_predecessor_invalid")
 tools={}
 for name,row in p2a.mapping(cfg.get("tools")).items():
  q=p2a.resolve(str(p2a.mapping(row).get("path") or ""));tools[name]=q
  if not q.is_file() or p2a.sha256_file(q)!=p2a.mapping(row).get("sha256"):faults.append(f"tool_invalid:{name}")
 store=p2a.resolve(str(cfg.get("store") or ""));expected_store=p2a.mapping(environment.get("retained_shared_store"))
 if not store.is_dir() or p2a.rel(store)!=expected_store.get("path"):faults.append("store_path_invalid")
 elif verify_store and base.tree_identity(store)!=expected_store:faults.append("store_identity_invalid")
 rows=p2a.dicts(cfg.get("rows"));bound={}
 if [int(r.get("index") or 0) for r in rows]!=EXPECTED or int(cfg.get("expected_task_count") or 0)!=3:faults.append("row_denominator_invalid")
 for row in rows:
  index=int(row["index"]);manager=str(row.get("manager") or "");lock=p2a.resolve(str(row.get("lock") or ""));archives={}
  if manager not in {"uv","cargo"}:faults.append(f"task_{index}:manager_invalid")
  if manager=="uv" and str(row.get("python_tool") or "") not in tools:faults.append(f"task_{index}:python_tool_invalid")
  if not lock.is_file() or p2a.sha256_file(lock)!=row.get("lock_sha256"):faults.append(f"task_{index}:lock_invalid")
  for side in ("parent","target"):
   binding=p2a.mapping(row.get(f"{side}_archive"));archive=p2a.resolve(str(binding.get("path") or ""));archives[side]=archive
   if not archive.is_file() or p2a.sha256_file(archive)!=binding.get("sha256") or base.archive_root(archive)!=binding.get("root"):faults.append(f"task_{index}:{side}_archive_invalid")
  observed=base.archive_changes(archives["parent"],archives["target"]);declared=sorted(p2a.strings(row.get("target_changed_paths")));evaluators=sorted(p2a.strings(row.get("common_evaluator_paths")));forbidden=set(p2a.strings(row.get("forbidden_transplant_paths")))
  if observed!=declared:faults.append(f"task_{index}:change_partition_invalid")
  if not evaluators or not set(evaluators).issubset(declared) or set(evaluators)&forbidden:faults.append(f"task_{index}:evaluator_partition_invalid")
  for evaluator in evaluators:
   payload=base.archive_member(archives["target"],evaluator);expected=p2a.mapping(row.get("common_evaluator_sha256")).get(evaluator)
   if payload is None or hashlib.sha256(payload).hexdigest()!=expected:faults.append(f"task_{index}:evaluator_identity_invalid:{evaluator}")
  if manager=="uv" and (not p2a.strings(row.get("python_path_roots")) or any(Path(v).is_absolute() or ".." in Path(v).parts for v in p2a.strings(row.get("python_path_roots")))):faults.append(f"task_{index}:python_path_invalid")
  bound[index]={"config":row,"lock":lock,"archives":archives}
 authority=p2a.mapping(cfg.get("authority"));allowed={"offline_exact_lock_replay_authorized","hidden_target_evaluator_transplant_authorized","serial_parent_target_verifier_execution_authorized","disposable_copy_on_write_cache_clone_authorized","untrusted_rust_compile_authorized","complete_diagnostics_authorized"}
 if any(authority.get(k) is not True for k in allowed) or any(v is not False for k,v in authority.items() if k not in allowed):faults.append("authority_invalid")
 return cfg,{"sources":sources,"tools":tools,"store":store,"rows":bound},sorted(set(faults))

def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);store=base.tree_identity(bound["store"]) if not faults else {};return finish(cfg,path,faults,[],False,store,store,0,0,{})

def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path,verify_store=False);store_before=base.tree_identity(bound["store"]) if not faults else {};expected=p2a.mapping(p2a.mapping(bound.get("sources")).get("environment")).get("retained_shared_store")
 if not faults and store_before!=expected:faults.append("store_identity_invalid")
 reserve=int(p2a.mapping(cfg.get("limits"))["minimum_free_bytes_after_execution"]);peak=int(p2a.mapping(cfg.get("limits"))["required_incremental_peak_bytes"])
 if faults:return finish(cfg,path,faults,[],False,store_before,store_before,0,0,{})
 if shutil.disk_usage(ROOT).free-peak<reserve:return finish(cfg,path,["current_fit_boundary_closed"],[],False,store_before,store_before,0,0,{})
 rows=[];executions=installs=0;clones={}
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-replacement-verifier-",dir="/private/tmp") as raw:
  batch=Path(raw).resolve();caches={}
  for manager in ("cargo","uv"):
   destination=batch/"cache"/manager;destination.parent.mkdir(parents=True,exist_ok=True);clones[manager]=base.clone_tree(bound["tools"]["cp"],bound["store"]/manager,destination);caches[manager]=destination
   if clones[manager].get("returncode")!=0:faults.append(f"cache_clone_failed:{manager}")
  if not faults:
   for index in EXPECTED:
    if shutil.disk_usage(ROOT).free<reserve:rows.append(base.not_executed_row(bound["rows"][index]["config"],"INCONCLUSIVE_EXPERIMENT_HOST_RESOURCE_BOUNDARY"));continue
    row,count,installed=base.execute_row(cfg,bound,bound["rows"][index],batch,caches);rows.append(row);executions+=count;installs+=installed;shutil.rmtree(batch/f"task-{index:02d}")
    if shutil.disk_usage(ROOT).free<reserve:faults.append("free_space_reserve_postflight_boundary_hit");break
 store_after=base.tree_identity(bound["store"])
 if store_after!=store_before:faults.append("retained_store_mutated")
 return finish(cfg,path,sorted(set(faults)),rows,True,store_before,store_after,executions,installs,clones)

def finish(cfg,path,faults,rows,executed,before,after,executions,installs,clones):
 qualified=sum(r.get("disposition")=="QUALIFIED_COMMON_EVALUATOR_PARENT_FAIL_TARGET_PASS" for r in rows);valid=not faults and (not executed or len(rows)==3)
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if executed and valid else ("PAUSED" if not executed and valid else "RED"),"state":"THREE_REPLACEMENT_MATCHED_VERIFIERS_EXECUTED_WITH_SCOPED_DISPOSITIONS" if executed and valid else ("READY_FOR_THREE_REPLACEMENT_MATCHED_VERIFIERS" if valid else "REPLACEMENT_MATCHED_VERIFIER_INVALID"),"faults":sorted(set(faults)),"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":executed,"task_count":len(rows),"qualified_task_count":qualified,"inconclusive_task_count":len(rows)-qualified,"rows":rows,"retained_store_before":before,"retained_store_after":after,"retained_store_unchanged":before==after,"cache_clone_receipts":clones,"package_installations":installs,"source_build_executions":sum(1 for r in rows if r.get("manager")=="cargo" for side in ("parent","target") if p2a.mapping(r.get(side))),"project_installations":0,"repository_runner_executions":executions,"parent_target_or_evaluator_executions":executions,"network_enabled_calls":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"panel_admitted":False,"partial_panel_admission_forbidden":True,"target_production_transplant_count":0,"project_selected_output_cap":None,"maximum_inference":cfg.get("maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
