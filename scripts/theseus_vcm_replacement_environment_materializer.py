#!/usr/bin/env python3
"""Materialize and offline-replay exact environments for VCM replacement rows."""
from __future__ import annotations
import argparse,json,shutil,sys,tempfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import theseus_assistant_p2a as p2a  # noqa:E402
import theseus_vcm_six_row_environment_materializer as base  # noqa:E402
POLICY="project_theseus_vcm_replacement_environment_materializer_v1";DEFAULT_CONFIG=ROOT/"configs"/"theseus_vcm_replacement_environment_materializer.json";EXPECTED=[12,13,35]

def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--config",default=p2a.rel(DEFAULT_CONFIG));p.add_argument("--out",default="");p.add_argument("--execute",action="store_true");a=p.parse_args();path=p2a.resolve(a.config);cfg=p2a.read_json(path);r=execute(path) if a.execute else preflight_report(path);p2a.write_json(p2a.resolve(a.out or cfg["report"]),r);print(json.dumps({k:r.get(k) for k in ("trigger_state","state","faults","execution_performed","task_count","qualified_task_count","inconclusive_task_count","network_enabled_materializations","network_denied_replays","package_installations","repository_runner_executions","candidate_or_control_calls","external_reference_calls")},indent=2,sort_keys=True));return 0 if r["trigger_state"] in {"GREEN","PAUSED"} else 2

def preflight(path:Path=DEFAULT_CONFIG)->tuple[dict[str,Any],dict[str,Any],list[str]]:
 cfg=p2a.read_json(path);faults=[]
 if cfg.get("policy")!=POLICY:faults.append("policy_invalid")
 sources={}
 for row in p2a.dicts(cfg.get("bindings")):
  q=p2a.resolve(str(row.get("path") or ""))
  if not q.is_file() or p2a.sha256_file(q)!=row.get("sha256"):faults.append(f"binding_invalid:{row.get('id')}")
  if row.get("kind")=="json" and q.is_file():sources[str(row.get("id"))]=p2a.read_json(q)
 resolution=sources.get("resolution",{});audit=sources.get("resolution_audit",{});closures=sources.get("closures",{})
 if resolution.get("trigger_state")!="GREEN" or resolution.get("qualified_task_count")!=3 or audit.get("trigger_state")!="GREEN" or audit.get("qualified_task_count")!=3:faults.append("resolution_predecessor_invalid")
 tools={}
 for name,row in p2a.mapping(cfg.get("tools")).items():
  q=p2a.resolve(str(p2a.mapping(row).get("path") or ""));tools[name]=str(q)
  if not q.is_file() or p2a.sha256_file(q)!=p2a.mapping(row).get("sha256"):faults.append(f"tool_invalid:{name}")
 resolution_rows={int(r.get("index") or 0):r for r in p2a.dicts(resolution.get("rows"))};closure_rows={int(r.get("campaign_index") or 0):r for r in p2a.dicts(closures.get("tasks"))};bound={};rows=p2a.dicts(cfg.get("rows"))
 if [int(r.get("index") or 0) for r in rows]!=EXPECTED:faults.append("row_denominator_invalid")
 for row in rows:
  index=int(row["index"]);producer=resolution_rows.get(index,{});receipt=p2a.mapping(p2a.mapping(producer.get("receipt")).get("lock"));lock=p2a.resolve(str(row.get("lock") or ""));closure=closure_rows.get(index,{});target=next((a for a in p2a.dicts(closure.get("artifacts")) if a.get("label")=="target"),{});archive=p2a.resolve(str(target.get("normalized") or ""))
  if producer.get("repository")!=row.get("repository") or producer.get("disposition")!="RESOLUTION_QUALIFIED_IMMUTABLE_LOCK":faults.append(f"task_{index}:resolution_alignment_invalid")
  if not lock.is_file() or p2a.sha256_file(lock)!=row.get("sha256") or receipt.get("sha256")!=row.get("sha256") or receipt.get("package_count")!=row.get("package_count"):faults.append(f"task_{index}:lock_invalid")
  if not archive.is_file() or p2a.sha256_file(archive)!=target.get("normalized_sha256"):faults.append(f"task_{index}:archive_invalid")
  if row.get("manager")=="uv" and row.get("python_tool") not in tools:faults.append(f"task_{index}:python_tool_invalid")
  if row.get("manager") not in {"uv","cargo"}:faults.append(f"task_{index}:manager_invalid")
  bound[index]={"config":row,"lock":lock,"resolver":{"archive":archive},"target":target}
 authority=p2a.mapping(cfg.get("authority"));allowed={"serialized_network_dependency_materialization_authorized","network_denied_offline_replay_authorized","wheel_only_python_installation_authorized","cargo_fetch_without_build_authorized","shared_store_retention_authorized","disposable_environment_authorized"}
 if any(authority.get(k) is not True for k in allowed) or any(v is not False for k,v in authority.items() if k not in allowed):faults.append("authority_invalid")
 store=p2a.resolve(str(cfg.get("store") or ""))
 if store!=(ROOT/"runtime/vcm_evaluator/dependency_store/replacement-environments-v1").resolve():faults.append("store_invalid")
 return cfg,{"sources":sources,"tools":tools,"rows":bound,"store":store},sorted(set(faults))

def preflight_report(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);return finish(cfg,path,faults,[],False,{})

def execute(path:Path=DEFAULT_CONFIG)->dict[str,Any]:
 cfg,bound,faults=preflight(path);store=bound.get("store",Path("/invalid"));limits=p2a.mapping(cfg.get("limits"));free_before=shutil.disk_usage(ROOT).free;required=int(limits["maximum_shared_store_bytes"])+int(limits["maximum_disposable_and_temporary_bytes"]);reserve=int(limits["minimum_free_bytes_after_execution"])
 if faults:return finish(cfg,path,faults,[],False,{})
 if store.exists():return finish(cfg,path,["immutable_shared_store_already_exists"],[],False,{})
 if free_before-required<reserve:return finish(cfg,path,["free_space_reserve_preflight_boundary_hit"],[],False,{})
 store.mkdir(parents=True);rows=[]
 with tempfile.TemporaryDirectory(prefix="theseus-vcm-replacement-env-",dir="/private/tmp") as raw:
  batch=Path(raw).resolve()
  for index in EXPECTED:
   item=bound["rows"][index];rows.append(base.materialize_row(cfg,bound,p2a.mapping(item["config"]),item,batch,store))
   if base.tree_identity(store)["bytes"]>int(limits["maximum_shared_store_bytes"]):faults.append("shared_store_size_boundary_hit");break
   if shutil.disk_usage(ROOT).free<reserve:faults.append("free_space_reserve_postflight_boundary_hit");break
 if len(rows)!=3:faults.append("row_denominator_not_completed")
 return finish(cfg,path,sorted(set(faults)),rows,True,base.tree_identity(store),free_before)

def finish(cfg,path,faults,rows,executed,store,free_before=None):
 qualified=sum(r.get("disposition")=="ENVIRONMENT_MATERIALIZATION_QUALIFIED" for r in rows);online=sum("online_sync" in p2a.mapping(r.get("receipts")) or "online_fetch" in p2a.mapping(r.get("receipts")) for r in rows);offline=sum("offline_sync" in p2a.mapping(r.get("receipts")) or "offline_fetch" in p2a.mapping(r.get("receipts")) for r in rows);green=executed and not faults and qualified==3
 return {"policy":POLICY,"created_utc":p2a.now(),"trigger_state":"GREEN" if green else ("PAUSED" if not faults else "RED"),"state":"THREE_REPLACEMENT_ENVIRONMENTS_QUALIFIED" if green else ("READY_FOR_THREE_REPLACEMENT_ENVIRONMENT_MATERIALIZATIONS" if not executed and not faults else "REPLACEMENT_ENVIRONMENT_SCOPED_DISPOSITIONS"),"faults":faults,"config":{"path":p2a.rel(path),"sha256":p2a.sha256_file(path)},"execution_performed":executed,"task_count":len(rows),"qualified_task_count":qualified,"inconclusive_task_count":len(rows)-qualified,"rows":rows,"retained_shared_store":store,"free_bytes_before":free_before if free_before is not None else shutil.disk_usage(ROOT).free,"free_bytes_after":shutil.disk_usage(ROOT).free,"network_enabled_materializations":online,"network_denied_replays":offline,"package_installations":sum(2 for r in rows if r.get("manager")=="uv" and r.get("disposition")=="ENVIRONMENT_MATERIALIZATION_QUALIFIED"),"cargo_dependency_fetches":sum(2 for r in rows if r.get("manager")=="cargo" and r.get("disposition")=="ENVIRONMENT_MATERIALIZATION_QUALIFIED"),"source_build_executions":0,"project_installations":0,"repository_runner_executions":0,"parent_target_or_evaluator_executions":0,"candidate_or_control_calls":0,"external_reference_calls":0,"teacher_calls":0,"panel_admitted":False,"partial_panel_admission_forbidden":True,"project_selected_output_cap":None,"maximum_inference":cfg.get("maximum_inference")}
if __name__=="__main__":raise SystemExit(main())
